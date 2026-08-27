"""MLB 부상자명단(IL) 데이터 수집 파이프라인 — 설계 초안.

목적: Lahman에는 없는 부상 데이터를, 이미 게임 데이터용으로 쓰기로 한 MLB Stats API의
transactions 엔드포인트에서 가져와 player_season 에 붙일 수 있는 형태로 만든다.
(참고: 이 API는 개인·비상업·비대량 사용 시 무료, API 키 불필요 — 이미 검증함)

전체 흐름
  1) Chadwick Bureau Register 로 player_id(Lahman) <-> MLB Stats API person.id 크로스워크 확보
  2) MLB Stats API /api/v1/transactions 를 연도별로 순회 호출
  3) description 텍스트에서 "injured list"/"disabled list" 관련 거래만 필터링
  4) player_id + season 단위로 집계 (해당 시즌에 IL 등재가 있었는지 0/1, 등재 횟수, 사유 텍스트)
  5) player_injury_stints.csv 로 저장 → A의 load.py 가 Supabase 에 적재

실행 전 준비물
  - `uv add requests` (아직 없다면)
  - 인터넷 연결. API 키는 필요 없음.

이 파일은 설계 초안입니다. 실제 프로젝트에 편입할 때는 A가 src/adapters/mlb_api.py 의
스타일(로깅, 재시도, config 분리 등)에 맞춰 다듬어야 합니다.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

# ── 설정 ──────────────────────────────────────────────────────────
START_YEAR = 2000
END_YEAR = 2025  # 학습 구간과 동일하게. 당해 연도(2026)는 별도로 필요시 추가 호출
SEASON_MONTHS = (3, 11)  # 시즌 관련 거래는 대략 3월(스프링캠프)~11월(월드시리즈)에 몰림
REQUEST_DELAY_SEC = 0.3  # MLB Stats API 에 과도한 부하를 주지 않기 위한 최소 대기

TRANSACTIONS_URL = "https://statsapi.mlb.com/api/v1/transactions"
# 실제로 확인함: 파일 하나가 아니라 16개로 샤딩되어 있음 (people-0.csv ~ people-f.csv)
CHADWICK_REGISTER_BASE = "https://raw.githubusercontent.com/chadwickbureau/register/master/data"
CHADWICK_SHARDS = "0123456789abcdef"

# 부상자명단 관련 거래인지 판별하는 키워드 (MLB 거래 설명문 기준)
INJURY_PATTERNS = re.compile(
    r"injured list|disabled list|\bIL\b|\bDL\b", re.IGNORECASE
)
# "activated from the injured list" 처럼 복귀 거래는 별도로 구분해둔다 (복귀일 = IL 종료일 추정)
ACTIVATED_PATTERN = re.compile(r"activated", re.IGNORECASE)


# ── 1) ID 크로스워크 ─────────────────────────────────────────────
def load_id_crosswalk(cache_dir: Path) -> pd.DataFrame:
    """Chadwick Bureau Register 에서 Lahman playerID <-> MLB person.id 매핑을 가져온다.

    Register 는 key_bbref(Baseball-Reference ID) 컬럼을 갖고 있고, Lahman People.csv 의
    bbrefID 와 값이 동일하다 — 이 컬럼으로 조인하면 이름/생년월일 매칭 없이 정확히 연결된다.

    실제 파일은 16개로 샤딩되어 있다 (data/people-0.csv ~ people-f.csv, key_person의
    첫 hex 자리 기준). 직접 확인함 — 컬럼명은 key_mlbam / key_bbref.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for shard in CHADWICK_SHARDS:
        cache_path = cache_dir / f"chadwick_people-{shard}.csv"
        if not cache_path.exists():
            resp = requests.get(f"{CHADWICK_REGISTER_BASE}/people-{shard}.csv", timeout=60)
            resp.raise_for_status()
            cache_path.write_bytes(resp.content)
        frames.append(pd.read_csv(cache_path, usecols=["key_mlbam", "key_bbref"], low_memory=False))

    reg = pd.concat(frames, ignore_index=True).dropna()
    reg["key_mlbam"] = reg["key_mlbam"].astype(int)
    return reg.rename(columns={"key_bbref": "bbrefID", "key_mlbam": "mlbam_id"})


def build_player_id_map(people_csv: Path, cache_dir: Path) -> pd.DataFrame:
    """player_id(Lahman) -> mlbam_id(MLB Stats API) 매핑 테이블."""
    people = pd.read_csv(people_csv, usecols=["playerID", "bbrefID"])
    crosswalk = load_id_crosswalk(cache_dir)
    merged = people.merge(crosswalk, on="bbrefID", how="inner")
    return merged[["playerID", "mlbam_id"]].drop_duplicates()


# ── 2) 거래내역 수집 ──────────────────────────────────────────────
@dataclass
class TransactionsFetcher:
    session: requests.Session

    def fetch_range(self, start_date: str, end_date: str) -> list[dict]:
        resp = self.session.get(
            TRANSACTIONS_URL,
            params={"startDate": start_date, "endDate": end_date},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("transactions", [])

    def fetch_year(self, year: int) -> list[dict]:
        """연도 하나를 월 단위로 쪼개서 호출한다 (한 번에 너무 큰 범위를 요청하지 않기 위함)."""
        all_tx: list[dict] = []
        for month in range(SEASON_MONTHS[0], SEASON_MONTHS[1] + 1):
            start = f"{year}-{month:02d}-01"
            end_month = month + 1 if month < 12 else 1
            end_year = year if month < 12 else year + 1
            end = f"{end_year}-{end_month:02d}-01"
            all_tx.extend(self.fetch_range(start, end))
            time.sleep(REQUEST_DELAY_SEC)
        return all_tx


def fetch_all_transactions(start_year: int, end_year: int) -> pd.DataFrame:
    fetcher = TransactionsFetcher(session=requests.Session())
    rows = []
    for year in range(start_year, end_year + 1):
        print(f"[transactions] {year} 수집 중...")
        for tx in fetcher.fetch_year(year):
            person = tx.get("person") or {}
            rows.append(
                {
                    "mlbam_id": person.get("id"),
                    "season": year,
                    "date": tx.get("date"),
                    "type_desc": tx.get("typeDesc"),
                    "description": tx.get("description", ""),
                }
            )
    return pd.DataFrame(rows)


# ── 3) 부상 관련 거래만 필터 ───────────────────────────────────────
def filter_injury_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    is_injury = tx["description"].str.contains(INJURY_PATTERNS, na=False)
    injury_tx = tx[is_injury].copy()
    injury_tx["is_activation"] = injury_tx["description"].str.contains(
        ACTIVATED_PATTERN, na=False
    )
    return injury_tx


# ── 4) 선수×시즌 단위 집계 ─────────────────────────────────────────
def aggregate_to_player_season(injury_tx: pd.DataFrame, id_map: pd.DataFrame) -> pd.DataFrame:
    merged = injury_tx.merge(id_map, on="mlbam_id", how="inner")  # mlbam_id -> playerID 역매핑
    placed = merged[~merged["is_activation"]]

    agg = (
        placed.groupby(["playerID", "season"])
        .agg(
            il_stint_count=("date", "count"),
            first_il_date=("date", "min"),
            injury_note_sample=("description", "first"),  # 참고용, 확정 사실 아님을 명시할 것
        )
        .reset_index()
    )
    agg["had_injury"] = 1  # 이 표에 있는 행은 전부 IL 등재가 확인된 선수·시즌
    return agg.rename(columns={"playerID": "player_id"})


# ── 실행 ───────────────────────────────────────────────────────────
def run(lahman_dir: Path, cache_dir: Path, out_path: Path) -> pd.DataFrame:
    id_map = build_player_id_map(lahman_dir / "People.csv", cache_dir)
    print(f"ID 매칭: {len(id_map):,}명")

    tx = fetch_all_transactions(START_YEAR, END_YEAR)
    print(f"전체 거래: {len(tx):,}건")

    injury_tx = filter_injury_transactions(tx)
    print(f"부상 관련 거래: {len(injury_tx):,}건")

    result = aggregate_to_player_season(injury_tx, id_map)
    print(f"선수x시즌 부상 기록: {len(result):,}건")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"저장 완료: {out_path}")
    return result


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent
    run(
        lahman_dir=ROOT / "data" / "raw" / "lahman",  # 실제 프로젝트 경로로 조정 필요
        cache_dir=ROOT / ".cache",
        out_path=ROOT / "player_injury_stints.csv",
    )
