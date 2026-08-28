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
# strength.py의 YEAR_FLOOR(2009)보다 이전 데이터는 어차피 안 쓰여서 재수집
# 범위에서 뺐다 - API 호출 수를 줄인다. END_YEAR는 2026(진행중 시즌)까지 포함.
START_YEAR = 2009
END_YEAR = 2026
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


# ── 3b) 복귀일 매칭 (2026-08-28 추가) ────────────────────────────────
# 예전엔 "activated"(복귀) 거래를 감지만 해두고 실제로 안 썼다 - IL 등재일만
# 있고 해제일이 없어서 "실제 며칠 결장했는지"를 몰랐다(docs/label_spec.md
# Rev.5 초안이 지적한 바로 그 한계). merge_asof로 각 등재 거래에 그 이후
# 가장 가까운 같은 선수의 "activated" 거래를 매칭해 복귀일을 근사한다.
#
# 한계 (근사치임, 확정 아님) - 정직하게 문서화해둔다:
#   - 한 선수가 매칭 전에 여러 번 등재되면(재등재 등) 같은 복귀 거래가
#     여러 등재에 중복 매칭될 수 있다.
#   - 시즌이 끝날 때까지 복귀 거래가 안 잡히면(시즌아웃 부상 등) 복귀일은
#     결측으로 남는다 - "복귀 못함"으로 임의 확정하지 않는다.
#   - MAX_RECOVERY_DAYS 제한 없이 그냥 forward 매칭만 하면, 어떤 선수는
#     그 뒤로 "activated" 거래가 수년간 하나도 없다가 완전히 무관한(다른
#     부상의) 복귀 거래에 매칭돼 복귀일수가 9,000일 넘게 나오는 사고가
#     실측으로 발생했다 - 명백히 다른 사건인데 잘못 이어붙인 것이므로
#     tolerance로 매칭 자체를 막는다(억지로 값을 만들지 않고 결측 처리).
MAX_RECOVERY_DAYS = 400  # 토미존 서저리 등 최장기 재활도 넉넉히 포함하는 상한


def match_recovery_dates(injury_tx: pd.DataFrame, id_map: pd.DataFrame) -> pd.DataFrame:
    merged = injury_tx.merge(id_map, on="mlbam_id", how="inner")
    merged["date"] = pd.to_datetime(merged["date"])

    placements = merged[~merged["is_activation"]].sort_values("date").reset_index(drop=True)
    activations = (
        merged[merged["is_activation"]][["playerID", "date"]]
        .rename(columns={"date": "return_date"})
        .sort_values("return_date")
        .reset_index(drop=True)
    )

    matched = pd.merge_asof(
        placements, activations,
        left_on="date", right_on="return_date",
        by="playerID", direction="forward",
        tolerance=pd.Timedelta(days=MAX_RECOVERY_DAYS),
    )
    matched["recovery_days"] = (matched["return_date"] - matched["date"]).dt.days
    return matched


# ── 4) 선수×시즌 단위 집계 ─────────────────────────────────────────
def aggregate_to_player_season(injury_tx: pd.DataFrame, id_map: pd.DataFrame) -> pd.DataFrame:
    matched = match_recovery_dates(injury_tx, id_map)

    agg = (
        matched.groupby(["playerID", "season"])
        .agg(
            il_stint_count=("date", "count"),
            first_il_date=("date", "min"),
            injury_note_sample=("description", "first"),  # 참고용, 확정 사실 아님을 명시할 것
            total_recovery_days=("recovery_days", lambda s: s.fillna(0).sum()),
            unresolved_stints=("recovery_days", lambda s: s.isna().sum()),
        )
        .reset_index()
    )
    agg["had_injury"] = 1  # 이 표에 있는 행은 전부 IL 등재가 확인된 선수·시즌

    # reason.py가 이미 기대하고 있던 컬럼명(MODEL_FEATURE_CANDIDATES,
    # add_reason_features의 "injury_risk_score" 분기) 그대로 채운다 -
    # reason.py 쪽은 전혀 안 고쳐도 이 컬럼이 생기는 순간 자동으로 쓰인다.
    # 60일(장기 IL 기준) 대비 실제 결장일수 비율로 0~1 스케일.
    agg["injury_risk_score"] = (agg["total_recovery_days"] / 60).clip(upper=1.0)
    # 복귀 거래가 안 잡힌 스틴트(시즌 안에 복귀 못 봄 - 심각했을 가능성)가
    # 있으면, 관측된 회복일수만으로 과소평가되지 않게 최소 심각도를 보정한다.
    unresolved = agg["unresolved_stints"] > 0
    agg.loc[unresolved, "injury_risk_score"] = agg.loc[unresolved, "injury_risk_score"].clip(lower=0.7)

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
        lahman_dir=ROOT / "data" / "raw" / "lahman",
        cache_dir=ROOT / ".cache",
        out_path=ROOT / "data" / "final" / "player_injury_stints.csv",  # build.py가 직접 읽는 위치
    )
