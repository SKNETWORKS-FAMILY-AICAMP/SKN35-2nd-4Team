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

import numpy as np
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


def fetch_all_transactions(start_year: int, end_year: int,
                           cache_dir: Path | None = None) -> pd.DataFrame:
    """MLB 거래 기록 수집. 연도별로 캐시해 재실행 비용을 없앤다.

    18개 시즌치를 매번 API 에서 다시 받으면 수 분이 걸리고 MLB 서버에도
    부담이라, 연도 단위로 캐시한다. 진행 중인 시즌(END_YEAR)은 계속 갱신되므로
    캐시하지 않는다 — 오래된 스냅샷을 확정 데이터처럼 쓰면 안 된다.
    """
    fetcher = TransactionsFetcher(session=requests.Session())
    frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        cache_path = (cache_dir / f"transactions-{year}.csv") if cache_dir else None
        is_current = year >= END_YEAR  # 진행 중 시즌은 캐시 금지
        if cache_path and cache_path.exists() and not is_current:
            frames.append(pd.read_csv(cache_path))
            print(f"[transactions] {year} 캐시 사용")
            continue

        print(f"[transactions] {year} 수집 중...")
        year_rows = []
        for tx in fetcher.fetch_year(year):
            person = tx.get("person") or {}
            year_rows.append(
                {
                    "mlbam_id": person.get("id"),
                    "season": year,
                    "date": tx.get("date"),
                    "type_desc": tx.get("typeDesc"),
                    "description": tx.get("description", ""),
                }
            )
        year_df = pd.DataFrame(year_rows)
        if cache_path and not is_current and not year_df.empty:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            year_df.to_csv(cache_path, index=False)
        frames.append(year_df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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
FULL_SEASON_DAYS = 180   # 부상 심각도 스케일 기준 — 한 시즌 통째 결장 = 1.0
SEASON_END_MONTH, SEASON_END_DAY = 10, 1  # 복귀 미확인 스틴트의 결장 종료 추정 시점
SEASON_SPAN_DAYS = 214  # 3/1~10/1 — 한 시즌에 결장 가능한 최대 일수


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

    # 복귀 거래가 안 잡힌 스틴트 = 그 시즌 안에 복귀가 관측되지 않음.
    # 시즌 종료일(대략 10/1)까지 계속 결장한 것으로 보고 일수를 추정한다.
    # 임의 상수(예전의 0.7 하한)와 달리 등재 시점에 따라 값이 달라져서
    # 분포가 한 점에 뭉치지 않는다. 관측된 회복과 섞이지 않게 별도 컬럼.
    season_end = pd.to_datetime(
        matched["season"].astype(int).astype(str) + f"-{SEASON_END_MONTH:02d}-{SEASON_END_DAY:02d}"
    )
    est = (season_end - matched["date"]).dt.days
    matched["estimated_days"] = (
        est.where(matched["recovery_days"].isna())
        .clip(lower=0, upper=MAX_RECOVERY_DAYS)
        .fillna(0)
    )
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
            # sum 이 아니라 max: 미복귀 스틴트가 여러 개면 각각 "등재일~시즌종료"를
            # 더하게 되어 같은 잔여기간을 중복으로 센다(실측: Verlander 2026 이
            # 추정 324일 = 한 시즌보다 길게 나왔다). 가장 이른 미복귀 등재일부터
            # 시즌 끝까지가 곧 최대 추정 결장이므로 max 가 맞다.
            _est_days=("estimated_days", "max"),
        )
        .reset_index()
    )
    estimated_days = agg.pop("_est_days")
    agg["had_injury"] = 1  # 이 표에 있는 행은 전부 IL 등재가 확인된 선수·시즌

    # ── 부상 심각도 점수 ────────────────────────────────────────────
    # reason.py가 이미 기대하고 있던 컬럼명(MODEL_FEATURE_CANDIDATES,
    # add_reason_features의 "injury_risk_score" 분기) 그대로 채운다 -
    # reason.py 쪽은 전혀 안 고쳐도 이 컬럼이 생기는 순간 자동으로 쓰인다.
    #
    # [2026-08-30 수정] 이전 계산식은 (결장일수/60).clip(upper=1) 에
    # "복귀 미확인이면 0.7로 바닥 보정"이었는데, 0~1 척도에 값이 두 군데로
    # 뭉치는 문제가 있었다(실측: 양수 17,679건 중 0.70에 25.1%, 1.00에 29.5%
    # — 합쳐서 54.6%가 딱 두 값). 그 결과
    #   (1) reason.py가 상위 25% 지점으로 잡는 임계값이 천장 1.00에 붙어버려
    #       "천장에 닿았는가"만 묻는 사실상 이진 판정이 됐고,
    #   (2) 60일 이상 결장(부상자의 37.7%)이 전부 1.00으로 뭉개져 60일과
    #       400일 결장이 구분되지 않았으며,
    #   (3) 0.70은 측정된 심각도가 아니라 임의의 하한인데 화면에는 실측치처럼
    #       보였다.
    # 두 가지를 바꾼다:
    #   A. 복귀 기록이 없는 스틴트는 "시즌 종료까지 결장한 것"으로 보고 IL
    #      등재일부터 시즌 종료일까지를 추정 결장일수로 쓴다 — 날짜에 따라
    #      값이 달라지므로 한 점에 뭉치지 않고, 임의 상수보다 근거가 있다.
    #   B. 선형 60일 상한 대신 한 시즌(180일) 기준 제곱근 스케일을 쓴다.
    #      제곱근이라 짧은 IL 사이의 차이는 살아있고, 장기 결장도 천장에
    #      닿기 전까지 계속 구분된다.
    # 추정으로 채운 일수는 injury_days_estimated 로 따로 남겨 화면에서
    # "측정값"과 "추정값"을 구분할 수 있게 한다.
    agg["injury_days_estimated"] = estimated_days
    # 한 시즌에 결장할 수 있는 최대치로 자른다. match_recovery_dates 의 알려진
    # 한계(같은 복귀 거래가 여러 등재에 중복 매칭될 수 있음) 때문에 합계가
    # 비현실적으로 커질 수 있다 — 실측에서 한 선수·시즌 유효일수가 1,792일까지
    # 나왔다. 점수는 어차피 FULL_SEASON_DAYS 에서 포화하지만, 저장되는 일수
    # 자체가 말이 안 되면 나중에 그 컬럼을 쓰는 쪽이 오해한다.
    effective_days = (
        agg["total_recovery_days"] + agg["injury_days_estimated"]
    ).clip(upper=SEASON_SPAN_DAYS)
    agg["injury_effective_days"] = effective_days
    agg["injury_risk_score"] = np.sqrt(
        (effective_days / FULL_SEASON_DAYS).clip(lower=0.0, upper=1.0)
    )

    return agg.rename(columns={"playerID": "player_id"})


# ── 실행 ───────────────────────────────────────────────────────────
def run(lahman_dir: Path, cache_dir: Path, out_path: Path) -> pd.DataFrame:
    id_map = build_player_id_map(lahman_dir / "People.csv", cache_dir)
    print(f"ID 매칭: {len(id_map):,}명")

    tx = fetch_all_transactions(START_YEAR, END_YEAR, cache_dir=cache_dir)
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
