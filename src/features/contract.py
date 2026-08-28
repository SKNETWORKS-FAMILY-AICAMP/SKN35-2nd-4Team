"""features_v1 계약 정의 + 실제 데이터 로더.

이 파일이 features_v1.parquet 의 **스키마 명세**다.
실제 데이터만 사용하며, 테스트용 목업/가짜 데이터는 제공하지 않는다.

담당: D (Day 1)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data" / "final" / "features_v1.parquet"

START_YEAR, END_YEAR = 2009, 2025
LABEL_END_YEAR = 2024          # 2025는 다음 시즌이 없어 라벨 생성 불가
SPLIT = {"train": (2009, 2021), "valid": (2022, 2023), "test": (2024, 2024)}

FA_ELIGIBLE_EXP = 6            # MLB FA 자격 서비스 타임. exp < 6 이면 FA 불가 (확정 규칙)

LEAGUE_CLASSES = ["mlb", "kbo"]
PATH_CLASSES = ["trade", "offseason_move", "league_exit"]        # L2 — 이탈자에게만 존재
FA_RELEASE_CLASSES = ["release_certain", "fa_est", "release_est"]  # L2b — offseason_move 에게만 존재
ROLE_CLASSES = ["B", "P", "TWO"]

# ── 계약 ───────────────────────────────────────────────────────────
# 컬럼 추가는 허용, 삭제·이름변경은 금지. 변경이 필요하면 features_v2 를 만든다.
# Rev.4 — 이탈 라벨을 4층(L1/L2/L2b/L3)으로 분리. FA와 방출은 의사결정 성격이
# 정반대이므로 하나의 클래스로 합치지 않는다 (12-2 절대 하지 말 것).
SCHEMA: dict[str, str] = {
    # 키
    "player_id": "object",
    "season": "int64",
    "team_last": "object",
    "franch_id": "object",
    "league": "object",        # mlb / kbo ← Rev.4 신규. 리그 전이의 파티션 키
    # 식별·역할
    "role": "object",          # B / P / TWO
    "age": "float64",          # 실전 피처로 사용 — 결측 불허 (validate() 에서 강제)
    "exp": "int64",            # 리그 경력 시즌 수 — L2b 판별 키 (FA_ELIGIBLE_EXP)
    "n_stint": "int64",        # >= 2 이면 시즌 중 트레이드
    # 출전
    "g_ratio": "float64",      # G / team_games. raw G 는 사용하지 않는다
    "g_ratio_prev": "float64",
    "g_chg": "float64",
    # 전력
    "off_score": "float64",    # 0~100, AB<50 이면 NaN
    "pit_score": "float64",    # 0~100, IP<30 이면 NaN
    "def_score": "float64",
    "overall_score": "float64",
    # 정규화 지표 (시즌 내 z-score)
    "ops_z": "float64",
    "ops_z_prev": "float64",
    "era_z": "float64",
    "whip_z": "float64",
    # 팀 맥락
    "team_wr": "float64",
    # 라벨 — 관측 가능성에 따라 4층으로 분리 (Rev.4 3장)
    "y_departed": "float64",     # L1  0/1 — 다음 시즌 동일 franch_id 인가
    "y_path": "object",         # L2  PATH_CLASSES — 이탈자만. 잔류자는 NaN
    "y_fa_release": "object",   # L2b FA_RELEASE_CLASSES — offseason_move 만. 그 외 NaN ← Rev.4 신규
    "y_returned": "float64",     # L3  리그이탈자만 0/1, 그 외 NaN
    "y_core_departed": "float64", # L1' 핵심 이탈위험 타깃; release_certain은 NaN
    "y_next_score": "float64",  # D 회귀 타깃: 다음 시즌 overall_score
}

KEY = ["player_id", "season"]
LABEL_COLS = ["y_departed", "y_path", "y_fa_release", "y_returned", "y_core_departed", "y_next_score"]
# league 는 모델 피처가 아니라 파티션 컨텍스트다. 피처로 넣으면 리그 중립성이 깨진다 (7-7)
FEATURE_COLS = [c for c in SCHEMA if c not in KEY + LABEL_COLS + ["team_last", "franch_id", "league"]]

# 리그 내 상대값이 아닌 절대 스탯 컬럼. features_v1 에 섞이면 KBO 전이가 깨지므로 차단한다 (7-7)
ABSOLUTE_STAT_COLS = {
    "h", "hr", "rbi", "so", "bb", "sb", "cs", "r", "2b", "3b",
    "w", "l", "sv", "ip", "er", "era", "whip",
    "ab", "pa", "g", "gs", "avg", "obp", "slg", "ops", "salary",
}



def validate(df: pd.DataFrame) -> None:
    """계약 위반을 즉시 잡는다. B의 실제 데이터도 이 검사를 통과해야 한다."""
    missing = set(SCHEMA) - set(df.columns)
    assert not missing, f"컬럼 누락: {sorted(missing)}"
    assert df.duplicated(KEY).sum() == 0, "player_id + season 중복"
    assert df.g_ratio.max() <= 1.05, f"g_ratio 이상: {df.g_ratio.max()}"
    assert df.season.between(START_YEAR, END_YEAR).all(), "학습 구간 이탈"

    # age 는 이제 실전 피처로 쓴다 — 결측 허용하지 않는다 (전력 나이 곡선 등에 필수)
    assert df.age.notna().all(), "age 결측 존재 — 나이 피처를 실제로 쓰기로 했으므로 결측 불허"
    assert df.age.between(15, 50).all(), f"age 범위 이상: {df.age.min()}~{df.age.max()}"

    bad_league = set(df.league.dropna()) - set(LEAGUE_CLASSES)
    assert not bad_league, f"정의되지 않은 league: {bad_league}"

    # 절대값 피처 차단 — 리그 내 상대값으로 통일되지 않으면 KBO 전이가 깨진다 (7-7)
    leaked = ABSOLUTE_STAT_COLS & {c.lower() for c in df.columns}
    assert not leaked, f"절대값 피처가 섞여 있음 (리그 내 상대값으로 변환할 것): {leaked}"

    lab = df[df.season <= LABEL_END_YEAR]

    bad_path = set(lab.y_path.dropna()) - set(PATH_CLASSES)
    assert not bad_path, f"정의되지 않은 y_path: {bad_path}"
    assert lab.y_departed.notna().all(), "라벨 구간에 y_departed 결측"
    # L2 는 이탈자에게만 존재한다 — 잔류자(y_departed=0)에 y_path 가 있으면 라벨링 오류
    stayed = lab.y_departed == 0
    assert lab.loc[stayed, "y_path"].isna().all(), "잔류자에게 y_path 가 존재함 (L1/L2 불일치)"
    departed = lab.y_departed == 1
    assert lab.loc[departed, "y_path"].notna().all(), "이탈자에게 y_path 가 결측 (L1/L2 불일치)"

    # L2b 는 오프시즌 이적자에게만 존재한다 — FA 와 방출을 하나로 합치지 않는다 (12-2)
    bad_fa = set(lab.y_fa_release.dropna()) - set(FA_RELEASE_CLASSES)
    assert not bad_fa, f"정의되지 않은 y_fa_release: {bad_fa}"
    offseason = lab.y_path == "offseason_move"
    assert lab.loc[offseason, "y_fa_release"].notna().all(), "오프시즌 이적자에게 y_fa_release 결측"
    assert lab.loc[~offseason, "y_fa_release"].isna().all(), "오프시즌 이적이 아닌데 y_fa_release 존재"
    # exp < 6 은 FA 자격이 제도상 불가능하다 — 한쪽 방향으로만 확정되는 비대칭 규칙 (3-3)
    certain = lab.y_fa_release == "release_certain"
    assert (lab.loc[certain, "exp"] < FA_ELIGIBLE_EXP).all(), "release_certain 인데 exp>=6 (서비스 타임 규칙 위반)"

    bad_core = set(lab.y_core_departed.dropna().astype(float)) - {0.0, 1.0}
    assert not bad_core, f"정의되지 않은 y_core_departed: {bad_core}"
    assert lab.loc[certain, "y_core_departed"].isna().all(), "release_certain 행의 y_core_departed는 결측이어야 함"
    non_release = ~certain
    assert (lab.loc[non_release, "y_core_departed"].reset_index(drop=True) ==
            lab.loc[non_release, "y_departed"].reset_index(drop=True)).all(), "y_core_departed와 y_departed 불일치"

    # 시즌별 정규화가 됐는지 — 연도별 평균 전력이 비슷해야 한다
    drift = df.groupby("season").overall_score.mean().std()
    assert drift < 5, f"시즌 간 전력 평균 편차 과다: {drift:.2f}"


def split(df: pd.DataFrame, part: str) -> pd.DataFrame:
    lo, hi = SPLIT[part]
    return df[df.season.between(lo, hi)]


def load_features() -> pd.DataFrame:
    """실제 features_v1.parquet만 로드하고 계약을 검증한다."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"실제 features 파일이 없습니다: {FEATURES_PATH}. "
            "목업 데이터로 대체하지 않고 실제 데이터 연결을 확인하세요."
        )

    df = pd.read_parquet(FEATURES_PATH)
    validate(df)
    return df
