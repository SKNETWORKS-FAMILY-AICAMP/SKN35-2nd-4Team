"""features_v1 계약 정의 + 목업 생성기.

이 파일이 features_v1.parquet 의 **스키마 명세**다.
B는 이 스키마를 만족하는 실제 데이터를 만들고, 그 전까지 나머지 인원은
make_mock() 으로 생성한 가짜 데이터로 모델·평가·화면을 개발한다.

실제 파일이 도착하면 load_features() 가 자동으로 실제 데이터를 쓴다.
호출부 코드는 한 줄도 바꿀 필요가 없다.

담당: D (Day 1)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data" / "processed" / "features_v1.parquet"

START_YEAR, END_YEAR = 2000, 2025
LABEL_END_YEAR = 2024          # 2025는 다음 시즌이 없어 라벨 생성 불가
SPLIT = {"train": (2000, 2021), "valid": (2022, 2023), "test": (2024, 2024)}

REASON_CLASSES = ["stay", "trade", "fa_release", "league_exit"]
ROLE_CLASSES = ["B", "P", "TWO"]

# ── 계약 ───────────────────────────────────────────────────────────
# 컬럼 추가는 허용, 삭제·이름변경은 금지. 변경이 필요하면 features_v2 를 만든다.
SCHEMA: dict[str, str] = {
    # 키
    "player_id": "object",
    "season": "int64",
    "team_last": "object",
    "franch_id": "object",
    # 식별·역할
    "role": "object",          # B / P / TWO
    "age": "float64",
    "exp": "int64",            # 리그 경력 시즌 수
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
    "allstar": "int64",
    # 라벨
    "y_departed": "float64",   # L1  0/1
    "y_reason": "object",      # L2  REASON_CLASSES
    "y_returned": "float64",   # L3  리그이탈자만 0/1, 그 외 NaN
    "y_next_score": "float64", # D 회귀 타깃: 다음 시즌 overall_score
}

KEY = ["player_id", "season"]
LABEL_COLS = ["y_departed", "y_reason", "y_returned", "y_next_score"]
FEATURE_COLS = [c for c in SCHEMA if c not in KEY + LABEL_COLS + ["team_last", "franch_id"]]


def make_mock(n_players: int = 2200, seed: int = 42) -> pd.DataFrame:
    """계약을 만족하는 가짜 데이터. 실제 데이터의 분포를 대략 흉내낸다.

    라벨 분포는 실측값에 맞춤: 잔류 50.6 / FA·방출 22.6 / 리그이탈 20.6 / 트레이드 6.2
    """
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(30)]
    rows = []

    for i in range(n_players):
        pid = f"mock{i:05d}"
        debut = int(rng.integers(START_YEAR, END_YEAR))
        career = int(rng.integers(1, 11))   # 상한은 아래 break 가 처리한다
        team = rng.choice(teams)
        role = rng.choice(ROLE_CLASSES, p=[0.62, 0.36, 0.02])
        peak = rng.normal(60, 15)

        for k in range(career):
            season = debut + k
            if season > END_YEAR:
                break
            age = 22 + k + rng.integers(0, 4)
            # 27세 정점의 aging curve
            curve = -0.045 * (age - 27) ** 2
            score = float(np.clip(peak + curve + rng.normal(0, 4), 5, 99))
            g_ratio = float(np.clip(rng.beta(2, 2) * 1.05, 0.02, 1.0))
            rows.append(
                dict(
                    player_id=pid, season=season, team_last=team,
                    franch_id=team, role=role, age=float(age), exp=k,
                    n_stint=int(rng.choice([1, 2, 3], p=[0.86, 0.12, 0.02])),
                    g_ratio=g_ratio,
                    off_score=score if role in ("B", "TWO") else np.nan,
                    pit_score=score if role in ("P", "TWO") else np.nan,
                    def_score=float(np.clip(rng.normal(55, 12), 0, 100)),
                    overall_score=score,
                    ops_z=float(rng.normal(0, 1)),
                    era_z=float(rng.normal(0, 1)),
                    whip_z=float(rng.normal(0, 1)),
                    team_wr=float(np.clip(rng.normal(0.5, 0.07), 0.25, 0.75)),
                    allstar=int(rng.random() < 0.05),
                )
            )
            if rng.random() < 0.40:      # 이적
                team = rng.choice(teams)

    df = pd.DataFrame(rows).sort_values(KEY).reset_index(drop=True)

    g = df.groupby("player_id")
    df["g_ratio_prev"] = g.g_ratio.shift(1)
    df["g_chg"] = df.g_ratio / df.g_ratio_prev.replace(0, np.nan)
    df["ops_z_prev"] = g.ops_z.shift(1)
    df["y_next_score"] = g.overall_score.shift(-1)

    # 라벨 — 실제 판별 로직과 동일한 규칙으로 생성
    nxt_season = g.season.shift(-1)
    nxt_franch = g.franch_id.shift(-1)
    in_league = nxt_season == df.season + 1
    same_team = nxt_franch == df.franch_id

    df["y_reason"] = np.select(
        [in_league & same_team, df.n_stint >= 2, in_league],
        ["stay", "trade", "fa_release"],
        default="league_exit",
    )
    df["y_departed"] = (df.y_reason != "stay").astype(float)

    # L3 — 리그이탈자가 t+2 이내 재등장하는가
    seasons = g.season.apply(set).to_dict()
    df["y_returned"] = [
        float(bool(seasons[p] & {s + 2, s + 3})) if r == "league_exit" else np.nan
        for p, s, r in zip(df.player_id, df.season, df.y_reason)
    ]

    # 라벨 생성 불가 구간 마스킹
    last = df.season > LABEL_END_YEAR
    df.loc[last, LABEL_COLS] = np.nan
    df.loc[last, "y_reason"] = None

    return df[list(SCHEMA)]


def validate(df: pd.DataFrame) -> None:
    """계약 위반을 즉시 잡는다. B의 실제 데이터도 이 검사를 통과해야 한다."""
    missing = set(SCHEMA) - set(df.columns)
    assert not missing, f"컬럼 누락: {sorted(missing)}"
    assert df.duplicated(KEY).sum() == 0, "player_id + season 중복"
    assert df.g_ratio.max() <= 1.05, f"g_ratio 이상: {df.g_ratio.max()}"
    assert df.season.between(START_YEAR, END_YEAR).all(), "학습 구간 이탈"

    lab = df[df.season <= LABEL_END_YEAR]
    bad = set(lab.y_reason.dropna()) - set(REASON_CLASSES)
    assert not bad, f"정의되지 않은 y_reason: {bad}"
    assert lab.y_departed.notna().all(), "라벨 구간에 y_departed 결측"

    # 시즌별 정규화가 됐는지 — 연도별 평균 전력이 비슷해야 한다
    drift = df.groupby("season").overall_score.mean().std()
    assert drift < 5, f"시즌 간 전력 평균 편차 과다: {drift:.2f}"


def split(df: pd.DataFrame, part: str) -> pd.DataFrame:
    lo, hi = SPLIT[part]
    return df[df.season.between(lo, hi)]


def load_features(allow_mock: bool = True) -> pd.DataFrame:
    """실제 파일이 있으면 그것을, 없으면 목업을 반환한다.

    B가 features_v1.parquet 을 커밋하는 순간, 호출부 수정 없이 실제 데이터로 전환된다.
    """
    if FEATURES_PATH.exists():
        df = pd.read_parquet(FEATURES_PATH)
        validate(df)
        return df
    if not allow_mock:
        raise FileNotFoundError(f"{FEATURES_PATH} 없음")
    print(f"[mock] {FEATURES_PATH.name} 없음 — 목업 데이터 사용 중")
    df = make_mock()
    validate(df)
    return df


if __name__ == "__main__":
    d = make_mock()
    validate(d)
    print(f"rows {len(d):,}  cols {d.shape[1]}")
    print("\n라벨 분포(%)")
    print((d.y_reason.value_counts(normalize=True) * 100).round(1).to_string())
    print("\n분할")
    for p in SPLIT:
        print(f"  {p:6s} {len(split(d, p)):,}")
    print(f"\n피처 {len(FEATURE_COLS)}개: {FEATURE_COLS}")