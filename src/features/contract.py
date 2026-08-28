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
    "age": "float64",
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
    "allstar": "int64",
    # 부상 — MLB Stats API transactions(IL 등재 기록) 기반 실측치.
    # Lahman 자체엔 부상자 명단이 없어 "추정"만 가능하다고 했었는데, 이제 실제 관측값이라
    # L3 해석 레이어가 아니라 당당한 피처로 들어간다. exp<6 규칙처럼 한쪽은 확정이다:
    # had_injury=1(그 시즌 IL 등재 있었음)은 확정, =0은 "기록 매칭 실패"와 "진짜 무사고"가
    # 섞여 있을 수 있음 — ID 크로스워크가 못 찾은 선수는 0으로 채워지기 때문 (완전한 반증은 아님).
    "had_injury": "int64",     # 그 시즌 IL 등재 여부 0/1
    "il_stint_count": "int64", # 그 시즌 IL 등재 횟수
    # 라벨 — 관측 가능성에 따라 4층으로 분리 (Rev.4 3장)
    "y_departed": "float64",     # L1  0/1 — 다음 시즌 동일 franch_id 인가
    "y_path": "object",         # L2  PATH_CLASSES — 이탈자만. 잔류자는 NaN
    "y_fa_release": "object",   # L2b FA_RELEASE_CLASSES — offseason_move 만. 그 외 NaN ← Rev.4 신규
    "y_returned": "float64",     # L3  리그이탈자만 0/1, 그 외 NaN
    "y_next_score": "float64",  # D 회귀 타깃: 다음 시즌 overall_score
}

KEY = ["player_id", "season"]
LABEL_COLS = ["y_departed", "y_path", "y_fa_release", "y_returned", "y_next_score"]
# league 는 모델 피처가 아니라 파티션 컨텍스트다. 피처로 넣으면 리그 중립성이 깨진다 (7-7)
FEATURE_COLS = [c for c in SCHEMA if c not in KEY + LABEL_COLS + ["team_last", "franch_id", "league"]]

# 리그 내 상대값이 아닌 절대 스탯 컬럼. features_v1 에 섞이면 KBO 전이가 깨지므로 차단한다 (7-7)
ABSOLUTE_STAT_COLS = {
    "h", "hr", "rbi", "so", "bb", "sb", "cs", "r", "2b", "3b",
    "w", "l", "sv", "ip", "er", "era", "whip",
    "ab", "pa", "g", "gs", "avg", "obp", "slg", "ops", "salary",
}


def make_mock(n_players: int = 2200, seed: int = 42) -> pd.DataFrame:
    """계약을 만족하는 가짜 데이터. 실제 데이터의 분포를 대략 흉내낸다.

    라벨 분포는 실측값에 맞춤 (Rev.4 3-6): 잔류 50.6 / 오프시즌이적 22.6
    (방출확정 68.2% + FA·방출추정 31.8%) / 리그이탈 20.6 (복귀 23.3%) / 트레이드 6.2
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
                    franch_id=team, league="mlb", role=role, age=float(age), exp=k,
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
                    # 실측 기준 대략적인 시즌당 IL 등재 비율(~18%)을 흉내낸다
                    had_injury=int(rng.random() < 0.18),
                    il_stint_count=0,  # 아래서 had_injury=1 인 행만 다시 채움
                )
            )
            if rng.random() < 0.40:      # 이적
                team = rng.choice(teams)

    df = pd.DataFrame(rows).sort_values(KEY).reset_index(drop=True)

    injured = df.had_injury == 1
    df.loc[injured, "il_stint_count"] = rng.integers(1, 3, size=int(injured.sum()))
    # 부상 있었던 시즌은 출전 비중이 깎이는 게 자연스럽다
    df.loc[injured, "g_ratio"] = (df.loc[injured, "g_ratio"] * rng.uniform(0.4, 0.85, size=int(injured.sum()))).clip(0.02, 1.0)

    g = df.groupby("player_id")
    df["g_ratio_prev"] = g.g_ratio.shift(1)
    df["g_chg"] = df.g_ratio / df.g_ratio_prev.replace(0, np.nan)
    df["ops_z_prev"] = g.ops_z.shift(1)
    df["y_next_score"] = g.overall_score.shift(-1)

    # 라벨 — 판별 규칙 3-5 와 동일한 로직으로 생성
    nxt_season = g.season.shift(-1)
    nxt_franch = g.franch_id.shift(-1)
    has_next = nxt_season == df.season + 1              # 다음 시즌 리그 내 기록 존재
    same_franch = has_next & (nxt_franch == df.franch_id)

    # L1 — 다음 시즌 동일 franch_id 가 아니면 이탈
    df["y_departed"] = (~same_franch).astype(float)

    # L2 — 트레이드(시즌 중 팀 변경) 우선, 다음은 오프시즌 이적, 나머지는 리그이탈
    is_trade = df.n_stint >= 2
    is_offseason_move = has_next & ~same_franch & ~is_trade
    is_league_exit = ~has_next & ~is_trade
    df["y_path"] = np.select(
        [is_trade, is_offseason_move, is_league_exit],
        PATH_CLASSES,
        default=None,
    )
    df.loc[same_franch, "y_path"] = None                # 잔류자는 이탈 경로가 없다

    # L2b — 서비스 타임 규칙 (Rev.4 3-3). exp<6 은 방출 확정, exp>=6 은 성적 추이로 추정
    score_trend = df.overall_score - g.overall_score.shift(1)
    is_offseason = df.y_path == "offseason_move"
    release_certain = is_offseason & (df.exp < FA_ELIGIBLE_EXP)
    release_est = is_offseason & (df.exp >= FA_ELIGIBLE_EXP) & (score_trend < -5)
    fa_est = is_offseason & (df.exp >= FA_ELIGIBLE_EXP) & ~(score_trend < -5)
    df["y_fa_release"] = np.select(
        [release_certain, release_est, fa_est],
        FA_RELEASE_CLASSES,
        default=None,
    )

    # L3 — 리그이탈자가 t+2 이내 재등장하는가
    seasons = g.season.apply(set).to_dict()
    df["y_returned"] = [
        float(bool(seasons[p] & {s + 2, s + 3})) if r == "league_exit" else np.nan
        for p, s, r in zip(df.player_id, df.season, df.y_path)
    ]

    # 라벨 생성 불가 구간 마스킹 — 2025는 다음 시즌이 없어 라벨을 만들 수 없다
    last = df.season > LABEL_END_YEAR
    df.loc[last, ["y_departed", "y_returned", "y_next_score"]] = np.nan
    df.loc[last, ["y_path", "y_fa_release"]] = None

    return df[list(SCHEMA)]


def validate(df: pd.DataFrame) -> None:
    """계약 위반을 즉시 잡는다. B의 실제 데이터도 이 검사를 통과해야 한다."""
    missing = set(SCHEMA) - set(df.columns)
    assert not missing, f"컬럼 누락: {sorted(missing)}"
    assert df.duplicated(KEY).sum() == 0, "player_id + season 중복"
    assert df.g_ratio.max() <= 1.05, f"g_ratio 이상: {df.g_ratio.max()}"
    assert df.season.between(START_YEAR, END_YEAR).all(), "학습 구간 이탈"
    assert df.had_injury.isin([0, 1]).all(), "had_injury 는 0/1 이어야 함"
    assert (df.il_stint_count >= 0).all(), "il_stint_count 는 음수일 수 없음"
    assert (df.loc[df.had_injury == 0, "il_stint_count"] == 0).all(), "부상 없는데 il_stint_count>0"

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
    print("\nL2 라벨 분포(%) — 잔류자는 NaN")
    print((d.y_path.value_counts(normalize=True) * 100).round(1).to_string())
    print("\nL2b 라벨 분포(%) — 오프시즌 이적자만")
    print((d.y_fa_release.value_counts(normalize=True) * 100).round(1).to_string())
    print("\n분할")
    for p in SPLIT:
        print(f"  {p:6s} {len(split(d, p)):,}")
    print(f"\n피처 {len(FEATURE_COLS)}개: {FEATURE_COLS}")