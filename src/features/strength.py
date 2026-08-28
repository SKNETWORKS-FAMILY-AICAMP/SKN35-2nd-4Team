"""
strength.py (v4 - 2009년 이후로 범위 축소 + 부상 페널티 공식 통합)
B - 전력 점수 계산 모듈

=== v4 변경 사항 ===
1. YEAR_FLOOR = 2009 로 전체 데이터 범위 축소.
   부상 데이터(player_injury_stints.csv)가 2000~2008년은 기록이 거의 없다가
   (연 5~15건) 2009년부터 갑자기 촘촘해짐(연 700건+) - 실제로 부상이 급증한 게
   아니라 기록 수집 체계가 2009년쯤부터 갖춰진 것으로 보임. 이 상태로 전체
   기간에 부상 페널티를 적용하면 2000년대 초반 선수만 부당하게 페널티를 덜
   받는 시대 편향이 생겨서, 아예 계산 범위를 2009년부터로 맞춤.
   [팀 공유 필요] 이 변경으로 D의 팀 공통 분할 기준(학습 1998/2000~2021)이
   실질적으로 '학습 2009~2021'이 됨 - A/C/D/E 전원에게 공유 필요.

2. 부상 페널티를 전력점수 공식에 직접 통합 (별도 피처 아님, 사용자 결정).
   il_stint_count(부상 횟수) 비례 페널티: 1회당 5% 감점, 최대 30% 캐핑.
   playing_time_ratio와 마찬가지로 곱연산으로 적용한 뒤, 시즌 내 Min-Max로
   0~100 재정규화 (두 개의 곱연산 할인 요소가 겹쳐도 스케일 해석이 유지되도록).

=== 이전 이력 (v1~v3) ===
v1: 원본 Lahman CSV (2B, 3B, R 모두 존재) 기준
v2: A의 1차 DB 적재본에서 2B/3B/R이 빠져서 SLG 근사(H+3*HR) + R가중치
    재분배(0.588/0.235/0.176)로 임시 대응
v3: A가 batting_stats.csv에 R, 2B, 3B 컬럼을 추가 -> 원래 공식 복원.
    pitching_stats.csv는 ER 대신 era만 있어 ER=era*IP/9로 역산.
    teams.csv에 팀 전체 IPouts가 없어 pitching_stats를 팀-연도 단위로 합산.
"""

import pandas as pd
import numpy as np

YEAR_FLOOR = 2009  # 부상 데이터 신뢰 구간 시작 - 전체 계산이 이 연도부터 시작됨

# ── shrinkage K: 연도별 적응형 ──────────────────────────────────────
# 예전엔 K를 연도 무관 고정값(타자 50/투수 20/수비 150)으로 썼다. 문제는
# 2020년(60경기 단축시즌)처럼 표본 자체가 작은 해에는 같은 K가 상대적으로
# 훨씬 더 강하게 리그 평균 쪽으로 당겨버리고, 반대로 그 해 중앙값 표본이
# 원래보다 큰 해에는 상대적으로 덜 당긴다 - 절대값 K는 "그 해 표본이 원래
# 이 정도는 된다"는 가정에 암묵적으로 의존하는데, 실제로는 해마다 다르다.
#
# 그래서 K를 고정값이 아니라 "그 해 중앙값 표본의 일정 비율"로 다시 정의한다
# (FRACTION은 2009~2019년 평균 중앙값 기준으로 예전 고정 K와 같은 강도가
# 나오도록 역산한 값 - 정상 시즌에서는 이전과 거의 동일하게 동작하고,
# 2020년 같은 이례적인 해에만 자동으로 조정된다). 이게 "가중치를 연도별로
# 더 촘촘하게" 계산해달라는 요청의 핵심 — shrinkage 강도 자체가 그 해의
# 실제 데이터 규모에 맞춰 매년 다시 계산된다.
BATTING_SHRINKAGE_FRACTION = 0.72   # 2009~2019 평균 중앙값 AB(≈69) 기준 K≈50 재현
PITCHING_SHRINKAGE_FRACTION = 0.44  # 평균 중앙값 IP(≈45) 기준 K≈20 재현
FIELDING_SHRINKAGE_FRACTION = 5.7   # 평균 중앙값 TC(≈26) 기준 K≈150 재현
SHRINKAGE_K_FLOOR = 5  # 극단적으로 표본이 작은 해에도 shrinkage가 사실상 사라지지 않게 하는 최소값


def _adaptive_k(sample_size: pd.Series, year: pd.Series, fraction: float) -> pd.Series:
    """그 해 중앙값 표본 * fraction. SHRINKAGE_K_FLOOR 아래로는 내려가지 않는다."""
    median_by_year = sample_size.groupby(year).transform("median")
    return (median_by_year * fraction).clip(lower=SHRINKAGE_K_FLOOR)

# 타자 전력 가중치 (2B/3B/R 데이터 복원됨 - 원래 문서 공식 그대로 사용)
BATTING_WEIGHTS = {"OPS": 0.5, "HR": 0.2, "RBI": 0.15, "R": 0.15}

# 수비 전력 가중치. Lahman Fielding.csv에는 타구 위치·난이도 데이터가 없어
# UZR/DRS 같은 정밀 지표는 계산 불가 - PO(자살)/A(보살)/E(실책)/G(경기)로
# 구할 수 있는 가장 기본적인 근사치만 쓴다(v1 한계, 문서화된 채로 남겨둠).
#   FPCT(수비율)  = (PO+A) / (PO+A+E)      -- 정확도
#   RFG(Range Factor per Game) = (PO+A) / G -- 수비 관여도(활동 범위)
# 포지션 난이도 보정(유격수 vs 1루수 등)은 하지 않는다 - 같은 기준으로 비교됨.
FIELDING_WEIGHTS = {"FPCT": 0.4, "RFG": 0.6}

# 부상 페널티: il_stint_count(부상 횟수) 1회당 5% 감점, 최대 30%까지 캐핑
INJURY_PENALTY_PER_STINT = 0.05
INJURY_PENALTY_CAP = 0.30


def standardize_injuries(df: pd.DataFrame) -> pd.DataFrame:
    """player_injury_stints.csv 표준화. player_id+season 유니크, 중복 없음(사전 확인됨)."""
    return df.rename(columns={
        "player_id": "playerID", "season": "yearID",
    })[["playerID", "yearID", "il_stint_count"]]


def _apply_injury_penalty(df: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """
    부상 횟수(il_stint_count) 비례 페널티. 기록 없는 선수-연도는 0회로 간주
    (YEAR_FLOOR=2009부터만 계산하므로 부상 데이터 신뢰 구간 내에서만 적용됨).
    """
    out = df.merge(injuries, on=["playerID", "yearID"], how="left")
    out["il_stint_count"] = out["il_stint_count"].fillna(0)
    out["injury_penalty_factor"] = 1 - (out["il_stint_count"] * INJURY_PENALTY_PER_STINT).clip(upper=INJURY_PENALTY_CAP)
    return out


# ---------------------------------------------------------------------------
# 컬럼 표준화 (새 스키마 -> 기존 내부 로직이 쓰던 이름)
# ---------------------------------------------------------------------------

def standardize_batting(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={
        "player_id": "playerID", "year": "yearID", "team_id": "teamID",
        "g": "G", "ab": "AB", "h": "H", "hr": "HR", "rbi": "RBI",
        "bb": "BB", "hbp": "HBP", "sf": "SF", "r": "R", "2b": "2B", "3b": "3B",
    })


def standardize_pitching(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={
        "player_id": "playerID", "year": "yearID", "team_id": "teamID",
        "g": "G", "gs": "GS", "ipouts": "IPouts", "era": "ERA",
        "h": "H", "bb": "BB", "so": "SO",
    })


def standardize_fielding(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={
        "player_id": "playerID", "year": "yearID", "team_id": "teamID",
        "position": "POS", "g": "G", "po": "PO", "a": "A", "e": "E", "dp": "DP",
    })


def standardize_teams(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={
        "year": "yearID", "team_id": "teamID", "franch_id": "franchID",
        "lg_id": "lgID", "w": "W", "l": "L", "win_rate": "winRate", "g": "team_games",
    })


# ---------------------------------------------------------------------------
# 시즌 단위 집계 (트레이드/stint 처리)
# ---------------------------------------------------------------------------

def aggregate_batting_to_season(batting: pd.DataFrame) -> pd.DataFrame:
    """player-year 단위로 stint(트레이드) 합산."""
    counting_cols = ["G", "AB", "R", "H", "2B", "3B", "HR", "RBI", "BB", "HBP", "SF"]
    agg = batting.groupby(["playerID", "yearID"], as_index=False)[counting_cols].sum()

    last_team = (
        batting.sort_values("stint")
        .groupby(["playerID", "yearID"], as_index=False)
        .last()[["playerID", "yearID", "teamID"]]
    )
    return agg.merge(last_team, on=["playerID", "yearID"])


def aggregate_pitching_to_season(pitching: pd.DataFrame) -> pd.DataFrame:
    """
    player-year 단위로 stint(트레이드) 합산.
    ER은 raw 컬럼이 없어 era*IP/9로 역산 후 합산 (ipouts=0인 행은 ER=0 처리).
    """
    p = pitching.copy()
    p["IP_stint"] = p["IPouts"] / 3
    p["ER"] = np.where(p["IPouts"] > 0, p["ERA"] * p["IP_stint"] / 9, 0.0)

    counting_cols = ["G", "IPouts", "ER", "H", "BB", "SO"]
    agg = p.groupby(["playerID", "yearID"], as_index=False)[counting_cols].sum()

    last_team = (
        p.sort_values("stint")
        .groupby(["playerID", "yearID"], as_index=False)
        .last()[["playerID", "yearID", "teamID"]]
    )
    return agg.merge(last_team, on=["playerID", "yearID"])


def aggregate_fielding_to_season(fielding: pd.DataFrame) -> pd.DataFrame:
    """player-year 단위로 stint(트레이드)·포지션 합산.

    한 시즌에 여러 포지션을 겸했어도(예: 유틸리티 선수) 수비 기여도는 하나의
    점수로 보므로 포지션 구분 없이 합산한다 - 포지션별 세분화는 v1 범위 밖.
    """
    counting_cols = ["G", "PO", "A", "E", "DP"]
    agg = fielding.groupby(["playerID", "yearID"], as_index=False)[counting_cols].sum()

    last_team = (
        fielding.sort_values("stint")
        .groupby(["playerID", "yearID"], as_index=False)
        .last()[["playerID", "yearID", "teamID"]]
    )
    return agg.merge(last_team, on=["playerID", "yearID"])


def derive_primary_position(fielding_std: pd.DataFrame) -> pd.DataFrame:
    """선수-시즌별 가장 많이 소화한 수비 포지션 (Fielding.csv, position별 G 최댓값).

    aggregate_fielding_to_season()과 달리 포지션을 합치지 않고 그대로 남긴다 —
    "이 선수가 이 시즌 주로 어디를 지켰는가"가 목적이라 포지션별 분리가 핵심.
    수비 기록이 아예 없는 선수-시즌(순수 지명타자 등)은 결과에 없음(NaN으로
    남게 되고, 호출부가 필요하면 "DH" 등으로 보정한다).
    """
    fs = fielding_std[fielding_std["yearID"] >= YEAR_FLOOR]
    by_pos = fs.groupby(["playerID", "yearID", "POS"], as_index=False)["G"].sum()
    idx = by_pos.groupby(["playerID", "yearID"])["G"].idxmax()
    return (
        by_pos.loc[idx, ["playerID", "yearID", "POS"]]
        .rename(columns={"POS": "primary_position"})
        .reset_index(drop=True)
    )


def _season_denominators(teams_std: pd.DataFrame, pitching_std: pd.DataFrame) -> pd.DataFrame:
    """
    연도별 출전비율 분모.
      - season_games: teams.csv의 team_games를 연도별 중앙값으로 (팀마다 158~163 정도 편차 흡수)
      - season_IP: pitching_stats를 (yearID, teamID)로 합산한 팀 전체 이닝의 연도별 중앙값
        (모든 아웃카운트는 정확히 한 명의 투수에게 귀속되므로, 선수별 IPouts를
        팀-연도 단위로 다 더하면 팀 전체 투구이닝과 정확히 일치함 - 근사 아님)
    """
    season_games = teams_std.groupby("yearID")["team_games"].median().rename("season_games")

    team_ip = pitching_std.groupby(["yearID", "teamID"])["IPouts"].sum().reset_index()
    season_ip = (team_ip.groupby("yearID")["IPouts"].median() / 3).rename("season_IP")

    return pd.concat([season_games, season_ip], axis=1).reset_index()


# ---------------------------------------------------------------------------
# 비율 지표 계산 (shrinkage 적용)
# ---------------------------------------------------------------------------

def compute_shrunk_ops(batting_season: pd.DataFrame) -> pd.DataFrame:
    """OBP, SLG 모두 정확히 계산 (2B/3B 복원됨)."""
    b = batting_season[batting_season["AB"] > 0].copy()

    obp_num = b["H"] + b["BB"] + b["HBP"]
    obp_den = b["AB"] + b["BB"] + b["HBP"] + b["SF"]
    total_bases = (b["H"] - b["2B"] - b["3B"] - b["HR"]) + 2 * b["2B"] + 3 * b["3B"] + 4 * b["HR"]

    b["obs_OPS"] = (obp_num / obp_den) + (total_bases / b["AB"])

    league_avg = b.groupby("yearID").apply(
        lambda g: (g["obs_OPS"] * g["AB"]).sum() / g["AB"].sum()
    )
    b["league_avg_OPS"] = b["yearID"].map(league_avg)

    k = _adaptive_k(b["AB"], b["yearID"], BATTING_SHRINKAGE_FRACTION)
    b["OPS"] = (b["obs_OPS"] * b["AB"] + b["league_avg_OPS"] * k) / (b["AB"] + k)

    return b


def compute_shrunk_pitching_rates(pitching_season: pd.DataFrame) -> pd.DataFrame:
    p = pitching_season[pitching_season["IPouts"] > 0].copy()
    p["IP"] = p["IPouts"] / 3

    league_era = p.groupby("yearID").apply(lambda g: (g["ER"].sum() * 9) / g["IP"].sum())
    league_whip = p.groupby("yearID").apply(lambda g: (g["BB"].sum() + g["H"].sum()) / g["IP"].sum())
    league_so9 = p.groupby("yearID").apply(lambda g: (g["SO"].sum() * 9) / g["IP"].sum())

    p["league_avg_ERA"] = p["yearID"].map(league_era)
    p["league_avg_WHIP"] = p["yearID"].map(league_whip)
    p["league_avg_SO9"] = p["yearID"].map(league_so9)

    k = _adaptive_k(p["IP"], p["yearID"], PITCHING_SHRINKAGE_FRACTION)
    p["ERA"] = (p["ER"] * 9 + p["league_avg_ERA"] * k) / (p["IP"] + k)
    p["WHIP"] = ((p["BB"] + p["H"]) + p["league_avg_WHIP"] * k) / (p["IP"] + k)
    p["SO9"] = (p["SO"] * 9 + p["league_avg_SO9"] * k) / (p["IP"] + k)

    return p


def compute_shrunk_fielding_rates(fielding_season: pd.DataFrame) -> pd.DataFrame:
    """FPCT(수비율)에 리그 평균 shrinkage 적용. RFG는 표본이 이미 G로 나눈
    비율이라 shrinkage 없이 그대로 쓴다(BB/H 처럼 합산 후 나누는 값이 아님)."""
    f = fielding_season[fielding_season["G"] > 0].copy()
    f["TC"] = f["PO"] + f["A"] + f["E"]  # 수비 기회(자살+보살+실책)

    # 수비 기회가 아예 없는 선수-시즌(지명타자 전업 등)은 수비율을 정의할 수
    # 없다 - 0으로 채우면 "수비를 못한다"는 거짓 신호가 되므로 그대로 제외한다.
    f = f[f["TC"] > 0].copy()

    f["obs_FPCT"] = (f["PO"] + f["A"]) / f["TC"]
    league_fpct = f.groupby("yearID").apply(lambda g: (g["PO"].sum() + g["A"].sum()) / g["TC"].sum())
    f["league_avg_FPCT"] = f["yearID"].map(league_fpct)

    k = _adaptive_k(f["TC"], f["yearID"], FIELDING_SHRINKAGE_FRACTION)
    f["FPCT"] = (f["obs_FPCT"] * f["TC"] + f["league_avg_FPCT"] * k) / (f["TC"] + k)
    f["RFG"] = (f["PO"] + f["A"]) / f["G"]  # Range Factor per Game

    return f


def _zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def _minmax_0_100(s: pd.Series) -> pd.Series:
    return (s - s.min()) / (s.max() - s.min()) * 100


# ---------------------------------------------------------------------------
# 전력 점수
# ---------------------------------------------------------------------------

def compute_batting_strength(batting: pd.DataFrame, teams: pd.DataFrame,
                              pitching_for_denom: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """
    타자 전력 점수 (0~100).
    공식: OPS(0.5) + HR(0.2) + RBI(0.15) + R(0.15) - 문서 원안 그대로.
    이후 출전비율(playing_time_ratio)과 부상 페널티(injury_penalty_factor)를
    곱연산으로 적용하고, 시즌 내 Min-Max로 다시 0~100 스케일을 맞춘다.
    """
    batting = batting[batting["yearID"] >= YEAR_FLOOR]
    season = aggregate_batting_to_season(batting)
    b = compute_shrunk_ops(season)

    for col in ["OPS", "HR", "RBI", "R"]:
        b[f"{col}_z"] = b.groupby("yearID")[col].transform(_zscore)

    b["batting_strength_raw"] = sum(b[f"{c}_z"] * w for c, w in BATTING_WEIGHTS.items())
    b["batting_strength_before_pt"] = b.groupby("yearID")["batting_strength_raw"].transform(_minmax_0_100)

    denom = _season_denominators(teams, pitching_for_denom)
    b = b.merge(denom[["yearID", "season_games"]], on="yearID", how="left")
    b["playing_time_ratio"] = (b["G"] / b["season_games"]).clip(upper=1.0)

    b = _apply_injury_penalty(b, injuries)
    b["batting_strength_scaled"] = (
        b["batting_strength_before_pt"] * b["playing_time_ratio"] * b["injury_penalty_factor"]
    )
    # 출전비율과 부상 페널티 두 개의 곱연산 할인이 겹치므로, 시즌 내에서 다시
    # Min-Max 0~100으로 재정규화해서 "최고 점수 = 100" 해석을 유지한다.
    b["batting_strength"] = b.groupby("yearID")["batting_strength_scaled"].transform(_minmax_0_100)

    return b[["playerID", "yearID", "teamID", "AB", "G", "OPS", "HR", "RBI", "R",
              "playing_time_ratio", "il_stint_count", "injury_penalty_factor",
              "batting_strength_before_pt", "batting_strength"]]


def compute_pitching_strength(pitching: pd.DataFrame, teams: pd.DataFrame,
                               injuries: pd.DataFrame) -> pd.DataFrame:
    """
    투수 전력 점수 (0~100).
    공식: ERA(0.35) + WHIP(0.30) + SO9(0.20) + IP(0.15) - 기존과 동일, 컬럼만 표준화됨.
    """
    pitching = pitching[pitching["yearID"] >= YEAR_FLOOR]
    season = aggregate_pitching_to_season(pitching)
    p = compute_shrunk_pitching_rates(season)

    for col in ["ERA", "WHIP", "SO9", "IP"]:
        p[f"{col}_z"] = p.groupby("yearID")[col].transform(_zscore)

    p["pitching_strength_raw"] = (
        (-p["ERA_z"]) * 0.35
        + (-p["WHIP_z"]) * 0.30
        + p["SO9_z"] * 0.20
        + p["IP_z"] * 0.15
    )
    p["pitching_strength_before_pt"] = p.groupby("yearID")["pitching_strength_raw"].transform(_minmax_0_100)

    denom = _season_denominators(teams, pitching)
    p = p.merge(denom[["yearID", "season_IP", "season_games"]], on="yearID", how="left")
    p["playing_time_ratio"] = (p["IP"] / p["season_IP"]).clip(upper=1.0)

    # g_ratio_games: 팀 총 이닝 대비 이 투수의 이닝 점유율(playing_time_ratio)은
    # 팀 투수진이 10명 이상으로 나눠 가지는 자원이라 "얼마나 자주 경기에
    # 나왔나"를 보여주지 못한다(마무리가 65경기를 던져도 이닝 점유율은 낮게
    # 나옴). contract.py의 g_ratio 정의(G/team_games)와 맞춰 "몇 경기에
    # 나왔는가" 기준의 별도 비율을 만든다 - 점수 스케일링(위 줄)에는
    # playing_time_ratio를 그대로 쓰고, g_ratio 산출에만 이 컬럼을 쓴다.
    p["g_ratio_games"] = (p["G"] / p["season_games"]).clip(upper=1.0)

    p = _apply_injury_penalty(p, injuries)
    p["pitching_strength_scaled"] = (
        p["pitching_strength_before_pt"] * p["playing_time_ratio"] * p["injury_penalty_factor"]
    )

    # 타자와 달리 투수는 한 명이 팀 전체 이닝(season_IP)의 최대 15~20% 정도밖에
    # 차지할 수 없어(팀에 투수가 10명 이상 있으므로), before_pt(최대 100) x ratio(최대 ~0.18)를
    # 하면 최종 점수가 구조적으로 20점 근처에서 눌린다. 부상 페널티까지 곱해지므로
    # 시즌 내에서 다시 한번 Min-Max 정규화를 적용해 0~100 스케일을 맞춘다.
    p["pitching_strength"] = p.groupby("yearID")["pitching_strength_scaled"].transform(_minmax_0_100)

    return p[["playerID", "yearID", "teamID", "IP", "G", "ERA", "WHIP", "SO9",
              "playing_time_ratio", "g_ratio_games", "il_stint_count", "injury_penalty_factor",
              "pitching_strength_before_pt", "pitching_strength"]]


def compute_fielding_strength(fielding: pd.DataFrame, teams: pd.DataFrame,
                               pitching_for_denom: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """
    수비 전력 점수 (0~100).
    공식: FPCT(0.4) + RFG(0.6) - FIELDING_WEIGHTS 주석 참고.
    batting_strength/pitching_strength와 동일한 절차(z-score -> min-max ->
    출전비율 -> 부상페널티 -> 재정규화)를 그대로 따른다.
    """
    fielding = fielding[fielding["yearID"] >= YEAR_FLOOR]
    season = aggregate_fielding_to_season(fielding)
    f = compute_shrunk_fielding_rates(season)

    for col in ["FPCT", "RFG"]:
        f[f"{col}_z"] = f.groupby("yearID")[col].transform(_zscore)

    f["fielding_strength_raw"] = sum(f[f"{c}_z"] * w for c, w in FIELDING_WEIGHTS.items())
    f["fielding_strength_before_pt"] = f.groupby("yearID")["fielding_strength_raw"].transform(_minmax_0_100)

    # season_games는 teams_std에서만 나온다 - pitching_for_denom은 _season_denominators
    # 시그니처를 맞추기 위한 것일 뿐 여기선 실제로 쓰이지 않는다(batting과 동일 패턴).
    denom = _season_denominators(teams, pitching_for_denom)
    f = f.merge(denom[["yearID", "season_games"]], on="yearID", how="left")
    f["playing_time_ratio"] = (f["G"] / f["season_games"]).clip(upper=1.0)

    f = _apply_injury_penalty(f, injuries)
    f["fielding_strength_scaled"] = (
        f["fielding_strength_before_pt"] * f["playing_time_ratio"] * f["injury_penalty_factor"]
    )
    f["fielding_strength"] = f.groupby("yearID")["fielding_strength_scaled"].transform(_minmax_0_100)

    return f[["playerID", "yearID", "teamID", "TC", "G", "FPCT", "RFG",
              "playing_time_ratio", "il_stint_count", "injury_penalty_factor",
              "fielding_strength_before_pt", "fielding_strength"]]


def build_features_v1(
    batting_result: pd.DataFrame,
    pitching_result: pd.DataFrame,
    fielding_result: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """타자/투수/수비 전력 점수를 player-year 단위로 병합.

    outer join - NaN은 "그 역할로 안 뛴 시즌"을 뜻한다(수비 기록이 아예 없는
    지명타자 전업 시즌 등도 fielding_strength가 자연히 NaN이 됨).
    fielding_result가 없으면(과거 호출부 호환용) 3자 병합을 건너뛰고 이전과
    동일하게 동작한다 - def_score는 호출부가 NaN으로 채워야 한다.
    """
    b = batting_result.rename(columns={"teamID": "bat_teamID"})
    p = pitching_result.rename(columns={"teamID": "pit_teamID"})

    merged = b.merge(p, on=["playerID", "yearID"], how="outer", suffixes=("", "_pit"))
    merged["is_batter"] = merged["batting_strength"].notna()
    merged["is_pitcher"] = merged["pitching_strength"].notna()

    if fielding_result is not None:
        d = fielding_result.rename(columns={"teamID": "def_teamID"})
        merged = merged.merge(d, on=["playerID", "yearID"], how="outer", suffixes=("", "_def"))
        merged["is_fielder"] = merged["fielding_strength"].notna()

    # 최종 features 경계에서는 contract.py / labels.py와 동일한 공통 키를 사용한다.
    # 내부 전력 계산은 Lahman 호환 이름을 유지한다.
    merged = merged.rename(columns={
        "playerID": "player_id",
        "yearID": "season",
        "bat_teamID": "bat_team_id",
        "pit_teamID": "pit_team_id",
        "def_teamID": "def_team_id",
    })

    return merged


if __name__ == "__main__":
    import os

    teams_raw = pd.read_csv("data/final/teams.csv")
    batting_raw = pd.read_csv("data/final/batting_stats.csv")
    pitching_raw = pd.read_csv("data/final/pitching_stats.csv")
    fielding_raw = pd.read_csv("data/final/fielding_stats.csv")
    injuries_raw = pd.read_csv("data/final/player_injury_stints.csv")

    teams = standardize_teams(teams_raw)
    batting = standardize_batting(batting_raw)
    pitching = standardize_pitching(pitching_raw)
    fielding = standardize_fielding(fielding_raw)
    injuries = standardize_injuries(injuries_raw)

    batting_result = compute_batting_strength(batting, teams, pitching, injuries)
    pitching_result = compute_pitching_strength(pitching, teams, injuries)
    fielding_result = compute_fielding_strength(fielding, teams, pitching, injuries)
    features_v1 = build_features_v1(batting_result, pitching_result, fielding_result)