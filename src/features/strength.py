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

BATTING_SHRINKAGE_K = 50
PITCHING_SHRINKAGE_K = 20  # 이닝(IP) 단위

# 타자 전력 가중치 (2B/3B/R 데이터 복원됨 - 원래 문서 공식 그대로 사용)
BATTING_WEIGHTS = {"OPS": 0.5, "HR": 0.2, "RBI": 0.15, "R": 0.15}

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
        "bb": "BB", "hbp": "HBP", "sf": "SF", "R": "R", "2B": "2B", "3B": "3B",
    })


def standardize_pitching(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={
        "player_id": "playerID", "year": "yearID", "team_id": "teamID",
        "g": "G", "gs": "GS", "ipouts": "IPouts", "era": "ERA",
        "h": "H", "bb": "BB", "so": "SO",
    })


def standardize_teams(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={
        "year": "yearID", "team_id": "teamID", "franch_id": "franchID",
        "lg_id": "lgID", "w": "W", "l": "L", "win_rate": "winRate",
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

    k = BATTING_SHRINKAGE_K
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

    k = PITCHING_SHRINKAGE_K
    p["ERA"] = (p["ER"] * 9 + p["league_avg_ERA"] * k) / (p["IP"] + k)
    p["WHIP"] = ((p["BB"] + p["H"]) + p["league_avg_WHIP"] * k) / (p["IP"] + k)
    p["SO9"] = (p["SO"] * 9 + p["league_avg_SO9"] * k) / (p["IP"] + k)

    return p


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
    p = p.merge(denom[["yearID", "season_IP"]], on="yearID", how="left")
    p["playing_time_ratio"] = (p["IP"] / p["season_IP"]).clip(upper=1.0)

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
              "playing_time_ratio", "il_stint_count", "injury_penalty_factor",
              "pitching_strength_before_pt", "pitching_strength"]]


def build_features_v1(batting_result: pd.DataFrame, pitching_result: pd.DataFrame) -> pd.DataFrame:
    """타자/투수 전력 점수를 player-year 단위로 병합 (outer join, NaN=해당 역할로 안 뛴 시즌)."""
    b = batting_result.rename(columns={"teamID": "bat_teamID"})
    p = pitching_result.rename(columns={"teamID": "pit_teamID"})

    merged = b.merge(p, on=["playerID", "yearID"], how="outer", suffixes=("", "_pit"))
    merged["is_batter"] = merged["batting_strength"].notna()
    merged["is_pitcher"] = merged["pitching_strength"].notna()

    # 최종 features 경계에서는 contract.py / labels.py와 동일한 공통 키를 사용한다.
    # 내부 전력 계산은 Lahman 호환 이름을 유지한다.
    merged = merged.rename(columns={
        "playerID": "player_id",
        "yearID": "season",
        "bat_teamID": "bat_team_id",
        "pit_teamID": "pit_team_id",
    })

    return merged


if __name__ == "__main__":
    import os

    teams_raw = pd.read_csv("data/final/teams.csv")
    batting_raw = pd.read_csv("data/final/batting_stats.csv")
    pitching_raw = pd.read_csv("data/final/pitching_stats.csv")
    injuries_raw = pd.read_csv("data/final/player_injury_stints.csv")

    teams = standardize_teams(teams_raw)
    batting = standardize_batting(batting_raw)
    pitching = standardize_pitching(pitching_raw)
    injuries = standardize_injuries(injuries_raw)

    batting_result = compute_batting_strength(batting, teams, pitching, injuries)
    pitching_result = compute_pitching_strength(pitching, teams, injuries)
    features_v1 = build_features_v1(batting_result, pitching_result)