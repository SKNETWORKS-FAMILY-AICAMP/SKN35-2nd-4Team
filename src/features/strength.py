"""
=======================================================================================
strength.py

전력 점수 Module

타자 전력 (OPS 0.5 + HR 0.2 + RBI 0.15 + R 0.15)
투수 전력 (ERA 0.35 + WHIP 0.30 + SO9 0.20 + IP 0.15)
시즌 내 z-score 정규화 -> Min-Max 0~100

설계 노트:
  - OPS/ERA/WHIP 같은 비율 지표는 표본(AB, IP)이 적을수록 극단값이 나오기 쉬움
    (예: 1타수 1안타 = OPS 3.000, 1이닝 5실점 = ERA 45.00).
  - 이를 하드 컷오프(AB>=50 등)로 잘라내면 저활약 선수-시즌 전체가
    데이터셋에서 사라져, 이탈(퇴출) 예측에 가장 중요한 케이스를 잃게 됨.
  - 대신 베이지안 축소(shrinkage)를 적용: 표본이 적을수록 리그 평균 쪽으로
    끌어당기고, 표본이 많을수록 관측값을 그대로 신뢰함.
        보정값 = (관측값 * 표본크기 + 리그평균 * k) / (표본크기 + k)
    k는 "가상의 리그평균 표본 크기"로, 타자는 k=50(AB), 투수는 k=20(IP) 사용.
    -> 규정 타석/이닝급 선수는 거의 영향 없음 (검증: 평균 차이 0.01~0.07),
       저활약 선수는 리그 평균 근처로 안정화됨 (극단치 완전 제거).
=======================================================================================
"""

import pandas as pd
import numpy as np

# 축소(shrinkage) 강도. 타자는 AB 50개, 투수는 IP 20이닝만큼의
# 리그평균 가상표본을 더해준다는 의미.
BATTING_SHRINKAGE_K = 50
PITCHING_SHRINKAGE_K = 20  # 이닝(IP) 단위


def compute_shrunk_ops(batting: pd.DataFrame) -> pd.DataFrame:
    """선수-시즌 단위 OPS를 계산하고, 표본 크기 기반으로 리그 평균 쪽으로 축소한다."""
    b = batting[batting["AB"] > 0].copy()

    obp_num = b["H"] + b["BB"] + b["HBP"]
    obp_den = b["AB"] + b["BB"] + b["HBP"] + b["SF"]
    slg_num = (b["H"] - b["2B"] - b["3B"] - b["HR"]) + 2 * b["2B"] + 3 * b["3B"] + 4 * b["HR"]

    b["obs_OPS"] = (obp_num / obp_den) + (slg_num / b["AB"])

    # 연도별 리그 평균 OPS (AB 가중평균)
    league_avg = b.groupby("yearID").apply(
        lambda g: (g["obs_OPS"] * g["AB"]).sum() / g["AB"].sum()
    )
    b["league_avg_OPS"] = b["yearID"].map(league_avg)

    k = BATTING_SHRINKAGE_K
    b["OPS"] = (b["obs_OPS"] * b["AB"] + b["league_avg_OPS"] * k) / (b["AB"] + k)

    return b


def _zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def _minmax_0_100(s: pd.Series) -> pd.Series:
    return (s - s.min()) / (s.max() - s.min()) * 100


def aggregate_batting_to_season(batting: pd.DataFrame) -> pd.DataFrame:
    """
    트레이드(stint) 처리: 한 시즌에 여러 팀을 거친 선수를 player-year 단위 한 행으로
    합산한다. 원인 라벨(잔류/이탈, franchID 비교)이 시즌 단위 판단이라
    stint 단위로 남겨두면 조인이 애매해지므로, 전력 계산 전에 먼저 합친다.
    representative team은 시즌 마지막 소속팀(가장 큰 stint)을 사용.
    """
    counting_cols = ["G", "AB", "R", "H", "2B", "3B", "HR", "RBI", "BB", "SO", "HBP", "SF"]
    agg = batting.groupby(["playerID", "yearID"], as_index=False)[counting_cols].sum()

    last_team = (
        batting.sort_values("stint")
        .groupby(["playerID", "yearID"], as_index=False)
        .last()[["playerID", "yearID", "teamID"]]
    )
    return agg.merge(last_team, on=["playerID", "yearID"])


def aggregate_pitching_to_season(pitching: pd.DataFrame) -> pd.DataFrame:
    """투수 버전 시즌 집계 (aggregate_batting_to_season과 동일한 이유)."""
    counting_cols = ["G", "IPouts", "ER", "H", "BB", "SO"]
    agg = pitching.groupby(["playerID", "yearID"], as_index=False)[counting_cols].sum()

    last_team = (
        pitching.sort_values("stint")
        .groupby(["playerID", "yearID"], as_index=False)
        .last()[["playerID", "yearID", "teamID"]]
    )
    return agg.merge(last_team, on=["playerID", "yearID"])


def _season_denominators(teams: pd.DataFrame) -> pd.DataFrame:
    """
    연도별 '시즌 전체 게임수/이닝수' (팀 간 158~163경기 정도의 편차는 있지만
    2020년 60경기처럼 연도 단위로는 명확히 구분됨).
    트레이드로 여러 팀을 거친 선수는 특정 팀 하나를 기준으로 삼기 애매하므로
    팀별 값 대신 연도별 중앙값을 출전 비율의 분모로 사용한다.
    """
    season_games = teams.groupby("yearID")["G"].median().rename("season_games")
    season_ip = (teams.groupby("yearID")["IPouts"].median() / 3).rename("season_IP")
    return pd.concat([season_games, season_ip], axis=1).reset_index()


def compute_batting_strength(batting: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """
    타자 전력 점수 (0~100) 계산. player-year 단위로 반환 (트레이드 stint는 사전 합산).
    공식: OPS 0.5 + HR 0.2 + RBI 0.15 + R 0.15 (각 항목은 시즌 내 z-score)
    이후 출전 비율(G/season_games)을 곱해 최종 점수를 산출한다.
    """
    season = aggregate_batting_to_season(batting)
    b = compute_shrunk_ops(season)

    components = ["OPS", "HR", "RBI", "R"]
    weights = {"OPS": 0.5, "HR": 0.2, "RBI": 0.15, "R": 0.15}

    for col in components:
        b[f"{col}_z"] = b.groupby("yearID")[col].transform(_zscore)

    b["batting_strength_raw"] = sum(b[f"{c}_z"] * w for c, w in weights.items())
    b["batting_strength_before_pt"] = b.groupby("yearID")["batting_strength_raw"].transform(_minmax_0_100)

    denom = _season_denominators(teams)
    b = b.merge(denom[["yearID", "season_games"]], on="yearID", how="left")
    b["playing_time_ratio"] = (b["G"] / b["season_games"]).clip(upper=1.0)
    b["batting_strength"] = b["batting_strength_before_pt"] * b["playing_time_ratio"]

    return b[["playerID", "yearID", "teamID", "AB", "G", "OPS", "HR", "RBI", "R",
              "playing_time_ratio", "batting_strength_before_pt", "batting_strength"]]


def compute_shrunk_pitching_rates(pitching: pd.DataFrame) -> pd.DataFrame:
    """ERA, WHIP, SO9을 계산하고 표본 크기(IP) 기반으로 리그 평균 쪽으로 축소한다."""
    p = pitching[pitching["IPouts"] > 0].copy()
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


def compute_pitching_strength(pitching: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """
    투수 전력 점수 (0~100) 계산. player-year 단위로 반환 (트레이드 stint는 사전 합산).
    공식: ERA 0.35 + WHIP 0.30 + SO9 0.20 + IP 0.15 (각 항목은 시즌 내 z-score)
    ERA/WHIP은 낮을수록 좋은 지표이므로 z-score 부호를 반전해서 합산한다.
    이후 출전 비율(IP/season_IP)을 곱해 최종 점수를 산출한다.
    """
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

    denom = _season_denominators(teams)
    p = p.merge(denom[["yearID", "season_IP"]], on="yearID", how="left")
    p["playing_time_ratio"] = (p["IP"] / p["season_IP"]).clip(upper=1.0)
    p["pitching_strength"] = p["pitching_strength_before_pt"] * p["playing_time_ratio"]

    return p[["playerID", "yearID", "teamID", "IP", "G", "ERA", "WHIP", "SO9",
              "playing_time_ratio", "pitching_strength_before_pt", "pitching_strength"]]


def build_features_v1(batting_result: pd.DataFrame, pitching_result: pd.DataFrame) -> pd.DataFrame:
    """
    타자/투수 전력 점수를 player-year(playerID+yearID) 단위로 병합한다.

    - outer join: 둘 중 한쪽에만 있으면 다른 쪽은 NaN (오타니처럼 둘 다 있는 선수는
      두 점수 모두 채워짐). LightGBM 등 트리 기반 모델은 NaN을 자연스럽게 분기
      처리하므로 0으로 채우지 않는다 (0으로 채우면 "최악의 성적"으로 잘못 해석될 위험).
    - is_batter / is_pitcher 플래그를 별도로 둬서, 결측이 "그 해 그 역할로 뛰지
      않았다"는 의미임을 모델과 사람 모두 명확히 구분할 수 있게 한다.
    - teamID가 다를 경우(매우 드묾, 이중포지션 선수가 두 역할에서 다른 팀으로
      트레이드된 경우) bat_teamID / pit_teamID로 분리 보관.
    """
    b = batting_result.rename(columns={"teamID": "bat_teamID"})
    p = pitching_result.rename(columns={"teamID": "pit_teamID"})

    merged = b.merge(p, on=["playerID", "yearID"], how="outer", suffixes=("", "_pit"))
    merged["is_batter"] = merged["batting_strength"].notna()
    merged["is_pitcher"] = merged["pitching_strength"].notna()

    return merged


if __name__ == "__main__":
    teams = pd.read_csv("data/processed/Teams.csv")

    batting = pd.read_csv("data/processed/Batting.csv")
    batting_result = compute_batting_strength(batting, teams)

    pitching = pd.read_csv("data/processed/Pitching.csv")
    pitching_result = compute_pitching_strength(pitching, teams)

    features_v1 = build_features_v1(batting_result, pitching_result)

    

    print(features_v1.shape)

    print(
        features_v1[features_v1.yearID == 2024]
        .sort_values("batting_strength", ascending=False)
        .head(5)[["playerID", "yearID", "AB", "OPS", "batting_strength"]]
    )