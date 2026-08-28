"""
strength.py (v3 - 2B/3B/R 컬럼 복원 반영)
B - 전력 점수 계산 모듈

=== 스키마 변경 이력 ===
v1: 원본 Lahman CSV (2B, 3B, R 모두 존재) 기준
v2: A의 1차 DB 적재본에서 2B/3B/R이 빠져서 SLG 근사(H+3*HR) + R가중치
    재분배(0.588/0.235/0.176)로 임시 대응
v3 (현재): A가 batting_stats.csv에 R, 2B, 3B 컬럼을 추가함
    -> 원래 문서 공식(OPS 0.5 + HR 0.2 + RBI 0.15 + R 0.15) 그대로 복원,
       SLG도 2루타/3루타 반영한 정확한 계산으로 복원.
    컬럼명 표기가 대문자로 옴에 유의 (g,ab,h,hr,rbi,bb,hbp,sf는 소문자,
    R,2B,3B만 대문자) - standardize_batting()에서 그대로 매핑.

=== pitching_stats.csv는 그대로 (ER 대신 era만 존재) ===
  -> ER = era * IP / 9 로 역산해서 stint 합산 (v2와 동일).
=== teams.csv도 그대로 (팀 전체 IPouts 없음) ===
  -> season_IP는 pitching_stats.csv를 팀-연도 단위로 합산해서 만듦
     (근사 아니고 정의상 정확히 일치).
""" 

import pandas as pd
import numpy as np

BATTING_SHRINKAGE_K = 50
PITCHING_SHRINKAGE_K = 20  # 이닝(IP) 단위

# 타자 전력 가중치 (2B/3B/R 데이터 복원됨 - 원래 문서 공식 그대로 사용)
BATTING_WEIGHTS = {"OPS": 0.5, "HR": 0.2, "RBI": 0.15, "R": 0.15}


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
                              pitching_for_denom: pd.DataFrame) -> pd.DataFrame:
    """
    타자 전력 점수 (0~100).
    공식: OPS(0.5) + HR(0.2) + RBI(0.15) + R(0.15) - 문서 원안 그대로.
    """
    season = aggregate_batting_to_season(batting)
    b = compute_shrunk_ops(season)

    for col in ["OPS", "HR", "RBI", "R"]:
        b[f"{col}_z"] = b.groupby("yearID")[col].transform(_zscore)

    b["batting_strength_raw"] = sum(b[f"{c}_z"] * w for c, w in BATTING_WEIGHTS.items())
    b["batting_strength_before_pt"] = b.groupby("yearID")["batting_strength_raw"].transform(_minmax_0_100)

    denom = _season_denominators(teams, pitching_for_denom)
    b = b.merge(denom[["yearID", "season_games"]], on="yearID", how="left")
    b["playing_time_ratio"] = (b["G"] / b["season_games"]).clip(upper=1.0)
    b["batting_strength"] = b["batting_strength_before_pt"] * b["playing_time_ratio"]

    return b[["playerID", "yearID", "teamID", "AB", "G", "OPS", "HR", "RBI", "R",
              "playing_time_ratio", "batting_strength_before_pt", "batting_strength"]]


def compute_pitching_strength(pitching: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """
    투수 전력 점수 (0~100).
    공식: ERA(0.35) + WHIP(0.30) + SO9(0.20) + IP(0.15) - 기존과 동일, 컬럼만 표준화됨.
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

    denom = _season_denominators(teams, pitching)
    p = p.merge(denom[["yearID", "season_IP"]], on="yearID", how="left")
    p["playing_time_ratio"] = (p["IP"] / p["season_IP"]).clip(upper=1.0)
    p["pitching_strength"] = p["pitching_strength_before_pt"] * p["playing_time_ratio"]

    return p[["playerID", "yearID", "teamID", "IP", "G", "ERA", "WHIP", "SO9",
              "playing_time_ratio", "pitching_strength_before_pt", "pitching_strength"]]


def build_features_v1(batting_result: pd.DataFrame, pitching_result: pd.DataFrame) -> pd.DataFrame:
    """타자/투수 전력 점수를 player-year 단위로 병합 (outer join, NaN=해당 역할로 안 뛴 시즌)."""
    b = batting_result.rename(columns={"teamID": "bat_teamID"})
    p = pitching_result.rename(columns={"teamID": "pit_teamID"})

    merged = b.merge(p, on=["playerID", "yearID"], how="outer", suffixes=("", "_pit"))
    merged["is_batter"] = merged["batting_strength"].notna()
    merged["is_pitcher"] = merged["pitching_strength"].notna()

    return merged


if __name__ == "__main__":
    import os

    teams_raw = pd.read_csv("data/final/lahman/teams.csv")
    batting_raw = pd.read_csv("data/final/lahman/batting_stats.csv")
    pitching_raw = pd.read_csv("data/final/lahman/pitching_stats.csv")

    teams = standardize_teams(teams_raw)
    batting = standardize_batting(batting_raw)
    pitching = standardize_pitching(pitching_raw)

    batting_result = compute_batting_strength(batting, teams, pitching)
    pitching_result = compute_pitching_strength(pitching, teams)
    features_v1 = build_features_v1(batting_result, pitching_result)

    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/features_v1.parquet"
    features_v1.to_parquet(out_path, index=False)

    print(f"저장 완료: {out_path}")
    print(f"행 수: {len(features_v1)}, 열 수: {features_v1.shape[1]}")
    print(f"playerID+yearID 중복: {features_v1.duplicated(subset=['playerID','yearID']).sum()}건")
    print("\n[참고용] 2024시즌 타자 전력 상위 5명:")
    print(
        features_v1[features_v1.yearID == 2024]
        .sort_values("batting_strength", ascending=False)
        .head(5)[["playerID", "yearID", "AB", "OPS", "batting_strength"]]
    )