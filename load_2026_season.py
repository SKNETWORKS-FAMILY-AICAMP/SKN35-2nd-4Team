"""2026시즌(진행중) 데이터를 기존 파이프라인에 병합한다.

입력 (전부 repo 루트에 이미 있음)
    mlb_2026_batting_stats.csv / mlb_2026_pitching_stats.csv / mlb_2026_team_standings.csv
        -- fetch_mlb_2026_stats.py로 받은 MLB Stats API 원본 (playerPool=all로 재수집,
           벤치 선수까지 포함해 타자 714명/투수 826명)
    2026_.csv
        -- build_id_crosswalk.py 출력(player_id_crosswalk.csv)과 동일한 스키마.
           Lahman player_id <-> MLBAM mlbam_id 매핑, Chadwick Register 기반.

MLBAM team id -> Lahman team_id는 build_id_crosswalk.py의 TEAM_CROSSWALK
정적 표(실측 대조 완료)를 그대로 재사용한다 - 새로 만들지 않는다.

주의 (fetch_mlb_2026_stats.py 자체 경고 그대로 승계)
    2026시즌은 아직 끝나지 않았다(게임 수 136/162, 약 84%). 그래서:
    - stint을 전부 1로 둔다 - MLB Stats API의 season 집계는 이미 시즌 전체를
      합산해서 주기 때문에(팀 이동 있어도 한 행) Lahman처럼 스틴트별로
      쪼개져 있지 않다. n_stint(트레이드 감지)는 2026 시즌에서는 정확하지
      않을 수 있다는 뜻 - 알려진 한계로 남겨둔다.
    - y_departed 등 라벨은 다음 시즌(2027) 데이터가 없어 전부 결측(censored)
      처리된다 - 지금 2025가 그런 것과 완전히 동일한 처리다. 학습에 쓰이는
      2009~2024 구간에는 전혀 영향 없다.
    - 표본이 작은 시즌은 D가 이번에 추가한 연도별 적응형 shrinkage
      (strength.py, FRACTION 기반 K)가 자동으로 더 세게 리그 평균 쪽으로
      당겨준다 - 2020 단축시즌과 동일한 메커니즘.

크로스워크에 없는 선수(2026 신인 등 Lahman에 아직 없는 선수)는 조용히
빠진다 - 임의 ID를 만들지 않는다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_FINAL = ROOT / "data" / "final"
YEAR = 2026

# build_id_crosswalk.py의 TEAM_CROSSWALK 그대로 (실측 대조 완료된 표) - 새로 안 만듦
TEAM_CROSSWALK = [
    ("ARI", 109, "Arizona Diamondbacks"), ("ATH", 133, "Athletics"), ("ATL", 144, "Atlanta Braves"),
    ("BAL", 110, "Baltimore Orioles"), ("BOS", 111, "Boston Red Sox"), ("CHA", 145, "Chicago White Sox"),
    ("CHN", 112, "Chicago Cubs"), ("CIN", 113, "Cincinnati Reds"), ("CLE", 114, "Cleveland Guardians"),
    ("COL", 115, "Colorado Rockies"), ("DET", 116, "Detroit Tigers"), ("HOU", 117, "Houston Astros"),
    ("KCA", 118, "Kansas City Royals"), ("LAA", 108, "Los Angeles Angels"), ("LAN", 119, "Los Angeles Dodgers"),
    ("MIA", 146, "Miami Marlins"), ("MIL", 158, "Milwaukee Brewers"), ("MIN", 142, "Minnesota Twins"),
    ("NYA", 147, "New York Yankees"), ("NYN", 121, "New York Mets"), ("PHI", 143, "Philadelphia Phillies"),
    ("PIT", 134, "Pittsburgh Pirates"), ("SDN", 135, "San Diego Padres"), ("SEA", 136, "Seattle Mariners"),
    ("SFN", 137, "San Francisco Giants"), ("SLN", 138, "St. Louis Cardinals"), ("TBA", 139, "Tampa Bay Rays"),
    ("TEX", 140, "Texas Rangers"), ("TOR", 141, "Toronto Blue Jays"), ("WAS", 120, "Washington Nationals"),
]
MLBAM_TO_LAHMAN_TEAM = {mlbam: lahman for lahman, mlbam, _ in TEAM_CROSSWALK}


def load_player_crosswalk() -> dict[int, str]:
    cw = pd.read_csv(ROOT / "2026_.csv")
    return dict(zip(cw["mlbam_id"], cw["player_id"]))


def _lg_id_lookup() -> dict[str, str]:
    """team_id -> lg_id(AL/NL), 정적이라 해마다 안 바뀜. 재실행해도 안정적이도록
    2026 이전 시즌 중 최신 연도를 기준으로 삼는다(이 스크립트 재실행으로
    teams.csv에 이미 2026이 들어있어도 max(year)가 2026으로 안 튀게)."""
    teams_prev = pd.read_csv(DATA_FINAL / "teams.csv")
    teams_prev = teams_prev[teams_prev.year < YEAR]
    latest = teams_prev[teams_prev.year == teams_prev.year.max()]
    return dict(zip(latest["team_id"], latest["lg_id"]))


def build_batting_rows(player_cw: dict[int, str], lg_map: dict[str, str]) -> pd.DataFrame:
    src = pd.read_csv(ROOT / "mlb_2026_batting_stats.csv")
    team_ids = src["team_id"].map(MLBAM_TO_LAHMAN_TEAM)
    out = pd.DataFrame({
        "stint": 1,
        "player_id": src["player_id"].map(player_cw),
        "year": YEAR,
        "team_id": team_ids,
        "lg_id": team_ids.map(lg_map),
        "g": src["gamesPlayed"], "ab": src["atBats"], "r": src["runs"], "h": src["hits"],
        "2b": src["doubles"], "3b": src["triples"], "hr": src["homeRuns"], "rbi": src["rbi"],
        "sb": src["stolenBases"], "cs": src["caughtStealing"], "bb": src["baseOnBalls"],
        "so": src["strikeOuts"], "ibb": src["intentionalWalks"], "hbp": src["hitByPitch"],
        "sh": src["sacBunts"], "sf": src["sacFlies"], "gidp": src["groundIntoDoublePlay"],
    })
    before = len(out)
    out = out.dropna(subset=["player_id", "team_id"])
    print(f"  타자: crosswalk 매칭 {len(out):,}/{before:,} (미매칭 {before - len(out):,}명은 제외 - 임의 ID 생성 안 함)")
    return out


def build_pitching_rows(player_cw: dict[int, str], lg_map: dict[str, str]) -> pd.DataFrame:
    src = pd.read_csv(ROOT / "mlb_2026_pitching_stats.csv")
    team_ids = src["team_id"].map(MLBAM_TO_LAHMAN_TEAM)
    out = pd.DataFrame({
        "stint": 1,
        "player_id": src["player_id"].map(player_cw),
        "year": YEAR,
        "team_id": team_ids,
        "lg_id": team_ids.map(lg_map),
        "w": src["wins"], "l": src["losses"], "g": src["gamesPitched"], "gs": src["gamesStarted"],
        "sv": src["saves"], "ipouts": src["outs"], "h": src["hits"], "er": src["earnedRuns"],
        "hr": src["homeRuns"], "bb": src["baseOnBalls"], "so": src["strikeOuts"], "era": src["era"],
        "hbp": src["hitByPitch"], "r": src["runs"],
    })
    before = len(out)
    out = out.dropna(subset=["player_id", "team_id"])
    print(f"  투수: crosswalk 매칭 {len(out):,}/{before:,} (미매칭 {before - len(out):,}명은 제외)")
    return out


# MLB Stats API는 외야를 LF/CF/RF로 세분해서 준다. Lahman Fielding.csv(2009~2025)는
# 전부 "OF"로 뭉뚱그려져 있다 - 그대로 두면 2026시즌 외야수는 "같은 포지션 후보
# 추천"에서 과거 외야수를 전혀 못 찾는다(RF != OF로 취급되어 매칭 0건, 실측 확인).
# primary_position 파생 로직(strength.py derive_primary_position)의 입력 단계에서
# 정규화해 데이터셋 전체의 포지션 granularity를 통일한다.
_OUTFIELD_POSITIONS = {"LF", "CF", "RF"}


def build_fielding_rows(player_cw: dict[int, str], lg_map: dict[str, str]) -> pd.DataFrame:
    src = pd.read_csv(ROOT / "mlb_2026_fielding_stats.csv")
    team_ids = src["team_id"].map(MLBAM_TO_LAHMAN_TEAM)
    position = src["position"].where(~src["position"].isin(_OUTFIELD_POSITIONS), "OF")
    out = pd.DataFrame({
        "stint": 1,
        "position": position,
        "player_id": src["player_id"].map(player_cw),
        "year": YEAR,
        "team_id": team_ids,
        "lg_id": team_ids.map(lg_map),
        "g": src["gamesPlayed"], "po": src["putOuts"], "a": src["assists"],
        "e": src["errors"], "dp": src["doublePlays"],
    })
    before = len(out)
    out = out.dropna(subset=["player_id", "team_id"])
    print(f"  수비: crosswalk 매칭 {len(out):,}/{before:,} (미매칭 {before - len(out):,}행은 제외)")

    # LF/CF/RF를 OF로 합치면서 같은 선수가 여러 외야 포지션을 겸했으면
    # (player_id, year, stint, team_id, position) 키가 중복된다 - 실측 210명.
    # 그냥 두면 뒤 호출부(append_unique)의 drop_duplicates(keep="last")가
    # 조용히 한쪽 스탯을 버린다 - 반드시 합산해야 한다.
    counting = ["g", "po", "a", "e", "dp"]
    before_group = len(out)
    grouped = out.groupby(
        ["stint", "position", "player_id", "year", "team_id", "lg_id"], as_index=False
    )[counting].sum()
    if before_group != len(grouped):
        print(f"  외야 포지션(LF/CF/RF) 합산으로 {before_group - len(grouped)}행 -> OF 단일 행 통합")
    return grouped


def build_team_rows() -> pd.DataFrame:
    """team_id별 정적 정보(franch_id/div_id/lg_id/park/name)는 2025시즌 행에서 그대로 가져온다
    (구단명·연고지·리그·디비전은 해마다 바뀌지 않음 - 2026에 새로 찾을 필요 없음)."""
    teams_prev = pd.read_csv(DATA_FINAL / "teams.csv")
    teams_prev = teams_prev[teams_prev.year < YEAR]  # 재실행 시 2026 자기 자신을 기준으로 삼지 않게
    static = (
        teams_prev[teams_prev.year == teams_prev.year.max()]
        .set_index("team_id")[["lg_id", "franch_id", "div_id", "name", "park"]]
    )

    src = pd.read_csv(ROOT / "mlb_2026_team_standings.csv")
    src = src.copy()
    src["team_id"] = src["team_id"].map(MLBAM_TO_LAHMAN_TEAM)
    src = src.dropna(subset=["team_id"])

    rows = []
    for _, row in src.iterrows():
        tid = row["team_id"]
        if tid not in static.index:
            continue
        s = static.loc[tid]
        rows.append({
            "year": YEAR, "team_id": tid, "lg_id": s["lg_id"], "franch_id": s["franch_id"],
            "div_id": s["div_id"], "rank": row["division_rank"], "g": row["games_played"],
            "w": row["wins"], "l": row["losses"], "win_rate": row["win_pct"],
            "name": s["name"], "park": s["park"],
        })
    out = pd.DataFrame(rows)
    print(f"  팀: {len(out)}/30")
    return out


def append_unique(path: Path, new_rows: pd.DataFrame, key_cols: list[str]) -> None:
    existing = pd.read_csv(path)
    existing = existing[~existing["year"].eq(YEAR)]  # 재실행 안전 - 이번 연도 행은 새로 교체
    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined = combined.drop_duplicates(subset=key_cols, keep="last")
    combined.to_csv(path, index=False)
    print(f"  {path.name}: {len(existing):,} -> {len(combined):,}행 저장")


def main() -> None:
    print(f"[1/5] ID 크로스워크 로드")
    player_cw = load_player_crosswalk()
    lg_map = _lg_id_lookup()
    print(f"  선수 크로스워크 {len(player_cw):,}명, 팀 크로스워크 {len(MLBAM_TO_LAHMAN_TEAM)}개")

    print(f"\n[2/5] {YEAR}시즌 타자 스탯 변환")
    batting_2026 = build_batting_rows(player_cw, lg_map)

    print(f"\n[3/5] {YEAR}시즌 투수 스탯 변환")
    pitching_2026 = build_pitching_rows(player_cw, lg_map)

    print(f"\n[4/5] {YEAR}시즌 수비 스탯 변환")
    fielding_2026 = build_fielding_rows(player_cw, lg_map)

    print(f"\n[5/5] {YEAR}시즌 팀 스탠딩 변환")
    teams_2026 = build_team_rows()

    print(f"\n기존 data/final/*.csv에 병합 (재실행해도 안전 - {YEAR}년 행만 교체)")
    append_unique(DATA_FINAL / "batting_stats.csv", batting_2026, ["player_id", "year", "stint"])
    append_unique(DATA_FINAL / "pitching_stats.csv", pitching_2026, ["player_id", "year", "stint"])
    append_unique(DATA_FINAL / "fielding_stats.csv", fielding_2026, ["player_id", "year", "stint", "position"])
    append_unique(DATA_FINAL / "teams.csv", teams_2026, ["team_id", "year"])

    print(f"\n완료. src/features/contract.py의 END_YEAR를 2026으로 올리고 build.py를 다시 돌릴 것.")


if __name__ == "__main__":
    main()
