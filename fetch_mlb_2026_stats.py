"""
2026시즌 MLB 타자/투수/팀 스탯 수집 스크립트

데이터 출처: MLB Stats API (https://statsapi.mlb.com) — mlb.com 웹페이지를 긁는 게
아니라 MLB이 공개한 JSON API 엔드포인트를 직접 호출한다. mlb.com/stats 페이지는
이용약관 1조(xi)에서 자동화 스크립트를 명시적으로 금지하지만, statsapi.mlb.com은
성격이 다른 공개 데이터 API이고 pybaseball, MLB-StatsAPI 같은 오픈소스 프로젝트에서
개인/연구 목적으로 널리 쓰인다. 다만 이것도 MLB 소유 서비스이므로, 상업적 재배포나
대량 크롤링은 하지 말고 이 프로젝트(학습용, 비상업)처럼 적당한 빈도로만 쓸 것.

⚠️ 중요 — "시즌 진행 중" 데이터라는 점 반드시 인지할 것
--------------------------------------------------------
오늘(스크립트 실행 시점) 기준 2026시즌은 아직 끝나지 않았다. 즉 이 스크립트로
받는 타자/투수 스탯은 "시즌 전체"가 아니라 "지금까지 치른 경기까지의 누적치"다.

- HR/RBI/타석수/이닝 같은 "누적(counting) 스탯"은 2025시즌 풀시즌 값과 그대로
  비교하면 안 된다 — 당연히 더 적게 나오고, 이걸 "왜곡"으로 오인할 수 있다.
- AVG/OBP/OPS/ERA/WHIP 같은 "비율(rate) 스탯"은 표본 크기가 작을 뿐 비교 자체는
  가능하지만, 시즌 초반일수록 변동성이 커서 표준적인 통계적 노이즈로 봐야 한다.
- 팀 win_pct(승률)도 마찬가지로 게임 수가 적을수록 변동성이 크다.

즉, 원인 진단 없이 이 데이터를 곧바로 features_v1.parquet에 섞으면 "2025 대비
2026이 이상하다"는 현상이 실제 트렌드 변화가 아니라 이 표본 크기 문제 때문에
생길 수 있다. 병합하기 전에 반드시 games_played/innings_pitched 기준으로
필터링하거나, 시즌 진행률을 피처로 같이 넣는 걸 권장한다.

사용법
------
    pip install requests pandas
    python fetch_mlb_2026_stats.py

출력 (현재 폴더에 저장):
    mlb_2026_batting_stats.csv
    mlb_2026_pitching_stats.csv
    mlb_2026_team_standings.csv
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://statsapi.mlb.com/api/v1"
SEASON = 2026
SPORT_ID = 1  # MLB (마이너리그 등은 다른 sportId)
OUT_DIR = Path(".")
HEADERS = {"User-Agent": "SKN-AI-2nd-Project/1.0 (educational, non-commercial)"}


def _get(url: str, params: dict) -> dict:
    resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_player_stats(group: str, season: int = SEASON) -> pd.DataFrame:
    """group: 'hitting' 또는 'pitching'. 개인 통산이 아니라 해당 시즌 누적 스탯만 가져온다."""
    params = {
        "stats": "season",
        "group": group,
        "season": season,
        "sportId": SPORT_ID,
        "gameType": "R",  # 정규시즌만 (스프링캠프/포스트시즌 제외)
        "limit": 2000,     # 한 시즌 로스터 전체를 한 번에 받기에 충분히 큰 값
        # playerPool 기본값은 "규정타석/이닝을 채운 리더"만 준다(실측 확인 -
        # 이 파라미터 없이는 타자 139명/투수 53명뿐, all로 주면 714명/532명).
        # 벤치·백업 선수까지 다 있어야 "같은 포지션 후보" 매칭 등이 의미있다.
        "playerPool": "all",
    }
    data = _get(f"{BASE_URL}/stats", params)

    rows = []
    for block in data.get("stats", []):
        for split in block.get("splits", []):
            player = split.get("player", {})
            team = split.get("team", {})
            stat = split.get("stat", {})
            row = {
                "player_id": player.get("id"),
                "player_name": player.get("fullName"),
                "team_id": team.get("id"),
                "team_name": team.get("name"),
                "season": season,
                **stat,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) >= 2000:
        print(
            f"[경고] {group} 응답이 limit(2000)에 딱 맞게 찼음 — "
            "페이지네이션이 필요할 수 있으니 결과 행 수를 확인할 것"
        )
    return df


def fetch_fielding_stats(season: int = SEASON) -> pd.DataFrame:
    """포지션별 수비 스탯. group=hitting/pitching과 달리 선수당 여러 행이 나올
    수 있다(포지션마다 한 행 - Lahman Fielding.csv와 동일한 구조). 실측
    2,579행이라 limit=2000이면 잘림 - 5000으로 넉넉히 잡는다."""
    params = {
        "stats": "season",
        "group": "fielding",
        "season": season,
        "sportId": SPORT_ID,
        "gameType": "R",
        "limit": 5000,
        "playerPool": "all",
    }
    data = _get(f"{BASE_URL}/stats", params)

    rows = []
    for block in data.get("stats", []):
        for split in block.get("splits", []):
            player = split.get("player", {})
            team = split.get("team", {})
            stat = split.get("stat", {})
            position = split.get("position", {})
            rows.append({
                "player_id": player.get("id"),
                "player_name": player.get("fullName"),
                "team_id": team.get("id"),
                "team_name": team.get("name"),
                "season": season,
                "position": position.get("abbreviation"),
                "gamesPlayed": stat.get("gamesPlayed"),
                "putOuts": stat.get("putOuts"),
                "assists": stat.get("assists"),
                "errors": stat.get("errors"),
                "doublePlays": stat.get("doublePlays"),
            })

    df = pd.DataFrame(rows)
    if len(df) >= 5000:
        print(f"[경고] fielding 응답이 limit(5000)에 딱 맞게 찼음 — 페이지네이션 필요 가능성")
    return df


def fetch_team_standings(season: int = SEASON) -> pd.DataFrame:
    """AL(103)/NL(104) 전체 팀 스탠딩. division/league 랭킹까지 포함."""
    params = {"leagueId": "103,104", "season": season}
    data = _get(f"{BASE_URL}/standings", params)

    rows = []
    for record in data.get("records", []):
        division = record.get("division", {})
        league = record.get("league", {})
        for team_record in record.get("teamRecords", []):
            team = team_record.get("team", {})
            rows.append(
                {
                    "team_id": team.get("id"),
                    "team_name": team.get("name"),
                    "season": season,
                    "league_id": league.get("id"),
                    "division_id": division.get("id"),
                    "wins": team_record.get("wins"),
                    "losses": team_record.get("losses"),
                    "games_played": team_record.get("gamesPlayed"),
                    "win_pct": team_record.get("winningPercentage"),
                    "division_rank": team_record.get("divisionRank"),
                    "league_rank": team_record.get("leagueRank"),
                    "games_back": team_record.get("gamesBack"),
                    "run_differential": team_record.get("runDifferential"),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    print(f"[1/4] {SEASON}시즌 타자 스탯 수집 중...")
    batting = fetch_player_stats("hitting")
    batting.to_csv(OUT_DIR / f"mlb_{SEASON}_batting_stats.csv", index=False)
    print(f"  -> {len(batting):,}명, mlb_{SEASON}_batting_stats.csv 저장")

    time.sleep(1)  # API에 과도한 연속 요청을 피한다

    print(f"[2/4] {SEASON}시즌 투수 스탯 수집 중...")
    pitching = fetch_player_stats("pitching")
    pitching.to_csv(OUT_DIR / f"mlb_{SEASON}_pitching_stats.csv", index=False)
    print(f"  -> {len(pitching):,}명, mlb_{SEASON}_pitching_stats.csv 저장")

    time.sleep(1)

    print(f"[3/4] {SEASON}시즌 수비 스탯 수집 중...")
    fielding = fetch_fielding_stats()
    fielding.to_csv(OUT_DIR / f"mlb_{SEASON}_fielding_stats.csv", index=False)
    print(f"  -> {len(fielding):,}행, mlb_{SEASON}_fielding_stats.csv 저장")

    time.sleep(1)

    print(f"[4/4] {SEASON}시즌 팀 스탠딩 수집 중...")
    standings = fetch_team_standings()
    standings.to_csv(OUT_DIR / f"mlb_{SEASON}_team_standings.csv", index=False)
    print(f"  -> {len(standings):,}팀, mlb_{SEASON}_team_standings.csv 저장")

    print("\n완료. games_played / inningsPitched 컬럼으로 표본 크기부터 확인하고 병합할 것.")


if __name__ == "__main__":
    main()
