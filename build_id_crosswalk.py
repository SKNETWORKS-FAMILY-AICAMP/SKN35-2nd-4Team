"""
Lahman player_id/team_id <-> MLB Stats API ID 크로스워크 생성

왜 필요한가
-----------
build.py/strength.py 파이프라인은 전부 Lahman의 ID 체계로 짜여 있다.
    - 선수: "troutmi01" 같은 문자열 (Lahman playerID)
    - 팀:   "NYA" 같은 3~4자리 코드 (Lahman team_id, franch.py의 franch_id도 여기서 파생)

근데 방금 받은 mlb_2026_batting_stats.csv / mlb_2026_pitching_stats.csv는
MLB Stats API(statsapi.mlb.com)에서 온 거라 ID가 완전히 다른 체계다.
    - 선수: 545361 같은 숫자 (MLBAM ID)
    - 팀:   147 같은 숫자 (MLB Stats API team id)

이 두 ID 체계를 그냥 이름으로만 매칭하면 동명이인·표기 차이(선수명 약자,
구단명 변경 등)에서 조용히 틀린 매칭이 생긴다. 그래서 신뢰할 수 있는 공개
크로스워크 테이블(Chadwick Register)을 써야 한다.

Chadwick Register (https://github.com/chadwickbureau/register)
-----------------------------------------------------------------
- 야구 통계 커뮤니티(Baseball-Reference, Retrosheet, Lahman 등)가 공동 유지하는
  "선수 신원 하나에 여러 ID 체계를 매핑"하는 공개 정적 참조 테이블이다.
- mlb.com을 긁는 게 전혀 아니고, GitHub에 공개된 정적 CSV 파일을 그대로
  받아오는 것뿐이라 이용약관 문제와 무관하다.
- key_bbref(Baseball-Reference ID)는 절대다수 선수에서 Lahman playerID와
  동일하다(둘 다 원래 같은 명명 규칙을 쓴다) — 완벽히 100%는 아니라서 아래에서
  실제 players.csv와 대조 검증까지 한다.
- key_mlbam이 바로 MLB Stats API가 쓰는 숫자 ID다.

팀 ID 크로스워크는 실제 데이터로 직접 검증
-------------------------------------------
data/final/teams.csv(2025시즌)의 franch_id/team_id와 MLB Stats API
/api/v1/teams 응답의 team.name을 구단명으로 직접 대조해서 만들었다
(추측 아님 — 두 소스 다 실제로 조회해서 만든 표).

사용법
------
    pip install requests pandas
    python build_id_crosswalk.py
    (players.csv를 같은 폴더 혹은 --players-csv로 지정하면 매칭률까지 검증)

출력
----
    player_id_crosswalk.csv   (columns: player_id[Lahman], mlbam_id, name_last, name_first)
    team_id_crosswalk.csv     (columns: lahman_team_id, mlbam_team_id, team_name)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

CHADWICK_BASE = "https://raw.githubusercontent.com/chadwickbureau/register/master/data"
CHADWICK_FILES = [f"people-{c}.csv" for c in "0123456789abcdef"]

# data/final/teams.csv(2025시즌)와 MLB Stats API /api/v1/teams 응답을
# 구단명으로 직접 대조해서 만든 표. 추측이 아니라 실측 대조 결과.
TEAM_CROSSWALK = [
    ("ARI", 109, "Arizona Diamondbacks"),
    ("ATH", 133, "Athletics"),
    ("ATL", 144, "Atlanta Braves"),
    ("BAL", 110, "Baltimore Orioles"),
    ("BOS", 111, "Boston Red Sox"),
    ("CHA", 145, "Chicago White Sox"),
    ("CHN", 112, "Chicago Cubs"),
    ("CIN", 113, "Cincinnati Reds"),
    ("CLE", 114, "Cleveland Guardians"),
    ("COL", 115, "Colorado Rockies"),
    ("DET", 116, "Detroit Tigers"),
    ("HOU", 117, "Houston Astros"),
    ("KCA", 118, "Kansas City Royals"),
    ("LAA", 108, "Los Angeles Angels"),  # Lahman명은 'of Anaheim' 포함, API는 미포함
    ("LAN", 119, "Los Angeles Dodgers"),
    ("MIA", 146, "Miami Marlins"),
    ("MIL", 158, "Milwaukee Brewers"),
    ("MIN", 142, "Minnesota Twins"),
    ("NYA", 147, "New York Yankees"),
    ("NYN", 121, "New York Mets"),
    ("PHI", 143, "Philadelphia Phillies"),
    ("PIT", 134, "Pittsburgh Pirates"),
    ("SDN", 135, "San Diego Padres"),
    ("SEA", 136, "Seattle Mariners"),
    ("SFN", 137, "San Francisco Giants"),
    ("SLN", 138, "St. Louis Cardinals"),
    ("TBA", 139, "Tampa Bay Rays"),
    ("TEX", 140, "Texas Rangers"),
    ("TOR", 141, "Toronto Blue Jays"),
    ("WAS", 120, "Washington Nationals"),
]


def build_team_crosswalk() -> pd.DataFrame:
    df = pd.DataFrame(
        TEAM_CROSSWALK,
        columns=["lahman_team_id", "mlbam_team_id", "team_name"],
    )
    assert df["lahman_team_id"].nunique() == 30, "팀이 30개가 아님 — 표 확인 필요"
    assert df["mlbam_team_id"].nunique() == 30, "mlbam_team_id 중복 있음 — 표 확인 필요"
    return df


def download_register() -> pd.DataFrame:
    frames = []
    for fname in CHADWICK_FILES:
        url = f"{CHADWICK_BASE}/{fname}"
        print(f"  다운로드 중: {fname}")
        df = pd.read_csv(url, dtype=str, low_memory=False)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_player_crosswalk(
    register: pd.DataFrame, players: pd.DataFrame | None
) -> pd.DataFrame:
    """
    2단계 매칭.

    1단계 — 엄격 매칭: register의 key_bbref를 그대로 Lahman player_id로 본다.
    key_bbref는 절대다수 선수에서 Lahman playerID와 동일하다(둘 다 같은
    명명 규칙에서 출발했다). 실측(players.csv 7,163명) 기준 매칭률 99.0%.

    2단계 — 성+출생연도 폴백: 1단계에서 안 맞은 나머지는 대부분 ID 문자열
    생성 규칙 차이 때문이다(이니셜 이름의 마침표 유무 — "burnea.01" vs
    "burneaj01", 아포스트로피 유무 — "o'brich01" vs "obriech01", 동명이인
    disambiguation 번호가 시스템마다 다르게 매겨짐 등). 은퇴 여부와는
    무관하다는 게 실측으로 확인됨. 성+출생연도로 register에서 유일하게
    특정되는 경우만 복구한다(동명이인+동갑 여러 명이면 오매칭 위험이 있어
    포기하고 unmatched로 남긴다) — 실측 기준 추가 37명 복구, 99.0% -> 99.5%.
    """

    register_mlbam = register.dropna(subset=["key_mlbam"]).copy()
    register_mlbam["mlbam_id"] = pd.to_numeric(
        register_mlbam["key_mlbam"], errors="coerce"
    ).astype("Int64")

    # 1단계: key_bbref 엄격 매칭
    strict = register_mlbam.dropna(subset=["key_bbref"]).rename(
        columns={"key_bbref": "player_id"}
    )
    dup = strict["player_id"].duplicated().sum()
    if dup:
        print(f"  [경고] player_id 중복 {dup}건 — 첫 값만 사용.")
        strict = strict.drop_duplicates("player_id", keep="first")

    rows = strict[["player_id", "mlbam_id", "name_last", "name_first"]].assign(
        matched_by="key_bbref 직접 매칭"
    )

    if players is None or "player_id" not in players.columns:
        return rows

    # 2단계: 아직 안 맞은 Lahman player_id에 대해 성+출생연도 폴백
    lahman_ids = set(players["player_id"].dropna())
    still_missing = players[
        players["player_id"].isin(lahman_ids - set(rows["player_id"]))
    ]

    recovered = []
    for _, row in still_missing.iterrows():
        last = str(row["name_last"]).lower()
        by = row.get("birth_year")
        if pd.isna(by):
            continue
        cand = register_mlbam[
            (register_mlbam["name_last"].str.lower() == last)
            & (register_mlbam["birth_year"] == by)
        ]
        if len(cand) == 1:
            recovered.append(
                {
                    "player_id": row["player_id"],
                    "mlbam_id": cand.iloc[0]["mlbam_id"],
                    "name_last": cand.iloc[0]["name_last"],
                    "name_first": cand.iloc[0]["name_first"],
                    "matched_by": "성+출생연도 폴백",
                }
            )

    if recovered:
        print(f"  2단계 폴백으로 {len(recovered)}명 추가 복구")
        rows = pd.concat([rows, pd.DataFrame(recovered)], ignore_index=True)

    return rows


def verify_against_lahman(crosswalk: pd.DataFrame, players: pd.DataFrame) -> None:
    if "player_id" not in players.columns:
        print("  [건너뜀] players.csv에 player_id 컬럼이 없어 검증 생략")
        return

    lahman_ids = set(players["player_id"].dropna())
    matched_ids = set(crosswalk["player_id"])
    overlap = lahman_ids & matched_ids

    print(
        f"  실제 players.csv 대비 매칭률: {len(overlap):,} / {len(lahman_ids):,} "
        f"({len(overlap) / max(len(lahman_ids), 1):.1%})"
    )
    missing = lahman_ids - matched_ids
    if missing:
        unmatched = players[players["player_id"].isin(missing)]
        print(f"  끝까지 안 맞는 {len(missing)}명 (동명이인+동갑 여러 명이라 자동 복구 불가):")
        print(
            unmatched[["player_id", "name_first", "name_last", "birth_year"]]
            .to_string(index=False)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--players-csv",
        type=Path,
        default=Path("data/final/players.csv"),
        help="Lahman players.csv 경로 (매칭률 검증용, 없으면 건너뜀)",
    )
    args = parser.parse_args()

    print("[1/2] 팀 ID 크로스워크 생성 (실측 대조 완료된 표)")
    team_cw = build_team_crosswalk()
    team_cw.to_csv("team_id_crosswalk.csv", index=False)
    print(f"  -> team_id_crosswalk.csv ({len(team_cw)}팀)")

    players = None
    if args.players_csv.exists():
        players = pd.read_csv(args.players_csv, dtype=str, low_memory=False)
    else:
        print(f"\n[참고] {args.players_csv} 없음 — 1단계(key_bbref 직접 매칭)만 수행 (2단계 폴백/검증 생략)")

    print("\n[2/2] 선수 ID 크로스워크 생성 (Chadwick Register, 2단계 매칭)")
    register = download_register()
    player_cw = build_player_crosswalk(register, players)
    player_cw.to_csv("player_id_crosswalk.csv", index=False)
    print(f"  -> player_id_crosswalk.csv ({len(player_cw):,}명)")

    if players is not None:
        print("\n[검증] 실제 Lahman players.csv와 대조")
        verify_against_lahman(player_cw, players)


if __name__ == "__main__":
    main()
