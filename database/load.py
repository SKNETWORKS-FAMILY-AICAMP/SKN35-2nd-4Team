"""로컬 데이터 -> Supabase 적재.

지난 논의(수파베이스 스코프)에서 정한 순서를 따른다: 화면에 실제로 표시되는
"가공 계층"(player_season/team_season/games)을 먼저 채운다. 원천 5개 테이블
(batting_stats/pitching_stats/fielding_stats/appearances/allstar)은 스키마만
만들어두고 비워둔다 — 지금 당장 화면 어디서도 안 읽어서 우선순위가 낮다.
필요해지면 이 파일에 로더만 추가하면 된다(테이블은 이미 있음).

FK 순서 때문에 참조 테이블(franchises/teams/players)을 먼저 채워야 한다 —
player_season/team_season/games이 이 세 테이블을 참조한다.

재실행해도 안전하다 — 테이블마다 TRUNCATE 후 새로 채운다(부분 실패로 중복
적재되는 것을 막기 위함). 원천 5개 테이블은 건드리지 않는다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from psycopg2.extras import execute_values

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.storage.supabase_client import connection  # noqa: E402

DATA_FINAL = ROOT / "data" / "final"
DATA_PROCESSED = ROOT / "data" / "processed"


def _load_table(conn, table: str, columns: list[str], df: pd.DataFrame) -> None:
    """DataFrame을 table에 통째로 다시 적재한다(TRUNCATE 후 INSERT)."""
    df = df[columns].copy().astype(object)
    # NaN -> None 이어야 psycopg2가 SQL NULL로 보낸다. astype(object)로 먼저
    # 바꿔야 float 컬럼의 NaN이 nan::float 그대로 남지 않고 확실히 None이 된다
    # (안 그러면 date 컬럼에 NaN이 섞였을 때 "type mismatch: date vs float"
    # 오류가 남 — 실측 확인됨).
    df = df.where(pd.notna(df), None)
    rows = [tuple(r) for r in df.itertuples(index=False, name=None)]

    with conn.cursor() as cur:
        cur.execute(f'TRUNCATE TABLE "{table}" CASCADE;')
        col_list = ", ".join(f'"{c}"' for c in columns)
        execute_values(
            cur, f'INSERT INTO "{table}" ({col_list}) VALUES %s', rows, page_size=2000
        )
    conn.commit()
    print(f"  {table}: {len(rows):,}행 적재")


def load_franchises(conn) -> None:
    df = pd.read_csv(DATA_FINAL / "franchises.csv")
    _load_table(conn, "franchises", ["franch_id", "franch_name"], df)


def load_teams(conn) -> None:
    df = pd.read_csv(DATA_FINAL / "teams.csv")
    cols = ["year", "team_id", "lg_id", "franch_id", "div_id", "rank", "g", "w", "l", "win_rate", "name", "park"]
    _load_table(conn, "teams", cols, df)


def load_players(conn) -> None:
    df = pd.read_csv(DATA_FINAL / "players.csv")
    cols = ["player_id", "birth_year", "name_first", "name_last", "bats", "throws", "debut", "final_game"]
    _load_table(conn, "players", cols, df)


def load_player_season(conn) -> None:
    """features_v1.parquet = contract.py 계약 = player_season 테이블과 컬럼이 거의 동일하다."""
    df = pd.read_parquet(DATA_FINAL / "features_v1.parquet")
    cols = [
        "player_id", "season", "team_last", "franch_id", "league", "role", "age", "exp", "n_stint",
        "g_ratio", "g_ratio_prev", "g_chg", "off_score", "pit_score", "def_score", "overall_score",
        "ops_z", "ops_z_prev", "era_z", "whip_z", "team_wr",
        "y_departed", "y_path", "y_fa_release", "y_returned", "y_next_score",
    ]
    # allstar는 contract.SCHEMA에서 뺐다(D 확정) — player_season 테이블엔 컬럼이
    # 남아있어(nullable) 그냥 NULL로 채운다. 스키마를 다시 뜯을 필요는 없다.
    df = df.copy()
    df["allstar"] = None
    cols = cols[:21] + ["allstar"] + cols[21:]
    _load_table(conn, "player_season", cols, df)


def load_team_season(conn) -> None:
    df = pd.read_csv(DATA_FINAL / "team_season.csv")
    cols = ["year", "team_id", "bat_strength", "pit_strength", "def_strength", "win_rate", "pred_rank", "risk_index"]
    _load_table(conn, "team_season", cols, df)


def load_games(conn) -> None:
    df = pd.read_csv(DATA_FINAL / "games.csv")
    # games.csv의 league 컬럼은 AL/NL(아메리칸/내셔널리그)인데, 스키마의 league는
    # "mlb 또는 kbo"(데이터 출처 리그, KBO 전이 대비용 — player_season과 같은 뜻)라
    # 값 종류가 완전히 다르다. 지금 KBO 데이터는 하나도 없으므로 전부 'mlb'로
    # 채운다 — AL/NL 구분이 필요해지면 별도 컬럼을 스키마에 추가할 것.
    df["league"] = "mlb"
    cols = [
        "game_pk", "season", "game_date", "league", "home_team", "away_team",
        "home_strength", "away_strength", "home_sp_era", "away_sp_era",
        "home_rest", "away_rest", "home_last10", "away_last10", "y_home_win",
    ]
    _load_table(conn, "games", cols, df)


def run() -> None:
    print("=" * 60)
    print("Supabase 적재 시작 (UI 표시 계층 우선)")
    print("=" * 60)
    with connection() as conn:
        print("\n[1/6] franchises (참조)")
        load_franchises(conn)
        print("[2/6] teams (참조)")
        load_teams(conn)
        print("[3/6] players (참조)")
        load_players(conn)
        print("[4/6] player_season (가공 — 선수 리포트 화면)")
        load_player_season(conn)
        print("[5/6] team_season (가공 — 구단 상황실/홈 화면)")
        load_team_season(conn)
        print("[6/6] games (가공 — 오늘 경기 화면)")
        load_games(conn)
    print("\n원천 5개 테이블(batting_stats 등)은 이번엔 비워뒀습니다 — 필요해지면")
    print("이 파일에 load_* 함수만 추가하면 됩니다(스키마는 이미 있음).")
    print("완료.")


if __name__ == "__main__":
    run()
