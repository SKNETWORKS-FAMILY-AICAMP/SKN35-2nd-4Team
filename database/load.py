"""
load.py
=======
data/final/*.csv 파일들을 Supabase PostgreSQL(schema.sql로 만든 테이블)에 적재한다.

사용법
------
    # 1) 연결 정보를 환경변수로 설정 (.env 파일도 지원, python-dotenv 필요)
    #    Supabase 대시보드 > Project Settings > Database > Connection string 에서 복사
    export DATABASE_URL="postgresql://postgres:[PASSWORD]@db.xxxxxxxx.supabase.co:5432/postgres"

    # 2) 전체 테이블 적재
    python load.py

    # 3) 특정 테이블만 적재
    python load.py --tables franchises players teams

    # 4) DB에 실제로 연결하지 않고, 무엇을 적재할지만 미리 확인 (검증용)
    python load.py --dry-run

    # 5) 재적재 전 기존 데이터 삭제(TRUNCATE) 후 적재 - 개발 중 재실행할 때 사용
    python load.py --truncate

동작 방식
--------
- data 폴더는 win_rate.py/game.py와 동일한 방식으로 자동 탐색한다
  (스크립트 위치에서 상위로 올라가며 data/final 또는 data 를 탐색).
- 테이블은 FK 의존관계 순서대로 적재한다:
    franchises -> players -> teams -> team_season
    -> batting_stats / pitching_stats / fielding_stats / appearances / allstar
    -> games -> player_season
- 이미 있는 행은 PK 기준 ON CONFLICT DO NOTHING 으로 건너뛴다 (재실행 안전).
- pandas NaN은 SQL NULL로 자동 변환한다.

schema.sql과 CSV 간 차이 때문에 이 스크립트가 자동으로 보정하는 것들
--------------------------------------------------------------
1) batting_stats: CSV는 소문자 r/2b/3b, 테이블은 대문자/숫자시작이라 quoted
   식별자 "R"/"2B"/"3B" 로 정의되어 있음 -> 적재 시 컬럼명을 "R","2B","3B"로 변경.
2) games.league, player_season.league: 테이블은 CHECK(league IN ('mlb','kbo'))
   인데 CSV의 league는 AL/NL(야구 리그)이 들어있음 -> 이 프로젝트는 MLB 데이터만
   다루므로 적재 시 league 컬럼을 전부 상수 'mlb'로 덮어씀. AL/NL 정보가 필요하면
   기존 lg_id 개념의 별도 컬럼을 스키마에 추가해야 하며, 이 스크립트만으로는
   복원할 수 없다는 점에 유의.
3) player_season.role: 테이블은 CHECK(role IN ('B','P','TWO'))인데 CSV는
   P/IF/OF/C/DH/PH 6종 세부 포지션이 들어있음 -> P는 그대로 'P', 나머지
   (IF/OF/C/DH/PH)는 전부 'B'(타자)로 매핑. 두 가지 이상 겸업한 '이도류(TWO)'
   선수를 식별할 데이터가 없어 'TWO'는 사용하지 않음(단순화된 근사치).
4) player_season.era_z: 테이블 컬럼명은 era_z인데 CSV는 cra_z로 되어 있어
   이름만 맞춰서 그대로 옮김 (두 값 모두 현재 전부 NULL이라 값 손실 없음).
5) teams.win_rate: schema 주석에 "load.py에서 계산해 적재"라고 되어 있어,
   CSV 값을 신뢰하지 않고 w/(w+l)로 다시 계산해서 넣음.

주의: 위 2), 3)번은 데이터 의미가 바뀌는 보정이다. 실제 팀 프로젝트에서
AL/NL 구분이나 세부 포지션이 필요하면, 이 변환을 적용하기 전에 스키마를
수정하는 편이 낫다. 필요하면 알려주면 이 스크립트도 같이 고칠 수 있다.
"""

import os
import sys

# database/ 에서 직접 실행해도 src.* 를 찾을 수 있게 리포 루트를 넣는다
# (app/pages/*.py, src/models/game.py 와 동일한 가드).
_ROOT_FOR_IMPORT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT_FOR_IMPORT not in sys.path:
    sys.path.insert(0, _ROOT_FOR_IMPORT)
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv 미설치 시 환경변수를 직접 export했다고 가정하고 진행

import psycopg2
from psycopg2.extras import execute_values


# ----------------------------------------------------------------------
# 0. 데이터 폴더 자동 탐색 (win_rate.py / game.py와 동일한 방식)
# ----------------------------------------------------------------------
def find_data_dir(start_dir, filename="games.csv", max_up=8):
    current = os.path.abspath(start_dir)
    tried = []
    patterns = [("data",), ("data", "final")]
    for _ in range(max_up + 1):
        for pattern in patterns:
            candidate = os.path.join(current, *pattern)
            tried.append(candidate)
            if os.path.isfile(os.path.join(candidate, filename)):
                return candidate, tried
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return None, tried


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_env_override = os.environ.get("LOAD_DATA_DIR")
if _env_override:
    DATA_DIR = _env_override
    _SEARCH_TRIED = [f"(환경변수 LOAD_DATA_DIR 사용) {DATA_DIR}"]
else:
    _found, _SEARCH_TRIED = find_data_dir(BASE_DIR)
    DATA_DIR = _found if _found else os.path.join(BASE_DIR, "data", "final")


# ----------------------------------------------------------------------
# 1. 테이블별 변환 함수 (CSV -> schema.sql 컬럼에 맞춘 DataFrame)
# ----------------------------------------------------------------------
def _to_records(df, columns):
    """DataFrame -> (컬럼순서에 맞춘) 튜플 리스트, NaN은 None으로 변환."""
    df = df[columns].copy()
    df = df.astype(object).where(pd.notnull(df), None)
    return list(df.itertuples(index=False, name=None))


def transform_franchises(df):
    return df[["franch_id", "franch_name"]]


def transform_players(df):
    return df[["player_id", "birth_year", "name_first", "name_last", "bats", "throws", "debut", "final_game"]]


def transform_teams(df):
    df = df.copy()
    # schema 주석: win_rate는 load.py에서 계산해 적재
    df["win_rate"] = df["w"] / (df["w"] + df["l"])
    return df[["year", "team_id", "lg_id", "franch_id", "div_id", "rank", "g", "w", "l", "win_rate", "name", "park"]]


def transform_team_season(df):
    return df[["year", "team_id", "bat_strength", "pit_strength", "def_strength", "win_rate", "pred_rank", "risk_index"]]


def transform_batting_stats(df):
    df = df.rename(columns={"r": "R", "2b": "2B", "3b": "3B"})
    return df[["year", "stint", "player_id", "team_id", "lg_id", "g", "ab", "R", "h",
               "2B", "3B", "hr", "rbi", "sb", "cs", "bb", "so", "ibb", "hbp", "sh", "sf", "gidp"]]


def transform_pitching_stats(df):
    return df[["year", "stint", "player_id", "team_id", "lg_id", "w", "l", "g", "gs", "sv",
               "ipouts", "h", "er", "hr", "bb", "so", "era", "hbp", "r"]]


def transform_fielding_stats(df):
    return df[["year", "stint", "position", "player_id", "team_id", "lg_id", "g", "po", "a", "e", "dp"]]


def transform_appearances(df):
    return df[["year", "team_id", "lg_id", "player_id", "g_all", "g_batting", "g_defense",
               "g_p", "g_c", "g_1b", "g_2b", "g_3b", "g_ss", "g_lf", "g_cf", "g_rf", "g_of", "g_dh"]]


def transform_allstar(df):
    return df[["player_id", "year", "game_num", "team_id"]]


def transform_games(df):
    df = df.copy()
    # league CHECK(mlb/kbo) 대응: CSV의 AL/NL은 야구 하위리그이지 스포츠 종류가 아님.
    # 이 프로젝트는 MLB만 다루므로 상수로 덮어씀 (AL/NL 세부구분은 유실됨).
    df["league"] = "mlb"
    return df[["game_pk", "season", "game_date", "league", "home_team", "away_team",
               "home_strength", "away_strength", "home_sp_era", "away_sp_era",
               "home_rest", "away_rest", "home_last10", "away_last10", "y_home_win"]]


ROLE_MAP = {"P": "P", "IF": "B", "OF": "B", "C": "B", "DH": "B", "PH": "B"}


def transform_player_season(df):
    df = df.copy()
    df["league"] = "mlb"  # games와 동일한 이유로 상수 덮어씀 (AL/NL -> mlb)
    df["role"] = df["role"].map(ROLE_MAP).fillna("B")  # TWO(이도류)는 식별 불가 -> 미사용
    df = df.rename(columns={"cra_z": "era_z"})
    return df[["player_id", "season", "team_last", "franch_id", "league", "role", "age", "exp",
               "n_stint", "g_ratio", "g_ratio_prev", "g_chg", "off_score", "pit_score", "def_score",
               "overall_score", "ops_z", "ops_z_prev", "era_z", "whip_z", "team_wr", "allstar",
               "y_departed", "y_path", "y_fa_release", "y_returned", "y_next_score"]]


def transform_features_v1(df):
    """features_v1 — contract.SCHEMA 컬럼만 그 순서대로.

    build.py 가 validate() 를 통과시킨 산출물이라 값 변환은 하지 않는다.
    다른 테이블과 달리 원본이 parquet 이라 config 에 reader="parquet" 를 준다.
    """
    from src.features import contract

    cols = [c for c in contract.SCHEMA if c in df.columns]
    missing = [c for c in contract.SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"features_v1 에 계약 컬럼이 없습니다: {missing}")
    return df[cols]


def transform_player_injury_stints(df):
    """mlb_injury_pipeline.py 산출물.

    injury_days_estimated(추정으로 채운 결장일수)는 total_recovery_days(실측)와
    반드시 분리해서 적재한다 — 추정을 실측처럼 쓰면 안 되기 때문.
    파이프라인을 예전 버전으로 돌린 CSV 에는 새 컬럼이 없을 수 있어 방어한다.
    """
    df = df.copy()
    for col in ("injury_days_estimated", "injury_effective_days"):
        if col not in df.columns:
            df[col] = None
    return df[["player_id", "season", "il_stint_count", "first_il_date",
               "injury_note_sample", "total_recovery_days", "unresolved_stints",
               "had_injury", "injury_days_estimated", "injury_effective_days",
               "injury_risk_score"]]


# ----------------------------------------------------------------------
# 2. 테이블 적재 순서 & 설정 (FK 의존관계 순서 반드시 지켜야 함)
# ----------------------------------------------------------------------
TABLE_CONFIGS = [
    {"name": "franchises",     "csv": "franchises.csv",     "pk": ["franch_id"],
     "transform": transform_franchises},
    {"name": "players",        "csv": "players.csv",        "pk": ["player_id"],
     "transform": transform_players},
    {"name": "teams",          "csv": "teams.csv",          "pk": ["year", "team_id"],
     "transform": transform_teams},
    {"name": "team_season",    "csv": "team_season.csv",    "pk": ["year", "team_id"],
     "transform": transform_team_season},
    {"name": "batting_stats",  "csv": "batting_stats.csv",  "pk": ["year", "stint", "player_id"],
     "transform": transform_batting_stats},
    {"name": "pitching_stats", "csv": "pitching_stats.csv", "pk": ["year", "stint", "player_id"],
     "transform": transform_pitching_stats},
    {"name": "fielding_stats", "csv": "fielding_stats.csv", "pk": ["year", "stint", "position", "player_id"],
     "transform": transform_fielding_stats},
    {"name": "appearances",    "csv": "appearances.csv",    "pk": ["year", "team_id", "player_id"],
     "transform": transform_appearances},
    {"name": "allstar",        "csv": "allstar.csv",        "pk": ["year", "game_num", "player_id"],
     "transform": transform_allstar},
    {"name": "games",          "csv": "games.csv",          "pk": ["game_pk"],
     "transform": transform_games},
    {"name": "player_season",  "csv": "player_season.csv",  "pk": ["player_id", "season"],
     "transform": transform_player_season},
    # [2026-08-31 추가] 화면이 실제로 읽는 두 산출물.
    # features_v1 은 player_season 파생물이라 반드시 그 뒤에 적재한다.
    {"name": "features_v1",    "csv": "features_v1.parquet", "pk": ["player_id", "season"],
     "transform": transform_features_v1, "reader": "parquet"},
    {"name": "player_injury_stints", "csv": "player_injury_stints.csv",
     "pk": ["player_id", "season"], "transform": transform_player_injury_stints},
]


# ----------------------------------------------------------------------
# 3. DB 적재
# ----------------------------------------------------------------------
def get_connection():
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError(
            "환경변수 DATABASE_URL이 설정되어 있지 않습니다.\n"
            "  Supabase 대시보드 > Project Settings > Database > Connection string 에서\n"
            "  'URI' 형식의 문자열을 복사해 아래처럼 설정하세요.\n"
            '    export DATABASE_URL="postgresql://postgres:[PASSWORD]@db.xxxx.supabase.co:5432/postgres"\n'
            "  (PowerShell: $env:DATABASE_URL = \"...\")"
        )
    return psycopg2.connect(dsn)


def load_table(conn, config, dry_run=False, truncate=False):
    name = config["name"]
    csv_path = os.path.join(DATA_DIR, config["csv"])

    if not os.path.exists(csv_path):
        print(f"  [건너뜀] {name}: {csv_path} 파일이 없습니다.")
        return 0

    # 대부분 CSV 지만 features_v1 은 parquet 이다(계약 산출물의 원본 형식).
    df = (pd.read_parquet(csv_path) if config.get("reader") == "parquet"
          else pd.read_csv(csv_path))
    df = config["transform"](df)
    columns = list(df.columns)
    records = _to_records(df, columns)

    print(f"  {name}: {len(records)}행 준비됨 (컬럼: {', '.join(columns)})")

    if dry_run:
        if records:
            print(f"    샘플 1행: {records[0]}")
        return len(records)

    cur = conn.cursor()
    if truncate:
        cur.execute(f'TRUNCATE TABLE "{name}" CASCADE;')

    col_list = ", ".join(f'"{c}"' for c in columns)
    pk_list = ", ".join(f'"{c}"' for c in config["pk"])
    sql = f'INSERT INTO "{name}" ({col_list}) VALUES %s ON CONFLICT ({pk_list}) DO NOTHING'

    execute_values(cur, sql, records, page_size=1000)
    conn.commit()
    cur.close()
    return len(records)


# ----------------------------------------------------------------------
# 4. 메인
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="data/final CSV -> Supabase PostgreSQL 적재")
    parser.add_argument("--tables", nargs="*", default=None,
                         help="적재할 테이블 이름들 (지정 안 하면 전체). 예: --tables franchises players teams")
    parser.add_argument("--dry-run", action="store_true",
                         help="DB에 연결하지 않고 준비된 데이터만 확인")
    parser.add_argument("--truncate", action="store_true",
                         help="적재 전 해당 테이블 데이터를 TRUNCATE (재실행/초기화용)")
    args = parser.parse_args()

    print(f"데이터 폴더: {DATA_DIR}")
    if not os.path.isdir(DATA_DIR):
        print("\n[오류] 데이터 폴더를 찾지 못했습니다. 탐색한 경로:")
        for p in _SEARCH_TRIED:
            print(f"  - {p}")
        print("\n환경변수 LOAD_DATA_DIR로 직접 지정할 수 있습니다.")
        sys.exit(1)

    configs = TABLE_CONFIGS
    if args.tables:
        wanted = set(args.tables)
        configs = [c for c in TABLE_CONFIGS if c["name"] in wanted]
        missing = wanted - {c["name"] for c in configs}
        if missing:
            print(f"[경고] 알 수 없는 테이블명: {missing}")

    conn = None
    if not args.dry_run:
        conn = get_connection()
        print("Supabase PostgreSQL 연결 성공")

    print(f"\n적재 순서: {' -> '.join(c['name'] for c in configs)}")
    if args.dry_run:
        print("(--dry-run: 실제 DB에는 적재하지 않습니다)")
    if args.truncate:
        print("(--truncate: 적재 전 기존 데이터를 삭제합니다)")

    total = 0
    for config in configs:
        print(f"\n[{config['name']}] 적재 중...")
        try:
            n = load_table(conn, config, dry_run=args.dry_run, truncate=args.truncate)
            total += n
        except Exception as e:
            print(f"  [실패] {config['name']}: {e}")
            if conn:
                conn.rollback()
            raise

    if conn:
        conn.close()

    print(f"\n완료. 총 {total}행 처리됨.")


if __name__ == "__main__":
    main()