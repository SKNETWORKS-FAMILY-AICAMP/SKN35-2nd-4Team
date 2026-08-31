"""Supabase 조회 — 서비스/화면 계층이 로컬 parquet 대신 쓸 수 있는 함수들.

지금 당장 app/*.py 는 여전히 로컬 features_v1.parquet 를 직접 읽는다 — 이
파일은 그 전환을 위한 준비물이다(교체는 각 페이지에서 pd.read_parquet(...)
호출을 이 함수들로 바꾸기만 하면 됨, 반환 스키마를 동일하게 맞춰뒀다).

전부 pandas.read_sql 로 DataFrame을 반환한다 — 호출부 코드(adapt_features_v1
등)를 그대로 재사용할 수 있게 하기 위함이다.
"""

from __future__ import annotations

import pandas as pd

from src.storage.supabase_client import connection


def fetch_player_season(season: int | None = None) -> pd.DataFrame:
    """player_season 전체 또는 특정 시즌. contract.py 스키마와 컬럼이 동일하다."""
    query = 'SELECT * FROM "player_season"'
    params: tuple = ()
    if season is not None:
        query += ' WHERE "season" = %s'
        params = (season,)
    with connection() as conn:
        return pd.read_sql(query, conn, params=params)


def fetch_team_players(team_id: str, season: int) -> pd.DataFrame:
    """특정 구단·시즌의 선수단. 선수 리포트/구단 상황실 화면이 쓰는 형태."""
    query = 'SELECT * FROM "player_season" WHERE "team_last" = %s AND "season" = %s'
    with connection() as conn:
        return pd.read_sql(query, conn, params=(team_id, season))


def fetch_team_season(season: int | None = None) -> pd.DataFrame:
    """team_season 전체 또는 특정 시즌 — 예상 순위 화면이 쓰는 형태."""
    query = 'SELECT * FROM "team_season"'
    params: tuple = ()
    if season is not None:
        query += ' WHERE "year" = %s'
        params = (season,)
    with connection() as conn:
        return pd.read_sql(query, conn, params=params)


def fetch_games(season: int | None = None, game_date: str | None = None) -> pd.DataFrame:
    """games 전체 또는 시즌/날짜로 필터 — 오늘 경기 화면이 쓰는 형태."""
    query = 'SELECT * FROM "games" WHERE 1=1'
    params: list = []
    if season is not None:
        query += ' AND "season" = %s'
        params.append(season)
    if game_date is not None:
        query += ' AND "game_date" = %s'
        params.append(game_date)
    with connection() as conn:
        return pd.read_sql(query, conn, params=tuple(params))


def fetch_player_name_lookup() -> dict[str, str]:
    """player_id -> "이름 성". 카드/드롭다운 표시용 (People.csv 대신 DB에서)."""
    query = 'SELECT "player_id", "name_first", "name_last" FROM "players"'
    with connection() as conn:
        df = pd.read_sql(query, conn)
    names = (df["name_first"].fillna("") + " " + df["name_last"].fillna("")).str.strip()
    return dict(zip(df["player_id"], names))


# ── [2026-08-31 추가] 화면이 직접 읽던 두 산출물 ──────────────────────
# app/ 은 지금까지 features_v1.parquet / player_injury_stints.csv 를 파일로
# 읽었다. 배포 후 DB에서 끌어오려면 이 두 함수가 필요하다.
# 반환 컬럼·dtype 을 파일을 읽었을 때와 같게 맞춰서, 호출부(adapt_features_v1,
# _merge_injury 등)를 그대로 재사용할 수 있게 한다.

def fetch_features_v1(season: int | None = None) -> pd.DataFrame:
    """features_v1 전체 또는 특정 시즌.

    contract.SCHEMA 와 컬럼이 1:1로 대응한다. 화면은 보통 전체를 읽는다 —
    전 시즌 이력이 있어야 overall_score_delta 같은 파생값을 만들 수 있기 때문
    (predict_reason_tags 가 다시즌을 요구한다).
    """
    query = 'SELECT * FROM "features_v1"'
    params: tuple = ()
    if season is not None:
        query += ' WHERE "season" = %s'
        params = (season,)
    with connection() as conn:
        df = pd.read_sql(query, conn, params=params or None)

    # contract.validate() 가 기대하는 dtype 으로 되돌린다. Postgres 왕복에서
    # int 컬럼에 NULL 이 하나라도 있으면 float 으로 올라오기 때문.
    from src.features import contract

    for col, dtype in contract.SCHEMA.items():
        if col not in df.columns:
            continue
        if dtype == "int64" and df[col].notna().all():
            df[col] = df[col].astype("int64")
    return df


def fetch_injury_stints() -> pd.DataFrame:
    """player_injury_stints 전체. reason.py 의 부상 피처 병합에 쓰인다."""
    with connection() as conn:
        return pd.read_sql('SELECT * FROM "player_injury_stints"', conn)
