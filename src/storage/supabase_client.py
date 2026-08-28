"""Supabase(Postgres) 연결 헬퍼.

DATABASE_URL은 .env에서 읽는다(사용자가 이미 발급받아 넣어둔 값). API 키를
코드에 하드코딩하지 않는다 — .env는 .gitignore에 포함되어야 한다.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import psycopg2
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=ROOT / ".env")


def get_connection():
    """새 연결을 연다. 호출부가 직접 닫아야 한다 — 짧은 스크립트/배치 적재용.

    Streamlit 앱처럼 매 rerun마다 새로 열면 안 되는 곳에서는 connection()
    컨텍스트 매니저를 쓸 것.
    """
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL이 .env에 없습니다. Supabase 프로젝트 설정 > Database에서 "
            "Connection string(URI)을 복사해 .env에 DATABASE_URL=... 로 추가하세요."
        )
    return psycopg2.connect(url, connect_timeout=15)


@contextmanager
def connection() -> Iterator["psycopg2.extensions.connection"]:
    """with 블록이 끝나면 자동으로 닫는다."""
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


def healthcheck() -> bool:
    """연결 + 스키마 존재 여부를 빠르게 확인한다. CLI/디버깅용."""
    try:
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public' ORDER BY table_name;"
                )
                tables = [r[0] for r in cur.fetchall()]
        print(f"연결 성공. 테이블 {len(tables)}개: {tables}")
        return True
    except Exception as exc:  # noqa: BLE001 — CLI 헬스체크는 원인 그대로 보여주는 게 낫다
        print(f"연결 실패: {type(exc).__name__}: {exc}")
        return False


if __name__ == "__main__":
    healthcheck()
