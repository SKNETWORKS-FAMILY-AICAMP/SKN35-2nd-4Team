"""데이터 출처 한 곳으로 모으기 — Supabase 우선, 실패하면 리포의 로컬 파일.

배경
----
화면은 원래 `pd.read_parquet("data/final/features_v1.parquet")` 처럼 로컬
파일을 직접 읽었다. Streamlit Cloud 배포 후에는 Supabase 에서 끌어와야 하는데,
호출부를 전부 DB 호출로 바꿔버리면 DB 가 잠깐 끊겼을 때 앱 전체가 죽는다.

그래서 여기서 한 겹 감싼다:
    1) DB 연결 정보가 있으면 Supabase 에서 읽는다
    2) 실패하면(미설정·네트워크·테이블 없음) 리포에 들어있는 파일로 넘어간다
    3) 어느 쪽을 썼는지 기록해 화면 하단에 표시한다

로컬 파일은 리포에 함께 배포되므로 폴백은 항상 가능하다 — DB 장애가 곧
서비스 장애가 되지 않게 하는 것이 목적이다.

주의
----
- 모델 파일(models/*.pkl, *.pt)은 DB 대상이 아니다. 산출물이라 리포에 둔다.
- st.cache_data 로 감싸서 rerun 마다 DB 를 두들기지 않는다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

FEATURES_PATH = ROOT / "data" / "final" / "features_v1.parquet"
INJURY_PATH = ROOT / "data" / "final" / "player_injury_stints.csv"
TEAM_SEASON_PATH = ROOT / "data" / "final" / "team_season.csv"

# 이번 세션에서 실제로 어느 출처를 썼는지 (화면 표시용)
_SOURCES: dict[str, str] = {}


def data_sources() -> dict[str, str]:
    """{데이터 이름: 'supabase' | 'local'}."""
    return dict(_SOURCES)


def _db_enabled() -> bool:
    """DB 연결 정보가 있는지. 없으면 시도조차 하지 않는다(불필요한 예외 방지)."""
    try:
        from src.storage.supabase_client import _database_url

        return bool(_database_url())
    except Exception:
        return False


def _try_db(name: str, fetch, fallback):
    """DB 우선, 실패하면 로컬. 어느 쪽을 썼는지 _SOURCES 에 남긴다.

    예외를 넓게 잡는 이유: psycopg2 연결 오류·타임아웃·테이블 없음·스키마
    불일치가 모두 다른 예외로 오는데, 어느 쪽이든 폴백 동작은 같아야 한다.
    사용자에게 traceback 을 던지는 대신 로컬 데이터로 계속 굴러가는 것이 낫다.
    """
    if _db_enabled():
        try:
            df = fetch()
            if df is not None and not df.empty:
                _SOURCES[name] = "supabase"
                return df
        except Exception as exc:  # noqa: BLE001 — 원인과 무관하게 폴백이 정답
            _SOURCES[name] = f"local (DB 실패: {type(exc).__name__})"
            return fallback()
    _SOURCES[name] = "local"
    return fallback()


@st.cache_data(show_spinner=False, ttl=600)
def load_features() -> pd.DataFrame:
    """features_v1 — 앱의 주 데이터."""
    def _db():
        from src.storage.queries import fetch_features_v1

        return fetch_features_v1()

    return _try_db("features_v1", _db, lambda: pd.read_parquet(FEATURES_PATH))


@st.cache_data(show_spinner=False, ttl=600)
def load_injury() -> pd.DataFrame | None:
    """부상 스틴트. 어느 쪽에도 없으면 None — 호출부가 0으로 채운다."""
    def _db():
        from src.storage.queries import fetch_injury_stints

        return fetch_injury_stints()

    def _local():
        return pd.read_csv(INJURY_PATH) if INJURY_PATH.exists() else None

    try:
        return _try_db("player_injury_stints", _db, _local)
    except (OSError, ValueError):
        return None


@st.cache_data(show_spinner=False, ttl=600)
def load_team_season() -> pd.DataFrame | None:
    """팀 시즌 성적 (홈 화면 예상 순위)."""
    def _db():
        from src.storage.queries import fetch_team_season

        return fetch_team_season()

    def _local():
        return pd.read_csv(TEAM_SEASON_PATH) if TEAM_SEASON_PATH.exists() else None

    try:
        return _try_db("team_season", _db, _local)
    except (OSError, ValueError):
        return None


def source_caption() -> str:
    """화면 하단에 붙일 출처 안내. DB 장애를 조용히 숨기지 않기 위한 것."""
    src = data_sources()
    if not src:
        return ""
    if all(v == "supabase" for v in src.values()):
        return "데이터 출처: Supabase"
    if all(v == "local" for v in src.values()):
        return "데이터 출처: 리포 내 로컬 파일 (DB 미설정)"
    failed = [f"{k}({v})" for k, v in src.items() if v.startswith("local (DB 실패")]
    if failed:
        return "데이터 출처: 일부 로컬 폴백 — " + ", ".join(failed)
    return "데이터 출처: " + ", ".join(f"{k}={v}" for k, v in src.items())
