"""구단 상황실 — E 시뮬레이션을 features_v1과 연결한다."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.models.recommend import adapt_features_v1
from src.service.simulation import TeamStrength, calculate_team_strength, simulate
from ui.theme import inject_css, init_state, page_header, require_team, section, topbar, wrap

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data" / "processed" / "features_v1.parquet"


@st.cache_data(show_spinner=False)
def load_players() -> pd.DataFrame:
    return adapt_features_v1(pd.read_parquet(FEATURES_PATH))


def predict_win_rate(strength: TeamStrength) -> float:
    return float(np.clip(0.35 + strength.overall * 0.003, 0.25, 0.75))


def make_rank_predictor(season_players: pd.DataFrame):
    baselines = {
        team: calculate_team_strength(group).overall
        for team, group in season_players.groupby("team_last")
    }

    def predict_rank(strength: TeamStrength) -> int:
        return 1 + sum(value > strength.overall for value in baselines.values())

    return predict_rank


st.set_page_config(page_title="구단 상황실", page_icon="⚾", layout="wide", initial_sidebar_state="collapsed")
inject_css()
init_state()
team_code = require_team()
topbar("구단 상황실")

with wrap():
    page_header("구단 상황실", "이탈이 팀 전력과 순위에 미치는 영향")

    try:
        players = load_players()
    except FileNotFoundError:
        st.error("data/processed/features_v1.parquet 파일이 없습니다.")
        st.stop()
    except ImportError:
        st.error("parquet을 읽기 위한 pyarrow 설치가 필요합니다.")
        st.stop()
    except ValueError as exc:
        st.error(f"features_v1 데이터 계약 오류: {exc}")
        st.stop()
    except Exception as exc:
        # 파일 손상 등 예상하지 못한 로딩 오류도 traceback 대신 화면에 안내한다.
        st.error(f"features_v1을 읽는 중 오류가 발생했습니다: {exc}")
        st.stop()

    quality = players.attrs.get("data_quality", {})
    if quality.get("excluded_rows", 0) or quality.get("imputed_g_ratio", 0):
        st.info(
            f"데이터 정리: {quality.get('excluded_rows', 0)}행 제외, "
            f"출전 비중 {quality.get('imputed_g_ratio', 0)}건 보정"
        )

    season = int(players["season"].max())
    season_players = players.loc[players["season"] == season].copy()
    team_players = season_players.loc[season_players["team_last"] == team_code].copy()
    if team_players.empty:
        st.warning(f"{season}시즌 {team_code} 선수 데이터가 없습니다.")
        st.stop()

    team_players = team_players.sort_values("overall_score", ascending=False)
    default_player = st.session_state.get("selected_player_id")
    ids = team_players["player_id"].astype(str).tolist()
    selected_id = st.selectbox(
        "이탈 시뮬레이션 선수",
        ids,
        index=ids.index(default_player) if default_player in ids else 0,
        format_func=lambda pid: f"{pid} · 전력 {team_players.loc[team_players.player_id.astype(str).eq(pid), 'overall_score'].iloc[0]:.1f}",
    )
    st.session_state.selected_player_id = selected_id

    result = simulate(
        team_players,
        selected_id,
        predict_win_rate,
        rank_predictor=make_rank_predictor(season_players),
    )

    section("순위 변동 패널", f"{season}시즌 features_v1 기반")
    c1, c2, c3 = st.columns(3)
    c1.metric("현재 예상 승률", f"{result.current_win_rate:.1%}")
    c2.metric("이탈 후 예상 승률", f"{result.after_departure_win_rate:.1%}", f"{result.impact:+.1%}p")
    c3.metric("예상 순위", f"{result.rank_before}위 → {result.rank_after}위")
    st.caption("승률·순위 함수는 simulation.py에 주입되어 실제 예측 함수로 교체할 수 있습니다.")

    section("전력 로스터", "전력 순")
    roster = team_players[["player_id", "role", "overall_score", "g_ratio"]].head(15).copy()
    roster.columns = ["선수", "역할", "전력", "출전 비중"]
    st.dataframe(roster, hide_index=True, use_container_width=True)

    if st.button("선수 리포트에서 대체 후보 보기", type="primary"):
        st.switch_page("pages/2_Player_Report.py")
