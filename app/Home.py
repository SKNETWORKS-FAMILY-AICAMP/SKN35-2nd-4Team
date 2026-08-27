"""진입 화면 — 리그 조회 + 구단 선택.

오늘 경기·예상 순위는 A 파트(mlb_api.py, win_rate.py)가 아직 비어 있어 실데이터가
없다. 가짜 숫자를 채우는 대신 무엇이 필요한지 명시한다. 구단 선택은 실제 MLB
팀·디비전 정보(예측이 아닌 사실)라 그대로 보여준다.
"""

import streamlit as st

from ui.theme import DIVISIONS, inject_css, init_state, page_header, placeholder, section

st.set_page_config(page_title="구단 선택", page_icon="⚾", layout="wide", initial_sidebar_state="collapsed")
inject_css()
init_state()

with st.container(key="hero"):
    page_header("GENERAL MANAGER MODE", "오늘의 리그를 확인하고 맡을 구단을 선택하세요")

    tab_today, tab_rank, tab_pick = st.tabs(["오늘 경기", "예상 순위", "구단 선택"])

    with tab_today:
        placeholder(
            "오늘 경기 목록",
            "`src/adapters/mlb_api.py`(A) 로 MLB Stats API에서 경기 일정을 수집하고, "
            "`src/models/game.py`(A) 로 승부를 예측해야 표시할 수 있습니다.",
        )

    with tab_rank:
        placeholder(
            "예상 순위",
            "`src/models/win_rate.py`(A) 로 팀 승률을 예측하고 "
            "`team_season.pred_rank` 컬럼을 채워야 표시할 수 있습니다.",
        )

    with tab_pick:
        for division, teams in DIVISIONS.items():
            st.markdown(f'<div class="gm-division-label">{division}</div>', unsafe_allow_html=True)
            cols = st.columns(5)
            for col, (code, name) in zip(cols, teams):
                with col:
                    if st.button(f"**{code}**\n\n{name}", key=f"pick_{code}", use_container_width=True):
                        st.session_state.team_code = code
                        st.switch_page("pages/1_Club_Operations_Center.py")
