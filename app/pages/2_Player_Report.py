"""선수 리포트 — 단장 브리핑 · 승률 트랙 · 영입 시뮬레이션 · 근거.

선수 단위 이탈 확률(B) · 이탈 유형(C) · 대체 후보(E) 가 전부 있어야 성립하는
화면이라, 지금은 어떤 파트가 무엇을 만들어야 하는지만 보여준다.
"""

import streamlit as st

from ui.theme import inject_css, init_state, page_header, placeholder

st.set_page_config(page_title="선수 리포트", page_icon="⚾", layout="wide")
inject_css()
init_state()

if not st.session_state.get("team_code"):
    st.info("먼저 진입 화면에서 구단을 선택해주세요.")
    if st.button("← 구단 선택으로"):
        st.switch_page("Home.py")
    st.stop()

page_header("선수 리포트", "선수 단위 이탈 예측 · 영입 시뮬레이션")

placeholder(
    "단장 브리핑 (이탈 확률 · 승률 영향 · 대체 시나리오)",
    "아래가 전부 갖춰져야 표시됩니다 — 지금은 아무것도 학습되지 않았습니다.",
)

st.markdown("**필요한 것**")
st.markdown(
    "- `src/features/labels.py`(C) — L1/L2/L2b/L3 실제 라벨\n"
    "- `src/models/departure.py`(B) — 이탈 여부(L1) 학습\n"
    "- `src/models/reason.py`(C) — 이탈 유형(L2/L2b) 학습\n"
    "- `src/models/recommend.py`(E) — 대체 후보 추천 (코드는 있음, 실데이터 대기)\n"
    "- `src/service/simulation.py`(E) — 시뮬레이션 엔진 (코드는 있음, 실데이터 대기)"
)

if st.button("← 구단 상황실로"):
    st.switch_page("pages/1_Club_Operations_Center.py")
