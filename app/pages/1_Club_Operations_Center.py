"""구단 상황실 — 순위 변동 패널 · 이탈 위험 로스터 · 오늘 경기.

셋 다 features_v1(B) + departure.py(B) + win_rate.py(A) 가 있어야 실데이터가
나온다. 지금은 아직 mock 데이터뿐이라(사용자 요청에 따라 mock은 화면에 숨김),
무엇이 필요한지만 보여준다.
"""

import streamlit as st

from ui.theme import inject_css, init_state, page_header, placeholder, require_team, section, topbar, wrap

st.set_page_config(page_title="구단 상황실", page_icon="⚾", layout="wide", initial_sidebar_state="collapsed")
inject_css()
init_state()
require_team()
topbar("구단 상황실")

with wrap():
    page_header("구단 상황실", "오늘의 결정이 필요한 것부터")

    section("순위 변동 패널")
    placeholder(
        "이탈 시 순위 변동 예측",
        "`src/models/win_rate.py`(A) + `src/models/departure.py`(B) 학습 결과 + "
        "`src/service/simulation.py`(E) 연동이 필요합니다.",
    )

    section("이탈 위험 로스터")
    placeholder(
        "선수별 이탈 확률",
        "`features_v1.parquet`(B 실데이터) + `src/models/departure.py`(B, L1 모델 학습) 이 필요합니다. "
        "지금은 `src/features/contract.py` 목업 스키마만 존재합니다.",
    )

    section("오늘 경기")
    placeholder(
        "우리 팀 오늘 경기",
        "`src/adapters/mlb_api.py`(A) 로 경기 일정을 수집해야 표시할 수 있습니다.",
    )
