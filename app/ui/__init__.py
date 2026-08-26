"""Streamlit mock-up 공통 UI와 예시 데이터."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


TEAMS = {
    "AL EAST": {
        "NYY": ("New York Yankees", "3위 · 62승"),
        "BAL": ("Baltimore Orioles", "5위 · 58승"),
        "BOS": ("Boston Red Sox", "8위 · 54승"),
        "TBR": ("Tampa Bay Rays", "11위 · 51승"),
        "TOR": ("Toronto Blue Jays", "14위 · 48승"),
    },
    "NL WEST": {
        "LAD": ("Los Angeles Dodgers", "1위 · 71승"),
        "SDP": ("San Diego Padres", "4위 · 60승"),
        "ARI": ("Arizona Diamondbacks", "9위 · 53승"),
        "SFG": ("San Francisco Giants", "12위 · 50승"),
        "COL": ("Colorado Rockies", "28위 · 38승"),
    },
    "기타 지구": {
        "PHI": ("Philadelphia Phillies", "2위 · 66승"),
        "ATL": ("Atlanta Braves", "6위 · 57승"),
        "CLE": ("Cleveland Guardians", "7위 · 56승"),
        "HOU": ("Houston Astros", "10위 · 52승"),
        "MIL": ("Milwaukee Brewers", "13위 · 49승"),
    },
}

TEAM_LOOKUP = {
    code: {"name": name, "record": record, "division": division}
    for division, teams in TEAMS.items()
    for code, (name, record) in teams.items()
}

PLAYERS = {
    "A. Judge": {"initials": "AJ", "position": "RF", "age": 34, "score": 91.2, "risk": 78},
    "G. Cole": {"initials": "GC", "position": "SP", "age": 36, "score": 94.1, "risk": 61},
    "A. Volpe": {"initials": "AV", "position": "SS", "age": 25, "score": 78.5, "risk": 19},
    "M. Rizzo": {"initials": "MR", "position": "1B", "age": 36, "score": 72.8, "risk": 47},
}


def init_state() -> None:
    st.session_state.setdefault("league", "MLB")
    st.session_state.setdefault("team_code", "NYY")
    st.session_state.setdefault("player_name", "A. Judge")
    st.session_state.setdefault("scenario", "트레이드 · 시즌 전체")


def sidebar_context() -> None:
    init_state()
    team = TEAM_LOOKUP[st.session_state.team_code]
    with st.sidebar:
        st.caption("GENERAL MANAGER MODE")
        st.title(st.session_state.team_code)
        st.write(team["name"])
        st.caption(team["record"])
        st.divider()
        codes = list(TEAM_LOOKUP)
        selected = st.selectbox(
            "구단 변경",
            codes,
            index=codes.index(st.session_state.team_code),
            format_func=lambda code: f"{code} · {TEAM_LOOKUP[code]['name']}",
        )
        st.session_state.team_code = selected
        st.caption("시즌 예상 승률")
        st.metric("", "61.3%")


def team_name() -> str:
    return TEAM_LOOKUP[st.session_state.team_code]["name"]


def render_mockup(screen: str) -> None:
    """원본 목업의 CSS/마크업을 그대로 사용해 지정 화면만 렌더링한다."""
    init_state()
    source = Path(__file__).parent / "templates" / "mockup.html"
    streamlit_style_source = Path(__file__).parent / "templates" / "streamlit_style.html"
    css_source = Path(__file__).parent / "css" / "mockup.css"
    streamlit_css_source = Path(__file__).parent / "css" / "streamlit.css"
    js_source = Path(__file__).parent / "js" / "mockup.js"
    html = source.read_text(encoding="utf-8")
    streamlit_style = streamlit_style_source.read_text(encoding="utf-8")
    css = css_source.read_text(encoding="utf-8")
    streamlit_css = streamlit_css_source.read_text(encoding="utf-8")
    javascript = js_source.read_text(encoding="utf-8")
    html = html.replace("__MOCKUP_CSS__", css)
    html = html.replace("__MOCKUP_JS__", javascript)
    streamlit_style = streamlit_style.replace("__STREAMLIT_CSS__", streamlit_css)
    team = TEAM_LOOKUP[st.session_state.team_code]
    html = html.replace("{{TEAM_CODE}}", st.session_state.team_code)
    html = html.replace("{{TEAM_NAME}}", team["name"])
    html = html.replace("{{SCREEN}}", screen)
    height = {"home": 1040, "war": 1050, "report": 1250, "about": 1200}[screen]
    st.markdown(streamlit_style, unsafe_allow_html=True)
    components.html(html, height=height, scrolling=True)
