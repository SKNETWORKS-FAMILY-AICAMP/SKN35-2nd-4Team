"""네이티브 Streamlit 컴포넌트용 공용 스타일 + 참조 데이터.

목업(mockup.html)의 색상 토큰을 그대로 가져와 실데이터가 있는 위젯에 입힌다.
PLAYERS/가짜 순위 같은 목업 예시 데이터는 여기 포함하지 않는다 — 실제 모델이
없는 항목은 화면에서 아예 숨기거나 placeholder() 로 "준비중"임을 명시한다.
"""

from __future__ import annotations

import streamlit as st

CSS = """
<style>
:root{
  --navy:#16325C; --navy-soft:#E9EEF6; --ink:#1F2937; --muted:#6B7280; --faint:#9CA3AF;
  --line:#E5E7EB; --paper:#F5F4F1; --card:#FFF;
  --risk:#D94F4F; --risk-bg:#FCEDED; --gain:#15805E; --gain-bg:#E6F4EF;
  --warn:#B4700A; --warn-bg:#FBF2DF;
}
.gm-kicker{font-size:12px;color:var(--muted);letter-spacing:1.2px;margin-bottom:6px;font-weight:600}
.gm-title{font-size:28px;font-weight:700;letter-spacing:-.5px;margin:0 0 4px;color:var(--ink)}
.gm-desc{font-size:13.5px;color:var(--muted);margin:0 0 18px}
.gm-section{font-size:12px;color:var(--faint);font-weight:600;letter-spacing:.4px;
  margin:22px 0 8px;text-transform:none}
.gm-card{background:var(--card);border:1px solid var(--line);border-radius:14px;
  padding:16px 18px;margin-bottom:10px}
.gm-badge{display:inline-block;font-size:10.5px;padding:2px 9px;border-radius:10px;font-weight:600}
.gm-badge.warn{background:var(--warn-bg);color:var(--warn)}
.gm-badge.gain{background:var(--gain-bg);color:var(--gain)}
.gm-badge.risk{background:var(--risk-bg);color:var(--risk)}
.gm-placeholder{background:var(--card);border:1px dashed var(--line);border-radius:14px;
  padding:22px 22px;color:var(--muted);font-size:13px;line-height:1.7}
.gm-placeholder b{color:var(--ink)}
</style>
"""

# 실제 MLB 30개 팀 · 6개 디비전 (정적 참조 데이터 — 예측값이 아니라 사실이라 표시해도 됨)
DIVISIONS: dict[str, list[tuple[str, str]]] = {
    "AL East": [
        ("NYY", "New York Yankees"), ("BOS", "Boston Red Sox"), ("TOR", "Toronto Blue Jays"),
        ("TBR", "Tampa Bay Rays"), ("BAL", "Baltimore Orioles"),
    ],
    "AL Central": [
        ("CLE", "Cleveland Guardians"), ("MIN", "Minnesota Twins"), ("CHW", "Chicago White Sox"),
        ("DET", "Detroit Tigers"), ("KCR", "Kansas City Royals"),
    ],
    "AL West": [
        ("HOU", "Houston Astros"), ("SEA", "Seattle Mariners"), ("TEX", "Texas Rangers"),
        ("LAA", "Los Angeles Angels"), ("ATH", "Athletics"),
    ],
    "NL East": [
        ("ATL", "Atlanta Braves"), ("NYM", "New York Mets"), ("PHI", "Philadelphia Phillies"),
        ("MIA", "Miami Marlins"), ("WSN", "Washington Nationals"),
    ],
    "NL Central": [
        ("MIL", "Milwaukee Brewers"), ("CHC", "Chicago Cubs"), ("STL", "St. Louis Cardinals"),
        ("CIN", "Cincinnati Reds"), ("PIT", "Pittsburgh Pirates"),
    ],
    "NL West": [
        ("LAD", "Los Angeles Dodgers"), ("SDP", "San Diego Padres"), ("SFG", "San Francisco Giants"),
        ("ARI", "Arizona Diamondbacks"), ("COL", "Colorado Rockies"),
    ],
}
TEAM_NAMES: dict[str, str] = {code: name for teams in DIVISIONS.values() for code, name in teams}


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def init_state() -> None:
    st.session_state.setdefault("team_code", None)


def page_header(kicker: str, title: str, desc: str = "") -> None:
    st.markdown(
        f'<div class="gm-kicker">{kicker}</div>'
        f'<div class="gm-title">{title}</div>'
        + (f'<div class="gm-desc">{desc}</div>' if desc else ""),
        unsafe_allow_html=True,
    )


def section(label: str) -> None:
    st.markdown(f'<div class="gm-section">{label}</div>', unsafe_allow_html=True)


def placeholder(what: str, needs: str) -> None:
    """아직 실데이터가 없는 기능임을 명시한다 — 가짜 값으로 채우지 않는다."""
    st.markdown(
        f'<div class="gm-placeholder">🚧 <b>{what}</b> — 아직 준비되지 않았습니다.<br>'
        f'필요: {needs}</div>',
        unsafe_allow_html=True,
    )


def selected_team() -> str | None:
    init_state()
    return st.session_state.get("team_code")
