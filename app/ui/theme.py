"""네이티브 Streamlit 컴포넌트용 공용 스타일 + 참조 데이터.

목업(mockup_1.html)의 색상 토큰·컴포넌트 디자인을 그대로 가져와 실데이터가 있는
위젯에 입힌다. PLAYERS/가짜 순위 같은 목업 예시 데이터는 여기 포함하지 않는다 —
실제 모델이 없는 항목은 placeholder() 로 "준비중"임을 명시한다.

주의 — st.markdown('<div>') ... st.markdown('</div>') 는 그 사이의 다른 위젯을
실제로 감싸지 않는다(각각 독립된 블록으로 렌더링됨). 위젯을 특정 스타일로
스코프하려면 반드시 st.container(key=...) 를 쓴다 — Streamlit이 그 안의 내용을
".st-key-<key>" 클래스를 가진 실제 div 로 감싸준다.
"""

from __future__ import annotations

from contextlib import contextmanager

import streamlit as st

FONT_LINK = (
    '<link rel="stylesheet" '
    'href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable.min.css">'
)

CSS = """
<style>
:root{
  --navy:#16325C; --navy-2:#1D3E73; --navy-soft:#E9EEF6; --ink:#1F2937; --muted:#6B7280; --faint:#9CA3AF;
  --line:#E5E7EB; --paper:#F5F4F1; --card:#FFF;
  --risk:#D94F4F; --risk-bg:#FCEDED; --gain:#15805E; --gain-bg:#E6F4EF;
  --warn:#B4700A; --warn-bg:#FBF2DF; --violet:#6D4FC2; --violet-bg:#EFEAFB;
  --shadow-sm: 0 1px 2px rgba(16,24,40,.04), 0 1px 3px rgba(16,24,40,.06);
  --shadow-md: 0 4px 10px rgba(16,24,40,.06), 0 2px 4px rgba(16,24,40,.05);
}
html, body, [class*="css"] { font-family: "Pretendard Variable", Pretendard, -apple-system, "Malgun Gothic", sans-serif !important; }

/* Streamlit 기본 크롬 제거 — 우리 topbar 로 대체 */
[data-testid="stSidebar"], [data-testid="stHeader"], [data-testid="stToolbar"], footer { display:none !important; }
.block-container { max-width: 100% !important; padding: 0 0 60px !important; }
.stApp { background: var(--paper); }
#MainMenu { visibility:hidden; }
div[data-testid="stVerticalBlockBorderWrapper"] { gap:0; }
[data-testid="stAppViewContainer"] { animation: gm-fade .35s ease; }
@keyframes gm-fade { from{opacity:0; transform:translateY(4px)} to{opacity:1; transform:translateY(0)} }

/* ── topbar (st.container(key="topbar")) ── */
.st-key-topbar{background:linear-gradient(180deg,var(--navy-2),var(--navy));padding:14px 26px;
  margin-bottom:0;box-shadow:var(--shadow-md);position:relative;z-index:5}
.st-key-topbar .mt-ab{width:34px;height:34px;border-radius:10px;background:#fff;color:var(--navy);
  display:flex;align-items:center;justify-content:center;font-size:12.5px;font-weight:800;
  box-shadow:0 2px 6px rgba(0,0,0,.18);letter-spacing:.3px}
.st-key-topbar .mt-n{font-size:14px;font-weight:700;line-height:1.2;color:#fff}
.st-key-topbar .mt-s{font-size:10.5px;color:rgba(255,255,255,.55)}
.st-key-topbar [data-testid="stPageLink"] p { font-size:13px !important; color:rgba(255,255,255,.7) !important; margin:0 !important; transition:color .15s; }
.st-key-topbar [data-testid="stPageLink"]:hover p { color:#fff !important; }
.st-key-nav_active [data-testid="stPageLink"]{background:#fff !important;border-radius:8px !important;
  padding:4px 2px !important;box-shadow:0 2px 8px rgba(0,0,0,.15)}
.st-key-nav_active [data-testid="stPageLink"] p{color:var(--navy) !important;font-weight:700 !important}

/* ── 컨텐츠 wrap (st.container(key="wrap")) ── */
.st-key-wrap{max-width:960px;margin:0 auto;padding:28px 26px 0}
.gm-kicker{font-size:12px;color:var(--muted);letter-spacing:1.4px;margin-bottom:6px;font-weight:700;
  text-transform:uppercase}
.gm-title{font-size:27px;font-weight:800;letter-spacing:-.6px;margin:0 0 4px;color:var(--ink)}
.gm-desc{font-size:13.5px;color:var(--muted);margin:0 0 6px}
.gm-section{font-size:12.5px;color:var(--ink);font-weight:700;letter-spacing:.2px;
  margin:28px 0 12px;display:flex;align-items:center;gap:8px;justify-content:space-between}
.gm-section > span:first-child{display:flex;align-items:center;gap:8px}
.gm-section > span:first-child::before{content:"";width:4px;height:14px;border-radius:2px;
  background:var(--navy);display:inline-block}

/* ── 카드 / 배지 / KPI (순수 HTML 문자열이라 중첩 문제 없음) ── */
.gm-card{background:var(--card);border:1px solid var(--line);border-radius:14px;
  padding:18px 20px;margin-bottom:10px;box-shadow:var(--shadow-sm);transition:box-shadow .2s, transform .2s}
.gm-card:hover{box-shadow:var(--shadow-md);transform:translateY(-1px)}
.gm-badge{display:inline-block;font-size:10.5px;padding:3px 10px;border-radius:20px;font-weight:700;
  letter-spacing:.2px}
.gm-badge.warn{background:var(--warn-bg);color:var(--warn)}
.gm-badge.gain{background:var(--gain-bg);color:var(--gain)}
.gm-badge.risk{background:var(--risk-bg);color:var(--risk)}
.gm-badge.navy{background:var(--navy-soft);color:var(--navy)}
.gm-badge.violet{background:var(--violet-bg);color:var(--violet)}
.gm-placeholder{background:linear-gradient(180deg,#FAFAF9,var(--card));border:1px dashed #D7D3CC;
  border-radius:14px;padding:18px 20px;margin-bottom:10px;display:flex;gap:14px;align-items:flex-start;
  transition:border-color .2s}
.gm-placeholder:hover{border-color:var(--navy)}
.gm-ph-icon{width:34px;height:34px;border-radius:10px;background:var(--warn-bg);color:var(--warn);
  display:flex;align-items:center;justify-content:center;font-size:15px;flex-shrink:0}
.gm-ph-body b{color:var(--ink);font-size:13.5px}
.gm-ph-body .desc{color:var(--muted);font-size:12.5px;line-height:1.6;margin-top:3px}
.gm-kpi-l{font-size:11.5px;color:var(--muted);display:flex;align-items:center;gap:6px}
.gm-kpi-v{font-size:25px;font-weight:800;margin-top:4px;color:var(--ink);letter-spacing:-.4px}
.gm-kpi-accent{border-left:3px solid var(--navy);padding-left:14px}

/* 진행률 바 */
.gm-progress-wrap{display:flex;align-items:center;gap:12px;margin-bottom:14px}
.gm-progress-track{flex:1;height:8px;background:var(--line);border-radius:4px;overflow:hidden}
.gm-progress-fill{height:100%;background:linear-gradient(90deg,var(--navy),#3E6FB0);border-radius:4px;
  transition:width .4s ease}
.gm-progress-label{font-size:12px;color:var(--muted);white-space:nowrap;font-weight:600}

/* ── hero (st.container(key="hero")) — 진입화면 ── */
.st-key-hero{background:radial-gradient(1200px 500px at 15% -10%, #23477E 0%, var(--navy) 55%);
  padding:52px 26px 40px}
.gm-header-block{max-width:900px;margin:0 auto 6px}
.st-key-hero .gm-kicker{color:rgba(255,255,255,.55)}
.st-key-hero .gm-title{color:#fff;font-size:33px;text-shadow:0 1px 2px rgba(0,0,0,.1)}
.st-key-hero .gm-desc{color:rgba(255,255,255,.65);font-size:14px;margin-bottom:6px}

.st-key-hero [data-testid="stTabs"] [role="tablist"]{background:rgba(255,255,255,.08);padding:4px;
  border-radius:11px;gap:2px;width:fit-content;max-width:900px;margin:0 auto;border-bottom:none !important;
  box-shadow:none !important}
.st-key-hero [data-testid="stTab"]{color:rgba(255,255,255,.7) !important;border-radius:8px !important;
  font-size:12.5px !important;padding:8px 16px !important;height:auto !important}
.st-key-hero [data-testid="stTab"] p{color:inherit !important;font-size:12.5px !important}
.st-key-hero [data-testid="stTab"][aria-selected="true"]{background:#fff !important;color:var(--navy) !important;font-weight:600 !important}
.st-key-hero [data-testid="stTab"] [data-testid="stMarkdownContainer"]{color:inherit !important}
.st-key-hero [data-testid="stTabs"] [data-baseweb="tab-highlight"]{display:none !important}
.st-key-hero [data-testid="stTabs"] [data-baseweb="tab-border"]{display:none !important}
.st-key-hero [data-testid="stTabPanel"]{max-width:900px;margin:0 auto;padding-left:0;padding-right:0}

.st-key-hero div[data-testid="stButton"] button{
  background:rgba(255,255,255,.07) !important;border:1px solid rgba(255,255,255,.14) !important;
  border-radius:12px !important;color:#fff !important;padding:12px 8px !important;
  width:100%;white-space:pre-wrap;line-height:1.5;font-size:13px !important;
  transition:all .18s cubic-bezier(.2,.8,.2,1);min-height:64px;box-shadow:0 1px 2px rgba(0,0,0,.12)}
.st-key-hero div[data-testid="stButton"] button:hover{
  background:#fff !important;color:var(--navy) !important;border-color:#fff !important;
  transform:translateY(-3px) scale(1.015);box-shadow:0 10px 20px rgba(0,0,0,.25)}
.st-key-hero div[data-testid="stButton"] button:active{transform:translateY(-1px) scale(1)}
.st-key-hero div[data-testid="stButton"] button p{ color:inherit !important; }
.gm-division-label{font-size:11px;color:rgba(255,255,255,.45);letter-spacing:1px;margin:22px 0 9px;
  text-transform:uppercase;font-weight:700;display:flex;align-items:center;gap:8px}
.gm-division-label::after{content:"";flex:1;height:1px;background:rgba(255,255,255,.12)}

/* ── 일반(밝은 배경) 버튼 — 뒤로가기 등 ── */
div[data-testid="stButton"] button{
  border-radius:20px !important;border:1px solid var(--line) !important;background:#fff !important;
  color:var(--muted) !important;font-size:12.5px !important;padding:6px 16px !important;
  box-shadow:var(--shadow-sm);transition:all .15s}
div[data-testid="stButton"] button:hover{border-color:var(--navy) !important;color:var(--navy) !important;
  box-shadow:var(--shadow-md);transform:translateY(-1px)}
.st-key-hero div[data-testid="stButton"] button:hover{color:var(--navy) !important}

/* dataframe 라운딩 */
[data-testid="stDataFrame"]{border-radius:12px;overflow:hidden;border:1px solid var(--line);
  box-shadow:var(--shadow-sm)}

/* 카드 안 테이블 행 호버 */
.gm-card table tr:hover td{background:var(--paper)}
.gm-card table td{border-bottom:1px solid var(--line)}
.gm-card table tr:last-child td{border-bottom:none}

/* expander 살짝 카드화 */
[data-testid="stExpander"]{border-radius:14px !important;border:1px solid var(--line) !important;
  box-shadow:var(--shadow-sm);overflow:hidden}
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

PAGES = [
    ("구단 상황실", "pages/1_Club_Operations_Center.py"),
    ("선수 리포트", "pages/2_Player_Report.py"),
    ("모델 정보", "pages/3_Model_Information.py"),
]


def inject_css() -> None:
    # st.markdown 은 큰 <style> 블록을 마크다운 파서가 중간에 텍스트로 흘려버리는
    # 경우가 있다. st.html() 은 마크다운 파싱을 거치지 않고 그대로 주입한다.
    st.html(FONT_LINK + CSS)


def init_state() -> None:
    st.session_state.setdefault("team_code", None)


def selected_team() -> str | None:
    init_state()
    return st.session_state.get("team_code")


def topbar(active_label: str) -> None:
    """목업의 navy 상단바 — 팀 배지 + 3탭 네비게이션 (실제 DOM 중첩: st.container 사용)."""
    team_code = selected_team() or "?"
    team_name = TEAM_NAMES.get(team_code, "구단 미선택")

    with st.container(key="topbar"):
        c_badge, c_name, *c_tabs, c_change = st.columns([0.5, 2] + [1.1] * len(PAGES) + [1])
        with c_badge:
            st.markdown(f'<div class="mt-ab">{team_code}</div>', unsafe_allow_html=True)
        with c_name:
            st.markdown(
                f'<div class="mt-n">{team_name}</div><div class="mt-s">GM Mode</div>',
                unsafe_allow_html=True,
            )
        for i, (col, (label, target)) in enumerate(zip(c_tabs, PAGES)):
            with col:
                if label == active_label:
                    with st.container(key="nav_active"):
                        st.page_link(target, label=label)
                else:
                    st.page_link(target, label=label)
        with c_change:
            st.page_link("Home.py", label="⟲ 구단 변경")


@contextmanager
def wrap():
    """본문 컨텐츠를 960px 중앙 정렬 영역으로 감싼다."""
    with st.container(key="wrap"):
        yield


def require_team() -> str:
    """팀이 선택되지 않았으면 안내 후 중단한다."""
    team_code = selected_team()
    if not team_code:
        with wrap():
            st.info("먼저 진입 화면에서 구단을 선택해주세요.")
            st.page_link("Home.py", label="← 구단 선택으로")
        st.stop()
    return team_code


def page_header(kicker: str, title: str, desc: str = "") -> None:
    st.markdown(
        '<div class="gm-header-block">'
        f'<div class="gm-kicker">{kicker}</div>'
        f'<div class="gm-title">{title}</div>'
        + (f'<div class="gm-desc">{desc}</div>' if desc else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def section(label: str, sublabel: str = "") -> None:
    st.markdown(
        f'<div class="gm-section"><span>{label}</span>'
        + (f'<span style="font-weight:400;color:var(--faint)">{sublabel}</span>' if sublabel else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def placeholder(what: str, needs: str) -> None:
    """아직 실데이터가 없는 기능임을 명시한다 — 가짜 값으로 채우지 않는다."""
    st.markdown(
        '<div class="gm-placeholder">'
        '<div class="gm-ph-icon">🚧</div>'
        '<div class="gm-ph-body">'
        f'<b>{what}</b>'
        f'<div class="desc">필요: {needs}</div>'
        "</div></div>",
        unsafe_allow_html=True,
    )


def badge(text: str, kind: str = "navy") -> str:
    return f'<span class="gm-badge {kind}">{text}</span>'


def kpi_card(label: str, value: str, icon: str = "", color: str = "var(--ink)") -> None:
    icon_html = f"{icon} " if icon else ""
    st.markdown(
        f'<div class="gm-card gm-kpi-accent">'
        f'<div class="gm-kpi-l">{icon_html}{label}</div>'
        f'<div class="gm-kpi-v" style="color:{color}">{value}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def progress_bar(done: int, total: int, label: str = "") -> None:
    pct = 0 if total == 0 else round(done / total * 100)
    st.markdown(
        '<div class="gm-progress-wrap">'
        f'<div class="gm-progress-track"><div class="gm-progress-fill" style="width:{pct}%"></div></div>'
        f'<div class="gm-progress-label">{label or f"{done}/{total} 완료"}</div>'
        "</div>",
        unsafe_allow_html=True,
    )
