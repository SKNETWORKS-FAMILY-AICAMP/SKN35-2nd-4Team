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

# ── 색상 유틸 (팀 아이덴티티 색을 --navy 계열 토큰으로 변환할 때 사용) ──


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _mix_hex(hex_a: str, hex_b: str, t: float) -> str:
    """hex_a 와 hex_b 를 t(0~1) 비율로 섞는다. t=0 -> a, t=1 -> b."""
    ra, ga, ba = _hex_to_rgb(hex_a)
    rb, gb, bb = _hex_to_rgb(hex_b)
    r = round(ra + (rb - ra) * t)
    g = round(ga + (gb - ga) * t)
    b = round(ba + (bb - ba) * t)
    return f"#{r:02x}{g:02x}{b:02x}"

FONT_LINK = (
    '<link rel="stylesheet" '
    'href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable.min.css">'
)

CSS = """
<style>
:root{
  --navy:#16325C; --navy-2:#1D3E73; --navy-soft:#E9EEF6; --team-accent:#3E6FB0; --hero-glow:#23477E;
  --ink:#1F2937; --muted:#6B7280; --faint:#9CA3AF;
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
  margin-bottom:0;box-shadow:var(--shadow-md);position:relative;z-index:5;
  border-bottom:3px solid var(--team-accent);transition:background .3s ease,border-color .3s ease}
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
.st-key-hero{background:radial-gradient(1200px 500px at 15% -10%, var(--hero-glow) 0%, var(--navy) 55%);
  padding:52px 26px 40px;transition:background .3s ease}
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
  box-shadow:var(--shadow-sm);transition:all .15s;position:relative;overflow:hidden}
div[data-testid="stButton"] button:hover{border-color:var(--navy) !important;color:var(--navy) !important;
  box-shadow:var(--shadow-md);transform:translateY(-1px)}
.st-key-hero div[data-testid="stButton"] button:hover{color:var(--navy) !important}

/* 버튼 hover 시 카드처럼 스치는 광택 — 모든 stButton 공통 */
@keyframes gm-shine{0%{transform:translateX(-140%) skewX(-18deg)}100%{transform:translateX(240%) skewX(-18deg)}}
div[data-testid="stButton"] button::after{content:"";position:absolute;top:0;left:0;width:38%;height:100%;
  background:linear-gradient(115deg,transparent,rgba(255,255,255,.55),transparent);
  transform:translateX(-140%) skewX(-18deg);pointer-events:none}
div[data-testid="stButton"] button:hover::after{animation:gm-shine .7s ease forwards}
@media (prefers-reduced-motion: reduce){
  div[data-testid="stButton"] button:hover::after{animation:none}
}

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

/* ══ 영입 후보 카드 — FIFA 얼티밋팀 스타일 ══ */
.gm-pcard-row{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:6px}
.gm-pcard{position:relative;width:100%;border-radius:16px;padding:14px 12px 12px;overflow:hidden;
  animation:gm-pop .55s cubic-bezier(.2,.9,.25,1.15) both;animation-delay:calc(var(--i,0) * 100ms);
  transition:transform .18s ease, box-shadow .18s ease;border:1px solid rgba(255,255,255,.12)}
.gm-pcard::after{content:"";position:absolute;inset:0;background:
  linear-gradient(115deg,rgba(255,255,255,.35) 0%,rgba(255,255,255,0) 30%);pointer-events:none}
.gm-pcard:hover{transform:translateY(-5px) scale(1.025)}
@keyframes gm-pop{from{opacity:0;transform:translateY(22px) scale(.82) rotate(-2deg)}
  to{opacity:1;transform:translateY(0) scale(1) rotate(0)}}

.gm-pcard.tier-gold{background:linear-gradient(160deg,#7A5B15,#E3B438 45%,#F8E29A 58%,#8A6417);
  box-shadow:0 8px 18px rgba(180,130,10,.35)}
.gm-pcard.tier-silver{background:linear-gradient(160deg,#57616D,#A6B0BC 45%,#EAEEF2 58%,#636E7A);
  box-shadow:0 8px 18px rgba(60,70,90,.22)}
.gm-pcard.tier-bronze{background:linear-gradient(160deg,#5A3A22,#9C6A3E 45%,#C99B6C 58%,#5A3A22);
  box-shadow:0 8px 18px rgba(90,50,20,.22)}
.gm-pcard.selected{box-shadow:0 0 0 3px var(--navy),0 12px 26px rgba(22,50,92,.4);transform:translateY(-4px)}
.gm-pcard.selected::before{content:"✓ 선택됨";position:absolute;top:8px;right:8px;font-size:9.5px;
  font-weight:800;color:var(--navy);background:#fff;border-radius:20px;padding:2px 8px;z-index:2;
  box-shadow:0 1px 4px rgba(0,0,0,.25)}

.gm-pcard .pc-top{display:flex;justify-content:space-between;align-items:flex-start;position:relative;z-index:1}
.gm-pcard .pc-ovr{font-size:25px;font-weight:800;color:#fff;line-height:1;text-shadow:0 1px 3px rgba(0,0,0,.4)}
.gm-pcard .pc-pos{font-size:10.5px;font-weight:700;color:rgba(255,255,255,.9);margin-top:3px;letter-spacing:.4px}
.gm-pcard .pc-rank{font-size:9.5px;font-weight:800;color:#fff;background:rgba(0,0,0,.28);
  border-radius:20px;padding:3px 8px;height:fit-content}
.gm-pcard .pc-avatar{width:58px;height:58px;border-radius:50%;margin:10px auto 7px;display:flex;
  align-items:center;justify-content:center;font-size:17px;font-weight:800;color:#fff;
  background:radial-gradient(circle at 35% 28%, rgba(255,255,255,.4), rgba(255,255,255,.08) 65%);
  border:2.5px solid var(--team-accent);text-shadow:0 1px 3px rgba(0,0,0,.45);
  position:relative;z-index:1;overflow:hidden;
  box-shadow:0 2px 6px rgba(0,0,0,.25), inset 0 0 10px rgba(0,0,0,.15)}
.gm-pcard .pc-avatar svg{position:absolute;inset:0;width:100%;height:100%;opacity:.4;color:#fff}
.gm-pcard .pc-avatar .pc-initials{position:relative;z-index:1;letter-spacing:.5px}
.gm-pcard .pc-name{text-align:center;font-size:12.5px;font-weight:800;color:#fff;
  text-shadow:0 1px 2px rgba(0,0,0,.35);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
  position:relative;z-index:1}
.gm-pcard .pc-team{text-align:center;font-size:10px;color:rgba(255,255,255,.75);margin-bottom:9px;
  position:relative;z-index:1;letter-spacing:.3px}
.gm-pcard .pc-stats{display:flex;flex-direction:column;gap:4px;margin-bottom:9px;position:relative;z-index:1}
.gm-pcard .pc-stat-row{display:flex;align-items:center;gap:6px;font-size:9.5px;color:rgba(255,255,255,.92);
  font-weight:600}
.gm-pcard .pc-stat-label{width:28px;font-weight:800;flex-shrink:0}
.gm-pcard .pc-stat-track{flex:1;height:5px;border-radius:3px;background:rgba(0,0,0,.4);overflow:hidden;
  box-shadow:inset 0 1px 2px rgba(0,0,0,.3)}
.gm-pcard .pc-stat-fill{height:100%;background:linear-gradient(90deg,#fff,#EAF1FF);border-radius:3px;
  box-shadow:0 0 4px rgba(255,255,255,.8)}
.gm-pcard .pc-net{text-align:center;font-size:12px;font-weight:800;border-radius:8px;padding:4px 0;
  background:rgba(0,0,0,.24);color:#fff;position:relative;z-index:1}

/* 카드 바로 아래 선택 버튼 — 카드와 한 세트로 보이게 */
.st-key-pcard_section div[data-testid="stButton"] button{width:100%;font-size:11px !important;
  padding:6px 4px !important;margin-top:6px;border-radius:9px !important;font-weight:700 !important}
.st-key-pcard_section div[data-testid="stButton"] button:hover{
  border-color:var(--team-accent) !important;box-shadow:0 4px 12px rgba(0,0,0,.15) !important}
.st-key-pcard_section div[data-testid="stButton"] button:disabled{
  background:var(--navy) !important;color:#fff !important;border-color:var(--navy) !important;opacity:1 !important;
  box-shadow:inset 0 0 0 2px var(--team-accent) !important}
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

# 구단별 (주 색상, 보조 색상) — 각 구단이 실제로 쓰는 브랜드 컬러(공개된 사실 정보).
# 선택된 팀에 따라 --navy 계열 토큰을 이 색으로 갈아끼워 화면 전체에 아이덴티티를 준다.
TEAM_COLORS: dict[str, tuple[str, str]] = {
    "NYY": ("#0C2340", "#C4CED4"), "BOS": ("#BD3039", "#0C2340"), "TOR": ("#134A8E", "#1D2D5C"),
    "TBR": ("#092C5C", "#8FBCE6"), "BAL": ("#DF4601", "#000000"),
    "CLE": ("#00385D", "#E50022"), "MIN": ("#002B5C", "#D31145"), "CHW": ("#27251F", "#C4CED4"),
    "DET": ("#0C2340", "#FA4616"), "KCR": ("#004687", "#BD9B60"),
    "HOU": ("#002D62", "#EB6E1F"), "SEA": ("#0C2C56", "#005C5C"), "TEX": ("#003278", "#C0111F"),
    "LAA": ("#BA0021", "#003263"), "ATH": ("#003831", "#EFB21E"),
    "ATL": ("#13274F", "#CE1141"), "NYM": ("#002D72", "#FF5910"), "PHI": ("#E81828", "#002D72"),
    "MIA": ("#00A3E0", "#EF3340"), "WSN": ("#AB0003", "#14225A"),
    "MIL": ("#12284B", "#FFC52F"), "CHC": ("#0E3386", "#CC3433"), "STL": ("#C41E3A", "#0C2340"),
    "CIN": ("#C6011F", "#000000"), "PIT": ("#27251F", "#FDB827"),
    "LAD": ("#005A9C", "#A5ACAF"), "SDP": ("#2F241D", "#FFC425"), "SFG": ("#FD5A1E", "#27251F"),
    "ARI": ("#A71930", "#E3D4AD"), "COL": ("#33006F", "#C4CED4"),
}


def _team_theme_css(team_code: str | None) -> str:
    """선택된 팀의 브랜드 컬러로 --navy 계열 토큰을 오버라이드하는 <style> 블록."""
    primary, secondary = TEAM_COLORS.get(team_code or "", ("#16325C", "#1D3E73"))
    navy_2 = _mix_hex(primary, "#FFFFFF", 0.18)
    navy_soft = _mix_hex(primary, "#FFFFFF", 0.9)
    hero_glow = _mix_hex(primary, secondary, 0.4)
    return (
        "<style>:root{"
        f"--navy:{primary}; --navy-2:{navy_2}; --navy-soft:{navy_soft}; "
        f"--team-accent:{secondary}; --hero-glow:{hero_glow};"
        "}</style>"
    )

PAGES = [
    ("구단 상황실", "pages/1_Club_Operations_Center.py"),
    ("선수 리포트", "pages/2_Player_Report.py"),
    ("모델 정보", "pages/3_Model_Information.py"),
]


def inject_css() -> None:
    # st.markdown 은 큰 <style> 블록을 마크다운 파서가 중간에 텍스트로 흘려버리는
    # 경우가 있다. st.html() 은 마크다운 파싱을 거치지 않고 그대로 주입한다.
    team_code = st.session_state.get("team_code")
    st.html(FONT_LINK + CSS + _team_theme_css(team_code))


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


# 실제 선수 사진은 초상권 때문에 쓸 수 없어, 특정 인물을 특정하지 않는 일반 실루엣으로
# "카드에 사진이 있는" 느낌만 준다. 이니셜 배지를 그 위에 겹쳐 식별성을 준다.
_PLAYER_SILHOUETTE_SVG = (
    '<svg viewBox="0 0 100 100" preserveAspectRatio="xMidYMax slice" aria-hidden="true">'
    '<circle cx="50" cy="36" r="19" fill="currentColor"/>'
    '<path d="M12 104C12 72 28 54 50 54C72 54 88 72 88 104Z" fill="currentColor"/>'
    "</svg>"
)


def _card_tier(ovr: float) -> str:
    if ovr >= 65:
        return "gold"
    if ovr >= 50:
        return "silver"
    return "bronze"


def player_card_html(
    *,
    index: int,
    rank: int,
    ovr: float,
    position_label: str,
    name: str,
    team: str,
    stat_rows: list[tuple[str, float]],
    net_effect_pct: float,
    selected: bool = False,
) -> str:
    """FIFA 얼티밋팀 느낌의 영입 후보 카드 HTML. 실사진이 없어 이니셜 아바타로 대체한다."""
    tier = _card_tier(ovr)
    initials = "".join(part[0] for part in name.replace("-", " ").split()[:2]).upper() or "?"
    stats_html = "".join(
        f'<div class="pc-stat-row"><span class="pc-stat-label">{label}</span>'
        f'<span class="pc-stat-track"><span class="pc-stat-fill" style="width:{max(0, min(100, pct)):.0f}%"></span></span>'
        f"</div>"
        for label, pct in stat_rows
    )
    net_sign = "+" if net_effect_pct >= 0 else ""
    net_color = "var(--gain)" if net_effect_pct >= 0 else "var(--risk)"
    selected_cls = " selected" if selected else ""
    return (
        f'<div class="gm-pcard tier-{tier}{selected_cls}" style="--i:{index}">'
        '<div class="pc-top">'
        f'<div><div class="pc-ovr">{ovr:.0f}</div><div class="pc-pos">{position_label}</div></div>'
        f'<div class="pc-rank">#{rank} 추천</div>'
        "</div>"
        f'<div class="pc-avatar">{_PLAYER_SILHOUETTE_SVG}<span class="pc-initials">{initials}</span></div>'
        f'<div class="pc-name">{name}</div>'
        f'<div class="pc-team">{team}</div>'
        f'<div class="pc-stats">{stats_html}</div>'
        f'<div class="pc-net" style="color:{net_color};background:rgba(255,255,255,.9)">'
        f"{net_sign}{net_effect_pct:.1f}%p</div>"
        "</div>"
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
