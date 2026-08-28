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
.gm-kicker{font-size:13px;color:var(--muted);letter-spacing:1.6px;margin-bottom:7px;font-weight:800;
  text-transform:uppercase}
.gm-title{font-size:31px;font-weight:800;letter-spacing:-.6px;margin:0 0 6px;color:var(--ink)}
.gm-desc{font-size:15.5px;color:var(--muted);margin:0 0 8px;line-height:1.55}
.gm-section{font-size:14.5px;color:var(--ink);font-weight:800;letter-spacing:.2px;
  margin:30px 0 14px;display:flex;align-items:center;gap:8px;justify-content:space-between}
.gm-section > span:first-child{display:flex;align-items:center;gap:8px}
.gm-section > span:first-child::before{content:"";width:4px;height:14px;border-radius:2px;
  background:var(--navy);display:inline-block}
.gm-section-icon{width:17px;height:17px;color:var(--navy);flex-shrink:0}

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
.gm-ph-body b{color:var(--ink);font-size:15px}
.gm-ph-body .desc{color:var(--muted);font-size:13.5px;line-height:1.65;margin-top:4px}
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
.st-key-hero .gm-title{color:#fff;font-size:38px;text-shadow:0 1px 2px rgba(0,0,0,.15)}
.st-key-hero .gm-desc{color:rgba(255,255,255,.7);font-size:16px;margin-bottom:8px}

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
.gm-pcard.selected{box-shadow:0 0 0 3px var(--navy),0 12px 26px rgba(22,50,92,.4);transform:translateY(-4px);
  animation:gm-swap-in .55s cubic-bezier(.2,.9,.25,1.3)}
@keyframes gm-swap-in{
  0%{transform:translateY(-4px) scale(.9) rotateY(-25deg);box-shadow:0 0 0 0 rgba(22,50,92,0)}
  55%{transform:translateY(-8px) scale(1.06) rotateY(8deg)}
  100%{transform:translateY(-4px) scale(1) rotateY(0);box-shadow:0 0 0 3px var(--navy),0 12px 26px rgba(22,50,92,.4)}
}
.gm-pcard.selected::before{content:"✓ 선택됨";position:absolute;top:8px;right:8px;font-size:9.5px;
  font-weight:800;color:var(--navy);background:#fff;border-radius:20px;padding:2px 8px;z-index:3;
  box-shadow:0 1px 4px rgba(0,0,0,.25);animation:gm-badge-pop .4s cubic-bezier(.2,.9,.25,1.4) .1s both}
@keyframes gm-badge-pop{from{opacity:0;transform:scale(0)}to{opacity:1;transform:scale(1)}}
.gm-pcard.selected::after{content:"";position:absolute;inset:-3px;border-radius:18px;z-index:0;
  box-shadow:0 0 0 0 rgba(22,50,92,.5);animation:gm-ring-pulse 1s ease-out .1s}
@keyframes gm-ring-pulse{0%{box-shadow:0 0 0 0 rgba(22,50,92,.5)}100%{box-shadow:0 0 0 16px rgba(22,50,92,0)}}

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
.gm-pcard .pc-photo{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;z-index:2;
  animation:gm-photo-in .5s ease .15s both}
@keyframes gm-photo-in{from{opacity:0;transform:scale(1.15)}to{opacity:1;transform:scale(1)}}
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

/* ══ 오늘 경기 — VS 매치업 카드 ══ */
.gm-vs-grid{display:flex;flex-direction:column;gap:10px}
.gm-vs-card{position:relative;background:linear-gradient(135deg,var(--navy-2),var(--navy));
  border-radius:16px;padding:14px 18px;overflow:hidden;color:#fff;
  animation:gm-pop .5s cubic-bezier(.2,.9,.25,1.15) both;animation-delay:calc(var(--i,0) * 70ms);
  box-shadow:0 6px 16px rgba(0,0,0,.18)}
.gm-vs-card::before{content:"";position:absolute;inset:0;
  background:radial-gradient(600px 120px at var(--team-a-x,20%) -20%, var(--team-accent) 0%, transparent 60%);
  opacity:.35;pointer-events:none}
.gm-vs-row{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:10px;position:relative;z-index:1}
.gm-vs-team{display:flex;flex-direction:column;gap:2px;min-width:0}
.gm-vs-team.away{text-align:left}
.gm-vs-team.home{text-align:right;align-items:flex-end}
.gm-vs-name{font-size:14.5px;font-weight:800;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.gm-vs-tag{font-size:10px;color:rgba(255,255,255,.55);letter-spacing:.5px;text-transform:uppercase}
.gm-vs-mid{display:flex;flex-direction:column;align-items:center;gap:2px}
.gm-vs-bolt{font-size:11px;font-weight:800;color:rgba(255,255,255,.5);letter-spacing:1px}
.gm-vs-bar{width:74px;height:7px;border-radius:4px;overflow:hidden;background:rgba(255,255,255,.15);
  display:flex}
.gm-vs-bar-away{height:100%;background:linear-gradient(90deg,#9CA9C9,#DCE3F2)}
.gm-vs-bar-home{height:100%;background:linear-gradient(90deg,var(--team-accent),#fff)}
.gm-vs-winner{font-size:10px;font-weight:800;color:#FFE9A8;letter-spacing:.3px;margin-top:2px}

/* ══ 예상 순위 — 리빌 인터랙션 ══ */
.gm-standing-row{display:flex;align-items:center;gap:14px;background:var(--card);border:1px solid var(--line);
  border-radius:12px;padding:10px 16px;margin-bottom:7px;box-shadow:var(--shadow-sm);
  animation:gm-pop .45s cubic-bezier(.2,.9,.25,1.15) both;animation-delay:calc(var(--i,0) * 90ms)}
.gm-standing-row.gm-rank-1{background:linear-gradient(90deg,#FFF8E6,var(--card) 55%);
  border-color:#F0CB6B;box-shadow:0 4px 14px rgba(212,160,20,.18)}
.gm-standing-rank{width:30px;height:30px;border-radius:50%;display:flex;align-items:center;justify-content:center;
  font-size:13px;font-weight:800;color:#fff;background:var(--navy);flex-shrink:0}
.gm-standing-row.gm-rank-1 .gm-standing-rank{background:linear-gradient(140deg,#E3B438,#8A6417)}
.gm-standing-row.gm-rank-2 .gm-standing-rank{background:linear-gradient(140deg,#A6B0BC,#636E7A)}
.gm-standing-row.gm-rank-3 .gm-standing-rank{background:linear-gradient(140deg,#C99B6C,#5A3A22)}
.gm-standing-name{flex:1;font-size:14px;font-weight:700;color:var(--ink)}
.gm-standing-bar-track{width:120px;height:8px;border-radius:4px;background:var(--line);overflow:hidden}
.gm-standing-bar-fill{height:100%;background:linear-gradient(90deg,var(--navy),var(--team-accent));
  border-radius:4px;animation:gm-bar-grow .7s cubic-bezier(.2,.8,.2,1) both;animation-delay:calc(var(--i,0) * 90ms + .1s)}
@keyframes gm-bar-grow{from{width:0}}
.gm-standing-pct{width:52px;text-align:right;font-size:13px;font-weight:800;color:var(--ink);
  font-variant-numeric:tabular-nums}

/* ══ 구단 선택 — 미국 지도 위에 "둥둥 뜨는" 핀 마커 ══ */
.gm-usmap-wrap{position:relative;max-width:900px;margin:0 auto;
  filter:drop-shadow(0 20px 40px rgba(0,0,0,.35))}
.gm-usmap-wrap{border-radius:16px;overflow:hidden}
.gm-usmap-wrap svg{width:100%;height:auto;display:block;overflow:hidden}
.gm-usmap-state{fill:rgba(255,255,255,.07);stroke:rgba(255,255,255,.22);stroke-width:1;
  transition:fill .2s}
.gm-map-marker{cursor:pointer}
.gm-map-marker .pin-shadow{transform-box:fill-box;transform-origin:center;
  animation:gm-shadow-breathe 2.6s ease-in-out infinite;animation-delay:calc(var(--i,0) * .12s)}
.gm-map-marker .pin-float{transform-box:fill-box;transform-origin:center bottom;
  animation:gm-pin-bob 2.6s ease-in-out infinite;animation-delay:calc(var(--i,0) * .12s);
  transition:filter .15s ease}
@keyframes gm-pin-bob{0%,100%{transform:translateY(0)}50%{transform:translateY(-5px)}}
@keyframes gm-shadow-breathe{0%,100%{opacity:.4;transform:scaleX(1)}50%{opacity:.2;transform:scaleX(.75)}}
.gm-usmap-wrap a{text-decoration:none;outline:none}
.gm-map-marker:hover .pin-float{animation-play-state:paused;transform:translateY(-9px) scale(1.18);
  filter:drop-shadow(0 8px 10px rgba(0,0,0,.5))}
.gm-map-marker .pin-ring{transform-box:fill-box;animation:gm-map-pulse 2.4s ease-out infinite;
  animation-delay:calc(var(--i,0) * .12s);transform-origin:center bottom}
.gm-map-marker .pin-code{font-size:7.5px;font-weight:800;fill:var(--navy);text-anchor:middle;
  pointer-events:none}
.gm-map-marker .pin-name{opacity:0;transition:opacity .15s ease;pointer-events:none}
.gm-map-marker:hover .pin-name{opacity:1}
.gm-map-marker .pin-name rect{fill:rgba(10,18,32,.92)}
.gm-map-marker .pin-name text{font-size:11px;font-weight:700;fill:#fff;text-anchor:middle}
@keyframes gm-map-pulse{0%{transform:scale(1);opacity:.5}100%{transform:scale(2.6);opacity:0}}
.gm-map-marker .pop-wrap{animation:gm-pop .5s cubic-bezier(.2,.9,.25,1.2) both;animation-delay:calc(var(--i,0) * 35ms)}

/* ══ 스트림릿 기본 위젯을 앱 톤에 맞게 재구성 ══ */
div[data-testid="stRadioGroup"]{display:flex;gap:6px;background:var(--paper);padding:5px;
  border-radius:13px;border:1px solid var(--line);flex-wrap:wrap}
label[data-testid="stRadioOption"]{flex:1;min-width:fit-content;display:flex;align-items:center;
  justify-content:center;padding:9px 16px;border-radius:9px;cursor:pointer;
  transition:background .18s,box-shadow .18s;background:transparent}
label[data-testid="stRadioOption"] > div > div > div:first-child{display:none !important}
label[data-testid="stRadioOption"] [data-testid="stMarkdownContainer"] p{
  font-size:13.5px !important;font-weight:600;color:var(--muted);transition:color .18s}
label[data-testid="stRadioOption"]:has(input:checked){background:var(--navy);
  box-shadow:0 3px 10px rgba(22,50,92,.35)}
label[data-testid="stRadioOption"]:has(input:checked) [data-testid="stMarkdownContainer"] p{
  color:#fff !important;font-weight:800}
label[data-testid="stRadioOption"]:hover [data-testid="stMarkdownContainer"] p{color:var(--navy)}
.st-key-hero label[data-testid="stRadioOption"] [data-testid="stMarkdownContainer"] p{
  color:rgba(255,255,255,.75)}
.st-key-hero div[data-testid="stRadioGroup"]{background:rgba(255,255,255,.08);border-color:rgba(255,255,255,.14)}

div[data-testid="stSelectbox"] .react-aria-ComboBox > div{
  border-radius:13px !important;border:1.5px solid var(--line) !important;
  box-shadow:var(--shadow-sm) !important;transition:border-color .15s,box-shadow .15s !important}
div[data-testid="stSelectbox"] .react-aria-ComboBox:focus-within > div,
div[data-testid="stSelectbox"] .react-aria-ComboBox > div:hover{
  border-color:var(--navy) !important;box-shadow:var(--shadow-md) !important}
div[data-testid="stSelectbox"] .react-aria-ComboBox input{
  font-weight:700 !important;font-size:14.5px !important;color:var(--ink) !important}

div[data-testid="stElementToolbar"]{display:none !important}
div[data-testid="stDataFrame"]{border-radius:14px !important;box-shadow:var(--shadow-md) !important}

label[data-testid="stWidgetLabel"] p{font-size:13.5px !important;font-weight:700 !important;
  color:var(--ink) !important;margin-bottom:2px !important}
.st-key-hero label[data-testid="stWidgetLabel"] p{color:rgba(255,255,255,.8) !important}
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


# 섹션 헤더용 소형 라인 아이콘 (Feather 스타일, 16x16 로컬 좌표) — SVG로
# 통일해 스트림릿 기본 이모지/텍스트 느낌을 줄인다.
SECTION_ICONS: dict[str, str] = {
    "trophy": '<path d="M6 2h12v3a6 6 0 0 1-12 0z"/><path d="M6 3H3a3 3 0 0 0 3 5"/>'
              '<path d="M18 3h3a3 3 0 0 1-3 5"/><path d="M12 14v4"/><path d="M8 22h8"/>'
              '<path d="M10 18h4v4h-4z"/>',
    "chart": '<path d="M4 20V10"/><path d="M11 20V4"/><path d="M18 20v-7"/>',
    "team": '<circle cx="9" cy="7" r="3.2"/><path d="M2.5 20a6.5 6.5 0 0 1 13 0"/>'
            '<circle cx="17.5" cy="8" r="2.6"/><path d="M15 12.5a5.5 5.5 0 0 1 6.5 5.4"/>',
    "target": '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="4.5"/><circle cx="12" cy="12" r=".8" fill="currentColor"/>',
    "shield": '<path d="M12 2 4 5v6c0 5.2 3.4 8.7 8 9 4.6-.3 8-3.8 8-9V5z"/>',
    "calendar": '<rect x="3.5" y="5" width="17" height="16" rx="2.5"/><path d="M3.5 10h17"/>'
                '<path d="M8 3v4"/><path d="M16 3v4"/>',
    "swap": '<path d="M4 8h13"/><path d="M13 4l4 4-4 4"/><path d="M20 16H7"/><path d="M11 12l-4 4 4 4"/>',
}


def _section_icon_svg(icon: str) -> str:
    body = SECTION_ICONS.get(icon, "")
    if not body:
        return ""
    return (
        '<svg class="gm-section-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        f"{body}</svg>"
    )


def section(label: str, sublabel: str = "", icon: str | None = None) -> None:
    icon_html = _section_icon_svg(icon) if icon else ""
    st.markdown(
        f'<div class="gm-section"><span>{icon_html}{label}</span>'
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
    photo_url: str | None = None,
) -> str:
    """FIFA 얼티밋팀 느낌의 영입 후보 카드 HTML.

    photo_url이 있으면 MLB 공식 헤드샷을 씌운다(핫링크, 저장 안 함). 로딩
    실패하면 onerror로 자기 자신을 숨겨서 밑에 깔린 실루엣+이니셜이 그대로
    드러난다 — 깨진 이미지 아이콘 대신 항상 뭔가는 보이게.
    """
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
    photo_html = (
        f'<img class="pc-photo" src="{photo_url}" alt="" loading="lazy" '
        'onerror="this.style.display=\'none\'"/>'
        if photo_url else ""
    )
    return (
        f'<div class="gm-pcard tier-{tier}{selected_cls}" style="--i:{index}">'
        '<div class="pc-top">'
        f'<div><div class="pc-ovr">{ovr:.0f}</div><div class="pc-pos">{position_label}</div></div>'
        f'<div class="pc-rank">#{rank} 추천</div>'
        "</div>"
        f'<div class="pc-avatar">{_PLAYER_SILHOUETTE_SVG}<span class="pc-initials">{initials}</span>{photo_html}</div>'
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


# ── 구단 선택 화면의 미국 지도 (Wikimedia Commons "Blank US Map", public-domain
# 스타일 템플릿의 주(州) 외곽선을 그대로 사용 + 떠 있는 핀 마커 오버레이) ──
_US_MAP_SVG = """<div class="gm-usmap-wrap"><svg viewBox="-10 -35 979 638" xmlns="http://www.w3.org/2000/svg"><path class="gm-usmap-state" d="m 643,467.4 .4,-7.3 -.9,-1.2 -1.7,-.7 -2.5,-2.8 .5,-2.9 48.8,-5.1 -.7,-2.2 -1.5,-1.5 -.5,-1.4 .6,-6.3 -2.4,-5.7 .5,-2.6 .3,-3.7 2.2,-3.8 -.2,-1.1 -1.7,-1 v -3.2 l -1.8,-1.9 -2.9,-6.1 -12.9,-45.8 -45.7,4 1.3,2 -1.3,67 4.4,33.2 .9,-.5 1.3,.1 .6,.4 .8,-.1 2,-3.8 v -2.3 l 1.1,-1.1 1.4,.5 3.4,6.4 v .9 l -3.3,2.2 3.5,-.4 4.9,-1.6 z"/><path class="gm-usmap-state" d="m 139.6,387.6 3,-2.2 .8,-2.4 -1,-1.6 -1.8,-.2 -1.1,-1.6 1.1,-6.9 1.6,-.3 2.4,-3.2 1.6,-7 2.4,-3.6 4.8,-1.7 1.3,-1.3 -.4,-1.9 -2.3,-2.5 -1.2,-5.8 -1.4,-1.8 -1.3,-3.4 .9,-2.1 1.4,-3 .5,-2.9 -.5,-4.9 1,-13.6 3.5,-.6 3.7,1.4 1.2,2.7 h 2 l 2.4,-2.9 3.4,-17.5 46.2,8.2 40,6 -17.4,124.1 -37.3,-5.4 -64.2,-37.5 .5,-2.9 2,-1.8 z"/><path class="gm-usmap-state" d="m 584.2,367 .9,-2.2 1.2,.5 .7,-1 -.8,-.7 .3,-1.5 -1.1,-.9 .6,-1 -.1,-1.5 -1.1,-.1 .8,-.8 1.3,.8 .3,-1.4 -.4,-1.1 .1,-.7 2,.6 -.4,-1.5 1.6,-1.3 -.5,-.9 -1.1,.1 -.6,-.9 .9,-.9 1.6,-.2 .5,-.8 1.4,-.2 -.1,-.8 -.9,-.9 v -.5 h 1.5 l .4,-.7 -1.4,-1 -.1,-.6 -11.2,.8 2.8,-5.1 1.7,-1.5 v -2.2 l -1.6,-2.5 -39.8,2 -39.1,.7 4.1,24.4 -.7,39 2.6,2.3 2.8,-1.3 3.2,.8 .2,11.9 52.3,-1.3 1.2,-1.5 .5,-3 -1.5,-2.3 -.5,-2.2 .9,-.7 v -.8 l -1.7,-1.1 -.1,-.7 1.6,-.9 -1.2,-1.1 1.7,-7.1 3.4,-1.6 v -.8 l -1.1,-1.4 2.9,-5.4 h 1.9 l 1.5,-1.2 -.3,-5.2 3.1,-4.5 1.8,-.6 -.5,-3.1 z"/><path class="gm-usmap-state" d="m 69.4,365.6 3.4,5.2 -1.4,.1 -1.8,-1.9 z m 1.9,-9.8 1.8,4.1 2.6,1 .7,-.6 -1.3,-2.5 -2.6,-2.4 z m -19.9,-19 v 2.4 l 2,1.2 4.4,-.2 1,-1 -3.1,-.2 z m -5.9,.1 3.3,.5 1.4,2.2 h -3.8 z m 47.9,45.5 -1,-3 .2,-3 -.4,-7.9 -1.8,-4.8 -1.2,-1.4 -.6,-1.5 -7,-8.6 -3.6,.1 -2,-1.9 1.1,-1.8 -.7,-3.7 -2.2,-1.2 -3.9,-.6 -2.8,-1.3 -1.5,-1.9 -4.5,-6.6 -2.7,-2.2 -3.7,-.5 -3.1,-2.3 -4.7,-1.5 -2.8,-.3 -2.5,-2.5 .2,-2.8 .8,-4.8 1.8,-5.1 -1.4,-1.6 -4,-9.4 -2.7,-3.7 -.4,-3 -1.6,-2.3 .2,-2.5 -2,-5 -2.9,-2.7 .6,-7.1 2.4,-.8 1.8,-3.1 -.4,-3.2 -1,-.9 h -2.5 l -2.5,-3.3 -1.5,-3.5 v -7.5 l 1.2,-4.2 .2,-2.1 2.5,.2 -.1,1.6 -.8,.7 v 2.5 l 3.7,3.2 v -4.7 l -1.4,-3.4 .5,-1.1 -1,-1.7 2.8,-1.5 -1.9,-3 -1.4,.5 -1.5,3.8 .5,1.3 -.8,1 -.9,-.1 -5.4,-6.1 .7,-5.6 -1.1,-3.9 -6.5,-12.8 .8,-10.7 2.3,-3.6 .2,-6.4 -5.5,-11.1 .3,-5.2 6.9,-7.5 1.7,-2.4 -.1,-1.4 4,-9.2 .1,-8.4 .9,-2.5 66.1,18.6 -16.4,63.1 1.1,3.5 70.4,105 -.9,2.1 1.3,3.4 1.4,1.8 1.2,5.8 2.3,2.5 .4,1.9 -1.3,1.3 -4.8,1.7 -2.4,3.6 -1.6,7 -2.4,3.2 -1.6,.3 -1.1,6.9 1.1,1.6 1.8,.2 1,1.6 -.8,2.4 -3,2.2 -2.2,-.1 z"/><path class="gm-usmap-state" d="m 374.6,323.3 -16.5,-1 -51.7,-4.8 -52.6,-6.5 11.5,-88.3 44.9,5.7 37.5,3.4 33.1,2.4 -1.4,22.1 z"/><path class="gm-usmap-state" d="m 873.5,178.9 .4,-1.1 -3.2,-12.3 -.1,-.3 -14.9,3.4 v .7 l -.9,.3 -.5,-.7 -10.5,2.4 2.8,16.3 1.8,1.5 -3.5,3.4 1.7,2.2 5.4,-4.5 1.7,-1.3 h .8 l 2.4,-3.1 1.4,.1 2.9,-1.1 h 2.1 l 5.3,-2.7 2.8,-.9 1,-1 1.5,.5 z"/><path class="gm-usmap-state" d="m 822.2,226.6 -1.6,.3 -1.5,1.1 -1.2,2.1 7.6,27.1 10.9,-2.3 -2.2,-7.6 -1.1,.5 -3.3,-2.6 -.5,-1.7 -1.8,-1 -.2,-3.7 -2.1,-2.2 -1.1,-.8 -1.2,-1.1 -.4,-3.2 .3,-2.1 1,-2.2 z"/><path class="gm-usmap-state" d="m 751.7,445.1 -4,-.7 -1.7,-.9 -2.2,1.4 v 2.5 l 1.4,2.1 -.5,4.3 -2.1,.6 -1,-1.1 -.6,-3.2 -50.1,3.3 -3.3,-6 -48.8,5.1 -.5,2.9 2.5,2.8 1.7,.7 .9,1.2 -.4,7.3 -1.1,.6 .5,.4 1,-.3 .7,-.8 10.5,-2.7 9.2,-.5 8.1,1.9 8.5,5 2.4,.8 2.2,2 -.1,2.7 h 2.4 l 1.9,-1 2.5,.1 2,-.8 2.9,-2 3.1,-2.9 1.1,-.4 .6,.5 h 1.4 l .5,-.8 -.5,-1.2 -.6,-.6 .2,-.8 2,-1.1 5,-.4 .8,1 1,.1 2.3,1 3,1.8 1.2,1.7 1.1,1.2 2.8,1.4 v 2.4 l 2.8,1.9 1,.1 1.6,1.4 .7,1.6 1,.2 .8,2.1 .7,.6 1,-1.1 2.9,.1 .5,1.4 1.1,.9 v 1.3 l 2.9,2.2 .2,9.6 -1.8,5.8 1,1.2 -.2,3.4 -.8,1.4 .7,1.2 2.3,2.3 .3,1.5 .8,1 -.4,-1.9 1.3,-.6 .8,-3.6 -3,-1.2 .1,-.6 2.6,-.4 .9,2.6 1.1,.6 .1,-2 1.1,.3 .6,.8 -.1,.7 -2.9,4.2 -.2,1.1 -1.7,1.9 v 1.1 l 3.7,3.8 5.3,7.9 1.8,2.1 v 1.8 l 2.8,4.6 2.3,.6 .7,-1.2 -2.1,.3 -3,-4.5 .2,-1.4 1.5,-.8 v -1.5 l -.6,-1.3 .9,-.9 .4,.9 .7,.5 v 4 l -1.2,-.6 -.8,.9 1.4,1.6 1,2.6 1.2,-.6 2.3,1.2 2.1,2.2 1.6,5.1 3.1,4.8 .8,-1.3 2.8,-.5 3.2,1.3 .3,1.7 3.3,3.8 .1,1.1 2.2,2.7 -.7,.5 v 2.7 l 2.7,1.4 h 1.5 l 2.7,-1.8 1.5,.3 1.1,.4 2.3,-1.7 .2,-.7 1.2,.3 2.4,-1.7 1.3,-2.3 -.7,-3.2 -.2,-1.3 1.1,-4 .6,-.2 .6,1.6 .8,-1.8 -.8,-7.2 -.4,-10.5 -1,-6.8 -.7,-1.7 -6.6,-11.1 -5.2,-9.1 -2.2,-3.3 -1.3,-3.6 -.2,-3.4 .9,-.3 v -.9 l -1.1,-2.2 -4,-4 -7.6,-9.7 -5.7,-10.4 -4.3,-10.7 -.6,-3.7 -1.2,-1 -.5,-3.8 z m 9.2,134.5 1.7,-.1 -.7,-1 z m 7.3,-1.1 v -.7 l 1.6,-.2 3.7,-3.3 1.5,-.6 2.4,-.9 .3,1.3 1.7,.8 -2.6,1.2 h -2.4 l -3.9,2.5 z m 17.2,-7.6 -3,1.4 -1,1.3 1.1,.1 z m 3.8,-2.9 -1.1,.3 -1.4,2 1.1,-.2 1.5,-1.6 z m 8.3,-15.7 -1.7,5.6 -.8,1 -1,2.6 -1.2,1.6 -.7,1.7 -1.9,2.2 v .9 l 2.7,-2.8 2.4,-3.5 .6,-2 2.1,-4.9 z"/><path class="gm-usmap-state" d="m 761.8,414.1 v 1.4 l -4.2,6.2 -1.2,.2 1.5,.5 v 2 l -.9,1.1 -.6,6 -2.3,6.2 .5,2 .7,5.1 -3.6,.3 -4,-.7 -1.7,-.9 -2.2,1.4 v 2.5 l 1.4,2.1 -.5,4.3 -2.1,.6 -1,-1.1 -.6,-3.2 -50.1,3.3 -3.3,-6 -.7,-2.2 -1.5,-1.5 -.5,-1.4 .6,-6.3 -2.4,-5.7 .5,-2.6 .3,-3.7 2.2,-3.8 -.2,-1.1 -1.7,-1 v -3.2 l -1.8,-1.9 -2.9,-6.1 -12.9,-45.8 22.9,-2.9 21.4,-3 -.1,1.9 -1.9,1 -1.4,3.2 .2,1.3 6.1,3.8 2.6,-.3 3.1,4 .4,1.7 4.2,5.1 2.6,1.7 1.4,.2 2.2,1.6 1.1,2.2 2,1.6 1.8,.5 2.7,2.7 .1,1.4 2.6,2.8 5,2.3 3.6,6.7 .3,2.7 3.9,2.1 2.5,4.8 .8,3.1 4.2,.4 z"/><path class="gm-usmap-state" d="m 165.3,183.1 -24.4,-5.4 8.5,-37.3 2.9,-5.8 .4,-2.1 .8,-.9 -.9,-2 -2.9,-1.2 .2,-4.2 4,-5.8 2.5,-.8 1.6,-2.3 -.1,-1.6 1.8,-1.6 3.2,-5.5 4.2,-4.8 -.5,-3.2 -3.5,-3.1 -1.6,-3.6 1.1,-4.3 -.7,-4 12.7,-56.1 14.2,3 -4.8,22 3.7,7.4 -1.6,4.8 3.6,4.8 1.9,.7 3.9,8.3 v 2.1 l 2.3,3 h .9 l 1.4,2.1 h 3.2 v 1.6 l -7.1,17 -.5,4.1 1.4,.5 1.6,2.6 2.8,-1.4 3.6,-2.4 1.9,1.9 .5,2.5 -.5,3.2 2.5,9.7 2.6,3.5 2.3,1.4 .4,3 v 4.1 l 2.3,2.3 1.6,-2.3 6.9,1.6 2.1,-1.2 9,1.7 2.8,-3.3 1.8,-.6 1.2,1.8 1.6,4.1 .9,.1 -8.5,54.8 -47.9,-8.2 z"/><path class="gm-usmap-state" d="m 623.5,265.9 -1,5.2 v 2 l 2.4,3.5 v .7 l -.3,.9 .9,1.9 -.3,2.4 -1.6,1.8 -1.3,4.2 -3.8,5.3 -.1,7 h -1 l .9,1.9 v .9 l -2.2,2.7 .1,1.1 1.5,2.2 -.1,.9 -3.7,.6 -.6,1.2 -1.2,-.6 -1,.5 -.4,3.3 1.7,1.8 -.4,2.4 -1.5,.3 -6.9,-3 -4,3.7 .3,1.8 h -2.8 l -1.4,-1.5 -1.8,-3.8 v -1.9 l .8,-.6 .1,-1.3 -1.7,-1.9 -.9,-2.5 -2.7,-4.1 -4.8,-1.3 -7.4,-7.1 -.4,-2.4 2.8,-7.6 -.4,-1.9 1.2,-1.1 v -1.3 l -2.8,-1.5 -3,-.7 -3.4,1.2 -1.3,-2.3 .6,-1.9 -.7,-2.4 -8.6,-8.4 -2.2,-1.5 -2.5,-5.9 -1.2,-5.4 1.4,-3.7 .7,-.7 .1,-2.3 -.7,-.9 1,-1.5 1.8,-.6 .9,-.3 1,-1.2 v -2.4 l 1.7,-2.4 .5,-.5 .1,-3.5 -.9,-1.4 -1,-.3 -1.1,-1.6 1,-4 3,-.8 h 2.4 l 4.2,-1.8 1.7,-2.2 .1,-2.4 1.1,-1.3 1.3,-3.2 -.1,-2.6 -2.8,-3.5 h -1.2 l -.9,-1.1 .2,-1.6 -1.7,-1.7 -2.5,-1.3 .5,-.6 45.9,-2.8 .1,4.6 3.4,4.6 1.2,4.1 1.6,3.2 z"/><path class="gm-usmap-state" d="m 629.2,214.8 -5.1,2.3 -4.7,-1.4 4.1,50.2 -1,5.2 v 2 l 2.4,3.5 v .7 l -.3,.9 .9,1.9 -.3,2.4 -1.6,1.8 -1.3,4.2 -3.8,5.3 -.1,7 h -1 l .9,1.9 1.1,.8 .6,-1 -.7,-1.7 4.6,-.5 .2,1.2 1.1,.2 .4,-.9 -.6,-1.3 .3,-.8 1.3,.8 1.7,-.4 1.7,.6 3.4,2.1 1.8,-2.8 3.5,-2.2 3,3.3 1.6,-2.1 .3,-2.7 3.8,-2.3 .2,1.3 1.9,1.2 3,-.2 1.2,-.7 .1,-3.4 2.5,-3.7 4.6,-4.4 -.1,-1.7 1.2,-3.8 2.2,1 6.7,-4.5 -.4,-1.7 -1.5,-2.1 1,-1.9 -6.6,-57.2 -.1,-1.4 -32.4,3.4 z"/><path class="gm-usmap-state" d="m 556.9,183 2.1,1.6 .6,1.1 -1.6,3.3 -.1,2.5 2,5.5 2.7,1.5 3.3,.7 1.3,2.8 -.5,.6 2.5,1.3 1.7,1.7 -.2,1.6 .9,1.1 h 1.2 l 2.8,3.5 .1,2.6 -1.3,3.2 -1.1,1.3 -.1,2.4 -1.7,2.2 -4.2,1.8 h -2.4 l -3,.8 -1,4 1.1,1.6 1,.3 .9,1.4 -.1,3.5 -.5,.5 -1.7,2.4 v 2.4 l -1,1.2 -.9,.3 -1.8,.6 -1,1.5 .7,.9 -.1,2.3 -.7,.7 -1.5,-.8 -1.1,-1.1 -.6,-1.6 -1.7,-1.3 -14.3,.8 -27.2,1.2 -25.9,-.1 -1.8,-4.4 .7,-2.2 -.8,-3.3 .2,-2.9 -1.3,-.7 -.4,-6.1 -2.8,-5 -.2,-3.7 -2.2,-4.3 -1.3,-3.7 v -1.4 l -.6,-1.7 v -2.3 l -.5,-.9 -.7,-1.7 -.3,-1.3 -1.3,-1.2 1,-4.3 1.7,-5.1 -.7,-2 -1.3,-.4 -.4,-1.6 1,-.5 .1,-1.1 -1.3,-1.5 .1,-1.6 2.2,.1 h 28.2 l 36.3,-.9 18.6,-.7 z"/><path class="gm-usmap-state" d="m 459.1,259.5 -43.7,-1.2 -36,-2 -4.8,67 67.7,2.9 62,.1 -.5,-48.1 -3.2,-.7 -2.6,-4.7 -2.5,-2.5 .5,-2.3 2.7,-2.6 .1,-1.2 -1.5,-2.1 -.9,1 -2,-.6 -2.9,-3 z"/><path class="gm-usmap-state" d="m 692.1,322.5 -20.5,1.4 -5.2,.8 -17.4,1 -2.6,.8 -22.6,2 -.7,-.6 h -3.7 l 1.2,3.2 -.6,.9 -23.3,1.5 1,-2.7 1.4,.9 .7,-.4 1.2,-4.1 -1,-1 1,-2 .2,-.9 -1.3,-.8 -.3,-1.8 4,-3.7 6.9,3 1.5,-.3 .4,-2.4 -1.7,-1.8 .4,-3.3 1,-.5 1.2,.6 .6,-1.2 3.7,-.6 .1,-.9 -1.5,-2.2 -.1,-1.1 2.2,-2.7 0,-.9 1.1,.8 .6,-1 -.7,-1.7 4.6,-.5 .2,1.2 1.1,.2 .4,-.9 -.6,-1.3 .3,-.8 1.3,.8 1.7,-.4 1.7,.6 3.4,2.1 1.8,-2.8 3.5,-2.2 3,3.3 1.6,-2.1 .3,-2.7 3.8,-2.3 .2,1.3 1.9,1.2 3,-.2 1.2,-.7 .1,-3.4 2.5,-3.7 4.6,-4.4 -.1,-1.7 1.2,-3.8 2.2,1 6.7,-4.5 -.4,-1.7 -1.5,-2.1 1,-1.9 1.3,.5 2.2,.1 1.9,-.8 2.9,1.2 2.2,3.4 v 1 l 4.1,.7 2.3,-.2 1.9,2.1 2.2,.2 v -1 l 1.9,-.8 3,.8 1.2,.8 1.3,-.7 h .9 l .6,-1.7 3.4,-1.8 .5,.8 .8,2.9 3.5,1.4 1.2,2.1 -.1,1.1 .6,1 -.6,3.6 1.9,1.6 .8,1.1 1,.6 -.1,.9 4.4,5.6 h 1.4 l 1.5,1.8 1.2,.3 1.4,-.1 -4.9,6.6 -2.9,1 -3,3 -.4,2.2 -2.1,1.3 -.1,1.7 -1.4,1.4 -1.8,.5 -.5,1.9 -1,.4 -6.9,4.2 z m -98,11.3 -.7,-.7 .2,-1 h 1.1 l .7,.7 -.3,1 z"/><path class="gm-usmap-state" d="m 602.5,472.8 -1.2,-1.8 .3,-1.3 -4.8,-6.8 .9,-4.6 1,-1.4 .1,-1.4 -36,2 1.7,-11.9 2.4,-4.8 6,-8.4 -1.8,-2.5 h 2 v -3.3 l -2.4,-2.5 .5,-1.7 -1.2,-1 -1.6,-7.1 .6,-1.4 -52.3,1.3 .5,19.9 .7,3.4 2.6,2.8 .7,5.4 3.8,4.6 .8,4.3 h 1 l -.1,7.3 -3.3,6.4 1.3,2.3 -1.3,1.5 .7,3 -.1,4.3 -2.2,3.5 -.1,.8 -1.7,1.2 1,1.8 1.2,1.1 1.6,-1.3 5.3,-.9 6.1,-.1 9.6,3.8 8,1 1.5,-1.4 1.8,-.2 4.8,2.2 1.6,-.4 1.1,-1.5 -4.2,-1.8 -2.2,1 -1.1,-.2 -1.4,-2 3.3,-2.2 1.6,-.1 v 1.7 l 1.5,-.1 3.4,-.3 .4,2.3 1.1,.4 .6,1.9 4.8,1 1.7,1.6 v .7 h -1.2 l -1.5,1.7 1.7,1.2 5.4,1 2.7,2.8 4.4,-1 -3.7,.2 -.1,-.6 2.8,-.7 .2,-1.8 1.2,-.3 v -1.4 l 1.1,.1 v 1.6 l 2.5,.1 .8,-1.9 .9,.3 .2,2.5 1.2,.2 -1.8,2 2.6,-.9 2,-1.1 2.9,-3.3 h -.7 l -1.3,1.2 -.4,-.1 -.5,-.8 .9,-1.2 v -2.3 l 1.1,-.8 .7,.7 1,-.8 1,-.1 .6,1.3 -.6,1.9 h 2.4 l 5.1,1.7 .5,1.3 1.6,1.4 2.8,.1 1.3,.7 1.8,-1 .9,-1.7 v -1.7 h -1.4 l -1.2,-1.4 -1.1,-1.1 -3.2,-.9 -2.6,.2 -4.2,-2.4 v -2.3 l 1.3,-1 2.4,.6 -3.1,-1.6 .2,-.8 h 3.6 l 2.6,-3.5 -2.6,-1.8 .8,-1.5 -1.2,-.8 h -.8 l -2,2.1 v 2.1 l -.6,.7 -1.1,-.1 -1.6,-1.4 h -1.3 v -1.5 l .6,-.7 .8,.7 1.7,-1.6 .7,-1.6 .8,-.3 z m -10.3,-2.7 1.9,1 .8,1.1 2.5,.1 1.5,.8 .2,1.4 -.4,.6 -.9,-1.5 -1.4,1.2 -.9,1.4 -2.8,.8 -1.6,.1 -3.7,-1 .1,-1.7 2,-2 1.1,-2.4 z m -4.7,1.2 v 1.1 l -1.8,2 h -1.2 v -2.2 l 1.6,-1.5 z"/><path class="gm-usmap-state" d="m 875,128.7 .6,4 3.2,2 .8,2.2 2.3,1.4 1.4,-.3 1,-3 -.8,-2.9 1.6,-.9 .5,-2.8 -.6,-1.3 3.3,-1.9 -2.2,-2.3 .9,-2.4 1.4,-2.2 .5,3.2 1.6,-2 1.3,.9 1.2,-.8 v -1.7 l 3.2,-1.3 .3,-2.9 2.5,-.2 2.7,-3.7 v -.7 l -.9,-.5 -.1,-3.3 .6,-1.1 .2,1.6 1,-.5 -.2,-3.2 -.9,.3 -.1,1.2 -1.2,-1.4 .9,-1.4 .6,.1 1.1,-.4 .5,2.8 2,-.3 2.9,.7 v -1 l -1.1,-1.2 1.3,.1 .1,-2.3 .6,.8 .3,1.9 2.1,1.5 .2,-1 .9,-.2 -.3,-.8 .8,-.6 -.1,-1.6 -1.6,-.2 -2,.7 1.4,-1.6 .7,-.8 1.3,-.2 .4,1.3 1.7,1.6 .4,-2.1 2.3,-1.2 -.9,-1.3 .1,-1.7 1.1,.5 h .7 l 1.7,-1.4 .4,-2.3 2.2,.3 .1,-.7 .2,-1.6 .5,1.4 1.5,-1 2.3,-4.1 -.1,-2.2 -1.4,-2 -3,-3.2 h -1.9 l -.8,2.2 -2.9,-3 .3,-.8 v -1.5 l -1.6,-4.5 -.8,-.2 -.7,.4 h -4.8 l -.3,-3.6 -8.1,-26 -7.3,-3.7 -2.9,-.1 -6.7,6.6 -2.7,-1 -1,-3.9 h -2.7 l -6.9,19.5 .7,6.2 -1.7,2.4 -.4,4.6 1.3,3.7 .8,.2 v 1.6 l -1.6,4.5 -1.5,1.4 -1.3,2.2 -.4,7.8 -2.4,-1 -1.5,.4 z m 34.6,-24.7 -1,.8 v 1.3 l .7,-.8 .9,.8 .4,-.5 1.1,.2 -1,-.8 .4,-.8 z m -1.7,2.6 -1,1.1 .5,.4 -.1,1 h 1.1 v -1.8 z m -3,-1.6 .9,1.3 1,.5 .3,-1 v -1.8 l -1.3,-.7 -.4,1.2 z m -1,5 -1.7,-1.7 1.6,-2.4 .8,.3 .2,1.1 1,.8 v 1.1 l -1,1 z"/><path class="gm-usmap-state" d="m 822.9,269.3 0,-1.7 h -.8 l 0,1.8 z m 11.8,-3.9 1.2,-2.2 .1,-2.5 -.6,-.6 -.7,.9 -.2,2.1 -.8,1.4 -.3,1.1 -4.6,1.6 -.7,.8 -1.3,.2 -.4,.9 -1.3,.6 -.3,-2.5 .4,-.7 -.8,-.5 .2,-1.5 -1.6,1 v -2 l 1.2,-.3 -1.9,-.4 -.7,-.8 .4,-1.3 -.8,-.6 -.7,1.6 .5,.8 -.7,.6 -1.1,.5 -2,-1 -.2,-1.2 -1,-1.1 -1.4,-1.7 1.5,-.8 -1,-.6 v -.9 l .6,-1 1.7,-.3 -1.4,-.6 -.1,-.7 -1.3,-.1 -.4,1.1 -.6,.3 .1,-3.4 1,-1 .8,.7 .1,-1.6 -1,-.9 -.9,1.1 -1,1.4 -.6,-1 .2,-2.4 .9,-1 .9,.9 1.2,-.7 -.4,-1.7 -1,1 -.9,-2.1 -.2,-1.7 1.1,-2.4 1.1,-1.4 1.4,-.2 -.5,-.8 .5,-.6 -.3,-.7 .2,-2.1 -1.5,.4 -.8,1.1 1,1.3 -2.6,3.6 -.9,-.4 -.7,.9 -.6,2.2 -1.8,.5 1.3,.6 1.3,1.3 -.2,.7 .9,1.2 -1.1,1 .5,.3 -.5,1.3 v 2.1 l -.5,1.3 .9,1.1 .7,3.4 1.3,1.4 1.6,1.4 .4,2.8 1.6,2 .4,1.4 v 1 h -.7 l -1.5,-1.2 -.4,.2 -1.2,-.2 -1.7,-1.4 -1.4,-.3 -1,.5 -1.2,-.3 -.4,.2 -1.7,-.8 -1,-1 -1,-1.3 -.6,-.2 -.8,.7 -1.6,1.3 -1.1,-.8 -.4,-2.3 .8,-2.1 -.3,-.5 .3,-.4 -.7,-1 1,-.1 1,-.9 .4,-1.8 1.7,-2.6 -2.6,-1.8 -1,1.7 -.6,-.6 h -1 l -.6,-.1 -.4,-.4 .1,-.5 -1.7,-.6 -.8,.3 -1.2,-.1 -.7,-.7 -.5,-.2 -.2,-.7 .6,-.8 v -.9 l -1.2,-.2 -1,-.9 -.9,.1 -1.6,-.3 -.9,-.4 .2,-1.6 -1,-.5 -.2,-.7 h -.7 l -.8,-1.2 .2,-1 -2.6,.4 -2.2,-1.6 -1.4,.3 -.9,1.4 h -1.3 l -1.7,2.9 -3.3,.4 -1.9,-1 -2.6,3.8 -2.2,-.3 -3.1,3.9 -.9,1.6 -1.8,1.6 -1.7,-11.4 60.5,-11.8 7.6,27.1 10.9,-2.3 0,5.3 -.1,3.1 -1,1.8 z m -13.4,-1.8 -1.3,.9 .8,1.8 1.7,.8 -.4,-1.6 z"/><path class="gm-usmap-state" d="m 899.9,174.2 h 3.4 l .9,-.6 .1,-1.3 -1.9,-1.8 .4,1 -1.5,1.5 h -2.3 l .1,.8 z m -9,1.8 -1.2,-.6 1,-.8 .6,-2.1 1.2,-1 .8,-.2 .6,.9 1.1,.2 .6,-.6 .5,1.9 -1.3,.3 -2.8,.7 z m -34.9,-23.4 18.4,-3.8 1,-1.5 .3,-1.7 1.9,-.6 .5,-1.1 1.7,-1.1 1.3,.3 1.7,3.3 1,.4 1.1,-1.3 .8,1.3 v 1.1 l -3,2.4 .2,.8 -.9,1 .4,.8 -1.3,.3 .9,1.2 -.8,.7 .6,1 .9,-.2 .3,-.8 1.1,.6 h 1.8 l 2.5,2.6 .2,2.6 1.8,.1 .8,1.1 .6,2 1,.7 h 1.9 l 1.9,-.1 .8,-.9 1.6,-1.2 1.1,-.3 -1.2,-2.1 -.3,.9 -1.5,-3.6 h -.8 l -.4,.9 -1.2,-1 1.3,-1.1 1.8,.4 2.3,2.1 1.3,2.7 1.2,3.3 -1,2.8 v -1.8 l -.7,-1 -3.5,2.3 -.9,-.3 -1.6,1 -.1,1.2 -2.2,1.2 -2,2.1 -2,1.9 h -1.2 l 3.3,-3.3 .5,-1.9 -.5,-.6 -.3,-1.3 -.9,-.1 -.1,1.3 -1,1.2 h -1.2 l -.3,1.1 .4,1.2 -1.2,1.1 -1.1,-.2 -.4,1 -1.4,-3 -1.3,-1.1 -2.6,-1.3 -.6,-2.2 h -.8 l -.7,-2.6 -6.5,2 -.1,-.3 -14.9,3.4 v .7 l -.9,.3 -.5,-.7 -10.5,2.4 -.7,-1 .5,-15 z"/><path class="gm-usmap-state" d="m 663.3,209.8 .1,1.4 21.4,-3.5 .5,-1.2 3.9,-5.9 v -4.3 l .8,-2.1 2.2,-.8 2,-7.8 1,-.5 1,.6 -.2,.6 -1.1,.8 .3,.9 .8,.4 1.9,-1.4 .4,-9.8 -1.6,-2.3 -1.2,-3.7 v -2.5 l -2.3,-4.4 v -1.8 l -1.2,-3.3 -2.3,-3 -2.9,-1 -4.8,3 -2.5,4.6 -.2,.9 -3,3.5 -1.5,-.2 -2.9,-2.8 -.1,-3.4 1.5,-1.9 2,-.2 1.2,-1.7 .2,-4 .8,-.8 1.1,-.1 .9,-1.7 -.2,-9.6 -.3,-1.3 -1.2,-1.2 -1.7,-1 -.1,-1.8 .7,-.6 1.8,.8 -.3,-1.7 -1.9,-2.7 -.7,-1.6 -1.1,-1.1 h -2.2 l -8.1,-2.9 -1.4,-1.7 -3.1,-.3 -1.2,.3 -4.4,-2.3 h -1.4 l .5,1 -2.7,-.1 .1,.6 .6,.6 -2.5,2.1 .1,1.8 1.5,2.3 1.5,.2 v .6 l -1.5,.5 -2.1,-.1 -2.8,2.5 .1,2.5 .4,5.8 -2.2,3.4 .8,-4.5 -.8,-.6 -.9,5.3 -1,-2.3 .5,-2.3 -.5,-1 .6,-1.3 -.6,-1.1 1,-1 v -1.2 l -1.3,.6 -1.3,3.1 -.7,.7 -1.3,2.4 -1.7,-.2 -.1,1.2 h -1.6 l .2,1.5 .2,2 -3,1.2 .1,1.3 1,1.7 -.1,5.2 -1.3,4.4 -1.7,2.5 1.2,1.4 .8,3.5 -1,2.5 -.2,2.1 1.7,3.4 2.5,4.9 1.2,1.9 1.6,6.9 -.1,8.8 -.9,3.9 -2,3.2 -.9,3.7 -2,3 -1.2,1 z m -95.8,-96.8 3,3.8 17,3.8 1.4,1 4,.8 .7,.5 2.8,-.2 4.9,.8 1.4,1.5 -1,1 .8,.8 3.8,.7 1.2,1.2 .1,4.4 -1.3,2.8 2,.1 1,-.8 .9,.8 -1.1,3.1 1,1.6 1.2,.3 .8,-1.8 2.9,-4.6 1.6,-6 2.3,-2 -.5,-1.6 .5,-.9 1,1.6 -.3,2.2 2.9,-2.2 .2,-2.3 2.1,.6 .8,-1.6 .7,.6 -.7,1.5 -1,.5 -1,2 1.4,1.8 1.1,-.5 -.5,-.7 1,-1.5 1.9,-1.7 h .8 l .2,-2.6 2,-1.8 7.9,-.5 1.9,-3.1 3.8,-.3 3.8,1.2 4.2,2.7 .7,-.2 -.2,-3.5 .7,-.2 4.5,1.1 1.5,-.2 2.9,-.7 1.7,.4 1.8,.1 v -1.1 l -.7,-.9 -1.5,-.2 -1.1,-.8 .5,-1.4 -.8,-.3 -2.6,.1 -.1,-1 1.1,-.8 .6,.8 .5,-1.8 -.7,-.7 .7,-.2 -1.4,-1.3 .3,-1.3 .1,-1.9 h -1.3 l -1.5,1 -1.9,.1 -.5,1.8 -1.9,.2 -.3,-1.2 -2.2,.1 -1,1.2 -.7,-.1 -.2,-.8 -2.6,.4 -.1,-4.8 1,-2 -.7,-.1 -1.8,1.1 h -2.2 l -3.8,2.7 -6.2,.3 -4.1,.8 -1.9,1.5 -1.4,1.3 -2.5,1.7 -.3,.8 -.6,-1.7 -1.3,-.6 v .6 l .7,.7 v 1.3 l -1.5,-.6 h -.6 l -.3,1.2 -2,-1.9 -1.3,-.2 -1.3,1.5 -3.2,-.1 -.5,-1.4 -2,-1.9 -1.3,-1.6 v -.7 l -1.1,-1.4 -2.6,-1.2 -3.3,-.1 -1.1,-.9 h -1.4 l -.7,.4 -2.2,2.2 -.7,1.1 -1,-.7 .2,-1 .8,-2.1 3.2,-5 .8,-.2 1.7,-1.9 .7,-1.6 3,-.6 .8,-.6 -.1,-1 -.5,-.5 -4.5,.2 -2,.5 -2.6,1.2 -1.2,1.2 -1.7,2.2 -1.8,1 -3.3,3.4 -.4,1.6 -7.4,4.6 -4,.5 -1.8,.4 -2.3,3 -1.8,.7 -4.4,2.3 z m 100.7,3.8 3.8,.1 .6,-.5 -.2,-2 -1.7,-1.8 -1.9,.1 -.1,.5 1.1,.4 -1.6,.8 -.3,1 -.6,-.6 -.4,.8 z m -75.1,-41.9 -2.3,.2 -2.7,1.9 -7.1,5.3 .8,1 1.8,.3 2.8,-2 -1.1,-.5 2.3,-1.6 h 1 l 3,-1.9 -.1,-.9 z m 41.1,62.8 v 1 l 2.1,1.6 -.2,-2.4 z m -.7,2.8 1.1,.1 v .9 h -1 z m 21.4,-21.3 v .9 l .8,-.2 v -.5 z m 4.7,3.1 -.1,-1.1 -1.6,-.2 -.6,-.4 h -.9 l -.4,.3 .9,.4 1.1,1.1 z m -18,1.2 -.1,1.1 -.3,.7 .2,2.2 .4,.3 .7,.1 .5,-.9 .1,-1.6 -.3,-.6 -.1,-1.1 z"/><path class="gm-usmap-state" d="m 464.7,68.6 -1.1,2.8 .8,1.4 -.3,5.1 -.5,1.1 2.7,9.1 1.3,2.5 .7,14 1,2.7 -.4,5.8 2.9,7.4 .3,5.8 -.1,2.1 -.1,2.2 -.9,2 -3.1,1.9 -.3,1.2 1.7,2.5 .4,1.8 2.6,.6 1.5,1.9 -.2,39.5 h 28.2 l 36.3,-.9 18.6,-.7 -1.1,-4.5 -.2,-3 -2.2,-3 -2.8,-.7 -5.2,-3.6 -.6,-3.3 -6.3,-3.1 -.2,-1.3 h -3.3 l -2.2,-2.6 -2,-1.3 .7,-5.1 -.9,-1.6 .5,-5.4 1,-1.8 -.3,-2.7 -1.2,-1.3 -1.8,-.3 v -1.7 l 2.8,-5.8 5.9,-3.9 -.4,-13 .9,.4 .6,-.5 .1,-1.1 .9,-.6 1.4,1.2 .7,-.1 v 0 l -1.2,-2.2 4.3,-3.1 3.1,-3.7 1.6,-.8 4.7,-5.9 6.3,-5.8 3.9,-2.1 6.3,-2.7 7.6,-4.5 -.6,-.4 -3.7,.7 -2.8,.1 -1,-1.6 -1.4,-.9 -9.8,1.2 -1,-2.8 -1.6,-.1 -1.7,.8 -3.7,3.1 h -4.1 l -2.1,-1 -.3,-1.7 -3.9,-.8 -.6,-1.6 -.7,-1.3 -1,.9 -2.6,.1 -9.9,-5.5 h -2.9 l -.8,-.7 -3.1,1.3 -.8,1.3 -3.3,.8 -1.3,-.2 v -1.7 l -.7,-.9 h -5.9 l -.4,-1.4 h -2.6 l -1.1,.4 -2.4,-1.7 .3,-1.4 -.6,-2.4 -.7,-1.1 -.2,-3 -1,-3.1 -2.1,-1.6 h -2.9 l .1,8 -30.9,-.4 z"/><path class="gm-usmap-state" d="m 623.8,468.6 -5,.1 -2.4,-1.5 -7.9,2.5 -.9,-.7 -.5,.2 -.1,1.6 -.6,.1 -2.6,2.7 -.7,-.1 -.6,-.7 -1.2,-1.8 .3,-1.3 -4.8,-6.8 .9,-4.6 1,-1.4 .1,-1.4 -36,2 1.7,-11.9 2.4,-4.8 6,-8.4 -1.8,-2.5 h 2 v -3.3 l -2.4,-2.5 .5,-1.7 -1.2,-1 -1.6,-7.1 .6,-1.4 1.2,-1.5 .5,-3 -1.5,-2.3 -.5,-2.2 .9,-.7 v -.8 l -1.7,-1.1 -.1,-.7 1.6,-.9 -1.2,-1.1 1.7,-7.1 3.4,-1.6 v -.8 l -1.1,-1.4 2.9,-5.4 h 1.9 l 1.5,-1.2 -.3,-5.2 3.1,-4.5 1.8,-.6 -.5,-3.1 38.3,-2.6 1.3,2 -1.3,67 4.4,33.2 z"/><path class="gm-usmap-state" d="m 555.3,248.9 -1.1,-1.1 -.6,-1.6 -1.7,-1.3 -14.3,.8 -27.2,1.2 -25.9,-.1 1.3,1.3 -.3,1.4 2.1,3.7 3.9,6.3 2.9,3 2,.6 .9,-1 1.5,2.1 -.1,1.2 -2.7,2.6 -.5,2.3 2.5,2.5 2.6,4.7 3.2,.7 .5,48.1 .2,10.8 39.1,-.7 39.8,-2 1.6,2.5 v 2.2 l -1.7,1.5 -2.8,5.1 11.2,-.8 1,-2 1.2,-.5 v -.7 l -1.2,-1.1 -.6,-1 1.7,.2 .8,-.7 -1.4,-1.5 1.4,-.5 .1,-1 -.6,-1 v -1.3 l -.7,-.7 .2,-1 h 1.1 l .7,.7 -.3,1 .8,.7 .8,-1 1,-2.7 1.4,.9 .7,-.4 1.2,-4.1 -1,-1 1,-2 .2,-.9 -1.3,-.8 h -2.8 l -1.4,-1.5 -1.8,-3.8 v -1.9 l .8,-.6 .1,-1.3 -1.7,-1.9 -.9,-2.5 -2.7,-4.1 -4.8,-1.3 -7.4,-7.1 -.4,-2.4 2.8,-7.6 -.4,-1.9 1.2,-1.1 v -1.3 l -2.8,-1.5 -3,-.7 -3.4,1.2 -1.3,-2.3 .6,-1.9 -.7,-2.4 -8.6,-8.4 -2.2,-1.5 -2.5,-5.9 -1.2,-5.4 1.4,-3.7 z"/><path class="gm-usmap-state" d="m 247,130.5 57.3,7.9 51,5.3 2,-20.7 5.2,-66.7 -53.5,-5.6 -54.3,-7.7 -65.9,-12.5 -4.8,22 3.7,7.4 -1.6,4.8 3.6,4.8 1.9,.7 3.9,8.3 v 2.1 l 2.3,3 h .9 l 1.4,2.1 h 3.2 v 1.6 l -7.1,17 -.5,4.1 1.4,.5 1.6,2.6 2.8,-1.4 3.6,-2.4 1.9,1.9 .5,2.5 -.5,3.2 2.5,9.7 2.6,3.5 2.3,1.4 .4,3 v 4.1 l 2.3,2.3 1.6,-2.3 6.9,1.6 2.1,-1.2 9,1.7 2.8,-3.3 1.8,-.6 1.2,1.8 1.6,4.1 .9,.1 z"/><path class="gm-usmap-state" d="m 402.5,191.1 38,1.6 3.4,3.2 1.7,.2 2.1,2 1.8,-.1 1.8,-2 1.5,.6 1,-.7 .7,.5 .9,-.4 .7,.4 .9,-.4 1,.5 1.4,-.6 2,.6 .6,1.1 6.1,2.2 1.2,1.3 .9,2.6 1.8,.7 1.5,-.2 .5,.9 v 2.3 l .6,1.7 v 1.4 l 1.3,3.7 2.2,4.3 .2,3.7 2.8,5 .4,6.1 1.3,.7 -.2,2.9 .8,3.3 -.7,2.2 1.8,4.4 1.3,1.3 -.3,1.4 2.1,3.7 3.9,6.3 h -32.4 l -43.7,-1.2 -36,-2 1.4,-22.1 -33.1,-2.4 3.7,-44.2 z"/><path class="gm-usmap-state" d="m 167.6,296.8 -3.4,17.5 -2.4,2.9 h -2 l -1.2,-2.7 -3.7,-1.4 -3.5,.6 -1,13.6 .5,4.9 -.5,2.9 -1.4,3 -70.4,-105 -1.1,-3.5 16.4,-63.1 47,11.2 24.4,5.4 23.3,4.7 z"/><path class="gm-usmap-state" d="m 862.6,93.6 -1.3,.1 -1,-1.1 -1.9,1.4 -.5,6.1 1.2,2.3 -1.1,3.5 2.1,2.8 -.4,1.7 .1,1.3 -1.1,2.1 -1.4,.4 -.6,1.3 -2.1,1 -.7,1.5 1.4,3.4 -.5,2.5 .5,1.5 -1,1.9 .4,1.9 -1.3,1.9 .2,2.2 -.7,1.1 .7,4.5 .7,1.5 -.5,2.6 .9,1.8 -.2,2.5 -.5,1.3 -.1,1.4 2.1,2.6 18.4,-3.8 1,-1.5 .3,-1.7 1.9,-.6 .5,-1.1 1.7,-1.1 1.3,.3 .8,-4.8 -2.3,-1.4 -.8,-2.2 -3.2,-2 -.6,-4 -11.9,-36.8 z"/><path class="gm-usmap-state" d="m 842.5,195.4 -14.6,-4.9 -1.8,2.5 .1,2.2 -3,5.4 1.5,1.8 -.7,2 -1,1 .5,3.6 2.7,.9 1,2.8 2.1,1.1 4.2,3.2 -3.3,2.6 -1.6,2.3 -1.8,3 -1.6,.6 -1.4,1.7 -1,2.2 -.3,2.1 .8,.9 .4,2.3 1.2,.6 2.4,1.5 1.8,.8 1.6,.8 .1,1.1 .8,.1 1.1,-1.2 .8,.4 2.1,.2 -.2,2.9 .2,2.5 1.8,-.7 1.5,-3.9 1.6,-4.8 2.9,-2.8 .6,-3.5 -.6,-1.2 1.7,-2.9 v -1.2 l -.7,-1.1 1.2,-2.7 -.3,-3.6 -.6,-8.2 -1.2,-1.4 v 1.4 l .5,.6 h -1.1 l -.6,-.4 -1.3,-.2 -.9,.6 -1.2,-1.6 .7,-1.7 v -1 l 1.7,-.7 .8,-2.1 z"/><path class="gm-usmap-state" d="m 357.5,332.9 h -.8 l -7.9,99.3 -31.8,-2.6 -34.4,-3.6 -.3,3 2,2.2 -30.8,-4.1 -1.4,10.2 -15.7,-2.2 17.4,-124.1 52.6,6.5 51.7,4.8 z"/><path class="gm-usmap-state" d="m 872.9,181.6 -1.3,.1 -.5,1 z m -30.6,22.7 .7,.6 1.3,-.3 1.1,.3 .9,-1.3 h 1.9 l 2.4,-.9 5.1,-2.1 -.5,-.5 -1.9,.8 -2,.9 .2,-.8 2.6,-1.1 .8,-1 1.2,.1 4.1,-2.3 v .7 l -4.2,3 4.5,-2.8 1.7,-2.2 1.5,-.1 4.5,-3.1 3.2,-3.1 3,-2.3 1,-1.2 -1.7,-.1 -1,1.2 -.2,.7 -.9,.7 -.8,-1.1 -1.7,1 -.1,.9 -.9,-.2 .5,-.9 -1.2,-.7 -.6,.9 .9,.3 .2,.5 -.3,.5 -1.4,2.6 h -1.9 l .9,-1.8 .9,-.6 .3,-1.7 1.4,-1.6 .9,-.8 1.5,-.7 -1.2,-.2 -.7,.9 h -.7 l -1.1,.8 -.2,1 -2.2,2.1 -.4,.9 -1.4,.9 -7.7,1.9 .2,.9 -.9,.7 -2,.3 -1,-.6 -.2,1.1 -1.1,-.4 .1,1 -1.2,-.1 -1.2,.5 -.2,1.1 h -1 l .2,1 h -.7 l .2,1 -1.8,.4 -1.5,2.3 z m -.8,-.4 -1.6,.4 v 1 l -.7,1.6 .6,.7 2.4,-2.3 -.1,-.9 z m -10.1,-95.2 -.6,1.9 1.4,.9 -.4,1.5 .5,3.2 2.2,2.3 -.4,2.2 .6,2 -.4,1 -.3,3.8 3.1,6.7 -.8,1.8 .9,2.2 .9,-1.6 1.9,1.5 3,14.2 -.5,2 1.1,1 -.5,15 .7,1 2.8,16.3 1.8,1.5 -3.5,3.4 1.7,2.2 -1.3,3.3 -1.5,1.7 -1.5,2.3 -.2,-.7 .4,-5.9 -14.6,-4.9 -1.6,-1.1 -1.9,.3 -3,-2.2 -3,-5.8 h -2 l -.4,-1.5 -1.7,-1.1 -70.5,13.9 -.8,-6 4.3,-3.9 .6,-1.7 3.9,-2.5 .6,-2.4 2.3,-2 .8,-1.1 -1.7,-3.3 -1.7,-.5 -1.8,-3 -.2,-3.2 7.6,-3.9 8.2,-1.6 h 4.4 l 3.2,1.6 .9,-.1 1.8,-1.6 3.4,-.7 h 3 l 2.6,-1.3 2.5,-2.6 2.4,-3.1 1.9,-.4 1.1,-.5 .4,-3.2 -1.4,-2.7 -1.2,-.7 2,-1.3 -.1,-1.8 h -1.5 l -2.3,-1.4 -.1,-3.1 6.2,-6.1 .7,-2.4 3.7,-6.3 5.9,-6.4 2.1,-1.7 2.5,.1 20.6,-5.2 z"/><path class="gm-usmap-state" d="m 829,300.1 -29.1,6.1 -39.4,7.3 -29.4,3.5 v 5.2 l -1.5,-.1 -1.4,1.2 -2.4,5.2 -2.6,-1.1 -3.5,2.5 -.7,2.1 -1.5,1.2 -.8,-.8 -.1,-1.5 -.8,-.2 -4,3.3 -.6,3.4 -4.7,2.4 -.5,1.2 -3.2,2.6 -3.6,.5 -4.6,3 -.8,4.1 -1.3,.9 -1.5,-.1 -1.4,1.3 -.1,4.9 21.4,-3 4.4,-1.9 1.3,-.1 7.3,-4.3 23.2,-2.2 .4,.5 -.2,1.4 .7,.3 1.2,-1.5 3.3,3 .1,2.6 19.7,-2.8 24.5,17.1 4,-2.2 3,-.7 h 1.7 l 1.1,1.1 .8,-2 .6,-5 1.7,-3.9 5.4,-6.1 4.1,-3.5 5.4,-2.3 2.5,-.4 1.3,.4 .7,1.1 3.3,-6.6 3.3,-5.3 -.7,-.3 -4.4,6.8 -.5,-.8 2,-2.2 -.4,-1.5 -2,-.5 1,1.3 -1.2,.1 -1.2,-1.8 -1.2,2 -1.6,.2 1,-2.7 .7,-1.7 -.2,-2.9 -2.2,-.1 .9,-.9 1.1,.3 2.7,.1 .8,-.5 h 2.3 l 2,-1.9 .2,-3.2 1.3,-1.4 1.2,-.2 1.3,-1 -.5,-3.7 -2.2,-3.8 -2.7,-.2 -.9,1.6 -.5,-1 -2.7,.2 -1.2,.4 -1.9,1.2 -.3,-.4 h -.9 l -1.8,1.2 -2.6,.5 v -1.3 l .8,-1 1,.7 h 1 l 1.7,-2.1 3.7,-1.7 2,-2.2 h 2.4 l .8,1.3 1.7,.8 -.5,-1.5 -.3,-1.6 -2.8,-3.1 -.3,-1.4 -.4,1 -.9,-1.3 z m 7,31 2.7,-2.5 4.6,-3.3 v -3.7 l -.4,-3.1 -1.7,-4.2 1.5,1.4 1,3.2 .4,7.6 -1.7,.4 -3.1,2.4 -3.2,3.2 z m 1.9,-19.3 -.9,-.2 v 1 l 2.5,2.2 -.2,-1.4 z m 2.9,2.1 -1.4,-2.8 -2.2,-3.4 -2.4,-3 -2.2,-4.3 -.8,-.7 2.2,4.3 .3,1.3 3.4,5.5 1.8,2.1 z"/><path class="gm-usmap-state" d="m 464.7,68.6 -1.1,2.8 .8,1.4 -.3,5.1 -.5,1.1 2.7,9.1 1.3,2.5 .7,14 1,2.7 -.4,5.8 2.9,7.4 .3,5.8 -.1,2.1 -29.5,-.4 -46,-2.1 -39.2,-2.9 5.2,-66.7 44.5,3.4 55.3,1.6 z"/><path class="gm-usmap-state" d="m 685.7,208.8 1.9,-.4 3,1.3 2.1,.6 .7,.9 h 1 l 1,-1.5 1.3,.8 h 1.5 l -.1,1 -3.1,.5 -2,1.1 1.9,.8 1.6,-1.5 2.4,-.4 2.2,1.5 1.5,-.1 2.5,-1.7 3.6,-2.1 5.2,-.3 4.9,-5.9 3.8,-3.1 9.3,-5.1 4.9,29.9 -2.2,1.2 1.4,2.1 -.1,2.2 .6,2 -1.1,3.4 -.1,5.4 -1,3.6 .5,1.1 -.4,2.2 -1.1,.5 -2,3.3 -1.8,2 h -.6 l -1.8,1.7 -1.3,-1.2 -1.5,1.8 -.3,1.2 h -1.3 l -1.3,2.2 .1,2.1 -1,.5 1.4,1.1 v 1.9 l -1,.2 -.7,.8 -1,.5 -.6,-2.1 -1.6,-.5 -1,2.3 -.3,2.2 -1.1,1.3 1.3,3.6 -1.5,.8 -.4,3.5 h -1.5 l -3.2,1.4 -1.2,-2.1 -3.5,-1.4 -.8,-2.9 -.5,-.8 -3.4,1.8 -.6,1.7 h -.9 l -1.3,.7 -1.2,-.8 -3,-.8 -1.9,.8 v 1 l -2.2,-.2 -1.9,-2.1 -2.3,.2 -4.1,-.7 v -1 l -2.2,-3.4 -2.9,-1.2 -1.9,.8 -2.2,-.1 -1.3,-.5 -6.6,-57.2 21.4,-3.5 z"/><path class="gm-usmap-state" d="m 501.5,398.6 -4.6,-3.8 -2.2,-.9 -.5,1.6 -5.1,.3 -.6,-1.5 -5,2.5 -1.6,-.7 -3.7,.3 -.6,1.7 -3.6,.9 -1.3,-1.2 -1.2,.1 -2,-1.8 -2.1,.7 -2,-.5 -1.8,-2 -2.5,4.2 -1.2,.8 -1,-1.8 .3,-2 -1.2,-.7 -2.3,2.5 -1.7,-1.2 -.1,-1.5 -1.3,.5 -2.6,-1.7 -3,2.6 -2.3,-1.1 .7,-2.1 -2.3,.1 -1.9,-3 -3.5,-1.1 -2,2.3 -2.3,-2.2 -1.4,.4 -2,.1 -3.5,-1.9 -2.3,.1 -1.2,-.7 -.5,-2.9 -2.3,-1.7 -1.1,1.5 -1.4,-1 -1.2,-.4 -1.1,1 -1.5,-.3 -2.5,-3 -2.7,-1.3 1.4,-42.7 -52.6,-3.2 .6,-10.6 16.5,1 67.7,2.9 62,.1 .2,10.8 4.1,24.4 -.7,39 z"/><path class="gm-usmap-state" d="m 93.9,166.5 47,11.2 8.5,-37.3 2.9,-5.8 .4,-2.1 .8,-.9 -.9,-2 -2.9,-1.2 .2,-4.2 4,-5.8 2.5,-.8 1.6,-2.3 -.1,-1.6 1.8,-1.6 3.2,-5.5 4.2,-4.8 -.5,-3.2 -3.5,-3.1 -1.6,-3.6 -30.3,-7.3 -2.8,1 -5.4,-.9 -1.8,-.9 -1.5,1.2 -3.3,-.4 -4.5,.5 -.9,.7 -4.2,-.4 -.8,-1.6 -1.2,-.2 -4.4,1.3 -1.6,-1.1 -2.2,.8 -.2,-1.8 -2.3,-1.2 -1.5,-.2 -1,-1.1 -3,.3 -1.2,-.8 h -1.2 l -1.2,.9 -5.5,.7 -6.6,-4.2 1.1,-5.6 -.4,-4.1 -3.2,-3.7 -3.7,.1 -.4,-1.1 .4,-1.2 -.7,-.8 -1,.1 -1.1,1.3 -1.5,-.2 -.5,-1.1 -1,-.1 -.7,.6 -2,-1.9 v 4.3 l -1.3,1.3 -1.1,3.5 -.1,2.3 -4.5,12.3 -13.2,31.3 -3.2,4.6 -1.6,-.1 .1,2.1 -5.2,7.1 -.3,3.3 1,1.3 .1,2.4 -1.2,1.1 -1.2,3 .1,5.7 1.2,2.9 z"/><path class="gm-usmap-state" d="m 826.3,189.4 -1.9,.3 -3,-2.2 -3,-5.8 h -2 l -.4,-1.5 -1.7,-1.1 -70.5,13.9 -.8,-6 -4.2,3.4 -.9,.1 -2.7,3 -3.3,1.7 4.9,29.9 3.2,19.7 17.4,-2.9 60.5,-11.8 1.2,-2.1 1.5,-1.1 1.6,-.3 1.6,.6 1.4,-1.7 1.6,-.6 1.8,-3 1.6,-2.3 3.3,-2.6 -4.2,-3.2 -2.1,-1.1 -1,-2.8 -2.7,-.9 -.5,-3.6 1,-1 .7,-2 -1.5,-1.8 3,-5.4 -.1,-2.2 1.8,-2.5 z"/><path class="gm-usmap-state" d="m 883.2,170.7 -1.3,-1.1 -2.6,-1.3 -.6,-2.2 h -.8 l -.7,-2.6 -6.5,2 3.2,12.3 -.4,1.1 .4,1.8 5.6,-3.6 .1,-3 -.8,-.8 .4,-.6 -.1,-1.3 -.9,-.7 1.2,-.4 -.9,-1.6 1.8,.7 .3,1.4 .7,1.2 -1.4,-.8 1.1,1.7 -.3,1.2 -.6,-1.1 v 2.5 l .6,-.9 .4,.9 1.3,-1.5 -.2,-2.5 1.4,3.1 1,-.9 z m -4.7,12.2 h .9 l .5,-.6 -.8,-1.3 -.7,.7 z"/><path class="gm-usmap-state" d="m 772.3,350.2 -19.7,2.8 -.1,-2.6 -3.3,-3 -1.2,1.5 -.7,-.3 .2,-1.4 -.4,-.5 -23.2,2.2 -7.3,4.3 -1.3,.1 -4.4,1.9 -.1,1.9 -1.9,1 -1.4,3.2 .2,1.3 6.1,3.8 2.6,-.3 3.1,4 .4,1.7 4.2,5.1 2.6,1.7 1.4,.2 2.2,1.6 1.1,2.2 2,1.6 1.8,.5 2.7,2.7 .1,1.4 2.6,2.8 5,2.3 3.6,6.7 .3,2.7 3.9,2.1 2.5,4.8 .8,3.1 4.2,.4 .8,-1.5 h .6 l 1.8,-1.5 .5,-2 3.2,-2.1 .3,-2.4 -1.2,-.9 .8,-.7 .8,.4 1.3,-.4 1.8,-2.1 3.8,-1.8 1.6,-2.4 .1,-.7 4.8,-4.4 -.1,-.5 -.9,-.8 1.1,-1.5 h .8 l .4,.5 .7,-.8 h 1.3 l .6,-1.5 2.3,-2.1 -.3,-5.4 .8,-2.3 3.6,-6.2 2.4,-2.2 2.2,-1.1 z"/><path class="gm-usmap-state" d="m 396.5,125.9 46,2.1 29.5,.4 -.1,2.2 -.9,2 -3.1,1.9 -.3,1.2 1.7,2.5 .4,1.8 2.6,.6 1.5,1.9 -.2,39.5 -2.2,-.1 -.1,1.6 1.3,1.5 -.1,1.1 -1,.5 .4,1.6 1.3,.4 .7,2 -1.7,5.1 -1,4.3 1.3,1.2 .3,1.3 .7,1.7 -1.5,.2 -1.8,-.7 -.9,-2.6 -1.2,-1.3 -6.1,-2.2 -.6,-1.1 -2,-.6 -1.4,.6 -1,-.5 -.9,.4 -.7,-.4 -.9,.4 -.7,-.5 -1,.7 -1.5,-.6 -1.8,2 -1.8,.1 -2.1,-2 -1.7,-.2 -3.4,-3.2 -38,-1.6 -51.1,-3.5 3.9,-43.9 2,-20.7 z"/><path class="gm-usmap-state" d="m 620.9,365.1 45.7,-4 22.9,-2.9 .1,-4.9 1.4,-1.3 1.5,.1 1.3,-.9 .8,-4.1 4.6,-3 3.6,-.5 3.2,-2.6 .5,-1.2 4.7,-2.4 .6,-3.4 4,-3.3 .8,.2 .1,1.5 .8,.8 1.5,-1.2 .7,-2.1 3.5,-2.5 2.6,1.1 2.4,-5.2 1.4,-1.2 1.5,.1 0,-5.2 .3,-.7 -4.6,.5 -.2,1 -28.9,3.3 -5.6,1.4 -20.5,1.4 -5.2,.8 -17.4,1 -2.6,.8 -22.6,2 -.7,-.6 h -3.7 l 1.2,3.2 -.6,.9 -23.3,1.5 -.8,1 -.8,-.7 h -1 v 1.3 l .6,1 -.1,1 -1.4,.5 1.4,1.5 -.8,.7 -1.7,-.2 .6,1 1.2,1.1 v .7 l -1.2,.5 -1,2 .1,.6 1.4,1 -.4,.7 h -1.5 v .5 l .9,.9 .1,.8 -1.4,.2 -.5,.8 -1.6,.2 -.9,.9 .6,.9 1.1,-.1 .5,.9 -1.6,1.3 .4,1.5 -2,-.6 -.1,.7 .4,1.1 -.3,1.4 -1.3,-.8 -.8,.8 1.1,.1 .1,1.5 -.6,1 1.1,.9 -.3,1.5 .8,.7 -.7,1 -1.2,-.5 -.9,2.2 -1.6,.7 z"/><path class="gm-usmap-state" d="m 282.3,429 .3,-3 34.4,3.6 31.8,2.6 7.9,-99.3 .8,0 52.6,3.2 -1.4,42.7 2.7,1.3 2.5,3 1.5,.3 1.1,-1 1.2,.4 1.4,1 1.1,-1.5 2.3,1.7 .5,2.9 1.2,.7 2.3,-.1 3.5,1.9 2,-.1 1.4,-.4 2.3,2.2 2,-2.3 3.5,1.1 1.9,3 2.3,-.1 -.7,2.1 2.3,1.1 3,-2.6 2.6,1.7 1.3,-.5 .1,1.5 1.7,1.2 2.3,-2.5 1.2,.7 -.3,2 1,1.8 1.2,-.8 2.5,-4.2 1.8,2 2,.5 2.1,-.7 2,1.8 1.2,-.1 1.3,1.2 3.6,-.9 .6,-1.7 3.7,-.3 1.6,.7 5,-2.5 .6,1.5 5.1,-.3 .5,-1.6 2.2,.9 4.6,3.8 6.4,1.9 2.6,2.3 2.8,-1.3 3.2,.8 .2,11.9 .5,19.9 .7,3.4 2.6,2.8 .7,5.4 3.8,4.6 .8,4.3 h 1 l -.1,7.3 -3.3,6.4 1.3,2.3 -1.3,1.5 .7,3 -.1,4.3 -2.2,3.5 -.1,.8 -1.7,1.2 1,1.8 1.2,1.1 -3.5,.3 -8.4,3.9 -3.5,1.4 -1.8,1.8 -.7,-.5 2.1,-2.3 1.8,-.7 .5,-.9 -2.9,-.1 -.7,-.8 .8,-2 -.9,-1.8 h -.6 l -2.4,1.3 -1.9,2.6 .3,1.7 3.3,3.4 1.3,.3 v .8 l -2.3,1.6 -4.9,4 -4,3.9 -3.2,1.4 -5,3 -3.7,2 -4.5,1.9 -4.1,2.5 3.2,-3 v -1.1 l .6,-.8 -.2,-1.8 -1.5,-.1 -1.1,1.5 -2.6,1.3 -1.8,-1.2 -.3,-1.7 h -1.5 l .8,2.2 1.4,.7 1.2,.9 1.8,1.6 -.7,.8 -3.9,1.7 -1.7,.1 -1.2,-1.2 -.5,2.1 .5,1.1 -2.7,2 -1.5,.2 -.8,.7 -.4,1.7 -1.8,3.3 -1.6,.7 -1.6,-.6 -1.8,1.1 .3,1.4 1.3,.8 1,.8 -1.8,3.5 -.3,2.8 -1,1.7 -1.4,1 -2.9,.4 1.8,.6 1.9,-.6 -.4,3.2 -1.1,-.1 .2,1.2 .3,1.4 -1.3,.9 v 3.1 l 1.6,1.4 .6,3.1 -.4,2.2 -1,.4 .4,1.5 1.1,.4 .8,1.7 v 2.6 l 1.1,2.1 2.2,2.6 -.1,.7 -2.2,-.2 -1.6,1.4 .2,1.4 -.9,-.3 -1.4,-.2 -3.4,-3.7 -2.3,-.6 h -7.1 l -2.8,-.8 -3.6,-3 -1.7,-1 -2.1,.1 -3.2,-2.6 -5.4,-1.6 v -1.3 l -1.4,-1.8 -.9,-4.7 -1.1,-1.7 -1.7,-1.4 v -1.6 l -1.4,-.6 .6,-2.6 -.3,-2.2 -1.3,-1.4 .7,-3 -.8,-3.2 -1.7,-1.4 h -1.1 l -4,-3.5 .1,-1.9 -.8,-1.7 -.8,-.2 -.9,-2.4 -2,-1.6 -2.9,-2.5 -.2,-2.1 -1,-.7 .2,-1.6 .5,-.7 -1.4,-1.5 .1,-.7 -2,-2.2 .1,-2.1 -2.7,-4.9 -.1,-1.7 -1.8,-3.1 -5.1,-4.8 v -1.1 l -3.3,-1.7 -.1,-1.8 -1.2,-.4 v -.7 l -.8,-.2 -2.1,-2.8 h -.8 l -.7,-.6 -1.3,1.1 h -2.2 l -2.6,-1.1 h -4.6 l -4.2,-2.1 -1.3,1.9 -2.2,-.6 -3.3,1.2 -1.7,2.8 -2,3.2 -1.1,4.4 -1.4,1.2 -1.1,.1 -.9,1.6 -1.3,.6 -.1,1.8 -2.9,.1 -1.8,-1.5 h -1 l -2,-2.9 -3.6,-.5 -1.7,-2.3 -1.3,-.2 -2.1,-.8 -3.4,-3.4 .2,-.8 -1.6,-1.2 -1,-.1 -3.4,-3.1 -.1,-2 -2.3,-4 .2,-1.6 -.7,-1.3 .8,-1.5 -.1,-2.4 -2.6,-4.1 -.6,-4.2 -1.6,-1.6 v -1 l -1.2,-.2 -.7,-1.1 -2.4,-1.7 -.9,-.1 -1.9,-1.6 v -1.1 l -2.9,-1.8 -.6,-2.1 -2.6,-2.3 -3.2,-4.4 -3,-1.3 -2.1,-1.8 .2,-1.2 -1.3,-1.4 -1.7,-3.7 -2.4,-1 z m 174.9,138.3 .8,.1 -.6,-4.8 -3.5,-12.3 -.2,-8.1 4.9,-10.5 6.1,-8.2 7.2,-5.1 v -.7 h -.8 l -2.6,1 -3.6,2.3 -.7,1.5 -8.2,11.6 -2.8,7.9 v 8.8 l 3.6,12 z"/><path class="gm-usmap-state" d="m 233.2,217.9 3.3,-21.9 -47.9,-8.2 -21,109 46.2,8.2 40,6 11.5,-88.3 z"/><path class="gm-usmap-state" d="m 859.1,102.4 -1.1,3.5 2.1,2.8 -.4,1.7 .1,1.3 -1.1,2.1 -1.4,.4 -.6,1.3 -2.1,1 -.7,1.5 1.4,3.4 -.5,2.5 .5,1.5 -1,1.9 .4,1.9 -1.3,1.9 .2,2.2 -.7,1.1 .7,4.5 .7,1.5 -.5,2.6 .9,1.8 -.2,2.5 -.5,1.3 -.1,1.4 2.1,2.6 -12.4,2.7 -1.1,-1 .5,-2 -3,-14.2 -1.9,-1.5 -.9,1.6 -.9,-2.2 .8,-1.8 -3.1,-6.7 .3,-3.8 .4,-1 -.6,-2 .4,-2.2 -2.2,-2.3 -.5,-3.2 .4,-1.5 -1.4,-.9 .6,-1.9 -.8,-1.7 27.3,-6.9 z"/><path class="gm-usmap-state" d="m 834.7,265.4 -1.1,2.8 .5,1.1 .4,-1.1 .8,-3.1 z m -34.6,-7 -.7,-1 1,-.1 1,-.9 .4,-1.8 -.2,-.5 .1,-.5 -.3,-.7 -.6,-.5 -.4,-.1 -.5,-.4 -.6,-.6 h -1 l -.6,-.1 -.4,-.4 .1,-.5 -1.7,-.6 -.8,.3 -1.2,-.1 -.7,-.7 -.5,-.2 -.2,-.7 .6,-.8 v -.9 l -1.2,-.2 -1,-.9 -.9,.1 -1.6,-.3 -.4,.7 -.4,1.6 -.5,2.3 -10,-5.2 -.2,.9 .9,1.6 -.8,2.3 .1,2.9 -1.2,.8 -.5,2.1 -.9,.8 -1.4,1.8 -.9,.8 -1,2.5 -2.4,-1.1 -2.3,8.5 -1.3,1.6 -2.8,-.5 -1.3,-1.9 -2.3,-.7 -.1,4.7 -1.4,1.7 .4,1.5 -2.1,2.2 .4,1.9 -3.7,6.3 -1,3.3 1.5,1.2 -1.5,1.9 .1,1.4 -2.3,2 -.7,-1.1 -4.3,3.1 -1.5,-1 -.6,1.4 .8,.5 -.5,.9 -5.5,2.4 -3,-1.8 -.8,1.7 -1.9,1.8 -2.3,.1 -4.4,-2.3 -.1,-1.5 -1.5,-.7 .8,-1.2 -.7,-.6 -4.9,6.6 -2.9,1 -3,3 -.4,2.2 -2.1,1.3 -.1,1.7 -1.4,1.4 -1.8,.5 -.5,1.9 -1,.4 -6.9,4.2 28.9,-3.3 .2,-1 4.6,-.5 -.3,.7 29.4,-3.5 39.4,-7.3 29.1,-6.1 -.6,-1.2 .4,-.1 .9,.9 -.1,-1.4 -.3,-1.9 1.6,1.2 .9,2.1 v -1.3 l -3.4,-5.5 v -1.2 l -.7,-.8 -1.3,.7 .5,1.4 h -.8 l -.4,-1 -.6,.9 -.9,-1.1 -2.1,-.1 -.2,.7 1.5,2.1 -1.4,-.7 -.5,-1 -.4,.8 -.8,.1 -1.5,1.7 .3,-1.6 v -1.4 l -1.5,-.7 -1.8,-.5 -.2,-1.7 -.6,-1.3 -.6,1.1 -1.7,-1 -2,.3 .2,-.9 1.5,-.2 .9,.5 1.7,-.8 .9,.4 .5,1 v .7 l 1.9,.4 .3,.9 .9,.4 .9,1.2 1.4,-1.6 h .6 l -.1,-2.1 -1.3,1 -.6,-.9 1.5,-.2 -1.2,-.9 -1.2,.6 -.1,-1.7 -1.7,.2 -2.2,-1.1 -1.8,-2.2 3.6,2.2 .9,.3 1.7,-.8 -1.7,-.9 .6,-.6 -1,-.5 .8,-.2 -.3,-.9 1.1,.9 .4,-.8 .4,1.3 1.2,.8 .6,-.5 -.5,-.6 -.1,-2.5 -1.1,-.1 -1.6,-.8 .9,-1.1 -2,-.1 -.4,-.5 -1.4,.6 -1.4,-.8 -.5,-1.2 -2.1,-1.2 -2.1,-1.8 -2.2,-1.9 3,1.3 .9,1.2 2.1,.7 2.3,2.5 .2,-1.7 .6,1.3 2.3,.5 v -4 l -.8,-1.1 1.1,.4 .1,-1.6 -3.1,-1.4 -1.6,-.2 -1.3,-.2 .3,-1.2 -1.5,-.3 -.1,-.6 h -1.8 l -.2,.8 -.7,-1 h -2.7 l -1,-.4 -.2,-1 -1.2,-.6 -.4,-1.5 -.6,-.4 -.7,1.1 -.9,.2 -.9,.7 h -1.5 l -.9,-1.3 .4,-3.1 .5,-2.4 .6,.5 z m 21.9,11.6 .9,-.1 0,-.6 -.8,.1 z m 7.5,14.2 -1,2.7 1.2,-1.3 z m -1.8,-15.3 .7,.3 -.2,1.9 -.5,-.5 -1.3,1 1,.4 -1.8,4.4 .1,8.1 1.9,3.1 .5,-1.5 .4,-2.7 -.3,-2.3 .7,-.9 -.2,-1.4 1.2,-.6 -.6,-.5 .5,-.7 .8,1.1 -.2,1.1 -.4,3.9 1.1,-2.2 .4,-3.1 .1,-3 -.3,-2 .6,-2.3 1.1,-1.8 .1,-2.2 .3,-.9 -4.6,1.6 -.7,.8 z"/><path class="gm-usmap-state" d="m 161.9,83.6 .7,4 -1.1,4.3 -30.3,-7.3 -2.8,1 -5.4,-.9 -1.8,-.9 -1.5,1.2 -3.3,-.4 -4.5,.5 -.9,.7 -4.2,-.4 -.8,-1.6 -1.2,-.2 -4.4,1.3 -1.6,-1.1 -2.2,.8 -.2,-1.8 -2.3,-1.2 -1.5,-.2 -1,-1.1 -3,.3 -1.2,-.8 h -1.2 l -1.2,.9 -5.5,.7 -6.6,-4.2 1.1,-5.6 -.4,-4.1 -3.2,-3.7 -3.7,.1 -.4,-1.1 .4,-1.2 -.7,-.8 -1,.1 -2.1,-1.5 -1.2,.4 -2,-.1 -.7,-1.5 -1.6,-.3 2.5,-7.5 -.7,6 .5,.5 v -2 l .8,-.2 1.1,2.3 -.5,-2.2 1.2,-4.2 1.8,.4 -1.1,-2 -1,.3 -1.5,-.4 .2,-4.2 .2,1.5 .9,.5 .6,-1.6 h 3.2 l -2.2,-1.2 -1.7,-1.9 -1.4,1.6 1.2,-3.1 -.3,-4.6 -.2,-3.6 .9,-6.1 -.5,-2 -1.4,-2.1 .1,-4 .4,-2.7 2,-2.3 -.7,-1.4 .2,-.6 .9,.1 7.8,7.6 4.7,1.9 5.1,2.5 3.2,-.1 .2,3 1,-1.6 h .7 l .6,2.7 .5,-2.6 1.4,-.2 .5,.7 -1.1,.6 .1,1.6 .7,-1.5 h 1.1 l -.4,2.6 -1.1,-.8 .4,1.4 -.1,1.5 -.8,.7 -2.5,2.9 1.2,-3.4 -1.6,.4 -.4,2.1 -3.8,2.8 -.4,1 -2.1,2.2 -.1,1 h 2.2 l 2.4,-.2 .5,-.9 -3.9,.5 v -.6 l 2.6,-2.8 1.8,-.8 1.9,-.2 1,-1.6 3,-2.3 v -1.4 h 1.1 l .1,4 h -1.5 l -.6,.8 -1.1,-.9 .3,1.1 v 1.7 l -.7,.7 -.3,-1.6 -.8,.8 .7,.6 -.9,1.1 h 1.3 l .7,-.5 .1,2 -1,1.9 -.9,1 -.1,1.8 -1,-.2 -.2,-1.4 .9,-1.1 -.8,-.5 -.8,.7 -.7,2.2 -.8,.9 -.1,-2 .8,-1.1 -.2,-1.1 -1.2,1.2 .1,2.2 -.6,.4 -2.1,-.4 -1.3,1.2 2.2,-.6 -.2,2.2 1,-1.8 .4,1.4 .5,-1 .7,1.8 h .7 l .7,-.8 .6,-.1 2,-1.9 .2,-1.2 .8,.6 .3,.9 .7,-.3 .1,-1.2 h 1.3 l .2,-2.9 -.1,-2.7 .9,.3 -.7,-2.1 1.4,-.8 .2,-2.4 2.3,-2.2 1,.1 .3,-1.4 -1.2,-1.4 -.1,-3.5 -.8,.9 .7,2.9 -.6,.1 -.6,-1.9 -.6,-.5 .3,-2.3 1.8,-.1 .3,.7 .3,-1.6 -1.6,-1.7 -.6,-1.6 -.2,2 .9,1.1 -.7,.4 -1,-.8 -1.8,1.3 1.5,.5 .2,2.4 -.3,1.8 .9,-1.3 1.4,2.3 -.4,1.9 h -1.5 v -1.2 l -1.5,-1.2 .5,-3 -1.9,-2.6 2.7,-3 .6,-4.1 h .9 l 1.4,3.2 v -2.6 l 1.2,.3 v -3.3 l -.9,-.8 -1.2,2.5 -1,-3 1.3,-.1 -1.5,-4.9 1.9,-.6 25.4,7.5 31.7,8 23.6,5.5 z m -78.7,-39.4 h .5 l .1,.8 -.5,.3 .1,.6 -.7,.4 -.2,-.9 .5,-.4 z m 5,-4.3 -1.2,1.9 -.1,.8 .4,.2 .5,-.6 1.1,.1 z m -.4,-21.6 .5,.6 1.3,-.3 .2,-1 1.2,-1.8 -1,-.4 -.7,1.6 -.1,-1.6 -1.1,.2 -.7,1.4 z m 3.2,-5.5 .7,1.5 -.9,.2 -.8,.4 -.2,-2.4 z m -2.7,-1.6 -1.1,-.2 .5,1.4 z m -1,2.5 .8,.4 -.4,1.1 1.7,-.5 -.2,-2.2 -.9,-.2 z m -2.7,-.4 .3,2.7 1.6,1.3 .6,-1.9 -1.1,-2.2 z m 1.9,-1.1 -1.1,-1 -.9,.1 1.8,1.5 z m 3.2,-7 h -1.2 v .8 l 1.2,.6 z m -.9,32.5 .4,-2.7 h -1.1 l -.2,1.9 z"/><path class="gm-usmap-state" d="m 723.4,297.5 -.8,1.2 1.5,.7 .1,1.5 4.4,2.3 2.3,-.1 1.9,-1.8 .8,-1.7 3,1.8 5.5,-2.4 .5,-.9 -.8,-.5 .6,-1.4 1.5,1 4.3,-3.1 .7,1.1 2.3,-2 -.1,-1.4 1.5,-1.9 -1.5,-1.2 1,-3.3 3.7,-6.3 -.4,-1.9 2.1,-2.2 -.4,-1.5 1.4,-1.7 .1,-4.7 2.3,.7 1.3,1.9 2.8,.5 1.3,-1.6 2.3,-8.5 2.4,1.1 1,-2.5 .9,-.8 1.4,-1.8 .9,-.8 .5,-2.1 1.2,-.8 -.1,-2.9 .8,-2.3 -.9,-1.6 .2,-.9 10,5.2 .5,-2.3 .4,-1.6 .4,-.7 -.9,-.4 .2,-1.6 -1,-.5 -.2,-.7 h -.7 l -.8,-1.2 .2,-1 -2.6,.4 -2.2,-1.6 -1.4,.3 -.9,1.4 h -1.3 l -1.7,2.9 -3.3,.4 -1.9,-1 -2.6,3.8 -2.2,-.3 -3.1,3.9 -.9,1.6 -1.8,1.6 -1.7,-11.4 -17.4,2.9 -3.2,-19.7 -2.2,1.2 1.4,2.1 -.1,2.2 .6,2 -1.1,3.4 -.1,5.4 -1,3.6 .5,1.1 -.4,2.2 -1.1,.5 -2,3.3 -1.8,2 h -.6 l -1.8,1.7 -1.3,-1.2 -1.5,1.8 -.3,1.2 h -1.3 l -1.3,2.2 .1,2.1 -1,.5 1.4,1.1 v 1.9 l -1,.2 -.7,.8 -1,.5 -.6,-2.1 -1.6,-.5 -1,2.3 -.3,2.2 -1.1,1.3 1.3,3.6 -1.5,.8 -.4,3.5 h -1.5 l -3.2,1.4 -.1,1.1 .6,1 -.6,3.6 1.9,1.6 .8,1.1 1,.6 -.1,.9 4.4,5.6 h 1.4 l 1.5,1.8 1.2,.3 1.4,-.1 z"/><path class="gm-usmap-state" d="m 611,144 -2.9,.8 .2,2.3 -2.4,3.4 -.2,3.1 .6,.7 .8,-.7 .5,-1.6 2,-1.1 1.6,-4.2 3.5,-1.1 .8,-3.3 .7,-.9 .4,-2.1 1.8,-1.1 v -1.5 l 1,-.9 1.4,.1 v 2 l -1,.1 .5,1.2 -.7,2.2 -.6,.1 -1.2,4.5 -.7,.5 -2.8,7.2 -.3,4.2 .6,2 .1,1.3 -2.4,1.9 .3,1.9 -.9,3.1 .3,1.6 .4,3.7 -1.1,4.1 -1.5,5 1,1.5 -.3,.3 .8,1.7 -.5,1.1 1.1,.9 v 2.7 l 1.3,1.5 -.4,3 .3,4 -45.9,2.8 -1.3,-2.8 -3.3,-.7 -2.7,-1.5 -2,-5.5 .1,-2.5 1.6,-3.3 -.6,-1.1 -2.1,-1.6 -.2,-2.6 -1.1,-4.5 -.2,-3 -2.2,-3 -2.8,-.7 -5.2,-3.6 -.6,-3.3 -6.3,-3.1 -.2,-1.3 h -3.3 l -2.2,-2.6 -2,-1.3 .7,-5.1 -.9,-1.6 .5,-5.4 1,-1.8 -.3,-2.7 -1.2,-1.3 -1.8,-.3 v -1.7 l 2.8,-5.8 5.9,-3.9 -.4,-13 .9,.4 .6,-.5 .1,-1.1 .9,-.6 1.4,1.2 .7,-.1 h 2.6 l 6.8,-2.6 .3,-1 h 1.2 l .7,-1.2 .4,.8 1.8,-.9 1.8,-1.7 .3,.5 1,-1 2.2,1.6 -.8,1.6 -1.2,1.4 .5,1.5 -1.4,1.6 .4,.9 2.3,-1.1 v -1.4 l 3.3,1.9 1.9,.7 1.9,.7 3,3.8 17,3.8 1.4,1 4,.8 .7,.5 2.8,-.2 4.9,.8 1.4,1.5 -1,1 .8,.8 3.8,.7 1.2,1.2 .1,4.4 -1.3,2.8 2,.1 1,-.8 .9,.8 -1.1,3.1 1,1.6 1.2,.3 z m -49.5,-37.3 -.5,.1 -1.5,1.6 .2,.5 1.5,-.6 v -.6 l .9,-.3 z m 1.6,-1.1 -1,.3 -.2,.7 .9,-.1 z m -1.3,-1.6 -.2,.9 h 1.7 l .6,-.4 .1,-1 z m 2.8,-3 -.3,1.9 1.2,-.5 .1,-1.4 z m 58.3,31.9 -2,.3 -.4,1.3 1.3,1.7 z"/><path class="gm-usmap-state" d="m 355.3,143.7 -51,-5.3 -57.3,-7.9 -2,10.7 -8.5,54.8 -3.3,21.9 32.1,4.8 44.9,5.7 37.5,3.4 3.7,-44.2 z"/><a href="?pick=NYY" target="_top"><g class="gm-map-marker" transform="translate(823,188)"><g class="pop-wrap" style="--i:0"><ellipse class="pin-shadow" cx="0" cy="2" rx="7" ry="2.3" fill="rgba(0,0,0,.4)"/><g class="pin-float"><circle class="pin-ring" cx="0" cy="-18" r="10" fill="#0C2340" opacity=".55"/><path class="pin-body" d="M0,0 C-7,-9 -10,-13 -10,-18 C-10,-24.5 -5.5,-29.5 0,-29.5 C5.5,-29.5 10,-24.5 10,-18 C10,-13 7,-9 0,0 Z" fill="#0C2340" stroke="#fff" stroke-width="1.4"/><circle cx="0" cy="-18" r="6.2" fill="#fff"/><text class="pin-code" x="0" y="-15.3">NY</text><g class="pin-name" transform="translate(0,-40)"><rect x="-49.6" y="-11" width="99.2" height="20" rx="6"/><text x="0" y="3">New York Yankees</text></g></g></g><title>New York Yankees</title></g></a><a href="?pick=NYM" target="_top"><g class="gm-map-marker" transform="translate(838,202)"><g class="pop-wrap" style="--i:1"><ellipse class="pin-shadow" cx="0" cy="2" rx="7" ry="2.3" fill="rgba(0,0,0,.4)"/><g class="pin-float"><circle class="pin-ring" cx="0" cy="-18" r="10" fill="#002D72" opacity=".55"/><path class="pin-body" d="M0,0 C-7,-9 -10,-13 -10,-18 C-10,-24.5 -5.5,-29.5 0,-29.5 C5.5,-29.5 10,-24.5 10,-18 C10,-13 7,-9 0,0 Z" fill="#002D72" stroke="#fff" stroke-width="1.4"/><circle cx="0" cy="-18" r="6.2" fill="#fff"/><text class="pin-code" x="0" y="-15.3">NY</text><g class="pin-name" transform="translate(0,-40)"><rect x="-40.3" y="-11" width="80.6" height="20" rx="6"/><text x="0" y="3">New York Mets</text></g></g></g><title>New York Mets</title></g></a><a href="?pick=BOS" target="_top"><g class="gm-map-marker" transform="translate(868,155)"><g class="pop-wrap" style="--i:2"><ellipse class="pin-shadow" cx="0" cy="2" rx="7" ry="2.3" fill="rgba(0,0,0,.4)"/><g class="pin-float"><circle class="pin-ring" cx="0" cy="-18" r="10" fill="#BD3039" opacity=".55"/><path class="pin-body" d="M0,0 C-7,-9 -10,-13 -10,-18 C-10,-24.5 -5.5,-29.5 0,-29.5 C5.5,-29.5 10,-24.5 10,-18 C10,-13 7,-9 0,0 Z" fill="#BD3039" stroke="#fff" stroke-width="1.4"/><circle cx="0" cy="-18" r="6.2" fill="#fff"/><text class="pin-code" x="0" y="-15.3">BO</text><g class="pin-name" transform="translate(0,-40)"><rect x="-43.4" y="-11" width="86.8" height="20" rx="6"/><text x="0" y="3">Boston Red Sox</text></g></g></g><title>Boston Red Sox</title></g></a><a href="?pick=TOR" target="_top"><g class="gm-map-marker" transform="translate(778,92)"><g class="pop-wrap" style="--i:3"><ellipse class="pin-shadow" cx="0" cy="2" rx="7" ry="2.3" fill="rgba(0,0,0,.4)"/><g class="pin-float"><circle class="pin-ring" cx="0" cy="-18" r="10" fill="#134A8E" opacity=".55"/><path class="pin-body" d="M0,0 C-7,-9 -10,-13 -10,-18 C-10,-24.5 -5.5,-29.5 0,-29.5 C5.5,-29.5 10,-24.5 10,-18 C10,-13 7,-9 0,0 Z" fill="#134A8E" stroke="#fff" stroke-width="1.4"/><circle cx="0" cy="-18" r="6.2" fill="#fff"/><text class="pin-code" x="0" y="-15.3">TO</text><g class="pin-name" transform="translate(0,-40)"><rect x="-52.7" y="-11" width="105.4" height="20" rx="6"/><text x="0" y="3">Toronto Blue Jays</text></g></g></g><title>Toronto Blue Jays</title></g></a><a href="?pick=TBR" target="_top"><g class="gm-map-marker" transform="translate(690,470)"><g class="pop-wrap" style="--i:4"><ellipse class="pin-shadow" cx="0" cy="2" rx="7" ry="2.3" fill="rgba(0,0,0,.4)"/><g class="pin-float"><circle class="pin-ring" cx="0" cy="-18" r="10" fill="#092C5C" opacity=".55"/><path class="pin-body" d="M0,0 C-7,-9 -10,-13 -10,-18 C-10,-24.5 -5.5,-29.5 0,-29.5 C5.5,-29.5 10,-24.5 10,-18 C10,-13 7,-9 0,0 Z" fill="#092C5C" stroke="#fff" stroke-width="1.4"/><circle cx="0" cy="-18" r="6.2" fill="#fff"/><text class="pin-code" x="0" y="-15.3">TB</text><g class="pin-name" transform="translate(0,-40)"><rect x="-43.4" y="-11" width="86.8" height="20" rx="6"/><text x="0" y="3">Tampa Bay Rays</text></g></g></g><title>Tampa Bay Rays</title></g></a><a href="?pick=BAL" target="_top"><g class="gm-map-marker" transform="translate(800,248)"><g class="pop-wrap" style="--i:5"><ellipse class="pin-shadow" cx="0" cy="2" rx="7" ry="2.3" fill="rgba(0,0,0,.4)"/><g class="pin-float"><circle class="pin-ring" cx="0" cy="-18" r="10" fill="#DF4601" opacity=".55"/><path class="pin-body" d="M0,0 C-7,-9 -10,-13 -10,-18 C-10,-24.5 -5.5,-29.5 0,-29.5 C5.5,-29.5 10,-24.5 10,-18 C10,-13 7,-9 0,0 Z" fill="#DF4601" stroke="#fff" stroke-width="1.4"/><circle cx="0" cy="-18" r="6.2" fill="#fff"/><text class="pin-code" x="0" y="-15.3">BA</text><g class="pin-name" transform="translate(0,-40)"><rect x="-52.7" y="-11" width="105.4" height="20" rx="6"/><text x="0" y="3">Baltimore Orioles</text></g></g></g><title>Baltimore Orioles</title></g></a><a href="?pick=CLE" target="_top"><g class="gm-map-marker" transform="translate(700,215)"><g class="pop-wrap" style="--i:6"><ellipse class="pin-shadow" cx="0" cy="2" rx="7" ry="2.3" fill="rgba(0,0,0,.4)"/><g class="pin-float"><circle class="pin-ring" cx="0" cy="-18" r="10" fill="#00385D" opacity=".55"/><path class="pin-body" d="M0,0 C-7,-9 -10,-13 -10,-18 C-10,-24.5 -5.5,-29.5 0,-29.5 C5.5,-29.5 10,-24.5 10,-18 C10,-13 7,-9 0,0 Z" fill="#00385D" stroke="#fff" stroke-width="1.4"/><circle cx="0" cy="-18" r="6.2" fill="#fff"/><text class="pin-code" x="0" y="-15.3">CL</text><g class="pin-name" transform="translate(0,-40)"><rect x="-58.9" y="-11" width="117.8" height="20" rx="6"/><text x="0" y="3">Cleveland Guardians</text></g></g></g><title>Cleveland Guardians</title></g></a><a href="?pick=MIN" target="_top"><g class="gm-map-marker" transform="translate(515,120)"><g class="pop-wrap" style="--i:7"><ellipse class="pin-shadow" cx="0" cy="2" rx="7" ry="2.3" fill="rgba(0,0,0,.4)"/><g class="pin-float"><circle class="pin-ring" cx="0" cy="-18" r="10" fill="#002B5C" opacity=".55"/><path class="pin-body" d="M0,0 C-7,-9 -10,-13 -10,-18 C-10,-24.5 -5.5,-29.5 0,-29.5 C5.5,-29.5 10,-24.5 10,-18 C10,-13 7,-9 0,0 Z" fill="#002B5C" stroke="#fff" stroke-width="1.4"/><circle cx="0" cy="-18" r="6.2" fill="#fff"/><text class="pin-code" x="0" y="-15.3">MI</text><g class="pin-name" transform="translate(0,-40)"><rect x="-46.5" y="-11" width="93.0" height="20" rx="6"/><text x="0" y="3">Minnesota Twins</text></g></g></g><title>Minnesota Twins</title></g></a><a href="?pick=CHW" target="_top"><g class="gm-map-marker" transform="translate(608,224)"><g class="pop-wrap" style="--i:8"><ellipse class="pin-shadow" cx="0" cy="2" rx="7" ry="2.3" fill="rgba(0,0,0,.4)"/><g class="pin-float"><circle class="pin-ring" cx="0" cy="-18" r="10" fill="#27251F" opacity=".55"/><path class="pin-body" d="M0,0 C-7,-9 -10,-13 -10,-18 C-10,-24.5 -5.5,-29.5 0,-29.5 C5.5,-29.5 10,-24.5 10,-18 C10,-13 7,-9 0,0 Z" fill="#27251F" stroke="#fff" stroke-width="1.4"/><circle cx="0" cy="-18" r="6.2" fill="#fff"/><text class="pin-code" x="0" y="-15.3">CH</text><g class="pin-name" transform="translate(0,-40)"><rect x="-52.7" y="-11" width="105.4" height="20" rx="6"/><text x="0" y="3">Chicago White Sox</text></g></g></g><title>Chicago White Sox</title></g></a><a href="?pick=DET" target="_top"><g class="gm-map-marker" transform="translate(655,172)"><g class="pop-wrap" style="--i:9"><ellipse class="pin-shadow" cx="0" cy="2" rx="7" ry="2.3" fill="rgba(0,0,0,.4)"/><g class="pin-float"><circle class="pin-ring" cx="0" cy="-18" r="10" fill="#0C2340" opacity=".55"/><path class="pin-body" d="M0,0 C-7,-9 -10,-13 -10,-18 C-10,-24.5 -5.5,-29.5 0,-29.5 C5.5,-29.5 10,-24.5 10,-18 C10,-13 7,-9 0,0 Z" fill="#0C2340" stroke="#fff" stroke-width="1.4"/><circle cx="0" cy="-18" r="6.2" fill="#fff"/><text class="pin-code" x="0" y="-15.3">DE</text><g class="pin-name" transform="translate(0,-40)"><rect x="-43.4" y="-11" width="86.8" height="20" rx="6"/><text x="0" y="3">Detroit Tigers</text></g></g></g><title>Detroit Tigers</title></g></a><a href="?pick=KCR" target="_top"><g class="gm-map-marker" transform="translate(498,282)"><g class="pop-wrap" style="--i:10"><ellipse class="pin-shadow" cx="0" cy="2" rx="7" ry="2.3" fill="rgba(0,0,0,.4)"/><g class="pin-float"><circle class="pin-ring" cx="0" cy="-18" r="10" fill="#004687" opacity=".55"/><path class="pin-body" d="M0,0 C-7,-9 -10,-13 -10,-18 C-10,-24.5 -5.5,-29.5 0,-29.5 C5.5,-29.5 10,-24.5 10,-18 C10,-13 7,-9 0,0 Z" fill="#004687" stroke="#fff" stroke-width="1.4"/><circle cx="0" cy="-18" r="6.2" fill="#fff"/><text class="pin-code" x="0" y="-15.3">KC</text><g class="pin-name" transform="translate(0,-40)"><rect x="-55.8" y="-11" width="111.6" height="20" rx="6"/><text x="0" y="3">Kansas City Royals</text></g></g></g><title>Kansas City Royals</title></g></a><a href="?pick=HOU" target="_top"><g class="gm-map-marker" transform="translate(455,500)"><g class="pop-wrap" style="--i:11"><ellipse class="pin-shadow" cx="0" cy="2" rx="7" ry="2.3" fill="rgba(0,0,0,.4)"/><g class="pin-float"><circle class="pin-ring" cx="0" cy="-18" r="10" fill="#002D62" opacity=".55"/><path class="pin-body" d="M0,0 C-7,-9 -10,-13 -10,-18 C-10,-24.5 -5.5,-29.5 0,-29.5 C5.5,-29.5 10,-24.5 10,-18 C10,-13 7,-9 0,0 Z" fill="#002D62" stroke="#fff" stroke-width="1.4"/><circle cx="0" cy="-18" r="6.2" fill="#fff"/><text class="pin-code" x="0" y="-15.3">HO</text><g class="pin-name" transform="translate(0,-40)"><rect x="-43.4" y="-11" width="86.8" height="20" rx="6"/><text x="0" y="3">Houston Astros</text></g></g></g><title>Houston Astros</title></g></a><a href="?pick=SEA" target="_top"><g class="gm-map-marker" transform="translate(112,65)"><g class="pop-wrap" style="--i:12"><ellipse class="pin-shadow" cx="0" cy="2" rx="7" ry="2.3" fill="rgba(0,0,0,.4)"/><g class="pin-float"><circle class="pin-ring" cx="0" cy="-18" r="10" fill="#0C2C56" opacity=".55"/><path class="pin-body" d="M0,0 C-7,-9 -10,-13 -10,-18 C-10,-24.5 -5.5,-29.5 0,-29.5 C5.5,-29.5 10,-24.5 10,-18 C10,-13 7,-9 0,0 Z" fill="#0C2C56" stroke="#fff" stroke-width="1.4"/><circle cx="0" cy="-18" r="6.2" fill="#fff"/><text class="pin-code" x="0" y="-15.3">SE</text><g class="pin-name" transform="translate(0,-40)"><rect x="-49.6" y="-11" width="99.2" height="20" rx="6"/><text x="0" y="3">Seattle Mariners</text></g></g></g><title>Seattle Mariners</title></g></a><a href="?pick=TEX" target="_top"><g class="gm-map-marker" transform="translate(410,400)"><g class="pop-wrap" style="--i:13"><ellipse class="pin-shadow" cx="0" cy="2" rx="7" ry="2.3" fill="rgba(0,0,0,.4)"/><g class="pin-float"><circle class="pin-ring" cx="0" cy="-18" r="10" fill="#003278" opacity=".55"/><path class="pin-body" d="M0,0 C-7,-9 -10,-13 -10,-18 C-10,-24.5 -5.5,-29.5 0,-29.5 C5.5,-29.5 10,-24.5 10,-18 C10,-13 7,-9 0,0 Z" fill="#003278" stroke="#fff" stroke-width="1.4"/><circle cx="0" cy="-18" r="6.2" fill="#fff"/><text class="pin-code" x="0" y="-15.3">TE</text><g class="pin-name" transform="translate(0,-40)"><rect x="-40.3" y="-11" width="80.6" height="20" rx="6"/><text x="0" y="3">Texas Rangers</text></g></g></g><title>Texas Rangers</title></g></a><a href="?pick=LAA" target="_top"><g class="gm-map-marker" transform="translate(108,332)"><g class="pop-wrap" style="--i:14"><ellipse class="pin-shadow" cx="0" cy="2" rx="7" ry="2.3" fill="rgba(0,0,0,.4)"/><g class="pin-float"><circle class="pin-ring" cx="0" cy="-18" r="10" fill="#BA0021" opacity=".55"/><path class="pin-body" d="M0,0 C-7,-9 -10,-13 -10,-18 C-10,-24.5 -5.5,-29.5 0,-29.5 C5.5,-29.5 10,-24.5 10,-18 C10,-13 7,-9 0,0 Z" fill="#BA0021" stroke="#fff" stroke-width="1.4"/><circle cx="0" cy="-18" r="6.2" fill="#fff"/><text class="pin-code" x="0" y="-15.3">LA</text><g class="pin-name" transform="translate(0,-40)"><rect x="-55.8" y="-11" width="111.6" height="20" rx="6"/><text x="0" y="3">Los Angeles Angels</text></g></g></g><title>Los Angeles Angels</title></g></a><a href="?pick=ATH" target="_top"><g class="gm-map-marker" transform="translate(88,218)"><g class="pop-wrap" style="--i:15"><ellipse class="pin-shadow" cx="0" cy="2" rx="7" ry="2.3" fill="rgba(0,0,0,.4)"/><g class="pin-float"><circle class="pin-ring" cx="0" cy="-18" r="10" fill="#003831" opacity=".55"/><path class="pin-body" d="M0,0 C-7,-9 -10,-13 -10,-18 C-10,-24.5 -5.5,-29.5 0,-29.5 C5.5,-29.5 10,-24.5 10,-18 C10,-13 7,-9 0,0 Z" fill="#003831" stroke="#fff" stroke-width="1.4"/><circle cx="0" cy="-18" r="6.2" fill="#fff"/><text class="pin-code" x="0" y="-15.3">AT</text><g class="pin-name" transform="translate(0,-40)"><rect x="-27.9" y="-11" width="55.8" height="20" rx="6"/><text x="0" y="3">Athletics</text></g></g></g><title>Athletics</title></g></a><a href="?pick=ATL" target="_top"><g class="gm-map-marker" transform="translate(710,396)"><g class="pop-wrap" style="--i:16"><ellipse class="pin-shadow" cx="0" cy="2" rx="7" ry="2.3" fill="rgba(0,0,0,.4)"/><g class="pin-float"><circle class="pin-ring" cx="0" cy="-18" r="10" fill="#13274F" opacity=".55"/><path class="pin-body" d="M0,0 C-7,-9 -10,-13 -10,-18 C-10,-24.5 -5.5,-29.5 0,-29.5 C5.5,-29.5 10,-24.5 10,-18 C10,-13 7,-9 0,0 Z" fill="#13274F" stroke="#fff" stroke-width="1.4"/><circle cx="0" cy="-18" r="6.2" fill="#fff"/><text class="pin-code" x="0" y="-15.3">AT</text><g class="pin-name" transform="translate(0,-40)"><rect x="-43.4" y="-11" width="86.8" height="20" rx="6"/><text x="0" y="3">Atlanta Braves</text></g></g></g><title>Atlanta Braves</title></g></a><a href="?pick=PHI" target="_top"><g class="gm-map-marker" transform="translate(812,228)"><g class="pop-wrap" style="--i:17"><ellipse class="pin-shadow" cx="0" cy="2" rx="7" ry="2.3" fill="rgba(0,0,0,.4)"/><g class="pin-float"><circle class="pin-ring" cx="0" cy="-18" r="10" fill="#E81828" opacity=".55"/><path class="pin-body" d="M0,0 C-7,-9 -10,-13 -10,-18 C-10,-24.5 -5.5,-29.5 0,-29.5 C5.5,-29.5 10,-24.5 10,-18 C10,-13 7,-9 0,0 Z" fill="#E81828" stroke="#fff" stroke-width="1.4"/><circle cx="0" cy="-18" r="6.2" fill="#fff"/><text class="pin-code" x="0" y="-15.3">PH</text><g class="pin-name" transform="translate(0,-40)"><rect x="-65.1" y="-11" width="130.2" height="20" rx="6"/><text x="0" y="3">Philadelphia Phillies</text></g></g></g><title>Philadelphia Phillies</title></g></a><a href="?pick=MIA" target="_top"><g class="gm-map-marker" transform="translate(756,536)"><g class="pop-wrap" style="--i:18"><ellipse class="pin-shadow" cx="0" cy="2" rx="7" ry="2.3" fill="rgba(0,0,0,.4)"/><g class="pin-float"><circle class="pin-ring" cx="0" cy="-18" r="10" fill="#00A3E0" opacity=".55"/><path class="pin-body" d="M0,0 C-7,-9 -10,-13 -10,-18 C-10,-24.5 -5.5,-29.5 0,-29.5 C5.5,-29.5 10,-24.5 10,-18 C10,-13 7,-9 0,0 Z" fill="#00A3E0" stroke="#fff" stroke-width="1.4"/><circle cx="0" cy="-18" r="6.2" fill="#fff"/><text class="pin-code" x="0" y="-15.3">MI</text><g class="pin-name" transform="translate(0,-40)"><rect x="-40.3" y="-11" width="80.6" height="20" rx="6"/><text x="0" y="3">Miami Marlins</text></g></g></g><title>Miami Marlins</title></g></a><a href="?pick=WSN" target="_top"><g class="gm-map-marker" transform="translate(802,252)"><g class="pop-wrap" style="--i:19"><ellipse class="pin-shadow" cx="0" cy="2" rx="7" ry="2.3" fill="rgba(0,0,0,.4)"/><g class="pin-float"><circle class="pin-ring" cx="0" cy="-18" r="10" fill="#AB0003" opacity=".55"/><path class="pin-body" d="M0,0 C-7,-9 -10,-13 -10,-18 C-10,-24.5 -5.5,-29.5 0,-29.5 C5.5,-29.5 10,-24.5 10,-18 C10,-13 7,-9 0,0 Z" fill="#AB0003" stroke="#fff" stroke-width="1.4"/><circle cx="0" cy="-18" r="6.2" fill="#fff"/><text class="pin-code" x="0" y="-15.3">WS</text><g class="pin-name" transform="translate(0,-40)"><rect x="-62.0" y="-11" width="124.0" height="20" rx="6"/><text x="0" y="3">Washington Nationals</text></g></g></g><title>Washington Nationals</title></g></a><a href="?pick=MIL" target="_top"><g class="gm-map-marker" transform="translate(598,170)"><g class="pop-wrap" style="--i:20"><ellipse class="pin-shadow" cx="0" cy="2" rx="7" ry="2.3" fill="rgba(0,0,0,.4)"/><g class="pin-float"><circle class="pin-ring" cx="0" cy="-18" r="10" fill="#12284B" opacity=".55"/><path class="pin-body" d="M0,0 C-7,-9 -10,-13 -10,-18 C-10,-24.5 -5.5,-29.5 0,-29.5 C5.5,-29.5 10,-24.5 10,-18 C10,-13 7,-9 0,0 Z" fill="#12284B" stroke="#fff" stroke-width="1.4"/><circle cx="0" cy="-18" r="6.2" fill="#fff"/><text class="pin-code" x="0" y="-15.3">MI</text><g class="pin-name" transform="translate(0,-40)"><rect x="-52.7" y="-11" width="105.4" height="20" rx="6"/><text x="0" y="3">Milwaukee Brewers</text></g></g></g><title>Milwaukee Brewers</title></g></a><a href="?pick=CHC" target="_top"><g class="gm-map-marker" transform="translate(590,212)"><g class="pop-wrap" style="--i:21"><ellipse class="pin-shadow" cx="0" cy="2" rx="7" ry="2.3" fill="rgba(0,0,0,.4)"/><g class="pin-float"><circle class="pin-ring" cx="0" cy="-18" r="10" fill="#0E3386" opacity=".55"/><path class="pin-body" d="M0,0 C-7,-9 -10,-13 -10,-18 C-10,-24.5 -5.5,-29.5 0,-29.5 C5.5,-29.5 10,-24.5 10,-18 C10,-13 7,-9 0,0 Z" fill="#0E3386" stroke="#fff" stroke-width="1.4"/><circle cx="0" cy="-18" r="6.2" fill="#fff"/><text class="pin-code" x="0" y="-15.3">CH</text><g class="pin-name" transform="translate(0,-40)"><rect x="-37.2" y="-11" width="74.4" height="20" rx="6"/><text x="0" y="3">Chicago Cubs</text></g></g></g><title>Chicago Cubs</title></g></a><a href="?pick=STL" target="_top"><g class="gm-map-marker" transform="translate(598,300)"><g class="pop-wrap" style="--i:22"><ellipse class="pin-shadow" cx="0" cy="2" rx="7" ry="2.3" fill="rgba(0,0,0,.4)"/><g class="pin-float"><circle class="pin-ring" cx="0" cy="-18" r="10" fill="#C41E3A" opacity=".55"/><path class="pin-body" d="M0,0 C-7,-9 -10,-13 -10,-18 C-10,-24.5 -5.5,-29.5 0,-29.5 C5.5,-29.5 10,-24.5 10,-18 C10,-13 7,-9 0,0 Z" fill="#C41E3A" stroke="#fff" stroke-width="1.4"/><circle cx="0" cy="-18" r="6.2" fill="#fff"/><text class="pin-code" x="0" y="-15.3">ST</text><g class="pin-name" transform="translate(0,-40)"><rect x="-58.9" y="-11" width="117.8" height="20" rx="6"/><text x="0" y="3">St. Louis Cardinals</text></g></g></g><title>St. Louis Cardinals</title></g></a><a href="?pick=CIN" target="_top"><g class="gm-map-marker" transform="translate(672,275)"><g class="pop-wrap" style="--i:23"><ellipse class="pin-shadow" cx="0" cy="2" rx="7" ry="2.3" fill="rgba(0,0,0,.4)"/><g class="pin-float"><circle class="pin-ring" cx="0" cy="-18" r="10" fill="#C6011F" opacity=".55"/><path class="pin-body" d="M0,0 C-7,-9 -10,-13 -10,-18 C-10,-24.5 -5.5,-29.5 0,-29.5 C5.5,-29.5 10,-24.5 10,-18 C10,-13 7,-9 0,0 Z" fill="#C6011F" stroke="#fff" stroke-width="1.4"/><circle cx="0" cy="-18" r="6.2" fill="#fff"/><text class="pin-code" x="0" y="-15.3">CI</text><g class="pin-name" transform="translate(0,-40)"><rect x="-46.5" y="-11" width="93.0" height="20" rx="6"/><text x="0" y="3">Cincinnati Reds</text></g></g></g><title>Cincinnati Reds</title></g></a><a href="?pick=PIT" target="_top"><g class="gm-map-marker" transform="translate(748,222)"><g class="pop-wrap" style="--i:24"><ellipse class="pin-shadow" cx="0" cy="2" rx="7" ry="2.3" fill="rgba(0,0,0,.4)"/><g class="pin-float"><circle class="pin-ring" cx="0" cy="-18" r="10" fill="#27251F" opacity=".55"/><path class="pin-body" d="M0,0 C-7,-9 -10,-13 -10,-18 C-10,-24.5 -5.5,-29.5 0,-29.5 C5.5,-29.5 10,-24.5 10,-18 C10,-13 7,-9 0,0 Z" fill="#27251F" stroke="#fff" stroke-width="1.4"/><circle cx="0" cy="-18" r="6.2" fill="#fff"/><text class="pin-code" x="0" y="-15.3">PI</text><g class="pin-name" transform="translate(0,-40)"><rect x="-55.8" y="-11" width="111.6" height="20" rx="6"/><text x="0" y="3">Pittsburgh Pirates</text></g></g></g><title>Pittsburgh Pirates</title></g></a><a href="?pick=LAD" target="_top"><g class="gm-map-marker" transform="translate(100,340)"><g class="pop-wrap" style="--i:25"><ellipse class="pin-shadow" cx="0" cy="2" rx="7" ry="2.3" fill="rgba(0,0,0,.4)"/><g class="pin-float"><circle class="pin-ring" cx="0" cy="-18" r="10" fill="#005A9C" opacity=".55"/><path class="pin-body" d="M0,0 C-7,-9 -10,-13 -10,-18 C-10,-24.5 -5.5,-29.5 0,-29.5 C5.5,-29.5 10,-24.5 10,-18 C10,-13 7,-9 0,0 Z" fill="#005A9C" stroke="#fff" stroke-width="1.4"/><circle cx="0" cy="-18" r="6.2" fill="#fff"/><text class="pin-code" x="0" y="-15.3">LA</text><g class="pin-name" transform="translate(0,-40)"><rect x="-58.9" y="-11" width="117.8" height="20" rx="6"/><text x="0" y="3">Los Angeles Dodgers</text></g></g></g><title>Los Angeles Dodgers</title></g></a><a href="?pick=SDP" target="_top"><g class="gm-map-marker" transform="translate(68,338)"><g class="pop-wrap" style="--i:26"><ellipse class="pin-shadow" cx="0" cy="2" rx="7" ry="2.3" fill="rgba(0,0,0,.4)"/><g class="pin-float"><circle class="pin-ring" cx="0" cy="-18" r="10" fill="#2F241D" opacity=".55"/><path class="pin-body" d="M0,0 C-7,-9 -10,-13 -10,-18 C-10,-24.5 -5.5,-29.5 0,-29.5 C5.5,-29.5 10,-24.5 10,-18 C10,-13 7,-9 0,0 Z" fill="#2F241D" stroke="#fff" stroke-width="1.4"/><circle cx="0" cy="-18" r="6.2" fill="#fff"/><text class="pin-code" x="0" y="-15.3">SD</text><g class="pin-name" transform="translate(0,-40)"><rect x="-49.6" y="-11" width="99.2" height="20" rx="6"/><text x="0" y="3">San Diego Padres</text></g></g></g><title>San Diego Padres</title></g></a><a href="?pick=SFG" target="_top"><g class="gm-map-marker" transform="translate(58,258)"><g class="pop-wrap" style="--i:27"><ellipse class="pin-shadow" cx="0" cy="2" rx="7" ry="2.3" fill="rgba(0,0,0,.4)"/><g class="pin-float"><circle class="pin-ring" cx="0" cy="-18" r="10" fill="#FD5A1E" opacity=".55"/><path class="pin-body" d="M0,0 C-7,-9 -10,-13 -10,-18 C-10,-24.5 -5.5,-29.5 0,-29.5 C5.5,-29.5 10,-24.5 10,-18 C10,-13 7,-9 0,0 Z" fill="#FD5A1E" stroke="#fff" stroke-width="1.4"/><circle cx="0" cy="-18" r="6.2" fill="#fff"/><text class="pin-code" x="0" y="-15.3">SF</text><g class="pin-name" transform="translate(0,-40)"><rect x="-62.0" y="-11" width="124.0" height="20" rx="6"/><text x="0" y="3">San Francisco Giants</text></g></g></g><title>San Francisco Giants</title></g></a><a href="?pick=ARI" target="_top"><g class="gm-map-marker" transform="translate(190,350)"><g class="pop-wrap" style="--i:28"><ellipse class="pin-shadow" cx="0" cy="2" rx="7" ry="2.3" fill="rgba(0,0,0,.4)"/><g class="pin-float"><circle class="pin-ring" cx="0" cy="-18" r="10" fill="#A71930" opacity=".55"/><path class="pin-body" d="M0,0 C-7,-9 -10,-13 -10,-18 C-10,-24.5 -5.5,-29.5 0,-29.5 C5.5,-29.5 10,-24.5 10,-18 C10,-13 7,-9 0,0 Z" fill="#A71930" stroke="#fff" stroke-width="1.4"/><circle cx="0" cy="-18" r="6.2" fill="#fff"/><text class="pin-code" x="0" y="-15.3">AR</text><g class="pin-name" transform="translate(0,-40)"><rect x="-62.0" y="-11" width="124.0" height="20" rx="6"/><text x="0" y="3">Arizona Diamondbacks</text></g></g></g><title>Arizona Diamondbacks</title></g></a><a href="?pick=COL" target="_top"><g class="gm-map-marker" transform="translate(300,258)"><g class="pop-wrap" style="--i:29"><ellipse class="pin-shadow" cx="0" cy="2" rx="7" ry="2.3" fill="rgba(0,0,0,.4)"/><g class="pin-float"><circle class="pin-ring" cx="0" cy="-18" r="10" fill="#33006F" opacity=".55"/><path class="pin-body" d="M0,0 C-7,-9 -10,-13 -10,-18 C-10,-24.5 -5.5,-29.5 0,-29.5 C5.5,-29.5 10,-24.5 10,-18 C10,-13 7,-9 0,0 Z" fill="#33006F" stroke="#fff" stroke-width="1.4"/><circle cx="0" cy="-18" r="6.2" fill="#fff"/><text class="pin-code" x="0" y="-15.3">CO</text><g class="pin-name" transform="translate(0,-40)"><rect x="-49.6" y="-11" width="99.2" height="20" rx="6"/><text x="0" y="3">Colorado Rockies</text></g></g></g><title>Colorado Rockies</title></g></a></svg></div>"""


def us_map_html() -> str:
    """마커를 클릭하면 ?pick=CODE 로 이동한다 — Home.py가 st.query_params로 읽어서 팀을 선택한다."""
    return _US_MAP_SVG
