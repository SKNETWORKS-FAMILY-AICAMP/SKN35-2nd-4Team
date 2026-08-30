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

import math
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
/* ══════════════════════════════════════════════════════════════════
   야간 스타디움 다크 테마 (2026-08-29 전면 개편)
   조명탑이 켜진 밤 경기장 = 이 서비스의 무대. 모든 표면은 그 위에 뜬
   유리판(글래스모피즘)이고, 팀 컬러가 조명처럼 배경을 물들인다.
   ── 컴포넌트가 전부 var() 를 통해 색을 쓰므로 이 :root 만 바꾸면
      전체 톤이 함께 뒤집힌다(하드코딩 색을 새로 넣지 말 것).
   ══════════════════════════════════════════════════════════════════ */
:root{
  --navy:#0F1B33; --navy-2:#16264A; --navy-soft:#1B2A4A; --team-accent:#4E8FD6; --hero-glow:#1E3E77;
  --ink:#EEF3FF; --muted:#93A2BF; --faint:#64748B;
  --line:rgba(255,255,255,.10); --paper:#080D1A; --card:rgba(255,255,255,.045);
  --card-solid:#111A2E;
  --gold:#FFC94D; --gold-dim:#B8871F;
  --risk:#FF6B6B; --risk-bg:rgba(255,107,107,.14); --gain:#3FD17B; --gain-bg:rgba(63,209,123,.14);
  --warn:#FFC94D; --warn-bg:rgba(255,201,77,.14); --violet:#A78BFA; --violet-bg:rgba(167,139,250,.14);
  --shadow-sm: 0 1px 2px rgba(0,0,0,.35), 0 2px 8px rgba(0,0,0,.28);
  --shadow-md: 0 8px 20px rgba(0,0,0,.42), 0 2px 6px rgba(0,0,0,.3);
  --glass-blur: saturate(150%) blur(14px);
}
html, body, [class*="css"] { font-family: "Pretendard Variable", Pretendard, -apple-system, "Malgun Gothic", sans-serif !important; }

/* Streamlit 기본 크롬 제거 — 우리 topbar 로 대체 */
[data-testid="stSidebar"], [data-testid="stHeader"], [data-testid="stToolbar"], footer { display:none !important; }
.block-container { max-width: 100% !important; padding: 0 0 60px !important; }
#MainMenu { visibility:hidden; }
div[data-testid="stVerticalBlockBorderWrapper"] { gap:0; }
[data-testid="stAppViewContainer"] { animation: gm-fade .35s ease; }
@keyframes gm-fade { from{opacity:0; transform:translateY(4px)} to{opacity:1; transform:translateY(0)} }

/* ── 무대: 밤 경기장 ────────────────────────────────────────────────
   .stApp 자체에 별(반복 radial-gradient) + 팀컬러 조명 + 지평선 글로우를
   깔고, ::after 로 좌우 조명탑 빛기둥을 얹는다. 콘텐츠는 그 위에 뜬다. */
.stApp{
  background:
    radial-gradient(1400px 620px at 50% 104%, rgba(78,143,214,.20) 0%, transparent 62%),
    radial-gradient(900px 460px at 8% -6%, var(--hero-glow) 0%, transparent 58%),
    radial-gradient(900px 460px at 92% -6%, var(--hero-glow) 0%, transparent 58%),
    linear-gradient(180deg,#050912 0%, var(--paper) 42%, #060B17 100%);
  background-attachment: fixed;
  color: var(--ink);
}
.stApp::before{
  content:"";position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:
    radial-gradient(1.4px 1.4px at 12% 18%, rgba(255,255,255,.65), transparent),
    radial-gradient(1.2px 1.2px at 68% 8%,  rgba(255,255,255,.5),  transparent),
    radial-gradient(1.6px 1.6px at 84% 26%, rgba(255,255,255,.6),  transparent),
    radial-gradient(1.1px 1.1px at 33% 32%, rgba(255,255,255,.4),  transparent),
    radial-gradient(1.3px 1.3px at 52% 15%, rgba(255,255,255,.45), transparent),
    radial-gradient(1.1px 1.1px at 22% 6%,  rgba(255,255,255,.4),  transparent);
  animation: gm-twinkle 5.5s ease-in-out infinite;
}
@keyframes gm-twinkle{0%,100%{opacity:.55}50%{opacity:.95}}
/* 좌우 조명탑 빛기둥 — 첨부 영상의 플러드라이트 두 줄기 */
.stApp::after{
  content:"";position:fixed;inset:0 0 auto;height:78vh;pointer-events:none;z-index:0;
  background:
    conic-gradient(from 168deg at 6% 0%,  transparent 0deg, rgba(120,180,255,.13) 8deg, transparent 20deg),
    conic-gradient(from 172deg at 94% 0%, transparent 0deg, rgba(120,180,255,.13) 8deg, transparent 20deg);
  animation: gm-floodbeam 7s ease-in-out infinite;
}
@keyframes gm-floodbeam{0%,100%{opacity:.6}50%{opacity:1}}
[data-testid="stAppViewContainer"]{position:relative;z-index:1}
@media (prefers-reduced-motion: reduce){
  .stApp::before,.stApp::after{animation:none}
}

/* ── 3D 파티클 레이어 ──────────────────────────────────────────────
   perspective 를 컨테이너에 걸고 각 입자를 translate3d 의 Z 축으로 밀어
   진짜 원근을 만든다. 뒤(음수 Z)에 있는 입자는 브라우저가 알아서 작게
   그려주고, 여기에 blur/opacity 를 더해 공기원근까지 흉내낸다.
   pointer-events:none 이라 클릭을 절대 가로채지 않는다. */
.gm-particles{position:fixed;inset:0;pointer-events:none;z-index:0;overflow:hidden;
  perspective:760px;perspective-origin:50% 42%;transform-style:preserve-3d}
.gm-particle{position:absolute;left:var(--x);bottom:-6vh;
  width:var(--s);height:var(--s);border-radius:50%;opacity:0;
  background:radial-gradient(circle at 34% 32%,
    rgba(255,255,255,.98), rgba(168,208,255,.55) 52%, rgba(120,170,255,0) 74%);
  filter:blur(var(--blur));
  animation:gm-particle-rise var(--dur) linear var(--d) infinite}
@keyframes gm-particle-rise{
  0%{opacity:0;transform:translate3d(0,0,var(--z)) scale(.55)}
  10%{opacity:var(--op)}
  85%{opacity:var(--op)}
  100%{opacity:0;transform:translate3d(var(--dx),-112vh,var(--z)) scale(1.25)}}

/* 보케 — 초점이 나간 큰 광원. 아주 느리게 부유하며 깊이를 한 겹 더 만든다 */
.gm-bokeh{position:absolute;left:var(--x);top:var(--y);
  width:var(--s);height:var(--s);border-radius:50%;
  background:radial-gradient(circle,
    color-mix(in srgb, var(--team-accent) 70%, #8FC0FF) 0%, transparent 68%);
  opacity:var(--op);filter:blur(26px);
  animation:gm-bokeh-drift var(--dur) ease-in-out var(--d) infinite}
@keyframes gm-bokeh-drift{
  0%,100%{transform:translate3d(0,0,-260px) scale(1)}
  50%{transform:translate3d(38px,-46px,-160px) scale(1.16)}}

/* 전경 레이어 — 콘텐츠(z-index:1) 위에 뜬다. 클릭은 절대 막지 않는다. */
.gm-particles-front{position:fixed;inset:0;pointer-events:none;z-index:3;overflow:hidden;
  perspective:520px}
.gm-particle-front{position:absolute;left:var(--x);bottom:-12vh;
  width:var(--s);height:var(--s);border-radius:50%;opacity:0;
  background:radial-gradient(circle at 36% 34%,
    rgba(255,255,255,.9), rgba(150,195,255,.4) 55%, transparent 72%);
  filter:blur(7px);
  animation:gm-particle-front var(--dur) linear var(--d) infinite}
@keyframes gm-particle-front{
  0%{opacity:0;transform:translate3d(0,0,90px) scale(.7)}
  14%{opacity:var(--op)}
  80%{opacity:var(--op)}
  100%{opacity:0;transform:translate3d(var(--dx),-125vh,180px) scale(1.5)}}

@media (prefers-reduced-motion: reduce){
  .gm-particles,.gm-particles-front{display:none}
}

/* ── 큰 숫자 등장 연출 ──────────────────────────────────────────────
   주의: 이 CSS 문자열 안에는 꺾쇠괄호를 절대 넣지 말 것(주석 안이라도!).
   st.html() 의 HTML 새니타이저는 CSS 주석을 구분하지 않아서, 꺾쇠가 하나라도
   있으면 스타일 블록 전체를 통째로 폐기한다(실측 확인). 그래서 CSS 카운트업에
   쓰는 @property 규칙(값 타입을 꺾쇠로 표기해야 함)은 이 프로젝트에서 못 쓴다
   — 대신 "아래에서 솟아오르며 선명해지는" 연출로 대체한다. */
.gm-rise{display:inline-block;animation:gm-rise .7s cubic-bezier(.2,.9,.25,1.1) both;
  animation-delay:calc(var(--i,0) * 90ms)}
@keyframes gm-rise{from{opacity:0;transform:translateY(14px) scale(.9);filter:blur(6px)}
  to{opacity:1;transform:translateY(0) scale(1);filter:blur(0)}}
@media (prefers-reduced-motion: reduce){ .gm-rise{animation:none} }

/* ── topbar (st.container(key="topbar")) ── */
.st-key-topbar{background:linear-gradient(180deg,var(--navy-2),var(--navy));padding:14px 26px;
  margin-bottom:0;box-shadow:var(--shadow-md);position:relative;z-index:5;
  border-bottom:3px solid var(--team-accent);transition:background .3s ease,border-color .3s ease}
.st-key-topbar .mt-ab{position:relative;width:38px;height:38px;border-radius:11px;
  background:rgba(255,255,255,.96);color:var(--navy);overflow:hidden;
  display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:800;
  box-shadow:0 2px 8px rgba(0,0,0,.28);letter-spacing:.3px}
.st-key-topbar .mt-ab img{position:absolute;inset:3px;width:calc(100% - 6px);height:calc(100% - 6px);
  object-fit:contain}
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

/* ── 카드 / 배지 / KPI (순수 HTML 문자열이라 중첩 문제 없음) ──
   유리판: 반투명 + backdrop-blur + 상단 1px 하이라이트(빛 받는 모서리). */
.gm-card{background:var(--card);border:1px solid var(--line);border-radius:16px;
  padding:18px 20px;margin-bottom:10px;box-shadow:var(--shadow-sm);
  -webkit-backdrop-filter:var(--glass-blur);backdrop-filter:var(--glass-blur);
  position:relative;overflow:hidden;
  transition:box-shadow .22s, transform .22s, border-color .22s}
.gm-card::before{content:"";position:absolute;top:0;left:14px;right:14px;height:1px;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.35),transparent);pointer-events:none}
.gm-card:hover{box-shadow:var(--shadow-md);transform:translateY(-2px);
  border-color:color-mix(in srgb, var(--team-accent) 45%, transparent)}
.gm-badge{display:inline-block;font-size:10.5px;padding:3px 10px;border-radius:20px;font-weight:700;
  letter-spacing:.2px}
.gm-badge.warn{background:var(--warn-bg);color:var(--warn)}
.gm-badge.gain{background:var(--gain-bg);color:var(--gain)}
.gm-badge.risk{background:var(--risk-bg);color:var(--risk)}
.gm-badge.navy{background:var(--navy-soft);color:var(--navy)}
.gm-badge.violet{background:var(--violet-bg);color:var(--violet)}
.gm-placeholder{background:linear-gradient(180deg,rgba(255,255,255,.05),rgba(255,255,255,.02));
  border:1px dashed rgba(255,255,255,.22);
  border-radius:16px;padding:18px 20px;margin-bottom:10px;display:flex;gap:14px;align-items:flex-start;
  transition:border-color .2s}
.gm-placeholder:hover{border-color:var(--team-accent)}
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
/* stTabPanel 은 flex 아이템이라 flex:0 1 auto 로는 내용 폭(≈324px)까지
   쪼그라든다 — width:100% 를 줘야 max-width 까지 실제로 펼쳐진다(실측 확인). */
.st-key-hero [data-testid="stTabPanel"]{max-width:1040px;width:100%;flex:1 1 auto;
  margin:0 auto;padding-left:0;padding-right:0}

.st-key-hero div[data-testid="stButton"] button{
  background:rgba(255,255,255,.07) !important;border:1px solid rgba(255,255,255,.14) !important;
  border-radius:12px !important;color:#fff !important;padding:12px 8px !important;
  width:100%;white-space:pre-wrap;line-height:1.5;font-size:13px !important;
  transition:all .18s cubic-bezier(.2,.8,.2,1);min-height:64px;box-shadow:0 1px 2px rgba(0,0,0,.12)}
/* 구단 선택 버튼 hover — 팀 컬러 조명이 켜지듯 */
.st-key-hero div[data-testid="stButton"] button:hover{
  background:linear-gradient(160deg,rgba(255,255,255,.16),rgba(255,255,255,.06)) !important;
  color:#fff !important;border-color:var(--gold) !important;
  transform:translateY(-4px) scale(1.02);
  box-shadow:0 14px 28px rgba(0,0,0,.45), 0 0 22px rgba(255,201,77,.35)}
.st-key-hero div[data-testid="stButton"] button:active{transform:translateY(-1px) scale(1)}
.st-key-hero div[data-testid="stButton"] button p{ color:inherit !important; }
.st-key-hero div[data-testid="stButton"] button:hover p{ color:#fff !important; }
.gm-division-label{font-size:11px;color:rgba(255,255,255,.45);letter-spacing:1px;margin:22px 0 9px;
  text-transform:uppercase;font-weight:700;display:flex;align-items:center;gap:8px}
.gm-division-label::after{content:"";flex:1;height:1px;background:rgba(255,255,255,.12)}

/* ── 일반 버튼 — 뒤로가기 등 (다크 유리 알약) ── */
div[data-testid="stButton"] button{
  border-radius:20px !important;border:1px solid var(--line) !important;
  background:rgba(255,255,255,.06) !important;
  -webkit-backdrop-filter:var(--glass-blur);backdrop-filter:var(--glass-blur);
  color:var(--ink) !important;font-size:12.5px !important;padding:6px 16px !important;
  box-shadow:var(--shadow-sm);transition:all .15s;position:relative;overflow:hidden}
div[data-testid="stButton"] button:hover{
  border-color:var(--team-accent) !important;color:#fff !important;
  background:color-mix(in srgb, var(--team-accent) 26%, transparent) !important;
  box-shadow:var(--shadow-md), 0 0 18px color-mix(in srgb, var(--team-accent) 35%, transparent);
  transform:translateY(-1px)}
div[data-testid="stButton"] button:hover p{color:#fff !important}

/* 버튼 hover 시 카드처럼 스치는 광택 — 모든 stButton 공통 */
@keyframes gm-shine{0%{transform:translateX(-140%) skewX(-18deg)}100%{transform:translateX(240%) skewX(-18deg)}}
div[data-testid="stButton"] button::after{content:"";position:absolute;top:0;left:0;width:38%;height:100%;
  background:linear-gradient(115deg,transparent,rgba(255,255,255,.55),transparent);
  transform:translateX(-140%) skewX(-18deg);pointer-events:none}
div[data-testid="stButton"] button:hover::after{animation:gm-shine .7s ease forwards}
@media (prefers-reduced-motion: reduce){
  div[data-testid="stButton"] button:hover::after{animation:none}
}

/* dataframe 라운딩 — 다크 유리판 위에 얹는다 */
[data-testid="stDataFrame"]{border-radius:14px;overflow:hidden;border:1px solid var(--line);
  box-shadow:var(--shadow-sm);background:rgba(255,255,255,.03)}

/* 카드 안 테이블 행 호버 */
.gm-card table tr:hover td{background:rgba(255,255,255,.05)}
.gm-card table td{border-bottom:1px solid var(--line)}
.gm-card table tr:last-child td{border-bottom:none}

/* expander 살짝 카드화 */
[data-testid="stExpander"]{border-radius:16px !important;border:1px solid var(--line) !important;
  box-shadow:var(--shadow-sm);overflow:hidden;background:var(--card) !important;
  -webkit-backdrop-filter:var(--glass-blur);backdrop-filter:var(--glass-blur)}
[data-testid="stExpander"] summary{color:var(--ink) !important}

/* Streamlit 위젯 라벨/캡션이 다크 위에서 묻히지 않게 */
[data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"], .stCaption{
  color:var(--muted) !important}
[data-testid="stMarkdownContainer"] p{color:var(--ink)}
[data-testid="stAlert"]{background:rgba(78,143,214,.12) !important;border:1px solid var(--line) !important;
  border-radius:14px !important;color:var(--ink) !important}

/* ══ 영입 후보 카드 — FIFA 얼티밋팀 스타일 ══ */
.gm-pcard-row{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:6px}
.gm-pcard{position:relative;width:100%;border-radius:16px;padding:14px 12px 12px;overflow:hidden;
  animation:gm-pop .55s cubic-bezier(.2,.9,.25,1.15) both;animation-delay:calc(var(--i,0) * 100ms);
  transition:transform .18s ease, box-shadow .18s ease;border:1px solid rgba(255,255,255,.12)}
.gm-pcard::after{content:"";position:absolute;inset:0;background:
  linear-gradient(115deg,rgba(255,255,255,.16) 0%,rgba(255,255,255,0) 34%);pointer-events:none}
@keyframes gm-pop{from{opacity:0;transform:translateY(22px) scale(.82) rotate(-2deg)}
  to{opacity:1;transform:translateY(0) scale(1) rotate(0)}}

/* 등급 카드 — 첨부 영상의 "어두운 카드 본체 + 등급색으로 빛나는 테두리".
   밝은 메탈 그라디언트를 그대로 쓰면 다크 테마에서 흰 글씨·게이지가 묻혀
   읽기 어려워진다(실측). 본체를 어둡게 깔고 등급색은 테두리·상단 글로우·
   숫자에만 쓰면 대비가 확실히 살아난다. */
.gm-pcard.tier-gold{--tier:#FFC94D;
  background:radial-gradient(120% 80% at 50% -10%, rgba(255,201,77,.30), transparent 62%),
             linear-gradient(165deg,#241B08,#12151F 62%);
  border-color:rgba(255,201,77,.55);
  box-shadow:0 10px 24px rgba(0,0,0,.5), 0 0 22px rgba(255,201,77,.28)}
.gm-pcard.tier-silver{--tier:#C9D6E8;
  background:radial-gradient(120% 80% at 50% -10%, rgba(201,214,232,.24), transparent 62%),
             linear-gradient(165deg,#1A2130,#10141E 62%);
  border-color:rgba(201,214,232,.42);
  box-shadow:0 10px 24px rgba(0,0,0,.5), 0 0 18px rgba(201,214,232,.16)}
.gm-pcard.tier-bronze{--tier:#D89A63;
  background:radial-gradient(120% 80% at 50% -10%, rgba(216,154,99,.24), transparent 62%),
             linear-gradient(165deg,#231710,#11141E 62%);
  border-color:rgba(216,154,99,.42);
  box-shadow:0 10px 24px rgba(0,0,0,.5), 0 0 18px rgba(216,154,99,.18)}
/* 등급색을 실제로 읽히는 곳에만 얹는다 (게이지 색은 .pc-stat-fill 본 규칙에서 처리) */
.gm-pcard .pc-ovr{color:var(--tier,#fff)}
.gm-pcard .pc-avatar{border-color:var(--tier,var(--team-accent))}
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
/* 트랙: 어두운 카드 위에서는 검은 트랙이 사라지므로 밝은 반투명으로 */
.gm-pcard .pc-stat-track{flex:1;height:5px;border-radius:3px;background:rgba(255,255,255,.16);
  overflow:hidden;box-shadow:inset 0 1px 2px rgba(0,0,0,.45)}
/* display:block 이 없으면 span 이 inline 으로 남아 width/height 가 아예 먹지
   않는다 — 게이지가 계속 0px 로 그려지고 있었다(실측으로 확인한 기존 버그). */
.gm-pcard .pc-stat-fill{display:block;height:100%;border-radius:3px;
  background:linear-gradient(90deg,var(--tier,#fff),#fff);
  box-shadow:0 0 6px var(--tier,rgba(255,255,255,.8))}
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

/* 카드 자체에 perspective() 를 transform 함수 체인에 넣어 부모 요소 손대지 않고
   카드 개별 3D 기울임을 준다 — hover 시 살짝 입체적으로 들리는 느낌 */
.gm-pcard:hover{transform:perspective(900px) rotateX(4deg) rotateY(-5deg) translateY(-6px) scale(1.035)}
@media (prefers-reduced-motion: reduce){ .gm-pcard:hover{transform:translateY(-3px)} }

/* ── 레전더리 등급(ovr 80+) — 홀로그래픽 트레이딩카드 느낌 ──
   ::after 는 기존 고정 대각선 하이라이트가 쓰고 있어서, 움직이는 홀로 시트는
   ::before 에 새로 둔다(레전더리 카드에만 존재) */
.gm-pcard.tier-legendary{
  background:linear-gradient(155deg,#1B0F3A,#5B2A9E 30%,#B33FA0 52%,#3FA8C9 74%,#1B0F3A);
  background-size:280% 280%;border-color:rgba(255,255,255,.35);
  animation:gm-pop .55s cubic-bezier(.2,.9,.25,1.15) both,
            gm-holo-drift 6s ease-in-out infinite,
            gm-legend-glow 2.4s ease-in-out infinite;
  animation-delay:calc(var(--i,0) * 100ms), 0s, 0s}
@keyframes gm-holo-drift{0%,100%{background-position:0% 30%}50%{background-position:100% 70%}}
@keyframes gm-legend-glow{
  0%,100%{box-shadow:0 0 0 1px rgba(255,255,255,.22),0 0 20px rgba(179,63,160,.45),0 8px 18px rgba(0,0,0,.4)}
  50%{box-shadow:0 0 0 1px rgba(255,255,255,.4),0 0 36px rgba(63,168,201,.6),0 8px 22px rgba(0,0,0,.45)}}
.gm-pcard.tier-legendary::before{content:"";position:absolute;inset:0;z-index:1;pointer-events:none;
  background:linear-gradient(115deg,transparent 32%,rgba(255,255,255,.65) 46%,rgba(160,255,255,.4) 51%,transparent 66%);
  background-size:260% 260%;mix-blend-mode:overlay;
  animation:gm-holo-sweep 3.4s ease-in-out infinite}
@keyframes gm-holo-sweep{0%{background-position:220% 220%}100%{background-position:-40% -40%}}
.gm-pcard.tier-legendary .pc-rank{background:linear-gradient(120deg,#B33FA0,#3FA8C9);
  box-shadow:0 0 8px rgba(179,63,160,.6)}
@media (prefers-reduced-motion: reduce){
  .gm-pcard.tier-legendary{animation:gm-pop .55s cubic-bezier(.2,.9,.25,1.15) both}
  .gm-pcard.tier-legendary::before{animation:none}
}

/* ══ 선수 프로파일 히어로 카드 (선수 리포트 — 레이더 차트 + 큰 등번호 워터마크) ══ */
.gm-hero-card{position:relative;border-radius:20px;padding:26px 28px;overflow:hidden;color:#fff;
  background:radial-gradient(900px 340px at 12% -10%, var(--team-accent) 0%, var(--navy) 55%),
             linear-gradient(155deg,var(--navy-2),var(--navy));
  box-shadow:0 14px 32px rgba(0,0,0,.28);margin-bottom:12px;
  display:flex;flex-wrap:wrap;gap:22px;align-items:center;
  animation:gm-pop .5s cubic-bezier(.2,.9,.25,1.1) both}
.gm-hero-watermark{position:absolute;right:-8px;bottom:-38px;font-size:168px;font-weight:800;
  color:rgba(255,255,255,.07);line-height:1;letter-spacing:-6px;pointer-events:none;user-select:none}
.gm-hero-id{position:relative;z-index:1;display:flex;flex-direction:column;align-items:flex-start;
  gap:10px;min-width:180px;flex:1}
.gm-hero-avatar{width:84px;height:84px;border-radius:50%;display:flex;align-items:center;justify-content:center;
  font-size:24px;font-weight:800;color:#fff;position:relative;overflow:hidden;
  background:radial-gradient(circle at 35% 28%, rgba(255,255,255,.4), rgba(255,255,255,.08) 65%);
  border:3px solid var(--team-accent);box-shadow:0 4px 14px rgba(0,0,0,.35), inset 0 0 14px rgba(0,0,0,.2)}
.gm-hero-avatar svg{position:absolute;inset:0;width:100%;height:100%;opacity:.4;color:#fff}
.gm-hero-avatar img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;z-index:2}
.gm-hero-avatar .pc-initials{position:relative;z-index:1}
.gm-hero-name{font-size:20px;font-weight:800;text-shadow:0 1px 3px rgba(0,0,0,.3)}
.gm-hero-role{font-size:12px;color:rgba(255,255,255,.65);letter-spacing:.4px;margin-top:-4px}
.gm-hero-ovr-row{display:flex;align-items:baseline;gap:6px}
.gm-hero-ovr{font-size:40px;font-weight:800;line-height:1;text-shadow:0 2px 6px rgba(0,0,0,.35)}
.gm-hero-ovr-label{font-size:10.5px;color:rgba(255,255,255,.55);letter-spacing:1px;text-transform:uppercase}
.gm-hero-percentile{display:flex;align-items:center;gap:7px;font-size:11px;color:rgba(255,255,255,.6)}
.gm-hero-percentile .gm-badge{font-size:9.5px;padding:2px 8px}
.gm-hero-chips{display:flex;gap:6px;flex-wrap:wrap}
.gm-hero-chip{font-size:10.5px;font-weight:800;padding:4px 9px;border-radius:8px;color:#fff;
  background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.18)}
.gm-hero-radar{position:relative;z-index:1;flex-shrink:0;margin-left:auto}
/* 첨부 영상의 카드 모션 — 아주 느리게 좌우로 기울며 떠 있는 느낌.
   과하면 읽기 불편하므로 각도(1.2deg)와 부유 폭(5px)을 작게 잡는다. */
.gm-hero-card{transform-style:preserve-3d;
  animation:gm-pop .5s cubic-bezier(.2,.9,.25,1.1) both, gm-hero-float 9s ease-in-out 1s infinite}
@keyframes gm-hero-float{
  0%,100%{transform:perspective(1200px) rotateY(-1.2deg) rotateX(.5deg) translateY(0)}
  50%    {transform:perspective(1200px) rotateY(1.2deg)  rotateX(-.5deg) translateY(-5px)}}
/* 조명이 카드 표면을 훑고 지나가는 광택 */
.gm-hero-card::after{content:"";position:absolute;inset:0;pointer-events:none;z-index:2;
  background:linear-gradient(115deg,transparent 34%,rgba(255,255,255,.10) 47%,transparent 60%);
  background-size:280% 280%;animation:gm-hero-sheen 7s ease-in-out 1.5s infinite}
@keyframes gm-hero-sheen{0%{background-position:200% 200%}55%,100%{background-position:-60% -60%}}
@media (prefers-reduced-motion: reduce){
  .gm-hero-card{animation:gm-pop .5s cubic-bezier(.2,.9,.25,1.1) both}
  .gm-hero-card::after{animation:none}
}
/* 히어로 하단 스탯 링 묶음 */
.gm-hero-rings{display:flex;gap:14px;flex-wrap:wrap;position:relative;z-index:1;margin-top:2px}
.gm-radar-fill{animation:gm-radar-grow .7s cubic-bezier(.2,.8,.2,1) .1s both;transform-origin:center}
@keyframes gm-radar-grow{from{transform:scale(0);opacity:0}to{transform:scale(1);opacity:1}}
.gm-radar-dot{animation:gm-badge-pop .35s cubic-bezier(.2,.9,.25,1.4) both}

/* ══ 추세 그래프 (선수 리포트) — 실선(실측)이 그려지는 느낌 + 예측 구간 점선 ══ */
.gm-trend-card{background:linear-gradient(155deg,var(--navy-2),var(--navy));border-radius:14px;
  padding:14px 10px 6px;margin-bottom:10px;box-shadow:var(--shadow-sm);overflow:hidden}
.gm-trend-card svg{display:block;width:100%;height:auto}
.gm-trend-line{stroke-dasharray:1400;stroke-dashoffset:1400;animation:gm-trend-draw 1.1s cubic-bezier(.2,.7,.2,1) .1s forwards}
@keyframes gm-trend-draw{to{stroke-dashoffset:0}}
@media (prefers-reduced-motion: reduce){ .gm-trend-line{animation:none;stroke-dashoffset:0} }

/* ══ 다이아몬드 라인업 — 로스터를 그라운드 위에 ══ */
.gm-diamond-wrap{position:relative;border-radius:20px;overflow:hidden;margin-bottom:12px;
  background:radial-gradient(700px 300px at 50% 0%, rgba(78,143,214,.18), transparent 65%),
             linear-gradient(180deg,rgba(10,18,38,.85) 0%, rgba(6,11,23,.6) 100%);
  border:1px solid var(--line);box-shadow:var(--shadow-md);
  padding:6px 6px 2px}
.gm-diamond{display:block;width:100%;height:auto}
.gm-dia-node{opacity:0;animation:gm-dia-in .5s cubic-bezier(.2,.9,.25,1.2) forwards;
  animation-delay:var(--d,0s);transform-origin:center;cursor:default}
@keyframes gm-dia-in{from{opacity:0;transform:translateY(14px) scale(.7)}
  to{opacity:1;transform:translateY(0) scale(1)}}
/* SVG 필터 참조(filter:url(#...)) 대신 drop-shadow 로 글로우를 준다 — 참조
   id 없이도 같은 효과가 나고, SVG 정의와 CSS 사이의 의존을 만들지 않는다. */
.gm-dia-node:hover{filter:brightness(1.18) drop-shadow(0 0 10px var(--tone))}
/* 위험도가 높은 선수는 링이 계속 맥동한다 — 그라운드를 훑으면 바로 눈에 띈다 */
.gm-dia-pulse-hot .gm-dia-halo{animation:gm-dia-ring 1.5s ease-out infinite;
  animation-delay:calc(var(--d,0s) + .3s)}
.gm-dia-pulse-warm .gm-dia-halo{animation:gm-dia-ring 2.6s ease-out infinite;
  animation-delay:calc(var(--d,0s) + .3s)}
@keyframes gm-dia-ring{0%{r:26;opacity:.75;stroke-width:2.5}100%{r:44;opacity:0;stroke-width:.5}}
@media (prefers-reduced-motion: reduce){
  .gm-dia-node{opacity:1;animation:none}
  .gm-dia-pulse-hot .gm-dia-halo,.gm-dia-pulse-warm .gm-dia-halo{animation:none}
}

/* ══ 원형 게이지 ══ */
.gm-ring{position:relative;display:inline-flex;align-items:center;justify-content:center}
.gm-ring svg{display:block}
.gm-ring-fill{stroke-dashoffset:var(--circ);
  animation:gm-ring-draw 1.1s cubic-bezier(.2,.8,.2,1) .2s forwards;
  filter:drop-shadow(0 0 6px var(--tone))}
@keyframes gm-ring-draw{to{stroke-dashoffset:var(--off)}}
.gm-ring-mid{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;
  justify-content:center;gap:1px;pointer-events:none}
.gm-ring-v{font-size:21px;font-weight:800;color:var(--ink);line-height:1;
  font-variant-numeric:tabular-nums}
.gm-ring-l{font-size:9px;font-weight:700;color:var(--muted);letter-spacing:.6px;text-transform:uppercase}
@media (prefers-reduced-motion: reduce){ .gm-ring-fill{animation:none;stroke-dashoffset:var(--off)} }

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
.gm-vs-name{font-size:14.5px;font-weight:800;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
  max-width:100%}
/* align-items:flex-end(.home) 은 자식을 stretch 하지 않아 overflow:hidden 이
   콘텐츠 폭에 안 걸리는 문제가 있었다 — max-width:100% 로 그리드 트랙 폭에 강제로 고정 */
.gm-vs-tag{font-size:10px;color:rgba(255,255,255,.55);letter-spacing:.5px;text-transform:uppercase}
.gm-vs-mid{display:flex;flex-direction:column;align-items:center;gap:2px}
.gm-vs-bolt{font-size:11px;font-weight:800;color:rgba(255,255,255,.5);letter-spacing:1px}
.gm-vs-bar{width:74px;height:7px;border-radius:4px;overflow:hidden;background:rgba(255,255,255,.15);
  display:flex}
.gm-vs-bar-away{height:100%;background:linear-gradient(90deg,#9CA9C9,#DCE3F2)}
.gm-vs-bar-home{height:100%;background:linear-gradient(90deg,var(--team-accent),#fff)}
.gm-vs-winner{font-size:10px;font-weight:800;color:#FFE9A8;letter-spacing:.3px;margin-top:2px}

/* 야간 경기장 조명 느낌 — 카드 위쪽 모서리에서 은은하게 스치는 두 줄기 빛 */
.gm-vs-card::after{content:"";position:absolute;inset:0;pointer-events:none;z-index:0;
  background:
    linear-gradient(200deg, rgba(255,255,255,.16) 0%, transparent 22%),
    linear-gradient(340deg, rgba(255,255,255,.1) 0%, transparent 28%);
  opacity:.8;animation:gm-floodlight 5s ease-in-out infinite}
@keyframes gm-floodlight{0%,100%{opacity:.55}50%{opacity:1}}

/* 모델 예측 배지 — "실시간 중계"가 아니라 "AI 예측"임을 정직하게 표시하면서도
   레퍼런스의 LIVE 펄스 뱃지 느낌을 그대로 가져온다 */
.gm-live-badge{position:absolute;top:10px;right:14px;z-index:2;display:flex;align-items:center;gap:5px;
  font-size:9px;font-weight:800;letter-spacing:.6px;color:rgba(255,255,255,.85);
  white-space:nowrap}  /* nowrap 없으면 "AI 예측"이 좁은 칸에서 두 줄로 쪼개진다 */
.gm-live-dot{width:6px;height:6px;border-radius:50%;background:#3FD17B;flex-shrink:0;
  box-shadow:0 0 0 0 rgba(63,209,123,.6);animation:gm-live-pulse 1.6s ease-out infinite}
@keyframes gm-live-pulse{0%{box-shadow:0 0 0 0 rgba(63,209,123,.55)}70%{box-shadow:0 0 0 7px rgba(63,209,123,0)}
  100%{box-shadow:0 0 0 0 rgba(63,209,123,0)}}
@media (prefers-reduced-motion: reduce){
  .gm-vs-card::after{animation:none}
  .gm-live-dot{animation:none}
}

/* 좁은 화면(모바일 폭)에서는 3컬럼 그리드가 팀명을 겹치게 만들어 세로 스택으로 전환 */
@media (max-width: 480px){
  .gm-vs-row{grid-template-columns:1fr;justify-items:center;text-align:center;gap:6px}
  .gm-vs-team.away, .gm-vs-team.home{align-items:center;text-align:center}
  .gm-vs-name{white-space:normal;max-width:100%}
  .gm-live-badge{position:static;justify-content:center;margin-bottom:6px}
}

/* ══ 승부예측 챌린지 — 사용자가 먼저 찍고, 그 다음 AI 예측이 열린다 ══
   네이버 스포츠 승부예측처럼 "내 선택 → 결과 공개" 순서를 지키는 게 핵심.
   AI 확률을 먼저 보여주면 사용자가 그걸 따라 찍게 되어 재미가 사라진다. */
.gm-pred-board{display:flex;align-items:center;gap:18px;flex-wrap:wrap;
  background:radial-gradient(600px 200px at 12% -30%, rgba(255,201,77,.18), transparent 62%),
             var(--card);
  border:1px solid rgba(255,201,77,.28);border-radius:18px;padding:16px 22px;margin-bottom:14px;
  -webkit-backdrop-filter:var(--glass-blur);backdrop-filter:var(--glass-blur);
  box-shadow:var(--shadow-md)}
.gm-pred-board-stat{display:flex;flex-direction:column;gap:2px;min-width:78px}
.gm-pred-board-v{font-size:26px;font-weight:800;line-height:1;color:var(--gold);
  font-variant-numeric:tabular-nums;text-shadow:0 0 16px rgba(255,201,77,.45)}
.gm-pred-board-l{font-size:10.5px;font-weight:700;color:var(--muted);letter-spacing:.6px}
.gm-pred-board-msg{flex:1;min-width:200px;font-size:13px;color:var(--ink);line-height:1.6}
.gm-pred-streak{font-size:11px;font-weight:800;color:var(--gold);
  background:rgba(255,201,77,.14);border:1px solid rgba(255,201,77,.3);
  border-radius:20px;padding:4px 12px;white-space:nowrap}

/* 경기 카드 */
.gm-pred-card{position:relative;border-radius:18px;padding:14px 18px 6px;margin-bottom:4px;overflow:hidden;
  background:linear-gradient(150deg,rgba(255,255,255,.06),rgba(255,255,255,.02));
  border:1px solid var(--line);box-shadow:var(--shadow-sm);
  animation:gm-pop .5s cubic-bezier(.2,.9,.25,1.15) both;animation-delay:calc(var(--i,0) * 60ms)}
.gm-pred-card.picked{border-color:color-mix(in srgb, var(--gold) 45%, transparent)}
.gm-pred-date{font-size:10.5px;font-weight:800;color:var(--faint);letter-spacing:1px;margin-bottom:8px;
  display:flex;align-items:center;gap:8px}
.gm-pred-date::after{content:"";flex:1;height:1px;background:var(--line)}
.gm-pred-q{font-size:12.5px;color:var(--muted);text-align:center;margin:2px 0 8px;font-weight:700}
.gm-pred-vs{display:flex;align-items:center;justify-content:center;gap:14px;margin-bottom:6px}
.gm-pred-side{flex:1;display:flex;align-items:center;justify-content:center;gap:9px;
  font-size:14px;font-weight:800;color:var(--ink)}
.gm-pred-side.away{justify-content:flex-end}
.gm-pred-side.home{justify-content:flex-start}
/* 구단 로고 — 어두운 배경에서 묻히는 로고가 있어 밝은 원판 위에 올린다 */
.gm-pred-logo{width:32px;height:32px;border-radius:50%;flex-shrink:0;
  background:rgba(255,255,255,.92);padding:3px;object-fit:contain;
  box-shadow:0 2px 8px rgba(0,0,0,.35);transition:transform .18s}
.gm-pred-card:hover .gm-pred-logo{transform:scale(1.08)}
.gm-pred-bolt{font-size:11px;font-weight:800;color:var(--faint);letter-spacing:1.4px;
  animation:gm-bolt-glow 2.6s ease-in-out infinite}
@keyframes gm-bolt-glow{0%,100%{opacity:.5;text-shadow:none}
  50%{opacity:1;text-shadow:0 0 12px var(--team-accent)}}

/* 공개 연출 — 카드가 뒤집히듯 열린다 */
.gm-pred-reveal{animation:gm-flip-in .62s cubic-bezier(.2,.85,.25,1.05) both;transform-origin:center}
@keyframes gm-flip-in{
  0%{opacity:0;transform:perspective(1000px) rotateX(-72deg) scale(.94)}
  62%{opacity:1;transform:perspective(1000px) rotateX(8deg) scale(1.01)}
  100%{opacity:1;transform:perspective(1000px) rotateX(0) scale(1)}}

/* 양쪽으로 갈라지는 확률 바 — 가운데(50%)에서 실제 확률까지 벌어진다 */
.gm-pred-bar{position:relative;height:26px;border-radius:9px;overflow:hidden;display:flex;
  background:rgba(255,255,255,.07);border:1px solid var(--line);margin:8px 0 6px}
.gm-pred-bar-away,.gm-pred-bar-home{height:100%;display:flex;align-items:center;
  font-size:11px;font-weight:800;color:#fff;
  animation:gm-pred-split .95s cubic-bezier(.2,.85,.2,1) .12s both}
.gm-pred-bar-away{justify-content:flex-start;padding-left:9px;
  background:linear-gradient(90deg,var(--away-c,#5B7FB9),rgba(91,127,185,.55))}
.gm-pred-bar-home{justify-content:flex-end;padding-right:9px;
  background:linear-gradient(90deg,rgba(214,110,110,.55),var(--home-c,#D66E6E))}
@keyframes gm-pred-split{from{width:50%}}
.gm-pred-bar-mid{position:absolute;left:50%;top:0;bottom:0;width:2px;
  background:rgba(255,255,255,.5);transform:translateX(-1px);z-index:2}

/* 판정 배지 */
.gm-pred-verdict{display:flex;align-items:center;gap:9px;flex-wrap:wrap;
  padding:9px 12px;border-radius:12px;margin:6px 0 4px;font-size:12.5px;font-weight:700}
.gm-pred-verdict.hit{background:rgba(63,209,123,.13);border:1px solid rgba(63,209,123,.4);color:var(--gain)}
.gm-pred-verdict.miss{background:rgba(255,201,77,.12);border:1px solid rgba(255,201,77,.36);color:var(--warn)}
.gm-pred-verdict-icon{font-size:17px;animation:gm-verdict-pop .55s cubic-bezier(.2,.9,.25,1.5) .3s both}
@keyframes gm-verdict-pop{from{opacity:0;transform:scale(0) rotate(-40deg)}
  to{opacity:1;transform:scale(1) rotate(0)}}
.gm-pred-mine{font-weight:800;color:var(--ink)}

/* 4개 모델 개별 투표 칩 — "왜 이렇게 예측했나"를 한 줄로 */
.gm-pred-models{display:flex;gap:6px;flex-wrap:wrap;align-items:center;
  padding:6px 0 10px;font-size:10px;color:var(--muted)}
.gm-pred-chip{font-size:9.5px;font-weight:800;padding:3px 8px;border-radius:7px;
  border:1px solid var(--line);background:rgba(255,255,255,.05);color:var(--muted);
  animation:gm-badge-pop .4s cubic-bezier(.2,.9,.25,1.4) both;
  animation-delay:calc(.45s + var(--j,0) * 70ms)}
.gm-pred-chip.agree{border-color:rgba(63,209,123,.45);color:var(--gain);background:rgba(63,209,123,.1)}
.gm-pred-conf{font-size:10px;font-weight:800;padding:3px 9px;border-radius:7px;
  background:rgba(255,255,255,.07);border:1px solid var(--line);color:var(--ink);white-space:nowrap}

/* 선택 버튼을 카드와 한 몸으로 */
.st-key-predpick div[data-testid="stButton"] button{width:100%;min-height:46px;
  border-radius:12px !important;font-size:13px !important;font-weight:800 !important;
  background:rgba(255,255,255,.05) !important}
.st-key-predpick div[data-testid="stButton"] button:hover{
  border-color:var(--gold) !important;
  background:rgba(255,201,77,.16) !important;
  box-shadow:0 8px 20px rgba(0,0,0,.4), 0 0 20px rgba(255,201,77,.3);
  transform:translateY(-2px) scale(1.015)}
@media (prefers-reduced-motion: reduce){
  .gm-pred-reveal,.gm-pred-bar-away,.gm-pred-bar-home,.gm-pred-verdict-icon,.gm-pred-chip{animation:none}
  .gm-pred-bolt{animation:none}
}

.gm-icon{display:inline-block;vertical-align:-.14em;flex-shrink:0}

/* ══ 전력 로스터 리스트 ══
   st.dataframe 은 숫자를 소수점 15자리까지 뿌리고 위험도가 색으로 안 읽힌다.
   전력/이탈위험을 막대로 만들어 훑기만 해도 순위가 보이게 한다. */
.gm-roster{display:flex;flex-direction:column;gap:5px;margin-bottom:8px}
.gm-roster-head,.gm-roster-row{display:grid;
  grid-template-columns:26px minmax(96px,1.5fr) 52px 1.25fr 1.25fr minmax(96px,1fr);
  gap:11px;align-items:center}
.gm-roster-head{padding:0 13px 3px;font-size:10px;font-weight:800;color:var(--faint);letter-spacing:.6px}
.gm-roster-row{padding:9px 13px;border-radius:12px;background:rgba(255,255,255,.035);
  border:1px solid var(--line);transition:background .16s, border-color .16s, transform .16s;
  animation:gm-pop .42s cubic-bezier(.2,.9,.25,1.15) both;animation-delay:calc(var(--i,0) * 32ms)}
.gm-roster-row:hover{background:rgba(255,255,255,.075);transform:translateX(3px);
  border-color:color-mix(in srgb, var(--team-accent) 45%, transparent)}
.gm-roster-rank{font-size:11px;font-weight:800;color:var(--faint);text-align:center;
  font-variant-numeric:tabular-nums}
.gm-roster-name{display:flex;align-items:center;gap:8px;font-size:13px;font-weight:700;
  color:var(--ink);min-width:0}
.gm-roster-name span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
/* 선수 얼굴 — 로스터를 이름 나열이 아니라 "사람 목록"으로 읽히게 한다 */
.gm-roster-face{width:28px;height:28px;border-radius:50%;flex-shrink:0;object-fit:cover;
  background:linear-gradient(160deg,rgba(255,255,255,.14),rgba(255,255,255,.04));
  border:1.5px solid var(--line);transition:border-color .16s, transform .16s}
.gm-roster-row:hover .gm-roster-face{border-color:var(--team-accent);transform:scale(1.08)}
.gm-roster-role{font-size:9.5px;font-weight:800;padding:3px 7px;border-radius:6px;text-align:center;
  border:1px solid var(--line);color:var(--muted);background:rgba(255,255,255,.05);white-space:nowrap}
.gm-roster-metric{display:flex;align-items:center;gap:7px}
.gm-roster-num{font-size:11.5px;font-weight:800;width:38px;flex-shrink:0;text-align:right;
  font-variant-numeric:tabular-nums;color:var(--ink)}
.gm-roster-track{flex:1;height:6px;border-radius:4px;background:rgba(255,255,255,.10);overflow:hidden}
.gm-roster-fill{display:block;height:100%;border-radius:4px;
  animation:gm-bar-grow .75s cubic-bezier(.2,.8,.2,1) both;animation-delay:calc(var(--i,0) * 32ms + .12s)}
.gm-roster-tag{font-size:10px;font-weight:700;color:var(--muted);overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap}
@media (max-width: 760px){
  .gm-roster-head{display:none}
  .gm-roster-row{grid-template-columns:26px 1fr;gap:6px 10px}
  .gm-roster-row .gm-roster-role{justify-self:start}
}
@media (prefers-reduced-motion: reduce){
  .gm-roster-row,.gm-roster-fill{animation:none}
}

/* ══ 신호군 설명 ("복합 요인"이 왜 복합인지) ══ */
.gm-sig-box{margin-top:8px;padding:9px 11px;border-radius:11px;
  background:rgba(255,255,255,.045);border:1px solid var(--line);text-align:left}
.gm-sig-head{font-size:11.5px;color:var(--muted);line-height:1.55;margin-bottom:7px}
.gm-sig-head b{color:var(--ink)}
.gm-sig{display:flex;align-items:flex-start;gap:7px;font-size:11px;padding:3px 0;
  color:var(--faint)}
.gm-sig-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;margin-top:4px;
  background:rgba(255,255,255,.18);box-shadow:none;transition:background .2s}
.gm-sig-label{flex-shrink:0;font-weight:800;width:52px}
.gm-sig-detail{flex:1;font-variant-numeric:tabular-nums;line-height:1.45;word-break:keep-all}
/* 켜진 신호군만 색과 글로우로 도드라지게 — 훑으면 몇 개가 켜졌는지 바로 보인다 */
.gm-sig.on{color:var(--ink)}
.gm-sig.on .gm-sig-dot{background:var(--risk);box-shadow:0 0 8px var(--risk)}
.gm-sig.on .gm-sig-label{color:var(--risk)}
.gm-sig-note{margin-top:7px;padding-top:6px;border-top:1px solid var(--line);
  font-size:10px;color:var(--faint);line-height:1.6}

/* ══ 이탈위험 TOP3 카드 — 선수 얼굴이 주인공 ══ */
.gm-risk-card{position:relative;overflow:visible;padding-top:38px}
.gm-risk-face{position:absolute;top:-24px;left:50%;transform:translateX(-50%);
  width:62px;height:62px;border-radius:50%;overflow:hidden;
  background:linear-gradient(160deg,rgba(255,255,255,.16),rgba(255,255,255,.05));
  border:2.5px solid var(--tone,var(--risk));
  box-shadow:0 6px 18px rgba(0,0,0,.45), 0 0 16px color-mix(in srgb, var(--tone,red) 45%, transparent);
  transition:transform .2s}
.gm-risk-face img{width:100%;height:100%;object-fit:cover;display:block}
.gm-risk-card:hover .gm-risk-face{transform:translateX(-50%) scale(1.07)}
/* 위험도 색으로 맥동하는 링 — 카드를 훑을 때 시선을 잡는다 */
.gm-risk-face::after{content:"";position:absolute;inset:-2.5px;border-radius:50%;
  border:2px solid var(--tone,var(--risk));
  animation:gm-risk-ring 2.2s ease-out infinite;pointer-events:none}
@keyframes gm-risk-ring{0%{transform:scale(1);opacity:.7}100%{transform:scale(1.5);opacity:0}}
@media (prefers-reduced-motion: reduce){ .gm-risk-face::after{animation:none;opacity:.35} }

/* ══ 순위 변동 임팩트 패널 ══ */
.gm-impact{display:grid;grid-template-columns:1fr auto 1fr;gap:18px;align-items:center;
  border-radius:18px;padding:18px 22px;margin-bottom:10px;
  background:linear-gradient(150deg,rgba(255,255,255,.06),rgba(255,255,255,.02));
  border:1px solid var(--line);box-shadow:var(--shadow-sm)}
.gm-impact-side{text-align:center}
.gm-impact-l{font-size:10.5px;font-weight:800;color:var(--muted);letter-spacing:.6px;margin-bottom:5px}
.gm-impact-v{font-size:30px;font-weight:800;line-height:1;color:var(--ink);
  font-variant-numeric:tabular-nums;letter-spacing:-.8px}
.gm-impact-arrow{display:flex;flex-direction:column;align-items:center;gap:5px;min-width:76px}
.gm-impact-arrow-icon{font-size:20px;color:var(--team-accent);
  animation:gm-arrow-slide 1.8s ease-in-out infinite}
@keyframes gm-arrow-slide{0%,100%{transform:translateX(-3px);opacity:.6}
  50%{transform:translateX(3px);opacity:1}}
.gm-impact-delta{font-size:12px;font-weight:800;padding:3px 10px;border-radius:20px;white-space:nowrap}
.gm-impact-delta.up{background:var(--gain-bg);color:var(--gain)}
.gm-impact-delta.down{background:var(--risk-bg);color:var(--risk)}
.gm-impact-delta.flat{background:rgba(255,255,255,.07);color:var(--muted)}
@media (prefers-reduced-motion: reduce){ .gm-impact-arrow-icon{animation:none} }

/* ══ 모델 카드 그리드 (모델 정보) ══
   원시 dataframe 한 덩어리로는 "어느 모델이 좋은지"가 안 읽힌다. 태스크별로
   묶고, 각 모델의 대표 지표를 큰 숫자 + 게이지로 보여준다. */
.gm-mcard-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(212px,1fr));gap:11px;
  margin-bottom:6px}
.gm-mcard{position:relative;overflow:hidden;border-radius:16px;padding:14px 15px 12px;
  background:linear-gradient(155deg,rgba(255,255,255,.06),rgba(255,255,255,.02));
  border:1px solid var(--line);box-shadow:var(--shadow-sm);
  animation:gm-pop .5s cubic-bezier(.2,.9,.25,1.15) both;animation-delay:calc(var(--i,0) * 55ms);
  transition:transform .2s, box-shadow .2s, border-color .2s}
.gm-mcard:hover{transform:translateY(-3px);box-shadow:var(--shadow-md);
  border-color:color-mix(in srgb, var(--team-accent) 50%, transparent)}
/* 태스크 1위 모델에 왕관 + 금색 테두리 */
.gm-mcard.best{border-color:rgba(255,201,77,.6);
  box-shadow:var(--shadow-md), 0 0 20px rgba(255,201,77,.22)}
.gm-mcard-crown{position:absolute;top:9px;right:10px;color:var(--gold);
  animation:gm-badge-pop .5s cubic-bezier(.2,.9,.25,1.5) .35s both}
.gm-mcard-head{display:flex;align-items:center;gap:6px;margin-bottom:8px;flex-wrap:wrap}
.gm-mcard-name{font-size:13px;font-weight:800;color:var(--ink);word-break:break-all}
.gm-mcard-kind{font-size:9px;font-weight:800;padding:2px 7px;border-radius:6px;
  border:1px solid var(--line);color:var(--muted);background:rgba(255,255,255,.05)}
.gm-mcard-metric{display:flex;align-items:baseline;gap:6px;margin-bottom:3px}
.gm-mcard-v{font-size:27px;font-weight:800;line-height:1;color:var(--ink);
  font-variant-numeric:tabular-nums;letter-spacing:-.6px}
.gm-mcard.best .gm-mcard-v{color:var(--gold);text-shadow:0 0 18px rgba(255,201,77,.4)}
.gm-mcard-mlabel{font-size:10px;font-weight:800;color:var(--muted);letter-spacing:.5px}
.gm-mcard-sub{font-size:10.5px;color:var(--faint);margin-bottom:9px;line-height:1.5}
.gm-mcard-bar{height:6px;border-radius:4px;background:rgba(255,255,255,.10);overflow:hidden}
.gm-mcard-bar-fill{display:block;height:100%;border-radius:4px;
  background:linear-gradient(90deg,var(--team-accent),var(--gold));
  animation:gm-bar-grow .9s cubic-bezier(.2,.8,.2,1) both;animation-delay:calc(var(--i,0) * 55ms + .2s)}
.gm-mcard.best .gm-mcard-bar-fill{background:linear-gradient(90deg,var(--gold),#FFF0C2)}
.gm-task-head{display:flex;align-items:center;gap:9px;margin:16px 0 9px;font-size:12px;
  font-weight:800;color:var(--muted);letter-spacing:.6px}
.gm-task-head::after{content:"";flex:1;height:1px;background:var(--line)}
@media (prefers-reduced-motion: reduce){
  .gm-mcard,.gm-mcard-bar-fill,.gm-mcard.best::before{animation:none}
}

/* ══ 운명의 구단 배정 카드 — 뽑기 결과가 조명 아래 나타나듯 ══ */
.gm-draw-card{position:relative;overflow:hidden;text-align:center;margin:12px auto 6px;max-width:520px;
  border-radius:20px;padding:22px 26px;
  /* -30% 로 두면 그라디언트 대부분이 카드 위로 잘려나가 팀 컬러가 거의 안 보인다.
     0% 로 내리고 반경을 키워서 팀 컬러가 카드 상단을 실제로 물들이게 한다. */
  background:radial-gradient(620px 340px at 50% 0%,
               color-mix(in srgb, var(--team-accent) 85%, transparent), transparent 72%),
             linear-gradient(160deg,rgba(255,255,255,.07),rgba(255,255,255,.02));
  border:1px solid color-mix(in srgb, var(--team-accent) 50%, transparent);
  box-shadow:var(--shadow-md), 0 0 34px color-mix(in srgb, var(--team-accent) 30%, transparent);
  animation:gm-draw-in .7s cubic-bezier(.2,.9,.25,1.15) both}
@keyframes gm-draw-in{
  0%{opacity:0;transform:translateY(18px) scale(.9) rotateX(-24deg)}
  100%{opacity:1;transform:translateY(0) scale(1) rotateX(0)}}
/* 표면을 훑는 조명 */
.gm-draw-card::after{content:"";position:absolute;inset:0;pointer-events:none;
  background:linear-gradient(115deg,transparent 36%,rgba(255,255,255,.16) 48%,transparent 60%);
  background-size:250% 250%;animation:gm-draw-sheen 2.6s ease-in-out .35s infinite}
@keyframes gm-draw-sheen{0%{background-position:200% 200%}60%,100%{background-position:-60% -60%}}
.gm-draw-label{font-size:11px;font-weight:800;letter-spacing:1.6px;color:rgba(255,255,255,.6);
  text-transform:uppercase}
.gm-draw-name{font-size:27px;font-weight:800;color:#fff;margin:6px 0 4px;letter-spacing:-.4px;
  text-shadow:0 2px 14px rgba(0,0,0,.5)}
.gm-draw-sub{font-size:12.5px;font-weight:700;color:rgba(255,255,255,.75)}
@media (prefers-reduced-motion: reduce){
  .gm-draw-card,.gm-draw-card::after{animation:none}
}

/* ══ 예상 순위 — 리빌 인터랙션 ══ */
.gm-standing-row{display:flex;align-items:center;gap:14px;background:var(--card);border:1px solid var(--line);
  border-radius:12px;padding:10px 16px;margin-bottom:7px;box-shadow:var(--shadow-sm);
  animation:gm-pop .45s cubic-bezier(.2,.9,.25,1.15) both;animation-delay:calc(var(--i,0) * 90ms)}
/* 상위 3팀은 시상대처럼 — 금/은/동 색이 행 전체에 은은하게 번지고 테두리가 빛난다 */
.gm-standing-row.gm-rank-1{background:linear-gradient(90deg,rgba(255,201,77,.16),var(--card) 58%);
  border-color:rgba(255,201,77,.55);box-shadow:0 6px 20px rgba(0,0,0,.42),0 0 20px rgba(255,201,77,.18)}
.gm-standing-row.gm-rank-2{background:linear-gradient(90deg,rgba(201,214,232,.13),var(--card) 58%);
  border-color:rgba(201,214,232,.42)}
.gm-standing-row.gm-rank-3{background:linear-gradient(90deg,rgba(216,154,99,.13),var(--card) 58%);
  border-color:rgba(216,154,99,.42)}
.gm-standing-rank{width:30px;height:30px;border-radius:50%;display:flex;align-items:center;justify-content:center;
  font-size:13px;font-weight:800;color:#fff;background:rgba(255,255,255,.10);flex-shrink:0;
  border:1px solid rgba(255,255,255,.16);font-variant-numeric:tabular-nums}
.gm-standing-row.gm-rank-1 .gm-standing-rank{background:linear-gradient(140deg,#FFD97A,#B8871F);
  color:#2A1F05;border-color:rgba(255,217,122,.8);box-shadow:0 0 12px rgba(255,201,77,.55)}
.gm-standing-row.gm-rank-2 .gm-standing-rank{background:linear-gradient(140deg,#E4ECF7,#8C99AB);color:#1A2130}
.gm-standing-row.gm-rank-3 .gm-standing-rank{background:linear-gradient(140deg,#E0AE7C,#8A5A32);color:#2A1810}
.gm-standing-logo{width:26px;height:26px;object-fit:contain;flex-shrink:0;
  background:rgba(255,255,255,.92);border-radius:50%;padding:2px;
  box-shadow:0 1px 5px rgba(0,0,0,.3)}
.gm-standing-name{flex:1;font-size:14px;font-weight:700;color:var(--ink)}
.gm-standing-bar-track{width:120px;height:8px;border-radius:4px;background:rgba(255,255,255,.12);overflow:hidden}
.gm-standing-bar-fill{height:100%;background:linear-gradient(90deg,var(--navy),var(--team-accent));
  border-radius:4px;animation:gm-bar-grow .7s cubic-bezier(.2,.8,.2,1) both;animation-delay:calc(var(--i,0) * 90ms + .1s)}
@keyframes gm-bar-grow{from{width:0}}
.gm-standing-pct{width:52px;text-align:right;font-size:13px;font-weight:800;color:var(--ink);
  font-variant-numeric:tabular-nums}

/* ══ 구단 선택 지도 — 각 구단이 밤 경기장의 조명탑처럼 서 있다 ══ */
.gm-usmap-wrap{position:relative;width:100%;margin:0 auto;border-radius:20px;overflow:hidden;
  padding:6px 0;
  background:radial-gradient(760px 340px at 50% 118%, rgba(78,143,214,.16), transparent 68%);
  filter:drop-shadow(0 22px 46px rgba(0,0,0,.45))}
.gm-usmap-wrap svg{width:100%;height:auto;display:block;overflow:visible}
.gm-usmap-wrap a{text-decoration:none;outline:none}
.gm-usmap-state{fill:url(#gm-map-plate);stroke:rgba(140,180,240,.30);stroke-width:1;
  transition:fill .25s, stroke .25s}
.gm-usmap-wrap:hover .gm-usmap-state{stroke:rgba(140,180,240,.42)}

/* 지도를 천천히 훑고 지나가는 레이더 스캔 (아주 은은하게 — 띠가 도드라지면
   지형보다 스캔이 먼저 보여서 오히려 지저분해진다) */
.gm-map-scan{animation:gm-map-scan 9s linear infinite;pointer-events:none;opacity:.55;
  mix-blend-mode:screen}
@keyframes gm-map-scan{0%{transform:translateX(-200px)}100%{transform:translateX(1010px)}}

/* 주의: 이 g 요소는 SVG transform="translate(x,y)" 속성으로 위치가 정해진다.
   CSS 애니메이션이 transform 을 건드리면 그 속성을 덮어써서 30개 핀이 전부
   원점(좌상단)으로 뭉친다(실측 확인) — 등장 연출은 opacity 로만 한다. */
.gm-pin{cursor:pointer;opacity:0;animation:gm-pin-in .45s ease both;
  animation-delay:calc(var(--i,0) * 30ms)}
@keyframes gm-pin-in{to{opacity:1}}
/* 지면 글로우 — 조명이 땅에 떨어진 자리 */
.gm-pin-ground{opacity:.30;filter:blur(2.5px);transition:opacity .2s}
/* 퍼져나가는 레이더 링 */
.gm-pin-radar{transform-box:fill-box;transform-origin:center;opacity:0;
  animation:gm-pin-radar 3.2s ease-out infinite;animation-delay:calc(var(--i,0) * .1s)}
@keyframes gm-pin-radar{0%{transform:scale(.5);opacity:.7}100%{transform:scale(2.8);opacity:0}}
/* 빛기둥 — 아래에서 위로 옅어지고, 숨쉬듯 밝기가 변한다 */
.gm-pin-beam{opacity:.30;
  -webkit-mask-image:linear-gradient(to top, rgba(0,0,0,.95), transparent);
  mask-image:linear-gradient(to top, rgba(0,0,0,.95), transparent);
  animation:gm-pin-beam 3.4s ease-in-out infinite;animation-delay:calc(var(--i,0) * .1s);
  transition:opacity .2s}
@keyframes gm-pin-beam{0%,100%{opacity:.22}50%{opacity:.48}}
/* 조명 꼭대기 오브 — 위아래로 아주 조금 부유 */
.gm-pin-orb{transform-box:fill-box;transform-origin:center;
  animation:gm-pin-bob 3.4s ease-in-out infinite;animation-delay:calc(var(--i,0) * .1s);
  transition:transform .18s ease}
@keyframes gm-pin-bob{0%,100%{transform:translateY(0)}50%{transform:translateY(-3px)}}

/* 호버 — 조명이 확 켜지고 이름표가 뜬다 */
.gm-pin:hover .gm-pin-beam{opacity:.75;animation-play-state:paused}
.gm-pin:hover .gm-pin-ground{opacity:.6}
.gm-pin:hover .gm-pin-orb{animation-play-state:paused;transform:translateY(-4px) scale(1.35)}
.gm-pin-plate{opacity:0;transform-box:fill-box;transition:opacity .16s ease;pointer-events:none}
.gm-pin:hover .gm-pin-plate{opacity:1}
.gm-pin:hover{filter:drop-shadow(0 0 12px var(--pin,#fff))}
/* 스포트라이트 — SVG 에는 z-index 가 없어서 호버한 핀을 맨 앞으로 끌어올릴 수
   없다(형제 순서로만 겹침이 결정됨). 그래서 반대로 나머지를 어둡게 낮춰
   이름표가 옆 구단 핀에 가려 읽히지 않던 문제를 해결한다(북동부 밀집 구간
   에서 실측). 부수적으로 "지금 이 구단을 보고 있다"는 집중 효과도 생긴다. */
.gm-usmap-wrap:hover .gm-pin{opacity:.28;transition:opacity .18s ease}
.gm-usmap-wrap:hover .gm-pin:hover{opacity:1}

@media (prefers-reduced-motion: reduce){
  .gm-pin,.gm-pin-radar,.gm-pin-beam,.gm-pin-orb,.gm-map-scan{animation:none}
  .gm-pin-radar{opacity:.3}
}

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


# ── MLB 공식 팀 ID (로고 CDN 용) ──────────────────────────────────
# mlb_2026_team_standings.csv 의 team_id 를 UI 팀코드에 매핑한 것.
# ARI 만 원본이 "D-backs" 라 자동 매칭이 안 돼 직접 넣었다.
MLB_TEAM_IDS: dict[str, int] = {
    "ARI": 109,
    "ATH": 133,
    "ATL": 144,
    "BAL": 110,
    "BOS": 111,
    "CHC": 112,
    "CHW": 145,
    "CIN": 113,
    "CLE": 114,
    "COL": 115,
    "DET": 116,
    "HOU": 117,
    "KCR": 118,
    "LAA": 108,
    "LAD": 119,
    "MIA": 146,
    "MIL": 158,
    "MIN": 142,
    "NYM": 121,
    "NYY": 147,
    "PHI": 143,
    "PIT": 134,
    "SDP": 135,
    "SEA": 136,
    "SFG": 137,
    "STL": 138,
    "TBR": 139,
    "TEX": 140,
    "TOR": 141,
    "WSN": 120
}


def team_logo_url(team_code: str | None) -> str | None:
    """MLB 공식 팀 로고 SVG URL. 매핑에 없으면 None(호출부가 대체 표시).

    선수 헤드샷(photos.py)과 동일하게 MLB CDN 을 핫링크한다 — 파일을
    내려받아 저장하지 않는다.
    """
    tid = MLB_TEAM_IDS.get(team_code or "")
    return f"https://www.mlbstatic.com/team-logos/{tid}.svg" if tid else None

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


# ══════════════════════════════════════════════════════════════════════
# 3D 파티클 레이어 — 밤 경기장 공기 중에 떠다니는 빛 입자
#
# Streamlit 은 st.html 로 넣은 <script> 를 실행하지 않으므로 JS 파티클 엔진을
# 쓸 수 없다. 대신 CSS perspective + translate3d 로 진짜 원근을 준다:
# z 가 뒤로 갈수록(음수) 작고 흐릿하고 어둡게, 앞으로 올수록 크고 선명하게
# 보이도록 blur/opacity/size 를 z 와 함께 계산해 둔다 — 레이어가 서로 다른
# 속도로 흐르면서 시차(parallax)가 생긴다.
# 좌표·속도는 난수를 쓰되 seed 를 고정한다 — 리런마다 파티클이 튀면
# 화면이 산만해지기 때문.
# ══════════════════════════════════════════════════════════════════════

_PARTICLE_COUNT = 26
_BOKEH_COUNT = 7
_FRONT_COUNT = 9


def _particle_layer_html() -> str:
    import random

    rng = random.Random(20260830)  # 고정 seed — 리런해도 같은 배치
    parts = []

    for _ in range(_PARTICLE_COUNT):
        z = rng.uniform(-340, 130)          # 깊이
        depth = (z + 340) / 470             # 0(멀다) ~ 1(가깝다)
        size = 2.0 + depth * 4.4            # 가까울수록 크게
        blur = (1 - depth) * 2.6            # 멀수록 흐리게
        op = 0.18 + depth * 0.5             # 가까울수록 진하게
        parts.append(
            '<span class="gm-particle" style="'
            f"--x:{rng.uniform(0, 100):.1f}%;"
            f"--s:{size:.1f}px;"
            f"--z:{z:.0f}px;"
            f"--blur:{blur:.2f}px;"
            f"--op:{op:.2f};"
            f"--dx:{rng.uniform(-70, 70):.0f}px;"
            # 가까운 입자가 빨리 지나가야 시차가 산다
            f"--dur:{rng.uniform(26, 54) - depth * 10:.1f}s;"
            f"--d:-{rng.uniform(0, 40):.1f}s"
            '"></span>'
        )

    for _ in range(_BOKEH_COUNT):
        parts.append(
            '<span class="gm-bokeh" style="'
            f"--x:{rng.uniform(-5, 100):.1f}%;"
            f"--y:{rng.uniform(5, 95):.1f}%;"
            f"--s:{rng.uniform(90, 230):.0f}px;"
            f"--op:{rng.uniform(.05, .13):.3f};"
            f"--dur:{rng.uniform(30, 60):.1f}s;"
            f"--d:-{rng.uniform(0, 30):.1f}s"
            '"></span>'
        )

    # 전경(near-field) — 콘텐츠보다 앞에 떠서 카메라 바로 앞 먼지처럼 보인다.
    # 뒤 레이어만 있으면 카드에 다 가려서 깊이가 안 느껴진다. 앞뒤로 감싸야
    # 콘텐츠가 "공간 속에 놓인" 것처럼 읽힌다. 가독성을 해치지 않도록
    # 개수는 적게, 흐림은 세게, 불투명도는 아주 낮게 잡는다.
    front = []
    for _ in range(_FRONT_COUNT):
        front.append(
            '<span class="gm-particle-front" style="'
            f"--x:{rng.uniform(0, 100):.1f}%;"
            f"--s:{rng.uniform(9, 26):.0f}px;"
            f"--op:{rng.uniform(.05, .12):.3f};"
            f"--dx:{rng.uniform(-120, 120):.0f}px;"
            f"--dur:{rng.uniform(15, 26):.1f}s;"
            f"--d:-{rng.uniform(0, 22):.1f}s"
            '"></span>'
        )

    return (
        f'<div class="gm-particles" aria-hidden="true">{"".join(parts)}</div>'
        f'<div class="gm-particles-front" aria-hidden="true">{"".join(front)}</div>'
    )


def inject_css() -> None:
    # st.markdown 은 큰 <style> 블록을 마크다운 파서가 중간에 텍스트로 흘려버리는
    # 경우가 있다. st.html() 은 마크다운 파싱을 거치지 않고 그대로 주입한다.
    team_code = st.session_state.get("team_code")
    st.html(FONT_LINK + CSS + _team_theme_css(team_code) + _particle_layer_html())


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
            # 팀 약칭 텍스트 대신 MLB 공식 로고. 로고 로드 실패 시 onerror 로
            # img 를 숨기면 뒤에 깔린 약칭이 그대로 드러난다(이중 안전장치).
            logo = team_logo_url(team_code)
            logo_img = (
                f'<img src="{logo}" alt="" loading="lazy" '
                "onerror=\"this.style.display='none'\"/>" if logo else ""
            )
            st.markdown(
                f'<div class="mt-ab"><span>{team_code}</span>{logo_img}</div>',
                unsafe_allow_html=True,
            )
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
            st.page_link("Home.py", label="구단 변경")


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
    if ovr >= 80:
        return "legendary"
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


def radar_chart_svg(
    axes: list[tuple[str, float]],
    *,
    max_value: float = 100.0,
    size: int = 210,
    color: str = "var(--team-accent)",
) -> str:
    """레이더(스타) 차트 SVG. NBA 대시보드 레퍼런스 스타일 — 값은 0~max_value.

    축이 3개 미만이면 다각형이 의미가 없어 막대로 대체하지 않고 그대로 그린다
    (호출부가 최소 3축을 보장해야 함).
    """
    n = len(axes)
    if n < 3:
        raise ValueError("radar_chart_svg 는 최소 3개 축이 필요합니다")

    # 좌/우 끝 라벨(예: "경험", "출전율")이 text-anchor="end/start"로 바깥쪽으로
    # 뻗어나가는데, viewBox를 size 그대로 두면 그 텍스트가 0/size 경계 밖으로
    # 잘려서 "험"처럼 앞글자가 잘린 채로 렌더링됐다(실측 확인) — 가로로
    # margin을 더 줘서 라벨이 잘리지 않게 한다. 원(그리드/데이터)은 원래
    # size 기준 그대로 두고, 라벨 공간만큼만 캔버스를 넓힌다.
    margin_x = max(30, size * 0.16)
    view_w = size + margin_x * 2
    view_h = size
    cx = view_w / 2
    cy = view_h / 2
    r_max = size * 0.32
    label_r = size * 0.44

    def _point(i: int, r: float) -> tuple[float, float]:
        angle = -math.pi / 2 + i * (2 * math.pi / n)
        return cx + r * math.cos(angle), cy + r * math.sin(angle)

    # 배경 그리드 (25/50/75/100%) — 동심 폴리곤
    grid_polys = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (_point(i, r_max * frac) for i in range(n)))
        grid_polys.append(f'<polygon points="{pts}" fill="none" stroke="rgba(255,255,255,.14)" stroke-width="1"/>')

    axis_lines = []
    labels_html = []
    for i, (label, _value) in enumerate(axes):
        x, y = _point(i, r_max)
        axis_lines.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{x:.1f}" y2="{y:.1f}" '
                           'stroke="rgba(255,255,255,.16)" stroke-width="1"/>')
        lx, ly = _point(i, label_r)
        # 좌/우 라벨은 텍스트가 잘리지 않게 anchor 를 위치에 맞춘다
        anchor = "middle" if abs(lx - cx) < size * 0.06 else ("start" if lx > cx else "end")
        labels_html.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" dominant-baseline="middle" '
            f'font-size="{max(9, size * 0.052):.0f}" font-weight="700" fill="rgba(255,255,255,.75)">{label}</text>'
        )

    data_pts = [_point(i, r_max * max(0.0, min(1.0, value / max_value))) for i, (_l, value) in enumerate(axes)]
    data_poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in data_pts)
    dots = "".join(
        f'<circle class="gm-radar-dot" cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}" '
        f'stroke="#fff" stroke-width="1.2" style="animation-delay:{.5 + i * .05}s"/>'
        for i, (x, y) in enumerate(data_pts)
    )
    grad_id = f"gm-radar-grad-{abs(hash(tuple(a for a, _ in axes))) % 100000}"

    return (
        f'<svg class="gm-radar" width="{view_w:.0f}" height="{view_h:.0f}" viewBox="0 0 {view_w:.0f} {view_h:.0f}" '
        'xmlns="http://www.w3.org/2000/svg">'
        f'<defs><radialGradient id="{grad_id}" cx="50%" cy="50%" r="65%">'
        f'<stop offset="0%" stop-color="{color}" stop-opacity=".55"/>'
        f'<stop offset="100%" stop-color="{color}" stop-opacity=".12"/>'
        "</radialGradient></defs>"
        + "".join(grid_polys)
        + "".join(axis_lines)
        + f'<polygon class="gm-radar-fill" points="{data_poly}" fill="url(#{grad_id})" '
        f'stroke="{color}" stroke-width="2" stroke-linejoin="round"/>'
        + dots
        + "".join(labels_html)
        + "</svg>"
    )


def player_hero_card_html(
    *,
    name: str,
    role_label: str,
    team: str,
    ovr: float,
    radar_axes: list[tuple[str, float]],
    chips: list[str] | None = None,
    photo_url: str | None = None,
    league_rank: int | None = None,
    league_total: int | None = None,
    rings: list[tuple[str, float]] | None = None,
) -> str:
    """선수 리포트용 큰 프로파일 카드 — 왼쪽 아이덴티티 + 오른쪽 레이더 차트.

    NBA 대시보드 레퍼런스(선수 사진 + 등번호 워터마크 + 레이더)를 이 프로젝트의
    실제 데이터(overall_score 등)로 재구성한 것 — 사진이 없으면 실루엣+이니셜로
    자연스럽게 대체된다(player_card_html 과 동일한 폴백 원칙).

    league_rank/league_total: 같은 시즌 리그 전체에서의 순위(1=최고). 전력 점수는
    시즌별 min-max 정규화라서 그 시즌 최저 선수는 항상 정확히 0.00, 최고 선수는
    항상 정확히 100.00이 나오는 구조적 특성이 있다 — "0.00"만 보면 계산이 깨진
    것처럼 보일 수 있어, 리그 내 순위/퍼센타일을 함께 보여줘서 "정규화상 바닥"과
    "실제 능력치 0"을 구분해준다.
    """
    initials = "".join(part[0] for part in name.replace("-", " ").split()[:2]).upper() or "?"
    photo_html = (
        f'<img src="{photo_url}" alt="" loading="lazy" onerror="this.style.display=\'none\'"/>'
        if photo_url else ""
    )
    chips_html = "".join(f'<span class="gm-hero-chip">{c}</span>' for c in (chips or []))

    rank_html = ""
    if league_rank is not None and league_total:
        pct_from_top = (league_rank - 1) / league_total * 100
        if ovr <= 0.05:
            note = badge(icon("alert", 10) + " 시즌 최저치", "risk")
            detail = f"리그 {league_total}명 중 {league_rank}위 — 정규화 특성상 매 시즌 1명은 항상 0.00이 됩니다"
        elif ovr >= 99.95:
            note = badge(icon("trophy", 10) + " 시즌 최고치", "gain")
            detail = f"리그 {league_total}명 중 {league_rank}위"
        elif pct_from_top <= 10:
            note = badge(f"상위 {max(pct_from_top, 0.1):.0f}%", "gain")
            detail = f"리그 {league_total}명 중 {league_rank}위"
        elif pct_from_top >= 90:
            note = badge(f"하위 {100 - pct_from_top:.0f}%", "warn")
            detail = f"리그 {league_total}명 중 {league_rank}위"
        else:
            note = ""
            detail = f"리그 {league_total}명 중 {league_rank}위"
        rank_html = f'<div class="gm-hero-percentile">{note}<span>{detail}</span></div>'

    return (
        '<div class="gm-hero-card">'
        f'<div class="gm-hero-watermark">{ovr:.0f}</div>'
        '<div class="gm-hero-id">'
        f'<div class="gm-hero-avatar">{_PLAYER_SILHOUETTE_SVG}<span class="pc-initials">{initials}</span>{photo_html}</div>'
        '<div>'
        f'<div class="gm-hero-name">{name}</div>'
        f'<div class="gm-hero-role">{role_label} · {team}</div>'
        "</div>"
        '<div class="gm-hero-ovr-row">'
        f'<span class="gm-hero-ovr">{ovr:.0f}</span>'
        '<span class="gm-hero-ovr-label">종합 전력</span>'
        "</div>"
        f'{rank_html}'
        f'<div class="gm-hero-chips">{chips_html}</div>'
        + (
            '<div class="gm-hero-rings">'
            + "".join(
                stat_ring_svg(v, lbl, size=76, tone=(tone[0] if tone else "var(--team-accent)"))
                for lbl, v, *tone in rings
            )
            + "</div>"
            if rings else ""
        )
        + "</div>"
        f'<div class="gm-hero-radar">{radar_chart_svg(radar_axes)}</div>'
        "</div>"
    )


def trend_chart_svg(
    seasons: list[int],
    values: list[float],
    *,
    future_season: int | None = None,
    future_value: float | None = None,
    max_value: float = 100.0,
    width: int = 560,
    height: int = 200,
    color: str = "var(--team-accent)",
) -> str:
    """시즌별 전력 추세 라인차트. future_season/future_value가 있으면 마지막
    실측 지점에서 점선으로 이어지는 "예측" 구간을 덧붙인다(D의 다음 시즌 전력
    예측 모델 — 확정 결과가 아니라 모델 추정이므로 점선+다른 색으로
    시각적으로 분리한다. 실측과 예측을 같은 실선으로 섞어 보여주지 않는다.
    어느 모델을 쓰는지는 next_strength.py 호출부(NEXT_STRENGTH_PATH)가 결정
    하므로 여기에 모델명을 하드코딩하지 않는다 — 모델 교체 시 이 주석이
    stale해지는 걸 방지).
    """
    if len(seasons) != len(values):
        raise ValueError("seasons와 values 길이가 같아야 합니다")
    pad_l, pad_r, pad_t, pad_b = 34, 18, 16, 26
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    all_seasons = list(seasons) + ([future_season] if future_season is not None else [])
    n = len(all_seasons)
    if n < 2:
        step = 0.0
    else:
        step = plot_w / (n - 1)

    def _xy(i: int, value: float) -> tuple[float, float]:
        x = pad_l + i * step
        y = pad_t + (1 - max(0.0, min(1.0, value / max_value))) * plot_h
        return x, y

    # 격자선 (0/25/50/75/100)
    grid = []
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        gy = pad_t + (1 - frac) * plot_h
        grid.append(
            f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{width - pad_r}" y2="{gy:.1f}" '
            'stroke="rgba(255,255,255,.08)" stroke-width="1"/>'
        )
        grid.append(
            f'<text x="{pad_l - 8}" y="{gy + 3:.1f}" text-anchor="end" font-size="9" '
            f'fill="rgba(255,255,255,.4)">{frac * max_value:.0f}</text>'
        )

    hist_pts = [_xy(i, v) for i, v in enumerate(values)]
    hist_path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(hist_pts))
    hist_dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{color}" stroke="#0A1220" stroke-width="1.5"/>'
        for x, y in hist_pts
    )
    labels = "".join(
        f'<text x="{x:.1f}" y="{height - 8}" text-anchor="middle" font-size="9.5" '
        f'fill="rgba(255,255,255,.55)">{s}</text>'
        for (x, _y), s in zip(hist_pts, seasons)
    )

    future_html = ""
    if future_season is not None and future_value is not None and hist_pts:
        fx, fy = _xy(n - 1, future_value)
        lx, ly = hist_pts[-1]
        future_html = (
            f'<line x1="{lx:.1f}" y1="{ly:.1f}" x2="{fx:.1f}" y2="{fy:.1f}" '
            f'stroke="#FFC94D" stroke-width="2.5" stroke-dasharray="5,4" stroke-linecap="round"/>'
            f'<circle cx="{fx:.1f}" cy="{fy:.1f}" r="4.5" fill="none" stroke="#FFC94D" stroke-width="2.5"/>'
            f'<circle cx="{fx:.1f}" cy="{fy:.1f}" r="2" fill="#FFC94D"/>'
            f'<text x="{fx:.1f}" y="{height - 8}" text-anchor="middle" font-size="9.5" '
            f'font-weight="800" fill="#FFC94D">{future_season}?</text>'
        )

    return (
        f'<svg class="gm-trend" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'xmlns="http://www.w3.org/2000/svg">'
        + "".join(grid)
        + f'<path class="gm-trend-line" d="{hist_path}" fill="none" stroke="{color}" '
        'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'
        + hist_dots
        + future_html
        + labels
        + "</svg>"
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
# 주(州) 외곽선 path 모음 — 지도의 고정 지형. 마커는 아래에서 코드로 생성한다
_US_STATE_PATHS = """<path class="gm-usmap-state" d="m 643,467.4 .4,-7.3 -.9,-1.2 -1.7,-.7 -2.5,-2.8 .5,-2.9 48.8,-5.1 -.7,-2.2 -1.5,-1.5 -.5,-1.4 .6,-6.3 -2.4,-5.7 .5,-2.6 .3,-3.7 2.2,-3.8 -.2,-1.1 -1.7,-1 v -3.2 l -1.8,-1.9 -2.9,-6.1 -12.9,-45.8 -45.7,4 1.3,2 -1.3,67 4.4,33.2 .9,-.5 1.3,.1 .6,.4 .8,-.1 2,-3.8 v -2.3 l 1.1,-1.1 1.4,.5 3.4,6.4 v .9 l -3.3,2.2 3.5,-.4 4.9,-1.6 z"/><path class="gm-usmap-state" d="m 139.6,387.6 3,-2.2 .8,-2.4 -1,-1.6 -1.8,-.2 -1.1,-1.6 1.1,-6.9 1.6,-.3 2.4,-3.2 1.6,-7 2.4,-3.6 4.8,-1.7 1.3,-1.3 -.4,-1.9 -2.3,-2.5 -1.2,-5.8 -1.4,-1.8 -1.3,-3.4 .9,-2.1 1.4,-3 .5,-2.9 -.5,-4.9 1,-13.6 3.5,-.6 3.7,1.4 1.2,2.7 h 2 l 2.4,-2.9 3.4,-17.5 46.2,8.2 40,6 -17.4,124.1 -37.3,-5.4 -64.2,-37.5 .5,-2.9 2,-1.8 z"/><path class="gm-usmap-state" d="m 584.2,367 .9,-2.2 1.2,.5 .7,-1 -.8,-.7 .3,-1.5 -1.1,-.9 .6,-1 -.1,-1.5 -1.1,-.1 .8,-.8 1.3,.8 .3,-1.4 -.4,-1.1 .1,-.7 2,.6 -.4,-1.5 1.6,-1.3 -.5,-.9 -1.1,.1 -.6,-.9 .9,-.9 1.6,-.2 .5,-.8 1.4,-.2 -.1,-.8 -.9,-.9 v -.5 h 1.5 l .4,-.7 -1.4,-1 -.1,-.6 -11.2,.8 2.8,-5.1 1.7,-1.5 v -2.2 l -1.6,-2.5 -39.8,2 -39.1,.7 4.1,24.4 -.7,39 2.6,2.3 2.8,-1.3 3.2,.8 .2,11.9 52.3,-1.3 1.2,-1.5 .5,-3 -1.5,-2.3 -.5,-2.2 .9,-.7 v -.8 l -1.7,-1.1 -.1,-.7 1.6,-.9 -1.2,-1.1 1.7,-7.1 3.4,-1.6 v -.8 l -1.1,-1.4 2.9,-5.4 h 1.9 l 1.5,-1.2 -.3,-5.2 3.1,-4.5 1.8,-.6 -.5,-3.1 z"/><path class="gm-usmap-state" d="m 69.4,365.6 3.4,5.2 -1.4,.1 -1.8,-1.9 z m 1.9,-9.8 1.8,4.1 2.6,1 .7,-.6 -1.3,-2.5 -2.6,-2.4 z m -19.9,-19 v 2.4 l 2,1.2 4.4,-.2 1,-1 -3.1,-.2 z m -5.9,.1 3.3,.5 1.4,2.2 h -3.8 z m 47.9,45.5 -1,-3 .2,-3 -.4,-7.9 -1.8,-4.8 -1.2,-1.4 -.6,-1.5 -7,-8.6 -3.6,.1 -2,-1.9 1.1,-1.8 -.7,-3.7 -2.2,-1.2 -3.9,-.6 -2.8,-1.3 -1.5,-1.9 -4.5,-6.6 -2.7,-2.2 -3.7,-.5 -3.1,-2.3 -4.7,-1.5 -2.8,-.3 -2.5,-2.5 .2,-2.8 .8,-4.8 1.8,-5.1 -1.4,-1.6 -4,-9.4 -2.7,-3.7 -.4,-3 -1.6,-2.3 .2,-2.5 -2,-5 -2.9,-2.7 .6,-7.1 2.4,-.8 1.8,-3.1 -.4,-3.2 -1,-.9 h -2.5 l -2.5,-3.3 -1.5,-3.5 v -7.5 l 1.2,-4.2 .2,-2.1 2.5,.2 -.1,1.6 -.8,.7 v 2.5 l 3.7,3.2 v -4.7 l -1.4,-3.4 .5,-1.1 -1,-1.7 2.8,-1.5 -1.9,-3 -1.4,.5 -1.5,3.8 .5,1.3 -.8,1 -.9,-.1 -5.4,-6.1 .7,-5.6 -1.1,-3.9 -6.5,-12.8 .8,-10.7 2.3,-3.6 .2,-6.4 -5.5,-11.1 .3,-5.2 6.9,-7.5 1.7,-2.4 -.1,-1.4 4,-9.2 .1,-8.4 .9,-2.5 66.1,18.6 -16.4,63.1 1.1,3.5 70.4,105 -.9,2.1 1.3,3.4 1.4,1.8 1.2,5.8 2.3,2.5 .4,1.9 -1.3,1.3 -4.8,1.7 -2.4,3.6 -1.6,7 -2.4,3.2 -1.6,.3 -1.1,6.9 1.1,1.6 1.8,.2 1,1.6 -.8,2.4 -3,2.2 -2.2,-.1 z"/><path class="gm-usmap-state" d="m 374.6,323.3 -16.5,-1 -51.7,-4.8 -52.6,-6.5 11.5,-88.3 44.9,5.7 37.5,3.4 33.1,2.4 -1.4,22.1 z"/><path class="gm-usmap-state" d="m 873.5,178.9 .4,-1.1 -3.2,-12.3 -.1,-.3 -14.9,3.4 v .7 l -.9,.3 -.5,-.7 -10.5,2.4 2.8,16.3 1.8,1.5 -3.5,3.4 1.7,2.2 5.4,-4.5 1.7,-1.3 h .8 l 2.4,-3.1 1.4,.1 2.9,-1.1 h 2.1 l 5.3,-2.7 2.8,-.9 1,-1 1.5,.5 z"/><path class="gm-usmap-state" d="m 822.2,226.6 -1.6,.3 -1.5,1.1 -1.2,2.1 7.6,27.1 10.9,-2.3 -2.2,-7.6 -1.1,.5 -3.3,-2.6 -.5,-1.7 -1.8,-1 -.2,-3.7 -2.1,-2.2 -1.1,-.8 -1.2,-1.1 -.4,-3.2 .3,-2.1 1,-2.2 z"/><path class="gm-usmap-state" d="m 751.7,445.1 -4,-.7 -1.7,-.9 -2.2,1.4 v 2.5 l 1.4,2.1 -.5,4.3 -2.1,.6 -1,-1.1 -.6,-3.2 -50.1,3.3 -3.3,-6 -48.8,5.1 -.5,2.9 2.5,2.8 1.7,.7 .9,1.2 -.4,7.3 -1.1,.6 .5,.4 1,-.3 .7,-.8 10.5,-2.7 9.2,-.5 8.1,1.9 8.5,5 2.4,.8 2.2,2 -.1,2.7 h 2.4 l 1.9,-1 2.5,.1 2,-.8 2.9,-2 3.1,-2.9 1.1,-.4 .6,.5 h 1.4 l .5,-.8 -.5,-1.2 -.6,-.6 .2,-.8 2,-1.1 5,-.4 .8,1 1,.1 2.3,1 3,1.8 1.2,1.7 1.1,1.2 2.8,1.4 v 2.4 l 2.8,1.9 1,.1 1.6,1.4 .7,1.6 1,.2 .8,2.1 .7,.6 1,-1.1 2.9,.1 .5,1.4 1.1,.9 v 1.3 l 2.9,2.2 .2,9.6 -1.8,5.8 1,1.2 -.2,3.4 -.8,1.4 .7,1.2 2.3,2.3 .3,1.5 .8,1 -.4,-1.9 1.3,-.6 .8,-3.6 -3,-1.2 .1,-.6 2.6,-.4 .9,2.6 1.1,.6 .1,-2 1.1,.3 .6,.8 -.1,.7 -2.9,4.2 -.2,1.1 -1.7,1.9 v 1.1 l 3.7,3.8 5.3,7.9 1.8,2.1 v 1.8 l 2.8,4.6 2.3,.6 .7,-1.2 -2.1,.3 -3,-4.5 .2,-1.4 1.5,-.8 v -1.5 l -.6,-1.3 .9,-.9 .4,.9 .7,.5 v 4 l -1.2,-.6 -.8,.9 1.4,1.6 1,2.6 1.2,-.6 2.3,1.2 2.1,2.2 1.6,5.1 3.1,4.8 .8,-1.3 2.8,-.5 3.2,1.3 .3,1.7 3.3,3.8 .1,1.1 2.2,2.7 -.7,.5 v 2.7 l 2.7,1.4 h 1.5 l 2.7,-1.8 1.5,.3 1.1,.4 2.3,-1.7 .2,-.7 1.2,.3 2.4,-1.7 1.3,-2.3 -.7,-3.2 -.2,-1.3 1.1,-4 .6,-.2 .6,1.6 .8,-1.8 -.8,-7.2 -.4,-10.5 -1,-6.8 -.7,-1.7 -6.6,-11.1 -5.2,-9.1 -2.2,-3.3 -1.3,-3.6 -.2,-3.4 .9,-.3 v -.9 l -1.1,-2.2 -4,-4 -7.6,-9.7 -5.7,-10.4 -4.3,-10.7 -.6,-3.7 -1.2,-1 -.5,-3.8 z m 9.2,134.5 1.7,-.1 -.7,-1 z m 7.3,-1.1 v -.7 l 1.6,-.2 3.7,-3.3 1.5,-.6 2.4,-.9 .3,1.3 1.7,.8 -2.6,1.2 h -2.4 l -3.9,2.5 z m 17.2,-7.6 -3,1.4 -1,1.3 1.1,.1 z m 3.8,-2.9 -1.1,.3 -1.4,2 1.1,-.2 1.5,-1.6 z m 8.3,-15.7 -1.7,5.6 -.8,1 -1,2.6 -1.2,1.6 -.7,1.7 -1.9,2.2 v .9 l 2.7,-2.8 2.4,-3.5 .6,-2 2.1,-4.9 z"/><path class="gm-usmap-state" d="m 761.8,414.1 v 1.4 l -4.2,6.2 -1.2,.2 1.5,.5 v 2 l -.9,1.1 -.6,6 -2.3,6.2 .5,2 .7,5.1 -3.6,.3 -4,-.7 -1.7,-.9 -2.2,1.4 v 2.5 l 1.4,2.1 -.5,4.3 -2.1,.6 -1,-1.1 -.6,-3.2 -50.1,3.3 -3.3,-6 -.7,-2.2 -1.5,-1.5 -.5,-1.4 .6,-6.3 -2.4,-5.7 .5,-2.6 .3,-3.7 2.2,-3.8 -.2,-1.1 -1.7,-1 v -3.2 l -1.8,-1.9 -2.9,-6.1 -12.9,-45.8 22.9,-2.9 21.4,-3 -.1,1.9 -1.9,1 -1.4,3.2 .2,1.3 6.1,3.8 2.6,-.3 3.1,4 .4,1.7 4.2,5.1 2.6,1.7 1.4,.2 2.2,1.6 1.1,2.2 2,1.6 1.8,.5 2.7,2.7 .1,1.4 2.6,2.8 5,2.3 3.6,6.7 .3,2.7 3.9,2.1 2.5,4.8 .8,3.1 4.2,.4 z"/><path class="gm-usmap-state" d="m 165.3,183.1 -24.4,-5.4 8.5,-37.3 2.9,-5.8 .4,-2.1 .8,-.9 -.9,-2 -2.9,-1.2 .2,-4.2 4,-5.8 2.5,-.8 1.6,-2.3 -.1,-1.6 1.8,-1.6 3.2,-5.5 4.2,-4.8 -.5,-3.2 -3.5,-3.1 -1.6,-3.6 1.1,-4.3 -.7,-4 12.7,-56.1 14.2,3 -4.8,22 3.7,7.4 -1.6,4.8 3.6,4.8 1.9,.7 3.9,8.3 v 2.1 l 2.3,3 h .9 l 1.4,2.1 h 3.2 v 1.6 l -7.1,17 -.5,4.1 1.4,.5 1.6,2.6 2.8,-1.4 3.6,-2.4 1.9,1.9 .5,2.5 -.5,3.2 2.5,9.7 2.6,3.5 2.3,1.4 .4,3 v 4.1 l 2.3,2.3 1.6,-2.3 6.9,1.6 2.1,-1.2 9,1.7 2.8,-3.3 1.8,-.6 1.2,1.8 1.6,4.1 .9,.1 -8.5,54.8 -47.9,-8.2 z"/><path class="gm-usmap-state" d="m 623.5,265.9 -1,5.2 v 2 l 2.4,3.5 v .7 l -.3,.9 .9,1.9 -.3,2.4 -1.6,1.8 -1.3,4.2 -3.8,5.3 -.1,7 h -1 l .9,1.9 v .9 l -2.2,2.7 .1,1.1 1.5,2.2 -.1,.9 -3.7,.6 -.6,1.2 -1.2,-.6 -1,.5 -.4,3.3 1.7,1.8 -.4,2.4 -1.5,.3 -6.9,-3 -4,3.7 .3,1.8 h -2.8 l -1.4,-1.5 -1.8,-3.8 v -1.9 l .8,-.6 .1,-1.3 -1.7,-1.9 -.9,-2.5 -2.7,-4.1 -4.8,-1.3 -7.4,-7.1 -.4,-2.4 2.8,-7.6 -.4,-1.9 1.2,-1.1 v -1.3 l -2.8,-1.5 -3,-.7 -3.4,1.2 -1.3,-2.3 .6,-1.9 -.7,-2.4 -8.6,-8.4 -2.2,-1.5 -2.5,-5.9 -1.2,-5.4 1.4,-3.7 .7,-.7 .1,-2.3 -.7,-.9 1,-1.5 1.8,-.6 .9,-.3 1,-1.2 v -2.4 l 1.7,-2.4 .5,-.5 .1,-3.5 -.9,-1.4 -1,-.3 -1.1,-1.6 1,-4 3,-.8 h 2.4 l 4.2,-1.8 1.7,-2.2 .1,-2.4 1.1,-1.3 1.3,-3.2 -.1,-2.6 -2.8,-3.5 h -1.2 l -.9,-1.1 .2,-1.6 -1.7,-1.7 -2.5,-1.3 .5,-.6 45.9,-2.8 .1,4.6 3.4,4.6 1.2,4.1 1.6,3.2 z"/><path class="gm-usmap-state" d="m 629.2,214.8 -5.1,2.3 -4.7,-1.4 4.1,50.2 -1,5.2 v 2 l 2.4,3.5 v .7 l -.3,.9 .9,1.9 -.3,2.4 -1.6,1.8 -1.3,4.2 -3.8,5.3 -.1,7 h -1 l .9,1.9 1.1,.8 .6,-1 -.7,-1.7 4.6,-.5 .2,1.2 1.1,.2 .4,-.9 -.6,-1.3 .3,-.8 1.3,.8 1.7,-.4 1.7,.6 3.4,2.1 1.8,-2.8 3.5,-2.2 3,3.3 1.6,-2.1 .3,-2.7 3.8,-2.3 .2,1.3 1.9,1.2 3,-.2 1.2,-.7 .1,-3.4 2.5,-3.7 4.6,-4.4 -.1,-1.7 1.2,-3.8 2.2,1 6.7,-4.5 -.4,-1.7 -1.5,-2.1 1,-1.9 -6.6,-57.2 -.1,-1.4 -32.4,3.4 z"/><path class="gm-usmap-state" d="m 556.9,183 2.1,1.6 .6,1.1 -1.6,3.3 -.1,2.5 2,5.5 2.7,1.5 3.3,.7 1.3,2.8 -.5,.6 2.5,1.3 1.7,1.7 -.2,1.6 .9,1.1 h 1.2 l 2.8,3.5 .1,2.6 -1.3,3.2 -1.1,1.3 -.1,2.4 -1.7,2.2 -4.2,1.8 h -2.4 l -3,.8 -1,4 1.1,1.6 1,.3 .9,1.4 -.1,3.5 -.5,.5 -1.7,2.4 v 2.4 l -1,1.2 -.9,.3 -1.8,.6 -1,1.5 .7,.9 -.1,2.3 -.7,.7 -1.5,-.8 -1.1,-1.1 -.6,-1.6 -1.7,-1.3 -14.3,.8 -27.2,1.2 -25.9,-.1 -1.8,-4.4 .7,-2.2 -.8,-3.3 .2,-2.9 -1.3,-.7 -.4,-6.1 -2.8,-5 -.2,-3.7 -2.2,-4.3 -1.3,-3.7 v -1.4 l -.6,-1.7 v -2.3 l -.5,-.9 -.7,-1.7 -.3,-1.3 -1.3,-1.2 1,-4.3 1.7,-5.1 -.7,-2 -1.3,-.4 -.4,-1.6 1,-.5 .1,-1.1 -1.3,-1.5 .1,-1.6 2.2,.1 h 28.2 l 36.3,-.9 18.6,-.7 z"/><path class="gm-usmap-state" d="m 459.1,259.5 -43.7,-1.2 -36,-2 -4.8,67 67.7,2.9 62,.1 -.5,-48.1 -3.2,-.7 -2.6,-4.7 -2.5,-2.5 .5,-2.3 2.7,-2.6 .1,-1.2 -1.5,-2.1 -.9,1 -2,-.6 -2.9,-3 z"/><path class="gm-usmap-state" d="m 692.1,322.5 -20.5,1.4 -5.2,.8 -17.4,1 -2.6,.8 -22.6,2 -.7,-.6 h -3.7 l 1.2,3.2 -.6,.9 -23.3,1.5 1,-2.7 1.4,.9 .7,-.4 1.2,-4.1 -1,-1 1,-2 .2,-.9 -1.3,-.8 -.3,-1.8 4,-3.7 6.9,3 1.5,-.3 .4,-2.4 -1.7,-1.8 .4,-3.3 1,-.5 1.2,.6 .6,-1.2 3.7,-.6 .1,-.9 -1.5,-2.2 -.1,-1.1 2.2,-2.7 0,-.9 1.1,.8 .6,-1 -.7,-1.7 4.6,-.5 .2,1.2 1.1,.2 .4,-.9 -.6,-1.3 .3,-.8 1.3,.8 1.7,-.4 1.7,.6 3.4,2.1 1.8,-2.8 3.5,-2.2 3,3.3 1.6,-2.1 .3,-2.7 3.8,-2.3 .2,1.3 1.9,1.2 3,-.2 1.2,-.7 .1,-3.4 2.5,-3.7 4.6,-4.4 -.1,-1.7 1.2,-3.8 2.2,1 6.7,-4.5 -.4,-1.7 -1.5,-2.1 1,-1.9 1.3,.5 2.2,.1 1.9,-.8 2.9,1.2 2.2,3.4 v 1 l 4.1,.7 2.3,-.2 1.9,2.1 2.2,.2 v -1 l 1.9,-.8 3,.8 1.2,.8 1.3,-.7 h .9 l .6,-1.7 3.4,-1.8 .5,.8 .8,2.9 3.5,1.4 1.2,2.1 -.1,1.1 .6,1 -.6,3.6 1.9,1.6 .8,1.1 1,.6 -.1,.9 4.4,5.6 h 1.4 l 1.5,1.8 1.2,.3 1.4,-.1 -4.9,6.6 -2.9,1 -3,3 -.4,2.2 -2.1,1.3 -.1,1.7 -1.4,1.4 -1.8,.5 -.5,1.9 -1,.4 -6.9,4.2 z m -98,11.3 -.7,-.7 .2,-1 h 1.1 l .7,.7 -.3,1 z"/><path class="gm-usmap-state" d="m 602.5,472.8 -1.2,-1.8 .3,-1.3 -4.8,-6.8 .9,-4.6 1,-1.4 .1,-1.4 -36,2 1.7,-11.9 2.4,-4.8 6,-8.4 -1.8,-2.5 h 2 v -3.3 l -2.4,-2.5 .5,-1.7 -1.2,-1 -1.6,-7.1 .6,-1.4 -52.3,1.3 .5,19.9 .7,3.4 2.6,2.8 .7,5.4 3.8,4.6 .8,4.3 h 1 l -.1,7.3 -3.3,6.4 1.3,2.3 -1.3,1.5 .7,3 -.1,4.3 -2.2,3.5 -.1,.8 -1.7,1.2 1,1.8 1.2,1.1 1.6,-1.3 5.3,-.9 6.1,-.1 9.6,3.8 8,1 1.5,-1.4 1.8,-.2 4.8,2.2 1.6,-.4 1.1,-1.5 -4.2,-1.8 -2.2,1 -1.1,-.2 -1.4,-2 3.3,-2.2 1.6,-.1 v 1.7 l 1.5,-.1 3.4,-.3 .4,2.3 1.1,.4 .6,1.9 4.8,1 1.7,1.6 v .7 h -1.2 l -1.5,1.7 1.7,1.2 5.4,1 2.7,2.8 4.4,-1 -3.7,.2 -.1,-.6 2.8,-.7 .2,-1.8 1.2,-.3 v -1.4 l 1.1,.1 v 1.6 l 2.5,.1 .8,-1.9 .9,.3 .2,2.5 1.2,.2 -1.8,2 2.6,-.9 2,-1.1 2.9,-3.3 h -.7 l -1.3,1.2 -.4,-.1 -.5,-.8 .9,-1.2 v -2.3 l 1.1,-.8 .7,.7 1,-.8 1,-.1 .6,1.3 -.6,1.9 h 2.4 l 5.1,1.7 .5,1.3 1.6,1.4 2.8,.1 1.3,.7 1.8,-1 .9,-1.7 v -1.7 h -1.4 l -1.2,-1.4 -1.1,-1.1 -3.2,-.9 -2.6,.2 -4.2,-2.4 v -2.3 l 1.3,-1 2.4,.6 -3.1,-1.6 .2,-.8 h 3.6 l 2.6,-3.5 -2.6,-1.8 .8,-1.5 -1.2,-.8 h -.8 l -2,2.1 v 2.1 l -.6,.7 -1.1,-.1 -1.6,-1.4 h -1.3 v -1.5 l .6,-.7 .8,.7 1.7,-1.6 .7,-1.6 .8,-.3 z m -10.3,-2.7 1.9,1 .8,1.1 2.5,.1 1.5,.8 .2,1.4 -.4,.6 -.9,-1.5 -1.4,1.2 -.9,1.4 -2.8,.8 -1.6,.1 -3.7,-1 .1,-1.7 2,-2 1.1,-2.4 z m -4.7,1.2 v 1.1 l -1.8,2 h -1.2 v -2.2 l 1.6,-1.5 z"/><path class="gm-usmap-state" d="m 875,128.7 .6,4 3.2,2 .8,2.2 2.3,1.4 1.4,-.3 1,-3 -.8,-2.9 1.6,-.9 .5,-2.8 -.6,-1.3 3.3,-1.9 -2.2,-2.3 .9,-2.4 1.4,-2.2 .5,3.2 1.6,-2 1.3,.9 1.2,-.8 v -1.7 l 3.2,-1.3 .3,-2.9 2.5,-.2 2.7,-3.7 v -.7 l -.9,-.5 -.1,-3.3 .6,-1.1 .2,1.6 1,-.5 -.2,-3.2 -.9,.3 -.1,1.2 -1.2,-1.4 .9,-1.4 .6,.1 1.1,-.4 .5,2.8 2,-.3 2.9,.7 v -1 l -1.1,-1.2 1.3,.1 .1,-2.3 .6,.8 .3,1.9 2.1,1.5 .2,-1 .9,-.2 -.3,-.8 .8,-.6 -.1,-1.6 -1.6,-.2 -2,.7 1.4,-1.6 .7,-.8 1.3,-.2 .4,1.3 1.7,1.6 .4,-2.1 2.3,-1.2 -.9,-1.3 .1,-1.7 1.1,.5 h .7 l 1.7,-1.4 .4,-2.3 2.2,.3 .1,-.7 .2,-1.6 .5,1.4 1.5,-1 2.3,-4.1 -.1,-2.2 -1.4,-2 -3,-3.2 h -1.9 l -.8,2.2 -2.9,-3 .3,-.8 v -1.5 l -1.6,-4.5 -.8,-.2 -.7,.4 h -4.8 l -.3,-3.6 -8.1,-26 -7.3,-3.7 -2.9,-.1 -6.7,6.6 -2.7,-1 -1,-3.9 h -2.7 l -6.9,19.5 .7,6.2 -1.7,2.4 -.4,4.6 1.3,3.7 .8,.2 v 1.6 l -1.6,4.5 -1.5,1.4 -1.3,2.2 -.4,7.8 -2.4,-1 -1.5,.4 z m 34.6,-24.7 -1,.8 v 1.3 l .7,-.8 .9,.8 .4,-.5 1.1,.2 -1,-.8 .4,-.8 z m -1.7,2.6 -1,1.1 .5,.4 -.1,1 h 1.1 v -1.8 z m -3,-1.6 .9,1.3 1,.5 .3,-1 v -1.8 l -1.3,-.7 -.4,1.2 z m -1,5 -1.7,-1.7 1.6,-2.4 .8,.3 .2,1.1 1,.8 v 1.1 l -1,1 z"/><path class="gm-usmap-state" d="m 822.9,269.3 0,-1.7 h -.8 l 0,1.8 z m 11.8,-3.9 1.2,-2.2 .1,-2.5 -.6,-.6 -.7,.9 -.2,2.1 -.8,1.4 -.3,1.1 -4.6,1.6 -.7,.8 -1.3,.2 -.4,.9 -1.3,.6 -.3,-2.5 .4,-.7 -.8,-.5 .2,-1.5 -1.6,1 v -2 l 1.2,-.3 -1.9,-.4 -.7,-.8 .4,-1.3 -.8,-.6 -.7,1.6 .5,.8 -.7,.6 -1.1,.5 -2,-1 -.2,-1.2 -1,-1.1 -1.4,-1.7 1.5,-.8 -1,-.6 v -.9 l .6,-1 1.7,-.3 -1.4,-.6 -.1,-.7 -1.3,-.1 -.4,1.1 -.6,.3 .1,-3.4 1,-1 .8,.7 .1,-1.6 -1,-.9 -.9,1.1 -1,1.4 -.6,-1 .2,-2.4 .9,-1 .9,.9 1.2,-.7 -.4,-1.7 -1,1 -.9,-2.1 -.2,-1.7 1.1,-2.4 1.1,-1.4 1.4,-.2 -.5,-.8 .5,-.6 -.3,-.7 .2,-2.1 -1.5,.4 -.8,1.1 1,1.3 -2.6,3.6 -.9,-.4 -.7,.9 -.6,2.2 -1.8,.5 1.3,.6 1.3,1.3 -.2,.7 .9,1.2 -1.1,1 .5,.3 -.5,1.3 v 2.1 l -.5,1.3 .9,1.1 .7,3.4 1.3,1.4 1.6,1.4 .4,2.8 1.6,2 .4,1.4 v 1 h -.7 l -1.5,-1.2 -.4,.2 -1.2,-.2 -1.7,-1.4 -1.4,-.3 -1,.5 -1.2,-.3 -.4,.2 -1.7,-.8 -1,-1 -1,-1.3 -.6,-.2 -.8,.7 -1.6,1.3 -1.1,-.8 -.4,-2.3 .8,-2.1 -.3,-.5 .3,-.4 -.7,-1 1,-.1 1,-.9 .4,-1.8 1.7,-2.6 -2.6,-1.8 -1,1.7 -.6,-.6 h -1 l -.6,-.1 -.4,-.4 .1,-.5 -1.7,-.6 -.8,.3 -1.2,-.1 -.7,-.7 -.5,-.2 -.2,-.7 .6,-.8 v -.9 l -1.2,-.2 -1,-.9 -.9,.1 -1.6,-.3 -.9,-.4 .2,-1.6 -1,-.5 -.2,-.7 h -.7 l -.8,-1.2 .2,-1 -2.6,.4 -2.2,-1.6 -1.4,.3 -.9,1.4 h -1.3 l -1.7,2.9 -3.3,.4 -1.9,-1 -2.6,3.8 -2.2,-.3 -3.1,3.9 -.9,1.6 -1.8,1.6 -1.7,-11.4 60.5,-11.8 7.6,27.1 10.9,-2.3 0,5.3 -.1,3.1 -1,1.8 z m -13.4,-1.8 -1.3,.9 .8,1.8 1.7,.8 -.4,-1.6 z"/><path class="gm-usmap-state" d="m 899.9,174.2 h 3.4 l .9,-.6 .1,-1.3 -1.9,-1.8 .4,1 -1.5,1.5 h -2.3 l .1,.8 z m -9,1.8 -1.2,-.6 1,-.8 .6,-2.1 1.2,-1 .8,-.2 .6,.9 1.1,.2 .6,-.6 .5,1.9 -1.3,.3 -2.8,.7 z m -34.9,-23.4 18.4,-3.8 1,-1.5 .3,-1.7 1.9,-.6 .5,-1.1 1.7,-1.1 1.3,.3 1.7,3.3 1,.4 1.1,-1.3 .8,1.3 v 1.1 l -3,2.4 .2,.8 -.9,1 .4,.8 -1.3,.3 .9,1.2 -.8,.7 .6,1 .9,-.2 .3,-.8 1.1,.6 h 1.8 l 2.5,2.6 .2,2.6 1.8,.1 .8,1.1 .6,2 1,.7 h 1.9 l 1.9,-.1 .8,-.9 1.6,-1.2 1.1,-.3 -1.2,-2.1 -.3,.9 -1.5,-3.6 h -.8 l -.4,.9 -1.2,-1 1.3,-1.1 1.8,.4 2.3,2.1 1.3,2.7 1.2,3.3 -1,2.8 v -1.8 l -.7,-1 -3.5,2.3 -.9,-.3 -1.6,1 -.1,1.2 -2.2,1.2 -2,2.1 -2,1.9 h -1.2 l 3.3,-3.3 .5,-1.9 -.5,-.6 -.3,-1.3 -.9,-.1 -.1,1.3 -1,1.2 h -1.2 l -.3,1.1 .4,1.2 -1.2,1.1 -1.1,-.2 -.4,1 -1.4,-3 -1.3,-1.1 -2.6,-1.3 -.6,-2.2 h -.8 l -.7,-2.6 -6.5,2 -.1,-.3 -14.9,3.4 v .7 l -.9,.3 -.5,-.7 -10.5,2.4 -.7,-1 .5,-15 z"/><path class="gm-usmap-state" d="m 663.3,209.8 .1,1.4 21.4,-3.5 .5,-1.2 3.9,-5.9 v -4.3 l .8,-2.1 2.2,-.8 2,-7.8 1,-.5 1,.6 -.2,.6 -1.1,.8 .3,.9 .8,.4 1.9,-1.4 .4,-9.8 -1.6,-2.3 -1.2,-3.7 v -2.5 l -2.3,-4.4 v -1.8 l -1.2,-3.3 -2.3,-3 -2.9,-1 -4.8,3 -2.5,4.6 -.2,.9 -3,3.5 -1.5,-.2 -2.9,-2.8 -.1,-3.4 1.5,-1.9 2,-.2 1.2,-1.7 .2,-4 .8,-.8 1.1,-.1 .9,-1.7 -.2,-9.6 -.3,-1.3 -1.2,-1.2 -1.7,-1 -.1,-1.8 .7,-.6 1.8,.8 -.3,-1.7 -1.9,-2.7 -.7,-1.6 -1.1,-1.1 h -2.2 l -8.1,-2.9 -1.4,-1.7 -3.1,-.3 -1.2,.3 -4.4,-2.3 h -1.4 l .5,1 -2.7,-.1 .1,.6 .6,.6 -2.5,2.1 .1,1.8 1.5,2.3 1.5,.2 v .6 l -1.5,.5 -2.1,-.1 -2.8,2.5 .1,2.5 .4,5.8 -2.2,3.4 .8,-4.5 -.8,-.6 -.9,5.3 -1,-2.3 .5,-2.3 -.5,-1 .6,-1.3 -.6,-1.1 1,-1 v -1.2 l -1.3,.6 -1.3,3.1 -.7,.7 -1.3,2.4 -1.7,-.2 -.1,1.2 h -1.6 l .2,1.5 .2,2 -3,1.2 .1,1.3 1,1.7 -.1,5.2 -1.3,4.4 -1.7,2.5 1.2,1.4 .8,3.5 -1,2.5 -.2,2.1 1.7,3.4 2.5,4.9 1.2,1.9 1.6,6.9 -.1,8.8 -.9,3.9 -2,3.2 -.9,3.7 -2,3 -1.2,1 z m -95.8,-96.8 3,3.8 17,3.8 1.4,1 4,.8 .7,.5 2.8,-.2 4.9,.8 1.4,1.5 -1,1 .8,.8 3.8,.7 1.2,1.2 .1,4.4 -1.3,2.8 2,.1 1,-.8 .9,.8 -1.1,3.1 1,1.6 1.2,.3 .8,-1.8 2.9,-4.6 1.6,-6 2.3,-2 -.5,-1.6 .5,-.9 1,1.6 -.3,2.2 2.9,-2.2 .2,-2.3 2.1,.6 .8,-1.6 .7,.6 -.7,1.5 -1,.5 -1,2 1.4,1.8 1.1,-.5 -.5,-.7 1,-1.5 1.9,-1.7 h .8 l .2,-2.6 2,-1.8 7.9,-.5 1.9,-3.1 3.8,-.3 3.8,1.2 4.2,2.7 .7,-.2 -.2,-3.5 .7,-.2 4.5,1.1 1.5,-.2 2.9,-.7 1.7,.4 1.8,.1 v -1.1 l -.7,-.9 -1.5,-.2 -1.1,-.8 .5,-1.4 -.8,-.3 -2.6,.1 -.1,-1 1.1,-.8 .6,.8 .5,-1.8 -.7,-.7 .7,-.2 -1.4,-1.3 .3,-1.3 .1,-1.9 h -1.3 l -1.5,1 -1.9,.1 -.5,1.8 -1.9,.2 -.3,-1.2 -2.2,.1 -1,1.2 -.7,-.1 -.2,-.8 -2.6,.4 -.1,-4.8 1,-2 -.7,-.1 -1.8,1.1 h -2.2 l -3.8,2.7 -6.2,.3 -4.1,.8 -1.9,1.5 -1.4,1.3 -2.5,1.7 -.3,.8 -.6,-1.7 -1.3,-.6 v .6 l .7,.7 v 1.3 l -1.5,-.6 h -.6 l -.3,1.2 -2,-1.9 -1.3,-.2 -1.3,1.5 -3.2,-.1 -.5,-1.4 -2,-1.9 -1.3,-1.6 v -.7 l -1.1,-1.4 -2.6,-1.2 -3.3,-.1 -1.1,-.9 h -1.4 l -.7,.4 -2.2,2.2 -.7,1.1 -1,-.7 .2,-1 .8,-2.1 3.2,-5 .8,-.2 1.7,-1.9 .7,-1.6 3,-.6 .8,-.6 -.1,-1 -.5,-.5 -4.5,.2 -2,.5 -2.6,1.2 -1.2,1.2 -1.7,2.2 -1.8,1 -3.3,3.4 -.4,1.6 -7.4,4.6 -4,.5 -1.8,.4 -2.3,3 -1.8,.7 -4.4,2.3 z m 100.7,3.8 3.8,.1 .6,-.5 -.2,-2 -1.7,-1.8 -1.9,.1 -.1,.5 1.1,.4 -1.6,.8 -.3,1 -.6,-.6 -.4,.8 z m -75.1,-41.9 -2.3,.2 -2.7,1.9 -7.1,5.3 .8,1 1.8,.3 2.8,-2 -1.1,-.5 2.3,-1.6 h 1 l 3,-1.9 -.1,-.9 z m 41.1,62.8 v 1 l 2.1,1.6 -.2,-2.4 z m -.7,2.8 1.1,.1 v .9 h -1 z m 21.4,-21.3 v .9 l .8,-.2 v -.5 z m 4.7,3.1 -.1,-1.1 -1.6,-.2 -.6,-.4 h -.9 l -.4,.3 .9,.4 1.1,1.1 z m -18,1.2 -.1,1.1 -.3,.7 .2,2.2 .4,.3 .7,.1 .5,-.9 .1,-1.6 -.3,-.6 -.1,-1.1 z"/><path class="gm-usmap-state" d="m 464.7,68.6 -1.1,2.8 .8,1.4 -.3,5.1 -.5,1.1 2.7,9.1 1.3,2.5 .7,14 1,2.7 -.4,5.8 2.9,7.4 .3,5.8 -.1,2.1 -.1,2.2 -.9,2 -3.1,1.9 -.3,1.2 1.7,2.5 .4,1.8 2.6,.6 1.5,1.9 -.2,39.5 h 28.2 l 36.3,-.9 18.6,-.7 -1.1,-4.5 -.2,-3 -2.2,-3 -2.8,-.7 -5.2,-3.6 -.6,-3.3 -6.3,-3.1 -.2,-1.3 h -3.3 l -2.2,-2.6 -2,-1.3 .7,-5.1 -.9,-1.6 .5,-5.4 1,-1.8 -.3,-2.7 -1.2,-1.3 -1.8,-.3 v -1.7 l 2.8,-5.8 5.9,-3.9 -.4,-13 .9,.4 .6,-.5 .1,-1.1 .9,-.6 1.4,1.2 .7,-.1 v 0 l -1.2,-2.2 4.3,-3.1 3.1,-3.7 1.6,-.8 4.7,-5.9 6.3,-5.8 3.9,-2.1 6.3,-2.7 7.6,-4.5 -.6,-.4 -3.7,.7 -2.8,.1 -1,-1.6 -1.4,-.9 -9.8,1.2 -1,-2.8 -1.6,-.1 -1.7,.8 -3.7,3.1 h -4.1 l -2.1,-1 -.3,-1.7 -3.9,-.8 -.6,-1.6 -.7,-1.3 -1,.9 -2.6,.1 -9.9,-5.5 h -2.9 l -.8,-.7 -3.1,1.3 -.8,1.3 -3.3,.8 -1.3,-.2 v -1.7 l -.7,-.9 h -5.9 l -.4,-1.4 h -2.6 l -1.1,.4 -2.4,-1.7 .3,-1.4 -.6,-2.4 -.7,-1.1 -.2,-3 -1,-3.1 -2.1,-1.6 h -2.9 l .1,8 -30.9,-.4 z"/><path class="gm-usmap-state" d="m 623.8,468.6 -5,.1 -2.4,-1.5 -7.9,2.5 -.9,-.7 -.5,.2 -.1,1.6 -.6,.1 -2.6,2.7 -.7,-.1 -.6,-.7 -1.2,-1.8 .3,-1.3 -4.8,-6.8 .9,-4.6 1,-1.4 .1,-1.4 -36,2 1.7,-11.9 2.4,-4.8 6,-8.4 -1.8,-2.5 h 2 v -3.3 l -2.4,-2.5 .5,-1.7 -1.2,-1 -1.6,-7.1 .6,-1.4 1.2,-1.5 .5,-3 -1.5,-2.3 -.5,-2.2 .9,-.7 v -.8 l -1.7,-1.1 -.1,-.7 1.6,-.9 -1.2,-1.1 1.7,-7.1 3.4,-1.6 v -.8 l -1.1,-1.4 2.9,-5.4 h 1.9 l 1.5,-1.2 -.3,-5.2 3.1,-4.5 1.8,-.6 -.5,-3.1 38.3,-2.6 1.3,2 -1.3,67 4.4,33.2 z"/><path class="gm-usmap-state" d="m 555.3,248.9 -1.1,-1.1 -.6,-1.6 -1.7,-1.3 -14.3,.8 -27.2,1.2 -25.9,-.1 1.3,1.3 -.3,1.4 2.1,3.7 3.9,6.3 2.9,3 2,.6 .9,-1 1.5,2.1 -.1,1.2 -2.7,2.6 -.5,2.3 2.5,2.5 2.6,4.7 3.2,.7 .5,48.1 .2,10.8 39.1,-.7 39.8,-2 1.6,2.5 v 2.2 l -1.7,1.5 -2.8,5.1 11.2,-.8 1,-2 1.2,-.5 v -.7 l -1.2,-1.1 -.6,-1 1.7,.2 .8,-.7 -1.4,-1.5 1.4,-.5 .1,-1 -.6,-1 v -1.3 l -.7,-.7 .2,-1 h 1.1 l .7,.7 -.3,1 .8,.7 .8,-1 1,-2.7 1.4,.9 .7,-.4 1.2,-4.1 -1,-1 1,-2 .2,-.9 -1.3,-.8 h -2.8 l -1.4,-1.5 -1.8,-3.8 v -1.9 l .8,-.6 .1,-1.3 -1.7,-1.9 -.9,-2.5 -2.7,-4.1 -4.8,-1.3 -7.4,-7.1 -.4,-2.4 2.8,-7.6 -.4,-1.9 1.2,-1.1 v -1.3 l -2.8,-1.5 -3,-.7 -3.4,1.2 -1.3,-2.3 .6,-1.9 -.7,-2.4 -8.6,-8.4 -2.2,-1.5 -2.5,-5.9 -1.2,-5.4 1.4,-3.7 z"/><path class="gm-usmap-state" d="m 247,130.5 57.3,7.9 51,5.3 2,-20.7 5.2,-66.7 -53.5,-5.6 -54.3,-7.7 -65.9,-12.5 -4.8,22 3.7,7.4 -1.6,4.8 3.6,4.8 1.9,.7 3.9,8.3 v 2.1 l 2.3,3 h .9 l 1.4,2.1 h 3.2 v 1.6 l -7.1,17 -.5,4.1 1.4,.5 1.6,2.6 2.8,-1.4 3.6,-2.4 1.9,1.9 .5,2.5 -.5,3.2 2.5,9.7 2.6,3.5 2.3,1.4 .4,3 v 4.1 l 2.3,2.3 1.6,-2.3 6.9,1.6 2.1,-1.2 9,1.7 2.8,-3.3 1.8,-.6 1.2,1.8 1.6,4.1 .9,.1 z"/><path class="gm-usmap-state" d="m 402.5,191.1 38,1.6 3.4,3.2 1.7,.2 2.1,2 1.8,-.1 1.8,-2 1.5,.6 1,-.7 .7,.5 .9,-.4 .7,.4 .9,-.4 1,.5 1.4,-.6 2,.6 .6,1.1 6.1,2.2 1.2,1.3 .9,2.6 1.8,.7 1.5,-.2 .5,.9 v 2.3 l .6,1.7 v 1.4 l 1.3,3.7 2.2,4.3 .2,3.7 2.8,5 .4,6.1 1.3,.7 -.2,2.9 .8,3.3 -.7,2.2 1.8,4.4 1.3,1.3 -.3,1.4 2.1,3.7 3.9,6.3 h -32.4 l -43.7,-1.2 -36,-2 1.4,-22.1 -33.1,-2.4 3.7,-44.2 z"/><path class="gm-usmap-state" d="m 167.6,296.8 -3.4,17.5 -2.4,2.9 h -2 l -1.2,-2.7 -3.7,-1.4 -3.5,.6 -1,13.6 .5,4.9 -.5,2.9 -1.4,3 -70.4,-105 -1.1,-3.5 16.4,-63.1 47,11.2 24.4,5.4 23.3,4.7 z"/><path class="gm-usmap-state" d="m 862.6,93.6 -1.3,.1 -1,-1.1 -1.9,1.4 -.5,6.1 1.2,2.3 -1.1,3.5 2.1,2.8 -.4,1.7 .1,1.3 -1.1,2.1 -1.4,.4 -.6,1.3 -2.1,1 -.7,1.5 1.4,3.4 -.5,2.5 .5,1.5 -1,1.9 .4,1.9 -1.3,1.9 .2,2.2 -.7,1.1 .7,4.5 .7,1.5 -.5,2.6 .9,1.8 -.2,2.5 -.5,1.3 -.1,1.4 2.1,2.6 18.4,-3.8 1,-1.5 .3,-1.7 1.9,-.6 .5,-1.1 1.7,-1.1 1.3,.3 .8,-4.8 -2.3,-1.4 -.8,-2.2 -3.2,-2 -.6,-4 -11.9,-36.8 z"/><path class="gm-usmap-state" d="m 842.5,195.4 -14.6,-4.9 -1.8,2.5 .1,2.2 -3,5.4 1.5,1.8 -.7,2 -1,1 .5,3.6 2.7,.9 1,2.8 2.1,1.1 4.2,3.2 -3.3,2.6 -1.6,2.3 -1.8,3 -1.6,.6 -1.4,1.7 -1,2.2 -.3,2.1 .8,.9 .4,2.3 1.2,.6 2.4,1.5 1.8,.8 1.6,.8 .1,1.1 .8,.1 1.1,-1.2 .8,.4 2.1,.2 -.2,2.9 .2,2.5 1.8,-.7 1.5,-3.9 1.6,-4.8 2.9,-2.8 .6,-3.5 -.6,-1.2 1.7,-2.9 v -1.2 l -.7,-1.1 1.2,-2.7 -.3,-3.6 -.6,-8.2 -1.2,-1.4 v 1.4 l .5,.6 h -1.1 l -.6,-.4 -1.3,-.2 -.9,.6 -1.2,-1.6 .7,-1.7 v -1 l 1.7,-.7 .8,-2.1 z"/><path class="gm-usmap-state" d="m 357.5,332.9 h -.8 l -7.9,99.3 -31.8,-2.6 -34.4,-3.6 -.3,3 2,2.2 -30.8,-4.1 -1.4,10.2 -15.7,-2.2 17.4,-124.1 52.6,6.5 51.7,4.8 z"/><path class="gm-usmap-state" d="m 872.9,181.6 -1.3,.1 -.5,1 z m -30.6,22.7 .7,.6 1.3,-.3 1.1,.3 .9,-1.3 h 1.9 l 2.4,-.9 5.1,-2.1 -.5,-.5 -1.9,.8 -2,.9 .2,-.8 2.6,-1.1 .8,-1 1.2,.1 4.1,-2.3 v .7 l -4.2,3 4.5,-2.8 1.7,-2.2 1.5,-.1 4.5,-3.1 3.2,-3.1 3,-2.3 1,-1.2 -1.7,-.1 -1,1.2 -.2,.7 -.9,.7 -.8,-1.1 -1.7,1 -.1,.9 -.9,-.2 .5,-.9 -1.2,-.7 -.6,.9 .9,.3 .2,.5 -.3,.5 -1.4,2.6 h -1.9 l .9,-1.8 .9,-.6 .3,-1.7 1.4,-1.6 .9,-.8 1.5,-.7 -1.2,-.2 -.7,.9 h -.7 l -1.1,.8 -.2,1 -2.2,2.1 -.4,.9 -1.4,.9 -7.7,1.9 .2,.9 -.9,.7 -2,.3 -1,-.6 -.2,1.1 -1.1,-.4 .1,1 -1.2,-.1 -1.2,.5 -.2,1.1 h -1 l .2,1 h -.7 l .2,1 -1.8,.4 -1.5,2.3 z m -.8,-.4 -1.6,.4 v 1 l -.7,1.6 .6,.7 2.4,-2.3 -.1,-.9 z m -10.1,-95.2 -.6,1.9 1.4,.9 -.4,1.5 .5,3.2 2.2,2.3 -.4,2.2 .6,2 -.4,1 -.3,3.8 3.1,6.7 -.8,1.8 .9,2.2 .9,-1.6 1.9,1.5 3,14.2 -.5,2 1.1,1 -.5,15 .7,1 2.8,16.3 1.8,1.5 -3.5,3.4 1.7,2.2 -1.3,3.3 -1.5,1.7 -1.5,2.3 -.2,-.7 .4,-5.9 -14.6,-4.9 -1.6,-1.1 -1.9,.3 -3,-2.2 -3,-5.8 h -2 l -.4,-1.5 -1.7,-1.1 -70.5,13.9 -.8,-6 4.3,-3.9 .6,-1.7 3.9,-2.5 .6,-2.4 2.3,-2 .8,-1.1 -1.7,-3.3 -1.7,-.5 -1.8,-3 -.2,-3.2 7.6,-3.9 8.2,-1.6 h 4.4 l 3.2,1.6 .9,-.1 1.8,-1.6 3.4,-.7 h 3 l 2.6,-1.3 2.5,-2.6 2.4,-3.1 1.9,-.4 1.1,-.5 .4,-3.2 -1.4,-2.7 -1.2,-.7 2,-1.3 -.1,-1.8 h -1.5 l -2.3,-1.4 -.1,-3.1 6.2,-6.1 .7,-2.4 3.7,-6.3 5.9,-6.4 2.1,-1.7 2.5,.1 20.6,-5.2 z"/><path class="gm-usmap-state" d="m 829,300.1 -29.1,6.1 -39.4,7.3 -29.4,3.5 v 5.2 l -1.5,-.1 -1.4,1.2 -2.4,5.2 -2.6,-1.1 -3.5,2.5 -.7,2.1 -1.5,1.2 -.8,-.8 -.1,-1.5 -.8,-.2 -4,3.3 -.6,3.4 -4.7,2.4 -.5,1.2 -3.2,2.6 -3.6,.5 -4.6,3 -.8,4.1 -1.3,.9 -1.5,-.1 -1.4,1.3 -.1,4.9 21.4,-3 4.4,-1.9 1.3,-.1 7.3,-4.3 23.2,-2.2 .4,.5 -.2,1.4 .7,.3 1.2,-1.5 3.3,3 .1,2.6 19.7,-2.8 24.5,17.1 4,-2.2 3,-.7 h 1.7 l 1.1,1.1 .8,-2 .6,-5 1.7,-3.9 5.4,-6.1 4.1,-3.5 5.4,-2.3 2.5,-.4 1.3,.4 .7,1.1 3.3,-6.6 3.3,-5.3 -.7,-.3 -4.4,6.8 -.5,-.8 2,-2.2 -.4,-1.5 -2,-.5 1,1.3 -1.2,.1 -1.2,-1.8 -1.2,2 -1.6,.2 1,-2.7 .7,-1.7 -.2,-2.9 -2.2,-.1 .9,-.9 1.1,.3 2.7,.1 .8,-.5 h 2.3 l 2,-1.9 .2,-3.2 1.3,-1.4 1.2,-.2 1.3,-1 -.5,-3.7 -2.2,-3.8 -2.7,-.2 -.9,1.6 -.5,-1 -2.7,.2 -1.2,.4 -1.9,1.2 -.3,-.4 h -.9 l -1.8,1.2 -2.6,.5 v -1.3 l .8,-1 1,.7 h 1 l 1.7,-2.1 3.7,-1.7 2,-2.2 h 2.4 l .8,1.3 1.7,.8 -.5,-1.5 -.3,-1.6 -2.8,-3.1 -.3,-1.4 -.4,1 -.9,-1.3 z m 7,31 2.7,-2.5 4.6,-3.3 v -3.7 l -.4,-3.1 -1.7,-4.2 1.5,1.4 1,3.2 .4,7.6 -1.7,.4 -3.1,2.4 -3.2,3.2 z m 1.9,-19.3 -.9,-.2 v 1 l 2.5,2.2 -.2,-1.4 z m 2.9,2.1 -1.4,-2.8 -2.2,-3.4 -2.4,-3 -2.2,-4.3 -.8,-.7 2.2,4.3 .3,1.3 3.4,5.5 1.8,2.1 z"/><path class="gm-usmap-state" d="m 464.7,68.6 -1.1,2.8 .8,1.4 -.3,5.1 -.5,1.1 2.7,9.1 1.3,2.5 .7,14 1,2.7 -.4,5.8 2.9,7.4 .3,5.8 -.1,2.1 -29.5,-.4 -46,-2.1 -39.2,-2.9 5.2,-66.7 44.5,3.4 55.3,1.6 z"/><path class="gm-usmap-state" d="m 685.7,208.8 1.9,-.4 3,1.3 2.1,.6 .7,.9 h 1 l 1,-1.5 1.3,.8 h 1.5 l -.1,1 -3.1,.5 -2,1.1 1.9,.8 1.6,-1.5 2.4,-.4 2.2,1.5 1.5,-.1 2.5,-1.7 3.6,-2.1 5.2,-.3 4.9,-5.9 3.8,-3.1 9.3,-5.1 4.9,29.9 -2.2,1.2 1.4,2.1 -.1,2.2 .6,2 -1.1,3.4 -.1,5.4 -1,3.6 .5,1.1 -.4,2.2 -1.1,.5 -2,3.3 -1.8,2 h -.6 l -1.8,1.7 -1.3,-1.2 -1.5,1.8 -.3,1.2 h -1.3 l -1.3,2.2 .1,2.1 -1,.5 1.4,1.1 v 1.9 l -1,.2 -.7,.8 -1,.5 -.6,-2.1 -1.6,-.5 -1,2.3 -.3,2.2 -1.1,1.3 1.3,3.6 -1.5,.8 -.4,3.5 h -1.5 l -3.2,1.4 -1.2,-2.1 -3.5,-1.4 -.8,-2.9 -.5,-.8 -3.4,1.8 -.6,1.7 h -.9 l -1.3,.7 -1.2,-.8 -3,-.8 -1.9,.8 v 1 l -2.2,-.2 -1.9,-2.1 -2.3,.2 -4.1,-.7 v -1 l -2.2,-3.4 -2.9,-1.2 -1.9,.8 -2.2,-.1 -1.3,-.5 -6.6,-57.2 21.4,-3.5 z"/><path class="gm-usmap-state" d="m 501.5,398.6 -4.6,-3.8 -2.2,-.9 -.5,1.6 -5.1,.3 -.6,-1.5 -5,2.5 -1.6,-.7 -3.7,.3 -.6,1.7 -3.6,.9 -1.3,-1.2 -1.2,.1 -2,-1.8 -2.1,.7 -2,-.5 -1.8,-2 -2.5,4.2 -1.2,.8 -1,-1.8 .3,-2 -1.2,-.7 -2.3,2.5 -1.7,-1.2 -.1,-1.5 -1.3,.5 -2.6,-1.7 -3,2.6 -2.3,-1.1 .7,-2.1 -2.3,.1 -1.9,-3 -3.5,-1.1 -2,2.3 -2.3,-2.2 -1.4,.4 -2,.1 -3.5,-1.9 -2.3,.1 -1.2,-.7 -.5,-2.9 -2.3,-1.7 -1.1,1.5 -1.4,-1 -1.2,-.4 -1.1,1 -1.5,-.3 -2.5,-3 -2.7,-1.3 1.4,-42.7 -52.6,-3.2 .6,-10.6 16.5,1 67.7,2.9 62,.1 .2,10.8 4.1,24.4 -.7,39 z"/><path class="gm-usmap-state" d="m 93.9,166.5 47,11.2 8.5,-37.3 2.9,-5.8 .4,-2.1 .8,-.9 -.9,-2 -2.9,-1.2 .2,-4.2 4,-5.8 2.5,-.8 1.6,-2.3 -.1,-1.6 1.8,-1.6 3.2,-5.5 4.2,-4.8 -.5,-3.2 -3.5,-3.1 -1.6,-3.6 -30.3,-7.3 -2.8,1 -5.4,-.9 -1.8,-.9 -1.5,1.2 -3.3,-.4 -4.5,.5 -.9,.7 -4.2,-.4 -.8,-1.6 -1.2,-.2 -4.4,1.3 -1.6,-1.1 -2.2,.8 -.2,-1.8 -2.3,-1.2 -1.5,-.2 -1,-1.1 -3,.3 -1.2,-.8 h -1.2 l -1.2,.9 -5.5,.7 -6.6,-4.2 1.1,-5.6 -.4,-4.1 -3.2,-3.7 -3.7,.1 -.4,-1.1 .4,-1.2 -.7,-.8 -1,.1 -1.1,1.3 -1.5,-.2 -.5,-1.1 -1,-.1 -.7,.6 -2,-1.9 v 4.3 l -1.3,1.3 -1.1,3.5 -.1,2.3 -4.5,12.3 -13.2,31.3 -3.2,4.6 -1.6,-.1 .1,2.1 -5.2,7.1 -.3,3.3 1,1.3 .1,2.4 -1.2,1.1 -1.2,3 .1,5.7 1.2,2.9 z"/><path class="gm-usmap-state" d="m 826.3,189.4 -1.9,.3 -3,-2.2 -3,-5.8 h -2 l -.4,-1.5 -1.7,-1.1 -70.5,13.9 -.8,-6 -4.2,3.4 -.9,.1 -2.7,3 -3.3,1.7 4.9,29.9 3.2,19.7 17.4,-2.9 60.5,-11.8 1.2,-2.1 1.5,-1.1 1.6,-.3 1.6,.6 1.4,-1.7 1.6,-.6 1.8,-3 1.6,-2.3 3.3,-2.6 -4.2,-3.2 -2.1,-1.1 -1,-2.8 -2.7,-.9 -.5,-3.6 1,-1 .7,-2 -1.5,-1.8 3,-5.4 -.1,-2.2 1.8,-2.5 z"/><path class="gm-usmap-state" d="m 883.2,170.7 -1.3,-1.1 -2.6,-1.3 -.6,-2.2 h -.8 l -.7,-2.6 -6.5,2 3.2,12.3 -.4,1.1 .4,1.8 5.6,-3.6 .1,-3 -.8,-.8 .4,-.6 -.1,-1.3 -.9,-.7 1.2,-.4 -.9,-1.6 1.8,.7 .3,1.4 .7,1.2 -1.4,-.8 1.1,1.7 -.3,1.2 -.6,-1.1 v 2.5 l .6,-.9 .4,.9 1.3,-1.5 -.2,-2.5 1.4,3.1 1,-.9 z m -4.7,12.2 h .9 l .5,-.6 -.8,-1.3 -.7,.7 z"/><path class="gm-usmap-state" d="m 772.3,350.2 -19.7,2.8 -.1,-2.6 -3.3,-3 -1.2,1.5 -.7,-.3 .2,-1.4 -.4,-.5 -23.2,2.2 -7.3,4.3 -1.3,.1 -4.4,1.9 -.1,1.9 -1.9,1 -1.4,3.2 .2,1.3 6.1,3.8 2.6,-.3 3.1,4 .4,1.7 4.2,5.1 2.6,1.7 1.4,.2 2.2,1.6 1.1,2.2 2,1.6 1.8,.5 2.7,2.7 .1,1.4 2.6,2.8 5,2.3 3.6,6.7 .3,2.7 3.9,2.1 2.5,4.8 .8,3.1 4.2,.4 .8,-1.5 h .6 l 1.8,-1.5 .5,-2 3.2,-2.1 .3,-2.4 -1.2,-.9 .8,-.7 .8,.4 1.3,-.4 1.8,-2.1 3.8,-1.8 1.6,-2.4 .1,-.7 4.8,-4.4 -.1,-.5 -.9,-.8 1.1,-1.5 h .8 l .4,.5 .7,-.8 h 1.3 l .6,-1.5 2.3,-2.1 -.3,-5.4 .8,-2.3 3.6,-6.2 2.4,-2.2 2.2,-1.1 z"/><path class="gm-usmap-state" d="m 396.5,125.9 46,2.1 29.5,.4 -.1,2.2 -.9,2 -3.1,1.9 -.3,1.2 1.7,2.5 .4,1.8 2.6,.6 1.5,1.9 -.2,39.5 -2.2,-.1 -.1,1.6 1.3,1.5 -.1,1.1 -1,.5 .4,1.6 1.3,.4 .7,2 -1.7,5.1 -1,4.3 1.3,1.2 .3,1.3 .7,1.7 -1.5,.2 -1.8,-.7 -.9,-2.6 -1.2,-1.3 -6.1,-2.2 -.6,-1.1 -2,-.6 -1.4,.6 -1,-.5 -.9,.4 -.7,-.4 -.9,.4 -.7,-.5 -1,.7 -1.5,-.6 -1.8,2 -1.8,.1 -2.1,-2 -1.7,-.2 -3.4,-3.2 -38,-1.6 -51.1,-3.5 3.9,-43.9 2,-20.7 z"/><path class="gm-usmap-state" d="m 620.9,365.1 45.7,-4 22.9,-2.9 .1,-4.9 1.4,-1.3 1.5,.1 1.3,-.9 .8,-4.1 4.6,-3 3.6,-.5 3.2,-2.6 .5,-1.2 4.7,-2.4 .6,-3.4 4,-3.3 .8,.2 .1,1.5 .8,.8 1.5,-1.2 .7,-2.1 3.5,-2.5 2.6,1.1 2.4,-5.2 1.4,-1.2 1.5,.1 0,-5.2 .3,-.7 -4.6,.5 -.2,1 -28.9,3.3 -5.6,1.4 -20.5,1.4 -5.2,.8 -17.4,1 -2.6,.8 -22.6,2 -.7,-.6 h -3.7 l 1.2,3.2 -.6,.9 -23.3,1.5 -.8,1 -.8,-.7 h -1 v 1.3 l .6,1 -.1,1 -1.4,.5 1.4,1.5 -.8,.7 -1.7,-.2 .6,1 1.2,1.1 v .7 l -1.2,.5 -1,2 .1,.6 1.4,1 -.4,.7 h -1.5 v .5 l .9,.9 .1,.8 -1.4,.2 -.5,.8 -1.6,.2 -.9,.9 .6,.9 1.1,-.1 .5,.9 -1.6,1.3 .4,1.5 -2,-.6 -.1,.7 .4,1.1 -.3,1.4 -1.3,-.8 -.8,.8 1.1,.1 .1,1.5 -.6,1 1.1,.9 -.3,1.5 .8,.7 -.7,1 -1.2,-.5 -.9,2.2 -1.6,.7 z"/><path class="gm-usmap-state" d="m 282.3,429 .3,-3 34.4,3.6 31.8,2.6 7.9,-99.3 .8,0 52.6,3.2 -1.4,42.7 2.7,1.3 2.5,3 1.5,.3 1.1,-1 1.2,.4 1.4,1 1.1,-1.5 2.3,1.7 .5,2.9 1.2,.7 2.3,-.1 3.5,1.9 2,-.1 1.4,-.4 2.3,2.2 2,-2.3 3.5,1.1 1.9,3 2.3,-.1 -.7,2.1 2.3,1.1 3,-2.6 2.6,1.7 1.3,-.5 .1,1.5 1.7,1.2 2.3,-2.5 1.2,.7 -.3,2 1,1.8 1.2,-.8 2.5,-4.2 1.8,2 2,.5 2.1,-.7 2,1.8 1.2,-.1 1.3,1.2 3.6,-.9 .6,-1.7 3.7,-.3 1.6,.7 5,-2.5 .6,1.5 5.1,-.3 .5,-1.6 2.2,.9 4.6,3.8 6.4,1.9 2.6,2.3 2.8,-1.3 3.2,.8 .2,11.9 .5,19.9 .7,3.4 2.6,2.8 .7,5.4 3.8,4.6 .8,4.3 h 1 l -.1,7.3 -3.3,6.4 1.3,2.3 -1.3,1.5 .7,3 -.1,4.3 -2.2,3.5 -.1,.8 -1.7,1.2 1,1.8 1.2,1.1 -3.5,.3 -8.4,3.9 -3.5,1.4 -1.8,1.8 -.7,-.5 2.1,-2.3 1.8,-.7 .5,-.9 -2.9,-.1 -.7,-.8 .8,-2 -.9,-1.8 h -.6 l -2.4,1.3 -1.9,2.6 .3,1.7 3.3,3.4 1.3,.3 v .8 l -2.3,1.6 -4.9,4 -4,3.9 -3.2,1.4 -5,3 -3.7,2 -4.5,1.9 -4.1,2.5 3.2,-3 v -1.1 l .6,-.8 -.2,-1.8 -1.5,-.1 -1.1,1.5 -2.6,1.3 -1.8,-1.2 -.3,-1.7 h -1.5 l .8,2.2 1.4,.7 1.2,.9 1.8,1.6 -.7,.8 -3.9,1.7 -1.7,.1 -1.2,-1.2 -.5,2.1 .5,1.1 -2.7,2 -1.5,.2 -.8,.7 -.4,1.7 -1.8,3.3 -1.6,.7 -1.6,-.6 -1.8,1.1 .3,1.4 1.3,.8 1,.8 -1.8,3.5 -.3,2.8 -1,1.7 -1.4,1 -2.9,.4 1.8,.6 1.9,-.6 -.4,3.2 -1.1,-.1 .2,1.2 .3,1.4 -1.3,.9 v 3.1 l 1.6,1.4 .6,3.1 -.4,2.2 -1,.4 .4,1.5 1.1,.4 .8,1.7 v 2.6 l 1.1,2.1 2.2,2.6 -.1,.7 -2.2,-.2 -1.6,1.4 .2,1.4 -.9,-.3 -1.4,-.2 -3.4,-3.7 -2.3,-.6 h -7.1 l -2.8,-.8 -3.6,-3 -1.7,-1 -2.1,.1 -3.2,-2.6 -5.4,-1.6 v -1.3 l -1.4,-1.8 -.9,-4.7 -1.1,-1.7 -1.7,-1.4 v -1.6 l -1.4,-.6 .6,-2.6 -.3,-2.2 -1.3,-1.4 .7,-3 -.8,-3.2 -1.7,-1.4 h -1.1 l -4,-3.5 .1,-1.9 -.8,-1.7 -.8,-.2 -.9,-2.4 -2,-1.6 -2.9,-2.5 -.2,-2.1 -1,-.7 .2,-1.6 .5,-.7 -1.4,-1.5 .1,-.7 -2,-2.2 .1,-2.1 -2.7,-4.9 -.1,-1.7 -1.8,-3.1 -5.1,-4.8 v -1.1 l -3.3,-1.7 -.1,-1.8 -1.2,-.4 v -.7 l -.8,-.2 -2.1,-2.8 h -.8 l -.7,-.6 -1.3,1.1 h -2.2 l -2.6,-1.1 h -4.6 l -4.2,-2.1 -1.3,1.9 -2.2,-.6 -3.3,1.2 -1.7,2.8 -2,3.2 -1.1,4.4 -1.4,1.2 -1.1,.1 -.9,1.6 -1.3,.6 -.1,1.8 -2.9,.1 -1.8,-1.5 h -1 l -2,-2.9 -3.6,-.5 -1.7,-2.3 -1.3,-.2 -2.1,-.8 -3.4,-3.4 .2,-.8 -1.6,-1.2 -1,-.1 -3.4,-3.1 -.1,-2 -2.3,-4 .2,-1.6 -.7,-1.3 .8,-1.5 -.1,-2.4 -2.6,-4.1 -.6,-4.2 -1.6,-1.6 v -1 l -1.2,-.2 -.7,-1.1 -2.4,-1.7 -.9,-.1 -1.9,-1.6 v -1.1 l -2.9,-1.8 -.6,-2.1 -2.6,-2.3 -3.2,-4.4 -3,-1.3 -2.1,-1.8 .2,-1.2 -1.3,-1.4 -1.7,-3.7 -2.4,-1 z m 174.9,138.3 .8,.1 -.6,-4.8 -3.5,-12.3 -.2,-8.1 4.9,-10.5 6.1,-8.2 7.2,-5.1 v -.7 h -.8 l -2.6,1 -3.6,2.3 -.7,1.5 -8.2,11.6 -2.8,7.9 v 8.8 l 3.6,12 z"/><path class="gm-usmap-state" d="m 233.2,217.9 3.3,-21.9 -47.9,-8.2 -21,109 46.2,8.2 40,6 11.5,-88.3 z"/><path class="gm-usmap-state" d="m 859.1,102.4 -1.1,3.5 2.1,2.8 -.4,1.7 .1,1.3 -1.1,2.1 -1.4,.4 -.6,1.3 -2.1,1 -.7,1.5 1.4,3.4 -.5,2.5 .5,1.5 -1,1.9 .4,1.9 -1.3,1.9 .2,2.2 -.7,1.1 .7,4.5 .7,1.5 -.5,2.6 .9,1.8 -.2,2.5 -.5,1.3 -.1,1.4 2.1,2.6 -12.4,2.7 -1.1,-1 .5,-2 -3,-14.2 -1.9,-1.5 -.9,1.6 -.9,-2.2 .8,-1.8 -3.1,-6.7 .3,-3.8 .4,-1 -.6,-2 .4,-2.2 -2.2,-2.3 -.5,-3.2 .4,-1.5 -1.4,-.9 .6,-1.9 -.8,-1.7 27.3,-6.9 z"/><path class="gm-usmap-state" d="m 834.7,265.4 -1.1,2.8 .5,1.1 .4,-1.1 .8,-3.1 z m -34.6,-7 -.7,-1 1,-.1 1,-.9 .4,-1.8 -.2,-.5 .1,-.5 -.3,-.7 -.6,-.5 -.4,-.1 -.5,-.4 -.6,-.6 h -1 l -.6,-.1 -.4,-.4 .1,-.5 -1.7,-.6 -.8,.3 -1.2,-.1 -.7,-.7 -.5,-.2 -.2,-.7 .6,-.8 v -.9 l -1.2,-.2 -1,-.9 -.9,.1 -1.6,-.3 -.4,.7 -.4,1.6 -.5,2.3 -10,-5.2 -.2,.9 .9,1.6 -.8,2.3 .1,2.9 -1.2,.8 -.5,2.1 -.9,.8 -1.4,1.8 -.9,.8 -1,2.5 -2.4,-1.1 -2.3,8.5 -1.3,1.6 -2.8,-.5 -1.3,-1.9 -2.3,-.7 -.1,4.7 -1.4,1.7 .4,1.5 -2.1,2.2 .4,1.9 -3.7,6.3 -1,3.3 1.5,1.2 -1.5,1.9 .1,1.4 -2.3,2 -.7,-1.1 -4.3,3.1 -1.5,-1 -.6,1.4 .8,.5 -.5,.9 -5.5,2.4 -3,-1.8 -.8,1.7 -1.9,1.8 -2.3,.1 -4.4,-2.3 -.1,-1.5 -1.5,-.7 .8,-1.2 -.7,-.6 -4.9,6.6 -2.9,1 -3,3 -.4,2.2 -2.1,1.3 -.1,1.7 -1.4,1.4 -1.8,.5 -.5,1.9 -1,.4 -6.9,4.2 28.9,-3.3 .2,-1 4.6,-.5 -.3,.7 29.4,-3.5 39.4,-7.3 29.1,-6.1 -.6,-1.2 .4,-.1 .9,.9 -.1,-1.4 -.3,-1.9 1.6,1.2 .9,2.1 v -1.3 l -3.4,-5.5 v -1.2 l -.7,-.8 -1.3,.7 .5,1.4 h -.8 l -.4,-1 -.6,.9 -.9,-1.1 -2.1,-.1 -.2,.7 1.5,2.1 -1.4,-.7 -.5,-1 -.4,.8 -.8,.1 -1.5,1.7 .3,-1.6 v -1.4 l -1.5,-.7 -1.8,-.5 -.2,-1.7 -.6,-1.3 -.6,1.1 -1.7,-1 -2,.3 .2,-.9 1.5,-.2 .9,.5 1.7,-.8 .9,.4 .5,1 v .7 l 1.9,.4 .3,.9 .9,.4 .9,1.2 1.4,-1.6 h .6 l -.1,-2.1 -1.3,1 -.6,-.9 1.5,-.2 -1.2,-.9 -1.2,.6 -.1,-1.7 -1.7,.2 -2.2,-1.1 -1.8,-2.2 3.6,2.2 .9,.3 1.7,-.8 -1.7,-.9 .6,-.6 -1,-.5 .8,-.2 -.3,-.9 1.1,.9 .4,-.8 .4,1.3 1.2,.8 .6,-.5 -.5,-.6 -.1,-2.5 -1.1,-.1 -1.6,-.8 .9,-1.1 -2,-.1 -.4,-.5 -1.4,.6 -1.4,-.8 -.5,-1.2 -2.1,-1.2 -2.1,-1.8 -2.2,-1.9 3,1.3 .9,1.2 2.1,.7 2.3,2.5 .2,-1.7 .6,1.3 2.3,.5 v -4 l -.8,-1.1 1.1,.4 .1,-1.6 -3.1,-1.4 -1.6,-.2 -1.3,-.2 .3,-1.2 -1.5,-.3 -.1,-.6 h -1.8 l -.2,.8 -.7,-1 h -2.7 l -1,-.4 -.2,-1 -1.2,-.6 -.4,-1.5 -.6,-.4 -.7,1.1 -.9,.2 -.9,.7 h -1.5 l -.9,-1.3 .4,-3.1 .5,-2.4 .6,.5 z m 21.9,11.6 .9,-.1 0,-.6 -.8,.1 z m 7.5,14.2 -1,2.7 1.2,-1.3 z m -1.8,-15.3 .7,.3 -.2,1.9 -.5,-.5 -1.3,1 1,.4 -1.8,4.4 .1,8.1 1.9,3.1 .5,-1.5 .4,-2.7 -.3,-2.3 .7,-.9 -.2,-1.4 1.2,-.6 -.6,-.5 .5,-.7 .8,1.1 -.2,1.1 -.4,3.9 1.1,-2.2 .4,-3.1 .1,-3 -.3,-2 .6,-2.3 1.1,-1.8 .1,-2.2 .3,-.9 -4.6,1.6 -.7,.8 z"/><path class="gm-usmap-state" d="m 161.9,83.6 .7,4 -1.1,4.3 -30.3,-7.3 -2.8,1 -5.4,-.9 -1.8,-.9 -1.5,1.2 -3.3,-.4 -4.5,.5 -.9,.7 -4.2,-.4 -.8,-1.6 -1.2,-.2 -4.4,1.3 -1.6,-1.1 -2.2,.8 -.2,-1.8 -2.3,-1.2 -1.5,-.2 -1,-1.1 -3,.3 -1.2,-.8 h -1.2 l -1.2,.9 -5.5,.7 -6.6,-4.2 1.1,-5.6 -.4,-4.1 -3.2,-3.7 -3.7,.1 -.4,-1.1 .4,-1.2 -.7,-.8 -1,.1 -2.1,-1.5 -1.2,.4 -2,-.1 -.7,-1.5 -1.6,-.3 2.5,-7.5 -.7,6 .5,.5 v -2 l .8,-.2 1.1,2.3 -.5,-2.2 1.2,-4.2 1.8,.4 -1.1,-2 -1,.3 -1.5,-.4 .2,-4.2 .2,1.5 .9,.5 .6,-1.6 h 3.2 l -2.2,-1.2 -1.7,-1.9 -1.4,1.6 1.2,-3.1 -.3,-4.6 -.2,-3.6 .9,-6.1 -.5,-2 -1.4,-2.1 .1,-4 .4,-2.7 2,-2.3 -.7,-1.4 .2,-.6 .9,.1 7.8,7.6 4.7,1.9 5.1,2.5 3.2,-.1 .2,3 1,-1.6 h .7 l .6,2.7 .5,-2.6 1.4,-.2 .5,.7 -1.1,.6 .1,1.6 .7,-1.5 h 1.1 l -.4,2.6 -1.1,-.8 .4,1.4 -.1,1.5 -.8,.7 -2.5,2.9 1.2,-3.4 -1.6,.4 -.4,2.1 -3.8,2.8 -.4,1 -2.1,2.2 -.1,1 h 2.2 l 2.4,-.2 .5,-.9 -3.9,.5 v -.6 l 2.6,-2.8 1.8,-.8 1.9,-.2 1,-1.6 3,-2.3 v -1.4 h 1.1 l .1,4 h -1.5 l -.6,.8 -1.1,-.9 .3,1.1 v 1.7 l -.7,.7 -.3,-1.6 -.8,.8 .7,.6 -.9,1.1 h 1.3 l .7,-.5 .1,2 -1,1.9 -.9,1 -.1,1.8 -1,-.2 -.2,-1.4 .9,-1.1 -.8,-.5 -.8,.7 -.7,2.2 -.8,.9 -.1,-2 .8,-1.1 -.2,-1.1 -1.2,1.2 .1,2.2 -.6,.4 -2.1,-.4 -1.3,1.2 2.2,-.6 -.2,2.2 1,-1.8 .4,1.4 .5,-1 .7,1.8 h .7 l .7,-.8 .6,-.1 2,-1.9 .2,-1.2 .8,.6 .3,.9 .7,-.3 .1,-1.2 h 1.3 l .2,-2.9 -.1,-2.7 .9,.3 -.7,-2.1 1.4,-.8 .2,-2.4 2.3,-2.2 1,.1 .3,-1.4 -1.2,-1.4 -.1,-3.5 -.8,.9 .7,2.9 -.6,.1 -.6,-1.9 -.6,-.5 .3,-2.3 1.8,-.1 .3,.7 .3,-1.6 -1.6,-1.7 -.6,-1.6 -.2,2 .9,1.1 -.7,.4 -1,-.8 -1.8,1.3 1.5,.5 .2,2.4 -.3,1.8 .9,-1.3 1.4,2.3 -.4,1.9 h -1.5 v -1.2 l -1.5,-1.2 .5,-3 -1.9,-2.6 2.7,-3 .6,-4.1 h .9 l 1.4,3.2 v -2.6 l 1.2,.3 v -3.3 l -.9,-.8 -1.2,2.5 -1,-3 1.3,-.1 -1.5,-4.9 1.9,-.6 25.4,7.5 31.7,8 23.6,5.5 z m -78.7,-39.4 h .5 l .1,.8 -.5,.3 .1,.6 -.7,.4 -.2,-.9 .5,-.4 z m 5,-4.3 -1.2,1.9 -.1,.8 .4,.2 .5,-.6 1.1,.1 z m -.4,-21.6 .5,.6 1.3,-.3 .2,-1 1.2,-1.8 -1,-.4 -.7,1.6 -.1,-1.6 -1.1,.2 -.7,1.4 z m 3.2,-5.5 .7,1.5 -.9,.2 -.8,.4 -.2,-2.4 z m -2.7,-1.6 -1.1,-.2 .5,1.4 z m -1,2.5 .8,.4 -.4,1.1 1.7,-.5 -.2,-2.2 -.9,-.2 z m -2.7,-.4 .3,2.7 1.6,1.3 .6,-1.9 -1.1,-2.2 z m 1.9,-1.1 -1.1,-1 -.9,.1 1.8,1.5 z m 3.2,-7 h -1.2 v .8 l 1.2,.6 z m -.9,32.5 .4,-2.7 h -1.1 l -.2,1.9 z"/><path class="gm-usmap-state" d="m 723.4,297.5 -.8,1.2 1.5,.7 .1,1.5 4.4,2.3 2.3,-.1 1.9,-1.8 .8,-1.7 3,1.8 5.5,-2.4 .5,-.9 -.8,-.5 .6,-1.4 1.5,1 4.3,-3.1 .7,1.1 2.3,-2 -.1,-1.4 1.5,-1.9 -1.5,-1.2 1,-3.3 3.7,-6.3 -.4,-1.9 2.1,-2.2 -.4,-1.5 1.4,-1.7 .1,-4.7 2.3,.7 1.3,1.9 2.8,.5 1.3,-1.6 2.3,-8.5 2.4,1.1 1,-2.5 .9,-.8 1.4,-1.8 .9,-.8 .5,-2.1 1.2,-.8 -.1,-2.9 .8,-2.3 -.9,-1.6 .2,-.9 10,5.2 .5,-2.3 .4,-1.6 .4,-.7 -.9,-.4 .2,-1.6 -1,-.5 -.2,-.7 h -.7 l -.8,-1.2 .2,-1 -2.6,.4 -2.2,-1.6 -1.4,.3 -.9,1.4 h -1.3 l -1.7,2.9 -3.3,.4 -1.9,-1 -2.6,3.8 -2.2,-.3 -3.1,3.9 -.9,1.6 -1.8,1.6 -1.7,-11.4 -17.4,2.9 -3.2,-19.7 -2.2,1.2 1.4,2.1 -.1,2.2 .6,2 -1.1,3.4 -.1,5.4 -1,3.6 .5,1.1 -.4,2.2 -1.1,.5 -2,3.3 -1.8,2 h -.6 l -1.8,1.7 -1.3,-1.2 -1.5,1.8 -.3,1.2 h -1.3 l -1.3,2.2 .1,2.1 -1,.5 1.4,1.1 v 1.9 l -1,.2 -.7,.8 -1,.5 -.6,-2.1 -1.6,-.5 -1,2.3 -.3,2.2 -1.1,1.3 1.3,3.6 -1.5,.8 -.4,3.5 h -1.5 l -3.2,1.4 -.1,1.1 .6,1 -.6,3.6 1.9,1.6 .8,1.1 1,.6 -.1,.9 4.4,5.6 h 1.4 l 1.5,1.8 1.2,.3 1.4,-.1 z"/><path class="gm-usmap-state" d="m 611,144 -2.9,.8 .2,2.3 -2.4,3.4 -.2,3.1 .6,.7 .8,-.7 .5,-1.6 2,-1.1 1.6,-4.2 3.5,-1.1 .8,-3.3 .7,-.9 .4,-2.1 1.8,-1.1 v -1.5 l 1,-.9 1.4,.1 v 2 l -1,.1 .5,1.2 -.7,2.2 -.6,.1 -1.2,4.5 -.7,.5 -2.8,7.2 -.3,4.2 .6,2 .1,1.3 -2.4,1.9 .3,1.9 -.9,3.1 .3,1.6 .4,3.7 -1.1,4.1 -1.5,5 1,1.5 -.3,.3 .8,1.7 -.5,1.1 1.1,.9 v 2.7 l 1.3,1.5 -.4,3 .3,4 -45.9,2.8 -1.3,-2.8 -3.3,-.7 -2.7,-1.5 -2,-5.5 .1,-2.5 1.6,-3.3 -.6,-1.1 -2.1,-1.6 -.2,-2.6 -1.1,-4.5 -.2,-3 -2.2,-3 -2.8,-.7 -5.2,-3.6 -.6,-3.3 -6.3,-3.1 -.2,-1.3 h -3.3 l -2.2,-2.6 -2,-1.3 .7,-5.1 -.9,-1.6 .5,-5.4 1,-1.8 -.3,-2.7 -1.2,-1.3 -1.8,-.3 v -1.7 l 2.8,-5.8 5.9,-3.9 -.4,-13 .9,.4 .6,-.5 .1,-1.1 .9,-.6 1.4,1.2 .7,-.1 h 2.6 l 6.8,-2.6 .3,-1 h 1.2 l .7,-1.2 .4,.8 1.8,-.9 1.8,-1.7 .3,.5 1,-1 2.2,1.6 -.8,1.6 -1.2,1.4 .5,1.5 -1.4,1.6 .4,.9 2.3,-1.1 v -1.4 l 3.3,1.9 1.9,.7 1.9,.7 3,3.8 17,3.8 1.4,1 4,.8 .7,.5 2.8,-.2 4.9,.8 1.4,1.5 -1,1 .8,.8 3.8,.7 1.2,1.2 .1,4.4 -1.3,2.8 2,.1 1,-.8 .9,.8 -1.1,3.1 1,1.6 1.2,.3 z m -49.5,-37.3 -.5,.1 -1.5,1.6 .2,.5 1.5,-.6 v -.6 l .9,-.3 z m 1.6,-1.1 -1,.3 -.2,.7 .9,-.1 z m -1.3,-1.6 -.2,.9 h 1.7 l .6,-.4 .1,-1 z m 2.8,-3 -.3,1.9 1.2,-.5 .1,-1.4 z m 58.3,31.9 -2,.3 -.4,1.3 1.3,1.7 z"/><path class="gm-usmap-state" d="m 355.3,143.7 -51,-5.3 -57.3,-7.9 -2,10.7 -8.5,54.8 -3.3,21.9 32.1,4.8 44.9,5.7 37.5,3.4 3.7,-44.2 z"/>"""

# 구단 핀 좌표(viewBox 기준) — 기존 지도에서 그대로 가져왔다
_TEAM_PINS = [
    {
        "code": "NYY",
        "x": 823.0,
        "y": 188.0,
        "color": "#0C2340",
        "short": "NY",
        "name": "New York Yankees"
    },
    {
        "code": "NYM",
        "x": 838.0,
        "y": 202.0,
        "color": "#002D72",
        "short": "NY",
        "name": "New York Mets"
    },
    {
        "code": "BOS",
        "x": 868.0,
        "y": 155.0,
        "color": "#BD3039",
        "short": "BO",
        "name": "Boston Red Sox"
    },
    {
        "code": "TOR",
        "x": 778.0,
        "y": 92.0,
        "color": "#134A8E",
        "short": "TO",
        "name": "Toronto Blue Jays"
    },
    {
        "code": "TBR",
        "x": 690.0,
        "y": 470.0,
        "color": "#092C5C",
        "short": "TB",
        "name": "Tampa Bay Rays"
    },
    {
        "code": "BAL",
        "x": 800.0,
        "y": 248.0,
        "color": "#DF4601",
        "short": "BA",
        "name": "Baltimore Orioles"
    },
    {
        "code": "CLE",
        "x": 700.0,
        "y": 215.0,
        "color": "#00385D",
        "short": "CL",
        "name": "Cleveland Guardians"
    },
    {
        "code": "MIN",
        "x": 515.0,
        "y": 120.0,
        "color": "#002B5C",
        "short": "MI",
        "name": "Minnesota Twins"
    },
    {
        "code": "CHW",
        "x": 608.0,
        "y": 224.0,
        "color": "#27251F",
        "short": "CH",
        "name": "Chicago White Sox"
    },
    {
        "code": "DET",
        "x": 655.0,
        "y": 172.0,
        "color": "#0C2340",
        "short": "DE",
        "name": "Detroit Tigers"
    },
    {
        "code": "KCR",
        "x": 498.0,
        "y": 282.0,
        "color": "#004687",
        "short": "KC",
        "name": "Kansas City Royals"
    },
    {
        "code": "HOU",
        "x": 455.0,
        "y": 500.0,
        "color": "#002D62",
        "short": "HO",
        "name": "Houston Astros"
    },
    {
        "code": "SEA",
        "x": 112.0,
        "y": 65.0,
        "color": "#0C2C56",
        "short": "SE",
        "name": "Seattle Mariners"
    },
    {
        "code": "TEX",
        "x": 410.0,
        "y": 400.0,
        "color": "#003278",
        "short": "TE",
        "name": "Texas Rangers"
    },
    {
        "code": "LAA",
        "x": 108.0,
        "y": 332.0,
        "color": "#BA0021",
        "short": "LA",
        "name": "Los Angeles Angels"
    },
    {
        "code": "ATH",
        "x": 88.0,
        "y": 218.0,
        "color": "#003831",
        "short": "AT",
        "name": "Athletics"
    },
    {
        "code": "ATL",
        "x": 710.0,
        "y": 396.0,
        "color": "#13274F",
        "short": "AT",
        "name": "Atlanta Braves"
    },
    {
        "code": "PHI",
        "x": 812.0,
        "y": 228.0,
        "color": "#E81828",
        "short": "PH",
        "name": "Philadelphia Phillies"
    },
    {
        "code": "MIA",
        "x": 756.0,
        "y": 536.0,
        "color": "#00A3E0",
        "short": "MI",
        "name": "Miami Marlins"
    },
    {
        "code": "WSN",
        "x": 802.0,
        "y": 252.0,
        "color": "#AB0003",
        "short": "WS",
        "name": "Washington Nationals"
    },
    {
        "code": "MIL",
        "x": 598.0,
        "y": 170.0,
        "color": "#12284B",
        "short": "MI",
        "name": "Milwaukee Brewers"
    },
    {
        "code": "CHC",
        "x": 590.0,
        "y": 212.0,
        "color": "#0E3386",
        "short": "CH",
        "name": "Chicago Cubs"
    },
    {
        "code": "STL",
        "x": 598.0,
        "y": 300.0,
        "color": "#C41E3A",
        "short": "ST",
        "name": "St. Louis Cardinals"
    },
    {
        "code": "CIN",
        "x": 672.0,
        "y": 275.0,
        "color": "#C6011F",
        "short": "CI",
        "name": "Cincinnati Reds"
    },
    {
        "code": "PIT",
        "x": 748.0,
        "y": 222.0,
        "color": "#27251F",
        "short": "PI",
        "name": "Pittsburgh Pirates"
    },
    {
        "code": "LAD",
        "x": 100.0,
        "y": 340.0,
        "color": "#005A9C",
        "short": "LA",
        "name": "Los Angeles Dodgers"
    },
    {
        "code": "SDP",
        "x": 68.0,
        "y": 338.0,
        "color": "#2F241D",
        "short": "SD",
        "name": "San Diego Padres"
    },
    {
        "code": "SFG",
        "x": 58.0,
        "y": 258.0,
        "color": "#FD5A1E",
        "short": "SF",
        "name": "San Francisco Giants"
    },
    {
        "code": "ARI",
        "x": 190.0,
        "y": 350.0,
        "color": "#A71930",
        "short": "AR",
        "name": "Arizona Diamondbacks"
    },
    {
        "code": "COL",
        "x": 300.0,
        "y": 258.0,
        "color": "#33006F",
        "short": "CO",
        "name": "Colorado Rockies"
    }
]



def us_map_html() -> str:
    """구단 선택 지도. 마커 클릭 시 ?pick=CODE 로 이동한다
    (Home.py 가 st.query_params 로 읽어서 팀을 선택).

    마커는 "밤 경기장의 조명탑"을 형상화했다 — 지면에 팀 컬러 글로우가 깔리고,
    거기서 빛기둥이 솟아오르며, 꼭대기에 팀 컬러 오브가 떠 있다. 전체 테마
    (야간 스타디움)와 같은 언어를 쓰는 것이 목적이다.
    핀 하나당 노드가 많아 마크업을 문자열로 하드코딩하지 않고 여기서 생성한다.
    """
    markers = []
    for i, p in enumerate(_TEAM_PINS):
        x, y, code, color = p["x"], p["y"], p["code"], p["color"]
        # 팀 색조는 유지하면서 밝기만 끌어올린다 (진한 빨강 구단이 남색으로
        # 바뀌어버리는 걸 막기 위해 team_glow_color 대신 boost 를 쓴다)
        glow = boost_for_dark(color)
        logo = team_logo_url(code)
        name = p["name"]
        plate_w = max(66.0, _approx_text_width(name, 6.4) + 20)

        markers.append(
            f'<a href="?pick={code}" target="_top">'
            f'<g class="gm-pin" transform="translate({x},{y})" style="--i:{i};--pin:{glow}">'
            # 지면 글로우 — 조명이 땅을 비추는 원
            f'<ellipse class="gm-pin-ground" cx="0" cy="0" rx="13" ry="4.6" fill="{glow}"/>'
            # 퍼져나가는 레이더 링
            f'<ellipse class="gm-pin-radar" cx="0" cy="0" rx="9" ry="3.2" fill="none" '
            f'stroke="{glow}" stroke-width="1.4"/>'
            # 빛기둥 (아래는 넓고 위로 갈수록 좁아지는 사다리꼴)
            f'<path class="gm-pin-beam" d="M-6,0 L-2.4,-29 L2.4,-29 L6,0 Z" fill="{glow}"/>'
            # 조명 꼭대기에 구단 로고 — "무슨 팀인지" 지도에서 바로 읽히게.
            # MLB 공식 로고 SVG 를 핫링크하고, 로드 실패 시 onerror 로 숨겨
            # 아래 팀컬러 오브만 남는다(가짜 로고를 그리지 않는다).
            '<g class="gm-pin-orb">'
            # 바깥 글로우(팀 컬러) → 밝은 원판 → 로고 순.
            # MLB 공식 로고는 밝은 배경 전제로 만들어져서 어두운 원판 위에 얹으면
            # 검은 윤곽이 뭉개진다(실측) — 원판을 밝게 깔고 팀 컬러는 테두리로만 쓴다.
            f'<circle cx="0" cy="-33" r="16" fill="{glow}" opacity=".28"/>'
            f'<circle cx="0" cy="-33" r="12.6" fill="rgba(248,251,255,.97)" '
            f'stroke="{glow}" stroke-width="2.2"/>'
            + (
                f'<image href="{logo}" x="-9.6" y="-42.6" width="19.2" height="19.2" '
                'preserveAspectRatio="xMidYMid meet" '
                "onerror=\"this.style.display='none'\"/>"
                if logo else
                f'<circle cx="0" cy="-33" r="4" fill="{glow}"/>'
            )
            + "</g>"
            # 호버 시 뜨는 이름표
            f'<g class="gm-pin-plate" transform="translate(0,-51)">'
            f'<rect x="{-plate_w / 2:.1f}" y="-10" width="{plate_w:.1f}" height="19" rx="7" '
            f'fill="rgba(6,11,23,.95)" stroke="{glow}" stroke-width="1"/>'
            f'<text x="0" y="3.5" text-anchor="middle" font-size="10" font-weight="700" '
            f'fill="#fff">{name}</text></g>'
            f"<title>{name}</title>"
            "</g></a>"
        )

    return (
        '<div class="gm-usmap-wrap"><svg viewBox="-10 -35 979 638" '
        'xmlns="http://www.w3.org/2000/svg" role="img" aria-label="MLB 구단 위치 지도">'
        "<defs>"
        # 지도 판 자체의 은은한 그라디언트
        '<linearGradient id="gm-map-plate" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0%" stop-color="rgba(255,255,255,.10)"/>'
        '<stop offset="100%" stop-color="rgba(255,255,255,.03)"/></linearGradient>'
        "</defs>"
        + _US_STATE_PATHS
        # 지도를 훑고 지나가는 레이더 스캔 라인
        + '<g class="gm-map-scan"><rect x="-10" y="-35" width="150" height="638" '
        'fill="url(#gm-scan-grad)"/></g>'
        '<defs><linearGradient id="gm-scan-grad" x1="0" y1="0" x2="1" y2="0">'
        '<stop offset="0%" stop-color="rgba(120,180,255,0)"/>'
        '<stop offset="55%" stop-color="rgba(120,180,255,.16)"/>'
        '<stop offset="100%" stop-color="rgba(120,180,255,0)"/></linearGradient></defs>'
        + "".join(markers)
        + "</svg></div>"
    )


# ══════════════════════════════════════════════════════════════════════
# 다이아몬드 라인업 — 로스터를 실제 야구장 수비 위치에 배치해서 보여준다
#
# 첨부 레퍼런스(모바일 야구 게임 LINE-UP 화면)의 아이디어를 이 서비스의
# 실제 데이터로 재구성한 것. 표로만 보던 로스터를 "우리 팀 수비 그림"으로
# 바꾸면 GM이 "어느 포지션이 비었는지 / 어디가 위험한지"를 한눈에 본다
# — 이탈위험이 높은 선수일수록 붉게 타오르는 링이 돈다.
#
# 좌표는 viewBox(0 0 640 520) 기준 수비 위치 실제 배치를 따른다.
# ══════════════════════════════════════════════════════════════════════

# 실제 수비 위치 좌표 (viewBox 0 0 640 560). 홈플레이트 (320,424),
# 2루 (320,212), 1루 (426,318), 3루 (214,318), 마운드 (320,318) 기준으로
# 각 야수가 실제로 서는 자리에 배치한다.
# 외야는 LF/CF/RF 세 자리다 — features_v1 의 primary_position 은 좌/중/우를
# 구분하지 않고 전부 "OF" 라서, 예전에는 외야수 8명 중 1명만 그려지고
# 나머지가 통째로 사라졌다(실측). 이제 OF 상위 3명을 세 자리에 나눠 세운다.
_DIAMOND_SLOTS: dict[str, tuple[float, float, str]] = {
    # 이름표(원 아래 30px 높이 pill)까지 고려해 자리를 벌려놓았다 — 좁히면
    # 유격수 이름표가 3루수 포지션 라벨을 덮는다(실측으로 조정한 값).
    "LF": (146, 206, "좌익수"),
    "CF": (320, 138, "중견수"),
    "RF": (494, 206, "우익수"),
    "SS": (238, 232, "유격수"),
    "2B": (402, 232, "2루수"),
    "3B": (148, 322, "3루수"),
    "1B": (492, 322, "1루수"),
    "P":  (320, 310, "투수"),
    "C":  (320, 458, "포수"),
    "DH": (556, 436, "지명타자"),
}

# 데이터의 primary_position -> 그릴 슬롯 목록(우선순위 순).
# OF 한 종류가 세 자리로 펼쳐지는 것이 핵심.
_POSITION_SLOTS: dict[str, list[str]] = {
    "OF": ["CF", "LF", "RF"],
    "SS": ["SS"], "2B": ["2B"], "3B": ["3B"], "1B": ["1B"],
    "P": ["P"], "C": ["C"], "DH": ["DH"],
}


def _approx_text_width(text: str, font_size: float) -> float:
    """SVG 텍스트의 대략적인 픽셀 폭. 라벨 뒤에 깔 배경 pill 크기를 잡는 용도.

    한글은 폭이 거의 글자 크기와 같고(정사각에 가까움) 라틴 문자는 그 절반쯤이라
    두 가지로만 나눠 근사한다 — 정확한 측정은 브라우저에서만 가능하므로,
    조금 넉넉하게 잡아 글자가 pill 밖으로 삐져나오지 않게 한다.
    """
    units = sum(1.0 if ord(ch) > 0x2E80 else 0.55 for ch in text)
    return units * font_size


def _risk_tone(risk: float) -> str:
    """이탈위험 → 색. 임계값은 UI 전반(선수 리포트 카드)과 같은 기준."""
    if risk >= 0.70:
        return "var(--risk)"
    if risk >= 0.45:
        return "var(--warn)"
    return "var(--gain)"


def diamond_lineup_svg(players: list[dict]) -> str:
    """포지션별 주전을 그라운드 위에 얼굴로 배치한 SVG.

    players: [{"position","name","ovr","risk","photo"}] — 전력 내림차순 정렬된
             전체 후보(포지션 중복 허용). 이 함수가 자리별로 배분한다.

    데이터가 없는 자리는 점선 빈 슬롯으로 남겨 "공백"을 그대로 보여준다
    — 가짜로 채우지 않는다.
    """
    # 포지션별로 전력 순서대로 담아두고, 슬롯 우선순위대로 한 명씩 꺼내 세운다
    pool: dict[str, list[dict]] = {}
    for p in players:
        pool.setdefault(str(p.get("position")), []).append(p)

    assigned: dict[str, dict] = {}
    for pos, slots in _POSITION_SLOTS.items():
        queue = pool.get(pos, [])
        for slot, person in zip(slots, queue):
            assigned[slot] = person

    nodes: list[str] = []
    for idx, (slot, (cx, cy, label)) in enumerate(_DIAMOND_SLOTS.items()):
        person = assigned.get(slot)
        delay = 0.35 + idx * 0.06

        if person is None:
            nodes.append(
                f'<g class="gm-dia-node gm-dia-empty" style="--d:{delay}s">'
                f'<circle cx="{cx}" cy="{cy}" r="25" fill="rgba(255,255,255,.04)" '
                'stroke="rgba(255,255,255,.22)" stroke-width="1.5" stroke-dasharray="4 4"/>'
                f'<text x="{cx}" y="{cy + 4}" text-anchor="middle" font-size="11" '
                f'font-weight="800" fill="rgba(255,255,255,.45)">{slot}</text>'
                f'<text x="{cx}" y="{cy + 44}" text-anchor="middle" font-size="10.5" '
                f'fill="rgba(255,255,255,.35)">{label} 공백</text>'
                "</g>"
            )
            continue

        risk = float(person.get("risk") or 0.0)
        ovr = float(person.get("ovr") or 0.0)
        name = str(person.get("name") or "")
        photo = person.get("photo")
        tone = _risk_tone(risk)
        short = name if len(name) <= 12 else name[:11] + "…"
        pulse = "gm-dia-pulse-hot" if risk >= 0.70 else ("gm-dia-pulse-warm" if risk >= 0.45 else "")

        risk_text = f"이탈 {risk * 100:.0f}%"
        pill_w = max(_approx_text_width(short, 11.0), _approx_text_width(risk_text, 9.5)) + 16
        pill_x = cx - pill_w / 2
        pill_y = cy + 32
        clip_id = f"gm-dia-clip-{slot}"

        # 얼굴이 있으면 원 안을 사진으로 채우고 OVR 은 우하단 배지로 뺀다.
        # 없으면 예전처럼 원 가운데에 OVR 숫자를 크게 둔다.
        if photo:
            face = (
                f'<clipPath id="{clip_id}"><circle cx="{cx}" cy="{cy}" r="24"/></clipPath>'
                f'<circle cx="{cx}" cy="{cy}" r="24" fill="#0B1424"/>'
                f'<image href="{photo}" x="{cx - 24}" y="{cy - 24}" width="48" height="48" '
                f'clip-path="url(#{clip_id})" preserveAspectRatio="xMidYMid slice" '
                "onerror=\"this.style.display='none'\"/>"
                f'<circle cx="{cx}" cy="{cy}" r="24" fill="none" stroke="{tone}" stroke-width="3"/>'
                f'<circle cx="{cx + 18}" cy="{cy + 17}" r="11.5" fill="rgba(6,11,23,.95)" '
                f'stroke="{tone}" stroke-width="1.6"/>'
                f'<text x="{cx + 18}" y="{cy + 21}" text-anchor="middle" font-size="11" '
                f'font-weight="800" fill="#fff">{ovr:.0f}</text>'
            )
        else:
            face = (
                f'<circle cx="{cx}" cy="{cy}" r="24" fill="rgba(8,14,28,.88)" '
                f'stroke="{tone}" stroke-width="3"/>'
                f'<text x="{cx}" y="{cy + 6}" text-anchor="middle" font-size="17" '
                f'font-weight="800" fill="#fff">{ovr:.0f}</text>'
            )

        nodes.append(
            f'<g class="gm-dia-node {pulse}" style="--d:{delay}s;--tone:{tone}">'
            f'<circle class="gm-dia-halo" cx="{cx}" cy="{cy}" r="29" fill="none" '
            f'stroke="{tone}" stroke-width="2" opacity=".55"/>'
            # 포지션 라벨은 원 위쪽에 — 얼굴을 가리지 않는다
            f'<text x="{cx}" y="{cy - 30}" text-anchor="middle" font-size="9.5" '
            f'font-weight="800" fill="rgba(255,255,255,.72)" letter-spacing=".6">{slot}</text>'
            + face
            + f'<rect x="{pill_x:.1f}" y="{pill_y:.1f}" width="{pill_w:.1f}" height="30" '
            'rx="9" fill="rgba(6,11,23,.86)" stroke="rgba(255,255,255,.14)" stroke-width="1"/>'
            f'<text x="{cx}" y="{pill_y + 13:.1f}" text-anchor="middle" font-size="11" '
            f'font-weight="700" fill="#fff">{short}</text>'
            f'<text x="{cx}" y="{pill_y + 25:.1f}" text-anchor="middle" font-size="9.5" '
            f'font-weight="800" fill="{tone}">{risk_text}</text>'
            "</g>"
        )

    return (
        '<div class="gm-diamond-wrap">'
        '<svg class="gm-diamond" viewBox="0 0 640 560" xmlns="http://www.w3.org/2000/svg" '
        'role="img" aria-label="포지션별 로스터 배치도">'
        "<defs>"
        '<radialGradient id="gm-turf" cx="50%" cy="76%" r="78%">'
        '<stop offset="0%" stop-color="#1F8A55"/><stop offset="58%" stop-color="#14603A"/>'
        '<stop offset="100%" stop-color="#0B3B24"/></radialGradient>'
        '<linearGradient id="gm-dirt" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0%" stop-color="#C08A55"/><stop offset="100%" stop-color="#9A6B3E"/></linearGradient>'
        "</defs>"
        '<path d="M320 470 L36 246 A360 360 0 0 1 604 246 Z" fill="url(#gm-turf)"/>'
        '<path d="M320 470 L150 336 A250 250 0 0 1 236 268 Z" fill="rgba(255,255,255,.045)"/>'
        '<path d="M320 470 L320 232 A250 250 0 0 1 404 268 Z" fill="rgba(255,255,255,.045)"/>'
        '<path d="M320 470 L490 336 A250 250 0 0 0 404 268 Z" fill="rgba(0,0,0,.05)"/>'
        '<path d="M320 470 L168 318 A215 215 0 0 1 472 318 Z" fill="url(#gm-dirt)" opacity=".92"/>'
        '<path d="M320 424 L214 318 L320 212 L426 318 Z" fill="url(#gm-turf)"/>'
        '<path d="M320 424 L214 318 L320 212 L426 318 Z" fill="none" '
        'stroke="rgba(255,255,255,.85)" stroke-width="2.5"/>'
        '<rect x="313" y="417" width="14" height="14" fill="#fff" transform="rotate(45 320 424)"/>'
        '<rect x="207" y="311" width="14" height="14" fill="#fff" transform="rotate(45 214 318)"/>'
        '<rect x="313" y="205" width="14" height="14" fill="#fff" transform="rotate(45 320 212)"/>'
        '<rect x="419" y="311" width="14" height="14" fill="#fff" transform="rotate(45 426 318)"/>'
        '<circle cx="320" cy="318" r="26" fill="url(#gm-dirt)"/>'
        '<rect x="315" y="314" width="10" height="5" rx="1" fill="#fff" opacity=".9"/>'
        '<path d="M320 470 L60 210" stroke="rgba(255,255,255,.5)" stroke-width="2"/>'
        '<path d="M320 470 L580 210" stroke="rgba(255,255,255,.5)" stroke-width="2"/>'
        + "".join(nodes)
        + "</svg></div>"
    )


def stat_ring_svg(value: float, label: str, *, max_value: float = 100.0,
                  size: int = 96, tone: str = "var(--team-accent)") -> str:
    """원형 게이지 — 값이 0에서부터 그려지며 차오른다(stroke-dashoffset)."""
    r = size / 2 - 8
    circ = 2 * math.pi * r
    pct = max(0.0, min(1.0, value / max_value if max_value else 0.0))
    offset = circ * (1 - pct)
    return (
        f'<div class="gm-ring" style="--tone:{tone}">'
        f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}">'
        f'<circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" '
        'stroke="rgba(255,255,255,.10)" stroke-width="7"/>'
        f'<circle class="gm-ring-fill" cx="{size/2}" cy="{size/2}" r="{r}" fill="none" '
        f'stroke="{tone}" stroke-width="7" stroke-linecap="round" '
        f'stroke-dasharray="{circ:.1f}" style="--circ:{circ:.1f};--off:{offset:.1f}" '
        f'transform="rotate(-90 {size/2} {size/2})"/>'
        "</svg>"
        f'<div class="gm-ring-mid"><span class="gm-ring-v">{value:.0f}</span>'
        f'<span class="gm-ring-l">{label}</span></div>'
        "</div>"
    )


# ══════════════════════════════════════════════════════════════════════
# 승부예측 챌린지 — 사용자가 먼저 고르고 그 다음 AI 예측이 열린다
#
# 네이버 스포츠 승부예측의 재미 구조를 그대로 가져왔다: 확률을 미리 보여주면
# 그걸 따라 찍게 되므로, "내 선택 → 공개 → 비교" 순서를 반드시 지킨다.
# 정답(실제 경기 결과)이 아직 없는 미래 경기이므로 "맞았다/틀렸다"가 아니라
# "AI와 같게 봤다/다르게 봤다"로 정직하게 표현한다.
# ══════════════════════════════════════════════════════════════════════

def confidence_label(prob: float) -> tuple[str, str]:
    """승리확률 → (표현, 이모지). 50%에 가까울수록 박빙."""
    edge = abs(prob - 0.5)
    if edge >= 0.12:
        return "압도적 우세", "flame"
    if edge >= 0.06:
        return "우세", "trend-up"
    if edge >= 0.025:
        return "근소 우세", "pinch"
    return "초박빙", "scale"


def prediction_reveal_html(
    *,
    away_name: str,
    home_name: str,
    away_pct: float,
    home_pct: float,
    ai_winner_name: str,
    user_pick_name: str,
    model_votes: list[tuple[str, bool]],
    away_color: str = "#5B7FB9",
    home_color: str = "#D66E6E",
) -> str:
    """선택 후 공개되는 결과 카드.

    model_votes: [(모델명, AI최종예측과 같은 쪽에 투표했는가)] — 4개 모델이
    얼마나 한목소리인지 보여준다. 3:1로 갈렸으면 그만큼 불확실하다는 뜻.
    """
    agreed = user_pick_name == ai_winner_name
    top_pct = max(away_pct, home_pct)
    conf_text, conf_icon = confidence_label(top_pct / 100)
    agree_n = sum(1 for _, ok in model_votes if ok)

    chips = "".join(
        f'<span class="gm-pred-chip{" agree" if ok else ""}" style="--j:{j}">{name}</span>'
        for j, (name, ok) in enumerate(model_votes)
    )

    if agreed:
        verdict = (
            '<div class="gm-pred-verdict hit">'
            '<span class="gm-pred-verdict-icon">' + icon("target", 17) + "</span>"
            f'<span>내 선택 <span class="gm-pred-mine">{user_pick_name}</span> — '
            f'AI도 같게 봤습니다</span></div>'
        )
    else:
        verdict = (
            '<div class="gm-pred-verdict miss">'
            '<span class="gm-pred-verdict-icon">' + icon("diverge", 17) + "</span>"
            f'<span>내 선택 <span class="gm-pred-mine">{user_pick_name}</span> · '
            f'AI 예측 <span class="gm-pred-mine">{ai_winner_name}</span> — 다르게 봤습니다</span></div>'
        )

    return (
        '<div class="gm-pred-reveal">'
        f'<div class="gm-pred-bar" style="--away-c:{away_color};--home-c:{home_color}">'
        f'<div class="gm-pred-bar-away" style="width:{away_pct:.1f}%">{away_pct:.0f}%</div>'
        f'<div class="gm-pred-bar-home" style="width:{home_pct:.1f}%">{home_pct:.0f}%</div>'
        '<div class="gm-pred-bar-mid"></div>'
        "</div>"
        + verdict
        + '<div class="gm-pred-models">'
        f'<span class="gm-pred-conf">{icon(conf_icon, 11)} {conf_text}</span>'
        f"<span>모델 {agree_n}/{len(model_votes)} 동의</span>"
        + chips
        + "</div></div>"
    )


def prediction_scoreboard_html(picked: int, total: int, agreed: int, streak: int) -> str:
    """상단 스코어보드 — 참여 현황과 AI 일치율."""
    rate = (agreed / picked * 100) if picked else 0.0
    if picked == 0:
        msg = "경기를 하나씩 골라보세요. 선택하는 순간 AI 예측이 열립니다."
    elif rate >= 80:
        msg = "AI와 거의 같은 눈으로 보고 계시네요. 단장 자질이 보입니다."
    elif rate >= 50:
        msg = "AI와 절반 이상 같은 판단입니다. 갈린 경기가 진짜 박빙인 경기예요."
    else:
        msg = "AI와 다르게 보신 경기가 많습니다 — 그만큼 이변 가능성을 읽고 계신 걸 수도."

    streak_html = (
        f'<span class="gm-pred-streak">{icon("flame", 11)} 연속 {streak}경기 일치</span>' if streak >= 2 else ""
    )
    return (
        '<div class="gm-pred-board">'
        f'<div class="gm-pred-board-stat"><span class="gm-pred-board-v">{picked}<span '
        f'style="font-size:14px;color:var(--muted)">/{total}</span></span>'
        '<span class="gm-pred-board-l">참여</span></div>'
        f'<div class="gm-pred-board-stat"><span class="gm-pred-board-v">{agreed}</span>'
        '<span class="gm-pred-board-l">AI와 일치</span></div>'
        f'<div class="gm-pred-board-stat"><span class="gm-pred-board-v">{rate:.0f}%</span>'
        '<span class="gm-pred-board-l">일치율</span></div>'
        f'<div class="gm-pred-board-msg">{msg}</div>'
        f"{streak_html}"
        "</div>"
    )


def team_glow_color(team_code: str | None, fallback: str = "#4E8FD6") -> str:
    """다크 배경에서 조명으로 쓸 팀 컬러.

    TEAM_COLORS 는 (primary, secondary) 쌍인데 팀에 따라 한쪽이 검정/남색이라
    (예: CIN 은 secondary 가 #000000) 그대로 쓰면 다크 테마에서 색이 아예
    안 보인다. 두 색 중 더 밝은 쪽을 골라 쓰고, 둘 다 너무 어두우면 기본
    액센트로 물러난다.
    """
    pair = TEAM_COLORS.get(team_code or "", ())
    best, best_lum = fallback, -1.0
    for hex_color in pair:
        try:
            r, g, b = _hex_to_rgb(hex_color)
        except (ValueError, TypeError):
            continue
        # 상대휘도(간이) — 사람 눈 가중치
        lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255
        if lum > best_lum:
            best, best_lum = hex_color, lum
    # 너무 어두우면(거의 검정) 조명 역할을 못 하므로 기본 액센트를 쓴다
    return fallback if best_lum < 0.14 else best


# ══════════════════════════════════════════════════════════════════════
# 모델 카드 — 태스크마다 지표 종류가 달라서(회귀/이진분류/다중분류) 표 한 장에
# 넣으면 빈 칸투성이가 된다. 각 모델의 "대표 지표" 하나를 골라 크게 보여주고,
# 나머지는 보조 줄로 내린다.
# ══════════════════════════════════════════════════════════════════════

# task -> (대표지표 컬럼, 표시라벨, 0~1 정규화 함수)
_PRIMARY_METRIC: dict[str, tuple[str, str, "callable"]] = {
    "strength":  ("r2",       "R²",  lambda v: max(0.0, min(1.0, v))),
    # AUC 는 0.5 가 무작위 — 0.5~1.0 구간을 0~1 로 펴야 막대가 실제 실력차를 보여준다
    "win_rate":  ("roc_auc",  "AUC", lambda v: max(0.0, min(1.0, (v - 0.5) * 2))),
    "departure": ("roc_auc",  "AUC", lambda v: max(0.0, min(1.0, (v - 0.5) * 2))),
    "reason":    ("macro_f1", "Macro F1", lambda v: max(0.0, min(1.0, v))),
    "recommend": ("precision_at_3", "P@3", lambda v: max(0.0, min(1.0, v * 5))),
}

_TASK_KO = {
    "strength": "전력 예측 (회귀)",
    "win_rate": "승부 예측 (이진분류)",
    "departure": "이탈 예측 (이진분류)",
    "reason": "이탈 원인 태그 (다중분류)",
    "recommend": "대체 선수 추천",
}


def model_card_html(
    *,
    name: str,
    kind: str,
    owner: str,
    task: str,
    metrics: dict,
    is_best: bool = False,
    index: int = 0,
) -> str:
    """모델 하나를 카드로. 대표 지표는 크게, 보조 지표는 작게."""
    spec = _PRIMARY_METRIC.get(task)
    primary_v, primary_label, gauge = None, "", 0.0
    if spec:
        col, primary_label, norm = spec
        raw = metrics.get(col)
        if raw is not None and not (isinstance(raw, float) and math.isnan(raw)):
            primary_v = float(raw)
            gauge = norm(primary_v)

    # 보조 지표 — 태스크와 무관하게 있는 것만 골라 붙인다
    subs = []
    for col, fmt in (
        ("mae", "MAE {:.2f}"), ("rmse", "RMSE {:.2f}"),
        ("f1", "F1 {:.3f}"), ("accuracy", "정확도 {:.3f}"),
        ("n_test", "n={:.0f}"),
    ):
        v = metrics.get(col)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            subs.append(fmt.format(float(v)))
    sub_text = " · ".join(subs[:3]) if subs else "지표 없음"

    value_html = (
        f'<span class="gm-mcard-v">{primary_v:.3f}</span>'
        f'<span class="gm-mcard-mlabel">{primary_label}</span>'
        if primary_v is not None
        else '<span class="gm-mcard-v" style="font-size:17px;color:var(--faint)">지표 없음</span>'
    )

    return (
        f'<div class="gm-mcard{" best" if is_best else ""}" style="--i:{index}">'
        + (f'<span class="gm-mcard-crown">{icon("crown", 15)}</span>' if is_best else "")
        + '<div class="gm-mcard-head">'
        f'{badge(owner, "navy")}'
        f'<span class="gm-mcard-kind">{kind.upper()}</span>'
        "</div>"
        f'<div class="gm-mcard-name" style="margin-bottom:7px">{name}</div>'
        f'<div class="gm-mcard-metric">{value_html}</div>'
        f'<div class="gm-mcard-sub">{sub_text}</div>'
        '<div class="gm-mcard-bar">'
        f'<span class="gm-mcard-bar-fill" style="width:{gauge * 100:.1f}%"></span>'
        "</div></div>"
    )


def model_task_section_html(task: str, cards_html: str) -> str:
    """태스크 헤더 + 그 태스크의 카드 그리드."""
    label = _TASK_KO.get(task, task)
    return f'<div class="gm-task-head">{label}</div><div class="gm-mcard-grid">{cards_html}</div>'


def roster_list_html(rows: list[dict]) -> str:
    """전력 로스터 리스트.

    rows: [{"name","role","ovr","g_ratio","risk","tag"}] — 이미 정렬된 순서대로.
    전력·이탈위험을 막대로 그려서 표를 훑기만 해도 분포가 보이게 한다.
    이탈위험 막대는 위험도에 따라 색이 바뀐다(초록→노랑→빨강).
    """
    head = (
        '<div class="gm-roster-head">'
        "<span>#</span><span>선수</span><span>역할</span>"
        "<span>전력</span><span>이탈위험</span><span>연관 요인</span>"
        "</div>"
    )
    items = []
    for i, r in enumerate(rows):
        ovr = float(r.get("ovr") or 0.0)
        risk = r.get("risk")
        has_risk = risk is not None and not (isinstance(risk, float) and math.isnan(risk))
        risk_v = float(risk) if has_risk else 0.0
        tone = _risk_tone(risk_v)
        risk_cell = (
            f'<span class="gm-roster-num" style="color:{tone}">{risk_v * 100:.0f}%</span>'
            f'<span class="gm-roster-track"><span class="gm-roster-fill" '
            f'style="width:{risk_v * 100:.1f}%;background:{tone}"></span></span>'
            if has_risk
            else '<span class="gm-roster-num" style="color:var(--faint)">—</span>'
            '<span class="gm-roster-track"></span>'
        )
        items.append(
            f'<div class="gm-roster-row" style="--i:{i}">'
            f'<span class="gm-roster-rank">{i + 1}</span>'
            + (
                '<span class="gm-roster-name">'
                + (
                    f'<img class="gm-roster-face" src="{r["photo"]}" alt="" loading="lazy" '
                    "onerror=\"this.style.visibility='hidden'\"/>"
                    if r.get("photo") else '<span class="gm-roster-face"></span>'
                )
                + f'<span>{r.get("name", "")}</span></span>'
            )
            +
            f'<span class="gm-roster-role">{r.get("role", "")}</span>'
            '<span class="gm-roster-metric">'
            f'<span class="gm-roster-num">{ovr:.1f}</span>'
            '<span class="gm-roster-track"><span class="gm-roster-fill" '
            f'style="width:{max(0.0, min(100.0, ovr)):.1f}%;'
            'background:linear-gradient(90deg,var(--team-accent),#9FC4F0)"></span></span>'
            "</span>"
            f'<span class="gm-roster-metric">{risk_cell}</span>'
            f'<span class="gm-roster-tag">{r.get("tag", "")}</span>'
            "</div>"
        )
    return f'<div class="gm-roster">{head}{"".join(items)}</div>'


def impact_panel_html(before: float, after: float, *, label_before: str,
                      label_after: str, unit: str = "%", decimals: int = 2,
                      higher_is_better: bool = True,
                      delta_words: tuple[str, str] | None = None,
                      delta_unit: str | None = None) -> str:
    """이탈 전/후 값을 좌우로 놓고 가운데에 변화량을 띄우는 임팩트 패널.

    delta_words: (나빠졌을 때, 좋아졌을 때) 표현. 순위처럼 "%p"가 말이 안 되는
        단위에 쓴다 — 없으면 기존처럼 부호와 함께 "%p"를 붙인다.
    """
    delta = after - before
    if abs(delta) < 10 ** (-decimals) / 2:
        cls, sign = "flat", "변화 없음"
    else:
        good = (delta > 0) if higher_is_better else (delta < 0)
        cls = "up" if good else "down"
        if delta_words is not None:
            # 순위: "+1위p" 처럼 %p 규칙을 그대로 붙이면 말이 안 된다.
            word = delta_words[1] if good else delta_words[0]
            sign = f"{abs(delta):.{decimals}f}{delta_unit or unit} {word}"
        else:
            sign = f"{delta:+.{decimals}f}{unit}p"
    return (
        '<div class="gm-impact">'
        f'<div class="gm-impact-side"><div class="gm-impact-l">{label_before}</div>'
        f'<div class="gm-impact-v">{before:.{decimals}f}{unit}</div></div>'
        '<div class="gm-impact-arrow">'
        '<span class="gm-impact-arrow-icon">→</span>'
        f'<span class="gm-impact-delta {cls}">{sign}</span></div>'
        f'<div class="gm-impact-side"><div class="gm-impact-l">{label_after}</div>'
        f'<div class="gm-impact-v">{after:.{decimals}f}{unit}</div></div>'
        "</div>"
    )


def boost_for_dark(hex_color: str, *, min_l: float = 0.58, min_s: float = 0.55) -> str:
    """어두운 배경에서 확실히 빛나도록 색의 밝기를 끌어올린다(색조는 유지).

    team_glow_color() 처럼 "두 팀 컬러 중 밝은 쪽"을 고르는 방식은 진한 빨강
    계열 구단(신시내티 #C6011F, 워싱턴 #AB0003)에서 실패한다 — 빨강은 상대휘도
    가중치가 낮아(0.2126) 눈에는 선명해도 계산상 어둡게 나오고, 결국 팀 색을
    버리고 남색을 고르는 일이 생긴다(LAA 에서 실측). 그래서 색을 갈아끼우지
    않고 HLS 로 바꿔 명도/채도만 최소치까지 올린다 — 빨강은 빨강으로 남는다.
    거의 무채색(검정·회색 유니폼)은 채도를 억지로 올리면 엉뚱한 색이 되므로
    명도만 올린다.
    """
    import colorsys

    try:
        r, g, b = _hex_to_rgb(hex_color)
    except (ValueError, TypeError):
        return hex_color
    h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    l = max(l, min_l)
    if s > 0.12:  # 유채색일 때만 채도 보정
        s = max(s, min_s)
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return "#{:02X}{:02X}{:02X}".format(round(r2 * 255), round(g2 * 255), round(b2 * 255))


# ══════════════════════════════════════════════════════════════════════
# SVG 아이콘 세트 — 이모지 대체
#
# 이모지는 OS/브라우저마다 모양·색·크기가 제각각이라(같은 화면에서 애플/윈도우가
# 전혀 다르게 보인다) 디자인 톤을 맞출 수 없고, currentColor 를 못 따라간다.
# 여기 아이콘은 전부 stroke="currentColor" 라서 놓인 자리의 색을 그대로 입는다.
# ══════════════════════════════════════════════════════════════════════

_ICON_PATHS: dict[str, str] = {
    # 과녁 — 예측 적중
    "target": ('<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3.4"/>'
               '<path d="M12 1.8v3M12 19.2v3M1.8 12h3M19.2 12h3"/>'),
    # 갈림길 — 예측 불일치
    "diverge": '<path d="M6 21V9m0 0L3 12M6 9l3 3M18 3v12m0 0l-3-3m3 3l3-3"/>',
    # 불꽃 — 압도적 우세 / 연속 기록
    "flame": ('<path d="M12 22c3.9 0 6.5-2.5 6.5-6 0-4.2-4-6-5-9.6C12.8 8.6 11 9.4 11 11.5'
              ' 11 9 9 7.6 9 5.4 6.8 7.5 5.5 9.7 5.5 12.4c0 3.7 2.6 9.6 6.5 9.6Z"/>'),
    # 상승 화살표 — 우세
    "trend-up": '<path d="M3 17l6-6 4 4 7-7"/><path d="M15 8h5v5"/>',
    # 저울 — 초박빙
    "scale": ('<path d="M12 4v16M5 20h14M3 9l3.5-5L10 9M14 9l3.5-5L21 9"/>'
              '<path d="M3 9a3.5 3.5 0 0 0 7 0M14 9a3.5 3.5 0 0 0 7 0"/>'),
    # 좁은 간격 — 근소 우세
    "pinch": '<path d="M9 5v14M15 5v14"/><path d="M11.4 12h1.2"/>',
    # 주먹(원정) — 원정팀 선택
    "away": ('<path d="M6 11V8.5a2 2 0 0 1 4 0V11m0-1.5V7a2 2 0 0 1 4 0v3m0-1a2 2 0 0 1 4 0v5.5'
             'a6.5 6.5 0 0 1-13 0V12a2 2 0 0 1 4 0"/>'),
    # 홈플레이트 — 홈팀 선택
    "home": '<path d="M4 4h16v9l-8 7-8-7Z"/>',
    # 왕관 — 1위
    "crown": '<path d="M3 8l4 4 5-7 5 7 4-4v10H3Z"/><path d="M3 20h18"/>',
    # 트로피
    "trophy": ('<path d="M7 4h10v5a5 5 0 0 1-10 0Z"/><path d="M7 6H4v2a3 3 0 0 0 3 3"/>'
               '<path d="M17 6h3v2a3 3 0 0 1-3 3"/><path d="M12 14v3M9 20h6"/>'),
    # 경고 삼각형 — 복합 요인
    "alert": '<path d="M12 3.5 2.5 20h19Z"/><path d="M12 10v4.5M12 17.4v.2"/>',
    # 반창고 — 부상 연관
    "bandage": ('<rect x="2.6" y="8.4" width="18.8" height="7.2" rx="3.6" '
                'transform="rotate(-45 12 12)"/><path d="M10 10.5v.2M13.9 14.4v.2M13.9 10.5v.2M10 14.4v.2"/>'),
    # 하락 화살표 — 성적 하락
    "trend-down": '<path d="M3 7l6 6 4-4 7 7"/><path d="M20 11v5h-5"/>',
    # 시계 — 베테랑 시기
    "clock": '<circle cx="12" cy="12" r="8.5"/><path d="M12 7v5.3l3.2 2"/>',
    # 물음표 — 판단 근거 부족
    "question": ('<circle cx="12" cy="12" r="8.5"/>'
                 '<path d="M9.6 9.4a2.5 2.5 0 1 1 3.3 2.4c-.6.2-.9.8-.9 1.5v.4"/>'
                 '<path d="M12 17.2v.2"/>'),
    # 사이렌 — 이탈위험
    "siren": ('<path d="M6 19v-6a6 6 0 0 1 12 0v6"/><path d="M3.5 19h17"/>'
              '<path d="M12 3.5V5M4.8 6.3 5.9 7.4M19.2 6.3 18.1 7.4"/>'),
    # 근육 — 전력
    "muscle": ('<path d="M5 8c2-1.5 4.5-1 5.6.8 1 1.6 1 3.2.4 4.6 2.5-1 5.3.4 6 2.6'
               '.7 2.2-.6 4-3 4H8c-2.2 0-4-1.8-4-4Z"/>'),
    # 새로고침
    "refresh": '<path d="M20 11a8 8 0 1 0-2.3 6"/><path d="M20 4v7h-7"/>',
    # 뒤로 — 구단 변경
    "swap": '<path d="M4 8h13l-3-3m3 3-3 3"/><path d="M20 16H7l3-3m-3 3 3 3"/>',
}


def icon(name: str, size: float = 14, *, stroke_width: float = 1.9,
         cls: str = "", style: str = "") -> str:
    """인라인 SVG 아이콘. currentColor 를 따르므로 놓인 자리의 색을 그대로 입는다."""
    path = _ICON_PATHS.get(name)
    if not path:
        return ""
    return (
        f'<svg class="gm-icon {cls}" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="currentColor" stroke-width="{stroke_width}" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'style="{style}" aria-hidden="true">{path}</svg>'
    )
