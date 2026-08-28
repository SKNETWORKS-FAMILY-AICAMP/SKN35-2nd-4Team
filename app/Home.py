"""진입 화면 — 리그 조회 + 구단 선택.

오늘 경기(game.py) · 예상 순위(team_season.win_rate)는 A가 실제로 만들어둔
`data/final/predictions/remaining_games_predictions.csv` / `team_season.csv`를
그대로 읽는다. 파일이 없으면 placeholder()로 무엇이 필요한지 명시한다 — 가짜
숫자를 채우지 않는다.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# streamlit run 은 app/ 을 sys.path[0] 으로 잡아 리포 루트의 src.* 가 안 잡힐 때가
# 있다 — 3_Model_Information.py 와 동일한 가드.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# team_season.csv/게임 예측 파일은 라만 원본 팀코드(NYA/LAN/CHN 등)를 쓴다 —
# recommend.py가 이미 만들어둔 매핑을 그대로 재사용해 UI 코드로 맞춘다.
from src.models.recommend import LAHMAN_TEAM_TO_UI  # noqa: E402
from ui.theme import (  # noqa: E402
    DIVISIONS,
    TEAM_COLORS,
    TEAM_NAMES,
    inject_css,
    init_state,
    page_header,
    placeholder,
    section,
    us_map_html,
)

st.set_page_config(page_title="구단 선택", page_icon="⚾", layout="wide", initial_sidebar_state="collapsed")
inject_css()
init_state()

# 지도 마커 클릭(?pick=CODE)을 여기서 소비한다 — 실제 <a href> 이동이라 새
# 쿼리파라미터가 붙은 채로 이 스크립트가 다시 실행된다.
_picked = st.query_params.get("pick")
if _picked and _picked in TEAM_NAMES:
    st.session_state.team_code = _picked
    st.query_params.clear()
    st.switch_page("pages/1_Club_Operations_Center.py")

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = ROOT / "data" / "final" / "predictions" / "remaining_games_predictions.csv"
TEAM_SEASON_PATH = ROOT / "data" / "final" / "team_season.csv"


# LAHMAN_TEAM_TO_UI(recommend.py)에 없는 코드 — 탬파베이가 데이터 소스에 따라
# 2글자(TB)/3글자(TBA) 라만 코드가 섞여 나온다(직접 확인함). 여기서만 보충한다.
_EXTRA_TEAM_CODE_FIX = {"TBA": "TBR"}


def _to_ui_code(code: str) -> str:
    return LAHMAN_TEAM_TO_UI.get(code) or _EXTRA_TEAM_CODE_FIX.get(code, code)


with st.container(key="hero"):
    page_header("GENERAL MANAGER MODE", "오늘의 리그를 확인하고 맡을 구단을 선택하세요")

    tab_today, tab_rank, tab_pick = st.tabs(["오늘 경기", "예상 순위", "구단 선택"])

    # ── 오늘 경기: VS 매치업 카드 ──
    with tab_today:
        if not PREDICTIONS_PATH.exists():
            placeholder(
                "오늘 경기 목록",
                "`src/adapters/mlb_api.py`(A) 로 MLB Stats API에서 경기 일정을 수집하고, "
                "`src/models/game.py`(A) 로 승부를 예측해야 표시할 수 있습니다.",
            )
        else:
            games = pd.read_csv(PREDICTIONS_PATH)
            today = pd.Timestamp.now().strftime("%Y-%m-%d")
            todays_games = games[games.game_date == today]
            if todays_games.empty:
                next_date = games.game_date.min()
                todays_games = games[games.game_date == next_date]
                st.caption(f"오늘 예정된 경기가 없어 다음 경기일({next_date})을 표시합니다.")

            rows_html = []
            for i, row in enumerate(todays_games.itertuples()):
                home_code = _to_ui_code(row.home_team)
                away_code = _to_ui_code(row.away_team)
                winner_code = _to_ui_code(row.predicted_winner)
                home_name = TEAM_NAMES.get(home_code, row.home_team)
                away_name = TEAM_NAMES.get(away_code, row.away_team)
                winner_name = TEAM_NAMES.get(winner_code, row.predicted_winner)
                home_pct = row.ensemble_home_win_proba * 100
                away_pct = 100 - home_pct
                accent = TEAM_COLORS.get(winner_code, ("#3E6FB0", "#3E6FB0"))[1]
                rows_html.append(
                    f'<div class="gm-vs-card" style="--i:{i};--team-accent:{accent}">'
                    '<span class="gm-live-badge"><span class="gm-live-dot"></span>AI 예측</span>'
                    '<div class="gm-vs-row">'
                    f'<div class="gm-vs-team away"><span class="gm-vs-tag">원정</span>'
                    f'<span class="gm-vs-name">{away_name}</span></div>'
                    '<div class="gm-vs-mid">'
                    f'<span class="gm-vs-bolt">VS</span>'
                    '<div class="gm-vs-bar">'
                    f'<div class="gm-vs-bar-away" style="width:{away_pct:.0f}%"></div>'
                    f'<div class="gm-vs-bar-home" style="width:{home_pct:.0f}%"></div>'
                    "</div>"
                    f'<span class="gm-vs-winner">🏆 {winner_name} 우세 {max(home_pct, away_pct):.0f}%</span>'
                    "</div>"
                    f'<div class="gm-vs-team home"><span class="gm-vs-tag">홈</span>'
                    f'<span class="gm-vs-name">{home_name}</span></div>'
                    "</div></div>"
                )
            st.markdown(f'<div class="gm-vs-grid">{"".join(rows_html)}</div>', unsafe_allow_html=True)
            st.caption("game.py(LogReg + RandomForest + PyTorch MLP 앙상블) 예측 — 실제 미래 결과가 아닌 모델 추정치입니다.")

    # ── 예상 순위: 리빌 인터랙션 ──
    with tab_rank:
        if not TEAM_SEASON_PATH.exists():
            placeholder(
                "예상 순위",
                "`src/models/win_rate.py`(A) 로 팀 승률을 예측하고 "
                "`team_season.pred_rank` 컬럼을 채워야 표시할 수 있습니다.",
            )
        else:
            revealed = st.session_state.get("standings_revealed", False)
            if not revealed:
                st.markdown(
                    '<div style="text-align:center;padding:26px 0 10px">'
                    '<div style="font-size:15px;color:rgba(255,255,255,.75);margin-bottom:14px">'
                    "예측 모델이 계산한 이번 시즌 순위, 공개할까요?</div></div>",
                    unsafe_allow_html=True,
                )
                c1, c2, c3 = st.columns([1, 1, 1])
                with c2:
                    if st.button("🏆 순위 공개하기", key="reveal_standings", use_container_width=True):
                        st.session_state.standings_revealed = True
                        st.rerun()
            else:
                team_season = pd.read_csv(TEAM_SEASON_PATH)
                if team_season.pred_rank.notna().any():
                    latest_year = team_season.loc[team_season.pred_rank.notna(), "year"].max()
                    ranked = team_season[team_season.year == latest_year].sort_values("pred_rank")
                    rank_col = "pred_rank"
                else:
                    # pred_rank(시즌 시뮬레이션 결과)가 아직 안 채워져 있으면, 같은 파일의
                    # 실제 win_rate로 대신 순위를 매긴다 — 최신 완결 시즌 기준.
                    latest_year = team_season.year.max()
                    ranked = team_season[team_season.year == latest_year].sort_values("win_rate", ascending=False)
                    ranked = ranked.reset_index(drop=True)
                    ranked["pred_rank"] = ranked.index + 1
                    rank_col = "pred_rank"
                    st.caption(
                        f"team_season.pred_rank가 아직 없어 {latest_year}시즌 실제 승률로 대신 순위를 매깁니다."
                    )

                rows_html = []
                for i, row in enumerate(ranked.itertuples()):
                    code = _to_ui_code(row.team_id)
                    name = TEAM_NAMES.get(code, row.team_id)
                    rank = int(getattr(row, rank_col))
                    pct = row.win_rate * 100
                    accent = TEAM_COLORS.get(code, ("#16325C", "#3E6FB0"))[1]
                    rank_cls = f" gm-rank-{rank}" if rank <= 3 else ""
                    rows_html.append(
                        f'<div class="gm-standing-row{rank_cls}" style="--i:{i}">'
                        f'<div class="gm-standing-rank">{rank}</div>'
                        f'<div class="gm-standing-name">{name}</div>'
                        '<div class="gm-standing-bar-track">'
                        f'<div class="gm-standing-bar-fill" style="--i:{i};width:{pct:.1f}%;'
                        f'background:linear-gradient(90deg,var(--navy),{accent})"></div></div>'
                        f'<div class="gm-standing-pct">{pct:.1f}%</div>'
                        "</div>"
                    )
                st.markdown("".join(rows_html), unsafe_allow_html=True)
                if st.button("↻ 다시 감추기", key="hide_standings"):
                    st.session_state.standings_revealed = False
                    st.rerun()

    # ── 구단 선택: 미국 지도 + 목록 ──
    with tab_pick:
        st.markdown(
            '<div style="text-align:center;color:rgba(255,255,255,.6);font-size:13px;margin-bottom:6px">'
            "지도의 마커를 클릭하거나, 아래 목록에서 구단을 선택하세요</div>",
            unsafe_allow_html=True,
        )
        st.markdown(us_map_html(), unsafe_allow_html=True)

        with st.expander("목록에서 선택"):
            for division, teams in DIVISIONS.items():
                st.markdown(f'<div class="gm-division-label">{division}</div>', unsafe_allow_html=True)
                cols = st.columns(5)
                for col, (code, name) in zip(cols, teams):
                    with col:
                        if st.button(f"**{code}**\n\n{name}", key=f"pick_{code}", use_container_width=True):
                            st.session_state.team_code = code
                            st.switch_page("pages/1_Club_Operations_Center.py")
