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
from ui.theme import DIVISIONS, TEAM_NAMES, inject_css, init_state, page_header, placeholder, section  # noqa: E402

st.set_page_config(page_title="구단 선택", page_icon="⚾", layout="wide", initial_sidebar_state="collapsed")
inject_css()
init_state()

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
                # 오늘자 경기가 없으면(휴식일 등) 가장 가까운 다음 날짜를 보여준다.
                next_date = games.game_date.min()
                todays_games = games[games.game_date == next_date]
                st.caption(f"오늘 예정된 경기가 없어 다음 경기일({next_date})을 표시합니다.")

            display = todays_games.copy()
            display["home"] = display.home_team.map(_to_ui_code).map(TEAM_NAMES).fillna(display.home_team)
            display["away"] = display.away_team.map(_to_ui_code).map(TEAM_NAMES).fillna(display.away_team)
            display["winner"] = display.predicted_winner.map(_to_ui_code).map(TEAM_NAMES).fillna(display.predicted_winner)
            display["홈팀 승리확률"] = (display.ensemble_home_win_proba * 100).round(1).astype(str) + "%"
            display = display.rename(columns={"away": "원정", "home": "홈", "winner": "예상 승리팀"})
            st.dataframe(
                display[["원정", "홈", "홈팀 승리확률", "예상 승리팀"]],
                hide_index=True,
                width="stretch",
            )
            st.caption("game.py(LogReg+RF+MLP 앙상블) 예측 — 실제 미래 결과가 아닌 모델 추정치입니다.")

    with tab_rank:
        if not TEAM_SEASON_PATH.exists():
            placeholder(
                "예상 순위",
                "`src/models/win_rate.py`(A) 로 팀 승률을 예측하고 "
                "`team_season.pred_rank` 컬럼을 채워야 표시할 수 있습니다.",
            )
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

            display = ranked.copy()
            display["팀"] = display.team_id.map(_to_ui_code).map(TEAM_NAMES).fillna(display.team_id)
            display["승률"] = (display.win_rate * 100).round(1).astype(str) + "%"
            display = display.rename(columns={rank_col: "순위"})
            st.dataframe(
                display[["순위", "팀", "승률"]],
                hide_index=True,
                width="stretch",
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
