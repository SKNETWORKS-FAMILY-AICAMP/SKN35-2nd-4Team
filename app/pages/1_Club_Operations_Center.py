"""구단 상황실 — E 시뮬레이션을 features_v1과 연결한다."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# streamlit run 은 app/ 을 sys.path[0] 으로 잡아 리포 루트의 src.* 가 안 잡힐 때가
# 있다(다른 페이지가 먼저 실행돼 루트를 추가해둔 경우에만 우연히 성공). 페이지
# 단독 진입에서도 항상 되도록 3_Model_Information.py 와 동일한 가드를 둔다.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.models.recommend import adapt_features_v1  # noqa: E402
from src.service.simulation import TeamStrength, calculate_team_strength, simulate  # noqa: E402
from ui.risk import (  # noqa: E402
    REASON_DISPLAY,
    load_departure_model,
    load_reason_model,
    predict_departure_risk,
    predict_reason_tags,
    reason_badge_html,
)
from ui.theme import badge, inject_css, init_state, page_header, require_team, section, topbar, wrap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data" / "final" / "features_v1.parquet"
DEPARTURE_MODEL_PATH = ROOT / "models" / "departure_lgbm.pkl"
REASON_MODEL_PATH = ROOT / "models" / "reason_rf.pkl"
PEOPLE_PATH = ROOT / "data" / "processed" / "People.csv"

ROLE_LABEL = {"B": "타자", "P": "투수", "TWO": "투타겸업"}


@st.cache_data(show_spinner=False)
def load_players() -> pd.DataFrame:
    return adapt_features_v1(pd.read_parquet(FEATURES_PATH))


@st.cache_data(show_spinner=False)
def load_name_lookup() -> dict[str, str]:
    if not PEOPLE_PATH.exists():
        return {}
    people = pd.read_csv(PEOPLE_PATH, usecols=["playerID", "nameFirst", "nameLast"])
    names = (people["nameFirst"].fillna("") + " " + people["nameLast"].fillna("")).str.strip()
    return dict(zip(people["playerID"], names))


def predict_win_rate(strength: TeamStrength) -> float:
    return float(np.clip(0.35 + strength.overall * 0.003, 0.25, 0.75))


def make_rank_predictor(season_players: pd.DataFrame):
    baselines = {
        team: calculate_team_strength(group).overall
        for team, group in season_players.groupby("team_last")
    }

    def predict_rank(strength: TeamStrength) -> int:
        return 1 + sum(value > strength.overall for value in baselines.values())

    return predict_rank


st.set_page_config(page_title="구단 상황실", page_icon="⚾", layout="wide", initial_sidebar_state="collapsed")
inject_css()
init_state()
team_code = require_team()
topbar("구단 상황실")

with wrap():
    page_header("구단 상황실", "이탈이 팀 전력과 순위에 미치는 영향")

    try:
        players = load_players()
    except FileNotFoundError:
        st.error("data/final/features_v1.parquet 파일이 없습니다.")
        st.stop()
    except ImportError:
        st.error("parquet을 읽기 위한 pyarrow 설치가 필요합니다.")
        st.stop()
    except ValueError as exc:
        st.error(f"features_v1 데이터 계약 오류: {exc}")
        st.stop()
    except Exception as exc:
        # 파일 손상 등 예상하지 못한 로딩 오류도 traceback 대신 화면에 안내한다.
        st.error(f"features_v1을 읽는 중 오류가 발생했습니다: {exc}")
        st.stop()

    quality = players.attrs.get("data_quality", {})
    if quality.get("excluded_rows", 0) or quality.get("imputed_g_ratio", 0):
        st.info(
            f"데이터 정리: {quality.get('excluded_rows', 0)}행 제외, "
            f"출전 비중 {quality.get('imputed_g_ratio', 0)}건 보정"
        )

    season = int(players["season"].max())
    season_players = players.loc[players["season"] == season].copy()
    team_players = season_players.loc[season_players["team_last"] == team_code].copy()
    if team_players.empty:
        st.warning(f"{season}시즌 {team_code} 선수 데이터가 없습니다.")
        st.stop()

    team_players = team_players.sort_values("overall_score", ascending=False).copy()

    # ── 이탈위험 + 연관 요인 (이 서비스의 핵심 — "누가 떠날 위험이 큰가"를
    # 로스터 전체에 대해 먼저 보여준다. 전력순 정렬만 있던 기존 화면의 공백) ──
    names = load_name_lookup()
    departure_model = load_departure_model(
        DEPARTURE_MODEL_PATH.stat().st_mtime_ns if DEPARTURE_MODEL_PATH.exists() else 0
    )
    reason_model = load_reason_model(
        REASON_MODEL_PATH.stat().st_mtime_ns if REASON_MODEL_PATH.exists() else 0
    )
    team_players["departure_risk"] = predict_departure_risk(departure_model, team_players)
    reason_tags = predict_reason_tags(reason_model, players, team_players["player_id"].astype(str))
    reason_map = dict(zip(reason_tags["player_id"], reason_tags["reason_tag"]))
    team_players["reason_tag"] = team_players["player_id"].astype(str).map(reason_map)
    team_players["이름"] = team_players["player_id"].astype(str).map(lambda pid: names.get(pid, pid))

    default_player = st.session_state.get("selected_player_id")
    ids = team_players["player_id"].astype(str).tolist()

    sort_label = st.radio(
        "로스터 정렬",
        ["🚨 이탈위험순", "💪 전력순"],
        horizontal=True,
        key="roster_sort",
    )
    ranked = (
        team_players.sort_values("departure_risk", ascending=False, na_position="last")
        if sort_label.endswith("이탈위험순")
        else team_players.sort_values("overall_score", ascending=False)
    )

    selected_id = st.selectbox(
        "이탈 시뮬레이션 선수",
        ranked["player_id"].astype(str).tolist(),
        index=(
            ranked["player_id"].astype(str).tolist().index(default_player)
            if default_player in ids else 0
        ),
        format_func=lambda pid: (
            f"{names.get(pid, pid)} · 전력 {team_players.loc[team_players.player_id.astype(str).eq(pid), 'overall_score'].iloc[0]:.1f}"
            f" · 이탈위험 {team_players.loc[team_players.player_id.astype(str).eq(pid), 'departure_risk'].iloc[0]:.0%}"
        ),
    )
    st.session_state.selected_player_id = selected_id

    result = simulate(
        team_players,
        selected_id,
        predict_win_rate,
        rank_predictor=make_rank_predictor(season_players),
    )

    section("순위 변동 패널", f"{season}시즌 features_v1 기반", icon="target")
    c1, c2, c3 = st.columns(3)
    c1.metric("현재 예상 승률", f"{result.current_win_rate:.2%}")
    c2.metric("이탈 후 예상 승률", f"{result.after_departure_win_rate:.2%}", f"{result.impact:+.2%}p")
    c3.metric("예상 순위", f"{result.rank_before}위 → {result.rank_after}위")
    st.caption("승률·순위 함수는 simulation.py에 주입되어 실제 예측 함수로 교체할 수 있습니다.")

    top_risk = ranked.dropna(subset=["departure_risk"]).nlargest(3, "departure_risk")
    if not top_risk.empty and sort_label.endswith("이탈위험순"):
        section("이탈위험 TOP 3", "모델 추정 — 인과관계 단정 아님", icon="shield")
        rc = st.columns(3)
        for col, (_, r) in zip(rc, top_risk.iterrows()):
            with col:
                badge_html = reason_badge_html(r.get("reason_tag", ""))
                col.markdown(
                    f'<div class="gm-card" style="text-align:center">'
                    f'<div style="font-weight:800;font-size:14.5px">{r["이름"]}</div>'
                    f'<div class="gm-kpi-v" style="color:var(--risk);font-size:22px;margin:4px 0">'
                    f'{r["departure_risk"]:.0%}</div>'
                    f'{badge_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    section("전력 로스터", sort_label, icon="team")
    roster = ranked[["이름", "role", "overall_score", "g_ratio", "departure_risk", "reason_tag"]].head(20).copy()
    roster["role"] = roster["role"].map(ROLE_LABEL).fillna(roster["role"])
    roster["departure_risk"] = roster["departure_risk"].map(lambda v: f"{v:.0%}" if pd.notna(v) else "—")
    roster["reason_tag"] = roster["reason_tag"].fillna("").map(
        lambda t: f"{REASON_DISPLAY[t][1]} {REASON_DISPLAY[t][0]}" if t in REASON_DISPLAY else ""
    )
    roster.columns = ["선수", "역할", "전력", "출전 비중", "이탈위험", "연관 요인(모델 추정)"]
    st.dataframe(roster, hide_index=True, use_container_width=True)

    if st.button("선수 리포트에서 대체 후보 보기", type="primary"):
        st.switch_page("pages/2_Player_Report.py")
