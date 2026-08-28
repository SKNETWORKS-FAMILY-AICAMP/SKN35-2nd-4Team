"""선수 리포트 — E 추천과 시뮬레이션을 features_v1에 연결한다."""

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

from src.models.recommend import (  # noqa: E402
    AutoencoderRecommender,
    ReplacementRecommender,
    adapt_features_v1,
    load_knn_artifact,
)
from src.service.simulation import (  # noqa: E402
    TeamStrength,
    calculate_team_strength,
    evaluate_replacements,
    simulate,
)
from ui.theme import inject_css, init_state, page_header, player_card_html, require_team, section, topbar, wrap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data" / "processed" / "features_v1.parquet"
KNN_PATH = ROOT / "models" / "recommend_knn.pkl"
AUTOENCODER_PATH = ROOT / "models" / "recommend_autoencoder.pt"
PEOPLE_PATH = ROOT / "data" / "processed" / "People.csv"

ROLE_LABEL = {"B": "타자", "P": "투수", "TWO": "투타겸업"}


@st.cache_data(show_spinner=False)
def load_players() -> pd.DataFrame:
    return adapt_features_v1(pd.read_parquet(FEATURES_PATH))


@st.cache_data(show_spinner=False)
def load_name_lookup() -> dict[str, str]:
    """Lahman People.csv 로 playerID → 실명을 매핑한다. 없으면 빈 dict(= ID 그대로 표시)."""
    if not PEOPLE_PATH.exists():
        return {}
    people = pd.read_csv(PEOPLE_PATH, usecols=["playerID", "nameFirst", "nameLast"])
    names = (people["nameFirst"].fillna("") + " " + people["nameLast"].fillna("")).str.strip()
    return dict(zip(people["playerID"], names))


@st.cache_resource(show_spinner=False)
def load_saved_knn(data_version: int, model_version: int) -> ReplacementRecommender:
    """저장된 KNN 설정을 최신 features_v1 카탈로그에 연결한다."""
    del data_version, model_version
    players = adapt_features_v1(pd.read_parquet(FEATURES_PATH))
    return load_knn_artifact(KNN_PATH, players)


@st.cache_resource(show_spinner=False)
def load_saved_autoencoder(data_version: int, model_version: int) -> AutoencoderRecommender:
    """저장된 Autoencoder 가중치와 전처리 통계를 최신 카탈로그에 연결한다."""
    del data_version, model_version
    players = adapt_features_v1(pd.read_parquet(FEATURES_PATH))
    return AutoencoderRecommender.load_artifact(AUTOENCODER_PATH, players)


def _num(row: pd.Series, col: str, default: float = 0.0) -> float:
    """row[col] 이 없거나 NaN 이면 default. (`x or default` 는 NaN 이 truthy라 못 걸러낸다.)"""
    value = row.get(col)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    return float(value)


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


st.set_page_config(page_title="선수 리포트", page_icon="⚾", layout="wide", initial_sidebar_state="collapsed")
inject_css()
init_state()
team_code = require_team()
topbar("선수 리포트")

with wrap():
    page_header("선수 리포트", "대체 후보를 비교하고 영입 효과를 시뮬레이션합니다")

    names = load_name_lookup()

    try:
        players = load_players()
    except FileNotFoundError:
        st.error("data/processed/features_v1.parquet 파일이 없습니다.")
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

    team_players = team_players.sort_values("overall_score", ascending=False)
    player_ids = team_players["player_id"].astype(str).tolist()
    stored_id = st.session_state.get("selected_player_id")
    selected_id = st.selectbox(
        "이탈 대상 선수",
        player_ids,
        index=player_ids.index(stored_id) if stored_id in player_ids else 0,
        format_func=lambda pid: (
            f"{names.get(pid, pid)} · {ROLE_LABEL.get(team_players.loc[team_players.player_id.astype(str).eq(pid), 'role'].iloc[0], '')}"
            f" · 전력 {team_players.loc[team_players.player_id.astype(str).eq(pid), 'overall_score'].iloc[0]:.1f}"
        ),
    )
    st.session_state.selected_player_id = selected_id

    # 문서 F6-2: 트레이드=시즌 전체, FA=오프시즌, 방출=즉시.
    scenario_labels = {
        "트레이드 · 시즌 전체": "trade",
        "FA · 오프시즌": "fa",
        "방출 · 즉시": "release",
    }
    selected_scenario_label = st.radio(
        "이탈 유형별 시나리오",
        list(scenario_labels),
        horizontal=True,
    )
    scenario = scenario_labels[selected_scenario_label]

    recommender_kind = st.radio(
        "추천 모델",
        ["KNN 코사인", "Autoencoder"],
        horizontal=True,
    )

    try:
        if recommender_kind == "KNN 코사인":
            knn = load_saved_knn(
                FEATURES_PATH.stat().st_mtime_ns,
                KNN_PATH.stat().st_mtime_ns,
            )
            candidates = knn.recommend(
                selected_id,
                season,
                n_recommendations=5,
            )
        else:
            autoencoder = load_saved_autoencoder(
                FEATURES_PATH.stat().st_mtime_ns,
                AUTOENCODER_PATH.stat().st_mtime_ns,
            )
            candidates = autoencoder.recommend(
                selected_id,
                season,
                n_recommendations=5,
            )
    except (ValueError, RuntimeError) as exc:
        st.warning(str(exc))
        st.stop()

    filter_note = candidates.attrs.get("filter_note", "")
    rank_predictor = make_rank_predictor(season_players)
    try:
        evaluated = evaluate_replacements(
            team_players,
            selected_id,
            candidates,
            predict_win_rate,
            rank_predictor=rank_predictor,
            scenario=scenario,
        )
    except ValueError as exc:
        st.warning(str(exc))
        st.stop()

    # 일부 후보만 실패한 경우 성공한 후보 결과는 계속 보여준다.
    evaluation_errors = evaluated.attrs.get("evaluation_errors", [])
    if filter_note:
        st.caption(filter_note)
    reconstruction_loss = candidates.attrs.get("reconstruction_loss")
    if reconstruction_loss is not None:
        st.caption(f"Autoencoder 재구성 손실(MSE): {reconstruction_loss:.4f}")
    if evaluation_errors:
        st.warning(
            f"후보 {len(candidates)}명 중 {len(evaluation_errors)}명은 결측 또는 범위 오류로 제외했습니다."
        )

    # ── 영입 후보 카드 (FIFA UT 스타일 선택 UI) ──
    section("영입 후보", "카드를 클릭하듯 골라보세요 · 선택 즉시 아래 시뮬레이션이 갱신됩니다")

    evaluated = evaluated.reset_index(drop=True)
    candidate_ids = evaluated["player_id"].astype(str).tolist()
    sel_key = f"pcard_sel::{team_code}::{selected_id}::{scenario}::{recommender_kind}"
    if st.session_state.get(sel_key) not in candidate_ids:
        st.session_state[sel_key] = candidate_ids[0]  # 기본값: 최우선 추천 후보

    with st.container(key="pcard_section"):
        cols = st.columns(len(candidate_ids))
        for i, (col, (_, row)) in enumerate(zip(cols, evaluated.iterrows())):
            pid = str(row["player_id"])
            is_selected = st.session_state[sel_key] == pid
            role = row.get("role", "B")

            stat_rows: list[tuple[str, float]] = []
            if role == "P":
                stat_rows.append(("투구", _num(row, "pit_score")))
            elif role == "TWO":
                stat_rows.append(("타격", _num(row, "off_score")))
                stat_rows.append(("투구", _num(row, "pit_score")))
            else:
                stat_rows.append(("타격", _num(row, "off_score")))
            stat_rows.append(("출전", _num(row, "g_ratio") * 100))
            if pd.notna(row.get("similarity")):
                stat_rows.append(("적합", _num(row, "similarity") * 100))

            with col:
                st.markdown(
                    player_card_html(
                        index=i,
                        rank=int(row.get("recommendation_rank", i + 1)),
                        ovr=_num(row, "overall_score"),
                        position_label=ROLE_LABEL.get(role, role),
                        name=names.get(pid, pid),
                        team=str(row.get("team_last", "")),
                        stat_rows=stat_rows,
                        net_effect_pct=float(row["net_effect"]) * 100,
                        selected=is_selected,
                    ),
                    unsafe_allow_html=True,
                )
                if st.button(
                    "선택됨" if is_selected else "이 선수 영입",
                    key=f"pick_{pid}",
                    disabled=is_selected,
                    width="stretch",
                ):
                    st.session_state[sel_key] = pid
                    st.rerun()

    replacement_id = st.session_state[sel_key]
    replacement = evaluated.loc[evaluated["player_id"].astype(str) == replacement_id].iloc[0]
    departing_name = names.get(selected_id, selected_id)
    replacement_name = names.get(replacement_id, replacement_id)

    result = simulate(
        team_players,
        selected_id,
        predict_win_rate,
        replacement_player=replacement,
        rank_predictor=rank_predictor,
        scenario=scenario,
    )

    section("단장 브리핑")
    st.markdown(
        '<div class="gm-card">'
        f'{result.scenario_label}({result.effective_timing} · {result.absence_scope}) 시나리오에서 '
        f'<b>{departing_name}</b> 이탈 시 승률은 <b>{result.current_win_rate:.1%} → {result.after_departure_win_rate:.1%}</b>로 변합니다.<br>'
        f'<b>{replacement_name}</b> 투입 후 <b>{result.after_replacement_win_rate:.1%}</b>, '
        f'예상 순위는 <b>{result.rank_before}위 → {result.rank_after}위</b>입니다.'
        '</div>',
        unsafe_allow_html=True,
    )

    section("승률에 미치는 영향")
    c1, c2, c3 = st.columns(3)
    c1.metric("이탈 영향", f"{result.impact:+.1%}p")
    c2.metric("대체 효과", f"{result.replacement_effect:+.1%}p")
    c3.metric("최종 변화", f"{result.net_effect:+.1%}p")

    with st.expander("이 예측의 근거"):
        used = [
            column for column in
            ["overall_score", "off_score", "pit_score", "g_ratio", "ops_z", "era_z", "whip_z"]
            if column in season_players.columns
        ]
        st.write(f"비교 피처: {', '.join(used)}")
        st.caption("동일 시즌·동일 역할 후보를 표준화한 뒤 코사인 유사도를 계산합니다.")
