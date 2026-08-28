"""선수 리포트 — E 추천과 시뮬레이션을 features_v1에 연결한다."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

# Streamlit이 app/pages를 실행 기준으로 잡아도 프로젝트 패키지를 찾도록 한다.
ROOT = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "app"
for import_path in (ROOT, APP_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from src.models.recommend import (
    AutoencoderRecommender,
    ReplacementRecommender,
    adapt_features_v1,
    load_knn_artifact,
)
from src.service.simulation import (
    TeamStrength,
    calculate_team_strength,
    evaluate_replacements,
    simulate,
)
from ui.theme import inject_css, init_state, page_header, require_team, section, topbar, wrap

FEATURES_PATH = ROOT / "data" / "processed" / "features_v1.parquet"
KNN_PATH = ROOT / "models" / "recommend_knn.pkl"
AUTOENCODER_PATH = ROOT / "models" / "recommend_autoencoder.pt"


@st.cache_data(show_spinner=False)
def load_players() -> pd.DataFrame:
    return adapt_features_v1(pd.read_parquet(FEATURES_PATH))


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
        format_func=lambda pid: f"{pid} · {team_players.loc[team_players.player_id.astype(str).eq(pid), 'role'].iloc[0]} · 전력 {team_players.loc[team_players.player_id.astype(str).eq(pid), 'overall_score'].iloc[0]:.1f}",
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
    st.caption(f"현재 추천 모델: {recommender_kind}")
    model_top = candidates.sort_values("rank", kind="stable").iloc[0]
    st.info(
        f"{recommender_kind} 유사도 1순위: {model_top['player_id']} "
        f"(유사도 {model_top['similarity']:.4f}) · "
        "최종 영입 순위는 문서 F6-3의 net effect와 예상 순위로 다시 계산합니다."
    )
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

    candidate_ids = evaluated["player_id"].astype(str).tolist()
    replacement_id = st.selectbox(
        "대체 후보",
        candidate_ids,
        format_func=lambda pid: (
            f"{pid} · 예상 {int(evaluated.loc[evaluated.player_id.astype(str).eq(pid), 'rank_after'].iloc[0])}위"
            f" · 최종 변화 {evaluated.loc[evaluated.player_id.astype(str).eq(pid), 'net_effect'].iloc[0]:+.2%}p"
        ),
    )
    replacement = evaluated.loc[evaluated["player_id"].astype(str) == replacement_id].iloc[0]

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
        f'<b>{selected_id}</b> 이탈 시 승률은 '
        f'<b>{result.current_win_rate:.2%} → {result.after_departure_win_rate:.2%}</b>로 변합니다. '
        f'(impact {result.impact:+.2%}p)<br>'
        f'<b>{replacement_id}</b> 투입 후 <b>{result.after_replacement_win_rate:.2%}</b>, '
        f'예상 순위는 <b>{result.rank_before}위 → {result.rank_after}위</b>입니다.'
        '</div>',
        unsafe_allow_html=True,
    )

    section("승률에 미치는 영향")
    c1, c2, c3 = st.columns(3)
    c1.metric("이탈 영향", f"{result.impact:+.2%}p")
    c2.metric("대체 효과", f"{result.replacement_effect:+.2%}p")
    c3.metric("최종 변화", f"{result.net_effect:+.2%}p")

    section("영입 시뮬레이션", "예상 순위·net effect 우선")
    display = evaluated[
        [
            "recommendation_rank", "rank", "recommender", "player_id", "team_last",
            "role", "similarity",
            "after_replacement_win_rate", "replacement_effect", "net_effect", "rank_after",
        ]
    ].copy()
    display.columns = [
        "추천 순위", "모델 원순위", "추천 모델", "선수", "소속", "역할", "유사도",
        "대체 후 승률", "대체 효과", "최종 변화", "예상 순위",
    ]
    st.dataframe(display, hide_index=True, use_container_width=True)

    with st.expander("이 예측의 근거"):
        used = [
            column for column in
            ["overall_score", "off_score", "pit_score", "g_ratio", "ops_z", "era_z", "whip_z"]
            if column in season_players.columns
        ]
        st.write(f"비교 피처: {', '.join(used)}")
        st.caption("동일 시즌·동일 역할 후보를 표준화한 뒤 코사인 유사도를 계산합니다.")
