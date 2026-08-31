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
from src.models.next_strength import (  # noqa: E402
    apply_next_strength_projection,
    load_next_strength_model,
    predict_all_next_season_strength,
    predict_next_season_strength,
)
from src.service.simulation import (  # noqa: E402
    TeamStrength,
    calculate_team_strength,
    evaluate_replacements,
    simulate,
)
from src.models.recommend_policy_ranker import RecommendationPolicyRanker  # noqa: E402
from src.service.recommendation_scoring import (  # noqa: E402
    blend_policy_model_score,
    replacement_need_score,
    score_recommendations,
    select_recommendation_slots,
)
from ui.winrate import predict_win_rate_from_strength, win_rate_caption  # noqa: E402
from ui.datasource import load_features as load_features_df, source_caption  # noqa: E402
from ui.photos import headshot_url, load_mlbam_lookup  # noqa: E402
from ui.risk import (  # noqa: E402
    evidence_html,
    load_departure_model,
    load_reason_model,
    load_reason_thresholds,
    predict_departure_risk,
    predict_reason_tags,
    reason_badge_html,
    reason_explain_html,
    reason_proba_html,
)
from ui.theme import (  # noqa: E402
    badge,
    icon,
    inject_css,
    init_state,
    page_header,
    player_card_html,
    player_hero_card_html,
    _risk_tone,
    require_team,
    section,
    topbar,
    trend_chart_svg,
    wrap,
)

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data" / "final" / "features_v1.parquet"
KNN_PATH = ROOT / "models" / "recommend_knn.pkl"
AUTOENCODER_PATH = ROOT / "models" / "recommend_autoencoder.pt"
NEXT_STRENGTH_PATH = ROOT / "models" / "strength_xgb.ubj"
POLICY_RANKER_PATH = ROOT / "models" / "recommend_policy_ranker.ubj"
# strength_xgb(R² 0.560)가 strength_mlp(R² 0.472)보다 확실히 정확한데
# 실제 서비스(다음 시즌 예측 추세선)는 계속 strength_mlp를 쓰고 있었다 —
# 둘 다 LAG_FEATURES 기반 2D 입력이라 인터페이스가 동일해서 경로만 바꾸면 됨
# (next_strength.py의 feature_names_in_ 체크로 호환성 확인 완료).
DEPARTURE_MODEL_PATH = ROOT / "models" / "departure_lgbm.pkl"
REASON_MODEL_PATH = ROOT / "models" / "reason_rf.pkl"
PEOPLE_PATH = ROOT / "data" / "processed" / "People.csv"
PLAYERS_PATH = ROOT / "data" / "final" / "players.csv"

ROLE_LABEL = {"B": "타자", "P": "투수", "TWO": "투타겸업"}
POSITION_LABEL = {
    "P": "투수", "C": "포수", "1B": "1루수", "2B": "2루수", "3B": "3루수",
    "SS": "유격수", "OF": "외야수", "DH": "지명타자",
}


def _adapted_catalog() -> pd.DataFrame:
    """E의 추천 계약으로 변환 + position 별칭 부여.

    recommend.py의 ReplacementRecommender._filter_candidates()는 catalog에
    "position" 컬럼이 있으면 role(B/P/TWO)보다 먼저 그 컬럼으로 후보를 좁힌다
    (이미 구현돼 있던 로직 — E 파일은 안 건드림). features_v1에 D가 추가한
    primary_position을 그 이름으로 얹어주기만 하면 "같은 포지션 우선 추천"이
    바로 켜진다.
    """
    df = adapt_features_v1(pd.read_parquet(FEATURES_PATH))
    model = load_next_strength_model(NEXT_STRENGTH_PATH)
    projected = predict_all_next_season_strength(df, model)
    df = df.merge(projected, on=["player_id", "season"], how="left")
    # Supabase 우선, 실패 시 리포 내 parquet 폴백 (ui/datasource.py)
    if "primary_position" in df.columns:
        df["position"] = df["primary_position"]
    return df


@st.cache_data(show_spinner=False)
def load_players() -> pd.DataFrame:
    return _adapted_catalog()


@st.cache_data(show_spinner=False)
def load_name_lookup() -> dict[str, str]:
    """playerID → 실명 매핑. data/final/players.csv(2026 신인 195명 포함, features_v1의
    전체 player_id를 100% 커버)를 우선으로 쓰고, Lahman People.csv를 보조로 합친다 —
    People.csv 단독으로는 2026 신인 184명이 빠져서 로스터/추천 카드에 이름 대신
    player_id(예: "acunajo01")가 그대로 노출되는 문제가 있었다."""
    names: dict[str, str] = {}
    if PEOPLE_PATH.exists():
        people = pd.read_csv(PEOPLE_PATH, usecols=["playerID", "nameFirst", "nameLast"])
        pnames = (people["nameFirst"].fillna("") + " " + people["nameLast"].fillna("")).str.strip()
        names.update(dict(zip(people["playerID"], pnames)))
    if PLAYERS_PATH.exists():
        players = pd.read_csv(PLAYERS_PATH, usecols=["player_id", "name_first", "name_last"])
        fnames = (players["name_first"].fillna("") + " " + players["name_last"].fillna("")).str.strip()
        names.update({pid: n for pid, n in zip(players["player_id"], fnames) if n})
    return names


@st.cache_resource(show_spinner=False)
def load_saved_knn(data_version: int, model_version: int) -> ReplacementRecommender:
    """저장된 KNN 설정을 최신 features_v1 카탈로그에 연결한다."""
    del data_version, model_version
    return load_knn_artifact(KNN_PATH, _adapted_catalog())


@st.cache_resource(show_spinner=False)
def load_saved_autoencoder(data_version: int, model_version: int) -> AutoencoderRecommender:
    """저장된 Autoencoder 가중치와 전처리 통계를 최신 카탈로그에 연결한다."""
    del data_version, model_version
    return AutoencoderRecommender.load_artifact(AUTOENCODER_PATH, _adapted_catalog())


@st.cache_resource(show_spinner=False)
def load_policy_ranker(model_version: int) -> RecommendationPolicyRanker:
    del model_version
    return RecommendationPolicyRanker.load(POLICY_RANKER_PATH)


@st.cache_data(show_spinner=False)
def load_next_strength_projections(
    data_version: int,
    model_version: int,
) -> pd.DataFrame:
    """최신 features_v1을 D의 XGBoost에 연결해 t+1 전력을 계산한다."""
    del data_version, model_version
    players = adapt_features_v1(load_features_df())
    model = load_next_strength_model(NEXT_STRENGTH_PATH)
    return predict_next_season_strength(players, model)


def _num(row: pd.Series, col: str, default: float = 0.0) -> float:
    """row[col] 이 없거나 NaN 이면 default. (`x or default` 는 NaN 이 truthy라 못 걸러낸다.)"""
    value = row.get(col)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    return float(value)


def predict_win_rate(strength: TeamStrength) -> float:
    # 계수는 ui/winrate.py 에 실데이터(510개 팀·시즌)로 적합해 두었다.
    # 예전엔 이 자리에 검증 안 된 상수(0.35 + overall*0.003)가 두 페이지에
    # 복붙돼 있었고, 실제 33.4%p 승률 차이를 3.9%p 로 압축하고 있었다.
    return predict_win_rate_from_strength(strength.overall)


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
    photo_lookup = load_mlbam_lookup()
    departure_model = load_departure_model(
        DEPARTURE_MODEL_PATH.stat().st_mtime_ns if DEPARTURE_MODEL_PATH.exists() else 0
    )
    reason_model = load_reason_model(
        REASON_MODEL_PATH.stat().st_mtime_ns if REASON_MODEL_PATH.exists() else 0
    )
    reason_thresholds = load_reason_thresholds(
        FEATURES_PATH.stat().st_mtime_ns if FEATURES_PATH.exists() else 0
    )

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
            f" · 전력 {team_players.loc[team_players.player_id.astype(str).eq(pid), 'overall_score'].iloc[0]:.2f}"
        ),
    )
    st.session_state.selected_player_id = selected_id

    # ── 이탈 대상 프로파일 히어로 카드 (레이더 차트 + 이탈위험 + 연관 요인) ──
    section("이탈 대상 프로파일", "이탈이 이 서비스의 핵심 질문입니다 — 전력뿐 아니라 위험도·연관 요인까지", icon="target")
    selected_row = team_players.loc[team_players.player_id.astype(str) == selected_id].iloc[0]
    sel_role = selected_row.get("role", "B")
    sel_position = selected_row.get("primary_position")
    radar_axes: list[tuple[str, float]] = [("종합", _num(selected_row, "overall_score"))]
    if sel_role in ("B", "TWO"):
        radar_axes.append(("타격", _num(selected_row, "off_score")))
    if sel_role in ("P", "TWO"):
        radar_axes.append(("투구", _num(selected_row, "pit_score")))
    if pd.notna(selected_row.get("def_score")):
        radar_axes.append(("수비", _num(selected_row, "def_score")))
    radar_axes.append(("출전율", _num(selected_row, "g_ratio") * 100))
    radar_axes.append(("경험", min(_num(selected_row, "exp") / 15 * 100, 100)))

    departure_risk = predict_departure_risk(departure_model, selected_row.to_frame().T).iloc[0]
    reason_df = predict_reason_tags(reason_model, players, [selected_id])
    reason_row = reason_df.iloc[0] if not reason_df.empty else None
    reason_tag = reason_row["reason_tag"] if reason_row is not None else ""
    replacement_need = replacement_need_score(
        float(departure_risk) if pd.notna(departure_risk) else 0.0,
        reason_row["reason_proba"] if reason_row is not None else {},
    )

    chips = [
        f"나이 {_num(selected_row, 'age'):.0f}세",
        f"경력 {_num(selected_row, 'exp'):.0f}년",
        f"{season}시즌",
    ]
    if pd.notna(sel_position):
        chips.append(POSITION_LABEL.get(sel_position, sel_position))
    chips.append(f"교체 필요도 {replacement_need:.0%}")

    # 리그 전체(같은 시즌, 전 구단) 대비 순위 — overall_score는 시즌별 min-max
    # 정규화라서 그 시즌 최저 선수는 항상 정확히 0.00이 나오는 구조적 특성이 있다.
    # "0.00"만 보면 계산 오류처럼 보일 수 있어 리그 순위/퍼센타일을 함께 보여준다.
    ovr_val = _num(selected_row, "overall_score")
    league_pool = season_players["overall_score"].dropna()
    league_total = int(len(league_pool))
    league_rank = int((league_pool > ovr_val).sum()) + 1 if league_total else None

    st.markdown(
        player_hero_card_html(
            name=names.get(selected_id, selected_id),
            role_label=ROLE_LABEL.get(sel_role, sel_role),
            team=team_code,
            ovr=ovr_val,
            radar_axes=radar_axes,
            chips=chips,
            photo_url=headshot_url(selected_id, photo_lookup),
            league_rank=league_rank,
            league_total=league_total,
            # 레이더는 "모양"으로 균형을 보여주고, 링은 핵심 3개를 숫자로 못박는다
            # (첨부한 선수 대시보드 레퍼런스의 스탯 타일 역할).
            rings=[
                ("전력", ovr_val, "var(--team-accent)"),
                ("출전율", _num(selected_row, "g_ratio") * 100, "var(--violet)"),
                # 이탈위험만 의미색(위험할수록 붉게) — 나머지와 같은 색이면
                # "98%"가 좋은 수치인지 나쁜 수치인지 한눈에 안 읽힌다.
                ("이탈위험", float(departure_risk) * 100, _risk_tone(float(departure_risk))),
            ],
        ),
        unsafe_allow_html=True,
    )
    if ovr_val == 0.0 and pd.notna(
        selected_row.get("off_score")
        if sel_role in ("B", "TWO")
        else selected_row.get("pit_score")
    ):
        role_score_name = "공격" if sel_role in ("B", "TWO") else "투구"
        defense_note = (
            f" 수비 전력은 {_num(selected_row, 'def_score'):.1f}점으로 별도 표시됩니다."
            if pd.notna(selected_row.get("def_score"))
            else ""
        )
        st.info(
            f"전력 0.0은 결측치가 아니라 {season}시즌 비교 집단에서 "
            f"{role_score_name} 전력 최저값이라는 뜻입니다.{defense_note} "
            "현재 종합 전력은 역할별 공격/투구 점수를 사용하며 수비 점수를 "
            "합산하지 않습니다."
        )

    risk_c1, risk_c2 = st.columns([1, 2])
    with risk_c1:
        if pd.notna(departure_risk):
            risk_kind = "risk" if departure_risk >= 0.5 else ("warn" if departure_risk >= 0.3 else "gain")
            risk_label = "위험 높음" if departure_risk >= 0.5 else ("주의" if departure_risk >= 0.3 else "안정적")
            st.markdown(
                f'<div class="gm-card" style="text-align:center">'
                f'<div class="gm-kpi-l">{icon("siren", 12)} 모델 추정 이탈위험</div>'
                f'<div class="gm-kpi-v" style="color:var(--{risk_kind})">{departure_risk:.0%}</div>'
                f'{badge(risk_label, risk_kind)}'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("이탈위험 모델을 불러올 수 없습니다.")
    with risk_c2:
        badge_html = reason_badge_html(reason_tag)
        if badge_html and reason_row is not None:
            st.markdown(
                f'<div class="gm-card">'
                f'<div class="gm-kpi-l" style="margin-bottom:8px">📋 모델 추정 연관 요인</div>'
                f'{badge_html}'
                f'<div style="margin-top:10px">{reason_proba_html(reason_row["reason_proba"])}</div>'
                f'{reason_explain_html(reason_tag, reason_row, reason_thresholds)}'
                f'{evidence_html(reason_row, reason_thresholds)}'
                f'<div style="margin-top:8px;font-size:11px;color:var(--muted)">'
                f'이탈이 확정된 사실이 아니라, 현재 피처 프로필이 과거 이탈 사례 중 '
                f'어떤 유형과 비슷한지에 대한 모델 추정입니다 — 인과관계 단정 아님.</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("연관 요인 모델을 불러올 수 없습니다.")

    # ── 전력 추세 그래프 (실측 + 가능하면 다음 시즌 예측) ──
    section("전력 추세", "실선: 실측 시즌 · 노란 점선: 다음 시즌 예측(strength_xgb, 확정 아님)", icon="chart")
    history = players.loc[players.player_id.astype(str) == selected_id].sort_values("season")
    trend_seasons = history["season"].astype(int).tolist()
    trend_values = history["overall_score"].fillna(0).tolist()

    future_season = None
    future_value = None
    try:
        trend_projections = load_next_strength_projections(
            FEATURES_PATH.stat().st_mtime_ns,
            NEXT_STRENGTH_PATH.stat().st_mtime_ns,
        )
        proj_row = trend_projections.loc[trend_projections.player_id.astype(str) == selected_id]
        if not proj_row.empty:
            future_season = int(proj_row["prediction_season"].iloc[0])
            future_value = float(proj_row["predicted_next_overall_score"].iloc[0])
    except (FileNotFoundError, OSError, ValueError):
        pass  # 예측 모델이 없어도 실측 추세만으로 그래프는 그린다

    if len(trend_seasons) >= 2:
        st.markdown(
            f'<div class="gm-trend-card">'
            f'{trend_chart_svg(trend_seasons, trend_values, future_season=future_season, future_value=future_value)}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("추세를 보여줄 과거 시즌 기록이 2시즌 미만입니다.")

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

    # 추천 모델(KNN 코사인 vs Autoencoder)은 GM이 신경 쓸 일이 아니다 — 둘 다
    # 돌려서 후보 풀을 합치고, 겹치면 유사도 높은 쪽만 남긴다. "어떤 알고리즘"이
    # 아니라 "누가 좋은 후보인가"만 화면에 남도록 내부 구현으로 감춘다.
    POOL_SIZE = 8
    candidate_pool: list[pd.DataFrame] = []
    recommender_errors: list[str] = []
    try:
        knn = load_saved_knn(FEATURES_PATH.stat().st_mtime_ns, KNN_PATH.stat().st_mtime_ns)
        candidate_pool.append(knn.recommend(selected_id, season, n_recommendations=POOL_SIZE))
    except (ValueError, RuntimeError) as exc:
        recommender_errors.append(f"KNN: {exc}")
    try:
        autoencoder = load_saved_autoencoder(
            FEATURES_PATH.stat().st_mtime_ns, AUTOENCODER_PATH.stat().st_mtime_ns
        )
        candidate_pool.append(autoencoder.recommend(selected_id, season, n_recommendations=POOL_SIZE))
    except ImportError:
        # torch 미설치 환경(경량 배포)에서는 Autoencoder 추천을 건너뛴다.
        # KNN 만으로도 후보 풀이 채워지므로 화면은 정상 동작한다 — torch 는
        # CUDA 의존까지 끌고 와서 Streamlit Cloud 메모리를 크게 먹는다.
        recommender_errors.append("Autoencoder: torch 미설치 환경이라 건너뜀(KNN 추천만 사용)")
    except (ValueError, RuntimeError) as exc:
        recommender_errors.append(f"Autoencoder: {exc}")

    if not candidate_pool:
        st.warning(" / ".join(recommender_errors) or "추천 후보를 찾을 수 없습니다.")
        st.stop()

    filter_notes = {c.attrs.get("filter_note", "") for c in candidate_pool} - {""}
    reconstruction_losses = [c.attrs.get("reconstruction_loss") for c in candidate_pool if c.attrs.get("reconstruction_loss") is not None]

    combined = pd.concat(candidate_pool, ignore_index=True)
    combined = combined.sort_values("similarity", ascending=False).drop_duplicates("player_id", keep="first")

    # 포지션을 아는 선수는 동일 포지션 후보만 다음 단계로 통과시킨다.
    if pd.notna(sel_position) and "position" in combined.columns:
        combined = combined.loc[combined["position"].eq(sel_position)].copy()
        if combined.empty:
            st.warning("동일 포지션의 영입 후보가 없습니다.")
            st.stop()

    # 여기서는 5명으로 자르지 않는다 - evaluate_replacements 이후 이탈위험 기준으로
    # 다시 추려야 하므로, 그 전까지는 풀을 넉넉히 유지한다.
    candidates = combined.reset_index(drop=True)
    candidates.attrs["filter_note"] = " / ".join(sorted(filter_notes))
    if reconstruction_losses:
        candidates.attrs["reconstruction_loss"] = float(np.mean(reconstruction_losses))

    filter_note = candidates.attrs.get("filter_note", "")
    simulation_players = season_players
    simulation_team_players = team_players
    simulation_candidates = candidates

    # FA는 오프시즌 의사결정이므로 D가 예측한 다음 시즌 전력으로 계산한다.
    if scenario == "fa":
        try:
            projections = load_next_strength_projections(
                FEATURES_PATH.stat().st_mtime_ns,
                NEXT_STRENGTH_PATH.stat().st_mtime_ns,
            )
            simulation_players = apply_next_strength_projection(
                season_players, projections
            )
            simulation_team_players = apply_next_strength_projection(
                team_players, projections
            )
            simulation_candidates = apply_next_strength_projection(
                candidates, projections
            )
            st.caption(
                f"FA 시나리오는 D strength_xgb의 {season + 1}시즌 예측 전력을 반영합니다."
            )
        except (FileNotFoundError, OSError, ValueError) as exc:
            # 모델 또는 계약 데이터가 없으면 현재 시즌 값으로 조용히 대체하지 않는다.
            st.warning(f"다음 시즌 전력 모델을 적용할 수 없습니다: {exc}")
            st.stop()

    rank_predictor = make_rank_predictor(simulation_players)
    try:
        evaluated = evaluate_replacements(
            simulation_team_players,
            selected_id,
            simulation_candidates,
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

    evaluated["departure_risk"] = predict_departure_risk(departure_model, evaluated)
    candidate_reasons = predict_reason_tags(
        reason_model, players, evaluated["player_id"].astype(str)
    )
    reason_probability_map = dict(
        zip(
            candidate_reasons["player_id"].astype(str),
            candidate_reasons["reason_proba"],
            strict=True,
        )
    )
    reason_injury_score_map = dict(
        zip(
            candidate_reasons["player_id"].astype(str),
            candidate_reasons["reason_injury_score"],
            strict=True,
        )
    )
    evaluated = score_recommendations(
        evaluated,
        season_players,
        reason_probability_map,
        reason_injury_scores=reason_injury_score_map,
        target_reason_probabilities=(
            reason_row["reason_proba"] if reason_row is not None else {}
        ),
    )
    try:
        policy_ranker = load_policy_ranker(POLICY_RANKER_PATH.stat().st_mtime_ns)
        policy_scores = policy_ranker.predict(selected_row, evaluated)
        evaluated = blend_policy_model_score(
            evaluated, policy_scores, model_weight=0.05
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        # 정책 모델이 없어도 해석 가능한 규칙 점수는 계속 제공한다.
        evaluated["rule_recommend_score"] = evaluated["final_recommend_score"]
        evaluated["policy_model_score"] = np.nan
    evaluated = select_recommendation_slots(evaluated)
    if len(evaluated) < 5:
        st.caption(f"동일 포지션 후보가 {len(evaluated)}명뿐입니다.")

    # ── 영입 후보 카드 (FIFA UT 스타일 선택 UI) ──
    section(
        "영입 후보",
        "포지션·전력·성장성·영입 가능성·건강·승률 회복·비용 효율 종합 순",
        icon="swap",
    )

    candidate_ids = evaluated["player_id"].astype(str).tolist()
    sel_key = f"pcard_sel::{team_code}::{selected_id}::{scenario}"
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
            if pd.notna(row.get("departure_risk")):
                stat_rows.append(("이탈률", _num(row, "departure_risk") * 100))
            stat_rows.append(("예상전력", _num(row, "predicted_next_overall_score")))
            stat_rows.append(("성장성", _num(row, "growth_potential") * 100))
            stat_rows.append(("가성비", _num(row, "cost_efficiency") * 100))
            stat_rows.append(("종합점수", _num(row, "final_recommend_score") * 100))
            if pd.notna(row.get("policy_model_score")):
                stat_rows.append(("정책모델", _num(row, "policy_model_score") * 100))

            with col:
                match_badges = []
                tier_label = str(row.get("recommend_tier_label", ""))
                if tier_label:
                    tier_kind = "gain" if tier_label == "최적 영입 후보" else (
                        "warn" if tier_label.startswith("참고용") else "navy"
                    )
                    match_badges.append(badge(tier_label, tier_kind))
                if row.get("matched_on") == "position":
                    pos = row.get("position") or row.get("primary_position")
                    match_badges.append(badge(f"같은 포지션 ({POSITION_LABEL.get(pos, pos)})", "gain"))
                if pd.notna(row.get("departure_risk")) and row["departure_risk"] >= 0.5:
                    match_badges.append(badge("영입 가능성 높음", "warn"))
                match_badges.append(
                    badge(
                        f"팀 가치 {int(row.get('team_player_rank', 16))}위 · 비용 {_num(row, 'acquisition_cost_weight'):.2f}",
                        "navy",
                    )
                )
                match_badges.append(
                    badge(
                        f"영입 가능 {_num(row, 'market_availability'):.0%} · 위험 {_num(row, 'harm_risk'):.0%}",
                        "gain" if _num(row, "market_availability") >= _num(row, "harm_risk") else "warn",
                    )
                )
                if _num(row, "harm_risk") >= 0.3:
                    match_badges.append(badge("영입 위험 주의", "risk"))
                if match_badges:
                    st.markdown(
                        f'<div style="display:flex;gap:4px;flex-wrap:wrap;margin-bottom:4px">{"".join(match_badges)}</div>',
                        unsafe_allow_html=True,
                    )
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
                        photo_url=headshot_url(pid, photo_lookup),
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
        simulation_team_players,
        selected_id,
        predict_win_rate,
        replacement_player=replacement,
        rank_predictor=rank_predictor,
        scenario=scenario,
    )

    section("단장 브리핑", icon="target")

    # 방출은 즉시 이탈 결과를, 트레이드와 FA는 대체 완료 결과를 브리핑 기준으로 쓴다.
    if scenario == "release":
        briefing_win_rate = result.after_departure_win_rate
        briefing_rank = rank_predictor(result.after_departure_strength)
        briefing_detail = (
            f'즉시 이탈 기준 예상 순위는 <b>{result.rank_before}위 → '
            f'{briefing_rank}위</b>입니다. 이후 <b>{replacement_name}</b> 투입 시 '
            f'승률은 <b>{result.after_replacement_win_rate:.2%}</b>까지 회복됩니다.'
        )
    else:
        briefing_win_rate = result.after_replacement_win_rate
        briefing_detail = (
            f'<b>{replacement_name}</b> 반영 후 예상 순위는 '
            f'<b>{result.rank_before}위 → {result.rank_after}위</b>입니다.'
        )

    # 후보가 선택된 트레이드·FA 결과에는 대체 완료 승률이 반드시 있어야 한다.
    if briefing_win_rate is None:
        st.error("브리핑에 표시할 최종 승률이 없습니다.")
        st.stop()

    st.markdown(
        '<div class="gm-card">'
        f'{result.scenario_label}({result.effective_timing} · {result.absence_scope}) 시나리오에서 '
        f'<b>{departing_name}</b> 관련 의사결정 기준 승률은 '
        f'<b>{result.current_win_rate:.2%} → {briefing_win_rate:.2%}</b>입니다.<br>'
        f'{briefing_detail}'
        '</div>',
        unsafe_allow_html=True,
    )

    section("승률에 미치는 영향", icon="chart")
    c1, c2, c3 = st.columns(3)
    c1.metric("이탈 영향", f"{result.impact:+.2%}p")
    c2.metric("대체 효과", f"{result.replacement_effect:+.2%}p")
    c3.metric("최종 변화", f"{result.net_effect:+.2%}p")

    with st.expander("이 예측의 근거"):
        used = [
            column for column in
            ["overall_score", "off_score", "pit_score", "g_ratio", "ops_z", "era_z", "whip_z"]
            if column in season_players.columns
        ]
        st.write(f"비교 피처: {', '.join(used)}")
        st.caption("동일 시즌·동일 역할 후보를 표준화한 뒤 코사인 유사도를 계산합니다.")
