"""이탈 위험 + 연관 요인 — B(departure)·C(reason)가 저장한 모델을 화면에 연결한다.

이 파일은 B/C의 모델을 학습/수정하지 않는다. app/pages/2_Player_Report.py가
이미 E의 recommend.py·next_strength.py를 같은 방식(저장된 모델을 불러와
predict만 호출)으로 쓰고 있는 것과 동일한 패턴이다.

표시 원칙 (reason.py 상단 docstring 그대로 계승)
    원인 태그는 확정된 이탈 사실의 사후 설명이 아니라, 현재 활동 중인
    선수에게도 "지금 피처 프로필이 과거 이탈자들의 어떤 유형과 비슷한가"를
    모델이 추정한 것이다. 화면에는 반드시 "연관 요인(모델 추정)"으로 표시하고
    "~때문에 이탈함" 같은 인과 단정 문구를 쓰지 않는다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.models.departure import DepartureLGBM
from src.models.reason import ReasonRandomForest, add_reason_features

ROOT = Path(__file__).resolve().parents[2]

# 연관 요인 태그 -> (표시 라벨, 아이콘, 배지 색상 키). "~때문" 같은 인과 단정
# 문구를 피하고 전부 "연관"/"추정"으로 끝맺는다 (reason.py 표시 원칙).
REASON_DISPLAY: dict[str, tuple[str, str, str]] = {
    "injury_associated": ("부상 연관", "🩹", "risk"),
    "performance_decline": ("성적 하락 연관", "📉", "warn"),
    "career_stage": ("베테랑 시기 연관", "🕰️", "violet"),
    "mixed": ("복합 요인 연관", "⚠️", "risk"),
    "unknown": ("판단 근거 부족", "❔", "navy"),
}


@st.cache_resource(show_spinner=False)
def load_departure_model(model_version: int) -> DepartureLGBM | None:
    """등록된 이탈위험 모델 중 test AUC가 가장 높은 것을 쓴다(현재 departure_lgbm)."""
    del model_version
    try:
        return DepartureLGBM.load("departure_lgbm")
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner=False)
def load_reason_model(model_version: int) -> ReasonRandomForest | None:
    try:
        return ReasonRandomForest.load("reason_rf")
    except FileNotFoundError:
        return None


def predict_departure_risk(model: DepartureLGBM, players: pd.DataFrame) -> pd.Series:
    """player-season 행별 이탈위험 확률(0~1). 모델이 없으면 전부 NaN.

    단일 행을 ``series.to_frame().T``로 넘기면 전 컬럼이 object dtype이 되어
    LightGBM이 거부한다(실측 확인) - 호출부가 어떤 형태로 넘기든 방어적으로
    숫자형 변환한다.
    """
    if model is None or players.empty:
        return pd.Series(np.nan, index=players.index)
    numeric = players.reindex(columns=model.feature_names).apply(pd.to_numeric, errors="coerce")
    proba = model.predict_proba(numeric)
    pos_idx = 1 if len(model.classes_) < 2 else int(np.argmax([float(c) for c in model.classes_]))
    return pd.Series(proba[:, pos_idx], index=players.index)


def predict_reason_tags(model: ReasonRandomForest, all_seasons: pd.DataFrame, target_ids) -> pd.DataFrame:
    """target_ids(현재 시즌 player_id 목록)에 대해 연관 요인 태그를 추정한다.

    overall_score_delta(전 시즌 대비 변화)를 계산하려면 그 선수의 과거 시즌
    행이 같이 필요하다 — all_seasons는 features_v1 전체(다시즌)를 넘겨야 한다.
    had_injury/il_stint_count는 features_v1에 아직 없어(부상 데이터 별도 파이프라인
    미병합) next_strength.py와 동일하게 0으로 채운다 — 값을 지어내지 않고
    "부상 신호 없음"으로 보수적으로 처리한다는 뜻.

    Returns:
        target_ids 순서의 DataFrame: player_id, reason_tag(원본 클래스명), 없으면 빈 문자열
    """
    if model is None or not len(target_ids):
        return pd.DataFrame({"player_id": list(target_ids), "reason_tag": ""})

    df = all_seasons.copy()
    if "had_injury" not in df.columns:
        df["had_injury"] = 0.0
    if "il_stint_count" not in df.columns:
        df["il_stint_count"] = 0.0

    try:
        featured = add_reason_features(df)
    except (KeyError, ValueError):
        return pd.DataFrame({"player_id": list(target_ids), "reason_tag": ""})

    latest = (
        featured[featured["player_id"].astype(str).isin([str(i) for i in target_ids])]
        .sort_values("season")
        .groupby("player_id", as_index=False)
        .last()
    )
    if latest.empty:
        return pd.DataFrame({"player_id": list(target_ids), "reason_tag": ""})

    try:
        X = latest.reindex(columns=model.feature_names)
        tags = model.predict(X)
    except (KeyError, ValueError):
        return pd.DataFrame({"player_id": list(target_ids), "reason_tag": ""})

    return pd.DataFrame({"player_id": latest["player_id"].astype(str).values, "reason_tag": tags})


def reason_badge_html(tag: str) -> str:
    label, icon, kind = REASON_DISPLAY.get(tag, ("", "", "navy"))
    if not label:
        return ""
    return f'<span class="gm-badge {kind}">{icon} {label}</span>'
