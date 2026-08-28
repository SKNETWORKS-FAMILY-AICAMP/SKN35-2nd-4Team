"""이탈 위험 + 연관 요인 — B(departure)·C(reason)가 저장한 모델을 화면에 연결한다.

이 파일은 B/C의 모델을 학습/수정하지 않는다. app/pages/2_Player_Report.py가
이미 E의 recommend.py·next_strength.py를 같은 방식(저장된 모델을 불러와
predict만 호출)으로 쓰고 있는 것과 동일한 패턴이다.

표시 원칙 (reason.py 상단 docstring 그대로 계승)
    원인 태그는 확정된 이탈 사실의 사후 설명이 아니라, 현재 활동 중인
    선수에게도 "지금 피처 프로필이 과거 이탈자들의 어떤 유형과 비슷한가"를
    모델이 추정한 것이다. 화면에는 반드시 "연관 요인(모델 추정)"으로 표시하고
    "~때문에 이탈함" 같은 인과 단정 문구를 쓰지 않는다.

2026-08-28 추가: "판단 근거 부족/복합 요인"만 뜨면 이해가 안 된다는 피드백 —
    태그 하나만 보여주지 않고 (1) 클래스별 확률 상위 항목, (2) 그 판정에 실제로
    쓰인 수치(나이·경력·최근 전력 변화·출전비중 변화)를 훈련 때와 동일한
    임계값과 나란히 보여준다. reason.py의 규칙/모델을 새로 만들지 않고,
    이미 있는 것을 더 투명하게 보여주는 것 — "세분화"를 태그 종류를 늘리는
    대신(그러려면 규칙 자체를 다시 설계해야 함, C 영역) 근거를 더 촘촘히
    보여주는 방향으로 풀었다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.models.departure import DepartureLGBM
from src.models.reason import ReasonRandomForest, ReasonThresholds, add_reason_features, fit_reason_thresholds

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


@st.cache_resource(show_spinner=False)
def load_reason_thresholds(data_version: int) -> ReasonThresholds | None:
    """reason.py 학습 때와 동일한 기준(2021년까지만)으로 임계값을 다시 계산한다.

    "이 선수 수치가 임계값을 넘었는지"를 보여주려면 학습 때 쓴 것과 정확히
    같은 임계값이어야 의미가 있다 — 대충 다른 기준으로 비교하지 않는다.
    """
    del data_version
    try:
        from src.features import contract

        df = contract.load_features()
    except Exception:
        return None
    df = df.copy()
    if "had_injury" not in df.columns:
        df["had_injury"] = 0.0
    try:
        featured = add_reason_features(df)
        return fit_reason_thresholds(featured)
    except (KeyError, ValueError):
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


# 근거로 보여줄 원본 수치 + 사람이 읽을 라벨. reason.py의 판정에 실제로 쓰이는
# 값들이다(assign_reason_labels 참고) - 지어낸 부가 정보가 아니라 판정 근거 그 자체.
_EVIDENCE_COLS = ["age", "exp", "overall_score_delta", "g_chg", "reason_injury_score"]


def predict_reason_tags(model: ReasonRandomForest, all_seasons: pd.DataFrame, target_ids) -> pd.DataFrame:
    """target_ids(현재 시즌 player_id 목록)에 대해 연관 요인 태그 + 확률분포 + 근거 수치를 추정한다.

    overall_score_delta(전 시즌 대비 변화)를 계산하려면 그 선수의 과거 시즌
    행이 같이 필요하다 — all_seasons는 features_v1 전체(다시즌)를 넘겨야 한다.
    had_injury/il_stint_count는 features_v1에 아직 없어(부상 데이터 별도 파이프라인
    미병합) next_strength.py와 동일하게 0으로 채운다 — 값을 지어내지 않고
    "부상 신호 없음"으로 보수적으로 처리한다는 뜻.

    Returns:
        player_id, reason_tag, reason_proba(dict[str,float] 클래스별 확률),
        + _EVIDENCE_COLS. 모델/데이터가 없으면 reason_tag=""인 빈 값들.
    """
    empty = pd.DataFrame({
        "player_id": [str(i) for i in target_ids],
        "reason_tag": "",
        "reason_proba": [{}] * len(target_ids),
        **{c: np.nan for c in _EVIDENCE_COLS},
    })
    if model is None or not len(target_ids):
        return empty

    df = all_seasons.copy()
    if "had_injury" not in df.columns:
        df["had_injury"] = 0.0
    if "il_stint_count" not in df.columns:
        df["il_stint_count"] = 0.0

    try:
        featured = add_reason_features(df)
    except (KeyError, ValueError):
        return empty

    latest = (
        featured[featured["player_id"].astype(str).isin([str(i) for i in target_ids])]
        .sort_values("season")
        .groupby("player_id", as_index=False)
        .last()
    )
    if latest.empty:
        return empty

    try:
        X = latest.reindex(columns=model.feature_names)
        tags = model.predict(X)
        proba = model.predict_proba(X)
    except (KeyError, ValueError):
        return empty

    proba_dicts = [dict(zip(model.classes_, row)) for row in proba]
    out = pd.DataFrame({
        "player_id": latest["player_id"].astype(str).values,
        "reason_tag": tags,
        "reason_proba": proba_dicts,
    })
    for col in _EVIDENCE_COLS:
        out[col] = latest[col].to_numpy() if col in latest.columns else np.nan
    return out


def reason_badge_html(tag: str) -> str:
    label, icon, kind = REASON_DISPLAY.get(tag, ("", "", "navy"))
    if not label:
        return ""
    return f'<span class="gm-badge {kind}">{icon} {label}</span>'


def reason_proba_html(proba: dict, *, top_n: int = 3) -> str:
    """클래스별 확률을 막대 목록으로 - 1등 태그만 보여주면 "0.52 vs 0.48이라
    거의 반반이었다" 같은 정보가 사라진다. 상위 top_n개를 막대로 보여준다."""
    if not proba:
        return ""
    ranked = sorted(proba.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    rows = []
    for tag, p in ranked:
        label, icon, _kind = REASON_DISPLAY.get(tag, (tag, "", "navy"))
        rows.append(
            '<div style="display:flex;align-items:center;gap:6px;font-size:11.5px;margin-bottom:3px">'
            f'<span style="width:92px;flex-shrink:0;color:var(--muted)">{icon} {label}</span>'
            '<span style="flex:1;height:6px;border-radius:3px;background:var(--line);overflow:hidden">'
            f'<span style="display:block;height:100%;width:{p * 100:.0f}%;background:var(--navy);border-radius:3px"></span>'
            "</span>"
            f'<span style="width:34px;text-align:right;font-weight:700;font-variant-numeric:tabular-nums">{p:.0%}</span>'
            "</div>"
        )
    return "".join(rows)


def evidence_html(row: dict | pd.Series, thresholds: ReasonThresholds | None) -> str:
    """판정에 쓰인 실제 수치를 임계값과 나란히 보여준다 — "복합 요인"이 왜
    복합인지, "판단 근거 부족"이 정말 아무 신호가 없는 건지 숫자로 확인 가능하게.
    임계값 방향(넘으면 위험 신호인지 아닌지)은 reason.py의 assign_reason_labels
    조건식 그대로 따른다.
    """

    def _get(key: str) -> float | None:
        value = row.get(key) if isinstance(row, (dict, pd.Series)) else None
        return None if value is None or (isinstance(value, float) and np.isnan(value)) else float(value)

    age, exp = _get("age"), _get("exp")
    delta, g_chg = _get("overall_score_delta"), _get("g_chg")
    injury = _get("reason_injury_score")

    lines = []
    if age is not None:
        over = thresholds is not None and age >= thresholds.career_age
        lines.append(("나이", f"{age:.0f}세", f"임계 {thresholds.career_age:.0f}세↑" if thresholds else "", over))
    if exp is not None:
        over = thresholds is not None and exp >= thresholds.career_exp
        lines.append(("경력", f"{exp:.0f}년", f"임계 {thresholds.career_exp:.0f}년↑" if thresholds else "", over))
    if delta is not None:
        over = thresholds is not None and delta <= thresholds.score_delta
        lines.append(("전력 변화(전 시즌 대비)", f"{delta:+.1f}점", f"임계 {thresholds.score_delta:+.1f}↓" if thresholds else "", over))
    if g_chg is not None:
        over = thresholds is not None and g_chg <= thresholds.g_change
        lines.append(("출전비중 변화", f"{g_chg:+.2f}", f"임계 {thresholds.g_change:+.2f}↓" if thresholds else "", over))
    if injury is not None:
        over = thresholds is not None and injury >= thresholds.injury_risk and injury > 0
        lines.append(("부상 신호", f"{injury:.2f}", f"임계 {thresholds.injury_risk:.2f}↑" if thresholds else "", over))

    if not lines:
        return ""

    rows = "".join(
        '<div style="display:flex;justify-content:space-between;gap:8px;font-size:11.5px;'
        f'padding:3px 0;color:{"var(--risk)" if flag else "var(--muted)"}">'
        f'<span>{"🔺 " if flag else ""}{label}</span>'
        f'<span style="font-weight:700;font-variant-numeric:tabular-nums">{val} <span style="font-weight:400;opacity:.6">({thr})</span></span>'
        "</div>"
        for label, val, thr, flag in lines
    )
    return (
        '<div style="margin-top:6px;padding-top:6px;border-top:1px solid var(--line)">'
        f"{rows}"
        '<div style="font-size:10px;color:var(--faint);margin-top:4px">🔺 = 훈련 구간(~2021) 기준 임계값을 넘은 항목</div>'
        "</div>"
    )
