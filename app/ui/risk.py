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

from ui.theme import icon
from src.models.departure import DepartureLGBM
from src.models.reason import ReasonRandomForest, ReasonThresholds, add_reason_features, fit_reason_thresholds

ROOT = Path(__file__).resolve().parents[2]

# 연관 요인 태그 -> (표시 라벨, 아이콘, 배지 색상 키). "~때문" 같은 인과 단정
# 문구를 피하고 전부 "연관"/"추정"으로 끝맺는다 (reason.py 표시 원칙).
REASON_DISPLAY: dict[str, tuple[str, str, str]] = {
    "injury_associated": ("부상 연관", "bandage", "risk"),
    "performance_decline": ("성적 하락 연관", "trend-down", "warn"),
    "career_stage": ("베테랑 시기 연관", "clock", "violet"),
    "mixed": ("복합 요인 연관", "alert", "risk"),
    # [2026-08-30] unknown 이 라벨의 48%를 차지해 "판단 근거 부족"만 뜨는 문제로
    # reason.py 에서 두 클래스를 분리했다. 원인을 단정하지 않고 관측된 패턴을
    # 이름 붙인 것이라 라벨도 "~중 이동"으로 끝맺는다.
    "early_career_move": ("저연차 이동 연관", "swap", "violet"),
    "stable_performance_move": ("성적 유지 중 이동", "trend-up", "gain"),
    "unknown": ("판단 근거 부족", "question", "navy"),
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


INJURY_STINTS_PATH = ROOT / "data" / "final" / "player_injury_stints.csv"


@st.cache_data(show_spinner=False)
def _load_real_injury_data(data_version: int) -> pd.DataFrame | None:
    """mlb_injury_pipeline.py가 만든 실제 IL 데이터(2026-08-28부터 복귀일 매칭 포함,
    injury_risk_score 포함). 파일이 없으면 None - 호출부가 0-fill로 대체한다."""
    del data_version
    if not INJURY_STINTS_PATH.exists():
        return None
    try:
        return pd.read_csv(INJURY_STINTS_PATH)
    except (OSError, ValueError):
        return None


def _merge_injury(df: pd.DataFrame) -> pd.DataFrame:
    """실제 IL 데이터를 (player_id, season)로 붙인다. 매칭 안 되는 행은 0으로
    채운다 — "부상 없음"이 아니라 "이 데이터에서 IL 기록이 관측 안 됨"이라는
    reason.py의 기존 구분을 그대로 따른다(merge_injury_data와 동일한 취지,
    reason.py 자체는 안 건드림 - injury_risk_score 컬럼명도 그쪽이 이미
    기대하던 이름 그대로라 reason.py 코드 변경 없이 자동으로 쓰인다)."""
    injury = _load_real_injury_data(
        INJURY_STINTS_PATH.stat().st_mtime_ns if INJURY_STINTS_PATH.exists() else 0
    )
    out = df.copy()
    if injury is None:
        if "had_injury" not in out.columns:
            out["had_injury"] = 0.0
        if "il_stint_count" not in out.columns:
            out["il_stint_count"] = 0.0
        return out

    keep = [c for c in ["player_id", "season", "had_injury", "il_stint_count", "injury_risk_score"]
            if c in injury.columns]
    out = out.merge(injury[keep], on=["player_id", "season"], how="left")
    out["had_injury"] = out["had_injury"].fillna(0.0)
    out["il_stint_count"] = out["il_stint_count"].fillna(0.0)
    return out


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
    df = _merge_injury(df)
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
    had_injury/il_stint_count/injury_risk_score는 features_v1에 없어
    mlb_injury_pipeline.py가 만든 실제 IL 데이터(_merge_injury)를 여기서
    직접 붙인다 - 그 파일도 없으면 next_strength.py와 동일하게 0으로
    채운다(값을 지어내지 않고 "부상 신호 없음"으로 보수적으로 처리).

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

    df = _merge_injury(all_seasons)

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
    # REASON_DISPLAY 의 두 번째 값은 이모지가 아니라 theme.icon() 의 아이콘
    # 이름이다 — 이모지는 OS/브라우저마다 모양이 달라 톤이 안 맞아서 SVG로 교체했다.
    label, icon_name, kind = REASON_DISPLAY.get(tag, ("", "", "navy"))
    if not label:
        return ""
    return f'<span class="gm-badge {kind}">{icon(icon_name, 10)} {label}</span>'


def reason_proba_html(proba: dict, *, top_n: int = 3) -> str:
    """클래스별 확률을 막대 목록으로 - 1등 태그만 보여주면 "0.52 vs 0.48이라
    거의 반반이었다" 같은 정보가 사라진다. 상위 top_n개를 막대로 보여준다."""
    if not proba:
        return ""
    ranked = sorted(proba.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    rows = []
    for tag, p in ranked:
        label, icon_name, _kind = REASON_DISPLAY.get(tag, (tag, "", "navy"))
        rows.append(
            '<div style="display:flex;align-items:center;gap:6px;font-size:11.5px;margin-bottom:3px">'
            f'<span style="width:92px;flex-shrink:0;color:var(--muted)">{icon(icon_name, 10)} {label}</span>'
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


# ──────────────────────────────────────────────────────────────────────
# "복합 요인 연관"이 왜 복합인지 설명하기
#
# reason.py의 assign_reason_labels()는 개별 수치가 아니라 "신호군" 3개를
# 각각 켜고, 2개 이상 켜지면 primary_reason="mixed"로 정한다:
#
#   부상    : injury_record_matched AND had_injury>0 AND injury_score >= 임계
#   성적하락 : overall_score_delta <= 임계  OR  g_chg <= 임계   (둘 중 하나면 켜짐)
#   경력단계 : y_path=="league_exit" AND (age >= 임계 OR exp >= 임계)
#
# 기존 화면은 수치를 나열만 해서 "그래서 왜 복합인지"가 안 보였다 —
# 수치가 어느 군에 속하고 그 군이 켜졌는지를 묶어서 보여준다.
#
# 주의: 위 조건에는 "이미 이탈했는가"(departed / league_exit) 게이트가 들어
# 있는데, 현역 선수는 그걸 알 수 없다. 그래서 여기서는 게이트를 뺀 "조건
# 충족 여부"만 판정하고, 화면에도 그렇게 표기한다 — 확정 라벨이 아니다.
# ──────────────────────────────────────────────────────────────────────

def reason_signal_groups(row, thresholds: ReasonThresholds | None) -> list[dict]:
    """신호군 3개를 평가한다. 각 군의 활성 여부와 그 근거가 된 조건을 함께 준다."""
    if thresholds is None:
        return []

    def _get(key: str) -> float | None:
        value = row.get(key) if isinstance(row, (dict, pd.Series)) else None
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return float(value)

    age, exp = _get("age"), _get("exp")
    delta, g_chg = _get("overall_score_delta"), _get("g_chg")
    injury = _get("reason_injury_score")

    groups: list[dict] = []

    hits = []
    if injury is not None and injury >= thresholds.injury_risk and injury > 0:
        hits.append(f"부상 신호 {injury:.2f} ≥ {thresholds.injury_risk:.2f}")
    groups.append({
        "key": "injury_associated", "label": "부상",
        "active": bool(hits), "hits": hits,
        "miss": f"부상 신호 {injury:.2f} (임계 {thresholds.injury_risk:.2f})" if injury is not None else "부상 기록 없음",
    })

    hits = []
    if delta is not None and delta <= thresholds.score_delta:
        hits.append(f"전력 {delta:+.1f}점 ≤ {thresholds.score_delta:+.1f}")
    if g_chg is not None and g_chg <= thresholds.g_change:
        hits.append(f"출전비중 {g_chg:+.2f} ≤ {thresholds.g_change:+.2f}")
    groups.append({
        "key": "performance_decline", "label": "성적 하락",
        "active": bool(hits), "hits": hits,
        "miss": "하락 폭이 임계 미만",
    })

    hits = []
    if age is not None and age >= thresholds.career_age:
        hits.append(f"나이 {age:.0f}세 ≥ {thresholds.career_age:.0f}세")
    if exp is not None and exp >= thresholds.career_exp:
        hits.append(f"경력 {exp:.0f}년 ≥ {thresholds.career_exp:.0f}년")
    groups.append({
        "key": "career_stage", "label": "경력 단계",
        "active": bool(hits), "hits": hits,
        "miss": "나이·경력 모두 임계 미만",
    })

    return groups


def reason_explain_html(tag: str, row, thresholds: ReasonThresholds | None,
                        *, compact: bool = False) -> str:
    """태그가 왜 그렇게 나왔는지를 신호군 단위로 설명한다."""
    groups = reason_signal_groups(row, thresholds)
    if not groups:
        return ""
    n_active = sum(1 for g in groups if g["active"])

    # 주의: 화면의 태그는 "모델 예측"이고, 아래 신호군은 "학습 라벨을 만들 때
    # 쓴 규칙"을 지금 수치에 그대로 적용한 결과다. 모델은 규칙을 외운 게 아니라
    # 과거 이탈 사례의 패턴을 학습한 것이라 둘이 갈릴 수 있다 — 갈릴 때
    # "N개가 켜져서 복합"이라고 쓰면 없는 인과를 단정하는 것이 된다.
    rule_note = f"학습 규칙 기준으로 지금 켜진 신호군은 <b>{n_active}개</b>입니다."

    if tag == "mixed":
        if n_active >= 2:
            head = (
                "모델이 <b>복합 요인</b>으로 봤습니다. " + rule_note
                + " 규칙에서도 2개 이상이면 복합으로 분류하므로 서로 맞아떨어집니다."
            )
        else:
            head = (
                "모델이 <b>복합 요인</b>으로 봤습니다. " + rule_note
                + " 규칙만 보면 복합 조건(2개 이상)에 못 미치는데, 모델은 규칙을 그대로"
                " 따르지 않고 과거 이탈 사례의 패턴으로 판단하기 때문에 이렇게 갈릴 수"
                " 있습니다 — 아래 수치를 직접 보고 판단하세요."
            )
    elif tag == "unknown":
        head = (
            "모델이 <b>판단 근거 부족</b>으로 봤습니다. " + rule_note
            + " 위험이 없다는 뜻이 아니라, 분류할 만한 뚜렷한 신호가 안 잡혔다는 뜻입니다."
        )
    elif tag == "early_career_move":
        head = (
            "모델이 <b>저연차 이동 연관</b>으로 봤습니다. " + rule_note
            + " 부상·성적하락·베테랑 신호가 아니라, 팀 이동이 잦은 경력 초반 구간에"
            " 해당한다는 관측입니다 — 이동 원인을 단정하는 것이 아닙니다."
        )
    elif tag == "stable_performance_move":
        head = (
            "모델이 <b>성적 유지 중 이동</b>으로 봤습니다. " + rule_note
            + " 성적이 떨어지지도, 부상 신호가 있지도 않은 상태의 이동이라"
            " 구단 사정(트레이드·로스터 조정 등)일 가능성을 함께 보셔야 합니다."
        )
    else:
        label = REASON_DISPLAY.get(tag, (tag, "", ""))[0]
        head = f"모델이 <b>{label}</b>으로 봤습니다. " + rule_note

    items = []
    for g in groups:
        on = g["active"]
        detail = " · ".join(g["hits"]) if on else g["miss"]
        items.append(
            f'<div class="gm-sig{" on" if on else ""}">'
            f'<span class="gm-sig-dot"></span>'
            f'<span class="gm-sig-label">{g["label"]}</span>'
            f'<span class="gm-sig-detail">{detail}</span>'
            "</div>"
        )

    note = (
        ""
        if compact
        else '<div class="gm-sig-note">신호군 정의는 학습 라벨을 만들 때 쓴 규칙(reason.py)과 '
             '같습니다. 다만 (1) 화면의 태그는 규칙이 아니라 <b>모델 예측</b>이고, '
             '(2) 규칙에 있는 "이미 이탈했는가" 조건은 현역 선수에게 확인할 수 없어 빼고 '
             '계산했습니다. 확정된 사실이 아니라 참고 근거입니다.</div>'
    )
    return f'<div class="gm-sig-box"><div class="gm-sig-head">{head}</div>{"".join(items)}{note}</div>'
