"""대체 선수 후보의 영입비용과 원인별 위험을 포함한 최종 점수 계산."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def acquisition_cost_weight(rank: float) -> float:
    """팀 내 선수 가치 순위를 프로젝트의 영입비용 구간으로 변환한다."""
    rank = int(rank)
    if rank <= 3:
        return 1.0
    if rank <= 6:
        return 0.85
    if rank <= 9:
        return 0.70
    if rank <= 12:
        return 0.55
    if rank <= 15:
        return 0.40
    return 0.25


def add_team_player_rank(players: pd.DataFrame) -> pd.DataFrame:
    """시즌·팀 안에서 overall_score 내림차순 선수 가치 순위를 산정한다."""
    required = {"player_id", "season", "team_last", "overall_score"}
    missing = required - set(players.columns)
    if missing:
        raise ValueError(f"선수 가치 랭크 필수 컬럼 누락: {sorted(missing)}")
    ranked = players.copy()
    ranked["team_player_rank"] = (
        ranked.groupby(["season", "team_last"], dropna=False)["overall_score"]
        .rank(method="first", ascending=False, na_option="bottom")
        .astype(int)
    )
    return ranked


def score_recommendations(
    candidates: pd.DataFrame,
    season_players: pd.DataFrame,
    reason_probabilities: Mapping[str, Mapping[str, float]],
    *,
    reason_injury_scores: Mapping[str, float] | None = None,
    target_reason_probabilities: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """비용·가용성·건강·전력·성장·승률 회복을 합쳐 추천 순위를 만든다."""
    if candidates.empty:
        return candidates.copy()

    ranks = add_team_player_rank(season_players)[
        ["player_id", "season", "team_last", "team_player_rank"]
    ].copy()
    ranks["player_id"] = ranks["player_id"].astype(str)
    scored = candidates.copy()
    scored["player_id"] = scored["player_id"].astype(str)
    scored = scored.merge(
        ranks,
        on=["player_id", "season", "team_last"],
        how="left",
        validate="many_to_one",
    )
    # 후보가 시즌 로스터에 없으면 비용을 임의 추정하지 않고 최하위 그룹으로 둔다.
    scored["team_player_rank"] = scored["team_player_rank"].fillna(16).astype(int)
    scored["acquisition_cost_weight"] = scored["team_player_rank"].map(
        acquisition_cost_weight
    )
    scored["cost_efficiency"] = 1.0 - scored["acquisition_cost_weight"]

    departure = pd.to_numeric(scored.get("departure_risk", 0.0), errors="coerce").fillna(0.0)
    availability: list[float] = []
    harm: list[float] = []
    profile_similarity: list[float] = []
    reason_classes = (
        "career_stage",
        "early_career_move",
        "injury_associated",
        "mixed",
        "moderate_performance_decline",
        "performance_decline",
        "stable_performance_move",
    )
    target_vector = np.asarray(
        [(target_reason_probabilities or {}).get(name, 0.0) for name in reason_classes],
        dtype=float,
    )
    for player_id, dep in zip(scored["player_id"], departure, strict=True):
        proba = reason_probabilities.get(str(player_id), {}) or {}
        positive = float(proba.get("early_career_move", 0.0)) + float(
            proba.get("stable_performance_move", 0.0)
        )
        adverse = (
            float(proba.get("injury_associated", 0.0))
            + float(proba.get("performance_decline", 0.0))
            + float(proba.get("moderate_performance_decline", 0.0))
            + float(proba.get("career_stage", 0.0))
        )
        availability.append(float(np.clip(dep * positive, 0.0, 1.0)))
        harm.append(float(np.clip(dep * adverse, 0.0, 1.0)))
        candidate_vector = np.asarray(
            [proba.get(name, 0.0) for name in reason_classes], dtype=float
        )
        denominator = np.linalg.norm(target_vector) * np.linalg.norm(candidate_vector)
        profile_similarity.append(
            0.0 if denominator == 0.0 else float(candidate_vector @ target_vector / denominator)
        )
    scored["market_availability"] = availability
    scored["harm_risk"] = harm
    scored["reason_profile_similarity"] = profile_similarity

    similarity = pd.to_numeric(scored.get("similarity", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    predicted = pd.to_numeric(
        scored.get("predicted_next_overall_score", scored.get("overall_score", 0.0)),
        errors="coerce",
    ).fillna(0.0)
    # FA 시뮬레이션은 overall_score를 예측값으로 교체하므로 성장성의 기준은
    # 교체 전에 보존한 current_overall_score를 우선 사용한다.
    overall = pd.to_numeric(
        scored.get("current_overall_score", scored.get("overall_score", 0.0)),
        errors="coerce",
    ).fillna(0.0)
    age = pd.to_numeric(scored.get("age", 30.0), errors="coerce").fillna(30.0)
    projected_gain = ((predicted - overall + 10.0) / 20.0).clip(0.0, 1.0)
    youth = ((32.0 - age) / 12.0).clip(0.0, 1.0)
    # 성장성은 확정 피처가 없어 합의 가능한 임시 계약으로 분리한다:
    # D 예상 전력 상승분 60% + 32세 이전 나이 여력 40%.
    scored["growth_potential"] = 0.6 * projected_gain + 0.4 * youth
    injury_scores = reason_injury_scores or {}
    scored["reason_injury_score"] = scored["player_id"].map(injury_scores)
    scored["reason_injury_score"] = pd.to_numeric(
        scored["reason_injury_score"], errors="coerce"
    ).fillna(0.0).clip(0.0, 1.0)
    scored["health_stability"] = 1.0 - scored["reason_injury_score"]
    next_strength = (predicted / 100.0).clip(0.0, 1.0)
    win_recovery = (
        pd.to_numeric(scored.get("net_effect", 0.0), errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .div(0.10)
        .clip(0.0, 1.0)
    )

    scored["final_recommend_score"] = (
        0.25 * similarity
        + 0.20 * next_strength
        + 0.15 * scored["growth_potential"]
        + 0.15 * scored["market_availability"]
        + 0.10 * scored["health_stability"]
        + 0.10 * win_recovery
        + 0.05 * scored["cost_efficiency"]
        - 0.15 * scored["harm_risk"]
    )

    # 특히 동일 비용인 16위 이하에서 적합도→성장성→위험→회복량으로 동점을 푼다.
    scored = scored.sort_values(
        [
            "final_recommend_score",
            "similarity",
            "growth_potential",
            "health_stability",
            "net_effect",
        ],
        ascending=[False, False, False, False, False],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    scored["recommendation_rank"] = np.arange(1, len(scored) + 1)
    return scored


def blend_policy_model_score(
    scored: pd.DataFrame,
    policy_scores: np.ndarray,
    *,
    model_weight: float = 0.05,
) -> pd.DataFrame:
    """검증 성능이 제한적인 정책 모델을 규칙 점수에 보수적으로 결합한다."""
    if not 0.0 <= model_weight <= 1.0:
        raise ValueError("model_weight는 0과 1 사이여야 합니다.")
    if len(scored) != len(policy_scores):
        raise ValueError("후보 수와 정책 모델 점수 수가 다릅니다.")
    blended = scored.copy()
    blended["rule_recommend_score"] = blended["final_recommend_score"]
    blended["policy_model_score"] = np.asarray(policy_scores, dtype=float)
    blended["final_recommend_score"] = (
        (1.0 - model_weight) * blended["rule_recommend_score"]
        + model_weight * blended["policy_model_score"]
    )
    blended = blended.sort_values(
        [
            "final_recommend_score",
            "similarity",
            "growth_potential",
            "health_stability",
            "net_effect",
        ],
        ascending=[False, False, False, False, False],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    blended["recommendation_rank"] = np.arange(1, len(blended) + 1)
    return blended


def replacement_need_score(
    departure_risk: float,
    reason_probabilities: Mapping[str, float],
) -> float:
    """기존 선수의 부상·하락·경력단계 연관 교체 필요도를 계산한다."""
    adverse = sum(
        float(reason_probabilities.get(name, 0.0))
        for name in (
            "injury_associated",
            "performance_decline",
            "moderate_performance_decline",
            "career_stage",
        )
    )
    return float(np.clip(float(departure_risk) * adverse, 0.0, 1.0))


def select_recommendation_slots(scored: pd.DataFrame) -> pd.DataFrame:
    """최적 2명→유사 후보 2명→참고 1명의 캐스케이드 슬롯을 구성한다."""
    if scored.empty:
        return scored.copy()
    selected: list[pd.Series] = []
    used: set[str] = set()

    optimal = scored.loc[
        scored["departure_risk"].ge(0.60)
        & scored["market_availability"].ge(0.50)
        & scored["harm_risk"].le(0.35)
    ].sort_values("final_recommend_score", ascending=False)
    for _, row in optimal.head(2).iterrows():
        row = row.copy()
        row["recommend_tier_label"] = "최적 영입 후보"
        selected.append(row)
        used.add(str(row["player_id"]))

    remaining = scored.loc[~scored["player_id"].astype(str).isin(used)].copy()
    top_count = min(len(remaining), max(4, int(np.ceil(len(remaining) * 0.30))))
    similar_ids = set(remaining.nlargest(top_count, "similarity")["player_id"].astype(str))
    similar_ids.update(
        remaining.nlargest(top_count, "reason_profile_similarity")["player_id"].astype(str)
    )
    similar = remaining.loc[remaining["player_id"].astype(str).isin(similar_ids)].sort_values(
        "final_recommend_score", ascending=False
    )
    for _, row in similar.iterrows():
        if len(selected) == 4:
            break
        row = row.copy()
        row["recommend_tier_label"] = "스탯·이탈원인 유사 후보"
        selected.append(row)
        used.add(str(row["player_id"]))

    # 앞 단계가 부족하면 순수 스탯 유사도로 최대 5명까지 완화한다.
    fallback = scored.loc[~scored["player_id"].astype(str).isin(used)].sort_values(
        "similarity", ascending=False
    )
    for _, row in fallback.iterrows():
        if len(selected) == 5:
            break
        row = row.copy()
        row["recommend_tier_label"] = "참고용 (시장 이탈 신호 없음)"
        selected.append(row)
        used.add(str(row["player_id"]))

    out = pd.DataFrame(selected).reset_index(drop=True)
    out["recommend_tier"] = np.arange(1, len(out) + 1)
    out["recommendation_rank"] = out["recommend_tier"]
    return out


__all__ = [
    "acquisition_cost_weight",
    "add_team_player_rank",
    "blend_policy_model_score",
    "replacement_need_score",
    "score_recommendations",
    "select_recommendation_slots",
]
