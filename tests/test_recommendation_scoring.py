import pandas as pd

from src.service.recommendation_scoring import (
    acquisition_cost_weight,
    add_team_player_rank,
    blend_policy_model_score,
    score_recommendations,
    select_recommendation_slots,
)


def test_acquisition_cost_bands() -> None:
    assert [acquisition_cost_weight(rank) for rank in (1, 3, 4, 7, 10, 13, 16, 30)] == [
        1.0, 1.0, 0.85, 0.70, 0.55, 0.40, 0.25, 0.25
    ]


def test_team_player_rank_is_not_recommendation_or_team_rank() -> None:
    players = pd.DataFrame(
        {
            "player_id": ["a", "b", "c"],
            "season": [2026] * 3,
            "team_last": ["T"] * 3,
            "overall_score": [90.0, 70.0, 80.0],
            "recommendation_rank": [3, 1, 2],
            "pred_rank": [1, 1, 1],
        }
    )
    ranked = add_team_player_rank(players).set_index("player_id")
    assert ranked["team_player_rank"].to_dict() == {"a": 1, "b": 3, "c": 2}


def test_low_cost_only_helps_after_fit_and_risk_components() -> None:
    players = pd.DataFrame(
        {
            "player_id": ["expensive", "cheap"],
            "season": [2026, 2026],
            "team_last": ["A", "B"],
            "overall_score": [90.0, 50.0],
        }
    )
    candidates = pd.DataFrame(
        {
            "player_id": ["expensive", "cheap"],
            "season": [2026, 2026],
            "team_last": ["A", "B"],
            "overall_score": [90.0, 50.0],
            "predicted_next_overall_score": [90.0, 50.0],
            "similarity": [0.95, 0.20],
            "age": [27, 27],
            "departure_risk": [0.4, 0.4],
            "replacement_effect": [0.05, 0.01],
            "net_effect": [0.04, 0.01],
        }
    )
    reasons = {
        "expensive": {"stable_performance_move": 1.0},
        "cheap": {"stable_performance_move": 1.0},
    }
    scored = score_recommendations(
        candidates,
        players,
        reasons,
        reason_injury_scores={"expensive": 0.1, "cheap": 0.1},
    )

    assert scored.iloc[0]["player_id"] == "expensive"
    assert {
            "team_player_rank",
            "acquisition_cost_weight",
            "cost_efficiency",
            "market_availability",
            "harm_risk",
            "final_recommend_score",
    }.issubset(scored.columns)


def test_policy_model_is_only_blended_by_configured_weight() -> None:
    scored = pd.DataFrame(
        {
            "player_id": ["a", "b"],
            "final_recommend_score": [0.8, 0.7],
            "similarity": [0.8, 0.7],
            "growth_potential": [0.5, 0.5],
            "harm_risk": [0.1, 0.1],
            "health_stability": [0.9, 0.9],
            "replacement_effect": [0.03, 0.03],
            "net_effect": [0.03, 0.03],
        }
    )
    result = blend_policy_model_score(scored, [0.0, 1.0], model_weight=0.05)

    by_id = result.set_index("player_id")
    assert by_id.loc["a", "final_recommend_score"] == 0.76
    assert by_id.loc["b", "final_recommend_score"] == 0.715


def test_moderate_decline_is_included_in_harm_risk() -> None:
    players = pd.DataFrame(
        {"player_id": ["p"], "season": [2026], "team_last": ["A"], "overall_score": [50.0]}
    )
    candidates = players.assign(
        predicted_next_overall_score=50.0,
        similarity=0.8,
        age=28,
        departure_risk=0.8,
        replacement_effect=0.02,
        net_effect=0.02,
    )
    scored = score_recommendations(
        candidates,
        players,
        {"p": {"moderate_performance_decline": 0.5}},
        reason_injury_scores={"p": 0.25},
    )

    assert scored.iloc[0]["harm_risk"] == 0.4
    assert scored.iloc[0]["health_stability"] == 0.75


def test_slot_cascade_keeps_five_and_labels_fallback_source() -> None:
    scored = pd.DataFrame(
        {
            "player_id": [f"p{i}" for i in range(6)],
            "departure_risk": [0.2] * 6,
            "market_availability": [0.1] * 6,
            "harm_risk": [0.1] * 6,
            "final_recommend_score": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            "similarity": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            "reason_profile_similarity": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )
    selected = select_recommendation_slots(scored)

    assert len(selected) == 5
    assert "최적 영입 후보" not in set(selected["recommend_tier_label"])
    assert selected.iloc[-1]["recommend_tier_label"] == "참고용 (시장 이탈 신호 없음)"
