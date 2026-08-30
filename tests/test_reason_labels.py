"""C 담당 이탈 연관 요인 라벨의 우선순위 회귀 테스트."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models.reason import (
    ReasonThresholds,
    assign_observed_departure_events,
    assign_reason_labels,
    merge_transaction_evidence,
)


def _row(**overrides) -> dict:
    row = {
        "player_id": "player-1",
        "season": 2024,
        "age": 27.0,
        "exp": 3.0,
        "g_chg": 0.1,
        "overall_score": 50.0,
        "overall_score_delta": 1.0,
        "y_departed": 1.0,
        "y_path": "offseason_move",
        "y_fa_release": pd.NA,
        "reason_injury_score": 0.0,
        "injury_score_source": "observed_fallback",
        "injury_record_matched": False,
        "had_injury": 0.0,
    }
    row.update(overrides)
    return row


def _label(**overrides) -> pd.Series:
    frame = pd.DataFrame([_row(**overrides)])
    # 실제 훈련 데이터에서 g_chg 하위 분위수는 음수다. 테스트에서는 임계값
    # 적합 자체가 아니라 라벨 우선순위만 검증하므로 대표값을 명시한다.
    thresholds = ReasonThresholds(g_change=-0.1)
    return assign_reason_labels(frame, thresholds).iloc[0]


class ReasonLabelPriorityTest(unittest.TestCase):
    def test_early_career_is_used_only_after_strong_reasons_are_absent(self) -> None:
        result = _label(exp=1.0, overall_score_delta=8.0)
        self.assertEqual(result["primary_reason"], "early_career_move")
        self.assertEqual(result["reason_tags"], ("early_career_move",))

        injured = _label(
            exp=1.0,
            had_injury=1.0,
            injury_record_matched=True,
            reason_injury_score=1.0,
        )
        self.assertEqual(injured["primary_reason"], "injury_associated")

    def test_stable_performance_requires_observed_nonnegative_delta(self) -> None:
        stable = _label(exp=3.0, overall_score_delta=0.0)
        self.assertEqual(stable["primary_reason"], "stable_performance_move")

        missing = _label(exp=3.0, overall_score_delta=np.nan)
        self.assertEqual(missing["primary_reason"], "limited_history")
        self.assertEqual(missing["evidence_level"], "insufficient")

    def test_observed_events_are_separate_from_predictive_reason(self) -> None:
        rows = pd.DataFrame(
            [
                _row(
                    player_id="trade",
                    y_path="trade",
                    transaction_trade_confirmed=True,
                ),
                _row(
                    player_id="release",
                    y_fa_release="release_certain",
                ),
                _row(
                    player_id="fa",
                    y_fa_release="fa_est",
                ),
            ]
        )
        events = assign_observed_departure_events(rows).set_index("player_id")
        self.assertEqual(events.loc["trade", "departure_event_type"], "transaction_trade")
        self.assertEqual(events.loc["trade", "departure_event_evidence"], "confirmed_event")
        self.assertEqual(
            events.loc["release", "departure_event_type"],
            "roster_release_waiver",
        )
        self.assertEqual(events.loc["fa", "departure_event_type"], "free_agent_market")

        # 실제 거래 사건은 사후 관측값이고, 원인 모델의 정답에는 들어가지 않는다.
        reason = _label(y_path="trade", overall_score_delta=-3.0)
        self.assertEqual(reason["primary_reason"], "moderate_performance_decline")

    def test_moderate_decline_and_unresolved_injury_are_separate(self) -> None:
        moderate = _label(overall_score_delta=-3.0)
        self.assertEqual(
            moderate["primary_reason"],
            "moderate_performance_decline",
        )

        unresolved_row = _row(
            overall_score_delta=np.nan,
            had_injury=1.0,
            unresolved_stints=1.0,
        )
        unresolved_event = assign_observed_departure_events(
            pd.DataFrame([unresolved_row])
        ).iloc[0]
        self.assertEqual(
            unresolved_event["departure_event_type"],
            "injury_roster_move",
        )
        unresolved_reason = _label(**unresolved_row)
        self.assertEqual(unresolved_reason["primary_reason"], "limited_history")

    def test_transaction_window_maps_spring_to_previous_season(self) -> None:
        players = pd.DataFrame(
            [
                {"player_id": "p1", "season": 2024},
                {"player_id": "p1", "season": 2025},
            ]
        )
        transactions = pd.DataFrame(
            [
                {
                    "mlbam_id": 100,
                    "date": "2025-02-10",
                    "type_desc": "Declared Free Agency",
                    "description": "Player elected free agency.",
                },
                {
                    "mlbam_id": 100,
                    "date": "2025-07-10",
                    "type_desc": "Optioned",
                    "description": "Player optioned to Triple-A.",
                },
            ]
        )
        crosswalk = pd.DataFrame([{"player_id": "p1", "mlbam_id": 100}])

        result = merge_transaction_evidence(players, transactions, crosswalk)
        season_2024 = result.loc[result["season"].eq(2024)].iloc[0]
        season_2025 = result.loc[result["season"].eq(2025)].iloc[0]
        self.assertTrue(season_2024["transaction_fa_confirmed"])
        self.assertFalse(season_2025["transaction_option_confirmed"])

    def test_non_departed_row_has_no_reason_label(self) -> None:
        result = _label(y_departed=0.0)
        self.assertTrue(pd.isna(result["primary_reason"]))
        self.assertEqual(result["reason_tags"], tuple())


if __name__ == "__main__":
    unittest.main()
