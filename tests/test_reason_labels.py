"""C 담당 이탈 연관 요인 라벨의 우선순위 회귀 테스트."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models.reason import ReasonThresholds, assign_reason_labels


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
        self.assertEqual(missing["primary_reason"], "unknown")

    def test_non_departed_row_has_no_reason_label(self) -> None:
        result = _label(y_departed=0.0)
        self.assertTrue(pd.isna(result["primary_reason"]))
        self.assertEqual(result["reason_tags"], tuple())


if __name__ == "__main__":
    unittest.main()
