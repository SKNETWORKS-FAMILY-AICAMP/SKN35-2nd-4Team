"""시뮬레이션에서 공통으로 사용하는 데이터 계약과 시나리오 설정."""

from typing import Literal

REQUIRED_COLUMNS = {"player_id", "overall_score", "g_ratio"}
SCORE_COLUMNS = ("overall_score", "off_score", "pit_score", "def_score")

# Rev.4 F6-2에 정의된 E 담당 유형별 시나리오만 지원한다.
DepartureScenario = Literal["trade", "fa", "release"]
SCENARIO_META: dict[DepartureScenario, dict[str, str]] = {
    "trade": {
        "label": "트레이드",
        "timing": "시즌 중",
        "absence_scope": "시즌 전체",
    },
    "fa": {
        "label": "FA",
        "timing": "오프시즌",
        "absence_scope": "다음 시즌",
    },
    "release": {
        "label": "방출",
        "timing": "즉시",
        "absence_scope": "잔여 시즌",
    },
}


__all__ = [
    "REQUIRED_COLUMNS",
    "SCENARIO_META",
    "SCORE_COLUMNS",
    "DepartureScenario",
]
