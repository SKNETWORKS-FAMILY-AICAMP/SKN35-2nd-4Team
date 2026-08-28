"""시뮬레이션 계층이 외부와 주고받는 타입 정의."""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from src.service.simulation_constants import DepartureScenario


@dataclass(frozen=True)
class TeamStrength:
    """한 시점의 팀 전력 요약."""

    overall: float
    offense: float | None
    pitching: float | None
    defense: float | None
    player_count: int

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


@dataclass(frozen=True)
class SimulationResult:
    """화면과 API가 공통으로 사용하는 시뮬레이션 결과."""

    removed_player_id: str
    replacement_player_id: str | None
    scenario: DepartureScenario
    scenario_label: str
    effective_timing: str
    absence_scope: str
    season_progress: float
    application_ratio: float
    current_strength: TeamStrength
    after_departure_strength: TeamStrength
    after_replacement_strength: TeamStrength | None
    current_win_rate: float
    after_departure_win_rate: float
    after_replacement_win_rate: float | None
    effective_after_departure_win_rate: float
    effective_after_replacement_win_rate: float | None
    impact: float
    replacement_effect: float | None
    net_effect: float | None
    effective_impact: float
    effective_replacement_effect: float | None
    effective_net_effect: float | None
    rank_before: int | None
    rank_after: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# 예측 모델은 팀 전력을 받아 검증 가능한 단일 값을 반환한다.
WinRatePredictor = Callable[[TeamStrength], float]
RankPredictor = Callable[[TeamStrength], int]


@dataclass(frozen=True)
class DepartureContext:
    """후보가 달라도 변하지 않는 이탈 시뮬레이션의 공통 계산 결과."""

    removed_player_id: str
    remaining_players: pd.DataFrame
    predict_win_rate: WinRatePredictor
    rank_predictor: RankPredictor | None
    current_strength: TeamStrength
    departure_strength: TeamStrength
    current_win_rate: float
    departure_win_rate: float
    rank_before: int | None
    rank_after_departure: int | None


__all__ = [
    "DepartureContext",
    "RankPredictor",
    "SimulationResult",
    "TeamStrength",
    "WinRatePredictor",
]
