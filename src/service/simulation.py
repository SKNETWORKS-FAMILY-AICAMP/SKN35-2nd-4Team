"""선수 이탈/대체 What-if 시뮬레이션 인터페이스.

이 모듈은 모델이나 Streamlit에 의존하지 않는 순수 함수로 유지한다.
승률 모델 담당자는 ``WinRatePredictor`` 규약에 맞는 함수를 전달하고,
화면 담당자는 ``SimulationResult.to_dict()`` 결과만 사용하면 된다.

기본 계산 규칙
--------------
* 선수 기여도: ``overall_score * g_ratio``
* 팀 전력: 출전 비중(``g_ratio``)을 가중치로 사용한 선수 전력 평균
* 선수 이탈: 대상 선수를 제외한 뒤 팀 전력을 다시 계산
* 대체 투입: 이탈 선수를 제외하고 대체 선수를 추가한 뒤 다시 계산

팀 전력에서 선수 점수를 직접 빼지 않는 것이 핵심이다.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"player_id", "overall_score", "g_ratio"}

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
    current_strength: TeamStrength
    after_departure_strength: TeamStrength
    after_replacement_strength: TeamStrength | None
    current_win_rate: float
    after_departure_win_rate: float
    after_replacement_win_rate: float | None
    impact: float
    replacement_effect: float | None
    net_effect: float | None
    rank_before: int | None
    rank_after: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# A 담당 승률 모델과의 연결 규약. 0~1 확률 하나를 반환해야 한다.
WinRatePredictor = Callable[[TeamStrength], float]
RankPredictor = Callable[[TeamStrength], int]


def _validate_players(players: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(players.columns)

    if missing:
        raise ValueError(f"선수 데이터 필수 컬럼 누락: {sorted(missing)}")

    if players.empty:
        raise ValueError("팀 선수 데이터가 비어 있습니다.")

    if players["player_id"].duplicated().any():
        duplicated = players.loc[players["player_id"].duplicated(), "player_id"].tolist()
        raise ValueError(f"팀 데이터에 player_id가 중복됩니다: {duplicated[:5]}")

    if players["overall_score"].isna().any():
        raise ValueError("overall_score에 결측치가 있습니다.")

    if players["g_ratio"].isna().any() or (players["g_ratio"] < 0).any():
        raise ValueError("g_ratio는 결측이 없는 0 이상의 값이어야 합니다.")

    if float(players["g_ratio"].sum()) <= 0:
        raise ValueError("팀 전체 g_ratio 합은 0보다 커야 합니다.")


def _weighted_score(players: pd.DataFrame, column: str) -> float | None:
    """결측 점수를 제외하고 출전 비중 가중평균을 계산한다."""
    if column not in players.columns:
        return None

    valid = players[column].notna() & players["g_ratio"].notna() & (players["g_ratio"] > 0)

    if not valid.any():
        return None

    return float(np.average(players.loc[valid, column], weights=players.loc[valid, "g_ratio"]))


def calculate_team_strength(players: pd.DataFrame) -> TeamStrength:
    """선수 목록을 출전 비중으로 가중해 팀 전력을 계산한다."""
    _validate_players(players)
    return TeamStrength(
        overall=_weighted_score(players, "overall_score"),  # type: ignore[arg-type]
        offense=_weighted_score(players, "off_score"),
        pitching=_weighted_score(players, "pit_score"),
        defense=_weighted_score(players, "def_score"),
        player_count=len(players),
    )


def _as_probability(value: float, label: str) -> float:
    probability = float(value)
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{label}은 0~1 사이의 유한한 확률이어야 합니다: {value}")
    return probability


def _as_rank(value: int, label: str) -> int:
    """순위 함수의 NaN·0·음수 결과가 화면까지 전파되지 않게 차단한다."""
    numeric = float(value)
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric < 1:
        raise ValueError(f"{label}은 1 이상의 정수여야 합니다: {value}")
    return int(numeric)


def _one_player(players: pd.DataFrame, player_id: str, label: str) -> pd.DataFrame:
    selected = players.loc[players["player_id"] == player_id]
    if len(selected) != 1:
        raise ValueError(f"{label} player_id '{player_id}'를 정확히 1명 찾을 수 없습니다.")
    return selected.copy()


def simulate(
    team_players: pd.DataFrame,
    removed_player_id: str,
    predict_win_rate: WinRatePredictor,
    *,
    replacement_player: pd.Series | Mapping[str, Any] | None = None,
    rank_predictor: RankPredictor | None = None,
    scenario: DepartureScenario = "trade",
) -> SimulationResult:
    """선수 이탈 및 선택적 대체 투입 결과를 계산한다.

    Parameters
    ----------
    team_players:
        동일 팀·동일 시즌의 선수 데이터. 필수 컬럼은 ``player_id``,
        ``overall_score``, ``g_ratio``이며, 세부 전력 컬럼은 선택 사항이다.
    removed_player_id:
        현재 팀에서 제외할 선수 ID.
    predict_win_rate:
        ``TeamStrength``를 받아 0~1 예상 승률을 반환하는 함수.
    replacement_player:
        외부 후보 한 명의 행. 없으면 이탈 시나리오까지만 계산한다.

    Returns
    -------
    SimulationResult
        변화량은 비율 단위다. 예를 들어 ``-0.055``는 ``-5.5%p``이다.
    """
    if scenario not in SCENARIO_META:
        raise ValueError(f"지원하지 않는 이탈 시나리오입니다: {scenario}")

    _validate_players(team_players)
    _one_player(team_players, removed_player_id, "이탈 선수")

    remaining = team_players.loc[team_players["player_id"] != removed_player_id].copy()
    if remaining.empty:
        raise ValueError("이탈 후 팀에 남는 선수가 없어 시뮬레이션할 수 없습니다.")

    current_strength = calculate_team_strength(team_players)
    departure_strength = calculate_team_strength(remaining)

    current_wr = _as_probability(predict_win_rate(current_strength), "현재 예상 승률")
    departure_wr = _as_probability(predict_win_rate(departure_strength), "이탈 후 예상 승률")

    replacement_id: str | None = None
    replacement_strength: TeamStrength | None = None
    replacement_wr: float | None = None

    if replacement_player is not None:
        replacement = pd.DataFrame([dict(replacement_player)])
        _validate_players(replacement)
        replacement_id = str(replacement.iloc[0]["player_id"])

        if replacement_id in set(remaining["player_id"].astype(str)):
            raise ValueError(f"대체 선수 '{replacement_id}'가 이미 현재 팀에 있습니다.")

        replaced_team = pd.concat([remaining, replacement], ignore_index=True, sort=False)
        replacement_strength = calculate_team_strength(replaced_team)
        replacement_wr = _as_probability(
            predict_win_rate(replacement_strength), "대체 후 예상 승률"
        )

    impact = departure_wr - current_wr
    replacement_effect = None if replacement_wr is None else replacement_wr - departure_wr
    net_effect = None if replacement_wr is None else replacement_wr - current_wr

    # 순위표 계산은 팀 전체 상태를 아는 외부 모델/A 파트에서 주입한다.
    # 순위표가 없는 mock 단계에서는 두 값을 None으로 둔다.
    rank_before = (
        None
        if rank_predictor is None
        else _as_rank(rank_predictor(current_strength), "현재 예상 순위")
    )
    final_strength = replacement_strength or departure_strength
    rank_after = (
        None
        if rank_predictor is None
        else _as_rank(rank_predictor(final_strength), "대체 후 예상 순위")
    )

    return SimulationResult(
        removed_player_id=str(removed_player_id),
        replacement_player_id=replacement_id,
        scenario=scenario,
        scenario_label=SCENARIO_META[scenario]["label"],
        effective_timing=SCENARIO_META[scenario]["timing"],
        absence_scope=SCENARIO_META[scenario]["absence_scope"],
        current_strength=current_strength,
        after_departure_strength=departure_strength,
        after_replacement_strength=replacement_strength,
        current_win_rate=current_wr,
        after_departure_win_rate=departure_wr,
        after_replacement_win_rate=replacement_wr,
        impact=impact,
        replacement_effect=replacement_effect,
        net_effect=net_effect,
        rank_before=rank_before,
        rank_after=rank_after,
    )


def evaluate_replacements(
    team_players: pd.DataFrame,
    removed_player_id: str,
    candidates: pd.DataFrame,
    predict_win_rate: WinRatePredictor,
    *,
    rank_predictor: RankPredictor | None = None,
    scenario: DepartureScenario = "trade",
) -> pd.DataFrame:
    """추천 후보별 시뮬레이션 결과를 의사결정 우선순위로 재정렬한다.

    정렬 기준은 대체 후 순위(오름차순), net_effect와 replacement_effect
    (내림차순), 코사인 유사도(내림차순)다.
    """
    if candidates.empty:
        raise ValueError("평가할 대체 후보가 없습니다.")

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for _, candidate in candidates.iterrows():
        try:
            result = simulate(
                team_players,
                removed_player_id,
                predict_win_rate,
                replacement_player=candidate,
                rank_predictor=rank_predictor,
                scenario=scenario,
            )
        except (ValueError, TypeError, FloatingPointError) as exc:
            # 한 후보의 결측·범위 오류 때문에 전체 추천 결과를 버리지 않는다.
            errors.append(
                {"player_id": str(candidate.get("player_id", "?")), "reason": str(exc)}
            )
            continue
        row = candidate.to_dict()
        row.update(
            after_replacement_win_rate=result.after_replacement_win_rate,
            replacement_effect=result.replacement_effect,
            net_effect=result.net_effect,
            rank_after=result.rank_after,
        )
        rows.append(row)

    if not rows:
        reasons = "; ".join(f"{item['player_id']}: {item['reason']}" for item in errors[:3])
        raise ValueError(f"모든 대체 후보의 시뮬레이션이 실패했습니다. {reasons}")

    evaluated = pd.DataFrame(rows)
    sort_columns: list[str] = []
    ascending: list[bool] = []
    if rank_predictor is not None:
        sort_columns.append("rank_after")
        ascending.append(True)
    sort_columns.extend(["net_effect", "replacement_effect"])
    ascending.extend([False, False])
    if "similarity" in evaluated.columns:
        sort_columns.append("similarity")
        ascending.append(False)

    evaluated = evaluated.sort_values(
        sort_columns,
        ascending=ascending,
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)
    evaluated["recommendation_rank"] = np.arange(1, len(evaluated) + 1)
    evaluated.attrs["evaluation_errors"] = errors
    evaluated.attrs["requested_candidates"] = len(candidates)
    return evaluated


__all__ = [
    "SimulationResult",
    "DepartureScenario",
    "SCENARIO_META",
    "TeamStrength",
    "WinRatePredictor",
    "RankPredictor",
    "calculate_team_strength",
    "evaluate_replacements",
    "simulate",
]

if __name__ == "__main__":
    def __demo_win_rate_predictor(team: TeamStrength) -> float:
        """데모용 승률 예측기. 실제 모델 연결 후 교체할 함수다."""
        return max(0.0, min(1.0, team.overall / 100.0))

    team_players = pd.DataFrame(
        [
            {
                "player_id": "star_hitter",
                "overall_score": 92.0,
                "off_score": 95.0,
                "pit_score": None,
                "def_score": 78.0,
                "g_ratio": 0.95,
            },
            {
                "player_id": "regular_hitter",
                "overall_score": 76.0,
                "off_score": 79.0,
                "pit_score": None,
                "def_score": 72.0,
                "g_ratio": 0.85,
            },
            {
                "player_id": "starting_pitcher",
                "overall_score": 81.0,
                "off_score": None,
                "pit_score": 88.0,
                "def_score": 65.0,
                "g_ratio": 0.75,
            },
            {
                "player_id": "relief_pitcher",
                "overall_score": 70.0,
                "off_score": None,
                "pit_score": 74.0,
                "def_score": 60.0,
                "g_ratio": 0.55,
            },
        ]
    )

    # 다른 팀 또는 FA 시장에서 데려온다고 가정한 대체 선수
    replacement_player = {
        "player_id": "replacement_hitter",
        "overall_score": 84.0,
        "off_score": 87.0,
        "pit_score": None,
        "def_score": 75.0,
        "g_ratio": 0.90,
    }

    result = simulate(
        team_players=team_players,
        removed_player_id="star_hitter",
        predict_win_rate=__demo_win_rate_predictor,
        replacement_player=replacement_player,
        # 실제 구현에서는 전체 팀 순위표를 조회하는 A 담당 함수를 주입한다.
        rank_predictor=lambda team: 3 if team.overall >= 78 else 6,
    )

    print("=== 선수 이탈·대체 시뮬레이션 예제 ===")
    print(f"시나리오: {result.scenario_label} · {result.effective_timing} · {result.absence_scope}")
    print(f"이탈 선수: {result.removed_player_id}")
    print(f"대체 선수: {result.replacement_player_id}")
    print(f"현재 예상 승률:       {result.current_win_rate:.1%}")
    print(f"이탈 후 예상 승률:    {result.after_departure_win_rate:.1%}")
    print(f"대체 후 예상 승률:    {result.after_replacement_win_rate:.1%}")
    print(f"이탈 영향:            {result.impact:+.1%}p")
    print(f"대체 효과:            {result.replacement_effect:+.1%}p")
    print(f"최종 변화:            {result.net_effect:+.1%}p")
    print(f"현재 순위:            {result.rank_before}위")
    print(f"대체 후 순위:         {result.rank_after}위")
