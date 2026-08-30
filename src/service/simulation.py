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

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from src.service.simulation_constants import *
from src.service.simulation_types import *

"""==== 지역 함수 ===="""

def _normalize_players(players: pd.DataFrame) -> pd.DataFrame:
    """계산에 사용할 ID와 숫자 컬럼을 한 가지 표현으로 정규화한다."""
    missing = REQUIRED_COLUMNS - set(players.columns)

    # 필수 컬럼이 없으면 이후 계산의 의미가 성립하지 않으므로 즉시 중단한다.
    if missing:
        raise ValueError(f"선수 데이터 필수 컬럼 누락: {sorted(missing)}")

    # 빈 로스터는 가중평균과 이탈 시뮬레이션의 기준이 될 수 없다.
    if players.empty:
        raise ValueError("팀 선수 데이터가 비어 있습니다.")

    normalized = players.copy()

    # 선수 식별자는 문자열 변환 전에 결측 여부를 확인해 'nan' ID 생성을 막는다.
    if normalized["player_id"].isna().any():
        raise ValueError("player_id에 결측치가 있습니다.")
    normalized["player_id"] = normalized["player_id"].astype(str)

    # 실제로 존재하는 선택 점수 컬럼만 숫자 정규화 대상에 포함한다.
    numeric_columns = [
        column
        for column in (*SCORE_COLUMNS, "g_ratio")
        if column in normalized.columns
    ]
    for column in numeric_columns:
        original = normalized[column]
        numeric = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & numeric.isna()
        infinite = numeric.notna() & ~np.isfinite(numeric)

        # 숫자로 해석할 수 없거나 무한한 값은 가중평균을 오염시키므로 차단한다.
        if invalid.any() or infinite.any():
            raise ValueError(f"{column}은 유한한 숫자 또는 결측치여야 합니다.")

        normalized[column] = numeric

    # 한 선수가 여러 행이면 이탈 대상을 유일하게 결정할 수 없다.
    if normalized["player_id"].duplicated().any():
        duplicated = normalized.loc[
            normalized["player_id"].duplicated(), "player_id"
        ].tolist()
        raise ValueError(f"팀 데이터에 player_id가 중복됩니다: {duplicated[:5]}")

    # 전체 전력은 모든 선수에게 필수이므로 결측 행을 허용하지 않는다.
    if normalized["overall_score"].isna().any():
        raise ValueError("overall_score에 결측치가 있습니다.")

    # 출전 비중은 가중치로 사용되므로 결측값과 음수를 허용하지 않는다.
    if normalized["g_ratio"].isna().any() or (normalized["g_ratio"] < 0).any():
        raise ValueError("g_ratio는 결측이 없는 0 이상의 값이어야 합니다.")

    # 모든 가중치가 0이면 가중평균을 계산할 수 없다.
    if float(normalized["g_ratio"].sum()) <= 0:
        raise ValueError("팀 전체 g_ratio 합은 0보다 커야 합니다.")

    return normalized


def _weighted_score(players: pd.DataFrame, column: str) -> float | None:
    """결측 점수를 제외하고 출전 비중 가중평균을 계산한다."""

    # 선택 점수 컬럼이 없는 데이터 계약은 해당 세부 전력을 제공하지 않는다.
    if column not in players.columns:
        return None

    # 점수가 존재하고 출전 비중이 양수인 선수만 가중평균 계산에 포함한다.
    valid = players[column].notna() & players["g_ratio"].notna() & (players["g_ratio"] > 0)

    # 유효한 점수와 양수 가중치 조합이 없으면 계산 결과를 제공하지 않는다.
    if not valid.any():
        return None

    return float(np.average(players.loc[valid, column], weights=players.loc[valid, "g_ratio"]))


# ── 대체 수준(replacement level) ─────────────────────────────────────
# [2026-08-30 추가] 예전에는 이탈 선수를 로스터에서 그냥 빼기만 했다. 그러면
# 팀 전력이 "남은 선수들의 가중평균"으로 다시 계산되면서, 그 선수가 소화하던
# 출전 시간을 남은 선수 평균이 공짜로 메운다고 가정하는 셈이 된다. 그래서
#   - 핵심 선수가 빠져도 승률이 거의 안 내려가고(46명 평균이라 지분이 작음)
#   - 평균 이하 선수가 빠지면 오히려 팀이 좋아지는
# 비현실적인 결과가 나왔다(실측: 최고 선수 이탈 -0.63%p, 약체 이탈 +0.33%p).
#
# 실제로는 빈 자리를 마이너/웨이버에서 데려온 대체 수준 선수가 메운다.
# 그 수준은 세이버메트릭스 관례상 "대체 수준 선수로만 채운 팀의 승률 .294"로
# 정의된다. 우리 승률 매핑(app/ui/winrate.py 의 실데이터 적합식
# win_rate = 0.1786 + 0.006704 * strength)을 역산하면 그 승률에 해당하는
# 전력 점수가 나온다: (0.294 - 0.1786) / 0.006704 = 17.2
REPLACEMENT_LEVEL_SCORE = 17.2

_SCORE_COLUMNS = ("overall_score", "off_score", "pit_score", "def_score")


def _replacement_filler(departing: pd.Series, template: pd.DataFrame) -> pd.DataFrame:
    """이탈한 선수의 출전 시간을 대체 수준 선수가 그대로 메운다고 본 가상 행.

    출전 비중(g_ratio)은 떠난 선수 것을 그대로 물려받고, 점수만 대체 수준으로
    낮춘다 — "그 자리는 비지 않고 누군가 뛴다"는 현실을 반영하는 것이 목적이다.
    """
    row = {col: np.nan for col in template.columns}
    row["player_id"] = f"__replacement__{departing['player_id']}"
    row["g_ratio"] = departing["g_ratio"]
    for col in _SCORE_COLUMNS:
        if col in template.columns and pd.notna(departing.get(col)):
            row[col] = REPLACEMENT_LEVEL_SCORE
    return pd.DataFrame([row])


def _calculate_validated_strength(players: pd.DataFrame) -> TeamStrength:
    """이미 정규화·검증된 선수 목록의 전력을 계산한다."""
    return TeamStrength(
        overall=float(_weighted_score(players, "overall_score")),
        offense=_weighted_score(players, "off_score"),
        pitching=_weighted_score(players, "pit_score"),
        defense=_weighted_score(players, "def_score"),
        player_count=len(players),
    )


def _as_probability(value: float, label: str) -> float:
    probability = float(value)

    # 모델의 비정상 출력이 화면과 후속 변화량 계산으로 전파되지 않게 한다.
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{label}은 0~1 사이의 유한한 확률이어야 합니다: {value}")
    return probability


def _as_rank(value: int, label: str) -> int:
    """순위 함수의 NaN·0·음수 결과가 화면까지 전파되지 않게 차단한다."""
    numeric = float(value)
    # 순위는 유한한 1 이상의 정수만 유효한 도메인 값이다.
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric < 1:
        raise ValueError(f"{label}은 1 이상의 정수여야 합니다: {value}")

    return int(numeric)


def _prepare_departure(
    team_players: pd.DataFrame,
    removed_player_id: str,
    predict_win_rate: WinRatePredictor,
    rank_predictor: RankPredictor | None,
) -> DepartureContext:
    """이탈 전후 상태를 한 번 계산해 후보 평가에서 재사용한다."""
    players = _normalize_players(team_players)
    normalized_id = str(removed_player_id)

    selected = players.loc[players["player_id"] == normalized_id]

    # 이탈 대상은 누락되거나 중복되지 않은 단일 행이어야 한다.
    if len(selected) != 1:
        raise ValueError(
            f"이탈 선수 player_id '{normalized_id}'를 정확히 1명 찾을 수 없습니다."
        )

    remaining = players.loc[players["player_id"] != normalized_id].copy()

    # 마지막 선수까지 제거하면 이탈 후 팀 전력을 정의할 수 없다.
    if remaining.empty:
        raise ValueError("이탈 후 팀에 남는 선수가 없어 시뮬레이션할 수 없습니다.")

    current_strength = _calculate_validated_strength(players)
    # 빈 자리를 대체 수준 선수가 메운 상태로 이탈 후 전력을 계산한다.
    # remaining 자체는 손대지 않는다 — 실제 대체 선수를 영입하는 경로가
    # 그 자리에 진짜 선수를 넣어야 하기 때문(가상 채움과 중복되면 안 됨).
    departure_roster = pd.concat(
        [remaining, _replacement_filler(selected.iloc[0], remaining)],
        ignore_index=True,
        sort=False,
    )
    departure_strength = _calculate_validated_strength(departure_roster)
    current_win_rate = _as_probability(
        predict_win_rate(current_strength), "현재 예상 승률"
    )
    departure_win_rate = _as_probability(
        predict_win_rate(departure_strength), "이탈 후 예상 승률"
    )

    # 순위 예측기가 주입되지 않은 경우 화면에서 순위 미제공 상태를 표시한다.
    rank_before = (
        None
        if rank_predictor is None
        else _as_rank(rank_predictor(current_strength), "현재 예상 순위")
    )

    # 동일한 규칙으로 이탈 후 순위도 선택적으로 계산한다.
    rank_after_departure = (
        None
        if rank_predictor is None
        else _as_rank(rank_predictor(departure_strength), "이탈 후 예상 순위")
    )
    return DepartureContext(
        removed_player_id=normalized_id,
        remaining_players=remaining,
        predict_win_rate=predict_win_rate,
        rank_predictor=rank_predictor,
        current_strength=current_strength,
        departure_strength=departure_strength,
        current_win_rate=current_win_rate,
        departure_win_rate=departure_win_rate,
        rank_before=rank_before,
        rank_after_departure=rank_after_departure,
    )


def _simulate_from_context(
    context: DepartureContext,
    replacement_player: pd.Series | Mapping[str, Any] | None,
    scenario: DepartureScenario,
) -> SimulationResult:
    """준비된 이탈 상태에 선택적 대체 선수를 적용한다."""
    replacement_id: str | None = None
    replacement_strength: TeamStrength | None = None
    replacement_win_rate: float | None = None
    rank_after = context.rank_after_departure

    # 대체 선수가 전달된 경우에만 영입 후 전력과 예측값을 추가로 계산한다.
    if replacement_player is not None:
        replacement = _normalize_players(pd.DataFrame([dict(replacement_player)]))
        replacement_id = str(replacement.iloc[0]["player_id"])

        # 현재 로스터에 남아 있는 선수는 외부 대체 후보가 될 수 없다.
        if replacement_id in set(context.remaining_players["player_id"]):
            raise ValueError(f"대체 선수 '{replacement_id}'가 이미 현재 팀에 있습니다.")

        replaced_team = pd.concat(
            [context.remaining_players, replacement], ignore_index=True, sort=False
        )
        replacement_strength = _calculate_validated_strength(replaced_team)
        replacement_win_rate = _as_probability(
            context.predict_win_rate(replacement_strength), "대체 후 예상 승률"
        )

        # 순위 예측기가 제공된 경우에만 대체 후 순위를 갱신한다.
        if context.rank_predictor is not None:
            rank_after = _as_rank(
                context.rank_predictor(replacement_strength), "대체 후 예상 순위"
            )

    impact = context.departure_win_rate - context.current_win_rate

    # 대체 선수가 없으면 비교 기준이 없으므로 대체 효과를 계산하지 않는다.
    replacement_effect = (
        None
        if replacement_win_rate is None
        else replacement_win_rate - context.departure_win_rate
    )

    # 대체 선수가 있을 때만 최초 전력 대비 최종 변화량을 계산한다.
    net_effect = (
        None
        if replacement_win_rate is None
        else replacement_win_rate - context.current_win_rate
    )
    meta = SCENARIO_META[scenario]

    return SimulationResult(
        removed_player_id=context.removed_player_id,
        replacement_player_id=replacement_id,
        scenario=scenario,
        scenario_label=meta["label"],
        effective_timing=meta["timing"],
        absence_scope=meta["absence_scope"],
        current_strength=context.current_strength,
        after_departure_strength=context.departure_strength,
        after_replacement_strength=replacement_strength,
        current_win_rate=context.current_win_rate,
        after_departure_win_rate=context.departure_win_rate,
        after_replacement_win_rate=replacement_win_rate,
        impact=impact,
        replacement_effect=replacement_effect,
        net_effect=net_effect,
        rank_before=context.rank_before,
        rank_after=rank_after,
    )


"""==== 전역 함수 ===="""

def calculate_team_strength(players: pd.DataFrame) -> TeamStrength:
    """선수 목록을 출전 비중으로 가중해 팀 전력을 계산한다."""
    return _calculate_validated_strength(_normalize_players(players))

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

    # 정의된 메타데이터가 없는 시나리오는 결과 설명을 구성할 수 없다.
    if scenario not in SCENARIO_META:
        raise ValueError(f"지원하지 않는 이탈 시나리오입니다: {scenario}")

    context = _prepare_departure(
        team_players,
        removed_player_id,
        predict_win_rate,
        rank_predictor,
    )
    return _simulate_from_context(context, replacement_player, scenario)


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

    문서 F6-3에 따라 net effect(내림차순), 예상 순위(오름차순),
    replacement effect와 코사인 유사도(내림차순) 순으로 정렬한다.
    """
    # 비교할 후보가 없으면 평가표와 추천 순위를 만들 수 없다.
    if candidates.empty:
        raise ValueError("평가할 대체 후보가 없습니다.")

    # 단건 시뮬레이션과 동일한 시나리오 계약을 후보 평가에도 적용한다.
    if scenario not in SCENARIO_META:
        raise ValueError(f"지원하지 않는 이탈 시나리오입니다: {scenario}")

    context = _prepare_departure(
        team_players,
        removed_player_id,
        predict_win_rate,
        rank_predictor,
    )

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for _, candidate in candidates.iterrows():
        try:
            result = _simulate_from_context(context, candidate, scenario)
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

    # 유효한 후보가 하나도 없으면 실패 원인을 모아 호출자에게 전달한다.
    if not rows:
        reasons = "; ".join(f"{item['player_id']}: {item['reason']}" for item in errors[:3])
        raise ValueError(f"모든 대체 후보의 시뮬레이션이 실패했습니다. {reasons}")

    evaluated = pd.DataFrame(rows)
    sort_columns: list[str] = []
    ascending: list[bool] = []

    sort_columns.append("net_effect")
    ascending.append(False)

    # net effect가 같을 때는 문서의 다음 기준인 예상 순위를 사용한다.
    if rank_predictor is not None:
        sort_columns.append("rank_after")
        ascending.append(True)
    sort_columns.append("replacement_effect")
    ascending.append(False)

    # 추천 모델이 유사도를 제공한 경우 동률 후보의 추가 정렬 기준으로 사용한다.
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
    "SCENARIO_META",
    "DepartureScenario",
    "RankPredictor",
    "SimulationResult",
    "TeamStrength",
    "WinRatePredictor",
    "calculate_team_strength",
    "evaluate_replacements",
    "simulate",
]
