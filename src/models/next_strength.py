"""D의 저장된 MLP로 선수별 다음 시즌 전력을 예측한다."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.models.strength_ts import LAG_FEATURES, add_lag_features


def load_next_strength_model(path: str | Path) -> Any:
    """학습된 D 전력 모델을 불러오고 입력 계약을 확인한다."""
    path = Path(path)
    if path.suffix.lower() in {".json", ".ubj"}:
        from xgboost import XGBRegressor

        model = XGBRegressor()
        model.load_model(path)
    else:
        model = joblib.load(path)
    feature_names = list(getattr(model, "feature_names_in_", []))

    # 현재 D 계약에 없는 피처를 요구하는 다른 모델 파일은 연결하지 않는다.
    if feature_names and not set(feature_names) <= set(LAG_FEATURES):
        raise ValueError("다음 시즌 전력 모델의 입력 피처 계약이 현재 코드와 다릅니다.")
    return model


def predict_next_season_strength(
    players: pd.DataFrame,
    model: Any,
) -> pd.DataFrame:
    """features_v1 전체 이력에서 최신 시즌 선수의 t+1 전력을 만든다."""
    required = {"player_id", "season", "age", "exp", "team_wr"}
    missing = required - set(players.columns)

    # D가 확정한 실전 피처 계약이 없으면 임의 값으로 모델 입력을 만들지 않는다.
    if missing:
        raise ValueError(f"다음 시즌 전력 예측 피처 누락: {sorted(missing)}")

    enriched = players.copy()
    enriched["player_id"] = enriched["player_id"].astype(str)
    enriched = enriched.sort_values(["player_id", "season"]).copy()

    # update된 lag 함수는 부상 이력을 참조하지만 현재 확정 features_v1에는 아직 없다.
    if "had_injury" not in enriched.columns:
        enriched["had_injury"] = 0.0

    featured = add_lag_features(enriched)

    # 타자·투수 전용 지표가 원본에 없더라도 학습 때와 같은 0 보정을 적용한다.
    for column in LAG_FEATURES:
        if column not in featured.columns:
            featured[column] = 0.0

    latest_season = int(featured["season"].max())
    latest = featured.loc[featured["season"] == latest_season].copy()

    # 최신 시즌 선수가 없으면 다음 시즌 예측 대상도 정의할 수 없다.
    if latest.empty:
        raise ValueError("다음 시즌 전력을 예측할 최신 시즌 선수가 없습니다.")

    model_features = list(getattr(model, "feature_names_in_", LAG_FEATURES))
    predicted = np.asarray(
        model.predict(latest[model_features].fillna(0.0)), dtype=float
    )

    # 비정상 모델 출력은 승률 계산에 전달하지 않는다.
    if not np.isfinite(predicted).all():
        raise ValueError("다음 시즌 전력 모델이 유한하지 않은 값을 반환했습니다.")

    latest["predicted_next_overall_score"] = np.clip(predicted, 0.0, 100.0)
    latest["prediction_season"] = latest_season + 1
    return latest[
        ["player_id", "prediction_season", "predicted_next_overall_score"]
    ].reset_index(drop=True)


def apply_next_strength_projection(
    players: pd.DataFrame,
    projections: pd.DataFrame,
) -> pd.DataFrame:
    """시뮬레이션 입력의 현재 점수를 D의 t+1 예측 점수로 교체한다."""
    projected = players.copy()
    projected["player_id"] = projected["player_id"].astype(str)
    score_by_player = projections.set_index("player_id")[
        "predicted_next_overall_score"
    ]
    next_scores = projected["player_id"].map(score_by_player)

    # 일부 선수의 예측이 빠지면 현재·미래 점수가 섞이므로 계산을 중단한다.
    if next_scores.isna().any():
        missing = projected.loc[next_scores.isna(), "player_id"].tolist()
        raise ValueError(f"다음 시즌 전력 예측이 없는 선수가 있습니다: {missing[:5]}")

    projected["current_overall_score"] = projected["overall_score"]
    projected["overall_score"] = next_scores.astype(float)
    return projected


__all__ = [
    "apply_next_strength_projection",
    "load_next_strength_model",
    "predict_next_season_strength",
]
