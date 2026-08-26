"""KNN 기반 대체 선수 추천.

추천 순서
---------
1. 같은 시즌 선수만 사용한다.
2. ``position`` 컬럼이 있으면 동일 포지션, 없으면 동일 ``role``로 제한한다.
3. 최소 출전 비중(``g_ratio``)을 충족한 선수만 남긴다.
4. 필요하면 현재 팀 소속 선수를 제외한다.
5. 표준화된 전력 지표의 코사인 유사도가 높은 순으로 반환한다.

``features_v1``에 실제 데이터가 생기기 전에도 ``contract.make_mock()`` 결과로
동작한다. 추천은 정답 라벨을 학습하는 분류가 아니라 선수 카탈로그를 검색하는
방식이므로, 데이터 누수를 막기 위해 항상 대상 선수와 같은 시즌끼리 비교한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REQUIRED_COLUMNS = {
    "player_id",
    "season",
    "team_last",
    "role",
    "g_ratio",
    "overall_score",
}

COMMON_FEATURES = ["overall_score", "def_score", "g_ratio", "age", "exp"]
ROLE_FEATURES = {
    "B": ["off_score", "ops_z"],
    "P": ["pit_score", "era_z", "whip_z"],
    "TWO": ["off_score", "pit_score", "ops_z", "era_z", "whip_z"],
}


@dataclass(frozen=True)
class RecommendationConfig:
    """추천 후보 필터와 이웃 탐색 설정."""

    min_g_ratio: float = 0.10
    exclude_same_team: bool = True
    metric: str = "cosine"

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_g_ratio <= 1.05:
            raise ValueError("min_g_ratio는 0~1.05 사이여야 합니다.")
        if self.metric != "cosine":
            raise ValueError("현재 추천기는 cosine 거리만 지원합니다.")


class ReplacementRecommender:
    """동일 역할/포지션 선수 중 전력이 유사한 대체 후보를 찾는다."""

    def __init__(self, **config: Any) -> None:
        self.config = RecommendationConfig(**config)
        self.catalog_: pd.DataFrame | None = None
        self.feature_names_: list[str] = []

    def fit(self, players: pd.DataFrame) -> "ReplacementRecommender":
        """추천에 사용할 선수-시즌 카탈로그를 저장하고 검증한다."""
        missing = REQUIRED_COLUMNS - set(players.columns)
        if missing:
            raise ValueError(f"추천 데이터 필수 컬럼 누락: {sorted(missing)}")
        if players.empty:
            raise ValueError("추천 데이터가 비어 있습니다.")
        if players.duplicated(["player_id", "season"]).any():
            raise ValueError("player_id + season이 중복된 행이 있습니다.")
        if players["g_ratio"].dropna().lt(0).any():
            raise ValueError("g_ratio는 0 이상이어야 합니다.")

        self.catalog_ = players.copy()
        return self

    def recommend(
        self,
        player_id: str,
        season: int,
        *,
        n_recommendations: int = 3,
    ) -> pd.DataFrame:
        """대상 선수와 유사한 대체 후보를 코사인 유사도 순으로 반환한다.

        반환값에는 원본 식별·전력 컬럼과 다음 컬럼이 추가된다.

        * ``similarity``: 0~1 코사인 유사도
        * ``distance``: 코사인 거리
        * ``rank``: 추천 순위
        * ``matched_on``: ``position`` 또는 ``role``
        """
        catalog = self._catalog()
        if n_recommendations < 1:
            raise ValueError("n_recommendations는 1 이상이어야 합니다.")

        target_rows = catalog.loc[
            (catalog["player_id"].astype(str) == str(player_id))
            & (catalog["season"] == season)
        ]
        if len(target_rows) != 1:
            raise ValueError(
                f"player_id='{player_id}', season={season}인 선수를 정확히 1명 찾을 수 없습니다."
            )
        target = target_rows.iloc[0]

        candidates, matched_on = self._filter_candidates(catalog, target)
        if candidates.empty:
            raise ValueError("동일 역할/포지션과 최소 출전 기준을 만족하는 대체 후보가 없습니다.")

        features = self._select_features(candidates, target)
        self.feature_names_ = features

        # 후보와 질의 선수를 함께 변환해야 같은 결측치 대체값/스케일을 사용한다.
        matrix = pd.concat([target.to_frame().T, candidates], ignore_index=True)[features]
        matrix = matrix.apply(pd.to_numeric, errors="coerce")
        pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        transformed = pipeline.fit_transform(matrix)

        n_neighbors = min(n_recommendations, len(candidates))
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric=self.config.metric)
        knn.fit(transformed[1:])
        distances, indices = knn.kneighbors(transformed[[0]])

        result = candidates.iloc[indices[0]].copy()
        result["distance"] = distances[0]
        result["similarity"] = np.clip(1.0 - distances[0], 0.0, 1.0)
        result["rank"] = np.arange(1, len(result) + 1)
        result["matched_on"] = matched_on

        first = [
            "rank",
            "player_id",
            "season",
            "team_last",
            "role",
            "matched_on",
            "similarity",
            "distance",
        ]
        remaining = [column for column in result.columns if column not in first]
        return result[first + remaining].reset_index(drop=True)

    def _catalog(self) -> pd.DataFrame:
        if self.catalog_ is None:
            raise RuntimeError("fit()을 먼저 호출하세요.")
        return self.catalog_

    def _filter_candidates(
        self, catalog: pd.DataFrame, target: pd.Series
    ) -> tuple[pd.DataFrame, str]:
        candidates = catalog.loc[catalog["season"] == target["season"]].copy()
        candidates = candidates.loc[
            candidates["player_id"].astype(str) != str(target["player_id"])
        ]
        candidates = candidates.loc[candidates["g_ratio"] >= self.config.min_g_ratio]

        # 실제 포지션 컬럼이 도착하면 role보다 더 구체적인 position을 우선한다.
        if "position" in catalog.columns and pd.notna(target.get("position")):
            candidates = candidates.loc[candidates["position"] == target["position"]]
            matched_on = "position"
        else:
            candidates = candidates.loc[candidates["role"] == target["role"]]
            matched_on = "role"

        if self.config.exclude_same_team:
            candidates = candidates.loc[candidates["team_last"] != target["team_last"]]
        return candidates, matched_on

    @staticmethod
    def _select_features(candidates: pd.DataFrame, target: pd.Series) -> list[str]:
        role = str(target["role"])
        preferred = COMMON_FEATURES + ROLE_FEATURES.get(role, [])
        available = [
            column
            for column in preferred
            if column in candidates.columns
            and column in target.index
            and (candidates[column].notna().any() or pd.notna(target[column]))
        ]
        if not available:
            raise ValueError("코사인 유사도를 계산할 전력 피처가 없습니다.")
        return available


def recommend_replacements(
    players: pd.DataFrame,
    player_id: str,
    season: int,
    *,
    n_recommendations: int = 3,
    min_g_ratio: float = 0.10,
    exclude_same_team: bool = True,
) -> pd.DataFrame:
    """한 번의 호출로 추천 결과를 얻는 편의 함수."""
    recommender = ReplacementRecommender(
        min_g_ratio=min_g_ratio,
        exclude_same_team=exclude_same_team,
    ).fit(players)
    return recommender.recommend(
        player_id,
        season,
        n_recommendations=n_recommendations,
    )


__all__ = [
    "RecommendationConfig",
    "ReplacementRecommender",
    "recommend_replacements",
]
