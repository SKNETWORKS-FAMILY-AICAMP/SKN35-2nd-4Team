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

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.models.recommend_constants import *
from src.models.recommend_types import RecommendationConfig


class ReplacementRecommender:
    """동일 역할/포지션 선수 중 전력이 유사한 대체 후보를 찾는다."""

    def __init__(self, **config: Any) -> None:
        self.config = RecommendationConfig(**config)
        self.catalog_: pd.DataFrame | None = None
        self.feature_names_: list[str] = []
        self.filter_note_: str = ""

    def fit(self, players: pd.DataFrame) -> "ReplacementRecommender":
        """추천에 사용할 선수-시즌 카탈로그를 저장하고 검증한다."""
        missing = REQUIRED_COLUMNS - set(players.columns)

        # 필수 계약 컬럼이 없으면 후보 필터와 특징 계산을 수행할 수 없다.
        if missing:
            raise ValueError(f"추천 데이터 필수 컬럼 누락: {sorted(missing)}")

        # 빈 카탈로그에서는 추천 대상과 후보를 찾을 수 없다.
        if players.empty:
            raise ValueError("추천 데이터가 비어 있습니다.")

        # 선수·시즌 키가 중복되면 추천 대상을 단일 행으로 결정할 수 없다.
        if players.duplicated(["player_id", "season"]).any():
            raise ValueError("player_id + season이 중복된 행이 있습니다.")

        # 출전 비중은 후보 자격을 판정하는 기준이므로 음수를 허용하지 않는다.
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

        # 반환할 추천 수는 최소 한 명 이상이어야 한다.
        if n_recommendations < 1:
            raise ValueError("n_recommendations는 1 이상이어야 합니다.")

        target_rows = catalog.loc[
            (catalog["player_id"].astype(str) == str(player_id))
            & (catalog["season"] == season)
        ]

        # 추천 대상은 지정한 선수·시즌 조합에서 정확히 한 행이어야 한다.
        if len(target_rows) != 1:
            raise ValueError(
                f"player_id='{player_id}', season={season}인 선수를 정확히 1명 찾을 수 없습니다."
            )
        target = target_rows.iloc[0]

        candidates, matched_on = self._filter_candidates(catalog, target)

        # 필터를 통과한 후보가 없으면 유사도 모델을 실행할 수 없다.
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
        result["recommender"] = "knn"

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
        # 계산 결과에 없는 원본 컬럼만 뒤에 이어 붙여 핵심 컬럼 순서를 유지한다.
        remaining = [column for column in result.columns if column not in first]
        result = result[first + remaining].reset_index(drop=True)
        result.attrs["filter_note"] = self.filter_note_
        return result

    def _catalog(self) -> pd.DataFrame:

        # 학습 카탈로그가 준비되지 않은 상태의 추천 호출을 차단한다.
        if self.catalog_ is None:
            raise RuntimeError("fit()을 먼저 호출하세요.")

        return self.catalog_

    def _filter_candidates(
        self, catalog: pd.DataFrame, target: pd.Series
    ) -> tuple[pd.DataFrame, str]:
        pool = catalog.loc[catalog["season"] == target["season"]].copy()
        pool = pool.loc[pool["player_id"].astype(str) != str(target["player_id"])]

        # 설정에서 요청한 경우 현재 소속팀 선수는 외부 대체 후보에서 제외한다.
        if self.config.exclude_same_team:
            pool = pool.loc[pool["team_last"] != target["team_last"]]

        # 포지션 후보가 없을 때는 더 넓은 역할(B/P/TWO) 후보군으로 완화한다.
        if "position" in catalog.columns and pd.notna(target.get("position")):
            position_pool = pool.loc[pool["position"] == target["position"]]
        else:
            position_pool = pd.DataFrame()

        # 동일 포지션 후보가 있으면 역할보다 구체적인 포지션 조건을 우선한다.
        if not position_pool.empty:
            matched_pool, matched_on = position_pool, "position"
        else:
            matched_pool = pool.loc[pool["role"] == target["role"]]
            matched_on = "role"

        candidates = matched_pool.loc[matched_pool["g_ratio"] >= self.config.min_g_ratio]
        self.filter_note_ = "기본 출전 기준 적용"

        # 기본 기준 후보가 없을 때만 최소 출전 비중을 한 차례 완화한다.
        if candidates.empty and self.config.min_g_ratio > 0.05:
            # 후보가 0명인 경우에만 최소 출전 기준을 0.05까지 낮춘다.
            candidates = matched_pool.loc[matched_pool["g_ratio"] >= 0.05]
            self.filter_note_ = "후보 부족으로 최소 출전 비중을 0.05로 완화"
        return candidates, matched_on

    @staticmethod
    def _select_features(candidates: pd.DataFrame, target: pd.Series) -> list[str]:
        role = str(target["role"])
        preferred = COMMON_FEATURES + ROLE_FEATURES.get(role, [])

        # 후보와 대상 양쪽에서 사용할 수 있고 실제 값이 있는 피처만 선택한다.
        available = [
            column
            for column in preferred
            if column in candidates.columns
            and column in target.index
            and (candidates[column].notna().any() or pd.notna(target[column]))
        ]

        # 유효한 공통 피처가 없으면 코사인 유사도를 정의할 수 없다.
        if not available:
            raise ValueError("코사인 유사도를 계산할 전력 피처가 없습니다.")
        return available


class AutoencoderRecommender(ReplacementRecommender):
    """Autoencoder 잠재 벡터의 코사인 유사도로 대체 선수를 추천한다."""

    def __init__(
        self,
        *,
        latent_dim: int = 3,
        hidden_dim: int = 8,
        epochs: int = 300,
        learning_rate: float = 0.01,
        batch_size: int = 64,
        validation_fraction: float = 0.2,
        seed: int = 42,
        **config: Any,
    ) -> None:
        super().__init__(**config)

        # 신경망 차원과 학습 횟수는 모두 양의 정수 범위여야 한다.
        if latent_dim < 1 or hidden_dim < 1 or epochs < 1:
            raise ValueError("latent_dim, hidden_dim, epochs는 1 이상이어야 합니다.")
        # mini-batch 크기는 최소 한 행 이상이어야 한다.
        if batch_size < 1:
            raise ValueError("batch_size는 1 이상이어야 합니다.")
        # Train과 Validation이 모두 남도록 검증 비율은 0과 1 사이여야 한다.
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction은 0과 1 사이여야 합니다.")

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.validation_fraction = validation_fraction
        self.seed = seed
        self.imputer_: SimpleImputer | None = None
        self.scaler_: StandardScaler | None = None
        self.model_: Any = None
        self.reconstruction_loss_: float | None = None
        self.train_loss_: float | None = None
        self.validation_loss_: float | None = None
        self.best_epoch_: int | None = None
        self.training_history_: list[dict[str, float | int]] = []

    def fit(self, players: pd.DataFrame) -> "AutoencoderRecommender":
        """전체 선수 카탈로그에서 비지도 방식으로 전력 표현을 학습한다."""
        super().fit(players)
        catalog = self._catalog()
        preferred = list(dict.fromkeys(COMMON_FEATURES + sum(ROLE_FEATURES.values(), [])))

        # 카탈로그에 존재하고 실제 학습값이 있는 피처만 사용한다.
        self.feature_names_ = [
            column for column in preferred
            if column in catalog.columns and catalog[column].notna().any()
        ]

        # 학습 가능한 피처가 없으면 Autoencoder 입력 행렬을 만들 수 없다.
        if not self.feature_names_:
            raise ValueError("Autoencoder를 학습할 전력 피처가 없습니다.")

        # 최소 두 행은 있어야 Train과 Validation에 한 행 이상 배정할 수 있다.
        if len(catalog) < 2:
            raise ValueError("Autoencoder Train/Validation 분할에는 최소 2행이 필요합니다.")

        # 전체 선수-시즌 행을 고정 seed로 섞어 재현 가능한 검증 세트를 만든다.
        train_indices, validation_indices = train_test_split(
            catalog.index,
            test_size=self.validation_fraction,
            random_state=self.seed,
            shuffle=True,
        )
        train_matrix = catalog.loc[train_indices, self.feature_names_].apply(
            pd.to_numeric, errors="coerce"
        )
        validation_matrix = catalog.loc[
            validation_indices, self.feature_names_
        ].apply(pd.to_numeric, errors="coerce")
        # 전처리 통계는 Train에서만 학습하고 Validation에는 transform만 적용한다.
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler_ = StandardScaler()
        train_transformed = self.scaler_.fit_transform(
            self.imputer_.fit_transform(train_matrix)
        )
        validation_transformed = self.scaler_.transform(
            self.imputer_.transform(validation_matrix)
        )

        # torch는 Autoencoder를 실제로 사용할 때만 불러와 KNN 경로를 가볍게 유지한다.
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from src.models._torch_autoencoder_net import PlayerAutoencoderNet

        torch.manual_seed(self.seed)
        train_tensor = torch.as_tensor(train_transformed, dtype=torch.float32)
        validation_tensor = torch.as_tensor(validation_transformed, dtype=torch.float32)
        generator = torch.Generator().manual_seed(self.seed)
        train_loader = DataLoader(
            TensorDataset(train_tensor),
            batch_size=min(self.batch_size, len(train_tensor)),
            shuffle=True,
            generator=generator,
        )
        self.model_ = PlayerAutoencoderNet(
            n_features=train_tensor.shape[1],
            hidden_dim=self.hidden_dim,
            latent_dim=min(self.latent_dim, train_tensor.shape[1]),
        )
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.MSELoss()

        best_state: dict[str, Any] | None = None
        best_validation_loss = float("inf")
        self.training_history_ = []
        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            train_loss_sum = 0.0
            for (batch,) in train_loader:
                optimizer.zero_grad()
                reconstructed = self.model_(batch)
                batch_loss = loss_fn(reconstructed, batch)
                batch_loss.backward()
                optimizer.step()
                train_loss_sum += float(batch_loss.detach().cpu()) * len(batch)
            train_loss = train_loss_sum / len(train_tensor)

            self.model_.eval()
            with torch.no_grad():
                validation_loss = float(
                    loss_fn(self.model_(validation_tensor), validation_tensor).cpu()
                )
            self.training_history_.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                }
            )

            # 전체 epoch 중 validation loss가 가장 낮은 가중치를 저장한다.
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_state = {
                    name: value.detach().clone()
                    for name, value in self.model_.state_dict().items()
                }
                self.best_epoch_ = epoch
                self.train_loss_ = train_loss

        # 첫 epoch은 항상 무한대보다 개선되므로 최적 상태가 반드시 존재한다.
        if best_state is None:
            raise RuntimeError("Autoencoder 최적 가중치를 저장하지 못했습니다.")
        self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.validation_loss_ = best_validation_loss
        self.reconstruction_loss_ = best_validation_loss
        return self

    def recommend(
        self,
        player_id: str,
        season: int,
        *,
        n_recommendations: int = 3,
    ) -> pd.DataFrame:
        """잠재 공간에서 대상 선수와 가장 가까운 후보를 반환한다."""

        # 모델과 전처리기가 모두 준비된 이후에만 잠재 벡터를 계산할 수 있다.
        if self.model_ is None or self.imputer_ is None or self.scaler_ is None:
            raise RuntimeError("fit()을 먼저 호출하세요.")

        # 반환할 추천 수는 최소 한 명 이상이어야 한다.
        if n_recommendations < 1:
            raise ValueError("n_recommendations는 1 이상이어야 합니다.")

        catalog = self._catalog()
        target_rows = catalog.loc[
            catalog["player_id"].astype(str).eq(str(player_id))
            & catalog["season"].eq(season)
        ]

        # 추천 대상은 지정한 선수·시즌 조합에서 정확히 한 행이어야 한다.
        if len(target_rows) != 1:
            raise ValueError(
                f"player_id='{player_id}', season={season}인 선수를 정확히 1명 찾을 수 없습니다."
            )
        target = target_rows.iloc[0]
        candidates, matched_on = self._filter_candidates(catalog, target)

        # 잠재 공간에서 비교할 후보가 없으면 추천 결과를 만들 수 없다.
        if candidates.empty:
            raise ValueError("Autoencoder 추천 조건을 만족하는 대체 후보가 없습니다.")

        matrix = pd.concat([target.to_frame().T, candidates], ignore_index=True)
        matrix = matrix[self.feature_names_].apply(pd.to_numeric, errors="coerce")
        transformed = self.scaler_.transform(self.imputer_.transform(matrix))

        import torch

        with torch.no_grad():
            latent = self.model_.encode(torch.as_tensor(transformed, dtype=torch.float32))
            latent = latent.cpu().numpy()
        target_vector = latent[0]
        candidate_vectors = latent[1:]
        denominator = np.linalg.norm(candidate_vectors, axis=1) * np.linalg.norm(target_vector)
        similarity = np.divide(
            candidate_vectors @ target_vector,
            denominator,
            out=np.zeros(len(candidate_vectors), dtype=float),
            where=denominator > 0,
        )
        order = np.argsort(-similarity, kind="stable")[: min(n_recommendations, len(candidates))]

        result = candidates.iloc[order].copy()
        result["similarity"] = np.clip(similarity[order], -1.0, 1.0)
        result["distance"] = 1.0 - result["similarity"]
        result["rank"] = np.arange(1, len(result) + 1)
        result["matched_on"] = matched_on
        result["recommender"] = "autoencoder"
        result.attrs["filter_note"] = self.filter_note_
        result.attrs["reconstruction_loss"] = self.reconstruction_loss_
        return result.reset_index(drop=True)

    @classmethod
    def load_artifact(
        cls,
        path: str | Path,
        players: pd.DataFrame,
    ) -> "AutoencoderRecommender":
        """저장된 Autoencoder와 전처리 통계를 복원해 현재 카탈로그에 연결한다."""
        import torch
        from src.models._torch_autoencoder_net import PlayerAutoencoderNet

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(
            latent_dim=int(checkpoint["latent_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            epochs=1,
            batch_size=int(checkpoint.get("batch_size", 64)),
            validation_fraction=float(checkpoint.get("validation_fraction", 0.2)),
        )

        # 추천 대상 카탈로그는 최신 features_v1을 사용하고 학습 가중치만 복원한다.
        ReplacementRecommender.fit(obj, players)
        obj.feature_names_ = list(checkpoint["feature_names"])
        missing = set(obj.feature_names_) - set(players.columns)

        # 저장 모델이 요구하는 피처가 없으면 동일한 입력 공간을 복원할 수 없다.
        if missing:
            raise ValueError(f"Autoencoder 저장 피처가 현재 데이터에 없습니다: {sorted(missing)}")

        matrix = players[obj.feature_names_].apply(pd.to_numeric, errors="coerce")
        obj.imputer_ = SimpleImputer(
            strategy="median", keep_empty_features=True
        ).fit(matrix)
        obj.imputer_.statistics_ = np.asarray(checkpoint["imputer_statistics"])
        imputed = obj.imputer_.transform(matrix)
        obj.scaler_ = StandardScaler().fit(imputed)
        obj.scaler_.mean_ = np.asarray(checkpoint["scaler_mean"])
        obj.scaler_.scale_ = np.asarray(checkpoint["scaler_scale"])
        obj.scaler_.var_ = obj.scaler_.scale_ ** 2

        obj.model_ = PlayerAutoencoderNet(
            n_features=len(obj.feature_names_),
            hidden_dim=obj.hidden_dim,
            latent_dim=min(obj.latent_dim, len(obj.feature_names_)),
        )
        obj.model_.load_state_dict(checkpoint["state_dict"])
        obj.model_.eval()
        obj.reconstruction_loss_ = float(checkpoint["reconstruction_loss"])
        train_loss = checkpoint.get("train_loss")
        validation_loss = checkpoint.get("validation_loss")
        # 이전 형식의 아티팩트에는 학습 손실이 없으므로 선택적으로 복원한다.
        obj.train_loss_ = None if train_loss is None else float(train_loss)
        # 이전 형식의 아티팩트에는 검증 손실이 없으므로 선택적으로 복원한다.
        obj.validation_loss_ = (
            None if validation_loss is None else float(validation_loss)
        )
        obj.best_epoch_ = checkpoint.get("best_epoch")
        obj.training_history_ = list(checkpoint.get("training_history", []))
        return obj


def load_knn_artifact(path: str | Path, players: pd.DataFrame) -> ReplacementRecommender:
    """저장된 KNN 설정을 복원하고 최신 선수 카탈로그를 연결한다."""
    with Path(path).open("rb") as stream:
        recommender = pickle.load(stream)

    # 다른 객체가 저장된 파일을 추천기로 잘못 사용하는 상황을 차단한다.
    if not isinstance(recommender, ReplacementRecommender):
        raise TypeError("저장 파일이 ReplacementRecommender가 아닙니다.")

    return recommender.fit(players)


def _clean_feature_rows(players: pd.DataFrame) -> pd.DataFrame:
    """복구 가능한 결측은 보정하고 계약 자체가 깨진 행만 제외한다."""
    out = players.copy()
    original_count = len(out)

    # 문자열이나 inf가 들어와도 계산 계층에는 유한한 숫자만 전달한다.
    for column in ["overall_score", "g_ratio"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )

    # 출전 비중은 동일 시즌·역할 중앙값으로 보정하고, 그룹 전체가 결측이면
    # 전체 중앙값을 사용한다. 전력 점수는 임의 보정하지 않고 해당 행을 제외한다.
    missing_ratio = int(out["g_ratio"].isna().sum())
    group_median = out.groupby(["season", "role"])["g_ratio"].transform("median")
    out["g_ratio"] = out["g_ratio"].fillna(group_median).fillna(out["g_ratio"].median())
    out.loc[out["g_ratio"] < 0, "g_ratio"] = np.nan
    out["g_ratio"] = out["g_ratio"].clip(upper=1.05)

    required_rows = ["player_id", "season", "team_last", "role", "overall_score", "g_ratio"]
    out = out.dropna(subset=required_rows).copy()

    # 선수·시즌 중복은 추천 대상과 다음 시즌 평가의 키를 모호하게 만든다.
    if out.duplicated(["player_id", "season"]).any():
        raise ValueError("features_v1에 player_id + season 중복 행이 있습니다.")

    out = out.reset_index(drop=True)
    out.attrs["data_quality"] = {
        "original_rows": original_count,
        "usable_rows": len(out),
        "excluded_rows": original_count - len(out),
        "imputed_g_ratio": missing_ratio,
    }
    return out


def adapt_features_v1(players: pd.DataFrame) -> pd.DataFrame:
    """B의 features_v1을 E 추천·시뮬레이션 계약으로 변환한다.

    문서 계약 컬럼이 이미 있으면 복사본을 반환하고, 현재 B 산출물의
    ``playerID/yearID`` 스키마도 동일 인터페이스로 변환한다.
    """
    # 이미 추천 계약을 만족하면 레거시 컬럼 변환 없이 품질 정리만 수행한다.
    if REQUIRED_COLUMNS <= set(players.columns):
        return _clean_feature_rows(players)

    legacy_required = {
        "playerID", "yearID", "bat_teamID", "pit_teamID",
        "batting_strength_before_pt", "pitching_strength_before_pt",
        "playing_time_ratio", "playing_time_ratio_pit", "is_batter", "is_pitcher",
    }
    missing = legacy_required - set(players.columns)
    # 레거시 스키마 변환에 필요한 원본 컬럼이 없으면 변환을 중단한다.
    if missing:
        raise ValueError(f"features_v1 변환에 필요한 컬럼 누락: {sorted(missing)}")

    out = pd.DataFrame(index=players.index)
    out["player_id"] = players["playerID"].astype(str)
    out["season"] = players["yearID"].astype(int)
    out["team_last"] = players["bat_teamID"].fillna(players["pit_teamID"])
    out["team_last"] = out["team_last"].replace(LAHMAN_TEAM_TO_UI)

    batter = players["is_batter"].fillna(False).astype(bool)
    pitcher = players["is_pitcher"].fillna(False).astype(bool)
    out["role"] = np.select([batter & pitcher, pitcher], ["TWO", "P"], default="B")
    out["g_ratio"] = pd.concat(
        [players["playing_time_ratio"], players["playing_time_ratio_pit"]], axis=1
    ).max(axis=1, skipna=True)
    out["off_score"] = players["batting_strength_before_pt"]
    out["pit_score"] = players["pitching_strength_before_pt"]
    out["def_score"] = np.nan
    out["overall_score"] = pd.concat(
        [out["off_score"], out["pit_score"]], axis=1
    ).mean(axis=1, skipna=True)

    # 추천 품질을 높일 수 있는 현재 B 피처는 계약명으로 함께 전달한다.
    # OPS가 있을 때만 시즌별 표준화 타격 피처를 추가한다.
    if "OPS" in players:
        out["ops_z"] = players.groupby("yearID")["OPS"].transform(
            lambda s: (s - s.mean()) / s.std()
        )

    # ERA가 있을 때만 시즌별 표준화 투수 피처를 추가한다.
    if "ERA" in players:
        out["era_z"] = players.groupby("yearID")["ERA"].transform(
            lambda s: (s - s.mean()) / s.std()
        )

    # WHIP가 있을 때만 시즌별 표준화 투수 피처를 추가한다.
    if "WHIP" in players:
        out["whip_z"] = players.groupby("yearID")["WHIP"].transform(
            lambda s: (s - s.mean()) / s.std()
        )

    return _clean_feature_rows(out)


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


def recommend_replacements_autoencoder(
    players: pd.DataFrame,
    player_id: str,
    season: int,
    *,
    n_recommendations: int = 3,
    min_g_ratio: float = 0.10,
    exclude_same_team: bool = True,
    epochs: int = 300,
    batch_size: int = 64,
    validation_fraction: float = 0.2,
) -> pd.DataFrame:
    """Autoencoder 학습부터 추천까지 한 번에 수행하는 편의 함수."""
    recommender = AutoencoderRecommender(
        min_g_ratio=min_g_ratio,
        exclude_same_team=exclude_same_team,
        epochs=epochs,
        batch_size=batch_size,
        validation_fraction=validation_fraction,
    ).fit(players)
    return recommender.recommend(
        player_id,
        season,
        n_recommendations=n_recommendations,
    )


def precision_at_k_next_strength(
    players: pd.DataFrame,
    recommender: ReplacementRecommender,
    *,
    season: int = 2024,
    k: int = 3,
    max_queries: int | None = 200,
) -> dict[str, float | int]:
    """다음 시즌 실제 전력을 정답으로 사용해 추천 P@K를 계산한다.

    질의 선수의 t+1 전력과 가장 가까운 동일 역할·타 팀 후보 K명을 정답으로
    정의한다. 추천기는 t시점 피처만 보고 후보를 고르므로 시간 누수를 피한다.
    """
    # Precision@K의 K는 최소 한 개의 추천을 요구한다.
    if k < 1:
        raise ValueError("k는 1 이상이어야 합니다.")

    current = players.loc[players["season"] == season].copy()
    next_scores = players.loc[
        players["season"] == season + 1, ["player_id", "overall_score"]
    ].rename(columns={"overall_score": "next_overall_score"})
    current = current.merge(next_scores, on="player_id", how="inner")

    # 다음 시즌 실제 전력이 없으면 추천 적중 여부를 정의할 수 없다.
    if current.empty:
        raise ValueError(f"{season + 1}시즌 실제 전력이 없어 P@{k}를 계산할 수 없습니다.")

    # 실행시간과 재현성을 위해 전력 상위 질의를 고정적으로 사용한다.
    queries = current.sort_values("overall_score", ascending=False)

    # 평가 상한이 지정된 경우에만 상위 질의 수를 제한한다.
    if max_queries is not None:
        queries = queries.head(max_queries)

    scores: list[float] = []
    skipped = 0
    for _, target in queries.iterrows():
        relevant_pool = current.loc[
            current["role"].eq(target["role"])
            & current["team_last"].ne(target["team_last"])
            & current["player_id"].ne(target["player_id"])
            & current["g_ratio"].ge(recommender.config.min_g_ratio)
        ].copy()

        # 정답 후보가 K명보다 적은 질의는 Precision@K 비교가 불가능하다.
        if len(relevant_pool) < k:
            skipped += 1
            continue

        relevant_pool["next_score_gap"] = (
            relevant_pool["next_overall_score"] - target["next_overall_score"]
        ).abs()
        relevant = set(relevant_pool.nsmallest(k, "next_score_gap")["player_id"].astype(str))
        try:
            predicted = recommender.recommend(
                str(target["player_id"]), season, n_recommendations=k
            )
        except ValueError:
            skipped += 1
            continue
        recommended = set(predicted["player_id"].astype(str))
        scores.append(len(relevant & recommended) / k)

    # 모든 질의가 제외됐다면 평균 정밀도를 계산할 수 없다.
    if not scores:
        raise ValueError(f"평가 가능한 질의가 없어 P@{k}를 계산할 수 없습니다.")
    return {
        f"precision_at_{k}": float(np.mean(scores)),
        "evaluated_queries": len(scores),
        "skipped_queries": skipped,
        "evaluation_season": season,
    }


def save_recommendation_models(
    knn: ReplacementRecommender,
    autoencoder: AutoencoderRecommender,
    knn_metrics: dict[str, float | int],
    autoencoder_metrics: dict[str, float | int],
) -> tuple[Path, Path]:
    """E 담당 ML/DL 모델과 비교 지표를 담당자별 레지스트리에 저장한다."""

    # 두 모델이 모두 학습된 상태에서만 일관된 아티팩트 묶음을 저장한다.
    if knn.catalog_ is None or autoencoder.model_ is None:
        raise RuntimeError("두 추천 모델을 모두 fit()한 뒤 저장하세요.")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

    knn_path = MODEL_DIR / "recommend_knn.pkl"
    with knn_path.open("wb") as stream:
        pickle.dump(knn, stream)

    import torch

    autoencoder_path = MODEL_DIR / "recommend_autoencoder.pt"
    torch.save(
        {
            "state_dict": autoencoder.model_.state_dict(),
            "feature_names": autoencoder.feature_names_,
            "latent_dim": autoencoder.latent_dim,
            "hidden_dim": autoencoder.hidden_dim,
            "batch_size": autoencoder.batch_size,
            "validation_fraction": autoencoder.validation_fraction,
            "imputer_statistics": autoencoder.imputer_.statistics_,
            "scaler_mean": autoencoder.scaler_.mean_,
            "scaler_scale": autoencoder.scaler_.scale_,
            "reconstruction_loss": autoencoder.reconstruction_loss_,
            "train_loss": autoencoder.train_loss_,
            "validation_loss": autoencoder.validation_loss_,
            "best_epoch": autoencoder.best_epoch_,
            "training_history": autoencoder.training_history_,
        },
        autoencoder_path,
    )

    knn_features = [
        column
        for column in dict.fromkeys(COMMON_FEATURES + sum(ROLE_FEATURES.values(), []))
        if column in knn.catalog_.columns and knn.catalog_[column].notna().any()
    ]
    entries = [
        {
            "name": "recommend_knn", "task": "recommend", "kind": "ml", "owner": "E",
            "format": "pickle", "path": "models/recommend_knn.pkl",
            "features": knn_features, "n_features": len(knn_features),
            "metrics": knn_metrics,
            "note": "동일 시즌·역할 후보 KNN 코사인 추천",
        },
        {
            "name": "recommend_autoencoder", "task": "recommend", "kind": "dl", "owner": "E",
            "format": "torch", "path": "models/recommend_autoencoder.pt",
            "features": autoencoder.feature_names_, "n_features": len(autoencoder.feature_names_),
            "metrics": {**autoencoder_metrics, "reconstruction_loss": autoencoder.reconstruction_loss_},
            "note": "Autoencoder 잠재 벡터 코사인 추천",
        },
    ]
    for entry in entries:
        entry.update(
            classes=[],
            params={},
            saved_at=datetime.now().isoformat(timespec="seconds"),
        )
        path = REGISTRY_DIR / f"{entry['name']}.json"
        path.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
    return knn_path, autoencoder_path


__all__ = [
    "RecommendationConfig",
    "AutoencoderRecommender",
    "ReplacementRecommender",
    "adapt_features_v1",
    "load_knn_artifact",
    "recommend_replacements",
    "recommend_replacements_autoencoder",
    "precision_at_k_next_strength",
    "save_recommendation_models",
]
