"""과거 다음 시즌 대체 적합도를 학습하는 LightGBM LambdaRank 모델."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.service.recommendation_scoring import (
    acquisition_cost_weight,
    add_team_player_rank,
)

PAIR_BASE_FEATURES = [
    "overall_score",
    "predicted_next_overall_score",
    "g_ratio",
    "age",
    "exp",
    "off_score",
    "ops_z",
    "pit_score",
    "era_z",
    "whip_z",
]


class RecommendationPolicyRanker:
    """Autoencoder 후보를 선수 쌍의 미래 대체 적합도로 재정렬한다."""

    def __init__(self, *, seed: int = 42) -> None:
        self.seed = seed
        self.model_: Any = None
        self.feature_names_: list[str] = []
        self.training_end_season_: int | None = None
        self.best_iteration_: int | None = None
        self.candidate_pool_size_: int | None = None
        self.retrieval_strategy_: str | None = None

    @staticmethod
    def _pair_matrix(target: pd.Series, candidates: pd.DataFrame) -> pd.DataFrame:
        data: dict[str, pd.Series | float] = {}
        for feature in PAIR_BASE_FEATURES:
            if feature not in candidates or feature not in target.index:
                continue
            values = pd.to_numeric(candidates[feature], errors="coerce").reset_index(drop=True)
            target_value = pd.to_numeric(pd.Series([target[feature]]), errors="coerce").iloc[0]
            data[f"candidate_{feature}"] = values
            data[f"target_{feature}"] = float(target_value)
            data[f"diff_{feature}"] = values - target_value
            data[f"abs_diff_{feature}"] = (values - target_value).abs()
        data["candidate_team_player_rank"] = pd.to_numeric(
            candidates["team_player_rank"], errors="coerce"
        ).reset_index(drop=True)
        data["candidate_cost_efficiency"] = pd.to_numeric(
            candidates["cost_efficiency"], errors="coerce"
        ).reset_index(drop=True)
        data["candidate_retrieval_similarity"] = pd.to_numeric(
            candidates.get("retrieval_similarity", candidates.get("similarity", 0.0)),
            errors="coerce",
        ).reset_index(drop=True)
        return pd.DataFrame(data).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def _queries(current: pd.DataFrame, limit: int) -> pd.DataFrame:
        ordered = current.sort_values(["overall_score", "player_id"], ascending=[False, True])
        groups = [group.reset_index(drop=True) for _, group in ordered.groupby("role", sort=True)]
        rows: list[pd.Series] = []
        offset = 0
        while len(rows) < min(limit, len(ordered)):
            added = False
            for group in groups:
                if offset < len(group):
                    rows.append(group.iloc[offset])
                    added = True
                    if len(rows) == limit:
                        break
            if not added:
                break
            offset += 1
        return pd.DataFrame(rows)

    def fit(
        self,
        players: pd.DataFrame,
        *,
        training_end_season: int = 2023,
        queries_per_season: int = 100,
        candidate_recommenders: list[tuple[Any, int]] | None = None,
        projection_pool_size: int = 20,
    ) -> RecommendationPolicyRanker:
        ranked = add_team_player_rank(players)
        ranked["cost_efficiency"] = 1.0 - ranked["team_player_rank"].map(
            acquisition_cost_weight
        )
        seasons = set(ranked["season"].astype(int))
        train_seasons = sorted(
            year
            for year in seasons
            if year < training_end_season and year + 1 in seasons
        )
        validation_seasons = [training_end_season]
        X_train, y_train, train_groups = self._build_dataset(
            ranked,
            train_seasons,
            queries_per_season,
            candidate_recommenders,
            projection_pool_size,
        )
        X_validation, y_validation, validation_groups = self._build_dataset(
            ranked,
            validation_seasons,
            queries_per_season,
            candidate_recommenders,
            projection_pool_size,
        )

        from lightgbm import early_stopping, log_evaluation

        validation_model = self._new_model(n_estimators=2_000)
        validation_model.fit(
            X_train,
            y_train,
            group=train_groups,
            eval_set=[(X_validation, y_validation)],
            eval_group=[validation_groups],
            eval_at=[3, 5],
            callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
        )
        self.best_iteration_ = int(validation_model.best_iteration_ or 500)

        X_final = pd.concat([X_train, X_validation], ignore_index=True)
        y_final = np.concatenate([y_train, y_validation])
        final_groups = train_groups + validation_groups
        self.model_ = self._new_model(n_estimators=self.best_iteration_)
        self.model_.fit(X_final, y_final, group=final_groups)
        self.feature_names_ = list(X_final.columns)
        self.training_end_season_ = training_end_season
        configured_pool = sum(size for _, size in (candidate_recommenders or []))
        self.candidate_pool_size_ = configured_pool + projection_pool_size
        self.retrieval_strategy_ = "autoencoder+knn+d_projection"
        return self

    def _build_dataset(
        self,
        ranked: pd.DataFrame,
        seasons: list[int],
        queries_per_season: int,
        candidate_recommenders: list[tuple[Any, int]] | None,
        projection_pool_size: int,
    ) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
        matrices: list[pd.DataFrame] = []
        labels: list[np.ndarray] = []
        groups: list[int] = []
        for season in seasons:
            current = ranked.loc[ranked["season"].eq(season)].copy()
            next_scores = ranked.loc[
                ranked["season"].eq(season + 1), ["player_id", "overall_score"]
            ].rename(columns={"overall_score": "actual_next_score"})
            current = current.merge(next_scores, on="player_id", how="inner")
            for _, target in self._queries(current, queries_per_season).iterrows():
                candidates = self._candidate_pool(
                    current,
                    target,
                    season,
                    candidate_recommenders,
                    projection_pool_size,
                )
                if len(candidates) < 3:
                    continue
                gap = (candidates["actual_next_score"] - target["actual_next_score"]).abs()
                gap_rank = gap.rank(method="first")
                relevance = np.where(gap_rank <= 3, 3, np.where(gap_rank <= 10, 2, 0))
                relevance = np.where((gap <= 5.0) & (relevance == 0), 1, relevance)
                # Ranker 정답은 미래 대체 적합도에 집중한다. 이동 가능성과 비용
                # 효율은 B/C 모델 출력과 최종 비즈니스 규칙에서 별도로 반영한다.
                matrices.append(self._pair_matrix(target, candidates))
                labels.append(relevance.astype(float))
                groups.append(len(candidates))
        if not matrices:
            raise ValueError("정책 Ranker 학습 쌍이 없습니다.")
        return (
            pd.concat(matrices, ignore_index=True),
            np.concatenate(labels),
            groups,
        )

    @staticmethod
    def _candidate_pool(
        current: pd.DataFrame,
        target: pd.Series,
        season: int,
        candidate_recommenders: list[tuple[Any, int]] | None,
        projection_pool_size: int,
    ) -> pd.DataFrame:
        candidates = current.loc[
            current["player_id"].ne(target["player_id"])
            & current["team_last"].ne(target["team_last"])
            & current["g_ratio"].ge(0.10)
        ].copy()
        position = target.get("position", target.get("primary_position"))
        position_col = "position" if "position" in candidates else "primary_position"
        if pd.notna(position) and position_col in candidates:
            eligible = candidates.loc[candidates[position_col].eq(position)].copy()
        else:
            eligible = candidates.loc[candidates["role"].eq(target["role"])].copy()

        eligible["retrieval_similarity"] = 0.0
        eligible_ids = eligible["player_id"].astype(str)
        ids: set[str] = set()
        for recommender, pool_size in candidate_recommenders or []:
            try:
                generated = recommender.recommend(
                    str(target["player_id"]),
                    season,
                    n_recommendations=pool_size,
                )
                generated_ids = generated["player_id"].astype(str)
                ids.update(generated_ids)
                similarity_map = dict(
                    zip(
                        generated_ids,
                        pd.to_numeric(generated["similarity"], errors="coerce").fillna(0.0),
                        strict=True,
                    )
                )
                generated_similarity = eligible_ids.map(similarity_map).fillna(0.0)
                eligible["retrieval_similarity"] = np.maximum(
                    eligible["retrieval_similarity"], generated_similarity
                )
            except (RuntimeError, ValueError):
                continue
        if "predicted_next_overall_score" in eligible:
            projection_gap = (
                pd.to_numeric(eligible["predicted_next_overall_score"], errors="coerce")
                - float(target["predicted_next_overall_score"])
            ).abs()
            ids.update(
                eligible.loc[projection_gap.nsmallest(projection_pool_size).index, "player_id"].astype(str)
            )
            projection_similarity = 1.0 / (1.0 + projection_gap)
            eligible["retrieval_similarity"] = np.maximum(
                eligible["retrieval_similarity"], projection_similarity.fillna(0.0)
            )
        if ids:
            return eligible.loc[eligible["player_id"].astype(str).isin(ids)].copy()
        return eligible

    def _new_model(self, *, n_estimators: int) -> Any:
        from lightgbm import LGBMRanker

        return LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            label_gain=[0, 1, 3, 7, 15, 31],
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=0.1,
            random_state=self.seed,
            verbosity=-1,
        )

    def predict(self, target: pd.Series, candidates: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("fit() 또는 load()를 먼저 호출하세요.")
        matrix = self._pair_matrix(target, candidates).reindex(
            columns=self.feature_names_, fill_value=0.0
        )
        raw = np.asarray(self.model_.predict(matrix), dtype=float)
        return pd.Series(raw).rank(method="average", pct=True).to_numpy()

    def save(self, path: str | Path) -> Path:
        if self.model_ is None:
            raise RuntimeError("fit()을 먼저 호출하세요.")
        path = Path(path)
        booster = getattr(self.model_, "booster_", self.model_)
        # LightGBM 네이티브 파일 API는 Windows에서 한글·공백이 포함된 경로를
        # 열지 못할 수 있다. Python이 경로를 처리하고 모델 문자열만 넘긴다.
        path.write_text(booster.model_to_string(), encoding="utf-8")
        path.with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "feature_names": self.feature_names_,
                    "training_end_season": self.training_end_season_,
                    "seed": self.seed,
                    "model_type": "lightgbm_lambdarank",
                    "best_iteration": self.best_iteration_,
                    "candidate_pool_size": self.candidate_pool_size_,
                    "retrieval_strategy": self.retrieval_strategy_,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> RecommendationPolicyRanker:
        from lightgbm import Booster

        path = Path(path)
        meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        obj = cls(seed=int(meta["seed"]))
        # model_file은 LightGBM의 네이티브 fopen을 사용해 Windows의 한글 경로에서
        # 실패할 수 있으므로 Python으로 읽은 모델 문자열을 전달한다.
        obj.model_ = Booster(model_str=path.read_text(encoding="utf-8"))
        obj.feature_names_ = list(meta["feature_names"])
        obj.training_end_season_ = int(meta["training_end_season"])
        obj.best_iteration_ = int(meta.get("best_iteration") or 0) or None
        obj.candidate_pool_size_ = int(meta.get("candidate_pool_size") or 0) or None
        obj.retrieval_strategy_ = meta.get("retrieval_strategy")
        return obj


__all__ = ["RecommendationPolicyRanker"]
