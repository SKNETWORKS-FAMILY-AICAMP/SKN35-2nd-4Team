"""과거 다음 시즌 성과·실제 이동·비용 효율을 학습하는 추천 정책 Ranker."""

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
    """선수 쌍의 미래 적합도와 영입 현실성을 함께 정렬한다."""

    def __init__(self, *, seed: int = 42) -> None:
        self.seed = seed
        self.model_: Any = None
        self.feature_names_: list[str] = []
        self.training_end_season_: int | None = None

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
        queries_per_season: int = 40,
    ) -> RecommendationPolicyRanker:
        ranked = add_team_player_rank(players)
        ranked["cost_efficiency"] = 1.0 - ranked["team_player_rank"].map(
            acquisition_cost_weight
        )
        matrices: list[pd.DataFrame] = []
        labels: list[np.ndarray] = []
        groups: list[int] = []
        seasons = set(ranked["season"].astype(int))
        for season in sorted(year for year in seasons if year <= training_end_season and year + 1 in seasons):
            current = ranked.loc[ranked["season"].eq(season)].copy()
            next_scores = ranked.loc[
                ranked["season"].eq(season + 1), ["player_id", "overall_score"]
            ].rename(columns={"overall_score": "actual_next_score"})
            current = current.merge(next_scores, on="player_id", how="inner")
            for _, target in self._queries(current, queries_per_season).iterrows():
                candidates = current.loc[
                    current["player_id"].ne(target["player_id"])
                    & current["team_last"].ne(target["team_last"])
                    & current["g_ratio"].ge(0.10)
                ].copy()
                position = target.get("position", target.get("primary_position"))
                position_col = "position" if "position" in candidates else "primary_position"
                if pd.notna(position) and position_col in candidates:
                    candidates = candidates.loc[candidates[position_col].eq(position)]
                else:
                    candidates = candidates.loc[candidates["role"].eq(target["role"])]
                if len(candidates) < 3:
                    continue
                gap = (candidates["actual_next_score"] - target["actual_next_score"]).abs()
                gap_rank = gap.rank(method="first")
                relevance = np.where(gap_rank <= 3, 3, np.where(gap_rank <= 10, 2, 0))
                relevance = np.where((gap <= 5.0) & (relevance == 0), 1, relevance)
                moved = candidates["y_path"].isin(["offseason_move", "trade"]).to_numpy()
                affordable = candidates["team_player_rank"].gt(12).to_numpy()
                relevance = np.clip(relevance + moved.astype(int) + affordable.astype(int), 0, 5)
                matrices.append(self._pair_matrix(target, candidates))
                labels.append(relevance.astype(float))
                groups.append(len(candidates))
        if not matrices:
            raise ValueError("정책 Ranker 학습 쌍이 없습니다.")

        from xgboost import XGBRanker

        X = pd.concat(matrices, ignore_index=True)
        y = np.concatenate(labels)
        self.model_ = XGBRanker(
            objective="rank:ndcg",
            eval_metric="ndcg@5",
            n_estimators=350,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=self.seed,
        ).fit(X, y, group=groups, verbose=False)
        self.feature_names_ = list(X.columns)
        self.training_end_season_ = training_end_season
        return self

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
        self.model_.save_model(path)
        path.with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "feature_names": self.feature_names_,
                    "training_end_season": self.training_end_season_,
                    "seed": self.seed,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> RecommendationPolicyRanker:
        from xgboost import XGBRanker

        path = Path(path)
        meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        obj = cls(seed=int(meta["seed"]))
        obj.model_ = XGBRanker()
        obj.model_.load_model(path)
        obj.feature_names_ = list(meta["feature_names"])
        obj.training_end_season_ = int(meta["training_end_season"])
        return obj


__all__ = ["RecommendationPolicyRanker"]
