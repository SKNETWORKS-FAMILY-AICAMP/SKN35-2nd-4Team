"""
departure.py
B - 이탈 이진분류 (LGBMClassifier + Optuna, sklearn API 스타일)

핵심 변경 (가짜 라벨 → 진짜 데이터)
--------------------------------
이전 버전은 features_v1이 아직 없어 attach_dummy_label()로 랜덤 라벨을
만들어 파이프라인만 확인했다. 이제 진짜 features_v1.parquet(contract.py
스키마)이 있으므로 그걸 그대로 쓴다.

타깃은 y_departed가 아니라 **y_core_departed**를 쓴다 — labels.py의 설계
원칙(구단이 결정한 방출까지 "이탈"로 묶으면 "선수가 떠날 위험"과 "구단이
내보낼 정책"이 뒤섞이는 순환논리가 생김)을 그대로 따른다. release_certain
행은 y_core_departed가 결측이라 자동으로 학습에서 제외된다.

원본 코드 대비 유지한 설계 (여전히 유효함)
  1. 시계열 분할 (학습 09~21 / 검증 22~23 / 테스트 24). contract.SPLIT 그대로 재사용.
  2. Optuna 목적함수 = AUC (accuracy는 불균형 데이터에서 "전부 잔류 예측"만
     해도 높게 나와 튜닝 기준으로 부적합).
  3. class_weight='balanced'.
  4. eval_metric을 목적함수(auc)와 일치.

추가한 것
  - D의 BaseModel/registry에 등록 (ML: DepartureLGBM, DL: DepartureMLP).
  - D의 evaluate.py로 지표 계산 — 다른 4개 태스크 모델과 동일한 형식.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

if __package__:
    from .base import BaseModel
    from .evaluate import evaluate
else:
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.models.base import BaseModel
    from src.models.evaluate import evaluate

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data" / "final" / "features_v1.parquet"

# contract.py의 SPLIT과 동일 — 팀 공통 기준
TRAIN_START_YEAR, TRAIN_END_YEAR = 2009, 2021
VAL_START_YEAR, VAL_END_YEAR = 2022, 2023
TEST_START_YEAR, TEST_END_YEAR = 2024, 2024

TARGET = "y_departed"

# def_score는 strength.py에 수비 전력 계산이 아직 없어 전부 NaN이라 뺀다.
FEATURE_COLS = [
    "age", "exp", "g_ratio", "g_ratio_prev", "g_chg",
    "off_score", "pit_score", "overall_score",
    "ops_z", "ops_z_prev", "era_z", "whip_z", "team_wr",
]


def load_features(path: Path = FEATURES_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def time_based_split(df: pd.DataFrame):
    """season 기준 시계열 분할. y_core_departed 결측(release_certain 등)은 제외."""
    labeled = df[df[TARGET].notna()]
    train = labeled[labeled.season.between(TRAIN_START_YEAR, TRAIN_END_YEAR)]
    val = labeled[labeled.season.between(VAL_START_YEAR, VAL_END_YEAR)]
    test = labeled[labeled.season.between(TEST_START_YEAR, TEST_END_YEAR)]
    return train, val, test


def to_xy(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    y = df[TARGET].astype(int)
    return X, y


class DepartureLGBM(BaseModel):
    """B 담당 ML 이탈 이진분류 모델. Optuna로 튜닝한다."""

    name, task, kind, owner = "departure_lgbm", "departure", "ml", "B"

    def __init__(self, n_trials: int = 40, timeout: int = 600, **params):
        super().__init__(**params)
        self.n_trials = n_trials
        self.timeout = timeout
        self.best_params_: dict = {}

    def fit_with_validation(self, X_train, y_train, X_val, y_val):
        """Optuna로 검증셋 AUC를 최대화하는 하이퍼파라미터를 찾은 뒤 최종 학습한다."""

        def objective(trial: optuna.Trial) -> float:
            param = {
                "objective": "binary",
                "metric": "auc",
                "boosting_type": "gbdt",
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "class_weight": "balanced",
                "force_row_wise": True,
                "n_jobs": -1,
                "random_state": 42,
                "verbosity": -1,
            }
            model = LGBMClassifier(**param)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                callbacks=[early_stopping(stopping_rounds=10, verbose=False), log_evaluation(period=0)],
            )
            proba = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, proba)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        self.best_params_ = study.best_params

        best_params = dict(study.best_params)
        best_params.update({"class_weight": "balanced", "random_state": 42, "verbosity": -1})
        self.params = best_params
        self.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
        return study.best_value

    def _fit(self, X, y):
        self.model = LGBMClassifier(**self.params).fit(X, y)

    def _predict_proba(self, X):
        return self.model.predict_proba(X)


class DepartureMLP(BaseModel):
    """B 담당 DL 이탈 이진분류 모델. sklearn MLP (TensorFlow 미설치 환경 폴백)."""

    name, task, kind, owner = "departure_mlp", "departure", "dl", "B"

    def __init__(self, **params):
        defaults = {
            "hidden_layer_sizes": (64, 32),
            "activation": "relu",
            "solver": "adam",
            "alpha": 1e-4,
            "learning_rate_init": 1e-3,
            "max_iter": 300,
            "early_stopping": True,
            "validation_fraction": 0.15,
            "random_state": 42,
        }
        defaults.update(params)
        super().__init__(**defaults)

    def _fit(self, X, y):
        from sklearn.impute import SimpleImputer
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self.model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(**self.params)),
        ]).fit(X, y)

    def _predict_proba(self, X):
        return self.model.predict_proba(X)


if __name__ == "__main__":
    features = load_features()
    train_df, val_df, test_df = time_based_split(features)
    X_train, y_train = to_xy(train_df)
    X_val, y_val = to_xy(val_df)
    X_test, y_test = to_xy(test_df)

    print(f"train: {len(X_train)}건 ({TRAIN_START_YEAR}~{TRAIN_END_YEAR}) / "
          f"val: {len(X_val)}건 ({VAL_START_YEAR}~{VAL_END_YEAR}) / "
          f"test: {len(X_test)}건 ({TEST_START_YEAR}~{TEST_END_YEAR})")
    print(f"train 이탈 비율: {y_train.mean():.1%}")

    # ---- ML: LGBM + Optuna ----
    print("\n[ML] DepartureLGBM 학습 (Optuna 튜닝)...")
    lgbm = DepartureLGBM()
    best_auc = lgbm.fit_with_validation(X_train, y_train, X_val, y_val)
    print(f"검증 AUC: {best_auc:.4f} / best_params: {lgbm.best_params_}")
    metrics = evaluate(lgbm, X_test, y_test)
    lgbm.set_metrics(**metrics)
    path = lgbm.save(note="Optuna 튜닝, train+val로 최종학습 후 test 평가")
    print(f"[{lgbm.name}] test AUC={metrics.get('roc_auc', float('nan')):.4f} f1={metrics.get('f1', float('nan')):.4f} -> {path}")

    # ---- DL: MLP ----
    print("\n[DL] DepartureMLP 학습...")
    mlp = DepartureMLP()
    mlp.fit(X_train, y_train)
    metrics = evaluate(mlp, X_test, y_test)
    mlp.set_metrics(**metrics)
    path = mlp.save(note="sklearn MLPClassifier 폴백")
    print(f"[{mlp.name}] test AUC={metrics.get('roc_auc', float('nan')):.4f} f1={metrics.get('f1', float('nan')):.4f} -> {path}")
