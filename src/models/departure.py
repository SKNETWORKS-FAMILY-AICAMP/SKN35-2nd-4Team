"""
departure.py
B - 이탈 이진분류 (LGBMClassifier + Optuna, sklearn API 스타일)

원본 코드 대비 수정 사항:
  1. train_test_split(랜덤) -> 시계열 분할 (학습 ≤2021 / 검증 22~23 / 테스트 24~25)
     D의 Day1 Step2 팀 공통 기준. 랜덤 분할은 미래 데이터로 학습하고 과거로
     테스트하는 누수를 만들 수 있어 D의 누수 감시(Day2 Step7)에 걸림.
  2. print(f'...{accuracy_score:.4f}') 버그 수정 (함수 자체를 출력하려던 오타)
     -> print(f'...{accuracy:.4f}')
  3. Optuna 목적함수를 accuracy -> AUC로 변경.
     이탈 데이터는 잔류가 훨씬 많은 불균형 데이터라, accuracy는 "전부 잔류로
     예측"만 해도 80%+ 나올 수 있어 튜닝 기준으로 부적합 (실제로 더미 라벨
     테스트에서 이 현상이 그대로 재현됨: accuracy 81%인데 이탈 recall 0%).
  4. class_weight='balanced' 추가 (문서에 명시된 class_weight 확인 항목).
  5. eval_metric을 목적함수(auc)와 일치시킴 (기존엔 binary_error로 따로 놀았음).

TODO: C의 실제 라벨 도착 시 attach_dummy_label() 호출부만 교체.
"""

import numpy as np
import pandas as pd
import optuna
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report

FEATURES_PATH = "data/processed/features_v1.parquet"

TRAIN_END_YEAR = 2021
VAL_START_YEAR, VAL_END_YEAR = 2022, 2023
TEST_START_YEAR, TEST_END_YEAR = 2024, 2025

FEATURE_COLS = [
    "batting_strength", "pitching_strength", "OPS", "HR", "RBI", "R",
    "ERA", "WHIP", "SO9", "IP", "AB",
    "playing_time_ratio", "playing_time_ratio_pit",
    "is_batter", "is_pitcher",
]


def load_features(path: str = FEATURES_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def attach_dummy_label(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """TODO: C 라벨 도착 시 실제 merge로 교체. 지금은 파이프라인 확인용."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["departed"] = rng.choice([0, 1], size=len(out), p=[0.8, 0.2])
    return out


def time_based_split(df: pd.DataFrame):
    """랜덤 분할 대신 연도 기준 시계열 분할 (D의 팀 공통 기준)."""
    train = df[df.yearID <= TRAIN_END_YEAR]
    val = df[(df.yearID >= VAL_START_YEAR) & (df.yearID <= VAL_END_YEAR)]
    test = df[(df.yearID >= TEST_START_YEAR) & (df.yearID <= TEST_END_YEAR)]
    return train, val, test


def to_xy(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    for col in ["is_batter", "is_pitcher"]:
        X[col] = X[col].astype(int)
    y = df["departed"]
    return X, y


if __name__ == "__main__":
    features = load_features()
    features = attach_dummy_label(features)  # TODO: C 라벨 도착 시 교체

    train_df, val_df, test_df = time_based_split(features)
    X_train, y_train = to_xy(train_df)
    X_val, y_val = to_xy(val_df)
    X_test, y_test = to_xy(test_df)

    print(f"train: {len(X_train)}건 (~{TRAIN_END_YEAR}) / "
          f"val: {len(X_val)}건 ({VAL_START_YEAR}~{VAL_END_YEAR}) / "
          f"test: {len(X_test)}건 ({TEST_START_YEAR}~{TEST_END_YEAR})")

    # ---- 기본 모델 (튜닝 전 베이스라인) ----
    baseline = LGBMClassifier(random_state=42, class_weight="balanced", verbosity=-1)
    baseline.fit(X_train, y_train)
    preds = baseline.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"[베이스라인] Test accuracy: {accuracy:.4f}")  # 수정: accuracy_score(함수) -> accuracy(변수)

    # ---- Optuna 튜닝 ----
    # 검증은 반드시 시계열 val(2022~2023) 사용, 목적함수는 accuracy가 아닌 AUC
    # (불균형 데이터에서 accuracy는 "전부 잔류 예측"만 해도 높게 나와 튜닝 기준으로 부적합).
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
            "class_weight": "balanced",  # 문서 명시 항목: class_weight 확인
            "force_row_wise": True,
            "n_jobs": -1,
            "random_state": 42,
            "verbosity": -1,
        }

        model = LGBMClassifier(**param)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",  # objective/목적함수와 일치시킴 (기존 binary_error와 불일치 수정)
            callbacks=[early_stopping(stopping_rounds=10, verbose=False), log_evaluation(period=0)],
        )

        proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, proba)  # accuracy -> AUC로 변경 (불균형 데이터 대응)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=40, timeout=600)  # n_trials 100 -> 40 + timeout (B 부하 고려)

    print("\nBest parameters found: ", study.best_params)
    print("Best validation AUC: ", study.best_value)

    # ---- 최적 파라미터로 최종 학습 + 테스트 평가 ----
    best_params = study.best_params
    best_params.update({"class_weight": "balanced", "random_state": 42, "verbosity": -1})
    best_model = LGBMClassifier(**best_params)
    best_model.fit(X_train, y_train)

    preds = best_model.predict(X_test)
    proba = best_model.predict_proba(X_test)[:, 1]

    print(f"\n=== 최종 테스트 평가 (2024~2025) ===")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"AUC: {roc_auc_score(y_test, proba):.4f}")
    print(f"F1: {f1_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds))