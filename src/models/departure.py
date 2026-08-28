"""
departure.py
B - 이탈 이진분류 (LGBMClassifier + Optuna, sklearn API 스타일)

원본 코드 대비 수정 사항:
  1. train_test_split(랜덤) -> 시계열 분할 (학습 09~21 / 검증 22~23 / 테스트 24~25)
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
  6. 더미 라벨 생성 로직 제거 -> C가 생성한 y_core_departed 사용.
  7. 기존 레거시 컬럼(yearID, batting_strength 등) 제거 -> features_v1 새 스키마 사용.
"""

import pandas as pd
import optuna
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report

FEATURES_PATH = "data/final/features_v1.parquet"

TRAIN_START_YEAR, TRAIN_END_YEAR = 2009, 2021
VAL_START_YEAR, VAL_END_YEAR = 2022, 2023
TEST_START_YEAR, TEST_END_YEAR = 2024, 2025

# features_v1의 새 스키마 기준.
# y_* 라벨 컬럼은 누수 방지를 위해 feature에서 제외한다.
FEATURE_COLS = [
    "off_score",
    "pit_score",
    "g_ratio",
    "ops_z",
    "era_z",
    "whip_z",
    "overall_score",
    "g_ratio_prev",
    "g_chg",
    "def_score",
    "team_wr",
    "age",
    "exp",
    "n_stint",
    "allstar",
    "role",
]

TARGET_COL = "y_core_departed"


def load_features(path: str = FEATURES_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def attach_real_label(df: pd.DataFrame) -> pd.DataFrame:
    """C가 생성한 실제 이탈 라벨 y_core_departed를 사용한다.

    더미/랜덤 라벨은 생성하지 않는다.
    """
    if TARGET_COL not in df.columns:
        raise KeyError(
            f"실제 라벨 컬럼 '{TARGET_COL}'이 features_v1.parquet에 없습니다. "
            "build.py에서 labels.py 결과가 병합되었는지 확인하세요."
        )

    out = df.copy()
    out[TARGET_COL] = pd.to_numeric(out[TARGET_COL], errors="coerce")

    invalid = out[TARGET_COL].dropna().loc[
        ~out[TARGET_COL].dropna().isin([0, 1])
    ]
    if len(invalid):
        raise ValueError(
            f"{TARGET_COL}에 0/1이 아닌 값이 있습니다: "
            f"{sorted(invalid.unique().tolist())}"
        )

    out = out.dropna(subset=[TARGET_COL]).copy()
    out[TARGET_COL] = out[TARGET_COL].astype(int)

    return out

def time_based_split(df: pd.DataFrame):
    """랜덤 분할 대신 season 기준 시계열 분할."""
    if "season" not in df.columns:
        raise KeyError("features_v1.parquet에 season 컬럼이 없습니다.")

    train = df[(df.season >= TRAIN_START_YEAR) & (df.season <= TRAIN_END_YEAR)].copy()
    val = df[(df.season >= VAL_START_YEAR) & (df.season <= VAL_END_YEAR)].copy()
    test = df[(df.season >= TEST_START_YEAR) & (df.season <= TEST_END_YEAR)].copy()

    return train, val, test


def to_xy(df: pd.DataFrame):
    missing = [col for col in FEATURE_COLS if col not in df.columns]
    if missing:
        raise KeyError(
            f"features_v1.parquet에 필요한 feature 컬럼이 없습니다: {missing}"
        )

    X = df[FEATURE_COLS].copy()

    # role은 contract상 문자열 컬럼이므로 모델 입력용 숫자로 변환한다.
    role_map = {"P": 0, "B": 1, "TWO": 2}
    X["role"] = X["role"].map(role_map)

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([float("inf"), float("-inf")], pd.NA)
    X = X.apply(pd.to_numeric, errors="coerce")

    y = df[TARGET_COL].astype(int)

    return X, y


if __name__ == "__main__":
    features = load_features()
    features = attach_real_label(features)

    train_df, val_df, test_df = time_based_split(features)
    X_train, y_train = to_xy(train_df)
    X_val, y_val = to_xy(val_df)
    X_test, y_test = to_xy(test_df)

    print(f"train: {len(X_train)}건 ({TRAIN_START_YEAR}~{TRAIN_END_YEAR}) / "
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