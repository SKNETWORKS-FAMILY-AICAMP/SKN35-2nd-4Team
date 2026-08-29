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
  - D의 BaseModel/registry에 등록 (ML: DepartureLGBM, DL: DepartureMLP/DepartureLSTM).
  - D의 evaluate.py로 지표 계산 — 다른 4개 태스크 모델과 동일한 형식.

DL 시퀀스 모델 (DepartureLSTM)
  DepartureMLP는 한 시즌의 스냅샷(단일 행)만 본다 — "최근 하락세인지"는
  ops_z_prev/g_ratio_prev 두 컬럼으로만 짐작한다. DepartureLSTM은 D의
  strength_ts.py가 이미 갖춘 시퀀스 인프라(build_sequences, MaskedLSTMNet)를
  그대로 재사용해 선수별 최근 SEQ_LEN 시즌 흐름 전체를 보고 이탈위험을 예측한다.
  회귀(y_next_score) → 분류(y_core_departed)로 타깃만 바뀌므로,
  build_sequences()에 target= 파라미터를 추가해 그대로 호출한다 (중복 구현 없음).

[2026-08-29 수정 - LSTM 성능 개선]
  DepartureLSTM/StrengthLSTM 둘 다 build_sequences()가 만든 3D 텐서를 정규화
  없이 그대로 torch.tensor()에 넣고 있었다. SEQ_FEATURES를 보면 스케일이
  완전히 다른 값들이 섞여 있다: off_score/pit_score/overall_score(0~100),
  ops_z/era_z(대략 -3~3), g_ratio(0~1), age(대략 20~40), had_injury(0/1).
  DepartureMLP는 파이프라인 안에 StandardScaler가 있는데 LSTM 경로만 빠져
  있었다 - strength_ts.py 주석에도 "LSTM이 MLP 폴백을 못 넘어서는 문제
  확인됨"이라고 이미 기록되어 있던 것과 정확히 일치하는 원인.
  fit_seq_scaler/apply_seq_scaler로 train 구간 통계만으로 표준화한 뒤
  build_sequences()에 넘기도록 수정 (val/test 정보 누수 방지 - train 구간
  에서만 평균/표준편차 계산). build_sequences() 자체(D 소유)는 건드리지
  않음 - pad_value=0.0 도 정규화 후에는 "평균값"을 뜻하게 되어 오히려
  의미가 더 자연스러워짐 (기존 raw 스케일에서는 age=0처럼 있을 수 없는
  값으로 패딩되고 있었음).
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
    from .strength_ts import SEQ_FEATURES, SEQ_LEN, _limit_torch_threads, build_sequences
else:
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.models.base import BaseModel
    from src.models.evaluate import evaluate
    from src.models.strength_ts import SEQ_FEATURES, SEQ_LEN, _limit_torch_threads, build_sequences

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data" / "final" / "features_v1.parquet"

# contract.py의 SPLIT과 동일 — 팀 공통 기준
TRAIN_START_YEAR, TRAIN_END_YEAR = 2009, 2021
VAL_START_YEAR, VAL_END_YEAR = 2022, 2023
TEST_START_YEAR, TEST_END_YEAR = 2024, 2024

# y_departed로 바꾸지 말 것 — release_certain(구단이 방출)까지 "이탈위험"에 섞이면
# "선수가 떠날 위험"과 "구단이 내보낼 정책"이 뒤섞이는 순환논리가 생긴다
# (labels.py의 설계 원칙, 위 docstring 참고). y_core_departed는 release_certain
# 행이 자동으로 결측 처리돼서 학습에서 빠진다.
TARGET = "y_core_departed"

# def_score: strength.compute_fielding_strength()가 실제 값을 채우게 되면서
# (2026-08-28) 다시 포함시켰다. 수비 기록이 아예 없는 선수-시즌(지명타자 등)은
# 여전히 NaN — LGBM은 결측을 자체 처리하고, MLP/LSTM 경로는 SimpleImputer로
# 중앙값 대체한다(둘 다 기존에도 다른 컬럼 결측을 이렇게 다뤄왔음).
FEATURE_COLS = [
    "age", "exp", "g_ratio", "g_ratio_prev", "g_chg",
    "off_score", "pit_score", "def_score", "overall_score",
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


# ---------------------------------------------------------------------------
# [신규] 시퀀스 피처 정규화 - train 구간 통계만 사용 (val/test 누수 방지)
# ---------------------------------------------------------------------------

def fit_seq_scaler(df: pd.DataFrame, features: list[str] = SEQ_FEATURES) -> dict:
    """TRAIN_START_YEAR~TRAIN_END_YEAR 구간에서만 평균/표준편차를 계산한다."""
    train_rows = df[df.season.between(TRAIN_START_YEAR, TRAIN_END_YEAR)]
    mean = train_rows[features].mean()
    std = train_rows[features].std().replace(0, 1.0)  # 상수 컬럼(std=0) 방어 - 0으로 나누기 방지
    return {"mean": mean, "std": std}


def apply_seq_scaler(df: pd.DataFrame, scaler: dict, features: list[str] = SEQ_FEATURES) -> pd.DataFrame:
    """스케일러를 적용한 복사본을 반환한다 (원본 df는 건드리지 않음)."""
    d = df.copy()
    d[features] = (d[features] - scaler["mean"]) / scaler["std"]
    return d


def prepare_departure_sequences(df: pd.DataFrame, seq_len: int = SEQ_LEN):
    """DepartureLSTM용 3D 시퀀스. strength_ts.build_sequences()를 target만 바꿔 재사용.

    release_certain 등으로 y_core_departed가 결측인 행은 자기 자신의 샘플로는
    안 만들어지지만(build_sequences가 건너뜀), 그 선수의 과거 시즌 히스토리로는
    여전히 쓰인다 — time_based_split처럼 미리 걸러내면 안 되는 이유다.

    [수정] build_sequences() 호출 전에 train 구간 통계로 정규화한다.
    이전에는 raw 스케일(off_score 0~100 vs ops_z -3~3 등)을 그대로 넣어
    LSTM이 큰 스케일 피처에 그래디언트가 지배당해 학습이 잘 안 됐다.

    Returns:
        X: (n, seq_len, n_features), y: (n,), meta: player_id, season
    """
    d = df.copy()
    # had_injury는 현재 확정 features_v1에 아직 없다 — next_strength.py와 동일하게
    # 0.0으로 채운다(strength_ts.SEQ_FEATURES가 이 컬럼을 요구함).
    if "had_injury" not in d.columns:
        d["had_injury"] = 0.0

    scaler = fit_seq_scaler(d, features=SEQ_FEATURES)
    d = apply_seq_scaler(d, scaler, features=SEQ_FEATURES)

    return build_sequences(d, seq_len=seq_len, features=SEQ_FEATURES, target=TARGET)


def _split_sequences_by_season(X, y, meta: pd.DataFrame):
    """meta.season 기준으로 시퀀스를 train/val/test로 나눈다 (time_based_split의 시퀀스판)."""
    season = meta["season"].to_numpy()
    train_mask = (season >= TRAIN_START_YEAR) & (season <= TRAIN_END_YEAR)
    val_mask = (season >= VAL_START_YEAR) & (season <= VAL_END_YEAR)
    test_mask = (season >= TEST_START_YEAR) & (season <= TEST_END_YEAR)
    return (
        (X[train_mask], y[train_mask]),
        (X[val_mask], y[val_mask]),
        (X[test_mask], y[test_mask]),
    )


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


class DepartureLSTM(BaseModel):
    """B 담당 DL 이탈 이진분류 모델 — 시퀀스 버전 (PyTorch, D의 MaskedLSTMNet 재사용).

    입력이 3D(선수별 최근 SEQ_LEN 시즌)라 fit/predict에 DataFrame이 아닌 ndarray를
    넘긴다. prepare_departure_sequences()로 만든 X를 그대로 쓴다 (train 구간
    통계로 이미 정규화되어 있음).

    strength_ts.MaskedLSTMNet은 회귀용으로 만들어졌지만 마지막 층이 스칼라 1개를
    내는 구조라 그대로 재사용 가능하다 — 여기서는 그 스칼라를 로짓으로 해석하고
    BCEWithLogitsLoss로 학습한다(MSELoss 대신). 새 nn.Module을 따로 만들지 않는다.
    """

    name, task, kind, owner = "departure_lstm", "departure", "dl", "B"

    def _fit(self, X, y):
        import torch
        import torch.nn as nn

        from ._torch_lstm_net import MaskedLSTMNet

        _limit_torch_threads(torch)

        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32")
        n_feat = X.shape[2]
        units = self.params.get("units", 32)
        dropout = self.params.get("dropout", 0.3)
        epochs = self.params.get("epochs", 60)
        batch_size = self.params.get("batch_size", 64)
        patience = self.params.get("patience", 7)

        # validation_split=0.15 와 동일 — StrengthLSTM과 같은 방식으로 내부 홀드아웃
        n_val = max(1, int(len(X) * 0.15))
        Xtr, Xva = torch.tensor(X[:-n_val]), torch.tensor(X[-n_val:])
        ytr, yva = torch.tensor(y[:-n_val]), torch.tensor(y[-n_val:])

        # 이탈은 소수 클래스라 LGBM/MLP와 동일하게 class_weight='balanced' 취지로
        # pos_weight를 준다 (train 쪽 비율로만 계산 — 검증/테스트 정보 누수 방지).
        n_pos = max(1.0, float(ytr.sum().item()))
        n_neg = max(1.0, float(len(ytr) - ytr.sum().item()))
        pos_weight = torch.tensor(n_neg / n_pos)

        self.model = MaskedLSTMNet(n_feat, units, dropout)
        opt = torch.optim.Adam(self.model.parameters(), weight_decay=1e-5)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        val_loss_fn = nn.BCEWithLogitsLoss()  # 검증 손실은 클래스 가중치 없이 비교

        best_state, best_val, bad_epochs = None, float("inf"), 0
        for _ in range(epochs):
            self.model.train()
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), batch_size):
                idx = perm[i : i + batch_size]
                opt.zero_grad()
                logits = self.model(Xtr[idx])
                loss = loss_fn(logits, ytr[idx])
                loss.backward()
                opt.step()

            self.model.eval()
            with torch.no_grad():
                val_loss = val_loss_fn(self.model(Xva), yva).item()

            if val_loss < best_val:
                best_val, bad_epochs = val_loss, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()

    def _predict_proba(self, X):
        import torch

        _limit_torch_threads(torch)

        self.model.eval()
        with torch.no_grad():
            Xt = torch.tensor(np.asarray(X, dtype="float32"))
            p1 = torch.sigmoid(self.model(Xt)).numpy().ravel()
        return np.stack([1.0 - p1, p1], axis=1)

    def _align(self, X):
        # 3D 입력은 컬럼 정렬 개념이 없다 (StrengthLSTM과 동일)
        return X


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

    # ---- DL: MLP (단일 시즌 스냅샷) ----
    print("\n[DL] DepartureMLP 학습...")
    mlp = DepartureMLP()
    mlp.fit(X_train, y_train)
    metrics = evaluate(mlp, X_test, y_test)
    mlp.set_metrics(**metrics)
    path = mlp.save(note="sklearn MLPClassifier 폴백")
    print(f"[{mlp.name}] test AUC={metrics.get('roc_auc', float('nan')):.4f} f1={metrics.get('f1', float('nan')):.4f} -> {path}")

    # ---- DL: LSTM (선수별 최근 시즌 시퀀스, D의 strength_ts.py 인프라 재사용) ----
    print("\n[DL] DepartureLSTM 학습 (시퀀스)...")
    X_seq, y_seq, meta_seq = prepare_departure_sequences(features)
    (Xtr_seq, ytr_seq), (Xva_seq, yva_seq), (Xte_seq, yte_seq) = _split_sequences_by_season(
        X_seq, y_seq, meta_seq
    )
    print(f"  시퀀스 train {len(Xtr_seq)}건 / val {len(Xva_seq)}건 / test {len(Xte_seq)}건 "
          f"(SEQ_LEN={SEQ_LEN}, features={len(SEQ_FEATURES)}개)")

    lstm = DepartureLSTM()
    lstm.fit(Xtr_seq, ytr_seq)
    metrics = evaluate(lstm, Xte_seq, yte_seq)
    lstm.set_metrics(**metrics)
    path = lstm.save(note=f"PyTorch MaskedLSTM, SEQ_LEN={SEQ_LEN}, 선수별 최근 시즌 흐름 재사용(D), train구간 정규화 적용")
    print(f"[{lstm.name}] test AUC={metrics.get('roc_auc', float('nan')):.4f} f1={metrics.get('f1', float('nan')):.4f} -> {path}")

    # 참고용: 진짜 VAL 구간(22~23) 지표도 출력 — LGBM처럼 Optuna 튜닝에 쓰진 않지만
    # 시퀀스 분할이 의도대로 됐는지 눈으로 확인하는 용도
    if len(Xva_seq):
        val_metrics = evaluate(lstm, Xva_seq, yva_seq)
        print(f"  (참고) val({VAL_START_YEAR}~{VAL_END_YEAR}) AUC="
              f"{val_metrics.get('roc_auc', float('nan')):.4f} f1={val_metrics.get('f1', float('nan')):.4f}")