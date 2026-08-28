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


def prepare_departure_sequences(df: pd.DataFrame, seq_len: int = SEQ_LEN):
    """DepartureLSTM용 3D 시퀀스. strength_ts.build_sequences()를 target만 바꿔 재사용.

    release_certain 등으로 y_core_departed가 결측인 행은 자기 자신의 샘플로는
    안 만들어지지만(build_sequences가 건너뜀), 그 선수의 과거 시즌 히스토리로는
    여전히 쓰인다 — time_based_split처럼 미리 걸러내면 안 되는 이유다.

    Returns:
        X: (n, seq_len, n_features), y: (n,), meta: player_id, season
    """
    d = df.copy()
    # had_injury는 현재 확정 features_v1에 아직 없다 — next_strength.py와 동일하게
    # 0.0으로 채운다(strength_ts.SEQ_FEATURES가 이 컬럼을 요구함).
    if "had_injury" not in d.columns:
        d["had_injury"] = 0.0
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
    넘긴다. prepare_departure_sequences()로 만든 X를 그대로 쓴다.

    strength_ts.MaskedLSTMNet은 회귀용으로 만들어졌지만 마지막 층이 스칼라 1개를
    내는 구조라 그대로 재사용 가능하다 — 여기서는 그 스칼라를 로짓으로 해석하고
    BCEWithLogitsLoss로 학습한다(MSELoss 대신). 새 nn.Module을 따로 만들지 않는다.
    """

    name, task, kind, owner = "departure_lstm", "departure", "dl", "B"

    def fit_with_validation(self, X_train, y_train, X_val, y_val, n_trials: int = 25, timeout: int = 300):
        """DepartureLGBM과 동일한 패턴 — Optuna로 검증셋 AUC를 최대화하는
        하이퍼파라미터(units/dropout/lr/weight_decay/batch_size)를 찾은 뒤
        train+val로 최종 학습한다. 예전엔 units=32/dropout=0.3 고정값이었는데
        (departure_lgbm은 튜닝하면서 LSTM만 안 하는 게 불공평한 비교였음),
        LGBM(0.828)과 6점 가까이 차이나던 걸 좁히는 게 목적.
        """
        import numpy as _np
        import torch
        from sklearn.metrics import roc_auc_score

        X_train = _np.asarray(X_train, dtype="float32")
        y_train = _np.asarray(y_train, dtype="float32")
        X_val = _np.asarray(X_val, dtype="float32")
        y_val = _np.asarray(y_val, dtype="float32")

        def objective(trial: optuna.Trial) -> float:
            params = {
                "units": trial.suggest_categorical("units", [16, 32, 48, 64]),
                "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
                "epochs": 40,
                "patience": 6,
            }
            trial_model = DepartureLSTM(**params)
            trial_model._fit_arrays(X_train, y_train, X_val, y_val)
            proba = trial_model._predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, proba)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        self.best_params_ = study.best_params

        best_params = dict(study.best_params)
        best_params.setdefault("epochs", 60)
        best_params.setdefault("patience", 7)
        self.params = best_params
        # 최종 학습은 train+val을 합쳐서 - LGBM과 동일한 관례. 내부 얼리스토핑용
        # 홀드아웃은 _fit_arrays가 그 안에서 다시 15% 떼어 쓴다.
        combined_X = _np.concatenate([X_train, X_val])
        combined_y = _np.concatenate([y_train, y_val])
        # _fit_arrays()를 직접 부르면 BaseModel.fit()을 안 거쳐서 classes_가
        # 빈 리스트로 남는다 - evaluate.py의 _binary()가 model.classes_[pos_idx]로
        # 색인하다 IndexError로 죽는다(실측 확인). self.fit()을 불러야
        # classes_/feature_names가 제대로 채워진다 - self.fit()은 내부적으로
        # self._fit()을 호출하고, DepartureLSTM._fit()은 X_val 없이
        # _fit_arrays()를 호출하므로(위에서 의도한 내부 15% 홀드아웃 경로) 결과는 동일하다.
        self.fit(combined_X, combined_y)
        return study.best_value

    def _fit(self, X, y):
        self._fit_arrays(np.asarray(X, dtype="float32"), np.asarray(y, dtype="float32"))

    def _fit_arrays(self, X, y, X_val=None, y_val=None):
        """실제 학습 루프. X_val/y_val을 명시적으로 주면 그걸 얼리스토핑 검증에
        쓰고(Optuna 트라이얼용 - 진짜 val 구간), 안 주면 StrengthLSTM과 동일하게
        X 뒤쪽 15%를 내부 홀드아웃으로 뗀다(train+val 합본 최종학습용)."""
        import torch
        import torch.nn as nn

        from ._torch_lstm_net import MaskedLSTMNet

        _limit_torch_threads(torch)

        n_feat = X.shape[2]
        units = self.params.get("units", 32)
        dropout = self.params.get("dropout", 0.3)
        epochs = self.params.get("epochs", 60)
        batch_size = self.params.get("batch_size", 64)
        patience = self.params.get("patience", 7)
        lr = self.params.get("lr", 1e-3)
        weight_decay = self.params.get("weight_decay", 1e-5)

        if X_val is not None and y_val is not None:
            Xtr, Xva = torch.tensor(X), torch.tensor(np.asarray(X_val, dtype="float32"))
            ytr, yva = torch.tensor(y), torch.tensor(np.asarray(y_val, dtype="float32"))
        else:
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
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
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
    best_auc = lstm.fit_with_validation(Xtr_seq, ytr_seq, Xva_seq, yva_seq)
    print(f"검증 AUC: {best_auc:.4f} / best_params: {lstm.best_params_}")
    metrics = evaluate(lstm, Xte_seq, yte_seq)
    lstm.set_metrics(**metrics)
    path = lstm.save(note=f"PyTorch MaskedLSTM, SEQ_LEN={SEQ_LEN}, Optuna 튜닝, train+val로 최종학습")
    print(f"[{lstm.name}] test AUC={metrics.get('roc_auc', float('nan')):.4f} f1={metrics.get('f1', float('nan')):.4f} -> {path}")
