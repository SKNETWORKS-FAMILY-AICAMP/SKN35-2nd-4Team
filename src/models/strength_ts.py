"""다음 시즌 전력 예측 — D 담당.

타깃: y_next_score (다음 시즌 overall_score). 5개 태스크 중 유일한 회귀다.

  ML  StrengthXGB  과거 시즌을 컬럼으로 펼쳐(lag) 트리 모델로 예측
  DL  StrengthLSTM 최근 N시즌을 3D 텐서로 넣어 시퀀스로 예측

주의 — 생존 편향
  y_next_score 는 다음 시즌 기록이 있어야 존재한다. 즉 이탈한 선수는 학습에서 빠진다.
  이 모델은 "리그에 남은 선수" 만 보고 학습하므로 낙관 편향이 있다.
  대체 선수 추천에 쓸 때는 이탈 모델(B·C)과 결합해 보정해야 한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseModel

TARGET = "y_next_score"
SEQ_LEN = 5          # LSTM 이 보는 시즌 수
PEAK_AGE = 27        # 야구 선수 기량 정점. aging curve 의 꼭짓점

# 시퀀스에 넣을 지표. 선수의 "상태"를 나타내는 연속값만 쓴다
SEQ_FEATURES = [
    "overall_score",
    "off_score",
    "pit_score",
    "g_ratio",
    "ops_z",
    "era_z",
    "age",
]


# ── 피처 엔지니어링 ─────────────────────────────────────────────────
def add_lag_features(df: pd.DataFrame, n_lags: int = 3) -> pd.DataFrame:
    """과거 시즌을 컬럼으로 펼친다. 트리 모델은 이 형태여야 시계열을 학습한다.

    groupby('player_id') 없이 shift 하면 앞 선수의 성적이 섞인다. 최다 실수 지점.
    """
    d = df.sort_values(["player_id", "season"]).copy()
    g = d.groupby("player_id")

    for k in range(1, n_lags + 1):
        d[f"score_lag{k}"] = g.overall_score.shift(k)
        d[f"gratio_lag{k}"] = g.g_ratio.shift(k)

    # 이동평균 — shift(1) 을 먼저 해야 당해 성적이 섞이지 않는다 (누수 방지)
    prev = g.overall_score.shift(1)
    d["score_ma3"] = prev.groupby(d.player_id).rolling(3, min_periods=1).mean().values
    d["score_trend"] = d.score_lag1 - d.score_lag2          # 직전 상승/하락폭
    d["score_std3"] = prev.groupby(d.player_id).rolling(3, min_periods=1).std().values

    # aging curve — 27세 정점의 포물선을 잡으려면 2차항이 필요하다
    d["age_c"] = d.age - PEAK_AGE
    d["age_sq"] = d.age_c ** 2
    d["past_peak"] = (d.age > PEAK_AGE).astype(int)

    return d


# overall_score - 현재 시즌 종합 전력 점수 - 지금 얼마나 잘하는 선수인가
# score_lag1 - 1시즌 전 overall_score - 작년 실력
# score_lag2 - 2시즌 전 overall_score - 재작년 실력
# score_lag3 - 3시즌 전 overall_score - 3년 전 실력
# score_maa3 - 최근 3시즌 종합점수 평균 - 최근 3년 평균 실력
# score_trend - [lag1 - lag2] - 작년보다 실력이 올랐는지/떨어졌는지
# score_std3 - 최근 3시즌 점수의 표준편차 - 실력이 얼마나 들쭉날쭉한지

LAG_FEATURES = [
    "overall_score", "score_lag1", "score_lag2", "score_lag3",
    "score_ma3", "score_trend", "score_std3",
    "g_ratio", "gratio_lag1", "gratio_lag2",
    "age", "age_c", "age_sq", "past_peak", "exp",
    "ops_z", "era_z", "team_wr",
]


def make_xy(df: pd.DataFrame, features: list[str] | None = None):
    """타깃이 있는 행만 남긴다. 결측 타깃은 학습에 쓸 수 없다."""
    features = features or LAG_FEATURES
    d = df.dropna(subset=[TARGET])
    return d[features].fillna(0), d[TARGET]


# ── 3D 텐서 (LSTM 입력) ────────────────────────────────────────────
def build_sequences(
    df: pd.DataFrame,
    seq_len: int = SEQ_LEN,
    features: list[str] | None = None,
    pad_value: float = 0.0,
):
    """(샘플, 시즌, 지표) 3D 텐서로 변환한다. LSTM 작업의 90% 가 여기다.

    선수마다 뛴 시즌 수가 다르므로 앞을 pad_value 로 채운다.
    Masking 레이어가 이 값을 무시하게 되어 있으므로 pad_value 를 바꾸면 안 된다.

    Returns:
        X: (n, seq_len, n_features)
        y: (n,)
        meta: player_id, season — 예측 결과를 되돌려 붙일 때 사용
    """
    features = features or SEQ_FEATURES
    d = df.sort_values(["player_id", "season"]).copy()
    d[features] = d[features].fillna(0.0)

    X, y, meta = [], [], []
    for pid, grp in d.groupby("player_id", sort=False):
        vals = grp[features].to_numpy(dtype="float32")
        tgt = grp[TARGET].to_numpy(dtype="float32")
        seasons = grp["season"].to_numpy()

        for i in range(len(grp)):
            if np.isnan(tgt[i]):        # 타깃 없는 시즌은 건너뛴다
                continue
            lo = max(0, i - seq_len + 1)
            window = vals[lo : i + 1]
            if len(window) < seq_len:   # 앞을 패딩 (pre-padding)
                pad = np.full((seq_len - len(window), len(features)), pad_value, dtype="float32")
                window = np.vstack([pad, window])
            X.append(window)
            y.append(tgt[i])
            meta.append((pid, int(seasons[i])))

    if not X:
        raise ValueError("시퀀스를 만들 수 없다. TARGET 이 전부 결측인지 확인할 것")

    return (
        np.stack(X),
        np.asarray(y, dtype="float32"),
        pd.DataFrame(meta, columns=["player_id", "season"]),
    )


# ── ML: XGBoost ────────────────────────────────────────────────────
# 과거 성적과 나이 등을 보고 다음 시즌 전력을 회귀 예측
class StrengthXGB(BaseModel):
    """과거 시즌을 lag 컬럼으로 펼쳐 학습하는 회귀 모델."""

    name, task, kind, owner = "strength_xgb", "strength", "ml", "D"

    def _fit(self, X, y):
        from xgboost import XGBRegressor

        defaults = dict(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        defaults.update(self.params)
        self.model = XGBRegressor(**defaults).fit(X, y)

    def _predict(self, X):
        return self.model.predict(X)

    def importance(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_, index=self.feature_names
        ).sort_values(ascending=False)


# ── DL: LSTM ───────────────────────────────────────────────────────
# 최근 5시즌의 흐름을 순서대로 보고 다음 시즌 전력을 예측
class StrengthLSTM(BaseModel):
    """최근 SEQ_LEN 시즌을 시퀀스로 읽는 회귀 모델.

    입력이 3D 라 fit/predict 에 DataFrame 이 아닌 ndarray 를 넘긴다.
    build_sequences() 로 만든 X 를 그대로 사용한다.
    """

    name, task, kind, owner = "strength_lstm", "strength", "dl", "D"

    def _fit(self, X, y):
        from tensorflow import keras
        from tensorflow.keras import layers

        X = np.asarray(X, dtype="float32")
        seq_len, n_feat = X.shape[1], X.shape[2]

        self.model = keras.Sequential(
            [
                keras.Input(shape=(seq_len, n_feat)),
                # 패딩된 0 을 무시한다. 이게 없으면 신인의 빈 시즌을 실제 0점으로 학습한다
                layers.Masking(mask_value=0.0),
                layers.LSTM(self.params.get("units", 32)),
                layers.Dropout(self.params.get("dropout", 0.2)),
                layers.Dense(16, activation="relu"),
                layers.Dense(1),
            ]
        )
        self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
        self.history = self.model.fit(
            X,
            np.asarray(y, dtype="float32"),
            epochs=self.params.get("epochs", 40),
            batch_size=self.params.get("batch_size", 64),
            validation_split=0.15,
            callbacks=cb,
            verbose=self.params.get("verbose", 0),
        )

    def _predict(self, X):
        return self.model.predict(np.asarray(X, dtype="float32"), verbose=0).ravel()

    def _align(self, X):
        # 3D 입력은 컬럼 정렬 개념이 없다
        return X


# ── DL 폴백: tensorflow 설치 실패 시 ────────────────────────────────
# tensorflow를 사용할 수 없으면 MLP를 대체 모델로 사용
class StrengthMLP(BaseModel):
    """tensorflow 를 못 쓸 때의 대체 DL 모델. 2D lag 피처를 그대로 쓴다."""

    name, task, kind, owner = "strength_mlp", "strength", "dl", "D"

    def _fit(self, X, y):
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        defaults = dict(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            early_stopping=True,
            random_state=42,
        )
        defaults.update(self.params)
        # 스케일러는 파이프라인 안에 둔다. 밖에서 전체 fit 하면 테스트 정보가 샌다
        self.model = make_pipeline(StandardScaler(), MLPRegressor(**defaults)).fit(X, y)

    def _predict(self, X):
        return self.model.predict(X)
    
    
# ===========================================================
