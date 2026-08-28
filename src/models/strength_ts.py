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
    "had_injury",  # 그 시즌 IL 등재 여부 — 성적 하락이 노쇠인지 부상인지 구분하는 신호
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

    # 작년에 부상이 있었다면, 올해 성적 하락을 노쇠가 아니라 회복 과정으로 볼 근거가 된다
    d["injury_lag1"] = g.had_injury.shift(1)

    return d

# 성적 관련 피처들
# [만약 score_lag1 = 85, score_lag2 = 80
# score_trend = 85 - 80 = +- 5
# 최근 실력이 상승하고 있다는 뜻]
# overall_score - 현재 시즌 종합 전력 점수 - 지금 얼마나 잘하는 선수인가
# score_lag1 - 1시즌 전 overall_score - 작년 실력
# score_lag2 - 2시즌 전 overall_score - 재작년 실력
# score_lag3 - 3시즌 전 overall_score - 3년 전 실력
# score_maa3 - 최근 3시즌 종합점수 평균 - 최근 3년 평균 실력
# score_trend - [lag1 - lag2] - 작년보다 실력이 올랐는지/떨어졌는지
# score_std3 - 최근 3시즌 점수의 표준편차 - 실력이 얼마나 들쭉날쭉한지

# 경기 참여 관련 피처들
# [만약 g_ratio = 0.9라면 -> 팀 경기의 90% 정도를 출전했다는 의미]
# g_ratio - 경기 참여 비율 - 팀 경기 중 얼마나 많이 뛰었는가
# gratio_lag1 - 1시즌 전 경기 참여 비율 - 작년 출전 비율
# gratio-lag2 - 2시즌 전 경기 참여 비율 - 재작년 출전 비율

# 나이 관련
# [예를들어 30살이면:
# age = 30, age_c = 30 - 27 = 3, age_sq = 3^2 = 9, past_peak = 1
# 즉 선수의 나이에 따른 기량 변화를 모델에게 알려주는 변수]
# age - 현재 나이 - 현재 몇 살인가
# age_c - [age - 27] - 전성기 27세에서 얼마나 떨어져 있는가
# age_sq - age_c^2 - 전성기에서 멀어질수록 영향이 커지는 정도
# past_peak - 27세 초과 여부 - 전성기를 지났는가?

# 경험 관련
# [정확히 어떤 방식으로 계산했는지는 원본 데이터의 exp 정의를 봐야 하지만, 일반적으로 프로 경력 연수나 누적 경험을 의미]
# exp - 선수 경험/경력 지표 - 얼마나 경험이 많은 선수인가

# 선수 세부 성적
# [z-score는 0 -> 평균 정도, +1 -> 평균보다 좋음, -1 -> 평균보다 나쁨
#  다만 ERA는 낮을수록 좋은 지표라서 era_z의 방향은 전처리 과정에서 어떻게 정의했는지 확인해봐야 함]

# 팀 성적
# [만약 team_wr = 0.650이면 팀이 전체 경기의 65%를 승리했다는 의미, 선수 개인 능력뿐 아니라 소속 팀의 환경/전력도 다음 
# 시즌 전력에 영향을 줄 수 있어서 넣음]
# team_wr - 팀 숭률(Win Rate) - 선수가 속한 팀이 얼마나 잘하고 있는가

# 이 선수의 현재 실력 + 과거 실력 + 성장/하락 추세 + 나이 + 출전량 + 개인 성적 + 팀 성적을
# 종합해서 다음 시즌 실력을 예측

LAG_FEATURES = [
    "overall_score", "score_lag1", "score_lag2", "score_lag3",
    "score_ma3", "score_trend", "score_std3",

    "g_ratio", "gratio_lag1", "gratio_lag2",

    "age", "age_c", "age_sq", "past_peak", "exp",

    "ops_z", "era_z", "team_wr",

    "had_injury", "injury_lag1",
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
    target: str = TARGET,
    pad_value: float = 0.0,
):
    """(샘플, 시즌, 지표) 3D 텐서로 변환한다. LSTM 작업의 90% 가 여기다.

    선수마다 뛴 시즌 수가 다르므로 앞을 pad_value 로 채운다.
    Masking 레이어가 이 값을 무시하게 되어 있으므로 pad_value 를 바꾸면 안 된다.

    target 을 바꾸면 다른 태스크에서도 그대로 재사용할 수 있다 — 예:
    departure.py 가 y_core_departed 로 시퀀스를 만들 때 이 함수를 그대로 쓴다
    (선수별 과거 시즌 흐름을 3D 텐서로 접는 로직은 태스크와 무관하다).

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
        tgt = grp[target].to_numpy(dtype="float32")
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


# ── DL: LSTM (PyTorch) ────────────────────────────────────────────
# 최근 5시즌의 흐름을 순서대로 보고 다음 시즌 전력을 예측
#
# torch 는 이 파일 최상단이 아니라 _fit/_predict 안에서 지연 임포트한다 (아래 참고).
# nn.Module 정의는 _torch_lstm_net.py 로 분리했다 — 이유 2가지:
#   1. torch.save() 는 내부적으로 pickle 을 쓰는데 함수 안 closure 클래스는 저장이 안 된다.
#   2. torch 를 xgboost 보다 먼저 이 프로세스에 로드하면 macOS(arm64)에서 두 라이브러리의
#      OpenMP/Accelerate 초기화가 충돌해 세그폴트가 난다. StrengthXGB 가 먼저 xgboost 를
#      쓴 뒤에만 torch 가 들어오도록, import 시점을 여기(실제 fit 호출 시점)까지 미룬다.
#
# 순서를 미뤄도 xgboost 와 torch 가 각자 멀티스레드 BLAS/OpenMP 를 같이 쓰면 여전히
# 데드락이 난다(같은 프로세스, macOS arm64에서 재현됨). torch 쪽 스레드를 1개로 고정해
# 충돌을 피한다 — LSTM 자체가 가벼워서 성능 손해는 거의 없다.
_torch_threads_limited = False


def _limit_torch_threads(torch) -> None:
    global _torch_threads_limited
    if not _torch_threads_limited:
        torch.set_num_threads(1)
        _torch_threads_limited = True


class StrengthLSTM(BaseModel):
    """최근 SEQ_LEN 시즌을 시퀀스로 읽는 회귀 모델 (PyTorch).

    입력이 3D 라 fit/predict 에 DataFrame 이 아닌 ndarray 를 넘긴다.
    build_sequences() 로 만든 X 를 그대로 사용한다.
    """

    name, task, kind, owner = "strength_lstm", "strength", "dl", "D"

    def _fit(self, X, y):
        import torch
        import torch.nn as nn

        from ._torch_lstm_net import MaskedLSTMNet

        _limit_torch_threads(torch)

        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32")
        n_feat = X.shape[2]
        units = self.params.get("units", 32)
        dropout = self.params.get("dropout", 0.2)
        epochs = self.params.get("epochs", 40)
        batch_size = self.params.get("batch_size", 64)
        patience = self.params.get("patience", 5)

        # validation_split=0.15 와 동일 — 뒤쪽 15% 를 검증에 쓴다
        n_val = max(1, int(len(X) * 0.15))
        Xtr, Xva = torch.tensor(X[:-n_val]), torch.tensor(X[-n_val:])
        ytr, yva = torch.tensor(y[:-n_val]), torch.tensor(y[-n_val:])

        self.model = MaskedLSTMNet(n_feat, units, dropout)
        opt = torch.optim.Adam(self.model.parameters())
        loss_fn = nn.MSELoss()

        best_state, best_val, bad_epochs = None, float("inf"), 0
        for _ in range(epochs):
            self.model.train()
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), batch_size):
                idx = perm[i : i + batch_size]
                opt.zero_grad()
                pred = self.model(Xtr[idx])
                loss = loss_fn(pred, ytr[idx])
                loss.backward()
                opt.step()

            self.model.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model(Xva), yva).item()

            # EarlyStopping(patience=5, restore_best_weights=True) 와 동일한 로직
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

    def _predict(self, X):
        import torch

        _limit_torch_threads(torch)

        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(np.asarray(X, dtype="float32"))
            return self.model(X).numpy().ravel()

    def _align(self, X):
        # 3D 입력은 컬럼 정렬 개념이 없다
        return X


# ── DL 폴백: torch 설치 실패 시 ──────────────────────────────────────
# torch 를 사용할 수 없으면 MLP를 대체 모델로 사용
class StrengthMLP(BaseModel):
    """torch 를 못 쓸 때의 대체 DL 모델. 2D lag 피처를 그대로 쓴다."""

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
# 결과 활용
# 모델이 예측한 y_next_score를 이용해 다음 시즌 선수 전력을 예상
# 이 결과를 B, C의 선수 이탈 예측과 결합해서 대체 선수 추천에 활용
# 최종적으로 "이 선수가 이탈하면 -> 팀 전력이 얼마나 떨어지고 -> 누구로 대체할 것인가"를 시뮬레이션하는 데 사용"
# ===========================================================