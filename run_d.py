"""D 파트 전체 동작 확인.

    uv run python run_d.py

목업 데이터로 3개 모델(XGBoost / LSTM / MLP)을 학습하고 평가한 뒤
models/registry/ 에 결과를 저장한다.
features_v1.parquet 이 생기면 자동으로 실제 데이터로 전환된다.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from src.features.contract import FEATURES_PATH, load_features, split  # noqa: E402
from src.models.base import registry_table  # noqa: E402
from src.models.evaluate import evaluate  # noqa: E402
from src.models.strength_ts import (  # noqa: E402
    LAG_FEATURES,
    StrengthLSTM,
    StrengthMLP,
    StrengthXGB,
    add_lag_features,
    build_sequences,
    make_xy,
)


def line(title):
    print(f"\n{'─' * 58}\n{title}\n{'─' * 58}")


line("1. 데이터 불러오기")
raw = load_features()
print(f"   출처   : {'실제 데이터' if FEATURES_PATH.exists() else '목업 (가짜)'}")
print(f"   행 / 열: {len(raw):,} / {raw.shape[1]}")
print(f"   시즌   : {int(raw.season.min())} ~ {int(raw.season.max())}")
print("\n   라벨 분포(%)")
for k, v in (raw.y_reason.value_counts(normalize=True) * 100).round(1).items():
    print(f"     {k:<12} {v:>5}")

line("2. 데이터 미리보기")
cols = ["player_id", "season", "age", "g_ratio", "overall_score", "y_departed", "y_reason", "y_next_score"]
print(raw[cols].head(8).to_string(index=False))

line("3. lag 피처 생성")
df = add_lag_features(raw)
tr, va, te = split(df, "train"), split(df, "valid"), split(df, "test")
print(f"   train {len(tr):,} / valid {len(va):,} / test {len(te):,}")
print(f"   피처 {len(LAG_FEATURES)}개")
print("\n   한 선수의 시간 흐름 (lag 가 제대로 밀렸는지 확인)")
pid = df[df.groupby("player_id").player_id.transform("size") >= 5].player_id.iloc[0]
print(
    df[df.player_id == pid][
        ["season", "age", "overall_score", "score_lag1", "score_ma3", "y_next_score"]
    ].round(1).to_string(index=False)
)

line("4. ML — XGBoost")
Xtr, ytr = make_xy(tr)
Xte, yte = make_xy(te)
xgb = StrengthXGB().fit(Xtr, ytr)
r_xgb = evaluate(xgb, Xte, yte, verbose=True)
xgb.set_metrics(**r_xgb).save()
print("\n   상위 피처")
for k, v in xgb.importance().head(6).round(3).items():
    print(f"     {k:<16} {v}")

line("5. 3D 텐서 만들기")
Xs, ys, _ = build_sequences(tr)
Xs_te, ys_te, _ = build_sequences(te)
print(f"   train {Xs.shape}  =  (샘플, 시즌, 지표)")
print(f"   test  {Xs_te.shape}")
print("\n   첫 샘플의 overall_score 5시즌 (앞의 0 = 패딩, Masking 이 무시함)")
print(f"     {np.round(Xs[0][:, 0], 1)}")

line("6. DL — LSTM")
try:
    lstm = StrengthLSTM(epochs=15).fit(Xs, ys)
    r_lstm = evaluate(lstm, Xs_te, ys_te, verbose=True)
    lstm.set_metrics(**r_lstm).save()
except ImportError:
    print("   tensorflow 미설치 — 건너뜀")

line("7. DL 폴백 — sklearn MLP")
mlp = StrengthMLP().fit(Xtr, ytr)
r_mlp = evaluate(mlp, Xte, yte, verbose=True)
mlp.set_metrics(**r_mlp).save()

line("8. 최종 비교표  (화면⑤가 읽는 것과 동일)")
t = registry_table()
show = [c for c in ["model", "task", "kind", "owner", "mae", "rmse", "r2", "baseline_mae"] if c in t.columns]
print(t[show].to_string(index=False))

print("\n   MAE = 평균 몇 점 틀리는가 (전력 점수 0~100 기준)")
print("   baseline_mae 는 그냥 평균으로 찍었을 때의 오차")
print("\n   models/registry/ 에 JSON 이 생성되었습니다.")