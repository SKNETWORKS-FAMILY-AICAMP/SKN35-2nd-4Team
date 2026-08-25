"""모델 평가 — 태스크별 지표 분기.

10개 모델(5명 × ML/DL)이 전부 이 함수로 지표를 뽑는다.
각자 다른 지표를 쓰면 화면⑤의 비교표가 성립하지 않는다.

    from src.models.evaluate import evaluate
    m.set_metrics(**evaluate(m, X_test, y_test))
    m.save()

반환값은 to_py() 를 거쳐 전부 파이썬 기본형이므로 그대로 json.dump 가능하다.

담당: D
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from .base import CLASSIFICATION_TASKS, REGRESSION_TASKS, to_py

BINARY_TASKS = {"win_rate", "departure"}
MULTICLASS_TASKS = {"reason", "recommend"}


def evaluate(model, X, y, verbose: bool = False) -> dict:
    """model.task 에 따라 지표를 계산한다.

    Args:
        model: BaseModel 하위 인스턴스 (fit 완료 상태)
        X: 피처 DataFrame
        y: 정답 Series/array
        verbose: True 면 콘솔에도 출력

    Returns:
        직렬화 가능한 dict. set_metrics(**결과) 로 바로 넘길 수 있다.
    """
    task = model.task
    y_true = np.asarray(y).ravel()

    if task in REGRESSION_TASKS:
        out = _regression(model, X, y_true)
    elif task in BINARY_TASKS:
        out = _binary(model, X, y_true)
    elif task in MULTICLASS_TASKS:
        out = _multiclass(model, X, y_true)
    else:
        raise ValueError(f"알 수 없는 task: {task}")

    out["n_test"] = int(len(y_true))
    out = to_py(out)

    if verbose:
        print(format_metrics(model.name, out))
    return out


# ── 회귀 (D: strength) ─────────────────────────────────────────────
def _regression(model, X, y_true) -> dict:
    pred = np.asarray(model.predict(X)).ravel()
    mask = ~np.isnan(y_true)
    if mask.sum() == 0:
        raise ValueError("정답이 전부 결측이다. y_next_score 의 NaN 을 먼저 제거할 것")
    y_true, pred = y_true[mask], pred[mask]

    mae = mean_absolute_error(y_true, pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2_score(y_true, pred),
        # 평균으로 찍는 것 대비 얼마나 나은지. R² 가 음수일 때 해석을 돕는다
        "baseline_mae": mean_absolute_error(y_true, np.full_like(y_true, y_true.mean())),
    }


# ── 이진 분류 (A: win_rate, B: departure) ───────────────────────────
def _binary(model, X, y_true) -> dict:
    proba = np.asarray(model.predict_proba(X))
    # 양성 클래스 확률 열을 classes_ 기준으로 찾는다 (열 순서 가정 금지)
    pos_idx = _positive_index(model)
    p1 = proba[:, pos_idx]
    pred = np.asarray(model.predict(X)).ravel()

    out = {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }
    # 단일 클래스만 존재하면 AUC/LogLoss 가 정의되지 않는다
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, p1)
        out["log_loss"] = log_loss(y_true, np.clip(p1, 1e-15, 1 - 1e-15))
    else:
        out["roc_auc"] = None
        out["log_loss"] = None

    out["confusion_matrix"] = confusion_matrix(y_true, pred).tolist()
    out["positive_rate"] = float(np.mean(pred == model.classes_[pos_idx]))
    return out


def _positive_index(model) -> int:
    """양성 클래스의 열 인덱스. classes_ 가 [0.0, 1.0] 또는 [0, 1] 형태."""
    if not model.classes_:
        return 1
    try:
        return int(np.argmax([float(c) for c in model.classes_]))
    except (TypeError, ValueError):
        return len(model.classes_) - 1


# ── 다중 분류 (C: reason, E: recommend) ─────────────────────────────
def _multiclass(model, X, y_true) -> dict:
    pred = np.asarray(model.predict(X)).ravel()
    labels = list(model.classes_) if model.classes_ else sorted(set(y_true.tolist()))

    per_class = f1_score(y_true, pred, labels=labels, average=None, zero_division=0)
    support = [int((y_true == c).sum()) for c in labels]

    out = {
        "accuracy": accuracy_score(y_true, pred),
        # 소수 클래스를 동등하게 취급한다. 트레이드가 6% 뿐이라 accuracy 는 의미가 없다
        "macro_f1": f1_score(y_true, pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, pred, labels=labels, average="weighted", zero_division=0),
        "f1_per_class": {str(c): float(v) for c, v in zip(labels, per_class)},
        "support_per_class": {str(c): n for c, n in zip(labels, support)},
        "labels": [str(c) for c in labels],
        "confusion_matrix": confusion_matrix(y_true, pred, labels=labels).tolist(),
    }

    # 확률 출력이 가능하면 다중 log_loss 도 기록
    try:
        proba = np.asarray(model.predict_proba(X))
        if proba.shape[1] == len(labels) and len(np.unique(y_true)) > 1:
            out["log_loss"] = log_loss(y_true, proba, labels=labels)
    except (NotImplementedError, TypeError, ValueError):
        pass

    return out


# ── 출력 보조 ──────────────────────────────────────────────────────
def format_metrics(name: str, m: dict) -> str:
    """콘솔·노트북용 한 줄 요약."""
    skip = {"confusion_matrix", "f1_per_class", "support_per_class", "labels"}
    body = "  ".join(
        f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
        for k, v in m.items()
        if k not in skip and v is not None
    )
    return f"[{name}] {body}"


def confusion_frame(m: dict):
    """혼동행렬을 라벨이 붙은 DataFrame 으로. 화면⑤에서 사용한다."""
    import pandas as pd

    cm = m.get("confusion_matrix")
    if cm is None:
        return None
    labels = m.get("labels") or [str(i) for i in range(len(cm))]
    return pd.DataFrame(
        cm,
        index=[f"실제 {l}" for l in labels],
        columns=[f"예측 {l}" for l in labels],
    )


def compare(*models):
    """여러 모델의 지표를 한 표로. 이미 set_metrics 된 모델을 넘긴다."""
    import pandas as pd

    skip = {"confusion_matrix", "f1_per_class", "support_per_class", "labels"}
    rows = []
    for m in models:
        row = {"model": m.name, "task": m.task, "kind": m.kind.upper(), "owner": m.owner}
        row.update({k: v for k, v in m.metrics.items() if k not in skip})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["task", "kind"]).reset_index(drop=True)