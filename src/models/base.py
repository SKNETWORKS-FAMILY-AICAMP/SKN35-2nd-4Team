"""모델 공통 인터페이스.

5명이 만드는 10개 모델(ML 5 + DL 5)이 모두 이 클래스를 상속한다.
save() 호출 시 models/registry/{name}.json 이 자동 생성되고,
화면⑤는 load_registry() 만 읽어서 비교표를 그린다. 취합 작업이 발생하지 않는다.

레지스트리를 단일 파일이 아니라 담당자별 파일로 나눈 이유는 머지 충돌 방지다.

담당: D (Day 1 오전 확정. 이후 수정은 전원 합의)
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
REGISTRY_DIR = MODEL_DIR / "registry"

# contract.py 의 태스크 정의와 일치해야 한다
CLASSIFICATION_TASKS = {"win_rate", "departure", "reason", "recommend"}
REGRESSION_TASKS = {"strength"}
TASKS = CLASSIFICATION_TASKS | REGRESSION_TASKS
KINDS = {"ml", "dl"}
OWNERS = {"A", "B", "C", "D", "E"}


def to_py(obj: Any) -> Any:
    """numpy 타입을 파이썬 기본형으로 변환한다.

    sklearn 지표는 전부 numpy 타입이라 이 처리 없이는 json.dump 가 TypeError 를 낸다.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return round(float(obj), 4)
    if isinstance(obj, np.ndarray):
        return [to_py(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(x) for x in obj]
    if isinstance(obj, float):
        return round(obj, 4)
    return obj


def _is_keras(model) -> bool:
    """keras 모델인지 판정. tensorflow 미설치 환경에서도 안전하다."""
    return type(model).__module__.split(".")[0] in ("keras", "tensorflow")


def _is_torch(model) -> bool:
    """torch.nn.Module 인지 판정 (MRO 로 검사 — torch 미설치 환경에서도 안전하다)."""
    return any(c.__module__.split(".")[0] == "torch" for c in type(model).__mro__)


class BaseModel:
    """모든 모델의 부모.

    하위 클래스는 클래스 변수 4개를 선언하고, 분류면 _fit/_predict_proba,
    회귀면 _fit/_predict 를 구현한다.

        class DepartureLGBM(BaseModel):
            name, task, kind, owner = "departure_lgbm", "departure", "ml", "B"

            def _fit(self, X, y):
                self.model = LGBMClassifier(**self.params).fit(X, y)

            def _predict_proba(self, X):
                return self.model.predict_proba(X)
    """

    name: str = ""
    task: str = ""
    kind: str = ""
    owner: str = ""

    def __init__(self, **params):
        for attr in ("name", "task", "kind", "owner"):
            if not getattr(self, attr):
                raise ValueError(f"{type(self).__name__}: 클래스 변수 '{attr}' 를 선언할 것")
        if self.task not in TASKS:
            raise ValueError(f"task 는 {sorted(TASKS)} 중 하나여야 함 (받은 값: {self.task})")
        if self.kind not in KINDS:
            raise ValueError(f"kind 는 {sorted(KINDS)} 중 하나여야 함 (받은 값: {self.kind})")
        if self.owner not in OWNERS:
            raise ValueError(f"owner 는 {sorted(OWNERS)} 중 하나여야 함 (받은 값: {self.owner})")

        self.params = params
        self.model = None
        self.feature_names: list[str] = []
        self.classes_: list = []
        self.metrics: dict = {}

    @property
    def is_regression(self) -> bool:
        return self.task in REGRESSION_TASKS

    # 하위 클래스가 구현
    def _fit(self, X, y):
        raise NotImplementedError

    def _predict_proba(self, X):
        raise NotImplementedError("분류 모델은 _predict_proba 를 구현할 것")

    def _predict(self, X):
        raise NotImplementedError("회귀 모델은 _predict 를 구현할 것")

    # 공통
    def fit(self, X, y):
        self.feature_names = list(getattr(X, "columns", []))
        if not self.is_regression:
            self.classes_ = sorted(set(np.asarray(y).ravel().tolist()))
        self._fit(X, y)
        return self

    def predict_proba(self, X):
        if self.is_regression:
            raise TypeError(f"{self.task} 는 회귀 태스크다. predict() 를 쓸 것")
        self._check_fitted()
        return self._predict_proba(self._align(X))

    def predict(self, X):
        self._check_fitted()
        Xa = self._align(X)
        if self.is_regression:
            return np.asarray(self._predict(Xa)).ravel()
        proba = np.asarray(self._predict_proba(Xa))
        idx = proba.argmax(axis=1)
        return np.array([self.classes_[i] for i in idx]) if self.classes_ else idx

    def _check_fitted(self):
        if self.model is None:
            raise RuntimeError(f"{self.name}: fit() 을 먼저 호출할 것")

    def _align(self, X):
        """학습 때와 컬럼 순서가 달라도 안전하게 맞춘다."""
        if self.feature_names and hasattr(X, "reindex"):
            missing = set(self.feature_names) - set(X.columns)
            if missing:
                raise KeyError(f"{self.name}: 피처 누락 {sorted(missing)}")
            return X.reindex(columns=self.feature_names)
        return X

    def set_metrics(self, **metrics):
        """evaluate.py 가 계산한 지표를 붙인다."""
        self.metrics.update(to_py(metrics))
        return self

    # 저장 · 로드
    def save(self, note: str = "") -> Path:
        self._check_fitted()
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

        # kind 가 아니라 실제 객체 타입으로 판단한다.
        # DL 이어도 sklearn MLP 같은 폴백 모델은 pickle 로 저장해야 한다.
        if _is_keras(self.model):
            fmt = "keras"
        elif _is_torch(self.model):
            fmt = "torch"
        else:
            fmt = "pickle"

        if fmt == "keras":
            path = MODEL_DIR / f"{self.name}.keras"
            self.model.save(path)
        elif fmt == "torch":
            import torch

            path = MODEL_DIR / f"{self.name}.pt"
            torch.save(self.model, path)
        else:
            path = MODEL_DIR / f"{self.name}.pkl"
            with open(path, "wb") as f:
                pickle.dump(self.model, f)

        entry = {
            "name": self.name,
            "task": self.task,
            "kind": self.kind,
            "owner": self.owner,
            "is_regression": self.is_regression,
            "format": fmt,
            # 레지스트리는 Git으로 공유되고 Linux 배포 환경에서도 읽힌다.
            # Windows에서 저장해도 JSON에는 POSIX 구분자를 사용한다.
            "path": path.relative_to(ROOT).as_posix(),
            "n_features": len(self.feature_names),
            "features": self.feature_names,
            "classes": to_py(self.classes_),
            "params": {k: str(v) for k, v in self.params.items()},
            "metrics": self.metrics,
            "note": note,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(REGISTRY_DIR / f"{self.name}.json", "w", encoding="utf-8") as f:
            json.dump(to_py(entry), f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load(cls, name: str) -> "BaseModel":
        meta_path = REGISTRY_DIR / f"{name}.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"레지스트리에 없음: {name}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        obj = cls.__new__(cls)
        obj.name = meta["name"]
        obj.task = meta["task"]
        obj.kind = meta["kind"]
        obj.owner = meta["owner"]
        obj.feature_names = meta["features"]
        obj.classes_ = meta.get("classes", [])
        obj.metrics = meta.get("metrics", {})
        obj.params = {}

        path = ROOT / meta["path"]
        fmt = meta.get("format", "keras" if meta["kind"] == "dl" else "pickle")
        if fmt == "keras":
            from tensorflow import keras

            obj.model = keras.models.load_model(path)
        elif fmt == "torch":
            import torch

            obj.model = torch.load(path, weights_only=False)
            obj.model.eval()
        else:
            with open(path, "rb") as f:
                obj.model = pickle.load(f)
        return obj

    def __repr__(self) -> str:
        m = " ".join(f"{k}={v}" for k, v in list(self.metrics.items())[:3])
        return f"<{self.name} [{self.owner}/{self.kind}] {m}>"


def load_registry() -> list[dict]:
    """등록된 모든 모델 메타를 읽는다. 화면⑤가 이것만 호출한다."""
    if not REGISTRY_DIR.exists():
        return []
    out = []
    for p in sorted(REGISTRY_DIR.glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            print(f"[registry] 손상된 파일 건너뜀: {p.name}")
    return out


# 표에 펼치면 읽을 수 없는 중첩 지표. 상세 조회는 load_registry() 로 한다.
NESTED_METRICS = {"confusion_matrix", "f1_per_class", "support_per_class", "labels"}


def registry_table(include_nested: bool = False):
    """비교표용 DataFrame. 지표 키는 모델마다 달라 union 으로 펼친다."""
    import pandas as pd

    rows = []
    for e in load_registry():
        row = {
            "model": e["name"],
            "task": e["task"],
            "kind": e["kind"].upper(),
            "owner": e["owner"],
            "n_features": e["n_features"],
        }
        metrics = e.get("metrics", {})
        if not include_nested:
            metrics = {k: v for k, v in metrics.items() if k not in NESTED_METRICS}
        row.update(metrics)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["task", "kind"]).reset_index(drop=True)
