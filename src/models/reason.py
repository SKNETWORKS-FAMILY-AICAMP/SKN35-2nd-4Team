"""선수 이탈 원인 태그 생성과 다중분류 모델.

이 모듈은 관측된 L1/L2/L2b/L3 라벨을 변경하지 않는다. 시즌 t까지 관측된
부상·성적·생애주기 정보를 이용해 이탈자에게 원인 보조 태그를 만들고,
RandomForest와 MLP가 그 규칙 기반 라벨을 학습할 수 있는 인터페이스를 제공한다.

주의
----
원인 라벨은 공식 방출·은퇴 사유가 아니라 약한 지도학습(weak supervision)을
위한 추정값이다. 서비스 화면에는 반드시 "연관 요인" 또는 "모델 추정"으로
표시한다. 자세한 기준은 ``docs/label_spec.md``를 따른다.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

if __package__:
    from .base import BaseModel
else:  # VS Code에서 이 파일을 직접 실행하는 경우도 지원한다.
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from src.models.base import BaseModel


KEY = ["player_id", "season"]
REASON_CLASSES = [
    "injury_associated",
    "performance_decline",
    "career_stage",
    "mixed",
    "unknown",
]

BASE_REQUIRED_COLUMNS = [
    "player_id",
    "season",
    "age",
    "exp",
    "g_chg",
    "overall_score",
    "y_departed",
    "y_path",
]

INJURY_RAW_COLUMNS = [
    "had_injury",
    "il_stint_count",
    "first_il_date",
    "injury_note_sample",
]

# 학습 입력은 모두 시즌 t 종료 시점에 알 수 있는 수치만 사용한다.
# y_departed/y_path 등 정답 라벨은 원인 모델의 입력에서 제외한다.
MODEL_FEATURE_CANDIDATES = [
    "age",
    "exp",
    "g_ratio",
    "g_ratio_prev",
    "g_chg",
    "overall_score",
    "overall_score_delta",
    "ops_z",
    "ops_z_prev",
    "era_z",
    "whip_z",
    "team_wr",
    "allstar",
    "had_injury",
    "il_stint_count",
    "injury_severity_score",
    "injury_frequency_score",
    "injury_risk_score",
]


@dataclass(frozen=True)
class ReasonThresholds:
    """훈련 구간에서만 계산하는 원인 태그 임계값."""

    injury_risk: float = 0.5
    score_delta: float = -5.0
    g_change: float = 0.7
    career_age: float = 34.0
    career_exp: float = 10.0


@dataclass(frozen=True)
class ReasonConfig:
    """원인 태그 생성 설정."""

    train_end_year: int = 2021
    injury_quantile: float = 0.75
    decline_quantile: float = 0.25
    career_quantile: float = 0.75

    def __post_init__(self) -> None:
        for name in ("injury_quantile", "decline_quantile", "career_quantile"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name}은 0과 1 사이여야 합니다: {value}")


def _require_columns(df: pd.DataFrame, columns: Iterable[str], *, name: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{name}에 필요한 컬럼이 없습니다: {missing}")


def merge_injury_data(
    player_season: pd.DataFrame,
    injury: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """선수 시즌 데이터에 부상 관측값을 안전하게 연결한다.

    ``injury``가 ``None``이면 이미 player_season에 들어 있는 부상 컬럼을 사용한다.
    외부 부상 데이터를 넘기면 ``player_id + season``으로 left join한다.

    조인되지 않은 행의 0은 "부상 없음"이 아니라 "현재 데이터에서 IL 기록이
    관측되지 않음"을 뜻한다. 이를 구분하기 위해 ``injury_record_matched``도 만든다.
    """

    _require_columns(player_season, KEY, name="player_season")
    out = player_season.copy()

    if injury is not None:
        _require_columns(injury, KEY, name="injury")
        if injury.duplicated(KEY).any():
            raise ValueError("부상 데이터에 player_id + season 중복이 있습니다.")

        injury_cols = [column for column in INJURY_RAW_COLUMNS if column in injury.columns]
        optional_scores = [
            column
            for column in (
                "injury_severity_score",
                "injury_frequency_score",
                "injury_risk_score",
            )
            if column in injury.columns
        ]
        overlapping = set(injury_cols + optional_scores) & set(out.columns)
        if overlapping:
            out = out.drop(columns=sorted(overlapping))

        source = injury[KEY + injury_cols + optional_scores].copy()
        source["injury_record_matched"] = True
        out = out.merge(source, on=KEY, how="left", validate="one_to_one")
    elif "injury_record_matched" not in out.columns:
        if "had_injury" in out.columns:
            out["injury_record_matched"] = out["had_injury"].notna()
        else:
            out["injury_record_matched"] = False

    out["injury_record_matched"] = out["injury_record_matched"].fillna(False).astype(bool)

    if "had_injury" not in out.columns:
        out["had_injury"] = 0.0
    else:
        out["had_injury"] = pd.to_numeric(out["had_injury"], errors="coerce").fillna(0.0)

    if "il_stint_count" not in out.columns:
        out["il_stint_count"] = 0.0
    else:
        out["il_stint_count"] = pd.to_numeric(
            out["il_stint_count"], errors="coerce"
        ).fillna(0.0)

    return out


def add_reason_features(player_season: pd.DataFrame) -> pd.DataFrame:
    """원인 분석에 필요한 파생값을 생성한다."""

    _require_columns(player_season, BASE_REQUIRED_COLUMNS, name="player_season")
    out = player_season.sort_values(KEY, kind="stable").reset_index(drop=True).copy()

    previous_score = out.groupby("player_id", sort=False)["overall_score"].shift(1)
    out["overall_score_delta"] = out["overall_score"] - previous_score

    # B의 통합 위험점수가 있으면 그대로 사용한다. 아직 없으면 가중치를 새로
    # 만들지 않고 공식 IL 기록 관측 여부만 0/1 대체값으로 사용한다.
    if "injury_risk_score" in out.columns:
        risk = pd.to_numeric(out["injury_risk_score"], errors="coerce")
        out["reason_injury_score"] = risk.where(risk.notna(), out["had_injury"])
        out["injury_score_source"] = np.where(risk.notna(), "b_feature", "observed_fallback")
    else:
        out["reason_injury_score"] = out["had_injury"].astype(float)
        out["injury_score_source"] = "observed_fallback"

    return out


def _safe_quantile(series: pd.Series, q: float, fallback: float) -> float:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return float(fallback)
    return float(numeric.quantile(q))


def fit_reason_thresholds(
    player_season: pd.DataFrame,
    config: ReasonConfig | None = None,
) -> ReasonThresholds:
    """훈련 기간만 사용해 원인 태그 임계값을 계산한다."""

    cfg = config or ReasonConfig()
    _require_columns(
        player_season,
        ["season", "reason_injury_score", "overall_score_delta", "g_chg", "age", "exp"],
        name="reason features",
    )
    train = player_season[player_season["season"] <= cfg.train_end_year]
    if train.empty:
        raise ValueError(f"{cfg.train_end_year}년 이하 훈련 데이터가 없습니다.")

    positive_injury = train.loc[train["reason_injury_score"] > 0, "reason_injury_score"]
    return ReasonThresholds(
        injury_risk=_safe_quantile(positive_injury, cfg.injury_quantile, 0.5),
        score_delta=_safe_quantile(
            train["overall_score_delta"], cfg.decline_quantile, -5.0
        ),
        g_change=_safe_quantile(train["g_chg"], cfg.decline_quantile, 0.7),
        career_age=_safe_quantile(train["age"], cfg.career_quantile, 34.0),
        career_exp=_safe_quantile(train["exp"], cfg.career_quantile, 10.0),
    )


def assign_reason_labels(
    player_season: pd.DataFrame,
    thresholds: ReasonThresholds,
) -> pd.DataFrame:
    """이탈자에게 원인 보조 태그와 근거 수준을 부여한다.

    반환되는 ``primary_reason``은 다중분류 학습용 대표 태그이고,
    ``reason_tags``에는 동시에 활성화된 모든 태그를 튜플로 보존한다.
    잔류자와 라벨 검열 행의 primary_reason은 결측이다.
    """

    required = BASE_REQUIRED_COLUMNS + [
        "reason_injury_score",
        "injury_score_source",
        "injury_record_matched",
        "overall_score_delta",
    ]
    _require_columns(player_season, required, name="reason features")
    out = player_season.copy()

    departed = out["y_departed"].eq(1.0)
    injury = (
        departed
        & out["injury_record_matched"]
        & out["had_injury"].gt(0)
        & out["reason_injury_score"].ge(thresholds.injury_risk)
    )
    performance = departed & (
        out["overall_score_delta"].le(thresholds.score_delta)
        | out["g_chg"].le(thresholds.g_change)
    )
    career = (
        departed
        & out["y_path"].eq("league_exit")
        & (
            out["age"].ge(thresholds.career_age)
            | out["exp"].ge(thresholds.career_exp)
        )
    )

    flag_frame = pd.DataFrame(
        {
            "injury_associated": injury.fillna(False),
            "performance_decline": performance.fillna(False),
            "career_stage": career.fillna(False),
        },
        index=out.index,
    )

    tags: list[tuple[str, ...]] = []
    primary: list[object] = []
    evidence: list[object] = []

    for idx, is_departed in departed.items():
        if not is_departed:
            tags.append(tuple())
            primary.append(pd.NA)
            evidence.append(pd.NA)
            continue

        active = tuple(name for name, value in flag_frame.loc[idx].items() if bool(value))
        tags.append(active or ("unknown",))

        if len(active) >= 2:
            primary.append("mixed")
        elif len(active) == 1:
            primary.append(active[0])
        else:
            primary.append("unknown")

        if injury.loc[idx] and out.loc[idx, "injury_score_source"] == "b_feature":
            evidence.append("estimated")
        elif injury.loc[idx]:
            evidence.append("associated")
        elif active:
            evidence.append("estimated")
        else:
            evidence.append("unknown")

    out["reason_tags"] = tags
    out["primary_reason"] = pd.Series(primary, index=out.index, dtype="object")
    out["evidence_level"] = pd.Series(evidence, index=out.index, dtype="object")
    out["reason_explanation"] = [
        _explanation_for(tags_value, evidence_value)
        for tags_value, evidence_value in zip(out["reason_tags"], out["evidence_level"])
    ]

    bad_classes = set(out["primary_reason"].dropna()) - set(REASON_CLASSES)
    if bad_classes:
        raise ValueError(f"정의되지 않은 원인 클래스: {sorted(bad_classes)}")
    return out


def _explanation_for(tags: tuple[str, ...], evidence_level: object) -> str | None:
    if pd.isna(evidence_level):
        return None
    if tags == ("unknown",) or not tags:
        return "현재 데이터로 이탈 연관 요인을 판단하기 어렵습니다."

    messages = {
        "injury_associated": "IL 등재 기록이 이탈 위험과 함께 관측됨",
        "performance_decline": "최근 성적 또는 출전 비중 하락이 관측됨",
        "career_stage": "연령·경력상 생애주기 요인이 리그 이탈과 함께 관측됨",
    }
    return "; ".join(messages[tag] for tag in tags if tag in messages)


def build_reason_dataset(
    player_season: pd.DataFrame,
    injury: pd.DataFrame | None = None,
    *,
    config: ReasonConfig | None = None,
    thresholds: ReasonThresholds | None = None,
) -> tuple[pd.DataFrame, ReasonThresholds]:
    """부상 연결부터 원인 라벨 생성까지 한 번에 수행한다."""

    merged = merge_injury_data(player_season, injury)
    featured = add_reason_features(merged)
    fitted_thresholds = thresholds or fit_reason_thresholds(featured, config)
    labeled = assign_reason_labels(featured, fitted_thresholds)
    return labeled, fitted_thresholds


def to_reason_xy(reason_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """원인 라벨이 존재하는 이탈자 행을 모델 입력 X와 정답 y로 변환한다."""

    _require_columns(reason_data, ["primary_reason"], name="reason_data")
    feature_columns = [
        column for column in MODEL_FEATURE_CANDIDATES if column in reason_data.columns
    ]
    if not feature_columns:
        raise ValueError("원인 모델에 사용할 수 있는 수치 피처가 없습니다.")

    eligible = reason_data["primary_reason"].notna()
    X = reason_data.loc[eligible, feature_columns].copy()
    y = reason_data.loc[eligible, "primary_reason"].astype(str)

    for column in X.columns:
        X[column] = pd.to_numeric(X[column], errors="coerce")
    return X, y


class ReasonRandomForest(BaseModel):
    """C 담당 ML 원인 다중분류 모델."""

    name = "reason_rf"
    task = "reason"
    kind = "ml"
    owner = "C"

    def __init__(self, **params):
        defaults = {
            "n_estimators": 300,
            "max_depth": 12,
            "min_samples_leaf": 3,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }
        defaults.update(params)
        super().__init__(**defaults)

    def _fit(self, X, y):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline

        self.model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", RandomForestClassifier(**self.params)),
            ]
        ).fit(X, y)

    def _predict_proba(self, X):
        return self.model.predict_proba(X)


class ReasonMLP(BaseModel):
    """C 담당 DL MLP 원인 다중분류 모델.

    프로젝트 공통 저장기가 TensorFlow 미설치 환경에서도 동작하도록 sklearn MLP를
    폴백 구현으로 사용한다. 레지스트리의 kind는 계획대로 ``dl``을 유지한다.
    """

    name = "reason_mlp"
    task = "reason"
    kind = "dl"
    owner = "C"

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
        self._label_encoder = None

    def _fit(self, X, y):
        from sklearn.impute import SimpleImputer
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        # sklearn 1.8에서 early_stopping=True인 MLPClassifier에 문자열
        # 타깃을 전달하면 내부 검증 과정에서 TypeError가 발생할 수 있다.
        # primary_reason을 정수로 인코딩하되, BaseModel.classes_와 같은
        # 정렬 순서를 사용하여 predict_proba의 열 순서를 유지한다.
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)

        encoded_classes = self._label_encoder.classes_.tolist()
        if self.classes_ and encoded_classes != list(self.classes_):
            raise ValueError(
                "BaseModel.classes_와 LabelEncoder 클래스 순서가 "
                "일치하지 않습니다. "
                f"BaseModel={self.classes_}, LabelEncoder={encoded_classes}"
            )

        self.model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", MLPClassifier(**self.params)),
            ]
        ).fit(X, y_encoded)

    def _predict_proba(self, X):
        return self.model.predict_proba(X)


def threshold_metadata(thresholds: ReasonThresholds) -> dict[str, float]:
    """결과서와 모델 메모에 저장할 수 있는 임계값 사전을 반환한다."""

    return {key: float(value) for key, value in asdict(thresholds).items()}


if __name__ == "__main__":
    from src.features.contract import SPLIT, load_features
    from src.models.evaluate import evaluate

    ROOT = Path(__file__).resolve().parents[2]
    injury_path = ROOT / "data" / "final" / "player_injury_stints.csv"
    if not injury_path.exists():
        raise FileNotFoundError(f"실제 부상 데이터가 없습니다: {injury_path}")

    features = load_features()  # contract.validate()까지 통과한 데이터만 사용
    injury = pd.read_csv(injury_path)

    reason_data, fitted = build_reason_dataset(features, injury)
    X, y = to_reason_xy(reason_data)

    print("=" * 60)
    print("실제 원인 데이터 생성 완료")
    print("=" * 60)
    print(f"전체 features_v1: {len(features):,}행")
    print(f"부상 데이터: {len(injury):,}행")
    print(f"원인 모델 입력: X={X.shape}, y={len(y):,}")

    print("\n[primary_reason 분포]")
    print(y.value_counts(dropna=False).to_string())

    print("\n[primary_reason 비율(%)]")
    print(
        y.value_counts(normalize=True, dropna=False)
        .mul(100)
        .round(2)
        .to_string()
    )

    print("\n[학습 구간에서 계산된 임계값]")
    for name, value in threshold_metadata(fitted).items():
        print(f"{name}: {value:.4f}")

    duplicate_count = int(reason_data.duplicated(KEY).sum())
    print(f"\n[player_id + season 중복]: {duplicate_count:,}건")

    season = reason_data.loc[X.index, "season"]
    train_lo, train_hi = SPLIT["train"]
    test_lo, test_hi = SPLIT["test"]
    train_mask = season.between(train_lo, train_hi)
    test_mask = season.between(test_lo, test_hi)
    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]
    print(f"\ntrain: {len(X_train):,}건 ({train_lo}~{train_hi}) / test: {len(X_test):,}건 ({test_lo}~{test_hi})")

    for model_cls in (ReasonRandomForest, ReasonMLP):
        model = model_cls()
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        model.set_metrics(**metrics)
        path = model.save(note="실제 features_v1 + player_injury_stints.csv로 학습")
        print(f"\n[{model.name}] macro_f1={metrics.get('macro_f1', float('nan')):.4f} -> {path}")