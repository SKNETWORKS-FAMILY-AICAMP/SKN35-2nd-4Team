"""선수 이탈 원인 태그 생성과 다중분류 모델.

이 모듈은 관측된 L1/L2/L2b/L3 라벨을 변경하지 않는다. 시즌 t까지 관측된
부상·성적·생애주기 정보를 이용해 이탈자에게 원인 보조 태그를 만들고,
RandomForest와 MLP가 그 규칙 기반 라벨을 학습할 수 있는 인터페이스를 제공한다.

주의
----
원인 라벨은 공식 방출·은퇴 사유가 아니라 약한 지도학습(weak supervision)을
위한 추정값이다. 서비스 화면에는 반드시 "연관 요인" 또는 "모델 추정"으로
표시한다. 자세한 기준은 ``docs/label_spec.md``를 따른다.

원인 태그 적용 범위 (팀 결정, 2026-08-27)
------------------------------------------
``이탈유형정의_v2``의 원래 설계원칙은 원인 태그를 방출·은퇴에만 붙이고
FA·트레이드는 제외하는 것이었다. 팀 논의 결과, 부상은 방출·은퇴뿐 아니라
트레이드·FA 이적의 배경 요인으로도 실제로 작용할 수 있다고 보고(예: 부상
이력이 있는 선수를 방출 대신 트레이드로 정리, 부상 이후 시장가치 하락으로
낮은 조건에 FA 계약) **원인 태그를 전체 이탈 유형(trade/offseason_move/
league_exit)에 다시 적용하기로 확장 결정**했다. `docs/label_spec.md`와
`이탈유형정의_v2_원인태그포함.md`에도 이 결정을 반영해야 한다.

단, 이 태그는 여전히 "구단·선수가 이 이유로 떠났다"는 확정 원인이 아니라
"이탈 시점 전후로 이런 신호가 함께 관측됐다"는 약한 지도학습 추정치다.
`reason_explanation`의 문구도 "~때문에 이탈함"이 아니라 "~가 함께 관측됨"
형태로 이미 인과관계를 단정하지 않게 되어 있으니, 화면에 노출할 때도 이
표현을 그대로 유지할 것.

참고: 이 원인 태그의 적용 범위는 대체 선수 추천 점수식(포지션 적합성 +
전력 유사도 + 낮은 이탈 위험 + 세부 스탯 유사성)과는 별개다. 추천 점수식의
"이탈 위험"은 `departure.py`가 예측하는 핵심 이탈확률(`y_departed`/
`y_core_departed`)이고, 여기서 다루는 `primary_reason`/`reason_tags`는 그
이탈이 왜 일어났을 가능성이 있는지를 사후 설명하는 별도 출력이다. 즉 원인
태그 범위를 넓혀도 추천 점수식 자체는 바뀌지 않는다.
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
    "y_fa_release",
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
    """방출·은퇴에 한해 원인 보조 태그와 근거 수준을 부여한다.

    반환되는 ``primary_reason``은 다중분류 학습용 대표 태그이고,
    ``reason_tags``에는 동시에 활성화된 모든 태그를 튜플로 보존한다.
    잔류자와 라벨 검열 행의 ``primary_reason``은 결측이다. 팀 결정(2026-08-27)에
    따라 원인 태그는 trade/offseason_move/league_exit 전체 이탈 유형에 적용한다
    (모듈 docstring "원인 태그 적용 범위" 참고) — 단 career_stage(생애주기)는
    성격상 은퇴·리그이탈에만 의미가 있으므로 league_exit로 한정한다.
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
    is_retirement_bucket = out["y_path"].eq("league_exit")
    reason_eligible = departed

    injury = (
        reason_eligible
        & out["injury_record_matched"]
        & out["had_injury"].gt(0)
        & out["reason_injury_score"].ge(thresholds.injury_risk)
    )
    performance = reason_eligible & (
        out["overall_score_delta"].le(thresholds.score_delta)
        | out["g_chg"].le(thresholds.g_change)
    )
    career = (
        reason_eligible
        & is_retirement_bucket
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

    for idx, is_eligible in reason_eligible.items():
        if not is_eligible:
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

    def _fit(self, X, y):
        from sklearn.impute import SimpleImputer
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self.model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", MLPClassifier(**self.params)),
            ]
        ).fit(X, y)

    def _predict_proba(self, X):
        return self.model.predict_proba(X)


def threshold_metadata(thresholds: ReasonThresholds) -> dict[str, float]:
    """결과서와 모델 메모에 저장할 수 있는 임계값 사전을 반환한다."""

    return {key: float(value) for key, value in asdict(thresholds).items()}


if __name__ == "__main__":
    from src.features.contract import make_mock
    from src.features.labels import LABEL_COLUMNS, build_labels

    mock = make_mock(n_players=500)
    raw = mock.drop(columns=LABEL_COLUMNS, errors="ignore")
    labeled = build_labels(raw)

    # 직접 실행 시 파이프라인 연결만 확인한다. 모델 파일은 저장하지 않는다.
    injury_mock = labeled.loc[
        labeled.index % 7 == 0, ["player_id", "season"]
    ].copy()
    injury_mock["had_injury"] = 1
    injury_mock["il_stint_count"] = 1

    reason_data, fitted = build_reason_dataset(labeled, injury_mock)
    X, y = to_reason_xy(reason_data)
    print(f"원인 데이터 생성 완료: X={X.shape}, y={len(y):,}")
    print(f"원인 분포:\n{y.value_counts().to_string()}")
    print(f"임계값: {threshold_metadata(fitted)}")
