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
    "early_career_move",
    "stable_performance_move",
    "moderate_performance_decline",
    "limited_history",
    "mixed",
]

DEPARTURE_EVENT_TYPES = [
    "transaction_trade",
    "roster_release_waiver",
    "free_agent_market",
    "injury_roster_move",
    "minor_league_option",
    "league_exit",
    "unresolved_event",
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
    "unresolved_stints",
]

TRANSACTION_EVIDENCE_COLUMNS = [
    "transaction_trade_confirmed",
    "transaction_release_confirmed",
    "transaction_fa_confirmed",
    "transaction_option_confirmed",
    "transaction_injury_confirmed",
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
    # 저연차와 성적 유지·상승은 기존 원인 신호가 하나도 없는 이탈자에게만
    # 적용하는 보조 규칙이다. exp <= 1은 루키·2년 차 구간을 뜻하고,
    # 전력 변화 0 이상은 전년 대비 유지·상승이 실제 관측된 경우만 뜻한다.
    early_career_max_exp: float = 1.0
    stable_score_delta: float = 0.0


@dataclass(frozen=True)
class ReasonConfig:
    """원인 태그 생성 설정."""

    train_end_year: int = 2021
    injury_quantile: float = 0.75
    decline_quantile: float = 0.25
    career_quantile: float = 0.75
    # 이 경력(년) 이하를 "저연차"로 본다. unknown 집단의 40.3%가 여기 해당하고,
    # MLB는 40인 로스터/옵션 규정 때문에 저연차 선수의 팀 이동이 실제로 잦다.
    early_career_exp: float = 1.0

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

    if "unresolved_stints" not in out.columns:
        out["unresolved_stints"] = 0.0
    else:
        out["unresolved_stints"] = pd.to_numeric(
            out["unresolved_stints"], errors="coerce"
        ).fillna(0.0)

    return out


def merge_transaction_evidence(
    player_season: pd.DataFrame,
    transactions: pd.DataFrame | None = None,
    crosswalk: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """시즌 종료 전후의 MLB 트랜잭션을 이탈 근거로 연결한다.

    트레이드는 해당 시즌 달력연도 전체에서 확인한다. FA·방출·옵션·IL 이동은
    정규시즌 종료 무렵인 9월부터 다음 해 4월까지의 기록만 시즌 ``t``의
    오프시즌 근거로 사용한다. 같은 연도에 있었다는 이유만으로 무관한 거래를
    붙이지 않도록 실제 라벨 판정에서는 ``y_path``/``y_fa_release``를 추가
    가드로 사용한다.

    트랜잭션 자료가 없을 때도 파이프라인과 서비스 추론이 동작하도록 모든
    확인 플래그를 False로 만든다. 이 플래그들은 정답 라벨 생성에만 쓰며 모델
    입력에는 포함하지 않아 미래 거래정보 누수를 막는다.
    """

    _require_columns(player_season, KEY, name="player_season")
    out = player_season.copy()
    for column in TRANSACTION_EVIDENCE_COLUMNS:
        out[column] = False

    if transactions is None:
        return out

    tx = transactions.copy()
    _require_columns(tx, ["date", "type_desc", "description"], name="transactions")

    if "player_id" not in tx.columns:
        if crosswalk is None:
            raise ValueError("mlbam_id 트랜잭션에는 player_id crosswalk가 필요합니다.")
        _require_columns(crosswalk, ["player_id", "mlbam_id"], name="crosswalk")
        mapping = crosswalk[["player_id", "mlbam_id"]].dropna().copy()
        mapping["mlbam_id"] = pd.to_numeric(mapping["mlbam_id"], errors="coerce")
        mapping = mapping.dropna(subset=["mlbam_id"]).drop_duplicates("mlbam_id")
        tx["mlbam_id"] = pd.to_numeric(tx.get("mlbam_id"), errors="coerce")
        tx = tx.merge(mapping, on="mlbam_id", how="inner", validate="many_to_one")

    tx["date"] = pd.to_datetime(tx["date"], errors="coerce")
    tx = tx.dropna(subset=["player_id", "date"]).copy()
    if tx.empty:
        return out

    tx["type_norm"] = tx["type_desc"].fillna("").str.strip().str.casefold()
    tx["description_norm"] = tx["description"].fillna("").str.casefold()
    year = tx["date"].dt.year.astype(int)
    month = tx["date"].dt.month.astype(int)

    # 9~12월 거래는 같은 시즌의 종료 이후 근거, 1~4월 거래는 직전 시즌의
    # 오프시즌 근거로 귀속한다. 5~8월의 비트레이드 거래는 시즌 중 로스터
    # 운영일 수 있어 이탈 사유 근거에서 제외한다.
    tx["offseason_for"] = np.select(
        [month.ge(9), month.le(4)],
        [year, year - 1],
        default=np.nan,
    )

    tx["is_trade"] = tx["type_norm"].eq("trade")
    tx["is_release"] = tx["type_norm"].isin(
        {
            "released",
            "designated for assignment",
            "outrighted",
            "claimed off waivers",
        }
    )
    tx["is_fa"] = tx["type_norm"].isin(
        {"declared free agency", "signed as free agent"}
    )
    tx["is_option"] = tx["type_norm"].eq("optioned")
    tx["is_injury"] = tx["description_norm"].str.contains(
        r"injured list|disabled list", regex=True, na=False
    )

    evidence_frames: list[pd.DataFrame] = []

    trade = tx.loc[tx["is_trade"], ["player_id"]].copy()
    if not trade.empty:
        trade["season"] = year.loc[trade.index].astype(int)
        trade["transaction_trade_confirmed"] = True
        evidence_frames.append(trade)

    offseason = tx[tx["offseason_for"].notna()].copy()
    if not offseason.empty:
        offseason["season"] = offseason["offseason_for"].astype(int)
        offseason = offseason.rename(
            columns={
                "is_release": "transaction_release_confirmed",
                "is_fa": "transaction_fa_confirmed",
                "is_option": "transaction_option_confirmed",
                "is_injury": "transaction_injury_confirmed",
            }
        )
        evidence_frames.append(
            offseason[["player_id", "season"] + TRANSACTION_EVIDENCE_COLUMNS[1:]]
        )

    if not evidence_frames:
        return out

    evidence = pd.concat(evidence_frames, ignore_index=True, sort=False)
    for column in TRANSACTION_EVIDENCE_COLUMNS:
        if column not in evidence.columns:
            evidence[column] = False
        evidence[column] = evidence[column].fillna(False).astype(bool)

    evidence = (
        evidence.groupby(KEY, as_index=False)[TRANSACTION_EVIDENCE_COLUMNS]
        .max()
    )
    out = out.drop(columns=TRANSACTION_EVIDENCE_COLUMNS).merge(
        evidence,
        on=KEY,
        how="left",
        validate="one_to_one",
    )
    out[TRANSACTION_EVIDENCE_COLUMNS] = out[TRANSACTION_EVIDENCE_COLUMNS].fillna(False)
    return out


def assign_observed_departure_events(player_season: pd.DataFrame) -> pd.DataFrame:
    """관측 완료 이탈자의 실제 이동 사건을 원인 예측 라벨과 분리해 저장한다.

    거래·FA·방출은 사건 이후에 알게 되는 결과이므로 ``primary_reason``의 모델
    타깃으로 쓰면 미래정보 누수와 심각한 희소 클래스 문제가 생긴다. 대신
    ``departure_event_type``에 사후 관측값으로 보존하고, 현재 선수의 원인 모델은
    시즌 t까지 알 수 있는 부상·성적·경력 피처만 학습한다.
    """

    _require_columns(player_season, ["y_departed", "y_path"], name="player_season")
    out = player_season.copy()
    for column in TRANSACTION_EVIDENCE_COLUMNS:
        if column not in out.columns:
            out[column] = False

    departed = out["y_departed"].eq(1.0)
    y_fa_release = out.get(
        "y_fa_release",
        pd.Series(pd.NA, index=out.index, dtype="object"),
    )
    unresolved_stints = pd.to_numeric(
        out.get(
            "unresolved_stints",
            pd.Series(0.0, index=out.index, dtype="float64"),
        ),
        errors="coerce",
    ).fillna(0.0)

    event = pd.Series(pd.NA, index=out.index, dtype="object")
    event_evidence = pd.Series(pd.NA, index=out.index, dtype="object")

    def apply(mask: pd.Series, value: str, *, confirmed_column: str | None = None) -> None:
        available = departed & event.isna() & mask.fillna(False)
        event.loc[available] = value
        if confirmed_column is None:
            event_evidence.loc[available] = "strong_proxy"
        else:
            confirmed = out[confirmed_column].fillna(False).astype(bool)
            event_evidence.loc[available & confirmed] = "confirmed_event"
            event_evidence.loc[available & ~confirmed] = "strong_proxy"

    apply(
        out["y_path"].eq("trade"),
        "transaction_trade",
        confirmed_column="transaction_trade_confirmed",
    )
    apply(
        out["transaction_release_confirmed"]
        | y_fa_release.isin(["release_certain", "release_est"]),
        "roster_release_waiver",
        confirmed_column="transaction_release_confirmed",
    )
    apply(
        out["transaction_fa_confirmed"] | y_fa_release.eq("fa_est"),
        "free_agent_market",
        confirmed_column="transaction_fa_confirmed",
    )
    apply(
        out["transaction_injury_confirmed"]
        | unresolved_stints.gt(0),
        "injury_roster_move",
        confirmed_column="transaction_injury_confirmed",
    )
    apply(
        out["transaction_option_confirmed"],
        "minor_league_option",
        confirmed_column="transaction_option_confirmed",
    )
    apply(out["y_path"].eq("league_exit"), "league_exit")
    apply(departed, "unresolved_event")

    out["departure_event_type"] = event
    out["departure_event_evidence"] = event_evidence
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
    config: ReasonConfig | None = None,
) -> pd.DataFrame:
    """이탈자에게 원인 보조 태그와 근거 수준을 부여한다.

    반환되는 ``primary_reason``은 다중분류 학습용 대표 태그이고,
    ``reason_tags``에는 동시에 활성화된 모든 태그를 튜플로 보존한다.
    잔류자와 라벨 검열 행의 primary_reason은 결측이다.
    """

    cfg_early_career_exp = (config or ReasonConfig()).early_career_exp
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

    # 기존 세 요인은 비교적 직접적인 관측 근거(IL, 성적·출전 하락,
    # 리그 이탈과 베테랑 시기)를 가진다. 새 태그는 이 근거들과 겹치지 않는
    # 잔여 집단에서만 적용하여 mixed 클래스가 인위적으로 불어나는 것을 막는다.
    has_strong_reason = injury | performance | career
    early_career = (
        departed
        & ~has_strong_reason
        & out["exp"].le(thresholds.early_career_max_exp)
    )
    stable_performance = (
        departed
        & ~has_strong_reason
        & ~early_career
        & out["overall_score_delta"].notna()
        & out["overall_score_delta"].ge(thresholds.stable_score_delta)
    )

    # Rev.6: 거래·FA·방출은 사후 관측 사건(departure_event_type)으로 분리한다.
    # 원인 모델에는 시즌 t에 실제로 알 수 있는 피처만 남겨 미래정보 누수를
    # 막는다. 기존 태그에 들지 않은 관측 음수 변화는 완만한 하락, 변화량 자체가
    # 없으면 비교 이력 부족으로 분리한다.
    remaining = departed & ~has_strong_reason & ~early_career & ~stable_performance

    moderate_decline = (
        remaining
        & out["overall_score_delta"].notna()
        & out["overall_score_delta"].lt(thresholds.stable_score_delta)
    )
    remaining = remaining & ~moderate_decline

    limited_history = remaining

    flag_frame = pd.DataFrame(
        {
            "injury_associated": injury.fillna(False),
            "performance_decline": performance.fillna(False),
            "career_stage": career.fillna(False),
            "early_career_move": early_career.fillna(False),
            "stable_performance_move": stable_performance.fillna(False),
            "moderate_performance_decline": moderate_decline.fillna(False),
            "limited_history": limited_history.fillna(False),
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
        tags.append(active or ("limited_history",))

        if len(active) >= 2:
            primary.append("mixed")
        elif len(active) == 1:
            primary.append(active[0])
        else:
            primary.append("limited_history")

        primary_value = primary[-1]
        if primary_value == "limited_history":
            evidence.append("insufficient")
        else:
            evidence.append("strong_proxy")

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
    if tags == ("limited_history",) or not tags:
        return "현재 데이터로 이탈 연관 요인을 판단하기 어렵습니다."

    messages = {
        "injury_associated": "IL 등재 기록이 이탈 위험과 함께 관측됨",
        "performance_decline": "최근 성적 또는 출전 비중 하락이 관측됨",
        "career_stage": "연령·경력상 생애주기 요인이 리그 이탈과 함께 관측됨",
        "early_career_move": "저연차 구간의 선수 이동과 유사한 특성이 관측됨",
        "stable_performance_move": "전년 대비 전력이 유지·상승한 이동과 유사한 특성이 관측됨",
        "moderate_performance_decline": "강한 하락 임계에는 못 미치지만 전력 하락이 관측됨",
        "limited_history": "직전 비교 시즌이 없어 전력 변화 근거가 제한됨",
    }
    return "; ".join(messages[tag] for tag in tags if tag in messages)


def build_reason_dataset(
    player_season: pd.DataFrame,
    injury: pd.DataFrame | None = None,
    *,
    transactions: pd.DataFrame | None = None,
    crosswalk: pd.DataFrame | None = None,
    config: ReasonConfig | None = None,
    thresholds: ReasonThresholds | None = None,
) -> tuple[pd.DataFrame, ReasonThresholds]:
    """부상·트랜잭션 연결부터 원인 라벨 생성까지 한 번에 수행한다."""

    merged = merge_injury_data(player_season, injury)
    merged = merge_transaction_evidence(merged, transactions, crosswalk)
    merged = assign_observed_departure_events(merged)
    featured = add_reason_features(merged)
    fitted_thresholds = thresholds or fit_reason_thresholds(featured, config)
    labeled = assign_reason_labels(featured, fitted_thresholds, config)
    return labeled, fitted_thresholds


def to_reason_xy(reason_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """예측 가능한 원인 라벨을 모델 입력 X와 정답 y로 변환한다.

    ``limited_history``는 원인이 아니라 비교 데이터의 가용 상태다. 또한 최신
    검증구간에는 정답 표본이 없어 모델 클래스로 넣으면 macro F1만 인위적으로
    낮아진다. 이 값은 서비스에서 결측 패턴을 직접 확인해 규칙으로 표시한다.
    """

    _require_columns(reason_data, ["primary_reason"], name="reason_data")
    feature_columns = [
        column for column in MODEL_FEATURE_CANDIDATES if column in reason_data.columns
    ]
    if not feature_columns:
        raise ValueError("원인 모델에 사용할 수 있는 수치 피처가 없습니다.")

    eligible = (
        reason_data["primary_reason"].notna()
        & reason_data["primary_reason"].ne("limited_history")
    )
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
            "class_weight": "balanced_subsample",
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

    root = Path(__file__).resolve().parents[2]
    injury_path = (
        root
        / "data"
        / "final"
        / "player_injury_stints.csv"
    )
    crosswalk_path = root / "2026_.csv"
    transaction_paths = sorted((root / ".cache").glob("transactions-*.csv"))

    if not injury_path.exists():
        raise FileNotFoundError(
            f"실제 부상 데이터가 없습니다: {injury_path}"
        )

    # contract.validate()를 통과한 실제 데이터만 사용한다.
    features = load_features()
    injury = pd.read_csv(injury_path)
    crosswalk = pd.read_csv(crosswalk_path) if crosswalk_path.exists() else None
    transactions = (
        pd.concat(
            [pd.read_csv(path) for path in transaction_paths],
            ignore_index=True,
        )
        if transaction_paths
        else None
    )

    reason_data, fitted = build_reason_dataset(
        player_season=features,
        injury=injury,
        transactions=transactions,
        crosswalk=crosswalk,
    )

    X, y = to_reason_xy(reason_data)

    print("=" * 60)
    print("실제 원인 데이터 생성 완료")
    print("=" * 60)
    print(f"전체 features_v1: {len(features):,}행")
    print(f"부상 데이터: {len(injury):,}행")
    print(
        "트랜잭션 데이터: "
        f"{0 if transactions is None else len(transactions):,}행"
    )
    print(f"원인 모델 입력: X={X.shape}, y={len(y):,}")

    print("\n[primary_reason 분포]")
    reason_distribution = reason_data.loc[
        reason_data["y_departed"].eq(1.0),
        "primary_reason",
    ]
    print(
        reason_distribution.value_counts(dropna=False).to_string()
    )

    print("\n[primary_reason 비율(%)]")
    print(
        reason_distribution.value_counts(
            normalize=True,
            dropna=False,
        )
        .mul(100)
        .round(2)
        .to_string()
    )

    print("\n[학습 구간에서 계산된 임계값]")
    for name, value in threshold_metadata(fitted).items():
        print(f"{name}: {value:.4f}")

    duplicate_count = int(
        reason_data.duplicated(KEY).sum()
    )

    print(
        f"\n[player_id + season 중복]: "
        f"{duplicate_count:,}건"
    )

    # ---------------------------------------------------------
    # 시계열 학습·검증·테스트 분할
    # ---------------------------------------------------------

    season = reason_data.loc[X.index, "season"]

    train_lo, train_hi = SPLIT["train"]
    valid_lo, valid_hi = SPLIT["valid"]
    test_lo, test_hi = SPLIT["test"]

    train_mask = season.between(
        train_lo,
        train_hi,
    )
    valid_mask = season.between(
        valid_lo,
        valid_hi,
    )
    test_mask = season.between(
        test_lo,
        test_hi,
    )

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]

    X_valid = X.loc[valid_mask]
    y_valid = y.loc[valid_mask]

    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]

    print(
        f"\ntrain: {len(X_train):,}건 "
        f"({train_lo}~{train_hi})"
    )
    print(
        f"valid: {len(X_valid):,}건 "
        f"({valid_lo}~{valid_hi})"
    )
    print(
        f"test: {len(X_test):,}건 "
        f"({test_lo}~{test_hi})"
    )

    # ---------------------------------------------------------
    # RandomForest 및 MLP 학습·평가·저장
    # ---------------------------------------------------------

    for model_cls in (
        ReasonRandomForest,
        ReasonMLP,
    ):
        model = model_cls()
        model.fit(
            X_train,
            y_train,
        )

        valid_metrics = evaluate(
            model,
            X_valid,
            y_valid,
        )

        test_metrics = evaluate(
            model,
            X_test,
            y_test,
        )

        model.set_metrics(
            **test_metrics,
            valid_macro_f1=valid_metrics["macro_f1"],
            n_valid=len(y_valid),
        )

        path = model.save(
            note=(
                "실제 features_v1 및 "
                "player_injury_stints.csv/MLB 트랜잭션으로 학습 "
                "(Rev.6 잔여 unknown 재라벨링, 2026-08-30)"
            )
        )

        print(f"\n[{model.name}]")
        print(
            "valid macro_f1="
            f"{valid_metrics['macro_f1']:.4f}"
        )
        print(
            "test macro_f1="
            f"{test_metrics['macro_f1']:.4f}"
        )
        print(f"저장 경로: {path}")
