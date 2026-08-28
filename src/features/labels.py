"""선수 시즌 단위 이탈 라벨 생성 및 검증.

이 모듈은 ``docs/label_spec.md``와 ``src/features/contract.py``의 Rev.4 계약을
구현한다.

라벨 계층
---------
L1 ``y_departed``
    다음 시즌에도 같은 ``franch_id``에 소속되는지 여부.
L2 ``y_path``
    이탈자에게만 부여하는 관측 경로: trade / offseason_move / league_exit.
L2b ``y_fa_release``
    offseason_move에만 부여하는 FA·방출 규칙 기반 분류.
L3 ``y_returned``
    league_exit 이후 지정된 시즌 안에 리그 기록이 다시 등장하는지 여부.
L1' ``y_core_departed`` (추가)
    구단이 결정한 방출(release_certain)을 제외한 "선수 주도 이탈위험" 타깃.
    ``departure.py`` 등 핵심 이탈위험 예측 모델은 ``y_departed`` 대신 이 컬럼을
    사용해야 한다. 그렇지 않으면 구단이 내보낸 선수를 "이탈 위험이 높은 선수"로
    학습하는 순환논리가 발생한다(설계원칙, docs/label_spec.md 참고).

부상은 관측 정답 라벨이 아니라 원인 보조 정보이므로 이 파일에서 라벨을
변경하는 데 사용하지 않는다. 부상 관련 설명은 ``src/models/reason.py``가
담당한다.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

if __package__:
    from . import contract as _contract
else:  # VS Code에서 이 파일을 직접 실행하는 경우도 지원한다.
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from src.features import contract as _contract

FA_ELIGIBLE_EXP = _contract.FA_ELIGIBLE_EXP
FA_RELEASE_CLASSES = _contract.FA_RELEASE_CLASSES
PATH_CLASSES = _contract.PATH_CLASSES


KEY = ["player_id", "season"]
# y_core_departed는 contract.py의 정식 L1' 타깃이며 핵심 이탈위험 모델이 사용한다.
LABEL_COLUMNS = list(_contract.LABEL_COLS)
REQUIRED_COLUMNS = [
    "player_id",
    "season",
    "franch_id",
    "n_stint",
    "exp",
    "overall_score",
]


@dataclass(frozen=True)
class LabelConfig:
    """라벨 생성 규칙.

    Attributes
    ----------
    data_end_year:
        원천 데이터에서 완전히 관측된 마지막 시즌. ``None``이면 입력 데이터의
        최대 season을 사용한다.
    fa_eligible_exp:
        프로젝트에서 사용하는 FA 자격 경력 시즌 기준.
    release_score_drop:
        현재 시즌 overall_score - 직전 관측 시즌 overall_score가 이 값보다
        작으면 ``release_est``로 분류한다.
    return_offsets:
        league_exit 시즌을 t라고 할 때 복귀를 탐색할 시즌 간격. Rev.4 기본값은
        t+2와 t+3이다.
    """

    data_end_year: int | None = None
    fa_eligible_exp: int = FA_ELIGIBLE_EXP
    release_score_drop: float = -5.0
    return_offsets: tuple[int, ...] = (2, 3)

    def __post_init__(self) -> None:
        offsets = tuple(sorted(set(self.return_offsets)))
        if not offsets or any(offset < 1 for offset in offsets):
            raise ValueError("return_offsets는 1 이상의 정수를 하나 이상 포함해야 합니다.")
        object.__setattr__(self, "return_offsets", offsets)


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"라벨 생성에 필요한 컬럼이 없습니다: {missing}")


def _prepare_player_season(df: pd.DataFrame) -> pd.DataFrame:
    """입력을 복사하고 키·판정 컬럼의 기본 품질을 검사한다."""

    _require_columns(df, REQUIRED_COLUMNS)
    out = df.copy()

    if out.empty:
        raise ValueError("라벨을 생성할 선수 시즌 데이터가 비어 있습니다.")
    if out[KEY].isna().any().any():
        raise ValueError("player_id와 season에는 결측값이 있을 수 없습니다.")
    if out.duplicated(KEY).any():
        examples = out.loc[out.duplicated(KEY, keep=False), KEY].head(5).to_dict("records")
        raise ValueError(f"player_id + season 중복이 있습니다. 예시: {examples}")
    if out[["franch_id", "n_stint", "exp"]].isna().any().any():
        raise ValueError("franch_id, n_stint, exp에는 결측값이 있을 수 없습니다.")

    for col in ("season", "n_stint", "exp"):
        numeric = pd.to_numeric(out[col], errors="coerce")
        if numeric.isna().any():
            raise ValueError(f"{col}에는 숫자로 변환할 수 없는 값이 있습니다.")
        out[col] = numeric.astype("int64")

    if (out["n_stint"] < 1).any():
        raise ValueError("n_stint는 1 이상이어야 합니다.")
    if (out["exp"] < 0).any():
        raise ValueError("exp는 0 이상이어야 합니다.")

    return out.sort_values(KEY, kind="stable").reset_index(drop=True)


def _resolve_end_year(df: pd.DataFrame, config: LabelConfig) -> int:
    observed_end = int(df["season"].max())
    if config.data_end_year is None:
        return observed_end
    if config.data_end_year > observed_end:
        raise ValueError(
            "data_end_year가 입력 데이터의 최대 season보다 큽니다: "
            f"{config.data_end_year} > {observed_end}"
        )
    return int(config.data_end_year)


def build_labels(
    player_season: pd.DataFrame,
    config: LabelConfig | None = None,
    *,
    validate: bool = True,
) -> pd.DataFrame:
    """기존 선수 시즌 데이터에 L1/L2/L2b/L3 라벨을 생성한다.

    기존 라벨 컬럼이 있으면 Rev.4 규칙으로 다시 계산해 덮어쓴다. 반환 데이터는
    ``player_id, season`` 순으로 정렬된 복사본이며 입력 데이터는 변경하지 않는다.

    Parameters
    ----------
    player_season:
        최소한 ``REQUIRED_COLUMNS``를 포함하는 선수 시즌 데이터.
    config:
        라벨 생성 설정. 생략하면 입력의 최대 시즌과 Rev.4 기본 규칙을 사용한다.
    validate:
        생성 후 계층·값·검열 규칙을 검사할지 여부.
    """

    cfg = config or LabelConfig()
    out = _prepare_player_season(player_season)
    data_end_year = _resolve_end_year(out, cfg)

    grouped = out.groupby("player_id", sort=False)
    next_season = grouped["season"].shift(-1)
    next_franch = grouped["franch_id"].shift(-1)
    previous_score = grouped["overall_score"].shift(1)

    has_next_season = next_season.eq(out["season"] + 1)
    same_franchise = has_next_season & next_franch.eq(out["franch_id"])
    l1_observable = out["season"] < data_end_year

    # L1: 다음 시즌 동일 프랜차이즈가 아니면 이탈. 마지막 관측 시즌은 검열한다.
    out["y_departed"] = np.where(
        l1_observable,
        (~same_franchise).astype(float),
        np.nan,
    )

    # L2: 이탈자에게만 관측 경로를 부여한다.
    #
    # 주의: Lahman에는 공식 거래(Transactions) 테이블이 없어 "trade"를 직접
    # 확인할 방법이 없다. 여기서는 "이번 시즌 안에 이미 스틴트가 2개 이상이었다
    # (시즌 중 팀 이동을 실제로 겪었다)"를 근거로 삼는 추정치이며, 시즌 t와
    # t+1 사이의 이동 방식이 진짜 트레이드였는지를 확인해주지는 않는다. 팀
    # 리뷰에서 이 한계를 문서화하기로 함(C파트_진행상황_검토 참고).
    #
    # 이전 버전 버그: has_next_season 조건 없이 n_stint만으로 is_trade를
    # 판정했더니, 마지막 시즌에 시즌 중 트레이드를 겪고 그대로 리그를 완전히
    # 떠난(다음 시즌 기록 자체가 없는) 선수까지 "trade"로 잘못 분류되고
    # league_exit·y_returned 계산에서 통째로 빠지는 문제가 있었다. has_next_season을
    # 추가해 "다음 시즌 기록이 아예 없는 경우"는 항상 league_exit로 가도록 고쳤다.
    departed = out["y_departed"].eq(1.0)
    is_trade = departed & out["n_stint"].ge(2) & has_next_season
    is_offseason_move = departed & ~is_trade & has_next_season
    is_league_exit = departed & ~is_trade & ~has_next_season

    out["y_path"] = pd.Series(pd.NA, index=out.index, dtype="object")
    out.loc[is_trade, "y_path"] = "trade"
    out.loc[is_offseason_move, "y_path"] = "offseason_move"
    out.loc[is_league_exit, "y_path"] = "league_exit"

    # L2b: 오프시즌 이적자만 FA·방출을 구분한다.
    score_change = out["overall_score"] - previous_score
    is_l2b = out["y_path"].eq("offseason_move")
    release_certain = is_l2b & out["exp"].lt(cfg.fa_eligible_exp)
    release_est = (
        is_l2b
        & out["exp"].ge(cfg.fa_eligible_exp)
        & score_change.lt(cfg.release_score_drop)
    )
    fa_est = is_l2b & ~release_certain & ~release_est

    out["y_fa_release"] = pd.Series(pd.NA, index=out.index, dtype="object")
    out.loc[release_certain, "y_fa_release"] = "release_certain"
    out.loc[release_est, "y_fa_release"] = "release_est"
    out.loc[fa_est, "y_fa_release"] = "fa_est"

    # L3: 관찰된 복귀는 즉시 1, 관찰 창이 끝난 미복귀만 0, 나머지는 NaN.
    seasons_by_player = grouped["season"].agg(lambda values: set(values)).to_dict()
    max_return_offset = max(cfg.return_offsets)
    returned_values: list[float] = []

    for player_id, season, path in out[["player_id", "season", "y_path"]].itertuples(
        index=False, name=None
    ):
        if pd.isna(path) or path != "league_exit":
            returned_values.append(np.nan)
            continue

        candidate_seasons = {season + offset for offset in cfg.return_offsets}
        returned = bool(seasons_by_player[player_id] & candidate_seasons)
        followup_complete = season + max_return_offset <= data_end_year

        if returned:
            returned_values.append(1.0)
        elif followup_complete:
            returned_values.append(0.0)
        else:
            returned_values.append(np.nan)

    out["y_returned"] = pd.Series(returned_values, index=out.index, dtype="float64")

    # y_core_departed: departure.py 등 "선수 주도 이탈위험"을 예측하는 핵심
    # 모델의 타깃. y_departed를 그대로 쓰면 trade/offseason_move/league_exit은
    # 물론 구단이 결정한 방출(release_certain)까지 전부 "이탈"로 묶여
    # 순환논리가 생긴다(선수가 떠날 위험 vs 구단이 내보낼 정책을 분리한다는
    # 설계원칙 위반). release_certain 행만 결측 처리해 핵심 타깃에서 제외한다.
    # release_est/fa_est는 아직 확정할 수 없는 추정치이므로, 더 보수적으로
    # 가려낼지는 팀이 별도로 정할 사항이며 여기서는 y_departed 값을 그대로 둔다.
    confirmed_release = out["y_fa_release"].eq("release_certain")
    out["y_core_departed"] = out["y_departed"].where(~confirmed_release, np.nan)

    if validate:
        validate_labels(out, LabelConfig(
            data_end_year=data_end_year,
            fa_eligible_exp=cfg.fa_eligible_exp,
            release_score_drop=cfg.release_score_drop,
            return_offsets=cfg.return_offsets,
        ))

    return out


def validate_labels(
    labeled: pd.DataFrame,
    config: LabelConfig | None = None,
) -> None:
    """라벨 값, 계층 관계, 관찰기간 검열 규칙을 검사한다.

    규칙 위반 시 ``ValueError``를 발생시키며, 성공 시 반환값은 없다.
    """

    _require_columns(labeled, REQUIRED_COLUMNS + LABEL_COLUMNS)
    df = labeled.sort_values(KEY, kind="stable").reset_index(drop=True)
    cfg = config or LabelConfig()
    data_end_year = _resolve_end_year(df, cfg)

    if df.duplicated(KEY).any():
        raise ValueError("player_id + season 중복이 있습니다.")

    allowed_departed = {0.0, 1.0}
    bad_departed = set(df["y_departed"].dropna().astype(float)) - allowed_departed
    if bad_departed:
        raise ValueError(f"정의되지 않은 y_departed 값: {sorted(bad_departed)}")

    bad_paths = set(df["y_path"].dropna()) - set(PATH_CLASSES)
    if bad_paths:
        raise ValueError(f"정의되지 않은 y_path 값: {sorted(bad_paths)}")

    bad_l2b = set(df["y_fa_release"].dropna()) - set(FA_RELEASE_CLASSES)
    if bad_l2b:
        raise ValueError(f"정의되지 않은 y_fa_release 값: {sorted(bad_l2b)}")

    observable_l1 = df["season"] < data_end_year
    censored_l1 = ~observable_l1
    if df.loc[observable_l1, "y_departed"].isna().any():
        raise ValueError("L1 판정 가능 구간에 y_departed 결측이 있습니다.")
    if df.loc[censored_l1, ["y_departed", "y_path", "y_fa_release"]].notna().any().any():
        raise ValueError("마지막 관측 시즌의 L1/L2/L2b는 결측이어야 합니다.")

    stayed = df["y_departed"].eq(0.0)
    departed = df["y_departed"].eq(1.0)
    if df.loc[stayed, "y_path"].notna().any():
        raise ValueError("잔류자에게 y_path가 존재합니다.")
    if df.loc[departed, "y_path"].isna().any():
        raise ValueError("이탈자에게 y_path가 없습니다.")

    offseason = df["y_path"].eq("offseason_move")
    if df.loc[offseason, "y_fa_release"].isna().any():
        raise ValueError("offseason_move 행에 y_fa_release가 없습니다.")
    if df.loc[~offseason, "y_fa_release"].notna().any():
        raise ValueError("offseason_move가 아닌 행에 y_fa_release가 존재합니다.")

    release_certain = df["y_fa_release"].eq("release_certain")
    if df.loc[release_certain, "exp"].ge(cfg.fa_eligible_exp).any():
        raise ValueError("release_certain 행의 exp가 FA 자격 기준 이상입니다.")

    # trade는 반드시 다음 시즌 기록이 있는(has_next_season) 이탈자여야 한다.
    # league_exit(다음 시즌 기록 자체가 없음)와 겹치면 안 된다.
    trade = df["y_path"].eq("trade")
    league_exit = df["y_path"].eq("league_exit")
    if (trade & league_exit).any():
        raise ValueError("trade와 league_exit가 동시에 참인 행이 있습니다.")

    bad_returned = set(df["y_returned"].dropna().astype(float)) - {0.0, 1.0}
    if bad_returned:
        raise ValueError(f"정의되지 않은 y_returned 값: {sorted(bad_returned)}")

    if df.loc[~league_exit, "y_returned"].notna().any():
        raise ValueError("league_exit가 아닌 행에 y_returned가 존재합니다.")

    seasons_by_player = df.groupby("player_id")["season"].agg(lambda values: set(values)).to_dict()
    max_return_offset = max(cfg.return_offsets)
    for player_id, season, returned in df.loc[
        league_exit, ["player_id", "season", "y_returned"]
    ].itertuples(index=False, name=None):
        candidates = {season + offset for offset in cfg.return_offsets}
        observed_return = bool(seasons_by_player[player_id] & candidates)
        followup_complete = season + max_return_offset <= data_end_year

        if observed_return and returned != 1.0:
            raise ValueError(f"관측된 복귀가 y_returned=1이 아닙니다: {player_id}, {season}")
        if not observed_return and followup_complete and returned != 0.0:
            raise ValueError(f"관찰 완료 미복귀가 y_returned=0이 아닙니다: {player_id}, {season}")
        if not observed_return and not followup_complete and pd.notna(returned):
            raise ValueError(f"관찰기간 부족 행의 y_returned는 결측이어야 합니다: {player_id}, {season}")

    # y_core_departed: release_certain 행만 결측이고, 나머지는 y_departed와
    # 동일해야 한다.
    bad_core = set(df["y_core_departed"].dropna().astype(float)) - allowed_departed
    if bad_core:
        raise ValueError(f"정의되지 않은 y_core_departed 값: {sorted(bad_core)}")
    if df.loc[release_certain, "y_core_departed"].notna().any():
        raise ValueError("release_certain 행의 y_core_departed는 결측이어야 합니다.")

    non_release = ~release_certain
    same_as_departed = (df["y_core_departed"] == df["y_departed"]) | (
        df["y_core_departed"].isna() & df["y_departed"].isna()
    )
    if not same_as_departed[non_release].all():
        raise ValueError("release_certain이 아닌 행의 y_core_departed가 y_departed와 다릅니다.")


def label_summary(labeled: pd.DataFrame) -> dict[str, pd.Series]:
    """팀 점검용 라벨별 건수 요약을 반환한다."""

    _require_columns(labeled, LABEL_COLUMNS)
    return {
        column: labeled[column].value_counts(dropna=False)
        for column in LABEL_COLUMNS
    }

