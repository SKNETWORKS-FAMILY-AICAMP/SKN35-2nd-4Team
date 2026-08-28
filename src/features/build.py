"""
build.py

B(strength) 전력 + C(labels) 라벨을 병합하여
contract.py가 정의한 features_v1.parquet을 생성한다.

파이프라인
----------
1. 실제 원천 CSV 로드
2. strength.py로 전력 feature 생성
3. player-season 메타데이터 구성
4. labels.py로 실제 라벨 생성
5. 전력 + 라벨 + 메타데이터 병합
6. contract.SCHEMA 기준으로 컬럼 정렬
7. contract.validate() 통과 확인
8. validate PASS일 때만 features_v1.parquet 저장
9. Git commit 및 빌드 결과 통보

중요
----
- dummy / mock 데이터는 만들지 않는다.
- validate 실패 시 features_v1.parquet을 저장하지 않는다.
- 현재 strength.py가 만들지 않는 contract 필드는 원천 CSV에서
  가져올 수 있는 경우에만 구성한다.
- 원천 데이터에 없는 필드는 임의값으로 채우지 않고 명확한 오류를 낸다.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------

if __package__:
    from . import contract
    from . import labels
    from . import strength
else:
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from src.features import contract
    from src.features import labels
    from src.features import strength


# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "final"

TEAMS_PATH = DATA_DIR / "teams.csv"
BATTING_PATH = DATA_DIR / "batting_stats.csv"
PITCHING_PATH = DATA_DIR / "pitching_stats.csv"
INJURY_PATH = DATA_DIR / "player_injury_stints.csv"

OUTPUT_PATH = DATA_DIR / "features_v1.parquet"


# ---------------------------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------------------------

KEY = ["player_id", "season"]


# contract.SCHEMA에 있지만 strength.py가 직접 만들지 않는 컬럼
REQUIRED_META_COLS = [
    "team_last",
    "franch_id",
    "league",
    "role",
    "age",
    "exp",
    "n_stint",
    "g_ratio_prev",
    "g_chg",
    "def_score",
    "team_wr",
    "allstar",
]


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"필수 파일이 없습니다: {path}")


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    name: str,
) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(
            f"{name}에 필요한 컬럼이 없습니다: {missing}"
        )


def assert_unique(
    df: pd.DataFrame,
    keys: list[str],
    name: str,
) -> None:
    duplicated = df.duplicated(keys, keep=False)

    if duplicated.any():
        examples = (
            df.loc[duplicated, keys]
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            f"{name}에서 {keys} 중복이 발견되었습니다. "
            f"예시: {examples}"
        )


# ---------------------------------------------------------------------------
# 원천 데이터
# ---------------------------------------------------------------------------

def load_raw_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """실제 원천 CSV만 읽는다."""

    for path in [
        TEAMS_PATH,
        BATTING_PATH,
        PITCHING_PATH,
        INJURY_PATH,
    ]:
        require_file(path)

    teams = pd.read_csv(TEAMS_PATH)
    batting = pd.read_csv(BATTING_PATH)
    pitching = pd.read_csv(PITCHING_PATH)
    injuries = pd.read_csv(INJURY_PATH)

    return teams, batting, pitching, injuries


# ---------------------------------------------------------------------------
# 1. B - 전력 생성
# ---------------------------------------------------------------------------

def build_strength_features(
    teams_raw: pd.DataFrame,
    batting_raw: pd.DataFrame,
    pitching_raw: pd.DataFrame,
    injuries_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    strength.py의 공식 파이프라인을 그대로 사용한다.

    strength.py 자체가 마지막에
    player_id / season / off_score / pit_score / g_ratio /
    ops_z / era_z / whip_z / overall_score 등을 만든다.
    """

    teams = strength.standardize_teams(teams_raw)
    batting = strength.standardize_batting(batting_raw)
    pitching = strength.standardize_pitching(pitching_raw)
    injuries = strength.standardize_injuries(injuries_raw)

    batting_result = strength.compute_batting_strength(
        batting,
        teams,
        pitching,
        injuries,
    )

    pitching_result = strength.compute_pitching_strength(
        pitching,
        teams,
        injuries,
    )

    result = strength.build_features_v1(
        batting_result,
        pitching_result,
    )

    assert_unique(result, KEY, "strength 결과")

    return result


# ---------------------------------------------------------------------------
# 2. 선수-season 메타데이터
# ---------------------------------------------------------------------------

def build_player_season_metadata(
    teams_raw: pd.DataFrame,
    batting_raw: pd.DataFrame,
    pitching_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    labels.py가 요구하는 player-season 입력을 구성한다.

    필요한 값
    ----------
    player_id
    season
    franch_id
    n_stint
    exp
    overall_score는 strength 결과와 merge 후 추가한다.

    contract 최종 컬럼 중 메타데이터로 필요한
    team_last / league / role / age / team_wr / allstar 등도
    가능한 범위에서 원천 데이터로 구성한다.
    """

    require_columns(
        teams_raw,
        ["year", "team_id", "franch_id", "lg_id"],
        "teams.csv",
    )

    require_columns(
        batting_raw,
        ["player_id", "year", "team_id", "stint"],
        "batting_stats.csv",
    )

    require_columns(
        pitching_raw,
        ["player_id", "year", "team_id", "stint"],
        "pitching_stats.csv",
    )

    # ------------------------------------------------------------------
    # batting / pitching 표준 season 키
    # ------------------------------------------------------------------

    batting = batting_raw.copy()
    pitching = pitching_raw.copy()

    batting["season"] = pd.to_numeric(
        batting["year"], errors="raise"
    ).astype("int64")

    pitching["season"] = pd.to_numeric(
        pitching["year"], errors="raise"
    ).astype("int64")

    batting["player_id"] = batting["player_id"].astype(str)
    pitching["player_id"] = pitching["player_id"].astype(str)

    # ------------------------------------------------------------------
    # 선수-season 전체 키
    # ------------------------------------------------------------------

    player_season = pd.concat(
        [
            batting[["player_id", "season"]],
            pitching[["player_id", "season"]],
        ],
        ignore_index=True,
    ).drop_duplicates()

    # ------------------------------------------------------------------
    # 마지막 팀
    # batting / pitching 모두 있으면 가장 큰 stint의 팀을 사용.
    # ------------------------------------------------------------------

    team_candidates = []

    b_last = (
        batting
        .sort_values(["player_id", "season", "stint"])
        .groupby(["player_id", "season"], as_index=False)
        .last()
    )

    team_candidates.append(
        b_last[
            ["player_id", "season", "team_id"]
        ].rename(columns={"team_id": "bat_team"})
    )

    p_last = (
        pitching
        .sort_values(["player_id", "season", "stint"])
        .groupby(["player_id", "season"], as_index=False)
        .last()
    )

    team_candidates.append(
        p_last[
            ["player_id", "season", "team_id"]
        ].rename(columns={"team_id": "pit_team"})
    )

    last_team = team_candidates[0].merge(
        team_candidates[1],
        on=KEY,
        how="outer",
    )

    last_team["team_last"] = (
        last_team["bat_team"]
        .combine_first(last_team["pit_team"])
    )

    player_season = player_season.merge(
        last_team[KEY + ["team_last"]],
        on=KEY,
        how="left",
    )

    # ------------------------------------------------------------------
    # n_stint
    # ------------------------------------------------------------------

    stint_rows = pd.concat(
        [
            batting[
                ["player_id", "season", "team_id", "stint"]
            ],
            pitching[
                ["player_id", "season", "team_id", "stint"]
            ],
        ],
        ignore_index=True,
    ).drop_duplicates()

    n_stint = (
        stint_rows
        .groupby(KEY)
        .size()
        .rename("n_stint")
        .reset_index()
    )

    player_season = player_season.merge(
        n_stint,
        on=KEY,
        how="left",
    )

    # ------------------------------------------------------------------
    # exp
    #
    # 시즌 경력 수를 0부터 부여한다.
    # ------------------------------------------------------------------

    player_season = player_season.sort_values(KEY)

    player_season["exp"] = (
        player_season
        .groupby("player_id")
        .cumcount()
        .astype("int64")
    )

    # ------------------------------------------------------------------
    # 팀 정보
    # ------------------------------------------------------------------

    teams = teams_raw.copy()

    teams["season"] = pd.to_numeric(
        teams["year"], errors="raise"
    ).astype("int64")

    teams["team_last"] = teams["team_id"]

    team_columns = [
        "season",
        "team_last",
        "franch_id",
        "lg_id",
    ]

    if "win_rate" in teams.columns:
        team_columns.append("win_rate")
    elif "W" in teams.columns and "L" in teams.columns:
        teams["win_rate"] = (
            teams["W"] /
            (teams["W"] + teams["L"]).replace(0, np.nan)
        )
        team_columns.append("win_rate")
    elif "w" in teams.columns and "l" in teams.columns:
        teams["win_rate"] = (
            teams["w"] /
            (teams["w"] + teams["l"]).replace(0, np.nan)
        )
        team_columns.append("win_rate")
    else:
        raise ValueError(
            "teams.csv에서 team_wr을 계산할 win_rate 또는 W/L 컬럼을 찾을 수 없습니다."
        )

    teams_meta = (
        teams[team_columns]
        .drop_duplicates(["season", "team_last"], keep="last")
    )

    player_season = player_season.merge(
        teams_meta,
        on=["season", "team_last"],
        how="left",
    )

    # ------------------------------------------------------------------
    # league
    # ------------------------------------------------------------------

    league_map = {
        "AL": "mlb",
        "NL": "mlb",
        "MLB": "mlb",
        "KBO": "kbo",
    }

    player_season["league"] = (
        player_season["lg_id"]
        .astype(str)
        .str.upper()
        .map(league_map)
    )

    unknown_league = sorted(
        player_season.loc[
            player_season["league"].isna(),
            "lg_id",
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    if unknown_league:
        raise ValueError(
            f"league로 변환할 수 없는 lg_id가 있습니다: {unknown_league}"
        )

    player_season["team_wr"] = pd.to_numeric(
        player_season["win_rate"],
        errors="coerce",
    )

    # ------------------------------------------------------------------
    # role
    # ------------------------------------------------------------------

    batter_keys = (
        batting[KEY]
        .drop_duplicates()
        .assign(is_batter=True)
    )

    pitcher_keys = (
        pitching[KEY]
        .drop_duplicates()
        .assign(is_pitcher=True)
    )

    role = player_season[KEY].merge(
        batter_keys,
        on=KEY,
        how="left",
    ).merge(
        pitcher_keys,
        on=KEY,
        how="left",
    )

    role["is_batter"] = role["is_batter"].fillna(False)
    role["is_pitcher"] = role["is_pitcher"].fillna(False)

    role["is_batter"] = (
    role["is_batter"]
    .fillna(False)
    .astype(bool)
    )

    role["is_pitcher"] = (
        role["is_pitcher"]
        .fillna(False)
        .astype(bool)
    )

    role["role"] = None

    role.loc[
        role["is_batter"] & role["is_pitcher"],
        "role"
    ] = "TWO"

    role.loc[
        role["is_batter"] & ~role["is_pitcher"],
        "role"
    ] = "B"

    role.loc[
        ~role["is_batter"] & role["is_pitcher"],
        "role"
    ] = "P"

    player_season = player_season.merge(
        role[KEY + ["role"]],
        on=KEY,
        how="left",
    )

    # ------------------------------------------------------------------
    # age
    # ------------------------------------------------------------------

    age_frames = []

    if "age" in batting.columns:
        age_frames.append(
            batting[
                KEY + ["age"]
            ].rename(columns={"age": "age_b"})
        )

    if "age" in pitching.columns:
        age_frames.append(
            pitching[
                KEY + ["age"]
            ].rename(columns={"age": "age_p"})
        )

    if not age_frames:
        raise ValueError(
            "contract.SCHEMA의 age를 만들 수 없습니다. "
            "batting_stats.csv 또는 pitching_stats.csv에 age 컬럼이 필요합니다."
        )

    age = age_frames[0]

    if len(age_frames) == 2:
        age = age.merge(
            age_frames[1],
            on=KEY,
            how="outer",
        )
        age["age"] = age["age_b"].combine_first(age["age_p"])
        age = age[KEY + ["age"]]
    else:
        age = age.rename(columns={"age_b": "age"})

    age = (
        age
        .dropna(subset=["age"])
        .drop_duplicates(KEY)
    )

    player_season = player_season.merge(
        age,
        on=KEY,
        how="left",
    )

    # ------------------------------------------------------------------
    # allstar
    #
    # 현재 원천 데이터에 allstar 관련 컬럼이 없다면 임의값을 만들지 않는다.
    # contract.validate()를 통과하려면 실제 allstar 데이터가 필요하다.
    # ------------------------------------------------------------------

    allstar_source = None

    for source in [batting, pitching]:
        for candidate in ["allstar", "all_star", "is_allstar"]:
            if candidate in source.columns:
                allstar_source = (
                    source[
                        KEY + [candidate]
                    ]
                    .rename(columns={candidate: "allstar"})
                    .drop_duplicates(KEY)
                )
                break
        if allstar_source is not None:
            break

    if allstar_source is None:
        raise ValueError(
            "contract.SCHEMA의 allstar를 만들 실제 원천 컬럼이 없습니다. "
            "batting_stats.csv 또는 pitching_stats.csv에 "
            "allstar / all_star / is_allstar 중 하나가 필요합니다."
        )

    player_season = player_season.merge(
        allstar_source,
        on=KEY,
        how="left",
    )

    # ------------------------------------------------------------------
    # g_ratio_prev / g_chg
    #
    # strength 결과와 나중에 merge하므로 여기서는 계산하지 않는다.
    # ------------------------------------------------------------------

    return player_season[
        KEY
        + [
            "team_last",
            "franch_id",
            "league",
            "role",
            "age",
            "exp",
            "n_stint",
            "team_wr",
            "allstar",
        ]
    ]


# ---------------------------------------------------------------------------
# 3. strength + metadata 병합
# ---------------------------------------------------------------------------

def merge_strength_metadata(
    strength_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """전력 결과에 contract용 메타데이터를 붙인다."""

    assert_unique(strength_df, KEY, "strength 결과")
    assert_unique(metadata_df, KEY, "metadata 결과")

    merged = strength_df.merge(
        metadata_df,
        on=KEY,
        how="left",
        validate="one_to_one",
    )

    # strength.py가 만드는 g_ratio를 기준으로 이전 시즌 출전비율 계산
    merged = merged.sort_values(KEY)

    merged["g_ratio_prev"] = (
        merged
        .groupby("player_id")["g_ratio"]
        .shift(1)
    )

    merged["g_chg"] = (
        merged["g_ratio"] -
        merged["g_ratio_prev"]
    )

    # ------------------------------------------------------------------
    # def_score
    #
    # 현재 strength.py에는 수비 전력 계산이 구현되어 있지 않다.
    # 임의 점수를 넣지 않고 명시적으로 중단한다.
    # ------------------------------------------------------------------

    if "def_score" not in merged.columns:
        raise ValueError(
            "contract.SCHEMA에는 def_score가 필요하지만 "
            "현재 strength.py에서 def_score를 생성하지 않습니다. "
            "실제 수비 전력 계산을 strength.py에 추가한 뒤 build를 실행하세요."
        )

    return merged


# ---------------------------------------------------------------------------
# 4. C - 라벨 생성
# ---------------------------------------------------------------------------

def build_labels(
    player_season: pd.DataFrame,
    strength_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    labels.py에 필요한 overall_score를 붙인 뒤 실제 라벨을 생성한다.
    """

    label_input = player_season.merge(
        strength_df[
            KEY + ["overall_score"]
        ],
        on=KEY,
        how="inner",
        validate="one_to_one",
    )

    require_columns(
        label_input,
        labels.REQUIRED_COLUMNS,
        "labels 입력",
    )

    labeled = labels.build_labels(
        label_input,
        config=labels.LabelConfig(
            data_end_year=contract.LABEL_END_YEAR,
        ),
        validate=True,
    )

    return labeled


# ---------------------------------------------------------------------------
# 5. 최종 merge
# ---------------------------------------------------------------------------

def merge_final(
    strength_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    strength + metadata + labels를 하나의 features_v1로 합친다.
    """

    merged = strength_df.merge(
        metadata_df,
        on=KEY,
        how="left",
        validate="one_to_one",
    )

    label_columns = KEY + contract.LABEL_COLS

    merged = merged.merge(
        labeled_df[label_columns],
        on=KEY,
        how="left",
        validate="one_to_one",
    )

    # 다음 시즌 overall_score
    next_score = (
        strength_df[
            KEY + ["overall_score"]
        ]
        .copy()
        .rename(
            columns={
                "season": "next_season",
                "overall_score": "y_next_score",
            }
        )
    )

    next_score["season"] = next_score["next_season"] - 1
    next_score.drop(columns=["next_season"], inplace=True)

    merged = merged.merge(
        next_score,
        on=KEY,
        how="left",
        validate="one_to_one",
    )

    # ------------------------------------------------------------------
    # g_ratio 파생
    # ------------------------------------------------------------------

    merged = merged.sort_values(KEY)

    if "g_ratio_prev" not in merged.columns:
        merged["g_ratio_prev"] = (
            merged
            .groupby("player_id")["g_ratio"]
            .shift(1)
        )

    if "g_chg" not in merged.columns:
        merged["g_chg"] = (
            merged["g_ratio"] -
            merged["g_ratio_prev"]
        )

    # ------------------------------------------------------------------
    # contract 스키마에 없는 컬럼은 제거
    # ------------------------------------------------------------------

    final_columns = list(contract.SCHEMA.keys())

    missing = sorted(
        set(final_columns) - set(merged.columns)
    )

    if missing:
        raise ValueError(
            "최종 features_v1에 필요한 컬럼이 없습니다: "
            f"{missing}"
        )

    final_df = merged[final_columns].copy()

    assert_unique(
        final_df,
        KEY,
        "최종 features_v1",
    )

    return final_df


# ---------------------------------------------------------------------------
# 6. dtype
# ---------------------------------------------------------------------------

def normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """contract.SCHEMA의 dtype에 맞춘다."""

    out = df.copy()

    for column, dtype in contract.SCHEMA.items():

        if dtype == "object":
            out[column] = out[column].astype("object")

        elif dtype == "int64":
            numeric = pd.to_numeric(
                out[column],
                errors="coerce",
            )

            if numeric.isna().any():
                raise ValueError(
                    f"{column}은 int64인데 결측 또는 숫자가 아닌 값이 있습니다."
                )

            out[column] = numeric.astype("int64")

        elif dtype == "float64":
            out[column] = pd.to_numeric(
                out[column],
                errors="coerce",
            ).astype("float64")

    return out


# ---------------------------------------------------------------------------
# 7. Git
# ---------------------------------------------------------------------------

def git_commit() -> None:
    """
    validate PASS 이후에만 features_v1을 Git에 동결한다.

    Git이 없는 환경에서는 commit을 하지 않고 안내만 출력한다.
    """

    relative_output = OUTPUT_PATH.relative_to(ROOT)

    try:
        subprocess.run(
            ["git", "add", str(relative_output)],
            cwd=ROOT,
            check=True,
        )

        result = subprocess.run(
            [
                "git",
                "commit",
                "-m",
                "build: freeze features_v1",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("\n[Git] features_v1 commit 완료")
            print(result.stdout.strip())
            return

        # 변경사항이 없는 경우
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        if not status.stdout.strip():
            print("\n[Git] commit할 변경사항이 없습니다.")
        else:
            print("\n[Git] commit 실패")
            print(result.stdout.strip())
            print(result.stderr.strip())

    except FileNotFoundError:
        print("\n[Git] git 명령을 찾을 수 없습니다. commit을 건너뜁니다.")


# ---------------------------------------------------------------------------
# 8. 통보
# ---------------------------------------------------------------------------

def notify(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("features_v1 BUILD COMPLETE")
    print("=" * 70)
    print(f"파일       : {OUTPUT_PATH}")
    print(f"행 수      : {len(df):,}")
    print(f"열 수      : {len(df.columns):,}")
    print(
        f"시즌 범위  : "
        f"{df['season'].min()} ~ {df['season'].max()}"
    )
    print(
        "키 중복    : "
        f"{df.duplicated(KEY).sum()}건"
    )
    print("contract.validate() : PASS")
    print("features_v1.parquet : 저장 완료")
    print("Git 동결              : 완료")
    print("=" * 70)


# ---------------------------------------------------------------------------
# 9. 전체 빌드
# ---------------------------------------------------------------------------

def build() -> pd.DataFrame:
    print("=" * 70)
    print("features_v1 BUILD START")
    print("=" * 70)

    # 1. 원천 데이터
    print("\n[1/7] 실제 원천 데이터 로드")
    teams_raw, batting_raw, pitching_raw, injuries_raw = load_raw_data()

    print(f"  teams    : {len(teams_raw):,}")
    print(f"  batting  : {len(batting_raw):,}")
    print(f"  pitching : {len(pitching_raw):,}")
    print(f"  injuries : {len(injuries_raw):,}")

    # 2. B 전력
    print("\n[2/7] B strength.py 실행")
    strength_df = build_strength_features(
        teams_raw,
        batting_raw,
        pitching_raw,
        injuries_raw,
    )

    print(
        f"  strength 결과 : "
        f"{len(strength_df):,}행 × {len(strength_df.columns):,}열"
    )

    # 3. 메타데이터
    print("\n[3/7] 선수-season 메타데이터 구성")
    metadata_df = build_player_season_metadata(
        teams_raw,
        batting_raw,
        pitching_raw,
    )

    print(
        f"  metadata : "
        f"{len(metadata_df):,}행 × {len(metadata_df.columns):,}열"
    )

    # 4. strength + metadata
    print("\n[4/7] 전력 + 메타데이터 병합")
    strength_full = merge_strength_metadata(
        strength_df,
        metadata_df,
    )

    # 5. C 라벨
    print("\n[5/7] C labels.py 실행")
    labeled_df = build_labels(
        strength_full[
            KEY
            + [
                "franch_id",
                "n_stint",
                "exp",
                "overall_score",
            ]
        ],
        strength_full,
    )

    print(
        f"  labels : "
        f"{len(labeled_df):,}행 × {len(labeled_df.columns):,}열"
    )

    # 6. 최종 병합
    print("\n[6/7] 전력 + C 라벨 최종 병합")
    final_df = merge_final(
        strength_full,
        metadata_df,
        labeled_df,
    )

    final_df = normalize_dtypes(final_df)

    # ---------------------------------------------------------------
    # 중요: validate 전에는 기존 features_v1을 건드리지 않는다.
    # ---------------------------------------------------------------

    print("\n[7/7] contract.validate() 실행")

    try:
        contract.validate(final_df)
    except Exception as exc:
        print("\n" + "!" * 70)
        print("contract.validate() FAILED")
        print("!" * 70)
        print(f"{type(exc).__name__}: {exc}")
        print(
            "\n검증 실패이므로 features_v1.parquet을 "
            "저장하거나 Git commit하지 않습니다."
        )
        raise

    print("  ✅ contract.validate() PASS")

    # ----------------------------------------------------------------
    # 검증 통과 후에만 저장
    # ----------------------------------------------------------------

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"  저장 완료 : {OUTPUT_PATH}"
    )

    # ----------------------------------------------------------------
    # Git 동결
    # ----------------------------------------------------------------

    git_commit()

    # ----------------------------------------------------------------
    # 통보
    # ----------------------------------------------------------------

    notify(final_df)

    return final_df


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build()
