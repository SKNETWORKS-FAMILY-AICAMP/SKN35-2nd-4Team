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
FIELDING_PATH = DATA_DIR / "fielding_stats.csv"
INJURY_PATH = DATA_DIR / "player_injury_stints.csv"
PLAYERS_PATH = DATA_DIR / "players.csv"  # age(=season-birth_year) 계산용

OUTPUT_PATH = DATA_DIR / "features_v1.parquet"


# ---------------------------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------------------------

KEY = ["player_id", "season"]


# contract.SCHEMA에 있지만 strength.py가 직접 만들지 않는 컬럼
# def_score는 더 이상 여기 없다 - strength.compute_fielding_strength()가 만들고
# merge_strength_metadata()가 strength_df에서 그대로 가져온다(metadata_df 출처 아님).
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
    "team_wr",
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
    pd.DataFrame,
    pd.DataFrame,
]:
    """실제 원천 CSV만 읽는다."""

    for path in [
        TEAMS_PATH,
        BATTING_PATH,
        PITCHING_PATH,
        FIELDING_PATH,
        INJURY_PATH,
        PLAYERS_PATH,
    ]:
        require_file(path)

    teams = pd.read_csv(TEAMS_PATH)
    batting = pd.read_csv(BATTING_PATH)
    pitching = pd.read_csv(PITCHING_PATH)
    fielding = pd.read_csv(FIELDING_PATH)
    injuries = pd.read_csv(INJURY_PATH)
    players = pd.read_csv(PLAYERS_PATH)

    return teams, batting, pitching, fielding, injuries, players


# ---------------------------------------------------------------------------
# 1. B - 전력 생성
# ---------------------------------------------------------------------------

def build_strength_features(
    teams_raw: pd.DataFrame,
    batting_raw: pd.DataFrame,
    pitching_raw: pd.DataFrame,
    fielding_raw: pd.DataFrame,
    injuries_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    strength.py의 공식 파이프라인을 그대로 사용한다.

    strength.py 자체가 마지막에
    player_id / season / off_score / pit_score / fielding_strength_before_pt /
    g_ratio / ops_z / era_z / whip_z / overall_score 등을 만든다.
    (fielding_strength_before_pt는 merge_strength_metadata에서 def_score로 옮겨진다.)
    """

    teams = strength.standardize_teams(teams_raw)
    batting = strength.standardize_batting(batting_raw)
    pitching = strength.standardize_pitching(pitching_raw)
    fielding = strength.standardize_fielding(fielding_raw)
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

    fielding_result = strength.compute_fielding_strength(
        fielding,
        teams,
        pitching,
        injuries,
    )

    result = strength.build_features_v1(
        batting_result,
        pitching_result,
        fielding_result,
    )

    # primary_position은 전력 점수가 아니라 포지션 메타데이터라 build_features_v1
    # 안이 아니라 여기서 별도로 붙인다 - "같은 포지션 후보 추천"에 쓰인다.
    primary_position = strength.derive_primary_position(fielding).rename(
        columns={"playerID": "player_id", "yearID": "season"}
    )
    result = result.merge(primary_position, on=KEY, how="left")

    assert_unique(result, KEY, "strength 결과")

    return result


# ---------------------------------------------------------------------------
# 2. 선수-season 메타데이터
# ---------------------------------------------------------------------------

def build_player_season_metadata(
    teams_raw: pd.DataFrame,
    batting_raw: pd.DataFrame,
    pitching_raw: pd.DataFrame,
    players_raw: pd.DataFrame,
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
    team_last / league / role / age / team_wr 등도
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
    # age — players.csv의 birth_year로 계산한다 (season - birth_year).
    # batting/pitching 원본엔 age 컬럼 자체가 없다(실측 확인됨). 정확한 생일
    # 기준은 아니고 "그 해 나이"로 근사하는 것 — 실전 피처로 쓰기로 했으므로
    # (contract.py) 결측 없이 채워야 한다.
    # ------------------------------------------------------------------

    players = players_raw.copy()
    players["player_id"] = players["player_id"].astype(str)
    birth_year = players[["player_id", "birth_year"]].dropna().drop_duplicates("player_id")

    player_season = player_season.merge(birth_year, on="player_id", how="left")
    player_season["age"] = player_season["season"] - player_season["birth_year"]

    missing_age = sorted(
        player_season.loc[player_season["age"].isna(), "player_id"].unique()
    )
    if missing_age:
        raise ValueError(
            f"players.csv에 birth_year가 없어 age를 못 만든 선수 {len(missing_age)}명: "
            f"{missing_age[:10]}{'...' if len(missing_age) > 10 else ''}"
        )
    player_season = player_season.drop(columns=["birth_year"])

    # ------------------------------------------------------------------
    # allstar — contract.SCHEMA에서 뺐다(사용 안 하기로 결정, D 확정). 여기서
    # 만들지 않는다.
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # strength.py는 라만 호환 이름(playing_time_ratio, batting_strength_*  등)만
    # 만들고 contract 이름(g_ratio/off_score/pit_score/overall_score/*_z)으로
    # 바꿔주는 단계가 없었다 — 여기서 만든다. E의 recommend.adapt_features_v1()과
    # 동일한 공식을 쓴다(off_score/pit_score는 출전·부상 반영 *전* 스킬 점수;
    # g_ratio가 출전 비중을 별도로 담당하므로 겹치지 않게 하기 위한 결정 — D 확정).
    # ------------------------------------------------------------------

    # g_ratio: contract.py 정의(G/team_games) 그대로 - 타자는 playing_time_ratio
    # (G/season_games)를 쓰고, 투수는 g_ratio_games(G_pit/season_games)를 쓴다.
    # 예전엔 투수 쪽에 playing_time_ratio_pit(IP/팀총IP)을 그대로 넣었는데,
    # 이건 이닝 점유율이라 "몇 경기에 나왔나"와 단위가 다르다 - 마무리투수처럼
    # 경기는 많이 나오지만(G 높음) 이닝은 짧은(IP 점유율 낮음) 선수가 "거의 안
    # 뛴 선수"로 왜곡됐다(실측 확인 - 2022년 이후 투수 g_ratio 중앙값이 절반
    # 가까이 떨어짐, 팀당 투수 수가 늘어난 추세와 겹쳐 더 심해짐). 점수
    # 스케일링(pitching_strength)에는 여전히 이닝 점유율을 쓴다 - 거기서는
    # 맞는 기준이다.
    merged["g_ratio"] = merged[["playing_time_ratio", "g_ratio_games"]].max(
        axis=1, skipna=True
    )
    merged["off_score"] = merged["batting_strength_before_pt"]
    merged["pit_score"] = merged["pitching_strength_before_pt"]

    # 투타겸업(off_score/pit_score 둘 다 존재 = role"TWO"와 동치)은 평균을 내면
    # "타격도 투구도 어중간한" 실체 없는 점수가 나온다(2026-08-28 팀 결정) — 두
    # 역할 중 진짜 의미있게 뛴 쪽을 그 선수의 전력으로 본다.
    #
    # playing_time_ratio(타격 G/팀경기)와 playing_time_ratio_pit(투구 IP/팀총IP)를
    # 그대로 비교하면 안 된다 - 단위가 다르다(경기 비율 vs 이닝 비율). 구원투수가
    # 대타 상황 등으로 어쩌다 1타석 서면 Lahman 타격 G는 "그 경기에 있었다"는
    # 이유만으로 등판 경기수와 똑같이 잡혀서(AB=1인데 G=33 식) 진짜 겸업처럼
    # 오분류된다(실측 확인 - aardsda01 2015). AB/IP 최소선을 넘겨야 그 역할이
    # "진짜" 있었던 것으로 인정한다.
    AB_QUALIFYING = 100  # 대타 1타석 수준과 진짜 타격 출전을 가르는 최소 타수
    IP_QUALIFYING = 20   # PITCHING_SHRINKAGE_K와 동일 - "의미있는 투구량"의 최소선

    both = merged["off_score"].notna() & merged["pit_score"].notna()
    bat_qualifies = both & (merged["AB"].fillna(0) >= AB_QUALIFYING)
    pit_qualifies = both & (merged["IP"].fillna(0) >= IP_QUALIFYING)
    only_bat_real = bat_qualifies & ~pit_qualifies
    only_pit_real = pit_qualifies & ~bat_qualifies
    both_real = bat_qualifies & pit_qualifies
    # 그 시즌 출전비중이 더 높은 쪽 - 둘 다 진짜로 뛴 진짜 겸업(오타니형)에만 적용
    batting_heavier = merged["playing_time_ratio"].fillna(-1) >= merged["playing_time_ratio_pit"].fillna(-1)

    merged["overall_score"] = merged[["off_score", "pit_score"]].mean(axis=1, skipna=True)  # 기본값
    merged.loc[only_bat_real, "overall_score"] = merged.loc[only_bat_real, "off_score"]
    merged.loc[only_pit_real, "overall_score"] = merged.loc[only_pit_real, "pit_score"]
    merged.loc[both_real & batting_heavier, "overall_score"] = merged.loc[both_real & batting_heavier, "off_score"]
    merged.loc[both_real & ~batting_heavier, "overall_score"] = merged.loc[both_real & ~batting_heavier, "pit_score"]
    # neither_real(둘 다 트리비얼한 겸업 - 아주 드문 엣지케이스)은 평균(기본값) 유지

    # primary_position은 Fielding.csv 실측이라 기본적으로 안 건드린다. 다만
    # 위 판정에서 타격이 주 역할로 뽑혔는데 필드 기록이 "P"뿐인 경우(수비 이닝
    # 없이 지명타자로만 나온 겸업 선수)는 포지션 매칭에 "투수"로 잡히면 안
    # 되므로 DH로 보정한다.
    batting_is_primary = only_bat_real | (both_real & batting_heavier)
    dh_fallback = batting_is_primary & (merged["primary_position"] == "P")
    merged.loc[dh_fallback, "primary_position"] = "DH"

    # 타격 기록은 있는데(off_score 존재) 수비 기록이 아예 없는 선수-시즌(순수
    # 지명타자)은 derive_primary_position()이 애초에 만들지 못해 NaN으로 남는다
    # - Lahman Fielding.csv엔 "DH"라는 포지션 자체가 없기 때문(수비를 안 하므로).
    # "같은 포지션 추천"에서 이런 선수가 누락되지 않도록 DH로 명시한다.
    pure_dh = merged["primary_position"].isna() & merged["off_score"].notna()
    merged.loc[pure_dh, "primary_position"] = "DH"

    # role도 같은 AB/IP 최소선으로 되돌린다. build_player_season_metadata()의
    # role은 raw batting/pitching KEY 존재만 보고 TWO를 매겨서(대타 1타석도
    # is_batter=True) 구원투수 대다수가 "투타겸업"으로 잘못 표시되는 문제가
    # 있었다(실측 확인 - 화면에서 릴리버들이 전부 투타겸업으로 보임). 여기서
    # off_score/pit_score의 신뢰 근거(both/bat_qualifies/pit_qualifies)를 그대로
    # 재사용해 진짜 겸업이 아니면 B/P로 되돌린다.
    merged.loc[only_bat_real, "role"] = "B"
    merged.loc[only_pit_real, "role"] = "P"
    # both_real(오타니형) / neither_real(둘 다 트리비얼 — 극히 드묾)은 TWO 유지

    # off_score/pit_score 둘 중 하나가 NaN인 TWO(원래 role 계산이 AB=0인 행도
    # "타격 기록 있음"으로 셈)는 위 both 기반 분기를 아예 못 타서 안 고쳐진다 —
    # 이쪽은 판단 기준(AB/IP 최소선)도 필요 없다, 애초에 한쪽 점수 자체가 없다.
    merged.loc[merged["off_score"].isna() & merged["pit_score"].notna(), "role"] = "P"
    merged.loc[merged["pit_score"].isna() & merged["off_score"].notna(), "role"] = "B"

    # 수비 전력: strength.compute_fielding_strength()가 만든 fielding_strength_before_pt를
    # 그대로 def_score로 옮긴다(off_score/pit_score와 동일하게 출전·부상 반영 *전* 값).
    # 수비 기록이 아예 없는 선수-시즌(지명타자 전업 등)은 자연히 NaN으로 남는다.
    merged["def_score"] = merged["fielding_strength_before_pt"]

    merged["ops_z"] = merged.groupby("season")["OPS"].transform(lambda s: (s - s.mean()) / s.std())
    merged["era_z"] = merged.groupby("season")["ERA"].transform(lambda s: (s - s.mean()) / s.std())
    merged["whip_z"] = merged.groupby("season")["WHIP"].transform(lambda s: (s - s.mean()) / s.std())

    # 이전 시즌 출전비율(g_ratio_prev)·변화량(g_chg) 계산
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

    merged["ops_z_prev"] = (
        merged
        .groupby("player_id")["ops_z"]
        .shift(1)
    )

    # ------------------------------------------------------------------
    # def_score 안전장치
    #
    # 컬럼 존재는 위에서 이미 보장되지만(대입문), fielding 파이프라인이
    # 조용히 깨져서 전부 NaN만 나오는 상황(예: merge 키가 하나도 안 맞음)은
    # 별개 문제다 - "컬럼은 있는데 쓸모없는 상태"를 여기서 fail-fast로 잡는다.
    # ------------------------------------------------------------------

    if merged["def_score"].notna().sum() == 0:
        raise ValueError(
            "def_score가 전부 결측입니다 - fielding_strength 파이프라인이 "
            "silent하게 깨졌을 가능성이 높습니다(merge 키 불일치 등). "
            "build_strength_features()의 fielding_result 병합을 확인하세요."
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
        # labels.py의 data_end_year는 "그 시즌 자체는 censor(라벨 보류)"하는
        # 배타적 경계다(l1_observable = season < data_end_year). 반면
        # contract.LABEL_END_YEAR는 "그 시즌까지는 라벨이 있어야 한다"는
        # 포함 경계다(validate()의 season <= LABEL_END_YEAR). 그대로 넘기면
        # LABEL_END_YEAR 자기 자신이 censor돼서 y_departed가 결측으로
        # 나온다(실측 확인됨) — +1 해서 배타/포함 정의를 맞춘다.
        config=labels.LabelConfig(
            data_end_year=contract.LABEL_END_YEAR + 1,
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

    주의: 호출부에서 strength_df 자리에 넘기는 strength_full은 이미
    merge_strength_metadata()에서 metadata_df와 합쳐진 상태다(team_last/
    franch_id/... 다 포함). 그런데도 여기서 metadata_df를 다시 merge하면
    같은 컬럼명이 겹쳐 pandas가 자동으로 _x/_y를 붙여버려서, 원래 이름의
    컬럼이 하나도 안 남는다 — 그래서 최종 병합 직전에 필수 컬럼이 통째로
    "없다"는 에러가 났었다(실측). y_next_score도 마찬가지로 labeled_df가
    이미 만들어서 주므로 여기서 또 계산하지 않는다. metadata_df는 이미
    strength_df 안에 들어있어 여기서는 쓰지 않는다.
    """

    merged = strength_df.copy()

    label_columns = KEY + contract.LABEL_COLS

    merged = merged.merge(
        labeled_df[label_columns],
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
    teams_raw, batting_raw, pitching_raw, fielding_raw, injuries_raw, players_raw = load_raw_data()

    print(f"  teams    : {len(teams_raw):,}")
    print(f"  batting  : {len(batting_raw):,}")
    print(f"  pitching : {len(pitching_raw):,}")
    print(f"  fielding : {len(fielding_raw):,}")
    print(f"  injuries : {len(injuries_raw):,}")
    print(f"  players  : {len(players_raw):,}")

    # 2. B 전력
    print("\n[2/7] B strength.py 실행")
    strength_df = build_strength_features(
        teams_raw,
        batting_raw,
        pitching_raw,
        fielding_raw,
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
        players_raw,
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
