"""MLB 경기 일정·결과 수집 파이프라인 — 설계 초안.

목적: win_rate.py 학습용 "끝난 경기 결과"와 game.py 서빙용 "오늘 일정"을
동일한 스케줄 엔드포인트에서 가져온다. mlb_injury_pipeline.py 와 같은 성격의
초안이다 — A가 src/adapters/mlb_api.py 스타일(로깅, 재시도, config 분리 등)에
맞춰 다듬어서 편입하면 된다. API 키 불필요, 개인·비상업 사용 무료 (이미 검증됨).

이 파일이 "만드는" 것과 "안 만드는" 것
  만든다 : game_pk, game_date, home/away_team, home/away_sp_era,
           home/away_rest, home/away_last10, y_home_win
           (= schema.sql 의 games 테이블에서 MLB API 로만 알 수 있는 컬럼들)
  안 만든다: home_strength / away_strength — 이건 team_season(D/E 담당 계산 로직,
           calculate_team_strength)에서 나오는 값이라 win_rate.py 가 조인해야 한다.
           이 파일 결과에 home_strength/away_strength 컬럼은 NaN 으로 비워둔다.

전체 흐름
  1) /api/v1/schedule 를 날짜범위로 호출 (하루씩 반복 X — 범위 조회 됨, 직접 확인함)
  2) 끝난 경기(status=Final)는 y_home_win 확정, 오늘 경기는 y_home_win=NaN
  3) 각 팀의 직전 경기 대비 휴식일수(rest) 계산 — 로컬 계산, API 추가 호출 불필요
  4) 각 팀의 "그 경기 시점 이전" 10경기 승률(last10) 계산 — 로컬 계산
     (shift(1) 로 자기 자신 경기는 제외 — 미래 정보 누수 방지)
  5) 선발투수 시즌 ERA — /api/v1/people/{id}/stats 별도 호출, 시즌 단위로 캐싱
     (주의: 이건 "그 경기 시점까지의" ERA가 아니라 "시즌 최종/현재" ERA로 근사한
     것이다 — 완벽한 walk-forward 피처가 아님. 학습 파이프라인부터 굴러가게
     하는 것을 우선하고, 여유되면 날짜별 롤링으로 정교화할 것.)

실행 전 준비물
  - `uv add requests` (아직 없다면, mlb_injury_pipeline.py 에서 이미 추가했다면 불필요)
  - 인터넷 연결. API 키는 필요 없음.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

# ── 설정 ──────────────────────────────────────────────────────────
# MLB Stats API 의 team.abbreviation 은 프로젝트 전역에서 쓰는 UI 코드(theme.py의
# TEAM_COLORS/TEAM_NAMES, recommend.py의 LAHMAN_TEAM_TO_UI 변환 후 코드)와 6개 팀이
# 다르다 — /api/v1/teams 로 30팀 전부 직접 대조해서 확인함. 이 매핑 없이 조인하면
# 이 6개 팀 경기가 team_season 과 조인 안 되고 조용히 NaN 처리돼서 알아채기 어렵다.
MLB_API_TEAM_TO_UI = {
    "AZ": "ARI", "CWS": "CHW", "KC": "KCR", "SD": "SDP", "SF": "SFG",
    "TB": "TBR", "WSH": "WSN",
    # 어슬레틱스는 2025년에 "Oakland" 를 떼고 개명(오클랜드 -> 새크라멘토 임시 연고,
    # 라스베이거스 이전 예정) — 그래서 2023~24 경기는 API 가 "OAK", 2025+ 는 "ATH"로
    # 돌려준다. 같은 프랜차이즈인데 시즌마다 코드가 달라서 조인이 깨짐 — 직접 겪음
    # (features_v1 쪽은 LAHMAN_TEAM_TO_UI 가 이미 전 시즌 "ATH"로 통일해놨음).
    "OAK": "ATH",
}
SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
PERSON_STATS_URL = "https://statsapi.mlb.com/api/v1/people/{person_id}/stats"
REQUEST_DELAY_SEC = 0.3  # MLB Stats API 에 과도한 부하를 주지 않기 위한 최소 대기
SEASON_START_MD = "03-01"  # 스프링캠프 끝~정규시즌 시작 여유 포함
SEASON_END_MD = "11-15"  # 월드시리즈까지 여유 포함
LAST_N = 10  # "최근 10경기" 의 N


# ── 1) 일정/결과 수집 ─────────────────────────────────────────────
@dataclass
class ScheduleFetcher:
    session: requests.Session

    def fetch_range(self, start_date: str, end_date: str, game_type: str = "R") -> list[dict]:
        resp = self.session.get(
            SCHEDULE_URL,
            params={
                "sportId": 1,
                "startDate": start_date,
                "endDate": end_date,
                "gameType": game_type,
                # team 을 명시적으로 hydrate 하지 않으면 team 객체에 abbreviation 이 안 실린다
                # (id/name/link 만 옴) — 직접 확인함.
                "hydrate": "team,probablePitcher",
            },
            timeout=30,
        )
        resp.raise_for_status()
        games: list[dict] = []
        for day in resp.json().get("dates", []):
            games.extend(day.get("games", []))
        return games


def _extract_game_row(game: dict) -> dict:
    """스케줄 응답 하나(경기 1건)를 games 테이블 형태 dict 로 변환."""
    home = game["teams"]["home"]
    away = game["teams"]["away"]
    state = game["status"]["abstractGameState"]  # "Final" | "Live" | "Preview" 등

    y_home_win = None
    if state == "Final":
        # 연장/취소 등으로 isWinner 가 둘 다 False/None 인 이례적 케이스 방어
        if home.get("isWinner"):
            y_home_win = 1.0
        elif away.get("isWinner"):
            y_home_win = 0.0

    return {
        "game_pk": game["gamePk"],
        "season": int(game["season"]),
        "game_date": game["officialDate"],
        "league": "mlb",
        "status": state,
        "home_team": MLB_API_TEAM_TO_UI.get(home["team"]["abbreviation"], home["team"]["abbreviation"]),
        "away_team": MLB_API_TEAM_TO_UI.get(away["team"]["abbreviation"], away["team"]["abbreviation"]),
        "home_sp_id": (home.get("probablePitcher") or {}).get("id"),
        "away_sp_id": (away.get("probablePitcher") or {}).get("id"),
        "y_home_win": y_home_win,
    }


def fetch_schedule(start_date: str, end_date: str, game_type: str = "R") -> pd.DataFrame:
    """지정 기간의 경기 목록. 하루씩 반복하지 않고 범위로 한 번에 받는다.

    주의(직접 겪은 버그): MLB 스케줄 API 가 같은 game_pk 를 두 번 주는 경우가
    있다(연기 후 재편성 등 사유 추정 — 넓은 날짜범위로 한 번에 조회할 때 관찰됨).
    game_pk 는 PK 라 중복이 있으면 이후 add_rest_days/add_last10 의 merge 가
    game_pk 당 여러 번 매칭되면서 기하급수적으로 뻥튀기된다(실측: 2,430경기가
    19,804행으로 불어남). 그래서 여기서 바로 dedup 한다 — Final 로 확정된
    쪽을 우선 남긴다.
    """
    fetcher = ScheduleFetcher(session=requests.Session())
    games = fetcher.fetch_range(start_date, end_date, game_type)
    df = pd.DataFrame([_extract_game_row(g) for g in games])
    if df.empty:
        return df

    status_rank = {"Final": 3, "Live": 2, "Preview": 1}
    df["_rank"] = df["status"].map(status_rank).fillna(0)
    df = (
        df.sort_values("_rank")
        .drop_duplicates(subset="game_pk", keep="last")
        .drop(columns="_rank")
        .reset_index(drop=True)
    )
    return df


def fetch_finished_games(season: int) -> pd.DataFrame:
    """win_rate.py 학습용 — 그 시즌에 끝난 경기만."""
    df = fetch_schedule(f"{season}-{SEASON_START_MD}", f"{season}-{SEASON_END_MD}")
    if df.empty:
        return df
    return df[df["status"] == "Final"].reset_index(drop=True)


def fetch_today_games() -> pd.DataFrame:
    """game.py 서빙용 — 오늘 날짜 전체(끝난 경기가 섞여 있어도 status 로 구분 가능)."""
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    return fetch_schedule(today, today)


# ── 2) 휴식일수 / 최근 10경기 — 로컬 계산, 추가 API 호출 없음 ──────────
def _to_long_form(games: pd.DataFrame) -> pd.DataFrame:
    """홈/원정 두 컬럼짜리 games 를 '팀 하나당 한 행'으로 펼친다 (rest/last10 계산용)."""
    home = games[["game_pk", "game_date", "home_team", "y_home_win"]].rename(
        columns={"home_team": "team"}
    )
    home["win"] = home["y_home_win"]
    away = games[["game_pk", "game_date", "away_team", "y_home_win"]].rename(
        columns={"away_team": "team"}
    )
    away["win"] = 1.0 - away["y_home_win"]
    long_form = pd.concat([home, away], ignore_index=True)
    long_form["game_date"] = pd.to_datetime(long_form["game_date"])
    return long_form.sort_values(["team", "game_date"]).reset_index(drop=True)


def add_rest_days(games: pd.DataFrame) -> pd.DataFrame:
    """직전 경기 이후 며칠 쉬었는지(연속 경기=0일). 시즌 첫 경기는 NaN."""
    long_form = _to_long_form(games)
    long_form["rest_days"] = (
        long_form.groupby("team")["game_date"].diff().dt.days.sub(1).clip(lower=0)
    )
    rest = long_form[["game_pk", "team", "rest_days"]]

    out = games.merge(
        rest.rename(columns={"team": "home_team", "rest_days": "home_rest"}),
        on=["game_pk", "home_team"],
        how="left",
    ).merge(
        rest.rename(columns={"team": "away_team", "rest_days": "away_rest"}),
        on=["game_pk", "away_team"],
        how="left",
    )
    return out


def add_last10(games: pd.DataFrame) -> pd.DataFrame:
    """각 팀의 '이 경기 이전' 최근 10경기 승률. shift(1) 로 당일 결과 누수를 막는다.

    끝난 경기가 없는 오늘자 스케줄(fetch_today_games)에 쓰려면, 먼저
    fetch_finished_games 로 받은 과거 결과와 concat 한 뒤 이 함수를 돌려야
    last10 이 채워진다 — 오늘 경기 자체엔 y_home_win 이 없어도, 그 팀의
    "과거" 경기들에서 승패를 끌어오기 때문.
    """
    long_form = _to_long_form(games)
    long_form["last10"] = long_form.groupby("team")["win"].transform(
        lambda s: s.shift(1).rolling(LAST_N, min_periods=1).mean()
    )
    last10 = long_form[["game_pk", "team", "last10"]]

    out = games.merge(
        last10.rename(columns={"team": "home_team", "last10": "home_last10"}),
        on=["game_pk", "home_team"],
        how="left",
    ).merge(
        last10.rename(columns={"team": "away_team", "last10": "away_last10"}),
        on=["game_pk", "away_team"],
        how="left",
    )
    return out


# ── 3) 선발투수 시즌 ERA — 시즌 단위로 캐싱 ─────────────────────────
def fetch_season_era_lookup(
    person_ids: list[int], season: int, cache_dir: Path
) -> dict[int, float]:
    """선발투수 id -> 그 시즌 ERA. 사람 수만큼이 아니라 캐시 파일 하나로 관리.

    주의: '시즌 최종(또는 현재까지 누적)' ERA다. 그 경기 시점까지의 ERA가
    아니므로 학습 데이터에는 약간의 미래정보 누수가 있다 — 초안 단계에서는
    감수하고, 필요해지면 게임 로그 기반 롤링 ERA로 교체할 것.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"pitcher_era_{season}.csv"

    cached: dict[int, float] = {}
    if cache_path.exists():
        cached = dict(
            pd.read_csv(cache_path).itertuples(index=False, name=None)
        )

    ids_needed = {pid for pid in person_ids if pid and pid not in cached}
    if not ids_needed:
        return cached

    session = requests.Session()
    for pid in ids_needed:
        resp = session.get(
            PERSON_STATS_URL.format(person_id=int(pid)),
            params={"stats": "season", "group": "pitching", "season": season},
            timeout=30,
        )
        resp.raise_for_status()
        stats = resp.json().get("stats", [])
        era = None
        if stats and stats[0].get("splits"):
            era_str = stats[0]["splits"][0].get("stat", {}).get("era")
            if era_str not in (None, "-.--"):
                era = float(era_str)
        if era is not None:
            cached[int(pid)] = era
        time.sleep(REQUEST_DELAY_SEC)

    pd.DataFrame(list(cached.items()), columns=["person_id", "era"]).to_csv(
        cache_path, index=False
    )
    return cached


def add_starter_era(games: pd.DataFrame, season: int, cache_dir: Path) -> pd.DataFrame:
    ids = pd.concat([games["home_sp_id"], games["away_sp_id"]]).dropna().unique().tolist()
    era_lookup = fetch_season_era_lookup(ids, season, cache_dir)
    out = games.copy()
    out["home_sp_era"] = out["home_sp_id"].map(era_lookup)
    out["away_sp_era"] = out["away_sp_id"].map(era_lookup)
    return out


# ── 4) 팀 전력 조인 — B의 strength.py 결과(features_v1) + E의 calculate_team_strength ──
def build_team_strength_table(features_path: Path) -> pd.DataFrame:
    """시즌×팀 단위 전력 점수. B strength.py 출력을 E의 계산 로직으로 집계한다.

    features_v1 은 2000~2025 시즌만 있다(직접 확인함) — 2026처럼 아직 라만/B
    파이프라인에 없는 시즌은 여기 안 나온다. add_team_strength() 에서 그런
    시즌은 "그 팀의 가장 최근에 알려진 시즌" 전력으로 대체한다.
    """
    import sys

    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.models.recommend import adapt_features_v1  # noqa: PLC0415
    from src.service.simulation import calculate_team_strength  # noqa: PLC0415

    raw = pd.read_parquet(features_path)
    players = adapt_features_v1(raw)

    rows = []
    for (season, team), group in players.groupby(["season", "team_last"]):
        rows.append(
            {"season": int(season), "team": team, "strength": calculate_team_strength(group).overall}
        )
    return pd.DataFrame(rows)


def add_team_strength(games: pd.DataFrame, features_path: Path) -> pd.DataFrame:
    """games 에 home_strength/away_strength 를 채운다.

    해당 시즌 데이터가 없으면(예: 진행 중인 2026시즌) 그 팀의 가장 최근
    시즌 전력으로 대체한다 — "지금 로스터가 대략 작년이랑 비슷한 수준일
    것"이라는 근사다. 완벽하진 않지만, 아예 NaN으로 비우는 것보다는 낫다.
    """
    strength = build_team_strength_table(features_path)
    latest = (
        strength.sort_values("season")
        .groupby("team")
        .last()
        .rename(columns={"strength": "latest_strength"})
        .reset_index()
    )

    out = games.copy()
    for side in ("home", "away"):
        col = f"{side}_team"
        merged = out.merge(
            strength.rename(columns={"team": col, "strength": f"{side}_strength_exact"}),
            on=["season", col],
            how="left",
        )
        merged = merged.merge(
            latest.rename(columns={"team": col}), on=col, how="left"
        )
        out[f"{side}_strength"] = merged[f"{side}_strength_exact"].fillna(merged["latest_strength"])
    return out


# ── 실행 예시 ─────────────────────────────────────────────────────
def build_training_table_multi(
    cache_dir: Path, seasons: list[int], features_path: Path | None = None
) -> pd.DataFrame:
    """여러 시즌을 합쳐서 학습 테이블을 만든다. 진행 중인 시즌을 넣어도

    fetch_finished_games 가 status=Final 인 경기만 걸러내므로, 아직 안 끝난
    시즌을 넣어도 그때까지 끝난 경기만 자동으로 포함되고 나머지는 무시된다
    (즉 마지막 원소로 "현재 진행 중인 시즌"을 넣어도 안전하다).
    """
    frames = [build_training_table(season, cache_dir, features_path) for season in seasons]
    return pd.concat(frames, ignore_index=True)


def build_training_table(season: int, cache_dir: Path, features_path: Path | None = None) -> pd.DataFrame:
    """win_rate.py 학습용 games 테이블.

    features_path 를 주면 B strength.py 결과(features_v1)를 조인해 home_strength/
    away_strength 를 실제로 채운다. 안 주면(기본값) 전처럼 NaN 으로 비워둔다 —
    features_v1 없이도 나머지 파이프라인이 독립적으로 도는지 테스트할 때 쓸 것.
    """
    finished = fetch_finished_games(season)
    print(f"[{season}] 끝난 경기 {len(finished):,}건")

    finished = add_rest_days(finished)
    finished = add_last10(finished)
    finished = add_starter_era(finished, season, cache_dir)

    if features_path is not None:
        finished = add_team_strength(finished, features_path)
    else:
        finished["home_strength"] = pd.NA
        finished["away_strength"] = pd.NA

    cols = [
        "game_pk", "season", "game_date", "league", "home_team", "away_team",
        "home_strength", "away_strength", "home_sp_era", "away_sp_era",
        "home_rest", "away_rest", "home_last10", "away_last10", "y_home_win",
    ]
    return finished[cols]


def build_today_table(
    cache_dir: Path, recent_history: pd.DataFrame | None = None, features_path: Path | None = None
) -> pd.DataFrame:
    """game.py 서빙용 — 오늘 경기. last10 을 채우려면 과거 결과(recent_history)가 필요.

    recent_history 는 build_training_table(당해 시즌) 로 미리 받아둔 결과를 넘기면 된다.
    """
    today = fetch_today_games()
    if today.empty:
        return today

    season = int(today["season"].iloc[0])
    combined = pd.concat([recent_history, today], ignore_index=True) if recent_history is not None else today
    combined = add_rest_days(combined)
    combined = add_last10(combined)

    today_only = combined[combined["game_pk"].isin(today["game_pk"])].copy()
    today_only = add_starter_era(today_only, season, cache_dir)
    if features_path is not None:
        today_only = add_team_strength(today_only, features_path)
    else:
        today_only["home_strength"] = pd.NA
        today_only["away_strength"] = pd.NA
    return today_only


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent
    CACHE = ROOT / ".cache"
    FEATURES_PATH = ROOT / "data" / "processed" / "features_v1.parquet"

    # 마지막 원소(2026)는 진행 중인 시즌 — 지금까지 끝난 경기만 자동으로 걸러져 들어간다.
    SEASONS = [2023, 2024, 2025, 2026]
    train = build_training_table_multi(CACHE, seasons=SEASONS, features_path=FEATURES_PATH)
    train.to_csv(ROOT / "games_train.csv", index=False)
    print(f"저장 완료: games_train.csv ({len(train):,}행, 시즌 {SEASONS})")
    print(f"home_strength 결측: {train['home_strength'].isna().sum()}/{len(train)}")

    today_games = build_today_table(CACHE, recent_history=train, features_path=FEATURES_PATH)
    print(f"오늘 경기 {len(today_games):,}건")
    if not today_games.empty:
        print(today_games[["home_team", "away_team", "home_strength", "away_strength", "home_sp_era", "away_sp_era"]])
