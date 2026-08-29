"""
game.py
=======
MLB 개별 경기 승부 예측 - 머신러닝/딥러닝 모델 학습 + 2026 잔여경기 예측

목표
----
1) 과거 경기 데이터(games.csv, 2010~2025)로 이진분류 모델을 학습한다.
2) 2026시즌 진행 데이터(mlb-2026-asplayed.csv)를 읽어, 이미 끝난 경기(Final)로
   "현재까지의 팀 성적"을 계산하고, 아직 안 한 경기(Scheduled)의 승패를 예측한다.

왜 team_season을 안 쓰고 경기 로그만으로 피처를 다시 만드는가
--------------------------------------------------------
기존 win_rate.py는 team_season.csv의 "그 시즌 최종 승률(win_rate)"을 홈/원정 팀
강함 지표로 사용했다. 하지만 이건 시즌이 다 끝난 뒤에나 알 수 있는 값이라, "시즌
중간에 잔여 경기를 예측"하는 이번 목적에는 맞지 않는다(미래 정보가 섞이는
data leakage). 그래서 이 스크립트는 batting/pitching 등 시즌 집계 파일 대신,
"그 경기 시점까지의 누적 승률(to-date win rate)"을 직접 계산해서 사용한다.
이렇게 하면:
  - 과거 데이터 학습 때도 각 경기 시점까지의 정보만 사용 (leakage 없음)
  - 2026 예측 때도 완전히 동일한 방식으로 "지금까지의 성적"을 계산해서 그대로 적용
    가능 (학습/예측 피처 산출 방식이 100% 동일)

사용 피처
--------
- home_strength_td / away_strength_td : 그 경기 이전까지의 "시즌 누적 승률"
- home_rest / away_rest               : 직전 경기와의 휴식일수
- home_last10 / away_last10           : 직전 10경기 승률
- 위 3쌍의 diff(홈-원정 차이)값

데이터 경로
----------
실제 프로젝트 구조(스크린샷 기준):

    SKN35-2nd-4Team/
    ├── data/
    │   └── final/                 <- 데이터 파일들이 있는 실제 위치
    │       ├── batting_stats.csv
    │       ├── fielding_stats.csv
    │       ├── franchises.csv
    │       ├── games.csv          <- 과거 경기 데이터 (2010~2025)
    │       ├── managers_stats.csv
    │       ├── mlb_api.csv        <- 2026시즌 경기결과 + 잔여경기 일정
    │       ├── pitching_stats.csv
    │       ├── player_injury_stints.csv
    │       ├── player_season.csv
    │       ├── players.csv
    │       ├── team_season.csv
    │       └── teams.csv
    └── src/
        └── models/
            ├── data/               <- (모델별 로컬 캐시용 폴더, 비어있어도 무방)
            ├── __init__.py
            ├── base.py
            ├── evaluate.py
            ├── game.py             <- 이 스크립트
            └── ...

win_rate.py와 동일하게, game.py 위치에서 시작해 상위 폴더로 올라가며
games.csv가 실제로 들어있는 "data" 또는 "data/final" 폴더를 자동으로 찾는다
(find_data_dir). src/models/game.py 기준으로 두 단계 위인 SKN35-2nd-4Team/data/final/
을 자동으로 찾아내며, mlb_api.csv도 같은 폴더에서 함께 찾는다.
못 찾으면 환경변수 GAME_DATA_DIR로 직접 지정 가능하다.

모델
----
- 머신러닝: LogisticRegression, RandomForestClassifier
- 딥러닝  : PyTorch MLP

실행 방법
--------
    python game.py

출력
----
- 콘솔에 모델별 Accuracy / ROC-AUC 등 평가 지표
- data/models/ 에 학습된 모델 저장
- data/predictions/remaining_games_predictions.csv : 2026 잔여경기 예측 결과
  (날짜, 홈팀, 원정팀, 모델별 홈팀 승리확률·예측승자)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import joblib

# xgboost는 torch보다 먼저 import해야 한다 — macOS arm64에서 torch를 먼저
# import한 뒤 xgboost를 나중에 import(함수 안에서 지연 import 포함)하면
# 같은 프로세스 안에서 세그폴트가 난다(직접 겪음, D의 strength_ts.py에서도
# 동일 이슈 있었음). 그래서 여기서 미리 import해 순서를 고정해둔다.
import xgboost  # noqa: F401

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# import 순서를 맞춰도 xgboost와 torch가 같이 멀티스레드 BLAS/OpenMP를 쓰면
# 데드락이 날 수 있다(macOS arm64, D의 strength_ts.py에서 겪은 것과 동일 원인).
# LogReg/RF/XGBoost는 이미 다 끝난 뒤에 DL을 돌리므로 성능 손해는 거의 없다.
torch.set_num_threads(1)

import sys as _sys
from pathlib import Path as _Path

_ROOT_FOR_IMPORT = _Path(__file__).resolve().parents[2]
if str(_ROOT_FOR_IMPORT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT_FOR_IMPORT))

from src.models.base import BaseModel  # noqa: E402
# game.py 자체에 evaluate(name, y_true, y_pred, y_proba)라는 로컬 함수가 이미
# 있어서(모듈 아래에서 재정의됨) 이름이 겹친다 — d_evaluate로 별칭.
from src.models.evaluate import evaluate as d_evaluate  # noqa: E402


# ----------------------------------------------------------------------
# 0. 경로 설정 (win_rate.py와 동일한 자동 탐색 방식)
# ----------------------------------------------------------------------
def find_data_dir(start_dir, filename="games.csv", max_up=8):
    """start_dir(스크립트 위치)부터 상위 폴더로 올라가며 filename이 실제로 있는
    폴더를 찾는다. 각 단계에서 <폴더>/data 와 <폴더>/data/final 두 가지를 검사한다."""
    current = os.path.abspath(start_dir)
    tried = []
    patterns = [("data",), ("data", "final")]
    for _ in range(max_up + 1):
        for pattern in patterns:
            candidate = os.path.join(current, *pattern)
            tried.append(candidate)
            if os.path.isfile(os.path.join(candidate, filename)):
                return candidate, tried
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return None, tried


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_env_override = os.environ.get("GAME_DATA_DIR")
if _env_override:
    DATA_DIR = _env_override
    _SEARCH_TRIED = [f"(환경변수 GAME_DATA_DIR 사용) {DATA_DIR}"]
else:
    _found, _SEARCH_TRIED = find_data_dir(BASE_DIR)
    DATA_DIR = _found if _found else os.path.join(BASE_DIR, "data")

MODEL_DIR = os.path.join(DATA_DIR, "models")
PRED_DIR = os.path.join(DATA_DIR, "predictions")

GAMES_PATH = os.path.join(DATA_DIR, "games.csv")
GAMES_2026_PATH = os.path.join(DATA_DIR, "mlb_api.csv")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ----------------------------------------------------------------------
# 1. 팀명 매핑 (2026 파일은 정식 구단명, 과거 데이터는 3자리 코드 사용)
# ----------------------------------------------------------------------
TEAM_NAME_TO_ID = {
    "Arizona Diamondbacks": "ARI", "Athletics": "ATH", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS", "Chicago Cubs": "CHN",
    "Chicago White Sox": "CHA", "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU",
    "Kansas City Royals": "KCA", "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAN",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN",
    "New York Mets": "NYN", "New York Yankees": "NYA", "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT", "San Diego Padres": "SDN", "San Francisco Giants": "SFN",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "SLN", "Tampa Bay Rays": "TBA",
    "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR", "Washington Nationals": "WAS",
}


# ----------------------------------------------------------------------
# 2. 경기 로그 -> "그 시점까지" 피처 계산 (학습/예측 공통 함수)
# ----------------------------------------------------------------------
def build_time_aware_features(games):
    """games: [season, game_date(datetime), home_team, away_team, y_home_win(NaN 허용)]
    (+ 위 컬럼 외에 추가로 들어있는 컬럼, 예: 'Status'는 그대로 보존되어 함께 반환됨)
    각 경기 이전까지의 누적 승률/최근10경기승률/휴식일수를 팀 관점으로 계산해
    home_*, away_* 컬럼으로 되돌려준다. 미래 정보 유출 없이(shift 사용) 계산한다.
    더블헤더처럼 (날짜, 홈팀, 원정팀)이 같은 경기가 있어도 game_pk로 고유하게
    처리하므로 행이 중복 생성되지 않는다."""
    games = games.sort_values(["game_date", "home_team", "away_team"]).reset_index(drop=True)
    games["game_pk"] = range(1, len(games) + 1)

    home_rows = games[["game_pk", "season", "game_date", "home_team", "y_home_win"]].copy()
    home_rows.columns = ["game_pk", "season", "game_date", "team", "win"]
    home_rows["is_home"] = 1

    away_rows = games[["game_pk", "season", "game_date", "away_team", "y_home_win"]].copy()
    away_rows.columns = ["game_pk", "season", "game_date", "team", "win"]
    away_rows["win"] = 1 - away_rows["win"]
    away_rows["is_home"] = 0

    long = pd.concat([home_rows, away_rows], ignore_index=True)
    long = long.sort_values(["team", "game_date", "game_pk"]).reset_index(drop=True)

    # 직전 경기와의 휴식일수
    long["prev_date"] = long.groupby(["team", "season"])["game_date"].shift(1)
    long["rest"] = (long["game_date"] - long["prev_date"]).dt.days

    # 결과가 아직 없는(미래) 경기는 win이 NaN -> 누적/롤링 계산에서 자동 제외됨
    win_shift = long.groupby(["team", "season"])["win"].shift(1)

    # 시즌 누적 승률(to-date): 그 경기 이전까지 치른 경기의 승률
    long["games_so_far"] = long.groupby(["team", "season"])["win"].transform(
        lambda s: s.shift(1).expanding().count()
    )
    long["wins_so_far"] = long.groupby(["team", "season"])["win"].transform(
        lambda s: s.shift(1).expanding().sum()
    )
    long["strength_td"] = (long["wins_so_far"] / long["games_so_far"]).fillna(0.5)  # 시즌 첫 경기는 중립 0.5

    # 최근 10경기 승률
    long["last10"] = long.groupby(["team", "season"])["win"].transform(
        lambda s: s.shift(1).rolling(window=10, min_periods=1).mean()
    )

    home_feat = long[long["is_home"] == 1][["game_pk", "rest", "strength_td", "last10"]].rename(
        columns={"rest": "home_rest", "strength_td": "home_strength", "last10": "home_last10"})
    away_feat = long[long["is_home"] == 0][["game_pk", "rest", "strength_td", "last10"]].rename(
        columns={"rest": "away_rest", "strength_td": "away_strength", "last10": "away_last10"})

    games = games.merge(home_feat, on="game_pk", how="left").merge(away_feat, on="game_pk", how="left")

    # 시즌 첫 경기 등 결측 -> 중립값
    games["home_rest"] = games["home_rest"].fillna(4)
    games["away_rest"] = games["away_rest"].fillna(4)
    games["home_last10"] = games["home_last10"].fillna(0.5)
    games["away_last10"] = games["away_last10"].fillna(0.5)

    games["strength_diff"] = games["home_strength"] - games["away_strength"]
    games["rest_diff"] = games["home_rest"] - games["away_rest"]
    games["last10_diff"] = games["home_last10"] - games["away_last10"]

    games = add_pitching_quality(games)
    games = add_batting_defense_quality(games)

    return games


_PITCHING_QUALITY_CACHE = None


def load_pitching_quality_table():
    """team_season.csv의 pit_strength(선발+불펜 합산 투수진 전력)를 시즌x팀 단위로 불러온다.

    home_sp_era/away_sp_era를 games.csv에 두려고 했지만 실제로는 채워지지
    않았다(직접 확인함, 37,339행 전부 결측). 대신 이미 채워져 있는
    team_season.pit_strength를 쓴다 — 선발만이 아니라 로테이션+불펜을 합친
    투수진 전체 전력이라 "불펜 반영"이라는 목적에 더 잘 맞는다. 시즌 단위
    값이라 그 시즌 안에서는 날짜별로 변하지 않는다는 근사가 있다.
    """
    global _PITCHING_QUALITY_CACHE
    if _PITCHING_QUALITY_CACHE is not None:
        return _PITCHING_QUALITY_CACHE

    path = os.path.join(DATA_DIR, "team_season.csv")
    if not os.path.exists(path):
        print(f"[안내] team_season.csv를 못 찾아 투수진 전력 피처를 건너뜁니다: {path}")
        _PITCHING_QUALITY_CACHE = pd.DataFrame(columns=["year", "team_id", "pit_strength"])
        return _PITCHING_QUALITY_CACHE

    ts = pd.read_csv(path, usecols=["year", "team_id", "pit_strength"])
    _PITCHING_QUALITY_CACHE = ts
    return ts


def add_pitching_quality(games):
    """home/away_pitching_quality, pitching_quality_diff 컬럼을 추가한다.

    team_season.csv는 2025시즌까지만 있다 — 2026시즌 경기는 그 팀의 가장
    최근 시즌 값으로 대체한다(로스터가 갑자기 확 바뀌진 않는다는 근사).
    매칭 자체가 안 되는 팀은 리그 평균으로 채운다.
    """
    ts = load_pitching_quality_table()
    if ts.empty:
        games["home_pitching_quality"] = 0.0
        games["away_pitching_quality"] = 0.0
        games["pitching_quality_diff"] = 0.0
        return games

    latest = ts.sort_values("year").groupby("team_id")["pit_strength"].last()
    league_avg = float(ts["pit_strength"].mean())

    for side in ("home", "away"):
        merged = games.merge(
            ts.rename(columns={"year": "season", "team_id": f"{side}_team"}),
            on=["season", f"{side}_team"],
            how="left",
        )["pit_strength"]
        fallback = games[f"{side}_team"].map(latest)
        games[f"{side}_pitching_quality"] = merged.fillna(fallback).fillna(league_avg).to_numpy()

    games["pitching_quality_diff"] = games["home_pitching_quality"] - games["away_pitching_quality"]
    return games


_TEAM_QUALITY_CACHE = None


def load_team_quality_table():
    """team_season.csv에서 타격/수비 전력(bat_strength/def_strength)을 불러온다.

    [2026-08-29 추가] FEATURE_COLS를 보면 투수진 전력(pitching_quality)만 있고
    팀 타격력·수비력이 아예 안 들어가 있었다 — win_rate 세 모델(logreg/xgb/mlp)
    전부 AUC 0.58~0.59로 똑같이 낮았던 게(모델을 바꿔도 안 오르는 걸 보면 모델
    문제가 아니라 피처 문제) 이 구멍 때문일 가능성이 커서 추가한다.
    load_pitching_quality_table()과 캐시를 분리한 이유: 기존에 검증된
    pitching_quality 피처 동작을 건드리지 않기 위해서다(그 함수는 그대로 둠).
    """
    global _TEAM_QUALITY_CACHE
    if _TEAM_QUALITY_CACHE is not None:
        return _TEAM_QUALITY_CACHE

    path = os.path.join(DATA_DIR, "team_season.csv")
    if not os.path.exists(path):
        print(f"[안내] team_season.csv를 못 찾아 타격/수비 전력 피처를 건너뜁니다: {path}")
        _TEAM_QUALITY_CACHE = pd.DataFrame(columns=["year", "team_id", "bat_strength", "def_strength"])
        return _TEAM_QUALITY_CACHE

    ts = pd.read_csv(path, usecols=["year", "team_id", "bat_strength", "def_strength"])
    _TEAM_QUALITY_CACHE = ts
    return ts


def add_batting_defense_quality(games):
    """home/away_batting_quality, home/away_defense_quality와 그 diff를 추가한다.

    add_pitching_quality()와 동일한 패턴(team_season.csv, 시즌 매칭 안 되면
    그 팀의 가장 최근 시즌 값으로 폴백, 그래도 없으면 리그 평균으로 폴백)을
    타격/수비 전력에도 그대로 적용한다.
    """
    ts = load_team_quality_table()
    if ts.empty:
        for side in ("home", "away"):
            games[f"{side}_batting_quality"] = 0.0
            games[f"{side}_defense_quality"] = 0.0
        games["batting_quality_diff"] = 0.0
        games["defense_quality_diff"] = 0.0
        return games

    latest_bat = ts.sort_values("year").groupby("team_id")["bat_strength"].last()
    latest_def = ts.sort_values("year").groupby("team_id")["def_strength"].last()
    league_avg_bat = float(ts["bat_strength"].mean())
    league_avg_def = float(ts["def_strength"].mean())

    for side in ("home", "away"):
        merged = games.merge(
            ts.rename(columns={"year": "season", "team_id": f"{side}_team"}),
            on=["season", f"{side}_team"],
            how="left",
        )
        fallback_bat = games[f"{side}_team"].map(latest_bat)
        fallback_def = games[f"{side}_team"].map(latest_def)
        games[f"{side}_batting_quality"] = (
            merged["bat_strength"].fillna(fallback_bat).fillna(league_avg_bat).to_numpy()
        )
        games[f"{side}_defense_quality"] = (
            merged["def_strength"].fillna(fallback_def).fillna(league_avg_def).to_numpy()
        )

    games["batting_quality_diff"] = games["home_batting_quality"] - games["away_batting_quality"]
    games["defense_quality_diff"] = games["home_defense_quality"] - games["away_defense_quality"]
    return games


FEATURE_COLS = [
    "home_strength", "away_strength", "strength_diff",
    "home_rest", "away_rest", "rest_diff",
    "home_last10", "away_last10", "last10_diff",
    "home_pitching_quality", "away_pitching_quality", "pitching_quality_diff",
    "home_batting_quality", "away_batting_quality", "batting_quality_diff",
    "home_defense_quality", "away_defense_quality", "defense_quality_diff",
]
TARGET_COL = "y_home_win"


# ----------------------------------------------------------------------
# 3. 과거 데이터(games.csv) 로드 & 피처 생성
# ----------------------------------------------------------------------
def load_historical_games(path=GAMES_PATH):
    if not os.path.exists(path):
        tried_str = "\n".join(f"    - {p}" for p in _SEARCH_TRIED)
        raise FileNotFoundError(
            f"games.csv를 찾을 수 없습니다: {path}\n"
            f"\n  다음 경로들에서 찾아봤지만 없었습니다:\n{tried_str}\n"
            f"\n  환경변수로 직접 지정하려면:\n"
            f"    $env:GAME_DATA_DIR = \"C:\\경로\\data\"   (PowerShell)\n"
            f"    python game.py"
        )
    df = pd.read_csv(path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df[["season", "game_date", "home_team", "away_team", "y_home_win"]].copy()
    return build_time_aware_features(df)


# ----------------------------------------------------------------------
# 4. 2026 시즌 데이터 로드 (완료경기 -> 성적 반영, 예정경기 -> 예측대상)
# ----------------------------------------------------------------------
def load_2026_games(path=GAMES_2026_PATH):
    """2026시즌 진행 데이터(mlb_api.csv)를 로드한다.
    (2026년 MLB 경기결과 + 잔여경기 일정 데이터, data/final/mlb_api.csv 위치)"""
    if not os.path.exists(path):
        print(f"\n[안내] 2026 시즌 파일을 찾을 수 없습니다: {path}")
        print("  잔여경기 예측 단계는 건너뜁니다. mlb_api.csv를 data 폴더에 넣고 다시 실행하세요.")
        return None

    raw = pd.read_csv(path, encoding="latin1")
    raw["home_team"] = raw["Home"].map(TEAM_NAME_TO_ID)
    raw["away_team"] = raw["Away"].map(TEAM_NAME_TO_ID)

    unmapped = raw[raw["home_team"].isna() | raw["away_team"].isna()]
    if len(unmapped) > 0:
        unknown_names = set(unmapped["Home"]).union(set(unmapped["Away"])) - set(TEAM_NAME_TO_ID.keys())
        print(f"[경고] 팀명 매핑 실패 {len(unmapped)}건, 알 수 없는 팀명: {unknown_names}")
        raw = raw.dropna(subset=["home_team", "away_team"])

    raw["game_date"] = pd.to_datetime(raw["Date"])
    raw["season"] = 2026

    # 완료된 경기(Final)는 실제 스코어로 y_home_win 계산, 예정 경기(Scheduled)는 NaN
    is_final = raw["Status"] == "Final"
    raw["y_home_win"] = np.nan
    raw.loc[is_final, "y_home_win"] = (
        raw.loc[is_final, "Home Score"] > raw.loc[is_final, "Away Score"]
    ).astype(float)

    df = raw[["season", "game_date", "home_team", "away_team", "y_home_win", "Status"]].copy()
    featured = build_time_aware_features(df)
    return featured


# ----------------------------------------------------------------------
# 5. 시간 기준 Train/Test 분할
# ----------------------------------------------------------------------
def time_based_split(df, train_seasons=range(2010, 2023), test_seasons=range(2023, 2026)):
    train_df = df[df["season"].isin(train_seasons)].copy()
    test_df = df[df["season"].isin(test_seasons)].copy()
    return train_df, test_df


# ----------------------------------------------------------------------
# 6. 평가 유틸
# ----------------------------------------------------------------------
def evaluate(name, y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)

    print(f"\n[{name}] 평가 결과")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print("  Confusion Matrix:")
    print("  ", confusion_matrix(y_true, y_pred))

    return {"model": name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "roc_auc": auc}


# ----------------------------------------------------------------------
# 7. PyTorch DL 모델
# ----------------------------------------------------------------------
class WinPredictorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(32, 16)):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(0.2)]
            prev_dim = h
        layers += [nn.Linear(prev_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_dl_model(X_train, y_train, X_val, y_val, epochs=60, batch_size=256, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = WinPredictorMLP(input_dim=X_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss, best_state, patience, patience_cnt = float("inf"), None, 8, 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss, best_state, patience_cnt = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_cnt += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d} | val_loss={val_loss:.4f}")
        if patience_cnt >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, device


def predict_dl_model(model, device, X):
    model.eval()
    with torch.no_grad():
        proba = torch.sigmoid(model(torch.tensor(X, dtype=torch.float32).to(device))).cpu().numpy()
    return (proba >= 0.5).astype(int), proba


# ----------------------------------------------------------------------
# 7.5 D의 공통 레지스트리(BaseModel)에 등록하기 위한 얇은 래퍼
#
# task="win_rate"로 등록한다 — base.py의 TASKS에는 "game"이라는 태스크가
# 없다(5개 태스크 x ML/DL = 10모델 스펙: strength/win_rate/departure/reason/
# recommend뿐). game.py가 실제로 예측하는 건 "이 경기, 이 상대와 붙었을 때"의
# 개별 경기 승부라 win_rate 태스크의 정의(evaluate.py의 BINARY_TASKS)와
# 정확히 일치한다.
# ----------------------------------------------------------------------
class WinRateLogReg(BaseModel):
    name, task, kind, owner = "win_rate_logreg", "win_rate", "ml", "A"

    def _fit(self, X, y):
        self.model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED).fit(X, y)

    def _predict_proba(self, X):
        return self.model.predict_proba(X)


class WinRateXGB(BaseModel):
    name, task, kind, owner = "win_rate_xgb", "win_rate", "ml", "A"

    def fit_with_validation(self, X_train, y_train, X_val, y_val, n_trials: int = 30, timeout: int = 180):
        """strength_xgb/departure_lgbm과 동일한 패턴(2026-08-29 추가) - Optuna로
        검증 AUC를 최대화하는 하이퍼파라미터를 찾은 뒤 train+val로 최종 학습한다.
        예전엔 n_estimators=300/max_depth=4/lr=0.03 고정값이었다."""
        import optuna
        from sklearn.metrics import roc_auc_score
        from xgboost import XGBClassifier

        def objective(trial: optuna.Trial) -> float:
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                max_depth=trial.suggest_int("max_depth", 3, 8),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
            )
            trial_model = XGBClassifier(
                **params, random_state=RANDOM_SEED, eval_metric="logloss", n_jobs=-1,
            ).fit(X_train, y_train)
            proba = trial_model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, proba)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        self.best_params_ = study.best_params
        self.params = dict(study.best_params)

        # 최종 학습은 train+val을 합쳐서 - 다른 태스크(strength/departure)와 동일한 관례.
        combined_X = np.concatenate([X_train, X_val])
        combined_y = np.concatenate([y_train, y_val])
        self.fit(combined_X, combined_y)
        return study.best_value

    def _fit(self, X, y):
        from xgboost import XGBClassifier

        defaults = dict(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            reg_alpha=0.0, min_child_weight=1,
        )
        defaults.update(self.params)
        self.model = XGBClassifier(
            **defaults, random_state=RANDOM_SEED, eval_metric="logloss", n_jobs=-1,
        ).fit(X, y)

    def _predict_proba(self, X):
        return self.model.predict_proba(X)


class WinRateMLP(BaseModel):
    name, task, kind, owner = "win_rate_mlp", "win_rate", "dl", "A"

    def _fit(self, X, y):
        n_val = max(1, int(len(X) * 0.2))
        X_tr, X_val = X[:-n_val], X[-n_val:]
        y_tr, y_val = y[:-n_val], y[-n_val:]
        self.model, _device = train_dl_model(X_tr, y_tr, X_val, y_val)

    def _predict_proba(self, X):
        device = next(self.model.parameters()).device
        _, proba = predict_dl_model(self.model, device, X)
        proba = proba.ravel()
        return np.column_stack([1 - proba, proba])


# ----------------------------------------------------------------------
# 8. 메인 파이프라인
# ----------------------------------------------------------------------
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)

    print("=" * 60)
    print("1) 과거 경기 데이터(2010~2025) 로드 및 피처 생성")
    print("=" * 60)
    df = load_historical_games()
    print(f"전체 경기 수: {len(df)}")

    train_df, test_df = time_based_split(df)
    print(f"Train(2010~2022): {len(train_df)}경기 / Test(2023~2025): {len(test_df)}경기")

    X_train_raw = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COL].values
    X_test_raw = test_df[FEATURE_COLS].values
    y_test = test_df[TARGET_COL].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    results = []

    # ---------------- Logistic Regression ----------------
    print("\n" + "=" * 60)
    print("2) 머신러닝 - Logistic Regression")
    print("=" * 60)
    logreg = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    logreg.fit(X_train, y_train)
    pred, proba = logreg.predict(X_test), logreg.predict_proba(X_test)[:, 1]
    results.append(evaluate("Logistic Regression", y_test, pred, proba))
    joblib.dump(logreg, os.path.join(MODEL_DIR, "game_logreg.joblib"))

    # ---------------- Random Forest ----------------
    print("\n" + "=" * 60)
    print("3) 머신러닝 - Random Forest")
    print("=" * 60)
    rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=20,
                                 random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred, proba = rf.predict(X_test), rf.predict_proba(X_test)[:, 1]
    results.append(evaluate("Random Forest", y_test, pred, proba))
    joblib.dump(rf, os.path.join(MODEL_DIR, "game_rf.joblib"))

    # ---------------- XGBoost ----------------
    print("\n" + "=" * 60)
    print("3.5) 머신러닝 - XGBoost")
    print("=" * 60)
    from xgboost import XGBClassifier

    xgb = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=RANDOM_SEED, eval_metric="logloss", n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    pred, proba = xgb.predict(X_test), xgb.predict_proba(X_test)[:, 1]
    results.append(evaluate("XGBoost", y_test, pred, proba))
    joblib.dump(xgb, os.path.join(MODEL_DIR, "game_xgb.joblib"))

    # ---------------- PyTorch MLP ----------------
    print("\n" + "=" * 60)
    print("4) 딥러닝 - PyTorch MLP")
    print("=" * 60)
    n_val = int(len(X_train) * 0.2)
    X_tr, X_val = X_train[:-n_val], X_train[-n_val:]
    y_tr, y_val = y_train[:-n_val], y_train[-n_val:]
    dl_model, device = train_dl_model(X_tr, y_tr, X_val, y_val)
    pred, proba = predict_dl_model(dl_model, device, X_test)
    results.append(evaluate("PyTorch MLP (DL)", y_test, pred.ravel(), proba.ravel()))
    torch.save(dl_model.state_dict(), os.path.join(MODEL_DIR, "game_dl.pt"))

    joblib.dump(scaler, os.path.join(MODEL_DIR, "game_scaler.joblib"))

    # ------------------------------------------------------------------
    # D의 공통 레지스트리(models/registry/*.json)에도 등록한다 — 이게 있어야
    # Streamlit "모델 정보" 화면(3_Model_Information.py)에 뜬다. 위에서 이미
    # 만든 X_train/y_train/X_test/y_test를 그대로 재사용해 새로 학습한다(로컬
    # logreg/dl_model 변수를 직접 재사용하지 않는 이유: BaseModel.fit()이
    # feature_names/classes_ 같은 공통 메타를 자동으로 채워줘서, 다른 4개
    # 태스크 모델들과 완전히 같은 방식으로 저장·비교되게 하기 위함).
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("4.5) D 공통 레지스트리(BaseModel) 등록")
    print("=" * 60)
    # win_rate_xgb만 Optuna로 튜닝한다 - train 구간의 마지막 15%(시간순 뒤쪽,
    # WinRateMLP._fit()의 내부 홀드아웃과 동일한 방식)를 검증셋으로 뗀다.
    n_val = max(1, int(len(X_train) * 0.15))
    X_tr_opt, X_val_opt = X_train[:-n_val], X_train[-n_val:]
    y_tr_opt, y_val_opt = y_train[:-n_val], y_train[-n_val:]

    for model_cls in (WinRateLogReg, WinRateXGB, WinRateMLP):
        registry_model = model_cls()
        if model_cls is WinRateXGB:
            best_auc = registry_model.fit_with_validation(X_tr_opt, y_tr_opt, X_val_opt, y_val_opt)
            print(f"  [{registry_model.name}] 검증 AUC={best_auc:.4f} best_params={registry_model.best_params_}")
        else:
            registry_model.fit(X_train, y_train)
        metrics = d_evaluate(registry_model, X_test, y_test)
        registry_model.set_metrics(**metrics)
        saved_path = registry_model.save(note="game.py 경기 단위 승부예측 (홈/원정 전력·휴식·최근10·타격/투구/수비 전력)")
        print(f"  [{registry_model.name}] auc={metrics.get('roc_auc', float('nan')):.4f} -> {saved_path}")

    print("\n" + "=" * 60)
    print("5) 모델 성능 비교")
    print("=" * 60)
    result_df = pd.DataFrame(results)
    print(result_df.to_string(index=False))
    result_df.to_csv(os.path.join(MODEL_DIR, "game_model_comparison.csv"), index=False)

    # --------------------------------------------------------------
    print("\n" + "=" * 60)
    print("6) 2026시즌 잔여경기 예측")
    print("=" * 60)
    games_2026 = load_2026_games()
    if games_2026 is not None:
        remaining = games_2026[games_2026["Status"] == "Scheduled"].copy()
        print(f"2026시즌 완료경기: {(games_2026['Status']=='Final').sum()}경기")
        print(f"2026시즌 잔여경기: {len(remaining)}경기 -> 예측 진행")

        X_rem_raw = remaining[FEATURE_COLS].values
        X_rem = scaler.transform(X_rem_raw)

        remaining["logreg_home_win_proba"] = logreg.predict_proba(X_rem)[:, 1]
        remaining["rf_home_win_proba"] = rf.predict_proba(X_rem)[:, 1]
        remaining["xgb_home_win_proba"] = xgb.predict_proba(X_rem)[:, 1]
        _, dl_proba = predict_dl_model(dl_model, device, X_rem)
        remaining["dl_home_win_proba"] = dl_proba.ravel()

        # 4개 모델 평균을 최종 확률로 사용
        remaining["ensemble_home_win_proba"] = remaining[
            ["logreg_home_win_proba", "rf_home_win_proba", "xgb_home_win_proba", "dl_home_win_proba"]
        ].mean(axis=1)
        remaining["predicted_winner"] = np.where(
            remaining["ensemble_home_win_proba"] >= 0.5, remaining["home_team"], remaining["away_team"]
        )

        out_cols = ["game_date", "home_team", "away_team",
                    "logreg_home_win_proba", "rf_home_win_proba", "xgb_home_win_proba", "dl_home_win_proba",
                    "ensemble_home_win_proba", "predicted_winner"]
        out = remaining[out_cols].sort_values("game_date")
        out_path = os.path.join(PRED_DIR, "remaining_games_predictions.csv")
        out.to_csv(out_path, index=False)

        print(f"\n예측 결과 저장: {out_path}")
        print(out.head(10).to_string(index=False))
    else:
        print("2026 잔여경기 예측을 건너뛰었습니다 (파일 없음).")

    print(f"\n모델 파일은 {MODEL_DIR} 에 저장되었습니다.")


if __name__ == "__main__":
    main()