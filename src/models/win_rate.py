"""
win_rate.py
===========
MLB 개별 경기 단위 승패 이진분류 모델 (2010~2025) - Logistic Regression

목표
----
games 테이블(경기 단위: home_team vs away_team)의 y_home_win(홈팀 승=1 / 패=0)을
경기 시작 "전"에 알 수 있는 정보(팀 시즌 강함 지표, 최근 폼, 휴식일수)만으로 예측한다.

데이터 경로
----------
실제 프로젝트 구조(스크린샷 기준):

    SKN35-2nd-4Team/
    └── src/
        └── models/                <- win_rate.py가 있는 폴더
            ├── data/               <- games.csv, team_season.csv 등이 여기 위치
            │   ├── games.csv
            │   └── team_season.csv
            ├── __init__.py
            ├── base.py
            ├── evaluate.py
            ├── game.py
            ├── ...
            └── win_rate.py

즉 win_rate.py와 data 폴더가 "같은 폴더(src/models/)" 안에 있다.
스크립트는 자신의 위치에서 시작해 상위 폴더로 올라가며 "data/games.csv"
또는 "data/final/games.csv"(과거 구조)를 자동으로 탐색한다(find_data_dir 함수).
찾지 못하면 탐색한 모든 경로를 에러 메시지로 보여주며, 그래도 안 되면
환경변수 WINRATE_DATA_DIR로 직접 지정할 수 있다.

사용 데이터
----------
- games.csv       : 경기 단위 데이터 (2010~2025, 37,339경기)
- team_season.csv : 팀-시즌 단위 지표 (bat_strength, pit_strength, def_strength)

모델
----
- Logistic Regression (scikit-learn)

평가 분할 방식
-------------
시즌 기준 시간 분할(Time-based split)을 기본으로 사용한다.
  - Train : 2010 ~ 2022 시즌 (30,050경기, 약 80%)
  - Test  : 2023 ~ 2025 시즌 ( 7,289경기, 약 20%)
실제 운영 시나리오(과거로 미래를 예측)와 동일한 조건으로 평가하기 위해서이며,
무작위 분할(random split)은 미래 시즌의 정보가 과거 학습에 섞여 들어가는
data leakage 위험이 있어 사용하지 않았다.

실행 방법
--------
    cd SKN35-2nd-4Team
    python win_rate.py

출력
----
- 콘솔에 Accuracy / Precision / Recall / F1 / ROC-AUC / 회귀계수(feature 영향력) 출력
- data/final/models/ 에 학습된 모델 파일 저장
  (logreg_model.joblib, scaler.joblib)
- data/final/models/model_result.csv : 성능 지표 결과표
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import joblib

# ----------------------------------------------------------------------
# 0. 경로 설정
# ----------------------------------------------------------------------
# 실제 프로젝트 구조(이미지 기준): SKN35-2nd-4Team/src/models/data/games.csv
#   즉 win_rate.py와 data 폴더가 "같은 폴더(src/models/)" 안에 있다.
#   과거 구조(SKN35-2nd-4Team/data/final/games.csv)도 함께 지원하도록,
#   games.csv가 실제로 존재하는 폴더를 아래 두 패턴으로 각 상위 폴더에서 찾는다:
#     - <폴더>/data/games.csv        <- 이미지의 현재 구조
#     - <폴더>/data/final/games.csv  <- 과거 구조
#
# 우선순위:
#   1) 환경변수 WINRATE_DATA_DIR 이 설정되어 있으면 그 경로를 그대로 사용 (가장 확실한 방법)
#   2) 스크립트 위치("start_dir")부터 상위 폴더로 올라가며 위 두 패턴을 자동 탐색
#   3) 못 찾으면, 탐색했던 모든 경로 목록과 함께 에러를 출력
def find_data_dir(start_dir, filename="games.csv", max_up=8):
    """start_dir(스크립트 위치)부터 상위 폴더로 올라가며 filename이 실제로 있는
    폴더를 찾는다. 각 단계에서 <폴더>/data 와 <폴더>/data/final 두 가지를 모두 검사한다.
    반환값: (찾은 경로 또는 None, 탐색을 시도한 경로 목록)"""
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
        if parent == current:   # 파일시스템 루트에 도달
            break
        current = parent
    return None, tried


BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # 이 스크립트가 있는 위치

_env_override = os.environ.get("WINRATE_DATA_DIR")
if _env_override:
    DATA_DIR = _env_override
    _SEARCH_TRIED = [f"(환경변수 WINRATE_DATA_DIR 사용) {DATA_DIR}"]
else:
    _found, _SEARCH_TRIED = find_data_dir(BASE_DIR)
    DATA_DIR = _found if _found else os.path.join(BASE_DIR, "data")

MODEL_DIR = os.path.join(DATA_DIR, "models")
GAMES_PATH = os.path.join(DATA_DIR, "games.csv")
TEAM_SEASON_PATH = os.path.join(DATA_DIR, "team_season.csv")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ----------------------------------------------------------------------
# 1. 데이터 로드 & 피처 결합
# ----------------------------------------------------------------------
def load_and_build_features(games_path=GAMES_PATH, team_season_path=TEAM_SEASON_PATH):
    """games + team_season을 결합해 경기 단위 학습용 피처 테이블을 만든다."""
    if not os.path.exists(games_path):
        tried_str = "\n".join(f"    - {p}" for p in _SEARCH_TRIED)
        raise FileNotFoundError(
            f"games.csv를 찾을 수 없습니다: {games_path}\n"
            f"\n  다음 경로들에서 games.csv를 찾아봤지만 없었습니다:\n{tried_str}\n"
            f"\n  해결 방법 (택 1):\n"
            f"  1) PowerShell에서 실제 games.csv 위치를 확인:\n"
            f"       Get-ChildItem -Recurse -Filter games.csv\n"
            f"     찾은 경로를 아래 2번 방법에 사용하세요.\n"
            f"  2) 환경변수로 데이터 폴더를 직접 지정 후 재실행:\n"
            f"       $env:WINRATE_DATA_DIR = \"C:\\경로\\SKN35-2nd-4Team\\src\\models\\data\"\n"
            f"       python win_rate.py\n"
            f"  3) 이 스크립트 상단의 DATA_DIR 변수에 절대경로를 직접 대입해도 됩니다."
        )
    games = pd.read_csv(games_path)
    team_season = pd.read_csv(team_season_path)[
        ["year", "team_id", "bat_strength", "pit_strength", "def_strength"]
    ]

    df = games.merge(
        team_season.rename(columns={
            "year": "season", "team_id": "home_team",
            "bat_strength": "home_bat", "pit_strength": "home_pit", "def_strength": "home_def",
        }),
        on=["season", "home_team"], how="left",
    )
    df = df.merge(
        team_season.rename(columns={
            "year": "season", "team_id": "away_team",
            "bat_strength": "away_bat", "pit_strength": "away_pit", "def_strength": "away_def",
        }),
        on=["season", "away_team"], how="left",
    )

    # 시즌 개막 첫 경기는 rest/last10이 결측(직전 경기가 없음) -> 중립값으로 대체
    #   - rest: 시즌 첫 경기는 4일 휴식(스프링캠프 이후 정상 휴식)으로 가정
    #   - last10: 직전 기록이 없으므로 0.5(중립) 승률로 가정
    df["home_rest"] = df["home_rest"].fillna(4)
    df["away_rest"] = df["away_rest"].fillna(4)
    df["home_last10"] = df["home_last10"].fillna(0.5)
    df["away_last10"] = df["away_last10"].fillna(0.5)

    # 경기 전에 알 수 없는 정보(선발투수 ERA)는 이번 데이터셋에서 결측 -> 제외
    # (home_sp_era / away_sp_era 컬럼은 사용하지 않음)

    # 파생 피처: 홈-원정 차이값 (팀 간 상대적 우위를 직접적으로 표현)
    df["strength_diff"] = df["home_strength"] - df["away_strength"]
    df["bat_diff"] = df["home_bat"] - df["away_bat"]
    df["pit_diff"] = df["home_pit"] - df["away_pit"]      # ERA는 낮을수록 좋음(주의: 부호 반전 안 함, 모델이 학습)
    df["def_diff"] = df["home_def"] - df["away_def"]
    df["last10_diff"] = df["home_last10"] - df["away_last10"]
    df["rest_diff"] = df["home_rest"] - df["away_rest"]

    return df


FEATURE_COLS = [
    "home_strength", "away_strength", "strength_diff",
    "home_bat", "away_bat", "bat_diff",
    "home_pit", "away_pit", "pit_diff",
    "home_def", "away_def", "def_diff",
    "home_rest", "away_rest", "rest_diff",
    "home_last10", "away_last10", "last10_diff",
]
TARGET_COL = "y_home_win"


# ----------------------------------------------------------------------
# 2. 시간 기준 Train/Test 분할
# ----------------------------------------------------------------------
def time_based_split(df, train_seasons=range(2010, 2023), test_seasons=range(2023, 2026)):
    train_df = df[df["season"].isin(train_seasons)].copy()
    test_df = df[df["season"].isin(test_seasons)].copy()
    return train_df, test_df


# ----------------------------------------------------------------------
# 3. 평가 유틸
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
    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["away_win", "home_win"]))

    return {"model": name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "roc_auc": auc}


# ----------------------------------------------------------------------
# 4. 메인 파이프라인
# ----------------------------------------------------------------------
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 60)
    print("1) 데이터 로드 및 피처 결합")
    print("=" * 60)
    df = load_and_build_features()
    print(f"전체 경기 수: {len(df)}")

    train_df, test_df = time_based_split(df)
    print(f"Train(2010~2022): {len(train_df)}경기 / Test(2023~2025): {len(test_df)}경기")

    X_train_raw = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COL].values
    X_test_raw = test_df[FEATURE_COLS].values
    y_test = test_df[TARGET_COL].values

    # 표준화 (계수 크기를 피처 간 비교 가능하게 만들기 위해 필수)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # --------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2) Logistic Regression 학습")
    print("=" * 60)
    logreg = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    logreg.fit(X_train, y_train)

    pred = logreg.predict(X_test)
    proba = logreg.predict_proba(X_test)[:, 1]
    result = evaluate("Logistic Regression", y_test, pred, proba)

    print("\n  회귀계수(표준화된 피처 기준, 절댓값이 클수록 영향력 큼):")
    coef = pd.Series(logreg.coef_[0], index=FEATURE_COLS).sort_values(key=np.abs, ascending=False)
    print(coef.to_string())

    # --------------------------------------------------------------
    joblib.dump(logreg, os.path.join(MODEL_DIR, "logreg_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

    result_df = pd.DataFrame([result])
    result_df.to_csv(os.path.join(MODEL_DIR, "model_result.csv"), index=False)
    coef.to_csv(os.path.join(MODEL_DIR, "logreg_coefficients.csv"), header=["coefficient"])

    print(f"\n모델 파일 및 결과표가 {MODEL_DIR} 에 저장되었습니다.")
    print("  - logreg_model.joblib")
    print("  - scaler.joblib")
    print("  - model_result.csv")
    print("  - logreg_coefficients.csv")


if __name__ == "__main__":
    main()