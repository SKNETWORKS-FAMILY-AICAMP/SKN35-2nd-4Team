"""대체 선수 추천 모델이 공통으로 사용하는 데이터 계약과 경로 설정."""

from pathlib import Path

REQUIRED_COLUMNS = {
    "player_id",
    "season",
    "team_last",
    "role",
    "g_ratio",
    "overall_score",
}

LAHMAN_TEAM_TO_UI = {
    "CHA": "CHW",
    "CHN": "CHC",
    "KCA": "KCR",
    "LAN": "LAD",
    "NYA": "NYY",
    "NYN": "NYM",
    "SDN": "SDP",
    "SFN": "SFG",
    "SLN": "STL",
    "TB": "TBR",
    "TBA": "TBR",  # 라만 teams.csv는 TBA, 다른 표는 TB를 쓴다 (직접 확인함)
    "WAS": "WSN",
}

# 추천 거리는 현재 시즌의 종합 전력과 출전 비중만 공통으로 사용한다.
# 역할별 세부 전력·표준화 지표는 ROLE_FEATURES에서 추가한다.
COMMON_FEATURES = ["overall_score", "predicted_next_overall_score", "g_ratio"]
ROLE_FEATURES = {
    "B": ["off_score", "ops_z"],
    "P": ["pit_score", "era_z", "whip_z"],
    "TWO": ["off_score", "pit_score", "ops_z", "era_z", "whip_z"],
}

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
REGISTRY_DIR = MODEL_DIR / "registry"

__all__ = [
    "COMMON_FEATURES",
    "LAHMAN_TEAM_TO_UI",
    "MODEL_DIR",
    "REGISTRY_DIR",
    "REQUIRED_COLUMNS",
    "ROLE_FEATURES",
    "ROOT",
]
