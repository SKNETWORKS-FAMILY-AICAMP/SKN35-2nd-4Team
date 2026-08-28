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

# age·exp는 서로 강하게 중복되고 고정 평가에서 추천 정확도를 낮춰 제외한다.
# def_score는 현재 features_v1에 유효값이 없어 실제 거리 계산에 사용할 수 없다.
COMMON_FEATURES = ["overall_score", "g_ratio"]
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
