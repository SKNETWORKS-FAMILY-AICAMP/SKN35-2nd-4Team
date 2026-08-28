"""대체 선수 추천 모델의 설정 타입 정의."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RecommendationConfig:
    """추천 후보 필터와 이웃 탐색 설정."""

    min_g_ratio: float = 0.10
    exclude_same_team: bool = True
    metric: str = "cosine"

    def __post_init__(self) -> None:
        # 출전 비중의 데이터 계약 범위를 벗어난 필터 기준은 허용하지 않는다.
        if not 0.0 <= self.min_g_ratio <= 1.05:
            raise ValueError("min_g_ratio는 0~1.05 사이여야 합니다.")
        # 현재 구현과 거리-유사도 변환은 코사인 거리만 지원한다.
        if self.metric != "cosine":
            raise ValueError("현재 추천기는 cosine 거리만 지원합니다.")


__all__ = ["RecommendationConfig"]
