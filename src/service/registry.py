"""모델 레지스트리 조회 — 서비스 계층.

담당자별 models/registry/{name}.json 을 취합해 비교표로 낸다.
실제 취합 로직은 models/base.py 에 있다 — 여기는 app 이 서비스 계층만
바라보게 하는 얇은 재노출이다 (7-1 의존성 단방향: app → service → models).

담당: D
"""

from __future__ import annotations

from src.models.base import load_registry, registry_table


def list_models(task: str | None = None, owner: str | None = None) -> list[dict]:
    """등록된 모델 메타 목록. task/owner 로 필터링할 수 있다."""
    entries = load_registry()
    if task:
        entries = [e for e in entries if e["task"] == task]
    if owner:
        entries = [e for e in entries if e["owner"] == owner]
    return entries


def comparison_table(include_nested: bool = False):
    """화면③(모델 정보)이 그대로 렌더링하는 비교표."""
    return registry_table(include_nested=include_nested)
