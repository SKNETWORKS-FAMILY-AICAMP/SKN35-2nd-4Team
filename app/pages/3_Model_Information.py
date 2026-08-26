"""모델 정보 — 유일하게 실데이터로 채울 수 있는 화면.

models/registry/*.json 에 실제로 학습·저장된 모델만 보여준다 (지금은 D 담당
strength_xgb/strength_mlp/strength_lstm 3개뿐). 나머지 7개는 "미학습"으로
투명하게 표시한다 — 가짜 성능 숫자를 채우지 않는다.
"""

import os
import sys

import pandas as pd
import streamlit as st

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.service.registry import comparison_table, list_models  # noqa: E402
from ui.theme import inject_css, page_header, placeholder, section  # noqa: E402

st.set_page_config(page_title="모델 정보", page_icon="⚾", layout="wide")
inject_css()

page_header("모델 정보", "예측에 사용된 모델과 성능, 데이터의 한계를 공개합니다.")

# 기획서 8-1 기준 전체 10개 모델 계획 대비 실제 등록 현황
PLANNED = {
    "win_rate": ("A", ["win_rate_lr / win_rate_lgbm", "win_rate_mlp"]),
    "game": ("A", ["game_lgbm", "game_mlp"]),
    "departure": ("B", ["departure_lgbm", "departure_mlp"]),
    "reason": ("C", ["reason_rf", "reason_mlp"]),
    "strength": ("D", ["strength_xgb", "strength_lstm (또는 strength_mlp 폴백)"]),
    "recommend": ("E", ["recommend_knn", "recommend_autoencoder"]),
}

section("학습 완료된 모델")
entries = list_models()
if not entries:
    placeholder("모델 레지스트리", "아직 저장된 모델이 없습니다. `models/registry/*.json` 확인 필요.")
else:
    table = comparison_table()
    show_cols = [c for c in ["model", "task", "kind", "owner", "mae", "rmse", "r2", "baseline_mae", "n_test"] if c in table.columns]
    st.dataframe(table[show_cols], use_container_width=True, hide_index=True)

    strength_rows = table[table.task == "strength"] if "task" in table.columns else pd.DataFrame()
    if not strength_rows.empty and "mae" in strength_rows.columns:
        best = strength_rows.loc[strength_rows.mae.idxmin()]
        c1, c2, c3 = st.columns(3)
        c1.metric("최고 성능 모델", best["model"])
        c2.metric("MAE (전력 점수 0~100 기준)", f"{best.mae:.2f}")
        if "baseline_mae" in best and pd.notna(best.baseline_mae):
            improve = (1 - best.mae / best.baseline_mae) * 100
            c3.metric("평균 대입 대비 개선", f"{improve:.0f}%", help=f"baseline MAE {best.baseline_mae:.2f}")

trained_tasks = {e["task"] for e in entries}

section("아직 학습되지 않은 모델")
missing = {task: v for task, v in PLANNED.items() if task not in trained_tasks}
if missing:
    rows = [
        {"담당": owner, "태스크": task, "예정 모델": " / ".join(names)}
        for task, (owner, names) in missing.items()
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.success("계획된 10개 모델이 모두 등록되었습니다.")

section("데이터와 한계")
with st.expander("펼치기"):
    st.markdown(
        "| 항목 | 내용 |\n|---|---|\n"
        "| 학습 데이터 | Lahman Baseball Database 2000–2025 (CC BY-SA 3.0) |\n"
        "| 경기 데이터 | MLB Stats API (개인·비상업 무료) — 아직 수집 코드 미작성 |\n"
        "| 분할 | Train 2000–2021 / Valid 2022–2023 / Test 2024 |\n"
    )
    st.markdown(
        "- 시뮬레이션 결과는 실제 미래 승률이 아니라 모델 기반의 가상 시나리오입니다\n"
        "- 부상 태그는 복귀 패턴으로 추정한 값이며 확정 사실이 아닙니다 (아직 미구현)\n"
        "- 이탈 유형(L2) 분류는 트레이드가 전체의 6%뿐이라 성능에 구조적 한계가 있습니다\n"
        "- 전력 예측(D)은 리그 잔류 선수만으로 학습되어 생존 편향이 존재합니다"
    )

section("아시안게임 이벤트")
placeholder("조별리그 예측 + 차출 이탈 시뮬레이션", "F9(C, 선택 기능) — 아직 미구현입니다.")
