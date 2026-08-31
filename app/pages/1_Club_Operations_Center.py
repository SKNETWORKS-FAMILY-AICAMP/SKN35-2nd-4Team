"""구단 상황실 — E 시뮬레이션을 features_v1과 연결한다."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# streamlit run 은 app/ 을 sys.path[0] 으로 잡아 리포 루트의 src.* 가 안 잡힐 때가
# 있다(다른 페이지가 먼저 실행돼 루트를 추가해둔 경우에만 우연히 성공). 페이지
# 단독 진입에서도 항상 되도록 3_Model_Information.py 와 동일한 가드를 둔다.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.models.recommend import adapt_features_v1  # noqa: E402
from src.service.simulation import TeamStrength, calculate_team_strength, simulate  # noqa: E402
from ui.winrate import predict_win_rate_from_strength, win_rate_caption  # noqa: E402
from ui.datasource import load_features as load_features_df, source_caption  # noqa: E402
from ui.photos import headshot_url, load_mlbam_lookup  # noqa: E402
from ui.risk import (  # noqa: E402
    REASON_DISPLAY,
    evidence_html,
    load_departure_model,
    load_reason_model,
    load_reason_thresholds,
    predict_departure_risk,
    predict_reason_tags,
    reason_badge_html,
    reason_explain_html,
    reason_proba_html,
)
from ui.theme import (  # noqa: E402
    badge,
    diamond_lineup_svg,
    icon,
    _risk_tone,
    impact_panel_html,
    inject_css,
    init_state,
    page_header,
    require_team,
    roster_list_html,
    section,
    topbar,
    wrap,
)

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data" / "final" / "features_v1.parquet"
DEPARTURE_MODEL_PATH = ROOT / "models" / "departure_lgbm.pkl"
REASON_MODEL_PATH = ROOT / "models" / "reason_rf.pkl"
PEOPLE_PATH = ROOT / "data" / "processed" / "People.csv"
PLAYERS_PATH = ROOT / "data" / "final" / "players.csv"

ROLE_LABEL = {"B": "타자", "P": "투수", "TWO": "투타겸업"}


@st.cache_data(show_spinner=False)
def load_players() -> pd.DataFrame:
    # Supabase 우선, 실패 시 리포 내 parquet 폴백 (ui/datasource.py)
    return adapt_features_v1(load_features_df())


@st.cache_data(show_spinner=False)
def load_name_lookup() -> dict[str, str]:
    """playerID → 실명 매핑. data/final/players.csv(2026 신인 195명 포함, features_v1의
    전체 player_id를 100% 커버)를 우선으로 쓰고, Lahman People.csv를 보조로 합친다 —
    People.csv 단독으로는 2026 신인 184명이 빠져서 로스터 표에 이름 대신 player_id가
    그대로 노출되는 문제가 있었다."""
    names: dict[str, str] = {}
    if PEOPLE_PATH.exists():
        people = pd.read_csv(PEOPLE_PATH, usecols=["playerID", "nameFirst", "nameLast"])
        pnames = (people["nameFirst"].fillna("") + " " + people["nameLast"].fillna("")).str.strip()
        names.update(dict(zip(people["playerID"], pnames)))
    if PLAYERS_PATH.exists():
        players = pd.read_csv(PLAYERS_PATH, usecols=["player_id", "name_first", "name_last"])
        fnames = (players["name_first"].fillna("") + " " + players["name_last"].fillna("")).str.strip()
        names.update({pid: n for pid, n in zip(players["player_id"], fnames) if n})
    return names


def predict_win_rate(strength: TeamStrength) -> float:
    # 계수는 ui/winrate.py 에 실데이터(510개 팀·시즌)로 적합해 두었다.
    # 예전엔 이 자리에 검증 안 된 상수(0.35 + overall*0.003)가 두 페이지에
    # 복붙돼 있었고, 실제 33.4%p 승률 차이를 3.9%p 로 압축하고 있었다.
    return predict_win_rate_from_strength(strength.overall)


def make_rank_predictor(season_players: pd.DataFrame):
    baselines = {
        team: calculate_team_strength(group).overall
        for team, group in season_players.groupby("team_last")
    }

    def predict_rank(strength: TeamStrength) -> int:
        return 1 + sum(value > strength.overall for value in baselines.values())

    return predict_rank


st.set_page_config(page_title="구단 상황실", page_icon="⚾", layout="wide", initial_sidebar_state="collapsed")
inject_css()
init_state()
team_code = require_team()
topbar("구단 상황실")

with wrap():
    page_header("구단 상황실", "이탈이 팀 전력과 순위에 미치는 영향")

    try:
        players = load_players()
    except FileNotFoundError:
        st.error("data/final/features_v1.parquet 파일이 없습니다.")
        st.stop()
    except ImportError:
        st.error("parquet을 읽기 위한 pyarrow 설치가 필요합니다.")
        st.stop()
    except ValueError as exc:
        st.error(f"features_v1 데이터 계약 오류: {exc}")
        st.stop()
    except Exception as exc:
        # 파일 손상 등 예상하지 못한 로딩 오류도 traceback 대신 화면에 안내한다.
        st.error(f"features_v1을 읽는 중 오류가 발생했습니다: {exc}")
        st.stop()

    quality = players.attrs.get("data_quality", {})
    if quality.get("excluded_rows", 0) or quality.get("imputed_g_ratio", 0):
        st.info(
            f"데이터 정리: {quality.get('excluded_rows', 0)}행 제외, "
            f"출전 비중 {quality.get('imputed_g_ratio', 0)}건 보정"
        )

    season = int(players["season"].max())
    season_players = players.loc[players["season"] == season].copy()
    team_players = season_players.loc[season_players["team_last"] == team_code].copy()
    if team_players.empty:
        st.warning(f"{season}시즌 {team_code} 선수 데이터가 없습니다.")
        st.stop()

    team_players = team_players.sort_values("overall_score", ascending=False).copy()

    # ── 이탈위험 + 연관 요인 (이 서비스의 핵심 — "누가 떠날 위험이 큰가"를
    # 로스터 전체에 대해 먼저 보여준다. 전력순 정렬만 있던 기존 화면의 공백) ──
    names = load_name_lookup()
    photo_lookup = load_mlbam_lookup()
    departure_model = load_departure_model(
        DEPARTURE_MODEL_PATH.stat().st_mtime_ns if DEPARTURE_MODEL_PATH.exists() else 0
    )
    reason_model = load_reason_model(
        REASON_MODEL_PATH.stat().st_mtime_ns if REASON_MODEL_PATH.exists() else 0
    )
    reason_thresholds = load_reason_thresholds(
        FEATURES_PATH.stat().st_mtime_ns if FEATURES_PATH.exists() else 0
    )
    team_players["departure_risk"] = predict_departure_risk(departure_model, team_players)
    reason_tags = predict_reason_tags(reason_model, players, team_players["player_id"].astype(str))
    reason_tags = reason_tags.set_index("player_id")
    team_players["_pid_str"] = team_players["player_id"].astype(str)
    team_players["reason_tag"] = team_players["_pid_str"].map(reason_tags["reason_tag"])
    team_players["reason_proba"] = team_players["_pid_str"].map(reason_tags["reason_proba"])
    for col in ["age", "exp", "overall_score_delta", "g_chg", "reason_injury_score"]:
        team_players[f"_ev_{col}"] = team_players["_pid_str"].map(reason_tags[col])
    team_players["이름"] = team_players["_pid_str"].map(lambda pid: names.get(pid, pid))

    default_player = st.session_state.get("selected_player_id")
    ids = team_players["player_id"].astype(str).tolist()

    sort_label = st.radio(
        "로스터 정렬",
        ["이탈위험순", "전력순"],
        horizontal=True,
        key="roster_sort",
    )
    ranked = (
        team_players.sort_values("departure_risk", ascending=False, na_position="last")
        if sort_label == "이탈위험순"
        else team_players.sort_values("overall_score", ascending=False)
    )

    selected_id = st.selectbox(
        "이탈 시뮬레이션 선수",
        ranked["player_id"].astype(str).tolist(),
        index=(
            ranked["player_id"].astype(str).tolist().index(default_player)
            if default_player in ids else 0
        ),
        format_func=lambda pid: (
            f"{names.get(pid, pid)} · 전력 {team_players.loc[team_players.player_id.astype(str).eq(pid), 'overall_score'].iloc[0]:.1f}"
            f" · 이탈위험 {team_players.loc[team_players.player_id.astype(str).eq(pid), 'departure_risk'].iloc[0]:.0%}"
        ),
    )
    st.session_state.selected_player_id = selected_id

    selected_data = team_players.loc[
        team_players["player_id"].astype(str).eq(selected_id)
    ].iloc[0]
    selected_score = pd.to_numeric(
        pd.Series([selected_data.get("overall_score")]), errors="coerce"
    ).iloc[0]
    if pd.notna(selected_score) and float(selected_score) == 0.0:
        defense = pd.to_numeric(
            pd.Series([selected_data.get("def_score")]), errors="coerce"
        ).iloc[0]
        defense_text = (
            f" · 수비 전력 {float(defense):.1f}점(별도)"
            if pd.notna(defense)
            else ""
        )
        st.caption(
            f"전력 0.0 = {season}시즌 역할별 공격/투구 전력의 비교 집단 최저값"
            f"이며 결측치가 아닙니다{defense_text}."
        )

    result = simulate(
        team_players,
        selected_id,
        predict_win_rate,
        rank_predictor=make_rank_predictor(season_players),
    )

    section("순위 변동 패널", f"{season}시즌 features_v1 기반", icon="target")
    # st.metric 3개로는 "이탈 전 → 후"의 인과가 안 읽힌다. 좌우 배치 + 가운데
    # 변화량으로 바꿔서 한 줄로 이야기가 되게 한다.
    st.markdown(
        impact_panel_html(
            result.current_win_rate * 100,
            result.after_departure_win_rate * 100,
            label_before="현재 예상 승률",
            label_after="이탈 후 예상 승률",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        impact_panel_html(
            float(result.rank_before),
            float(result.rank_after),
            label_before="현재 예상 순위",
            label_after="이탈 후 예상 순위",
            unit="위",
            decimals=0,
            higher_is_better=False,  # 순위는 숫자가 작을수록 좋다
            delta_words=("하락", "상승"),
            delta_unit="계단",
        ),
        unsafe_allow_html=True,
    )
    st.caption(win_rate_caption())
    # DB 폴백이 조용히 일어나면 안 된다 — 출처를 항상 표시한다
    _src = source_caption()
    if _src:
        st.caption(_src)

    top_risk = ranked.dropna(subset=["departure_risk"]).nlargest(3, "departure_risk")
    if not top_risk.empty and sort_label.endswith("이탈위험순"):
        section("이탈위험 TOP 3", "모델 추정 — 인과관계 단정 아님, 아래 근거 수치 참고", icon="shield")
        rc = st.columns(3)
        for col, (_, r) in zip(rc, top_risk.iterrows()):
            with col:
                badge_html = reason_badge_html(r.get("reason_tag", ""))
                evidence = {
                    "age": r.get("age"),
                    "exp": r.get("exp"),
                    "g_chg": r.get("g_chg"),
                    "overall_score_delta": r.get("_ev_overall_score_delta"),
                    "reason_injury_score": r.get("_ev_reason_injury_score"),
                }
                # 선수 얼굴을 카드 맨 위에 — 이름만 있는 카드보다 "이 사람이
                # 떠날 수 있다"가 훨씬 실감나게 읽힌다. 위험도에 따라 테두리 색이
                # 바뀌고, 링이 맥동한다.
                risk_v = float(r["departure_risk"])
                photo = headshot_url(str(r["player_id"]), photo_lookup)
                tone = _risk_tone(risk_v)
                face = (
                    f'<img src="{photo}" alt="" loading="lazy" '
                    "onerror=\"this.style.visibility='hidden'\"/>" if photo else ""
                )
                col.markdown(
                    f'<div class="gm-card gm-risk-card" style="text-align:center;--tone:{tone}">'
                    f'<div class="gm-risk-face">{face}</div>'
                    f'<div style="font-weight:800;font-size:14.5px">{r["이름"]}</div>'
                    f'<div class="gm-kpi-v" style="color:{tone};font-size:22px;margin:4px 0">'
                    f'{risk_v:.0%}</div>'
                    f'{badge_html}'
                    f'<div style="text-align:left">{reason_proba_html(r.get("reason_proba") or {}, top_n=2)}</div>'
                    f'{reason_explain_html(r.get("reason_tag", ""), evidence, reason_thresholds, compact=True)}'
                    f'{evidence_html(evidence, reason_thresholds)}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── 다이아몬드 라인업 — 포지션별 대표 선수를 그라운드에 배치 ──
    # 포지션별로 "전력이 가장 높은 선수"를 그 자리의 주전으로 본다. 표에서는
    # 안 보이던 "어느 포지션이 비었나 / 어디가 위험한가"가 한눈에 들어온다.
    section("다이아몬드 라인업", f"{season}시즌 · 포지션별 주전(전력 1위) · 링이 붉을수록 이탈위험 높음", icon="team")
    lineup_rows: list[dict] = []
    if "primary_position" in team_players.columns:
        # 예전에는 포지션당 1명만 남기고 잘라서(drop_duplicates) 외야수 8명 중
        # 1명만 그려졌다. 이제 전력 내림차순 전체를 넘기고, 자리 배분(외야 3자리)은
        # diamond_lineup_svg 가 한다.
        starters = (
            team_players.dropna(subset=["primary_position"])
            .sort_values("overall_score", ascending=False)
        )
        for _, r in starters.iterrows():
            lineup_rows.append({
                "position": r["primary_position"],
                "name": r["이름"],
                "photo": headshot_url(str(r["player_id"]), photo_lookup),
                "ovr": float(r["overall_score"]) if pd.notna(r["overall_score"]) else 0.0,
                "risk": float(r["departure_risk"]) if pd.notna(r["departure_risk"]) else 0.0,
            })
    if lineup_rows:
        st.markdown(diamond_lineup_svg(lineup_rows), unsafe_allow_html=True)
        st.caption(
            "포지션별 전력 1위 선수를 주전으로 표시합니다. 점선 슬롯은 해당 포지션에 "
            "등록된 선수가 없다는 뜻입니다(가짜로 채우지 않습니다)."
        )
    else:
        st.caption("primary_position이 없어 라인업 배치도를 표시할 수 없습니다.")

    section("전력 로스터", sort_label, icon="team")
    # st.dataframe 은 전력을 49.296755655208635 처럼 뿌리고 이탈위험이 색으로
    # 안 읽힌다 — 막대 리스트로 바꿔서 훑기만 해도 분포가 보이게 한다.
    roster_rows = [
        {
            "name": r["이름"],
            "photo": headshot_url(str(r["player_id"]), photo_lookup),
            "role": ROLE_LABEL.get(r.get("role"), r.get("role") or ""),
            "ovr": r.get("overall_score"),
            "risk": r.get("departure_risk"),
            "tag": (
                icon(REASON_DISPLAY[r["reason_tag"]][1], 11) + " " + REASON_DISPLAY[r["reason_tag"]][0]
                if r.get("reason_tag") in REASON_DISPLAY else ""
            ),
        }
        for _, r in ranked.head(20).iterrows()
    ]
    st.markdown(roster_list_html(roster_rows), unsafe_allow_html=True)

    if st.button("선수 리포트에서 대체 후보 보기", type="primary"):
        st.switch_page("pages/2_Player_Report.py")
