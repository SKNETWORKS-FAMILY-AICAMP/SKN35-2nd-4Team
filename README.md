# ⚾ BMS (Baseball Manager Simulation)

MLB 30개 구단 중 하나를 맡아 **오늘 경기 승부를 예측**하고, **이탈 위험이 있는 선수를 파악**하고, **대체 선수를 추천**받아 **다음 시즌 팀 전력**까지 시뮬레이션해보는 야구 데이터 분석 웹앱입니다. Lahman 야구 데이터베이스(2009~2025)와 MLB Stats API(2026 진행 시즌)를 바탕으로, 5가지 예측 태스크마다 머신러닝·딥러닝 모델을 각각 학습해 총 13개 모델을 비교합니다.

> SKN35-2nd-4Team 팀의 SK네트웍스 Family AI캠프 2차 프로젝트입니다.

Streamlit으로 제작되었으며 [Streamlit Community Cloud](https://streamlit.io/cloud) 배포를 기준으로 설계되어 있습니다.

## 목차

- [프로젝트 개요](#프로젝트-개요)
- [주요 기능](#주요-기능)
- [예측 모델](#예측-모델)
- [기술 스택](#기술-스택)
- [프로젝트 구조](#프로젝트-구조)
- [데이터](#데이터)
- [시작하기](#시작하기)
- [배포](#배포)
- [테스트](#테스트)
- [설계 원칙](#설계-원칙)
- [알려진 한계 / TODO](#알려진-한계--todo)
- [팀](#팀)

## 역할
## 역할

<table align="center">

<!-- ===================== -->
<!--        1행 : 3명       -->
<!-- ===================== -->

<tr>

<!-- 이승희 -->
<td align="center" width="33.33%" valign="top">

<h3>🔴 이승희</h3>

<sub><b>@lshee9008</b></sub>

<br><br>

<img 
src="https://github.com/user-attachments/assets/10848653-7a3b-4454-89bc-ed90b69587d5" 
width="220"
/>

<br><br>

<b>🖥️ UI & 실험 관리 인프라 구축</b>

<br><br>

시계열 회귀 기반의<br>
ML · DL 모델 비교 및 검증

</td>


<!-- 나치훈 -->
<td align="center" width="33.33%" valign="top">

<h3>🔵 나치훈</h3>

<sub><b>@Nachihun</b></sub>

<br><br>

<img 
src="https://github.com/user-attachments/assets/e12965a5-52b9-4d08-83ba-767d9dace284" 
width="220"
/>

<br><br>

<b>⚙️ 다중 소스 데이터 파이프라인</b>

<br><br>

팀 전력 데이터를 기반으로 한<br>
승률 · 경기 예측 모델 구축

</td>


<!-- 김도영 -->
<td align="center" width="33.33%" valign="top">

<h3>🟡 김도영</h3>

<sub><b>@Do-0-K</b></sub>

<br><br>

<img 
src="https://github.com/user-attachments/assets/4b5535fe-3265-4343-b9f2-0b0138d7b842" 
width="220"
/>

<br><br>

<b>📊 야구 도메인 데이터 피처화</b>

<br><br>

클래스 불균형을 고려한<br>
선수 이탈 예측 모델링

</td>

</tr>


<!-- ===================== -->
<!--       행 간격         -->
<!-- ===================== -->

<tr>
<td colspan="3" height="50"></td>
</tr>


<!-- ===================== -->
<!--     2행 : 가운데 2명   -->
<!-- ===================== -->

<tr>

<td colspan="3" align="center">

<table align="center">

<tr>


<!-- 권준호 -->
<td align="center" width="300" valign="top">

<h3>🟣 권준호</h3>

<sub><b>@kweonjunho37-boop</b></sub>

<br><br>

<img 
src="https://github.com/user-attachments/assets/5c70e433-9b2e-4076-b928-5756384d68dd" 
width="220"
/>

<br><br>

<b>👑 TEAM LEADER · Labeling · Presentation</b>

<br><br>

팀장 · 발표 총괄<br>
관측 가능성 기반 4층 라벨 체계 설계<br>
제도 규칙을 활용한 준지도 라벨링

</td>


<!-- 두 카드 사이 간격 -->
<td width="100"></td>


<!-- 유지호 -->
<td align="center" width="300" valign="top">

<h3>🟢 유지호</h3>

<sub><b>@dbdbdb123</b></sub>

<br><br>

<img 
src="https://github.com/user-attachments/assets/42da58d7-9f05-4bd0-907b-695659f6b0ec" 
width="220"
/>

<br><br>

<b>🎯 Recommendation · Decision Simulator</b>

<br><br>

예측 모델을<br>
의사결정 시뮬레이터로 통합하는<br>
추천 시스템 설계

</td>


</tr>

</table>

</td>

</tr>

</table>
## 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 주제 | MLB 구단 단장(General Manager) 관점의 선수단 운영 시뮬레이션 |
| 데이터 범위 | Lahman DB 2009~2025시즌(완결) + MLB Stats API 2026시즌(진행 중) |
| 태스크 수 | 5개 (승부예측 / 이탈 예측 / 이탈 원인 / 다음 시즌 전력 / 대체 선수 추천) |
| 모델 수 | 태스크당 ML + DL 최소 1쌍, 총 13개 모델을 `models/registry/`에 등록 |
| 프런트엔드 | Streamlit 멀티페이지 앱 (`app/`) |
| 데이터베이스 | Supabase(PostgreSQL) — 스키마·적재 완료, 화면 연동은 준비 단계 |

## 주요 기능

### 1. 진입 화면 — `app/Home.py`
- **오늘 경기 승부 예측 챌린지**: 사용자가 먼저 승자를 고르면 4개 모델(LogReg·RandomForest·XGBoost·PyTorch MLP) 앙상블의 예측이 공개되는 인터랙션. 미국 동부시(EDT) 기준으로 "오늘 경기"를 판별하고 한국 시간도 함께 표기합니다.
- **예상 순위 리빌**: 시즌 예측 순위(또는 최신 완결 시즌 실제 승률)를 버튼 클릭으로 공개.
- **구단 선택**: 미국 지도에서 조명탑(마커) 클릭 또는 목록에서 30개 구단 중 하나를 선택해 단장 모드에 진입.

### 2. 구단 상황실 — `app/pages/1_Club_Operations_Center.py`
- 선택한 구단의 로스터, 팀 전력(공격/투구/수비), 예상 승률을 확인.
- 선수 이탈 시 **What-if 시뮬레이션**(`src/service/simulation.py`)으로 팀 전력 변화를 즉시 계산.

### 3. 선수 리포트 — `app/pages/2_Player_Report.py`
- 선수별 이탈 위험도, 이탈 연관 요인(부상/성적 하락/커리어 단계 등), 다음 시즌 전력 예측 추이.
- KNN + Autoencoder + LightGBM LambdaRank 정책 랭커를 결합한 **대체 선수 추천**과 슬롯 선정.

### 4. 모델 정보 — `app/pages/3_Model_Information.py`
- 5개 태스크 × ML/DL 계획 대비 실제 학습·등록된 모델 현황을 투명하게 공개(미학습 모델은 "미학습"으로 표시, 가짜 성능 숫자를 채우지 않음).

## 예측 모델

`src/models/base.py`의 공통 `BaseModel` 인터페이스를 상속하는 5개 태스크입니다. 담당자(A~E)별로 모델을 저장하면 `models/registry/{name}.json`이 자동 생성되어 병합 충돌 없이 성능을 취합합니다.

| 태스크 | 담당 | 설명 | ML 모델 (테스트 성능) | DL 모델 (테스트 성능) |
|---|---|---|---|---|
| **win_rate** (경기 승부예측) | A | 경기 시작 전 정보(누적 승률·휴식일수·최근 10경기 등)만으로 홈팀 승패 예측 | LogReg — Acc 0.576 / AUC 0.608 · XGBoost — Acc 0.580 / AUC 0.607 | PyTorch MLP — Acc 0.577 / AUC 0.606 |
| **departure** (이탈 예측) | B | 선수의 다음 시즌 이탈(`y_core_departed`) 이진분류 | LightGBM(Optuna) — Acc 0.744 / AUC 0.828 | MLP — Acc 0.744 / AUC 0.825 · **LSTM(시퀀스)** — Acc 0.738 / AUC 0.826 |
| **reason** (이탈 원인) | C | 이탈 연관 요인 다중분류(부상/성적하락/커리어단계 등 7클래스, 약지도학습) | RandomForest — Acc 0.882 / macro F1 0.861 | MLP — Acc 0.838 / macro F1 0.799 |
| **strength** (다음 시즌 전력) | D | 유일한 회귀 태스크. 다음 시즌 `overall_score` 예측 | XGBoost(lag 피처) — R² 0.558 / MAE 11.63 | MLP — R² 0.472 · **LSTM(5시즌 시퀀스)** — R² 0.510 |
| **recommend** (대체 선수 추천) | E | 이탈 선수를 대체할 후보를 검색·랭킹 | KNN 코사인 유사도, **정책 랭커(LightGBM LambdaRank)** — nDCG@5 0.337 | Autoencoder 잠재 유사도 — nDCG@5 0.286 |

> 성능 수치는 `models/registry/*.json`에 기록된 실제 테스트셋 결과이며, 앱의 "모델 정보" 화면에서 그대로 확인할 수 있습니다. `win_rate` 태스크는 경기 단위로 미래 정보 누수 없이(to-date 누적 승률만 사용) 피처를 만들기 때문에 이탈/전력 태스크보다 정확도가 낮게 나옵니다.

추가로 `game.py`는 2026시즌 잔여 경기를 위 앙상블로 예측해 `data/final/predictions/remaining_games_predictions.csv`를 생성하고, Home 화면의 "오늘 경기" 탭이 이를 그대로 읽습니다.

## 기술 스택

- **언어/런타임**: Python 3.12, [uv](https://docs.astral.sh/uv/) 프로젝트 매니저
- **웹 프레임워크**: Streamlit
- **ML/DL**: scikit-learn, XGBoost, LightGBM, PyTorch, Optuna(하이퍼파라미터 튜닝)
- **데이터 처리**: pandas, numpy, pyarrow(Parquet)
- **데이터베이스**: PostgreSQL(Supabase), SQLAlchemy, psycopg2
- **시각화**: Plotly, 커스텀 SVG/HTML(테마 컴포넌트)
- **데이터 수집**: MLB Stats API(공개 JSON API), Lahman Baseball Database, Chadwick Bureau Register(ID 크로스워크)

## 프로젝트 구조

```
SKN35-2nd-4Team/
├── app/                        # Streamlit 앱
│   ├── Home.py                 # 진입 화면 (오늘 경기 / 예상 순위 / 구단 선택)
│   ├── pages/
│   │   ├── 1_Club_Operations_Center.py
│   │   ├── 2_Player_Report.py
│   │   └── 3_Model_Information.py
│   └── ui/                     # 테마, CSS/JS, 데이터 로더, 위험도·승률 표시 컴포넌트
├── src/
│   ├── adapters/                # 외부 API 어댑터 (MLB Stats API)
│   ├── features/                # features_v1 스키마 정의(contract.py), 빌드 파이프라인, 라벨링, 전력 계산
│   ├── models/                  # 5개 태스크 모델 + 공통 BaseModel/registry
│   ├── service/                 # 시뮬레이션, 추천 스코어링, 레지스트리 조회 (Streamlit 비의존 순수 로직)
│   └── storage/                 # Supabase 연결 및 조회 함수
├── database/                    # schema.sql, load.py (CSV → Supabase 적재)
├── data/
│   ├── raw/                     # Lahman/KBO 원본
│   ├── processed/               # 중간 가공 데이터
│   └── final/                   # features_v1.parquet 등 앱이 실제로 읽는 산출물
├── models/                      # 학습된 모델 아티팩트(.pkl/.pt/.ubj) + registry/*.json
├── notebooks/                   # 태스크별 실험 노트북 (a_win_rate ~ e_recommend)
├── tests/                       # 라벨링·추천 스코어링 회귀 테스트
├── docs/                        # 배포 가이드, 라벨 스펙 문서
├── *.py (루트)                  # 2026시즌 데이터 수집·병합용 보조 스크립트
├── pyproject.toml / requirements.txt
└── .streamlit/                  # 다크 테마, secrets 템플릿
```

## 데이터

- **Lahman Baseball Database**: 선수·팀·타격/투구/수비/올스타 기록 원본 (2009~2025시즌으로 필터링해 사용).
- **MLB Stats API** (`statsapi.mlb.com`): 2026시즌 진행 중 스탯·일정·부상자 명단(IL)을 공개 JSON 엔드포인트로 수집. 비상업·개인/연구 목적 사용 범위 내에서만 사용합니다.
- **Chadwick Bureau Register**: Lahman `player_id` ↔ MLBAM ID 크로스워크 생성에 사용.
- **`features_v1.parquet`**: 위 데이터를 통합한 프로젝트 표준 피처 계약(`src/features/contract.py`). 학습/검증/테스트 시즌 분할은 각각 2009–2021 / 2022–2023 / 2024이며, 2025~2026은 라벨 신뢰도(관측/잠정/센서드)를 구분해 관리합니다.

원본 CSV 자체는 리포에 포함되어 있으나(`data/raw`), 실제 앱과 모델은 `data/final/`의 정제된 산출물만 사용합니다.

## 시작하기

### 요구 사항
- Python 3.12
- [uv](https://docs.astral.sh/uv/) (권장) 또는 pip

### 설치 및 실행 (uv)

```bash
git clone https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN35-2nd-4Team.git
cd SKN35-2nd-4Team

uv sync                      # pyproject.toml 기준 의존성 설치
cp .env.example .env         # 필요 시 DATABASE_URL 등 채우기 (Supabase 연동 시에만 필수)

uv run streamlit run app/Home.py
```

### pip으로 실행하는 경우

배포 환경과 동일하게, 앱 실행에 실제로 필요한 패키지만 정리된 `requirements.txt`를 사용합니다(전체 실습용 패키지는 `pyproject.toml` 참고).

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/Home.py
```

### 모델 재학습 / 파이프라인 스크립트

```bash
uv run python run_d.py                         # D(strength) 3개 모델 목업/실데이터 학습 + registry 등록 확인
uv run python -m src.storage.supabase_client   # Supabase 연결 확인
uv run python database/load.py --dry-run       # CSV → Supabase 적재 미리보기
```

각 태스크 모델(`src/models/win_rate.py`, `departure.py`, `reason.py`, `strength_ts.py`, `recommend.py`)은 모듈을 직접 실행하거나 `notebooks/`의 대응 노트북에서 재현할 수 있습니다.

## 배포

**Streamlit Community Cloud + Supabase** 구조를 기준으로 합니다. 앱은 리포에 포함된 `data/final/*.parquet`와 `models/*` 아티팩트를 직접 읽으며, Supabase는 스키마·적재까지 준비되어 있으나 화면은 아직 파일 기반으로 동작합니다.

- Main file path: `app/Home.py`
- Python 버전: 3.12 (`.python-version`)
- 배포용 의존성 목록은 `requirements.txt`(경량화된 목록)를 사용하고, `pyproject.toml`의 실습용 대용량 패키지(torch CUDA 빌드, mediapipe, opencv 등)는 제외됩니다.
- Supabase 연동 시크릿은 `.streamlit/secrets.toml.example`을 참고해 Cloud 대시보드의 Secrets에 등록합니다.

자세한 절차와 리포 용량 최적화 방법은 [`docs/deploy.md`](docs/deploy.md)를 참고하세요.

## 테스트

```bash
uv run python -m unittest discover tests
```

- `tests/test_reason_labels.py`: 이탈 원인 라벨 우선순위·병합 로직 회귀 테스트
- `tests/test_recommendation_scoring.py`: 추천 스코어링(비용 가중치, 팀 내 순위 등) 단위 테스트

## 설계 원칙

- **가짜 데이터로 화면을 채우지 않는다.** 아직 만들어지지 않은 예측 파일이 있으면 어떤 계산이 필요한지 명시하는 placeholder를 보여줍니다.
- **미래 정보 누수(data leakage) 방지.** `game.py`는 시즌 최종 승률 대신 "그 경기 시점까지의 누적 성적"만 사용합니다.
- **모델 추정과 관측 사실을 분리한다.** 이탈 원인 태그는 "~때문에 이탈함"이 아니라 "모델이 추정한 연관 요인"으로만 표시합니다(자세한 기준은 [`docs/label_spec.md`](docs/label_spec.md)).
- **생존 편향 인지.** 다음 시즌 전력 모델은 리그에 남은 선수만 학습하므로 낙관 편향이 있으며, 대체 선수 추천 시 이탈 모델과 결합해 보정합니다.
- **담당자별 파일 분리로 병합 충돌 최소화.** 모델 레지스트리를 태스크별 단일 파일이 아닌 모델별 JSON으로 저장합니다.

## 알려진 한계 / TODO

- `config/kbo.yaml`, `config/mlb.yaml`, `docs/data_inventory.md`는 아직 작성되지 않은 빈 파일입니다.
- Supabase 연동은 완료되었으나 화면(`app/pages/*`)은 아직 `features_v1.parquet`를 직접 읽습니다 — `src/storage/queries.py`로 전환이 남아 있습니다.
- `win_rate` 경기 단위 예측 모델의 정확도(~58%)는 상대적으로 낮은 편으로, 추가 피처 엔지니어링의 여지가 있습니다.
- 2026시즌은 진행 중 데이터이므로 관련 통계·순위는 시즌 종료 전까지 잠정치입니다.

## 팀

SK네트웍스 Family AI캠프 35기 2차 프로젝트 4팀이 BMS(Baseball Manager Simulation)를 만들었습니다. GitHub 기여자: `lshee9008`, `Do-0-K`, `dbdbdb123`, `kweonjunho`, `Nachihun`.

프로젝트 내부적으로 5개 예측 태스크(승부예측 · 이탈예측 · 이탈원인 · 다음시즌전력 · 추천)를 담당자 A~E로 나누어 병렬로 진행했습니다.
