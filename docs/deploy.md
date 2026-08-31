# 배포 가이드 — Streamlit Community Cloud + Supabase

## 현재 구조

```
Streamlit Cloud (앱)  ──읽기──▶  리포지토리 안의 로컬 파일
                                  data/final/features_v1.parquet
                                  models/*.pkl, *.pt

Supabase (Postgres)   ──────▶  아직 앱이 읽지 않음 (준비만 완료)
```

앱은 **리포에 들어있는 parquet/모델 파일을 직접 읽는다.** Supabase 는 스키마와
적재까지 끝나 있지만(`player_season` 등 11개 테이블) 화면은 아직 연결되어 있지
않다 — `src/storage/queries.py` 가 그 전환을 위한 준비물이다. 즉 **지금 배포하는
데 Supabase 는 필수가 아니다.**

---

## 1. Streamlit Community Cloud 배포

### 준비된 것
| 파일 | 역할 |
|---|---|
| `requirements.txt` | 런타임 의존성 (pyproject 와 별도) |
| `.streamlit/config.toml` | 다크 테마 — Cloud 가 자동으로 읽는다 |
| `.streamlit/secrets.toml.example` | 시크릿 템플릿 (실제 파일은 gitignore) |

### 배포 설정
- **Repository**: `SKNETWORKS-FAMILY-AICAMP/SKN35-2nd-4Team`
- **Branch**: 배포용 브랜치 (예: `main`)
- **Main file path**: `app/Home.py`
- **Python version**: 3.12 (`.python-version` 과 맞출 것)

### Secrets (Supabase 를 연결할 때만)
앱 대시보드 → **Settings → Secrets** 에 붙여넣는다. 파일을 올리는 게 아니다.
내용은 `.streamlit/secrets.toml.example` 참고.

`supabase_client.py` 는 환경변수(`DATABASE_URL`)와 `st.secrets` 를 **둘 다**
찾는다. 로컬은 `.env`, Cloud 는 Secrets — 호출부는 바꿀 필요 없다.

### requirements.txt 를 따로 두는 이유
`pyproject.toml` 에는 분석·크롤링·CV 실습 패키지가 함께 들어 있는데 배포되는
앱은 그중 하나도 쓰지 않는다. 그대로 설치하면:
- 설치본이 수 GB (torch 501MB, cv2 138MB, mediapipe 99MB)
- `pyautogui` 는 디스플레이가 필요해 헤드리스 환경에서 실패

`requirements.txt` 는 `app/` 진입점에서 실제로 도달하는 모듈만 추린 것이다.
**의존성을 추가/제거하면 이 파일도 같이 손봐야 한다.**

torch 는 CPU 휠(`--extra-index-url`)을 쓴다 — 기본 인덱스의 CUDA 빌드는 2GB가
넘어 배포 한도를 넘긴다. 앱에서 torch 가 필요한 곳은
`models/recommend_autoencoder.pt` 하나뿐이다.

---

## 2. Supabase

### 현재 상태 (확인 완료)
연결 정상, 테이블 11개 적재됨:
`allstar, appearances, batting_stats, fielding_stats, franchises, games,
pitching_stats, player_season, players, team_season, teams`

### 관련 파일
| 파일 | 역할 |
|---|---|
| `database/schema.sql` | 테이블 정의 |
| `database/load.py` | CSV → Supabase 적재 |
| `src/storage/supabase_client.py` | 연결 (psycopg2 직결) |
| `src/storage/queries.py` | 조회 함수 — 화면 전환용 준비물 |

### 연결 확인
```bash
uv run python -m src.storage.supabase_client
```

### 화면을 DB 로 전환하려면
각 페이지의 `pd.read_parquet(FEATURES_PATH)` 를 `queries.py` 함수로 바꾸면 된다
(반환 스키마를 동일하게 맞춰뒀다). 다만 **`features_v1` 은 아직 테이블이 없다** —
`player_season` 에서 파생되는 계약 산출물이라, 전환하려면 그 테이블을 먼저
만들거나 앱에서 파생 로직을 돌려야 한다.

---

## 3. 배포 전 확인할 것

### 리포 용량 (주의)
추적 파일 258MB, `.git` 137MB. Cloud 가 매 배포마다 클론하므로 빌드가 느리다.
줄일 수 있는 것:

| 대상 | 크기 | 비고 |
|---|---|---|
| `.cache/transactions-*.csv` | 66MB | **API 에서 재생성 가능** — 커밋 불필요 |
| `models/reason_rf.pkl` | 32MB | RandomForest. 트리 수/깊이로 줄일 수 있음 |
| `data/raw/lahman/*.csv` | ~25MB | 원본. 앱은 `data/final/` 만 읽는다 |

`.gitignore` 에 `.cache/transactions-*.csv` 를 넣어뒀지만 **이미 커밋된 파일은
gitignore 로 빠지지 않는다.** 실제로 빼려면:
```bash
git rm -r --cached .cache/transactions-*.csv
```
(로컬 파일은 남고 추적만 해제된다. 재수집은 `mlb_injury_pipeline.py` 가
연도별 캐시를 다시 만든다.)

### 메모리
Community Cloud 는 리소스 한도가 있다. 앱이 상시 메모리에 올리는 것:
- `reason_rf.pkl` 32MB + `recommend_knn.pkl` 6MB + torch 런타임
- `features_v1.parquet` (24,578행)

`@st.cache_resource` 로 모델을, `@st.cache_data` 로 데이터를 캐싱하고 있어
재실행마다 다시 읽지는 않는다. 다만 **reason_rf 32MB 가 가장 큰 부담**이라
메모리 문제가 나면 여기부터 줄이는 게 효과적이다.

### 외부 CDN 의존
선수 헤드샷과 구단 로고는 MLB CDN 을 핫링크한다(`img.mlbstatic.com`,
`www.mlbstatic.com`). 파일을 저장하지 않으므로 리포 용량에는 영향이 없고,
로드 실패 시 `onerror` 로 숨겨져 레이아웃이 깨지지 않는다.
