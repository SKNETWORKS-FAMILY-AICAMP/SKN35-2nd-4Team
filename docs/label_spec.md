# 라벨 스펙 (Rev.5 — 이탈 관측·경로·복귀·설명 분리)

이 문서는 `src/features/labels.py`와 `src/models/reason.py`가 앞으로 따라야 할
목표 명세다. Rev.5의 목적은 이탈확률이 높은 선수에게도 관측 여부와 추정
근거를 구분하여 설명하고, 기존 `unknown`에 섞여 있던 검열·부상 미복귀·
성적·베테랑 요인을 분리하는 것이다.

현재 코드에 이미 구현된 Rev.4 컬럼은 하위 호환을 위해 유지한다. Rev.5 신규
컬럼은 단계적으로 구현하며, 코드와 문서가 일시적으로 다른 동안에는 아래
"구현 상태"를 따른다.

## 구현 상태

| 구분 | 상태 |
|---|---|
| `y_departed`, `y_path`, `y_fa_release`, `y_returned`, `y_core_departed` | Rev.4 구현 완료 |
| `primary_reason`, `reason_tags`, `evidence_level` | Rev.5 1차 재라벨링 구현 완료 |
| `label_status`, `y_transaction_type` | Rev.5 구현 예정 |
| `injury_recovery_status`, `injury_duration_bucket`, `injury_recurrent` | Rev.5 구현 예정 |
| `early_career_move`, `stable_performance_move` | Rev.5 구현 완료 |
| 영입가치·방출위험·베테랑가치·은퇴위험 세분화 | 트랜잭션 근거 확보 후 2차 구현 |
| 현재 시즌 원인 추론 | 화면 연결 구현, 신규 모델 재학습 필요 |

---

## 1. 설계 원칙

1. **관측 사실과 모델 추정을 분리한다.** 소속·트랜잭션·IL 등재·복귀일은
   관측값이고, "잘해서 영입", "못해서 방출", "은퇴 위험"은 데이터 기반
   연관 요인 또는 모델 추정값이다.
2. **`NaN`은 클래스가 아니다.** 미래 관측기간이 부족한 행은
   `label_status=censored`로 관리하며, `insufficient_evidence`와 섞지 않는다.
3. **확률과 원인 신뢰도를 분리한다.** 이탈확률 90%가 원인 신뢰도 90%를
   뜻하지 않는다. 화면에는 두 값을 별도로 표시한다.
4. **원인은 다중 라벨을 허용한다.** 부상·성적 하락·커리어 단계가 동시에
   존재할 수 있으므로 `reason_tags`에는 모든 활성 요인을 보존한다.
5. **인과관계를 단정하지 않는다.** "부상 때문에 이탈"이 아니라
   "IL 등재 기록이 높은 이탈 위험과 함께 관측됨"으로 표현한다.
6. **임계값은 훈련 구간에서만 계산한다.** 시즌·역할별 분위수도 검증·테스트
   구간을 사용하지 않는다.
7. **현재 시즌은 추론 대상으로 유지한다.** 정답 라벨이 아직 없어도 저장된
   원인 모델로 예상 원인과 신뢰도를 제공한다.

---

## 2. 공통 관측 상태 — `label_status`

`label_status`는 정답값과 별도로 라벨을 신뢰할 수 있는 상태인지 나타낸다.

| 값 | 의미 |
|---|---|
| `observed` | 필요한 후속 시즌 또는 공식 트랜잭션이 관측되어 확정 가능 |
| `provisional` | 시즌 진행 중 자료로 임시 판정. 학습 정답에서는 제외 |
| `censored` | 필요한 후속 관측기간이 끝나지 않아 판정 불가 |

규칙:

- `censored`와 `provisional` 행은 지도학습 정답에서 제외한다.
- 데이터에 다음 시즌 기록이 없다는 사실만으로 `league_exit`, 방출, 은퇴를
  확정하지 않는다.
- `insufficient_evidence`는 관측기간이 충분하지만 원인 신호가 부족한 경우에만
  사용한다.

---

## 3. L1 — `y_departed` (이탈 여부)

**정의**: 다음 시즌에도 같은 `franch_id`에 소속되는가.

- `0.0` = 다음 시즌 동일 프랜차이즈 잔류
- `1.0` = 다음 시즌 다른 프랜차이즈 또는 관측 완료 후 리그 이탈
- `NaN` = `label_status`가 `provisional` 또는 `censored`

2026년 자료로 2025년 라벨을 만들 때는 다음 원칙을 적용한다.

- 시즌이 종료되고 소속·트랜잭션이 확정된 자료는 `observed`로 승격한다.
- 시즌 중간 스냅샷은 `provisional`이며 모델 학습에서 제외한다.
- 2026년 기록이 없더라도 관찰기간이 끝나기 전에는 이탈로 확정하지 않는다.

---

## 4. L2 — 이탈 경로

### 4.1 `y_path` — Rev.4 호환 컬럼

| 값 | 조건 |
|---|---|
| `trade` | 시즌 중 복수 스틴트가 관측된 이동 proxy |
| `offseason_move` | 다음 시즌 다른 프랜차이즈 기록 관측 |
| `league_exit` | 후속 관찰 완료 후 리그 기록 없음 |

`n_stint >= 2`는 이동 관측 proxy이지 공식 트레이드 확정값이 아니다.

### 4.2 `y_transaction_type` — Rev.5 신규 관측 컬럼

공식 트랜잭션 또는 신뢰 가능한 거래 데이터가 있을 때만 부여한다.

| 값 | 의미 |
|---|---|
| `trade` | 공식 트레이드 관측 |
| `fa_move` | FA 자격 및 타 구단 계약 관측 |
| `release_dfa` | 방출·DFA 관측 |
| `waiver_move` | 웨이버 이동 관측 |
| `league_exit` | 관찰 완료 후 MLB 기록 없음 |
| `retirement_confirmed` | 공식 은퇴 발표 또는 은퇴 트랜잭션 확인 |
| `unknown_departure` | 이탈 사실은 확인됐지만 거래 유형 근거 부족 |

공식 자료가 없으면 `fa_move`, `release_dfa`, `retirement_confirmed`를 확정
라벨로 만들지 않는다.

### 4.3 `y_fa_release` — Rev.4 호환 추정 컬럼

| 값 | 의미 |
|---|---|
| `release_certain` | `exp < 6`으로 FA 자격이 제도상 불가능한 오프시즌 이동 |
| `fa_est` | FA 가능성이 높은 proxy |
| `release_est` | 방출 가능성이 높은 proxy |

`fa_est`와 `release_est`는 관측 트랜잭션이 아니라 추정값이므로
`y_transaction_type`과 동일하게 취급하지 않는다.

---

## 5. L3 — 복귀 라벨

### 5.1 `y_returned` — 리그 복귀

`y_path == league_exit`인 선수에게만 적용한다. 기본 관찰창은 이탈 후 2~3시즌이다.

- `1.0` = 관찰창 안에 MLB 기록 재등장
- `0.0` = 관찰창이 완료됐지만 MLB 기록 없음
- `NaN` = 관찰창 미완료(`censored`)

### 5.2 `injury_recovery_status` — 부상 복귀 상태

| 값 | 조건 |
|---|---|
| `no_injury_record` | 신뢰 가능한 수집 구간에서 IL 기록이 관측되지 않음 |
| `returned_same_season` | IL 등재 후 같은 시즌 복귀일 또는 경기 출전 관측 |
| `returned_next_season` | 다음 시즌 복귀 출전 관측 |
| `no_return_observed` | 정해진 관찰창이 끝났지만 복귀 출전 없음 |
| `censored` | 부상 이후 관찰창이 아직 끝나지 않음 |

`no_return_observed`는 은퇴 또는 부상으로 인한 이탈을 확정하지 않는다. 다른
트랜잭션·성적·나이 근거와 결합하는 연관 신호로만 사용한다.

### 5.3 `injury_duration_bucket` — 실제 결장기간

IL 등재일부터 복귀일 또는 부상 이후 첫 경기일까지의 실제 날짜 차이를 사용한다.

| 값 | 실제 결장기간 |
|---|---|
| `short` | 15일 이하 |
| `medium` | 16~59일 |
| `long` | 60일 이상 |
| `unknown` | 복귀일 또는 시작일이 없어 계산 불가 |

7일·10일·15일·60일 IL 지정은 심각도 보조정보로 사용할 수 있지만 실제
결장일수와 동일하게 취급하지 않는다.

### 5.4 `injury_recurrent`

- `1` = 같은 시즌 `il_stint_count >= 2`
- `0` = 같은 시즌 IL 0~1회

반복 여부는 복귀기간과 동시에 존재할 수 있으므로 별도 컬럼으로 유지한다.

---

## 6. L1' — `y_core_departed` (핵심 이탈위험 타깃)

Rev.4 호환을 위해 유지한다. 구단 결정이 확실한 `release_certain`은 선수 주도
이탈위험 학습에서 제외하고, 나머지는 `y_departed`를 사용한다.

```python
confirmed_release = y_fa_release == "release_certain"
y_core_departed = y_departed.where(~confirmed_release, NaN)
```

Rev.5에서는 `y_transaction_type`이 확정되면 다음처럼 관측 근거를 우선한다.

- `release_dfa`: 선수 이탈위험 모델 타깃에서 제외하거나 별도 구단 의사결정
  모델의 양성 클래스로 사용
- `trade`, `fa_move`: 선수 이동위험 모델의 양성 클래스로 사용
- `unknown_departure`: 핵심 모델 학습 포함 여부를 실험으로 비교

---

## 7. 이탈 연관 원인 — Rev.5 1차 구현

원인 태그는 관측된 이탈을 설명하는 약지도 라벨이자, 현재 시즌 고위험 선수에게
예측되는 설명값이다. `reason_tags`에는 여러 태그를 함께 저장하고
`primary_reason`은 화면 표시용 최상위 태그로만 사용한다.

| 태그 | 데이터 기반 판정 원칙 |
|---|---|
| `injury_associated` | 확인된 IL 기록과 부상 위험점수가 임계값 이상 |
| `performance_decline` | 전력 변화 또는 출전비중 변화가 훈련구간 하위 임계값 이하 |
| `career_stage` | `league_exit`이면서 나이 또는 경력이 훈련구간 상위 임계값 이상 |
| `early_career_move` | 위 세 원인이 없고 경력 `exp <= 1`인 이탈자 |
| `stable_performance_move` | 위 세 원인과 저연차 태그가 없고 전력 변화가 관측되며 `overall_score_delta >= 0` |
| `mixed` | 부상·성적하락·커리어 단계 중 2개 이상 동시 활성화 |
| `unknown` | 관측 완료 이탈자지만 위 어느 조건도 충족하지 못함 |

### 7.1 대표 태그 우선순위

1. 부상·성적하락·커리어 단계 태그를 먼저 계산한다.
2. 이 중 2개 이상이면 `mixed`, 1개면 해당 태그를 대표 태그로 쓴다.
3. 강한 원인이 하나도 없을 때만 `early_career_move`를 적용한다.
4. 저연차도 아닐 때만 `stable_performance_move`를 적용한다.
5. 마지막 잔여 집단만 `unknown`으로 유지한다.

새 두 태그를 잔여 집단에만 적용하는 이유는 동일 선수가 저연차이면서 성적도
상승한 경우 `mixed`가 인위적으로 증가하는 것을 막기 위해서다. 이 우선순위는
`src/models/reason.py`와 회귀 테스트에 고정한다.

### 7.2 2026-08-30 최신 데이터 기준 분포

| 태그 | 건수 | 비율 |
|---|---:|---:|
| `performance_decline` | 2,715 | 23.74% |
| `early_career_move` | 2,215 | 19.37% |
| `mixed` | 1,923 | 16.82% |
| `stable_performance_move` | 1,906 | 16.67% |
| `unknown` | 1,379 | 12.06% |
| `career_stage` | 810 | 7.08% |
| `injury_associated` | 488 | 4.27% |

기존 `unknown` 5,500건(48.09%) 중 저연차 2,215건과 전력 유지·상승이 관측된
1,906건을 분리했다. 최신 부상 위험점수 재계산 후 남은 1,379건은 근거가
부족하므로 억지로 다른 원인에
배정하지 않는다.

### 7.3 원인 근거 수준 — `reason_evidence_level`

| 값 | 의미 |
|---|---|
| `confirmed_event` | 공식 트랜잭션·IL·복귀·은퇴 자료가 존재 |
| `strong_proxy` | 관측 이동과 강한 성적·부상·경력 신호가 함께 존재 |
| `model_estimate` | 현재 시즌 미관측 선수에 대한 원인 모델 예측 |
| `insufficient` | 관찰 완료 후에도 설명 신호 부족 |
| `censored` | 원인 평가에 필요한 후속 관찰기간 부족 |

### 7.4 2차 세분화 후보

`acquisition_value`, `performance_release_risk`, `veteran_value`,
`retirement_risk`는 비즈니스 설명력이 더 높은 후보지만, 현재 성적·경력만으로
영입·방출·은퇴를 확정하면 순환논리가 생긴다. 공식 트랜잭션 유형 또는 독립된
근거가 연결된 뒤 별도 2차 태그로 구현한다.

---

## 8. 현재 시즌 원인 추론

현재 시즌은 `y_departed`가 `NaN`이어도 이탈확률이 높으면 원인 모델 추론
대상에 포함한다. 학습용 `to_reason_xy()`와 별도로 라벨을 요구하지 않는 추론용
입력 함수를 둔다.

화면 표시 순서:

1. `departure` 모델의 이탈확률
2. `reason` 모델의 최상위 원인과 원인 신뢰도
3. 관측 근거(IL 횟수, 전력·출전 변화, 나이·경력)
4. `reason_evidence_level=model_estimate` 표시

`primary_reason == unknown`이거나 최고 원인 확률이 서비스 임계값보다
낮을 때만 "판단 근거 부족"을 표시한다. 미래 라벨이 없다는 이유만으로 이 문구를
표시하지 않는다.

---

## 9. 2026년 데이터 반영 정책

- 시즌 종료 전 데이터는 `provisional`로 저장하고 학습 정답에서 제외한다.
- 시즌 종료 후 2025년 L1·L2를 재생성하고 2025년을 추가 시계열 홀드아웃으로
  평가한다.
- `y_returned`와 `no_return_observed`는 정의된 관찰창이 끝날 때까지
  `censored`를 유지한다.
- Lahman ID와 MLBAM ID 매핑, 데이터 기준일(`label_as_of_date`)과 출처를 함께
  기록한다.
- 2026년 자료로 2025년을 확정하더라도 2026년 예측에는 다시 미래 라벨이 없으므로
  현재 시즌 원인 추론 기능을 제거하지 않는다.

---

## 10. 평가 및 결과서 해석

- 불균형 다중분류는 Accuracy보다 Macro F1과 클래스별 Recall/F1을 우선한다.
- 원인 모델은 약지도 규칙을 학습하므로 높은 성능을 실제 인과 정확도로 표현하지
  않는다.
- `early_career_move`는 `exp`로 라벨을 만들고 모델 입력에도 `exp`를 사용하므로
  F1이 매우 높게 나오는 것은 규칙 복원에 가깝다. 이 클래스의 높은 F1을 독립적인
  예측력으로 홍보하지 않는다.
- 직접 라벨 생성에 사용한 피처를 모델 입력에서도 사용한 결과와 제외한 결과를
  함께 비교해 규칙 복원 성능과 독립 신호 성능을 구분한다.
- `censored`/`provisional` 행은 학습·검증·테스트 정답에서 모두 제외한다.
- 화면에는 "관측 원인", "강한 proxy", "모델 추정"을 구분하여 표시한다.

## 11. 절대 하지 않을 것

- 미래 데이터가 없다는 이유로 `y_departed=0` 또는 `league_exit`로 채우기
- 부상 미복귀를 곧바로 은퇴 확정으로 저장하기
- 높은 전력과 이적만으로 "타팀이 원해서 영입"을 확정 사실로 표현하기
- 낮은 전력과 이탈만으로 방출을 확정하기
- `censored`를 `insufficient_evidence` 또는 `unknown`으로 합치기
- 이탈확률을 원인 모델 신뢰도로 표시하기
