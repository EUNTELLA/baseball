# 제구 성공 확률 예측 프로젝트 기본 지식

이 문서는 프로젝트를 이해하고 구현하기 전에 알아야 할 야구 데이터, 시계열 검증, 머신러닝 및 확률 예측의 핵심 개념을 정리합니다.

공식 문제는 2019~2024 시즌의 투구 단위 데이터와 학습용 Target으로 패턴을 학습하고, 2025 시즌 각 투구의 제구 성공 확률을 예측하는 것입니다.

## 1. 문제를 머신러닝 언어로 바꾸기

| 대회 표현 | 머신러닝 표현 |
| --- | --- |
| 투구별 입력 정보 | Feature, 독립변수, 설명변수 |
| 제구 성공 여부 | Target, Label, 종속변수 |
| 2019~2024 데이터 | Training data |
| 2025 데이터 | Test 또는 Inference data |
| 제구 성공 확률 | Positive class probability |
| 제출 결과 평가 | Generalization performance 측정 |

이 문제의 목표는 규칙으로 성공 여부를 직접 결정하는 것이 아니라, 과거 데이터에서 입력 정보와 Target 사이의 관계를 학습해 보지 못한 미래 투구의 성공 확률을 추정하는 것입니다.

## 2. 지도학습과 데이터 기반 학습

### 지도학습

지도학습은 입력과 정답이 함께 있는 데이터로 모델을 학습하는 방법입니다.

```text
경기 상황 + 이력 + Trackman 과거 로그 → 제구 성공 여부
```

학습이 끝나면 Target이 없는 2025 데이터에 모델을 적용합니다.

### 데이터 기반과 모델 기반

`Data-driven`과 `Model-driven`은 문맥에 따라 다르게 사용됩니다.

| 접근 | 의미 | 이 프로젝트의 예 |
| --- | --- | --- |
| 데이터 기반 | 데이터에서 패턴을 직접 학습 | Logistic Regression, Gradient Boosting |
| 규칙 기반 | 사람이 명시한 조건으로 판단 | `볼 3개이면 실패 확률 증가` 같은 고정 규칙 |
| 도메인 모델 기반 | 야구 지식과 가정을 수식 또는 구조로 표현 | 투수별 기본 제구력과 상황 효과를 분리한 통계 모델 |

좋은 결과는 한쪽만 선택하기보다 야구 도메인 지식으로 유효한 특징을 만들고, 데이터로 그 관계와 중요도를 학습할 때 나오는 경우가 많습니다.

## 3. 분류와 확률 예측

### 이진 분류

Target이 성공과 실패 두 상태라면 이진 분류 문제입니다. 실제 Target 값과 성공 클래스 인코딩은 공식 별첨 자료에서 확인해야 합니다.

### 클래스와 확률의 차이

```text
분류 결과: 성공
확률 결과: 성공 확률 0.73
```

대회는 성공 확률을 요구하므로 `predict()`의 클래스보다 `predict_proba()`의 확률이 중요합니다.

### 확률 보정

예측값이 `0.7`인 투구 100개 중 실제로 약 70개가 성공한다면 확률이 잘 보정됐다고 말합니다.

관련 키워드:

- Calibration
- Reliability diagram
- Platt scaling
- Isotonic regression
- CalibratedClassifierCV

확률 보정 역시 미래 검증 데이터를 학습에 사용하지 않도록 시간 순서 안에서 수행해야 합니다.

## 4. 시계열 데이터

### 시계열 또는 시간 순서 데이터란

관측치에 시간 순서가 있고 과거가 미래에 영향을 주는 데이터입니다. 투구 데이터는 시즌, 경기 날짜, 경기 내부 투구 순서를 가집니다.

일반적인 표 데이터처럼 모든 행을 무작위로 섞으면 미래 기록으로 과거를 예측하는 상황이 생길 수 있습니다.

### 시간 기반 검증

이 프로젝트의 기본 검증 구조는 다음과 같습니다.

| Fold | 학습 | 검증 |
| --- | --- | --- |
| 1 | 2019~2022 | 2023 |
| 2 | 2019~2023 | 2024 |
| 최종 | 2019~2024 | 2025 예측 |

학습 구간이 점점 늘어나는 방식을 `Expanding window validation`이라고 합니다.

함께 알아둘 키워드:

- Temporal split
- Time series cross-validation
- Walk-forward validation
- Expanding window
- Rolling window
- Backtesting

### 시계열과 순차 의존성

모든 시간 데이터가 ARIMA 같은 전통적 시계열 모델을 요구하는 것은 아닙니다. 이 프로젝트는 여러 투수의 투구가 섞인 투구 단위 패널 데이터에 가깝습니다.

따라서 다음 접근을 비교할 수 있습니다.

- 과거 통계를 특징으로 만든 표 형태 분류 모델
- 투수별 효과를 반영하는 계층 모델
- 투구 순서를 직접 처리하는 순차 모델

우선은 해석과 검증이 쉬운 표 형태 모델을 기준으로 삼습니다.

## 5. 데이터 누수

### 데이터 누수란

예측 시점에는 알 수 없는 정보가 학습 Feature에 들어가는 문제입니다. 검증 점수는 매우 좋아지지만 실제 2025 예측에서는 재현되지 않습니다.

### 이 프로젝트의 대표 누수 사례

- 현재 투구 Target을 포함한 누적 성공률
- 경기 종료 후 확정된 기록
- 2025년 이후 정보를 이용한 선수 평균
- 전체 데이터로 전처리한 뒤 학습·검증을 분리
- 검증 시즌 Target으로 만든 투수별 통계
- 현재 투구 이후의 구종 또는 Trackman 측정값

### `asof`의 의미

`As of time`은 특정 시점까지 알 수 있었던 정보만 사용한다는 뜻입니다. `asof_*` 컬럼도 이름만 믿지 말고 계산 기준 시점을 확인해야 합니다.

### `shift(1)`

현재 행을 제외하고 과거까지만 집계하기 위한 핵심 연산입니다.

```text
현재 투구의 과거 성공률
= 현재 투구 직전까지의 성공 수 / 현재 투구 직전까지의 투구 수
```

먼저 투수별로 날짜와 투구 순서를 정렬한 뒤 누적 통계를 계산하고 한 행 이동해야 합니다.

관련 키워드:

- Target leakage
- Look-ahead bias
- Future leakage
- Point-in-time correct feature
- Leakage-safe aggregation

## 6. 특징 공학

특징 공학은 원본 데이터를 모델이 패턴을 학습하기 좋은 변수로 바꾸는 과정입니다.

### 경기 상황 특징

| 원본 정보 | 파생변수 예 |
| --- | --- |
| 볼·스트라이크 | 볼카운트 조합, 불리한 카운트 여부 |
| 아웃 | 2아웃 여부 |
| 이닝 | 초반·중반·후반, 연장 여부 |
| 점수차 | 리드·동점·열세, 접전 여부 |
| 주자 | 주자 수, 득점권 주자 여부, 만루 여부 |

### 과거 흐름 특징

- 시즌 누적 성공률
- 최근 5·10·20개 투구 성공률
- 최근 1·3경기 성공률
- 연속 성공 또는 실패 길이
- 직전 등판 후 휴식일
- 구종별 누적 성공률과 사용 비율

서로 다른 기간을 함께 사용하면 장기 실력과 단기 컨디션을 나눠 볼 수 있습니다.

### Trackman 이력 특징

- 투수·구종별 평균 구속
- 최근 구속과 장기 평균의 차이
- 회전수 평균과 변동성
- 수평·수직 무브먼트 편차
- 릴리스 높이·좌우 위치의 편차
- 익스텐션 변화

현재 투구 이후에만 측정 가능한 값을 사용하지 않는 것이 우선입니다.

### 상호작용

한 변수의 효과가 다른 변수에 따라 달라지는 관계입니다.

예:

- 투수 × 구종
- 볼카운트 × 구종 사용 패턴
- LI × 점수차
- 주자 상황 × 릴리스 안정성
- 최근 컨디션 × 경기 후반

트리 기반 모델은 이런 비선형 상호작용을 비교적 자연스럽게 학습합니다.

## 7. 전처리

### 결측치

결측은 단순한 오류일 수도 있고, 해당 장비나 선수가 측정 대상이 아니었다는 정보일 수도 있습니다.

처리 후보:

- 중앙값 또는 평균 대체
- 투수·구종별 중앙값 대체
- `UNKNOWN` 범주 추가
- 결측 여부를 별도 Feature로 추가

대체 기준은 학습 데이터에서만 계산하고 검증·평가 데이터에 적용해야 합니다.

### 범주형 변수

구종이나 익명 ID처럼 문자열로 표현된 변수입니다.

주요 처리 방법:

- One-hot encoding
- Ordinal encoding
- Target encoding
- CatBoost의 범주형 처리

Target encoding은 Target을 사용하므로 반드시 Fold 내부에서 계산해야 합니다.

### 수치형 변수

구속, 회전수, LI 같은 연속형 변수입니다. Logistic Regression이나 거리 기반 모델은 표준화가 도움이 될 수 있지만 트리 모델은 보통 필수가 아닙니다.

### 익명 ID

익명 ID는 이름이 없다는 뜻이지 정보가 없다는 뜻은 아닙니다. 투수별 과거 수준을 표현할 수 있지만, 처음 등장하는 선수와 적은 표본 문제를 고려해야 합니다.

관련 키워드:

- Cold start
- High-cardinality categorical feature
- Smoothing
- Shrinkage

## 8. 기준 모델

복잡한 모델보다 먼저 단순 모델을 만들어야 개선 여부를 판단할 수 있습니다.

| 모델 | 역할 |
| --- | --- |
| 전체 평균 성공률 | 최소 기준선 |
| 투수별 과거 성공률 | 개인 차이를 반영한 기준 |
| Logistic Regression | 선형 관계를 학습하는 해석 가능한 기준 |
| Gradient Boosting | 비선형성과 변수 상호작용 학습 |

기준 모델보다 검증 점수가 좋아야 새로운 특징이나 복잡한 모델을 유지할 근거가 생깁니다.

## 9. 주요 모델

### Logistic Regression

입력 변수의 가중합을 성공 확률로 변환합니다. 빠르고 해석하기 쉬우며 좋은 기준 모델입니다.

### Random Forest

여러 결정 트리를 결합합니다. 비선형 관계를 처리하지만 확률 보정과 시간·메모리 비용을 확인해야 합니다.

### Gradient Boosting

이전 트리의 오류를 보완하는 트리를 순차적으로 추가합니다.

관련 라이브러리:

- HistGradientBoosting
- XGBoost
- LightGBM
- CatBoost

### 계층적 모델

리그 전체 효과와 투수 개인 효과를 함께 추정합니다. 표본이 적은 투수의 극단적 성공률을 전체 평균 쪽으로 완화할 수 있습니다.

관련 키워드:

- Hierarchical model
- Mixed-effects model
- Partial pooling
- Bayesian shrinkage

### 순차 모델

투구 순서 자체를 입력으로 처리합니다. 데이터 양과 검증 구조가 충분할 때 검토합니다.

- Markov model
- Hidden Markov Model
- RNN, LSTM, GRU
- Transformer

복잡한 순차 모델이 표 형태 부스팅 모델보다 항상 좋은 것은 아닙니다.

## 10. 평가 지표

PDF에는 공식 평가 지표가 공개되어 있지 않습니다. 공식 지표가 나오면 모델 선택 기준도 그 지표에 맞춰야 합니다.

### Log Loss

정답에 낮은 확률을 강하게 부여한 예측에 큰 페널티를 줍니다.

```text
Log Loss = -평균[y log(p) + (1-y) log(1-p)]
```

작을수록 좋습니다.

### Brier Score

예측 확률과 실제 Target의 평균 제곱 차이입니다. 작을수록 좋습니다.

### ROC-AUC

성공 투구가 실패 투구보다 높은 점수를 받는지를 평가합니다. 확률의 보정 상태를 직접 평가하는 지표는 아닙니다.

### 클래스 불균형

성공 또는 실패가 지나치게 많으면 Accuracy만으로 모델을 판단하기 어렵습니다.

확인할 항목:

- Target 비율
- 시즌별 Target 비율
- 투수별 표본 수
- PR-AUC
- 가중치 적용 전후 성능

## 11. 일반화와 분포 변화

### 일반화

학습에서 보지 않은 데이터에도 모델이 잘 작동하는 능력입니다. 대회의 Hidden Target 평가는 일반화 성능을 확인합니다.

### 과적합

학습 데이터를 지나치게 외워 검증·평가 성능이 떨어지는 현상입니다.

대응:

- 시간 기반 검증
- 모델 복잡도 제한
- 정규화
- Early stopping
- 불필요한 Feature 제거

### 분포 변화

2019~2024와 2025의 환경이 달라질 수 있습니다.

예:

- 새 투수와 타자
- 구종 분류 체계 변화
- Trackman 장비 또는 측정 방식 변화
- 리그 환경과 규칙 변화
- 선수의 노화, 부상, 성장

관련 키워드:

- Distribution shift
- Dataset shift
- Covariate shift
- Concept drift
- Population stability

## 12. 실험 설계

모든 실험은 같은 검증 Fold와 같은 지표에서 비교해야 합니다.

기록할 내용:

- 실험 ID와 날짜
- 데이터 버전
- Feature 목록
- 학습·검증 시즌
- 모델과 하이퍼파라미터
- Fold별 점수와 평균
- 실행 시간
- 결론과 다음 가설

여러 모델의 Out-of-fold 예측을 저장하면 모델 비교, 확률 보정 및 앙상블에 활용할 수 있습니다.

관련 키워드:

- Reproducibility
- Random seed
- Experiment tracking
- Out-of-fold prediction
- Ablation study
- Hyperparameter tuning
- Ensemble, blending, stacking

## 13. 파이프라인

현재 프로젝트가 지향하는 실행 순서는 다음과 같습니다.

```text
데이터 계약 로드
      ↓
학습·평가 데이터 검증
      ↓
시간 순서 정렬
      ↓
누수 없는 과거 특징 생성
      ↓
2023·2024 시간 기반 검증
      ↓
2019~2024 전체 학습
      ↓
2025 성공 확률 예측
      ↓
제출 파일 검증
```

공식 데이터가 오면 가장 먼저 해야 할 일은 임시 컬럼과 실제 컬럼을 매핑하고, 모든 변수의 생성 기준 시점을 확인하는 것입니다.

## 14. 공식 데이터 수령 후 체크리스트

- [ ] Train, Test, Sample Submission 파일 확인
- [ ] Target 이름, 값, 성공 클래스 확인
- [ ] 공식 평가 지표 확인
- [ ] ID 컬럼과 제출 행 순서 확인
- [ ] 날짜와 경기 내 투구 순서 확인
- [ ] 각 `asof_*` 변수의 기준 시점 확인
- [ ] Trackman 변수가 과거 누적인지 현재 투구 값인지 확인
- [ ] 시즌별 행 수와 Target 비율 확인
- [ ] 결측치와 이상값 확인
- [ ] 2025년에 처음 등장한 익명 ID 확인
- [ ] 중복 행과 중복 투구 ID 확인
- [ ] 임시 데이터 계약을 공식 스키마로 교체

## 15. 핵심 키워드 빠른 목록

| 분야 | 키워드 |
| --- | --- |
| 문제 유형 | Supervised learning, Binary classification, Probability prediction |
| 시간 | Temporal split, Walk-forward, Expanding window, Backtesting |
| 누수 | Target leakage, Look-ahead bias, Point-in-time correctness, `shift(1)` |
| 특징 | Feature engineering, Rolling statistics, Interaction, Recency |
| 전처리 | Imputation, Encoding, Scaling, Pipeline |
| 모델 | Logistic Regression, Gradient Boosting, Hierarchical model |
| 확률 | Calibration, Log Loss, Brier Score |
| 검증 | Generalization, Overfitting, OOF prediction, Ablation |
| 변화 | Distribution shift, Concept drift, Cold start |
| 운영 | Reproducibility, Data contract, Schema validation, Inference |

처음에는 모든 용어를 완벽히 이해하기보다, **예측 시점에 알 수 있는 정보만 사용한다**, **과거로 미래를 검증한다**, **확률의 품질을 평가한다**는 세 원칙을 기준으로 구현하면 됩니다.
