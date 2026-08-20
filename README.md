# 야구 투구 제구 성공 확률 예측

투구 직전까지 확인 가능한 경기 상황과 선수 정보, 과거 투구 이력을 활용해 각 투구의 `control_success` 확률을 예측합니다.

> 본 대회는 모델 파일과 추론 코드를 `submit.zip`으로 제출하는 코드 제출 대회입니다.

## 대회 개요

최근 스포츠 현장에서는 선수의 경기력 분석과 전략 수립을 위한 데이터 기반 의사결정의 중요성이 빠르게 커지고 있습니다. 특히 야구에서 투구의 제구력은 실점 억제, 볼카운트 운영, 타자 대응 전략에 직접적인 영향을 주는 핵심 요소입니다.

기존에는 평균자책점, 볼넷 수, 스트라이크 비율과 같은 경기 후 집계 지표로 제구력을 평가하는 경우가 많았습니다. 하지만 실제 경기에서는 매 투구 직전의 볼카운트, 주자 상황, 타자·투수 특성 및 과거 투구 이력 등 다양한 정보가 복합적으로 작용합니다.

이번 해커톤은 단순한 결과 통계가 아닌 **투구가 이루어지기 전까지 확인 가능한 정보만으로 해당 투구의 제구 성공 가능성을 예측**하는 실전형 AI 모델 개발을 목표로 합니다. 2019~2024년의 과거 투구 특성을 담은 Trackman 데이터가 보조 데이터로 제공됩니다.

## 문제 정의

- **주제:** 투구 단위의 제구 성공 확률 예측 AI 모델 개발
- **예측 대상:** `control_success`
- **학습 Target:** 제구 성공 `1`, 제구 실패 `0`
- **제출 예측값:** 각 투구의 제구 성공 확률(0 이상 1 이하)
- **핵심 제약:** 현재 투구 이전 시점에 확인 가능한 정보만 사용

다음 세 가지 경우는 제구 실패이며, 그 외 유효한 투구는 제구 성공으로 정의됩니다.

1. 스트라이크존 가운데 부근으로 들어간 공
2. 스트라이크존에서 크게 벗어난 공
3. 포수의 요구 방향과 반대로 들어간 공

Phase 2 결과와 코드 검증을 바탕으로 약 100명의 Phase 3 진출자를 선발합니다. Phase 3은 1박 2일 오프라인 해커톤으로 진행되며, 세부 과제는 추후 안내됩니다.

## 저장소 구성

```text
.
├── baseline/
│   ├── [Baseline_Train]_RandomForest...ipynb
│   └── [Baseline_Inference]_RandomForest...ipynb
├── open/
│   ├── data/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── sample_submission.csv
│   │   └── trackman_history.csv
│   ├── data_description.md
│   └── submit.zip
├── open.zip
└── README.md
```

- `train.csv`: 학습 입력 및 정답 데이터, 1,475,092행 × 49열
- `test.csv`: 컬럼 형식 확인용 샘플 5행 × 48열. 평가 시 실제 비공개 데이터로 교체
- `sample_submission.csv`: 제출 형식 확인용 샘플 5행 × 2열
- `trackman_history.csv`: 2019~2024년 Trackman 과거 로그, 1,793,078행 × 30열
- `open/submit.zip`: 운영진 베이스라인 기반 제출 파일 예시

전체 컬럼 정의는 [데이터 설명서](open/data_description.md), 현재 코드에서 사용하는 구조와 파생 정보는 [데이터 구조 문서](doc/DATA_STRUCTURE.md)를 참고하세요.

모델별 실험 내용과 리더보드 제출 결과는 [실험 및 제출 기록](EXPERIMENTS.md)에 누적합니다.

## 데이터 사용 원칙

평가 데이터의 각 행은 독립적으로 예측해야 합니다. 운영진이 제공한 `asof_*` 컬럼은 투구 직전까지의 과거 기록으로 계산되었으므로 사용할 수 있습니다.

다음 정보 또는 피처는 사용할 수 없습니다.

- 현재 투구 이후에 확정되는 정보
- 현재 투구의 실제 위치, 코스, 판정, 결과, 구종 및 Trackman 측정값
- 2025년 Trackman 데이터
- 평가 데이터의 다른 행으로 만든 선수·팀·월별 통계
- 평가 데이터 내부 빈도, 분포, target encoding, rolling 및 expanding 피처
- 평가 데이터 전체를 확인한 뒤 만든 사후 보정값

## 평가

<details>
<summary><strong>평가 지표 및 점수 산식 보기</strong></summary>

본 대회는 실제 정답에 가까운 확률을 예측할수록 높은 점수를 받는 **Brier Skill Score** 기반 평가를 사용합니다.

```text
Brier Score = mean((p_i - y_i)^2)
r = mean(y_i)
평균 제구율 Brier Score = r × (1 - r)

Score = max(0, 100000 × (1 - Brier Score / 평균 제구율 Brier Score))
```

- `p_i`: i번째 샘플의 제구 성공 예측 확률
- `y_i`: i번째 샘플의 실제 정답(0 또는 1)
- `r`: 전체 평가 데이터의 평균 제구 성공률(비공개)
- Public Score: 전체 테스트 데이터 100%
- Private Score: 대회 종료 시점의 Public Score

</details>

<details>
<summary><strong>평가 절차 및 Phase 3 선발 기준 보기</strong></summary>

- LG Aimers 수료 조건: Phase 1 이수 및 Phase 2 Public Score 549.51 이상
- 기준 점수는 운영진 베이스라인 추론 코드를 운영진 평가 환경에서 실행한 결과로 산정
- 1차 평가: 리더보드 Private Score 100%
- 동점자는 대회 리더보드 순위 산정 방식 적용
- 2차 평가: Phase 3 진출 희망 팀의 제출 코드 검증
- Private 리더보드 상위 참가자 약 100명은 코드 및 PPT 필수 제출
- 코드·PPT 제출과 검증을 모두 통과한 상위 참가자가 Phase 3 진출

</details>

## 코드 제출

<details open>
<summary><strong>submit.zip 구성 및 실행 규칙 보기</strong></summary>

압축 파일의 최상위 구조는 아래와 정확히 일치해야 합니다. 별도의 상위 폴더를 추가하면 설치 오류가 발생할 수 있습니다.

```text
submit.zip
├── model/              # 학습된 모델 가중치
├── script.py           # 평가 서버에서 자동 실행되는 추론 코드
└── requirements.txt    # 추론에 필요한 패키지와 버전
```

평가 서버는 압축을 해제한 뒤 아래 항목을 자동으로 추가합니다.

```text
submit.zip
├── model/
├── script.py
├── requirements.txt
├── data/               # 읽기 전용 실제 평가 데이터
└── output/             # 예측 결과 저장 경로
    └── submission.csv
```

`script.py`는 반드시 `output/submission.csv`를 생성해야 합니다. 결과 파일은 평가 서버의 `test.csv`와 동일한 `row_id`를 포함하고, `control_success` 열에 0 이상 1 이하의 확률을 기록해야 합니다.

</details>

<details>
<summary><strong>제출 및 실행 제한 보기</strong></summary>

| 항목 | 제한 |
| --- | --- |
| 제출 ZIP 파일 | 최대 10GB |
| 압축 해제 후 용량 | 최대 32GB |
| 패키지 설치 시간 | 최대 10분 |
| 전체 추론 시간 | 최대 10분, 245,789개 샘플 |
| 운영체제 | Ubuntu 22.04.5 LTS |
| Python | 3.11.15 |
| CPU | 6 vCPU |
| RAM | 28GB |
| GPU | NVIDIA L4, VRAM 22.4GiB |
| CUDA | 12.8 |
| 인터넷 | 패키지 설치 외 비활성화 |

</details>

## 참고 링크

- [대회 평가 안내](https://dacon.io/competitions/official/236743/overview/evaluation)
- [코드 제출 가이드](https://cfiles.dacon.co.kr/competitions/236564/guide.html)

제출 전 반드시 최신 평가 탭과 코드 제출 가이드를 다시 확인하세요.
