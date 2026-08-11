# 0826 실험

베이스라인 점수 개선을 위한 네 가지 실험입니다. 모든 모델 비교에는 2022~2024년 expanding-window 시간 검증을 사용합니다.

## Google Colab 준비

Google Drive에 저장소를 올린 뒤 Drive를 마운트하고, 저장소 루트로 이동합니다. 아래 구조에서 `0826/common.py`와 `open/data/train.csv`가 모두 있어야 합니다.

```text
baseball/
├── 0826/
│   ├── common.py
│   └── *.ipynb
└── open/data/train.csv
```

```python
from google.colab import drive
drive.mount("/content/drive")

# 실제 Drive 경로에 맞게 수정
%cd /content/drive/MyDrive/baseball
```

2번과 4번 노트북은 첫 실행 시 `catboost==1.2.10`을 설치합니다.

## 실행 순서

아래 노트북을 번호 순서대로 열어 전체 셀을 실행합니다.

1. `01_time_cv_random_forest.ipynb`: 운영진 RandomForest의 다중 시간 Fold 기준점
2. `02_catboost_time_cv.ipynb`: 선수·팀 ID를 범주형으로 처리하는 CatBoost
3. `03_probability_calibration.ipynb`: 이전 연도 OOF만 사용하는 walk-forward 확률 보정
4. `04_feature_engineering_catboost.ipynb`: 카운트, 좌우 조합, 득점권, 경기 국면, 최근 추세 등의 파생 피처
5. `05_final_train_submission.ipynb`: 전체 데이터 RandomForest 재학습, 확률 보정, 제출 ZIP 생성 및 샘플 검증

실행 결과는 `0826/results/`에 저장됩니다. 모델 성능 비교 시 리더보드가 아닌 동일 Fold의 Brier Score와 Competition Score를 우선 확인합니다.

05는 앞선 검증에서 선택한 RandomForest를 2019~2024년 전체 데이터로 재학습합니다. 비교를 위해 원본 제출본과 `-0.01056616` 확률 보정 제출본을 모두 생성합니다.

## 제출 결과

2026-08-11에 동일한 RandomForest 모델의 원본 확률과 보정 확률을 각각 제출했습니다.

| 제출본 | 확률 보정 | Public Score | 추론 시간 | 용도 |
| --- | ---: | ---: | ---: | --- |
| `submit_rf_raw.zip` | 없음 | 549.6413938266 | 2초 | 비교 기준 및 백업 |
| `submit_rf_calibrated.zip` | `-0.01056616` | **675.5498778599** | 2초 | **현재 최고 점수 및 최종 후보** |

보정 제출본은 raw 제출본보다 `+125.9084840333`, 기존 베이스라인 `549.5119345223`보다 `+126.0379433376` 높은 Public Score를 기록했습니다. 모델과 피처는 동일하고 확률 보정만 다르므로, 이번 비교에서 확인된 차이는 상수 오프셋 보정의 효과입니다. 다만 Public 데이터 과적합 가능성이 있으므로 최종 판단 시 시간 기반 로컬 검증과 Private Score를 함께 확인합니다.

대회에는 ZIP을 풀거나 이름을 내부적으로 변경하지 않고 `submit_rf_calibrated.zip`을 업로드합니다. 플랫폼 화면에서는 업로드 파일명이 `submit.zip`으로 표시될 수 있습니다.

## 의존성

```text
pandas
numpy
scikit-learn
joblib
catboost==1.2.10
```
