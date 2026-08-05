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

실행 결과는 `0826/results/`에 저장됩니다. 모델 성능 비교 시 리더보드가 아닌 동일 Fold의 Brier Score와 Competition Score를 우선 확인합니다.

## 의존성

```text
pandas
numpy
scikit-learn
joblib
catboost==1.2.10
```
