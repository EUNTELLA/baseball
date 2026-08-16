# CatBoost 투수 역할 피처 선별 검증

현재 최고 모델의 CatBoost d6·FE10 성공모델 구조에 다음 두 피처만 추가합니다.

- `pitcher_role_score`: 학습 구간의 투수별 1~3회 투구 비중을 스무딩한 선발 성향
- `inning_over_role`: 현재 이닝과 역할상 예상 등판 이닝의 차이

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
!nvidia-smi
!pip -q install catboost==1.2.8
!rm -rf /content/baseball /content/dataset
!git clone https://github.com/EUNTELLA/baseball.git /content/baseball
!unzip -q -o /content/drive/MyDrive/open.zip -d /content/dataset
```

```python
!python /content/baseball/0816/08_catboost_pitcher_role_screen_colab.py \
  --train /content/dataset/data/train.csv \
  --task-type GPU \
  --output /content/drive/MyDrive/0816_catboost_pitcher_role_result.json
```

기준 모델과 역할 피처 모델을 각각 7시드로 학습합니다. 원본 Brier Score가 개선되고, 두 모델의 예측 평균을 2024 실제 평균으로 동일하게 맞춘 진단에서도 `+5`를 넘을 때만 전체 제출 빌드로 진행합니다. 실제 평균 맞춤은 구조적 분해능 진단에만 사용하며 제출 코드에는 포함하지 않습니다.
