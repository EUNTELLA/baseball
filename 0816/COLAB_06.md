# submit012 League-rate baseline Colab 검증

기준 모델은 Public Score `998.0030076995`의 `012_shift_full`입니다. 이 실험은 기존 모델을 바로 대체하지 않고 3시드 2024 시간 검증을 먼저 수행합니다.

## 1. Google Drive 준비

`train.csv`를 Google Drive의 `LG_Aimers/train.csv`에 저장합니다.

## 2. Colab 실행

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
!pip -q install catboost==1.2.8
!git clone https://github.com/EUNTELLA/baseball.git /content/baseball
```

```python
!python /content/baseball/0816/06_submit012_league_baseline_colab.py \
  --train /content/drive/MyDrive/LG_Aimers/train.csv \
  --output /content/drive/MyDrive/LG_Aimers/0816_submit012_league_baseline_result.json
```

피처 정의와 submit012 비교용 OOF 파일이 `0816`에 포함되어 있어 다른 저장소를 클론하거나 파일을 따로 업로드할 필요가 없습니다.

## 판정 기준

- `score_delta > 3.0`: 7시드 전체 학습 및 새 제출본 제작 진행
- `score_delta <= 3.0`: 후보 폐기, `submit012.zip` 유지

2024 실제 평균은 결과 보고에만 사용합니다. 모델 baseline은 2021~2023 시즌 평균으로 2024를 외삽하여 계산합니다. 테스트 데이터 전체 평균이나 행간 통계는 사용하지 않습니다.
