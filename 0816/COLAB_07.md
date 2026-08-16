# submit012 League-rate 7시드 제출 후보 생성

06 선별 검증 결과 `+40.1010`을 확인한 뒤 실행하는 전체 빌드입니다.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
!nvidia-smi
!pip -q install catboost==1.2.8
!rm -rf /content/baseball /content/dataset /content/league_build
!git clone https://github.com/EUNTELLA/baseball.git /content/baseball
!unzip -q -o /content/drive/MyDrive/open.zip -d /content/dataset
```

```python
!python /content/baseball/0816/07_submit012_league_baseline_build_colab.py \
  --train /content/dataset/data/train.csv \
  --task-type GPU \
  --work-dir /content/league_build \
  --output-zip /content/drive/MyDrive/submit013_league_baseline.zip
```

완료 산출물:

- `/content/drive/MyDrive/submit013_league_baseline.zip`
- `/content/drive/MyDrive/submit013_league_baseline.json`

기존 submit012의 보조모델과 고정 offset은 유지합니다. 성공모델만 league-rate baseline 7시드로 교체하고, 기존 전역 logit shift는 제거합니다. 2025 baseline은 학습 결과로 확정한 고정값이며 test 평균을 사용하지 않습니다.
