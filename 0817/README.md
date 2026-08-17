# 0817 실험

`997.3951851847`의 공식 train 추세 기반 CatBoost 제출본을 새 기준으로 시작합니다. 0816의 기준 모델 코드와 피처 정의는 그대로 참조하되, 2026-08-17부터 시작한 실험 코드와 결과는 이 폴더에 분리합니다.

## 01. CatBoost 다중 시즌 소규모 파라미터 선별

`01_catboost_multiseason_tuning_colab.py`는 공식 train의 2022·2023·2024 시즌을 각각 검증으로 사용합니다. depth `5/6/7`, learning rate `0.03/0.05`, L2 `1/3/5/10`에서 한 변수 중심의 7개 설정을 먼저 1시드로 비교하고, 기존 기준과 최상위 도전자 하나를 3시드로 재확인합니다. 총 33회 학습입니다.

### Colab GPU 준비

런타임 유형에서 T4 GPU를 선택한 뒤 먼저 GPU와 CatBoost 버전을 확인합니다. 이 저장소의 Colab 검증에서 사용한 `catboost==1.2.8`로 맞춥니다.

```bash
!nvidia-smi
!pip uninstall -y catboost
!pip install -q catboost==1.2.8
```

설치 후에는 반드시 `런타임 → 세션 다시 시작`을 실행합니다. 다시 연결한 다음 `!nvidia-smi`가 GPU 정보를 출력하는지 확인하고 아래 실험을 시작합니다. `CUDA error 35`는 학습 코드가 아니라 현재 런타임의 CUDA 드라이버와 CatBoost 런타임이 맞지 않을 때 발생합니다.

```bash
%cd /content/baseball
!git pull

!python /content/baseball/0817/01_catboost_multiseason_tuning_colab.py \
  --train /content/dataset/data/train.csv \
  --output /content/drive/MyDrive/0817_catboost_multiseason_tuning.json \
  --task-type GPU
```

같은 평균 점수의 평균이 기준보다 `+3` 초과, 최악 시즌이 `-2` 이상, 원본 평균이 `-10` 이상일 때만 7시드 제출본 제작으로 진행합니다. 이 단계에서는 ZIP을 만들지 않습니다.
