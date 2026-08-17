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

결과: `d6_lr03_l2_3`이 3시드 확인에서 같은 평균 평균 `+4.4015`, 최악 시즌 `+2.4662`, 원본 평균 `+4.9291`로 세 기준을 모두 통과했습니다.

## 02. 통과 파라미터 7시드 train-only 제출본

기존 TEST 예측 평균은 사용하지 않습니다. 2019~2023 학습 → 2024 OOF 예측으로 MR/wayoff offset을 다시 적합하고, 공식 train 시즌 추세 목표 `0.4777935316`에 맞는 고정 shift를 학습 단계에서 저장합니다.

```bash
!python /content/baseball/0817/02_catboost_tuned_build_colab.py \
  --train /content/dataset/data/train.csv \
  --test /content/dataset/data/test.csv \
  --sample /content/dataset/data/sample_submission.csv \
  --output-dir /content/drive/MyDrive/0817 \
  --task-type GPU
```

출력 ZIP은 `/content/drive/MyDrive/0817/submit_catboost_lr03_l2_3_train_only.zip`입니다.

빌드 결과 7시드 2024 성공모델 검증 점수는 `774.7030`, offset 적용 후 복원 라벨 253,116행 점수는 `774.3808 → 800.1367`입니다. 재적합 계수는 `b=-0.1106166`, `c=0.0105778`이며 train-only 기준 평균 `0.4950931`을 목표 `0.4777935`로 이동하는 고정 shift는 `-0.0698581`입니다. ZIP 무결성 검사와 5행 샘플 추론을 통과했습니다.

2026-08-17 제출 결과 Public Score는 `959.2282302569`였습니다. 공식 train 추세 기준 모델 `997.3951851847`보다 `38.1669549278` 낮아 최종 후보에서 제외합니다. 파라미터, offset 재적합, shift가 동시에 바뀌었고 특히 shift가 `-0.0384267`에서 `-0.0698581`로 크게 이동했으므로 다음 실험은 재학습 없이 구성 요소를 하나씩 분리해 비교합니다.

## 03. 평가 기준 기반 전체 파이프라인 walk-forward

검증 시즌 정답을 보정값 결정에 사용하지 않는 중첩 검증입니다. 2022·2023·2024 각각에 대해 직전 시즌에서 조기 종료 반복 수, MR/wayoff offset, 다음 시즌 성공률과 shift를 확정한 뒤 검증 시즌의 Brier Skill Score를 계산합니다. 기준과 후보의 차이는 성공모델 파라미터뿐입니다.

```bash
!python /content/baseball/0817/03_catboost_full_pipeline_walkforward_colab.py \
  --train /content/dataset/data/train.csv \
  --output /content/drive/MyDrive/0817_full_pipeline_walkforward.json \
  --task-type GPU
```

최종 점수 평균 `+5`, 3개 중 2개 시즌 개선, 최악 시즌 `-3` 이상, 2024 개선, 평균 보정 오차 비악화를 모두 만족할 때만 단일 변수 제출본을 만듭니다. 중간 결과는 fold가 끝날 때마다 JSON에 저장합니다.

결과는 평균 최종 점수 `+10.0668`, 2/3 시즌 개선, 최악 `-0.2776`이었지만 2024가 `-0.2776` 하락했고 평균 절대 보정 오차도 `+0.0002831` 악화됐습니다. 사전 채택 기준을 통과하지 못했으므로 새 파라미터 제출본은 만들지 않고 Public Score `997.3951851847` 모델을 유지합니다.

## 04. Brier 직접 최적화 CatBoost 회귀

기존 CatBoostClassifier의 Logloss 확률과 CatBoostRegressor의 RMSE 예측을 비교합니다. RMSE는 대회 핵심인 Brier의 제곱오차를 직접 최소화합니다. 2022·2023에서 회귀 혼합 비율을 선택하고 2024는 확인 전용으로 남겨 선택 과적합을 줄입니다.

```bash
!python /content/baseball/0817/04_catboost_brier_regression_screen_colab.py \
  --train /content/dataset/data/train.csv \
  --output /content/drive/MyDrive/0817_catboost_brier_regression.json \
  --task-type GPU
```

개발 시즌 raw 평균 `+3`, 최악 `-2`, 같은 평균 개선과 2024 raw `+3`·같은 평균 개선을 모두 만족할 때만 전체 파이프라인 검증으로 진행합니다.

결과는 개발 시즌에서 선택된 회귀 비중이 `0.0`이었습니다. 즉 RMSE 회귀를 섞지 않은 기존 Logloss 분류 모델이 가장 좋았으며, 2024 확인도 동일 모델 비교가 되어 차이 `0`으로 끝났습니다. 판정은 `reject_brier_regression_axis`이며 이 모델 축은 종료합니다.

## 05. 성공·MR·wayoff 확률 단순합 제약

서로소 라벨인 성공, MR, wayoff를 각각 3시드 CatBoost로 예측한 뒤 `1-p_mr-p_wayoff`, `p_success/(p_success+p_mr+p_wayoff)` 및 기존 성공확률과의 고정 혼합을 비교합니다. 추가 보정 계수는 적합하지 않습니다. 2022·2023에서 결합 방식을 선택하고 2024는 확인에만 사용합니다.

```bash
!python /content/baseball/0817/05_catboost_failure_simplex_screen_colab.py \
  --train /content/dataset/data/train.csv \
  --output /content/drive/MyDrive/0817_catboost_failure_simplex.json \
  --task-type GPU
```

개발 시즌 raw 평균 `+5`, 최악 `-2`, 같은 평균 개선과 2024 raw `+5`·같은 평균 개선을 모두 만족할 때만 전체 파이프라인 검증으로 진행합니다.

선별 결과 `blend_complement_10`이 개발 raw 평균 `+5.6301`, 최악 `+3.9737`, 같은 평균 평균 `+5.2463`으로 선택됐습니다. 2024에서도 raw `+5.3559`, 같은 평균 `+6.8821`로 확인되어 전체 파이프라인 검증으로 진행합니다.

## 06. 실패 여집합 10% 전체 파이프라인 검증

기존 성공확률 90%와 실패 여집합 10%를 혼합한 뒤 기존 offset을 대체하는 방식과, 혼합 후 offset을 다시 적합하는 방식을 비교합니다. 직전 시즌에서 반복 수·offset·shift를 확정하는 중첩 walk-forward이며 기존 offset 파이프라인이 기준입니다.

```bash
!python /content/baseball/0817/06_failure_simplex_full_pipeline_colab.py \
  --train /content/dataset/data/train.csv \
  --output /content/drive/MyDrive/0817_failure_simplex_full_pipeline.json \
  --task-type GPU
```

개발 평균 `+5`, 최악 `-2`, 평균 보정 오차 비악화와 2024 `+5`·보정 오차 비악화를 모두 만족할 때만 제출 ZIP 하나를 만듭니다.

결과는 `simplex_replace_offset`이 개발 시즌 평균 `+207.8028`, 최악 `+163.2570`, 평균 절대 보정 오차 `-0.0025843`으로 크게 개선됐습니다. 그러나 2024에서 shift 전에는 약 `+6.17`이었지만 최종 shift 적용 후 `-2.1561`로 뒤집혔고 평균 절대 오차도 `+0.0002981` 악화됐습니다. 사전 기준에 따라 `keep_997_baseline`으로 판정하며 이 구성의 ZIP은 만들지 않습니다.

## 07. 실패 여집합 공통 shift 민감도

후보별 shift 재계산을 제거하고 기준과 후보에 동일한 고정 shift를 적용합니다. 기준은 직전 시즌에서 적합한 기존 offset, 후보는 offset을 제거한 실패 여집합 10% 혼합입니다. 현재 `997` 모델의 배포 shift `-0.0384267193`과 주변 `±0.01`을 핵심 판정에 사용합니다.

```bash
!python /content/baseball/0817/07_failure_simplex_shared_shift_validation_colab.py \
  --train /content/dataset/data/train.csv \
  --output /content/drive/MyDrive/0817_failure_simplex_shared_shift.json \
  --task-type GPU
```

배포 shift에서 개발 평균 `+5`, 최악 `-2`, 2024 `+5`, 2024 주변 shift 두 값 모두 양수이고 평균 오차 악화가 `0.001` 이하일 때만 기존 `997` ZIP을 단일 변수로 패치합니다.

결과는 배포 shift에서 개발 평균 `+180.2663`, 최악 `+171.1284`로 강했지만 2024 개선은 `+2.4134`로 채택 기준 `+5`에 미달했습니다. 2024 주변 shift에서도 `+1.6925~+3.1345`로 방향은 일관되게 양수였고 평균 오차는 `-0.0003609` 개선됐습니다. 작은 개선 신호는 확인했지만 제출 기회 대비 근거가 부족해 `keep_997_baseline`으로 판정합니다.
