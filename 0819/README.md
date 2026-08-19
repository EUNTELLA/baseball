# 0819 실험

0819 시작 기준은 공식 train 추세를 사용한 CatBoost의 Public Score `997.3951851847`입니다. 모델을 분리하지 않고 기존 모델의 과거 예측 오차를 보정한 결과, 새로운 Public Score `1029.0832235020`을 기록했습니다.

## 01. 투수별 과거 예측 오차 보정 3종 선별

같은 손 조합, 2스트라이크, 주자 존재 여부에 따라 투수별 예측 오차가 달라지는지를 계산합니다. 표본이 적은 투수의 값은 0에 가깝게 줄이고, 각 행의 조건에 맞는 값만 기존 확률에 반영합니다.

- 같은 손 조합 보정: 완화 상수 `1000`
- 2스트라이크 보정: 완화 상수 `1000`
- 주자 존재 보정: 완화 상수 `2000`
- 검증 시즌 S의 표 원천: S-2와 S-1을 학습하지 않고 얻은 예측 오차
- 평가 지표: 공식 Brier Skill Score
- 통과 조건: 2023·2024 BSS 증분이 모두 양수이고 2024가 `+5` 이상

이번 코드는 구조 선별 단계이므로 성공모델만 사용합니다. 통과할 때만 MR/wayoff offset과 공식 train 추세 shift를 포함한 전체 `997` 파이프라인 검증 및 제출본 제작으로 진행합니다.

```bash
!python /content/baseball/0819/01_catboost_residual_differential_screen_colab.py \
  --train /content/dataset/data/train.csv \
  --output /content/drive/MyDrive/0819_catboost_residual_differential.json \
  --task-type GPU
```

### 결과

- 2023 BSS 증분: `+31.7693760408`
- 2024 BSS 증분: `+36.2943178506`
- 판정: `continue_full_997_pipeline`

두 시즌에서 공식 점수가 함께 개선되어 전체 파이프라인 검증으로 진행합니다.

## 02. 전체 997 파이프라인 검증

01에서 만든 행 단위 보정을 성공확률에 더한 뒤, 기존과 같은 MR/wayoff offset과 공식 train 시즌 추세 shift를 기준·후보에 공통 적용합니다. 2023은 개발 확인, 2024는 최종 확인 역할이며 해당 시즌을 학습하지 않은 예측 오차만 사용합니다.

```bash
!python /content/baseball/0819/02_residual_differential_full_pipeline_colab.py \
  --train /content/dataset/data/train.csv \
  --output /content/drive/MyDrive/0819_residual_differential_full_pipeline.json \
  --task-type GPU
```

2023·2024 BSS가 모두 양수이고, 2024 BSS가 `+5` 이상이며 평균 오차가 `0.001`보다 크게 악화되지 않을 때만 제출 ZIP을 생성합니다.

### 결과

- 2023 전체 파이프라인 BSS 증분: `+8.6274`
- 2024 전체 파이프라인 BSS 증분: `+57.3687`
- 2024 절대 평균 오차 변화: `-0.0001573` (개선)
- 공식 지표 판정: 제출 후보 제작
- 해석: 두 시즌 모두 공식 점수가 개선됐으므로 기존 997을 보존하면서 별도 후보 제작

## 03. 7시드 예측 오차 보정 제출 후보 제작

기존 Public `997.3951851847` ZIP은 수정하지 않습니다. 2023·2024를 각각 직전 시즌까지만 학습한 7시드 모델로 예측하고, 그 오차로 만든 세 보정표와 행 단위 추론 코드만 기존 ZIP 사본에 추가합니다.

```bash
!python /content/baseball/0819/03_build_residual_differential_submission_colab.py \
  --train /content/dataset/data/train.csv \
  --test /content/dataset/data/test.csv \
  --sample /content/dataset/data/sample_submission.csv \
  --report /content/drive/MyDrive/0819_residual_differential_build.json
```

생성 파일은 `/content/baseball/0819/results/submit_catboost_residual_differential.zip`입니다.

### 결과

- 예측 오차표: 같은 손 499명, 2스트라이크 508명, 주자 존재 499명
- 중앙 보정 차이 절댓값: `0.0042121`, `0.0027521`, `0.0016151`
- ZIP 무결성 오류: 없음
- 샘플 추론: 5행, 결측 0, 범위 `0.3999831~0.4954599`
- 샘플 예측 평균: `0.4529250`
- 제출일: `2026-08-19 21:08:52`
- Public Score: `1029.0832235020`
- 기존 `997.3951851847` 대비: `+31.6880383173`
- 판정: 새 규정 준수 최고 모델로 채택

## 04. 보정 구성과 적용 강도 비교

현재 최고 `1029.0832235020`을 만든 세 보정의 기여를 분리합니다. 같은 손 조합, 2스트라이크, 주자 존재의 7개 단독·조합과 적용 강도 `0.5, 0.75, 1.0, 1.25`를 비교합니다.

비교 비용을 줄이기 위해 선별 단계는 기준 CatBoost 성공모델 3시드의 과거 시즌 예측을 사용합니다. 여기서 선택된 한 구성만 기존 offset·shift를 포함한 전체 과정으로 다시 확인합니다.

두 확인 시즌에서 모두 공식 점수가 개선되는 후보만 남기고, 그중 낮은 쪽의 개선값이 가장 큰 구성을 선택합니다. 이 단계에서는 새 제출 ZIP을 만들지 않습니다.

```bash
!python /content/baseball/0819/04_error_adjustment_weight_screen_colab.py \
  --train /content/dataset/data/train.csv \
  --output /content/drive/MyDrive/0819_error_adjustment_weight_screen.json \
  --task-type GPU
```

### 결과

- 선택 구성: 같은 손 조합 + 2스트라이크
- 적용 강도: `1.25`
- 2023 BSS 증분: `+32.0812`
- 2024 BSS 증분: `+33.1534`
- 판정: 전체 offset·shift 과정에서 재검증

## 05. 선택 구성 전체 과정 검증

04에서 선택한 두 보정을 `1.25배` 적용하고 기존 MR/wayoff offset과 공식 train 추세 shift를 포함해 다시 확인합니다.

```bash
!python /content/baseball/0819/02_residual_differential_full_pipeline_colab.py \
  --train /content/dataset/data/train.csv \
  --output /content/drive/MyDrive/0819_hand_two_strikes_weight125_full.json \
  --task-type GPU \
  --axes hand,two_strikes \
  --weight 1.25
```

이 결과가 현재 3종 보정의 전체 과정 결과보다 안정적일 때만 새 제출본을 만듭니다.

## 후속 실험 범위

모델 계열 교체, 데이터 분할형 전문가, 단순 용량 확대, 광범위한 사후 변환은 재검토 보류로 분류했습니다. 이후에는 현재 최고 모델 위의 행 단위 오차 보정과 F 행 조건부 후처리를 중심으로 진행합니다. 상세 기준은 `doc/EXPERIMENT_FILTER.md`에 기록했습니다.
