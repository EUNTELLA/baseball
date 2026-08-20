# 0820 실험

## 01. F 행 전용 잔차 제어기

현재 Public 최고 `1029.0832235020` 구조를 유지하고 F 행에서 발생한 성공모델의 순방향 예측 오차만 별도 CatBoost 회귀기로 학습합니다. R/F 직접 분류 모델을 대체하는 실험이 아니라 현재 예측 위에 작은 잔차만 더하는 실험입니다.

- 잔차 원천: 해당 시즌을 학습하지 않은 성공 확률 예측
- 제어기: depth 4, 250 iterations, lr 0.025, L2 100
- 적용 범위: `game_type=F` 행만
- 적용 강도: `0.05, 0.10, 0.15, 0.20, 0.30`
- 검증: 2021·2022 잔차 → 2023, 2022·2023 잔차 → 2024
- 기준선: 과거 오차 보정 3종 + 전역 MR/큰 이탈 offset + Train 추세 shift

```bash
!python /content/baseball/0820/01_f_residual_controller_screen_colab.py \
  --train /content/dataset/data/train.csv \
  --output /content/drive/MyDrive/0820_f_residual_controller_screen.json \
  --task-type GPU
```

같은 강도가 2023·2024에서 모두 개선되고 2024 BSS가 `+3` 이상일 때만 전체 학습 및 제출 ZIP 제작으로 진행합니다.

### 결과

- 선택된 적용 강도: 없음
- 판정: `keep_1029_champion`
- 결론: F 행 순방향 잔차를 별도 회귀기로 학습해도 시험한 동일 강도로 2023·2024를 함께 개선하지 못해 전체 학습 및 ZIP 제작 중단

## 02. 보조 채널 6시드 평균과 전역 shift 재계산

현재 MR·큰 이탈 보조 채널만 `3→6`시드로 늘립니다. 추가 시드는 `99, 1, 123`입니다. 6시드 결과는 현재 3시드 shift를 공유한 경우와 6시드 Train 예측으로 shift를 다시 계산한 경우를 분리해 비교합니다.

```bash
!python /content/baseball/0820/02_auxiliary_six_seed_calibration_screen_colab.py \
  --train /content/dataset/data/train.csv \
  --output /content/drive/MyDrive/0820_auxiliary_six_seed_calibration.json \
  --task-type GPU
```

동일 구성이 2023·2024에서 모두 개선되고 2024 BSS가 `+3` 이상일 때만 제출 빌드로 진행합니다.
