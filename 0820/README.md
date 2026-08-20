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
