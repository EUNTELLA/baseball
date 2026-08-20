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

### 결과

- 6시드 + 기존 shift: 2023 `+5.2875`, 2024 `-1.7550`
- 6시드 + Train 재계산 shift: 2023 `+5.3644`, 2024 `-1.7510`
- 선택 결과: 없음
- 판정: `keep_1029_champion`
- 결론: 보조 채널 시드 확대는 2023에서는 개선됐지만 2024로 전이되지 않았고, Train 기반 전역 shift 재계산도 하락을 복구하지 못해 제출 빌드 중단

## 03. 보조 채널 6시드 탐색 제출 빌드

정식 로컬 게이트는 통과하지 못했지만 남은 제출 기회로 전이 여부를 확인하기 위해 `aux6_train_recomputed_shift`를 별도 탐색 ZIP으로 만듭니다. 검증 JSON에서 선택 설정과 iteration을 읽고, production offset과 2025 shift에 필요한 2024 예측만 다시 생성합니다.

```bash
!python /content/baseball/0820/03_build_auxiliary_six_seed_probe_colab.py \
  --train /content/dataset/data/train.csv \
  --test /content/dataset/data/test.csv \
  --sample /content/dataset/data/sample_submission.csv \
  --validation-json /content/drive/MyDrive/0820_auxiliary_six_seed_calibration.json \
  --report /content/drive/MyDrive/0820_auxiliary_six_seed_probe_build.json \
  --task-type GPU
```

출력은 `/content/baseball/0820/results/submit_catboost_aux6_recomputed_shift_probe.zip`입니다. 현재 1029 ZIP은 보존하며 이 파일은 탐색 후보로만 취급합니다.

### 제출 결과

- Public Score: `1009.4332838027`
- 현재 최고 `1029.0832235020` 대비: `-19.6499396993`
- 판정: 폐기
- 해석: 보조 채널 6시드 확대와 함께 재계산된 `-0.0689738607` 전역 shift가 예측 수준을 과도하게 낮췄으며, 로컬 2024 하락 경고가 리더보드에서 더 크게 나타남
