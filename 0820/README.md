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

## 04. 동적 투수 기준확률 잔차 모델

공개 저장소 검토에서 반복적으로 성과가 있었던 `동적 기준확률 + 잔차 학습 + 3개 연도 순방향 검증` 원리만 독립 구현합니다. 외부 코드·모델·Trackman 매핑·고정 계수는 사용하지 않습니다.

공식 `asof_pitcher_n`, 커리어 성공률, 최근 1·3·5경기 성공률로 투수 기준확률을 만들고 CatBoostRegressor가 `정답-기준확률`을 학습합니다. 현재 직접 CatBoost 분류기와 2022·2023·2024에서 비교합니다.

```bash
!python /content/baseball/0820/04_dynamic_pitcher_baseline_residual_screen_colab.py \
  --train /content/dataset/data/train.csv \
  --output /content/drive/MyDrive/0820_dynamic_pitcher_baseline_residual.json \
  --task-type GPU
```

세 연도 모두 개선되고 2024 증분이 `+5` 이상일 때만 다중 잔차 채널과 행별 결합 단계로 진행합니다.

### 결과

- 2022 BSS 변화: `-44.5786`
- 2023 BSS 변화: `-1436.8538`
- 2024 BSS 변화: `-51.9202`
- pooled BSS 변화: `-504.9753`
- 오차 상관: 2022 `0.999456`, 2023 `0.990779`, 2024 `0.999274`
- 판정: `reject_dynamic_baseline_residual`
- 결론: 단일 동적 기준확률과 고정 잔차 회귀기는 직접 분류기보다 모든 연도에서 나쁘고 2023 체제 변화에서 크게 붕괴해 다중 채널 확장 중단

## 05. 다중 잔차 채널 구조 선별

외부 구조처럼 하나의 모델을 시드만 늘리지 않고, 서로 다른 입력 범위와 시간 감쇠를 가진 잔차 채널 3개를 먼저 만듭니다. 이 단계는 채널 구조 선별이므로 각 잔차 채널은 단일시드이며, 통과한 채널만 이후 3시드·6시드로 확대합니다.

- compact slow: 핵심 공식 피처, decay `0.75`, depth 6
- expanded slow: 전체 기존 피처, decay `0.55`, depth 8
- expanded recent: 전체 기존 피처, decay `0.30`, depth 8
- 후보: 각 단독 채널과 세 채널 동일가중 평균

```bash
!python /content/baseball/0820/05_multichannel_residual_architecture_screen_colab.py \
  --train /content/dataset/data/train.csv \
  --output /content/drive/MyDrive/0820_multichannel_residual_architecture.json \
  --task-type GPU
```

2022·2023·2024 모두 직접 분류기보다 개선되고 2024 증분이 `+5` 이상인 구조만 시드 앙상블 단계로 진행합니다.

### 결과

- compact slow: `-102.3655 / -1488.9070 / -94.7010`
- expanded slow: `-60.9727 / -1381.2311 / -47.4203`
- expanded recent: `-89.7824 / -1454.2702 / -64.3323`
- 동일가중 평균: `-38.4903 / -1381.5256 / -6.0587`
- 동일가중 평균 pooled 변화: `-469.2007`
- 선택 결과: 없음
- 판정: `reject_multichannel_residual_architecture`
- 결론: 피처 범위와 시간 감쇠를 달리해도 동적 기준확률 잔차 구조가 직접 분류기를 넘지 못해 시드 확대와 gate 단계 중단

## 04. 동적 투수 기준확률 잔차 모델

공개 저장소 검토에서 반복적으로 성과가 있었던 `동적 기준확률 + 잔차 학습 + 3개 연도 순방향 검증` 원리만 독립 구현합니다. 외부 코드·모델·Trackman 매핑·고정 계수는 사용하지 않습니다.

공식 `asof_pitcher_n`, 커리어 성공률, 최근 1·3·5경기 성공률로 투수 기준확률을 만들고 CatBoostRegressor가 `정답-기준확률`을 학습합니다. 현재 직접 CatBoost 분류기와 2022·2023·2024에서 비교합니다.

```bash
!python /content/baseball/0820/04_dynamic_pitcher_baseline_residual_screen_colab.py \
  --train /content/dataset/data/train.csv \
  --output /content/drive/MyDrive/0820_dynamic_pitcher_baseline_residual.json \
  --task-type GPU
```

세 연도 모두 개선되고 2024 증분이 `+5` 이상일 때만 다중 잔차 채널과 행별 결합 단계로 진행합니다.
## 외부 OOF 행 조건 오차 진단

외부 공개 OOF를 제출 예측에 섞지 않고, 동일한 공식 Train 행에서 우리 기준 OOF보다 오차가 작은 조건만 찾는다. target과 `pitcher_id`가 완전히 일치하지 않으면 즉시 중단한다.

```bash
python 0820/06_external_oof_error_condition_audit_colab.py \
  --train /content/dataset/data/train.csv \
  --external-root /content/baseball/LG-Aimers-9th \
  --output /content/drive/MyDrive/0820_external_oof_error_condition_audit.json \
  --task-type GPU
```

결과에서 2023·2024 모두 양수인 조건만 다음 독립 피처 실험 후보로 사용한다. 외부 예측값·계수·모델은 ZIP에 포함하지 않는다.

## 카운트·손 조합·LI 잔차 차등 선별

외부 OOF 진단에서 반복된 조건 구조만 공식 Train으로 독립 검증한다. 직전 시즌의 조건별 잔차에서 시즌 전체 잔차를 뺀 차등을 축소하여 다음 시즌에 적용한다.

```bash
python 0820/07_count_hand_leverage_differential_screen_colab.py \
  --train /content/dataset/data/train.csv \
  --output /content/drive/MyDrive/0820_count_hand_leverage_differential.json \
  --task-type GPU
```
