# 0816 실험

0814의 실제 제출 및 엄격 Walk-forward 결과를 바탕으로 calibration blend를 실험합니다.

## 기준 결과

| 모델 | Public Score | 판단 |
| --- | ---: | --- |
| 가중 RF + 고정 offset | 727.5961617195 | 안정 기준 |
| 가중 RF + Ridge meta | 743.6924889713 | affine 대비 하락 |
| 가중 RF + affine | **761.7509255482** | 현재 최고 |

## 01. Affine–Rates Ridge 혼합 검증

Affine 보정과 `asof_*` 비율만 사용하는 Ridge 보정의 확률을 혼합합니다.

```powershell
python 0816/01_calibration_blend_cv.py
```

혼합 비율은 `0.0~1.0`을 `0.1` 간격으로 비교합니다. 2024 내부 연속 5-fold와 `2023 보정 학습 → 2024 평가` 엄격 Walk-forward 결과를 모두 저장하며, 어느 한쪽에서만 좋아지는 후보는 채택하지 않습니다.

### 검증 결과

| Offset 비중 | Rates Ridge 비중 | 내부 CV Score | 엄격 Score |
| ---: | ---: | ---: | ---: |
| 1.0 | 0.0 | 588.21 | 573.25 |
| 0.9 | 0.1 | 594.09 | 584.98 |
| 0.8 | 0.2 | 599.14 | 591.62 |
| **0.7** | **0.3** | **603.36** | **593.17** |
| 0.6 | 0.4 | 606.75 | 589.65 |
| 0.5 | 0.5 | 609.31 | 581.04 |

엄격 Walk-forward를 1순위로 선택해 `Offset 0.7 + Rates Ridge 0.3`을 후속 후보로 채택합니다.

## 02. Offset–Rates Ridge 최종 후보

```powershell
python 0816/02_offset_rates_blend_submission.py
```

2024 OOF에서 학습한 offset 보정 확률 `70%`와 `asof_*` 비율 Ridge 확률 `30%`를 행 단위로 혼합합니다. 생성 파일은 `0816/results/submit_rf_weighted20_offset70_rates30.zip`입니다.

### 제출 결과

- **Public Score:** `747.3463802872`
- **affine 대비:** `-14.4045452610`
- **판단:** Ridge meta보다는 높지만 affine보다 낮아 최종 후보에서 제외

## 03. LightGBM 시간 순서 검증

보정 방식의 조합을 중단하고 기본 모델을 RandomForest에서 LightGBM으로 변경합니다.

```powershell
python 0816/03_lightgbm_time_validation.py
```

`2021~2022 → 2023`에서 보정식을 학습해 `2022~2023 → 2024` 예측에 적용하는 엄격 검증과, 2024 OOF에서의 affine 상한을 함께 비교합니다. RF 기준은 엄격 Score `171.71`, 2024 affine Score `598.21`, 실제 Public Score `761.7509255482`입니다.

## 04. RF–LightGBM 혼합 검증

```powershell
python 0816/04_rf_lightgbm_blend_validation.py
```

LightGBM 단독이 2024 affine 기준에서 RF보다 낮으므로 바로 제출하지 않습니다. LightGBM 비중 `0~0.5`를 탐색하고 엄격 검증과 2024 affine 검증에서 RF 단독을 모두 이길 때만 패키징합니다.

### 선택 결과

| 모델 | 엄격 Score | 2024 affine Score |
| --- | ---: | ---: |
| RF 100% | 171.71 | 598.21 |
| **RF 85% + LightGBM 15%** | **201.03** | **603.06** |

두 기준을 모두 개선하고 2024 affine 점수가 가장 높은 LightGBM 15% 혼합만 최종 패키징합니다.

## 05. 최종 제출 후보 생성

```powershell
python 0816/05_rf_lightgbm_blend_submission.py
```

최종 후보는 `0816/results/submit_rf85_lgb15_affine.zip` 하나입니다. 실제 검증 전 기준점은 기존 RF affine의 Public Score `761.7509255482`입니다.

## 06. CatBoost d6·FE10·7시드·실패모드 offset·고정 shift 기준 League-rate 선별 검증

Public Score `998.0030076995`의 `CatBoost depth 6 + FE 10개 + 7시드 평균 + MR/wayoff offset + 고정 logit shift` 모델을 새 기준점으로 변경했습니다(run ID `012_shift_full`). 동일한 `open.zip`, T4 GPU, CatBoost 3시드 조건에서 기존 성공모델 구조와 league-rate baseline을 다시 학습해 비교했습니다.

| 모델 | 2024 Score | 예측 평균 |
| --- | ---: | ---: |
| CatBoost d6 + FE10 성공모델 구조 | 769.00 | 0.49490 |
| **League-rate baseline** | **809.11** | **0.48283** |

- **Score 개선:** `+40.1010`
- **개별 시드:** `795.08 / 798.33 / 799.55`
- **2024 baseline 외삽값:** `0.487742`
- **판정:** `continue_to_7_seed_build`
- **결과 파일:** `0816/results/0816_submit012_league_baseline_result.json`

평균 편향 감소가 개선의 일부이므로 기준 모델의 전역 logit shift `-0.04163865`와 중복 적용하지 않습니다. 다음 단계는 league-rate 7시드 전체 학습, 2025 baseline 고정 저장 및 제출 ZIP 검증입니다.

## 07. League-rate 7시드 제출 후보 빌드

```text
0816/07_submit012_league_baseline_build_colab.py
```

기준 모델의 MR/wayoff 보조모델과 고정 offset을 유지하고 CatBoost d6·FE10 성공모델만 league-rate 7시드로 교체합니다. 2025 baseline은 학습·검증 결과로 확정한 `0.4819150787`을 meta에 저장하며, 기존 전역 logit shift `-0.04163865`는 제거합니다. Colab 실행 방법은 `0816/COLAB_07.md`에 기록합니다.

### 7시드 빌드 및 ZIP 실행 결과

- **검증 Brier:** `0.2477786359`
- **검증 Score:** `811.9434`
- **검증 예측 평균:** `0.4829028`
- **기존 CatBoost d6·FE10 동일 조건 대비:** `+42.9393`
- **최적 iteration:** `366 / 330 / 317 / 312 / 334 / 319 / 384`
- **ZIP 크기:** `2.8067 MiB`
- **ZIP 자체 실행:** 성공
- **샘플 출력:** 5행, 결측 0, 확률 범위 `0.4119411~0.4969956`
- **제출 후보 파일:** `submit013_league_baseline.zip`

샘플 test는 5행뿐이므로 샘플 예측 평균 `0.4563`은 모델 선택이나 전체 test 평균 추정에 사용하지 않습니다.
