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
