# 0811 실험

RandomForest의 Public Score가 raw `549.6413938266`에서 상수 오프셋 보정 후 `675.5498778599`로 상승한 원인을 검증하고, 보정값의 연도별 안정성을 확인합니다.

## 실행 순서

1. `01_offset_stability.ipynb`: 기존 RandomForest OOF 예측으로 2022~2024년 오프셋 안정성 비교
2. `02_recent_window_rf_submission.py`: 2023~2024 최근 데이터만 학습한 RandomForest 제출 ZIP 생성

## 최근-window RF

2023년만 학습하고 2024년을 검증한 결과, 기존 전체 기간 RF보다 시간 이동에 대한 성능이 개선됐습니다.

| 모델 | 2024 Raw Score | 평균 보정 후 Score |
| --- | ---: | ---: |
| 전체 기간 기존 RF | 415.57 | 약 507 |
| 2023 학습 recent RF | 560.10 | 597.64 |

최종 후보는 2023~2024년으로 학습하고 2023→2024 검증에서 측정한 평균 잔차 `-0.0096837591`을 적용합니다.

```powershell
python 0811/02_recent_window_rf_submission.py
```

생성 파일은 `0811/results/submit_rf_recent_calibrated.zip`입니다.

### 제출 결과

| 제출명 | 제출일 | Public Score | 추론 시간 | 이전 최고 대비 |
| --- | --- | ---: | ---: | ---: |
| `recent_window_randomforest_rf_submission` | 2026-08-14 18:45:02 | **704.320910303** | 3초 | **+28.7710324431** |
| `submit_rf_2024_only_calibrated_report` | 2026-08-14 19:05:04 | 699.3861996099 | 3초 | -4.9347106931 |

최근-window RF가 전체 기간 보정 RF의 `675.5498778599`를 넘어 현재 최고 제출본이 됐습니다. 오래된 2019~2022년 데이터를 제외하는 방향이 2025 평가 데이터의 시간 분포에 더 적합하다는 가설을 지지합니다.

## 단일 최근 시즌 추가 실험

동일한 RF 설정으로 한 시즌만 학습해 다음 시즌을 예측했습니다.

| 학습 → 검증 | Raw Score | 평균 보정 후 Score | 측정 편향 |
| --- | ---: | ---: | ---: |
| 2022 → 2023 | -916.63 | -766.03 | -0.01940337 |
| 2023 → 2024 | 560.10 | 597.64 | -0.00968376 |

단일 시즌 학습은 연도에 따라 편차가 크므로 안정적인 기본 모델로 확정하지 않습니다. 다만 2023년 이후 구조가 안정됐는지 확인하기 위해 2024년 단독 학습본을 추가 제출 후보로 생성합니다.

```powershell
python 0811/02_recent_window_rf_submission.py --train-years 2024 --output-stem submit_rf_2024_only_calibrated
```

2024년 단독 학습본의 Public Score는 `699.3861996099`로, 2023~2024년 학습본보다 `4.9347106931` 낮았습니다. 따라서 2023년 데이터를 완전히 제거하지 않고 최근 시즌에 가중치를 높이는 방향이 다음 후보입니다.

첫 노트북은 아래 위치에서 `01_rf_oof.npz`를 자동으로 찾습니다.

- `0811/results/01_rf_oof.npz`
- `0726/results/01_rf_oof.npz`
- `0826/results/01_rf_oof.npz` (이전 폴더명 호환)
- `/content/drive/MyDrive/baseball-results/01_rf_oof.npz`

파일이 없다면 `0726/01_time_cv_random_forest.ipynb`를 먼저 실행하거나 Google Drive에 저장한 OOF 파일을 `0811/results/`로 복사합니다.

## 산출물

- `results/01_offset_grid.csv`: 오프셋별 전체 및 연도별 지표
- `results/01_offset_stability.json`: 분석 조건, 연도별 최적값, 안정성 기준 권장값

오프셋은 Public 리더보드 점수가 아니라 시간 순서 OOF의 평균 Brier Score를 기준으로 선택합니다. 연도별 결과와 최악 연도 성능도 함께 확인하여 특정 연도에만 맞는 보정값을 피합니다.
