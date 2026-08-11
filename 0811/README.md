# 0811 실험

RandomForest의 Public Score가 raw `549.6413938266`에서 상수 오프셋 보정 후 `675.5498778599`로 상승한 원인을 검증하고, 보정값의 연도별 안정성을 확인합니다.

## 실행 순서

1. `01_offset_stability.ipynb`: 기존 RandomForest OOF 예측으로 2022~2024년 오프셋 안정성 비교
2. 이후 실험 후보: calibration 방식 비교, RandomForest seed 앙상블, RandomForest-CatBoost 혼합

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
