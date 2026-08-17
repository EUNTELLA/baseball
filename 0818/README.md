# 0818 실험

최종 기준은 공식 train 추세 CatBoost 제출의 Public Score `997.3951851847`입니다. 0817까지의 파라미터, Brier 회귀, 실패 여집합 보정 실험은 제출 기준을 통과하지 못했으므로 0818부터 구조가 다른 모델 축만 검증합니다.

## 01. 경기유형 R/F Mixture of Experts

공통 CatBoost와 R/F 경기 전용 CatBoost를 동일 피처와 3시드로 학습합니다. 각 평가 행은 자신의 `game_type` 전문가 예측만 사용하므로 test 행 간 집계가 없습니다. 전문가 비중 `0, 0.25, 0.5, 0.75, 1.0`을 비교합니다.

F 경기 성공률이 2022년 `0.7087`에서 2023년 `0.4729`로 급변했으므로 전환기 2023은 선택과 확인에서 제외합니다. 2021·2022로 비중을 선택하고 2024를 확인 전용으로 사용합니다.

```bash
!python /content/baseball/0818/01_game_type_mixture_of_experts_colab.py \
  --train /content/dataset/data/train.csv \
  --output /content/drive/MyDrive/0818_game_type_mixture_of_experts.json \
  --task-type GPU
```

개발 평균 `+5`, 최악 `-2`, 같은 평균 개선, 유형별 최악 `-5`와 2024 raw `+10`, 같은 평균 개선, 유형별 최악 `-5`를 모두 만족할 때만 전체 파이프라인 검증으로 진행합니다.
