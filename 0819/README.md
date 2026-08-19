# 0819 실험

최종 제출 기준은 공식 train 추세를 사용한 CatBoost의 Public Score `997.3951851847`입니다. 0819에서는 모델을 분리하지 않고, 기존 모델이 남긴 OOF 잔차에서 투수별 상황 차이만 추출하는 후처리 축을 검증합니다.

## 01. 투수별 OOF 잔차 차등 3축 선별

세 축은 `투수×같은손`, `투수×2스트라이크`, `투수×주자유무`입니다. 각 투수의 context 1과 0 잔차 평균 차이에 유효 표본수 기반 축소를 적용하고, 행별로 `+0.5d/-0.5d`를 더합니다.

- 손 차등: `k=1000`
- 2스트라이크 차등: `k=1000`
- 주자유무 차등: `k=2000`
- 검증 시즌 S의 표 원천: S-2와 S-1의 strictly OOF 잔차
- 주 지표: 공식 Brier Skill Score
- 보조 지표: `100000 × corr²`
- 통과 조건: 2023·2024 BSS 증분이 모두 양수이고 2024가 `+5` 이상

이번 코드는 구조 선별 단계이므로 성공모델만 사용합니다. 통과할 때만 MR/wayoff offset과 공식 train 추세 shift를 포함한 전체 `997` 파이프라인 검증 및 제출본 제작으로 진행합니다.

```bash
!python /content/baseball/0819/01_catboost_residual_differential_screen_colab.py \
  --train /content/dataset/data/train.csv \
  --output /content/drive/MyDrive/0819_catboost_residual_differential.json \
  --task-type GPU
```
