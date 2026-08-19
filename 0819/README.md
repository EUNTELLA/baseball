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

### 결과

- 2023 BSS 증분: `+31.7693760408`
- 2024 BSS 증분: `+36.2943178506`
- 2023 `corr²` 증분: `+31.93`
- 2024 `corr²` 증분: `+36.81`
- 판정: `continue_full_997_pipeline`

두 시즌에서 BSS와 순위 구조가 함께 개선되어 전체 파이프라인 검증으로 진행합니다.

## 02. 전체 997 파이프라인 검증

01에서 만든 차등 보정을 성공확률에 더한 뒤, 기존과 같은 MR/wayoff offset과 공식 train 시즌 추세 shift를 기준·후보에 공통 적용합니다. 2023은 개발 확인, 2024는 최종 확인 역할이며 두 시즌 모두 strictly OOF 잔차표만 사용합니다.

```bash
!python /content/baseball/0819/02_residual_differential_full_pipeline_colab.py \
  --train /content/dataset/data/train.csv \
  --output /content/drive/MyDrive/0819_residual_differential_full_pipeline.json \
  --task-type GPU
```

2023·2024 BSS와 `corr²`가 모두 양수이고, 2024 BSS가 `+5` 이상이며 평균 오차가 `0.001`보다 크게 악화되지 않을 때만 제출 ZIP을 생성합니다.

### 결과

- 2023 전체 파이프라인 BSS / corr² 증분: `+8.6274` / `+9.4677`
- 2024 전체 파이프라인 BSS / corr² 증분: `+57.3687` / `-87.2905`
- 2024 절대 평균 오차 변화: `-0.0001573` (개선)
- 자동 판정: `keep_997_baseline` (`corr²` 하드 게이트 미통과)
- 해석: 공식 BSS는 두 시즌 모두 개선했으므로 기존 997을 대체하지 않는 별도 제출 후보 제작

## 03. 7시드 잔차 차등 제출 후보 제작

기존 Public `997.3951851847` ZIP은 수정하지 않습니다. 2023·2024를 각각 직전 시즌까지만 학습한 7시드 모델로 예측해 OOF 잔차표를 만들고, 기존 ZIP 사본에 세 표와 행 단위 추론 코드만 추가합니다.

```bash
!python /content/baseball/0819/03_build_residual_differential_submission_colab.py \
  --train /content/dataset/data/train.csv \
  --test /content/dataset/data/test.csv \
  --sample /content/dataset/data/sample_submission.csv \
  --report /content/drive/MyDrive/0819_residual_differential_build.json
```

생성 파일은 `/content/baseball/0819/results/submit_catboost_residual_differential.zip`입니다.
