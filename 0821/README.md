# 2026-08-21 실험

## R 행 저강도 잔차 제출 탐침

시간 전방 검증에서는 R 잔차가 2023 음수·2024 양수였으므로 정식 승격하지 않는다. 사용자 지정 제출 탐침으로 가장 보수적인 강도 0.025만 빌드한다.

먼저 2024의 현재 최고 전체 anchor 예측을 저장한다.

```bash
python 0819/02_residual_differential_full_pipeline_colab.py \
  --train /content/dataset/data/train.csv \
  --output /content/drive/MyDrive/0821_r_residual_rebuild.json \
  --task-type GPU \
  --axes hand,two_strikes,runners_on \
  --weight 1.0 \
  --screen-r-residual \
  --anchor-output /content/drive/MyDrive/0821_anchor_2024.npz
```

현재 최고 ZIP이 Colab에 없다면 먼저 다시 만든다.

```bash
python 0819/03_build_residual_differential_submission_colab.py \
  --train /content/dataset/data/train.csv \
  --test /content/dataset/data/test.csv \
  --sample /content/dataset/data/sample_submission.csv \
  --report /content/drive/MyDrive/0821_base_1029_rebuild.json
```

마지막으로 R residual 모델을 학습하고 제출 ZIP을 만든다.

```bash
python 0821/01_build_r_residual_probe_colab.py \
  --train /content/dataset/data/train.csv \
  --anchor /content/drive/MyDrive/0821_anchor_2024.npz \
  --base-zip /content/baseball/0819/results/submit_catboost_residual_differential.zip \
  --test /content/dataset/data/test.csv \
  --sample /content/dataset/data/sample_submission.csv \
  --output-zip /content/drive/MyDrive/submit_catboost_r_residual_scale0025.zip \
  --report /content/drive/MyDrive/0821_r_residual_scale0025_build.json \
  --task-type GPU
```

제출 후보는 `/content/drive/MyDrive/submit_catboost_r_residual_scale0025.zip` 하나다.

### 빌드 결과

- R residual 학습 행: 2024 R 행 223,497개
- 모델: CatBoostRegressor seed 17/42/777, 각 1200회
- 적용 강도: 0.025
- Train 전체 평균 재계산이나 test 행간 집계 없음
- ZIP 무결성 검사: 통과
- 샘플 추론: 5행, 결측 0, 범위 `0.399282~0.494669`, 평균 `0.452750`
- 최종 파일: `/content/drive/MyDrive/submit_catboost_r_residual_scale0025.zip`

이 ZIP은 시간 전방 게이트를 통과한 정식 승격본이 아니라 2024의 강한 양수 신호가 서버에서 전이되는지 확인하는 1회 탐침이다. 기존 `1029.0832` 제출은 계속 기준 최고로 보존한다.
