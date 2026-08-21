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

### 리더보드 결과

- 제출 시각: 2026-08-21 11:09:10
- Public Score: `1031.5033329643`
- 이전 최고: `1029.0832235020`
- 개선: `+2.4201094623`
- 로컬 2024 증분 대비 서버 전이율: 약 `17.4%`

R 행 저강도 잔차가 서버에서도 같은 개선 방향을 보였으므로 이 ZIP을 새 최고로 승격한다. 다만 2023 전방 검증은 음수였으므로 더 큰 강도는 한 단계씩 보수적으로 확인한다.

## R 잔차 강도 0.05·0.075 후보 생성

새 최고 ZIP에 저장된 동일한 3시드 R 잔차 모델을 재사용하고 강도만 변경한다. 따라서 재학습 없이 두 후보를 만들며 각각 ZIP 무결성과 샘플 추론을 검사한다.

```bash
python 0821/02_repack_r_residual_scales_colab.py \
  --source-zip /content/drive/MyDrive/submit_catboost_r_residual_scale0025.zip \
  --test /content/dataset/data/test.csv \
  --sample /content/dataset/data/sample_submission.csv \
  --output-dir /content/drive/MyDrive \
  --report /content/drive/MyDrive/0821_r_residual_scale_candidates.json
```

생성 파일은 `submit_catboost_r_residual_scale0050.zip`과 `submit_catboost_r_residual_scale0075.zip`이다. 2023 전방 성능 하락을 고려한 우선 제출 후보는 `scale0050`이며, 두 파일을 동시에 제출하지 않는다.

실행 결과 두 후보 모두 23개 멤버의 ZIP 무결성 검사와 5행 독립 추론을 통과했다. 0.05 후보의 샘플 평균은 `0.452576`, 0.075 후보는 `0.452402`였으며 결측은 모두 0이다. 결과 원본은 `0821/results/0821_r_residual_scale_candidates.json`에 저장한다.

### 0.05 리더보드 결과

- 제출 시각: 2026-08-21 11:20:42
- Public Score: `1033.0126318779`
- 0.025 제출 대비 개선: `+1.5092989136`
- R 잔차 적용 전 3축 기준 대비 누적 개선: `+3.9294083759`

0.05를 새 최고로 승격한다. 0.075는 생성·검증된 상태로 보관하되 별도 판단 없이 연속 제출하지 않는다.

## F행 조건 오차 전이 선별

R행은 `scale=0.05`로 동결하고 F행만 대상으로 직전 시즌 조건별 예측 오차가 다음 시즌에 재현되는지 검사한다. 기존 F 전용 모델과 실패유형 실험을 반복하지 않고 카운트×손, 카운트×레버리지, 손×주자, 카운트×이닝, 카운트×손×주자 조건을 강하게 축소한다.

```bash
python 0821/03_f_condition_error_screen_colab.py \
  --train /content/dataset/data/train.csv \
  --output /content/drive/MyDrive/0821_f_condition_error_screen.json \
  --task-type GPU
```

2023·2024가 모두 양수이고 2024 증분이 `+2` 이상인 동일 설정만 다음 전체 파이프라인 검증으로 넘긴다. 이 단계에서는 제출 ZIP을 만들지 않는다.

실행 결과 `selected=null`이었다. 최상위 카운트×손×주자 후보도 2023 `-0.1182`, 2024 `+0.3706`에 그쳤고 다른 상위 조건도 모두 2023 음수였다. 조건 테이블 방식은 폐기하고 `scale0050` 챔피언을 유지한다. 결과 요약은 `0821/results/0821_f_condition_error_screen.json`에 저장한다.

## F행 다중 잔차 채널 재구성

R행은 `scale=0.05`로 고정하고 F행에만 장기 기억, 직전 시즌, 강한 최근가중 잔차 채널을 적용한다. 각 채널은 seed 17·42·777 평균이며, 단일 채널과 세 가지 혼합을 동일 강도에서 시간 전방 비교한다.

```bash
python 0821/04_f_multichannel_residual_screen_colab.py \
  --train /content/dataset/data/train.csv \
  --output /content/drive/MyDrive/0821_f_multichannel_residual_screen.json \
  --task-type GPU
```

2023·2024 동일 혼합·동일 강도가 모두 양수이고 2024가 `+3` 이상일 때만 실패유형·리그 전환 보조 신호를 추가하는 2단계 검증으로 넘어간다.
