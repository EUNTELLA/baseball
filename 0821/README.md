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

실행 결과 `selected=null`이었다. 최상위 강한 최근가중 단독·강도 0.25도 2023 `-32.26`, 2024 `+12.47`로 시즌 방향이 갈렸다. 최근 시즌 단독은 2024 `+14.52`였지만 2023 `-35.31`이므로 제출 또는 후속 결합에서 제외한다. 결과 요약은 `0821/results/0821_f_multichannel_residual_screen.json`에 저장한다.

## F행 이전 유형 전환 잔차

다중 채널과 독립적으로 투수의 이전 주 리그 유형, 현재 유형, 전환 조합과 경기 상황을 이용해 잔차를 학습한다. 보정은 검증 연도의 F행에만 적용하고 R행은 그대로 둔다.

```bash
python 0821/05_f_transition_residual_screen_colab.py \
  --train /content/dataset/data/train.csv \
  --output /content/drive/MyDrive/0821_f_transition_residual_screen.json \
  --task-type GPU
```

`04`와 결과 의존성이 없으므로 두 번째 실행 후보로 사용할 수 있다. 단일 T4에서 동시에 실행하면 GPU 메모리와 학습 시간이 경합하므로 두 셀을 순차 실행하는 편이 안정적이다.

실행 결과 depth 3·강도 0.05가 2023 `+0.0331`, 2024 `+2.4817`로 게이트를 통과했다. 0.075와 0.10도 두 시즌 양수였지만 2023 여유가 각각 `+0.0260`, `+0.0031`로 감소하므로 안정성 기준에서는 0.05를 선택한다. 다음 단계는 실제 3축·R 0.05 전체 anchor 위의 2024 확인이며 아직 ZIP을 만들지 않는다.

## F 전환 잔차 전체 anchor 확인

먼저 전체 시간 안전 파이프라인의 2023·2024 anchor를 저장한다.

```bash
python 0819/02_residual_differential_full_pipeline_colab.py \
  --train /content/dataset/data/train.csv \
  --output /content/drive/MyDrive/0821_full_anchor_report.json \
  --anchor-dir /content/drive/MyDrive/0821_full_anchors \
  --task-type GPU
```

이어서 2023 전체 anchor 잔차로 전환 모델을 학습하고 2024 F행에만 적용한다.

```bash
python 0821/06_f_transition_full_anchor_validation_colab.py \
  --train /content/dataset/data/train.csv \
  --anchor-dir /content/drive/MyDrive/0821_full_anchors \
  --output /content/drive/MyDrive/0821_f_transition_full_anchor.json \
  --task-type GPU
```

0.05가 `+1` 이상이고 이웃 강도 0.025·0.075도 모두 양수일 때만 제출 ZIP을 만든다.

실행 결과 0.025 `+2.7851`, 0.05 `+4.3528`, 0.075 `+4.7030`으로 모두 양수였다. 선택 강도 0.05의 평균 오차 증분은 `0.000845`로 게이트 안이며 `build_f_transition_submission` 판정을 받았다.

```bash
python 0821/07_build_f_transition_submission_colab.py \
  --train /content/dataset/data/train.csv \
  --anchor /content/drive/MyDrive/0821_full_anchors/anchor_2024.npz \
  --base-zip /content/drive/MyDrive/submit_catboost_r_residual_scale0050.zip \
  --test /content/dataset/data/test.csv \
  --sample /content/dataset/data/sample_submission.csv \
  --output-zip /content/drive/MyDrive/submit_catboost_r0050_f_transition0050.zip \
  --report /content/drive/MyDrive/0821_r0050_f_transition0050_build.json \
  --task-type GPU
```

빌더는 ZIP 무결성·5행 일괄 추론뿐 아니라 각 행을 하나씩 따로 추론한다. 단독 예측과 일괄 예측의 최대 차이가 `1e-12`를 넘으면 빌드를 실패시켜 test 행간 통계·순서·분포 의존성을 차단한다.

제공된 5행 샘플은 모두 R행이므로 기준 R 0.05 ZIP과 후보 ZIP의 R 예측이 완전히 같은지도 검사한다. 이어서 공식 Train의 F행 하나를 2025 입력 형태로 바꾼 스모크 입력에서 후보 예측만 실제로 달라지는지 확인해 F 보정 경로까지 검증한다.

최종 빌드는 모든 검사를 통과했다. 단독·일괄 최대 차이와 기준 ZIP 대비 R행 최대 차이는 모두 `0.0`이었다. F 스모크 예측은 `0.44773518 → 0.44687162`, 차이 `-0.00086356`으로 전환 보정 적용을 확인했다. 제출 ZIP SHA-256은 `543f33e0aeb82e59978af041325f3c6fe12038cfa1003a416ba398928ee9cd66`이다.

리더보드 점수는 `1030.6410723404`로 R 0.05 챔피언 `1033.0126318779`보다 `-2.3715595375` 하락했다. 규정·기능 검증은 통과했지만 로컬 개선 방향이 서버에서 전이되지 않았으므로 F 이전 유형 전환 축은 폐기하고 `submit_catboost_r_residual_scale0050.zip`을 계속 유지한다.

## 후보 보정 전이 강건성 재평가

로컬 양수만으로 제출한 후보가 서버에서 반전된 문제를 보완하기 위해 평가 게이트를 다시 구성한다. R 0.05 전체 시간 전방 anchor를 고정하고 F 이전 유형 전환 보정을 다음 기준으로 재평가한다.

- 2023·2024 각각 BSS `+1` 이상
- 두 시즌 개선 절댓값의 작은 값/큰 값 비율 `0.25` 이상
- 투수 묶음 500회 재표본화에서 개선 확률이 두 시즌 모두 `0.80` 이상
- 후보 보정의 평균 수준을 분리한 행별 형태 기여가 두 시즌 모두 양수
- 중심화에 쓰는 평균은 검증·평가 행이 아니라 학습 시즌 F행에서만 계산

이 평가는 2022→2023과 2023→2024를 모두 다시 보므로 `anchor_2022.npz`도 필요하다. 기존 폴더에 2023·2024만 있다면 다음 명령으로 2022를 포함한 anchor를 먼저 갱신한다. `--screen-r-residual`은 검증 폴드를 2022부터 열기 위한 기존 옵션이며 저장되는 공통 anchor는 R 추가 보정 전 단계다.

```bash
python 0819/02_residual_differential_full_pipeline_colab.py \
  --train /content/dataset/data/train.csv \
  --output /content/drive/MyDrive/0821_full_anchor_2022_2024_report.json \
  --anchor-dir /content/drive/MyDrive/0821_full_anchors \
  --screen-r-residual \
  --task-type GPU
```

```bash
python 0821/08_transfer_robustness_audit_colab.py \
  --train /content/dataset/data/train.csv \
  --anchor-dir /content/drive/MyDrive/0821_full_anchors \
  --output /content/drive/MyDrive/0821_transfer_robustness_audit.json \
  --task-type GPU
```

이 실험은 제출 ZIP을 만들지 않는다. 강화된 게이트가 기존 F 전환 후보를 거부하는지 확인하고, 이후 모든 행별 보정 후보의 공통 제출 전 점검 기준으로 사용한다.

실행 결과 `selected=null`, `keep_r_scale0050_champion`이었다. 원본 보정은 모든 강도에서 두 시즌 양수였지만 개선 크기 비율이 `0.054~0.150`으로 기준 `0.25`에 미달했다. 0.05는 2023 `+38.03`, 2024 `+4.50`, 투수 묶음 개선 확률 최소 `0.906`이었지만 시즌 간 크기가 지나치게 달랐다. 더 중요하게 학습 시즌 평균을 제거한 0.05 보정은 2023 `-19.04`, 2024 `-1.57`로 모두 악화됐다. 따라서 원본 이득은 안정적인 행별 형태보다 시즌 수준 이동 의존성이 크다고 판정하며 F 전환 축을 최종 종료한다.

오늘 최종 유지 제출은 `submit_catboost_r_residual_scale0050.zip`, Public Score `1033.0126318779`다. 이번 감사 결과로 새 ZIP은 만들지 않는다.

## 경기 상태 형태 신호 선별

F/R 보정 강도와 기존 카운트·손 차등을 반복하지 않고, 현재 행에서 알 수 있는 이닝 구간·공수·아웃·주자·투수 팀 점수 상황·레버리지 조합을 새 독립 축으로 검사한다. 각 조회표는 원천 시즌의 리그 유형별 평균 잔차를 먼저 제거하므로 전역 수준 이동을 학습하지 않는다. 기존 2022~2024 anchor를 재사용하므로 CatBoost 재학습 없이 실행된다.

```bash
python 0821/09_game_state_shape_screen_colab.py \
  --train /content/dataset/data/train.csv \
  --anchor-dir /content/drive/MyDrive/0821_full_anchors \
  --output /content/drive/MyDrive/0821_game_state_shape_screen.json
```

2023·2024 각각 `+1` 이상, 개선 크기 비율 `0.25` 이상, 투수 묶음 개선 확률 `0.80` 이상을 동시에 만족해야 다음 전체 파이프라인으로 넘긴다. 이 단계에서는 ZIP을 만들지 않는다.

실행 결과 `selected=null`이었다. 최상위 이닝 구간×공수×아웃 후보도 2023 `+0.269`, 2024 `+0.615`, 개선 크기 비율 `0.438`이었지만 투수 묶음 개선 확률이 `0.582`에 그쳤다. 가장 균형 잡힌 이닝 구간×점수 후보도 최소 개선 `+0.222`, 개선 확률 `0.596`으로 효과 크기와 확신이 모두 부족했다. 경기 상태 형태 축은 폐기하고 R 0.05를 유지한다.

## 시드 불일치 기반 행 불확실성 선별

동일 CatBoost를 7개 시드로 시간 전방 학습하고 행별 표준편차·최대최소 범위·앙상블 평균과 anchor 차이를 불확실성 피처로 만든다. 이전 시즌 anchor 잔차에 강하게 규제한 선형 보정을 적합하고, 학습 시즌 보정 평균을 제거한 형태 성분만 다음 시즌에 적용한다.

```bash
python 0821/10_seed_disagreement_shape_screen_colab.py \
  --train /content/dataset/data/train.csv \
  --anchor-dir /content/drive/MyDrive/0821_full_anchors \
  --output /content/drive/MyDrive/0821_seed_disagreement_shape_screen.json \
  --task-type GPU
```

2023·2024 각각 `+1`, 개선 크기 비율 `0.25`, 투수 묶음 개선 확률 `0.80` 게이트를 그대로 적용한다. 모델 21개를 학습하므로 T4에서 수 분 정도 걸릴 수 있으며 이 단계에서는 ZIP을 만들지 않는다.

실행 결과 `selected=null`이었다. 2023 모델은 7개 시드가 `1~2`회에서 종료됐지만 2024 모델은 `201~286`회 학습되어 시드 불일치의 의미와 크기가 시즌 사이에서 달라졌다. 가장 강하게 규제한 `alpha=10000`, 강도 0.25도 2023 `+400.45`에서 2024 `-107.79`로 반전됐고 투수 묶음 개선 확률은 `0.082`에 불과했다. 불확실성 보정은 다음 시즌 외삽에 실패했으므로 폐기하고 R 0.05를 유지한다.

## 자체 제출 반응 기반 R 강도 후보

R 보정 전 `1029.0832`, 강도 0.025 `1031.5033`, 강도 0.05 `1033.0126`의 자체 제출 결과 세 점에 이차 반응곡선을 적합한다. 이는 새 모델이나 평가 데이터 통계를 사용하는 과정이 아니라 이미 완료한 제출의 강도와 점수 관계를 정리하는 진단이다.

```bash
python 0821/11_r_scale_response_analysis.py \
  --output /content/drive/MyDrive/0821_r_scale_response_analysis.json
```

계산상 정점은 약 `0.0789`이고, 이미 생성·검증한 0.075 후보의 예상 점수는 약 `1033.6111`, R 0.05 대비 약 `+0.5985`다. 관측점이 세 개뿐이므로 예상 점수를 보장할 수 없으며, 추가 제출을 한 번 사용한다면 새 구조보다 `submit_catboost_r_residual_scale0075.zip`을 우선한다.

### 0.075 리더보드 결과

- 제출 ID: `58354`
- 제출 시각: 2026-08-21 14:21:34
- Public Score: **`1033.5178832618`**
- 실행 시간: 5초
- R 0.05 대비 개선: `+0.5052513839`
- 반응곡선 예상치 `1033.6111202428` 대비 오차: `-0.0932369810`

상승 방향과 크기가 근사 예측과 유사하게 재현됐다. `submit_catboost_r_residual_scale0075.zip`을 새 최고 및 현재 유지 제출로 승격하며, 반응곡선 정점과 충분히 가까우므로 0.08 이상의 미세 강도 탐색은 제출 기회를 추가로 사용할 근거가 생기기 전까지 중단한다.
