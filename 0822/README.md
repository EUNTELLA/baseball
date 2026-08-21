# 0822 anchor 재구성

현재 유지본은 R 잔차 0.075와 검증 전역 shift `-0.0416386466`을 적용해 Public `1034.5103741711`을 기록한 제출이다. 새 anchor는 이 유지본을 덮어쓰지 않고 시간 전방 구성요소부터 별도로 비교한다.

## 1단계: 구성요소 OOF 저장

```bash
python 0819/02_residual_differential_full_pipeline_colab.py \
  --train /content/dataset/data/train.csv \
  --output /content/drive/MyDrive/0822_anchor_component_build.json \
  --anchor-dir /content/drive/MyDrive/0822_anchor_components \
  --component-dir /content/drive/MyDrive/0822_anchor_components \
  --screen-r-residual \
  --task-type GPU
```

`--screen-r-residual`은 2022부터 검증 폴드를 생성하기 위해 사용한다. 저장되는 구성요소는 R 추가 보정 전의 성공확률, MR, 큰 이탈, offset, shift, 잔차 차등 anchor다.

## 2단계: 채널 결합 선별

```bash
python 0822/01_anchor_channel_stack_screen_colab.py \
  --component-dir /content/drive/MyDrive/0822_anchor_components \
  --train /content/dataset/data/train.csv \
  --output /content/drive/MyDrive/0822_anchor_channel_stack_screen.json
```

성공확률, 실패확률 여집합, offset 전·후 채널을 이전 시즌에서 anchor 평균에 맞추고 전체·F·R 영역에 `2.5~20%` 혼합한다. 두 시즌 각각 `+1`, 개선 크기 비율 `0.25`, 투수 묶음 개선 확률 `0.80`을 통과한 경우에만 전체 학습으로 넘긴다. 이후 R 0.075와 검증 shift까지 다시 적용한 완성 예측이 현재 최고를 이길 때만 제출 ZIP을 만든다.

선별 결과 `failure_complement`를 R행에 0.20 혼합한 후보가 2023 `+34.75`, 2024 `+103.00`, 개선 크기 비율 `0.337`, bootstrap 확률 `1.0`으로 통과했다. 0.05~0.20 전 구간도 모두 통과했으므로 현재 챔피언 구성까지 포함한 다음 검증으로 진행한다.

## 3단계: 현재 챔피언 위 재검증

```bash
python 0822/02_failure_complement_champion_validation_colab.py \
  --component-dir /content/drive/MyDrive/0822_anchor_components \
  --train /content/dataset/data/train.csv \
  --output /content/drive/MyDrive/0822_failure_complement_champion_validation.json \
  --task-type GPU
```

기존 anchor 잔차로 학습한 R 3시드 correction을 기준과 후보에 동일하게 적용하고 R 강도 0.075, 검증 shift 차이 `-0.0032119273`까지 공통 적용한다. 따라서 이 단계의 증분은 현재 챔피언 위에서 실패확률 여집합 혼합만 추가한 효과다.
