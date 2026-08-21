# 0822 anchor 재구성

현재 유지본은 R 잔차 0.075와 전역 shift 후보이며, 새 anchor는 제출과 분리해 시간 전방 구성요소부터 다시 비교한다.

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

성공확률, 실패확률 여집합, offset 전·후 채널을 이전 시즌에서 anchor 평균에 맞추고 전체·F·R 영역에 `2.5~20%` 혼합한다. 두 시즌 각각 `+1`, 개선 크기 비율 `0.25`, 투수 묶음 개선 확률 `0.80`을 통과한 경우에만 내일 전체 학습과 제출 ZIP 생성으로 진행한다.
