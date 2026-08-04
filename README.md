# 투구 단위 제구 성공 확률 예측 AI 모델

투구 직전까지 확인 가능한 경기 상황, 선수·주자 정보와 과거 이력을 이용해 각 투구의 `control_success = 1` 확률을 예측하는 LG Aimers 온라인 해커톤 Phase 2 프로젝트입니다.

모델과 파생변수에는 반드시 해당 투구 **이전 시점에 이용 가능한 정보만** 사용할 수 있습니다.

## 문제 정의

`control_success`는 다음과 같이 정의된 이진 Target입니다.

| 값 | 의미 |
| ---: | --- |
| `1` | 제구 성공 |
| `0` | 제구 실패 |

다음 세 가지 경우는 제구 실패이며, 그 외 유효한 투구는 제구 성공입니다.

1. 스트라이크존 가운데 부근으로 들어간 공
2. 스트라이크존에서 크게 벗어난 공
3. 포수의 요구 방향과 반대로 들어간 공

모델의 출력은 다음 확률입니다.

```text
P(control_success = 1 | 투구 직전까지의 정보)
```

## 공식 평가 지표

리더보드는 **Brier Skill Score**로 평가합니다. 예측 확률이 실제 정답에 가까울수록 높은 점수를 받습니다.

<details>
<summary><strong>평가 산식 펼치기</strong></summary>

```text
Brier Score = mean((p_i - y_i)^2)
r = mean(y_i)
평균 제구율 Brier Score = r × (1 - r)

Score = max(0, 100000 × (1 - Brier Score / 평균 제구율 Brier Score))
```

| 기호 | 의미 |
| --- | --- |
| `p_i` | i번째 투구의 제구 성공 예측 확률 |
| `y_i` | i번째 투구의 실제 정답 (`0` 또는 `1`) |
| `r` | 전체 평가 데이터의 평균 제구 성공률(비공개) |

- Public Score는 전체 테스트 데이터 100%로 계산합니다.
- Private Score는 대회 종료 시점의 Public Score입니다.
- 평균 제구율 기준 모델과 성능이 같거나 그보다 나쁘면 점수는 `0`입니다.

</details>

개발 과정에서는 전체 시간순 OOF Brier Skill Score로 모델을 선택하고 Brier Score, Log Loss, ROC-AUC를 보조 지표로 확인합니다.

## 평가 및 Phase 3 진출

<details>
<summary><strong>수료 및 선발 방식 펼치기</strong></summary>

### LG Aimers 수료 조건

- Phase 1 이수
- Phase 2 Public Score `549.51` 이상
- 기준 점수는 운영진 베이스라인 코드를 공식 평가 환경에서 실행한 결과

### Phase 3 선발

1. Private Score 100%로 1차 평가
2. 동점자는 기존 리더보드 순위 산정 방식 적용
3. Phase 3 진출 희망 팀 중 Private 상위 팀 약 100명은 코드와 PPT 제출
4. 제출 자료와 코드 검증을 모두 통과한 상위 팀이 Phase 3 진출

</details>

## 코드 제출 환경

| 항목 | 제한 |
| --- | --- |
| 테스트 샘플 | 245,789개 |
| 패키지 설치 | 10분 이하 |
| 추론 실행 | 10분 이하 |
| ZIP 크기 | 10GB 이하 |
| 압축 해제 후 크기 | 32GB 이하 |
| OS | Ubuntu 22.04.5 LTS |
| Python | 3.11.15 |
| CUDA | 12.8 |
| GPU | NVIDIA L4, VRAM 22.4GiB |
| CPU / RAM | 6 vCPU / 28GB |
| 인터넷 | 패키지 설치 외 비활성화 |

<details>
<summary><strong>submit.zip 구조 펼치기</strong></summary>

```text
submit.zip
├── model/
│   └── model.joblib
├── script.py
└── requirements.txt
```

평가 서버는 다음 항목을 자동으로 추가합니다.

```text
submit.zip
├── model/
├── script.py
├── requirements.txt
├── data/                  # 읽기 전용 평가 데이터
└── output/
    └── submission.csv     # script.py가 생성해야 하는 결과
```

- ZIP 내부에 별도의 최상위 폴더를 추가하지 않습니다.
- `requirements.txt`는 `pip install -r requirements.txt`로 설치 가능해야 합니다.
- `data/`는 읽기 전용이며 결과는 반드시 `output/submission.csv`에 저장합니다.

</details>

## 현재 구현

- 공식 Brier Skill Score 계산
- 시간순 Fold 및 전체 OOF Brier Skill Score 기반 모델 선택
- 평균 확률과 Logistic Regression 개발 기준 모델
- 현재 투구의 정답을 사용하지 않는 투수 과거 성공률 특징
- 실제 `data/`의 train, test, sample submission CSV 자동 식별
- 모델 아티팩트 `model/model.joblib` 저장
- 독립 실행형 제출 진입점 `script.py`
- 샘플 제출의 컬럼과 행 순서를 보존한 `output/submission.csv` 생성
- 합성 데이터 기반 회귀 테스트

현재 `configs/schema.json`의 입력 특징은 개발용 계약입니다. 공식 데이터가 제공되면 실제 컬럼명과 파일 구조에 맞춰 갱신해야 합니다. Target 이름은 공식 명칭인 `control_success`로 통일했습니다.

## 프로젝트 구조

```text
baseball/
├── script.py                         # 평가 서버 추론 진입점
├── requirements.txt                  # 제출용 최소 의존성
├── requirements-dev.txt              # 개발·시각화 의존성
├── requirements-notebook.txt         # Notebook 의존성
├── configs/schema.json                # 데이터 컬럼 계약
├── src/baseball_platform/
│   ├── competition_data.py           # 실제 대회 CSV 탐색
│   ├── evaluation.py                 # Brier Skill Score 및 모델 비교
│   ├── pipeline.py                   # 합성 데이터 개발 파이프라인
│   ├── models/                       # 모델
│   ├── transforms/                   # 누수 방지 특징 생성
│   ├── quality/                      # 데이터·제출 검증
│   └── validation/                   # 시간순 Fold
├── notebooks/                        # 분석 Notebook
└── tests/                            # 자동 테스트
```

`script.py`는 제출 ZIP에 `src/`를 포함하지 않아도 실행되도록 필요한 입력 탐색과 제출 저장 로직을 자체 포함합니다.

## 개발 파이프라인 실행

```powershell
python -m pip install -r requirements-dev.txt
$env:PYTHONPATH="src"
python -m baseball_platform.pipeline
```

기본 출력은 `data/generated/`에 저장됩니다.

- `synthetic_train.csv`, `synthetic_test.csv`
- `fold_results.csv`, `leaderboard.csv`
- `model_comparison.png`
- `model/model.joblib`
- `synthetic_submission.csv`

## 테스트

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests -v
```

## Notebook

```powershell
python -m pip install -r requirements-notebook.txt
jupyter lab
```

[모델 대시보드](notebooks/model_dashboard.ipynb)에서 Fold별 성능과 예측 확률 분포를 확인할 수 있습니다.

## 다음 작업

- [ ] 공식 데이터 스키마를 `configs/schema.json`에 반영
- [ ] 샘플 제출의 실제 예측 컬럼 확인
- [ ] 실제 데이터 기반 시간순 검증 구성
- [ ] 부스팅 모델과 확률 보정 비교
- [ ] Ubuntu/Python 3.11 환경에서 설치·추론 10분 제한 검증
- [ ] 최종 `submit.zip` 자동 패키징 검사
