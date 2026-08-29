# 0829 실행 파일 정리

2026-08-29에는 현재 자체 기준 최고점 주변에서 F profile, R response, R rebuild를 검증했다.

## 파일

- `01_train_f_postregime_profile_screen_colab.py`: F post-regime profile 선별
- `02_build_train_f_postregime_profile_submission_colab.py`: F post-regime profile 제출 ZIP 생성
- `03_repack_train_r_strict_response_colab.py`: R strict response 강도 후보 재패키징
- `04_train_r_gap_pocket_screen_colab.py`: R gap/pocket residual 선별
- `05_train_r_residual_rebuild_screen_colab.py`: R residual rebuild 선별
- `06_build_train_r_residual_rebuild_submission_colab.py`: R residual rebuild 제출 ZIP 생성

## 제출 결과

- `0.875`: `1065.089435112`, 현재 자체 기준 최고
- `catboost_rstrict0875_fptc010`: `1064.9405293377`, 하락
- `submit_catboost_rstrict0875_rrebuild030`: `1013.9671783434`, 하락

## 판정

- 현재 유지 후보는 `0.875` 계열이다.
- 2024 단일 fold에서 크게 오른 R rebuild는 서버 전이에 실패했으므로 폐기한다.
