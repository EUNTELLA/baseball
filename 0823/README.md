# 0823 실행 파일 정리

2026-08-23에 만든 OOF, F 경로, R 경로 실험 파일을 한 폴더에 모았다.

## 핵심 제출 흐름

- `03_build_strict_r_blend_submission_colab.py`: F general 기준에 R strict 경로를 결합
- `11_train_f_strict_oof_student_screen_colab.py`: strict OOF 기반 F student 검증
- `14_train_f_route_blend_oof_audit_colab.py`: F route 혼합 OOF 감사
- `18_oof_build_own_champion_oof_colab.py`: 자체 champion strict OOF 생성

## 재구성·검증 파일

- `01_strict_anchor_comparison_colab.py`
- `02_strict_anchor_champion_validation_colab.py`
- `04_r_multichannel_reconstruction_screen_colab.py`
- `05_failure_complement_strength_extension_colab.py`
- `06_strict_anchor_strength_extension_colab.py`
- `07_repack_failure_complement_strength_colab.py`
- `08_tree_prior_strict_blend_screen_colab.py`
- `09_futures_hard_route_audit_colab.py`
- `10_futures_stack_component_audit_colab.py`
- `12_train_f_own_multichannel_regime_screen_colab.py`
- `13_build_train_f_own_multichannel_regime_submission_colab.py`
- `15_build_train_f_level_gate_submission_colab.py`
- `16_train_r_direct_route_blend_screen_colab.py`
- `17_train_r_recent_weighted_direct_screen_colab.py`
- `19_oof_build_f_base_2022_colab.py`
- `20_oof_f_champion_residual_screen_colab.py`
- `21_oof_build_f_history_oof_colab.py`
- `22_oof_f_level_shape_transfer_audit_colab.py`
- `23_oof_f_prior_level_gate_screen_colab.py`

## 실행 원칙

- 제출용 ZIP과 데이터 파일은 폴더에 넣지 않는다.
- 추론 시 test 행 간 집계, rolling, 순위, 분포 보정은 사용하지 않는다.
- 후보는 OOF 결과와 서버 제출 결과를 분리해 판정한다.
