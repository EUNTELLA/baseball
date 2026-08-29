"""Futures 재구성 1단계 실행 진입점."""
from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "0823" / "10_futures_stack_component_audit_colab.py"
    runpy.run_path(str(target), run_name="__main__")
