"""현재 챔피언 위 strict model-only R 혼합을 0.20 이후로 확장 검증한다."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "0823" / "02_strict_anchor_champion_validation_colab.py"
EXTENDED_BLENDS = (0.20, 0.25, 0.30, 0.40)


def load_module():
    spec = importlib.util.spec_from_file_location("strict_champion_validation", SOURCE)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-source", type=Path, required=True)
    parser.add_argument("--component-dir", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    module = load_module()
    module.BLENDS = EXTENDED_BLENDS
    module.main(
        args.strict_source.resolve(), args.component_dir.resolve(), args.train.resolve(),
        args.output.resolve(), args.task_type,
    )
