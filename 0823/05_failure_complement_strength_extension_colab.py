"""서버 검증된 R 실패여집합 채널의 0.20 이후 강도를 확장 검증한다."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "0822" / "02_failure_complement_champion_validation_colab.py"
EXTENDED_BLENDS = (0.20, 0.25, 0.30, 0.40)


def load_module():
    spec = importlib.util.spec_from_file_location("failure_complement_validation", SOURCE)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-dir", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    module = load_module()
    module.BLENDS = EXTENDED_BLENDS
    module.main(
        args.component_dir.resolve(), args.train.resolve(), args.output.resolve(), args.task_type
    )
