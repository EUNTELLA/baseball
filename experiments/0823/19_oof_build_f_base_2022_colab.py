"""F residual의 첫 학습 연도를 위해 2022 general6 OOF를 만든다."""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "train_f" / "02_general_route_reconstruction_colab.py"
SEEDS = (42, 7, 2024, 99, 1, 123)

def load():
    spec = importlib.util.spec_from_file_location("f_base_2022", SOURCE)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def main(train_path, component_dir, output, task_type):
    module = load(); features = module.load_features()
    frame = pd.read_csv(train_path, encoding="utf-8-sig", low_memory=False)
    target = frame.control_success.astype(int).to_numpy()
    season = frame.season.astype(int).to_numpy()
    prediction, shift, iterations, seconds = module.train_fold(
        frame, target, season, 2022, SEEDS, task_type, features, 128)
    rows = frame.loc[season == 2022].reset_index(drop=True)
    asset = module.load_component(component_dir, 2022)
    if not np.array_equal(rows.row_id.astype(str), asset["row_id"].astype(str)):
        raise ValueError("2022 row_id 불일치")
    calibrated = module.sigmoid(module.logit(prediction) + shift)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, row_id=rows.row_id.astype(str).to_numpy(),
        target=asset["target"].astype(np.float32), game_type=rows.game_type.astype(str).to_numpy(),
        pitcher_id=rows.pitcher_id.astype(str).to_numpy(), p_f_general6=calibrated.astype(np.float32))
    report = {"year": 2022, "seeds": list(SEEDS), "iterations": iterations,
              "seconds": seconds, "calibration_shift": float(shift), "output": str(output)}
    output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--train",type=Path,required=True)
    p.add_argument("--component-dir",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--task-type",choices=("CPU","GPU"),default="GPU"); a=p.parse_args()
    main(a.train.resolve(),a.component_dir.resolve(),a.output.resolve(),a.task_type)
