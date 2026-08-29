"""Build a large-lever R residual rebuild submission package.

The residual model is trained from official Train OOF rows and is applied only
to R rows at inference.  Each test row is predicted independently: the injected
runtime uses the row's own features plus frozen model assets.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "train_r" / "03_r_segment_error_audit_colab.py"
ID = "row_id"
TARGET = "control_success"
TRAIN_YEAR = 2024
SEEDS = (17, 42, 777, 2024)
CAT_COLS = [
    "count", "hand", "same_hand", "base_state", "top_bottom",
    "pitcher_team_id", "batter_team_id", "p_band", "gap_band",
    "inning_band", "runner_band", "outs",
]


def load_audit_module():
    spec = importlib.util.spec_from_file_location("r_segment_audit", AUDIT_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(AUDIT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def number(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def logit(values):
    values = np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(values / (1 - values))


def make_features(rows: pd.DataFrame, anchor: np.ndarray) -> pd.DataFrame:
    result = pd.DataFrame(index=rows.index)
    balls = number(rows, "balls_before", -1).astype(int)
    strikes = number(rows, "strikes_before", -1).astype(int)
    outs = number(rows, "outs_before", 0).astype(int)
    runners = number(rows, "num_runners_on", 0).astype(int)
    inning = number(rows, "inning", 0).astype(int)
    pitcher = number(rows, "asof_pitcher_success_rate", 0.5)
    batter = number(rows, "asof_batter_success_rate", 0.5)
    pitcher_n = number(rows, "asof_pitcher_n", 0).clip(lower=0)
    batter_n = number(rows, "asof_batter_n", 0).clip(lower=0)
    recent1 = number(rows, "asof_pitcher_prev1_game_success_rate", pitcher)
    recent3 = number(rows, "asof_pitcher_prev3_game_success_rate", pitcher)
    recent5 = number(rows, "asof_pitcher_prev5_game_success_rate", pitcher)
    p = np.clip(np.asarray(anchor, dtype=float), 1e-6, 1 - 1e-6)
    gap = pitcher - batter

    result["anchor"] = p
    result["anchor_logit"] = logit(p)
    result["anchor_center"] = p - 0.5
    result["gap_raw"] = gap
    result["gap_smooth_100"] = (
        (pitcher_n * pitcher + 100 * 0.5) / (pitcher_n + 100)
        - (batter_n * batter + 100 * 0.5) / (batter_n + 100)
    )
    result["gap_smooth_500"] = (
        (pitcher_n * pitcher + 500 * 0.5) / (pitcher_n + 500)
        - (batter_n * batter + 500 * 0.5) / (batter_n + 500)
    )
    result["recent1_gap"] = recent1 - pitcher
    result["recent3_gap"] = recent3 - pitcher
    result["recent5_gap"] = recent5 - pitcher
    result["recent1_vs_batter"] = recent1 - batter
    result["recent3_vs_batter"] = recent3 - batter
    result["recent5_vs_batter"] = recent5 - batter
    result["pitcher_rate"] = pitcher
    result["batter_rate"] = batter
    result["log_pitcher_n"] = np.log1p(pitcher_n)
    result["log_batter_n"] = np.log1p(batter_n)
    result["balls"] = balls
    result["strikes"] = strikes
    result["count_advantage"] = strikes - balls
    result["outs_num"] = outs
    result["runners_num"] = runners
    result["inning"] = inning
    result["score_diff"] = number(rows, "score_diff_pitcher_team", 0)
    result["li"] = number(rows, "li", 1.0)
    result["middle_rate"] = number(rows, "asof_pitcher_middle_rate", 0.0)
    result["reverse_rate"] = number(rows, "asof_pitcher_reverse_rate", 0.0)
    result["count"] = balls.astype(str) + "-" + strikes.astype(str)
    result["hand"] = rows["pitcher_hand"].astype(str) + "_" + rows["batter_hand"].astype(str)
    result["same_hand"] = rows["pitcher_hand"].astype(str).eq(rows["batter_hand"].astype(str)).map(
        {True: "same", False: "opposite"}
    )
    result["base_state"] = rows["base_state"].astype(str)
    result["top_bottom"] = rows["top_bottom"].astype(str)
    result["pitcher_team_id"] = rows["pitcher_team_id"].astype(str)
    result["batter_team_id"] = rows["batter_team_id"].astype(str)
    result["p_band"] = pd.cut(
        pd.Series(p, index=rows.index),
        [0.0, 0.35, 0.42, 0.47, 0.52, 0.58, 0.65, 1.0],
        labels=False,
        include_lowest=True,
    ).astype(str)
    result["gap_band"] = pd.cut(
        gap,
        [-2, -0.10, -0.04, 0.0, 0.04, 0.10, 2],
        labels=False,
        include_lowest=True,
    ).astype(str)
    result["inning_band"] = np.select(
        [inning <= 3, inning <= 6, inning <= 9],
        ["early", "mid", "late"],
        default="extra",
    )
    result["runner_band"] = runners.clip(lower=0, upper=3).astype(str)
    result["outs"] = outs.clip(lower=0, upper=2).astype(str)
    for column in CAT_COLS:
        result[column] = result[column].astype(str)
    return result


def load_training_frame(train_path: Path, oof_dir: Path) -> pd.DataFrame:
    audit = load_audit_module()
    raw = pd.read_csv(train_path, encoding="utf-8-sig", low_memory=False)
    oof = audit.load_oof(oof_dir, TRAIN_YEAR)
    rows = raw.loc[raw["season"].astype(int).eq(TRAIN_YEAR)].copy().reset_index(drop=True)
    if len(rows) != len(oof):
        raise ValueError(f"{TRAIN_YEAR} row count mismatch: train={len(rows)} oof={len(oof)}")
    if not np.array_equal(rows[ID].astype(str).to_numpy(), oof[ID].astype(str).to_numpy()):
        raise ValueError(f"{TRAIN_YEAR} row_id alignment failed")
    frame = pd.concat(
        [rows.reset_index(drop=True), oof.drop(columns=[ID, "game_type", "pitcher_id"])],
        axis=1,
    )
    return frame.loc[frame["game_type"].astype(str).eq("R")].copy()


def train_models(train_frame: pd.DataFrame, model_dir: Path, task_type: str) -> list[dict]:
    from catboost import CatBoostRegressor, Pool

    x = make_features(train_frame, train_frame["p_champion"].to_numpy(float))
    y = train_frame["target"].to_numpy(float) - train_frame["p_champion"].to_numpy(float)
    cat_indices = [x.columns.get_loc(column) for column in CAT_COLS]
    pool = Pool(x, y, cat_features=cat_indices)
    model_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for seed in SEEDS:
        model = CatBoostRegressor(
            loss_function="RMSE",
            iterations=900,
            learning_rate=0.025,
            depth=4,
            l2_leaf_reg=500.0,
            random_seed=seed,
            task_type=task_type,
            allow_writing_files=False,
            verbose=False,
        )
        start = time.perf_counter()
        model.fit(pool)
        elapsed = float(time.perf_counter() - start)
        path = model_dir / f"r_residual_rebuild_{seed}.cbm"
        model.save_model(path)
        summaries.append({"seed": seed, "seconds": elapsed, "model": path.name})
        print(f"R residual rebuild seed={seed} sec={elapsed:.1f}", flush=True)
    return summaries


def extract_package(source_zip: Path, destination: Path) -> None:
    if not zipfile.is_zipfile(source_zip):
        raise ValueError(f"not a valid zip: {source_zip}")
    with zipfile.ZipFile(source_zip) as archive:
        error = archive.testzip()
        if error is not None:
            raise ValueError(f"source zip integrity failed: {error}")
        archive.extractall(destination)
    if not (destination / "script.py").exists():
        scripts = list(destination.rglob("script.py"))
        if len(scripts) != 1:
            raise ValueError(f"root script.py missing, script count={len(scripts)}")
        package = scripts[0].parent
        temporary = destination.parent / f"{destination.name}_flat"
        shutil.move(str(package), temporary)
        shutil.rmtree(destination)
        shutil.move(str(temporary), destination)
    if not (destination / "requirements.txt").exists():
        raise FileNotFoundError(destination / "requirements.txt")


def injection_block(probability_name: str) -> str:
    return f'''    # R residual rebuild: official Train OOF residual model, row-local inference only
    _rrb_meta_path = ROOT / "model" / "r_residual_rebuild_meta.json"
    if _rrb_meta_path.exists():
        import json as _rrb_json
        from catboost import CatBoostRegressor as _RRBCatBoostRegressor, Pool as _RRBPool
        with _rrb_meta_path.open("r", encoding="utf-8") as _rrb_stream:
            _rrb_meta = _rrb_json.load(_rrb_stream)

        def _rrb_num(_frame, _column, _default=0.0):
            if _column not in _frame.columns:
                return pd.Series(_default, index=_frame.index, dtype=float)
            return pd.to_numeric(_frame[_column], errors="coerce").fillna(_default)

        def _rrb_logit(_values):
            _values = np.clip(np.asarray(_values, dtype=float), 1e-6, 1 - 1e-6)
            return np.log(_values / (1 - _values))

        def _rrb_features(_rows, _anchor):
            _out = pd.DataFrame(index=_rows.index)
            _balls = _rrb_num(_rows, "balls_before", -1).astype(int)
            _strikes = _rrb_num(_rows, "strikes_before", -1).astype(int)
            _outs = _rrb_num(_rows, "outs_before", 0).astype(int)
            _runners = _rrb_num(_rows, "num_runners_on", 0).astype(int)
            _inning = _rrb_num(_rows, "inning", 0).astype(int)
            _pitcher = _rrb_num(_rows, "asof_pitcher_success_rate", 0.5)
            _batter = _rrb_num(_rows, "asof_batter_success_rate", 0.5)
            _pitcher_n = _rrb_num(_rows, "asof_pitcher_n", 0).clip(lower=0)
            _batter_n = _rrb_num(_rows, "asof_batter_n", 0).clip(lower=0)
            _recent1 = _rrb_num(_rows, "asof_pitcher_prev1_game_success_rate", _pitcher)
            _recent3 = _rrb_num(_rows, "asof_pitcher_prev3_game_success_rate", _pitcher)
            _recent5 = _rrb_num(_rows, "asof_pitcher_prev5_game_success_rate", _pitcher)
            _p = np.clip(np.asarray(_anchor, dtype=float), 1e-6, 1 - 1e-6)
            _gap = _pitcher - _batter
            _out["anchor"] = _p
            _out["anchor_logit"] = _rrb_logit(_p)
            _out["anchor_center"] = _p - 0.5
            _out["gap_raw"] = _gap
            _out["gap_smooth_100"] = (
                (_pitcher_n * _pitcher + 100 * 0.5) / (_pitcher_n + 100)
                - (_batter_n * _batter + 100 * 0.5) / (_batter_n + 100)
            )
            _out["gap_smooth_500"] = (
                (_pitcher_n * _pitcher + 500 * 0.5) / (_pitcher_n + 500)
                - (_batter_n * _batter + 500 * 0.5) / (_batter_n + 500)
            )
            _out["recent1_gap"] = _recent1 - _pitcher
            _out["recent3_gap"] = _recent3 - _pitcher
            _out["recent5_gap"] = _recent5 - _pitcher
            _out["recent1_vs_batter"] = _recent1 - _batter
            _out["recent3_vs_batter"] = _recent3 - _batter
            _out["recent5_vs_batter"] = _recent5 - _batter
            _out["pitcher_rate"] = _pitcher
            _out["batter_rate"] = _batter
            _out["log_pitcher_n"] = np.log1p(_pitcher_n)
            _out["log_batter_n"] = np.log1p(_batter_n)
            _out["balls"] = _balls
            _out["strikes"] = _strikes
            _out["count_advantage"] = _strikes - _balls
            _out["outs_num"] = _outs
            _out["runners_num"] = _runners
            _out["inning"] = _inning
            _out["score_diff"] = _rrb_num(_rows, "score_diff_pitcher_team", 0)
            _out["li"] = _rrb_num(_rows, "li", 1.0)
            _out["middle_rate"] = _rrb_num(_rows, "asof_pitcher_middle_rate", 0.0)
            _out["reverse_rate"] = _rrb_num(_rows, "asof_pitcher_reverse_rate", 0.0)
            _out["count"] = _balls.astype(str) + "-" + _strikes.astype(str)
            _out["hand"] = _rows["pitcher_hand"].astype(str) + "_" + _rows["batter_hand"].astype(str)
            _out["same_hand"] = _rows["pitcher_hand"].astype(str).eq(_rows["batter_hand"].astype(str)).map({{True: "same", False: "opposite"}})
            _out["base_state"] = _rows["base_state"].astype(str)
            _out["top_bottom"] = _rows["top_bottom"].astype(str)
            _out["pitcher_team_id"] = _rows["pitcher_team_id"].astype(str)
            _out["batter_team_id"] = _rows["batter_team_id"].astype(str)
            _out["p_band"] = pd.cut(
                pd.Series(_p, index=_rows.index),
                [0.0, 0.35, 0.42, 0.47, 0.52, 0.58, 0.65, 1.0],
                labels=False,
                include_lowest=True,
            ).astype(str)
            _out["gap_band"] = pd.cut(
                _gap,
                [-2, -0.10, -0.04, 0.0, 0.04, 0.10, 2],
                labels=False,
                include_lowest=True,
            ).astype(str)
            _out["inning_band"] = np.select(
                [_inning <= 3, _inning <= 6, _inning <= 9],
                ["early", "mid", "late"],
                default="extra",
            )
            _out["runner_band"] = _runners.clip(lower=0, upper=3).astype(str)
            _out["outs"] = _outs.clip(lower=0, upper=2).astype(str)
            for _column in _rrb_meta["cat_cols"]:
                _out[_column] = _out[_column].astype(str)
            return _out[_rrb_meta["feature_cols"]]

        _rrb_prob = {probability_name}
        _rrb_mask = test["game_type"].astype(str).eq("R").to_numpy()
        if _rrb_mask.any():
            _rrb_x = _rrb_features(test, _rrb_prob)
            _rrb_cat = [_rrb_x.columns.get_loc(_column) for _column in _rrb_meta["cat_cols"]]
            _rrb_pool = _RRBPool(_rrb_x, cat_features=_rrb_cat)
            _rrb_members = []
            for _rrb_model_name in _rrb_meta["models"]:
                _rrb_model = _RRBCatBoostRegressor()
                _rrb_model.load_model(str(ROOT / "model" / _rrb_model_name))
                _rrb_members.append(_rrb_model.predict(_rrb_pool))
            _rrb_residual = np.mean(_rrb_members, axis=0)
            if _rrb_meta["alpha"] > 0:
                _rrb_pitcher_n = _rrb_num(test, "asof_pitcher_n", 0).clip(lower=0).to_numpy(float)
                _rrb_batter_n = _rrb_num(test, "asof_batter_n", 0).clip(lower=0).to_numpy(float)
                _rrb_reliability = np.sqrt(
                    (_rrb_pitcher_n / (_rrb_pitcher_n + _rrb_meta["alpha"]))
                    * (_rrb_batter_n / (_rrb_batter_n + _rrb_meta["alpha"]))
                )
            else:
                _rrb_reliability = np.ones(len(test), dtype=float)
            _rrb_delta = np.clip(
                _rrb_meta["scale"] * _rrb_reliability * _rrb_residual,
                -_rrb_meta["cap"],
                _rrb_meta["cap"],
            )
            _rrb_prob[_rrb_mask] = np.clip(_rrb_prob[_rrb_mask] + _rrb_delta[_rrb_mask], 1e-6, 1 - 1e-6)
            {probability_name} = _rrb_prob
'''


def patch_script(script_path: Path) -> str:
    script = script_path.read_text(encoding="utf-8")
    normal_marker = "    pred_map = dict(zip(test[ID], p))\n"
    wrapper_marker = "    destination = ROOT / \"output\"\n"
    if normal_marker in script:
        script_path.write_text(script.replace(normal_marker, injection_block("p") + normal_marker), encoding="utf-8")
        return "probability_array_p"
    if wrapper_marker in script:
        script_path.write_text(script.replace(wrapper_marker, injection_block("result") + wrapper_marker), encoding="utf-8")
        return "wrapper_result"
    raise RuntimeError("prediction insertion marker not found")


def run_package(package: Path, test: pd.DataFrame, sample: pd.DataFrame, tag: str):
    run_dir = package.parent / f"verify_{tag}"
    shutil.copytree(package, run_dir)
    data = run_dir / "data"
    data.mkdir(exist_ok=True)
    test.to_csv(data / "test.csv", index=False, encoding="utf-8")
    sample.to_csv(data / "sample_submission.csv", index=False, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "script.py"],
        cwd=run_dir,
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
    )
    if completed.returncode:
        raise RuntimeError(f"{tag} inference failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
    result = pd.read_csv(run_dir / "output" / "submission.csv")
    return result, completed.stdout.strip()


def prediction_column(frame: pd.DataFrame) -> str:
    columns = [column for column in frame.columns if column != ID]
    if len(columns) != 1:
        raise ValueError(f"prediction columns={columns}")
    return columns[0]


def zip_directory(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            if not path.is_file() or "data" in relative.parts or "output" in relative.parts:
                continue
            archive.write(path, relative.as_posix())


def verify_singletons(package: Path, test: pd.DataFrame, sample: pd.DataFrame,
                      full_prediction: pd.DataFrame) -> float:
    column = prediction_column(full_prediction)
    indices = list(dict.fromkeys([0, 1, 2, len(test) // 2, len(test) - 2, len(test) - 1]))
    indices = [index for index in indices if 0 <= index < len(test)]
    differences = []
    for index in indices[:8]:
        one_test = test.iloc[[index]].copy()
        one_sample = sample.loc[sample[ID].astype(str).eq(str(one_test[ID].iloc[0]))].copy()
        if one_sample.empty:
            one_sample = sample.iloc[[0]].copy()
            one_sample[ID] = one_test[ID].iloc[0]
        one_prediction, _ = run_package(package, one_test, one_sample, f"single_{index}")
        full_value = full_prediction.loc[
            full_prediction[ID].astype(str).eq(str(one_test[ID].iloc[0])), column
        ].iloc[0]
        differences.append(abs(float(one_prediction[column].iloc[0]) - float(full_value)))
    return max(differences, default=0.0)


def main(source_zip: Path, train_path: Path, oof_dir: Path, test_path: Path,
         sample_path: Path, output_zip: Path, report_path: Path, scale: float,
         alpha: float, cap: float, task_type: str) -> None:
    if scale <= 0 or alpha < 0 or cap <= 0:
        raise ValueError("scale/cap은 양수, alpha는 0 이상이어야 합니다")
    train_frame = load_training_frame(train_path, oof_dir)
    test = pd.read_csv(test_path, encoding="utf-8-sig", low_memory=False)
    sample = pd.read_csv(sample_path, encoding="utf-8-sig", low_memory=False)
    source_sha = digest(source_zip)

    with tempfile.TemporaryDirectory(prefix="r_residual_rebuild_") as temporary:
        root = Path(temporary)
        source = root / "source"
        candidate = root / "candidate"
        extract_package(source_zip, source)
        shutil.copytree(source, candidate)
        model_dir = candidate / "model"
        model_summaries = train_models(train_frame, model_dir, task_type)
        feature_cols = make_features(train_frame.iloc[:2], train_frame["p_champion"].iloc[:2].to_numpy(float)).columns.tolist()
        meta = {
            "training_year": TRAIN_YEAR,
            "training_r_rows": int(len(train_frame)),
            "target": "control_success minus p_champion",
            "scale": float(scale),
            "alpha": float(alpha),
            "cap": float(cap),
            "seeds": list(SEEDS),
            "models": [row["model"] for row in model_summaries],
            "feature_cols": feature_cols,
            "cat_cols": CAT_COLS,
            "official_train_only": True,
            "test_aggregate_used": False,
        }
        (model_dir / "r_residual_rebuild_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        insertion_mode = patch_script(candidate / "script.py")
        source_prediction, source_stdout = run_package(source, test, sample, "source")
        candidate_prediction, candidate_stdout = run_package(candidate, test, sample, "candidate")
        source_column = prediction_column(source_prediction)
        candidate_column = prediction_column(candidate_prediction)
        comparison = test[[ID, "game_type"]].merge(
            source_prediction.rename(columns={source_column: "source"}), on=ID
        ).merge(
            candidate_prediction.rename(columns={candidate_column: "candidate"}), on=ID
        )
        delta = comparison["candidate"] - comparison["source"]
        r_delta = delta[comparison["game_type"].astype(str).eq("R")]
        f_delta = delta[comparison["game_type"].astype(str).eq("F")]
        if len(f_delta) and float(f_delta.abs().max()) > 1e-12:
            raise ValueError(f"F rows changed: {float(f_delta.abs().max())}")
        if candidate_prediction[candidate_column].isna().any():
            raise ValueError("candidate has missing predictions")
        if not candidate_prediction[candidate_column].between(0, 1).all():
            raise ValueError("candidate probabilities are out of range")
        maximum_singleton_difference = verify_singletons(candidate, test, sample, candidate_prediction)
        if maximum_singleton_difference > 1e-12:
            raise ValueError(f"row independence failed: {maximum_singleton_difference}")

        shutil.rmtree(candidate / "data", ignore_errors=True)
        shutil.rmtree(candidate / "output", ignore_errors=True)
        zip_directory(candidate, output_zip)

    with zipfile.ZipFile(output_zip) as archive:
        zip_error = archive.testzip()
        members = archive.namelist()
    if zip_error is not None:
        raise RuntimeError(f"candidate zip integrity failed: {zip_error}")
    report = {
        "experiment": "R residual rebuild submission",
        "official_train_only": True,
        "test_aggregate_used": False,
        "source_zip": str(source_zip),
        "source_sha256": source_sha,
        "output_zip": str(output_zip),
        "output_sha256": digest(output_zip),
        "training_year": TRAIN_YEAR,
        "training_r_rows": int(len(train_frame)),
        "scale": float(scale),
        "alpha": float(alpha),
        "cap": float(cap),
        "model_summaries": model_summaries,
        "insertion_mode": insertion_mode,
        "members": len(members),
        "script_count": sum(Path(name).name == "script.py" for name in members),
        "root_script": "script.py" in members,
        "nested_zip_count": sum(name.lower().endswith(".zip") for name in members),
        "zip_test_error": zip_error,
        "sample_rows": int(len(candidate_prediction)),
        "sample_missing": int(candidate_prediction[candidate_column].isna().sum()),
        "r_mean_delta": float(r_delta.mean()) if len(r_delta) else None,
        "r_max_absolute_delta": float(r_delta.abs().max()) if len(r_delta) else None,
        "f_max_absolute_delta": float(f_delta.abs().max()) if len(f_delta) else None,
        "maximum_singleton_difference": maximum_singleton_difference,
        "source_stdout": source_stdout,
        "candidate_stdout": candidate_stdout,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--oof-dir", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=0.30)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--cap", type=float, default=0.04)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    main(
        args.source_zip.resolve(),
        args.train.resolve(),
        args.oof_dir.resolve(),
        args.test.resolve(),
        args.sample.resolve(),
        args.output_zip.resolve(),
        args.report.resolve(),
        args.scale,
        args.alpha,
        args.cap,
        args.task_type,
    )
