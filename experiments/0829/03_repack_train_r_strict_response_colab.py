"""Build strict R blend response candidates from two verified packages.

The alternate package is assumed to represent a known effective R strict
strength, usually 0.05.  This runner creates nearby effective strengths by
calling the existing region-specific runtime blend builder.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BLEND_BUILDER = ROOT / "0823" / "03_build_strict_r_blend_submission_colab.py"
DEFAULT_STRENGTHS = (0.0625, 0.075, 0.0875, 0.10)


def load_builder():
    spec = importlib.util.spec_from_file_location("strict_r_blend_builder", BLEND_BUILDER)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(BLEND_BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def strength_tag(value: float):
    return f"{int(round(value * 10000)):04d}"


def main(current_zip: Path, alternate_zip: Path, test: Path, sample: Path,
         output_dir: Path, report: Path, base_strength: float, strengths: list[float]):
    if base_strength <= 0:
        raise ValueError("base_strength must be positive")
    builder = load_builder()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for strength in strengths:
        blend = strength / base_strength
        if not 0 < blend <= 2:
            raise ValueError(f"effective strength {strength} requires invalid blend {blend}")
        tag = strength_tag(strength)
        output_zip = output_dir / f"submit_catboost_fgeneral6_rstrict{tag}.zip"
        output_report = output_dir / f"strict_r_response_{tag}.json"
        builder.main(
            current_zip.resolve(),
            alternate_zip.resolve(),
            test.resolve(),
            sample.resolve(),
            output_zip.resolve(),
            output_report.resolve(),
            float(blend),
            "R",
        )
        payload = json.loads(output_report.read_text(encoding="utf-8"))
        candidates.append({
            "effective_strength": float(strength),
            "blend": float(blend),
            "output_zip": str(output_zip),
            "output_sha256": payload["output_sha256"],
            "members": payload["members"],
            "zip_test_error": payload["zip_test_error"],
            "sample_verification": payload["sample_verification"],
        })
    result = {
        "experiment": "strict R response candidate repack",
        "official_train_only": True,
        "test_aggregate_used": False,
        "current_zip": str(current_zip),
        "alternate_zip": str(alternate_zip),
        "base_strength": base_strength,
        "candidates": candidates,
        "recommended_order": [str(row["output_zip"]) for row in candidates],
        "note": "Submit at most one or two candidates after comparing with the existing 0.075 score.",
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved: {report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-zip", type=Path, required=True)
    parser.add_argument("--alternate-zip", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--base-strength", type=float, default=0.05)
    parser.add_argument(
        "--strengths",
        type=float,
        nargs="+",
        default=list(DEFAULT_STRENGTHS),
    )
    args = parser.parse_args()
    main(args.current_zip, args.alternate_zip, args.test, args.sample,
         args.output_dir, args.report, args.base_strength, args.strengths)
