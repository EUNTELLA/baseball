"""다음 피처 축을 정하기 위해 공식 데이터 파일과 식별자 연결 가능성을 점검한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main(data_dir: Path, output: Path) -> None:
    files = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        row = {"relative_path": str(path.relative_to(data_dir)), "bytes": path.stat().st_size,
               "suffix": path.suffix.lower()}
        if path.suffix.lower() in {".csv", ".gz", ".parquet"}:
            try:
                sample = pd.read_parquet(path).head(5) if path.suffix.lower() == ".parquet" else pd.read_csv(path, nrows=5, low_memory=False)
                row["columns"] = list(sample.columns)
                row["identifier_columns"] = [column for column in sample.columns
                                             if "id" in column.lower() or "pitcher" in column.lower()]
                row["trackman_columns"] = [column for column in sample.columns
                                           if any(token in column.lower() for token in
                                                  ("trackman", "spin", "break", "speed", "extension", "release"))]
            except Exception as error:
                row["read_error"] = f"{type(error).__name__}: {error}"
        files.append(row)
    train_candidates = [row for row in files if Path(row["relative_path"]).name.lower() == "train.csv"]
    trackman_candidates = [row for row in files if "trackman" in row["relative_path"].lower()
                           or row.get("trackman_columns")]
    report = {"experiment": "official data asset inventory",
              "data_dir": str(data_dir), "files": files,
              "train_candidates": train_candidates, "trackman_candidates": trackman_candidates,
              "decision_hint": ("audit_official_trackman_linkage" if trackman_candidates
                                  else "do_not_pursue_trackman_axis")}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train_candidates": train_candidates,
                      "trackman_candidates": trackman_candidates,
                      "decision_hint": report["decision_hint"]}, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.data_dir.resolve(), args.output.resolve())
