"""평가 서버에서 자동 실행되는 투구 제구 성공 확률 추론 진입점."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parent
DATA_DIRECTORY = ROOT / "data"
MODEL_PATH = ROOT / "model" / "model.joblib"
OUTPUT_PATH = ROOT / "output" / "submission.csv"


def main() -> None:
    test_path, sample_path = discover_inference_files(DATA_DIRECTORY)
    test_columns, test_rows = read_csv(test_path)
    artifact = joblib.load(MODEL_PATH)
    model, features, input_format, prediction_column = unpack_artifact(
        artifact, test_columns
    )
    probabilities = predict(model, test_rows, features, input_format)
    write_submission(sample_path, probabilities, prediction_column)


def discover_inference_files(data_directory: Path) -> tuple[Path, Path]:
    files = sorted(data_directory.rglob("*.csv"))
    samples = [
        path
        for path in files
        if "submission" in path.stem.lower() or "sample" in path.stem.lower()
    ]
    tests = [
        path
        for path in files
        if path not in samples and "control_success" not in read_header(path)
    ]
    if len(samples) != 1 or len(tests) != 1:
        raise RuntimeError(
            "data/에서 test CSV와 sample submission CSV를 각각 하나씩 "
            f"찾아야 합니다. test={tests}, sample={samples}"
        )
    return tests[0], samples[0]


def unpack_artifact(artifact, test_columns: list[str]):
    if isinstance(artifact, dict):
        if "constant_probability" in artifact:
            model = float(artifact["constant_probability"])
        else:
            model = artifact["model"]
        features = list(artifact.get("feature_columns", test_columns))
        input_format = artifact.get("input_format", "matrix")
        prediction_column = artifact.get(
            "prediction_column", "control_success"
        )
    else:
        model = artifact
        features = test_columns
        input_format = "matrix"
        prediction_column = "control_success"
    missing = sorted(set(features) - set(test_columns))
    if missing:
        raise ValueError(f"테스트 데이터에 모델 입력 컬럼이 없습니다: {missing}")
    return model, features, input_format, prediction_column


def predict(model, rows, features, input_format: str) -> list[float]:
    if input_format == "constant":
        return [min(1.0, max(0.0, float(model)))] * len(rows)
    if input_format == "rows":
        inputs = rows
    elif input_format == "matrix":
        inputs = [[convert(row[column]) for column in features] for row in rows]
    else:
        raise ValueError(f"지원하지 않는 input_format입니다: {input_format}")

    raw = model.predict_proba(inputs)
    if hasattr(raw, "ndim") and raw.ndim == 2:
        raw = raw[:, 1]
    probabilities = [float(value) for value in raw]
    if len(probabilities) != len(rows):
        raise ValueError("예측 개수와 테스트 행 개수가 다릅니다.")
    if any(not math.isfinite(value) for value in probabilities):
        raise ValueError("예측 결과에 유한하지 않은 값이 있습니다.")
    return [min(1.0, max(0.0, value)) for value in probabilities]


def write_submission(
    sample_path: Path, probabilities: list[float], prediction_column: str
) -> None:
    columns, rows = read_csv(sample_path)
    if len(rows) != len(probabilities):
        raise ValueError("샘플 제출과 예측 결과의 행 개수가 다릅니다.")
    if prediction_column not in columns:
        non_id_columns = [column for column in columns if "id" not in column.lower()]
        if len(non_id_columns) != 1:
            raise ValueError(
                f"예측 컬럼 {prediction_column!r}을 샘플 제출에서 찾지 못했습니다."
            )
        prediction_column = non_id_columns[0]
    for row, probability in zip(rows, probabilities, strict=True):
        row[prediction_column] = f"{probability:.10f}"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_header(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        try:
            return next(csv.reader(file))
        except StopIteration as exc:
            raise ValueError(f"빈 CSV 파일입니다: {path}") from exc


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV 헤더가 없습니다: {path}")
        return list(reader.fieldnames), list(reader)


def convert(value: str):
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


if __name__ == "__main__":
    main()
