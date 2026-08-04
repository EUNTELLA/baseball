"""코드 제출 환경의 실제 CSV 파일을 안전하게 식별한다."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CompetitionFiles:
    train: Path | None
    test: Path
    sample_submission: Path | None


def discover_competition_files(
    data_directory: str | Path,
    *,
    target_column: str = "control_success",
    require_train: bool = False,
) -> CompetitionFiles:
    """Target과 파일명을 이용해 train/test/sample submission을 찾는다."""
    root = Path(data_directory)
    csv_files = sorted(root.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"{root} 아래에 CSV 파일이 없습니다.")

    headers = {path: _read_header(path) for path in csv_files}
    sample_candidates = [
        path
        for path in csv_files
        if "submission" in path.stem.lower() or "sample" in path.stem.lower()
    ]
    train_candidates = [
        path
        for path, header in headers.items()
        if target_column in header and path not in sample_candidates
    ]
    test_candidates = [
        path
        for path, header in headers.items()
        if target_column not in header and path not in sample_candidates
    ]

    train = _choose(train_candidates, "train", required=require_train)
    test = _choose(test_candidates, "test", required=True)
    sample = _choose(sample_candidates, "sample_submission", required=False)
    return CompetitionFiles(train=train, test=test, sample_submission=sample)


def read_csv_rows(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV 헤더가 없습니다: {path}")
        return list(reader.fieldnames), list(reader)


def _read_header(path: Path) -> tuple[str, ...]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        try:
            return tuple(next(reader))
        except StopIteration as exc:
            raise ValueError(f"빈 CSV 파일입니다: {path}") from exc


def _choose(
    candidates: list[Path], kind: str, *, required: bool
) -> Path | None:
    if not candidates:
        if required:
            raise FileNotFoundError(f"{kind} CSV 파일을 찾지 못했습니다.")
        return None
    if len(candidates) == 1:
        return candidates[0]

    exact = [path for path in candidates if kind in path.stem.lower()]
    if len(exact) == 1:
        return exact[0]
    names = ", ".join(str(path) for path in candidates)
    raise ValueError(f"{kind} CSV 후보를 하나로 결정할 수 없습니다: {names}")
