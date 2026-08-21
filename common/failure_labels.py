"""공식 Train의 다음 누적 상태로 학습용 실패유형 라벨을 재구성한다."""
from __future__ import annotations

import numpy as np
import pandas as pd


def recover_failure_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """각 투수의 다음 행이 바로 다음 투구일 때만 middle/reverse를 복원한다."""
    grouped = frame.groupby("pitcher_id", sort=False)
    count = pd.to_numeric(frame["asof_pitcher_n"], errors="coerce")
    next_count = pd.to_numeric(grouped["asof_pitcher_n"].shift(-1), errors="coerce")
    available = next_count.eq(count + 1)
    recovered: dict[str, np.ndarray] = {}
    for name, column in (
        ("middle", "asof_pitcher_middle_rate"),
        ("reverse", "asof_pitcher_reverse_rate"),
    ):
        rate = pd.to_numeric(frame[column], errors="coerce")
        next_rate = pd.to_numeric(grouped[column].shift(-1), errors="coerce")
        increment = next_count * next_rate - count * rate
        available &= increment.notna()
        recovered[name] = increment.gt(0.5).to_numpy()

    result = pd.DataFrame({"row_id": frame["row_id"], **recovered})
    result[["middle", "reverse"]] = result[["middle", "reverse"]].astype(float)
    result.loc[~available.to_numpy(), ["middle", "reverse"]] = np.nan
    return result
