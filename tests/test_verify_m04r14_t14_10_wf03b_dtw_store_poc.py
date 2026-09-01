from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.m04r.verify_m04r14_t14_10_wf03b_dtw_store_poc import _slice


def test_independent_slice_handles_present_and_absent_ids() -> None:
    records = np.asarray([(1, 10), (1, 20), (3, 30)], dtype=[
        ("symbol_id", "<u4"), ("cutoff_ns", "<i8"),
    ])
    assert _slice(records, 1)["cutoff_ns"].tolist() == [10, 20]
    assert len(_slice(records, 2)) == 0
