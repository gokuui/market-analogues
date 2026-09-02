from types import SimpleNamespace

import numpy as np

from experiments.m04r.verify_m04r14_t14_10_wf03b_dtw_component_ladder import (
    _eligibility,
)
from market_analogues.packed_bound_store import OVERFLOW_DTYPE


def test_independent_eligibility_excludes_future_query_and_overlap() -> None:
    rows = np.zeros(5, dtype=OVERFLOW_DTYPE)
    rows["episode_id"] = [
        np.void(bytes.fromhex(f"{value:024x}")) for value in range(5)
    ]
    rows["cutoff_ns"] = [5, 20, 5, 5, 5]
    rows["symbol_id"] = [1, 1, 7, 7, 1]
    rows["quality_tier"] = [1, 1, 1, 1, 3]
    query = SimpleNamespace(
        latest_eligible_ns=10, episode_id=f"{2:024x}", symbol="QUERY",
        query_start_ns=4,
    )
    assert _eligibility(rows, query, 7).tolist() == [True, False, False, False, False]
