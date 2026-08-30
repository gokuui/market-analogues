from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.m04r import m04r14_untouched_authority as authority


def test_authority_controls_freeze_separate_32k_lane() -> None:
    value = {"search_contract": {"controls": {"block_rows": 4096,
        "deferred_alignments": True, "exact_workers_per_process": 1,
        "initial_frontier_rows": 16384, "maximum_frontier_rows": 32768,
        "numba_threads_per_process": 1, "processes": 8,
        "requested_positions": True, "seed_rows": 512,
        "sorted_joined_iqr_merge": True, "vector_lower_bounds": True}}}
    assert authority.controls(value)["maximum_frontier_rows"] == 32768
    value["search_contract"]["controls"]["maximum_frontier_rows"] = 16384
    with pytest.raises(authority.AuthorityError, match="controls"):
        authority.controls(value)


def test_authority_root_is_distinct_from_candidate() -> None:
    assert authority.OUTPUT != authority.contract.CANDIDATE_RELATIVE
    assert "authority" in authority.OUTPUT.name
