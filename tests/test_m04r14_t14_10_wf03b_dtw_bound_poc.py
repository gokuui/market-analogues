from pathlib import Path
from random import Random
import sys

from market_analogues.packed_bound_search import BoundProposal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.m04r.m04r14_t14_10_wf03b_dtw_bound_poc import (
    promotion_gate,
    retain_sample,
    selection_key,
)


def _row(index: int) -> BoundProposal:
    return BoundProposal(
        f"{index:024x}", f"S{index % 7}", index, "A", index / 1000,
        ("price",), False,
    )


def test_hash_sample_is_order_and_chunk_invariant() -> None:
    rows = [_row(index) for index in range(200)]
    expected = sorted(rows, key=lambda row: selection_key("query", row.episode_id))[:31]
    forward = []
    for first in range(0, len(rows), 13):
        forward = retain_sample(
            forward, rows[first:first + 13], query_id="query", sample_rows=31,
        )
    shuffled = rows.copy(); Random(19).shuffle(shuffled)
    reverse = retain_sample([], shuffled, query_id="query", sample_rows=31)
    assert forward == expected == reverse


def test_conflicting_duplicate_sample_fails_closed() -> None:
    import pytest
    from experiments.m04r.m04r14_t14_10_wf03b_dtw_bound_poc import DtwBoundPocError

    row = _row(1)
    conflicting = BoundProposal(
        row.episode_id, row.symbol, row.cutoff_ns, row.quality_tier,
        row.lower_bound + 1, row.routes, row.overflow_fallback,
    )
    with pytest.raises(DtwBoundPocError, match="conflicting"):
        retain_sample([row], [conflicting], query_id="query", sample_rows=2)


def test_automatic_promotion_gate_requires_safety_and_materiality() -> None:
    passing = [{
        "sample_rows": 100, "enhanced_pruned": 40,
        "enhanced_pruned_fraction": .40, "false_prunes": 0,
        "maximum_bound_excess": 0.0, "maximum_packed_native_excess": 0.0,
    }] * 3
    assert promotion_gate(passing)["auxiliary_store_authorized"] is True
    weak = [dict(row, enhanced_pruned=20, enhanced_pruned_fraction=.20)
            for row in passing]
    assert promotion_gate(weak)["auxiliary_store_authorized"] is False
    unsafe = [dict(row, false_prunes=1) for row in passing]
    assert promotion_gate(unsafe)["auxiliary_store_authorized"] is False
