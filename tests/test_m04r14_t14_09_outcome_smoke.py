from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import m04r14_t14_09_outcome_smoke as smoke
from experiments.m04r import verify_m04r14_t14_09_outcome_smoke as verifier


def test_authentic_smoke_links_are_exact_fixed_audit_sample_without_outcomes() -> None:
    links, registry = smoke._extract_links(ROOT)
    requests = smoke._bind_source_fingerprints(
        ROOT, smoke._requests(links), registry,
    )
    assert len(links) == 240
    assert len({(row["query_episode_id"], row["match_rank"]) for row in links}) == 240
    assert len(requests) <= 240
    assert all(len(row["expected_source_fingerprint"]) == 64 for row in requests)
    assert registry["source_lock"]["real_forward_outcomes_accessed"] is False


def test_episode_request_deduplication_rejects_conflicting_metadata() -> None:
    link = {
        "matched_episode_id": "episode", "matched_symbol": "ABC",
        "matched_cutoff": "2020-01-01",
    }
    assert smoke._requests([link, deepcopy(link)]) == [{
        "episode_id": "episode", "symbol": "ABC", "cutoff": "2020-01-01",
    }]
    conflict = {**link, "matched_symbol": "XYZ"}
    with pytest.raises(smoke.OutcomeSmokeError, match="conflicting"):
        smoke._requests([link, conflict])


def test_twelve_process_partition_is_stable_complete_and_disjoint() -> None:
    rows = [{
        "episode_id": f"episode-{index}", "symbol": f"S{index % 17}",
        "cutoff": "2020-01-01", "expected_source_fingerprint": f"fp-{index}",
    } for index in range(100)]
    first = smoke._groups(rows)
    second = smoke._groups(list(reversed(rows)))
    assert first == second
    flattened = [row["episode_id"] for group in first for row in group]
    assert len(first) <= 12
    assert len(flattened) == len(set(flattened)) == 100
    symbol_group = {
        row["symbol"]: index for index, group in enumerate(first) for row in group
    }
    assert all(symbol_group[row["symbol"]] == index for index, group in enumerate(first) for row in group)


def test_semantic_frame_digest_is_row_order_and_nullable_dtype_stable() -> None:
    frame = pd.DataFrame({
        "episode_id": ["b", "a"], "horizon_sessions": [20, 5],
        "barrier_touch_offset": pd.Series([pd.NA, 3], dtype="Int64"),
    })
    reordered = frame.iloc[::-1].reset_index(drop=True)
    assert smoke._frame_digest(frame, ("episode_id", "horizon_sessions")) \
        == smoke._frame_digest(reordered, ("episode_id", "horizon_sessions"))


def test_independent_verifier_does_not_import_production_outcome_formula() -> None:
    source = inspect.getsource(verifier)
    assert "market_analogues.causal_outcomes" not in source
    assert "reference_episode" in source
    assert smoke.OUTPUT != smoke.retrieval.OUTPUT


@pytest.mark.parametrize(
    "completion,query,complete,expected",
    [
        ("2020-01-10", "2020-01-10", True, {"eligible": True, "reason": "eligible"}),
        ("2020-01-11", "2020-01-10", True, {"eligible": False, "reason": "outcome_not_yet_observable"}),
        (None, "2020-01-10", False, {"eligible": False, "reason": "incomplete_horizon"}),
    ],
)
def test_independent_embargo_boundaries(completion, query, complete, expected) -> None:
    assert verifier._eligibility(completion, query, complete) == expected
