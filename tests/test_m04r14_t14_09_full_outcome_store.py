from __future__ import annotations

import inspect
from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import m04r14_t14_09_full_outcome_store as full
from experiments.m04r import verify_m04r14_t14_09_full_outcome_store as verifier


@pytest.fixture(scope="module")
def authentic_inventory():
    links, registry = full._full_links(ROOT)
    requests = full._requests(ROOT, links, registry)
    return links, registry, requests


def test_complete_inventory_is_exact_and_deduplicated_without_opening_futures(
    authentic_inventory,
) -> None:
    links, registry, requests = authentic_inventory
    assert len(registry["cases_data"]) == 3270
    assert len(links) == 65400
    assert len({(row["query_episode_id"], row["match_rank"]) for row in links}) == 65400
    assert len(requests) == 56378
    assert len({row["symbol"] for row in requests}) == 7325
    assert len(links) - len(requests) == 9022
    assert all(len(row["expected_source_fingerprint"]) == 64 for row in requests)
    assert registry["real_forward_outcomes_accessed"] is False


def test_twelve_full_partitions_are_stable_balanced_complete_and_symbol_disjoint(
    authentic_inventory,
) -> None:
    requests = authentic_inventory[2]
    groups = full._groups(requests)
    assert [len(group) for group in groups] == [
        4624, 4515, 4804, 4816, 5308, 4924,
        4492, 4591, 4481, 4616, 4462, 4745,
    ]
    assert groups == full._groups(list(reversed(requests)))
    flattened = [row["episode_id"] for group in groups for row in group]
    assert len(flattened) == len(set(flattened)) == 56378
    assigned: dict[str, int] = {}
    for index, group in enumerate(groups):
        for row in group:
            assert assigned.setdefault(row["symbol"], index) == index


def test_full_frame_casts_preserve_nullable_integer_schema() -> None:
    outcomes = pd.DataFrame({
        "episode_id": ["b", "a"], "horizon_sessions": [20, 5],
        "available_sessions": [20, 5], "complete": [True, False],
        "time_to_mfe": [1, None], "time_to_mae": [None, 2],
        "barrier_touch_offset": [None, None],
    })
    result = full._cast_outcomes(outcomes)
    assert list(result.episode_id) == ["a", "b"]
    assert str(result.time_to_mfe.dtype) == "Int64"
    assert str(result.barrier_touch_offset.dtype) == "Int64"
    assert str(result.complete.dtype) == "bool"


def test_chunked_semantic_digest_is_order_and_nullable_dtype_stable() -> None:
    frame = pd.DataFrame({
        "episode_id": ["b", "a"], "horizon_sessions": [20, 5],
        "barrier_touch_offset": pd.Series([pd.NA, 3], dtype="Int64"),
    })
    reversed_frame = frame.iloc[::-1].reset_index(drop=True)
    expected = full._frame_digest(frame, ("episode_id", "horizon_sessions"))
    assert full._frame_digest(
        reversed_frame, ("episode_id", "horizon_sessions"),
    ) == expected
    assert verifier._frame_digest(
        reversed_frame, ("episode_id", "horizon_sessions"),
    ) == expected
    records = verifier._records(frame, ("episode_id", "horizon_sessions"))
    assert verifier._record_digest(records) == expected


def test_partition_cache_start_is_idempotent_and_rejects_unexpected_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(full, "CACHE", Path("partition-cache"))
    prereg = {"preregistration_digest": "frozen-digest"}
    first = full._start_or_resume_cache(tmp_path, prereg, "h1")
    assert full._start_or_resume_cache(tmp_path, prereg, "h1") == first
    (first / "unexpected").mkdir()
    with pytest.raises(full.FullOutcomeError, match="unexpected"):
        full._start_or_resume_cache(tmp_path, prereg, "h1")


def test_independent_full_verifier_has_no_production_formula_import() -> None:
    source = inspect.getsource(verifier)
    assert "market_analogues.causal_outcomes" not in source
    assert "reference_prepared_episode" in source
    assert "ProcessPoolExecutor" in source
    assert full.OUTPUT != full.CACHE != full.VERIFICATION


@pytest.mark.parametrize(
    "completion,query,complete,expected",
    [
        ("2020-01-10", "2020-01-10", True, {"eligible": True, "reason": "eligible"}),
        ("2020-01-11", "2020-01-10", True, {"eligible": False, "reason": "outcome_not_yet_observable"}),
        (None, "2020-01-10", False, {"eligible": False, "reason": "incomplete_horizon"}),
    ],
)
def test_full_verifier_embargo_boundaries(completion, query, complete, expected) -> None:
    assert verifier._eligibility(completion, query, complete) == expected
