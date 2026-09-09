from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import (  # noqa: E402
    m04r14_t14_10_wf03d_outcome_store as subject,
)
from experiments.m04r import (  # noqa: E402
    verify_m04r14_t14_10_wf03d_outcome_store as verifier,
)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    links = pd.DataFrame([
        {"query_id": "q1", "query_cutoff": "2020-02-01T00:00:00",
         "method": "composite", "rank": 1, "matched_episode_id": "e1"},
        {"query_id": "q2", "query_cutoff": "2020-01-15T00:00:00",
         "method": "price_only", "rank": 1, "matched_episode_id": "e2"},
    ])
    outcomes = pd.DataFrame([
        {"episode_id": "e1", "horizon_sessions": 5,
         "completion_timestamp": "2020-02-01T00:00:00", "complete": True,
         "status": "complete"},
        {"episode_id": "e2", "horizon_sessions": 5,
         "completion_timestamp": "2020-02-02T00:00:00", "complete": True,
         "status": "complete"},
        {"episode_id": "e1", "horizon_sessions": 20,
         "completion_timestamp": None, "complete": False,
         "status": "source_end_before_horizon"},
        {"episode_id": "e2", "horizon_sessions": 20,
         "completion_timestamp": None, "complete": False,
         "status": "suspension_or_missing_session"},
    ])
    return links, outcomes


def test_causal_eligibility_includes_equal_cutoff_only() -> None:
    links, outcomes = _frames()
    result = subject._eligibility_table(
        links, outcomes, expected_links=2, expected_requests=2, horizons=(5, 20),
    )
    by_key = {
        (row.query_id, row.horizon_sessions): (row.eligible, row.reason)
        for row in result.itertuples(index=False)
    }
    assert by_key[("q1", 5)] == (True, "eligible")
    assert by_key[("q2", 5)] == (False, "outcome_not_yet_observable")
    assert by_key[("q1", 20)] == (False, "incomplete_horizon")
    assert by_key[("q2", 20)] == (False, "incomplete_horizon")


def test_independent_eligibility_formula_matches_producer() -> None:
    links, outcomes = _frames()
    observed = subject._eligibility_table(
        links, outcomes, expected_links=2, expected_requests=2, horizons=(5, 20),
    )
    original_horizons = verifier.producer.HORIZONS
    verifier.producer.HORIZONS = (5, 20)
    try:
        expected = verifier._expected_eligibility(links, outcomes)
    finally:
        verifier.producer.HORIZONS = original_horizons
    assert observed.equals(expected)


def test_eligibility_rejects_missing_episode_outcome() -> None:
    links, outcomes = _frames()
    with pytest.raises(subject.WalkForwardOutcomeError):
        subject._eligibility_table(
            links, outcomes.iloc[:-1], expected_links=2,
            expected_requests=2, horizons=(5, 20),
        )


def test_symbol_partition_is_deterministic_and_co_located() -> None:
    symbols: dict[int, str] = {}
    candidate = 0
    while len(symbols) < subject.PARTITIONS:
        symbol = f"S{candidate}"
        partition = int(sha256(symbol.encode()).hexdigest(), 16) % subject.PARTITIONS
        symbols.setdefault(partition, symbol)
        candidate += 1
    names = [symbols[index] for index in range(subject.PARTITIONS)] * 2
    requests = [{
        "episode_id": f"{index:024x}", "symbol": symbol, "cutoff": "2020-01-01",
    } for index, symbol in enumerate(names)]
    groups = subject._groups(requests)
    assert groups == subject._groups(list(reversed(requests)))
    locations = {}
    for index, group in enumerate(groups):
        for row in group:
            locations.setdefault(row["symbol"], set()).add(index)
    assert all(len(indices) == 1 for indices in locations.values())


def test_partition_receipt_timing_is_not_semantic() -> None:
    state = {"schema_version": subject.PARTITION_SCHEMA, "passed": True}
    receipt = {
        **state, "elapsed_seconds": 1.0,
        "result_digest": subject.stable_hash(state), "created_at": "now",
    }
    assert subject._receipt_valid(receipt, timing=True)
    receipt["passed"] = False
    assert not subject._receipt_valid(receipt, timing=True)


@pytest.mark.parametrize("field", [
    "cutoff", "source_fingerprint", "contract_digest", "source_content_digest",
])
def test_reuse_requires_every_frozen_identity_field(field: str) -> None:
    request = {"cutoff": "2020-01-01", "expected_source_fingerprint": "source"}
    observed = {
        "cutoff": "2020-01-01", "source_fingerprint": "source",
        "contract_digest": "contract", "source_content_digest": "content",
    }
    assert subject._reusable_identity_matches(
        request, observed, "contract", "content",
    )
    assert verifier._reusable_identity_matches(
        request, observed, "contract", "content",
    )
    changed = deepcopy(observed)
    changed[field] = "mutated"
    assert not subject._reusable_identity_matches(
        request, changed, "contract", "content",
    )
    assert not verifier._reusable_identity_matches(
        request, changed, "contract", "content",
    )
