from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import verify_m04r14_t14_10_wf03d_exclusion_audit as subject


def test_identity_digest_is_order_invariant_and_independent() -> None:
    first = subject._identity_digest({"02": "B", "01": "A"})
    assert first == subject._identity_digest({"01": "A", "02": "B"})
    assert first != subject._identity_digest({"01": "A", "02": "C"})


def test_method_rows_requires_all_four_sources() -> None:
    rows = subject._method_rows(
        {"retrieval": {"matches": []}}, {"matches": []},
        {"random_neighbors": [], "rank_neighbors": []},
    )
    assert [name for name, _values in rows] == [
        "composite", "price_only", "deterministic_random",
        "recent_return_volatility",
    ]
    with pytest.raises(subject.ExclusionAuditVerificationError, match="layout"):
        subject._method_rows({}, {"matches": []}, {
            "random_neighbors": [], "rank_neighbors": [],
        })
