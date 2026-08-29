from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.m04r import verify_m04r14_exposed_authority_comparison as verifier


def _case() -> dict:
    return {
        "registry_case_id": "case", "query_episode_id": "query",
        "matches": [{"episode_id": "one"}], "query_stock_prefix": "stock",
        "query_benchmark_prefix": "benchmark", "gate_passed": True,
        "certificate": {key: f"value-{key}" for key in verifier.STABLE_CERTIFICATE_FIELDS},
    }


def test_comparison_checks_ordered_matches_and_stable_certificate() -> None:
    candidate = _case()
    authority = copy.deepcopy(candidate)
    assert all(verifier._gates(candidate, authority).values())
    authority["matches"] = [{"episode_id": "other"}]
    assert verifier._gates(candidate, authority)["ordered_matches_equal"] is False
    authority = copy.deepcopy(candidate)
    authority["certificate"]["stop_threshold"] = "changed"
    assert verifier._gates(candidate, authority)["stable_certificate_fields_equal"] is False


def test_traversal_only_certificate_fields_are_deliberately_ignored() -> None:
    candidate = _case()
    authority = copy.deepcopy(candidate)
    candidate["certificate"]["rounds"] = [{"rows": 1000}]
    authority["certificate"]["rounds"] = [{"rows": 16000}]
    assert all(verifier._gates(candidate, authority).values())


def test_publish_is_create_only(tmp_path: Path) -> None:
    state = {"schema_version": verifier.SCHEMA, "result_digest": "a" * 64}
    root = tmp_path / "verification"
    verifier._publish(root, state)
    with pytest.raises(verifier.VerificationError, match="exists"):
        verifier._publish(root, state)
