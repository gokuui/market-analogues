from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.m04r import verify_m04r14_performance_qualification as verifier


def test_terminal_digest_boundary() -> None:
    value = {"passed": True, "created_at": "now", "result_digest": "old"}
    changed = copy.deepcopy(value); changed["created_at"] = "later"
    assert verifier._terminal_digest(value) == verifier._terminal_digest(changed)
    changed["passed"] = False
    assert verifier._terminal_digest(value) != verifier._terminal_digest(changed)


def test_counterfeit_resident_observation_is_rejected() -> None:
    value = {"ready": True, "mode": "validate-existing", "content_digest": "c",
        "seal_digest": "s", "ready_digest": "r", "fresh_validation_observation": {
            "content_digest": "c", "seal_digest": "s", "ready_digest": "r",
            "validation_seconds": 1.0, "observed_at": "now",
            "observation_digest": "counterfeit"}}
    with pytest.raises(verifier.VerificationError, match="reconstruction"):
        verifier._resident(value, "a" * 64)


def test_publish_is_create_only(tmp_path: Path) -> None:
    state = {"schema_version": verifier.SCHEMA, "result_digest": "a" * 64}
    root = tmp_path / "verification"
    verifier._publish(root, state)
    with pytest.raises(verifier.VerificationError, match="exists"):
        verifier._publish(root, state)
