from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.m04r import verify_m04r14_throughput_poc as verifier


def test_case_digest_layers_exclude_only_declared_measurements() -> None:
    value = {"semantic": 1, "proposal_seconds": 2.0, "created_at": "now",
             "result_digest": "old", "checkpoint_integrity_digest": "old"}
    changed = copy.deepcopy(value); changed["proposal_seconds"] = 3.0
    assert verifier._case_result_digest(value) == verifier._case_result_digest(changed)
    assert verifier._case_integrity_digest(value) != verifier._case_integrity_digest(changed)
    changed = copy.deepcopy(value); changed["semantic"] = 2
    assert verifier._case_result_digest(value) != verifier._case_result_digest(changed)


def test_finite_resource_boundary() -> None:
    assert verifier._finite(0)
    assert verifier._finite(1.5)
    assert not verifier._finite(True)
    assert not verifier._finite(-1)
    assert not verifier._finite(float("inf"))


def test_publish_is_create_only(tmp_path: Path) -> None:
    state = {"schema_version": verifier.SCHEMA, "result_digest": "a" * 64}
    root = tmp_path / "verification"
    verifier._publish(root, state)
    with pytest.raises(verifier.VerificationError, match="exists"):
        verifier._publish(root, state)
