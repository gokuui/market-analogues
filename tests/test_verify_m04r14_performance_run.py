from __future__ import annotations

import copy
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.m04r import verify_m04r14_performance_run as verifier


def test_measurement_digest_boundaries() -> None:
    value = {"semantic": 1, "proposal_seconds": 2.0, "created_at": "now",
             "result_digest": "old", "checkpoint_integrity_digest": "old"}
    changed = copy.deepcopy(value)
    changed["proposal_seconds"] = 3.0
    assert verifier._case_result_digest(value) == verifier._case_result_digest(changed)
    assert verifier._case_integrity_digest(value) != verifier._case_integrity_digest(changed)
    changed = copy.deepcopy(value)
    changed["semantic"] = 2
    assert verifier._case_result_digest(value) != verifier._case_result_digest(changed)


def test_finite_measurement_boundary() -> None:
    assert verifier._finite(0)
    assert verifier._finite(1.5)
    assert not verifier._finite(True)
    assert not verifier._finite(-1)
    assert not verifier._finite(float("inf"))
