from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.m04r import m04r14_performance_qualification as qualification


def test_terminal_digest_ignores_only_timestamp_and_digest() -> None:
    value = {"status": "complete", "passed": True, "created_at": "now", "result_digest": "old"}
    changed = {**value, "created_at": "later", "result_digest": "new"}
    assert qualification._terminal_digest(value) == qualification._terminal_digest(changed)
    changed["passed"] = False
    assert qualification._terminal_digest(value) != qualification._terminal_digest(changed)


def test_resident_binding_requires_matching_fresh_observation() -> None:
    value = {"ready": True, "mode": "validate-existing", "content_digest": "c",
        "seal_digest": "s", "ready_digest": "r", "fresh_validation_observation": {
            "content_digest": "c", "seal_digest": "s", "ready_digest": "r",
            "validation_seconds": 1.0, "observation_digest": "o"}}
    assert qualification._resident(value, json.dumps(value).encode())["validation_seconds"] == 1.0
    value["fresh_validation_observation"]["content_digest"] = "different"
    with pytest.raises(qualification.QualificationError, match="binding"):
        qualification._resident(value, json.dumps(value).encode())
