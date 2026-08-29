from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.m04r import verify_m04r14_untouched_registry as verifier


def test_forbidden_outcome_keys_are_recursive() -> None:
    assert not verifier._contains_forbidden({"cases": [{"cutoff": "2020"}]})
    assert verifier._contains_forbidden({"cases": [{"forward_return": 1.0}]})


def test_counterfeit_observation_tree_is_rejected_by_symbol_normalization() -> None:
    assert verifier._symbols([" abc ", "ABC", "def"]) == ["ABC", "DEF"]


def test_publish_is_create_only(tmp_path: Path) -> None:
    root = tmp_path / "verification"; state = {"result_digest": "a" * 64}
    verifier._publish(root, state)
    with pytest.raises(verifier.VerificationError, match="exists"):
        verifier._publish(root, state)


def test_records_normalize_numpy_and_timestamps() -> None:
    frame = pd.DataFrame({"timestamp": [pd.Timestamp("2020-01-01")], "value": [1]})
    assert verifier._records(frame) == [{"timestamp": "2020-01-01T00:00:00", "value": 1}]
