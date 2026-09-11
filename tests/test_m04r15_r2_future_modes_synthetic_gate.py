from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import verify_m04r15_r2_future_modes_synthetic_gate as verifier


RESULT = ROOT / "config/data/analogues/m04r15/r2-future-modes-synthetic-gate-v1/RESULT.json"


def _write(path: Path, value: dict) -> None:
    state = {key: item for key, item in value.items()
             if key not in {"result_digest", "created_at"}}
    value["result_digest"] = verifier._stable(state)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def test_independent_verifier_imports_no_producer_module() -> None:
    source = (ROOT / "experiments/m04r/verify_m04r15_r2_future_modes_synthetic_gate.py").read_text()
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert imported.isdisjoint(verifier.PRODUCER_MODULES)


def test_scalar_oracle_finds_exact_three_families() -> None:
    result = verifier._oracle_three_families()
    assert result["medoid_episode_ids"] == ["episode-01", "episode-04", "episode-07"]
    assert result["labels"] == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert result["mean_silhouette"] > 0.98


def test_real_synthetic_result_is_independently_verified() -> None:
    result = verifier.verify(ROOT)
    assert result["passed"] is True
    assert result["verified_case_count"] == 5
    assert result["producer_modules_imported"] is False
    assert result["real_future_path_store_opened"] is False
    assert result["real_future_path_mode_computation_authorized"] is False


def test_verifier_rejects_self_consistent_case_tamper(tmp_path: Path) -> None:
    value = deepcopy(json.loads(RESULT.read_text()))
    value["cases"]["three_separated_families"]["labels"][0] = 2
    path = tmp_path / "result.json"
    _write(path, value)
    with pytest.raises(verifier.SyntheticVerificationError, match="oracle"):
        verifier.verify(ROOT, path)


def test_verifier_rejects_self_consistent_claim_tamper(tmp_path: Path) -> None:
    value = deepcopy(json.loads(RESULT.read_text()))
    value["real_future_path_store_opened"] = True
    path = tmp_path / "result.json"
    _write(path, value)
    with pytest.raises(verifier.SyntheticVerificationError, match="state"):
        verifier.verify(ROOT, path)
