from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import verify_m04r15_r2_stability_synthetic_gate as verifier


RESULT = ROOT / "config/data/analogues/m04r15/r2-stability-synthetic-gate-v1/RESULT.json"


def _write(path: Path, value: dict) -> None:
    state = {key: item for key, item in value.items() if key not in {"result_digest", "created_at"}}
    value["result_digest"] = verifier._stable(state)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def test_stability_verifier_imports_no_producer_module() -> None:
    tree = ast.parse((ROOT / "experiments/m04r/verify_m04r15_r2_stability_synthetic_gate.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import): imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module: imported.add(node.module)
    assert imported.isdisjoint(verifier.PRODUCER_MODULES)


def test_independent_stability_oracle_has_expected_boundaries() -> None:
    stable = verifier._case([-1.01, -1, -.99, -.01, 0, .01, .99, 1, 1.01],
        [f"20{20 + i // 4}-Q{i % 4 + 1}" for i in range(9)], "stable")
    assert stable["status"] == "stable_multiple_modes"
    assert stable["selected_k"] == 3
    confounded = verifier._case([-1.01, -1, -.99, .99, 1, 1.01],
        ["2020-Q1"] * 3 + ["2020-Q2"] * 3, "date-confounded")
    assert confounded["status"] == "one_mode_fallback"


def test_real_stability_result_is_independently_reconstructed() -> None:
    result = verifier.verify(ROOT)
    assert result["passed"] is True
    assert result["verified_case_count"] == 4
    assert result["producer_modules_imported"] is False
    assert result["bounded_consumed_data_poc_authorized"] is True
    assert result["real_future_path_store_opened"] is False


def test_stability_verifier_rejects_self_consistent_mode_tamper(tmp_path: Path) -> None:
    value = deepcopy(json.loads(RESULT.read_text()))
    value["cases"]["stable_three_modes"]["selected_k"] = 2
    path = tmp_path / "result.json"; _write(path, value)
    with pytest.raises(verifier.StabilityVerificationError, match="reconstruction"):
        verifier.verify(ROOT, path)


def test_stability_verifier_rejects_self_consistent_boundary_tamper(tmp_path: Path) -> None:
    value = deepcopy(json.loads(RESULT.read_text()))
    value["real_full_build_authorized"] = True
    path = tmp_path / "result.json"; _write(path, value)
    with pytest.raises(verifier.StabilityVerificationError, match="boundary"):
        verifier.verify(ROOT, path)
