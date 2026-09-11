from __future__ import annotations

import ast
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import verify_m04r15_r2_stability_synthetic_gate as verifier


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
