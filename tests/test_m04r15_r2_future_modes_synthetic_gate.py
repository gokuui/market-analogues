from __future__ import annotations

import ast
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import verify_m04r15_r2_future_modes_synthetic_gate as verifier


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
