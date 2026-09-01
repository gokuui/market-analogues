from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import m04r14_t14_09_synthetic_gate as gate


def test_synthetic_gate_covers_frozen_edge_matrix() -> None:
    assert set(gate.KINDS) == {
        "no_touch", "favorable", "adverse", "ambiguous", "incomplete",
        "missing_benchmark_origin", "missing_benchmark_endpoint",
        "missing_stock_session", "insufficient_atr", "scaled",
    }


def test_synthetic_gate_passes_all_independent_and_parallel_checks() -> None:
    result = gate.execute(ROOT)
    assert result["passed"] is True
    assert all(result["gates"].values())
    assert len(result["case_results"]) == len(gate.KINDS)
    assert all(
        row["production_digest"] == row["oracle_digest"]
        for row in result["case_results"]
    )
    assert result["real_forward_outcomes_accessed"] is False
