from __future__ import annotations

import inspect
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import m04r14_t14_09_evidence_card_oracle as oracle
from experiments.m04r import m04r14_t14_09_evidence_card_synthetic_gate as gate


def test_synthetic_evidence_gate_passes_without_real_query_aggregation() -> None:
    result = gate.execute(ROOT)
    assert result["passed"] is True
    assert result["family_count"] == 6
    assert result["exact_production_reference_equality"] is True
    assert result["row_order_invariant"] is True
    assert result["outcome_mutation_preserved_ranks"] is True
    assert result["parquet_semantic_roundtrip"] is True
    assert result["query_level_real_outcome_aggregation_opened"] is False


def test_evidence_oracle_remains_formula_independent() -> None:
    source = inspect.getsource(oracle)
    assert "market_analogues.evidence_cards" not in source
    assert "reference_card" in source
