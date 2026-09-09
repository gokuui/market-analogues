from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import m04r14_t14_10_wf03d_exclusion_repair_poc as subject


def test_two_frozen_queries_cover_every_repair_method() -> None:
    assert set().union(*(set(value) for value in subject.EXPECTED_AFFECTED.values())) == {
        "composite", "price_only", "deterministic_random",
        "recent_return_volatility",
    }
    assert len(subject.QUERY_IDS) == 2
    assert {
        "experiments/m04r/m04r14_t14_10_wf03_baseline_store_full.py",
        "experiments/m04r/m04r14_t14_10_wf03b_dtw_component_ladder.py",
    }.issubset(subject.RUNTIME_FILES)


def test_affected_method_projection_uses_frozen_order() -> None:
    audit = {"methods": {
        method: {"affected_query_ids": (["q"] if method in {
            "composite", "deterministic_random",
        } else [])}
        for method in (
            "composite", "price_only", "deterministic_random",
            "recent_return_volatility",
        )
    }}
    assert subject._affected_for_query(audit, "q") == (
        "composite", "deterministic_random",
    )


def test_proof_emits_exact_top21_drop_result() -> None:
    rows = [{"symbol": value} for value in ["S0", "Q", *[f"S{i}" for i in range(1, 20)]]]
    selected, proof = subject._proof(rows, "Q")
    assert len(rows) == subject.SUPERSET_K
    assert len(selected) == subject.TOP_K
    assert all(value["symbol"] != "Q" for value in selected)
    assert proof["proof_kind"] == "exact_top_k_plus_one_drop_single_excluded_symbol"
