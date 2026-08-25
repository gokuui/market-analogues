from __future__ import annotations

import importlib.util
from pathlib import Path

from market_analogues.m04r_certified_search_verification import _certificate_digest


def _module():
    path = Path(__file__).parents[1] / "experiments" / "m04r" / "m04r11_build_authorities.py"
    spec = importlib.util.spec_from_file_location("m04r11_build_authorities", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(module):
    case = {
        "case_id": "nasdaq-QUERY-current-252", "episode_id": "a" * 24,
        "symbol": "QUERY", "cutoff": "2026-03-30T00:00:00",
        "query_start": "2025-04-01T00:00:00",
        "latest_eligible_cutoff": "2025-01-01T00:00:00",
        "stock_prefix": {"digest": "stock"},
        "benchmark_prefix": {"digest": "benchmark"},
    }
    matches = [{
        "episode_id": f"{index:024x}", "symbol": f"S{index:02d}",
        "cutoff": "2024-01-01T00:00:00", "total_distance": index / 100,
        "component_distances": {"price": index / 100},
        "alignment": [[0, 0]], "quality_tier": "A",
    } for index in range(20)]
    certificate = {
        "schema_version": "m04r-certified-packed-search-v5",
        "contract_digest": "certified-contract", "generation_id": "generation",
        "query_episode_id": case["episode_id"], "input_digest": "input",
        "eligible_candidates": 100, "exact_evaluated": 20, "safely_pruned": 80,
        "stopped_early": True, "stop_threshold": .5, "next_lower_bound": .6,
        "maximum_quantized_bound_excess": 0.0,
        "materialization_groups": 1, "sparse_symbols": 0, "batch_symbols": 1,
        "rounds": [],
    }
    certificate["result_digest"] = _certificate_digest({
        "certificate": certificate, "matches": matches,
    })
    contract = {
        "contract_digest": "authority-contract", "registry_digest": "registry",
        "generation_id": "generation",
        "search_contract": {"certified_search_contract_digest": "certified-contract"},
    }
    gates = module._case_gates(case, matches, certificate)
    payload = {
        "schema_version": module.CASE_SCHEMA, "status": "completed",
        "contract_digest": contract["contract_digest"],
        "registry_digest": contract["registry_digest"],
        "generation_id": contract["generation_id"],
        "registry_case_id": case["case_id"],
        "query_episode_id": case["episode_id"], "query_start": case["query_start"],
        "latest_eligible_cutoff": case["latest_eligible_cutoff"],
        "query_stock_prefix": case["stock_prefix"],
        "query_benchmark_prefix": case["benchmark_prefix"],
        "matches": matches, "certificate": certificate,
        "gates": gates, "gate_passed": all(gates.values()),
        "proposal_seconds": 1.0, "exact_seconds": 2.0, "peak_rss_mb": 3.0,
        "real_forward_outcomes_accessed": False, "created_at": "now",
    }
    payload["result_digest"] = module.case_result_digest(payload)
    payload["checkpoint_integrity_digest"] = module.checkpoint_integrity_digest(payload)
    return case, matches, certificate, contract, payload


def test_authority_checkpoint_binds_semantics_and_measured_integrity() -> None:
    module = _module()
    case, _, _, contract, payload = _fixture(module)
    assert payload["gate_passed"]
    assert module._case_checkpoint_valid(payload, case, contract)
    payload["exact_seconds"] = 99.0
    assert not module._case_checkpoint_valid(payload, case, contract)


def test_authority_gates_reject_order_accounting_and_overlap() -> None:
    module = _module()
    case, matches, certificate, _, _ = _fixture(module)
    matches[0], matches[1] = matches[1], matches[0]
    certificate["safely_pruned"] = 79
    matches[2]["symbol"] = case["symbol"]
    matches[2]["cutoff"] = "2025-06-01T00:00:00"
    gates = module._case_gates(case, matches, certificate)
    assert not gates["stable_distance_id_order"]
    assert not gates["candidate_accounting"]
    assert not gates["same_symbol_overlap_excluded"]


def test_authority_grouping_is_deterministic_complete_and_balanced() -> None:
    module = _module()
    cases = [{
        "case_id": f"case-{index}", "episode_id": f"{index:024x}",
        "active_source_universe": 100 + index,
    } for index in range(60)]
    first = module._groups(cases, 8)
    second = module._groups(list(reversed(cases)), 8)
    assert first == second
    assert sorted(row["case_id"] for group in first for row in group) == sorted(
        row["case_id"] for row in cases
    )
    loads = [sum(row["active_source_universe"] for row in group) for group in first]
    assert max(loads) - min(loads) <= max(row["active_source_universe"] for row in cases)
