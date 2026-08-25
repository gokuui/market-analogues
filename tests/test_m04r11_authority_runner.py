from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from market_analogues.m04r_certified_search_verification import _certificate_digest


def _module():
    path = Path(__file__).parents[1] / "experiments" / "m04r" / "m04r11_build_authorities.py"
    spec = importlib.util.spec_from_file_location("m04r11_build_authorities", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verifier_module():
    path = Path(__file__).parents[1] / "experiments" / "m04r" / "verify_m04r11_authorities.py"
    spec = importlib.util.spec_from_file_location("verify_m04r11_authorities", path)
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
        "rounds": [{
            "frontier_rows": 16_384, "exact_rows": 20,
            "next_lower_bound": .6, "constrained_threshold": .5,
            "selected_rows": 20, "certified": True,
            "proposal_digest": "prefix",
        }],
    }
    certificate["result_digest"] = _certificate_digest({
        "certificate": certificate, "matches": matches,
    })
    contract = {
        "contract_digest": "authority-contract", "registry_digest": "registry",
        "generation_id": "generation",
        "search_contract": {"certified_search_contract_digest": "certified-contract"},
        "certified_execution_contract": {"digest": "certified-contract"},
        "controls": {"maximum_frontier_rows": 32_768},
        "frontier_overflow_policy": module._overflow_policy(32_768),
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
        "primary_proposal_result_digest": "proposal",
        "proposal_result_digest": "proposal",
        "frontier_attempts": [{
            "maximum_frontier_rows": 32_768,
            "proposal_result_digest": "proposal", "exact_evaluated": 20,
            "stop_threshold_hex": (.5).hex(),
            "next_lower_bound_hex": (.6).hex(), "status": "certified",
        }],
        "frontier_overflow_recovery": False,
        "frontier_limit_rows": 32_768,
    }
    payload["gates"]["frontier_execution_policy"] = module._frontier_execution_valid(
        payload, contract,
    )
    payload["gate_passed"] = all(payload["gates"].values())
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


def test_frontier_execution_policy_rejects_skipped_or_rebound_attempts() -> None:
    module = _module()
    _, _, _, contract, payload = _fixture(module)
    assert module._frontier_execution_valid(payload, contract)
    payload["certificate"]["eligible_candidates"] = 100_000
    payload["frontier_attempts"] = [{
        "maximum_frontier_rows": 65_536,
        "proposal_result_digest": "recovery", "exact_evaluated": 20,
        "stop_threshold_hex": (.5).hex(),
        "next_lower_bound_hex": (.6).hex(), "status": "certified",
    }]
    payload["frontier_overflow_recovery"] = True
    payload["frontier_limit_rows"] = 65_536
    assert not module._frontier_execution_valid(payload, contract)
    payload["frontier_attempts"].insert(0, {
        "maximum_frontier_rows": 32_768,
        "proposal_result_digest": "proposal", "exact_evaluated": 32_768,
        "stop_threshold_hex": (.7).hex(),
        "next_lower_bound_hex": (.6).hex(), "status": "overflow",
    })
    payload["certificate"]["rounds"][0]["frontier_rows"] = 65_536
    assert module._frontier_execution_valid(payload, contract)
    payload["frontier_attempts"][0]["status"] = "certified"
    assert not module._frontier_execution_valid(payload, contract)


def test_overflow_recovery_doubles_and_preserves_exact_search(monkeypatch) -> None:
    module = _module()
    primary = SimpleNamespace(result_digest="primary", eligible_rows=100_000)
    recovered = SimpleNamespace(result_digest="recovered", eligible_rows=100_000)
    calls = []

    def search(*args, **kwargs):
        calls.append(kwargs["maximum_frontier_rows"])
        if kwargs["maximum_frontier_rows"] == 32_768:
            raise module.CertifiedFrontierOverflow(
                frontier_rows=32_768, eligible_candidates=100_000,
                exact_evaluated=32_768, stop_threshold=.7,
                next_lower_bound=.6,
            )
        certificate = SimpleNamespace(
            exact_evaluated=40_000, stop_threshold=.5, next_lower_bound=.6,
        )
        return SimpleNamespace(name="exact-result", certificate=certificate)

    def scan(*args, **kwargs):
        assert kwargs["route_quotas"] == {"composite": 65_537}
        return SimpleNamespace(reports=[recovered], elapsed_seconds=3.5)

    monkeypatch.setattr(module, "certified_packed_search", search)
    monkeypatch.setattr(module, "scan_packed_bound_proposals_many", scan)
    controls = {
        "initial_frontier_rows": 16_384, "maximum_frontier_rows": 32_768,
        "seed_rows": 512, "block_rows": 4_096,
        "exact_workers_per_process": 1,
    }
    result, attempts, seconds, proposal, measurements = (
        module._certified_with_overflow_recovery(
        object(), object(), object(), Path("/tmp/full"), "generation", controls,
        object(), primary,
        )
    )
    assert result.name == "exact-result"
    assert proposal is recovered
    assert seconds == 3.5
    assert calls == [32_768, 65_536]
    assert [row["status"] for row in measurements] == ["overflow", "certified"]
    assert attempts == [
        {"maximum_frontier_rows": 32_768,
         "proposal_result_digest": "primary", "exact_evaluated": 32_768,
         "stop_threshold_hex": (.7).hex(), "next_lower_bound_hex": (.6).hex(),
         "status": "overflow"},
        {"maximum_frontier_rows": 65_536,
         "proposal_result_digest": "recovered", "exact_evaluated": 40_000,
         "stop_threshold_hex": (.5).hex(), "next_lower_bound_hex": (.6).hex(),
         "status": "certified"},
    ]


def test_overflow_at_eligible_exhaustion_fails_closed(monkeypatch) -> None:
    module = _module()
    proposal = SimpleNamespace(result_digest="primary", eligible_rows=32_768)

    def search(*args, **kwargs):
        raise module.CertifiedFrontierOverflow(
            frontier_rows=32_768, eligible_candidates=32_768,
            exact_evaluated=32_768, stop_threshold=.7,
            next_lower_bound=None,
        )

    monkeypatch.setattr(module, "certified_packed_search", search)
    controls = {
        "initial_frontier_rows": 16_384, "maximum_frontier_rows": 32_768,
        "seed_rows": 512, "block_rows": 4_096,
        "exact_workers_per_process": 1,
    }
    try:
        module._certified_with_overflow_recovery(
            object(), object(), object(), Path("/tmp/full"), "generation",
            controls, object(), proposal,
        )
    except RuntimeError as exc:
        assert "eligible exhaustion" in str(exc)
    else:
        raise AssertionError("eligible-exhaustion overflow must fail closed")


def test_independent_verifier_agrees_on_manifest_and_frontier_policy() -> None:
    runner = _module()
    verifier = _verifier_module()
    _, _, _, contract, payload = _fixture(runner)
    assert runner._implementation_manifest() == verifier._implementation_manifest()
    assert verifier._frontier_execution_failure(payload, contract) is None
    payload["frontier_attempts"][0]["exact_evaluated"] = 19
    assert verifier._frontier_execution_failure(payload, contract) is not None
