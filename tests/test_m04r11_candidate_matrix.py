from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

from market_analogues.types import stable_hash


def _module():
    path = Path(__file__).parents[1] / "experiments" / "m04r" / "m04r11_candidate_matrix.py"
    spec = importlib.util.spec_from_file_location("m04r11_candidate_matrix", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _comparator_module():
    path = Path(__file__).parents[1] / "experiments" / "m04r" / "compare_m04r11_candidate_matrix.py"
    spec = importlib.util.spec_from_file_location("compare_m04r11_candidate_matrix", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_case_digest_binds_semantics_but_not_measurements() -> None:
    module = _module()
    payload = {
        "query_episode_id": "a" * 24,
        "candidates": [{"episode_id": "b" * 24}],
        "recall_at_20": 1.0,
        "cold_task_seconds": 100.0,
        "warm_second_seconds": 50.0,
        "created_at": "first",
    }
    digest = module.case_digest(payload)
    payload["cold_task_seconds"] = 101.0
    payload["warm_second_seconds"] = 51.0
    payload["created_at"] = "second"
    assert module.case_digest(payload) == digest
    payload["recall_at_20"] = .95
    assert module.case_digest(payload) != digest


def test_hard_frozen_candidate_identity_matches_registry() -> None:
    module = _module()
    registry = json.loads((
        Path(__file__).parents[1] / "config" / "data" / "analogues" / "m04r10"
        / "nasdaq-untouched-authority-registry" / "query-registry.json"
    ).read_text())
    assert registry["registry_digest"] == module.FROZEN_REGISTRY_DIGEST
    assert registry["search_contract"]["packed_generation_id"] == module.FROZEN_GENERATION_ID
    assert registry["search_contract"]["proposal_contract_digest"] == module.FROZEN_PROPOSAL_CONTRACT_DIGEST
    assert registry["search_contract"]["fast_route_quotas"] == module.FROZEN_ROUTE_QUOTAS
    assert registry["search_contract"]["request"] == module.FROZEN_REQUEST
    assert stable_hash([case["episode_id"] for case in registry["cases_data"]]) == module.FROZEN_CASE_ORDER_DIGEST


def test_matrix_digest_binds_gate_but_not_runtime() -> None:
    module = _module()
    payload = {"passed": True, "gates": {"recall": True}, "elapsed_seconds": 1.0, "created_at": "x"}
    digest = module.matrix_digest(payload)
    payload["elapsed_seconds"] = 2.0
    assert module.matrix_digest(payload) == digest
    payload["gates"]["recall"] = False
    assert module.matrix_digest(payload) != digest


def test_failed_matrix_is_not_sealed_and_complete_pass_is_sealed(
    tmp_path: Path,
) -> None:
    module = _module()
    matrix = {
        "schema_version": module.MATRIX_SCHEMA,
        "registry_digest": "registry",
        "producer_contract_digest": "producer",
        "completed_cases": 60,
        "worker_failures": [],
        "gates": {
            "all_60_candidate_pools_sealed": True,
            "all_safety_determinism_and_resource_gates": False,
        },
        "passed": False,
    }
    matrix["result_digest"] = module.matrix_digest(matrix)
    assert not module._write_success_seal(
        tmp_path, matrix, registry_digest="registry",
        producer_contract_digest="producer",
    )
    assert not (tmp_path / "SEALED.json").exists()

    matrix["gates"]["all_safety_determinism_and_resource_gates"] = True
    matrix["passed"] = True
    matrix["result_digest"] = module.matrix_digest(matrix)
    assert module._write_success_seal(
        tmp_path, matrix, registry_digest="registry",
        producer_contract_digest="producer",
    )
    seal = json.loads((tmp_path / "SEALED.json").read_text())
    assert seal["candidate_pools_sealed"] is True
    assert seal["candidate_matrix_digest"] == matrix["result_digest"]
    assert seal["seal_digest"] == module._seal_digest(seal)


def test_comparator_rejects_producer_drift_before_truth_open(tmp_path: Path) -> None:
    module = _comparator_module()
    registry = json.loads((
        Path(__file__).parents[1] / "config" / "data" / "analogues" / "m04r10"
        / "nasdaq-untouched-authority-registry" / "query-registry.json"
    ).read_text())
    physical_manifest = tmp_path / "manifest.json"
    physical_manifest.write_text("{}\n")
    deterministic = {
        "schema_version": module.PRODUCER_CONTRACT_SCHEMA,
        "registry_digest": module.FROZEN_REGISTRY_DIGEST,
        "ordered_query_ids": [
            case["episode_id"] for case in registry["cases_data"]
        ],
        "ordered_query_ids_digest": module.FROZEN_CASE_ORDER_DIGEST,
        "generation_id": module.FROZEN_GENERATION_ID,
        "physical_manifest_path": str(physical_manifest.resolve()),
        "physical_manifest_sha256": module.file_fingerprint(physical_manifest),
        "physical_rows": 10,
        "proposal_contract_digest": module.FROZEN_PROPOSAL_CONTRACT_DIGEST,
        "route_quotas": module.FROZEN_ROUTE_QUOTAS,
        "request": module.FROZEN_REQUEST,
        "scan_protocol": module.PRODUCER_SCAN_PROTOCOL,
        "performance_limits": module.PRODUCER_PERFORMANCE_LIMITS,
        "prefix_policy": "worker recomputes and exactly matches frozen stock and benchmark causal prefixes",
        "implementation_manifest": module._expected_producer_implementation_manifest(),
        "real_forward_outcomes_accessed": False,
    }
    contract = {
        **deterministic, "contract_digest": stable_hash(deterministic),
    }
    assert module.validate_preopen_producer_contract(contract, registry) == ()

    for changed in (
        {"schema_version": "candidate-recall-producer-contract-v1"},
        {"scan_protocol": {**module.PRODUCER_SCAN_PROTOCOL, "outer_threads": 2}},
        {"implementation_manifest": {"files": {}, "digest": stable_hash({})}},
    ):
        invalid = {**deepcopy(contract), **changed}
        invalid["contract_digest"] = stable_hash({
            key: value for key, value in invalid.items()
            if key != "contract_digest"
        })
        assert module.validate_preopen_producer_contract(invalid, registry)


def test_checkpoint_is_bound_to_registry_case_and_three_repeats() -> None:
    module = _module()
    comparator = _comparator_module()
    payload, case, _, _ = _valid_comparator_fixture(comparator)
    assert module._checkpoint_valid(
        payload, case=case, registry_digest=comparator.FROZEN_REGISTRY_DIGEST,
        generation_id=comparator.FROZEN_GENERATION_ID,
        proposal_contract_digest=comparator.FROZEN_PROPOSAL_CONTRACT_DIGEST,
        producer_contract_digest="producer",
        route_quotas={"composite": 1}, physical_rows=10,
        expected_query_start_ns=100, expected_latest_eligible_ns=90,
        expected_representation_digest="representation",
    )
    payload["scan_invariants"][1] = {**payload["scan_invariants"][1], "generation_id": "changed"}
    payload["result_digest"] = module.case_digest(payload)
    payload["checkpoint_integrity_digest"] = module.checkpoint_integrity_digest(payload)
    assert not module._checkpoint_valid(
        payload, case=case, registry_digest=comparator.FROZEN_REGISTRY_DIGEST,
        generation_id=comparator.FROZEN_GENERATION_ID,
        proposal_contract_digest=comparator.FROZEN_PROPOSAL_CONTRACT_DIGEST,
        producer_contract_digest="producer",
        route_quotas={"composite": 1}, physical_rows=10,
        expected_query_start_ns=100, expected_latest_eligible_ns=90,
        expected_representation_digest="representation",
    )


def test_checkpoint_integrity_binds_timing_when_semantic_digest_does_not() -> None:
    module = _module()
    payload = {
        "query_episode_id": "a" * 24, "cold_seconds": 50.0,
        "warm_second_seconds": 25.0, "peak_rss_mb": 100.0,
    }
    semantic = module.case_digest(payload)
    integrity = module.checkpoint_integrity_digest(payload)
    payload["cold_seconds"] = 51.0
    assert module.case_digest(payload) == semantic
    assert module.checkpoint_integrity_digest(payload) != integrity


def test_comparator_reconstructs_hex_candidate_digest() -> None:
    module = _comparator_module()
    producer = _module()

    class Row:
        episode_id = "b" * 24
        symbol = "TEST"
        cutoff_ns = 123
        quality_tier = "A"
        lower_bound = 0.25
        routes = ("composite", "stage")
        overflow_fallback = False

    payload = producer._candidate_payload(type("Report", (), {"candidates": (Row(),)})())
    from market_analogues.packed_bound_search import bound_proposal_candidate_digest
    assert module.reconstructed_candidate_digest(payload) == bound_proposal_candidate_digest((Row(),))


def _valid_comparator_fixture(module):
    row = {
        "episode_id": "b" * 24, "symbol": "OTHER", "cutoff_ns": 10,
        "quality_tier": "A", "lower_bound_hex": float(0.25).hex(),
        "routes": ["composite"], "overflow_fallback": False,
    }
    candidate_digest = module.reconstructed_candidate_digest([row])
    invariant = {
        "schema_version": "m04r-global-bound-proposal-v1",
        "generation_id": module.FROZEN_GENERATION_ID,
        "query_episode_id": "a" * 24, "rows_scanned": 10,
        "eligible_rows": 8, "eligible_main_rows": 7, "eligible_overflow_rows": 1,
        "route_counts": {"composite": 1}, "route_quotas": {"composite": 1},
        "candidate_count": 1, "candidate_digest": candidate_digest,
    }
    invariant["result_digest"] = module._scan_digest(invariant)
    contract = {
        "contract_digest": "producer", "physical_rows": 10,
        "route_quotas": {"composite": 1},
    }
    registry_case = {
        "case_id": "case-1", "episode_id": "a" * 24, "symbol": "QUERY",
        "stock_prefix": {"digest": "stock"},
        "benchmark_prefix": {"digest": "benchmark"},
    }
    authority = {"certificate": {"eligible_candidates": 8}}
    authority["result_digest"] = stable_hash(authority)
    gates = {
        "cold_advice_supported_and_applied": True,
        "three_scan_digest_and_block_order_invariance": True,
        "internal_eligible_row_accounting": True, "physical_row_accounting": True,
        "frozen_route_quotas": True,
        "zero_temporal_overlap_tier_duplicate_errors": True,
        "measurements_finite_nonnegative_and_task_contains_cold": True,
        "cold_scan_at_most_120_seconds": True,
        "second_warm_scan_at_most_60_seconds": True,
        "rss_at_most_1024_mib": True,
    }
    candidate = {
        "schema_version": "candidate-recall-case-v2-producer",
        "registry_case_id": "case-1", "query_episode_id": "a" * 24,
        "query_symbol": "QUERY", "query_start_ns": 100, "latest_eligible_ns": 90,
        "registry_digest": module.FROZEN_REGISTRY_DIGEST,
        "generation_id": module.FROZEN_GENERATION_ID,
        "proposal_contract_digest": module.FROZEN_PROPOSAL_CONTRACT_DIGEST,
        "producer_contract_digest": "producer",
        "query_stock_prefix": registry_case["stock_prefix"],
        "query_benchmark_prefix": registry_case["benchmark_prefix"],
        "query_representation_digest": "representation",
        "scan_invariants": [invariant, invariant, invariant], "candidates": [row],
        "candidate_digest_reconstructed": candidate_digest,
        "violations": {"duplicates": 0, "future": 0, "same_symbol_overlap": 0, "tier": 0},
        "cold_advice_applied": True, "cold_seconds": 100.0,
        "warm_first_seconds": 50.0, "warm_second_seconds": 50.0,
        "cold_task_seconds": 101.0, "peak_rss_mb": 100.0,
        "route_quotas": {"composite": 1}, "gates": gates, "passed": True,
        "real_forward_outcomes_accessed": False, "created_at": "now",
    }
    candidate["result_digest"] = module.candidate_case_digest(candidate)
    candidate["checkpoint_integrity_digest"] = module.checkpoint_integrity_digest(candidate)
    return candidate, registry_case, authority, contract


def test_comparator_recomputes_timing_and_rejects_malformed_identity() -> None:
    module = _comparator_module()
    candidate, registry_case, authority, contract = _valid_comparator_fixture(module)
    assert module.validate_candidate_case(candidate, registry_case, authority, contract) == []
    candidate["cold_seconds"] = 121.0
    assert module.validate_candidate_case(candidate, registry_case, authority, contract)
    candidate, registry_case, authority, contract = _valid_comparator_fixture(module)
    candidate["warm_second_seconds"] = -1.0
    candidate["result_digest"] = module.candidate_case_digest(candidate)
    candidate["checkpoint_integrity_digest"] = module.checkpoint_integrity_digest(candidate)
    assert module.validate_candidate_case(candidate, registry_case, authority, contract)
    candidate, registry_case, authority, contract = _valid_comparator_fixture(module)
    candidate["scan_invariants"][0]["schema_version"] = "wrong"
    candidate["scan_invariants"] = [candidate["scan_invariants"][0]] * 3
    candidate["result_digest"] = module.candidate_case_digest(candidate)
    candidate["checkpoint_integrity_digest"] = module.checkpoint_integrity_digest(candidate)
    assert module.validate_candidate_case(candidate, registry_case, authority, contract)
    candidate, registry_case, authority, contract = _valid_comparator_fixture(module)
    candidate["candidates"][0]["episode_id"] = "not-hex"
    assert module.validate_candidate_case(candidate, registry_case, authority, contract)
