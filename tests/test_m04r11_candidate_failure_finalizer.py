from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from market_analogues.types import stable_hash


def _module():
    path = (
        Path(__file__).parents[1] / "experiments" / "m04r"
        / "finalize_m04r11_candidate_failure.py"
    )
    spec = importlib.util.spec_from_file_location(
        "finalize_m04r11_candidate_failure", path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _experiment_module(name: str):
    path = Path(__file__).parents[1] / "experiments" / "m04r" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failure_digest_binds_evidence_but_not_timestamp() -> None:
    module = _module()
    payload = {
        "status": "terminal_performance_failure_after_interruption",
        "failed_case_id": module.EXPECTED_FAILED_CASE_ID,
        "created_at": "first",
    }
    digest = module.failure_digest(payload)
    payload["created_at"] = "second"
    assert module.failure_digest(payload) == digest
    payload["failed_case_id"] = "changed"
    assert module.failure_digest(payload) != digest


def _summary_fixture(module, *, terminal: bool):
    case = {
        "case_id": (
            module.EXPECTED_FAILED_CASE_ID if terminal else "nasdaq-BLDP-current-252"
        ),
        "episode_id": "a" * 24,
    }
    gates = {
        "cold_scan_at_most_120_seconds": not terminal,
        "second_warm_scan_at_most_60_seconds": True,
        "rss_at_most_1024_mib": True,
    }
    payload = {
        "gates": gates, "passed": not terminal,
        "result_digest": "semantic", "checkpoint_integrity_digest": "integrity",
        "cold_seconds": 267.57 if terminal else 79.0,
        "warm_second_seconds": 30.57, "peak_rss_mb": 800.0,
    }
    return case, payload


def test_terminal_sequence_requires_six_passes_and_exact_gabc_failure() -> None:
    module = _module()
    case, payload = _summary_fixture(module, terminal=False)
    assert module._terminal_summary(0, case, payload)["passed"] is True
    case, payload = _summary_fixture(module, terminal=True)
    summary = module._terminal_summary(6, case, payload)
    assert summary["false_gates"] == ["cold_scan_at_most_120_seconds"]

    invalid = {**payload, "gates": {**payload["gates"], "rss_at_most_1024_mib": False}}
    with pytest.raises(ValueError, match="classification"):
        module._terminal_summary(6, case, invalid)
    with pytest.raises(ValueError, match="pass status"):
        module._terminal_summary(0, case, payload)


def test_prefix_comparison_absence_and_exact_idempotency_fail_closed(
    tmp_path: Path,
) -> None:
    module = _module()
    module._require_exact_checkpoint_prefix({"a.json"}, {"a.json"})
    with pytest.raises(ValueError, match="filenames"):
        module._require_exact_checkpoint_prefix({"a.json"}, {"a.json", "b.json"})
    comparison_root = tmp_path / "comparison"
    module._require_comparison_absent(comparison_root)
    comparison_root.mkdir()
    (comparison_root / "RESULTS_OPENED.json").write_text("{}\n")
    with pytest.raises(ValueError, match="already opened"):
        module._require_comparison_absent(comparison_root)

    deterministic = {
        "schema_version": module.SCHEMA, "resume_authorized": False,
        "authority_results_opened": False,
    }
    existing = {
        **deterministic, "created_at": "now",
        "failure_digest": stable_hash(deterministic),
    }
    assert module._existing_failure_matches(existing, deterministic)
    existing["status"] = "drift"
    assert not module._existing_failure_matches(existing, deterministic)


def test_producer_and_comparator_refuse_terminal_failed_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = _experiment_module("m04r11_candidate_matrix")
    comparator = _experiment_module("compare_m04r11_candidate_matrix")
    frozen_failed_digest = producer.FAILED_V1_PRODUCER_CONTRACT_DIGEST
    contract_deterministic = {"schema_version": "test"}
    contract = {
        **contract_deterministic,
        "contract_digest": stable_hash(contract_deterministic),
    }
    monkeypatch.setattr(
        producer, "FAILED_V1_PRODUCER_CONTRACT_DIGEST",
        contract["contract_digest"],
    )
    deterministic = {
        "schema_version": producer.TERMINAL_FAILURE_SCHEMA,
        "registry_digest": producer.FROZEN_REGISTRY_DIGEST,
        "producer_contract_digest": contract["contract_digest"],
        "generation_id": producer.FROZEN_GENERATION_ID,
        "proposal_contract_digest": producer.FROZEN_PROPOSAL_CONTRACT_DIGEST,
        "status": "terminal_performance_failure_after_interruption",
        "completed_cases": 7,
        "failed_case_id": "nasdaq-GABC-current-252",
        "failed_gates": ["cold_scan_at_most_120_seconds"],
        "candidate_pools_sealed": False,
        "authority_results_opened": False,
        "candidate_authority_comparison_opened": False,
        "resume_authorized": False,
    }
    payload = {
        **deterministic, "created_at": "now",
        "failure_digest": stable_hash(deterministic),
    }
    (tmp_path / "FAILED.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="resume is forbidden"):
        producer.refuse_terminal_failure(tmp_path, contract)
    comparator_payload = {
        **payload,
        "producer_contract_digest": frozen_failed_digest,
    }
    comparator_deterministic = {
        key: value for key, value in comparator_payload.items()
        if key not in {"created_at", "failure_digest"}
    }
    comparator_payload["failure_digest"] = stable_hash(comparator_deterministic)
    (tmp_path / "FAILED.json").write_text(json.dumps(comparator_payload))
    with pytest.raises(ValueError, match="truth comparison is forbidden"):
        comparator._refuse_failed_candidate_root(tmp_path)
