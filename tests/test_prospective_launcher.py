from __future__ import annotations

import copy

import pytest

from market_analogues.prospective_launcher import (
    ProspectiveLaunchError, decide_launch, refresh_assessment, stable,
)


def sealed(state: dict, field: str) -> dict:
    return {**state, field: stable(state), "created_at": "now"}


def documents(ready: bool = False) -> tuple[dict, dict, dict]:
    result_state = {
        "contract_digest": "a" * 64, "readiness_passed": ready,
        "registry_creation_authorized": ready, "source_values_opened": False,
        "post_freeze_outcomes_opened": False,
    }
    result = sealed(result_state, "result_digest")
    verify_state = {
        "passed": True, "live_source_matched_at_verification": True,
        "producer_result_digest": result["result_digest"],
        "readiness_passed": ready, "registry_creation_authorized": ready,
        "source_values_opened": False, "post_freeze_outcomes_opened": False,
    }
    verification = sealed(verify_state, "verification_digest")
    contract = {
        "upstream": {
            "availability_contract_digest": "a" * 64,
            "blocked_result_digest": result["result_digest"],
            "blocked_verification_digest": verification["verification_digest"],
        },
        "launcher": {"blocked_action": "wait_for_source_refresh",
                     "ready_action": "run_registry_and_prediction"},
        "refresh": {"provider": "EODHD", "credential_environment_variable": "TOKEN",
                    "minimum_free_bytes": 100},
    }
    return contract, result, verification


def test_blocked_pair_cannot_reach_callbacks() -> None:
    contract, result, verification = documents()
    calls: list[str] = []
    decision = decide_launch(
        contract, result, verification,
        registry_callback=lambda: calls.append("registry"),
        prediction_callback=lambda: calls.append("prediction"),
    )
    assert decision.action == "wait_for_source_refresh"
    assert decision.registry_callback_calls == decision.prediction_callback_calls == 0
    assert calls == []


def test_tamper_mismatch_and_disagreement_fail_closed() -> None:
    contract, result, verification = documents()
    tampered = copy.deepcopy(result); tampered["readiness_passed"] = True
    with pytest.raises(ProspectiveLaunchError, match="seal differs"):
        decide_launch(contract, tampered, verification,
                      registry_callback=lambda: None, prediction_callback=lambda: None)
    disagreement = copy.deepcopy(verification)
    state = {key: value for key, value in disagreement.items()
             if key not in {"verification_digest", "created_at"}}
    state["readiness_passed"] = True
    disagreement = sealed(state, "verification_digest")
    with pytest.raises(ProspectiveLaunchError, match="flags disagree"):
        decide_launch(contract, result, disagreement,
                      registry_callback=lambda: None, prediction_callback=lambda: None)


def test_ready_pair_needs_new_identities_and_calls_once() -> None:
    contract, result, verification = documents(ready=True)
    with pytest.raises(ProspectiveLaunchError, match="blocked identities"):
        decide_launch(contract, result, verification,
                      registry_callback=lambda: None, prediction_callback=lambda: None)
    contract["upstream"]["blocked_result_digest"] = "b" * 64
    contract["upstream"]["blocked_verification_digest"] = "c" * 64
    calls: list[str] = []
    decision = decide_launch(
        contract, result, verification,
        registry_callback=lambda: calls.append("registry"),
        prediction_callback=lambda: calls.append("prediction"),
    )
    assert decision.action == "run_registry_and_prediction"
    assert calls == ["registry", "prediction"]


def test_refresh_assessment_records_presence_only() -> None:
    contract, _, _ = documents()
    value = refresh_assessment(
        contract, credential_present=False, free_bytes=200,
        canonical_stock_exists=True, canonical_benchmark_exists=True,
    )
    assert value["blocking_reasons"] == ["required_provider_credential_absent"]
    assert value["credential_value_recorded"] is False
    assert value["checks"]["capacity_passed"] is True
    assert value["canonical_inputs_modified"] is False
