from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.m04r import m04r15_m2_real_source_launcher_gate as gate


ROOT = Path(__file__).resolve().parents[1]


def inputs() -> tuple[dict, dict, dict]:
    return tuple(json.loads((ROOT / path).read_text()) for path in (
        gate.CONTRACT, gate.AVAILABILITY, gate.VERIFICATION,
    ))


def test_current_verified_pair_produces_only_wait_state() -> None:
    contract, availability, verification = inputs()
    state = gate.gate_state(
        contract, availability, verification, credential_present=False,
        free_bytes=20_000_000_000, canonical_stock_exists=True,
        canonical_benchmark_exists=True,
    )
    assert state["status"] == "verified_wait_state"
    assert state["decision"]["action"] == "wait_for_source_refresh"
    assert state["decision"]["registry_callback_calls"] == 0
    assert state["decision"]["prediction_callback_calls"] == 0
    assert state["refresh_assessment"]["blocking_reasons"] == [
        "required_provider_credential_absent"
    ]
    assert state["canonical_inputs_modified"] is False
    assert state["post_freeze_outcomes_opened"] is False


def test_contract_is_sealed_and_claims_are_false() -> None:
    contract, _, _ = inputs()
    state = {key: value for key, value in contract.items() if key != "contract_digest"}
    assert contract["contract_digest"] == gate.stable(state)
    assert not any(contract["claims"].values())
    assert contract["refresh"]["canonical_inputs_mutable_by_refresh"] is False


def test_result_publication_is_create_only(tmp_path: Path) -> None:
    path = tmp_path / "RESULT.json"
    gate.publish(path, {"status": "wait"})
    with pytest.raises(gate.GateError, match="create-only"):
        gate.publish(path, {"status": "wait"})
