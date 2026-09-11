from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from experiments.m04r import m04r15_m2_prospective_payload_adapter_gate as gate


ROOT = Path(__file__).resolve().parents[1]


def raw_inputs() -> dict[str, bytes]:
    return {name: gate.snapshot(ROOT / path) for name, path in gate.INPUTS.items()}


def test_consumed_month_builds_valid_prospective_documents() -> None:
    contract = json.loads((ROOT / gate.CONTRACT).read_text())
    evidence = gate.gate_state(raw_inputs(), contract)
    assert len(evidence["checks"]) == 7
    assert all(row["passed"] for row in evidence["checks"])
    assert evidence["inventory"] == {
        "registry_rows": 24, "target_score_rows": 72,
        "causal_history_rows": evidence["inventory"]["causal_history_rows"],
        "market_regime": evidence["inventory"]["market_regime"],
        "cutoff": "2025-07-31",
    }


def test_contract_and_consumed_input_bytes_are_exact() -> None:
    contract = gate.decode((ROOT / gate.CONTRACT).read_bytes(), ROOT / gate.CONTRACT)
    state = {key: value for key, value in contract.items() if key != "contract_digest"}
    assert contract["contract_digest"] == gate.stable(state)
    raw = raw_inputs()
    assert {name: sha256(raw[name]).hexdigest()
            for name in contract["consumed_fixture"]["input_sha256"]} == (
        contract["consumed_fixture"]["input_sha256"]
    )
    assert not any(contract["claims"].values())


def test_result_publication_is_create_only(tmp_path: Path) -> None:
    path = tmp_path / "RESULT.json"
    gate.publish(path, {"passed": True})
    with pytest.raises(gate.GateError, match="create-only"):
        gate.publish(path, {"passed": True})
