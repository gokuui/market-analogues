from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.m04r import m04r15_m2_prospective_listener_synthetic_gate as gate


ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_gate_checks_all_pass() -> None:
    contract = json.loads((ROOT / gate.CONTRACT).read_text())
    checks, inventory = gate.synthetic_checks(contract["contract_digest"])
    assert len(checks) == 8
    assert all(row["passed"] for row in checks)
    assert inventory["synthetic_queries"] == 24
    assert inventory["prediction_methods"] == [
        "candidate", "matched_causal_history", "locked_composite",
    ]


def test_contract_is_self_sealed_and_real_launch_is_disabled() -> None:
    contract = gate.decode((ROOT / gate.CONTRACT).read_bytes(), ROOT / gate.CONTRACT)
    state = {key: value for key, value in contract.items() if key != "contract_digest"}
    assert contract["contract_digest"] == gate.stable(state)
    assert contract["real_launch"]["currently_authorized"] is False
    assert not any(contract["claims"].values())


def test_result_publication_is_create_only(tmp_path: Path) -> None:
    path = tmp_path / "RESULT.json"
    gate.publish(path, {"passed": True})
    with pytest.raises(gate.GateError, match="create-only"):
        gate.publish(path, {"passed": True})
