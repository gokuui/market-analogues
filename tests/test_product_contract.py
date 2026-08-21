from __future__ import annotations

from copy import deepcopy
from html.parser import HTMLParser
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from market_analogues.cli import main
from market_analogues.product_contract import (
    ProductContractError,
    historical_outcome_eligibility,
    load_product_contract,
    validate_contract,
    validate_trial_ledger,
    write_contract_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "config" / "case-memory-contract.yaml"
LEDGER_PATH = ROOT / "config" / "case-memory-trials.yaml"


class _CanonicalContractParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.capture = False
        self.text: list[str] = []
        self.digest: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "body":
            self.digest = values.get("data-contract-digest")
        if tag == "pre" and values.get("id") == "canonical-contract":
            self.capture = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "pre" and self.capture:
            self.capture = False

    def handle_data(self, data: str) -> None:
        if self.capture:
            self.text.append(data)


def _contract_payload() -> dict:
    return yaml.safe_load(CONTRACT_PATH.read_text())


def test_repository_contract_and_trial_ledger_are_frozen_and_bound() -> None:
    contract = load_product_contract(CONTRACT_PATH)
    ledger = validate_trial_ledger(LEDGER_PATH, contract)
    assert len(contract.digest) == 64
    assert ledger["contract_digest"] == contract.digest
    assert ledger["trials"] == []
    assert contract.payload["decision"]["primary_mode"] == "after_close_daily"
    assert contract.payload["decision"]["modes"]["entry_open"]["enabled"] is False
    assert contract.payload["outcomes"]["outcomes_may_affect_similarity"] is False


@pytest.mark.parametrize(
    "mutation,expected",
    [
        (lambda p: p.update(status="draft"), "frozen"),
        (lambda p: p["purpose"].update(prediction_or_advice=True), "prediction_or_advice"),
        (lambda p: p["representation"].update(future_data_policy="allowed"), "future_data_policy"),
        (lambda p: p["outcomes"].update(outcomes_may_affect_similarity=True), "never affect"),
        (lambda p: p["evidence"]["abstain_when"].pop(), "abstention vocabulary"),
    ],
)
def test_contract_rejects_safety_regressions(mutation, expected: str) -> None:
    payload = deepcopy(_contract_payload())
    mutation(payload)
    with pytest.raises(ProductContractError, match=expected):
        validate_contract(payload)


def test_contract_digest_is_independent_of_yaml_key_order(tmp_path: Path) -> None:
    payload = _contract_payload()
    reordered = dict(reversed(list(payload.items())))
    second = tmp_path / "reordered.yaml"
    second.write_text(yaml.safe_dump(reordered, sort_keys=False))
    assert load_product_contract(second).digest == load_product_contract(CONTRACT_PATH).digest


def test_any_contract_change_breaks_the_frozen_trial_ledger_binding(tmp_path: Path) -> None:
    payload = _contract_payload()
    payload["retrieval"]["displayed_neighbors"] = 21
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(payload))
    changed_contract = load_product_contract(changed)
    with pytest.raises(ProductContractError, match="contract_digest does not match"):
        validate_trial_ledger(LEDGER_PATH, changed_contract)


def test_historical_outcome_embargo_uses_observed_sessions() -> None:
    bars = pd.DataFrame({"timestamp": pd.bdate_range("2020-01-01", periods=30)})
    cutoff = bars.timestamp.iloc[5]
    completion = bars.timestamp.iloc[10]

    too_early = historical_outcome_eligibility(bars, cutoff, 5, bars.timestamp.iloc[9])
    assert not too_early.eligible
    assert too_early.reason == "outcome_not_yet_observable"
    assert too_early.outcome_completion_timestamp == completion

    boundary = historical_outcome_eligibility(bars, cutoff, 5, completion)
    assert boundary.eligible
    assert boundary.reason == "eligible"
    assert boundary.available_sessions == 5


def test_historical_outcome_embargo_rejects_newer_and_incomplete_cases() -> None:
    bars = pd.DataFrame({"timestamp": pd.bdate_range("2020-01-01", periods=12)})
    newer = historical_outcome_eligibility(
        bars, bars.timestamp.iloc[8], 2, bars.timestamp.iloc[8],
    )
    assert not newer.eligible
    assert newer.reason == "analogue_not_earlier"

    incomplete = historical_outcome_eligibility(
        bars, bars.timestamp.iloc[9], 5, bars.timestamp.iloc[-1] + pd.Timedelta(days=10),
    )
    assert not incomplete.eligible
    assert incomplete.reason == "incomplete_horizon"
    assert incomplete.available_sessions == 2


def test_contract_html_and_json_are_generated_from_identical_payload(tmp_path: Path) -> None:
    contract = load_product_contract(CONTRACT_PATH)
    ledger = validate_trial_ledger(LEDGER_PATH, contract)
    machine = tmp_path / "contract.json"
    html = tmp_path / "contract.html"
    write_contract_artifacts(contract, ledger, machine, html)

    machine_payload = json.loads(machine.read_text())
    parser = _CanonicalContractParser()
    parser.feed(html.read_text())
    assert parser.digest == contract.digest
    assert json.loads("".join(parser.text)) == machine_payload["contract"] == contract.payload
    assert machine_payload["contract_digest"] == contract.digest


def test_contract_cli_writes_artifacts_and_gate(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    config = tmp_path / "datasets.yaml"
    config.write_text(yaml.safe_dump({
        "artifact_dir": str(artifacts),
        "datasets": {
            "demo": {
                "adapter": "directory", "path": str(tmp_path / "bars"),
                "format": "parquet",
            },
        },
    }))
    result = main([
        "verify-case-memory-contract", "--config", str(config),
        "--contract", str(CONTRACT_PATH), "--trial-ledger", str(LEDGER_PATH),
    ])
    assert result == 0
    contract = load_product_contract(CONTRACT_PATH)
    machine = artifacts / "contracts" / f"{contract.contract_id}.json"
    html = artifacts / "reports" / f"m00-{contract.contract_id}.html"
    gate = artifacts / "gates" / "m00_case_memory_contract.json"
    assert machine.exists() and html.exists() and gate.exists()
    gate_payload = json.loads(gate.read_text())
    assert gate_payload["passed"] is True
    assert gate_payload["metrics"]["contract_digest"] == contract.digest
    assert gate_payload["metrics"]["trial_count"] == 0
