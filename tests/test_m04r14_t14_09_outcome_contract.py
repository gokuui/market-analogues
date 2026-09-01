from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import verify_m04r14_t14_09_outcome_contract as verifier
from market_analogues.types import stable_hash


CONTRACT = ROOT / "config/m04r14-t14-09-outcome-contract.json"


def _write_contract(path: Path, value: dict) -> None:
    state = {key: item for key, item in value.items() if key != "contract_digest"}
    value["contract_digest"] = stable_hash(state)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def test_real_contract_is_frozen_and_bound() -> None:
    result = verifier.validate(ROOT)
    assert result["passed"] is True
    assert result["scheduled_queries"] == 3270
    assert result["retrieved_match_rows"] == 65400
    assert result["real_forward_outcomes_accessed"] is False


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda c: c["session_axis"].update(horizons=[5, 10, 20]), "session/ATR"),
        (lambda c: c["primary_barrier"].update(same_first_touch_bar="favorable"), "session/ATR"),
        (lambda c: c["causal_embargo"].update(outcomes_may_affect_similarity_rank_or_weight=True), "embargo"),
        (lambda c: c["censoring_and_limitations"].update(missing_delisting_return_imputed=True), "censor"),
        (lambda c: c["benchmark_alignment"].update(forward_fill=True), "benchmark"),
        (lambda c: c["claims"].update(profitability_claimed=True), "claim"),
    ],
)
def test_contract_rejects_semantic_weakening(tmp_path: Path, mutation, match: str) -> None:
    value = json.loads(CONTRACT.read_text())
    mutation(value)
    path = tmp_path / "contract.json"
    _write_contract(path, value)
    with pytest.raises(verifier.OutcomeContractError, match=match):
        verifier.validate(ROOT, path)


def test_contract_rejects_self_consistent_upstream_rewrite(tmp_path: Path) -> None:
    value = deepcopy(json.loads(CONTRACT.read_text()))
    value["retrieval_inputs"]["snapshot_case_manifest_digest"] = "0" * 64
    path = tmp_path / "contract.json"
    _write_contract(path, value)
    with pytest.raises(verifier.OutcomeContractError, match="retrieval evidence"):
        verifier.validate(ROOT, path)


def test_contract_rejects_unknown_fields_even_with_new_digest(tmp_path: Path) -> None:
    value = json.loads(CONTRACT.read_text())
    value["tuned_after_outcomes"] = False
    path = tmp_path / "contract.json"
    _write_contract(path, value)
    with pytest.raises(verifier.OutcomeContractError, match="contract keys"):
        verifier.validate(ROOT, path)
