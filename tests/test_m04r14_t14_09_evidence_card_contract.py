from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import verify_m04r14_t14_09_evidence_card_contract as verifier
from market_analogues.types import stable_hash


CONTRACT = ROOT / "config/m04r14-t14-09-evidence-card-contract.json"


def _write(path: Path, value: dict) -> None:
    state = {key: item for key, item in value.items() if key != "contract_digest"}
    value["contract_digest"] = stable_hash(state)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def test_real_evidence_contract_is_frozen_and_bound() -> None:
    result = verifier.validate(ROOT)
    assert result["passed"] is True
    assert result["query_count"] == 3270
    assert result["query_link_count"] == 65400
    assert result["query_level_outcome_aggregation_opened"] is False
    assert result["production_promotion_authorized"] is False


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda c: c["raw_evidence"].update(preserve_original_rank_1_through_20=False), "raw/dependence"),
        (lambda c: c["primary_evidence"].update(maximum_episodes_per_matched_symbol=2), "raw/dependence"),
        (lambda c: c["summaries"]["locked_weighted"].update(unnormalized_formula="1 / rank"), "weight"),
        (lambda c: c["abstention"].update(predictive_claim_status="prediction_allowed"), "abstention"),
        (lambda c: c["verification"].update(production_promotion_authorized=True), "verification"),
    ],
)
def test_evidence_contract_rejects_semantic_weakening(
    tmp_path: Path, mutation, match: str,
) -> None:
    value = deepcopy(json.loads(CONTRACT.read_text()))
    mutation(value)
    path = tmp_path / "contract.json"
    _write(path, value)
    with pytest.raises(verifier.EvidenceContractError, match=match):
        verifier.validate(ROOT, path)


def test_evidence_contract_rejects_self_consistent_upstream_rewrite(tmp_path: Path) -> None:
    value = deepcopy(json.loads(CONTRACT.read_text()))
    value["upstream"]["full_verification_result_digest"] = "0" * 64
    path = tmp_path / "contract.json"
    _write(path, value)
    with pytest.raises(verifier.EvidenceContractError, match="upstream"):
        verifier.validate(ROOT, path)


def test_evidence_contract_rejects_unknown_field(tmp_path: Path) -> None:
    value = json.loads(CONTRACT.read_text())
    value["outcome_tuned"] = False
    path = tmp_path / "contract.json"
    _write(path, value)
    with pytest.raises(verifier.EvidenceContractError, match="keys"):
        verifier.validate(ROOT, path)
