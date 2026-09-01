from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import verify_m04r14_t14_10_walk_forward_contract as verifier
from market_analogues.types import stable_hash


CONTRACT = ROOT / "config/m04r14-t14-10-walk-forward-contract.json"


def _write(path: Path, value: dict) -> None:
    state = {key: item for key, item in value.items() if key != "contract_digest"}
    value["contract_digest"] = stable_hash(state)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def test_real_walk_forward_contract_is_frozen_before_historical_queries() -> None:
    result = verifier.validate(ROOT)
    assert result["passed"] is True
    assert result["fold_count"] == 5
    assert result["target_queries_per_month"] == 24
    assert result["historical_walk_forward_query_outcomes_opened"] is False
    assert result["final_period_result_opened"] is False
    assert result["production_promotion_authorized"] is False


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda c: c["query_registry"].update(future_observations_may_affect_selection=True), "registry"),
        (lambda c: c["temporal_protocol"].update(purge_sessions=0), "temporal"),
        (lambda c: c["evidence_methods"].update(forced_score_lane_includes_abstained_queries=False), "evidence"),
        (lambda c: c["probability_construction"].update(no_posthoc_platt_isotonic_or_model_fit=False), "probability"),
        (lambda c: c["abstention"]["novelty"].update(threshold_quantile=0.99), "abstention"),
        (lambda c: c["baselines"].pop("price_only_similarity"), "baseline"),
        (lambda c: c["metrics"].update(hit_rate_alone_can_pass=True), "metric"),
        (lambda c: c["acceptance"].update(minimum_final_evaluable_queries=1), "acceptance"),
        (lambda c: c["verification"].update(final_period_result_opened=True), "verification"),
        (lambda c: c["verification"].update(production_promotion_authorized=True), "verification"),
    ],
)
def test_walk_forward_contract_rejects_semantic_weakening(
    tmp_path: Path, mutation, match: str,
) -> None:
    value = deepcopy(json.loads(CONTRACT.read_text()))
    mutation(value)
    path = tmp_path / "contract.json"
    _write(path, value)
    with pytest.raises(verifier.WalkForwardContractError, match=match):
        verifier.validate(ROOT, path)


def test_walk_forward_contract_rejects_self_consistent_upstream_rewrite(
    tmp_path: Path,
) -> None:
    value = deepcopy(json.loads(CONTRACT.read_text()))
    value["upstream"]["evidence_verification_result_digest"] = "0" * 64
    path = tmp_path / "contract.json"
    _write(path, value)
    with pytest.raises(verifier.WalkForwardContractError, match="upstream"):
        verifier.validate(ROOT, path)


def test_walk_forward_contract_keeps_missing_data_boundary_visible() -> None:
    value = json.loads(CONTRACT.read_text())
    assert value["baselines"]["sector_frequency"].startswith("unavailable_")
    assert value["acceptance"][
        "product_calibration_pass_requires_missing_sector_membership_delisting_event_cluster_controls"
    ] is True
    assert value["boundary"]["production_promotion_authorized"] is False
