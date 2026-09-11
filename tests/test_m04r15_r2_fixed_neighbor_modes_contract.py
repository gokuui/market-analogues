from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import verify_m04r15_r2_fixed_neighbor_modes_contract as verifier


CONTRACT = ROOT / "config/m04r15-r2-fixed-neighbor-modes-contract-v1.json"


def _write(path: Path, value: dict) -> None:
    state = {key: item for key, item in value.items() if key != "contract_digest"}
    value["contract_digest"] = verifier._stable(state)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def test_real_r2_contract_is_bound_without_opening_future_paths() -> None:
    result = verifier.validate(ROOT)
    assert result["passed"] is True
    assert result["observed"] == {
        "query_count": 3270,
        "query_link_count": 65400,
        "duplicate_matched_symbol_links_within_query": 810,
        "query_symbol_links": 50,
        "future_path_rows_from_parquet_metadata": 6917999,
    }
    assert result["future_path_columns_opened"] == []
    assert result["real_future_path_mode_computation_opened"] is False
    assert result["synthetic_gate_authorized"] is True
    assert result["predictive_claim_authorized"] is False


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda c: c["member_rules"].update(maximum_primary_episodes_per_matched_symbol=2), "member"),
        (lambda c: c["path_contract"].update(cluster_horizon_sessions=126), "path"),
        (lambda c: c["algorithm"].update(maximum_modes=8), "clustering"),
        (lambda c: c["stability"].update(block="iid_member"), "stability"),
        (lambda c: c["output"].update(context_slices="available"), "output"),
        (lambda c: c["claim_boundary"].update(predictive_claim_authorized=True), "claim"),
        (lambda c: c["verification"].update(real_future_path_mode_computation_opened=True), "verification"),
    ],
)
def test_r2_contract_rejects_self_consistent_semantic_weakening(
    tmp_path: Path, mutation, match: str,
) -> None:
    value = deepcopy(json.loads(CONTRACT.read_text()))
    mutation(value)
    path = tmp_path / "contract.json"
    _write(path, value)
    with pytest.raises(verifier.R2ContractError, match=match):
        verifier.validate(ROOT, path)


def test_r2_contract_rejects_self_consistent_upstream_rewrite(tmp_path: Path) -> None:
    value = deepcopy(json.loads(CONTRACT.read_text()))
    value["upstream"]["future_paths_sha256"] = "0" * 64
    path = tmp_path / "contract.json"
    _write(path, value)
    with pytest.raises(verifier.R2ContractError, match="upstream"):
        verifier.validate(ROOT, path)


def test_r2_contract_rejects_unknown_field(tmp_path: Path) -> None:
    value = deepcopy(json.loads(CONTRACT.read_text()))
    value["outcome_tuned"] = False
    path = tmp_path / "contract.json"
    _write(path, value)
    with pytest.raises(verifier.R2ContractError, match="keys"):
        verifier.validate(ROOT, path)
