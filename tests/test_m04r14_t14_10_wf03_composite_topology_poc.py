from copy import deepcopy
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import m04r14_t14_10_wf03_composite_topology_poc as subject


def _case(query_id: str) -> dict:
    certificate = {"result_digest": query_id, "elapsed_seconds": 1.0}
    value = {
        "query_id": query_id,
        "proposal_result_digest": f"proposal-{query_id}",
        "certificate": certificate,
        "matches": [{"episode_id": query_id}],
    }
    value["semantic_digest"] = subject.stable_hash(subject._case_semantics(value))
    return value


def test_topology_comparison_ignores_measurement_and_rejects_semantic_drift() -> None:
    sequential = {"cases": [_case(query_id) for query_id in subject.QUERY_IDS]}
    parallel = deepcopy(sequential)
    for row in parallel["cases"]:
        row["exact_seconds"] = 9.0
        row["certificate"]["elapsed_seconds"] = 8.0
    assert subject.compare_topologies(sequential, parallel)
    parallel["cases"][0]["matches"][0]["episode_id"] = "changed"
    assert not subject.compare_topologies(sequential, parallel)


def test_true_composite_contract_contains_all_nonprice_channels() -> None:
    contract = subject._contract()
    expected = {
        "coarse", "stage", "structural", "candle_volatility",
        "volume_shock", "market_context",
    }
    assert expected < set(subject.DistanceConfig().weights)
    assert all(subject.DistanceConfig().weights[channel] > 0 for channel in expected)
    assert contract["outcomes_or_labels_used"] is False
