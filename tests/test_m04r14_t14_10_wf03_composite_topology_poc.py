from copy import deepcopy
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import m04r14_t14_10_wf03_composite_topology_poc as subject
from experiments.m04r import verify_m04r14_t14_10_wf03_composite_topology_poc as verifier


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


def test_independent_certificate_validator_rejects_digest_tamper() -> None:
    certificate = {
        "schema_version": subject._contract()["schema_version"],
        "contract_digest": "contract", "generation_id": "generation",
        "query_episode_id": "a" * 24, "input_digest": "input",
        "eligible_candidates": 3, "exact_evaluated": 1, "safely_pruned": 2,
        "stopped_early": True, "stop_threshold": 1.0,
        "next_lower_bound": 1.1, "maximum_quantized_bound_excess": 0.0,
        "rounds": [],
        "native_bound_accounting": {
            "exact_dtw_evaluated": 1, "native_bound_evaluated": 2,
            "native_bound_pruned": 1, "packed_bound_pruned": 1,
        },
        "minimum_native_pruned_bound": 1.1,
        "threshold_closure_passes": [], "elapsed_seconds": 2.0,
    }
    matches = [{
        "episode_id": f"{index:024x}", "symbol": f"S{index}",
        "total_distance": float(index),
        "component_distances": {key: 0.0 for key in verifier.EXPECTED_COMPONENTS},
        "alignment": [], "quality_tier": "A",
    } for index in range(20)]
    certificate["result_digest"] = verifier._certificate_digest({
        "certificate": certificate, "matches": matches,
    })
    case = {
        "query_id": "a" * 24, "proposal_result_digest": "proposal",
        "certificate": certificate, "matches": matches,
    }
    semantics = verifier._case_semantics(case)
    assert isinstance(semantics, dict)
    assert semantics == subject._case_semantics(case)
    case["semantic_digest"] = subject.stable_hash(semantics)
    verifier.validate_certificate(case)
    case["certificate"]["result_digest"] = "0" * 64
    try:
        verifier.validate_certificate(case)
    except verifier.CompositeTopologyVerificationError:
        pass
    else:
        raise AssertionError("tampered certificate was accepted")


def test_verifier_normalizes_distinct_main_and_overflow_layouts() -> None:
    main = np.zeros(2, dtype=[
        ("episode_id", "V12"), ("cutoff_ns", "<i8"),
        ("symbol_id", "<u4"), ("quality_tier", "u1"),
        ("samples", "f4", (4,)),
    ])
    overflow = np.zeros(1, dtype=[
        ("episode_id", "V12"), ("cutoff_ns", "<i8"),
        ("symbol_id", "<u4"), ("quality_tier", "u1"),
        ("padding", "V3"),
    ])
    normalized = np.concatenate((
        verifier._metadata_records(main), verifier._metadata_records(overflow),
    ))
    assert normalized.dtype == verifier.METADATA_DTYPE
    assert len(normalized) == 3


def test_certificate_json_preserves_valid_incomplete_threshold() -> None:
    from dataclasses import dataclass

    @dataclass
    class Certificate:
        rounds: list
        threshold_closure_passes: list
        stop_threshold: float

    original = Certificate(
        rounds=[{"constrained_threshold": float("inf"),
                 "selected_rows": 19, "certified": False}],
        threshold_closure_passes=[], stop_threshold=0.5,
    )
    encoded = subject._certificate_json_value(original)
    assert encoded["rounds"][0]["constrained_threshold"] \
        == subject.POSITIVE_INFINITY_SENTINEL
    decoded = subject.decode_certificate_json_value(encoded)
    assert decoded["rounds"][0]["constrained_threshold"] == float("inf")


def test_certificate_json_rejects_nonfinite_final_value() -> None:
    from dataclasses import dataclass
    import pytest

    @dataclass
    class Certificate:
        rounds: list
        threshold_closure_passes: list
        stop_threshold: float

    with pytest.raises(subject.CompositeTopologyError):
        subject._certificate_json_value(Certificate([], [], float("inf")))
