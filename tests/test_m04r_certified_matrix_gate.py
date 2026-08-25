from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

from market_analogues.certified_packed_search import certified_packed_search_contract
from market_analogues.m04r_certified_search_verification import _certificate_digest
from market_analogues.types import stable_hash


def _module():
    path = (
        Path(__file__).parents[1] / "experiments" / "m04r"
        / "certified_packed_search_all12.py"
    )
    spec = importlib.util.spec_from_file_location("m04r_certified_matrix", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _module()
CASE_OMITTED = MODULE.CASE_OMITTED
CASE_SCHEMA = MODULE.CASE_SCHEMA
_aggregate = MODULE._aggregate
_comparison = MODULE._comparison
_deterministic = MODULE._deterministic
_matrix_deterministic = MODULE._matrix_deterministic
_valid_checkpoint = MODULE._valid_checkpoint


def _case(query_id: str, seconds: float = 10.0) -> tuple[
    dict[str, object], dict[str, object], dict[str, object]
]:
    generation = "b" * 64
    contract = certified_packed_search_contract()["digest"]
    matches = [{
        "episode_id": f"{index + 1:024x}",
        "symbol": f"S{index:02d}",
        "cutoff": "2020-01-01T00:00:00",
        "total_distance": 0.1 + index / 100,
        "component_distances": {"coarse": 0.2 + index / 100},
        "alignment": [[0, 0]],
        "quality_tier": "A",
    } for index in range(20)]
    authority: dict[str, object] = {
        "query_episode_id": query_id,
        "authority_digest": stable_hash({"query": query_id}),
        "matches": copy.deepcopy(matches),
    }
    build = {"generation_id": generation, "result_digest": "c" * 64}
    certificate: dict[str, object] = {
        "schema_version": "m04r-certified-packed-search-v1",
        "contract_digest": contract,
        "generation_id": generation,
        "query_episode_id": query_id,
        "input_digest": stable_hash({"input": query_id}),
        "eligible_candidates": 100,
        "exact_evaluated": 20,
        "safely_pruned": 80,
        "stopped_early": True,
        "stop_threshold": 0.29,
        "next_lower_bound": 0.30,
        "maximum_quantized_bound_excess": 0.0,
        "materialization_groups": 2,
        "sparse_symbols": 2,
        "batch_symbols": 0,
        "rounds": [{
            "frontier_rows": 100,
            "exact_rows": 20,
            "next_lower_bound": 0.30,
            "constrained_threshold": 0.29,
            "selected_rows": 20,
            "certified": True,
            "proposal_digest": "d" * 64,
        }],
    }
    comparison = _comparison(matches, authority["matches"])
    gates = {
        "ordered_ids_equal_authority": True,
        "alignments_equal_authority": True,
        "component_names_equal_authority": True,
        "total_delta_within_1e_7": True,
        "component_delta_within_1e_6": True,
        "candidate_accounting": True,
        "strict_stopping": True,
        "quantized_bound_safe": True,
        "runtime_within_600_seconds": True,
        "rss_within_1536_mib": True,
    }
    payload: dict[str, object] = {
        "schema_version": CASE_SCHEMA,
        "contract_digest": contract,
        "generation_id": generation,
        "full_build_evidence_digest": build["result_digest"],
        "query_episode_id": query_id,
        "authority_digest": authority["authority_digest"],
        "controls": {"block_rows": 7, "workers": 2},
        "matches": matches,
        **comparison,
        "certificate": certificate,
        "seconds": seconds,
        "peak_rss_mb": 100.0,
        "gates": gates,
        "gate_passed": True,
        "status": "completed",
        "real_forward_outcomes_accessed": False,
        "created_at": "now",
    }
    certificate_digest = _certificate_digest(payload)
    certificate["result_digest"] = certificate_digest
    payload["certificate_digest"] = certificate_digest
    payload["result_digest"] = stable_hash(_deterministic(payload, CASE_OMITTED))
    return payload, authority, build


def test_matrix_checkpoint_validation_and_timing_free_digest() -> None:
    cases = []
    expected = []
    build: dict[str, object] | None = None
    for index in range(12):
        query_id = f"{index + 1:024x}"
        case, authority, build = _case(query_id, 100.0 + index)
        assert _valid_checkpoint(case, authority, build)
        cases.append(case)
        expected.append(query_id)
    assert build is not None
    matrix = _aggregate(
        cases, [], build=build, expected_ids=expected, started_at="one",
    )
    assert matrix["gate_passed"] is True
    original_digest = matrix["result_digest"]
    matrix["started_at"] = "two"
    matrix["created_at"] = "later"
    matrix["total_seconds"] = 9999.0
    matrix["peak_rss_mb"] = 999.0
    for case in matrix["cases"]:
        case["seconds"] += 1
        case["peak_rss_mb"] += 1
    assert stable_hash(_matrix_deterministic(matrix)) == original_digest

    tampered = copy.deepcopy(cases[0])
    tampered["gates"]["strict_stopping"] = False
    tampered["result_digest"] = stable_hash(_deterministic(tampered, CASE_OMITTED))
    _, authority, build = _case(expected[0])
    assert not _valid_checkpoint(tampered, authority, build)
