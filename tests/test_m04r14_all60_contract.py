from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "experiments/m04r/m04r14_all60_contract.py"
SPEC = importlib.util.spec_from_file_location("m04r14_all60_contract_tested", PATH)
assert SPEC is not None and SPEC.loader is not None
contract = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = contract
SPEC.loader.exec_module(contract)


def test_descriptor_paths_schemas_keys_and_digest_are_deterministic() -> None:
    assert contract.DESCRIPTOR["paths"] == {
        "candidate": "config/data/analogues/m04r14/all60-certified-development-v1",
        "verifier": (
            "config/data/analogues/m04r14/all60-certified-development-v1-verification"
        ),
        "preregistration": "experiments/m04r/m04r14_all60_preregistered.json",
    }
    assert set(contract.SCHEMAS) == set(contract.FIELD_KEYS)
    assert all(tuple(sorted(keys)) == keys and len(keys) == len(set(keys))
               for keys in contract.FIELD_KEYS.values())
    assert contract.FIELD_KEYS["proposal"] == tuple(sorted(
        ("state", "digest", "measurement", "created_at")
    ))
    assert {"semantic", "measurement"} <= set(contract.FIELD_KEYS["case_bundle"])
    assert contract.DESCRIPTOR_DIGEST == contract.stable_digest(contract.DESCRIPTOR)
    assert len(bytes.fromhex(contract.DESCRIPTOR_DIGEST)) == 32
    assert contract.DESCRIPTOR_DIGEST == \
        "522336628b0d344eea866596484cee1a235c321041867699d5c36e5efacfbbc4"


def test_t14_02_verified_binding_freezes_selected_lane_and_artifact_shas() -> None:
    binding = contract.T14_02_BINDING
    assert binding["selected_exact_workers"] == 1
    assert binding["candidate_root"].endswith("exact-scheduler-poc-v2")
    assert binding["verification_root"].endswith("exact-scheduler-poc-v2-verification")
    digest_fields = [name for name in binding if name.endswith(("sha256", "digest"))]
    assert digest_fields
    assert all(len(bytes.fromhex(binding[name])) == 32 for name in digest_fields)
    assert binding["candidate_complete_sha256"] == \
        "e074be8b302764d9a3333635846819dfc81971f2fb0597f4f9cd50f620e54a54"
    assert binding["verification_result_digest"] == \
        "9de54c0e0b6737251d27bf73210e6119c80b66f2ff37624a1bcec74afd2e2ddd"


def test_execution_and_claim_boundaries_are_exact() -> None:
    policy = contract.EXECUTION_POLICY
    assert policy["exact_workers"] == 1
    assert policy["proposal_threads"] == 8
    assert policy["case_concurrency"] == 1
    assert policy["proposal_child_cpus"] == 8
    assert policy["exact_child_cpus"] == 1
    assert policy["proposal_must_persist_before_exact_spawn"] is True
    assert policy["run_hard_limit_seconds"] == 7200
    assert policy["performance_limits"] == {
        "forward_proposal_seconds": 120.0,
        "reverse_proposal_seconds": 60.0,
        "proposal_process_rss_mb": 1536.0,
        "exact_stage_p95_seconds": 120.0,
        "exact_stage_max_seconds": 180.0,
        "end_to_end_p95_seconds": 180.0,
        "end_to_end_max_seconds": 300.0,
        "p95_method": "nearest-rank-ceiling",
        "p95_order_index_zero_based_for_60": 56,
    }
    assert policy["semantic_policy"] == "exact-timing-free"
    assert policy["resume"] is False
    assert policy["case_started_before_child_spawn"] is True
    assert policy["case_completed_after_bundle_fsync"] is True
    assert contract.CLAIMS == {
        "development_only": True,
        "cases_previously_exposed": True,
        "direct_raw_authority_accessed_by_this_run": False,
        "authority_derived_prerequisite_evidence_accessed_by_this_run": True,
        "forward_outcomes_accessed_by_this_run": False,
        "raw_authority_or_outcome_paths_accepted": False,
        "production_promotion_authorized": False,
    }


def test_successful_tree_has_exact_case_leaves_and_120_ordered_events() -> None:
    tree = contract.successful_tree()
    assert tree == contract.DESCRIPTOR["successful_tree"]
    events = sorted(path for path in tree if path.startswith("events/"))
    case_leaves = sorted(path for path in tree if path.startswith("cases/"))
    assert len(tree) == 4 + 120 + 180 + 2 + 1
    assert len(events) == 120 and len(case_leaves) == 180
    for ordinal, query_id in enumerate(contract.QUERY_IDS):
        assert f"events/{ordinal * 2:03d}-CASE_STARTED-{query_id}.json" in tree
        assert f"events/{ordinal * 2 + 1:03d}-CASE_COMPLETED-{query_id}.json" in tree
        prefix = f"cases/{ordinal:03d}-{query_id}"
        assert f"{prefix}/PROPOSAL.json" in tree
        assert f"{prefix}/EXACT-w1.json" in tree
        assert f"{prefix}/CASE.json" in tree
    assert "COMPLETE.json" in tree and "INCOMPLETE.json" not in tree
    assert contract.successful_directories() == tuple(sorted((
        "cases", "events", *(f"cases/{ordinal:03d}-{query_id}"
        for ordinal, query_id in enumerate(contract.QUERY_IDS))
    )))


def test_every_valid_case_prefix_shape_is_gap_free_and_terminal() -> None:
    for completed in range(61):
        variants = (None, "started", "proposal", "exact", "case") \
            if completed < 60 else (None,)
        for stage in variants:
            tree = contract.incomplete_tree(completed, trailing_stage=stage)
            assert "INCOMPLETE.json" in tree and "COMPLETE.json" not in tree
            expected_trailing_leaves = {
                None: 0, "started": 0, "proposal": 1, "exact": 2, "case": 3,
            }[stage]
            assert sum(path.startswith("cases/") for path in tree) == \
                completed * 3 + expected_trailing_leaves
            assert sum(path.startswith("events/") for path in tree) == \
                completed * 2 + int(stage is not None)
            directories = contract.incomplete_directories(
                completed, trailing_stage=stage)
            assert len(directories) == 2 + completed + int(
                stage in {"proposal", "exact", "case"})
            for ordinal, query_id in enumerate(contract.QUERY_IDS[:completed]):
                assert f"cases/{ordinal:03d}-{query_id}/CASE.json" in tree
    for count in range(5):
        tree = contract.incomplete_tree(0, foundation_count=count)
        assert sum(path in tree for path in contract.FOUNDATION_PATHS) == count
    for count in range(3):
        tree = contract.incomplete_tree(60, aggregate_count=count)
        assert sum(path in tree for path in contract.AGGREGATE_PATHS) == count


@pytest.mark.parametrize("kwargs", [
    {"completed_cases": -1}, {"completed_cases": 61},
    {"completed_cases": 60, "trailing_stage": "started"},
    {"completed_cases": 1, "trailing_stage": "unknown"},
    {"completed_cases": 1, "foundation_count": 3},
    {"completed_cases": 59, "aggregate_count": 1},
    {"completed_cases": 60, "aggregate_count": 3},
])
def test_invalid_prefix_shapes_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        contract.incomplete_tree(**kwargs)


def test_query_duplicate_and_order_mutations_are_rejected() -> None:
    duplicate = list(contract.QUERY_IDS); duplicate[1] = duplicate[0]
    reversed_ids = list(reversed(contract.QUERY_IDS))
    missing = list(contract.QUERY_IDS[:-1])
    for values in (duplicate, reversed_ids, missing):
        with pytest.raises(ValueError):
            contract.validate_query_ids(values)
        with pytest.raises(ValueError):
            contract.successful_tree(values)


def _semantic() -> dict[str, object]:
    return {key: ({"bound": True} if key not in {
        "ordinal", "query_id", "case_id", "semantic_digest",
    } else {
        "ordinal": 0, "query_id": contract.QUERY_IDS[0], "case_id": "case-0",
        "semantic_digest": "a" * 64,
    }[key]) for key in contract.FIELD_KEYS["case_semantic"]}


def test_semantic_projection_excludes_measurement_and_timing() -> None:
    bundle = {"ordinal": 0, "query_id": contract.QUERY_IDS[0], "case_id": "case-0",
        "semantic": _semantic(), "measurement": {"wall_seconds": 1.0},
        "created_at": "later", "measurement_digest": "b" * 64,
        "bundle_digest": "c" * 64}
    first = contract.semantic_projection(bundle)
    mutated = dict(bundle)
    mutated["measurement"] = {"wall_seconds": 999.0, "elapsed_seconds": 888.0}
    mutated["created_at"] = "much-later"; mutated["measurement_digest"] = "d" * 64
    assert contract.semantic_projection(mutated) == first
    assert set(contract.TIMING_FIELDS).isdisjoint(first["semantic"])
    forged = dict(bundle); forged["semantic"] = {**_semantic(), "elapsed_seconds": 1.0}
    with pytest.raises(ValueError, match="key set|timing"):
        contract.semantic_projection(forged)
    nested = dict(bundle); nested_semantic = _semantic()
    nested_semantic["certificate"] = {"result_digest": "e" * 64, "elapsed_seconds": 1.0}
    nested["semantic"] = nested_semantic
    with pytest.raises(ValueError, match="timing"):
        contract.semantic_projection(nested)


def test_canonical_digest_rejects_nonfinite_values() -> None:
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            contract.stable_digest({"nested": [value]})
