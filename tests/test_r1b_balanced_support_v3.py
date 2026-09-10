from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from experiments.m04r import m04r14_r1b_balanced_partition_v3 as partition
from experiments.m04r import m04r14_r1b_balanced_support_v3 as support
from market_analogues.types import stable_hash


def test_support_summary_uses_frozen_episode_and_link_denominators() -> None:
    cohort = {"a": (0, 1, 2), "b": (1, 2)}
    result = support._summary("x", {"a": 4096, "b": 17}, cohort)
    assert result == {
        "design": "x", "supported_episodes": 1, "cohort_episodes": 2,
        "episode_coverage": 0.5, "supported_links": 3, "cohort_links": 5,
        "link_coverage": 0.6, "passes": False,
    }


def test_secondary_k_cannot_rescue_primary_k8() -> None:
    designs = [
        {"role": "primary", "global_structure_cells": 8, "passes": False},
        {"role": "secondary_sensitivity", "global_structure_cells": 12, "passes": True},
        {"role": "secondary_sensitivity", "global_structure_cells": 16, "passes": True},
    ]
    assert support._primary_support_pass(designs) is False
    with pytest.raises(support.BalancedSupportRunError, match="primary K8"):
        support._primary_support_pass(designs[1:])


def test_each_design_summary_uses_its_own_k_details() -> None:
    cohort = {"a": (0,)}
    eight = support._design_summary(
        8, {"a": 4096}, cohort, {"retained_k": 8, "leaf_sizes": [408, 409]}, 201,
    )
    sixteen = support._design_summary(
        16, {"a": 4096}, cohort, {"retained_k": 16, "leaf_sizes": [204, 205]}, 341,
    )
    assert (eight["retained_structure_cells"], eight["leaf_size_min"], eight["crossed_cells"]) == (8, 408, 201)
    assert (sixteen["retained_structure_cells"], sixteen["leaf_size_min"], sixteen["crossed_cells"]) == (16, 204, 341)


def test_partition_contract_is_exact_and_primary_is_k8() -> None:
    contract = partition._partition_contract()
    assert contract["requested_k"] == [8, 12, 16]
    assert contract["primary_k"] == 8
    assert contract["secondary_k"] == [12, 16]
    assert contract["parallel_reconstruction_workers"] == 12
    assert contract["no_merge_retry_or_coarsening"] is True
    assert "math.fsum" in contract["projection"]
    assert "relative eigengap" in contract["fallback"]


def test_partition_run_refuses_nonfrozen_worker_count_before_inputs(tmp_path: Path) -> None:
    with pytest.raises(partition.BalancedPartitionRunError, match="exactly 12 workers"):
        partition.run(tmp_path, workers=8)


def test_partition_run_has_no_candidate_or_support_authority_read() -> None:
    source = inspect.getsource(partition.run)
    for forbidden in ("CASES", "R1A_RESULT", "V2_RESULT", "V2_COHORT", "matched_support"):
        assert forbidden not in source


def test_partition_publish_is_create_only(tmp_path: Path) -> None:
    first = tmp_path / "first"; first.mkdir(); (first / "x").write_text("one")
    destination = tmp_path / "destination"
    partition._publish(first, destination)
    assert (destination / "x").read_text() == "one"
    second = tmp_path / "second"; second.mkdir(); (second / "x").write_text("two")
    with pytest.raises(partition.BalancedPartitionRunError, match="output exists"):
        partition._publish(second, destination)
    assert (destination / "x").read_text() == "one"


def test_html_states_support_only_boundary() -> None:
    html = support._html({
        "status": "support_inadequate",
        "designs": [{
            "design": "session_21_structure_8", "role": "primary",
            "retained_structure_cells": 8, "leaf_size_min": 408, "leaf_size_max": 409,
            "episode_coverage": 0.5, "link_coverage": 0.6, "passes": False,
        }],
    })
    assert "support statistics only" in html
    assert "No R1-B cohesion/specificity statistic" in html
    assert "K12/K16 are secondary only" in html


def test_support_publication_binds_sidecars_and_is_create_only(tmp_path: Path) -> None:
    state = {
        "schema_version": support.SCHEMA, "status": "support_inadequate",
        "passed": False, "designs": [],
    }
    result = support._publish_result(tmp_path, state, [{"episode_id": "e"}])
    root = tmp_path / partition.SUPPORT_OUTPUT
    assert {path.name for path in root.iterdir()} == {"RESULT.json", "SUPPORT.json", "report.html"}
    assert result["result_digest"] == stable_hash({
        key: value for key, value in result.items()
        if key not in {"result_digest", "created_at"}
    })
    assert partition._sha(root / "SUPPORT.json") == result["support_sha256"]
    assert partition._sha(root / "report.html") == result["report_sha256"]
    with pytest.raises(support.BalancedSupportRunError, match="support output exists"):
        support._publish_result(tmp_path, dict(state), [])


def test_failure_publication_preserves_exact_reason(tmp_path: Path) -> None:
    result = support._publish_failure(
        tmp_path, {"preregistration_digest": "p"}, "h1",
        "reproducibility_failed", "worker mismatch", "partition-digest",
    )
    assert result["status"] == "reproducibility_failed"
    assert result["reason"] == "worker mismatch"
    assert result["partition_result_digest"] == "partition-digest"
    assert result["passed"] is False


def _structural_fixture() -> tuple[dict, dict, dict, dict, list[str]]:
    query_ids = [f"q-{index:04d}" for index in range(3270)]
    assignments = [{"query_episode_id": value, "labels": {}} for value in query_ids]
    partitions = {}
    split = {
        "axis_digest": "a", "axis_mode": "leading_pca", "boundary_margin_hex": "0x1p+0",
        "leading_eigenvalue_hex": "0x1p+0", "left_boundary_hex": "0x0p+0",
        "left_child_path": "0", "left_rows": 1, "left_target_leaves": 1,
        "member_ids_digest": "m", "path": "root", "pivot_feature": 0,
        "relative_eigengap_hex": "0x1p+0", "right_boundary_hex": "0x1p+0",
        "right_child_path": "1", "right_rows": 1, "right_target_leaves": 1,
        "rows": 2, "target_leaves": 2,
    }
    for k in (8, 12, 16):
        labels = [index * k // 3270 for index in range(3270)]
        for index, label in enumerate(labels):
            assignments[index]["labels"][str(k)] = label
        sizes = [labels.count(label) for label in range(k)]
        partitions[str(k)] = {
            "status": "partition_valid", "requested_k": k, "retained_k": k,
            "leaf_paths": [f"leaf-{value}" for value in range(k)], "leaf_sizes": sizes,
            "membership_digest": stable_hash([
                [query_ids[index], label] for index, label in enumerate(labels)
            ]),
            "splits": [dict(split) for _ in range(k - 1)],
        }
    payload = {
        "schema_version": "m04r14-r1b-balanced-partitions-v1",
        "assignments": assignments, "partitions": partitions,
    }
    result = {
        "adequacy_labels_authorized": False, "b2_execution_authorized": False,
        "candidate_or_eligibility_inputs_accessed": False, "created_at": "time",
        "partitions_sha256": "p", "passed": True, "predictive_claim_authorized": False,
        "preregistration_commit": "h1", "preregistration_digest": "pre", "primary_k": 8,
        "production_promotion_authorized": False, "queries": 3270,
        "r1b_statistics_opened": False, "real_forward_outcomes_accessed": False,
        "registry_digest": "registry", "result_digest": "result",
        "schema_version": partition.SCHEMA, "secondary_k": [12, 16],
        "serial_parallel_partition_digest_identical": True,
        "serial_parallel_reconstruction_identical": True,
        "serial_parallel_transform_digest_identical": True,
        "status": "partition_valid", "transform_sha256": "t",
    }
    transform = {
        "constant_columns": [], "integer_midrank_digest": "i",
        "query_audits": [{
            "benchmark_prefix_digest": "b", "query_episode_id": value,
            "query_representation_digest": "r", "stock_prefix_digest": "s",
        } for value in query_ids],
        "query_ids_digest": partition.id_digest(query_ids), "raw_vector_digest": "raw",
        "schema_version": "m04r14-r1b-balanced-transform-v1", "shape": [3270, 141],
        "transformed_digest": "z",
    }
    prereg = {"preregistration_digest": "pre"}
    return result, transform, payload, prereg, query_ids


def test_partition_artifact_validator_recomputes_membership_and_closure() -> None:
    result, transform, payload, prereg, query_ids = _structural_fixture()
    support._validate_partition_artifacts(
        result, transform, payload, prereg, "h1", {"registry_digest": "registry"}, query_ids,
    )
    payload["assignments"][0]["labels"]["8"] = 1
    with pytest.raises(support.BalancedSupportRunError, match="leaf accounting|membership"):
        support._validate_partition_artifacts(
            result, transform, payload, prereg, "h1", {"registry_digest": "registry"}, query_ids,
        )
