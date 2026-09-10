from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments.m04r import m04r14_r1b_balanced_support_v3_verifier as verify
from market_analogues.types import stable_hash


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _synthetic_output_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(verify, "PARTITION_OUTPUT", Path("partition"))
    monkeypatch.setattr(verify, "SUPPORT_OUTPUT", Path("support"))
    partition_root = tmp_path / "partition"
    support_root = tmp_path / "support"
    query_ids = tuple(f"q-{index:04d}" for index in range(3270))
    transform = {"synthetic": "independently reconstructed transform"}
    assignments = [
        {"query_episode_id": query_id,
         "labels": {str(k): index % k for k in verify.K_VALUES}}
        for index, query_id in enumerate(query_ids)
    ]
    partitions = {
        "schema_version": "synthetic",
        "assignments": assignments,
        "partitions": {
            str(k): {
                "status": "partition_valid", "requested_k": k,
                "retained_k": k,
                "leaf_sizes": [3270 // k + (1 if index < 3270 % k else 0)
                               for index in range(k)],
                "splits": [{"path": "root", "k": k}],
            } for k in verify.K_VALUES
        },
    }
    _write_json(partition_root / "TRANSFORM.json", transform)
    _write_json(partition_root / "PARTITIONS.json", partitions)
    partition_state = {
        "schema_version": "m04r14-r1b-balanced-partition-v3",
        "status": "partition_valid", "passed": True,
        "preregistration_commit": "h1", "preregistration_digest": "prereg",
        "registry_digest": "registry", "queries": 3270,
        "serial_parallel_reconstruction_identical": True,
        "serial_parallel_transform_digest_identical": True,
        "serial_parallel_partition_digest_identical": True,
        "primary_k": 8, "secondary_k": [12, 16],
        "real_forward_outcomes_accessed": False,
        "candidate_or_eligibility_inputs_accessed": False,
        "r1b_statistics_opened": False, "b2_execution_authorized": False,
        "adequacy_labels_authorized": False, "predictive_claim_authorized": False,
        "production_promotion_authorized": False,
        "transform_sha256": verify._sha(partition_root / "TRANSFORM.json"),
        "partitions_sha256": verify._sha(partition_root / "PARTITIONS.json"),
    }
    partition_result = {
        **partition_state, "result_digest": stable_hash(partition_state),
        "created_at": "synthetic",
    }
    _write_json(partition_root / "RESULT.json", partition_result)
    cohort = {"episode": tuple(range(12))}
    eligible = {"episode": tuple(range(3270))}
    base_cells = tuple((index % 2,) for index in range(3270))
    base_support = {"episode": 4096}
    designs = []
    n1 = {}
    for k in verify.K_VALUES:
        crossed = tuple((*base_cells[index], index % k) for index in range(3270))
        support = verify._matched_support(eligible["episode"], cohort["episode"], crossed)
        n1[str(k)] = support
        designs.append(verify._summary(
            k, {"episode": support}, cohort, partitions["partitions"][str(k)],
            len(set(crossed)),
        ))
    support_rows = [{
        "episode_id": "episode", "observed_inbound_queries": 12,
        "eligible_queries": 3270, "n0_support": 4096, "n1_support": n1,
    }]
    _write_json(support_root / "SUPPORT.json", support_rows)
    support_state = {
        "schema_version": "m04r14-r1b-balanced-support-v3",
        "status": "support_pass_pending_independent_verification", "passed": True,
        "preregistration_commit": "h1", "preregistration_digest": "prereg",
        "partition_result_digest": partition_result["result_digest"],
        "v2_result_digest": "v2",
        "v2_verification_digest": verify.V2_VERIFICATION_DIGEST,
        "inventory": {"queries": 3270, "cohort_episodes": 369, "cohort_links": 2865},
        "designs": designs,
        "partition_status": {str(k): "partition_valid" for k in verify.K_VALUES},
        "primary_k": 8, "secondary_k": [12, 16],
        "n0_matching_design_support_verified": True,
        "primary_support_gate_passed": True,
        "passed_meaning": "producer support gate only; independent verification pending",
        "b2_execution_authorized": False, "real_forward_outcomes_accessed": False,
        "r1b_statistics_opened": False, "adequacy_labels_authorized": False,
        "predictive_claim_authorized": False, "production_promotion_authorized": False,
        "support_sha256": verify._sha(support_root / "SUPPORT.json"),
        "report_sha256": "pending",
    }
    provisional = {**support_state, "result_digest": "pending", "created_at": "synthetic"}
    report = verify._expected_html(provisional)
    (support_root / "report.html").write_text(report)
    support_state["report_sha256"] = verify._sha(support_root / "report.html")
    support_result = {
        **support_state, "result_digest": stable_hash(support_state),
        "created_at": "synthetic",
    }
    # The report renders only fields unaffected by replacing the provisional digest.
    assert verify._expected_html(support_result) == report
    _write_json(support_root / "RESULT.json", support_result)
    return {
        "arguments": (
            tmp_path, {"preregistration_digest": "prereg"}, "h1",
            {"registry_digest": "registry"}, query_ids, transform, partitions,
            cohort, eligible, base_cells, base_support, {"result_digest": "v2"},
        ),
        "partition_root": partition_root, "support_root": support_root,
    }


def test_strict_loader_rejects_duplicates_nonfinite_and_wrong_shape(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"x": 1, "x": 2}')
    with pytest.raises(verify.BalancedSupportVerificationError, match="duplicate JSON key"):
        verify._load(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"x": NaN}')
    with pytest.raises(verify.BalancedSupportVerificationError, match="non-finite JSON"):
        verify._load(nonfinite)
    wrong = tmp_path / "wrong.json"
    wrong.write_text("[]")
    with pytest.raises(verify.BalancedSupportVerificationError, match="JSON dict required"):
        verify._load(wrong)


def test_independent_midrank_transform_exact_ties_and_constants() -> None:
    values = np.asarray([
        [4.0, 7.0, 1.0], [1.0, 7.0, 2.0], [1.0, 7.0, 3.0], [9.0, 7.0, 4.0],
    ])
    transformed, integers, constants = verify._independent_midranks(values)
    np.testing.assert_array_equal(integers[:, 0], [6, 3, 3, 8])
    np.testing.assert_array_equal(integers[:, 1], [5, 5, 5, 5])
    np.testing.assert_array_equal(integers[:, 2], [2, 4, 6, 8])
    np.testing.assert_allclose(transformed[:, 0], [1 / 3, -2 / 3, -2 / 3, 1])
    np.testing.assert_array_equal(transformed[:, 1], 0.0)
    assert constants == (1,)


@pytest.mark.parametrize("leaves", [2, 3, 4, 5])
def test_independent_recursive_partition_is_balanced_and_deterministic(leaves: int) -> None:
    rng = np.random.default_rng(20260910)
    raw = rng.normal(size=(23, 7))
    raw[5] = raw[4]
    transformed, integers, _ = verify._independent_midranks(raw)
    ids = tuple(f"q-{index:02d}" for index in range(len(raw)))
    first = verify._independent_partition(transformed, integers, ids, leaves)
    second = verify._independent_partition(transformed, integers, ids, leaves)
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1:] == second[1:]
    assert set(first[2]) <= {len(raw) // leaves, (len(raw) + leaves - 1) // leaves}
    assert len(first[3]) == leaves - 1


def test_independent_partition_falls_back_deterministically_on_repeated_eigenvalue() -> None:
    raw = np.asarray([
        [-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0],
    ])
    transformed, integers, _ = verify._independent_midranks(raw)
    labels, _paths, sizes, splits = verify._independent_partition(
        transformed, integers, ("a", "b", "c", "d"), 2,
    )
    assert splits[0]["axis_mode"] == "exact_variance_fallback"
    assert splits[0]["pivot_feature"] == 0
    assert sorted(sizes) == [2, 2]
    assert set(map(int, labels)) == {0, 1}


def test_independent_matched_support_reconstructs_product_of_combinations() -> None:
    cells = ("a", "a", "a", "b", "b", "c")
    assert verify._matched_support((0, 1, 2, 3, 4, 5), (0, 1, 3), cells) == 6
    assert verify._matched_support((0, 1, 2, 3, 4, 5), (), cells) == 1
    assert verify._matched_support(tuple(range(100)), tuple(range(10)), ("a",) * 100) == 4096
    with pytest.raises(verify.BalancedSupportVerificationError, match="duplicate"):
        verify._matched_support((0, 0), (0,), ("a",))
    with pytest.raises(verify.BalancedSupportVerificationError, match="ineligible"):
        verify._matched_support((0,), (1,), ("a", "b"))


def test_design_roles_make_k8_primary_and_larger_k_sensitivity_only() -> None:
    cohort = {"a": (0, 1), "b": (1,)}
    detail = {"leaf_sizes": [2, 2]}
    primary = verify._summary(8, {"a": 4096, "b": 1}, cohort, detail, 4)
    sensitivity = verify._summary(12, {"a": 4096, "b": 4096}, cohort, detail, 4)
    assert primary["role"] == "primary" and primary["passes"] is False
    assert sensitivity["role"] == "secondary_sensitivity" and sensitivity["passes"] is True


def test_receipt_preserves_pending_scientific_authority() -> None:
    designs = [{"passes": True}, {"passes": True}, {"passes": True}]
    state = verify._receipt_state(
        {"preregistration_digest": "p"}, "h1", "verifier", {
            "partition_result_digest": "partition", "result_digest": "support",
        }, designs, {"verifier.py": "hash"},
    )
    assert state["passed"] is True
    assert state["primary_k8_support_gate_passed"] is True
    assert state["b2_contract_freeze_may_proceed"] is True
    assert state["b2_scientific_execution_authorized"] is False
    assert state["predictive_claim_authorized"] is False
    assert state["passed_meaning"] == "integrity and matching-support decision only"


def test_receipt_does_not_let_secondary_designs_rescue_failed_k8() -> None:
    state = verify._receipt_state(
        {"preregistration_digest": "p"}, "h1", "verifier", {
            "partition_result_digest": "partition", "result_digest": "support",
        }, [{"passes": False}, {"passes": True}, {"passes": True}],
        {"verifier.py": "hash"},
    )
    assert state["passed"] is True
    assert state["primary_k8_support_gate_passed"] is False
    assert state["support_decision_verified"] is True
    assert state["b2_contract_freeze_may_proceed"] is False


def test_verifier_does_not_import_producer_or_canonical_math_kernels() -> None:
    source = inspect.getsource(verify)
    for forbidden in (
        "from experiments.m04r import m04r14_r1b_balanced_partition_v3",
        "from experiments.m04r import m04r14_r1b_balanced_support_v3",
        "from market_analogues.balanced_partition import",
        "from market_analogues.adequacy_support import",
    ):
        assert forbidden not in source


def test_atomic_receipt_publication_is_create_only(tmp_path: Path) -> None:
    output = tmp_path / "receipt"
    verify._atomic_publish(output, {"passed": True})
    stored = json.loads((output / "VERIFIED.json").read_text())
    assert stored == {"passed": True}
    with pytest.raises(verify.BalancedSupportVerificationError, match="create-only"):
        verify._atomic_publish(output, {"passed": False})
    assert json.loads((output / "VERIFIED.json").read_text()) == {"passed": True}


def test_verification_digest_projection_excludes_only_time_and_digest() -> None:
    state = {"passed": True, "gates": {"x": True}}
    payload = {
        **state, "verification_digest": stable_hash(state), "created_at": "now",
    }
    assert payload["verification_digest"] == stable_hash({
        key: value for key, value in payload.items()
        if key not in {"verification_digest", "created_at"}
    })


def test_frozen_partition_contract_is_complete_and_exact() -> None:
    contract = verify._frozen_partition_contract()
    assert set(contract) == {
        "column_order", "fallback", "no_merge_retry_or_coarsening",
        "parallel_reconstruction_workers", "primary_k", "projection",
        "query_order", "rank", "requested_k", "scatter", "secondary_k", "shape",
    }
    assert contract["parallel_reconstruction_workers"] == 12
    assert contract["primary_k"] == 8 and contract["secondary_k"] == [12, 16]
    assert "exact float64 ties" in contract["rank"]
    assert "single-thread BLAS" in contract["scatter"]
    assert "math.fsum" in contract["projection"]


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("partition_contract", "rank", "changed"),
        ("partition_contract", "parallel_reconstruction_workers", 11),
        ("execution", "publication", "changed"),
        ("environment", "partition_blas_threads", 2),
        ("support_contract", "primary", "changed"),
    ],
)
def test_preregistration_refuses_semantic_mutation(
    monkeypatch: pytest.MonkeyPatch, section: str, field: str, value: Any,
) -> None:
    repository = Path(verify.__file__).resolve().parents[2]
    original_load = verify._load
    prereg = original_load(repository / verify.PREREGISTRATION)
    prereg[section][field] = value
    prereg["preregistration_digest"] = stable_hash({
        key: item for key, item in prereg.items() if key != "preregistration_digest"
    })

    def changed_load(path: Path, expected: type = dict) -> Any:
        if path == repository / verify.PREREGISTRATION:
            return prereg
        return original_load(path, expected)

    monkeypatch.setattr(verify, "_load", changed_load)
    monkeypatch.setattr(
        verify, "_preregistration_h1",
        lambda _repository, _prereg: "8890cff2a31b309db7816f8cb4b4ff73edcbbacb",
    )
    with pytest.raises(verify.BalancedSupportVerificationError,
                       match="preregistration semantics differ"):
        verify._validate_preregistration(repository)


def test_preregistration_refuses_runtime_hash_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Path(verify.__file__).resolve().parents[2]
    original_load = verify._load
    prereg = original_load(repository / verify.PREREGISTRATION)
    runtime = next(iter(prereg["runtime_sha256"]))
    prereg["runtime_sha256"][runtime] = "0" * 64
    prereg["preregistration_digest"] = stable_hash({
        key: item for key, item in prereg.items() if key != "preregistration_digest"
    })

    def changed_load(path: Path, expected: type = dict) -> Any:
        if path == repository / verify.PREREGISTRATION:
            return prereg
        return original_load(path, expected)

    monkeypatch.setattr(verify, "_load", changed_load)
    monkeypatch.setattr(
        verify, "_preregistration_h1",
        lambda _repository, _prereg: "8890cff2a31b309db7816f8cb4b4ff73edcbbacb",
    )
    with pytest.raises(verify.BalancedSupportVerificationError, match="runtime hash differs"):
        verify._validate_preregistration(repository)


def test_preregistration_h1_refuses_changed_h0_lineage() -> None:
    repository = Path(verify.__file__).resolve().parents[2]
    prereg = verify._load(repository / verify.PREREGISTRATION)
    prereg["implementation_commit"] = "0" * 40
    with pytest.raises(verify.BalancedSupportVerificationError,
                       match="preregistration H1 lineage differs"):
        verify._preregistration_h1(repository, prereg)


def test_exact_output_reconstruction_accepts_complete_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _synthetic_output_fixture(tmp_path, monkeypatch)
    result, rows, designs = verify._validate_outputs(*fixture["arguments"])
    assert result["status"] == "support_pass_pending_independent_verification"
    assert rows[0]["n1_support"] == {"8": 4096, "12": 4096, "16": 4096}
    assert designs[0]["role"] == "primary" and designs[0]["passes"] is True


@pytest.mark.parametrize(
    ("target", "mutation", "message"),
    [
        ("PARTITIONS.json", "assignment", "partition reconstruction differs"),
        ("PARTITIONS.json", "split", "partition reconstruction differs"),
        ("PARTITION_RESULT.json", "digest", "partition RESULT differs"),
        ("SUPPORT.json", "support", "support-row reconstruction differs"),
        ("RESULT.json", "status", "support RESULT reconstruction differs"),
        ("RESULT.json", "digest", "support RESULT reconstruction differs"),
        ("report.html", "report", "support report reconstruction differs"),
    ],
)
def test_exact_output_reconstruction_refuses_every_artifact_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    target: str, mutation: str, message: str,
) -> None:
    fixture = _synthetic_output_fixture(tmp_path, monkeypatch)
    partition_root = fixture["partition_root"]
    support_root = fixture["support_root"]
    if target == "PARTITIONS.json":
        path = partition_root / target
        value = json.loads(path.read_text())
        if mutation == "assignment":
            value["assignments"][0]["labels"]["8"] = 7
        else:
            value["partitions"]["8"]["splits"][0]["path"] = "changed"
        _write_json(path, value)
    elif target == "PARTITION_RESULT.json":
        path = partition_root / "RESULT.json"
        value = json.loads(path.read_text())
        value["result_digest"] = "changed"
        _write_json(path, value)
    elif target == "SUPPORT.json":
        path = support_root / target
        value = json.loads(path.read_text())
        value[0]["n0_support"] = 1
        _write_json(path, value)
    elif target == "RESULT.json":
        path = support_root / target
        value = json.loads(path.read_text())
        value[mutation] = "changed"
        _write_json(path, value)
    else:
        (support_root / target).write_text("changed")
        result_path = support_root / "RESULT.json"
        value = json.loads(result_path.read_text())
        value["report_sha256"] = verify._sha(support_root / target)
        value["result_digest"] = stable_hash({
            key: item for key, item in value.items()
            if key not in {"result_digest", "created_at"}
        })
        _write_json(result_path, value)
    with pytest.raises(verify.BalancedSupportVerificationError, match=message):
        verify._validate_outputs(*fixture["arguments"])
