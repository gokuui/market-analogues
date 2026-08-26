from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from market_analogues.types import stable_hash


def _module():
    path = (
        Path(__file__).parents[1] / "experiments" / "m04r"
        / "m04r11_candidate_v2_contract.py"
    )
    spec = importlib.util.spec_from_file_location(
        "m04r11_candidate_v2_contract", path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _registry() -> dict:
    return json.loads((
        Path(__file__).parents[1] / "config" / "data" / "analogues" / "m04r10"
        / "nasdaq-untouched-authority-registry" / "query-registry.json"
    ).read_text())


def _manifests() -> tuple[dict, dict]:
    files = {"experiments/m04r/example.py": "a" * 64}
    implementation = {"files": files, "digest": stable_hash(files)}
    environment = {"python": "3.11", "numpy": "test"}
    environment["digest"] = stable_hash(environment)
    return implementation, environment


def _source_pack(module, artifact_dir: Path) -> dict:
    roots = module.expected_roots(artifact_dir)
    generation = (
        Path(roots["source_full_root"]) / "store" / "generations"
        / module.FROZEN_GENERATION_ID
    )
    row_count = 100
    overflow_count = 2
    return {
        "generation_id": module.FROZEN_GENERATION_ID,
        "source_full_root": roots["source_full_root"],
        "manifest_path": str((generation / "manifest.json").resolve()),
        "manifest_sha256": "b" * 64,
        "manifest_digest": module.FROZEN_GENERATION_ID,
        "provenance_digest": "c" * 64,
        "rows_file": "bound-rows.bin",
        "rows_path": str((generation / "bound-rows.bin").resolve()),
        "rows_bytes": row_count * module.FROZEN_ROW_BYTES,
        "rows_sha256": "d" * 64,
        "row_count": row_count,
        "row_bytes": module.FROZEN_ROW_BYTES,
        "overflow_file": "overflow-exact-fallback.bin",
        "overflow_path": str((generation / "overflow-exact-fallback.bin").resolve()),
        "overflow_bytes": overflow_count * module.FROZEN_OVERFLOW_ROW_BYTES,
        "overflow_sha256": "e" * 64,
        "overflow_count": overflow_count,
        "overflow_row_bytes": module.FROZEN_OVERFLOW_ROW_BYTES,
        "physical_rows": row_count + overflow_count,
        "active_pointer_absent": True,
    }


def test_frozen_role_partition_and_confirmatory_first_execution_order() -> None:
    module = _module()
    registry = _registry()
    table = module.derive_role_table(registry)
    assert module.validate_role_table(registry, table) == ()
    assert stable_hash(table) == module.FROZEN_ROLE_TABLE_DIGEST
    assert [
        (row["case_id"], row["query_episode_id"]) for row in table[:7]
    ] == list(module.FROZEN_EXPOSED_CASES)
    assert sum(
        row["performance_role"] == module.EXPOSED_PERFORMANCE_ROLE
        for row in table
    ) == 7
    assert sum(
        row["performance_role"] == module.CONFIRMATORY_PERFORMANCE_ROLE
        for row in table
    ) == 53
    assert all(row["recall_role"] == module.BLIND_RECALL_ROLE for row in table)
    execution = module.execution_query_ids(registry)
    assert execution == [row["query_episode_id"] for row in table[7:] + table[:7]]
    assert stable_hash(execution) == module.FROZEN_EXECUTION_ORDER_DIGEST

    changed = deepcopy(table)
    changed[7]["performance_role"] = module.EXPOSED_PERFORMANCE_ROLE
    assert module.validate_role_table(registry, changed)


def test_exact_v2_roots_bind_actual_resident_full_root(tmp_path: Path) -> None:
    module = _module()
    roots = module.expected_roots(tmp_path)
    assert module.validate_exact_roots(roots, tmp_path) == ()
    assert roots["candidate_root"].endswith("m04r11/candidate-pools-v2")
    assert roots["comparison_root"].endswith("m04r11/candidate-comparison-v2")
    assert roots["verification_root"].endswith(
        "m04r11/candidate-comparison-verification-v2"
    )
    assert roots["resident_full_root"] == str((
        Path("/dev/shm/market-analogues/m04r11-candidate-v2")
        / module.FROZEN_GENERATION_ID
    ).resolve())

    drifted = dict(roots)
    drifted["resident_full_root"] = "/dev/shm/unbound-copy"
    assert "exact v2 roots differ" in module.validate_exact_roots(
        drifted, tmp_path,
    )
    missing = dict(roots)
    missing.pop("verification_root")
    assert module.validate_exact_roots(missing, tmp_path) == ("root keys differ",)


def test_predecessor_failure_is_exact_terminal_unopened_evidence() -> None:
    module = _module()
    payload = json.loads((
        Path(__file__).parents[1] / "config" / "data" / "analogues" / "m04r11"
        / "candidate-pools-v1" / "FAILED.json"
    ).read_text())
    assert module.validate_predecessor_failure(payload) == ()
    assert payload["failure_digest"] == module.PREDECESSOR_FAILURE["failure_digest"]

    changed = deepcopy(payload)
    changed["claims_policy"]["recall_holdout_cases_still_blind"] = 59
    changed["failure_digest"] = stable_hash({
        key: value for key, value in changed.items()
        if key not in {"created_at", "failure_digest"}
    })
    assert module.validate_predecessor_failure(changed)
    changed = deepcopy(payload)
    changed["completed_case_summaries"][6]["passed"] = True
    assert module.validate_predecessor_failure(changed)


def test_semantic_and_performance_digests_are_separate() -> None:
    module = _module()
    semantic = {
        "schema_version": module.SEMANTIC_CASE_SCHEMA,
        "query_episode_id": "a" * 24,
        "candidates": [{"episode_id": "b" * 24}],
        "created_at": "first",
    }
    performance = {
        "schema_version": module.PERFORMANCE_ATTEMPT_SCHEMA,
        "query_episode_id": "a" * 24,
        "semantic_digest": module.semantic_case_digest(semantic),
        "resident_first_seconds": 34.0,
        "resident_repeat_seconds": 38.0,
        "created_at": "first",
    }
    semantic_digest = module.semantic_case_digest(semantic)
    performance_digest = module.performance_attempt_digest(performance)
    semantic["created_at"] = "second"
    performance["created_at"] = "second"
    assert module.semantic_case_digest(semantic) == semantic_digest
    assert module.performance_attempt_digest(performance) == performance_digest
    performance["resident_first_seconds"] = 35.0
    assert module.performance_attempt_digest(performance) != performance_digest
    assert module.semantic_case_digest(semantic) == semantic_digest
    semantic["candidates"].append({"episode_id": "c" * 24})
    assert module.semantic_case_digest(semantic) != semantic_digest


def test_exact_contract_build_and_strict_drift_rejection(tmp_path: Path) -> None:
    module = _module()
    registry = _registry()
    source_pack = _source_pack(module, tmp_path)
    implementation, environment = _manifests()
    contract = module.build_producer_contract(
        registry, tmp_path, source_pack=source_pack,
        implementation_manifest=implementation,
        environment_manifest=environment,
    )
    assert contract["schema_version"] == module.PRODUCER_CONTRACT_SCHEMA
    assert contract["roots"] == module.expected_roots(tmp_path)
    assert contract["scan_protocol"] == module.SCAN_PROTOCOL
    assert contract["scan_protocol"][
        "maximum_semantic_recovery_attempts_per_case"
    ] == 0
    assert contract["scan_protocol"]["outer_threads"] == 8
    assert contract["scan_protocol"]["maximum_in_flight_blocks"] == 8
    assert contract["performance_limits"]["worker_rss_mib"] == 1_536.0
    assert "worker_rss_at_most_1536_mib" in contract[
        "performance_attempt_gates"
    ]
    assert contract["marker_and_resume_policy"]["resume_authorized"] is False
    assert contract["marker_and_resume_policy"][
        "partial_ledger_resume_authorized"
    ] is False
    assert contract["artifact_schemas"] == module.ARTIFACT_SCHEMAS
    assert contract["run_ledger_event_schema"] == module.RUN_LEDGER_EVENT_SCHEMA
    assert contract["run_ledger_head_schema"] == module.RUN_LEDGER_HEAD_SCHEMA
    assert contract["incomplete_schema"] == module.INCOMPLETE_SCHEMA
    assert contract["run_complete_schema"] == module.RUN_COMPLETE_SCHEMA
    assert contract["case_bundle_schema"] == module.CASE_BUNDLE_SCHEMA
    assert contract["marker_and_resume_policy"]["case_bundles_directory"] == (
        "case-bundles"
    )
    assert contract["marker_and_resume_policy"]["case_bundle_filename"] == (
        "{execution_ordinal:03d}-{query_episode_id}.json"
    )
    assert contract["comparison_policy"]["minimum_retained_per_case"] == 19
    assert contract["comparison_policy"]["minimum_retained_aggregate"] == 1_188
    assert contract["comparison_policy"]["performance_pass_required"] is False
    assert contract["claims_policy"]["semantic_seal_independent_of_performance"]
    assert contract["contract_digest"] == module.producer_contract_digest(contract)
    assert module.validate_producer_contract(
        contract, registry, tmp_path, expected_source_pack=source_pack,
        expected_implementation_manifest=implementation,
        expected_environment_manifest=environment,
    ) == ()

    mutations = []
    changed = deepcopy(contract)
    changed["roots"]["resident_full_root"] = "/dev/shm/drift"
    mutations.append(changed)
    changed = deepcopy(contract)
    changed["scan_protocol"]["cache_advice"] = "POSIX_FADV_DONTNEED"
    mutations.append(changed)
    changed = deepcopy(contract)
    changed["role_table"][0]["recall_role"] = "opened"
    mutations.append(changed)
    changed = deepcopy(contract)
    changed["performance_limits"]["resident_repeat_seconds"] = 61.0
    mutations.append(changed)
    changed = deepcopy(contract)
    changed["predecessor_failure"]["resume_authorized"] = True
    mutations.append(changed)
    for mutation in mutations:
        mutation["contract_digest"] = module.producer_contract_digest(mutation)
        assert module.validate_producer_contract(
            mutation, registry, tmp_path, expected_source_pack=source_pack,
            expected_implementation_manifest=implementation,
            expected_environment_manifest=environment,
        )


def test_source_pack_physical_accounting_and_manifests_fail_closed(
    tmp_path: Path,
) -> None:
    module = _module()
    roots = module.expected_roots(tmp_path)
    source_pack = _source_pack(module, tmp_path)
    assert module.validate_source_pack_binding(source_pack, roots) == ()
    changed = deepcopy(source_pack)
    changed["physical_rows"] += 1
    assert module.validate_source_pack_binding(changed, roots)
    changed = deepcopy(source_pack)
    changed["rows_path"] = str(tmp_path / "other.bin")
    assert module.validate_source_pack_binding(changed, roots)

    implementation, environment = _manifests()
    invalid_implementation = deepcopy(implementation)
    invalid_implementation["files"]["../escape.py"] = "f" * 64
    invalid_implementation["digest"] = stable_hash(invalid_implementation["files"])
    with pytest.raises(ValueError, match="implementation manifest"):
        module.build_producer_contract(
            _registry(), tmp_path, source_pack=source_pack,
            implementation_manifest=invalid_implementation,
            environment_manifest=environment,
        )


def test_actual_ready_and_frozen_binding_schemas_are_distinct() -> None:
    module = _module()
    from market_analogues.resident_store import (
        CONTENT_SCHEMA_VERSION, READY_SCHEMA_VERSION,
    )

    assert READY_SCHEMA_VERSION == "m04r-resident-packed-store-ready-v2"
    assert module.RESIDENT_READY_SCHEMA == READY_SCHEMA_VERSION
    assert module.RESIDENT_CONTENT_SCHEMA == CONTENT_SCHEMA_VERSION
    assert module.RESIDENT_BINDING_SCHEMA == (
        "candidate-resident-ready-binding-v2"
    )
    assert module.RESIDENT_BINDING_SCHEMA != module.RESIDENT_READY_SCHEMA
    assert module.ARTIFACT_SCHEMAS["resident_ready"] == READY_SCHEMA_VERSION
    assert module.ARTIFACT_SCHEMAS["resident_content"] == CONTENT_SCHEMA_VERSION
    assert module.ARTIFACT_SCHEMAS["resident_binding"] == (
        module.RESIDENT_BINDING_SCHEMA
    )


def test_committed_preregistration_is_exact_path_and_not_self_certifying(
    tmp_path: Path,
) -> None:
    module = _module()
    registry = _registry()
    repository_root = tmp_path / "repository"
    artifact_dir = tmp_path / "artifacts"
    source_pack = _source_pack(module, artifact_dir)
    implementation, environment = _manifests()
    document = module.build_preregistration_document(
        registry, artifact_dir, repository_root, source_pack=source_pack,
        implementation_manifest=implementation,
        environment_manifest=environment,
    )
    frozen_path = module.expected_preregistration_path(repository_root)
    assert str(frozen_path).endswith(module.PREREGISTRATION_RELATIVE_PATH)
    frozen_path.parent.mkdir(parents=True)
    frozen_path.write_text(json.dumps(document, sort_keys=True))
    assert module.load_and_validate_preregistration(
        registry, artifact_dir, repository_root,
        expected_source_pack=source_pack,
        expected_implementation_manifest=implementation,
        expected_environment_manifest=environment,
    ) == document

    drifted_implementation = deepcopy(implementation)
    drifted_implementation["files"]["experiments/m04r/new.py"] = "f" * 64
    drifted_implementation["digest"] = stable_hash(
        drifted_implementation["files"]
    )
    attacker_document = module.build_preregistration_document(
        registry, artifact_dir, repository_root, source_pack=source_pack,
        implementation_manifest=drifted_implementation,
        environment_manifest=environment,
    )
    alternate_path = repository_root / "generated-at-launch.json"
    alternate_path.write_text(json.dumps(attacker_document, sort_keys=True))
    failures = module.validate_preregistration_document(
        attacker_document, registry, artifact_dir, repository_root,
        observed_path=alternate_path, expected_source_pack=source_pack,
        expected_implementation_manifest=implementation,
        expected_environment_manifest=environment,
    )
    assert "preregistration was not loaded from exact committed path" in failures
    assert any("current exact frozen v2 contract" in value for value in failures)
    with pytest.raises(ValueError, match="committed v2 preregistration differs"):
        module.load_and_validate_preregistration(
            registry, artifact_dir, repository_root,
            expected_source_pack=source_pack,
            expected_implementation_manifest=drifted_implementation,
            expected_environment_manifest=environment,
        )


@pytest.mark.parametrize("mutation", ["ready_schema", "marker_policy"])
def test_preregistration_rejects_self_consistent_schema_and_marker_drift(
    tmp_path: Path, mutation: str,
) -> None:
    module = _module()
    registry = _registry()
    repository_root = tmp_path / "repository"
    artifact_dir = tmp_path / "artifacts"
    source_pack = _source_pack(module, artifact_dir)
    implementation, environment = _manifests()
    document = module.build_preregistration_document(
        registry, artifact_dir, repository_root, source_pack=source_pack,
        implementation_manifest=implementation,
        environment_manifest=environment,
    )
    changed = deepcopy(document)
    if mutation == "ready_schema":
        changed["artifact_schemas"]["resident_ready"] = "substitute-ready-v1"
        changed["producer_contract"]["artifact_schemas"][
            "resident_ready"
        ] = "substitute-ready-v1"
        changed["producer_contract"]["resident_ready_schema"] = (
            "substitute-ready-v1"
        )
    else:
        changed["marker_and_resume_policy"]["resume_authorized"] = True
        changed["producer_contract"]["marker_and_resume_policy"][
            "resume_authorized"
        ] = True
    changed["producer_contract"]["contract_digest"] = (
        module.producer_contract_digest(changed["producer_contract"])
    )
    changed["producer_contract_digest"] = changed["producer_contract"][
        "contract_digest"
    ]
    changed["preregistration_digest"] = module.preregistration_digest(changed)
    failures = module.validate_preregistration_document(
        changed, registry, artifact_dir, repository_root,
        observed_path=module.expected_preregistration_path(repository_root),
        expected_source_pack=source_pack,
        expected_implementation_manifest=implementation,
        expected_environment_manifest=environment,
    )
    assert failures
    assert any(
        text in " ".join(failures)
        for text in ("artifact schemas differ", "marker/no-resume policy differs")
    )
