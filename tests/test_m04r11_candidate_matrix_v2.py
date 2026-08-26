from __future__ import annotations

import base64
import importlib.util
import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import subprocess

import pytest

from market_analogues.packed_bound_search import PackedBoundQuery
from market_analogues.resident_store import READY_SCHEMA_VERSION
from market_analogues.types import stable_hash


def _module():
    path = (
        Path(__file__).parents[1] / "experiments" / "m04r"
        / "m04r11_candidate_matrix_v2.py"
    )
    spec = importlib.util.spec_from_file_location(
        "m04r11_candidate_matrix_v2", path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Report:
    schema_version = "proposal-v1"
    generation_id = "g"
    query_episode_id = "q"
    rows_scanned = 10
    eligible_rows = 8
    eligible_main_rows = 7
    eligible_overflow_rows = 1
    route_counts = {"composite": 1}
    route_quotas = {"composite": 1_000}
    candidates = ()
    candidate_digest = "candidate"
    result_digest = "result"
    block_rows = 1
    block_order = "forward"
    elapsed_seconds = 12.5
    peak_rss_mb = 99.0


def test_semantic_report_excludes_all_timing_and_execution_measurements() -> None:
    module = _module()
    observed = module._report_semantics(_Report())
    assert "elapsed_seconds" not in observed
    assert "peak_rss_mb" not in observed
    assert "block_rows" not in observed
    assert "block_order" not in observed


def test_frozen_resident_scan_order_threads_and_no_cache_advice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    calls = []

    def scan(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(module, "scan_packed_bound_proposals_threaded", scan)
    query = object()
    reports = module._run_resident_scans(
        Path("/resident/store"), query, {"composite": 1_000}, "provenance",
    )
    assert len(reports) == 3
    assert [
        (call[1]["block_rows"], call[1]["block_order"])
        for call in calls
    ] == [(4_096, "forward"), (4_097, "reverse"), (4_093, "forward")]
    assert all(call[1]["threads"] == 8 for call in calls)
    assert all(call[1]["verify_content"] is False for call in calls)
    assert all("fadvise" not in str(call).lower() for call in calls)


def test_actual_resident_ready_schema_and_binding_are_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    ready_path = tmp_path / "READY.json"
    ready = {
        "schema_version": READY_SCHEMA_VERSION,
        "content": {"generation_id": "g"},
        "content_digest": stable_hash({"generation_id": "g"}),
        "ready_digest": "r" * 64,
        "seal_digest": "s" * 64,
        "seal": {
            "generation_id": "g",
            "provenance_digest": "p",
            "mirror_store_root": "/resident/store",
            "storage_class": "tmpfs-resident-generation-v1",
            "latency_scope": "resident-ready query; not durable-storage disk-cold",
            "query_specific_inputs_used": False,
            "outcomes_or_labels_used": False,
        },
    }
    ready_path.write_text(json.dumps(ready))
    contract = {
        "contract_digest": "c" * 64,
        "resident_ready_schema": READY_SCHEMA_VERSION,
        "generation_id": "g",
        "resident_policy": {"reserve_bytes": 1},
    }
    observation = {
        "schema_version": READY_SCHEMA_VERSION,
        "content_digest": ready["content_digest"],
        "ready_digest": ready["ready_digest"],
        "seal_digest": ready["seal_digest"],
        "ready_file_sha256": sha256(ready_path.read_bytes()).hexdigest(),
        "ready_identity": {}, "file_identity_lease": {},
    }
    monkeypatch.setattr(module, "_ready_observation", lambda path: observation)
    validation_deterministic = {
        "ready_digest": ready["ready_digest"],
        "content_digest": ready["content_digest"],
        "seal_digest": ready["seal_digest"],
        "reserve_bytes": 1,
    }
    validation = {
        **validation_deterministic,
        "observation_digest": stable_hash(validation_deterministic),
    }
    binding = module._resident_binding(
        ready, contract_digest=contract["contract_digest"],
        ready_path=ready_path, validation_observation=validation,
    )
    assert base64.b64decode(binding["resident_ready_bytes_base64"]) == ready_path.read_bytes()
    assert binding["resident_ready_payload"] == ready
    assert module._validate_resident_binding(binding, contract, ready) == ()
    contract["resident_ready_schema"] = "invented-ready-schema"
    assert module._validate_resident_binding(binding, contract, ready)


def test_hash_chained_ledger_is_ordered_bounded_and_tamper_evident(
    tmp_path: Path,
) -> None:
    module = _module()
    first = module._append_ledger_event(tmp_path, "contract", "run_started", {"x": 1})
    second = module._append_ledger_event(tmp_path, "contract", "case_completed", {"x": 2})
    events, head = module._load_ledger(tmp_path, "contract")
    assert [row["event_index"] for row in events] == [0, 1]
    assert second["previous_event_digest"] == first["event_digest"]
    assert head["event_count"] == 2
    path = tmp_path / "ledger" / "events" / "000000.json"
    tampered = json.loads(path.read_text())
    tampered["details"]["x"] = 99
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="chain"):
        module._load_ledger(tmp_path, "contract")


def _registry_and_contract(module):
    cases = [{
        "case_id": f"case-{index}", "episode_id": f"{index:024x}",
    } for index in range(60)]
    registry = {"cases_data": cases}
    roles = [{
        "ordinal": index, "case_id": row["case_id"],
        "query_episode_id": row["episode_id"],
        "performance_role": (
            "exposed_recovery_regression" if index < 7
            else module.CONFIRMATORY_PERFORMANCE_ROLE
        ),
        "recall_role": "blind_primary",
    } for index, row in enumerate(cases)]
    execution = [row["episode_id"] for row in cases[7:] + cases[:7]]
    return registry, {
        "contract_digest": "contract", "role_table": roles,
        "execution_query_ids": execution,
    }


def test_registry_ordered_semantics_and_timing_failure_are_independent() -> None:
    module = _module()
    registry, contract = _registry_and_contract(module)
    semantic = {}
    attempts = {}
    for index, case in enumerate(registry["cases_data"]):
        query_id = case["episode_id"]
        semantic[query_id] = {
            "query_episode_id": query_id,
            "semantic_digest": f"s{index}", "passed": True,
            "real_forward_outcomes_accessed": False,
        }
        role = contract["role_table"][index]["performance_role"]
        passed = index != 20
        attempts[query_id] = {
            "query_episode_id": query_id, "performance_digest": f"p{index}",
            "performance_role": role, "attempt_ordinal": 1,
            "ready_start": {"ready_digest": "ready"},
            "ready_end": {"ready_digest": "ready"},
            "passed": passed,
        }
    events = [{
        "event_type": "case_completed",
        "details": {"semantic_digest": semantic[value]["semantic_digest"]},
    } for value in contract["execution_query_ids"]]
    resident = {
        "binding_digest": "resident", "resident_content_digest": "content",
    }
    semantic_matrix = module._semantic_matrix(
        contract, resident, resident, registry, semantic, events, 1.0,
    )
    assert semantic_matrix["passed"] is True
    assert semantic_matrix["ordered_query_episode_ids"] == [
        row["episode_id"] for row in registry["cases_data"]
    ]
    performance_matrix = module._performance_matrix(
        contract, registry, attempts, 2.0,
    )
    assert performance_matrix["passed"] is False
    assert performance_matrix["gates"][
        "all_53_confirmatory_primary_attempts_passed"
    ] is False
    final = module._terminal_document(
        module.PERFORMANCE_FINAL_SCHEMA, "final_digest", {
            "performance_passed": performance_matrix["passed"],
            "performance_matrix_digest": performance_matrix["result_digest"],
        },
    )
    assert final["performance_passed"] is False
    assert final["final_digest"] == module.terminal_digest(final, "final_digest")


def test_incomplete_or_terminal_roots_cannot_resume(tmp_path: Path) -> None:
    module = _module()
    module._assert_fresh_output(tmp_path)
    (tmp_path / "INCOMPLETE.json").write_text("{}\n")
    with pytest.raises(ValueError, match="terminal"):
        module._assert_fresh_output(tmp_path)
    (tmp_path / "INCOMPLETE.json").unlink()
    (tmp_path / "ledger").mkdir()
    (tmp_path / "ledger" / "HEAD.json").write_text("{}\n")
    with pytest.raises(ValueError, match="resume is not authorized"):
        module._assert_fresh_output(tmp_path)


def test_any_stray_artifact_contaminates_the_one_shot_root(tmp_path: Path) -> None:
    module = _module()
    (tmp_path / "candidate-contract.json").write_text("{}\n")
    with pytest.raises(ValueError, match="fresh root"):
        module._assert_fresh_output(tmp_path)


def _valid_case_evidence(module):
    query_id = "1" * 24
    quotas = dict(module.FROZEN_ROUTE_QUOTAS)
    candidates = []
    candidate_digest = module.reconstructed_candidate_digest(candidates)
    scan = {
        "schema_version": "m04r-global-bound-proposal-v1",
        "generation_id": module.FROZEN_GENERATION_ID,
        "query_episode_id": query_id,
        "rows_scanned": 10,
        "eligible_rows": 8,
        "eligible_main_rows": 7,
        "eligible_overflow_rows": 1,
        "route_counts": {name: 0 for name in quotas},
        "route_quotas": quotas,
        "candidate_count": 0,
        "candidate_digest": candidate_digest,
    }
    scan["result_digest"] = module.scan_result_digest(
        scan, module.FROZEN_PROPOSAL_CONTRACT_DIGEST,
    )
    expected_query = {
        "query_episode_id": query_id,
        "query_symbol": "ABC",
        "query_start_ns": 100,
        "latest_eligible_ns": 90,
        "query_stock_prefix": {"digest": "stock"},
        "query_benchmark_prefix": {"digest": "benchmark"},
        "query_representation_digest": "representation",
    }
    resident_observation = {
        "schema_version": READY_SCHEMA_VERSION,
        "content_digest": "content",
        "ready_digest": "ready",
        "seal_digest": "seal",
        "ready_file_sha256": "file",
        "ready_identity": {"st_ino": 1},
        "file_identity_lease": {"lease_digest": "lease"},
    }
    resident = {
        "binding_digest": "resident-binding",
        "resident_content_digest": "content",
        "resident_ready_observation": resident_observation,
    }
    contract = {"contract_digest": "contract"}
    case = {"case_id": "case", "episode_id": query_id}
    role = {
        "performance_role": module.CONFIRMATORY_PERFORMANCE_ROLE,
        "recall_role": "blind_primary",
    }
    semantic_gates = {
        "resident_content_matches_contract": True,
        "query_identity_prefix_and_representation_match": True,
        "three_scan_digest_and_block_order_invariance": True,
        "internal_eligible_row_accounting": True,
        "physical_row_accounting": True,
        "frozen_route_quotas": True,
        "candidate_digest_order_and_routes_reconstruct": True,
        "zero_temporal_overlap_tier_duplicate_errors": True,
        "real_forward_outcomes_excluded": True,
    }
    semantic = {
        "schema_version": module.SEMANTIC_CASE_SCHEMA,
        "producer_contract_digest": contract["contract_digest"],
        "registry_digest": module.FROZEN_REGISTRY_DIGEST,
        "generation_id": module.FROZEN_GENERATION_ID,
        "proposal_contract_digest": module.FROZEN_PROPOSAL_CONTRACT_DIGEST,
        "resident_content_digest": "content",
        "registry_case_id": case["case_id"],
        **expected_query,
        "performance_role": role["performance_role"],
        "recall_role": role["recall_role"],
        "scan_semantics": [deepcopy(scan) for _ in range(3)],
        "candidates": candidates,
        "candidate_digest_reconstructed": candidate_digest,
        "violations": {"duplicates": 0, "future": 0, "same_symbol_overlap": 0, "tier": 0},
        "gates": semantic_gates,
        "passed": True,
        "real_forward_outcomes_accessed": False,
        "created_at": "2026-08-26T00:00:00+00:00",
    }
    semantic["semantic_digest"] = module.semantic_case_digest(semantic)
    timings = {
        "resident_first_seconds": 1.0,
        "resident_reverse_seconds": 1.0,
        "resident_repeat_seconds": 1.0,
        "task_seconds": 4.0,
        "peak_rss_mb": 100.0,
    }
    performance_gates = {
        "same_ready_instance_at_start_and_end": True,
        "measurements_finite_nonnegative_and_task_contains_scans": True,
        "resident_first_scan_at_most_120_seconds": True,
        "resident_repeat_scan_at_most_60_seconds": True,
        "worker_rss_at_most_1536_mib": True,
        "primary_attempt_completed": True,
    }
    performance = {
        "schema_version": module.PERFORMANCE_ATTEMPT_SCHEMA,
        "producer_contract_digest": contract["contract_digest"],
        "registry_digest": module.FROZEN_REGISTRY_DIGEST,
        "generation_id": module.FROZEN_GENERATION_ID,
        "resident_binding_digest": resident["binding_digest"],
        "semantic_digest": semantic["semantic_digest"],
        "registry_case_id": case["case_id"],
        "query_episode_id": query_id,
        "performance_role": role["performance_role"],
        "attempt_ordinal": 1,
        "ready_start": deepcopy(resident_observation),
        "ready_end": deepcopy(resident_observation),
        "timings": timings,
        "gates": performance_gates,
        "passed": True,
        "real_forward_outcomes_accessed": False,
        "created_at": "2026-08-26T00:00:01+00:00",
    }
    performance["performance_digest"] = module.performance_attempt_digest(performance)
    return contract, case, role, expected_query, resident, semantic, performance


def test_parent_reconstructs_case_and_rejects_self_consistent_corruption() -> None:
    module = _module()
    contract, case, role, expected_query, resident, semantic, performance = (
        _valid_case_evidence(module)
    )
    module._strict_validate_case_evidence(
        semantic, performance, contract=contract, case=case, role=role,
        expected_query=expected_query, resident=resident, physical_rows=10,
    )
    corrupted = deepcopy(semantic)
    for scan in corrupted["scan_semantics"]:
        scan["route_counts"]["composite"] = 1
        scan["result_digest"] = module.scan_result_digest(
            scan, module.FROZEN_PROPOSAL_CONTRACT_DIGEST,
        )
    corrupted["semantic_digest"] = module.semantic_case_digest(corrupted)
    corrupted_performance = deepcopy(performance)
    corrupted_performance["semantic_digest"] = corrupted["semantic_digest"]
    corrupted_performance["performance_digest"] = module.performance_attempt_digest(
        corrupted_performance,
    )
    with pytest.raises(ValueError, match="rows, violations or gates"):
        module._strict_validate_case_evidence(
            corrupted, corrupted_performance, contract=contract, case=case,
            role=role, expected_query=expected_query, resident=resident,
            physical_rows=10,
        )


def test_atomic_case_bundle_corruption_is_detected() -> None:
    module = _module()
    contract, case, role, expected_query, resident, semantic, performance = (
        _valid_case_evidence(module)
    )
    bundle = module._case_bundle("contract", 0, semantic, performance)
    module._validate_case_bundle(
        bundle, contract=contract, case=case, role=role,
        expected_query=expected_query, resident=resident,
        physical_rows=10, execution_ordinal=0,
    )
    bundle["performance"]["timings"]["task_seconds"] = 99.0
    with pytest.raises(ValueError, match="bundle identity or digest"):
        module._validate_case_bundle(
            bundle, contract=contract, case=case, role=role,
            expected_query=expected_query, resident=resident,
            physical_rows=10, execution_ordinal=0,
        )


def test_crash_with_broken_ledger_still_writes_terminal_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setattr(
        module, "_append_ledger_event",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("crash")),
    )
    assert module._best_effort_mark_incomplete(
        tmp_path, {"contract_digest": "contract"}, {"binding_digest": "resident"},
        case_id="case", query_episode_id="1" * 24,
        reason="worker_crash", error_type="OSError",
    ) is True
    marker = json.loads((tmp_path / "INCOMPLETE.json").read_text())
    module._validate_incomplete(marker, "contract")
    assert marker["ledger_valid"] is True
    assert marker["incomplete_event_appended"] is False
    assert marker["resume_authorized"] is False


def test_mid_run_implementation_mutation_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    expected = {"files": {"x.py": "a" * 64}, "digest": "expected"}
    monkeypatch.setattr(
        module, "_implementation_manifest",
        lambda: {"files": {"x.py": "b" * 64}, "digest": "changed"},
    )
    with pytest.raises(ValueError, match="changed after preregistration"):
        module._require_implementation_manifest(expected)


def test_create_only_publication_never_overwrites_existing_target(
    tmp_path: Path,
) -> None:
    module = _module()
    target = tmp_path / "case-bundles" / f"000-{'1' * 24}.json"
    target.parent.mkdir()
    original = b'{"original":true}\n'
    target.write_bytes(original)
    with pytest.raises(FileExistsError):
        module._atomic_json_create(target, {"replacement": True})
    assert target.read_bytes() == original


def test_ready_observation_rejects_mixed_identity_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    observed = {
        "ready_digest": "r", "ready_file_sha256": "f",
        "seal_digest": "s", "content_digest": "c",
        "identity": {"st_ino": 1},
    }
    lease = {
        "ready_digest": "r", "ready_file_sha256": "f",
        "content_digest": "c", "files": {"ready": {"st_ino": 2}},
        "lease_digest": "l",
    }
    monkeypatch.setattr(module, "observe_ready_strict", lambda path: observed)
    monkeypatch.setattr(module, "resident_file_identity_lease", lambda path: lease)
    with pytest.raises(ValueError, match="between observation and identity lease"):
        module._ready_observation(Path("/resident/READY.json"))


def test_exact_artifact_tree_rejects_unregistered_file(tmp_path: Path) -> None:
    module = _module()
    (tmp_path / "candidate-contract.json").write_text("{}\n")
    module._assert_exact_artifact_files(tmp_path, ["candidate-contract.json"])
    (tmp_path / "authority-contract.json").write_text("{}\n")
    with pytest.raises(ValueError, match="exact run stage"):
        module._assert_exact_artifact_files(tmp_path, ["candidate-contract.json"])


def test_preregistration_requires_clean_tracked_single_file_head(tmp_path: Path) -> None:
    module = _module()

    def git(*args: str) -> None:
        subprocess.run(
            ["git", *args], cwd=tmp_path, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    git("init")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    implementation = tmp_path / "impl.py"
    implementation.write_text("VALUE = 1\n")
    git("add", "impl.py")
    git("commit", "-m", "implementation")
    preregistration = tmp_path / "preregistered.json"
    preregistration.write_text("{}\n")
    git("add", "preregistered.json")
    git("commit", "-m", "preregistration only")
    binding = module._git_preregistration_binding(
        tmp_path, preregistration, ["impl.py"],
    )
    assert len(binding["head_commit"]) == 40
    implementation.write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="tracked worktree"):
        module._git_preregistration_binding(
            tmp_path, preregistration, ["impl.py"],
        )
