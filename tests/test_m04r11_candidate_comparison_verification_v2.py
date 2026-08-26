from __future__ import annotations

from copy import deepcopy
import base64
import importlib.util
import json
from pathlib import Path
import sys
import subprocess

import pytest
import pandas as pd


def _module():
    path = (
        Path(__file__).parents[1] / "experiments" / "m04r"
        / "verify_m04r11_candidate_comparison_v2.py"
    )
    spec = importlib.util.spec_from_file_location("independent_m04r11_v2_verifier", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True))


def _registry() -> dict:
    return json.loads((
        Path(__file__).parents[1] / "config" / "data" / "analogues" / "m04r10"
        / "nasdaq-untouched-authority-registry" / "query-registry.json"
    ).read_text())


def _truth_ids(index: int) -> list[str]:
    return [f"{index + 1:08x}{10_000 + rank:016x}" for rank in range(20)]


def _digest_document(module, payload: dict, field: str, omitted: set[str]) -> dict:
    payload[field] = module._hash({
        key: value for key, value in payload.items() if key not in omitted | {field}
    })
    return payload


def _rechain_ledger(module, candidate_root: Path) -> None:
    previous = module.LEDGER_GENESIS
    paths = sorted((candidate_root / "ledger/events").glob("*.json"))
    events = []
    for path in paths:
        event = json.loads(path.read_text())
        event["previous_event_digest"] = previous
        event["event_digest"] = module._hash({
            key: value for key, value in event.items() if key != "event_digest"
        })
        _write(path, event)
        previous = event["event_digest"]
        events.append(event)
    head = json.loads((candidate_root / "ledger/HEAD.json").read_text())
    head["last_event_digest"] = previous
    head["head_digest"] = module._hash({
        key: value for key, value in head.items() if key != "head_digest"
    })
    _write(candidate_root / "ledger/HEAD.json", head)
    complete = json.loads((candidate_root / "RUN_COMPLETE.json").read_text())
    complete["ledger_last_event_digest"] = events[-1]["event_digest"]
    complete["ledger_head_digest"] = head["head_digest"]
    complete["complete_digest"] = module._hash({
        key: value for key, value in complete.items()
        if key not in {"created_at", "complete_digest"}
    })
    _write(candidate_root / "RUN_COMPLETE.json", complete)


def _build_synthetic(
    module, tmp_path: Path, retained: list[int] | None = None,
    *, performance_fail: bool = False,
) -> dict[str, Path]:
    retained = retained or [20] * 60
    artifact = tmp_path / "artifacts"
    repository = tmp_path / "repository"
    verifier_source = Path(module.__file__).read_text()
    verifier_copy = repository / "experiments/m04r/verify_m04r11_candidate_comparison_v2.py"
    verifier_copy.parent.mkdir(parents=True, exist_ok=True)
    verifier_copy.write_text(verifier_source)
    module.__file__ = str(verifier_copy)
    config_path = repository / "config/datasets.example.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("datasets: {}\n")
    roots = module.expected_roots(artifact)
    registry = _registry()
    module._query_context = lambda _config, case: {
        "query_episode_id": case["episode_id"], "query_symbol": case["symbol"],
        "query_start_ns": 100, "latest_eligible_ns": 1_000,
        "query_stock_prefix": case["stock_prefix"],
        "query_benchmark_prefix": case["benchmark_prefix"],
        "query_representation_digest": "9" * 64,
    }
    registry_root = Path(roots["registry_root"])
    _write(registry_root / "query-registry.json", registry)
    candidate_root = Path(roots["candidate_root"])
    authority_root = Path(roots["authority_root"])
    comparison_root = Path(roots["comparison_root"])
    output_root = Path(roots["verification_root"])

    for relative in module.IMPLEMENTATION_FILES:
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path != verifier_copy:
            path.write_text(f"# synthetic {relative}\n")
    authority_builder = repository / "experiments/m04r/m04r11_build_authorities.py"
    authority_builder.write_text("# synthetic authority builder\n")
    implementation_files = {
        relative: module._file_hash(repository / relative)
        for relative in module.IMPLEMENTATION_FILES
    }
    implementation = {
        "files": implementation_files, "digest": module._hash(implementation_files),
    }
    environment = module._environment_manifest()

    generation = (
        Path(roots["source_full_root"]) / "store" / "generations"
        / module.FROZEN_GENERATION_ID
    )
    generation.mkdir(parents=True)
    manifest_path = generation / "manifest.json"
    rows_path = generation / "bound-rows.bin"
    overflow_path = generation / "overflow-exact-fallback.bin"
    manifest_path.write_text(json.dumps({
        "pack_contract_digest": "1" * 64,
        "quantized_bound_contract_digest": "2" * 64,
    }))
    rows_path.write_bytes(b"r" * 2_432)
    overflow_path.write_bytes(b"o" * 32)
    source_pack = {
        "generation_id": module.FROZEN_GENERATION_ID,
        "source_full_root": roots["source_full_root"],
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": module._file_hash(manifest_path),
        "manifest_digest": module.FROZEN_GENERATION_ID,
        "provenance_digest": "a" * 64,
        "rows_file": rows_path.name, "rows_path": str(rows_path.resolve()),
        "rows_bytes": 2_432, "rows_sha256": module._file_hash(rows_path),
        "row_count": 1, "row_bytes": 2_432,
        "overflow_file": overflow_path.name,
        "overflow_path": str(overflow_path.resolve()), "overflow_bytes": 32,
        "overflow_sha256": module._file_hash(overflow_path),
        "overflow_count": 1, "overflow_row_bytes": 32,
        "physical_rows": 2, "active_pointer_absent": True,
    }
    roles = module._roles(registry)
    execution = module._execution(roles)
    ordered = [case["episode_id"] for case in registry["cases_data"]]
    contract = {
        "schema_version": module.PRODUCER_SCHEMA,
        "registry_digest": module.FROZEN_REGISTRY_DIGEST,
        "ordered_query_ids": ordered,
        "ordered_query_ids_digest": module.FROZEN_CASE_ORDER_DIGEST,
        "role_table": roles, "role_table_digest": module.FROZEN_ROLE_TABLE_DIGEST,
        "exposed_query_ids_digest": module.FROZEN_EXPOSED_IDS_DIGEST,
        "confirmatory_query_ids_digest": module.FROZEN_CONFIRMATORY_IDS_DIGEST,
        "execution_query_ids": execution,
        "execution_query_ids_digest": module.FROZEN_EXECUTION_ORDER_DIGEST,
        "generation_id": module.FROZEN_GENERATION_ID,
        "proposal_contract_digest": module.FROZEN_PROPOSAL_CONTRACT_DIGEST,
        "route_quotas": module.FROZEN_ROUTE_QUOTAS,
        "request": module.FROZEN_REQUEST, "roots": roots,
        "predecessor_failure": module.PREDECESSOR_FAILURE,
        "source_pack": source_pack, "implementation_manifest": implementation,
        "environment_manifest": environment,
        "artifact_schemas": module.ARTIFACT_SCHEMAS,
        "resident_policy": module.RESIDENT_POLICY,
        "resident_ready_schema": module.RESIDENT_READY_SCHEMA,
        "resident_binding_schema": module.RESIDENT_BINDING_SCHEMA,
        "scan_protocol": module.SCAN_PROTOCOL,
        "performance_limits": module.PERFORMANCE_LIMITS,
        "semantic_case_schema": module.SEMANTIC_CASE_SCHEMA,
        "performance_attempt_schema": module.PERFORMANCE_ATTEMPT_SCHEMA,
        "semantic_matrix_schema": module.SEMANTIC_MATRIX_SCHEMA,
        "semantic_seal_schema": module.SEMANTIC_SEAL_SCHEMA,
        "performance_matrix_schema": module.PERFORMANCE_MATRIX_SCHEMA,
        "performance_final_schema": module.PERFORMANCE_FINAL_SCHEMA,
        "run_ledger_event_schema": module.LEDGER_EVENT_SCHEMA,
        "run_ledger_head_schema": module.LEDGER_HEAD_SCHEMA,
        "incomplete_schema": "candidate-resident-incomplete-v1",
        "run_complete_schema": module.RUN_COMPLETE_SCHEMA,
        "case_bundle_schema": module.CASE_BUNDLE_SCHEMA,
        "results_opened_schema": module.RESULTS_OPENED_SCHEMA,
        "comparison_matrix_schema": module.COMPARISON_MATRIX_SCHEMA,
        "comparison_seal_schema": module.COMPARISON_SEAL_SCHEMA,
        "comparison_verification_schema": module.VERIFICATION_SCHEMA,
        "preregistration_schema": module.PREREGISTRATION_SCHEMA,
        "preregistration_relative_path": module.PREREGISTRATION_RELATIVE_PATH,
        "marker_and_resume_policy": module.MARKER_AND_RESUME_POLICY,
        "preregistration_policy": module.PREREGISTRATION_POLICY,
        "comparison_policy": module.COMPARISON_POLICY,
        "semantic_case_gates": list(module.SEMANTIC_GATES),
        "performance_attempt_gates": list(module.PERFORMANCE_GATES),
        "semantic_matrix_gates": list(module.SEMANTIC_MATRIX_GATES),
        "performance_matrix_gates": list(module.PERFORMANCE_MATRIX_GATES),
        "claims_policy": module.CLAIMS_POLICY,
        "real_forward_outcomes_accessed": False,
    }
    contract["contract_digest"] = module._hash(contract)
    _write(candidate_root / "candidate-contract.json", contract)
    prereg_path = repository / module.PREREGISTRATION_RELATIVE_PATH
    prereg = {
        "schema_version": module.PREREGISTRATION_SCHEMA,
        "relative_path": module.PREREGISTRATION_RELATIVE_PATH,
        "resolved_path": str(prereg_path.resolve()), "producer_contract": contract,
        "producer_contract_digest": contract["contract_digest"],
        "source_pack_binding_digest": module._hash(source_pack),
        "implementation_manifest_digest": implementation["digest"],
        "environment_manifest_digest": environment["digest"],
        "artifact_schemas": module.ARTIFACT_SCHEMAS,
        "marker_and_resume_policy": module.MARKER_AND_RESUME_POLICY,
        "comparison_policy": module.COMPARISON_POLICY,
        "preregistration_policy": module.PREREGISTRATION_POLICY,
        "authority_results_opened": False,
        "candidate_authority_comparison_opened": False,
        "real_forward_outcomes_accessed": False,
    }
    prereg["preregistration_digest"] = module._hash(prereg)
    _write(prereg_path, prereg)
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.name", "Verifier Test"], cwd=repository, check=True)
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "synthetic preregistration"], cwd=repository, check=True)
    head_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip()

    resident_content = module._resident_content(source_pack)
    content_digest = module._hash(resident_content)
    resident_root = Path(roots["resident_full_root"])
    store_root = resident_root / "store"
    generation_root = store_root / "generations" / module.FROZEN_GENERATION_ID
    identity_paths = {
        "ready": resident_root / "READY.json", "mirror": resident_root,
        "store": store_root, "generations": store_root / "generations",
        "generation": generation_root, "file_manifest": generation_root / "manifest.json",
        "file_rows": generation_root / rows_path.name,
        "file_overflow": generation_root / overflow_path.name,
    }
    identities = {
        name: {"path": str(path), "st_dev": 1, "st_ino": index + 1,
               "st_size": 1, "st_mtime_ns": 1, "st_ctime_ns": 1, "st_mode": 0o100644}
        for index, (name, path) in enumerate(identity_paths.items())
    }
    mount_binding = {"mount_id": 1, "parent_mount_id": 0, "major_minor": "0:1",
                     "mount_root": "/", "mount_point": "/dev/shm", "mount_options": [],
                     "optional_fields": [], "fs_type": "tmpfs", "mount_source": "tmpfs",
                     "super_options": [], "st_dev": 1}
    source_generation = Path(roots["source_full_root"]) / "store/generations" / module.FROZEN_GENERATION_ID
    source_paths = {"manifest": source_generation / "manifest.json", "rows": rows_path, "overflow": overflow_path}
    mirror_paths = {"manifest": generation_root / "manifest.json", "rows": generation_root / rows_path.name,
                    "overflow": generation_root / overflow_path.name}
    source_bindings = {
        name: {"path": str(source_paths[name]), **resident_content["source_files"][name], "st_dev": 2}
        for name in source_paths
    }
    mirror_bindings = {
        name: {"path": str(mirror_paths[name]), **resident_content["mirror_files"][name], "st_dev": 1}
        for name in mirror_paths
    }
    seal = {
        "generation_id": module.FROZEN_GENERATION_ID,
        "provenance_digest": source_pack["provenance_digest"],
        "pack_contract_digest": resident_content["pack_contract_digest"],
        "quantized_bound_contract_digest": resident_content["quantized_bound_contract_digest"],
        "source_store_root": str(Path(roots["source_full_root"]) / "store"),
        "mirror_root": str(resident_root), "mirror_store_root": str(store_root),
        "source_generation_st_dev": 2, "mirror_generation_st_dev": 1,
        "source_files": source_bindings, "mirror_files": mirror_bindings,
        "resident_mount": mount_binding, "mountinfo_path": "/proc/self/mountinfo",
        "resident_capacity_bytes": 2_000_000_000,
        "required_capacity_bytes": resident_content["physical_generation_bytes"] + 1_073_741_824,
        "reserve_bytes": 1_073_741_824,
        "physical_generation_bytes": resident_content["physical_generation_bytes"],
        "storage_class": "tmpfs-backed-generation-v1", "latency_scope": module.RESIDENT_LATENCY_SCOPE,
        "query_specific_inputs_used": False, "outcomes_or_labels_used": False,
        "real_forward_outcomes_accessed": False, "source_active_pointer_absent": True,
        "mirror_active_pointer_absent": True,
    }
    ready_payload = {
        "schema_version": module.RESIDENT_READY_SCHEMA, "mode": "validate-existing",
        "content": resident_content, "content_digest": content_digest,
        "seal": seal, "seal_digest": module._hash(seal),
        "capacity_observation": {
            "before": {"capacity_bytes": 2_000_000_000, "available_bytes": 1_500_000_000, "block_bytes": 4096},
            "after": {"capacity_bytes": 2_000_000_000, "available_bytes": 1_500_000_000, "block_bytes": 4096},
        },
        "startup_timings": {
            "source_content_verification_seconds": 1.0, "mirror_copy_seconds": 0.0,
            "mirror_content_verification_seconds": 1.0, "readiness_seal_seconds": 0.1,
            "total_before_ready_seconds": 2.1,
        },
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    ready_payload["ready_digest"] = module._hash(ready_payload)
    ready_bytes = (json.dumps(ready_payload, indent=2, sort_keys=True) + "\n").encode()
    ready_sha = module.sha256(ready_bytes).hexdigest()
    identities["ready"]["st_size"] = len(ready_bytes)
    lease = {
        "schema_version": module.FILE_IDENTITY_LEASE_SCHEMA,
        "ready_digest": ready_payload["ready_digest"], "ready_file_sha256": ready_sha,
        "content_digest": content_digest, "files": identities,
    }
    lease["lease_digest"] = module._hash(lease)
    ready_observation = {
        "schema_version": module.RESIDENT_READY_SCHEMA,
        "content_digest": content_digest, "ready_digest": ready_payload["ready_digest"],
        "seal_digest": ready_payload["seal_digest"], "ready_file_sha256": ready_sha,
        "ready_identity": identities["ready"], "file_identity_lease": lease,
    }
    validation = {
        "schema_version": module.VALIDATION_OBSERVATION_SCHEMA,
        "ready_digest": ready_observation["ready_digest"],
        "content_digest": ready_observation["content_digest"],
        "seal_digest": ready_observation["seal_digest"],
        "reserve_bytes": 1_073_741_824,
        "capacity": {"capacity_bytes": 2_000_000_000, "available_bytes": 1_500_000_000, "block_bytes": 4096},
        "mount": mount_binding,
        "source_content_verification_seconds": 1.0,
        "mirror_content_verification_seconds": 1.0, "validation_seconds": 2.0,
        "observed_at": "2026-01-01T00:00:00+00:00",
    }
    validation["observation_digest"] = module._hash(validation)
    resident = {
        "schema_version": module.RESIDENT_BINDING_SCHEMA,
        "producer_contract_digest": contract["contract_digest"],
        "resident_ready_path": str(resident_root / "READY.json"),
        "resident_ready_schema": module.RESIDENT_READY_SCHEMA,
        "resident_content_digest": ready_observation["content_digest"],
        "resident_ready_observation": ready_observation,
        "resident_ready_payload": ready_payload,
        "resident_ready_bytes_base64": base64.b64encode(ready_bytes).decode("ascii"),
        "validation_observation": validation,
        "generation_id": module.FROZEN_GENERATION_ID,
        "provenance_digest": source_pack["provenance_digest"],
        "mirror_store_root": str(store_root),
        "storage_class": "tmpfs-backed-generation-v1",
        "latency_scope": module.RESIDENT_LATENCY_SCOPE,
        "query_specific_inputs_used": False, "outcomes_or_labels_used": False,
        "real_forward_outcomes_accessed": False,
    }
    resident["binding_digest"] = module._hash(resident)
    _write(candidate_root / "RESIDENT_READY.json", resident)

    case_by_id = {case["episode_id"]: case for case in registry["cases_data"]}
    role_by_id = {role["query_episode_id"]: role for role in roles}
    semantics, attempts, bundles = {}, {}, []
    for ordinal, query_id in enumerate(execution):
        case = case_by_id[query_id]
        case_index = ordered.index(query_id)
        truth = _truth_ids(case_index)
        fillers = [
            f"{0xF0000000 + case_index:08x}{50_000 + rank:016x}"
            for rank in range(20 - retained[case_index])
        ]
        candidate_ids = truth[:retained[case_index]] + fillers
        candidates = [{
            "episode_id": episode_id, "symbol": "OTHER", "cutoff_ns": rank,
            "quality_tier": "A", "lower_bound_hex": float(rank + 1).hex(),
            "routes": ["composite"], "overflow_fallback": False,
        } for rank, episode_id in enumerate(candidate_ids)]
        candidate_digest = module._candidate_digest(candidates)
        route_counts = {
            name: (20 if name == "composite" else 0)
            for name in module.FROZEN_ROUTE_QUOTAS
        }
        scan = {
            "schema_version": "m04r-global-bound-proposal-v1",
            "generation_id": module.FROZEN_GENERATION_ID,
            "query_episode_id": query_id, "rows_scanned": 2,
            "eligible_rows": 2, "eligible_main_rows": 1,
            "eligible_overflow_rows": 1, "route_counts": route_counts,
            "route_quotas": module.FROZEN_ROUTE_QUOTAS,
            "candidate_count": 20, "candidate_digest": candidate_digest,
        }
        scan["result_digest"] = module._scan_digest(scan)
        semantic_gates = {name: True for name in module.SEMANTIC_GATES}
        semantic = {
            "schema_version": module.SEMANTIC_CASE_SCHEMA,
            "producer_contract_digest": contract["contract_digest"],
            "registry_digest": module.FROZEN_REGISTRY_DIGEST,
            "generation_id": module.FROZEN_GENERATION_ID,
            "proposal_contract_digest": module.FROZEN_PROPOSAL_CONTRACT_DIGEST,
            "resident_content_digest": resident["resident_content_digest"],
            "registry_case_id": case["case_id"], "query_episode_id": query_id,
            "query_symbol": case["symbol"], "query_start_ns": 100,
            "latest_eligible_ns": 1_000,
            "query_stock_prefix": case["stock_prefix"],
            "query_benchmark_prefix": case["benchmark_prefix"],
            "query_representation_digest": "9" * 64,
            "performance_role": role_by_id[query_id]["performance_role"],
            "recall_role": "blind_primary", "scan_semantics": [scan, scan, scan],
            "candidates": candidates,
            "candidate_digest_reconstructed": candidate_digest,
            "violations": {"duplicates": 0, "future": 0, "same_symbol_overlap": 0, "tier": 0},
            "gates": semantic_gates, "passed": True,
            "real_forward_outcomes_accessed": False, "created_at": "2026-01-01T00:00:00+00:00",
        }
        semantic["semantic_digest"] = module._hash({
            key: value for key, value in semantic.items() if key != "created_at"
        })
        fail_this = performance_fail and case_index == 7
        timings = {
            "resident_first_seconds": 121.0 if fail_this else 10.0,
            "resident_reverse_seconds": 10.0, "resident_repeat_seconds": 10.0,
            "task_seconds": 141.0 if fail_this else 30.0, "peak_rss_mb": 100.0,
        }
        performance_gates = {name: True for name in module.PERFORMANCE_GATES}
        if fail_this:
            performance_gates["resident_first_scan_at_most_120_seconds"] = False
        performance = {
            "schema_version": module.PERFORMANCE_ATTEMPT_SCHEMA,
            "producer_contract_digest": contract["contract_digest"],
            "registry_digest": module.FROZEN_REGISTRY_DIGEST,
            "generation_id": module.FROZEN_GENERATION_ID,
            "resident_binding_digest": resident["binding_digest"],
            "semantic_digest": semantic["semantic_digest"],
            "registry_case_id": case["case_id"], "query_episode_id": query_id,
            "performance_role": role_by_id[query_id]["performance_role"],
            "attempt_ordinal": 1, "ready_start": ready_observation,
            "ready_end": ready_observation, "timings": timings,
            "gates": performance_gates, "passed": all(performance_gates.values()),
            "real_forward_outcomes_accessed": False, "created_at": "2026-01-01T00:00:00+00:00",
        }
        performance["performance_digest"] = module._hash({
            key: value for key, value in performance.items() if key != "created_at"
        })
        bundle = {
            "schema_version": module.CASE_BUNDLE_SCHEMA,
            "producer_contract_digest": contract["contract_digest"],
            "execution_ordinal": ordinal, "query_episode_id": query_id,
            "semantic": semantic, "performance": performance,
        }
        bundle["bundle_digest"] = module._hash(bundle)
        _write(candidate_root / "case-bundles" / f"{ordinal:03d}-{query_id}.json", bundle)
        semantics[query_id], attempts[query_id] = semantic, performance
        bundles.append(bundle)

    events, previous = [], module.LEDGER_GENESIS
    def event(event_type: str, details: dict) -> None:
        nonlocal previous
        row = {
            "schema_version": module.LEDGER_EVENT_SCHEMA,
            "producer_contract_digest": contract["contract_digest"],
            "event_index": len(events), "previous_event_digest": previous,
            "event_type": event_type, "details": details, "created_at": "2026-01-01T00:00:00+00:00",
        }
        row["event_digest"] = module._hash(row)
        previous = row["event_digest"]
        events.append(row)
    git_binding = {
        "repository_root": str(repository.resolve()), "head_commit": head_commit,
        "preregistration_relative_path": module.PREREGISTRATION_RELATIVE_PATH,
        "preregistration_blob_sha256": module._file_hash(prereg_path),
        "tracked_worktree_clean": True, "index_clean": True,
    }
    git_binding["binding_digest"] = module._hash(git_binding)
    event("run_started", {
        "preregistration_digest": prereg["preregistration_digest"],
        "git_binding": git_binding, "resident_binding_digest": resident["binding_digest"],
        "resident_content_digest": resident["resident_content_digest"],
        "execution_query_ids_digest": contract["execution_query_ids_digest"],
    })
    for ordinal, query_id in enumerate(execution):
        case = case_by_id[query_id]
        event("case_started", {
            "execution_ordinal": ordinal, "registry_case_id": case["case_id"],
            "query_episode_id": query_id,
            "performance_role": role_by_id[query_id]["performance_role"],
            "implementation_manifest_digest": implementation["digest"],
            "resident_ready_digest": ready_observation["ready_digest"],
            "resident_content_digest": resident["resident_content_digest"],
        })
        bundle = bundles[ordinal]
        semantic, performance = bundle["semantic"], bundle["performance"]
        event("case_completed", {
            "execution_ordinal": ordinal,
            "registry_case_id": semantic["registry_case_id"],
            "query_episode_id": query_id,
            "performance_role": semantic["performance_role"],
            "bundle_digest": bundle["bundle_digest"],
            "semantic_digest": semantic["semantic_digest"],
            "performance_digest": performance["performance_digest"],
            "semantic_passed": True, "performance_passed": performance["passed"],
        })
    ordered_semantics = [semantics[value] for value in ordered]
    semantic_matrix_gates = {name: True for name in module.SEMANTIC_MATRIX_GATES}
    semantic_matrix = {
        "schema_version": module.SEMANTIC_MATRIX_SCHEMA,
        "producer_contract_digest": contract["contract_digest"],
        "resident_start_content_digest": resident["resident_content_digest"],
        "resident_end_content_digest": resident["resident_content_digest"],
        "ordered_query_episode_ids": ordered,
        "semantic_case_digests": [row["semantic_digest"] for row in ordered_semantics],
        "gates": semantic_matrix_gates, "passed": True,
        "real_forward_outcomes_accessed": False,
        "elapsed_seconds": 1.0, "created_at": "2026-01-01T00:00:00+00:00",
    }
    semantic_matrix["result_digest"] = module._hash({
        key: value for key, value in semantic_matrix.items()
        if key not in {"elapsed_seconds", "created_at"}
    })
    _write(candidate_root / "semantic-matrix.json", semantic_matrix)
    semantic_seal = {
        "schema_version": module.SEMANTIC_SEAL_SCHEMA,
        "producer_contract_digest": contract["contract_digest"],
        "semantic_matrix_digest": semantic_matrix["result_digest"],
        "resident_start_content_digest": resident["resident_content_digest"],
        "resident_end_content_digest": resident["resident_content_digest"],
        "semantic_cases": 60, "semantic_recall_ready": True,
        "performance_independent": True, "authority_results_opened": False,
        "production_promotion_authorized": False,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    semantic_seal["seal_digest"] = module._hash({
        key: value for key, value in semantic_seal.items() if key != "created_at"
    })
    _write(candidate_root / "SEMANTIC_SEALED.json", semantic_seal)
    ordered_attempts = [attempts[value] for value in ordered]
    confirmatory = [
        row for row in ordered_attempts
        if row["performance_role"] == "confirmatory_untouched"
    ]
    exposed = [
        row for row in ordered_attempts
        if row["performance_role"] == "exposed_recovery_regression"
    ]
    perf_passed = all(row["passed"] for row in ordered_attempts)
    performance_gates = {
        "exact_7_exposed_53_confirmatory_role_partition": True,
        "all_60_primary_attempts_accounted": True,
        "all_53_confirmatory_primary_attempts_passed": all(row["passed"] for row in confirmatory),
        "all_7_exposed_regression_attempts_passed": all(row["passed"] for row in exposed),
        "all_60_operational_limits_passed": perf_passed,
        "single_ready_instance_for_all_primary_attempts": True,
    }
    performance_matrix = {
        "schema_version": module.PERFORMANCE_MATRIX_SCHEMA,
        "producer_contract_digest": contract["contract_digest"],
        "ordered_query_episode_ids": ordered,
        "attempt_digests": [row["performance_digest"] for row in ordered_attempts],
        "confirmatory_query_episode_ids": [row["query_episode_id"] for row in confirmatory],
        "exposed_query_episode_ids": [row["query_episode_id"] for row in exposed],
        "gates": performance_gates, "passed": perf_passed,
        "claims_policy": module.CLAIMS_POLICY,
        "real_forward_outcomes_accessed": False,
        "elapsed_seconds": 1.0, "created_at": "2026-01-01T00:00:00+00:00",
    }
    performance_matrix["result_digest"] = module._hash({
        key: value for key, value in performance_matrix.items()
        if key not in {"elapsed_seconds", "created_at"}
    })
    _write(candidate_root / "performance-matrix.json", performance_matrix)
    performance_final = {
        "schema_version": module.PERFORMANCE_FINAL_SCHEMA,
        "producer_contract_digest": contract["contract_digest"],
        "performance_matrix_digest": performance_matrix["result_digest"],
        "resident_binding_digest": resident["binding_digest"],
        "performance_terminal": True, "performance_passed": perf_passed,
        "confirmatory_performance_cases": 53, "exposed_regression_cases": 7,
        "claims_policy": module.CLAIMS_POLICY,
        "authority_results_opened": False,
        "production_promotion_authorized": False, "created_at": "2026-01-01T00:00:00+00:00",
    }
    performance_final["final_digest"] = module._hash({
        key: value for key, value in performance_final.items() if key != "created_at"
    })
    _write(candidate_root / "PERFORMANCE_FINAL.json", performance_final)
    final_details = {
        "semantic_matrix_digest": semantic_matrix["result_digest"],
        "semantic_seal_digest": semantic_seal["seal_digest"],
        "performance_matrix_digest": performance_matrix["result_digest"],
        "performance_final_digest": performance_final["final_digest"],
        "performance_passed": perf_passed,
    }
    event("run_complete", final_details)
    for index, row in enumerate(events):
        _write(candidate_root / "ledger" / "events" / f"{index:06d}.json", row)
    head = {
        "schema_version": module.LEDGER_HEAD_SCHEMA,
        "producer_contract_digest": contract["contract_digest"],
        "event_count": len(events), "last_event_digest": events[-1]["event_digest"],
    }
    head["head_digest"] = module._hash(head)
    _write(candidate_root / "ledger" / "HEAD.json", head)
    run_complete = {
        "schema_version": module.RUN_COMPLETE_SCHEMA,
        "producer_contract_digest": contract["contract_digest"],
        "resident_binding_digest": resident["binding_digest"],
        "semantic_seal_digest": semantic_seal["seal_digest"],
        "performance_final_digest": performance_final["final_digest"],
        "ledger_last_event_digest": events[-1]["event_digest"],
        "ledger_head_digest": head["head_digest"], "semantic_passed": True,
        "performance_passed": perf_passed, "authority_results_opened": False,
        "production_promotion_authorized": False,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    run_complete["complete_digest"] = module._hash({
        key: value for key, value in run_complete.items() if key != "created_at"
    })
    _write(candidate_root / "RUN_COMPLETE.json", run_complete)

    authority_files = {
        "experiments/m04r/m04r11_build_authorities.py": module._file_hash(authority_builder)
    }
    controls = registry["search_contract"]["controls"]
    fake_digest_doc = lambda name: {"name": name, "digest": module._hash({"name": name})}
    authority_contract = {
        "schema_version": module.AUTHORITY_CONTRACT_SCHEMA,
        "registry_digest": module.FROZEN_REGISTRY_DIGEST,
        "generation_id": module.FROZEN_GENERATION_ID,
        "full_build_evidence_digest": "3" * 64,
        "search_contract": registry["search_contract"],
        "certified_execution_contract": module.certified_packed_search_contract(
            requested_positions=True, vector_lower_bounds=True, deferred_alignments=True,
            compact_scored=True, native_bound_deferral=True,
            streaming_threshold_closure=True, branch_aware_packed_bounds=True,
        ),
        "branch_bound_evidence": fake_digest_doc("branch"),
        "primary_proposal_contract": module.packed_bound_search_contract(branch_aware=True),
        "threshold_scan_contract": module.packed_bound_threshold_scan_contract(branch_aware=True),
        "controls": controls,
        "frontier_overflow_policy": module._overflow_policy(controls["maximum_frontier_rows"]),
        "execution_processes": controls["processes"],
        "expected_query_episode_ids": ordered,
        "selection_order": [case["case_id"] for case in registry["cases_data"]],
        "authority_root_policy": "write-isolated truth; no candidate result input",
        "runner_sha256": module._file_hash(authority_builder),
        "implementation_manifest": {
            "files": authority_files, "digest": module._hash(authority_files),
        },
        "real_forward_outcomes_accessed": False,
    }
    authority_contract["contract_digest"] = module._hash(authority_contract)
    authority_rows, authorities = [], {}
    for index, case in enumerate(registry["cases_data"]):
        matches = [{
            "episode_id": episode_id, "symbol": f"S{rank}", "cutoff": "1970-01-01T00:00:00+00:00",
            "total_distance": float(rank + 1),
            "component_distances": {"price": float(rank + 1)},
            "alignment": [[0, 0]], "quality_tier": "A",
        } for rank, episode_id in enumerate(_truth_ids(index))]
        certificate = {
            "schema_version": "certified-search-certificate-v1",
            "contract_digest": authority_contract["certified_execution_contract"]["digest"],
            "generation_id": module.FROZEN_GENERATION_ID,
            "query_episode_id": case["episode_id"], "input_digest": f"{index + 1:064x}",
            "eligible_candidates": 20, "exact_evaluated": 20, "safely_pruned": 0,
            "stopped_early": False, "stop_threshold": 20.0,
            "next_lower_bound": None, "maximum_quantized_bound_excess": 0.0,
            "rounds": [{"frontier_rows": 20, "exact_rows": 20, "constrained_threshold": 20.0}],
        }
        certificate["result_digest"] = module._certificate_digest(certificate, matches)
        authority = {
            "schema_version": module.AUTHORITY_CASE_SCHEMA, "status": "completed",
            "contract_digest": authority_contract["contract_digest"],
            "registry_digest": module.FROZEN_REGISTRY_DIGEST,
            "generation_id": module.FROZEN_GENERATION_ID,
            "registry_case_id": case["case_id"], "query_episode_id": case["episode_id"],
            "query_symbol": case["symbol"], "query_cutoff": case["cutoff"],
            "query_start": pd.Timestamp(100, tz="UTC").isoformat(),
            "latest_eligible_cutoff": pd.Timestamp(1_000, tz="UTC").isoformat(),
            "query_stock_prefix": case["stock_prefix"],
            "query_benchmark_prefix": case["benchmark_prefix"],
            "primary_proposal_result_digest": "4" * 64,
            "proposal_result_digest": "4" * 64,
            "proposal_seconds": 1.0, "amortized_proposal_seconds": 1.0,
            "frontier_attempts": [{
                "maximum_frontier_rows": controls["maximum_frontier_rows"],
                "proposal_result_digest": "4" * 64, "exact_evaluated": 20,
                "stop_threshold_hex": float(20).hex(), "next_lower_bound_hex": None,
                "status": "certified",
            }],
            "streaming_threshold_closure_used": False,
            "frontier_limit_rows": controls["maximum_frontier_rows"],
            "frontier_attempt_measurements": [{"maximum_frontier_rows": controls["maximum_frontier_rows"], "elapsed_seconds": 1.0, "status": "certified"}],
            "matches": matches, "certificate": certificate,
            "certificate_digest": certificate["result_digest"],
            "exact_seconds": 1.0, "final_exact_seconds": 1.0, "peak_rss_mb": 100.0,
            "real_forward_outcomes_accessed": False, "created_at": "2026-01-01T00:00:00+00:00",
        }
        authority["gates"] = module._authority_case_gates(authority, case, authority_contract)
        authority["gate_passed"] = all(authority["gates"].values())
        authority["result_digest"] = module._hash({
            key: value for key, value in authority.items()
            if key not in module.AUTHORITY_CASE_OMITTED
        })
        authority["checkpoint_integrity_digest"] = module._hash({
            key: value for key, value in authority.items()
            if key not in {"created_at", "checkpoint_integrity_digest"}
        })
        _write(authority_root / "cases" / f"{case['episode_id']}.json", authority)
        authorities[case["episode_id"]] = authority
        authority_rows.append({
            "registry_case_id": case["case_id"],
            "query_episode_id": case["episode_id"],
            "authority_digest": authority["result_digest"],
            "certificate_digest": authority["certificate_digest"],
            "eligible_candidates": authority["certificate"]["eligible_candidates"],
            "exact_evaluated": authority["certificate"]["exact_evaluated"],
        })
    measurements = [{
        "registry_case_id": case["case_id"], "proposal_seconds": 1.0,
        "exact_seconds": 1.0, "search_seconds": 2.0, "peak_rss_mb": 100.0,
    } for case in registry["cases_data"]]
    authority_matrix = {
        "schema_version": module.AUTHORITY_MATRIX_SCHEMA,
        "contract_digest": authority_contract["contract_digest"],
        "registry_digest": module.FROZEN_REGISTRY_DIGEST,
        "generation_id": module.FROZEN_GENERATION_ID, "cases": authority_rows,
        "branch_bound_evidence_digest": authority_contract["branch_bound_evidence"]["digest"],
        "measurements": measurements,
        "completed_cases": 60, "invalid_cases": [], "gate_passed": True,
        "p95_exact_seconds": 1.0, "maximum_exact_seconds": 1.0, "total_exact_seconds": 60.0,
        "p95_search_seconds": 2.0, "maximum_search_seconds": 2.0, "total_search_seconds": 120.0,
        "maximum_worker_rss_mb": 100.0, "streaming_threshold_closure_cases": 0,
        "worker_failures": [], "elapsed_seconds": 1.0,
        "gates": {"all_60_authorities_complete": True, "all_case_certificates_pass": True,
                  "no_worker_failures": True, "registry_order_exact": True,
                  "real_forward_outcomes_excluded": True},
        "candidate_results_opened": False, "real_forward_outcomes_accessed": False,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    authority_matrix["performance_gates"] = {
        "certified_p95_search_at_most_300_seconds": True,
        "certified_maximum_search_at_most_600_seconds": True,
        "certified_peak_rss_at_most_1536_mb": True,
    }
    authority_matrix["performance_gate_passed"] = True
    authority_matrix["measurement_integrity_digest"] = module._hash({
        "measurements": authority_matrix["measurements"], "p95_exact_seconds": 1.0,
        "maximum_exact_seconds": 1.0, "total_exact_seconds": 60.0,
        "p95_search_seconds": 2.0, "maximum_search_seconds": 2.0,
        "total_search_seconds": 120.0, "maximum_worker_rss_mb": 100.0,
        "performance_gates": authority_matrix["performance_gates"], "performance_gate_passed": True,
    })
    authority_matrix["result_digest"] = module._hash({
        key: value for key, value in authority_matrix.items()
        if key not in module.AUTHORITY_MATRIX_OMITTED
    })
    authority_seal = {
        "schema_version": module.AUTHORITY_SEAL_SCHEMA,
        "contract_digest": authority_contract["contract_digest"],
        "registry_digest": module.FROZEN_REGISTRY_DIGEST,
        "branch_bound_evidence_digest": authority_contract["branch_bound_evidence"]["digest"],
        "authority_matrix_digest": authority_matrix["result_digest"],
        "measurement_integrity_digest": authority_matrix["measurement_integrity_digest"],
        "authority_cases": 60, "authority_correctness_sealed": True,
        "seal_scope": "exact authority correctness only", "performance_gate_passed": True,
        "production_promotion_authorized": False,
        "candidate_results_opened": False, "real_forward_outcomes_accessed": False,
    }
    authority_seal["seal_digest"] = module._hash(authority_seal)
    _write(authority_root / "authority-contract.json", authority_contract)
    _write(authority_root / "authority-matrix.json", authority_matrix)
    (authority_root / "authority-matrix.html").write_text("<html>synthetic</html>\n")
    _write(authority_root / "SEALED.json", authority_seal)

    marker_expected = {
        "schema_version": module.RESULTS_OPENED_SCHEMA,
        "registry_digest": module.FROZEN_REGISTRY_DIGEST,
        "producer_contract_digest": contract["contract_digest"],
        "semantic_seal_digest": semantic_seal["seal_digest"],
        "performance_final_digest": performance_final["final_digest"],
        "run_complete_digest": run_complete["complete_digest"],
        "status": "authority results about to be opened exactly once",
    }
    marker = {
        **marker_expected, "created_at": "2026-01-01T00:00:00+00:00",
        "result_digest": module._hash(marker_expected),
    }
    _write(comparison_root / "RESULTS_OPENED.json", marker)
    comparison_cases, comparison_failures, total = [], [], 0
    for index, case in enumerate(registry["cases_data"]):
        query_id = case["episode_id"]
        candidate_ids = {row["episode_id"] for row in semantics[query_id]["candidates"]}
        truth = _truth_ids(index)
        retained_ids = [value for value in truth if value in candidate_ids]
        count = len(retained_ids)
        total += count
        case_failures = [] if count >= 19 else ["candidate retained fewer than 19 of 20"]
        comparison_cases.append({
            "registry_case_id": case["case_id"], "query_episode_id": query_id,
            "candidate_semantic_digest": semantics[query_id]["semantic_digest"],
            "authority_case_digest": authorities[query_id]["result_digest"],
            "candidate_count": 20, "retained_count": count,
            "recall_at_20": count / 20.0, "perfect_20_of_20": count == 20,
            "missing_authority_episode_ids": [
                value for value in truth if value not in candidate_ids
            ],
            "failures": case_failures, "passed": not case_failures,
        })
        comparison_failures.extend(f"{case['case_id']}:{value}" for value in case_failures)
    comparison_gates = {
        "all_60_authority_cases_valid": True,
        "every_case_retains_at_least_19_of_20": all(value >= 19 for value in retained),
        "aggregate_retains_at_least_1188_of_1200": total >= 1_188,
    }
    comparison_expected = {
        "schema_version": module.COMPARISON_MATRIX_SCHEMA,
        "registry_digest": module.FROZEN_REGISTRY_DIGEST,
        "producer_contract_digest": contract["contract_digest"],
        "semantic_seal_digest": semantic_seal["seal_digest"],
        "performance_final_digest": performance_final["final_digest"],
        "performance_passed": perf_passed,
        "results_opened_marker_digest": marker["result_digest"],
        "authority_contract_digest": authority_contract["contract_digest"],
        "authority_matrix_digest": authority_matrix["result_digest"],
        "authority_seal_digest": authority_seal["seal_digest"],
        "completed_cases": 60, "retained_total": total, "retained_denominator": 1_200,
        "minimum_retained_count": min(retained),
        "perfect_20_of_20_cases": sum(value == 20 for value in retained),
        "perfect_20_of_20_is_descriptive_only": True, "cases": comparison_cases,
        "failures": comparison_failures, "gates": comparison_gates,
        "passed": all(comparison_gates.values()), "candidate_results_opened": True,
        "production_promotion_authorized": False,
        "real_forward_outcomes_accessed": False,
    }
    comparison = {
        **comparison_expected, "created_at": "2026-01-01T00:00:01+00:00",
        "result_digest": module._hash(comparison_expected),
    }
    _write(comparison_root / "candidate-comparison.json", comparison)
    comparison_seal_expected = {
        "schema_version": module.COMPARISON_SEAL_SCHEMA,
        "registry_digest": module.FROZEN_REGISTRY_DIGEST,
        "producer_contract_digest": contract["contract_digest"],
        "comparison_digest": comparison["result_digest"],
        "results_opened_marker_digest": marker["result_digest"],
        "authority_seal_digest": authority_seal["seal_digest"],
        "candidate_results_opened": True,
        "comparison_gate_passed": comparison["passed"],
        "production_promotion_authorized": False,
    }
    comparison_seal = {
        **comparison_seal_expected, "created_at": "2026-01-01T00:00:02+00:00",
        "seal_digest": module._hash(comparison_seal_expected),
    }
    _write(comparison_root / "SEALED.json", comparison_seal)
    return {
        "config": config_path, "artifact_dir": artifact, "repository_root": repository,
        "registry_root": registry_root, "candidate_root": candidate_root,
        "authority_root": authority_root, "comparison_root": comparison_root,
        "output_root": output_root,
    }


def test_independent_verifier_accepts_terminal_performance_failure(tmp_path: Path) -> None:
    module = _module()
    paths = _build_synthetic(module, tmp_path, performance_fail=True)
    result = module.verify_comparison(**paths)
    assert result["passed"] is True
    assert result["comparison_gate_passed"] is True
    assert result["performance_terminal_fail_allowed"] is True
    assert (paths["output_root"] / "verification.json").is_file()


@pytest.mark.parametrize(
    ("retained", "comparison_passed"),
    [([18] + [20] * 59, False), ([19] * 13 + [20] * 47, False)],
)
def test_thresholds_are_reconstructed_exactly(
    tmp_path: Path, retained: list[int], comparison_passed: bool,
) -> None:
    module = _module()
    paths = _build_synthetic(module, tmp_path, retained)
    result = module.verify_comparison(**paths)
    assert result["passed"] is True
    assert result["comparison_gate_passed"] is comparison_passed


def test_nested_bundle_corruption_is_detected(tmp_path: Path) -> None:
    module = _module()
    paths = _build_synthetic(module, tmp_path)
    bundle_path = sorted((paths["candidate_root"] / "case-bundles").glob("*.json"))[0]
    bundle = json.loads(bundle_path.read_text())
    bundle["semantic"]["candidates"][0]["episode_id"] = "f" * 24
    _write(bundle_path, bundle)
    result = module.verify_comparison(**paths)
    assert result["passed"] is False
    assert "bundle" in result["failures"][0]


@pytest.mark.parametrize("mode", ["missing", "wrong"])
def test_missing_or_wrong_results_opened_marker_fails_closed(
    tmp_path: Path, mode: str,
) -> None:
    module = _module()
    paths = _build_synthetic(module, tmp_path)
    marker = paths["comparison_root"] / "RESULTS_OPENED.json"
    if mode == "missing":
        marker.unlink()
    else:
        payload = json.loads(marker.read_text())
        payload["status"] = "opened earlier"
        payload["result_digest"] = module._hash({
            key: value for key, value in payload.items()
            if key not in {"created_at", "result_digest"}
        })
        _write(marker, payload)
    result = module.verify_comparison(**paths)
    assert result["passed"] is False
    assert "RESULTS_OPENED" in result["failures"][0] or "FileNotFoundError" in result["failures"][0]


def test_comparator_shared_helper_tamper_cannot_change_verifier(tmp_path: Path) -> None:
    module = _module()
    paths = _build_synthetic(module, tmp_path)
    fake = type("Comparator", (), {"_authority_case_digest": staticmethod(lambda _: "0" * 64)})
    sys.modules["compare_m04r11_candidate_matrix_v2"] = fake
    try:
        result = module.verify_comparison(**paths)
    finally:
        sys.modules.pop("compare_m04r11_candidate_matrix_v2", None)
    assert result["passed"] is True
    source = Path(module.__file__).read_text()
    assert "from m04r11_candidate_v2_contract import" not in source
    assert "import m04r11_candidate_v2_contract" not in source
    assert "from compare_m04r11_candidate_matrix_v2 import" not in source


def test_exact_preregistration_policy_is_independently_frozen(tmp_path: Path) -> None:
    module = _module()
    paths = _build_synthetic(module, tmp_path)
    candidate_contract = paths["candidate_root"] / "candidate-contract.json"
    prereg_path = paths["repository_root"] / module.PREREGISTRATION_RELATIVE_PATH
    contract = json.loads(candidate_contract.read_text())
    contract["marker_and_resume_policy"]["resume_authorized"] = True
    contract["contract_digest"] = module._hash({
        key: value for key, value in contract.items() if key != "contract_digest"
    })
    prereg = json.loads(prereg_path.read_text())
    prereg["producer_contract"] = contract
    prereg["producer_contract_digest"] = contract["contract_digest"]
    prereg["marker_and_resume_policy"]["resume_authorized"] = True
    prereg["preregistration_digest"] = module._hash({
        key: value for key, value in prereg.items() if key != "preregistration_digest"
    })
    _write(candidate_contract, contract)
    _write(prereg_path, prereg)
    registry = json.loads((paths["registry_root"] / "query-registry.json").read_text())
    with pytest.raises(ValueError, match="exact contract"):
        module._validate_contract_and_prereg(
            registry, paths["artifact_dir"], paths["repository_root"], paths["candidate_root"],
        )


def test_resident_content_and_path_are_reconstructed_from_durable_pack(tmp_path: Path) -> None:
    module = _module()
    paths = _build_synthetic(module, tmp_path)
    contract = json.loads((paths["candidate_root"] / "candidate-contract.json").read_text())
    resident_path = paths["candidate_root"] / "RESIDENT_READY.json"
    resident = json.loads(resident_path.read_text())
    resident["resident_content_digest"] = "0" * 64
    resident["resident_ready_observation"]["content_digest"] = "0" * 64
    resident["binding_digest"] = module._hash({
        key: value for key, value in resident.items() if key != "binding_digest"
    })
    _write(resident_path, resident)
    with pytest.raises(ValueError, match="resident binding"):
        module._validate_resident(paths["candidate_root"], contract)


def test_exact_historical_ready_bytes_reconstruct_every_commitment(tmp_path: Path) -> None:
    module = _module()
    paths = _build_synthetic(module, tmp_path)
    contract = json.loads((paths["candidate_root"] / "candidate-contract.json").read_text())
    resident_path = paths["candidate_root"] / "RESIDENT_READY.json"
    resident = json.loads(resident_path.read_text())
    ready = base64.b64decode(resident["resident_ready_bytes_base64"])
    resident["resident_ready_bytes_base64"] = base64.b64encode(ready + b" ").decode("ascii")
    resident["binding_digest"] = module._hash({
        key: value for key, value in resident.items() if key != "binding_digest"
    })
    _write(resident_path, resident)
    with pytest.raises(ValueError, match="resident binding"):
        module._validate_resident(paths["candidate_root"], contract)


def test_query_context_and_scan_schema_are_not_self_attested(tmp_path: Path) -> None:
    module = _module()
    paths = _build_synthetic(module, tmp_path)
    registry = json.loads((paths["registry_root"] / "query-registry.json").read_text())
    contract = json.loads((paths["candidate_root"] / "candidate-contract.json").read_text())
    resident = module._validate_resident(paths["candidate_root"], contract)
    roles = module._roles(registry)
    execution = module._execution(roles)
    query_id = execution[0]
    case = next(row for row in registry["cases_data"] if row["episode_id"] == query_id)
    role = next(row for row in roles if row["query_episode_id"] == query_id)
    bundle = json.loads(sorted((paths["candidate_root"] / "case-bundles").glob("*.json"))[0].read_text())
    bundle["semantic"]["query_start_ns"] += 1
    bundle["semantic"]["semantic_digest"] = module._hash({
        key: value for key, value in bundle["semantic"].items()
        if key not in {"created_at", "semantic_digest"}
    })
    bundle["bundle_digest"] = module._hash({
        key: value for key, value in bundle.items() if key != "bundle_digest"
    })
    with pytest.raises(ValueError, match="semantic case identity"):
        module._validate_bundle(
            bundle, case, role, contract, resident, 0, module._query_context(paths["config"], case),
        )
    bundle = json.loads(sorted((paths["candidate_root"] / "case-bundles").glob("*.json"))[0].read_text())
    bundle["semantic"]["scan_semantics"][0]["schema_version"] = "wrong"
    bundle["semantic"]["scan_semantics"][1]["schema_version"] = "wrong"
    bundle["semantic"]["scan_semantics"][2]["schema_version"] = "wrong"
    for scan in bundle["semantic"]["scan_semantics"]:
        scan["result_digest"] = module._scan_digest(scan)
    bundle["semantic"]["semantic_digest"] = module._hash({
        key: value for key, value in bundle["semantic"].items()
        if key not in {"created_at", "semantic_digest"}
    })
    bundle["bundle_digest"] = module._hash({
        key: value for key, value in bundle.items() if key != "bundle_digest"
    })
    with pytest.raises(ValueError, match="scan semantics"):
        module._validate_bundle(
            bundle, case, role, contract, resident, 0, module._query_context(paths["config"], case),
        )


def test_authority_gates_and_measurement_integrity_are_reconstructed(tmp_path: Path) -> None:
    module = _module()
    paths = _build_synthetic(module, tmp_path)
    registry = json.loads((paths["registry_root"] / "query-registry.json").read_text())
    contexts = {row["episode_id"]: module._query_context(paths["config"], row) for row in registry["cases_data"]}
    case_path = sorted((paths["authority_root"] / "cases").glob("*.json"))[0]
    authority = json.loads(case_path.read_text())
    authority["gates"] = {"synthetic_exact": True}
    authority["result_digest"] = module._hash({
        key: value for key, value in authority.items() if key not in module.AUTHORITY_CASE_OMITTED
    })
    authority["checkpoint_integrity_digest"] = module._hash({
        key: value for key, value in authority.items() if key not in {"created_at", "checkpoint_integrity_digest"}
    })
    _write(case_path, authority)
    with pytest.raises(ValueError, match="authority case"):
        module._validate_authorities(registry, paths["authority_root"], paths["repository_root"], contexts)


@pytest.mark.parametrize("event_index", [0, 1, 121])
def test_every_ledger_stage_payload_is_reconstructed(tmp_path: Path, event_index: int) -> None:
    module = _module()
    paths = _build_synthetic(module, tmp_path)
    event_path = paths["candidate_root"] / "ledger/events" / f"{event_index:06d}.json"
    event = json.loads(event_path.read_text())
    event["details"] = {}
    _write(event_path, event)
    _rechain_ledger(module, paths["candidate_root"])
    registry = json.loads((paths["registry_root"] / "query-registry.json").read_text())
    contract = json.loads((paths["candidate_root"] / "candidate-contract.json").read_text())
    prereg = json.loads((paths["repository_root"] / module.PREREGISTRATION_RELATIVE_PATH).read_text())
    roles, execution = module._roles(registry), module._execution(module._roles(registry))
    resident = module._validate_resident(paths["candidate_root"], contract)
    contexts = {row["episode_id"]: module._query_context(paths["config"], row) for row in registry["cases_data"]}
    with pytest.raises(ValueError, match="(ledger run-start|ledger case-start|run-complete evidence)"):
        module._validate_candidate_aggregate(
            registry, paths["candidate_root"], contract, prereg, roles, execution,
            resident, contexts, paths["repository_root"],
        )


def test_authority_measurement_digest_is_not_self_attested(tmp_path: Path) -> None:
    module = _module()
    paths = _build_synthetic(module, tmp_path)
    registry = json.loads((paths["registry_root"] / "query-registry.json").read_text())
    contexts = {row["episode_id"]: module._query_context(paths["config"], row) for row in registry["cases_data"]}
    matrix_path = paths["authority_root"] / "authority-matrix.json"
    matrix = json.loads(matrix_path.read_text())
    matrix["measurements"][0]["search_seconds"] += 1
    _write(matrix_path, matrix)
    with pytest.raises(ValueError, match="authority aggregate"):
        module._validate_authorities(registry, paths["authority_root"], paths["repository_root"], contexts)


def test_extra_or_incomplete_candidate_tree_fails_closed(tmp_path: Path) -> None:
    module = _module()
    paths = _build_synthetic(module, tmp_path)
    _write(paths["candidate_root"] / "INCOMPLETE.json", {"status": "incomplete"})
    result = module.verify_comparison(**paths)
    assert result["passed"] is False
    assert "candidate terminal artifact tree differs" in result["failures"][0]


def test_repository_root_is_bound_before_any_input_read(tmp_path: Path) -> None:
    module = _module()
    paths = _build_synthetic(module, tmp_path)
    paths["repository_root"] = paths["authority_root"]
    with pytest.raises(ValueError, match="repository root differs"):
        module.verify_comparison(**paths)
