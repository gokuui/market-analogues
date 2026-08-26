"""Produce truth-blind M04R-11 v2 candidate evidence from a resident mirror.

The producer deliberately has no authority input.  It executes the 53 still-
untouched performance cases before the seven exposed recovery cases, writes
semantic and timing evidence separately, and publishes semantic/performance
terminal documents only after all sixty primary attempts are accounted for.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
from math import isfinite
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter
from typing import Any, Iterable, Mapping
from uuid import uuid4

import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import m04r11_candidate_v2_contract as v2_contract
from m04r11_candidate_v2_contract import (
    CASE_BUNDLE_SCHEMA, CLAIMS_POLICY, CONFIRMATORY_PERFORMANCE_ROLE,
    FROZEN_GENERATION_ID, FROZEN_PROPOSAL_CONTRACT_DIGEST,
    FROZEN_REGISTRY_DIGEST, FROZEN_ROUTE_QUOTAS,
    INCOMPLETE_SCHEMA, PERFORMANCE_ATTEMPT_GATES,
    PERFORMANCE_ATTEMPT_SCHEMA, PERFORMANCE_FINAL_SCHEMA,
    PERFORMANCE_LIMITS, PERFORMANCE_MATRIX_GATES, PERFORMANCE_MATRIX_SCHEMA,
    RESIDENT_BINDING_SCHEMA, RUN_COMPLETE_SCHEMA, RUN_LEDGER_EVENT_SCHEMA,
    RUN_LEDGER_HEAD_SCHEMA, SEMANTIC_CASE_GATES,
    SEMANTIC_CASE_SCHEMA, SEMANTIC_MATRIX_GATES, SEMANTIC_MATRIX_SCHEMA,
    SEMANTIC_SEAL_SCHEMA, derive_role_table,
    execution_query_ids, expected_roots, performance_attempt_digest,
    performance_matrix_digest, semantic_case_digest,
    semantic_matrix_digest, terminal_digest, validate_predecessor_paths,
    validate_producer_contract, validate_role_table,
)
from market_analogues.adapters import file_fingerprint, source_from_spec
from market_analogues.config import load_config
from market_analogues.episodes import build_episode
from market_analogues.m04r_candidate_evidence import (
    reconstructed_candidate_digest, scan_result_digest,
)
from market_analogues.m04r_validation_registry import validate_m04r_validation_registry
from market_analogues.packed_bound_search import (
    PackedBoundQuery, packed_bound_search_contract,
    scan_packed_bound_proposals_threaded,
)
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.representation import represent, representation_input_digest
from market_analogues.resident_store import (
    READY_SCHEMA_VERSION, observe_ready_strict,
    prepare_resident_mirror_observed, resident_file_identity_lease,
)
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import InstrumentKey, stable_hash


LEDGER_GENESIS = "0" * 64
IMPLEMENTATION_FILES = (
    "experiments/m04r/m04r11_candidate_matrix_v2.py",
    "experiments/m04r/m04r11_candidate_v2_contract.py",
    "experiments/m04r/compare_m04r11_candidate_matrix_v2.py",
    "experiments/m04r/verify_m04r11_candidate_comparison_v2.py",
    "experiments/m04r/prepare_m04r11_resident_mirror.py",
    "experiments/m04r/preregister_m04r11_candidate_v2.py",
    "experiments/m04r/preflight_m04r11_candidate_v2.py",
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("x") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json_create(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a complete JSON inode without ever replacing an existing target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _implementation_manifest() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    paths = [
        *(repository / name for name in IMPLEMENTATION_FILES),
        *sorted((repository / "src").rglob("*.py")),
    ]
    files = {
        str(path.relative_to(repository)): file_fingerprint(path)
        for path in paths
    }
    return {"files": files, "digest": stable_hash(files)}


def _environment_manifest() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "pandas", "numba"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    deterministic = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "cpu_count": os.cpu_count(),
        "numba_num_threads": os.environ.get("NUMBA_NUM_THREADS"),
    }
    return {**deterministic, "digest": stable_hash(deterministic)}


def _require_implementation_manifest(expected: Mapping[str, Any]) -> dict[str, Any]:
    observed = _implementation_manifest()
    if observed != dict(expected):
        raise ValueError("implementation files changed after preregistration")
    return observed


def _require_environment_manifest(expected: Mapping[str, Any]) -> dict[str, Any]:
    observed = _environment_manifest()
    if observed != dict(expected):
        raise ValueError("runtime environment changed after preregistration")
    return observed


def _preregistration_path() -> Path:
    repository = Path(__file__).resolve().parents[2]
    return v2_contract.expected_preregistration_path(repository)


def _git_preregistration_binding(
    repository: Path, preregistration: Path,
    implementation_files: Iterable[str] = (),
) -> dict[str, Any]:
    relative = str(preregistration.resolve().relative_to(repository.resolve()))

    def run(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments], cwd=repository, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )

    tracked = run("ls-files", "--error-unmatch", "--", relative)
    if tracked.returncode != 0:
        raise ValueError("committed preregistration is not tracked by git")
    if run("diff", "--quiet", "--", relative).returncode != 0:
        raise ValueError("committed preregistration has tracked worktree changes")
    if run("diff", "--cached", "--quiet", "--", relative).returncode != 0:
        raise ValueError("committed preregistration has staged changes")
    if run("diff", "--quiet").returncode != 0:
        raise ValueError("tracked worktree differs from HEAD before launch")
    if run("diff", "--cached", "--quiet").returncode != 0:
        raise ValueError("git index differs from HEAD before launch")
    head = run("rev-parse", "--verify", "HEAD")
    if head.returncode != 0:
        raise ValueError("cannot bind preregistration to repository HEAD")
    head_commit = head.stdout.strip()
    prereg_commit = run("log", "-1", "--format=%H", "--", relative)
    if prereg_commit.returncode != 0 or prereg_commit.stdout.strip() != head_commit:
        raise ValueError("preregistration was not committed at the launch HEAD")
    changed = run(
        "diff-tree", "--no-commit-id", "--name-only", "-r", head_commit,
    )
    changed_paths = sorted(filter(None, changed.stdout.splitlines()))
    if changed.returncode != 0 or changed_paths != [relative]:
        raise ValueError("preregistration must be the sole file in the launch HEAD commit")
    implementation_names = sorted(set(str(value) for value in implementation_files))
    if implementation_names:
        implementation_tracked = run(
            "ls-files", "--error-unmatch", "--", *implementation_names,
        )
        if implementation_tracked.returncode != 0:
            raise ValueError("a preregistered implementation file is not tracked by git")
    deterministic = {
        "repository_root": str(repository.resolve()),
        "head_commit": head_commit,
        "preregistration_relative_path": relative,
        "preregistration_blob_sha256": file_fingerprint(preregistration),
        "tracked_worktree_clean": True,
        "index_clean": True,
    }
    return {**deterministic, "binding_digest": stable_hash(deterministic)}


def _source_pack_binding(source_full_root: Path) -> dict[str, Any]:
    store = source_full_root / "store"
    loaded = load_packed_generation(
        store, FROZEN_GENERATION_ID, verify_content=True,
        validate_records=False,
    )
    manifest = loaded.manifest
    generation = store / "generations" / FROZEN_GENERATION_ID
    manifest_path = generation / "manifest.json"
    rows_path = generation / str(manifest["rows_file"])
    overflow_path = generation / str(manifest["overflow_file"])
    return {
        "generation_id": FROZEN_GENERATION_ID,
        "source_full_root": str(source_full_root.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": file_fingerprint(manifest_path),
        "manifest_digest": str(manifest["manifest_digest"]),
        "provenance_digest": str(manifest["provenance_digest"]),
        "rows_file": str(manifest["rows_file"]),
        "rows_path": str(rows_path.resolve()),
        "rows_bytes": int(manifest["rows_bytes"]),
        "rows_sha256": str(manifest["rows_sha256"]),
        "row_count": len(loaded.rows),
        "row_bytes": int(loaded.rows.dtype.itemsize),
        "overflow_file": str(manifest["overflow_file"]),
        "overflow_path": str(overflow_path.resolve()),
        "overflow_bytes": int(manifest["overflow_bytes"]),
        "overflow_sha256": str(manifest["overflow_sha256"]),
        "overflow_count": len(loaded.overflow),
        "overflow_row_bytes": int(loaded.overflow.dtype.itemsize),
        "physical_rows": len(loaded.rows) + len(loaded.overflow),
        "active_pointer_absent": not (store / "active.json").exists(),
    }


def _resident_binding(
    ready: Mapping[str, Any], *, contract_digest: str, ready_path: Path,
    validation_observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    seal = dict(ready["seal"])
    observation = _ready_observation(ready_path)
    ready_bytes = ready_path.read_bytes()
    if sha256(ready_bytes).hexdigest() != observation["ready_file_sha256"]:
        raise ValueError("resident READY changed while capturing durable bytes")
    if json.loads(ready_bytes) != dict(ready):
        raise ValueError("resident READY bytes differ from validated payload")
    deterministic = {
        "schema_version": RESIDENT_BINDING_SCHEMA,
        "producer_contract_digest": contract_digest,
        "resident_ready_path": str(ready_path.resolve()),
        "resident_ready_schema": ready["schema_version"],
        "resident_content_digest": ready["content_digest"],
        "resident_ready_observation": observation,
        "resident_ready_payload": dict(ready),
        "resident_ready_bytes_base64": base64.b64encode(ready_bytes).decode("ascii"),
        "validation_observation": (
            dict(validation_observation) if validation_observation is not None else None
        ),
        "generation_id": seal["generation_id"],
        "provenance_digest": seal["provenance_digest"],
        "mirror_store_root": seal["mirror_store_root"],
        "storage_class": seal["storage_class"],
        "latency_scope": seal["latency_scope"],
        "query_specific_inputs_used": seal["query_specific_inputs_used"],
        "outcomes_or_labels_used": seal["outcomes_or_labels_used"],
        "real_forward_outcomes_accessed": False,
    }
    return {**deterministic, "binding_digest": stable_hash(deterministic)}


def _validate_resident_binding(
    binding: Mapping[str, Any], contract: Mapping[str, Any], ready: Mapping[str, Any],
) -> tuple[str, ...]:
    failures: list[str] = []
    try:
        expected = _resident_binding(
            ready, contract_digest=str(contract["contract_digest"]),
            ready_path=Path(str(binding["resident_ready_path"])),
            validation_observation=binding.get("validation_observation"),
        )
        if dict(binding) != expected:
            failures.append("resident READY binding differs")
        if binding.get("schema_version") != RESIDENT_BINDING_SCHEMA:
            failures.append("resident binding schema differs")
        if not all((
            ready.get("schema_version") == READY_SCHEMA_VERSION,
            binding.get("resident_ready_schema") == READY_SCHEMA_VERSION,
            contract.get("resident_ready_schema") == READY_SCHEMA_VERSION,
        )):
            failures.append("actual resident READY schema differs from contract")
        if binding.get("generation_id") != contract.get("generation_id"):
            failures.append("resident generation differs from contract")
        if binding.get("resident_content_digest") != ready.get("content_digest"):
            failures.append("resident content digest differs")
        validation = binding.get("validation_observation")
        if not isinstance(validation, Mapping):
            failures.append("resident fresh validation observation is absent")
        else:
            deterministic = {
                key: value for key, value in validation.items()
                if key != "observation_digest"
            }
            if not all((
                validation.get("observation_digest") == stable_hash(deterministic),
                validation.get("ready_digest") == ready.get("ready_digest"),
                validation.get("content_digest") == ready.get("content_digest"),
                validation.get("seal_digest") == ready.get("seal_digest"),
                validation.get("reserve_bytes")
                == contract.get("resident_policy", {}).get("reserve_bytes"),
            )):
                failures.append("resident fresh validation observation differs")
        if binding.get("query_specific_inputs_used") is not False:
            failures.append("resident preparation used query-specific inputs")
        if binding.get("outcomes_or_labels_used") is not False:
            failures.append("resident preparation used outcomes or labels")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        failures.append(f"malformed resident binding:{type(exc).__name__}:{exc}")
    return tuple(sorted(set(failures)))


def _candidate_payload(report: Any) -> list[dict[str, Any]]:
    return [{
        "episode_id": row.episode_id,
        "symbol": row.symbol,
        "cutoff_ns": row.cutoff_ns,
        "quality_tier": row.quality_tier,
        "lower_bound_hex": row.lower_bound.hex(),
        "routes": list(row.routes),
        "overflow_fallback": row.overflow_fallback,
    } for row in report.candidates]


def _report_semantics(report: Any) -> dict[str, Any]:
    return {
        "schema_version": report.schema_version,
        "generation_id": report.generation_id,
        "query_episode_id": report.query_episode_id,
        "rows_scanned": report.rows_scanned,
        "eligible_rows": report.eligible_rows,
        "eligible_main_rows": report.eligible_main_rows,
        "eligible_overflow_rows": report.eligible_overflow_rows,
        "route_counts": dict(report.route_counts),
        "route_quotas": dict(report.route_quotas),
        "candidate_count": len(report.candidates),
        "candidate_digest": report.candidate_digest,
        "result_digest": report.result_digest,
    }


def _ready_observation(path: Path) -> dict[str, Any]:
    observed = observe_ready_strict(path)
    lease = resident_file_identity_lease(path)
    if not all((
        lease.get("ready_digest") == observed["ready_digest"],
        lease.get("ready_file_sha256") == observed["ready_file_sha256"],
        lease.get("content_digest") == observed["content_digest"],
        lease.get("files", {}).get("ready") == observed["identity"],
    )):
        raise ValueError("resident READY changed between observation and identity lease")
    return {
        "schema_version": READY_SCHEMA_VERSION,
        "content_digest": observed["content_digest"],
        "ready_digest": observed["ready_digest"],
        "seal_digest": observed["seal_digest"],
        "ready_file_sha256": observed["ready_file_sha256"],
        "ready_identity": observed["identity"],
        "file_identity_lease": lease,
    }


def _run_resident_scans(
    store_root: Path, query: PackedBoundQuery, route_quotas: Mapping[str, int],
    provenance_digest: str,
) -> tuple[Any, Any, Any]:
    """Execute the frozen three primary traversals without cache advice."""
    return tuple(
        scan_packed_bound_proposals_threaded(
            store_root, FROZEN_GENERATION_ID, query,
            route_quotas=route_quotas, block_rows=block_rows,
            block_order=order, threads=8, verify_content=False,
            expected_provenance_digest=provenance_digest,
        )
        for block_rows, order in (
            (4_096, "forward"), (4_097, "reverse"), (4_093, "forward"),
        )
    )  # type: ignore[return-value]


def _worker(
    config_path: str, resident_store_root: str, ready_path_text: str,
    case: dict[str, Any], role: dict[str, Any], route_quotas: dict[str, int],
    provenance_digest: str, physical_rows: int, resident_binding_digest: str,
    resident_content_digest: str, producer_contract_digest_value: str,
    expected_implementation_manifest: dict[str, Any],
    expected_ready_observation: dict[str, Any],
    expected_resident_binding: dict[str, Any],
) -> dict[str, Any]:
    task_started = perf_counter()
    ready_path = Path(ready_path_text)
    _require_implementation_manifest(expected_implementation_manifest)
    ready_start = _ready_observation(ready_path)
    binding_deterministic = {
        key: value for key, value in expected_resident_binding.items()
        if key != "binding_digest"
    }
    if not all((
        ready_start == expected_ready_observation,
        expected_resident_binding.get("binding_digest")
        == stable_hash(binding_deterministic),
        expected_resident_binding.get("producer_contract_digest")
        == producer_contract_digest_value,
        expected_resident_binding.get("resident_content_digest")
        == ready_start["content_digest"],
        expected_resident_binding.get("resident_ready_observation") == ready_start,
    )):
        raise ValueError("worker READY instance or resident binding differs before query")
    config = load_config(config_path)
    source = source_from_spec(config.datasets["nasdaq"])
    episode = build_episode(
        source, InstrumentKey("nasdaq", str(case["symbol"])),
        str(case["cutoff"]), int(case["lookback"]),
        str(case["representation_version"]),
    )
    if episode.key.id != case["episode_id"]:
        raise ValueError("rebuilt v2 query episode differs")
    instrument = InstrumentKey("nasdaq", str(case["symbol"]))
    stock_prefix = asdict(source.causal_prefix_fingerprint(
        instrument, str(case["cutoff"]),
    ))
    benchmark_raw = source.benchmark_causal_prefix_fingerprint(str(case["cutoff"]))
    benchmark_prefix = asdict(benchmark_raw) if benchmark_raw is not None else None
    if stock_prefix != case["stock_prefix"] or benchmark_prefix != case["benchmark_prefix"]:
        raise ValueError("v2 query causal prefix differs from frozen registry")
    representation = represent(episode)
    representation_digest = representation_input_digest(representation)
    latest = latest_eligible_cutoff(episode, 60)
    query = PackedBoundQuery(
        episode.key.id, episode.key.instrument.source_symbol,
        int(pd.Timestamp(episode.bars.timestamp.iloc[0]).value),
        int(latest.value), representation, ("A", "B"),
    )
    store_root = Path(resident_store_root)
    loaded = load_packed_generation(
        store_root, FROZEN_GENERATION_ID,
        expected_provenance_digest=provenance_digest,
        verify_content=False, validate_records=False,
    )
    scans = _run_resident_scans(
        store_root, query, route_quotas, provenance_digest,
    )
    ready_end = _ready_observation(ready_path)
    if ready_end != expected_ready_observation:
        raise ValueError("worker READY instance differs after query execution")
    task_seconds = perf_counter() - task_started
    peak_rss = max(
        *(report.peak_rss_mb for report in scans),
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1_024,
    )
    semantics = [_report_semantics(report) for report in scans]
    candidates = _candidate_payload(scans[0])
    candidate_ids = [row["episode_id"] for row in candidates]
    violations = {
        "duplicates": len(candidate_ids) - len(set(candidate_ids)),
        "future": sum(row["cutoff_ns"] > query.latest_eligible_ns for row in candidates),
        "same_symbol_overlap": sum(
            row["symbol"] == query.symbol and row["cutoff_ns"] >= query.query_start_ns
            for row in candidates
        ),
        "tier": sum(row["quality_tier"] not in query.quality_tiers for row in candidates),
    }
    candidate_digest = reconstructed_candidate_digest(candidates)
    semantic_gates = {
        "resident_content_matches_contract": (
            loaded.generation_id == FROZEN_GENERATION_ID
            and str(loaded.manifest["provenance_digest"]) == provenance_digest
            and ready_start["content_digest"] == resident_content_digest
            and ready_end["content_digest"] == resident_content_digest
        ),
        "query_identity_prefix_and_representation_match": (
            episode.key.id == case["episode_id"]
            and stock_prefix == case["stock_prefix"]
            and benchmark_prefix == case["benchmark_prefix"]
            and bool(representation_digest)
        ),
        "three_scan_digest_and_block_order_invariance": (
            semantics[0] == semantics[1] == semantics[2]
            and all(
                row["result_digest"]
                == scan_result_digest(row, FROZEN_PROPOSAL_CONTRACT_DIGEST)
                for row in semantics
            )
        ),
        "internal_eligible_row_accounting": (
            scans[0].eligible_rows
            == scans[0].eligible_main_rows + scans[0].eligible_overflow_rows
        ),
        "physical_row_accounting": scans[0].rows_scanned == physical_rows,
        "frozen_route_quotas": dict(scans[0].route_quotas) == route_quotas,
        "candidate_digest_order_and_routes_reconstruct": (
            candidate_digest == scans[0].candidate_digest
            and list(zip(
                [float.fromhex(row["lower_bound_hex"]) for row in candidates],
                candidate_ids,
            )) == sorted(zip(
                [float.fromhex(row["lower_bound_hex"]) for row in candidates],
                candidate_ids,
            ))
            and all(
                row["routes"] == sorted(set(row["routes"])) and row["routes"]
                for row in candidates
            )
        ),
        "zero_temporal_overlap_tier_duplicate_errors": not any(violations.values()),
        "real_forward_outcomes_excluded": all((
            loaded.manifest.get("real_forward_outcomes_accessed") is False,
            expected_resident_binding.get("outcomes_or_labels_used") is False,
            expected_resident_binding.get("real_forward_outcomes_accessed") is False,
        )),
    }
    if tuple(semantic_gates) != SEMANTIC_CASE_GATES:
        raise ValueError("v2 semantic gate names differ from frozen contract")
    semantic_deterministic = {
        "schema_version": SEMANTIC_CASE_SCHEMA,
        "producer_contract_digest": producer_contract_digest_value,
        "registry_digest": FROZEN_REGISTRY_DIGEST,
        "generation_id": FROZEN_GENERATION_ID,
        "proposal_contract_digest": FROZEN_PROPOSAL_CONTRACT_DIGEST,
        "resident_content_digest": resident_content_digest,
        "registry_case_id": case["case_id"],
        "query_episode_id": query.episode_id,
        "query_symbol": query.symbol,
        "query_start_ns": query.query_start_ns,
        "latest_eligible_ns": query.latest_eligible_ns,
        "query_stock_prefix": stock_prefix,
        "query_benchmark_prefix": benchmark_prefix,
        "query_representation_digest": representation_digest,
        "performance_role": role["performance_role"],
        "recall_role": role["recall_role"],
        "scan_semantics": semantics,
        "candidates": candidates,
        "candidate_digest_reconstructed": candidate_digest,
        "violations": violations,
        "gates": semantic_gates,
        "passed": all(semantic_gates.values()),
        "real_forward_outcomes_accessed": False,
    }
    semantic_payload = {
        **semantic_deterministic,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "semantic_digest": stable_hash(semantic_deterministic),
    }
    timings = {
        "resident_first_seconds": scans[0].elapsed_seconds,
        "resident_reverse_seconds": scans[1].elapsed_seconds,
        "resident_repeat_seconds": scans[2].elapsed_seconds,
        "task_seconds": task_seconds,
        "peak_rss_mb": peak_rss,
    }
    finite = all(isfinite(float(value)) and float(value) >= 0 for value in timings.values())
    performance_gates = {
        "same_ready_instance_at_start_and_end": ready_start == ready_end,
        "measurements_finite_nonnegative_and_task_contains_scans": (
            finite and task_seconds >= sum(report.elapsed_seconds for report in scans)
        ),
        "resident_first_scan_at_most_120_seconds": (
            finite and scans[0].elapsed_seconds <= PERFORMANCE_LIMITS["resident_first_seconds"]
        ),
        "resident_repeat_scan_at_most_60_seconds": (
            finite and scans[2].elapsed_seconds <= PERFORMANCE_LIMITS["resident_repeat_seconds"]
        ),
        "worker_rss_at_most_1536_mib": (
            finite and peak_rss <= PERFORMANCE_LIMITS["worker_rss_mib"]
        ),
        "primary_attempt_completed": (
            len(scans) == 3
            and all(report.query_episode_id == query.episode_id for report in scans)
            and ready_start == ready_end == expected_ready_observation
        ),
    }
    if tuple(performance_gates) != PERFORMANCE_ATTEMPT_GATES:
        raise ValueError("v2 performance gate names differ from frozen contract")
    performance_deterministic = {
        "schema_version": PERFORMANCE_ATTEMPT_SCHEMA,
        "producer_contract_digest": producer_contract_digest_value,
        "registry_digest": FROZEN_REGISTRY_DIGEST,
        "generation_id": FROZEN_GENERATION_ID,
        "resident_binding_digest": resident_binding_digest,
        "semantic_digest": semantic_payload["semantic_digest"],
        "registry_case_id": case["case_id"],
        "query_episode_id": query.episode_id,
        "performance_role": role["performance_role"],
        "attempt_ordinal": 1,
        "ready_start": ready_start,
        "ready_end": ready_end,
        "timings": timings,
        "gates": performance_gates,
        "passed": all(performance_gates.values()),
        "real_forward_outcomes_accessed": False,
    }
    performance_payload = {
        **performance_deterministic,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "performance_digest": stable_hash(performance_deterministic),
    }
    return {"semantic": semantic_payload, "performance": performance_payload}


def _expected_query_context(
    config_path: Path | str, case: Mapping[str, Any],
) -> dict[str, Any]:
    config = load_config(config_path)
    source = source_from_spec(config.datasets["nasdaq"])
    instrument = InstrumentKey("nasdaq", str(case["symbol"]))
    episode = build_episode(
        source, instrument, str(case["cutoff"]), int(case["lookback"]),
        str(case["representation_version"]),
    )
    stock_prefix = asdict(source.causal_prefix_fingerprint(
        instrument, str(case["cutoff"]),
    ))
    raw_benchmark = source.benchmark_causal_prefix_fingerprint(str(case["cutoff"]))
    benchmark_prefix = asdict(raw_benchmark) if raw_benchmark is not None else None
    representation = represent(episode)
    return {
        "query_episode_id": episode.key.id,
        "query_symbol": episode.key.instrument.source_symbol,
        "query_start_ns": int(pd.Timestamp(episode.bars.timestamp.iloc[0]).value),
        "latest_eligible_ns": int(latest_eligible_cutoff(episode, 60).value),
        "query_stock_prefix": stock_prefix,
        "query_benchmark_prefix": benchmark_prefix,
        "query_representation_digest": representation_input_digest(representation),
    }


def _exact_keys(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(payload) != expected:
        raise ValueError(f"{label} fields differ")


def _validate_timestamp(value: Any, label: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{label} timestamp differs")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} timestamp is malformed") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} timestamp lacks timezone")


def _strict_validate_case_evidence(
    semantic: Mapping[str, Any], performance: Mapping[str, Any], *,
    contract: Mapping[str, Any], case: Mapping[str, Any], role: Mapping[str, Any],
    expected_query: Mapping[str, Any], resident: Mapping[str, Any],
    physical_rows: int,
) -> None:
    _exact_keys(dict(semantic), {
        "schema_version", "producer_contract_digest", "registry_digest",
        "generation_id", "proposal_contract_digest", "resident_content_digest",
        "registry_case_id", "query_episode_id", "query_symbol",
        "query_start_ns", "latest_eligible_ns", "query_stock_prefix",
        "query_benchmark_prefix", "query_representation_digest",
        "performance_role", "recall_role", "scan_semantics", "candidates",
        "candidate_digest_reconstructed", "violations", "gates", "passed",
        "real_forward_outcomes_accessed", "created_at", "semantic_digest",
    }, "semantic case")
    _validate_timestamp(semantic["created_at"], "semantic case")
    if not all((
        semantic.get("schema_version") == SEMANTIC_CASE_SCHEMA,
        semantic.get("producer_contract_digest") == contract["contract_digest"],
        semantic.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
        semantic.get("generation_id") == FROZEN_GENERATION_ID,
        semantic.get("proposal_contract_digest") == FROZEN_PROPOSAL_CONTRACT_DIGEST,
        semantic.get("resident_content_digest") == resident["resident_content_digest"],
        semantic.get("registry_case_id") == case["case_id"],
        semantic.get("performance_role") == role["performance_role"],
        semantic.get("recall_role") == role["recall_role"],
        semantic.get("real_forward_outcomes_accessed") is False,
        all(semantic.get(key) == value for key, value in expected_query.items()),
        semantic.get("semantic_digest") == semantic_case_digest(semantic),
    )):
        raise ValueError("semantic identity, provenance or digest differs")

    scans = list(semantic["scan_semantics"])
    if len(scans) != 3:
        raise ValueError("semantic case does not contain exactly three scans")
    scan_fields = {
        "schema_version", "generation_id", "query_episode_id", "rows_scanned",
        "eligible_rows", "eligible_main_rows", "eligible_overflow_rows",
        "route_counts", "route_quotas", "candidate_count", "candidate_digest",
        "result_digest",
    }
    for scan in scans:
        _exact_keys(dict(scan), scan_fields, "scan semantic")
        if not all((
            scan.get("schema_version") == "m04r-global-bound-proposal-v1",
            scan.get("generation_id") == FROZEN_GENERATION_ID,
            scan.get("query_episode_id") == case["episode_id"],
            scan.get("rows_scanned") == physical_rows,
            scan.get("route_quotas") == dict(FROZEN_ROUTE_QUOTAS),
            scan.get("eligible_rows")
            == scan.get("eligible_main_rows") + scan.get("eligible_overflow_rows"),
            scan.get("result_digest")
            == scan_result_digest(scan, FROZEN_PROPOSAL_CONTRACT_DIGEST),
            all(
                type(scan.get(name)) is int and scan[name] >= 0
                for name in (
                    "rows_scanned", "eligible_rows", "eligible_main_rows",
                    "eligible_overflow_rows", "candidate_count",
                )
            ),
            all(
                type(value) is int and value >= 0
                for value in scan.get("route_counts", {}).values()
            ),
        )):
            raise ValueError("scan identity, accounting or result digest differs")

    candidates = list(semantic["candidates"])
    candidate_fields = {
        "episode_id", "symbol", "cutoff_ns", "quality_tier",
        "lower_bound_hex", "routes", "overflow_fallback",
    }
    bounds: list[float] = []
    ids: list[str] = []
    for candidate in candidates:
        _exact_keys(dict(candidate), candidate_fields, "candidate")
        bound = float.fromhex(str(candidate["lower_bound_hex"]))
        episode_id = str(candidate["episode_id"])
        if not all((
            isfinite(bound) and bound >= 0,
            len(episode_id) == 24,
            episode_id == episode_id.lower(),
            len(bytes.fromhex(episode_id)) == 12,
            type(candidate["cutoff_ns"]) is int,
            isinstance(candidate["symbol"], str) and bool(candidate["symbol"]),
            candidate["quality_tier"] in ("A", "B"),
            type(candidate["overflow_fallback"]) is bool,
            isinstance(candidate["routes"], list),
            candidate["routes"] == sorted(set(candidate["routes"])),
            bool(candidate["routes"]),
            set(candidate["routes"]).issubset(FROZEN_ROUTE_QUOTAS),
        )):
            raise ValueError("candidate metadata, routes or bound differs")
        bounds.append(bound)
        ids.append(episode_id)
    if list(zip(bounds, ids)) != sorted(zip(bounds, ids)):
        raise ValueError("candidate ordering differs")
    candidate_digest = reconstructed_candidate_digest(candidates)
    route_counts = {
        route_name: sum(route_name in row["routes"] for row in candidates)
        for route_name in FROZEN_ROUTE_QUOTAS
    }
    violations = {
        "duplicates": len(ids) - len(set(ids)),
        "future": sum(
            row["cutoff_ns"] > expected_query["latest_eligible_ns"]
            for row in candidates
        ),
        "same_symbol_overlap": sum(
            row["symbol"] == expected_query["query_symbol"]
            and row["cutoff_ns"] >= expected_query["query_start_ns"]
            for row in candidates
        ),
        "tier": sum(row["quality_tier"] not in ("A", "B") for row in candidates),
    }
    scan_invariant = scans[0] == scans[1] == scans[2]
    candidate_valid = all((
        candidate_digest == semantic.get("candidate_digest_reconstructed"),
        candidate_digest == scans[0].get("candidate_digest"),
        len(candidates) == scans[0].get("candidate_count"),
        route_counts == scans[0].get("route_counts"),
        all(
            route_counts[name] <= FROZEN_ROUTE_QUOTAS[name]
            for name in FROZEN_ROUTE_QUOTAS
        ),
    ))
    expected_semantic_gates = {
        "resident_content_matches_contract": (
            semantic.get("resident_content_digest")
            == resident["resident_content_digest"]
        ),
        "query_identity_prefix_and_representation_match": all(
            semantic.get(key) == value for key, value in expected_query.items()
        ),
        "three_scan_digest_and_block_order_invariance": (
            scan_invariant and all(
                row["result_digest"]
                == scan_result_digest(row, FROZEN_PROPOSAL_CONTRACT_DIGEST)
                for row in scans
            )
        ),
        "internal_eligible_row_accounting": (
            scans[0]["eligible_rows"]
            == scans[0]["eligible_main_rows"] + scans[0]["eligible_overflow_rows"]
        ),
        "physical_row_accounting": scans[0]["rows_scanned"] == physical_rows,
        "frozen_route_quotas": scans[0]["route_quotas"] == dict(FROZEN_ROUTE_QUOTAS),
        "candidate_digest_order_and_routes_reconstruct": candidate_valid,
        "zero_temporal_overlap_tier_duplicate_errors": not any(violations.values()),
        "real_forward_outcomes_excluded": (
            semantic.get("real_forward_outcomes_accessed") is False
        ),
    }
    if not all((
        semantic.get("violations") == violations,
        semantic.get("gates") == expected_semantic_gates,
        semantic.get("passed") is all(expected_semantic_gates.values()),
    )):
        raise ValueError("semantic rows, violations or gates differ")

    _exact_keys(dict(performance), {
        "schema_version", "producer_contract_digest", "registry_digest",
        "generation_id", "resident_binding_digest", "semantic_digest",
        "registry_case_id", "query_episode_id", "performance_role",
        "attempt_ordinal", "ready_start", "ready_end", "timings", "gates",
        "passed", "real_forward_outcomes_accessed", "created_at",
        "performance_digest",
    }, "performance attempt")
    _validate_timestamp(performance["created_at"], "performance attempt")
    timings = dict(performance["timings"])
    _exact_keys(timings, {
        "resident_first_seconds", "resident_reverse_seconds",
        "resident_repeat_seconds", "task_seconds", "peak_rss_mb",
    }, "performance timings")
    if any(type(value) not in {int, float} for value in timings.values()):
        raise ValueError("performance timing types differ")
    numeric = [float(value) for value in timings.values()]
    finite = all(isfinite(value) and value >= 0 for value in numeric)
    expected_performance_gates = {
        "same_ready_instance_at_start_and_end": (
            performance.get("ready_start") == resident["resident_ready_observation"]
            and performance.get("ready_end") == resident["resident_ready_observation"]
        ),
        "measurements_finite_nonnegative_and_task_contains_scans": (
            finite and timings["task_seconds"] >= sum(
                timings[name] for name in (
                    "resident_first_seconds", "resident_reverse_seconds",
                    "resident_repeat_seconds",
                )
            )
        ),
        "resident_first_scan_at_most_120_seconds": (
            finite and timings["resident_first_seconds"]
            <= PERFORMANCE_LIMITS["resident_first_seconds"]
        ),
        "resident_repeat_scan_at_most_60_seconds": (
            finite and timings["resident_repeat_seconds"]
            <= PERFORMANCE_LIMITS["resident_repeat_seconds"]
        ),
        "worker_rss_at_most_1536_mib": (
            finite and timings["peak_rss_mb"] <= PERFORMANCE_LIMITS["worker_rss_mib"]
        ),
        "primary_attempt_completed": (
            len(scans) == 3 and performance.get("attempt_ordinal") == 1
        ),
    }
    if not all((
        performance.get("schema_version") == PERFORMANCE_ATTEMPT_SCHEMA,
        performance.get("producer_contract_digest") == contract["contract_digest"],
        performance.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
        performance.get("generation_id") == FROZEN_GENERATION_ID,
        performance.get("resident_binding_digest") == resident["binding_digest"],
        performance.get("semantic_digest") == semantic["semantic_digest"],
        performance.get("registry_case_id") == case["case_id"],
        performance.get("query_episode_id") == case["episode_id"],
        performance.get("performance_role") == role["performance_role"],
        type(performance.get("attempt_ordinal")) is int,
        performance.get("attempt_ordinal") == 1,
        performance.get("real_forward_outcomes_accessed") is False,
        performance.get("gates") == expected_performance_gates,
        performance.get("passed") is all(expected_performance_gates.values()),
        performance.get("performance_digest")
        == performance_attempt_digest(performance),
    )):
        raise ValueError("performance identity, measurements, gates or digest differs")


def _bundle_digest(payload: Mapping[str, Any]) -> str:
    return stable_hash({
        key: value for key, value in payload.items() if key != "bundle_digest"
    })


def _case_bundle_path(root: Path, execution_ordinal: int, query_id: str) -> Path:
    if not (0 <= execution_ordinal < 60):
        raise ValueError("case bundle execution ordinal is outside 0..59")
    if len(query_id) != 24 or query_id.lower() != query_id:
        raise ValueError("case bundle query ID is invalid")
    bytes.fromhex(query_id)
    return root / "case-bundles" / f"{execution_ordinal:03d}-{query_id}.json"


def _case_bundle(
    contract_digest: str, execution_ordinal: int,
    semantic: Mapping[str, Any], performance: Mapping[str, Any],
) -> dict[str, Any]:
    deterministic = {
        "schema_version": CASE_BUNDLE_SCHEMA,
        "producer_contract_digest": contract_digest,
        "execution_ordinal": execution_ordinal,
        "query_episode_id": semantic["query_episode_id"],
        "semantic": dict(semantic),
        "performance": dict(performance),
    }
    return {**deterministic, "bundle_digest": stable_hash(deterministic)}


def _validate_case_bundle(
    payload: Mapping[str, Any], *, contract: Mapping[str, Any],
    case: Mapping[str, Any], role: Mapping[str, Any],
    expected_query: Mapping[str, Any], resident: Mapping[str, Any],
    physical_rows: int, execution_ordinal: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _exact_keys(dict(payload), {
        "schema_version", "producer_contract_digest", "execution_ordinal",
        "query_episode_id",
        "semantic", "performance", "bundle_digest",
    }, "case bundle")
    if not all((
        payload.get("schema_version") == CASE_BUNDLE_SCHEMA,
        payload.get("producer_contract_digest") == contract["contract_digest"],
        payload.get("execution_ordinal") == execution_ordinal,
        payload.get("query_episode_id") == case["episode_id"],
        payload.get("bundle_digest") == _bundle_digest(payload),
    )):
        raise ValueError("case bundle identity or digest differs")
    semantic = dict(payload["semantic"])
    performance = dict(payload["performance"])
    _strict_validate_case_evidence(
        semantic, performance, contract=contract, case=case, role=role,
        expected_query=expected_query, resident=resident,
        physical_rows=physical_rows,
    )
    return semantic, performance


def _ledger_event_digest(payload: Mapping[str, Any]) -> str:
    return stable_hash({
        key: value for key, value in payload.items() if key != "event_digest"
    })


def _ledger_head_digest(payload: Mapping[str, Any]) -> str:
    return stable_hash({
        key: value for key, value in payload.items() if key != "head_digest"
    })


def _load_ledger(root: Path, contract_digest: str) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    events_root = root / "ledger" / "events"
    paths = sorted(events_root.glob("*.json")) if events_root.exists() else []
    events: list[dict[str, Any]] = []
    previous = LEDGER_GENESIS
    for index, path in enumerate(paths):
        if path.name != f"{index:06d}.json":
            raise ValueError("v2 ledger filenames are not contiguous")
        event = json.loads(path.read_text())
        if set(event) != {
            "schema_version", "producer_contract_digest", "event_index",
            "previous_event_digest", "event_type", "details", "created_at",
            "event_digest",
        } or not isinstance(event.get("details"), dict):
            raise ValueError("v2 ledger event fields differ")
        _validate_timestamp(event.get("created_at"), "ledger event")
        if not all((
            event.get("schema_version") == RUN_LEDGER_EVENT_SCHEMA,
            event.get("event_index") == index,
            event.get("previous_event_digest") == previous,
            event.get("producer_contract_digest") == contract_digest,
            event.get("event_digest") == _ledger_event_digest(event),
        )):
            raise ValueError("v2 ledger chain differs")
        events.append(event)
        previous = str(event["event_digest"])
    head_path = root / "ledger" / "HEAD.json"
    head = json.loads(head_path.read_text()) if head_path.exists() else None
    if head is not None and set(head) != {
        "schema_version", "producer_contract_digest", "event_count",
        "last_event_digest", "head_digest",
    }:
        raise ValueError("v2 ledger head fields differ")
    if head is not None and not all((
        head.get("schema_version") == RUN_LEDGER_HEAD_SCHEMA,
        head.get("producer_contract_digest") == contract_digest,
        head.get("event_count") == len(events),
        head.get("last_event_digest") == previous,
        head.get("head_digest") == _ledger_head_digest(head),
    )):
        raise ValueError("v2 ledger head differs")
    if events and head is None:
        raise ValueError("v2 ledger events exist without HEAD")
    return events, head


def _append_ledger_event(
    root: Path, contract_digest: str, event_type: str,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    events, head = _load_ledger(root, contract_digest)
    previous = str(head["last_event_digest"]) if head else LEDGER_GENESIS
    deterministic = {
        "schema_version": RUN_LEDGER_EVENT_SCHEMA,
        "producer_contract_digest": contract_digest,
        "event_index": len(events),
        "previous_event_digest": previous,
        "event_type": event_type,
        "details": dict(details),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    event = {**deterministic, "event_digest": stable_hash(deterministic)}
    path = root / "ledger" / "events" / f"{len(events):06d}.json"
    _atomic_json_create(path, event)
    head_deterministic = {
        "schema_version": RUN_LEDGER_HEAD_SCHEMA,
        "producer_contract_digest": contract_digest,
        "event_count": len(events) + 1,
        "last_event_digest": event["event_digest"],
    }
    _atomic_json(
        root / "ledger" / "HEAD.json",
        {**head_deterministic, "head_digest": stable_hash(head_deterministic)},
    )
    return event


def _validate_execution_ledger(
    events: list[Mapping[str, Any]], *, contract: Mapping[str, Any],
    preregistration: Mapping[str, Any], git_binding: Mapping[str, Any],
    resident: Mapping[str, Any], execution: list[str],
    cases_by_id: Mapping[str, Mapping[str, Any]],
    roles_by_id: Mapping[str, Mapping[str, Any]],
    bundles_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    expected_types = ["run_started"] + [
        event_type for _ in execution
        for event_type in ("case_started", "case_completed")
    ]
    if [event.get("event_type") for event in events] != expected_types:
        raise ValueError("ledger start/completion sequence differs before sealing")
    run_details = dict(events[0]["details"])
    expected_run_details = {
        "preregistration_digest": preregistration["preregistration_digest"],
        "git_binding": dict(git_binding),
        "resident_binding_digest": resident["binding_digest"],
        "resident_content_digest": resident["resident_content_digest"],
        "execution_query_ids_digest": contract["execution_query_ids_digest"],
    }
    if run_details != expected_run_details:
        raise ValueError("ledger run-start binding differs")
    for ordinal, query_id in enumerate(execution):
        case = cases_by_id[query_id]
        role = roles_by_id[query_id]
        bundle = bundles_by_id[query_id]
        started = dict(events[1 + ordinal * 2]["details"])
        expected_started = {
            "execution_ordinal": ordinal,
            "registry_case_id": case["case_id"],
            "query_episode_id": query_id,
            "performance_role": role["performance_role"],
            "implementation_manifest_digest": contract["implementation_manifest"]["digest"],
            "resident_ready_digest": resident["resident_ready_observation"]["ready_digest"],
            "resident_content_digest": resident["resident_content_digest"],
        }
        if started != expected_started:
            raise ValueError("ledger case-start binding differs")
        semantic = bundle["semantic"]
        performance = bundle["performance"]
        completed = dict(events[2 + ordinal * 2]["details"])
        expected_completed = {
            "execution_ordinal": ordinal,
            "registry_case_id": case["case_id"],
            "query_episode_id": query_id,
            "performance_role": role["performance_role"],
            "bundle_digest": bundle["bundle_digest"],
            "semantic_digest": semantic["semantic_digest"],
            "performance_digest": performance["performance_digest"],
            "semantic_passed": semantic["passed"],
            "performance_passed": performance["passed"],
        }
        if completed != expected_completed:
            raise ValueError("ledger case-completion binding differs")


def _validate_incomplete(payload: Mapping[str, Any], contract_digest: str) -> None:
    _exact_keys(dict(payload), {
        "schema_version", "producer_contract_digest", "resident_binding_digest",
        "status", "registry_case_id", "query_episode_id", "reason",
        "error_type", "ledger_event_count", "ledger_head_digest",
        "ledger_valid", "incomplete_event_appended", "resume_authorized",
        "semantic_seal_written",
        "authority_results_opened", "production_promotion_authorized",
        "created_at", "incomplete_digest",
    }, "INCOMPLETE marker")
    deterministic = {
        key: value for key, value in payload.items()
        if key not in {"created_at", "incomplete_digest"}
    }
    if not all((
        payload.get("schema_version") == INCOMPLETE_SCHEMA,
        payload.get("producer_contract_digest") == contract_digest,
        payload.get("status") == "incomplete_fail_closed",
        payload.get("resume_authorized") is False,
        type(payload.get("semantic_seal_written")) is bool,
        payload.get("authority_results_opened") is False,
        payload.get("production_promotion_authorized") is False,
        payload.get("incomplete_digest") == stable_hash(deterministic),
    )):
        raise ValueError("INCOMPLETE marker differs")


def _best_effort_mark_incomplete(
    root: Path, contract: Mapping[str, Any], resident: Mapping[str, Any], *,
    case_id: str | None, query_episode_id: str | None, reason: str,
    error_type: str | None,
) -> bool:
    contract_digest = str(contract["contract_digest"])
    incomplete_event_appended = False
    try:
        _append_ledger_event(root, contract_digest, "run_incomplete", {
            "registry_case_id": case_id,
            "query_episode_id": query_episode_id,
            "reason": reason,
            "error_type": error_type,
        })
        incomplete_event_appended = True
    except BaseException:
        pass
    ledger_valid = False
    event_count = 0
    head_digest = None
    try:
        events, head = _load_ledger(root, contract_digest)
        ledger_valid = True
        event_count = len(events)
        head_digest = head["head_digest"] if head else None
    except BaseException:
        pass
    deterministic = {
        "schema_version": INCOMPLETE_SCHEMA,
        "producer_contract_digest": contract_digest,
        "resident_binding_digest": resident.get("binding_digest"),
        "status": "incomplete_fail_closed",
        "registry_case_id": case_id,
        "query_episode_id": query_episode_id,
        "reason": reason,
        "error_type": error_type,
        "ledger_event_count": event_count,
        "ledger_head_digest": head_digest,
        "ledger_valid": ledger_valid,
        "incomplete_event_appended": incomplete_event_appended,
        "resume_authorized": False,
        "semantic_seal_written": (root / "SEMANTIC_SEALED.json").is_file(),
        "authority_results_opened": False,
        "production_promotion_authorized": False,
    }
    marker = {
        **deterministic,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "incomplete_digest": stable_hash(deterministic),
    }
    try:
        path = root / "INCOMPLETE.json"
        if path.exists():
            _validate_incomplete(json.loads(path.read_text()), contract_digest)
        else:
            try:
                _atomic_json_create(path, marker)
            except FileExistsError:
                pass
            _validate_incomplete(json.loads(path.read_text()), contract_digest)
        return True
    except BaseException:
        return False


def _semantic_matrix(
    contract: Mapping[str, Any], resident_start: Mapping[str, Any],
    resident_end: Mapping[str, Any], registry: Mapping[str, Any],
    cases_by_id: Mapping[str, Mapping[str, Any]], events: Iterable[Mapping[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    ordered_ids = [str(case["episode_id"]) for case in registry["cases_data"]]
    ordered = [dict(cases_by_id[value]) for value in ordered_ids]
    completed = [
        event["details"].get("semantic_digest") for event in events
        if event.get("event_type") == "case_completed"
    ]
    gates = {
        "all_60_semantic_cases_present_in_registry_order": (
            len(ordered) == 60
            and [row["query_episode_id"] for row in ordered] == ordered_ids
        ),
        "all_semantic_case_gates_passed": all(row.get("passed") is True for row in ordered),
        "mirror_content_unchanged_before_and_after": (
            resident_start["resident_content_digest"]
            == resident_end["resident_content_digest"]
        ),
        "ledger_semantic_completion_matches_cases": (
            completed == [cases_by_id[value]["semantic_digest"] for value in contract["execution_query_ids"]]
        ),
        "real_forward_outcomes_excluded": all(
            row.get("real_forward_outcomes_accessed") is False for row in ordered
        ),
    }
    if tuple(gates) != SEMANTIC_MATRIX_GATES:
        raise ValueError("semantic matrix gates differ from frozen contract")
    deterministic = {
        "schema_version": SEMANTIC_MATRIX_SCHEMA,
        "producer_contract_digest": contract["contract_digest"],
        "resident_start_content_digest": resident_start["resident_content_digest"],
        "resident_end_content_digest": resident_end["resident_content_digest"],
        "ordered_query_episode_ids": ordered_ids,
        "semantic_case_digests": [row["semantic_digest"] for row in ordered],
        "gates": gates,
        "passed": all(gates.values()),
        "real_forward_outcomes_accessed": False,
    }
    return {
        **deterministic, "elapsed_seconds": elapsed_seconds,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result_digest": semantic_matrix_digest(deterministic),
    }


def _performance_matrix(
    contract: Mapping[str, Any], registry: Mapping[str, Any],
    attempts_by_id: Mapping[str, Mapping[str, Any]], elapsed_seconds: float,
) -> dict[str, Any]:
    roles = {row["query_episode_id"]: row for row in contract["role_table"]}
    ordered_ids = [str(case["episode_id"]) for case in registry["cases_data"]]
    attempts = [dict(attempts_by_id[value]) for value in ordered_ids]
    confirmatory = [
        row for row in attempts
        if roles[row["query_episode_id"]]["performance_role"]
        == CONFIRMATORY_PERFORMANCE_ROLE
    ]
    exposed = [row for row in attempts if row not in confirmatory]
    roles_match = all(
        row.get("performance_role")
        == roles[row["query_episode_id"]]["performance_role"]
        for row in attempts
    )
    ready_digests = {
        (stable_hash(row["ready_start"]), stable_hash(row["ready_end"]))
        for row in attempts
    }
    gates = {
        "exact_7_exposed_53_confirmatory_role_partition": (
            len(exposed) == 7 and len(confirmatory) == 53 and roles_match
        ),
        "all_60_primary_attempts_accounted": (
            len(attempts) == 60
            and all(row.get("attempt_ordinal") == 1 for row in attempts)
        ),
        "all_53_confirmatory_primary_attempts_passed": all(
            row.get("passed") is True for row in confirmatory
        ),
        "all_7_exposed_regression_attempts_passed": all(
            row.get("passed") is True for row in exposed
        ),
        "all_60_operational_limits_passed": all(
            row.get("passed") is True for row in attempts
        ),
        "single_ready_instance_for_all_primary_attempts": (
            len(ready_digests) == 1 and next(iter(ready_digests))[0] == next(iter(ready_digests))[1]
        ),
    }
    if tuple(gates) != PERFORMANCE_MATRIX_GATES:
        raise ValueError("performance matrix gates differ from frozen contract")
    deterministic = {
        "schema_version": PERFORMANCE_MATRIX_SCHEMA,
        "producer_contract_digest": contract["contract_digest"],
        "ordered_query_episode_ids": ordered_ids,
        "attempt_digests": [row["performance_digest"] for row in attempts],
        "confirmatory_query_episode_ids": [row["query_episode_id"] for row in confirmatory],
        "exposed_query_episode_ids": [row["query_episode_id"] for row in exposed],
        "gates": gates,
        "passed": all(gates.values()),
        "claims_policy": CLAIMS_POLICY,
        "real_forward_outcomes_accessed": False,
    }
    return {
        **deterministic, "elapsed_seconds": elapsed_seconds,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result_digest": performance_matrix_digest(deterministic),
    }


def _terminal_document(
    schema: str, digest_field: str, deterministic: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": schema, **dict(deterministic),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    payload[digest_field] = terminal_digest(payload, digest_field)
    return payload


def _assert_fresh_output(root: Path) -> None:
    forbidden = (
        "INCOMPLETE.json", "SEMANTIC_SEALED.json", "PERFORMANCE_FINAL.json",
        "RUN_COMPLETE.json",
    )
    if any((root / name).exists() for name in forbidden):
        raise ValueError("v2 candidate root is terminal; rerun is forbidden")
    if (root / "ledger" / "HEAD.json").exists():
        raise ValueError("v2 partial ledger exists; resume is not authorized")
    if root.exists() and any(root.iterdir()):
        raise ValueError("v2 candidate root is not empty; a fresh root is required")


def _assert_exact_artifact_files(root: Path, expected: Iterable[Path | str]) -> None:
    expected_names = sorted(str(Path(value)) for value in expected)
    observed_names: list[str] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("candidate artifact tree contains a symbolic link")
        if path.is_file():
            observed_names.append(str(path.relative_to(root)))
        elif not path.is_dir():
            raise ValueError("candidate artifact tree contains a special file")
    if sorted(observed_names) != expected_names:
        raise ValueError("candidate artifact tree differs from the exact run stage")


class _TerminalSemanticFailure(RuntimeError):
    pass


def _execute_started(
    args: argparse.Namespace, *, contract: dict[str, Any],
    preregistration: Mapping[str, Any], registry: Mapping[str, Any],
    source_pack: Mapping[str, Any], resident: Mapping[str, Any],
    reserve_bytes: int, started: float,
    git_binding: Mapping[str, Any], failure_context: dict[str, Any],
) -> int:
    ready_path = args.resident_root / "READY.json"
    cases_by_id = {
        str(case["episode_id"]): dict(case) for case in registry["cases_data"]
    }
    roles_by_id = {
        row["query_episode_id"]: row for row in derive_role_table(registry)
    }
    execution = execution_query_ids(registry)
    context = multiprocessing.get_context("spawn")
    for execution_ordinal, query_id in enumerate(execution):
        case = cases_by_id[query_id]
        role = roles_by_id[query_id]
        failure_context.update({
            "case_id": case["case_id"], "query_episode_id": query_id,
            "reason": "case_execution_failure",
        })
        implementation = _require_implementation_manifest(
            contract["implementation_manifest"],
        )
        expected_query = _expected_query_context(args.config, case)
        expected_ready = _ready_observation(ready_path)
        if expected_ready != resident["resident_ready_observation"]:
            raise ValueError("parent READY instance changed before case spawn")
        _append_ledger_event(
            args.output_root, contract["contract_digest"], "case_started", {
                "execution_ordinal": execution_ordinal,
                "registry_case_id": case["case_id"],
                "query_episode_id": query_id,
                "performance_role": role["performance_role"],
                "implementation_manifest_digest": implementation["digest"],
                "resident_ready_digest": expected_ready["ready_digest"],
                "resident_content_digest": expected_ready["content_digest"],
            },
        )
        with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
            result = executor.submit(
                _worker, str(args.config), resident["mirror_store_root"],
                str(ready_path), case, role, dict(FROZEN_ROUTE_QUOTAS),
                source_pack["provenance_digest"], source_pack["physical_rows"],
                resident["binding_digest"], resident["resident_content_digest"],
                contract["contract_digest"],
                dict(contract["implementation_manifest"]), dict(expected_ready),
                dict(resident),
            ).result()
        semantic_row = dict(result["semantic"])
        performance_row = dict(result["performance"])
        _strict_validate_case_evidence(
            semantic_row, performance_row, contract=contract, case=case,
            role=role, expected_query=expected_query, resident=resident,
            physical_rows=int(source_pack["physical_rows"]),
        )
        bundle = _case_bundle(
            contract["contract_digest"], execution_ordinal,
            semantic_row, performance_row,
        )
        bundle_path = _case_bundle_path(
            args.output_root, execution_ordinal, query_id,
        )
        _atomic_json_create(bundle_path, bundle)
        persisted = json.loads(bundle_path.read_text())
        persisted_semantic, persisted_performance = _validate_case_bundle(
            persisted, contract=contract, case=case, role=role,
            expected_query=expected_query, resident=resident,
            physical_rows=int(source_pack["physical_rows"]),
            execution_ordinal=execution_ordinal,
        )
        _append_ledger_event(
            args.output_root, contract["contract_digest"], "case_completed", {
                "execution_ordinal": execution_ordinal,
                "registry_case_id": case["case_id"],
                "query_episode_id": query_id,
                "performance_role": role["performance_role"],
                "bundle_digest": persisted["bundle_digest"],
                "semantic_digest": persisted_semantic["semantic_digest"],
                "performance_digest": persisted_performance["performance_digest"],
                "semantic_passed": persisted_semantic["passed"],
                "performance_passed": persisted_performance["passed"],
            },
        )
        if persisted_semantic["passed"] is not True:
            failure_context["reason"] = "semantic_gate_failure"
            raise _TerminalSemanticFailure("semantic gate failed; recovery is forbidden")

    failure_context.update({
        "case_id": None, "query_episode_id": None,
        "reason": "finalization_failure",
    })
    final_ready, final_validation_observation = prepare_resident_mirror_observed(
        args.source_full_root / "store", args.resident_root,
        FROZEN_GENERATION_ID,
        expected_provenance_digest=source_pack["provenance_digest"],
        reserve_bytes=reserve_bytes, validate_existing=True,
    )
    resident_end = _resident_binding(
        final_ready, contract_digest=contract["contract_digest"],
        ready_path=ready_path,
        validation_observation=final_validation_observation,
    )
    resident_failures = _validate_resident_binding(
        resident_end, contract, final_ready,
    )
    if resident_failures:
        raise ValueError(f"final resident READY binding differs:{resident_failures}")
    _require_implementation_manifest(contract["implementation_manifest"])
    _require_environment_manifest(contract["environment_manifest"])
    copied_contract = json.loads(
        (args.output_root / "candidate-contract.json").read_text()
    )
    if copied_contract != contract:
        raise ValueError("copied producer contract changed before sealing")
    contract_failures = validate_producer_contract(
        copied_contract, registry, load_config(args.config).artifact_dir,
        expected_source_pack=source_pack,
        expected_implementation_manifest=contract["implementation_manifest"],
        expected_environment_manifest=contract["environment_manifest"],
    )
    if contract_failures:
        raise ValueError(f"copied producer contract differs:{contract_failures}")
    prereg_path = _preregistration_path()
    if json.loads(prereg_path.read_text()) != preregistration:
        raise ValueError("committed preregistration changed before sealing")
    if _git_preregistration_binding(
        Path(__file__).resolve().parents[2], prereg_path,
        contract["implementation_manifest"]["files"],
    ) != git_binding:
        raise ValueError("repository HEAD or tracked state changed before sealing")

    semantic: dict[str, dict[str, Any]] = {}
    performance: dict[str, dict[str, Any]] = {}
    bundles: dict[str, dict[str, Any]] = {}
    for execution_ordinal, query_id in enumerate(execution):
        case = cases_by_id[query_id]
        role = roles_by_id[query_id]
        bundle_path = _case_bundle_path(
            args.output_root, execution_ordinal, query_id,
        )
        bundle = json.loads(bundle_path.read_text())
        semantic_row, performance_row = _validate_case_bundle(
            bundle, contract=contract, case=case, role=role,
            expected_query=_expected_query_context(args.config, case),
            resident=resident, physical_rows=int(source_pack["physical_rows"]),
            execution_ordinal=execution_ordinal,
        )
        semantic[query_id] = semantic_row
        performance[query_id] = performance_row
        bundles[query_id] = bundle
    observed_bundle_paths = sorted((args.output_root / "case-bundles").glob("*.json"))
    expected_bundle_paths = [
        _case_bundle_path(args.output_root, ordinal, query_id)
        for ordinal, query_id in enumerate(execution)
    ]
    if observed_bundle_paths != sorted(expected_bundle_paths):
        raise ValueError("on-disk case bundle set differs before sealing")

    preseal_files: list[Path | str] = [
        "candidate-contract.json", "RESIDENT_READY.json", "ledger/HEAD.json",
        *(path.relative_to(args.output_root) for path in expected_bundle_paths),
        *(f"ledger/events/{index:06d}.json" for index in range(121)),
    ]
    _assert_exact_artifact_files(args.output_root, preseal_files)

    events, _ = _load_ledger(args.output_root, contract["contract_digest"])
    _validate_execution_ledger(
        events, contract=contract, preregistration=preregistration,
        git_binding=git_binding, resident=resident, execution=execution,
        cases_by_id=cases_by_id, roles_by_id=roles_by_id,
        bundles_by_id=bundles,
    )
    semantic_matrix = _semantic_matrix(
        contract, resident, resident_end, registry, semantic, events,
        perf_counter() - started,
    )
    if semantic_matrix["passed"] is not True:
        failure_context["reason"] = "semantic_matrix_failure"
        raise _TerminalSemanticFailure("semantic matrix failed; recovery is forbidden")
    _require_implementation_manifest(contract["implementation_manifest"])
    _atomic_json_create(args.output_root / "semantic-matrix.json", semantic_matrix)
    semantic_seal = _terminal_document(
        SEMANTIC_SEAL_SCHEMA, "seal_digest", {
            "producer_contract_digest": contract["contract_digest"],
            "semantic_matrix_digest": semantic_matrix["result_digest"],
            "resident_start_content_digest": resident["resident_content_digest"],
            "resident_end_content_digest": resident_end["resident_content_digest"],
            "semantic_cases": 60,
            "semantic_recall_ready": True,
            "performance_independent": True,
            "authority_results_opened": False,
            "production_promotion_authorized": False,
        },
    )
    _atomic_json_create(args.output_root / "SEMANTIC_SEALED.json", semantic_seal)
    performance_matrix = _performance_matrix(
        contract, registry, performance, perf_counter() - started,
    )
    _require_implementation_manifest(contract["implementation_manifest"])
    _atomic_json_create(args.output_root / "performance-matrix.json", performance_matrix)
    performance_final = _terminal_document(
        PERFORMANCE_FINAL_SCHEMA, "final_digest", {
            "producer_contract_digest": contract["contract_digest"],
            "performance_matrix_digest": performance_matrix["result_digest"],
            "resident_binding_digest": resident["binding_digest"],
            "performance_terminal": True,
            "performance_passed": performance_matrix["passed"],
            "confirmatory_performance_cases": 53,
            "exposed_regression_cases": 7,
            "claims_policy": CLAIMS_POLICY,
            "authority_results_opened": False,
            "production_promotion_authorized": False,
        },
    )
    _atomic_json_create(args.output_root / "PERFORMANCE_FINAL.json", performance_final)
    _require_implementation_manifest(contract["implementation_manifest"])
    final_event = _append_ledger_event(
        args.output_root, contract["contract_digest"], "run_complete", {
            "semantic_matrix_digest": semantic_matrix["result_digest"],
            "semantic_seal_digest": semantic_seal["seal_digest"],
            "performance_matrix_digest": performance_matrix["result_digest"],
            "performance_final_digest": performance_final["final_digest"],
            "performance_passed": performance_matrix["passed"],
        },
    )
    _, head = _load_ledger(args.output_root, contract["contract_digest"])
    run_complete = _terminal_document(
        RUN_COMPLETE_SCHEMA, "complete_digest", {
            "producer_contract_digest": contract["contract_digest"],
            "resident_binding_digest": resident["binding_digest"],
            "semantic_seal_digest": semantic_seal["seal_digest"],
            "performance_final_digest": performance_final["final_digest"],
            "ledger_last_event_digest": final_event["event_digest"],
            "ledger_head_digest": head["head_digest"],
            "semantic_passed": True,
            "performance_passed": performance_matrix["passed"],
            "authority_results_opened": False,
            "production_promotion_authorized": False,
        },
    )
    _atomic_json_create(args.output_root / "RUN_COMPLETE.json", run_complete)
    _assert_exact_artifact_files(args.output_root, [
        *preseal_files,
        "semantic-matrix.json", "SEMANTIC_SEALED.json",
        "performance-matrix.json", "PERFORMANCE_FINAL.json",
        "RUN_COMPLETE.json", "ledger/events/000121.json",
    ])
    persisted_semantic_matrix = json.loads(
        (args.output_root / "semantic-matrix.json").read_text()
    )
    persisted_semantic_seal = json.loads(
        (args.output_root / "SEMANTIC_SEALED.json").read_text()
    )
    persisted_performance_matrix = json.loads(
        (args.output_root / "performance-matrix.json").read_text()
    )
    persisted_performance_final = json.loads(
        (args.output_root / "PERFORMANCE_FINAL.json").read_text()
    )
    persisted_complete = json.loads(
        (args.output_root / "RUN_COMPLETE.json").read_text()
    )
    final_events, final_head = _load_ledger(
        args.output_root, contract["contract_digest"],
    )
    if not all((
        persisted_semantic_matrix == semantic_matrix,
        persisted_semantic_matrix.get("result_digest")
        == semantic_matrix_digest(persisted_semantic_matrix),
        persisted_semantic_seal == semantic_seal,
        persisted_semantic_seal.get("seal_digest")
        == terminal_digest(persisted_semantic_seal, "seal_digest"),
        persisted_performance_matrix == performance_matrix,
        persisted_performance_matrix.get("result_digest")
        == performance_matrix_digest(persisted_performance_matrix),
        persisted_performance_final == performance_final,
        persisted_performance_final.get("final_digest")
        == terminal_digest(persisted_performance_final, "final_digest"),
        persisted_complete == run_complete,
        persisted_complete.get("complete_digest")
        == terminal_digest(persisted_complete, "complete_digest"),
        len(final_events) == 122,
        final_events[-1]["event_digest"] == run_complete["ledger_last_event_digest"],
        final_head is not None,
        final_head["head_digest"] == run_complete["ledger_head_digest"],
    )):
        raise ValueError("persisted terminal evidence differs after publication")
    _require_implementation_manifest(contract["implementation_manifest"])
    print(json.dumps({
        "run_complete": True,
        "semantic_passed": True,
        "performance_passed": performance_matrix["passed"],
        "semantic_seal_digest": semantic_seal["seal_digest"],
        "performance_final_digest": performance_final["final_digest"],
        "complete_digest": run_complete["complete_digest"],
    }, indent=2, sort_keys=True))
    return 0 if performance_matrix["passed"] else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--source-full-root", type=Path, required=True)
    parser.add_argument("--resident-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    config = load_config(args.config)
    roots = expected_roots(config.artifact_dir)
    observed_roots = {
        "registry_root": str(args.registry.parent.resolve()),
        "source_full_root": str(args.source_full_root.resolve()),
        "resident_full_root": str(args.resident_root.resolve()),
        "candidate_root": str(args.output_root.resolve()),
        "comparison_root": roots["comparison_root"],
        "verification_root": roots["verification_root"],
        "predecessor_candidate_root": roots["predecessor_candidate_root"],
        "predecessor_comparison_root": roots["predecessor_comparison_root"],
        "authority_root": roots["authority_root"],
    }
    if observed_roots != roots:
        raise ValueError("v2 exact roots differ")
    _assert_fresh_output(args.output_root)
    registry = json.loads(args.registry.read_text())
    failures = [
        *validate_m04r_validation_registry(
            source_from_spec(config.datasets["nasdaq"]), args.registry.parent,
        ),
        *validate_role_table(registry),
        *validate_predecessor_paths(config.artifact_dir),
    ]
    if failures:
        raise ValueError(f"v2 frozen prerequisites differ:{failures}")
    if packed_bound_search_contract()["digest"] != FROZEN_PROPOSAL_CONTRACT_DIGEST:
        raise ValueError("runtime proposal implementation contract differs")
    source_pack = _source_pack_binding(args.source_full_root)
    implementation = _implementation_manifest()
    environment = _environment_manifest()
    repository = Path(__file__).resolve().parents[2]
    preregistration = v2_contract.load_and_validate_preregistration(
        registry, config.artifact_dir, repository,
        expected_source_pack=source_pack,
        expected_implementation_manifest=implementation,
        expected_environment_manifest=environment,
    )
    preregistration_path = v2_contract.expected_preregistration_path(repository)
    contract = dict(preregistration["producer_contract"])
    git_binding = _git_preregistration_binding(
        repository, preregistration_path,
        contract["implementation_manifest"]["files"],
    )
    _require_implementation_manifest(contract["implementation_manifest"])
    _require_environment_manifest(contract["environment_manifest"])
    ready_path = args.resident_root / "READY.json"
    reserve_bytes = int(contract["resident_policy"]["reserve_bytes"])
    ready, readiness_validation_observation = prepare_resident_mirror_observed(
        args.source_full_root / "store", args.resident_root,
        FROZEN_GENERATION_ID,
        expected_provenance_digest=source_pack["provenance_digest"],
        reserve_bytes=reserve_bytes, validate_existing=True,
    )
    resident = _resident_binding(
        ready, contract_digest=contract["contract_digest"], ready_path=ready_path,
        validation_observation=readiness_validation_observation,
    )
    resident_failures = _validate_resident_binding(resident, contract, ready)
    if resident_failures:
        raise ValueError(f"resident READY binding differs:{resident_failures}")
    _require_implementation_manifest(contract["implementation_manifest"])
    failure_context: dict[str, Any] = {
        "case_id": None, "query_episode_id": None,
        "reason": "launch_publication_failure",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    owns_run = False
    try:
        _atomic_json_create(args.output_root / "candidate-contract.json", contract)
        owns_run = True
        _atomic_json_create(args.output_root / "RESIDENT_READY.json", resident)
        _append_ledger_event(
            args.output_root, contract["contract_digest"], "run_started", {
                "preregistration_digest": preregistration["preregistration_digest"],
                "git_binding": git_binding,
                "resident_binding_digest": resident["binding_digest"],
                "resident_content_digest": resident["resident_content_digest"],
                "execution_query_ids_digest": contract["execution_query_ids_digest"],
            },
        )
        return _execute_started(
            args, contract=contract, preregistration=preregistration,
            registry=registry, source_pack=source_pack, resident=resident,
            reserve_bytes=reserve_bytes, started=started,
            git_binding=git_binding, failure_context=failure_context,
        )
    except BaseException as exc:
        if not owns_run:
            raise
        marked = _best_effort_mark_incomplete(
            args.output_root, contract, resident,
            case_id=failure_context["case_id"],
            query_episode_id=failure_context["query_episode_id"],
            reason=failure_context["reason"], error_type=type(exc).__name__,
        )
        if isinstance(exc, _TerminalSemanticFailure) and marked:
            return 2
        raise


if __name__ == "__main__":
    raise SystemExit(main())
