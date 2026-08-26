"""Truth-blind exposed-case exact-worker and scheduler qualification.

This development-only POC calibrates exact-search worker counts on the four
already-open M04R-13 hard cases.  It never accepts an authority or outcome
path.  Proposal scans are performed once in each direction per preparation
and the forward proposal is reused by every exact attempt.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import stat
import subprocess
import sys
import platform
from threading import Lock
import tempfile
import time
from time import perf_counter
from typing import Any, Callable, Mapping, Protocol, Sequence

from market_analogues.certified_packed_search import certified_packed_search
from market_analogues.packed_bound_search import scan_packed_bound_proposals_threaded
from market_analogues.types import stable_hash


SCHEMA = "m04r14-exact-scheduler-poc-v1"
RUN_SCHEMA = "m04r14-exact-scheduler-run-started-v1"
PROPOSAL_SCHEMA = "m04r14-exact-scheduler-proposal-v1"
ATTEMPT_SCHEMA = "m04r14-exact-scheduler-attempt-v1"
SEMANTICS_SCHEMA = "m04r14-exact-scheduler-semantics-v1"
MEASUREMENTS_SCHEMA = "m04r14-exact-scheduler-measurements-v1"
COMPLETE_SCHEMA = "m04r14-exact-scheduler-complete-v1"
INCOMPLETE_SCHEMA = "m04r14-exact-scheduler-incomplete-v1"
PREREG_SELECTION_RULE = (
    "per-case max/min<=1.25; score=sum four 3-rep medians; "
    "smallest workers with score<=1.03*fastest"
)
FROZEN_QUERY_IDS = (
    "3307023dbe2164d025e788da", "3618af07dedd52fb3bdb1ccd",
    "9d7365581643bd93e85beb67", "99a0838725a09570b4a075ff",
)
FROZEN_CASE_IDS = (
    "nasdaq-JCTC-historical-252", "nasdaq-GBNY-current-252",
    "nasdaq-GBNY-historical-252", "nasdaq-ISPOW-current-252",
)
THROUGHPUT_EXTRA_QUERY_IDS = (
    "2d5fd014631f9d4ce97ac34d", "5d4e4365b63ff184cbf29f9a",
    "f2af4d63144103d695102676", "f9d5f80116223e7573d268fb",
)
THROUGHPUT_EXTRA_CASE_IDS = (
    "nasdaq-OABI-historical-252", "nasdaq-BTCY-current-252",
    "nasdaq-CUE-historical-252", "nasdaq-FRMM-historical-252",
)
THROUGHPUT_QUERY_IDS = FROZEN_QUERY_IDS + THROUGHPUT_EXTRA_QUERY_IDS
THROUGHPUT_CASE_IDS = FROZEN_CASE_IDS + THROUGHPUT_EXTRA_CASE_IDS
THROUGHPUT_SELECTION_TRANSCRIPT = (
    (7, "2d5fd014631f9d4ce97ac34d", "OABI", 1163, 3633057,
     "59ff3a5f86360413ae90faf92d41a031c36523a5d18f352c47b8a1dad3d83a96"),
    (21, "5d4e4365b63ff184cbf29f9a", "BTCY", 1170, 3786118,
     "613b0a1882b1da55a3757d0451f7096c1bb1d06fa7f4b6828cd901c8267b5c0b"),
    (35, "f2af4d63144103d695102676", "CUE", 1179, 3091909,
     "315c074efa90a39961e3b13048a5a7c2f4ff4d82a27cfe7d533c6d5311983e90"),
    (49, "f9d5f80116223e7573d268fb", "FRMM", 1253, 2676848,
     "409a18f6e03049e0dce5fa3de9b4a5dd21c6e16fbbe34208c1e2a7c19d44fa92"),
)
THROUGHPUT_SELECTION_DIGEST = "6542e99973abbaf81db2ff1e1e5bf107b45802027519642c902169e617c329be"
M11_ALL60_DIGEST = "0eb6aa3d9f3277ca4d1fab829b08981b982626c856683a5616eac92fc57c49a8"
M11_SEMANTIC_SEAL_DIGEST = "976781b3021700c54e64091b669507201f09aa4799b13e1f414af2e6979066a2"
M11_SEMANTIC_MATRIX_DIGEST = "9bc68539092ec0912f5bdcd0c0f063c35370e15757408914e44e0c1178f21af3"
M11_SEMANTIC_SEALED_SHA256 = "b9c63061a14c1c04f5ffb8bb7bd8d12b19cd3ec7aa945099cc4f87593ea9c492"
M11_RUN_COMPLETE_SHA256 = "de098543ae3b76e019742dc0d574468fb0bfc75d0afd4d5576b1bfc870c6ebd6"
M11_HISTORICAL_GIT_HEAD = "882436abeadb331537aa1b9a19dd72a4e91166b2"
M11_HISTORICAL_PREREG_SHA256 = "53646c847323b8a29d9e950e84d7cf267d514d4792a561fd786c0938fb82e940"
M11_PREREG_RELATIVE = Path("experiments/m04r/m04r11_candidate_v2_preregistered.json")
CATALOG_DIGEST = "5aedc6d5eb0890df3040c559cf70cf35123398506577f63e852030db132bd658"
ORACLE_RESULT_DIGEST = "ac2a8c84cf629060156ebc37f0c037a0a6731b1fffca9d349eeebdfd0eca0496"
WORKERS = (1, 2, 4, 8)
REPETITIONS = 3
PROPOSAL_THREADS = 8
THROUGHPUT_TASKS = 8
NUMERIC_ATOL = 1e-6
ENGINE_TOLERANCE = 1e-12
RUN_HARD_LIMIT_SECONDS = 6 * 60 * 60
CHILD_STARTUP_TIMEOUT_SECONDS = 600
CHILD_TASK_TIMEOUT_SECONDS = 3600
OUTPUT_RELATIVE = Path("config/data/analogues/m04r14/exact-scheduler-poc-v1")
PREREG_RELATIVE = Path("experiments/m04r/m04r14_exact_scheduler_poc_preregistered.json")
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    "NUMBA_NUM_THREADS", "NUMBA_THREADING_LAYER",
)


class SchedulerError(RuntimeError):
    pass


def _without(value: Mapping[str, Any], omitted: set[str]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in omitted}


def _strict_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise SchedulerError("evidence is not strict finite JSON") from exc


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    if path.is_symlink():
        raise SchedulerError("JSON input symlink is forbidden")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor); raw = b""
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block: break
            raw += block
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
        value.st_ctime_ns, value.st_mode,
    )
    if identity(before) != identity(after) or not stat.S_ISREG(before.st_mode):
        raise SchedulerError("JSON input identity changed")
    if expected_sha256 is not None and sha256(raw).hexdigest() != expected_sha256:
        raise SchedulerError("JSON input SHA differs")
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value: raise SchedulerError("duplicate JSON key")
            value[key] = item
        return value
    try:
        value = json.loads(
            raw, object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SchedulerError(f"nonfinite JSON token {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SchedulerError("invalid strict JSON") from exc
    if type(value) is not dict:
        raise SchedulerError("JSON top level differs")
    def finite(item: Any) -> bool:
        if type(item) is float:
            return math.isfinite(item)
        if type(item) is list:
            return all(finite(child) for child in item)
        if type(item) is dict:
            return all(finite(child) for child in item.values())
        return True
    if not finite(value):
        raise SchedulerError("nonfinite JSON number")
    return value


def _seal(state: Mapping[str, Any]) -> dict[str, Any]:
    deterministic = dict(state)
    return {"state": deterministic, "digest": stable_hash(deterministic)}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise SchedulerError(f"create-only evidence exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _strict_bytes(payload) + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary_path, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_incomplete(root: Path, exc: BaseException) -> None:
    existing = [
        {"path": str(path.relative_to(root)), "sha256": _sha(path)}
        for path in sorted(root.rglob("*.json"))
        if path.name != "INCOMPLETE.json"
    ]
    state = {
        "schema_version": INCOMPLETE_SCHEMA, "status": "incomplete",
        "error_type": type(exc).__name__, "error_message": str(exc),
        "existing_files": existing, "existing_files_digest": stable_hash(existing),
    }
    _atomic(root / "INCOMPLETE.json", {**_seal(state), "created_at": _timestamp()})


def _expected_terminal_paths(*, include_complete: bool) -> set[str]:
    paths = {"CONTRACT.json", "RUN_STARTED.json", "SEMANTICS.json",
             "MEASUREMENTS.json"}
    if include_complete: paths.add("COMPLETE.json")
    for repetition in range(REPETITIONS):
        for ordinal in range(4):
            base = f"primary/r{repetition}/c{ordinal}"
            paths.add(f"{base}/PROPOSAL.json")
            paths.update(f"{base}/EXACT-w{worker}.json" for worker in WORKERS)
    paths.add("throughput/PROPOSALS_COMPLETE.json")
    for index in range(THROUGHPUT_TASKS):
        base = f"throughput/t{index}"
        paths.update({f"{base}/PROPOSAL.json", f"{base}/EXACT-w1.json",
                      f"{base}/CONTROL-w1.json"})
    return paths


def _tree_files(root: Path) -> set[str]:
    files = set()
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise SchedulerError("output tree contains symlink/special entry")
        if stat.S_ISREG(mode): files.add(str(path.relative_to(root)))
    return files


def _exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise SchedulerError(f"{label} exact schema differs")
    return value


def _sealed(value: Any, label: str) -> dict[str, Any]:
    row = _exact_keys(value, {"state", "digest"}, label)
    if type(row["state"]) is not dict or row["digest"] != stable_hash(row["state"]):
        raise SchedulerError(f"{label} seal differs")
    return row


def _timestamp_valid(value: Any) -> bool:
    if type(value) is not str:
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed)


def _proposal_leaf(
    path: Path, *, payload: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _exact_keys(
        _read_json(path) if payload is None else dict(payload),
        {"state", "digest", "measurement", "created_at"},
        "proposal leaf",
    )
    if not _timestamp_valid(payload["created_at"]):
        raise SchedulerError("proposal timestamp differs")
    sealed = _sealed(_without(payload, {"measurement", "created_at"}), "proposal")
    state = sealed["state"]
    _exact_keys(state, {
        "schema_version", "task_id", "case_id", "query_id", "query_binding",
        "resident_lease_digests", "source_binding_before", "source_binding_after",
        "resident_snapshot", "forward", "reverse", "semantic_digest",
    }, "proposal state")
    if state["schema_version"] != PROPOSAL_SCHEMA \
            or state["source_binding_before"] != state["source_binding_after"] \
            or type(state["resident_lease_digests"]) is not list \
            or len(state["resident_lease_digests"]) != 4 \
            or len(set(state["resident_lease_digests"])) != 1 \
            or not all(_is_digest(value) for value in state["resident_lease_digests"]):
        raise SchedulerError("proposal binding differs")
    omitted = {"block_rows", "block_order", "elapsed_seconds", "peak_rss_mb"}
    if _without(state["forward"], omitted) != _without(state["reverse"], omitted) \
            or state["forward"].get("block_rows") != 4096 \
            or state["forward"].get("block_order") != "forward" \
            or state["reverse"].get("block_rows") != 4097 \
            or state["reverse"].get("block_order") != "reverse" \
            or state["semantic_digest"] != stable_hash(
                _without(state["forward"], omitted)
            ):
        raise SchedulerError("proposal forward/reverse reconstruction differs")
    measurement = payload["measurement"]
    expected_measurement_keys = {
        "wall_seconds", "forward_seconds", "reverse_seconds", "resources",
    }
    if type(measurement) is dict and "spawned_process" in measurement:
        expected_measurement_keys.add("spawned_process")
    _exact_keys(measurement, expected_measurement_keys, "proposal measurement")
    if any(
        type(measurement[key]) is not float or not math.isfinite(measurement[key])
        or measurement[key] < 0
        for key in ("wall_seconds", "forward_seconds", "reverse_seconds")
    ) or measurement["wall_seconds"] < (
        measurement["forward_seconds"] + measurement["reverse_seconds"]
    ):
        raise SchedulerError("proposal timing nesting differs")
    _validate_resource_evidence(measurement["resources"])
    if "spawned_process" in measurement:
        _validate_proposal_process(measurement["spawned_process"])
    return state, measurement


def _attempt_leaf(
    path: Path, *, control: bool = False,
    payload: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _exact_keys(
        _read_json(path) if payload is None else dict(payload),
        {"semantic", "measurement", "created_at"},
        "attempt leaf",
    )
    if not _timestamp_valid(payload["created_at"]):
        raise SchedulerError("attempt timestamp differs")
    seal = _sealed(payload["semantic"], "attempt semantic")
    return seal["state"], payload["measurement"]


def validate_terminal(
    root: Path, *, complete_payload: Mapping[str, Any] | None = None,
    repository: Path | None = None, production_backend: Any | None = None,
) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise SchedulerError("terminal root differs")
    if _tree_files(root) != _expected_terminal_paths(
        include_complete=complete_payload is None,
    ):
        raise SchedulerError("terminal exact tree differs")
    complete = _exact_keys(
        _read_json(root / "COMPLETE.json") if complete_payload is None
        else dict(complete_payload), {"state", "digest", "created_at"},
        "complete",
    )
    if not _timestamp_valid(complete["created_at"]):
        raise SchedulerError("complete timestamp differs")
    state = _sealed(_without(complete, {"created_at"}), "complete")["state"]
    _exact_keys(state, {
        "schema_version", "status", "semantic_digest", "measurement_digest",
        "selected_workers", "final_lease", "semantic_passed",
        "performance_selection_only", "production_promotion_authorized",
        "leaf_manifest", "leaf_manifest_digest",
    }, "complete state")
    if state["schema_version"] != COMPLETE_SCHEMA or state["status"] != "complete" \
            or state["semantic_passed"] is not True \
            or state["performance_selection_only"] is not True \
            or state["production_promotion_authorized"] is not False \
            or type(state["selected_workers"]) is not int \
            or state["selected_workers"] not in WORKERS:
        raise SchedulerError("complete terminal state differs")
    manifest = state.get("leaf_manifest")
    if type(manifest) is not list or state.get("leaf_manifest_digest") != stable_hash(manifest):
        raise SchedulerError("leaf manifest seal differs")
    expected_leaf_paths = _expected_terminal_paths(include_complete=False)
    if [row.get("path") for row in manifest] != sorted(expected_leaf_paths) \
            or any(set(row) != {"path", "sha256"} for row in manifest):
        raise SchedulerError("leaf manifest paths differ")
    leaf_shas = {row["path"]: row["sha256"] for row in manifest}
    leaf_payloads = {
        row["path"]: _read_json(root / row["path"], expected_sha256=row["sha256"])
        for row in manifest
    }
    contract = _sealed(leaf_payloads["CONTRACT.json"], "contract")
    contract_state = _exact_keys(contract["state"], {
        "schema_version", "status", "runtime_binding", "foundation",
        "output_root", "query_ids", "throughput_query_ids", "workers",
        "repetitions", "execution_policy", "selection_rule",
    }, "contract state")
    if contract_state["schema_version"] != "m04r14-exact-scheduler-preregistration-v1" \
            or contract_state["status"] != "frozen_before_run" \
            or contract_state["output_root"] != str(root.resolve()) \
            or contract_state["query_ids"] != list(FROZEN_QUERY_IDS) \
            or contract_state["throughput_query_ids"] != list(THROUGHPUT_QUERY_IDS) \
            or contract_state["workers"] != list(WORKERS) \
            or contract_state["repetitions"] != REPETITIONS \
            or contract_state["execution_policy"] != _execution_policy() \
            or contract_state["selection_rule"] != PREREG_SELECTION_RULE:
        raise SchedulerError("contract reconstruction differs")
    foundation = contract_state["foundation"]
    production_terminal = foundation.get("mode") not in {
        "test", "dependency-injected-test",
    }
    if production_terminal:
        if not isinstance(production_backend, ProductionBackend):
            raise SchedulerError("production terminal requires live backend binding")
        if production_backend.foundation() != foundation:
            raise SchedulerError("production foundation changed")
        _validate_runtime(
            contract_state["runtime_binding"],
            (Path(__file__).resolve().parents[2] if repository is None else repository),
        )
        observed_thread_environment = contract_state["runtime_binding"]["state"][
            "environment"
        ]["state"]["thread_environment"]
        production_child_environment = {
            key: (
                observed_thread_environment[key]
                if key == "NUMBA_THREADING_LAYER" else "1"
            )
            for key in THREAD_ENV_KEYS
        }
        production_contexts = {}
        for ordinal, case in enumerate(production_backend.cases):
            source, episode, request, packed = production_backend.module._case_context(
                production_backend.inputs, case,
            )
            production_contexts[case.query_id] = {
                "ordinal": ordinal, "packed": packed,
                "query_binding": production_backend.module.query_binding(
                    source, episode, request, packed,
                    production_backend.inputs.provenance_digest,
                ),
            }
    elif contract_state["runtime_binding"] != test_foundation():
        raise SchedulerError("test runtime binding differs")
    else:
        production_child_environment = None
    run = _exact_keys(
        leaf_payloads["RUN_STARTED.json"], {"state", "digest", "created_at"},
        "run started",
    )
    if not _timestamp_valid(run["created_at"]):
        raise SchedulerError("run-started timestamp differs")
    run_state = _sealed(_without(run, {"created_at"}), "run started")["state"]
    _exact_keys(run_state, {
        "schema_version", "status", "development_only", "cases_previously_exposed",
        "direct_raw_authority_accessed_by_this_run",
        "authority_derived_prerequisite_evidence_accessed_by_this_run",
        "forward_outcomes_accessed_by_this_run",
        "raw_authority_or_outcome_paths_accepted", "production_promotion_authorized",
        "query_ids", "case_ids", "workers", "repetitions", "proposal_threads",
        "throughput", "numeric_policy", "foundation", "runtime_binding",
        "preregistration_digest",
    }, "run-started state")
    if run_state["schema_version"] != RUN_SCHEMA or run_state["status"] != "started" \
            or run_state["development_only"] is not True \
            or run_state["cases_previously_exposed"] is not True \
            or run_state["direct_raw_authority_accessed_by_this_run"] is not False \
            or run_state[
                "authority_derived_prerequisite_evidence_accessed_by_this_run"
            ] is not True \
            or run_state["forward_outcomes_accessed_by_this_run"] is not False \
            or run_state["raw_authority_or_outcome_paths_accepted"] is not False \
            or run_state["production_promotion_authorized"] is not False \
            or run_state["query_ids"] != list(FROZEN_QUERY_IDS) \
            or run_state["case_ids"] != list(FROZEN_CASE_IDS) \
            or run_state["workers"] != list(WORKERS) \
            or run_state["repetitions"] != REPETITIONS \
            or run_state["proposal_threads"] != PROPOSAL_THREADS \
            or run_state["throughput"] != {
                "tasks": THROUGHPUT_TASKS, "distinct_queries": True,
                "maximum_concurrent_exact_tasks": 8, "workers_per_task": 1,
                "proposal_phase": "serial-before-barrier",
            } \
            or run_state["numeric_policy"] != {
                "semantic_digest_equality": "exact",
                "scorer_absolute_tolerance_hex": NUMERIC_ATOL.hex(),
                "engine_algorithm_tolerance_hex": ENGINE_TOLERANCE.hex(),
            } \
            or run_state["foundation"] != contract_state["foundation"] \
            or run_state["runtime_binding"] != contract_state["runtime_binding"] \
            or run_state["preregistration_digest"] != contract["digest"]:
        raise SchedulerError("run-started reconstruction differs")
    semantics = _sealed(leaf_payloads["SEMANTICS.json"], "semantics")
    measurements = _sealed(leaf_payloads["MEASUREMENTS.json"], "measurements")
    if semantics.get("digest") != state.get("semantic_digest") \
            or semantics.get("digest") != stable_hash(semantics.get("state")) \
            or measurements.get("digest") != state.get("measurement_digest") \
            or measurements.get("digest") != stable_hash(measurements.get("state")):
        raise SchedulerError("aggregate terminal links differ")
    semantic_state = _exact_keys(semantics["state"], {
        "schema_version", "status", "primary", "throughput",
        "case_reference_digests", "semantic_passed", "cases_previously_exposed",
        "direct_raw_authority_accessed_by_this_run",
        "authority_derived_prerequisite_evidence_accessed_by_this_run",
        "forward_outcomes_accessed_by_this_run",
    }, "semantics state")
    measurement_state = _exact_keys(measurements["state"], {
        "schema_version", "status", "primary", "throughput", "throughput_lane",
        "selection", "host_cpu_qualification", "primary_interactive_selected", "event_ledger",
        "event_ledger_digest",
    }, "measurements state")
    if semantic_state["schema_version"] != SEMANTICS_SCHEMA \
            or semantic_state["status"] != "semantic_pass" \
            or semantic_state["semantic_passed"] is not True \
            or semantic_state["cases_previously_exposed"] is not True \
            or semantic_state[
                "direct_raw_authority_accessed_by_this_run"
            ] is not False \
            or semantic_state[
                "authority_derived_prerequisite_evidence_accessed_by_this_run"
            ] is not True \
            or semantic_state["forward_outcomes_accessed_by_this_run"] is not False \
            or measurement_state["schema_version"] != MEASUREMENTS_SCHEMA \
            or measurement_state["status"] != "measured":
        raise SchedulerError("aggregate literal differs")
    if len(semantic_state["primary"]) != 48 \
            or len(semantic_state["throughput"]) != 8 \
            or len(measurement_state["primary"]) != 48 \
            or len(measurement_state["throughput"]) != 8:
        raise SchedulerError("aggregate matrix cardinality differs")
    primary_semantics: list[dict[str, Any]] = []
    primary_measurements: list[dict[str, Any]] = []
    references: dict[str, dict[str, Any]] = {}
    for repetition in range(REPETITIONS):
        for ordinal in _rotated(tuple(range(4)), repetition):
            proposal_relative = f"primary/r{repetition}/c{ordinal}/PROPOSAL.json"
            proposal, proposal_measurement = _proposal_leaf(
                root / proposal_relative, payload=leaf_payloads[proposal_relative],
            )
            if production_terminal and "spawned_process" not in proposal_measurement:
                raise SchedulerError("production proposal process evidence is absent")
            if production_terminal:
                _validate_production_proposal(
                    proposal, production_backend, production_contexts,
                )
                _validate_proposal_process_binding(
                    proposal_measurement["spawned_process"], proposal,
                    production_child_environment,
                )
            expected_task = f"primary-r{repetition}-c{ordinal}"
            if proposal["task_id"] != expected_task \
                    or proposal["case_id"] != FROZEN_CASE_IDS[ordinal] \
                    or proposal["query_id"] != FROZEN_QUERY_IDS[ordinal]:
                raise SchedulerError("primary proposal identity differs")
            for worker in _rotated(WORKERS, repetition + ordinal):
                attempt_relative = (
                    f"primary/r{repetition}/c{ordinal}/EXACT-w{worker}.json"
                )
                row, measurement_row = _attempt_leaf(
                    root / attempt_relative, payload=leaf_payloads[attempt_relative],
                )
                if production_terminal and (
                    "spawned_process" not in measurement_row.get("measurement", {})
                    or "source_binding_before" not in row.get("attempt", {})
                    or "source_binding_after" not in row.get("attempt", {})
                ):
                    raise SchedulerError("production exact process/source evidence is absent")
                _exact_keys(row, {
                    "lane", "repetition", "case_id", "query_id",
                    "workers", "task_id", "attempt",
                }, "primary semantic row")
                expected_identity = {
                    "lane": "primary", "repetition": repetition,
                    "case_id": FROZEN_CASE_IDS[ordinal],
                    "query_id": FROZEN_QUERY_IDS[ordinal], "workers": worker,
                    "task_id": expected_task,
                }
                if _without(row, {"attempt"}) != expected_identity:
                    raise SchedulerError("primary semantic identity differs")
                attempt = ExactAttempt(row["attempt"], measurement_row["measurement"])
                prepared = PreparedCase(
                    expected_task, proposal["case_id"], proposal["query_id"],
                    proposal, proposal_measurement,
                )
                _validate_attempt(prepared, attempt, worker)
                if production_terminal:
                    _validate_production_attempt(
                        proposal, attempt, production_backend,
                    )
                core = _attempt_semantic_core(attempt.semantic)
                if proposal["case_id"] in references \
                        and references[proposal["case_id"]] != core:
                    raise SchedulerError("primary exact semantic parity differs")
                references.setdefault(proposal["case_id"], core)
                primary_semantics.append(row)
                expected_measurement = {
                    "lane": "primary", "repetition": repetition,
                    "case_id": proposal["case_id"],
                    "workers": worker, "task_id": expected_task,
                    "proposal_measurement": proposal_measurement,
                    "measurement": attempt.measurement,
                }
                if measurement_row != expected_measurement:
                    raise SchedulerError("primary measurement leaf differs")
                primary_measurements.append(measurement_row)
    throughput_semantics: list[dict[str, Any]] = []
    throughput_measurements: list[dict[str, Any]] = []
    for index in range(THROUGHPUT_TASKS):
        proposal_relative = f"throughput/t{index}/PROPOSAL.json"
        proposal, proposal_measurement = _proposal_leaf(
            root / proposal_relative, payload=leaf_payloads[proposal_relative],
        )
        if production_terminal and "spawned_process" not in proposal_measurement:
            raise SchedulerError("production proposal process evidence is absent")
        if production_terminal:
            _validate_production_proposal(
                proposal, production_backend, production_contexts,
            )
            _validate_proposal_process_binding(
                proposal_measurement["spawned_process"], proposal,
                production_child_environment,
            )
        task_id = f"throughput-t{index}-c{index}"
        if proposal["task_id"] != task_id \
                or proposal["case_id"] != THROUGHPUT_CASE_IDS[index] \
                or proposal["query_id"] != THROUGHPUT_QUERY_IDS[index]:
            raise SchedulerError("throughput proposal identity differs")
        control_relative = f"throughput/t{index}/CONTROL-w1.json"
        exact_relative = f"throughput/t{index}/EXACT-w1.json"
        control_state, control_measurement = _attempt_leaf(
            root / control_relative, control=True,
            payload=leaf_payloads[control_relative],
        )
        attempt_state, attempt_measurement_row = _attempt_leaf(
            root / exact_relative, payload=leaf_payloads[exact_relative],
        )
        if production_terminal and (
            "spawned_process" not in control_measurement
            or "source_binding_before" not in control_state
            or "source_binding_after" not in control_state
            or "spawned_process" not in attempt_measurement_row.get("measurement", {})
            or "source_binding_before" not in attempt_state.get("attempt", {})
            or "source_binding_after" not in attempt_state.get("attempt", {})
        ):
            raise SchedulerError("production exact process/source evidence is absent")
        _exact_keys(attempt_state, {
            "lane", "task_index", "case_id", "query_id", "workers", "task_id",
            "attempt",
        }, "throughput semantic row")
        if _attempt_semantic_core(control_state) != _attempt_semantic_core(
            attempt_state["attempt"]
        ):
            raise SchedulerError("throughput control semantic parity differs")
        prepared = PreparedCase(
            task_id, proposal["case_id"], proposal["query_id"], proposal, {},
        )
        _validate_attempt(prepared, ExactAttempt(control_state, control_measurement), 1)
        _validate_attempt(
            prepared,
            ExactAttempt(attempt_state["attempt"], attempt_measurement_row["measurement"]),
            1,
        )
        if production_terminal:
            _validate_production_attempt(
                proposal, ExactAttempt(control_state, control_measurement),
                production_backend,
            )
            _validate_production_attempt(
                proposal,
                ExactAttempt(
                    attempt_state["attempt"], attempt_measurement_row["measurement"],
                ),
                production_backend,
            )
        if attempt_state["case_id"] in references \
                and _attempt_semantic_core(attempt_state["attempt"]) != references[
                    attempt_state["case_id"]
                ]:
            raise SchedulerError("throughput/primary semantic parity differs")
        expected_measurement = {
            "lane": "throughput-p8t1", "task_index": index,
            "case_id": proposal["case_id"], "workers": 1, "task_id": task_id,
            "measurement": attempt_measurement_row["measurement"],
        }
        if attempt_state != {
            "lane": "throughput-p8t1", "task_index": index,
            "case_id": proposal["case_id"], "query_id": proposal["query_id"],
            "workers": 1, "task_id": task_id, "attempt": attempt_state["attempt"],
        } or attempt_measurement_row != expected_measurement:
            raise SchedulerError("throughput leaf reconstruction differs")
        throughput_semantics.append(attempt_state)
        throughput_measurements.append(attempt_measurement_row)
    barrier = _sealed(
        leaf_payloads["throughput/PROPOSALS_COMPLETE.json"],
        "proposal barrier",
    )
    barrier_state = _exact_keys(barrier["state"], {
        "status", "query_ids", "proposal_semantic_digests",
    }, "proposal barrier state")
    if barrier_state != {
        "status": "all_eight_proposals_complete_before_timed_wave",
        "query_ids": list(THROUGHPUT_QUERY_IDS),
        "proposal_semantic_digests": [
            _proposal_leaf(
                root / f"throughput/t{index}/PROPOSAL.json",
                payload=leaf_payloads[f"throughput/t{index}/PROPOSAL.json"],
            )[0][
                "semantic_digest"
            ]
            for index in range(THROUGHPUT_TASKS)
        ],
    }:
        raise SchedulerError("proposal barrier reconstruction differs")
    if semantic_state["primary"] != primary_semantics \
            or semantic_state["throughput"] != throughput_semantics \
            or measurement_state["primary"] != primary_measurements \
            or measurement_state["throughput"] != throughput_measurements \
            or semantic_state["case_reference_digests"] != {
                key: stable_hash(value) for key, value in sorted(references.items())
            }:
        raise SchedulerError("aggregate leaf reconstruction differs")
    expected_selection = _stable_selection(primary_measurements)
    if measurement_state["selection"] != expected_selection \
            or state["selected_workers"] != expected_selection["state"]["selected_workers"]:
        raise SchedulerError("worker selection reconstruction differs")
    selected = expected_selection["state"]["selected_workers"]
    expected_interactive = [{
        "case_id": row["case_id"], "repetition": row["repetition"],
        "forward_proposal_seconds": row["proposal_measurement"]["forward_seconds"],
        "selected_exact_seconds": row["measurement"]["wall_seconds"],
        "interactive_seconds": (
            row["proposal_measurement"]["forward_seconds"]
            + row["measurement"]["wall_seconds"]
        ),
        "reverse_proposal_role": "semantic-parity-only-excluded",
    } for row in primary_measurements if row["workers"] == selected]
    if measurement_state["primary_interactive_selected"] != expected_interactive:
        raise SchedulerError("interactive latency reconstruction differs")
    cpu_qualification = measurement_state["host_cpu_qualification"]
    if type(cpu_qualification) is not dict or set(cpu_qualification) != {
        "schema_version", "required_cpus", "before", "after", "delta",
        "performance_valid",
    } or cpu_qualification != _host_cpu_qualification(
        cpu_qualification.get("before", {}), cpu_qualification.get("after", {}),
        required_cpus=8,
    ):
        raise SchedulerError("host CPU qualification differs")
    lane = _exact_keys(measurement_state["throughput_lane"], {
        "label", "wall_seconds", "resources", "tasks", "distinct_query_ids",
        "maximum_concurrent_exact_tasks", "workers_per_task",
        "observed_maximum_active_exact_tasks",
        "proposals_prepared_serially_with_threads",
        "serial_controls_excluded_from_lane_timing", "spawned_process_evidence",
    }, "throughput lane")
    if lane["label"] != "exact-stage exposed-host p8t1 microbenchmark" \
            or type(lane["wall_seconds"]) is not float \
            or not math.isfinite(lane["wall_seconds"]) or lane["wall_seconds"] < 0 \
            or lane["tasks"] != 8 \
            or lane["distinct_query_ids"] != list(THROUGHPUT_QUERY_IDS) \
            or lane["maximum_concurrent_exact_tasks"] != 8 \
            or lane["workers_per_task"] != 1 \
            or type(lane["observed_maximum_active_exact_tasks"]) is not int \
            or not 1 < lane["observed_maximum_active_exact_tasks"] <= 8 \
            or lane["proposals_prepared_serially_with_threads"] != PROPOSAL_THREADS \
            or lane["serial_controls_excluded_from_lane_timing"] is not True:
        raise SchedulerError("throughput lane reconstruction differs")
    _validate_resource_evidence(lane["resources"])
    batch_process = lane["spawned_process_evidence"]
    if production_terminal and batch_process is None:
        raise SchedulerError("production batch process evidence is absent")
    if batch_process is not None:
        batch_process = _exact_keys(batch_process, {
            "wall_seconds", "children", "host_vmstat_swap_before",
            "host_vmstat_swap_after", "host_swap_is_context_only",
            "process_swap_gate_passed", "ready_evidence", "release_evidence",
            "observed_concurrent_children", "release_to_all_children_exit_seconds",
        }, "throughput spawned process evidence")
        if type(batch_process["children"]) is not list \
                or len(batch_process["children"]) != 8 \
                or type(batch_process["ready_evidence"]) is not list \
                or len(batch_process["ready_evidence"]) != 8 \
                or batch_process["host_swap_is_context_only"] is not True \
                or batch_process["process_swap_gate_passed"] is not True \
                or batch_process["observed_concurrent_children"] != 8 \
                or not _nonnegative_number(batch_process["wall_seconds"]) \
                or not _nonnegative_number(
                    batch_process["release_to_all_children_exit_seconds"]
                ):
            raise SchedulerError("throughput spawned process values differ")
        host_before = _validate_vmstat(
            batch_process["host_vmstat_swap_before"],
            "throughput host vmstat before",
        )
        host_after = _validate_vmstat(
            batch_process["host_vmstat_swap_after"],
            "throughput host vmstat after",
        )
        if any(host_after[key] < host_before[key] for key in host_before) \
                or batch_process["release_to_all_children_exit_seconds"] > batch_process[
                    "wall_seconds"
                ]:
            raise SchedulerError("throughput process timing/vmstat differs")
        for child in batch_process["children"]:
            _validate_child_metric(child)
        ready_pids = []
        ready_cpus = []
        ready_monotonic = []
        for index, ready in enumerate(batch_process["ready_evidence"]):
            _exact_keys(ready, {
                "schema_version", "task_id", "pid", "workers", "cpu_affinity",
                "thread_environment", "ready_monotonic", "proposal_sha256",
                "resident_lease_digest",
            }, "throughput READY evidence")
            proposal_path = root / f"throughput/t{index}/PROPOSAL.json"
            exact_state, _ = _attempt_leaf(
                root / f"throughput/t{index}/EXACT-w1.json",
                payload=leaf_payloads[f"throughput/t{index}/EXACT-w1.json"],
            )
            expected_lease = exact_state["attempt"]["lease_before"]
            if ready["schema_version"] != "m04r14-exact-child-ready-v1" \
                    or ready["task_id"] != f"throughput-t{index}-c{index}" \
                    or ready["workers"] != 1 or len(ready["cpu_affinity"]) != 1 \
                    or ready["cpu_affinity"] != batch_process["children"][index]["cpus"] \
                    or ready["proposal_sha256"] != leaf_shas[
                        str(proposal_path.relative_to(root))
                    ] \
                    or ready["resident_lease_digest"] != expected_lease \
                    or not _is_digest(ready["resident_lease_digest"]) \
                    or ready["thread_environment"] != production_child_environment \
                    or not _nonnegative_number(ready["ready_monotonic"]):
                raise SchedulerError("throughput READY values differ")
            ready_pids.append(ready["pid"])
            ready_cpus.extend(ready["cpu_affinity"])
            ready_monotonic.append(ready["ready_monotonic"])
        release = _exact_keys(batch_process["release_evidence"], {
            "schema_version", "released_monotonic", "pids", "ready_digests",
        }, "throughput release evidence")
        if release["schema_version"] != "m04r14-exact-child-release-v1" \
                or release["pids"] != ready_pids \
                or not _nonnegative_number(release["released_monotonic"]) \
                or release["released_monotonic"] < max(ready_monotonic) \
                or release["ready_digests"] != [
                    stable_hash(value) for value in batch_process["ready_evidence"]
                ] or [child["pid"] for child in batch_process["children"]] != ready_pids \
                or len(set(ready_pids)) != 8 or len(set(ready_cpus)) != 8:
            raise SchedulerError("throughput release crosslink differs")
    final_lease = state["final_lease"]
    if foundation.get("mode") in {"test", "dependency-injected-test"}:
        _exact_keys(
            final_lease, {"resident_identity_digest", "source_lease_digest"},
            "test final lease",
        )
        if final_lease["resident_identity_digest"] != foundation.get(
            "resident_identity_digest"
        ) or final_lease["source_lease_digest"] != foundation.get(
            "source_lease_digest"
        ):
            raise SchedulerError("test final lease crosslink differs")
    else:
        _exact_keys(final_lease, {
            "resident_identity_digest", "source_binding_digests",
            "causal_input_shas",
        }, "production final lease")
        expected_sources: dict[str, str] = {}
        for ordinal in range(8):
            proposal_relative = (
                f"primary/r0/c{ordinal}/PROPOSAL.json" if ordinal < 4 else
                f"throughput/t{ordinal}/PROPOSAL.json"
            )
            proposal, _ = _proposal_leaf(
                root / proposal_relative, payload=leaf_payloads[proposal_relative],
            )
            expected_sources[str(ordinal)] = stable_hash(
                proposal["source_binding_before"]
            )
        if final_lease["resident_identity_digest"] != foundation["resident"][
            "identity_digest"
        ] or final_lease["source_binding_digests"] != expected_sources \
                or final_lease["causal_input_shas"] != foundation["causal_input_shas"]:
            raise SchedulerError("production final lease crosslink differs")
    ledger = measurement_state["event_ledger"]
    if type(ledger) is not list or measurement_state["event_ledger_digest"] != stable_hash(ledger):
        raise SchedulerError("event ledger seal differs")
    previous = None
    for sequence, event in enumerate(ledger):
        if type(event) is not dict or event.get("sequence") != sequence \
                or event.get("previous_event_digest") != previous:
            raise SchedulerError("event ledger chain differs")
        deterministic = _without(event, {"event_digest"})
        if event.get("event_digest") != stable_hash(deterministic):
            raise SchedulerError("event ledger event digest differs")
        previous = event["event_digest"]
    event_rows = [
        _without(row, {"previous_event_digest", "event_digest"}) for row in ledger
    ]
    expected_prefix: list[dict[str, Any]] = []
    for index in range(8):
        task_id = f"throughput-t{index}-c{index}"
        proposal_relative = f"throughput/t{index}/PROPOSAL.json"
        proposal, _ = _proposal_leaf(
            root / proposal_relative, payload=leaf_payloads[proposal_relative],
        )
        expected_prefix.extend((
            {"sequence": len(expected_prefix), "event": "proposal_started",
             "task_id": task_id, "active_proposals": 1},
            {"sequence": len(expected_prefix) + 1, "event": "proposal_completed",
             "task_id": task_id, "active_proposals": 0,
             "proposal_semantic_digest": proposal["semantic_digest"]},
        ))
    expected_prefix.append({
        "sequence": len(expected_prefix), "event": "barrier_released",
        "barrier_digest": barrier["digest"],
    })
    if event_rows[:len(expected_prefix)] != expected_prefix:
        raise SchedulerError("proposal event sequence differs")
    cursor = len(expected_prefix)
    batch_events = []
    while cursor < len(event_rows) and event_rows[cursor].get("event") in {
        "batch_exact_started", "batch_exact_ended",
    }:
        batch_events.append(event_rows[cursor]); cursor += 1
    task_ids = {f"throughput-t{index}-c{index}" for index in range(8)}
    if len(batch_events) != 16 \
            or {row["task_id"] for row in batch_events if row["event"] == "batch_exact_started"} != task_ids \
            or {row["task_id"] for row in batch_events if row["event"] == "batch_exact_ended"} != task_ids \
            or sum(row["event"] == "batch_exact_started" for row in batch_events) != 8 \
            or sum(row["event"] == "batch_exact_ended" for row in batch_events) != 8 \
            or any(set(row) != {"sequence", "event", "task_id", "active_exact_tasks"}
                   for row in batch_events):
        raise SchedulerError("batch event sequence differs")
    if batch_process is not None:
        expected_batch = [
            {
                "sequence": len(expected_prefix) + index,
                "event": "batch_exact_started",
                "task_id": f"throughput-t{index}-c{index}",
                "active_exact_tasks": 8,
            }
            for index in range(8)
        ] + [
            {
                "sequence": len(expected_prefix) + 8 + index,
                "event": "batch_exact_ended",
                "task_id": f"throughput-t{index}-c{index}",
                "active_exact_tasks": 0,
            }
            for index in range(8)
        ]
        if batch_events != expected_batch:
            raise SchedulerError("spawned batch event reconstruction differs")
        reconstructed_maximum_active = 8
    else:
        active_tasks: set[str] = set()
        started_tasks: set[str] = set()
        reconstructed_maximum_active = 0
        for event in batch_events:
            task_id = event["task_id"]
            if event["event"] == "batch_exact_started":
                if task_id in started_tasks or task_id in active_tasks:
                    raise SchedulerError("batch task started twice")
                started_tasks.add(task_id); active_tasks.add(task_id)
            else:
                if task_id not in active_tasks:
                    raise SchedulerError("batch task ended while inactive")
                active_tasks.remove(task_id)
            if event["active_exact_tasks"] != len(active_tasks):
                raise SchedulerError("batch active counter differs")
            reconstructed_maximum_active = max(
                reconstructed_maximum_active, len(active_tasks),
            )
        if active_tasks or started_tasks != task_ids:
            raise SchedulerError("batch active-set terminal differs")
    if lane["observed_maximum_active_exact_tasks"] != reconstructed_maximum_active:
        raise SchedulerError("throughput observed concurrency differs")
    for index in range(8):
        task_id = f"throughput-t{index}-c{index}"
        control_relative = f"throughput/t{index}/CONTROL-w1.json"
        control, _ = _attempt_leaf(
            root / control_relative, payload=leaf_payloads[control_relative],
        )
        expected = {
            "sequence": cursor, "event": "control_completed_after_wave",
            "task_id": task_id,
            "semantic_digest": stable_hash(_attempt_semantic_core(control)),
        }
        if cursor >= len(event_rows) or event_rows[cursor] != expected:
            raise SchedulerError("control event sequence differs")
        cursor += 1
    for index in range(8):
        task_id = f"throughput-t{index}-c{index}"
        exact_relative = f"throughput/t{index}/EXACT-w1.json"
        attempt, _ = _attempt_leaf(
            root / exact_relative, payload=leaf_payloads[exact_relative],
        )
        expected = {
            "sequence": cursor, "event": "batch_exact_completed",
            "task_id": task_id,
            "semantic_digest": stable_hash(_attempt_semantic_core(attempt["attempt"])),
        }
        if cursor >= len(event_rows) or event_rows[cursor] != expected:
            raise SchedulerError("batch-complete event sequence differs")
        cursor += 1
    if cursor != len(event_rows):
        raise SchedulerError("event ledger terminal length differs")
    if _tree_files(root) != _expected_terminal_paths(
        include_complete=complete_payload is None,
    ) or any(
        _read_json(root / row["path"], expected_sha256=row["sha256"])
        != leaf_payloads[row["path"]]
        for row in manifest
    ):
        raise SchedulerError("terminal tree changed during validation")
    if complete_payload is None and _read_json(root / "COMPLETE.json") != complete:
        raise SchedulerError("complete terminal changed during validation")
    return complete


def _resource_snapshot() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss_divisor = 1024 ** 2 if sys.platform == "darwin" else 1024
    swap_kib = 0
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmSwap:"):
                swap_kib = int(line.split()[1]); break
    except (OSError, ValueError, IndexError):
        swap_kib = 0
    affinity = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    )
    return {
        "user_cpu_seconds": float(usage.ru_utime),
        "system_cpu_seconds": float(usage.ru_stime),
        "minor_faults": int(usage.ru_minflt),
        "major_faults": int(usage.ru_majflt),
        "voluntary_context_switches": int(usage.ru_nvcsw),
        "involuntary_context_switches": int(usage.ru_nivcsw),
        "max_rss_mb": float(usage.ru_maxrss) / rss_divisor,
        "swap_kib": swap_kib,
        "cpu_affinity": affinity,
        "thread_environment": {key: os.environ.get(key) for key in THREAD_ENV_KEYS},
    }


def _resource_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    if before["cpu_affinity"] != after["cpu_affinity"] \
            or before["thread_environment"] != after["thread_environment"]:
        raise SchedulerError("affinity or thread environment changed during task")
    names = (
        "user_cpu_seconds", "system_cpu_seconds", "minor_faults", "major_faults",
        "voluntary_context_switches", "involuntary_context_switches", "swap_kib",
    )
    delta = {name: after[name] - before[name] for name in names}
    if any(value < 0 for value in delta.values()):
        raise SchedulerError("resource counter regressed")
    return {
        "before": dict(before), "after": dict(after), "delta": delta,
        "peak_rss_mb": max(before["max_rss_mb"], after["max_rss_mb"]),
        "swap_delta_kib": delta["swap_kib"],
    }


def _validate_resource_evidence(value: Any) -> None:
    row = _exact_keys(value, {
        "before", "after", "delta", "peak_rss_mb", "swap_delta_kib",
    }, "resource evidence")
    snapshot_keys = {
        "user_cpu_seconds", "system_cpu_seconds", "minor_faults", "major_faults",
        "voluntary_context_switches", "involuntary_context_switches", "max_rss_mb",
        "swap_kib", "cpu_affinity", "thread_environment",
    }
    for label in ("before", "after"):
        snapshot = _exact_keys(row[label], snapshot_keys, f"resource {label}")
        if any(
            type(snapshot[key]) is not float or not math.isfinite(snapshot[key])
            or snapshot[key] < 0
            for key in ("user_cpu_seconds", "system_cpu_seconds", "max_rss_mb")
        ) or any(
            type(snapshot[key]) is not int or snapshot[key] < 0
            for key in (
                "minor_faults", "major_faults", "voluntary_context_switches",
                "involuntary_context_switches", "swap_kib",
            )
        ) or type(snapshot["cpu_affinity"]) is not list \
                or len(set(snapshot["cpu_affinity"])) != len(snapshot["cpu_affinity"]) \
                or not all(type(cpu) is int and cpu >= 0 for cpu in snapshot["cpu_affinity"]) \
                or type(snapshot["thread_environment"]) is not dict \
                or set(snapshot["thread_environment"]) != set(THREAD_ENV_KEYS) \
                or any(value is not None and type(value) is not str
                       for value in snapshot["thread_environment"].values()):
            raise SchedulerError("resource snapshot types differ")
    reconstructed = _resource_delta(row["before"], row["after"])
    if row != reconstructed:
        raise SchedulerError("resource evidence reconstruction differs")


def _nonnegative_number(value: Any) -> bool:
    return type(value) in {int, float} and not isinstance(value, bool) \
        and math.isfinite(value) and value >= 0


def _is_digest(value: Any) -> bool:
    if type(value) is not str or len(value) != 64 or value.lower() != value:
        return False
    try:
        return len(bytes.fromhex(value)) == 32
    except ValueError:
        return False


def _validate_vmstat(value: Any, label: str) -> dict[str, int]:
    row = _exact_keys(value, {"pswpin", "pswpout"}, label)
    if any(type(row[key]) is not int or row[key] < 0 for key in row):
        raise SchedulerError(f"{label} values differ")
    return row


def _validate_child_metric(value: Any) -> None:
    row = _exact_keys(value, {
        "pid", "cpus", "wall_seconds", "user_cpu_seconds", "system_cpu_seconds",
        "minor_faults", "major_faults", "peak_rss_kib", "peak_hwm_kib",
        "wait4_max_rss_kib", "effective_peak_rss_kib", "peak_swap_kib",
        "final_swap_kib",
    }, "exact child process metric")
    if type(row["pid"]) is not int or row["pid"] <= 0 \
            or type(row["cpus"]) is not list or not row["cpus"] \
            or len(set(row["cpus"])) != len(row["cpus"]) \
            or not all(type(cpu) is int and cpu >= 0 for cpu in row["cpus"]) \
            or not all(_nonnegative_number(row[key]) for key in (
                "wall_seconds", "user_cpu_seconds", "system_cpu_seconds",
                "minor_faults", "major_faults", "peak_rss_kib", "peak_hwm_kib",
                "wait4_max_rss_kib", "effective_peak_rss_kib", "peak_swap_kib",
                "final_swap_kib",
            )) \
            or row["effective_peak_rss_kib"] != max(
                row["peak_rss_kib"], row["peak_hwm_kib"], row["wait4_max_rss_kib"],
            ) or row["peak_swap_kib"] != 0 or row["final_swap_kib"] != 0:
        raise SchedulerError("exact child process metric differs")


def _validate_proposal_process(value: Any) -> None:
    row = _exact_keys(value, {
        "pid", "wall_seconds", "cpus", "ready_evidence", "release_evidence",
        "startup_to_ready_seconds", "user_cpu_seconds", "system_cpu_seconds",
        "minor_faults", "major_faults", "peak_rss_kib", "peak_hwm_kib",
        "wait4_max_rss_kib", "effective_peak_rss_kib", "peak_swap_kib",
        "final_swap_kib", "host_vmstat_swap_before", "host_vmstat_swap_after",
        "host_swap_is_context_only",
    }, "proposal child process")
    if type(row["pid"]) is not int or row["pid"] <= 0 \
            or type(row["cpus"]) is not list or len(row["cpus"]) != PROPOSAL_THREADS \
            or len(set(row["cpus"])) != PROPOSAL_THREADS \
            or not all(_nonnegative_number(row[key]) for key in (
                "wall_seconds", "startup_to_ready_seconds", "user_cpu_seconds",
                "system_cpu_seconds", "minor_faults", "major_faults",
                "peak_rss_kib", "peak_hwm_kib", "wait4_max_rss_kib",
                "effective_peak_rss_kib", "peak_swap_kib", "final_swap_kib",
            )) \
            or row["effective_peak_rss_kib"] != max(
                row["peak_rss_kib"], row["peak_hwm_kib"], row["wait4_max_rss_kib"],
            ) or row["peak_swap_kib"] != 0 or row["final_swap_kib"] != 0 \
            or row["host_swap_is_context_only"] is not True \
            or row["startup_to_ready_seconds"] > row["wall_seconds"]:
        raise SchedulerError("proposal child process values differ")
    ready = _exact_keys(row["ready_evidence"], {
        "schema_version", "task_id", "pid", "cpu_affinity", "thread_environment",
        "resident_lease_digest", "ready_monotonic",
    }, "proposal READY evidence")
    release = _exact_keys(row["release_evidence"], {
        "schema_version", "task_id", "pid", "ready_digest", "released_monotonic",
    }, "proposal release evidence")
    if ready["schema_version"] != "m04r14-proposal-child-ready-v1" \
            or release["schema_version"] != "m04r14-proposal-child-release-v1" \
            or ready["pid"] != row["pid"] or release["pid"] != row["pid"] \
            or ready["cpu_affinity"] != row["cpus"] \
            or type(ready["task_id"]) is not str or not ready["task_id"] \
            or type(ready["thread_environment"]) is not dict \
            or set(ready["thread_environment"]) != set(THREAD_ENV_KEYS) \
            or any(value is not None and type(value) is not str
                   for value in ready["thread_environment"].values()) \
            or not _is_digest(ready["resident_lease_digest"]) \
            or not _nonnegative_number(ready["ready_monotonic"]) \
            or not _nonnegative_number(release["released_monotonic"]) \
            or release["released_monotonic"] < ready["ready_monotonic"] \
            or release["task_id"] != ready["task_id"] \
            or release["ready_digest"] != stable_hash(ready):
        raise SchedulerError("proposal process barrier crosslink differs")
    before = _validate_vmstat(
        row["host_vmstat_swap_before"], "proposal host vmstat before",
    )
    after = _validate_vmstat(
        row["host_vmstat_swap_after"], "proposal host vmstat after",
    )
    if any(after[key] < before[key] for key in before):
        raise SchedulerError("proposal host vmstat regressed")


def _validate_proposal_process_binding(
    value: Mapping[str, Any], proposal: Mapping[str, Any],
    expected_thread_environment: Mapping[str, Any] | None,
) -> None:
    ready = value["ready_evidence"]
    if ready["task_id"] != proposal["task_id"] \
            or ready["resident_lease_digest"] != proposal[
                "resident_lease_digests"
            ][0] \
            or expected_thread_environment is None \
            or ready["thread_environment"] != dict(expected_thread_environment):
        raise SchedulerError("proposal process/proposal binding differs")


def _validate_production_proposal(
    proposal: Mapping[str, Any], backend: Any,
    contexts: Mapping[str, Mapping[str, Any]],
) -> None:
    context = contexts.get(proposal["query_id"])
    resident = backend.resident
    expected_lease = resident["lease"]["lease_digest"]
    try:
        backend.module.validate_resident_snapshot(proposal["resident_snapshot"])
        backend.module.validate_proposal(
            proposal["forward"], context["packed"], backend.inputs,
        )
        backend.module.validate_proposal(
            proposal["reverse"], context["packed"], backend.inputs,
        )
    except BaseException as exc:
        raise SchedulerError("production proposal structure differs") from exc
    if context is None \
            or proposal["query_binding"] != context["query_binding"] \
            or proposal["source_binding_before"] != context["query_binding"] \
            or proposal["source_binding_after"] != context["query_binding"] \
            or proposal["resident_snapshot"] != resident \
            or proposal["resident_lease_digests"] != [expected_lease] * 4:
        raise SchedulerError("production proposal causal binding differs")


def _validate_production_attempt(
    proposal: Mapping[str, Any], attempt: ExactAttempt, backend: Any,
) -> None:
    resident = backend.resident
    expected_lease = resident["lease"]["lease_digest"]
    certificate = {
        **attempt.semantic["certificate"],
        "elapsed_seconds": attempt.measurement["engine_seconds"],
    }
    try:
        backend.module.validate_certificate_and_matches(
            certificate, attempt.semantic["matches"], proposal["query_id"],
            expected_input_digest=proposal["query_binding"][
                "certified_input_digest"
            ],
        )
    except BaseException as exc:
        raise SchedulerError("production exact certificate/matches differ") from exc
    if attempt.semantic["lease_before"] != expected_lease \
            or attempt.semantic["lease_after"] != expected_lease \
            or attempt.semantic["source_binding_before"] != proposal["query_binding"] \
            or attempt.semantic["source_binding_after"] != proposal["query_binding"]:
        raise SchedulerError("production exact causal binding differs")


def _vmstat_swap() -> dict[str, int]:
    values = {"pswpin": 0, "pswpout": 0}
    try:
        for line in Path("/proc/vmstat").read_text().splitlines():
            name, raw = line.split()
            if name in values:
                values[name] = int(raw)
    except (OSError, ValueError):
        pass
    return values


def _parse_cpu_list(raw: str) -> list[int]:
    cpus: set[int] = set()
    for part in raw.strip().split(","):
        if not part:
            continue
        bounds = part.split("-", 1)
        try:
            start = int(bounds[0]); end = int(bounds[-1])
        except ValueError as exc:
            raise SchedulerError("cgroup CPU list is malformed") from exc
        if start < 0 or end < start:
            raise SchedulerError("cgroup CPU list is malformed")
        cpus.update(range(start, end + 1))
    return sorted(cpus)


def _cgroup_cpu_snapshot() -> dict[str, Any]:
    try:
        unified = next(
            line.split("::", 1)[1] for line in Path("/proc/self/cgroup").read_text().splitlines()
            if line.startswith("0::")
        )
    except (OSError, StopIteration, IndexError) as exc:
        raise SchedulerError("cgroup v2 CPU binding is unavailable") from exc
    cgroup_root = Path("/sys/fs/cgroup")
    current = (cgroup_root / unified.lstrip("/")).resolve(strict=True)

    def inherited(name: str) -> tuple[Path, str]:
        candidate = current
        while candidate == cgroup_root or cgroup_root in candidate.parents:
            path = candidate / name
            try:
                raw = path.read_text().strip()
            except OSError:
                raw = ""
            if raw:
                return path, raw
            if candidate == cgroup_root:
                break
            candidate = candidate.parent
        raise SchedulerError(f"cgroup {name} binding is unavailable")

    cpuset_path, cpuset_raw = inherited("cpuset.cpus.effective")
    cpu_max_path, cpu_max_raw = inherited("cpu.max")
    maximum = cpu_max_raw.split()
    if len(maximum) != 2:
        raise SchedulerError("cgroup cpu.max is malformed")
    quota = None if maximum[0] == "max" else int(maximum[0])
    period = int(maximum[1])
    if (quota is not None and quota <= 0) or period <= 0:
        raise SchedulerError("cgroup cpu.max is invalid")
    stat_path, stat_raw = inherited("cpu.stat")
    try:
        cpu_stat = {
            key: int(value) for key, value in (
                line.split() for line in stat_raw.splitlines()
            )
        }
    except (ValueError, TypeError) as exc:
        raise SchedulerError("cgroup cpu.stat is malformed") from exc
    if not {"usage_usec", "nr_periods", "nr_throttled", "throttled_usec"} \
            <= set(cpu_stat) or any(value < 0 for value in cpu_stat.values()):
        raise SchedulerError("cgroup cpu.stat fields differ")
    return {
        "configuration": {
            "schema_version": "m04r14-cgroup-cpu-configuration-v1",
            "cgroup_path": str(current),
            "cpuset_path": str(cpuset_path),
            "effective_cpus": _parse_cpu_list(cpuset_raw),
            "cpu_max_path": str(cpu_max_path),
            "quota_usec": quota, "period_usec": period,
            "effective_quota_cpus": (
                None if quota is None else float(quota) / float(period)
            ),
            "cpu_stat_path": str(stat_path),
        },
        "cpu_stat": cpu_stat,
    }


def _host_cpu_qualification(
    before: Mapping[str, Any], after: Mapping[str, Any], *, required_cpus: int = 8,
) -> dict[str, Any]:
    if set(before) != {"configuration", "cpu_stat"} \
            or set(after) != {"configuration", "cpu_stat"} \
            or before["configuration"] != after["configuration"]:
        raise SchedulerError("cgroup CPU configuration changed")
    configuration = before["configuration"]
    cpus = configuration.get("effective_cpus")
    quota_cpus = configuration.get("effective_quota_cpus")
    if type(cpus) is not list or len(cpus) < required_cpus \
            or not all(type(cpu) is int and cpu >= 0 for cpu in cpus) \
            or (quota_cpus is not None and (
                type(quota_cpus) is not float or not math.isfinite(quota_cpus)
                or quota_cpus < required_cpus
            )):
        raise SchedulerError("cgroup CPU capacity is below required lane width")
    if set(before["cpu_stat"]) != set(after["cpu_stat"]):
        raise SchedulerError("cgroup cpu.stat schema changed")
    delta = {
        key: after["cpu_stat"][key] - before["cpu_stat"][key]
        for key in before["cpu_stat"]
    }
    if any(type(value) is not int or value < 0 for value in delta.values()) \
            or delta["nr_throttled"] != 0 or delta["throttled_usec"] != 0:
        raise SchedulerError("cgroup CPU throttling invalidates performance evidence")
    return {
        "schema_version": "m04r14-host-cpu-qualification-v1",
        "required_cpus": required_cpus, "before": dict(before),
        "after": dict(after), "delta": delta, "performance_valid": True,
    }


def _proc_memory(pid: int) -> dict[str, int]:
    values = {"VmRSS": 0, "VmHWM": 0, "VmSwap": 0}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            name = line.split(":", 1)[0]
            if name in values:
                values[name] = int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return values


@dataclass(frozen=True)
class PreparedCase:
    task_id: str
    case_id: str
    query_id: str
    semantic: dict[str, Any]
    measurement: dict[str, Any]
    opaque: Any = None


@dataclass(frozen=True)
class ExactAttempt:
    semantic: dict[str, Any]
    measurement: dict[str, Any]


class ExecutionBackend(Protocol):
    case_ids: tuple[str, ...]
    query_ids: tuple[str, ...]
    throughput_case_ids: tuple[str, ...]
    throughput_query_ids: tuple[str, ...]

    def foundation(self) -> dict[str, Any]: ...
    def prepare(self, case_ordinal: int, task_id: str) -> PreparedCase: ...
    def exact(self, prepared: PreparedCase, workers: int) -> ExactAttempt: ...
    def final_lease(self) -> dict[str, Any]: ...


def _m13(repository: Path) -> Any:
    path = repository / "experiments/m04r/m04r13_threaded_certified_exposed.py"
    spec = importlib.util.spec_from_file_location("m04r14_scheduler_m13", path)
    if spec is None or spec.loader is None:
        raise SchedulerError("M04R-13 implementation is absent")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_script(repository: Path, relative: str, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, repository / relative)
    if spec is None or spec.loader is None:
        raise SchedulerError(f"required script is absent: {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; spec.loader.exec_module(module)
    return module


def _git_blob(repository: Path, commit: str, relative: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative}"], cwd=repository,
        capture_output=True, check=False,
    )
    if result.returncode:
        raise SchedulerError(f"historical Git blob is absent: {relative}")
    return result.stdout


def _authenticate_m11_historical(repository: Path) -> dict[str, Any]:
    """Authenticate M11 evidence against its producing commit, not current code."""
    contract_path = (
        repository / "config/data/analogues/m04r11/candidate-pools-v2"
        / "candidate-contract.json"
    )
    contract = _read_json(contract_path)
    prereg_raw = _git_blob(
        repository, M11_HISTORICAL_GIT_HEAD, str(M11_PREREG_RELATIVE),
    )
    if sha256(prereg_raw).hexdigest() != M11_HISTORICAL_PREREG_SHA256:
        raise SchedulerError("historical M11 preregistration blob differs")
    try:
        historical_prereg = json.loads(prereg_raw)
    except json.JSONDecodeError as exc:
        raise SchedulerError("historical M11 preregistration is invalid") from exc
    implementation = contract.get("implementation_manifest")
    environment = contract.get("environment_manifest")
    if type(implementation) is not dict or set(implementation) != {"files", "digest"} \
            or type(implementation.get("files")) is not dict \
            or implementation.get("digest") != stable_hash(implementation["files"]):
        raise SchedulerError("M11 implementation manifest differs")
    if type(environment) is not dict or environment.get("digest") != stable_hash(
        _without(environment, {"digest"})
    ):
        raise SchedulerError("M11 environment manifest differs")
    if historical_prereg.get("implementation_manifest_digest") != implementation["digest"] \
            or historical_prereg.get("environment_manifest_digest") != environment["digest"]:
        raise SchedulerError("M11 preregistration/contract manifest link differs")
    if historical_prereg.get("producer_contract") != contract \
            or historical_prereg.get("producer_contract_digest") != contract.get(
                "contract_digest"
            ):
        raise SchedulerError("M11 historical preregistration/contract differs")
    files = implementation["files"]
    if len(files) != 83:
        raise SchedulerError("M11 historical implementation file count differs")
    for relative, expected in sorted(files.items()):
        if type(relative) is not str or type(expected) is not str \
                or sha256(_git_blob(
                    repository, M11_HISTORICAL_GIT_HEAD, relative,
                )).hexdigest() != expected:
            raise SchedulerError(f"M11 historical implementation blob differs: {relative}")
    return _seal({
        "git_head": M11_HISTORICAL_GIT_HEAD,
        "preregistration_path": str(M11_PREREG_RELATIVE),
        "preregistration_sha256": M11_HISTORICAL_PREREG_SHA256,
        "implementation_manifest_digest": implementation["digest"],
        "implementation_file_count": len(files),
        "environment_manifest": environment,
        "candidate_contract_sha256": _sha(contract_path),
    })


def _throughput_selection_binding(repository: Path) -> dict[str, Any]:
    historical = _authenticate_m11_historical(repository)
    bundle_root = repository / "config/data/analogues/m04r11/candidate-pools-v2/case-bundles"
    rows = []
    for path in sorted(bundle_root.glob("*.json")):
        value = _read_json(path)
        semantic = value["semantic"]
        scans = semantic["scan_semantics"]
        if len(scans) != 3 or any(scan != scans[0] for scan in scans[1:]):
            raise SchedulerError("M11 proposal repetitions differ")
        rows.append({
            "query_id": value["query_episode_id"],
            "symbol": semantic["query_symbol"],
            "candidate_count": scans[0]["candidate_count"],
            "eligible_rows": scans[0]["eligible_rows"],
            "rows_scanned": scans[0]["rows_scanned"],
            "bundle_digest": value["bundle_digest"],
        })
    if len(rows) != 60 or len({row["query_id"] for row in rows}) != 60:
        raise SchedulerError("M11 candidate transcript does not contain 60 cases")
    all60 = sorted(rows, key=lambda row: row["query_id"])
    if stable_hash(all60) != M11_ALL60_DIGEST:
        raise SchedulerError("M11 all-60 workload digest differs")
    non_hard = sorted(
        (row for row in rows if row["query_id"] not in FROZEN_QUERY_IDS),
        key=lambda row: (row["candidate_count"], row["query_id"]),
    )
    pool_state = {
        "schema_version": "m04r14-t14-02-opened-workload-pool-v1",
        "source": "m04r11-candidate-pools-v2/case-bundles",
        "excluded_hard_query_ids": sorted(FROZEN_QUERY_IDS),
        "sort": ["candidate_count:ascending", "query_id:ascending"],
        "rows": non_hard,
    }
    if stable_hash(pool_state) != THROUGHPUT_SELECTION_DIGEST:
        raise SchedulerError("M11 non-hard workload-pool digest differs")
    selected = [non_hard[index] for index in (6, 20, 34, 48)]
    expected = [
        {
            "query_id": row[1], "symbol": row[2], "candidate_count": row[3],
            "eligible_rows": row[4], "bundle_digest": row[5],
        }
        for row in THROUGHPUT_SELECTION_TRANSCRIPT
    ]
    if [{key: row[key] for key in expected[0]} for row in selected] != expected:
        raise SchedulerError("mechanical throughput selection differs")
    seal = _read_json(
        repository / "config/data/analogues/m04r11/candidate-pools-v2/SEMANTIC_SEALED.json"
    )
    matrix = _read_json(
        repository / "config/data/analogues/m04r11/candidate-pools-v2/semantic-matrix.json"
    )
    sealed_path = repository / "config/data/analogues/m04r11/candidate-pools-v2/SEMANTIC_SEALED.json"
    complete_path = repository / "config/data/analogues/m04r11/candidate-pools-v2/RUN_COMPLETE.json"
    if seal.get("seal_digest") != M11_SEMANTIC_SEAL_DIGEST \
            or matrix.get("result_digest") != M11_SEMANTIC_MATRIX_DIGEST:
        raise SchedulerError("M11 semantic source seals differ")
    if _sha(sealed_path) != M11_SEMANTIC_SEALED_SHA256 \
            or _sha(complete_path) != M11_RUN_COMPLETE_SHA256:
        raise SchedulerError("M11 terminal source bytes differ")
    state = {
        "rule": (
            "exclude hard4; ascending (candidate_count,query_id); four 14-row "
            "strata; select zero-based indices 6/20/34/48"
        ),
        "source_selection_digest": THROUGHPUT_SELECTION_DIGEST,
        "source_all60_digest": M11_ALL60_DIGEST,
        "semantic_seal_digest": M11_SEMANTIC_SEAL_DIGEST,
        "semantic_matrix_digest": M11_SEMANTIC_MATRIX_DIGEST,
        "full_non_hard_transcript": non_hard,
        "selected": selected,
        "historical_authentication": historical,
        "candidate_root_inventory": [
            {
                "path": str(path.relative_to(repository)),
                "sha256": _sha(path),
            }
            for path in sorted(
                (repository / "config/data/analogues/m04r11/candidate-pools-v2")
                .rglob("*.json")
            )
        ],
    }
    return _seal(state)


class ProductionBackend:
    """Real exposed-case backend; accepts no truth/authority/outcome path."""

    case_ids = FROZEN_CASE_IDS
    query_ids = FROZEN_QUERY_IDS
    throughput_case_ids = THROUGHPUT_CASE_IDS
    throughput_query_ids = THROUGHPUT_QUERY_IDS

    def __init__(
        self, *, repository: Path, config_path: Path, registry_root: Path,
        source_full_root: Path, resident_root: Path,
        require_host_cpus: bool = True,
    ) -> None:
        self.repository = repository.resolve()
        self.config_path = config_path.resolve()
        self.registry_root = registry_root.resolve()
        self.source_full_root = source_full_root.resolve()
        self.resident_root = resident_root.resolve()
        expected_roots = (
            (self.repository / "config/datasets.example.yaml").resolve(),
            (self.repository / "config/data/analogues/m04r10/nasdaq-untouched-authority-registry").resolve(),
            (self.repository / "config/data/analogues/poc/m04r/packed-bound-full").resolve(),
            (Path("/dev/shm/market-analogues/m04r11-candidate-v2")
             / "9fc6ae0ec4451133d8006897162f3443803fd30a3c44b78e80a476fbb18bb483").resolve(),
        )
        if (
            self.config_path, self.registry_root,
            self.source_full_root, self.resident_root,
        ) != expected_roots:
            raise SchedulerError("production input roots differ from frozen topology")
        self.module = _m13(repository)
        registry_path = registry_root / "query-registry.json"
        registry = _read_json(registry_path)
        all_cases = self.module._m12(repository)._validate_registry(registry)
        self._registry_case_ids = tuple(str(row["episode_id"]) for row in all_cases)
        by_query = {str(case["episode_id"]): case for case in all_cases}
        cases = tuple(
            self.module.CaseInput(index, dict(by_query[query_id]))
            for index, query_id in enumerate(THROUGHPUT_QUERY_IDS)
        )
        registry_digest = str(registry["registry_digest"])
        if tuple(case.query_id for case in cases) != THROUGHPUT_QUERY_IDS \
                or tuple(case.case_id for case in cases) != THROUGHPUT_CASE_IDS:
            raise SchedulerError("frozen exposed cases differ")
        self.inputs = self.module.Inputs(
            repository.resolve(), config_path.resolve(), registry_root.resolve(),
            (source_full_root / "store").resolve(), resident_root.resolve(),
            Path("/nonexistent/m04r14-scheduler-output"),
            self.module.GENERATION_ID, self.module.PROVENANCE_DIGEST,
            self.module.RESIDENT_RESERVE_BYTES, registry_digest, cases,
            stable_hash({"schema_version": SCHEMA}),
        )
        self.cases = cases
        self.resident = self.module.resident_full(
            self.inputs.source_store_root, self.inputs.resident_root,
            self.inputs.generation_id, self.inputs.provenance_digest,
            self.inputs.reserve_bytes,
        )
        if require_host_cpus and len(os.sched_getaffinity(0)) < 8:
            raise SchedulerError("scheduler POC requires at least eight CPUs")
        if require_host_cpus:
            cpu = _cgroup_cpu_snapshot()
            _host_cpu_qualification(cpu, cpu, required_cpus=8)
        self._scratch = tempfile.TemporaryDirectory(prefix="m04r14-exact-children-")
        self._child_counter = 0
        self._child_lock = Lock()
        self._proposal_leaves: dict[str, tuple[Path, str]] = {}
        self._source_bindings: dict[int, dict[str, Any]] = {}
        self._causal_input_shas = {
            "config": _sha(self.config_path),
            "registry": _sha(registry_path),
        }
        self._registry_state_digest = stable_hash(registry)

    def foundation(self) -> dict[str, Any]:
        catalog_path = (
            self.inputs.repository
            / "config/data/analogues/m04r14/evidence-catalog-v1/catalog.json"
        )
        oracle_path = (
            self.inputs.repository
            / "config/data/analogues/m04r14/adversarial-oracle-v1/oracle.json"
        )
        catalog = _read_json(catalog_path); oracle = _read_json(oracle_path)
        catalog_module = _load_script(
            self.inputs.repository,
            "experiments/m04r/m04r14_evidence_catalog.py",
            "m04r14_scheduler_catalog_validator",
        )
        oracle_module = _load_script(
            self.inputs.repository,
            "experiments/m04r/m04r14_adversarial_oracle.py",
            "m04r14_scheduler_oracle_validator",
        )
        catalog_module.validate_catalog(catalog, self.inputs.repository)
        oracle_module.validate_payload(oracle, require_production=True)
        if catalog.get("catalog_digest") != CATALOG_DIGEST \
                or oracle.get("result_digest") != ORACLE_RESULT_DIGEST \
                or oracle.get("passed") is not True:
            raise SchedulerError("M04R-14 prerequisite evidence differs")
        self._prerequisite_paths = {
            "catalog": (catalog_path, _sha(catalog_path)),
            "oracle": (oracle_path, _sha(oracle_path)),
        }
        return {
            "registry_digest": self.inputs.registry_digest,
            "generation_id": self.inputs.generation_id,
            "provenance_digest": self.inputs.provenance_digest,
            "resident": self.resident,
            "source_store_root": str(self.inputs.source_store_root),
            "config_path": str(self.config_path),
            "registry_root": str(self.registry_root),
            "resident_root": str(self.resident_root),
            "causal_input_shas": dict(self._causal_input_shas),
            "prerequisites": {
                "evidence_catalog_digest": CATALOG_DIGEST,
                "evidence_catalog_sha256": _sha(catalog_path),
                "adversarial_oracle_result_digest": ORACLE_RESULT_DIGEST,
                "adversarial_oracle_sha256": _sha(oracle_path),
            },
            "throughput_selection": _throughput_selection_binding(
                self.inputs.repository,
            ),
        }

    def _lease(self) -> str:
        return self.module.lease_exact(self.inputs, self.resident)

    def _proposal_command(self, request: Path, output: Path) -> list[str]:
        return [
            sys.executable, str(Path(__file__).resolve()), "_proposal-child",
            "--config", str(self.config_path), "--registry-root", str(self.registry_root),
            "--source-full-root", str(self.source_full_root),
            "--resident-root", str(self.resident_root),
            "--request", str(request), "--output", str(output),
        ]

    def _prepared_from_proposal_child(
        self, case_ordinal: int, task_id: str, semantic: dict[str, Any],
        measurement: dict[str, Any],
    ) -> PreparedCase:
        case = self.cases[case_ordinal]
        source, episode, search_request, packed = self.module._case_context(
            self.inputs, case,
        )
        self.module.validate_proposal(semantic["forward"], packed, self.inputs)
        self.module.validate_proposal(semantic["reverse"], packed, self.inputs)
        self._source_bindings[case_ordinal] = semantic["query_binding"]
        forward = self.module._proposal_report(semantic["forward"])
        return PreparedCase(
            task_id, case.case_id, case.query_id, semantic, measurement,
            (source, episode, search_request, packed, forward, case),
        )

    @staticmethod
    def _prepare_local(
        module: Any, inputs: Any, resident: Mapping[str, Any], case: Any,
        task_id: str,
    ) -> PreparedCase:
        before = _resource_snapshot(); started = perf_counter()
        source, episode, request, packed = module._case_context(inputs, case)
        binding = module.query_binding(
            source, episode, request, packed, inputs.provenance_digest,
        )
        lease = lambda: module.lease_exact(inputs, resident)
        leases = [lease()]
        forward = scan_packed_bound_proposals_threaded(
            Path(resident["store_root"]), inputs.generation_id, packed,
            route_quotas={"composite": module.PROPOSAL_QUOTA},
            block_rows=4096, block_order="forward", threads=PROPOSAL_THREADS,
            branch_aware=True, verify_content=False,
            expected_provenance_digest=inputs.provenance_digest,
        )
        leases.extend((lease(), lease()))
        reverse = scan_packed_bound_proposals_threaded(
            Path(resident["store_root"]), inputs.generation_id, packed,
            route_quotas={"composite": module.PROPOSAL_QUOTA},
            block_rows=4097, block_order="reverse", threads=PROPOSAL_THREADS,
            branch_aware=True, verify_content=False,
            expected_provenance_digest=inputs.provenance_digest,
        )
        leases.append(lease())
        binding_after = module.query_binding(
            source, episode, request, packed, inputs.provenance_digest,
        )
        if binding_after != binding:
            raise SchedulerError("source/query causal binding changed during proposals")
        fwd = module._proposal_payload(forward)
        rev = module._proposal_payload(reverse)
        module.validate_proposal(fwd, packed, inputs)
        module.validate_proposal(rev, packed, inputs)
        omitted = {"block_rows", "block_order", "elapsed_seconds", "peak_rss_mb"}
        if _without(fwd, omitted) != _without(rev, omitted):
            raise SchedulerError("forward/reverse proposal semantics differ")
        after = _resource_snapshot()
        semantic = {
            "schema_version": PROPOSAL_SCHEMA, "task_id": task_id,
            "case_id": case.case_id, "query_id": case.query_id,
            "query_binding": binding, "resident_lease_digests": leases,
            "source_binding_before": binding,
            "source_binding_after": binding_after,
            "resident_snapshot": dict(resident),
            "forward": fwd, "reverse": rev,
            "semantic_digest": stable_hash(_without(fwd, omitted)),
        }
        measurement = {
            "wall_seconds": perf_counter() - started,
            "forward_seconds": forward.elapsed_seconds,
            "reverse_seconds": reverse.elapsed_seconds,
            "resources": _resource_delta(before, after),
        }
        opaque = (source, episode, request, packed, forward, case)
        return PreparedCase(task_id, case.case_id, case.query_id,
                            semantic, measurement, opaque)

    def prepare(self, case_ordinal: int, task_id: str) -> PreparedCase:
        """Run each two-direction proposal preparation in a fresh 8-CPU child."""
        with self._child_lock:
            sequence = self._child_counter; self._child_counter += 1
        scratch = Path(self._scratch.name)
        request_path = scratch / f"proposal-request-{sequence:03d}.json"
        ready_path = scratch / f"proposal-ready-{sequence:03d}.json"
        release_path = scratch / f"proposal-release-{sequence:03d}.json"
        output_path = scratch / f"proposal-result-{sequence:03d}.json"
        request = {
            "schema_version": "m04r14-proposal-child-request-v1",
            "case_ordinal": case_ordinal, "task_id": task_id,
            "resident_snapshot": self.resident,
            "ready_path": str(ready_path), "release_path": str(release_path),
        }
        _atomic(request_path, request)
        affinity = sorted(os.sched_getaffinity(0))
        cpus = affinity[:PROPOSAL_THREADS]
        started = perf_counter(); started_monotonic = time.monotonic()
        host_before = _vmstat_swap()
        process = subprocess.Popen(
            self._proposal_command(request_path, output_path),
            cwd=self.repository, env=self._child_environment(),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            preexec_fn=lambda: os.sched_setaffinity(0, set(cpus)),
        )
        try:
            deadline = perf_counter() + CHILD_STARTUP_TIMEOUT_SECONDS
            while not ready_path.is_file():
                if process.poll() is not None:
                    raise SchedulerError("proposal child exited before READY")
                if perf_counter() > deadline:
                    raise SchedulerError("proposal child READY timed out")
                time.sleep(0.02)
            ready = _read_json(ready_path)
            _exact_keys(ready, {
                "schema_version", "task_id", "pid", "cpu_affinity",
                "thread_environment", "resident_lease_digest", "ready_monotonic",
            }, "proposal child READY")
            if ready["schema_version"] != "m04r14-proposal-child-ready-v1" \
                    or ready["task_id"] != task_id or ready["pid"] != process.pid \
                    or ready["cpu_affinity"] != cpus \
                    or ready["thread_environment"] != self._child_environment_binding():
                raise SchedulerError("proposal child READY binding differs")
            _atomic(release_path, {
                "schema_version": "m04r14-proposal-child-release-v1",
                "task_id": task_id, "pid": process.pid,
                "ready_digest": stable_hash(ready),
                "released_monotonic": time.monotonic(),
            })
            peak = {"VmRSS": 0, "VmHWM": 0, "VmSwap": 0}
            task_deadline = min(
                perf_counter() + CHILD_TASK_TIMEOUT_SECONDS,
                getattr(self, "_run_deadline_monotonic", float("inf")),
            )
            usage = None
            while True:
                observed = _proc_memory(process.pid)
                for key in peak: peak[key] = max(peak[key], observed[key])
                waited, status, usage = os.wait4(process.pid, os.WNOHANG)
                if waited:
                    process.returncode = os.waitstatus_to_exitcode(status); break
                if perf_counter() > task_deadline:
                    raise SchedulerError("proposal child exceeded one-hour timeout")
                time.sleep(0.02)
            if process.returncode != 0 or usage is None:
                raise SchedulerError(f"proposal child failed rc={process.returncode}")
        except BaseException as exc:
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired as cleanup_exc:
                        raise SchedulerError(
                            "proposal child could not be killed and reaped"
                        ) from cleanup_exc
            if process.returncode is None:
                raise SchedulerError("proposal child cleanup did not reap process") from exc
            raise
        payload = _exact_keys(
            _read_json(output_path), {"semantic", "measurement", "created_at"},
            "proposal child result",
        )
        semantic = _sealed(payload["semantic"], "proposal child semantic")["state"]
        if semantic.get("task_id") != task_id \
                or semantic.get("case_id") != self.cases[case_ordinal].case_id \
                or semantic.get("query_id") != self.cases[case_ordinal].query_id:
            raise SchedulerError("proposal child result identity differs")
        wait4_rss = int(usage.ru_maxrss * (1024 if sys.platform == "darwin" else 1))
        measurement = {
            **payload["measurement"],
            "spawned_process": {
                "pid": process.pid, "wall_seconds": perf_counter() - started,
                "cpus": cpus, "ready_evidence": ready,
                "release_evidence": _read_json(release_path),
                "startup_to_ready_seconds": max(
                    0.0, ready["ready_monotonic"] - started_monotonic,
                ),
                "user_cpu_seconds": float(usage.ru_utime),
                "system_cpu_seconds": float(usage.ru_stime),
                "minor_faults": int(usage.ru_minflt), "major_faults": int(usage.ru_majflt),
                "peak_rss_kib": peak["VmRSS"], "peak_hwm_kib": peak["VmHWM"],
                "wait4_max_rss_kib": wait4_rss,
                "effective_peak_rss_kib": max(peak["VmRSS"], peak["VmHWM"], wait4_rss),
                "peak_swap_kib": peak["VmSwap"],
                "final_swap_kib": payload["measurement"]["resources"]["after"]["swap_kib"],
                "host_vmstat_swap_before": host_before,
                "host_vmstat_swap_after": _vmstat_swap(),
                "host_swap_is_context_only": True,
            },
        }
        if peak["VmSwap"] != 0 \
                or measurement["spawned_process"]["final_swap_kib"] != 0:
            raise SchedulerError("proposal child used process-attributed swap")
        return self._prepared_from_proposal_child(
            case_ordinal, task_id, semantic, measurement,
        )

    def _exact_local(self, prepared: PreparedCase, workers: int) -> ExactAttempt:
        if workers not in WORKERS:
            raise SchedulerError("exact worker count is not frozen")
        source, episode, request, _packed, proposal, case = prepared.opaque
        before_lease = self._lease(); before = _resource_snapshot(); started = perf_counter()
        result = certified_packed_search(
            episode, source, request, Path(self.resident["store_root"]),
            self.inputs.generation_id, store_dataset_id="nasdaq",
            initial_frontier_rows=self.module.INITIAL_FRONTIER,
            maximum_frontier_rows=self.module.MAXIMUM_FRONTIER,
            seed_rows=self.module.SEED_ROWS, block_rows=4096, workers=workers,
            sparse_cutoff=8, tolerance=ENGINE_TOLERANCE, verify_content=False,
            requested_positions=True, vector_lower_bounds=True,
            deferred_alignments=True, compact_scored=True,
            native_bound_deferral=True, streaming_threshold_closure=True,
            branch_aware_packed_bounds=True, precomputed_proposal=proposal,
        )
        after = _resource_snapshot(); after_lease = self._lease()
        certificate = json.loads(json.dumps(asdict(result.certificate), allow_nan=False))
        matches = json.loads(json.dumps(
            [self.module._match(value) for value in result.matches], allow_nan=False,
        ))
        self.module.validate_certificate_and_matches(
            certificate, matches, case.query_id,
            expected_input_digest=prepared.semantic["query_binding"][
                "certified_input_digest"
            ],
        )
        binding_after = self.module.query_binding(
            source, episode, request, _packed, self.inputs.provenance_digest,
        )
        if binding_after != prepared.semantic["source_binding_before"]:
            raise SchedulerError("source/query causal binding changed during exact task")
        semantic = {
            "schema_version": ATTEMPT_SCHEMA, "case_id": case.case_id,
            "query_id": case.query_id, "workers": workers,
            "proposal_semantic_digest": prepared.semantic["semantic_digest"],
            "certificate": _without(certificate, {"elapsed_seconds"}),
            "matches": matches,
            "certificate_result_digest": certificate["result_digest"],
            "match_digest": stable_hash(matches),
            "lease_before": before_lease, "lease_after": after_lease,
            "source_binding_before": prepared.semantic["source_binding_before"],
            "source_binding_after": binding_after,
        }
        measurement = {
            "wall_seconds": perf_counter() - started,
            "engine_seconds": certificate["elapsed_seconds"],
            "resources": _resource_delta(before, after),
        }
        return ExactAttempt(semantic, measurement)

    def _child_request(
        self, prepared: PreparedCase, workers: int, *, release_path: Path | None,
    ) -> tuple[Path, Path, Path]:
        with self._child_lock:
            ordinal = self._child_counter; self._child_counter += 1
        root = Path(self._scratch.name)
        request_path = root / f"request-{ordinal:03d}.json"
        output_path = root / f"result-{ordinal:03d}.json"
        ready_path = root / f"ready-{ordinal:03d}.json"
        case_ordinal = self.throughput_query_ids.index(prepared.query_id)
        if prepared.task_id not in self._proposal_leaves:
            raise SchedulerError("exact attempt lacks a persisted proposal binding")
        proposal_path, proposal_sha = self._proposal_leaves[prepared.task_id]
        if _sha(proposal_path) != proposal_sha:
            raise SchedulerError("persisted proposal leaf changed")
        _atomic(request_path, {
            "schema_version": "m04r14-exact-child-request-v1",
            "case_ordinal": case_ordinal, "workers": workers,
            "task_id": prepared.task_id, "case_id": prepared.case_id,
            "query_id": prepared.query_id, "proposal_path": str(proposal_path),
            "proposal_sha256": proposal_sha,
            "ready_path": str(ready_path),
            "release_path": None if release_path is None else str(release_path),
        })
        return request_path, output_path, ready_path

    def bind_proposal(self, prepared: PreparedCase, path: Path) -> None:
        if prepared.task_id in self._proposal_leaves:
            raise SchedulerError("proposal task was bound twice")
        self._proposal_leaves[prepared.task_id] = (path.resolve(strict=True), _sha(path))

    def _command(self, request: Path, output: Path) -> list[str]:
        return [
            sys.executable, str(Path(__file__).resolve()), "_exact-child",
            "--config", str(self.config_path), "--registry-root", str(self.registry_root),
            "--source-full-root", str(self.source_full_root),
            "--resident-root", str(self.resident_root),
            "--request", str(request), "--output", str(output),
        ]

    @staticmethod
    def _child_environment() -> dict[str, str]:
        environment = dict(os.environ)
        for key in THREAD_ENV_KEYS:
            if key != "NUMBA_THREADING_LAYER":
                environment[key] = "1"
        return environment

    @staticmethod
    def _child_environment_binding() -> dict[str, str]:
        return {
            key: os.environ.get(key) if key == "NUMBA_THREADING_LAYER" else "1"
            for key in THREAD_ENV_KEYS
        }

    def _launch_children(
        self, values: Sequence[tuple[PreparedCase, int]],
    ) -> tuple[list[ExactAttempt], dict[str, Any]]:
        affinity = sorted(os.sched_getaffinity(0))
        if len(affinity) < len(values):
            raise SchedulerError("insufficient CPUs for spawned exact children")
        host_before = _vmstat_swap(); started = perf_counter()
        processes: list[dict[str, Any]] = []
        def terminate_remaining() -> None:
            for item in processes:
                process = item["process"]
                if process.returncode is None:
                    try: process.kill()
                    except ProcessLookupError: pass
            for item in processes:
                process = item["process"]
                if process.returncode is None:
                    try:
                        waited, child_status, _ = os.wait4(process.pid, 0)
                        if waited:
                            process.returncode = os.waitstatus_to_exitcode(child_status)
                    except ChildProcessError:
                        process.poll()
        release_path = (
            Path(self._scratch.name) / f"release-{self._child_counter:03d}.json"
            if len(values) > 1 else None
        )
        try:
            for index, (prepared, workers) in enumerate(values):
                request, output, ready = self._child_request(
                    prepared, workers, release_path=release_path,
                )
                cpu_set = (
                    {affinity[index]} if len(values) > 1
                    else set(affinity[:workers])
                )
                process = subprocess.Popen(
                    self._command(request, output), cwd=self.repository,
                    env=self._child_environment(), stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, text=True,
                    preexec_fn=lambda cpu_set=cpu_set: os.sched_setaffinity(0, cpu_set),
                )
                processes.append({
                    "process": process, "request": request, "output": output,
                    "ready": ready,
                    "cpus": sorted(cpu_set), "peak_rss_kib": 0, "peak_hwm_kib": 0,
                    "peak_swap_kib": 0, "started": perf_counter(), "usage": None,
                })
        except BaseException:
            terminate_remaining()
            raise
        ready_deadline = min(
            perf_counter() + CHILD_STARTUP_TIMEOUT_SECONDS,
            getattr(self, "_run_deadline_monotonic", float("inf")),
        )
        try:
            while not all(item["ready"].is_file() for item in processes):
                failed = [
                    item for item in processes
                    if not item["ready"].is_file()
                    and item["process"].poll() is not None
                ]
                if failed:
                    raise SchedulerError("exact child exited before READY barrier")
                if perf_counter() > ready_deadline:
                    raise SchedulerError("exact children did not reach READY barrier")
                time.sleep(0.02)
        except BaseException:
            terminate_remaining()
            raise
        try:
            ready_evidence = [_read_json(item["ready"]) for item in processes]
            for item, ready, (prepared, workers) in zip(
                processes, ready_evidence, values, strict=True,
            ):
                _exact_keys(ready, {
                    "schema_version", "task_id", "pid", "workers", "cpu_affinity",
                    "thread_environment", "ready_monotonic", "proposal_sha256",
                    "resident_lease_digest",
                }, "exact child READY")
                proposal_path, proposal_sha = self._proposal_leaves[prepared.task_id]
                if ready["schema_version"] != "m04r14-exact-child-ready-v1" \
                        or ready["task_id"] != prepared.task_id \
                        or ready["workers"] != workers \
                        or ready["proposal_sha256"] != proposal_sha \
                        or _sha(proposal_path) != proposal_sha \
                        or ready.get("pid") != item["process"].pid \
                        or ready.get("cpu_affinity") != item["cpus"] \
                        or ready.get("thread_environment") != self._child_environment_binding():
                    raise SchedulerError("exact child READY identity/environment differs")
            release_evidence = None
            wave_started = None
            if release_path is not None:
                release_evidence = {
                    "schema_version": "m04r14-exact-child-release-v1",
                    "released_monotonic": time.monotonic(),
                    "pids": [item["process"].pid for item in processes],
                    "ready_digests": [stable_hash(value) for value in ready_evidence],
                }
                _atomic(release_path, release_evidence)
                wave_started = perf_counter()
        except BaseException:
            terminate_remaining()
            raise
        remaining = set(range(len(processes)))
        try:
            while remaining:
                for index in tuple(remaining):
                    item = processes[index]; process = item["process"]
                    status = _proc_memory(process.pid)
                    item["peak_rss_kib"] = max(item["peak_rss_kib"], status["VmRSS"])
                    item["peak_hwm_kib"] = max(item["peak_hwm_kib"], status["VmHWM"])
                    item["peak_swap_kib"] = max(item["peak_swap_kib"], status["VmSwap"])
                    waited, child_status, usage = os.wait4(process.pid, os.WNOHANG)
                    if waited:
                        process.returncode = os.waitstatus_to_exitcode(child_status)
                        item["usage"] = usage; item["ended"] = perf_counter()
                        remaining.remove(index)
                    elif perf_counter() - item["started"] > CHILD_TASK_TIMEOUT_SECONDS \
                            or perf_counter() > getattr(
                                self, "_run_deadline_monotonic", float("inf"),
                            ):
                        raise SchedulerError("exact child exceeded one-hour hard timeout")
                if remaining:
                    time.sleep(0.02)
        except BaseException:
            terminate_remaining()
            raise
        attempts = []
        child_metrics = []
        for item in processes:
            process = item["process"]
            if process.returncode != 0:
                raise SchedulerError(
                    f"exact child failed rc={process.returncode}"
                )
            payload = _exact_keys(
                _read_json(item["output"]),
                {"semantic", "measurement", "created_at"},
                "exact child result",
            )
            if not _timestamp_valid(payload["created_at"]):
                raise SchedulerError("exact child result timestamp differs")
            semantic_seal = payload.get("semantic")
            if type(semantic_seal) is not dict \
                    or semantic_seal.get("digest") != stable_hash(
                        semantic_seal.get("state")
                    ):
                raise SchedulerError("exact child semantic seal differs")
            attempt = ExactAttempt(
                semantic_seal["state"], payload["measurement"],
            )
            prepared, workers = values[len(attempts)]
            _validate_attempt(prepared, attempt, workers)
            attempts.append(attempt)
            usage = item["usage"]
            wait4_rss_kib = int(
                usage.ru_maxrss * (1024 if sys.platform == "darwin" else 1)
            )
            child_metrics.append({
                "pid": process.pid, "cpus": item["cpus"],
                "wall_seconds": item["ended"] - item["started"],
                "user_cpu_seconds": float(usage.ru_utime),
                "system_cpu_seconds": float(usage.ru_stime),
                "minor_faults": int(usage.ru_minflt),
                "major_faults": int(usage.ru_majflt),
                "peak_rss_kib": item["peak_rss_kib"],
                "peak_hwm_kib": item["peak_hwm_kib"],
                "wait4_max_rss_kib": wait4_rss_kib,
                "effective_peak_rss_kib": max(
                    item["peak_rss_kib"], item["peak_hwm_kib"], wait4_rss_kib,
                ),
                "peak_swap_kib": item["peak_swap_kib"],
                "final_swap_kib": payload["measurement"]["resources"]["after"]["swap_kib"],
            })
        evidence = {
            "wall_seconds": perf_counter() - started, "children": child_metrics,
            "host_vmstat_swap_before": host_before,
            "host_vmstat_swap_after": _vmstat_swap(),
            "host_swap_is_context_only": True,
            "process_swap_gate_passed": all(
                row["peak_swap_kib"] == 0 and row["final_swap_kib"] == 0
                for row in child_metrics
            ),
            "ready_evidence": ready_evidence,
            "release_evidence": release_evidence,
            "observed_concurrent_children": len(ready_evidence),
            "release_to_all_children_exit_seconds": (
                None if wave_started is None else
                max(item["ended"] for item in processes) - wave_started
            ),
        }
        if not evidence["process_swap_gate_passed"]:
            raise SchedulerError("exact child used process-attributed swap")
        attempts = [
            ExactAttempt(
                attempt.semantic,
                {**attempt.measurement, "spawned_process": child_metrics[index]},
            )
            for index, attempt in enumerate(attempts)
        ]
        return attempts, evidence

    def exact(self, prepared: PreparedCase, workers: int) -> ExactAttempt:
        attempts, _ = self._launch_children(((prepared, workers),))
        return attempts[0]

    def exact_batch(
        self, prepared: Sequence[PreparedCase], workers: int,
    ) -> tuple[list[ExactAttempt], dict[str, Any]]:
        if len(prepared) != 8 or workers != 1:
            raise SchedulerError("production throughput batch must be p8t1")
        return self._launch_children(tuple((value, 1) for value in prepared))

    def final_lease(self) -> dict[str, Any]:
        registry = _read_json(self.registry_root / "query-registry.json")
        final_cases = self.module._m12(self.repository)._validate_registry(registry)
        if stable_hash(registry) != self._registry_state_digest \
                or str(registry.get("registry_digest")) != self.inputs.registry_digest \
                or tuple(str(row["episode_id"]) for row in final_cases) != self._registry_case_ids:
            raise SchedulerError("registry causal state changed")
        if self._causal_input_shas != {
            "config": _sha(self.config_path),
            "registry": _sha(self.registry_root / "query-registry.json"),
        }:
            raise SchedulerError("config/registry causal input changed")
        current = self.module.resident_full(
            self.inputs.source_store_root, self.inputs.resident_root,
            self.inputs.generation_id, self.inputs.provenance_digest,
            self.inputs.reserve_bytes,
        )
        if current != self.resident:
            raise SchedulerError("resident foundation changed")
        for name, (path, expected_sha) in self._prerequisite_paths.items():
            if _sha(path) != expected_sha:
                raise SchedulerError(f"{name} prerequisite changed before final seal")
        final_source = {}
        for ordinal, expected in sorted(self._source_bindings.items()):
            case = self.cases[ordinal]
            source, episode, request, packed = self.module._case_context(
                self.inputs, case,
            )
            observed = self.module.query_binding(
                source, episode, request, packed, self.inputs.provenance_digest,
            )
            if observed != expected:
                raise SchedulerError("source binding changed before final seal")
            final_source[str(ordinal)] = stable_hash(observed)
        return {
            "resident_identity_digest": current["identity_digest"],
            "source_binding_digests": final_source,
            "causal_input_shas": dict(self._causal_input_shas),
        }


def _rotated(values: Sequence[Any], offset: int) -> tuple[Any, ...]:
    offset %= len(values)
    return tuple(values[offset:]) + tuple(values[:offset])


def _attempt_semantic_core(value: Mapping[str, Any]) -> dict[str, Any]:
    return _without(value, {"workers", "lease_before", "lease_after"})


def _validate_attempt(
    prepared: PreparedCase, attempt: ExactAttempt, workers: int,
) -> None:
    semantic = attempt.semantic; measurement = attempt.measurement
    required = {
        "schema_version", "case_id", "query_id", "workers",
        "proposal_semantic_digest", "certificate", "matches",
        "certificate_result_digest", "match_digest", "lease_before", "lease_after",
    }
    if "source_binding_before" in semantic or "source_binding_after" in semantic:
        required |= {"source_binding_before", "source_binding_after"}
    if type(semantic) is not dict or set(semantic) != required \
            or semantic["schema_version"] != ATTEMPT_SCHEMA \
            or semantic["case_id"] != prepared.case_id \
            or semantic["query_id"] != prepared.query_id \
            or semantic["workers"] != workers \
            or semantic["proposal_semantic_digest"] != prepared.semantic["semantic_digest"]:
        raise SchedulerError("exact attempt schema/binding differs")
    if semantic["certificate_result_digest"] != semantic["certificate"].get(
        "result_digest"
    ) or semantic["match_digest"] != stable_hash(semantic["matches"]):
        raise SchedulerError("exact attempt semantic digests differ")
    if semantic["lease_before"] != semantic["lease_after"]:
        raise SchedulerError("resident lease changed during exact attempt")
    if "source_binding_before" in semantic and (
        semantic["source_binding_before"] != semantic["source_binding_after"]
        or semantic["source_binding_before"] != prepared.semantic["source_binding_before"]
    ):
        raise SchedulerError("source binding changed during exact attempt")
    if type(measurement) is not dict or any(
        type(value) is not float or not math.isfinite(value) or value < 0
        for name, value in measurement.items()
        if name.endswith("seconds")
    ):
        raise SchedulerError("exact attempt measurement differs")
    measurement_keys = {"wall_seconds", "engine_seconds", "resources"}
    if "spawned_process" in measurement:
        measurement_keys.add("spawned_process")
    _exact_keys(measurement, measurement_keys, "exact attempt measurement")
    if measurement["wall_seconds"] < measurement["engine_seconds"]:
        raise SchedulerError("exact attempt timing nesting differs")
    _validate_resource_evidence(measurement["resources"])
    if "spawned_process" in measurement:
        _validate_child_metric(measurement["spawned_process"])


def _stable_selection(primary_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_worker: dict[int, list[float]] = {worker: [] for worker in WORKERS}
    by_case_worker: dict[tuple[str, int], list[float]] = {}
    for row in primary_rows:
        worker = row["workers"]
        wall = row["measurement"]["wall_seconds"]
        if type(worker) is not int or worker not in WORKERS \
                or type(wall) is not float or not (0 <= wall < float("inf")):
            raise SchedulerError("primary measurement differs")
        by_worker[worker].append(wall)
        by_case_worker.setdefault((row["case_id"], worker), []).append(wall)
    if any(len(values) != len(FROZEN_QUERY_IDS) * REPETITIONS
           for values in by_worker.values()):
        raise SchedulerError("worker matrix is incomplete")
    stable = []
    summaries = {}
    for worker, values in by_worker.items():
        case_medians = []
        ratios = []
        for case_id in FROZEN_CASE_IDS:
            samples = sorted(by_case_worker[(case_id, worker)])
            ratios.append(max(samples) / max(min(samples), 1e-12))
            case_medians.append(samples[1])
        score = sum(case_medians)
        is_stable = max(ratios) <= 1.25
        summaries[str(worker)] = {
            "sample_count": len(values), "per_case_median_seconds": case_medians,
            "raw_score_seconds": score,
            "maximum_case_repeat_ratio": max(ratios), "stable": is_stable,
        }
        if is_stable:
            stable.append(worker)
    if not stable:
        raise SchedulerError("no exact-worker configuration is stable")
    fastest = min(summaries[str(worker)]["raw_score_seconds"] for worker in stable)
    selected = min(
        worker for worker in stable
        if summaries[str(worker)]["raw_score_seconds"] <= fastest * 1.03
    )
    state = {
        "rule": (
            "each case/worker max/min<=1.25; raw score=sum of four "
            "three-repetition medians; smallest worker with raw score<=1.03*fastest"
        ),
        "workers": summaries, "selected_workers": selected,
    }
    return _seal(state)


def _validate_backend(backend: ExecutionBackend) -> None:
    if backend.query_ids != FROZEN_QUERY_IDS or backend.case_ids != FROZEN_CASE_IDS:
        raise SchedulerError("backend exposed-case identity differs")
    if backend.throughput_query_ids != THROUGHPUT_QUERY_IDS \
            or backend.throughput_case_ids != THROUGHPUT_CASE_IDS:
        raise SchedulerError("backend throughput identity differs")


def _execution_policy() -> dict[str, Any]:
    return {
        "case_ids": list(FROZEN_CASE_IDS),
        "query_ids": list(FROZEN_QUERY_IDS),
        "throughput_case_ids": list(THROUGHPUT_CASE_IDS),
        "throughput_query_ids": list(THROUGHPUT_QUERY_IDS),
        "workers": list(WORKERS), "repetitions": REPETITIONS,
        "primary_order": [
            list(_rotated(tuple(range(4)), repetition))
            for repetition in range(REPETITIONS)
        ],
        "worker_order": [
            [list(_rotated(WORKERS, repetition + ordinal)) for ordinal in range(4)]
            for repetition in range(REPETITIONS)
        ],
        "proposal": {
            "fresh_spawned_child": True, "threads": PROPOSAL_THREADS,
            "cpu_count": PROPOSAL_THREADS, "quota": 16385,
            "forward": {"block_rows": 4096, "block_order": "forward"},
            "reverse": {"block_rows": 4097, "block_order": "reverse"},
            "branch_aware": True, "verify_content": False,
            "serial": True, "scans_per_preparation": 2,
        },
        "exact": {
            "fresh_spawned_child_per_attempt": True,
            "initial_frontier_rows": 1000, "maximum_frontier_rows": 16384,
            "seed_rows": 512, "block_rows": 4096, "sparse_cutoff": 8,
            "requested_positions": True, "vector_lower_bounds": True,
            "deferred_alignments": True, "compact_scored": True,
            "native_bound_deferral": True, "streaming_threshold_closure": True,
            "branch_aware_packed_bounds": True,
            "precomputed_forward_proposal": True,
        },
        "throughput": {
            "label": "exact-stage exposed-host p8t1 microbenchmark",
            "distinct_tasks": 8, "workers_per_task": 1,
            "proposal_preparation": "serial-before-durable-barrier",
            "timed_wave": "eight-ready-one-release-spawned-processes",
            "serial_controls": "after-wave-excluded-from-timing",
            "affinity": "one-distinct-logical-cpu-per-child",
            "supervisor_shares_host_capacity_and_is_not_claimed_nonoversubscribed": True,
            "excluded_from_worker_selection": True,
        },
        "selection": {
            "case_worker_repeat_max_min_ratio": 1.25,
            "score": "sum-of-four-three-repeat-raw-medians",
            "near_fastest_factor": 1.03,
            "tie": "smallest-worker-within-factor-no-rounding",
        },
        "numeric": {
            "semantic_identity": "byte/canonical-digest-exact",
            "evidence_atol_hex": NUMERIC_ATOL.hex(), "evidence_rtol": 0,
            "engine_algorithm_tolerance_hex": ENGINE_TOLERANCE.hex(),
        },
        "resource": {
            "child_process_swap_kib": 0,
            "host_vmstat_swap": "context-only",
            "peak_rss": "max-proc-rss-proc-hwm-wait4-ru_maxrss",
            "startup_timeout_seconds": CHILD_STARTUP_TIMEOUT_SECONDS,
            "task_timeout_seconds": CHILD_TASK_TIMEOUT_SECONDS,
            "run_hard_limit_seconds": RUN_HARD_LIMIT_SECONDS,
        },
        "thread_environment": {
            "numeric_thread_variables": "1",
            "numba_threading_layer": "preserve-preregistered-observed-value",
        },
        "lifecycle": {
            "create_only": True, "resume": False,
            "failure_terminal": INCOMPLETE_SCHEMA,
            "complete_published_last": True,
            "exact_terminal_tree": sorted(_expected_terminal_paths(include_complete=True)),
        },
        "claims": {
            "development_only": True, "cases_previously_exposed": True,
            "direct_raw_authority_accessed_by_this_run": False,
            "authority_derived_prerequisite_evidence_accessed_by_this_run": True,
            "forward_outcomes_accessed_by_this_run": False,
            "production_promotion_authorized": False,
        },
        "schemas": {
            "run": RUN_SCHEMA, "proposal": PROPOSAL_SCHEMA,
            "attempt": ATTEMPT_SCHEMA, "semantics": SEMANTICS_SCHEMA,
            "measurements": MEASUREMENTS_SCHEMA, "complete": COMPLETE_SCHEMA,
            "incomplete": INCOMPLETE_SCHEMA,
        },
    }


def _preregistration_payload(
    root: Path, foundation: Mapping[str, Any], runtime: Mapping[str, Any],
) -> dict[str, Any]:
    return _seal({
        "schema_version": "m04r14-exact-scheduler-preregistration-v1",
        "status": "frozen_before_run", "runtime_binding": dict(runtime),
        "foundation": dict(foundation), "output_root": str(root.resolve()),
        "query_ids": list(FROZEN_QUERY_IDS),
        "throughput_query_ids": list(THROUGHPUT_QUERY_IDS),
        "workers": list(WORKERS), "repetitions": REPETITIONS,
        "execution_policy": _execution_policy(),
        "selection_rule": PREREG_SELECTION_RULE,
    })


def _overlap(left: Path, right: Path) -> bool:
    left = left.resolve(); right = right.resolve()
    return left == right or left in right.parents or right in left.parents


def _reject_symlink_ancestry(path: Path) -> None:
    current = path.absolute()
    for candidate in (current, *current.parents):
        if candidate.exists() and candidate.is_symlink():
            raise SchedulerError(f"symlinked path ancestry is forbidden: {path}")


def _validate_topology(
    repository: Path, output: Path, preregistration: Path,
    config: Path, registry: Path, source: Path, resident: Path,
) -> None:
    for path in (output, preregistration, config, registry, source, resident):
        _reject_symlink_ancestry(path)
    protected = (
        config, registry, source, resident, repository / ".git",
        repository / "config/data/analogues/m04r13",
        repository / "config/data/analogues/m04r14/evidence-catalog-v1",
        repository / "config/data/analogues/m04r14/adversarial-oracle-v1",
        repository / "config/data/analogues/m04r11/candidate-pools-v2",
    )
    if output.resolve() == preregistration.resolve() \
            or any(_overlap(output, path) for path in protected) \
            or any(_overlap(preregistration, path) for path in protected):
        raise SchedulerError("production evidence topology overlaps protected inputs")


def _validate_prereg_launch_envelope(
    prereg: Mapping[str, Any], *, repository: Path, output: Path,
    prereg_path: Path, config: Path, registry: Path, source: Path, resident: Path,
) -> Mapping[str, Any]:
    sealed = _sealed(prereg, "launch preregistration")
    state = _exact_keys(sealed["state"], {
        "schema_version", "status", "runtime_binding", "foundation",
        "output_root", "query_ids", "throughput_query_ids", "workers",
        "repetitions", "execution_policy", "selection_rule",
    }, "launch preregistration state")
    foundation = _exact_keys(state["foundation"], {
        "registry_digest", "generation_id", "provenance_digest", "resident",
        "source_store_root", "config_path", "registry_root", "resident_root",
        "causal_input_shas", "prerequisites", "throughput_selection",
    }, "launch foundation")
    if state["schema_version"] != "m04r14-exact-scheduler-preregistration-v1" \
            or state["status"] != "frozen_before_run" \
            or state["output_root"] != str(output.resolve()) \
            or state["query_ids"] != list(FROZEN_QUERY_IDS) \
            or state["throughput_query_ids"] != list(THROUGHPUT_QUERY_IDS) \
            or state["workers"] != list(WORKERS) \
            or state["repetitions"] != REPETITIONS \
            or state["execution_policy"] != _execution_policy() \
            or state["selection_rule"] != PREREG_SELECTION_RULE \
            or foundation["config_path"] != str(config.resolve()) \
            or foundation["registry_root"] != str(registry.resolve()) \
            or foundation["source_store_root"] != str((source / "store").resolve()) \
            or foundation["resident_root"] != str(resident.resolve()):
        raise SchedulerError("launch preregistration envelope differs")
    _validate_runtime(state["runtime_binding"], repository)
    prereg_relative = prereg_path.resolve().relative_to(repository.resolve())
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(prereg_relative)],
        cwd=repository, capture_output=True, check=False,
    )
    if tracked.returncode:
        raise SchedulerError("launch preregistration is not tracked")
    _validate_h1_commit(
        repository, prereg_path.resolve(), state["runtime_binding"]["state"]["git_head"],
    )
    return state


def _validate_h1_commit(repository: Path, preregistration: Path, h0: str) -> None:
    """Require one direct, preregistration-only commit after frozen runtime H0."""
    repository = repository.resolve(strict=True)
    preregistration = preregistration.resolve(strict=True)
    try:
        relative = preregistration.relative_to(repository)
    except ValueError as exc:
        raise SchedulerError("preregistration is outside repository") from exc
    current = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    ancestry = subprocess.run(
        ["git", "rev-list", "--parents", "-n", "1", current], cwd=repository,
        text=True, capture_output=True, check=True,
    ).stdout.split()
    changed = subprocess.run(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", current],
        cwd=repository, text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    if ancestry != [current, h0] or changed != [str(relative)]:
        raise SchedulerError("launch requires one preregistration-only H1 commit")
    committed = subprocess.run(
        ["git", "show", f"{current}:{relative}"], cwd=repository,
        capture_output=True, check=True,
    ).stdout
    if sha256(committed).hexdigest() != _sha(preregistration):
        raise SchedulerError("H1 preregistration blob differs from worktree")


def execute(
    root: Path, backend: ExecutionBackend, *,
    concurrent: Callable[..., Any] = ThreadPoolExecutor,
    runtime_binding: Mapping[str, Any] | None = None,
    preregistration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute both lanes and publish create-only reconstructible checkpoints."""
    _validate_backend(backend)
    foundation = backend.foundation()
    runtime = dict(runtime_binding or test_foundation())
    expected_preregistration = _preregistration_payload(root, foundation, runtime)
    if preregistration is None:
        if isinstance(backend, ProductionBackend):
            raise SchedulerError("production scheduler requires frozen preregistration")
        preregistration = expected_preregistration
    if dict(preregistration) != expected_preregistration:
        raise SchedulerError("scheduler preregistration differs")
    if root.exists() or root.is_symlink():
        raise SchedulerError("scheduler output root must be absent")
    root.mkdir(parents=False)
    run_started_monotonic = perf_counter()
    run_deadline_monotonic = run_started_monotonic + RUN_HARD_LIMIT_SECONDS
    host_cpu_before = _cgroup_cpu_snapshot()
    if isinstance(backend, ProductionBackend):
        backend._run_deadline_monotonic = run_deadline_monotonic
    def check_run_deadline() -> None:
        if perf_counter() > run_deadline_monotonic:
            raise SchedulerError("scheduler exceeded six-hour hard limit")
    started_state = {
        "schema_version": RUN_SCHEMA, "status": "started",
        "development_only": True, "cases_previously_exposed": True,
        "direct_raw_authority_accessed_by_this_run": False,
        "authority_derived_prerequisite_evidence_accessed_by_this_run": True,
        "forward_outcomes_accessed_by_this_run": False,
        "raw_authority_or_outcome_paths_accepted": False,
        "production_promotion_authorized": False,
        "query_ids": list(FROZEN_QUERY_IDS), "case_ids": list(FROZEN_CASE_IDS),
        "workers": list(WORKERS), "repetitions": REPETITIONS,
        "proposal_threads": PROPOSAL_THREADS,
        "throughput": {
            "tasks": THROUGHPUT_TASKS, "distinct_queries": True,
            "maximum_concurrent_exact_tasks": 8, "workers_per_task": 1,
            "proposal_phase": "serial-before-barrier",
        },
        "numeric_policy": {
            "semantic_digest_equality": "exact",
            "scorer_absolute_tolerance_hex": NUMERIC_ATOL.hex(),
            "engine_algorithm_tolerance_hex": ENGINE_TOLERANCE.hex(),
        },
        "foundation": foundation,
        "runtime_binding": runtime,
        "preregistration_digest": preregistration["digest"],
    }
    try:
        _atomic(root / "CONTRACT.json", preregistration)
        _atomic(root / "RUN_STARTED.json", {
            **_seal(started_state), "created_at": _timestamp(),
        })
    except BaseException as exc:
        try: _write_incomplete(root, exc)
        except BaseException: pass
        raise
    primary_semantics: list[dict[str, Any]] = []
    primary_measurements: list[dict[str, Any]] = []
    throughput_semantics: list[dict[str, Any]] = []
    throughput_measurements: list[dict[str, Any]] = []
    reference: dict[str, dict[str, Any]] = {}
    try:
        for repetition in range(REPETITIONS):
            for ordinal in _rotated(tuple(range(4)), repetition):
                check_run_deadline()
                task_id = f"primary-r{repetition}-c{ordinal}"
                prepared = backend.prepare(ordinal, task_id)
                check_run_deadline()
                proposal_payload = {
                    **_seal(prepared.semantic), "measurement": prepared.measurement,
                    "created_at": _timestamp(),
                }
                task_root = root / "primary" / f"r{repetition}" / f"c{ordinal}"
                proposal_path = task_root / "PROPOSAL.json"
                _atomic(proposal_path, proposal_payload)
                binder = getattr(backend, "bind_proposal", None)
                if callable(binder):
                    binder(prepared, proposal_path)
                for worker in _rotated(WORKERS, repetition + ordinal):
                    check_run_deadline()
                    attempt = backend.exact(prepared, worker)
                    check_run_deadline()
                    _validate_attempt(prepared, attempt, worker)
                    core = _attempt_semantic_core(attempt.semantic)
                    prior = reference.setdefault(prepared.case_id, core)
                    if core != prior:
                        raise SchedulerError("exact semantics differ across workers/repeats")
                    semantic_row = {
                        "lane": "primary", "repetition": repetition,
                        "case_id": prepared.case_id, "query_id": prepared.query_id,
                        "workers": worker, "task_id": task_id,
                        "attempt": attempt.semantic,
                    }
                    measurement_row = {
                        "lane": "primary", "repetition": repetition,
                        "case_id": prepared.case_id, "workers": worker,
                        "task_id": task_id, "measurement": attempt.measurement,
                        "proposal_measurement": prepared.measurement,
                    }
                    _atomic(task_root / f"EXACT-w{worker}.json", {
                        "semantic": _seal(semantic_row),
                        "measurement": measurement_row, "created_at": _timestamp(),
                    })
                    primary_semantics.append(semantic_row)
                    primary_measurements.append(measurement_row)

        prepared_throughput: list[PreparedCase] = []
        throughput_order = tuple(range(THROUGHPUT_TASKS))
        event_ledger: list[dict[str, Any]] = []
        for task_index, ordinal in enumerate(throughput_order):
            check_run_deadline()
            task_id = f"throughput-t{task_index}-c{ordinal}"
            event_ledger.append({
                "sequence": len(event_ledger), "event": "proposal_started",
                "task_id": task_id, "active_proposals": 1,
            })
            prepared = backend.prepare(ordinal, task_id)
            check_run_deadline()
            prepared_throughput.append(prepared)
            proposal_path = root / "throughput" / f"t{task_index}" / "PROPOSAL.json"
            _atomic(proposal_path, {
                **_seal(prepared.semantic), "measurement": prepared.measurement,
                "created_at": _timestamp(),
            })
            binder = getattr(backend, "bind_proposal", None)
            if callable(binder):
                binder(prepared, proposal_path)
            event_ledger.append({
                "sequence": len(event_ledger), "event": "proposal_completed",
                "task_id": task_id, "active_proposals": 0,
                "proposal_semantic_digest": prepared.semantic["semantic_digest"],
            })
        barrier_state = {
            "status": "all_eight_proposals_complete_before_timed_wave",
            "query_ids": list(THROUGHPUT_QUERY_IDS),
            "proposal_semantic_digests": [
                value.semantic["semantic_digest"] for value in prepared_throughput
            ],
        }
        barrier = _seal(barrier_state)
        _atomic(root / "throughput" / "PROPOSALS_COMPLETE.json", barrier)
        event_ledger.append({
            "sequence": len(event_ledger), "event": "barrier_released",
            "barrier_digest": barrier["digest"],
        })
        lane_before = _resource_snapshot(); lane_started = perf_counter()
        active_lock = Lock(); active_exact = 0; maximum_active_exact = 0
        batch_process_evidence: dict[str, Any] | None = None

        def batch_exact(prepared: PreparedCase) -> ExactAttempt:
            nonlocal active_exact, maximum_active_exact
            with active_lock:
                active_exact += 1
                maximum_active_exact = max(maximum_active_exact, active_exact)
                event_ledger.append({
                    "sequence": len(event_ledger), "event": "batch_exact_started",
                    "task_id": prepared.task_id, "active_exact_tasks": active_exact,
                })
            try:
                return backend.exact(prepared, 1)
            finally:
                with active_lock:
                    active_exact -= 1
                    event_ledger.append({
                        "sequence": len(event_ledger), "event": "batch_exact_ended",
                        "task_id": prepared.task_id,
                        "active_exact_tasks": active_exact,
                    })

        completed: dict[int, tuple[PreparedCase, ExactAttempt]] = {}
        production_batch = getattr(backend, "exact_batch", None)
        if callable(production_batch):
            for prepared in prepared_throughput:
                event_ledger.append({
                    "sequence": len(event_ledger), "event": "batch_exact_started",
                    "task_id": prepared.task_id, "active_exact_tasks": 8,
                })
            attempts, batch_process_evidence = production_batch(
                prepared_throughput, 1,
            )
            check_run_deadline()
            if len(attempts) != 8:
                raise SchedulerError("spawned p8t1 batch result count differs")
            maximum_active_exact = 8
            for index, (prepared, attempt) in enumerate(zip(
                prepared_throughput, attempts, strict=True,
            )):
                completed[index] = (prepared, attempt)
                event_ledger.append({
                    "sequence": len(event_ledger), "event": "batch_exact_ended",
                    "task_id": prepared.task_id, "active_exact_tasks": 0,
                })
        else:
            with concurrent(max_workers=8) as pool:
                futures = {
                    pool.submit(batch_exact, prepared): (index, prepared)
                    for index, prepared in enumerate(prepared_throughput)
                }
                for future in as_completed(futures):
                    index, prepared = futures[future]
                    completed[index] = (prepared, future.result())
                    check_run_deadline()
        lane_after = _resource_snapshot(); lane_wall_seconds = perf_counter() - lane_started
        if batch_process_evidence is not None:
            lane_wall_seconds = batch_process_evidence[
                "release_to_all_children_exit_seconds"
            ]
        if maximum_active_exact <= 1 or maximum_active_exact > 8:
            raise SchedulerError("throughput lane did not demonstrate bounded concurrency")
        controls: dict[int, ExactAttempt] = {}
        for index, prepared in enumerate(prepared_throughput):
            check_run_deadline()
            control = backend.exact(prepared, 1)
            check_run_deadline()
            _validate_attempt(prepared, control, 1)
            controls[index] = control
            _atomic(root / "throughput" / f"t{index}" / "CONTROL-w1.json", {
                "semantic": _seal(control.semantic),
                "measurement": control.measurement, "created_at": _timestamp(),
            })
            event_ledger.append({
                "sequence": len(event_ledger), "event": "control_completed_after_wave",
                "task_id": prepared.task_id,
                "semantic_digest": stable_hash(_attempt_semantic_core(control.semantic)),
            })
        for index in range(THROUGHPUT_TASKS):
            prepared, attempt = completed[index]
            _validate_attempt(prepared, attempt, 1)
            expected = _attempt_semantic_core(controls[index].semantic)
            if _attempt_semantic_core(attempt.semantic) != expected:
                raise SchedulerError("throughput semantics differ from serial control")
            if prepared.case_id in reference and expected != reference[prepared.case_id]:
                raise SchedulerError("throughput hard-case control differs from primary")
            semantic_row = {
                "lane": "throughput-p8t1", "task_index": index,
                "case_id": prepared.case_id, "query_id": prepared.query_id,
                "workers": 1, "task_id": prepared.task_id,
                "attempt": attempt.semantic,
            }
            measurement_row = {
                "lane": "throughput-p8t1", "task_index": index,
                "case_id": prepared.case_id, "workers": 1,
                "task_id": prepared.task_id, "measurement": attempt.measurement,
            }
            _atomic(root / "throughput" / f"t{index}" / "EXACT-w1.json", {
                "semantic": _seal(semantic_row), "measurement": measurement_row,
                "created_at": _timestamp(),
            })
            throughput_semantics.append(semantic_row)
            throughput_measurements.append(measurement_row)
            event_ledger.append({
                "sequence": len(event_ledger), "event": "batch_exact_completed",
                "task_id": prepared.task_id,
                "semantic_digest": stable_hash(expected),
            })

        selection = _stable_selection(primary_measurements)
        selected_workers = selection["state"]["selected_workers"]
        interactive_rows = [{
            "case_id": row["case_id"], "repetition": row["repetition"],
            "forward_proposal_seconds": row["proposal_measurement"]["forward_seconds"],
            "selected_exact_seconds": row["measurement"]["wall_seconds"],
            "interactive_seconds": (
                row["proposal_measurement"]["forward_seconds"]
                + row["measurement"]["wall_seconds"]
            ),
            "reverse_proposal_role": "semantic-parity-only-excluded",
        } for row in primary_measurements if row["workers"] == selected_workers]
        final_lease = backend.final_lease()
        check_run_deadline()
        host_cpu_qualification = _host_cpu_qualification(
            host_cpu_before, _cgroup_cpu_snapshot(), required_cpus=8,
        )
        if isinstance(backend, ProductionBackend):
            _validate_runtime(runtime, backend.repository)
            if _throughput_selection_binding(backend.repository) != foundation[
                "throughput_selection"
            ]:
                raise SchedulerError("opened workload source changed before seal")
        chained_events = []
        previous_event_digest = None
        for row in event_ledger:
            event_state = {
                **row, "previous_event_digest": previous_event_digest,
            }
            event_digest = stable_hash(event_state)
            chained_events.append({**event_state, "event_digest": event_digest})
            previous_event_digest = event_digest
        semantic_state = {
            "schema_version": SEMANTICS_SCHEMA, "status": "semantic_pass",
            "primary": primary_semantics, "throughput": throughput_semantics,
            "case_reference_digests": {
                case_id: stable_hash(value) for case_id, value in sorted(reference.items())
            },
            "semantic_passed": True, "cases_previously_exposed": True,
            "direct_raw_authority_accessed_by_this_run": False,
            "authority_derived_prerequisite_evidence_accessed_by_this_run": True,
            "forward_outcomes_accessed_by_this_run": False,
        }
        measurement_state = {
            "schema_version": MEASUREMENTS_SCHEMA, "status": "measured",
            "primary": primary_measurements,
            "throughput": throughput_measurements,
            "throughput_lane": {
                "label": "exact-stage exposed-host p8t1 microbenchmark",
                "wall_seconds": lane_wall_seconds,
                "resources": _resource_delta(lane_before, lane_after),
                "tasks": THROUGHPUT_TASKS, "distinct_query_ids": list(
                    THROUGHPUT_QUERY_IDS
                ),
                "maximum_concurrent_exact_tasks": 8, "workers_per_task": 1,
                "observed_maximum_active_exact_tasks": maximum_active_exact,
                "proposals_prepared_serially_with_threads": PROPOSAL_THREADS,
                "serial_controls_excluded_from_lane_timing": True,
                "spawned_process_evidence": batch_process_evidence,
            },
            "selection": selection,
            "host_cpu_qualification": host_cpu_qualification,
            "primary_interactive_selected": interactive_rows,
            "event_ledger": chained_events,
            "event_ledger_digest": stable_hash(chained_events),
        }
        semantics = _seal(semantic_state); measurements = _seal(measurement_state)
        _atomic(root / "SEMANTICS.json", semantics)
        _atomic(root / "MEASUREMENTS.json", measurements)
        leaf_manifest = [
            {"path": str(path.relative_to(root)), "sha256": _sha(path)}
            for path in sorted(root.rglob("*.json"))
        ]
        if _tree_files(root) != _expected_terminal_paths(include_complete=False) \
                or {row["path"] for row in leaf_manifest} != _expected_terminal_paths(
                    include_complete=False,
                ):
            raise SchedulerError("pre-complete exact tree differs")
        complete_state = {
            "schema_version": COMPLETE_SCHEMA, "status": "complete",
            "semantic_digest": semantics["digest"],
            "measurement_digest": measurements["digest"],
            "selected_workers": selection["state"]["selected_workers"],
            "final_lease": final_lease, "semantic_passed": True,
            "performance_selection_only": True,
            "production_promotion_authorized": False,
            "leaf_manifest": leaf_manifest,
            "leaf_manifest_digest": stable_hash(leaf_manifest),
        }
        complete = {**_seal(complete_state), "created_at": _timestamp()}
        if isinstance(backend, ProductionBackend):
            _validate_runtime(runtime, backend.repository)
        if validate_terminal(
            root, complete_payload=complete,
            repository=(backend.repository if isinstance(backend, ProductionBackend) else None),
            production_backend=(backend if isinstance(backend, ProductionBackend) else None),
        ) != complete:
            raise SchedulerError("prepublication terminal reconstruction differs")
        check_run_deadline()
        _atomic(root / "COMPLETE.json", complete)
        return complete
    except BaseException as exc:
        if not (root / "COMPLETE.json").exists():
            try:
                _write_incomplete(root, exc)
            except BaseException:
                pass
        raise


def test_foundation() -> dict[str, Any]:
    return {"mode": "dependency-injected-test", "digest": "0" * 64}


def _environment_binding() -> dict[str, Any]:
    state = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "system": platform.system(), "machine": platform.machine(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "numba", "pyarrow")
        },
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "thread_environment": {key: os.environ.get(key) for key in THREAD_ENV_KEYS},
        "cgroup_cpu_configuration": _cgroup_cpu_snapshot()["configuration"],
    }
    return _seal(state)


RUNTIME_FIXED_FILES = (
    "config/datasets.example.yaml",
    "experiments/m04r/m04r14_exact_scheduler_poc.py",
    "experiments/m04r/verify_m04r14_exact_scheduler_poc.py",
    "experiments/m04r/m04r13_threaded_certified_exposed.py",
    "experiments/m04r/m04r14_evidence_catalog.py",
    "experiments/m04r/m04r14_adversarial_oracle.py",
    "experiments/m04r/compare_m04r11_candidate_matrix_v2.py",
    "experiments/m04r/m04r11_candidate_v2_contract.py",
    "experiments/m04r/m04r11_candidate_matrix_v2.py",
    "experiments/m04r/m04r12_quota_ladder_poc.py",
)


def _runtime_file_set(repository: Path, commit: str | None = None) -> tuple[str, ...]:
    if commit is None:
        command = ["git", "ls-files"]
    else:
        command = ["git", "ls-tree", "-r", "--name-only", commit]
    names = subprocess.run(
        command, cwd=repository, text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    selected = {
        name for name in names if name.startswith("src/market_analogues/")
        and name.endswith(".py")
    }
    selected.update(RUNTIME_FIXED_FILES)
    missing = [
        name for name in RUNTIME_FIXED_FILES
        if name not in names
    ]
    if missing:
        raise SchedulerError(f"mandatory runtime files are untracked: {missing}")
    return tuple(sorted(selected))


def _production_runtime(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repository, text=True, capture_output=True, check=True,
    ).stdout.strip()
    if status:
        raise SchedulerError("production POC requires globally clean Git state")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository,
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    tracked = _runtime_file_set(repository, head)
    files = {
        name: sha256((repository / name).read_bytes()).hexdigest()
        for name in sorted(tracked)
    }
    state = {
        "git_head": head, "files": files, "environment": _environment_binding(),
        "contracts": {
            "schema_version": SCHEMA, "execution_policy": _execution_policy(),
        },
    }
    return _seal(state)


def _validate_runtime(value: Mapping[str, Any], repository: Path) -> None:
    if type(value) is not dict or set(value) != {"state", "digest"} \
            or type(value.get("state")) is not dict \
            or value.get("digest") != stable_hash(value.get("state")):
        raise SchedulerError("runtime seal differs")
    state = value["state"]
    _exact_keys(state, {"git_head", "files", "environment", "contracts"}, "runtime state")
    if type(state["git_head"]) is not str or type(state["files"]) is not dict \
            or state["contracts"] != {
                "schema_version": SCHEMA, "execution_policy": _execution_policy(),
            } \
            or set(state["files"]) != set(_runtime_file_set(repository, state["git_head"])):
        raise SchedulerError("runtime manifest contract differs")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repository, text=True, capture_output=True, check=True,
    ).stdout.strip()
    if status:
        raise SchedulerError("launch requires globally clean Git state")
    current = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository,
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", state["git_head"], current],
        cwd=repository, check=False,
    ).returncode:
        raise SchedulerError("preregistered runtime commit is not an ancestor")
    for name, expected in state["files"].items():
        h0 = subprocess.run(
            ["git", "show", f"{state['git_head']}:{name}"], cwd=repository,
            capture_output=True, check=True,
        ).stdout
        head = subprocess.run(
            ["git", "show", f"{current}:{name}"], cwd=repository,
            capture_output=True, check=True,
        ).stdout
        if sha256(h0).hexdigest() != expected \
                or sha256(head).hexdigest() != expected \
                or _sha(repository / name) != expected:
            raise SchedulerError(f"runtime blob drifted: {name}")
    if state["environment"] != _environment_binding():
        raise SchedulerError("runtime environment drifted")


def _proposal_child(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--source-full-root", type=Path, required=True)
    parser.add_argument("--resident-root", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    repository = Path(__file__).resolve().parents[2]
    request = _exact_keys(_read_json(args.request), {
        "schema_version", "case_ordinal", "task_id", "resident_snapshot",
        "ready_path", "release_path",
    }, "proposal child request")
    ordinal = request["case_ordinal"]
    if request["schema_version"] != "m04r14-proposal-child-request-v1" \
            or type(ordinal) is not int or ordinal not in range(8) \
            or type(request["task_id"]) is not str:
        raise SchedulerError("proposal child request binding differs")
    module = _m13(repository)
    registry = module._read_json(args.registry_root / "query-registry.json")
    all_cases = module._m12(repository)._validate_registry(registry)
    by_query = {str(case["episode_id"]): case for case in all_cases}
    cases = tuple(
        module.CaseInput(index, dict(by_query[query_id]))
        for index, query_id in enumerate(THROUGHPUT_QUERY_IDS)
    )
    inputs = module.Inputs(
        repository.resolve(), args.config.resolve(), args.registry_root.resolve(),
        (args.source_full_root / "store").resolve(), args.resident_root.resolve(),
        Path("/nonexistent/m04r14-proposal-child"), module.GENERATION_ID,
        module.PROVENANCE_DIGEST, module.RESIDENT_RESERVE_BYTES,
        str(registry["registry_digest"]), cases, stable_hash({"schema_version": SCHEMA}),
    )
    resident = request["resident_snapshot"]
    module.validate_resident_snapshot(resident)
    lease = module.lease_exact(inputs, resident)
    ready = {
        "schema_version": "m04r14-proposal-child-ready-v1",
        "task_id": request["task_id"], "pid": os.getpid(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "thread_environment": {
            key: os.environ.get(key) for key in THREAD_ENV_KEYS
        },
        "resident_lease_digest": lease, "ready_monotonic": time.monotonic(),
    }
    _atomic(Path(request["ready_path"]), ready)
    release = Path(request["release_path"])
    deadline = time.monotonic() + CHILD_STARTUP_TIMEOUT_SECONDS
    while not release.is_file():
        if time.monotonic() > deadline:
            raise SchedulerError("proposal child release barrier timed out")
        time.sleep(0.01)
    release_payload = _read_json(release)
    if release_payload.get("schema_version") != "m04r14-proposal-child-release-v1" \
            or release_payload.get("task_id") != request["task_id"] \
            or release_payload.get("pid") != os.getpid() \
            or release_payload.get("ready_digest") != stable_hash(ready):
        raise SchedulerError("proposal child release binding differs")
    prepared = ProductionBackend._prepare_local(
        module, inputs, resident, cases[ordinal], request["task_id"],
    )
    _atomic(args.output, {
        "semantic": _seal(prepared.semantic), "measurement": prepared.measurement,
        "created_at": _timestamp(),
    })
    return 0


def _exact_child(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--source-full-root", type=Path, required=True)
    parser.add_argument("--resident-root", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    repository = Path(__file__).resolve().parents[2]
    request = _read_json(args.request)
    if request.get("schema_version") != "m04r14-exact-child-request-v1" \
            or set(request) != {
                "schema_version", "case_ordinal", "workers", "task_id",
                "case_id", "query_id", "proposal_path", "proposal_sha256",
                "ready_path", "release_path",
            }:
        raise SchedulerError("exact child request differs")
    module = _m13(repository)
    ordinal = request["case_ordinal"]
    if type(ordinal) is not int or ordinal not in range(8):
        raise SchedulerError("exact child case ordinal differs")
    proposal_path = Path(request["proposal_path"])
    if _sha(proposal_path) != request["proposal_sha256"]:
        raise SchedulerError("exact child proposal SHA differs")
    proposal_leaf = _read_json(
        proposal_path, expected_sha256=request["proposal_sha256"],
    )
    if proposal_leaf.get("digest") != stable_hash(proposal_leaf.get("state")):
        raise SchedulerError("exact child proposal seal differs")
    proposal_semantic = proposal_leaf["state"]
    registry = module._read_json(args.registry_root / "query-registry.json")
    all_cases = module._m12(repository)._validate_registry(registry)
    by_query = {str(case["episode_id"]): case for case in all_cases}
    cases = tuple(
        module.CaseInput(index, dict(by_query[query_id]))
        for index, query_id in enumerate(THROUGHPUT_QUERY_IDS)
    )
    inputs = module.Inputs(
        repository.resolve(), args.config.resolve(), args.registry_root.resolve(),
        (args.source_full_root / "store").resolve(), args.resident_root.resolve(),
        Path("/nonexistent/m04r14-exact-child"), module.GENERATION_ID,
        module.PROVENANCE_DIGEST, module.RESIDENT_RESERVE_BYTES,
        str(registry["registry_digest"]), cases, stable_hash({"schema_version": SCHEMA}),
    )
    resident = proposal_semantic["resident_snapshot"]
    module.validate_resident_snapshot(resident)
    module.lease_exact(inputs, resident)
    case = cases[ordinal]
    if request["case_id"] != case.case_id or request["query_id"] != case.query_id:
        raise SchedulerError("exact child case/query request differs")
    source, episode, search_request, packed = module._case_context(inputs, case)
    binding = module.query_binding(
        source, episode, search_request, packed, inputs.provenance_digest,
    )
    if proposal_semantic.get("query_binding") != binding \
            or proposal_semantic.get("query_id") != case.query_id:
        raise SchedulerError("exact child proposal/query binding differs")
    forward = module._proposal_report(proposal_semantic["forward"])
    module.validate_proposal(
        proposal_semantic["forward"], packed, inputs,
    )
    ready = {
        "schema_version": "m04r14-exact-child-ready-v1",
        "task_id": request["task_id"], "pid": os.getpid(),
        "workers": request["workers"], "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "thread_environment": {
            key: os.environ.get(key) for key in THREAD_ENV_KEYS
        },
        "ready_monotonic": time.monotonic(),
        "proposal_sha256": request["proposal_sha256"],
        "resident_lease_digest": module.lease_exact(inputs, resident),
    }
    _atomic(Path(request["ready_path"]), ready)
    if request["release_path"] is not None:
        release = Path(request["release_path"])
        deadline = time.monotonic() + 600
        while not release.is_file():
            if time.monotonic() > deadline:
                raise SchedulerError("child release barrier timed out")
            time.sleep(0.01)
    prepared = PreparedCase(
        request["task_id"], case.case_id, case.query_id,
        proposal_semantic, {},
        (source, episode, search_request, packed, forward, case),
    )
    backend = object.__new__(ProductionBackend)
    backend.module = module; backend.inputs = inputs; backend.resident = resident
    result = backend._exact_local(prepared, request["workers"])
    _atomic(args.output, {
        "semantic": _seal(result.semantic), "measurement": result.measurement,
        "created_at": _timestamp(),
    })
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    selected_argv = list(sys.argv[1:] if argv is None else argv)
    if selected_argv and selected_argv[0] == "_proposal-child":
        return _proposal_child(selected_argv[1:])
    if selected_argv and selected_argv[0] == "_exact-child":
        return _exact_child(selected_argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--source-full-root", type=Path, required=True)
    parser.add_argument("--resident-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregister", action="store_true")
    args = parser.parse_args(selected_argv)
    repository = Path(__file__).resolve().parents[2]
    output = (
        repository / OUTPUT_RELATIVE if args.output_root is None else args.output_root
    ).resolve()
    args.preregistration = args.preregistration.resolve()
    args.config = args.config.resolve()
    args.registry_root = args.registry_root.resolve()
    args.source_full_root = args.source_full_root.resolve()
    args.resident_root = args.resident_root.resolve()
    if output.resolve() != (repository / OUTPUT_RELATIVE).resolve() \
            or args.preregistration.resolve() != (repository / PREREG_RELATIVE).resolve():
        raise SchedulerError("production output/preregistration paths are frozen")
    _validate_topology(
        repository, output, args.preregistration, args.config,
        args.registry_root, args.source_full_root, args.resident_root,
    )
    if args.preregister:
        runtime = _production_runtime(repository)
        backend = ProductionBackend(
            repository=repository, config_path=args.config,
            registry_root=args.registry_root, source_full_root=args.source_full_root,
            resident_root=args.resident_root,
        )
        if output.exists() or output.is_symlink():
            raise SchedulerError("output root must be absent at preregistration")
        preregistration = _preregistration_payload(
            output, backend.foundation(), runtime,
        )
        _atomic(args.preregistration, preregistration)
        print(json.dumps(preregistration, indent=2, sort_keys=True))
        return 0
    preregistration = _read_json(args.preregistration)
    prereg_state = _validate_prereg_launch_envelope(
        preregistration, repository=repository, output=output,
        prereg_path=args.preregistration, config=args.config,
        registry=args.registry_root, source=args.source_full_root,
        resident=args.resident_root,
    )
    runtime = prereg_state["runtime_binding"]
    backend = ProductionBackend(
        repository=repository, config_path=args.config,
        registry_root=args.registry_root, source_full_root=args.source_full_root,
        resident_root=args.resident_root,
    )
    result = execute(
        output, backend, runtime_binding=runtime,
        preregistration=preregistration,
    )
    print(json.dumps({"runtime": runtime, "result": result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
