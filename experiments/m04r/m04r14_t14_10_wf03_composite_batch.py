"""Preregister and run all 3,936 true seven-group WF-03 searches."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import fcntl
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path
import resource
import stat
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence

from market_analogues.distance import DistanceConfig
from market_analogues.types import stable_hash

from experiments.m04r import m04r14_t14_10_wf03_composite_topology_poc as kernel
from experiments.m04r import m04r14_t14_10_wf03_composite_width_poc as width
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import verify_m04r14_t14_10_wf03_composite_topology_poc as kernel_verifier


SCHEMA = "m04r14-t14-10-wf03-composite-batch-preregistration-v1"
OUTPUT_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf03-composite-batch-v1"
)
PREREGISTRATION_RELATIVE = Path(
    "experiments/m04r/m04r14_t14_10_wf03_composite_batch_v1_preregistered.json"
)
WIDTH_VERIFICATION_RELATIVE = Path(
    "config/data/analogues/m04r14/"
    "t14-10-wf03-composite-width-poc-v1-verification/VERIFIED.json"
)
PROCESS_COUNT = 12
THREADS_PER_PROCESS = 1
EXPECTED_QUERIES = 3_936
EXPECTED_SCORED_QUERIES = 3_360
EXPECTED_WARMUP_QUERIES = 576
WIDTH_VERIFICATION_DIGEST = (
    "062dbc955cf68a71b775577754739dfebf5cfe6fe230206a3677790e3c2aae8c"
)
WIDTH_PRODUCER_DIGEST = (
    "5664dac25cd42c3c7fff3ccf7076c5e6067f7c15c6516d552aec0a407bb97f2c"
)
_REPOSITORY = Path(__file__).resolve().parents[2]
RUNTIME_FILES = (
    "config/datasets.example.yaml",
    "experiments/m04r/m04r14_t14_10_wf03_composite_batch.py",
    "experiments/m04r/m04r14_t14_10_wf03_composite_topology_poc.py",
    "experiments/m04r/m04r14_t14_10_wf03_composite_width_poc.py",
    "experiments/m04r/m04r14_t14_10_wf03_feasibility.py",
    "pyproject.toml",
) + tuple(
    str(path.relative_to(_REPOSITORY))
    for path in sorted((_REPOSITORY / "src/market_analogues").rglob("*.py"))
)


class CompositeBatchError(RuntimeError):
    pass


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repository, text=True,
        capture_output=True, check=False,
    )
    if result.returncode:
        raise CompositeBatchError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _is_digest(value: Any) -> bool:
    return type(value) is str and len(value) == 64 \
        and set(value).issubset("0123456789abcdef")


def _is_attempt_id(value: Any) -> bool:
    if type(value) is not str or not value.startswith("attempt-"):
        return False
    try:
        ordinal = int(value.removeprefix("attempt-"))
    except ValueError:
        return False
    return ordinal >= 1 and value == f"attempt-{ordinal:04d}"


def _resource_observation() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, separator, rest = line.partition(":")
        if separator and name in {"MemTotal", "MemAvailable"}:
            values[name] = int(rest.split()[0])
    effective_cpus = len(os.sched_getaffinity(0)) \
        if hasattr(os, "sched_getaffinity") else int(os.cpu_count() or 0)
    if set(values) != {"MemTotal", "MemAvailable"}:
        raise CompositeBatchError("host resource observation differs")
    return {
        "effective_cpus": effective_cpus,
        "memory_total_kib": values["MemTotal"],
        "memory_available_kib": values["MemAvailable"],
        "parent_peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def _require_resources() -> dict[str, int]:
    observation = _resource_observation()
    if observation["effective_cpus"] < PROCESS_COUNT:
        raise CompositeBatchError(
            f"requires {PROCESS_COUNT} effective CPUs, observed "
            f"{observation['effective_cpus']}"
        )
    return observation


def _replace_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _width_verification(repository: Path) -> dict[str, Any]:
    value = base._read(repository / WIDTH_VERIFICATION_RELATIVE)
    base._validate_seal(value, "verification_digest")
    if not all((
        value.get("schema_version")
            == "m04r14-t14-10-wf03-composite-width-verification-v1",
        value.get("status") == "complete",
        value.get("passed") is True,
        value.get("queries_verified") == 12,
        value.get("certificates_verified") == 24,
        value.get("links_verified") == 240,
        value.get("selected_topology") == "p12t1",
        value.get("verification_digest") == WIDTH_VERIFICATION_DIGEST,
        value.get("producer_result_digest") == WIDTH_PRODUCER_DIGEST,
        value.get("outcomes_or_labels_used") is False,
        value.get("historical_walk_forward_query_outcomes_opened") is False,
        value.get("final_period_result_opened") is False,
        value.get("production_promotion_authorized") is False,
    )):
        raise CompositeBatchError("verified width prerequisite differs")
    return value


def _inventory(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "queries": len(rows),
        "scored_queries": sum(row.get("scored") is True for row in rows),
        "warmup_queries": sum(row.get("scored") is False for row in rows),
        "months": len({row.get("cutoff") for row in rows}),
    }


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain"):
        raise CompositeBatchError("composite batch preregistration requires clean commit")
    output = repository / OUTPUT_RELATIVE
    if output.exists() or output.is_symlink():
        raise CompositeBatchError("composite batch output must be absent before freeze")
    registry, _by_id = base._registry(repository)
    rows = registry["queries_data"]
    inventory = _inventory(rows)
    if inventory["queries"] != EXPECTED_QUERIES \
            or inventory["scored_queries"] != EXPECTED_SCORED_QUERIES \
            or inventory["warmup_queries"] != EXPECTED_WARMUP_QUERIES:
        raise CompositeBatchError("composite batch inventory differs")
    verified_width = _width_verification(repository)
    resident = base._resident()
    state = {
        "schema_version": SCHEMA,
        "status": "frozen_before_all_query_true_composite_retrieval",
        "implementation_commit": _git(repository, "rev-parse", "HEAD"),
        "runtime_files": {path: _sha(repository / path) for path in RUNTIME_FILES},
        "inputs": {
            "registry_digest": registry["registry_digest"],
            "registry_sha256": _sha(repository / base.REGISTRY_FILE),
            "query_ids_digest": stable_hash([row["episode_id"] for row in rows]),
            "packed_generation_id": base.GENERATION_ID,
            "packed_provenance_digest": base.PROVENANCE_DIGEST,
            "resident_content_digest": resident["content_digest"],
            "width_verification_digest": verified_width["verification_digest"],
            "width_verification_sha256": _sha(
                repository / WIDTH_VERIFICATION_RELATIVE
            ),
            "width_producer_result_digest": verified_width["producer_result_digest"],
        },
        "inventory": inventory,
        "query_rows": rows,
        "contracts": {
            "retrieval": kernel._contract(),
            "distance_weights": DistanceConfig().weights,
        },
        "execution": {
            "processes": PROCESS_COUNT,
            "threads_per_process": THREADS_PER_PROCESS,
            "total_numerical_threads": PROCESS_COUNT * THREADS_PER_PROCESS,
            "start_method": "spawn",
            "scheduler": "rolling bounded queue, refill after each completed future",
            "worker_batch_queries": 1,
            "case_publication": "parent-validated create-only sealed JSON",
            "progress_publication": "atomic after every completed query",
            "attempt_publication": "create-only start and terminal receipts",
            "resume": "validate every receipt; schedule only absent query IDs",
            "output_root": str(output.resolve()),
        },
        "gates": {
            "all_query_certificates_close": True,
            "twenty_distinct_symbols_per_query": True,
            "all_seven_component_groups_present": True,
            "strict_causal_retrieval": True,
            "zero_worker_swap": True,
            "outcomes_or_labels_excluded": True,
            "independent_verification_required": True,
        },
        "claims": {
            "historical_query_retrieval_opened": True,
            "historical_walk_forward_query_outcomes_opened": False,
            "final_period_result_opened": False,
            "production_promotion_authorized": False,
        },
    }
    return base._sealed(state, "preregistration_digest")


def validate_preregistration(
    repository: Path, value: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if _git(repository, "status", "--porcelain"):
        raise CompositeBatchError("composite batch execution requires clean commit")
    base._validate_seal(value, "preregistration_digest")
    registry, _by_id = base._registry(repository)
    rows = registry["queries_data"]
    verified_width = _width_verification(repository)
    resident = base._resident()
    inputs = value.get("inputs", {})
    execution = value.get("execution", {})
    if not all((
        value.get("schema_version") == SCHEMA,
        value.get("status") == "frozen_before_all_query_true_composite_retrieval",
        value.get("query_rows") == rows,
        value.get("inventory") == _inventory(rows),
        value.get("inventory", {}).get("queries") == EXPECTED_QUERIES,
        value.get("inventory", {}).get("scored_queries") == EXPECTED_SCORED_QUERIES,
        value.get("inventory", {}).get("warmup_queries") == EXPECTED_WARMUP_QUERIES,
        value.get("contracts", {}).get("retrieval") == kernel._contract(),
        value.get("contracts", {}).get("distance_weights") == DistanceConfig().weights,
        inputs.get("registry_digest") == registry["registry_digest"],
        inputs.get("registry_sha256") == _sha(repository / base.REGISTRY_FILE),
        inputs.get("query_ids_digest")
            == stable_hash([row["episode_id"] for row in rows]),
        inputs.get("packed_generation_id") == base.GENERATION_ID,
        inputs.get("packed_provenance_digest") == base.PROVENANCE_DIGEST,
        inputs.get("resident_content_digest") == resident["content_digest"],
        inputs.get("width_verification_digest") == verified_width["verification_digest"],
        inputs.get("width_verification_sha256")
            == _sha(repository / WIDTH_VERIFICATION_RELATIVE),
        inputs.get("width_producer_result_digest") == WIDTH_PRODUCER_DIGEST,
        execution.get("processes") == PROCESS_COUNT,
        execution.get("threads_per_process") == THREADS_PER_PROCESS,
        execution.get("total_numerical_threads") == PROCESS_COUNT,
        execution.get("start_method") == "spawn",
        execution.get("worker_batch_queries") == 1,
        execution.get("output_root") == str((repository / OUTPUT_RELATIVE).resolve()),
        set(value.get("runtime_files", {})) == set(RUNTIME_FILES),
        value.get("claims") == {
            "historical_query_retrieval_opened": True,
            "historical_walk_forward_query_outcomes_opened": False,
            "final_period_result_opened": False,
            "production_promotion_authorized": False,
        },
    )):
        raise CompositeBatchError("composite batch preregistration differs")
    commit = value.get("implementation_commit")
    if type(commit) is not str:
        raise CompositeBatchError("composite batch implementation commit differs")
    _git(repository, "merge-base", "--is-ancestor", commit, "HEAD")
    for path, digest in value["runtime_files"].items():
        blob = subprocess.run(
            ["git", "show", f"{commit}:{path}"], cwd=repository,
            capture_output=True, check=False,
        )
        if blob.returncode or sha256(blob.stdout).hexdigest() != digest \
                or _sha(repository / path) != digest:
            raise CompositeBatchError(f"composite batch runtime differs: {path}")
    return registry, rows, resident


def _case_path(cases_root: Path, query_id: str) -> Path:
    if type(query_id) is not str or len(query_id) != 24 \
            or not set(query_id).issubset("0123456789abcdef"):
        raise CompositeBatchError("composite batch query ID differs")
    return cases_root / f"{query_id}.json"


def _case_semantics(value: Mapping[str, Any]) -> dict[str, Any]:
    return kernel._case_semantics(value["retrieval"])


def _validate_case(
    value: Mapping[str, Any], row: Mapping[str, Any],
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        base._validate_seal(value, "case_digest")
        retrieval = value["retrieval"]
        measurement = value["worker_measurement"]
        kernel_verifier.validate_certificate(retrieval)
        valid = all((
            value["schema_version"] == "m04r14-wf03-composite-batch-case-v1",
            value["status"] == "complete",
            value["query_id"] == row["episode_id"],
            value["case_id"] == row["case_id"],
            value["symbol"] == row["symbol"],
            value["cutoff"] == row["cutoff"],
            value["fold_id"] == row["fold_id"],
            value["fold_role"] == row["fold_role"],
            value["scored"] is bool(row["scored"]),
            value["preregistration_digest"] == preregistration["preregistration_digest"],
            value["packed_generation_id"] == base.GENERATION_ID,
            value["resident_content_digest"]
                == preregistration["inputs"]["resident_content_digest"],
            _is_attempt_id(value["attempt_id"]),
            _is_digest(value["resident_identity_digest"]),
            retrieval["query_id"] == row["episode_id"],
            retrieval["certificate"]["query_episode_id"] == row["episode_id"],
            retrieval["certificate"]["generation_id"] == base.GENERATION_ID,
            retrieval["certificate"]["contract_digest"]
                == preregistration["contracts"]["retrieval"]["digest"],
            retrieval["semantic_digest"] == stable_hash(kernel._case_semantics(retrieval)),
            value["semantic_digest"] == stable_hash(_case_semantics(value)),
            measurement["queries"] == 1,
            measurement["threads"] == THREADS_PER_PROCESS,
            measurement["swap_kib"] == 0,
            measurement["resident_identity_digest"] == value["resident_identity_digest"],
            type(measurement["elapsed_seconds"]) in (int, float),
            measurement["elapsed_seconds"] >= 0,
            type(measurement["proposal_seconds"]) in (int, float),
            measurement["proposal_seconds"] >= 0,
            type(measurement["peak_rss_mb"]) in (int, float),
            measurement["peak_rss_mb"] > 0,
            value["outcomes_or_labels_used"] is False,
            value["historical_walk_forward_query_outcomes_opened"] is False,
            value["final_period_result_opened"] is False,
        ))
    except (KeyError, TypeError, ValueError, base.FeasibilityError):
        valid = False
    if not valid:
        raise CompositeBatchError(f"composite batch case differs: {row['episode_id']}")
    return dict(value)


def _existing_case(
    path: Path, row: Mapping[str, Any], preregistration: Mapping[str, Any],
) -> dict[str, Any] | None:
    if path.is_symlink():
        raise CompositeBatchError("composite batch case is linked")
    if not path.exists():
        return None
    if not path.is_file():
        raise CompositeBatchError("composite batch case is not regular")
    return _validate_case(base._read(path), row, preregistration)


def _validate_worker_result(
    worker: Mapping[str, Any], row: Mapping[str, Any], resident: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        valid = all((
            worker["queries"] == 1,
            worker["threads"] == THREADS_PER_PROCESS,
            worker["swap_kib"] == 0,
            worker["resident_identity_digest"] == resident["identity_digest"],
            len(worker["cases"]) == 1,
            worker["cases"][0]["query_id"] == row["episode_id"],
        ))
    except (KeyError, TypeError):
        valid = False
    if not valid:
        raise CompositeBatchError(f"composite worker result differs: {row['episode_id']}")
    retrieval = dict(worker["cases"][0])
    kernel_verifier.validate_certificate(retrieval)
    measurement = {
        key: worker[key] for key in (
            "queries", "threads", "proposal_seconds", "elapsed_seconds",
            "peak_rss_mb", "swap_kib", "resident_identity_digest",
        )
    }
    return retrieval, measurement


def _publish_worker_result(
    worker: Mapping[str, Any], row: Mapping[str, Any], resident: Mapping[str, Any],
    attempt_id: str, preregistration: Mapping[str, Any], cases_root: Path,
) -> dict[str, Any]:
    retrieval, measurement = _validate_worker_result(worker, row, resident)
    state = {
        "schema_version": "m04r14-wf03-composite-batch-case-v1",
        "status": "complete", "query_id": row["episode_id"],
        "case_id": row["case_id"], "symbol": row["symbol"],
        "cutoff": row["cutoff"], "fold_id": row["fold_id"],
        "fold_role": row["fold_role"], "scored": bool(row["scored"]),
        "preregistration_digest": preregistration["preregistration_digest"],
        "packed_generation_id": base.GENERATION_ID,
        "resident_content_digest": resident["content_digest"],
        "resident_identity_digest": resident["identity_digest"],
        "attempt_id": attempt_id, "retrieval": retrieval,
        "worker_measurement": measurement,
        "semantic_digest": stable_hash(kernel._case_semantics(retrieval)),
        "outcomes_or_labels_used": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
    }
    sealed = base._sealed(state, "case_digest")
    _validate_case(sealed, row, preregistration)
    base._atomic(_case_path(cases_root, row["episode_id"]), sealed)
    return sealed


def _next_attempt(root: Path) -> Path:
    attempts = root / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    if attempts.is_symlink():
        raise CompositeBatchError("composite batch attempts root is linked")
    ordinals = []
    for path in attempts.iterdir():
        if path.is_symlink() or not path.is_dir() or not _is_attempt_id(path.name):
            raise CompositeBatchError("composite batch attempt layout differs")
        ordinals.append(int(path.name.removeprefix("attempt-")))
    target = attempts / f"attempt-{max(ordinals, default=0) + 1:04d}"
    target.mkdir()
    return target


def _attempt_history(
    root: Path, preregistration: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    attempts = root / "attempts"
    if not attempts.exists():
        return {}
    if attempts.is_symlink() or not attempts.is_dir():
        raise CompositeBatchError("composite batch attempts root differs")
    output = {}
    for path in sorted(attempts.iterdir()):
        if path.is_symlink() or not path.is_dir() or not _is_attempt_id(path.name):
            raise CompositeBatchError("composite batch attempt layout differs")
        names = {item.name for item in path.iterdir() if not item.name.startswith(".wf03-")}
        if "RUN_STARTED.json" not in names \
                or names - {"RUN_STARTED.json", "INTERRUPTED.json", "COMPLETE.json"} \
                or {"INTERRUPTED.json", "COMPLETE.json"}.issubset(names):
            raise CompositeBatchError("composite batch attempt files differ")
        started = base._read(path / "RUN_STARTED.json")
        base._validate_seal(started, "attempt_digest")
        if not all((
            started.get("schema_version") == "m04r14-wf03-composite-batch-attempt-v1",
            started.get("status") == "running",
            started.get("attempt_id") == path.name,
            started.get("preregistration_digest") == preregistration["preregistration_digest"],
            started.get("resident_content_digest")
                == preregistration["inputs"]["resident_content_digest"],
            _is_digest(started.get("resident_identity_digest")),
            type(started.get("receipts_reused_at_start")) is int,
            0 <= started.get("receipts_reused_at_start", -1) <= EXPECTED_QUERIES,
            type(started.get("resource_observation")) is dict,
            started.get("resource_observation", {}).get("effective_cpus", 0)
                >= PROCESS_COUNT,
            type(started.get("created_at")) is str,
        )):
            raise CompositeBatchError("composite batch attempt start differs")
        terminal_name = next(iter(names & {"INTERRUPTED.json", "COMPLETE.json"}), None)
        terminal = None
        if terminal_name is not None:
            terminal = base._read(path / terminal_name)
            base._validate_seal(terminal, "attempt_digest")
            expected_status = "complete" if terminal_name == "COMPLETE.json" else "interrupted"
            if not all((
                terminal.get("schema_version") == "m04r14-wf03-composite-batch-attempt-v1",
                terminal.get("status") == expected_status,
                terminal.get("attempt_id") == path.name,
                type(terminal.get("created_at")) is str,
                type(terminal.get("completed_this_attempt")) is int,
                0 <= terminal.get("completed_this_attempt", -1) <= EXPECTED_QUERIES,
                type(terminal.get("completed_total")) is int,
                0 <= terminal.get("completed_total", -1) <= EXPECTED_QUERIES,
            )):
                raise CompositeBatchError("composite batch attempt terminal differs")
            if terminal_name == "COMPLETE.json" and not all((
                terminal.get("completed_total") == EXPECTED_QUERIES,
                type(terminal.get("receipts_reused_at_start")) is int,
                terminal.get("receipts_reused_at_start", -1) >= 0,
                terminal.get("receipts_reused_at_start", 0)
                    + terminal.get("completed_this_attempt", 0) == EXPECTED_QUERIES,
                _is_digest(terminal.get("result_digest")),
            )):
                raise CompositeBatchError("composite batch completion differs")
            if terminal_name == "INTERRUPTED.json" and not all((
                type(terminal.get("error_type")) is str,
                type(terminal.get("error")) is str,
            )):
                raise CompositeBatchError("composite batch interruption differs")
        output[path.name] = {
            **started, "_terminal_name": terminal_name, "_terminal": terminal,
        }
    return output


def _validate_case_attempt(
    value: Mapping[str, Any], attempts: Mapping[str, Mapping[str, Any]],
) -> None:
    attempt = attempts.get(str(value.get("attempt_id")))
    if attempt is None or value.get("resident_identity_digest") \
            != attempt.get("resident_identity_digest"):
        raise CompositeBatchError("composite batch case attempt binding differs")


def _manifest(
    rows: Sequence[Mapping[str, Any]], results_by_id: Mapping[str, Mapping[str, Any]],
    cases_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = []
    manifest = []
    for row in rows:
        value = results_by_id.get(row["episode_id"])
        if value is None:
            raise CompositeBatchError("composite batch result is incomplete")
        ordered.append(dict(value))
        manifest.append({
            "query_id": value["query_id"], "case_digest": value["case_digest"],
            "sha256": _sha(_case_path(cases_root, value["query_id"])),
        })
    return ordered, manifest


def _build_result(
    rows: Sequence[Mapping[str, Any]], results_by_id: Mapping[str, Mapping[str, Any]],
    cases_root: Path, preregistration: Mapping[str, Any], attempt_id: str,
    attempts: int, started: float, resources: Mapping[str, Any],
) -> dict[str, Any]:
    ordered, manifest = _manifest(rows, results_by_id, cases_root)
    state = {
        "schema_version": "m04r14-t14-10-wf03-composite-batch-result-v1",
        "status": "complete", "passed": True,
        "queries": len(ordered),
        "scored_queries": sum(value["scored"] for value in ordered),
        "warmup_queries": sum(not value["scored"] for value in ordered),
        "months": preregistration["inventory"]["months"],
        "preregistration_digest": preregistration["preregistration_digest"],
        "packed_generation_id": base.GENERATION_ID,
        "resident_content_digest": preregistration["inputs"]["resident_content_digest"],
        "case_manifest_digest": stable_hash(manifest),
        "case_semantic_digest": stable_hash([
            value["semantic_digest"] for value in ordered
        ]),
        "minimum_eligible_candidates": min(
            value["retrieval"]["certificate"]["eligible_candidates"]
            for value in ordered
        ),
        "maximum_eligible_candidates": max(
            value["retrieval"]["certificate"]["eligible_candidates"]
            for value in ordered
        ),
        "maximum_worker_peak_rss_mb": max(
            value["worker_measurement"]["peak_rss_mb"] for value in ordered
        ),
        "all_worker_swap_zero": all(
            value["worker_measurement"]["swap_kib"] == 0 for value in ordered
        ),
        "terminal_attempt_id": attempt_id, "attempts": attempts,
        "resource_observation": dict(resources),
        "elapsed_seconds": perf_counter() - started,
        "outcomes_or_labels_used": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
        "independent_verification_authorized": True,
    }
    return base._sealed(state)


def _validate_terminal(
    value: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
    results_by_id: Mapping[str, Mapping[str, Any]], cases_root: Path,
    preregistration: Mapping[str, Any], attempts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        base._validate_seal(value)
        ordered, manifest = _manifest(rows, results_by_id, cases_root)
        valid = all((
            value["schema_version"] == "m04r14-t14-10-wf03-composite-batch-result-v1",
            value["status"] == "complete", value["passed"] is True,
            value["queries"] == EXPECTED_QUERIES,
            value["scored_queries"] == EXPECTED_SCORED_QUERIES,
            value["warmup_queries"] == EXPECTED_WARMUP_QUERIES,
            value["months"] == preregistration["inventory"]["months"],
            value["preregistration_digest"] == preregistration["preregistration_digest"],
            value["packed_generation_id"] == base.GENERATION_ID,
            value["resident_content_digest"]
                == preregistration["inputs"]["resident_content_digest"],
            value["case_manifest_digest"] == stable_hash(manifest),
            value["case_semantic_digest"]
                == stable_hash([item["semantic_digest"] for item in ordered]),
            value["all_worker_swap_zero"] is True,
            value["terminal_attempt_id"] in attempts,
            value["attempts"] == len(attempts),
            value["outcomes_or_labels_used"] is False,
            value["historical_walk_forward_query_outcomes_opened"] is False,
            value["final_period_result_opened"] is False,
            value["production_promotion_authorized"] is False,
            value["independent_verification_authorized"] is True,
        ))
    except (KeyError, TypeError, ValueError, base.FeasibilityError):
        valid = False
    if not valid:
        raise CompositeBatchError("composite batch terminal result differs")
    return dict(value)


def _reconcile_terminal(
    root: Path, result: Mapping[str, Any], attempts: Mapping[str, Mapping[str, Any]],
) -> None:
    attempt_id = str(result["terminal_attempt_id"])
    attempt = attempts[attempt_id]
    terminal_name = attempt.get("_terminal_name")
    if terminal_name == "INTERRUPTED.json":
        raise CompositeBatchError("composite batch result attempt was interrupted")
    if terminal_name == "COMPLETE.json":
        if attempt["_terminal"].get("result_digest") != result["result_digest"]:
            raise CompositeBatchError("composite batch completion result differs")
        return
    reused = int(attempt["receipts_reused_at_start"])
    base._atomic(root / "attempts" / attempt_id / "COMPLETE.json", base._sealed({
        "schema_version": "m04r14-wf03-composite-batch-attempt-v1",
        "status": "complete", "attempt_id": attempt_id,
        "completed_this_attempt": EXPECTED_QUERIES - reused,
        "completed_total": EXPECTED_QUERIES,
        "receipts_reused_at_start": reused,
        "result_digest": result["result_digest"], "created_at": base._now(),
    }, "attempt_digest"))


def _progress(
    root: Path, *, status: str, attempt_id: str, completed: int,
    total: int, started: float, last: Mapping[str, Any] | None = None,
    error: BaseException | None = None,
) -> None:
    state: dict[str, Any] = {
        "schema_version": "m04r14-wf03-composite-batch-progress-v1",
        "status": status, "attempt_id": attempt_id,
        "completed_queries": completed, "total_queries": total,
        "remaining_queries": total - completed,
        "elapsed_seconds": perf_counter() - started,
        "updated_at": base._now(),
    }
    if last is not None:
        state.update({
            "last_query_id": last["query_id"],
            "last_case_digest": last["case_digest"],
        })
    if error is not None:
        state.update({"error_type": type(error).__name__, "error": str(error)})
    _replace_json(root / "PROGRESS.json", state)


def _run_pending(
    repository: Path, resident: Mapping[str, Any], pending: Sequence[Mapping[str, Any]],
    results_by_id: dict[str, dict[str, Any]], cases_root: Path,
    attempt_id: str, preregistration: Mapping[str, Any], root: Path, started: float,
) -> int:
    if not pending:
        return 0
    context = multiprocessing.get_context("spawn")
    completed_this_attempt = 0
    iterator = iter(pending)
    with ProcessPoolExecutor(
        max_workers=PROCESS_COUNT, mp_context=context,
    ) as executor:
        active = {}
        for _ in range(min(PROCESS_COUNT, len(pending))):
            row = next(iterator)
            future = executor.submit(
                kernel._run_group, str(repository), resident["store_root"],
                (dict(row),), THREADS_PER_PROCESS,
            )
            active[future] = row
        while active:
            done, _not_done = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                row = active.pop(future)
                worker = future.result()
                value = _publish_worker_result(
                    worker, row, resident, attempt_id, preregistration, cases_root,
                )
                results_by_id[value["query_id"]] = value
                completed_this_attempt += 1
                _progress(
                    root, status="running", attempt_id=attempt_id,
                    completed=len(results_by_id), total=EXPECTED_QUERIES,
                    started=started, last=value,
                )
                print(
                    f"[composite-batch] completed={len(results_by_id)}/"
                    f"{EXPECTED_QUERIES} query={value['query_id']} "
                    f"seconds={value['worker_measurement']['elapsed_seconds']:.2f}",
                    flush=True,
                )
                try:
                    next_row = next(iterator)
                except StopIteration:
                    continue
                next_future = executor.submit(
                    kernel._run_group, str(repository), resident["store_root"],
                    (dict(next_row),), THREADS_PER_PROCESS,
                )
                active[next_future] = next_row
    return completed_this_attempt


def _execute_locked(
    repository: Path, preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    started = perf_counter()
    _registry, rows, resident = validate_preregistration(repository, preregistration)
    resources = _require_resources()
    root = repository / OUTPUT_RELATIVE
    if root.is_symlink() or root.exists() and not root.is_dir():
        raise CompositeBatchError("composite batch output path differs")
    if not root.exists():
        root.mkdir(parents=True)
        base._atomic(root / "CONTRACT.json", preregistration)
    elif base._read(root / "CONTRACT.json") != preregistration:
        raise CompositeBatchError("composite batch resume contract differs")
    cases_root = root / "cases"
    cases_root.mkdir(exist_ok=True)
    if cases_root.is_symlink():
        raise CompositeBatchError("composite batch cases root is linked")
    expected_names = {f"{row['episode_id']}.json" for row in rows}
    unexpected = {
        path.name for path in cases_root.iterdir()
        if not path.name.startswith(".wf03-") and path.name not in expected_names
    }
    if unexpected:
        raise CompositeBatchError("composite batch cases layout differs")
    attempts = _attempt_history(root, preregistration)
    results_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        existing = _existing_case(
            _case_path(cases_root, row["episode_id"]), row, preregistration,
        )
        if existing is not None:
            _validate_case_attempt(existing, attempts)
            results_by_id[row["episode_id"]] = existing
    if (root / "RESULT.json").exists():
        result = _validate_terminal(
            base._read(root / "RESULT.json"), rows, results_by_id, cases_root,
            preregistration, attempts,
        )
        _reconcile_terminal(root, result, attempts)
        _progress(
            root, status="complete", attempt_id=result["terminal_attempt_id"],
            completed=len(results_by_id), total=EXPECTED_QUERIES, started=started,
        )
        return result
    pending = [row for row in rows if row["episode_id"] not in results_by_id]
    attempt = _next_attempt(root)
    attempt_id = attempt.name
    base._atomic(attempt / "RUN_STARTED.json", base._sealed({
        "schema_version": "m04r14-wf03-composite-batch-attempt-v1",
        "status": "running", "attempt_id": attempt_id,
        "preregistration_digest": preregistration["preregistration_digest"],
        "resident_content_digest": resident["content_digest"],
        "resident_identity_digest": resident["identity_digest"],
        "receipts_reused_at_start": len(results_by_id),
        "resource_observation": resources, "created_at": base._now(),
    }, "attempt_digest"))
    completed_this_attempt = 0
    try:
        completed_this_attempt = _run_pending(
            repository, resident, pending, results_by_id, cases_root,
            attempt_id, preregistration, root, started,
        )
    except BaseException as exc:
        completed_this_attempt = len(results_by_id) - len(rows) + len(pending)
        _progress(
            root, status="interrupted", attempt_id=attempt_id,
            completed=len(results_by_id), total=EXPECTED_QUERIES,
            started=started, error=exc,
        )
        base._atomic(attempt / "INTERRUPTED.json", base._sealed({
            "schema_version": "m04r14-wf03-composite-batch-attempt-v1",
            "status": "interrupted", "attempt_id": attempt_id,
            "completed_this_attempt": completed_this_attempt,
            "completed_total": len(results_by_id),
            "error_type": type(exc).__name__, "error": str(exc),
            "created_at": base._now(),
        }, "attempt_digest"))
        raise
    result = _build_result(
        rows, results_by_id, cases_root, preregistration, attempt_id,
        len(attempts) + 1, started, resources,
    )
    base._atomic(root / "RESULT.json", result)
    base._atomic(attempt / "COMPLETE.json", base._sealed({
        "schema_version": "m04r14-wf03-composite-batch-attempt-v1",
        "status": "complete", "attempt_id": attempt_id,
        "completed_this_attempt": completed_this_attempt,
        "completed_total": len(results_by_id),
        "receipts_reused_at_start": len(results_by_id) - completed_this_attempt,
        "result_digest": result["result_digest"], "created_at": base._now(),
    }, "attempt_digest"))
    _progress(
        root, status="complete", attempt_id=attempt_id,
        completed=len(results_by_id), total=EXPECTED_QUERIES,
        started=started,
    )
    return result


def execute(repository: Path, preregistration: Mapping[str, Any]) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    output = repository / OUTPUT_RELATIVE
    lock_path = output.with_name(f"{output.name}.lock")
    if lock_path.is_symlink():
        raise CompositeBatchError("composite batch lock is linked")
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise CompositeBatchError("composite batch lock is not regular")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CompositeBatchError("another composite batch is running") from exc
        try:
            return _execute_locked(repository, preregistration)
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preregister", "run"))
    parser.add_argument("--repository", required=True, type=Path)
    args = parser.parse_args(argv)
    repository = args.repository.resolve(strict=True)
    path = repository / PREREGISTRATION_RELATIVE
    if args.mode == "preregister":
        base._atomic(path, build_preregistration(repository))
        return 0
    result = execute(repository, base._read(path))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
