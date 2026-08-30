"""Preregister and run the resumable T14-08 full-universe shadow snapshot."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter
import traceback
from typing import Any, Mapping, Sequence

from experiments.m04r import m04r11_build_authorities as engine
from experiments.m04r import m04r13_threaded_certified_exposed as m13
from experiments.m04r import m04r14_untouched_candidate_contract as certified
from market_analogues.types import stable_hash


SCHEMA = "m04r14-shadow-snapshot-v1"
PREREG_SCHEMA = "m04r14-shadow-preregistration-v1"
RESULT_SCHEMA = "m04r14-shadow-result-v1"
CONTRACT = Path("config/m04r14-shadow-contract.json")
REGISTRY = Path("config/data/analogues/m04r14/nasdaq-shadow-denominator-v1")
REGISTRY_VERIFICATION = Path(
    "config/data/analogues/m04r14/nasdaq-shadow-denominator-v1-verification"
)
PREREGISTRATION = Path("experiments/m04r/m04r14_shadow_preregistered.json")
OUTPUT = Path("config/data/analogues/m04r14/nasdaq-shadow-snapshot-v1")
CONFIG = Path("config/datasets.example.yaml")
SOURCE_FULL = Path("config/data/analogues/poc/m04r/packed-bound-full")
RUNTIME_FILES = (
    "config/m04r14-shadow-contract.json",
    "experiments/m04r/m04r14_shadow_run.py",
    "experiments/m04r/verify_m04r14_shadow_run.py",
    "experiments/m04r/m04r14_shadow_supervisor.py",
    "experiments/m04r/m04r14_resource_monitored_run.py",
    "experiments/m04r/m04r11_build_authorities.py",
    "experiments/m04r/m04r13_threaded_certified_exposed.py",
)


class ShadowRunError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ShadowRunError(f"regular file required: {path}")
    raw = path.read_bytes()

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ShadowRunError(f"duplicate JSON key: {path}:{key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw, object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ShadowRunError(f"non-finite JSON: {path}:{item}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ShadowRunError(f"invalid JSON: {path}") from exc
    if type(value) is not dict:
        raise ShadowRunError(f"JSON object required: {path}")
    return value, raw


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ShadowRunError(f"regular file required: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ShadowRunError(f"create-only target exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(
                value, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode() + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _git(repository: Path, *args: str, raw: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=repository, capture_output=True,
        text=not raw, check=True,
    )
    return result.stdout if raw else result.stdout.strip()


def _clean_head(repository: Path) -> str:
    if _git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise ShadowRunError("globally clean Git worktree required")
    return str(_git(repository, "rev-parse", "HEAD"))


def _manifest(repository: Path, head: str) -> dict[str, str]:
    tracked = set(str(_git(repository, "ls-tree", "-r", "--name-only", head)).splitlines())
    names = sorted({
        name for name in tracked
        if name.startswith("src/market_analogues/") and name.endswith(".py")
    } | set(RUNTIME_FILES))
    if any(name not in tracked for name in names):
        raise ShadowRunError("runtime manifest contains uncommitted files")
    return {
        name: sha256(_git(repository, "show", f"{head}:{name}", raw=True)).hexdigest()
        for name in names
    }


def _controls(contract: Mapping[str, Any]) -> dict[str, Any]:
    execution = contract["execution"]
    return {
        key: execution[key] for key in (
            "block_rows", "deferred_alignments", "exact_workers_per_process",
            "initial_frontier_rows", "maximum_frontier_rows",
            "numba_threads_per_process", "processes", "requested_positions",
            "seed_rows", "sorted_joined_iqr_merge", "vector_lower_bounds",
        )
    }


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    h0 = _clean_head(repository)
    contract, contract_raw = _read(repository / CONTRACT)
    registry_root = repository / REGISTRY
    registry, registry_raw = _read(registry_root / "query-registry.json")
    registry_seal, registry_seal_raw = _read(registry_root / "SEALED.json")
    verification_path = repository / REGISTRY_VERIFICATION / "VERIFIED.json"
    verification, verification_raw = _read(verification_path)
    cases = registry.get("cases_data")
    if not all((
        registry.get("passed") is True,
        registry.get("scheduled_queries") == 3270,
        registry.get("denominator_symbols") == 12080,
        type(cases) is list and len(cases) == 3270,
        verification.get("passed") is True,
        verification.get("registry_digest") == registry.get("registry_digest"),
        registry_seal.get("registry_digest") == registry.get("registry_digest"),
        registry.get("real_forward_outcomes_accessed") is False,
    )):
        raise ShadowRunError("sealed shadow denominator differs")
    state: dict[str, Any] = {
        "schema_version": PREREG_SCHEMA, "status": "frozen_before_shadow_run",
        "implementation_h0": h0,
        "runtime_files": _manifest(repository, h0),
        "contract_sha256": sha256(contract_raw).hexdigest(), "contract": contract,
        "registry_digest": registry["registry_digest"],
        "registry_sha256": sha256(registry_raw).hexdigest(),
        "registry_seal_sha256": sha256(registry_seal_raw).hexdigest(),
        "registry_verification_sha256": sha256(verification_raw).hexdigest(),
        "registry_verification_result_digest": verification["result_digest"],
        "source_lock_digest": registry["source_lock"]["source_lock_digest"],
        "denominator_symbols": 12080, "scheduled_queries": 3270,
        "query_ids_digest": stable_hash([row["episode_id"] for row in cases]),
        "case_ids_digest": stable_hash([row["case_id"] for row in cases]),
        "audit_sample": registry["audit_sample"],
        "audit_sample_digest": registry["audit_sample_digest"],
        "controls": _controls(contract),
        "roots": {
            "config": str((repository / CONFIG).resolve()),
            "registry": str(registry_root.resolve()),
            "source": str((repository / SOURCE_FULL).resolve()),
            "resident": str(certified.RESIDENT_ROOT.resolve()),
            "output": str((repository / OUTPUT).resolve()),
        },
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "effective_cpus": sorted(os.sched_getaffinity(0)),
        },
        "authority_accessed": False, "real_forward_outcomes_accessed": False,
        "production_promotion_authorized": False,
    }
    return {**state, "preregistration_digest": stable_hash(state)}


def _validate_preregistration(repository: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    h1 = _clean_head(repository)
    prereg, _ = _read(repository / PREREGISTRATION)
    state = {key: value for key, value in prereg.items() if key != "preregistration_digest"}
    if prereg.get("preregistration_digest") != stable_hash(state) \
            or prereg.get("schema_version") != PREREG_SCHEMA:
        raise ShadowRunError("preregistration seal differs")
    h0 = str(prereg.get("implementation_h0"))
    lineage = str(_git(repository, "rev-list", "--parents", "-n", "1", h1)).split()
    changed = str(_git(repository, "diff-tree", "--no-commit-id", "--name-only", "-r", h1)).splitlines()
    if lineage != [h1, h0] or changed != [PREREGISTRATION.as_posix()]:
        raise ShadowRunError("preregistration is not the sole direct-child change")
    for name, expected in prereg.get("runtime_files", {}).items():
        if sha256(_git(repository, "show", f"{h0}:{name}", raw=True)).hexdigest() != expected \
                or sha256(_git(repository, "show", f"{h1}:{name}", raw=True)).hexdigest() != expected \
                or _sha(repository / name) != expected:
            raise ShadowRunError(f"runtime file drifted: {name}")
    contract, contract_raw = _read(repository / CONTRACT)
    if prereg.get("contract_sha256") != sha256(contract_raw).hexdigest() \
            or prereg.get("contract") != contract \
            or prereg.get("controls") != _controls(contract):
        raise ShadowRunError("frozen contract differs")
    registry, registry_raw = _read(repository / REGISTRY / "query-registry.json")
    registry_seal, registry_seal_raw = _read(repository / REGISTRY / "SEALED.json")
    receipt, receipt_raw = _read(repository / REGISTRY_VERIFICATION / "VERIFIED.json")
    expected_roots = {
        "config": str((repository / CONFIG).resolve()),
        "registry": str((repository / REGISTRY).resolve()),
        "source": str((repository / SOURCE_FULL).resolve()),
        "resident": str(certified.RESIDENT_ROOT.resolve()),
        "output": str((repository / OUTPUT).resolve()),
    }
    if any((
        prereg.get("registry_digest") != registry.get("registry_digest"),
        prereg.get("registry_sha256") != sha256(registry_raw).hexdigest(),
        prereg.get("registry_seal_sha256") != sha256(registry_seal_raw).hexdigest(),
        prereg.get("registry_verification_sha256") != sha256(receipt_raw).hexdigest(),
        prereg.get("registry_verification_result_digest") != receipt.get("result_digest"),
        registry_seal.get("registry_digest") != registry.get("registry_digest"),
        receipt.get("registry_digest") != registry.get("registry_digest"),
        prereg.get("source_lock_digest") != registry.get("source_lock", {}).get("source_lock_digest"),
        prereg.get("denominator_symbols") != registry.get("denominator_symbols"),
        prereg.get("scheduled_queries") != registry.get("scheduled_queries"),
        prereg.get("query_ids_digest") != stable_hash([row["episode_id"] for row in registry["cases_data"]]),
        prereg.get("case_ids_digest") != stable_hash([row["case_id"] for row in registry["cases_data"]]),
        prereg.get("audit_sample") != registry.get("audit_sample"),
        prereg.get("audit_sample_digest") != registry.get("audit_sample_digest"),
        prereg.get("roots") != expected_roots,
    )):
        raise ShadowRunError("denominator/preregistration binding differs")
    return prereg, contract, registry


def _execution_contract(prereg: Mapping[str, Any], registry: Mapping[str, Any]) -> dict[str, Any]:
    state = {
        "schema_version": SCHEMA,
        "preregistration_digest": prereg["preregistration_digest"],
        "registry_digest": registry["registry_digest"],
        "generation_id": certified.GENERATION_ID,
        "controls": prereg["controls"],
        "authority_accessed": False, "real_forward_outcomes_accessed": False,
    }
    return {**state, "contract_digest": stable_hash(state)}


def _deterministic_case_digest(row: Mapping[str, Any]) -> str:
    return stable_hash({
        key: value for key, value in row.items() if key not in engine.CASE_RESULT_OMITTED
    })


def _integrity_digest(row: Mapping[str, Any]) -> str:
    return stable_hash({
        key: value for key, value in row.items()
        if key not in {"created_at", "checkpoint_integrity_digest"}
    })


def _case_valid(
    repository: Path, root: Path, case: Mapping[str, Any], execution: Mapping[str, Any],
) -> bool:
    path = root / "cases" / f"{case['episode_id']}.json"
    try:
        row, _ = _read(path)
        if not all((
            row.get("contract_digest") == execution["contract_digest"],
            row.get("registry_digest") == execution["registry_digest"],
            row.get("generation_id") == certified.GENERATION_ID,
            row.get("registry_case_id") == case["case_id"],
            row.get("query_episode_id") == case["episode_id"],
            row.get("query_stock_prefix") == case["stock_prefix"],
            row.get("query_benchmark_prefix") == case["benchmark_prefix"],
            row.get("gate_passed") is True,
            row.get("real_forward_outcomes_accessed") is False,
            type(row.get("matches")) is list and len(row["matches"]) == 20,
            row.get("result_digest") == _deterministic_case_digest(row),
            row.get("checkpoint_integrity_digest") == _integrity_digest(row),
        )):
            return False
        # The production-independent certificate validator checks ordered rows,
        # accounting, stopping and the certified input digest.
        case_input = m13.CaseInput(0, dict(case))
        inputs = m13.Inputs(
            repository, repository / CONFIG, repository / REGISTRY,
            repository / SOURCE_FULL / "store", certified.RESIDENT_ROOT, root,
            certified.GENERATION_ID, certified.PROVENANCE_DIGEST, 1024 ** 3,
            str(execution["registry_digest"]), (case_input,),
            str(execution["preregistration_digest"]),
        )
        source, episode, request, packed = m13._case_context(inputs, case_input)
        binding = m13.query_binding(
            source, episode, request, packed, certified.PROVENANCE_DIGEST,
        )
        certificate = {**row["certificate"], "elapsed_seconds": 0.0}
        m13.validate_certificate_and_matches(
            certificate, row["matches"], str(case["episode_id"]),
            expected_input_digest=binding["certified_input_digest"],
        )
        return True
    except Exception:
        return False


def _worker(
    repository: str, output: str, execution: dict[str, Any],
    cases: tuple[dict[str, Any], ...], controls: dict[str, Any],
) -> dict[str, Any]:
    root = Path(repository)
    return engine._worker(
        str(root / CONFIG), str(root / SOURCE_FULL), output,
        certified.GENERATION_ID, execution, cases, controls,
    )


def _groups(cases: Sequence[dict[str, Any]], count: int) -> list[tuple[dict[str, Any], ...]]:
    ordered = sorted(cases, key=lambda row: (
        -int(row.get("active_source_universe", 0)),
        sha256(str(row["case_id"]).encode()).hexdigest(),
    ))
    values: list[list[dict[str, Any]]] = [[] for _ in range(count)]
    loads = [0] * count
    for row in ordered:
        index = min(range(count), key=lambda item: (loads[item], len(values[item]), item))
        values[index].append(dict(row))
        loads[index] += int(row.get("active_source_universe", 0))
    return [tuple(value) for value in values if value]


def _promote_work(
    repository: Path, work_root: Path, root: Path,
    cases: Mapping[str, dict[str, Any]], execution: Mapping[str, Any],
) -> int:
    promoted = 0
    for path in sorted((work_root / "cases").glob("*.json")):
        case = cases.get(path.stem)
        if case is None:
            raise ShadowRunError(f"work tree contains unknown case: {path.name}")
        temporary_root = work_root
        if not _case_valid(repository, temporary_root, case, execution):
            raise ShadowRunError(f"work tree contains invalid case: {case['case_id']}")
        target = root / "cases" / path.name
        if target.exists():
            if _sha(target) != _sha(path):
                raise ShadowRunError(f"completed case collision: {case['case_id']}")
            path.unlink()
            continue
        os.replace(path, target)
        directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        promoted += 1
    return promoted


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    prereg, contract, registry = _validate_preregistration(repository)
    root = repository / OUTPUT
    execution = _execution_contract(prereg, registry)
    controls = dict(prereg["controls"])
    cases = [dict(row) for row in registry["cases_data"]]
    case_map = {str(row["episode_id"]): row for row in cases}
    if len(case_map) != len(cases):
        raise ShadowRunError("registry contains duplicate query IDs")
    if (root / "RESULT.json").exists():
        raise ShadowRunError("shadow snapshot is already terminal")
    if not root.exists():
        root.mkdir(parents=True)
        (root / "cases").mkdir()
        (root / "work").mkdir()
        (root / "attempts").mkdir()
        resident = m13.resident_full(
            repository / SOURCE_FULL / "store", certified.RESIDENT_ROOT,
            certified.GENERATION_ID, certified.PROVENANCE_DIGEST, 1024 ** 3,
        )
        started_state = {
            "schema_version": RESULT_SCHEMA, "status": "running",
            "preregistration_digest": prereg["preregistration_digest"],
            "contract_digest": execution["contract_digest"],
            "registry_digest": registry["registry_digest"],
            "resident_identity_digest": resident["identity_digest"],
            "source_content_digest": resident["content_digest"],
            "scheduled_queries": len(cases), "created_at": _now(),
            "authority_accessed": False, "real_forward_outcomes_accessed": False,
        }
        _atomic(root / "RUN_STARTED.json", {
            **started_state, "result_digest": stable_hash(started_state),
        })
    started_state, _ = _read(root / "RUN_STARTED.json")
    if started_state.get("preregistration_digest") != prereg["preregistration_digest"] \
            or started_state.get("contract_digest") != execution["contract_digest"]:
        raise ShadowRunError("existing shadow root belongs to another contract")

    # Recover only independently valid atomic files from interrupted worker roots.
    recovered = 0
    for work_root in sorted((root / "work").glob("attempt-*-group-*")):
        recovered += _promote_work(repository, work_root, root, case_map, execution)
    completed: set[str] = set()
    for path in sorted((root / "cases").glob("*.json")):
        case = case_map.get(path.stem)
        if case is None or not _case_valid(repository, root, case, execution):
            raise ShadowRunError(f"existing completed case is invalid: {path.name}")
        completed.add(path.stem)
    remaining = [row for row in cases if row["episode_id"] not in completed]
    attempt_index = len(list((root / "attempts").glob("*-STARTED.json")))
    attempt_id = f"{attempt_index:04d}"
    started = perf_counter()
    _atomic(root / "attempts" / f"{attempt_id}-STARTED.json", {
        "schema_version": RESULT_SCHEMA, "attempt": attempt_index,
        "completed_before": len(completed), "remaining_before": len(remaining),
        "recovered_before": recovered, "created_at": _now(),
    })
    failures: list[dict[str, Any]] = []
    if remaining:
        groups = _groups(remaining, min(int(controls["processes"]), len(remaining)))
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(groups), mp_context=context) as pool:
            futures = {}
            for index, group in enumerate(groups):
                work_root = root / "work" / f"attempt-{attempt_id}-group-{index:02d}"
                work_root.mkdir(parents=True, exist_ok=False)
                (work_root / "cases").mkdir()
                future = pool.submit(
                    _worker, str(repository), str(work_root), execution,
                    group, controls,
                )
                futures[future] = (index, group, work_root)
            for future in as_completed(futures):
                index, group, work_root = futures[future]
                try:
                    worker_result = future.result()
                    failure = None
                except BaseException as exc:
                    worker_result = None
                    failure = {
                        "group_index": index,
                        "case_ids": [row["case_id"] for row in group],
                        "error_type": type(exc).__name__, "message": str(exc),
                        "traceback": "".join(traceback.format_exception(exc)),
                    }
                    failures.append(failure)
                promoted = _promote_work(repository, work_root, root, case_map, execution)
                event = {
                    "schema_version": RESULT_SCHEMA, "attempt": attempt_index,
                    "group_index": index, "assigned_cases": len(group),
                    "promoted_cases": promoted, "worker_result": worker_result,
                    "failure": failure, "created_at": _now(),
                }
                _atomic(root / "attempts" / f"{attempt_id}-GROUP-{index:02d}.json", event)
    elapsed = perf_counter() - started
    completed = {
        path.stem for path in (root / "cases").glob("*.json")
        if path.stem in case_map and _case_valid(repository, root, case_map[path.stem], execution)
    }
    _atomic(root / "attempts" / f"{attempt_id}-COMPLETED.json", {
        "schema_version": RESULT_SCHEMA, "attempt": attempt_index,
        "elapsed_seconds": elapsed, "completed_after": len(completed),
        "remaining_after": len(cases) - len(completed), "worker_failures": failures,
        "created_at": _now(),
    })
    if failures or len(completed) != len(cases):
        return {
            "status": "incomplete", "completed": len(completed),
            "remaining": len(cases) - len(completed), "worker_failures": len(failures),
            "resume_authorized": True,
        }
    resident = m13.resident_full(
        repository / SOURCE_FULL / "store", certified.RESIDENT_ROOT,
        certified.GENERATION_ID, certified.PROVENANCE_DIGEST, 1024 ** 3,
    )
    if resident["identity_digest"] != started_state["resident_identity_digest"] \
            or resident["content_digest"] != started_state["source_content_digest"]:
        raise ShadowRunError("source/resident changed during shadow snapshot")
    paths = sorted((root / "cases").glob("*.json"))
    case_manifest = [{
        "path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
        "sha256": _sha(path),
    } for path in paths]
    attempt_rows = [_read(path)[0] for path in sorted((root / "attempts").glob("*-COMPLETED.json"))]
    active_seconds = sum(float(row["elapsed_seconds"]) for row in attempt_rows)
    throughput = len(cases) / active_seconds
    exact = []
    peak_rss = 0.0
    for path in paths:
        row, _ = _read(path)
        exact.append(float(row["exact_seconds"]))
        peak_rss = max(peak_rss, float(row["peak_rss_mb"]))
    p95 = sorted(exact)[math.ceil(.95 * len(exact)) - 1]
    perf = contract["performance"]
    batch_gates = {
        "active_wall_seconds": active_seconds <= float(perf["snapshot_wall_seconds_max"]),
        "minimum_throughput": throughput >= float(perf["minimum_queries_per_second"]),
        "case_reported_worker_rss": peak_rss <= float(perf["worker_peak_rss_mib_max"]),
        "no_worker_failures_in_terminal_attempts": all(
            not row["worker_failures"] for row in attempt_rows
        ),
    }
    state = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "preregistration_digest": prereg["preregistration_digest"],
        "contract_digest": execution["contract_digest"],
        "registry_digest": registry["registry_digest"],
        "scheduled_queries": len(cases), "completed_queries": len(paths),
        "case_manifest": case_manifest, "case_manifest_digest": stable_hash(case_manifest),
        "attempts": len(attempt_rows), "active_seconds": active_seconds,
        "queries_per_second": throughput, "exact_seconds_p95_descriptive": p95,
        "exact_seconds_max_descriptive": max(exact), "case_reported_peak_rss_mb": peak_rss,
        "batch_gates": batch_gates, "batch_performance_passed": all(batch_gates.values()),
        "semantic_passed": True, "resource_monitor_verification_pending": True,
        "resident_identity_digest": resident["identity_digest"],
        "source_content_digest": resident["content_digest"],
        "authority_accessed": False, "real_forward_outcomes_accessed": False,
        "production_promotion_authorized": False,
    }
    result = {**state, "result_digest": stable_hash(state), "created_at": _now()}
    _atomic(root / "RESULT.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preregister", "run"):
        item = sub.add_parser(name)
        item.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv)
    repository = args.repository.resolve(strict=True)
    if args.command == "preregister":
        value = build_preregistration(repository)
        _atomic(repository / PREREGISTRATION, value)
    else:
        value = execute(repository)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value.get("status") != "incomplete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
