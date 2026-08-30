"""Candidate-first producer for the sealed M04R-14 untouched registry.

This module accepts no authority, comparison, result-opened, or outcome path.
Its ``bind`` and ``preregister`` commands create the two lifecycle-only files;
``run`` is enabled only at the exact clean H0 -> H1 -> H2 Git boundary.
"""
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
from typing import Any, Iterable, Mapping, Sequence

from experiments.m04r import m04r11_build_authorities as engine
from experiments.m04r import m04r13_threaded_certified_exposed as m13
from experiments.m04r import m04r14_untouched_candidate_contract as contract


class CandidateError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise CandidateError(f"regular file required: {path}")
    value = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            value.update(block)
    return value.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise CandidateError(f"duplicate JSON key: {path}:{key}")
            result[key] = value
        return result
    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                CandidateError(f"non-finite JSON: {path}:{item}")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateError(f"invalid JSON: {path}") from exc
    if type(value) is not dict:
        raise CandidateError(f"JSON object required: {path}")
    return value


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise CandidateError(f"create-only target exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(contract.canonical_bytes(value) + b"\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    except BaseException:
        if temporary.exists(): temporary.unlink()
        raise


def _git(repository: Path, *args: str, raw: bool = False) -> str | bytes:
    try:
        result = subprocess.run(["git", *args], cwd=repository, check=True,
            capture_output=True, text=not raw)
    except subprocess.CalledProcessError as exc:
        raise CandidateError(f"Git operation failed: {' '.join(args)}") from exc
    return result.stdout if raw else result.stdout.strip()


def _clean_head(repository: Path) -> str:
    if _git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise CandidateError("globally clean Git worktree required")
    head = _git(repository, "rev-parse", "HEAD")
    if type(head) is not str or len(head) != 40:
        raise CandidateError("Git HEAD differs")
    return head


def _only_child(repository: Path, child: str, parent: str, path: Path) -> None:
    parents = str(_git(repository, "rev-list", "--parents", "-n", "1", child)).split()
    changed = str(_git(repository, "diff-tree", "--no-commit-id", "--name-only", "-r", child)).splitlines()
    if parents != [child, parent] or changed != [path.as_posix()]:
        raise CandidateError(f"{path.name} is not the sole direct-child change")


def _manifest(repository: Path, head: str) -> dict[str, str]:
    tracked = set(str(_git(repository, "ls-tree", "-r", "--name-only", head)).splitlines())
    files = sorted({name for name in tracked if name.startswith("src/market_analogues/")
                    and name.endswith(".py")} | set(contract.RUNTIME_FILES))
    if any(name not in tracked for name in files):
        raise CandidateError("runtime manifest contains untracked files")
    return {name: sha256(_git(repository, "show", f"{head}:{name}", raw=True)).hexdigest()
            for name in files}


def _registry_state(repository: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = repository / contract.REGISTRY_RELATIVE
    verification = repository / contract.REGISTRY_VERIFICATION_RELATIVE / "VERIFIED.json"
    registry = _read(root / "query-registry.json"); receipt = _read(verification)
    cases = registry.get("cases_data")
    if not all((
        registry.get("passed") is True,
        registry.get("schema_version") == "m04r14-untouched-authority-registry-v1",
        registry.get("registry_digest") == receipt.get("registry_digest"),
        receipt.get("status") == "verified", receipt.get("passed") is True,
        receipt.get("verified_symbols") == 36, receipt.get("verified_cases") == 72,
        receipt.get("real_forward_outcomes_accessed") is False,
        type(cases) is list, len(cases) == 72,
    )):
        raise CandidateError("sealed untouched registry/verification differs")
    query_ids = [str(row.get("episode_id")) for row in cases]
    case_ids = [str(row.get("case_id")) for row in cases]
    if len(set(query_ids)) != 72 or len(set(case_ids)) != 72:
        raise CandidateError("registry case identities differ")
    return registry, receipt


def build_binding(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True); h0 = _clean_head(repository)
    registry, receipt = _registry_state(repository)
    root = repository / contract.REGISTRY_RELATIVE
    state = {
        "schema_version": contract.BINDING_SCHEMA,
        "status": "frozen_before_candidate_preregistration",
        "contract_digest": contract.CONTRACT_DIGEST,
        "implementation_h0": h0,
        "runtime_files": _manifest(repository, h0),
        "registry_relative": contract.REGISTRY_RELATIVE.as_posix(),
        "registry_digest": registry["registry_digest"],
        "registry_sha256": _sha(root / "query-registry.json"),
        "registry_seal_sha256": _sha(root / "SEALED.json"),
        "registry_verification_sha256": _sha(
            repository / contract.REGISTRY_VERIFICATION_RELATIVE / "VERIFIED.json"),
        "registry_verification_result_digest": receipt["result_digest"],
        "registry_source_lock_digest": registry["source_lock"]["source_lock_digest"],
        "query_ids": [row["episode_id"] for row in registry["cases_data"]],
        "case_ids": [row["case_id"] for row in registry["cases_data"]],
        "real_forward_outcomes_accessed": False,
    }
    return {**state, "binding_digest": contract.digest(state)}


def validate_binding(repository: Path, binding: Mapping[str, Any], *, head: str) -> dict[str, Any]:
    value = dict(binding); digest = value.pop("binding_digest", None)
    if digest != contract.digest(value) or value.get("schema_version") != contract.BINDING_SCHEMA \
            or value.get("contract_digest") != contract.CONTRACT_DIGEST \
            or value.get("real_forward_outcomes_accessed") is not False:
        raise CandidateError("registry binding seal differs")
    h0 = str(value.get("implementation_h0")); _only_child(
        repository, head, h0, contract.BINDING_RELATIVE)
    registry, receipt = _registry_state(repository); root = repository / contract.REGISTRY_RELATIVE
    expected = {
        "registry_digest": registry["registry_digest"],
        "registry_sha256": _sha(root / "query-registry.json"),
        "registry_seal_sha256": _sha(root / "SEALED.json"),
        "registry_verification_sha256": _sha(
            repository / contract.REGISTRY_VERIFICATION_RELATIVE / "VERIFIED.json"),
        "registry_verification_result_digest": receipt["result_digest"],
        "registry_source_lock_digest": registry["source_lock"]["source_lock_digest"],
        "query_ids": [row["episode_id"] for row in registry["cases_data"]],
        "case_ids": [row["case_id"] for row in registry["cases_data"]],
    }
    if any(value.get(key) != item for key, item in expected.items()) \
            or value.get("runtime_files") != _manifest(repository, h0):
        raise CandidateError("registry binding content differs")
    for name, expected_sha in value["runtime_files"].items():
        if _sha(repository / name) != expected_sha:
            raise CandidateError(f"runtime file drifted: {name}")
    return dict(binding)


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True); h1 = _clean_head(repository)
    binding = validate_binding(repository, _read(repository / contract.BINDING_RELATIVE), head=h1)
    state = {
        "schema_version": contract.PREREGISTRATION_SCHEMA,
        "status": "frozen_before_candidate_run",
        "contract_digest": contract.CONTRACT_DIGEST,
        "implementation_h0": binding["implementation_h0"],
        "registry_binding_h1": h1,
        "registry_binding_digest": binding["binding_digest"],
        "registry_digest": binding["registry_digest"],
        "query_ids": binding["query_ids"], "case_ids": binding["case_ids"],
        "controls": contract.CONTROLS,
        "performance_limits": contract.PERFORMANCE_LIMITS,
        "claims": contract.CLAIMS,
        "roots": {
            "candidate": str((repository / contract.CANDIDATE_RELATIVE).resolve()),
            "config": str((repository / contract.CONFIG_RELATIVE).resolve()),
            "registry": str((repository / contract.REGISTRY_RELATIVE).resolve()),
            "source": str((repository / contract.SOURCE_FULL_RELATIVE).resolve()),
            "resident": str(contract.RESIDENT_ROOT.resolve()),
        },
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "effective_cpus": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else
                list(range(os.cpu_count() or 1)),
        },
        "real_forward_outcomes_accessed": False,
    }
    return {**state, "preregistration_digest": contract.digest(state)}


def validate_launch(repository: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    repository = repository.resolve(strict=True); h2 = _clean_head(repository)
    prereg = _read(repository / contract.PREREGISTRATION_RELATIVE)
    state = {key: value for key, value in prereg.items() if key != "preregistration_digest"}
    if prereg.get("preregistration_digest") != contract.digest(state) \
            or prereg.get("schema_version") != contract.PREREGISTRATION_SCHEMA \
            or prereg.get("contract_digest") != contract.CONTRACT_DIGEST \
            or prereg.get("controls") != contract.CONTROLS \
            or prereg.get("performance_limits") != contract.PERFORMANCE_LIMITS \
            or prereg.get("claims") != contract.CLAIMS:
        raise CandidateError("preregistration differs")
    h1 = str(prereg.get("registry_binding_h1")); _only_child(
        repository, h2, h1, contract.PREREGISTRATION_RELATIVE)
    binding = validate_binding(repository, _read(repository / contract.BINDING_RELATIVE), head=h1)
    if prereg.get("implementation_h0") != binding["implementation_h0"] \
            or prereg.get("registry_binding_digest") != binding["binding_digest"] \
            or prereg.get("registry_digest") != binding["registry_digest"] \
            or prereg.get("query_ids") != binding["query_ids"] \
            or prereg.get("case_ids") != binding["case_ids"]:
        raise CandidateError("preregistration/binding relationship differs")
    for name, expected_sha in binding["runtime_files"].items():
        h2_raw = _git(repository, "show", f"{h2}:{name}", raw=True)
        if sha256(h2_raw).hexdigest() != expected_sha or _sha(repository / name) != expected_sha:
            raise CandidateError(f"runtime H2 file drifted: {name}")
    expected_roots = {
        "candidate": str((repository / contract.CANDIDATE_RELATIVE).resolve()),
        "config": str((repository / contract.CONFIG_RELATIVE).resolve()),
        "registry": str((repository / contract.REGISTRY_RELATIVE).resolve()),
        "source": str((repository / contract.SOURCE_FULL_RELATIVE).resolve()),
        "resident": str(contract.RESIDENT_ROOT.resolve()),
    }
    if prereg.get("roots") != expected_roots:
        raise CandidateError("preregistered roots differ")
    root_text = json.dumps(prereg["roots"]).lower()
    if any(token in root_text for token in contract.FORBIDDEN_PATH_TOKENS):
        raise CandidateError("preregistration contains forbidden truth path")
    registry, _ = _registry_state(repository)
    return prereg, binding, registry


def balanced_groups(cases: Sequence[dict[str, Any]], processes: int) -> tuple[tuple[dict[str, Any], ...], ...]:
    if not 1 <= processes <= len(cases):
        raise CandidateError("process count differs")
    ordered = sorted(cases, key=lambda row: (
        -int(row.get("active_source_universe", 0)), sha256(str(row["case_id"]).encode()).hexdigest()))
    groups: list[list[dict[str, Any]]] = [[] for _ in range(processes)]; loads = [0] * processes
    for row in ordered:
        index = min(range(processes), key=lambda item: (loads[item], len(groups[item]), item))
        groups[index].append(dict(row)); loads[index] += int(row.get("active_source_universe", 0))
    return tuple(tuple(group) for group in groups)


def _run_group(repository: str, output: str, cases: tuple[dict[str, Any], ...],
               registry_digest: str, preregistration_digest: str) -> dict[str, Any]:
    repo = Path(repository)
    state = {"schema_version": contract.RESULT_SCHEMA,
        "registry_digest": registry_digest, "preregistration_digest": preregistration_digest,
        "contract_digest": contract.CONTRACT_DIGEST, "controls": contract.CONTROLS}
    execution = {**state, "contract_digest": contract.digest(state)}
    return engine._worker(str(repo / contract.CONFIG_RELATIVE),
        str(repo / contract.SOURCE_FULL_RELATIVE), output, contract.GENERATION_ID,
        execution, cases, contract.CONTROLS)


def _finite(value: Any) -> bool:
    if isinstance(value, float): return math.isfinite(value)
    if isinstance(value, dict): return all(_finite(item) for item in value.values())
    if isinstance(value, list): return all(_finite(item) for item in value)
    return True


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True); prereg, binding, registry = validate_launch(repository)
    root = repository / contract.CANDIDATE_RELATIVE
    if root.exists() or root.is_symlink():
        raise CandidateError("candidate root exists; retry/resume is forbidden")
    root.mkdir(parents=True); (root / "cases").mkdir()
    started = perf_counter()
    try:
        resident_before = m13.resident_full(repository / contract.SOURCE_FULL_RELATIVE / "store",
            contract.RESIDENT_ROOT, contract.GENERATION_ID, contract.PROVENANCE_DIGEST, 1024 ** 3)
        run_state = {"schema_version": contract.RESULT_SCHEMA, "status": "running",
            "contract_digest": contract.CONTRACT_DIGEST,
            "preregistration_digest": prereg["preregistration_digest"],
            "registry_binding_digest": binding["binding_digest"],
            "registry_digest": registry["registry_digest"],
            "resident_identity_digest": resident_before["identity_digest"],
            "source_content_digest": resident_before["content_digest"],
            "claims": contract.CLAIMS, "created_at": _now()}
        _atomic(root / "RUN_STARTED.json", {**run_state, "result_digest": contract.digest(run_state)})
        cases = [dict(row) for row in registry["cases_data"]]
        groups = balanced_groups(cases, int(contract.CONTROLS["processes"]))
        context = multiprocessing.get_context("spawn"); results: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=len(groups), mp_context=context) as pool:
            futures = {pool.submit(_run_group, str(repository), str(root), group,
                registry["registry_digest"], prereg["preregistration_digest"]): index
                for index, group in enumerate(groups)}
            for future in as_completed(futures):
                results.append({"group_index": futures[future], **future.result()})
        observed = [_read(path) for path in sorted((root / "cases").glob("*.json"))]
        by_query = {str(row.get("query_episode_id")): row for row in observed}
        if len(observed) != 72 or len(by_query) != 72 or set(by_query) != set(binding["query_ids"]):
            raise CandidateError("candidate case inventory differs")
        for expected in registry["cases_data"]:
            row = by_query[str(expected["episode_id"])]
            if not all((row.get("gate_passed") is True, row.get("real_forward_outcomes_accessed") is False,
                    row.get("registry_case_id") == expected["case_id"],
                    row.get("query_stock_prefix") == expected["stock_prefix"],
                    row.get("query_benchmark_prefix") == expected["benchmark_prefix"],
                    type(row.get("matches")) is list, len(row["matches"]) == 20,
                    type(row.get("certificate")) is dict, _finite(row))):
                raise CandidateError(f"candidate case gate differs: {expected['case_id']}")
        resident_after = m13.resident_full(repository / contract.SOURCE_FULL_RELATIVE / "store",
            contract.RESIDENT_ROOT, contract.GENERATION_ID, contract.PROVENANCE_DIGEST, 1024 ** 3)
        if resident_after["identity_digest"] != resident_before["identity_digest"] \
                or resident_after["content_digest"] != resident_before["content_digest"]:
            raise CandidateError("resident/source changed during candidate run")
        exact = [float(row["exact_seconds"]) for row in observed]; wall = perf_counter() - started
        p95 = sorted(exact)[math.ceil(.95 * len(exact)) - 1]; limits = contract.PERFORMANCE_LIMITS
        performance_passed = wall <= limits["candidate_wall_seconds_max"] \
            and p95 <= limits["case_exact_seconds_p95"] \
            and max(exact) <= limits["case_exact_seconds_max"]
        case_manifest = [{"path": path.relative_to(root).as_posix(), "sha256": _sha(path),
                          "bytes": path.stat().st_size}
            for path in sorted((root / "cases").glob("*.json"))]
        state = {"schema_version": contract.RESULT_SCHEMA, "status": "complete",
            "contract_digest": contract.CONTRACT_DIGEST,
            "preregistration_digest": prereg["preregistration_digest"],
            "registry_binding_digest": binding["binding_digest"],
            "registry_digest": registry["registry_digest"], "cases": 72, "symbols": 36,
            "controls": contract.CONTROLS, "groups": sorted(results, key=lambda row: row["group_index"]),
            "case_manifest": case_manifest, "case_manifest_digest": contract.digest(case_manifest),
            "wall_seconds": wall, "exact_seconds_p95": p95, "exact_seconds_max": max(exact),
            "semantic_passed": True, "performance_passed": performance_passed,
            "resident_identity_digest": resident_after["identity_digest"],
            "source_content_digest": resident_after["content_digest"],
            "authority_accessed": False, "real_forward_outcomes_accessed": False,
            "production_promotion_authorized": False}
        result = {**state, "result_digest": contract.digest(state), "created_at": _now()}
        _atomic(root / "RESULT.json", result)
        return result
    except BaseException as exc:
        if not (root / "RESULT.json").exists() and not (root / "FAILED.json").exists():
            _atomic(root / "FAILED.json", {"schema_version": contract.RESULT_SCHEMA,
                "status": "failed", "error_type": type(exc).__name__, "message": str(exc),
                "resume_authorized": False, "authority_open_authorized": False,
                "created_at": _now()})
        raise


def _publish_lifecycle(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); _atomic(path, value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    for name in ("bind", "preregister", "run"):
        item = sub.add_parser(name); item.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv); repository = args.repository.resolve(strict=True)
    if args.command == "bind":
        value = build_binding(repository); _publish_lifecycle(repository / contract.BINDING_RELATIVE, value)
    elif args.command == "preregister":
        value = build_preregistration(repository); _publish_lifecycle(
            repository / contract.PREREGISTRATION_RELATIVE, value)
    else:
        value = execute(repository)
    print(json.dumps(value, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
