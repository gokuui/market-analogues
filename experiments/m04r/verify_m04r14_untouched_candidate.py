"""Independent pre-open verifier for the M04R-14 untouched candidate run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from experiments.m04r import m04r13_threaded_certified_exposed as m13
from experiments.m04r import m04r14_untouched_candidate_contract as contract
from market_analogues.types import stable_hash


class VerificationError(RuntimeError):
    pass


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file(): raise VerificationError(f"regular file required: {path}")
    raw = path.read_bytes()
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in values:
            if key in result: raise VerificationError(f"duplicate JSON key: {path}:{key}")
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(VerificationError(f"non-finite: {item}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc: raise VerificationError(f"invalid JSON: {path}") from exc
    if type(value) is not dict: raise VerificationError(f"object required: {path}")
    return value, raw


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _git(repository: Path, *args: str, raw: bool = False) -> str | bytes:
    try:
        result = subprocess.run(["git", *args], cwd=repository, check=True,
            capture_output=True, text=not raw)
    except subprocess.CalledProcessError as exc: raise VerificationError("Git lifecycle differs") from exc
    return result.stdout if raw else result.stdout.strip()


def _only_child(repository: Path, child: str, parent: str, path: Path) -> None:
    lineage = str(_git(repository, "rev-list", "--parents", "-n", "1", child)).split()
    changed = str(_git(repository, "diff-tree", "--no-commit-id", "--name-only", "-r", child)).splitlines()
    if lineage != [child, parent] or changed != [path.as_posix()]:
        raise VerificationError("H0/H1/H2 lifecycle differs")


def _deterministic_case_digest(row: Mapping[str, Any]) -> str:
    omitted = {"created_at", "proposal_seconds", "amortized_proposal_seconds",
        "exact_seconds", "final_exact_seconds", "frontier_attempt_measurements",
        "peak_rss_mb", "result_digest", "checkpoint_integrity_digest"}
    return stable_hash({key: value for key, value in row.items() if key not in omitted})


def _integrity_digest(row: Mapping[str, Any]) -> str:
    return stable_hash({key: value for key, value in row.items()
        if key not in {"created_at", "checkpoint_integrity_digest"}})


def verify(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    prereg, prereg_raw = _read(repository / contract.PREREGISTRATION_RELATIVE)
    binding, binding_raw = _read(repository / contract.BINDING_RELATIVE)
    pstate = {key: value for key, value in prereg.items() if key != "preregistration_digest"}
    bstate = {key: value for key, value in binding.items() if key != "binding_digest"}
    if prereg.get("preregistration_digest") != contract.digest(pstate) \
            or binding.get("binding_digest") != contract.digest(bstate) \
            or prereg.get("contract_digest") != contract.CONTRACT_DIGEST \
            or binding.get("contract_digest") != contract.CONTRACT_DIGEST:
        raise VerificationError("lifecycle payload seal differs")
    h0 = str(binding.get("implementation_h0")); h1 = str(prereg.get("registry_binding_h1"))
    h2 = str(_git(repository, "rev-parse", "HEAD"))
    _only_child(repository, h1, h0, contract.BINDING_RELATIVE)
    _only_child(repository, h2, h1, contract.PREREGISTRATION_RELATIVE)
    if _git(repository, "status", "--porcelain"):
        raise VerificationError("worktree is not clean")
    files = binding.get("runtime_files")
    if type(files) is not dict or not files: raise VerificationError("runtime manifest differs")
    for name, expected in files.items():
        h0_raw = _git(repository, "show", f"{h0}:{name}", raw=True)
        h2_raw = _git(repository, "show", f"{h2}:{name}", raw=True)
        if sha256(h0_raw).hexdigest() != expected or sha256(h2_raw).hexdigest() != expected \
                or _sha(repository / name) != expected:
            raise VerificationError(f"runtime file drifted: {name}")
    registry_root = repository / contract.REGISTRY_RELATIVE
    registry, _ = _read(registry_root / "query-registry.json")
    receipt, _ = _read(repository / contract.REGISTRY_VERIFICATION_RELATIVE / "VERIFIED.json")
    if registry.get("registry_digest") != binding.get("registry_digest") \
            or receipt.get("registry_digest") != binding.get("registry_digest") \
            or _sha(registry_root / "query-registry.json") != binding.get("registry_sha256") \
            or _sha(registry_root / "SEALED.json") != binding.get("registry_seal_sha256"):
        raise VerificationError("registry binding differs")
    root = repository / contract.CANDIDATE_RELATIVE
    names = {path.name for path in root.iterdir()}
    if names != {"RUN_STARTED.json", "RESULT.json", "cases"} or any(path.is_symlink() for path in root.rglob("*")):
        raise VerificationError("candidate tree differs")
    started, _ = _read(root / "RUN_STARTED.json"); result, result_raw = _read(root / "RESULT.json")
    deterministic = {key: value for key, value in result.items() if key not in {"result_digest", "created_at"}}
    if result.get("result_digest") != contract.digest(deterministic) \
            or result.get("status") != "complete" or result.get("semantic_passed") is not True \
            or result.get("performance_passed") is not True \
            or result.get("authority_accessed") is not False \
            or result.get("real_forward_outcomes_accessed") is not False \
            or result.get("production_promotion_authorized") is not False \
            or started.get("preregistration_digest") != prereg.get("preregistration_digest"):
        raise VerificationError("candidate aggregate differs")
    paths = sorted((root / "cases").glob("*.json"))
    manifest = [{"path": path.relative_to(root).as_posix(), "sha256": _sha(path), "bytes": path.stat().st_size}
        for path in paths]
    if len(paths) != 72 or manifest != result.get("case_manifest") \
            or contract.digest(manifest) != result.get("case_manifest_digest"):
        raise VerificationError("candidate manifest differs")
    cases = [m13.CaseInput(index, dict(row)) for index, row in enumerate(registry["cases_data"])]
    inputs = m13.Inputs(repository, repository / contract.CONFIG_RELATIVE, registry_root,
        repository / contract.SOURCE_FULL_RELATIVE / "store", contract.RESIDENT_ROOT, root,
        contract.GENERATION_ID, contract.PROVENANCE_DIGEST, 1024 ** 3,
        str(registry["registry_digest"]), tuple(cases), str(prereg["preregistration_digest"]))
    by_query: dict[str, dict[str, Any]] = {}
    for path in paths:
        row, _ = _read(path); query_id = str(row.get("query_episode_id"))
        if query_id in by_query: raise VerificationError("duplicate candidate query")
        by_query[query_id] = row
    exact: list[float] = []
    for case in cases:
        row = by_query.get(case.query_id); expected = case.registry_case
        if row is None or row.get("registry_case_id") != case.case_id \
                or row.get("query_stock_prefix") != expected["stock_prefix"] \
                or row.get("query_benchmark_prefix") != expected["benchmark_prefix"] \
                or row.get("gate_passed") is not True or len(row.get("matches", [])) != 20 \
                or row.get("result_digest") != _deterministic_case_digest(row) \
                or row.get("checkpoint_integrity_digest") != _integrity_digest(row):
            raise VerificationError(f"candidate case differs: {case.case_id}")
        source, episode, request, packed = m13._case_context(inputs, case)
        query_binding = m13.query_binding(source, episode, request, packed, contract.PROVENANCE_DIGEST)
        certificate = {**row["certificate"], "elapsed_seconds": 0.0}
        m13.validate_certificate_and_matches(certificate, row["matches"], case.query_id,
            expected_input_digest=query_binding["certified_input_digest"])
        exact.append(float(row["exact_seconds"]))
    limits = contract.PERFORMANCE_LIMITS; p95 = sorted(exact)[math.ceil(.95 * 72) - 1]
    if float(result["wall_seconds"]) > limits["candidate_wall_seconds_max"] \
            or p95 != float(result["exact_seconds_p95"]) or max(exact) != float(result["exact_seconds_max"]) \
            or p95 > limits["case_exact_seconds_p95"] or max(exact) > limits["case_exact_seconds_max"]:
        raise VerificationError("candidate performance differs")
    resident = m13.resident_full(repository / contract.SOURCE_FULL_RELATIVE / "store",
        contract.RESIDENT_ROOT, contract.GENERATION_ID, contract.PROVENANCE_DIGEST, 1024 ** 3)
    if resident["identity_digest"] != result.get("resident_identity_digest") \
            or resident["content_digest"] != result.get("source_content_digest"):
        raise VerificationError("final source/resident lease differs")
    state = {"schema_version": contract.VERIFICATION_SCHEMA, "status": "verified", "passed": True,
        "candidate_result_digest": result["result_digest"], "candidate_result_sha256": sha256(result_raw).hexdigest(),
        "preregistration_sha256": sha256(prereg_raw).hexdigest(), "binding_sha256": sha256(binding_raw).hexdigest(),
        "registry_digest": registry["registry_digest"], "verified_cases": 72, "verified_matches": 1440,
        "exact_seconds_p95": p95, "exact_seconds_max": max(exact),
        "authority_accessed": False, "real_forward_outcomes_accessed": False,
        "results_open_authorized": True, "production_promotion_authorized": False}
    return {**state, "result_digest": contract.digest(state)}


def _publish(path: Path, state: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink(): raise VerificationError("verification root exists")
    path.mkdir(parents=False); descriptor = os.open(path / "VERIFIED.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json.dumps({**state, "created_at": datetime.now(timezone.utc).isoformat()},
            indent=2, sort_keys=True).encode() + b"\n"); handle.flush(); os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true"); args = parser.parse_args(argv)
    repository = args.repository.resolve(strict=True); state = verify(repository)
    if not args.dry_run: _publish(repository / contract.CANDIDATE_VERIFICATION_RELATIVE, state)
    print(json.dumps(state, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
