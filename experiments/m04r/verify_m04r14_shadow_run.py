"""Independently verify a completed T14-08 full-universe shadow snapshot."""
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

from experiments.m04r import m04r11_build_authorities as engine
from experiments.m04r import m04r13_threaded_certified_exposed as m13
from experiments.m04r import m04r14_untouched_candidate_contract as certified
from market_analogues.types import stable_hash


SCHEMA = "m04r14-shadow-snapshot-verification-v1"
PREREG_SCHEMA = "m04r14-shadow-preregistration-v1"
RESULT_SCHEMA = "m04r14-shadow-result-v1"
CONTRACT = Path("config/m04r14-shadow-contract.json")
REGISTRY = Path("config/data/analogues/m04r14/nasdaq-shadow-denominator-v1")
REGISTRY_VERIFICATION = Path("config/data/analogues/m04r14/nasdaq-shadow-denominator-v1-verification")
PREREGISTRATION = Path("experiments/m04r/m04r14_shadow_preregistered.json")
OUTPUT = Path("config/data/analogues/m04r14/nasdaq-shadow-snapshot-v1")
VERIFICATION = Path("config/data/analogues/m04r14/nasdaq-shadow-snapshot-v1-verification")
CONFIG = Path("config/datasets.example.yaml")
SOURCE_FULL = Path("config/data/analogues/poc/m04r/packed-bound-full")


class ShadowRunVerificationError(RuntimeError):
    pass


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ShadowRunVerificationError(f"regular file required: {path}")
    raw = path.read_bytes()

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ShadowRunVerificationError(f"duplicate JSON key: {path}:{key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw, object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ShadowRunVerificationError(f"non-finite JSON: {path}:{item}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ShadowRunVerificationError(f"invalid JSON: {path}") from exc
    if type(value) is not dict:
        raise ShadowRunVerificationError(f"JSON object required: {path}")
    return value, raw


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _git(repository: Path, *args: str, raw: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=repository, capture_output=True,
        text=not raw, check=True,
    )
    return result.stdout if raw else result.stdout.strip()


def _deterministic_case_digest(row: Mapping[str, Any]) -> str:
    return stable_hash({
        key: value for key, value in row.items() if key not in engine.CASE_RESULT_OMITTED
    })


def _integrity_digest(row: Mapping[str, Any]) -> str:
    return stable_hash({
        key: value for key, value in row.items()
        if key not in {"created_at", "checkpoint_integrity_digest"}
    })


def _execution_contract(prereg: Mapping[str, Any], registry: Mapping[str, Any]) -> dict[str, Any]:
    state = {
        "schema_version": "m04r14-shadow-snapshot-v1",
        "preregistration_digest": prereg["preregistration_digest"],
        "registry_digest": registry["registry_digest"],
        "generation_id": certified.GENERATION_ID,
        "controls": prereg["controls"],
        "authority_accessed": False, "real_forward_outcomes_accessed": False,
    }
    return {**state, "contract_digest": stable_hash(state)}


def verify(repository: Path, root: Path | None = None) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    root = (root or repository / OUTPUT).resolve(strict=True)
    prereg, prereg_raw = _read(repository / PREREGISTRATION)
    pstate = {key: value for key, value in prereg.items() if key != "preregistration_digest"}
    if prereg.get("schema_version") != PREREG_SCHEMA \
            or prereg.get("preregistration_digest") != stable_hash(pstate):
        raise ShadowRunVerificationError("preregistration seal differs")
    h1 = str(_git(repository, "rev-parse", "HEAD"))
    h0 = str(prereg.get("implementation_h0"))
    lineage = str(_git(repository, "rev-list", "--parents", "-n", "1", h1)).split()
    changed = str(_git(repository, "diff-tree", "--no-commit-id", "--name-only", "-r", h1)).splitlines()
    if lineage != [h1, h0] or changed != [PREREGISTRATION.as_posix()] \
            or _git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise ShadowRunVerificationError("H0/H1 lifecycle or worktree differs")
    for name, expected in prereg.get("runtime_files", {}).items():
        if sha256(_git(repository, "show", f"{h0}:{name}", raw=True)).hexdigest() != expected \
                or sha256(_git(repository, "show", f"{h1}:{name}", raw=True)).hexdigest() != expected \
                or _sha(repository / name) != expected:
            raise ShadowRunVerificationError(f"runtime source drifted: {name}")
    contract, contract_raw = _read(repository / CONTRACT)
    registry, registry_raw = _read(repository / REGISTRY / "query-registry.json")
    registry_seal, registry_seal_raw = _read(repository / REGISTRY / "SEALED.json")
    receipt, receipt_raw = _read(repository / REGISTRY_VERIFICATION / "VERIFIED.json")
    cases = registry.get("cases_data")
    if not all((
        prereg.get("contract_sha256") == sha256(contract_raw).hexdigest(),
        prereg.get("contract") == contract,
        prereg.get("registry_digest") == registry.get("registry_digest"),
        prereg.get("registry_sha256") == sha256(registry_raw).hexdigest(),
        prereg.get("registry_seal_sha256") == sha256(registry_seal_raw).hexdigest(),
        prereg.get("registry_verification_sha256") == sha256(receipt_raw).hexdigest(),
        prereg.get("registry_verification_result_digest") == receipt.get("result_digest"),
        type(cases) is list and len(cases) == 3270,
        prereg.get("query_ids_digest") == stable_hash([row["episode_id"] for row in cases]),
        prereg.get("case_ids_digest") == stable_hash([row["case_id"] for row in cases]),
        prereg.get("audit_sample") == registry.get("audit_sample"),
        prereg.get("audit_sample_digest") == registry.get("audit_sample_digest"),
    )):
        raise ShadowRunVerificationError("registry/preregistration binding differs")
    expected_names = {"RUN_STARTED.json", "RESULT.json", "cases", "work", "attempts"}
    if {path.name for path in root.iterdir()} != expected_names \
            or any(path.is_symlink() for path in root.rglob("*")):
        raise ShadowRunVerificationError("shadow result tree differs")
    for work_root in sorted((root / "work").iterdir()):
        if not work_root.is_dir() or set(path.name for path in work_root.iterdir()) != {"cases"} \
                or any((work_root / "cases").iterdir()):
            raise ShadowRunVerificationError("terminal work tree is not empty and canonical")
    started, _ = _read(root / "RUN_STARTED.json")
    result, result_raw = _read(root / "RESULT.json")
    execution = _execution_contract(prereg, registry)
    start_state = {key: value for key, value in started.items() if key != "result_digest"}
    if started.get("result_digest") != stable_hash(start_state) \
            or started.get("preregistration_digest") != prereg["preregistration_digest"] \
            or started.get("contract_digest") != execution["contract_digest"]:
        raise ShadowRunVerificationError("run-start seal differs")
    deterministic = {key: value for key, value in result.items() if key not in {"result_digest", "created_at"}}
    if result.get("result_digest") != stable_hash(deterministic) \
            or result.get("status") != "complete" \
            or result.get("semantic_passed") is not True \
            or result.get("authority_accessed") is not False \
            or result.get("real_forward_outcomes_accessed") is not False \
            or result.get("production_promotion_authorized") is not False:
        raise ShadowRunVerificationError("terminal result seal differs")
    paths = sorted((root / "cases").glob("*.json"))
    if len(paths) != 3270 or any(path.is_symlink() for path in paths):
        raise ShadowRunVerificationError("case inventory differs")
    manifest = [{
        "path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
        "sha256": _sha(path),
    } for path in paths]
    if result.get("case_manifest") != manifest \
            or result.get("case_manifest_digest") != stable_hash(manifest):
        raise ShadowRunVerificationError("case manifest differs")
    by_query: dict[str, tuple[dict[str, Any], bytes]] = {}
    for path in paths:
        row, raw = _read(path)
        query_id = str(row.get("query_episode_id"))
        if query_id in by_query:
            raise ShadowRunVerificationError("duplicate query result")
        by_query[query_id] = (row, raw)
    exact: list[float] = []
    peak_rss = 0.0
    verified_matches = 0
    inputs = m13.Inputs(
        repository, repository / CONFIG, repository / REGISTRY,
        repository / SOURCE_FULL / "store", certified.RESIDENT_ROOT, root,
        certified.GENERATION_ID, certified.PROVENANCE_DIGEST, 1024 ** 3,
        str(registry["registry_digest"]), (), str(prereg["preregistration_digest"]),
    )
    for ordinal, case in enumerate(cases):
        found = by_query.get(str(case["episode_id"]))
        if found is None:
            raise ShadowRunVerificationError(f"missing case: {case['case_id']}")
        row, _ = found
        if not all((
            row.get("contract_digest") == execution["contract_digest"],
            row.get("registry_digest") == registry["registry_digest"],
            row.get("generation_id") == certified.GENERATION_ID,
            row.get("registry_case_id") == case["case_id"],
            row.get("query_stock_prefix") == case["stock_prefix"],
            row.get("query_benchmark_prefix") == case["benchmark_prefix"],
            row.get("gate_passed") is True,
            row.get("real_forward_outcomes_accessed") is False,
            type(row.get("matches")) is list and len(row["matches"]) == 20,
            row.get("result_digest") == _deterministic_case_digest(row),
            row.get("checkpoint_integrity_digest") == _integrity_digest(row),
        )):
            raise ShadowRunVerificationError(f"case seal differs: {case['case_id']}")
        case_input = m13.CaseInput(ordinal, dict(case))
        case_inputs = m13.Inputs(
            inputs.repository, inputs.config_path, inputs.registry_root,
            inputs.source_store_root, inputs.resident_root, inputs.output_root,
            inputs.generation_id, inputs.provenance_digest, inputs.reserve_bytes,
            inputs.registry_digest, (case_input,), inputs.prereg_digest,
        )
        source, episode, request, packed = m13._case_context(case_inputs, case_input)
        binding = m13.query_binding(source, episode, request, packed, certified.PROVENANCE_DIGEST)
        certificate = {**row["certificate"], "elapsed_seconds": 0.0}
        m13.validate_certificate_and_matches(
            certificate, row["matches"], str(case["episode_id"]),
            expected_input_digest=binding["certified_input_digest"],
        )
        exact.append(float(row["exact_seconds"]))
        peak_rss = max(peak_rss, float(row["peak_rss_mb"]))
        verified_matches += len(row["matches"])
    attempt_paths = sorted((root / "attempts").glob("*-COMPLETED.json"))
    attempts = [_read(path)[0] for path in attempt_paths]
    started_paths = sorted((root / "attempts").glob("*-STARTED.json"))
    if len(started_paths) != len(attempt_paths) \
            or not attempts or [row["attempt"] for row in attempts] != list(range(len(attempts))) \
            or any(row.get("worker_failures") for row in attempts):
        raise ShadowRunVerificationError("attempt ledger differs or records worker failure")
    active_seconds = sum(float(row["elapsed_seconds"]) for row in attempts)
    throughput = len(cases) / active_seconds
    p95 = sorted(exact)[math.ceil(.95 * len(exact)) - 1]
    performance = contract["performance"]
    gates = {
        "active_wall_seconds": active_seconds <= float(performance["snapshot_wall_seconds_max"]),
        "minimum_throughput": throughput >= float(performance["minimum_queries_per_second"]),
        "case_reported_worker_rss": peak_rss <= float(performance["worker_peak_rss_mib_max"]),
        "no_worker_failures_in_terminal_attempts": True,
    }
    if result.get("batch_gates") != gates \
            or result.get("batch_performance_passed") is not all(gates.values()) \
            or float(result.get("active_seconds")) != active_seconds \
            or float(result.get("queries_per_second")) != throughput \
            or float(result.get("exact_seconds_p95_descriptive")) != p95 \
            or float(result.get("exact_seconds_max_descriptive")) != max(exact) \
            or float(result.get("case_reported_peak_rss_mb")) != peak_rss:
        raise ShadowRunVerificationError("batch metrics/gates differ")
    resident = m13.resident_full(
        repository / SOURCE_FULL / "store", certified.RESIDENT_ROOT,
        certified.GENERATION_ID, certified.PROVENANCE_DIGEST, 1024 ** 3,
    )
    if resident["identity_digest"] != result.get("resident_identity_digest") \
            or resident["content_digest"] != result.get("source_content_digest"):
        raise ShadowRunVerificationError("source/resident lease differs")
    state = {
        "schema_version": SCHEMA, "status": "verified", "passed": True,
        "shadow_result_digest": result["result_digest"],
        "shadow_result_sha256": sha256(result_raw).hexdigest(),
        "preregistration_sha256": sha256(prereg_raw).hexdigest(),
        "registry_digest": registry["registry_digest"],
        "verified_cases": len(cases), "verified_matches": verified_matches,
        "active_seconds": active_seconds, "queries_per_second": throughput,
        "exact_seconds_p95_descriptive": p95,
        "exact_seconds_max_descriptive": max(exact),
        "case_reported_peak_rss_mb": peak_rss,
        "batch_performance_passed": all(gates.values()),
        "resource_monitor_verification_pending": True,
        "audit_sample_verification_pending": True,
        "real_forward_outcomes_accessed": False,
        "production_promotion_authorized": False,
    }
    return {**state, "result_digest": stable_hash(state)}


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ShadowRunVerificationError("verification root exists")
    path.mkdir(parents=False)
    descriptor = os.open(path / "VERIFIED.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json.dumps({
            **value, "created_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--shadow-root", type=Path)
    parser.add_argument("--verification-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    repository = args.repository.resolve(strict=True)
    value = verify(repository, args.shadow_root)
    if not args.dry_run:
        _publish(args.verification_root or repository / VERIFICATION, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
