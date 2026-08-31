"""Verify T14-08 semantics after a host interruption prevented terminal publication.

This verifier never manufactures the producer's missing RESULT.json.  It binds the
complete case tree to the frozen preregistration and immutable source content,
validates every certified result, accounts for the dangling/recovery attempts,
and permanently rejects uninterrupted performance/resource qualification.
"""
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
from experiments.m04r import m04r14_shadow_run as producer
from experiments.m04r import m04r14_untouched_candidate_contract as certified
from experiments.m04r import verify_m04r14_shadow_run as base
from market_analogues import resident_store
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.types import stable_hash


SCHEMA = "m04r14-shadow-interrupted-semantic-verification-v1"
OUTPUT = Path("config/data/analogues/m04r14/nasdaq-shadow-snapshot-v1")
VERIFICATION = Path(
    "config/data/analogues/m04r14/nasdaq-shadow-interrupted-verification-v1"
)
REATTACHED = Path(
    "config/data/analogues/m04r14/nasdaq-shadow-reattached-supervision-v1"
)
RESUMED = Path(
    "config/data/analogues/m04r14/nasdaq-shadow-resume-supervision-v1"
)


class InterruptedVerificationError(RuntimeError):
    pass


def _git(repository: Path, *args: str, raw: bool = False, check: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=repository, capture_output=True,
        text=not raw, check=check,
    )
    return result.stdout if raw else result.stdout.strip()


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise InterruptedVerificationError(f"regular file required: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _time(value: Any, field: str) -> datetime:
    if type(value) is not str:
        raise InterruptedVerificationError(f"timestamp required: {field}")
    try:
        result = datetime.fromisoformat(value)
    except ValueError as exc:
        raise InterruptedVerificationError(f"invalid timestamp: {field}") from exc
    if result.tzinfo is None:
        raise InterruptedVerificationError(f"timezone required: {field}")
    return result.astimezone(timezone.utc)


def _sole_preregistration_commit(
    repository: Path, h0: str, preregistration_raw: bytes,
) -> str:
    rows = str(_git(repository, "rev-list", "--all", "--children")).splitlines()
    children: list[str] = []
    for row in rows:
        values = row.split()
        if values and values[0] == h0:
            children.extend(values[1:])
    accepted: list[str] = []
    for child in sorted(set(children)):
        lineage = str(_git(repository, "rev-list", "--parents", "-n", "1", child)).split()
        changed = str(_git(
            repository, "diff-tree", "--no-commit-id", "--name-only", "-r", child,
        )).splitlines()
        blob = _git(
            repository, "show", f"{child}:{base.PREREGISTRATION.as_posix()}",
            raw=True, check=False,
        )
        if lineage == [child, h0] and changed == [base.PREREGISTRATION.as_posix()] \
                and blob == preregistration_raw:
            accepted.append(child)
    if len(accepted) != 1:
        raise InterruptedVerificationError(
            f"expected one exact preregistration child; found {len(accepted)}"
        )
    h1 = accepted[0]
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", h1, "HEAD"], cwd=repository,
        capture_output=True, check=False,
    )
    if ancestry.returncode != 0:
        raise InterruptedVerificationError("verification HEAD does not descend from H1")
    return h1


def _source_content_digest(repository: Path) -> tuple[str, dict[str, Any]]:
    source_store = repository / base.SOURCE_FULL / "store"
    packed = load_packed_generation(
        source_store, certified.GENERATION_ID,
        expected_provenance_digest=certified.PROVENANCE_DIGEST,
        verify_content=True, validate_records=False,
    )
    manifest = packed.manifest
    physical = resident_store._physical_files(  # type: ignore[attr-defined]
        source_store, certified.GENERATION_ID, manifest,
    )
    bindings = resident_store._verified_bindings(physical, manifest)  # type: ignore[attr-defined]
    content_bindings = resident_store._content_bindings(bindings)  # type: ignore[attr-defined]
    physical_bytes = (
        int(manifest["rows_bytes"]) + int(manifest["overflow_bytes"])
        + physical["manifest"].stat().st_size
    )
    content = {
        "schema_version": resident_store.CONTENT_SCHEMA_VERSION,
        "generation_id": certified.GENERATION_ID,
        "provenance_digest": str(manifest["provenance_digest"]),
        "manifest_digest": str(manifest["manifest_digest"]),
        "pack_contract_digest": str(manifest["pack_contract_digest"]),
        "quantized_bound_contract_digest": str(
            manifest["quantized_bound_contract_digest"]
        ),
        "physical_generation_bytes": physical_bytes,
        "source_files": content_bindings,
        # The original mirror was byte-for-byte validated before RUN_STARTED.
        "mirror_files": content_bindings,
    }
    return stable_hash(content), content


def _attempt_ledger(
    root: Path, cases: Sequence[Mapping[str, Any]], rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    attempts = root / "attempts"
    started_paths = sorted(attempts.glob("*-STARTED.json"))
    completed_paths = sorted(attempts.glob("*-COMPLETED.json"))
    group_paths = sorted(attempts.glob("*-GROUP-*.json"))
    names = {path.name for path in attempts.iterdir()}
    expected_names = {path.name for path in started_paths + completed_paths + group_paths}
    if names != expected_names or any(path.is_symlink() for path in attempts.iterdir()):
        raise InterruptedVerificationError("attempt tree contains an unexpected entry")
    if [path.name for path in started_paths] != ["0000-STARTED.json", "0001-STARTED.json"] \
            or [path.name for path in completed_paths] != ["0001-COMPLETED.json"] \
            or [path.name for path in group_paths] != [
                f"0001-GROUP-{index:02d}.json" for index in range(8)
            ]:
        raise InterruptedVerificationError("interrupted attempt inventory differs")
    start0 = base._read(started_paths[0])[0]
    start1 = base._read(started_paths[1])[0]
    complete1 = base._read(completed_paths[0])[0]
    if start0 != {
        "schema_version": base.RESULT_SCHEMA, "attempt": 0,
        "completed_before": 0, "remaining_before": len(cases),
        "recovered_before": 0, "created_at": start0.get("created_at"),
    }:
        raise InterruptedVerificationError("initial attempt-start state differs")
    if start1 != {
        "schema_version": base.RESULT_SCHEMA, "attempt": 1,
        "completed_before": 1800, "remaining_before": len(cases) - 1800,
        "recovered_before": 1800, "created_at": start1.get("created_at"),
    }:
        raise InterruptedVerificationError("recovery attempt-start state differs")
    elapsed = complete1.get("elapsed_seconds")
    if type(elapsed) not in {int, float} or not math.isfinite(float(elapsed)) \
            or float(elapsed) < 0 or complete1 != {
                "schema_version": base.RESULT_SCHEMA, "attempt": 1,
                "elapsed_seconds": elapsed, "completed_after": len(cases),
                "remaining_after": 0, "worker_failures": [],
                "created_at": complete1.get("created_at"),
            }:
        raise InterruptedVerificationError("recovery attempt completion differs")
    t0 = _time(start0.get("created_at"), "attempt0.created_at")
    t1 = _time(start1.get("created_at"), "attempt1.created_at")
    tc = _time(complete1.get("created_at"), "attempt1.completed_at")
    if not t0 < t1 < tc:
        raise InterruptedVerificationError("attempt chronology differs")

    case_map = {str(case["episode_id"]): case for case in cases}
    assigned: list[str] = []
    groups: list[tuple[dict[str, Any], tuple[str, ...]]] = []
    for index, path in enumerate(group_paths):
        event = base._read(path)[0]
        worker = event.get("worker_result")
        ids = tuple(worker.get("query_episode_ids", ())) if type(worker) is dict else ()
        numeric = ("proposal_seconds", "elapsed_seconds", "peak_rss_mb")
        if not all((
            event.get("schema_version") == base.RESULT_SCHEMA,
            event.get("attempt") == 1,
            event.get("group_index") == index,
            event.get("failure") is None,
            type(worker) is dict,
            type(event.get("assigned_cases")) is int,
            event.get("assigned_cases") == event.get("promoted_cases") == len(ids),
            len(ids) in {183, 184}, len(ids) == len(set(ids)),
            all(type(worker.get(key)) in {int, float}
                and math.isfinite(float(worker[key])) and float(worker[key]) >= 0
                for key in numeric),
            _time(event.get("created_at"), f"group{index}.created_at") <= tc,
        )):
            raise InterruptedVerificationError(f"group event differs: {index}")
        if any(value not in case_map for value in ids):
            raise InterruptedVerificationError(f"group contains unknown query: {index}")
        assigned.extend(ids)
        groups.append((event, ids))
    if len(assigned) != 1470 or len(set(assigned)) != 1470:
        raise InterruptedVerificationError("recovery assignment coverage differs")
    assigned_set = set(assigned)
    recovered = set(case_map).difference(assigned_set)
    if len(recovered) != 1800:
        raise InterruptedVerificationError("recovered complement differs")
    expected_groups = producer._groups(
        [dict(case_map[value]) for value in assigned_set], 8,
    )
    for index, ((_, observed), expected) in enumerate(zip(groups, expected_groups, strict=True)):
        if observed != tuple(str(row["episode_id"]) for row in expected):
            raise InterruptedVerificationError(f"deterministic group differs: {index}")
    for query_id, row in rows.items():
        created = _time(row.get("created_at"), f"case[{query_id}].created_at")
        if query_id in assigned_set and not t1 <= created <= tc:
            raise InterruptedVerificationError("recovery-attempt case chronology differs")
        if query_id in recovered and not t0 <= created < t1:
            raise InterruptedVerificationError("interrupted-attempt case chronology differs")
    return {
        "dangling_attempts": [0], "completed_attempts": [1],
        "recovered_cases": len(recovered), "resume_computed_cases": len(assigned_set),
        "resume_elapsed_seconds": float(elapsed),
        "resume_group_event_digest": stable_hash([
            base._read(path)[0] for path in group_paths
        ]),
    }


def _interruption_evidence(repository: Path, ledger: Mapping[str, Any]) -> dict[str, Any]:
    old_heartbeat_path = repository / REATTACHED / "HEARTBEAT.json"
    attached_path = repository / REATTACHED / "ATTACHED.json"
    resume_start_path = repository / RESUMED / "LISTENER_STARTED.json"
    resume_end_path = repository / RESUMED / "LISTENER_COMPLETED.json"
    stderr_path = repository / RESUMED / "producer.stderr.log"
    old = base._read(old_heartbeat_path)[0]
    attached = base._read(attached_path)[0]
    resume_start = base._read(resume_start_path)[0]
    resume_end = base._read(resume_end_path)[0]
    stderr = stderr_path.read_text()
    if not all((
        attached.get("full_run_monitoring") is False,
        int(old.get("recoverable_work_cases", -1)) == ledger["recovered_cases"],
        int(old.get("maximum_tree_swap_kib", 0)) > 0,
        resume_end.get("producer_returncode") == 1,
        resume_end.get("canonical_cases") == 3270,
        resume_end.get("work_cases") == 0,
        resume_end.get("result_exists") is False,
        "validate-existing generation is absent or linked" in stderr,
    )):
        raise InterruptedVerificationError("interruption evidence differs")
    old_time = _time(old.get("observed_at"), "reattached.observed_at")
    resume_time = _time(resume_start.get("started_at"), "resume.started_at")
    if old_time >= resume_time:
        raise InterruptedVerificationError("monitor/resume chronology differs")
    artifact_paths = [
        attached_path, old_heartbeat_path, resume_start_path, resume_end_path, stderr_path,
    ]
    return {
        "interruption_observed": True,
        "continuous_resident_identity_preserved": False,
        "continuous_resource_monitoring_preserved": False,
        "observed_tree_swap_kib": int(old["maximum_tree_swap_kib"]),
        "last_pre_interruption_observation": old_time.isoformat(),
        "resume_listener_started": resume_time.isoformat(),
        "terminal_publisher_returncode": 1,
        "terminal_failure": "volatile resident generation absent after interruption",
        "artifact_sha256": {
            path.relative_to(repository).as_posix(): _sha(path) for path in artifact_paths
        },
    }


def _performance_status(
    contract: Mapping[str, Any], started0: Mapping[str, Any],
    ledger: Mapping[str, Any], interruption: Mapping[str, Any],
    peak_rss_mb: float,
) -> dict[str, Any]:
    first = _time(started0.get("created_at"), "attempt0.created_at")
    last = _time(
        interruption["last_pre_interruption_observation"], "last_pre_interruption",
    )
    resumed = _time(interruption["resume_listener_started"], "resume_listener_started")
    resume_seconds = float(ledger["resume_elapsed_seconds"])
    lower = (last - first).total_seconds() + resume_seconds
    upper = (resumed - first).total_seconds() + resume_seconds
    if not 0 < lower <= upper:
        raise InterruptedVerificationError("active-time bounds differ")
    performance = contract["performance"]
    descriptive = {
        "active_seconds_lower_bound": lower,
        "active_seconds_upper_bound": upper,
        "queries_per_second_lower_bound": 3270 / upper,
        "queries_per_second_upper_bound": 3270 / lower,
        "case_reported_peak_rss_mb": peak_rss_mb,
    }
    gates = {
        "wall_time_bound": upper <= float(performance["snapshot_wall_seconds_max"]),
        "throughput_bound": 3270 / upper >= float(performance["minimum_queries_per_second"]),
        "case_reported_worker_rss": peak_rss_mb <= float(
            performance["worker_peak_rss_mib_max"]
        ),
        "zero_observed_process_swap": interruption["observed_tree_swap_kib"] == 0,
        "continuous_resource_monitoring": False,
        "continuous_resident_identity": False,
        "producer_terminal_publication": False,
    }
    # Performance qualification is intentionally impossible after this interruption.
    return {
        **descriptive, "gates": gates,
        "performance_qualified": False,
        "resource_qualified": False,
    }


def verify(repository: Path, root: Path | None = None) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    root = (root or repository / OUTPUT).resolve(strict=True)
    verifier_path = Path(__file__).resolve(strict=True)
    prereg, prereg_raw = base._read(repository / base.PREREGISTRATION)
    pstate = {key: value for key, value in prereg.items() if key != "preregistration_digest"}
    if prereg.get("schema_version") != base.PREREG_SCHEMA \
            or prereg.get("preregistration_digest") != stable_hash(pstate):
        raise InterruptedVerificationError("preregistration seal differs")
    h0 = str(prereg.get("implementation_h0"))
    h1 = _sole_preregistration_commit(repository, h0, prereg_raw)
    for name, expected in prereg.get("runtime_files", {}).items():
        if sha256(_git(repository, "show", f"{h0}:{name}", raw=True)).hexdigest() != expected \
                or sha256(_git(repository, "show", f"{h1}:{name}", raw=True)).hexdigest() != expected \
                or _sha(repository / name) != expected:
            raise InterruptedVerificationError(f"runtime source drifted: {name}")

    contract, contract_raw = base._read(repository / base.CONTRACT)
    registry, registry_raw = base._read(repository / base.REGISTRY / "query-registry.json")
    registry_seal, registry_seal_raw = base._read(repository / base.REGISTRY / "SEALED.json")
    receipt, receipt_raw = base._read(
        repository / base.REGISTRY_VERIFICATION / "VERIFIED.json"
    )
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
    )):
        raise InterruptedVerificationError("registry/preregistration binding differs")

    expected_names = {"RUN_STARTED.json", "cases", "work", "attempts"}
    if {path.name for path in root.iterdir()} != expected_names \
            or any(path.is_symlink() for path in root.rglob("*")):
        raise InterruptedVerificationError("interrupted terminal tree differs")
    for work_root in sorted((root / "work").iterdir()):
        if not work_root.is_dir() \
                or set(path.name for path in work_root.iterdir()) != {"cases"} \
                or any((work_root / "cases").iterdir()):
            raise InterruptedVerificationError("work tree is not empty and canonical")
    started, _ = base._read(root / "RUN_STARTED.json")
    execution = base._execution_contract(prereg, registry)
    start_state = {key: value for key, value in started.items() if key != "result_digest"}
    if not all((
        started.get("result_digest") == stable_hash(start_state),
        started.get("status") == "running",
        started.get("scheduled_queries") == len(cases),
        started.get("preregistration_digest") == prereg["preregistration_digest"],
        started.get("contract_digest") == execution["contract_digest"],
        started.get("registry_digest") == registry["registry_digest"],
        started.get("authority_accessed") is False,
        started.get("real_forward_outcomes_accessed") is False,
    )):
        raise InterruptedVerificationError("run-start seal differs")
    source_content_digest, source_content = _source_content_digest(repository)
    if source_content_digest != started.get("source_content_digest"):
        raise InterruptedVerificationError("immutable source content differs from RUN_STARTED")

    paths = sorted((root / "cases").glob("*.json"))
    if len(paths) != len(cases) or any(path.is_symlink() for path in paths):
        raise InterruptedVerificationError("case inventory differs")
    manifest = [{
        "path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
        "sha256": _sha(path),
    } for path in paths]
    by_query: dict[str, dict[str, Any]] = {}
    for path in paths:
        row = base._read(path)[0]
        query_id = str(row.get("query_episode_id"))
        if query_id in by_query:
            raise InterruptedVerificationError("duplicate query result")
        by_query[query_id] = row

    inputs = m13.Inputs(
        repository, repository / base.CONFIG, repository / base.REGISTRY,
        repository / base.SOURCE_FULL / "store", certified.RESIDENT_ROOT, root,
        certified.GENERATION_ID, certified.PROVENANCE_DIGEST, 1024 ** 3,
        str(registry["registry_digest"]), (), str(prereg["preregistration_digest"]),
    )
    exact: list[float] = []
    peak_rss = 0.0
    verified_matches = 0
    for ordinal, case in enumerate(cases):
        row = by_query.get(str(case["episode_id"]))
        if row is None or not all((
            row.get("contract_digest") == execution["contract_digest"],
            row.get("registry_digest") == registry["registry_digest"],
            row.get("generation_id") == certified.GENERATION_ID,
            row.get("registry_case_id") == case["case_id"],
            row.get("query_episode_id") == case["episode_id"],
            row.get("query_stock_prefix") == case["stock_prefix"],
            row.get("query_benchmark_prefix") == case["benchmark_prefix"],
            row.get("gate_passed") is True,
            row.get("real_forward_outcomes_accessed") is False,
            type(row.get("matches")) is list and len(row["matches"]) == 20,
            row.get("result_digest") == base._deterministic_case_digest(row),
            row.get("checkpoint_integrity_digest") == base._integrity_digest(row),
        )):
            raise InterruptedVerificationError(f"case seal differs: {case['case_id']}")
        case_input = m13.CaseInput(ordinal, dict(case))
        one = m13.Inputs(
            inputs.repository, inputs.config_path, inputs.registry_root,
            inputs.source_store_root, inputs.resident_root, inputs.output_root,
            inputs.generation_id, inputs.provenance_digest, inputs.reserve_bytes,
            inputs.registry_digest, (case_input,), inputs.prereg_digest,
        )
        source, episode, request, packed = m13._case_context(one, case_input)
        binding = m13.query_binding(
            source, episode, request, packed, certified.PROVENANCE_DIGEST,
        )
        certificate = {**row["certificate"], "elapsed_seconds": 0.0}
        m13.validate_certificate_and_matches(
            certificate, row["matches"], str(case["episode_id"]),
            expected_input_digest=binding["certified_input_digest"],
        )
        exact_seconds = float(row["exact_seconds"])
        rss = float(row["peak_rss_mb"])
        if not all(math.isfinite(value) and value >= 0 for value in (exact_seconds, rss)):
            raise InterruptedVerificationError(f"non-finite measurement: {case['case_id']}")
        exact.append(exact_seconds)
        peak_rss = max(peak_rss, rss)
        verified_matches += 20

    ledger = _attempt_ledger(root, cases, by_query)
    interruption = _interruption_evidence(repository, ledger)
    started0 = base._read(root / "attempts/0000-STARTED.json")[0]
    performance = _performance_status(
        contract, started0, ledger, interruption, peak_rss,
    )
    p95 = sorted(exact)[math.ceil(.95 * len(exact)) - 1]
    state = {
        "schema_version": SCHEMA,
        "status": "verified_semantic_only_after_interruption",
        "semantic_passed": True,
        "overall_t14_08_passed": False,
        "producer_terminal_result_present": False,
        "verifier_path": verifier_path.relative_to(repository).as_posix(),
        "verifier_sha256": _sha(verifier_path),
        "implementation_h0": h0, "preregistration_h1": h1,
        "preregistration_digest": prereg["preregistration_digest"],
        "registry_digest": registry["registry_digest"],
        "contract_digest": execution["contract_digest"],
        "source_content_digest": source_content_digest,
        "source_content_binding_digest": stable_hash(source_content),
        "case_manifest_digest": stable_hash(manifest),
        "verified_cases": len(cases), "verified_matches": verified_matches,
        "exact_seconds_p95_descriptive": p95,
        "exact_seconds_max_descriptive": max(exact),
        "ledger": ledger, "interruption": interruption,
        "performance": performance,
        "audit_sample_verification_pending": True,
        "sustained_source_locks_completed": 1,
        "sustained_source_locks_required": int(
            contract["audit"]["daily_drift_minimum_distinct_source_locks"]
        ),
        "authority_accessed": False,
        "real_forward_outcomes_accessed": False,
        "production_promotion_authorized": False,
    }
    return {**state, "result_digest": stable_hash(state)}


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise InterruptedVerificationError("verification root exists")
    path.mkdir(parents=False)
    target = path / "VERIFIED.json"
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
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
