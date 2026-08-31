"""Independently verify the preregistered T14-08 12-case audit authority."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.m04r import m04r13_threaded_certified_exposed as m13
from experiments.m04r import m04r14_shadow_run as base
from experiments.m04r import m04r14_untouched_candidate_contract as certified
from market_analogues.types import stable_hash


SCHEMA = "m04r14-shadow-audit-authority-verification-v1"
PREREGISTRATION = Path("experiments/m04r/m04r14_shadow_audit_preregistered.json")
REGISTRY = Path("config/data/analogues/m04r14/nasdaq-shadow-denominator-v1")
AUTHORITY = Path("config/data/analogues/m04r14/nasdaq-shadow-audit-authority-v1")
OUTPUT = Path(
    "config/data/analogues/m04r14/nasdaq-shadow-audit-authority-v1-verification"
)
CONTROLS = {
    "block_rows": 4097, "deferred_alignments": True,
    "exact_workers_per_process": 1, "initial_frontier_rows": 16_384,
    "maximum_frontier_rows": 32_768, "numba_threads_per_process": 1,
    "processes": 8, "requested_positions": True, "seed_rows": 512,
    "sorted_joined_iqr_merge": True, "vector_lower_bounds": True,
}


class AuditVerificationError(RuntimeError):
    pass


def _selected(
    registry: Mapping[str, Any], sample: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    cases = registry.get("cases_data")
    if type(cases) is not list:
        raise AuditVerificationError("registry cases missing")
    by_id = {str(row.get("episode_id")): row for row in cases}
    if len(by_id) != 3270 or len(sample) != 12:
        raise AuditVerificationError("audit input cardinality differs")
    result: list[dict[str, Any]] = []
    keys = ("case_id", "episode_id", "symbol", "quality_tier", "liquidity_stratum")
    for item in sample:
        row = by_id.get(str(item.get("episode_id")))
        if row is None or any(row.get(key) != item.get(key) for key in keys):
            raise AuditVerificationError("sample/registry binding differs")
        result.append(dict(row))
    if len({row["episode_id"] for row in result}) != 12:
        raise AuditVerificationError("duplicate audit identity")
    return result


def _preregistration_child(repository: Path, raw: bytes, h0: str) -> str:
    candidates: list[str] = []
    for line in str(base._git(repository, "rev-list", "--all", "--children")).splitlines():
        values = line.split()
        if values and values[0] == h0:
            candidates.extend(values[1:])
    accepted: list[str] = []
    for child in sorted(set(candidates)):
        lineage = str(base._git(
            repository, "rev-list", "--parents", "-n", "1", child,
        )).split()
        changed = str(base._git(
            repository, "diff-tree", "--no-commit-id", "--name-only", "-r", child,
        )).splitlines()
        if lineage != [child, h0] or changed != [PREREGISTRATION.as_posix()]:
            continue
        blob = base._git(
            repository, "show", f"{child}:{PREREGISTRATION.as_posix()}", raw=True,
        )
        if blob == raw:
            accepted.append(child)
    if len(accepted) != 1:
        raise AuditVerificationError("audit preregistration lifecycle differs")
    return accepted[0]


def _execution(prereg: Mapping[str, Any], registry: Mapping[str, Any]) -> dict[str, Any]:
    state = {
        "schema_version": "m04r14-shadow-audit-authority-v1",
        "preregistration_digest": prereg["preregistration_digest"],
        "registry_digest": registry["registry_digest"],
        "generation_id": certified.GENERATION_ID,
        "controls": CONTROLS,
        "authority_role": "separate-16k-32k-certified-frontier",
        "candidate_case_root_accessed": False,
        "real_forward_outcomes_accessed": False,
    }
    return {**state, "contract_digest": stable_hash(state)}


def verify(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    prereg, prereg_raw = base._read(repository / PREREGISTRATION)
    prereg_state = {
        key: value for key, value in prereg.items() if key != "preregistration_digest"
    }
    if prereg.get("schema_version") != "m04r14-shadow-audit-preregistration-v1" \
            or prereg.get("preregistration_digest") != stable_hash(prereg_state) \
            or prereg.get("controls") != CONTROLS:
        raise AuditVerificationError("audit preregistration differs")
    h0 = str(prereg.get("implementation_h0"))
    h1 = _preregistration_child(repository, prereg_raw, h0)
    for name, expected in prereg.get("runtime_files", {}).items():
        if sha256(base._git(repository, "show", f"{h0}:{name}", raw=True)).hexdigest() != expected \
                or sha256(base._git(repository, "show", f"{h1}:{name}", raw=True)).hexdigest() != expected \
                or base._sha(repository / name) != expected:
            raise AuditVerificationError(f"audit runtime drifted: {name}")
    original, original_raw = base._read(repository / base.PREREGISTRATION)
    original_state = {
        key: value for key, value in original.items()
        if key != "preregistration_digest"
    }
    registry, registry_raw = base._read(repository / REGISTRY / "query-registry.json")
    sample = registry.get("audit_sample")
    if any((
        original.get("preregistration_digest") != stable_hash(original_state),
        prereg.get("shadow_preregistration_sha256") != sha256(original_raw).hexdigest(),
        prereg.get("shadow_preregistration_digest") != original.get("preregistration_digest"),
        prereg.get("registry_sha256") != sha256(registry_raw).hexdigest(),
        prereg.get("registry_digest") != registry.get("registry_digest"),
        prereg.get("source_lock_digest") != registry.get("source_lock", {}).get("source_lock_digest"),
        prereg.get("audit_sample") != sample,
        prereg.get("audit_sample_digest") != stable_hash(sample),
    )):
        raise AuditVerificationError("frozen audit inputs differ")
    cases = _selected(registry, sample)
    if prereg.get("audit_query_ids_digest") != stable_hash([
        row["episode_id"] for row in cases
    ]):
        raise AuditVerificationError("audit query digest differs")
    execution = _execution(prereg, registry)

    root = repository / AUTHORITY
    if {path.name for path in root.iterdir()} != {
        "RUN_STARTED.json", "AUTHORITY.json", "cases",
    } or any(path.is_symlink() for path in root.rglob("*")):
        raise AuditVerificationError("authority tree differs")
    started, _ = base._read(root / "RUN_STARTED.json")
    aggregate, aggregate_raw = base._read(root / "AUTHORITY.json")
    state = {
        key: value for key, value in aggregate.items()
        if key not in {"result_digest", "created_at"}
    }
    if not all((
        aggregate.get("result_digest") == stable_hash(state),
        aggregate.get("status") == "sealed",
        aggregate.get("semantic_passed") is True,
        aggregate.get("cases") == 12,
        aggregate.get("matches") == 240,
        aggregate.get("controls") == CONTROLS,
        aggregate.get("contract_digest") == execution["contract_digest"],
        aggregate.get("registry_digest") == registry["registry_digest"],
        aggregate.get("generation_id") == certified.GENERATION_ID,
        aggregate.get("preregistration_h1") == h1,
        aggregate.get("preregistration_digest") == prereg["preregistration_digest"],
        aggregate.get("candidate_case_root_accessed") is False,
        aggregate.get("real_forward_outcomes_accessed") is False,
        started.get("contract_digest") == execution["contract_digest"],
        started.get("source_content_digest") == aggregate.get("source_content_digest"),
        started.get("resident_identity_digest") == aggregate.get("resident_identity_digest"),
    )):
        raise AuditVerificationError("authority aggregate differs")
    paths = sorted((root / "cases").glob("*.json"))
    manifest = [{
        "path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
        "sha256": base._sha(path),
    } for path in paths]
    if len(paths) != 12 or manifest != aggregate.get("case_manifest") \
            or stable_hash(manifest) != aggregate.get("case_manifest_digest"):
        raise AuditVerificationError("authority case manifest differs")
    observed: dict[str, dict[str, Any]] = {}
    for path in paths:
        row, _ = base._read(path)
        query_id = str(row.get("query_episode_id"))
        if query_id in observed:
            raise AuditVerificationError("duplicate authority query")
        observed[query_id] = row
    if set(observed) != {str(row["episode_id"]) for row in cases}:
        raise AuditVerificationError("authority query coverage differs")

    inputs = m13.Inputs(
        repository, repository / base.CONFIG, repository / REGISTRY,
        repository / base.SOURCE_FULL / "store", certified.RESIDENT_ROOT, root,
        certified.GENERATION_ID, certified.PROVENANCE_DIGEST, 1024 ** 3,
        str(registry["registry_digest"]),
        tuple(m13.CaseInput(index, row) for index, row in enumerate(cases)),
        str(prereg["preregistration_digest"]),
    )
    original = (m13.INITIAL_FRONTIER, m13.MAXIMUM_FRONTIER)
    m13.INITIAL_FRONTIER, m13.MAXIMUM_FRONTIER = 16_384, 32_768
    verified_matches = 0
    try:
        for index, case in enumerate(cases):
            row = observed[str(case["episode_id"])]
            if not all((
                row.get("contract_digest") == execution["contract_digest"],
                row.get("registry_case_id") == case["case_id"],
                row.get("query_stock_prefix") == case["stock_prefix"],
                row.get("query_benchmark_prefix") == case["benchmark_prefix"],
                row.get("gate_passed") is True,
                type(row.get("matches")) is list and len(row["matches"]) == 20,
                row.get("result_digest") == base._deterministic_case_digest(row),
                row.get("checkpoint_integrity_digest") == base._integrity_digest(row),
                row.get("real_forward_outcomes_accessed") is False,
            )):
                raise AuditVerificationError(f"authority case differs: {case['case_id']}")
            case_input = m13.CaseInput(index, case)
            source, episode, request, packed = m13._case_context(inputs, case_input)
            binding = m13.query_binding(
                source, episode, request, packed, certified.PROVENANCE_DIGEST,
            )
            m13.validate_certificate_and_matches(
                {**row["certificate"], "elapsed_seconds": 0.0}, row["matches"],
                str(case["episode_id"]),
                expected_input_digest=binding["certified_input_digest"],
            )
            verified_matches += 20
    finally:
        m13.INITIAL_FRONTIER, m13.MAXIMUM_FRONTIER = original
    assigned = [
        query_id
        for group in aggregate.get("groups", [])
        for query_id in group.get("query_episode_ids", [])
    ]
    if len(assigned) != 12 or len(set(assigned)) != 12 or set(assigned) != set(observed):
        raise AuditVerificationError("authority group assignment differs")
    resident = m13.resident_full(
        repository / base.SOURCE_FULL / "store", certified.RESIDENT_ROOT,
        certified.GENERATION_ID, certified.PROVENANCE_DIGEST, 1024 ** 3,
    )
    if resident["content_digest"] != aggregate.get("source_content_digest") \
            or resident["identity_digest"] != aggregate.get("resident_identity_digest"):
        raise AuditVerificationError("authority resident binding differs")
    result_state = {
        "schema_version": SCHEMA, "status": "verified", "passed": True,
        "authority_result_digest": aggregate["result_digest"],
        "authority_result_sha256": sha256(aggregate_raw).hexdigest(),
        "preregistration_digest": prereg["preregistration_digest"],
        "registry_digest": registry["registry_digest"],
        "verified_cases": 12, "verified_matches": verified_matches,
        "candidate_case_root_accessed": False,
        "real_forward_outcomes_accessed": False,
        "comparison_authorized": True,
        "production_promotion_authorized": False,
    }
    return {**result_state, "result_digest": stable_hash(result_state)}


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise AuditVerificationError("audit verification root exists")
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
    args = parser.parse_args(argv)
    repository = args.repository.resolve(strict=True)
    value = verify(repository)
    _publish(repository / OUTPUT, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
