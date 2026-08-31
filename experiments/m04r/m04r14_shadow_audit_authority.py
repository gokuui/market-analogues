"""Build the preregistered 12-case T14-08 shadow audit authority.

The authority reads the frozen denominator and immutable packed source, but never
the 3,270 candidate case files or forward outcomes.  Its 16K->32K certified
frontier and 4,097-row scan schedule are deliberately distinct from the shadow
producer's 1K->16K/4,096-row schedule.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
import json
import multiprocessing
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

from experiments.m04r import m04r11_build_authorities as engine
from experiments.m04r import m04r13_threaded_certified_exposed as m13
from experiments.m04r import m04r14_shadow_run as base
from experiments.m04r import m04r14_untouched_candidate_contract as certified
from market_analogues.types import stable_hash


SCHEMA = "m04r14-shadow-audit-authority-v1"
PREREG_SCHEMA = "m04r14-shadow-audit-preregistration-v1"
PREREGISTRATION = Path("experiments/m04r/m04r14_shadow_audit_preregistered.json")
OUTPUT = Path("config/data/analogues/m04r14/nasdaq-shadow-audit-authority-v1")
CONTROLS = {
    "block_rows": 4097,
    "deferred_alignments": True,
    "exact_workers_per_process": 1,
    "initial_frontier_rows": 16_384,
    "maximum_frontier_rows": 32_768,
    "numba_threads_per_process": 1,
    "processes": 8,
    "requested_positions": True,
    "seed_rows": 512,
    "sorted_joined_iqr_merge": True,
    "vector_lower_bounds": True,
}
RUNTIME_FILES = (
    "experiments/m04r/m04r14_shadow_audit_authority.py",
    "experiments/m04r/verify_m04r14_shadow_audit_authority.py",
    "experiments/m04r/compare_m04r14_shadow_audit.py",
    "experiments/m04r/m04r11_build_authorities.py",
    "experiments/m04r/m04r13_threaded_certified_exposed.py",
)


class AuditAuthorityError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _manifest(repository: Path, head: str) -> dict[str, str]:
    tracked = set(str(base._git(
        repository, "ls-tree", "-r", "--name-only", head,
    )).splitlines())
    names = sorted({
        name for name in tracked
        if name.startswith("src/market_analogues/") and name.endswith(".py")
    } | set(RUNTIME_FILES))
    if any(name not in tracked for name in names):
        raise AuditAuthorityError("audit runtime contains an uncommitted file")
    return {
        name: sha256(base._git(
            repository, "show", f"{head}:{name}", raw=True,
        )).hexdigest()
        for name in names
    }


def _selected_cases(
    registry: Mapping[str, Any], sample: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    cases = registry.get("cases_data")
    if type(cases) is not list or len(cases) != 3270:
        raise AuditAuthorityError("shadow registry case inventory differs")
    by_id = {str(row.get("episode_id")): row for row in cases}
    if len(by_id) != len(cases) or len(sample) != 12:
        raise AuditAuthorityError("audit sample cardinality differs")
    selected: list[dict[str, Any]] = []
    sample_keys = (
        "case_id", "episode_id", "symbol", "quality_tier",
        "liquidity_stratum",
    )
    for audit_row in sample:
        row = by_id.get(str(audit_row.get("episode_id")))
        if row is None or any(row.get(key) != audit_row.get(key) for key in sample_keys):
            raise AuditAuthorityError("audit sample no longer binds registry")
        selected.append(dict(row))
    if len({row["episode_id"] for row in selected}) != 12:
        raise AuditAuthorityError("audit sample contains duplicate queries")
    return selected


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    h0 = base._clean_head(repository)
    original, original_raw = base._read(repository / base.PREREGISTRATION)
    original_state = {
        key: value for key, value in original.items()
        if key != "preregistration_digest"
    }
    registry, registry_raw = base._read(
        repository / base.REGISTRY / "query-registry.json"
    )
    sample = registry.get("audit_sample")
    if not all((
        original.get("preregistration_digest") == stable_hash(original_state),
        original.get("audit_sample") == sample,
        original.get("audit_sample_digest") == registry.get("audit_sample_digest"),
        registry.get("audit_sample_digest") == stable_hash(sample),
        registry.get("registry_digest") == original.get("registry_digest"),
    )):
        raise AuditAuthorityError("original shadow preregistration differs")
    selected = _selected_cases(registry, sample)
    state = {
        "schema_version": PREREG_SCHEMA,
        "status": "frozen_before_audit_authority",
        "implementation_h0": h0,
        "runtime_files": _manifest(repository, h0),
        "shadow_preregistration_sha256": sha256(original_raw).hexdigest(),
        "shadow_preregistration_digest": original["preregistration_digest"],
        "registry_sha256": sha256(registry_raw).hexdigest(),
        "registry_digest": registry["registry_digest"],
        "source_lock_digest": registry["source_lock"]["source_lock_digest"],
        "audit_sample": sample,
        "audit_sample_digest": registry["audit_sample_digest"],
        "audit_query_ids_digest": stable_hash([
            row["episode_id"] for row in selected
        ]),
        "controls": CONTROLS,
        "authority_method": (
            "full packed-universe lower-bound scan plus independently scheduled "
            "certified exact frontier and threshold closure"
        ),
        "candidate_case_root_accessed": False,
        "real_forward_outcomes_accessed": False,
        "production_promotion_authorized": False,
    }
    return {**state, "preregistration_digest": stable_hash(state)}


def _sole_preregistration_child(repository: Path, prereg_raw: bytes, h0: str) -> str:
    children: list[str] = []
    for line in str(base._git(repository, "rev-list", "--all", "--children")).splitlines():
        values = line.split()
        if values and values[0] == h0:
            children.extend(values[1:])
    accepted: list[str] = []
    for child in sorted(set(children)):
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
        if blob == prereg_raw:
            accepted.append(child)
    if len(accepted) != 1:
        raise AuditAuthorityError("expected one exact audit-preregistration child")
    return accepted[0]


def _validate_preregistration(
    repository: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], str]:
    if base._git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise AuditAuthorityError("globally clean Git worktree required")
    prereg, prereg_raw = base._read(repository / PREREGISTRATION)
    state = {key: value for key, value in prereg.items() if key != "preregistration_digest"}
    if prereg.get("schema_version") != PREREG_SCHEMA \
            or prereg.get("preregistration_digest") != stable_hash(state) \
            or prereg.get("controls") != CONTROLS:
        raise AuditAuthorityError("audit preregistration seal differs")
    h0 = str(prereg.get("implementation_h0"))
    h1 = _sole_preregistration_child(repository, prereg_raw, h0)
    if base._git(repository, "merge-base", "--is-ancestor", h1, "HEAD") != "":
        # A successful merge-base check prints nothing; failures raise in base._git.
        raise AuditAuthorityError("unexpected merge-base output")
    for name, expected in prereg.get("runtime_files", {}).items():
        if sha256(base._git(repository, "show", f"{h0}:{name}", raw=True)).hexdigest() != expected \
                or sha256(base._git(repository, "show", f"{h1}:{name}", raw=True)).hexdigest() != expected \
                or base._sha(repository / name) != expected:
            raise AuditAuthorityError(f"audit runtime drifted: {name}")
    original, original_raw = base._read(repository / base.PREREGISTRATION)
    registry, registry_raw = base._read(
        repository / base.REGISTRY / "query-registry.json"
    )
    sample = registry.get("audit_sample")
    if any((
        prereg.get("shadow_preregistration_sha256") != sha256(original_raw).hexdigest(),
        prereg.get("shadow_preregistration_digest") != original.get("preregistration_digest"),
        prereg.get("registry_sha256") != sha256(registry_raw).hexdigest(),
        prereg.get("registry_digest") != registry.get("registry_digest"),
        prereg.get("source_lock_digest") != registry.get("source_lock", {}).get("source_lock_digest"),
        prereg.get("audit_sample") != sample,
        prereg.get("audit_sample_digest") != stable_hash(sample),
    )):
        raise AuditAuthorityError("audit frozen inputs differ")
    selected = _selected_cases(registry, sample)
    if prereg.get("audit_query_ids_digest") != stable_hash([
        row["episode_id"] for row in selected
    ]):
        raise AuditAuthorityError("audit query identity digest differs")
    return prereg, registry, selected, h1


def _execution(prereg: Mapping[str, Any], registry: Mapping[str, Any]) -> dict[str, Any]:
    state = {
        "schema_version": SCHEMA,
        "preregistration_digest": prereg["preregistration_digest"],
        "registry_digest": registry["registry_digest"],
        "generation_id": certified.GENERATION_ID,
        "controls": CONTROLS,
        "authority_role": "separate-16k-32k-certified-frontier",
        "candidate_case_root_accessed": False,
        "real_forward_outcomes_accessed": False,
    }
    return {**state, "contract_digest": stable_hash(state)}


def _case_valid(
    repository: Path, root: Path, case: Mapping[str, Any], execution: Mapping[str, Any],
) -> bool:
    try:
        row, _ = base._read(root / "cases" / f"{case['episode_id']}.json")
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
            row.get("result_digest") == base._deterministic_case_digest(row),
            row.get("checkpoint_integrity_digest") == base._integrity_digest(row),
        )):
            return False
        case_input = m13.CaseInput(0, dict(case))
        inputs = m13.Inputs(
            repository, repository / base.CONFIG, repository / base.REGISTRY,
            repository / base.SOURCE_FULL / "store", certified.RESIDENT_ROOT, root,
            certified.GENERATION_ID, certified.PROVENANCE_DIGEST, 1024 ** 3,
            str(execution["registry_digest"]), (case_input,),
            str(execution["preregistration_digest"]),
        )
        source, episode, request, packed = m13._case_context(inputs, case_input)
        binding = m13.query_binding(
            source, episode, request, packed, certified.PROVENANCE_DIGEST,
        )
        original = (m13.INITIAL_FRONTIER, m13.MAXIMUM_FRONTIER)
        m13.INITIAL_FRONTIER, m13.MAXIMUM_FRONTIER = 16_384, 32_768
        try:
            m13.validate_certificate_and_matches(
                {**row["certificate"], "elapsed_seconds": 0.0}, row["matches"],
                str(case["episode_id"]),
                expected_input_digest=binding["certified_input_digest"],
            )
        finally:
            m13.INITIAL_FRONTIER, m13.MAXIMUM_FRONTIER = original
        return True
    except Exception:
        return False


def _worker(
    repository: str, output: str, execution: dict[str, Any],
    cases: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    root = Path(repository)
    return engine._worker(
        str(root / base.CONFIG), str(root / base.SOURCE_FULL), output,
        certified.GENERATION_ID, execution, cases, CONTROLS,
    )


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    prereg, registry, cases, h1 = _validate_preregistration(repository)
    root = repository / OUTPUT
    if root.exists() or root.is_symlink():
        raise AuditAuthorityError("audit authority root exists")
    root.mkdir(parents=True)
    (root / "cases").mkdir()
    execution = _execution(prereg, registry)
    started_at = perf_counter()
    resident_before = m13.resident_full(
        repository / base.SOURCE_FULL / "store", certified.RESIDENT_ROOT,
        certified.GENERATION_ID, certified.PROVENANCE_DIGEST, 1024 ** 3,
    )
    run = {
        "schema_version": SCHEMA, "status": "running",
        "preregistration_h1": h1,
        "preregistration_digest": prereg["preregistration_digest"],
        "contract_digest": execution["contract_digest"],
        "registry_digest": registry["registry_digest"],
        "generation_id": certified.GENERATION_ID,
        "controls": CONTROLS,
        "source_content_digest": resident_before["content_digest"],
        "resident_identity_digest": resident_before["identity_digest"],
        "scheduled_cases": 12,
        "candidate_case_root_accessed": False,
        "real_forward_outcomes_accessed": False,
        "created_at": _now(),
    }
    base._atomic(root / "RUN_STARTED.json", run)
    try:
        groups = base._groups(cases, min(CONTROLS["processes"], len(cases)))
        context = multiprocessing.get_context("spawn")
        results: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=len(groups), mp_context=context) as pool:
            futures = {
                pool.submit(
                    _worker, str(repository), str(root), execution, group,
                ): index
                for index, group in enumerate(groups)
            }
            for future in as_completed(futures):
                results.append({"group_index": futures[future], **future.result()})
        if not all(_case_valid(repository, root, case, execution) for case in cases):
            raise AuditAuthorityError("one or more audit authority cases failed validation")
        paths = sorted((root / "cases").glob("*.json"))
        if len(paths) != 12:
            raise AuditAuthorityError("audit authority case inventory differs")
        resident_after = m13.resident_full(
            repository / base.SOURCE_FULL / "store", certified.RESIDENT_ROOT,
            certified.GENERATION_ID, certified.PROVENANCE_DIGEST, 1024 ** 3,
        )
        if resident_after["content_digest"] != resident_before["content_digest"] \
                or resident_after["identity_digest"] != resident_before["identity_digest"]:
            raise AuditAuthorityError("source/resident identity changed during audit")
        manifest = [{
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": base._sha(path),
        } for path in paths]
        state = {
            "schema_version": SCHEMA, "status": "sealed", "semantic_passed": True,
            "preregistration_h1": h1,
            "preregistration_digest": prereg["preregistration_digest"],
            "contract_digest": execution["contract_digest"],
            "registry_digest": registry["registry_digest"],
            "generation_id": certified.GENERATION_ID,
            "controls": CONTROLS,
            "cases": 12, "matches": 240,
            "groups": sorted(results, key=lambda row: row["group_index"]),
            "case_manifest": manifest,
            "case_manifest_digest": stable_hash(manifest),
            "wall_seconds": perf_counter() - started_at,
            "source_content_digest": resident_after["content_digest"],
            "resident_identity_digest": resident_after["identity_digest"],
            "candidate_case_root_accessed": False,
            "real_forward_outcomes_accessed": False,
            "production_promotion_authorized": False,
        }
        result = {**state, "result_digest": stable_hash(state), "created_at": _now()}
        base._atomic(root / "AUTHORITY.json", result)
        return result
    except BaseException as exc:
        if not (root / "AUTHORITY.json").exists() and not (root / "FAILED.json").exists():
            base._atomic(root / "FAILED.json", {
                "schema_version": SCHEMA, "status": "failed",
                "error_type": type(exc).__name__, "message": str(exc),
                "resume_authorized": False, "created_at": _now(),
            })
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--build-preregistration", action="store_true")
    args = parser.parse_args(argv)
    repository = args.repository.resolve(strict=True)
    value = (
        build_preregistration(repository)
        if args.build_preregistration else execute(repository)
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
