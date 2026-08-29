"""Independent raw-artifact verifier for an M04R-14 grouped performance run."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.m04r import m04r14_all60_contract as contract


SCHEMA = "m04r14-performance-run-verification-v1"
BASELINE = Path("config/data/analogues/m04r14/all60-certified-development-v3")
REGISTRY = Path("config/data/analogues/m04r10/nasdaq-untouched-authority-registry/query-registry.json")
CASE_RESULT_OMITTED = {
    "created_at", "proposal_seconds", "amortized_proposal_seconds",
    "exact_seconds", "final_exact_seconds", "frontier_attempt_measurements",
    "peak_rss_mb", "result_digest", "checkpoint_integrity_digest",
}
CONTROLS = {
    "block_rows": 4_096, "deferred_alignments": True,
    "exact_workers_per_process": 1, "initial_frontier_rows": 1_000,
    "maximum_frontier_rows": 16_384, "numba_threads_per_process": 1,
    "requested_positions": True, "seed_rows": 512,
    "sorted_joined_iqr_merge": True, "vector_lower_bounds": True,
}


class VerificationError(RuntimeError):
    pass


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise VerificationError(f"regular file required: {path}")
    raw = path.read_bytes()

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise VerificationError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw, object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                VerificationError(f"non-finite JSON: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSON: {path}") from exc
    if type(value) is not dict:
        raise VerificationError(f"JSON object required: {path}")
    return value, raw


def _digest(value: Any) -> str:
    return contract.stable_digest(value)


def _sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _finite(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value)) and value >= 0


def _case_result_digest(value: Mapping[str, Any]) -> str:
    return _digest({key: item for key, item in value.items() if key not in CASE_RESULT_OMITTED})


def _case_integrity_digest(value: Mapping[str, Any]) -> str:
    return _digest({key: item for key, item in value.items()
                    if key not in {"created_at", "checkpoint_integrity_digest"}})


def _baseline(repository: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    root = repository / BASELINE
    complete, _ = _read(root / "COMPLETE.json")
    semantics, _ = _read(root / "SEMANTICS.json")
    measurements, _ = _read(root / "MEASUREMENTS.json")
    if not all((complete.get("status") == "complete",
                complete.get("semantic_passed") is True,
                semantics.get("semantic_passed") is True,
                measurements.get("summary", {}).get("cases") == 60)):
        raise VerificationError("baseline differs")
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "cases").glob("*/CASE.json")):
        value, _ = _read(path)
        semantic, measurement = value.get("semantic"), value.get("measurement")
        if type(semantic) is not dict or type(measurement) is not dict:
            raise VerificationError("baseline case differs")
        case_id = semantic.get("case_id")
        if type(case_id) is not str or case_id in rows:
            raise VerificationError("baseline case identity differs")
        rows[case_id] = {"semantic": semantic, "measurement": measurement}
    if len(rows) != 60:
        raise VerificationError("baseline count differs")
    return rows, {
        "complete_digest": complete["complete_digest"],
        "semantic_digest": semantics["semantic_digest"],
        "measurement_digest": measurements["measurement_digest"],
        "serial_total_end_to_end_seconds": measurements["summary"]["total_end_to_end_seconds"],
    }


def verify(run_root: Path, *, repository: Path, case_limit: int, processes: int) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    root = run_root.resolve(strict=True)
    if root == (repository / "config/data/analogues/m04r14/throughput-development-poc-v1").resolve(strict=True):
        pass
    elif repository not in root.parents:
        raise VerificationError("run root must be inside repository")
    if not 1 <= case_limit <= 60 or not 1 <= processes <= case_limit:
        raise VerificationError("requested shape differs")
    directories = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_dir()}
    files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if directories != {"cases"} or files != {"RESULT.json"} | {
        path.relative_to(root).as_posix() for path in (root / "cases").glob("*.json")
    } or len(files) != case_limit + 1 or any(path.is_symlink() for path in root.rglob("*")):
        raise VerificationError("run tree differs")

    result, result_raw = _read(root / "RESULT.json")
    result_input = {key: value for key, value in result.items() if key not in {"created_at", "result_digest"}}
    if result.get("result_digest") != _digest(result_input):
        raise VerificationError("result digest differs")
    baseline, baseline_binding = _baseline(repository)
    registry, _ = _read(repository / REGISTRY)
    registry_rows = registry.get("cases_data")
    if type(registry_rows) is not list or len(registry_rows) != 60:
        raise VerificationError("registry differs")
    registry_by_case = {row["case_id"]: row for row in registry_rows}
    selected = [case_id for case_id, _ in sorted(baseline.items(), key=lambda item: (
        -float(item[1]["measurement"]["exact_stage_seconds"]), item[0],
    ))[:case_limit]]
    if result.get("selected_case_ids") != selected:
        raise VerificationError("selected order differs")

    observed: dict[str, dict[str, Any]] = {}
    manifest: list[dict[str, str]] = [{"path": "RESULT.json", "sha256": _sha(result_raw)}]
    for relative in sorted(files - {"RESULT.json"}):
        value, raw = _read(root / relative)
        manifest.append({"path": relative, "sha256": _sha(raw)})
        case_id = value.get("registry_case_id")
        if type(case_id) is not str or case_id in observed or case_id not in selected:
            raise VerificationError("observed identity differs")
        semantic = baseline[case_id]["semantic"]
        registry_row = registry_by_case[case_id]
        if not all((value.get("query_episode_id") == semantic.get("query_id"),
                    value.get("proposal_result_digest") == semantic.get("forward_proposal", {}).get("result_digest"),
                    value.get("matches") == semantic.get("matches"),
                    value.get("certificate") == semantic.get("certificate"),
                    value.get("gate_passed") is True, value.get("status") == "completed",
                    value.get("query_symbol") == registry_row.get("symbol"),
                    value.get("query_cutoff") == registry_row.get("cutoff"),
                    value.get("query_stock_prefix") == registry_row.get("stock_prefix"),
                    value.get("query_benchmark_prefix") == registry_row.get("benchmark_prefix"),
                    value.get("real_forward_outcomes_accessed") is False,
                    value.get("result_digest") == _case_result_digest(value),
                    value.get("checkpoint_integrity_digest") == _case_integrity_digest(value),
                    all(_finite(value.get(key)) for key in (
                        "proposal_seconds", "amortized_proposal_seconds", "exact_seconds",
                        "final_exact_seconds", "peak_rss_mb")))):
            raise VerificationError(f"case reconstruction differs: {case_id}")
        if Path(relative).name != f"{value['query_episode_id']}.json":
            raise VerificationError("case filename differs")
        observed[case_id] = value

    groups = result.get("groups")
    if type(groups) is not list or len(groups) != processes:
        raise VerificationError("group count differs")
    group_ids: list[str] = []
    group_walls: list[float] = []
    for index, group in enumerate(groups):
        if type(group) is not dict or group.get("group_index") != index:
            raise VerificationError("group order differs")
        ids = group.get("query_episode_ids")
        if type(ids) is not list or not all(type(item) is str for item in ids):
            raise VerificationError("group coverage differs")
        if not all(_finite(group.get(key)) for key in ("elapsed_seconds", "peak_rss_mb", "proposal_seconds")):
            raise VerificationError("group measurement differs")
        group_ids.extend(ids)
        group_walls.append(float(group["elapsed_seconds"]))
    if len(group_ids) != case_limit or len(set(group_ids)) != case_limit \
            or set(group_ids) != {row["query_episode_id"] for row in observed.values()}:
        raise VerificationError("group query coverage differs")

    comparisons = result.get("comparisons")
    if type(comparisons) is not list or len(comparisons) != case_limit \
            or [row.get("case_id") for row in comparisons] != selected \
            or not all(row.get("passed") is True and all(row.get("gates", {}).values()) for row in comparisons):
        raise VerificationError("comparison coverage differs")
    serial_service = sum(
        float(baseline[case_id]["measurement"]["forward_proposal_seconds"])
        + float(baseline[case_id]["measurement"]["exact_stage_seconds"])
        for case_id in selected
    )
    expected_controls = {**CONTROLS, "processes": processes,
        "effective_cpus": list(range(8)), "grouping": "LPT-by-V3-exact-stage-seconds",
        "service_scans_per_query_group": 1,
        "reverse_scan_role": "separate-audit-not-service-throughput"}
    wall = result.get("wall_seconds")
    if not all((result.get("schema_version") == "m04r14-throughput-development-poc-v1",
                result.get("status") == "complete", result.get("development_only") is True,
                result.get("authority_or_outcome_accessed") is False,
                result.get("all_semantics_equal") is True, result.get("cases") == case_limit,
                result.get("baseline") == baseline_binding,
                result.get("registry_digest") == registry.get("registry_digest"),
                result.get("controls") == expected_controls, len(observed) == case_limit,
                result.get("baseline_serial_service_seconds") == serial_service,
                _finite(wall), float(wall) >= max(group_walls),
                float(wall) - max(group_walls) <= 120,
                result.get("speedup_vs_serial_service_sum") == serial_service / float(wall))):
        raise VerificationError("aggregate reconstruction differs")
    state = {
        "schema_version": SCHEMA, "status": "verified", "passed": True,
        "run_root": str(root), "run_result_digest": result["result_digest"],
        "run_result_sha256": _sha(result_raw), "verified_cases": case_limit,
        "processes": processes, "wall_seconds": wall,
        "maximum_group_wall_seconds": max(group_walls),
        "speedup_vs_serial_service_sum": result["speedup_vs_serial_service_sum"],
        "peak_reported_rss_mb": max(float(group["peak_rss_mb"]) for group in groups),
        "manifest": manifest, "manifest_digest": _digest(manifest),
        "all_semantics_equal": True,
    }
    state["result_digest"] = _digest(state)
    return state


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--case-limit", type=int, required=True)
    parser.add_argument("--processes", type=int, required=True)
    args = parser.parse_args(argv)
    state = verify(args.run_root, repository=args.repository,
                   case_limit=args.case_limit, processes=args.processes)
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
