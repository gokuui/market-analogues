"""Throughput-first grouped replay of the exposed M04R-14 V3 corpus.

This development POC accepts no authority or outcome path.  It uses the
already-sealed V3 candidate cases as semantic controls, shares one forward
packed-universe scan across each process's query group, and runs certified
completion with the same 1K -> 16K policy as V3.  The reverse traversal used
to qualify scan invariance remains audit evidence; it is deliberately not
repeated in the measured service-throughput lane.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

from experiments.m04r import m04r11_build_authorities as grouped_engine
from experiments.m04r import m04r13_threaded_certified_exposed as m13
from experiments.m04r import m04r14_all60_contract as all60_contract


SCHEMA = "m04r14-throughput-development-poc-v1"
DEFAULT_BASELINE = Path(
    "config/data/analogues/m04r14/all60-certified-development-v3"
)
DEFAULT_OUTPUT = Path(
    "config/data/analogues/m04r14/throughput-development-poc-v1"
)
CONTROLS = {
    "block_rows": 4_096,
    "deferred_alignments": True,
    "exact_workers_per_process": 1,
    "initial_frontier_rows": 1_000,
    "maximum_frontier_rows": 16_384,
    "numba_threads_per_process": 1,
    "requested_positions": True,
    "seed_rows": 512,
    "sorted_joined_iqr_merge": True,
    "vector_lower_bounds": True,
}


class ThroughputError(RuntimeError):
    """Raised when the POC cannot prove exact V3 parity."""


def available_cpus() -> tuple[int, ...]:
    """Return the CPUs this process may actually schedule on."""
    if hasattr(os, "sched_getaffinity"):
        cpus = tuple(sorted(os.sched_getaffinity(0)))
    else:  # pragma: no cover - Linux is the production platform
        cpus = tuple(range(os.cpu_count() or 1))
    if not cpus:
        raise ThroughputError("no effective CPUs are available")
    return cpus


def _read(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ThroughputError(f"duplicate JSON key in {path}: {value}")

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                reject(key)
            result[key] = value
        return result

    value = json.loads(
        path.read_text(), object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ThroughputError(f"non-finite JSON in {path}: {value}")
        ),
    )
    if type(value) is not dict:
        raise ThroughputError(f"JSON object required: {path}")
    return value


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ThroughputError(f"create-only target exists: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(all60_contract.canonical_bytes(value) + b"\n")
    temporary.replace(path)


def _baseline_cases(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    complete = _read(root / "COMPLETE.json")
    semantics = _read(root / "SEMANTICS.json")
    measurements = _read(root / "MEASUREMENTS.json")
    if not all((
        complete.get("status") == "complete",
        complete.get("semantic_passed") is True,
        semantics.get("semantic_passed") is True,
        semantics.get("all_certified") is True,
        semantics.get("forward_reverse_equal") is True,
        measurements.get("status") == "complete",
        measurements.get("summary", {}).get("cases") == 60,
    )):
        raise ThroughputError("V3 semantic baseline is not complete")
    cases: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "cases").glob("*/CASE.json")):
        bundle = _read(path)
        semantic = bundle.get("semantic")
        measurement = bundle.get("measurement")
        if type(semantic) is not dict or type(measurement) is not dict:
            raise ThroughputError(f"baseline case shape differs: {path}")
        case_id = semantic.get("case_id")
        if type(case_id) is not str or case_id in cases:
            raise ThroughputError("baseline case IDs differ")
        cases[case_id] = {"semantic": semantic, "measurement": measurement}
    if len(cases) != 60:
        raise ThroughputError("baseline must contain exactly 60 cases")
    return cases, {
        "complete_digest": complete["complete_digest"],
        "semantic_digest": semantics["semantic_digest"],
        "measurement_digest": measurements["measurement_digest"],
        "serial_total_end_to_end_seconds": measurements["summary"][
            "total_end_to_end_seconds"
        ],
    }


def balanced_groups(
    cases: Sequence[dict[str, Any]], baseline: Mapping[str, dict[str, Any]],
    processes: int,
) -> tuple[tuple[dict[str, Any], ...], ...]:
    if not 1 <= processes <= len(cases):
        raise ThroughputError("process count differs")
    ordered = sorted(cases, key=lambda case: (
        -float(baseline[str(case["case_id"])]["measurement"]["exact_stage_seconds"]),
        str(case["case_id"]),
    ))
    groups: list[list[dict[str, Any]]] = [[] for _ in range(processes)]
    loads = [0.0] * processes
    for case in ordered:
        index = min(range(processes), key=lambda value: (
            loads[value], len(groups[value]), value,
        ))
        groups[index].append(case)
        loads[index] += float(
            baseline[str(case["case_id"])]["measurement"]["exact_stage_seconds"]
        )
    return tuple(tuple(group) for group in groups)


def compare_case(
    observed: Mapping[str, Any], baseline: Mapping[str, Any],
) -> dict[str, bool]:
    semantic = baseline["semantic"]
    return {
        "case_id_equal": observed.get("registry_case_id") == semantic.get("case_id"),
        "query_id_equal": observed.get("query_episode_id") == semantic.get("query_id"),
        "proposal_equal": observed.get("proposal_result_digest")
        == semantic.get("forward_proposal", {}).get("result_digest"),
        "matches_equal": observed.get("matches") == semantic.get("matches"),
        "certificate_equal": observed.get("certificate") == semantic.get("certificate"),
        "engine_gate_passed": observed.get("gate_passed") is True,
    }


def _run_group(
    repository: str, output_root: str, cases: tuple[dict[str, Any], ...],
    registry_digest: str,
) -> dict[str, Any]:
    repo = Path(repository)
    contract_state = {
        "schema_version": SCHEMA,
        "registry_digest": registry_digest,
        "controls": CONTROLS,
        "baseline_complete_digest": _read(
            repo / DEFAULT_BASELINE / "COMPLETE.json"
        )["complete_digest"],
    }
    contract = {
        **contract_state,
        "contract_digest": all60_contract.stable_digest(contract_state),
    }
    return grouped_engine._worker(
        str(repo / m13.CONFIG_RELATIVE), str(m13.RESIDENT_ROOT), output_root,
        m13.GENERATION_ID, contract, cases, CONTROLS,
    )


def execute(
    repository: Path, output_root: Path, *, processes: int | None = None,
    case_limit: int = 60,
) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    baseline_root = repository / DEFAULT_BASELINE
    baseline, baseline_binding = _baseline_cases(baseline_root)
    registry = _read(repository / m13.REGISTRY_RELATIVE / "query-registry.json")
    rows = m13._m12(repository)._validate_registry(registry)
    if type(rows) is not list or len(rows) != 60 \
            or registry.get("registry_digest") != m13.REGISTRY_DIGEST:
        raise ThroughputError("registry binding differs")
    by_case = {str(row["case_id"]): dict(row) for row in rows}
    if set(by_case) != set(baseline):
        raise ThroughputError("registry/baseline case set differs")
    if not 1 <= case_limit <= 60:
        raise ThroughputError("case limit differs")
    selected_ids = [
        case_id for case_id, _ in sorted(baseline.items(), key=lambda item: (
            -float(item[1]["measurement"]["exact_stage_seconds"]), item[0],
        ))[:case_limit]
    ]
    selected = [by_case[case_id] for case_id in selected_ids]
    cpus = available_cpus()
    requested_processes = len(cpus) if processes is None else processes
    if not 1 <= requested_processes <= len(cpus):
        raise ThroughputError("process count exceeds effective CPU capacity")
    groups = balanced_groups(selected, baseline, min(requested_processes, case_limit))
    output_root = output_root.absolute()
    if output_root.exists() or output_root.is_symlink():
        raise ThroughputError("output root must be absent")
    output_root.mkdir(parents=True)
    (output_root / "cases").mkdir()
    started = perf_counter()
    context = multiprocessing.get_context("spawn")
    results: list[dict[str, Any]] = []
    try:
        with ProcessPoolExecutor(max_workers=len(groups), mp_context=context) as pool:
            futures = {
                pool.submit(
                    _run_group, str(repository), str(output_root), group,
                    str(registry["registry_digest"]),
                ): index for index, group in enumerate(groups)
            }
            for future in as_completed(futures):
                result = future.result()
                results.append({"group_index": futures[future], **result})
    except BaseException as exc:
        failure = {
            "schema_version": SCHEMA, "status": "failed",
            "error_type": type(exc).__name__, "message": str(exc),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic(output_root / "FAILED.json", failure)
        raise
    observed: dict[str, dict[str, Any]] = {}
    for path in sorted((output_root / "cases").glob("*.json")):
        row = _read(path); case_id = str(row.get("registry_case_id"))
        if case_id in observed:
            raise ThroughputError("duplicate observed case")
        observed[case_id] = row
    comparisons = []
    for case_id in selected_ids:
        gates = compare_case(observed.get(case_id, {}), baseline[case_id])
        comparisons.append({"case_id": case_id, "gates": gates,
                            "passed": all(gates.values())})
    wall = perf_counter() - started
    baseline_service_seconds = sum(
        float(baseline[case_id]["measurement"]["forward_proposal_seconds"])
        + float(baseline[case_id]["measurement"]["exact_stage_seconds"])
        for case_id in selected_ids
    )
    result_state = {
        "schema_version": SCHEMA, "status": "complete",
        "development_only": True, "authority_or_outcome_accessed": False,
        "baseline": baseline_binding, "registry_digest": registry["registry_digest"],
        "controls": {**CONTROLS, "processes": len(groups),
                     "effective_cpus": list(cpus),
                     "grouping": "LPT-by-V3-exact-stage-seconds",
                     "service_scans_per_query_group": 1,
                     "reverse_scan_role": "separate-audit-not-service-throughput"},
        "selected_case_ids": selected_ids,
        "groups": sorted(results, key=lambda row: row["group_index"]),
        "comparisons": comparisons,
        "cases": len(selected_ids), "wall_seconds": wall,
        "baseline_serial_service_seconds": baseline_service_seconds,
        "speedup_vs_serial_service_sum": baseline_service_seconds / wall,
        "all_semantics_equal": all(row["passed"] for row in comparisons),
    }
    result_state["result_digest"] = all60_contract.stable_digest(result_state)
    result = {**result_state, "created_at": datetime.now(timezone.utc).isoformat()}
    _atomic(output_root / "RESULT.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--processes", type=int)
    parser.add_argument("--case-limit", type=int, default=60)
    args = parser.parse_args(argv)
    output = args.output_root or args.repository / DEFAULT_OUTPUT
    result = execute(args.repository, output, processes=args.processes,
                     case_limit=args.case_limit)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
