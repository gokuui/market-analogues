"""Aggregate the frozen M04R-14 performance qualification from raw evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
from typing import Any, Mapping, Sequence

from experiments.m04r import m04r14_all60_contract as digest_contract
from experiments.m04r import verify_m04r14_performance_run as run_verifier


SCHEMA = "m04r14-performance-qualification-v1"
CONTRACT = Path("experiments/m04r/m04r14_performance_contract_v1.json")
OUTPUT = Path("config/data/analogues/m04r14/performance-qualification-v1")
V3 = Path("config/data/analogues/m04r14/all60-certified-development-v3")
FULL_RUNS = (
    ("canonical", Path("config/data/analogues/m04r14/throughput-development-poc-v1")),
    ("repeat_2", Path("config/data/analogues/m04r14/throughput-performance-repeat-2-v1")),
    ("repeat_3", Path("config/data/analogues/m04r14/throughput-performance-repeat-3-v1")),
    ("resource_monitored", Path("config/data/analogues/m04r14/throughput-resource-monitored-v1")),
)
CONCURRENCY_RUNS = tuple(
    (processes, Path(f"config/data/analogues/m04r14/performance-concurrency-p{processes}-hard8-v1"))
    for processes in (1, 2, 4, 8)
)
GROWTH = Path("config/data/analogues/m04r14/performance-growth-extension-hard12-v1")
FAULT = Path("config/data/analogues/m04r14/performance-operational-fault-v1/RESULT.json")
RESOURCE = Path("config/data/analogues/m04r14/performance-resource-monitor-v1/RESULT.json")


class QualificationError(RuntimeError):
    pass


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise QualificationError(f"regular file required: {path}")
    raw = path.read_bytes()
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise QualificationError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                QualificationError(f"non-finite JSON: {value}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationError(f"invalid JSON: {path}") from exc
    if type(value) is not dict:
        raise QualificationError(f"JSON object required: {path}")
    return value, raw


def _sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _digest(value: Any) -> str:
    return digest_contract.stable_digest(value)


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise QualificationError(f"create-only target exists: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(digest_contract.canonical_bytes(value) + b"\n")
        handle.flush(); os.fsync(handle.fileno())


def _terminal_digest(value: Mapping[str, Any]) -> str:
    return _digest({key: item for key, item in value.items()
                    if key not in {"created_at", "result_digest"}})


def _run(repository: Path, root: Path, cases: int, processes: int) -> dict[str, Any]:
    verified = run_verifier.verify(
        repository / root, repository=repository, case_limit=cases, processes=processes,
    )
    result, _ = _read(repository / root / "RESULT.json")
    group_walls = sorted(float(group["elapsed_seconds"]) for group in result["groups"])
    median = statistics.median(group_walls)
    return {
        "root": str(repository / root), "cases": cases, "processes": processes,
        "wall_seconds": verified["wall_seconds"],
        "speedup_vs_serial_service_sum": verified["speedup_vs_serial_service_sum"],
        "maximum_group_wall_seconds": verified["maximum_group_wall_seconds"],
        "median_group_wall_seconds": median,
        "maximum_group_wall_over_median": max(group_walls) / median,
        "peak_reported_rss_mb": verified["peak_reported_rss_mb"],
        "run_result_digest": verified["run_result_digest"],
        "verification_result_digest": verified["result_digest"],
        "manifest_digest": verified["manifest_digest"],
        "all_semantics_equal": verified["all_semantics_equal"],
    }


def _resident(value: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    observation = value.get("fresh_validation_observation")
    if type(observation) is not dict:
        raise QualificationError("resident observation differs")
    keys = ("content_digest", "seal_digest", "ready_digest")
    if value.get("ready") is not True or value.get("mode") != "validate-existing" \
            or any(value.get(key) != observation.get(key) for key in keys):
        raise QualificationError("resident binding differs")
    seconds = observation.get("validation_seconds")
    if type(seconds) not in {int, float} or not math.isfinite(float(seconds)) or seconds < 0:
        raise QualificationError("resident timing differs")
    return {"sha256": _sha(raw), **{key: value[key] for key in keys},
            "validation_seconds": seconds,
            "observation_digest": observation.get("observation_digest")}


def execute(repository: Path, resident_paths: Sequence[Path]) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    output = repository / OUTPUT
    if output.exists() or output.is_symlink():
        raise QualificationError("qualification root exists")
    if len(resident_paths) != 2:
        raise QualificationError("exactly two resident observations required")
    if subprocess.run(["git", "status", "--porcelain"], cwd=repository,
            text=True, capture_output=True, check=True).stdout:
        raise QualificationError("qualification requires clean Git")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository,
        text=True, capture_output=True, check=True).stdout.strip()
    contract, contract_raw = _read(repository / CONTRACT)
    full = [{"label": label, **_run(repository, root, 60, 8)}
            for label, root in FULL_RUNS]
    concurrency = [_run(repository, root, 8, processes)
                   for processes, root in CONCURRENCY_RUNS]
    growth = _run(repository, GROWTH, 12, 8)

    end_to_end: list[float] = []
    for path in sorted((repository / V3 / "cases").glob("*/CASE.json")):
        case, _ = _read(path)
        end_to_end.append(float(case["measurement"]["end_to_end_seconds"]))
    if len(end_to_end) != 60:
        raise QualificationError("interactive case count differs")
    ordered = sorted(end_to_end)
    interactive = {"cases": 60, "p95_method": "nearest-rank-ceiling",
        "end_to_end_seconds_p95": ordered[math.ceil(.95 * len(ordered)) - 1],
        "end_to_end_seconds_max": max(ordered)}

    stability_walls = [row["wall_seconds"] for row in full[:3]]
    stability = {
        "runs": len(stability_walls), "wall_seconds": stability_walls,
        "mean_wall_seconds": statistics.mean(stability_walls),
        "population_coefficient_of_variation":
            statistics.pstdev(stability_walls) / statistics.mean(stability_walls),
        "worst_to_best_wall_ratio": max(stability_walls) / min(stability_walls),
    }
    resident_values: list[dict[str, Any]] = []
    resident_raw: list[bytes] = []
    for path in resident_paths:
        value, raw = _read(path.resolve(strict=True))
        resident_values.append(_resident(value, raw)); resident_raw.append(raw)
    fault, fault_raw = _read(repository / FAULT)
    resource, resource_raw = _read(repository / RESOURCE)
    if fault.get("result_digest") != _terminal_digest(fault) \
            or resource.get("result_digest") != _terminal_digest(resource):
        raise QualificationError("operational evidence digest differs")

    interactive_contract = contract["interactive_lane"]
    batch_contract = contract["batch_lane"]
    stability_contract = contract["stability_lane"]
    operational_contract = contract["operational_lane"]
    concurrency_walls = [row["wall_seconds"] for row in concurrency]
    conservative_72 = max(stability_walls) + growth["wall_seconds"]
    gates = {
        "interactive_case_count": interactive["cases"] == interactive_contract["case_count"],
        "interactive_p95": interactive["end_to_end_seconds_p95"] <= interactive_contract["end_to_end_seconds_p95_max"],
        "interactive_max": interactive["end_to_end_seconds_max"] <= interactive_contract["end_to_end_seconds_max"],
        "all_full_run_semantics": all(row["all_semantics_equal"] for row in full),
        "all_full_run_walls": all(row["wall_seconds"] <= batch_contract["sixty_case_wall_seconds_max"] for row in full),
        "all_full_run_speedups": all(row["speedup_vs_serial_service_sum"] >= batch_contract["speedup_vs_serial_service_sum_min"] for row in full),
        "all_group_balance": all(row["maximum_group_wall_over_median"] <= batch_contract["maximum_group_wall_over_median_max"] for row in full),
        "stability_run_count": stability["runs"] >= stability_contract["complete_sixty_case_runs_required"],
        "stability_cv": stability["population_coefficient_of_variation"] <= stability_contract["wall_seconds_coefficient_of_variation_max"],
        "stability_ratio": stability["worst_to_best_wall_ratio"] <= stability_contract["worst_to_best_wall_ratio_max"],
        "concurrency_matrix_complete": [row["processes"] for row in concurrency] == contract["required_concurrency_matrix"],
        "concurrency_semantics": all(row["all_semantics_equal"] for row in concurrency),
        "concurrency_monotonic_speedup": all(a > b for a, b in zip(concurrency_walls, concurrency_walls[1:])),
        "growth_semantics": growth["all_semantics_equal"],
        "growth_72_wall": conservative_72 <= batch_contract["seventy_two_load_case_wall_seconds_max"],
        "resident_run_count": len(resident_values) >= operational_contract["resident_validate_existing_runs_required"],
        "resident_digest_equality": len({(row["content_digest"], row["seal_digest"], row["ready_digest"]) for row in resident_values}) == 1,
        "forced_termination": fault.get("passed") is True and fault.get("forced_signal") == "SIGKILL",
        "same_root_non_resumable": fault.get("same_root_retry_rejected") is True,
        "clean_restart_verified": fault.get("restart_verification", {}).get("passed") is True,
        "resource_monitor_verified": resource.get("passed") is True and all(resource.get("gates", {}).values()),
        "zero_tree_swap": resource.get("maximum_tree_swap_kib") == stability_contract["process_swap_bytes_max"],
        "zero_oom": resource.get("memory_events_delta", {}).get("oom") == stability_contract["oom_events_max"] and resource.get("memory_events_delta", {}).get("oom_kill") == 0,
    }
    state = {
        "schema_version": SCHEMA, "status": "complete", "passed": all(gates.values()),
        "git_head": head, "contract_sha256": _sha(contract_raw),
        "contract_schema_version": contract["schema_version"],
        "interactive": interactive, "full_runs": full, "stability": stability,
        "concurrency": concurrency, "growth_extension": growth,
        "conservative_72_wall_seconds": conservative_72,
        "resident_observations": resident_values,
        "operational_fault_result_digest": fault["result_digest"],
        "operational_fault_sha256": _sha(fault_raw),
        "resource_monitor_result_digest": resource["result_digest"],
        "resource_monitor_sha256": _sha(resource_raw), "gates": gates,
        "production_promotion_authorized": False,
        "next_gate": "t14-06-new-untouched-registry",
    }
    state["result_digest"] = _digest(state)
    output.mkdir(parents=True)
    _atomic(output / "RESIDENT_1.json", json.loads(resident_raw[0]))
    _atomic(output / "RESIDENT_2.json", json.loads(resident_raw[1]))
    _atomic(output / "RESULT.json", {**state,
        "created_at": datetime.now(timezone.utc).isoformat()})
    if not state["passed"]:
        raise QualificationError("performance qualification failed")
    return state


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--resident-observation", type=Path, action="append", required=True)
    args = parser.parse_args(argv)
    state = execute(args.repository, args.resident_observation)
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
