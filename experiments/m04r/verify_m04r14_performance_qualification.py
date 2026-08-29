"""Independent verifier for the M04R-14 performance qualification."""
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


SCHEMA = "m04r14-performance-qualification-verification-v1"
QUALIFICATION = Path("config/data/analogues/m04r14/performance-qualification-v1")
VERIFICATION = Path("config/data/analogues/m04r14/performance-qualification-v1-verification")
CONTRACT = Path("experiments/m04r/m04r14_performance_contract_v1.json")
PRODUCER_SOURCE = Path("experiments/m04r/m04r14_performance_qualification.py")
V3 = Path("config/data/analogues/m04r14/all60-certified-development-v3")
FULL_RUNS = (
    ("canonical", Path("config/data/analogues/m04r14/throughput-development-poc-v1")),
    ("repeat_2", Path("config/data/analogues/m04r14/throughput-performance-repeat-2-v1")),
    ("repeat_3", Path("config/data/analogues/m04r14/throughput-performance-repeat-3-v1")),
    ("resource_monitored", Path("config/data/analogues/m04r14/throughput-resource-monitored-v1")),
)
CONCURRENCY_RUNS = tuple((processes,
    Path(f"config/data/analogues/m04r14/performance-concurrency-p{processes}-hard8-v1"))
    for processes in (1, 2, 4, 8))
GROWTH = Path("config/data/analogues/m04r14/performance-growth-extension-hard12-v1")
FAULT = Path("config/data/analogues/m04r14/performance-operational-fault-v1/RESULT.json")
RESOURCE = Path("config/data/analogues/m04r14/performance-resource-monitor-v1/RESULT.json")
RESIDENT_SOURCE_SHAS = (
    "3c7c7815a80b830e6b1e3f77e3d61e21c6c33ba8e06dd8105a182685db53ac2c",
    "4131f1c47b120340dafec59ef59f93785343b1a8f5d4d9401e5e2f2bbbe05527",
)


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
        value = json.loads(raw, object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                VerificationError(f"non-finite JSON: {value}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSON: {path}") from exc
    if type(value) is not dict:
        raise VerificationError(f"JSON object required: {path}")
    return value, raw


def _digest(value: Any) -> str:
    return digest_contract.stable_digest(value)


def _sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _terminal_digest(value: Mapping[str, Any]) -> str:
    return _digest({key: item for key, item in value.items()
                    if key not in {"created_at", "result_digest"}})


def _git(repository: Path, *args: str) -> bytes:
    result = subprocess.run(["git", *args], cwd=repository,
        capture_output=True, check=False)
    if result.returncode:
        raise VerificationError(f"git validation failed: {' '.join(args)}")
    return result.stdout


def _run(repository: Path, root: Path, cases: int, processes: int) -> dict[str, Any]:
    verified = run_verifier.verify(repository / root, repository=repository,
        case_limit=cases, processes=processes)
    result, _ = _read(repository / root / "RESULT.json")
    group_walls = sorted(float(group["elapsed_seconds"]) for group in result["groups"])
    median = statistics.median(group_walls)
    return {"root": str(repository / root), "cases": cases, "processes": processes,
        "wall_seconds": verified["wall_seconds"],
        "speedup_vs_serial_service_sum": verified["speedup_vs_serial_service_sum"],
        "maximum_group_wall_seconds": verified["maximum_group_wall_seconds"],
        "median_group_wall_seconds": median,
        "maximum_group_wall_over_median": max(group_walls) / median,
        "peak_reported_rss_mb": verified["peak_reported_rss_mb"],
        "run_result_digest": verified["run_result_digest"],
        "verification_result_digest": verified["result_digest"],
        "manifest_digest": verified["manifest_digest"],
        "all_semantics_equal": verified["all_semantics_equal"]}


def _resident(value: Mapping[str, Any], source_sha: str) -> dict[str, Any]:
    observation = value.get("fresh_validation_observation")
    if type(observation) is not dict:
        raise VerificationError("resident observation differs")
    observation_input = {key: item for key, item in observation.items()
                         if key != "observation_digest"}
    keys = ("content_digest", "seal_digest", "ready_digest")
    seconds = observation.get("validation_seconds")
    if not all((value.get("ready") is True, value.get("mode") == "validate-existing",
                all(value.get(key) == observation.get(key) for key in keys),
                observation.get("observation_digest") == _digest(observation_input),
                type(seconds) in {int, float}, math.isfinite(float(seconds)), seconds >= 0)):
        raise VerificationError("resident reconstruction differs")
    return {"sha256": source_sha, **{key: value[key] for key in keys},
        "validation_seconds": seconds,
        "observation_digest": observation["observation_digest"]}


def verify(root: Path, *, repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    root = root.resolve(strict=True)
    if root != (repository / QUALIFICATION).resolve(strict=True):
        raise VerificationError("qualification root differs")
    entries = list(root.iterdir())
    if {entry.name for entry in entries} != {"RESULT.json", "RESIDENT_1.json", "RESIDENT_2.json"} \
            or any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise VerificationError("qualification tree differs")
    result, result_raw = _read(root / "RESULT.json")
    if result.get("result_digest") != _terminal_digest(result):
        raise VerificationError("qualification result digest differs")
    marker_head = result.get("git_head")
    if type(marker_head) is not str or len(marker_head) != 40:
        raise VerificationError("qualification Git head differs")
    _git(repository, "cat-file", "-e", f"{marker_head}^{{commit}}")
    if _git(repository, "show", f"{marker_head}:{PRODUCER_SOURCE.as_posix()}") \
            != (repository / PRODUCER_SOURCE).read_bytes():
        raise VerificationError("qualification producer source differs")
    contract, contract_raw = _read(repository / CONTRACT)
    full = [{"label": label, **_run(repository, run_root, 60, 8)}
            for label, run_root in FULL_RUNS]
    concurrency = [_run(repository, run_root, 8, processes)
                   for processes, run_root in CONCURRENCY_RUNS]
    growth = _run(repository, GROWTH, 12, 8)
    end_to_end: list[float] = []
    for path in sorted((repository / V3 / "cases").glob("*/CASE.json")):
        case, _ = _read(path)
        end_to_end.append(float(case["measurement"]["end_to_end_seconds"]))
    if len(end_to_end) != 60:
        raise VerificationError("interactive case count differs")
    ordered = sorted(end_to_end)
    interactive = {"cases": 60, "p95_method": "nearest-rank-ceiling",
        "end_to_end_seconds_p95": ordered[math.ceil(.95 * len(ordered)) - 1],
        "end_to_end_seconds_max": max(ordered)}
    stability_walls = [row["wall_seconds"] for row in full[:3]]
    stability = {"runs": len(stability_walls), "wall_seconds": stability_walls,
        "mean_wall_seconds": statistics.mean(stability_walls),
        "population_coefficient_of_variation": statistics.pstdev(stability_walls) / statistics.mean(stability_walls),
        "worst_to_best_wall_ratio": max(stability_walls) / min(stability_walls)}
    residents: list[dict[str, Any]] = []
    resident_manifest: list[dict[str, str]] = []
    for index, source_sha in enumerate(RESIDENT_SOURCE_SHAS, 1):
        value, raw = _read(root / f"RESIDENT_{index}.json")
        residents.append(_resident(value, source_sha))
        resident_manifest.append({"path": f"RESIDENT_{index}.json", "sha256": _sha(raw)})
    fault, fault_raw = _read(repository / FAULT)
    resource, resource_raw = _read(repository / RESOURCE)
    if fault.get("result_digest") != _terminal_digest(fault) \
            or resource.get("result_digest") != _terminal_digest(resource):
        raise VerificationError("operational evidence differs")
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
        "resident_run_count": len(residents) >= operational_contract["resident_validate_existing_runs_required"],
        "resident_digest_equality": len({(row["content_digest"], row["seal_digest"], row["ready_digest"]) for row in residents}) == 1,
        "forced_termination": fault.get("passed") is True and fault.get("forced_signal") == "SIGKILL",
        "same_root_non_resumable": fault.get("same_root_retry_rejected") is True,
        "clean_restart_verified": fault.get("restart_verification", {}).get("passed") is True,
        "resource_monitor_verified": resource.get("passed") is True and all(resource.get("gates", {}).values()),
        "zero_tree_swap": resource.get("maximum_tree_swap_kib") == stability_contract["process_swap_bytes_max"],
        "zero_oom": resource.get("memory_events_delta", {}).get("oom") == stability_contract["oom_events_max"] and resource.get("memory_events_delta", {}).get("oom_kill") == 0}
    expected = {"schema_version": "m04r14-performance-qualification-v1",
        "status": "complete", "passed": all(gates.values()), "git_head": marker_head,
        "contract_sha256": _sha(contract_raw),
        "contract_schema_version": contract["schema_version"], "interactive": interactive,
        "full_runs": full, "stability": stability, "concurrency": concurrency,
        "growth_extension": growth, "conservative_72_wall_seconds": conservative_72,
        "resident_observations": residents,
        "operational_fault_result_digest": fault["result_digest"],
        "operational_fault_sha256": _sha(fault_raw),
        "resource_monitor_result_digest": resource["result_digest"],
        "resource_monitor_sha256": _sha(resource_raw), "gates": gates,
        "production_promotion_authorized": False,
        "next_gate": "t14-06-new-untouched-registry"}
    expected["result_digest"] = _digest(expected)
    observed = {key: item for key, item in result.items() if key != "created_at"}
    if observed != expected or not expected["passed"]:
        raise VerificationError("qualification reconstruction differs")
    state = {"schema_version": SCHEMA, "status": "verified", "passed": True,
        "qualification_root": str(root), "qualification_result_digest": result["result_digest"],
        "qualification_result_sha256": _sha(result_raw),
        "resident_manifest": resident_manifest,
        "resident_manifest_digest": _digest(resident_manifest),
        "verified_full_runs": len(full), "verified_concurrency_runs": len(concurrency),
        "verified_interactive_cases": 60, "verified_growth_cases": 12,
        "all_gates_passed": True, "production_promotion_authorized": False}
    state["result_digest"] = _digest(state)
    return state


def _publish(path: Path, state: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise VerificationError("verification root exists")
    path.mkdir(parents=False)
    target = path / "VERIFIED.json"
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(digest_contract.canonical_bytes({**state,
            "created_at": datetime.now(timezone.utc).isoformat()}) + b"\n")
        handle.flush(); os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    repository = args.repository.resolve(strict=True)
    state = verify(repository / QUALIFICATION, repository=repository)
    if not args.dry_run:
        _publish(repository / VERIFICATION, state)
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
