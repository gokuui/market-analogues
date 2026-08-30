"""Outcome/authority-blind diagnosis of a terminal untouched candidate run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.m04r import m04r14_untouched_candidate_contract as contract


SCHEMA = "m04r14-untouched-candidate-failure-diagnostic-v1"
OUTPUT_RELATIVE = Path("config/data/analogues/m04r14/untouched-candidate-v1-diagnostic")


class DiagnosticError(RuntimeError):
    pass


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(), parse_constant=lambda item: (_ for _ in ()).throw(
        DiagnosticError(f"non-finite JSON: {path}:{item}")))
    if type(value) is not dict: raise DiagnosticError(f"object required: {path}")
    return value


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0: return None
    return float(np.corrcoef(left, right)[0, 1])


def diagnose(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    root = repository / contract.CANDIDATE_RELATIVE
    result = _read(root / "RESULT.json")
    registry = _read(repository / contract.REGISTRY_RELATIVE / "query-registry.json")
    rows = [_read(path) for path in sorted((root / "cases").glob("*.json"))]
    if len(rows) != 72 or result.get("semantic_passed") is not True \
            or result.get("performance_passed") is not False \
            or result.get("authority_accessed") is not False \
            or result.get("real_forward_outcomes_accessed") is not False:
        raise DiagnosticError("terminal failed-performance candidate required")
    registry_rows = {str(row["episode_id"]): row for row in registry["cases_data"]}
    exact = [float(row["exact_seconds"]) for row in rows]
    evaluated = [float(row["certificate"]["exact_evaluated"]) for row in rows]
    active = [float(registry_rows[str(row["query_episode_id"])]["active_source_universe"])
              for row in rows]
    limits = contract.PERFORMANCE_LIMITS
    hard = sorted(rows, key=lambda row: (-float(row["exact_seconds"]), str(row["registry_case_id"])))
    roles: dict[str, list[float]] = {"current": [], "historical": []}
    for row in rows:
        role = str(registry_rows[str(row["query_episode_id"])]["cutoff_role"])
        roles[role].append(float(row["exact_seconds"]))
    groups = list(result["groups"]); group_times = [float(row["elapsed_seconds"]) for row in groups]
    state = {
        "schema_version": SCHEMA, "status": "diagnosed",
        "candidate_result_digest": result["result_digest"],
        "registry_digest": registry["registry_digest"], "cases": 72,
        "semantic_cases_passed": sum(row.get("gate_passed") is True for row in rows),
        "performance_failure": {
            "wall_seconds": float(result["wall_seconds"]),
            "wall_limit_seconds": limits["candidate_wall_seconds_max"],
            "wall_passed": float(result["wall_seconds"]) <= limits["candidate_wall_seconds_max"],
            "exact_p95_seconds": float(result["exact_seconds_p95"]),
            "exact_p95_limit_seconds": limits["case_exact_seconds_p95"],
            "exact_max_seconds": float(result["exact_seconds_max"]),
            "exact_max_limit_seconds": limits["case_exact_seconds_max"],
            "cases_over_p95_limit": sum(value > limits["case_exact_seconds_p95"] for value in exact),
            "cases_over_max_limit": sum(value > limits["case_exact_seconds_max"] for value in exact),
        },
        "work_relationships": {
            "exact_seconds_mean": mean(exact),
            "exact_evaluated_mean": mean(evaluated),
            "exact_seconds_vs_exact_evaluated_correlation": _correlation(exact, evaluated),
            "exact_seconds_vs_active_universe_correlation": _correlation(exact, active),
            "current_exact_seconds_mean": mean(roles["current"]),
            "historical_exact_seconds_mean": mean(roles["historical"]),
        },
        "group_balance": {"groups": len(groups), "minimum_seconds": min(group_times),
            "maximum_seconds": max(group_times), "maximum_over_minimum": max(group_times) / min(group_times),
            "proposal_seconds_minimum": min(float(row["proposal_seconds"]) for row in groups),
            "proposal_seconds_maximum": max(float(row["proposal_seconds"]) for row in groups)},
        "hardest_cases": [{"case_id": row["registry_case_id"],
            "query_id": row["query_episode_id"], "exact_seconds": float(row["exact_seconds"]),
            "exact_evaluated": int(row["certificate"]["exact_evaluated"]),
            "eligible_candidates": int(row["certificate"]["eligible_candidates"]),
            "frontier_limit_rows": int(row["frontier_limit_rows"])} for row in hard[:12]],
        "decision": {
            "authority_open_authorized": False, "candidate_retry_authorized": False,
            "cohort_reusable_as_untouched": False,
            "next_step": "exposed hard-case concurrency/kernel POCs, then a new untouched registry",
            "frozen_limits_changed": False,
        },
        "authority_accessed": False, "real_forward_outcomes_accessed": False,
    }
    return {**state, "result_digest": contract.digest(state)}


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink(): raise DiagnosticError("diagnostic root exists")
    path.mkdir(parents=False); descriptor = os.open(path / "DIAGNOSTIC.json",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json.dumps({**value, "created_at": datetime.now(timezone.utc).isoformat()},
            indent=2, sort_keys=True).encode() + b"\n"); handle.flush(); os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv); repository = args.repository.resolve(strict=True); value = diagnose(repository)
    if not args.dry_run: _publish(repository / OUTPUT_RELATIVE, value)
    print(json.dumps(value, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
