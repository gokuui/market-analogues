from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import m04r14_shadow_run as producer
from experiments.m04r import verify_m04r14_shadow_interrupted as verifier


T0 = "2026-08-30T10:00:00+00:00"
T1 = "2026-08-31T05:00:00+00:00"
TC = "2026-08-31T17:00:00+00:00"


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _ledger_fixture(root: Path) -> tuple[list[dict], dict[str, dict]]:
    attempts = root / "attempts"
    attempts.mkdir(parents=True)
    cases = [{
        "case_id": f"case-{index:04d}", "episode_id": f"query-{index:04d}",
        "active_source_universe": 3000 + index % 17,
    } for index in range(3270)]
    assigned_cases = cases[1800:]
    groups = producer._groups(assigned_cases, 8)
    assigned = {str(row["episode_id"]) for group in groups for row in group}
    rows = {
        str(case["episode_id"]): {
            "created_at": "2026-08-30T20:00:00+00:00"
            if case["episode_id"] not in assigned else "2026-08-31T12:00:00+00:00"
        }
        for case in cases
    }
    _write(attempts / "0000-STARTED.json", {
        "schema_version": verifier.base.RESULT_SCHEMA, "attempt": 0,
        "completed_before": 0, "remaining_before": 3270,
        "recovered_before": 0, "created_at": T0,
    })
    _write(attempts / "0001-STARTED.json", {
        "schema_version": verifier.base.RESULT_SCHEMA, "attempt": 1,
        "completed_before": 1800, "remaining_before": 1470,
        "recovered_before": 1800, "created_at": T1,
    })
    _write(attempts / "0001-COMPLETED.json", {
        "schema_version": verifier.base.RESULT_SCHEMA, "attempt": 1,
        "elapsed_seconds": 40000.0, "completed_after": 3270,
        "remaining_after": 0, "worker_failures": [], "created_at": TC,
    })
    for index, group in enumerate(groups):
        ids = [str(row["episode_id"]) for row in group]
        _write(attempts / f"0001-GROUP-{index:02d}.json", {
            "schema_version": verifier.base.RESULT_SCHEMA, "attempt": 1,
            "group_index": index, "assigned_cases": len(ids),
            "promoted_cases": len(ids), "failure": None, "created_at": TC,
            "worker_result": {
                "query_episode_ids": ids, "proposal_seconds": 100.0,
                "elapsed_seconds": 200.0, "peak_rss_mb": 1000.0,
            },
        })
    return cases, rows


def test_interrupted_attempt_ledger_accepts_complete_deterministic_recovery(tmp_path: Path):
    cases, rows = _ledger_fixture(tmp_path)
    result = verifier._attempt_ledger(tmp_path, cases, rows)
    assert result["dangling_attempts"] == [0]
    assert result["completed_attempts"] == [1]
    assert result["recovered_cases"] == 1800
    assert result["resume_computed_cases"] == 1470


def test_interrupted_attempt_ledger_rejects_duplicate_assignment(tmp_path: Path):
    cases, rows = _ledger_fixture(tmp_path)
    path = tmp_path / "attempts/0001-GROUP-07.json"
    event = json.loads(path.read_text())
    event["worker_result"]["query_episode_ids"][0] = (
        json.loads((tmp_path / "attempts/0001-GROUP-00.json").read_text())
        ["worker_result"]["query_episode_ids"][0]
    )
    _write(path, event)
    with pytest.raises(verifier.InterruptedVerificationError, match="coverage"):
        verifier._attempt_ledger(tmp_path, cases, rows)


def test_interruption_can_never_publish_performance_success():
    result = verifier._performance_status(
        {"performance": {
            "snapshot_wall_seconds_max": 200000,
            "minimum_queries_per_second": 0.001,
            "worker_peak_rss_mib_max": 4096,
        }},
        {"created_at": T0},
        {"resume_elapsed_seconds": 40000.0},
        {
            "last_pre_interruption_observation": "2026-08-31T04:20:00+00:00",
            "resume_listener_started": T1,
            "observed_tree_swap_kib": 1,
        },
        3000.0,
    )
    assert result["gates"]["wall_time_bound"] is True
    assert result["gates"]["throughput_bound"] is True
    assert result["gates"]["case_reported_worker_rss"] is True
    assert result["gates"]["zero_observed_process_swap"] is False
    assert result["performance_qualified"] is False
    assert result["resource_qualified"] is False
