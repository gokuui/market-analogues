from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable, Sequence

import yaml

from .config import load_config


@dataclass(frozen=True)
class MatrixCase:
    dataset_id: str
    case_id: str
    episode_id: str


@dataclass(frozen=True)
class MatrixRunnerResult:
    passed: bool
    completed_cases: int
    required_cases: int
    failed_case: str | None
    progress_path: Path


CommandRunner = Callable[[Sequence[str]], int]


def authority_matrix_cases(
    config_path: Path, datasets: tuple[str, ...],
) -> tuple[MatrixCase, ...]:
    config = load_config(config_path)
    cases: list[MatrixCase] = []
    for dataset_id in datasets:
        if dataset_id not in config.datasets:
            raise ValueError(f"unknown dataset {dataset_id!r}")
        path = config.artifact_dir / "gate12" / dataset_id / "query-registry.yaml"
        payload = yaml.safe_load(path.read_text()) or {}
        records = payload.get("cases_data")
        if not isinstance(records, list) or len(records) != 12:
            raise ValueError(f"{dataset_id} registry must contain exactly 12 cases")
        seen: set[str] = set()
        for record in records:
            case_id = str(record["case_id"])
            if case_id in seen:
                raise ValueError(f"duplicate authority case {case_id}")
            seen.add(case_id)
            cases.append(MatrixCase(dataset_id, case_id, str(record["episode_id"])))
    return tuple(cases)


def _default_command_runner(command: Sequence[str]) -> int:
    return subprocess.run(list(command), check=False).returncode


def _write_progress(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def run_authority_matrix(
    config_path: Path,
    datasets: tuple[str, ...] = ("nse", "nasdaq"),
    *,
    disk_reserve_bytes: int = 5 * 1024 ** 3,
    maximum_rss_mb: float = 1024,
    command_runner: CommandRunner = _default_command_runner,
) -> MatrixRunnerResult:
    if disk_reserve_bytes < 0 or maximum_rss_mb <= 0:
        raise ValueError("resource limits must be non-negative and positive")
    config_path = config_path.resolve()
    config = load_config(config_path)
    cases = authority_matrix_cases(config_path, datasets)
    progress_path = (
        config.artifact_dir / "gate12" / "authorities" / "matrix-runner.json"
    )
    now = datetime.now(timezone.utc).isoformat()
    payload: dict[str, object] = {
        "schema_version": "gate12-matrix-runner-v1",
        "config": str(config_path),
        "datasets": list(datasets),
        "required_cases": len(cases),
        "completed_cases": 0,
        "status": "running",
        "started_at": now,
        "updated_at": now,
        "disk_reserve_bytes": disk_reserve_bytes,
        "maximum_rss_mb": maximum_rss_mb,
        "cases": {
            case.case_id: {
                "dataset": case.dataset_id,
                "episode_id": case.episode_id,
                "status": "pending",
            }
            for case in cases
        },
        "aggregates": {dataset_id: "pending" for dataset_id in datasets},
    }
    _write_progress(progress_path, payload)
    completed = 0
    case_records = payload["cases"]
    assert isinstance(case_records, dict)
    for dataset_id in datasets:
        for case in (item for item in cases if item.dataset_id == dataset_id):
            free = shutil.disk_usage(config.artifact_dir).free
            record = case_records[case.case_id]
            assert isinstance(record, dict)
            if free < disk_reserve_bytes:
                record["status"] = "blocked_disk_reserve"
                record["free_disk_bytes"] = free
                payload["status"] = "blocked"
                payload["failed_case"] = case.case_id
                payload["updated_at"] = datetime.now(timezone.utc).isoformat()
                _write_progress(progress_path, payload)
                print(
                    f"BLOCKED {case.case_id}: free disk {free} below reserve "
                    f"{disk_reserve_bytes}", flush=True,
                )
                return MatrixRunnerResult(
                    False, completed, len(cases), case.case_id, progress_path,
                )
            record.update({
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "free_disk_bytes_before": free,
            })
            payload["current_case"] = case.case_id
            payload["updated_at"] = datetime.now(timezone.utc).isoformat()
            _write_progress(progress_path, payload)
            print(
                f"START {completed + 1}/{len(cases)} {case.case_id} "
                f"free_disk={free}", flush=True,
            )
            command = (
                sys.executable, "-m", "market_analogues.cli",
                "build-gate12-authority", "--config", str(config_path),
                "--dataset", dataset_id, "--case-id", case.case_id,
                "--maximum-rss-mb", str(maximum_rss_mb),
            )
            exit_code = command_runner(command)
            record["exit_code"] = exit_code
            record["finished_at"] = datetime.now(timezone.utc).isoformat()
            record["free_disk_bytes_after"] = shutil.disk_usage(config.artifact_dir).free
            if exit_code != 0:
                record["status"] = "failed"
                payload["status"] = "failed"
                payload["failed_case"] = case.case_id
                payload["updated_at"] = datetime.now(timezone.utc).isoformat()
                _write_progress(progress_path, payload)
                print(f"FAILED {case.case_id} exit={exit_code}", flush=True)
                return MatrixRunnerResult(
                    False, completed, len(cases), case.case_id, progress_path,
                )
            completed += 1
            record["status"] = "passed"
            payload["completed_cases"] = completed
            payload["updated_at"] = datetime.now(timezone.utc).isoformat()
            _write_progress(progress_path, payload)
            print(f"PASS {completed}/{len(cases)} {case.case_id}", flush=True)
        aggregate = (
            sys.executable, "-m", "market_analogues.cli",
            "aggregate-gate12-authorities", "--config", str(config_path),
            "--dataset", dataset_id,
        )
        exit_code = command_runner(aggregate)
        aggregates = payload["aggregates"]
        assert isinstance(aggregates, dict)
        aggregates[dataset_id] = "passed" if exit_code == 0 else "failed"
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_progress(progress_path, payload)
        if exit_code != 0:
            payload["status"] = "failed"
            payload["failed_case"] = f"aggregate:{dataset_id}"
            _write_progress(progress_path, payload)
            return MatrixRunnerResult(
                False, completed, len(cases), f"aggregate:{dataset_id}", progress_path,
            )
    payload.pop("current_case", None)
    payload["status"] = "passed"
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_progress(progress_path, payload)
    return MatrixRunnerResult(True, completed, len(cases), None, progress_path)
