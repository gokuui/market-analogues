"""Continuously monitor swap/OOM resources during a full M04R-14 P8 run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from time import monotonic, sleep
from typing import Any, Mapping, Sequence

from experiments.m04r import m04r14_all60_contract as contract
from experiments.m04r import verify_m04r14_performance_run as run_verifier


SCHEMA = "m04r14-resource-monitored-run-v1"
RUN_ROOT = Path("config/data/analogues/m04r14/throughput-resource-monitored-v1")
EVIDENCE_ROOT = Path("config/data/analogues/m04r14/performance-resource-monitor-v1")
PERFORMANCE_CONTRACT = Path("experiments/m04r/m04r14_performance_contract_v1.json")
SAMPLE_INTERVAL_SECONDS = 0.25


class ResourceMonitorError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ResourceMonitorError(f"regular file required: {path}")
    return sha256(path.read_bytes()).hexdigest()


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ResourceMonitorError(f"create-only target exists: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(contract.canonical_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repository, text=True, capture_output=True, check=True,
    )
    return result.stdout.strip()


def _direct_children(pid: int) -> tuple[int, ...]:
    path = Path(f"/proc/{pid}/task/{pid}/children")
    try:
        content = path.read_text().strip()
    except (FileNotFoundError, ProcessLookupError):
        return ()
    return tuple(int(value) for value in content.split()) if content else ()


def _process_tree(root_pid: int) -> tuple[int, ...]:
    pending = [root_pid]
    observed: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in observed:
            continue
        observed.add(pid)
        pending.extend(_direct_children(pid))
    return tuple(sorted(observed))


def _status_kib(pid: int) -> dict[str, int] | None:
    path = Path(f"/proc/{pid}/status")
    try:
        lines = path.read_text().splitlines()
    except (FileNotFoundError, ProcessLookupError):
        return None
    result = {"VmRSS": 0, "VmHWM": 0, "VmSwap": 0}
    for line in lines:
        key, separator, rest = line.partition(":")
        if separator and key in result:
            parts = rest.split()
            if len(parts) != 2 or parts[1] != "kB":
                raise ResourceMonitorError(f"unexpected status unit for PID {pid}: {line}")
            result[key] = int(parts[0])
    return result


def _cgroup_root() -> Path | None:
    try:
        rows = Path("/proc/self/cgroup").read_text().splitlines()
    except FileNotFoundError:
        return None
    unified = [row.split("::", 1)[1] for row in rows if row.startswith("0::")]
    if len(unified) != 1:
        return None
    return Path("/sys/fs/cgroup") / unified[0].lstrip("/")


def _counter_file(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    result: dict[str, int] = {}
    for line in path.read_text().splitlines():
        key, value = line.split()
        result[key] = int(value)
    return result


def _counter_delta(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
    keys = set(before) | set(after)
    result = {key: after.get(key, 0) - before.get(key, 0) for key in sorted(keys)}
    if any(value < 0 for value in result.values()):
        raise ResourceMonitorError("cgroup counter decreased")
    return result


def _command(repository: Path, output: Path) -> list[str]:
    return [
        str(repository / ".venv/bin/python"), "-m",
        "experiments.m04r.m04r14_throughput_poc", "--repository", str(repository),
        "--output-root", str(output), "--processes", "8", "--case-limit", "60",
    ]


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    run_root = repository / RUN_ROOT
    evidence_root = repository / EVIDENCE_ROOT
    if any(path.exists() or path.is_symlink() for path in (run_root, evidence_root)):
        raise ResourceMonitorError("resource-monitor roots must be absent")
    if _git(repository, "status", "--porcelain"):
        raise ResourceMonitorError("resource-monitored run requires clean Git")
    head = _git(repository, "rev-parse", "HEAD")
    cgroup = _cgroup_root()
    memory_before = _counter_file(cgroup / "memory.events") if cgroup else {}
    swap_before = _counter_file(cgroup / "memory.swap.events") if cgroup else {}
    command = _command(repository, run_root)
    launched_at = datetime.now(timezone.utc).isoformat()
    process = subprocess.Popen(
        command, cwd=repository, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, start_new_session=True,
    )
    started = monotonic()
    samples = 0
    observed_pids: set[int] = set()
    maximum_tree_rss_kib = 0
    maximum_individual_rss_kib = 0
    maximum_tree_swap_kib = 0
    maximum_individual_swap_kib = 0
    while process.poll() is None:
        tree_rss = 0
        tree_swap = 0
        for pid in _process_tree(process.pid):
            status = _status_kib(pid)
            if status is None:
                continue
            observed_pids.add(pid)
            tree_rss += status["VmRSS"]
            tree_swap += status["VmSwap"]
            maximum_individual_rss_kib = max(maximum_individual_rss_kib, status["VmRSS"])
            maximum_individual_swap_kib = max(maximum_individual_swap_kib, status["VmSwap"])
        maximum_tree_rss_kib = max(maximum_tree_rss_kib, tree_rss)
        maximum_tree_swap_kib = max(maximum_tree_swap_kib, tree_swap)
        samples += 1
        sleep(SAMPLE_INTERVAL_SECONDS)
    stdout, stderr = process.communicate()
    monitored_seconds = monotonic() - started
    memory_after = _counter_file(cgroup / "memory.events") if cgroup else {}
    swap_after = _counter_file(cgroup / "memory.swap.events") if cgroup else {}
    memory_delta = _counter_delta(memory_before, memory_after)
    swap_delta = _counter_delta(swap_before, swap_after)
    if process.returncode != 0:
        raise ResourceMonitorError(f"monitored producer failed: {stderr[-2000:]}")
    verified = run_verifier.verify(
        run_root, repository=repository, case_limit=60, processes=8,
    )
    gates = {
        "run_verified": verified.get("passed") is True,
        "all_semantics_equal": verified.get("all_semantics_equal") is True,
        "maximum_tree_swap_zero": maximum_tree_swap_kib == 0,
        "maximum_individual_swap_zero": maximum_individual_swap_kib == 0,
        "cgroup_oom_zero": memory_delta.get("oom", 0) == 0,
        "cgroup_oom_kill_zero": memory_delta.get("oom_kill", 0) == 0,
        "cgroup_swap_fail_zero": swap_delta.get("fail", 0) == 0,
        "enough_samples": samples >= monitored_seconds / SAMPLE_INTERVAL_SECONDS * 0.90,
    }
    state = {
        "schema_version": SCHEMA, "status": "complete", "passed": all(gates.values()),
        "git_head": head, "contract_sha256": _sha(repository / PERFORMANCE_CONTRACT),
        "launched_at": launched_at, "command": command,
        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS, "samples": samples,
        "monitored_seconds": monitored_seconds, "observed_processes": len(observed_pids),
        "maximum_tree_rss_kib": maximum_tree_rss_kib,
        "maximum_individual_rss_kib": maximum_individual_rss_kib,
        "maximum_tree_swap_kib": maximum_tree_swap_kib,
        "maximum_individual_swap_kib": maximum_individual_swap_kib,
        "cgroup_path": str(cgroup) if cgroup else None,
        "memory_events_before": memory_before, "memory_events_after": memory_after,
        "memory_events_delta": memory_delta, "swap_events_before": swap_before,
        "swap_events_after": swap_after, "swap_events_delta": swap_delta,
        "producer_stdout_sha256": sha256(stdout.encode()).hexdigest(),
        "producer_stderr": stderr, "run_verification": verified, "gates": gates,
        "production_promotion_authorized": False,
    }
    state["result_digest"] = contract.stable_digest(state)
    evidence_root.mkdir(parents=True)
    _atomic(evidence_root / "RESULT.json", {
        **state, "created_at": datetime.now(timezone.utc).isoformat(),
    })
    if not state["passed"]:
        raise ResourceMonitorError("resource gates failed")
    return state


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    state = execute(parser.parse_args(argv).repository)
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
