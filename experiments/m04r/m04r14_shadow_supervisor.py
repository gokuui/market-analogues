"""Run, continuously monitor and independently verify the T14-08 snapshot."""
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

from experiments.m04r import m04r14_resource_monitored_run as monitor
from experiments.m04r import m04r14_shadow_run as runner
from experiments.m04r import verify_m04r14_shadow_run as verifier
from market_analogues.types import stable_hash


SCHEMA = "m04r14-shadow-supervision-v1"
EVIDENCE = Path("config/data/analogues/m04r14/nasdaq-shadow-supervision-v1")
SAMPLE_SECONDS = 0.25


class ShadowSupervisorError(RuntimeError):
    pass


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ShadowSupervisorError(f"create-only target exists: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json.dumps(
            value, indent=2, sort_keys=True, allow_nan=False,
        ).encode() + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=repository,
        text=True, capture_output=True, check=True,
    ).stdout:
        raise ShadowSupervisorError("supervised shadow requires clean Git")
    evidence = repository / EVIDENCE
    if evidence.exists() or evidence.is_symlink():
        raise ShadowSupervisorError("supervision evidence root already exists")
    evidence.mkdir(parents=True)
    stdout_path = evidence / "producer.stdout.log"
    stderr_path = evidence / "producer.stderr.log"
    command = [
        str(repository / ".venv/bin/python"), "-m",
        "experiments.m04r.m04r14_shadow_run", "run",
        "--repository", str(repository),
    ]
    cgroup = monitor._cgroup_root()
    memory_before = monitor._counter_file(cgroup / "memory.events") if cgroup else {}
    swap_before = monitor._counter_file(cgroup / "memory.swap.events") if cgroup else {}
    launched_at = datetime.now(timezone.utc).isoformat()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            command, cwd=repository, stdout=stdout, stderr=stderr,
            start_new_session=True,
        )
        started = monotonic()
        samples = 0
        observed: set[int] = set()
        max_tree_rss = max_individual_rss = 0
        max_tree_swap = max_individual_swap = 0
        while process.poll() is None:
            tree_rss = tree_swap = 0
            for pid in monitor._process_tree(process.pid):
                status = monitor._status_kib(pid)
                if status is None:
                    continue
                observed.add(pid)
                tree_rss += status["VmRSS"]
                tree_swap += status["VmSwap"]
                max_individual_rss = max(max_individual_rss, status["VmRSS"])
                max_individual_swap = max(max_individual_swap, status["VmSwap"])
            max_tree_rss = max(max_tree_rss, tree_rss)
            max_tree_swap = max(max_tree_swap, tree_swap)
            samples += 1
            sleep(SAMPLE_SECONDS)
        returncode = process.wait()
        elapsed = monotonic() - started
    memory_after = monitor._counter_file(cgroup / "memory.events") if cgroup else {}
    swap_after = monitor._counter_file(cgroup / "memory.swap.events") if cgroup else {}
    memory_delta = monitor._counter_delta(memory_before, memory_after)
    swap_delta = monitor._counter_delta(swap_before, swap_after)
    verification: dict[str, Any] | None = None
    verification_error: str | None = None
    if returncode == 0:
        try:
            verification = verifier.verify(repository)
            verifier._publish(repository / verifier.VERIFICATION, verification)
        except BaseException as exc:
            verification_error = f"{type(exc).__name__}: {exc}"
    contract, _ = runner._read(repository / runner.CONTRACT)
    limit_kib = int(float(contract["performance"]["worker_peak_rss_mib_max"]) * 1024)
    gates = {
        "producer_exit_zero": returncode == 0,
        "independent_snapshot_verification": verification is not None
            and verification.get("passed") is True,
        "maximum_individual_rss_within_limit": max_individual_rss <= limit_kib,
        "maximum_tree_swap_zero": max_tree_swap == 0,
        "maximum_individual_swap_zero": max_individual_swap == 0,
        "cgroup_oom_zero": memory_delta.get("oom", 0) == 0,
        "cgroup_oom_kill_zero": memory_delta.get("oom_kill", 0) == 0,
        "cgroup_swap_fail_zero": swap_delta.get("fail", 0) == 0,
        "sampling_coverage": samples >= elapsed / SAMPLE_SECONDS * .90,
    }
    state: dict[str, Any] = {
        "schema_version": SCHEMA, "status": "complete",
        "passed": all(gates.values()), "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True,
            capture_output=True, check=True,
        ).stdout.strip(),
        "launched_at": launched_at, "command": command,
        "sample_interval_seconds": SAMPLE_SECONDS, "samples": samples,
        "monitored_seconds": elapsed, "observed_processes": len(observed),
        "maximum_tree_rss_kib": max_tree_rss,
        "maximum_individual_rss_kib": max_individual_rss,
        "maximum_tree_swap_kib": max_tree_swap,
        "maximum_individual_swap_kib": max_individual_swap,
        "cgroup_path": str(cgroup) if cgroup else None,
        "memory_events_before": memory_before, "memory_events_after": memory_after,
        "memory_events_delta": memory_delta,
        "swap_events_before": swap_before, "swap_events_after": swap_after,
        "swap_events_delta": swap_delta,
        "producer_returncode": returncode,
        "producer_stdout_sha256": _sha(stdout_path),
        "producer_stderr_sha256": _sha(stderr_path),
        "snapshot_verification": verification,
        "snapshot_verification_error": verification_error,
        "gates": gates, "audit_sample_verification_pending": True,
        "real_forward_outcomes_accessed": False,
        "production_promotion_authorized": False,
    }
    state["result_digest"] = stable_hash(state)
    _atomic(evidence / "RESULT.json", {
        **state, "created_at": datetime.now(timezone.utc).isoformat(),
    })
    return state


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    value = execute(parser.parse_args(argv).repository)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
