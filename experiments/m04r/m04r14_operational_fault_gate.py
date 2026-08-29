"""Real forced-worker-failure and clean-restart gate for M04R-14."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import subprocess
from time import monotonic, sleep
from typing import Any, Mapping, Sequence

from experiments.m04r import m04r14_all60_contract as contract
from experiments.m04r import verify_m04r14_performance_run as run_verifier


SCHEMA = "m04r14-operational-fault-gate-v1"
FAILURE_ROOT = Path("config/data/analogues/m04r14/performance-forced-child-failure-v1")
RESTART_ROOT = Path("config/data/analogues/m04r14/performance-clean-restart-v1")
EVIDENCE_ROOT = Path("config/data/analogues/m04r14/performance-operational-fault-v1")
CONTRACT = Path("experiments/m04r/m04r14_performance_contract_v1.json")


class FaultGateError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise FaultGateError(f"regular file required: {path}")
    return sha256(path.read_bytes()).hexdigest()


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FaultGateError(f"create-only target exists: {path}")
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


def _children(pid: int) -> tuple[int, ...]:
    path = Path(f"/proc/{pid}/task/{pid}/children")
    if not path.is_file():
        return ()
    content = path.read_text().strip()
    return tuple(int(value) for value in content.split()) if content else ()


def _cmdline(pid: int) -> str:
    path = Path(f"/proc/{pid}/cmdline")
    try:
        return path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except FileNotFoundError:
        return ""


def _worker_child(pid: int) -> int | None:
    for child in _children(pid):
        if "multiprocessing.spawn" in _cmdline(child):
            return child
    return None


def _command(repository: Path, output: Path) -> list[str]:
    return [
        str(repository / ".venv/bin/python"), "-m",
        "experiments.m04r.m04r14_throughput_poc", "--repository", str(repository),
        "--output-root", str(output), "--processes", "1", "--case-limit", "1",
    ]


def _tree_manifest(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise FaultGateError(f"symlink in evidence tree: {path}")
        if path.is_file():
            rows.append({
                "path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                "sha256": _sha(path),
            })
    return rows


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    failure = repository / FAILURE_ROOT
    restart = repository / RESTART_ROOT
    evidence = repository / EVIDENCE_ROOT
    if any(path.exists() or path.is_symlink() for path in (failure, restart, evidence)):
        raise FaultGateError("operational gate roots must be absent")
    if _git(repository, "status", "--porcelain"):
        raise FaultGateError("operational gate requires clean Git")
    head = _git(repository, "rev-parse", "HEAD")
    command = _command(repository, failure)
    launched_at = datetime.now(timezone.utc).isoformat()
    process = subprocess.Popen(
        command, cwd=repository, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, start_new_session=True,
    )
    deadline = monotonic() + 180
    worker: int | None = None
    while monotonic() < deadline and process.poll() is None:
        worker = _worker_child(process.pid)
        if worker is not None:
            break
        sleep(0.05)
    if worker is None:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise FaultGateError("spawned worker was not observed")
    worker_cmdline = _cmdline(worker)
    os.kill(worker, signal.SIGKILL)
    try:
        stdout, stderr = process.communicate(timeout=180)
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise FaultGateError("producer did not fail closed after worker death") from exc
    if process.returncode == 0 or not (failure / "FAILED.json").is_file() \
            or (failure / "RESULT.json").exists():
        raise FaultGateError("forced worker failure did not fail closed")
    failure_manifest = _tree_manifest(failure)

    retry = subprocess.run(
        command, cwd=repository, text=True, capture_output=True, timeout=180,
    )
    if retry.returncode == 0 or "output root must be absent" not in retry.stderr:
        raise FaultGateError("partial root was unexpectedly resumable")

    restarted_at = datetime.now(timezone.utc).isoformat()
    restarted = subprocess.run(
        _command(repository, restart), cwd=repository, text=True,
        capture_output=True, timeout=900,
    )
    if restarted.returncode != 0:
        raise FaultGateError(f"clean restart failed: {restarted.stderr[-1000:]}")
    verified = run_verifier.verify(
        restart, repository=repository, case_limit=1, processes=1,
    )
    state = {
        "schema_version": SCHEMA, "status": "complete", "passed": True,
        "git_head": head, "contract_sha256": _sha(repository / CONTRACT),
        "launched_at": launched_at, "forced_worker_pid": worker,
        "forced_worker_cmdline": worker_cmdline, "forced_signal": "SIGKILL",
        "failed_process_returncode": process.returncode,
        "failed_stderr_tail": stderr[-2000:], "failed_stdout_tail": stdout[-2000:],
        "failure_root": str(failure), "failure_manifest": failure_manifest,
        "failure_manifest_digest": contract.stable_digest(failure_manifest),
        "same_root_retry_returncode": retry.returncode,
        "same_root_retry_rejected": True, "restarted_at": restarted_at,
        "restart_root": str(restart), "restart_verification": verified,
        "production_promotion_authorized": False,
    }
    state["result_digest"] = contract.stable_digest(state)
    evidence.mkdir(parents=True)
    _atomic(evidence / "RESULT.json", {
        **state, "created_at": datetime.now(timezone.utc).isoformat(),
    })
    return state


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    state = execute(parser.parse_args(argv).repository)
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
