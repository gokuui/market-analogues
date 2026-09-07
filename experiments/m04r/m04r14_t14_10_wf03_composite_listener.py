"""Wait once for the composite producer, then launch its independent verifier."""
from __future__ import annotations

import argparse
import errno
import os
from pathlib import Path
import select
import subprocess
from typing import Any, Mapping, Sequence

from experiments.m04r import m04r14_t14_10_wf03_composite_batch as producer
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import verify_m04r14_t14_10_wf03_composite_batch as verifier


class CompositeListenerError(RuntimeError):
    pass


def _wait_once(pid: int) -> str:
    """Block in the kernel on a stable process handle; never poll by time."""
    try:
        descriptor = os.pidfd_open(pid, 0)
    except ProcessLookupError:
        return "already_exited"
    except AttributeError as exc:
        raise CompositeListenerError("pidfd_open is unavailable") from exc
    try:
        waiter = select.poll()
        waiter.register(descriptor, select.POLLIN)
        events = waiter.poll()
        if not events:
            raise CompositeListenerError("process listener returned without event")
    finally:
        os.close(descriptor)
    return "process_exit_observed"


def _terminal_ready(repository: Path) -> tuple[bool, dict[str, Any]]:
    root = repository / producer.OUTPUT_RELATIVE
    progress_path = root / "PROGRESS.json"
    result_path = root / "RESULT.json"
    if not progress_path.exists() or not result_path.exists():
        return False, {
            "reason": "producer terminal artifacts are absent",
            "progress_exists": progress_path.exists(),
            "result_exists": result_path.exists(),
        }
    progress = base._read(progress_path)
    result = base._read(result_path)
    try:
        base._validate_seal(result)
        ready = all((
            progress.get("schema_version")
                == "m04r14-wf03-composite-batch-progress-v1",
            progress.get("status") == "complete",
            progress.get("completed_queries") == producer.EXPECTED_QUERIES,
            progress.get("remaining_queries") == 0,
            result.get("schema_version")
                == "m04r14-t14-10-wf03-composite-batch-result-v1",
            result.get("status") == "complete", result.get("passed") is True,
            result.get("queries") == producer.EXPECTED_QUERIES,
            result.get("independent_verification_authorized") is True,
        ))
    except (KeyError, TypeError, base.FeasibilityError):
        ready = False
    return ready, {
        "progress_status": progress.get("status"),
        "completed_queries": progress.get("completed_queries"),
        "remaining_queries": progress.get("remaining_queries"),
        "producer_result_digest": result.get("result_digest"),
    }


def listen(repository: Path, producer_pid: int) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if producer._git(repository, "status", "--porcelain"):
        raise CompositeListenerError("listener requires clean commit")
    root = repository / producer.OUTPUT_RELATIVE / "listener-v1"
    if root.exists() or root.is_symlink():
        raise CompositeListenerError("listener output already exists")
    root.mkdir(parents=True)
    preregistration = base._read(repository / producer.PREREGISTRATION_RELATIVE)
    started = base._sealed({
        "schema_version": "m04r14-wf03-composite-listener-v1",
        "status": "waiting", "producer_pid": producer_pid,
        "preregistration_digest": preregistration["preregistration_digest"],
        "listener_commit": producer._git(repository, "rev-parse", "HEAD"),
        "created_at": base._now(),
    }, "listener_digest")
    base._atomic(root / "LISTENER_STARTED.json", started)
    wait_mode = _wait_once(producer_pid)
    ready, terminal = _terminal_ready(repository)
    if not ready:
        failed = base._sealed({
            "schema_version": "m04r14-wf03-composite-listener-v1",
            "status": "producer_incomplete", "producer_pid": producer_pid,
            "wait_mode": wait_mode, "terminal_observation": terminal,
            "verifier_launched": False, "created_at": base._now(),
        }, "listener_digest")
        base._atomic(root / "LISTENER_FAILED.json", failed)
        return failed
    command = [
        str(repository / ".venv/bin/python"), "-m",
        "experiments.m04r.verify_m04r14_t14_10_wf03_composite_batch",
        "--repository", str(repository),
    ]
    completed = subprocess.run(
        command, cwd=repository, text=True, capture_output=True, check=False,
    )
    verification_path = repository / verifier.OUTPUT_RELATIVE / "VERIFIED.json"
    if completed.returncode or not verification_path.is_file():
        failed = base._sealed({
            "schema_version": "m04r14-wf03-composite-listener-v1",
            "status": "verification_failed", "producer_pid": producer_pid,
            "wait_mode": wait_mode, "terminal_observation": terminal,
            "verifier_launched": True, "verifier_returncode": completed.returncode,
            "verifier_stdout_tail": completed.stdout[-4000:],
            "verifier_stderr_tail": completed.stderr[-4000:],
            "created_at": base._now(),
        }, "listener_digest")
        base._atomic(root / "LISTENER_FAILED.json", failed)
        return failed
    verification = base._read(verification_path)
    base._validate_seal(verification, "verification_digest")
    state = base._sealed({
        "schema_version": "m04r14-wf03-composite-listener-v1",
        "status": "complete", "producer_pid": producer_pid,
        "wait_mode": wait_mode, "terminal_observation": terminal,
        "verifier_launched": True, "verifier_returncode": completed.returncode,
        "verification_digest": verification["verification_digest"],
        "verification_passed": verification.get("passed") is True,
        "created_at": base._now(),
    }, "listener_digest")
    base._atomic(root / "LISTENER_COMPLETED.json", state)
    return state


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--producer-pid", required=True, type=int)
    args = parser.parse_args(argv)
    try:
        result = listen(args.repository, args.producer_pid)
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            raise CompositeListenerError("producer process does not exist") from exc
        raise
    print(result)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
