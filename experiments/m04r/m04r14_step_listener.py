"""Durable fail-closed event listener for M04R-14 step transitions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from time import sleep
from typing import Any, Mapping, Sequence

from experiments.m04r import m04r14_untouched_candidate_contract as contract
from experiments.m04r import m04r14_untouched_failure_diagnostic as diagnostic
from experiments.m04r import verify_m04r14_untouched_batch_amendment as amended


SCHEMA = "m04r14-step-listener-v1"
ROOT = Path("config/data/analogues/m04r14/step-listener-v1")


class ListenerError(RuntimeError): pass


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if type(value) is not dict: raise ListenerError(f"object required: {path}")
    return value


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink(): raise ListenerError(f"create-only target exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def _event(root: Path, name: str, payload: Mapping[str, Any]) -> None:
    existing = sorted(root.glob("[0-9][0-9][0-9]-*.json")); sequence = len(existing)
    value = {"schema_version": SCHEMA, "sequence": sequence, "event": name,
        "payload": dict(payload), "created_at": datetime.now(timezone.utc).isoformat()}
    value["event_digest"] = contract.digest({key: item for key, item in value.items() if key != "created_at"})
    _atomic(root / f"{sequence:03d}-{name}.json", value)


def _run(repository: Path, module: str, *arguments: str) -> None:
    command = [sys.executable, "-m", module, "--repository", str(repository), *arguments]
    result = subprocess.run(command, cwd=repository, text=True, capture_output=True)
    if result.returncode:
        raise ListenerError(f"transition command failed ({module}): {result.stderr[-2000:]}")


def transition(repository: Path, root: Path) -> str:
    candidate = repository / contract.CANDIDATE_RELATIVE
    failed = candidate / "FAILED.json"; result_path = candidate / "RESULT.json"
    if failed.exists():
        failure = _read(failed); _event(root, "CANDIDATE_FAILED", {"error_type": failure.get("error_type")})
        _atomic(root / "TERMINAL.json", {"schema_version": SCHEMA, "status": "blocked_before_authority",
            "reason": "candidate_failed", "authority_open_authorized": False})
        return "blocked_before_authority"
    if not result_path.exists(): return "waiting_for_candidate_terminal"
    result = _read(result_path)
    if not list(root.glob("*-CANDIDATE_COMPLETE.json")):
        _event(root, "CANDIDATE_COMPLETE", {"result_digest": result.get("result_digest"),
            "semantic_passed": result.get("semantic_passed"),
            "performance_passed": result.get("performance_passed")})
    if result.get("semantic_passed") is not True:
        _atomic(root / "TERMINAL.json", {"schema_version": SCHEMA, "status": "blocked_before_authority",
            "reason": "candidate_semantic_failure", "authority_open_authorized": False})
        return "blocked_before_authority"
    if result.get("performance_passed") is False:
        diag_root = repository / diagnostic.OUTPUT_RELATIVE
        if not diag_root.exists():
            _run(repository, "experiments.m04r.m04r14_untouched_failure_diagnostic")
            _event(root, "FAILURE_DIAGNOSED", {"diagnostic": str(diag_root)})
        verification = repository / amended.OUTPUT
        if not verification.exists():
            _run(repository, "experiments.m04r.verify_m04r14_untouched_batch_amendment")
            _event(root, "BATCH_AMENDMENT_VERIFIED", {"verification": str(verification)})
        receipt = _read(verification / "VERIFIED.json")
        if receipt.get("passed") is not True or receipt.get("results_open_authorized") is not True:
            raise ListenerError("amended verification receipt differs")
        terminal = {"schema_version": SCHEMA, "status": "ready_for_results_open",
            "candidate_result_digest": result["result_digest"],
            "verification_result_digest": receipt["result_digest"],
            "original_performance_failure_preserved": True,
            "authority_open_authorized": True, "production_promotion_authorized": False}
        if not (root / "TERMINAL.json").exists(): _atomic(root / "TERMINAL.json", terminal)
        return "ready_for_results_open"
    raise ListenerError("unimplemented candidate terminal classification")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--watch", action="store_true"); parser.add_argument("--interval-seconds", type=float, default=2.0)
    args = parser.parse_args(argv); repository = args.repository.resolve(strict=True); root = repository / ROOT
    root.mkdir(parents=True, exist_ok=True); lock = (root / "listener.lock").open("w")
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc: raise ListenerError("listener is already running") from exc
    while True:
        state = transition(repository, root); print(json.dumps({"state": state}), flush=True)
        if state != "waiting_for_candidate_terminal" or not args.watch: return 0
        sleep(max(.25, args.interval_seconds))


if __name__ == "__main__": raise SystemExit(main())
