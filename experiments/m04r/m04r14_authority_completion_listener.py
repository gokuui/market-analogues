"""One-shot detached listener: authority terminal -> verify -> compare -> receipt."""
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
from typing import Any, Sequence

from experiments.m04r import compare_m04r14_untouched as comparison
from experiments.m04r import m04r14_untouched_authority as authority
from experiments.m04r import verify_m04r14_untouched_authority as verifier


ROOT = Path("config/data/analogues/m04r14/untouched-authority-listener-v2")


def _write(path: Path, value: dict[str, Any]) -> None:
    if path.exists(): raise RuntimeError(f"listener terminal exists: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"); handle.flush(); os.fsync(handle.fileno())


def _run(repository: Path, module: str) -> None:
    result = subprocess.run([sys.executable, "-m", module, "--repository", str(repository)],
        cwd=repository, text=True, capture_output=True)
    if result.returncode: raise RuntimeError(f"{module} failed: {result.stderr[-4000:]}")


def listen(repository: Path, interval: float) -> str:
    repository = repository.resolve(strict=True); root = repository / ROOT; root.mkdir(parents=True, exist_ok=True)
    lock = (root / "listener.lock").open("w"); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    authority_root = repository / authority.OUTPUT
    while not (authority_root / "AUTHORITY.json").exists():
        if (authority_root / "FAILED.json").exists():
            _write(root / "TERMINAL.json", {"status": "blocked", "reason": "authority_failed",
                "created_at": datetime.now(timezone.utc).isoformat()}); return "blocked"
        sleep(max(.25, interval))
    try:
        if not (repository / verifier.OUTPUT).exists():
            _run(repository, "experiments.m04r.verify_m04r14_untouched_authority")
        if not (repository / comparison.OUTPUT).exists():
            _run(repository, "experiments.m04r.compare_m04r14_untouched")
        result = json.loads((repository / comparison.OUTPUT / "RESULT.json").read_text())
        terminal = {"status": "complete", "passed": result.get("passed") is True,
            "matching_positions": result.get("matching_positions"),
            "comparison_result_digest": result.get("result_digest"),
            "real_forward_outcomes_accessed": False,
            "production_promotion_authorized": False,
            "created_at": datetime.now(timezone.utc).isoformat()}
        _write(root / "TERMINAL.json", terminal); return "complete"
    except BaseException as exc:
        _write(root / "TERMINAL.json", {"status": "blocked", "reason": type(exc).__name__,
            "message": str(exc), "created_at": datetime.now(timezone.utc).isoformat()}); raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=5.0); args = parser.parse_args(argv)
    print(json.dumps({"state": listen(args.repository, args.interval_seconds)})); return 0


if __name__ == "__main__": raise SystemExit(main())
