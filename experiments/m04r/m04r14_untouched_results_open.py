"""Publish the durable boundary that permits untouched authority construction."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Sequence

from experiments.m04r import m04r14_untouched_candidate_contract as contract
from experiments.m04r import verify_m04r14_untouched_batch_amendment as amended


SCHEMA = "m04r14-untouched-results-opened-v1"
OUTPUT = Path("config/data/analogues/m04r14/untouched-results-opened-v1")


class ResultsOpenError(RuntimeError): pass


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes(); value = json.loads(raw)
    if type(value) is not dict: raise ResultsOpenError(f"object required: {path}")
    return value, raw


def build(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    candidate, candidate_raw = _read(repository / contract.CANDIDATE_RELATIVE / "RESULT.json")
    receipt_path = repository / amended.OUTPUT / "VERIFIED.json"
    receipt, receipt_raw = _read(receipt_path)
    listener, listener_raw = _read(repository / "config/data/analogues/m04r14/step-listener-v1/TERMINAL.json")
    candidate_state = {key: value for key, value in candidate.items()
        if key not in {"created_at", "result_digest"}}
    receipt_state = {key: value for key, value in receipt.items()
        if key not in {"created_at", "result_digest"}}
    if not all((candidate.get("semantic_passed") is True,
            candidate.get("result_digest") == contract.digest(candidate_state),
            candidate.get("authority_accessed") is False,
            candidate.get("real_forward_outcomes_accessed") is False,
            receipt.get("passed") is True, receipt.get("results_open_authorized") is True,
            receipt.get("result_digest") == contract.digest(receipt_state),
            receipt.get("candidate_result_digest") == candidate.get("result_digest"),
            listener.get("status") == "ready_for_results_open",
            listener.get("verification_result_digest") == receipt.get("result_digest"))):
        raise ResultsOpenError("pre-open evidence differs")
    state = {"schema_version": SCHEMA, "status": "results_opened",
        "candidate_result_digest": candidate["result_digest"],
        "candidate_result_sha256": sha256(candidate_raw).hexdigest(),
        "preopen_verification_result_digest": receipt["result_digest"],
        "preopen_verification_sha256": sha256(receipt_raw).hexdigest(),
        "listener_terminal_sha256": sha256(listener_raw).hexdigest(),
        "original_performance_failure_preserved": True,
        "policy_classification": "post-run-user-authorized-grouped-batch-amendment",
        "authority_access_authorized": True, "outcome_access_authorized": False,
        "production_promotion_authorized": False}
    return {**state, "marker_digest": contract.digest(state)}


def publish(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink(): raise ResultsOpenError("results-open root exists")
    path.mkdir(parents=False); descriptor = os.open(path / "RESULTS_OPENED.json",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json.dumps({**value, "created_at": datetime.now(timezone.utc).isoformat()},
            indent=2, sort_keys=True).encode() + b"\n"); handle.flush(); os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv); repository = args.repository.resolve(strict=True); value = build(repository)
    publish(repository / OUTPUT, value); print(json.dumps(value, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
