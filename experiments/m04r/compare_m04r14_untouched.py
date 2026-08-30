"""Compare the sealed untouched candidate with its independently verified authority."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.m04r import m04r14_untouched_candidate_contract as contract
from experiments.m04r import verify_m04r14_untouched_authority as authority_verifier
from experiments.m04r import verify_m04r14_untouched_batch_amendment as candidate_verifier
from experiments.m04r import verify_m04r14_untouched_candidate as base


SCHEMA = "m04r14-untouched-candidate-authority-comparison-v1"
OUTPUT = Path("config/data/analogues/m04r14/untouched-comparison-v1")
AUTHORITY_ROOT = Path("config/data/analogues/m04r14/untouched-authority-v1")


class ComparisonError(RuntimeError): pass


def compare(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    candidate_receipt, _ = base._read(repository / candidate_verifier.OUTPUT / "VERIFIED.json")
    authority_receipt, _ = base._read(repository / authority_verifier.OUTPUT / "VERIFIED.json")
    candidate_state = {key: value for key, value in candidate_receipt.items()
        if key not in {"created_at", "result_digest"}}
    authority_state = {key: value for key, value in authority_receipt.items()
        if key not in {"created_at", "result_digest"}}
    if candidate_receipt.get("result_digest") != contract.digest(candidate_state) \
            or authority_receipt.get("result_digest") != contract.digest(authority_state) \
            or candidate_receipt.get("passed") is not True or candidate_receipt.get("results_open_authorized") is not True \
            or authority_receipt.get("passed") is not True \
            or authority_receipt.get("comparison_authorized") is not True:
        raise ComparisonError("upstream verification differs")
    candidate_root = repository / contract.CANDIDATE_RELATIVE / "cases"
    authority_root = repository / AUTHORITY_ROOT / "cases"
    candidate_rows = {}
    for path in candidate_root.glob("*.json"):
        row, _ = base._read(path); query_id = str(row["query_episode_id"])
        if query_id in candidate_rows: raise ComparisonError("duplicate candidate query")
        candidate_rows[query_id] = row
    authority_rows = {}
    for path in authority_root.glob("*.json"):
        row, _ = base._read(path); query_id = str(row["query_episode_id"])
        if query_id in authority_rows: raise ComparisonError("duplicate authority query")
        authority_rows[query_id] = row
    if len(candidate_rows) != 72 or set(candidate_rows) != set(authority_rows):
        raise ComparisonError("comparison case inventory differs")
    rows = []
    for query_id in sorted(candidate_rows):
        candidate = candidate_rows[query_id]; exact = authority_rows[query_id]
        if len(candidate.get("matches", [])) != 20 or len(exact.get("matches", [])) != 20:
            raise ComparisonError("comparison match inventory differs")
        matches_equal = candidate["matches"] == exact["matches"]
        case_id_equal = candidate["registry_case_id"] == exact["registry_case_id"]
        prefix_equal = candidate["query_stock_prefix"] == exact["query_stock_prefix"] \
            and candidate["query_benchmark_prefix"] == exact["query_benchmark_prefix"]
        rows.append({"query_id": query_id, "case_id": candidate["registry_case_id"],
            "case_id_equal": case_id_equal, "prefix_equal": prefix_equal,
            "matches_equal": matches_equal, "matching_positions": sum(
                left == right for left, right in zip(candidate["matches"], exact["matches"], strict=True)),
            "passed": case_id_equal and prefix_equal and matches_equal})
    matching = sum(row["matching_positions"] for row in rows)
    state = {"schema_version": SCHEMA, "status": "complete", "passed": all(row["passed"] for row in rows),
        "candidate_verification_result_digest": candidate_receipt["result_digest"],
        "authority_verification_result_digest": authority_receipt["result_digest"],
        "cases": 72, "positions": 1440, "matching_positions": matching,
        "case_comparisons": rows, "authority_accessed_after_results_open": True,
        "real_forward_outcomes_accessed": False, "production_promotion_authorized": False}
    if not state["passed"] or matching != 1440: raise ComparisonError("untouched comparison differs")
    return {**state, "result_digest": contract.digest(state)}


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink(): raise ComparisonError("comparison root exists")
    path.mkdir(parents=False); descriptor = os.open(path / "RESULT.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json.dumps({**value, "created_at": datetime.now(timezone.utc).isoformat()}, indent=2,
            sort_keys=True).encode() + b"\n"); handle.flush(); os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv); repository = args.repository.resolve(strict=True); value = compare(repository)
    _publish(repository / OUTPUT, value); print(json.dumps(value, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
