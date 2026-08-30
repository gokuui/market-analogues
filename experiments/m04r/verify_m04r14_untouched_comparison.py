"""Independent terminal verifier for the untouched candidate/authority comparison."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.m04r import m04r14_untouched_candidate_contract as contract
from experiments.m04r import verify_m04r14_untouched_candidate as evidence


SCHEMA = "m04r14-untouched-comparison-terminal-verification-v2"
COMPARISON = Path("config/data/analogues/m04r14/untouched-comparison-v1")
OUTPUT = Path("config/data/analogues/m04r14/untouched-comparison-v2-terminal-verification")
AUTHORITY_ROOT = Path("config/data/analogues/m04r14/untouched-authority-v1")
CANDIDATE_VERIFICATION = Path(
    "config/data/analogues/m04r14/untouched-candidate-v1-batch-verification"
)
AUTHORITY_VERIFICATION = Path(
    "config/data/analogues/m04r14/untouched-authority-v1-verification"
)


class TerminalVerificationError(RuntimeError): pass


def reconstruct_rows(candidate_rows: Mapping[str, Mapping[str, Any]],
                     authority_rows: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(candidate_rows) != 72 or set(candidate_rows) != set(authority_rows):
        raise TerminalVerificationError("terminal case inventory differs")
    rows = []
    for query_id in sorted(candidate_rows):
        candidate = candidate_rows[query_id]; exact = authority_rows[query_id]
        left = candidate.get("matches"); right = exact.get("matches")
        if type(left) is not list or type(right) is not list or len(left) != 20 or len(right) != 20:
            raise TerminalVerificationError("terminal match inventory differs")
        case_id_equal = candidate.get("registry_case_id") == exact.get("registry_case_id")
        prefix_equal = candidate.get("query_stock_prefix") == exact.get("query_stock_prefix") \
            and candidate.get("query_benchmark_prefix") == exact.get("query_benchmark_prefix")
        matches_equal = left == right
        rows.append({"query_id": query_id, "case_id": candidate.get("registry_case_id"),
            "case_id_equal": case_id_equal, "prefix_equal": prefix_equal,
            "matches_equal": matches_equal,
            "matching_positions": sum(a == b for a, b in zip(left, right, strict=True)),
            "passed": case_id_equal and prefix_equal and matches_equal})
    return rows


def _case_map(root: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted(root.glob("*.json")):
        row, _ = evidence._read(path); query_id = str(row.get("query_episode_id"))
        if query_id in result: raise TerminalVerificationError("duplicate terminal query")
        if row.get("gate_passed") is not True \
                or row.get("result_digest") != evidence._deterministic_case_digest(row) \
                or row.get("checkpoint_integrity_digest") != evidence._integrity_digest(row):
            raise TerminalVerificationError(f"terminal case seal differs: {query_id}")
        result[query_id] = row
    return result


def _sealed_payload(value: Mapping[str, Any]) -> bool:
    state = {key: item for key, item in value.items()
        if key not in {"created_at", "result_digest"}}
    return value.get("result_digest") == contract.digest(state)


def _introduced_source(repository: Path, relative: Path,
                       artifact_created_at: str) -> dict[str, str]:
    commits = str(evidence._git(repository, "log", "--diff-filter=A", "--format=%H",
        "--", relative.as_posix())).splitlines()
    if len(commits) != 1: raise TerminalVerificationError(f"source introduction differs: {relative}")
    commit = commits[0]; raw = evidence._git(repository, "show", f"{commit}:{relative.as_posix()}", raw=True)
    committed_at = str(evidence._git(repository, "show", "-s", "--format=%cI", commit))
    if committed_at >= artifact_created_at:
        raise TerminalVerificationError(f"artifact predates committed producer: {relative}")
    return {"path": relative.as_posix(), "commit": commit,
        "sha256": sha256(raw).hexdigest(), "commit_time": committed_at}


def verify(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    comparison, comparison_raw = evidence._read(repository / COMPARISON / "RESULT.json")
    candidate_receipt, candidate_raw = evidence._read(
        repository / CANDIDATE_VERIFICATION / "VERIFIED.json")
    authority_receipt, authority_raw = evidence._read(
        repository / AUTHORITY_VERIFICATION / "VERIFIED.json")
    deterministic = {key: value for key, value in comparison.items()
        if key not in {"result_digest", "created_at"}}
    if comparison.get("result_digest") != contract.digest(deterministic) \
            or comparison.get("status") != "complete" or comparison.get("passed") is not True \
            or not _sealed_payload(candidate_receipt) or not _sealed_payload(authority_receipt) \
            or candidate_receipt.get("passed") is not True or authority_receipt.get("passed") is not True \
            or comparison.get("candidate_verification_result_digest") != candidate_receipt.get("result_digest") \
            or comparison.get("authority_verification_result_digest") != authority_receipt.get("result_digest"):
        raise TerminalVerificationError("terminal upstream binding differs")
    candidate = _case_map(repository / contract.CANDIDATE_RELATIVE / "cases")
    exact = _case_map(repository / AUTHORITY_ROOT / "cases")
    rows = reconstruct_rows(candidate, exact)
    matching = sum(row["matching_positions"] for row in rows)
    if rows != comparison.get("case_comparisons") or not all(row["passed"] for row in rows) \
            or matching != 1440 or comparison.get("matching_positions") != 1440 \
            or comparison.get("authority_accessed_after_results_open") is not True \
            or comparison.get("real_forward_outcomes_accessed") is not False:
        raise TerminalVerificationError("terminal comparison reconstruction differs")
    marker, marker_raw = evidence._read(repository /
        "config/data/analogues/m04r14/untouched-results-opened-v1/RESULTS_OPENED.json")
    marker_state = {key: value for key, value in marker.items()
        if key not in {"created_at", "marker_digest"}}
    if marker.get("marker_digest") != contract.digest(marker_state) \
            or marker.get("status") != "results_opened" or marker.get("authority_access_authorized") is not True \
            or marker.get("outcome_access_authorized") is not False:
        raise TerminalVerificationError("terminal results-open boundary differs")
    authority_result, _ = evidence._read(repository / AUTHORITY_ROOT / "AUTHORITY.json")
    producer_sources = [
        _introduced_source(repository, Path("experiments/m04r/m04r14_untouched_authority.py"),
            str(authority_result["created_at"])),
        _introduced_source(repository, Path("experiments/m04r/compare_m04r14_untouched.py"),
            str(comparison["created_at"])),
    ]
    state = {"schema_version": SCHEMA, "status": "verified", "passed": True,
        "comparison_result_digest": comparison["result_digest"],
        "comparison_sha256": sha256(comparison_raw).hexdigest(),
        "candidate_verification_sha256": sha256(candidate_raw).hexdigest(),
        "authority_verification_sha256": sha256(authority_raw).hexdigest(),
        "results_open_marker_sha256": sha256(marker_raw).hexdigest(),
        "producer_sources": producer_sources,
        "producer_binding_limit": "post-run reconstruction; producer artifacts did not embed Git commit",
        "verified_cases": 72, "verified_positions": 1440,
        "exact_ordered_match_agreement": True,
        "original_performance_failure_preserved": True,
        "policy_classification": "post-run-user-authorized-grouped-batch-amendment",
        "real_forward_outcomes_accessed": False,
        "t14_07_complete": True, "production_promotion_authorized": False}
    return {**state, "result_digest": contract.digest(state)}


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink(): raise TerminalVerificationError("terminal verification root exists")
    path.mkdir(parents=False); descriptor = os.open(path / "VERIFIED.json",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json.dumps({**value, "created_at": datetime.now(timezone.utc).isoformat()},
            indent=2, sort_keys=True).encode() + b"\n"); handle.flush(); os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true"); args = parser.parse_args(argv)
    repository = args.repository.resolve(strict=True); value = verify(repository)
    if not args.dry_run: _publish(repository / OUTPUT, value)
    print(json.dumps(value, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
