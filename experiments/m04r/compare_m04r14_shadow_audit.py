"""Compare the 12 preregistered shadow cases with the verified audit authority."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.m04r import m04r14_shadow_run as base
from market_analogues.types import stable_hash


SCHEMA = "m04r14-shadow-audit-comparison-v1"
SEMANTIC_RECEIPT = Path(
    "config/data/analogues/m04r14/nasdaq-shadow-interrupted-verification-v1/VERIFIED.json"
)
AUTHORITY_RECEIPT = Path(
    "config/data/analogues/m04r14/nasdaq-shadow-audit-authority-v1-verification/VERIFIED.json"
)
REGISTRY = Path("config/data/analogues/m04r14/nasdaq-shadow-denominator-v1")
CANDIDATE = Path("config/data/analogues/m04r14/nasdaq-shadow-snapshot-v1")
AUTHORITY = Path("config/data/analogues/m04r14/nasdaq-shadow-audit-authority-v1")
OUTPUT = Path("config/data/analogues/m04r14/nasdaq-shadow-audit-comparison-v1")


class AuditComparisonError(RuntimeError):
    pass


def _receipt_valid(value: Mapping[str, Any]) -> bool:
    state = {
        key: item for key, item in value.items()
        if key not in {"result_digest", "created_at"}
    }
    return value.get("result_digest") == stable_hash(state)


def _compare_rows(
    candidate: Mapping[str, Any], authority: Mapping[str, Any],
) -> dict[str, Any]:
    left = candidate.get("matches")
    right = authority.get("matches")
    if type(left) is not list or type(right) is not list or len(left) != 20 or len(right) != 20:
        raise AuditComparisonError("twenty ordered matches required")
    case_equal = candidate.get("registry_case_id") == authority.get("registry_case_id")
    prefix_equal = (
        candidate.get("query_stock_prefix") == authority.get("query_stock_prefix")
        and candidate.get("query_benchmark_prefix") == authority.get("query_benchmark_prefix")
    )
    matching = sum(a == b for a, b in zip(left, right, strict=True))
    return {
        "query_id": candidate.get("query_episode_id"),
        "case_id": candidate.get("registry_case_id"),
        "case_id_equal": case_equal, "prefix_equal": prefix_equal,
        "matches_equal": left == right, "matching_positions": matching,
        "passed": case_equal and prefix_equal and left == right,
    }


def compare(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    semantic, _ = base._read(repository / SEMANTIC_RECEIPT)
    authority_receipt, _ = base._read(repository / AUTHORITY_RECEIPT)
    if not all((
        _receipt_valid(semantic), semantic.get("semantic_passed") is True,
        semantic.get("verified_cases") == 3270,
        semantic.get("verified_matches") == 65400,
        semantic.get("audit_sample_verification_pending") is True,
        semantic.get("real_forward_outcomes_accessed") is False,
        _receipt_valid(authority_receipt), authority_receipt.get("passed") is True,
        authority_receipt.get("comparison_authorized") is True,
        authority_receipt.get("verified_cases") == 12,
        authority_receipt.get("verified_matches") == 240,
        authority_receipt.get("candidate_case_root_accessed") is False,
        authority_receipt.get("real_forward_outcomes_accessed") is False,
    )):
        raise AuditComparisonError("upstream verification differs")
    registry, _ = base._read(repository / REGISTRY / "query-registry.json")
    sample = registry.get("audit_sample")
    if type(sample) is not list or len(sample) != 12 \
            or stable_hash(sample) != registry.get("audit_sample_digest") \
            or registry.get("registry_digest") != semantic.get("registry_digest") \
            or registry.get("registry_digest") != authority_receipt.get("registry_digest"):
        raise AuditComparisonError("audit sample binding differs")
    rows: list[dict[str, Any]] = []
    for item in sample:
        query_id = str(item["episode_id"])
        candidate, _ = base._read(repository / CANDIDATE / "cases" / f"{query_id}.json")
        authority, _ = base._read(repository / AUTHORITY / "cases" / f"{query_id}.json")
        if not all((
            candidate.get("query_episode_id") == query_id,
            authority.get("query_episode_id") == query_id,
            candidate.get("result_digest") == base._deterministic_case_digest(candidate),
            candidate.get("checkpoint_integrity_digest") == base._integrity_digest(candidate),
            authority.get("result_digest") == base._deterministic_case_digest(authority),
            authority.get("checkpoint_integrity_digest") == base._integrity_digest(authority),
            candidate.get("gate_passed") is True,
            authority.get("gate_passed") is True,
        )):
            raise AuditComparisonError(f"case seal differs: {item['case_id']}")
        rows.append(_compare_rows(candidate, authority))
    matching = sum(row["matching_positions"] for row in rows)
    state = {
        "schema_version": SCHEMA, "status": "complete",
        "passed": all(row["passed"] for row in rows) and matching == 240,
        "semantic_verification_result_digest": semantic["result_digest"],
        "authority_verification_result_digest": authority_receipt["result_digest"],
        "registry_digest": registry["registry_digest"],
        "audit_sample_digest": registry["audit_sample_digest"],
        "cases": 12, "positions": 240, "matching_positions": matching,
        "case_comparisons": rows,
        "authority_accessed_only_after_independent_verification": True,
        "real_forward_outcomes_accessed": False,
        "production_promotion_authorized": False,
    }
    if not state["passed"]:
        raise AuditComparisonError("shadow audit comparison differs")
    return {**state, "result_digest": stable_hash(state)}


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise AuditComparisonError("audit comparison root exists")
    path.mkdir(parents=False)
    descriptor = os.open(path / "RESULT.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json.dumps({
            **value, "created_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv)
    repository = args.repository.resolve(strict=True)
    value = compare(repository)
    _publish(repository / OUTPUT, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
