"""Independently reconstruct the WF-03D query-symbol exclusion audit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import subprocess
from typing import Any, Mapping

import pandas as pd

from market_analogues.types import stable_hash
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_10_wf03d_exclusion_audit as producer


SCHEMA = "m04r14-t14-10-wf03d-exclusion-audit-verification-v1"
OUTPUT_RELATIVE = Path(
    "config/data/analogues/m04r14/"
    "t14-10-wf03d-exclusion-audit-v1-verification"
)


class ExclusionAuditVerificationError(RuntimeError):
    pass


def _git(repository: Path, *args: str, raw: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=repository, capture_output=True,
        text=not raw, check=False,
    )
    if result.returncode:
        message = result.stderr if not raw else result.stderr.decode(errors="replace")
        raise ExclusionAuditVerificationError(message.strip() or "git command failed")
    return result.stdout if raw else result.stdout.strip()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _identity_digest(values: Mapping[str, str]) -> str:
    return stable_hash([
        {"episode_id": episode_id, "symbol": symbol}
        for episode_id, symbol in sorted(values.items())
    ])


def _method_rows(
    composite: Mapping[str, Any], price: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> tuple[tuple[str, list[Mapping[str, Any]]], ...]:
    try:
        result = (
            ("composite", composite["retrieval"]["matches"]),
            ("price_only", price["matches"]),
            ("deterministic_random", baseline["random_neighbors"]),
            ("recent_return_volatility", baseline["rank_neighbors"]),
        )
    except (KeyError, TypeError) as exc:
        raise ExclusionAuditVerificationError("source case layout differs") from exc
    if any(type(rows) is not list for _method, rows in result):
        raise ExclusionAuditVerificationError("source neighbour list differs")
    return result


def reconstruct(repository: Path) -> dict[str, Any]:
    registry, _ = base._registry(repository)
    rows = registry.get("queries_data")
    if type(rows) is not list or len(rows) != producer.EXPECTED_QUERIES:
        raise ExclusionAuditVerificationError("registry inventory differs")
    counts: dict[str, dict[str, Any]] = {
        method: {"links": 0, "ids": set(), "affected": [], "overlap": 0}
        for method in (
            "composite", "price_only", "deterministic_random",
            "recent_return_volatility",
        )
    }
    identities: dict[str, str] = {}
    query_impact = []
    for row in rows:
        query_id = str(row["episode_id"])
        source_cases = []
        for root in (
            producer.COMPOSITE_ROOT, producer.PRICE_ROOT, producer.BASELINE_ROOT,
        ):
            case = base._read(repository / root / "cases" / f"{query_id}.json")
            base._validate_seal(case, "case_digest")
            if not all((
                case.get("query_id") == query_id,
                case.get("case_id") == row["case_id"],
                case.get("symbol") == row["symbol"],
                case.get("cutoff") == row["cutoff"],
                case.get("outcomes_or_labels_used") is False,
                case.get("historical_walk_forward_query_outcomes_opened") is False,
                case.get("final_period_result_opened") is False,
            )):
                raise ExclusionAuditVerificationError(
                    f"source query binding differs: {query_id}"
                )
            source_cases.append(case)
        affected = []
        for method, neighbours in _method_rows(*source_cases):
            if len(neighbours) != producer.TOP_K:
                raise ExclusionAuditVerificationError(
                    f"source method count differs: {query_id}:{method}"
                )
            symbols = [item.get("symbol") for item in neighbours]
            identifiers = [item.get("episode_id") for item in neighbours]
            if len(set(symbols)) != producer.TOP_K \
                    or len(set(identifiers)) != producer.TOP_K:
                raise ExclusionAuditVerificationError(
                    f"source method diversity differs: {query_id}:{method}"
                )
            for episode_id, symbol in zip(identifiers, symbols, strict=True):
                if type(episode_id) is not str or type(symbol) is not str:
                    raise ExclusionAuditVerificationError("source identity type differs")
                try:
                    valid_id = len(bytes.fromhex(episode_id)) == 12
                except ValueError:
                    valid_id = False
                if not valid_id:
                    raise ExclusionAuditVerificationError("source episode ID differs")
                prior = identities.setdefault(episode_id, symbol)
                if prior != symbol:
                    raise ExclusionAuditVerificationError(
                        "source episode identity conflicts"
                    )
            same = sum(symbol == row["symbol"] for symbol in symbols)
            if same not in (0, 1):
                raise ExclusionAuditVerificationError("query symbol count differs")
            counts[method]["links"] += len(neighbours)
            counts[method]["ids"].update(identifiers)
            if same:
                counts[method]["affected"].append(query_id)
                affected.append(method)
        query_impact.append({"query_id": query_id, "affected_methods": affected})
    old_ids = set(pd.read_parquet(
        repository / producer.OLD_OUTCOMES, columns=["episode_id"],
    )["episode_id"].astype(str))
    methods = {}
    for method, values in counts.items():
        affected = sorted(values["affected"])
        methods[method] = {
            "links": values["links"],
            "unique_episodes": len(values["ids"]),
            "same_symbol_links": len(affected),
            "affected_queries": len(affected),
            "affected_query_ids": affected,
            "affected_query_digest": stable_hash(affected),
            "t14_09_outcome_overlap": len(values["ids"] & old_ids),
        }
    all_ids = set(identities)
    return {
        "registry_digest": registry["registry_digest"],
        "query_count": len(rows),
        "methods": methods,
        "total_links": sum(value["links"] for value in methods.values()),
        "unique_episodes": len(all_ids),
        "episode_identity_digest": _identity_digest(identities),
        "query_impact_digest": stable_hash(query_impact),
        "t14_09_unique_outcomes": len(old_ids),
        "t14_09_outcome_overlap": len(all_ids & old_ids),
        "outcomes_missing_from_t14_09": len(all_ids - old_ids),
    }


def verify(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain"):
        raise ExclusionAuditVerificationError("audit verifier requires clean commit")
    verifier_commit = str(_git(repository, "rev-parse", "HEAD"))
    verifier_path = Path(__file__).resolve()
    verifier_relative = verifier_path.relative_to(repository).as_posix()
    if sha256(_git(
        repository, "show", f"{verifier_commit}:{verifier_relative}", raw=True,
    )).hexdigest() != _sha(verifier_path):
        raise ExclusionAuditVerificationError("verifier Git binding differs")
    audit = base._read(repository / producer.OUTPUT_RELATIVE / "AUDIT.json")
    base._validate_seal(audit, "audit_digest")
    implementation_commit = audit.get("implementation_commit")
    if type(implementation_commit) is not str:
        raise ExclusionAuditVerificationError("producer commit binding differs")
    producer_relative = Path(producer.__file__).resolve().relative_to(repository).as_posix()
    producer_blob = _git(
        repository, "show", f"{implementation_commit}:{producer_relative}", raw=True,
    )
    if sha256(producer_blob).hexdigest() != audit.get("runtime_sha256"):
        raise ExclusionAuditVerificationError("producer runtime binding differs")
    contract = base._read(repository / producer.CONTRACT_RELATIVE)
    if audit.get("walk_forward_contract_digest") != contract.get("contract_digest") \
            or audit.get("walk_forward_contract_sha256") \
            != _sha(repository / producer.CONTRACT_RELATIVE):
        raise ExclusionAuditVerificationError("walk-forward contract binding differs")
    for name, root, verification_path in (
        ("composite", producer.COMPOSITE_ROOT, producer.COMPOSITE_VERIFICATION),
        ("price_only", producer.PRICE_ROOT, producer.PRICE_VERIFICATION),
        ("baselines", producer.BASELINE_ROOT, producer.BASELINE_VERIFICATION),
    ):
        result = base._read(repository / root / "RESULT.json")
        base._validate_seal(result)
        receipt = base._read(repository / verification_path)
        base._validate_seal(receipt, "verification_digest")
        expected = audit.get("upstream", {}).get(name, {})
        if not all((
            receipt.get("passed") is True,
            receipt.get("producer_result_digest") == result.get("result_digest"),
            expected.get("producer_result_digest") == result.get("result_digest"),
            expected.get("producer_result_sha256")
                == _sha(repository / root / "RESULT.json"),
            expected.get("verification_digest") == receipt.get("verification_digest"),
            expected.get("verification_sha256") == _sha(repository / verification_path),
        )):
            raise ExclusionAuditVerificationError(f"upstream binding differs: {name}")
    reconstructed = reconstruct(repository)
    for key, value in reconstructed.items():
        if audit.get(key) != value:
            raise ExclusionAuditVerificationError(f"audit reconstruction differs: {key}")
    if not all((
        audit.get("schema_version") == producer.SCHEMA,
        audit.get("status") == "repair_required_before_walk_forward_predictions",
        audit.get("outcomes_or_labels_used_for_repair_selection") is False,
        audit.get("historical_walk_forward_query_outcomes_opened") is False,
        audit.get("final_period_result_opened") is False,
        audit.get("production_promotion_authorized") is False,
    )):
        raise ExclusionAuditVerificationError("audit claim boundary differs")
    state = {
        "schema_version": SCHEMA,
        "status": "verified_repair_required_before_walk_forward_predictions",
        "passed": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "verifier_commit": verifier_commit,
        "verifier_runtime_sha256": _sha(verifier_path),
        "audit_digest": audit["audit_digest"],
        "audit_sha256": _sha(repository / producer.OUTPUT_RELATIVE / "AUDIT.json"),
        "query_count": reconstructed["query_count"],
        "total_links": reconstructed["total_links"],
        "unique_episodes": reconstructed["unique_episodes"],
        "method_affected_queries": {
            key: value["affected_queries"]
            for key, value in reconstructed["methods"].items()
        },
        "t14_09_outcome_overlap": reconstructed["t14_09_outcome_overlap"],
        "outcomes_missing_from_t14_09": reconstructed[
            "outcomes_missing_from_t14_09"
        ],
        "gates": {
            "all_3936_query_bindings_reconstructed": True,
            "all_314880_links_reconstructed": True,
            "all_four_method_impact_sets_exact": True,
            "episode_identity_union_exact": True,
            "t14_09_identity_only_overlap_exact": True,
            "repair_selection_outcome_blind": True,
        },
        "outcomes_or_labels_used_for_repair_selection": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
    }
    return base._sealed(state, "verification_digest")


def publish(repository: Path) -> Path:
    value = verify(repository)
    root = repository.resolve(strict=True) / OUTPUT_RELATIVE
    if root.exists() or root.is_symlink():
        raise ExclusionAuditVerificationError("audit verification output exists")
    root.mkdir(parents=True)
    path = root / "VERIFIED.json"
    base._atomic(path, value)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args()
    print(publish(args.repository))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
