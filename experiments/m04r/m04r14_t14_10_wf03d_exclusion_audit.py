"""Audit WF-03 neighbour stores against the stricter walk-forward symbol rule."""
from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import pandas as pd

from market_analogues.types import stable_hash
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base


SCHEMA = "m04r14-t14-10-wf03d-exclusion-audit-v1"
OUTPUT_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf03d-exclusion-audit-v1"
)
CONTRACT_RELATIVE = Path("config/m04r14-t14-10-walk-forward-contract.json")
COMPOSITE_ROOT = Path(
    "config/data/analogues/m04r14/t14-10-wf03-composite-batch-v2"
)
COMPOSITE_VERIFICATION = Path(
    "config/data/analogues/m04r14/"
    "t14-10-wf03-composite-batch-v2-verification/VERIFIED.json"
)
PRICE_ROOT = Path(
    "config/data/analogues/m04r14/t14-10-wf03-combined-batch-v5"
)
PRICE_VERIFICATION = Path(
    "config/data/analogues/m04r14/"
    "t14-10-wf03-combined-batch-v5-verification/VERIFIED.json"
)
BASELINE_ROOT = Path(
    "config/data/analogues/m04r14/t14-10-wf03-baseline-batch-v2"
)
BASELINE_VERIFICATION = Path(
    "config/data/analogues/m04r14/"
    "t14-10-wf03-baseline-batch-v2-verification/VERIFIED.json"
)
OLD_OUTCOMES = Path(
    "config/data/analogues/m04r14/"
    "t14-09-full-outcome-store-v1/episode-outcomes.parquet"
)
EXPECTED_QUERIES = 3_936
TOP_K = 20


class ExclusionAuditError(RuntimeError):
    pass


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repository, text=True, capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ExclusionAuditError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _verified_upstream(
    repository: Path, root: Path, verification_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = base._read(repository / root / "RESULT.json")
    base._validate_seal(result)
    verification = base._read(repository / verification_path)
    base._validate_seal(verification, "verification_digest")
    if verification.get("passed") is not True \
            or verification.get("producer_result_digest") != result["result_digest"] \
            or verification.get("outcomes_or_labels_used") is not False \
            or verification.get("historical_walk_forward_query_outcomes_opened") is not False \
            or verification.get("final_period_result_opened") is not False:
        raise ExclusionAuditError(f"upstream verification differs: {root}")
    return result, verification


def _groups(
    composite: Mapping[str, Any], price: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Sequence[Mapping[str, Any]]]:
    try:
        return {
            "composite": composite["retrieval"]["matches"],
            "price_only": price["matches"],
            "deterministic_random": baseline["random_neighbors"],
            "recent_return_volatility": baseline["rank_neighbors"],
        }
    except (KeyError, TypeError) as exc:
        raise ExclusionAuditError("neighbour case layout differs") from exc


def _episode_identity_digest(values: Mapping[str, str]) -> str:
    return stable_hash([
        {"episode_id": episode_id, "symbol": values[episode_id]}
        for episode_id in sorted(values)
    ])


def audit(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain"):
        raise ExclusionAuditError("exclusion audit requires a clean commit")
    contract = base._read(repository / CONTRACT_RELATIVE)
    if contract.get("evidence_methods", {}).get("same_query_symbol_excluded") is not True \
            or contract.get("verification", {}).get(
                "historical_walk_forward_query_outcomes_opened"
            ) is not False \
            or contract.get("verification", {}).get("final_period_result_opened") is not False:
        raise ExclusionAuditError("walk-forward exclusion contract differs")
    registry, _by_id = base._registry(repository)
    rows = registry.get("queries_data")
    if type(rows) is not list or len(rows) != EXPECTED_QUERIES:
        raise ExclusionAuditError("walk-forward registry differs")
    upstream = {}
    for name, root, verification in (
        ("composite", COMPOSITE_ROOT, COMPOSITE_VERIFICATION),
        ("price_only", PRICE_ROOT, PRICE_VERIFICATION),
        ("baselines", BASELINE_ROOT, BASELINE_VERIFICATION),
    ):
        result, verified = _verified_upstream(repository, root, verification)
        upstream[name] = {
            "producer_result_digest": result["result_digest"],
            "producer_result_sha256": _sha(repository / root / "RESULT.json"),
            "verification_digest": verified["verification_digest"],
            "verification_sha256": _sha(repository / verification),
        }
    counts = {
        method: {"links": 0, "unique_episodes": set(), "affected_queries": []}
        for method in (
            "composite", "price_only", "deterministic_random",
            "recent_return_volatility",
        )
    }
    all_episodes: set[str] = set()
    episode_symbols: dict[str, str] = {}
    query_inventory: list[dict[str, Any]] = []
    for row in rows:
        query_id = str(row["episode_id"])
        cases = (
            base._read(repository / COMPOSITE_ROOT / "cases" / f"{query_id}.json"),
            base._read(repository / PRICE_ROOT / "cases" / f"{query_id}.json"),
            base._read(repository / BASELINE_ROOT / "cases" / f"{query_id}.json"),
        )
        for case in cases:
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
                raise ExclusionAuditError(f"query binding differs: {query_id}")
        affected = []
        for method, neighbours in _groups(*cases).items():
            if type(neighbours) is not list or len(neighbours) != TOP_K:
                raise ExclusionAuditError(f"{method} inventory differs: {query_id}")
            symbols = [item.get("symbol") for item in neighbours]
            if any(type(value) is not str or not value for value in symbols) \
                    or len(symbols) != len(set(symbols)):
                raise ExclusionAuditError(f"{method} symbol diversity differs: {query_id}")
            identifiers = [item.get("episode_id") for item in neighbours]
            try:
                identifiers_valid = all(
                    type(value) is str and len(bytes.fromhex(value)) == 12
                    for value in identifiers
                )
            except ValueError:
                identifiers_valid = False
            if not identifiers_valid or len(identifiers) != len(set(identifiers)):
                raise ExclusionAuditError(f"{method} episode identity differs: {query_id}")
            same = sum(symbol == row["symbol"] for symbol in symbols)
            if same > 1:
                raise ExclusionAuditError(f"{method} query symbol repeats: {query_id}")
            counts[method]["links"] += len(neighbours)
            counts[method]["unique_episodes"].update(identifiers)
            if same:
                counts[method]["affected_queries"].append(query_id)
                affected.append(method)
            for episode_id, symbol in zip(identifiers, symbols, strict=True):
                prior = episode_symbols.setdefault(str(episode_id), str(symbol))
                if prior != symbol:
                    raise ExclusionAuditError("episode symbol differs across stores")
            all_episodes.update(str(value) for value in identifiers)
        query_inventory.append({"query_id": query_id, "affected_methods": affected})
    old_outcomes = pd.read_parquet(
        repository / OLD_OUTCOMES, columns=["episode_id"],
    )
    old_ids = set(old_outcomes["episode_id"].astype(str))
    methods = {}
    for method, values in counts.items():
        affected_ids = sorted(values["affected_queries"])
        unique_ids = values["unique_episodes"]
        methods[method] = {
            "links": values["links"],
            "unique_episodes": len(unique_ids),
            "same_symbol_links": len(affected_ids),
            "affected_queries": len(affected_ids),
            "affected_query_ids": affected_ids,
            "affected_query_digest": stable_hash(affected_ids),
            "t14_09_outcome_overlap": len(unique_ids & old_ids),
        }
    state = {
        "schema_version": SCHEMA,
        "status": "repair_required_before_walk_forward_predictions",
        "implementation_commit": _git(repository, "rev-parse", "HEAD"),
        "runtime_sha256": _sha(Path(__file__).resolve()),
        "walk_forward_contract_digest": contract["contract_digest"],
        "walk_forward_contract_sha256": _sha(repository / CONTRACT_RELATIVE),
        "registry_digest": registry["registry_digest"],
        "query_count": len(rows),
        "methods": methods,
        "total_links": sum(value["links"] for value in methods.values()),
        "unique_episodes": len(all_episodes),
        "episode_identity_digest": _episode_identity_digest(episode_symbols),
        "query_impact_digest": stable_hash(query_inventory),
        "t14_09_unique_outcomes": len(old_ids),
        "t14_09_outcome_overlap": len(all_episodes & old_ids),
        "outcomes_missing_from_t14_09": len(all_episodes - old_ids),
        "upstream": upstream,
        "outcomes_or_labels_used_for_repair_selection": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
    }
    return base._sealed(state, "audit_digest")


def publish(repository: Path) -> Path:
    result = audit(repository)
    root = repository.resolve(strict=True) / OUTPUT_RELATIVE
    if root.exists() or root.is_symlink():
        raise ExclusionAuditError("exclusion audit output already exists")
    root.mkdir(parents=True)
    path = root / "AUDIT.json"
    base._atomic(path, result)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args()
    path = publish(args.repository)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
