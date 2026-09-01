"""Verify the T14-09 evidence-card contract before query-level aggregation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from market_analogues.product_contract import load_product_contract
from market_analogues.types import stable_hash


SCHEMA = "m04r14-t14-09-evidence-card-contract-verification-v1"
CONTRACT = Path("config/m04r14-t14-09-evidence-card-contract.json")
M00 = Path("config/case-memory-contract.yaml")
OUTCOME_CONTRACT = Path("config/m04r14-t14-09-outcome-contract.json")
REGISTRY = Path("config/data/analogues/m04r14/nasdaq-shadow-denominator-v1/query-registry.json")
STORE = Path("config/data/analogues/m04r14/t14-09-full-outcome-store-v1/SEALED.json")
FULL_VERIFICATION = Path(
    "config/data/analogues/m04r14/t14-09-full-outcome-store-v1-verification/VERIFIED.json"
)
PLAN = Path("docs/t14-09-evidence-cards-plan.html")
OUTPUT = Path(
    "config/data/analogues/m04r14/t14-09-evidence-card-contract-verification-v1"
)
TOP_KEYS = {
    "schema_version", "contract_id", "status", "upstream", "population",
    "raw_evidence", "primary_evidence", "summaries", "sensitivity_panels",
    "abstention", "artifacts", "verification", "contract_digest",
}


class EvidenceContractError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise EvidenceContractError(f"regular file required: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise EvidenceContractError(f"regular JSON file required: {path}")
    raw = path.read_bytes()

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise EvidenceContractError(f"duplicate JSON key: {path}:{key}")
            result[key] = value
        return result

    value = json.loads(
        raw, object_pairs_hook=pairs,
        parse_constant=lambda item: (_ for _ in ()).throw(
            EvidenceContractError(f"non-finite JSON: {path}:{item}")
        ),
    )
    if type(value) is not dict:
        raise EvidenceContractError(f"JSON object required: {path}")
    return value, raw


def _receipt(value: Mapping[str, Any], *, timing: bool = False) -> bool:
    omitted = {"result_digest", "created_at"}
    if timing:
        omitted.add("partition_elapsed_seconds")
    return value.get("result_digest") == stable_hash({
        key: item for key, item in value.items() if key not in omitted
    })


def validate(repository: Path, contract_path: Path | None = None) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    path = (contract_path or repository / CONTRACT).resolve(strict=True)
    contract, raw = _read(path)
    if set(contract) != TOP_KEYS:
        raise EvidenceContractError("evidence contract keys differ")
    state = {key: value for key, value in contract.items() if key != "contract_digest"}
    if not all((
        contract.get("schema_version") == "m04r14-t14-09-evidence-card-contract-v1",
        contract.get("contract_id") == "nasdaq-descriptive-analogue-evidence-cards-v1",
        contract.get("status") == "frozen_before_query_level_outcome_aggregation",
        contract.get("contract_digest") == stable_hash(state),
    )):
        raise EvidenceContractError("evidence contract identity or self-digest differs")

    upstream = contract["upstream"]
    m00 = load_product_contract(repository / M00)
    outcome, outcome_raw = _read(repository / OUTCOME_CONTRACT)
    registry, registry_raw = _read(repository / REGISTRY)
    store, _ = _read(repository / STORE)
    verified, verified_raw = _read(repository / FULL_VERIFICATION)
    if not all((
        upstream.get("m00_contract_digest") == m00.digest,
        upstream.get("m00_sha256") == _sha(repository / M00),
        upstream.get("outcome_contract_digest") == outcome.get("contract_digest"),
        upstream.get("outcome_contract_sha256") == sha256(outcome_raw).hexdigest(),
        upstream.get("registry_digest") == registry.get("registry_digest"),
        upstream.get("registry_sha256") == sha256(registry_raw).hexdigest(),
        upstream.get("full_store_result_digest") == store.get("result_digest"),
        upstream.get("full_verification_result_digest") == verified.get("result_digest"),
        upstream.get("full_verification_sha256") == sha256(verified_raw).hexdigest(),
        _receipt(store, timing=True), store.get("passed") is True,
        _receipt(verified), verified.get("passed") is True,
        verified.get("evidence_cards_authorized") is True,
        verified.get("production_promotion_authorized") is False,
    )):
        raise EvidenceContractError("evidence upstream binding differs")

    population = contract["population"]
    raw_rules = contract["raw_evidence"]
    primary = contract["primary_evidence"]
    summaries = contract["summaries"]
    weighted = summaries.get("locked_weighted", {})
    if not all((
        population.get("dataset_id") == "nasdaq",
        population.get("query_count") == 3270,
        population.get("raw_neighbors_per_query") == 20,
        population.get("query_link_count") == 65400,
        population.get("unique_outcome_episode_count") == 56378,
        population.get("horizons_sessions") == [5, 10, 20, 40, 60, 126],
        population.get("primary_horizon_sessions") == 20,
        raw_rules.get("display_before_summaries") is True,
        raw_rules.get("preserve_original_rank_1_through_20") is True,
        raw_rules.get("show_counterexamples") is True,
        raw_rules.get("show_censored_and_ambiguous") is True,
        primary.get("exclude_query_symbol_to_separate_panel") is True,
        primary.get("maximum_episodes_per_matched_symbol") == 1,
        primary.get("duplicate_selection") == "lowest_original_match_rank",
        primary.get("event_cluster_contribution_cap") == 1,
        primary.get("event_cluster_identity_status") == "unavailable_no_point_in_time_cluster_labels",
        primary.get("minimum_effective_sample_size") == 10,
        primary.get("eligible_when") == "stored_complete_outcome_and_completion_timestamp_lte_query_cutoff",
        primary.get("censored_policy") == "show_but_exclude_from_measure_denominators",
        primary.get("ambiguous_primary_policy") == "show_but_exclude_from_directional_denominator",
        summaries.get("unweighted") is True,
        weighted.get("enabled") is True,
        weighted.get("unnormalized_formula") == "2 ** (-(original_match_rank - 1) / 10)",
        weighted.get("normalization") == "renormalize_within_each_eligible_measure_denominator",
        weighted.get("outcomes_cannot_change_weights") is True,
    )):
        raise EvidenceContractError("raw/dependence/weight semantics differ")

    sensitivity = contract["sensitivity_panels"]
    abstain = contract["abstention"]
    verification = contract["verification"]
    forbidden = set(abstain.get("forbidden_claims", []))
    if not all((
        sensitivity.get("neighbor_count") == "report_prefixes_5_10_15_20_using_identical_rules",
        sensitivity.get("same_regime_vs_different_regime") == "unavailable_no_point_in_time_regime_labels",
        sensitivity.get("same_sector_vs_cross_sector") == "unavailable_no_point_in_time_sector_history",
        abstain.get("predictive_claim_status") == "always_abstain_until_expanding_window_calibration_passes",
        abstain.get("always_present_reason") == "failed_calibration",
        {"calibrated_probability", "expected_profit", "buy_or_sell_recommendation", "guaranteed_future_direction"}.issubset(forbidden),
        verification.get("independent_reconstruction") is True,
        verification.get("all_queries_required") == 3270,
        verification.get("all_links_required") == 65400,
        verification.get("exact_row_and_summary_equality") is True,
        verification.get("outcome_mutation_must_not_change_raw_ranks") is True,
        verification.get("production_promotion_authorized") is False,
        verification.get("query_level_outcome_aggregation_opened") is False,
    )):
        raise EvidenceContractError("sensitivity/abstention/verification semantics differ")
    parser = HTMLParser()
    parser.feed((repository / PLAN).read_text())
    parser.close()
    result_state = {
        "schema_version": SCHEMA,
        "status": "verified_before_query_level_outcome_aggregation",
        "passed": True,
        "contract_digest": contract["contract_digest"],
        "contract_sha256": sha256(raw).hexdigest(),
        "full_verification_result_digest": verified["result_digest"],
        "query_count": 3270, "query_link_count": 65400,
        "query_level_outcome_aggregation_opened": False,
        "production_promotion_authorized": False,
    }
    return {**result_state, "result_digest": stable_hash(result_state)}


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise EvidenceContractError("evidence contract verification root exists")
    path.mkdir(parents=False)
    descriptor = os.open(path / "VERIFIED.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json.dumps({
            **value, "created_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    repository = args.repository.resolve(strict=True)
    result = validate(repository, args.contract)
    if not args.dry_run:
        _publish(repository / OUTPUT, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
