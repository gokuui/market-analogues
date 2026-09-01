"""Verify T14-10 walk-forward rules before historical query outcomes are opened."""
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


SCHEMA = "m04r14-t14-10-walk-forward-contract-verification-v1"
CONTRACT = Path("config/m04r14-t14-10-walk-forward-contract.json")
M00 = Path("config/case-memory-contract.yaml")
OUTCOME_CONTRACT = Path("config/m04r14-t14-09-outcome-contract.json")
EVIDENCE_CONTRACT = Path("config/m04r14-t14-09-evidence-card-contract.json")
EVIDENCE_STORE = Path(
    "config/data/analogues/m04r14/t14-09-evidence-card-store-v1/SEALED.json"
)
EVIDENCE_VERIFICATION = Path(
    "config/data/analogues/m04r14/t14-09-evidence-card-store-v1-verification/VERIFIED.json"
)
PLAN = Path("docs/t14-10-walk-forward-calibration-plan.html")
OUTPUT = Path(
    "config/data/analogues/m04r14/t14-10-walk-forward-contract-verification-v1"
)
TOP_KEYS = {
    "schema_version", "contract_id", "status", "upstream", "boundary",
    "query_registry", "temporal_protocol", "evidence_methods", "targets",
    "probability_construction", "continuous_forecasts", "abstention",
    "baselines", "metrics", "inference", "acceptance", "artifacts",
    "verification", "contract_digest",
}


class WalkForwardContractError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise WalkForwardContractError(f"regular file required: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise WalkForwardContractError(f"regular JSON file required: {path}")
    raw = path.read_bytes()

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise WalkForwardContractError(f"duplicate JSON key: {path}:{key}")
            result[key] = value
        return result

    value = json.loads(
        raw, object_pairs_hook=pairs,
        parse_constant=lambda item: (_ for _ in ()).throw(
            WalkForwardContractError(f"non-finite JSON: {path}:{item}")
        ),
    )
    if type(value) is not dict:
        raise WalkForwardContractError(f"JSON object required: {path}")
    return value, raw


def _receipt(value: Mapping[str, Any], *, timing: bool = False) -> bool:
    omitted = {"result_digest", "created_at"}
    if timing:
        omitted.add("elapsed_seconds")
    return value.get("result_digest") == stable_hash({
        key: item for key, item in value.items() if key not in omitted
    })


def validate(repository: Path, contract_path: Path | None = None) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    path = (contract_path or repository / CONTRACT).resolve(strict=True)
    contract, raw = _read(path)
    if set(contract) != TOP_KEYS:
        raise WalkForwardContractError("walk-forward contract keys differ")
    state = {key: value for key, value in contract.items() if key != "contract_digest"}
    if not all((
        contract.get("schema_version") == "m04r14-t14-10-walk-forward-contract-v1",
        contract.get("contract_id") == "nasdaq-analogue-walk-forward-falsification-v1",
        contract.get("status") == "frozen_before_historical_walk_forward_query_outcomes",
        contract.get("contract_digest") == stable_hash(state),
    )):
        raise WalkForwardContractError("walk-forward identity or self-digest differs")

    upstream = contract["upstream"]
    m00 = load_product_contract(repository / M00)
    outcome, outcome_raw = _read(repository / OUTCOME_CONTRACT)
    evidence, evidence_raw = _read(repository / EVIDENCE_CONTRACT)
    store, store_raw = _read(repository / EVIDENCE_STORE)
    verified, verified_raw = _read(repository / EVIDENCE_VERIFICATION)
    if not all((
        upstream.get("m00_contract_digest") == m00.digest,
        upstream.get("m00_sha256") == _sha(repository / M00),
        upstream.get("outcome_contract_digest") == outcome.get("contract_digest"),
        upstream.get("outcome_contract_sha256") == sha256(outcome_raw).hexdigest(),
        upstream.get("evidence_card_contract_digest") == evidence.get("contract_digest"),
        upstream.get("evidence_card_contract_sha256") == sha256(evidence_raw).hexdigest(),
        upstream.get("evidence_store_result_digest") == store.get("result_digest"),
        upstream.get("evidence_store_sha256") == sha256(store_raw).hexdigest(),
        upstream.get("evidence_verification_result_digest") == verified.get("result_digest"),
        upstream.get("evidence_verification_sha256") == sha256(verified_raw).hexdigest(),
        _receipt(store, timing=True), store.get("passed") is True,
        _receipt(verified), verified.get("passed") is True,
        verified.get("walk_forward_authorized") is True,
        verified.get("production_promotion_authorized") is False,
    )):
        raise WalkForwardContractError("walk-forward upstream binding differs")

    boundary = contract["boundary"]
    registry = contract["query_registry"]
    if not all((
        boundary.get("purpose") == "falsify_or_validate_historical_conditional_evidence_not_optimize_a_trading_rule",
        boundary.get("dataset_id") == "nasdaq",
        boundary.get("retrieval_and_outcome_formulas_frozen") is True,
        boundary.get("trading_costs_positions_and_profit_metrics_in_scope") is False,
        boundary.get("historical_membership_delisting_returns_sector_history_available") is False,
        boundary.get("production_promotion_authorized") is False,
        registry.get("cutoff_rule") == "last_observed_nasdaq_benchmark_session_of_each_calendar_month",
        registry.get("warmup_period") == ["2012-01-01", "2013-12-31"],
        registry.get("scored_period") == ["2014-01-01", "2025-08-31"],
        registry.get("locked_source_coverage_cutoff") == "2026-03-30",
        registry.get("target_queries_per_month") == 24,
        registry.get("target_per_cell_per_month") == 4,
        registry.get("underfilled_cell_policy") == "retain_shortfall_without_cross_cell_replacement",
        registry.get("minimum_required_prior_sessions") == 252,
        registry.get("future_observations_may_affect_selection") is False,
        registry.get("minimum_scored_queries") == 3000,
        registry.get("registry_must_freeze_before_retrieval") is True,
    )):
        raise WalkForwardContractError("query registry or product boundary differs")

    temporal = contract["temporal_protocol"]
    folds = temporal.get("folds")
    expected_folds = [
        ("development", "2014-01-01", "2017-12-31", "implementation_diagnostic_no_threshold_tuning"),
        ("validation_1", "2018-01-01", "2019-12-31", "locked_validation"),
        ("validation_2", "2020-01-01", "2021-12-31", "locked_validation"),
        ("validation_3", "2022-01-01", "2023-12-31", "locked_validation"),
        ("final_untouched", "2024-01-01", "2025-08-31", "single_open_final_test"),
    ]
    observed_folds = [
        (row.get("fold_id"), row.get("start"), row.get("end"), row.get("role"))
        for row in folds if type(row) is dict
    ] if type(folds) is list else []
    if not all((
        temporal.get("kind") == "purged_expanding_window",
        temporal.get("purge_sessions") == 126,
        temporal.get("outcome_eligible_for_evidence_when") == "completion_timestamp_lte_query_cutoff",
        observed_folds == expected_folds,
        temporal.get("final_period_open_once") is True,
        temporal.get("failed_fold_cannot_be_relabelled_or_removed") is True,
        temporal.get("representation_or_threshold_change_requires_new_version_and_new_final_period") is True,
    )):
        raise WalkForwardContractError("temporal folds or leakage boundary differs")

    methods = contract["evidence_methods"]
    targets = contract["targets"]
    primary = targets.get("primary", {})
    probability = contract["probability_construction"]
    continuous = contract["continuous_forecasts"]
    if not all((
        methods.get("primary") == "composite_top20_one_best_per_symbol_locked_rank_weight",
        methods.get("neighbor_prefixes") == [5, 10, 15, 20],
        methods.get("same_query_symbol_excluded") is True,
        methods.get("maximum_per_matched_symbol") == 1,
        methods.get("rank_weight_formula") == "2 ** (-(original_match_rank - 1) / 10)",
        methods.get("minimum_effective_rows") == 10,
        methods.get("outcomes_cannot_affect_retrieval_rank_or_weight") is True,
        methods.get("forced_score_lane_includes_abstained_queries") is True,
        primary.get("classes") == ["favorable_first", "adverse_first", "no_touch"],
        primary.get("ambiguous_same_first_touch_bar") == "excluded_and_reported",
        primary.get("censored_or_incomplete") == "excluded_and_reported",
        primary.get("favorable_barrier_atr") == 2.0,
        primary.get("adverse_barrier_atr") == 1.0,
        primary.get("horizon_sessions") == 20,
        probability.get("dirichlet_alpha_per_class") == 0.5,
        probability.get("beta_alpha") == 0.5,
        probability.get("no_posthoc_platt_isotonic_or_model_fit") is True,
        probability.get("probabilities_sum_to_one_tolerance") == 1e-15,
        continuous.get("quantiles") == [0.1, 0.25, 0.5, 0.75, 0.9],
        continuous.get("quantile_method") == "weighted_inverted_cdf",
    )):
        raise WalkForwardContractError("evidence probability or target semantics differ")

    abstain = contract["abstention"]
    novelty = abstain.get("novelty", {})
    instability = abstain.get("neighborhood_instability", {})
    baselines = contract["baselines"]
    required_baselines = {
        "unconditional_market_frequency", "regime_only_frequency",
        "deterministic_random_historical_neighbors", "recent_return_matched_neighbors",
        "price_only_similarity", "sector_frequency",
        "all_baselines_use_identical_target_smoothing_and_censor_rules",
    }
    if not all((
        novelty.get("reference") == "prior_scored_queries_only",
        novelty.get("minimum_reference_queries") == 250,
        novelty.get("threshold_quantile") == 0.95,
        novelty.get("threshold_method") == "linear_quantile",
        instability.get("maximum_allowed") == 0.2,
        abstain.get("poor_data_quality") == "always_present_for_missing_membership_delisting_sector_and_event_cluster_history",
        abstain.get("failed_calibration") == "always_present_until_this_contract_passes",
        abstain.get("abstention_cannot_remove_query_from_forced_score_lane") is True,
        set(baselines) == required_baselines,
        baselines.get("sector_frequency") == "unavailable_no_point_in_time_sector_history_no_substitution",
        baselines.get("all_baselines_use_identical_target_smoothing_and_censor_rules") is True,
    )):
        raise WalkForwardContractError("abstention or baseline semantics differ")

    metrics = contract["metrics"]
    inference = contract["inference"]
    acceptance = contract["acceptance"]
    if not all((
        metrics.get("primary") == "multiclass_brier_score",
        {"multiclass_log_loss", "brier_skill_score", "reliability_diagram", "equal_count_ece"}.issubset(metrics.get("required_probability", [])),
        metrics.get("reliability_bins") == 10,
        metrics.get("minimum_rows_per_reliability_bin") == 30,
        metrics.get("hit_rate_alone_can_pass") is False,
        inference.get("dependence_unit") == "calendar_month_mean_loss_not_individual_stock_rows",
        inference.get("paired_month_block_bootstrap_resamples") == 10000,
        inference.get("paired_month_block_bootstrap_seed") == 20260901,
        inference.get("paired_month_block_length") == 3,
        inference.get("diebold_mariano_hac_lags_months") == 3,
        inference.get("familywise_alpha") == 0.05,
        inference.get("multiple_comparison_correction") == "holm",
        inference.get("failure_to_reject_is_not_stated_as_proof") is True,
        acceptance.get("primary_lane") == "forced_score_all_evaluable_queries",
        acceptance.get("final_multiclass_brier_skill_strictly_positive_vs_every_available_baseline") is True,
        acceptance.get("holm_adjusted_one_sided_loss_test_vs_unconditional_and_price_only_below") == 0.05,
        acceptance.get("no_validation_fold_negative_brier_skill_vs_unconditional") is True,
        acceptance.get("minimum_final_nonabstained_fraction") == 0.5,
        acceptance.get("minimum_final_evaluable_queries") == 400,
        acceptance.get("all_leakage_manifest_and_censor_checks_pass") is True,
        acceptance.get("product_calibration_pass_requires_missing_sector_membership_delisting_event_cluster_controls") is True,
        acceptance.get("production_promotion_authorized") is False,
    )):
        raise WalkForwardContractError("metric inference or acceptance semantics differ")

    verification = contract["verification"]
    if not all((
        verification.get("synthetic_formula_oracle_before_real_run") is True,
        verification.get("outcome_blind_registry_verification_before_retrieval") is True,
        verification.get("independent_reconstruction") is True,
        verification.get("final_period_result_opened") is False,
        verification.get("historical_walk_forward_query_outcomes_opened") is False,
        verification.get("failed_trials_append_to_ledger") is True,
        verification.get("production_promotion_authorized") is False,
    )):
        raise WalkForwardContractError("verification boundary differs")
    parser = HTMLParser()
    parser.feed((repository / PLAN).read_text())
    parser.close()
    result_state = {
        "schema_version": SCHEMA,
        "status": "verified_before_historical_walk_forward_query_outcomes",
        "passed": True,
        "contract_digest": contract["contract_digest"],
        "contract_sha256": sha256(raw).hexdigest(),
        "evidence_store_result_digest": store["result_digest"],
        "evidence_verification_result_digest": verified["result_digest"],
        "warmup_period": registry["warmup_period"],
        "scored_period": registry["scored_period"],
        "fold_count": len(folds),
        "target_queries_per_month": registry["target_queries_per_month"],
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
    }
    return {**result_state, "result_digest": stable_hash(result_state)}


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise WalkForwardContractError("walk-forward contract verification root exists")
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
