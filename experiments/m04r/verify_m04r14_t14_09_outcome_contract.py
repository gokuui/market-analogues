"""Verify the T14-09 operational outcome contract without opening future bars."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from market_analogues.product_contract import load_product_contract
from market_analogues.types import stable_hash


SCHEMA = "m04r14-t14-09-outcome-contract-verification-v1"
CONTRACT = Path("config/m04r14-t14-09-outcome-contract.json")
SEMANTIC = Path(
    "config/data/analogues/m04r14/nasdaq-shadow-interrupted-verification-v1/VERIFIED.json"
)
AUDIT = Path(
    "config/data/analogues/m04r14/nasdaq-shadow-audit-comparison-v1/RESULT.json"
)
OUTPUT = Path(
    "config/data/analogues/m04r14/t14-09-outcome-contract-verification-v1"
)
TOP_KEYS = {
    "schema_version", "contract_id", "status", "contract_digest",
    "parent_contract", "retrieval_inputs", "scope", "origin", "session_axis",
    "atr", "primary_barrier", "measures", "benchmark_alignment",
    "causal_embargo", "censoring_and_limitations", "artifacts", "execution",
    "verification", "claims",
}


class OutcomeContractError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise OutcomeContractError(f"regular file required: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise OutcomeContractError(f"regular JSON file required: {path}")
    raw = path.read_bytes()

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise OutcomeContractError(f"duplicate JSON key: {path}:{key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw, object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                OutcomeContractError(f"non-finite JSON: {path}:{item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OutcomeContractError(f"invalid JSON: {path}") from exc
    if type(value) is not dict:
        raise OutcomeContractError(f"JSON object required: {path}")
    return value, raw


def _sealed_receipt(value: Mapping[str, Any]) -> bool:
    state = {
        key: item for key, item in value.items()
        if key not in {"result_digest", "created_at"}
    }
    return value.get("result_digest") == stable_hash(state)


def validate(repository: Path, contract_path: Path | None = None) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    path = (contract_path or repository / CONTRACT).resolve(strict=True)
    contract, raw = _read(path)
    if set(contract) != TOP_KEYS:
        raise OutcomeContractError(
            f"contract keys differ: missing={sorted(TOP_KEYS - set(contract))}, "
            f"unknown={sorted(set(contract) - TOP_KEYS)}"
        )
    state = {key: value for key, value in contract.items() if key != "contract_digest"}
    if contract.get("schema_version") != "m04r14-t14-09-outcome-contract-v1" \
            or contract.get("contract_id") != "nasdaq-causal-forward-outcomes-v1" \
            or contract.get("status") != "frozen_before_real_forward_outcome_access" \
            or contract.get("contract_digest") != stable_hash(state):
        raise OutcomeContractError("contract identity or self-digest differs")

    parent = contract["parent_contract"]
    product_path = repository / str(parent.get("source_path"))
    product = load_product_contract(product_path)
    if not all((
        parent.get("contract_id") == product.contract_id,
        parent.get("contract_digest") == product.digest,
        parent.get("source_sha256") == _sha(product_path),
        product.payload["outcomes"]["outcomes_may_affect_similarity"] is False,
        product.payload["retrieval"]["retrieval_must_be_frozen_before_outcomes"] is True,
    )):
        raise OutcomeContractError("parent M00 contract binding differs")

    retrieval = contract["retrieval_inputs"]
    semantic, semantic_raw = _read(repository / SEMANTIC)
    audit, audit_raw = _read(repository / AUDIT)
    if not all((
        _sealed_receipt(semantic), semantic.get("semantic_passed") is True,
        semantic.get("verified_cases") == 3270,
        semantic.get("verified_matches") == 65400,
        semantic.get("real_forward_outcomes_accessed") is False,
        _sealed_receipt(audit), audit.get("passed") is True,
        audit.get("matching_positions") == 240,
        audit.get("real_forward_outcomes_accessed") is False,
        retrieval.get("dataset") == "nasdaq",
        retrieval.get("registry_digest") == semantic.get("registry_digest"),
        retrieval.get("source_content_digest") == semantic.get("source_content_digest"),
        retrieval.get("snapshot_case_manifest_digest") == semantic.get("case_manifest_digest"),
        retrieval.get("snapshot_semantic_result_digest") == semantic.get("result_digest"),
        retrieval.get("snapshot_semantic_receipt_sha256") == sha256(semantic_raw).hexdigest(),
        retrieval.get("audit_comparison_result_digest") == audit.get("result_digest"),
        retrieval.get("audit_comparison_sha256") == sha256(audit_raw).hexdigest(),
        retrieval.get("scheduled_queries") == 3270,
        retrieval.get("retrieved_match_rows") == 65400,
        retrieval.get("retrieval_must_remain_byte_identical") is True,
    )):
        raise OutcomeContractError("frozen retrieval evidence binding differs")

    scope = contract["scope"]
    origin = contract["origin"]
    sessions = contract["session_axis"]
    atr = contract["atr"]
    barrier = contract["primary_barrier"]
    if not all((
        scope.get("decision_mode") == "after_close_daily",
        scope.get("intended_direction") == "long",
        scope.get("compute_once_key") == ["dataset", "episode_id", "source_fingerprint"],
        scope.get("one_outcome_record_per_unique_matched_episode") is True,
        scope.get("same_symbol_acquirer_substitution") is False,
        scope.get("real_forward_outcomes_accessed_before_contract") is False,
        origin.get("origin_price") == "stored_close_at_cutoff",
        origin.get("entry_execution_assumed") is False,
        origin.get("returns_are_descriptive_not_trade_returns") is True,
        origin.get("future_rows_begin") == "first_strictly_later_observed_stock_session",
        sessions.get("horizon_unit") == "subsequent_observed_stock_sessions",
        sessions.get("horizons") == [5, 10, 20, 40, 60, 126],
        sessions.get("calendar_day_approximation_forbidden") is True,
        sessions.get("weekend_or_holiday_forward_fill_forbidden") is True,
        atr.get("lookback_sessions") == 20,
        atr.get("method") == "simple_mean_true_range",
        atr.get("minimum_rows_including_previous_close") == 21,
        atr.get("future_values_allowed") is False,
        barrier.get("horizon_sessions") == 20,
        barrier.get("favorable_price") == "origin_close_plus_2_atr",
        barrier.get("adverse_price") == "origin_close_minus_1_atr",
        barrier.get("same_first_touch_bar") == "ambiguous_excluded_from_directional_denominator",
        barrier.get("no_touch_requires_complete_horizon") is True,
        barrier.get("barrier_prices_fixed_at_origin") is True,
    )):
        raise OutcomeContractError("origin/session/ATR/barrier semantics differ")
    labels = barrier.get("labels")
    if labels != [
        "favorable_first", "adverse_first", "ambiguous_same_first_touch_bar",
        "no_touch", "censored",
    ]:
        raise OutcomeContractError("primary barrier labels differ")

    measures = contract["measures"]
    path_contract = measures.get("normalized_future_path", {})
    benchmark = contract["benchmark_alignment"]
    embargo = contract["causal_embargo"]
    censoring = contract["censoring_and_limitations"]
    if not all((
        measures.get("close_return") == "endpoint_close_divided_by_origin_close_minus_1",
        measures.get("benchmark_relative_return") == "stock_gross_return_divided_by_benchmark_gross_return_minus_1",
        path_contract.get("maximum_steps") == 126,
        path_contract.get("one_row_per_observed_step") is True,
        path_contract.get("partial_paths_preserved") is True,
        path_contract.get("partial_paths_excluded_from_complete_horizon_denominators") is True,
        benchmark.get("origin_requires_exact_session") is True,
        benchmark.get("endpoint_requires_exact_session") is True,
        benchmark.get("forward_fill") is False,
        benchmark.get("nearest_session_substitution") is False,
        embargo.get("attach_outcomes_only_after_retrieval_is_frozen") is True,
        embargo.get("horizon_contributes_to_query_when") == "full_horizon_completion_timestamp_lte_query_cutoff",
        embargo.get("early_barrier_touch_does_not_shorten_embargo") is True,
        embargo.get("outcomes_may_affect_similarity_rank_or_weight") is False,
        censoring.get("source_end_before_horizon") == "right_censored_unknown_delisting_status",
        censoring.get("known_delisting_return_available") is False,
        censoring.get("missing_delisting_return_imputed") is False,
        censoring.get("corporate_action_adjustment_provenance") == "unknown_propagated_as_quality_warning",
        censoring.get("split_or_dividend_repair") == "forbidden_without_separate_fingerprinted_source",
    )):
        raise OutcomeContractError("measure/benchmark/embargo/censor semantics differ")

    artifacts = contract["artifacts"]
    execution = contract["execution"]
    verification = contract["verification"]
    claims = contract["claims"]
    if not all((
        artifacts.get("output_policy") == "create_only_atomic_then_seal",
        artifacts.get("timing_excluded_from_semantic_digest") is True,
        execution.get("load_each_symbol_once_per_process_partition") is True,
        execution.get("deduplicate_matched_episodes_before_computation") is True,
        execution.get("processes") == 12,
        execution.get("resume") == "validate_immutable_partition_before_reuse",
        execution.get("minimum_free_disk_gib_before_start") == 5,
        verification.get("synthetic_oracle_required_before_real_run") is True,
        verification.get("real_smoke") == "preregistered_12_case_audit_sample_coverage_only",
        verification.get("full_verifier") == "independent_recomputation_of_every_episode_outcome_and_path_row",
        verification.get("real_outcome_values_cannot_change_contract_or_retrieval") is True,
        verification.get("required_reconciliation", {}).get("scheduled_query_links") == 65400,
        all(value is False for key, value in claims.items() if key != "historical_descriptive_evidence_only"),
        claims.get("historical_descriptive_evidence_only") is True,
    )):
        raise OutcomeContractError("artifact/execution/verification/claim policy differs")
    result_state = {
        "schema_version": SCHEMA,
        "status": "verified_before_real_forward_outcome_access",
        "passed": True,
        "contract_digest": contract["contract_digest"],
        "contract_sha256": sha256(raw).hexdigest(),
        "parent_contract_digest": product.digest,
        "retrieval_semantic_result_digest": semantic["result_digest"],
        "audit_comparison_result_digest": audit["result_digest"],
        "scheduled_queries": 3270,
        "retrieved_match_rows": 65400,
        "real_forward_outcomes_accessed": False,
        "production_promotion_authorized": False,
    }
    return {**result_state, "result_digest": stable_hash(result_state)}


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise OutcomeContractError("verification root already exists")
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
    value = validate(repository, args.contract)
    if not args.dry_run:
        _publish(repository / OUTPUT, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
