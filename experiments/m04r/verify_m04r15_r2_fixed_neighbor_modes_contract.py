"""Independently verify the R2 contract without opening future outcome values."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq


SCHEMA = "m04r15-r2-fixed-neighbor-modes-contract-verification-v1"
CONTRACT = Path("config/m04r15-r2-fixed-neighbor-modes-contract-v1.json")
PLAN = Path("docs/r2-fixed-neighbor-future-modes-plan.html")
OUTCOME_ROOT = Path("config/data/analogues/m04r14/t14-09-full-outcome-store-v1")
OUTCOME_VERIFICATION = Path(
    "config/data/analogues/m04r14/t14-09-full-outcome-store-v1-verification/VERIFIED.json"
)
EVIDENCE_ROOT = Path("config/data/analogues/m04r14/t14-09-evidence-card-store-v1")
EVIDENCE_VERIFICATION = Path(
    "config/data/analogues/m04r14/t14-09-evidence-card-store-v1-verification/VERIFIED.json"
)
OUTPUT = Path(
    "config/data/analogues/m04r15/r2-fixed-neighbor-modes-contract-verification-v1"
)
TOP_KEYS = {
    "schema_version", "contract_id", "status", "upstream", "population",
    "member_rules", "path_contract", "algorithm", "stability", "output",
    "claim_boundary", "verification", "contract_digest",
}


class R2ContractError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise R2ContractError(message)


def _stable(value: Any) -> str:
    return sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()).hexdigest()


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    _require(path.is_file() and not path.is_symlink(), f"regular JSON required: {path}")
    raw = path.read_bytes()

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            _require(key not in value, f"duplicate JSON key: {path}:{key}")
            value[key] = item
        return value

    try:
        value = json.loads(
            raw, object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                R2ContractError(f"non-finite JSON: {path}:{token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise R2ContractError(f"invalid JSON: {path}") from error
    _require(type(value) is dict, f"JSON object required: {path}")
    return value, raw


def _sha(path: Path) -> str:
    _require(path.is_file() and not path.is_symlink(), f"regular file required: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _parquet(path: Path) -> pq.ParquetFile:
    _require(path.is_file() and not path.is_symlink(), f"regular Parquet required: {path}")
    try:
        return pq.ParquetFile(path)
    except Exception as error:
        raise R2ContractError(f"invalid Parquet: {path}") from error


def _validate_upstream(repository: Path, contract: Mapping[str, Any]) -> None:
    upstream = contract["upstream"]
    _require(set(upstream) == {
        "evidence_analogue_rows_sha256", "evidence_store_result_digest",
        "evidence_store_seal_sha256", "evidence_verification_result_digest",
        "evidence_verification_sha256", "future_paths_sha256",
        "outcome_store_result_digest", "outcome_store_seal_sha256",
        "outcome_verification_result_digest", "outcome_verification_sha256",
        "query_match_links_sha256",
    }, "R2 upstream keys differ")
    outcome_seal, _ = _read_json(repository / OUTCOME_ROOT / "SEALED.json")
    outcome_verified, _ = _read_json(repository / OUTCOME_VERIFICATION)
    evidence_seal, _ = _read_json(repository / EVIDENCE_ROOT / "SEALED.json")
    evidence_verified, _ = _read_json(repository / EVIDENCE_VERIFICATION)
    checks = (
        upstream.get("outcome_store_result_digest") == outcome_seal.get("result_digest"),
        upstream.get("outcome_store_seal_sha256") == _sha(repository / OUTCOME_ROOT / "SEALED.json"),
        upstream.get("outcome_verification_result_digest") == outcome_verified.get("result_digest"),
        upstream.get("outcome_verification_sha256") == _sha(repository / OUTCOME_VERIFICATION),
        upstream.get("evidence_store_result_digest") == evidence_seal.get("result_digest"),
        upstream.get("evidence_store_seal_sha256") == _sha(repository / EVIDENCE_ROOT / "SEALED.json"),
        upstream.get("evidence_verification_result_digest") == evidence_verified.get("result_digest"),
        upstream.get("evidence_verification_sha256") == _sha(repository / EVIDENCE_VERIFICATION),
        upstream.get("query_match_links_sha256") == _sha(repository / OUTCOME_ROOT / "query-match-links.parquet"),
        upstream.get("future_paths_sha256") == _sha(repository / OUTCOME_ROOT / "future-paths.parquet"),
        upstream.get("evidence_analogue_rows_sha256") == _sha(repository / EVIDENCE_ROOT / "analogue-evidence.parquet"),
        outcome_seal.get("passed") is True,
        outcome_seal.get("query_count") == 3270,
        outcome_seal.get("query_links") == 65400,
        outcome_seal.get("path_rows") == 6917999,
        outcome_seal.get("unique_episodes") == 56378,
        outcome_seal.get("outcomes_affected_retrieval") is False,
        outcome_verified.get("passed") is True,
        outcome_verified.get("store_result_digest") == outcome_seal.get("result_digest"),
        evidence_seal.get("passed") is True,
        evidence_seal.get("query_count") == 3270,
        evidence_seal.get("raw_analogue_rows") == 65400,
        evidence_seal.get("predictive_claims_emitted") is False,
        evidence_verified.get("passed") is True,
        evidence_verified.get("store_result_digest") == evidence_seal.get("result_digest"),
        evidence_verified.get("exact_independent_raw_equality") is True,
        evidence_verified.get("predictive_claims_emitted") is False,
    )
    _require(all(checks), "R2 upstream identity or verified state differs")


def _validate_semantics(contract: Mapping[str, Any]) -> None:
    population = contract["population"]
    members = contract["member_rules"]
    path = contract["path_contract"]
    algorithm = contract["algorithm"]
    stability = contract["stability"]
    output = contract["output"]
    claims = contract["claim_boundary"]
    verification = contract["verification"]
    _require(population == {
        "dataset_id": "nasdaq", "duplicate_matched_symbol_links_within_query": 810,
        "future_path_rows": 6917999, "query_count": 3270,
        "query_link_count": 65400, "query_symbol_links": 50,
        "raw_neighbors_per_query": 20, "unique_outcome_episode_count": 56378,
    }, "R2 population differs")
    _require(members == {
        "censored_or_invalid_members_disclosed": True,
        "duplicate_selection": "lowest_original_match_rank",
        "maximum_primary_episodes_per_matched_symbol": 1,
        "outcomes_may_change_primary_member_identity": False,
        "preserve_original_ranks_1_through_20": True,
        "query_symbol_matches": "disclose_in_separate_panel_and_exclude_from_primary_modes",
    }, "R2 member semantics differ")
    _require(path == {
        "cluster_horizon_sessions": 60,
        "completion": "all_steps_1_through_60_present_unique_expected_session_match_true_and_source_bindings_valid",
        "excluded_supporting_channels": [
            "atr_normalized_close_move", "high_return", "low_return",
            "sessions_61_through_126",
        ],
        "imputation": "forbidden",
        "nonfinite_values": "reject_member_and_disclose_reason",
        "views": [
            {"field": "close_return", "id": "absolute_close_return", "label": "cumulative close return"},
            {"field": "benchmark_relative_close_return", "id": "benchmark_relative_close_return", "label": "cumulative benchmark-relative close return"},
        ],
    }, "R2 path semantics differ")
    _require(algorithm == {
        "candidate_k": "integers_1_through_minimum_of_4_and_floor_complete_primary_members_divided_by_3",
        "distance": "mean_absolute_pointwise_difference_over_60_cumulative_return_values",
        "implementation": "deterministic_partitioning_around_medoids_build_and_swap",
        "maximum_modes": 4,
        "medoids_are_observed_paths": True,
        "minimum_members_per_displayed_mode": 3,
        "multi_mode_selection": "greatest_mean_silhouette_among_candidates_passing_size_separation_and_stability",
        "separation_minimum_mean_silhouette": 0.25,
        "tie_break": [
            "smaller_k_for_equal_silhouette", "lower_original_match_rank",
            "lexicographically_smaller_episode_id",
        ],
        "tie_equality": "exact_float64_equality_no_rounding",
    }, "R2 clustering semantics differ")
    _require(stability == {
        "block": "matched_cutoff_calendar_quarter",
        "bootstrap_replicates": 256,
        "fallback": "report_k1_if_no_multi_mode_candidate_passes_or_abstain_if_fewer_than_3_complete_primary_members",
        "minimum_valid_replicates": 205,
        "minimum_valid_replicate_fraction": 0.8,
        "randomness": "sha256_counter_stream_seeded_by_contract_digest_query_case_id_view_id_and_replicate",
        "refit": "weighted_pam_on_calendar_quarter_blocks_sampled_with_replacement_then_assign_all_base_members_to_bootstrap_medoids",
        "statistic": "adjusted_rand_index_against_full_sample_labels",
        "threshold_median_adjusted_rand_index": 0.8,
    }, "R2 stability semantics differ")
    _require(output == {
        "context_slices": "unavailable_until_verified_point_in_time_context_fields_exist",
        "disagreement_banner_required_for_materially_different_stable_modes": True,
        "evidence_panels": [
            "all_raw_members", "primary_excluded_and_censored_members",
            "actual_medoid_paths", "complete_mode_membership",
            "frequency_and_path_envelopes", "returns_mfe_mae_and_barriers",
            "separation_and_stability_diagnostics", "limitations_and_claim_boundary",
        ],
        "negative_analogue": "unavailable_until_outcome_blind_point_in_time_selector_is_verified",
        "version_pinned_cohort_handle": True,
    }, "R2 output semantics differ")
    _require(claims == {
        "allowed_claim": "historical_conditional_descriptive_evidence_only",
        "mode_labels_are_trader_pattern_names": False,
        "predictive_claim_authorized": False,
        "production_promotion_authorized": False,
        "trading_claim_authorized": False,
    }, "R2 claim semantics differ")
    _require(set(verification) == {
        "all_queries_required", "all_raw_links_required",
        "independent_full_reconstruction_after_build", "metadata_only_preflight",
        "outcome_columns_opened_during_contract_verification",
        "outcome_mutation_must_not_change_cohort_or_retrieval",
        "real_future_path_mode_computation_opened", "required_input_schemas",
        "synthetic_gate_required_before_real_paths",
    } and all((
        verification.get("all_queries_required") == 3270,
        verification.get("all_raw_links_required") == 65400,
        verification.get("metadata_only_preflight") is True,
        verification.get("outcome_columns_opened_during_contract_verification") is False,
        verification.get("real_future_path_mode_computation_opened") is False,
        verification.get("synthetic_gate_required_before_real_paths") is True,
        verification.get("independent_full_reconstruction_after_build") is True,
        verification.get("outcome_mutation_must_not_change_cohort_or_retrieval") is True,
    )), "R2 verification semantics differ")


def _validate_metadata(repository: Path, contract: Mapping[str, Any]) -> dict[str, int]:
    required = contract["verification"]["required_input_schemas"]
    link_path = repository / OUTCOME_ROOT / "query-match-links.parquet"
    path_path = repository / OUTCOME_ROOT / "future-paths.parquet"
    evidence_path = repository / EVIDENCE_ROOT / "analogue-evidence.parquet"
    link_file, path_file, evidence_file = map(_parquet, (link_path, path_path, evidence_path))
    _require(link_file.schema_arrow.names == required["query_match_links"], "query-link schema differs")
    _require(path_file.schema_arrow.names == required["future_paths"], "future-path schema differs")
    _require(evidence_file.schema_arrow.names == required["analogue_evidence"], "analogue-evidence schema differs")
    _require(link_file.metadata.num_rows == 65400, "query-link row count differs")
    _require(path_file.metadata.num_rows == 6917999, "future-path row count differs")
    _require(evidence_file.metadata.num_rows == 65400, "analogue-evidence row count differs")

    # Only outcome-blind identity/rank columns are opened. No future-path column is read.
    names = [
        "query_case_id", "query_symbol", "match_rank", "matched_episode_id",
        "matched_symbol", "matched_cutoff",
    ]
    table = pq.read_table(link_path, columns=names).to_pydict()
    rows = list(zip(*(table[name] for name in names), strict=True))
    queries: dict[str, list[tuple[Any, ...]]] = {}
    for row in rows:
        queries.setdefault(str(row[0]), []).append(row)
    _require(len(queries) == 3270, "query identity count differs")
    duplicate_symbols = 0
    query_symbol_links = 0
    for query_rows in queries.values():
        ranks = sorted(int(row[2]) for row in query_rows)
        _require(ranks == list(range(1, 21)), "per-query rank closure differs")
        _require(len({str(row[3]) for row in query_rows}) == 20, "matched episode uniqueness differs")
        seen: set[str] = set()
        for row in sorted(query_rows, key=lambda item: int(item[2])):
            symbol = str(row[4])
            duplicate_symbols += symbol in seen
            seen.add(symbol)
            query_symbol_links += str(row[1]) == symbol
    _require(duplicate_symbols == 810, "duplicate-symbol count differs")
    _require(query_symbol_links == 50, "query-symbol link count differs")
    return {
        "query_count": len(queries), "query_link_count": len(rows),
        "duplicate_matched_symbol_links_within_query": duplicate_symbols,
        "query_symbol_links": query_symbol_links,
        "future_path_rows_from_parquet_metadata": path_file.metadata.num_rows,
    }


def validate(repository: Path, contract_path: Path | None = None) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    selected = contract_path or repository / CONTRACT
    contract, raw = _read_json(selected.resolve(strict=True))
    _require(set(contract) == TOP_KEYS, "R2 contract keys differ")
    state = {key: value for key, value in contract.items() if key != "contract_digest"}
    _require(all((
        contract.get("schema_version") == "m04r15-r2-fixed-neighbor-modes-contract-v1",
        contract.get("contract_id") == "m04r15-r2-fixed-neighbor-future-path-modes-v1",
        contract.get("status") == "frozen_before_real_future_path_mode_computation",
        contract.get("contract_digest") == _stable(state),
    )), "R2 contract identity or self-digest differs")
    _validate_upstream(repository, contract)
    _validate_semantics(contract)
    observed = _validate_metadata(repository, contract)
    parser = HTMLParser()
    parser.feed((repository / PLAN).read_text(encoding="utf-8"))
    parser.close()
    names = [
        "query_case_id", "query_symbol", "match_rank", "matched_episode_id",
        "matched_symbol", "matched_cutoff",
    ]
    result_state = {
        "schema_version": SCHEMA,
        "status": "verified_before_real_future_path_mode_computation",
        "passed": True,
        "contract_digest": contract["contract_digest"],
        "contract_sha256": sha256(raw).hexdigest(),
        "observed": observed,
        "outcome_blind_columns_opened": names,
        "future_path_columns_opened": [],
        "real_future_path_mode_computation_opened": False,
        "synthetic_gate_authorized": True,
        "predictive_claim_authorized": False,
        "production_promotion_authorized": False,
        "trading_claim_authorized": False,
    }
    _require(len(names) == 6, "internal outcome-blind column declaration differs")
    return {**result_state, "result_digest": _stable(result_state)}


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    _require(not path.exists() and not path.is_symlink(), "R2 contract verification root exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".r2-contract-verification-", dir=path.parent))
    try:
        target = temporary / "VERIFIED.json"
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write((json.dumps({
                **value, "created_at": datetime.now(timezone.utc).isoformat(),
            }, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, path)
    except Exception:
        try:
            if (temporary / "VERIFIED.json").exists():
                (temporary / "VERIFIED.json").unlink()
            temporary.rmdir()
        except OSError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = validate(args.repository, args.contract)
    if not args.dry_run:
        _publish(args.repository.resolve(strict=True) / OUTPUT, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
