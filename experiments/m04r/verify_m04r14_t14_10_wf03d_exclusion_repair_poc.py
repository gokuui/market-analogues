"""Independently rerun and verify the bounded WF-03D exclusion-repair POC."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence

import numba
import numpy as np

from market_analogues.adapters import CachedOHLCVSource, source_from_spec
from market_analogues.baseline_feature_store import load_feature_generation
from market_analogues.baseline_neighbors import (
    build_baseline_rank_index, deterministic_random_neighbors,
    indexed_recent_return_volatility_neighbors, recent_return_volatility,
)
from market_analogues.config import load_config
from market_analogues.dtw_component_search import certified_staged_dtw_component_search
from market_analogues.episodes import build_episode
from market_analogues.packed_bound_search import PackedBoundQuery, _eligible_mask
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import InstrumentKey, SearchQuery
from market_analogues.certified_packed_search import certified_packed_search

from experiments.m04r import m04r14_t14_10_wf03_baseline_batch as baseline_batch
from experiments.m04r import m04r14_t14_10_wf03_baseline_poc as baseline_poc
from experiments.m04r import m04r14_t14_10_wf03_baseline_store_full as feature_store
from experiments.m04r import m04r14_t14_10_wf03_combined_batch as price_batch
from experiments.m04r import m04r14_t14_10_wf03_composite_topology_poc as composite_kernel
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_10_wf03b_dtw_component_ladder as dtw_ladder
from experiments.m04r import m04r14_t14_10_wf03d_exclusion_audit as audit
from experiments.m04r import m04r14_t14_10_wf03d_exclusion_repair_poc as producer


SCHEMA = "m04r14-t14-10-wf03d-exclusion-repair-poc-verification-v1"
OUTPUT_RELATIVE = Path(
    "config/data/analogues/m04r14/"
    "t14-10-wf03d-exclusion-repair-poc-v1-verification"
)


class ExclusionRepairPocVerificationError(RuntimeError):
    pass


def _git(repository: Path, *args: str, raw: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=repository, capture_output=True,
        text=not raw, check=False,
    )
    if result.returncode:
        message = result.stderr if not raw else result.stderr.decode(errors="replace")
        raise ExclusionRepairPocVerificationError(
            message.strip() or "git command failed"
        )
    return result.stdout if raw else result.stdout.strip()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _independent_subset(
    rows: Any, query_symbol: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if type(rows) is not list or len(rows) != producer.SUPERSET_K:
        raise ExclusionRepairPocVerificationError("top-21 prefix shape differs")
    try:
        symbols = [row["symbol"] for row in rows]
        identifiers = [row["episode_id"] for row in rows]
    except (KeyError, TypeError) as exc:
        raise ExclusionRepairPocVerificationError("top-21 identity differs") from exc
    if any(type(value) is not str or not value for value in symbols + identifiers) \
            or len(set(symbols)) != producer.SUPERSET_K \
            or len(set(identifiers)) != producer.SUPERSET_K \
            or symbols.count(query_symbol) != 1:
        raise ExclusionRepairPocVerificationError("top-21 diversity differs")
    selected = [dict(row) for row in rows if row["symbol"] != query_symbol]
    if len(selected) != producer.TOP_K:
        raise ExclusionRepairPocVerificationError("top-21 exclusion differs")
    proof = {
        "input_rows": producer.SUPERSET_K,
        "selected_rows": producer.TOP_K,
        "excluded_rows": 1,
        "query_symbol": query_symbol,
        "top_k": producer.TOP_K,
        "proof_kind": "exact_top_k_plus_one_drop_single_excluded_symbol",
    }
    return selected, proof


def _episode(source: Any, row: Mapping[str, Any]) -> Any:
    value = build_episode(
        source, InstrumentKey("nasdaq", str(row["symbol"])), str(row["cutoff"]),
        int(row["lookback"]), str(row["representation_version"]),
    )
    if value.key.id != row["episode_id"]:
        raise ExclusionRepairPocVerificationError("query reconstruction differs")
    return value


def _request(episode: Any) -> SearchQuery:
    return SearchQuery(
        episode.key, ("nasdaq",), ("A", "B"), producer.SUPERSET_K,
        False, True, base.MAX_PER_INSTRUMENT, base.MINIMUM_HISTORY_GAP,
    )


def _old_prefix(
    repository: Path, row: Mapping[str, Any], method: str,
) -> list[dict[str, Any]]:
    query_id = row["episode_id"]
    if method == "composite":
        value = base._read(
            repository / audit.COMPOSITE_ROOT / "cases" / f"{query_id}.json"
        )["retrieval"]["matches"]
    elif method == "price_only":
        value = base._read(
            repository / audit.PRICE_ROOT / "cases" / f"{query_id}.json"
        )["matches"]
    else:
        old = base._read(
            repository / audit.BASELINE_ROOT / "cases" / f"{query_id}.json"
        )
        value = old[
            "random_neighbors" if method == "deterministic_random"
            else "rank_neighbors"
        ]
    if type(value) is not list or len(value) != producer.TOP_K:
        raise ExclusionRepairPocVerificationError("upstream top-20 differs")
    return value


def verify(repository: Path) -> dict[str, Any]:
    started = perf_counter()
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain"):
        raise ExclusionRepairPocVerificationError("repair verifier requires clean commit")
    verifier_commit = str(_git(repository, "rev-parse", "HEAD"))
    verifier_path = Path(__file__).resolve()
    verifier_relative = verifier_path.relative_to(repository).as_posix()
    if sha256(_git(
        repository, "show", f"{verifier_commit}:{verifier_relative}", raw=True,
    )).hexdigest() != _sha(verifier_path):
        raise ExclusionRepairPocVerificationError("verifier Git binding differs")

    preregistration = base._read(repository / producer.PREREGISTRATION_RELATIVE)
    base._validate_seal(preregistration, "preregistration_digest")
    frozen_rows = producer.validate_preregistration(repository, preregistration)
    root = repository / producer.OUTPUT_RELATIVE
    if base._read(root / "CONTRACT.json") != preregistration:
        raise ExclusionRepairPocVerificationError("published contract differs")
    result = base._read(root / "RESULT.json")
    base._validate_seal(result)
    if not all((
        result.get("schema_version")
            == "m04r14-t14-10-wf03d-exclusion-repair-poc-result-v1",
        result.get("status") == "complete", result.get("passed") is True,
        result.get("preregistration_digest")
            == preregistration["preregistration_digest"],
        result.get("query_count") == len(producer.QUERY_IDS),
        result.get("method_runs") == sum(map(len, producer.EXPECTED_AFFECTED.values())),
        result.get("all_original_top20_prefixes_exact") is True,
        result.get("all_corrected_top20_exclude_query_symbol") is True,
        result.get("outcomes_or_labels_used") is False,
        result.get("historical_walk_forward_query_outcomes_opened") is False,
        result.get("final_period_result_opened") is False,
        result.get("production_promotion_authorized") is False,
        result.get("process_swap_kib") == 0,
    )):
        raise ExclusionRepairPocVerificationError("producer result contract differs")
    try:
        published = {row["query_id"]: row for row in result["queries"]}
    except (KeyError, TypeError) as exc:
        raise ExclusionRepairPocVerificationError("producer query layout differs") from exc
    if set(published) != set(producer.QUERY_IDS) \
            or len(published) != len(result["queries"]):
        raise ExclusionRepairPocVerificationError("producer query inventory differs")

    resident = base._resident()
    if resident["content_digest"] != preregistration["inputs"]["resident_content_digest"]:
        raise ExclusionRepairPocVerificationError("resident substrate differs")
    if price_batch._dtw_physical_identity(repository)["digest"] \
            != preregistration["inputs"]["dtw_physical_identity_digest"]:
        raise ExclusionRepairPocVerificationError("DTW substrate differs")
    numba.set_num_threads(producer.THREADS)
    packed_root = Path(resident["store_root"])
    packed = load_packed_generation(
        packed_root, base.GENERATION_ID, verify_content=False,
        validate_records=False, expected_provenance_digest=base.PROVENANCE_DIGEST,
    )
    feature_result = base._read(repository / producer.FEATURE_RESULT)
    features_loaded = load_feature_generation(
        repository / feature_store.OUTPUT_RELATIVE / "store",
        feature_result["generation_id"], packed_manifest=packed.manifest,
        verify_content=True,
    )
    records = np.concatenate((
        baseline_poc._neighbor_records(packed.rows),
        baseline_poc._neighbor_records(packed.overflow),
    ))
    feature_values = np.concatenate((
        features_loaded.rows, features_loaded.overflow,
    ))["values"]
    rank_index = build_baseline_rank_index(feature_values, records["episode_id"])
    source = CachedOHLCVSource(source_from_spec(
        load_config(repository / base.CONFIG_RELATIVE).datasets["nasdaq"]
    ), max_entries=None)
    prepared: dict[str, Any] = {}
    rerun_digests: dict[str, dict[str, str | None]] = {}

    for row in frozen_rows:
        query_id = row["episode_id"]
        case = published[query_id]
        if case.get("symbol") != row["symbol"] or case.get("cutoff") != row["cutoff"]:
            raise ExclusionRepairPocVerificationError("published query binding differs")
        try:
            methods = {value["method"]: value for value in case["methods"]}
        except (KeyError, TypeError) as exc:
            raise ExclusionRepairPocVerificationError("published method layout differs") from exc
        expected_methods = producer.EXPECTED_AFFECTED[query_id]
        if tuple(value["method"] for value in case["methods"]) != expected_methods \
                or set(methods) != set(expected_methods) \
                or len(methods) != len(case["methods"]):
            raise ExclusionRepairPocVerificationError("published method inventory differs")
        for method, value in methods.items():
            selected, proof = _independent_subset(value.get("superset_matches"), row["symbol"])
            if value.get("corrected_matches") != selected \
                    or value.get("subset_proof") != proof \
                    or value["superset_matches"][:producer.TOP_K] \
                        != _old_prefix(repository, row, method):
                raise ExclusionRepairPocVerificationError("published subset proof differs")

        episode = _episode(source, row)
        request = _request(episode)
        rerun_digests[query_id] = {}
        if "composite" in methods:
            search = certified_packed_search(
                episode, source, request, packed_root, base.GENERATION_ID,
                store_dataset_id="nasdaq",
                initial_frontier_rows=producer.INITIAL_FRONTIER,
                maximum_frontier_rows=producer.MAXIMUM_FRONTIER,
                seed_rows=producer.SEED_ROWS, block_rows=producer.BLOCK_ROWS,
                workers=producer.THREADS, tolerance=producer.TOLERANCE,
                verify_content=False, requested_positions=True,
                vector_lower_bounds=True, deferred_alignments=True,
                compact_scored=True, native_bound_deferral=True,
                streaming_threshold_closure=True,
                branch_aware_packed_bounds=True,
                proposal_threads=producer.THREADS,
            )
            rerun = [composite_kernel._match(value) for value in search.matches]
            certificate = methods["composite"].get("certificate", {})
            if rerun != methods["composite"]["superset_matches"] \
                    or certificate.get("result_digest") != search.certificate.result_digest \
                    or certificate.get("input_digest") != search.certificate.input_digest:
                raise ExclusionRepairPocVerificationError("composite exact rerun differs")
            rerun_digests[query_id]["composite"] = search.certificate.result_digest
        if "price_only" in methods:
            search = certified_staged_dtw_component_search(
                episode, source, request, packed_root, base.GENERATION_ID,
                repository / dtw_ladder.DTW_ROOT_RELATIVE,
                dtw_ladder.DTW_GENERATION_ID, store_dataset_id="nasdaq",
                seed_rows=producer.PRICE_SEED_ROWS, block_rows=producer.BLOCK_ROWS,
                rigid_threads=producer.THREADS, dtw_threads=producer.THREADS,
                exact_workers=producer.THREADS, tolerance=producer.TOLERANCE,
                verify_content=False, prepared_symbol_cache=prepared,
                adaptive_seed=True,
            )
            rerun = price_batch._matches(search)
            certificate = methods["price_only"].get("certificate", {})
            if rerun != methods["price_only"]["superset_matches"] \
                    or certificate.get("result_digest") != search.certificate.result_digest \
                    or certificate.get("input_digest") != search.certificate.input_digest:
                raise ExclusionRepairPocVerificationError("price exact rerun differs")
            rerun_digests[query_id]["price_only"] = search.certificate.result_digest

        packed_query = PackedBoundQuery(
            episode.key.id, episode.key.instrument.source_symbol,
            int(episode.bars.timestamp.iloc[0].value),
            int(latest_eligible_cutoff(episode, base.MINIMUM_HISTORY_GAP).value),
            composite_kernel.represent(episode), ("A", "B"),
        )
        symbol_id = packed.symbols.index(row["symbol"]) \
            if row["symbol"] in packed.symbols else None
        eligible = _eligible_mask(records, packed_query, symbol_id)
        if "deterministic_random" in methods:
            rerun = baseline_batch._neighbor_state(deterministic_random_neighbors(
                records["episode_id"], records["symbol_id"], eligible,
                packed.symbols, query_id, top_k=producer.SUPERSET_K,
            ))
            if rerun != methods["deterministic_random"]["superset_matches"]:
                raise ExclusionRepairPocVerificationError("random exact rerun differs")
            rerun_digests[query_id]["deterministic_random"] = None
        if "recent_return_volatility" in methods:
            query_features = recent_return_volatility(
                episode.bars["close"].to_numpy(dtype=np.float64)
            )
            rerun = baseline_batch._neighbor_state(
                indexed_recent_return_volatility_neighbors(
                    rank_index, feature_values, records["episode_id"],
                    records["symbol_id"], eligible, packed.symbols,
                    query_features, query_id, top_k=producer.SUPERSET_K,
                )
            )
            if rerun != methods["recent_return_volatility"]["superset_matches"]:
                raise ExclusionRepairPocVerificationError(
                    "return/volatility exact rerun differs"
                )
            rerun_digests[query_id]["recent_return_volatility"] = None

    if price_batch._dtw_physical_identity(repository)["digest"] \
            != preregistration["inputs"]["dtw_physical_identity_digest"]:
        raise ExclusionRepairPocVerificationError("DTW substrate changed during rerun")
    state = {
        "schema_version": SCHEMA,
        "status": "verified_bounded_exclusion_repair_poc",
        "passed": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "verifier_commit": verifier_commit,
        "verifier_runtime_sha256": _sha(verifier_path),
        "producer_result_digest": result["result_digest"],
        "producer_result_sha256": _sha(root / "RESULT.json"),
        "preregistration_digest": preregistration["preregistration_digest"],
        "query_count": len(frozen_rows),
        "method_runs": sum(len(value) for value in producer.EXPECTED_AFFECTED.values()),
        "rerun_certificate_digests": rerun_digests,
        "elapsed_seconds": perf_counter() - started,
        "gates": {
            "all_source_top20_prefixes_exact": True,
            "all_top21_lists_independently_rerun": True,
            "all_subset_proofs_independently_reconstructed": True,
            "all_corrected_lists_have_20_distinct_non_query_symbols": True,
            "all_verified_substrates_unchanged": True,
            "outcome_blind": True,
        },
        "outcomes_or_labels_used": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
        "full_exclusion_repair_authorized": True,
    }
    return base._sealed(state, "verification_digest")


def publish(repository: Path) -> Path:
    value = verify(repository)
    root = repository.resolve(strict=True) / OUTPUT_RELATIVE
    if root.exists() or root.is_symlink():
        raise ExclusionRepairPocVerificationError("repair verification output exists")
    root.mkdir(parents=True)
    path = root / "VERIFIED.json"
    base._atomic(path, value)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    args = parser.parse_args(argv)
    print(publish(args.repository))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
