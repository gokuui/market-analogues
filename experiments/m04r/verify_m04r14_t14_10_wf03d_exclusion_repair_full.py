"""Independently verify the complete WF-03D symbol-exclusion repair."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import math
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from market_analogues.adapters import CachedOHLCVSource, source_from_spec
from market_analogues.baseline_feature_store import load_feature_generation
from market_analogues.baseline_neighbors import (
    build_baseline_rank_index, deterministic_random_neighbors,
    indexed_recent_return_volatility_neighbors, recent_return_volatility,
)
from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.certified_packed_search import certified_packed_search_contract
from market_analogues.config import load_config
from market_analogues.dtw_component_search import staged_dtw_component_search_contract
from market_analogues.dtw_sample_store import load_dtw_sample_generation
from market_analogues.episodes import build_episode
from market_analogues.m04r_certified_search_verification import (
    _certificate_digest as composite_certificate_digest,
)
from market_analogues.packed_bound_search import PackedBoundQuery, _eligible_mask
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.representation import represent, representation_input_digest
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import EpisodeKey, InstrumentKey, SearchQuery, stable_hash

from experiments.m04r import m04r14_t14_10_wf03_baseline_batch as baseline_batch
from experiments.m04r import m04r14_t14_10_wf03_baseline_poc as baseline_poc
from experiments.m04r import m04r14_t14_10_wf03_baseline_store_full as feature_store
from experiments.m04r import m04r14_t14_10_wf03_combined_batch as price_batch
from experiments.m04r import m04r14_t14_10_wf03_composite_topology_poc as composite_kernel
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_10_wf03b_dtw_component_ladder as dtw_ladder
from experiments.m04r import m04r14_t14_10_wf03d_exclusion_audit as audit
from experiments.m04r import m04r14_t14_10_wf03d_exclusion_repair_full as producer
from experiments.m04r import verify_m04r14_t14_10_wf03_combined_batch as price_verifier


SCHEMA = "m04r14-t14-10-wf03d-exclusion-repair-full-verification-v1"
OUTPUT_RELATIVE = Path(
    "config/data/analogues/m04r14/"
    "t14-10-wf03d-exclusion-repair-full-v1-verification"
)
INVENTORY_DTYPE = np.dtype([
    ("episode_id", "V12"), ("cutoff_ns", "<i8"),
    ("symbol_id", "<u4"), ("quality_tier", "u1"),
])
QUALITY_CODES = {"A": 1, "B": 2}


class ExclusionRepairFullVerificationError(RuntimeError):
    pass


def _git(repository: Path, *args: str, raw: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=repository, capture_output=True,
        text=not raw, check=False,
    )
    if result.returncode:
        message = result.stderr if not raw else result.stderr.decode(errors="replace")
        raise ExclusionRepairFullVerificationError(
            message.strip() or "git command failed"
        )
    return result.stdout if raw else result.stdout.strip()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _inventory(main: np.ndarray, overflow: np.ndarray) -> np.ndarray:
    result = np.empty(len(main) + len(overflow), dtype=INVENTORY_DTYPE)
    first = len(main)
    for name in INVENTORY_DTYPE.names or ():
        result[name][:first] = main[name]
        result[name][first:] = overflow[name]
    return result


def _lookup(
    records: np.ndarray, order: np.ndarray, sorted_ids: np.ndarray,
    identifiers: Sequence[str],
) -> np.ndarray:
    try:
        requested = np.asarray([
            np.void(bytes.fromhex(value)) for value in identifiers
        ], dtype="V12")
    except ValueError as exc:
        raise ExclusionRepairFullVerificationError("invalid episode ID") from exc
    positions = np.searchsorted(sorted_ids, requested)
    if np.any(positions >= len(sorted_ids)) \
            or not np.array_equal(sorted_ids[positions], requested):
        raise ExclusionRepairFullVerificationError("episode absent from inventory")
    return order[positions]


def _request(episode: Any) -> SearchQuery:
    return SearchQuery(
        episode.key, ("nasdaq",), ("A", "B"), producer.SUPERSET_K,
        False, True, base.MAX_PER_INSTRUMENT, base.MINIMUM_HISTORY_GAP,
    )


def _episode(source: Any, row: Mapping[str, Any]) -> Any:
    value = build_episode(
        source, InstrumentKey("nasdaq", str(row["symbol"])), str(row["cutoff"]),
        int(row["lookback"]), str(row["representation_version"]),
    )
    if value.key.id != row["episode_id"]:
        raise ExclusionRepairFullVerificationError("query reconstruction differs")
    return value


def _composite_input_digest(
    source: Any, episode: Any, request: SearchQuery,
    packed_provenance_digest: str,
) -> str:
    benchmark = source.load_benchmark()
    return stable_hash({
        "query_stock_prefix": asdict(causal_prefix_digest(
            source.load(episode.key.instrument), episode.key.cutoff,
        )),
        "query_benchmark_prefix": (
            asdict(causal_prefix_digest(benchmark, episode.key.cutoff))
            if benchmark is not None else None
        ),
        "request": {
            "search_datasets": request.search_datasets,
            "quality_tiers": request.quality_tiers,
            "top_k": request.top_k,
            "cross_dataset": request.cross_dataset,
            "deduplicate_overlaps": request.deduplicate_overlaps,
            "max_per_instrument": request.max_per_instrument,
            "minimum_history_gap_bars": request.minimum_history_gap_bars,
        },
        "packed_provenance_digest": packed_provenance_digest,
        "query_representation_digest": representation_input_digest(
            represent(episode),
        ),
    })


def _price_input_digest(
    source: Any, episode: Any, request: SearchQuery,
    packed_manifest: Mapping[str, Any], dtw_manifest: Mapping[str, Any],
    contract_digest: str,
) -> str:
    return stable_hash({
        "query_stock_prefix": asdict(causal_prefix_digest(
            source.load(episode.key.instrument), episode.key.cutoff,
        )),
        "query_representation_digest": representation_input_digest(
            represent(episode),
        ),
        "request": asdict(request),
        "packed_generation_id": base.GENERATION_ID,
        "packed_provenance_digest": packed_manifest["provenance_digest"],
        "dtw_generation_id": dtw_ladder.DTW_GENERATION_ID,
        "dtw_provenance_digest": dtw_manifest["provenance_digest"],
        "contract_digest": contract_digest,
    })


def _independent_subset(
    matches: Any, query_symbol: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if type(matches) is not list or len(matches) != producer.SUPERSET_K:
        raise ExclusionRepairFullVerificationError("repair top-21 shape differs")
    try:
        symbols = [row["symbol"] for row in matches]
        identifiers = [row["episode_id"] for row in matches]
    except (KeyError, TypeError) as exc:
        raise ExclusionRepairFullVerificationError("repair identity differs") from exc
    if len(set(symbols)) != producer.SUPERSET_K \
            or len(set(identifiers)) != producer.SUPERSET_K \
            or symbols.count(query_symbol) != 1:
        raise ExclusionRepairFullVerificationError("repair diversity differs")
    selected = [dict(row) for row in matches if row["symbol"] != query_symbol]
    proof = {
        "input_rows": producer.SUPERSET_K,
        "selected_rows": producer.TOP_K, "excluded_rows": 1,
        "query_symbol": query_symbol, "top_k": producer.TOP_K,
        "proof_kind": "exact_top_k_plus_one_drop_single_excluded_symbol",
    }
    return selected, proof


def _validate_composite_certificate(
    receipt: Mapping[str, Any], episode: Any, source: Any,
    packed_manifest: Mapping[str, Any],
) -> None:
    try:
        certificate = composite_kernel.decode_certificate_json_value(
            receipt["certificate"]
        )
        matches = receipt["superset_matches"]
        accounting = certificate["native_bound_accounting"]
        distances = [float(match["total_distance"]) for match in matches]
        contract = certified_packed_search_contract(
            requested_positions=True, vector_lower_bounds=True,
            deferred_alignments=True, compact_scored=True,
            native_bound_deferral=True, streaming_threshold_closure=True,
            branch_aware_packed_bounds=True,
        )
        valid = all((
            certificate["contract_digest"] == contract["digest"],
            certificate["generation_id"] == base.GENERATION_ID,
            certificate["query_episode_id"] == episode.key.id,
            certificate["input_digest"] == _composite_input_digest(
                source, episode, _request(episode),
                packed_manifest["provenance_digest"],
            ),
            certificate["result_digest"] == composite_certificate_digest({
                "certificate": certificate, "matches": matches,
            }),
            certificate["eligible_candidates"]
                == accounting["exact_dtw_evaluated"]
                + accounting["native_bound_pruned"]
                + accounting["packed_bound_pruned"],
            certificate["exact_evaluated"] == accounting["exact_dtw_evaluated"],
            certificate["safely_pruned"]
                == accounting["native_bound_pruned"] + accounting["packed_bound_pruned"],
            certificate["maximum_quantized_bound_excess"] <= producer.TOLERANCE,
            distances == sorted(distances),
            math.isclose(
                distances[-1], certificate["stop_threshold"],
                rel_tol=0.0, abs_tol=producer.TOLERANCE,
            ),
            certificate["next_lower_bound"] is None
                or certificate["next_lower_bound"] + producer.TOLERANCE
                    >= certificate["stop_threshold"],
        ))
    except (KeyError, TypeError, ValueError) as exc:
        raise ExclusionRepairFullVerificationError(
            "composite certificate structure differs"
        ) from exc
    if not valid:
        raise ExclusionRepairFullVerificationError("composite certificate differs")


def _validate_price_certificate(
    receipt: Mapping[str, Any], episode: Any, source: Any,
    packed_manifest: Mapping[str, Any], dtw_manifest: Mapping[str, Any],
) -> None:
    try:
        certificate = receipt["certificate"]
        matches = receipt["superset_matches"]
        distances = [float.fromhex(match["distance_hex"]) for match in matches]
        contract = staged_dtw_component_search_contract(adaptive_seed=True)
        rigid_minimum = certificate["minimum_rigid_pruned"]
        combined_minimum = certificate["minimum_combined_pruned"]
        valid = all((
            certificate["contract_digest"] == contract["digest"],
            certificate["packed_generation_id"] == base.GENERATION_ID,
            certificate["dtw_generation_id"] == dtw_ladder.DTW_GENERATION_ID,
            certificate["query_episode_id"] == episode.key.id,
            certificate["input_digest"] == _price_input_digest(
                source, episode, _request(episode), packed_manifest,
                dtw_manifest, contract["digest"],
            ),
            certificate["result_digest"]
                == price_verifier._certificate_digest(certificate, matches),
            certificate["eligible_candidates"] == certificate["rigid_bound_evaluated"],
            certificate["rigid_bound_admitted"] <= certificate["rigid_bound_evaluated"],
            certificate["dtw_bound_evaluated"] <= certificate["rigid_bound_admitted"],
            certificate["combined_bound_admitted"] <= certificate["rigid_bound_admitted"],
            certificate["exact_evaluated"] >= certificate["seed_rows"],
            certificate["maximum_bound_excess"] <= producer.TOLERANCE,
            rigid_minimum is None or rigid_minimum > certificate["seed_threshold"],
            combined_minimum is None or combined_minimum > certificate["seed_threshold"],
            distances == sorted(distances),
            distances[-1] == certificate["final_threshold"],
        ))
    except (KeyError, TypeError, ValueError) as exc:
        raise ExclusionRepairFullVerificationError(
            "price certificate structure differs"
        ) from exc
    if not valid:
        raise ExclusionRepairFullVerificationError("price certificate differs")


def _source_matches(
    cases: Mapping[str, Mapping[str, Any]], method: str,
) -> list[dict[str, Any]]:
    if method == "composite":
        value = cases["composite"]["retrieval"]["matches"]
    elif method == "price_only":
        value = cases["price_only"]["matches"]
    else:
        value = cases["baselines"][
            "random_neighbors" if method == "deterministic_random"
            else "rank_neighbors"
        ]
    if type(value) is not list or len(value) != producer.TOP_K:
        raise ExclusionRepairFullVerificationError("source top-20 differs")
    return value


def verify(repository: Path) -> dict[str, Any]:
    started = perf_counter()
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain"):
        raise ExclusionRepairFullVerificationError("full repair verifier requires clean commit")
    verifier_commit = str(_git(repository, "rev-parse", "HEAD"))
    verifier_path = Path(__file__).resolve()
    relative = verifier_path.relative_to(repository).as_posix()
    if sha256(_git(repository, "show", f"{verifier_commit}:{relative}", raw=True)).hexdigest() \
            != _sha(verifier_path):
        raise ExclusionRepairFullVerificationError("verifier Git binding differs")

    preregistration = base._read(repository / producer.PREREGISTRATION_RELATIVE)
    base._validate_seal(preregistration, "preregistration_digest")
    rows, affected = producer.validate_preregistration(repository, preregistration)
    root = repository / producer.OUTPUT_RELATIVE
    if base._read(root / "CONTRACT.json") != preregistration:
        raise ExclusionRepairFullVerificationError("full repair contract differs")
    result = base._read(root / "RESULT.json"); base._validate_seal(result)
    manifest = base._read(root / "MANIFEST.json")
    base._validate_seal(manifest, "manifest_digest")
    if not all((
        result.get("schema_version")
            == "m04r14-t14-10-wf03d-exclusion-repair-full-result-v1",
        result.get("passed") is True, result.get("status") == "complete",
        result.get("preregistration_digest")
            == preregistration["preregistration_digest"],
        result.get("manifest_digest") == manifest["manifest_digest"],
        result.get("manifest_sha256") == _sha(root / "MANIFEST.json"),
        result.get("query_count") == producer.EXPECTED_QUERIES,
        result.get("affected_query_union") == producer.EXPECTED_AFFECTED_UNION,
        result.get("affected_method_repairs") == sum(map(len, affected.values())),
        result.get("outcomes_or_labels_used") is False,
        result.get("historical_walk_forward_query_outcomes_opened") is False,
        result.get("final_period_result_opened") is False,
        result.get("production_promotion_authorized") is False,
        manifest.get("schema_version")
            == "m04r14-t14-10-wf03d-exclusion-repair-manifest-v1",
        manifest.get("status") == "complete",
        manifest.get("preregistration_digest")
            == preregistration["preregistration_digest"],
        manifest.get("query_count") == producer.EXPECTED_QUERIES,
        manifest.get("method_links")
            == producer.EXPECTED_QUERIES * len(producer.METHODS),
        manifest.get("effective_neighbour_links")
            == producer.EXPECTED_QUERIES * len(producer.METHODS) * producer.TOP_K,
        manifest.get("affected_method_repairs") == sum(map(len, affected.values())),
        manifest.get("affected_query_union") == producer.EXPECTED_AFFECTED_UNION,
        manifest.get("outcomes_or_labels_used") is False,
        manifest.get("historical_walk_forward_query_outcomes_opened") is False,
        manifest.get("final_period_result_opened") is False,
    )):
        raise ExclusionRepairFullVerificationError("terminal publication differs")
    if type(manifest.get("queries")) is not list \
            or len(manifest["queries"]) != len(rows):
        raise ExclusionRepairFullVerificationError("manifest query inventory differs")

    resident = base._resident()
    packed = load_packed_generation(
        Path(resident["store_root"]), base.GENERATION_ID,
        expected_provenance_digest=base.PROVENANCE_DIGEST,
        verify_content=False, validate_records=False,
    )
    dtw = load_dtw_sample_generation(
        repository / dtw_ladder.DTW_ROOT_RELATIVE,
        dtw_ladder.DTW_GENERATION_ID, packed_manifest=packed.manifest,
        verify_content=False, validate_records=False,
    )
    records = _inventory(packed.rows, packed.overflow)
    order = np.argsort(records["episode_id"], kind="stable")
    sorted_ids = records["episode_id"][order]
    if len(np.unique(sorted_ids)) != len(sorted_ids):
        raise ExclusionRepairFullVerificationError("packed episode IDs are not unique")
    source = CachedOHLCVSource(source_from_spec(
        load_config(repository / base.CONFIG_RELATIVE).datasets["nasdaq"]
    ), max_entries=None)
    affected_sets = {method: set(values) for method, values in affected.items()}
    repaired_baselines: dict[tuple[str, str], list[dict[str, Any]]] = {}
    effective_states = []
    composite_certificates = 0
    price_certificates = 0

    for row, entry in zip(rows, manifest["queries"], strict=True):
        query_id = row["episode_id"]
        if any(entry.get(key) != row[key] for key in (
            "query_id", "case_id", "symbol", "cutoff",
        )) or type(entry.get("methods")) is not list \
                or [value.get("method") for value in entry["methods"]] \
                    != list(producer.METHODS):
            raise ExclusionRepairFullVerificationError("manifest query binding differs")
        paths = producer._case_paths(repository, query_id)
        cases = {name: base._read(path) for name, path in paths.items()}
        for case in cases.values():
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
                raise ExclusionRepairFullVerificationError("source query binding differs")
        source_sha = {name: _sha(path) for name, path in paths.items()}
        episode = _episode(source, row)
        latest_ns = int(latest_eligible_cutoff(
            episode, base.MINIMUM_HISTORY_GAP,
        ).value)
        for method_entry in entry["methods"]:
            method = method_entry["method"]
            source_matches = _source_matches(cases, method)
            if method_entry.get("source_case_sha256") != source_sha:
                raise ExclusionRepairFullVerificationError("source case identity differs")
            if query_id in affected_sets[method]:
                receipt_path = root / str(method_entry.get("repair_receipt", ""))
                if receipt_path != producer._receipt_path(root, method, query_id) \
                        or _sha(receipt_path) != method_entry.get("repair_receipt_sha256"):
                    raise ExclusionRepairFullVerificationError("repair path identity differs")
                receipt = base._read(receipt_path)
                base._validate_seal(receipt, "receipt_digest")
                selected, proof = _independent_subset(
                    receipt.get("superset_matches"), row["symbol"],
                )
                if not all((
                    receipt.get("schema_version")
                        == "m04r14-t14-10-wf03d-exclusion-repair-receipt-v1",
                    receipt.get("status") == "complete",
                    receipt.get("receipt_digest")
                        == method_entry.get("repair_receipt_digest"),
                    receipt.get("query_id") == query_id,
                    receipt.get("case_id") == row["case_id"],
                    receipt.get("symbol") == row["symbol"],
                    receipt.get("cutoff") == row["cutoff"],
                    receipt.get("method") == method,
                    receipt.get("preregistration_digest")
                        == preregistration["preregistration_digest"],
                    receipt.get("source_case_sha256") == source_sha,
                    receipt["superset_matches"][:producer.TOP_K] == source_matches,
                    receipt.get("corrected_matches") == selected,
                    receipt.get("subset_proof") == proof,
                    method_entry.get("kind") == "top21_drop_query_symbol",
                    method_entry.get("subset_proof") == proof,
                    receipt.get("outcomes_or_labels_used") is False,
                    receipt.get("historical_walk_forward_query_outcomes_opened") is False,
                    receipt.get("final_period_result_opened") is False,
                    receipt.get("production_promotion_authorized") is False,
                )):
                    raise ExclusionRepairFullVerificationError("repair receipt differs")
                matches = selected
                if method == "composite":
                    _validate_composite_certificate(
                        receipt, episode, source, packed.manifest,
                    )
                    composite_certificates += 1
                elif method == "price_only":
                    _validate_price_certificate(
                        receipt, episode, source, packed.manifest, dtw.manifest,
                    )
                    price_certificates += 1
                else:
                    if receipt.get("certificate") is not None:
                        raise ExclusionRepairFullVerificationError(
                            "baseline repair unexpectedly has certificate"
                        )
                    repaired_baselines[(query_id, method)] = receipt["superset_matches"]
            else:
                if method_entry.get("kind") != "upstream_top20_unchanged_over_subset" \
                        or any(match["symbol"] == row["symbol"] for match in source_matches) \
                        or method_entry.get("subset_proof") != {
                            "input_rows": producer.TOP_K,
                            "selected_rows": producer.TOP_K, "excluded_rows": 0,
                            "query_symbol": row["symbol"], "top_k": producer.TOP_K,
                            "proof_kind": "exact_top_k_unchanged_over_subset",
                        }:
                    raise ExclusionRepairFullVerificationError(
                        "unchanged-over-subset proof differs"
                    )
                matches = source_matches
            identifiers = [match["episode_id"] for match in matches]
            symbols = [match["symbol"] for match in matches]
            if len(matches) != producer.TOP_K \
                    or len(set(identifiers)) != producer.TOP_K \
                    or len(set(symbols)) != producer.TOP_K \
                    or row["symbol"] in symbols \
                    or method_entry.get("effective_matches_digest") != stable_hash(matches):
                raise ExclusionRepairFullVerificationError("effective matches differ")
            positions = _lookup(records, order, sorted_ids, identifiers)
            for match, position in zip(matches, positions, strict=True):
                symbol = packed.symbols[int(records["symbol_id"][position])]
                cutoff_ns = int(records["cutoff_ns"][position])
                cutoff = pd.Timestamp(cutoff_ns)
                quality = int(records["quality_tier"][position])
                expected_key = EpisodeKey(
                    InstrumentKey("nasdaq", symbol), cutoff,
                    int(row["lookback"]), str(row["representation_version"]),
                )
                if not all((
                    expected_key.id == match["episode_id"], symbol == match["symbol"],
                    cutoff_ns <= latest_ns,
                    cutoff_ns < int(pd.Timestamp(row["cutoff"]).value),
                    quality in QUALITY_CODES.values(),
                    match["episode_id"] != query_id,
                )):
                    raise ExclusionRepairFullVerificationError(
                        "effective analogue eligibility differs"
                    )
                if method in {"composite", "price_only"} and not all((
                    match.get("cutoff") == cutoff.isoformat(),
                    QUALITY_CODES.get(match.get("quality_tier")) == quality,
                )):
                    raise ExclusionRepairFullVerificationError(
                        "effective analogue metadata differs"
                    )
            effective_states.append({
                "query_id": query_id, "method": method,
                "matches_digest": stable_hash(matches),
            })

    feature_result = base._read(repository / producer.poc.FEATURE_RESULT)
    loaded = load_feature_generation(
        repository / feature_store.OUTPUT_RELATIVE / "store",
        feature_result["generation_id"], packed_manifest=packed.manifest,
        verify_content=True,
    )
    baseline_records = np.concatenate((
        baseline_poc._neighbor_records(packed.rows),
        baseline_poc._neighbor_records(packed.overflow),
    ))
    feature_values = np.concatenate((loaded.rows, loaded.overflow))["values"]
    rank_index = build_baseline_rank_index(feature_values, baseline_records["episode_id"])
    rows_by_id = {row["episode_id"]: row for row in rows}
    for (query_id, method), published in repaired_baselines.items():
        row = rows_by_id[query_id]
        episode = _episode(source, row)
        query = PackedBoundQuery(
            episode.key.id, episode.key.instrument.source_symbol,
            int(episode.bars.timestamp.iloc[0].value),
            int(latest_eligible_cutoff(episode, base.MINIMUM_HISTORY_GAP).value),
            represent(episode), ("A", "B"),
        )
        symbol_id = packed.symbols.index(row["symbol"]) \
            if row["symbol"] in packed.symbols else None
        eligible = _eligible_mask(baseline_records, query, symbol_id)
        if method == "deterministic_random":
            rerun = baseline_batch._neighbor_state(deterministic_random_neighbors(
                baseline_records["episode_id"], baseline_records["symbol_id"],
                eligible, packed.symbols, query_id, top_k=producer.SUPERSET_K,
            ))
        else:
            query_features = recent_return_volatility(
                episode.bars["close"].to_numpy(dtype=np.float64)
            )
            rerun = baseline_batch._neighbor_state(
                indexed_recent_return_volatility_neighbors(
                    rank_index, feature_values, baseline_records["episode_id"],
                    baseline_records["symbol_id"], eligible, packed.symbols,
                    query_features, query_id, top_k=producer.SUPERSET_K,
                )
            )
        if rerun != published:
            raise ExclusionRepairFullVerificationError("baseline exact rerun differs")

    if manifest.get("effective_inventory_digest") != stable_hash(effective_states) \
            or composite_certificates != producer.EXPECTED_AFFECTED["composite"] \
            or price_certificates != producer.EXPECTED_AFFECTED["price_only"] \
            or len(repaired_baselines) \
                != producer.EXPECTED_AFFECTED["deterministic_random"] \
                    + producer.EXPECTED_AFFECTED["recent_return_volatility"]:
        raise ExclusionRepairFullVerificationError("verified inventory totals differ")
    state = {
        "schema_version": SCHEMA,
        "status": "verified_full_symbol_exclusion_repair",
        "passed": True, "created_at": datetime.now(timezone.utc).isoformat(),
        "verifier_commit": verifier_commit,
        "verifier_runtime_sha256": _sha(verifier_path),
        "producer_result_digest": result["result_digest"],
        "producer_result_sha256": _sha(root / "RESULT.json"),
        "manifest_digest": manifest["manifest_digest"],
        "effective_inventory_digest": manifest["effective_inventory_digest"],
        "query_count": len(rows),
        "method_lanes": len(rows) * len(producer.METHODS),
        "effective_neighbour_links": len(effective_states) * producer.TOP_K,
        "repair_receipts": sum(map(len, affected.values())),
        "composite_certificates": composite_certificates,
        "price_certificates": price_certificates,
        "baseline_exact_reruns": len(repaired_baselines),
        "elapsed_seconds": perf_counter() - started,
        "gates": {
            "all_source_cases_and_identities_valid": True,
            "all_15744_method_lanes_resolved": True,
            "all_314880_effective_links_eligible": True,
            "all_212_repair_receipts_valid": True,
            "all_172_search_certificates_reconstructed": True,
            "all_40_baseline_repairs_exactly_rerun": True,
            "all_query_symbols_excluded": True,
            "outcome_blind": True,
        },
        "outcomes_or_labels_used": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
        "cross_store_manifest_construction_authorized": True,
    }
    return base._sealed(state, "verification_digest")


def publish(repository: Path) -> Path:
    value = verify(repository)
    root = repository.resolve(strict=True) / OUTPUT_RELATIVE
    if root.exists() or root.is_symlink():
        raise ExclusionRepairFullVerificationError("full repair verification output exists")
    root.mkdir(parents=True)
    path = root / "VERIFIED.json"; base._atomic(path, value)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    args = parser.parse_args(argv)
    print(publish(args.repository)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
