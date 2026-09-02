"""Independently verify all combined-retrieval receipts and sampled reruns."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from market_analogues.adapters import CachedOHLCVSource, source_from_spec
from market_analogues.config import load_config
from market_analogues.dtw_component_search import certified_staged_dtw_component_search
from market_analogues.dtw_sample_store import load_dtw_sample_generation
from market_analogues.episodes import build_episode
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import EpisodeKey, InstrumentKey, SearchQuery, stable_hash

from experiments.m04r import m04r14_t14_10_wf03_combined_batch as producer
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_10_wf03b_dtw_component_ladder as ladder


OUTPUT_RELATIVE = Path(
    "config/data/analogues/m04r14/"
    "t14-10-wf03-combined-batch-v1-verification"
)
AUTHORITY_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf03b-dtw-component-ladder-v1/"
    "cases/000-early-a69def453340e01048a52284/attempts/"
    "attempt-0001/EXACT.json"
)
AUTHORITY_QUERY_ID = "a69def453340e01048a52284"
RERUNS_PER_FOLD = 1
QUALITY_CODES = {"A": 1, "B": 2}
INVENTORY_DTYPE = np.dtype([
    ("episode_id", "V12"), ("cutoff_ns", "<i8"),
    ("symbol_id", "<u4"), ("quality_tier", "u1"),
])


class CombinedBatchVerificationError(RuntimeError):
    pass


def _seal_valid(value: Mapping[str, Any], field: str) -> bool:
    return type(value) is dict and value.get(field) == stable_hash({
        key: item for key, item in value.items() if key != field
    })


def _certificate_digest(
    certificate: Mapping[str, Any], matches: Sequence[Mapping[str, Any]],
) -> str:
    state = {
        "schema_version": certificate["schema_version"],
        "contract_digest": certificate["contract_digest"],
        "packed_generation_id": certificate["packed_generation_id"],
        "dtw_generation_id": certificate["dtw_generation_id"],
        "query_episode_id": certificate["query_episode_id"],
        "input_digest": certificate["input_digest"],
        "eligible_candidates": certificate["eligible_candidates"],
        "seed_rows": certificate["seed_rows"],
        "rigid_bound_evaluated": certificate["rigid_bound_evaluated"],
        "rigid_bound_admitted": certificate["rigid_bound_admitted"],
        "dtw_bound_evaluated": certificate["dtw_bound_evaluated"],
        "combined_bound_admitted": certificate["combined_bound_admitted"],
        "exact_evaluated": certificate["exact_evaluated"],
        "native_bound_pruned": certificate["native_bound_pruned"],
        "seed_threshold_hex": float(certificate["seed_threshold"]).hex(),
        "final_threshold_hex": float(certificate["final_threshold"]).hex(),
        "minimum_rigid_pruned_hex": (
            float(certificate["minimum_rigid_pruned"]).hex()
            if certificate["minimum_rigid_pruned"] is not None else None
        ),
        "minimum_combined_pruned_hex": (
            float(certificate["minimum_combined_pruned"]).hex()
            if certificate["minimum_combined_pruned"] is not None else None
        ),
        "maximum_bound_excess_hex": float(
            certificate["maximum_bound_excess"]
        ).hex(),
        "matches": [{
            "episode_id": row["episode_id"],
            "distance_hex": row["distance_hex"],
        } for row in matches],
        "outcomes_or_labels_used": False,
    }
    return stable_hash(state)


def select_rerun_sample(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["fold_id"]), []).append(row)
    selected = []
    for fold in sorted(grouped):
        ordered = sorted(grouped[fold], key=lambda row: (
            stable_hash({
                "purpose": "wf03-combined-independent-rerun-v1",
                "fold": fold, "query_id": row["episode_id"],
            }),
            row["episode_id"],
        ))
        if len(ordered) < RERUNS_PER_FOLD:
            raise CombinedBatchVerificationError("combined rerun fold is undersized")
        selected.extend(
            str(row["episode_id"]) for row in ordered[:RERUNS_PER_FOLD]
        )
    return selected


def _lookup_positions(
    sorted_ids: np.ndarray, order: np.ndarray, identifiers: Sequence[str],
) -> np.ndarray:
    try:
        requested = np.asarray([
            np.void(bytes.fromhex(value)) for value in identifiers
        ], dtype="V12")
    except ValueError as exc:
        raise CombinedBatchVerificationError("invalid analogue episode ID") from exc
    positions = np.searchsorted(sorted_ids, requested)
    if np.any(positions >= len(sorted_ids)) \
            or not np.array_equal(sorted_ids[positions], requested):
        raise CombinedBatchVerificationError("analogue episode is absent from store")
    return order[positions]


def _inventory(main: np.ndarray, overflow: np.ndarray) -> np.ndarray:
    result = np.empty(len(main) + len(overflow), dtype=INVENTORY_DTYPE)
    first = len(main)
    for name in INVENTORY_DTYPE.names or ():
        result[name][:first] = main[name]
        result[name][first:] = overflow[name]
    return result


def _independent_case(
    value: Mapping[str, Any], row: Mapping[str, Any],
    preregistration: Mapping[str, Any], records: np.ndarray,
    symbols: tuple[str, ...], id_order: np.ndarray, sorted_ids: np.ndarray,
    source: Any,
) -> dict[str, Any]:
    if not _seal_valid(value, "case_digest"):
        raise CombinedBatchVerificationError("combined case seal differs")
    try:
        certificate = value["certificate"]
        matches = value["matches"]
        distances = [float.fromhex(match["distance_hex"]) for match in matches]
        semantic = {
            "query_id": value["query_id"], "case_id": value["case_id"],
            "packed_generation_id": value["packed_generation_id"],
            "dtw_generation_id": value["dtw_generation_id"],
            "certificate_result_digest": certificate["result_digest"],
            "matches": matches,
        }
        rigid_minimum = certificate["minimum_rigid_pruned"]
        combined_minimum = certificate["minimum_combined_pruned"]
        predicates = (
            value["schema_version"] == "m04r14-wf03-combined-batch-case-v1",
            value["status"] == "complete",
            value["query_id"] == row["episode_id"],
            value["case_id"] == row["case_id"],
            value["symbol"] == row["symbol"], value["cutoff"] == row["cutoff"],
            value["fold_id"] == row["fold_id"],
            value["fold_role"] == row["fold_role"],
            value["scored"] is bool(row["scored"]),
            value["preregistration_digest"]
                == preregistration["preregistration_digest"],
            value["packed_generation_id"] == base.GENERATION_ID,
            value["dtw_generation_id"] == ladder.DTW_GENERATION_ID,
            value["contract_digest"] == preregistration["contract"]["digest"],
            value["outcomes_or_labels_used"] is False,
            value["historical_walk_forward_query_outcomes_opened"] is False,
            value["final_period_result_opened"] is False,
            value["process_swap_kib"] == 0,
            certificate["schema_version"]
                == preregistration["contract"]["schema_version"],
            certificate["contract_digest"] == preregistration["contract"]["digest"],
            certificate["packed_generation_id"] == base.GENERATION_ID,
            certificate["dtw_generation_id"] == ladder.DTW_GENERATION_ID,
            certificate["query_episode_id"] == row["episode_id"],
            certificate["eligible_candidates"]
                == certificate["rigid_bound_evaluated"],
            certificate["rigid_bound_admitted"]
                <= certificate["rigid_bound_evaluated"],
            certificate["dtw_bound_evaluated"]
                <= certificate["rigid_bound_admitted"],
            certificate["combined_bound_admitted"]
                <= certificate["rigid_bound_admitted"],
            certificate["exact_evaluated"] >= certificate["seed_rows"],
            certificate["maximum_bound_excess"] <= producer.TOLERANCE,
            certificate["final_threshold"] <= certificate["seed_threshold"]
                + producer.TOLERANCE,
            rigid_minimum is None or rigid_minimum > certificate["seed_threshold"],
            combined_minimum is None
                or combined_minimum > certificate["seed_threshold"],
            type(matches) is list and len(matches) == producer.TOP_K,
            len({match["symbol"] for match in matches}) == producer.TOP_K,
            len({match["episode_id"] for match in matches}) == producer.TOP_K,
            np.isfinite(distances).all(), distances == sorted(distances),
            distances[-1] == certificate["final_threshold"],
            certificate["result_digest"] == _certificate_digest(
                certificate, matches
            ),
            value["semantic_digest"] == stable_hash(semantic),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CombinedBatchVerificationError("combined case structure differs") from exc
    if not all(predicates):
        raise CombinedBatchVerificationError("combined case contract differs")

    episode = build_episode(
        source, InstrumentKey("nasdaq", str(row["symbol"])), str(row["cutoff"]),
        int(row["lookback"]), str(row["representation_version"]),
    )
    if episode.key.id != row["episode_id"]:
        raise CombinedBatchVerificationError("combined query reconstruction differs")
    latest_ns = int(latest_eligible_cutoff(
        episode, base.MINIMUM_HISTORY_GAP
    ).value)
    query_start_ns = int(episode.bars.timestamp.iloc[0].value)
    positions = _lookup_positions(
        sorted_ids, id_order, [match["episode_id"] for match in matches],
    )
    same_symbol = 0
    for match, position in zip(matches, positions, strict=True):
        symbol = symbols[int(records["symbol_id"][position])]
        cutoff_ns = int(records["cutoff_ns"][position])
        cutoff = pd.Timestamp(cutoff_ns)
        quality = int(records["quality_tier"][position])
        expected_key = EpisodeKey(
            InstrumentKey("nasdaq", symbol), cutoff,
            int(row["lookback"]), str(row["representation_version"]),
        )
        if not all((
            symbol == match["symbol"], cutoff.isoformat() == match["cutoff"],
            quality == QUALITY_CODES.get(match["quality_tier"]),
            expected_key.id == match["episode_id"],
            cutoff_ns <= latest_ns,
            cutoff_ns < int(pd.Timestamp(row["cutoff"]).value),
            match["episode_id"] != row["episode_id"],
        )):
            raise CombinedBatchVerificationError("combined analogue binding differs")
        if symbol == row["symbol"]:
            same_symbol += 1
            if cutoff_ns >= query_start_ns:
                raise CombinedBatchVerificationError("same-symbol analogue overlaps query")
    return {
        "query_id": row["episode_id"], "case_digest": value["case_digest"],
        "certificate_result_digest": certificate["result_digest"],
        "neighbor_position_digest": stable_hash([int(value) for value in positions]),
        "same_symbol_neighbors": same_symbol,
        "eligible_candidates": certificate["eligible_candidates"],
    }


def _rerun_matches(result: Any) -> list[dict[str, Any]]:
    return [{
        "episode_id": match.episode_key.id,
        "symbol": match.episode_key.instrument.source_symbol,
        "cutoff": match.episode_key.cutoff.isoformat(),
        "distance_hex": match.total_distance.hex(),
        "quality_tier": match.quality_tier,
    } for match in result.matches]


def _rerun_sample(
    repository: Path, rows: Sequence[Mapping[str, Any]], cases_root: Path,
    packed_root: Path,
) -> list[dict[str, Any]]:
    raw = source_from_spec(
        load_config(repository / base.CONFIG_RELATIVE).datasets["nasdaq"]
    )
    source = CachedOHLCVSource(raw, max_entries=None)
    prepared: dict[str, Any] = {}
    output = []
    for row in rows:
        published = base._read(producer._case_path(cases_root, row["episode_id"]))
        episode = build_episode(
            source, InstrumentKey("nasdaq", row["symbol"]), row["cutoff"],
            int(row["lookback"]), row["representation_version"],
        )
        request = SearchQuery(
            episode.key, ("nasdaq",), ("A", "B"), producer.TOP_K,
            False, True, base.MAX_PER_INSTRUMENT, base.MINIMUM_HISTORY_GAP,
        )
        result = certified_staged_dtw_component_search(
            episode, source, request, packed_root, base.GENERATION_ID,
            repository / ladder.DTW_ROOT_RELATIVE, ladder.DTW_GENERATION_ID,
            store_dataset_id="nasdaq", seed_rows=producer.SEED_ROWS,
            block_rows=producer.BLOCK_ROWS, rigid_threads=producer.THREADS,
            dtw_threads=producer.THREADS, exact_workers=producer.THREADS,
            tolerance=producer.TOLERANCE, verify_content=False,
            prepared_symbol_cache=prepared,
        )
        actual = _rerun_matches(result)
        if actual != published["matches"] \
                or result.certificate.result_digest \
                != published["certificate"]["result_digest"]:
            raise CombinedBatchVerificationError("sampled combined rerun differs")
        output.append({
            "query_id": row["episode_id"],
            "certificate_result_digest": result.certificate.result_digest,
            "matches_digest": stable_hash(actual),
        })
    return output


def verify(repository: Path) -> dict[str, Any]:
    started = perf_counter()
    repository = repository.resolve(strict=True)
    if producer._git(repository, "status", "--porcelain"):
        raise CombinedBatchVerificationError("combined verifier requires clean commit")
    verifier_commit = producer._git(repository, "rev-parse", "HEAD")
    verifier_relative = str(Path(__file__).resolve().relative_to(repository))
    verifier_sha256 = base._sha(repository / verifier_relative)
    blob = subprocess.run(
        ["git", "show", f"{verifier_commit}:{verifier_relative}"],
        cwd=repository, capture_output=True, check=False,
    )
    if blob.returncode or sha256(blob.stdout).hexdigest() != verifier_sha256:
        raise CombinedBatchVerificationError("combined verifier Git binding differs")
    preregistration = base._read(repository / producer.PREREGISTRATION_RELATIVE)
    registry, by_id = producer.validate_preregistration(
        repository, preregistration
    )
    root = repository / producer.OUTPUT_RELATIVE
    if base._read(root / "CONTRACT.json") != preregistration:
        raise CombinedBatchVerificationError("combined producer contract differs")
    result = base._read(root / "RESULT.json")
    if not _seal_valid(result, "result_digest") or not all((
        result.get("status") == "complete", result.get("passed") is True,
        result.get("queries") == 3_936,
        result.get("scored_queries") == 3_360,
        result.get("warmup_queries") == 576,
        result.get("outcomes_or_labels_used") is False,
        result.get("historical_walk_forward_query_outcomes_opened") is False,
        result.get("final_period_result_opened") is False,
        result.get("independent_verification_authorized") is True,
    )):
        raise CombinedBatchVerificationError("combined producer terminal differs")

    resident = base._resident()
    packed = load_packed_generation(
        Path(resident["store_root"]), base.GENERATION_ID,
        expected_provenance_digest=base.PROVENANCE_DIGEST,
        verify_content=True, validate_records=True,
    )
    load_dtw_sample_generation(
        repository / ladder.DTW_ROOT_RELATIVE, ladder.DTW_GENERATION_ID,
        packed_manifest=packed.manifest, verify_content=True,
        validate_records=True,
    )
    records = _inventory(packed.rows, packed.overflow)
    id_order = np.argsort(records["episode_id"], kind="stable")
    sorted_ids = records["episode_id"][id_order]
    source = source_from_spec(
        load_config(repository / base.CONFIG_RELATIVE).datasets["nasdaq"]
    )
    cases_root = root / "cases"
    observations = []
    case_manifest = []
    semantic_digests = []
    for row in registry["queries_data"]:
        path = producer._case_path(cases_root, row["episode_id"])
        value = base._read(path)
        observations.append(_independent_case(
            value, row, preregistration, records, packed.symbols,
            id_order, sorted_ids, source,
        ))
        case_manifest.append({
            "query_id": value["query_id"], "case_digest": value["case_digest"],
            "sha256": base._sha(path),
        })
        semantic_digests.append(value["semantic_digest"])
    if not all((
        result["case_manifest_digest"] == stable_hash(case_manifest),
        result["case_semantic_digest"] == stable_hash(semantic_digests),
        result["minimum_eligible_candidates"] == min(
            row["eligible_candidates"] for row in observations
        ),
        result["maximum_eligible_candidates"] == max(
            row["eligible_candidates"] for row in observations
        ),
    )):
        raise CombinedBatchVerificationError("combined aggregate differs")

    authority = base._read(repository / AUTHORITY_RELATIVE)
    published_authority = base._read(
        producer._case_path(cases_root, AUTHORITY_QUERY_ID)
    )
    expected_authority = [{
        key: match[key]
        for key in ("episode_id", "symbol", "cutoff", "distance_hex", "quality_tier")
    } for match in authority["matches"]]
    if published_authority["matches"] != expected_authority:
        raise CombinedBatchVerificationError("frozen scalar authority differs")

    sample_ids = select_rerun_sample(registry["queries_data"])
    reruns = _rerun_sample(
        repository, [by_id[value] for value in sample_ids], cases_root,
        Path(resident["store_root"]),
    )
    gates = {
        "producer_and_preregistration_valid": True,
        "packed_and_dtw_full_content_valid": True,
        "all_3936_case_seals_and_certificates_valid": True,
        "all_78720_analogue_links_resolve": True,
        "all_analogue_links_causal_and_nonoverlapping": True,
        "all_certificate_result_digests_independently_reconstructed": True,
        "frozen_scalar_authority_exact": True,
        "fold_stratified_reruns_exact": True,
        "outcomes_or_labels_excluded": True,
    }
    state = {
        "schema_version": "m04r14-t14-10-wf03-combined-batch-verification-v1",
        "status": "complete", "passed": all(gates.values()), "gates": gates,
        "producer_result_digest": result["result_digest"],
        "preregistration_digest": preregistration["preregistration_digest"],
        "verifier_commit": verifier_commit,
        "verifier_runtime_sha256": verifier_sha256,
        "queries_verified": len(observations),
        "analogue_links_verified": len(observations) * producer.TOP_K,
        "same_symbol_links_verified": sum(
            row["same_symbol_neighbors"] for row in observations
        ),
        "case_observation_digest": stable_hash(observations),
        "rerun_sample_ids": sample_ids,
        "rerun_digest": stable_hash(reruns),
        "rerun_queries": len(reruns),
        "authority_query_id": AUTHORITY_QUERY_ID,
        "elapsed_seconds": perf_counter() - started,
        "outcomes_or_labels_used": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "combined_batch_complete": True,
    }
    if not state["passed"]:
        raise CombinedBatchVerificationError("combined verification gates failed")
    return base._sealed(state, "verification_digest")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    value = verify(args.repository)
    if not args.dry_run:
        root = args.repository.resolve() / OUTPUT_RELATIVE
        if root.exists() or root.is_symlink():
            raise CombinedBatchVerificationError("combined verification exists")
        root.mkdir(parents=True)
        base._atomic(root / "VERIFIED.json", value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
