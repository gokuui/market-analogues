from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .adapters import OHLCVSource
from .causal_prefix import causal_prefix_digest
from .distance import (
    complete_representation_distance, representation_distance_lower_bound,
)
from .exact_batch import (
    exact_channel_rows, exact_representations_at_positions,
    materialize_exact_representations,
    sliding_exact_representations,
)
from .packed_bound_search import (
    BoundProposal, PackedBoundQuery, scan_packed_bound_proposals,
)
from .packed_bound_store import load_packed_generation
from .representation import Representation, represent
from .search import ScoredCandidate, eligible, latest_eligible_cutoff, select_scored
from .types import (
    AnalogueMatch, Episode, EpisodeKey, InstrumentKey, SearchQuery, stable_hash,
)


CERTIFIED_PACKED_SEARCH_VERSION = "m04r-certified-packed-search-v1"
CERTIFIED_PACKED_SEARCH_REQUESTED_VERSION = "m04r-certified-packed-search-v2"
CERTIFIED_PACKED_SEARCH_HYBRID_VERSION = "m04r-certified-packed-search-v3"


class CertifiedPackedSearchError(RuntimeError):
    pass


@dataclass(frozen=True)
class CompletionRound:
    frontier_rows: int
    exact_rows: int
    next_lower_bound: float | None
    constrained_threshold: float
    selected_rows: int
    certified: bool
    proposal_digest: str


@dataclass(frozen=True)
class PackedSearchCertificate:
    schema_version: str
    contract_digest: str
    generation_id: str
    query_episode_id: str
    input_digest: str
    eligible_candidates: int
    exact_evaluated: int
    safely_pruned: int
    stopped_early: bool
    stop_threshold: float
    next_lower_bound: float | None
    maximum_quantized_bound_excess: float
    materialization_groups: int
    sparse_symbols: int
    batch_symbols: int
    rounds: tuple[CompletionRound, ...]
    result_digest: str
    elapsed_seconds: float


@dataclass(frozen=True)
class CertifiedPackedSearchResult:
    matches: tuple[AnalogueMatch, ...]
    certificate: PackedSearchCertificate


def certified_packed_search_contract(
    *, requested_positions: bool = False,
    hybrid_requested_positions: bool = False,
) -> dict[str, Any]:
    if requested_positions and hybrid_requested_positions:
        raise ValueError("requested-position modes are mutually exclusive")
    if hybrid_requested_positions:
        version = CERTIFIED_PACKED_SEARCH_HYBRID_VERSION
    elif requested_positions:
        version = CERTIFIED_PACKED_SEARCH_REQUESTED_VERSION
    else:
        version = CERTIFIED_PACKED_SEARCH_VERSION
    payload: dict[str, Any] = {
        "schema_version": version,
        "frontier": (
            "globally stable quantized-bound prefix plus first omitted bound; "
            "expand geometrically until certified or fail closed"
        ),
        "stop_rule": (
            "constrained top-k is full and first omitted bound is strictly greater "
            "than kth native exact distance, or literal eligible-row exhaustion"
        ),
        "diversity_safety": (
            "recompute constrained selection after every completed frontier; an "
            "incomplete selection has infinite threshold and cannot certify"
        ),
        "exact_scoring": (
            "native distance-v1 with scalar reconstruction for sparse symbol groups "
            "and requested-cutoff vector construction for dense symbol groups"
            if hybrid_requested_positions else
            "native distance-v1 with per-symbol vector construction of only requested "
            "cutoff positions"
            if requested_positions else
            "native distance-v1 with adaptive sparse/full grouped reconstruction"
        ),
        "quantized_check": "stored lower bound <= recomputed native lower bound + tolerance",
        "source_check": "every materialized stock and benchmark matches packed causal-prefix provenance",
        "tie_rule": "ascending exact distance then 24-hex episode ID; strict frontier stop",
        "outcomes_or_labels_used": False,
    }
    payload["digest"] = stable_hash(payload)
    return payload


def _prefix_matches(frame: pd.DataFrame, expected: dict[str, Any]) -> bool:
    return asdict(causal_prefix_digest(frame, str(expected["requested_cutoff"]))) == expected


def _score_group(
    symbol: str,
    proposals: tuple[BoundProposal, ...],
    *,
    query: Episode,
    query_representation: Representation,
    source: OHLCVSource,
    request: SearchQuery,
    benchmark: pd.DataFrame,
    store_dataset_id: str,
    expected_prefix: dict[str, Any],
    maximum_candidate_cutoff: pd.Timestamp,
    sparse_cutoff: int,
    tolerance: float,
    requested_positions: bool,
    hybrid_requested_positions: bool,
) -> tuple[list[ScoredCandidate], float, str]:
    key = InstrumentKey(store_dataset_id, symbol)
    bars = source.load(key)
    if not _prefix_matches(bars, expected_prefix):
        raise CertifiedPackedSearchError(f"packed stock causal prefix is stale: {symbol}")
    latest = min(
        latest_eligible_cutoff(query, request.minimum_history_gap_bars),
        maximum_candidate_cutoff,
    )
    frame = bars[bars.timestamp <= latest].reset_index(drop=True)
    cutoff_to_position = {
        int(pd.Timestamp(value).value): index
        for index, value in enumerate(frame.timestamp)
    }
    use_batch = len(proposals) >= sparse_cutoff
    use_requested_batch = requested_positions or (
        hybrid_requested_positions and use_batch
    )
    representations: dict[int, Representation] = {}
    if use_requested_batch:
        requested = []
        for proposal in proposals:
            position = cutoff_to_position.get(proposal.cutoff_ns)
            if position is None or position + 1 < query.key.lookback:
                raise CertifiedPackedSearchError(
                    f"cannot reconstruct packed cutoff {symbol}:{proposal.cutoff_ns}"
                )
            requested.append(position)
        materialized = exact_representations_at_positions(
            frame, benchmark, positions=np.asarray(requested, dtype=int),
            lookback=query.key.lookback,
        )
        representations = {
            proposal.cutoff_ns: representation
            for proposal, representation in zip(proposals, materialized)
        }
    elif use_batch:
        batch = sliding_exact_representations(
            frame, benchmark, lookback=query.key.lookback, stride=5,
            batch_size=128,
        )
        representations = {
            int(pd.Timestamp(frame.timestamp.iloc[int(position)]).value): representation
            for position, representation in zip(batch.positions, batch.representations)
        }
    output: list[ScoredCandidate] = []
    maximum_excess = 0.0
    for proposal in proposals:
        position = cutoff_to_position.get(proposal.cutoff_ns)
        if position is None or position + 1 < query.key.lookback:
            raise CertifiedPackedSearchError(
                f"cannot reconstruct packed cutoff {symbol}:{proposal.cutoff_ns}"
            )
        window = frame.iloc[
            position - query.key.lookback + 1:position + 1
        ].reset_index(drop=True)
        episode = Episode(
            EpisodeKey(
                key, pd.Timestamp(proposal.cutoff_ns), query.key.lookback,
                query.key.representation_version,
            ),
            window, benchmark, proposal.quality_tier,
        )
        if episode.key.id != proposal.episode_id:
            raise CertifiedPackedSearchError("packed episode identity changed during reconstruction")
        if not eligible(query, episode, request):
            raise CertifiedPackedSearchError(f"packed proposal is ineligible: {proposal.episode_id}")
        if use_requested_batch or use_batch:
            candidate_representation = representations.get(proposal.cutoff_ns)
        else:
            positions, channels = exact_channel_rows(
                window, benchmark, lookback=query.key.lookback, stride=1,
            )
            materialized = materialize_exact_representations(channels)
            candidate_representation = (
                materialized[0]
                if len(positions) == 1 and len(materialized) == 1 else None
            )
        if candidate_representation is None:
            raise CertifiedPackedSearchError(
                f"batch omitted packed cutoff {symbol}:{proposal.cutoff_ns}"
            )
        native_lower, components, rigid = representation_distance_lower_bound(
            query_representation, candidate_representation,
        )
        excess = proposal.lower_bound - native_lower
        maximum_excess = max(maximum_excess, excess)
        if excess > tolerance:
            raise CertifiedPackedSearchError(
                f"quantized lower bound exceeds native bound for {proposal.episode_id}: {excess:.3e}"
            )
        total, exact_components, path = complete_representation_distance(
            query_representation, candidate_representation,
            native_lower, components, rigid,
        )
        output.append(ScoredCandidate(AnalogueMatch(
            episode.key, total, exact_components, path,
            episode.quality_tier, episode.quality_issues,
        ), episode))
    return output, maximum_excess, "batch" if use_batch else "sparse"


def _score_new_proposals(
    proposals: Iterable[BoundProposal],
    *,
    query: Episode,
    source: OHLCVSource,
    request: SearchQuery,
    store_dataset_id: str,
    manifest: dict[str, Any],
    workers: int,
    sparse_cutoff: int,
    tolerance: float,
    requested_positions: bool,
    hybrid_requested_positions: bool,
) -> tuple[list[ScoredCandidate], float, int, int]:
    if workers < 1 or sparse_cutoff < 1:
        raise ValueError("workers and sparse cutoff must be positive")
    grouped: dict[str, list[BoundProposal]] = {}
    for proposal in proposals:
        grouped.setdefault(proposal.symbol, []).append(proposal)
    provenance = manifest.get("provenance", {})
    prefixes = provenance.get("source_prefixes", {})
    benchmark_prefix = provenance.get("benchmark_prefix")
    if not isinstance(prefixes, dict) or not isinstance(benchmark_prefix, dict):
        raise CertifiedPackedSearchError("packed source-prefix provenance is incomplete")
    benchmark = source.load_benchmark()
    if benchmark is None or not _prefix_matches(benchmark, benchmark_prefix):
        raise CertifiedPackedSearchError("packed benchmark causal prefix is stale")
    maximum_cutoff = pd.Timestamp(str(benchmark_prefix["requested_cutoff"]))
    query_representation = represent(query)

    def one(item: tuple[str, list[BoundProposal]]):
        symbol, rows = item
        expected = prefixes.get(symbol)
        if not isinstance(expected, dict):
            raise CertifiedPackedSearchError(f"missing packed source prefix: {symbol}")
        return _score_group(
            symbol, tuple(rows), query=query,
            query_representation=query_representation, source=source,
            request=request, benchmark=benchmark,
            store_dataset_id=store_dataset_id, expected_prefix=expected,
            maximum_candidate_cutoff=maximum_cutoff,
            sparse_cutoff=sparse_cutoff, tolerance=tolerance,
            requested_positions=requested_positions,
            hybrid_requested_positions=hybrid_requested_positions,
        )

    items = sorted(grouped.items())
    if workers == 1:
        results = map(one, items)
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        results = executor.map(one, items)
    scored: list[ScoredCandidate] = []
    maximum_excess = 0.0
    sparse = batch = 0
    try:
        for values, excess, mode in results:
            scored.extend(values)
            maximum_excess = max(maximum_excess, excess)
            sparse += mode == "sparse"
            batch += mode == "batch"
    finally:
        if workers != 1:
            executor.shutdown(wait=True)
    return scored, maximum_excess, sparse, batch


def certified_packed_search(
    query: Episode,
    source: OHLCVSource,
    request: SearchQuery,
    store_root: Path,
    generation_id: str,
    *,
    store_dataset_id: str,
    initial_frontier_rows: int = 8_192,
    maximum_frontier_rows: int = 65_536,
    seed_rows: int = 512,
    block_rows: int = 2_048,
    workers: int = 8,
    sparse_cutoff: int = 8,
    tolerance: float = 1e-12,
    verify_content: bool = True,
    requested_positions: bool = False,
    hybrid_requested_positions: bool = False,
) -> CertifiedPackedSearchResult:
    if requested_positions and hybrid_requested_positions:
        raise ValueError("requested-position modes are mutually exclusive")
    if (
        initial_frontier_rows < request.top_k
        or maximum_frontier_rows < initial_frontier_rows
        or seed_rows < request.top_k
        or seed_rows > initial_frontier_rows
    ):
        raise ValueError("frontier limits do not cover requested top-k")
    if store_dataset_id != query.key.instrument.dataset_id and not request.cross_dataset:
        raise CertifiedPackedSearchError("cross-dataset packed search is not authorized")
    started = perf_counter()
    contract = certified_packed_search_contract(
        requested_positions=requested_positions,
        hybrid_requested_positions=hybrid_requested_positions,
    )
    loaded = load_packed_generation(
        store_root, generation_id, verify_content=verify_content,
        validate_records=False,
    )
    benchmark_input = source.load_benchmark()
    input_provenance = {
        "query_stock_prefix": asdict(causal_prefix_digest(
            source.load(query.key.instrument), query.key.cutoff,
        )),
        "query_benchmark_prefix": (
            asdict(causal_prefix_digest(benchmark_input, query.key.cutoff))
            if benchmark_input is not None else None
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
        "packed_provenance_digest": loaded.manifest["provenance_digest"],
    }
    input_digest = stable_hash(input_provenance)
    packed_query = PackedBoundQuery(
        query.key.id, query.key.instrument.source_symbol,
        int(pd.Timestamp(query.bars.timestamp.iloc[0]).value),
        int(latest_eligible_cutoff(query, request.minimum_history_gap_bars).value),
        represent(query), request.quality_tiers,
    )
    scored_by_id: dict[str, ScoredCandidate] = {}
    maximum_excess = 0.0
    sparse_symbols = batch_symbols = 0
    rounds: list[CompletionRound] = []
    frontier_rows = initial_frontier_rows
    eligible_count = -1
    while True:
        proposal = scan_packed_bound_proposals(
            store_root, generation_id, packed_query,
            route_quotas={"composite": frontier_rows + 1},
            block_rows=block_rows, verify_content=False,
        )
        eligible_count = proposal.eligible_rows
        frontier = proposal.candidates[:frontier_rows]
        next_lower = (
            proposal.candidates[frontier_rows].lower_bound
            if len(proposal.candidates) > frontier_rows else None
        )
        if not scored_by_id:
            pending = list(frontier[:seed_rows])
        else:
            pending = []
        while True:
            if pending:
                newly_scored, excess, sparse, batch = _score_new_proposals(
                    pending, query=query, source=source, request=request,
                    store_dataset_id=store_dataset_id, manifest=loaded.manifest,
                    workers=workers, sparse_cutoff=sparse_cutoff,
                    tolerance=tolerance,
                    requested_positions=requested_positions,
                    hybrid_requested_positions=hybrid_requested_positions,
                )
                scored_by_id.update({
                    row.match.episode_key.id: row for row in newly_scored
                })
                maximum_excess = max(maximum_excess, excess)
                sparse_symbols += sparse
                batch_symbols += batch
            selected = select_scored(list(scored_by_id.values()), request)
            threshold = (
                max(row.total_distance for row in selected)
                if len(selected) >= request.top_k else float("inf")
            )
            pending = [
                row for row in frontier
                if row.episode_id not in scored_by_id
                and row.lower_bound <= threshold
            ]
            certified = (
                len(selected) >= request.top_k and not pending
                and (next_lower is None or next_lower > threshold)
            )
            rounds.append(CompletionRound(
                len(frontier), len(scored_by_id), next_lower, threshold,
                len(selected), certified, proposal.candidate_digest,
            ))
            if certified or not pending:
                break
        if certified:
            break
        if len(frontier) >= eligible_count:
            if len(selected) < request.top_k:
                raise CertifiedPackedSearchError("eligible universe cannot fill constrained top-k")
            break
        if frontier_rows >= maximum_frontier_rows:
            raise CertifiedPackedSearchError(
                "certified frontier exceeds configured maximum; no result emitted"
            )
        frontier_rows = min(maximum_frontier_rows, frontier_rows * 2, eligible_count)

    matches = tuple(select_scored(list(scored_by_id.values()), request))
    threshold = max(row.total_distance for row in matches)
    next_lower = rounds[-1].next_lower_bound
    stopped_early = next_lower is not None
    deterministic = {
        "schema_version": contract["schema_version"],
        "contract_digest": contract["digest"],
        "generation_id": generation_id,
        "query_episode_id": query.key.id,
        "input_digest": input_digest,
        "eligible_candidates": eligible_count,
        "exact_evaluated": len(scored_by_id),
        "safely_pruned": eligible_count - len(scored_by_id),
        "stopped_early": stopped_early,
        "stop_threshold_hex": threshold.hex(),
        "next_lower_bound_hex": next_lower.hex() if next_lower is not None else None,
        "maximum_quantized_bound_excess_hex": maximum_excess.hex(),
        "rounds": [asdict(value) for value in rounds],
        "matches": [{
            "episode_id": match.episode_key.id,
            "total_hex": match.total_distance.hex(),
            "components": {
                key: value.hex() for key, value in sorted(match.component_distances.items())
            },
            "alignment": match.alignment,
        } for match in matches],
        "real_forward_outcomes_accessed": False,
    }
    digest = stable_hash(deterministic)
    certificate = PackedSearchCertificate(
        contract["schema_version"],
        contract["digest"],
        generation_id, query.key.id, input_digest,
        eligible_count, len(scored_by_id), eligible_count - len(scored_by_id),
        stopped_early, threshold, next_lower, maximum_excess,
        sparse_symbols + batch_symbols, sparse_symbols, batch_symbols,
        tuple(rounds), digest, perf_counter() - started,
    )
    return CertifiedPackedSearchResult(matches, certificate)
