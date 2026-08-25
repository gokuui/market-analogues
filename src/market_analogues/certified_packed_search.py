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
    complete_representation_distance, representation_distance,
    representation_distance_lower_bound,
)
from .exact_batch import (
    batch_representation_lower_bounds, exact_channel_rows,
    exact_representations_at_positions,
    materialize_exact_representations,
    sliding_exact_representations,
)
from .packed_bound_search import (
    BRANCH_AWARE_SEARCH_SCHEMA_VERSION, SEARCH_SCHEMA_VERSION,
    BoundProposal, BoundProposalReport, PackedBoundQuery,
    _packed_query_input_digest, bound_proposal_candidate_digest,
    scan_packed_bound_proposals,
    packed_bound_search_contract, scan_packed_bound_threshold,
)
from .packed_bound_store import load_packed_generation
from .representation import Representation, represent, representation_input_digest
from .search import ScoredCandidate, eligible, latest_eligible_cutoff, select_scored
from .types import (
    AnalogueMatch, Episode, EpisodeKey, InstrumentKey, SearchQuery, stable_hash,
)


CERTIFIED_PACKED_SEARCH_VERSION = "m04r-certified-packed-search-v1"
CERTIFIED_PACKED_SEARCH_REQUESTED_VERSION = "m04r-certified-packed-search-v2"
CERTIFIED_PACKED_SEARCH_HYBRID_VERSION = "m04r-certified-packed-search-v3"
CERTIFIED_PACKED_SEARCH_VECTOR_VERSION = "m04r-certified-packed-search-v4"
CERTIFIED_PACKED_SEARCH_DEFERRED_VERSION = "m04r-certified-packed-search-v5"
CERTIFIED_PACKED_SEARCH_COMPACT_VERSION = "m04r-certified-packed-search-v6"
CERTIFIED_PACKED_SEARCH_NATIVE_BOUND_VERSION = "m04r-certified-packed-search-v7"
CERTIFIED_PACKED_SEARCH_BRANCH_AWARE_VERSION = "m04r-certified-packed-search-v8"


class CertifiedPackedSearchError(RuntimeError):
    pass


class CertifiedFrontierOverflow(CertifiedPackedSearchError):
    """Typed fail-closed state for a correct search needing a larger frontier."""

    def __init__(
        self, *, frontier_rows: int, eligible_candidates: int,
        exact_evaluated: int, stop_threshold: float,
        next_lower_bound: float | None,
    ) -> None:
        super().__init__(
            "certified frontier exceeds configured maximum; no result emitted"
        )
        self.frontier_rows = frontier_rows
        self.eligible_candidates = eligible_candidates
        self.exact_evaluated = exact_evaluated
        self.stop_threshold = stop_threshold
        self.next_lower_bound = next_lower_bound


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
class NativeBoundAccounting:
    native_bound_evaluated: int
    exact_dtw_evaluated: int
    native_bound_pruned: int
    packed_bound_pruned: int


@dataclass(frozen=True)
class PackedSearchCertificateV7(PackedSearchCertificate):
    native_bound_accounting: NativeBoundAccounting
    minimum_native_pruned_bound: float | None
    threshold_closure_passes: tuple["ThresholdClosurePass", ...]


@dataclass(frozen=True)
class ThresholdClosurePass:
    lower_exclusive: float | None
    upper_inclusive: float
    admitted_rows: int
    cumulative_native_bound_evaluated: int
    cumulative_exact_dtw_evaluated: int
    selected_rows: int
    resulting_threshold: float
    minimum_packed_unclassified_bound: float | None
    minimum_native_pruned_bound: float | None
    excluded_prefix_digest: str
    admitted_set_digest: str
    scan_result_digest: str
    certified: bool


@dataclass(frozen=True)
class CertifiedPackedSearchResult:
    matches: tuple[AnalogueMatch, ...]
    certificate: PackedSearchCertificate | PackedSearchCertificateV7


@dataclass(frozen=True)
class CompactScoredCandidate:
    """Exact score without the candidate's heavyweight pandas episode frame."""
    match: AnalogueMatch
    instrument: InstrumentKey
    start_ns: int
    cutoff_ns: int


@dataclass(frozen=True)
class NativeBoundDeferredCandidate:
    """Compact native-bound proof; exact state is reconstructed only if reopened."""

    episode_key: EpisodeKey
    proposal: BoundProposal
    lower_bound: float


def certified_packed_search_contract(
    *, requested_positions: bool = False,
    hybrid_requested_positions: bool = False,
    vector_lower_bounds: bool = False,
    deferred_alignments: bool = False,
    compact_scored: bool = False,
    native_bound_deferral: bool = False,
    streaming_threshold_closure: bool = False,
    branch_aware_packed_bounds: bool = False,
) -> dict[str, Any]:
    if requested_positions and hybrid_requested_positions:
        raise ValueError("requested-position modes are mutually exclusive")
    if vector_lower_bounds and not requested_positions:
        raise ValueError("vector lower bounds require requested positions")
    if deferred_alignments and not vector_lower_bounds:
        raise ValueError("deferred alignments require vector lower bounds")
    if compact_scored and not deferred_alignments:
        raise ValueError("compact scored state requires deferred alignments")
    if streaming_threshold_closure and not (
        native_bound_deferral and compact_scored
    ):
        raise ValueError(
            "streaming threshold closure requires compact native-bound deferral"
        )
    if branch_aware_packed_bounds:
        version = CERTIFIED_PACKED_SEARCH_BRANCH_AWARE_VERSION
    elif native_bound_deferral:
        version = CERTIFIED_PACKED_SEARCH_NATIVE_BOUND_VERSION
    elif compact_scored:
        version = CERTIFIED_PACKED_SEARCH_COMPACT_VERSION
    elif deferred_alignments:
        version = CERTIFIED_PACKED_SEARCH_DEFERRED_VERSION
    elif vector_lower_bounds:
        version = CERTIFIED_PACKED_SEARCH_VECTOR_VERSION
    elif hybrid_requested_positions:
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
            "native distance-v1 with requested representations, vector non-DTW "
            "bounds and compiled two-row exact DTW distance; Python parent paths "
            "are reconstructed and parity-checked only for constrained final rows"
            if deferred_alignments else
            "native distance-v1 with per-symbol requested-cutoff vector construction "
            "and vectorized exact non-DTW lower bounds before scalar DTW completion"
            if vector_lower_bounds else
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
        "retained_exact_state": (
            "retain every exact-scored candidate as its match plus instrument and "
            "observed-window start/cutoff nanoseconds; discard no score; interval "
            "overlap is exact for contiguous windows from one instrument; reconstruct "
            "only final selected episodes for alignment"
            if compact_scored else "retain every exact-scored episode"
        ),
        "outcomes_or_labels_used": False,
    }
    if native_bound_deferral:
        payload["native_bound_completion"] = (
            "evaluate the exact native non-DTW partial sum before DTW; defer only "
            "bounds strictly above the current "
            "constrained threshold; complete equality ties; reopen deferred rows "
            "whenever constrained selection raises the threshold, and complete every "
            "deferred row when selection is incomplete"
        )
    if streaming_threshold_closure:
        payload["frontier_overflow_completion"] = (
            "at the configured sorted-prefix ceiling, stream every eligible packed "
            "row in inclusive threshold bands; retain prior exact/native-bound "
            "classifications; repeat only when constrained selection raises the "
            "threshold; certify when both minimum remaining packed and native "
            "bounds are strictly greater than the final threshold"
        )
    if branch_aware_packed_bounds:
        payload["packed_bound"] = (
            "branch-aware outward joined-IQR interval bound v2 over the immutable "
            "v1 float16/error-radius row; distance-v1 itself is unchanged"
        )
        payload["packed_bound_contract_digest"] = packed_bound_search_contract(
            branch_aware=True,
        )["digest"]
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
    vector_lower_bounds: bool,
    deferred_alignments: bool,
    native_bound_deferral: bool,
    completion_threshold: float,
) -> tuple[
    list[ScoredCandidate], list[NativeBoundDeferredCandidate], float, str,
]:
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
    lower_bounds: dict[int, tuple[float, dict[str, float], float]] = {}
    if vector_lower_bounds:
        ordered = tuple(representations[proposal.cutoff_ns] for proposal in proposals)
        bounded = batch_representation_lower_bounds(
            query_representation, ordered,
        )
        lower_bounds = {
            proposal.cutoff_ns: (
                float(bounded.totals[index]),
                {
                    name: float(values[index])
                    for name, values in bounded.components.items()
                },
                float(bounded.rigid_price[index]),
            )
            for index, proposal in enumerate(proposals)
        }
    output: list[ScoredCandidate] = []
    deferred: list[NativeBoundDeferredCandidate] = []
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
        if vector_lower_bounds:
            native_lower, components, rigid = lower_bounds[proposal.cutoff_ns]
        else:
            native_lower, components, rigid = representation_distance_lower_bound(
                query_representation, candidate_representation,
            )
        native_values = [native_lower, rigid, *components.values()]
        if not all(np.isfinite(value) and value >= 0 for value in native_values):
            raise CertifiedPackedSearchError(
                f"native lower bound is invalid for {proposal.episode_id}"
            )
        excess = proposal.lower_bound - native_lower
        if not np.isfinite(excess):
            raise CertifiedPackedSearchError(
                f"packed/native lower-bound comparison is invalid for {proposal.episode_id}"
            )
        maximum_excess = max(maximum_excess, excess)
        if excess > tolerance:
            raise CertifiedPackedSearchError(
                f"quantized lower bound exceeds native bound for {proposal.episode_id}: {excess:.3e}"
            )
        # Equality must complete so a distance/episode-ID tie can never be pruned.
        if native_bound_deferral and native_lower > completion_threshold:
            deferred.append(NativeBoundDeferredCandidate(
                episode.key, proposal, native_lower,
            ))
        else:
            total, exact_components, path = complete_representation_distance(
                query_representation, candidate_representation,
                native_lower, components, rigid,
                reconstruct_path=not deferred_alignments,
            )
            exact_values = [total, *exact_components.values()]
            if not all(np.isfinite(value) and value >= 0 for value in exact_values):
                raise CertifiedPackedSearchError(
                    f"completed exact distance is invalid for {proposal.episode_id}"
                )
            if total + tolerance < native_lower:
                raise CertifiedPackedSearchError(
                    f"completed exact distance violates native bound for {proposal.episode_id}"
                )
            output.append(ScoredCandidate(AnalogueMatch(
                episode.key, total, exact_components, path,
                episode.quality_tier, episode.quality_issues,
            ), episode))
    return output, deferred, maximum_excess, "batch" if use_batch else "sparse"


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
    vector_lower_bounds: bool,
    deferred_alignments: bool,
    compact_scored_records: bool = False,
    native_bound_deferral: bool = False,
    completion_threshold: float = float("inf"),
    benchmark_override: pd.DataFrame | None = None,
    query_representation_override: Representation | None = None,
) -> tuple[
    list[ScoredCandidate | CompactScoredCandidate],
    list[NativeBoundDeferredCandidate], float, int, int,
]:
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
    benchmark = (
        benchmark_override
        if benchmark_override is not None else source.load_benchmark()
    )
    if benchmark is None or not _prefix_matches(benchmark, benchmark_prefix):
        raise CertifiedPackedSearchError("packed benchmark causal prefix is stale")
    maximum_cutoff = pd.Timestamp(str(benchmark_prefix["requested_cutoff"]))
    query_representation = (
        query_representation_override
        if query_representation_override is not None else represent(query)
    )

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
            vector_lower_bounds=vector_lower_bounds,
            deferred_alignments=deferred_alignments,
            native_bound_deferral=native_bound_deferral,
            completion_threshold=completion_threshold,
        )

    items = sorted(grouped.items())
    if workers == 1:
        results = map(one, items)
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        results = executor.map(one, items)
    scored: list[ScoredCandidate | CompactScoredCandidate] = []
    deferred: list[NativeBoundDeferredCandidate] = []
    maximum_excess = 0.0
    sparse = batch = 0
    try:
        for values, deferred_values, excess, mode in results:
            if compact_scored_records:
                scored.extend(
                    CompactScoredCandidate(
                        item.match, item.episode.key.instrument,
                        int(pd.Timestamp(item.episode.bars.timestamp.iloc[0]).value),
                        int(pd.Timestamp(item.episode.bars.timestamp.iloc[-1]).value),
                    )
                    for item in values
                )
            else:
                scored.extend(values)
            deferred.extend(deferred_values)
            maximum_excess = max(maximum_excess, excess)
            sparse += mode == "sparse"
            batch += mode == "batch"
    finally:
        if workers != 1:
            executor.shutdown(wait=True)
    return scored, deferred, maximum_excess, sparse, batch


def _select_compact_scored(
    scored: Iterable[CompactScoredCandidate], request: SearchQuery,
) -> list[AnalogueMatch]:
    ranked = sorted(
        scored, key=lambda item: (item.match.total_distance, item.match.episode_key.id),
    )
    selected: list[CompactScoredCandidate] = []
    per_instrument: dict[InstrumentKey, int] = {}
    for item in ranked:
        if per_instrument.get(item.instrument, 0) >= request.max_per_instrument:
            continue
        if request.deduplicate_overlaps and any(
            item.instrument == chosen.instrument
            and item.start_ns <= chosen.cutoff_ns
            and chosen.start_ns <= item.cutoff_ns
            for chosen in selected
        ):
            continue
        selected.append(item)
        per_instrument[item.instrument] = per_instrument.get(item.instrument, 0) + 1
        if len(selected) >= request.top_k:
            break
    return [item.match for item in selected]


def _select_retained_scored(
    scored: Iterable[ScoredCandidate | CompactScoredCandidate],
    request: SearchQuery, *, compact: bool,
) -> list[AnalogueMatch]:
    values = list(scored)
    if compact:
        return _select_compact_scored(values, request)  # type: ignore[arg-type]
    return select_scored(values, request)  # type: ignore[arg-type]


def _close_native_bound_deferred(
    scored_by_id: dict[str, ScoredCandidate | CompactScoredCandidate],
    deferred_by_id: dict[str, NativeBoundDeferredCandidate],
    request: SearchQuery,
    *, compact: bool,
    complete_many: Any,
) -> tuple[list[AnalogueMatch], float, int]:
    """Reach a fixed point under non-monotone constrained selection.

    Adding a closer overlapping row can remove several selected rows and raise
    the kth threshold.  Therefore deferred rows are reconsidered after every
    completion wave.  A bound equal to the threshold is always completed.
    """
    completed = 0
    while True:
        selected = _select_retained_scored(
            scored_by_id.values(), request, compact=compact,
        )
        threshold = (
            max(row.total_distance for row in selected)
            if len(selected) >= request.top_k else float("inf")
        )
        ready = sorted(
            (
                value for value in deferred_by_id.values()
                if value.lower_bound <= threshold
            ),
            key=lambda value: (value.lower_bound, value.episode_key.id),
        )
        if not ready:
            return selected, threshold, completed
        completed_values = complete_many(ready)
        by_id = {value.match.episode_key.id: value for value in completed_values}
        if set(by_id) != {value.episode_key.id for value in ready}:
            raise CertifiedPackedSearchError("deferred native-bound completion differs")
        for value in ready:
            episode_id = value.episode_key.id
            result = by_id[episode_id]
            scored_by_id[episode_id] = result
            del deferred_by_id[episode_id]
            completed += 1


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
    vector_lower_bounds: bool = False,
    deferred_alignments: bool = False,
    compact_scored: bool = False,
    native_bound_deferral: bool = False,
    streaming_threshold_closure: bool = False,
    branch_aware_packed_bounds: bool = False,
    threshold_scan_block_order: str = "forward",
    precomputed_proposal: BoundProposalReport | None = None,
) -> CertifiedPackedSearchResult:
    if requested_positions and hybrid_requested_positions:
        raise ValueError("requested-position modes are mutually exclusive")
    if vector_lower_bounds and not requested_positions:
        raise ValueError("vector lower bounds require requested positions")
    if deferred_alignments and not vector_lower_bounds:
        raise ValueError("deferred alignments require vector lower bounds")
    if compact_scored and not deferred_alignments:
        raise ValueError("compact scored state requires deferred alignments")
    if streaming_threshold_closure and not (
        native_bound_deferral and compact_scored
    ):
        raise ValueError(
            "streaming threshold closure requires compact native-bound deferral"
        )
    if threshold_scan_block_order not in {"forward", "reverse"}:
        raise ValueError("threshold scan block order must be forward or reverse")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
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
        vector_lower_bounds=vector_lower_bounds,
        deferred_alignments=deferred_alignments,
        compact_scored=compact_scored,
        native_bound_deferral=native_bound_deferral,
        streaming_threshold_closure=streaming_threshold_closure,
        branch_aware_packed_bounds=branch_aware_packed_bounds,
    )
    loaded = load_packed_generation(
        store_root, generation_id, verify_content=verify_content,
        validate_records=False,
    )
    benchmark_input = source.load_benchmark()
    query_representation = represent(query)
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
        "query_representation_digest": representation_input_digest(
            query_representation,
        ),
    }
    input_digest = stable_hash(input_provenance)
    packed_query = PackedBoundQuery(
        query.key.id, query.key.instrument.source_symbol,
        int(pd.Timestamp(query.bars.timestamp.iloc[0]).value),
        int(latest_eligible_cutoff(query, request.minimum_history_gap_bars).value),
        query_representation, request.quality_tiers,
    )
    if precomputed_proposal is not None:
        candidates = precomputed_proposal.candidates
        candidate_ids = [row.episode_id for row in candidates]
        proposal_schema = (
            BRANCH_AWARE_SEARCH_SCHEMA_VERSION
            if branch_aware_packed_bounds else SEARCH_SCHEMA_VERSION
        )
        proposal_contract_digest = packed_bound_search_contract(
            branch_aware=branch_aware_packed_bounds,
        )["digest"]
        proposal_input_digest = (
            _packed_query_input_digest(packed_query)
            if branch_aware_packed_bounds else None
        )
        expected_result_digest = stable_hash({
            "schema_version": proposal_schema,
            "contract_digest": proposal_contract_digest,
            "generation_id": precomputed_proposal.generation_id,
            "query_episode_id": precomputed_proposal.query_episode_id,
            "rows_scanned": precomputed_proposal.rows_scanned,
            "eligible_rows": precomputed_proposal.eligible_rows,
            "eligible_main_rows": precomputed_proposal.eligible_main_rows,
            "eligible_overflow_rows": precomputed_proposal.eligible_overflow_rows,
            "route_counts": precomputed_proposal.route_counts,
            "route_quotas": precomputed_proposal.route_quotas,
            "candidate_digest": precomputed_proposal.candidate_digest,
            "real_forward_outcomes_accessed": False,
            **(
                {"input_digest": proposal_input_digest}
                if branch_aware_packed_bounds else {}
            ),
        })
        structurally_valid = all((
            precomputed_proposal.schema_version == proposal_schema,
            precomputed_proposal.generation_id == generation_id,
            precomputed_proposal.query_episode_id == query.key.id,
            not branch_aware_packed_bounds or all((
                precomputed_proposal.contract_digest
                == proposal_contract_digest,
                precomputed_proposal.input_digest == proposal_input_digest,
            )),
            int(precomputed_proposal.route_quotas.get("composite", 0))
            >= maximum_frontier_rows + 1,
            precomputed_proposal.rows_scanned
            == len(loaded.rows) + len(loaded.overflow),
            precomputed_proposal.eligible_rows
            == precomputed_proposal.eligible_main_rows
            + precomputed_proposal.eligible_overflow_rows,
            0 <= len(candidates) <= precomputed_proposal.eligible_rows,
            len(candidate_ids) == len(set(candidate_ids)),
            list(candidates) == sorted(
                candidates, key=lambda row: (row.lower_bound, row.episode_id),
            ),
            all(
                len(row.episode_id) == 24
                and row.episode_id.lower() == row.episode_id
                and np.isfinite(row.lower_bound) and row.lower_bound >= 0
                and row.quality_tier in {"A", "B"}
                and row.routes and tuple(sorted(set(row.routes))) == row.routes
                for row in candidates
            ),
            precomputed_proposal.candidate_digest
            == bound_proposal_candidate_digest(candidates),
            precomputed_proposal.result_digest == expected_result_digest,
        ))
        try:
            identifiers_valid = all(
                len(bytes.fromhex(value)) == 12 for value in candidate_ids
            )
        except ValueError:
            identifiers_valid = False
        if not structurally_valid or not identifiers_valid:
            raise CertifiedPackedSearchError("precomputed proposal report differs")
    scored_by_id: dict[str, ScoredCandidate | CompactScoredCandidate] = {}
    evaluated_ids: set[str] = set()
    deferred_by_id: dict[str, NativeBoundDeferredCandidate] = {}
    closure_passes: list[ThresholdClosurePass] = []
    maximum_excess = 0.0
    sparse_symbols = batch_symbols = 0
    rounds: list[CompletionRound] = []
    frontier_rows = initial_frontier_rows
    eligible_count = -1
    while True:
        if precomputed_proposal is None:
            proposal = scan_packed_bound_proposals(
                store_root, generation_id, packed_query,
                route_quotas={"composite": frontier_rows + 1},
                block_rows=block_rows,
                branch_aware=branch_aware_packed_bounds,
                verify_content=False,
            )
            proposal_candidates = proposal.candidates
            proposal_digest = proposal.candidate_digest
            eligible_count = proposal.eligible_rows
        else:
            eligible_count = precomputed_proposal.eligible_rows
            required = min(frontier_rows + 1, eligible_count)
            if len(precomputed_proposal.candidates) < required:
                raise CertifiedPackedSearchError(
                    "precomputed proposal does not cover certified frontier"
                )
            proposal_candidates = precomputed_proposal.candidates[:required]
            proposal_digest = bound_proposal_candidate_digest(
                proposal_candidates,
            )
        frontier = proposal_candidates[:frontier_rows]
        next_lower = (
            proposal_candidates[frontier_rows].lower_bound
            if len(proposal_candidates) > frontier_rows else None
        )
        if not evaluated_ids:
            pending = list(frontier[:seed_rows])
        else:
            pending = []
        threshold = float("inf")
        while True:
            if pending:
                newly_scored, newly_deferred, excess, sparse, batch = _score_new_proposals(
                    pending, query=query, source=source, request=request,
                    store_dataset_id=store_dataset_id, manifest=loaded.manifest,
                    workers=workers, sparse_cutoff=sparse_cutoff,
                    tolerance=tolerance,
                    requested_positions=requested_positions,
                    hybrid_requested_positions=hybrid_requested_positions,
                    vector_lower_bounds=vector_lower_bounds,
                    deferred_alignments=deferred_alignments,
                    compact_scored_records=compact_scored,
                    native_bound_deferral=native_bound_deferral,
                    completion_threshold=threshold,
                    benchmark_override=benchmark_input,
                    query_representation_override=query_representation,
                )
                pending_ids = {row.episode_id for row in pending}
                returned_ids = {
                    row.match.episode_key.id for row in newly_scored
                } | {
                    row.episode_key.id for row in newly_deferred
                }
                if returned_ids != pending_ids:
                    raise CertifiedPackedSearchError(
                        "native-bound completion did not account for every pending row"
                    )
                evaluated_ids.update(row.episode_id for row in pending)
                scored_by_id.update({
                    row.match.episode_key.id: row for row in newly_scored
                })
                deferred_by_id.update({
                    row.episode_key.id: row for row in newly_deferred
                })
                maximum_excess = max(maximum_excess, excess)
                sparse_symbols += sparse
                batch_symbols += batch
            def complete_ready(
                ready: list[NativeBoundDeferredCandidate],
            ) -> list[ScoredCandidate | CompactScoredCandidate]:
                nonlocal maximum_excess, sparse_symbols, batch_symbols
                values, deferred, excess, sparse, batch = _score_new_proposals(
                    (row.proposal for row in ready), query=query, source=source,
                    request=request, store_dataset_id=store_dataset_id,
                    manifest=loaded.manifest, workers=workers,
                    sparse_cutoff=sparse_cutoff, tolerance=tolerance,
                    requested_positions=requested_positions,
                    hybrid_requested_positions=hybrid_requested_positions,
                    vector_lower_bounds=vector_lower_bounds,
                    deferred_alignments=deferred_alignments,
                    compact_scored_records=compact_scored,
                    native_bound_deferral=False,
                    benchmark_override=benchmark_input,
                    query_representation_override=query_representation,
                )
                if deferred:
                    raise CertifiedPackedSearchError(
                        "forced native-bound completion deferred a candidate"
                    )
                maximum_excess = max(maximum_excess, excess)
                sparse_symbols += sparse
                batch_symbols += batch
                return values

            selected, threshold, _ = _close_native_bound_deferred(
                scored_by_id, deferred_by_id, request, compact=compact_scored,
                complete_many=complete_ready,
            )
            pending = [
                row for row in frontier
                if row.episode_id not in evaluated_ids
                and row.lower_bound <= threshold + tolerance
            ]
            minimum_native = min(
                (row.lower_bound for row in deferred_by_id.values()),
                default=None,
            )
            effective_next = min(
                (value for value in (next_lower, minimum_native) if value is not None),
                default=None,
            )
            certified = (
                len(selected) >= request.top_k and not pending
                and (
                    effective_next is None
                    or effective_next > threshold + tolerance
                )
            )
            rounds.append(CompletionRound(
                len(frontier), len(scored_by_id), effective_next, threshold,
                len(selected), certified, proposal_digest,
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
            if not streaming_threshold_closure:
                raise CertifiedFrontierOverflow(
                    frontier_rows=len(frontier), eligible_candidates=eligible_count,
                    exact_evaluated=len(scored_by_id), stop_threshold=threshold,
                    next_lower_bound=effective_next,
                )
            prior_upper: float | None = None
            # The packed lower bound is permitted to exceed the recomputed
            # native bound by ``tolerance``.  Reserve that complete margin when
            # pruning so an exact equality/ID tie can never be hidden inside
            # the accepted numerical error.
            admission_upper = threshold + tolerance
            prefix_exclusions = frozenset(row.episode_id for row in frontier)
            cumulative_admitted = 0
            while True:
                proposal_buffer: list[BoundProposal] = []

                def score_admitted(values: list[BoundProposal]) -> None:
                    nonlocal maximum_excess, sparse_symbols, batch_symbols
                    if not values:
                        return
                    value_ids = {row.episode_id for row in values}
                    if len(value_ids) != len(values) or value_ids & evaluated_ids:
                        raise CertifiedPackedSearchError(
                            "streaming threshold pass repeated an evaluated episode"
                        )
                    newly_scored, newly_deferred, excess, sparse, batch = (
                        _score_new_proposals(
                            values, query=query, source=source, request=request,
                            store_dataset_id=store_dataset_id,
                            manifest=loaded.manifest, workers=workers,
                            sparse_cutoff=sparse_cutoff, tolerance=tolerance,
                            requested_positions=requested_positions,
                            hybrid_requested_positions=hybrid_requested_positions,
                            vector_lower_bounds=vector_lower_bounds,
                            deferred_alignments=deferred_alignments,
                            compact_scored_records=compact_scored,
                            native_bound_deferral=True,
                            completion_threshold=admission_upper,
                            benchmark_override=benchmark_input,
                            query_representation_override=query_representation,
                        )
                    )
                    returned = {
                        row.match.episode_key.id for row in newly_scored
                    } | {row.episode_key.id for row in newly_deferred}
                    if returned != value_ids:
                        raise CertifiedPackedSearchError(
                            "streaming native-bound classification differs"
                        )
                    evaluated_ids.update(value_ids)
                    scored_by_id.update({
                        row.match.episode_key.id: row for row in newly_scored
                    })
                    deferred_by_id.update({
                        row.episode_key.id: row for row in newly_deferred
                    })
                    maximum_excess = max(maximum_excess, excess)
                    sparse_symbols += sparse
                    batch_symbols += batch

                def consume_admitted(values: tuple[BoundProposal, ...]) -> None:
                    proposal_buffer.extend(values)
                    if len(proposal_buffer) >= 32_768:
                        score_admitted(proposal_buffer)
                        proposal_buffer.clear()

                if prior_upper is None:
                    score_admitted([
                        row for row in frontier
                        if row.episode_id not in evaluated_ids
                    ])

                pass_report = scan_packed_bound_threshold(
                    store_root, generation_id, packed_query,
                    lower_exclusive=prior_upper,
                    upper_inclusive=admission_upper,
                    excluded_episode_ids=prefix_exclusions,
                    block_rows=block_rows,
                    block_order=threshold_scan_block_order,
                    branch_aware=branch_aware_packed_bounds,
                    verify_content=False,
                    consume=consume_admitted,
                )
                score_admitted(proposal_buffer)
                proposal_buffer.clear()
                if pass_report.eligible_rows != eligible_count:
                    raise CertifiedPackedSearchError(
                        "streaming threshold eligibility differs from prefix scan"
                    )
                if pass_report.excluded_eligible_rows != len(prefix_exclusions):
                    raise CertifiedPackedSearchError(
                        "streaming threshold prefix exclusions differ"
                    )
                cumulative_admitted += pass_report.admitted_rows
                if len(evaluated_ids) != len(prefix_exclusions) + cumulative_admitted:
                    raise CertifiedPackedSearchError(
                        "streaming threshold admission accounting differs"
                    )
                selected, threshold, _ = _close_native_bound_deferred(
                    scored_by_id, deferred_by_id, request,
                    compact=compact_scored, complete_many=complete_ready,
                )
                minimum_native = min(
                    (row.lower_bound for row in deferred_by_id.values()),
                    default=None,
                )
                minimum_packed = pass_report.minimum_above_upper
                effective_next = min(
                    (
                        value for value in (minimum_packed, minimum_native)
                        if value is not None
                    ),
                    default=None,
                )
                certified = (
                    len(selected) >= request.top_k
                    and (
                        effective_next is None
                        or effective_next > threshold + tolerance
                    )
                )
                closure_passes.append(ThresholdClosurePass(
                    prior_upper, admission_upper, pass_report.admitted_rows,
                    len(evaluated_ids), len(scored_by_id), len(selected), threshold,
                    minimum_packed, minimum_native,
                    pass_report.exclusions_digest,
                    pass_report.admitted_set_digest, pass_report.result_digest,
                    certified,
                ))
                if certified:
                    break
                if not np.isfinite(admission_upper):
                    raise CertifiedPackedSearchError(
                        "eligible universe cannot fill constrained top-k"
                    )
                next_admission_upper = threshold + tolerance
                if next_admission_upper <= admission_upper:
                    raise CertifiedPackedSearchError(
                        "streaming threshold closure made no certified progress"
                    )
                prior_upper, admission_upper = (
                    admission_upper, next_admission_upper,
                )
            break
        frontier_rows = min(maximum_frontier_rows, frontier_rows * 2, eligible_count)

    matches = tuple(_select_retained_scored(
        scored_by_id.values(), request, compact=compact_scored,
    ))
    if deferred_alignments:
        query_representation = represent(query)
        for match in matches:
            candidate = scored_by_id[match.episode_key.id]
            if compact_scored:
                bars = source.load(match.episode_key.instrument)
                frame = bars[bars.timestamp <= match.episode_key.cutoff]
                window = frame.tail(query.key.lookback).reset_index(drop=True)
                candidate_episode = Episode(
                    match.episode_key, window, source.load_benchmark(),
                    match.quality_tier, match.quality_issues,
                )
            else:
                candidate_episode = candidate.episode  # type: ignore[union-attr]
            candidate_representation = represent(candidate_episode)
            total, components, path = representation_distance(
                query_representation, candidate_representation,
            )
            component_delta = max(
                abs(components[name] - match.component_distances[name])
                for name in components
            )
            if (
                abs(total - match.total_distance) > 1e-12
                or component_delta > 1e-12
            ):
                raise CertifiedPackedSearchError(
                    "deferred alignment reconstruction changed exact distance"
                )
            match.alignment = path
    threshold = max(row.total_distance for row in matches)
    if closure_passes:
        final_pass = closure_passes[-1]
        next_lower = min(
            (
                value for value in (
                    final_pass.minimum_packed_unclassified_bound,
                    final_pass.minimum_native_pruned_bound,
                ) if value is not None
            ),
            default=None,
        )
    else:
        next_lower = rounds[-1].next_lower_bound
    stopped_early = next_lower is not None
    exact_count = len(scored_by_id) if native_bound_deferral else len(evaluated_ids)
    safely_pruned = eligible_count - exact_count
    native_accounting = NativeBoundAccounting(
        len(evaluated_ids), len(scored_by_id), len(deferred_by_id),
        eligible_count - len(evaluated_ids),
    )
    if native_bound_deferral and not all((
        not (set(scored_by_id) & set(deferred_by_id)),
        evaluated_ids == set(scored_by_id) | set(deferred_by_id),
        native_accounting.native_bound_evaluated
        == native_accounting.exact_dtw_evaluated
        + native_accounting.native_bound_pruned,
        eligible_count
        == native_accounting.exact_dtw_evaluated
        + native_accounting.native_bound_pruned
        + native_accounting.packed_bound_pruned,
    )):
        raise CertifiedPackedSearchError("native-bound candidate accounting differs")
    deterministic = {
        "schema_version": contract["schema_version"],
        "contract_digest": contract["digest"],
        "generation_id": generation_id,
        "query_episode_id": query.key.id,
        "input_digest": input_digest,
        "eligible_candidates": eligible_count,
        "exact_evaluated": exact_count,
        "safely_pruned": safely_pruned,
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
    if native_bound_deferral:
        deterministic["native_bound_accounting"] = asdict(native_accounting)
        deterministic["minimum_native_pruned_bound_hex"] = (
            min(row.lower_bound for row in deferred_by_id.values()).hex()
            if deferred_by_id else None
        )
        deterministic["threshold_closure_passes"] = [
            asdict(value) for value in closure_passes
        ]
    digest = stable_hash(deterministic)
    common_certificate = (
        contract["schema_version"], contract["digest"], generation_id,
        query.key.id, input_digest, eligible_count, exact_count, safely_pruned,
        stopped_early, threshold, next_lower, maximum_excess,
        sparse_symbols + batch_symbols, sparse_symbols, batch_symbols,
        tuple(rounds), digest, perf_counter() - started,
    )
    if native_bound_deferral:
        certificate = PackedSearchCertificateV7(
            *common_certificate, native_accounting,
            min((row.lower_bound for row in deferred_by_id.values()), default=None),
            tuple(closure_passes),
        )
    else:
        certificate = PackedSearchCertificate(*common_certificate)
    return CertifiedPackedSearchResult(matches, certificate)
