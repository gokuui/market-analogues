from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd

from .adapters import OHLCVSource
from .causal_prefix import causal_prefix_digest
from .exact_batch import (
    batch_exact_price_distances, batch_exact_rigid_price_distances,
    exact_price_representations_at_positions,
)
from .packed_bound_search import (
    COMPONENT_SEARCH_SCHEMA_VERSION,
    BoundProposal,
    BoundProposalReport,
    PackedBoundQuery,
    _packed_query_input_digest,
    bound_proposal_candidate_digest,
    packed_component_search_contract,
    scan_packed_component_bound_proposals_threaded,
)
from .packed_bound_store import load_packed_generation
from .representation import Representation, represent, representation_input_digest
from .search import latest_eligible_cutoff
from .types import AnalogueMatch, Episode, EpisodeKey, InstrumentKey, SearchQuery, stable_hash


PRICE_COMPONENT = "price"
COMPONENT_CERTIFICATE_SCHEMA = "certified-component-search-v1"


class CertifiedComponentSearchError(RuntimeError):
    pass


class ComponentFrontierOverflow(CertifiedComponentSearchError):
    def __init__(self, *, frontier_rows: int, eligible_candidates: int,
                 exact_evaluated: int, threshold: float,
                 next_lower_bound: float | None) -> None:
        super().__init__("component frontier exceeds configured maximum; no result emitted")
        self.frontier_rows = frontier_rows
        self.eligible_candidates = eligible_candidates
        self.exact_evaluated = exact_evaluated
        self.threshold = threshold
        self.next_lower_bound = next_lower_bound


@dataclass(frozen=True)
class ComponentCompletionRound:
    frontier_rows: int
    native_bound_evaluated: int
    exact_evaluated: int
    selected_rows: int
    constrained_threshold: float
    next_lower_bound: float | None
    certified: bool


@dataclass(frozen=True)
class ComponentSearchCertificate:
    schema_version: str
    contract_digest: str
    component: str
    generation_id: str
    query_episode_id: str
    input_digest: str
    proposal_digest: str
    eligible_candidates: int
    native_bound_evaluated: int
    exact_evaluated: int
    native_bound_pruned: int
    packed_bound_pruned: int
    stop_threshold: float
    next_lower_bound: float | None
    maximum_quantized_bound_excess: float
    rounds: tuple[ComponentCompletionRound, ...]
    result_digest: str
    elapsed_seconds: float


@dataclass(frozen=True)
class CertifiedComponentSearchResult:
    matches: tuple[AnalogueMatch, ...]
    certificate: ComponentSearchCertificate


@dataclass(frozen=True)
class _CompactComponentScore:
    match: AnalogueMatch
    instrument: InstrumentKey


@dataclass(frozen=True)
class _PreparedComponentSymbol:
    frame: pd.DataFrame
    timestamp_ns: np.ndarray


def _positions_at_cutoffs(timestamp_ns: np.ndarray, cutoff_ns: np.ndarray) -> np.ndarray:
    if timestamp_ns.ndim != 1 or cutoff_ns.ndim != 1 \
            or timestamp_ns.dtype != np.int64 or cutoff_ns.dtype != np.int64:
        raise CertifiedComponentSearchError("component timestamp arrays differ")
    # The former dictionary comprehension retained the last row for a duplicate
    # timestamp.  Right-sided search preserves that exact behavior.
    positions = np.searchsorted(timestamp_ns, cutoff_ns, side="right") - 1
    valid = positions >= 0
    if np.any(valid):
        valid[valid] &= timestamp_ns[positions[valid]] == cutoff_ns[valid]
    if not np.all(valid):
        raise CertifiedComponentSearchError("cannot reconstruct component proposal cutoff")
    return positions


def certified_component_search_contract(component: str = PRICE_COMPONENT) -> dict[str, object]:
    if component != PRICE_COMPONENT:
        raise CertifiedComponentSearchError("only the exact price component is implemented")
    state: dict[str, object] = {
        "schema_version": COMPONENT_CERTIFICATE_SCHEMA,
        "component": component,
        "distance": (
            "distance-v1 price component: 0.55 rigid price-channel distance plus "
            "0.45 exact banded multivariate DTW"
        ),
        "packed_bound": packed_component_search_contract(component)["digest"],
        "frontier": "geometric safe-bound prefix with strict next-bound closure",
        "selection": "ascending (exact component distance, episode ID), one row per symbol",
        "native_deferral": (
            "complete equality; prune native bounds strictly above the current top-k "
            "threshold; with max_per_instrument=1 additions cannot raise that threshold"
        ),
        "outcomes_or_labels_used": False,
    }
    return {**state, "digest": stable_hash(state)}


def _prefix_matches(frame: pd.DataFrame, expected: dict[str, object]) -> bool:
    return asdict(causal_prefix_digest(frame, str(expected["requested_cutoff"]))) == expected


def _select(values: Iterable[_CompactComponentScore], top_k: int) -> list[AnalogueMatch]:
    selected: list[AnalogueMatch] = []
    symbols: set[InstrumentKey] = set()
    for value in sorted(values, key=lambda item: (
            item.match.total_distance, item.match.episode_key.id)):
        if value.instrument in symbols:
            continue
        selected.append(value.match)
        symbols.add(value.instrument)
        if len(selected) == top_k:
            break
    return selected


def _score_group(
    symbol: str,
    proposals: tuple[BoundProposal, ...],
    *,
    query: Episode,
    query_representation: Representation,
    request: SearchQuery,
    benchmark: pd.DataFrame,
    store_dataset_id: str,
    prepared: _PreparedComponentSymbol,
    completion_threshold: float,
    tolerance: float,
    strengthened_proposal_bound: bool = False,
) -> tuple[list[_CompactComponentScore], int, float]:
    key = InstrumentKey(store_dataset_id, symbol)
    frame = prepared.frame
    requested_array = _positions_at_cutoffs(
        prepared.timestamp_ns,
        np.fromiter((row.cutoff_ns for row in proposals), dtype=np.int64),
    )
    if np.any(requested_array + 1 < query.key.lookback):
        raise CertifiedComponentSearchError("cannot reconstruct component proposal cutoff")
    requested = requested_array.tolist()
    representations = exact_price_representations_at_positions(
        frame, benchmark, positions=np.asarray(requested, dtype=int),
        lookback=query.key.lookback,
    )
    if len(representations) != len(proposals):
        raise CertifiedComponentSearchError("component representation batch differs")
    rigid_price = batch_exact_rigid_price_distances(
        query_representation, representations,
    )
    native_price = .55 * rigid_price
    output: list[_CompactComponentScore] = []
    native_pruned = 0
    maximum_excess = 0.0
    query_start_ns = int(query.bars.timestamp.iloc[0].value)
    query_latest = latest_eligible_cutoff(
        query, request.minimum_history_gap_bars,
    )
    exact_indexes: list[int] = []
    for index, (proposal, _candidate_representation) in enumerate(
            zip(proposals, representations, strict=True)):
        episode_key = EpisodeKey(
            key, pd.Timestamp(proposal.cutoff_ns), query.key.lookback,
            query.key.representation_version,
        )
        # The packed proposal already applied these predicates.  Recheck them
        # directly without materializing a 252-row DataFrame and Episode for
        # every exact candidate; only the immutable key and tier reach output.
        same_instrument_overlap = (
            key == query.key.instrument
            and proposal.cutoff_ns >= query_start_ns
        )
        if episode_key.id != proposal.episode_id \
                or episode_key.id == query.key.id \
                or episode_key.cutoff >= query.key.cutoff \
                or episode_key.cutoff > query_latest \
                or same_instrument_overlap \
                or proposal.quality_tier not in request.quality_tiers \
                or (request.search_datasets
                    and key.dataset_id not in request.search_datasets) \
                or (not request.cross_dataset
                    and key.dataset_id != query.key.instrument.dataset_id):
            raise CertifiedComponentSearchError("component proposal eligibility changed")
        native_lower = float(native_price[index])
        excess = proposal.lower_bound - native_lower
        if not np.isfinite(native_lower) or native_lower < 0 \
                or not np.isfinite(excess) \
                or not strengthened_proposal_bound and excess > tolerance:
            raise CertifiedComponentSearchError("packed component/native bound relation differs")
        if not strengthened_proposal_bound:
            maximum_excess = max(maximum_excess, excess)
        if native_lower > completion_threshold:
            native_pruned += 1
            continue
        exact_indexes.append(index)

    exact_prices = batch_exact_price_distances(
        query_representation,
        [representations[index] for index in exact_indexes],
        rigid_price[np.asarray(exact_indexes, dtype=int)],
    )
    for index, exact_price_value in zip(exact_indexes, exact_prices, strict=True):
        proposal = proposals[index]
        episode_key = EpisodeKey(
            key, pd.Timestamp(proposal.cutoff_ns), query.key.lookback,
            query.key.representation_version,
        )
        native_lower = float(native_price[index])
        exact_price = float(exact_price_value)
        if not np.isfinite(exact_price) \
                or exact_price + tolerance < native_lower \
                or strengthened_proposal_bound \
                and exact_price + tolerance < proposal.lower_bound:
            raise CertifiedComponentSearchError("exact component violates native bound")
        if strengthened_proposal_bound:
            maximum_excess = max(
                maximum_excess, proposal.lower_bound - exact_price,
            )
        output.append(_CompactComponentScore(AnalogueMatch(
            episode_key, exact_price, {PRICE_COMPONENT: exact_price}, [],
            proposal.quality_tier, (),
        ), key))
    return output, native_pruned, maximum_excess


def _score(
    proposals: Iterable[BoundProposal], *, query: Episode,
    query_representation: Representation, source: OHLCVSource,
    request: SearchQuery, store_dataset_id: str, manifest: dict[str, object],
    completion_threshold: float, tolerance: float, workers: int,
    benchmark: pd.DataFrame,
    strengthened_proposal_bound: bool = False,
    prepared_cache: dict[str, _PreparedComponentSymbol] | None = None,
) -> tuple[list[_CompactComponentScore], int, float]:
    grouped: dict[str, list[BoundProposal]] = {}
    for proposal in proposals:
        grouped.setdefault(proposal.symbol, []).append(proposal)
    provenance = manifest.get("provenance")
    if type(provenance) is not dict or type(provenance.get("source_prefixes")) is not dict \
            or type(provenance.get("benchmark_prefix")) is not dict:
        raise CertifiedComponentSearchError("packed prefix provenance is incomplete")
    prefixes = provenance["source_prefixes"]
    benchmark_prefix = provenance["benchmark_prefix"]
    if not _prefix_matches(benchmark, benchmark_prefix):
        raise CertifiedComponentSearchError("packed benchmark causal prefix is stale")
    maximum_cutoff = pd.Timestamp(str(benchmark_prefix["requested_cutoff"]))
    cache = {} if prepared_cache is None else prepared_cache

    def one(item: tuple[str, list[BoundProposal]]):
        symbol, rows = item
        expected = prefixes.get(symbol)
        if type(expected) is not dict:
            raise CertifiedComponentSearchError(f"missing packed prefix: {symbol}")
        prepared = cache.get(symbol)
        if prepared is None:
            key = InstrumentKey(store_dataset_id, symbol)
            bars = source.load(key)
            if not _prefix_matches(bars, expected):
                raise CertifiedComponentSearchError(
                    f"packed stock causal prefix is stale: {symbol}"
                )
            latest = min(
                latest_eligible_cutoff(query, request.minimum_history_gap_bars),
                maximum_cutoff,
            )
            frame = bars[bars.timestamp <= latest].reset_index(drop=True)
            prepared = _PreparedComponentSymbol(
                frame,
                np.ascontiguousarray(
                    frame["timestamp"].to_numpy(dtype="datetime64[ns]").view(np.int64)
                ),
            )
            cache[symbol] = prepared
        return _score_group(
            symbol, tuple(rows), query=query, query_representation=query_representation,
            request=request, benchmark=benchmark, store_dataset_id=store_dataset_id,
            prepared=prepared,
            completion_threshold=completion_threshold, tolerance=tolerance,
            strengthened_proposal_bound=strengthened_proposal_bound,
        )

    items = sorted(grouped.items())
    if workers == 1:
        results = map(one, items)
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        results = executor.map(one, items)
    output: list[_CompactComponentScore] = []
    native_pruned = 0
    maximum_excess = 0.0
    try:
        for values, pruned, excess in results:
            output.extend(values)
            native_pruned += pruned
            maximum_excess = max(maximum_excess, excess)
    finally:
        if workers != 1:
            executor.shutdown(wait=True)
    return output, native_pruned, maximum_excess


def certified_component_search(
    query: Episode,
    source: OHLCVSource,
    request: SearchQuery,
    store_root: Path,
    generation_id: str,
    *,
    store_dataset_id: str,
    component: str = PRICE_COMPONENT,
    initial_frontier_rows: int = 1_000,
    maximum_frontier_rows: int = 16_384,
    seed_rows: int = 512,
    block_rows: int = 4_096,
    proposal_threads: int = 8,
    workers: int = 1,
    tolerance: float = 1e-12,
    verify_content: bool = True,
    precomputed_proposal: BoundProposalReport | None = None,
) -> CertifiedComponentSearchResult:
    if component != PRICE_COMPONENT:
        raise CertifiedComponentSearchError("only price component search is implemented")
    if request.max_per_instrument != 1 or request.deduplicate_overlaps is not True:
        raise CertifiedComponentSearchError("component search requires one deduplicated row per symbol")
    if workers < 1 or proposal_threads < 1 or not np.isfinite(tolerance) or tolerance < 0:
        raise CertifiedComponentSearchError("component worker/tolerance policy differs")
    if initial_frontier_rows < request.top_k or seed_rows < request.top_k \
            or seed_rows > initial_frontier_rows \
            or maximum_frontier_rows < initial_frontier_rows:
        raise CertifiedComponentSearchError("component frontier policy differs")
    if store_dataset_id != query.key.instrument.dataset_id and not request.cross_dataset:
        raise CertifiedComponentSearchError("cross-dataset component search is not authorized")
    started = perf_counter()
    loaded = load_packed_generation(
        store_root, generation_id, verify_content=verify_content,
        validate_records=False,
    )
    benchmark = source.load_benchmark()
    if benchmark is None:
        raise CertifiedComponentSearchError("component search requires benchmark context")
    query_representation = represent(query)
    input_state = {
        "query_stock_prefix": asdict(causal_prefix_digest(
            source.load(query.key.instrument), query.key.cutoff,
        )),
        "query_benchmark_prefix": asdict(causal_prefix_digest(benchmark, query.key.cutoff)),
        "request": {
            "search_datasets": request.search_datasets,
            "quality_tiers": request.quality_tiers, "top_k": request.top_k,
            "cross_dataset": request.cross_dataset,
            "deduplicate_overlaps": request.deduplicate_overlaps,
            "max_per_instrument": request.max_per_instrument,
            "minimum_history_gap_bars": request.minimum_history_gap_bars,
        },
        "component": component,
        "packed_provenance_digest": loaded.manifest["provenance_digest"],
        "query_representation_digest": representation_input_digest(query_representation),
    }
    input_digest = stable_hash(input_state)
    packed_query = PackedBoundQuery(
        query.key.id, query.key.instrument.source_symbol,
        int(pd.Timestamp(query.bars.timestamp.iloc[0]).value),
        int(latest_eligible_cutoff(query, request.minimum_history_gap_bars).value),
        query_representation, request.quality_tiers,
    )
    if precomputed_proposal is None:
        precomputed_proposal = scan_packed_component_bound_proposals_threaded(
            store_root, generation_id, packed_query, component=component,
            quota=maximum_frontier_rows + 1, block_rows=block_rows,
            threads=proposal_threads, verify_content=False,
            expected_provenance_digest=str(loaded.manifest["provenance_digest"]),
        )
    contract = packed_component_search_contract(component)
    candidates = precomputed_proposal.candidates
    proposal_state = {
        "schema_version": COMPONENT_SEARCH_SCHEMA_VERSION,
        "contract_digest": contract["digest"], "component": component,
        "generation_id": generation_id, "query_episode_id": query.key.id,
        "rows_scanned": precomputed_proposal.rows_scanned,
        "eligible_rows": precomputed_proposal.eligible_rows,
        "eligible_main_rows": precomputed_proposal.eligible_main_rows,
        "eligible_overflow_rows": precomputed_proposal.eligible_overflow_rows,
        "route_counts": precomputed_proposal.route_counts,
        "route_quotas": precomputed_proposal.route_quotas,
        "candidate_digest": precomputed_proposal.candidate_digest,
        "real_forward_outcomes_accessed": False,
        "input_digest": _packed_query_input_digest(packed_query),
    }
    if not all((
        precomputed_proposal.schema_version == COMPONENT_SEARCH_SCHEMA_VERSION,
        precomputed_proposal.contract_digest == contract["digest"],
        precomputed_proposal.input_digest == proposal_state["input_digest"],
        precomputed_proposal.generation_id == generation_id,
        precomputed_proposal.query_episode_id == query.key.id,
        precomputed_proposal.route_quotas == {component: maximum_frontier_rows + 1},
        precomputed_proposal.route_counts == {component: len(candidates)},
        len(candidates) == min(maximum_frontier_rows + 1,
                               precomputed_proposal.eligible_rows),
        list(candidates) == sorted(candidates,
                                   key=lambda row: (row.lower_bound, row.episode_id)),
        all(row.routes == (component,) and np.isfinite(row.lower_bound)
            and row.lower_bound >= 0 for row in candidates),
        precomputed_proposal.candidate_digest == bound_proposal_candidate_digest(candidates),
        precomputed_proposal.result_digest == stable_hash(proposal_state),
    )):
        raise CertifiedComponentSearchError("precomputed component proposal differs")
    scored: dict[str, _CompactComponentScore] = {}
    evaluated: set[str] = set()
    native_pruned = 0
    exact_evaluated = 0
    maximum_excess = 0.0
    prepared_cache: dict[str, _PreparedComponentSymbol] = {}
    rounds: list[ComponentCompletionRound] = []
    frontier_rows = initial_frontier_rows
    threshold = float("inf")
    while True:
        frontier = candidates[:frontier_rows]
        pending = [row for row in frontier if row.episode_id not in evaluated]
        if not evaluated:
            pending = pending[:seed_rows]
        while pending:
            values, pruned, excess = _score(
                pending, query=query, query_representation=query_representation,
                source=source, request=request, store_dataset_id=store_dataset_id,
                manifest=loaded.manifest, completion_threshold=threshold,
                tolerance=tolerance, workers=workers, benchmark=benchmark,
                prepared_cache=prepared_cache,
            )
            evaluated.update(row.episode_id for row in pending)
            scored.update({row.match.episode_key.id: row for row in values})
            native_pruned += pruned
            exact_evaluated += len(values)
            maximum_excess = max(maximum_excess, excess)
            selected = _select(scored.values(), request.top_k)
            threshold = (selected[-1].total_distance
                         if len(selected) == request.top_k else float("inf"))
            pending = [row for row in frontier if row.episode_id not in evaluated]
        selected = _select(scored.values(), request.top_k)
        threshold = (selected[-1].total_distance
                     if len(selected) == request.top_k else float("inf"))
        next_lower = (candidates[frontier_rows].lower_bound
                      if len(candidates) > frontier_rows else None)
        certified = len(selected) == request.top_k and (
            next_lower is None or next_lower > threshold
        )
        rounds.append(ComponentCompletionRound(
            min(frontier_rows, len(candidates)), len(evaluated), exact_evaluated,
            len(selected), threshold, next_lower, certified,
        ))
        if certified:
            break
        if frontier_rows >= maximum_frontier_rows \
                or frontier_rows >= precomputed_proposal.eligible_rows:
            raise ComponentFrontierOverflow(
                frontier_rows=frontier_rows,
                eligible_candidates=precomputed_proposal.eligible_rows,
                exact_evaluated=exact_evaluated, threshold=threshold,
                next_lower_bound=next_lower,
            )
        frontier_rows = min(maximum_frontier_rows, frontier_rows * 2,
                            precomputed_proposal.eligible_rows)
    packed_pruned = precomputed_proposal.eligible_rows - len(evaluated)
    certificate_contract = certified_component_search_contract(component)
    deterministic = {
        "schema_version": COMPONENT_CERTIFICATE_SCHEMA,
        "contract_digest": certificate_contract["digest"], "component": component,
        "generation_id": generation_id, "query_episode_id": query.key.id,
        "input_digest": input_digest,
        "proposal_digest": precomputed_proposal.result_digest,
        "eligible_candidates": precomputed_proposal.eligible_rows,
        "native_bound_evaluated": len(evaluated),
        "exact_evaluated": exact_evaluated,
        "native_bound_pruned": native_pruned,
        "packed_bound_pruned": packed_pruned,
        "stop_threshold": threshold, "next_lower_bound": next_lower,
        "maximum_quantized_bound_excess": maximum_excess,
        "rounds": [asdict(row) for row in rounds],
        "matches": [{
            "episode_id": row.episode_key.id,
            "distance_hex": row.total_distance.hex(),
        } for row in selected],
        "outcomes_or_labels_used": False,
    }
    result_digest = stable_hash(deterministic)
    certificate = ComponentSearchCertificate(
        COMPONENT_CERTIFICATE_SCHEMA, str(certificate_contract["digest"]), component,
        generation_id, query.key.id, input_digest,
        precomputed_proposal.result_digest, precomputed_proposal.eligible_rows,
        len(evaluated), exact_evaluated, native_pruned, packed_pruned,
        threshold, next_lower, maximum_excess, tuple(rounds), result_digest,
        perf_counter() - started,
    )
    return CertifiedComponentSearchResult(tuple(selected), certificate)
