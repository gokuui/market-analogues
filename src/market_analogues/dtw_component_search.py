from __future__ import annotations

from dataclasses import asdict, dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import numba

from .dtw_sample_store import dtw_sample_lower_bounds, load_dtw_sample_generation
from .component_search import (
    ComponentFrontierOverflow,
    CertifiedComponentSearchError,
    _score,
    _select,
)
from .adapters import OHLCVSource
from .causal_prefix import causal_prefix_digest
from .packed_bound_search import (
    BoundProposal,
    PackedBoundQuery,
    _current_rss_mb,
    _ENTRY_DTYPE,
    _eligible_mask,
    _entries,
    _finalize,
    _packed_query_input_digest,
    _stable_bounded,
    packed_component_search_contract,
)
from .packed_bound_store import (
    load_packed_generation, packed_branch_aware_price_lower_bounds,
)
from .representation import represent, representation_input_digest
from .search import latest_eligible_cutoff
from .types import AnalogueMatch, Episode, SearchQuery, stable_hash


SCHEMA_VERSION = "m04r-dtw-component-bound-proposal-v1"


class DtwComponentSearchError(ValueError):
    pass


@dataclass(frozen=True)
class DtwComponentProposalReport:
    schema_version: str
    contract_digest: str
    packed_generation_id: str
    dtw_generation_id: str
    query_episode_id: str
    input_digest: str
    candidates: tuple[BoundProposal, ...]
    rows_scanned: int
    eligible_rows: int
    eligible_main_rows: int
    eligible_overflow_rows: int
    quota: int
    block_rows: int
    block_order: str
    kernel_threads: int
    elapsed_seconds: float
    peak_rss_mb: float
    candidate_digest: str
    result_digest: str


@dataclass(frozen=True)
class DtwComponentCompletionRound:
    frontier_rows: int
    bound_evaluated: int
    exact_evaluated: int
    selected_rows: int
    constrained_threshold: float
    next_lower_bound: float | None
    certified: bool


@dataclass(frozen=True)
class DtwComponentCertificate:
    schema_version: str
    contract_digest: str
    packed_generation_id: str
    dtw_generation_id: str
    query_episode_id: str
    input_digest: str
    proposal_digest: str
    eligible_candidates: int
    bound_evaluated: int
    exact_evaluated: int
    bound_pruned: int
    stop_threshold: float
    next_lower_bound: float | None
    maximum_bound_excess: float
    rounds: tuple[DtwComponentCompletionRound, ...]
    result_digest: str
    elapsed_seconds: float


@dataclass(frozen=True)
class CertifiedDtwComponentResult:
    matches: tuple[AnalogueMatch, ...]
    certificate: DtwComponentCertificate


def dtw_component_search_contract() -> dict[str, Any]:
    state = {
        "schema_version": SCHEMA_VERSION,
        "component": "price",
        "packed_component_contract_digest": packed_component_search_contract(
            "price"
        )["digest"],
        "formula": (
            "branch-aware packed price component (0.55 rigid contribution) plus "
            "0.45 times the aligned outward-corrected DTW interval lower bound"
        ),
        "alignment": (
            "main and overflow physical indexes must match the packed generation "
            "bound by the DTW generation manifest"
        ),
        "eligibility": (
            "quality tier and latest eligible cutoff before scoring; query ID and "
            "overlapping same-symbol windows excluded"
        ),
        "overflow": "packed overflow rows retain universal zero lower bound",
        "ranking": "ascending (combined price lower bound, episode ID)",
        "outcomes_or_labels_used": False,
    }
    return {**state, "digest": stable_hash(state)}


def certified_dtw_component_search_contract() -> dict[str, Any]:
    state = {
        "schema_version": "certified-dtw-component-search-v1",
        "component": "price",
        "proposal_contract_digest": dtw_component_search_contract()["digest"],
        "distance": (
            "distance-v1 price component: 0.55 rigid price-channel distance plus "
            "0.45 exact banded multivariate DTW"
        ),
        "completion": (
            "exactly score the combined-bound frontier and stop only when the next "
            "combined bound is strictly above the constrained top-20 threshold"
        ),
        "selection": "ascending (exact price distance, episode ID), one row per symbol",
        "outcomes_or_labels_used": False,
    }
    return {**state, "digest": stable_hash(state)}


def _entries_at_positions(
    records: np.ndarray, positions: np.ndarray, scores: np.ndarray,
) -> np.ndarray:
    output = np.empty(len(positions), dtype=_ENTRY_DTYPE)
    for name in ("episode_id", "cutoff_ns", "symbol_id", "quality_tier"):
        output[name] = records[name][positions]
    output["total"] = scores
    output["route_score"] = scores
    output["overflow"] = False
    return output


def _stable_main_positions(
    records: np.ndarray, scores: np.ndarray, quota: int,
) -> np.ndarray:
    positions = np.flatnonzero(np.isfinite(scores))
    if len(positions) <= quota:
        return positions
    values = scores[positions]
    boundary = float(np.partition(values, quota - 1)[quota - 1])
    lower = positions[values < boundary]
    tied = positions[values == boundary]
    needed = quota - len(lower)
    if needed < 0 or needed > len(tied):
        raise DtwComponentSearchError("combined stable partition accounting differs")
    identifiers = np.frombuffer(
        np.ascontiguousarray(records["episode_id"][tied]).tobytes(),
        dtype=np.dtype([("high", ">u8"), ("low", ">u4")]),
    )
    order = np.lexsort((identifiers["low"], identifiers["high"]))
    return np.concatenate((lower, tied[order[:needed]]))


def scan_dtw_component_bound_proposals(
    packed_root: Path,
    packed_generation_id: str,
    dtw_root: Path,
    dtw_generation_id: str,
    query: PackedBoundQuery,
    *,
    quota: int,
    block_rows: int = 4096,
    block_order: str = "forward",
    kernel_threads: int | None = None,
    verify_content: bool = True,
    expected_packed_provenance_digest: str | None = None,
) -> DtwComponentProposalReport:
    if type(quota) is not int or isinstance(quota, bool) or quota < 1:
        raise DtwComponentSearchError("combined component quota must be positive")
    if type(block_rows) is not int or isinstance(block_rows, bool) or block_rows < 1:
        raise DtwComponentSearchError("combined component block size must be positive")
    if block_order not in {"forward", "reverse"}:
        raise DtwComponentSearchError("combined component order must be forward or reverse")
    available_threads = int(numba.config.NUMBA_NUM_THREADS)
    threads = min(8, available_threads) if kernel_threads is None else kernel_threads
    if type(threads) is not int or isinstance(threads, bool) \
            or threads < 1 or threads > available_threads:
        raise DtwComponentSearchError("combined component kernel threads differ")
    packed = load_packed_generation(
        packed_root, packed_generation_id,
        expected_provenance_digest=expected_packed_provenance_digest,
        verify_content=verify_content, validate_records=False,
    )
    dtw = load_dtw_sample_generation(
        dtw_root, dtw_generation_id, packed_manifest=packed.manifest,
        verify_content=verify_content, validate_records=verify_content,
    )
    if len(packed.rows) != len(dtw.rows) \
            or len(packed.overflow) != len(dtw.overflow):
        raise DtwComponentSearchError("combined component store alignment differs")
    symbol_id = (
        packed.symbols.index(query.symbol) if query.symbol in packed.symbols else None
    )
    starts = list(range(0, len(packed.rows), block_rows))
    if block_order == "reverse":
        starts.reverse()
    scores = np.full(len(packed.rows), np.inf, dtype=np.float64)
    eligible_main = 0
    started = perf_counter()
    peak = _current_rss_mb()
    packed_threads = max(1, threads // 4)
    dtw_threads = threads - packed_threads
    parallel = dtw_threads >= 1 and threads >= 2

    def packed_score(block: np.ndarray) -> np.ndarray:
        numba.set_num_threads(packed_threads if parallel else threads)
        return np.asarray(
            packed_branch_aware_price_lower_bounds(query.representation, block),
            dtype=np.float64,
        )

    def dtw_score(block: np.ndarray) -> np.ndarray:
        numba.set_num_threads(dtw_threads if parallel else threads)
        return dtw_sample_lower_bounds(query.representation, block)

    executor = ThreadPoolExecutor(max_workers=2) if parallel else None
    try:
        for first in starts:
            last = min(first + block_rows, len(packed.rows))
            packed_block = np.asarray(packed.rows[first:last])
            dtw_block_records = np.asarray(dtw.rows[first:last])
            mask = _eligible_mask(packed_block, query, symbol_id)
            eligible_main += int(np.count_nonzero(mask))
            if not np.any(mask):
                continue
            if executor is None:
                rigid_all = packed_score(packed_block)
                dtw_all = dtw_score(dtw_block_records)
            else:
                packed_future = executor.submit(packed_score, packed_block)
                dtw_future = executor.submit(dtw_score, dtw_block_records)
                rigid_all = packed_future.result()
                dtw_all = dtw_future.result()
            combined = rigid_all + 0.45 * dtw_all
            if not np.isfinite(combined).all() or np.any(combined < rigid_all) \
                    or np.any(combined < 0):
                raise DtwComponentSearchError("combined component bound is invalid")
            scores[first:last][mask] = combined[mask]
            peak = max(peak, _current_rss_mb())
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    selected_positions = _stable_main_positions(packed.rows, scores, quota)
    heap = _entries_at_positions(
        packed.rows, selected_positions, scores[selected_positions],
    )
    overflow = np.asarray(packed.overflow)
    selected_overflow = overflow[_eligible_mask(overflow, query, symbol_id)]
    eligible_overflow = len(selected_overflow)
    if eligible_overflow:
        zeros = np.zeros(eligible_overflow, dtype=np.float64)
        heap = _stable_bounded(
            heap,
            _entries(selected_overflow, zeros, zeros, overflow=True),
            quota,
        )
    candidates, route_counts, candidate_digest = _finalize(
        {"price": heap}, packed.symbols,
    )
    if route_counts != {"price": len(candidates)}:
        raise DtwComponentSearchError("combined component route accounting differs")
    contract = dtw_component_search_contract()
    input_digest = stable_hash({
        "packed_query_input_digest": _packed_query_input_digest(query),
        "packed_generation_id": packed.generation_id,
        "packed_provenance_digest": packed.manifest["provenance_digest"],
        "dtw_generation_id": dtw.generation_id,
        "dtw_provenance_digest": dtw.manifest["provenance_digest"],
        "contract_digest": contract["digest"],
    })
    deterministic = {
        "schema_version": SCHEMA_VERSION,
        "contract_digest": contract["digest"],
        "packed_generation_id": packed.generation_id,
        "dtw_generation_id": dtw.generation_id,
        "query_episode_id": query.episode_id,
        "input_digest": input_digest,
        "rows_scanned": len(packed.rows) + len(packed.overflow),
        "eligible_rows": eligible_main + eligible_overflow,
        "eligible_main_rows": eligible_main,
        "eligible_overflow_rows": eligible_overflow,
        "quota": quota,
        "candidate_digest": candidate_digest,
        "outcomes_or_labels_used": False,
    }
    return DtwComponentProposalReport(
        SCHEMA_VERSION, str(contract["digest"]), packed.generation_id,
        dtw.generation_id, query.episode_id, input_digest, candidates,
        deterministic["rows_scanned"], deterministic["eligible_rows"],
        eligible_main, eligible_overflow, quota, block_rows, block_order,
        threads, float(perf_counter() - started), peak, candidate_digest,
        stable_hash(deterministic),
    )


def certified_dtw_component_search(
    query: Episode,
    source: OHLCVSource,
    request: SearchQuery,
    packed_root: Path,
    packed_generation_id: str,
    dtw_root: Path,
    dtw_generation_id: str,
    *,
    store_dataset_id: str,
    initial_frontier_rows: int = 1000,
    maximum_frontier_rows: int = 16_384,
    seed_rows: int = 512,
    block_rows: int = 4096,
    workers: int = 1,
    tolerance: float = 1e-12,
    verify_content: bool = True,
    precomputed_proposal: DtwComponentProposalReport | None = None,
) -> CertifiedDtwComponentResult:
    if request.max_per_instrument != 1 or request.deduplicate_overlaps is not True:
        raise DtwComponentSearchError("combined component search requires one row per symbol")
    if initial_frontier_rows < request.top_k or seed_rows < request.top_k \
            or seed_rows > initial_frontier_rows \
            or maximum_frontier_rows < initial_frontier_rows or workers < 1 \
            or not np.isfinite(tolerance) or tolerance < 0:
        raise DtwComponentSearchError("combined component completion policy differs")
    if store_dataset_id != query.key.instrument.dataset_id and not request.cross_dataset:
        raise DtwComponentSearchError("cross-dataset combined search is not authorized")
    started = perf_counter()
    packed = load_packed_generation(
        packed_root, packed_generation_id, verify_content=verify_content,
        validate_records=False,
    )
    dtw = load_dtw_sample_generation(
        dtw_root, dtw_generation_id, packed_manifest=packed.manifest,
        verify_content=verify_content, validate_records=verify_content,
    )
    benchmark = source.load_benchmark()
    if benchmark is None:
        raise DtwComponentSearchError("combined component search requires benchmark")
    query_representation = represent(query)
    packed_query = PackedBoundQuery(
        query.key.id, query.key.instrument.source_symbol,
        int(query.bars.timestamp.iloc[0].value),
        int(latest_eligible_cutoff(
            query, request.minimum_history_gap_bars,
        ).value),
        query_representation, request.quality_tiers,
    )
    if precomputed_proposal is None:
        precomputed_proposal = scan_dtw_component_bound_proposals(
            packed_root, packed_generation_id, dtw_root, dtw_generation_id,
            packed_query, quota=maximum_frontier_rows + 1,
            block_rows=block_rows, verify_content=False,
            expected_packed_provenance_digest=str(
                packed.manifest["provenance_digest"]
            ),
        )
    candidates = precomputed_proposal.candidates
    if not all((
        precomputed_proposal.schema_version == SCHEMA_VERSION,
        precomputed_proposal.contract_digest
        == dtw_component_search_contract()["digest"],
        precomputed_proposal.packed_generation_id == packed.generation_id,
        precomputed_proposal.dtw_generation_id == dtw.generation_id,
        precomputed_proposal.query_episode_id == query.key.id,
        precomputed_proposal.quota == maximum_frontier_rows + 1,
        len(candidates) == min(
            maximum_frontier_rows + 1, precomputed_proposal.eligible_rows,
        ),
        list(candidates) == sorted(
            candidates, key=lambda row: (row.lower_bound, row.episode_id),
        ),
        all(row.routes == ("price",) and np.isfinite(row.lower_bound)
            and row.lower_bound >= 0 for row in candidates),
    )):
        raise DtwComponentSearchError("combined component proposal differs")
    input_digest = stable_hash({
        "query_stock_prefix": asdict(causal_prefix_digest(
            source.load(query.key.instrument), query.key.cutoff,
        )),
        "query_representation_digest": representation_input_digest(
            query_representation
        ),
        "request": asdict(request),
        "packed_generation_id": packed.generation_id,
        "packed_provenance_digest": packed.manifest["provenance_digest"],
        "dtw_generation_id": dtw.generation_id,
        "dtw_provenance_digest": dtw.manifest["provenance_digest"],
    })
    scored: dict[str, Any] = {}
    evaluated: set[str] = set()
    exact_evaluated = 0
    maximum_excess = 0.0
    prepared_cache: dict[str, Any] = {}
    rounds = []
    frontier_rows = initial_frontier_rows
    threshold = float("inf")
    while True:
        frontier = candidates[:frontier_rows]
        pending = [row for row in frontier if row.episode_id not in evaluated]
        if not evaluated:
            pending = pending[:seed_rows]
        elif np.isfinite(threshold):
            pending = [row for row in pending if row.lower_bound <= threshold]
        while pending:
            try:
                values, _native_pruned, excess = _score(
                    pending, query=query,
                    query_representation=query_representation, source=source,
                    request=request, store_dataset_id=store_dataset_id,
                    manifest=packed.manifest, completion_threshold=threshold,
                    tolerance=tolerance, workers=workers, benchmark=benchmark,
                    strengthened_proposal_bound=True,
                    prepared_cache=prepared_cache,
                )
            except CertifiedComponentSearchError as exc:
                raise DtwComponentSearchError(
                    "combined exact completion failed"
                ) from exc
            evaluated.update(row.episode_id for row in pending)
            scored.update({row.match.episode_key.id: row for row in values})
            exact_evaluated += len(values)
            maximum_excess = max(maximum_excess, excess)
            selected = _select(scored.values(), request.top_k)
            threshold = (
                selected[-1].total_distance
                if len(selected) == request.top_k else float("inf")
            )
            pending = [
                row for row in frontier
                if row.episode_id not in evaluated and row.lower_bound <= threshold
            ]
        selected = _select(scored.values(), request.top_k)
        threshold = (
            selected[-1].total_distance
            if len(selected) == request.top_k else float("inf")
        )
        next_lower = (
            candidates[frontier_rows].lower_bound
            if len(candidates) > frontier_rows else None
        )
        certified = len(selected) == request.top_k and (
            next_lower is None or next_lower > threshold
        )
        rounds.append(DtwComponentCompletionRound(
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
        frontier_rows = min(
            maximum_frontier_rows, frontier_rows * 2,
            precomputed_proposal.eligible_rows,
        )
    contract = certified_dtw_component_search_contract()
    deterministic = {
        "schema_version": contract["schema_version"],
        "contract_digest": contract["digest"],
        "packed_generation_id": packed.generation_id,
        "dtw_generation_id": dtw.generation_id,
        "query_episode_id": query.key.id,
        "input_digest": input_digest,
        "proposal_digest": precomputed_proposal.result_digest,
        "eligible_candidates": precomputed_proposal.eligible_rows,
        "bound_evaluated": len(evaluated),
        "exact_evaluated": exact_evaluated,
        "bound_pruned": precomputed_proposal.eligible_rows - len(evaluated),
        "stop_threshold": threshold, "next_lower_bound": next_lower,
        "maximum_bound_excess": maximum_excess,
        "rounds": [asdict(row) for row in rounds],
        "matches": [{
            "episode_id": row.episode_key.id,
            "distance_hex": row.total_distance.hex(),
        } for row in selected],
        "outcomes_or_labels_used": False,
    }
    result_digest = stable_hash(deterministic)
    certificate = DtwComponentCertificate(
        str(contract["schema_version"]), str(contract["digest"]),
        packed.generation_id, dtw.generation_id, query.key.id, input_digest,
        precomputed_proposal.result_digest, precomputed_proposal.eligible_rows,
        len(evaluated), exact_evaluated,
        precomputed_proposal.eligible_rows - len(evaluated), threshold,
        next_lower, maximum_excess, tuple(rounds), result_digest,
        float(perf_counter() - started),
    )
    return CertifiedDtwComponentResult(tuple(selected), certificate)
