from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import Future, ThreadPoolExecutor
from hashlib import sha256
import os
from pathlib import Path
import resource
from time import perf_counter
from typing import AbstractSet, Any, Callable, Iterable, Mapping

import numba
import numpy as np

from .packed_bound_store import (
    PACK_DTYPE, TIER_CODES, TIER_NAMES, load_packed_generation, packed_lower_bounds,
    packed_branch_aware_lower_bounds,
    packed_bound_store_contract, prepare_packed_lower_bound_records,
    prepared_packed_branch_aware_lower_bounds, prepared_packed_lower_bounds,
)
from .quantized_bound import branch_aware_quantized_bound_contract
from .representation import Representation
from .types import stable_hash


SEARCH_SCHEMA_VERSION = "m04r-global-bound-proposal-v1"
BRANCH_AWARE_SEARCH_SCHEMA_VERSION = "m04r-global-bound-proposal-v2"
COMPONENT_SEARCH_SCHEMA_VERSION = "m04r-global-component-bound-proposal-v1"
BATCH_SEARCH_SCHEMA_VERSION = "m04r-global-bound-proposal-batch-v1"
BATCH_BRANCH_AWARE_SEARCH_SCHEMA_VERSION = "m04r-global-bound-proposal-batch-v2"
THRESHOLD_SCAN_SCHEMA_VERSION = "m04r-packed-bound-threshold-scan-v1"
BRANCH_AWARE_THRESHOLD_SCAN_SCHEMA_VERSION = "m04r-packed-bound-threshold-scan-v2"
COMPONENT_THRESHOLD_SCAN_SCHEMA_VERSION = "m04r-packed-component-threshold-scan-v1"
DEFAULT_ROUTE_QUOTAS: dict[str, int] = {
    # The composite route is the only authority-certified recall route.  The
    # component routes are additive proposal seeds and never replace it.
    "composite": 1_000,
    "stage": 64,
    "price": 64,
    "candle_volatility": 64,
    "volume_shock": 64,
    "market_context": 64,
    "structural": 64,
    "coarse": 64,
}


class PackedBoundSearchError(ValueError):
    pass


@dataclass(frozen=True)
class PackedBoundQuery:
    episode_id: str
    symbol: str
    query_start_ns: int
    latest_eligible_ns: int
    representation: Representation
    quality_tiers: tuple[str, ...] = ("A", "B")

    def __post_init__(self) -> None:
        if len(self.episode_id) != 24 or self.episode_id.lower() != self.episode_id:
            raise PackedBoundSearchError("query episode ID must contain 24 lowercase hex characters")
        try:
            bytes.fromhex(self.episode_id)
        except ValueError as exc:
            raise PackedBoundSearchError("query episode ID is not hexadecimal") from exc
        if self.query_start_ns > self.latest_eligible_ns:
            # This is normal only when a query is shorter than its required gap,
            # which cannot occur for the frozen 252/60 contract.
            raise PackedBoundSearchError("query start exceeds latest eligible cutoff")
        if (
            not self.quality_tiers
            or len(set(self.quality_tiers)) != len(self.quality_tiers)
            or any(value not in TIER_CODES for value in self.quality_tiers)
        ):
            raise PackedBoundSearchError("quality tiers must be a non-empty A/B subset")


@dataclass(frozen=True)
class BoundProposal:
    episode_id: str
    symbol: str
    cutoff_ns: int
    quality_tier: str
    lower_bound: float
    routes: tuple[str, ...]
    overflow_fallback: bool


@dataclass(frozen=True)
class BoundProposalReport:
    schema_version: str
    generation_id: str
    query_episode_id: str
    candidates: tuple[BoundProposal, ...]
    rows_scanned: int
    eligible_rows: int
    eligible_main_rows: int
    eligible_overflow_rows: int
    route_counts: Mapping[str, int]
    route_quotas: Mapping[str, int]
    block_rows: int
    block_order: str
    elapsed_seconds: float
    peak_rss_mb: float
    candidate_digest: str
    result_digest: str
    contract_digest: str | None = None
    input_digest: str | None = None


@dataclass(frozen=True)
class BoundProposalBatchReport:
    schema_version: str
    generation_id: str
    query_episode_ids: tuple[str, ...]
    reports: tuple[BoundProposalReport, ...]
    physical_rows_scanned: int
    logical_rows_evaluated: int
    block_rows: int
    block_order: str
    elapsed_seconds: float
    peak_rss_mb: float
    result_digest: str


def packed_component_search_contract(component: str) -> dict[str, Any]:
    """Contract for a route-specific safe-bound frontier.

    Unlike the additive multi-route proposal report, ``lower_bound`` in this
    report is the named component bound itself.  It may therefore certify an
    exact component search when paired with component-specific completion.
    """
    allowed = {
        "coarse", "stage", "price", "candle_volatility", "volume_shock",
        "market_context", "structural",
    }
    if component not in allowed:
        raise PackedBoundSearchError("component bound route is unsupported")
    state = {
        "schema_version": COMPONENT_SEARCH_SCHEMA_VERSION,
        "component": component,
        "eligibility": (
            "quality tier before scoring; cutoff <= latest eligible cutoff; "
            "same-symbol windows intersecting the query and query ID excluded"
        ),
        "ranking": "ascending (named branch-aware component lower bound, episode ID)",
        "overflow": "unquantizable rows receive zero and cannot be omitted",
        "lower_bound_field": "named component bound, not composite total",
        "outcomes_or_labels_used": False,
    }
    return {**state, "digest": stable_hash(state)}


def packed_component_threshold_scan_contract(component: str) -> dict[str, Any]:
    proposal = packed_component_search_contract(component)
    state = {
        "schema_version": COMPONENT_THRESHOLD_SCAN_SCHEMA_VERSION,
        "component": component,
        "component_proposal_contract_digest": proposal["digest"],
        "admission": "named safe component bound in (lower exclusive, upper inclusive]",
        "ordering": "streaming physical order; admitted set digest is order-independent",
        "overflow": "zero-bound fallback admitted only when lower is absent",
        "outcomes_or_labels_used": False,
    }
    return {**state, "digest": stable_hash(state)}


@dataclass(frozen=True)
class BoundThresholdScanReport:
    """Evidence for one streamed inclusive packed-bound admission band."""

    schema_version: str
    contract_digest: str
    generation_id: str
    query_episode_id: str
    input_digest: str
    exclusions_digest: str
    lower_exclusive: float | None
    upper_inclusive: float
    rows_scanned: int
    eligible_rows: int
    eligible_main_rows: int
    eligible_overflow_rows: int
    excluded_eligible_rows: int
    admitted_rows: int
    minimum_above_upper: float | None
    admitted_set_digest: str
    block_rows: int
    block_order: str
    elapsed_seconds: float
    peak_rss_mb: float
    result_digest: str


_ENTRY_DTYPE = np.dtype([
    ("episode_id", "V12"),
    ("cutoff_ns", "<i8"),
    ("symbol_id", "<u4"),
    ("quality_tier", "u1"),
    ("total", "<f8"),
    ("route_score", "<f8"),
    ("overflow", "?"),
])


def packed_bound_search_contract(*, branch_aware: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": (
            BRANCH_AWARE_SEARCH_SCHEMA_VERSION
            if branch_aware else SEARCH_SCHEMA_VERSION
        ),
        "eligibility": (
            "quality tier before scoring; cutoff <= latest eligible cutoff; "
            "same-symbol windows intersecting the query are excluded; query ID excluded"
        ),
        "ranking": "ascending (route lower bound, 12-byte episode ID)",
        "composite_route": (
            "authority-certified error-corrected distance-v1 quantized lower bound; "
            "top 1000 is never displaced by component routes"
        ),
        "component_routes": (
            "bounded additive proposal seeds calculated from the same certified pack; "
            "not recall certificates and never allowed to remove a composite candidate"
        ),
        "overflow": (
            "metadata-only sidecar rows enter every route at universal lower bound zero "
            "and require native exact materialization"
        ),
        "io": "bounded positional reads; no full-pack mmap traversal",
        "default_route_quotas": DEFAULT_ROUTE_QUOTAS,
        "outcomes_or_labels_used": False,
    }
    if branch_aware:
        payload["branch_aware_bound_contract_digest"] = (
            branch_aware_quantized_bound_contract()["digest"]
        )
        payload["query_binding"] = (
            "result report binds the complete packed-query identity, eligibility "
            "range, tiers and exact representation arrays"
        )
    payload["digest"] = stable_hash(payload)
    return payload


def packed_bound_batch_search_contract(*, branch_aware: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": (
            BATCH_BRANCH_AWARE_SEARCH_SCHEMA_VERSION
            if branch_aware else BATCH_SEARCH_SCHEMA_VERSION
        ),
        "scalar_contract_digest": packed_bound_search_contract(
            branch_aware=branch_aware,
        )["digest"],
        "io": "each physical pack block is read once and offered to every query",
        "query_isolation": (
            "eligibility, bounded route heaps, row accounting, candidate ordering "
            "and scalar result digest remain independent per query"
        ),
        "ordering": "input query order; query episode IDs must be unique",
        "outcomes_or_labels_used": False,
    }
    payload["digest"] = stable_hash(payload)
    return payload


def packed_bound_threshold_scan_contract(*, branch_aware: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": (
            BRANCH_AWARE_THRESHOLD_SCAN_SCHEMA_VERSION
            if branch_aware else THRESHOLD_SCAN_SCHEMA_VERSION
        ),
        "packed_store_contract_digest": packed_bound_store_contract()["digest"],
        "eligibility": packed_bound_search_contract(
            branch_aware=branch_aware,
        )["eligibility"],
        "band": "strict lower-exclusive and inclusive upper; all upper ties admitted",
        "overflow": "universal bound zero; streamed in bounded blocks",
        "set_commitment": (
            "count plus xor and modular sum of unique SHA-256 row commitments; "
            "generation integrity independently guarantees unique episode IDs"
        ),
        "outcomes_or_labels_used": False,
    }
    if branch_aware:
        payload["branch_aware_bound_contract_digest"] = (
            branch_aware_quantized_bound_contract()["digest"]
        )
    payload["digest"] = stable_hash(payload)
    return payload


def _packed_query_input_digest(query: PackedBoundQuery) -> str:
    digest = sha256()
    representation = query.representation
    for name, values in (
        ("coarse", representation.coarse),
        ("stage", representation.stage),
        ("structural", representation.structural),
    ):
        digest.update(name.encode())
        digest.update(np.asarray(values, dtype="<f8").tobytes())
    for name in sorted(representation.samples_48):
        digest.update(name.encode())
        values = representation.samples_48[name]
        if values is None:
            digest.update(b"\0")
        else:
            digest.update(b"\1")
            digest.update(np.asarray(values, dtype="<f8").tobytes())
    return stable_hash({
        "episode_id": query.episode_id,
        "symbol": query.symbol,
        "query_start_ns": query.query_start_ns,
        "latest_eligible_ns": query.latest_eligible_ns,
        "quality_tiers": query.quality_tiers,
        "packed_bound_representation_sha256": digest.hexdigest(),
    })


def _current_rss_mb() -> float:
    try:
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / 1024 ** 2
    except (OSError, ValueError, IndexError):
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _empty_entries() -> np.ndarray:
    return np.empty(0, dtype=_ENTRY_DTYPE)


def _stable_bounded(entries: np.ndarray, incoming: np.ndarray, quota: int) -> np.ndarray:
    if quota < 1:
        raise PackedBoundSearchError("route quotas must be positive")
    merged = incoming if not len(entries) else np.concatenate((entries, incoming))
    if not len(merged):
        return _empty_entries()
    if not np.isfinite(merged["route_score"]).all() or np.any(merged["route_score"] < 0):
        raise PackedBoundSearchError("route lower bounds must be finite and nonnegative")
    if len(merged) <= quota:
        return merged.copy()
    scores = merged["route_score"]
    boundary = float(np.partition(scores, quota - 1)[quota - 1])
    lower = np.flatnonzero(scores < boundary)
    tied = np.flatnonzero(scores == boundary)
    needed = quota - len(lower)
    if needed < 0 or needed > len(tied):
        raise PackedBoundSearchError("stable quota partition accounting differs")
    tied_identifiers = np.frombuffer(
        merged["episode_id"][tied].tobytes(),
        dtype=np.dtype([("high", ">u8"), ("low", ">u4")]),
    )
    tied_order = np.lexsort((
        tied_identifiers["low"], tied_identifiers["high"],
    ))
    selected = np.concatenate((lower, tied[tied_order[:needed]]))
    return merged[selected].copy()


def _eligible_mask(records: np.ndarray, query: PackedBoundQuery, symbol_id: int | None) -> np.ndarray:
    query_identifier = np.void(bytes.fromhex(query.episode_id))
    mask = (
        (records["cutoff_ns"] <= query.latest_eligible_ns)
        & (records["episode_id"] != query_identifier)
    )
    if set(query.quality_tiers) != set(TIER_CODES):
        allowed = np.asarray(
            [TIER_CODES[value] for value in query.quality_tiers], dtype=np.uint8,
        )
        mask &= np.isin(records["quality_tier"], allowed)
    if symbol_id is not None:
        mask &= ~(
            (records["symbol_id"] == symbol_id)
            & (records["cutoff_ns"] >= query.query_start_ns)
        )
    return mask


def _entries(records: np.ndarray, totals: np.ndarray, scores: np.ndarray, *, overflow: bool) -> np.ndarray:
    output = np.empty(len(records), dtype=_ENTRY_DTYPE)
    output["episode_id"] = records["episode_id"]
    output["cutoff_ns"] = records["cutoff_ns"]
    output["symbol_id"] = records["symbol_id"]
    output["quality_tier"] = records["quality_tier"]
    output["total"] = totals
    output["route_score"] = scores
    output["overflow"] = overflow
    return output


def _finalize(
    heaps: Mapping[str, np.ndarray], symbols: tuple[str, ...],
) -> tuple[tuple[BoundProposal, ...], dict[str, int], str]:
    by_id: dict[bytes, dict[str, Any]] = {}
    route_counts: dict[str, int] = {}
    for route, entries in heaps.items():
        route_counts[route] = len(entries)
        for entry in entries:
            episode_id = bytes(entry["episode_id"])
            current = by_id.get(episode_id)
            if current is None:
                symbol_id = int(entry["symbol_id"])
                if symbol_id >= len(symbols):
                    raise PackedBoundSearchError("candidate symbol ID exceeds dictionary")
                current = {
                    "episode_id": episode_id.hex(),
                    "symbol": symbols[symbol_id],
                    "cutoff_ns": int(entry["cutoff_ns"]),
                    "quality_tier": next(
                        name for name, code in TIER_CODES.items()
                        if code == int(entry["quality_tier"])
                    ),
                    "lower_bound": float(entry["total"]),
                    "routes": set(),
                    "overflow_fallback": bool(entry["overflow"]),
                }
                by_id[episode_id] = current
            elif (
                current["cutoff_ns"] != int(entry["cutoff_ns"])
                or current["symbol"] != symbols[int(entry["symbol_id"])]
                or current["lower_bound"] != float(entry["total"])
                or current["overflow_fallback"] != bool(entry["overflow"])
            ):
                raise PackedBoundSearchError("duplicate episode metadata differs across routes")
            current["routes"].add(route)
    candidates = tuple(sorted((
        BoundProposal(
            value["episode_id"], value["symbol"], value["cutoff_ns"],
            value["quality_tier"], value["lower_bound"],
            tuple(sorted(value["routes"])), value["overflow_fallback"],
        ) for value in by_id.values()
    ), key=lambda value: (value.lower_bound, value.episode_id)))
    return candidates, route_counts, bound_proposal_candidate_digest(candidates)


def bound_proposal_candidate_digest(
    candidates: Iterable[BoundProposal],
) -> str:
    digest_payload = [{
        "episode_id": value.episode_id,
        "symbol": value.symbol,
        "cutoff_ns": value.cutoff_ns,
        "quality_tier": value.quality_tier,
        "lower_bound_hex": value.lower_bound.hex(),
        "routes": list(value.routes),
        "overflow_fallback": value.overflow_fallback,
    } for value in candidates]
    return stable_hash(digest_payload)


def scan_packed_bound_proposals(
    store_root: Path,
    generation_id: str,
    query: PackedBoundQuery,
    *,
    route_quotas: Mapping[str, int] | None = None,
    block_rows: int = 2_048,
    block_order: str = "forward",
    branch_aware: bool = False,
    verify_content: bool = True,
    expected_provenance_digest: str | None = None,
) -> BoundProposalReport:
    """Select a global stable proposal union without materializing the pack.

    ``block_order='reverse'`` exists for verifier use.  It changes physical scan
    order only; the ordered result must remain byte-identical.
    """
    if block_rows < 1:
        raise PackedBoundSearchError("block rows must be positive")
    if block_order not in {"forward", "reverse"}:
        raise PackedBoundSearchError("block order must be forward or reverse")
    quotas = dict(route_quotas or DEFAULT_ROUTE_QUOTAS)
    if "composite" not in quotas or quotas["composite"] < 1_000:
        raise PackedBoundSearchError("the certified composite route requires quota >= 1000")
    allowed_routes = {"composite", "coarse", "stage", "structural", "price", "candle_volatility", "volume_shock", "market_context"}
    if set(quotas) - allowed_routes:
        raise PackedBoundSearchError("route quotas contain unsupported components")
    if any(type(value) is not int or value < 1 for value in quotas.values()):
        raise PackedBoundSearchError("route quotas must be positive integers")

    loaded = load_packed_generation(
        store_root, generation_id,
        expected_provenance_digest=expected_provenance_digest,
        verify_content=verify_content, validate_records=False,
    )
    symbol_id = loaded.symbols.index(query.symbol) if query.symbol in loaded.symbols else None
    heaps = {route: _empty_entries() for route in quotas}
    started = perf_counter()
    peak = _current_rss_mb()
    eligible_main = 0
    pack_path = loaded.root / "generations" / loaded.generation_id / str(loaded.manifest["rows_file"])
    offsets: Iterable[int] = range(0, len(loaded.rows), block_rows)
    if block_order == "reverse":
        offsets = reversed(tuple(offsets))
    with pack_path.open("rb") as handle:
        for first in offsets:
            count = min(block_rows, len(loaded.rows) - first)
            raw = os.pread(handle.fileno(), count * PACK_DTYPE.itemsize, first * PACK_DTYPE.itemsize)
            if len(raw) != count * PACK_DTYPE.itemsize:
                raise PackedBoundSearchError("short positional read from packed generation")
            block = np.frombuffer(raw, dtype=PACK_DTYPE, count=count)
            selected = block[_eligible_mask(block, query, symbol_id)]
            eligible_main += len(selected)
            if len(selected):
                bounded = (
                    packed_branch_aware_lower_bounds(
                        query.representation, selected,
                    )
                    if branch_aware else
                    packed_lower_bounds(query.representation, selected)
                )
                route_values = {"composite": bounded.totals, **bounded.components}
                for route, quota in quotas.items():
                    incoming = _entries(
                        selected, bounded.totals,
                        np.asarray(route_values[route], dtype=np.float64),
                        overflow=False,
                    )
                    heaps[route] = _stable_bounded(heaps[route], incoming, quota)
            peak = max(peak, _current_rss_mb())

    overflow = np.asarray(loaded.overflow)
    selected_overflow = overflow[_eligible_mask(overflow, query, symbol_id)]
    eligible_overflow = len(selected_overflow)
    if eligible_overflow:
        zeros = np.zeros(eligible_overflow, dtype=np.float64)
        incoming = _entries(selected_overflow, zeros, zeros, overflow=True)
        for route, quota in quotas.items():
            heaps[route] = _stable_bounded(heaps[route], incoming, quota)
    candidates, route_counts, candidate_digest = _finalize(heaps, loaded.symbols)
    elapsed = perf_counter() - started
    schema_version = (
        BRANCH_AWARE_SEARCH_SCHEMA_VERSION
        if branch_aware else SEARCH_SCHEMA_VERSION
    )
    contract_digest = packed_bound_search_contract(
        branch_aware=branch_aware,
    )["digest"]
    input_digest = _packed_query_input_digest(query) if branch_aware else None
    deterministic = {
        "schema_version": schema_version,
        "contract_digest": contract_digest,
        "generation_id": loaded.generation_id,
        "query_episode_id": query.episode_id,
        "rows_scanned": len(loaded.rows) + len(loaded.overflow),
        "eligible_rows": eligible_main + eligible_overflow,
        "eligible_main_rows": eligible_main,
        "eligible_overflow_rows": eligible_overflow,
        "route_counts": route_counts,
        "route_quotas": quotas,
        "candidate_digest": candidate_digest,
        "real_forward_outcomes_accessed": False,
    }
    if branch_aware:
        deterministic["input_digest"] = input_digest
    return BoundProposalReport(
        schema_version, loaded.generation_id, query.episode_id,
        candidates, deterministic["rows_scanned"], deterministic["eligible_rows"],
        eligible_main, eligible_overflow, route_counts, quotas, block_rows,
        block_order, elapsed, peak, candidate_digest, stable_hash(deterministic),
        contract_digest if branch_aware else None, input_digest,
    )


def _bounded_ordered_thread_results(
    executor: ThreadPoolExecutor, function: Callable[[int], Any],
    offsets: Iterable[int], maximum_in_flight: int,
) -> Iterable[Any]:
    iterator = iter(offsets)
    pending: list[Future[Any]] = []
    for _ in range(maximum_in_flight):
        try:
            pending.append(executor.submit(function, next(iterator)))
        except StopIteration:
            break
    while pending:
        future = pending.pop(0)
        yield future.result()
        try:
            pending.append(executor.submit(function, next(iterator)))
        except StopIteration:
            pass


def scan_packed_bound_proposals_threaded(
    store_root: Path,
    generation_id: str,
    query: PackedBoundQuery,
    *,
    route_quotas: Mapping[str, int] | None = None,
    block_rows: int = 4_096,
    block_order: str = "forward",
    threads: int = 4,
    branch_aware: bool = False,
    verify_content: bool = True,
    expected_provenance_digest: str | None = None,
) -> BoundProposalReport:
    """Bounded parallel scoring with exact scalar-v1/v2 ordered reduction."""
    if block_rows < 1 or threads < 1:
        raise PackedBoundSearchError("block rows and threads must be positive")
    if block_order not in {"forward", "reverse"}:
        raise PackedBoundSearchError("block order must be forward or reverse")
    quotas = dict(route_quotas or DEFAULT_ROUTE_QUOTAS)
    if "composite" not in quotas or quotas["composite"] < 1_000:
        raise PackedBoundSearchError("the certified composite route requires quota >= 1000")
    allowed_routes = {
        "composite", "coarse", "stage", "structural", "price",
        "candle_volatility", "volume_shock", "market_context",
    }
    if set(quotas) - allowed_routes or any(
        type(value) is not int or value < 1 for value in quotas.values()
    ):
        raise PackedBoundSearchError("threaded route quotas are invalid")
    loaded = load_packed_generation(
        store_root, generation_id,
        expected_provenance_digest=expected_provenance_digest,
        verify_content=verify_content, validate_records=False,
    )
    symbol_id = loaded.symbols.index(query.symbol) if query.symbol in loaded.symbols else None
    heaps = {route: _empty_entries() for route in quotas}
    started = perf_counter()
    peak = _current_rss_mb()
    eligible_main = 0
    pack_path = loaded.root / "generations" / loaded.generation_id / str(loaded.manifest["rows_file"])
    offsets = list(range(0, len(loaded.rows), block_rows))
    if block_order == "reverse":
        offsets.reverse()
    with pack_path.open("rb") as handle:
        descriptor = handle.fileno()

        def score_block(first: int) -> tuple[int, dict[str, np.ndarray]]:
            numba.set_num_threads(1)
            count = min(block_rows, len(loaded.rows) - first)
            raw = os.pread(
                descriptor, count * PACK_DTYPE.itemsize,
                first * PACK_DTYPE.itemsize,
            )
            if len(raw) != count * PACK_DTYPE.itemsize:
                raise PackedBoundSearchError("short threaded positional read")
            block = np.frombuffer(raw, dtype=PACK_DTYPE, count=count)
            selected = block[_eligible_mask(block, query, symbol_id)]
            if not len(selected):
                return 0, {route: _empty_entries() for route in quotas}
            bounded = (
                packed_branch_aware_lower_bounds(query.representation, selected)
                if branch_aware else
                packed_lower_bounds(query.representation, selected)
            )
            route_values = {"composite": bounded.totals, **bounded.components}
            return len(selected), {
                route: _entries(
                    selected, bounded.totals,
                    np.asarray(route_values[route], dtype=np.float64),
                    overflow=False,
                ) for route in quotas
            }

        with ThreadPoolExecutor(
            max_workers=threads, thread_name_prefix="legacy-bound",
        ) as executor:
            for eligible, entries in _bounded_ordered_thread_results(
                executor, score_block, offsets, threads,
            ):
                eligible_main += eligible
                for route, quota in quotas.items():
                    heaps[route] = _stable_bounded(
                        heaps[route], entries[route], quota,
                    )
                peak = max(peak, _current_rss_mb())
    overflow = np.asarray(loaded.overflow)
    selected_overflow = overflow[_eligible_mask(overflow, query, symbol_id)]
    eligible_overflow = len(selected_overflow)
    if eligible_overflow:
        zeros = np.zeros(eligible_overflow, dtype=np.float64)
        incoming = _entries(selected_overflow, zeros, zeros, overflow=True)
        for route, quota in quotas.items():
            heaps[route] = _stable_bounded(heaps[route], incoming, quota)
    candidates, route_counts, candidate_digest = _finalize(heaps, loaded.symbols)
    elapsed = perf_counter() - started
    schema_version = (
        BRANCH_AWARE_SEARCH_SCHEMA_VERSION
        if branch_aware else SEARCH_SCHEMA_VERSION
    )
    contract_digest = packed_bound_search_contract(
        branch_aware=branch_aware,
    )["digest"]
    input_digest = _packed_query_input_digest(query) if branch_aware else None
    deterministic = {
        "schema_version": schema_version,
        "contract_digest": contract_digest,
        "generation_id": loaded.generation_id,
        "query_episode_id": query.episode_id,
        "rows_scanned": len(loaded.rows) + len(loaded.overflow),
        "eligible_rows": eligible_main + eligible_overflow,
        "eligible_main_rows": eligible_main,
        "eligible_overflow_rows": eligible_overflow,
        "route_counts": route_counts, "route_quotas": quotas,
        "candidate_digest": candidate_digest,
        "real_forward_outcomes_accessed": False,
    }
    if branch_aware:
        deterministic["input_digest"] = input_digest
    return BoundProposalReport(
        schema_version, loaded.generation_id, query.episode_id,
        candidates, deterministic["rows_scanned"], deterministic["eligible_rows"],
        eligible_main, eligible_overflow, route_counts, quotas, block_rows,
        block_order, elapsed, peak, candidate_digest, stable_hash(deterministic),
        contract_digest if branch_aware else None, input_digest,
    )


def scan_packed_component_bound_proposals_threaded(
    store_root: Path,
    generation_id: str,
    query: PackedBoundQuery,
    *,
    component: str,
    quota: int,
    block_rows: int = 4_096,
    block_order: str = "forward",
    threads: int = 4,
    verify_content: bool = True,
    expected_provenance_digest: str | None = None,
) -> BoundProposalReport:
    """Return a stable branch-aware frontier ranked by one safe component bound.

    Multi-route reports intentionally retain the composite total in each
    candidate.  A component certificate cannot use that field.  This dedicated
    report instead stores the named route score as ``lower_bound`` and cannot be
    mistaken for a composite frontier because its schema and contract differ.
    """
    contract = packed_component_search_contract(component)
    if type(quota) is not int or isinstance(quota, bool) or quota < 1:
        raise PackedBoundSearchError("component quota must be a positive integer")
    if block_rows < 1 or threads < 1:
        raise PackedBoundSearchError("block rows and threads must be positive")
    if block_order not in {"forward", "reverse"}:
        raise PackedBoundSearchError("block order must be forward or reverse")
    loaded = load_packed_generation(
        store_root, generation_id,
        expected_provenance_digest=expected_provenance_digest,
        verify_content=verify_content, validate_records=False,
    )
    symbol_id = loaded.symbols.index(query.symbol) if query.symbol in loaded.symbols else None
    heap = _empty_entries()
    started = perf_counter()
    peak = _current_rss_mb()
    eligible_main = 0
    pack_path = (
        loaded.root / "generations" / loaded.generation_id
        / str(loaded.manifest["rows_file"])
    )
    offsets = list(range(0, len(loaded.rows), block_rows))
    if block_order == "reverse":
        offsets.reverse()
    with pack_path.open("rb") as handle:
        descriptor = handle.fileno()

        def score_block(first: int) -> tuple[int, np.ndarray]:
            numba.set_num_threads(1)
            count = min(block_rows, len(loaded.rows) - first)
            raw = os.pread(
                descriptor, count * PACK_DTYPE.itemsize,
                first * PACK_DTYPE.itemsize,
            )
            if len(raw) != count * PACK_DTYPE.itemsize:
                raise PackedBoundSearchError("short component positional read")
            block = np.frombuffer(raw, dtype=PACK_DTYPE, count=count)
            selected = block[_eligible_mask(block, query, symbol_id)]
            if not len(selected):
                return 0, _empty_entries()
            bounded = packed_branch_aware_lower_bounds(query.representation, selected)
            scores = np.asarray(bounded.components[component], dtype=np.float64)
            # Store the component score in both fields: _stable_bounded ranks
            # route_score and _finalize publishes total as lower_bound.
            return len(selected), _entries(
                selected, scores, scores, overflow=False,
            )

        with ThreadPoolExecutor(
            max_workers=threads, thread_name_prefix="component-bound",
        ) as executor:
            for eligible, entries in _bounded_ordered_thread_results(
                executor, score_block, offsets, threads,
            ):
                eligible_main += eligible
                heap = _stable_bounded(heap, entries, quota)
                peak = max(peak, _current_rss_mb())
    overflow = np.asarray(loaded.overflow)
    selected_overflow = overflow[_eligible_mask(overflow, query, symbol_id)]
    eligible_overflow = len(selected_overflow)
    if eligible_overflow:
        zeros = np.zeros(eligible_overflow, dtype=np.float64)
        heap = _stable_bounded(
            heap, _entries(selected_overflow, zeros, zeros, overflow=True), quota,
        )
    candidates, route_counts, candidate_digest = _finalize(
        {component: heap}, loaded.symbols,
    )
    input_digest = _packed_query_input_digest(query)
    deterministic = {
        "schema_version": COMPONENT_SEARCH_SCHEMA_VERSION,
        "contract_digest": contract["digest"], "component": component,
        "generation_id": loaded.generation_id,
        "query_episode_id": query.episode_id,
        "rows_scanned": len(loaded.rows) + len(loaded.overflow),
        "eligible_rows": eligible_main + eligible_overflow,
        "eligible_main_rows": eligible_main,
        "eligible_overflow_rows": eligible_overflow,
        "route_counts": route_counts, "route_quotas": {component: quota},
        "candidate_digest": candidate_digest,
        "real_forward_outcomes_accessed": False,
        "input_digest": input_digest,
    }
    return BoundProposalReport(
        COMPONENT_SEARCH_SCHEMA_VERSION, loaded.generation_id, query.episode_id,
        candidates, deterministic["rows_scanned"], deterministic["eligible_rows"],
        eligible_main, eligible_overflow, route_counts, {component: quota},
        block_rows, block_order, perf_counter() - started, peak,
        candidate_digest, stable_hash(deterministic), contract["digest"], input_digest,
    )


def scan_packed_bound_threshold(
    store_root: Path,
    generation_id: str,
    query: PackedBoundQuery,
    *,
    upper_inclusive: float,
    consume: Callable[[tuple[BoundProposal, ...]], None],
    lower_exclusive: float | None = None,
    excluded_episode_ids: AbstractSet[str] = frozenset(),
    block_rows: int = 2_048,
    block_order: str = "forward",
    branch_aware: bool = False,
    verify_content: bool = True,
    expected_provenance_digest: str | None = None,
    component: str | None = None,
) -> BoundThresholdScanReport:
    """Stream every eligible row in ``(lower, upper]`` without a global heap.

    The callback bounds live proposal memory to one physical pack block.  The
    admitted-set digest is deliberately independent of block size and scan
    direction so a verifier can classify the same immutable generation through
    a different traversal.  ``excluded_episode_ids`` is intended only for a
    previously certified prefix; later closure bands use ``lower_exclusive``.
    """
    if block_rows < 1:
        raise PackedBoundSearchError("block rows must be positive")
    if block_order not in {"forward", "reverse"}:
        raise PackedBoundSearchError("block order must be forward or reverse")
    if not callable(consume):
        raise PackedBoundSearchError("threshold consumer must be callable")
    component_contract = (
        packed_component_threshold_scan_contract(component)
        if component is not None else None
    )
    if component is not None and not branch_aware:
        raise PackedBoundSearchError("component threshold scan requires branch-aware bounds")
    upper = float(upper_inclusive)
    lower = None if lower_exclusive is None else float(lower_exclusive)
    if np.isnan(upper) or upper < 0 or lower is not None and (
        not np.isfinite(lower) or lower < 0 or lower >= upper
    ):
        raise PackedBoundSearchError("threshold scan bounds are invalid")
    try:
        if any(
            len(value) != 24 or value.lower() != value
            or len(bytes.fromhex(value)) != 12
            for value in excluded_episode_ids
        ):
            raise ValueError
    except ValueError as exc:
        raise PackedBoundSearchError(
            "threshold exclusions contain an invalid episode ID"
        ) from exc

    loaded = load_packed_generation(
        store_root, generation_id,
        expected_provenance_digest=expected_provenance_digest,
        verify_content=verify_content, validate_records=False,
    )
    symbol_id = loaded.symbols.index(query.symbol) if query.symbol in loaded.symbols else None
    started = perf_counter()
    peak = _current_rss_mb()
    eligible_main = eligible_overflow = excluded_eligible = admitted_rows = 0
    minimum_above = float("inf")
    digest_xor = 0
    digest_sum = 0
    modulus = 1 << 256

    def fresh_records(records: np.ndarray) -> np.ndarray:
        nonlocal excluded_eligible
        if not len(records) or not excluded_episode_ids:
            return records
        keep = np.fromiter(
            (
                bytes(record["episode_id"]).hex() not in excluded_episode_ids
                for record in records
            ),
            dtype=bool, count=len(records),
        )
        excluded_eligible += int((~keep).sum())
        return records[keep]

    def emit(records: np.ndarray, totals: np.ndarray, *, overflow: bool) -> None:
        nonlocal admitted_rows, digest_xor, digest_sum
        proposals: list[BoundProposal] = []
        for record, total_value in zip(records, totals, strict=True):
            episode_id = bytes(record["episode_id"]).hex()
            numeric_symbol_id = int(record["symbol_id"])
            if numeric_symbol_id >= len(loaded.symbols):
                raise PackedBoundSearchError("candidate symbol ID exceeds dictionary")
            quality_code = int(record["quality_tier"])
            if quality_code not in TIER_NAMES:
                raise PackedBoundSearchError("candidate quality tier differs")
            proposal = BoundProposal(
                episode_id, loaded.symbols[numeric_symbol_id],
                int(record["cutoff_ns"]), TIER_NAMES[quality_code],
                float(total_value),
                ((component,) if component is not None else ("composite",)),
                overflow,
            )
            proposals.append(proposal)
            row_hash = int(stable_hash({
                "episode_id": proposal.episode_id,
                "symbol": proposal.symbol,
                "cutoff_ns": proposal.cutoff_ns,
                "quality_tier": proposal.quality_tier,
                "lower_bound_hex": proposal.lower_bound.hex(),
                "overflow_fallback": proposal.overflow_fallback,
            }), 16)
            digest_xor ^= row_hash
            digest_sum = (digest_sum + row_hash) % modulus
        if proposals:
            admitted_rows += len(proposals)
            consume(tuple(proposals))

    pack_path = (
        loaded.root / "generations" / loaded.generation_id
        / str(loaded.manifest["rows_file"])
    )
    offsets: Iterable[int] = range(0, len(loaded.rows), block_rows)
    if block_order == "reverse":
        offsets = reversed(offsets)
    with pack_path.open("rb") as handle:
        for first in offsets:
            count = min(block_rows, len(loaded.rows) - first)
            raw = os.pread(
                handle.fileno(), count * PACK_DTYPE.itemsize,
                first * PACK_DTYPE.itemsize,
            )
            if len(raw) != count * PACK_DTYPE.itemsize:
                raise PackedBoundSearchError("short positional read from packed generation")
            block = np.frombuffer(raw, dtype=PACK_DTYPE, count=count)
            selected = block[_eligible_mask(block, query, symbol_id)]
            eligible_main += len(selected)
            fresh = fresh_records(selected)
            if len(fresh):
                bounded = (
                    packed_branch_aware_lower_bounds(query.representation, fresh)
                    if branch_aware else packed_lower_bounds(query.representation, fresh)
                )
                totals = np.asarray(
                    bounded.totals if component is None
                    else bounded.components[component], dtype=np.float64,
                )
                if not np.isfinite(totals).all() or np.any(totals < 0):
                    raise PackedBoundSearchError(
                        "threshold scan produced an invalid packed lower bound"
                    )
                above = totals > upper
                if np.any(above):
                    minimum_above = min(minimum_above, float(np.min(totals[above])))
                admitted = totals <= upper
                if lower is not None:
                    admitted &= totals > lower
                if np.any(admitted):
                    emit(fresh[admitted], totals[admitted], overflow=False)
            peak = max(peak, _current_rss_mb())

    overflow_offsets: Iterable[int] = range(0, len(loaded.overflow), block_rows)
    if block_order == "reverse":
        overflow_offsets = reversed(overflow_offsets)
    overflow_admitted = lower is None
    for first in overflow_offsets:
        block = loaded.overflow[first:first + block_rows]
        selected = block[_eligible_mask(block, query, symbol_id)]
        eligible_overflow += len(selected)
        fresh = fresh_records(selected)
        if len(fresh) and overflow_admitted:
            emit(
                fresh, np.zeros(len(fresh), dtype=np.float64), overflow=True,
            )
        peak = max(peak, _current_rss_mb())
    admitted_set_digest = stable_hash({
        "rows": admitted_rows,
        "xor": f"{digest_xor:064x}",
        "sum": f"{digest_sum:064x}",
    })
    minimum = None if not np.isfinite(minimum_above) else minimum_above
    schema_version = (
        COMPONENT_THRESHOLD_SCAN_SCHEMA_VERSION if component is not None else
        BRANCH_AWARE_THRESHOLD_SCAN_SCHEMA_VERSION if branch_aware else
        THRESHOLD_SCAN_SCHEMA_VERSION
    )
    contract_digest = (
        component_contract["digest"] if component_contract is not None else
        packed_bound_threshold_scan_contract(branch_aware=branch_aware)["digest"]
    )
    input_digest = _packed_query_input_digest(query)
    exclusions_digest = stable_hash(sorted(excluded_episode_ids))
    eligible_rows = eligible_main + eligible_overflow
    deterministic = {
        "schema_version": schema_version,
        "contract_digest": contract_digest,
        "generation_id": loaded.generation_id,
        "query_episode_id": query.episode_id,
        "input_digest": input_digest,
        "exclusions_digest": exclusions_digest,
        "lower_exclusive_hex": lower.hex() if lower is not None else None,
        "upper_inclusive_hex": upper.hex(),
        "rows_scanned": len(loaded.rows) + len(loaded.overflow),
        "eligible_rows": eligible_rows,
        "eligible_main_rows": eligible_main,
        "eligible_overflow_rows": eligible_overflow,
        "excluded_eligible_rows": excluded_eligible,
        "admitted_rows": admitted_rows,
        "minimum_above_upper_hex": minimum.hex() if minimum is not None else None,
        "admitted_set_digest": admitted_set_digest,
        "real_forward_outcomes_accessed": False,
    }
    if component is not None:
        deterministic["component"] = component
    return BoundThresholdScanReport(
        schema_version, contract_digest, loaded.generation_id,
        query.episode_id, input_digest, exclusions_digest, lower, upper,
        deterministic["rows_scanned"], eligible_rows, eligible_main,
        eligible_overflow, excluded_eligible, admitted_rows, minimum,
        admitted_set_digest, block_rows, block_order, perf_counter() - started,
        peak, stable_hash(deterministic),
    )


def scan_packed_bound_proposals_many(
    store_root: Path,
    generation_id: str,
    queries: Iterable[PackedBoundQuery],
    *,
    route_quotas: Mapping[str, int] | None = None,
    block_rows: int = 2_048,
    block_order: str = "forward",
    branch_aware: bool = False,
    verify_content: bool = True,
    expected_provenance_digest: str | None = None,
) -> BoundProposalBatchReport:
    """Run scalar-equivalent proposal selection with one shared physical read."""
    query_rows = tuple(queries)
    if not query_rows:
        raise PackedBoundSearchError("batch queries must be non-empty")
    query_ids = tuple(query.episode_id for query in query_rows)
    if len(set(query_ids)) != len(query_ids):
        raise PackedBoundSearchError("batch query episode IDs must be unique")
    if block_rows < 1:
        raise PackedBoundSearchError("block rows must be positive")
    if block_order not in {"forward", "reverse"}:
        raise PackedBoundSearchError("block order must be forward or reverse")
    quotas = dict(route_quotas or DEFAULT_ROUTE_QUOTAS)
    if "composite" not in quotas or quotas["composite"] < 1_000:
        raise PackedBoundSearchError("the certified composite route requires quota >= 1000")
    allowed_routes = {
        "composite", "coarse", "stage", "structural", "price",
        "candle_volatility", "volume_shock", "market_context",
    }
    if set(quotas) - allowed_routes:
        raise PackedBoundSearchError("route quotas contain unsupported components")
    if any(type(value) is not int or value < 1 for value in quotas.values()):
        raise PackedBoundSearchError("route quotas must be positive integers")

    loaded = load_packed_generation(
        store_root, generation_id,
        expected_provenance_digest=expected_provenance_digest,
        verify_content=verify_content, validate_records=False,
    )
    states = [{
        "query": query,
        "symbol_id": (
            loaded.symbols.index(query.symbol)
            if query.symbol in loaded.symbols else None
        ),
        "heaps": {route: _empty_entries() for route in quotas},
        "eligible_main": 0,
    } for query in query_rows]
    started = perf_counter()
    peak = _current_rss_mb()
    pack_path = (
        loaded.root / "generations" / loaded.generation_id
        / str(loaded.manifest["rows_file"])
    )
    offsets: Iterable[int] = range(0, len(loaded.rows), block_rows)
    if block_order == "reverse":
        offsets = reversed(tuple(offsets))
    with pack_path.open("rb") as handle:
        for first in offsets:
            count = min(block_rows, len(loaded.rows) - first)
            raw = os.pread(
                handle.fileno(), count * PACK_DTYPE.itemsize,
                first * PACK_DTYPE.itemsize,
            )
            if len(raw) != count * PACK_DTYPE.itemsize:
                raise PackedBoundSearchError("short positional read from packed generation")
            block = np.frombuffer(raw, dtype=PACK_DTYPE, count=count)
            prepared = prepare_packed_lower_bound_records(block)
            for state in states:
                query = state["query"]
                eligible_mask = _eligible_mask(
                    block, query, state["symbol_id"],
                )
                selected = block[eligible_mask]
                state["eligible_main"] += len(selected)
                if not len(selected):
                    continue
                bounded = (
                    prepared_packed_branch_aware_lower_bounds(
                        query.representation, prepared, eligible_mask,
                    )
                    if branch_aware else
                    prepared_packed_lower_bounds(
                        query.representation, prepared, eligible_mask,
                    )
                )
                route_values = {"composite": bounded.totals, **bounded.components}
                for route, quota in quotas.items():
                    incoming = _entries(
                        selected, bounded.totals,
                        np.asarray(route_values[route], dtype=np.float64),
                        overflow=False,
                    )
                    state["heaps"][route] = _stable_bounded(
                        state["heaps"][route], incoming, quota,
                    )
            peak = max(peak, _current_rss_mb())

    overflow = np.asarray(loaded.overflow)
    report_values = []
    for state in states:
        query = state["query"]
        selected_overflow = overflow[_eligible_mask(
            overflow, query, state["symbol_id"],
        )]
        eligible_overflow = len(selected_overflow)
        if eligible_overflow:
            zeros = np.zeros(eligible_overflow, dtype=np.float64)
            incoming = _entries(selected_overflow, zeros, zeros, overflow=True)
            for route, quota in quotas.items():
                state["heaps"][route] = _stable_bounded(
                    state["heaps"][route], incoming, quota,
                )
        candidates, route_counts, candidate_digest = _finalize(
            state["heaps"], loaded.symbols,
        )
        eligible_main = int(state["eligible_main"])
        scalar_schema = (
            BRANCH_AWARE_SEARCH_SCHEMA_VERSION
            if branch_aware else SEARCH_SCHEMA_VERSION
        )
        scalar_contract_digest = packed_bound_search_contract(
            branch_aware=branch_aware,
        )["digest"]
        input_digest = (
            _packed_query_input_digest(query) if branch_aware else None
        )
        deterministic = {
            "schema_version": scalar_schema,
            "contract_digest": scalar_contract_digest,
            "generation_id": loaded.generation_id,
            "query_episode_id": query.episode_id,
            "rows_scanned": len(loaded.rows) + len(loaded.overflow),
            "eligible_rows": eligible_main + eligible_overflow,
            "eligible_main_rows": eligible_main,
            "eligible_overflow_rows": eligible_overflow,
            "route_counts": route_counts,
            "route_quotas": quotas,
            "candidate_digest": candidate_digest,
            "real_forward_outcomes_accessed": False,
        }
        if branch_aware:
            deterministic["input_digest"] = input_digest
        report_values.append((
            query, candidates, eligible_main, eligible_overflow,
            route_counts, candidate_digest, stable_hash(deterministic),
            scalar_contract_digest if branch_aware else None, input_digest,
        ))
    elapsed = perf_counter() - started
    reports = tuple(BoundProposalReport(
        (
            BRANCH_AWARE_SEARCH_SCHEMA_VERSION
            if branch_aware else SEARCH_SCHEMA_VERSION
        ), loaded.generation_id, query.episode_id,
        candidates, len(loaded.rows) + len(loaded.overflow),
        eligible_main + eligible_overflow, eligible_main, eligible_overflow,
        route_counts, quotas, block_rows, block_order, elapsed, peak,
        candidate_digest, result_digest, contract_digest, input_digest,
    ) for (
        query, candidates, eligible_main, eligible_overflow,
        route_counts, candidate_digest, result_digest, contract_digest,
        input_digest,
    ) in report_values)
    physical_rows = len(loaded.rows) + len(loaded.overflow)
    batch_schema = (
        BATCH_BRANCH_AWARE_SEARCH_SCHEMA_VERSION
        if branch_aware else BATCH_SEARCH_SCHEMA_VERSION
    )
    deterministic = {
        "schema_version": batch_schema,
        "contract_digest": packed_bound_batch_search_contract(
            branch_aware=branch_aware,
        )["digest"],
        "generation_id": loaded.generation_id,
        "query_episode_ids": query_ids,
        "per_query_result_digests": [report.result_digest for report in reports],
        "physical_rows_scanned": physical_rows,
        "logical_rows_evaluated": physical_rows * len(query_rows),
        "real_forward_outcomes_accessed": False,
    }
    return BoundProposalBatchReport(
        batch_schema, loaded.generation_id, query_ids, reports,
        physical_rows, physical_rows * len(query_rows), block_rows, block_order,
        elapsed, peak, stable_hash(deterministic),
    )
