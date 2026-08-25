from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import resource
from time import perf_counter
from typing import Any, Iterable, Mapping

import numpy as np

from .packed_bound_store import (
    PACK_DTYPE, TIER_CODES, load_packed_generation, packed_lower_bounds,
    prepare_packed_lower_bound_records, prepared_packed_lower_bounds,
)
from .representation import Representation
from .types import stable_hash


SEARCH_SCHEMA_VERSION = "m04r-global-bound-proposal-v1"
BATCH_SEARCH_SCHEMA_VERSION = "m04r-global-bound-proposal-batch-v1"
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


_ENTRY_DTYPE = np.dtype([
    ("episode_id", "V12"),
    ("cutoff_ns", "<i8"),
    ("symbol_id", "<u4"),
    ("quality_tier", "u1"),
    ("total", "<f8"),
    ("route_score", "<f8"),
    ("overflow", "?"),
])


def packed_bound_search_contract() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SEARCH_SCHEMA_VERSION,
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
    payload["digest"] = stable_hash(payload)
    return payload


def packed_bound_batch_search_contract() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": BATCH_SEARCH_SCHEMA_VERSION,
        "scalar_contract_digest": packed_bound_search_contract()["digest"],
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
    digest_payload = [{
        "episode_id": value.episode_id,
        "symbol": value.symbol,
        "cutoff_ns": value.cutoff_ns,
        "quality_tier": value.quality_tier,
        "lower_bound_hex": value.lower_bound.hex(),
        "routes": list(value.routes),
        "overflow_fallback": value.overflow_fallback,
    } for value in candidates]
    return candidates, route_counts, stable_hash(digest_payload)


def scan_packed_bound_proposals(
    store_root: Path,
    generation_id: str,
    query: PackedBoundQuery,
    *,
    route_quotas: Mapping[str, int] | None = None,
    block_rows: int = 2_048,
    block_order: str = "forward",
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
                bounded = packed_lower_bounds(query.representation, selected)
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
    deterministic = {
        "schema_version": SEARCH_SCHEMA_VERSION,
        "contract_digest": packed_bound_search_contract()["digest"],
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
    return BoundProposalReport(
        SEARCH_SCHEMA_VERSION, loaded.generation_id, query.episode_id,
        candidates, deterministic["rows_scanned"], deterministic["eligible_rows"],
        eligible_main, eligible_overflow, route_counts, quotas, block_rows,
        block_order, elapsed, peak, candidate_digest, stable_hash(deterministic),
    )


def scan_packed_bound_proposals_many(
    store_root: Path,
    generation_id: str,
    queries: Iterable[PackedBoundQuery],
    *,
    route_quotas: Mapping[str, int] | None = None,
    block_rows: int = 2_048,
    block_order: str = "forward",
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
                bounded = prepared_packed_lower_bounds(
                    query.representation, prepared, eligible_mask,
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
        deterministic = {
            "schema_version": SEARCH_SCHEMA_VERSION,
            "contract_digest": packed_bound_search_contract()["digest"],
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
        report_values.append((
            query, candidates, eligible_main, eligible_overflow,
            route_counts, candidate_digest, stable_hash(deterministic),
        ))
    elapsed = perf_counter() - started
    reports = tuple(BoundProposalReport(
        SEARCH_SCHEMA_VERSION, loaded.generation_id, query.episode_id,
        candidates, len(loaded.rows) + len(loaded.overflow),
        eligible_main + eligible_overflow, eligible_main, eligible_overflow,
        route_counts, quotas, block_rows, block_order, elapsed, peak,
        candidate_digest, result_digest,
    ) for (
        query, candidates, eligible_main, eligible_overflow,
        route_counts, candidate_digest, result_digest,
    ) in report_values)
    physical_rows = len(loaded.rows) + len(loaded.overflow)
    deterministic = {
        "schema_version": BATCH_SEARCH_SCHEMA_VERSION,
        "contract_digest": packed_bound_batch_search_contract()["digest"],
        "generation_id": loaded.generation_id,
        "query_episode_ids": query_ids,
        "per_query_result_digests": [report.result_digest for report in reports],
        "physical_rows_scanned": physical_rows,
        "logical_rows_evaluated": physical_rows * len(query_rows),
        "real_forward_outcomes_accessed": False,
    }
    return BoundProposalBatchReport(
        BATCH_SEARCH_SCHEMA_VERSION, loaded.generation_id, query_ids, reports,
        physical_rows, physical_rows * len(query_rows), block_rows, block_order,
        elapsed, peak, stable_hash(deterministic),
    )
