"""Production B0-05 shared-priority concentration sensitivity.

The runner is outcome blind.  It consumes only the joint H1 contract, its
explicitly sealed case/risk-set authorities, and the frozen A/B packed
candidate metadata.  Every replicate hashes and orders the complete candidate
universe once, then filters those common orders for all queries.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from hashlib import sha256
from html import escape
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import resource
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

import numpy as np

from experiments.m04r import m04r14_r1b_joint_b005_b2_contract as joint
from market_analogues.adequacy import concentration_metrics, nearest_rank
from market_analogues.adequacy_shared_priority import (
    EPISODE_DOMAIN, SYMBOL_DOMAIN, SelectionConfig, priority_digest,
)
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.types import stable_hash


SCHEMA = "m04r14-r1b-b005-shared-priority-v1"
SHARD_SCHEMA = "m04r14-r1b-b005-shared-priority-shard-v1"
BINDING_SCHEMA = "m04r14-r1b-b005-shared-priority-binding-v1"
OUTPUT = Path(joint.OUTPUTS["b005"])
PACKED_DURABLE = Path("config/data/analogues/poc/m04r/packed-bound-full/store")
R1A_PREREG = joint.R1A_PREREG
DEFAULT_SHARD_SIZE = 8


class SharedPriorityRunError(RuntimeError):
    """Fail-closed B0-05 execution error."""


@dataclass(frozen=True)
class CandidateUniverse:
    """Compact, canonical complete-universe metadata."""

    episode_ids: np.ndarray  # V12 raw episode IDs
    cutoffs: np.ndarray  # int64 nanoseconds
    symbol_ids: np.ndarray  # uint32
    local_ordinals: np.ndarray  # int64, per-symbol five-session ordinal
    symbols: tuple[str, ...]
    starts: np.ndarray  # int64 offsets in symbol/cutoff order
    stops: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.episode_ids)
        if not all(len(value) == n for value in (
            self.cutoffs, self.symbol_ids, self.local_ordinals,
        )):
            raise SharedPriorityRunError("universe column lengths differ")
        if self.episode_ids.dtype.kind != "V" or self.episode_ids.dtype.itemsize != 12:
            raise SharedPriorityRunError("episode IDs must be raw V12 values")
        if len(self.starts) != len(self.symbols) or len(self.stops) != len(self.symbols):
            raise SharedPriorityRunError("symbol offsets differ")
        if n < 1 or len(self.symbols) < 1 or len(set(self.symbols)) != len(self.symbols):
            raise SharedPriorityRunError("empty or duplicate universe identities")


@dataclass(frozen=True, order=True)
class QueryRisk:
    query_id: str
    symbol_id: int
    start_ns: int
    latest_ns: int
    eligible_count: int


@dataclass(frozen=True)
class CompleteOrders:
    global_indices: np.ndarray
    hierarchical_indices: np.ndarray
    symbol_priority_order: np.ndarray
    counters: Mapping[str, int]


@dataclass(frozen=True)
class TieOrders:
    episode_ids: np.ndarray
    symbol_ids: np.ndarray


_FORK_EPISODE_IDS: np.ndarray | None = None
_FORK_PRIORITY_BYTES: Any = None


def _fork_episode_hash_range(arguments: tuple[int, int, int, int]) -> int:
    """Hash one disjoint range into an inherited shared-memory array."""

    seed, replicate, first, stop = arguments
    if _FORK_EPISODE_IDS is None or _FORK_PRIORITY_BYTES is None:
        raise SharedPriorityRunError("fork priority worker is not initialized")
    domain = EPISODE_DOMAIN.encode("utf-8")
    prefix = (len(domain).to_bytes(2, "big") + domain
              + seed.to_bytes(8, "big") + replicate.to_bytes(4, "big")
              + (24).to_bytes(2, "big"))
    output = np.frombuffer(_FORK_PRIORITY_BYTES, dtype=np.uint8).reshape(-1, 32)
    for index in range(first, stop):
        identifier = bytes(_FORK_EPISODE_IDS[index]).hex().encode("ascii")
        output[index] = np.frombuffer(sha256(prefix + identifier).digest(), dtype=np.uint8)
    return stop - first


class ForkEpisodeHasher:
    """Persistent fork workers writing full SHA-256 values to shared memory."""

    def __init__(self, episode_ids: np.ndarray, workers: int):
        require(type(workers) is int and workers >= 1, "workers must be positive")
        require("fork" in mp.get_all_start_methods(), "production hashing requires fork")
        self.workers = min(workers, len(episode_ids))
        self.length = len(episode_ids)
        self._context = mp.get_context("fork")
        self._buffer = self._context.RawArray("B", self.length * 32)
        global _FORK_EPISODE_IDS, _FORK_PRIORITY_BYTES
        _FORK_EPISODE_IDS = episode_ids
        _FORK_PRIORITY_BYTES = self._buffer
        self._pool = self._context.Pool(self.workers)
        block = (self.length + self.workers - 1) // self.workers
        self._ranges = tuple(
            (first, min(first + block, self.length))
            for first in range(0, self.length, block)
        )

    def hashes(self, *, seed: int, replicate: int) -> np.ndarray:
        completed = self._pool.map(
            _fork_episode_hash_range,
            ((seed, replicate, first, stop) for first, stop in self._ranges),
        )
        require(sum(completed) == self.length, "parallel episode hash accounting differs")
        return np.frombuffer(self._buffer, dtype=np.uint8).reshape(self.length, 32)

    def close(self) -> None:
        self._pool.close(); self._pool.join()

    def terminate(self) -> None:
        self._pool.terminate(); self._pool.join()

    def __enter__(self) -> "ForkEpisodeHasher":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is None:
            self.close()
        else:
            self.terminate()
        global _FORK_EPISODE_IDS, _FORK_PRIORITY_BYTES
        _FORK_EPISODE_IDS = None; _FORK_PRIORITY_BYTES = None


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SharedPriorityRunError(message)


def _json_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path, expected: type = dict) -> Any:
    try:
        value = json.loads(
            path.read_bytes(), object_pairs_hook=_json_pairs,
            parse_constant=lambda token: require(False, f"non-finite JSON: {token}"),
        )
    except (OSError, ValueError) as error:
        raise SharedPriorityRunError(f"unreadable JSON: {path}") from error
    require(isinstance(value, expected), f"expected JSON {expected.__name__}: {path}")
    return value


def file_sha(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"regular file required: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def require_regular_beneath(repository: Path, path: Path) -> None:
    root = repository.resolve()
    require(path.is_file() and not path.is_symlink(), f"regular file required: {path}")
    require(path.resolve().is_relative_to(root), f"file escapes repository: {path}")
    current = path.parent
    while current != root:
        require(not current.is_symlink(), f"symlink parent forbidden: {path}")
        require(current != current.parent, f"file is not beneath repository: {path}")
        current = current.parent


def atomic_json(path: Path, payload: Any) -> None:
    """Durable, create-only JSON publication."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid4().hex}"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise SharedPriorityRunError(f"create-only publication exists: {path}") from error
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_scientific_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _priority_chunk(arguments: tuple[tuple[bytes, ...], str, int, int]) -> bytes:
    identifiers, domain, seed, replicate = arguments
    return b"".join(priority_digest(
        domain=domain, seed=seed, replicate=replicate,
        identifier=value.decode("ascii"),
    ) for value in identifiers)


def priority_array(
    identifiers: Sequence[bytes], *, domain: str, seed: int, replicate: int,
    workers: int = 1,
) -> np.ndarray:
    """Hash every entity exactly once and return full 256-bit priorities.

    This portable implementation is also the serial/parallel synthetic oracle.
    The complete input is partitioned once; no query appears in this API.
    """

    require(type(workers) is int and workers >= 1, "workers must be positive")
    values = tuple(bytes(value) for value in identifiers)
    require(values and len(set(values)) == len(values), "priority identities must be unique")
    chunks = tuple(
        tuple(values[first::workers]) for first in range(min(workers, len(values)))
    )
    arguments = tuple((chunk, domain, seed, replicate) for chunk in chunks)
    if len(arguments) == 1:
        results = (_priority_chunk(arguments[0]),)
    else:
        with ProcessPoolExecutor(max_workers=len(arguments)) as executor:
            results = tuple(executor.map(_priority_chunk, arguments))
    # Strided chunking is re-interleaved into original universe order.
    output = np.empty((len(values), 32), dtype=np.uint8)
    for offset, block in enumerate(results):
        matrix = np.frombuffer(block, dtype=np.uint8).reshape(-1, 32)
        output[offset::len(arguments)] = matrix
    return output


def _void_priorities(matrix: np.ndarray) -> np.ndarray:
    require(matrix.dtype == np.uint8 and matrix.ndim == 2 and matrix.shape[1] == 32,
            "priority array must be uint8[N,32]")
    return np.ascontiguousarray(matrix).view("V32").reshape(-1)


def build_complete_orders(
    universe: CandidateUniverse, *, seed: int, replicate: int, workers: int = 1,
    episode_hashes: np.ndarray | None = None,
    symbol_hashes: np.ndarray | None = None,
    tie_orders: TieOrders | None = None,
) -> CompleteOrders:
    """Build exactly one global and one hierarchical complete-universe order."""

    if episode_hashes is None:
        episode_ascii = tuple(bytes(value).hex().encode("ascii") for value in universe.episode_ids)
        episode_hashes = priority_array(
            episode_ascii, domain=EPISODE_DOMAIN, seed=seed,
            replicate=replicate, workers=workers,
        )
    if symbol_hashes is None:
        symbol_ascii = tuple(value.encode("utf-8") for value in universe.symbols)
        symbol_hashes = priority_array(
            symbol_ascii, domain=SYMBOL_DOMAIN, seed=seed,
            replicate=replicate, workers=min(workers, len(symbol_ascii)),
        )
    require(len(episode_hashes) == len(universe.episode_ids), "episode hash count differs")
    require(len(symbol_hashes) == len(universe.symbols), "symbol hash count differs")
    ep_priority = _void_priorities(episode_hashes)
    sym_priority = _void_priorities(symbol_hashes)

    # Stable two-pass sorts exactly implement the explicit ID tie breakers.
    if tie_orders is None:
        tie_orders = TieOrders(
            np.argsort(universe.episode_ids, kind="stable").astype(np.int64, copy=False),
            np.argsort(np.asarray(universe.symbols, dtype="U"), kind="stable").astype(np.int64, copy=False),
        )
    require(len(tie_orders.episode_ids) == len(universe.episode_ids)
            and len(tie_orders.symbol_ids) == len(universe.symbols), "tie-order dimensions differ")
    episode_id_order = tie_orders.episode_ids
    global_indices = episode_id_order[
        np.argsort(ep_priority[episode_id_order], kind="stable")
    ].astype(np.int64, copy=False)
    symbol_id_order = tie_orders.symbol_ids
    symbol_priority_order = symbol_id_order[
        np.argsort(sym_priority[symbol_id_order], kind="stable")
    ].astype(np.int64, copy=False)
    symbol_rank = np.empty(len(universe.symbols), dtype=np.int64)
    symbol_rank[symbol_priority_order] = np.arange(len(symbol_priority_order), dtype=np.int64)
    global_rank = np.empty(len(global_indices), dtype=np.int64)
    global_rank[global_indices] = np.arange(len(global_indices), dtype=np.int64)
    # The episode component of a hierarchical key is exactly the global key.
    hierarchical_indices = np.lexsort((
        global_rank, symbol_rank[universe.symbol_ids],
    )).astype(np.int64, copy=False)
    # Both are compositions of argsort permutations; an additional O(N log N)
    # uniqueness sort here would duplicate work on every production replicate.
    return CompleteOrders(
        global_indices, hierarchical_indices, symbol_priority_order,
        {
            "episode_hashes": len(universe.episode_ids),
            "symbol_hashes": len(universe.symbols),
            "global_orders": 1, "hierarchical_orders": 1,
            "per_query_full_universe_hashes": 0,
            "per_query_full_universe_sorts": 0,
        },
    )


def _eligible(universe: CandidateUniverse, query: QueryRisk, index: int) -> bool:
    cutoff = int(universe.cutoffs[index]); symbol = int(universe.symbol_ids[index])
    return cutoff <= query.latest_ns and not (
        symbol == query.symbol_id and cutoff >= query.start_ns
    )


def select_from_order(
    universe: CandidateUniverse, query: QueryRisk, ordered: Iterable[int],
    *, config: SelectionConfig = SelectionConfig(),
) -> tuple[int, ...]:
    """Filter one common order by one exact causal risk set and greedily select."""

    selected: list[int] = []
    by_symbol: Counter[int] = Counter()
    for raw_index in ordered:
        index = int(raw_index)
        if not _eligible(universe, query, index):
            continue
        symbol = int(universe.symbol_ids[index])
        if by_symbol[symbol] >= config.max_per_symbol:
            continue
        ordinal = int(universe.local_ordinals[index])
        if any(
            int(universe.symbol_ids[prior]) == symbol
            and abs(ordinal - int(universe.local_ordinals[prior])) * 5 <= 251
            for prior in selected
        ):
            continue
        selected.append(index); by_symbol[symbol] += 1
        if len(selected) == config.top_k:
            return tuple(selected)
    raise SharedPriorityRunError(f"risk set cannot satisfy selector: {query.query_id}")


def _select_without_own_exclusion(
    universe: CandidateUniverse, *, latest_ns: int, ordered: Iterable[int],
    config: SelectionConfig,
) -> tuple[int, ...]:
    selected: list[int] = []; by_symbol: Counter[int] = Counter()
    for raw_index in ordered:
        index = int(raw_index)
        if int(universe.cutoffs[index]) > latest_ns:
            continue
        symbol = int(universe.symbol_ids[index]); ordinal = int(universe.local_ordinals[index])
        if by_symbol[symbol] >= config.max_per_symbol or any(
            int(universe.symbol_ids[prior]) == symbol
            and abs(ordinal - int(universe.local_ordinals[prior])) * 5 <= 251
            for prior in selected
        ):
            continue
        selected.append(index); by_symbol[symbol] += 1
        if len(selected) == config.top_k:
            return tuple(selected)
    raise SharedPriorityRunError("date-only risk set cannot satisfy selector")


def select_global_queries(
    universe: CandidateUniverse, queries: Sequence[QueryRisk], order: np.ndarray,
    *, config: SelectionConfig,
) -> tuple[tuple[int, ...], ...]:
    """Reuse a date-only selection unless its result contains the query symbol."""

    by_latest: dict[int, tuple[int, ...]] = {}
    output = []
    for query in queries:
        base = by_latest.get(query.latest_ns)
        if base is None:
            base = _select_without_own_exclusion(
                universe, latest_ns=query.latest_ns, ordered=order, config=config,
            )
            by_latest[query.latest_ns] = base
        if any(int(universe.symbol_ids[index]) == query.symbol_id for index in base):
            output.append(select_from_order(universe, query, order, config=config))
        else:
            output.append(base)
    return tuple(output)


def select_hierarchical_queries(
    universe: CandidateUniverse, queries: Sequence[QueryRisk], orders: CompleteOrders,
    *, config: SelectionConfig,
) -> tuple[tuple[int, ...], ...]:
    """Exploit exact symbol separability while retaining the complete flat order."""

    counts = universe.stops - universe.starts
    rank_stops = np.cumsum(counts[orders.symbol_priority_order], dtype=np.int64)
    rank_starts = rank_stops - counts[orders.symbol_priority_order]
    common: dict[tuple[int, int], tuple[int, ...]] = {}
    own: dict[tuple[int, int, int], tuple[int, ...]] = {}

    def symbol_choices(query: QueryRisk, rank: int, symbol_id: int) -> tuple[int, ...]:
        is_own = symbol_id == query.symbol_id
        key: tuple[int, ...] = ((query.latest_ns, symbol_id, query.start_ns)
                                if is_own else (query.latest_ns, symbol_id))
        cache = own if is_own else common
        if key in cache:
            return cache[key]
        selected: list[int] = []
        block = orders.hierarchical_indices[int(rank_starts[rank]):int(rank_stops[rank])]
        for raw_index in block:
            index = int(raw_index); cutoff = int(universe.cutoffs[index])
            if cutoff > query.latest_ns or (is_own and cutoff >= query.start_ns):
                continue
            ordinal = int(universe.local_ordinals[index])
            if any(abs(ordinal - int(universe.local_ordinals[prior])) * 5 <= 251
                   for prior in selected):
                continue
            selected.append(index)
            if len(selected) == config.max_per_symbol:
                break
        cache[key] = tuple(selected)
        return cache[key]

    output = []
    for query in queries:
        selected: list[int] = []
        for rank, raw_symbol in enumerate(orders.symbol_priority_order):
            selected.extend(symbol_choices(query, rank, int(raw_symbol)))
            if len(selected) >= config.top_k:
                selected = selected[:config.top_k]
                break
        require(len(selected) == config.top_k,
                f"hierarchical risk set cannot satisfy selector: {query.query_id}")
        output.append(tuple(selected))
    return tuple(output)


def _selection_digest(
    universe: CandidateUniverse, query_ids: Sequence[str], selections: Sequence[Sequence[int]],
    *, family: str,
) -> str:
    require(len(query_ids) == len(selections), "selection digest rows differ")
    digest = sha256(f"{SCHEMA}/selection/{family}/v1".encode("ascii"))
    digest.update(len(query_ids).to_bytes(4, "big"))
    for query_id, selected in zip(query_ids, selections, strict=True):
        encoded = query_id.encode("utf-8")
        require(len(encoded) <= 0xFFFF, "query identifier too long")
        digest.update(len(encoded).to_bytes(2, "big")); digest.update(encoded)
        digest.update(len(selected).to_bytes(2, "big"))
        for index in selected:
            digest.update(bytes(universe.episode_ids[int(index)]))
    return digest.hexdigest()


def validate_complete_selections(
    universe: CandidateUniverse, selections: Sequence[Sequence[int]], *, top_k: int,
) -> None:
    """Refuse partial, duplicated, or out-of-universe selections before hashing."""

    require(bool(selections), "selection rows are empty")
    for selected in selections:
        indices = tuple(int(value) for value in selected)
        require(len(indices) == top_k, "selection is not exact top-k")
        require(len(set(indices)) == top_k, "selection contains a duplicate episode")
        require(all(0 <= value < len(universe.episode_ids) for value in indices),
                "selection index outside universe")


def metrics_for_selections(
    universe: CandidateUniverse, selections: Sequence[Sequence[int]],
) -> dict[str, float | int]:
    require(len(selections) >= 2 and all(len(row) > 0 for row in selections),
            "complete nonempty query selections required")
    episode_counts: Counter[int] = Counter()
    symbol_counts: Counter[int] = Counter()
    repeated = twice = thrice = 0
    for selected in selections:
        local = Counter(int(universe.symbol_ids[int(index)]) for index in selected)
        repeated += any(value > 1 for value in local.values())
        twice += sum(value == 2 for value in local.values())
        thrice += sum(value == 3 for value in local.values())
        for raw_index in selected:
            index = int(raw_index)
            episode_counts[index] += 1
            symbol_counts[int(universe.symbol_ids[index])] += 1
    pairs = sum(value * (value - 1) // 2 for value in episode_counts.values())
    result = concentration_metrics(
        list(episode_counts.values()), list(symbol_counts.values()),
        episode_population=len(universe.episode_ids), symbol_population=len(universe.symbols),
    )
    queries = len(selections)
    result.update({
        "query_any_repeated_symbol_fraction": repeated / queries,
        "repeated_symbol_twice_per_query": twice / queries,
        "repeated_symbol_thrice_per_query": thrice / queries,
        "mean_pairwise_query_episode_overlap": pairs / (queries * (queries - 1) / 2),
    })
    require(set(result) == set(joint.METRICS), "metric closure differs")
    return result


def compute_replicate(
    universe: CandidateUniverse, queries: Sequence[QueryRisk], *, seed: int,
    replicate: int, workers: int = 1, config: SelectionConfig = SelectionConfig(),
    episode_hashes: np.ndarray | None = None,
    symbol_hashes: np.ndarray | None = None,
    tie_orders: TieOrders | None = None,
) -> dict[str, Any]:
    """Compute one deterministic replicate from two shared complete orders."""

    ordered_queries = tuple(sorted(queries))
    require(len(ordered_queries) == len({row.query_id for row in ordered_queries}),
            "duplicate query identity")
    orders = build_complete_orders(
        universe, seed=seed, replicate=replicate, workers=workers,
        episode_hashes=episode_hashes, symbol_hashes=symbol_hashes,
        tie_orders=tie_orders,
    )
    global_selected = select_global_queries(
        universe, ordered_queries, orders.global_indices, config=config,
    )
    hierarchical_selected = select_hierarchical_queries(
        universe, ordered_queries, orders, config=config,
    )
    validate_complete_selections(universe, global_selected, top_k=config.top_k)
    validate_complete_selections(universe, hierarchical_selected, top_k=config.top_k)
    query_ids = tuple(row.query_id for row in ordered_queries)
    counters = dict(orders.counters)
    counters.update({
        "query_risk_sets": len(ordered_queries),
        "global_risk_set_filters": len(ordered_queries),
        "hierarchical_risk_set_filters": len(ordered_queries),
        "total_risk_set_filters": 2 * len(ordered_queries),
    })
    result = {
        "replicate": replicate,
        "global": {
            "metrics": metrics_for_selections(universe, global_selected),
            "selection_digest": _selection_digest(
                universe, query_ids, global_selected, family="global",
            ),
        },
        "hierarchical": {
            "metrics": metrics_for_selections(universe, hierarchical_selected),
            "selection_digest": _selection_digest(
                universe, query_ids, hierarchical_selected, family="hierarchical",
            ),
        },
        "work_counters": counters,
        "selection_completeness": {"global": True, "hierarchical": True,
                                   "exact_top_k": config.top_k},
    }
    return result


def _validate_queries(universe: CandidateUniverse, queries: Sequence[QueryRisk]) -> None:
    require(queries and tuple(queries) == tuple(sorted(queries)), "queries must be sorted")
    require(len({row.query_id for row in queries}) == len(queries), "duplicate queries")
    all_cutoffs = np.sort(universe.cutoffs, kind="stable")
    for query in queries:
        require(0 <= query.symbol_id < len(universe.symbols), "query symbol outside universe")
        require(query.start_ns <= query.latest_ns, "query interval differs")
        total = int(np.searchsorted(all_cutoffs, query.latest_ns, side="right"))
        first, stop = int(universe.starts[query.symbol_id]), int(universe.stops[query.symbol_id])
        own = universe.cutoffs[first:stop]
        removed = int(np.searchsorted(own, query.latest_ns, side="right")
                      - np.searchsorted(own, query.start_ns, side="left"))
        eligible = total - removed
        require(eligible == query.eligible_count, f"risk-set count differs: {query.query_id}")


def _actual_from_cases(
    repository: Path, prereg: Mapping[str, Any], universe: CandidateUniverse,
) -> tuple[tuple[QueryRisk, ...], dict[str, float | int], str]:
    manifest = prereg["authorities"]["case_manifest"]
    manifest_by_path = {str(row["path"]): row for row in manifest}
    require(len(manifest_by_path) == len(manifest) == 3270, "case manifest differs")
    symbol_ids = {symbol: index for index, symbol in enumerate(universe.symbols)}
    universe_id_order = np.argsort(universe.episode_ids, kind="stable")
    sorted_universe_ids = universe.episode_ids[universe_id_order]
    queries: list[QueryRisk] = []; observed: list[tuple[int, ...]] = []
    identity_rows: list[dict[str, Any]] = []
    for relative in sorted(manifest_by_path):
        row = manifest_by_path[relative]; path = repository / relative
        require(file_sha(path) == row["sha256"] and path.stat().st_size == row["bytes"],
                f"case authority changed: {relative}")
        case = load_json(path)
        require(case.get("gate_passed") is True and case.get("real_forward_outcomes_accessed") is False,
                f"invalid outcome-blind case: {relative}")
        query_id = str(case["query_episode_id"]); symbol = str(case["query_symbol"])
        require(symbol in symbol_ids, f"query symbol outside universe: {query_id}")
        query = QueryRisk(
            query_id=query_id, symbol_id=symbol_ids[symbol],
            start_ns=int(np.datetime64(case["query_start"]).astype("datetime64[ns]").astype(np.int64)),
            latest_ns=int(np.datetime64(case["latest_eligible_cutoff"]).astype("datetime64[ns]").astype(np.int64)),
            eligible_count=int(case["certificate"]["eligible_candidates"]),
        )
        matches = case.get("matches")
        require(isinstance(matches, list) and len(matches) == 20, "case top20 differs")
        distances = [float(match["total_distance"]) for match in matches]
        identities = [str(match["episode_id"]) for match in matches]
        require(all(math.isfinite(value) and value >= 0 for value in distances)
                and list(zip(distances, identities)) == sorted(zip(distances, identities)),
                "observed match ordering differs")
        indices: list[int] = []
        for rank, match in enumerate(matches, 1):
            episode_id = str(match["episode_id"])
            try:
                raw_id = np.void(bytes.fromhex(episode_id))
            except ValueError as error:
                raise SharedPriorityRunError(f"invalid observed episode ID: {episode_id}") from error
            position = int(np.searchsorted(sorted_universe_ids, raw_id))
            require(position < len(sorted_universe_ids) and sorted_universe_ids[position] == raw_id,
                    f"observed episode outside universe: {episode_id}")
            index = int(universe_id_order[position])
            require(universe.symbols[int(universe.symbol_ids[index])] == str(match["symbol"]),
                    "observed episode symbol differs")
            require(int(universe.cutoffs[index]) == int(np.datetime64(match["cutoff"]).astype("datetime64[ns]").astype(np.int64)),
                    "observed episode cutoff differs")
            indices.append(index)
            identity_rows.append({"query_id": query_id, "episode_id": episode_id, "rank": rank})
        require(len(set(indices)) == 20, "duplicate observed episode")
        require(all(_eligible(universe, query, index) for index in indices),
                "observed match outside exact risk set")
        by_symbol: Counter[int] = Counter(int(universe.symbol_ids[index]) for index in indices)
        require(max(by_symbol.values()) <= 3, "observed match violates cap three")
        for left, first_index in enumerate(indices):
            for second_index in indices[left + 1:]:
                if int(universe.symbol_ids[first_index]) == int(universe.symbol_ids[second_index]):
                    require(abs(int(universe.local_ordinals[first_index])
                                - int(universe.local_ordinals[second_index])) * 5 > 251,
                            "observed match violates inclusive overlap")
        queries.append(query)
        observed.append(tuple(indices))
    order = np.argsort(np.asarray([row.query_id for row in queries], dtype="U"), kind="stable")
    sorted_queries = tuple(queries[int(i)] for i in order)
    sorted_observed = tuple(observed[int(i)] for i in order)
    _validate_queries(universe, sorted_queries)
    require([row.query_id for row in sorted_queries] == prereg["authorities"]["population"]["query_ids"],
            "query identity population differs")
    validate_complete_selections(universe, sorted_observed, top_k=20)
    return sorted_queries, metrics_for_selections(universe, sorted_observed), stable_hash(identity_rows)


def _packed_manifest_records(prereg: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = prereg["authorities"]["reuse_authority_manifest"]
    return {str(row["path"]): row for row in rows if "/packed-bound-full/store/generations/" in str(row["path"])}


def load_production_inputs(
    repository: Path, prereg: Mapping[str, Any],
) -> tuple[CandidateUniverse, tuple[QueryRisk, ...], dict[str, float | int], str]:
    """Reconstruct exact metadata without opening an R1-A scientific result."""

    r1a_path = repository / R1A_PREREG
    expected_r1a_sha = prereg["authorities"]["authority_file_sha256"][str(R1A_PREREG)]
    require(file_sha(r1a_path) == expected_r1a_sha, "R1-A preregistration changed")
    r1a = load_json(r1a_path)
    require(r1a.get("preregistration_digest") == stable_hash({
        key: value for key, value in r1a.items() if key != "preregistration_digest"
    }), "R1-A preregistration digest differs")
    inputs = r1a["inputs"]; generation = str(inputs["packed_generation_id"])
    root = repository / PACKED_DURABLE
    generation_root = root / "generations" / generation
    require(generation_root.is_dir() and not generation_root.is_symlink(), "packed generation missing")
    records = _packed_manifest_records(prereg)
    manifest_relative = (generation_root / "manifest.json").relative_to(repository).as_posix()
    require(manifest_relative in records, "packed manifest absent from H1 authority")
    require(file_sha(generation_root / "manifest.json") == records[manifest_relative]["sha256"],
            "packed manifest changed")
    manifest = load_json(generation_root / "manifest.json")
    for field, manifest_field in (("rows_file", "rows_sha256"), ("overflow_file", "overflow_sha256")):
        packed_path = generation_root / str(manifest[field])
        require_regular_beneath(repository, packed_path)
        relative = packed_path.relative_to(repository).as_posix()
        require(relative in records and records[relative]["sha256"] == manifest[manifest_field]
                and records[relative]["bytes"] == (generation_root / str(manifest[field])).stat().st_size,
                f"packed {field} authority differs")
    loaded = load_packed_generation(
        root, generation, expected_provenance_digest=str(inputs["packed_provenance_digest"]),
        verify_content=True, validate_records=True,
    )
    require(loaded.manifest.get("real_forward_outcomes_accessed") is False, "packed source not outcome blind")
    total = len(loaded.rows) + len(loaded.overflow)
    compact = np.empty(total, dtype=[("episode_id", "V12"), ("cutoff", "<i8"), ("symbol", "<u4")])
    cursor = 0
    for source in (loaded.rows, loaded.overflow):
        for first in range(0, len(source), 1 << 17):
            block = source[first:first + (1 << 17)]; stop = cursor + len(block)
            compact["episode_id"][cursor:stop] = block["episode_id"]
            compact["cutoff"][cursor:stop] = block["cutoff_ns"]
            compact["symbol"][cursor:stop] = block["symbol_id"]
            cursor = stop
    order = np.lexsort((compact["episode_id"], compact["cutoff"], compact["symbol"]))
    compact = compact[order]
    require(len(np.unique(compact["episode_id"])) == total, "packed episode IDs duplicate")
    counts = np.bincount(compact["symbol"], minlength=len(loaded.symbols))
    stops = np.cumsum(counts, dtype=np.int64); starts = stops - counts
    local = np.empty(total, dtype=np.int64)
    for symbol_id, (first, stop) in enumerate(zip(starts, stops, strict=True)):
        first = int(first); stop = int(stop)
        require(np.all(compact["symbol"][first:stop] == symbol_id), "symbol ordering differs")
        require(stop - first < 2 or np.all(np.diff(compact["cutoff"][first:stop]) > 0),
                "symbol cutoffs not strictly increasing")
        local[first:stop] = np.arange(stop - first, dtype=np.int64)
        prefix_rows = int(loaded.manifest["provenance"]["source_prefixes"][loaded.symbols[symbol_id]]["rows"])
        require(stop - first == max((prefix_rows - 252) // 5 + 1, 0), "stride accounting differs")
    universe = CandidateUniverse(
        compact["episode_id"], compact["cutoff"], compact["symbol"], local,
        loaded.symbols, starts, stops,
    )
    expected_work = joint.specification()["b005"]["work_counters_per_replicate"]
    require(total == expected_work["episode_hashes"]
            and len(loaded.symbols) == expected_work["symbol_hashes"],
            "candidate inventory differs")
    queries, actual, identity_digest = _actual_from_cases(repository, prereg, universe)
    return universe, queries, actual, identity_digest


def _binding(
    prereg: Mapping[str, Any], universe: CandidateUniverse, queries: Sequence[QueryRisk],
    *, h1_commit: str, retrieval_identity_digest: str, replicates: int, shard_size: int,
) -> dict[str, Any]:
    ids_digest = sha256(b"".join(bytes(value) for value in universe.episode_ids)).hexdigest()
    query_rows = [{"query_id": row.query_id, "symbol_id": row.symbol_id,
                   "start_ns": row.start_ns, "latest_ns": row.latest_ns,
                   "eligible_count": row.eligible_count} for row in queries]
    state = {
        "schema_version": BINDING_SCHEMA,
        "h1_commit": h1_commit,
        "joint_preregistration_digest": prereg["preregistration_digest"],
        "runtime_sha256": prereg["runtime_sha256"],
        "episode_ids_sha256": ids_digest,
        "symbols_digest": stable_hash(list(universe.symbols)),
        "queries_digest": stable_hash(query_rows),
        "candidate_episodes": len(universe.episode_ids),
        "candidate_symbols": len(universe.symbols),
        "queries": len(queries), "replicates": replicates,
        "replicate_indices": list(range(replicates)), "shard_size": shard_size,
        "shard_ranges": [list(value) for value in _shard_ranges(replicates, shard_size)],
        "seed": int(prereg["specification"]["b005"]["seed"]),
        "priority_domains": {"episode": EPISODE_DOMAIN, "symbol": SYMBOL_DOMAIN},
        "metric_directions": prereg["specification"]["b005"]["metrics"],
        "top_k": int(prereg["specification"]["b005"]["top_k"]),
        "max_per_symbol": 3,
        "real_forward_outcomes_accessed": False,
        "retrieval_identity_digest": retrieval_identity_digest,
    }
    return {**state, "binding_digest": stable_hash(state)}


def _shard_ranges(replicates: int, shard_size: int) -> tuple[tuple[int, int], ...]:
    require(type(replicates) is int and replicates >= 1, "replicates must be positive")
    require(type(shard_size) is int and shard_size >= 1, "shard size must be positive")
    return tuple((first, min(first + shard_size, replicates)) for first in range(0, replicates, shard_size))


def _validate_shard(
    path: Path, *, binding_digest: str, first: int, stop: int,
    expected_counters: Mapping[str, int] | None = None, expected_top_k: int | None = None,
) -> dict[str, Any]:
    require(not path.is_symlink(), f"symlink shard forbidden: {path}")
    value = load_json(path)
    require(set(value) == {"schema_version", "binding_digest", "replicate_range", "rows", "shard_digest"},
            f"shard field closure differs: {path}")
    require(value["schema_version"] == SHARD_SCHEMA and value["binding_digest"] == binding_digest,
            f"shard binding differs: {path}")
    require(value["replicate_range"] == [first, stop], f"shard range differs: {path}")
    require([row.get("replicate") for row in value["rows"]] == list(range(first, stop)),
            f"shard replicate accounting differs: {path}")
    for row in value["rows"]:
        require(set(row) == {"replicate", "global", "hierarchical", "work_counters",
                             "selection_completeness"}, f"replicate field closure differs: {path}")
        require(type(row["replicate"]) is int, f"replicate type differs: {path}")
        require(row["selection_completeness"] == {
            "global": True, "hierarchical": True, "exact_top_k": expected_top_k,
        } if expected_top_k is not None else isinstance(row["selection_completeness"], dict),
                f"selection completeness differs: {path}")
        if expected_counters is not None:
            require(row["work_counters"] == expected_counters, f"work counters differ: {path}")
        for family in ("global", "hierarchical"):
            payload = row[family]
            require(isinstance(payload, dict) and set(payload) == {"metrics", "selection_digest"},
                    f"null-family field closure differs: {path}")
            require(set(payload["metrics"]) == set(joint.METRICS), f"metric closure differs: {path}")
            require(all(type(value) in {int, float} and math.isfinite(float(value))
                        for value in payload["metrics"].values()), f"non-finite metric: {path}")
            digest = payload["selection_digest"]
            require(type(digest) is str and len(digest) == 64
                    and all(token in "0123456789abcdef" for token in digest),
                    f"selection digest malformed: {path}")
    state = {key: item for key, item in value.items() if key != "shard_digest"}
    require(value["shard_digest"] == stable_hash(state), f"shard digest differs: {path}")
    return value


def _summaries(
    actual: Mapping[str, float | int], rows: Sequence[Mapping[str, Any]],
    directions: Mapping[str, str], family: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in joint.METRICS:
        observed = actual[metric]; values = [float(row[family]["metrics"][metric]) for row in rows]
        direction = directions[metric]
        require(direction in {"higher_is_more_concentrated", "lower_is_more_concentrated"},
                "metric direction differs")
        tail = sum(value >= float(observed) for value in values) if direction.startswith("higher") \
            else sum(value <= float(observed) for value in values)
        result[metric] = {
            "observed": observed, "null_mean": math.fsum(values) / len(values),
            "null_p05": nearest_rank(values, .05), "null_p50": nearest_rank(values, .50),
            "null_p95": nearest_rank(values, .95), "null_p99": nearest_rank(values, .99),
            "concentration_direction": direction, "inclusive_tail_count": tail,
            "concentration_tail_monte_carlo_p": (1 + tail) / (len(values) + 1),
        }
    return result


def run_experiment(
    output: Path, *, binding: Mapping[str, Any], universe: CandidateUniverse,
    queries: Sequence[QueryRisk], actual: Mapping[str, float | int], workers: int,
    stop_after_shards: int | None = None,
) -> dict[str, Any]:
    """Run or resume exact create-only shards; useful with synthetic fixtures."""

    require(binding.get("binding_digest") == stable_hash({
        key: value for key, value in binding.items() if key != "binding_digest"
    }), "binding digest differs")
    require(binding.get("real_forward_outcomes_accessed") is False, "outcome-blind binding required")
    require(binding.get("priority_domains") == {"episode": EPISODE_DOMAIN, "symbol": SYMBOL_DOMAIN},
            "priority domains differ")
    require(binding.get("candidate_episodes") == len(universe.episode_ids)
            and binding.get("candidate_symbols") == len(universe.symbols)
            and binding.get("queries") == len(queries), "bound inventory differs")
    require(binding.get("replicate_indices") == list(range(int(binding["replicates"]))),
            "bound replicate indices differ")
    require(binding.get("shard_ranges") == [list(value) for value in _shard_ranges(
        int(binding["replicates"]), int(binding["shard_size"]),
    )], "bound shard ranges differ")
    require(set(actual) == set(joint.METRICS), "observed metric closure differs")
    require(not output.is_symlink(), "symlink output root forbidden")
    output.mkdir(parents=True, exist_ok=True)
    binding_path = output / "BINDING.json"
    if binding_path.exists() or binding_path.is_symlink():
        require(not binding_path.is_symlink(), "symlink run binding forbidden")
        require(load_json(binding_path) == binding, "existing run binding differs")
    else:
        atomic_json(binding_path, binding)
    shards = output / "shards"
    require(not shards.is_symlink(), "symlink shard root forbidden")
    shards.mkdir(exist_ok=True)
    allowed_names = {f"shard-{first:04d}-{stop:04d}.json" for first, stop in _shard_ranges(
        int(binding["replicates"]), int(binding["shard_size"]),
    )}
    temporary_pattern = re.compile(r"^\.(shard-\d{4}-\d{4}\.json)\.tmp-\d+-[0-9a-f]{32}$")
    for path in tuple(shards.iterdir()):
        match = temporary_pattern.fullmatch(path.name)
        if match is not None and match.group(1) in allowed_names:
            require(path.is_file() and not path.is_symlink(),
                    f"unsafe unpublished shard staging: {path.name}")
            path.unlink()
    extras = {path.name for path in shards.iterdir()} - allowed_names
    require(not extras, f"unexpected or partial shard artifacts: {sorted(extras)}")
    started_wall = perf_counter(); completed_now = 0
    self_before = resource.getrusage(resource.RUSAGE_SELF)
    children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    seed = int(binding["seed"]); config = SelectionConfig(
        top_k=int(binding["top_k"]), max_per_symbol=int(binding["max_per_symbol"]),
    )
    expected_per = {
        "episode_hashes": len(universe.episode_ids), "symbol_hashes": len(universe.symbols),
        "global_orders": 1, "hierarchical_orders": 1,
        "per_query_full_universe_hashes": 0, "per_query_full_universe_sorts": 0,
        "query_risk_sets": len(queries),
        "global_risk_set_filters": len(queries),
        "hierarchical_risk_set_filters": len(queries),
        "total_risk_set_filters": 2 * len(queries),
    }
    tie_orders = TieOrders(
        np.argsort(universe.episode_ids, kind="stable").astype(np.int64, copy=False),
        np.argsort(np.asarray(universe.symbols, dtype="U"), kind="stable").astype(np.int64, copy=False),
    )
    ranges = _shard_ranges(int(binding["replicates"]), int(binding["shard_size"]))
    missing: list[tuple[int, int]] = []
    for first, stop in ranges:
        path = shards / f"shard-{first:04d}-{stop:04d}.json"
        if path.exists() or path.is_symlink():
            _validate_shard(path, binding_digest=binding["binding_digest"], first=first, stop=stop,
                            expected_counters=expected_per, expected_top_k=config.top_k)
        else:
            missing.append((first, stop))
    hasher_context = (ForkEpisodeHasher(universe.episode_ids, workers)
                      if missing and workers > 1 else nullcontext(None))
    with hasher_context as fork_hasher:
        for first, stop in missing:
            path = shards / f"shard-{first:04d}-{stop:04d}.json"
            rows = []
            for replicate in range(first, stop):
                episode_hashes = None if fork_hasher is None else fork_hasher.hashes(
                    seed=seed, replicate=replicate,
                )
                # Only 11,584 symbol entities exist in production; hashing them
                # serially avoids an extra interprocess transfer and is still
                # exactly once per symbol and replicate.
                symbol_hashes = priority_array(
                    tuple(value.encode("utf-8") for value in universe.symbols),
                    domain=SYMBOL_DOMAIN, seed=seed, replicate=replicate, workers=1,
                )
                rows.append(compute_replicate(
                    universe, queries, seed=seed, replicate=replicate,
                    workers=1 if episode_hashes is not None else workers, config=config,
                    episode_hashes=episode_hashes, symbol_hashes=symbol_hashes,
                    tie_orders=tie_orders,
                ))
            state = {"schema_version": SHARD_SCHEMA, "binding_digest": binding["binding_digest"],
                     "replicate_range": [first, stop], "rows": rows}
            atomic_json(path, {**state, "shard_digest": stable_hash(state)})
            completed_now += 1
            if stop_after_shards is not None and completed_now >= stop_after_shards:
                return {"status": "incomplete", "completed_shards_this_invocation": completed_now}
    all_rows = []
    shard_digests = []
    for first, stop in ranges:
        value = _validate_shard(
            shards / f"shard-{first:04d}-{stop:04d}.json",
            binding_digest=binding["binding_digest"], first=first, stop=stop,
            expected_counters=expected_per, expected_top_k=config.top_k,
        )
        all_rows.extend(value["rows"]); shard_digests.append(value["shard_digest"])
    require([row["replicate"] for row in all_rows] == list(range(int(binding["replicates"]))),
            "aggregate replicate accounting differs")
    directions = binding["metric_directions"]
    scientific = {
        "actual": dict(actual),
        "global_comparison": _summaries(actual, all_rows, directions, "global"),
        "hierarchical_comparison": _summaries(actual, all_rows, directions, "hierarchical"),
        "replicates_digest": stable_hash(all_rows), "shard_digests": shard_digests,
    }
    require(all(row["work_counters"] == expected_per for row in all_rows), "work counters differ")
    gates = {"h1_validated": True, "outcomes_excluded": True,
             "shared_episode_priority_both_nulls": True,
             "one_complete_order_per_family_per_replicate": True,
             "zero_per_query_full_universe_hash_or_sort": True,
             "replicate_accounting_complete": True,
             "all_15_metrics_complete": True,
             "every_selection_exact_top20_unique": (
                 int(binding["top_k"]) == 20 and all(
                     row["selection_completeness"] == {
                         "global": True, "hierarchical": True, "exact_top_k": 20,
                     } for row in all_rows)
             )}
    require(all(gates.values()), "one or more B0-05 execution gates failed")
    state = {
        "schema_version": SCHEMA, "status": "diagnostic_complete_pending_independent_verification",
        "passed": True, "binding_digest": binding["binding_digest"],
        "joint_preregistration_digest": binding["joint_preregistration_digest"],
        "inventory": {"queries": len(queries), "candidate_episodes": len(universe.episode_ids),
                      "candidate_symbols": len(universe.symbols), "replicates": len(all_rows),
                      "null_families": 2},
        "scientific": scientific, "scientific_digest": sha256(_canonical_scientific_bytes(scientific)).hexdigest(),
        "work_counters_per_replicate": expected_per,
        "work_counters_total": {key: value * len(all_rows) for key, value in expected_per.items()},
        "gates": gates,
        "claims": {"adequacy_labels_authorized": False, "predictive_claim_authorized": False,
                   "production_promotion_authorized": False,
                   "real_forward_outcomes_accessed": False},
    }
    deterministic = {**state, "result_digest": stable_hash(state)}
    self_after = resource.getrusage(resource.RUSAGE_SELF)
    children_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    self_cpu = ((self_after.ru_utime + self_after.ru_stime)
                - (self_before.ru_utime + self_before.ru_stime))
    children_cpu = ((children_after.ru_utime + children_after.ru_stime)
                    - (children_before.ru_utime + children_before.ru_stime))
    result = {**deterministic, "performance": {
        "workers": workers, "wall_seconds_this_invocation": perf_counter() - started_wall,
        "cpu_seconds_this_invocation": self_cpu + children_cpu,
        "self_cpu_seconds_this_invocation": self_cpu,
        "child_cpu_seconds_this_invocation": children_cpu,
        "self_maximum_resident_set_kib": int(self_after.ru_maxrss),
        "child_maximum_resident_set_kib": int(children_after.ru_maxrss),
        "completed_shards_this_invocation": completed_now,
        "reused_shards": len(shard_digests) - completed_now,
    }}
    result_path = output / "RESULT.json"
    if result_path.exists() or result_path.is_symlink():
        require(not result_path.is_symlink(), "symlink result forbidden")
        prior = load_json(result_path)
        require({key: value for key, value in prior.items() if key != "performance"} == deterministic,
                "existing result differs")
        result = prior
    else:
        atomic_json(result_path, result)
    html = """<!doctype html><html lang='en'><head><meta charset='utf-8'><title>B0-05 shared-priority sensitivity</title></head><body>""" \
        f"<h1>B0-05: {escape(result['status'])}</h1><p>Result <code>{result['result_digest']}</code>.</p>" \
        "<p>Outcome-blind global-episode and hierarchical-symbol shared-priority sensitivities. No adequacy, prediction, ranking, or trading claim is authorized before independent verification and later stages.</p></body></html>\n"
    report = output / "report.html"
    if report.exists() or report.is_symlink():
        require(not report.is_symlink(), "symlink report forbidden")
        require(report.read_text(encoding="utf-8") == html, "existing report differs")
    else:
        temporary = report.parent / f".{report.name}.tmp-{os.getpid()}-{uuid4().hex}"
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(html); handle.flush(); os.fsync(handle.fileno())
            os.link(temporary, report)
            directory = os.open(report.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except FileExistsError as error:
            raise SharedPriorityRunError("create-only report exists") from error
        finally:
            temporary.unlink(missing_ok=True)
    return result


def execute(repository: Path, *, workers: int = 12) -> dict[str, Any]:
    repository = repository.resolve()
    require(workers == 12, "production B0-05 requires the frozen 12 workers")
    h1 = joint.validate_h1(repository)
    prereg = load_json(repository / joint.PREREGISTRATION)
    require(h1 == joint.git(repository, "rev-parse", "HEAD"), "H1 changed after validation")
    universe, queries, actual, identity_digest = load_production_inputs(repository, prereg)
    spec = prereg["specification"]["b005"]
    require(spec == joint.specification()["b005"], "B0-05 specification differs")
    require(spec["shard_replicates"] == DEFAULT_SHARD_SIZE and spec["shards"] == 64
            and spec["production_workers"] == workers, "production shard/worker contract differs")
    require(spec["work_counters_per_replicate"] == {
        "episode_hashes": len(universe.episode_ids), "symbol_hashes": len(universe.symbols),
        "global_orders": 1, "hierarchical_orders": 1,
        "per_query_full_universe_hashes": 0, "per_query_full_universe_sorts": 0,
        "query_risk_sets": len(queries),
        "global_risk_set_filters": len(queries),
        "hierarchical_risk_set_filters": len(queries),
        "total_risk_set_filters": 2 * len(queries),
    }, "production work-counter contract differs")
    binding = _binding(
        prereg, universe, queries, h1_commit=h1,
        retrieval_identity_digest=identity_digest, replicates=int(spec["replicates"]),
        shard_size=DEFAULT_SHARD_SIZE,
    )
    return run_experiment(
        repository / OUTPUT, binding=binding, universe=universe, queries=queries,
        actual=actual, workers=workers,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(argv)
    print(json.dumps(execute(args.repository, workers=args.workers), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
