"""Exact, outcome-blind shared-priority null-selection primitives.

This module is deliberately small and data-source agnostic.  It defines the
randomness used by the B0-05 global and hierarchical concentration nulls, but
does not know about prices, outcomes, observations, or a particular market.

The priority construction is part of the statistical contract.  In particular
we never reduce a SHA-256 value to a platform-sized integer: the complete
256-bit digest is used, and an identifier is an explicit deterministic tie
breaker.  A 16-bit bucket implementation is provided solely as a small
synthetic/reference oracle; it produces the identical order as a full sort.
Production must precompute the two complete-universe orders once per replicate
and filter those orders for every query risk set.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Callable, Iterable, Literal, Mapping, Sequence


class SharedPriorityError(ValueError):
    """Raised when a null draw cannot meet its frozen selection contract."""


EPISODE_DOMAIN = "market-analogues/r1b/b0-05/h-episode/v1"
SYMBOL_DOMAIN = "market-analogues/r1b/b0-05/h-symbol/v1"
_U64_MAX = (1 << 64) - 1
_U32_MAX = (1 << 32) - 1


@dataclass(frozen=True, order=True)
class SharedPriorityCandidate:
    """Minimal causal geometry needed by the shared-priority selector.

    ``episode_id`` and ``symbol_id`` are opaque, canonical UTF-8 identifiers.
    The caller must provide the same exact spellings in all queries.  Intervals
    use integer coordinates on one caller-defined axis.  Production B0-05 uses
    contiguous-window session coordinates ``[5*i, 5*i+251]``; tests may use
    smaller synthetic coordinates.
    """

    episode_id: str
    symbol_id: str
    start_coordinate: int
    cutoff_coordinate: int


@dataclass(frozen=True)
class SelectionConfig:
    """Frozen production diversity rule, configurable for synthetic tests."""

    top_k: int = 20
    max_per_symbol: int = 3

    def __post_init__(self) -> None:
        if type(self.top_k) is not int or self.top_k < 1:
            raise SharedPriorityError("top_k must be a positive integer")
        if type(self.max_per_symbol) is not int or self.max_per_symbol < 1:
            raise SharedPriorityError("max_per_symbol must be a positive integer")


@dataclass(frozen=True)
class QuerySelection:
    """Both preregistered null selections for a single query risk set."""

    query_id: str
    global_episode_ids: tuple[str, ...]
    hierarchical_episode_ids: tuple[str, ...]


PriorityProvider = Callable[[SharedPriorityCandidate], bytes]


def _require_identifier(value: str, label: str) -> bytes:
    if type(value) is not str or not value:
        raise SharedPriorityError(f"{label} must be a non-empty str")
    encoded = value.encode("utf-8")
    if len(encoded) > 0xFFFF:
        raise SharedPriorityError(f"{label} UTF-8 encoding exceeds uint16 length")
    return encoded


def _lp16(value: bytes) -> bytes:
    """Length-prefix bytes with an unsigned big-endian uint16."""

    if len(value) > 0xFFFF:
        raise SharedPriorityError("encoded identifier exceeds uint16 length")
    return len(value).to_bytes(2, "big") + value


def _validate_draw(seed: int, replicate: int) -> None:
    if type(seed) is not int or not 0 <= seed <= _U64_MAX:
        raise SharedPriorityError("seed must be a uint64")
    if type(replicate) is not int or not 0 <= replicate <= _U32_MAX:
        raise SharedPriorityError("replicate must be a uint32")


def priority_preimage(*, domain: str, seed: int, replicate: int, identifier: str) -> bytes:
    """Return the exact domain-separated SHA-256 preimage.

    Encoding is ``lp16(utf8(domain)) || u64be(seed) || u32be(replicate) ||
    lp16(utf8(identifier))``.  It intentionally has no implicit text encoding,
    separator, host byte order, or random-generator state.
    """

    domain_bytes = _require_identifier(domain, "domain")
    identifier_bytes = _require_identifier(identifier, "identifier")
    _validate_draw(seed, replicate)
    return (
        _lp16(domain_bytes)
        + seed.to_bytes(8, "big", signed=False)
        + replicate.to_bytes(4, "big", signed=False)
        + _lp16(identifier_bytes)
    )


def priority_digest(*, domain: str, seed: int, replicate: int, identifier: str) -> bytes:
    """Return all 32 SHA-256 bytes for an identifier and null replicate."""

    return sha256(priority_preimage(
        domain=domain, seed=seed, replicate=replicate, identifier=identifier,
    )).digest()


def episode_priority(*, seed: int, replicate: int, episode_id: str) -> bytes:
    """The common ``H_episode`` used unchanged by both null families."""

    return priority_digest(
        domain=EPISODE_DOMAIN, seed=seed, replicate=replicate, identifier=episode_id,
    )


def symbol_priority(*, seed: int, replicate: int, symbol_id: str) -> bytes:
    """The separate ``H_symbol`` used only by the hierarchical null."""

    return priority_digest(
        domain=SYMBOL_DOMAIN, seed=seed, replicate=replicate, identifier=symbol_id,
    )


def intervals_overlap(left: SharedPriorityCandidate, right: SharedPriorityCandidate) -> bool:
    """Production's inclusive same-symbol interval-overlap predicate."""

    return (
        left.symbol_id == right.symbol_id
        and left.start_coordinate <= right.cutoff_coordinate
        and right.start_coordinate <= left.cutoff_coordinate
    )


def _validate_candidates(candidates: Sequence[SharedPriorityCandidate]) -> None:
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, SharedPriorityCandidate):
            raise SharedPriorityError("candidates must be SharedPriorityCandidate instances")
        _require_identifier(candidate.episode_id, "episode_id")
        _require_identifier(candidate.symbol_id, "symbol_id")
        if (type(candidate.start_coordinate) is not int
                or type(candidate.cutoff_coordinate) is not int):
            raise SharedPriorityError("candidate coordinates must be integers")
        if candidate.start_coordinate > candidate.cutoff_coordinate:
            raise SharedPriorityError("candidate start_coordinate cannot exceed cutoff_coordinate")
        if candidate.episode_id in seen:
            raise SharedPriorityError("duplicate episode_id in a risk set")
        seen.add(candidate.episode_id)


def _digest(value: bytes) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise SharedPriorityError("priority provider must return exactly 32 bytes")
    return value


def _global_key(candidate: SharedPriorityCandidate, priority_for: PriorityProvider) -> tuple[bytes, str]:
    return _digest(priority_for(candidate)), candidate.episode_id


def _hierarchical_key(
    candidate: SharedPriorityCandidate,
    episode_priority_for: PriorityProvider,
    symbol_priority_for: Callable[[str], bytes],
) -> tuple[bytes, str, bytes, str]:
    return (
        _digest(symbol_priority_for(candidate.symbol_id)), candidate.symbol_id,
        _digest(episode_priority_for(candidate)), candidate.episode_id,
    )


def exact_bucket_order(
    candidates: Iterable[SharedPriorityCandidate], *, key_for: Callable[[SharedPriorityCandidate], tuple],
) -> tuple[SharedPriorityCandidate, ...]:
    """Reference-sort a full priority key via 16-bit buckets without approximation.

    The first two digest bytes are a monotone prefix of every supported key.
    We only sort populated buckets, and inside each bucket sort the *complete*
    key.  Thus concatenating buckets 0..65535 is definitionally identical to
    sorting all keys together; no low-bit truncation enters the ordering.
    """

    buckets: dict[int, list[tuple[tuple, SharedPriorityCandidate]]] = defaultdict(list)
    for candidate in candidates:
        key = key_for(candidate)
        if not key or not isinstance(key[0], bytes) or len(key[0]) != 32:
            raise SharedPriorityError("priority key must begin with a 32-byte digest")
        bucket = int.from_bytes(key[0][:2], "big", signed=False)
        buckets[bucket].append((key, candidate))
    ordered: list[SharedPriorityCandidate] = []
    for bucket in sorted(buckets):
        ordered.extend(candidate for _, candidate in sorted(buckets[bucket], key=lambda item: item[0]))
    return tuple(ordered)


def greedy_accept(
    ordered: Iterable[SharedPriorityCandidate], *, config: SelectionConfig = SelectionConfig(),
) -> tuple[SharedPriorityCandidate, ...]:
    """Apply unchanged inclusive-overlap, cap-three, top-20 selection.

    A risk set that cannot produce exactly ``top_k`` valid entries is an error;
    returning a partial null neighbourhood would silently alter its statistic.
    """

    selected: list[SharedPriorityCandidate] = []
    counts: Counter[str] = Counter()
    for candidate in ordered:
        if counts[candidate.symbol_id] >= config.max_per_symbol:
            continue
        if any(intervals_overlap(candidate, prior) for prior in selected):
            continue
        selected.append(candidate)
        counts[candidate.symbol_id] += 1
        if len(selected) == config.top_k:
            return tuple(selected)
    raise SharedPriorityError("risk set cannot satisfy the diversity contract")


def global_order(
    candidates: Iterable[SharedPriorityCandidate], *, seed: int, replicate: int,
    priority_for: PriorityProvider | None = None,
) -> tuple[SharedPriorityCandidate, ...]:
    """Order one risk set by its common full ``H_episode`` priority."""

    _validate_draw(seed, replicate)
    materialized = tuple(candidates)
    _validate_candidates(materialized)
    if priority_for is None:
        priority_for = lambda candidate: episode_priority(
            seed=seed, replicate=replicate, episode_id=candidate.episode_id,
        )
    return exact_bucket_order(materialized, key_for=lambda candidate: _global_key(candidate, priority_for))


def hierarchical_order(
    candidates: Iterable[SharedPriorityCandidate], *, seed: int, replicate: int,
    episode_priority_for: PriorityProvider | None = None,
    symbol_priority_for: Callable[[str], bytes] | None = None,
) -> tuple[SharedPriorityCandidate, ...]:
    """Order by ``H_symbol`` then common ``H_episode``, both with ID ties."""

    _validate_draw(seed, replicate)
    materialized = tuple(candidates)
    _validate_candidates(materialized)
    if episode_priority_for is None:
        episode_priority_for = lambda candidate: episode_priority(
            seed=seed, replicate=replicate, episode_id=candidate.episode_id,
        )
    if symbol_priority_for is None:
        symbol_priority_for = lambda symbol_id: symbol_priority(
            seed=seed, replicate=replicate, symbol_id=symbol_id,
        )
    return exact_bucket_order(
        materialized,
        key_for=lambda candidate: _hierarchical_key(
            candidate, episode_priority_for, symbol_priority_for,
        ),
    )


def select_global(
    candidates: Iterable[SharedPriorityCandidate], *, seed: int, replicate: int,
    config: SelectionConfig = SelectionConfig(), priority_for: PriorityProvider | None = None,
) -> tuple[SharedPriorityCandidate, ...]:
    return greedy_accept(
        global_order(candidates, seed=seed, replicate=replicate, priority_for=priority_for),
        config=config,
    )


def select_hierarchical(
    candidates: Iterable[SharedPriorityCandidate], *, seed: int, replicate: int,
    config: SelectionConfig = SelectionConfig(),
    episode_priority_for: PriorityProvider | None = None,
    symbol_priority_for: Callable[[str], bytes] | None = None,
) -> tuple[SharedPriorityCandidate, ...]:
    return greedy_accept(
        hierarchical_order(
            candidates, seed=seed, replicate=replicate,
            episode_priority_for=episode_priority_for,
            symbol_priority_for=symbol_priority_for,
        ),
        config=config,
    )


def evaluate_query_selections(
    query_risk_sets: Mapping[str, Sequence[SharedPriorityCandidate]], *, seed: int,
    replicate: int, config: SelectionConfig = SelectionConfig(), workers: int = 1,
) -> tuple[QuerySelection, ...]:
    """Synthetic/reference evaluator with deterministic parallel semantics.

    This validates shared episode identity across risk sets.  It deliberately
    does not implement the production reuse optimization and must never be used
    for the full 3,270-query experiment.
    """

    if type(workers) is not int or workers < 1:
        raise SharedPriorityError("workers must be a positive integer")
    query_items = tuple(
        (query_id, tuple(candidates))
        for query_id, candidates in sorted(query_risk_sets.items())
    )
    if len({query_id for query_id, _ in query_items}) != len(query_items):
        raise SharedPriorityError("duplicate query_id")
    episode_identity: dict[str, tuple[str, int, int]] = {}
    for query_id, candidates in query_items:
        _require_identifier(query_id, "query_id")
        _validate_candidates(candidates)
        for candidate in candidates:
            identity = (
                candidate.symbol_id,
                candidate.start_coordinate,
                candidate.cutoff_coordinate,
            )
            prior = episode_identity.setdefault(candidate.episode_id, identity)
            if prior != identity:
                raise SharedPriorityError(
                    "shared episode_id has conflicting symbol or interval coordinates"
                )

    def one(item: tuple[str, Sequence[SharedPriorityCandidate]]) -> QuerySelection:
        query_id, candidates = item
        return QuerySelection(
            query_id=query_id,
            global_episode_ids=tuple(row.episode_id for row in select_global(
                candidates, seed=seed, replicate=replicate, config=config,
            )),
            hierarchical_episode_ids=tuple(row.episode_id for row in select_hierarchical(
                candidates, seed=seed, replicate=replicate, config=config,
            )),
        )

    if workers == 1:
        return tuple(one(item) for item in query_items)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        completed = tuple(executor.map(one, query_items))
    return tuple(sorted(completed, key=lambda row: row.query_id))


def selection_semantic_digest(selections: Sequence[QuerySelection]) -> str:
    """Canonical content digest for serial/parallel identity assertions."""

    seen: set[str] = set()
    rows = []
    for row in selections:
        if not isinstance(row, QuerySelection):
            raise SharedPriorityError("selections must contain QuerySelection instances")
        _require_identifier(row.query_id, "query_id")
        if row.query_id in seen:
            raise SharedPriorityError("duplicate query_id in selections")
        seen.add(row.query_id)
        if not all(type(identifier) is str and identifier for identifier in (
            *row.global_episode_ids, *row.hierarchical_episode_ids,
        )):
            raise SharedPriorityError("selected episode IDs must be non-empty str")
        rows.append({
            "query_id": row.query_id,
            "global_episode_ids": list(row.global_episode_ids),
            "hierarchical_episode_ids": list(row.hierarchical_episode_ids),
        })
    rows.sort(key=lambda row: row["query_id"])
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256(encoded).hexdigest()


__all__ = [
    "EPISODE_DOMAIN", "SYMBOL_DOMAIN", "QuerySelection", "SelectionConfig",
    "SharedPriorityCandidate", "SharedPriorityError", "episode_priority",
    "evaluate_query_selections", "exact_bucket_order", "global_order", "greedy_accept",
    "hierarchical_order", "intervals_overlap", "priority_digest", "priority_preimage",
    "select_global", "select_hierarchical", "selection_semantic_digest", "symbol_priority",
]
