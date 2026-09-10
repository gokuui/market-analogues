"""Outcome-blind neighbourhood adequacy primitives.

The functions here deliberately know nothing about prices, returns, or outcomes.  They
operate on causal candidate identities and interval geometry only.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class NullCandidate:
    """The minimum metadata needed by the production diversity selector."""

    symbol_id: int
    ordinal: int
    start_ns: int
    cutoff_ns: int


def intervals_overlap(a: NullCandidate, b: NullCandidate) -> bool:
    """Match the production selector's inclusive interval-overlap rule."""

    return (
        a.symbol_id == b.symbol_id
        and a.start_ns <= b.cutoff_ns
        and b.start_ns <= a.cutoff_ns
    )


def greedy_accept(
    ordered: Iterable[NullCandidate], *, top_k: int = 20, max_per_symbol: int = 3,
) -> tuple[NullCandidate, ...]:
    """Apply the exact cap/overlap acceptance rule to an already ranked stream."""

    selected: list[NullCandidate] = []
    counts: Counter[int] = Counter()
    for item in ordered:
        if counts[item.symbol_id] >= max_per_symbol:
            continue
        if any(intervals_overlap(item, prior) for prior in selected):
            continue
        selected.append(item)
        counts[item.symbol_id] += 1
        if len(selected) == top_k:
            break
    return tuple(selected)


def random_priority_selection(
    *,
    rng: np.random.Generator,
    eligible_count: int,
    candidate_at: Callable[[int], NullCandidate],
    top_k: int = 20,
    max_per_symbol: int = 3,
    maximum_draws: int | None = None,
) -> tuple[NullCandidate, ...]:
    """Sample the greedy result under iid continuous random candidate distances.

    Repeated uniform draws with duplicate rejection expose a uniformly random
    permutation of the eligible candidates.  Passing that stream through
    :func:`greedy_accept` is therefore exactly the production selector under the
    stated null (ties have probability zero).
    """

    if eligible_count < top_k or top_k < 1 or max_per_symbol < 1:
        raise ValueError("invalid null-selection dimensions")
    if maximum_draws is not None and maximum_draws < top_k:
        raise ValueError("maximum draws cannot be smaller than top-k")
    drawn: set[int] = set()
    selected: list[NullCandidate] = []
    per_symbol: Counter[int] = Counter()
    draws = 0
    while len(selected) < top_k:
        if (maximum_draws is not None and draws >= maximum_draws) \
                or len(drawn) == eligible_count:
            raise ValueError("risk set cannot satisfy the diversity contract")
        index = int(rng.integers(0, eligible_count))
        draws += 1
        if index in drawn:
            continue
        drawn.add(index)
        item = candidate_at(index)
        if per_symbol[item.symbol_id] >= max_per_symbol:
            continue
        if any(intervals_overlap(item, prior) for prior in selected):
            continue
        selected.append(item)
        per_symbol[item.symbol_id] += 1
    return tuple(selected)


def top_share(counts: Sequence[int], fraction: float) -> float:
    if not counts or not 0 < fraction <= 1:
        return 0.0
    ordered = sorted((int(value) for value in counts if value > 0), reverse=True)
    if not ordered:
        return 0.0
    take = max(1, int(np.ceil(len(ordered) * fraction)))
    return float(sum(ordered[:take]) / sum(ordered))


def hhi(counts: Sequence[int]) -> float:
    total = float(sum(counts))
    return float(sum((value / total) ** 2 for value in counts)) if total else 0.0


def gini_with_zeros(counts: Sequence[int], population: int) -> float:
    """Gini over a known population without materialising its zero counts."""

    positive = np.sort(np.asarray([value for value in counts if value > 0], dtype=float))
    if population < len(positive) or population < 1 or not len(positive):
        return 0.0
    full_indices = np.arange(population - len(positive) + 1, population + 1, dtype=float)
    return float((2.0 * np.dot(full_indices, positive) / (population * positive.sum()))
                 - (population + 1.0) / population)


def concentration_metrics(
    episode_counts: Sequence[int], symbol_counts: Sequence[int], *, episode_population: int,
) -> dict[str, float | int]:
    """Global concentration statistics used identically for real and null links."""

    episode = [int(value) for value in episode_counts if value > 0]
    symbol = [int(value) for value in symbol_counts if value > 0]
    return {
        "episode_unique": len(episode),
        "episode_max": max(episode, default=0),
        "episode_top_1_percent_share": top_share(episode, 0.01),
        "episode_hhi": hhi(episode),
        "episode_effective_number": 1.0 / hhi(episode) if hhi(episode) else 0.0,
        "episode_gini": gini_with_zeros(episode, episode_population),
        "symbol_unique": len(symbol),
        "symbol_max": max(symbol, default=0),
        "symbol_top_1_percent_share": top_share(symbol, 0.01),
        "symbol_hhi": hhi(symbol),
        "symbol_effective_number": 1.0 / hhi(symbol) if hhi(symbol) else 0.0,
    }


def nearest_rank(values: Sequence[float], probability: float) -> float:
    if not values or not 0 < probability <= 1:
        raise ValueError("nearest-rank percentile requires data and 0 < p <= 1")
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, int(np.ceil(probability * len(ordered))) - 1)]
