from __future__ import annotations

from collections import Counter
from math import comb
from typing import Hashable, Iterable, Mapping, Sequence

import numpy as np


class AdequacySupportError(ValueError):
    pass


def causally_eligible(
    *, candidate_symbol: str, candidate_cutoff_ns: int,
    query_symbol: str, query_start_ns: int, latest_eligible_ns: int,
) -> bool:
    return (
        candidate_cutoff_ns <= latest_eligible_ns
        and (candidate_symbol != query_symbol or candidate_cutoff_ns < query_start_ns)
    )


def aligned_value(main: np.ndarray, overflow: np.ndarray, index: int) -> np.ndarray:
    if index < 0 or index >= len(main) + len(overflow):
        raise AdequacySupportError("aligned feature index is outside the store")
    return main[index] if index < len(main) else overflow[index - len(main)]


def capped_matched_set_count(
    eligible_cells: Mapping[Hashable, int],
    observed_cells: Mapping[Hashable, int],
    *,
    cap: int,
) -> int:
    """Return the exact product of combinations, saturated at ``cap``."""
    if cap < 1:
        raise AdequacySupportError("support cap must be positive")
    support = 1
    for cell, selected in observed_cells.items():
        available = int(eligible_cells.get(cell, 0))
        selected = int(selected)
        if selected < 0 or available < selected:
            return 0
        support *= comb(available, selected)
        if support >= cap:
            return cap
    return support


def matched_support(
    eligible_indices: Iterable[int],
    observed_indices: Iterable[int],
    cells: Sequence[Hashable],
    *,
    cap: int,
) -> int:
    eligible = tuple(int(value) for value in eligible_indices)
    observed = tuple(int(value) for value in observed_indices)
    if len(set(eligible)) != len(eligible) or len(set(observed)) != len(observed):
        raise AdequacySupportError("query membership contains duplicates")
    eligible_set = set(eligible)
    if not set(observed).issubset(eligible_set):
        raise AdequacySupportError("observed queries are not a subset of causal eligibility")
    if any(value < 0 or value >= len(cells) for value in (*eligible, *observed)):
        raise AdequacySupportError("query membership index is outside the cell vector")
    return capped_matched_set_count(
        Counter(cells[index] for index in eligible),
        Counter(cells[index] for index in observed),
        cap=cap,
    )


def deterministic_terciles(values: Sequence[float], ids: Sequence[str]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) != len(ids) or len(set(ids)) != len(ids):
        raise AdequacySupportError("tercile inputs differ")
    output = np.full(len(array), -1, dtype=np.int8)
    valid = [index for index, value in enumerate(array) if np.isfinite(value)]
    for rank, index in enumerate(sorted(valid, key=lambda item: (array[item], ids[item]))):
        output[index] = min(2, rank * 3 // max(len(valid), 1))
    return output


def farthest_first_partition(
    vectors: np.ndarray,
    ids: Sequence[str],
    *,
    clusters: int,
    minimum_size: int,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Deterministic farthest-first cells with small cells merged by medoid distance."""
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or len(values) != len(ids) or len(set(ids)) != len(ids):
        raise AdequacySupportError("partition inputs differ")
    if not 1 <= clusters <= len(values) or minimum_size < 1:
        raise AdequacySupportError("partition limits differ")
    if not np.isfinite(values).all():
        raise AdequacySupportError("partition vectors must be finite")
    first = min(range(len(ids)), key=lambda index: ids[index])
    medoids = [first]
    nearest = np.sum((values - values[first]) ** 2, axis=1)
    for _ in range(1, clusters):
        available = (index for index in range(len(values)) if index not in medoids)
        best = max(available, key=lambda index: (nearest[index], ids[index]))
        medoids.append(best)
        nearest = np.minimum(nearest, np.sum((values - values[best]) ** 2, axis=1))
    distances = np.column_stack([
        np.sum((values - values[index]) ** 2, axis=1) for index in medoids
    ])
    labels = np.argmin(distances, axis=1).astype(np.int32)
    active = set(range(len(medoids)))
    while len(active) > 1:
        counts = Counter(int(value) for value in labels)
        small = [cell for cell in sorted(active) if counts.get(cell, 0) < minimum_size]
        if not small:
            break
        cell = min(small, key=lambda item: (counts.get(item, 0), ids[medoids[item]]))
        targets = sorted(active - {cell})
        target = min(
            targets,
            key=lambda item: (
                float(np.sum((values[medoids[cell]] - values[medoids[item]]) ** 2)),
                ids[medoids[item]],
            ),
        )
        labels[labels == cell] = target
        active.remove(cell)
    ordered = sorted(active, key=lambda item: ids[medoids[item]])
    remap = {old: new for new, old in enumerate(ordered)}
    return np.asarray([remap[int(value)] for value in labels], dtype=np.int32), tuple(
        medoids[item] for item in ordered
    )
