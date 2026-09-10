"""Deterministic continuous-rank nearest controls for post-signal studies."""
from __future__ import annotations

from hashlib import sha256
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata


class RankMatchingError(ValueError): pass


def percentile_rank_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 4 or not len(matrix) or not np.isfinite(matrix).all():
        raise RankMatchingError("rank matching requires a finite n-by-four matrix")
    return np.column_stack([rankdata(matrix[:, column], method="average") / len(matrix) for column in range(4)])


def _digest(contract_digest: str, signal_id: str, symbol: str) -> str:
    return sha256(f"{contract_digest}|{signal_id}|{symbol}".encode("utf-8")).hexdigest()


def _exact_order(
    names: np.ndarray, candidate_indices: np.ndarray, candidate_points: np.ndarray,
    event_point: np.ndarray, contract_digest: str, signal_id: str, controls: int,
) -> tuple[np.ndarray, np.ndarray]:
    squared = np.sum((candidate_points - event_point) ** 2, axis=1)
    ordered = sorted(range(len(candidate_indices)), key=lambda index: (
        float(squared[index]), _digest(contract_digest, signal_id, str(names[candidate_indices[index]])),
        str(names[candidate_indices[index]]),
    ))[:controls]
    return candidate_indices[ordered].astype(np.int64), squared[ordered]


def select_rank_nearest_batch(
    *, symbols: Sequence[str], percentile_ranks: np.ndarray, candidate_indices: Sequence[int],
    event_indices: Sequence[int], contract_digest: str, signal_ids: Sequence[str], controls: int = 5,
    coordinate_weights: Sequence[float] = (1., 1., 1., 1.),
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return exact deterministic top-k controls for each event using a batched spatial frontier."""
    names = np.asarray(symbols, dtype=object); ranks = np.asarray(percentile_ranks, dtype=np.float64)
    candidates = np.asarray(candidate_indices, dtype=np.int64); events = np.asarray(event_indices, dtype=np.int64)
    weights = np.asarray(coordinate_weights, dtype=np.float64)
    if ranks.shape != (len(names), 4) or len(events) != len(signal_ids) or controls < 1:
        raise RankMatchingError("rank matching request differs")
    if weights.shape != (4,) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise RankMatchingError("coordinate weights must be four finite positive values")
    if len(np.unique(candidates)) != len(candidates) or len(np.unique(events)) != len(events):
        raise RankMatchingError("rank matching indices must be unique")
    if not len(events): return []
    if not len(candidates):
        empty = np.empty(0, dtype=np.int64); distances = np.empty(0, dtype=np.float64)
        return [(empty.copy(), distances.copy()) for _ in events]
    if (candidates < 0).any() or (candidates >= len(names)).any() or (events < 0).any() or (events >= len(names)).any():
        raise RankMatchingError("rank matching index is outside the cross-section")
    scale = np.sqrt(weights); candidate_points = ranks[candidates] * scale; query_points = ranks[events] * scale
    frontier = min(len(candidates), max(controls, 16)); tree = cKDTree(candidate_points)
    distances, local = tree.query(query_points, k=frontier)
    if frontier == 1:
        distances = np.asarray(distances)[:, None]; local = np.asarray(local)[:, None]
    results = []
    for row, (event_index, signal_id) in enumerate(zip(events, signal_ids)):
        local_frontier = np.atleast_1d(local[row]).astype(np.int64)
        if len(candidates) > frontier:
            kth = float(np.atleast_1d(distances[row])[min(controls, frontier) - 1])
            last = float(np.atleast_1d(distances[row])[-1])
            if math_isclose(last, kth):
                radius = kth * (1. + 1e-12) + 1e-15
                local_frontier = np.asarray(tree.query_ball_point(query_points[row], radius), dtype=np.int64)
        chosen_candidates = candidates[local_frontier]
        chosen_points = candidate_points[local_frontier]
        selected, squared = _exact_order(
            names, chosen_candidates, chosen_points, query_points[row], contract_digest, str(signal_id), controls,
        )
        if str(names[event_index]) in set(str(names[index]) for index in selected):
            raise RankMatchingError("event symbol entered its control set")
        results.append((selected, squared))
    return results


def math_isclose(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-14 * max(1., abs(left), abs(right))


def exhaustive_rank_nearest(
    *, symbols: Sequence[str], percentile_ranks: np.ndarray, candidate_indices: Sequence[int],
    event_index: int, contract_digest: str, signal_id: str, controls: int = 5,
    coordinate_weights: Sequence[float] = (1., 1., 1., 1.),
) -> tuple[np.ndarray, np.ndarray]:
    names = np.asarray(symbols, dtype=object); ranks = np.asarray(percentile_ranks, dtype=float)
    candidates = np.asarray(candidate_indices, dtype=np.int64); weights = np.asarray(coordinate_weights, dtype=float)
    if weights.shape != (4,) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise RankMatchingError("coordinate weights must be four finite positive values")
    scale = np.sqrt(weights)
    return _exact_order(names, candidates, ranks[candidates] * scale, ranks[int(event_index)] * scale,
                        contract_digest, signal_id, controls)
