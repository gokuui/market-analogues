from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np

from .types import stable_hash


class BaselineNeighborError(ValueError):
    pass


@dataclass(frozen=True)
class BaselineNeighbor:
    episode_id: str
    symbol: str
    distance: float | None
    order_key: str


@dataclass(frozen=True)
class BaselineRankIndex:
    orders: tuple[np.ndarray, np.ndarray, np.ndarray]
    finite: np.ndarray


def baseline_neighbor_contract() -> dict[str, Any]:
    state = {
        "schema_version": "wf03-baseline-neighbors-v1",
        "random": {
            "symbol_order": "sha256(wf03-random-symbol-v1, query ID, symbol)",
            "episode_order": "sha256(wf03-random-episode-v1, query ID, episode ID)",
            "selection": "first eligible episode within each of first 20 eligible symbols",
        },
        "recent_return_volatility": {
            "features": [
                "close[-1]/close[-21]-1",
                "close[-1]/close[-64]-1",
                "sample stddev of final 20 close log returns",
            ],
            "ranks": (
                "normalized ordinal ranks among eligible episodes plus query; "
                "ascending (finite value, episode ID) with query ID in the same tie rule"
            ),
            "distance": "unweighted L1 across the three normalized ranks",
            "selection": "ascending (distance, episode ID), first episode per symbol",
        },
        "eligibility": "supplied causal packed-store mask; all feature rows finite",
        "outcomes_or_labels_used": False,
    }
    return {**state, "digest": stable_hash(state)}


def recent_return_volatility(close: np.ndarray) -> np.ndarray:
    values = np.asarray(close, dtype=np.float64)
    if values.ndim != 1 or len(values) < 64:
        return np.full(3, np.nan, dtype=np.float64)
    return recent_return_volatility_at_positions(
        values, np.asarray([len(values) - 1], dtype=np.int64),
    )[0]


def recent_return_volatility_at_positions(
    close: np.ndarray, positions: np.ndarray,
) -> np.ndarray:
    values = np.asarray(close, dtype=np.float64)
    requested = np.asarray(positions)
    if values.ndim != 1 or requested.ndim != 1 \
            or requested.dtype.kind not in "iu" \
            or len(requested) and (int(np.min(requested)) < 63
                                   or int(np.max(requested)) >= len(values)):
        raise BaselineNeighborError("return/volatility feature positions differ")
    if not len(requested):
        return np.empty((0, 3), dtype=np.float64)
    windows = np.lib.stride_tricks.sliding_window_view(values, 64)[requested - 63]
    valid = np.isfinite(windows).all(axis=1) & (windows > 0).all(axis=1)
    output = np.full((len(requested), 3), np.nan, dtype=np.float64)
    if np.any(valid):
        selected = windows[valid]
        output[valid, 0] = selected[:, -1] / selected[:, -21] - 1.0
        output[valid, 1] = selected[:, -1] / selected[:, 0] - 1.0
        output[valid, 2] = np.std(
            np.diff(np.log(selected[:, -21:]), axis=1), axis=1, ddof=1,
        )
    return output


def _digest(domain: bytes, query_id: str, value: bytes) -> bytes:
    digest = sha256()
    digest.update(domain); digest.update(b"\0")
    digest.update(bytes.fromhex(query_id)); digest.update(b"\0"); digest.update(value)
    return digest.digest()


def _validate_arrays(
    episode_ids: np.ndarray, symbol_ids: np.ndarray, eligible: np.ndarray,
    symbols: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray(episode_ids)
    symbol_values = np.asarray(symbol_ids)
    mask = np.asarray(eligible)
    if ids.ndim != 1 or ids.dtype.itemsize != 12 \
            or symbol_values.shape != ids.shape or mask.shape != ids.shape \
            or symbol_values.dtype.kind not in "ui" or mask.dtype != np.bool_ \
            or len(symbols) == 0 \
            or len(symbol_values) and int(np.max(symbol_values)) >= len(symbols):
        raise BaselineNeighborError("baseline neighbor arrays differ")
    return ids, symbol_values, mask


def deterministic_random_neighbors(
    episode_ids: np.ndarray,
    symbol_ids: np.ndarray,
    eligible: np.ndarray,
    symbols: tuple[str, ...],
    query_id: str,
    *,
    top_k: int = 20,
) -> tuple[BaselineNeighbor, ...]:
    ids, symbol_values, mask = _validate_arrays(
        episode_ids, symbol_ids, eligible, symbols,
    )
    if type(top_k) is not int or isinstance(top_k, bool) or top_k < 1:
        raise BaselineNeighborError("random baseline top-k differs")
    positions = np.flatnonzero(mask)
    eligible_symbols = np.unique(symbol_values[positions])
    ordered_symbols = sorted((int(value) for value in eligible_symbols), key=lambda value: (
        _digest(b"wf03-random-symbol-v1", query_id, symbols[value].encode()),
        symbols[value],
    ))[:top_k]
    selected_symbols = np.zeros(len(symbols), dtype=np.bool_)
    selected_symbols[ordered_symbols] = True
    best: dict[int, tuple[bytes, bytes]] = {}
    for position in positions[selected_symbols[symbol_values[positions]]]:
        raw = bytes(ids[position])
        symbol_id = int(symbol_values[position])
        key = _digest(b"wf03-random-episode-v1", query_id, raw)
        current = best.get(symbol_id)
        if current is None or (key, raw) < current:
            best[symbol_id] = (key, raw)
    return tuple(BaselineNeighbor(
        best[symbol_id][1].hex(), symbols[symbol_id], None,
        best[symbol_id][0].hex(),
    ) for symbol_id in ordered_symbols)


def _ordinal_ranks_with_query(
    values: np.ndarray, episode_ids: np.ndarray, query_value: float, query_id: str,
) -> tuple[np.ndarray, float]:
    n = len(values)
    combined_values = np.concatenate((values, np.asarray([query_value])))
    combined_ids = np.concatenate((
        np.asarray(episode_ids, dtype="V12"),
        np.asarray([np.void(bytes.fromhex(query_id))], dtype="V12"),
    ))
    raw = np.frombuffer(np.ascontiguousarray(combined_ids).tobytes(), dtype=np.dtype([
        ("high", ">u8"), ("low", ">u4"),
    ]))
    order = np.lexsort((raw["low"], raw["high"], combined_values))
    ranks = np.empty(n + 1, dtype=np.float64)
    ranks[order] = np.arange(n + 1, dtype=np.float64) / max(n, 1)
    return ranks[:n], float(ranks[n])


def _identifier_fields(episode_ids: np.ndarray) -> np.ndarray:
    return np.frombuffer(
        np.ascontiguousarray(episode_ids, dtype="V12").tobytes(),
        dtype=np.dtype([("high", ">u8"), ("low", ">u4")]),
    )


def build_baseline_rank_index(
    features: np.ndarray, episode_ids: np.ndarray,
) -> BaselineRankIndex:
    matrix = np.asarray(features, dtype=np.float64)
    ids = np.asarray(episode_ids)
    if matrix.ndim != 2 or matrix.shape[1:] != (3,) \
            or ids.ndim != 1 or len(ids) != len(matrix) or ids.dtype.itemsize != 12:
        raise BaselineNeighborError("return/volatility rank-index arrays differ")
    finite = np.isfinite(matrix).all(axis=1)
    if np.any(np.isfinite(matrix).any(axis=1) != finite):
        raise BaselineNeighborError("return/volatility rank-index finiteness differs")
    positions = np.flatnonzero(finite)
    identifiers = _identifier_fields(ids[positions])
    orders = tuple(
        positions[np.lexsort((
            identifiers["low"], identifiers["high"], matrix[positions, column],
        ))].astype(np.int64, copy=False)
        for column in range(3)
    )
    return BaselineRankIndex(orders, finite)


def indexed_recent_return_volatility_neighbors(
    index: BaselineRankIndex,
    features: np.ndarray,
    episode_ids: np.ndarray,
    symbol_ids: np.ndarray,
    eligible: np.ndarray,
    symbols: tuple[str, ...],
    query_features: np.ndarray,
    query_id: str,
    *,
    top_k: int = 20,
) -> tuple[BaselineNeighbor, ...]:
    ids, symbol_values, mask = _validate_arrays(
        episode_ids, symbol_ids, eligible, symbols,
    )
    matrix = np.asarray(features, dtype=np.float64)
    query = np.asarray(query_features, dtype=np.float64)
    if matrix.shape != (len(ids), 3) or query.shape != (3,) \
            or index.finite.shape != (len(ids),) or index.finite.dtype != np.bool_ \
            or len(index.orders) != 3 \
            or type(top_k) is not int or isinstance(top_k, bool) or top_k < 1 \
            or not np.isfinite(query).all():
        raise BaselineNeighborError("indexed return/volatility arrays differ")
    mask = mask & index.finite
    positions = np.flatnonzero(mask)
    count = len(positions)
    if not count:
        return ()
    distance = np.zeros(len(ids), dtype=np.float64)
    query_raw = np.void(bytes.fromhex(query_id))
    for column, order in enumerate(index.orders):
        if order.ndim != 1 or order.dtype.kind not in "iu" \
                or len(order) != int(index.finite.sum()):
            raise BaselineNeighborError("return/volatility rank-index order differs")
        sorted_eligible = mask[order]
        cumulative = np.cumsum(sorted_eligible, dtype=np.int64)
        sorted_values = matrix[order, column]
        left = int(np.searchsorted(sorted_values, query[column], side="left"))
        right = int(np.searchsorted(sorted_values, query[column], side="right"))
        equal_ids = np.ascontiguousarray(ids[order[left:right]], dtype="V12")
        insertion = left + int(np.searchsorted(equal_ids, query_raw, side="left"))
        query_less = int(cumulative[insertion - 1]) if insertion else 0
        query_rank = query_less / max(count, 1)
        candidate_sorted_positions = np.flatnonzero(sorted_eligible)
        candidate_order = order[candidate_sorted_positions]
        candidate_ranks = (
            np.arange(count, dtype=np.float64)
            + (candidate_sorted_positions >= insertion)
        ) / max(count, 1)
        distance[candidate_order] += np.abs(candidate_ranks - query_rank)
    selected_symbols = symbol_values[positions].astype(np.int64, copy=False)
    minima = np.full(len(symbols), np.inf, dtype=np.float64)
    np.minimum.at(minima, selected_symbols, distance[positions])
    tied = positions[distance[positions] == minima[selected_symbols]]
    tied_identifiers = _identifier_fields(ids[tied])
    tied_order = np.lexsort((
        tied_identifiers["low"], tied_identifiers["high"], symbol_values[tied],
    ))
    ordered_tied = tied[tied_order]
    _unique_symbols, first = np.unique(
        symbol_values[ordered_tied], return_index=True,
    )
    best = ordered_tied[first]
    best_identifiers = _identifier_fields(ids[best])
    ranking = np.lexsort((
        best_identifiers["low"], best_identifiers["high"], distance[best],
    ))
    return tuple(BaselineNeighbor(
        bytes(ids[position]).hex(), symbols[int(symbol_values[position])],
        float(distance[position]), "",
    ) for position in best[ranking[:top_k]])


def recent_return_volatility_neighbors(
    features: np.ndarray,
    episode_ids: np.ndarray,
    symbol_ids: np.ndarray,
    eligible: np.ndarray,
    symbols: tuple[str, ...],
    query_features: np.ndarray,
    query_id: str,
    *,
    top_k: int = 20,
) -> tuple[BaselineNeighbor, ...]:
    ids, symbol_values, mask = _validate_arrays(
        episode_ids, symbol_ids, eligible, symbols,
    )
    matrix = np.asarray(features, dtype=np.float64)
    query = np.asarray(query_features, dtype=np.float64)
    if matrix.shape != (len(ids), 3) or query.shape != (3,) \
            or type(top_k) is not int or isinstance(top_k, bool) or top_k < 1:
        raise BaselineNeighborError("return/volatility baseline arrays differ")
    mask = mask & np.isfinite(matrix).all(axis=1)
    if not np.isfinite(query).all():
        raise BaselineNeighborError("query return/volatility features differ")
    positions = np.flatnonzero(mask)
    selected_features = matrix[positions]
    selected_ids = ids[positions]
    distances = np.zeros(len(positions), dtype=np.float64)
    for column in range(3):
        ranks, query_rank = _ordinal_ranks_with_query(
            selected_features[:, column], selected_ids, query[column], query_id,
        )
        distances += np.abs(ranks - query_rank)
    raw = np.frombuffer(np.ascontiguousarray(selected_ids).tobytes(), dtype=np.dtype([
        ("high", ">u8"), ("low", ">u4"),
    ]))
    order = np.lexsort((raw["low"], raw["high"], distances))
    output = []
    seen = set()
    for local in order:
        position = positions[int(local)]
        symbol_id = int(symbol_values[position])
        if symbol_id in seen:
            continue
        seen.add(symbol_id)
        output.append(BaselineNeighbor(
            bytes(ids[position]).hex(), symbols[symbol_id],
            float(distances[local]), "",
        ))
        if len(output) == top_k:
            break
    return tuple(output)
