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
    if values.ndim != 1 or len(values) < 64 or not np.isfinite(values[-64:]).all() \
            or np.any(values[-64:] <= 0):
        return np.full(3, np.nan, dtype=np.float64)
    log_returns = np.diff(np.log(values[-21:]))
    output = np.asarray([
        values[-1] / values[-21] - 1.0,
        values[-1] / values[-64] - 1.0,
        np.std(log_returns, ddof=1),
    ], dtype=np.float64)
    return output if np.isfinite(output).all() else np.full(3, np.nan)


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
    best: dict[int, tuple[bytes, bytes]] = {}
    for position in np.flatnonzero(mask):
        raw = bytes(ids[position])
        symbol_id = int(symbol_values[position])
        key = _digest(b"wf03-random-episode-v1", query_id, raw)
        current = best.get(symbol_id)
        if current is None or (key, raw) < current:
            best[symbol_id] = (key, raw)
    ordered_symbols = sorted(best, key=lambda value: (
        _digest(b"wf03-random-symbol-v1", query_id, symbols[value].encode()),
        symbols[value],
    ))
    return tuple(BaselineNeighbor(
        best[symbol_id][1].hex(), symbols[symbol_id], None,
        best[symbol_id][0].hex(),
    ) for symbol_id in ordered_symbols[:top_k])


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
