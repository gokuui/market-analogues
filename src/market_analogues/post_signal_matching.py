"""Outcome-blind matching kernels for prospective post-signal studies."""
from __future__ import annotations

from hashlib import sha256
from itertools import product
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import rankdata


class PostSignalMatchingError(ValueError):
    pass


def cross_sectional_deciles(values: Sequence[float]) -> np.ndarray:
    """Return deterministic average-rank deciles for one date/population cross-section."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise PostSignalMatchingError("matching values must be a nonempty finite vector")
    ranks = rankdata(array, method="average")
    return np.clip(np.ceil(10.0 * ranks / len(array)), 1, 10).astype(np.uint8)


def event_id(contract_digest: str, signal_name: str, symbol: str, signal_date: str) -> str:
    payload = f"{contract_digest}|{signal_name}|{symbol}|{signal_date}".encode("utf-8")
    return sha256(payload).hexdigest()


def control_order_digest(contract_digest: str, signal_id: str, candidate_symbol: str) -> str:
    payload = f"{contract_digest}|{signal_id}|{candidate_symbol}".encode("utf-8")
    return sha256(payload).hexdigest()


def bucket_members(deciles: np.ndarray, eligible: Sequence[bool]) -> dict[tuple[int, ...], np.ndarray]:
    matrix = np.asarray(deciles)
    mask = np.asarray(eligible, dtype=bool)
    if matrix.ndim != 2 or matrix.shape[1] != 4 or len(matrix) != len(mask):
        raise PostSignalMatchingError("four-decile matrix and eligibility mask differ")
    members: dict[tuple[int, ...], list[int]] = {}
    for index in np.flatnonzero(mask):
        members.setdefault(tuple(int(value) for value in matrix[index]), []).append(int(index))
    return {key: np.asarray(value, dtype=np.int64) for key, value in members.items()}


def select_control_indices(
    *, symbols: Sequence[str], members: Mapping[tuple[int, ...], np.ndarray],
    all_eligible_indices: Sequence[int], event_symbol: str, event_deciles: Sequence[int],
    contract_digest: str, signal_id: str, controls: int = 5,
) -> tuple[np.ndarray, str]:
    """Select unique controls using the frozen exact/nearby/same-date ladder."""
    names = np.asarray(symbols, dtype=object)
    eligible = np.asarray(all_eligible_indices, dtype=np.int64)
    key = tuple(int(value) for value in event_deciles)
    if len(key) != 4 or any(value < 1 or value > 10 for value in key) or controls < 1:
        raise PostSignalMatchingError("matching request is invalid")
    exact = members.get(key, np.empty(0, dtype=np.int64))
    if len(exact) >= controls:
        pool, tier = exact, "exact_all_four_deciles"
    else:
        nearby = []
        axes = [range(max(1, value - 1), min(10, value + 1) + 1) for value in key]
        for neighbor in product(*axes):
            found = members.get(tuple(neighbor))
            if found is not None:
                nearby.append(found)
        within_one = np.concatenate(nearby) if nearby else np.empty(0, dtype=np.int64)
        if len(within_one) >= controls:
            pool, tier = within_one, "within_one_bucket_all_four"
        else:
            pool, tier = eligible, "same_date_unmatched"
    if len(pool):
        pool = pool[names[pool] != event_symbol]
    ordered = sorted(
        (int(index) for index in pool),
        key=lambda index: (control_order_digest(contract_digest, signal_id, str(names[index])), str(names[index])),
    )
    return np.asarray(ordered[:controls], dtype=np.int64), tier
