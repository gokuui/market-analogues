"""Deterministic matching and dependence-aware inference for the Stockbee study."""
from __future__ import annotations

from hashlib import sha256
from itertools import product
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import rankdata


class StockbeeControlError(ValueError):
    pass


def cross_sectional_deciles(values: Sequence[float]) -> np.ndarray:
    """Assign average-rank deciles, with the largest observation in decile ten."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise StockbeeControlError("decile inputs must be a nonempty finite vector")
    ranks = rankdata(array, method="average")
    return np.clip(np.ceil(10.0 * ranks / len(array)), 1, 10).astype(np.uint8)


def bucket_members(deciles: np.ndarray, eligible: Sequence[bool]) -> dict[tuple[int, ...], np.ndarray]:
    matrix = np.asarray(deciles)
    mask = np.asarray(eligible, dtype=bool)
    if matrix.ndim != 2 or len(matrix) != len(mask) or matrix.shape[1] != 4:
        raise StockbeeControlError("four-decile matrix and eligibility mask differ")
    members: dict[tuple[int, ...], list[int]] = {}
    for index in np.flatnonzero(mask):
        members.setdefault(tuple(int(value) for value in matrix[index]), []).append(int(index))
    return {key: np.asarray(value, dtype=np.int64) for key, value in members.items()}


def control_order_digest(contract_digest: str, event_id: str, candidate_symbol: str) -> str:
    payload = f"{contract_digest}|{event_id}|{candidate_symbol}".encode("utf-8")
    return sha256(payload).hexdigest()


def select_control_indices(
    *, symbols: Sequence[str], members: Mapping[tuple[int, ...], np.ndarray],
    all_eligible_indices: Sequence[int], event_symbol: str, event_deciles: Sequence[int],
    contract_digest: str, event_id: str, controls: int = 5,
) -> tuple[np.ndarray, str]:
    """Select a deterministic unique control set under the frozen relaxation ladder."""
    names = np.asarray(symbols, dtype=object)
    all_indices = np.asarray(all_eligible_indices, dtype=np.int64)
    key = tuple(int(value) for value in event_deciles)
    if len(key) != 4 or any(value < 1 or value > 10 for value in key) or controls < 1:
        raise StockbeeControlError("matching request is invalid")

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
            pool, tier = all_indices, "same_date_horizon_unmatched"
    if len(pool):
        pool = pool[names[pool] != event_symbol]
    ordered = sorted(
        (int(index) for index in pool),
        key=lambda index: (control_order_digest(contract_digest, event_id, str(names[index])), str(names[index])),
    )
    return np.asarray(ordered[:controls], dtype=np.int64), tier


def moving_block_positive_inference(
    monthly_differences: Sequence[float], *, resamples: int, block_length: int, seed: int,
) -> tuple[float, float, float, float]:
    """Mean, one-sided positive-effect p-value, and percentile confidence interval."""
    values = np.asarray(monthly_differences, dtype=np.float64)
    if values.ndim != 1 or len(values) < block_length or block_length < 1 or resamples < 1 \
            or not np.isfinite(values).all():
        raise StockbeeControlError("block-bootstrap inputs are invalid")
    observed = float(values.mean())
    centered = values - observed
    starts = np.arange(len(values) - block_length + 1)
    blocks_needed = int(np.ceil(len(values) / block_length))
    rng = np.random.Generator(np.random.PCG64(seed))
    null_at_least_observed = 0
    bootstrap_means = np.empty(resamples, dtype=np.float64)
    for sample_index in range(resamples):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        positions = np.concatenate([
            np.arange(start, start + block_length, dtype=np.int64) for start in chosen
        ])[:len(values)]
        null_at_least_observed += bool(float(centered[positions].mean()) >= observed)
        bootstrap_means[sample_index] = float(values[positions].mean())
    lower, upper = np.quantile(bootstrap_means, [.025, .975])
    return observed, float((null_at_least_observed + 1) / (resamples + 1)), float(lower), float(upper)
