from __future__ import annotations

from hashlib import sha256
import math

import numpy as np
import pytest

from market_analogues.stockbee_controls import (
    StockbeeControlError, bucket_members, control_order_digest,
    cross_sectional_deciles, moving_block_positive_inference, select_control_indices,
)


def test_average_rank_deciles_are_deterministic_at_ties() -> None:
    values = np.array([1., 2., 2., 4., 5., 6., 7., 8., 9., 10.])
    observed = cross_sectional_deciles(values)
    assert observed.tolist() == [1, 3, 3, 4, 5, 6, 7, 8, 9, 10]
    assert cross_sectional_deciles([7., 7., 7.]).tolist() == [7, 7, 7]


def test_control_selection_uses_first_sufficient_tier_and_hash_order() -> None:
    symbols = np.array(["EVENT", "A", "B", "C", "D", "E", "F", "G"], dtype=object)
    deciles = np.array([
        [5, 5, 5, 5], [5, 5, 5, 5], [5, 5, 5, 5], [5, 5, 5, 5],
        [5, 5, 5, 5], [5, 5, 5, 5], [6, 6, 6, 6], [9, 9, 9, 9],
    ], dtype=np.uint8)
    eligible = np.array([False, True, True, True, True, True, True, True])
    members = bucket_members(deciles, eligible)
    selected, tier = select_control_indices(
        symbols=symbols, members=members, all_eligible_indices=np.flatnonzero(eligible),
        event_symbol="EVENT", event_deciles=deciles[0], contract_digest="contract", event_id="event",
    )
    expected = sorted(range(1, 6), key=lambda i: (control_order_digest("contract", "event", symbols[i]), symbols[i]))
    assert tier == "exact_all_four_deciles"
    assert selected.tolist() == expected
    assert len(set(symbols[selected])) == 5


def test_control_selection_relaxes_then_retains_shortfall() -> None:
    symbols = np.array(["EVENT", "A", "B", "C", "D", "E", "F"], dtype=object)
    deciles = np.array([
        [5, 5, 5, 5], [5, 5, 5, 5], [5, 5, 5, 5], [4, 5, 5, 5],
        [6, 6, 6, 6], [5, 4, 5, 5], [10, 10, 10, 10],
    ], dtype=np.uint8)
    eligible = np.array([False, True, True, True, True, True, True])
    selected, tier = select_control_indices(
        symbols=symbols, members=bucket_members(deciles, eligible),
        all_eligible_indices=np.flatnonzero(eligible), event_symbol="EVENT", event_deciles=deciles[0],
        contract_digest="c", event_id="e",
    )
    assert tier == "within_one_bucket_all_four" and len(selected) == 5
    short, short_tier = select_control_indices(
        symbols=symbols[:4], members=bucket_members(deciles[:4], eligible[:4]),
        all_eligible_indices=np.flatnonzero(eligible[:4]), event_symbol="EVENT", event_deciles=deciles[0],
        contract_digest="c", event_id="short",
    )
    assert short_tier == "same_date_horizon_unmatched" and len(short) == 3


def _bootstrap_reference(values: np.ndarray, resamples: int, block: int, seed: int) -> tuple[float, float, float, float]:
    observed = values.mean(); centered = values - observed
    starts = np.arange(len(values) - block + 1); needed = math.ceil(len(values) / block)
    rng = np.random.Generator(np.random.PCG64(seed)); null = 0; estimates = []
    for _ in range(resamples):
        chosen = rng.choice(starts, size=needed, replace=True)
        indices = np.concatenate([np.arange(start, start + block) for start in chosen])[:len(values)]
        null += centered[indices].mean() >= observed; estimates.append(values[indices].mean())
    low, high = np.quantile(estimates, [.025, .975])
    return float(observed), (null + 1) / (resamples + 1), float(low), float(high)


def test_positive_block_bootstrap_matches_independent_seeded_oracle() -> None:
    values = np.array([-.1, .2, .3, .05, .4, .2, -.05, .1])
    observed = moving_block_positive_inference(values, resamples=200, block_length=3, seed=9)
    assert observed == _bootstrap_reference(values, 200, 3, 9)


@pytest.mark.parametrize("function,args", [
    (cross_sectional_deciles, ([1., np.nan],)),
    (bucket_members, (np.ones((2, 3)), [True, True])),
    (lambda values: moving_block_positive_inference(values, resamples=10, block_length=3, seed=1), ([1., 2.],)),
])
def test_control_kernels_reject_invalid_inputs(function, args) -> None:
    with pytest.raises(StockbeeControlError): function(*args)
