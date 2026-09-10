from __future__ import annotations

import numpy as np

from experiments.m04r import m04r14_t14_12_final_matches as final
from experiments.m04r import verify_m04r14_t14_12_final_matches as oracle
from market_analogues.post_signal_rank_matching import percentile_rank_matrix, select_rank_nearest_batch


def test_final_weighted_batch_matches_independent_oracle() -> None:
    names = np.array(["EVENT", "A", "B", "C", "D", "E", "F"], dtype=object)
    raw = np.array([
        [3., 5., 3., 3.], [3., 4., 3., 3.], [2., 5., 3., 3.], [3., 6., 3., 3.],
        [4., 5., 3., 3.], [3., 5., 2., 3.], [3., 5., 4., 3.],
    ])
    ranks = percentile_rank_matrix(raw); candidates = np.arange(1, len(names)); events = np.array([0])
    actual = select_rank_nearest_batch(symbols=names, percentile_ranks=ranks, candidate_indices=candidates,
        event_indices=events, contract_digest="contract", signal_ids=["signal"],
        coordinate_weights=final.COORDINATE_WEIGHTS)
    expected = oracle._batch(names, oracle._ranks(raw), candidates, events, "contract", ["signal"])
    assert np.array_equal(actual[0][0], expected[0][0])
    assert np.array_equal(actual[0][1], expected[0][1])


def test_final_weight_is_selected_minimum_passing_weight() -> None:
    assert final.SELECTED_WEIGHT == 4
    assert final.COORDINATE_WEIGHTS == (1., 4., 1., 1.)
    assert final.MATCH_TIER.endswith("weight4")
