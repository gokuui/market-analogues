from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from market_analogues.post_signal_rank_matching import (
    RankMatchingError, exhaustive_rank_nearest, percentile_rank_matrix, select_rank_nearest_batch,
)
from experiments.m04r import m04r14_t14_12_rank_matches as stage


def test_percentile_ranks_are_average_rank_and_scale_invariant() -> None:
    values = np.array([[1., 3., 2., 8.], [2., 3., 4., 7.], [3., 9., 6., 6.]])
    observed = percentile_rank_matrix(values)
    assert observed[:, 0].tolist() == [1 / 3, 2 / 3, 1.]
    assert observed[:, 1].tolist() == [.5, .5, 1.]
    assert np.array_equal(observed, percentile_rank_matrix(values * np.array([2., 4., 8., 16.])))


@pytest.mark.parametrize("seed", [1, 7, 29, 103])
def test_batched_tree_matches_exhaustive_oracle_with_ties_and_shortfall(seed: int) -> None:
    rng = np.random.default_rng(seed); values = rng.normal(size=(80, 4)); values[20:30] = values[10]
    ranks = percentile_rank_matrix(values); symbols = np.array([f"S{i:03d}" for i in range(80)], dtype=object)
    candidates = np.arange(10, 80); events = np.arange(5); ids = [f"event-{i}" for i in events]
    observed = select_rank_nearest_batch(symbols=symbols, percentile_ranks=ranks, candidate_indices=candidates,
                                         event_indices=events, contract_digest="contract", signal_ids=ids)
    for event, signal_id, (selected, squared) in zip(events, ids, observed):
        expected, distances = exhaustive_rank_nearest(symbols=symbols, percentile_ranks=ranks,
            candidate_indices=candidates, event_index=event, contract_digest="contract", signal_id=signal_id)
        assert selected.tolist() == expected.tolist() and np.array_equal(squared, distances)
    short = select_rank_nearest_batch(symbols=symbols, percentile_ranks=ranks, candidate_indices=[10, 11],
                                      event_indices=[0], contract_digest="contract", signal_ids=["short"])[0]
    assert len(short[0]) == 2


def test_rank_matcher_rejects_bad_shapes_and_duplicate_indices() -> None:
    with pytest.raises(RankMatchingError): percentile_rank_matrix(np.ones((3, 3)))
    with pytest.raises(RankMatchingError):
        select_rank_nearest_batch(symbols=["A", "B"], percentile_ranks=np.ones((2, 4)),
                                  candidate_indices=[1, 1], event_indices=[0],
                                  contract_digest="c", signal_ids=["e"])


def test_v2_year_checkpoint_is_atomic_and_restartable(tmp_path) -> None:
    rows = []
    for index in range(12):
        rows.append({"symbol": f"S{index:02d}", "signal_date": pd.Timestamp("2024-01-03"),
            "signal_position": 300, "investable": True, "prior_return_63": index / 100,
            "prior_volatility_20": .1 + index / 1000, "prior_close": 10. + index,
            "prior_median_dollar_volume_20": 2e6 + index,
            "benchmark_signal_day_return": .01, "benchmark_return_20": .02,
            "benchmark_return_63": .03, "benchmark_volatility_20": .04,
            "up_close_at_risk": True, "up_close_4pct": index == 0, "up_close_signal_event": index == 0,
            "bullish_range_expansion_at_risk": True, "bullish_range_expansion_4pct": index == 1,
            "bullish_range_expansion_signal_event": index == 1})
    cache = tmp_path / "cache"; cache.mkdir(); frame = pd.DataFrame(rows)
    first = stage._write_year(2024, frame, cache, "contract")
    second = stage._write_year(2024, frame, cache, "contract")
    assert first == second and first["event_match_rows"] == 4 and first["control_identity_rows"] == 20
    controls = pd.read_parquet(cache / "year-2024/control-identities.parquet")
    assert controls.squared_rank_distance.ge(0).all()
    assert controls.groupby(["population", "signal_name"]).match_rank.apply(
        lambda values: list(values) == [1, 2, 3, 4, 5],
    ).all()
