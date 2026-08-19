from __future__ import annotations

import numpy as np
import pandas as pd

from market_analogues.representation import COARSE_LAYOUT, dense_channels, represent
from market_analogues.types import Episode, EpisodeKey, InstrumentKey


def _episode(bars: pd.DataFrame, benchmark: pd.DataFrame | None = None) -> Episode:
    canonical = bars.rename(columns={"date": "timestamp"}).reset_index(drop=True)
    key = EpisodeKey(InstrumentKey("test", "AAA"), canonical.timestamp.iloc[-1], len(canonical), "dense-v1")
    return Episode(key, canonical, benchmark)


def test_representation_shape_and_required_information(bars: pd.DataFrame) -> None:
    ep = _episode(bars.iloc[-252:])
    rep = represent(ep)
    assert rep.coarse.shape == (128,)
    assert np.isfinite(rep.coarse).all()
    required = {
        "close_path", "return", "overnight", "intraday", "range_pct",
        "volume_robust_z", "return_shock_z", "compression_ratio",
        "distance_high_63", "distance_ma_20", "benchmark_path", "relative_path",
    }
    assert required <= set(rep.channels)
    assert sum(COARSE_LAYOUT.values()) == 128


def test_price_and_volume_unit_scaling_is_invariant(bars: pd.DataFrame) -> None:
    base = bars.iloc[-252:].copy()
    scaled = base.copy()
    scaled[["open", "high", "low", "close"]] *= 17.5
    scaled["volume"] *= 100
    a = dense_channels(_episode(base))
    b = dense_channels(_episode(scaled))
    compare = [
        "close_path", "return", "overnight", "intraday", "range_pct",
        "body_atr", "upper_wick_atr", "lower_wick_atr", "close_location",
        "atr_pct", "volume_robust_z", "return_shock_z", "compression_ratio",
        "distance_high_63", "distance_ma_20", "distance_ma_50",
    ]
    np.testing.assert_allclose(a[compare], b[compare], equal_nan=True, atol=1e-10)


def test_future_mutation_cannot_change_cutoff_representation(bars: pd.DataFrame) -> None:
    original = bars.copy()
    cutoff_index = 220
    before = original.iloc[:cutoff_index].copy()
    mutated = original.copy()
    mutated.loc[cutoff_index:, ["open", "high", "low", "close", "volume"]] *= 50
    after = mutated.iloc[:cutoff_index].copy()
    np.testing.assert_array_equal(represent(_episode(before)).coarse, represent(_episode(after)).coarse)


def test_benchmark_future_mutation_cannot_change_representation(bars: pd.DataFrame) -> None:
    stock = bars.iloc[:220].copy()
    benchmark = bars.rename(columns={"date": "timestamp"})[["timestamp", "close"]].copy()
    cutoff = stock.date.iloc[-1]
    first_benchmark = benchmark[benchmark.timestamp <= cutoff].copy()
    benchmark.loc[benchmark.timestamp > cutoff, "close"] *= 1000
    second_benchmark = benchmark[benchmark.timestamp <= cutoff].copy()
    np.testing.assert_array_equal(
        represent(_episode(stock, first_benchmark)).coarse,
        represent(_episode(stock, second_benchmark)).coarse,
    )


def test_zero_volume_is_handled_without_infinite_features(bars: pd.DataFrame) -> None:
    sample = bars.iloc[-252:].copy()
    sample.loc[sample.index[::7], "volume"] = 0
    channels = dense_channels(_episode(sample))
    assert not np.isinf(channels.select_dtypes(include=["number"]).to_numpy()).any()
