from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.decision.chart_value import FEATURES, benchmark_features, stock_features
from experiments.p0.features import p0_stock_features
from experiments.p0.labels import ladder_label_scalar, ladder_labels


def _random_ohlc(n: int, seed: int, gap_prob: float = 0.05):
    rng = np.random.default_rng(seed)
    close = 50 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
    open_ = close * np.exp(rng.normal(0, 0.01, n))
    gaps = rng.random(n) < gap_prob
    open_[gaps] *= np.exp(rng.normal(0, 0.12, gaps.sum()))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.02, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.02, n)))
    return open_, high, low, close


def test_vectorized_labels_match_scalar_oracle():
    for seed in range(5):
        o, h, l, c = _random_ohlc(400, seed)
        for horizon in (5, 20, 40):
            lab = ladder_labels(o, h, l, c, horizon)
            for origin in range(25, len(c) - horizon - 1):
                expected = ladder_label_scalar(o, h, l, c, origin, horizon)
                assert expected is not None
                assert lab["cls"][origin] == expected[0], (seed, horizon, origin)
                assert np.isclose(lab["realized_r"][origin], expected[1])
            assert (lab["cls"][len(c) - horizon - 1:] == -1).all()


def test_all_classes_occur_and_stop_counts_first():
    o, h, l, c = _random_ohlc(3000, 11)
    cls = ladder_labels(o, h, l, c, 20)["cls"]
    assert set(np.unique(cls[cls >= 0])) == set(range(6))
    # a bar that touches both +5R and the stop is a stop-out with no reach
    n = 60
    o = np.full(n, 100.0); c = np.full(n, 100.0)
    h = np.full(n, 101.0); l = np.full(n, 99.0)  # ATR = 2, R = 2.5
    h[31], l[31] = 120.0, 90.0
    lab = ladder_labels(o, h, l, c, 5)
    assert lab["cls"][30] == 0 and np.isclose(lab["realized_r"][30], -1.0)


def test_gap_through_stop_exits_at_open():
    n = 60
    o = np.full(n, 100.0); c = np.full(n, 100.0)
    h = np.full(n, 101.0); l = np.full(n, 99.0)
    o[33], h[33], l[33], c[33] = 90.0, 91.0, 89.0, 90.0  # gap 4R below entry
    lab = ladder_labels(o, h, l, c, 5)
    assert lab["cls"][30] == 0 and np.isclose(lab["realized_r"][30], -4.0)


def test_features_do_not_see_the_future():
    o, h, l, c = _random_ohlc(700, 3)
    ts = pd.bdate_range("2010-01-01", periods=len(c))
    stock = pd.DataFrame({"timestamp": ts, "open": o, "high": h, "low": l, "close": c,
                          "volume": np.random.default_rng(3).integers(1e4, 1e6, len(c))})
    bench = benchmark_features(pd.DataFrame({"timestamp": ts, "close": c[::-1].copy()}))
    full = stock_features(stock, bench)
    for cut in (400, 550, 650):
        truncated = stock_features(stock.iloc[:cut + 1], bench)
        pd.testing.assert_series_equal(full.iloc[cut][list(FEATURES)],
                                       truncated.iloc[cut][list(FEATURES)])


def test_p0_features_are_causal_and_survive_flat_bars():
    o, h, l, c = _random_ohlc(700, 5)
    h[500:505] = l[500:505] = c[500:505] = o[500:505]  # locked bars
    ts = pd.bdate_range("2010-01-01", periods=len(c))
    stock = pd.DataFrame({"timestamp": ts, "open": o, "high": h, "low": l, "close": c,
                          "volume": np.random.default_rng(5).integers(1e4, 1e6, len(c))})
    bench = benchmark_features(pd.DataFrame({"timestamp": ts, "close": c[::-1].copy()}))
    full = p0_stock_features(stock, bench)
    assert full.iloc[510][list(FEATURES)].notna().all()
    for cut in (450, 510, 650):
        truncated = p0_stock_features(stock.iloc[:cut + 1], bench)
        pd.testing.assert_series_equal(full.iloc[cut][list(FEATURES)],
                                       truncated.iloc[cut][list(FEATURES)])
