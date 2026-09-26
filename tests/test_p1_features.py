from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.p1.features import E1, E2, e1_features, e2_features


def _stock(n=900, seed=1):
    rng = np.random.default_rng(seed)
    close = 40 * np.exp(np.cumsum(rng.normal(0.0005, 0.025, n)))
    open_ = close * np.exp(rng.normal(0, 0.01, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.015, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.015, n)))
    return pd.DataFrame({"timestamp": pd.bdate_range("2012-01-02", periods=n), "open": open_,
                         "high": high, "low": low, "close": close,
                         "volume": rng.integers(1e4, 1e6, n).astype(float)})


def test_e1_is_causal():
    stock = _stock()
    bench = np.exp(np.cumsum(np.random.default_rng(2).normal(0, 0.01, len(stock))))
    full = e1_features(stock, bench)
    for cut in (400, 620, 880):
        part = e1_features(stock.iloc[:cut + 1], bench[:cut + 1])
        pd.testing.assert_series_equal(full.iloc[cut][list(E1)], part.iloc[cut][list(E1)],
                                       check_names=False)


def test_e2_is_causal_and_populated():
    stock = _stock()
    rows = np.arange(300, len(stock), 7)
    full = e2_features(stock, rows)
    assert full["down_1_depth"].notna().mean() > 0.9
    # depth is in the current ATR; confirmation used the ATR when the pivot formed
    assert full["down_1_depth"].dropna().median() > 1.5
    for cut in (405, 615, 874):
        part = e2_features(stock.iloc[:cut + 1], np.array([cut]))
        pd.testing.assert_series_equal(full.loc[cut, list(E2)], part.loc[cut, list(E2)],
                                       check_names=False)
