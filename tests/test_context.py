from __future__ import annotations

import numpy as np
import pandas as pd

from market_analogues.context import align_benchmark, relative_channels


def _canonical(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(columns={"date": "timestamp"})


def test_benchmark_alignment_is_exact_and_does_not_fill_missing(bars: pd.DataFrame) -> None:
    stock = _canonical(bars.iloc[:30].copy())
    benchmark = stock[["timestamp", "close"]].copy()
    missing_timestamp = benchmark.loc[10, "timestamp"]
    benchmark = benchmark.drop(index=10).reset_index(drop=True)

    aligned = align_benchmark(stock, benchmark)

    row = aligned[aligned.timestamp == missing_timestamp].iloc[0]
    assert np.isnan(row.benchmark_close)
    assert np.isnan(row.benchmark_return)


def test_relative_return_is_stock_return_minus_market_return(bars: pd.DataFrame) -> None:
    stock = _canonical(bars.iloc[:30].copy())
    benchmark = stock[["timestamp", "close"]].copy()
    benchmark["close"] = 100 * np.exp(np.arange(len(benchmark)) * 0.001)
    aligned = align_benchmark(stock, benchmark)
    relative = relative_channels(stock, aligned)
    expected = np.log(stock.close.iloc[-1] / stock.close.iloc[-2]) - 0.001
    assert relative.relative_return.iloc[-1] == pytest.approx(expected)


import pytest
