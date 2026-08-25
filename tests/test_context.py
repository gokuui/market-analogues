from __future__ import annotations

import numpy as np
import pandas as pd

from market_analogues.context import (
    align_benchmark, align_benchmark_close, relative_channels,
)


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


def test_close_lookup_matches_last_duplicate_merge_semantics(
    bars: pd.DataFrame,
) -> None:
    stock = _canonical(bars.iloc[:30].copy())
    benchmark = stock[["timestamp", "close"]].copy()
    duplicate = benchmark.iloc[[7]].copy()
    duplicate["close"] *= 2
    benchmark = pd.concat((benchmark, duplicate), ignore_index=True)
    benchmark = benchmark.drop(index=10).reset_index(drop=True)
    right = benchmark[["timestamp", "close"]].drop_duplicates(
        "timestamp", keep="last",
    ).rename(columns={"close": "benchmark_close"})
    expected = pd.DataFrame({
        "timestamp": stock.timestamp.to_numpy(),
    }).merge(
        right, on="timestamp", how="left", sort=False,
    ).benchmark_close.to_numpy(float)
    actual = align_benchmark_close(stock, benchmark)
    np.testing.assert_array_equal(actual, expected)
    assert actual[7] == duplicate.close.iloc[0]
    assert np.isnan(actual[10])


def test_relative_return_is_stock_return_minus_market_return(bars: pd.DataFrame) -> None:
    stock = _canonical(bars.iloc[:30].copy())
    benchmark = stock[["timestamp", "close"]].copy()
    benchmark["close"] = 100 * np.exp(np.arange(len(benchmark)) * 0.001)
    aligned = align_benchmark(stock, benchmark)
    relative = relative_channels(stock, aligned)
    expected = np.log(stock.close.iloc[-1] / stock.close.iloc[-2]) - 0.001
    assert relative.relative_return.iloc[-1] == pytest.approx(expected)


import pytest
