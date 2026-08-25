from __future__ import annotations

import numpy as np
import pandas as pd


def align_benchmark_close(
    bars: pd.DataFrame, benchmark: pd.DataFrame | None,
) -> np.ndarray:
    """Return exact-timestamp benchmark closes without a full DataFrame merge."""
    if benchmark is None or benchmark.empty:
        return np.full(len(bars), np.nan)
    right = benchmark[["timestamp", "close"]].drop_duplicates(
        "timestamp", keep="last",
    )
    index = pd.Index(right["timestamp"])
    positions = index.get_indexer(pd.Index(bars["timestamp"]))
    values = pd.to_numeric(right["close"], errors="coerce").to_numpy(float)
    output = np.full(len(bars), np.nan)
    matched = positions >= 0
    output[matched] = values[positions[matched]]
    return output


def align_benchmark(bars: pd.DataFrame, benchmark: pd.DataFrame | None) -> pd.DataFrame:
    """Exact-timestamp benchmark alignment; unknown bars remain missing."""
    result = pd.DataFrame({"timestamp": bars["timestamp"].to_numpy()})
    if benchmark is None or benchmark.empty:
        for col in ["benchmark_close", "benchmark_return", "benchmark_path",
                    "benchmark_volatility", "benchmark_drawdown"]:
            result[col] = np.nan
        return result
    result["benchmark_close"] = align_benchmark_close(bars, benchmark)
    close = result["benchmark_close"]
    result["benchmark_return"] = np.log(close / close.shift(1))
    first = close.dropna()
    anchor = first.iloc[0] if len(first) else np.nan
    result["benchmark_path"] = np.log(close / anchor)
    result["benchmark_volatility"] = result["benchmark_return"].rolling(20, min_periods=10).std()
    result["benchmark_drawdown"] = close / close.cummax() - 1
    return result


def relative_channels(bars: pd.DataFrame, aligned: pd.DataFrame) -> pd.DataFrame:
    stock_return = np.log(bars["close"] / bars["close"].shift(1))
    excess = stock_return - aligned["benchmark_return"]
    relative_path = excess.fillna(0).cumsum()
    relative_path[aligned["benchmark_return"].isna()] = np.nan
    return pd.DataFrame({
        "relative_return": excess,
        "relative_path": relative_path,
    }, index=bars.index)
