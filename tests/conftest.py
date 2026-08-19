from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def make_bars(n: int = 300, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="B"),
        "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.integers(10_000, 1_000_000, n),
    })


@pytest.fixture
def bars() -> pd.DataFrame:
    return make_bars()


@pytest.fixture
def directory_dataset(tmp_path: Path, bars: pd.DataFrame) -> Path:
    root = tmp_path / "bars"
    root.mkdir()
    bars.to_parquet(root / "AAA.parquet", index=False)
    (bars.assign(close=bars.close * 2, open=bars.open * 2,
                 high=bars.high * 2, low=bars.low * 2)
         .to_parquet(root / "BBB.parquet", index=False))
    return root

