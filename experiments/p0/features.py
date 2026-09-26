"""P0 causal features: the decision study's 22 features, made robust to flat bars.

In the decision study, range_contraction_10_63 and close_location_20 divided by
(high - low) with zero ranges turned into NaN, so one flat (e.g. circuit-locked)
bar blanked those features for 10-63 sessions. On NSE this silently dropped 58%
of strategy entries, and they were systematically different trades. Here a flat
bar contributes a zero range, the close location skips flat bars, and remaining
gaps stay NaN for the gradient-boosted heads to route natively. The original
function is left unchanged so the decision results stay reproducible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.decision.chart_value import FEATURES, HISTORY, stock_features

REQUIRED = ("ret_5", "ret_20", "ret_63", "ret_126", "ret_252", "vol_20", "vol_63",
            "atr_pct_20", "dist_high_252", "dist_low_252", "dist_sma_50", "dist_sma_200",
            "log_dollar_volume_20")


def p0_stock_features(stock: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame:
    frame = stock_features(stock, bench)
    high, low, close = (stock[c].astype(float).to_numpy() for c in ("high", "low", "close"))
    spread = high - low
    rng = pd.Series(spread / close)
    frame["range_contraction_10_63"] = (rng.rolling(10).mean()
                                        / rng.rolling(63).mean().replace(0.0, np.nan)).to_numpy()
    location = pd.Series(np.where(spread > 0, (close - low) / np.where(spread > 0, spread, 1), np.nan))
    frame["close_location_20"] = location.rolling(20, min_periods=5).mean().to_numpy()
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame.iloc[:HISTORY] = np.nan
    return frame.loc[:, list(FEATURES)]
