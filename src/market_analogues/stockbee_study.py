"""Vectorized, causal feature and forward-outcome kernel for the Stockbee study."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


class StockbeeStudyError(ValueError):
    pass


@dataclass(frozen=True)
class StockbeeKernelSpec:
    horizons: tuple[int, ...] = (21, 63)
    prior_sessions: int = 252
    winner_return: float = .25
    move_threshold: float = .04
    range_multiple: float = 1.5


def _window_count(values: np.ndarray, starts: np.ndarray, width: int) -> np.ndarray:
    cumulative = np.concatenate(([0], np.cumsum(values.astype(np.int64))))
    return cumulative[starts + width] - cumulative[starts]


def _prior_rolling(values: np.ndarray, width: int, reducer: str) -> np.ndarray:
    series = pd.Series(values)
    rolling = series.rolling(width, min_periods=width)
    reduced = rolling.median() if reducer == "median" else rolling.std(ddof=0)
    return reduced.shift(1).to_numpy(dtype=np.float64)


def validate_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    required = ("timestamp", "open", "high", "low", "close", "volume")
    if any(name not in frame for name in required):
        raise StockbeeStudyError("OHLCV columns are incomplete")
    result = frame.loc[:, required].copy()
    result["timestamp"] = pd.to_datetime(result.timestamp)
    if result.timestamp.isna().any() or not result.timestamp.is_monotonic_increasing \
            or result.timestamp.duplicated().any():
        raise StockbeeStudyError("timestamps must be unique and strictly increasing")
    return result.reset_index(drop=True)


def valid_ohlcv_rows(frame: pd.DataFrame) -> np.ndarray:
    bars = validate_ohlcv(frame)
    ohlc = bars[["open", "high", "low", "close"]].to_numpy(dtype=np.float64)
    volume = bars.volume.to_numpy(dtype=np.float64)
    return (
        np.isfinite(ohlc).all(axis=1) & (ohlc > 0).all(axis=1)
        & np.isfinite(volume) & (volume >= 0)
        & (ohlc[:, 1] >= np.maximum.reduce((ohlc[:, 0], ohlc[:, 2], ohlc[:, 3])))
        & (ohlc[:, 2] <= np.minimum.reduce((ohlc[:, 0], ohlc[:, 1], ohlc[:, 3])))
    )


def symbol_risk_rows(
    frame: pd.DataFrame, symbol: str, spec: StockbeeKernelSpec = StockbeeKernelSpec(),
) -> pd.DataFrame:
    bars = validate_ohlcv(frame)
    n = len(bars)
    if len(set(spec.horizons)) != len(spec.horizons) or any(h < 5 for h in spec.horizons):
        raise StockbeeStudyError("horizons must be unique and at least five sessions")
    if n <= spec.prior_sessions + min(spec.horizons):
        return pd.DataFrame()
    open_ = bars.open.to_numpy(dtype=np.float64); high = bars.high.to_numpy(dtype=np.float64)
    low = bars.low.to_numpy(dtype=np.float64); close = bars.close.to_numpy(dtype=np.float64)
    volume = bars.volume.to_numpy(dtype=np.float64); timestamps = bars.timestamp.to_numpy()
    valid_rows = valid_ohlcv_rows(bars)
    previous = np.roll(close, 1); previous[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        close_return = close / previous - 1
        true_range_fraction = np.maximum.reduce((high - low, np.abs(high - previous), np.abs(low - previous))) / previous
    close_return[~valid_rows] = np.nan; true_range_fraction[~valid_rows] = np.nan
    prior_range_median = _prior_rolling(true_range_fraction, 20, "median")
    up_move = close_return >= spec.move_threshold
    range_day = true_range_fraction >= spec.move_threshold
    bullish_expansion = range_day & (close > open_) & (true_range_fraction >= spec.range_multiple * prior_range_median)
    dollar_volume = close * volume; dollar_volume[~valid_rows] = np.nan
    prior_median_dollar_volume = _prior_rolling(dollar_volume, 20, "median")
    with np.errstate(divide="ignore", invalid="ignore"):
        log_returns = np.log(close / previous)
    log_returns[~valid_rows] = np.nan
    prior_volatility = _prior_rolling(log_returns, 20, "std")
    indices = np.arange(n)
    prior_return_63 = np.full(n, np.nan)
    valid_prior = indices >= 64
    prior_return_63[valid_prior] = close[indices[valid_prior] - 1] / close[indices[valid_prior] - 64] - 1
    rows: list[pd.DataFrame] = []
    for horizon in sorted(spec.horizons):
        starts = np.arange(spec.prior_sessions, n - horizon + 1)
        if not len(starts): continue
        invalid_cumulative = np.concatenate(([0], np.cumsum((~valid_rows).astype(np.int64))))
        complete = (
            invalid_cumulative[starts + horizon]
            - invalid_cumulative[starts - spec.prior_sessions]
        ) == 0
        starts = starts[complete]
        if not len(starts): continue
        endpoint = starts + horizon - 1
        outcome = close[endpoint] / close[starts - 1] - 1
        state: dict[str, object] = {
            "symbol": str(symbol), "start": pd.to_datetime(timestamps[starts]),
            "start_position": starts, "horizon_sessions": horizon,
            "forward_close_return": outcome, "winner_25pct": outcome >= spec.winner_return,
            "start_close": close[starts], "prior_return_63": prior_return_63[starts],
            "prior_volatility_20": prior_volatility[starts],
            "prior_median_dollar_volume_20": prior_median_dollar_volume[starts],
        }
        for name, exposure in (
            ("up_close_4pct", up_move), ("true_range_4pct", range_day),
            ("bullish_range_expansion_4pct", bullish_expansion),
        ):
            start_count = exposure[starts].astype(np.int16)
            first5_count = _window_count(exposure, starts, 5).astype(np.int16)
            full_count = _window_count(exposure, starts, horizon).astype(np.int16)
            pre20_count = _window_count(exposure, starts - 20, 20).astype(np.int16)
            state[f"{name}_start_day"] = start_count.astype(bool)
            state[f"{name}_first_5_sessions"] = first5_count > 0
            state[f"{name}_full_move"] = full_count > 0
            state[f"{name}_pre_start_20_sessions"] = pre20_count > 0
            state[f"{name}_first_5_count"] = first5_count
            state[f"{name}_full_move_count"] = full_count
        result = pd.DataFrame(state)
        result["investable"] = (result.start_close >= 5) & (result.prior_median_dollar_volume_20 >= 1_000_000)
        rows.append(result)
    if not rows: return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(
        ["horizon_sessions", "start_position"], kind="stable",
    ).reset_index(drop=True)


def clustered_winners(risk_rows: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "horizon_sessions", "start_position", "winner_25pct", "forward_close_return"}
    if not required.issubset(risk_rows.columns):
        raise StockbeeStudyError("risk rows are incomplete")
    winners = risk_rows.loc[risk_rows.winner_25pct].sort_values(
        ["symbol", "horizon_sessions", "start_position"], kind="stable",
    ).copy()
    if winners.empty:
        return winners.assign(event_run_length=pd.Series(dtype="int64"), event_peak_return=pd.Series(dtype="float64"))
    discontinuity = winners.groupby(["symbol", "horizon_sessions"], sort=False).start_position.diff().fillna(2).ne(1)
    winners["event_run"] = discontinuity.groupby([winners.symbol, winners.horizon_sessions]).cumsum().astype(int)
    grouped = winners.groupby(["symbol", "horizon_sessions", "event_run"], sort=True)
    representatives = grouped.head(1).copy()
    lengths = grouped.size().rename("event_run_length")
    peaks = grouped.forward_close_return.max().rename("event_peak_return")
    representatives = representatives.merge(
        pd.concat([lengths, peaks], axis=1).reset_index(),
        on=["symbol", "horizon_sessions", "event_run"], validate="one_to_one",
    )
    return representatives.sort_values(["start", "symbol", "horizon_sessions"], kind="stable").reset_index(drop=True)
