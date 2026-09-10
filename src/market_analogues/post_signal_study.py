"""Causal signal-day panel and strictly subsequent outcome kernel for T14-12."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from market_analogues.stockbee_study import validate_ohlcv, valid_ohlcv_rows


class PostSignalStudyError(ValueError):
    pass


@dataclass(frozen=True)
class PostSignalSpec:
    horizons: tuple[int, ...] = (5, 20, 60)
    prior_sessions: int = 252
    refractory_sessions: int = 20
    move_threshold: float = .04
    range_multiple: float = 1.5
    favorable_atr: float = 2.0
    adverse_atr: float = 1.0


def _rolling_prior(values: np.ndarray, width: int, reducer: str) -> np.ndarray:
    rolling = pd.Series(values).rolling(width, min_periods=width)
    result = rolling.median() if reducer == "median" else rolling.std(ddof=0)
    return result.shift(1).to_numpy(np.float64)


def _window_count(values: np.ndarray, starts: np.ndarray, width: int) -> np.ndarray:
    cumulative = np.concatenate(([0], np.cumsum(values.astype(np.int64))))
    return cumulative[starts + width] - cumulative[starts]


def _benchmark_sessions(frame: pd.DataFrame) -> pd.DataFrame:
    source = frame.rename(columns={"date": "timestamp"}) if "date" in frame and "timestamp" not in frame else frame
    required = ("timestamp", "open", "high", "low", "close")
    if any(column not in source for column in required):
        raise PostSignalStudyError("benchmark columns are incomplete")
    result = source.loc[:, required].copy(); result["timestamp"] = pd.to_datetime(result.timestamp)
    numeric = result[["open", "high", "low", "close"]].to_numpy(float)
    if result.timestamp.isna().any() or result.timestamp.duplicated().any() \
            or not result.timestamp.is_monotonic_increasing or not np.isfinite(numeric).all() \
            or (numeric <= 0).any():
        raise PostSignalStudyError("benchmark sessions are invalid")
    return result.reset_index(drop=True)


def _future_extreme(values: np.ndarray, horizon: int, reducer: str) -> np.ndarray:
    result = np.full(len(values), np.nan)
    if len(values) <= horizon: return result
    windows = np.lib.stride_tricks.sliding_window_view(values[1:], horizon)
    reduced = windows.max(axis=1) if reducer == "max" else windows.min(axis=1)
    result[:len(reduced)] = reduced
    return result


def _barrier_codes(
    high: np.ndarray, low: np.ndarray, entry: np.ndarray, atr: np.ndarray,
    complete: np.ndarray, horizon: int, favorable_multiple: float, adverse_multiple: float,
) -> np.ndarray:
    # -1 censored, 0 no touch, 1 favorable first, 2 adverse first, 3 same-bar ambiguous.
    codes = np.full(len(high), -1, dtype=np.int8)
    if len(high) <= horizon: return codes
    high_windows = np.lib.stride_tricks.sliding_window_view(high[1:], horizon)
    low_windows = np.lib.stride_tricks.sliding_window_view(low[1:], horizon)
    valid_positions = np.flatnonzero(complete[:len(high_windows)] & np.isfinite(atr[:len(high_windows)]))
    if not len(valid_positions): return codes
    upper = entry[valid_positions] + favorable_multiple * atr[valid_positions]
    lower = entry[valid_positions] - adverse_multiple * atr[valid_positions]
    up = high_windows[valid_positions] >= upper[:, None]
    down = low_windows[valid_positions] <= lower[:, None]
    any_up, any_down = up.any(axis=1), down.any(axis=1)
    first_up = np.where(any_up, up.argmax(axis=1), horizon + 1)
    first_down = np.where(any_down, down.argmax(axis=1), horizon + 1)
    resolved = np.zeros(len(valid_positions), dtype=np.int8)
    resolved[first_up < first_down] = 1; resolved[first_down < first_up] = 2
    resolved[(first_up == first_down) & any_up & any_down] = 3
    codes[valid_positions] = resolved
    return codes


def symbol_post_signal_panel(
    frame: pd.DataFrame, benchmark_frame: pd.DataFrame, symbol: str,
    spec: PostSignalSpec = PostSignalSpec(),
) -> pd.DataFrame:
    """Return causal daily eligibility plus outcomes that begin at the next session open."""
    if tuple(sorted(set(spec.horizons))) != tuple(spec.horizons) or tuple(spec.horizons) != (5, 20, 60) \
            or spec.prior_sessions < 64 or spec.refractory_sessions < 1:
        raise PostSignalStudyError("post-signal specification differs")
    bars = validate_ohlcv(frame); benchmark = _benchmark_sessions(benchmark_frame)
    n = len(bars)
    if n <= spec.prior_sessions: return pd.DataFrame()
    open_ = bars.open.to_numpy(float); high = bars.high.to_numpy(float)
    low = bars.low.to_numpy(float); close = bars.close.to_numpy(float)
    volume = bars.volume.to_numpy(float); timestamps = bars.timestamp.to_numpy()
    valid = valid_ohlcv_rows(bars)
    previous = np.roll(close, 1); previous[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        close_return = close / previous - 1
        true_range = np.maximum.reduce((high - low, np.abs(high - previous), np.abs(low - previous)))
        true_range_fraction = true_range / previous
        log_return = np.log(close / previous)
    close_return[~valid] = np.nan; true_range[~valid] = np.nan
    true_range_fraction[~valid] = np.nan; log_return[~valid] = np.nan
    prior_range_median = _rolling_prior(true_range_fraction, 20, "median")
    prior_volatility = _rolling_prior(log_return, 20, "std")
    dollar_volume = close * volume; dollar_volume[~valid] = np.nan
    prior_dollar_volume = _rolling_prior(dollar_volume, 20, "median")
    up_close = close_return >= spec.move_threshold
    bullish_expansion = (
        (true_range_fraction >= spec.move_threshold) & (close > open_)
        & (true_range_fraction >= spec.range_multiple * prior_range_median)
    )
    positions = np.arange(spec.prior_sessions, n)
    invalid_cumulative = np.concatenate(([0], np.cumsum((~valid).astype(np.int64))))
    causal_valid = invalid_cumulative[positions + 1] - invalid_cumulative[positions - spec.prior_sessions] == 0
    positions = positions[causal_valid]
    if not len(positions): return pd.DataFrame()
    up_prior_count = _window_count(up_close, positions - spec.refractory_sessions, spec.refractory_sessions)
    expansion_prior_count = _window_count(
        bullish_expansion, positions - spec.refractory_sessions, spec.refractory_sessions,
    )

    benchmark_timestamps = benchmark.timestamp.to_numpy()
    benchmark_positions = {pd.Timestamp(value): index for index, value in enumerate(benchmark_timestamps)}
    stock_benchmark_position = np.asarray(
        [benchmark_positions.get(pd.Timestamp(value), -1) for value in timestamps], dtype=np.int64,
    )
    calendar_break = stock_benchmark_position[1:] != stock_benchmark_position[:-1] + 1
    calendar_break_cumulative = np.concatenate(([0], np.cumsum(calendar_break.astype(np.int64))))
    positions = positions[stock_benchmark_position[positions] >= 0]
    if not len(positions): return pd.DataFrame()
    # Recompute refractory counts after the exact-market-session filter changed the selected vector.
    up_prior_count = _window_count(up_close, positions - spec.refractory_sessions, spec.refractory_sessions)
    expansion_prior_count = _window_count(
        bullish_expansion, positions - spec.refractory_sessions, spec.refractory_sessions,
    )
    benchmark_index = stock_benchmark_position[positions]
    benchmark_close = benchmark.close.to_numpy(float); benchmark_open = benchmark.open.to_numpy(float)
    benchmark_signal_return = np.full(len(positions), np.nan)
    benchmark_return_20 = np.full(len(positions), np.nan); benchmark_return_63 = np.full(len(positions), np.nan)
    benchmark_volatility_20 = np.full(len(positions), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        benchmark_log_return = np.log(benchmark_close[1:] / benchmark_close[:-1])
    for output_index, market_index in enumerate(benchmark_index):
        if market_index >= 1: benchmark_signal_return[output_index] = benchmark_close[market_index] / benchmark_close[market_index - 1] - 1
        if market_index >= 20:
            benchmark_return_20[output_index] = benchmark_close[market_index] / benchmark_close[market_index - 20] - 1
            benchmark_volatility_20[output_index] = np.std(benchmark_log_return[market_index - 20:market_index], ddof=0)
        if market_index >= 63: benchmark_return_63[output_index] = benchmark_close[market_index] / benchmark_close[market_index - 63] - 1

    state: dict[str, object] = {
        "symbol": str(symbol), "signal_date": pd.to_datetime(timestamps[positions]),
        "signal_position": positions, "prior_close": close[positions - 1],
        "prior_return_63": close[positions - 1] / close[positions - 64] - 1,
        "prior_volatility_20": prior_volatility[positions],
        "prior_median_dollar_volume_20": prior_dollar_volume[positions],
        "signal_close": close[positions], "signal_atr_20": pd.Series(true_range).rolling(20, min_periods=20).mean().to_numpy()[positions],
        "up_close_4pct": up_close[positions], "up_close_at_risk": up_prior_count == 0,
        "up_close_signal_event": up_close[positions] & (up_prior_count == 0),
        "bullish_range_expansion_4pct": bullish_expansion[positions],
        "bullish_range_expansion_at_risk": expansion_prior_count == 0,
        "bullish_range_expansion_signal_event": bullish_expansion[positions] & (expansion_prior_count == 0),
        "benchmark_signal_day_return": benchmark_signal_return,
        "benchmark_return_20": benchmark_return_20, "benchmark_return_63": benchmark_return_63,
        "benchmark_volatility_20": benchmark_volatility_20,
    }
    state["investable"] = (state["prior_close"] >= 5) & (state["prior_median_dollar_volume_20"] >= 1_000_000)
    future_invalid_cumulative = invalid_cumulative
    for horizon in spec.horizons:
        complete = np.zeros(len(positions), dtype=bool)
        status = np.full(len(positions), "source_end_before_horizon", dtype=object)
        endpoint_positions = positions + horizon
        within_stock = endpoint_positions < n
        candidates = np.flatnonzero(within_stock)
        if len(candidates):
            p = positions[candidates]; endpoint = endpoint_positions[candidates]
            future_valid = future_invalid_cumulative[endpoint + 1] - future_invalid_cumulative[p + 1] == 0
            market_start = stock_benchmark_position[p]
            market_complete = (
                (calendar_break_cumulative[endpoint] - calendar_break_cumulative[p] == 0)
                & (market_start + horizon < len(benchmark))
            )
            complete[candidates] = future_valid & market_complete
            status[candidates] = np.where(
                ~future_valid, "invalid_future_ohlcv",
                np.where(~market_complete, "missing_benchmark_session", "complete"),
            )
        entry = np.full(len(positions), np.nan); endpoint_close = np.full(len(positions), np.nan)
        close_outcome = np.full(len(positions), np.nan); relative_log = np.full(len(positions), np.nan)
        mfe = np.full(len(positions), np.nan); mae = np.full(len(positions), np.nan)
        if complete.any():
            selected = np.flatnonzero(complete); p = positions[selected]; endpoint = p + horizon
            entry[selected] = open_[p + 1]; endpoint_close[selected] = close[endpoint]
            stock_gross = endpoint_close[selected] / entry[selected]
            market_position = stock_benchmark_position[p]
            market_gross = benchmark_close[market_position + horizon] / benchmark_open[market_position + 1]
            close_outcome[selected] = stock_gross - 1
            relative_log[selected] = np.log(stock_gross) - np.log(market_gross)
            max_high = _future_extreme(high, horizon, "max")[p]
            min_low = _future_extreme(low, horizon, "min")[p]
            mfe[selected] = max_high / entry[selected] - 1; mae[selected] = min_low / entry[selected] - 1
        state[f"complete_{horizon}"] = complete
        state[f"status_{horizon}"] = status
        state[f"entry_open_{horizon}"] = entry
        state[f"endpoint_close_return_{horizon}"] = close_outcome
        state[f"benchmark_relative_log_return_{horizon}"] = relative_log
        state[f"endpoint_gain_25pct_{horizon}"] = np.where(complete, close_outcome >= .25, np.nan)
        state[f"maximum_favorable_excursion_{horizon}"] = mfe
        state[f"maximum_adverse_excursion_{horizon}"] = mae
        if horizon == 20:
            complete_by_position = np.zeros(n, dtype=bool); complete_by_position[positions] = complete
            entry_by_position = np.full(n, np.nan); entry_by_position[positions] = entry
            atr_by_position = np.full(n, np.nan); atr_by_position[positions] = np.asarray(state["signal_atr_20"])
            codes = _barrier_codes(
                high, low, entry_by_position, atr_by_position, complete_by_position, horizon,
                spec.favorable_atr, spec.adverse_atr,
            )
            state["barrier_code_20"] = codes[positions]
    return pd.DataFrame(state).sort_values("signal_position", kind="stable").reset_index(drop=True)
