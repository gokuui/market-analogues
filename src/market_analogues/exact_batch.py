from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import warnings

from numba import njit
import numpy as np
import pandas as pd

from .context import align_benchmark_close
from .distance import GROUPS, DistanceConfig
from .representation import COARSE_LAYOUT, Representation


EXACT_BATCH_VERSION = "exact-batch-v1"
EPS = 1e-12


@dataclass(frozen=True)
class ExactBatch:
    positions: np.ndarray
    representations: tuple[Representation, ...]
    seconds: float


@dataclass(frozen=True)
class LowerBoundBatch:
    totals: np.ndarray
    components: dict[str, np.ndarray]
    rigid_price: np.ndarray


_DTW_CHANNELS = ("close_path", "atr_pct", "volume_robust_z", "relative_path")
_DTW_SAMPLES = 64


@njit(cache=True, nogil=True)
def _batch_exact_price_distances_compiled(
    query: np.ndarray,
    query_presence: np.ndarray,
    candidates: np.ndarray,
    candidate_presence: np.ndarray,
    rigid_price: np.ndarray,
    band_fraction: float,
) -> np.ndarray:
    """Complete distance-v1's price component without per-row allocations.

    This deliberately preserves the scalar implementation's channel scaling,
    DTW predecessor order and path-length normalization.  Parallelism remains
    outside this kernel because production groups candidates by symbol and
    scores those independent groups concurrently.
    """
    rows = len(candidates)
    output = np.empty(rows, dtype=np.float64)
    band = max(int(_DTW_SAMPLES * band_fraction), 1)
    for row in range(rows):
        included = np.zeros(len(_DTW_CHANNELS), dtype=np.bool_)
        scales = np.ones(len(_DTW_CHANNELS), dtype=np.float64)
        count = 0
        for channel in range(len(_DTW_CHANNELS)):
            if not query_presence[channel] or not candidate_presence[row, channel]:
                continue
            included[channel] = True
            count += 1
            joined = np.empty(2 * _DTW_SAMPLES, dtype=np.float64)
            for index in range(_DTW_SAMPLES):
                joined[index] = query[channel, index]
                joined[_DTW_SAMPLES + index] = candidates[row, channel, index]
            ordered = np.sort(joined)
            # NumPy's default linear percentile interpolates from the lower
            # value; retain that operation order for scalar bit parity.
            q25 = ordered[31] + (ordered[32] - ordered[31]) * .75
            q75 = ordered[95] + (ordered[96] - ordered[95]) * .25
            scale = q75 - q25
            if scale < 1e-8:
                mean = 0.0
                for value in joined:
                    mean += value
                mean /= len(joined)
                squared = 0.0
                for value in joined:
                    delta = value - mean
                    squared += delta * delta
                scale = np.sqrt(squared / len(joined))
            scales[channel] = max(scale, 1e-6)
        if count == 0:
            output[row] = .55 * rigid_price[row]
            continue

        previous = np.full(_DTW_SAMPLES + 1, np.inf)
        previous_length = np.zeros(_DTW_SAMPLES + 1, dtype=np.int64)
        previous[0] = 0.0
        for query_index in range(1, _DTW_SAMPLES + 1):
            current = np.full(_DTW_SAMPLES + 1, np.inf)
            current_length = np.zeros(_DTW_SAMPLES + 1, dtype=np.int64)
            first = max(1, query_index - band)
            last = min(_DTW_SAMPLES, query_index + band)
            for candidate_index in range(first, last + 1):
                value = previous[candidate_index]
                length = previous_length[candidate_index]
                if current[candidate_index - 1] < value:
                    value = current[candidate_index - 1]
                    length = current_length[candidate_index - 1]
                if previous[candidate_index - 1] < value:
                    value = previous[candidate_index - 1]
                    length = previous_length[candidate_index - 1]
                squared = 0.0
                for channel in range(len(_DTW_CHANNELS)):
                    if included[channel]:
                        left = query[channel, query_index - 1] / scales[channel]
                        right = (
                            candidates[row, channel, candidate_index - 1]
                            / scales[channel]
                        )
                        delta = left - right
                        squared += delta * delta
                current[candidate_index] = value + np.sqrt(squared / count)
                current_length[candidate_index] = length + 1
            previous = current
            previous_length = current_length
        dtw = previous[_DTW_SAMPLES] / max(
            previous_length[_DTW_SAMPLES], 1,
        )
        output[row] = .55 * rigid_price[row] + .45 * dtw
    return output


def batch_exact_price_distances(
    query: Representation,
    candidates: tuple[Representation, ...] | list[Representation],
    rigid_price: np.ndarray,
    config: DistanceConfig | None = None,
) -> np.ndarray:
    """Return exact distance-v1 price components for a candidate batch."""
    config = config or DistanceConfig()
    rows = len(candidates)
    rigid = np.asarray(rigid_price, dtype=np.float64)
    if rigid.shape != (rows,) or not np.isfinite(rigid).all() \
            or np.any(rigid < 0) or not np.isfinite(config.dtw_band_fraction) \
            or config.dtw_band_fraction < 0:
        raise ValueError("exact price batch inputs differ")
    if not rows:
        return np.empty(0, dtype=np.float64)
    query_values = np.zeros(
        (len(_DTW_CHANNELS), _DTW_SAMPLES), dtype=np.float64,
    )
    query_presence = np.zeros(len(_DTW_CHANNELS), dtype=bool)
    candidate_values = np.zeros(
        (rows, len(_DTW_CHANNELS), _DTW_SAMPLES), dtype=np.float64,
    )
    candidate_presence = np.zeros((rows, len(_DTW_CHANNELS)), dtype=bool)
    for channel, name in enumerate(_DTW_CHANNELS):
        values = query.samples_64.get(name)
        if values is not None:
            array = np.asarray(values, dtype=np.float64)
            if array.shape != (_DTW_SAMPLES,) or not np.isfinite(array).all():
                raise ValueError(f"query exact price channel differs: {name}")
            query_values[channel] = array
            query_presence[channel] = True
        for row, candidate in enumerate(candidates):
            values = candidate.samples_64.get(name)
            if values is None:
                continue
            array = np.asarray(values, dtype=np.float64)
            if array.shape != (_DTW_SAMPLES,) or not np.isfinite(array).all():
                raise ValueError(f"candidate exact price channel differs: {name}")
            candidate_values[row, channel] = array
            candidate_presence[row, channel] = True
    return _batch_exact_price_distances_compiled(
        query_values, query_presence, candidate_values, candidate_presence,
        rigid, config.dtw_band_fraction,
    )


def batch_exact_rigid_price_distances(
    query: Representation,
    candidates: tuple[Representation, ...] | list[Representation],
) -> np.ndarray:
    """Return the exact pre-DTW rigid price distance for a candidate batch."""
    rows = len(candidates)
    if not rows:
        return np.empty(0, dtype=np.float64)
    distances: list[np.ndarray] = []
    included: list[np.ndarray] = []
    for name in GROUPS["price"]:
        query_values = query.samples_48.get(name)
        candidate_values = [candidate.samples_48.get(name) for candidate in candidates]
        present = np.asarray([values is not None for values in candidate_values])
        if query_values is None:
            distances.append(np.where(present, 2.0, 0.0))
            included.append(present)
            continue
        query_array = np.asarray(query_values, dtype=np.float64)
        if query_array.shape != (48,) or not np.isfinite(query_array).all():
            raise ValueError(f"query rigid price channel differs: {name}")
        matrix = np.stack([
            np.asarray(values, dtype=np.float64)
            if values is not None else np.zeros(48, dtype=np.float64)
            for values in candidate_values
        ])
        if matrix.shape != (rows, 48) or not np.isfinite(matrix).all():
            raise ValueError(f"candidate rigid price channel differs: {name}")
        joined = np.c_[matrix, np.broadcast_to(query_array, matrix.shape)]
        quartiles = np.percentile(joined, [75, 25], axis=1)
        scale = quartiles[0] - quartiles[1]
        scale = np.where(scale < 1e-8, np.std(joined, axis=1), scale)
        scale = np.maximum(scale, 1e-6)
        values = np.sqrt(np.mean(
            ((matrix - query_array) / scale[:, None]) ** 2, axis=1,
        ))
        values[~present] = 2.0
        distances.append(values)
        included.append(np.ones(rows, dtype=bool))
    count = np.sum(included, axis=0)
    return np.divide(
        np.sum(distances, axis=0), count,
        out=np.zeros(rows, dtype=np.float64), where=count > 0,
    )


def _rolling_windows(rows: np.ndarray, window: int) -> np.ndarray:
    padded = np.pad(
        np.asarray(rows, dtype=float), ((0, 0), (window - 1, 0)),
        constant_values=np.nan,
    )
    return np.lib.stride_tricks.sliding_window_view(padded, window, axis=1)


def _rolling_mean(rows: np.ndarray, window: int, minimum: int) -> np.ndarray:
    windows = _rolling_windows(rows, window)
    valid = np.isfinite(windows)
    count = valid.sum(axis=2)
    total = np.where(valid, windows, 0.0).sum(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = total / count
    result[count < minimum] = np.nan
    return result


def _rolling_extreme(
    rows: np.ndarray, window: int, minimum: int, *, maximum: bool,
) -> np.ndarray:
    windows = _rolling_windows(rows, window)
    count = np.isfinite(windows).sum(axis=2)
    with np.errstate(all="ignore"):
        result = (
            np.nanmax(windows, axis=2) if maximum
            else np.nanmin(windows, axis=2)
        )
    result[count < minimum] = np.nan
    return result


def _rolling_median(rows: np.ndarray, window: int, minimum: int) -> np.ndarray:
    windows = _rolling_windows(rows, window)
    valid = np.isfinite(windows)
    count = valid.sum(axis=2)
    ordered = np.sort(np.where(valid, windows, np.inf), axis=2)
    lower = np.take_along_axis(
        ordered, np.maximum((count - 1) // 2, 0)[..., None], axis=2,
    )[..., 0]
    upper = np.take_along_axis(
        ordered, np.maximum(count // 2, 0)[..., None], axis=2,
    )[..., 0]
    result = (lower + upper) / 2
    result[count < minimum] = np.nan
    return result


def _rolling_robust_z(rows: np.ndarray, window: int) -> np.ndarray:
    minimum = max(5, window // 2)
    median = _rolling_median(rows, window, minimum)
    mad = _rolling_median(np.abs(rows - median), window, minimum)
    denominator = 1.4826 * np.where(mad == 0, np.nan, mad)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (rows - median) / denominator


def _log_ratio(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.clip(left, EPS, None) / np.clip(right, EPS, None))


def _previous(rows: np.ndarray) -> np.ndarray:
    return np.c_[np.full(len(rows), np.nan), rows[:, :-1]]


def _benchmark_channel_rows(market: np.ndarray) -> dict[str, np.ndarray]:
    previous = _previous(market)
    market_return = _log_ratio(market, previous)
    anchor = np.full((len(market), 1), np.nan)
    for row, values in enumerate(market):
        valid = values[np.isfinite(values)]
        if len(valid):
            anchor[row, 0] = valid[0]
    market_path = _log_ratio(market, anchor)
    state = np.maximum.accumulate(np.where(np.isfinite(market), market, -np.inf), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = market / state - 1
    drawdown[~np.isfinite(market)] = np.nan
    return {
        "benchmark_close": market,
        "benchmark_return": market_return,
        "benchmark_path": market_path,
        "benchmark_drawdown": drawdown,
    }


def _benchmark_channels(
    frame: pd.DataFrame,
    benchmark: pd.DataFrame | None,
    lookback: int,
    stride: int,
) -> dict[str, np.ndarray]:
    raw = align_benchmark_close(frame, benchmark)
    market = np.lib.stride_tricks.sliding_window_view(raw, lookback)[::stride]
    return _benchmark_channel_rows(market)


def _exact_channels_from_rows(
    open_rows: np.ndarray,
    high_rows: np.ndarray,
    low_rows: np.ndarray,
    close_rows: np.ndarray,
    volume_rows: np.ndarray,
    benchmark_rows: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build exact channels from explicit episode-by-session input matrices."""
    previous_close = _previous(close_rows)
    true_range = np.fmax.reduce([
        high_rows - low_rows,
        np.abs(high_rows - previous_close),
        np.abs(low_rows - previous_close),
    ])
    atr = _rolling_mean(true_range, 14, 5)
    span = high_rows - low_rows
    span[span == 0] = np.nan
    body = close_rows - open_rows
    stock_return = _log_ratio(close_rows, previous_close)
    channels = {
        "close_path": _log_ratio(close_rows, close_rows[:, :1]),
        "return": stock_return,
        "overnight": _log_ratio(open_rows, previous_close),
        "intraday": _log_ratio(close_rows, open_rows),
        "range_pct": _log_ratio(high_rows, low_rows),
        "body_atr": body / np.where(atr == 0, np.nan, atr),
        "upper_wick_atr": (
            high_rows - np.maximum(open_rows, close_rows)
        ) / np.where(atr == 0, np.nan, atr),
        "lower_wick_atr": (
            np.minimum(open_rows, close_rows) - low_rows
        ) / np.where(atr == 0, np.nan, atr),
        "close_location": (close_rows - low_rows) / span,
        "atr_pct": atr / close_rows,
        "volume_robust_z": _rolling_robust_z(
            np.log(np.clip(volume_rows, EPS, None)), 20,
        ),
        "return_shock_z": _rolling_robust_z(stock_return, 63),
        "distance_high_63": close_rows / _rolling_extreme(
            close_rows, 63, 20, maximum=True,
        ) - 1,
        "distance_ma_20": close_rows / _rolling_mean(close_rows, 20, 10) - 1,
        "distance_ma_50": close_rows / _rolling_mean(close_rows, 50, 20) - 1,
    }
    high_20 = _rolling_extreme(high_rows, 20, 10, maximum=True)
    low_20 = _rolling_extreme(low_rows, 20, 10, maximum=False)
    high_63 = _rolling_extreme(high_rows, 63, 20, maximum=True)
    low_63 = _rolling_extreme(low_rows, 63, 20, maximum=False)
    range_20 = high_20 / low_20 - 1
    range_63 = high_63 / low_63 - 1
    channels["compression_ratio"] = range_20 / np.where(
        range_63 == 0, np.nan, range_63,
    )
    context = _benchmark_channel_rows(benchmark_rows)
    channels.update(context)
    relative_return = stock_return - context["benchmark_return"]
    relative_path = np.cumsum(
        np.where(np.isfinite(relative_return), relative_return, 0), axis=1,
    )
    relative_path[~np.isfinite(context["benchmark_return"])] = np.nan
    channels["relative_return"] = relative_return
    channels["relative_path"] = relative_path
    for name, rows in channels.items():
        channels[name] = np.where(np.isfinite(rows), rows, np.nan)
    return channels


def exact_channel_rows(
    frame: pd.DataFrame,
    benchmark: pd.DataFrame | None,
    *,
    lookback: int,
    stride: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if len(frame) < lookback:
        return np.empty(0, dtype=int), {}
    values = {
        name: pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
        for name in ("open", "high", "low", "close", "volume")
    }
    window = lambda array: np.lib.stride_tricks.sliding_window_view(
        array, lookback,
    )[::stride]
    open_rows, high_rows, low_rows = (
        window(values[name]) for name in ("open", "high", "low")
    )
    close_rows, volume_rows = window(values["close"]), window(values["volume"])
    benchmark_values = align_benchmark_close(frame, benchmark)
    benchmark_rows = window(benchmark_values)
    channels = _exact_channels_from_rows(
        open_rows, high_rows, low_rows, close_rows, volume_rows,
        benchmark_rows,
    )
    positions = np.arange(lookback - 1, len(frame), stride, dtype=int)
    return positions, channels


def exact_representations_at_positions(
    frame: pd.DataFrame,
    benchmark: pd.DataFrame | None,
    *,
    positions: np.ndarray,
    lookback: int,
) -> tuple[Representation, ...]:
    """Materialize only explicitly requested cutoff positions in one vector batch."""
    requested = np.asarray(positions, dtype=int)
    if requested.ndim != 1:
        raise ValueError("requested positions must be one-dimensional")
    if lookback < 2:
        raise ValueError("lookback must be at least two")
    if not len(requested):
        return ()
    if len(np.unique(requested)) != len(requested):
        raise ValueError("requested positions must be unique")
    if np.any(requested < lookback - 1) or np.any(requested >= len(frame)):
        raise ValueError("requested position cannot provide the complete lookback")
    values = {
        name: pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
        for name in ("open", "high", "low", "close", "volume")
    }
    offsets = np.arange(lookback, dtype=int)
    indices = requested[:, None] - lookback + 1 + offsets
    benchmark_values = align_benchmark_close(frame, benchmark)
    channels = _exact_channels_from_rows(
        values["open"][indices], values["high"][indices],
        values["low"][indices], values["close"][indices],
        values["volume"][indices], benchmark_values[indices],
    )
    return materialize_exact_representations(channels)


def _exact_price_channels_from_rows(
    open_rows: np.ndarray,
    high_rows: np.ndarray,
    low_rows: np.ndarray,
    close_rows: np.ndarray,
    volume_rows: np.ndarray,
    benchmark_rows: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build only channels consumed by distance-v1's exact price component."""
    previous_close = _previous(close_rows)
    true_range = np.fmax.reduce([
        high_rows - low_rows,
        np.abs(high_rows - previous_close),
        np.abs(low_rows - previous_close),
    ])
    atr = _rolling_mean(true_range, 14, 5)
    stock_return = _log_ratio(close_rows, previous_close)
    context = _benchmark_channel_rows(benchmark_rows)
    relative_return = stock_return - context["benchmark_return"]
    relative_path = np.cumsum(
        np.where(np.isfinite(relative_return), relative_return, 0), axis=1,
    )
    relative_path[~np.isfinite(context["benchmark_return"])] = np.nan
    channels = {
        "close_path": _log_ratio(close_rows, close_rows[:, :1]),
        "return": stock_return,
        "overnight": _log_ratio(open_rows, previous_close),
        "intraday": _log_ratio(close_rows, open_rows),
        "distance_high_63": close_rows / _rolling_extreme(
            close_rows, 63, 20, maximum=True,
        ) - 1,
        "distance_ma_20": close_rows / _rolling_mean(
            close_rows, 20, 10,
        ) - 1,
        "atr_pct": atr / close_rows,
        "volume_robust_z": _rolling_robust_z(
            np.log(np.clip(volume_rows, EPS, None)), 20,
        ),
        "relative_path": relative_path,
    }
    return {
        name: np.where(np.isfinite(rows), rows, np.nan)
        for name, rows in channels.items()
    }


def exact_price_representations_at_positions(
    frame: pd.DataFrame,
    benchmark: pd.DataFrame | None,
    *,
    positions: np.ndarray,
    lookback: int,
) -> tuple[Representation, ...]:
    """Materialize only the nine sampled channels required by exact price."""
    requested = np.asarray(positions, dtype=int)
    if requested.ndim != 1 or lookback < 2 or not len(requested) \
            or len(np.unique(requested)) != len(requested) \
            or np.any(requested < lookback - 1) or np.any(requested >= len(frame)):
        raise ValueError("requested exact-price positions differ")
    values = {
        name: pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
        for name in ("open", "high", "low", "close", "volume")
    }
    indices = (
        requested[:, None] - lookback + 1 + np.arange(lookback, dtype=int)
    )
    benchmark_values = align_benchmark_close(frame, benchmark)
    channels = _exact_price_channels_from_rows(
        values["open"][indices], values["high"][indices],
        values["low"][indices], values["close"][indices],
        values["volume"][indices], benchmark_values[indices],
    )
    names_48 = GROUPS["price"]
    names_64 = _DTW_CHANNELS
    sampled_48 = {
        name: _resample_rows(channels[name], 48, optional=True)
        for name in names_48
    }
    sampled_64 = {
        name: _resample_rows(channels[name], 64, optional=True)
        for name in names_64
    }
    empty = np.empty(0, dtype=np.float64)
    empty_channels = pd.DataFrame()
    return tuple(
        Representation(
            empty_channels, empty,
            {
                name: samples[row].copy() if present[row] else None
                for name, (samples, present) in sampled_48.items()
            },
            {
                name: samples[row].copy() if present[row] else None
                for name, (samples, present) in sampled_64.items()
            },
            empty, empty,
        )
        for row in range(len(requested))
    )


def _resample_rows(
    rows: np.ndarray,
    count: int,
    *,
    optional: bool,
) -> tuple[np.ndarray, np.ndarray]:
    sampled = np.zeros((len(rows), count), dtype=float)
    present = np.zeros(len(rows), dtype=bool)
    source_positions = np.arange(rows.shape[1])
    target_positions = np.linspace(0, rows.shape[1] - 1, count)
    minimum = max(3, rows.shape[1] // 5)
    for row, values in enumerate(rows):
        valid = np.isfinite(values)
        if optional and valid.sum() < minimum:
            continue
        if not valid.any():
            continue
        sampled[row] = np.interp(
            target_positions, source_positions[valid], values[valid],
        )
        present[row] = True
    return sampled, present


def _stage_rows(channels: dict[str, np.ndarray], stages: int = 12) -> np.ndarray:
    close_path = channels["close_path"]
    boundaries = np.linspace(0, close_path.shape[1], stages + 1, dtype=int)
    parts: list[np.ndarray] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        end = max(end, start + 1)
        path = close_path[:, start:end]
        net = np.zeros(len(path))
        for row, values in enumerate(path):
            valid = values[np.isfinite(values)]
            if len(valid) > 1:
                net[row] = valid[-1] - valid[0]
        returns = channels["return"][:, start:end]
        valid_count = np.isfinite(returns).sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            volatility = np.nanstd(returns, axis=1, ddof=1)
        volume_values = channels["volume_robust_z"][:, start:end]
        volume_count = np.isfinite(volume_values).sum(axis=1)
        volume = np.full(len(volume_values), np.nan)
        np.divide(
            np.nansum(volume_values, axis=1), volume_count,
            out=volume, where=volume_count > 0,
        )
        volatility[valid_count < 2] = np.nan
        relative = np.nansum(channels["relative_return"][:, start:end], axis=1)
        parts.extend([
            net[:, None] / .08,
            volatility[:, None] / .02,
            volume[:, None],
            relative[:, None] / .08,
        ])
    return np.nan_to_num(
        np.concatenate(parts, axis=1), nan=0.0, posinf=0.0, neginf=0.0,
    )


def _structural_rows_reference(close_path: np.ndarray) -> np.ndarray:
    close = np.exp(np.where(np.isfinite(close_path), close_path, 0.0))
    output: list[np.ndarray] = []
    for threshold in (.03, .06, .12):
        mode = np.ones(len(close), dtype=np.int8)
        extreme_index = np.zeros(len(close), dtype=int)
        extreme = close[:, 0].copy()
        count = np.zeros(len(close), dtype=int)
        amplitude_sum = np.zeros(len(close))
        duration_sum = np.zeros(len(close))
        for index in range(1, close.shape[1]):
            price = close[:, index]
            tracking_high = mode >= 0
            new_high = tracking_high & (price >= extreme)
            extreme[new_high] = price[new_high]
            extreme_index[new_high] = index
            down = tracking_high & ~new_high & (price <= extreme * (1 - threshold))
            count[down] += 1
            amplitude_sum[down] += np.abs(price[down] / extreme[down] - 1)
            duration_sum[down] += index - extreme_index[down]
            mode[down] = -1
            extreme[down] = price[down]
            extreme_index[down] = index

            tracking_low = mode <= 0
            new_low = tracking_low & (price <= extreme)
            extreme[new_low] = price[new_low]
            extreme_index[new_low] = index
            up = tracking_low & ~new_low & (price >= extreme * (1 + threshold))
            count[up] += 1
            amplitude_sum[up] += np.abs(price[up] / extreme[up] - 1)
            duration_sum[up] += index - extreme_index[up]
            mode[up] = 1
            extreme[up] = price[up]
            extreme_index[up] = index
        nonzero = count > 0
        mean_amplitude = np.zeros(len(close))
        mean_duration = np.zeros(len(close))
        mean_amplitude[nonzero] = amplitude_sum[nonzero] / count[nonzero]
        mean_duration[nonzero] = duration_sum[nonzero] / count[nonzero]
        output.extend([
            count[:, None] / max(close.shape[1], 1),
            mean_amplitude[:, None],
            mean_duration[:, None] / max(close.shape[1], 1),
        ])
    return np.concatenate(output, axis=1)


@njit(cache=True, nogil=True)
def _structural_rows_compiled(close: np.ndarray) -> np.ndarray:
    rows, width = close.shape
    output = np.empty((rows, 9), dtype=np.float64)
    thresholds = (.03, .06, .12)
    for row in range(rows):
        column = 0
        for threshold in thresholds:
            mode = 1
            extreme = close[row, 0]
            extreme_index = 0
            count = 0
            amplitude_sum = 0.0
            duration_sum = 0.0
            for index in range(1, width):
                price = close[row, index]
                if mode >= 0:
                    if price >= extreme:
                        extreme = price
                        extreme_index = index
                    elif price <= extreme * (1.0 - threshold):
                        count += 1
                        amplitude_sum += abs(price / extreme - 1.0)
                        duration_sum += index - extreme_index
                        mode = -1
                        extreme = price
                        extreme_index = index
                else:
                    if price <= extreme:
                        extreme = price
                        extreme_index = index
                    elif price >= extreme * (1.0 + threshold):
                        count += 1
                        amplitude_sum += abs(price / extreme - 1.0)
                        duration_sum += index - extreme_index
                        mode = 1
                        extreme = price
                        extreme_index = index
            output[row, column] = count / max(width, 1)
            output[row, column + 1] = (
                amplitude_sum / count if count else 0.0
            )
            output[row, column + 2] = (
                (duration_sum / count if count else 0.0) / max(width, 1)
            )
            column += 3
    return output


def _structural_rows(close_path: np.ndarray) -> np.ndarray:
    close = np.exp(np.where(np.isfinite(close_path), close_path, 0.0))
    return _structural_rows_compiled(close)


def materialize_exact_representations(
    channels: dict[str, np.ndarray],
) -> tuple[Representation, ...]:
    """Materialize distance-v1 fields from one canonical matrix of channels.

    Scalar queries and sliding candidates both call this function.  Channel
    matrices are shaped ``(episodes, sessions)`` and retain NaN as missingness.
    """
    if not channels:
        return ()
    row_counts = {len(rows) for rows in channels.values()}
    widths = {rows.shape[1] for rows in channels.values()}
    if len(row_counts) != 1 or len(widths) != 1:
        raise ValueError("exact channel matrices disagree in shape")
    names_48 = tuple(sorted(name for names in GROUPS.values() for name in names))
    names_64 = ("atr_pct", "close_path", "relative_path", "volume_robust_z")
    sampled_48: dict[str, tuple[np.ndarray, np.ndarray]] = {
        name: _resample_rows(channels[name], 48, optional=True) for name in names_48
    }
    sampled_64: dict[str, tuple[np.ndarray, np.ndarray]] = {
        name: _resample_rows(channels[name], 64, optional=True) for name in names_64
    }
    coarse_parts = []
    for name, count in COARSE_LAYOUT.items():
        sampled, _ = _resample_rows(channels[name], count, optional=False)
        coarse_parts.append(sampled.astype(np.float32))
    coarse = np.concatenate(coarse_parts, axis=1).astype(np.float32)
    stage = _stage_rows(channels)
    structural = _structural_rows(channels["close_path"])
    representations = []
    for row in range(next(iter(row_counts))):
        representations.append(Representation(
            pd.DataFrame(), coarse[row],
            {
                name: values[row].copy() if present[row] else None
                for name, (values, present) in sampled_48.items()
            },
            {
                name: values[row].copy() if present[row] else None
                for name, (values, present) in sampled_64.items()
            },
            stage[row], structural[row],
        ))
    return tuple(representations)


def sliding_exact_representations(
    bars: pd.DataFrame,
    benchmark: pd.DataFrame | None,
    *,
    lookback: int,
    stride: int = 5,
    batch_size: int | None = None,
) -> ExactBatch:
    if lookback < 2:
        raise ValueError("lookback must be at least two")
    if stride < 1:
        raise ValueError("stride must be positive")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive")
    started = perf_counter()
    frame = bars.reset_index(drop=True)
    total_windows = max((len(frame) - lookback) // stride + 1, 0)
    if batch_size is not None and total_windows > batch_size:
        all_positions: list[np.ndarray] = []
        all_representations: list[Representation] = []
        for first in range(0, total_windows, batch_size):
            count = min(batch_size, total_windows - first)
            segment_start = first * stride
            segment_end = segment_start + lookback + (count - 1) * stride
            segment = frame.iloc[segment_start:segment_end].reset_index(drop=True)
            chunk = sliding_exact_representations(
                segment, benchmark, lookback=lookback, stride=stride,
            )
            all_positions.append(chunk.positions + segment_start)
            all_representations.extend(chunk.representations)
        return ExactBatch(
            np.concatenate(all_positions), tuple(all_representations),
            perf_counter() - started,
        )
    positions, channels = exact_channel_rows(
        frame, benchmark, lookback=lookback, stride=stride,
    )
    if not len(positions):
        return ExactBatch(positions, (), perf_counter() - started)
    representations = materialize_exact_representations(channels)
    return ExactBatch(
        positions, representations, perf_counter() - started,
    )


def batch_representation_lower_bounds(
    query: Representation,
    candidates: tuple[Representation, ...] | list[Representation],
    config: DistanceConfig | None = None,
) -> LowerBoundBatch:
    config = config or DistanceConfig()
    if config.samples_per_channel != 48:
        raise ValueError("batch lower bounds require the exact 48-sample contract")
    if not candidates:
        empty = np.empty(0, dtype=float)
        return LowerBoundBatch(empty, {}, empty)
    coarse = np.stack([candidate.coarse for candidate in candidates]).astype(
        np.float64, copy=False,
    )
    query_coarse = np.asarray(query.coarse, dtype=np.float64)
    joined = np.c_[coarse, np.broadcast_to(query_coarse, coarse.shape)]
    coarse_scale = np.maximum(np.std(joined, axis=1), 1e-6)
    components: dict[str, np.ndarray] = {
        "coarse": np.sqrt(np.mean(
            ((coarse - query_coarse) / coarse_scale[:, None]) ** 2,
            axis=1,
        )),
        "stage": np.sqrt(np.mean(
            (np.stack([candidate.stage for candidate in candidates]) - query.stage) ** 2,
            axis=1,
        )),
    }
    for group, names in GROUPS.items():
        channel_distances = []
        channel_included = []
        for name in names:
            query_values = query.samples_48.get(name)
            candidate_values = [candidate.samples_48.get(name) for candidate in candidates]
            present = np.asarray([values is not None for values in candidate_values])
            if query_values is None:
                distances = np.where(present, 2.0, 0.0)
                channel_distances.append(distances)
                channel_included.append(present)
                continue
            matrix = np.stack([
                values if values is not None else np.zeros(48)
                for values in candidate_values
            ])
            joined_values = np.c_[
                matrix, np.broadcast_to(query_values, matrix.shape),
            ]
            quartiles = np.percentile(joined_values, [75, 25], axis=1)
            scale = quartiles[0] - quartiles[1]
            scale = np.where(scale < 1e-8, np.std(joined_values, axis=1), scale)
            scale = np.maximum(scale, 1e-6)
            distances = np.sqrt(np.mean(
                ((matrix - query_values) / scale[:, None]) ** 2,
                axis=1,
            ))
            distances[~present] = 2.0
            channel_distances.append(distances)
            channel_included.append(np.ones(len(candidates), dtype=bool))
        included = np.sum(channel_included, axis=0)
        components[group] = np.divide(
            np.sum(channel_distances, axis=0), included,
            out=np.zeros(len(candidates), dtype=float), where=included > 0,
        )
    rigid_price = components["price"].copy()
    components["price"] = .55 * rigid_price
    components["structural"] = np.sqrt(np.mean(
        (
            np.stack([candidate.structural for candidate in candidates])
            - query.structural
        ) ** 2,
        axis=1,
    ))
    totals = sum(
        config.weights.get(name, 0.0) * values.astype(float)
        for name, values in components.items()
    )
    return LowerBoundBatch(np.asarray(totals), components, rigid_price)
