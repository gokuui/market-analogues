from __future__ import annotations

import numpy as np
import pandas as pd

from .context import align_benchmark
from .types import Episode


VIEW_NAMES = (
    "coarse", "price_shape", "stage", "candle_volatility", "volume_shock",
    "market_context", "structural",
)
CANDIDATE_VIEW_VERSION = "cheap-v2"


def _values(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return np.zeros(len(values), dtype=float)
    positions = np.arange(len(values))
    return np.interp(positions, positions[valid], values[valid])


def _row_correlation(rows: np.ndarray, query: np.ndarray) -> np.ndarray:
    left = rows - rows.mean(axis=1, keepdims=True)
    right = query - query.mean()
    denominator = np.sqrt(np.sum(left * left, axis=1)) * np.sqrt(np.sum(right * right))
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = (left @ right) / denominator
    return 1 - np.nan_to_num(correlation, nan=-1.0, posinf=-1.0, neginf=-1.0)


def _sampled_windows(values: np.ndarray, lookback: int, stride: int, samples: int) -> np.ndarray:
    windows = np.lib.stride_tricks.sliding_window_view(values, lookback)[::stride]
    offsets = np.rint(np.linspace(0, lookback - 1, min(samples, lookback))).astype(int)
    return windows[:, offsets]


def _sample(values: np.ndarray, samples: int) -> np.ndarray:
    offsets = np.rint(np.linspace(0, len(values) - 1, min(samples, len(values)))).astype(int)
    return values[offsets]


def _path_distance(rows: np.ndarray, query: np.ndarray, shape_weight: float = .65) -> np.ndarray:
    rows = rows - rows[:, :1]
    query = query - query[0]
    shape = _row_correlation(rows, query)
    magnitude = np.sqrt(np.mean((rows - query) ** 2, axis=1))
    return shape_weight * shape + (1 - shape_weight) * magnitude


def _robust_rows(rows: np.ndarray) -> np.ndarray:
    center = np.median(rows, axis=1, keepdims=True)
    scale = np.percentile(rows, 75, axis=1, keepdims=True) - np.percentile(
        rows, 25, axis=1, keepdims=True,
    )
    fallback = np.std(rows, axis=1, keepdims=True)
    scale = np.where(scale > 1e-8, scale, fallback)
    return (rows - center) / np.maximum(scale, 1e-6)


def _stage_rows(
    close_rows: np.ndarray,
    volume_rows: np.ndarray,
    market_rows: np.ndarray | None,
    stages: int = 12,
) -> np.ndarray:
    paths = close_rows - close_rows[:, :1]
    returns = np.diff(close_rows, axis=1, prepend=close_rows[:, :1])
    volume_z = _robust_rows(volume_rows)
    relative = returns if market_rows is None else (
        returns - np.diff(market_rows, axis=1, prepend=market_rows[:, :1])
    )
    boundaries = np.linspace(0, close_rows.shape[1], stages + 1, dtype=int)
    parts: list[np.ndarray] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        end = max(end, start + 1)
        block_path = paths[:, start:end]
        parts.extend([
            (block_path[:, -1] - block_path[:, 0])[:, None] / .08,
            np.std(returns[:, start:end], axis=1, ddof=1)[:, None] / .02,
            np.mean(volume_z[:, start:end], axis=1)[:, None],
            np.sum(relative[:, start:end], axis=1)[:, None] / .08,
        ])
    return np.nan_to_num(np.concatenate(parts, axis=1), nan=0.0, posinf=0.0, neginf=0.0)


def sliding_view_distances(
    query_bars: pd.DataFrame,
    candidate_bars: pd.DataFrame,
    *,
    stride: int = 5,
    query_benchmark: pd.DataFrame | None = None,
    candidate_benchmark: pd.DataFrame | None = None,
) -> dict[str, np.ndarray]:
    """Compute independent, vectorized candidate views for every sliding window."""
    lookback = len(query_bars)
    if lookback < 2 or len(candidate_bars) < lookback:
        return {name: np.empty(0, dtype=float) for name in VIEW_NAMES}

    query_close = np.log(np.clip(_values(query_bars.close), 1e-12, None))
    candidate_close = np.log(np.clip(_values(candidate_bars.close), 1e-12, None))
    close_windows = np.lib.stride_tricks.sliding_window_view(
        candidate_close, lookback,
    )[::stride]
    views: dict[str, np.ndarray] = {
        "price_shape": _path_distance(close_windows, query_close),
    }

    stage_rows = _sampled_windows(candidate_close, lookback, stride, 16)

    query_range = np.log(
        np.clip(_values(query_bars.high), 1e-12, None)
        / np.clip(_values(query_bars.low), 1e-12, None)
    )
    candidate_range = np.log(
        np.clip(_values(candidate_bars.high), 1e-12, None)
        / np.clip(_values(candidate_bars.low), 1e-12, None)
    )
    query_body = np.log(
        np.clip(_values(query_bars.close), 1e-12, None)
        / np.clip(_values(query_bars.open), 1e-12, None)
    )
    candidate_body = np.log(
        np.clip(_values(candidate_bars.close), 1e-12, None)
        / np.clip(_values(candidate_bars.open), 1e-12, None)
    )
    candle_rows = np.c_[
        _sampled_windows(candidate_range, lookback, stride, 16) / .02,
        _sampled_windows(candidate_body, lookback, stride, 16) / .02,
    ]
    candle_query = np.r_[_sample(query_range, 16) / .02, _sample(query_body, 16) / .02]
    views["candle_volatility"] = np.sqrt(np.mean((candle_rows - candle_query) ** 2, axis=1))

    query_volume = np.log(np.clip(_values(query_bars.volume), 1e-12, None))
    candidate_volume = np.log(np.clip(_values(candidate_bars.volume), 1e-12, None))
    full_volume_rows = np.lib.stride_tricks.sliding_window_view(
        candidate_volume, lookback,
    )[::stride]
    volume_rows = _robust_rows(_sampled_windows(candidate_volume, lookback, stride, 16))
    volume_query = _robust_rows(_sample(query_volume, 16)[None, :])[0]
    query_shock = np.abs(np.diff(query_close, prepend=query_close[0]))
    candidate_shock = np.abs(np.diff(candidate_close, prepend=candidate_close[0]))
    shock_rows = _robust_rows(_sampled_windows(candidate_shock, lookback, stride, 16))
    shock_query = _robust_rows(_sample(query_shock, 16)[None, :])[0]
    volume_shock_rows = np.c_[volume_rows, shock_rows]
    views["volume_shock"] = np.sqrt(np.mean(
        (volume_shock_rows - np.r_[volume_query, shock_query]) ** 2, axis=1,
    ))

    close_paths = stage_rows - stage_rows[:, :1]
    query_path = _sample(query_close, 16)
    query_path -= query_path[0]
    drawdowns = close_paths - np.maximum.accumulate(close_paths, axis=1)
    query_drawdown = query_path - np.maximum.accumulate(query_path)
    direction = np.sign(np.diff(close_paths, axis=1))
    query_direction = np.sign(np.diff(query_path))
    views["structural"] = (
        .6 * np.sqrt(np.mean((drawdowns - query_drawdown) ** 2, axis=1))
        + .4 * np.mean(direction != query_direction, axis=1)
    )

    query_context = align_benchmark(query_bars.reset_index(drop=True), query_benchmark)
    candidate_context = align_benchmark(candidate_bars.reset_index(drop=True), candidate_benchmark)
    query_market = _values(query_context.benchmark_close)
    candidate_market = _values(candidate_context.benchmark_close)
    if (
        query_context.benchmark_close.notna().sum() >= max(3, lookback // 5)
        and candidate_context.benchmark_close.notna().sum() >= lookback
    ):
        query_market = np.log(np.clip(query_market, 1e-12, None))
        candidate_market = np.log(np.clip(candidate_market, 1e-12, None))
        market_rows = _sampled_windows(candidate_market, lookback, stride, 16)
        market_query = _sample(query_market, 16)
        relative_rows = stage_rows - market_rows
        relative_query = _sample(query_close, 16) - market_query
        views["market_context"] = .5 * _path_distance(market_rows, market_query) + .5 * _path_distance(
            relative_rows, relative_query,
        )
        full_market_rows: np.ndarray | None = np.lib.stride_tricks.sliding_window_view(
            candidate_market, lookback,
        )[::stride]
        query_market_full: np.ndarray | None = query_market
    else:
        # Missing context is neutral rather than an artificial best match.
        views["market_context"] = np.full(len(close_windows), 1.0)
        full_market_rows = None
        query_market_full = None

    candidate_stage = _stage_rows(close_windows, full_volume_rows, full_market_rows)
    query_stage = _stage_rows(
        query_close[None, :], query_volume[None, :],
        query_market_full[None, :] if query_market_full is not None else None,
    )[0]
    views["stage"] = np.sqrt(np.mean((candidate_stage - query_stage) ** 2, axis=1))

    close_64 = _sampled_windows(candidate_close, lookback, stride, 64)
    query_close_64 = _sample(query_close, 64)
    close_64 -= close_64[:, :1]
    query_close_64 -= query_close_64[0]
    range_16 = _sampled_windows(candidate_range, lookback, stride, 16) / .02
    query_range_16 = _sample(query_range, 16) / .02
    volume_16 = _robust_rows(_sampled_windows(candidate_volume, lookback, stride, 16))
    query_volume_16 = _robust_rows(_sample(query_volume, 16)[None, :])[0]
    if full_market_rows is not None and query_market_full is not None:
        market_16 = _sampled_windows(candidate_market, lookback, stride, 16)
        query_market_16 = _sample(query_market_full, 16)
        market_16 -= market_16[:, :1]
        query_market_16 -= query_market_16[0]
        relative_offsets = np.rint(
            np.linspace(0, close_64.shape[1] - 1, market_16.shape[1]),
        ).astype(int)
        relative_16 = close_64[:, relative_offsets] - market_16
        query_relative_16 = query_close_64[relative_offsets] - query_market_16
    else:
        market_16 = np.zeros((len(close_64), 16))
        query_market_16 = np.zeros(16)
        relative_16 = np.zeros((len(close_64), 16))
        query_relative_16 = np.zeros(16)
    coarse_rows = np.c_[close_64, range_16, volume_16, relative_16, market_16]
    coarse_query = np.r_[
        query_close_64, query_range_16, query_volume_16,
        query_relative_16, query_market_16,
    ]
    joined_mean = (coarse_rows.sum(axis=1) + coarse_query.sum()) / (
        coarse_rows.shape[1] + len(coarse_query)
    )
    joined_variance = (
        np.sum((coarse_rows - joined_mean[:, None]) ** 2, axis=1)
        + np.sum((coarse_query[None, :] - joined_mean[:, None]) ** 2, axis=1)
    ) / (coarse_rows.shape[1] + len(coarse_query))
    coarse_scale = np.maximum(np.sqrt(joined_variance), 1e-6)
    views["coarse"] = np.sqrt(np.mean((coarse_rows - coarse_query) ** 2, axis=1)) / coarse_scale
    return views


def episode_view_distances(query: Episode, candidate: Episode) -> dict[str, float]:
    """Return the same cheap views used by the universe scanner for one episode."""
    if len(query.bars) != len(candidate.bars):
        raise ValueError("query and candidate lookbacks must match")
    values = sliding_view_distances(
        query.bars, candidate.bars, stride=1,
        query_benchmark=query.benchmark, candidate_benchmark=candidate.benchmark,
    )
    if any(len(distance) != 1 for distance in values.values()):
        raise ValueError("an episode pair must produce exactly one distance per view")
    return {name: float(values[name][0]) for name in VIEW_NAMES}
