from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .candidate_views import VIEW_NAMES
from .context import align_benchmark
from .types import Episode


VIEW_SIGNATURE_VERSION = "view-signature-v1"

LAYOUT: dict[str, tuple[int, int]] = {
    "close_path": (0, 64),
    "range": (64, 80),
    "body": (80, 96),
    "volume": (96, 112),
    "shock": (112, 128),
    "market_path": (128, 144),
    "relative_path": (144, 160),
    "stage": (160, 208),
    "drawdown": (208, 224),
    "direction": (224, 239),
    "context_available": (239, 240),
}
SIGNATURE_DIMENSIONS = 240


def _values(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return np.zeros(len(values), dtype=float)
    positions = np.arange(len(values))
    return np.interp(positions, positions[valid], values[valid])


def _sample(values: np.ndarray, count: int) -> np.ndarray:
    if not len(values):
        return np.zeros(count, dtype=float)
    positions = np.linspace(0, len(values) - 1, count)
    return np.interp(positions, np.arange(len(values)), values)


def _robust(values: np.ndarray) -> np.ndarray:
    center = float(np.median(values))
    scale = float(np.percentile(values, 75) - np.percentile(values, 25))
    if scale <= 1e-8:
        scale = float(np.std(values))
    return (values - center) / max(scale, 1e-6)


def _sample_rows(rows: np.ndarray, count: int) -> np.ndarray:
    positions = np.linspace(0, rows.shape[1] - 1, count)
    left = np.floor(positions).astype(int)
    right = np.ceil(positions).astype(int)
    weight = positions - left
    return rows[:, left] * (1 - weight) + rows[:, right] * weight


def _robust_rows(rows: np.ndarray) -> np.ndarray:
    center = np.median(rows, axis=1, keepdims=True)
    scale = np.percentile(rows, 75, axis=1, keepdims=True) - np.percentile(
        rows, 25, axis=1, keepdims=True,
    )
    fallback = np.std(rows, axis=1, keepdims=True)
    scale = np.where(scale > 1e-8, scale, fallback)
    return (rows - center) / np.maximum(scale, 1e-6)


def _stage_signature(
    close: np.ndarray,
    volume: np.ndarray,
    market: np.ndarray | None,
    stages: int = 12,
) -> np.ndarray:
    path = close - close[0]
    returns = np.diff(close, prepend=close[0])
    volume_z = _robust(volume)
    relative = returns if market is None else returns - np.diff(market, prepend=market[0])
    boundaries = np.linspace(0, len(close), stages + 1, dtype=int)
    values: list[float] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        end = max(end, start + 1)
        block = path[start:end]
        values.extend([
            float(block[-1] - block[0]) / .08,
            float(np.std(returns[start:end], ddof=1)) / .02 if end - start > 1 else 0.0,
            float(np.mean(volume_z[start:end])),
            float(np.sum(relative[start:end])) / .08,
        ])
    return np.nan_to_num(np.asarray(values), nan=0.0, posinf=0.0, neginf=0.0)


def _stage_matrix(
    close: np.ndarray,
    volume: np.ndarray,
    market: np.ndarray | None,
    stages: int = 12,
) -> np.ndarray:
    paths = close - close[:, :1]
    returns = np.diff(close, axis=1, prepend=close[:, :1])
    volume_z = _robust_rows(volume)
    relative = returns if market is None else (
        returns - np.diff(market, axis=1, prepend=market[:, :1])
    )
    boundaries = np.linspace(0, close.shape[1], stages + 1, dtype=int)
    parts: list[np.ndarray] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        end = max(end, start + 1)
        block = paths[:, start:end]
        parts.extend([
            (block[:, -1] - block[:, 0])[:, None] / .08,
            np.std(returns[:, start:end], axis=1, ddof=1)[:, None] / .02,
            np.mean(volume_z[:, start:end], axis=1)[:, None],
            np.sum(relative[:, start:end], axis=1)[:, None] / .08,
        ])
    return np.nan_to_num(
        np.concatenate(parts, axis=1), nan=0.0, posinf=0.0, neginf=0.0,
    )


@dataclass(frozen=True)
class ViewSignature:
    vector: np.ndarray

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector, dtype=np.float32)
        if vector.shape != (SIGNATURE_DIMENSIONS,):
            raise ValueError(
                f"view signature must have {SIGNATURE_DIMENSIONS} values; got {vector.shape}"
            )
        if not np.isfinite(vector).all():
            raise ValueError("view signature contains non-finite values")
        object.__setattr__(self, "vector", vector)

    def part(self, name: str) -> np.ndarray:
        start, end = LAYOUT[name]
        return self.vector[start:end]


def episode_view_signature(episode: Episode) -> ViewSignature:
    bars = episode.bars.reset_index(drop=True)
    close = np.log(np.clip(_values(bars.close), 1e-12, None))
    open_ = np.log(np.clip(_values(bars.open), 1e-12, None))
    high = np.log(np.clip(_values(bars.high), 1e-12, None))
    low = np.log(np.clip(_values(bars.low), 1e-12, None))
    volume = np.log(np.clip(_values(bars.volume), 1e-12, None))

    close_path = _sample(close - close[0], 64)
    range_ = _sample(high - low, 16) / .02
    body = _sample(close - open_, 16) / .02
    volume_sample = _robust(_sample(volume, 16))
    shock = np.abs(np.diff(close, prepend=close[0]))
    shock_sample = _robust(_sample(shock, 16))

    context = align_benchmark(bars, episode.benchmark)
    context_count = int(context.benchmark_close.notna().sum())
    context_available = context_count >= max(3, len(bars) // 5)
    if context_available:
        market = np.log(np.clip(_values(context.benchmark_close), 1e-12, None))
        market_path = _sample(market - market[0], 16)
        relative_path = _sample((close - close[0]) - (market - market[0]), 16)
        stage_market: np.ndarray | None = market
    else:
        market_path = np.zeros(16)
        relative_path = np.zeros(16)
        stage_market = None

    stage = _stage_signature(close, volume, stage_market)
    structural_path = _sample(close - close[0], 16)
    drawdown = structural_path - np.maximum.accumulate(structural_path)
    direction = np.sign(np.diff(structural_path))
    vector = np.r_[
        close_path, range_, body, volume_sample, shock_sample,
        market_path, relative_path, stage, drawdown, direction,
        float(context_available),
    ]
    return ViewSignature(vector)


def sliding_episode_signatures(
    bars: pd.DataFrame,
    benchmark: pd.DataFrame | None,
    *,
    lookback: int,
    stride: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorize every fixed-size view signature for one instrument shard."""
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    if stride < 1:
        raise ValueError("stride must be positive")
    if len(bars) < lookback:
        return np.empty(0, dtype=int), np.empty((0, SIGNATURE_DIMENSIONS), dtype=np.float32)
    frame = bars.reset_index(drop=True)
    close = np.log(np.clip(_values(frame.close), 1e-12, None))
    open_ = np.log(np.clip(_values(frame.open), 1e-12, None))
    high = np.log(np.clip(_values(frame.high), 1e-12, None))
    low = np.log(np.clip(_values(frame.low), 1e-12, None))
    volume = np.log(np.clip(_values(frame.volume), 1e-12, None))
    window = lambda values: np.lib.stride_tricks.sliding_window_view(values, lookback)[::stride]
    close_rows, open_rows = window(close), window(open_)
    high_rows, low_rows, volume_rows = window(high), window(low), window(volume)
    positions = np.arange(lookback - 1, len(frame), stride, dtype=int)

    close_path = _sample_rows(close_rows - close_rows[:, :1], 64)
    range_ = _sample_rows(high_rows - low_rows, 16) / .02
    body = _sample_rows(close_rows - open_rows, 16) / .02
    volume_sample = _robust_rows(_sample_rows(volume_rows, 16))
    shock_rows = np.abs(np.diff(close_rows, axis=1, prepend=close_rows[:, :1]))
    shock_sample = _robust_rows(_sample_rows(shock_rows, 16))

    context = align_benchmark(frame, benchmark)
    raw_market = pd.to_numeric(context.benchmark_close, errors="coerce").to_numpy(dtype=float)
    valid_rows = window(np.isfinite(raw_market).astype(float)).sum(axis=1)
    context_available = valid_rows >= max(3, lookback // 5)
    if np.isfinite(raw_market).any():
        market = np.log(np.clip(_values(context.benchmark_close), 1e-12, None))
        market_rows = window(market)
        market_path = _sample_rows(market_rows - market_rows[:, :1], 16)
        relative_path = _sample_rows(
            (close_rows - close_rows[:, :1]) - (market_rows - market_rows[:, :1]), 16,
        )
        stage_with_market = _stage_matrix(close_rows, volume_rows, market_rows)
        stage_without_market = _stage_matrix(close_rows, volume_rows, None)
        stage = np.where(context_available[:, None], stage_with_market, stage_without_market)
    else:
        market_path = np.zeros((len(close_rows), 16))
        relative_path = np.zeros((len(close_rows), 16))
        stage = _stage_matrix(close_rows, volume_rows, None)
    market_path[~context_available] = 0.0
    relative_path[~context_available] = 0.0

    structural_path = _sample_rows(close_rows - close_rows[:, :1], 16)
    drawdown = structural_path - np.maximum.accumulate(structural_path, axis=1)
    direction = np.sign(np.diff(structural_path, axis=1))
    matrix = np.c_[
        close_path, range_, body, volume_sample, shock_sample,
        market_path, relative_path, stage, drawdown, direction,
        context_available.astype(float),
    ].astype(np.float32)
    if matrix.shape != (len(positions), SIGNATURE_DIMENSIONS):
        raise AssertionError(f"unexpected sliding signature shape {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("sliding signatures contain non-finite values")
    return positions, matrix


def _parts(matrix: np.ndarray, name: str) -> np.ndarray:
    start, end = LAYOUT[name]
    return matrix[:, start:end]


def _path_distance(rows: np.ndarray, query: np.ndarray, shape_weight: float = .65) -> np.ndarray:
    left = rows - rows.mean(axis=1, keepdims=True)
    right = query - query.mean()
    denominator = np.sqrt(np.sum(left * left, axis=1)) * np.sqrt(np.sum(right * right))
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = (left @ right) / denominator
    shape = 1 - np.nan_to_num(correlation, nan=-1.0, posinf=-1.0, neginf=-1.0)
    magnitude = np.sqrt(np.mean((rows - query) ** 2, axis=1))
    return shape_weight * shape + (1 - shape_weight) * magnitude


def signature_view_distances(
    query: ViewSignature,
    candidates: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compare one query with a matrix of persisted query-independent signatures."""
    matrix = np.asarray(candidates, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.ndim != 2 or matrix.shape[1] != SIGNATURE_DIMENSIONS:
        raise ValueError(
            f"candidate matrix must be n x {SIGNATURE_DIMENSIONS}; got {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("candidate signatures contain non-finite values")
    q = query.vector.astype(float)
    close_rows, close_query = _parts(matrix, "close_path"), query.part("close_path")
    price = _path_distance(close_rows, close_query)

    stage = np.sqrt(np.mean((_parts(matrix, "stage") - query.part("stage")) ** 2, axis=1))
    candle_rows = np.c_[_parts(matrix, "range"), _parts(matrix, "body")]
    candle_query = np.r_[query.part("range"), query.part("body")]
    candle = np.sqrt(np.mean((candle_rows - candle_query) ** 2, axis=1))
    volume_rows = np.c_[_parts(matrix, "volume"), _parts(matrix, "shock")]
    volume_query = np.r_[query.part("volume"), query.part("shock")]
    volume = np.sqrt(np.mean((volume_rows - volume_query) ** 2, axis=1))

    market_rows = _parts(matrix, "market_path")
    relative_rows = _parts(matrix, "relative_path")
    context = .5 * _path_distance(market_rows, query.part("market_path")) + .5 * _path_distance(
        relative_rows, query.part("relative_path"),
    )
    available = (_parts(matrix, "context_available")[:, 0] > .5) & (
        query.part("context_available")[0] > .5
    )
    context = np.where(available, context, 1.0)

    drawdown = np.sqrt(np.mean(
        (_parts(matrix, "drawdown") - query.part("drawdown")) ** 2, axis=1,
    ))
    direction = np.mean(
        _parts(matrix, "direction") != query.part("direction"), axis=1,
    )
    structural = .6 * drawdown + .4 * direction

    coarse_rows = np.c_[
        close_rows, _parts(matrix, "range"), _parts(matrix, "volume"),
        relative_rows, market_rows,
    ]
    coarse_query = np.r_[
        close_query, query.part("range"), query.part("volume"),
        query.part("relative_path"), query.part("market_path"),
    ]
    joined_mean = (coarse_rows.sum(axis=1) + coarse_query.sum()) / (
        coarse_rows.shape[1] + len(coarse_query)
    )
    variance = (
        np.sum((coarse_rows - joined_mean[:, None]) ** 2, axis=1)
        + np.sum((coarse_query[None, :] - joined_mean[:, None]) ** 2, axis=1)
    ) / (coarse_rows.shape[1] + len(coarse_query))
    coarse = np.sqrt(np.mean((coarse_rows - coarse_query) ** 2, axis=1)) / np.maximum(
        np.sqrt(variance), 1e-6,
    )
    values = {
        "coarse": coarse, "price_shape": price, "stage": stage,
        "candle_volatility": candle, "volume_shock": volume,
        "market_context": context, "structural": structural,
    }
    if tuple(values) != VIEW_NAMES:
        raise AssertionError("persisted view order disagrees with candidate view order")
    return values
