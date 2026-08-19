from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .representation import Representation, resample_optional, stage_signature, structural_signature


GROUPS: dict[str, tuple[str, ...]] = {
    "price": ("close_path", "return", "overnight", "intraday", "distance_high_63", "distance_ma_20"),
    "candle_volatility": ("range_pct", "body_atr", "upper_wick_atr", "lower_wick_atr", "atr_pct", "compression_ratio"),
    "volume_shock": ("volume_robust_z", "return_shock_z"),
    "market_context": ("benchmark_path", "benchmark_return", "benchmark_drawdown", "relative_path", "relative_return"),
}


@dataclass(frozen=True)
class DistanceConfig:
    weights: dict[str, float] = field(default_factory=lambda: {
        "coarse": .08, "stage": .30, "price": .23, "candle_volatility": .10,
        "volume_shock": .10, "market_context": .10, "structural": .09,
    })
    samples_per_channel: int = 48
    dtw_band_fraction: float = .12


def _fill_and_resample(values: pd.Series, n: int) -> np.ndarray | None:
    return resample_optional(values, n)


def _robust_scale_pair(a: np.ndarray, b: np.ndarray, preserve_level: bool) -> tuple[np.ndarray, np.ndarray]:
    joined = np.r_[a, b]
    scale = np.nanpercentile(joined, 75) - np.nanpercentile(joined, 25)
    if scale < 1e-8:
        scale = np.nanstd(joined)
    scale = max(float(scale), 1e-6)
    if preserve_level:
        return a / scale, b / scale
    center = float(np.nanmedian(joined))
    return (a - center) / scale, (b - center) / scale


def channel_distance(a: pd.DataFrame, b: pd.DataFrame, names: tuple[str, ...], n: int = 48) -> float:
    distances: list[float] = []
    for name in names:
        if name not in a or name not in b:
            continue
        x, y = _fill_and_resample(a[name], n), _fill_and_resample(b[name], n)
        if x is None and y is None:
            continue
        if x is None or y is None:
            distances.append(2.0)
            continue
        # Level is meaningful for paths, drawdowns, volatility and distances;
        # unit invariance has already been handled in representation.
        x, y = _robust_scale_pair(x, y, preserve_level=True)
        distances.append(float(np.sqrt(np.mean((x - y) ** 2))))
    return float(np.mean(distances)) if distances else 0.0


def _sampled_channel_distance(
    a: Representation, b: Representation, names: tuple[str, ...],
) -> float:
    distances: list[float] = []
    for name in names:
        x, y = a.samples_48.get(name), b.samples_48.get(name)
        if x is None and y is None:
            continue
        if x is None or y is None:
            distances.append(2.0)
            continue
        left, right = _robust_scale_pair(x, y, preserve_level=True)
        distances.append(float(np.sqrt(np.mean((left - right) ** 2))))
    return float(np.mean(distances)) if distances else 0.0


def bounded_dtw(x: np.ndarray, y: np.ndarray, band_fraction: float = .12) -> tuple[float, list[tuple[int, int]]]:
    """Exact Sakoe-Chiba-banded DTW reference implementation."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim == 1:
        x, y = x[:, None], y[:, None]
    n, m = len(x), len(y)
    band = max(abs(n - m), int(max(n, m) * band_fraction), 1)
    costs = np.full((n + 1, m + 1), np.inf)
    costs[0, 0] = 0.0
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m, i + band) + 1):
            choices = ((costs[i - 1, j], (i - 1, j)),
                       (costs[i, j - 1], (i, j - 1)),
                       (costs[i - 1, j - 1], (i - 1, j - 1)))
            previous, prev_idx = min(choices, key=lambda item: item[0])
            local = float(np.sqrt(np.mean((x[i - 1] - y[j - 1]) ** 2)))
            costs[i, j] = local + previous
            parent[i, j] = prev_idx
    if not np.isfinite(costs[n, m]):
        return float("inf"), []
    path: list[tuple[int, int]] = []
    cursor = (n, m)
    while cursor != (0, 0):
        i, j = cursor
        if i and j:
            path.append((i - 1, j - 1))
        cursor = parent[cursor]
    path.reverse()
    return float(costs[n, m] / max(len(path), 1)), path


def _dtw_price(a: pd.DataFrame, b: pd.DataFrame, band_fraction: float) -> tuple[float, list[tuple[int, int]]]:
    names = ("close_path", "atr_pct", "volume_robust_z", "relative_path")
    left, right = [], []
    for name in names:
        x, y = _fill_and_resample(a[name], 64), _fill_and_resample(b[name], 64)
        if x is None or y is None:
            continue
        x, y = _robust_scale_pair(x, y, preserve_level=True)
        left.append(x)
        right.append(y)
    if not left:
        return 0.0, []
    return bounded_dtw(np.column_stack(left), np.column_stack(right), band_fraction)


def _dtw_matrices(a: Representation, b: Representation) -> tuple[np.ndarray, np.ndarray]:
    left, right = [], []
    for name in ("close_path", "atr_pct", "volume_robust_z", "relative_path"):
        x, y = a.samples_64.get(name), b.samples_64.get(name)
        if x is None or y is None:
            continue
        x, y = _robust_scale_pair(x, y, preserve_level=True)
        left.append(x)
        right.append(y)
    if not left:
        return np.empty((0, 0)), np.empty((0, 0))
    return np.column_stack(left), np.column_stack(right)


def _dtw_representation(
    a: Representation, b: Representation, band_fraction: float,
) -> tuple[float, list[tuple[int, int]]]:
    left, right = _dtw_matrices(a, b)
    if not len(left):
        return 0.0, []
    return bounded_dtw(left, right, band_fraction)


def _lb_keogh_one_way(query: np.ndarray, candidate: np.ndarray, band: int) -> float:
    if not len(query) or not len(candidate):
        return 0.0
    total = 0.0
    for index, value in enumerate(candidate):
        start, end = max(0, index - band), min(len(query), index + band + 1)
        neighborhood = query[start:end]
        lower, upper = neighborhood.min(axis=0), neighborhood.max(axis=0)
        deviation = np.maximum(lower - value, 0.0) + np.minimum(upper - value, 0.0)
        total += float(np.sqrt(np.mean(deviation * deviation)))
    # bounded_dtw returns cumulative cost divided by path length. Every candidate
    # point must be visited and a path can contain at most n+m-1 positions.
    return total / max(len(query) + len(candidate) - 1, 1)


def representation_dtw_lower_bound(
    a: Representation, b: Representation, band_fraction: float = .12,
) -> float:
    """Symmetric multivariate LB_Keogh bound for the normalized DTW component."""
    left, right = _dtw_matrices(a, b)
    if not len(left):
        return 0.0
    band = max(abs(len(left) - len(right)), int(max(len(left), len(right)) * band_fraction), 1)
    return max(
        _lb_keogh_one_way(left, right, band),
        _lb_keogh_one_way(right, left, band),
    )


def structural_distance(a: pd.DataFrame, b: pd.DataFrame) -> float:
    x, y = structural_signature(a), structural_signature(b)
    return float(np.sqrt(np.mean((x - y) ** 2)))


def stage_distance(a: pd.DataFrame, b: pd.DataFrame, stages: int = 12) -> float:
    """Compare the ordered low-frequency development a chart reader perceives."""
    x, y = stage_signature(a, stages), stage_signature(b, stages)
    return float(np.sqrt(np.mean((x - y) ** 2)))


def representation_distance(a: Representation, b: Representation, config: DistanceConfig | None = None) -> tuple[float, dict[str, float], list[tuple[int, int]]]:
    lower_bound, components, rigid_price = representation_distance_lower_bound(a, b, config)
    return complete_representation_distance(
        a, b, lower_bound, components, rigid_price, config,
    )


def representation_distance_lower_bound(
    a: Representation,
    b: Representation,
    config: DistanceConfig | None = None,
    *,
    include_dtw_bound: bool = False,
) -> tuple[float, dict[str, float], float]:
    """Exact non-DTW work plus a safe zero lower bound for the DTW remainder."""
    config = config or DistanceConfig()
    coarse_scale = max(float(np.std(np.r_[a.coarse, b.coarse])), 1e-6)
    components = {"coarse": float(np.sqrt(np.mean(((a.coarse - b.coarse) / coarse_scale) ** 2)))}
    components["stage"] = float(np.sqrt(np.mean((a.stage - b.stage) ** 2)))
    for group, names in GROUPS.items():
        if config.samples_per_channel == 48:
            components[group] = _sampled_channel_distance(a, b, names)
        else:
            components[group] = channel_distance(
                a.channels, b.channels, names, config.samples_per_channel,
            )
    rigid_price = components["price"]
    dtw_lower_bound = (
        representation_dtw_lower_bound(a, b, config.dtw_band_fraction)
        if include_dtw_bound else 0.0
    )
    components["price"] = .55 * rigid_price + .45 * dtw_lower_bound
    components["structural"] = float(np.sqrt(np.mean((a.structural - b.structural) ** 2)))
    lower_bound = sum(
        config.weights.get(name, 0.0) * value for name, value in components.items()
    )
    return float(lower_bound), components, float(rigid_price)


def complete_representation_distance(
    a: Representation,
    b: Representation,
    lower_bound: float,
    lower_components: dict[str, float],
    rigid_price: float,
    config: DistanceConfig | None = None,
) -> tuple[float, dict[str, float], list[tuple[int, int]]]:
    """Complete a previously computed safe lower bound with exact bounded DTW."""
    config = config or DistanceConfig()
    dtw, path = _dtw_representation(a, b, config.dtw_band_fraction)
    components = dict(lower_components)
    dtw_lower_bound = max((components["price"] - .55 * rigid_price) / .45, 0.0)
    components["price"] = .55 * rigid_price + .45 * dtw
    total = lower_bound + config.weights.get("price", 0.0) * .45 * (
        dtw - dtw_lower_bound
    )
    return float(total), components, path
