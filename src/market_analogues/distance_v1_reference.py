from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .representation import Representation
from .types import stable_hash


REFERENCE_GROUPS: dict[str, tuple[str, ...]] = {
    "price": (
        "close_path", "return", "overnight", "intraday",
        "distance_high_63", "distance_ma_20",
    ),
    "candle_volatility": (
        "range_pct", "body_atr", "upper_wick_atr", "lower_wick_atr",
        "atr_pct", "compression_ratio",
    ),
    "volume_shock": ("volume_robust_z", "return_shock_z"),
    "market_context": (
        "benchmark_path", "benchmark_return", "benchmark_drawdown",
        "relative_path", "relative_return",
    ),
}
REFERENCE_WEIGHTS = {
    "coarse": .08,
    "stage": .30,
    "price": .23,
    "candle_volatility": .10,
    "volume_shock": .10,
    "market_context": .10,
    "structural": .09,
}
DTW_CHANNELS = ("close_path", "atr_pct", "volume_robust_z", "relative_path")


@dataclass(frozen=True)
class ReferenceDistanceResult:
    total: float
    components: dict[str, float]
    alignment: tuple[tuple[int, int], ...]


def distance_v1_contract() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "market-analogue-distance-v1-contract",
        "representation_version": "dense-v1",
        "weights": REFERENCE_WEIGHTS,
        "component_groups": REFERENCE_GROUPS,
        "coarse": {
            "values": 128,
            "pair_scale": "population_std(concat(left,right)), floor 1e-6",
            "distance": "RMSE((left-right)/pair_scale)",
        },
        "stage": {"values": 48, "distance": "RMSE(left-right)"},
        "rigid_groups": {
            "samples_per_channel": 48,
            "pair_scale": (
                "IQR(concat(left,right)); when IQR < 1e-8 use population std; "
                "floor final scale at 1e-6"
            ),
            "both_missing": "omit channel from group mean",
            "one_missing": "include fixed distance 2.0",
            "both_present": "RMSE((left-right)/pair_scale)",
            "empty_group": 0.0,
        },
        "price_blend": {"rigid_fraction": .55, "dtw_fraction": .45},
        "dtw": {
            "channels": DTW_CHANNELS,
            "samples_per_channel": 64,
            "channel_rule": "omit channel unless present on both sides",
            "pair_scale": "same IQR/std/floor rule independently per channel",
            "local_cost": "sqrt(mean((left[i]-right[j])^2 across retained channels))",
            "band": "max(abs(n-m), floor(max(n,m)*0.12), 1)",
            "predecessor_tie_order": ["vertical", "horizontal", "diagonal"],
            "normalization": "accumulated local cost / realized alignment path length",
            "no_common_channel": {"distance": 0.0, "alignment": []},
        },
        "certified_bounds": {
            "native_symmetric_lb_keogh_v1": {
                "scope": "the pair-scaled finite DTW matrices used by distance-v1",
                "one_way": (
                    "for every candidate position j, measure RMS deviation outside the "
                    "axis-aligned query envelope spanning all legal |i-j|<=band positions; "
                    "sum these deviations and divide by n+m-1"
                ),
                "symmetry": "maximum of query-to-candidate and candidate-to-query one-way bounds",
                "proof": [
                    "Every candidate position is visited by every legal warping path.",
                    "Its envelope deviation is no greater than the local RMS cost of any legal aligned query position.",
                    "The sum of one minimum contribution per candidate position is no greater than total path cost.",
                    "Every legal path has at most n+m-1 positions, so division by n+m-1 cannot exceed division by realized path length.",
                    "Each directional value is therefore <= normalized exact DTW; their maximum remains <= exact DTW.",
                ],
                "pair_scaling_condition": (
                    "the bound and exact DTW consume the identical already pair-scaled matrices; "
                    "the proof makes no claim across different scalings"
                ),
            },
            "full_quantized_bound": "not promoted here; proof/error-radius hardening is M04R-04",
        },
        "structural": {"values": 9, "distance": "RMSE(left-right)"},
        "total": "sum(component_weight * component_distance)",
        "numeric_contract": {
            "quartile_method": "linear interpolation at (N-1)*q on sorted finite values",
            "population_std_ddof": 0,
            "production_dtype": "stored samples may originate as float32/float16; scoring promotes to float64",
            "required_output": "finite total and components for valid Representation inputs",
        },
        "retrieval_contract": {
            "temporal": (
                "exclude identity/overlap; candidate cutoff earlier than query and no later "
                "than the query-session minimum-gap cutoff"
            ),
            "scope": "quality tiers and dataset/cross-dataset policy must admit candidate",
            "primary_sort": ["total_distance ascending", "episode_id ascending"],
            "constraints": [
                "maximum contributions per instrument",
                "optional same-instrument timestamp-overlap suppression",
            ],
        },
        "metric_diagnostics": {
            "identity": "expected zero for identical finite representations",
            "symmetry": "required",
            "triangle_inequality": "diagnostic only; not claimed",
        },
        "outcomes_or_labels_used": False,
    }
    payload["digest"] = stable_hash(payload)
    return payload


def _finite(values: np.ndarray) -> list[float]:
    return sorted(float(value) for value in np.asarray(values).reshape(-1) if math.isfinite(float(value)))


def _percentile(values: np.ndarray, quantile: float) -> float:
    ordered = _finite(values)
    if not ordered:
        return math.nan
    location = (len(ordered) - 1) * quantile
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    fraction = location - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _population_std(values: np.ndarray) -> float:
    finite = _finite(values)
    if not finite:
        return math.nan
    mean = math.fsum(finite) / len(finite)
    return math.sqrt(math.fsum((value - mean) ** 2 for value in finite) / len(finite))


def _pair_scale(left: np.ndarray, right: np.ndarray) -> float:
    joined = np.concatenate((np.asarray(left, dtype=float), np.asarray(right, dtype=float)))
    scale = _percentile(joined, .75) - _percentile(joined, .25)
    if scale < 1e-8:
        scale = _population_std(joined)
    return max(float(scale), 1e-6)


def _rmse(left: np.ndarray, right: np.ndarray, scale: float = 1.0) -> float:
    x = np.asarray(left, dtype=float).reshape(-1)
    y = np.asarray(right, dtype=float).reshape(-1)
    if len(x) != len(y):
        raise ValueError("RMSE inputs differ in length")
    return math.sqrt(math.fsum(((float(a) - float(b)) / scale) ** 2 for a, b in zip(x, y)) / len(x))


def _rigid_group(
    left: Representation,
    right: Representation,
    names: tuple[str, ...],
) -> float:
    values: list[float] = []
    for name in names:
        x = left.samples_48.get(name)
        y = right.samples_48.get(name)
        if x is None and y is None:
            continue
        if x is None or y is None:
            values.append(2.0)
            continue
        values.append(_rmse(x, y, _pair_scale(x, y)))
    return math.fsum(values) / len(values) if values else 0.0


def reference_bounded_dtw(
    left: np.ndarray,
    right: np.ndarray,
    band_fraction: float = .12,
) -> tuple[float, tuple[tuple[int, int], ...]]:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("DTW inputs must be two matrices with equal channel count")
    n, m = len(x), len(y)
    band = max(abs(n - m), math.floor(max(n, m) * band_fraction), 1)
    costs = [[math.inf] * (m + 1) for _ in range(n + 1)]
    parents: dict[tuple[int, int], tuple[int, int]] = {}
    costs[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m, i + band) + 1):
            value = costs[i - 1][j]
            parent = (i - 1, j)
            if costs[i][j - 1] < value:
                value = costs[i][j - 1]
                parent = (i, j - 1)
            if costs[i - 1][j - 1] < value:
                value = costs[i - 1][j - 1]
                parent = (i - 1, j - 1)
            local = math.sqrt(math.fsum(
                (float(a) - float(b)) ** 2 for a, b in zip(x[i - 1], y[j - 1])
            ) / x.shape[1])
            costs[i][j] = value + local
            parents[(i, j)] = parent
    if not math.isfinite(costs[n][m]):
        return math.inf, ()
    path: list[tuple[int, int]] = []
    cursor = (n, m)
    while cursor != (0, 0):
        i, j = cursor
        if i and j:
            path.append((i - 1, j - 1))
        cursor = parents[cursor]
    path.reverse()
    return costs[n][m] / max(len(path), 1), tuple(path)


def reference_representation_distance(
    left: Representation,
    right: Representation,
) -> ReferenceDistanceResult:
    coarse_joined = np.concatenate((
        np.asarray(left.coarse, dtype=float), np.asarray(right.coarse, dtype=float),
    ))
    coarse_scale = max(_population_std(coarse_joined), 1e-6)
    components = {
        "coarse": _rmse(left.coarse, right.coarse, coarse_scale),
        "stage": _rmse(left.stage, right.stage),
    }
    for group, names in REFERENCE_GROUPS.items():
        components[group] = _rigid_group(left, right, names)
    rigid_price = components["price"]
    dtw_left: list[np.ndarray] = []
    dtw_right: list[np.ndarray] = []
    for name in DTW_CHANNELS:
        x = left.samples_64.get(name)
        y = right.samples_64.get(name)
        if x is None or y is None:
            continue
        scale = _pair_scale(x, y)
        dtw_left.append(np.asarray(x, dtype=float) / scale)
        dtw_right.append(np.asarray(y, dtype=float) / scale)
    if dtw_left:
        dtw, path = reference_bounded_dtw(
            np.column_stack(dtw_left), np.column_stack(dtw_right), .12,
        )
    else:
        dtw, path = 0.0, ()
    components["price"] = .55 * rigid_price + .45 * dtw
    components["structural"] = _rmse(left.structural, right.structural)
    total = math.fsum(
        REFERENCE_WEIGHTS[name] * components[name] for name in REFERENCE_WEIGHTS
    )
    if not math.isfinite(total) or any(not math.isfinite(value) for value in components.values()):
        raise ValueError("distance-v1 reference produced non-finite output")
    return ReferenceDistanceResult(total, components, path)
