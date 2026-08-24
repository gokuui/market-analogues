from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .exact_aligned_features import SAMPLES_48_NAMES
from .quantized_bound import QuantizedBoundRow
from .representation import Representation


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
    "coarse": .08, "stage": .30, "price": .23,
    "candle_volatility": .10, "volume_shock": .10,
    "market_context": .10, "structural": .09,
}


@dataclass(frozen=True)
class ReferenceQuantizedLowerBound:
    total: float
    components: dict[str, float]
    rigid_price: float


def _values(array: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(array).reshape(-1)]


def _rms_difference(left: np.ndarray, right: np.ndarray) -> float:
    x, y = _values(left), _values(right)
    return math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(x, y)) / len(x))


def _std(values: list[float]) -> float:
    mean = math.fsum(values) / len(values)
    return math.sqrt(math.fsum((value - mean) ** 2 for value in values) / len(values))


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * quantile
    lower, upper = math.floor(location), math.ceil(location)
    fraction = location - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _numerator(query: np.ndarray, stored: np.ndarray, radius: float) -> float:
    return max(_rms_difference(query, stored) - radius, 0.0)


def reference_quantized_lower_bound(
    query: Representation,
    row: QuantizedBoundRow,
) -> ReferenceQuantizedLowerBound:
    radii = _values(row.error_radii)
    count = len(SAMPLES_48_NAMES)
    rms_radii = radii[1:1 + count]
    max_radii = radii[1 + count:1 + 2 * count]
    coarse_stored = row.coarse.astype(np.float64)
    coarse_query = np.asarray(query.coarse, dtype=np.float64)
    coarse_joined = _values(coarse_stored) + _values(coarse_query)
    components = {
        "coarse": _numerator(coarse_query, coarse_stored, radii[0]) / max(
            _std(coarse_joined) + radii[0] / math.sqrt(2.0), 1e-6,
        ),
        "stage": _numerator(query.stage, row.stage, radii[-2]),
        "structural": _numerator(query.structural, row.structural, radii[-1]),
    }
    index_by_name = {name: index for index, name in enumerate(SAMPLES_48_NAMES)}
    for group, names in REFERENCE_GROUPS.items():
        group_values: list[float] = []
        for name in names:
            index = index_by_name[name]
            query_values = query.samples_48.get(name)
            candidate_present = bool(row.presence[index])
            if query_values is None and not candidate_present:
                continue
            if query_values is None or not candidate_present:
                group_values.append(2.0)
                continue
            stored = row.samples_48[index].astype(np.float64)
            joined = _values(stored) + _values(query_values)
            denominator = max(
                _percentile(joined, .75) - _percentile(joined, .25)
                + 2.0 * max_radii[index],
                _std(joined) + rms_radii[index] / math.sqrt(2.0),
                1e-6,
            )
            group_values.append(
                _numerator(query_values, stored, rms_radii[index]) / denominator
            )
        components[group] = (
            math.fsum(group_values) / len(group_values) if group_values else 0.0
        )
    rigid_price = components["price"]
    components["price"] = .55 * rigid_price
    total = math.fsum(
        REFERENCE_WEIGHTS[name] * components[name] for name in REFERENCE_WEIGHTS
    )
    return ReferenceQuantizedLowerBound(total, components, rigid_price)
