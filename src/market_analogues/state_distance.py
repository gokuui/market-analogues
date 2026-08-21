from __future__ import annotations

import numpy as np

from .multiresolution import MultiResolutionState, REQUIRED_HORIZONS
from .types import stable_hash


HORIZON_WEIGHTS = {252: .18, 126: .20, 63: .24, 21: .20, 10: .11, 5: .07}

CHANNEL_SCALES = {
    "close_path": .15, "return": .02, "overnight": .015, "intraday": .015,
    "range_pct": .02, "body_atr": 1.0, "upper_wick_atr": .7,
    "lower_wick_atr": .7, "close_location": .35, "atr_pct": .015,
    "volume_robust_z": 1.5, "return_shock_z": 1.5,
    "distance_high_63": .10, "distance_ma_20": .08, "distance_ma_50": .10,
    "compression_ratio": .60, "benchmark_close": 250.0,
    "benchmark_return": .015, "benchmark_path": .10,
    "benchmark_volatility": .012, "benchmark_drawdown": .10,
    "relative_return": .02, "relative_path": .12,
}

CHANNEL_WEIGHTS = {
    "close_path": 3.0, "return": 1.2, "overnight": .5, "intraday": .7,
    "range_pct": 1.0, "body_atr": .5, "upper_wick_atr": .3,
    "lower_wick_atr": .3, "close_location": .5, "atr_pct": 1.2,
    "volume_robust_z": 1.3, "return_shock_z": .8,
    "distance_high_63": 1.4, "distance_ma_20": 1.0, "distance_ma_50": 1.0,
    "compression_ratio": 1.5, "benchmark_close": 0.0,
    "benchmark_return": .4, "benchmark_path": .5,
    "benchmark_volatility": .4, "benchmark_drawdown": .5,
    "relative_return": .8, "relative_path": 1.0,
}

SUMMARY_SCALES = {
    "net_log_return": .15, "maximum_drawdown": .12,
    "distance_from_window_high": .12, "realized_volatility": .015,
    "median_range_pct": .02, "range_contraction": .60,
    "latest_overnight": .02, "latest_close_location": .35,
    "latest_atr_pct": .015, "latest_compression_ratio": .60,
    "mean_volume_robust_z": 1.2, "latest_volume_robust_z": 1.5,
    "up_down_volume_z_spread": 1.5, "relative_log_return": .12,
    "benchmark_log_return": .10, "benchmark_context_fraction": .30,
    "directional_events_3pct": 4.0, "directional_events_6pct": 3.0,
    "directional_events_12pct": 2.0,
}

SUMMARY_WEIGHTS = {
    "net_log_return": 2.0, "maximum_drawdown": 1.5,
    "distance_from_window_high": 1.5, "realized_volatility": 1.0,
    "median_range_pct": .8, "range_contraction": 1.4,
    "latest_overnight": .3, "latest_close_location": .4,
    "latest_atr_pct": 1.0, "latest_compression_ratio": 1.2,
    "mean_volume_robust_z": .7, "latest_volume_robust_z": .7,
    "up_down_volume_z_spread": .9, "relative_log_return": 1.0,
    "benchmark_log_return": .4, "benchmark_context_fraction": .8,
    "directional_events_3pct": .8, "directional_events_6pct": 1.0,
    "directional_events_12pct": 1.2,
}

REBASED_CHANNELS = {"close_path", "relative_path", "benchmark_path", "benchmark_close"}

TOPOLOGY_HORIZON_WEIGHTS = {252: .55, 126: .30, 63: .15}
TOPOLOGY_SMOOTHING_WINDOWS = (1, 3, 5)
TOPOLOGY_LANDMARK_FRACTION = .25
TOPOLOGY_SLOPE_FRACTION = .75
TOPOLOGY_FRACTION = .20


def state_distance_contract() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "masked-multiresolution-distance-v1",
        "horizon_weights": HORIZON_WEIGHTS,
        "channel_scales": CHANNEL_SCALES,
        "channel_weights": CHANNEL_WEIGHTS,
        "summary_scales": SUMMARY_SCALES,
        "summary_weights": SUMMARY_WEIGHTS,
        "rebased_channels": sorted(REBASED_CHANNELS),
        "channel_fraction": .68,
        "summary_fraction": .32,
        "missing_one_side_penalty": 2.0,
        "distance_clip": 4.0,
    }
    payload["digest"] = stable_hash(payload)
    return payload


def state_distance_v2_contract() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "masked-multiresolution-distance-v2",
        "base_distance_digest": state_distance_contract()["digest"],
        "base_fraction": 1.0 - TOPOLOGY_FRACTION,
        "topology_fraction": TOPOLOGY_FRACTION,
        "topology": {
            "channel": "close_path",
            "horizon_weights": TOPOLOGY_HORIZON_WEIGHTS,
            "smoothing_windows": list(TOPOLOGY_SMOOTHING_WINDOWS),
            "landmark_fraction": TOPOLOGY_LANDMARK_FRACTION,
            "slope_fraction": TOPOLOGY_SLOPE_FRACTION,
            "landmark_scales": {
                "path_level": .12,
                "turn_location_fraction": .15,
                "drawdown": .12,
                "recovery": .12,
            },
            "slope_scale": .025,
            "distance_clip": 4.0,
        },
    }
    payload["digest"] = stable_hash(payload)
    return payload


def _sample_distance(left, right, name: str) -> float | None:
    common = left.observed & right.observed
    union = left.observed | right.observed
    if not union.any():
        return None
    mismatch = float(np.mean(left.observed != right.observed))
    if not common.any():
        return 2.0 + mismatch
    x = left.values[common].astype(float)
    y = right.values[common].astype(float)
    if name in REBASED_CHANNELS:
        x = x - x[0]
        y = y - y[0]
    scale = CHANNEL_SCALES[name]
    rmse = float(np.sqrt(np.mean(((x - y) / scale) ** 2)))
    return min(rmse, 4.0) + mismatch


def _summary_distance(x: float | None, y: float | None, name: str) -> float | None:
    if x is None and y is None:
        return None
    if x is None or y is None:
        return 2.0
    return min(abs(x - y) / SUMMARY_SCALES[name], 4.0)


def multiresolution_state_distance(
    left: MultiResolutionState,
    right: MultiResolutionState,
) -> tuple[float, dict[str, float]]:
    horizon_scores: dict[str, float] = {}
    total = 0.0
    for horizon in REQUIRED_HORIZONS:
        a, b = left.views[horizon], right.views[horizon]
        channel_total = channel_weight = 0.0
        for name, weight in CHANNEL_WEIGHTS.items():
            if weight <= 0:
                continue
            value = _sample_distance(a.samples[name], b.samples[name], name)
            if value is not None:
                channel_total += weight * value
                channel_weight += weight
        summary_total = summary_weight = 0.0
        for name, weight in SUMMARY_WEIGHTS.items():
            value = _summary_distance(a.summary[name], b.summary[name], name)
            if value is not None:
                summary_total += weight * value
                summary_weight += weight
        channel_score = channel_total / channel_weight if channel_weight else 0.0
        summary_score = summary_total / summary_weight if summary_weight else 0.0
        score = .68 * channel_score + .32 * summary_score
        horizon_scores[str(horizon)] = score
        total += HORIZON_WEIGHTS[horizon] * score
    return float(total), horizon_scores


def _edge_smoothed(values: np.ndarray, window: int) -> np.ndarray:
    if window == 1:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _topology_horizon_distance(left, right) -> float | None:
    a = left.samples["close_path"]
    b = right.samples["close_path"]
    common = a.observed & b.observed
    if int(common.sum()) < 8:
        return None
    x = a.values[common].astype(float)
    y = b.values[common].astype(float)
    x -= x[0]
    y -= y[0]
    n = len(x)

    def landmarks(values: np.ndarray) -> np.ndarray:
        running_peak = np.maximum.accumulate(values)
        drawdown = values - running_peak
        trough = int(np.argmin(values))
        deepest_drawdown = int(np.argmin(drawdown))
        return np.asarray([
            values.min() / .12,
            values.max() / .12,
            values[-1] / .12,
            (trough / (n - 1)) / .15,
            (deepest_drawdown / (n - 1)) / .15,
            drawdown.min() / .12,
            (values[-1] - values[trough]) / .12,
        ])

    landmark_distance = min(
        float(np.sqrt(np.mean((landmarks(x) - landmarks(y)) ** 2))), 4.0,
    )
    slope_distances = []
    for window in TOPOLOGY_SMOOTHING_WINDOWS:
        x_slope = np.gradient(_edge_smoothed(x, window))
        y_slope = np.gradient(_edge_smoothed(y, window))
        slope_distances.append(min(
            float(np.sqrt(np.mean(((x_slope - y_slope) / .025) ** 2))), 4.0,
        ))
    return (
        TOPOLOGY_LANDMARK_FRACTION * landmark_distance
        + TOPOLOGY_SLOPE_FRACTION * float(np.mean(slope_distances))
    )


def multiresolution_state_distance_v2(
    left: MultiResolutionState,
    right: MultiResolutionState,
) -> tuple[float, dict[str, float]]:
    base, _ = multiresolution_state_distance(left, right)
    topology_total = topology_weight = 0.0
    topology_components: dict[str, float] = {}
    for horizon, weight in TOPOLOGY_HORIZON_WEIGHTS.items():
        value = _topology_horizon_distance(left.views[horizon], right.views[horizon])
        if value is None:
            continue
        topology_components[f"topology_{horizon}"] = value
        topology_total += weight * value
        topology_weight += weight
    topology = topology_total / topology_weight if topology_weight else 0.0
    total = (1.0 - TOPOLOGY_FRACTION) * base + TOPOLOGY_FRACTION * topology
    return float(total), {"base": base, "topology": topology, **topology_components}
