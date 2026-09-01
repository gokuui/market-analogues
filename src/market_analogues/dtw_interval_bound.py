from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .distance import representation_distance
from .representation import Representation
from .types import stable_hash


DTW_CHANNELS = ("close_path", "atr_pct", "volume_robust_z", "relative_path")
SAMPLES = 64


class DtwIntervalBoundError(ValueError):
    pass


@dataclass(frozen=True)
class QuantizedDtwSamples:
    centers: np.ndarray
    channel_error_radii: np.ndarray
    presence: np.ndarray


def dtw_interval_bound_contract() -> dict[str, object]:
    state: dict[str, object] = {
        "schema_version": "quantized-dtw-interval-bound-v1",
        "channels": list(DTW_CHANNELS), "samples_per_channel": SAMPLES,
        "storage": (
            "float16 sample centers, one outward-rounded float32 maximum absolute "
            "quantization radius and one presence bit per channel"
        ),
        "pair_scale_upper": (
            "joint query/candidate interval range, bounded below by 1e-6; this "
            "dominates the exact joint IQR or constant-series std fallback"
        ),
        "path_bound": (
            "maximum of the two one-way sums of minimum interval local costs "
            "inside the exact Sakoe-Chiba band, divided by n+m-1"
        ),
        "composition": (
            "existing safe 0.55 rigid-price bound plus 0.45 times this DTW bound"
        ),
        "outcomes_or_labels_used": False,
    }
    return {**state, "digest": stable_hash(state)}


def quantize_dtw_samples(representation: Representation) -> QuantizedDtwSamples:
    centers = np.zeros((len(DTW_CHANNELS), SAMPLES), dtype=np.float16)
    radii = np.zeros(len(DTW_CHANNELS), dtype=np.float32)
    presence = np.zeros(len(DTW_CHANNELS), dtype=bool)
    for channel, name in enumerate(DTW_CHANNELS):
        raw = representation.samples_64.get(name)
        if raw is None:
            continue
        values = np.asarray(raw, dtype=np.float64)
        if values.shape != (SAMPLES,) or not np.isfinite(values).all() \
                or np.any(np.abs(values) > np.finfo(np.float16).max):
            raise DtwIntervalBoundError(f"DTW channel cannot be quantized: {name}")
        quantized = values.astype(np.float16)
        error = float(np.max(np.abs(values - quantized.astype(np.float64))))
        radius = np.nextafter(np.float32(error), np.float32(np.inf))
        if not np.isfinite(radius):
            raise DtwIntervalBoundError(f"DTW channel radius is invalid: {name}")
        centers[channel] = quantized
        radii[channel] = radius
        presence[channel] = True
    return QuantizedDtwSamples(centers, radii, presence)


def validate_quantized_dtw_samples(value: QuantizedDtwSamples) -> None:
    if not all((
        isinstance(value.centers, np.ndarray), value.centers.dtype == np.float16,
        value.centers.shape == (len(DTW_CHANNELS), SAMPLES),
        isinstance(value.channel_error_radii, np.ndarray),
        value.channel_error_radii.dtype == np.float32,
        value.channel_error_radii.shape == (len(DTW_CHANNELS),),
        isinstance(value.presence, np.ndarray), value.presence.dtype == bool,
        value.presence.shape == (len(DTW_CHANNELS),),
        np.isfinite(value.centers).all(),
        np.isfinite(value.channel_error_radii).all(),
        np.all(value.channel_error_radii >= 0),
        np.all(value.centers[~value.presence] == 0),
        np.all(value.channel_error_radii[~value.presence] == 0),
    )):
        raise DtwIntervalBoundError("quantized DTW sample row differs")


def quantized_dtw_lower_bound(
    query: Representation,
    candidate: QuantizedDtwSamples,
    *,
    band_fraction: float = 0.12,
) -> float:
    validate_quantized_dtw_samples(candidate)
    if not np.isfinite(band_fraction) or band_fraction < 0:
        raise DtwIntervalBoundError("DTW band fraction is invalid")
    included = []
    for channel, name in enumerate(DTW_CHANNELS):
        query_values = query.samples_64.get(name)
        if query_values is not None and candidate.presence[channel]:
            values = np.asarray(query_values, dtype=np.float64)
            if values.shape != (SAMPLES,) or not np.isfinite(values).all():
                raise DtwIntervalBoundError(f"query DTW channel differs: {name}")
            included.append((channel, values))
    if not included:
        return 0.0
    band = max(int(SAMPLES * band_fraction), 1)
    query_matrix = []
    center_matrix = []
    radius_vector = []
    denominators = []
    for channel, query_values in included:
        center = candidate.centers[channel].astype(np.float64)
        radius = float(candidate.channel_error_radii[channel])
        upper = max(float(np.max(query_values)), float(np.max(center + radius)))
        lower = min(float(np.min(query_values)), float(np.min(center - radius)))
        query_matrix.append(query_values)
        center_matrix.append(center)
        radius_vector.append(radius)
        denominators.append(max(upper - lower, 1e-6))
    queries = np.asarray(query_matrix, dtype=np.float64).T
    centers = np.asarray(center_matrix, dtype=np.float64).T
    radii = np.asarray(radius_vector, dtype=np.float64)
    scales = np.asarray(denominators, dtype=np.float64)
    difference = np.maximum(
        np.abs(queries[:, None, :] - centers[None, :, :]) - radii,
        0.0,
    ) / scales
    local = np.sqrt(np.mean(difference * difference, axis=2))
    indices = np.arange(SAMPLES)
    local[np.abs(indices[:, None] - indices[None, :]) > band] = np.inf
    # Every valid path visits every row and every column.  Either one-way sum
    # is therefore no larger than its cumulative path cost; their maximum is
    # still safe.  A path has at most n+m-1 cells, so this remains a bound on
    # the exact path-normalized distance.
    query_to_candidate = float(np.sum(np.min(local, axis=1)))
    candidate_to_query = float(np.sum(np.min(local, axis=0)))
    return max(query_to_candidate, candidate_to_query) / (2 * SAMPLES - 1)


def exact_price_component(query: Representation, candidate: Representation) -> float:
    return float(representation_distance(query, candidate)[1]["price"])


def verify_dtw_interval_bound(
    query: Representation,
    candidate: Representation,
    *,
    rigid_price_lower_bound: float,
    tolerance: float = 1e-12,
) -> tuple[float, float]:
    if not np.isfinite(rigid_price_lower_bound) or rigid_price_lower_bound < 0:
        raise DtwIntervalBoundError("rigid price lower bound is invalid")
    quantized = quantize_dtw_samples(candidate)
    dtw_lower = quantized_dtw_lower_bound(query, quantized)
    combined = rigid_price_lower_bound + 0.45 * dtw_lower
    exact = exact_price_component(query, candidate)
    if combined > exact + tolerance:
        raise DtwIntervalBoundError("combined quantized price bound exceeds exact distance")
    return combined, exact
