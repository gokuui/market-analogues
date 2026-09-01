from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

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
        "schema_version": "quantized-dtw-interval-bound-v2",
        "channels": list(DTW_CHANNELS), "samples_per_channel": SAMPLES,
        "storage": (
            "float16 sample centers, one outward-rounded float32 maximum absolute "
            "quantization radius and one presence bit per channel"
        ),
        "pair_scale_upper": (
            "outward joint interval IQR upper when its lower bound proves the "
            "exact IQR branch; otherwise max(interval IQR upper, RMS-based std "
            "upper), bounded below by 1e-6"
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


def quantized_dtw_orders(value: QuantizedDtwSamples) -> np.ndarray:
    validate_quantized_dtw_samples(value)
    return np.argsort(value.centers, axis=1, kind="stable").astype(np.uint8)


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


def _pair_scale_upper(
    query_values: np.ndarray, center: np.ndarray, radius: float,
) -> float:
    candidate_lower = center - radius
    candidate_upper = center + radius
    lower_values = np.r_[query_values, candidate_lower]
    upper_values = np.r_[query_values, candidate_upper]
    lower_q25, lower_q75 = np.percentile(lower_values, (25, 75))
    upper_q25, upper_q75 = np.percentile(upper_values, (25, 75))
    magnitude = max(
        float(np.max(np.abs(lower_values))),
        float(np.max(np.abs(upper_values))), 1.0,
    )
    rounding = 16 * np.finfo(np.float64).eps * magnitude
    iqr_lower = max(float(lower_q75 - upper_q25) - rounding, 0.0)
    iqr_upper = max(float(upper_q75 - lower_q25) + rounding, 0.0)
    if iqr_lower >= 1e-8:
        upper = iqr_upper
    else:
        # std(values) is no larger than RMS(values-reference) for any fixed
        # reference.  Endpoint distances bound every candidate interval value.
        reference = (float(np.min(lower_values)) + float(np.max(upper_values))) / 2
        query_deviation = np.abs(query_values - reference)
        candidate_deviation = np.maximum(
            np.abs(candidate_lower - reference),
            np.abs(candidate_upper - reference),
        )
        std_upper = float(np.sqrt(np.mean(np.r_[
            query_deviation * query_deviation,
            candidate_deviation * candidate_deviation,
        ]))) + rounding
        upper = max(iqr_upper, std_upper)
    return float(np.nextafter(max(upper, 1e-6), np.inf))


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
        query_matrix.append(query_values)
        center_matrix.append(center)
        radius_vector.append(radius)
        denominators.append(_pair_scale_upper(query_values, center, radius))
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


@njit(cache=True, nogil=True)
def _merged_rank(
    query_sorted: np.ndarray, centers: np.ndarray, order: np.ndarray,
    shift: float, target: int,
) -> float:
    query_index = candidate_index = 0
    value = 0.0
    for _position in range(target + 1):
        query_value = query_sorted[query_index] if query_index < SAMPLES else np.inf
        candidate_value = (
            centers[order[candidate_index]] + shift
            if candidate_index < SAMPLES else np.inf
        )
        if query_value <= candidate_value:
            value = query_value
            query_index += 1
        else:
            value = candidate_value
            candidate_index += 1
    return value


@njit(cache=True, nogil=True)
def _compiled_pair_scale_upper(
    query: np.ndarray, query_sorted: np.ndarray,
    centers: np.ndarray, order: np.ndarray, radius: float,
) -> float:
    lower31 = _merged_rank(query_sorted, centers, order, -radius, 31)
    lower32 = _merged_rank(query_sorted, centers, order, -radius, 32)
    lower95 = _merged_rank(query_sorted, centers, order, -radius, 95)
    lower96 = _merged_rank(query_sorted, centers, order, -radius, 96)
    upper31 = _merged_rank(query_sorted, centers, order, radius, 31)
    upper32 = _merged_rank(query_sorted, centers, order, radius, 32)
    upper95 = _merged_rank(query_sorted, centers, order, radius, 95)
    upper96 = _merged_rank(query_sorted, centers, order, radius, 96)
    lower_q25 = .25 * lower31 + .75 * lower32
    lower_q75 = .75 * lower95 + .25 * lower96
    upper_q25 = .25 * upper31 + .75 * upper32
    upper_q75 = .75 * upper95 + .25 * upper96
    minimum = np.inf
    maximum = -np.inf
    magnitude = 1.0
    for index in range(SAMPLES):
        qvalue = query[index]
        low = centers[index] - radius
        high = centers[index] + radius
        minimum = min(minimum, qvalue, low)
        maximum = max(maximum, qvalue, high)
        magnitude = max(magnitude, abs(qvalue), abs(low), abs(high))
    rounding = 16 * np.finfo(np.float64).eps * magnitude
    iqr_lower = max(lower_q75 - upper_q25 - rounding, 0.0)
    iqr_upper = max(upper_q75 - lower_q25 + rounding, 0.0)
    if iqr_lower >= 1e-8:
        upper = iqr_upper
    else:
        reference = (minimum + maximum) / 2
        squared = 0.0
        for index in range(SAMPLES):
            qdelta = abs(query[index] - reference)
            cdelta = max(
                abs(centers[index] - radius - reference),
                abs(centers[index] + radius - reference),
            )
            squared += qdelta * qdelta + cdelta * cdelta
        upper = max(iqr_upper, np.sqrt(squared / (2 * SAMPLES)) + rounding)
    return np.nextafter(max(upper, 1e-6), np.inf)


@njit(cache=True, nogil=True, parallel=True)
def _quantized_dtw_lower_bounds_compiled(
    query: np.ndarray, query_sorted: np.ndarray, query_presence: np.ndarray,
    centers: np.ndarray, orders: np.ndarray, radii: np.ndarray,
    presence: np.ndarray, band: int,
) -> np.ndarray:
    output = np.zeros(len(centers), dtype=np.float64)
    for row in prange(len(centers)):
        included = np.zeros(len(DTW_CHANNELS), dtype=np.bool_)
        denominators = np.ones(len(DTW_CHANNELS), dtype=np.float64)
        count = 0
        for channel in range(len(DTW_CHANNELS)):
            if query_presence[channel] and presence[row, channel]:
                included[channel] = True
                count += 1
                denominators[channel] = _compiled_pair_scale_upper(
                    query[channel], query_sorted[channel], centers[row, channel],
                    orders[row, channel], radii[row, channel],
                )
        if count == 0:
            continue
        candidate_minima = np.full(SAMPLES, np.inf)
        query_sum = 0.0
        for query_index in range(SAMPLES):
            query_minimum = np.inf
            first = max(0, query_index - band)
            last = min(SAMPLES, query_index + band + 1)
            for candidate_index in range(first, last):
                squared = 0.0
                for channel in range(len(DTW_CHANNELS)):
                    if included[channel]:
                        difference = max(
                            abs(query[channel, query_index]
                                - centers[row, channel, candidate_index])
                            - radii[row, channel], 0.0,
                        ) / denominators[channel]
                        squared += difference * difference
                local = np.sqrt(squared / count)
                query_minimum = min(query_minimum, local)
                candidate_minima[candidate_index] = min(
                    candidate_minima[candidate_index], local,
                )
            query_sum += query_minimum
        candidate_sum = 0.0
        for index in range(SAMPLES):
            candidate_sum += candidate_minima[index]
        output[row] = max(query_sum, candidate_sum) / (2 * SAMPLES - 1)
    return output


def quantized_dtw_lower_bounds(
    query: Representation, centers: np.ndarray, orders: np.ndarray,
    radii: np.ndarray, presence: np.ndarray, *, band_fraction: float = .12,
) -> np.ndarray:
    center_values = np.asarray(centers)
    order_values = np.asarray(orders)
    radius_values = np.asarray(radii)
    presence_values = np.asarray(presence)
    rows = len(center_values) if center_values.ndim else 0
    if not all((
        center_values.dtype in (np.dtype(np.float16), np.dtype(np.float32)),
        center_values.shape == (rows, len(DTW_CHANNELS), SAMPLES),
        order_values.dtype == np.uint8,
        order_values.shape == center_values.shape,
        radius_values.dtype in (np.dtype(np.float32), np.dtype(np.float64)),
        radius_values.shape == (rows, len(DTW_CHANNELS)),
        presence_values.dtype == bool,
        presence_values.shape == (rows, len(DTW_CHANNELS)),
        np.isfinite(center_values).all(), np.isfinite(radius_values).all(),
        np.all(radius_values >= 0),
        np.isfinite(band_fraction), band_fraction >= 0,
    )):
        raise DtwIntervalBoundError("quantized DTW batch differs")
    expected_order = np.arange(SAMPLES, dtype=np.uint8)
    if rows and np.any(np.sort(order_values, axis=2) != expected_order):
        raise DtwIntervalBoundError("quantized DTW order is not a permutation")
    query_values = np.zeros((len(DTW_CHANNELS), SAMPLES), dtype=np.float64)
    query_presence = np.zeros(len(DTW_CHANNELS), dtype=bool)
    for channel, name in enumerate(DTW_CHANNELS):
        values = query.samples_64.get(name)
        if values is None:
            continue
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (SAMPLES,) or not np.isfinite(values).all():
            raise DtwIntervalBoundError(f"query DTW channel differs: {name}")
        query_values[channel] = values
        query_presence[channel] = True
    band = max(int(SAMPLES * band_fraction), 1)
    return _quantized_dtw_lower_bounds_compiled(
        query_values, np.sort(query_values, axis=1), query_presence,
        center_values.astype(np.float32, copy=False), order_values,
        radius_values.astype(np.float64, copy=False), presence_values, band,
    )


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
