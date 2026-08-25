from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .distance import DistanceConfig, GROUPS, representation_distance_lower_bound
from .exact_aligned_features import SAMPLES_48_NAMES
from .representation import Representation
from .types import stable_hash


QUANTIZED_BOUND_VERSION = "distance-v1-full-f16-bound-v1"
FLOAT16_MAX = float(np.finfo(np.float16).max)
ERROR_NAMES = (
    "coarse_rms",
    *(f"samples_48_rms:{name}" for name in SAMPLES_48_NAMES),
    *(f"samples_48_max:{name}" for name in SAMPLES_48_NAMES),
    "stage_rms",
    "structural_rms",
)
FLOAT16_VALUE_COUNT = 128 + len(SAMPLES_48_NAMES) * 48 + 48 + 9
ERROR_VALUE_COUNT = len(ERROR_NAMES)
PRESENCE_BYTES = (len(SAMPLES_48_NAMES) + 7) // 8
UNALIGNED_ROW_BYTES = (
    FLOAT16_VALUE_COUNT * 2 + ERROR_VALUE_COUNT * 4 + PRESENCE_BYTES
    + 12 + 8 + 4 + 1
)
PACKED_ROW_BYTES = ((UNALIGNED_ROW_BYTES + 63) // 64) * 64


class QuantizedBoundError(ValueError):
    pass


@dataclass(frozen=True)
class QuantizedBoundRow:
    coarse: np.ndarray
    samples_48: np.ndarray
    presence: np.ndarray
    stage: np.ndarray
    structural: np.ndarray
    error_radii: np.ndarray

    def __post_init__(self) -> None:
        fields = {
            "coarse": (self.coarse, np.float16, (128,)),
            "samples_48": (self.samples_48, np.float16, (len(SAMPLES_48_NAMES), 48)),
            "presence": (self.presence, np.bool_, (len(SAMPLES_48_NAMES),)),
            "stage": (self.stage, np.float16, (48,)),
            "structural": (self.structural, np.float16, (9,)),
            "error_radii": (self.error_radii, np.float32, (ERROR_VALUE_COUNT,)),
        }
        for name, (value, dtype, shape) in fields.items():
            array = np.asarray(value, dtype=dtype)
            if array.shape != shape:
                raise QuantizedBoundError(f"{name} shape {array.shape} != {shape}")
            if name != "presence" and not np.isfinite(array).all():
                raise QuantizedBoundError(f"{name} contains non-finite values")
            if name == "error_radii" and np.any(array < 0):
                raise QuantizedBoundError("error radii must be nonnegative")
            object.__setattr__(self, name, array)


@dataclass(frozen=True)
class QuantizedLowerBound:
    total: float
    components: dict[str, float]
    rigid_price: float


@dataclass(frozen=True)
class QuantizedLowerBoundBatch:
    totals: np.ndarray
    components: dict[str, np.ndarray]
    rigid_price: np.ndarray


def quantized_bound_contract() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": QUANTIZED_BOUND_VERSION,
        "distance_contract": "market-analogue-distance-v1-contract",
        "stored_fields": {
            "coarse": [128, "float16"],
            "samples_48": [len(SAMPLES_48_NAMES), 48, "float16"],
            "stage": [48, "float16"],
            "structural": [9, "float16"],
            "presence": [len(SAMPLES_48_NAMES), "packed bits"],
            "error_radii": [ERROR_VALUE_COUNT, "outward-rounded float32"],
        },
        "excluded_fields": (
            "64-sample DTW fields are omitted; their nonnegative weighted contribution "
            "is bounded by zero"
        ),
        "overflow_policy": (
            "reject the row on any non-finite value or finite value outside float16; "
            "never clip, saturate, or emit a bound"
        ),
        "numerator_proof": (
            "reverse triangle inequality: max(RMS(stored-query)-candidate_error_rms,0) "
            "<= RMS(native_candidate-query)"
        ),
        "denominator_proof": {
            "std": (
                "standard deviation is the norm of the centered vector; changing only "
                "the candidate half changes joined RMS by candidate_error_rms/sqrt(2)"
            ),
            "iqr": (
                "each linear-interpolation quantile is L-infinity Lipschitz; candidate "
                "error_max therefore expands IQR by at most 2*error_max"
            ),
            "branch": (
                "max(IQR upper, std upper, 1e-6) upper-bounds either distance-v1 "
                "IQR branch, its std fallback, and the scale floor"
            ),
        },
        "aggregation_proof": (
            "missingness inclusion counts are exact, all component weights are "
            "nonnegative, and price retains 55% rigid while assigning zero to DTW"
        ),
        "layout": {
            "float16_values": FLOAT16_VALUE_COUNT,
            "float32_error_values": ERROR_VALUE_COUNT,
            "presence_bytes": PRESENCE_BYTES,
            "episode_id_bytes": 12,
            "cutoff_bytes": 8,
            "symbol_id_bytes": 4,
            "quality_tier_bytes": 1,
            "unaligned_row_bytes": UNALIGNED_ROW_BYTES,
            "alignment_bytes": 64,
            "packed_row_bytes": PACKED_ROW_BYTES,
        },
        "outcomes_or_labels_used": False,
    }
    payload["digest"] = stable_hash(payload)
    return payload


def _outward_float32(value: float) -> np.float32:
    if not np.isfinite(value) or value < 0:
        raise QuantizedBoundError("invalid error radius")
    rounded = np.float32(value)
    if float(rounded) < value:
        rounded = np.nextafter(rounded, np.float32(np.inf), dtype=np.float32)
    return rounded


def _quantize(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    native = np.asarray(values, dtype=np.float64)
    if not np.isfinite(native).all():
        raise QuantizedBoundError("native field contains non-finite values")
    if np.any(np.abs(native) > FLOAT16_MAX):
        raise QuantizedBoundError("native field exceeds float16 range")
    stored = native.astype(np.float16)
    if not np.isfinite(stored).all():
        raise QuantizedBoundError("float16 conversion is non-finite")
    error = native - stored.astype(np.float64)
    rms = float(np.sqrt(np.mean(error * error)))
    maximum = float(np.max(np.abs(error)))
    return stored, rms, maximum


def quantize_bound_row(representation: Representation) -> QuantizedBoundRow:
    coarse, coarse_rms, _ = _quantize(representation.coarse)
    stage, stage_rms, _ = _quantize(representation.stage)
    structural, structural_rms, _ = _quantize(representation.structural)
    samples = np.zeros((len(SAMPLES_48_NAMES), 48), dtype=np.float16)
    presence = np.zeros(len(SAMPLES_48_NAMES), dtype=bool)
    sample_rms = np.zeros(len(SAMPLES_48_NAMES), dtype=np.float32)
    sample_max = np.zeros(len(SAMPLES_48_NAMES), dtype=np.float32)
    for index, name in enumerate(SAMPLES_48_NAMES):
        values = representation.samples_48.get(name)
        if values is None:
            continue
        stored, rms, maximum = _quantize(values)
        samples[index] = stored
        presence[index] = True
        sample_rms[index] = _outward_float32(rms)
        sample_max[index] = _outward_float32(maximum)
    radii = np.r_[
        _outward_float32(coarse_rms), sample_rms, sample_max,
        _outward_float32(stage_rms), _outward_float32(structural_rms),
    ].astype(np.float32)
    return QuantizedBoundRow(
        coarse, samples, presence, stage, structural, radii,
    )


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2)))


def _numerator(query: np.ndarray, stored: np.ndarray, radius: float) -> float:
    return max(_rms(np.asarray(stored, dtype=np.float64) - np.asarray(query, dtype=np.float64)) - radius, 0.0)


def quantized_representation_lower_bound(
    query: Representation,
    row: QuantizedBoundRow,
    config: DistanceConfig | None = None,
) -> QuantizedLowerBound:
    config = config or DistanceConfig()
    radii = row.error_radii.astype(np.float64)
    sample_rms = radii[1:1 + len(SAMPLES_48_NAMES)]
    sample_max = radii[1 + len(SAMPLES_48_NAMES):1 + 2 * len(SAMPLES_48_NAMES)]
    stage_radius, structural_radius = radii[-2:]
    coarse_stored = row.coarse.astype(np.float64)
    coarse_query = np.asarray(query.coarse, dtype=np.float64)
    coarse_joined = np.r_[coarse_stored, coarse_query]
    coarse_denominator = max(
        float(np.std(coarse_joined)) + float(radii[0]) / np.sqrt(2.0), 1e-6,
    )
    components = {
        "coarse": _numerator(coarse_query, coarse_stored, float(radii[0])) / coarse_denominator,
        "stage": _numerator(query.stage, row.stage, float(stage_radius)),
        "structural": _numerator(query.structural, row.structural, float(structural_radius)),
    }
    by_name = {name: index for index, name in enumerate(SAMPLES_48_NAMES)}
    for group, names in GROUPS.items():
        distances: list[float] = []
        for name in names:
            index = by_name[name]
            query_values = query.samples_48.get(name)
            candidate_present = bool(row.presence[index])
            if query_values is None and not candidate_present:
                continue
            if query_values is None or not candidate_present:
                distances.append(2.0)
                continue
            stored = row.samples_48[index].astype(np.float64)
            query_array = np.asarray(query_values, dtype=np.float64)
            joined = np.r_[stored, query_array]
            denominator = max(
                float(np.percentile(joined, 75) - np.percentile(joined, 25))
                + 2.0 * float(sample_max[index]),
                float(np.std(joined)) + float(sample_rms[index]) / np.sqrt(2.0),
                1e-6,
            )
            distances.append(
                _numerator(query_array, stored, float(sample_rms[index])) / denominator
            )
        components[group] = float(np.mean(distances)) if distances else 0.0
    rigid_price = components["price"]
    components["price"] = .55 * rigid_price
    total = sum(config.weights[name] * components[name] for name in config.weights)
    if not np.isfinite(total):
        raise QuantizedBoundError("quantized lower bound is non-finite")
    return QuantizedLowerBound(float(total), components, float(rigid_price))


def quantized_batch_lower_bounds(
    query: Representation,
    rows: tuple[QuantizedBoundRow, ...] | list[QuantizedBoundRow],
    config: DistanceConfig | None = None,
) -> QuantizedLowerBoundBatch:
    config = config or DistanceConfig()
    if not rows:
        empty = np.empty(0, dtype=np.float64)
        return QuantizedLowerBoundBatch(empty, {}, empty)
    radii = np.stack([row.error_radii for row in rows]).astype(np.float64)
    return quantized_array_lower_bounds(
        query,
        np.stack([row.coarse for row in rows]),
        np.stack([row.samples_48 for row in rows]),
        np.stack([row.presence for row in rows]),
        np.stack([row.stage for row in rows]),
        np.stack([row.structural for row in rows]),
        radii,
        config,
    )


def quantized_array_lower_bounds(
    query: Representation,
    coarse: np.ndarray,
    samples: np.ndarray,
    presence: np.ndarray,
    stage: np.ndarray,
    structural: np.ndarray,
    radii: np.ndarray,
    config: DistanceConfig | None = None,
) -> QuantizedLowerBoundBatch:
    """Vectorized bound kernel over array-backed rows, including mmap views."""
    config = config or DistanceConfig()
    coarse = np.asarray(coarse)
    samples = np.asarray(samples)
    presence = np.asarray(presence, dtype=bool)
    stage = np.asarray(stage)
    structural = np.asarray(structural)
    radii = np.asarray(radii)
    row_count = len(coarse)
    count = len(SAMPLES_48_NAMES)
    expected = {
        "coarse": (coarse.shape, (row_count, 128)),
        "samples": (samples.shape, (row_count, count, 48)),
        "presence": (presence.shape, (row_count, count)),
        "stage": (stage.shape, (row_count, 48)),
        "structural": (structural.shape, (row_count, 9)),
        "radii": (radii.shape, (row_count, ERROR_VALUE_COUNT)),
    }
    for name, (observed, required) in expected.items():
        if observed != required:
            raise QuantizedBoundError(f"{name} batch shape {observed} != {required}")
    if not row_count:
        empty = np.empty(0, dtype=np.float64)
        return QuantizedLowerBoundBatch(empty, {}, empty)
    if not all(np.isfinite(value).all() for value in (
        coarse, samples, stage, structural, radii,
    )) or np.any(radii < 0):
        raise QuantizedBoundError("quantized array batch contains invalid values")
    radii = radii.astype(np.float64)

    def numerator(query_values: np.ndarray, stored: np.ndarray, radius: np.ndarray) -> np.ndarray:
        observed = np.sqrt(np.mean(
            (stored.astype(np.float64) - np.asarray(query_values, dtype=np.float64)) ** 2,
            axis=1,
        ))
        return np.maximum(observed - radius, 0.0)

    def interquartile_range(joined: np.ndarray) -> np.ndarray:
        # One partition produces both quantiles.  Separate percentile calls
        # partition the identical 96-value rows twice.
        quartiles = np.percentile(joined, (25, 75), axis=1)
        return quartiles[1] - quartiles[0]

    coarse = coarse.astype(np.float64)
    coarse_error = radii[:, 0]
    coarse_joined = np.c_[coarse, np.broadcast_to(query.coarse, coarse.shape)]
    components: dict[str, np.ndarray] = {
        "coarse": numerator(query.coarse, coarse, coarse_error) / np.maximum(
            np.std(coarse_joined, axis=1) + coarse_error / np.sqrt(2.0), 1e-6,
        ),
    }
    stage = stage.astype(np.float64)
    structural = structural.astype(np.float64)
    components["stage"] = numerator(query.stage, stage, radii[:, -2])
    components["structural"] = numerator(
        query.structural, structural, radii[:, -1],
    )
    samples = samples.astype(np.float64)
    rms_radii = radii[:, 1:1 + count]
    max_radii = radii[:, 1 + count:1 + 2 * count]
    index_by_name = {name: index for index, name in enumerate(SAMPLES_48_NAMES)}
    for group, names in GROUPS.items():
        distances: list[np.ndarray] = []
        included: list[np.ndarray] = []
        for name in names:
            index = index_by_name[name]
            query_values = query.samples_48.get(name)
            candidate_present = presence[:, index]
            if query_values is None:
                distances.append(np.where(candidate_present, 2.0, 0.0))
                included.append(candidate_present)
                continue
            stored = samples[:, index]
            joined = np.c_[stored, np.broadcast_to(query_values, stored.shape)]
            denominator = np.maximum.reduce((
                interquartile_range(joined) + 2.0 * max_radii[:, index],
                np.std(joined, axis=1) + rms_radii[:, index] / np.sqrt(2.0),
                np.full(row_count, 1e-6),
            ))
            value = numerator(query_values, stored, rms_radii[:, index]) / denominator
            value[~candidate_present] = 2.0
            distances.append(value)
            included.append(np.ones(row_count, dtype=bool))
        included_count = np.sum(included, axis=0)
        components[group] = np.divide(
            np.sum(distances, axis=0), included_count,
            out=np.zeros(row_count, dtype=np.float64), where=included_count > 0,
        )
    rigid_price = components["price"].copy()
    components["price"] = .55 * rigid_price
    totals = sum(
        config.weights[name] * components[name] for name in config.weights
    )
    if not np.isfinite(totals).all():
        raise QuantizedBoundError("quantized lower-bound batch is non-finite")
    return QuantizedLowerBoundBatch(totals, components, rigid_price)


def verify_bound_against_native(
    query: Representation,
    candidate: Representation,
    *,
    tolerance: float = 1e-12,
) -> QuantizedLowerBound:
    result = quantized_representation_lower_bound(query, quantize_bound_row(candidate))
    native_total, native_components, _ = representation_distance_lower_bound(query, candidate)
    if result.total > native_total + tolerance:
        raise QuantizedBoundError("quantized total exceeds native lower bound")
    for name, value in result.components.items():
        if value > native_components[name] + tolerance:
            raise QuantizedBoundError(f"quantized {name} exceeds native component")
    return result
