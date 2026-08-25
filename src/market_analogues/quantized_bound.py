from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from numba import njit, prange
import numpy as np

from .distance import DistanceConfig, GROUPS, representation_distance_lower_bound
from .exact_aligned_features import SAMPLES_48_NAMES
from .representation import Representation
from .types import stable_hash


QUANTIZED_BOUND_VERSION = "distance-v1-full-f16-bound-v1"
BRANCH_AWARE_QUANTIZED_BOUND_VERSION = "distance-v1-full-f16-bound-v2"
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


@njit(cache=True, nogil=True, parallel=True)
def _joined_iqr_compiled(
    samples: np.ndarray, query_samples: np.ndarray,
) -> np.ndarray:
    """Exact linear 25/75 percentiles over each joined 48+48 row."""
    rows, channels, width = samples.shape
    output = np.empty((rows, channels), dtype=np.float64)
    for row in prange(rows):
        joined = np.empty(width * 2, dtype=np.float64)
        for channel in range(channels):
            for index in range(width):
                joined[index] = samples[row, channel, index]
                joined[width + index] = query_samples[channel, index]
            joined.sort()
            lower = joined[24] - (joined[24] - joined[23]) * .25
            upper = joined[71] + (joined[72] - joined[71]) * .25
            output[row, channel] = upper - lower
    return output


@njit(cache=True, nogil=True, parallel=True)
def _joined_iqr_sorted_compiled(
    sorted_samples: np.ndarray, sorted_query_samples: np.ndarray,
) -> np.ndarray:
    """Exact joined IQR by merging sorted 48-value candidate/query rows."""
    rows, channels, width = sorted_samples.shape
    output = np.empty((rows, channels), dtype=np.float64)
    for row in prange(rows):
        for channel in range(channels):
            candidate_index = 0
            query_index = 0
            lower_left = 0.0
            lower_right = 0.0
            upper_left = 0.0
            upper_right = 0.0
            for merged_index in range(73):
                if (
                    candidate_index < width
                    and (
                        query_index >= width
                        or sorted_samples[row, channel, candidate_index]
                        <= sorted_query_samples[channel, query_index]
                    )
                ):
                    value = sorted_samples[row, channel, candidate_index]
                    candidate_index += 1
                else:
                    value = sorted_query_samples[channel, query_index]
                    query_index += 1
                if merged_index == 23:
                    lower_left = value
                elif merged_index == 24:
                    lower_right = value
                elif merged_index == 71:
                    upper_left = value
                elif merged_index == 72:
                    upper_right = value
            lower = lower_right - (lower_right - lower_left) * .25
            upper = upper_left + (upper_right - upper_left) * .25
            output[row, channel] = upper - lower
    return output


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


@dataclass(frozen=True)
class PreparedQuantizedBoundArrays:
    """Validated float64 candidate arrays reusable across query bounds."""

    coarse: np.ndarray
    samples: np.ndarray
    presence: np.ndarray
    stage: np.ndarray
    structural: np.ndarray
    radii: np.ndarray
    sorted_samples: np.ndarray | None = None

    @property
    def row_count(self) -> int:
        return len(self.coarse)

    def select(self, mask: np.ndarray) -> PreparedQuantizedBoundArrays:
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != (self.row_count,):
            raise QuantizedBoundError("prepared bound selection shape differs")
        return PreparedQuantizedBoundArrays(
            self.coarse[selected], self.samples[selected], self.presence[selected],
            self.stage[selected], self.structural[selected], self.radii[selected],
            (
                self.sorted_samples[selected]
                if self.sorted_samples is not None else None
            ),
        )


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


def branch_aware_quantized_bound_contract() -> dict[str, Any]:
    """A tighter admissible scorer over the unchanged v1 packed row layout."""
    payload = {
        "schema_version": BRANCH_AWARE_QUANTIZED_BOUND_VERSION,
        "stored_row_contract_digest": quantized_bound_contract()["digest"],
        "distance_contract": "market-analogue-distance-v1-contract",
        "denominator": (
            "outward joined-IQR interval [observed-2*max_error, "
            "observed+2*max_error]; use IQR upper when lower >= 1e-8, "
            "std upper when upper < 1e-8, otherwise max of both; every "
            "branch retains the 1e-6 floor"
        ),
        "boundary_rule": (
            "native IQR exactly 1e-8 uses the IQR branch; uncertain interval "
            "boundaries use the conservative maximum"
        ),
        "numerator_aggregation_and_layout": "identical to stored-row v1",
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
    return _quantized_representation_lower_bound(
        query, row, config, branch_aware=False,
    )


def branch_aware_quantized_representation_lower_bound(
    query: Representation,
    row: QuantizedBoundRow,
    config: DistanceConfig | None = None,
) -> QuantizedLowerBound:
    return _quantized_representation_lower_bound(
        query, row, config, branch_aware=True,
    )


def _quantized_representation_lower_bound(
    query: Representation,
    row: QuantizedBoundRow,
    config: DistanceConfig | None,
    *,
    branch_aware: bool,
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
            observed_iqr = float(
                np.percentile(joined, 75) - np.percentile(joined, 25)
            )
            raw_iqr_upper = observed_iqr + 2.0 * float(sample_max[index])
            raw_std_upper = (
                float(np.std(joined))
                + float(sample_rms[index]) / np.sqrt(2.0)
            )
            if branch_aware:
                iqr_upper = float(np.nextafter(raw_iqr_upper, np.inf))
                std_upper = float(np.nextafter(raw_std_upper, np.inf))
                iqr_lower = float(np.nextafter(
                    observed_iqr - 2.0 * float(sample_max[index]), -np.inf,
                ))
                if iqr_lower >= 1e-8:
                    denominator = max(iqr_upper, 1e-6)
                elif iqr_upper < 1e-8:
                    denominator = max(std_upper, 1e-6)
                else:
                    denominator = max(iqr_upper, std_upper, 1e-6)
            else:
                denominator = max(raw_iqr_upper, raw_std_upper, 1e-6)
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


def branch_aware_quantized_batch_lower_bounds(
    query: Representation,
    rows: tuple[QuantizedBoundRow, ...] | list[QuantizedBoundRow],
    config: DistanceConfig | None = None,
) -> QuantizedLowerBoundBatch:
    config = config or DistanceConfig()
    if not rows:
        empty = np.empty(0, dtype=np.float64)
        return QuantizedLowerBoundBatch(empty, {}, empty)
    prepared = prepare_quantized_bound_arrays(
        np.stack([row.coarse for row in rows]),
        np.stack([row.samples_48 for row in rows]),
        np.stack([row.presence for row in rows]),
        np.stack([row.stage for row in rows]),
        np.stack([row.structural for row in rows]),
        np.stack([row.error_radii for row in rows]).astype(np.float64),
    )
    return branch_aware_prepared_quantized_array_lower_bounds(
        query, prepared, config,
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
    prepared = prepare_quantized_bound_arrays(
        coarse, samples, presence, stage, structural, radii,
    )
    return prepared_quantized_array_lower_bounds(query, prepared, config)


def prepare_quantized_bound_arrays(
    coarse: np.ndarray,
    samples: np.ndarray,
    presence: np.ndarray,
    stage: np.ndarray,
    structural: np.ndarray,
    radii: np.ndarray,
    *,
    sort_samples: bool = False,
) -> PreparedQuantizedBoundArrays:
    """Validate and convert query-invariant packed arrays exactly once."""
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
    if not all(np.isfinite(value).all() for value in (
        coarse, samples, stage, structural, radii,
    )) or np.any(radii < 0):
        raise QuantizedBoundError("quantized array batch contains invalid values")
    samples64 = samples.astype(np.float64)
    return PreparedQuantizedBoundArrays(
        coarse.astype(np.float64), samples64, presence,
        stage.astype(np.float64), structural.astype(np.float64),
        radii.astype(np.float64),
        np.sort(samples64, axis=2) if sort_samples else None,
    )


def prepared_quantized_array_lower_bounds(
    query: Representation,
    prepared: PreparedQuantizedBoundArrays,
    config: DistanceConfig | None = None,
) -> QuantizedLowerBoundBatch:
    return _prepared_quantized_array_lower_bounds(
        query, prepared, config, branch_aware=False,
    )


def branch_aware_prepared_quantized_array_lower_bounds(
    query: Representation,
    prepared: PreparedQuantizedBoundArrays,
    config: DistanceConfig | None = None,
) -> QuantizedLowerBoundBatch:
    return _prepared_quantized_array_lower_bounds(
        query, prepared, config, branch_aware=True,
    )


def _prepared_quantized_array_lower_bounds(
    query: Representation,
    prepared: PreparedQuantizedBoundArrays,
    config: DistanceConfig | None,
    *,
    branch_aware: bool,
) -> QuantizedLowerBoundBatch:
    """Evaluate a query against already validated and converted candidates."""
    config = config or DistanceConfig()
    coarse = prepared.coarse
    samples = prepared.samples
    presence = prepared.presence
    stage = prepared.stage
    structural = prepared.structural
    radii = prepared.radii
    row_count = prepared.row_count
    count = len(SAMPLES_48_NAMES)
    if not row_count:
        empty = np.empty(0, dtype=np.float64)
        return QuantizedLowerBoundBatch(empty, {}, empty)

    def numerator(query_values: np.ndarray, stored: np.ndarray, radius: np.ndarray) -> np.ndarray:
        observed = np.sqrt(np.mean(
            (stored.astype(np.float64) - np.asarray(query_values, dtype=np.float64)) ** 2,
            axis=1,
        ))
        return np.maximum(observed - radius, 0.0)

    coarse_error = radii[:, 0]
    coarse_joined = np.c_[coarse, np.broadcast_to(query.coarse, coarse.shape)]
    components: dict[str, np.ndarray] = {
        "coarse": numerator(query.coarse, coarse, coarse_error) / np.maximum(
            np.std(coarse_joined, axis=1) + coarse_error / np.sqrt(2.0), 1e-6,
        ),
    }
    components["stage"] = numerator(query.stage, stage, radii[:, -2])
    components["structural"] = numerator(
        query.structural, structural, radii[:, -1],
    )
    query_samples = np.zeros((count, 48), dtype=np.float64)
    for index, name in enumerate(SAMPLES_48_NAMES):
        values = query.samples_48.get(name)
        if values is not None:
            query_samples[index] = values
    joined_iqr = (
        _joined_iqr_sorted_compiled(
            prepared.sorted_samples, np.sort(query_samples, axis=1),
        )
        if prepared.sorted_samples is not None
        else _joined_iqr_compiled(samples, query_samples)
    )
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
            raw_iqr_upper = (
                joined_iqr[:, index] + 2.0 * max_radii[:, index]
            )
            if branch_aware:
                iqr_upper = np.nextafter(raw_iqr_upper, np.inf)
                iqr_lower = np.nextafter(
                    joined_iqr[:, index] - 2.0 * max_radii[:, index], -np.inf,
                )
                iqr_certain = iqr_lower >= 1e-8
                denominator = np.maximum(iqr_upper, 1e-6)
                needs_std = ~iqr_certain
                if np.any(needs_std):
                    joined = np.c_[
                        stored[needs_std],
                        np.broadcast_to(query_values, stored.shape)[needs_std],
                    ]
                    std_upper = np.nextafter(
                        np.std(joined, axis=1)
                        + rms_radii[needs_std, index] / np.sqrt(2.0),
                        np.inf,
                    )
                    fallback_certain = iqr_upper[needs_std] < 1e-8
                    denominator[needs_std] = np.where(
                        fallback_certain,
                        np.maximum(std_upper, 1e-6),
                        np.maximum.reduce((
                            iqr_upper[needs_std], std_upper,
                            np.full(np.sum(needs_std), 1e-6),
                        )),
                    )
            else:
                joined = np.c_[
                    stored, np.broadcast_to(query_values, stored.shape),
                ]
                denominator = np.maximum.reduce((
                    raw_iqr_upper,
                    np.std(joined, axis=1)
                    + rms_radii[:, index] / np.sqrt(2.0),
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
