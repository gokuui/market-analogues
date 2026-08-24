from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .distance import GROUPS, DistanceConfig
from .exact_aligned_features import SAMPLES_48_NAMES
from .representation import Representation
from .types import stable_hash


PROPOSAL_V2_VERSION = "exact-aligned-proposal-v2"
PROPOSAL_ROUTES = (
    "coarse", "stage", "price", "candle_volatility", "volume_shock",
    "market_context", "structural",
)
PROPOSAL_POOLS = {
    1_000: {"per_component": 100, "composite": 300},
    5_000: {"per_component": 500, "composite": 1_500},
    10_000: {"per_component": 1_000, "composite": 3_000},
    20_000: {"per_component": 2_000, "composite": 6_000},
}


@dataclass(frozen=True)
class ProposalLayout:
    dimensions: int
    group_samples: dict[str, int]
    coarse_samples: int


LAYOUTS: dict[int, ProposalLayout] = {
    192: ProposalLayout(192, {
        "price": 8, "candle_volatility": 7,
        "volume_shock": 8, "market_context": 5,
    }, 4),
    240: ProposalLayout(240, {
        "price": 10, "candle_volatility": 9,
        "volume_shock": 10, "market_context": 8,
    }, 9),
    320: ProposalLayout(320, {
        "price": 14, "candle_volatility": 13,
        "volume_shock": 14, "market_context": 12,
    }, 13),
}


def _field_order(layout: ProposalLayout) -> tuple[tuple[str, str, int], ...]:
    fields: list[tuple[str, str, int]] = [
        ("stage", "stage", 48), ("structural", "structural", 9),
    ]
    for group in GROUPS:
        fields.extend(
            (group, name, layout.group_samples[group]) for name in GROUPS[group]
        )
    fields.append(("coarse", "coarse", layout.coarse_samples))
    return tuple(fields)


def proposal_v2_contract() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": PROPOSAL_V2_VERSION,
        "representation_version": "dense-v1",
        "layouts": {
            str(dimensions): {
                "dimensions": layout.dimensions,
                "group_samples": layout.group_samples,
                "coarse_samples": layout.coarse_samples,
                "field_order": _field_order(layout),
            }
            for dimensions, layout in LAYOUTS.items()
        },
        "presence": [len(SAMPLES_48_NAMES), "separate exact bits"],
        "projection": "linear interpolation of exact-aligned 48-sample fields and coarse vector",
        "scoring": {
            "stage": "native 48-value RMSE",
            "structural": "native 9-value RMSE",
            "groups": "distance-v1 pair IQR/std scaling, masks and missing penalty on projected fields",
            "price": "0.55 projected rigid group + 0.45 projected close-path distance",
            "coarse": "pair-standardized RMSE of projected coarse vector",
            "weights": DistanceConfig().weights,
        },
        "route_union": {
            "routes": ["composite", *PROPOSAL_ROUTES],
            **{f"pool_{pool}": quotas for pool, quotas in PROPOSAL_POOLS.items()},
            "deduplicate": True,
        },
        "candidate_dtype": {
            "float32": "native proposal baseline",
            "float16": (
                "saturating diagnostic only; count source values outside finite "
                "float16 and reject float16 selection if any occur"
            ),
            "selection_requires_native_scoring_confirmation": True,
        },
        "per_query_weights": False,
        "certified_bound": False,
        "outcomes_or_labels_used": False,
    }
    payload["digest"] = stable_hash(payload)
    return payload


@dataclass(frozen=True)
class ProposalSignatureV2:
    dimensions: int
    vector: np.ndarray
    presence: np.ndarray

    def __post_init__(self) -> None:
        if self.dimensions not in LAYOUTS:
            raise ValueError("unsupported proposal layout")
        vector = np.asarray(self.vector, dtype=np.float32)
        presence = np.asarray(self.presence, dtype=bool)
        if vector.shape != (self.dimensions,):
            raise ValueError("proposal vector has wrong shape")
        if presence.shape != (len(SAMPLES_48_NAMES),):
            raise ValueError("proposal mask has wrong shape")
        if not np.isfinite(vector).all():
            raise ValueError("proposal vector is non-finite")
        object.__setattr__(self, "vector", vector)
        object.__setattr__(self, "presence", presence)


@dataclass(frozen=True)
class ProposalBatchV2:
    dimensions: int
    vectors: np.ndarray
    presence: np.ndarray

    def __post_init__(self) -> None:
        vectors = np.asarray(self.vectors, dtype=np.float32)
        presence = np.asarray(self.presence, dtype=bool)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimensions:
            raise ValueError("proposal batch vectors have wrong shape")
        if presence.shape != (len(vectors), len(SAMPLES_48_NAMES)):
            raise ValueError("proposal batch masks have wrong shape")
        if not np.isfinite(vectors).all():
            raise ValueError("proposal batch vectors are non-finite")
        object.__setattr__(self, "vectors", vectors)
        object.__setattr__(self, "presence", presence)


def _sample_matrix(values: np.ndarray, count: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError("proposal source samples must be a non-empty matrix")
    positions = np.linspace(0, matrix.shape[1] - 1, count)
    left = np.floor(positions).astype(int)
    right = np.ceil(positions).astype(int)
    fraction = positions - left
    return matrix[:, left] * (1.0 - fraction) + matrix[:, right] * fraction


def proposal_signatures_v2(
    representations: tuple[Representation, ...] | list[Representation],
    dimensions: int,
) -> ProposalBatchV2:
    if dimensions not in LAYOUTS:
        raise ValueError("unsupported proposal layout")
    layout = LAYOUTS[dimensions]
    row_count = len(representations)
    if not row_count:
        return ProposalBatchV2(
            dimensions, np.empty((0, dimensions), dtype=np.float32),
            np.empty((0, len(SAMPLES_48_NAMES)), dtype=bool),
        )
    values: list[np.ndarray] = []
    presence = np.zeros((row_count, len(SAMPLES_48_NAMES)), dtype=bool)
    index_by_name = {name: index for index, name in enumerate(SAMPLES_48_NAMES)}
    for group, name, count in _field_order(layout):
        if group == "stage":
            values.append(np.stack([item.stage for item in representations]))
        elif group == "structural":
            values.append(np.stack([item.structural for item in representations]))
        elif group == "coarse":
            values.append(_sample_matrix(
                np.stack([item.coarse for item in representations]), count,
            ))
        else:
            source = [item.samples_48.get(name) for item in representations]
            present = np.asarray([item is not None for item in source])
            matrix = np.stack([
                np.zeros(48, dtype=np.float64) if item is None else item
                for item in source
            ])
            values.append(_sample_matrix(matrix, count))
            presence[:, index_by_name[name]] = present
    vectors = np.concatenate(values, axis=1).astype(np.float32)
    if vectors.shape != (row_count, dimensions):
        raise AssertionError(
            f"layout {dimensions} materialized {vectors.shape} values"
        )
    return ProposalBatchV2(dimensions, vectors, presence)


def proposal_signature_v2(
    representation: Representation,
    dimensions: int,
) -> ProposalSignatureV2:
    batch = proposal_signatures_v2([representation], dimensions)
    return ProposalSignatureV2(dimensions, batch.vectors[0], batch.presence[0])


def _slices(layout: ProposalLayout) -> dict[tuple[str, str], slice]:
    output = {}
    offset = 0
    for group, name, count in _field_order(layout):
        output[(group, name)] = slice(offset, offset + count)
        offset += count
    return output


def proposal_v2_distances(
    query: ProposalSignatureV2,
    candidate_vectors: np.ndarray,
    candidate_presence: np.ndarray,
) -> dict[str, np.ndarray]:
    layout = LAYOUTS[query.dimensions]
    matrix = np.asarray(candidate_vectors, dtype=np.float64)
    masks = np.asarray(candidate_presence, dtype=bool)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if masks.ndim == 1:
        masks = masks[None, :]
    if matrix.shape[1] != query.dimensions or masks.shape != (
        len(matrix), len(SAMPLES_48_NAMES),
    ):
        raise ValueError("proposal candidate matrix shape differs")
    if not np.isfinite(matrix).all():
        raise ValueError("proposal candidate matrix is non-finite")
    slices = _slices(layout)
    q = query.vector.astype(np.float64)
    components = {
        "stage": np.sqrt(np.mean(
            (matrix[:, slices[("stage", "stage")]] - q[slices[("stage", "stage")]]) ** 2,
            axis=1,
        )),
        "structural": np.sqrt(np.mean(
            (matrix[:, slices[("structural", "structural")]] - q[slices[("structural", "structural")]]) ** 2,
            axis=1,
        )),
    }
    index_by_name = {name: index for index, name in enumerate(SAMPLES_48_NAMES)}
    projected_channel_distances: dict[str, np.ndarray] = {}
    for group, names in GROUPS.items():
        distances = []
        included = []
        for name in names:
            index = index_by_name[name]
            candidate_present = masks[:, index]
            query_present = bool(query.presence[index])
            if not query_present:
                value = np.where(candidate_present, 2.0, 0.0)
                distances.append(value)
                projected_channel_distances[name] = value
                included.append(candidate_present)
                continue
            candidate = matrix[:, slices[(group, name)]]
            query_values = q[slices[(group, name)]]
            joined = np.c_[candidate, np.broadcast_to(query_values, candidate.shape)]
            scale = np.percentile(joined, 75, axis=1) - np.percentile(joined, 25, axis=1)
            scale = np.where(scale < 1e-8, np.std(joined, axis=1), scale)
            scale = np.maximum(scale, 1e-6)
            value = np.sqrt(np.mean((candidate - query_values) ** 2, axis=1)) / scale
            value[~candidate_present] = 2.0
            distances.append(value)
            projected_channel_distances[name] = value
            included.append(np.ones(len(matrix), dtype=bool))
        counts = np.sum(included, axis=0)
        components[group] = np.divide(
            np.sum(distances, axis=0), counts,
            out=np.zeros(len(matrix), dtype=np.float64), where=counts > 0,
        )
    # The exact price component is 55% rigid price channels and 45% elastic
    # multivariate DTW.  A flat proposal scan cannot run DTW for every row, so
    # its preregistered, query-independent proxy uses the projected close path
    # for the elastic share.  Keep this distinct from the rigid group average:
    # accidentally reusing that average would collapse the blend to one term.
    components["price"] = (
        .55 * components["price"]
        + .45 * projected_channel_distances["close_path"]
    )
    coarse_candidate = matrix[:, slices[("coarse", "coarse")]]
    coarse_query = q[slices[("coarse", "coarse")]]
    coarse_joined = np.c_[coarse_candidate, np.broadcast_to(coarse_query, coarse_candidate.shape)]
    components["coarse"] = np.sqrt(np.mean(
        (coarse_candidate - coarse_query) ** 2, axis=1,
    )) / np.maximum(np.std(coarse_joined, axis=1), 1e-6)
    weights = DistanceConfig().weights
    components["composite"] = sum(
        weights[name] * components[name] for name in weights
    )
    if any(not np.isfinite(value).all() for value in components.values()):
        raise ValueError("proposal distance is non-finite")
    return components


def proposal_v2_distances_many(
    queries: tuple[ProposalSignatureV2, ...] | list[ProposalSignatureV2],
    candidate_vectors: np.ndarray,
    candidate_presence: np.ndarray,
) -> dict[str, np.ndarray]:
    """Vectorized proposal distances shaped ``(queries, candidates)``.

    This is mathematically identical to repeated ``proposal_v2_distances``
    calls but amortizes pair-scaling work and Python dispatch during the
    multi-authority/full-universe experiment.
    """
    if not queries:
        raise ValueError("at least one proposal query is required")
    dimensions = queries[0].dimensions
    if any(query.dimensions != dimensions for query in queries):
        raise ValueError("proposal queries use different layouts")
    layout = LAYOUTS[dimensions]
    matrix = np.asarray(candidate_vectors, dtype=np.float64)
    masks = np.asarray(candidate_presence, dtype=bool)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if masks.ndim == 1:
        masks = masks[None, :]
    if matrix.shape[1] != dimensions or masks.shape != (
        len(matrix), len(SAMPLES_48_NAMES),
    ):
        raise ValueError("proposal candidate matrix shape differs")
    if not np.isfinite(matrix).all():
        raise ValueError("proposal candidate matrix is non-finite")
    query_matrix = np.stack([query.vector for query in queries]).astype(np.float64)
    query_masks = np.stack([query.presence for query in queries])
    slices = _slices(layout)
    query_count, candidate_count = len(queries), len(matrix)

    def rmse(field: tuple[str, str]) -> np.ndarray:
        section = slices[field]
        return np.sqrt(np.mean(
            (matrix[None, :, section] - query_matrix[:, None, section]) ** 2,
            axis=2,
        ))

    components = {
        "stage": rmse(("stage", "stage")),
        "structural": rmse(("structural", "structural")),
    }
    index_by_name = {name: index for index, name in enumerate(SAMPLES_48_NAMES)}
    projected_channel_distances: dict[str, np.ndarray] = {}
    for group, names in GROUPS.items():
        distances = []
        included = []
        for name in names:
            index = index_by_name[name]
            candidate_present = masks[:, index][None, :]
            query_present = query_masks[:, index][:, None]
            section = slices[(group, name)]
            candidate = matrix[:, section]
            query_values = query_matrix[:, section]
            joined = np.concatenate((
                np.broadcast_to(
                    candidate[None, :, :],
                    (query_count, candidate_count, candidate.shape[1]),
                ),
                np.broadcast_to(
                    query_values[:, None, :],
                    (query_count, candidate_count, query_values.shape[1]),
                ),
            ), axis=2)
            scale = np.percentile(joined, 75, axis=2) - np.percentile(
                joined, 25, axis=2,
            )
            scale = np.where(scale < 1e-8, np.std(joined, axis=2), scale)
            scale = np.maximum(scale, 1e-6)
            value = np.sqrt(np.mean(
                (candidate[None, :, :] - query_values[:, None, :]) ** 2,
                axis=2,
            )) / scale
            value = np.where(
                query_present,
                np.where(candidate_present, value, 2.0),
                np.where(candidate_present, 2.0, 0.0),
            )
            distances.append(value)
            included.append(query_present | candidate_present)
            projected_channel_distances[name] = value
        counts = np.sum(included, axis=0)
        components[group] = np.divide(
            np.sum(distances, axis=0), counts,
            out=np.zeros((query_count, candidate_count), dtype=np.float64),
            where=counts > 0,
        )
    components["price"] = (
        .55 * components["price"]
        + .45 * projected_channel_distances["close_path"]
    )
    coarse = slices[("coarse", "coarse")]
    candidate_coarse = matrix[:, coarse]
    query_coarse = query_matrix[:, coarse]
    coarse_joined = np.concatenate((
        np.broadcast_to(
            candidate_coarse[None, :, :],
            (query_count, candidate_count, candidate_coarse.shape[1]),
        ),
        np.broadcast_to(
            query_coarse[:, None, :],
            (query_count, candidate_count, query_coarse.shape[1]),
        ),
    ), axis=2)
    components["coarse"] = np.sqrt(np.mean(
        (candidate_coarse[None, :, :] - query_coarse[:, None, :]) ** 2,
        axis=2,
    )) / np.maximum(np.std(coarse_joined, axis=2), 1e-6)
    weights = DistanceConfig().weights
    components["composite"] = sum(
        weights[name] * components[name] for name in weights
    )
    if any(
        value.shape != (query_count, candidate_count)
        or not np.isfinite(value).all()
        for value in components.values()
    ):
        raise ValueError("proposal multi-query distance is invalid")
    return components


def proposal_v2_route_admitted(
    component_ranks: dict[str, int], composite_rank: int, pool: int,
) -> bool:
    """Whether a row survives the frozen deterministic route-union quotas."""
    if pool not in PROPOSAL_POOLS:
        raise ValueError(f"unsupported proposal pool: {pool}")
    if set(component_ranks) != set(PROPOSAL_ROUTES):
        raise ValueError("component ranks do not match frozen proposal routes")
    if composite_rank < 1 or any(rank < 1 for rank in component_ranks.values()):
        raise ValueError("proposal ranks are one-based positive integers")
    quotas = PROPOSAL_POOLS[pool]
    return (
        composite_rank <= quotas["composite"]
        or min(component_ranks.values()) <= quotas["per_component"]
    )


def proposal_v2_storage_bytes(dimensions: int, dtype: str) -> int:
    """Unaligned signature bytes: numeric fields plus packed presence bits."""
    if dimensions not in LAYOUTS:
        raise ValueError("unsupported proposal layout")
    itemsize = np.dtype(dtype).itemsize
    if np.dtype(dtype).kind != "f":
        raise ValueError("proposal dtype must be floating point")
    mask_bytes = (len(SAMPLES_48_NAMES) + 7) // 8
    return dimensions * itemsize + mask_bytes
