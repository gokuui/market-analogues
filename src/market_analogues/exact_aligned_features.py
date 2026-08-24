from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .distance import GROUPS
from .exact_batch import sliding_exact_representations
from .representation import Representation, represent
from .types import Episode, stable_hash


EXACT_FEATURE_KERNEL_VERSION = "exact-aligned-feature-kernel-v1"
SAMPLES_48_NAMES = tuple(sorted(name for names in GROUPS.values() for name in names))
SAMPLES_64_NAMES = ("atr_pct", "close_path", "relative_path", "volume_robust_z")
PRESENCE_NAMES = tuple(
    [f"samples_48:{name}" for name in SAMPLES_48_NAMES]
    + [f"samples_64:{name}" for name in SAMPLES_64_NAMES]
)


def _layout() -> dict[str, tuple[int, int]]:
    offset = 0
    layout: dict[str, tuple[int, int]] = {}
    fields = [("coarse", 128)]
    fields += [(f"samples_48:{name}", 48) for name in SAMPLES_48_NAMES]
    fields += [(f"samples_64:{name}", 64) for name in SAMPLES_64_NAMES]
    fields += [("stage", 48), ("structural", 9), ("presence", len(PRESENCE_NAMES))]
    for name, width in fields:
        layout[name] = (offset, offset + width)
        offset += width
    return layout


EXACT_FEATURE_LAYOUT = _layout()
EXACT_FEATURE_DIMENSIONS = EXACT_FEATURE_LAYOUT["presence"][1]


def exact_feature_contract() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": EXACT_FEATURE_KERNEL_VERSION,
        "representation_version": "dense-v1",
        "dimensions": EXACT_FEATURE_DIMENSIONS,
        "dtype": "float64",
        "layout": EXACT_FEATURE_LAYOUT,
        "samples_48_names": SAMPLES_48_NAMES,
        "samples_64_names": SAMPLES_64_NAMES,
        "presence_names": PRESENCE_NAMES,
        "missing_encoding": "zero-filled values plus one explicit presence bit per optional field",
        "source": (
            "the same materialize_exact_representations output used by scalar queries "
            "and sliding exact candidates"
        ),
        "lossless_for_distance_v1_fields": True,
        "legacy_view_signature_equivalent": False,
        "proposal_status": (
            "exact-aligned source row for M04R-05 layout experiments; not a compact "
            "production proposal and not a certified bound"
        ),
        "outcomes_or_labels_used": False,
    }
    payload["digest"] = stable_hash(payload)
    return payload


@dataclass(frozen=True)
class ExactAlignedFeatures:
    vector: np.ndarray

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector, dtype=np.float64)
        if vector.shape != (EXACT_FEATURE_DIMENSIONS,):
            raise ValueError(
                f"exact feature row must have {EXACT_FEATURE_DIMENSIONS} values; "
                f"got {vector.shape}"
            )
        if not np.isfinite(vector).all():
            raise ValueError("exact feature row contains non-finite values")
        presence = vector[slice(*EXACT_FEATURE_LAYOUT["presence"])]
        if not np.isin(presence, (0.0, 1.0)).all():
            raise ValueError("exact feature presence values must be zero or one")
        object.__setattr__(self, "vector", vector)

    @property
    def digest(self) -> str:
        return stable_hash({
            "version": EXACT_FEATURE_KERNEL_VERSION,
            "dtype": self.vector.dtype.str,
            "shape": self.vector.shape,
            "bytes": self.vector.tobytes().hex(),
        })


def representation_to_exact_features(
    representation: Representation,
) -> ExactAlignedFeatures:
    values: list[np.ndarray] = [np.asarray(representation.coarse, dtype=np.float64)]
    presence: list[float] = []
    for collection_name, names in (
        ("samples_48", SAMPLES_48_NAMES),
        ("samples_64", SAMPLES_64_NAMES),
    ):
        collection = getattr(representation, collection_name)
        width = 48 if collection_name == "samples_48" else 64
        for name in names:
            sample = collection.get(name)
            presence.append(float(sample is not None))
            values.append(
                np.zeros(width, dtype=np.float64)
                if sample is None else np.asarray(sample, dtype=np.float64)
            )
    values.extend((
        np.asarray(representation.stage, dtype=np.float64),
        np.asarray(representation.structural, dtype=np.float64),
        np.asarray(presence, dtype=np.float64),
    ))
    return ExactAlignedFeatures(np.concatenate(values))


def exact_features_to_representation(features: ExactAlignedFeatures) -> Representation:
    vector = features.vector

    def part(name: str) -> np.ndarray:
        return vector[slice(*EXACT_FEATURE_LAYOUT[name])].copy()

    presence = part("presence").astype(bool)
    presence_by_name = dict(zip(PRESENCE_NAMES, presence))
    samples_48 = {
        name: (
            part(f"samples_48:{name}")
            if presence_by_name[f"samples_48:{name}"] else None
        )
        for name in SAMPLES_48_NAMES
    }
    samples_64 = {
        name: (
            part(f"samples_64:{name}")
            if presence_by_name[f"samples_64:{name}"] else None
        )
        for name in SAMPLES_64_NAMES
    }
    return Representation(
        pd.DataFrame(), part("coarse").astype(np.float32),
        samples_48, samples_64, part("stage"), part("structural"),
    )


def episode_exact_features(episode: Episode) -> ExactAlignedFeatures:
    return representation_to_exact_features(represent(episode))


def sliding_exact_features(
    bars: pd.DataFrame,
    benchmark: pd.DataFrame | None,
    *,
    lookback: int,
    stride: int = 5,
    batch_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    batch = sliding_exact_representations(
        bars, benchmark, lookback=lookback, stride=stride, batch_size=batch_size,
    )
    if not batch.representations:
        return batch.positions, np.empty((0, EXACT_FEATURE_DIMENSIONS), dtype=np.float64)
    matrix = np.vstack([
        representation_to_exact_features(item).vector
        for item in batch.representations
    ])
    return batch.positions, matrix
