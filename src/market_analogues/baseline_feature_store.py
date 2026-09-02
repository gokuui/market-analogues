from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .baseline_neighbors import (
    baseline_neighbor_contract,
    recent_return_volatility_at_positions,
)
from .types import stable_hash


FEATURE_DTYPE = np.dtype([("values", "<f8", (3,))], align=False)
SCHEMA_VERSION = "wf03-return-volatility-feature-store-v1"


class BaselineFeatureStoreError(ValueError):
    pass


@dataclass(frozen=True)
class BaselineFeatureGeneration:
    generation_id: str
    manifest: dict[str, Any]
    rows: np.ndarray
    overflow: np.ndarray


def baseline_feature_store_contract() -> dict[str, Any]:
    state = {
        "schema_version": SCHEMA_VERSION,
        "row_bytes": FEATURE_DTYPE.itemsize,
        "dtype": FEATURE_DTYPE.descr,
        "features": ["return_20", "return_63", "log_return_volatility_20"],
        "invalid": "all three binary64 values are NaN; partial/non-finite rows forbidden",
        "alignment": "one physical row per bound-store main and overflow row",
        "neighbor_contract_digest": baseline_neighbor_contract()["digest"],
        "outcomes_or_labels_used": False,
    }
    return {**state, "digest": stable_hash(state)}


def make_feature_record(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,) or not (np.isfinite(array).all() or np.isnan(array).all()):
        raise BaselineFeatureStoreError("feature row must be finite or wholly missing")
    output = np.empty(1, dtype=FEATURE_DTYPE)
    output["values"][0] = array
    return output


def validate_feature_records(records: np.ndarray) -> None:
    if records.dtype != FEATURE_DTYPE or records.ndim != 1:
        raise BaselineFeatureStoreError("feature record dtype differs")
    values = records["values"]
    valid = np.isfinite(values).all(axis=1)
    missing = np.isnan(values).all(axis=1)
    if not np.all(valid | missing):
        raise BaselineFeatureStoreError("feature record finiteness differs")


def features_for_packed_records(
    frame: pd.DataFrame, records: np.ndarray,
) -> np.ndarray:
    fields = records.dtype.fields
    if "timestamp" not in frame or "close" not in frame \
            or records.ndim != 1 or fields is None or "cutoff_ns" not in fields:
        raise BaselineFeatureStoreError("feature build inputs differ")
    output = np.empty(len(records), dtype=FEATURE_DTYPE)
    output["values"] = np.nan
    if not len(records):
        return output
    timestamps = np.ascontiguousarray(
        frame["timestamp"].to_numpy(dtype="datetime64[ns]").view(np.int64)
    )
    if len(timestamps) != len(frame) \
            or len(timestamps) > 1 and np.any(timestamps[1:] < timestamps[:-1]):
        raise BaselineFeatureStoreError("feature source timestamps are not ordered")
    cutoffs = np.asarray(records["cutoff_ns"], dtype=np.int64)
    positions = np.searchsorted(timestamps, cutoffs, side="right") - 1
    found = positions >= 0
    valid_positions = np.flatnonzero(found)
    found[valid_positions] = (
        timestamps[positions[valid_positions]] == cutoffs[valid_positions]
    )
    if not np.all(found):
        raise BaselineFeatureStoreError("packed feature cutoff is absent from source")
    buildable = positions >= 63
    if np.any(buildable):
        close = frame["close"].to_numpy(dtype=np.float64)
        output["values"][buildable] = recent_return_volatility_at_positions(
            close, positions[buildable].astype(np.int64),
        )
    validate_feature_records(output)
    return output


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".feature-", dir=path.parent)
    temp = Path(temporary)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _concatenate(paths: Iterable[Path], target: Path) -> int:
    count = 0
    with target.open("wb") as output:
        for path in paths:
            size = path.stat().st_size
            if size % FEATURE_DTYPE.itemsize:
                raise BaselineFeatureStoreError("feature shard size differs")
            count += size // FEATURE_DTYPE.itemsize
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1 << 20), b""):
                    output.write(block)
        output.flush(); os.fsync(output.fileno())
    return count


def _mapped(path: Path, count: int) -> np.ndarray:
    return (np.memmap(path, dtype=FEATURE_DTYPE, mode="r")
            if count else np.empty(0, dtype=FEATURE_DTYPE))


def write_feature_generation_from_shards(
    root: Path,
    row_shards: Iterable[Path],
    overflow_shards: Iterable[Path],
    *,
    packed_manifest: dict[str, Any],
    provenance: dict[str, Any],
) -> str:
    root.mkdir(parents=True, exist_ok=True)
    building = Path(tempfile.mkdtemp(prefix="feature-generation-", dir=root))
    try:
        rows_path = building / "features.bin"
        overflow_path = building / "overflow-features.bin"
        row_count = _concatenate(row_shards, rows_path)
        overflow_count = _concatenate(overflow_shards, overflow_path)
        if row_count != int(packed_manifest["row_count"]) \
                or overflow_count != int(packed_manifest["overflow_count"]):
            raise BaselineFeatureStoreError("feature/packed row alignment differs")
        state = {
            "schema_version": SCHEMA_VERSION,
            "contract_digest": baseline_feature_store_contract()["digest"],
            "packed_generation_id": packed_manifest["manifest_digest"],
            "packed_provenance_digest": packed_manifest["provenance_digest"],
            "row_count": row_count, "overflow_count": overflow_count,
            "row_bytes": FEATURE_DTYPE.itemsize,
            "rows_file": rows_path.name, "overflow_file": overflow_path.name,
            "rows_sha256": _sha(rows_path), "overflow_sha256": _sha(overflow_path),
            "provenance": provenance,
            "provenance_digest": stable_hash(provenance),
            "outcomes_or_labels_used": False,
        }
        generation_id = stable_hash(state)
        manifest = {**state, "manifest_digest": generation_id}
        _atomic_json(building / "manifest.json", manifest)
        validate_feature_records(_mapped(rows_path, row_count))
        validate_feature_records(_mapped(overflow_path, overflow_count))
        destination = root / "generations" / generation_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise BaselineFeatureStoreError("feature generation already exists")
        os.replace(building, destination)
        return generation_id
    finally:
        if building.exists():
            for path in building.iterdir(): path.unlink()
            building.rmdir()


def load_feature_generation(
    root: Path, generation_id: str, *, packed_manifest: dict[str, Any],
    verify_content: bool = True,
) -> BaselineFeatureGeneration:
    directory = root / "generations" / generation_id
    manifest = json.loads((directory / "manifest.json").read_text())
    state = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    if not all((
        manifest.get("manifest_digest") == generation_id == stable_hash(state),
        manifest.get("schema_version") == SCHEMA_VERSION,
        manifest.get("contract_digest") == baseline_feature_store_contract()["digest"],
        manifest.get("packed_generation_id") == packed_manifest["manifest_digest"],
        manifest.get("packed_provenance_digest") == packed_manifest["provenance_digest"],
        manifest.get("row_count") == packed_manifest["row_count"],
        manifest.get("overflow_count") == packed_manifest["overflow_count"],
        manifest.get("provenance_digest") == stable_hash(manifest.get("provenance")),
    )):
        raise BaselineFeatureStoreError("feature generation manifest differs")
    rows_path = directory / manifest["rows_file"]
    overflow_path = directory / manifest["overflow_file"]
    if rows_path.stat().st_size != manifest["row_count"] * FEATURE_DTYPE.itemsize \
            or overflow_path.stat().st_size != manifest["overflow_count"] * FEATURE_DTYPE.itemsize:
        raise BaselineFeatureStoreError("feature generation size differs")
    if verify_content and (_sha(rows_path) != manifest["rows_sha256"]
                           or _sha(overflow_path) != manifest["overflow_sha256"]):
        raise BaselineFeatureStoreError("feature generation content differs")
    rows = _mapped(rows_path, int(manifest["row_count"]))
    overflow = _mapped(overflow_path, int(manifest["overflow_count"]))
    validate_feature_records(rows); validate_feature_records(overflow)
    return BaselineFeatureGeneration(generation_id, manifest, rows, overflow)
