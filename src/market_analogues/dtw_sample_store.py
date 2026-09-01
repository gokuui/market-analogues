from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np

from .dtw_interval_bound import (
    DTW_CHANNELS,
    SAMPLES,
    QuantizedDtwSamples,
    dtw_interval_bound_contract,
    quantize_dtw_samples,
    quantized_dtw_lower_bounds,
    quantized_dtw_orders,
    validate_quantized_dtw_samples,
)
from .representation import Representation
from .types import stable_hash


DTW_SAMPLE_STORE_SCHEMA = "m04r-dtw-sample-store-v1"
_BASE_DTYPE = np.dtype([
    ("centers", "<f2", (len(DTW_CHANNELS), SAMPLES)),
    ("orders", "u1", (len(DTW_CHANNELS), SAMPLES)),
    ("radii", "<f4", (len(DTW_CHANNELS),)),
    ("presence", "u1"),
], align=False)
DTW_SAMPLE_ROW_BYTES = 788
if _BASE_DTYPE.itemsize > DTW_SAMPLE_ROW_BYTES:
    raise RuntimeError("DTW sample fields exceed the aligned row size")
DTW_SAMPLE_DTYPE = np.dtype([
    *_BASE_DTYPE.descr,
    ("padding", f"V{DTW_SAMPLE_ROW_BYTES - _BASE_DTYPE.itemsize}"),
], align=False)


class DtwSampleStoreError(ValueError):
    pass


@dataclass(frozen=True)
class LoadedDtwSampleGeneration:
    root: Path
    generation_id: str
    manifest: dict[str, Any]
    rows: np.ndarray
    overflow: np.ndarray


def dtw_sample_store_contract() -> dict[str, Any]:
    fields = {}
    for descriptor in DTW_SAMPLE_DTYPE.descr:
        name, storage = descriptor[:2]
        fields[name] = {
            "offset": int(DTW_SAMPLE_DTYPE.fields[name][1]), "storage": storage,
            "shape": list(descriptor[2]) if len(descriptor) == 3 else [],
        }
    state: dict[str, Any] = {
        "schema_version": DTW_SAMPLE_STORE_SCHEMA,
        "bound_contract_digest": dtw_interval_bound_contract()["digest"],
        "row_bytes": DTW_SAMPLE_DTYPE.itemsize, "row_fields": fields,
        "alignment": (
            "main and overflow arrays align one-for-one by physical row index "
            "with one immutable packed-bound generation"
        ),
        "order": "stable ascending uint8 permutation of each temporal center channel",
        "presence": "low four little-endian bits correspond to bound channels",
        "invalid_quantization": "all channels absent, yielding the safe zero DTW bound",
        "outcomes_or_labels_used": False,
    }
    return {**state, "digest": stable_hash(state)}


def make_dtw_sample_record(representation: Representation) -> np.ndarray:
    return make_dtw_sample_record_from_quantized(quantize_dtw_samples(representation))


def make_dtw_sample_record_from_quantized(
    quantized: QuantizedDtwSamples,
) -> np.ndarray:
    validate_quantized_dtw_samples(quantized)
    output = np.zeros(1, dtype=DTW_SAMPLE_DTYPE)
    output["centers"][0] = quantized.centers
    output["orders"][0] = quantized_dtw_orders(quantized)
    output["radii"][0] = quantized.channel_error_radii
    output["presence"][0] = np.packbits(
        quantized.presence.astype(np.uint8), bitorder="little",
    )[0]
    return output


def make_zero_dtw_sample_record() -> np.ndarray:
    output = np.zeros(1, dtype=DTW_SAMPLE_DTYPE)
    output["orders"][0] = np.arange(SAMPLES, dtype=np.uint8)
    return output


def _presence(records: np.ndarray) -> np.ndarray:
    return np.unpackbits(
        np.asarray(records["presence"], dtype=np.uint8)[:, None],
        axis=1, bitorder="little",
    )[:, :len(DTW_CHANNELS)].astype(bool)


def validate_dtw_sample_records(records: np.ndarray, *, block_rows: int = 16_384) -> None:
    if records.dtype != DTW_SAMPLE_DTYPE or block_rows < 1:
        raise DtwSampleStoreError("DTW sample record dtype/block differs")
    expected = np.arange(SAMPLES, dtype=np.uint8)
    for first in range(0, len(records), block_rows):
        block = records[first:first + block_rows]
        centers = np.asarray(block["centers"])
        radii = np.asarray(block["radii"])
        orders = np.asarray(block["orders"])
        presence = _presence(block)
        if not np.isfinite(centers).all() or not np.isfinite(radii).all() \
                or np.any(radii < 0) \
                or np.any(np.sort(orders, axis=2) != expected):
            raise DtwSampleStoreError("DTW sample record values differ")
        ordered = np.take_along_axis(centers, orders.astype(np.int64), axis=2)
        if np.any(ordered[:, :, 1:] < ordered[:, :, :-1]):
            raise DtwSampleStoreError("DTW sample order does not sort centers")
        absent = ~presence
        if np.any(centers[absent] != 0) or np.any(radii[absent] != 0):
            raise DtwSampleStoreError("absent DTW channel contains values")


def dtw_sample_lower_bounds(query: Representation, records: np.ndarray) -> np.ndarray:
    validate_dtw_sample_records(records)
    return quantized_dtw_lower_bounds(
        query, records["centers"], records["orders"], records["radii"],
        _presence(records),
    )


def _sha(path: Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("w") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def _packed_binding(manifest: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "manifest_digest", "rows_sha256", "overflow_sha256", "row_count",
        "overflow_count", "provenance_digest",
    )
    if any(key not in manifest for key in required):
        raise DtwSampleStoreError("packed generation binding is incomplete")
    return {key: manifest[key] for key in required}


def write_dtw_sample_generation_from_shards(
    root: Path, row_shards: Sequence[Path], overflow_shards: Sequence[Path],
    *, packed_manifest: Mapping[str, Any], provenance: Mapping[str, Any],
) -> str:
    if len(row_shards) != len(overflow_shards):
        raise DtwSampleStoreError("DTW main/overflow shard lists differ")
    packed = _packed_binding(packed_manifest)
    build = root / ".building" / uuid4().hex
    build.mkdir(parents=True, exist_ok=False)
    counts = {"rows": 0, "overflow": 0}
    try:
        for filename, shards, label in (
            ("dtw-samples.bin", row_shards, "rows"),
            ("dtw-overflow-samples.bin", overflow_shards, "overflow"),
        ):
            destination = build / filename
            with destination.open("wb") as output:
                for path in shards:
                    if not path.is_file() or path.stat().st_size % DTW_SAMPLE_DTYPE.itemsize:
                        raise DtwSampleStoreError("DTW shard byte size differs")
                    shard = np.fromfile(path, dtype=DTW_SAMPLE_DTYPE)
                    validate_dtw_sample_records(shard)
                    counts[label] += len(shard)
                    with path.open("rb") as source:
                        shutil.copyfileobj(source, output, 8 * 1024 * 1024)
                output.flush(); os.fsync(output.fileno())
        if counts["rows"] != int(packed["row_count"]) \
                or counts["overflow"] != int(packed["overflow_count"]):
            raise DtwSampleStoreError("DTW/packed row alignment differs")
        row_path = build / "dtw-samples.bin"
        overflow_path = build / "dtw-overflow-samples.bin"
        deterministic = {
            "schema_version": DTW_SAMPLE_STORE_SCHEMA,
            "contract_digest": dtw_sample_store_contract()["digest"],
            "packed_generation": packed,
            "row_count": counts["rows"], "overflow_count": counts["overflow"],
            "row_bytes": DTW_SAMPLE_DTYPE.itemsize,
            "rows_file": row_path.name, "overflow_file": overflow_path.name,
            "rows_bytes": row_path.stat().st_size,
            "overflow_bytes": overflow_path.stat().st_size,
            "rows_sha256": _sha(row_path), "overflow_sha256": _sha(overflow_path),
            "provenance": dict(provenance),
            "provenance_digest": stable_hash(dict(provenance)),
            "real_forward_outcomes_accessed": False,
        }
        generation_id = stable_hash(deterministic)
        _atomic_json(build / "manifest.json", {
            **deterministic, "manifest_digest": generation_id,
        })
        final = root / "generations" / generation_id
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            shutil.rmtree(build)
        else:
            os.replace(build, final)
        load_dtw_sample_generation(root, generation_id, packed_manifest=packed_manifest)
        return generation_id
    except Exception:
        if build.exists():
            shutil.rmtree(build)
        raise


def load_dtw_sample_generation(
    root: Path, generation_id: str, *, packed_manifest: Mapping[str, Any],
    verify_content: bool = True, validate_records: bool = True,
) -> LoadedDtwSampleGeneration:
    if not generation_id or any(value not in "0123456789abcdef" for value in generation_id):
        raise DtwSampleStoreError("DTW generation ID is unsafe")
    generation = root / "generations" / generation_id
    try:
        manifest = json.loads((generation / "manifest.json").read_text())
    except Exception as exc:
        raise DtwSampleStoreError("cannot read DTW generation manifest") from exc
    deterministic = {key: value for key, value in manifest.items()
                     if key != "manifest_digest"}
    if manifest.get("manifest_digest") != generation_id \
            or stable_hash(deterministic) != generation_id \
            or manifest.get("schema_version") != DTW_SAMPLE_STORE_SCHEMA \
            or manifest.get("contract_digest") != dtw_sample_store_contract()["digest"] \
            or manifest.get("packed_generation") != _packed_binding(packed_manifest):
        raise DtwSampleStoreError("DTW generation manifest differs")
    paths = (
        ("rows", generation / str(manifest.get("rows_file", ""))),
        ("overflow", generation / str(manifest.get("overflow_file", ""))),
    )
    arrays = []
    for label, path in paths:
        count = int(manifest["row_count" if label == "rows" else "overflow_count"])
        expected_size = int(manifest[f"{label}_bytes"])
        if not path.is_file() or path.stat().st_size != expected_size \
                or expected_size != count * DTW_SAMPLE_DTYPE.itemsize \
                or verify_content and _sha(path) != manifest[f"{label}_sha256"]:
            raise DtwSampleStoreError(f"DTW {label} file differs")
        array = (np.memmap(path, dtype=DTW_SAMPLE_DTYPE, mode="r")
                 if count else np.empty(0, dtype=DTW_SAMPLE_DTYPE))
        if validate_records:
            validate_dtw_sample_records(array)
        arrays.append(array)
    if manifest.get("provenance_digest") != stable_hash(manifest.get("provenance", {})):
        raise DtwSampleStoreError("DTW generation provenance differs")
    return LoadedDtwSampleGeneration(
        root, generation_id, manifest, arrays[0], arrays[1],
    )
