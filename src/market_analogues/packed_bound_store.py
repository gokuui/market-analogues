from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable
from uuid import uuid4

import numpy as np

from .exact_aligned_features import SAMPLES_48_NAMES
from .quantized_bound import (
    ERROR_VALUE_COUNT, PACKED_ROW_BYTES, QuantizedBoundRow,
    QuantizedLowerBoundBatch, quantized_array_lower_bounds,
    quantized_bound_contract,
)
from .representation import Representation
from .types import stable_hash


PACK_SCHEMA_VERSION = "m04r-packed-bound-store-v1"
ACTIVE_SCHEMA_VERSION = "m04r-packed-bound-active-v1"
OVERFLOW_POLICY = (
    "metadata-only exact-source fallback; routing bound is zero; no clipped or "
    "saturated quantized values"
)
TIER_CODES = {"A": 1, "B": 2}
TIER_NAMES = {value: key for key, value in TIER_CODES.items()}


class PackedBoundStoreError(ValueError):
    pass


_BASE_FIELDS: list[tuple[Any, ...]] = [
    ("episode_id", "V12"),
    ("cutoff_ns", "<i8"),
    ("symbol_id", "<u4"),
    ("quality_tier", "u1"),
    ("presence", "u1", (3,)),
    ("coarse", "<f2", (128,)),
    ("samples_48", "<f2", (len(SAMPLES_48_NAMES), 48)),
    ("stage", "<f2", (48,)),
    ("structural", "<f2", (9,)),
    ("error_radii", "<f4", (ERROR_VALUE_COUNT,)),
]
_BASE_DTYPE = np.dtype(_BASE_FIELDS, align=False)
if _BASE_DTYPE.itemsize > PACKED_ROW_BYTES:
    raise RuntimeError("quantized row fields exceed the frozen packed row size")
PACK_DTYPE = np.dtype([
    *_BASE_FIELDS,
    ("padding", f"V{PACKED_ROW_BYTES - _BASE_DTYPE.itemsize}"),
], align=False)
OVERFLOW_DTYPE = np.dtype([
    ("episode_id", "V12"), ("cutoff_ns", "<i8"),
    ("symbol_id", "<u4"), ("quality_tier", "u1"), ("padding", "V7"),
], align=False)


@dataclass(frozen=True)
class LoadedPackedGeneration:
    root: Path
    generation_id: str
    manifest: dict[str, Any]
    rows: np.ndarray
    overflow: np.ndarray
    symbols: tuple[str, ...]


def packed_bound_store_contract() -> dict[str, Any]:
    fields = {}
    for descriptor in PACK_DTYPE.descr:
        name, storage = descriptor[:2]
        fields[name] = {
            "offset": int(PACK_DTYPE.fields[name][1]),
            "storage": storage,
            "shape": list(descriptor[2]) if len(descriptor) == 3 else [],
        }
    payload: dict[str, Any] = {
        "schema_version": PACK_SCHEMA_VERSION,
        "quantized_bound_contract": quantized_bound_contract()["digest"],
        "byte_order": "explicit little-endian for multibyte scalar values",
        "row_bytes": PACK_DTYPE.itemsize,
        "row_fields": fields,
        "overflow_row_bytes": OVERFLOW_DTYPE.itemsize,
        "overflow_policy": OVERFLOW_POLICY,
        "episode_id_encoding": "24 lowercase hexadecimal characters as 12 bytes",
        "quality_tier_codes": TIER_CODES,
        "ordering": "symbol dictionary ID ascending, cutoff nanoseconds ascending",
        "generation": (
            "immutable content-addressed directory; activation is one atomic JSON "
            "pointer replacement"
        ),
        "outcomes_or_labels_used": False,
    }
    payload["digest"] = stable_hash(payload)
    return payload


def _episode_bytes(episode_id: str) -> bytes:
    if len(episode_id) != 24 or episode_id.lower() != episode_id:
        raise PackedBoundStoreError("episode ID must be 24 lowercase hex characters")
    try:
        value = bytes.fromhex(episode_id)
    except ValueError as exc:
        raise PackedBoundStoreError("episode ID is not hexadecimal") from exc
    if len(value) != 12:
        raise PackedBoundStoreError("episode ID does not encode 12 bytes")
    return value


def decode_episode_id(value: np.void | bytes) -> str:
    return bytes(value).hex()


def make_packed_record(
    episode_id: str, cutoff_ns: int, symbol_id: int, quality_tier: str,
    row: QuantizedBoundRow,
) -> np.ndarray:
    if quality_tier not in TIER_CODES:
        raise PackedBoundStoreError(f"unsupported quality tier {quality_tier!r}")
    if symbol_id < 0 or symbol_id > np.iinfo(np.uint32).max:
        raise PackedBoundStoreError("symbol ID is outside uint32")
    output = np.zeros(1, dtype=PACK_DTYPE)
    output["episode_id"][0] = np.void(_episode_bytes(episode_id))
    output["cutoff_ns"][0] = cutoff_ns
    output["symbol_id"][0] = symbol_id
    output["quality_tier"][0] = TIER_CODES[quality_tier]
    output["presence"][0] = np.packbits(
        row.presence.astype(np.uint8), bitorder="little",
    )
    output["coarse"][0] = row.coarse
    output["samples_48"][0] = row.samples_48
    output["stage"][0] = row.stage
    output["structural"][0] = row.structural
    output["error_radii"][0] = row.error_radii
    return output


def make_overflow_record(
    episode_id: str, cutoff_ns: int, symbol_id: int, quality_tier: str,
) -> np.ndarray:
    if quality_tier not in TIER_CODES:
        raise PackedBoundStoreError(f"unsupported quality tier {quality_tier!r}")
    output = np.zeros(1, dtype=OVERFLOW_DTYPE)
    output["episode_id"][0] = np.void(_episode_bytes(episode_id))
    output["cutoff_ns"][0] = cutoff_ns
    output["symbol_id"][0] = symbol_id
    output["quality_tier"][0] = TIER_CODES[quality_tier]
    return output


def unpack_quantized_rows(records: np.ndarray) -> list[QuantizedBoundRow]:
    if records.dtype != PACK_DTYPE:
        raise PackedBoundStoreError("record dtype differs from packed contract")
    output = []
    for record in records:
        presence = np.unpackbits(
            np.asarray(record["presence"], dtype=np.uint8), bitorder="little",
        )[:len(SAMPLES_48_NAMES)].astype(bool)
        output.append(QuantizedBoundRow(
            np.asarray(record["coarse"]), np.asarray(record["samples_48"]),
            presence, np.asarray(record["stage"]),
            np.asarray(record["structural"]),
            np.asarray(record["error_radii"]),
        ))
    return output


def packed_lower_bounds(
    query: Representation, records: np.ndarray,
) -> QuantizedLowerBoundBatch:
    if records.dtype != PACK_DTYPE:
        raise PackedBoundStoreError("record dtype differs from packed contract")
    presence = np.unpackbits(
        np.asarray(records["presence"], dtype=np.uint8),
        axis=1, bitorder="little",
    )[:, :len(SAMPLES_48_NAMES)].astype(bool)
    return quantized_array_lower_bounds(
        query, records["coarse"], records["samples_48"], presence,
        records["stage"], records["structural"], records["error_radii"],
    )


def _file_sha256(path: Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validate_order(records: np.ndarray, label: str) -> None:
    if not len(records):
        return
    symbols = np.asarray(records["symbol_id"], dtype=np.uint64)
    cutoffs = np.asarray(records["cutoff_ns"], dtype=np.int64)
    if np.any(symbols[1:] < symbols[:-1]) or np.any(
        (symbols[1:] == symbols[:-1]) & (cutoffs[1:] <= cutoffs[:-1])
    ):
        raise PackedBoundStoreError(
            f"{label} rows are not unique symbol/cutoff ascending"
        )


def write_packed_generation(
    root: Path,
    rows: np.ndarray,
    overflow: np.ndarray,
    symbols: Iterable[str],
    provenance: dict[str, Any],
    *,
    activate: bool = True,
) -> str:
    rows = np.asarray(rows)
    overflow = np.asarray(overflow)
    if rows.dtype != PACK_DTYPE or overflow.dtype != OVERFLOW_DTYPE:
        raise PackedBoundStoreError("generation arrays differ from frozen dtypes")
    symbol_tuple = tuple(str(value) for value in symbols)
    if len(symbol_tuple) != len(set(symbol_tuple)):
        raise PackedBoundStoreError("symbol dictionary contains duplicates")
    _validate_order(rows, "main")
    _validate_order(overflow, "overflow")
    for array, label in ((rows, "main"), (overflow, "overflow")):
        if len(array) and int(np.max(array["symbol_id"])) >= len(symbol_tuple):
            raise PackedBoundStoreError(f"{label} symbol ID exceeds dictionary")
        if len(array) and not set(np.unique(array["quality_tier"])).issubset(TIER_NAMES):
            raise PackedBoundStoreError(f"{label} contains unsupported quality tier")
    ids = [decode_episode_id(value) for value in rows["episode_id"]]
    ids.extend(decode_episode_id(value) for value in overflow["episode_id"])
    if len(ids) != len(set(ids)):
        raise PackedBoundStoreError("generation contains duplicate episode IDs")

    build_root = root / ".building" / uuid4().hex
    build_root.mkdir(parents=True, exist_ok=False)
    try:
        row_path = build_root / "bound-rows.bin"
        overflow_path = build_root / "overflow-exact-fallback.bin"
        rows.tofile(row_path)
        overflow.tofile(overflow_path)
        for path in (row_path, overflow_path):
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        deterministic = {
            "schema_version": PACK_SCHEMA_VERSION,
            "pack_contract_digest": packed_bound_store_contract()["digest"],
            "quantized_bound_contract_digest": quantized_bound_contract()["digest"],
            "row_count": len(rows),
            "overflow_count": len(overflow),
            "eligible_row_count": len(rows) + len(overflow),
            "row_bytes": PACK_DTYPE.itemsize,
            "overflow_row_bytes": OVERFLOW_DTYPE.itemsize,
            "rows_file": row_path.name,
            "overflow_file": overflow_path.name,
            "rows_bytes": row_path.stat().st_size,
            "overflow_bytes": overflow_path.stat().st_size,
            "rows_sha256": _file_sha256(row_path),
            "overflow_sha256": _file_sha256(overflow_path),
            "symbols": list(symbol_tuple),
            "provenance": provenance,
            "provenance_digest": stable_hash(provenance),
            "overflow_policy": OVERFLOW_POLICY,
            "real_forward_outcomes_accessed": False,
        }
        generation_id = stable_hash(deterministic)
        manifest = {**deterministic, "manifest_digest": generation_id}
        _atomic_json(build_root / "manifest.json", manifest)
        generations = root / "generations"
        generations.mkdir(parents=True, exist_ok=True)
        final = generations / generation_id
        if final.exists():
            shutil.rmtree(build_root)
            load_packed_generation(root, generation_id)
        else:
            os.replace(build_root, final)
        if activate:
            activate_packed_generation(root, generation_id)
        return generation_id
    except Exception:
        if build_root.exists():
            shutil.rmtree(build_root)
        raise


def _load_manifest(root: Path, generation_id: str) -> tuple[Path, dict[str, Any]]:
    if not generation_id or any(value not in "0123456789abcdef" for value in generation_id):
        raise PackedBoundStoreError("generation ID is unsafe")
    generation = root / "generations" / generation_id
    manifest_path = generation / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        raise PackedBoundStoreError(f"cannot read generation manifest: {exc}") from exc
    deterministic = {
        key: value for key, value in manifest.items() if key != "manifest_digest"
    }
    if stable_hash(deterministic) != generation_id:
        raise PackedBoundStoreError("manifest digest mismatch")
    if manifest.get("manifest_digest") != generation_id:
        raise PackedBoundStoreError("manifest self-digest mismatch")
    return generation, manifest


def load_packed_generation(
    root: Path, generation_id: str | None = None,
    *, expected_provenance_digest: str | None = None,
) -> LoadedPackedGeneration:
    if generation_id is None:
        active_path = root / "active.json"
        try:
            active = json.loads(active_path.read_text())
        except Exception as exc:
            raise PackedBoundStoreError(f"cannot read active generation: {exc}") from exc
        deterministic = {key: value for key, value in active.items() if key != "active_digest"}
        if active.get("schema_version") != ACTIVE_SCHEMA_VERSION or (
            active.get("active_digest") != stable_hash(deterministic)
        ):
            raise PackedBoundStoreError("active pointer digest mismatch")
        generation_id = str(active.get("generation_id", ""))
    generation, manifest = _load_manifest(root, generation_id)
    if manifest.get("schema_version") != PACK_SCHEMA_VERSION:
        raise PackedBoundStoreError("packed generation schema differs")
    if manifest.get("pack_contract_digest") != packed_bound_store_contract()["digest"]:
        raise PackedBoundStoreError("packed generation contract differs")
    if manifest.get("quantized_bound_contract_digest") != quantized_bound_contract()["digest"]:
        raise PackedBoundStoreError("packed quantized-bound contract differs")
    if manifest.get("overflow_policy") != OVERFLOW_POLICY:
        raise PackedBoundStoreError("packed overflow policy differs")
    if (
        int(manifest.get("row_bytes", -1)) != PACK_DTYPE.itemsize
        or int(manifest.get("overflow_row_bytes", -1)) != OVERFLOW_DTYPE.itemsize
    ):
        raise PackedBoundStoreError("packed row layout differs")
    provenance_digest = stable_hash(manifest.get("provenance", {}))
    if manifest.get("provenance_digest") != provenance_digest:
        raise PackedBoundStoreError("packed generation provenance digest differs")
    if (
        expected_provenance_digest is not None
        and provenance_digest != expected_provenance_digest
    ):
        raise PackedBoundStoreError("packed generation source provenance is stale")
    row_path = generation / str(manifest.get("rows_file", ""))
    overflow_path = generation / str(manifest.get("overflow_file", ""))
    expected = (
        (row_path, int(manifest.get("rows_bytes", -1)), str(manifest.get("rows_sha256", ""))),
        (overflow_path, int(manifest.get("overflow_bytes", -1)), str(manifest.get("overflow_sha256", ""))),
    )
    for path, size, digest in expected:
        if not path.is_file() or path.stat().st_size != size:
            raise PackedBoundStoreError(f"packed file size differs: {path.name}")
        if _file_sha256(path) != digest:
            raise PackedBoundStoreError(f"packed file digest differs: {path.name}")
    if row_path.stat().st_size != int(manifest["row_count"]) * PACK_DTYPE.itemsize:
        raise PackedBoundStoreError("main packed row count differs")
    if overflow_path.stat().st_size != int(manifest["overflow_count"]) * OVERFLOW_DTYPE.itemsize:
        raise PackedBoundStoreError("overflow packed row count differs")
    rows = (
        np.memmap(row_path, dtype=PACK_DTYPE, mode="r")
        if row_path.stat().st_size else np.empty(0, dtype=PACK_DTYPE)
    )
    overflow = (
        np.memmap(overflow_path, dtype=OVERFLOW_DTYPE, mode="r")
        if overflow_path.stat().st_size else np.empty(0, dtype=OVERFLOW_DTYPE)
    )
    symbols = tuple(str(value) for value in manifest.get("symbols", []))
    if len(symbols) != len(set(symbols)):
        raise PackedBoundStoreError("packed symbol dictionary is invalid")
    _validate_order(rows, "main")
    _validate_order(overflow, "overflow")
    for array, label in ((rows, "main"), (overflow, "overflow")):
        if len(array) and int(np.max(array["symbol_id"])) >= len(symbols):
            raise PackedBoundStoreError(f"{label} symbol ID exceeds dictionary")
        if len(array) and not set(np.unique(array["quality_tier"])).issubset(TIER_NAMES):
            raise PackedBoundStoreError(f"{label} quality tier differs")
    ids = [decode_episode_id(value) for value in rows["episode_id"]]
    ids.extend(decode_episode_id(value) for value in overflow["episode_id"])
    if len(ids) != len(set(ids)):
        raise PackedBoundStoreError("packed generation contains duplicate episode IDs")
    return LoadedPackedGeneration(root, generation_id, manifest, rows, overflow, symbols)


def activate_packed_generation(root: Path, generation_id: str) -> Path:
    load_packed_generation(root, generation_id)
    deterministic = {
        "schema_version": ACTIVE_SCHEMA_VERSION,
        "generation_id": generation_id,
    }
    active = {**deterministic, "active_digest": stable_hash(deterministic)}
    path = root / "active.json"
    _atomic_json(path, active)
    return path
