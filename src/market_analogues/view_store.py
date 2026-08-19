from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from time import perf_counter

import numpy as np
import pandas as pd

from .adapters import OHLCVSource
from .scan import _quality_issues
from .types import EpisodeKey, InstrumentKey
from .view_signatures import (
    SIGNATURE_DIMENSIONS, VIEW_SIGNATURE_VERSION, sliding_episode_signatures,
)


VIEW_SHARD_SCHEMA_VERSION = 3


class ViewShardError(ValueError):
    pass


@dataclass(frozen=True)
class ViewShardMetadata:
    schema_version: int
    signature_version: str
    dataset_id: str
    symbol: str
    lookback: int
    stride: int
    representation_version: str
    source_fingerprint: str
    benchmark_fingerprint: str | None
    quality_tier: str
    rows: int
    dimensions: int
    storage_dtype: str
    content_digest: str


@dataclass(frozen=True)
class LoadedViewShard:
    path: Path
    metadata: ViewShardMetadata
    episode_ids: np.ndarray
    cutoffs_ns: np.ndarray
    signatures: np.ndarray


@dataclass(frozen=True)
class ViewStoreBuildReport:
    passed: bool
    dataset_id: str
    instruments_considered: int
    instruments_built: int
    instruments_reused: int
    quality_skipped: int
    rows: int
    failures: tuple[str, ...]
    manifest_path: Path
    manifest_digest: str
    seconds: float


def _array_digest(
    episode_ids: np.ndarray,
    cutoffs_ns: np.ndarray,
    signatures: np.ndarray,
) -> str:
    digest = sha256()
    for episode_id in episode_ids.astype(str):
        digest.update(episode_id.encode())
        digest.update(b"\0")
    digest.update(np.asarray(cutoffs_ns, dtype="<i8").tobytes(order="C"))
    digest.update(np.asarray(signatures, dtype="<f4").tobytes(order="C"))
    return digest.hexdigest()


def _safe_symbol(symbol: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", symbol).strip("._") or "symbol"
    suffix = sha256(symbol.encode()).hexdigest()[:10]
    return f"{stem[:80]}-{suffix}"


def shard_path(
    root: Path,
    instrument: InstrumentKey,
    lookback: int,
    stride: int,
) -> Path:
    return (
        root / instrument.dataset_id / f"lookback-{lookback}" / f"stride-{stride}"
        / f"{_safe_symbol(instrument.source_symbol)}.npz"
    )


def _metadata_from_json(value: str) -> ViewShardMetadata:
    try:
        return ViewShardMetadata(**json.loads(value))
    except Exception as exc:
        raise ViewShardError(f"invalid shard metadata: {exc}") from exc


def load_view_shard(path: Path) -> LoadedViewShard:
    try:
        with np.load(path, allow_pickle=False) as payload:
            required = {"metadata_json", "episode_ids", "cutoffs_ns", "signatures"}
            missing = required.difference(payload.files)
            if missing:
                raise ViewShardError(f"missing arrays: {sorted(missing)}")
            metadata = _metadata_from_json(str(payload["metadata_json"].item()))
            episode_ids = np.asarray(payload["episode_ids"]).astype(str)
            cutoffs_ns = np.asarray(payload["cutoffs_ns"], dtype=np.int64)
            raw_signatures = np.asarray(payload["signatures"])
    except ViewShardError:
        raise
    except Exception as exc:
        raise ViewShardError(f"cannot load view shard {path}: {exc}") from exc
    if metadata.schema_version != VIEW_SHARD_SCHEMA_VERSION:
        raise ViewShardError(
            f"unsupported shard schema {metadata.schema_version}; "
            f"expected {VIEW_SHARD_SCHEMA_VERSION}"
        )
    if metadata.signature_version != VIEW_SIGNATURE_VERSION:
        raise ViewShardError(
            f"stale signature version {metadata.signature_version}; "
            f"expected {VIEW_SIGNATURE_VERSION}"
        )
    if metadata.storage_dtype not in {"float16", "float32"}:
        raise ViewShardError(f"unsupported signature storage dtype {metadata.storage_dtype}")
    if str(raw_signatures.dtype) != metadata.storage_dtype:
        raise ViewShardError(
            f"signature dtype {raw_signatures.dtype} disagrees with metadata {metadata.storage_dtype}"
        )
    signatures = raw_signatures.astype(np.float32)
    if signatures.shape != (metadata.rows, SIGNATURE_DIMENSIONS):
        raise ViewShardError(
            f"signature shape {signatures.shape} disagrees with metadata rows/dimensions"
        )
    if len(episode_ids) != metadata.rows or len(cutoffs_ns) != metadata.rows:
        raise ViewShardError("metadata row count disagrees with candidate arrays")
    if len(set(episode_ids)) != len(episode_ids):
        raise ViewShardError("view shard contains duplicate episode IDs")
    if not np.isfinite(signatures).all():
        raise ViewShardError("view shard contains non-finite signatures")
    actual_digest = _array_digest(episode_ids, cutoffs_ns, signatures)
    if actual_digest != metadata.content_digest:
        raise ViewShardError(
            f"content digest mismatch: {actual_digest} != {metadata.content_digest}"
        )
    return LoadedViewShard(path, metadata, episode_ids, cutoffs_ns, signatures)


def _write_shard(
    path: Path,
    metadata: ViewShardMetadata,
    episode_ids: np.ndarray,
    cutoffs_ns: np.ndarray,
    signatures: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata_json=np.asarray(json.dumps(asdict(metadata), sort_keys=True)),
            episode_ids=episode_ids.astype("U24"),
            cutoffs_ns=cutoffs_ns.astype(np.int64),
            signatures=signatures.astype(metadata.storage_dtype),
        )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _matches_expected(
    metadata: ViewShardMetadata,
    *,
    instrument: InstrumentKey,
    lookback: int,
    stride: int,
    representation_version: str,
    source_fingerprint: str,
    benchmark_fingerprint: str | None,
    quality_tier: str,
    storage_dtype: str,
) -> bool:
    return (
        metadata.dataset_id == instrument.dataset_id
        and metadata.symbol == instrument.source_symbol
        and metadata.lookback == lookback
        and metadata.stride == stride
        and metadata.representation_version == representation_version
        and metadata.source_fingerprint == source_fingerprint
        and metadata.benchmark_fingerprint == benchmark_fingerprint
        and metadata.quality_tier == quality_tier
        and metadata.dimensions == SIGNATURE_DIMENSIONS
        and (
            metadata.storage_dtype == storage_dtype
            or storage_dtype == "float16" and metadata.storage_dtype == "float32"
        )
    )


def _resolved_storage_dtype(signatures: np.ndarray, requested: str) -> str:
    if requested == "float16" and (
        signatures.size
        and float(np.max(np.abs(signatures))) > float(np.finfo(np.float16).max)
    ):
        return "float32"
    return requested


def build_view_shard(
    source: OHLCVSource,
    instrument: InstrumentKey,
    output_root: Path,
    *,
    lookback: int,
    stride: int,
    representation_version: str,
    quality_tier: str = "A",
    quality_issues: tuple[str, ...] = (),
    rebuild_invalid: bool = False,
    benchmark: pd.DataFrame | None = None,
    benchmark_fingerprint: str | None = None,
    storage_dtype: str = "float16",
) -> tuple[LoadedViewShard, bool]:
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    if stride < 1:
        raise ValueError("stride must be positive")
    if storage_dtype not in {"float16", "float32"}:
        raise ValueError("storage_dtype must be float16 or float32")
    path = shard_path(output_root, instrument, lookback, stride)
    source_fingerprint = source.fingerprint(instrument)
    if benchmark_fingerprint is None:
        benchmark_fingerprint = source.benchmark_fingerprint()
    if path.exists():
        try:
            loaded = load_view_shard(path)
        except ViewShardError:
            if not rebuild_invalid:
                raise
        else:
            if _matches_expected(
                loaded.metadata, instrument=instrument, lookback=lookback,
                stride=stride, representation_version=representation_version,
                source_fingerprint=source_fingerprint, quality_tier=quality_tier,
                benchmark_fingerprint=benchmark_fingerprint,
                storage_dtype=storage_dtype,
            ):
                return loaded, True

    bars = source.load(instrument)
    if benchmark is None:
        benchmark = source.load_benchmark()
    positions, signature_matrix = sliding_episode_signatures(
        bars, benchmark, lookback=lookback, stride=stride,
    )
    episode_ids: list[str] = []
    cutoffs: list[int] = []
    for position in positions:
        cutoff = pd.Timestamp(bars.timestamp.iloc[position])
        key = EpisodeKey(instrument, cutoff, lookback, representation_version)
        episode_ids.append(key.id)
        cutoffs.append(int(cutoff.value))
    id_array = np.asarray(episode_ids, dtype="U24")
    cutoff_array = np.asarray(cutoffs, dtype=np.int64)
    actual_storage_dtype = _resolved_storage_dtype(signature_matrix, storage_dtype)
    if actual_storage_dtype != storage_dtype:
        # Preserve extreme-but-finite market data rather than clipping it or
        # allowing a float16 cast to create infinities. The requested float16
        # mode is a compact-preferred policy, not a lossy requirement.
        actual_storage_dtype = "float32"
    stored_signatures = signature_matrix.astype(actual_storage_dtype).astype(np.float32)
    metadata = ViewShardMetadata(
        VIEW_SHARD_SCHEMA_VERSION, VIEW_SIGNATURE_VERSION,
        instrument.dataset_id, instrument.source_symbol, lookback, stride,
        representation_version, source_fingerprint, benchmark_fingerprint,
        quality_tier,
        len(id_array), SIGNATURE_DIMENSIONS, actual_storage_dtype,
        _array_digest(id_array, cutoff_array, stored_signatures),
    )
    _write_shard(path, metadata, id_array, cutoff_array, stored_signatures)
    return load_view_shard(path), False


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def build_view_store(
    source: OHLCVSource,
    quality: pd.DataFrame,
    output_root: Path,
    *,
    lookbacks: tuple[int, ...] = (63, 126, 252),
    stride: int = 5,
    representation_version: str = "dense-v1",
    workers: int = 1,
    instrument_limit: int | None = None,
    rebuild_invalid: bool = False,
    storage_dtype: str = "float16",
) -> ViewStoreBuildReport:
    if workers < 1:
        raise ValueError("workers must be positive")
    started = perf_counter()
    qmap = {str(row.symbol): row for row in quality.itertuples(index=False)}
    benchmark = source.load_benchmark()
    benchmark_fingerprint = source.benchmark_fingerprint()
    instruments = source.instruments()
    if instrument_limit is not None:
        instruments = instruments[:instrument_limit]
    jobs: list[tuple[InstrumentKey, int, object | None]] = []
    quality_skipped = 0
    for instrument in instruments:
        record = qmap.get(instrument.source_symbol)
        tier = str(record.tier) if record is not None else "A"
        if tier not in {"A", "B"}:
            quality_skipped += 1
            continue
        jobs.extend((instrument, lookback, record) for lookback in sorted(set(lookbacks)))

    def run(job: tuple[InstrumentKey, int, object | None]):
        instrument, lookback, record = job
        tier = str(record.tier) if record is not None else "A"
        try:
            shard, reused = build_view_shard(
                source, instrument, output_root, lookback=lookback, stride=stride,
                representation_version=representation_version, quality_tier=tier,
                quality_issues=_quality_issues(record), rebuild_invalid=rebuild_invalid,
                benchmark=benchmark,
                benchmark_fingerprint=benchmark_fingerprint,
                storage_dtype=storage_dtype,
            )
            return shard, reused, None
        except Exception as exc:
            return None, False, f"{instrument}:{lookback}:{type(exc).__name__}:{exc}"

    if workers == 1:
        results = map(run, jobs)
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        results = executor.map(run, jobs)
    shards: list[LoadedViewShard] = []
    failures: list[str] = []
    built = reused_count = 0
    try:
        for shard, reused, failure in results:
            if failure:
                failures.append(failure)
                continue
            assert shard is not None
            shards.append(shard)
            if reused:
                reused_count += 1
            else:
                built += 1
    finally:
        if workers != 1:
            executor.shutdown(wait=True)

    current_records = [
        {"path": str(shard.path.relative_to(output_root)), **asdict(shard.metadata)}
        for shard in sorted(shards, key=lambda item: str(item.path))
    ]
    dataset_id = source.instruments()[0].dataset_id if source.instruments() else "unknown"
    manifest = output_root / dataset_id / "manifest.json"
    retained_records: list[dict[str, object]] = []
    if manifest.exists():
        try:
            previous = json.loads(manifest.read_text())
        except Exception as exc:
            failures.append(f"existing manifest is invalid: {type(exc).__name__}:{exc}")
        else:
            compatible = (
                previous.get("schema_version") == VIEW_SHARD_SCHEMA_VERSION
                and previous.get("signature_version") == VIEW_SIGNATURE_VERSION
                and previous.get("representation_version") == representation_version
                and previous.get("benchmark_fingerprint") == benchmark_fingerprint
                and previous.get("stride") == stride
                and previous.get("storage_dtype") == storage_dtype
            )
            if compatible and isinstance(previous.get("shards"), list):
                replaced = set(lookbacks)
                retained_records = [
                    record for record in previous["shards"]
                    if int(record["lookback"]) not in replaced
                ]
    shard_records = sorted(
        retained_records + current_records,
        key=lambda record: (int(record["lookback"]), str(record["symbol"]), str(record["path"])),
    )
    digest_payload = json.dumps(shard_records, sort_keys=True, separators=(",", ":"))
    manifest_digest = sha256(digest_payload.encode()).hexdigest()
    _write_manifest(manifest, {
        "schema_version": VIEW_SHARD_SCHEMA_VERSION,
        "signature_version": VIEW_SIGNATURE_VERSION,
        "dataset_id": dataset_id,
        "lookbacks": sorted({int(record["lookback"]) for record in shard_records}),
        "stride": stride,
        "representation_version": representation_version,
        "benchmark_fingerprint": benchmark_fingerprint,
        "storage_dtype": storage_dtype,
        "shards": shard_records,
        "manifest_digest": manifest_digest,
        "failures": failures,
    })
    return ViewStoreBuildReport(
        bool(shards) and not failures, dataset_id, len(instruments), built,
        reused_count, quality_skipped, sum(shard.metadata.rows for shard in shards),
        tuple(failures), manifest, manifest_digest, perf_counter() - started,
    )
