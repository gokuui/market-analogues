"""Build the preregistered, outcome-blind R1-B B2 geometry authority.

The producer is intentionally separate from the localization statistic.  It
reconstructs exact causal float64 chart vectors from the source files frozen in
the joint H1, verifies the already-sealed query transform, and publishes only
geometry and causal eligibility.  It never reads outcomes or later R1-B
diagnostics.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict
import fcntl
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import resource
import shutil
import stat
import sys
import time
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.m04r import m04r14_r1b_joint_b005_b2_contract as joint
from market_analogues.adapters import DirectorySource, canonicalize
from market_analogues.adequacy_localization import (
    DIMENSIONS, candidate_specificity_ranks, chart_distances,
    empirical_chart_transform,
)
from market_analogues.balanced_partition import (
    array_digest, id_digest, integer_array_digest, midrank_transform,
)
from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.config import DatasetSpec, load_config_bytes
from market_analogues.representation import represent, representation_input_digest
from market_analogues.types import Episode, EpisodeKey, InstrumentKey, stable_hash


SCHEMA = "m04r14-r1b-b2-geometry-v1"
IDS_SCHEMA = "m04r14-r1b-b2-geometry-ids-v1"
SHARD_SCHEMA = "m04r14-r1b-b2-geometry-distance-shard-v1"
OUTPUT = Path("config/data/analogues/m04r14/r1b-b2-geometry-v1")
WORK = Path("config/data/analogues/m04r14/.r1b-b2-geometry-v1.work")
ARRAY_NAMES = (
    "raw_queries", "raw_candidates", "transformed_queries",
    "transformed_candidates", "query_pair_distances",
    "query_candidate_distances", "specificity_ranks", "causal_eligibility",
)
FLOAT_ARRAYS = frozenset(ARRAY_NAMES) - {"causal_eligibility"}
CHUNK_ROWS = 64


class GeometryError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GeometryError(message)


def _load_json(path: Path, expected: type = dict) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=pairs,
                           parse_constant=lambda token: require(False, f"nonfinite JSON: {token}"))
    except (OSError, ValueError) as error:
        raise GeometryError(f"unreadable JSON: {path}") from error
    require(isinstance(value, expected), f"expected JSON {expected.__name__}: {path}")
    return value


def _file_sha(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"regular file required: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _bound_bytes(record: Mapping[str, Any]) -> bytes:
    """Read, authenticate and return the one byte buffer that will be decoded."""
    path = Path(str(record["path"]))
    require(path.is_absolute() and not path.is_symlink(), f"regular absolute file required: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise GeometryError(f"source open failed: {path}") from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"regular file required: {path}")
        digest = sha256(); blocks: list[bytes] = []
        while block := os.read(descriptor, 8 << 20):
            digest.update(block); blocks.append(block)
        after = os.fstat(descriptor)
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                f"source changed while reading: {path}")
        content = b"".join(blocks)
        require(len(content) == int(record["bytes"]) == before.st_size
                and digest.hexdigest() == record["sha256"], f"source bytes differ: {path}")
        return content
    except OSError as error:
        raise GeometryError(f"source read failed: {path}") from error
    finally:
        os.close(descriptor)


def _bound_frame(record: Mapping[str, Any], spec: DatasetSpec, *, symbol: str,
                 benchmark: bool = False) -> pd.DataFrame:
    content = _bound_bytes(record)
    if benchmark:
        require(spec.benchmark is not None, "benchmark specification missing")
        fmt = spec.benchmark.format or Path(str(record["path"])).suffix.lstrip(".").lower()
        frame_spec = DatasetSpec(
            dataset_id=spec.dataset_id, adapter="directory", path=spec.benchmark.path.parent,
            format=fmt, timestamp_column=spec.benchmark.timestamp_column or spec.timestamp_column,
            timezone=spec.timezone, interval=spec.interval, column_map=spec.column_map,
        )
    else:
        fmt = spec.format; frame_spec = spec
    try:
        if fmt == "parquet":
            raw = pd.read_parquet(io.BytesIO(content))
        elif fmt == "csv":
            raw = pd.read_csv(io.BytesIO(content))
        else:
            raise GeometryError(f"unsupported bound source format: {fmt}")
    except (OSError, ValueError) as error:
        raise GeometryError(f"bound source decode failed: {record['path']}") from error
    return canonicalize(raw, frame_spec, symbol=symbol)


def _validate_dataset_spec(spec: DatasetSpec, manifest: Mapping[str, Any]) -> None:
    frozen = json.loads(json.dumps(asdict(spec), default=str))
    require(frozen == manifest.get("dataset_spec"),
            "frozen NASDAQ dataset specification differs")


def _canonical_float(values: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype="<f8", order="C")
    require(array.shape == shape and np.isfinite(array).all(), f"{name} shape or finiteness differs")
    array = np.ascontiguousarray(array)
    array[array == 0.0] = 0.0
    require(not np.signbit(array[array == 0.0]).any(), f"{name} contains negative zero")
    return array


def _array_semantic_digest(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = sha256()
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid4().hex}"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise GeometryError(f"create-only path exists: {path}") from error
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid4().hex}"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush(); os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise GeometryError(f"create-only path exists: {path}") from error
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_directory(temporary: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.parent / f".{destination.name}.publish.lock"
    descriptor = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        require(not destination.exists() and not destination.is_symlink(), f"output exists: {destination}")
        current = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(current)
        finally:
            os.close(current)
        os.rename(temporary, destination)
        parent = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _manifest_records(prereg: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    manifest = prereg["authorities"]["external_source_manifest"]
    require(manifest.get("manifest_digest") == stable_hash({
        key: value for key, value in manifest.items() if key != "manifest_digest"
    }), "external source manifest digest differs")
    records = {str(row["symbol"]): row for row in manifest["stock_files"]}
    require(len(records) == manifest["stock_files_count"] == len(manifest["required_symbols"])
            and sorted(records) == manifest["required_symbols"], "external stock manifest identities differ")
    return manifest, records


def verify_external_source_manifest(prereg: Mapping[str, Any]) -> None:
    """Rehash the exact H1-bound files before any OHLCV decoding."""
    manifest, records = _manifest_records(prereg)
    for row in (manifest["config"], manifest["benchmark"], *records.values()):
        path = Path(str(row["path"]))
        _bound_bytes(row)


def _vector(episode: Episode) -> tuple[np.ndarray, str]:
    represented = represent(episode)
    stage = represented.stage.astype(np.float64).reshape(12, 4, order="C")[:, :3].ravel(order="C")
    vector = np.r_[represented.coarse[:96].astype(np.float64), stage,
                   represented.structural[:9].astype(np.float64)]
    vector = _canonical_float(vector, (DIMENSIONS,), "chart vector")
    return vector, representation_input_digest(represented)


def reconstruct_query(
    query_id: str, row: Mapping[str, Any], stock: pd.DataFrame, benchmark: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    cutoff = pd.Timestamp(str(row["cutoff"]))
    stock_prefix = asdict(causal_prefix_digest(stock, cutoff))
    benchmark_prefix = asdict(causal_prefix_digest(benchmark, cutoff))
    require(stock_prefix == row["stock_prefix"], f"query stock prefix differs: {query_id}")
    require(benchmark_prefix == row["benchmark_prefix"], f"query benchmark prefix differs: {query_id}")
    lookback = int(row["lookback"])
    require(lookback == 252 and str(row["representation_version"]) == "dense-v1",
            f"query representation contract differs: {query_id}")
    eligible = stock.loc[pd.to_datetime(stock["timestamp"]) <= cutoff]
    require(len(eligible) >= lookback, f"query history differs: {query_id}")
    window = eligible.tail(lookback).copy().reset_index(drop=True)
    actual_cutoff = pd.Timestamp(window["timestamp"].iloc[-1])
    require(actual_cutoff == cutoff, f"query cutoff is absent: {query_id}")
    episode = Episode(
        EpisodeKey(InstrumentKey("nasdaq", str(row["symbol"])), actual_cutoff,
                   lookback, "dense-v1"),
        window, benchmark.loc[pd.to_datetime(benchmark["timestamp"]) <= actual_cutoff].copy(),
        str(row["quality_tier"]),
    )
    require(episode.key.id == query_id, f"query EpisodeKey differs: {query_id}")
    vector, representation_digest = _vector(episode)
    return vector, {
        "query_episode_id": query_id,
        "query_representation_digest": representation_digest,
        "stock_prefix_digest": stock_prefix["digest"],
        "benchmark_prefix_digest": benchmark_prefix["digest"],
    }


def reconstruct_candidate(
    row: Mapping[str, Any], stock: pd.DataFrame, benchmark: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    episode_id = str(row["episode_id"])
    cutoff = pd.Timestamp(int(row["cutoff_ns"]))
    lookback = 252
    eligible = stock.loc[pd.to_datetime(stock["timestamp"]) <= cutoff]
    require(len(eligible) >= lookback, f"candidate history differs: {episode_id}")
    window = eligible.tail(lookback).copy().reset_index(drop=True)
    actual_cutoff = pd.Timestamp(window["timestamp"].iloc[-1])
    require(actual_cutoff == cutoff, f"candidate cutoff is absent: {episode_id}")
    key = EpisodeKey(InstrumentKey("nasdaq", str(row["symbol"])), actual_cutoff,
                     lookback, "dense-v1")
    require(key.id == episode_id, f"candidate EpisodeKey differs: {episode_id}")
    stock_prefix = asdict(causal_prefix_digest(stock, cutoff))
    benchmark_prefix = asdict(causal_prefix_digest(benchmark, cutoff))
    episode = Episode(key, window,
                      benchmark.loc[pd.to_datetime(benchmark["timestamp"]) <= cutoff].copy(), "A")
    vector, representation_digest = _vector(episode)
    return vector, {
        "episode_id": episode_id, "symbol": str(row["symbol"]),
        "cutoff_ns": int(row["cutoff_ns"]), "lookback": lookback,
        "representation_version": "dense-v1",
        "representation_digest": representation_digest,
        "stock_prefix": stock_prefix, "benchmark_prefix": benchmark_prefix,
    }


def reconstruct_vectors(
    repository: Path, prereg: Mapping[str, Any], *, workers: int,
) -> tuple[np.ndarray, np.ndarray, tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    require(type(workers) is int and workers in {1, 12}, "reconstruction workers must be 1 or 12")
    population = prereg["authorities"]["population"]
    query_ids = tuple(population["query_ids"])
    candidate_ids = tuple(population["cohort_ids"])
    registry_rows = prereg["authorities"]["sealed_query_transform"]["query_audits"]
    # The audit sidecar contains no source rows; the registry is an H1-bound authority.
    registry_path = repository / joint.REGISTRY
    expected_registry_sha = prereg["authorities"]["authority_file_sha256"][str(joint.REGISTRY)]
    require(_file_sha(registry_path) == expected_registry_sha, "frozen registry bytes differ")
    registry = _load_json(registry_path)
    by_query = {str(row["episode_id"]): row for row in registry["cases_data"]}
    by_candidate = {str(row["episode_id"]): row for row in population["episodes"]}
    require(set(by_query) == set(query_ids) and set(by_candidate) == set(candidate_ids),
            "reconstruction identities differ")
    manifest, records = _manifest_records(prereg)
    config = load_config_bytes(_bound_bytes(manifest["config"]), path=manifest["config"]["path"])
    spec = config.datasets.get("nasdaq")
    require(spec is not None and spec.adapter == "directory", "NASDAQ directory source differs")
    _validate_dataset_spec(spec, manifest)
    require(spec.benchmark is not None
            and spec.benchmark.path == Path(str(manifest["benchmark"]["path"])),
            "NASDAQ benchmark resolver differs")
    source = DirectorySource(spec)
    require(type(source) is DirectorySource, "NASDAQ source adapter differs")
    for symbol, record in records.items():
        require(source._files.get(symbol) == Path(str(record["path"])), f"source resolver differs: {symbol}")
    benchmark = _bound_frame(manifest["benchmark"], spec, symbol="__benchmark__", benchmark=True)

    query_output: list[tuple[np.ndarray, dict[str, Any]] | None] = [None] * len(query_ids)
    candidate_output: list[tuple[np.ndarray, dict[str, Any]] | None] = [None] * len(candidate_ids)
    query_index = {value: index for index, value in enumerate(query_ids)}
    candidate_index = {value: index for index, value in enumerate(candidate_ids)}
    symbol_queries: dict[str, list[str]] = {}
    symbol_candidates: dict[str, list[str]] = {}
    for value in query_ids:
        symbol_queries.setdefault(str(by_query[value]["symbol"]), []).append(value)
    for value in candidate_ids:
        symbol_candidates.setdefault(str(by_candidate[value]["symbol"]), []).append(value)
    symbols = tuple(sorted(set(symbol_queries) | set(symbol_candidates)))

    def one(symbol: str) -> tuple[str, list[tuple[str, tuple[np.ndarray, dict[str, Any]]]], list[tuple[str, tuple[np.ndarray, dict[str, Any]]]]]:
        stock = _bound_frame(records[symbol], spec, symbol=symbol)
        queries = [(value, reconstruct_query(value, by_query[value], stock, benchmark))
                   for value in symbol_queries.get(symbol, ())]
        candidates = [(value, reconstruct_candidate(by_candidate[value], stock, benchmark))
                      for value in symbol_candidates.get(symbol, ())]
        return symbol, queries, candidates

    if workers == 1:
        reconstructed = map(one, symbols)
    else:
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="b2-geometry-source")
        reconstructed = executor.map(one, symbols)
    try:
        for _, queries, candidates in reconstructed:
            for value, result in queries:
                query_output[query_index[value]] = result
            for value, result in candidates:
                candidate_output[candidate_index[value]] = result
    finally:
        if workers != 1:
            executor.shutdown()
    require(all(value is not None for value in query_output + candidate_output), "reconstruction is incomplete")
    require(_file_sha(registry_path) == expected_registry_sha,
            "frozen registry changed during reconstruction")
    query_values = tuple(value for value in query_output if value is not None)
    candidate_values = tuple(value for value in candidate_output if value is not None)
    raw_queries = _canonical_float([value[0] for value in query_values],
                                   (len(query_ids), DIMENSIONS), "raw queries")
    raw_candidates = _canonical_float([value[0] for value in candidate_values],
                                      (len(candidate_ids), DIMENSIONS), "raw candidates")
    query_audits = tuple(value[1] for value in query_values)
    candidate_audits = tuple(value[1] for value in candidate_values)
    require(list(query_audits) == registry_rows, "sealed query reconstruction audits differ")
    return raw_queries, raw_candidates, query_audits, candidate_audits


_DISTANCE_QUERIES: np.ndarray | None = None
_DISTANCE_CANDIDATES: np.ndarray | None = None


def _distance_worker_init(queries: np.ndarray, candidates: np.ndarray) -> None:
    global _DISTANCE_QUERIES, _DISTANCE_CANDIDATES
    _DISTANCE_QUERIES = queries
    _DISTANCE_CANDIDATES = candidates


def _distance_worker(bounds: tuple[int, int]) -> tuple[int, int, np.ndarray, np.ndarray]:
    require(_DISTANCE_QUERIES is not None and _DISTANCE_CANDIDATES is not None,
            "distance worker is uninitialized")
    start, stop = bounds
    rows = _DISTANCE_QUERIES[start:stop]
    return start, stop, chart_distances(rows, _DISTANCE_QUERIES), chart_distances(rows, _DISTANCE_CANDIDATES)


def _shard_paths(work: Path, start: int, stop: int) -> tuple[Path, Path, Path]:
    stem = f"rows-{start:05d}-{stop:05d}"
    root = work / stem
    return root / "METADATA.json", root / "query_pair.npy", root / "query_candidate.npy"


def _load_shard(
    work: Path, start: int, stop: int, *, binding: str, queries: int, candidates: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    meta_path, qq_path, qc_path = _shard_paths(work, start, stop)
    root = meta_path.parent
    if not root.exists() and not root.is_symlink():
        return None
    require(root.is_dir() and not root.is_symlink(), f"invalid geometry shard root: {start}:{stop}")
    require({path.name for path in root.iterdir()} == {
        "METADATA.json", "query_pair.npy", "query_candidate.npy",
    }, f"partial or unexpected geometry shard: {start}:{stop}")
    require(all(path.is_file() and not path.is_symlink() for path in (meta_path, qq_path, qc_path)),
            f"nonregular geometry shard member: {start}:{stop}")
    meta = _load_json(meta_path)
    qq_bytes = _bound_bytes({"path": str(qq_path), "bytes": qq_path.stat().st_size,
                             "sha256": meta.get("query_pair", {}).get("sha256")})
    qc_bytes = _bound_bytes({"path": str(qc_path), "bytes": qc_path.stat().st_size,
                             "sha256": meta.get("query_candidate", {}).get("sha256")})
    require(meta == {
        "schema_version": SHARD_SCHEMA, "binding_digest": binding,
        "start": start, "stop": stop,
        "query_pair": {"path": qq_path.name, "sha256": sha256(qq_bytes).hexdigest(),
                       "shape": [stop - start, queries], "dtype": "<f8"},
        "query_candidate": {"path": qc_path.name, "sha256": sha256(qc_bytes).hexdigest(),
                            "shape": [stop - start, candidates], "dtype": "<f8"},
    }, f"geometry shard metadata differs: {start}:{stop}")
    try:
        qq = np.load(io.BytesIO(qq_bytes), allow_pickle=False)
        qc = np.load(io.BytesIO(qc_bytes), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise GeometryError(f"unreadable geometry shard: {start}:{stop}") from error
    qq = _canonical_float(qq, (stop - start, queries), "query-pair shard")
    qc = _canonical_float(qc, (stop - start, candidates), "query-candidate shard")
    return qq, qc


def _write_shard(
    work: Path, start: int, stop: int, qq: np.ndarray, qc: np.ndarray, *, binding: str,
) -> None:
    meta_path, qq_path, qc_path = _shard_paths(work, start, stop)
    destination = meta_path.parent
    require(not destination.exists() and not destination.is_symlink(),
            f"create-only geometry shard exists: {start}:{stop}")
    qq = _canonical_float(qq, qq.shape, "query-pair shard")
    qc = _canonical_float(qc, qc.shape, "query-candidate shard")
    temporary = work / f".{destination.name}.tmp-{os.getpid()}-{uuid4().hex}"
    try:
        temporary.mkdir()
        temp_meta = temporary / meta_path.name
        temp_qq = temporary / qq_path.name
        temp_qc = temporary / qc_path.name
        _atomic_npy(temp_qq, qq); _atomic_npy(temp_qc, qc)
        _atomic_json(temp_meta, {
            "schema_version": SHARD_SCHEMA, "binding_digest": binding,
            "start": start, "stop": stop,
            "query_pair": {"path": qq_path.name, "sha256": _file_sha(temp_qq),
                           "shape": list(qq.shape), "dtype": "<f8"},
            "query_candidate": {"path": qc_path.name, "sha256": _file_sha(temp_qc),
                                "shape": list(qc.shape), "dtype": "<f8"},
        })
        descriptor = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.rename(temporary, destination)
        except FileExistsError as error:
            raise GeometryError(f"create-only geometry shard exists: {start}:{stop}") from error
        descriptor = os.open(work, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists() and temporary.is_dir() and not temporary.is_symlink():
            shutil.rmtree(temporary)


def distance_matrices(
    queries: np.ndarray, candidates: np.ndarray, *, workers: int, work: Path | None = None,
    binding: str = "synthetic", chunk_rows: int = CHUNK_ROWS,
) -> tuple[np.ndarray, np.ndarray]:
    queries = _canonical_float(queries, (len(queries), DIMENSIONS), "transformed queries")
    candidates = _canonical_float(candidates, (len(candidates), DIMENSIONS), "transformed candidates")
    require(type(workers) is int and workers in {1, 12}, "distance workers must be 1 or 12")
    require(type(chunk_rows) is int and chunk_rows > 0 and len(queries) > 0 and len(candidates) > 0,
            "distance chunk contract differs")
    if work is not None:
        require(not work.is_symlink(), "geometry work directory may not be a symlink")
        work.mkdir(parents=True, exist_ok=True)
    bounds = tuple((start, min(start + chunk_rows, len(queries)))
                   for start in range(0, len(queries), chunk_rows))
    if work is not None:
        allowed = {_shard_paths(work, start, stop)[0].parent.name for start, stop in bounds}
        staging_prefixes = tuple(f".{name}.tmp-" for name in sorted(allowed))
        for path in tuple(work.iterdir()):
            if (not path.is_symlink() and path.is_dir()
                    and path.name.startswith(staging_prefixes)):
                shutil.rmtree(path)
        unexpected = [path.name for path in work.iterdir()
                      if path.name not in allowed or path.is_symlink() or not path.is_dir()]
        require(not unexpected, f"unexpected geometry work artifacts: {sorted(unexpected)}")
    qq = np.empty((len(queries), len(queries)), dtype="<f8")
    qc = np.empty((len(queries), len(candidates)), dtype="<f8")
    missing: list[tuple[int, int]] = []
    for start, stop in bounds:
        shard = _load_shard(work, start, stop, binding=binding,
                            queries=len(queries), candidates=len(candidates)) if work else None
        if shard is None:
            missing.append((start, stop))
        else:
            qq[start:stop], qc[start:stop] = shard
    if missing:
        if workers == 1:
            _distance_worker_init(queries, candidates)
            generated = map(_distance_worker, missing)
            executor = None
        else:
            executor = ProcessPoolExecutor(max_workers=workers, initializer=_distance_worker_init,
                                           initargs=(queries, candidates))
            generated = (future.result() for future in as_completed(
                [executor.submit(_distance_worker, bounds) for bounds in missing]
            ))
        try:
            for start, stop, query_pair, query_candidate in generated:
                query_pair = _canonical_float(query_pair, (stop - start, len(queries)), "query-pair shard")
                query_candidate = _canonical_float(query_candidate, (stop - start, len(candidates)), "query-candidate shard")
                if work is not None:
                    _write_shard(work, start, stop, query_pair, query_candidate, binding=binding)
                qq[start:stop], qc[start:stop] = query_pair, query_candidate
        finally:
            if executor is not None:
                executor.shutdown()
    qq = _canonical_float(qq, (len(queries), len(queries)), "query-pair distances")
    qc = _canonical_float(qc, (len(queries), len(candidates)), "query-candidate distances")
    require((qq >= 0).all() and (qc >= 0).all() and np.array_equal(qq, qq.T)
            and np.all(qq.diagonal() == 0.0), "distance geometry invariants differ")
    return qq, qc


def causal_eligibility(prereg: Mapping[str, Any]) -> np.ndarray:
    population = prereg["authorities"]["population"]
    query_ids = tuple(population["query_ids"]); candidate_ids = tuple(population["cohort_ids"])
    query_index = {value: index for index, value in enumerate(query_ids)}
    episodes = {str(row["episode_id"]): row for row in population["episodes"]}
    require(set(episodes) == set(candidate_ids), "eligibility episode identities differ")
    result = np.zeros((len(query_ids), len(candidate_ids)), dtype=np.bool_)
    for column, episode_id in enumerate(candidate_ids):
        eligible = tuple(str(value) for value in episodes[episode_id]["eligible_query_ids"])
        require(len(set(eligible)) == len(eligible) and set(eligible) <= set(query_ids),
                f"eligibility IDs differ: {episode_id}")
        result[[query_index[value] for value in eligible], column] = True
    return result


def _ids_payload(prereg: Mapping[str, Any]) -> dict[str, Any]:
    population = prereg["authorities"]["population"]
    query_ids = list(population["query_ids"]); candidate_ids = list(population["cohort_ids"])
    episodes = {str(row["episode_id"]): row for row in population["episodes"]}
    return {
        "schema_version": IDS_SCHEMA,
        "query_ids": query_ids, "candidate_ids": candidate_ids,
        "query_ids_digest": id_digest(query_ids), "candidate_ids_digest": id_digest(candidate_ids),
        "population_digest": population["population_digest"],
        "candidate_keys": [{"episode_id": value, "dataset_id": "nasdaq",
                            "symbol": episodes[value]["symbol"],
                            "cutoff_ns": episodes[value]["cutoff_ns"], "lookback": 252,
                            "representation_version": "dense-v1"} for value in candidate_ids],
    }


def _array_record(path: Path, array: np.ndarray) -> dict[str, Any]:
    dtype = "|b1" if array.dtype == np.bool_ else "<f8"
    return {"path": path.name, "sha256": _file_sha(path),
            "semantic_digest": _array_semantic_digest(array),
            "shape": list(array.shape), "dtype": dtype, "order": "C"}


def compute(
    repository: Path, prereg: Mapping[str, Any], *, workers: int, work: Path,
    source_verified_by_h1: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    start = time.monotonic()
    cpu_start = time.process_time()
    children_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    if not source_verified_by_h1:
        verify_external_source_manifest(prereg)
    raw_queries, raw_candidates, query_audits, candidate_audits = reconstruct_vectors(
        repository, prereg, workers=workers,
    )
    # Detect a rewrite racing the decode.  A mixed old/new in-memory snapshot
    # is never allowed to escape merely because decoding itself succeeded.
    verify_external_source_manifest(prereg)
    sealed = prereg["authorities"]["sealed_query_transform"]
    population = prereg["authorities"]["population"]
    query_ids = population["query_ids"]
    unsupported = [row for row in population["episodes"] if not row["primary"]]
    require(raw_queries.shape == (3270, DIMENSIONS)
            and raw_candidates.shape == (369, DIMENSIONS)
            and len(population["primary_ids"]) == 357 and len(unsupported) == 12,
            "frozen B2 geometry inventory differs")
    require(id_digest(query_ids) == sealed["query_ids_digest"], "sealed query ID digest differs")
    require(array_digest(raw_queries) == sealed["raw_vector_digest"], "sealed raw query digest differs")
    transformed_queries, transformed_candidates, constants = empirical_chart_transform(
        raw_queries, raw_candidates,
    )
    transformed_queries = _canonical_float(transformed_queries, raw_queries.shape, "transformed queries")
    transformed_candidates = _canonical_float(transformed_candidates, raw_candidates.shape, "transformed candidates")
    sealed_reconstruction, integer_midranks, balanced_constants = midrank_transform(raw_queries)
    sealed_reconstruction = _canonical_float(sealed_reconstruction, raw_queries.shape,
                                             "sealed transform reconstruction")
    require(np.array_equal(transformed_queries, sealed_reconstruction)
            and constants == balanced_constants, "query transform implementations disagree")
    require(integer_array_digest(integer_midranks) == sealed["integer_midrank_digest"],
            "sealed integer-midrank digest differs")
    require(array_digest(transformed_queries) == sealed["transformed_digest"], "sealed transformed query digest differs")
    require(list(constants) == sealed["constant_columns"], "sealed constant columns differ")
    ids = _ids_payload(prereg)
    binding = stable_hash({
        "schema_version": SHARD_SCHEMA, "preregistration_digest": prereg["preregistration_digest"],
        "runtime_sha256": prereg["runtime_sha256"], "query_ids_digest": ids["query_ids_digest"],
        "candidate_ids_digest": ids["candidate_ids_digest"],
        "transformed_queries": array_digest(transformed_queries),
        "transformed_candidates": array_digest(transformed_candidates), "chunk_rows": CHUNK_ROWS,
    })
    query_pair, query_candidate = distance_matrices(
        transformed_queries, transformed_candidates, workers=workers, work=work,
        binding=binding, chunk_rows=CHUNK_ROWS,
    )
    eligibility = causal_eligibility(prereg)
    specificity = candidate_specificity_ranks(query_candidate, eligibility)
    require(specificity.shape == eligibility.shape and np.isnan(specificity[~eligibility]).all()
            and np.isfinite(specificity[eligibility]).all(), "specificity/eligibility invariants differ")
    arrays = {
        "raw_queries": raw_queries, "raw_candidates": raw_candidates,
        "transformed_queries": transformed_queries, "transformed_candidates": transformed_candidates,
        "query_pair_distances": query_pair, "query_candidate_distances": query_candidate,
        "specificity_ranks": np.ascontiguousarray(specificity.astype("<f8")),
        "causal_eligibility": np.ascontiguousarray(eligibility),
    }
    audit = {
        "sealed_transform_sha256": prereg["authorities"]["authority_file_sha256"][str(joint.TRANSFORM)],
        "sealed_query_ids_digest": sealed["query_ids_digest"],
        "sealed_raw_query_digest": sealed["raw_vector_digest"],
        "sealed_transformed_query_digest": sealed["transformed_digest"],
        "sealed_integer_midrank_digest": sealed["integer_midrank_digest"],
        "sealed_query_audits_digest": stable_hash(sealed["query_audits"]),
        "reconstructed_query_audits_digest": stable_hash(list(query_audits)),
        "candidate_audits_digest": stable_hash(list(candidate_audits)),
        "candidate_audits": list(candidate_audits),
        "constant_columns": list(constants), "full_specificity_denominator": 369,
        "unsupported_candidates_included": len(unsupported),
        "query_reconstruction_workers": workers, "distance_workers": workers,
        "distance_chunk_rows": CHUNK_ROWS,
        "distance_shards": math.ceil(len(raw_queries) / CHUNK_ROWS),
        "distance_shard_binding_digest": binding,
        "performance": {
            "wall_seconds": time.monotonic() - start,
            "parent_cpu_seconds": time.process_time() - cpu_start,
            "children_cpu_seconds": (
                resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime
                + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime
                - children_start.ru_utime - children_start.ru_stime
            ),
            "parent_lifetime_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "children_lifetime_peak_rss_mib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0,
            "blas_threads": 1,
        },
        "synthetic_1_12_byte_identity_h0_test": "test_distance_serial_parallel_byte_identity and test_reconstruction_serial_threaded_byte_identity",
    }
    return arrays, ids, audit


def run(repository: Path, *, workers: int = 12) -> dict[str, Any]:
    require(workers == 12, "production geometry requires exactly 12 workers")
    repository = repository.resolve(); output = repository / OUTPUT; work = repository / WORK
    output.parent.mkdir(parents=True, exist_ok=True)
    run_lock = output.parent / f".{output.name}.run.lock"
    lock_descriptor = os.open(run_lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        require(not output.exists() and not output.is_symlink(), "geometry output already exists")
        prereg = _load_json(repository / joint.PREREGISTRATION)
        h1 = joint.validate_h1(repository, prereg)
        # validate_h1 has just rehashed the complete external manifest.  A
        # second verification inside compute, after decoding, closes the race;
        # avoid a redundant third full-file pass before decoding.
        with threadpool_limits(limits=1, user_api="blas"):
            arrays, ids, audit = compute(repository, prereg, workers=workers, work=work,
                                        source_verified_by_h1=True)
        require(set(arrays) == set(ARRAY_NAMES), "geometry array closure differs")
        temporary = output.parent / f".{output.name}.tmp-{os.getpid()}-{uuid4().hex}"
        try:
            temporary.mkdir(parents=True)
            _atomic_json(temporary / "IDS.json", ids)
            records: dict[str, Any] = {}
            for name in ARRAY_NAMES:
                array = arrays[name]
                if name in FLOAT_ARRAYS:
                    if name == "specificity_ranks":
                        require(np.isnan(array[~arrays["causal_eligibility"]]).all()
                                and np.isfinite(array[arrays["causal_eligibility"]]).all(),
                                "specificity nonfinite placement differs")
                    else:
                        require(np.isfinite(array).all(), f"nonfinite output array: {name}")
                    array[array == 0.0] = 0.0
                _atomic_npy(temporary / f"{name}.npy", array)
                records[name] = _array_record(temporary / f"{name}.npy", array)
            source_manifest = prereg["authorities"]["external_source_manifest"]
            state = {
                "schema_version": SCHEMA, "status": "geometry_complete_pending_independent_verification",
                "passed": True, "preregistration_commit": h1,
                "preregistration_digest": prereg["preregistration_digest"],
                "implementation_commit": prereg["implementation_commit"],
                "ids": {"path": "IDS.json", "sha256": _file_sha(temporary / "IDS.json"),
                        "query_ids_digest": ids["query_ids_digest"],
                        "candidate_ids_digest": ids["candidate_ids_digest"],
                        "population_digest": ids["population_digest"]},
                "arrays": records,
                "source_manifest": {"manifest_digest": source_manifest["manifest_digest"],
                                    "source_lock_digest": source_manifest["source_lock_digest"],
                                    "stock_files_count": source_manifest["stock_files_count"],
                                    "config_sha256": source_manifest["config"]["sha256"],
                                    "benchmark_sha256": source_manifest["benchmark"]["sha256"],
                                    "stock_files_digest": stable_hash(source_manifest["stock_files"])},
                "transform_audit": audit,
                "claims": {"real_forward_outcomes_accessed": False, "b2_statistics_computed": False,
                           "predictive_claim_authorized": False, "production_promotion_authorized": False},
            }
            state["geometry_digest"] = stable_hash(state)
            _atomic_json(temporary / "RESULT.json", state)
            _publish_directory(temporary, output)
            return state
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(argv)
    try:
        result = run(args.repository, workers=args.workers)
    except GeometryError as error:
        print(f"geometry unresolved: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
