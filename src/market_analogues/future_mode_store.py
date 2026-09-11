"""Shared immutable path indexing and atomic partitions for R2 mode builds."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


PATH_COLUMNS = (
    "benchmark_relative_close_return",
    "close_return",
    "contract_digest",
    "cutoff",
    "episode_id",
    "expected_session_match",
    "source_content_digest",
    "source_fingerprint",
    "step",
    "timestamp",
)
DICTIONARY_COLUMNS = (
    "contract_digest",
    "cutoff",
    "episode_id",
    "source_content_digest",
    "source_fingerprint",
    "timestamp",
)
PARTITION_SCHEMA = "m04r15-r2-mode-partition-v1"


class FutureModeStoreError(RuntimeError):
    """Raised when an immutable input or partition fails closed."""


def stable_digest(value: Any) -> str:
    return sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode() + b"\n"


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
        value.st_ctime_ns, value.st_mode,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class EpisodeSlice:
    start: int
    stop: int


class ArrowPathIndex:
    """One dictionary-encoded Arrow image with O(1) contiguous episode lookup."""

    def __init__(
        self, table: pa.Table, slices: Mapping[str, EpisodeSlice], source: Path,
    ) -> None:
        self._table = table
        self._slices = dict(slices)
        self.source = source

    @classmethod
    def load(cls, path: Path) -> "ArrowPathIndex":
        path = Path(path)
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except OSError as error:
            raise FutureModeStoreError(f"unsafe or missing path store: {path}") from error
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise FutureModeStoreError(f"regular path store required: {path}")
            if _identity(before) != _identity(os.stat(path, follow_symlinks=False)):
                raise FutureModeStoreError(f"path store identity differs: {path}")
            with os.fdopen(os.dup(descriptor), "rb") as handle:
                table = pq.read_table(
                    handle, columns=list(PATH_COLUMNS),
                    read_dictionary=list(DICTIONARY_COLUMNS),
                ).combine_chunks()
            after = os.fstat(descriptor)
            pathname_after = os.stat(path, follow_symlinks=False)
            if _identity(before) != _identity(after) or _identity(after) != _identity(pathname_after):
                raise FutureModeStoreError(f"path store changed during preload: {path}")
        except FutureModeStoreError:
            raise
        except Exception as error:
            raise FutureModeStoreError(f"invalid path store: {path}") from error
        finally:
            os.close(descriptor)

        if tuple(table.column_names) != PATH_COLUMNS or table.num_rows < 1:
            raise FutureModeStoreError("path store schema or row count differs")
        episode_column = table.column("episode_id")
        if episode_column.num_chunks != 1:
            raise FutureModeStoreError("episode column did not combine")
        encoded = episode_column.chunk(0)
        if not pa.types.is_dictionary(encoded.type):
            raise FutureModeStoreError("episode column is not dictionary encoded")
        if encoded.null_count:
            raise FutureModeStoreError("episode identity contains nulls")
        codes = encoded.indices.to_numpy(zero_copy_only=True)
        boundaries = np.flatnonzero(codes[1:] != codes[:-1]) + 1
        starts = np.concatenate((np.array([0], dtype=np.int64), boundaries))
        stops = np.concatenate((boundaries, np.array([len(codes)], dtype=np.int64)))
        run_codes = codes[starts]
        if len(np.unique(run_codes)) != len(run_codes):
            raise FutureModeStoreError("episode rows are not physically contiguous")
        dictionary = encoded.dictionary
        slices = {
            str(dictionary[int(code)].as_py()): EpisodeSlice(int(start), int(stop))
            for code, start, stop in zip(run_codes, starts, stops, strict=True)
        }
        if "" in slices or len(slices) != len(run_codes):
            raise FutureModeStoreError("episode identity index differs")
        return cls(table, slices, path.resolve())

    @property
    def row_count(self) -> int:
        return self._table.num_rows

    @property
    def episode_count(self) -> int:
        return len(self._slices)

    @property
    def episode_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._slices))

    @property
    def image_bytes(self) -> int:
        return self._table.nbytes

    def rows(self, episode_id: str) -> tuple[dict[str, Any], ...]:
        location = self._slices.get(str(episode_id))
        if location is None:
            return ()
        return tuple(self._table.slice(
            location.start, location.stop - location.start,
        ).to_pylist())


_SHARED_PATH_INDEX: ArrowPathIndex | None = None


def install_shared_path_index(index: ArrowPathIndex) -> None:
    """Install the immutable image before forking full-build workers."""
    global _SHARED_PATH_INDEX
    if _SHARED_PATH_INDEX is not None and _SHARED_PATH_INDEX is not index:
        raise FutureModeStoreError("a different shared path index is already installed")
    _SHARED_PATH_INDEX = index


def shared_episode_digest(episode_id: str) -> str:
    """Return canonical lookup bytes through the inherited read-only image."""
    if _SHARED_PATH_INDEX is None:
        raise FutureModeStoreError("shared path index is not installed")
    return stable_digest(list(_SHARED_PATH_INDEX.rows(episode_id)))


def partition_query_ids(query_ids: Sequence[str], count: int) -> tuple[tuple[str, ...], ...]:
    """Split sorted unique query IDs into deterministic contiguous partitions."""
    if count < 1:
        raise FutureModeStoreError("partition count must be positive")
    values = tuple(sorted(str(value) for value in query_ids))
    if not values or "" in values or len(set(values)) != len(values):
        raise FutureModeStoreError("query inventory must be nonempty and unique")
    count = min(count, len(values))
    width, remainder = divmod(len(values), count)
    groups: list[tuple[str, ...]] = []
    offset = 0
    for index in range(count):
        size = width + (1 if index < remainder else 0)
        groups.append(values[offset:offset + size])
        offset += size
    if offset != len(values) or any(not group for group in groups):
        raise FutureModeStoreError("partition accounting differs")
    return tuple(groups)


def partition_payload(
    partition_id: int, query_ids: Sequence[str], results: Sequence[Mapping[str, Any]],
    *, contract_digest: str,
) -> dict[str, Any]:
    ordered_ids = tuple(str(value) for value in query_ids)
    if not ordered_ids or "" in ordered_ids or tuple(sorted(set(ordered_ids))) != ordered_ids:
        raise FutureModeStoreError("partition query IDs must be sorted, nonempty and unique")
    try:
        rows = sorted(
            (dict(value) for value in results),
            key=lambda row: str(row["query_case_id"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise FutureModeStoreError("partition result rows differ") from error
    if partition_id < 0 or not contract_digest:
        raise FutureModeStoreError("partition identity differs")
    if tuple(str(row.get("query_case_id", "")) for row in rows) != ordered_ids:
        raise FutureModeStoreError("partition query/result inventory differs")
    state = {
        "schema_version": PARTITION_SCHEMA,
        "partition_id": partition_id,
        "contract_digest": contract_digest,
        "query_ids": list(ordered_ids),
        "query_ids_digest": stable_digest(list(ordered_ids)),
        "results": rows,
        "results_digest": stable_digest(rows),
    }
    return {**state, "partition_digest": stable_digest(state)}


def _strict_json(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise FutureModeStoreError(f"regular partition file required: {path}")
    raw = path.read_bytes()

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise FutureModeStoreError(f"duplicate JSON key: {path}:{key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw, object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                FutureModeStoreError(f"nonfinite JSON: {path}:{item}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FutureModeStoreError(f"invalid partition JSON: {path}") from error
    if type(value) is not dict:
        raise FutureModeStoreError("partition JSON object required")
    return value, raw


def validate_partition(
    directory: Path, partition_id: int, query_ids: Sequence[str],
    *, contract_digest: str,
) -> dict[str, Any]:
    if directory.is_symlink() or not directory.is_dir():
        raise FutureModeStoreError(f"regular partition directory required: {directory}")
    entries = sorted(path.name for path in directory.iterdir())
    if entries != ["PARTITION.json"]:
        raise FutureModeStoreError("partition layout differs")
    value, raw = _strict_json(directory / "PARTITION.json")
    state = {key: item for key, item in value.items() if key != "partition_digest"}
    expected_ids = list(map(str, query_ids))
    results = value.get("results")
    if type(results) is not list or any(type(row) is not dict for row in results):
        raise FutureModeStoreError("partition results must be JSON objects")
    try:
        sealed = all((
            value.get("schema_version") == PARTITION_SCHEMA,
            value.get("partition_id") == partition_id,
            value.get("contract_digest") == contract_digest,
            value.get("query_ids") == expected_ids,
            value.get("query_ids_digest") == stable_digest(expected_ids),
            value.get("results_digest") == stable_digest(results),
            value.get("partition_digest") == stable_digest(state),
            raw == canonical_json(value),
        ))
    except (TypeError, ValueError) as error:
        raise FutureModeStoreError("partition seal is not canonical JSON") from error
    if not sealed:
        raise FutureModeStoreError("partition seal differs")
    if [row.get("query_case_id") for row in results] != expected_ids:
        raise FutureModeStoreError("partition result identities differ")
    return value


def _conflict_tag(target: Path) -> str:
    path = target / "PARTITION.json"
    if path.is_file() and not path.is_symlink():
        return sha256(path.read_bytes()).hexdigest()[:16]
    metadata = os.lstat(target)
    return stable_digest([target.name, metadata.st_mode, metadata.st_size])[:16]


def quarantine_partition(root: Path, target: Path) -> Path:
    conflicts = root / "conflicts"
    conflicts.mkdir(exist_ok=True)
    destination = conflicts / f"{target.name}-{_conflict_tag(target)}"
    if destination.exists() or destination.is_symlink():
        raise FutureModeStoreError(f"conflict quarantine target exists: {destination}")
    os.rename(target, destination)
    _fsync_directory(conflicts)
    _fsync_directory(root / "partitions")
    return destination


def recover_stale_partition_temporaries(root: Path) -> tuple[str, ...]:
    """Move crash-left staging directories aside before a single-parent resume."""
    partitions = root / "partitions"
    if not partitions.exists():
        return ()
    if partitions.is_symlink() or not partitions.is_dir():
        raise FutureModeStoreError(f"regular partitions directory required: {partitions}")
    interrupted = root / "interrupted"
    recovered: list[str] = []
    for candidate in sorted(partitions.iterdir(), key=lambda path: path.name):
        if not candidate.name.startswith(".partition-"):
            continue
        if candidate.is_symlink() or not candidate.is_dir():
            raise FutureModeStoreError(f"unsafe stale partition temporary: {candidate}")
        interrupted.mkdir(exist_ok=True)
        tag = stable_digest([
            candidate.name,
            sorted(child.name for child in candidate.iterdir()),
        ])[:16]
        destination = interrupted / f"{candidate.name[1:]}-{tag}"
        if destination.exists() or destination.is_symlink():
            raise FutureModeStoreError(
                f"interrupted partition target exists: {destination}"
            )
        os.rename(candidate, destination)
        recovered.append(destination.name)
    if recovered:
        _fsync_directory(interrupted)
        _fsync_directory(partitions)
    return tuple(recovered)


def publish_partition(
    root: Path, partition_id: int, query_ids: Sequence[str],
    results: Sequence[Mapping[str, Any]], *, contract_digest: str,
) -> str:
    """Publish one canonical partition, reusing equality and quarantining conflicts."""
    root.mkdir(parents=True, exist_ok=True)
    partitions = root / "partitions"
    partitions.mkdir(exist_ok=True)
    name = f"partition-{partition_id:04d}"
    target = partitions / name
    payload = partition_payload(
        partition_id, query_ids, results, contract_digest=contract_digest,
    )
    encoded = canonical_json(payload)
    if target.exists() or target.is_symlink():
        try:
            existing = validate_partition(
                target, partition_id, query_ids, contract_digest=contract_digest,
            )
        except FutureModeStoreError as error:
            quarantined = quarantine_partition(root, target)
            raise FutureModeStoreError(
                f"conflicting partition quarantined at {quarantined}"
            ) from error
        if canonical_json(existing) == encoded:
            return "reused"
        quarantined = quarantine_partition(root, target)
        raise FutureModeStoreError(f"conflicting partition quarantined at {quarantined}")

    temporary = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=partitions))
    try:
        path = temporary / "PARTITION.json"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(temporary)
        os.rename(temporary, target)
        _fsync_directory(partitions)
    except BaseException:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
        raise
    return "created"
