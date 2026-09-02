from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import shutil
import stat
from time import perf_counter
from typing import Any, Iterator, Mapping
from uuid import uuid4

from .packed_bound_store import load_packed_generation
from .types import stable_hash


READY_SCHEMA_VERSION = "m04r-resident-packed-store-ready-v2"
CONTENT_SCHEMA_VERSION = "m04r-resident-packed-store-content-v1"
VALIDATION_OBSERVATION_SCHEMA_VERSION = "m04r-resident-validation-observation-v1"
FILE_IDENTITY_LEASE_SCHEMA_VERSION = "m04r-resident-file-identity-lease-v1"

_HEX = frozenset("0123456789abcdef")
_READY_KEYS = frozenset({
    "schema_version", "mode", "content", "content_digest", "seal",
    "seal_digest", "capacity_observation", "startup_timings", "created_at",
    "ready_digest",
})
_CONTENT_KEYS = frozenset({
    "schema_version", "generation_id", "provenance_digest", "manifest_digest",
    "pack_contract_digest", "quantized_bound_contract_digest",
    "physical_generation_bytes", "source_files", "mirror_files",
})
_SEAL_KEYS = frozenset({
    "generation_id", "provenance_digest", "pack_contract_digest",
    "quantized_bound_contract_digest", "source_store_root", "mirror_root",
    "mirror_store_root", "source_generation_st_dev", "mirror_generation_st_dev",
    "source_files", "mirror_files", "resident_mount", "mountinfo_path",
    "resident_capacity_bytes", "required_capacity_bytes", "reserve_bytes",
    "physical_generation_bytes", "storage_class", "latency_scope",
    "query_specific_inputs_used", "outcomes_or_labels_used",
    "real_forward_outcomes_accessed", "source_active_pointer_absent",
    "mirror_active_pointer_absent",
})
_FILE_KEYS = frozenset({"path", "bytes", "sha256", "st_dev"})
_CONTENT_FILE_KEYS = frozenset({"bytes", "sha256"})
_CAPACITY_KEYS = frozenset({"capacity_bytes", "available_bytes", "block_bytes"})
_MOUNT_KEYS = frozenset({
    "mount_id", "parent_mount_id", "major_minor", "mount_root", "mount_point",
    "mount_options", "optional_fields", "fs_type", "mount_source",
    "super_options", "st_dev",
})
_TIMING_KEYS = frozenset({
    "source_content_verification_seconds", "mirror_copy_seconds",
    "mirror_content_verification_seconds", "readiness_seal_seconds",
    "total_before_ready_seconds",
})
_FILE_NAMES = frozenset({"manifest", "rows", "overflow"})
_LATENCY_SCOPE = (
    "tmpfs-backed empirical query; not durable-storage disk-cold and not "
    "guaranteed unswappable RAM"
)


class ResidentStoreError(ValueError):
    pass


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and value == value.lower()
        and set(value).issubset(_HEX)
    )


def _exact_int(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ResidentStoreError(f"resident READY {label} must be an integer >= {minimum}")
    return value


def _file_sha256(path: Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _unescape_mount_path(value: str) -> str:
    output = bytearray()
    encoded = value.encode()
    index = 0
    while index < len(encoded):
        if (
            encoded[index:index + 1] == b"\\" and index + 3 < len(encoded)
            and all(48 <= byte <= 55 for byte in encoded[index + 1:index + 4])
        ):
            output.append(int(encoded[index + 1:index + 4], 8))
            index += 4
        else:
            output.append(encoded[index])
            index += 1
    return os.fsdecode(bytes(output))


def _mount_binding(path: Path, mountinfo_path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    matches: list[tuple[int, dict[str, Any]]] = []
    try:
        lines = mountinfo_path.read_text().splitlines()
    except OSError as exc:
        raise ResidentStoreError(f"cannot read mountinfo: {exc}") from exc
    for line in lines:
        try:
            left_text, right_text = line.split(" - ", 1)
            left, right = left_text.split(), right_text.split()
            mount_point = Path(_unescape_mount_path(left[4])).resolve()
            if resolved != mount_point and mount_point not in resolved.parents:
                continue
            binding = {
                "mount_id": int(left[0]), "parent_mount_id": int(left[1]),
                "major_minor": left[2], "mount_root": _unescape_mount_path(left[3]),
                "mount_point": str(mount_point), "mount_options": left[5].split(","),
                "optional_fields": left[6:], "fs_type": right[0],
                "mount_source": _unescape_mount_path(right[1]),
                "super_options": right[2].split(","),
            }
        except (IndexError, ValueError) as exc:
            raise ResidentStoreError(f"malformed mountinfo line: {line!r}") from exc
        matches.append((len(str(mount_point)), binding))
    if not matches:
        raise ResidentStoreError(f"no mountinfo entry contains {resolved}")
    binding = max(matches, key=lambda value: value[0])[1]
    observed = resolved.stat()
    device = f"{os.major(observed.st_dev)}:{os.minor(observed.st_dev)}"
    if binding["major_minor"] != device:
        raise ResidentStoreError("mountinfo device differs from tmpfs-backed path st_dev")
    binding["st_dev"] = int(observed.st_dev)
    return binding


def _capacity_binding(path: Path) -> dict[str, int]:
    values = os.statvfs(path)
    unit = int(values.f_frsize or values.f_bsize)
    return {
        "capacity_bytes": unit * int(values.f_blocks),
        "available_bytes": unit * int(values.f_bavail),
        "block_bytes": unit,
    }


def _physical_files(
    store_root: Path, generation_id: str, manifest: Mapping[str, Any],
) -> dict[str, Path]:
    generation = store_root / "generations" / generation_id
    rows_file = str(manifest["rows_file"])
    overflow_file = str(manifest["overflow_file"])
    if any(
        not value or Path(value).name != value
        for value in (rows_file, overflow_file)
    ):
        raise ResidentStoreError("packed generation contains an unsafe filename")
    return {
        "manifest": generation / "manifest.json",
        "rows": generation / rows_file,
        "overflow": generation / overflow_file,
    }


def _plain_directory(path: Path, within: Path | None = None) -> Path:
    try:
        observed = path.lstat()
    except OSError as exc:
        raise ResidentStoreError(f"cannot inspect tmpfs-backed directory {path}: {exc}") from exc
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
        raise ResidentStoreError(f"tmpfs-backed directory is not plain: {path}")
    resolved = path.resolve(strict=True)
    if within is not None:
        root = within.resolve(strict=True)
        if resolved != root and not resolved.is_relative_to(root):
            raise ResidentStoreError(f"tmpfs-backed directory escapes mirror: {path}")
    return resolved


def _plain_file(path: Path, generation: Path, st_dev: int) -> Path:
    try:
        observed = path.lstat()
    except OSError as exc:
        raise ResidentStoreError(f"cannot inspect tmpfs-backed file {path}: {exc}") from exc
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
        raise ResidentStoreError(f"tmpfs-backed file is not plain regular: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(generation.resolve(strict=True)):
        raise ResidentStoreError(f"tmpfs-backed file escapes generation: {path}")
    if observed.st_dev != st_dev or resolved.stat().st_dev != st_dev:
        raise ResidentStoreError(f"tmpfs-backed file device differs from mount: {path}")
    return resolved


def _validate_mirror_layout(
    mirror: Path, store: Path, generation_id: str, manifest: Mapping[str, Any],
    mount_st_dev: int,
) -> dict[str, Path]:
    root = _plain_directory(mirror)
    directories = (
        _plain_directory(store, root),
        _plain_directory(store / "generations", root),
        _plain_directory(store / "generations" / generation_id, root),
    )
    if root.stat().st_dev != mount_st_dev or any(
        value.stat().st_dev != mount_st_dev for value in directories
    ):
        raise ResidentStoreError("tmpfs-backed directory device differs from mount")
    generation = directories[-1]
    return {
        name: _plain_file(path, generation, mount_st_dev)
        for name, path in _physical_files(store, generation_id, manifest).items()
    }


def _file_binding(
    path: Path, verified_bytes: int | None = None,
    verified_sha256: str | None = None,
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    observed = resolved.stat()
    if verified_bytes is not None and observed.st_size != verified_bytes:
        raise ResidentStoreError(f"verified file size differs: {resolved}")
    return {
        "path": str(resolved), "bytes": int(observed.st_size),
        "sha256": verified_sha256 or _file_sha256(resolved),
        "st_dev": int(observed.st_dev),
    }


def _verified_bindings(
    files: Mapping[str, Path], manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        "manifest": _file_binding(files["manifest"]),
        "rows": _file_binding(
            files["rows"], int(manifest["rows_bytes"]), str(manifest["rows_sha256"]),
        ),
        "overflow": _file_binding(
            files["overflow"], int(manifest["overflow_bytes"]),
            str(manifest["overflow_sha256"]),
        ),
    }


def _content_bindings(
    bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        name: {"bytes": int(value["bytes"]), "sha256": str(value["sha256"])}
        for name, value in bindings.items()
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def _exclusive_lock(mirror: Path) -> Iterator[None]:
    path = mirror / ".resident-store.lock"
    if path.is_symlink():
        raise ResidentStoreError("resident preparation lock must not be a symlink")
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ResidentStoreError(f"cannot open resident preparation lock: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ResidentStoreError("resident preparation lock is not regular")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _reclaim_abandoned_builds(mirror_store: Path) -> None:
    build_parent = mirror_store.parent / ".building"
    if build_parent.exists() or build_parent.is_symlink():
        _plain_directory(build_parent, mirror_store.parent)
        for abandoned in build_parent.iterdir():
            if (
                len(abandoned.name) != 32
                or any(value not in _HEX for value in abandoned.name)
            ):
                raise ResidentStoreError(
                    "resident staging contains an unexpected entry"
                )
            _plain_directory(abandoned, build_parent)
            shutil.rmtree(abandoned)


def _copy_generation(
    source_store: Path, mirror_store: Path, generation_id: str,
    manifest: Mapping[str, Any],
) -> None:
    final = mirror_store / "generations" / generation_id
    if final.exists() or final.is_symlink():
        raise ResidentStoreError("generation exists; use validate-existing mode")
    _reclaim_abandoned_builds(mirror_store)
    build_parent = mirror_store.parent / ".building"
    if not build_parent.exists():
        build_parent.mkdir()
    build_root = build_parent / uuid4().hex
    staged = build_root / "store" / "generations" / generation_id
    staged.mkdir(parents=True)
    try:
        for source in _physical_files(source_store, generation_id, manifest).values():
            destination = staged / source.name
            with source.open("rb") as input_handle, destination.open("xb") as output_handle:
                shutil.copyfileobj(input_handle, output_handle, 8 * 1024 * 1024)
                output_handle.flush()
                os.fsync(output_handle.fileno())
        final.parent.mkdir(parents=True, exist_ok=True)
        os.rename(staged, final)
        descriptor = os.open(final.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if build_root.exists():
            shutil.rmtree(build_root)


def _strict_json(raw: bytes) -> dict[str, Any]:
    def invalid_constant(value: str) -> None:
        raise ValueError(f"non-finite constant {value}")
    try:
        payload = json.loads(raw.decode("utf-8"), parse_constant=invalid_constant)
    except Exception as exc:
        raise ResidentStoreError(f"cannot parse resident READY: {exc}") from exc
    if type(payload) is not dict:
        raise ResidentStoreError("resident READY must be one JSON object")
    return payload


def _file_identity(path: Path) -> dict[str, Any]:
    observed = path.lstat()
    if stat.S_ISLNK(observed.st_mode) or not (
        stat.S_ISREG(observed.st_mode) or stat.S_ISDIR(observed.st_mode)
    ):
        raise ResidentStoreError(f"resident identity target is linked or special: {path}")
    return {
        "path": str(path.resolve(strict=True)), "st_dev": int(observed.st_dev),
        "st_ino": int(observed.st_ino), "st_size": int(observed.st_size),
        "st_mtime_ns": int(observed.st_mtime_ns),
        "st_ctime_ns": int(observed.st_ctime_ns), "st_mode": int(observed.st_mode),
    }


def _read_once(path: Path) -> tuple[bytes, dict[str, Any]]:
    if path.is_symlink():
        raise ResidentStoreError("resident READY must not be a symlink")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ResidentStoreError(f"cannot open resident READY: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ResidentStoreError("resident READY is not regular")
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        after = os.fstat(descriptor)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, name) != getattr(after, name) for name in fields):
            raise ResidentStoreError("resident READY changed during single read")
        raw = b"".join(chunks)
        if len(raw) != after.st_size:
            raise ResidentStoreError("resident READY read length differs")
        identity = {
            "path": str(path.resolve(strict=True)), "st_dev": int(after.st_dev),
            "st_ino": int(after.st_ino), "st_size": int(after.st_size),
            "st_mtime_ns": int(after.st_mtime_ns),
            "st_ctime_ns": int(after.st_ctime_ns), "st_mode": int(after.st_mode),
        }
        return raw, identity
    finally:
        os.close(descriptor)


def _validate_file_map(payload: Any, label: str, *, content: bool = False) -> None:
    fields = _CONTENT_FILE_KEYS if content else _FILE_KEYS
    if type(payload) is not dict or set(payload) != _FILE_NAMES:
        raise ResidentStoreError(f"resident READY {label} fields differ")
    for name, binding in payload.items():
        if type(binding) is not dict or set(binding) != fields:
            raise ResidentStoreError(f"resident READY {label}.{name} fields differ")
        _exact_int(binding.get("bytes"), f"{label}.{name}.bytes")
        if not _is_digest(binding.get("sha256")):
            raise ResidentStoreError(f"resident READY {label}.{name} hash differs")
        if not content:
            _exact_int(binding.get("st_dev"), f"{label}.{name}.st_dev")
            if not isinstance(binding.get("path"), str) or not Path(binding["path"]).is_absolute():
                raise ResidentStoreError(f"resident READY {label}.{name} path differs")


def _validate_content(payload: Any) -> None:
    if type(payload) is not dict or set(payload) != _CONTENT_KEYS:
        raise ResidentStoreError("resident READY content fields differ")
    if payload.get("schema_version") != CONTENT_SCHEMA_VERSION:
        raise ResidentStoreError("resident READY content schema differs")
    for name in (
        "generation_id", "provenance_digest", "manifest_digest",
        "pack_contract_digest", "quantized_bound_contract_digest",
    ):
        if not _is_digest(payload.get(name)):
            raise ResidentStoreError(f"resident READY content {name} differs")
    _exact_int(payload.get("physical_generation_bytes"), "content physical bytes", 1)
    _validate_file_map(payload.get("source_files"), "content source files", content=True)
    _validate_file_map(payload.get("mirror_files"), "content mirror files", content=True)
    if payload["source_files"] != payload["mirror_files"]:
        raise ResidentStoreError("resident READY source/mirror content differs")
    if payload["generation_id"] != payload["manifest_digest"]:
        raise ResidentStoreError("resident READY generation/manifest digest differs")
    if sum(value["bytes"] for value in payload["source_files"].values()) != (
        payload["physical_generation_bytes"]
    ):
        raise ResidentStoreError("resident READY physical content bytes differ")


def _validate_seal(payload: Any) -> None:
    if type(payload) is not dict or set(payload) != _SEAL_KEYS:
        raise ResidentStoreError("resident READY seal fields differ")
    for name in (
        "generation_id", "provenance_digest", "pack_contract_digest",
        "quantized_bound_contract_digest",
    ):
        if not _is_digest(payload.get(name)):
            raise ResidentStoreError(f"resident READY seal {name} differs")
    for name in ("source_store_root", "mirror_root", "mirror_store_root", "mountinfo_path"):
        if not isinstance(payload.get(name), str) or not Path(payload[name]).is_absolute():
            raise ResidentStoreError(f"resident READY seal {name} differs")
    for name in (
        "source_generation_st_dev", "mirror_generation_st_dev",
        "resident_capacity_bytes", "required_capacity_bytes", "reserve_bytes",
        "physical_generation_bytes",
    ):
        _exact_int(payload.get(name), f"seal {name}")
    _validate_file_map(payload.get("source_files"), "source files")
    _validate_file_map(payload.get("mirror_files"), "mirror files")
    mount = payload.get("resident_mount")
    if type(mount) is not dict or set(mount) != _MOUNT_KEYS:
        raise ResidentStoreError("resident READY mount fields differ")
    if mount.get("fs_type") != "tmpfs" or type(mount.get("st_dev")) is not int:
        raise ResidentStoreError("resident READY mount differs")
    for name in ("mount_id", "parent_mount_id"):
        _exact_int(mount.get(name), f"mount {name}")
    for name in (
        "major_minor", "mount_root", "mount_point", "fs_type", "mount_source",
    ):
        if not isinstance(mount.get(name), str):
            raise ResidentStoreError(f"resident READY mount {name} differs")
    for name in ("mount_options", "optional_fields", "super_options"):
        if type(mount.get(name)) is not list or not all(
            isinstance(value, str) for value in mount[name]
        ):
            raise ResidentStoreError(f"resident READY mount {name} differs")
    if payload.get("storage_class") != "tmpfs-backed-generation-v1":
        raise ResidentStoreError("resident READY storage class differs")
    if payload.get("latency_scope") != _LATENCY_SCOPE:
        raise ResidentStoreError("resident READY latency scope differs")
    for name in (
        "query_specific_inputs_used", "outcomes_or_labels_used",
        "real_forward_outcomes_accessed",
    ):
        if payload.get(name) is not False:
            raise ResidentStoreError(f"resident READY {name} differs")
    for name in ("source_active_pointer_absent", "mirror_active_pointer_absent"):
        if payload.get(name) is not True:
            raise ResidentStoreError(f"resident READY {name} differs")
    if payload["mirror_generation_st_dev"] != mount["st_dev"] or any(
        value["st_dev"] != mount["st_dev"]
        for value in payload["mirror_files"].values()
    ):
        raise ResidentStoreError("resident READY mirror device binding differs")
    if payload["required_capacity_bytes"] != (
        payload["physical_generation_bytes"] + payload["reserve_bytes"]
    ) or payload["resident_capacity_bytes"] < payload["required_capacity_bytes"]:
        raise ResidentStoreError("resident READY capacity accounting differs")
    mirror = Path(payload["mirror_root"])
    store = Path(payload["mirror_store_root"])
    expected_generation = store / "generations" / str(payload["generation_id"])
    if store.parent != mirror or any(
        not Path(value["path"]).is_relative_to(expected_generation)
        for value in payload["mirror_files"].values()
    ):
        raise ResidentStoreError("resident READY mirror path containment differs")


def observe_ready_strict(path: Path) -> dict[str, Any]:
    """Read one READY inode once and return a strict race-resistant observation."""
    raw, identity = _read_once(path)
    payload = _strict_json(raw)
    if set(payload) != _READY_KEYS:
        raise ResidentStoreError("resident READY top-level fields differ")
    if payload.get("schema_version") != READY_SCHEMA_VERSION:
        raise ResidentStoreError("resident READY schema differs")
    if payload.get("mode") not in {"copy-and-validate", "validate-existing"}:
        raise ResidentStoreError("resident READY mode differs")
    _validate_content(payload.get("content"))
    _validate_seal(payload.get("seal"))
    content, seal = payload["content"], payload["seal"]
    if not _is_digest(payload.get("content_digest")) or (
        payload["content_digest"] != stable_hash(content)
    ):
        raise ResidentStoreError("resident READY content digest differs")
    if not _is_digest(payload.get("seal_digest")) or payload["seal_digest"] != stable_hash(seal):
        raise ResidentStoreError("resident READY seal digest differs")
    if any((
        seal["generation_id"] != content["generation_id"],
        seal["provenance_digest"] != content["provenance_digest"],
        seal["pack_contract_digest"] != content["pack_contract_digest"],
        seal["quantized_bound_contract_digest"]
        != content["quantized_bound_contract_digest"],
        seal["physical_generation_bytes"] != content["physical_generation_bytes"],
        _content_bindings(seal["source_files"]) != content["source_files"],
        _content_bindings(seal["mirror_files"]) != content["mirror_files"],
    )):
        raise ResidentStoreError("resident READY content/seal differs")
    capacity = payload.get("capacity_observation")
    if type(capacity) is not dict or set(capacity) != {"before", "after"}:
        raise ResidentStoreError("resident READY capacity fields differ")
    for stage, values in capacity.items():
        if type(values) is not dict or set(values) != _CAPACITY_KEYS:
            raise ResidentStoreError(f"resident READY capacity {stage} fields differ")
        for name in _CAPACITY_KEYS:
            _exact_int(values.get(name), f"capacity {stage}.{name}")
        if values["available_bytes"] > values["capacity_bytes"]:
            raise ResidentStoreError(f"resident READY capacity {stage} invalid")
    timings = payload.get("startup_timings")
    if type(timings) is not dict or set(timings) != _TIMING_KEYS:
        raise ResidentStoreError("resident READY timing fields differ")
    if any(
        type(value) not in {int, float} or not isfinite(value) or value < 0
        for value in timings.values()
    ):
        raise ResidentStoreError("resident READY timing value differs")
    if not isinstance(payload.get("created_at"), str):
        raise ResidentStoreError("resident READY created_at differs")
    try:
        created = datetime.fromisoformat(payload["created_at"])
    except ValueError as exc:
        raise ResidentStoreError("resident READY created_at malformed") from exc
    if created.tzinfo is None:
        raise ResidentStoreError("resident READY created_at lacks timezone")
    deterministic = {key: value for key, value in payload.items() if key != "ready_digest"}
    if not _is_digest(payload.get("ready_digest")) or (
        payload["ready_digest"] != stable_hash(deterministic)
    ):
        raise ResidentStoreError("resident READY document digest differs")
    return {
        "payload": payload, "ready_digest": payload["ready_digest"],
        "ready_file_sha256": sha256(raw).hexdigest(),
        "seal_digest": payload["seal_digest"],
        "content_digest": payload["content_digest"], "identity": identity,
    }


def resident_file_identity_lease(ready_path: Path) -> dict[str, Any]:
    """Return cheap dev/inode/size/time/mode state for worker pre/post checks."""
    observation = observe_ready_strict(ready_path)
    seal = observation["payload"]["seal"]
    mirror, store = Path(seal["mirror_root"]), Path(seal["mirror_store_root"])
    generation = store / "generations" / str(seal["generation_id"])
    paths = {
        "ready": ready_path, "mirror": mirror, "store": store,
        "generations": store / "generations", "generation": generation,
        **{f"file_{name}": Path(value["path"]) for name, value in seal["mirror_files"].items()},
    }
    deterministic = {
        "schema_version": FILE_IDENTITY_LEASE_SCHEMA_VERSION,
        "ready_digest": observation["ready_digest"],
        "ready_file_sha256": observation["ready_file_sha256"],
        "content_digest": observation["content_digest"],
        "files": {name: _file_identity(path) for name, path in paths.items()},
    }
    return {**deterministic, "lease_digest": stable_hash(deterministic)}


def _validate_ready(
    path: Path, content: Mapping[str, Any], seal: Mapping[str, Any], reserve_bytes: int,
) -> dict[str, Any]:
    payload = observe_ready_strict(path)["payload"]
    if type(payload["seal"]["reserve_bytes"]) is not int or (
        payload["seal"]["reserve_bytes"] != reserve_bytes
    ):
        raise ResidentStoreError("resident READY reserve differs from caller policy")
    if payload["content_digest"] != stable_hash(dict(content)):
        raise ResidentStoreError("resident READY content differs from verified bytes")
    if payload["seal_digest"] != stable_hash(dict(seal)):
        raise ResidentStoreError("resident READY seal differs from physical state")
    return payload


def _prepare_observed(
    source_store_root: Path, mirror_root: Path, generation_id: str, *,
    expected_provenance_digest: str | None, reserve_bytes: int,
    validate_existing: bool, mountinfo_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not _is_digest(generation_id):
        raise ResidentStoreError("generation ID must be 64 lowercase hex characters")
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ResidentStoreError("tmpfs reserve must be a nonnegative integer")
    source_store = source_store_root.resolve(strict=True)
    source_active = source_store / "active.json"
    if source_active.exists() or source_active.is_symlink():
        raise ResidentStoreError("source active.json is forbidden")
    unresolved = mirror_root.expanduser().absolute()
    ancestor = unresolved
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if _mount_binding(ancestor, mountinfo_path)["fs_type"] != "tmpfs":
        raise ResidentStoreError("mirror must be tmpfs-backed")
    unresolved.mkdir(parents=True, exist_ok=True)
    mirror = _plain_directory(unresolved)
    mount = _mount_binding(mirror, mountinfo_path)
    if mount["fs_type"] != "tmpfs":
        raise ResidentStoreError("resolved mirror is not tmpfs-backed")

    with _exclusive_lock(mirror):
        started = perf_counter()
        source_started = perf_counter()
        source = load_packed_generation(
            source_store, generation_id,
            expected_provenance_digest=expected_provenance_digest,
            verify_content=True, validate_records=False,
        )
        source_seconds = perf_counter() - source_started
        if source_active.exists() or source_active.is_symlink():
            raise ResidentStoreError("source active.json appeared during validation")
        manifest = source.manifest
        source_files = _physical_files(source_store, generation_id, manifest)
        physical_bytes = (
            int(manifest["rows_bytes"]) + int(manifest["overflow_bytes"])
            + source_files["manifest"].stat().st_size
        )
        store, ready_path = mirror / "store", mirror / "READY.json"
        if not validate_existing:
            _reclaim_abandoned_builds(store)
        capacity_before = _capacity_binding(mirror)
        required = physical_bytes + reserve_bytes
        if capacity_before["capacity_bytes"] < required:
            raise ResidentStoreError("tmpfs logical capacity is insufficient")
        if validate_existing and capacity_before["available_bytes"] < reserve_bytes:
            raise ResidentStoreError("tmpfs available capacity is below caller reserve")
        if ready_path.is_symlink():
            raise ResidentStoreError("resident READY must not be a symlink")
        if not validate_existing and ready_path.exists():
            raise ResidentStoreError("resident READY exists; refusing replacement")
        mirror_active = store / "active.json"
        if mirror_active.exists() or mirror_active.is_symlink():
            raise ResidentStoreError("mirror active.json is forbidden")
        copy_started = perf_counter()
        if validate_existing:
            generation = store / "generations" / generation_id
            if not generation.is_dir() or generation.is_symlink():
                raise ResidentStoreError("validate-existing generation is absent or linked")
        else:
            if capacity_before["available_bytes"] < required:
                raise ResidentStoreError("tmpfs available capacity is insufficient for copy")
            if store.exists() or store.is_symlink():
                _plain_directory(store, mirror)
            _copy_generation(source_store, store, generation_id, manifest)
        copy_seconds = perf_counter() - copy_started
        mirror_files = _validate_mirror_layout(
            mirror, store, generation_id, manifest, int(mount["st_dev"]),
        )
        if mirror_active.exists() or mirror_active.is_symlink():
            raise ResidentStoreError("mirror active.json appeared during validation")
        mirror_started = perf_counter()
        mirrored = load_packed_generation(
            store, generation_id,
            expected_provenance_digest=str(manifest["provenance_digest"]),
            verify_content=True, validate_records=False,
        )
        mirror_seconds = perf_counter() - mirror_started
        if mirrored.manifest != manifest:
            raise ResidentStoreError("tmpfs-backed manifest differs from source")
        source_bindings = _verified_bindings(source_files, manifest)
        mirror_bindings = _verified_bindings(mirror_files, manifest)
        if _content_bindings(source_bindings) != _content_bindings(mirror_bindings):
            raise ResidentStoreError("tmpfs-backed files differ from exact source")
        capacity_after = _capacity_binding(mirror)
        if capacity_after["available_bytes"] < reserve_bytes:
            raise ResidentStoreError("tmpfs available capacity fell below caller reserve")
        content: dict[str, Any] = {
            "schema_version": CONTENT_SCHEMA_VERSION, "generation_id": generation_id,
            "provenance_digest": str(manifest["provenance_digest"]),
            "manifest_digest": str(manifest["manifest_digest"]),
            "pack_contract_digest": str(manifest["pack_contract_digest"]),
            "quantized_bound_contract_digest": str(manifest["quantized_bound_contract_digest"]),
            "physical_generation_bytes": physical_bytes,
            "source_files": _content_bindings(source_bindings),
            "mirror_files": _content_bindings(mirror_bindings),
        }
        seal: dict[str, Any] = {
            "generation_id": generation_id,
            "provenance_digest": str(manifest["provenance_digest"]),
            "pack_contract_digest": str(manifest["pack_contract_digest"]),
            "quantized_bound_contract_digest": str(manifest["quantized_bound_contract_digest"]),
            "source_store_root": str(source_store), "mirror_root": str(mirror),
            "mirror_store_root": str(store.resolve(strict=True)),
            "source_generation_st_dev": int(
                (source_store / "generations" / generation_id).stat().st_dev
            ),
            "mirror_generation_st_dev": int((store / "generations" / generation_id).stat().st_dev),
            "source_files": source_bindings, "mirror_files": mirror_bindings,
            "resident_mount": mount, "mountinfo_path": str(mountinfo_path.absolute()),
            "resident_capacity_bytes": capacity_after["capacity_bytes"],
            "required_capacity_bytes": required, "reserve_bytes": reserve_bytes,
            "physical_generation_bytes": physical_bytes,
            "storage_class": "tmpfs-backed-generation-v1", "latency_scope": _LATENCY_SCOPE,
            "query_specific_inputs_used": False, "outcomes_or_labels_used": False,
            "real_forward_outcomes_accessed": False,
            "source_active_pointer_absent": True, "mirror_active_pointer_absent": True,
        }
        if ready_path.exists():
            ready = _validate_ready(ready_path, content, seal, reserve_bytes)
        else:
            elapsed = perf_counter() - started
            timings = {
                "source_content_verification_seconds": source_seconds,
                "mirror_copy_seconds": copy_seconds,
                "mirror_content_verification_seconds": mirror_seconds,
                "readiness_seal_seconds": max(
                    elapsed - source_seconds - copy_seconds - mirror_seconds, 0.0,
                ),
                "total_before_ready_seconds": elapsed,
            }
            deterministic: dict[str, Any] = {
                "schema_version": READY_SCHEMA_VERSION,
                "mode": "validate-existing" if validate_existing else "copy-and-validate",
                "content": content, "content_digest": stable_hash(content),
                "seal": seal, "seal_digest": stable_hash(seal),
                "capacity_observation": {"before": capacity_before, "after": capacity_after},
                "startup_timings": timings,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            ready = {**deterministic, "ready_digest": stable_hash(deterministic)}
            _atomic_json(ready_path, ready)
            ready = observe_ready_strict(ready_path)["payload"]
        observed_at = datetime.now(timezone.utc).isoformat()
        validation_deterministic: dict[str, Any] = {
            "schema_version": VALIDATION_OBSERVATION_SCHEMA_VERSION,
            "ready_digest": ready["ready_digest"], "content_digest": ready["content_digest"],
            "seal_digest": ready["seal_digest"], "reserve_bytes": reserve_bytes,
            "capacity": capacity_after, "mount": mount,
            "source_content_verification_seconds": source_seconds,
            "mirror_content_verification_seconds": mirror_seconds,
            "validation_seconds": perf_counter() - started, "observed_at": observed_at,
        }
        observation = {
            **validation_deterministic,
            "observation_digest": stable_hash(validation_deterministic),
        }
        return ready, observation


def prepare_resident_mirror_observed(
    source_store_root: Path, mirror_root: Path, generation_id: str, *,
    expected_provenance_digest: str | None = None, reserve_bytes: int = 1024 ** 3,
    validate_existing: bool = False,
    mountinfo_path: Path = Path("/proc/self/mountinfo"),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return immutable READY plus a fresh, separately digested live observation."""
    return _prepare_observed(
        source_store_root, mirror_root, generation_id,
        expected_provenance_digest=expected_provenance_digest,
        reserve_bytes=reserve_bytes, validate_existing=validate_existing,
        mountinfo_path=mountinfo_path,
    )


def prepare_resident_mirror(
    source_store_root: Path, mirror_root: Path, generation_id: str, *,
    expected_provenance_digest: str | None = None, reserve_bytes: int = 1024 ** 3,
    validate_existing: bool = False,
    mountinfo_path: Path = Path("/proc/self/mountinfo"),
) -> dict[str, Any]:
    """Prepare exact tmpfs-backed bytes without claiming unswappable RAM."""
    ready, _ = prepare_resident_mirror_observed(
        source_store_root, mirror_root, generation_id,
        expected_provenance_digest=expected_provenance_digest,
        reserve_bytes=reserve_bytes, validate_existing=validate_existing,
        mountinfo_path=mountinfo_path,
    )
    return ready
