from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

import market_analogues.resident_store as resident_store

from market_analogues.packed_bound_store import (
    OVERFLOW_DTYPE, PackedBoundStoreError, make_packed_record,
    write_packed_generation,
)
from market_analogues.quantized_bound import quantize_bound_row
from market_analogues.representation import represent
from market_analogues.resident_store import (
    CONTENT_SCHEMA_VERSION, READY_SCHEMA_VERSION, ResidentStoreError,
    _mount_binding, observe_ready_strict, prepare_resident_mirror,
    prepare_resident_mirror_observed, resident_file_identity_lease,
)
from market_analogues.synthetic import generate_case
from market_analogues.types import stable_hash


def _generation(root: Path) -> tuple[str, str]:
    representation = represent(generate_case("rounded_base", 771).episode)
    record = make_packed_record(
        f"{771:024x}", 1_700_000_000_000_000_000, 0, "A",
        quantize_bound_row(representation),
    )
    provenance = {"source": "fixture", "prefix": "sealed"}
    generation = write_packed_generation(
        root, record, np.empty(0, dtype=OVERFLOW_DTYPE),
        ("AAA",), provenance, activate=False,
    )
    return generation, stable_hash(provenance)


def _escape_mount_path(path: Path) -> str:
    return str(path).replace("\\", "\\134").replace(" ", "\\040")


def _mountinfo(path: Path, mount_point: Path, fs_type: str = "tmpfs") -> Path:
    device = mount_point.stat().st_dev
    value = path / "mountinfo"
    value.write_text(
        f"91 22 {os.major(device)}:{os.minor(device)} / "
        f"{_escape_mount_path(mount_point.resolve())} rw,nosuid - "
        f"{fs_type} {fs_type} rw,size=1g\n"
    )
    return value


def test_mount_binding_uses_decoded_longest_containing_mount(tmp_path: Path) -> None:
    mount = tmp_path / "resident mount"
    nested = mount / "nested"
    nested.mkdir(parents=True)
    device = mount.stat().st_dev
    info = tmp_path / "mountinfo"
    info.write_text(
        f"1 0 {os.major(device)}:{os.minor(device)} / / rw - ext4 /dev/root rw\n"
        f"2 1 {os.major(device)}:{os.minor(device)} / {_escape_mount_path(mount)} "
        "rw,nosuid shared:4 - tmpfs tmpfs rw,size=1g\n"
    )
    binding = _mount_binding(nested, info)
    assert binding["fs_type"] == "tmpfs"
    assert binding["mount_point"] == str(mount.resolve())
    assert binding["optional_fields"] == ["shared:4"]


def test_copy_publishes_exact_ready_document_last(tmp_path: Path) -> None:
    source = tmp_path / "source"
    mirror_parent = tmp_path / "resident"
    mirror_parent.mkdir()
    generation, provenance = _generation(source)
    info = _mountinfo(tmp_path, mirror_parent)
    mirror = mirror_parent / "mirror"
    result = prepare_resident_mirror(
        source, mirror, generation,
        expected_provenance_digest=provenance, reserve_bytes=0,
        mountinfo_path=info,
    )
    ready = mirror / "READY.json"
    assert ready.is_file()
    assert json.loads(ready.read_text()) == result
    assert result["schema_version"] == READY_SCHEMA_VERSION
    assert result["mode"] == "copy-and-validate"
    assert result["seal"]["storage_class"] == "tmpfs-backed-generation-v1"
    assert "not guaranteed unswappable RAM" in result["seal"]["latency_scope"]
    assert result["content"]["schema_version"] == CONTENT_SCHEMA_VERSION
    assert result["content_digest"] == stable_hash(result["content"])
    assert not set(result["content"]) & {
        "startup_timings", "capacity_observation", "source_store_root",
        "mirror_store_root", "resident_mount",
    }
    assert result["seal"]["mountinfo_path"] == str(info.absolute())
    assert not result["seal"]["query_specific_inputs_used"]
    assert not result["seal"]["outcomes_or_labels_used"]
    for name in ("manifest", "rows", "overflow"):
        source_file = result["seal"]["source_files"][name]
        mirror_file = result["seal"]["mirror_files"][name]
        assert source_file["bytes"] == mirror_file["bytes"]
        assert source_file["sha256"] == mirror_file["sha256"]
        assert Path(source_file["path"]).is_absolute()
        assert Path(mirror_file["path"]).is_absolute()
    assert set(result["startup_timings"]) == {
        "source_content_verification_seconds", "mirror_copy_seconds",
        "mirror_content_verification_seconds", "readiness_seal_seconds",
        "total_before_ready_seconds",
    }


def test_validate_existing_seals_without_recopy(tmp_path: Path) -> None:
    source = tmp_path / "source"
    mirror_parent = tmp_path / "resident"
    mirror_parent.mkdir()
    generation, provenance = _generation(source)
    info = _mountinfo(tmp_path, mirror_parent)
    mirror = mirror_parent / "mirror"
    destination = mirror / "store" / "generations" / generation
    destination.parent.mkdir(parents=True)
    shutil.copytree(source / "generations" / generation, destination)
    rows = destination / "bound-rows.bin"
    original = rows.stat().st_mtime_ns
    result = prepare_resident_mirror(
        source, mirror, generation,
        expected_provenance_digest=provenance, reserve_bytes=0,
        validate_existing=True, mountinfo_path=info,
    )
    assert result["mode"] == "validate-existing"
    assert result["startup_timings"]["mirror_copy_seconds"] < 0.1
    assert rows.stat().st_mtime_ns == original


def test_copy_reclaims_only_valid_abandoned_staging_directory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    resident = tmp_path / "resident"
    resident.mkdir()
    generation, provenance = _generation(source)
    info = _mountinfo(tmp_path, resident)
    mirror = resident / "mirror"
    abandoned = mirror / ".building" / ("a" * 32)
    abandoned.mkdir(parents=True)
    (abandoned / "partial.bin").write_bytes(b"partial")
    result = prepare_resident_mirror(
        source, mirror, generation,
        expected_provenance_digest=provenance, reserve_bytes=0,
        mountinfo_path=info,
    )
    assert result["mode"] == "copy-and-validate"
    assert not abandoned.exists()
    assert (mirror / "READY.json").is_file()


def test_copy_refuses_unrecognized_staging_instead_of_deleting_it(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    resident = tmp_path / "resident"
    resident.mkdir()
    generation, provenance = _generation(source)
    info = _mountinfo(tmp_path, resident)
    mirror = resident / "mirror"
    unexpected = mirror / ".building" / "do-not-delete"
    unexpected.mkdir(parents=True)
    marker = unexpected / "marker"
    marker.write_text("preserve")
    with pytest.raises(ResidentStoreError, match="unexpected entry"):
        prepare_resident_mirror(
            source, mirror, generation,
            expected_provenance_digest=provenance, reserve_bytes=0,
            mountinfo_path=info,
        )
    assert marker.read_text() == "preserve"


@pytest.mark.parametrize(
    "failure", ["non_tmpfs", "capacity", "provenance", "tamper"],
)
def test_failures_never_publish_ready(tmp_path: Path, failure: str) -> None:
    source = tmp_path / "source"
    mirror_parent = tmp_path / "resident"
    mirror_parent.mkdir()
    generation, provenance = _generation(source)
    info = _mountinfo(
        tmp_path, mirror_parent, "ext4" if failure == "non_tmpfs" else "tmpfs",
    )
    mirror = mirror_parent / "mirror"
    kwargs = {
        "expected_provenance_digest": provenance,
        "reserve_bytes": 0,
        "mountinfo_path": info,
    }
    if failure == "capacity":
        kwargs["reserve_bytes"] = 1 << 80
    if failure == "provenance":
        kwargs["expected_provenance_digest"] = "0" * 64
    if failure == "tamper":
        destination = mirror / "store" / "generations" / generation
        destination.parent.mkdir(parents=True)
        shutil.copytree(source / "generations" / generation, destination)
        rows = destination / "bound-rows.bin"
        rows.write_bytes(rows.read_bytes()[:-1])
        kwargs["validate_existing"] = True
    expected_error = (
        PackedBoundStoreError
        if failure in {"tamper", "provenance"} else ResidentStoreError
    )
    with pytest.raises(expected_error):
        prepare_resident_mirror(source, mirror, generation, **kwargs)
    assert not (mirror / "READY.json").exists()


def test_existing_ready_is_revalidated_and_drift_fails(tmp_path: Path) -> None:
    source = tmp_path / "source"
    mirror_parent = tmp_path / "resident"
    mirror_parent.mkdir()
    generation, provenance = _generation(source)
    info = _mountinfo(tmp_path, mirror_parent)
    mirror = mirror_parent / "mirror"
    first = prepare_resident_mirror(
        source, mirror, generation,
        expected_provenance_digest=provenance, reserve_bytes=0,
        mountinfo_path=info,
    )
    repeated = prepare_resident_mirror(
        source, mirror, generation,
        expected_provenance_digest=provenance, reserve_bytes=0,
        validate_existing=True, mountinfo_path=info,
    )
    assert repeated == first
    ready = mirror / "READY.json"
    payload = json.loads(ready.read_text())
    payload["seal"]["storage_class"] = "drifted"
    ready.write_text(json.dumps(payload))
    with pytest.raises(ResidentStoreError):
        prepare_resident_mirror(
            source, mirror, generation,
            expected_provenance_digest=provenance, reserve_bytes=0,
            validate_existing=True, mountinfo_path=info,
        )


def test_fresh_validation_observation_is_separate_from_immutable_ready(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    resident = tmp_path / "resident"
    resident.mkdir()
    generation, provenance = _generation(source)
    info = _mountinfo(tmp_path, resident)
    mirror = resident / "mirror"
    ready, first = prepare_resident_mirror_observed(
        source, mirror, generation, expected_provenance_digest=provenance,
        reserve_bytes=0, mountinfo_path=info,
    )
    repeated, second = prepare_resident_mirror_observed(
        source, mirror, generation, expected_provenance_digest=provenance,
        reserve_bytes=0, validate_existing=True, mountinfo_path=info,
    )
    assert repeated == ready
    assert first["ready_digest"] == second["ready_digest"] == ready["ready_digest"]
    assert first["content_digest"] == second["content_digest"]
    assert first["observation_digest"] == stable_hash({
        key: value for key, value in first.items() if key != "observation_digest"
    })
    assert second["observed_at"] >= first["observed_at"]


@pytest.mark.parametrize("mutation", ["extra", "bool_as_int", "mode"])
def test_strict_single_read_ready_rejects_shape_type_and_mode_drift(
    tmp_path: Path, mutation: str,
) -> None:
    source = tmp_path / "source"
    resident = tmp_path / "resident"
    resident.mkdir()
    generation, provenance = _generation(source)
    info = _mountinfo(tmp_path, resident)
    mirror = resident / "mirror"
    prepare_resident_mirror(
        source, mirror, generation, expected_provenance_digest=provenance,
        reserve_bytes=0, mountinfo_path=info,
    )
    ready_path = mirror / "READY.json"
    payload = json.loads(ready_path.read_text())
    if mutation == "extra":
        payload["unexpected"] = True
    elif mutation == "bool_as_int":
        payload["capacity_observation"]["before"]["block_bytes"] = False
    else:
        payload["mode"] = "invented"
    deterministic = {
        key: value for key, value in payload.items() if key != "ready_digest"
    }
    payload["ready_digest"] = stable_hash(deterministic)
    ready_path.write_text(json.dumps(payload))
    with pytest.raises(ResidentStoreError):
        observe_ready_strict(ready_path)


def test_ready_symlink_and_linked_generation_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    resident = tmp_path / "resident"
    resident.mkdir()
    generation, provenance = _generation(source)
    info = _mountinfo(tmp_path, resident)

    mirror = resident / "linked-generation"
    generation_link = mirror / "store" / "generations" / generation
    generation_link.parent.mkdir(parents=True)
    generation_link.symlink_to(source / "generations" / generation, target_is_directory=True)
    with pytest.raises(ResidentStoreError, match="linked"):
        prepare_resident_mirror(
            source, mirror, generation, expected_provenance_digest=provenance,
            reserve_bytes=0, validate_existing=True, mountinfo_path=info,
        )

    valid = resident / "valid"
    prepare_resident_mirror(
        source, valid, generation, expected_provenance_digest=provenance,
        reserve_bytes=0, mountinfo_path=info,
    )
    ready = valid / "READY.json"
    external = resident / "external-ready.json"
    ready.rename(external)
    ready.symlink_to(external)
    with pytest.raises(ResidentStoreError, match="symlink"):
        observe_ready_strict(ready)


def test_store_and_row_symlink_escapes_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    resident = tmp_path / "resident"
    resident.mkdir()
    generation, provenance = _generation(source)
    info = _mountinfo(tmp_path, resident)

    linked_store = resident / "linked-store"
    linked_store.mkdir()
    (linked_store / "store").symlink_to(source, target_is_directory=True)
    with pytest.raises(ResidentStoreError):
        prepare_resident_mirror(
            source, linked_store, generation,
            expected_provenance_digest=provenance, reserve_bytes=0,
            validate_existing=True, mountinfo_path=info,
        )

    linked_row = resident / "linked-row"
    destination = linked_row / "store" / "generations" / generation
    destination.parent.mkdir(parents=True)
    shutil.copytree(source / "generations" / generation, destination)
    rows = destination / "bound-rows.bin"
    rows.unlink()
    rows.symlink_to(source / "generations" / generation / "bound-rows.bin")
    with pytest.raises(ResidentStoreError, match="plain regular"):
        prepare_resident_mirror(
            source, linked_row, generation,
            expected_provenance_digest=provenance, reserve_bytes=0,
            validate_existing=True, mountinfo_path=info,
        )


def test_active_pointers_and_reserve_drift_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    resident = tmp_path / "resident"
    resident.mkdir()
    generation, provenance = _generation(source)
    info = _mountinfo(tmp_path, resident)
    mirror = resident / "mirror"
    prepare_resident_mirror(
        source, mirror, generation, expected_provenance_digest=provenance,
        reserve_bytes=0, mountinfo_path=info,
    )
    with pytest.raises(ResidentStoreError, match="reserve"):
        prepare_resident_mirror(
            source, mirror, generation, expected_provenance_digest=provenance,
            reserve_bytes=1, validate_existing=True, mountinfo_path=info,
        )
    (mirror / "store" / "active.json").write_text("{}")
    with pytest.raises(ResidentStoreError, match="mirror active"):
        prepare_resident_mirror(
            source, mirror, generation, expected_provenance_digest=provenance,
            reserve_bytes=0, validate_existing=True, mountinfo_path=info,
        )
    (mirror / "store" / "active.json").unlink()
    (source / "active.json").write_text("{}")
    with pytest.raises(ResidentStoreError, match="source active"):
        prepare_resident_mirror(
            source, mirror, generation, expected_provenance_digest=provenance,
            reserve_bytes=0, validate_existing=True, mountinfo_path=info,
        )


def test_file_identity_lease_detects_in_place_mutation(tmp_path: Path) -> None:
    source = tmp_path / "source"
    resident = tmp_path / "resident"
    resident.mkdir()
    generation, provenance = _generation(source)
    info = _mountinfo(tmp_path, resident)
    mirror = resident / "mirror"
    prepare_resident_mirror(
        source, mirror, generation, expected_provenance_digest=provenance,
        reserve_bytes=0, mountinfo_path=info,
    )
    ready = mirror / "READY.json"
    before = resident_file_identity_lease(ready)
    rows = mirror / "store" / "generations" / generation / "bound-rows.bin"
    original = rows.read_bytes()
    rows.write_bytes(original)
    after = resident_file_identity_lease(ready)
    assert before["lease_digest"] != after["lease_digest"]
    assert before["files"]["file_rows"]["st_ctime_ns"] <= (
        after["files"]["file_rows"]["st_ctime_ns"]
    )


def test_single_read_observation_rejects_identity_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready = tmp_path / "READY.json"
    ready.write_text("{}")
    actual_fstat = os.fstat
    calls = 0

    def changing_fstat(descriptor: int):
        nonlocal calls
        calls += 1
        observed = actual_fstat(descriptor)
        if calls == 1:
            return observed
        values = {
            name: getattr(observed, name) for name in (
                "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns",
                "st_mode",
            )
        }
        values["st_ctime_ns"] += 1
        return SimpleNamespace(**values)

    monkeypatch.setattr(resident_store.os, "fstat", changing_fstat)
    with pytest.raises(ResidentStoreError, match="changed during single read"):
        observe_ready_strict(ready)
