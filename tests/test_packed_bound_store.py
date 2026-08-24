from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from market_analogues.packed_bound_store import (
    OVERFLOW_DTYPE, PACK_DTYPE, PackedBoundStoreError,
    activate_packed_generation, decode_episode_id, load_packed_generation,
    make_overflow_record, make_packed_record, packed_bound_store_contract,
    packed_lower_bounds,
    unpack_quantized_rows, write_packed_generation,
)
from market_analogues.quantized_bound import (
    PACKED_ROW_BYTES, quantize_bound_row, quantized_representation_lower_bound,
)
from market_analogues.representation import represent
from market_analogues.synthetic import generate_case
from market_analogues.types import stable_hash


def _records(seed: int, symbol_id: int = 0) -> tuple[np.ndarray, object]:
    representation = represent(generate_case("rounded_base", seed).episode)
    row = quantize_bound_row(representation)
    record = make_packed_record(
        f"{seed:024x}", 1_600_000_000_000_000_000 + seed,
        symbol_id, "A", row,
    )
    return record, representation


def test_packed_contract_round_trip_preserves_bound_exactly() -> None:
    first, query = _records(1)
    second, _ = _records(2)
    records = np.concatenate((first, second))
    restored = unpack_quantized_rows(records)
    assert PACK_DTYPE.itemsize == PACKED_ROW_BYTES == 2432
    assert packed_bound_store_contract()["row_bytes"] == 2432
    assert decode_episode_id(records[0]["episode_id"]) == f"{1:024x}"
    native = quantized_representation_lower_bound(query, quantize_bound_row(query))
    packed = quantized_representation_lower_bound(query, restored[0])
    assert packed.total == native.total
    assert packed.components == native.components
    vector = packed_lower_bounds(query, records)
    for index, row in enumerate(restored):
        scalar = quantized_representation_lower_bound(query, row)
        assert vector.totals[index] == scalar.total


def test_generation_is_deterministic_resumable_and_atomically_rollbackable(
    tmp_path: Path,
) -> None:
    first, _ = _records(1)
    overflow = make_overflow_record(f"{2:024x}", 2, 0, "B")
    provenance = {"stocks": {"AAA": "digest"}, "benchmark": "digest"}
    generation_one = write_packed_generation(
        tmp_path, first, overflow, ("AAA",), provenance,
    )
    repeated = write_packed_generation(
        tmp_path, first.copy(), overflow.copy(), ("AAA",), provenance,
    )
    assert repeated == generation_one
    loaded = load_packed_generation(tmp_path)
    assert loaded.generation_id == generation_one
    assert len(loaded.rows) == 1 and len(loaded.overflow) == 1
    assert loaded.manifest["eligible_row_count"] == 2
    assert load_packed_generation(
        tmp_path, expected_provenance_digest=stable_hash(provenance),
    ).generation_id == generation_one
    with pytest.raises(PackedBoundStoreError, match="provenance is stale"):
        load_packed_generation(
            tmp_path, expected_provenance_digest=stable_hash({"revision": 99}),
        )

    second, _ = _records(3)
    generation_two = write_packed_generation(
        tmp_path, second, np.empty(0, dtype=OVERFLOW_DTYPE),
        ("AAA",), {"revision": 2},
    )
    assert generation_two != generation_one
    assert load_packed_generation(tmp_path).generation_id == generation_two
    activate_packed_generation(tmp_path, generation_one)
    assert load_packed_generation(tmp_path).generation_id == generation_one


@pytest.mark.parametrize(
    "target", ["bound-rows.bin", "missing-pack", "manifest.json", "active.json"],
)
def test_generation_rejects_corruption_truncation_and_manifest_tamper(
    tmp_path: Path, target: str,
) -> None:
    record, _ = _records(4)
    generation = write_packed_generation(
        tmp_path, record, np.empty(0, dtype=OVERFLOW_DTYPE),
        ("AAA",), {"revision": 1},
    )
    if target == "missing-pack":
        path = tmp_path / "generations" / generation / "bound-rows.bin"
        path.unlink()
    elif target == "active.json":
        path = tmp_path / target
        payload = json.loads(path.read_text())
        payload["generation_id"] = "0" * 64
        path.write_text(json.dumps(payload))
    elif target == "manifest.json":
        path = tmp_path / "generations" / generation / target
        payload = json.loads(path.read_text())
        payload["row_count"] += 1
        path.write_text(json.dumps(payload))
    else:
        path = tmp_path / "generations" / generation / target
        path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(PackedBoundStoreError):
        load_packed_generation(tmp_path)


def test_incremental_generation_equals_clean_rebuild(tmp_path: Path) -> None:
    first, _ = _records(10)
    second, _ = _records(11)
    provenance = {"prefix": "after-append"}
    incremental_root = tmp_path / "incremental"
    clean_root = tmp_path / "clean"
    write_packed_generation(
        incremental_root, first, np.empty(0, dtype=OVERFLOW_DTYPE),
        ("AAA",), {"prefix": "before-append"},
    )
    incremental = write_packed_generation(
        incremental_root, np.concatenate((first, second)),
        np.empty(0, dtype=OVERFLOW_DTYPE), ("AAA",), provenance,
    )
    clean = write_packed_generation(
        clean_root, np.concatenate((first, second)),
        np.empty(0, dtype=OVERFLOW_DTYPE), ("AAA",), provenance,
    )
    assert incremental == clean
    assert load_packed_generation(incremental_root).manifest == (
        load_packed_generation(clean_root).manifest
    )


def test_generation_rejects_duplicates_and_noncanonical_order(tmp_path: Path) -> None:
    record, _ = _records(5)
    duplicate = record.copy()
    duplicate["cutoff_ns"][0] += 1
    with pytest.raises(PackedBoundStoreError, match="duplicate episode"):
        write_packed_generation(
            tmp_path, np.concatenate((record, duplicate)),
            np.empty(0, dtype=OVERFLOW_DTYPE), ("AAA",), {},
        )
    later = record.copy()
    later["episode_id"][0] = np.void(bytes.fromhex(f"{6:024x}"))
    later["cutoff_ns"][0] = record["cutoff_ns"][0] - 1
    with pytest.raises(PackedBoundStoreError, match="ascending"):
        write_packed_generation(
            tmp_path, np.concatenate((record, later)),
            np.empty(0, dtype=OVERFLOW_DTYPE), ("AAA",), {},
        )
