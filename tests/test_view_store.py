from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_analogues.adapters import DirectorySource
from market_analogues.config import DatasetSpec
from market_analogues.view_signatures import SIGNATURE_DIMENSIONS
from market_analogues.view_store import (
    ViewShardError, build_view_store, load_view_shard,
)


def _source(path: Path) -> DirectorySource:
    return DirectorySource(DatasetSpec(
        "test", "directory", path, "parquet", timestamp_column="date",
    ))


def _quality() -> pd.DataFrame:
    return pd.DataFrame({"symbol": ["AAA", "BBB"], "tier": ["A", "B"], "issues": ["", ""]})


def test_view_store_round_trip_resume_and_parallel_digest(
    directory_dataset, tmp_path,
) -> None:
    serial_root = tmp_path / "serial"
    parallel_root = tmp_path / "parallel"
    first = build_view_store(
        _source(directory_dataset), _quality(), serial_root,
        lookbacks=(63,), stride=20, workers=1,
    )
    assert first.passed
    assert first.instruments_built == 2
    assert first.instruments_reused == 0
    assert first.rows > 0
    resumed = build_view_store(
        _source(directory_dataset), _quality(), serial_root,
        lookbacks=(63,), stride=20, workers=2,
    )
    assert resumed.passed
    assert resumed.instruments_built == 0
    assert resumed.instruments_reused == 2
    assert resumed.manifest_digest == first.manifest_digest
    parallel = build_view_store(
        _source(directory_dataset), _quality(), parallel_root,
        lookbacks=(63,), stride=20, workers=2,
    )
    assert parallel.manifest_digest == first.manifest_digest
    shards = sorted(serial_root.glob("**/*.npz"))
    loaded = load_view_shard(shards[0])
    assert loaded.signatures.shape[1] == SIGNATURE_DIMENSIONS
    assert loaded.metadata.storage_dtype == "float16"
    assert loaded.metadata.content_digest


def test_view_store_detects_corruption(directory_dataset, tmp_path) -> None:
    root = tmp_path / "store"
    build_view_store(
        _source(directory_dataset), _quality(), root,
        lookbacks=(126,), stride=50, instrument_limit=1,
    )
    shard = next(root.glob("**/*.npz"))
    payload = bytearray(shard.read_bytes())
    payload[len(payload) // 2] ^= 0xFF
    shard.write_bytes(payload)
    with pytest.raises(ViewShardError):
        load_view_shard(shard)
    failed = build_view_store(
        _source(directory_dataset), _quality(), root,
        lookbacks=(126,), stride=50, instrument_limit=1,
    )
    assert not failed.passed
    rebuilt = build_view_store(
        _source(directory_dataset), _quality(), root,
        lookbacks=(126,), stride=50, instrument_limit=1,
        rebuild_invalid=True,
    )
    assert rebuilt.passed
    assert np.isfinite(load_view_shard(shard).signatures).all()


def test_view_store_manifest_merges_independently_built_lookbacks(
    directory_dataset, tmp_path,
) -> None:
    root = tmp_path / "layers"
    first = build_view_store(
        _source(directory_dataset), _quality(), root,
        lookbacks=(63,), stride=10, instrument_limit=1,
    )
    second = build_view_store(
        _source(directory_dataset), _quality(), root,
        lookbacks=(126,), stride=10, instrument_limit=1,
    )
    manifest = pd.read_json(second.manifest_path, typ="series")
    assert manifest.lookbacks == [63, 126]
    assert len(manifest.shards) == 2
    assert first.manifest_digest != second.manifest_digest
