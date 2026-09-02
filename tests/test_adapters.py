from pathlib import Path

import pandas as pd
import pytest

from market_analogues.adapters import (
    CachedOHLCVSource, DirectorySource, LongTableSource, SourceError,
)
from market_analogues.config import DatasetSpec
from market_analogues.types import InstrumentKey


def test_directory_source_common_contract(directory_dataset):
    spec = DatasetSpec("demo", "directory", directory_dataset, "parquet", timestamp_column="date")
    source = DirectorySource(spec)
    assert [x.source_symbol for x in source.instruments()] == ["AAA", "BBB"]
    frame = source.load(InstrumentKey("demo", "AAA"))
    assert list(frame.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert frame.timestamp.is_monotonic_increasing


def test_adapter_never_modifies_source(directory_dataset):
    spec = DatasetSpec("demo", "directory", directory_dataset, "parquet", timestamp_column="date")
    source = DirectorySource(spec)
    key = source.instruments()[0]
    before = source.fingerprint(key)
    source.load(key)
    assert source.fingerprint(key) == before


def test_explicit_batch_cache_reuses_frames_and_isolates_callers(directory_dataset):
    spec = DatasetSpec(
        "demo", "directory", directory_dataset, "parquet",
        timestamp_column="date",
    )
    cached = CachedOHLCVSource(DirectorySource(spec), max_entries=2)
    first = cached.load(InstrumentKey("demo", "AAA"))
    first.loc[0, "close"] = -1
    second = cached.load(InstrumentKey("demo", "AAA"))
    assert second.loc[0, "close"] != -1
    assert cached.cache_state() == {
        "entries": 1, "max_entries": 2, "hits": 1, "misses": 1,
    }
    cached.load(InstrumentKey("demo", "BBB"))
    assert cached.cache_state()["entries"] == 2
    preload = CachedOHLCVSource(DirectorySource(spec), max_entries=None)
    state = preload.preload(tuple(preload.instruments()), workers=2)
    assert state == {
        "entries": 2, "max_entries": None, "hits": 0, "misses": 2,
    }


def test_long_table_matches_directory(tmp_path, bars):
    frame = pd.concat([bars.assign(symbol="AAA"), bars.assign(symbol="BBB")])
    path = tmp_path / "long.parquet"
    frame.to_parquet(path, index=False)
    spec = DatasetSpec("demo", "long_table", path, "parquet", symbol_from="column",
                       symbol_column="symbol", timestamp_column="date")
    source = LongTableSource(spec)
    assert len(source.instruments()) == 2
    assert len(source.load(InstrumentKey("demo", "AAA"))) == len(bars)


def test_missing_column_is_rejected(directory_dataset):
    bad = pd.DataFrame({"date": ["2020-01-01"], "close": [1]})
    bad.to_parquet(directory_dataset / "BAD.parquet", index=False)
    source = DirectorySource(DatasetSpec("demo", "directory", directory_dataset, "parquet", timestamp_column="date"))
    with pytest.raises(SourceError, match="missing required"):
        source.load(InstrumentKey("demo", "BAD"))
