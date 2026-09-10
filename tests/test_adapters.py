from pathlib import Path

import pandas as pd
import pytest

from market_analogues.adapters import (
    CachedOHLCVSource, DirectorySource, LongTableSource,
    PrefixLockedOHLCVSource, SourceError,
)
from market_analogues.config import DatasetSpec
from market_analogues.config import BenchmarkSpec
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


def test_long_table_supports_external_benchmark_and_rejects_foreign_key(
    tmp_path, bars,
):
    frame = pd.concat([bars.assign(symbol="AAA"), bars.assign(symbol="BBB")])
    path = tmp_path / "long.parquet"
    benchmark_path = tmp_path / "benchmark.parquet"
    frame.to_parquet(path, index=False)
    bars.assign(close=bars.close * 1.01).to_parquet(benchmark_path, index=False)
    spec = DatasetSpec(
        "demo", "long_table", path, "parquet", symbol_from="column",
        symbol_column="symbol", timestamp_column="date",
        benchmark=BenchmarkSpec(path=benchmark_path),
    )
    source = LongTableSource(spec)
    benchmark = source.load_benchmark()
    assert benchmark is not None
    assert list(benchmark.columns) == [
        "timestamp", "open", "high", "low", "close", "volume",
    ]
    assert source.benchmark_fingerprint()
    with pytest.raises(KeyError):
        source.load(InstrumentKey("foreign", "AAA"))


def test_missing_column_is_rejected(directory_dataset):
    bad = pd.DataFrame({"date": ["2020-01-01"], "close": [1]})
    bad.to_parquet(directory_dataset / "BAD.parquet", index=False)
    source = DirectorySource(DatasetSpec("demo", "directory", directory_dataset, "parquet", timestamp_column="date"))
    with pytest.raises(SourceError, match="missing required"):
        source.load(InstrumentKey("demo", "BAD"))


def test_prefix_locked_source_ignores_future_append_and_rejects_history_revision(
    tmp_path, bars,
):
    root = tmp_path / "data"
    root.mkdir()
    path = root / "AAA.parquet"
    benchmark_path = tmp_path / "benchmark.parquet"
    bars.to_parquet(path, index=False)
    bars.to_parquet(benchmark_path, index=False)
    spec = DatasetSpec(
        "demo", "directory", root, "parquet", timestamp_column="date",
        benchmark=BenchmarkSpec(benchmark_path),
    )
    cutoff = bars.date.iloc[-11]
    locked = PrefixLockedOHLCVSource(DirectorySource(spec), cutoff)
    key = InstrumentKey("demo", "AAA")
    stock_digest = locked.fingerprint(key)
    benchmark_digest = locked.benchmark_fingerprint()
    assert locked.load(key).timestamp.max() == cutoff

    appended = pd.concat([bars, pd.DataFrame([{
        **bars.iloc[-1].to_dict(),
        "date": pd.Timestamp(bars.date.iloc[-1]) + pd.offsets.BDay(),
    }])], ignore_index=True)
    appended.to_parquet(path, index=False)
    appended.to_parquet(benchmark_path, index=False)
    appended_lock = PrefixLockedOHLCVSource(DirectorySource(spec), cutoff)
    assert appended_lock.fingerprint(key) == stock_digest
    assert appended_lock.benchmark_fingerprint() == benchmark_digest

    revised = appended.copy()
    revised.loc[20, "close"] *= 1.01
    revised.to_parquet(path, index=False)
    assert PrefixLockedOHLCVSource(DirectorySource(spec), cutoff).fingerprint(key) != stock_digest
    with pytest.raises(SourceError, match="exceeds"):
        appended_lock.causal_prefix_fingerprint(
            key, pd.Timestamp(cutoff) + pd.offsets.BDay(),
        )
