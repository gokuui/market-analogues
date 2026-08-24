from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from market_analogues.adapters import DirectorySource, file_fingerprint
from market_analogues.causal_prefix import (
    CausalPrefixError, causal_prefix_digest, prefix_generation_digest,
)
from market_analogues.config import BenchmarkSpec, DatasetSpec
from market_analogues.types import InstrumentKey


def _canonical(bars: pd.DataFrame) -> pd.DataFrame:
    return bars.rename(columns={"date": "timestamp"}).reset_index(drop=True)


def test_prefix_digest_survives_rewrite_and_future_append(
    bars: pd.DataFrame, tmp_path: Path,
) -> None:
    path = tmp_path / "AAA.parquet"
    bars.to_parquet(path, index=False, compression="snappy")
    source = DirectorySource(DatasetSpec(
        "test", "directory", tmp_path, "parquet", timestamp_column="date",
    ))
    key = InstrumentKey("test", "AAA")
    cutoff = bars.date.iloc[-11]
    before_file = file_fingerprint(path)
    before = source.causal_prefix_fingerprint(key, cutoff)

    bars.astype({"volume": "float64"}).to_parquet(
        path, index=False, compression="gzip",
    )
    rewritten = DirectorySource(source.spec)
    assert file_fingerprint(path) != before_file
    assert rewritten.causal_prefix_fingerprint(key, cutoff).digest == before.digest

    appended = pd.concat([bars, pd.DataFrame([{
        **bars.iloc[-1].to_dict(),
        "date": pd.Timestamp(bars.date.iloc[-1]) + pd.offsets.BDay(),
        "close": float(bars.close.iloc[-1]) * 1.01,
    }])], ignore_index=True)
    appended.to_parquet(path, index=False)
    appended_source = DirectorySource(source.spec)
    assert appended_source.causal_prefix_fingerprint(key, cutoff).digest == before.digest


def test_prefix_digest_rejects_every_historical_change(bars: pd.DataFrame) -> None:
    frame = _canonical(bars.iloc[:120].copy())
    cutoff = frame.timestamp.iloc[-1]
    expected = causal_prefix_digest(frame, cutoff).digest
    revised = frame.copy()
    revised.loc[20, "close"] *= 1.01
    assert causal_prefix_digest(revised, cutoff).digest != expected
    deleted = frame.drop(index=20).reset_index(drop=True)
    assert causal_prefix_digest(deleted, cutoff).digest != expected
    inserted = pd.concat([
        frame.iloc[:20],
        frame.iloc[[19]].assign(
            timestamp=pd.Timestamp(frame.timestamp.iloc[19]) + pd.Timedelta(hours=1),
        ),
        frame.iloc[20:],
    ], ignore_index=True)
    assert causal_prefix_digest(inserted, cutoff).digest != expected


def test_prefix_digest_rejects_duplicate_and_reordered_timestamps(
    bars: pd.DataFrame,
) -> None:
    frame = _canonical(bars.iloc[:100].copy())
    duplicate = frame.copy()
    duplicate.loc[20, "timestamp"] = duplicate.loc[19, "timestamp"]
    with pytest.raises(CausalPrefixError, match="duplicate"):
        causal_prefix_digest(duplicate, frame.timestamp.iloc[-1])
    reordered = frame.copy()
    reordered.iloc[[20, 21]] = reordered.iloc[[21, 20]].to_numpy()
    with pytest.raises(CausalPrefixError, match="strictly ordered"):
        causal_prefix_digest(reordered, frame.timestamp.iloc[-1])


def test_source_records_raw_reordering_instead_of_hiding_it(
    bars: pd.DataFrame, tmp_path: Path,
) -> None:
    reordered = bars.iloc[:100].copy()
    reordered.iloc[[20, 21]] = reordered.iloc[[21, 20]].to_numpy()
    reordered.to_parquet(tmp_path / "AAA.parquet", index=False)
    source = DirectorySource(DatasetSpec(
        "test", "directory", tmp_path, "parquet", timestamp_column="date",
    ))
    with pytest.raises(CausalPrefixError, match="reordered during canonicalization"):
        source.causal_prefix_fingerprint(
            InstrumentKey("test", "AAA"), bars.date.iloc[99],
        )


def test_benchmark_prefix_obeys_same_append_and_revision_rules(
    bars: pd.DataFrame, tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    bars.to_parquet(data / "AAA.parquet", index=False)
    benchmark_path = tmp_path / "benchmark.parquet"
    bars.to_parquet(benchmark_path, index=False)
    spec = DatasetSpec(
        "test", "directory", data, "parquet", timestamp_column="date",
        benchmark=BenchmarkSpec(benchmark_path),
    )
    cutoff = bars.date.iloc[-11]
    expected = DirectorySource(spec).benchmark_causal_prefix_fingerprint(cutoff)
    assert expected is not None
    future_changed = bars.copy()
    future_changed.loc[future_changed.date > cutoff, "close"] *= 2
    future_changed.to_parquet(benchmark_path, index=False)
    assert DirectorySource(spec).benchmark_causal_prefix_fingerprint(cutoff).digest == expected.digest
    historical_changed = bars.copy()
    historical_changed.loc[historical_changed.date <= cutoff, "close"] *= 1.001
    historical_changed.to_parquet(benchmark_path, index=False)
    assert DirectorySource(spec).benchmark_causal_prefix_fingerprint(cutoff).digest != expected.digest


def test_generation_changes_for_new_query_but_not_old_prefix_append(
    bars: pd.DataFrame,
) -> None:
    frame = _canonical(bars.iloc[:100].copy())
    old_cutoff = frame.timestamp.iloc[-1]
    old_prefix = causal_prefix_digest(frame, old_cutoff)
    old_generation = prefix_generation_digest(
        dataset_id="test", query_cutoff=old_cutoff,
        representation_version="dense-v1", stock_prefixes={"AAA": old_prefix},
        benchmark_prefix=None,
    )
    appended = pd.concat([frame, pd.DataFrame([{
        **frame.iloc[-1].to_dict(),
        "timestamp": pd.Timestamp(old_cutoff) + pd.offsets.BDay(),
    }])], ignore_index=True)
    assert causal_prefix_digest(appended, old_cutoff).digest == old_prefix.digest
    new_cutoff = appended.timestamp.iloc[-1]
    new_prefix = causal_prefix_digest(appended, new_cutoff)
    new_generation = prefix_generation_digest(
        dataset_id="test", query_cutoff=new_cutoff,
        representation_version="dense-v1", stock_prefixes={"AAA": new_prefix},
        benchmark_prefix=None,
    )
    assert new_generation != old_generation
