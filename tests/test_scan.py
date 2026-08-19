from __future__ import annotations

import numpy as np
import pandas as pd

from market_analogues.adapters import DirectorySource
from market_analogues.candidate_views import VIEW_NAMES
from market_analogues.config import DatasetSpec
from market_analogues.episodes import build_episode
from market_analogues.scan import (
    _quality_issues, scan_frame, scan_frame_multiview, scan_universe_candidates,
)
from market_analogues.synthetic import generate_case, transform_case
from market_analogues.types import InstrumentKey, SearchQuery


def test_vectorized_scan_recovers_inserted_historical_path() -> None:
    base = generate_case("trend_contraction_breakout", 4, n=126)
    query = transform_case(base, name="later", price_scale=8, time_shift_days=3000).episode
    rng = np.random.default_rng(9)
    prefix_close = 80 * np.exp(np.cumsum(rng.normal(0, .02, 80)))
    target = base.episode.bars.copy()
    ratio = prefix_close[-1] / target.close.iloc[0]
    target[["open", "high", "low", "close"]] *= ratio
    prefix = pd.DataFrame({
        "timestamp": pd.date_range("1990-01-01", periods=80, freq="B"),
        "open": prefix_close, "high": prefix_close * 1.01,
        "low": prefix_close * .99, "close": prefix_close,
        "volume": np.full(80, 100_000),
    })
    target["timestamp"] = pd.date_range(prefix.timestamp.iloc[-1] + pd.offsets.BDay(), periods=len(target), freq="B")
    candidate = pd.concat([prefix, target], ignore_index=True)
    hits, windows = scan_frame(
        query, candidate, InstrumentKey("test", "candidate"), stride=1, per_instrument=3,
    )
    assert windows == len(candidate) - query.key.lookback + 1
    assert hits[0].cutoff == target.timestamp.iloc[-1]
    assert hits[0].distance == pytest.approx(0, abs=1e-10)


def test_multiview_scan_recovers_inserted_historical_path() -> None:
    base = generate_case("trend_contraction_breakout", 4, n=126)
    query = transform_case(base, name="later", price_scale=8, volume_scale=3,
                           time_shift_days=3000).episode
    rng = np.random.default_rng(9)
    prefix_close = 80 * np.exp(np.cumsum(rng.normal(0, .02, 80)))
    target = base.episode.bars.copy()
    ratio = prefix_close[-1] / target.close.iloc[0]
    target[["open", "high", "low", "close"]] *= ratio
    prefix = pd.DataFrame({
        "timestamp": pd.date_range("1990-01-01", periods=80, freq="B"),
        "open": prefix_close, "high": prefix_close * 1.01,
        "low": prefix_close * .99, "close": prefix_close,
        "volume": np.full(80, 100_000),
    })
    target["timestamp"] = pd.date_range(
        prefix.timestamp.iloc[-1] + pd.offsets.BDay(), periods=len(target), freq="B",
    )
    candidate = pd.concat([prefix, target], ignore_index=True)
    hits, windows = scan_frame_multiview(
        query, candidate, InstrumentKey("test", "candidate"), stride=1,
        per_instrument=1,
    )
    assert windows == len(candidate) - query.key.lookback + 1
    assert target.timestamp.iloc[-1] in {hit.cutoff for hit in hits}
    assert all(dict(hit.view_distances).keys() == set(VIEW_NAMES) for hit in hits)


def test_mass_and_vector_scans_agree_when_stumpy_is_installed() -> None:
    pytest.importorskip("stumpy")
    base = generate_case("rounded_base", 5, n=126)
    query = transform_case(base, name="later", time_shift_days=3000).episode
    candidate = generate_case("rounded_base", 8, n=400).episode.bars
    vector, _ = scan_frame(query, candidate, InstrumentKey("test", "candidate"), backend="vector")
    mass, _ = scan_frame(query, candidate, InstrumentKey("test", "candidate"), backend="mass")
    assert [hit.cutoff for hit in mass] == [hit.cutoff for hit in vector]
    np.testing.assert_allclose(
        [hit.distance for hit in mass], [hit.distance for hit in vector], rtol=1e-6, atol=1e-8,
    )


def test_missing_quality_issues_do_not_become_literal_nan() -> None:
    record = type("Quality", (), {"issues": np.nan})()
    assert _quality_issues(None) == ()
    assert _quality_issues(record) == ()


def test_parallel_and_serial_universe_scans_are_identical(directory_dataset) -> None:
    source = DirectorySource(DatasetSpec(
        "test", "directory", directory_dataset, "parquet", timestamp_column="date",
    ))
    query = build_episode(source, InstrumentKey("test", "AAA"), "2021-12-31", 126, "dense-v1")
    request = SearchQuery(query.key, ("test",), ("A", "B"), 5)
    serial = scan_universe_candidates(
        query, source, request, stride=5, candidate_pool=10, workers=1,
    )
    parallel = scan_universe_candidates(
        query, source, request, stride=5, candidate_pool=10, workers=2,
    )
    assert serial.failures == parallel.failures == ()
    assert serial.windows_scanned == parallel.windows_scanned
    assert [(h.instrument, h.cutoff) for h in serial.hits] == [
        (h.instrument, h.cutoff) for h in parallel.hits
    ]
    np.testing.assert_allclose(
        [h.distance for h in serial.hits], [h.distance for h in parallel.hits],
    )


def test_parallel_and_serial_multiview_scans_are_identical(directory_dataset) -> None:
    source = DirectorySource(DatasetSpec(
        "test", "directory", directory_dataset, "parquet", timestamp_column="date",
    ))
    query = build_episode(source, InstrumentKey("test", "AAA"), "2021-12-31", 126, "dense-v1")
    request = SearchQuery(query.key, ("test",), ("A", "B"), 5)
    serial = scan_universe_candidates(
        query, source, request, stride=5, candidate_pool=10, workers=1,
        candidate_strategy="multiview",
    )
    parallel = scan_universe_candidates(
        query, source, request, stride=5, candidate_pool=10, workers=2,
        candidate_strategy="multiview",
    )
    assert serial.failures == parallel.failures == ()
    assert [(h.instrument, h.cutoff) for h in serial.hits] == [
        (h.instrument, h.cutoff) for h in parallel.hits
    ]
    np.testing.assert_allclose(
        [h.distance for h in serial.hits], [h.distance for h in parallel.hits],
    )


import pytest
