from __future__ import annotations

import pandas as pd
import pytest

from market_analogues.adapters import DirectorySource
from market_analogues.config import BenchmarkSpec, DatasetSpec
from market_analogues.episodes import build_episode
from market_analogues.types import InstrumentKey, SearchQuery
from market_analogues.view_search import persisted_exact_search, search_view_store
from market_analogues.view_store import ViewShardError, build_view_store


def test_persisted_search_is_deterministic_and_temporally_eligible(
    directory_dataset, tmp_path,
) -> None:
    source = DirectorySource(DatasetSpec(
        "test", "directory", directory_dataset, "parquet", timestamp_column="date",
    ))
    quality = pd.DataFrame({
        "symbol": ["AAA", "BBB"], "tier": ["A", "B"], "issues": ["", ""],
    })
    root = tmp_path / "store"
    build_view_store(
        source, quality, root, lookbacks=(63,), stride=5, workers=2,
    )
    query = build_episode(
        source, InstrumentKey("test", "AAA"), "2021-12-31", 63, "dense-v1",
    )
    request = SearchQuery(
        query.key, ("test",), ("A", "B"), 5, minimum_history_gap_bars=20,
    )
    first = search_view_store(
        query, request, root, candidate_pool=12, per_instrument_view=3,
    )
    second = search_view_store(
        query, request, root, candidate_pool=12, per_instrument_view=3,
    )
    assert first.manifest_digest == second.manifest_digest
    assert [(hit.episode_id, hit.fusion_score) for hit in first.hits] == [
        (hit.episode_id, hit.fusion_score) for hit in second.hits
    ]
    assert first.shards_loaded == 2
    assert first.rows_considered > 0
    assert len(first.hits) <= 12
    assert all(hit.cutoff < query.key.cutoff for hit in first.hits)
    same_symbol = [hit for hit in first.hits if hit.instrument == query.key.instrument]
    assert all(hit.cutoff < query.bars.timestamp.iloc[0] for hit in same_symbol)


def test_persisted_exact_search_validates_and_prunes(
    directory_dataset, tmp_path,
) -> None:
    source = DirectorySource(DatasetSpec(
        "test", "directory", directory_dataset, "parquet", timestamp_column="date",
    ))
    quality = pd.DataFrame({
        "symbol": ["AAA", "BBB"], "tier": ["A", "B"], "issues": ["", ""],
    })
    root = tmp_path / "store"
    build_view_store(source, quality, root, lookbacks=(63,), stride=5, workers=2)
    query = build_episode(
        source, InstrumentKey("test", "AAA"), "2021-12-31", 63, "dense-v1",
    )
    request = SearchQuery(
        query.key, ("test",), ("A", "B"), 5, minimum_history_gap_bars=20,
    )

    first = persisted_exact_search(
        query, source, request, root, candidate_pool=12,
        per_instrument_view=3, workers=2, quality=quality,
    )
    second = persisted_exact_search(
        query, source, request, root, candidate_pool=12,
        per_instrument_view=3, workers=1, quality=quality,
    )

    assert [match.episode_key.id for match in first.matches] == [
        match.episode_key.id for match in second.matches
    ]
    assert first.candidates_materialized == len(first.candidate_search.hits)
    assert first.fingerprints_validated == first.candidate_search.shards_loaded
    assert first.pruning.exact_evaluated <= first.pruning.eligible_candidates
    assert all(match.quality_tier in {"A", "B"} for match in first.matches)
    changed_quality = quality.copy()
    changed_quality.loc[changed_quality.symbol == "BBB", "tier"] = "QUARANTINED"
    with pytest.raises(ViewShardError, match="quality tier changed"):
        persisted_exact_search(
            query, source, request, root, candidate_pool=12,
            per_instrument_view=3, quality=changed_quality,
        )


def test_persisted_exact_search_rejects_source_changed_after_build(
    directory_dataset, tmp_path,
) -> None:
    source = DirectorySource(DatasetSpec(
        "test", "directory", directory_dataset, "parquet", timestamp_column="date",
    ))
    quality = pd.DataFrame({
        "symbol": ["AAA", "BBB"], "tier": ["A", "B"], "issues": ["", ""],
    })
    root = tmp_path / "store"
    build_view_store(source, quality, root, lookbacks=(63,), stride=5)
    query = build_episode(
        source, InstrumentKey("test", "AAA"), "2021-12-31", 63, "dense-v1",
    )
    request = SearchQuery(
        query.key, ("test",), ("A", "B"), 5, minimum_history_gap_bars=20,
    )
    changed = pd.read_parquet(directory_dataset / "BBB.parquet")
    changed.loc[0, "close"] *= 1.01
    changed.to_parquet(directory_dataset / "BBB.parquet", index=False)

    with pytest.raises(ViewShardError, match="source changed"):
        persisted_exact_search(
            query, source, request, root, candidate_pool=12,
            per_instrument_view=3, quality=quality,
        )


def test_persisted_exact_search_rejects_benchmark_changed_after_build(
    directory_dataset, tmp_path,
) -> None:
    benchmark_path = tmp_path / "benchmark.parquet"
    benchmark = pd.read_parquet(directory_dataset / "AAA.parquet")
    benchmark.to_parquet(benchmark_path, index=False)
    source = DirectorySource(DatasetSpec(
        "test", "directory", directory_dataset, "parquet", timestamp_column="date",
        benchmark=BenchmarkSpec(benchmark_path),
    ))
    quality = pd.DataFrame({
        "symbol": ["AAA", "BBB"], "tier": ["A", "B"], "issues": ["", ""],
    })
    root = tmp_path / "store"
    build_view_store(source, quality, root, lookbacks=(63,), stride=5)
    query = build_episode(
        source, InstrumentKey("test", "AAA"), "2021-12-31", 63, "dense-v1",
    )
    request = SearchQuery(
        query.key, ("test",), ("A", "B"), 5, minimum_history_gap_bars=20,
    )
    benchmark.loc[0, "close"] *= 1.01
    benchmark.to_parquet(benchmark_path, index=False)

    with pytest.raises(ViewShardError, match="benchmark changed"):
        persisted_exact_search(
            query, source, request, root, candidate_pool=12,
            per_instrument_view=3, quality=quality,
        )
