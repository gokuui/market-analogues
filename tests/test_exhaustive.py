import json
from pathlib import Path

import pandas as pd
import pytest

from market_analogues.adapters import DirectorySource
from market_analogues.config import BenchmarkSpec, DatasetSpec
from market_analogues.episodes import build_episode
from market_analogues.exhaustive import (
    FrontierError, build_exact_frontier, exhaustive_frontier_search,
    load_frontier_shard,
)
from market_analogues.search import SearchCandidate, exact_search
from market_analogues.types import InstrumentKey, SearchQuery


def _setup(directory_dataset: Path, bars: pd.DataFrame):
    source = DirectorySource(DatasetSpec(
        "test", "directory", directory_dataset, "parquet", timestamp_column="date",
    ))
    query = build_episode(
        source, InstrumentKey("test", "AAA"), bars.date.iloc[-1], 63, "dense-v1",
    )
    request = SearchQuery(
        query.key, ("test",), ("A", "B"), 3,
        minimum_history_gap_bars=20, max_per_instrument=3,
    )
    quality = pd.DataFrame({
        "symbol": ["AAA", "BBB"], "tier": ["A", "A"], "issues": ["", ""],
    })
    return source, query, request, quality


def _frontier_candidates(source, query, manifest_path: Path):
    manifest = json.loads(manifest_path.read_text())
    candidates = []
    for record in manifest["shards"]:
        shard = load_frontier_shard(manifest_path.parents[1] / record["path"])
        key = InstrumentKey(shard.metadata.dataset_id, shard.metadata.symbol)
        for cutoff in shard.cutoffs_ns:
            episode = build_episode(
                source, key, pd.Timestamp(int(cutoff)), shard.metadata.lookback,
                shard.metadata.representation_version, shard.metadata.quality_tier,
            )
            candidates.append(SearchCandidate.from_episode(episode))
    return candidates


def test_frontier_resume_and_exhaustive_search_match_brute_force(
    directory_dataset: Path, bars: pd.DataFrame, tmp_path: Path,
) -> None:
    source, query, request, quality = _setup(directory_dataset, bars)
    root = tmp_path / "frontier"
    first = build_exact_frontier(
        query, source, request, quality, root, stride=5, batch_size=16,
    )
    assert first.passed
    assert first.instruments_built == 2 and first.instruments_reused == 0
    second = build_exact_frontier(
        query, source, request, quality, root, stride=5, batch_size=7,
    )
    assert second.passed
    assert second.instruments_built == 0 and second.instruments_reused == 2
    assert second.manifest_digest == first.manifest_digest

    exhaustive = exhaustive_frontier_search(
        query, source, request, root, quality=quality,
        frontier_batch_rows=1, representation_cache_shards=2,
    )
    brute = exact_search(
        query, _frontier_candidates(source, query, first.manifest_path), request,
    )
    assert [match.episode_key.id for match in exhaustive.matches] == [
        match.episode_key.id for match in brute
    ]
    assert [match.total_distance for match in exhaustive.matches] == pytest.approx(
        [match.total_distance for match in brute], abs=1e-12,
    )
    assert exhaustive.certificate.eligible_candidates == first.eligible_rows
    assert (
        exhaustive.certificate.exact_evaluated
        + exhaustive.certificate.safely_pruned
        == first.eligible_rows
    )
    assert exhaustive.certificate.maximum_recomputed_bound_delta <= 1e-12


def test_frontier_chunk_size_is_deterministic_and_missing_shard_resumes(
    directory_dataset: Path, bars: pd.DataFrame, tmp_path: Path,
) -> None:
    source, query, request, quality = _setup(directory_dataset, bars)
    first = build_exact_frontier(
        query, source, request, quality, tmp_path / "one",
        stride=5, batch_size=5,
    )
    second = build_exact_frontier(
        query, source, request, quality, tmp_path / "two",
        stride=5, batch_size=100,
    )
    assert first.manifest_digest == second.manifest_digest
    manifest = json.loads(first.manifest_path.read_text())
    missing = (tmp_path / "one") / manifest["shards"][0]["path"]
    missing.unlink()
    resumed = build_exact_frontier(
        query, source, request, quality, tmp_path / "one",
        stride=5, batch_size=11,
    )
    assert resumed.passed
    assert resumed.instruments_built == 1 and resumed.instruments_reused == 1
    assert resumed.manifest_digest == first.manifest_digest


def test_frontier_corruption_and_stale_source_fail_visibly(
    directory_dataset: Path, bars: pd.DataFrame, tmp_path: Path,
) -> None:
    source, query, request, quality = _setup(directory_dataset, bars)
    root = tmp_path / "frontier"
    result = build_exact_frontier(query, source, request, quality, root, stride=5)
    manifest = json.loads(result.manifest_path.read_text())
    path = root / manifest["shards"][0]["path"]
    path.write_bytes(path.read_bytes()[:100])
    with pytest.raises(FrontierError):
        exhaustive_frontier_search(query, source, request, root, quality=quality)
    failed = build_exact_frontier(query, source, request, quality, root, stride=5)
    assert not failed.passed and "cannot load frontier shard" in failed.failures[0]
    repaired = build_exact_frontier(
        query, source, request, quality, root, stride=5, rebuild_invalid=True,
    )
    assert repaired.passed
    repaired_manifest = json.loads(repaired.manifest_path.read_text())
    repaired_manifest["shards"][0]["rows"] += 1
    repaired.manifest_path.write_text(json.dumps(repaired_manifest))
    with pytest.raises(FrontierError, match="manifest digest mismatch"):
        exhaustive_frontier_search(query, source, request, root, quality=quality)
    repaired = build_exact_frontier(
        query, source, request, quality, root, stride=5,
    )
    assert repaired.passed
    changed_quality = quality.copy()
    changed_quality.loc[changed_quality.symbol == "BBB", "tier"] = "B"
    with pytest.raises(FrontierError, match="quality provenance is stale"):
        exhaustive_frontier_search(
            query, source, request, root, quality=changed_quality,
        )
    changed_request = SearchQuery(
        query.key, ("test",), ("A", "B"), 3,
        minimum_history_gap_bars=21, max_per_instrument=3,
    )
    with pytest.raises(FrontierError, match="request scope is stale"):
        exhaustive_frontier_search(
            query, source, changed_request, root, quality=quality,
        )

    query_source = pd.read_parquet(directory_dataset / "AAA.parquet")
    changed_query_source = query_source.copy()
    changed_query_source.loc[0, "close"] *= 1.001
    changed_query_source.to_parquet(directory_dataset / "AAA.parquet", index=False)
    with pytest.raises(FrontierError, match="query source fingerprint is stale"):
        exhaustive_frontier_search(query, source, request, root, quality=quality)
    query_source.to_parquet(directory_dataset / "AAA.parquet", index=False)

    changed = pd.read_parquet(directory_dataset / "BBB.parquet")
    changed.loc[0, "close"] *= 1.001
    changed.to_parquet(directory_dataset / "BBB.parquet", index=False)
    with pytest.raises(FrontierError, match="source fingerprint is stale"):
        exhaustive_frontier_search(query, source, request, root, quality=quality)


def test_frontier_benchmark_mutation_fails_visibly(
    directory_dataset: Path, bars: pd.DataFrame, tmp_path: Path,
) -> None:
    benchmark_path = directory_dataset / "MARKET.parquet"
    bars.to_parquet(benchmark_path, index=False)
    source = DirectorySource(DatasetSpec(
        "test", "directory", directory_dataset, "parquet", timestamp_column="date",
        benchmark=BenchmarkSpec(benchmark_path, timestamp_column="date"),
    ))
    query = build_episode(
        source, InstrumentKey("test", "AAA"), bars.date.iloc[-1], 63, "dense-v1",
    )
    request = SearchQuery(
        query.key, ("test",), ("A", "B"), 3,
        minimum_history_gap_bars=20, max_per_instrument=3,
    )
    quality = pd.DataFrame({
        "symbol": ["AAA", "BBB"], "tier": ["A", "A"], "issues": ["", ""],
    })
    root = tmp_path / "frontier"
    assert build_exact_frontier(query, source, request, quality, root, stride=5).passed
    changed = bars.copy()
    changed.loc[0, "close"] *= 1.001
    changed.to_parquet(benchmark_path, index=False)
    with pytest.raises(FrontierError, match="benchmark fingerprint is stale"):
        exhaustive_frontier_search(query, source, request, root, quality=quality)


@pytest.mark.parametrize("keyword", ["stride", "batch_size"])
def test_frontier_rejects_invalid_controls(
    keyword: str, directory_dataset: Path, bars: pd.DataFrame, tmp_path: Path,
) -> None:
    source, query, request, quality = _setup(directory_dataset, bars)
    arguments = {"stride": 5, "batch_size": 16}
    arguments[keyword] = 0
    with pytest.raises(ValueError):
        build_exact_frontier(
            query, source, request, quality, tmp_path, **arguments,
        )
