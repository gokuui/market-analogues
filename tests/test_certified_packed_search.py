from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from market_analogues.adapters import DirectorySource
from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.certified_packed_search import certified_packed_search
from market_analogues.config import BenchmarkSpec, DatasetSpec
from market_analogues.episodes import build_episode
from market_analogues.exact_batch import sliding_exact_representations
from market_analogues.packed_bound_store import (
    OVERFLOW_DTYPE, make_packed_record, write_packed_generation,
)
from market_analogues.quantized_bound import quantize_bound_row
from market_analogues.search import SearchCandidate, exact_search, latest_eligible_cutoff
from market_analogues.types import EpisodeKey, InstrumentKey, SearchQuery


def test_certified_pack_exhaustion_matches_brute_force(
    directory_dataset: Path, bars: pd.DataFrame, tmp_path: Path,
) -> None:
    benchmark_path = directory_dataset / "MARKET.parquet"
    bars.to_parquet(benchmark_path, index=False)
    source = DirectorySource(DatasetSpec(
        "test", "directory", directory_dataset, "parquet",
        timestamp_column="date",
        benchmark=BenchmarkSpec(benchmark_path, timestamp_column="date"),
    ))
    query = build_episode(
        source, InstrumentKey("test", "AAA"), bars.date.iloc[-1], 63,
        "dense-v1",
    )
    request = SearchQuery(
        query.key, ("test",), ("A", "B"), 3,
        minimum_history_gap_bars=20, max_per_instrument=3,
    )
    benchmark = source.load_benchmark()
    assert benchmark is not None
    maximum_cutoff = pd.Timestamp(bars.date.iloc[-1])
    symbols = ("AAA", "BBB")
    rows = []
    brute_candidates = []
    latest = latest_eligible_cutoff(query, request.minimum_history_gap_bars)
    for symbol_id, symbol in enumerate(symbols):
        key = InstrumentKey("test", symbol)
        frame = source.load(key)
        batch = sliding_exact_representations(
            frame, benchmark, lookback=63, stride=5, batch_size=32,
        )
        for position, representation in zip(batch.positions, batch.representations):
            cutoff = pd.Timestamp(frame.timestamp.iloc[int(position)])
            episode_id = EpisodeKey(key, cutoff, 63, "dense-v1").id
            rows.append(make_packed_record(
                episode_id, int(cutoff.value), symbol_id, "A",
                quantize_bound_row(representation),
            ))
            if cutoff <= latest:
                brute_candidates.append(SearchCandidate.from_episode(build_episode(
                    source, key, cutoff, 63, "dense-v1", "A",
                )))
    provenance = {
        "source_prefixes": {
            symbol: asdict(causal_prefix_digest(
                source.load(InstrumentKey("test", symbol)), maximum_cutoff,
            )) for symbol in symbols
        },
        "benchmark_prefix": asdict(causal_prefix_digest(
            benchmark, maximum_cutoff,
        )),
    }
    store_root = tmp_path / "store"
    generation = write_packed_generation(
        store_root, np.concatenate(rows), np.empty(0, dtype=OVERFLOW_DTYPE),
        symbols, provenance, activate=False,
    )
    result = certified_packed_search(
        query, source, request, store_root, generation,
        store_dataset_id="test", initial_frontier_rows=1_000,
        maximum_frontier_rows=1_000, block_rows=7, workers=2,
        sparse_cutoff=3, seed_rows=20,
    )
    repeated = certified_packed_search(
        query, source, request, store_root, generation,
        store_dataset_id="test", initial_frontier_rows=1_000,
        maximum_frontier_rows=1_000, block_rows=13, workers=1,
        sparse_cutoff=3, seed_rows=20, verify_content=False,
    )
    brute = exact_search(query, brute_candidates, request)
    assert [row.episode_key.id for row in result.matches] == [
        row.episode_key.id for row in brute
    ]
    assert [row.total_distance for row in result.matches] == [
        row.total_distance for row in brute
    ]
    certificate = result.certificate
    assert (
        certificate.exact_evaluated + certificate.safely_pruned
        == certificate.eligible_candidates
    )
    assert not certificate.stopped_early
    assert certificate.next_lower_bound is None
    assert certificate.maximum_quantized_bound_excess <= 1e-12
    assert repeated.certificate.result_digest == certificate.result_digest
    assert repeated.certificate.input_digest == certificate.input_digest
