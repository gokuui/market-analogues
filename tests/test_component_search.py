from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_analogues.adapters import DirectorySource
from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.component_search import (
    CertifiedComponentSearchError,
    certified_component_search,
    certified_component_search_contract,
)
from market_analogues.config import BenchmarkSpec, DatasetSpec
from market_analogues.distance import representation_distance
from market_analogues.episodes import build_episode
from market_analogues.exact_batch import sliding_exact_representations
from market_analogues.packed_bound_store import (
    OVERFLOW_DTYPE,
    make_packed_record,
    write_packed_generation,
)
from market_analogues.quantized_bound import quantize_bound_row
from market_analogues.representation import represent
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import EpisodeKey, InstrumentKey, SearchQuery


def test_certified_price_component_matches_exhaustive_distinct_symbol_oracle(
    directory_dataset: Path, bars: pd.DataFrame, tmp_path: Path,
) -> None:
    benchmark_path = directory_dataset / "MARKET.parquet"
    bars.to_parquet(benchmark_path, index=False)
    source = DirectorySource(DatasetSpec(
        "test", "directory", directory_dataset, "parquet",
        timestamp_column="date",
        benchmark=BenchmarkSpec(benchmark_path, timestamp_column="date"),
    ))
    query = build_episode(source, InstrumentKey("test", "AAA"),
                          bars.date.iloc[-1], 63, "dense-v1")
    request = SearchQuery(
        query.key, ("test",), ("A", "B"), 2,
        minimum_history_gap_bars=20, max_per_instrument=1,
    )
    benchmark = source.load_benchmark()
    assert benchmark is not None
    latest = latest_eligible_cutoff(query, request.minimum_history_gap_bars)
    symbols = ("AAA", "BBB")
    rows = []
    exact = []
    query_representation = represent(query)
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
            if cutoff <= latest and not (symbol == "AAA" and cutoff >= query.bars.timestamp.iloc[0]):
                distance = representation_distance(query_representation, representation)[1]["price"]
                exact.append((distance, episode_id, symbol, cutoff))
    maximum_cutoff = pd.Timestamp(bars.date.iloc[-1])
    provenance = {
        "source_prefixes": {
            symbol: asdict(causal_prefix_digest(
                source.load(InstrumentKey("test", symbol)), maximum_cutoff,
            )) for symbol in symbols
        },
        "benchmark_prefix": asdict(causal_prefix_digest(benchmark, maximum_cutoff)),
    }
    store = tmp_path / "store"
    generation = write_packed_generation(
        store, np.concatenate(rows), np.empty(0, dtype=OVERFLOW_DTYPE),
        symbols, provenance, activate=False,
    )
    result = certified_component_search(
        query, source, request, store, generation, store_dataset_id="test",
        initial_frontier_rows=1_000, maximum_frontier_rows=1_000,
        seed_rows=20, block_rows=13, proposal_threads=3, workers=2,
    )
    expected = []
    seen = set()
    for distance, episode_id, symbol, cutoff in sorted(exact):
        if symbol in seen:
            continue
        seen.add(symbol)
        expected.append((episode_id, distance))
        if len(expected) == 2:
            break
    assert [row.episode_key.id for row in result.matches] == [row[0] for row in expected]
    np.testing.assert_allclose(
        [row.total_distance for row in result.matches],
        [row[1] for row in expected], rtol=0, atol=1e-12,
    )
    certificate = result.certificate
    assert certificate.exact_evaluated + certificate.native_bound_pruned \
        + certificate.packed_bound_pruned == certificate.eligible_candidates
    assert certificate.stop_threshold == result.matches[-1].total_distance
    assert certificate.next_lower_bound is None \
        or certificate.next_lower_bound > certificate.stop_threshold
    assert certificate.contract_digest == certified_component_search_contract()["digest"]


def test_component_search_rejects_non_distinct_symbol_contract(
    directory_dataset: Path, bars: pd.DataFrame,
) -> None:
    benchmark_path = directory_dataset / "MARKET.parquet"
    bars.to_parquet(benchmark_path, index=False)
    source = DirectorySource(DatasetSpec(
        "test", "directory", directory_dataset, "parquet",
        timestamp_column="date",
        benchmark=BenchmarkSpec(benchmark_path, timestamp_column="date"),
    ))
    query = build_episode(source, InstrumentKey("test", "AAA"),
                          bars.date.iloc[-1], 63, "dense-v1")
    request = SearchQuery(query.key, ("test",), ("A",), 2, max_per_instrument=2)
    with pytest.raises(CertifiedComponentSearchError, match="one deduplicated"):
        certified_component_search(
            query, source, request, Path("/nonexistent"), "generation",
            store_dataset_id="test",
        )
