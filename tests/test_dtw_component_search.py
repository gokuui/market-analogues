from pathlib import Path
from dataclasses import asdict

import numpy as np
import numba
import pandas as pd
import pytest

from market_analogues.dtw_component_search import (
    certified_dtw_component_search, certified_staged_dtw_component_search,
    certified_dtw_component_search_contract,
    _staged_seed_proposals,
    dtw_component_search_contract, staged_dtw_component_search_contract,
    scan_dtw_component_bound_proposals,
)
from market_analogues.adapters import DirectorySource
from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.config import BenchmarkSpec, DatasetSpec
from market_analogues.distance import representation_distance
from market_analogues.episodes import build_episode
from market_analogues.exact_batch import sliding_exact_representations
from market_analogues.dtw_interval_bound import exact_price_component
from market_analogues.dtw_sample_store import (
    make_dtw_sample_record,
    write_dtw_sample_generation_from_shards,
)
from market_analogues.packed_bound_search import PackedBoundQuery
from market_analogues.packed_bound_store import (
    OVERFLOW_DTYPE,
    make_packed_record,
    write_packed_generation,
)
from market_analogues.quantized_bound import quantize_bound_row
from market_analogues.representation import represent
from market_analogues.synthetic import generate_case
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import EpisodeKey, InstrumentKey, SearchQuery


class _CountingDirectorySource(DirectorySource):
    def __init__(self, spec: DatasetSpec):
        super().__init__(spec)
        self.load_counts: dict[str, int] = {}

    def load(self, key: InstrumentKey) -> pd.DataFrame:
        self.load_counts[key.source_symbol] = (
            self.load_counts.get(key.source_symbol, 0) + 1
        )
        return super().load(key)


def _stores(tmp_path: Path):
    query = represent(generate_case("trend_contraction_breakout", 170_000).episode)
    candidates = [
        represent(generate_case(family, 170_010 + index).episode)
        for index, family in enumerate((
            "rounded_base", "trend_contraction_breakout", "steady_trend",
            "failed_breakout", "rounded_base", "volatile_reversal",
        ))
    ]
    packed_rows = np.concatenate([
        make_packed_record(
            f"{100 + index:024x}", 100 + index, 0, "A",
            quantize_bound_row(candidate),
        )
        for index, candidate in enumerate(candidates)
    ])
    packed_root = tmp_path / "packed"
    packed_generation = write_packed_generation(
        packed_root, packed_rows, np.empty(0, dtype=OVERFLOW_DTYPE),
        ("AAA",), {"purpose": "combined-component-test"}, activate=False,
    )
    from market_analogues.packed_bound_store import load_packed_generation

    packed = load_packed_generation(
        packed_root, packed_generation, verify_content=False, validate_records=False,
    )
    shard = tmp_path / "dtw-main.bin"
    overflow = tmp_path / "dtw-overflow.bin"
    np.concatenate([make_dtw_sample_record(value) for value in candidates]).tofile(shard)
    np.empty(0, dtype=make_dtw_sample_record(candidates[0]).dtype).tofile(overflow)
    dtw_root = tmp_path / "dtw"
    dtw_generation = write_dtw_sample_generation_from_shards(
        dtw_root, [shard], [overflow], packed_manifest=packed.manifest,
        provenance={"purpose": "combined-component-test"},
    )
    packed_query = PackedBoundQuery(
        f"{999:024x}", "QQQ", 10, 1_000, query, ("A", "B"),
    )
    return (
        packed_root, packed_generation, dtw_root, dtw_generation,
        packed_query, candidates,
    )


def test_combined_component_frontier_is_safe_and_order_independent(
    tmp_path: Path,
) -> None:
    packed_root, packed_generation, dtw_root, dtw_generation, query, candidates = (
        _stores(tmp_path)
    )
    reports = [
        scan_dtw_component_bound_proposals(
            packed_root, packed_generation, dtw_root, dtw_generation, query,
            quota=4, block_rows=block_rows, block_order=order,
            kernel_threads=threads,
            verify_content=False,
        )
        for block_rows, order, threads in (
            (2, "forward", 1),
            (5, "reverse", min(2, int(numba.config.NUMBA_NUM_THREADS))),
        )
    ]
    assert reports[0].candidate_digest == reports[1].candidate_digest
    assert reports[0].result_digest == reports[1].result_digest
    assert reports[0].contract_digest == dtw_component_search_contract()["digest"]
    exact = {
        f"{100 + index:024x}": exact_price_component(query.representation, value)
        for index, value in enumerate(candidates)
    }
    assert len(reports[0].candidates) == 4
    assert all(
        row.lower_bound <= exact[row.episode_id] + 1e-12
        for row in reports[0].candidates
    )


def test_combined_component_report_binds_auxiliary_generation(tmp_path: Path) -> None:
    packed_root, packed_generation, dtw_root, dtw_generation, query, _ = _stores(
        tmp_path
    )
    report = scan_dtw_component_bound_proposals(
        packed_root, packed_generation, dtw_root, dtw_generation, query,
        quota=3, verify_content=False,
    )
    assert report.packed_generation_id == packed_generation
    assert report.dtw_generation_id == dtw_generation
    assert report.input_digest


def test_adaptive_seed_expands_stable_prefix_until_symbols_are_diverse() -> None:
    representation = represent(generate_case("rounded_base", 811).episode)
    symbols = ("AAA", "BBB", "CCC")
    records = np.concatenate([
        make_packed_record(
            f"{index + 1:024x}", index + 1, index // 4, "A",
            quantize_bound_row(representation),
        )
        for index in range(12)
    ])
    scores = np.arange(12, dtype=np.float64)
    fixed, fixed_distinct = _staged_seed_proposals(
        records, scores, np.empty(0, dtype=OVERFLOW_DTYPE), symbols,
        initial_seed_rows=4, eligible_main=12, eligible_candidates=12,
        top_k=3, adaptive_seed=False,
    )
    adaptive, adaptive_distinct = _staged_seed_proposals(
        records, scores, np.empty(0, dtype=OVERFLOW_DTYPE), symbols,
        initial_seed_rows=4, eligible_main=12, eligible_candidates=12,
        top_k=3, adaptive_seed=True,
    )
    assert len(fixed) == 4
    assert fixed_distinct == 1
    assert len(adaptive) == 12
    assert adaptive_distinct == 3
    assert [row.episode_id for row in adaptive[:4]] \
        == [row.episode_id for row in fixed]
    assert staged_dtw_component_search_contract()["schema_version"] \
        == "certified-staged-dtw-component-search-v1"
    assert staged_dtw_component_search_contract()["digest"] \
        == "989bb4268fa38c7b8c4407c5ae15beb89292caa3029fb72807462f52c0be7224"
    adaptive_contract = staged_dtw_component_search_contract(adaptive_seed=True)
    assert adaptive_contract["schema_version"] \
        == "certified-adaptive-staged-dtw-component-search-v2"
    assert adaptive_contract["digest"] \
        == "14957dac821a9a8f4c8d703984acff4dbc1eab622e782f8aa8ca45bfad2463ec"
    with pytest.raises(ValueError, match="adaptive seed policy must be boolean"):
        staged_dtw_component_search_contract(adaptive_seed=1)  # type: ignore[arg-type]


def test_adaptive_seed_reports_genuinely_insufficient_symbol_universe() -> None:
    representation = represent(generate_case("rounded_base", 812).episode)
    records = np.concatenate([
        make_packed_record(
            f"{index + 1:024x}", index + 1, 0, "A",
            quantize_bound_row(representation),
        )
        for index in range(8)
    ])
    with pytest.raises(
        ValueError, match="fewer eligible symbols than top-k",
    ):
        _staged_seed_proposals(
            records, np.arange(8, dtype=np.float64),
            np.empty(0, dtype=OVERFLOW_DTYPE), ("AAA",),
            initial_seed_rows=2, eligible_main=8, eligible_candidates=8,
            top_k=2, adaptive_seed=True,
        )


def test_certified_combined_component_matches_exhaustive_oracle(
    directory_dataset: Path, bars: pd.DataFrame, tmp_path: Path,
) -> None:
    benchmark_path = directory_dataset / "MARKET.parquet"
    bars.to_parquet(benchmark_path, index=False)
    source = _CountingDirectorySource(DatasetSpec(
        "test", "directory", directory_dataset, "parquet",
        timestamp_column="date",
        benchmark=BenchmarkSpec(benchmark_path, timestamp_column="date"),
    ))
    query = build_episode(
        source, InstrumentKey("test", "AAA"), bars.date.iloc[-1], 63, "dense-v1",
    )
    request = SearchQuery(
        query.key, ("test",), ("A", "B"), 2,
        minimum_history_gap_bars=20, max_per_instrument=1,
    )
    benchmark = source.load_benchmark()
    assert benchmark is not None
    latest = latest_eligible_cutoff(query, request.minimum_history_gap_bars)
    symbols = ("AAA", "BBB")
    packed_rows = []
    dtw_rows = []
    exact = []
    query_representation = represent(query)
    for symbol_id, symbol in enumerate(symbols):
        key = InstrumentKey("test", symbol)
        frame = source.load(key)
        batch = sliding_exact_representations(
            frame, benchmark, lookback=63, stride=5, batch_size=32,
        )
        for position, representation in zip(
            batch.positions, batch.representations, strict=True,
        ):
            cutoff = pd.Timestamp(frame.timestamp.iloc[int(position)])
            episode_id = EpisodeKey(key, cutoff, 63, "dense-v1").id
            packed_rows.append(make_packed_record(
                episode_id, int(cutoff.value), symbol_id, "A",
                quantize_bound_row(representation),
            ))
            dtw_rows.append(make_dtw_sample_record(representation))
            if cutoff <= latest and not (
                symbol == "AAA" and cutoff >= query.bars.timestamp.iloc[0]
            ):
                exact.append((
                    representation_distance(query_representation, representation)[1]["price"],
                    episode_id, symbol,
                ))
    maximum = pd.Timestamp(bars.date.iloc[-1])
    provenance = {
        "source_prefixes": {
            symbol: asdict(causal_prefix_digest(
                source.load(InstrumentKey("test", symbol)), maximum,
            )) for symbol in symbols
        },
        "benchmark_prefix": asdict(causal_prefix_digest(benchmark, maximum)),
    }
    packed_root = tmp_path / "oracle-packed"
    packed_generation = write_packed_generation(
        packed_root, np.concatenate(packed_rows),
        np.empty(0, dtype=OVERFLOW_DTYPE), symbols, provenance, activate=False,
    )
    from market_analogues.packed_bound_store import load_packed_generation

    packed = load_packed_generation(
        packed_root, packed_generation, verify_content=False, validate_records=False,
    )
    main_shard = tmp_path / "oracle-dtw.bin"
    overflow_shard = tmp_path / "oracle-dtw-overflow.bin"
    np.concatenate(dtw_rows).tofile(main_shard)
    np.empty(0, dtype=dtw_rows[0].dtype).tofile(overflow_shard)
    dtw_root = tmp_path / "oracle-dtw"
    dtw_generation = write_dtw_sample_generation_from_shards(
        dtw_root, [main_shard], [overflow_shard],
        packed_manifest=packed.manifest, provenance={"purpose": "oracle"},
    )
    source.load_counts.clear()
    result = certified_dtw_component_search(
        query, source, request, packed_root, packed_generation,
        dtw_root, dtw_generation, store_dataset_id="test",
        initial_frontier_rows=1000, maximum_frontier_rows=1000,
        seed_rows=20, block_rows=13, workers=2,
    )
    assert source.load_counts["AAA"] <= 2
    assert source.load_counts["BBB"] <= 1
    source.load_counts.clear()
    prepared_symbols = {}
    staged = certified_staged_dtw_component_search(
        query, source, request, packed_root, packed_generation,
        dtw_root, dtw_generation, store_dataset_id="test",
        seed_rows=20, block_rows=13, rigid_threads=2, dtw_threads=2,
        exact_workers=2, prepared_symbol_cache=prepared_symbols,
    )
    expected = []
    seen = set()
    for distance, episode_id, symbol in sorted(exact):
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
    assert result.certificate.next_lower_bound is None \
        or result.certificate.next_lower_bound > result.certificate.stop_threshold
    assert result.certificate.contract_digest \
        == certified_dtw_component_search_contract()["digest"]
    assert [row.episode_key.id for row in staged.matches] \
        == [row[0] for row in expected]
    np.testing.assert_allclose(
        [row.total_distance for row in staged.matches],
        [row[1] for row in expected], rtol=0, atol=1e-12,
    )
    assert staged.certificate.contract_digest \
        == staged_dtw_component_search_contract()["digest"]
    assert staged.certificate.final_threshold <= staged.certificate.seed_threshold
    assert staged.certificate.rigid_bound_admitted \
        >= staged.certificate.combined_bound_admitted
    # The query prefix needs one load of AAA; exact completion then prepares
    # each candidate symbol once even though its frontier takes multiple batches.
    assert source.load_counts["AAA"] <= 2
    assert source.load_counts["BBB"] <= 1
    assert set(prepared_symbols) == {"AAA", "BBB"}
    adaptive = certified_staged_dtw_component_search(
        query, source, request, packed_root, packed_generation,
        dtw_root, dtw_generation, store_dataset_id="test",
        seed_rows=20, block_rows=13, rigid_threads=2, dtw_threads=2,
        exact_workers=2, prepared_symbol_cache=prepared_symbols,
        adaptive_seed=True,
    )
    assert adaptive.matches == staged.matches
    assert adaptive.certificate.contract_digest == (
        staged_dtw_component_search_contract(adaptive_seed=True)["digest"]
    )
    source.load_counts.clear()
    repeated = certified_staged_dtw_component_search(
        query, source, request, packed_root, packed_generation,
        dtw_root, dtw_generation, store_dataset_id="test",
        seed_rows=20, block_rows=13, rigid_threads=2, dtw_threads=2,
        exact_workers=2, prepared_symbol_cache=prepared_symbols,
    )
    assert repeated.matches == staged.matches
    assert source.load_counts == {"AAA": 1}
