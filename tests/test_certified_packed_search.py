from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest

from market_analogues.adapters import DirectorySource
from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.certified_packed_search import (
    CompactScoredCandidate,
    _close_native_bound_deferred,
    _select_compact_scored,
    certified_packed_search,
)
from market_analogues.config import BenchmarkSpec, DatasetSpec
from market_analogues.episodes import build_episode
from market_analogues.exact_batch import _stage_rows, sliding_exact_representations
from market_analogues.packed_bound_store import (
    OVERFLOW_DTYPE, make_packed_record, write_packed_generation,
)
from market_analogues.packed_bound_search import (
    BRANCH_AWARE_SEARCH_SCHEMA_VERSION, SEARCH_SCHEMA_VERSION,
    BoundProposal, BoundProposalReport, PackedBoundQuery,
    _packed_query_input_digest, bound_proposal_candidate_digest,
    packed_bound_search_contract, scan_packed_bound_proposals,
)
from market_analogues.quantized_bound import quantize_bound_row
from market_analogues.representation import represent, representation_input_digest
from market_analogues.search import (
    ScoredCandidate, SearchCandidate, exact_search, latest_eligible_cutoff,
    select_scored,
)
from market_analogues.synthetic import generate_case
from market_analogues.types import (
    AnalogueMatch, Episode, EpisodeKey, InstrumentKey, SearchQuery, stable_hash,
)


def test_representation_input_digest_binds_exact_and_alignment_samples() -> None:
    original = represent(generate_case("rounded_base", 900).episode)
    changed_48 = dict(original.samples_48)
    changed_48["close_path"] = changed_48["close_path"].copy()
    changed_48["close_path"][0] += 1.0
    changed_64 = dict(original.samples_64)
    changed_64["close_path"] = changed_64["close_path"].copy()
    changed_64["close_path"][0] += 1.0
    assert representation_input_digest(original) != representation_input_digest(
        type(original)(
            original.channels, original.coarse, changed_48,
            original.samples_64, original.stage, original.structural,
        ),
    )
    assert representation_input_digest(original) != representation_input_digest(
        type(original)(
            original.channels, original.coarse, original.samples_48,
            changed_64, original.stage, original.structural,
        ),
    )


def test_certified_pack_exhaustion_matches_brute_force(
    directory_dataset: Path, bars: pd.DataFrame, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    requested = certified_packed_search(
        query, source, request, store_root, generation,
        store_dataset_id="test", initial_frontier_rows=1_000,
        maximum_frontier_rows=1_000, block_rows=11, workers=2,
        sparse_cutoff=3, seed_rows=20, verify_content=False,
        requested_positions=True,
    )
    hybrid = certified_packed_search(
        query, source, request, store_root, generation,
        store_dataset_id="test", initial_frontier_rows=1_000,
        maximum_frontier_rows=1_000, block_rows=9, workers=2,
        sparse_cutoff=3, seed_rows=20, verify_content=False,
        hybrid_requested_positions=True,
    )
    vector = certified_packed_search(
        query, source, request, store_root, generation,
        store_dataset_id="test", initial_frontier_rows=1_000,
        maximum_frontier_rows=1_000, block_rows=15, workers=2,
        sparse_cutoff=3, seed_rows=20, verify_content=False,
        requested_positions=True, vector_lower_bounds=True,
    )
    deferred = certified_packed_search(
        query, source, request, store_root, generation,
        store_dataset_id="test", initial_frontier_rows=1_000,
        maximum_frontier_rows=1_000, block_rows=17, workers=2,
        sparse_cutoff=3, seed_rows=20, verify_content=False,
        requested_positions=True, vector_lower_bounds=True,
        deferred_alignments=True,
    )
    compact = certified_packed_search(
        query, source, request, store_root, generation,
        store_dataset_id="test", initial_frontier_rows=1_000,
        maximum_frontier_rows=1_000, block_rows=21, workers=2,
        sparse_cutoff=3, seed_rows=20, verify_content=False,
        requested_positions=True, vector_lower_bounds=True,
        deferred_alignments=True, compact_scored=True,
    )
    native = certified_packed_search(
        query, source, request, store_root, generation,
        store_dataset_id="test", initial_frontier_rows=1_000,
        maximum_frontier_rows=1_000, block_rows=21, workers=2,
        sparse_cutoff=3, seed_rows=20, verify_content=False,
        requested_positions=True, vector_lower_bounds=True,
        deferred_alignments=True, compact_scored=True,
        native_bound_deferral=True,
    )
    native_repeated = certified_packed_search(
        query, source, request, store_root, generation,
        store_dataset_id="test", initial_frontier_rows=1_000,
        maximum_frontier_rows=1_000, block_rows=25, workers=1,
        sparse_cutoff=3, seed_rows=20, verify_content=False,
        requested_positions=True, vector_lower_bounds=True,
        deferred_alignments=True, compact_scored=True,
        native_bound_deferral=True,
    )
    packed_query = PackedBoundQuery(
        query.key.id, query.key.instrument.source_symbol,
        int(query.bars.timestamp.iloc[0].value),
        int(latest_eligible_cutoff(
            query, request.minimum_history_gap_bars,
        ).value),
        represent(query), request.quality_tiers,
    )
    proposal = scan_packed_bound_proposals(
        store_root, generation, packed_query,
        route_quotas={"composite": 1_001}, block_rows=19,
        verify_content=False,
    )
    branch_proposal = scan_packed_bound_proposals(
        store_root, generation, packed_query,
        route_quotas={"composite": 1_001}, block_rows=13,
        branch_aware=True, verify_content=False,
    )
    precomputed = certified_packed_search(
        query, source, request, store_root, generation,
        store_dataset_id="test", initial_frontier_rows=1_000,
        maximum_frontier_rows=1_000, block_rows=23, workers=2,
        sparse_cutoff=3, seed_rows=20, verify_content=False,
        requested_positions=True, vector_lower_bounds=True,
        deferred_alignments=True, precomputed_proposal=proposal,
    )
    branch = certified_packed_search(
        query, source, request, store_root, generation,
        store_dataset_id="test", initial_frontier_rows=1_000,
        maximum_frontier_rows=1_000, block_rows=13, workers=2,
        sparse_cutoff=3, seed_rows=20, verify_content=False,
        requested_positions=True, vector_lower_bounds=True,
        deferred_alignments=True, compact_scored=True,
        native_bound_deferral=True, streaming_threshold_closure=True,
        branch_aware_packed_bounds=True,
        precomputed_proposal=branch_proposal,
    )
    streaming = certified_packed_search(
        query, source, request, store_root, generation,
        store_dataset_id="test", initial_frontier_rows=3,
        maximum_frontier_rows=3, block_rows=17, workers=2,
        sparse_cutoff=3, seed_rows=3, verify_content=False,
        requested_positions=True, vector_lower_bounds=True,
        deferred_alignments=True, compact_scored=True,
        native_bound_deferral=True, streaming_threshold_closure=True,
        precomputed_proposal=proposal,
    )
    streaming_reverse = certified_packed_search(
        query, source, request, store_root, generation,
        store_dataset_id="test", initial_frontier_rows=3,
        maximum_frontier_rows=3, block_rows=11, workers=1,
        sparse_cutoff=3, seed_rows=3, verify_content=False,
        requested_positions=True, vector_lower_bounds=True,
        deferred_alignments=True, compact_scored=True,
        native_bound_deferral=True, streaming_threshold_closure=True,
        threshold_scan_block_order="reverse", precomputed_proposal=proposal,
    )
    brute = exact_search(query, brute_candidates, request)
    assert [row.episode_key.id for row in result.matches] == [
        row.episode_key.id for row in brute
    ]
    assert [row.total_distance for row in result.matches] == [
        row.total_distance for row in brute
    ]
    assert [row.episode_key.id for row in streaming.matches] == [
        row.episode_key.id for row in brute
    ]
    assert streaming.certificate.threshold_closure_passes
    assert streaming.certificate.threshold_closure_passes[-1].certified
    assert streaming_reverse.certificate.result_digest == (
        streaming.certificate.result_digest
    )
    assert [row.episode_key.id for row in streaming_reverse.matches] == [
        row.episode_key.id for row in brute
    ]
    assert streaming.certificate.native_bound_accounting.native_bound_evaluated + (
        streaming.certificate.native_bound_accounting.packed_bound_pruned
    ) == streaming.certificate.eligible_candidates
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
    assert [row.episode_key.id for row in requested.matches] == [
        row.episode_key.id for row in result.matches
    ]
    assert [row.total_distance for row in requested.matches] == [
        row.total_distance for row in result.matches
    ]
    assert [row.component_distances for row in requested.matches] == [
        row.component_distances for row in result.matches
    ]
    assert [row.alignment for row in requested.matches] == [
        row.alignment for row in result.matches
    ]
    assert requested.certificate.contract_digest != certificate.contract_digest
    assert requested.certificate.schema_version == "m04r-certified-packed-search-v2"
    assert requested.certificate.input_digest == certificate.input_digest
    assert requested.certificate.exact_evaluated == certificate.exact_evaluated
    assert [row.episode_key.id for row in hybrid.matches] == [
        row.episode_key.id for row in result.matches
    ]
    assert [row.total_distance for row in hybrid.matches] == [
        row.total_distance for row in result.matches
    ]
    assert [row.component_distances for row in hybrid.matches] == [
        row.component_distances for row in result.matches
    ]
    assert [row.alignment for row in hybrid.matches] == [
        row.alignment for row in result.matches
    ]
    assert hybrid.certificate.schema_version == "m04r-certified-packed-search-v3"
    assert hybrid.certificate.contract_digest not in {
        certificate.contract_digest, requested.certificate.contract_digest,
    }
    assert hybrid.certificate.input_digest == certificate.input_digest
    assert hybrid.certificate.exact_evaluated == certificate.exact_evaluated
    assert [row.episode_key.id for row in vector.matches] == [
        row.episode_key.id for row in result.matches
    ]
    np.testing.assert_allclose(
        [row.total_distance for row in vector.matches],
        [row.total_distance for row in result.matches], rtol=0, atol=1e-12,
    )
    for actual, expected in zip(vector.matches, result.matches):
        assert actual.component_distances.keys() == expected.component_distances.keys()
        np.testing.assert_allclose(
            list(actual.component_distances.values()),
            list(expected.component_distances.values()), rtol=0, atol=1e-12,
        )
        assert actual.alignment == expected.alignment
    assert vector.certificate.schema_version == "m04r-certified-packed-search-v4"
    assert vector.certificate.contract_digest not in {
        certificate.contract_digest, requested.certificate.contract_digest,
        hybrid.certificate.contract_digest,
    }
    assert vector.certificate.input_digest == certificate.input_digest
    assert vector.certificate.exact_evaluated == certificate.exact_evaluated
    assert [row.episode_key.id for row in deferred.matches] == [
        row.episode_key.id for row in vector.matches
    ]
    np.testing.assert_allclose(
        [row.total_distance for row in deferred.matches],
        [row.total_distance for row in vector.matches], rtol=0, atol=1e-12,
    )
    for actual, expected in zip(deferred.matches, vector.matches):
        assert actual.component_distances.keys() == expected.component_distances.keys()
        np.testing.assert_allclose(
            list(actual.component_distances.values()),
            list(expected.component_distances.values()), rtol=0, atol=1e-12,
        )
        assert actual.alignment == expected.alignment
    assert deferred.certificate.schema_version == "m04r-certified-packed-search-v5"
    assert deferred.certificate.contract_digest not in {
        certificate.contract_digest, requested.certificate.contract_digest,
        hybrid.certificate.contract_digest, vector.certificate.contract_digest,
    }
    assert deferred.certificate.input_digest == certificate.input_digest
    assert deferred.certificate.exact_evaluated == certificate.exact_evaluated
    assert [row.episode_key.id for row in compact.matches] == [
        row.episode_key.id for row in deferred.matches
    ]
    np.testing.assert_allclose(
        [row.total_distance for row in compact.matches],
        [row.total_distance for row in deferred.matches], rtol=0, atol=1e-12,
    )
    assert [row.component_distances for row in compact.matches] == [
        row.component_distances for row in deferred.matches
    ]
    assert [row.alignment for row in compact.matches] == [
        row.alignment for row in deferred.matches
    ]
    assert compact.certificate.schema_version == "m04r-certified-packed-search-v6"
    assert compact.certificate.exact_evaluated == deferred.certificate.exact_evaluated
    assert compact.certificate.safely_pruned == deferred.certificate.safely_pruned
    assert [row.episode_key.id for row in native.matches] == [
        row.episode_key.id for row in compact.matches
    ]
    np.testing.assert_allclose(
        [row.total_distance for row in native.matches],
        [row.total_distance for row in compact.matches], rtol=0, atol=1e-12,
    )
    for actual, expected in zip(native.matches, compact.matches):
        assert actual.component_distances.keys() == expected.component_distances.keys()
        np.testing.assert_allclose(
            list(actual.component_distances.values()),
            list(expected.component_distances.values()), rtol=0, atol=1e-12,
        )
    assert [row.alignment for row in native.matches] == [
        row.alignment for row in compact.matches
    ]
    assert native.certificate.schema_version == "m04r-certified-packed-search-v7"
    assert native_repeated.certificate.result_digest == native.certificate.result_digest
    assert native_repeated.certificate.contract_digest == native.certificate.contract_digest
    accounting = native.certificate.native_bound_accounting
    assert accounting.native_bound_pruned > 0
    assert accounting.exact_dtw_evaluated < accounting.native_bound_evaluated
    assert accounting.native_bound_evaluated == (
        accounting.exact_dtw_evaluated + accounting.native_bound_pruned
    )
    assert native.certificate.eligible_candidates == (
        accounting.exact_dtw_evaluated + accounting.native_bound_pruned
        + accounting.packed_bound_pruned
    )
    assert native.certificate.exact_evaluated == accounting.exact_dtw_evaluated
    assert native.certificate.safely_pruned == (
        accounting.native_bound_pruned + accounting.packed_bound_pruned
    )
    if native.certificate.minimum_native_pruned_bound is not None:
        assert (
            native.certificate.minimum_native_pruned_bound
            > native.certificate.stop_threshold
        )
    assert precomputed.certificate.result_digest == deferred.certificate.result_digest
    assert [row.episode_key.id for row in precomputed.matches] == [
        row.episode_key.id for row in deferred.matches
    ]
    assert branch.certificate.schema_version == "m04r-certified-packed-search-v8"
    assert [row.episode_key.id for row in branch.matches] == [
        row.episode_key.id for row in brute
    ]
    np.testing.assert_allclose(
        [row.total_distance for row in branch.matches],
        [row.total_distance for row in brute], rtol=0, atol=1e-12,
    )

    import market_analogues.certified_packed_search as certified_module
    from dataclasses import replace
    from market_analogues.certified_packed_search import CertifiedPackedSearchError
    from market_analogues.exact_batch import LowerBoundBatch

    with pytest.raises(CertifiedPackedSearchError, match="precomputed proposal"):
        certified_packed_search(
            query, source, request, store_root, generation,
            store_dataset_id="test", initial_frontier_rows=1_000,
            maximum_frontier_rows=1_000, block_rows=23, workers=1,
            sparse_cutoff=3, seed_rows=20, verify_content=False,
            requested_positions=True, vector_lower_bounds=True,
            deferred_alignments=True,
            precomputed_proposal=replace(proposal, result_digest="invalid"),
        )
    with pytest.raises(CertifiedPackedSearchError, match="precomputed proposal"):
        certified_packed_search(
            query, source, request, store_root, generation,
            store_dataset_id="test", initial_frontier_rows=1_000,
            maximum_frontier_rows=1_000, block_rows=23, workers=1,
            sparse_cutoff=3, seed_rows=20, verify_content=False,
            requested_positions=True, vector_lower_bounds=True,
            deferred_alignments=True, compact_scored=True,
            native_bound_deferral=True, streaming_threshold_closure=True,
            branch_aware_packed_bounds=True,
            precomputed_proposal=proposal,
        )

    changed_samples = dict(packed_query.representation.samples_48)
    changed_samples["close_path"] = changed_samples["close_path"].copy()
    changed_samples["close_path"][0] += 1.0
    changed_representation = replace(
        packed_query.representation, samples_48=changed_samples,
    )
    stale_same_id = scan_packed_bound_proposals(
        store_root, generation,
        replace(packed_query, representation=changed_representation),
        route_quotas={"composite": 1_001}, branch_aware=True,
        verify_content=False,
    )
    with pytest.raises(CertifiedPackedSearchError, match="precomputed proposal"):
        certified_packed_search(
            query, source, request, store_root, generation,
            store_dataset_id="test", initial_frontier_rows=1_000,
            maximum_frontier_rows=1_000, block_rows=23, workers=1,
            sparse_cutoff=3, seed_rows=20, verify_content=False,
            requested_positions=True, vector_lower_bounds=True,
            deferred_alignments=True, compact_scored=True,
            native_bound_deferral=True, streaming_threshold_closure=True,
            branch_aware_packed_bounds=True,
            precomputed_proposal=stale_same_id,
        )

    original_bounds = certified_module.batch_representation_lower_bounds
    with monkeypatch.context() as patch:
        def invalid_bounds(query_representation, candidates):
            bounded = original_bounds(query_representation, candidates)
            totals = bounded.totals.copy()
            totals[0] = np.nan
            return LowerBoundBatch(totals, bounded.components, bounded.rigid_price)

        patch.setattr(
            certified_module, "batch_representation_lower_bounds", invalid_bounds,
        )
        with pytest.raises(CertifiedPackedSearchError, match="native lower bound"):
            certified_packed_search(
                query, source, request, store_root, generation,
                store_dataset_id="test", initial_frontier_rows=1_000,
                maximum_frontier_rows=1_000, block_rows=23, workers=1,
                sparse_cutoff=3, seed_rows=20, verify_content=False,
                requested_positions=True, vector_lower_bounds=True,
                deferred_alignments=True, compact_scored=True,
                native_bound_deferral=True, precomputed_proposal=proposal,
            )

    original_complete = certified_module.complete_representation_distance
    with monkeypatch.context() as patch:
        def invalid_complete(*args, **kwargs):
            total, components, path = original_complete(*args, **kwargs)
            return np.nan, components, path

        patch.setattr(
            certified_module, "complete_representation_distance", invalid_complete,
        )
        with pytest.raises(CertifiedPackedSearchError, match="exact distance"):
            certified_packed_search(
                query, source, request, store_root, generation,
                store_dataset_id="test", initial_frontier_rows=1_000,
                maximum_frontier_rows=1_000, block_rows=23, workers=1,
                sparse_cutoff=3, seed_rows=20, verify_content=False,
                requested_positions=True, vector_lower_bounds=True,
                deferred_alignments=True, compact_scored=True,
                native_bound_deferral=True, precomputed_proposal=proposal,
            )

    # Exercise the complete streamed state machine across a threshold rise.
    # The second band admits a row exactly equal to its inclusive upper bound.
    from types import SimpleNamespace

    controlled = proposal.candidates[:6]
    controlled_ids = [row.episode_id for row in controlled]
    distances = {
        controlled_ids[0]: .2,
        controlled_ids[1]: .3,
        controlled_ids[2]: 1_000_000.,
        controlled_ids[3]: 2_000_000.,
        controlled_ids[4]: .1,
        controlled_ids[5]: 3_000_000.,
    }
    intervals = {
        controlled_ids[0]: (0, 9), controlled_ids[1]: (10, 19),
        controlled_ids[2]: (20, 29), controlled_ids[3]: (30, 39),
        controlled_ids[4]: (5, 15), controlled_ids[5]: (40, 49),
    }
    selection_instrument = InstrumentKey("test", "CONTROLLED")

    def controlled_score(proposals, **kwargs):
        output = []
        for row in proposals:
            key = EpisodeKey(
                InstrumentKey("test", row.symbol), pd.Timestamp(row.cutoff_ns),
                query.key.lookback, query.key.representation_version,
            )
            start, cutoff = intervals[row.episode_id]
            distance = distances[row.episode_id]
            output.append(CompactScoredCandidate(
                AnalogueMatch(key, distance, {"price": distance}),
                selection_instrument, start, cutoff,
            ))
        return output, [], 0., 0, 0

    scan_calls = []

    def controlled_scan(*args, lower_exclusive=None, upper_inclusive,
                        excluded_episode_ids, consume, **kwargs):
        index = len(scan_calls)
        expected = (
            (None, 1_000_001., controlled[4], 2_000_000.)
            if index == 0 else
            (1_000_001., 2_000_001., controlled[5], 3_000_000.)
        )
        assert lower_exclusive == expected[0]
        assert upper_inclusive == expected[1]
        consume((expected[2],))
        scan_calls.append(index)
        return SimpleNamespace(
            eligible_rows=proposal.eligible_rows,
            excluded_eligible_rows=4,
            admitted_rows=1,
            minimum_above_upper=expected[3],
            exclusions_digest="prefix-digest",
            admitted_set_digest=f"set-{index}",
            result_digest=f"scan-{index}",
        )

    final_distances = iter((.1, 1_000_000., 2_000_000.))
    with monkeypatch.context() as patch:
        patch.setattr(certified_module, "_score_new_proposals", controlled_score)
        patch.setattr(certified_module, "scan_packed_bound_threshold", controlled_scan)
        patch.setattr(
            certified_module, "representation_distance",
            lambda *args, **kwargs: (
                (value := next(final_distances)), {"price": value}, [(0, 0)],
            ),
        )
        two_pass = certified_packed_search(
            query, source, request, store_root, generation,
            store_dataset_id="test", initial_frontier_rows=4,
            maximum_frontier_rows=4, block_rows=7, workers=1,
            sparse_cutoff=3, seed_rows=4, verify_content=False,
            requested_positions=True, vector_lower_bounds=True,
            deferred_alignments=True, compact_scored=True,
            native_bound_deferral=True, streaming_threshold_closure=True,
            precomputed_proposal=proposal, tolerance=1.0,
        )
    passes = two_pass.certificate.threshold_closure_passes
    assert len(passes) == 2
    assert passes[0].resulting_threshold == 2_000_000.
    assert passes[0].upper_inclusive == 1_000_001.
    assert passes[1].lower_exclusive == 1_000_001.
    assert passes[1].upper_inclusive == 2_000_001.
    assert passes[1].certified


def test_threaded_one_pass_proposal_drives_exact_logical_widening_once(
    directory_dataset: Path, bars: pd.DataFrame, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time
    import market_analogues.certified_packed_search as certified_module
    from dataclasses import replace

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
    quantized = quantize_bound_row(represent(query))
    packed_rows = np.concatenate([
        make_packed_record(
            f"{index + 1:024x}", index + 1, 0, "A", quantized,
        )
        for index in range(5_000)
    ])
    store_root = tmp_path / "threaded-proposal-store"
    generation = write_packed_generation(
        store_root, packed_rows, np.empty(0, dtype=OVERFLOW_DTYPE),
        ("FAKE",), {"purpose": "threaded-certified-proposal-test"},
        activate=False,
    )
    packed_query = PackedBoundQuery(
        query.key.id, query.key.instrument.source_symbol,
        int(query.bars.timestamp.iloc[0].value),
        int(latest_eligible_cutoff(
            query, request.minimum_history_gap_bars,
        ).value),
        represent(query), request.quality_tiers,
    )
    keys = [
        EpisodeKey(
            InstrumentKey("test", "FAKE"),
            pd.Timestamp("2000-01-01") + pd.Timedelta(minutes=index),
            63, "dense-v1",
        )
        for index in range(4_097)
    ]
    first = sorted(keys[:4_000], key=lambda value: value.id)
    last = sorted(keys[4_000:], key=lambda value: value.id)
    ordered = first + last
    proposals = tuple(
        BoundProposal(
            key.id, "FAKE", int(key.cutoff.value), "A",
            0.0 if index < 4_000 else 101.0,
            ("composite",), False,
        )
        for index, key in enumerate(ordered)
    )
    episodes = {
        key.id: Episode(key, pd.DataFrame({"timestamp": []}), quality_tier="A")
        for key in ordered
    }

    def proposal_report(
        quota: int, *, branch_aware: bool, block_rows: int,
        block_order: str = "forward",
    ) -> BoundProposalReport:
        selected = proposals[:quota]
        candidate_digest = bound_proposal_candidate_digest(selected)
        schema = (
            BRANCH_AWARE_SEARCH_SCHEMA_VERSION
            if branch_aware else SEARCH_SCHEMA_VERSION
        )
        contract_digest = packed_bound_search_contract(
            branch_aware=branch_aware,
        )["digest"]
        proposal_input_digest = (
            _packed_query_input_digest(packed_query) if branch_aware else None
        )
        deterministic = {
            "schema_version": schema, "contract_digest": contract_digest,
            "generation_id": generation, "query_episode_id": query.key.id,
            "rows_scanned": 5_000, "eligible_rows": 5_000,
            "eligible_main_rows": 5_000, "eligible_overflow_rows": 0,
            "route_counts": {"composite": len(selected)},
            "route_quotas": {"composite": quota},
            "candidate_digest": candidate_digest,
            "real_forward_outcomes_accessed": False,
            **({"input_digest": proposal_input_digest} if branch_aware else {}),
        }
        return BoundProposalReport(
            schema, generation, query.key.id, selected, 5_000, 5_000, 5_000, 0,
            {"composite": len(selected)}, {"composite": quota}, block_rows,
            block_order, .01, 100.0, candidate_digest, stable_hash(deterministic),
            contract_digest if branch_aware else None, proposal_input_digest,
        )

    def exact_score(values, **_kwargs):
        output = []
        for proposal in values:
            episode = episodes[proposal.episode_id]
            distance = 100.0 if proposal.lower_bound == 0.0 else 200.0
            output.append(ScoredCandidate(
                AnalogueMatch(
                    episode.key, distance, {"price": distance},
                    quality_tier="A",
                ),
                episode,
            ))
        return output, [], 0.0, 1, 0

    scalar_calls: list[int] = []
    threaded_calls: list[dict[str, object]] = []

    def scalar_scan(*_args, route_quotas, block_rows, branch_aware, **_kwargs):
        quota = route_quotas["composite"]
        scalar_calls.append(quota)
        return proposal_report(
            quota, branch_aware=branch_aware, block_rows=block_rows,
        )

    def threaded_scan(*_args, route_quotas, block_rows, threads,
                      branch_aware, expected_provenance_digest, **_kwargs):
        threaded_calls.append({
            "route_quotas": route_quotas, "block_rows": block_rows,
            "threads": threads, "branch_aware": branch_aware,
            "expected_provenance_digest": expected_provenance_digest,
        })
        time.sleep(.01)
        return proposal_report(
            route_quotas["composite"], branch_aware=branch_aware,
            block_rows=block_rows,
        )

    with monkeypatch.context() as patch:
        patch.setattr(certified_module, "_score_new_proposals", exact_score)
        patch.setattr(certified_module, "scan_packed_bound_proposals", scalar_scan)
        patch.setattr(
            certified_module, "scan_packed_bound_proposals_threaded",
            threaded_scan,
        )
        scalar = certified_packed_search(
            query, source, request, store_root, generation,
            store_dataset_id="test", initial_frontier_rows=1_000,
            maximum_frontier_rows=4_096, seed_rows=1_000, block_rows=73,
            verify_content=False,
        )
        threaded = certified_packed_search(
            query, source, request, store_root, generation,
            store_dataset_id="test", initial_frontier_rows=1_000,
            maximum_frontier_rows=4_096, seed_rows=1_000, block_rows=73,
            proposal_threads=6, verify_content=False,
        )
        scalar_branch = certified_packed_search(
            query, source, request, store_root, generation,
            store_dataset_id="test", initial_frontier_rows=1_000,
            maximum_frontier_rows=4_096, seed_rows=1_000, block_rows=73,
            branch_aware_packed_bounds=True, verify_content=False,
        )
        threaded_branch = certified_packed_search(
            query, source, request, store_root, generation,
            store_dataset_id="test", initial_frontier_rows=1_000,
            maximum_frontier_rows=4_096, seed_rows=1_000, block_rows=73,
            branch_aware_packed_bounds=True, proposal_threads=4,
            verify_content=False,
        )
        bad_v1 = replace(
            proposal_report(4_097, branch_aware=False, block_rows=73),
            contract_digest="not-allowed-for-v1",
        )
        patch.setattr(
            certified_module, "scan_packed_bound_proposals_threaded",
            lambda *_args, **_kwargs: bad_v1,
        )
        with pytest.raises(
            certified_module.CertifiedPackedSearchError,
            match="precomputed proposal",
        ):
            certified_packed_search(
                query, source, request, store_root, generation,
                store_dataset_id="test", initial_frontier_rows=1_000,
                maximum_frontier_rows=4_096, seed_rows=1_000,
                proposal_threads=2, verify_content=False,
            )
        bad_v2 = replace(
            proposal_report(4_097, branch_aware=True, block_rows=73),
            input_digest="stale-query",
        )
        patch.setattr(
            certified_module, "scan_packed_bound_proposals_threaded",
            lambda *_args, **_kwargs: bad_v2,
        )
        with pytest.raises(
            certified_module.CertifiedPackedSearchError,
            match="precomputed proposal",
        ):
            certified_packed_search(
                query, source, request, store_root, generation,
                store_dataset_id="test", initial_frontier_rows=1_000,
                maximum_frontier_rows=4_096, seed_rows=1_000,
                branch_aware_packed_bounds=True, proposal_threads=2,
                verify_content=False,
            )

    assert scalar_calls == [1_001, 2_001, 4_001] * 2
    provenance_digest = stable_hash({
        "purpose": "threaded-certified-proposal-test",
    })
    assert threaded_calls == [
        {
            "route_quotas": {"composite": 4_097}, "block_rows": 73,
            "threads": 6, "branch_aware": False,
            "expected_provenance_digest": provenance_digest,
        },
        {
            "route_quotas": {"composite": 4_097}, "block_rows": 73,
            "threads": 4, "branch_aware": True,
            "expected_provenance_digest": provenance_digest,
        },
    ]
    assert list(dict.fromkeys(
        row.frontier_rows for row in threaded.certificate.rounds
    )) == [
        1_000, 2_000, 4_000,
    ]
    assert threaded.certificate.result_digest == scalar.certificate.result_digest
    assert threaded.matches == scalar.matches
    assert threaded_branch.certificate.result_digest == (
        scalar_branch.certificate.result_digest
    )
    assert threaded_branch.matches == scalar_branch.matches
    assert threaded_branch.certificate.schema_version == (
        "m04r-certified-packed-search-v8"
    )
    assert threaded.certificate.elapsed_seconds >= .01
    brute = select_scored(exact_score(proposals)[0], request)
    assert threaded.matches == tuple(brute)


def test_requested_position_modes_are_mutually_exclusive(
    directory_dataset: Path, bars: pd.DataFrame, tmp_path: Path,
) -> None:
    from market_analogues.certified_packed_search import (
        certified_packed_search_contract,
    )

    with np.testing.assert_raises(ValueError):
        certified_packed_search_contract(
            requested_positions=True, hybrid_requested_positions=True,
        )
    with np.testing.assert_raises(ValueError):
        certified_packed_search_contract(vector_lower_bounds=True)
    with np.testing.assert_raises(ValueError):
        certified_packed_search_contract(deferred_alignments=True)
    with np.testing.assert_raises(ValueError):
        certified_packed_search_contract(compact_scored=True)
    native_contract = certified_packed_search_contract(native_bound_deferral=True)
    assert native_contract["schema_version"] == "m04r-certified-packed-search-v7"
    assert native_contract == certified_packed_search_contract(
        native_bound_deferral=True,
    )
    branch_contract = certified_packed_search_contract(
        branch_aware_packed_bounds=True,
    )
    assert branch_contract["schema_version"] == "m04r-certified-packed-search-v8"
    assert branch_contract["packed_bound"].startswith("branch-aware")


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), -1.0])
def test_certified_search_rejects_invalid_tolerance(tolerance: float) -> None:
    query_key = EpisodeKey(
        InstrumentKey("test", "QUERY"), pd.Timestamp("2024-12-31"), 10, "v1",
    )
    query = Episode(query_key, pd.DataFrame())
    request = SearchQuery(query_key, ("test",), ("A",), 1)
    with pytest.raises(ValueError, match="tolerance"):
        certified_packed_search(
            query, object(), request, Path("unused"), "generation",
            store_dataset_id="test", initial_frontier_rows=1,
            maximum_frontier_rows=1, seed_rows=1, tolerance=tolerance,
        )


@pytest.mark.parametrize("proposal_threads", [0, -1, 1.5, True])
def test_certified_search_rejects_invalid_proposal_thread_count(
    proposal_threads: object,
) -> None:
    query_key = EpisodeKey(
        InstrumentKey("test", "QUERY"), pd.Timestamp("2024-12-31"), 10, "v1",
    )
    query = Episode(query_key, pd.DataFrame())
    request = SearchQuery(query_key, ("test",), ("A",), 1)
    with pytest.raises(ValueError, match="proposal threads"):
        certified_packed_search(
            query, object(), request, Path("unused"), "generation",
            store_dataset_id="test", initial_frontier_rows=1,
            maximum_frontier_rows=1, seed_rows=1,
            proposal_threads=proposal_threads,  # type: ignore[arg-type]
        )


def test_certified_search_rejects_two_precomputed_proposal_sources() -> None:
    query_key = EpisodeKey(
        InstrumentKey("test", "QUERY"), pd.Timestamp("2024-12-31"), 10, "v1",
    )
    query = Episode(query_key, pd.DataFrame())
    request = SearchQuery(query_key, ("test",), ("A",), 1)
    with pytest.raises(ValueError, match="mutually exclusive"):
        certified_packed_search(
            query, object(), request, Path("unused"), "generation",
            store_dataset_id="test", initial_frontier_rows=1,
            maximum_frontier_rows=1, seed_rows=1, proposal_threads=1,
            precomputed_proposal=object(),  # type: ignore[arg-type]
        )


def test_native_bound_closure_reopens_threshold_and_completes_equality_tie() -> None:
    from types import SimpleNamespace

    instrument = InstrumentKey("test", "S")
    other = InstrumentKey("test", "Y")
    query_key = EpisodeKey(
        InstrumentKey("test", "QUERY"), pd.Timestamp("2024-12-31"), 252, "v1",
    )
    request = SearchQuery(query_key, ("test",), ("A",), 3, max_per_instrument=3)

    def compact(label: str, distance: float, start: int, cutoff: int,
                key_instrument: InstrumentKey = instrument) -> CompactScoredCandidate:
        key = EpisodeKey(
            key_instrument,
            pd.Timestamp("2020-01-01") + pd.Timedelta(days=cutoff),
            252, f"v1-{label}",
        )
        return CompactScoredCandidate(
            AnalogueMatch(key, distance, {"price": distance}),
            key_instrument, start, cutoff,
        )

    initial = [
        compact("A", .20, 0, 251), compact("B", .30, 252, 503),
        compact("C", .40, 504, 755), compact("D", .50, 756, 1007),
    ]
    exact_x = compact("X", .10, 126, 377)
    exact_y = compact("Y", .50, 0, 251, other)
    deferred = {
        exact_x.match.episode_key.id: SimpleNamespace(
            lower_bound=.10,
            episode_key=exact_x.match.episode_key,
        ),
        exact_y.match.episode_key.id: SimpleNamespace(
            lower_bound=.50,
            episode_key=exact_y.match.episode_key,
        ),
    }
    exact = {
        exact_x.match.episode_key.id: exact_x,
        exact_y.match.episode_key.id: exact_y,
    }
    completed: list[str] = []

    def complete(values: list[object]) -> list[CompactScoredCandidate]:
        episode_ids = [value.episode_key.id for value in values]  # type: ignore[attr-defined]
        completed.extend(episode_ids)
        return [exact[episode_id] for episode_id in episode_ids]

    selected, threshold, count = _close_native_bound_deferred(
        {row.match.episode_key.id: row for row in initial}, deferred,
        request, compact=True, complete_many=complete,
    )
    assert completed == [
        exact_x.match.episode_key.id, exact_y.match.episode_key.id,
    ]
    assert count == 2
    assert len(selected) == 3
    assert threshold == .5
    assert not deferred


def test_native_bound_closure_completes_all_when_selection_is_incomplete() -> None:
    from types import SimpleNamespace

    instrument = InstrumentKey("test", "S")
    query_key = EpisodeKey(
        InstrumentKey("test", "QUERY"), pd.Timestamp("2024-12-31"), 10, "v1",
    )
    request = SearchQuery(query_key, ("test",), ("A",), 2, max_per_instrument=2)
    first_key = EpisodeKey(instrument, pd.Timestamp("2020-01-10"), 10, "first")
    second_key = EpisodeKey(instrument, pd.Timestamp("2020-02-10"), 10, "second")
    first = CompactScoredCandidate(
        AnalogueMatch(first_key, .1, {"price": .1}), instrument, 0, 9,
    )
    second = CompactScoredCandidate(
        AnalogueMatch(second_key, 1000., {"price": 1000.}), instrument, 20, 29,
    )
    deferred = {
        second_key.id: SimpleNamespace(
            lower_bound=999., episode_key=second_key,
        ),
    }
    selected, threshold, count = _close_native_bound_deferred(
        {first_key.id: first}, deferred, request, compact=True,
        complete_many=lambda _: [second],
    )
    assert count == 1
    assert len(selected) == 2
    assert threshold == 1000.
    assert not deferred


def test_compact_exact_state_is_selection_equivalent_with_overlap_and_ties() -> None:
    query_key = EpisodeKey(
        InstrumentKey("test", "QUERY"), pd.Timestamp("2024-12-31"), 10, "v1",
    )
    request = SearchQuery(
        query_key, ("test",), ("A",), 20,
        deduplicate_overlaps=True, max_per_instrument=3,
    )
    scored = []
    for symbol_index in range(12):
        instrument = InstrumentKey("test", f"S{symbol_index:02d}")
        for row in range(30):
            start = pd.Timestamp("2020-01-01") + pd.Timedelta(days=row * 3)
            timestamps = pd.date_range(start, periods=10, freq="D")
            key = EpisodeKey(instrument, timestamps[-1], 10, "v1")
            # Repeated distances exercise stable episode-ID tie ordering.
            distance = float((row * 7 + symbol_index * 3) % 17) / 10
            episode = Episode(key, pd.DataFrame({"timestamp": timestamps}))
            match = AnalogueMatch(key, distance, {"price": distance})
            scored.append(ScoredCandidate(match, episode))
    full = select_scored(scored, request)
    compact = [
        CompactScoredCandidate(
            item.match, item.episode.key.instrument,
            int(item.episode.bars.timestamp.iloc[0].value),
            int(item.episode.bars.timestamp.iloc[-1].value),
        )
        for item in reversed(scored)
    ]
    reduced = _select_compact_scored(compact, request)
    assert [row.episode_key.id for row in reduced] == [
        row.episode_key.id for row in full
    ]
    assert len(compact) == len(scored)


def test_compact_exact_state_retains_candidate_exposed_by_later_overlap() -> None:
    instrument = InstrumentKey("test", "S")
    query_key = EpisodeKey(
        InstrumentKey("test", "QUERY"), pd.Timestamp("2024-12-31"), 252, "v1",
    )
    request = SearchQuery(query_key, ("test",), ("A",), 3, max_per_instrument=3)
    rows = [
        ("A", .20, 0, 251), ("B", .30, 252, 503),
        ("C", .40, 504, 755), ("D", .50, 756, 1007),
        ("X", .10, 126, 377),
    ]
    compact = []
    labels = {}
    for label, distance, start, cutoff in rows:
        key = EpisodeKey(
            instrument, pd.Timestamp("2020-01-01") + pd.Timedelta(days=cutoff),
            252, f"v1-{label}",
        )
        labels[key.id] = label
        compact.append(CompactScoredCandidate(
            AnalogueMatch(key, distance, {"price": distance}),
            instrument, start, cutoff,
        ))
    selected = _select_compact_scored(compact, request)
    assert [labels[row.episode_key.id] for row in selected] == ["X", "C", "D"]


def test_optional_empty_volume_stages_are_warning_free_across_threads() -> None:
    rows = 16
    width = 252
    channels = {
        "close_path": np.zeros((rows, width)),
        "return": np.zeros((rows, width)),
        "relative_return": np.zeros((rows, width)),
        "volume_robust_z": np.full((rows, width), np.nan),
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with ThreadPoolExecutor(max_workers=8) as executor:
            output = list(executor.map(
                lambda _: _stage_rows(channels), range(64),
            ))
    runtime = [
        item for item in caught if issubclass(item.category, RuntimeWarning)
    ]
    assert not runtime
    assert all(np.isfinite(value).all() for value in output)
