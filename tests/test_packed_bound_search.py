from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from market_analogues.packed_bound_search import (
    COMPONENT_SEARCH_SCHEMA_VERSION, PackedBoundQuery, PackedBoundSearchError,
    packed_component_search_contract, packed_component_threshold_scan_contract,
    scan_packed_bound_proposals,
    scan_packed_component_bound_proposals_threaded,
    scan_packed_bound_proposals_many, scan_packed_bound_proposals_threaded,
    scan_packed_bound_threshold,
)
from market_analogues.packed_bound_store import (
    OVERFLOW_DTYPE, make_overflow_record, make_packed_record,
    write_packed_generation,
)
from market_analogues.quantized_bound import quantize_bound_row
from market_analogues.representation import represent
from market_analogues.synthetic import generate_case


def _store(tmp_path: Path, count: int = 1_037) -> tuple[Path, str, object]:
    query_representation = represent(generate_case("rounded_base", 900).episode)
    quantized = quantize_bound_row(query_representation)
    rows = []
    # Physical order and episode-ID tie order deliberately disagree.
    for index in range(count):
        rows.append(make_packed_record(
            f"{count - index + 100:024x}", index + 1, 0,
            "A" if index % 2 == 0 else "B", quantized,
        ))
    overflow = make_overflow_record(f"{1:024x}", 1, 1, "B")
    root = tmp_path / "store"
    generation = write_packed_generation(
        root, np.concatenate(rows), overflow, ("AAA", "BBB"),
        {"purpose": "global-bound-search-test"}, activate=False,
    )
    return root, generation, query_representation


def _query(representation: object, **overrides: object) -> PackedBoundQuery:
    values = {
        "episode_id": f"{999_999:024x}",
        "symbol": "QQQ",
        "query_start_ns": 10,
        "latest_eligible_ns": 10_000,
        "representation": representation,
        "quality_tiers": ("A", "B"),
    }
    values.update(overrides)
    return PackedBoundQuery(**values)  # type: ignore[arg-type]


def test_global_selection_is_stable_across_blocks_and_scan_order(tmp_path: Path) -> None:
    root, generation, representation = _store(tmp_path)
    reports = [
        scan_packed_bound_proposals(
            root, generation, _query(representation), block_rows=block_rows,
            block_order=order, verify_content=False,
        )
        for block_rows, order in ((1, "forward"), (29, "forward"), (64, "reverse"))
    ]
    assert len({report.candidate_digest for report in reports}) == 1
    assert len({report.result_digest for report in reports}) == 1
    assert all(report.rows_scanned == 1_038 for report in reports)
    assert all(report.eligible_rows == 1_038 for report in reports)
    # The zero-bound exact fallback is never omitted and every route keeps its
    # own stable bounded result without displacing composite top-1000 rows.
    assert all(report.candidates[0].episode_id == f"{1:024x}" for report in reports)
    assert all(report.candidates[0].overflow_fallback for report in reports)
    composite = {
        row.episode_id for row in reports[0].candidates
        if "composite" in row.routes
    }
    assert len(composite) == 1_000
    expected = {f"{1:024x}"} | {
        f"{value:024x}" for value in range(101, 1_100)
    }
    assert composite == expected


def test_threaded_legacy_scan_is_scalar_exact_and_order_stable(tmp_path: Path) -> None:
    root, generation, representation = _store(tmp_path)
    query = _query(
        representation, symbol="AAA", query_start_ns=500,
        latest_eligible_ns=900, quality_tiers=("A",),
    )
    scalar = scan_packed_bound_proposals(
        root, generation, query, block_rows=31, block_order="forward",
        verify_content=False,
    )
    threaded = tuple(
        scan_packed_bound_proposals_threaded(
            root, generation, query, block_rows=block_rows,
            block_order=order, threads=threads, verify_content=False,
        )
        for block_rows, order, threads in (
            (17, "forward", 2), (29, "reverse", 4), (64, "forward", 4),
        )
    )
    for report in threaded:
        assert report.candidates == scalar.candidates
        assert report.candidate_digest == scalar.candidate_digest
        assert report.result_digest == scalar.result_digest
        assert report.rows_scanned == scalar.rows_scanned
        assert report.eligible_rows == scalar.eligible_rows
        assert report.eligible_main_rows == scalar.eligible_main_rows
        assert report.eligible_overflow_rows == scalar.eligible_overflow_rows
        assert report.route_counts == scalar.route_counts
        assert report.route_quotas == scalar.route_quotas
        assert report.contract_digest is None
        assert report.input_digest is None

    with pytest.raises(PackedBoundSearchError, match="positive"):
        scan_packed_bound_proposals_threaded(
            root, generation, query, threads=0, verify_content=False,
        )
    with pytest.raises(PackedBoundSearchError, match="block order"):
        scan_packed_bound_proposals_threaded(
            root, generation, query, block_order="shuffled",
            verify_content=False,
        )


def _assert_semantic_report_parity(left: object, right: object) -> None:
    for name in (
        "schema_version", "generation_id", "query_episode_id", "candidates",
        "rows_scanned", "eligible_rows", "eligible_main_rows",
        "eligible_overflow_rows", "route_counts", "route_quotas",
        "candidate_digest", "result_digest", "contract_digest", "input_digest",
    ):
        assert getattr(left, name) == getattr(right, name)


def test_threaded_branch_aware_is_scalar_exact_across_order_blocks_threads_ties_and_overflow(
    tmp_path: Path,
) -> None:
    root, generation, representation = _store(tmp_path)
    query = _query(representation)
    scalar = scan_packed_bound_proposals(
        root, generation, query, block_rows=37, branch_aware=True,
        verify_content=False,
    )
    assert scalar.schema_version == "m04r-global-bound-proposal-v2"
    assert scalar.contract_digest is not None and scalar.input_digest is not None
    assert any(row.overflow_fallback for row in scalar.candidates)
    # All main rows are identical packed representations, exercising the
    # score/episode-ID tie boundary independently of physical block order.
    for block_rows, order, threads in (
        (1, "forward", 1), (17, "forward", 2),
        (29, "reverse", 3), (257, "reverse", 7),
    ):
        threaded = scan_packed_bound_proposals_threaded(
            root, generation, query, block_rows=block_rows,
            block_order=order, threads=threads, branch_aware=True,
            verify_content=False,
        )
        _assert_semantic_report_parity(threaded, scalar)


def test_component_frontier_publishes_component_bound_and_is_order_stable(
    tmp_path: Path,
) -> None:
    root, generation, representation = _store(tmp_path)
    query = _query(representation)
    reports = tuple(
        scan_packed_component_bound_proposals_threaded(
            root, generation, query, component="price", quota=127,
            block_rows=block_rows, block_order=order, threads=threads,
            verify_content=False,
        )
        for block_rows, order, threads in (
            (1, "forward", 1), (23, "forward", 3), (41, "reverse", 7),
        )
    )
    first = reports[0]
    assert first.schema_version == COMPONENT_SEARCH_SCHEMA_VERSION
    assert first.contract_digest == packed_component_search_contract("price")["digest"]
    assert first.route_quotas == {"price": 127}
    assert first.route_counts == {"price": 127}
    assert len(first.candidates) == 127
    assert all(row.routes == ("price",) for row in first.candidates)
    assert all(row.lower_bound == 0 for row in first.candidates)
    for report in reports[1:]:
        _assert_semantic_report_parity(report, first)

    with pytest.raises(PackedBoundSearchError, match="unsupported"):
        packed_component_search_contract("composite")
    with pytest.raises(PackedBoundSearchError, match="positive"):
        scan_packed_component_bound_proposals_threaded(
            root, generation, query, component="price", quota=0,
            verify_content=False,
        )


def test_threaded_branch_aware_preserves_exclusions_and_zero_or_short_heaps(
    tmp_path: Path,
) -> None:
    root, generation, representation = _store(tmp_path, count=8)
    excluded = _query(
        representation, episode_id=f"{108:024x}", symbol="AAA",
        query_start_ns=5, latest_eligible_ns=8, quality_tiers=("A",),
    )
    scalar = scan_packed_bound_proposals(
        root, generation, excluded, block_rows=3, branch_aware=True,
        verify_content=False,
    )
    threaded = scan_packed_bound_proposals_threaded(
        root, generation, excluded, block_rows=5, block_order="reverse",
        threads=6, branch_aware=True, verify_content=False,
    )
    _assert_semantic_report_parity(threaded, scalar)
    assert scalar.eligible_rows == 1
    assert all(row.episode_id != excluded.episode_id for row in scalar.candidates)
    assert all(row.symbol == "AAA" and row.cutoff_ns < 5 for row in scalar.candidates)
    assert all(
        count < scalar.route_quotas[route]
        for route, count in scalar.route_counts.items()
    )

    empty = _query(
        representation, query_start_ns=0, latest_eligible_ns=0,
    )
    empty_scalar = scan_packed_bound_proposals(
        root, generation, empty, block_rows=2, branch_aware=True,
        verify_content=False,
    )
    empty_threaded = scan_packed_bound_proposals_threaded(
        root, generation, empty, block_rows=11, block_order="reverse",
        threads=8, branch_aware=True, verify_content=False,
    )
    _assert_semantic_report_parity(empty_threaded, empty_scalar)
    assert empty_scalar.candidates == ()
    assert set(empty_scalar.route_counts.values()) == {0}


def test_threaded_branch_aware_matches_scalar_at_adversarial_float16_boundaries(
    tmp_path: Path,
) -> None:
    query_representation = represent(generate_case("steady_trend", 41_001).episode)
    candidate = represent(generate_case("rounded_base", 41_002).episode)
    boundary = np.asarray([
        0.0,
        float(np.nextafter(np.float16(0), np.float16(1))),
        float(np.nextafter(np.float16(1), np.float16(2))),
        -float(np.nextafter(np.float16(1), np.float16(2))),
        np.finfo(np.float16).max,
        -np.finfo(np.float16).max,
    ])
    boundary_candidate = replace(
        candidate, coarse=np.resize(boundary, 128).astype(np.float64),
    )
    candidates = (query_representation, candidate, boundary_candidate)
    rows = []
    for index in range(15):
        rows.append(make_packed_record(
            f"{index + 200:024x}", index + 1, 0, "A",
            quantize_bound_row(candidates[index % len(candidates)]),
        ))
    root = tmp_path / "branch-boundary-store"
    generation = write_packed_generation(
        root, np.concatenate(rows),
        make_overflow_record(f"{1:024x}", 1, 1, "B"),
        ("AAA", "BBB"), {"purpose": "threaded-branch-boundary-test"},
        activate=False,
    )
    query = _query(query_representation)
    scalar = scan_packed_bound_proposals(
        root, generation, query, block_rows=4, branch_aware=True,
        verify_content=False,
    )
    assert all(np.isfinite(row.lower_bound) for row in scalar.candidates)
    for block_rows, order, threads in (
        (2, "forward", 2), (7, "reverse", 4), (64, "forward", 8),
    ):
        threaded = scan_packed_bound_proposals_threaded(
            root, generation, query, block_rows=block_rows,
            block_order=order, threads=threads, branch_aware=True,
            verify_content=False,
        )
        _assert_semantic_report_parity(threaded, scalar)


def test_eligibility_precedes_ranking_and_rejects_bad_contracts(tmp_path: Path) -> None:
    root, generation, representation = _store(tmp_path, count=8)
    # AAA cutoffs 5..8 overlap the query; alternating B rows and the B-tier
    # overflow sidecar are excluded by the requested A-only tier.
    report = scan_packed_bound_proposals(
        root, generation,
        _query(
            representation, symbol="AAA", query_start_ns=5,
            latest_eligible_ns=8, quality_tiers=("A",),
        ),
        verify_content=True,
    )
    assert report.rows_scanned == 9
    assert report.eligible_rows == 2
    assert all(row.symbol == "AAA" and row.cutoff_ns < 5 for row in report.candidates)
    assert all(row.quality_tier == "A" for row in report.candidates)
    assert not any(row.overflow_fallback for row in report.candidates)

    with pytest.raises(PackedBoundSearchError, match="quota >= 1000"):
        scan_packed_bound_proposals(
            root, generation, _query(representation),
            route_quotas={"composite": 999}, verify_content=False,
        )
    with pytest.raises(PackedBoundSearchError, match="block order"):
        scan_packed_bound_proposals(
            root, generation, _query(representation),
            block_order="shuffled", verify_content=False,
        )


def test_full_96_bit_query_id_is_excluded_even_with_trailing_zero(tmp_path: Path) -> None:
    root, generation, representation = _store(tmp_path, count=156)
    query_id = f"{256:024x}"
    assert query_id.endswith("00")
    report = scan_packed_bound_proposals(
        root, generation,
        _query(representation, episode_id=query_id),
        verify_content=False,
    )
    assert report.eligible_rows == 156
    assert query_id not in {row.episode_id for row in report.candidates}
    with pytest.raises(PackedBoundSearchError, match="lowercase"):
        _query(representation, episode_id="ABCDEF0123456789ABCDEF01")


def test_shared_scan_is_scalar_equivalent_and_order_stable(tmp_path: Path) -> None:
    root, generation, representation = _store(tmp_path)
    queries = (
        _query(representation, episode_id=f"{900_001:024x}"),
        _query(
            representation, episode_id=f"{900_002:024x}", symbol="AAA",
            query_start_ns=500, latest_eligible_ns=900,
            quality_tiers=("A",),
        ),
    )
    scalar = tuple(scan_packed_bound_proposals(
        root, generation, query, block_rows=31, verify_content=False,
    ) for query in queries)
    batches = tuple(scan_packed_bound_proposals_many(
        root, generation, queries, block_rows=block_rows, block_order=order,
        verify_content=False,
    ) for block_rows, order in ((17, "forward"), (64, "reverse")))
    assert all(batch.query_episode_ids == tuple(
        query.episode_id for query in queries
    ) for batch in batches)
    assert all(batch.physical_rows_scanned == 1_038 for batch in batches)
    assert all(batch.logical_rows_evaluated == 2_076 for batch in batches)
    assert len({batch.result_digest for batch in batches}) == 1
    for batch in batches:
        for shared, independent in zip(batch.reports, scalar):
            assert shared.candidates == independent.candidates
            assert shared.candidate_digest == independent.candidate_digest
            assert shared.result_digest == independent.result_digest
            assert shared.eligible_rows == independent.eligible_rows
            assert shared.route_counts == independent.route_counts

    branch_scalar = tuple(scan_packed_bound_proposals(
        root, generation, query, block_rows=23, branch_aware=True,
        verify_content=False,
    ) for query in queries)
    branch_batch = scan_packed_bound_proposals_many(
        root, generation, queries, block_rows=19, block_order="reverse",
        branch_aware=True, verify_content=False,
    )
    assert branch_batch.schema_version == "m04r-global-bound-proposal-batch-v2"
    for shared, independent in zip(branch_batch.reports, branch_scalar):
        assert shared.schema_version == "m04r-global-bound-proposal-v2"
        assert shared.candidates == independent.candidates
        assert shared.result_digest == independent.result_digest
        assert shared.contract_digest == independent.contract_digest
        assert shared.input_digest == independent.input_digest
        assert shared.contract_digest is not None
        assert shared.input_digest is not None
    assert all(report.contract_digest is None for report in scalar)
    assert all(report.input_digest is None for report in scalar)

    changed_samples = dict(representation.samples_48)
    changed_samples["close_path"] = changed_samples["close_path"].copy()
    changed_samples["close_path"][0] += 1.0
    changed_representation = type(representation)(
        representation.channels, representation.coarse, changed_samples,
        representation.samples_64, representation.stage,
        representation.structural,
    )
    changed_report = scan_packed_bound_proposals(
        root, generation, _query(
            changed_representation, episode_id=queries[0].episode_id,
        ),
        block_rows=23, branch_aware=True, verify_content=False,
    )
    assert changed_report.input_digest != branch_scalar[0].input_digest
    assert changed_report.result_digest != branch_scalar[0].result_digest


def test_shared_scan_rejects_empty_and_duplicate_queries(tmp_path: Path) -> None:
    root, generation, representation = _store(tmp_path, count=8)
    query = _query(representation)
    with pytest.raises(PackedBoundSearchError, match="non-empty"):
        scan_packed_bound_proposals_many(
            root, generation, (), verify_content=False,
        )
    with pytest.raises(PackedBoundSearchError, match="unique"):
        scan_packed_bound_proposals_many(
            root, generation, (query, query), verify_content=False,
        )


def test_threshold_scan_streams_complete_tied_bands_and_is_order_independent(
    tmp_path: Path,
) -> None:
    query_representation = represent(generate_case("rounded_base", 900).episode)
    rows = []
    kinds = (
        "rounded_base", "trend_contraction_breakout",
        "failed_breakout", "volatile_reversal",
    )
    for index in range(48):
        candidate = represent(generate_case(kinds[index % len(kinds)], 2_000 + index).episode)
        rows.append(make_packed_record(
            f"{index + 100:024x}", index + 1, 0, "A",
            quantize_bound_row(candidate),
        ))
    overflow_id = f"{1:024x}"
    root = tmp_path / "threshold-store"
    generation = write_packed_generation(
        root, np.concatenate(rows),
        make_overflow_record(overflow_id, 1, 1, "B"),
        ("AAA", "BBB"), {"purpose": "threshold-scan-test"}, activate=False,
    )
    query = _query(query_representation)
    ranked = scan_packed_bound_proposals(
        root, generation, query, verify_content=False,
    ).candidates
    boundary = ranked[len(ranked) // 2].lower_bound
    prefix_id = ranked[0].episode_id
    expected_first = {
        row.episode_id for row in ranked
        if row.lower_bound <= boundary and row.episode_id != prefix_id
    }

    reports = []
    admitted_sets = []
    for block_rows, order in ((7, "forward"), (11, "reverse")):
        admitted = []
        reports.append(scan_packed_bound_threshold(
            root, generation, query, upper_inclusive=boundary,
            excluded_episode_ids=frozenset({prefix_id}),
            block_rows=block_rows, block_order=order, verify_content=False,
            consume=lambda batch, output=admitted: output.extend(batch),
        ))
        admitted_sets.append({row.episode_id for row in admitted})
    assert admitted_sets == [expected_first, expected_first]
    assert reports[0].result_digest == reports[1].result_digest
    assert reports[0].admitted_set_digest == reports[1].admitted_set_digest
    assert reports[0].minimum_above_upper is not None
    assert reports[0].minimum_above_upper > boundary

    second = []
    second_report = scan_packed_bound_threshold(
        root, generation, query, lower_exclusive=boundary,
        upper_inclusive=float("inf"), block_rows=9, verify_content=False,
        consume=lambda batch: second.extend(batch),
    )
    assert second_report.minimum_above_upper is None
    assert {row.episode_id for row in second} == {
        row.episode_id for row in ranked if row.lower_bound > boundary
    }
    assert expected_first | {prefix_id} | {row.episode_id for row in second} == {
        row.episode_id for row in ranked
    }


def test_component_threshold_scan_is_complete_and_order_independent(
    tmp_path: Path,
) -> None:
    root, generation, representation = _store(tmp_path, count=128)
    query = _query(representation)
    frontier = scan_packed_component_bound_proposals_threaded(
        root, generation, query, component="price", quota=129,
        threads=3, verify_content=False,
    )
    boundary = frontier.candidates[64].lower_bound
    reports = []
    admitted = []
    for block_rows, order in ((7, "forward"), (13, "reverse")):
        values = []
        reports.append(scan_packed_bound_threshold(
            root, generation, query, component="price", branch_aware=True,
            upper_inclusive=boundary, block_rows=block_rows, block_order=order,
            verify_content=False,
            consume=lambda rows, output=values: output.extend(rows),
        ))
        admitted.append(values)
    assert reports[0].schema_version == "m04r-packed-component-threshold-scan-v1"
    assert reports[0].contract_digest == packed_component_threshold_scan_contract("price")["digest"]
    assert reports[0].result_digest == reports[1].result_digest
    assert reports[0].admitted_set_digest == reports[1].admitted_set_digest
    assert {row.episode_id for row in admitted[0]} == {row.episode_id for row in admitted[1]}
    assert all(row.routes == ("price",) and row.lower_bound <= boundary
               for rows in admitted for row in rows)

    with pytest.raises(PackedBoundSearchError, match="branch-aware"):
        scan_packed_bound_threshold(
            root, generation, query, component="price",
            upper_inclusive=boundary, verify_content=False,
            consume=lambda _rows: None,
        )

def test_threshold_scan_fails_closed_on_invalid_bounds_and_exclusions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import market_analogues.packed_bound_search as packed_search

    root, generation, representation = _store(tmp_path, count=8)
    query = _query(representation)
    for invalid in (float("nan"), -1.0, float("inf")):
        monkeypatch.setattr(
            packed_search, "packed_lower_bounds",
            lambda _query_representation, records, value=invalid: SimpleNamespace(
                totals=np.full(len(records), value),
            ),
        )
        with pytest.raises(PackedBoundSearchError, match="invalid packed lower bound"):
            scan_packed_bound_threshold(
                root, generation, query, upper_inclusive=float("inf"),
                consume=lambda _: None, verify_content=False,
            )
    with pytest.raises(PackedBoundSearchError, match="invalid episode ID"):
        scan_packed_bound_threshold(
            root, generation, query, upper_inclusive=1.0,
            excluded_episode_ids=frozenset({"g" * 24}),
            consume=lambda _: None, verify_content=False,
        )
