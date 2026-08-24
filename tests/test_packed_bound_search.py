from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from market_analogues.packed_bound_search import (
    PackedBoundQuery, PackedBoundSearchError, scan_packed_bound_proposals,
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
