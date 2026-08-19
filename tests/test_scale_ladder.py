from pathlib import Path

import pandas as pd

from market_analogues.adapters import DirectorySource
from market_analogues.config import DatasetSpec
from market_analogues.episodes import build_episode
from market_analogues.scale_ladder import (
    deterministic_instrument_prefixes, run_scale_ladder,
    write_scale_ladder_report,
)
from market_analogues.types import InstrumentKey, SearchQuery


def _inputs(directory_dataset: Path, bars: pd.DataFrame):
    for index, symbol in enumerate(("CCC", "DDD"), 3):
        bars.assign(
            open=bars.open * index, high=bars.high * index,
            low=bars.low * index, close=bars.close * index,
        ).to_parquet(directory_dataset / f"{symbol}.parquet", index=False)
    source = DirectorySource(DatasetSpec(
        "test", "directory", directory_dataset, "parquet", timestamp_column="date",
    ))
    query = build_episode(
        source, InstrumentKey("test", "AAA"), bars.date.iloc[-1], 63, "dense-v1",
    )
    request = SearchQuery(
        query.key, ("test",), ("A", "B"), 1,
        minimum_history_gap_bars=20, max_per_instrument=1,
    )
    quality = pd.DataFrame({
        "symbol": ["AAA", "BBB", "CCC", "DDD"],
        "tier": ["A", "A", "B", "QUARANTINED"],
        "issues": ["", "", "", "bad"],
    })
    return source, query, request, quality


def test_scale_prefixes_are_nested_deterministic_and_quality_filtered(
    directory_dataset: Path, bars: pd.DataFrame,
) -> None:
    source, _, request, quality = _inputs(directory_dataset, bars)
    fractions = (0.25, 0.5, 1.0)
    first = deterministic_instrument_prefixes(
        source, quality, request, fractions, seed="locked",
    )
    second = deterministic_instrument_prefixes(
        source, quality.sample(frac=1, random_state=4), request, fractions,
        seed="locked",
    )
    assert first == second
    assert set(first[0]).issubset(first[1])
    assert set(first[1]).issubset(first[2])
    assert len(first[2]) == 3
    assert all(key.source_symbol != "DDD" for key in first[2])


def test_scale_ladder_runs_resume_repeat_and_report(
    directory_dataset: Path, bars: pd.DataFrame, tmp_path: Path,
) -> None:
    source, query, request, quality = _inputs(directory_dataset, bars)
    result = run_scale_ladder(
        query, source, request, quality, tmp_path / "ladder",
        fractions=(1.0,), seed="locked", batch_size=11,
        frontier_batch_rows=1, representation_cache_shards=2,
        maximum_rss_mb=4096, disk_reserve_bytes=0,
        maximum_projected_hours=1,
    )
    assert result.passed
    rung = result.rungs[0]
    assert rung.selected_instruments == result.total_eligible_instruments == 3
    assert rung.result_digest == rung.repeated_digest
    assert rung.eligible_candidates == rung.exact_evaluated + rung.safely_pruned
    assert rung.projected_full_search_seconds == rung.search_seconds
    report = write_scale_ladder_report(result, tmp_path / "report.html")
    assert "deterministic scale ladder: PASS" in report.read_text()
