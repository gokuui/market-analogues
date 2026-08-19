from pathlib import Path

import pandas as pd

from market_analogues.adapters import DirectorySource
from market_analogues.config import DatasetSpec
from market_analogues.episodes import build_episode
from market_analogues.production_verification import (
    verify_production_search, write_production_search_report,
)
from market_analogues.types import InstrumentKey, SearchQuery
from market_analogues.view_store import build_view_store


def test_repeated_production_search_is_stable_and_budgeted(
    directory_dataset: Path, tmp_path: Path,
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
        query.key, ("test",), ("A", "B"), 3, minimum_history_gap_bars=20,
    )

    result = verify_production_search(
        query, source, request, root, quality=quality,
        candidate_pool=12, per_instrument_view=3, workers=2,
        repeat=2, max_seconds=30, max_rss_mb=4096,
    )

    assert result.passed, result.failures
    assert result.metrics["repeat"] == 2
    assert result.metrics["maximum_repeat_delta"] == 0
    assert result.runs.digest.nunique() == 1
    report = write_production_search_report(result, tmp_path / "production.html")
    report_text = report.read_text().lower()
    assert "production persisted search" in report_text
    assert "benchmark fingerprint" in report_text
