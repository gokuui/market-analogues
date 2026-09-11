from pathlib import Path

from market_analogues.adapters import DirectorySource
from market_analogues.config import DatasetSpec
from market_analogues.episodes import build_episode
from market_analogues.quality import audit_source
from market_analogues.types import InstrumentKey, SearchQuery
from market_analogues.universe import verify_universe, write_universe_report


def test_universe_verifier_reconciles_coverage_and_writes_report(
    directory_dataset: Path, tmp_path: Path,
) -> None:
    source = DirectorySource(DatasetSpec(
        "test", "directory", directory_dataset, "parquet", timestamp_column="date",
    ))
    quality = audit_source(source)
    query = build_episode(source, InstrumentKey("test", "AAA"), "2021-12-31", 126, "dense-v1")
    request = SearchQuery(query.key, ("test",), ("A", "B"), 3)
    result = verify_universe(
        query, source, request, quality, stride=10, reference_pool=10,
        comparison_pools=(2, 5), recall_pool=5, minimum_pool_recall=0,
        workers=2, max_seconds=60, max_rss_mb=16384,
    )
    assert result.passed
    assert result.metrics["instruments_considered"] == 2
    assert result.metrics["instruments_scanned"] == 2
    assert result.metrics["candidate_digest"]
    output = write_universe_report(result, tmp_path / "universe.html")
    assert "Complete-universe verification" in output.read_text()
