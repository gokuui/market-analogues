from pathlib import Path

from market_analogues.adapters import DirectorySource
from market_analogues.config import DatasetSpec
from market_analogues.oracle import run_oracle_suite, write_oracle_artifacts
from market_analogues.quality import audit_source


def test_sampled_oracle_is_exhaustive_and_deterministic(
    directory_dataset: Path, tmp_path: Path,
) -> None:
    source = DirectorySource(DatasetSpec(
        "test", "directory", directory_dataset, "parquet", timestamp_column="date",
    ))
    quality = audit_source(source)
    kwargs = dict(
        symbols=("AAA",), lookbacks=(126,), cutoff_quantiles=(1.0,),
        per_stratum=2, windows_per_instrument=4, top_k=1,
        coarse_pools=(1, 2), minimum_candidates=1, workers=2,
    )
    first = run_oracle_suite(source, quality, **kwargs)
    second = run_oracle_suite(source, quality, **kwargs)
    assert first.passed and second.passed
    assert first.cases[0].metrics["oracle_digest"] == second.cases[0].metrics["oracle_digest"]
    assert len(first.cases[0].ranking) == first.cases[0].metrics["candidates_scored"]
    report = write_oracle_artifacts(first, tmp_path / "oracle")
    assert report.exists()
    assert "exhaustive sampled oracle" in report.read_text().lower()
