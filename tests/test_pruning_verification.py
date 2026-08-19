from pathlib import Path

from market_analogues.adapters import DirectorySource
from market_analogues.config import DatasetSpec
from market_analogues.oracle import run_oracle_suite, write_oracle_artifacts
from market_analogues.pruning_verification import (
    verify_exact_safe_pruning, write_pruning_report,
)
from market_analogues.quality import audit_source


def test_pruning_verifier_matches_persisted_exact_oracle(
    directory_dataset: Path, tmp_path: Path,
) -> None:
    source = DirectorySource(DatasetSpec(
        "test", "directory", directory_dataset, "parquet", timestamp_column="date",
    ))
    quality = audit_source(source)
    oracle = run_oracle_suite(
        source, quality, symbols=("AAA",), lookbacks=(126,),
        cutoff_quantiles=(1.0,), per_stratum=2, windows_per_instrument=8,
        top_k=2, coarse_pools=(2, 4), minimum_candidates=1,
    )
    oracle_directory = tmp_path / "oracle"
    write_oracle_artifacts(oracle, oracle_directory)

    result = verify_exact_safe_pruning(source, oracle_directory)

    assert result.passed, result.failures
    assert result.metrics["cases"] == 1
    assert result.metrics["maximum_distance_delta"] < 1e-12
    assert result.metrics["maximum_component_delta"] < 1e-12
    assert result.cases.default_exact_evaluated.iloc[0] <= result.cases.eligible_candidates.iloc[0]
    assert result.cases.bound_exact_evaluated.iloc[0] <= result.cases.default_exact_evaluated.iloc[0]
    report = write_pruning_report(result, tmp_path / "pruning.html")
    assert "exact-safe progressive pruning" in report.read_text().lower()

