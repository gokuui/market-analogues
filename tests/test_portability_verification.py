from pathlib import Path

from market_analogues.portability_verification import (
    run_portability_verification, write_portability_verification,
)


def test_cross_adapter_end_to_end_verifier(tmp_path: Path) -> None:
    root = tmp_path / "portable"
    result = run_portability_verification(root)
    assert result.passed, result.failures
    assert result.variants == (
        "directory-parquet", "directory-csv",
        "long-table-parquet", "long-table-csv",
    )
    assert result.maximum_distance_delta <= 1e-12
    assert result.retrieval_semantics_equal
    assert result.same_format_identity_equal
    assert result.evidence_rows_equivalent
    assert result.evidence_summary_equivalent
    assert result.maximum_evidence_numeric_delta <= 1e-12
    assert result.source_files_unchanged
    assert result.future_mutation_retrieval_invariant
    assert result.future_mutation_outcomes_changed
    machine, html = write_portability_verification(result, root)
    assert machine.exists()
    assert "PASS" in html.read_text()
