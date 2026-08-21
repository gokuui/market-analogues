from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from market_analogues.adapters import DirectorySource
from market_analogues.cli import main
from market_analogues.config import BenchmarkSpec, DatasetSpec
from market_analogues.data_ledger import (
    DataLedgerError,
    REQUIRED_CAPABILITIES,
    build_data_ledger,
    load_availability_declaration,
    write_data_ledger_artifacts,
)
from market_analogues.gates import GateReport
from market_analogues.quality import audit_source


def _availability(tmp_path: Path, dataset_id: str, *, benchmark: str = "unavailable") -> Path:
    capabilities = {
        name: {
            "status": (
                "available" if name == "source_ohlcv"
                else benchmark if name == "benchmark_ohlcv"
                else "unavailable"
            ),
            "reason": f"declared test state for {name}",
        }
        for name in REQUIRED_CAPABILITIES
    }
    path = tmp_path / "availability.yaml"
    path.write_text(yaml.safe_dump({
        "schema_version": "data-availability-v1",
        "universe_boundary": "configured_source_instruments_only",
        "datasets": {dataset_id: {"capabilities": capabilities}},
    }))
    return path


def _source(path: Path, *, benchmark: Path | None = None) -> tuple[DatasetSpec, DirectorySource]:
    spec = DatasetSpec(
        "demo", "directory", path, "parquet", timestamp_column="date",
        benchmark=BenchmarkSpec(benchmark) if benchmark else None,
    )
    return spec, DirectorySource(spec)


def test_availability_declaration_requires_every_capability(tmp_path: Path) -> None:
    path = _availability(tmp_path, "demo")
    payload = yaml.safe_load(path.read_text())
    payload["datasets"]["demo"]["capabilities"].pop("delisting_returns")
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(DataLedgerError, match="capability keys differ"):
        load_availability_declaration(path)


def test_data_ledger_rejects_capability_contradictions(
    directory_dataset: Path, tmp_path: Path,
) -> None:
    spec, source = _source(directory_dataset)
    quality = audit_source(source)
    path = _availability(tmp_path, "demo")
    payload = yaml.safe_load(path.read_text())
    payload["datasets"]["demo"]["capabilities"]["source_ohlcv"]["status"] = "unknown"
    path.write_text(yaml.safe_dump(payload))
    result = build_data_ledger(
        spec, source, quality, load_availability_declaration(path),
    )
    assert not result.passed
    assert "configured source OHLCV is not declared available" in result.failures


def test_data_ledger_accounts_every_source_and_has_stable_digest(
    directory_dataset: Path, tmp_path: Path,
) -> None:
    spec, source = _source(directory_dataset)
    quality = audit_source(source)
    declaration = load_availability_declaration(_availability(tmp_path, "demo"))
    as_of = pd.to_datetime(quality.last_timestamp).max()
    first = build_data_ledger(spec, source, quality, declaration, workers=1, as_of=as_of)
    second = build_data_ledger(spec, source, quality, declaration, workers=2, as_of=as_of)
    assert first.passed and second.passed
    assert first.metrics["source_instruments"] == 2
    assert first.metrics["accounted_instruments"] == 2
    assert first.metrics["usable_instruments"] == 2
    assert first.metrics["fingerprints_verified"] == 2
    assert first.metrics["freshness_status"] == "current_within_7_days"
    assert first.metrics["current_usable_instruments_within_7_days"] == 2
    assert first.metrics["current_coverage_fraction_of_usable"] == 1
    assert first.metrics["current_after_close_analysis_available"] is True
    assert first.metrics["instrument_ledger_digest"] == second.metrics["instrument_ledger_digest"]

    machine1, html1, paths1 = write_data_ledger_artifacts([first], declaration, tmp_path / "one")
    machine2, _, _ = write_data_ledger_artifacts([second], declaration, tmp_path / "two")
    payload1 = json.loads(machine1.read_text())
    payload2 = json.loads(machine2.read_text())
    assert payload1["ledger_digest"] == payload2["ledger_digest"]
    assert paths1[0].exists()
    assert "configured_source_instruments_only" in html1.read_text()


def test_data_ledger_detects_source_mutation_after_quality_audit(
    directory_dataset: Path, bars: pd.DataFrame, tmp_path: Path,
) -> None:
    spec, source = _source(directory_dataset)
    quality = audit_source(source)
    changed = bars.copy()
    changed.loc[20, "close"] *= 1.01
    changed.loc[20, "high"] = max(changed.loc[20, "high"], changed.loc[20, "close"])
    changed.to_parquet(directory_dataset / "AAA.parquet", index=False)
    mutated_source = DirectorySource(spec)
    declaration = load_availability_declaration(_availability(tmp_path, "demo"))
    result = build_data_ledger(spec, mutated_source, quality, declaration)
    assert not result.passed
    assert result.metrics["fingerprint_mismatches"] == 1
    assert "source fingerprints changed" in result.failures[0]


def test_load_error_is_explicitly_quarantined_and_accounted(tmp_path: Path) -> None:
    root = tmp_path / "bars"
    root.mkdir()
    pd.DataFrame({"date": ["2020-01-01"], "close": [1.0]}).to_parquet(
        root / "BROKEN.parquet", index=False,
    )
    spec, source = _source(root)
    quality = audit_source(source)
    declaration = load_availability_declaration(_availability(tmp_path, "demo"))
    result = build_data_ledger(spec, source, quality, declaration)
    assert result.passed
    assert result.metrics["accounted_instruments"] == 1
    assert result.metrics["quarantined_load_errors"] == 1
    assert result.metrics["fingerprints_unverified_load_error"] == 1
    assert result.instruments.iloc[0].accounting_status == "quarantined_load_error"


def test_missing_quality_row_and_missing_declared_benchmark_fail(
    directory_dataset: Path, tmp_path: Path,
) -> None:
    spec, source = _source(directory_dataset)
    quality = audit_source(source).iloc[:1].copy()
    declaration = load_availability_declaration(
        _availability(tmp_path, "demo", benchmark="available"),
    )
    result = build_data_ledger(spec, source, quality, declaration)
    assert not result.passed
    assert result.metrics["missing_quality_records"] == 1
    assert any("benchmark declared available" in failure for failure in result.failures)


def test_configured_benchmark_is_independently_audited(
    directory_dataset: Path, bars: pd.DataFrame, tmp_path: Path,
) -> None:
    benchmark = tmp_path / "benchmark.parquet"
    bars.to_parquet(benchmark, index=False)
    spec, source = _source(directory_dataset, benchmark=benchmark)
    quality = audit_source(source)
    declaration = load_availability_declaration(
        _availability(tmp_path, "demo", benchmark="available"),
    )
    result = build_data_ledger(spec, source, quality, declaration)
    assert result.passed
    assert result.metrics["benchmark"]["tier"] == "A"
    assert len(result.metrics["benchmark"]["source_hash"]) == 64


def test_data_ledger_cli_requires_m00_and_writes_gate(
    directory_dataset: Path, tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    config = tmp_path / "datasets.yaml"
    config.write_text(yaml.safe_dump({
        "artifact_dir": str(artifacts),
        "datasets": {
            "demo": {
                "adapter": "directory", "path": str(directory_dataset),
                "format": "parquet", "timestamp_column": "date",
            },
        },
    }))
    quality_dir = artifacts / "quality"
    quality_dir.mkdir(parents=True)
    spec, source = _source(directory_dataset)
    audit_source(source, quality_dir / "demo.parquet")
    GateReport("m00_case_memory_contract", True).write(artifacts / "gates")
    availability = _availability(tmp_path, "demo")
    assert main([
        "build-data-ledger", "--config", str(config),
        "--availability", str(availability), "--datasets", "demo",
        "--workers", "2",
    ]) == 0
    gate = json.loads((artifacts / "gates" / "m01_point_in_time_data_ledger.json").read_text())
    assert gate["passed"] is True
    assert gate["metrics"]["datasets"]["demo"]["accounted_instruments"] == 2
    assert (artifacts / "data-ledger" / "data-ledger.html").exists()
