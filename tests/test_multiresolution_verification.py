from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from market_analogues.adapters import DirectorySource
from market_analogues.cli import main
from market_analogues.config import DatasetSpec
from market_analogues.gates import GateReport
from market_analogues.multiresolution_verification import (
    verify_multiresolution_state,
    write_multiresolution_verification,
)
from market_analogues.quality import audit_source


def _inputs(directory_dataset: Path):
    spec = DatasetSpec(
        "demo", "directory", directory_dataset, "parquet", timestamp_column="date",
    )
    source = DirectorySource(spec)
    quality = audit_source(source)
    registry = pd.DataFrame([
        {
            "case_id": f"demo-{symbol}-current-252",
            "symbol": symbol,
            "cutoff": str(source.load(key).timestamp.iloc[-1]),
            "representation_version": "dense-v1",
        }
        for key in source.instruments()
        for symbol in [key.source_symbol]
    ])
    return spec, source, quality, registry


def test_multiresolution_verifier_covers_synthetic_and_real_cases(
    directory_dataset: Path, tmp_path: Path,
) -> None:
    _, source, quality, registry = _inputs(directory_dataset)
    result = verify_multiresolution_state(
        [("demo", source, quality, registry)],
        maximum_total_seconds=20,
        maximum_case_seconds=3,
        maximum_rss_mb=16384,
    )
    assert result.passed
    assert result.metrics["required_horizons"] == [252, 126, 63, 21, 10, 5]
    assert result.metrics["field_contract_fields"] == 42
    assert result.metrics["future_mutation_digest_equal"] is True
    assert result.metrics["future_mutation_max_delta"] == 0
    assert result.metrics["price_volume_unit_max_delta"] <= 1e-6
    assert result.metrics["missing_context_mask_errors"] == 0
    assert result.metrics["real_cases_passed"] == 2

    machine, html, fields = write_multiresolution_verification(result, tmp_path / "report")
    payload = json.loads(machine.read_text())
    assert payload["passed"] is True
    assert len(payload["real_cases"]) == 2
    assert "six-resolution point-in-time state" in html.read_text()
    assert len(json.loads(fields.read_text())) == 42


def test_multiresolution_cli_is_gated_and_writes_artifacts(
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
    _, _, quality, registry = _inputs(directory_dataset)
    quality_path = artifacts / "quality" / "demo.parquet"
    quality_path.parent.mkdir(parents=True)
    quality.to_parquet(quality_path, index=False)
    registry_path = artifacts / "gate12" / "demo" / "query-registry.parquet"
    registry_path.parent.mkdir(parents=True)
    registry.to_parquet(registry_path, index=False)
    GateReport("m01_point_in_time_data_ledger", True).write(artifacts / "gates")
    assert main([
        "verify-multiresolution-state", "--config", str(config),
        "--datasets", "demo", "--maximum-total-seconds", "20",
        "--maximum-case-seconds", "3", "--maximum-rss-mb", "16384",
    ]) == 0
    gate = json.loads(
        (artifacts / "gates" / "m02_multiresolution_chart_state.json").read_text()
    )
    assert gate["passed"] is True
    assert gate["metrics"]["real_cases_passed"] == 2
    assert (artifacts / "m02-multiresolution" / "m02-multiresolution-verification.html").exists()
