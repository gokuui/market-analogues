from pathlib import Path

import pandas as pd
import yaml

from market_analogues.adapters import DirectorySource
from market_analogues.config import BenchmarkSpec, DatasetSpec
from market_analogues.gate12_registry import (
    build_gate12_registry, validate_gate12_registry, write_gate12_registry,
)


def test_gate12_registry_is_independent_deterministic_and_fingerprint_locked(
    bars: pd.DataFrame, tmp_path: Path,
) -> None:
    data = tmp_path / "bars"
    data.mkdir()
    selected = [
        ("AH", "A", "high"), ("AL", "A", "low"),
        ("AM", "A", "middle"), ("BH", "B", "high"),
        ("BL", "B", "low"), ("BM", "B", "middle"),
    ]
    for position, symbol in enumerate(["Q", "C", *[row[0] for row in selected]], 1):
        frame = bars.copy()
        frame["close"] *= 1 + position / 100
        frame.to_parquet(data / f"{symbol}.parquet", index=False)
    benchmark_path = tmp_path / "benchmark.parquet"
    bars.to_parquet(benchmark_path, index=False)
    source = DirectorySource(DatasetSpec(
        "test", "directory", data, "parquet", timestamp_column="date",
        benchmark=BenchmarkSpec(benchmark_path),
    ))
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    pd.DataFrame([{"case_id": "sample", "query": "test:Q"}]).to_parquet(
        oracle / "oracle-summary.parquet", index=False,
    )
    pd.DataFrame([{"symbol": "C"}]).to_parquet(
        oracle / "sample.parquet", index=False,
    )
    liquidity = pd.DataFrame([
        {
            "symbol": symbol, "quality_tier": tier,
            "liquidity_stratum": stratum, "rows": len(bars),
            "median_dollar_volume_252": float(index + 1) * 1000,
        }
        for index, (symbol, tier, stratum) in enumerate(selected)
    ])
    liquidity.to_parquet(oracle / "liquidity-strata.parquet", index=False)
    quality = pd.DataFrame({
        "symbol": ["Q", "C", *[row[0] for row in selected]],
        "last_timestamp": [bars.date.iloc[-1]] * 8,
        "tier": ["A", "A", *[row[1] for row in selected]],
    })

    first = build_gate12_registry(
        source, oracle, lookback=63, minimum_rows=250,
        minimum_future_sessions=60, quality=quality,
    )
    liquidity.sample(frac=1, random_state=7).to_parquet(
        oracle / "liquidity-strata.parquet", index=False,
    )
    second = build_gate12_registry(
        source, oracle, lookback=63, minimum_rows=250,
        minimum_future_sessions=60, quality=quality,
    )

    assert first.passed and second.passed
    assert first.metrics["registry_digest"] == second.metrics["registry_digest"]
    assert first.metrics["selected_symbols"] == sorted(row[0] for row in selected)
    assert len(first.cases) == 12
    assert set(first.cases.symbol).isdisjoint({"Q", "C"})
    assert first.cases[first.cases.cutoff_role == "historical"].future_sessions_at_lock.min() >= 60
    yaml_path, parquet_path, html_path = write_gate12_registry(first, tmp_path / "registry")
    assert parquet_path.exists() and html_path.exists()
    assert validate_gate12_registry(source, yaml_path, quality) == ()
    tampered_path = tmp_path / "tampered.yaml"
    tampered = yaml.safe_load(yaml_path.read_text())
    tampered["cases_data"][0]["cutoff"] = "1999-01-01T00:00:00"
    tampered_path.write_text(yaml.safe_dump(tampered, sort_keys=False))
    assert "registry digest mismatch" in validate_gate12_registry(
        source, tampered_path, quality,
    )
    changed_quality = quality.copy()
    changed_quality.loc[changed_quality.symbol == "AH", "tier"] = "B"
    assert "quality tier is stale for test:AH" in ";".join(
        validate_gate12_registry(source, yaml_path, changed_quality)
    )

    changed_benchmark = pd.read_parquet(benchmark_path)
    changed_benchmark.loc[0, "close"] *= 1.01
    changed_benchmark.to_parquet(benchmark_path, index=False)
    assert "benchmark fingerprint is stale" in ";".join(
        validate_gate12_registry(source, yaml_path, quality)
    )

    changed = pd.read_parquet(data / "AH.parquet")
    changed.loc[0, "close"] *= 1.01
    changed.to_parquet(data / "AH.parquet", index=False)
    assert "stale for test:AH" in ";".join(
        validate_gate12_registry(source, yaml_path, quality)
    )
