from pathlib import Path

import json
import pandas as pd

from market_analogues.adapters import DirectorySource
from market_analogues.config import BenchmarkSpec, DatasetSpec
from market_analogues.m04r_batch_registry import (
    build_m04r_batch_registry, validate_m04r_batch_registry,
    write_m04r_batch_registry,
)


def test_m04r_batch_registry_freezes_24_independent_real_queries(
    bars: pd.DataFrame, tmp_path: Path,
) -> None:
    data = tmp_path / "bars"
    data.mkdir()
    selected = [
        (f"{tier}{stratum[0].upper()}{index}", tier, stratum)
        for tier in ("A", "B")
        for stratum in ("high", "low", "middle")
        for index in (1, 2)
    ]
    symbols = ["Q", "C", *[row[0] for row in selected]]
    for position, symbol in enumerate(symbols, 1):
        frame = bars.copy()
        frame["close"] *= 1 + position / 100
        frame.to_parquet(data / f"{symbol}.parquet", index=False)
    benchmark = tmp_path / "benchmark.parquet"
    bars.to_parquet(benchmark, index=False)
    source = DirectorySource(DatasetSpec(
        "test", "directory", data, "parquet", timestamp_column="date",
        benchmark=BenchmarkSpec(benchmark),
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
            "median_dollar_volume_252": float(position + 1) * 1000,
        }
        for position, (symbol, tier, stratum) in enumerate(selected)
    ])
    liquidity.to_parquet(oracle / "liquidity-strata.parquet", index=False)
    quality = pd.DataFrame({
        "symbol": symbols,
        "last_timestamp": [bars.date.iloc[-1]] * len(symbols),
        "tier": ["A", "A", *[row[1] for row in selected]],
    })

    first = build_m04r_batch_registry(
        source, oracle, quality, lookback=63, minimum_rows=250,
        minimum_future_sessions=60,
    )
    liquidity.sample(frac=1, random_state=19).to_parquet(
        oracle / "liquidity-strata.parquet", index=False,
    )
    second = build_m04r_batch_registry(
        source, oracle, quality, lookback=63, minimum_rows=250,
        minimum_future_sessions=60,
    )

    assert first.passed and second.passed
    assert first.metrics["registry_digest"] == second.metrics["registry_digest"]
    assert len(first.cases) == 24
    assert first.cases.symbol.nunique() == 12
    assert set(first.cases.cutoff_role) == {"historical", "current"}
    assert set(first.cases.symbol).isdisjoint({"Q", "C"})
    json_path, parquet_path, html_path = write_m04r_batch_registry(
        first, tmp_path / "registry",
    )
    assert parquet_path.exists() and html_path.exists()
    assert validate_m04r_batch_registry(source, json_path, quality, oracle) == ()

    tampered = json.loads(json_path.read_text())
    tampered["cases_data"][0]["cutoff"] = "1999-01-01T00:00:00"
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered))
    failures = validate_m04r_batch_registry(source, tampered_path, quality, oracle)
    assert "registry digest mismatch" in failures
    assert any("episode identity is stale" in value for value in failures)
    assert "registry deterministic selection differs" in failures
