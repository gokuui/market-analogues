from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from market_analogues.adapters import DirectorySource
from market_analogues.config import BenchmarkSpec, DatasetSpec
from market_analogues.m04r_validation_registry import (
    EXPECTED_CASES, build_m04r_validation_registry,
    validate_m04r_validation_registry, write_m04r_validation_registry,
)
from market_analogues.types import stable_hash


def _fixture(tmp_path: Path) -> tuple[DirectorySource, pd.DataFrame, pd.DataFrame, Path]:
    dates = pd.date_range("2010-01-04", periods=4_200, freq="B")
    position = np.arange(len(dates), dtype=float)
    benchmark_close = 100 * np.exp(.00035 * position + .42 * np.sin(position / 170))
    benchmark_frame = pd.DataFrame({
        "date": dates, "open": benchmark_close, "high": benchmark_close * 1.01,
        "low": benchmark_close * .99, "close": benchmark_close,
        "volume": np.full(len(dates), 10_000_000),
    })
    benchmark = tmp_path / "benchmark.parquet"
    benchmark_frame.to_parquet(benchmark, index=False)
    data = tmp_path / "bars"
    data.mkdir()
    metadata = []
    counter = 0
    for tier in ("A", "B"):
        for stratum in ("high", "low", "middle"):
            for index in range(5):
                symbol = f"{tier}{stratum[0].upper()}{index}"
                mode = counter % 3
                slope = (.0017, -.0013, .0)[mode]
                oscillation = (.03, .50, .02)[mode] * np.sin(position / (13 + mode))
                close = 20 * np.exp(slope * position + oscillation)
                volume = np.full(len(dates), 100_000 + counter * 1_000, dtype=float)
                if tier == "B":
                    volume[::5] = 0
                frame = pd.DataFrame({
                    "date": dates, "open": close * .997, "high": close * 1.01,
                    "low": close * .99, "close": close, "volume": volume,
                })
                frame.to_parquet(data / f"{symbol}.parquet", index=False)
                metadata.append((symbol, tier, stratum))
                counter += 1
    source = DirectorySource(DatasetSpec(
        "nasdaq", "directory", data, "parquet", timestamp_column="date",
        benchmark=BenchmarkSpec(benchmark),
    ))
    selected_quality = pd.DataFrame({
        "symbol": [row[0] for row in metadata],
        "first_timestamp": [dates[0]] * len(metadata),
        "last_timestamp": [dates[-1]] * len(metadata),
    })
    phantom = pd.DataFrame({
        "symbol": [f"PHANTOM{index}" for index in range(8_001)],
        "first_timestamp": [pd.Timestamp("2020-01-02")] * 8_001,
        "last_timestamp": [dates[-1]] * 8_001,
    })
    quality = pd.concat([selected_quality, phantom], ignore_index=True)
    liquidity = pd.DataFrame([{
        "symbol": symbol, "quality_tier": tier, "liquidity_stratum": stratum,
        "rows": len(dates), "median_dollar_volume_252": float(counter + 1) * 1_000,
    } for counter, (symbol, tier, stratum) in enumerate(metadata)])
    return source, quality, liquidity, data


def _ledger() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "m04r-contamination-ledger-v1",
        "normalization": "strip then uppercase; exact source-symbol exclusion",
        "sources": [], "excluded_symbols": [], "excluded_symbol_count": 0,
        "excluded_symbol_digest": stable_hash([]),
        "known_identity_alias_groups": [], "alias_limit": "fixture",
    }
    payload["ledger_digest"] = stable_hash(payload)
    return payload


def test_registry_is_deterministic_balanced_and_causally_locked(tmp_path: Path) -> None:
    source, quality, liquidity, data = _fixture(tmp_path)
    first = build_m04r_validation_registry(
        source, quality, liquidity, _ledger(), {"real_forward_outcomes_accessed": False},
    )
    second = build_m04r_validation_registry(
        source, quality.sample(frac=1, random_state=7),
        liquidity.sample(frac=1, random_state=11), _ledger(),
        {"real_forward_outcomes_accessed": False},
    )
    assert first.passed and second.passed
    assert len(first.cases) == EXPECTED_CASES
    assert first.metrics["registry_digest"] == second.metrics["registry_digest"]
    assert set(first.cases.benchmark_regime) == {"up", "down", "sideways"}
    assert set(first.cases.morphology_stratum) == {"advance", "decline", "range"}
    assert set(first.cases.data_context) == {"full", "sparse"}
    directory = tmp_path / "registry"
    write_m04r_validation_registry(first, directory)
    assert validate_m04r_validation_registry(source, directory) == ()

    selected = str(first.cases.symbol.iloc[0])
    path = data / f"{selected}.parquet"
    original = pd.read_parquet(path)
    appended = pd.concat([
        original,
        original.tail(1).assign(date=original.date.iloc[-1] + pd.Timedelta(days=1)),
    ], ignore_index=True)
    appended.to_parquet(path, index=False)
    appended_source = DirectorySource(source.spec)
    assert validate_m04r_validation_registry(appended_source, directory) == ()

    revised = appended.copy()
    revised.loc[10, "close"] *= 1.01
    revised.to_parquet(path, index=False)
    revised_source = DirectorySource(source.spec)
    assert any(
        "stock causal prefix mismatch" in value
        for value in validate_m04r_validation_registry(revised_source, directory)
    )


def test_registry_rejects_transcript_and_outcome_tampering(tmp_path: Path) -> None:
    source, quality, liquidity, _ = _fixture(tmp_path)
    result = build_m04r_validation_registry(
        source, quality, liquidity, _ledger(), {"real_forward_outcomes_accessed": False},
    )
    directory = tmp_path / "registry"
    json_path, _ = write_m04r_validation_registry(result, directory)
    transcript_path = directory / "selection-transcript.parquet"
    transcript = pd.read_parquet(transcript_path)
    transcript.loc[0, "candidate_rank"] = 99
    transcript.to_parquet(transcript_path, index=False)
    failures = validate_m04r_validation_registry(source, directory)
    assert "selection transcript digest mismatch" in failures
    assert any("selection ranks are not contiguous" in value for value in failures)

    payload = json.loads(json_path.read_text())
    payload["cases_data"][0]["forward_return"] = 1.0
    json_path.write_text(json.dumps(payload))
    failures = validate_m04r_validation_registry(source, directory)
    assert "registry digest mismatch" in failures
    assert "registry contains a forbidden outcome/setup field" in failures
