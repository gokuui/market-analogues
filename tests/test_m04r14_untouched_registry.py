from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from market_analogues.adapters import DirectorySource
from market_analogues.config import BenchmarkSpec, DatasetSpec
from market_analogues.m04r_validation_registry import build_m04r_validation_registry
from market_analogues.types import stable_hash
from experiments.m04r import m04r14_untouched_registry as registry


SCHEDULE = (("down", "2010s"), ("sideways", "2010s"), ("up", "2010s"),
            ("down", "2020s"), ("sideways", "2020s"), ("up", "2020s"))


def test_core_supports_six_slot_registry_without_changing_defaults(tmp_path: Path) -> None:
    dates = pd.date_range("2010-01-04", periods=4_200, freq="B")
    position = np.arange(len(dates), dtype=float)
    benchmark_close = 100 * np.exp(.00035 * position + .42 * np.sin(position / 170))
    benchmark_path = tmp_path / "benchmark.parquet"
    pd.DataFrame({"date": dates, "open": benchmark_close, "high": benchmark_close * 1.01,
        "low": benchmark_close * .99, "close": benchmark_close,
        "volume": np.full(len(dates), 10_000_000)}).to_parquet(benchmark_path, index=False)
    bars_root = tmp_path / "bars"; bars_root.mkdir()
    metadata = []
    for cell, (tier, stratum) in enumerate((
        (tier, stratum) for tier in ("A", "B") for stratum in ("high", "low", "middle"))):
        for index in range(6):
            symbol = f"{tier}{stratum[0]}{index}"
            mode = index % 3
            slope = (.0017, -.0013, 0.0)[mode]
            oscillation = (.03, .50, .02)[mode] * np.sin(position / (13 + mode))
            close = 20 * np.exp(slope * position + oscillation)
            volume = np.full(len(dates), 100_000 + cell * 10_000 + index)
            if tier == "B": volume[::5] = 0
            pd.DataFrame({"date": dates, "open": close * .99, "high": close * 1.01,
                "low": close * .98, "close": close, "volume": volume}).to_parquet(
                    bars_root / f"{symbol}.parquet", index=False)
            metadata.append((symbol, tier, stratum))
    source = DirectorySource(DatasetSpec("nasdaq", "directory", bars_root, "parquet",
        timestamp_column="date", benchmark=BenchmarkSpec(benchmark_path)))
    selected_quality = pd.DataFrame({"symbol": [row[0] for row in metadata],
        "first_timestamp": [dates[0]] * 36, "last_timestamp": [dates[-1]] * 36})
    phantom = pd.DataFrame({"symbol": [f"PHANTOM{index}" for index in range(8_001)],
        "first_timestamp": [pd.Timestamp("2020-01-02")] * 8_001,
        "last_timestamp": [dates[-1]] * 8_001})
    quality = pd.concat([selected_quality, phantom], ignore_index=True)
    liquidity = pd.DataFrame([{"symbol": symbol, "quality_tier": tier,
        "liquidity_stratum": stratum, "rows": len(dates),
        "median_dollar_volume_252": float(index + 1)}
        for index, (symbol, tier, stratum) in enumerate(metadata)])
    ledger = {"excluded_symbols": [], "excluded_symbol_count": 0,
        "ledger_digest": stable_hash([])}
    result = build_m04r_validation_registry(source, quality, liquidity, ledger,
        {"real_forward_outcomes_accessed": False}, seed="fixture-six",
        target_schedule=SCHEDULE, symbols_per_cell=6,
        schema_version="fixture-six-v1")
    assert result.passed
    assert len(result.cases) == 72 and result.cases.symbol.nunique() == 36
    assert result.metrics["symbols_per_cell"] == 6
    assert result.metrics["target_schedule"] == [list(value) for value in SCHEDULE]


def test_contract_is_frozen_six_by_six() -> None:
    contract = json.loads((ROOT / "config/m04r14-untouched-registry-contract.json").read_text())
    assert contract["expected_symbols"] == 36 and contract["expected_cases"] == 72
    assert len(contract["required_cells"]) == 6 and len(contract["target_schedule"]) == 6
    assert contract["outcome_firewall"]["real_forward_outcomes_accessed"] is False


def test_expanded_ledger_adds_all_previous_selected_symbols(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    design = tmp_path / "design.yaml"; design.write_text("symbols: []\n")
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps({"registry_digest": "previous-digest",
        "cases_data": [{"symbol": f"OLD{index // 2}"} for index in range(60)]}))
    base = {"sources": [{"name": "base", "path": str(design),
        "file_sha256": "x", "symbol_count": 1, "symbol_digest": "y",
        "symbols": ["BASE"]}], "known_identity_alias_groups": [],
        "alias_limit": "fixture", "ledger_digest": "base-digest"}
    monkeypatch.setattr(registry, "build_contamination_ledger", lambda *_: base)
    value = registry.expanded_ledger(tmp_path, design, previous)
    assert value["previous_registry_cases"] == 60
    assert value["previous_registry_symbols"] == 30
    assert set(f"OLD{index}" for index in range(30)).issubset(value["excluded_symbols"])
    assert value["ledger_digest"] == stable_hash({
        key: item for key, item in value.items() if key != "ledger_digest"})


def test_publish_is_atomic_create_only_and_sealed(tmp_path: Path) -> None:
    frame = pd.DataFrame([{"case_id": "case", "quality_tier": "A",
        "liquidity_stratum": "low", "selection_target_regime": "up",
        "selection_target_era": "2020s", "cutoff_role": "current",
        "cutoff": "2026-01-01", "symbol": "NEW"}])
    result = SimpleNamespace(universe=frame[["symbol"]], transcript=frame[["symbol"]],
        cases=frame, metrics={"schema_version": registry.SCHEMA}, passed=True,
        failures=(), contamination_ledger={"ledger_digest": "ledger"})
    output = tmp_path / "registry"
    sealed = registry._publish(result, output, {"source_lock_digest": "lock"})
    assert sealed["passed"] is True
    assert {path.name for path in output.iterdir()} == {
        "SEALED.json", "query-registry.html", "query-registry.json",
        "query-registry.parquet", "selection-transcript.parquet",
        "selection-universe.parquet",
    }
    with pytest.raises(registry.RegistryError, match="exists"):
        registry._publish(result, output, {"source_lock_digest": "lock"})
