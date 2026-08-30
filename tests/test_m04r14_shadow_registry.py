from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import m04r14_shadow_registry as shadow
from market_analogues.types import InstrumentKey


def _bars(end: str, rows: int = 320) -> pd.DataFrame:
    timestamps = pd.bdate_range(end=end, periods=rows)
    close = np.linspace(10.0, 20.0, rows)
    return pd.DataFrame({
        "timestamp": timestamps, "open": close - .1, "high": close + .2,
        "low": close - .2, "close": close, "volume": np.full(rows, 1000.0),
    })


class Source:
    def __init__(self, values: dict[str, pd.DataFrame], benchmark: pd.DataFrame):
        self.values = values
        self.benchmark = benchmark

    def instruments(self):
        return [InstrumentKey("nasdaq", symbol) for symbol in sorted(self.values)]

    def load(self, key):
        return self.values[key.source_symbol].copy()

    def load_benchmark(self):
        return self.benchmark.copy()


def test_shadow_denominator_accounts_for_every_symbol_and_temporal_skip():
    contract = json.loads((ROOT / "config/m04r14-shadow-contract.json").read_text())
    values: dict[str, pd.DataFrame] = {}
    quality_rows = []
    liquidity_rows = []
    for tier in ("A", "B"):
        for stratum in ("low", "middle", "high"):
            for index in range(2):
                symbol = f"{tier}{stratum[0].upper()}{index}"
                bars = _bars("2026-03-25")
                values[symbol] = bars
                quality_rows.append({
                    "symbol": symbol, "source_hash": f"hash-{symbol}", "rows": len(bars),
                    "first_timestamp": bars.timestamp.iloc[0],
                    "last_timestamp": bars.timestamp.iloc[-1], "tier": tier,
                })
                liquidity_rows.append({
                    "symbol": symbol, "quality_tier": tier,
                    "liquidity_stratum": stratum, "median_dollar_volume_252": 1_000_000.0,
                })

    extras = {
        "QUAR": ("QUARANTINED", _bars("2026-03-27", 50), True),
        "SHORT": ("A", _bars("2026-03-27", 200), True),
        "STALE": ("A", _bars("2020-01-31"), True),
        "NOLIQ": ("A", _bars("2026-03-27"), False),
        "LATE": ("A", _bars("2026-04-07"), True),
    }
    for symbol, (tier, bars, has_liquidity) in extras.items():
        values[symbol] = bars
        quality_rows.append({
            "symbol": symbol, "source_hash": f"hash-{symbol}", "rows": len(bars),
            "first_timestamp": bars.timestamp.iloc[0],
            "last_timestamp": bars.timestamp.iloc[-1], "tier": tier,
        })
        if has_liquidity and tier != "QUARANTINED":
            liquidity_rows.append({
                "symbol": symbol, "quality_tier": tier,
                "liquidity_stratum": "low", "median_dollar_volume_252": 1.0,
            })
    # Metadata-only and source-only members prove that the denominator is a union.
    quality_rows.append({
        "symbol": "NOSOURCE", "source_hash": "missing", "rows": 320,
        "first_timestamp": pd.Timestamp("2025-01-01"),
        "last_timestamp": pd.Timestamp("2026-03-27"), "tier": "A",
    })
    liquidity_rows.append({
        "symbol": "NOSOURCE", "quality_tier": "A", "liquidity_stratum": "low",
        "median_dollar_volume_252": 1.0,
    })
    values["NOQUALITY"] = _bars("2026-03-27")
    quality = pd.DataFrame(quality_rows)
    liquidity = pd.DataFrame(liquidity_rows)
    benchmark = _bars("2026-05-08", 500)
    source = Source(values, benchmark)
    denominator, cases, sample, latest = shadow._classify(
        ROOT, contract, source, quality, liquidity, pd.Timestamp("2025-12-31"),
        sorted(values),
    )
    assert len(denominator) == len(set(values) | set(quality.symbol) | set(liquidity.symbol))
    assert len(cases) == 12
    assert len(sample) == 12
    assert cases.episode_id.is_unique and denominator.symbol.is_unique
    reasons = dict(zip(denominator.symbol, denominator.skip_reason, strict=True))
    assert reasons["QUAR"] == "quarantined_quality"
    assert reasons["SHORT"] == "insufficient_query_history"
    assert reasons["STALE"] == "stale_at_source_lock"
    assert reasons["NOLIQ"] == "missing_liquidity_metadata"
    assert reasons["LATE"] == "packed_temporal_coverage_unavailable"
    assert reasons["NOSOURCE"] == "missing_source_file"
    assert reasons["NOQUALITY"] == "missing_quality_metadata"
    assert latest == pd.Timestamp("2026-04-07")
    counts = pd.DataFrame(sample).groupby(["quality_tier", "liquidity_stratum"]).size()
    assert len(counts) == 6 and set(counts) == {2}


def test_records_support_nested_prefixes_and_reject_non_finite_json():
    frame = pd.DataFrame([{"symbol": "X", "prefix": {"digest": "a"}, "value": np.nan}])
    assert shadow._records(frame) == [{"symbol": "X", "prefix": {"digest": "a"}, "value": None}]
    try:
        json.dumps({"value": float("nan")}, allow_nan=False)
    except ValueError:
        pass
    else:
        raise AssertionError("strict JSON unexpectedly accepted NaN")
