from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd

from .adapters import OHLCVSource
from .types import InstrumentKey


@dataclass
class QualityRecord:
    dataset_id: str
    symbol: str
    source_hash: str
    rows: int
    first_timestamp: str | None
    last_timestamp: str | None
    duplicate_timestamps: int
    missing_required: int
    invalid_ohlc: int
    nonpositive_prices: int
    negative_volume: int
    zero_volume: int
    extreme_discontinuities: int
    max_gap_multiple: float
    tier: str
    issues: str


def qualify_frame(key: InstrumentKey, df: pd.DataFrame, source_hash: str) -> QualityRecord:
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing_required = int(df[required].isna().any(axis=1).sum())
    duplicates = int(df["timestamp"].duplicated().sum())
    prices = df[["open", "high", "low", "close"]]
    nonpositive = int((prices <= 0).any(axis=1).sum())
    invalid = int(((df["low"] > df[["open", "close", "high"]].min(axis=1)) |
                   (df["high"] < df[["open", "close", "low"]].max(axis=1))).sum())
    negative_volume = int((df["volume"] < 0).sum())
    zero_volume = int((df["volume"] == 0).sum())
    returns = np.log(df["close"].where(df["close"] > 0)).diff().abs()
    extreme = int((returns > np.log(3)).sum())
    deltas = df["timestamp"].diff().dropna().dt.total_seconds()
    positive = deltas[deltas > 0]
    median_delta = float(positive.median()) if len(positive) else 0.0
    max_gap_multiple = float(positive.max() / median_delta) if median_delta and len(positive) else 0.0
    critical = duplicates + missing_required + invalid + nonpositive + negative_volume
    issues: list[str] = []
    for name, count in [
        ("duplicate_timestamp", duplicates), ("missing_required", missing_required),
        ("invalid_ohlc", invalid), ("nonpositive_price", nonpositive),
        ("negative_volume", negative_volume), ("extreme_discontinuity", extreme),
        ("zero_volume", zero_volume),
    ]:
        if count:
            issues.append(f"{name}:{count}")
    if len(df) < 126 or critical:
        tier = "QUARANTINED"
    elif len(df) < 252 or extreme or zero_volume > max(5, int(len(df) * 0.02)):
        tier = "B"
    else:
        tier = "A"
    return QualityRecord(
        dataset_id=key.dataset_id, symbol=key.source_symbol, source_hash=source_hash,
        rows=len(df), first_timestamp=str(df["timestamp"].iloc[0]) if len(df) else None,
        last_timestamp=str(df["timestamp"].iloc[-1]) if len(df) else None,
        duplicate_timestamps=duplicates, missing_required=missing_required,
        invalid_ohlc=invalid, nonpositive_prices=nonpositive,
        negative_volume=negative_volume, zero_volume=zero_volume,
        extreme_discontinuities=extreme, max_gap_multiple=max_gap_multiple,
        tier=tier, issues=";".join(issues),
    )


def audit_source(source: OHLCVSource, output: Path | None = None) -> pd.DataFrame:
    records = []
    for key in source.instruments():
        try:
            records.append(qualify_frame(key, source.load(key), source.fingerprint(key)))
        except Exception as exc:
            records.append(QualityRecord(
                key.dataset_id, key.source_symbol, "", 0, None, None, 0, 1, 0, 0,
                0, 0, 0, 0.0, "QUARANTINED", f"load_error:{type(exc).__name__}:{exc}",
            ))
    df = pd.DataFrame([asdict(x) for x in records]).sort_values(["dataset_id", "symbol"])
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output, index=False)
    return df

