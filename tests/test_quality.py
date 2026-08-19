import pandas as pd

from market_analogues.adapters import DirectorySource
from market_analogues.config import DatasetSpec
from market_analogues.quality import audit_source, qualify_frame
from market_analogues.types import InstrumentKey


def test_clean_frame_is_tier_a(bars):
    bars = bars.rename(columns={"date": "timestamp"})
    record = qualify_frame(InstrumentKey("x", "AAA"), bars, "hash")
    assert record.tier == "A"
    assert record.invalid_ohlc == 0


def test_critical_invalid_frame_is_quarantined(bars):
    bars = bars.rename(columns={"date": "timestamp"})
    bars.loc[10, "high"] = bars.loc[10, "low"] - 1
    record = qualify_frame(InstrumentKey("x", "AAA"), bars, "hash")
    assert record.tier == "QUARANTINED"
    assert record.invalid_ohlc == 1


def test_audit_accounting(directory_dataset, tmp_path):
    source = DirectorySource(DatasetSpec("demo", "directory", directory_dataset, "parquet", timestamp_column="date"))
    output = tmp_path / "quality.parquet"
    report = audit_source(source, output)
    assert len(report) == len(source.instruments())
    assert output.exists()
    assert set(report.tier) == {"A"}

