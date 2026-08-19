from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_analogues.adapters import DirectorySource
from market_analogues.config import DatasetSpec
from market_analogues.exact_storage_feasibility import (
    _deserialize_layout,
    _pack,
    _serialize_layout,
    _unpack,
    verify_exact_storage_feasibility,
    write_exact_storage_report,
)
from market_analogues.episodes import build_episode
from market_analogues.representation import represent
from market_analogues.types import InstrumentKey


def test_exact_storage_feasibility_preserves_native_and_selects_streaming(
    bars: pd.DataFrame, tmp_path: Path,
) -> None:
    data = tmp_path / "bars"
    data.mkdir()
    bars.to_parquet(data / "AAA.parquet", index=False)
    changed = bars.copy()
    scale = np.linspace(.8, 1.35, len(changed))
    for column in ("open", "high", "low", "close"):
        changed[column] *= scale
    changed["volume"] = changed["volume"].iloc[::-1].to_numpy()
    changed.to_parquet(data / "BBB.parquet", index=False)
    source = DirectorySource(DatasetSpec(
        "test", "directory", data, "parquet", timestamp_column="date",
    ))
    registry = pd.DataFrame([
        {
            "symbol": symbol, "lookback": 63,
            "representation_version": "dense-v1", "quality_tier": "A",
        }
        for symbol in ("AAA", "BBB")
    ])
    result = verify_exact_storage_feasibility(
        {"test": source}, {"test": registry}, total_universe_rows=10_000,
        sample_windows_per_symbol=2, comparison_queries=1,
        comparison_candidates=2, disk_reserve_bytes=0,
        available_bytes=1,
    )

    assert result.passed
    assert result.metrics["selected_design"] == "two_pass_streaming"
    assert not result.metrics["full_exact_store_feasible"]
    native = result.layouts.set_index("layout").loc["native"]
    assert native.fidelity_passed
    assert native.maximum_field_delta == 0
    assert native.maximum_lower_bound_delta <= 1e-12
    assert native.maximum_total_delta <= 1e-12
    assert result.layouts.set_index("layout").loc["float32"].maximum_field_delta > 0
    report = write_exact_storage_report(result, tmp_path / "feasibility.html")
    assert "two_pass_streaming" in report.read_text()


def test_exact_layout_round_trip_handles_missing_constant_and_extreme_values(
    directory_dataset: Path, bars: pd.DataFrame, monkeypatch,
) -> None:
    source = DirectorySource(DatasetSpec(
        "test", "directory", directory_dataset, "parquet", timestamp_column="date",
    ))
    episode = build_episode(
        source, InstrumentKey("test", "AAA"), bars.date.iloc[-1], 63, "dense-v1",
    )
    ordinary = represent(episode)
    altered_48 = dict(ordinary.samples_48)
    altered_64 = dict(ordinary.samples_64)
    altered_48[sorted(altered_48)[0]] = None
    altered_48[sorted(altered_48)[1]] = np.zeros(48)
    altered_64[sorted(altered_64)[0]] = np.full(64, np.finfo(np.float64).max / 1e300)
    edge = replace(
        ordinary, samples_48=altered_48, samples_64=altered_64,
        stage=np.linspace(-1e6, 1e6, 48),
        structural=np.zeros(9),
    )
    coarse, values, masks, names_48, names_64 = _pack([ordinary, edge], "native")
    payload = _serialize_layout(
        coarse, values, masks, names_48, names_64, "native",
    )
    loaded = _deserialize_layout(payload)
    restored = _unpack(*loaded[:5])

    assert np.array_equal(restored[1].coarse, edge.coarse)
    assert np.array_equal(restored[1].stage, edge.stage)
    assert np.array_equal(restored[1].structural, edge.structural)
    assert restored[1].samples_48[sorted(altered_48)[0]] is None
    assert np.array_equal(
        restored[1].samples_64[sorted(altered_64)[0]],
        altered_64[sorted(altered_64)[0]],
    )
    with pytest.raises(ValueError):
        _deserialize_layout(payload[:len(payload) // 2])
    monkeypatch.setattr(
        "market_analogues.exact_storage_feasibility.LAYOUT_VERSION",
        "exact-representation-layout-future",
    )
    with pytest.raises(ValueError, match="unsupported exact layout version"):
        _deserialize_layout(payload)


@pytest.mark.parametrize(
    "keyword,value",
    [
        ("sample_windows_per_symbol", 0),
        ("comparison_queries", 0),
        ("comparison_candidates", 0),
        ("tolerance", -1),
        ("disk_reserve_bytes", -1),
        ("frontier_bytes_per_row", -1),
        ("compression_safety_factor", .99),
        ("available_bytes", -1),
    ],
)
def test_exact_storage_feasibility_rejects_invalid_controls(
    keyword: str, value: float,
) -> None:
    with pytest.raises(ValueError):
        verify_exact_storage_feasibility(
            {}, {}, total_universe_rows=1, **{keyword: value},
        )
