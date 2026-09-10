from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from market_analogues.post_signal_matching import (
    PostSignalMatchingError, bucket_members, control_order_digest, cross_sectional_deciles,
    event_id, select_control_indices,
)
from experiments.m04r import m04r14_t14_12_matches as stage
from experiments.m04r import verify_m04r14_t14_12_matches as verifier


def test_deciles_and_identifiers_have_independent_exact_oracles() -> None:
    values = np.array([1., 2., 2., 4., 5., 6., 7., 8., 9., 10.])
    assert cross_sectional_deciles(values).tolist() == verifier._deciles(values).tolist()
    signal_id = event_id("contract", "up_close_4pct", "XYZ", "2024-01-03")
    assert signal_id == verifier._event_id("contract", "up_close_4pct", "XYZ", "2024-01-03")
    assert control_order_digest("contract", signal_id, "ABC") == verifier._selection_digest("contract", signal_id, "ABC")


def test_signal_specific_at_risk_selection_matches_independent_oracle() -> None:
    names = np.array(["EVENT", "A", "B", "C", "D", "E", "F", "G"], dtype=object)
    deciles = np.array([
        [5, 5, 5, 5], [5, 5, 5, 5], [5, 5, 5, 5], [5, 5, 5, 5],
        [5, 5, 5, 5], [5, 5, 5, 5], [6, 6, 6, 6], [9, 9, 9, 9],
    ], dtype=np.uint8)
    eligible = np.array([False, True, True, True, True, True, False, True])
    observed, tier = select_control_indices(
        symbols=names, members=bucket_members(deciles, eligible),
        all_eligible_indices=np.flatnonzero(eligible), event_symbol="EVENT", event_deciles=deciles[0],
        contract_digest="contract", signal_id="signal",
    )
    expected, expected_tier = verifier._select(names, deciles, eligible, "EVENT", deciles[0], "contract", "signal")
    assert observed.tolist() == expected and tier == expected_tier == "exact_all_four_deciles"


def test_outcome_columns_are_structurally_excluded_from_matching_reader() -> None:
    assert tuple(stage.CAUSAL_COLUMNS) == verifier.VERIFY_COLUMNS
    assert "complete_20" not in stage.CAUSAL_COLUMNS
    assert not any("endpoint" in name or "maximum_" in name or "barrier" in name for name in stage.CAUSAL_COLUMNS)
    assert set(stage.MATCH_COLUMNS).issubset(stage.CAUSAL_COLUMNS)
    assert set(stage.MARKET_COLUMNS).issubset(stage.CAUSAL_COLUMNS)


def _year_fixture() -> pd.DataFrame:
    rows = []
    for index in range(12):
        rows.append({
            "symbol": f"S{index:02d}", "signal_date": pd.Timestamp("2024-01-03"), "signal_position": 300,
            "investable": True, "prior_return_63": index / 100, "prior_volatility_20": .1 + index / 1000,
            "prior_close": 10. + index, "prior_median_dollar_volume_20": 2_000_000. + index,
            "benchmark_signal_day_return": .01, "benchmark_return_20": .02,
            "benchmark_return_63": .03, "benchmark_volatility_20": .04,
            "up_close_at_risk": True, "up_close_4pct": index == 0, "up_close_signal_event": index == 0,
            "bullish_range_expansion_at_risk": True,
            "bullish_range_expansion_4pct": index == 1,
            "bullish_range_expansion_signal_event": index == 1,
        })
    return pd.DataFrame(rows)


def test_year_checkpoint_is_atomic_restartable_and_contains_no_outcome(tmp_path: Path) -> None:
    cache = tmp_path / "cache"; cache.mkdir()
    first = stage._write_year(2024, _year_fixture(), cache, "contract")
    second = stage._write_year(2024, _year_fixture(), cache, "contract")
    assert first == second and first["passed"] and first["outcome_columns_read"] == []
    root = cache / "year-2024"
    events = pd.read_parquet(root / "event-matches.parquet")
    controls = pd.read_parquet(root / "control-identities.parquet")
    assert len(events) == 4 and len(controls) == 20
    assert set(events.control_count) == {5} and controls.groupby(["population", "signal_name"]).size().eq(5).all()
    assert not any("return_20" in name and not name.startswith("benchmark_") for name in events.columns)


def test_year_checkpoint_retains_an_empty_signal_year(tmp_path: Path) -> None:
    frame = _year_fixture()
    for column in ("up_close_4pct", "up_close_signal_event", "bullish_range_expansion_4pct",
                   "bullish_range_expansion_signal_event"):
        frame[column] = False
    cache = tmp_path / "cache"; cache.mkdir()
    seal = stage._write_year(2024, frame, cache, "contract")
    assert seal["event_match_rows"] == 0 and seal["control_identity_rows"] == 0
    assert pd.read_parquet(cache / "year-2024/event-matches.parquet").empty
    assert pd.read_parquet(cache / "year-2024/control-identities.parquet").empty


@pytest.mark.parametrize("function,args", [
    (cross_sectional_deciles, ([1., np.nan],)),
    (bucket_members, (np.ones((2, 3)), [True, True])),
])
def test_matching_kernels_reject_invalid_inputs(function, args) -> None:
    with pytest.raises(PostSignalMatchingError): function(*args)
