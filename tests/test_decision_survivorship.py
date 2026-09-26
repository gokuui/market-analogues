from __future__ import annotations

import pandas as pd

from experiments.decision import survivorship_audit as audit


def _universe(ended_per_year: int) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    sessions = pd.bdate_range("2015-01-01", "2024-12-31")
    rows = []
    for index in range(500):
        rows.append({"symbol": f"S{index}", "first": sessions[0], "last": sessions[-1],
                     "bars": len(sessions), "last_close": 20.0,
                     "final_60_log_return": 0.05, "median_dollar_volume_final_60": 1e6})
    for year in range(2016, 2024):
        for index in range(ended_per_year):
            rows.append({"symbol": f"D{year}-{index}", "first": sessions[0],
                         "last": pd.Timestamp(year=year, month=6, day=3),
                         "bars": 1000, "last_close": 0.5,
                         "final_60_log_return": -0.6, "median_dollar_volume_final_60": 1e4})
    return pd.DataFrame(rows), sessions


def test_survivor_snapshot_is_flagged():
    symbols, sessions = _universe(ended_per_year=2)
    result = audit.summarize(symbols, sessions)
    assert result["median_yearly_attrition_percent"] < audit.LOW_ATTRITION_PERCENT
    assert result["reading"] == "likely_survivor_snapshot_outcomes_biased_upward"


def test_realistic_attrition_is_not_flagged():
    symbols, sessions = _universe(ended_per_year=40)
    result = audit.summarize(symbols, sessions)
    assert result["median_yearly_attrition_percent"] > audit.LOW_ATTRITION_PERCENT
    assert result["final_60_log_return_median"]["ended_early"] < 0
    assert result["ended_early_last_close_below_1_percent"] == 100.0
