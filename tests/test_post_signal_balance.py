from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import m04r14_t14_12_balance as stage
from experiments.m04r import verify_m04r14_t14_12_balance as verifier


def test_balance_transforms_match_independent_oracle() -> None:
    values = {"prior_return_63": np.array([-.2, .3]), "prior_volatility_20": np.array([0., .2]),
              "prior_close": np.array([5., 100.]), "prior_median_dollar_volume_20": np.array([0., 2e6])}
    for name, array in values.items():
        assert np.array_equal(stage._transform(name, array), verifier._transform(name, array))


def test_summary_standardized_difference_and_histogram_are_exact() -> None:
    state = stage._empty_stats()
    stage._add(state, np.array([1., 2., 3.]), np.array([1., 2., 3.]), np.array([0, 1, 2]))
    row = stage._summary_rows({"broad|up_close_4pct|overall|all|prior_return_63": state})[0]
    assert row["standardized_mean_difference"] == 0.
    assert row["mean_absolute_decile_gap"] == 1.
    assert row["p95_absolute_decile_gap"] == 2


def test_vectorized_year_balance_matches_independent_reconstruction() -> None:
    panel = pd.DataFrame({
        "signal_date": [pd.Timestamp("2024-01-03")] * 3, "symbol": ["E", "A", "B"],
        "prior_return_63": [.1, .1, .2], "prior_volatility_20": [.2, .2, .3],
        "prior_close": [10., 10., 20.], "prior_median_dollar_volume_20": [2e6, 2e6, 3e6],
    })
    controls = pd.DataFrame({
        "signal_date": [pd.Timestamp("2024-01-03")] * 2, "population": ["broad"] * 2,
        "signal_name": ["up_close_4pct"] * 2, "event_symbol": ["E"] * 2,
        "control_symbol": ["A", "B"], "match_tier": ["same_date_unmatched"] * 2,
        **{f"event_{name}_decile": [5, 5] for name in stage.TRANSFORMS},
        **{f"control_{name}_decile": [5, 6] for name in stage.TRANSFORMS},
    })
    observed_stats, observed_reuse = stage._year_stats(2024, panel, controls)
    expected_stats, expected_reuse = verifier._reconstruct_year(2024, panel, controls)
    assert observed_stats.keys() == expected_stats.keys() and observed_reuse == expected_reuse
    for key in observed_stats: verifier._assert_nested(expected_stats[key], observed_stats[key], key)
