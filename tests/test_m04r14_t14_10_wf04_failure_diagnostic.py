from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import m04r14_t14_10_wf04_failure_diagnostic as diagnosis


def _scores(count: int = 30) -> pd.DataFrame:
    rows = []
    for index in range(count):
        route = diagnosis.CLASSES[index % 3]
        for lane, brier, probability in (
            ("composite", .4 + index / 1000, (.3, .6, .1)),
            ("unconditional_market_frequency", .5, (.25, .65, .1)),
            ("regime_only_frequency", .49, (.26, .64, .1)),
            ("deterministic_random", .55, (.34, .55, .11)),
            ("recent_return_volatility", .48, (.3, .6, .1)),
            ("price_only", .47, (.31, .59, .1)),
            ("composite_unweighted", .42, (.3, .6, .1)),
        ):
            rows.append({
                "query_id": f"q{index:02d}", "query_cutoff": pd.Timestamp("2024-01-31"),
                "month": "2024-01", "calendar_year": 2024, "fold_id": "final_untouched",
                "route_status": route, "query_regime": "trend=up|volatility=mid",
                "quality_tier": "A", "liquidity_stratum": "low", "quality_liquidity_cell": "A|low",
                "research_dynamic_selective": True, "risk_score": index + 1., "eligible_rows": 20,
                "effective_rows": 18., "multiclass_evaluable": True, "multiclass_brier": brier,
                "multiclass_log_loss": brier + .2, "lane": lane,
                "favorable_probability": probability[0], "adverse_probability": probability[1],
                "no_touch_probability": probability[2],
            })
    return pd.DataFrame(rows)


def _features(count: int = 30) -> pd.DataFrame:
    return pd.DataFrame([{
        "query_id": f"q{index:02d}", "neighborhood_instability": index / 100,
        "novelty_threshold": 2., "favorable_rows": 6, "adverse_rows": 12,
        "no_touch_rows": 2, "ambiguous_rows": 0, "censored_rows": 0, "unavailable_rows": 0,
    } for index in range(count)])


def test_query_comparisons_pair_every_baseline_and_preserve_quintiles() -> None:
    result = diagnosis.query_comparisons(_scores(), _features())
    assert len(result) == 30 * len(diagnosis.BASELINES)
    assert not result.duplicated(["query_id", "baseline_lane"]).any()
    q = result.groupby("query_id").distance_quintile.nunique()
    assert (q == 1).all()
    row = result.loc[(result.query_id == "q00") & (result.baseline_lane == "unconditional_market_frequency")].iloc[0]
    assert row.brier_difference == pytest.approx(-.1)


def test_segment_report_keeps_all_predeclared_cells_above_floor() -> None:
    comparisons = diagnosis.query_comparisons(_scores(), _features())
    result = diagnosis.segment_performance(comparisons)
    overall = result.loc[(result.dimension == "overall") & (result.scope == "final_untouched")]
    assert set(overall.baseline_lane) == set(diagnosis.BASELINES)
    unconditional = overall.loc[overall.baseline_lane == "unconditional_market_frequency"].iloc[0]
    assert unconditional.query_pairs == 30
    assert unconditional.composite_mean_brier == pytest.approx(np.mean([.4 + i / 1000 for i in range(30)]))


def test_probability_behavior_reports_bias_sharpness_and_auc() -> None:
    result = diagnosis.probability_behavior(_scores())
    row = result.loc[
        (result.scope == "final_untouched") & (result.lane == "composite")
        & (result.class_name == "favorable_first")
    ].iloc[0]
    assert row.rows == 30
    assert row.observed_frequency == pytest.approx(1 / 3)
    assert row.mean_probability == pytest.approx(.3)
    assert row.probability_bias == pytest.approx(-1 / 30)
    assert row.probability_standard_deviation == pytest.approx(0, abs=1e-15)
    assert row.one_vs_rest_auc == .5


def test_quintile_retains_missingness_explicitly() -> None:
    result = diagnosis._quintile(pd.Series([1., 2., np.nan, 3., 4., 5.]))
    assert result.iloc[2] == "unavailable"
    assert set(result.drop(index=2)) == {"Q1", "Q2", "Q3", "Q4", "Q5"}
