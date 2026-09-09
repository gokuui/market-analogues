from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import m04r14_t14_10_wf04_nonfinal_evaluation as evaluation


def _registry() -> pd.DataFrame:
    return pd.DataFrame([{
        "query_id": "q1", "cutoff": pd.Timestamp("2020-03-31"), "fold_id": "validation_2",
        "calendar_year": 2020, "quality_tier": "A", "liquidity_stratum": "low",
        "quality_liquidity_cell": "A|low",
    }])


def _raw() -> pd.DataFrame:
    rows = []
    for lane in evaluation.CONTINUOUS_LANES:
        rows.append({
            "query_id": "q1", "query_case_id": "c1", "query_cutoff": "2020-03-31",
            "month": "2020-03", "fold_id": "validation_2", "scored": True,
            "lane": lane, "prefix": 20, "favorable_probability": .6,
            "adverse_probability": .3, "no_touch_probability": .1,
            "directional_favorable_probability": .7, "directional_adverse_probability": .3,
            "nearest_composite_distance": 1.2, "eligible_rows": 20, "effective_rows": 18.,
            "abstention_reasons": "poor_data_quality|failed_calibration",
            "selective_lane": False,
        })
    return pd.DataFrame(rows)


def _baselines() -> pd.DataFrame:
    return pd.DataFrame([{
        "query_id": "q1", "query_case_id": "c1", "query_cutoff": "2020-03-31",
        "month": "2020-03", "fold_id": "validation_2", "scored": True, "lane": lane,
        "prior_eligible_rows": 100, "favorable_probability": .4,
        "adverse_probability": .35, "no_touch_probability": .25,
        "directional_favorable_probability": .55, "directional_adverse_probability": .45,
    } for lane in ("unconditional_market_frequency", "regime_only_frequency")])


def _outcomes() -> pd.DataFrame:
    return pd.DataFrame([{
        "query_id": "q1", "horizon_sessions": horizon, "complete": True,
        "barrier_label": "favorable_first", "barrier_status": "complete", "status": "complete",
        "query_regime": "trend=up|volatility=mid", "close_return": .12,
        "benchmark_relative_return": .08, "maximum_favorable_excursion": .2,
        "maximum_adverse_excursion": -.04,
    } for horizon in (5, 10, 20, 40, 60, 126)])


def test_nonfinal_probability_join_preserves_literal_and_dynamic_selectivity() -> None:
    cutoffs = {"validation_2": pd.Timestamp("2021-07-02")}
    result = evaluation._probability_scores(
        _raw(), _baselines(), _outcomes(), _registry(), cutoffs,
    )
    assert len(result) == 7
    composite = result.loc[result.lane == "composite"].iloc[0]
    assert composite.purged_evaluation_included
    assert not composite.literal_selective
    assert composite.research_dynamic_selective
    assert composite.multiclass_evaluable
    assert composite.multiclass_brier == .26


def test_continuous_join_scores_only_available_measure() -> None:
    forecasts = pd.DataFrame([{
        "query_id": "q1", "query_cutoff": "2020-03-31", "month": "2020-03",
        "fold_id": "validation_2", "lane": "composite", "horizon_sessions": 20,
        "measure": "close_return", "q10": 0., "q25": .05, "q50": .1,
        "q75": .15, "q90": .2,
    }])
    result = evaluation._continuous_scores(
        forecasts, _outcomes(), _registry(), {"validation_2": pd.Timestamp("2021-07-02")},
    )
    assert len(result) == 1
    assert result.iloc[0].evaluable
    assert result.iloc[0].median_absolute_error == pytest.approx(.02)


def test_final_prediction_is_rejected_before_any_score() -> None:
    raw = _raw().copy()
    raw["fold_id"] = "final_untouched"
    raw["query_cutoff"] = "2024-01-31"
    baseline = _baselines().copy()
    baseline["fold_id"] = "final_untouched"
    baseline["query_cutoff"] = "2024-01-31"
    try:
        evaluation._probability_scores(
            raw, baseline, _outcomes(), _registry(), {"final_untouched": None},
        )
    except evaluation.WalkForwardEvaluationError as error:
        assert "final prediction" in str(error)
    else:
        raise AssertionError("final prediction entered the non-final scorer")


def test_complete_aggregate_surface_is_deterministic_on_synthetic_months() -> None:
    rows = []
    for fold_index, fold in enumerate(evaluation.SCORED_NONFINAL_FOLDS):
        for month_index in range(6):
            query_id = f"{fold}-{month_index}"
            route = evaluation.PRIMARY_CLASSES[(fold_index + month_index) % 3]
            for lane_index, lane in enumerate(evaluation.PROBABILITY_LANES):
                favorable = .5 - lane_index * .01
                adverse = .3 + lane_index * .005
                rows.append({
                    "query_id": query_id, "query_cutoff": f"2020-{month_index + 1:02d}-28",
                    "month": f"2020-{month_index + 1:02d}", "calendar_year": 2020,
                    "fold_id": fold, "lane": lane, "query_regime": "r",
                    "quality_liquidity_cell": "A|low", "purged_evaluation_included": True,
                    "multiclass_evaluable": True, "directional_evaluable": route != "no_touch",
                    "route_status": route,
                    "multiclass_brier": (lane_index + 1) / 100 + month_index / 1000,
                    "multiclass_log_loss": (lane_index + 1) / 10,
                    "directional_brier": (lane_index + 1) / 50,
                    "directional_log_loss": (lane_index + 1) / 20,
                    "literal_selective": False, "research_dynamic_selective": True,
                    "risk_score": float(month_index + 1),
                    "favorable_probability": favorable, "adverse_probability": adverse,
                    "no_touch_probability": 1 - favorable - adverse,
                })
    scores = pd.DataFrame(rows)
    fold_metrics = evaluation._fold_metrics(scores)
    stability = evaluation._stability_metrics(scores)
    calibration = evaluation._calibration(scores)
    risk = evaluation._risk_coverage(scores)
    inference = evaluation._inference(scores)
    assert len(fold_metrics) == 5 * len(evaluation.PROBABILITY_LANES)
    assert set(stability.dimension) == {
        "fold", "calendar_year", "benchmark_regime", "quality_liquidity_cell",
    }
    assert len(calibration) > 0 and np.isfinite(calibration.ece).all()
    assert len(risk) == 50
    assert len(inference) == 5 * len(evaluation.COMPARISON_BASELINES)
    assert np.isfinite(inference.bootstrap_lower_pvalue).all()
