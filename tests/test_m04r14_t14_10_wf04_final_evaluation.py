from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import m04r14_t14_10_wf04_final_evaluation as final
from tests.test_m04r14_t14_10_wf04_nonfinal_evaluation import (
    _baselines,
    _outcomes,
    _raw,
    _registry,
)


def test_final_score_adapter_changes_only_the_fold_boundary() -> None:
    registry = _registry(); registry["fold_id"] = final.FINAL_FOLD
    raw = _raw(); raw["fold_id"] = final.FINAL_FOLD
    baselines = _baselines(); baselines["fold_id"] = final.FINAL_FOLD
    forecasts = pd.DataFrame([{
        "query_id": "q1", "query_cutoff": "2020-03-31", "month": "2020-03",
        "fold_id": final.FINAL_FOLD, "lane": "composite", "horizon_sessions": 20,
        "measure": "close_return", "q10": 0., "q25": .05, "q50": .1,
        "q75": .15, "q90": .2,
    }])
    scores, continuous = final._score_final(
        raw, baselines, forecasts, _outcomes(), registry,
    )
    assert len(scores) == 7 and len(continuous) == 1
    assert set(scores.fold_id) == {final.FINAL_FOLD}
    assert set(continuous.fold_id) == {final.FINAL_FOLD}
    assert scores.purged_evaluation_included.all()
    assert continuous.purged_evaluation_included.all()
    composite = scores.loc[scores.lane == "composite"].iloc[0]
    assert composite.multiclass_brier == .26
    assert continuous.iloc[0].median_absolute_error == pytest.approx(.02)


def test_final_scope_drops_nonfinal_and_renames_fold_dimension() -> None:
    frame = pd.DataFrame([
        {"scope": "validation_3", "dimension": "fold", "value": "validation_3", "x": 1},
        {"scope": "validation_pooled", "dimension": "fold", "value": "validation_3", "x": 2},
    ])
    result = final._final_scope(frame)
    assert result.to_dict("records") == [{
        "scope": final.FINAL_FOLD, "dimension": "fold", "value": final.FINAL_FOLD, "x": 1,
    }]


def test_aggregate_adapter_supplies_unique_rows_to_each_legacy_scope() -> None:
    scores = pd.DataFrame([{"query_id": "q1", "fold_id": final.FINAL_FOLD, "x": 1}])
    result = final._complete_nonfinal_fold_surface(scores)
    assert set(result.fold_id) == set(final.nonfinal.SCORED_NONFINAL_FOLDS)
    assert result.query_id.nunique() == len(final.nonfinal.SCORED_NONFINAL_FOLDS)


def test_report_preserves_failed_decision_and_literal_fraction() -> None:
    metrics = {"fold-metrics.parquet": pd.DataFrame([{
        "lane": "composite", "multiclass_evaluable_rows": 410,
        "mean_multiclass_brier": .6, "mean_multiclass_log_loss": 1.0,
    }])}
    coverage = {
        "final_query_rows": 480, "multiclass_evaluable_queries": 410,
        "literal_nonabstained_fraction": 0.0,
    }
    decision = {
        "research_calibration_pass": False,
        "gates": {"minimum_final_nonabstained_fraction_0_5": False},
    }
    rendered = final._html(metrics, coverage, decision)
    assert "Research calibration pass: <b>False</b>" in rendered
    assert "literal non-abstained fraction: 0.000" in rendered
    assert "minimum_final_nonabstained_fraction_0_5: <b>FAIL</b>" in rendered
