from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from market_analogues.walk_forward_evaluation import (
    continuous_score_record,
    fold_purge_cutoffs,
    probability_score_record,
    query_inside_purged_fold,
    split_abstention_reasons,
)
from market_analogues.walk_forward_scoring import WalkForwardScoringError


def test_probability_scores_exclude_ambiguous_and_preserve_directional_boundary() -> None:
    probability = [.6, .3, .1]
    directional = [.7, .3]
    favorable = probability_score_record(probability, directional, "favorable_first")
    assert favorable["multiclass_evaluable"] is True
    assert favorable["directional_evaluable"] is True
    assert favorable["multiclass_brier"] == pytest.approx(.26)
    assert favorable["multiclass_log_loss"] == pytest.approx(-math.log(.6))
    assert favorable["directional_brier"] == pytest.approx(.18)

    no_touch = probability_score_record(probability, directional, "no_touch")
    assert no_touch["multiclass_evaluable"] is True
    assert no_touch["directional_evaluable"] is False
    assert np.isnan(no_touch["directional_brier"])

    ambiguous = probability_score_record(
        probability, directional, "ambiguous_same_first_touch_bar",
    )
    assert ambiguous["multiclass_evaluable"] is False
    assert np.isnan(ambiguous["multiclass_brier"])


def test_continuous_scores_match_manual_intervals_and_missing_policy() -> None:
    forecast = {.1: 1., .25: 2., .5: 3., .75: 4., .9: 5.}
    inside = continuous_score_record(3.5, forecast)
    assert inside["evaluable"] is True
    assert inside["median_absolute_error"] == .5
    assert inside["central_50_score"] == 2.
    assert inside["central_50_covered"] is True
    assert inside["central_80_width"] == 4.

    tail = continuous_score_record(0., forecast)
    assert tail["central_50_score"] == 10.
    assert tail["central_80_score"] == pytest.approx(14.)
    assert tail["central_80_covered"] is False

    missing = continuous_score_record(None, forecast)
    assert missing["evaluable"] is False
    assert missing["central_50_covered"] is None

    with pytest.raises(WalkForwardScoringError, match="monotonic"):
        continuous_score_record(2., {.1: 1., .25: 3., .5: 2., .75: 4., .9: 5.})


def test_fold_purge_uses_session_count_and_leaves_final_unpurged() -> None:
    sessions = pd.bdate_range("2019-01-01", "2021-12-31")
    folds = [
        {"fold_id": "first", "start": "2019-01-01", "end": "2019-12-31"},
        {"fold_id": "second", "start": "2020-01-01", "end": "2020-12-31"},
        {"fold_id": "final", "start": "2021-01-01", "end": "2021-12-31"},
    ]
    cutoffs = fold_purge_cutoffs(sessions, folds, purge_sessions=5)
    prior = sessions[sessions < pd.Timestamp("2020-01-01")]
    assert cutoffs["first"] == prior[-6]
    assert query_inside_purged_fold(prior[-6], "first", cutoffs)
    assert not query_inside_purged_fold(prior[-5], "first", cutoffs)
    assert cutoffs["final"] is None
    assert query_inside_purged_fold("2021-12-31", "final", cutoffs)


def test_literal_and_research_dynamic_abstention_are_not_conflated() -> None:
    permanent, dynamic = split_abstention_reasons(
        "poor_data_quality|failed_calibration|historically_novel_query",
    )
    assert permanent == ("failed_calibration", "poor_data_quality")
    assert dynamic == ("historically_novel_query",)
    with pytest.raises(WalkForwardScoringError, match="unknown abstention"):
        split_abstention_reasons("convenient_posthoc_reason")
