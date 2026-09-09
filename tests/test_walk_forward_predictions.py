from __future__ import annotations

import math

import numpy as np
import pytest

from market_analogues.walk_forward_predictions import (
    PRIMARY_CLASSES,
    continuous_prediction,
    directional_prediction,
    pointwise_path_prediction,
    rank_weight,
    rank_weights,
    route_prediction,
)
from market_analogues.walk_forward_scoring import WalkForwardScoringError


def test_rank_weights_follow_locked_original_rank_formula() -> None:
    observed = rank_weights([1, 11, 21], weighted=True)
    assert np.array_equal(observed, np.asarray([1.0, .5, .25]))
    assert rank_weight(1) == 1.0
    assert np.array_equal(rank_weights([2, 7], weighted=False), np.ones(2))


def test_route_prediction_preserves_exclusion_accounting_and_original_ranks() -> None:
    prediction = route_prediction(
        ["favorable_first", "ambiguous_same_first_touch_bar", None,
         "adverse_first", "censored", "no_touch"],
        [1, 2, 3, 4, 5, 6], weighted=True,
    )
    weights = np.asarray([rank_weight(1), rank_weight(4), rank_weight(6)])
    expected = (weights + .5) / (weights.sum() + 1.5)
    assert np.array_equal(np.asarray(prediction.probabilities), expected)
    assert prediction.eligible_rows == 3
    assert prediction.effective_rows == pytest.approx(
        weights.sum() ** 2 / (weights @ weights), rel=0, abs=1e-15,
    )
    assert (prediction.favorable_rows, prediction.adverse_rows, prediction.no_touch_rows) == (1, 1, 1)
    assert (prediction.ambiguous_rows, prediction.censored_rows, prediction.unavailable_rows) == (1, 1, 1)


def test_route_prediction_empty_eligible_mass_is_uniform() -> None:
    prediction = route_prediction([None, "censored"], [1, 2], weighted=True)
    assert prediction.probabilities == (1 / 3, 1 / 3, 1 / 3)
    assert prediction.eligible_rows == 0
    assert prediction.effective_rows == 0


def test_unweighted_lane_is_distinct_and_half_smoothed() -> None:
    routes = ["favorable_first", "adverse_first", "favorable_first"]
    prediction = route_prediction(routes, [1, 10, 20], weighted=False)
    assert np.array_equal(
        prediction.probabilities, np.asarray([2.5, 1.5, .5]) / 4.5,
    )


def test_directional_probability_excludes_no_touch_and_half_smooths() -> None:
    prediction = directional_prediction(
        ["favorable_first", "no_touch", "adverse_first", None],
        [1, 2, 3, 4], weighted=False,
    )
    assert prediction.favorable_probability == .5
    assert prediction.adverse_probability == .5
    assert prediction.eligible_rows == 2
    assert prediction.effective_rows == 2


def test_continuous_prediction_excludes_only_missing_measure() -> None:
    prediction = continuous_prediction(
        [4.0, None, 1.0, float("nan"), 2.0], [1, 2, 3, 4, 5],
        weighted=False,
    )
    assert prediction.quantiles == (1.0, 1.0, 2.0, 4.0, 4.0)
    assert prediction.eligible_rows == 3
    assert prediction.effective_rows == 3


def test_pointwise_path_prediction_uses_available_step_only() -> None:
    value, rows, ess = pointwise_path_prediction(
        [None, 3.0, 1.0], [1, 2, 3], weighted=False,
    )
    assert (value, rows, ess) == (1.0, 2, 2.0)


@pytest.mark.parametrize(
    "call",
    [
        lambda: rank_weight(0),
        lambda: rank_weights([1, 1], weighted=True),
        lambda: route_prediction(["unknown"], [1], weighted=True),
        lambda: route_prediction([PRIMARY_CLASSES[0]], [], weighted=True),
        lambda: continuous_prediction([math.inf], [1], weighted=True),
    ],
)
def test_invalid_prediction_inputs_fail_closed(call) -> None:
    with pytest.raises(WalkForwardScoringError):
        call()
