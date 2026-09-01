from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest, norm

from market_analogues.walk_forward_scoring import (
    WalkForwardScoringError,
    abstention_reasons,
    brier_skill,
    calendar_month_mean_losses,
    central_interval_score,
    diebold_mariano_hac_lower_pvalue,
    effective_sample_size,
    equal_count_calibration,
    expanding_frequency,
    holm_adjust,
    interval_coverage_tests,
    log_loss,
    moving_block_bootstrap_lower_pvalue,
    multiclass_brier,
    pinball_loss,
    pointwise_weighted_median,
    regime_frequency,
    risk_coverage_curve,
    smoothed_class_probabilities,
    weighted_inverted_cdf,
)


CLASSES = ("favorable_first", "adverse_first", "no_touch")


def test_half_smoothed_multiclass_and_zero_count_oracles() -> None:
    probabilities = smoothed_class_probabilities(
        ["favorable_first", "favorable_first", "no_touch"], [1.0, .5, 2.0], CLASSES,
    )
    np.testing.assert_array_equal(probabilities, np.array([2.0, .5, 2.5]) / 5.0)
    np.testing.assert_array_equal(expanding_frequency([], CLASSES), np.repeat(1 / 3, 3))
    assert np.all(probabilities > 0) and probabilities.sum() == 1.0


def test_effective_sample_size_uses_frozen_exponent_spelling() -> None:
    weights = [1.0, .5, .25]
    assert effective_sample_size(weights) == sum(weights) ** 2 / sum(x * x for x in weights)


def test_proper_scores_and_skill_match_manual_values() -> None:
    p = [.6, .3, .1]
    assert multiclass_brier(p, 0) == pytest.approx((.6 - 1) ** 2 + .3 ** 2 + .1 ** 2)
    assert log_loss(p, 0) == pytest.approx(-math.log(.6))
    assert brier_skill(.18, .24) == pytest.approx(.25)
    with pytest.raises(WalkForwardScoringError):
        log_loss([1.0, 0.0], 0)


def test_weighted_inverted_cdf_handles_ties_and_boundaries() -> None:
    observed = weighted_inverted_cdf([4, 1, 2, 2], [1, 1, 2, 1], [0, .1, .5, .9, 1])
    np.testing.assert_array_equal(observed, [1, 1, 2, 4, 4])


def test_pointwise_path_median_excludes_only_missing_measure() -> None:
    result = pointwise_weighted_median([[1, np.nan, 4], [2, 3, np.nan]], [1, 2])
    np.testing.assert_equal(result, [2, 3, 4])


def test_pinball_and_interval_scores_match_piecewise_oracle() -> None:
    assert pinball_loss(3, 2, .25) == pytest.approx(.25)
    assert pinball_loss(1, 2, .25) == pytest.approx(.75)
    assert central_interval_score(2, 1, 3, .8) == 2
    assert central_interval_score(0, 1, 3, .8) == pytest.approx(12)
    assert central_interval_score(4, 1, 3, .8) == pytest.approx(12)


def test_equal_count_calibration_uses_stable_bins_and_weighted_ece() -> None:
    probability = np.linspace(.05, .95, 60)
    outcome = np.array([0] * 30 + [1] * 30)
    frame, ece = equal_count_calibration(probability, outcome, bins=10, minimum_rows=30)
    assert len(frame) == 2 and frame.rows.tolist() == [30, 30]
    expected = np.mean([
        abs(probability[:30].mean()), abs(probability[30:].mean() - 1),
    ])
    assert ece == pytest.approx(expected)


def test_abstention_thresholds_are_strict_and_warmup_is_not_novel() -> None:
    history = np.arange(250, dtype=float)
    threshold = np.quantile(history, .95, method="linear")
    base = abstention_reasons(
        eligible_primary_rows=9, nearest_distance=10_000,
        prior_nearest_distances=history[:249], favorable_prefix_probabilities=[.4, .61],
    )
    assert base == (
        "insufficient_effective_sample_size", "unstable_neighborhood",
        "poor_data_quality", "failed_calibration",
    )
    assert "historically_novel_query" not in abstention_reasons(
        eligible_primary_rows=10, nearest_distance=threshold,
        prior_nearest_distances=history, favorable_prefix_probabilities=[.4, .6],
    )
    assert "historically_novel_query" in abstention_reasons(
        eligible_primary_rows=10, nearest_distance=np.nextafter(threshold, np.inf),
        prior_nearest_distances=history, favorable_prefix_probabilities=[.4, .6],
    )


def test_regime_baseline_falls_back_at_49_but_not_50() -> None:
    labels = [CLASSES[index % 3] for index in range(60)]
    regimes = ["target"] * 50 + ["other"] * 10
    direct, fallback = regime_frequency(labels, regimes, "target", CLASSES)
    assert fallback is False
    np.testing.assert_array_equal(
        direct, smoothed_class_probabilities(labels[:50], np.ones(50), CLASSES),
    )
    _, fallback = regime_frequency(labels[:49], regimes[:49], "target", CLASSES)
    assert fallback is True


def test_calendar_month_aggregation_does_not_treat_stocks_as_independent() -> None:
    frame = calendar_month_mean_losses(
        ["2020-01-02", "2020-01-30", "2020-02-03"], [.1, .3, .4], [.2, .4, .5],
    )
    assert frame.month.tolist() == ["2020-01", "2020-02"]
    np.testing.assert_allclose(frame.model_loss, [.2, .4])
    np.testing.assert_allclose(frame.loss_difference, [-.1, -.1])


def test_risk_coverage_retains_lowest_risk_with_stable_ties() -> None:
    frame = risk_coverage_curve([.4, .1, .3, .2], [4, 1, 3, 2], coverage_levels=[.5, 1])
    assert frame.retained_rows.tolist() == [2, 4]
    np.testing.assert_allclose(frame.mean_loss, [.15, .25])


def _bootstrap_reference(values: np.ndarray, resamples: int, block: int, seed: int) -> float:
    observed = values.mean()
    centered = values - observed
    rng = np.random.Generator(np.random.PCG64(seed))
    starts = np.arange(len(values) - block + 1)
    needed = math.ceil(len(values) / block)
    boot = []
    for _ in range(resamples):
        chosen = rng.choice(starts, size=needed, replace=True)
        sample = np.hstack([centered[start:start + block] for start in chosen])[:len(values)]
        boot.append(sample.mean())
    return (sum(value <= observed for value in boot) + 1) / (resamples + 1)


def test_block_bootstrap_is_seeded_centered_and_reproducible() -> None:
    values = np.array([-.4, -.2, .1, -.3, 0, -.1])
    observed, pvalue = moving_block_bootstrap_lower_pvalue(
        values, resamples=200, block_length=3, seed=7,
    )
    assert observed == pytest.approx(values.mean())
    assert pvalue == _bootstrap_reference(values, 200, 3, 7)


def test_dm_hac_matches_separate_newey_west_calculation() -> None:
    values = np.array([-.4, -.2, .1, -.3, 0, -.1, -.2, .05])
    mean, statistic, pvalue = diebold_mariano_hac_lower_pvalue(values, lags=3)
    centered = values - values.mean()
    variance = centered @ centered / len(values)
    variance += sum(
        2 * (1 - lag / 4) * (centered[lag:] @ centered[:-lag] / len(values))
        for lag in range(1, 4)
    )
    expected = values.mean() / math.sqrt(max(0, variance) / len(values))
    assert mean == pytest.approx(values.mean())
    assert statistic == pytest.approx(expected)
    assert pvalue == pytest.approx(norm.cdf(expected))


def test_holm_is_monotone_and_uses_strict_alpha() -> None:
    result = holm_adjust({"a": .01, "b": .03, "c": .2})
    assert result == {"a": (.03, True), "b": (.06, False), "c": (.2, False)}


def test_coverage_tests_bind_marginal_and_dependence_checks() -> None:
    hits = np.array(([1] * 8 + [0] * 2) * 10)
    result = interval_coverage_tests(hits, nominal_coverage=.8, dynamic_lags=3)
    assert result.observations == 100 and result.successes == 80
    assert result.empirical_coverage == .8
    assert result.exact_marginal_pvalue == pytest.approx(binomtest(80, 100, .8).pvalue)
    assert 0 <= result.christoffersen_independence_pvalue <= 1
    assert 0 <= result.dynamic_binary_pvalue <= 1


@pytest.mark.parametrize("function,args", [
    (smoothed_class_probabilities, (["unknown"], [1], CLASSES)),
    (weighted_inverted_cdf, ([1, np.nan], [1, 1], [.5])),
    (moving_block_bootstrap_lower_pvalue, ([1, 2],)),
    (interval_coverage_tests, ([1, 2, 1, 0],)),
])
def test_adversarial_inputs_are_rejected(function, args) -> None:
    with pytest.raises((WalkForwardScoringError, TypeError)):
        function(*args)
