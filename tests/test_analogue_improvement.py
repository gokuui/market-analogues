from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.m04r import m04r15_m0_analogue_improvement_synthetic_gate as synthetic
from market_analogues.analogue_improvement import (
    AnalogueImprovementError,
    PRIMARY_CLASSES,
    brier_reliability_resolution,
    classwise_calibration_gate,
    coverage_gate,
    evaluate_primary_improvement,
    monthly_paired_differences,
    moving_block_lower_inference,
    probability_losses,
    require_causal_prediction_order,
    require_outcome_mutation_invariance,
    validate_chronological_folds,
    weighted_empirical_crps,
)


def fixture(rows_per_month: int = 3):
    months = pd.period_range("2026-01", periods=48, freq="M")
    labels = []; cutoffs = []; folds = []
    candidate = []; matched = []; locked = []
    for month_index, month in enumerate(months):
        for row in range(rows_per_month):
            position = (month_index + row) % 3
            labels.append(PRIMARY_CLASSES[position])
            cutoffs.append(month.to_timestamp("M"))
            folds.append(f"fold-{month_index // 12}")
            candidate.append([.075, .075, .075]); candidate[-1][position] = .85
            matched.append([1/3, 1/3, 1/3])
            locked.append([.275, .275, .275]); locked[-1][position] = .45
    return labels, cutoffs, folds, candidate, matched, locked


def naive_crps(values, weights, observed):
    values = np.asarray(values, float); weights = np.asarray(weights, float); total = weights.sum()
    return float(np.sum(weights * np.abs(values-observed))/total
                 - np.sum(weights[:,None]*weights[None,:]*np.abs(values[:,None]-values[None,:]))/(2*total**2))


def test_contract_digest_and_no_claim_boundary() -> None:
    path = Path("config/analogue-improvement-metric-contract-v1.json")
    value = json.loads(path.read_text()); digest = value.pop("contract_digest")
    assert digest == sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert value["claims"]["real_forward_outcomes_opened_by_M0"] is False
    assert value["primary_target"]["categorical_labels_at_other_horizons"] is False


def test_contract_decoder_rejects_duplicate_and_nonfinite(tmp_path: Path) -> None:
    with pytest.raises(synthetic.SyntheticGateError, match="duplicate"):
        synthetic.decode_json(b'{"a":1,"a":2}', tmp_path / "x")
    with pytest.raises(synthetic.SyntheticGateError, match="nonfinite"):
        synthetic.decode_json(b'{"a":NaN}', tmp_path / "x")


def test_probability_losses_match_naive_oracle() -> None:
    labels = ["favorable_first", "no_touch"]
    probabilities = [[.6, .3, .1], [.2, .2, .6]]
    brier, logarithmic = probability_losses(labels, probabilities)
    np.testing.assert_allclose(brier, [.26, .24])
    np.testing.assert_allclose(logarithmic, [-math.log(.6), -math.log(.6)])
    with pytest.raises(AnalogueImprovementError):
        probability_losses(labels, [[.6, .3, .2], [.2, .2, .6]])


@pytest.mark.parametrize("values,weights,observed", [
    ([0.], [2.], 1.), ([-1., 0., 2.], [1., 2., 4.], .5),
    ([2., 2., -3., 7.], [.1, 9., 1., 2.], 2.),
])
def test_weighted_empirical_crps_matches_quadratic_oracle(values, weights, observed) -> None:
    actual = weighted_empirical_crps(values, weights, observed)
    assert actual == pytest.approx(naive_crps(values, weights, observed), abs=1e-15)
    assert weighted_empirical_crps(values, np.asarray(weights)*17, observed) == pytest.approx(actual)


def test_monthly_pairs_average_stocks_before_inference() -> None:
    value = monthly_paired_differences(
        ["2026-01-02", "2026-01-30", "2026-02-03"], [.1, .3, .2], [.4, .4, .4],
    )
    np.testing.assert_allclose(value, [-.2, -.2])


def test_block_inference_matches_independent_seeded_oracle() -> None:
    values = np.asarray([-.4, -.2, -.3, -.1, -.5, -.2, -.3, -.1])
    result = moving_block_lower_inference(values, resamples=200, block_length=3, seed=7)
    centered = values - values.mean(); starts = np.arange(6); rng = np.random.Generator(np.random.PCG64(7))
    null = []; raw = []
    for _ in range(200):
        selected = rng.choice(starts, size=3, replace=True)
        positions = np.concatenate([np.arange(start,start+3) for start in selected])[:8]
        null.append(centered[positions].mean()); raw.append(values[positions].mean())
    assert result.one_sided_lower_pvalue == (sum(value <= values.mean() for value in null)+1)/201
    assert result.simultaneous_upper == pytest.approx(np.quantile(raw, .975))


def test_coverage_gate_requires_all_evaluable_forecasts_and_every_fold() -> None:
    folds = [f"f{i//10}" for i in range(40)]
    passed = coverage_gate(folds, [True]*40, [True]*40)
    assert passed.passed and passed.forecasted_evaluable == 40
    failed = coverage_gate(folds, [True]*40, [False]+[True]*39)
    assert failed.passed is False
    evaluable = [False, False] + [True]*38
    assert coverage_gate(folds, evaluable, [True]*40).passed is False


def test_causal_order_accepts_boundary_and_refuses_leakage() -> None:
    require_causal_prediction_order(
        ["2026-02-01"], ["2026-02-01"], ["2026-02-02"], ["2026-02-03"],
    )
    with pytest.raises(AnalogueImprovementError, match="mature"):
        require_causal_prediction_order(
            ["2026-02-01"], ["2026-02-02"], ["2026-02-02"], ["2026-02-03"],
        )
    with pytest.raises(AnalogueImprovementError, match="sealed"):
        require_causal_prediction_order(
            ["2026-02-01"], ["2026-02-01"], ["2026-02-03"], ["2026-02-03"],
        )


def test_future_outcome_mutation_must_not_change_neighbors_or_predictions() -> None:
    ids = [["e1", "e2"], ["e3"]]; probabilities = [[.6,.3,.1],[.2,.2,.6]]
    require_outcome_mutation_invariance(ids, ids, probabilities, probabilities)
    with pytest.raises(AnalogueImprovementError, match="neighbor"):
        require_outcome_mutation_invariance(ids, [["e2","e1"],["e3"]], probabilities, probabilities)
    changed = [[.5,.4,.1],[.2,.2,.6]]
    with pytest.raises(AnalogueImprovementError, match="probabilities"):
        require_outcome_mutation_invariance(ids, ids, probabilities, changed)


def test_four_fold_sixty_session_purge_and_overlap_refusal() -> None:
    sessions = pd.bdate_range("2020-01-01", "2024-12-31")
    folds = [{"fold_id":f"f{i}","start":f"{2020+i}-01-01","end":f"{2020+i}-12-31"} for i in range(4)]
    limits = validate_chronological_folds(sessions, folds)
    assert set(limits) == {"f0","f1","f2","f3"} and limits["f3"] is None
    folds[1]["start"] = "2020-12-01"
    with pytest.raises(AnalogueImprovementError, match="overlap"):
        validate_chronological_folds(sessions, folds)


def test_calibration_and_brier_decomposition_are_finite_and_deterministic() -> None:
    labels, cutoffs, _, candidate, _, _ = fixture()
    first = classwise_calibration_gate(labels, candidate, cutoffs, resamples=200, block_length=6)
    second = classwise_calibration_gate(labels, candidate, cutoffs, resamples=200, block_length=6)
    assert asdict(first) == asdict(second) and first.passed
    disclosure = brier_reliability_resolution(labels, candidate)
    assert set(disclosure) == {"reliability","resolution","uncertainty","binning_reconstruction","mean_brier"}
    assert all(np.isfinite(value) and value >= 0 for value in disclosure.values())


def test_primary_candidate_must_beat_both_baselines_in_every_fold() -> None:
    labels, cutoffs, folds, candidate, matched, locked = fixture()
    decision = evaluate_primary_improvement(
        labels=labels, candidate_probabilities=candidate, matched_probabilities=matched,
        locked_probabilities=locked, cutoffs=cutoffs, fold_ids=folds,
        coverage_passed=True, calibration_passed=True, leakage_passed=True,
        determinism_passed=True, performance_passed=True, resamples=200, block_length=6,
    )
    assert decision.passed and all(decision.gates.values())
    assert all(value > 0 for value in decision.brier_skill.values())
    failed = evaluate_primary_improvement(
        labels=labels, candidate_probabilities=locked, matched_probabilities=matched,
        locked_probabilities=locked, cutoffs=cutoffs, fold_ids=folds,
        coverage_passed=True, calibration_passed=True, leakage_passed=True,
        determinism_passed=True, performance_passed=True, resamples=200, block_length=6,
    )
    assert not failed.passed and "positive_skill_both_comparators" in failed.reasons

    unstable = [row[:] for row in candidate]
    for index, fold in enumerate(folds):
        if fold == "fold-3":
            correct = PRIMARY_CLASSES.index(labels[index]); wrong = (correct + 1) % 3
            unstable[index] = [.075,.075,.075]; unstable[index][wrong] = .85
    era_failure = evaluate_primary_improvement(
        labels=labels, candidate_probabilities=unstable, matched_probabilities=matched,
        locked_probabilities=locked, cutoffs=cutoffs, fold_ids=folds,
        coverage_passed=True, calibration_passed=True, leakage_passed=True,
        determinism_passed=True, performance_passed=True, resamples=200, block_length=6,
    )
    assert era_failure.gates["positive_skill_both_comparators"] is True
    assert era_failure.gates["no_negative_fold"] is False and not era_failure.passed


@pytest.mark.parametrize("guardrail", ["coverage_passed", "calibration_passed", "leakage_passed", "determinism_passed", "performance_passed"])
def test_no_scientific_strength_can_rescue_failed_guardrail(guardrail: str) -> None:
    labels, cutoffs, folds, candidate, matched, locked = fixture()
    gates = dict(coverage_passed=True, calibration_passed=True, leakage_passed=True,
                 determinism_passed=True, performance_passed=True); gates[guardrail] = False
    result = evaluate_primary_improvement(
        labels=labels, candidate_probabilities=candidate, matched_probabilities=matched,
        locked_probabilities=locked, cutoffs=cutoffs, fold_ids=folds,
        resamples=50, block_length=6, **gates,
    )
    assert result.passed is False
