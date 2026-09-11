"""Frozen proper-score and guardrail primitives for analogue improvement M0."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from market_analogues.walk_forward_scoring import (
    WalkForwardScoringError,
    brier_skill,
    equal_count_calibration,
    holm_adjust,
    log_loss,
    multiclass_brier,
)
from market_analogues.walk_forward_evaluation import fold_purge_cutoffs


PRIMARY_CLASSES = ("favorable_first", "adverse_first", "no_touch")
MANDATORY_COMPARATORS = ("matched_causal_base_rate", "locked_retriever")


class AnalogueImprovementError(ValueError):
    pass


@dataclass(frozen=True)
class BlockInference:
    observations: int
    mean_difference: float
    one_sided_lower_pvalue: float
    percentile_lower: float
    simultaneous_upper: float


@dataclass(frozen=True)
class CoverageGate:
    registered: int
    evaluable: int
    forecasted_evaluable: int
    overall_evaluable_fraction: float
    per_fold_evaluable_fraction: Mapping[str, float]
    passed: bool


@dataclass(frozen=True)
class CalibrationGate:
    passed: bool
    class_mean_residual: Mapping[str, float]
    class_ece: Mapping[str, float]
    holm_adjusted_pvalue: Mapping[str, float]
    simultaneous_interval: Mapping[str, tuple[float, float]]


@dataclass(frozen=True)
class ImprovementDecision:
    passed: bool
    gates: Mapping[str, bool]
    reasons: tuple[str, ...]
    brier_skill: Mapping[str, float]
    log_loss_difference: Mapping[str, float]
    fold_brier_skill: Mapping[str, Mapping[str, float]]
    inference: Mapping[str, Mapping[str, float | int]]
    holm_adjusted_pvalue: Mapping[str, float]


def _probability_matrix(values: Sequence[Sequence[float]], rows: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1:] != (len(PRIMARY_CLASSES),) \
            or (rows is not None and len(array) != rows) or not len(array) \
            or not np.isfinite(array).all() or (array <= 0).any() \
            or np.any(np.abs(array.sum(axis=1) - 1.0) > 1e-15):
        raise AnalogueImprovementError("strictly positive normalized three-class probabilities required")
    return array


def _label_positions(labels: Sequence[str]) -> np.ndarray:
    positions = {name: index for index, name in enumerate(PRIMARY_CLASSES)}
    if len(labels) == 0 or set(labels).difference(positions):
        raise AnalogueImprovementError("primary labels are invalid")
    return np.asarray([positions[value] for value in labels], dtype=np.int64)


def probability_losses(
    labels: Sequence[str], probabilities: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact row-level multiclass Brier and natural-log losses."""
    observed = _label_positions(labels)
    matrix = _probability_matrix(probabilities, len(observed))
    brier = np.asarray([
        multiclass_brier(row, int(position)) for row, position in zip(matrix, observed)
    ], dtype=np.float64)
    logarithmic = np.asarray([
        log_loss(row, int(position)) for row, position in zip(matrix, observed)
    ], dtype=np.float64)
    return brier, logarithmic


def weighted_empirical_crps(
    values: Sequence[float], weights: Sequence[float], observed: float,
) -> float:
    """O(n log n) CRPS for a normalized weighted empirical distribution."""
    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if x.ndim != 1 or w.ndim != 1 or not len(x) or len(x) != len(w) \
            or not np.isfinite(x).all() or not np.isfinite(w).all() \
            or (w <= 0).any() or not np.isfinite(observed):
        raise AnalogueImprovementError("CRPS inputs must be finite with positive weights")
    order = np.argsort(x, kind="stable")
    x = x[order]; w = w[order]
    total = math.fsum(map(float, w))
    first = math.fsum(float(weight) * abs(float(value) - float(observed))
                      for value, weight in zip(x, w)) / total
    prior_weight = 0.0; prior_weighted_value = 0.0; pair_half = 0.0
    for value, weight in zip(x, w):
        numeric_value = float(value); numeric_weight = float(weight)
        pair_half += numeric_weight * (numeric_value * prior_weight - prior_weighted_value)
        prior_weight += numeric_weight
        prior_weighted_value += numeric_weight * numeric_value
    result = first - pair_half / (total * total)
    if result < 0 and result >= -16 * np.finfo(np.float64).eps * max(1.0, abs(first)):
        result = 0.0
    if not np.isfinite(result) or result < 0:
        raise AnalogueImprovementError("CRPS numerical result is invalid")
    return float(result)


def monthly_paired_differences(
    cutoffs: Sequence[pd.Timestamp | str], candidate_losses: Sequence[float],
    baseline_losses: Sequence[float],
) -> np.ndarray:
    if not (len(cutoffs) == len(candidate_losses) == len(baseline_losses)) \
            or len(cutoffs) == 0:
        raise AnalogueImprovementError("paired monthly inputs differ")
    frame = pd.DataFrame({
        "month": pd.to_datetime(tuple(cutoffs)).to_period("M").astype(str),
        "difference": np.asarray(candidate_losses, dtype=np.float64)
        - np.asarray(baseline_losses, dtype=np.float64),
    })
    if not np.isfinite(frame.difference).all():
        raise AnalogueImprovementError("paired losses must be finite")
    return frame.groupby("month", sort=True).difference.mean().to_numpy(dtype=np.float64)


def moving_block_lower_inference(
    monthly_differences: Sequence[float], *, resamples: int = 10_000,
    block_length: int = 6, seed: int = 20_260_911,
    simultaneous_comparisons: int = 2, confidence: float = .95,
) -> BlockInference:
    """Lower-tail centered test and Bonferroni simultaneous percentile upper bound."""
    values = np.asarray(monthly_differences, dtype=np.float64)
    if values.ndim != 1 or len(values) < block_length or block_length < 1 \
            or resamples < 1 or simultaneous_comparisons < 1 \
            or not 0 < confidence < 1 or not np.isfinite(values).all():
        raise AnalogueImprovementError("block inference inputs are invalid")
    observed = float(values.mean()); centered = values - observed
    starts = np.arange(len(values) - block_length + 1)
    blocks = int(math.ceil(len(values) / block_length))
    rng = np.random.Generator(np.random.PCG64(seed))
    null_count = 0; means = np.empty(resamples, dtype=np.float64)
    for replicate in range(resamples):
        chosen = rng.choice(starts, size=blocks, replace=True)
        positions = np.concatenate([
            np.arange(start, start + block_length, dtype=np.int64) for start in chosen
        ])[:len(values)]
        null_count += bool(float(centered[positions].mean()) <= observed)
        means[replicate] = float(values[positions].mean())
    tail = (1.0 - confidence) / simultaneous_comparisons
    lower, upper = np.quantile(means, [tail, 1.0 - tail])
    return BlockInference(
        observations=len(values), mean_difference=observed,
        one_sided_lower_pvalue=float((null_count + 1) / (resamples + 1)),
        percentile_lower=float(lower), simultaneous_upper=float(upper),
    )


def classwise_calibration_gate(
    labels: Sequence[str], probabilities: Sequence[Sequence[float]],
    cutoffs: Sequence[pd.Timestamp | str], *, resamples: int = 10_000,
    block_length: int = 6, seed: int = 20_260_931,
) -> CalibrationGate:
    """Monthly block-bootstrap test of mean class residuals with Holm control."""
    observed = _label_positions(labels)
    matrix = _probability_matrix(probabilities, len(observed))
    if len(cutoffs) != len(observed):
        raise AnalogueImprovementError("calibration cutoff length differs")
    months = pd.to_datetime(tuple(cutoffs)).to_period("M").astype(str)
    means: dict[str, float] = {}; eces: dict[str, float] = {}
    raw_p: dict[str, float] = {}; intervals: dict[str, tuple[float, float]] = {}
    for class_index, class_name in enumerate(PRIMARY_CLASSES):
        truth = (observed == class_index).astype(np.int64)
        residual = matrix[:, class_index] - truth
        monthly = pd.DataFrame({"month": months, "residual": residual}).groupby(
            "month", sort=True).residual.mean().to_numpy(dtype=np.float64)
        if len(monthly) < block_length:
            raise AnalogueImprovementError("insufficient calibration months")
        actual = float(monthly.mean())
        if abs(actual) <= 16 * np.finfo(np.float64).eps:
            actual = 0.0
        centered = monthly - actual
        starts = np.arange(len(monthly) - block_length + 1)
        blocks = int(math.ceil(len(monthly) / block_length))
        rng = np.random.Generator(np.random.PCG64(seed + class_index))
        null_count = 0; bootstrap = np.empty(resamples, dtype=np.float64)
        for replicate in range(resamples):
            chosen = rng.choice(starts, size=blocks, replace=True)
            positions = np.concatenate([
                np.arange(start, start + block_length, dtype=np.int64) for start in chosen
            ])[:len(monthly)]
            null_count += bool(abs(float(centered[positions].mean())) >= abs(actual))
            bootstrap[replicate] = float(monthly[positions].mean())
        tail = .05 / (2 * len(PRIMARY_CLASSES))
        lower, upper = np.quantile(bootstrap, [tail, 1 - tail])
        _, ece = equal_count_calibration(matrix[:, class_index], truth)
        means[class_name] = actual; eces[class_name] = ece
        raw_p[class_name] = float((null_count + 1) / (resamples + 1))
        intervals[class_name] = (float(lower), float(upper))
    adjusted = holm_adjust(raw_p)
    return CalibrationGate(
        passed=not any(rejected for _, rejected in adjusted.values()),
        class_mean_residual=means, class_ece=eces,
        holm_adjusted_pvalue={key: value[0] for key, value in adjusted.items()},
        simultaneous_interval=intervals,
    )


def brier_reliability_resolution(
    labels: Sequence[str], probabilities: Sequence[Sequence[float]], *, bins: int = 10,
) -> Mapping[str, float]:
    """Fixed-width multiclass Murphy decomposition disclosure."""
    observed = _label_positions(labels); matrix = _probability_matrix(probabilities, len(observed))
    if type(bins) is not int or bins < 2:
        raise AnalogueImprovementError("at least two fixed-width bins are required")
    reliability = 0.0; resolution = 0.0; uncertainty = 0.0
    for class_index in range(len(PRIMARY_CLASSES)):
        truth = (observed == class_index).astype(np.float64)
        climatology = float(truth.mean())
        uncertainty += climatology * (1.0 - climatology)
        positions = np.minimum((matrix[:, class_index] * bins).astype(np.int64), bins - 1)
        for cell in range(bins):
            selected = positions == cell
            if not selected.any():
                continue
            weight = float(selected.mean())
            forecast = float(matrix[selected, class_index].mean())
            frequency = float(truth[selected].mean())
            reliability += weight * (forecast - frequency) ** 2
            resolution += weight * (frequency - climatology) ** 2
    return {
        "reliability": float(reliability), "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "binning_reconstruction": float(reliability - resolution + uncertainty),
        "mean_brier": float(probability_losses(labels, matrix)[0].mean()),
    }


def validate_chronological_folds(
    benchmark_sessions: Sequence[pd.Timestamp | str],
    folds: Sequence[Mapping[str, object]], *, purge_sessions: int = 60,
) -> Mapping[str, pd.Timestamp | None]:
    if len(folds) < 4 or purge_sessions < 60:
        raise AnalogueImprovementError("at least four folds and a 60-session purge are required")
    ordered = sorted(folds, key=lambda row: pd.Timestamp(str(row["start"])))
    ids = [str(row["fold_id"]) for row in ordered]
    if len(ids) != len(set(ids)):
        raise AnalogueImprovementError("fold IDs duplicate")
    for index, row in enumerate(ordered):
        start = pd.Timestamp(str(row["start"])); end = pd.Timestamp(str(row["end"]))
        if start > end or (index and start <= pd.Timestamp(str(ordered[index - 1]["end"]))):
            raise AnalogueImprovementError("fold periods overlap or are invalid")
    try:
        return fold_purge_cutoffs(benchmark_sessions, ordered, purge_sessions=purge_sessions)
    except WalkForwardScoringError as error:
        raise AnalogueImprovementError(str(error)) from error


def coverage_gate(
    fold_ids: Sequence[str], evaluable: Sequence[bool], forecasted: Sequence[bool],
    *, minimum_evaluable_fraction: float = .9,
) -> CoverageGate:
    folds = np.asarray(tuple(fold_ids), dtype=object)
    labels = np.asarray(tuple(evaluable), dtype=np.bool_)
    predictions = np.asarray(tuple(forecasted), dtype=np.bool_)
    if folds.ndim != 1 or not len(folds) or len(labels) != len(folds) \
            or len(predictions) != len(folds) or len(set(map(str, folds))) < 4 \
            or not 0 < minimum_evaluable_fraction <= 1:
        raise AnalogueImprovementError("coverage inputs are invalid")
    per_fold = {
        str(fold): float(labels[folds == fold].mean()) for fold in sorted(set(map(str, folds)))
    }
    overall = float(labels.mean())
    forecast_complete = bool(np.all(predictions[labels]))
    passed = bool(forecast_complete and overall >= minimum_evaluable_fraction
                  and all(value >= minimum_evaluable_fraction for value in per_fold.values()))
    return CoverageGate(
        registered=len(folds), evaluable=int(labels.sum()),
        forecasted_evaluable=int(np.count_nonzero(labels & predictions)),
        overall_evaluable_fraction=overall, per_fold_evaluable_fraction=per_fold,
        passed=passed,
    )


def require_causal_prediction_order(
    query_cutoffs: Sequence[pd.Timestamp | str],
    neighbor_completion_times: Sequence[pd.Timestamp | str],
    prediction_sealed_times: Sequence[pd.Timestamp | str],
    outcome_opened_times: Sequence[pd.Timestamp | str],
) -> None:
    sizes = {len(query_cutoffs), len(neighbor_completion_times),
             len(prediction_sealed_times), len(outcome_opened_times)}
    if len(sizes) != 1 or not sizes or next(iter(sizes)) == 0:
        raise AnalogueImprovementError("causal order vectors differ")
    query = pd.to_datetime(tuple(query_cutoffs)); neighbor = pd.to_datetime(tuple(neighbor_completion_times))
    sealed = pd.to_datetime(tuple(prediction_sealed_times)); opened = pd.to_datetime(tuple(outcome_opened_times))
    if bool(np.any(neighbor > query)):
        raise AnalogueImprovementError("neighbor outcome was not fully mature at query cutoff")
    if bool(np.any(sealed >= opened)):
        raise AnalogueImprovementError("prediction was not sealed before outcome opening")


def require_outcome_mutation_invariance(
    original_neighbor_ids: Sequence[Sequence[str]],
    mutated_neighbor_ids: Sequence[Sequence[str]],
    original_probabilities: Sequence[Sequence[float]],
    mutated_probabilities: Sequence[Sequence[float]],
) -> None:
    """Require outcome mutation to leave retrieval and sealed forecasts byte-identical."""
    original_ids = tuple(tuple(row) for row in original_neighbor_ids)
    mutated_ids = tuple(tuple(row) for row in mutated_neighbor_ids)
    if original_ids != mutated_ids:
        raise AnalogueImprovementError("future outcome mutation changed neighbor identity or order")
    original = _probability_matrix(original_probabilities)
    mutated = _probability_matrix(mutated_probabilities, len(original))
    if not np.array_equal(original, mutated):
        raise AnalogueImprovementError("future outcome mutation changed sealed probabilities")


def evaluate_primary_improvement(
    *, labels: Sequence[str], candidate_probabilities: Sequence[Sequence[float]],
    matched_probabilities: Sequence[Sequence[float]],
    locked_probabilities: Sequence[Sequence[float]],
    cutoffs: Sequence[pd.Timestamp | str], fold_ids: Sequence[str],
    coverage_passed: bool, calibration_passed: bool, leakage_passed: bool,
    determinism_passed: bool, performance_passed: bool,
    resamples: int = 10_000, block_length: int = 6, seed: int = 20_260_911,
) -> ImprovementDecision:
    """Apply the frozen conjunctive 20-session acceptance decision."""
    rows = len(labels)
    if len(cutoffs) != rows or len(fold_ids) != rows or len(set(fold_ids)) < 4:
        raise AnalogueImprovementError("at least four complete chronological folds are required")
    candidate_brier, candidate_log = probability_losses(labels, candidate_probabilities)
    baselines = {
        "matched_causal_base_rate": probability_losses(labels, matched_probabilities),
        "locked_retriever": probability_losses(labels, locked_probabilities),
    }
    skills = {name: brier_skill(float(candidate_brier.mean()), float(score[0].mean()))
              for name, score in baselines.items()}
    log_differences = {name: float(candidate_log.mean() - score[1].mean())
                       for name, score in baselines.items()}
    folds: dict[str, dict[str, float]] = {}
    fold_array = np.asarray(tuple(map(str, fold_ids)), dtype=object)
    for fold in sorted(set(map(str, fold_ids))):
        selected = fold_array == fold
        if not selected.any():
            raise AnalogueImprovementError("empty fold")
        folds[fold] = {name: brier_skill(float(candidate_brier[selected].mean()),
                                         float(score[0][selected].mean()))
                       for name, score in baselines.items()}
    inferred = {}
    for offset, (name, score) in enumerate(baselines.items()):
        differences = monthly_paired_differences(cutoffs, candidate_brier, score[0])
        inferred[name] = moving_block_lower_inference(
            differences, resamples=resamples, block_length=block_length, seed=seed + offset,
        )
    adjusted = holm_adjust({name: value.one_sided_lower_pvalue
                            for name, value in inferred.items()}, alpha=.05)
    gates = {
        "positive_skill_both_comparators": all(value > 0 for value in skills.values()),
        "holm_pvalues_below_0_05": all(value[1] for value in adjusted.values()),
        "simultaneous_upper_bounds_below_zero": all(
            value.simultaneous_upper < 0 for value in inferred.values()),
        "no_negative_fold": all(value > 0 for row in folds.values() for value in row.values()),
        "log_loss_not_worse": all(value <= 0 for value in log_differences.values()),
        "coverage": bool(coverage_passed), "calibration": bool(calibration_passed),
        "leakage": bool(leakage_passed), "determinism": bool(determinism_passed),
        "performance": bool(performance_passed),
    }
    reasons = tuple(name for name, passed in gates.items() if not passed)
    return ImprovementDecision(
        passed=not reasons, gates=gates, reasons=reasons, brier_skill=skills,
        log_loss_difference=log_differences, fold_brier_skill=folds,
        inference={name: asdict(value) for name, value in inferred.items()},
        holm_adjusted_pvalue={name: value[0] for name, value in adjusted.items()},
    )
