from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import binomtest, chi2, norm


class WalkForwardScoringError(ValueError):
    pass


@dataclass(frozen=True)
class CoverageTests:
    observations: int
    successes: int
    empirical_coverage: float
    exact_marginal_pvalue: float
    christoffersen_independence_pvalue: float
    dynamic_binary_pvalue: float


def _weights(values: Sequence[float]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all() \
            or (result <= 0).any():
        raise WalkForwardScoringError("weights must be a finite positive vector")
    return result


def smoothed_class_probabilities(
    labels: Sequence[str], weights: Sequence[float], classes: Sequence[str], *, alpha: float = .5,
) -> np.ndarray:
    if alpha <= 0 or not np.isfinite(alpha):
        raise WalkForwardScoringError("smoothing alpha must be finite and positive")
    if len(labels) != len(weights):
        raise WalkForwardScoringError("label and weight lengths differ")
    if len(classes) < 2 or len(classes) != len(set(classes)):
        raise WalkForwardScoringError("classes must be unique")
    if not len(labels):
        return np.full(len(classes), 1.0 / len(classes), dtype=np.float64)
    values = _weights(weights)
    positions = {label: index for index, label in enumerate(classes)}
    if unknown := set(labels).difference(positions):
        raise WalkForwardScoringError(f"unknown labels: {sorted(unknown)}")
    mass = np.zeros(len(classes), dtype=np.float64)
    for label, weight in zip(labels, values):
        mass[positions[label]] += weight
    return (mass + alpha) / (float(mass.sum()) + alpha * len(classes))


def effective_sample_size(weights: Sequence[float]) -> float:
    values = _weights(weights)
    return float(values.sum() ** 2 / np.square(values).sum())


def multiclass_brier(probabilities: Sequence[float], observed_index: int) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() \
            or (values < 0).any() or abs(float(values.sum()) - 1.0) > 1e-15:
        raise WalkForwardScoringError("probabilities must be finite and sum to one")
    if not 0 <= observed_index < len(values):
        raise WalkForwardScoringError("observed class index is invalid")
    truth = np.zeros(len(values), dtype=np.float64)
    truth[observed_index] = 1.0
    return float(np.square(values - truth).sum())


def log_loss(probabilities: Sequence[float], observed_index: int) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    if not 0 <= observed_index < len(values) or not np.isfinite(values).all() \
            or (values <= 0).any() or abs(float(values.sum()) - 1.0) > 1e-15:
        raise WalkForwardScoringError("strictly positive normalized probabilities required")
    return float(-np.log(values[observed_index]))


def brier_skill(model_mean: float, baseline_mean: float) -> float:
    if not np.isfinite(model_mean) or not np.isfinite(baseline_mean) or baseline_mean <= 0:
        raise WalkForwardScoringError("finite scores and positive baseline required")
    return float(1.0 - model_mean / baseline_mean)


def weighted_inverted_cdf(
    values: Sequence[float], weights: Sequence[float], quantiles: Sequence[float],
) -> np.ndarray:
    observations = np.asarray(values, dtype=np.float64)
    mass = _weights(weights)
    requested = np.asarray(quantiles, dtype=np.float64)
    if observations.ndim != 1 or len(observations) != len(mass) \
            or not np.isfinite(observations).all():
        raise WalkForwardScoringError("weighted quantile observations are invalid")
    if requested.ndim != 1 or not len(requested) or not np.isfinite(requested).all() \
            or (requested < 0).any() or (requested > 1).any():
        raise WalkForwardScoringError("quantiles must be within zero and one")
    order = np.argsort(observations, kind="stable")
    sorted_values = observations[order]
    cumulative = np.cumsum(mass[order])
    targets = requested * cumulative[-1]
    positions = np.searchsorted(cumulative, targets, side="left")
    positions = np.clip(positions, 0, len(sorted_values) - 1)
    return sorted_values[positions]


def pointwise_weighted_median(
    paths: Sequence[Sequence[float]], weights: Sequence[float],
) -> np.ndarray:
    matrix = np.asarray(paths, dtype=np.float64)
    mass = _weights(weights)
    if matrix.ndim != 2 or matrix.shape[0] != len(mass) or np.isinf(matrix).any():
        raise WalkForwardScoringError("path forecast inputs are invalid")
    result = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    for column in range(matrix.shape[1]):
        valid = ~np.isnan(matrix[:, column])
        if valid.any():
            result[column] = weighted_inverted_cdf(
                matrix[valid, column], mass[valid], [.5],
            )[0]
    return result


def pinball_loss(observed: float, forecast: float, quantile: float) -> float:
    if not all(np.isfinite(value) for value in (observed, forecast, quantile)) \
            or not 0 < quantile < 1:
        raise WalkForwardScoringError("pinball inputs are invalid")
    error = observed - forecast
    return float(max(quantile * error, (quantile - 1.0) * error))


def central_interval_score(observed: float, lower: float, upper: float, coverage: float) -> float:
    if not all(np.isfinite(value) for value in (observed, lower, upper, coverage)) \
            or lower > upper or not 0 < coverage < 1:
        raise WalkForwardScoringError("interval inputs are invalid")
    alpha = 1.0 - coverage
    penalty = 0.0
    if observed < lower:
        penalty = 2.0 * (lower - observed) / alpha
    elif observed > upper:
        penalty = 2.0 * (observed - upper) / alpha
    return float(upper - lower + penalty)


def equal_count_calibration(
    probabilities: Sequence[float], outcomes: Sequence[int], *, bins: int = 10,
    minimum_rows: int = 30,
) -> tuple[pd.DataFrame, float]:
    probability = np.asarray(probabilities, dtype=np.float64)
    outcome = np.asarray(outcomes, dtype=np.int64)
    if len(probability) != len(outcome) or not len(probability) \
            or not np.isfinite(probability).all() or (probability < 0).any() \
            or (probability > 1).any() or not set(outcome).issubset({0, 1}):
        raise WalkForwardScoringError("calibration inputs are invalid")
    usable_bins = min(int(bins), max(1, len(probability) // int(minimum_rows)))
    order = np.argsort(probability, kind="stable")
    rows: list[dict[str, float | int]] = []
    for index, positions in enumerate(np.array_split(order, usable_bins), 1):
        rows.append({
            "bin": index, "rows": len(positions),
            "minimum_probability": float(probability[positions].min()),
            "maximum_probability": float(probability[positions].max()),
            "mean_probability": float(probability[positions].mean()),
            "observed_frequency": float(outcome[positions].mean()),
            "absolute_gap": float(abs(probability[positions].mean() - outcome[positions].mean())),
        })
    frame = pd.DataFrame(rows)
    ece = float(np.average(frame.absolute_gap, weights=frame.rows))
    return frame, ece


def abstention_reasons(
    *, eligible_primary_rows: int, nearest_distance: float,
    prior_nearest_distances: Sequence[float], favorable_prefix_probabilities: Sequence[float],
    data_quality_missing: bool = True, calibration_passed: bool = False,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if eligible_primary_rows < 10:
        reasons.append("insufficient_effective_sample_size")
    history = np.asarray(prior_nearest_distances, dtype=np.float64)
    if len(history) >= 250:
        if not np.isfinite(nearest_distance) or not np.isfinite(history).all():
            raise WalkForwardScoringError("novelty distances are invalid")
        threshold = float(np.quantile(history, .95, method="linear"))
        if nearest_distance > threshold:
            reasons.append("historically_novel_query")
    prefix = np.asarray(favorable_prefix_probabilities, dtype=np.float64)
    if len(prefix) and (
        not np.isfinite(prefix).all() or (prefix < 0).any() or (prefix > 1).any()
    ):
        raise WalkForwardScoringError("prefix probabilities are invalid")
    if len(prefix) and float(prefix.max() - prefix.min()) > .2:
        reasons.append("unstable_neighborhood")
    if data_quality_missing:
        reasons.append("poor_data_quality")
    if not calibration_passed:
        reasons.append("failed_calibration")
    return tuple(reasons)


def expanding_frequency(
    prior_labels: Sequence[str], classes: Sequence[str], *, alpha: float = .5,
) -> np.ndarray:
    return smoothed_class_probabilities(prior_labels, np.ones(len(prior_labels)), classes, alpha=alpha)


def regime_frequency(
    prior_labels: Sequence[str], prior_regimes: Sequence[str], query_regime: str,
    classes: Sequence[str], *, minimum_same_regime: int = 50, alpha: float = .5,
) -> tuple[np.ndarray, bool]:
    if len(prior_labels) != len(prior_regimes):
        raise WalkForwardScoringError("baseline label and regime lengths differ")
    selected = [label for label, regime in zip(prior_labels, prior_regimes) if regime == query_regime]
    fallback = len(selected) < minimum_same_regime
    return expanding_frequency(prior_labels if fallback else selected, classes, alpha=alpha), fallback


def calendar_month_mean_losses(
    cutoffs: Sequence[pd.Timestamp | str], model_losses: Sequence[float],
    baseline_losses: Sequence[float],
) -> pd.DataFrame:
    if not (len(cutoffs) == len(model_losses) == len(baseline_losses)):
        raise WalkForwardScoringError("paired loss lengths differ")
    frame = pd.DataFrame({
        "month": pd.to_datetime(cutoffs).to_period("M").astype(str),
        "model_loss": np.asarray(model_losses, dtype=np.float64),
        "baseline_loss": np.asarray(baseline_losses, dtype=np.float64),
    })
    if frame[["model_loss", "baseline_loss"]].isna().any().any() \
            or not np.isfinite(frame[["model_loss", "baseline_loss"]].to_numpy()).all():
        raise WalkForwardScoringError("paired losses are non-finite")
    result = frame.groupby("month", sort=True, as_index=False).mean()
    result["loss_difference"] = result.model_loss - result.baseline_loss
    return result


def risk_coverage_curve(
    losses: Sequence[float], risk_scores: Sequence[float], *,
    coverage_levels: Sequence[float] = tuple(np.linspace(.1, 1.0, 10)),
) -> pd.DataFrame:
    loss = np.asarray(losses, dtype=np.float64)
    risk = np.asarray(risk_scores, dtype=np.float64)
    coverage = np.asarray(coverage_levels, dtype=np.float64)
    if loss.ndim != 1 or len(loss) != len(risk) or not len(loss) \
            or not np.isfinite(loss).all() or not np.isfinite(risk).all() \
            or not len(coverage) or not np.isfinite(coverage).all() \
            or (coverage <= 0).any() or (coverage > 1).any():
        raise WalkForwardScoringError("risk-coverage inputs are invalid")
    order = np.argsort(risk, kind="stable")
    rows = []
    for level in coverage:
        retained = max(1, int(np.ceil(level * len(loss))))
        selected = order[:retained]
        rows.append({
            "requested_coverage": float(level), "retained_rows": retained,
            "empirical_coverage": float(retained / len(loss)),
            "mean_loss": float(loss[selected].mean()),
            "maximum_retained_risk_score": float(risk[selected].max()),
        })
    return pd.DataFrame(rows)


def moving_block_bootstrap_lower_pvalue(
    loss_differences: Sequence[float], *, resamples: int = 10_000,
    block_length: int = 3, seed: int = 20_260_901,
) -> tuple[float, float]:
    values = np.asarray(loss_differences, dtype=np.float64)
    if values.ndim != 1 or len(values) < block_length or not np.isfinite(values).all() \
            or block_length < 1 or resamples < 1:
        raise WalkForwardScoringError("block-bootstrap inputs are invalid")
    observed = float(values.mean())
    centered = values - observed
    starts = np.arange(len(values) - block_length + 1)
    blocks_needed = int(np.ceil(len(values) / block_length))
    rng = np.random.Generator(np.random.PCG64(seed))
    count = 0
    for _ in range(resamples):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([centered[start:start + block_length] for start in chosen])[:len(values)]
        count += bool(float(sample.mean()) <= observed)
    return observed, float((count + 1) / (resamples + 1))


def diebold_mariano_hac_lower_pvalue(
    loss_differences: Sequence[float], *, lags: int = 3,
) -> tuple[float, float, float]:
    values = np.asarray(loss_differences, dtype=np.float64)
    if values.ndim != 1 or len(values) <= lags or lags < 0 or not np.isfinite(values).all():
        raise WalkForwardScoringError("DM/HAC inputs are invalid")
    centered = values - values.mean()
    n = len(values)
    long_run = float(np.dot(centered, centered) / n)
    for lag in range(1, lags + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / n)
        long_run += 2.0 * (1.0 - lag / (lags + 1.0)) * covariance
    long_run = max(0.0, long_run)
    if long_run == 0:
        statistic = -np.inf if values.mean() < 0 else (np.inf if values.mean() > 0 else 0.0)
    else:
        statistic = float(values.mean() / np.sqrt(long_run / n))
    return float(values.mean()), float(statistic), float(norm.cdf(statistic))


def holm_adjust(pvalues: Mapping[str, float], *, alpha: float = .05) -> dict[str, tuple[float, bool]]:
    if not pvalues or not 0 < alpha < 1 or any(
        not np.isfinite(value) or not 0 <= value <= 1 for value in pvalues.values()
    ):
        raise WalkForwardScoringError("Holm inputs are invalid")
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, tuple[float, bool]] = {}
    running = 0.0
    total = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, (total - index) * value)
        corrected = min(1.0, running)
        adjusted[name] = (float(corrected), bool(corrected < alpha))
    return adjusted


def _bernoulli_log_likelihood(successes: float, trials: float) -> float:
    if trials == 0:
        return 0.0
    failures = trials - successes
    probability = successes / trials
    return float(xlogy(successes, probability) + xlogy(failures, 1.0 - probability))


def interval_coverage_tests(
    covered: Sequence[bool | int], *, nominal_coverage: float, dynamic_lags: int = 3,
) -> CoverageTests:
    hits = np.asarray(covered, dtype=np.int64)
    if hits.ndim != 1 or len(hits) <= dynamic_lags or not set(hits).issubset({0, 1}) \
            or not 0 < nominal_coverage < 1 or dynamic_lags < 1:
        raise WalkForwardScoringError("coverage-test inputs are invalid")
    successes = int(hits.sum())
    marginal = float(binomtest(successes, len(hits), nominal_coverage).pvalue)
    previous, current = hits[:-1], hits[1:]
    n00 = int(((previous == 0) & (current == 0)).sum())
    n01 = int(((previous == 0) & (current == 1)).sum())
    n10 = int(((previous == 1) & (current == 0)).sum())
    n11 = int(((previous == 1) & (current == 1)).sum())
    independent = _bernoulli_log_likelihood(n01 + n11, n00 + n01 + n10 + n11)
    markov = _bernoulli_log_likelihood(n01, n00 + n01) \
        + _bernoulli_log_likelihood(n11, n10 + n11)
    lr = max(0.0, 2.0 * (markov - independent))
    christoffersen = float(chi2.sf(lr, 1))
    centered = hits.astype(float) - nominal_coverage
    response = centered[dynamic_lags:]
    design = np.column_stack([
        np.ones(len(response)),
        *[centered[dynamic_lags - lag:len(centered) - lag] for lag in range(1, dynamic_lags + 1)],
    ])
    coefficients = np.linalg.pinv(design.T @ design) @ design.T @ response
    statistic = float(
        response @ design @ coefficients / (nominal_coverage * (1.0 - nominal_coverage))
    )
    dynamic = float(chi2.sf(max(0.0, statistic), design.shape[1]))
    return CoverageTests(
        len(hits), successes, float(hits.mean()), marginal, christoffersen, dynamic,
    )
