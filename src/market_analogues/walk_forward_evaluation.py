from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from market_analogues.walk_forward_scoring import (
    WalkForwardScoringError,
    central_interval_score,
    log_loss,
    multiclass_brier,
    pinball_loss,
)


PRIMARY_CLASSES = ("favorable_first", "adverse_first", "no_touch")
DIRECTIONAL_CLASSES = ("favorable_first", "adverse_first")
EXCLUDED_ROUTES = ("ambiguous_same_first_touch_bar", "censored")
PERMANENT_PRODUCT_BLOCKERS = ("failed_calibration", "poor_data_quality")
DYNAMIC_ABSTENTION_REASONS = (
    "historically_novel_query",
    "insufficient_effective_sample_size",
    "unstable_neighborhood",
)


@dataclass(frozen=True)
class ProbabilityScores:
    route_status: str
    multiclass_evaluable: bool
    multiclass_brier: float
    multiclass_log_loss: float
    directional_evaluable: bool
    directional_brier: float
    directional_log_loss: float


@dataclass(frozen=True)
class ContinuousScores:
    evaluable: bool
    pinball_q10: float
    pinball_q25: float
    pinball_q50: float
    pinball_q75: float
    pinball_q90: float
    median_absolute_error: float
    central_50_score: float
    central_50_width: float
    central_50_covered: bool | None
    central_80_score: float
    central_80_width: float
    central_80_covered: bool | None


def _nan() -> float:
    return float("nan")


def score_probabilities(
    probabilities: Sequence[float], directional_probabilities: Sequence[float],
    observed_route: str | None,
) -> ProbabilityScores:
    route = "unavailable" if observed_route is None else str(observed_route)
    allowed = set(PRIMARY_CLASSES + EXCLUDED_ROUTES)
    if route != "unavailable" and route not in allowed:
        raise WalkForwardScoringError(f"unknown observed route: {route}")
    multiclass_evaluable = route in PRIMARY_CLASSES
    directional_evaluable = route in DIRECTIONAL_CLASSES
    if multiclass_evaluable:
        observed_index = PRIMARY_CLASSES.index(route)
        primary_brier = multiclass_brier(probabilities, observed_index)
        primary_log = log_loss(probabilities, observed_index)
    else:
        primary_brier = primary_log = _nan()
    if directional_evaluable:
        observed_index = DIRECTIONAL_CLASSES.index(route)
        directional = np.asarray(directional_probabilities, dtype=np.float64)
        if directional.shape != (2,):
            raise WalkForwardScoringError("two directional probabilities required")
        directional_brier = multiclass_brier(directional, observed_index)
        directional_log = log_loss(directional, observed_index)
    else:
        directional_brier = directional_log = _nan()
    return ProbabilityScores(
        route_status=route,
        multiclass_evaluable=multiclass_evaluable,
        multiclass_brier=primary_brier,
        multiclass_log_loss=primary_log,
        directional_evaluable=directional_evaluable,
        directional_brier=directional_brier,
        directional_log_loss=directional_log,
    )


def score_continuous(
    observed: float | None, quantiles: Mapping[float, float | None],
) -> ContinuousScores:
    ordered = tuple(float(q) for q in (.1, .25, .5, .75, .9))
    values = []
    for quantile in ordered:
        value = quantiles.get(quantile)
        values.append(_nan() if value is None else float(value))
    if observed is None or not np.isfinite(float(observed)) or not np.isfinite(values).all():
        return ContinuousScores(
            False, *([_nan()] * 8), None, *([_nan()] * 2), None,
        )
    if np.any(np.diff(values) < 0):
        raise WalkForwardScoringError("forecast quantiles are not monotonic")
    actual = float(observed)
    losses = [pinball_loss(actual, value, q) for q, value in zip(ordered, values)]
    width50 = values[3] - values[1]
    width80 = values[4] - values[0]
    return ContinuousScores(
        evaluable=True,
        pinball_q10=losses[0], pinball_q25=losses[1], pinball_q50=losses[2],
        pinball_q75=losses[3], pinball_q90=losses[4],
        median_absolute_error=abs(actual - values[2]),
        central_50_score=central_interval_score(actual, values[1], values[3], .5),
        central_50_width=width50,
        central_50_covered=bool(values[1] <= actual <= values[3]),
        central_80_score=central_interval_score(actual, values[0], values[4], .8),
        central_80_width=width80,
        central_80_covered=bool(values[0] <= actual <= values[4]),
    )


def split_abstention_reasons(serialized: str | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    reasons = tuple(sorted(filter(None, str(serialized or "").split("|"))))
    known = set(PERMANENT_PRODUCT_BLOCKERS + DYNAMIC_ABSTENTION_REASONS)
    if unknown := set(reasons).difference(known):
        raise WalkForwardScoringError(f"unknown abstention reasons: {sorted(unknown)}")
    permanent = tuple(reason for reason in reasons if reason in PERMANENT_PRODUCT_BLOCKERS)
    dynamic = tuple(reason for reason in reasons if reason in DYNAMIC_ABSTENTION_REASONS)
    return permanent, dynamic


def fold_purge_cutoffs(
    benchmark_sessions: Sequence[pd.Timestamp | str], folds: Sequence[Mapping[str, object]],
    *, purge_sessions: int,
) -> dict[str, pd.Timestamp | None]:
    sessions = pd.DatetimeIndex(pd.to_datetime(tuple(benchmark_sessions))).sort_values().unique()
    if purge_sessions < 1 or len(sessions) <= purge_sessions or sessions.has_duplicates:
        raise WalkForwardScoringError("benchmark purge calendar is invalid")
    ordered = sorted(folds, key=lambda fold: str(fold["start"]))
    result: dict[str, pd.Timestamp | None] = {}
    for index, fold in enumerate(ordered):
        fold_id = str(fold["fold_id"])
        if index == len(ordered) - 1:
            result[fold_id] = None
            continue
        next_start = pd.Timestamp(str(ordered[index + 1]["start"]))
        prior = sessions[sessions < next_start]
        if len(prior) <= purge_sessions:
            raise WalkForwardScoringError(f"insufficient purge calendar for {fold_id}")
        result[fold_id] = pd.Timestamp(prior[-purge_sessions - 1])
    return result


def query_inside_purged_fold(
    cutoff: pd.Timestamp | str, fold_id: str, purge_cutoffs: Mapping[str, pd.Timestamp | None],
) -> bool:
    if fold_id not in purge_cutoffs:
        raise WalkForwardScoringError(f"unknown fold: {fold_id}")
    maximum = purge_cutoffs[fold_id]
    return maximum is None or pd.Timestamp(cutoff) <= maximum


def probability_score_record(
    probabilities: Sequence[float], directional_probabilities: Sequence[float],
    observed_route: str | None,
) -> dict[str, object]:
    return asdict(score_probabilities(probabilities, directional_probabilities, observed_route))


def continuous_score_record(
    observed: float | None, quantiles: Mapping[float, float | None],
) -> dict[str, object]:
    return asdict(score_continuous(observed, quantiles))
