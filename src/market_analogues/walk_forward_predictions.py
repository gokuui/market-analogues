from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from market_analogues.walk_forward_scoring import (
    WalkForwardScoringError,
    effective_sample_size,
    smoothed_class_probabilities,
    weighted_inverted_cdf,
)


PRIMARY_CLASSES = ("favorable_first", "adverse_first", "no_touch")
EXCLUDED_ROUTES = ("ambiguous_same_first_touch_bar", "censored")
QUANTILES = (.1, .25, .5, .75, .9)


@dataclass(frozen=True)
class RoutePrediction:
    probabilities: tuple[float, float, float]
    eligible_rows: int
    effective_rows: float
    favorable_rows: int
    adverse_rows: int
    no_touch_rows: int
    ambiguous_rows: int
    censored_rows: int
    unavailable_rows: int


@dataclass(frozen=True)
class ContinuousPrediction:
    quantiles: tuple[float, ...]
    eligible_rows: int
    effective_rows: float


@dataclass(frozen=True)
class DirectionalPrediction:
    favorable_probability: float
    adverse_probability: float
    eligible_rows: int
    effective_rows: float


def rank_weight(rank: int) -> float:
    if type(rank) is not int or rank < 1:
        raise WalkForwardScoringError("rank must be a positive integer")
    return float(2.0 ** (-(rank - 1) / 10.0))


def rank_weights(ranks: Sequence[int], *, weighted: bool) -> np.ndarray:
    values = tuple(ranks)
    if not values or any(type(rank) is not int or rank < 1 for rank in values):
        raise WalkForwardScoringError("ranks must be a nonempty positive integer vector")
    if len(values) != len(set(values)):
        raise WalkForwardScoringError("ranks must be unique")
    if not weighted:
        return np.ones(len(values), dtype=np.float64)
    return np.asarray([rank_weight(rank) for rank in values], dtype=np.float64)


def route_prediction(
    routes: Sequence[str | None], ranks: Sequence[int], *, weighted: bool,
    alpha: float = .5,
) -> RoutePrediction:
    if len(routes) != len(ranks) or not routes:
        raise WalkForwardScoringError("route and rank vectors must be equally nonempty")
    unknown = {
        route for route in routes
        if route is not None and route not in PRIMARY_CLASSES + EXCLUDED_ROUTES
    }
    if unknown:
        raise WalkForwardScoringError(f"unknown barrier routes: {sorted(unknown)}")
    positions = [index for index, route in enumerate(routes) if route in PRIMARY_CLASSES]
    labels = [str(routes[index]) for index in positions]
    selected_ranks = [int(ranks[index]) for index in positions]
    weights = rank_weights(selected_ranks, weighted=weighted) if positions else np.empty(0)
    probabilities = smoothed_class_probabilities(
        labels, weights, PRIMARY_CLASSES, alpha=alpha,
    )
    counts = {name: labels.count(name) for name in PRIMARY_CLASSES}
    return RoutePrediction(
        probabilities=tuple(float(value) for value in probabilities),
        eligible_rows=len(positions),
        effective_rows=effective_sample_size(weights) if len(weights) else 0.0,
        favorable_rows=counts["favorable_first"],
        adverse_rows=counts["adverse_first"],
        no_touch_rows=counts["no_touch"],
        ambiguous_rows=sum(route == "ambiguous_same_first_touch_bar" for route in routes),
        censored_rows=sum(route == "censored" for route in routes),
        unavailable_rows=sum(route is None for route in routes),
    )


def continuous_prediction(
    values: Sequence[float | None], ranks: Sequence[int], *, weighted: bool,
    quantiles: Sequence[float] = QUANTILES,
) -> ContinuousPrediction:
    if len(values) != len(ranks) or not values:
        raise WalkForwardScoringError("continuous and rank vectors must be equally nonempty")
    selected_values: list[float] = []
    selected_ranks: list[int] = []
    for value, rank in zip(values, ranks):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        numeric = float(value)
        if not np.isfinite(numeric):
            raise WalkForwardScoringError("continuous values must be finite or missing")
        selected_values.append(numeric)
        selected_ranks.append(int(rank))
    if not selected_values:
        return ContinuousPrediction(
            tuple(float("nan") for _ in quantiles), 0, 0.0,
        )
    weights = rank_weights(selected_ranks, weighted=weighted)
    forecast = weighted_inverted_cdf(selected_values, weights, quantiles)
    return ContinuousPrediction(
        tuple(float(value) for value in forecast), len(selected_values),
        effective_sample_size(weights),
    )


def directional_prediction(
    routes: Sequence[str | None], ranks: Sequence[int], *, weighted: bool,
    alpha: float = .5,
) -> DirectionalPrediction:
    if len(routes) != len(ranks) or not routes:
        raise WalkForwardScoringError("route and rank vectors must be equally nonempty")
    allowed = PRIMARY_CLASSES + EXCLUDED_ROUTES
    if unknown := {route for route in routes if route is not None and route not in allowed}:
        raise WalkForwardScoringError(f"unknown barrier routes: {sorted(unknown)}")
    positions = [
        index for index, route in enumerate(routes)
        if route in ("favorable_first", "adverse_first")
    ]
    labels = [str(routes[index]) for index in positions]
    selected_ranks = [int(ranks[index]) for index in positions]
    weights = rank_weights(selected_ranks, weighted=weighted) if positions else np.empty(0)
    probability = smoothed_class_probabilities(
        labels, weights, ("favorable_first", "adverse_first"), alpha=alpha,
    )
    return DirectionalPrediction(
        favorable_probability=float(probability[0]),
        adverse_probability=float(probability[1]),
        eligible_rows=len(positions),
        effective_rows=effective_sample_size(weights) if len(weights) else 0.0,
    )


def pointwise_path_prediction(
    values: Sequence[float | None], ranks: Sequence[int], *, weighted: bool,
) -> tuple[float, int, float]:
    prediction = continuous_prediction(
        values, ranks, weighted=weighted, quantiles=(.5,),
    )
    return prediction.quantiles[0], prediction.eligible_rows, prediction.effective_rows
