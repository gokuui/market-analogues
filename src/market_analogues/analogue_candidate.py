"""Frozen market-agnostic candidate distribution primitives for M04R-15 M1."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PRIMARY_CLASSES = ("favorable_first", "adverse_first", "no_touch")
COMPONENTS = (
    "matched_causal_history",
    "composite",
    "price_only",
    "recent_return_volatility",
)
MIXTURE_WEIGHTS: Mapping[str, float] = {
    "matched_causal_history": .4,
    "composite": .1,
    "price_only": .2,
    "recent_return_volatility": .3,
}
FALLBACK_LEVELS = (
    ("exact", ("market_regime", "prefix_quality_class", "trailing_liquidity_cell")),
    ("regime_and_liquidity", ("market_regime", "trailing_liquidity_cell")),
    ("regime", ("market_regime",)),
)


class AnalogueCandidateError(ValueError):
    pass


@dataclass(frozen=True)
class MatchedProbability:
    probabilities: tuple[float, float, float]
    fallback_level: str
    support_rows: int
    mature_evaluable_rows: int


def _probability_vector(values: Sequence[float]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all() \
            or (result <= 0).any() \
            or abs(float(result.sum()) - 1.0) > 1e-15:
        raise AnalogueCandidateError("strictly positive normalized three-class probabilities required")
    return result


def _timestamp(value: object, field: str) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise AnalogueCandidateError(f"invalid {field}") from error
    if pd.isna(result):
        raise AnalogueCandidateError(f"invalid {field}")
    if result.tzinfo is not None:
        result = result.tz_convert("UTC").tz_localize(None)
    return result


def matched_causal_probability(
    outcomes: Sequence[Mapping[str, Any]], *, query_cutoff: object,
    market_regime: str, prefix_quality_class: str,
    trailing_liquidity_cell: str, minimum_support: int = 30,
    half_count: float = .5,
) -> MatchedProbability:
    """Return the frozen causal cell frequency with deterministic fallback.

    Every supplied row must have a unique ``query_id`` and the five fields used
    below. Non-primary labels and not-yet-mature rows are retained by the caller
    but cannot enter the probability mass.
    """
    if type(minimum_support) is not int or isinstance(minimum_support, bool) \
            or minimum_support < 1 or not math.isfinite(half_count) or half_count <= 0:
        raise AnalogueCandidateError("matched-history support/smoothing differs")
    cells = {
        "market_regime": market_regime,
        "prefix_quality_class": prefix_quality_class,
        "trailing_liquidity_cell": trailing_liquidity_cell,
    }
    if any(type(value) is not str or not value for value in cells.values()):
        raise AnalogueCandidateError("query matching cells must be non-empty strings")
    cutoff = _timestamp(query_cutoff, "query cutoff")
    seen: set[str] = set()
    mature: list[Mapping[str, Any]] = []
    required = {"query_id", "origin_cutoff", "completion_timestamp", "label", *cells}
    for row in outcomes:
        if type(row) is not dict or not required.issubset(row):
            raise AnalogueCandidateError("matched-history row fields differ")
        query_id = row["query_id"]
        if type(query_id) is not str or not query_id or query_id in seen:
            raise AnalogueCandidateError("matched-history query IDs must be unique")
        seen.add(query_id)
        completion = row["completion_timestamp"]
        label = row["label"]
        if completion is None or label not in PRIMARY_CLASSES:
            continue
        origin = _timestamp(row["origin_cutoff"], "outcome origin")
        completed = _timestamp(completion, "outcome completion")
        if completed <= origin:
            raise AnalogueCandidateError("outcome completion must follow its origin")
        if origin < cutoff and completed <= cutoff:
            mature.append(row)

    selected: list[Mapping[str, Any]] | None = None
    selected_level = "unconditional"
    for level, fields in FALLBACK_LEVELS:
        candidate = [row for row in mature if all(row[field] == cells[field] for field in fields)]
        if len(candidate) >= minimum_support:
            selected = candidate
            selected_level = level
            break
    if selected is None:
        selected = mature
        selected_level = "unconditional" if selected else "uniform_no_history"
    counts = np.asarray([
        sum(row["label"] == class_name for row in selected)
        for class_name in PRIMARY_CLASSES
    ], dtype=np.float64)
    probabilities = (counts + half_count) / (float(counts.sum()) + half_count * 3)
    return MatchedProbability(
        tuple(map(float, probabilities)), selected_level, len(selected), len(mature),
    )


def matched_causal_probabilities_batch(
    outcomes: Sequence[Mapping[str, Any]], queries: Sequence[Mapping[str, Any]], *,
    minimum_support: int = 30, half_count: float = .5,
) -> Mapping[str, MatchedProbability]:
    """Incrementally form the same causal baseline for cutoff-ordered queries."""
    if type(minimum_support) is not int or isinstance(minimum_support, bool) \
            or minimum_support < 1 or not math.isfinite(half_count) or half_count <= 0:
        raise AnalogueCandidateError("matched-history support/smoothing differs")
    fields = ("market_regime", "prefix_quality_class", "trailing_liquidity_cell")
    required_outcome = {"query_id", "origin_cutoff", "completion_timestamp", "label", *fields}
    required_query = {"query_id", "query_cutoff", *fields}
    outcome_ids: set[str] = set()
    events: list[tuple[pd.Timestamp, str, str, tuple[str, str, str]]] = []
    for row in outcomes:
        if type(row) is not dict or not required_outcome.issubset(row):
            raise AnalogueCandidateError("matched-history row fields differ")
        query_id = row["query_id"]
        if type(query_id) is not str or not query_id or query_id in outcome_ids:
            raise AnalogueCandidateError("matched-history query IDs must be unique")
        outcome_ids.add(query_id)
        label = row["label"]
        completion = row["completion_timestamp"]
        if completion is None or label not in PRIMARY_CLASSES:
            continue
        origin = _timestamp(row["origin_cutoff"], "outcome origin")
        completed = _timestamp(completion, "outcome completion")
        if completed <= origin:
            raise AnalogueCandidateError("outcome completion must follow its origin")
        cells = tuple(row[field] for field in fields)
        if any(type(value) is not str or not value for value in cells):
            raise AnalogueCandidateError("outcome matching cells must be non-empty strings")
        events.append((completed, query_id, str(label), cells))

    query_ids: set[str] = set()
    ordered_queries: list[tuple[pd.Timestamp, str, tuple[str, str, str]]] = []
    for query in queries:
        if type(query) is not dict or not required_query.issubset(query):
            raise AnalogueCandidateError("matched-history query fields differ")
        query_id = query["query_id"]
        cells = tuple(query[field] for field in fields)
        if type(query_id) is not str or not query_id or query_id in query_ids:
            raise AnalogueCandidateError("matched-history forecast query IDs must be unique")
        if any(type(value) is not str or not value for value in cells):
            raise AnalogueCandidateError("query matching cells must be non-empty strings")
        query_ids.add(query_id)
        ordered_queries.append((_timestamp(query["query_cutoff"], "query cutoff"), query_id, cells))

    events.sort(key=lambda item: (item[0], item[1]))
    ordered_queries.sort(key=lambda item: (item[0], item[1]))
    unconditional = np.zeros(3, dtype=np.int64)
    exact: dict[tuple[str, str, str], np.ndarray] = {}
    regime_liquidity: dict[tuple[str, str], np.ndarray] = {}
    regime: dict[str, np.ndarray] = {}
    class_position = {name: index for index, name in enumerate(PRIMARY_CLASSES)}
    cursor = 0
    output: dict[str, MatchedProbability] = {}
    for cutoff, query_id, cells in ordered_queries:
        while cursor < len(events) and events[cursor][0] <= cutoff:
            _completion, event_id, label, event_cells = events[cursor]
            # A completed event necessarily originates before this cutoff; its
            # origin/completion ordering was validated above.
            position = class_position[label]
            unconditional[position] += 1
            exact.setdefault(event_cells, np.zeros(3, dtype=np.int64))[position] += 1
            regime_liquidity.setdefault(
                (event_cells[0], event_cells[2]), np.zeros(3, dtype=np.int64),
            )[position] += 1
            regime.setdefault(event_cells[0], np.zeros(3, dtype=np.int64))[position] += 1
            cursor += 1
        choices = (
            ("exact", exact.get(cells)),
            ("regime_and_liquidity", regime_liquidity.get((cells[0], cells[2]))),
            ("regime", regime.get(cells[0])),
        )
        selected_level = "unconditional"
        selected = unconditional
        for level, counts in choices:
            if counts is not None and int(counts.sum()) >= minimum_support:
                selected_level = level
                selected = counts
                break
        support = int(selected.sum())
        if int(unconditional.sum()) == 0:
            selected_level = "uniform_no_history"
        probabilities = (selected.astype(np.float64) + half_count) \
            / (support + half_count * len(PRIMARY_CLASSES))
        output[query_id] = MatchedProbability(
            tuple(map(float, probabilities)), selected_level, support,
            int(unconditional.sum()),
        )
    return output


def candidate_probabilities(
    component_probabilities: Mapping[str, Sequence[float]],
) -> tuple[float, float, float]:
    """Form the single frozen convex-mixture categorical distribution."""
    if set(component_probabilities) != set(COMPONENTS) \
            or set(MIXTURE_WEIGHTS) != set(COMPONENTS) \
            or not math.isclose(math.fsum(MIXTURE_WEIGHTS.values()), 1.0,
                                rel_tol=0.0, abs_tol=1e-15):
        raise AnalogueCandidateError("candidate component closure differs")
    output = np.zeros(3, dtype=np.float64)
    for component in COMPONENTS:
        output += MIXTURE_WEIGHTS[component] * _probability_vector(
            component_probabilities[component]
        )
    if not np.isfinite(output).all() or (output <= 0).any() \
            or abs(float(output.sum()) - 1.0) > 1e-15:
        raise AnalogueCandidateError("candidate probability result differs")
    return tuple(map(float, output))


def candidate_empirical_distribution(
    components: Mapping[str, tuple[Sequence[float], Sequence[float]]],
) -> tuple[np.ndarray, np.ndarray]:
    """Combine four empirical CDF components without quadratic materialization.

    Values retain deterministic component/input order. Each component's weights
    are normalized internally and then receive the frozen mixture mass. A value
    repeated across retrieval methods intentionally receives consensus mass.
    """
    if set(components) != set(COMPONENTS):
        raise AnalogueCandidateError("empirical component closure differs")
    values: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for component in COMPONENTS:
        raw_values, raw_weights = components[component]
        x = np.asarray(raw_values, dtype=np.float64)
        w = np.asarray(raw_weights, dtype=np.float64)
        if x.ndim != 1 or w.ndim != 1 or not len(x) or len(x) != len(w) \
                or not np.isfinite(x).all() or not np.isfinite(w).all() or (w <= 0).any():
            raise AnalogueCandidateError("empirical component values/weights differ")
        total = math.fsum(map(float, w))
        values.append(x.copy())
        weights.append(w * (MIXTURE_WEIGHTS[component] / total))
    combined_values = np.concatenate(values)
    combined_weights = np.concatenate(weights)
    if abs(math.fsum(map(float, combined_weights)) - 1.0) > 1e-15:
        raise AnalogueCandidateError("empirical mixture weight mass differs")
    return combined_values, combined_weights
