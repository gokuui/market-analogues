"""Frozen descriptive aggregation for T14-09 analogue evidence cards."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .types import stable_hash


HORIZONS = (5, 10, 20, 40, 60, 126)
PREFIXES = (5, 10, 15, 20)
MEASURES = (
    "close_return", "benchmark_relative_return",
    "maximum_favorable_excursion", "maximum_adverse_excursion",
    "mfe_atr", "mae_atr",
)
BARRIER_LABELS = (
    "favorable_first", "adverse_first", "no_touch",
    "ambiguous_same_first_touch_bar", "censored",
)
MINIMUM_EFFECTIVE_SAMPLE = 10
WEIGHT_HALF_LIFE_RANKS = 10.0


class EvidenceCardError(ValueError):
    pass


def rank_weight(rank: int) -> float:
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= 20:
        raise EvidenceCardError("original match rank must be an integer in [1, 20]")
    return 2.0 ** (-(rank - 1) / WEIGHT_HALF_LIFE_RANKS)


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _linear_quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise EvidenceCardError("quantile requires values")
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _weighted_quantile(values: Sequence[tuple[float, float]], q: float) -> float:
    ordered = sorted(values, key=lambda item: item[0])
    total = math.fsum(weight for _, weight in ordered)
    threshold = q * total
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return ordered[-1][0]


def _measure_summary(
    rows: Sequence[Mapping[str, Any]], horizon: int, measure: str,
) -> dict[str, Any]:
    values: list[tuple[float, float]] = []
    for row in rows:
        value = _finite(row["outcomes_by_horizon"][str(horizon)].get(measure))
        if value is not None:
            values.append((value, rank_weight(int(row["match_rank"]))))
    if not values:
        return {"count": 0, "unweighted": None, "locked_weighted": None}
    raw = [value for value, _ in values]
    weights = [weight for _, weight in values]
    weight_sum = math.fsum(weights)
    unweighted = {
        "mean": math.fsum(raw) / len(raw),
        "median": _linear_quantile(raw, .5),
        "q25_linear": _linear_quantile(raw, .25),
        "q75_linear": _linear_quantile(raw, .75),
        "minimum": min(raw), "maximum": max(raw),
        "positive_count": sum(value > 0 for value in raw),
        "negative_count": sum(value < 0 for value in raw),
        "zero_count": sum(value == 0 for value in raw),
    }
    weighted = {
        "weight_sum": weight_sum,
        "effective_sample_size": weight_sum ** 2 / math.fsum(w * w for w in weights),
        "mean": math.fsum(value * weight for value, weight in values) / weight_sum,
        "median_inverted_cdf": _weighted_quantile(values, .5),
        "q25_inverted_cdf": _weighted_quantile(values, .25),
        "q75_inverted_cdf": _weighted_quantile(values, .75),
        "minimum": min(raw), "maximum": max(raw),
        "positive_weight": math.fsum(weight for value, weight in values if value > 0),
        "negative_weight": math.fsum(weight for value, weight in values if value < 0),
        "zero_weight": math.fsum(weight for value, weight in values if value == 0),
    }
    return {"count": len(values), "unweighted": unweighted, "locked_weighted": weighted}


def _eligible(row: Mapping[str, Any], horizon: int) -> bool:
    eligibility = row["eligibility_by_horizon"].get(str(horizon), {})
    outcome = row["outcomes_by_horizon"].get(str(horizon), {})
    return eligibility.get("eligible") is True and outcome.get("complete") is True


def _select(rows: Sequence[Mapping[str, Any]], query_symbol: str, prefix: int) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        if int(row["match_rank"]) > prefix or row["matched_symbol"] == query_symbol:
            continue
        selected.setdefault(str(row["matched_symbol"]), row)
    return sorted(selected.values(), key=lambda row: int(row["match_rank"]))


def _prefix_summary(rows: Sequence[Mapping[str, Any]], query_symbol: str, prefix: int) -> dict[str, Any]:
    effective = _select(rows, query_symbol, prefix)
    horizons: dict[str, Any] = {}
    for horizon in HORIZONS:
        eligible = [row for row in effective if _eligible(row, horizon)]
        weights = [rank_weight(int(row["match_rank"])) for row in eligible]
        weight_sum = math.fsum(weights)
        horizons[str(horizon)] = {
            "eligible_rows": len(eligible),
            "weighted_effective_sample_size": (
                weight_sum ** 2 / math.fsum(weight * weight for weight in weights)
                if weights else 0.0
            ),
            "measures": {
                measure: _measure_summary(eligible, horizon, measure)
                for measure in MEASURES
            },
        }
    primary_rows = [row for row in effective if _eligible(row, 20)]
    barrier_counts = {label: 0 for label in BARRIER_LABELS}
    barrier_weights = {label: 0.0 for label in BARRIER_LABELS}
    for row in primary_rows:
        label = str(row["outcomes_by_horizon"]["20"]["barrier_label"])
        if label not in barrier_counts:
            raise EvidenceCardError(f"unknown primary barrier label: {label}")
        barrier_counts[label] += 1
        barrier_weights[label] += rank_weight(int(row["match_rank"]))
    barrier_counts["resolved_directional_denominator"] = (
        barrier_counts["favorable_first"] + barrier_counts["adverse_first"]
    )
    barrier_weights["resolved_directional_denominator"] = (
        barrier_weights["favorable_first"] + barrier_weights["adverse_first"]
    )
    return {
        "neighbor_prefix": prefix,
        "effective_rows_before_horizon_eligibility": len(effective),
        "effective_match_ranks": [int(row["match_rank"]) for row in effective],
        "eligible_by_horizon": horizons,
        "primary_barrier_unweighted_counts": barrier_counts,
        "primary_barrier_locked_weighted_mass": barrier_weights,
    }


def build_evidence_card(
    rows: Sequence[Mapping[str, Any]], *, contract_digest: str,
    provenance: Mapping[str, str],
) -> dict[str, Any]:
    if len(rows) != 20:
        raise EvidenceCardError("an evidence card requires exactly 20 raw neighbours")
    ordered = sorted((dict(row) for row in rows), key=lambda row: int(row["match_rank"]))
    if [int(row["match_rank"]) for row in ordered] != list(range(1, 21)):
        raise EvidenceCardError("raw ranks must be exactly 1 through 20")
    query_ids = {str(row["query_episode_id"]) for row in ordered}
    query_cases = {str(row["query_case_id"]) for row in ordered}
    query_symbols = {str(row["query_symbol"]) for row in ordered}
    query_cutoffs = {str(row["query_cutoff"]) for row in ordered}
    if any(len(values) != 1 for values in (query_ids, query_cases, query_symbols, query_cutoffs)):
        raise EvidenceCardError("raw rows do not identify one query")
    query_symbol = next(iter(query_symbols))
    for row in ordered:
        if set(row.get("eligibility_by_horizon", {})) != {str(h) for h in HORIZONS} \
                or set(row.get("outcomes_by_horizon", {})) != {str(h) for h in HORIZONS}:
            raise EvidenceCardError("raw row horizon inventory differs")
    prefixes = {str(prefix): _prefix_summary(ordered, query_symbol, prefix) for prefix in PREFIXES}
    primary_eligible = prefixes["20"]["eligible_by_horizon"]["20"]["eligible_rows"]
    reasons = ["failed_calibration", "poor_data_quality"]
    if primary_eligible < MINIMUM_EFFECTIVE_SAMPLE:
        reasons.append("insufficient_effective_sample_size")
    raw = {
        "raw_links": 20,
        "raw_unique_episodes": len({row["matched_episode_id"] for row in ordered}),
        "raw_unique_symbols": len({row["matched_symbol"] for row in ordered}),
        "same_symbol_links": sum(row["matched_symbol"] == query_symbol for row in ordered),
    }
    displayed_rows: list[dict[str, Any]] = []
    for row in ordered:
        displayed = dict(row)
        visible: dict[str, Any] = {}
        for horizon in HORIZONS:
            key = str(horizon)
            eligibility = row["eligibility_by_horizon"][key]
            outcome = row["outcomes_by_horizon"][key]
            if eligibility.get("eligible") is True:
                visible[key] = dict(outcome)
            elif eligibility.get("reason") == "incomplete_horizon":
                visible[key] = {
                    "withheld": True, "reason": "incomplete_horizon",
                    "status": outcome.get("status"),
                    "barrier_label": outcome.get("barrier_label") if horizon == 20 else None,
                }
            else:
                visible[key] = {
                    "withheld": True,
                    "reason": str(eligibility.get("reason", "not_eligible")),
                }
        displayed["outcomes_by_horizon"] = visible
        displayed_rows.append(displayed)
    state = {
        "schema_version": "m04r14-t14-09-evidence-card-v1",
        "contract_digest": contract_digest,
        "query_case_id": next(iter(query_cases)),
        "query_episode_id": next(iter(query_ids)),
        "query_symbol": query_symbol,
        "query_cutoff": next(iter(query_cutoffs)),
        "raw_sample_counts": raw,
        "raw_analogue_rows": displayed_rows,
        "same_symbol_panel_ranks": [
            row["match_rank"] for row in ordered if row["matched_symbol"] == query_symbol
        ],
        "neighbor_sensitivity": prefixes,
        "primary_summary": prefixes["20"],
        "context_availability": {
            "event_cluster": "unavailable_no_point_in_time_cluster_labels",
            "sector": "unavailable_no_point_in_time_sector_history",
            "regime": "unavailable_no_point_in_time_regime_labels",
            "alternative_structure_ranking": "not_certified",
        },
        "predictive_claim_status": "abstain",
        "abstention_reasons": reasons,
        "allowed_claim": "historical_conditional_descriptive_evidence_only",
        "provenance": dict(sorted(provenance.items())),
    }
    return {**state, "card_digest": stable_hash(state)}
