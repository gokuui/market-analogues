"""Independent scalar oracle for T14-09 descriptive evidence-card aggregation."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from market_analogues.types import stable_hash


HORIZONS = (5, 10, 20, 40, 60, 126)
PREFIXES = (5, 10, 15, 20)
MEASURES = (
    "close_return", "benchmark_relative_return",
    "maximum_favorable_excursion", "maximum_adverse_excursion", "mfe_atr", "mae_atr",
)
LABELS = (
    "favorable_first", "adverse_first", "no_touch",
    "ambiguous_same_first_touch_bar", "censored",
)


def _weight(rank: int) -> float:
    return math.pow(2.0, -(rank - 1) / 10.0)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _linear(values: list[float], probability: float) -> float:
    values = sorted(values)
    location = probability * (len(values) - 1)
    left = int(location)
    right = min(left + 1, len(values) - 1)
    fraction = location - left
    return values[left] * (1.0 - fraction) + values[right] * fraction


def _inverse(values: list[tuple[float, float]], probability: float) -> float:
    values = sorted(values)
    target = probability * math.fsum(weight for _, weight in values)
    running = 0.0
    for value, weight in values:
        running = math.fsum((running, weight))
        if running >= target:
            return value
    return values[-1][0]


def _measure(rows: Sequence[Mapping[str, Any]], horizon: int, name: str) -> dict[str, Any]:
    pairs = []
    for row in rows:
        value = _number(row["outcomes_by_horizon"][str(horizon)].get(name))
        if value is not None:
            pairs.append((value, _weight(int(row["match_rank"]))))
    if not pairs:
        return {"count": 0, "unweighted": None, "locked_weighted": None}
    values = [value for value, _ in pairs]
    weights = [weight for _, weight in pairs]
    total = math.fsum(weights)
    return {
        "count": len(pairs),
        "unweighted": {
            "mean": math.fsum(values) / len(values),
            "median": _linear(values, .5),
            "q25_linear": _linear(values, .25),
            "q75_linear": _linear(values, .75),
            "minimum": min(values), "maximum": max(values),
            "positive_count": len([value for value in values if value > 0]),
            "negative_count": len([value for value in values if value < 0]),
            "zero_count": len([value for value in values if value == 0]),
        },
        "locked_weighted": {
            "weight_sum": total,
            "effective_sample_size": total * total / math.fsum(w * w for w in weights),
            "mean": math.fsum(value * weight for value, weight in pairs) / total,
            "median_inverted_cdf": _inverse(pairs, .5),
            "q25_inverted_cdf": _inverse(pairs, .25),
            "q75_inverted_cdf": _inverse(pairs, .75),
            "minimum": min(values), "maximum": max(values),
            "positive_weight": math.fsum(weight for value, weight in pairs if value > 0),
            "negative_weight": math.fsum(weight for value, weight in pairs if value < 0),
            "zero_weight": math.fsum(weight for value, weight in pairs if value == 0),
        },
    }


def _is_eligible(row: Mapping[str, Any], horizon: int) -> bool:
    key = str(horizon)
    return row["eligibility_by_horizon"][key].get("eligible") is True \
        and row["outcomes_by_horizon"][key].get("complete") is True


def _panel(rows: Sequence[Mapping[str, Any]], query_symbol: str, prefix: int) -> dict[str, Any]:
    chosen = []
    symbols: set[str] = set()
    for row in rows:
        symbol = str(row["matched_symbol"])
        if int(row["match_rank"]) <= prefix and symbol != query_symbol and symbol not in symbols:
            symbols.add(symbol)
            chosen.append(row)
    horizon_panels = {}
    for horizon in HORIZONS:
        eligible = [row for row in chosen if _is_eligible(row, horizon)]
        weights = [_weight(int(row["match_rank"])) for row in eligible]
        total = math.fsum(weights)
        horizon_panels[str(horizon)] = {
            "eligible_rows": len(eligible),
            "weighted_effective_sample_size": (
                total * total / math.fsum(weight * weight for weight in weights)
                if weights else 0.0
            ),
            "measures": {name: _measure(eligible, horizon, name) for name in MEASURES},
        }
    counts = {label: 0 for label in LABELS}
    masses = {label: 0.0 for label in LABELS}
    for row in chosen:
        if not _is_eligible(row, 20):
            continue
        label = str(row["outcomes_by_horizon"]["20"]["barrier_label"])
        counts[label] += 1
        masses[label] += _weight(int(row["match_rank"]))
    counts["resolved_directional_denominator"] = counts["favorable_first"] + counts["adverse_first"]
    masses["resolved_directional_denominator"] = masses["favorable_first"] + masses["adverse_first"]
    return {
        "neighbor_prefix": prefix,
        "effective_rows_before_horizon_eligibility": len(chosen),
        "effective_match_ranks": [int(row["match_rank"]) for row in chosen],
        "eligible_by_horizon": horizon_panels,
        "primary_barrier_unweighted_counts": counts,
        "primary_barrier_locked_weighted_mass": masses,
    }


def reference_card(
    input_rows: Sequence[Mapping[str, Any]], *, contract_digest: str,
    provenance: Mapping[str, str],
) -> dict[str, Any]:
    rows = sorted((dict(row) for row in input_rows), key=lambda row: int(row["match_rank"]))
    if len(rows) != 20 or [int(row["match_rank"]) for row in rows] != list(range(1, 21)):
        raise ValueError("oracle raw inventory differs")
    query_symbol = str(rows[0]["query_symbol"])
    prefixes = {str(prefix): _panel(rows, query_symbol, prefix) for prefix in PREFIXES}
    visible_rows = []
    for source in rows:
        row = dict(source)
        visible = {}
        for horizon in HORIZONS:
            key = str(horizon)
            eligibility = source["eligibility_by_horizon"][key]
            outcome = source["outcomes_by_horizon"][key]
            if eligibility.get("eligible") is True:
                visible[key] = dict(outcome)
            elif eligibility.get("reason") == "incomplete_horizon":
                visible[key] = {
                    "withheld": True, "reason": "incomplete_horizon",
                    "status": outcome.get("status"),
                    "barrier_label": outcome.get("barrier_label") if horizon == 20 else None,
                }
            else:
                visible[key] = {"withheld": True, "reason": str(eligibility.get("reason", "not_eligible"))}
        row["outcomes_by_horizon"] = visible
        visible_rows.append(row)
    primary_count = prefixes["20"]["eligible_by_horizon"]["20"]["eligible_rows"]
    reasons = ["failed_calibration", "poor_data_quality"]
    if primary_count < 10:
        reasons.append("insufficient_effective_sample_size")
    state = {
        "schema_version": "m04r14-t14-09-evidence-card-v1",
        "contract_digest": contract_digest,
        "query_case_id": str(rows[0]["query_case_id"]),
        "query_episode_id": str(rows[0]["query_episode_id"]),
        "query_symbol": query_symbol,
        "query_cutoff": str(rows[0]["query_cutoff"]),
        "raw_sample_counts": {
            "raw_links": 20,
            "raw_unique_episodes": len({row["matched_episode_id"] for row in rows}),
            "raw_unique_symbols": len({row["matched_symbol"] for row in rows}),
            "same_symbol_links": sum(row["matched_symbol"] == query_symbol for row in rows),
        },
        "raw_analogue_rows": visible_rows,
        "same_symbol_panel_ranks": [row["match_rank"] for row in rows if row["matched_symbol"] == query_symbol],
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
