"""Adapt verified registry/probability outputs into a prospective prediction batch."""
from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping

import numpy as np
import pandas as pd

from market_analogues.analogue_candidate import matched_causal_probabilities_batch
from market_analogues.prospective_batch import stable


class ProspectiveAdapterError(RuntimeError):
    pass


WEIGHTS = {
    "matched_causal_history": .4,
    "composite": .1,
    "price_only": .2,
    "recent_return_volatility": .3,
}
PROBABILITY_COLUMNS = (
    "favorable_probability", "adverse_probability", "no_touch_probability",
)
SCORE_COLUMNS = {"query_id", "query_cutoff", "lane", *PROBABILITY_COLUMNS}
REGISTRY_COLUMNS = {
    "episode_id", "symbol", "cutoff", "quality_tier", "liquidity_stratum",
    "selection_hash", "stock_prefix_digest", "benchmark_prefix_digest",
}
HISTORY_COLUMNS = {
    "query_id", "origin_cutoff", "completion_timestamp", "label",
    "market_regime", "quality_tier", "liquidity_stratum",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProspectiveAdapterError(message)


def _plain_probabilities(values: np.ndarray) -> list[float]:
    require(values.shape == (3,) and np.isfinite(values).all()
            and (values >= 0).all() and (values <= 1).all()
            and abs(float(values.sum()) - 1) <= 1e-12,
            "probability vector differs")
    return [float(value) for value in values]


def build_prospective_documents(
    *, registry: pd.DataFrame, lane_scores: pd.DataFrame,
    prior_history: pd.DataFrame, market_regime: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build the source, registry, and probability documents for one month."""
    require(set(registry.columns) == REGISTRY_COLUMNS, "registry column closure differs")
    require(set(lane_scores.columns) == SCORE_COLUMNS, "score column closure differs")
    require(set(prior_history.columns) == HISTORY_COLUMNS, "history column closure differs")
    require(len(registry) == 24 and not registry.episode_id.duplicated().any(),
            "24 unique registry rows required")
    cutoffs = set(registry.cutoff.astype(str))
    require(len(cutoffs) == 1, "one registry cutoff required")
    cutoff = pd.Timestamp(next(iter(cutoffs)))
    batch_id = cutoff.strftime("%Y-%m")
    require(all(pd.Timestamp(value) == cutoff for value in lane_scores.query_cutoff),
            "score cutoff differs")
    required_lanes = {"composite", "price_only", "recent_return_volatility"}
    require(set(lane_scores.lane) == required_lanes
            and not lane_scores.duplicated(["query_id", "lane"]).any()
            and len(lane_scores) == 72,
            "score lane closure differs")
    query_ids = registry.episode_id.astype(str).tolist()
    require(set(lane_scores.query_id.astype(str)) == set(query_ids),
            "score query closure differs")
    cell_counts = registry.groupby(
        ["quality_tier", "liquidity_stratum"], observed=True,
    ).size().to_dict()
    require(cell_counts == {
        (quality, liquidity): 4
        for quality in ("A", "B") for liquidity in ("low", "middle", "high")
    }, "registry cell balance differs")

    if len(prior_history):
        require(not prior_history.query_id.duplicated().any(), "history query IDs differ")
        origins = pd.to_datetime(prior_history.origin_cutoff)
        completions = pd.to_datetime(prior_history.completion_timestamp)
        require((origins < completions).all() and (completions <= cutoff).all(),
                "history is not fully causal for cutoff")
    records = [{
        "query_id": str(row.query_id), "origin_cutoff": row.origin_cutoff,
        "completion_timestamp": row.completion_timestamp, "label": str(row.label),
        "market_regime": str(row.market_regime),
        "prefix_quality_class": str(row.quality_tier),
        "trailing_liquidity_cell": str(row.liquidity_stratum),
    } for row in prior_history.itertuples(index=False)]
    forecasts = [{
        "query_id": str(row.episode_id), "query_cutoff": row.cutoff,
        "market_regime": market_regime,
        "prefix_quality_class": str(row.quality_tier),
        "trailing_liquidity_cell": str(row.liquidity_stratum),
    } for row in registry.itertuples(index=False)]
    matched = matched_causal_probabilities_batch(records, forecasts)
    indexed = {
        lane: lane_scores[lane_scores.lane == lane].set_index("query_id")
        for lane in required_lanes
    }
    rows = []
    for query_id in query_ids:
        components: dict[str, list[float]] = {
            "matched_causal_history": _plain_probabilities(
                np.asarray(matched[query_id].probabilities, dtype=np.float64),
            ),
        }
        for lane in required_lanes:
            components[lane] = _plain_probabilities(
                indexed[lane].loc[query_id, list(PROBABILITY_COLUMNS)].to_numpy(
                    dtype=np.float64,
                ),
            )
        candidate = sum(
            WEIGHTS[name] * np.asarray(components[name], dtype=np.float64)
            for name in WEIGHTS
        )
        probabilities = {
            "candidate": _plain_probabilities(candidate),
            "matched_causal_history": components["matched_causal_history"],
            "locked_composite": components["composite"],
        }
        provenance = {
            "query_id": query_id, "cutoff": cutoff.isoformat(),
            "component_weights": WEIGHTS, "component_probabilities": components,
            "matched_fallback_level": matched[query_id].fallback_level,
            "matched_support_rows": matched[query_id].support_rows,
        }
        rows.append({
            "query_id": query_id, "probabilities": probabilities,
            "provenance_digest": stable(provenance),
        })

    registry_rows = [{
        "query_id": str(row.episode_id), "symbol": str(row.symbol),
        "quality_tier": str(row.quality_tier),
        "liquidity_stratum": str(row.liquidity_stratum),
        "selection_hash": str(row.selection_hash),
        "stock_prefix_digest": str(row.stock_prefix_digest),
    } for row in registry.itertuples(index=False)]
    query_digest = stable(query_ids)
    benchmark_digests = set(registry.benchmark_prefix_digest.astype(str))
    require(len(benchmark_digests) == 1, "benchmark prefix identity differs")
    source_manifest = [{
        "query_id": row["query_id"], "symbol": row["symbol"],
        "stock_prefix_digest": row["stock_prefix_digest"],
    } for row in registry_rows]
    source = {
        "schema_version": "prospective-source-lock-v1", "batch_id": batch_id,
        "cutoff": cutoff.date().isoformat(),
        "maximum_source_timestamp": cutoff.date().isoformat(),
        "stock_prefix_manifest_digest": stable(source_manifest),
        "benchmark_prefix_digest": next(iter(benchmark_digests)),
        "source_values_after_cutoff_opened": False,
    }
    registry_document = {
        "schema_version": "prospective-query-registry-v1", "batch_id": batch_id,
        "cutoff": cutoff.date().isoformat(), "queries": registry_rows,
        "query_digest": query_digest, "selection_used_outcomes": False,
    }
    prediction_document = {
        "schema_version": "prospective-probability-predictions-v1",
        "batch_id": batch_id, "cutoff": cutoff.date().isoformat(), "rows": rows,
        "query_digest": query_digest, "query_outcomes_opened": False,
        "source_values_after_cutoff_opened": False,
    }
    return source, registry_document, prediction_document
