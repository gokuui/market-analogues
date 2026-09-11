from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from market_analogues.prospective_adapter import (
    ProspectiveAdapterError,
    build_prospective_documents,
)
from market_analogues.prospective_batch import seal_prediction_batch, validate_prediction_batch


def fixtures() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    registry_rows = []
    index = 0
    for quality in ("A", "B"):
        for liquidity in ("low", "middle", "high"):
            for _ in range(4):
                registry_rows.append({
                    "episode_id": f"q{index:02d}", "symbol": f"S{index:02d}",
                    "cutoff": "2026-09-30", "quality_tier": quality,
                    "liquidity_stratum": liquidity, "selection_hash": f"{index:064x}",
                    "stock_prefix_digest": f"{index + 100:064x}",
                    "benchmark_prefix_digest": "b" * 64,
                })
                index += 1
    scores = []
    for row in registry_rows:
        for lane, probabilities in {
            "composite": (.6, .2, .2), "price_only": (.5, .3, .2),
            "recent_return_volatility": (.4, .4, .2),
        }.items():
            scores.append({
                "query_id": row["episode_id"], "query_cutoff": row["cutoff"],
                "lane": lane, "favorable_probability": probabilities[0],
                "adverse_probability": probabilities[1],
                "no_touch_probability": probabilities[2],
            })
    history = pd.DataFrame([{
        "query_id": f"old-{i:02d}", "origin_cutoff": "2026-01-02",
        "completion_timestamp": "2026-02-02",
        "label": ("favorable_first", "adverse_first", "no_touch")[i % 3],
        "market_regime": "trend=up|volatility=low", "quality_tier": "A",
        "liquidity_stratum": "low",
    } for i in range(36)])
    return pd.DataFrame(registry_rows), pd.DataFrame(scores), history


def test_adapter_builds_balanced_valid_sealable_batch(tmp_path: Path) -> None:
    registry, scores, history = fixtures()
    documents = build_prospective_documents(
        registry=registry, lane_scores=scores, prior_history=history,
        market_regime="trend=up|volatility=low",
    )
    source, query_registry, predictions = documents
    assert query_registry["selection_used_outcomes"] is False
    assert predictions["query_outcomes_opened"] is False
    assert len(predictions["rows"]) == 24
    first = predictions["rows"][0]["probabilities"]
    expected = [
        .4 * first["matched_causal_history"][i] + .1 * (.6, .2, .2)[i]
        + .2 * (.5, .3, .2)[i] + .3 * (.4, .4, .2)[i]
        for i in range(3)
    ]
    assert first["candidate"] == pytest.approx(expected, abs=1e-15)
    seal_prediction_batch(
        tmp_path, contract_digest="1" * 64, source_lock=source,
        registry=query_registry, predictions=predictions, created_at="now",
    )
    assert validate_prediction_batch(tmp_path / "batch-2026-09")["query_count"] == 24


def test_adapter_refuses_noncausal_history_and_outcome_bearing_scores() -> None:
    registry, scores, history = fixtures()
    history.loc[0, "completion_timestamp"] = "2026-10-01"
    with pytest.raises(ProspectiveAdapterError, match="fully causal"):
        build_prospective_documents(
            registry=registry, lane_scores=scores, prior_history=history,
            market_regime="trend=up|volatility=low",
        )
    registry, scores, history = fixtures()
    scores["route_status"] = "favorable_first"
    with pytest.raises(ProspectiveAdapterError, match="score column closure"):
        build_prospective_documents(
            registry=registry, lane_scores=scores, prior_history=history,
            market_regime="trend=up|volatility=low",
        )


def test_registry_requires_exact_cell_balance() -> None:
    registry, scores, history = fixtures()
    registry.loc[0, "liquidity_stratum"] = "high"
    with pytest.raises(ProspectiveAdapterError, match="cell balance"):
        build_prospective_documents(
            registry=registry, lane_scores=scores, prior_history=history,
            market_regime="trend=up|volatility=low",
        )
