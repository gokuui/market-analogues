from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from market_analogues.adapters import DirectorySource
from market_analogues.config import DatasetSpec
from market_analogues.episodes import build_episode
from market_analogues.m04r_incident import (
    M04RIncidentResult,
    _loss_reason,
    trace_view_store_targets,
    write_m04r_incident,
)
from market_analogues.types import InstrumentKey, SearchQuery, stable_hash
from market_analogues.view_search import search_view_store
from market_analogues.view_store import build_view_store


def test_target_trace_has_exact_parity_with_production_candidate_route(
    directory_dataset: Path, tmp_path: Path,
) -> None:
    source = DirectorySource(DatasetSpec(
        "test", "directory", directory_dataset, "parquet", timestamp_column="date",
    ))
    quality = pd.DataFrame({
        "symbol": ["AAA", "BBB"], "tier": ["A", "B"], "issues": ["", ""],
    })
    root = tmp_path / "store"
    build_view_store(source, quality, root, lookbacks=(63,), stride=5, workers=2)
    query = build_episode(
        source, InstrumentKey("test", "AAA"), "2021-12-31", 63, "dense-v1",
    )
    request = SearchQuery(
        query.key, ("test",), ("A", "B"), 5, minimum_history_gap_bars=20,
    )
    production = search_view_store(
        query, request, root, candidate_pool=12, per_instrument_view=3,
    )
    targets = tuple({
        "episode_id": hit.episode_id,
        "symbol": hit.instrument.source_symbol,
    } for hit in production.hits)
    traces, metrics = trace_view_store_targets(
        query, request, root, targets, candidate_pool=12, per_instrument_view=3,
    )
    expected = {
        hit.episode_id: rank for rank, hit in enumerate(production.hits, 1)
    }
    assert metrics["candidate_pool_returned"] == len(production.hits)
    assert metrics["rows_considered"] == production.rows_considered
    assert metrics["local_candidates"] == production.local_candidates
    assert {episode_id: row["candidate_pool_rank"] for episode_id, row in traces.items()} == expected
    assert all(row["eligible_in_view_store"] for row in traces.values())
    assert all(row["local_admitted"] for row in traces.values())
    assert all(_loss_reason(row) == "in_candidate_pool" for row in traces.values())


def test_loss_reason_separates_local_and_fusion_failures() -> None:
    base = {
        "present_in_view_store": True,
        "eligible_in_view_store": True,
        "local_admitted": True,
        "candidate_pool_rank": 1,
    }
    assert _loss_reason(base) == "in_candidate_pool"
    assert _loss_reason({**base, "candidate_pool_rank": None}) == "fusion_pool_loss"
    assert _loss_reason({**base, "local_admitted": False}) == "local_cap_loss"
    assert _loss_reason({**base, "eligible_in_view_store": False}) == "ineligible_in_view_store"
    assert _loss_reason({**base, "present_in_view_store": False}) == "missing_from_view_store"


def test_incident_artifacts_are_deterministic_and_digest_bound(tmp_path: Path) -> None:
    metrics = {
        "schema_version": "m04r-incident-attribution-v1",
        "outcome_inputs_opened": 0,
        "real_forward_outcomes_accessed": False,
    }
    cases = ({
        "dataset_id": "test", "symbol": "AAA", "cutoff": "2020-01-01",
        "status": "analyzed", "reason_counts": {"in_candidate_pool": 1},
        "authority_neighbors": [{
            "authority_rank": 1, "symbol": "BBB", "cutoff": "2019-01-01",
            "global_fusion_rank": 2, "candidate_pool_rank": 1,
            "final_loss_reason": "in_candidate_pool",
        }],
    },)
    deterministic = {
        "schema_version": "m04r-incident-attribution-v1",
        "metrics": metrics, "failures": [], "cases": list(cases),
    }
    digest = stable_hash(deterministic)
    result = M04RIncidentResult(True, metrics, (), cases, digest)
    machine = tmp_path / "incident.json"
    html = tmp_path / "incident.html"
    write_m04r_incident(result, machine, html)
    first = machine.read_bytes()
    write_m04r_incident(result, machine, html)
    assert machine.read_bytes() == first
    assert json.loads(first)["baseline_digest"] == digest
    assert digest in html.read_text()
