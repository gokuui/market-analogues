from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest

from market_analogues.prospective_batch import (
    ProspectiveBatchError,
    decide_next_batch,
    guarded_outcome_load,
    seal_prediction_batch,
    stable,
    validate_prediction_batch,
)


CONTRACT = "1" * 64


def documents(batch: str = "2026-09", cutoff: str = "2026-09-30") -> tuple[dict, dict, dict]:
    source = {
        "schema_version": "prospective-source-lock-v1", "batch_id": batch,
        "cutoff": cutoff, "maximum_source_timestamp": cutoff,
        "stock_prefix_manifest_digest": "2" * 64,
        "benchmark_prefix_digest": "3" * 64,
        "source_values_after_cutoff_opened": False,
    }
    queries = [{
        "query_id": "q1", "symbol": "AAA", "quality_tier": "A",
        "liquidity_stratum": "high", "selection_hash": "4" * 64,
        "stock_prefix_digest": "5" * 64,
    }, {
        "query_id": "q2", "symbol": "BBB", "quality_tier": "B",
        "liquidity_stratum": "low", "selection_hash": "6" * 64,
        "stock_prefix_digest": "7" * 64,
    }]
    ids = [row["query_id"] for row in queries]
    registry = {
        "schema_version": "prospective-query-registry-v1", "batch_id": batch,
        "cutoff": cutoff, "queries": queries, "query_digest": stable(ids),
        "selection_used_outcomes": False,
    }
    rows = [{
        "query_id": query_id,
        "probabilities": {
            "candidate": [.5, .3, .2],
            "matched_causal_history": [.4, .4, .2],
            "locked_composite": [.6, .2, .2],
        },
        "provenance_digest": "8" * 64,
    } for query_id in ids]
    predictions = {
        "schema_version": "prospective-probability-predictions-v1",
        "batch_id": batch, "cutoff": cutoff, "rows": rows,
        "query_digest": stable(ids), "query_outcomes_opened": False,
        "source_values_after_cutoff_opened": False,
    }
    return source, registry, predictions


def test_listener_waits_runs_and_requires_exact_prefix() -> None:
    sessions = pd.bdate_range("2026-09-01", "2026-11-30")
    freeze = pd.Timestamp("2026-09-11")
    waiting = decide_next_batch(
        freeze=freeze, as_of=pd.Timestamp("2026-09-20"),
        benchmark_sessions=sessions[:14], completed=[], eligible_stock_files={},
        minimum_stocks=1000,
    )
    assert waiting.action == "wait_for_completed_month"
    source_wait = decide_next_batch(
        freeze=freeze, as_of=pd.Timestamp("2026-10-02"),
        benchmark_sessions=sessions, completed=[], eligible_stock_files={"2026-09-30": 999},
        minimum_stocks=1000,
    )
    assert source_wait.action == "wait_for_stock_source"
    ready = decide_next_batch(
        freeze=freeze, as_of=pd.Timestamp("2026-10-02"),
        benchmark_sessions=sessions, completed=[], eligible_stock_files={"2026-09-30": 1000},
        minimum_stocks=1000,
    )
    assert (ready.action, ready.batch_id, ready.cutoff) == (
        "run_prediction_batch", "2026-09", "2026-09-30",
    )
    with pytest.raises(ProspectiveBatchError, match="exact chronological prefix"):
        decide_next_batch(
            freeze=freeze, as_of=pd.Timestamp("2026-11-02"),
            benchmark_sessions=sessions,
            completed=[{"batch_id": "2026-10", "cutoff": "2026-10-30"}],
            eligible_stock_files={}, minimum_stocks=1000,
        )


def test_listener_collection_complete_requires_exact_twelve_month_prefix() -> None:
    sessions = pd.bdate_range("2026-09-01", "2027-10-31")
    cutoffs = []
    for period, group in pd.Series(sessions, index=sessions.to_period("M")).groupby(level=0):
        if period >= pd.Period("2026-09", freq="M"):
            cutoff = pd.Timestamp(group.max())
            cutoffs.append({"batch_id": cutoff.strftime("%Y-%m"),
                            "cutoff": cutoff.date().isoformat()})
    decision = decide_next_batch(
        freeze=pd.Timestamp("2026-09-11"), as_of=pd.Timestamp("2027-11-01"),
        benchmark_sessions=sessions, completed=cutoffs[:12], eligible_stock_files={},
        minimum_stocks=1000,
    )
    assert decision.action == "collection_complete"


def test_invalid_probabilities_and_post_cutoff_source_are_refused(tmp_path: Path) -> None:
    source, registry, predictions = documents()
    source["maximum_source_timestamp"] = "2026-10-01"
    with pytest.raises(ProspectiveBatchError, match="crosses prediction cutoff"):
        seal_prediction_batch(
            tmp_path, contract_digest=CONTRACT, source_lock=source,
            registry=registry, predictions=predictions, created_at="now",
        )
    source["maximum_source_timestamp"] = source["cutoff"]
    predictions["rows"][0]["probabilities"]["candidate"] = [.5, .5, .5]
    with pytest.raises(ProspectiveBatchError, match="invalid probabilities"):
        seal_prediction_batch(
            tmp_path, contract_digest=CONTRACT, source_lock=source,
            registry=registry, predictions=predictions, created_at="now",
        )


def test_interrupted_prediction_batch_resumes_identically(tmp_path: Path) -> None:
    source, registry, predictions = documents()
    with pytest.raises(ProspectiveBatchError, match="injected interruption"):
        seal_prediction_batch(
            tmp_path, contract_digest=CONTRACT, source_lock=source,
            registry=registry, predictions=predictions,
            created_at="2026-10-01T00:00:00Z", interrupt_after_prerequisites=True,
        )
    seal = seal_prediction_batch(
        tmp_path, contract_digest=CONTRACT, source_lock=source,
        registry=registry, predictions=predictions, created_at="2026-10-01T00:00:00Z",
    )
    assert seal_prediction_batch(
        tmp_path, contract_digest=CONTRACT, source_lock=source,
        registry=registry, predictions=predictions, created_at="ignored-on-resume",
    ) == seal
    assert validate_prediction_batch(tmp_path / "batch-2026-09") == seal


def test_interruption_after_seal_resumes_with_original_timestamp(tmp_path: Path) -> None:
    source, registry, predictions = documents()
    with pytest.raises(ProspectiveBatchError, match="after prediction seal"):
        seal_prediction_batch(
            tmp_path, contract_digest=CONTRACT, source_lock=source,
            registry=registry, predictions=predictions, created_at="original",
            interrupt_after_seal=True,
        )
    seal = seal_prediction_batch(
        tmp_path, contract_digest=CONTRACT, source_lock=source,
        registry=registry, predictions=predictions, created_at="new-time",
    )
    assert seal["created_at"] == "original"


def test_conflicting_resume_and_tampering_are_rejected(tmp_path: Path) -> None:
    source, registry, predictions = documents()
    with pytest.raises(ProspectiveBatchError):
        seal_prediction_batch(
            tmp_path, contract_digest=CONTRACT, source_lock=source,
            registry=registry, predictions=predictions,
            created_at="now", interrupt_after_prerequisites=True,
        )
    changed = copy.deepcopy(predictions)
    changed["rows"][0]["probabilities"]["candidate"] = [.4, .4, .2]
    with pytest.raises(ProspectiveBatchError, match="restart prerequisite differs"):
        seal_prediction_batch(
            tmp_path, contract_digest=CONTRACT, source_lock=source,
            registry=registry, predictions=changed, created_at="later",
        )
    seal_prediction_batch(
        tmp_path, contract_digest=CONTRACT, source_lock=source,
        registry=registry, predictions=predictions, created_at="now",
    )
    path = tmp_path / "batch-2026-09" / "PREDICTIONS.json"
    path.write_text(path.read_text().replace("0.5", "0.4", 1))
    with pytest.raises(ProspectiveBatchError):
        validate_prediction_batch(tmp_path / "batch-2026-09")


def test_completed_batch_refuses_conflicting_requested_inputs(tmp_path: Path) -> None:
    source, registry, predictions = documents()
    seal_prediction_batch(
        tmp_path, contract_digest=CONTRACT, source_lock=source,
        registry=registry, predictions=predictions, created_at="now",
    )
    changed = copy.deepcopy(predictions)
    changed["rows"][0]["probabilities"]["candidate"] = [.4, .4, .2]
    with pytest.raises(ProspectiveBatchError, match="completed batch differs"):
        seal_prediction_batch(
            tmp_path, contract_digest=CONTRACT, source_lock=source,
            registry=registry, predictions=changed, created_at="later",
        )


def test_outcome_loader_is_unreachable_until_seal_and_maturity(tmp_path: Path) -> None:
    calls = []
    loader = lambda ids: calls.append(list(ids)) or {"ids": list(ids)}
    sessions = pd.bdate_range("2026-09-01", periods=100)
    with pytest.raises(ProspectiveBatchError):
        guarded_outcome_load(
            tmp_path / "missing", benchmark_sessions=sessions,
            source_as_of=sessions[-1], wall_clock=sessions[-1], horizon_sessions=60,
            loader=loader,
        )
    assert calls == []
    source, registry, predictions = documents()
    seal_prediction_batch(
        tmp_path, contract_digest=CONTRACT, source_lock=source,
        registry=registry, predictions=predictions, created_at="now",
    )
    batch = tmp_path / "batch-2026-09"
    with pytest.raises(ProspectiveBatchError, match="before maturity"):
        guarded_outcome_load(
            batch, benchmark_sessions=sessions,
            source_as_of=pd.Timestamp("2026-10-01"), wall_clock=pd.Timestamp("2026-12-01"),
            horizon_sessions=60, loader=loader,
        )
    assert calls == []
    result = guarded_outcome_load(
        batch, benchmark_sessions=sessions, source_as_of=sessions[-1],
        wall_clock=sessions[-1] + pd.Timedelta(days=1), horizon_sessions=60,
        loader=loader,
    )
    assert result == {"ids": ["q1", "q2"]}
    assert calls == [["q1", "q2"]]


def test_future_outcome_mutation_cannot_change_prediction_seal(tmp_path: Path) -> None:
    source, registry, predictions = documents()
    seal = seal_prediction_batch(
        tmp_path, contract_digest=CONTRACT, source_lock=source,
        registry=registry, predictions=predictions, created_at="now",
    )
    future = {"q1": "favorable_first", "q2": "adverse_first"}
    digest_before = seal["result_digest"]
    future["q1"] = "no_touch"
    assert validate_prediction_batch(tmp_path / "batch-2026-09")["result_digest"] == digest_before
