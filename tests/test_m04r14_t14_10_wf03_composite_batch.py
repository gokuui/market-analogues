from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from market_analogues.types import stable_hash

from experiments.m04r import m04r14_t14_10_wf03_composite_batch as subject


QUERY_ID = "a" * 24


def _row(query_id: str = QUERY_ID) -> dict:
    return {
        "episode_id": query_id, "case_id": f"case-{query_id}",
        "symbol": f"S{query_id[:2]}", "cutoff": "2014-03-31T00:00:00",
        "fold_id": "development",
        "fold_role": "implementation_diagnostic_no_threshold_tuning",
        "scored": True,
    }


def _preregistration() -> dict:
    return {
        "preregistration_digest": "b" * 64,
        "inputs": {"resident_content_digest": "c" * 64},
        "contracts": {"retrieval": {"digest": "d" * 64}},
        "inventory": {"months": 1},
    }


def _retrieval(query_id: str = QUERY_ID) -> dict:
    value = {
        "query_id": query_id, "proposal_result_digest": "e" * 64,
        "certificate": {
            "query_episode_id": query_id,
            "generation_id": subject.base.GENERATION_ID,
            "contract_digest": "d" * 64,
            "eligible_candidates": 100,
        },
        "matches": [],
    }
    value["semantic_digest"] = stable_hash(subject.kernel._case_semantics(value))
    return value


def _worker(query_id: str = QUERY_ID, *, swap: int = 0) -> dict:
    return {
        "queries": 1, "threads": 1, "proposal_seconds": 1.0,
        "elapsed_seconds": 2.0, "peak_rss_mb": 500.0,
        "swap_kib": swap, "resident_identity_digest": "f" * 64,
        "cases": [_retrieval(query_id)],
    }


def test_frozen_execution_uses_verified_twelve_by_one_shape() -> None:
    assert subject.PROCESS_COUNT == 12
    assert subject.THREADS_PER_PROCESS == 1
    assert subject.PROCESS_COUNT * subject.THREADS_PER_PROCESS == 12


def test_inventory_counts_scored_and_warmup() -> None:
    assert subject._inventory([{"cutoff": "a", "scored": True},
                               {"cutoff": "a", "scored": False},
                               {"cutoff": "b", "scored": True}]) == {
        "queries": 3, "scored_queries": 2, "warmup_queries": 1, "months": 2,
    }


def test_worker_result_rejects_swap(monkeypatch) -> None:
    monkeypatch.setattr(subject.kernel_verifier, "validate_certificate", lambda value: None)
    with pytest.raises(subject.CompositeBatchError):
        subject._validate_worker_result(
            _worker(swap=1), _row(), {"identity_digest": "f" * 64},
        )


def test_publish_is_create_only_and_round_trips(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(subject.kernel_verifier, "validate_certificate", lambda value: None)
    resident = {
        "identity_digest": "f" * 64, "content_digest": "c" * 64,
    }
    cases = tmp_path / "cases"
    value = subject._publish_worker_result(
        _worker(), _row(), resident, "attempt-0001", _preregistration(), cases,
    )
    assert subject._existing_case(
        cases / f"{QUERY_ID}.json", _row(), _preregistration(),
    ) == value
    with pytest.raises(subject.base.FeasibilityError):
        subject._publish_worker_result(
            _worker(), _row(), resident, "attempt-0001", _preregistration(), cases,
        )


def test_case_validation_rejects_query_binding_drift(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(subject.kernel_verifier, "validate_certificate", lambda value: None)
    resident = {
        "identity_digest": "f" * 64, "content_digest": "c" * 64,
    }
    value = subject._publish_worker_result(
        _worker(), _row(), resident, "attempt-0001", _preregistration(),
        tmp_path / "cases",
    )
    changed = {**value, "symbol": "DRIFT"}
    changed.pop("case_digest")
    changed = subject.base._sealed(changed, "case_digest")
    with pytest.raises(subject.CompositeBatchError):
        subject._validate_case(changed, _row(), _preregistration())


def test_manifest_restores_registry_order(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(subject, "_sha", lambda path: path.stem)
    rows = [_row("a" * 24), _row("b" * 24)]
    results = {
        "b" * 24: {"query_id": "b" * 24, "case_digest": "2"},
        "a" * 24: {"query_id": "a" * 24, "case_digest": "1"},
    }
    ordered, manifest = subject._manifest(rows, results, tmp_path)
    assert [value["query_id"] for value in ordered] == ["a" * 24, "b" * 24]
    assert [value["query_id"] for value in manifest] == ["a" * 24, "b" * 24]


def test_manifest_refuses_missing_query(tmp_path: Path) -> None:
    with pytest.raises(subject.CompositeBatchError):
        subject._manifest([_row()], {}, tmp_path)


def test_attempt_history_accepts_unfinished_restart_boundary(tmp_path: Path) -> None:
    root = tmp_path / "output"
    attempt = root / "attempts" / "attempt-0001"
    attempt.mkdir(parents=True)
    started = subject.base._sealed({
        "schema_version": "m04r14-wf03-composite-batch-attempt-v2",
        "status": "running", "attempt_id": "attempt-0001",
        "preregistration_digest": "b" * 64,
        "resident_content_digest": "c" * 64,
        "resident_identity_digest": "f" * 64,
        "receipts_reused_at_start": 0,
        "resource_observation": {"effective_cpus": 12},
        "created_at": "now",
    }, "attempt_digest")
    subject.base._atomic(attempt / "RUN_STARTED.json", started)
    history = subject._attempt_history(root, _preregistration())
    assert history["attempt-0001"]["_terminal_name"] is None
    assert subject._next_attempt(root).name == "attempt-0002"


def test_attempt_history_rejects_dual_terminal(tmp_path: Path) -> None:
    root = tmp_path / "output"
    attempt = root / "attempts" / "attempt-0001"
    attempt.mkdir(parents=True)
    for name in ("RUN_STARTED.json", "INTERRUPTED.json", "COMPLETE.json"):
        (attempt / name).write_text("{}")
    with pytest.raises(subject.CompositeBatchError):
        subject._attempt_history(root, _preregistration())


def test_rolling_scheduler_never_exceeds_frozen_width(monkeypatch, tmp_path: Path) -> None:
    state = {"outstanding": 0, "maximum": 0, "max_workers": None}

    class FakeFuture:
        def __init__(self, row):
            self.row = row

        def result(self):
            return _worker(self.row["episode_id"])

    class FakeExecutor:
        def __init__(self, max_workers, mp_context):
            state["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, *args):
            row = args[2][0]
            state["outstanding"] += 1
            state["maximum"] = max(state["maximum"], state["outstanding"])
            return FakeFuture(row)

    def fake_wait(active, return_when):
        future = next(iter(active))
        state["outstanding"] -= 1
        return {future}, set(active) - {future}

    def fake_publish(worker, row, resident, attempt_id, preregistration, cases_root):
        return {"query_id": row["episode_id"], "case_digest": "f" * 64,
                "worker_measurement": {"elapsed_seconds": 2.0}}

    monkeypatch.setattr(subject, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(subject, "wait", fake_wait)
    monkeypatch.setattr(subject, "_publish_worker_result", fake_publish)
    monkeypatch.setattr(subject, "_progress", lambda *args, **kwargs: None)
    rows = [_row(f"{index:024x}") for index in range(14)]
    results = {}
    completed = subject._run_pending(
        tmp_path, {"store_root": str(tmp_path)}, rows, results,
        tmp_path / "cases", "attempt-0001", _preregistration(), tmp_path, 0.0,
    )
    assert completed == 14
    assert len(results) == 14
    assert state == {"outstanding": 0, "maximum": 12, "max_workers": 12}
