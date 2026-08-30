from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.m04r import m04r14_step_listener as listener
from experiments.m04r import m04r14_untouched_candidate_contract as contract
from experiments.m04r import m04r14_untouched_failure_diagnostic as diagnostic


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))


def test_listener_waits_without_polluting_candidate_root(tmp_path: Path) -> None:
    state_root = tmp_path / "listener"; state_root.mkdir()
    assert listener.transition(tmp_path, state_root) == "waiting_for_candidate_terminal"
    assert list(state_root.iterdir()) == []


def test_listener_fails_closed_on_candidate_failure(tmp_path: Path) -> None:
    state_root = tmp_path / "listener"; state_root.mkdir()
    _write(tmp_path / contract.CANDIDATE_RELATIVE / "FAILED.json", {"error_type": "WorkerError"})
    assert listener.transition(tmp_path, state_root) == "blocked_before_authority"
    terminal = json.loads((state_root / "TERMINAL.json").read_text())
    assert terminal["authority_open_authorized"] is False
    assert terminal["reason"] == "candidate_failed"


def test_listener_fails_closed_on_semantic_failure(tmp_path: Path) -> None:
    state_root = tmp_path / "listener"; state_root.mkdir()
    _write(tmp_path / contract.CANDIDATE_RELATIVE / "RESULT.json", {
        "result_digest": "a" * 64, "semantic_passed": False, "performance_passed": False})
    assert listener.transition(tmp_path, state_root) == "blocked_before_authority"
    assert json.loads((state_root / "TERMINAL.json").read_text())["reason"] == "candidate_semantic_failure"


def test_diagnostic_correlation_handles_constant_and_linear_inputs() -> None:
    assert diagnostic._correlation([1, 1], [1, 2]) is None
    assert diagnostic._correlation([1, 2, 3], [2, 4, 6]) == 1.0


def test_batch_amendment_preserves_original_failure() -> None:
    value = json.loads((ROOT / "config/m04r14-untouched-batch-policy-amendment.json").read_text())
    assert value["original_performance_passed"] is False
    assert value["original_failure_preserved"] is True
    assert value["candidate_rerun_authorized"] is False
    assert value["batch_acceptance"]["candidate_wall_seconds_max"] == 2400.0
