from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import m04r14_t14_10_wf03_composite_listener as subject


def test_terminal_ready_requires_both_complete_artifacts(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / subject.producer.OUTPUT_RELATIVE
    root.mkdir(parents=True)
    ready, observation = subject._terminal_ready(tmp_path)
    assert ready is False
    assert observation == {
        "reason": "producer terminal artifacts are absent",
        "progress_exists": False, "result_exists": False,
    }


def test_terminal_ready_rejects_running_progress(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / subject.producer.OUTPUT_RELATIVE
    root.mkdir(parents=True)
    progress = {
        "schema_version": "m04r14-wf03-composite-batch-progress-v1",
        "status": "running", "completed_queries": 100, "remaining_queries": 3836,
    }
    result = subject.base._sealed({
        "schema_version": "m04r14-t14-10-wf03-composite-batch-result-v1",
        "status": "complete", "passed": True, "queries": 3936,
        "independent_verification_authorized": True,
    })
    subject.base._atomic(root / "PROGRESS.json", progress)
    subject.base._atomic(root / "RESULT.json", result)
    ready, observation = subject._terminal_ready(tmp_path)
    assert ready is False
    assert observation["progress_status"] == "running"


def test_terminal_ready_accepts_sealed_complete_state(tmp_path: Path) -> None:
    root = tmp_path / subject.producer.OUTPUT_RELATIVE
    root.mkdir(parents=True)
    progress = {
        "schema_version": "m04r14-wf03-composite-batch-progress-v1",
        "status": "complete", "completed_queries": 3936, "remaining_queries": 0,
    }
    result = subject.base._sealed({
        "schema_version": "m04r14-t14-10-wf03-composite-batch-result-v1",
        "status": "complete", "passed": True, "queries": 3936,
        "independent_verification_authorized": True,
    })
    subject.base._atomic(root / "PROGRESS.json", progress)
    subject.base._atomic(root / "RESULT.json", result)
    ready, observation = subject._terminal_ready(tmp_path)
    assert ready is True
    assert observation["producer_result_digest"] == result["result_digest"]
