from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.m04r import m04r14_t14_10_wf03_feasibility as subject
from market_analogues.packed_bound_search import BoundProposal, BoundProposalReport


def proposal() -> BoundProposalReport:
    candidate = BoundProposal(
        "0123456789abcdef01234567", "XYZ", 123, "A", 0.25,
        ("composite",), False,
    )
    return BoundProposalReport(
        "schema", "generation", "abcdef0123456789abcdef01", (candidate,),
        10, 9, 8, 1, {"composite": 1}, {"composite": 1},
        4_096, "forward", 1.5, 22.0, "candidate", "result", "contract", "input",
    )


def attempt_complete(attempt: Path, query_id: str) -> dict:
    subject._atomic(attempt / "RUN_STARTED.json", {"state": "running"})
    manifest = [{
        "path": "RUN_STARTED.json",
        "sha256": subject._sha(attempt / "RUN_STARTED.json"),
    }]
    value = subject._sealed({
        "schema_version": subject.ATTEMPT_SCHEMA, "status": "complete",
        "query_id": query_id, "leaf_manifest": manifest,
        "leaf_manifest_digest": subject.stable_hash(manifest),
    }, "complete_digest")
    subject._atomic(attempt / "COMPLETE.json", value)
    return value


def test_proposal_json_round_trip_is_exact() -> None:
    original = proposal()
    payload = subject._proposal_payload(original)
    restored = subject._proposal_report(payload)
    assert restored == original
    assert payload["candidates"][0]["lower_bound_hex"] == "0x1.0000000000000p-2"


def test_attempt_numbers_are_monotone_and_prior_attempts_survive(tmp_path: Path) -> None:
    first = subject._next_attempt(tmp_path)
    subject._atomic(first / "RUN_STARTED.json", {"state": "running"})
    second = subject._next_attempt(tmp_path)
    assert first.name == "attempt-0001"
    assert second.name == "attempt-0002"
    assert (first / "RUN_STARTED.json").is_file()


def test_atomic_is_create_only(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    subject._atomic(path, {"value": 1})
    with pytest.raises(subject.FeasibilityError, match="create-only"):
        subject._atomic(path, {"value": 2})
    assert subject._read(path) == {"value": 1}


def test_case_terminal_requires_seal_and_bound_attempt(tmp_path: Path) -> None:
    query_id = "abcdef0123456789abcdef01"
    attempt = tmp_path / "attempts" / "attempt-0001"
    attempt.mkdir(parents=True)
    attempt_value = attempt_complete(attempt, query_id)
    terminal = subject._sealed({
        "schema_version": subject.CASE_SCHEMA, "status": "complete",
        "query_id": query_id, "case_id": "case", "label": "early",
        "attempt_relative": "attempts/attempt-0001",
        "attempt_complete_sha256": subject._sha(attempt / "COMPLETE.json"),
        "attempt_complete_digest": attempt_value["complete_digest"], "created_at": "now",
    }, "complete_digest")
    subject._atomic(tmp_path / "COMPLETE.json", terminal)
    assert subject._case_complete_valid(tmp_path, query_id) == terminal


def test_case_terminal_rejects_wrong_attempt_hash(tmp_path: Path) -> None:
    query_id = "abcdef0123456789abcdef01"
    attempt = tmp_path / "attempts" / "attempt-0001"
    attempt.mkdir(parents=True)
    attempt_value = attempt_complete(attempt, query_id)
    terminal = subject._sealed({
        "schema_version": subject.CASE_SCHEMA, "status": "complete",
        "query_id": query_id, "case_id": "case", "label": "early",
        "attempt_relative": "attempts/attempt-0001",
        "attempt_complete_sha256": "0" * 64,
        "attempt_complete_digest": attempt_value["complete_digest"], "created_at": "now",
    }, "complete_digest")
    subject._atomic(tmp_path / "COMPLETE.json", terminal)
    with pytest.raises(subject.FeasibilityError, match="case terminal"):
        subject._case_complete_valid(tmp_path, query_id)


def test_seal_detects_mutation() -> None:
    value = subject._sealed({"status": "complete"}, "complete_digest")
    subject._validate_seal(value, "complete_digest")
    value["status"] = "failed"
    with pytest.raises(subject.FeasibilityError, match="complete_digest"):
        subject._validate_seal(value, "complete_digest")
