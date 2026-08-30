from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.m04r import m04r14_untouched_candidate as candidate
from experiments.m04r import m04r14_untouched_candidate_contract as contract


def _cases(count: int = 12) -> list[dict[str, object]]:
    return [{"case_id": f"case-{index:02d}", "episode_id": f"{index:024d}",
             "active_source_universe": (index * 7919) % 10000}
            for index in range(count)]


def test_contract_freezes_qualified_p8_lane_and_claim_boundary() -> None:
    assert contract.CONTROLS["processes"] == 8
    assert contract.CONTROLS["initial_frontier_rows"] == 1000
    assert contract.CONTROLS["maximum_frontier_rows"] == 16384
    assert contract.CLAIMS == {
        "authority_accessed": False, "real_forward_outcomes_accessed": False,
        "candidate_first": True, "cases": 72, "symbols": 36, "top_k": 20,
        "production_promotion_authorized": False,
    }
    assert contract.CONTRACT_DIGEST == contract.digest(contract.CONTRACT_STATE)


def test_balanced_groups_are_complete_disjoint_and_deterministic() -> None:
    cases = _cases()
    first = candidate.balanced_groups(cases, 8)
    second = candidate.balanced_groups(list(reversed(cases)), 8)
    assert first == second
    flattened = [row["case_id"] for group in first for row in group]
    assert sorted(flattened) == sorted(row["case_id"] for row in cases)
    assert len(flattened) == len(set(flattened)) == 12
    assert all(group for group in first)


@pytest.mark.parametrize("processes", [0, 13])
def test_balanced_groups_reject_invalid_process_count(processes: int) -> None:
    with pytest.raises(candidate.CandidateError, match="process count"):
        candidate.balanced_groups(_cases(), processes)


def test_atomic_publication_is_create_only(tmp_path: Path) -> None:
    path = tmp_path / "value.json"; candidate._atomic(path, {"value": 1})
    assert json.loads(path.read_text()) == {"value": 1}
    with pytest.raises(candidate.CandidateError, match="create-only"):
        candidate._atomic(path, {"value": 2})


def test_failed_started_root_cannot_be_reused(tmp_path: Path) -> None:
    root = tmp_path / contract.CANDIDATE_RELATIVE
    root.mkdir(parents=True)
    with pytest.raises(candidate.CandidateError, match="retry/resume"):
        # The launch validation is deliberately bypassed in this unit test so
        # the terminal root policy itself is isolated.
        original = candidate.validate_launch
        try:
            candidate.validate_launch = lambda repository: ({}, {}, {})  # type: ignore[assignment]
            candidate.execute(tmp_path)
        finally:
            candidate.validate_launch = original


def test_preopen_verifier_does_not_import_candidate_producer() -> None:
    path = Path(__file__).parents[1] / "experiments/m04r/verify_m04r14_untouched_candidate.py"
    source = path.read_text()
    assert "import m04r14_untouched_candidate as" not in source
    assert "from experiments.m04r.m04r14_untouched_candidate import" not in source
