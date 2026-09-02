from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r.m04r14_t14_10_wf03b_dtw_component_ladder import (
    DtwComponentLadderError,
    FRONTIER_LEVELS,
    PROPOSAL_QUOTA,
    _proposal_equal,
    _terminal,
)


def _report(**changes):
    state = {
        "contract_digest": "contract",
        "packed_generation_id": "packed",
        "dtw_generation_id": "dtw",
        "query_episode_id": "query",
        "input_digest": "input",
        "candidates": ("a", "b"),
        "rows_scanned": 9,
        "eligible_rows": 7,
        "eligible_main_rows": 6,
        "eligible_overflow_rows": 1,
        "quota": 2,
        "candidate_digest": "candidates",
        "result_digest": "result",
        "block_rows": 4,
        "block_order": "forward",
        "kernel_threads": 1,
        "elapsed_seconds": 1.0,
        "peak_rss_mb": 2.0,
    }
    state.update(changes)
    return SimpleNamespace(**state)


def test_ladder_levels_and_proposal_parity_are_frozen() -> None:
    assert FRONTIER_LEVELS == (16_384, 32_768, 65_536, 131_072, 262_144)
    assert PROPOSAL_QUOTA == 262_145
    assert _proposal_equal(
        _report(),
        _report(block_rows=4097, block_order="reverse", elapsed_seconds=8.0),
    )
    assert not _proposal_equal(_report(), _report(candidates=("a", "c")))


def test_case_terminal_revalidates_referenced_attempt(tmp_path: Path) -> None:
    case = tmp_path / "case"
    attempt = case / "attempts" / "attempt-0001"
    attempt.mkdir(parents=True)
    leaf = base._sealed({"status": "complete"}, "complete_digest")
    base._atomic(attempt / "COMPLETE.json", leaf)
    terminal = base._sealed({
        "status": "complete",
        "attempt_relative": "attempts/attempt-0001",
        "attempt_complete_sha256": base._sha(attempt / "COMPLETE.json"),
        "attempt_complete_digest": leaf["complete_digest"],
    }, "terminal_digest")
    base._atomic(case / "COMPLETE.json", terminal)
    assert _terminal(case) == terminal

    malformed = tmp_path / "malformed"
    malformed_attempt = malformed / "attempts" / "attempt-0001"
    malformed_attempt.mkdir(parents=True)
    base._atomic(malformed_attempt / "COMPLETE.json", leaf)
    base._atomic(malformed / "COMPLETE.json", base._sealed({
        **{key: value for key, value in terminal.items()
           if key != "terminal_digest"},
        "attempt_complete_sha256": "0" * 64,
    }, "terminal_digest"))
    with pytest.raises(DtwComponentLadderError):
        _terminal(malformed)
