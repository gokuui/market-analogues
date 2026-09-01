from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.m04r import verify_m04r14_t14_10_wf03_feasibility as subject
from market_analogues.types import stable_hash


def test_seal_accepts_exact_state_and_rejects_mutation() -> None:
    state = {"status": "verified", "passed": True}
    value = {**state, "verification_digest": stable_hash(state)}
    subject._seal(value, "verification_digest")
    value["passed"] = False
    with pytest.raises(subject.VerificationError, match="verification_digest"):
        subject._seal(value, "verification_digest")


def test_strict_reader_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"a":1,"a":2}\n')
    with pytest.raises(subject.VerificationError, match="duplicate"):
        subject._read(path)


def test_semantic_proposal_omits_only_measurements_and_scan_order() -> None:
    value = {
        "candidate_digest": "a" * 64, "block_rows": 4096,
        "block_order": "forward", "elapsed_seconds": 1.0, "peak_rss_mb": 2.0,
        "eligible_rows": 10,
    }
    assert subject._semantic_proposal(value) == {
        "candidate_digest": "a" * 64, "eligible_rows": 10,
    }
