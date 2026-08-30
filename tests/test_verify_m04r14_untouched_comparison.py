from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.m04r import verify_m04r14_untouched_comparison as verifier


def _rows() -> dict[str, dict]:
    result = {}
    for index in range(72):
        query = f"{index:024d}"
        result[query] = {"registry_case_id": f"case-{index}",
            "query_stock_prefix": {"digest": f"s-{index}"},
            "query_benchmark_prefix": {"digest": f"b-{index}"},
            "matches": [{"episode_id": f"{index}-{position}", "distance": position / 10}
                        for position in range(20)]}
    return result


def test_reconstruction_requires_exact_ordered_agreement() -> None:
    candidate = _rows(); authority = _rows()
    rows = verifier.reconstruct_rows(candidate, authority)
    assert len(rows) == 72
    assert sum(row["matching_positions"] for row in rows) == 1440
    assert all(row["passed"] for row in rows)
    authority["000000000000000000000000"]["matches"].reverse()
    changed = verifier.reconstruct_rows(candidate, authority)
    assert changed[0]["passed"] is False
    assert changed[0]["matching_positions"] < 20


def test_reconstruction_rejects_missing_case() -> None:
    candidate = _rows(); authority = _rows(); authority.pop(next(iter(authority)))
    with pytest.raises(verifier.TerminalVerificationError, match="inventory"):
        verifier.reconstruct_rows(candidate, authority)


def test_terminal_publish_is_create_only(tmp_path: Path) -> None:
    root = tmp_path / "verification"; verifier._publish(root, {"passed": True})
    with pytest.raises(verifier.TerminalVerificationError, match="exists"):
        verifier._publish(root, {"passed": True})
