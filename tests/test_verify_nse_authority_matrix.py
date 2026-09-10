from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import verify_m04r14_e2e_nse_authority_matrix as verifier


def _row(symbol: str, position: int, total: float, episode_id: str) -> dict:
    return {
        "symbol": symbol, "position": position, "total": total,
        "episode_id": episode_id,
    }


def test_independent_selector_enforces_overlap_and_instrument_caps() -> None:
    rows = [
        _row("AAA", 100, .01, "a1"),
        _row("AAA", 110, .02, "a2"),  # overlaps a1
        _row("AAA", 400, .03, "a3"),
        _row("AAA", 700, .04, "a4"),
        _row("AAA", 1000, .05, "a5"),  # fourth non-overlap; cap excludes it
        *[_row(f"S{i:02}", 100, .10 + i / 1000, f"s{i:02}") for i in range(30)],
    ]
    selected = verifier._select(rows, lookback=252)
    assert len(selected) == 20
    assert [row["episode_id"] for row in selected[:4]] == ["a1", "a3", "a4", "s00"]
    assert "a2" not in {row["episode_id"] for row in selected}
    assert "a5" not in {row["episode_id"] for row in selected}


def test_independent_selector_uses_episode_id_for_equal_distance() -> None:
    rows = [_row(f"S{i}", i * 300, 1.0, value) for i, value in enumerate(
        ["z", "b", "a", "m"] + [f"x{i:02}" for i in range(20)]
    )]
    selected = verifier._select(rows, lookback=252)
    assert [row["episode_id"] for row in selected] == sorted(
        row["episode_id"] for row in rows
    )[:20]


def test_semantic_digest_omits_only_named_fields() -> None:
    left = {"value": 3, "elapsed": 1.0, "result_digest": "old"}
    right = {"value": 3, "elapsed": 9.0, "result_digest": "new"}
    assert verifier._semantic_digest(left, "elapsed", "result_digest") \
        == verifier._semantic_digest(right, "elapsed", "result_digest")
