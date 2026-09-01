from pathlib import Path
import sys

from market_analogues.packed_bound_search import BoundProposal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.m04r.verify_m04r14_t14_10_wf03b_dtw_bound_poc import _key, _retain


def _row(index: int) -> BoundProposal:
    return BoundProposal(
        f"{index:024x}", f"S{index % 3}", index, "A", index / 100,
        ("price",), False,
    )


def test_independent_sampler_is_reverse_chunk_invariant() -> None:
    rows = [_row(index) for index in range(5_000)]
    expected = sorted(rows, key=lambda row: _key("Q", row.episode_id))[:4_096]
    retained = []
    for first in range(len(rows), 0, -137):
        retained = _retain(rows[max(0, first - 137):first], retained, "Q")
    assert retained == expected
