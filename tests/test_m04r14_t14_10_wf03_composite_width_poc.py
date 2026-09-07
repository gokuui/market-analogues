from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import m04r14_t14_10_wf03_composite_width_poc as subject


def test_width_topologies_use_all_twelve_cores() -> None:
    assert subject.TOPOLOGIES == (("P4T3.json", 4, 3), ("P12T1.json", 12, 1))
    assert all(groups * threads == 12 for _, groups, threads in subject.TOPOLOGIES)


def test_width_semantic_map_ignores_measurements() -> None:
    case = {
        "query_id": "a" * 24, "proposal_result_digest": "proposal",
        "certificate": {"result_digest": "result", "elapsed_seconds": 1.0},
        "matches": [], "exact_seconds": 2.0,
    }
    changed = {**case, "exact_seconds": 9.0,
               "certificate": {**case["certificate"], "elapsed_seconds": 8.0}}
    assert subject._semantic_map({"cases": [case]}) \
        == subject._semantic_map({"cases": [changed]})
