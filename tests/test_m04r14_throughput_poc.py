from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.m04r import m04r14_throughput_poc as poc


def _baseline(case_id: str, exact: float) -> dict:
    return {
        "semantic": {
            "case_id": case_id, "query_id": f"q-{case_id}",
            "forward_proposal": {"result_digest": f"p-{case_id}"},
            "matches": [{"episode_id": f"m-{case_id}"}],
            "certificate": {"result_digest": f"c-{case_id}"},
        },
        "measurement": {"exact_stage_seconds": exact},
    }


def test_balanced_groups_are_deterministic_lpt() -> None:
    cases = [{"case_id": value} for value in "abcd"]
    baseline = {
        "a": _baseline("a", 8), "b": _baseline("b", 7),
        "c": _baseline("c", 6), "d": _baseline("d", 5),
    }
    groups = poc.balanced_groups(cases, baseline, 2)
    assert [[row["case_id"] for row in group] for group in groups] == [
        ["a", "d"], ["b", "c"],
    ]
    with pytest.raises(poc.ThroughputError, match="process count"):
        poc.balanced_groups(cases, baseline, 0)


def test_available_cpus_are_unique_and_nonempty() -> None:
    cpus = poc.available_cpus()
    assert cpus
    assert cpus == tuple(sorted(set(cpus)))


def test_case_comparison_requires_every_semantic_boundary() -> None:
    baseline = _baseline("a", 1)
    observed = {
        "registry_case_id": "a", "query_episode_id": "q-a",
        "proposal_result_digest": "p-a",
        "matches": baseline["semantic"]["matches"],
        "certificate": baseline["semantic"]["certificate"],
        "gate_passed": True,
    }
    assert all(poc.compare_case(observed, baseline).values())
    for key in ("proposal_result_digest", "matches", "certificate", "gate_passed"):
        changed = copy.deepcopy(observed)
        changed[key] = False
        assert not all(poc.compare_case(changed, baseline).values())
