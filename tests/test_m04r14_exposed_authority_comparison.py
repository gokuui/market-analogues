from __future__ import annotations
import copy
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from experiments.m04r import m04r14_exposed_authority_comparison as comparison

def _case():
    cert={k: k for k in comparison.STABLE_CERTIFICATE_FIELDS}
    return {"registry_case_id":"case","query_episode_id":"query","matches":[{"id":1}],
        "query_stock_prefix":{"digest":"s"},"query_benchmark_prefix":{"digest":"b"},
        "certificate":cert,"gate_passed":True}

def test_comparison_requires_every_boundary_and_ignores_frontier_only_fields():
    left=_case(); right=copy.deepcopy(left)
    left["certificate"]["rounds"]=[1]; right["certificate"]["rounds"]=[2]
    assert all(comparison.compare_case(left,right).values())
    for key in ("matches","query_stock_prefix","query_benchmark_prefix","gate_passed"):
        changed=copy.deepcopy(right); changed[key]=False
        assert not all(comparison.compare_case(left,changed).values())
    changed=copy.deepcopy(right); changed["certificate"]["eligible_candidates"]="changed"
    assert not all(comparison.compare_case(left,changed).values())
