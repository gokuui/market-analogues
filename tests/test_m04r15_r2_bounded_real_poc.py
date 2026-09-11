from __future__ import annotations
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.m04r import m04r15_r2_bounded_real_poc as poc


def test_selection_summary_exposes_real_medoid_identity_and_no_claims()->None:
    from market_analogues.future_modes import Member,PreparedPath,pairwise_l1,select_modes
    paths=tuple(PreparedPath(Member(i+1,f"e{i}",f"S{i}","2020-01-02","s"),(x,)*4,tuple(str(j)for j in range(4)))
                for i,x in enumerate([-1,-.9,-1.1,1,.9,1.1]))
    selection=select_modes(pairwise_l1(paths),[p.member.key for p in paths],
        [f"202{i}-Q1"for i in range(6)],contract_digest="a"*64,query_case_id="q",view_id="v",replicates=16)
    result=poc._selection_summary(selection,paths)
    assert result["status"]=="stable_multiple_modes" and result["selected_k"]==2
    assert set(result["member_to_mode"])=={f"e{i}"for i in range(6)}
