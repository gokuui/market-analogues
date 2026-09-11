from __future__ import annotations
import ast
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.m04r import verify_m04r15_r2_bounded_real_poc as verifier
def test_real_poc_verifier_imports_no_producer_module():
    tree=ast.parse((ROOT/"experiments/m04r/verify_m04r15_r2_bounded_real_poc.py").read_text());imports=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):imports.update(x.name for x in node.names)
        elif isinstance(node,ast.ImportFrom)and node.module:imports.add(node.module)
    assert imports.isdisjoint(verifier.PRODUCER_MODULES)


def test_every_bounded_real_poc_panel_is_independently_reconstructed():
    result=verifier.verify(ROOT)
    assert result["passed"] is True
    assert result["verified_query_count"]==32 and result["verified_view_count"]==64
    assert result["mode_status_counts"]=={"one_mode_fallback":53,"stable_multiple_modes":11}
    assert result["full_query_build_authorized"] is True
    assert result["predictive_claim_authorized"] is False
