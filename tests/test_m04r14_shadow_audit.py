from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import compare_m04r14_shadow_audit as comparison
from experiments.m04r import m04r14_shadow_audit_authority as authority
from experiments.m04r import verify_m04r14_shadow_audit_authority as verifier


def test_audit_authority_uses_distinct_exact_schedule() -> None:
    assert authority.CONTROLS == verifier.CONTROLS
    assert authority.CONTROLS["block_rows"] == 4097
    assert authority.CONTROLS["initial_frontier_rows"] == 16_384
    assert authority.CONTROLS["maximum_frontier_rows"] == 32_768
    assert authority.OUTPUT != authority.base.OUTPUT


def test_audit_selection_requires_exact_preregistered_identity() -> None:
    cases = [{
        "case_id": f"case-{index}", "episode_id": f"query-{index}",
        "symbol": f"S{index}", "quality_tier": "A",
        "liquidity_stratum": "high",
    } for index in range(3270)]
    sample = deepcopy(cases[:12])
    assert len(authority._selected_cases({"cases_data": cases}, sample)) == 12
    sample[0]["symbol"] = "DRIFT"
    with pytest.raises(authority.AuditAuthorityError, match="binds registry"):
        authority._selected_cases({"cases_data": cases}, sample)


def test_independent_verifier_selection_rejects_duplicates() -> None:
    cases = [{
        "case_id": f"case-{index}", "episode_id": f"query-{index}",
        "symbol": f"S{index}", "quality_tier": "B",
        "liquidity_stratum": "middle",
    } for index in range(3270)]
    sample = deepcopy(cases[:12])
    sample[-1] = deepcopy(sample[0])
    with pytest.raises(verifier.AuditVerificationError, match="duplicate"):
        verifier._selected({"cases_data": cases}, sample)


def test_ordered_comparison_rejects_one_rank_swap() -> None:
    matches = [{"episode_id": f"e{index}", "total_distance": float(index)}
               for index in range(20)]
    candidate = {
        "registry_case_id": "case", "query_episode_id": "query",
        "query_stock_prefix": "stock", "query_benchmark_prefix": "benchmark",
        "matches": matches,
    }
    exact = deepcopy(candidate)
    assert comparison._compare_rows(candidate, exact)["matching_positions"] == 20
    exact["matches"][5], exact["matches"][6] = exact["matches"][6], exact["matches"][5]
    result = comparison._compare_rows(candidate, exact)
    assert result["passed"] is False
    assert result["matching_positions"] == 18
