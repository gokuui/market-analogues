from __future__ import annotations

import json
from pathlib import Path

from market_analogues.m04r_proposal_verification import (
    EVIDENCE_NONDETERMINISTIC_FIELDS, verify_m04r_proposal_v2,
    write_m04r_proposal_verification,
)
from market_analogues.proposal_v2 import (
    LAYOUTS, PROPOSAL_POOLS, PROPOSAL_ROUTES, proposal_v2_contract,
    proposal_v2_storage_bytes,
)
from market_analogues.types import stable_hash


def _passing_evidence() -> dict:
    ranks = [{
        "authority_rank": index + 1,
        "episode_id": f"episode-{index}",
        "symbol": f"S{index}",
        "lower_route_ranks": {route: index + 1 for route in PROPOSAL_ROUTES},
        "upper_route_ranks": {route: index + 1 for route in PROPOSAL_ROUTES},
        "lower_composite_rank": index + 1,
        "upper_composite_rank": index + 1,
        "ties_including_target": {
            route: 1 for route in (*PROPOSAL_ROUTES, "composite")
        },
    } for index in range(20)]
    pool_recalls = {
        str(pool): {
            "admitted": 20, "total": 20, "recall": 1.0,
            "conservative_upper_tie_rank": True,
        }
        for pool in PROPOSAL_POOLS
    }
    layouts = {
        str(dimensions): {
            dtype: {
                "pool_recalls": pool_recalls,
                "target_ranks": ranks,
                "target_exact_spearman": {
                    route: .5 for route in (*PROPOSAL_ROUTES, "composite")
                },
                "passed": True,
            }
            for dtype in ("float32", "float16")
        }
        for dimensions in LAYOUTS
    }
    cases = [{
        "query_episode_id": f"query-{index}",
        "authority_digest": f"authority-{index}",
        "symbol": f"Q{index}", "cutoff": "2020-01-01",
        "eligible_rows": 3_000_000,
        "authority_eligible_rows": 3_000_000,
        "row_accounting_matches": True,
        "targets_seen_once": True,
        "layouts": layouts,
    } for index in range(12)]
    synthetic = [{
        "dimensions": dimensions, "dtype": dtype, "family": family,
        "clone_input_position": 28,
        "route_ranks": {
            route: 1 for route in (*PROPOSAL_ROUTES, "composite")
        },
        "passed": True,
    } for dimensions in LAYOUTS for dtype in ("float32", "float16")
      for family in range(5)]
    payload = {
        "schema_version": "m04r-proposal-v2-authority-gate-v1",
        "contract_digest": proposal_v2_contract()["digest"],
        "authority_cases": cases,
        "authority_case_count": 12,
        "universe_symbols": 11_584,
        "evaluated_symbols": 11_584,
        "is_full_universe": True,
        "stride": 5,
        "source_prefix_count": 11_584,
        "source_prefixes": {
            f"S{index}": {"digest": f"prefix-{index}"}
            for index in range(11_584)
        },
        "synthetic_gate": {
            "cases": synthetic, "case_count": 30, "all_passed": True,
            "adversarial_local_cap_depth": 27,
        },
        "layout_float32_pass": {str(dimensions): True for dimensions in LAYOUTS},
        "layout_float16_pass": {str(dimensions): False for dimensions in LAYOUTS},
        "mean_target_exact_spearman": {},
        "storage_projection": {
            str(dimensions): {
                dtype: {
                    "signature_bytes": proposal_v2_storage_bytes(dimensions, dtype),
                    "projected_3_82m_gib": 1.0,
                }
                for dtype in ("float32", "float16")
            } for dimensions in LAYOUTS
        },
        "quantization_sensitivity": {},
        "quantization_overflow_rows": {
            str(dimensions): 1 for dimensions in LAYOUTS
        },
        "quantization_maximum_absolute_value": {
            str(dimensions): 70_000.0 for dimensions in LAYOUTS
        },
        "quantization_overflow_examples": {
            str(dimensions): [{"symbol": "BAD", "cutoff": "2000-01-01"}]
            for dimensions in LAYOUTS
        },
        "selected_dimensions": 192,
        "selected_dtype": "float32",
        "selected_signature_bytes": proposal_v2_storage_bytes(192, "float32"),
        "selection_rule": "frozen",
        "all_cases_passed": True,
        "real_forward_outcomes_accessed": False,
    }
    payload["source_scope_digest"] = stable_hash(payload["source_prefixes"])
    deterministic = {
        key: value for key, value in payload.items()
        if key not in EVIDENCE_NONDETERMINISTIC_FIELDS
    }
    payload["result_digest"] = stable_hash(deterministic)
    return payload


def test_proposal_verifier_recomputes_selection_and_writes_html(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps(_passing_evidence()))
    result = verify_m04r_proposal_v2(evidence)
    assert result.passed, result.failures
    assert result.metrics["selected_dimensions"] == 192
    assert result.metrics["selected_dtype"] == "float32"
    machine, html, contract = write_m04r_proposal_verification(
        result, tmp_path / "report",
    )
    assert machine.exists() and contract.exists()
    assert "proposal v2: <span class=\"pass\">PASS" in html.read_text()


def test_proposal_verifier_rejects_tampering_even_with_refreshed_digest(
    tmp_path: Path,
) -> None:
    payload = _passing_evidence()
    target = payload["authority_cases"][0]["layouts"]["192"]["float32"][
        "target_ranks"
    ][0]
    target["upper_composite_rank"] = 99_999
    target["upper_route_ranks"] = {
        route: 99_999 for route in PROPOSAL_ROUTES
    }
    deterministic = {
        key: value for key, value in payload.items()
        if key not in EVIDENCE_NONDETERMINISTIC_FIELDS
    }
    payload["result_digest"] = stable_hash(deterministic)
    evidence = tmp_path / "tampered.json"
    evidence.write_text(json.dumps(payload))
    result = verify_m04r_proposal_v2(evidence)
    assert not result.passed
    assert any("reported pool" in failure for failure in result.failures)
