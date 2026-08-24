from __future__ import annotations

import json
from pathlib import Path

from market_analogues.m04r_quantized_rank_verification import (
    OMITTED, verify_m04r_quantized_ranks,
    write_m04r_quantized_rank_verification,
)
from market_analogues.quantized_bound import quantized_bound_contract
from market_analogues.types import stable_hash


def _passing_evidence() -> dict:
    cases = []
    for case_index in range(12):
        upper = 343 if case_index == 3 else 80
        targets = [{
            "authority_rank": target_index + 1,
            "episode_id": f"episode-{case_index}-{target_index}",
            "symbol": f"T{case_index}-{target_index}",
            "quantized_bound": float(target_index),
            "lower_rank": upper if target_index == 19 else target_index + 1,
            "upper_rank": upper if target_index == 19 else target_index + 1,
            "ties_including_target": 1,
        } for target_index in range(20)]
        ranks = [target["upper_rank"] for target in targets]
        cases.append({
            "query_episode_id": f"query-{case_index}",
            "authority_digest": f"authority-{case_index}",
            "eligible_rows": 3_000_000,
            "authority_eligible_rows": 3_000_000,
            "row_accounting_matches": True,
            "targets_seen_once": True,
            "targets": targets,
            "maximum_target_rank": max(ranks),
            "recall": {
                str(quota): sum(rank <= quota for rank in ranks) / 20
                for quota in (100, 500, 1_000, 2_000)
            },
        })
    prefixes = {
        f"S{index}": {"digest": f"prefix-{index}"}
        for index in range(11_001)
    }
    benchmark_prefix = {"digest": "benchmark-prefix", "rows": 5_000}
    payload = {
        "schema_version": "m04r-quantized-bound-rank-gate-v1",
        "contract_digest": quantized_bound_contract()["digest"],
        "authority_cases": cases,
        "authority_case_count": 12,
        "universe_symbols": len(prefixes),
        "source_prefix_count": len(prefixes),
        "source_prefixes": prefixes,
        "benchmark_prefix": benchmark_prefix,
        "source_scope_digest": stable_hash({
            "stocks": prefixes, "benchmark": benchmark_prefix,
        }),
        "stride": 5,
        "maximum_target_rank": 343,
        "top_100_recall_passed": False,
        "top_1000_recall_passed": True,
        "all_row_accounting_passed": True,
        "rank_gate_passed": True,
        "overflow_rows": 2,
        "overflow_examples": [{
            "symbol": "BAD", "cutoff": "2000-01-01", "routing_bound": 0.0,
        }],
        "overflow_routing_policy": (
            "no quantized row emitted; route with universal safe bound zero and "
            "require exact/float32 sidecar"
        ),
        "real_forward_outcomes_accessed": False,
    }
    deterministic = {
        key: value for key, value in payload.items() if key not in OMITTED
    }
    payload["result_digest"] = stable_hash(deterministic)
    return payload


def _refresh_digest(payload: dict) -> None:
    deterministic = {
        key: value for key, value in payload.items() if key not in OMITTED
    }
    payload["result_digest"] = stable_hash(deterministic)


def test_quantized_rank_verifier_recomputes_gate_and_writes_html(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps(_passing_evidence()))
    result = verify_m04r_quantized_ranks(evidence)
    assert result.passed, result.failures
    assert result.metrics["maximum_target_rank"] == 343
    assert result.metrics["top_100_recall_passed"] is False
    machine, html = write_m04r_quantized_rank_verification(
        result, tmp_path / "report",
    )
    assert machine.exists()
    assert '<span class="pass">PASS</span>' in html.read_text()


def test_quantized_rank_verifier_rejects_refreshed_rank_tampering(
    tmp_path: Path,
) -> None:
    payload = _passing_evidence()
    payload["authority_cases"][0]["targets"][0]["upper_rank"] = 1_001
    _refresh_digest(payload)
    evidence = tmp_path / "tampered.json"
    evidence.write_text(json.dumps(payload))
    result = verify_m04r_quantized_ranks(evidence)
    assert not result.passed
    assert any("recall" in failure or "maximum" in failure for failure in result.failures)


def test_quantized_rank_verifier_rejects_unsafe_overflow_policy(
    tmp_path: Path,
) -> None:
    payload = _passing_evidence()
    payload["overflow_routing_policy"] = "clip overflowing values"
    _refresh_digest(payload)
    evidence = tmp_path / "unsafe.json"
    evidence.write_text(json.dumps(payload))
    result = verify_m04r_quantized_ranks(evidence)
    assert not result.passed
    assert "quantized overflow routing policy differs" in result.failures
