from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from market_analogues.m04r_global_proposal_verification import (
    EVIDENCE_OMITTED, FULL_BUILD_OMITTED,
    verify_m04r_global_bound_proposal,
)
from market_analogues.m04r_quantized_rank_verification import OMITTED as RANK_OMITTED
from market_analogues.packed_bound_search import (
    DEFAULT_ROUTE_QUOTAS, packed_bound_search_contract,
)
from market_analogues.packed_bound_store import (
    make_overflow_record, make_packed_record, write_packed_generation,
)
from market_analogues.quantized_bound import quantize_bound_row, quantized_bound_contract
from market_analogues.representation import represent
from market_analogues.synthetic import generate_case
from market_analogues.types import stable_hash


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True))


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    query_id = "f" * 24
    composite_ids = [f"{value:024x}" for value in range(1, 1_001)]
    prefixes = {
        f"S{index:05d}": {"digest": f"{index:064x}"[-64:]}
        for index in range(11_000)
    }
    benchmark = {"digest": "b" * 64, "rows": 1}
    cases = []
    for case_index in range(12):
        case_id = query_id if case_index == 0 else f"{10_000 + case_index:024x}"
        targets = [{
            "episode_id": composite_ids[index],
            "lower_rank": index + 1,
            "upper_rank": index + 1,
            "ties_including_target": 1,
            "quantized_bound": float(index) / 100,
        } for index in range(20)]
        cases.append({
            "query_episode_id": case_id,
            "authority_digest": f"{case_index + 1:064x}",
            "eligible_rows": 1_000,
            "authority_eligible_rows": 1_000,
            "row_accounting_matches": True,
            "targets_seen_once": True,
            "maximum_target_rank": 20,
            "targets": targets,
            "recall": {str(value): 1.0 for value in (100, 500, 1_000, 2_000)},
        })
    rank: dict[str, object] = {
        "schema_version": "m04r-quantized-bound-rank-gate-v1",
        "contract_digest": quantized_bound_contract()["digest"],
        "real_forward_outcomes_accessed": False,
        "universe_symbols": 11_000,
        "source_prefixes": prefixes,
        "source_prefix_count": 11_000,
        "benchmark_prefix": benchmark,
        "source_scope_digest": stable_hash({"stocks": prefixes, "benchmark": benchmark}),
        "authority_case_count": 12,
        "authority_cases": cases,
        "stride": 5,
        "maximum_target_rank": 20,
        "top_100_recall_passed": True,
        "top_1000_recall_passed": True,
        "all_row_accounting_passed": True,
        "rank_gate_passed": True,
        "overflow_rows": 1,
        "overflow_examples": [{
            "symbol": "BBB", "cutoff": "2020-01-01", "routing_bound": 0.0,
        }],
        "overflow_routing_policy": (
            "no quantized row emitted; route with universal safe bound zero and "
            "require exact/float32 sidecar"
        ),
    }
    rank["result_digest"] = stable_hash({
        key: value for key, value in rank.items() if key not in RANK_OMITTED
    })
    rank_path = tmp_path / "rank.json"
    _write(rank_path, rank)

    representation = represent(generate_case("rounded_base", 42).episode)
    quantized = quantize_bound_row(representation)
    rows = np.concatenate([
        make_packed_record(composite_ids[index], index + 1, 0, "A", quantized)
        for index in range(999)
    ])
    overflow = make_overflow_record(composite_ids[-1], 1, 1, "B")
    store_root = tmp_path / "store"
    generation = write_packed_generation(
        store_root, rows, overflow, ("AAA", "BBB"),
        {"rank_evidence_digest": rank["result_digest"]}, activate=False,
    )

    top_digest = "c" * 64
    selection = {"query_episode_id": query_id, "method": "fixture"}
    build: dict[str, object] = {
        "schema_version": "m04r-packed-bound-full-build-v1",
        "generation_id": generation,
        "gate_passed": True,
        "shadow_generation": True,
        "benchmark_selection": selection,
        "warm_scans_second": [{"top_1000_digest": top_digest}],
    }
    build["result_digest"] = stable_hash({
        key: value for key, value in build.items() if key not in FULL_BUILD_OMITTED
    })
    build_path = tmp_path / "build.json"
    _write(build_path, build)

    report = {
        "generation_id": generation,
        "query_episode_id": query_id,
        "rows_scanned": 1_000,
        "eligible_rows": 1_000,
        "eligible_main_rows": 999,
        "eligible_overflow_rows": 1,
        "candidate_count": 1_001,
        "composite_count": 1_000,
        "route_counts": DEFAULT_ROUTE_QUOTAS,
        "route_quotas": DEFAULT_ROUTE_QUOTAS,
        "candidate_digest": "d" * 64,
        "top_1000_score_digest": top_digest,
        "duplicate_candidates": 0,
        "future_candidates": 0,
        "overlapping_same_symbol_candidates": 0,
        "overflow_candidates": 1,
        "composite_episode_ids": composite_ids,
    }
    gates = {
        "full_row_accounting": True,
        "forward_reverse_and_block_invariance": True,
        "top_1000_scores_equal_m04r_06c": True,
        "all_selected_authority_targets_retained": True,
        "all_12_rank_evidence_supports_composite_quota": True,
        "no_duplicates_or_temporal_violations": True,
        "composite_quota_preserved": True,
        "overflow_fallback_exercised": True,
        "rss_within_768_mib": True,
        "each_scan_within_600_seconds": True,
    }
    evidence: dict[str, object] = {
        "schema_version": "m04r-global-bound-proposal-gate-v1",
        "contract_digest": packed_bound_search_contract()["digest"],
        "generation_id": generation,
        "full_build_evidence_digest": build["result_digest"],
        "query_episode_id": query_id,
        "selection_method": selection,
        "block_rows_forward": 7,
        "block_rows_reverse": 13,
        "forward": report,
        "reverse": dict(report),
        "target_retention": {
            "target_count": 20,
            "retained_count": 20,
            "missing_ids": [],
            "rank_evidence_digest": rank["result_digest"],
            "rank_evidence_all_12_top_1000": True,
            "rank_evidence_maximum_target_rank": 20,
        },
        "prior_top_1000_score_digest": top_digest,
        "gates": gates,
        "gate_passed": True,
        "forward_seconds": 1.0,
        "reverse_seconds": 1.1,
        "peak_rss_mb": 100.0,
        "real_forward_outcomes_accessed": False,
    }
    evidence["result_digest"] = stable_hash({
        key: value for key, value in evidence.items()
        if key not in EVIDENCE_OMITTED
    })
    evidence_path = tmp_path / "evidence.json"
    _write(evidence_path, evidence)
    return evidence_path, store_root, build_path, rank_path


def test_global_proposal_verifier_passes_and_rejects_target_tamper(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    assert verify_m04r_global_bound_proposal(*paths).passed
    evidence = json.loads(paths[0].read_text())
    evidence["forward"]["composite_episode_ids"][0] = "e" * 24
    _write(paths[0], evidence)
    result = verify_m04r_global_bound_proposal(*paths)
    assert not result.passed
    assert "global proposal evidence digest differs" in result.failures
    assert "global composite does not retain the selected exact authority" in result.failures
