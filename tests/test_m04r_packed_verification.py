from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from market_analogues.m04r_packed_verification import (
    EVIDENCE_OMITTED, verify_m04r_packed_bound_poc,
    write_m04r_packed_bound_verification,
)
from market_analogues.packed_bound_store import (
    OVERFLOW_DTYPE, PACK_DTYPE, make_overflow_record, make_packed_record,
    packed_bound_store_contract, write_packed_generation,
)
from market_analogues.quantized_bound import quantize_bound_row
from market_analogues.representation import represent
from market_analogues.synthetic import generate_case
from market_analogues.types import stable_hash


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    symbols = [f"S{index:03d}" for index in range(116)]
    prefixes = {symbol: {"digest": f"prefix-{symbol}"} for symbol in symbols}
    benchmark = {"digest": "benchmark", "rows": 5_000}
    selection = {
        "method": "sha256 order with known-overflow symbols forced inside fixed ceil(1%) size",
        "fraction": .01, "universe_count": 11_584,
        "sample_count": 116, "forced_overflow_symbols": [symbols[0]],
        "symbols": symbols,
    }
    rank = {
        "result_digest": "rank-evidence", "source_prefixes": prefixes,
        "benchmark_prefix": benchmark,
    }
    rank_path = tmp_path / "rank.json"
    rank_path.write_text(json.dumps(rank))
    provenance = {
        "rank_evidence_digest": rank["result_digest"],
        "source_prefixes": prefixes, "benchmark_prefix": benchmark,
        "selection": selection, "selection_digest": stable_hash(selection),
    }
    representation = represent(generate_case("rounded_base", 77).episode)
    rows = make_packed_record(
        f"{1:024x}", 1, 0, "A", quantize_bound_row(representation),
    )
    overflow = make_overflow_record(f"{2:024x}", 2, 0, "B")
    roots = [tmp_path / "parallel", tmp_path / "serial"]
    generation_ids = [write_packed_generation(
        root / "store", rows, overflow, symbols, provenance,
    ) for root in roots]
    assert generation_ids[0] == generation_ids[1]
    full_rows = 3_820_000
    projected_bytes = (
        full_rows * PACK_DTYPE.itemsize
        + math.ceil(full_rows * len(overflow) / len(rows)) * OVERFLOW_DTYPE.itemsize
    )
    scans = [{
        "query_episode_id": f"query-{index}",
        "eligible_rows": 2, "full_eligible_rows": 200,
        "projection_factor": 100.0, "overflow_eligible_rows": 1,
        "top_1000_digest": f"digest-{index}", "minimum_bound": 0.0,
        "seconds": .01, "projected_full_seconds": 1.0,
    } for index in range(12)]
    evidence = {
        "schema_version": "m04r-packed-bound-1pct-poc-v1",
        "pack_contract_digest": packed_bound_store_contract()["digest"],
        "generation_id": generation_ids[0], "sample": selection,
        "rows": 1, "overflow_rows": 1, "eligible_rows": 2,
        "pack_bytes": PACK_DTYPE.itemsize + OVERFLOW_DTYPE.itemsize,
        "projected_full_rows": full_rows,
        "projected_full_bytes": projected_bytes,
        "projected_full_gib": projected_bytes / 1024 ** 3,
        "cold_cache_advised": True, "cold_scan": scans[0],
        "warm_scans_first": scans, "warm_scans_second": scans,
        "maximum_warm_seconds": .01, "projected_warm_seconds": 1.0,
        "projected_cold_seconds": 1.0, "scan_deterministic": True,
        "capacity_passed": True, "warm_latency_passed": True,
        "cold_latency_passed": True, "overflow_sidecar_exercised": True,
        "resume_evidence": {
            "schema_version": "m04r-packed-bound-1pct-poc-v1",
            "sample_symbols": 116, "interrupted_completed_symbols": 20,
            "interrupted_remaining_symbols": 96,
            "resumed_completed_symbols": 116, "resume_completed": True,
        },
        "real_forward_outcomes_accessed": False, "poc_passed": True,
        "peak_rss_mb": 100.0,
    }
    deterministic = {
        key: value for key, value in evidence.items() if key not in EVIDENCE_OMITTED
    }
    evidence["result_digest"] = stable_hash(deterministic)
    evidence_paths = []
    for root in roots:
        path = root / "packed-bound-1pct.json"
        path.write_text(json.dumps(evidence))
        evidence_paths.append(path)
    return evidence_paths[0], evidence_paths[1], roots[0] / "store", roots[1] / "store", rank_path


def test_packed_verifier_recomputes_physical_and_performance_gates(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)
    result = verify_m04r_packed_bound_poc(*arguments)
    assert result.passed, result.failures
    machine, html = write_m04r_packed_bound_verification(
        result, tmp_path / "report",
    )
    assert machine.exists()
    assert '<span class="pass">PASS</span>' in html.read_text()


def test_packed_verifier_rejects_refreshed_projection_tampering(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)
    payload = json.loads(arguments[0].read_text())
    payload["warm_scans_second"][0]["projected_full_seconds"] = .1
    deterministic = {
        key: value for key, value in payload.items() if key not in EVIDENCE_OMITTED
    }
    payload["result_digest"] = stable_hash(deterministic)
    arguments[0].write_text(json.dumps(payload))
    result = verify_m04r_packed_bound_poc(*arguments)
    assert not result.passed
    assert "parallel query-specific latency projection differs" in result.failures
