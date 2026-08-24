from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from market_analogues.m04r_full_pack_verification import (
    EVIDENCE_OMITTED, verify_m04r_full_pack,
    write_m04r_full_pack_verification,
)
from market_analogues.packed_bound_store import (
    OVERFLOW_DTYPE, PACK_DTYPE, make_overflow_record, make_packed_record,
    packed_bound_store_contract, write_packed_generation,
)
from market_analogues.quantized_bound import quantize_bound_row
from market_analogues.representation import represent
from market_analogues.synthetic import generate_case
from market_analogues.types import stable_hash


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    symbols = [f"S{index:05d}" for index in range(11_584)]
    prefixes = {symbol: {"digest": f"prefix-{symbol}"} for symbol in symbols}
    benchmark = {"digest": "benchmark", "rows": 5_000}
    rank = {
        "result_digest": "rank-evidence", "source_prefixes": prefixes,
        "benchmark_prefix": benchmark, "overflow_rows": 35,
        "authority_cases": [{
            "query_episode_id": f"query-{index}", "eligible_rows": 36,
        } for index in range(12)],
    }
    rank_path = tmp_path / "rank.json"
    rank_path.write_text(json.dumps(rank))
    selection = {
        "method": "complete sha256-ordered A/B universe", "fraction": 1.0,
        "universe_count": 11_584, "sample_count": 11_584,
        "forced_overflow_symbols": [], "symbols": symbols,
    }
    provenance = {
        "rank_evidence_digest": rank["result_digest"],
        "source_prefixes": prefixes, "benchmark_prefix": benchmark,
        "selection": selection, "selection_digest": stable_hash(selection),
    }
    representation = represent(generate_case("rounded_base", 88).episode)
    rows = make_packed_record(
        f"{1:024x}", 1, 0, "A", quantize_bound_row(representation),
    )
    overflow = np.concatenate([
        make_overflow_record(f"{index:024x}", index, 0, "B")
        for index in range(2, 37)
    ]).astype(OVERFLOW_DTYPE)
    store = tmp_path / "store"
    generation = write_packed_generation(
        store, rows, overflow, symbols, provenance, activate=False,
    )
    scans = [{
        "query_episode_id": f"query-{index}",
        "eligible_rows": 36, "full_eligible_rows": 36,
        "projection_factor": 1.0, "overflow_eligible_rows": 35,
        "top_1000_digest": f"digest-{index}", "minimum_bound": 0.0,
        "seconds": 1.0, "projected_full_seconds": 1.0,
    } for index in range(12)]
    one_percent = {
        "result_digest": "one-percent-evidence",
        "warm_scans_second": [
            {**row, "projected_full_seconds": 2.0 if index == 0 else 1.0}
            for index, row in enumerate(scans)
        ],
    }
    one_percent_path = tmp_path / "one-percent.json"
    one_percent_path.write_text(json.dumps(one_percent))
    pack_bytes = PACK_DTYPE.itemsize + 35 * OVERFLOW_DTYPE.itemsize
    evidence = {
        "schema_version": "m04r-packed-bound-full-build-v1",
        "pack_contract_digest": packed_bound_store_contract()["digest"],
        "generation_id": generation, "sample": selection,
        "rows": 1, "overflow_rows": 35, "eligible_rows": 36,
        "pack_bytes": pack_bytes, "projected_full_rows": 3_820_000,
        "projected_full_bytes": pack_bytes,
        "projected_full_gib": pack_bytes / 1024 ** 3,
        "cold_cache_advised": True, "cold_scan": scans[0],
        "warm_scans_first": scans[:1], "warm_scans_second": scans[:1],
        "full_authority_row_counts": [{
            "query_episode_id": f"query-{index}",
            "eligible_rows": 36, "full_eligible_rows": 36,
            "overflow_eligible_rows": 35, "row_accounting_matches": True,
        } for index in range(12)],
        "authority_row_accounting_passed": True,
        "benchmark_selection": {
            "method": "maximum query-specific projected warm seconds in sealed 1% evidence",
            "source_evidence_digest": one_percent["result_digest"],
            "query_episode_id": "query-0",
        },
        "maximum_warm_seconds": 1.0, "projected_warm_seconds": 1.0,
        "projected_cold_seconds": 1.0,
        "scan_deterministic": True, "capacity_passed": True,
        "scan_rss_passed": True,
        "warm_latency_passed": True, "cold_latency_passed": True,
        "overflow_sidecar_exercised": True, "shadow_generation": True,
        "scan_io_mode": "bounded positional reads over raw immutable pack",
        "resume_evidence": {
            "schema_version": "m04r-packed-bound-full-build-v1",
            "sample_symbols": 11_584, "interrupted_completed_symbols": 100,
            "interrupted_remaining_symbols": 11_484,
            "resumed_completed_symbols": 11_584, "resume_completed": True,
        },
        "real_forward_outcomes_accessed": False,
        "poc_passed": True, "gate_passed": True, "peak_rss_mb": 100.0,
        "scan_peak_rss_mb": 100.0, "validation_peak_rss_mb": 200.0,
    }
    deterministic = {
        key: value for key, value in evidence.items() if key not in EVIDENCE_OMITTED
    }
    evidence["result_digest"] = stable_hash(deterministic)
    evidence_path = tmp_path / "packed-bound-full.json"
    evidence_path.write_text(json.dumps(evidence))
    return evidence_path, store, rank_path, one_percent_path


def test_full_pack_verifier_recomputes_shadow_and_authority_gates(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)
    result = verify_m04r_full_pack(*arguments)
    assert result.passed, result.failures
    machine, html = write_m04r_full_pack_verification(
        result, tmp_path / "report",
    )
    assert machine.exists()
    assert '<span class="pass">PASS</span>' in html.read_text()


def test_full_pack_verifier_rejects_refreshed_row_tampering(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    payload = json.loads(arguments[0].read_text())
    payload["full_authority_row_counts"][0]["eligible_rows"] = 35
    deterministic = {
        key: value for key, value in payload.items() if key not in EVIDENCE_OMITTED
    }
    payload["result_digest"] = stable_hash(deterministic)
    arguments[0].write_text(json.dumps(payload))
    result = verify_m04r_full_pack(*arguments)
    assert not result.passed
    assert "full packed authority metadata row accounting differs" in result.failures
