"""Produce and seal the truth-blind M04R-11 fast candidate pools."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from html import escape
import json
import multiprocessing
import os
from pathlib import Path
import resource
from math import isfinite
from time import perf_counter
from typing import Any

import pandas as pd

from market_analogues.adapters import file_fingerprint, source_from_spec
from market_analogues.config import load_config
from market_analogues.episodes import build_episode
from market_analogues.m04r_validation_registry import validate_m04r_validation_registry
from market_analogues.m04r_candidate_evidence import (
    candidate_checkpoint_integrity_digest, candidate_semantic_digest,
    validate_truth_blind_candidate_case,
)
from market_analogues.packed_bound_search import (
    PackedBoundQuery, bound_proposal_candidate_digest,
    packed_bound_search_contract, scan_packed_bound_proposals_threaded,
)
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.representation import represent, representation_input_digest
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import InstrumentKey, stable_hash


CASE_SCHEMA = "candidate-recall-case-v2-producer"
MATRIX_SCHEMA = "candidate-recall-matrix-v2-producer"
SEAL_SCHEMA = "candidate-recall-producer-seal-v1"
CONTRACT_SCHEMA = "candidate-recall-producer-contract-v2"
SCAN_THREADS = 4
FROZEN_REGISTRY_DIGEST = "0a4da732f91375a091775cb04e6e77c8d136ade47d7f4d16508a2d9a6555361e"
FROZEN_GENERATION_ID = "9fc6ae0ec4451133d8006897162f3443803fd30a3c44b78e80a476fbb18bb483"
FROZEN_PROPOSAL_CONTRACT_DIGEST = "059db6d78bbe62e7588e1fe2408fe244040b501d190bdf7b211a7ecd7aadb96c"
FROZEN_CASE_ORDER_DIGEST = "db1741d50eab19217d6264f051ca3bccf2663c9a54a7e6e846e8689410fe2b47"
FROZEN_ROUTE_QUOTAS = {
    "candle_volatility": 64, "coarse": 64, "composite": 1_000,
    "market_context": 64, "price": 64, "stage": 64,
    "structural": 64, "volume_shock": 64,
}
FROZEN_REQUEST = {
    "cross_dataset": False, "deduplicate_overlaps": True,
    "max_per_instrument": 3, "minimum_history_gap_bars": 60,
    "quality_tiers": ["A", "B"], "top_k": 20,
}
CASE_OMITTED = {
    "created_at", "cold_seconds", "warm_first_seconds", "warm_second_seconds",
    "cold_task_seconds", "peak_rss_mb", "result_digest",
    "checkpoint_integrity_digest",
}
MATRIX_OMITTED = {"created_at", "elapsed_seconds", "result_digest"}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def case_digest(payload: dict[str, Any]) -> str:
    return candidate_semantic_digest(payload)


def matrix_digest(payload: dict[str, Any]) -> str:
    return stable_hash({key: value for key, value in payload.items() if key not in MATRIX_OMITTED})


def checkpoint_integrity_digest(payload: dict[str, Any]) -> str:
    return candidate_checkpoint_integrity_digest(payload)


def _seal_digest(payload: dict[str, Any]) -> str:
    return stable_hash({
        key: value for key, value in payload.items()
        if key not in {"created_at", "seal_digest"}
    })


def _write_success_seal(
    output_root: Path, matrix: dict[str, Any], *, registry_digest: str,
    producer_contract_digest: str,
) -> bool:
    """Seal only a complete matrix whose safety/performance gates all passed."""
    if not all((
        matrix.get("schema_version") == MATRIX_SCHEMA,
        matrix.get("registry_digest") == registry_digest,
        matrix.get("producer_contract_digest") == producer_contract_digest,
        matrix.get("completed_cases") == 60,
        matrix.get("worker_failures") == [],
        matrix.get("gates") == {
            "all_60_candidate_pools_sealed": True,
            "all_safety_determinism_and_resource_gates": True,
        },
        matrix.get("passed") is True,
        matrix.get("result_digest") == matrix_digest(matrix),
    )):
        return False
    deterministic = {
        "schema_version": SEAL_SCHEMA, "registry_digest": registry_digest,
        "producer_contract_digest": producer_contract_digest,
        "candidate_matrix_digest": matrix["result_digest"],
        "completed_cases": 60, "candidate_pools_sealed": True,
        "comparison_results_opened": False,
        "production_promotion_authorized": False,
    }
    seal = {
        **deterministic, "created_at": datetime.now(timezone.utc).isoformat(),
        "seal_digest": stable_hash(deterministic),
    }
    _atomic_json(output_root / "SEALED.json", seal)
    return True


def _implementation_manifest() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    paths = [Path(__file__).resolve(), *sorted((repository / "src").rglob("*.py"))]
    files = {str(path.relative_to(repository)): file_fingerprint(path) for path in paths}
    return {"files": files, "digest": stable_hash(files)}


def _producer_contract(
    registry: dict[str, Any], manifest_path: Path, physical_rows: int,
) -> dict[str, Any]:
    deterministic = {
        "schema_version": CONTRACT_SCHEMA,
        "registry_digest": FROZEN_REGISTRY_DIGEST,
        "ordered_query_ids": [case["episode_id"] for case in registry["cases_data"]],
        "ordered_query_ids_digest": FROZEN_CASE_ORDER_DIGEST,
        "generation_id": FROZEN_GENERATION_ID,
        "physical_manifest_path": str(manifest_path.resolve()),
        "physical_manifest_sha256": file_fingerprint(manifest_path),
        "physical_rows": physical_rows,
        "proposal_contract_digest": FROZEN_PROPOSAL_CONTRACT_DIGEST,
        "route_quotas": FROZEN_ROUTE_QUOTAS,
        "request": FROZEN_REQUEST,
        "scan_protocol": {
            "execution": "serial fresh spawned process per query",
            "engine": "bounded ordered four-thread legacy-v1 scan",
            "outer_threads": SCAN_THREADS,
            "numba_threads_per_scorer": 1,
            "maximum_in_flight_blocks": SCAN_THREADS,
            "reduction": "strict requested physical block order into unchanged stable route heaps",
            "cold": {"advice": "POSIX_FADV_DONTNEED", "block_rows": 4_096, "order": "forward"},
            "warm_first": {"block_rows": 4_097, "order": "reverse"},
            "warm_second": {"block_rows": 4_093, "order": "forward"},
        },
        "performance_limits": {"cold_seconds": 120.0, "warm_second_seconds": 60.0, "rss_mib": 1_024.0},
        "prefix_policy": "worker recomputes and exactly matches frozen stock and benchmark causal prefixes",
        "implementation_manifest": _implementation_manifest(),
        "real_forward_outcomes_accessed": False,
    }
    return {**deterministic, "contract_digest": stable_hash(deterministic)}


def _checkpoint_valid(
    payload: dict[str, Any], *, case: dict[str, Any], registry_digest: str,
    generation_id: str, proposal_contract_digest: str, producer_contract_digest: str,
    route_quotas: dict[str, int], physical_rows: int,
    expected_query_start_ns: int, expected_latest_eligible_ns: int,
    expected_representation_digest: str,
) -> bool:
    return not validate_truth_blind_candidate_case(
        payload, expected_case=case, registry_digest=registry_digest,
        generation_id=generation_id,
        proposal_contract_digest=proposal_contract_digest,
        producer_contract_digest=producer_contract_digest,
        route_quotas=route_quotas, physical_rows=physical_rows,
        expected_query_start_ns=expected_query_start_ns,
        expected_latest_eligible_ns=expected_latest_eligible_ns,
        expected_representation_digest=expected_representation_digest,
    )


def _advise_cold(path: Path) -> bool:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return False
    with path.open("rb") as handle:
        os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return True


def _candidate_payload(report: Any) -> list[dict[str, Any]]:
    return [{
        "episode_id": row.episode_id, "symbol": row.symbol,
        "cutoff_ns": row.cutoff_ns, "quality_tier": row.quality_tier,
        "lower_bound_hex": row.lower_bound.hex(), "routes": list(row.routes),
        "overflow_fallback": row.overflow_fallback,
    } for row in report.candidates]


def _report_invariants(report: Any) -> dict[str, Any]:
    return {
        "schema_version": report.schema_version,
        "generation_id": report.generation_id,
        "query_episode_id": report.query_episode_id,
        "rows_scanned": report.rows_scanned, "eligible_rows": report.eligible_rows,
        "eligible_main_rows": report.eligible_main_rows,
        "eligible_overflow_rows": report.eligible_overflow_rows,
        "route_counts": dict(report.route_counts),
        "route_quotas": dict(report.route_quotas),
        "candidate_count": len(report.candidates),
        "candidate_digest": report.candidate_digest,
        "result_digest": report.result_digest,
    }


def _worker(
    config_path: str, full_root: str, output_root: str, generation_id: str,
    case: dict[str, Any], route_quotas: dict[str, int], provenance_digest: str,
    producer_contract_digest: str, physical_rows: int,
) -> dict[str, Any]:
    task_started = perf_counter()
    config = load_config(config_path)
    source = source_from_spec(config.datasets["nasdaq"])
    episode = build_episode(
        source, InstrumentKey("nasdaq", str(case["symbol"])),
        str(case["cutoff"]), int(case["lookback"]),
        str(case["representation_version"]),
    )
    if episode.key.id != case["episode_id"]:
        raise ValueError("rebuilt query episode ID differs from frozen registry")
    instrument = InstrumentKey("nasdaq", str(case["symbol"]))
    observed_stock_prefix = asdict(source.causal_prefix_fingerprint(instrument, str(case["cutoff"])))
    observed_benchmark_prefix_raw = source.benchmark_causal_prefix_fingerprint(str(case["cutoff"]))
    observed_benchmark_prefix = (
        asdict(observed_benchmark_prefix_raw) if observed_benchmark_prefix_raw is not None else None
    )
    if observed_stock_prefix != case["stock_prefix"] or observed_benchmark_prefix != case["benchmark_prefix"]:
        raise ValueError("worker causal source prefix differs from frozen registry")
    representation = represent(episode)
    latest = latest_eligible_cutoff(episode, 60)
    query = PackedBoundQuery(
        episode.key.id, episode.key.instrument.source_symbol,
        int(pd.Timestamp(episode.bars.timestamp.iloc[0]).value), int(latest.value),
        representation, ("A", "B"),
    )
    store_root = Path(full_root) / "store"
    loaded = load_packed_generation(
        store_root, generation_id, expected_provenance_digest=provenance_digest,
        verify_content=False, validate_records=False,
    )
    pack_path = loaded.root / "generations" / loaded.generation_id / str(loaded.manifest["rows_file"])
    cold_advice_applied = _advise_cold(pack_path)
    cold = scan_packed_bound_proposals_threaded(
        store_root, generation_id, query, route_quotas=route_quotas,
        block_rows=4_096, block_order="forward", threads=SCAN_THREADS,
        verify_content=False, expected_provenance_digest=provenance_digest,
    )
    cold_task_seconds = perf_counter() - task_started
    warm_first = scan_packed_bound_proposals_threaded(
        store_root, generation_id, query, route_quotas=route_quotas,
        block_rows=4_097, block_order="reverse", threads=SCAN_THREADS,
        verify_content=False, expected_provenance_digest=provenance_digest,
    )
    warm_second = scan_packed_bound_proposals_threaded(
        store_root, generation_id, query, route_quotas=route_quotas,
        block_rows=4_093, block_order="forward", threads=SCAN_THREADS,
        verify_content=False, expected_provenance_digest=provenance_digest,
    )
    candidates = _candidate_payload(cold)
    candidate_ids = [row["episode_id"] for row in candidates]
    violations = {
        "duplicates": len(candidate_ids) - len(set(candidate_ids)),
        "future": sum(row["cutoff_ns"] > query.latest_eligible_ns for row in candidates),
        "same_symbol_overlap": sum(
            row["symbol"] == query.symbol and row["cutoff_ns"] >= query.query_start_ns
            for row in candidates
        ),
        "tier": sum(row["quality_tier"] not in query.quality_tiers for row in candidates),
    }
    invariants = [_report_invariants(value) for value in (cold, warm_first, warm_second)]
    repeated = invariants[0] == invariants[1] == invariants[2]
    peak_rss = max(
        cold.peak_rss_mb, warm_first.peak_rss_mb, warm_second.peak_rss_mb,
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1_024,
    )
    gates = {
        "cold_advice_supported_and_applied": cold_advice_applied,
        "three_scan_digest_and_block_order_invariance": repeated,
        "internal_eligible_row_accounting": cold.eligible_rows == cold.eligible_main_rows + cold.eligible_overflow_rows,
        "physical_row_accounting": cold.rows_scanned == physical_rows,
        "frozen_route_quotas": dict(cold.route_quotas) == route_quotas,
        "zero_temporal_overlap_tier_duplicate_errors": not any(violations.values()),
        "measurements_finite_nonnegative_and_task_contains_cold": all(
            isfinite(value) and value >= 0
            for value in (
                cold.elapsed_seconds, warm_first.elapsed_seconds,
                warm_second.elapsed_seconds, cold_task_seconds, peak_rss,
            )
        ) and cold_task_seconds >= cold.elapsed_seconds,
        "cold_scan_at_most_120_seconds": cold.elapsed_seconds <= 120.0,
        "second_warm_scan_at_most_60_seconds": warm_second.elapsed_seconds <= 60.0,
        "rss_at_most_1024_mib": peak_rss <= 1_024.0,
    }
    payload: dict[str, Any] = {
        "schema_version": CASE_SCHEMA,
        "registry_case_id": case["case_id"], "registry_digest": case["registry_digest"],
        "query_episode_id": query.episode_id, "query_symbol": query.symbol,
        "query_start_ns": query.query_start_ns, "latest_eligible_ns": query.latest_eligible_ns,
        "query_stock_prefix": case["stock_prefix"],
        "query_benchmark_prefix": case["benchmark_prefix"],
        "query_representation_digest": representation_input_digest(representation),
        "generation_id": generation_id,
        "proposal_contract_digest": packed_bound_search_contract()["digest"],
        "producer_contract_digest": producer_contract_digest,
        "route_quotas": route_quotas, "scan_invariants": invariants,
        "candidates": candidates,
        "candidate_digest_reconstructed": bound_proposal_candidate_digest(cold.candidates),
        "violations": violations, "cold_advice_applied": cold_advice_applied,
        "cold_seconds": cold.elapsed_seconds, "cold_task_seconds": cold_task_seconds,
        "warm_first_seconds": warm_first.elapsed_seconds,
        "warm_second_seconds": warm_second.elapsed_seconds,
        "peak_rss_mb": peak_rss, "gates": gates, "passed": all(gates.values()),
        "real_forward_outcomes_accessed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    payload["result_digest"] = case_digest(payload)
    payload["checkpoint_integrity_digest"] = checkpoint_integrity_digest(payload)
    validation_failures = validate_truth_blind_candidate_case(
        payload, expected_case=case, registry_digest=case["registry_digest"],
        generation_id=generation_id,
        proposal_contract_digest=packed_bound_search_contract()["digest"],
        producer_contract_digest=producer_contract_digest,
        route_quotas=route_quotas, physical_rows=physical_rows,
        expected_query_start_ns=query.query_start_ns,
        expected_latest_eligible_ns=query.latest_eligible_ns,
        expected_representation_digest=representation_input_digest(representation),
    )
    if validation_failures:
        raise ValueError(f"generated candidate evidence differs:{validation_failures}")
    _atomic_json(Path(output_root) / "cases" / f"{query.episode_id}.json", payload)
    return {
        "case_id": case["case_id"], "query_episode_id": query.episode_id,
        "result_digest": payload["result_digest"], "passed": payload["passed"],
        "cold_seconds": cold.elapsed_seconds, "warm_second_seconds": warm_second.elapsed_seconds,
        "peak_rss_mb": peak_rss,
    }


def _render(path: Path, payload: dict[str, Any]) -> None:
    status = "PASS" if payload["passed"] else "FAIL"
    path.write_text(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>M04R-11 truth-blind candidate producer</title><style>"
        "body{font-family:system-ui;max-width:1200px;margin:2rem auto}"
        "pre{white-space:pre-wrap}.pass{color:#075}.fail{color:#a20}</style></head><body>"
        f"<h1 class=\"{status.lower()}\">Candidate producer: {status}</h1>"
        "<p>All candidate pools are sealed without reading exact-neighbour results or outcomes.</p><pre>"
        f"{escape(json.dumps(payload, indent=2, sort_keys=True))}</pre></body></html>"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    preliminary_config = load_config(args.config)
    expected_output_root = preliminary_config.artifact_dir / "m04r11" / "candidate-pools-v1"
    if args.output_root.resolve() != expected_output_root.resolve():
        raise ValueError("one-shot candidate output path differs")
    output_resolved = args.output_root.resolve()
    for protected in (args.registry.parent.resolve(), args.full_root.resolve()):
        if output_resolved == protected or output_resolved.is_relative_to(protected) or protected.is_relative_to(output_resolved):
            raise ValueError("candidate output root overlaps a frozen input root")
    if any((args.output_root / name).exists() for name in (
        "authority-contract.json", "authority-matrix.json",
    )):
        raise ValueError("candidate output root contains authority artifacts")
    config = preliminary_config
    source = source_from_spec(config.datasets["nasdaq"])
    failures = validate_m04r_validation_registry(source, args.registry.parent)
    if failures:
        raise ValueError(f"sealed registry differs: {failures}")
    registry = json.loads(args.registry.read_text())
    cases = list(registry["cases_data"])
    search_contract = registry["search_contract"]
    if not all((
        len(cases) == 60,
        registry.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
        stable_hash([case["episode_id"] for case in cases]) == FROZEN_CASE_ORDER_DIGEST,
        search_contract.get("packed_generation_id") == FROZEN_GENERATION_ID,
        search_contract.get("proposal_contract_digest") == FROZEN_PROPOSAL_CONTRACT_DIGEST,
        packed_bound_search_contract()["digest"] == FROZEN_PROPOSAL_CONTRACT_DIGEST,
        search_contract.get("fast_route_quotas") == FROZEN_ROUTE_QUOTAS,
        search_contract.get("request") == FROZEN_REQUEST,
        search_contract.get("real_forward_outcomes_accessed") is False,
    )):
        raise ValueError("frozen candidate contract differs")
    generation_id = str(search_contract["packed_generation_id"])
    loaded = load_packed_generation(
        args.full_root / "store", generation_id,
        expected_provenance_digest=str(search_contract["packed_provenance_digest"]),
        verify_content=True, validate_records=False,
    )
    if (args.full_root / "store" / "active.json").exists():
        raise ValueError("frozen candidate generation unexpectedly activated")
    route_quotas = {key: int(value) for key, value in search_contract["fast_route_quotas"].items()}
    proposal_contract_digest = packed_bound_search_contract()["digest"]
    manifest_path = Path(str(search_contract["packed_generation_manifest_path"]))
    if not all((
        file_fingerprint(manifest_path) == search_contract["packed_generation_manifest_sha256"],
        loaded.manifest.get("manifest_digest") == FROZEN_GENERATION_ID,
        loaded.manifest.get("provenance_digest") == search_contract["packed_provenance_digest"],
        loaded.manifest.get("pack_contract_digest") == search_contract["packed_bound_contract_digest"],
    )):
        raise ValueError("frozen physical manifest binding differs")
    producer_contract = _producer_contract(
        registry, manifest_path, len(loaded.rows) + len(loaded.overflow),
    )
    contract_path = args.output_root / "candidate-contract.json"
    existing_seal_path = args.output_root / "SEALED.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != producer_contract:
            raise ValueError("existing candidate producer contract differs")
    else:
        if existing_seal_path.exists() or any((args.output_root / "cases").glob("*.json")):
            raise ValueError("candidate artifacts exist without their producer contract")
        _atomic_json(contract_path, producer_contract)
    if existing_seal_path.exists():
        existing_seal = json.loads(existing_seal_path.read_text())
        existing_matrix = json.loads((args.output_root / "candidate-matrix.json").read_text())
        existing_cases: list[dict[str, Any]] = []
        for raw in cases:
            episode = build_episode(
                source, InstrumentKey("nasdaq", str(raw["symbol"])),
                str(raw["cutoff"]), int(raw["lookback"]),
                str(raw["representation_version"]),
            )
            candidate = json.loads((
                args.output_root / "cases" / f"{raw['episode_id']}.json"
            ).read_text())
            case = {**raw, "registry_digest": registry["registry_digest"]}
            if not _checkpoint_valid(
                candidate, case=case, registry_digest=registry["registry_digest"],
                generation_id=generation_id,
                proposal_contract_digest=proposal_contract_digest,
                producer_contract_digest=producer_contract["contract_digest"],
                route_quotas=route_quotas,
                physical_rows=int(producer_contract["physical_rows"]),
                expected_query_start_ns=int(episode.bars.timestamp.iloc[0].value),
                expected_latest_eligible_ns=int(latest_eligible_cutoff(episode, 60).value),
                expected_representation_digest=representation_input_digest(
                    represent(episode),
                ),
            ) or candidate.get("passed") is not True:
                raise ValueError("existing sealed candidate checkpoint differs")
            existing_cases.append(candidate)
        expected_existing_gates = {
            "all_60_candidate_pools_sealed": (
                len(existing_cases) == 60
                and existing_matrix.get("worker_failures") == []
            ),
            "all_safety_determinism_and_resource_gates": (
                len(existing_cases) == 60
                and all(row["passed"] for row in existing_cases)
            ),
        }
        if not all((
            existing_matrix.get("schema_version") == MATRIX_SCHEMA,
            existing_matrix.get("registry_digest") == registry["registry_digest"],
            existing_matrix.get("producer_contract_digest")
            == producer_contract["contract_digest"],
            existing_matrix.get("proposal_contract_digest")
            == proposal_contract_digest,
            existing_matrix.get("generation_id") == generation_id,
            existing_matrix.get("execution")
            == "serial isolated spawned query processes",
            existing_matrix.get("completed_cases") == 60,
            existing_matrix.get("case_result_digests")
            == [row["result_digest"] for row in existing_cases],
            existing_matrix.get("maximum_cold_seconds")
            == max(row["cold_seconds"] for row in existing_cases),
            existing_matrix.get("maximum_second_warm_seconds")
            == max(row["warm_second_seconds"] for row in existing_cases),
            existing_matrix.get("maximum_peak_rss_mb")
            == max(row["peak_rss_mb"] for row in existing_cases),
            existing_matrix.get("gates") == expected_existing_gates,
            existing_matrix.get("passed") is True,
            existing_matrix.get("result_digest") == matrix_digest(existing_matrix),
            existing_seal.get("schema_version") == SEAL_SCHEMA,
            existing_seal.get("registry_digest") == registry["registry_digest"],
            existing_seal.get("producer_contract_digest") == producer_contract["contract_digest"],
            existing_seal.get("candidate_matrix_digest") == existing_matrix.get("result_digest"),
            existing_seal.get("candidate_pools_sealed") is True,
            existing_seal.get("completed_cases") == 60,
            existing_seal.get("comparison_results_opened") is False,
            existing_seal.get("production_promotion_authorized") is False,
            existing_seal.get("seal_digest") == _seal_digest(existing_seal),
        )):
            raise ValueError("existing candidate seal is invalid")
        print("candidate pools are already sealed; refusing to regenerate one-shot evidence")
        return 0
    results: dict[str, dict[str, Any]] = {}
    worker_failures: list[str] = []
    started = perf_counter()
    for raw in cases:
        case = {**raw, "registry_digest": registry["registry_digest"]}
        query_id = str(case["episode_id"])
        checkpoint_episode = build_episode(
            source, InstrumentKey("nasdaq", str(case["symbol"])),
            str(case["cutoff"]), int(case["lookback"]),
            str(case["representation_version"]),
        )
        checkpoint_representation = represent(checkpoint_episode)
        checkpoint_start_ns = int(pd.Timestamp(checkpoint_episode.bars.timestamp.iloc[0]).value)
        checkpoint_latest_ns = int(latest_eligible_cutoff(checkpoint_episode, 60).value)
        case_path = args.output_root / "cases" / f"{query_id}.json"
        if case_path.is_file():
            existing = json.loads(case_path.read_text())
            if _checkpoint_valid(
                existing, case=case, registry_digest=registry["registry_digest"],
                generation_id=generation_id,
                proposal_contract_digest=proposal_contract_digest,
                producer_contract_digest=producer_contract["contract_digest"],
                route_quotas=route_quotas,
                physical_rows=int(producer_contract["physical_rows"]),
                expected_query_start_ns=checkpoint_start_ns,
                expected_latest_eligible_ns=checkpoint_latest_ns,
                expected_representation_digest=representation_input_digest(
                    checkpoint_representation,
                ),
            ):
                results[query_id] = {
                    "case_id": case["case_id"], "query_episode_id": query_id,
                    "result_digest": existing["result_digest"], "passed": existing["passed"],
                    "cold_seconds": existing["cold_seconds"],
                    "warm_second_seconds": existing["warm_second_seconds"],
                    "peak_rss_mb": existing["peak_rss_mb"],
                }
                print(f"[candidate:{case['case_id']}] validated checkpoint", flush=True)
                continue
        context = multiprocessing.get_context("spawn")
        try:
            # Fresh and serial prevents shared cache advice and inherited high-water
            # RSS from corrupting the cold/warm measurements.
            with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
                result = executor.submit(
                    _worker, str(args.config), str(args.full_root), str(args.output_root),
                    generation_id, case, route_quotas,
                    str(search_contract["packed_provenance_digest"]),
                    producer_contract["contract_digest"],
                    int(producer_contract["physical_rows"]),
                ).result()
            results[query_id] = result
            print(
                f"[candidate:{case['case_id']}] {'PASS' if result['passed'] else 'FAIL'} "
                f"cold={result['cold_seconds']:.2f}s warm2={result['warm_second_seconds']:.2f}s",
                flush=True,
            )
        except Exception as exc:
            worker_failures.append(f"{case['case_id']}:{type(exc).__name__}:{exc}")
            print(f"[candidate:{case['case_id']}] ERROR {exc}", flush=True)
    ordered = [results[str(case["episode_id"])] for case in cases if str(case["episode_id"]) in results]
    producer_gates = {
        "all_60_candidate_pools_sealed": len(ordered) == 60 and not worker_failures,
        "all_safety_determinism_and_resource_gates": len(ordered) == 60 and all(row["passed"] for row in ordered),
    }
    deterministic = {
        "schema_version": MATRIX_SCHEMA, "registry_digest": registry["registry_digest"],
        "producer_contract_digest": producer_contract["contract_digest"],
        "proposal_contract_digest": packed_bound_search_contract()["digest"],
        "generation_id": loaded.generation_id, "execution": "serial isolated spawned query processes",
        "completed_cases": len(ordered),
        "maximum_cold_seconds": max((row["cold_seconds"] for row in ordered), default=None),
        "maximum_second_warm_seconds": max((row["warm_second_seconds"] for row in ordered), default=None),
        "maximum_peak_rss_mb": max((row["peak_rss_mb"] for row in ordered), default=None),
        "case_result_digests": [row["result_digest"] for row in ordered],
        "worker_failures": worker_failures, "gates": producer_gates,
        "passed": all(producer_gates.values()), "real_forward_outcomes_accessed": False,
    }
    payload = {**deterministic, "elapsed_seconds": perf_counter() - started,
               "created_at": datetime.now(timezone.utc).isoformat()}
    payload["result_digest"] = matrix_digest(payload)
    _atomic_json(args.output_root / "candidate-matrix.json", payload)
    _render(args.output_root / "candidate-matrix.html", payload)
    sealed = _write_success_seal(
        args.output_root, payload,
        registry_digest=registry["registry_digest"],
        producer_contract_digest=producer_contract["contract_digest"],
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if sealed else 2


if __name__ == "__main__":
    raise SystemExit(main())
