"""Independently reconstruct the sealed M04R-11 candidate comparison."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from html import escape
import json
from math import isfinite
from pathlib import Path
from typing import Any

import pandas as pd

from market_analogues.adapters import file_fingerprint, source_from_spec
from market_analogues.config import load_config
from market_analogues.episodes import build_episode
from market_analogues.m04r_validation_registry import validate_m04r_validation_registry
from market_analogues.m04r_candidate_evidence import validate_truth_blind_candidate_case
from market_analogues.packed_bound_search import (
    BoundProposal, bound_proposal_candidate_digest, packed_bound_search_contract,
)
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.representation import represent, representation_input_digest
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import InstrumentKey, stable_hash


SCHEMA = "candidate-recall-comparison-verification-v1"
CANDIDATE_CASE_OMITTED = {
    "created_at", "cold_seconds", "warm_first_seconds", "warm_second_seconds",
    "cold_task_seconds", "peak_rss_mb", "result_digest",
    "checkpoint_integrity_digest",
}
CANDIDATE_MATRIX_OMITTED = {"created_at", "elapsed_seconds", "result_digest"}
AUTHORITY_CASE_OMITTED = {
    "created_at", "proposal_seconds", "amortized_proposal_seconds",
    "exact_seconds", "final_exact_seconds", "frontier_attempt_measurements",
    "peak_rss_mb", "result_digest", "checkpoint_integrity_digest",
}
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


def _digest_without(payload: dict[str, Any], omitted: set[str]) -> str:
    return stable_hash({key: value for key, value in payload.items() if key not in omitted})


def _seal_digest(payload: dict[str, Any]) -> str:
    return _digest_without(payload, {"created_at", "seal_digest"})


def _checkpoint_integrity_digest(payload: dict[str, Any]) -> str:
    return _digest_without(payload, {"created_at", "checkpoint_integrity_digest"})


def _candidate_digest(rows: list[dict[str, Any]]) -> str:
    return bound_proposal_candidate_digest(BoundProposal(
        str(row["episode_id"]), str(row["symbol"]), int(row["cutoff_ns"]),
        str(row["quality_tier"]), float.fromhex(str(row["lower_bound_hex"])),
        tuple(str(value) for value in row["routes"]), bool(row["overflow_fallback"]),
    ) for row in rows)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    expected_paths = {
        "candidate": config.artifact_dir / "m04r11" / "candidate-pools-v1",
        "authority": config.artifact_dir / "m04r11" / "authorities-sealed-v4",
        "comparison": config.artifact_dir / "m04r11" / "candidate-comparison-v1",
        "output": config.artifact_dir / "m04r11" / "candidate-comparison-verification-v1",
    }
    if not all((
        args.candidate_root.resolve() == expected_paths["candidate"].resolve(),
        args.authority_root.resolve() == expected_paths["authority"].resolve(),
        args.comparison_root.resolve() == expected_paths["comparison"].resolve(),
        args.output_root.resolve() == expected_paths["output"].resolve(),
    )):
        raise ValueError("candidate verification input or output path differs")
    source = source_from_spec(config.datasets["nasdaq"])
    failures = list(validate_m04r_validation_registry(source, args.registry.parent))
    registry = json.loads(args.registry.read_text())
    candidate_matrix = json.loads((args.candidate_root / "candidate-matrix.json").read_text())
    candidate_seal = json.loads((args.candidate_root / "SEALED.json").read_text())
    authority_matrix = json.loads((args.authority_root / "authority-matrix.json").read_text())
    authority_seal = json.loads((args.authority_root / "SEALED.json").read_text())
    comparison = json.loads((args.comparison_root / "candidate-comparison.json").read_text())
    comparison_seal = json.loads((args.comparison_root / "SEALED.json").read_text())
    results_opened = json.loads((args.comparison_root / "RESULTS_OPENED.json").read_text())
    producer_contract = json.loads((args.candidate_root / "candidate-contract.json").read_text())
    if candidate_matrix.get("result_digest") != _digest_without(candidate_matrix, CANDIDATE_MATRIX_OMITTED):
        failures.append("candidate matrix digest differs")
    if candidate_seal.get("seal_digest") != _seal_digest(candidate_seal):
        failures.append("candidate seal digest differs")
    if authority_seal.get("seal_digest") != _seal_digest(authority_seal):
        failures.append("authority seal digest differs")
    if comparison.get("result_digest") != _digest_without(comparison, {"created_at", "result_digest"}):
        failures.append("comparison digest differs")
    if comparison_seal.get("seal_digest") != _seal_digest(comparison_seal):
        failures.append("comparison seal digest differs")
    if not all((
        registry.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
        stable_hash([case["episode_id"] for case in registry["cases_data"]]) == FROZEN_CASE_ORDER_DIGEST,
        producer_contract.get("schema_version")
        == "candidate-recall-producer-contract-v2",
        producer_contract.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
        producer_contract.get("ordered_query_ids")
        == [case["episode_id"] for case in registry["cases_data"]],
        producer_contract.get("generation_id") == FROZEN_GENERATION_ID,
        producer_contract.get("proposal_contract_digest") == FROZEN_PROPOSAL_CONTRACT_DIGEST,
        producer_contract.get("route_quotas") == FROZEN_ROUTE_QUOTAS,
        producer_contract.get("request") == FROZEN_REQUEST,
        producer_contract.get("ordered_query_ids_digest") == FROZEN_CASE_ORDER_DIGEST,
        registry.get("search_contract", {}).get("fast_route_quotas") == FROZEN_ROUTE_QUOTAS,
        registry.get("search_contract", {}).get("request") == FROZEN_REQUEST,
        producer_contract.get("contract_digest") == stable_hash({
            key: value for key, value in producer_contract.items() if key != "contract_digest"
        }),
        producer_contract.get("real_forward_outcomes_accessed") is False,
        candidate_seal.get("producer_contract_digest") == producer_contract.get("contract_digest"),
        candidate_seal.get("candidate_matrix_digest") == candidate_matrix.get("result_digest"),
        candidate_seal.get("completed_cases") == 60,
        candidate_seal.get("candidate_pools_sealed") is True,
        candidate_seal.get("comparison_results_opened") is False,
        comparison_seal.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
        comparison_seal.get("candidate_seal_digest") == candidate_seal.get("seal_digest"),
        comparison_seal.get("authority_seal_digest") == authority_seal.get("seal_digest"),
        comparison_seal.get("comparison_digest") == comparison.get("result_digest"),
        comparison_seal.get("results_opened_marker_digest") == results_opened.get("result_digest"),
        comparison_seal.get("candidate_results_opened") is True,
        comparison_seal.get("comparison_gate_passed") is comparison.get("passed"),
        comparison_seal.get("production_promotion_authorized") is False,
        results_opened.get("result_digest") == _digest_without(results_opened, {"created_at", "result_digest"}),
        results_opened.get("candidate_matrix_digest") == candidate_matrix.get("result_digest"),
        results_opened.get("candidate_seal_digest") == candidate_seal.get("seal_digest"),
        results_opened.get("producer_contract_digest") == producer_contract.get("contract_digest"),
    )):
        failures.append("frozen identity, producer contract, or comparison seal differs")
    repository = Path(__file__).resolve().parents[2]
    implementation_files = producer_contract.get("implementation_manifest", {}).get("files", {})
    observed_implementation_files = {
        relative: file_fingerprint(repository / relative)
        for relative in implementation_files
        if (repository / relative).is_file()
    }
    if not all((
        observed_implementation_files == implementation_files,
        stable_hash(observed_implementation_files)
        == producer_contract.get("implementation_manifest", {}).get("digest"),
        comparison.get("comparison_implementation_sha256")
        == file_fingerprint(repository / "experiments/m04r/compare_m04r11_candidate_matrix.py"),
    )):
        failures.append("producer or comparator implementation binding differs")
    search_contract = registry["search_contract"]
    generation_id = str(search_contract["packed_generation_id"])
    try:
        loaded = load_packed_generation(
            args.full_root / "store", generation_id,
            expected_provenance_digest=str(search_contract["packed_provenance_digest"]),
            verify_content=True, validate_records=False,
        )
    except Exception as exc:
        failures.append(f"physical generation invalid:{type(exc).__name__}:{exc}")
        loaded = None
    expected_cases: list[dict[str, Any]] = []
    expected_case_digests: list[str] = []
    cold: list[float] = []
    warm: list[float] = []
    rss: list[float] = []
    for registry_case in registry["cases_data"]:
        query_id = str(registry_case["episode_id"])
        candidate = json.loads((args.candidate_root / "cases" / f"{query_id}.json").read_text())
        authority = json.loads((args.authority_root / "cases" / f"{query_id}.json").read_text())
        episode = build_episode(
            source, InstrumentKey("nasdaq", str(registry_case["symbol"])),
            str(registry_case["cutoff"]), int(registry_case["lookback"]),
            str(registry_case["representation_version"]),
        )
        representation_digest = representation_input_digest(represent(episode))
        observed_stock_prefix = asdict(source.causal_prefix_fingerprint(
            InstrumentKey("nasdaq", str(registry_case["symbol"])), str(registry_case["cutoff"]),
        ))
        observed_benchmark_raw = source.benchmark_causal_prefix_fingerprint(str(registry_case["cutoff"]))
        observed_benchmark_prefix = asdict(observed_benchmark_raw) if observed_benchmark_raw is not None else None
        latest_ns = int(latest_eligible_cutoff(episode, 60).value)
        invariants = candidate.get("scan_invariants") or []
        rows = candidate.get("candidates") or []
        rebuilt_candidate_digest = _candidate_digest(rows)
        candidate_ids = [str(row["episode_id"]) for row in rows]
        truth_ids = [str(row["episode_id"]) for row in authority["matches"]]
        retained = [value for value in truth_ids if value in set(candidate_ids)]
        violations = {
            "duplicates": len(candidate_ids) - len(set(candidate_ids)),
            "future": sum(int(row["cutoff_ns"]) > latest_ns for row in rows),
            "same_symbol_overlap": sum(
                row["symbol"] == registry_case["symbol"]
                and int(row["cutoff_ns"]) >= int(pd.Timestamp(episode.bars.timestamp.iloc[0]).value)
                for row in rows
            ),
            "tier": sum(row["quality_tier"] not in ("A", "B") for row in rows),
        }
        repeated = len(invariants) == 3 and invariants[0] == invariants[1] == invariants[2]
        first = invariants[0] if invariants else {}
        expected_candidate_gates = {
            "cold_advice_supported_and_applied": candidate.get("cold_advice_applied") is True,
            "three_scan_digest_and_block_order_invariance": repeated,
            "internal_eligible_row_accounting": bool(invariants) and int(invariants[0]["eligible_rows"]) == int(invariants[0]["eligible_main_rows"]) + int(invariants[0]["eligible_overflow_rows"]),
            "physical_row_accounting": bool(invariants) and int(invariants[0]["rows_scanned"]) == int(producer_contract["physical_rows"]),
            "frozen_route_quotas": candidate.get("route_quotas") == search_contract["fast_route_quotas"],
            "zero_temporal_overlap_tier_duplicate_errors": not any(violations.values()),
            "measurements_finite_nonnegative_and_task_contains_cold": all(
                isfinite(float(candidate[key])) and float(candidate[key]) >= 0
                for key in ("cold_seconds", "warm_first_seconds", "warm_second_seconds", "cold_task_seconds", "peak_rss_mb")
            ) and float(candidate["cold_task_seconds"]) >= float(candidate["cold_seconds"]),
            "cold_scan_at_most_120_seconds": isfinite(float(candidate["cold_seconds"])) and 0 <= float(candidate["cold_seconds"]) <= 120.0,
            "second_warm_scan_at_most_60_seconds": isfinite(float(candidate["warm_second_seconds"])) and 0 <= float(candidate["warm_second_seconds"]) <= 60.0,
            "rss_at_most_1024_mib": isfinite(float(candidate["peak_rss_mb"])) and 0 <= float(candidate["peak_rss_mb"]) <= 1_024.0,
        }
        scan_digest_valid = all(
            invariant.get("result_digest") == stable_hash({
                "schema_version": invariant["schema_version"],
                "contract_digest": packed_bound_search_contract()["digest"],
                "generation_id": invariant["generation_id"],
                "query_episode_id": invariant["query_episode_id"],
                "rows_scanned": invariant["rows_scanned"],
                "eligible_rows": invariant["eligible_rows"],
                "eligible_main_rows": invariant["eligible_main_rows"],
                "eligible_overflow_rows": invariant["eligible_overflow_rows"],
                "route_counts": invariant["route_counts"], "route_quotas": invariant["route_quotas"],
                "candidate_digest": invariant["candidate_digest"],
                "real_forward_outcomes_accessed": False,
            }) for invariant in invariants
        )
        local_failures = []
        local_failures.extend(validate_truth_blind_candidate_case(
            candidate, expected_case=registry_case,
            registry_digest=FROZEN_REGISTRY_DIGEST,
            generation_id=FROZEN_GENERATION_ID,
            proposal_contract_digest=FROZEN_PROPOSAL_CONTRACT_DIGEST,
            producer_contract_digest=producer_contract["contract_digest"],
            route_quotas=producer_contract["route_quotas"],
            physical_rows=int(producer_contract["physical_rows"]),
            expected_query_start_ns=int(episode.bars.timestamp.iloc[0].value),
            expected_latest_eligible_ns=latest_ns,
            expected_representation_digest=representation_digest,
        ))
        bounds = [float.fromhex(str(row["lower_bound_hex"])) for row in rows]
        route_counts = {
            route: sum(route in row["routes"] for row in rows)
            for route in producer_contract["route_quotas"]
        }
        metadata_and_order_valid = all((
            len(candidate_ids) == len(set(candidate_ids)),
            all(len(value) == 24 and value == value.lower() for value in candidate_ids),
            list(zip(bounds, candidate_ids)) == sorted(zip(bounds, candidate_ids)),
            all(
                row["routes"] == sorted(set(row["routes"]))
                and bool(row["routes"])
                and set(row["routes"]).issubset(producer_contract["route_quotas"])
                for row in rows
            ),
            bool(invariants) and len(rows) == int(first["candidate_count"]),
            bool(invariants) and route_counts == first["route_counts"],
        ))
        if not all((
            episode.key.id == query_id,
            candidate.get("query_representation_digest") == representation_digest,
            observed_stock_prefix == registry_case["stock_prefix"],
            observed_benchmark_prefix == registry_case["benchmark_prefix"],
            candidate.get("query_stock_prefix") == registry_case["stock_prefix"],
            candidate.get("query_benchmark_prefix") == registry_case["benchmark_prefix"],
            candidate.get("result_digest") == _digest_without(candidate, CANDIDATE_CASE_OMITTED),
            candidate.get("checkpoint_integrity_digest") == _checkpoint_integrity_digest(candidate),
            candidate.get("producer_contract_digest") == producer_contract["contract_digest"],
            authority.get("result_digest") == _digest_without(authority, AUTHORITY_CASE_OMITTED),
            rebuilt_candidate_digest == candidate.get("candidate_digest_reconstructed"),
            bool(invariants) and rebuilt_candidate_digest == invariants[0].get("candidate_digest"),
            scan_digest_valid, candidate.get("violations") == violations,
            metadata_and_order_valid,
            candidate.get("gates") == expected_candidate_gates,
            candidate.get("passed") is all(expected_candidate_gates.values()),
            bool(invariants) and int(invariants[0]["eligible_rows"]) == int(authority["certificate"]["eligible_candidates"]),
        )):
            local_failures.append("candidate/query/authority reconstruction differs")
        recall = len(retained) / 20.0
        if candidate.get("passed") is not True:
            local_failures.append("candidate safety/determinism/performance gate failed")
        if recall < .95:
            local_failures.append("candidate recall below 19/20")
        authority_rows = {
            str(row["query_episode_id"]): row for row in authority_matrix["cases"]
        }
        authority_row = authority_rows.get(query_id) or {}
        if not all((
            authority_row.get("registry_case_id") == registry_case["case_id"],
            authority_row.get("authority_digest") == authority.get("result_digest"),
            authority_row.get("certificate_digest") == authority.get("certificate_digest"),
        )):
            local_failures.append("authority matrix/case binding differs")
        expected = {
            "registry_case_id": registry_case["case_id"], "query_episode_id": query_id,
            "candidate_case_digest": candidate["result_digest"],
            "authority_case_digest": authority["result_digest"],
            "candidate_count": len(candidate_ids), "retained_count": len(retained),
            "recall_at_20": recall,
            "missing_authority_episode_ids": [value for value in truth_ids if value not in set(retained)],
            "failures": local_failures, "passed": not local_failures,
        }
        expected_cases.append(expected)
        expected_case_digests.append(candidate["result_digest"])
        cold.append(float(candidate["cold_seconds"]))
        warm.append(float(candidate["warm_second_seconds"]))
        rss.append(float(candidate["peak_rss_mb"]))
        failures.extend(f"{registry_case['case_id']}:{value}" for value in local_failures)
    recalls = [row["recall_at_20"] for row in expected_cases]
    expected_comparison_failures = [
        f"{row['registry_case_id']}:{value}"
        for row in expected_cases for value in row["failures"]
    ]
    expected_comparison_gates = {
        "all_60_comparisons_valid": len(expected_cases) == 60 and not expected_comparison_failures,
        "every_case_at_least_19_of_20": len(expected_cases) == 60 and min(recalls, default=0) >= .95,
        "mean_recall_at_least_99_percent": len(expected_cases) == 60 and sum(recalls) / 60 >= .99,
    }
    expected_producer_gates = {
        "all_60_candidate_pools_sealed": len(expected_cases) == 60 and candidate_matrix.get("worker_failures") == [],
        "all_safety_determinism_and_resource_gates": len(expected_cases) == 60 and all(
            json.loads((args.candidate_root / "cases" / f"{row['query_episode_id']}.json").read_text())["passed"]
            for row in expected_cases
        ),
    }
    if not all((
        loaded is not None and loaded.generation_id == generation_id,
        loaded is not None and int(producer_contract["physical_rows"]) == len(loaded.rows) + len(loaded.overflow),
        file_fingerprint(Path(producer_contract["physical_manifest_path"])) == producer_contract["physical_manifest_sha256"],
        producer_contract.get("scan_protocol") == {
            "execution": "serial fresh spawned process per query",
            "engine": "bounded ordered four-thread legacy-v1 scan",
            "outer_threads": 4,
            "numba_threads_per_scorer": 1,
            "maximum_in_flight_blocks": 4,
            "reduction": "strict requested physical block order into unchanged stable route heaps",
            "cold": {"advice": "POSIX_FADV_DONTNEED", "block_rows": 4_096, "order": "forward"},
            "warm_first": {"block_rows": 4_097, "order": "reverse"},
            "warm_second": {"block_rows": 4_093, "order": "forward"},
        },
        producer_contract.get("performance_limits") == {"cold_seconds": 120.0, "warm_second_seconds": 60.0, "rss_mib": 1_024.0},
        candidate_matrix.get("case_result_digests") == expected_case_digests,
        candidate_matrix.get("producer_contract_digest") == producer_contract["contract_digest"],
        candidate_matrix.get("completed_cases") == 60,
        candidate_matrix.get("maximum_cold_seconds") == max(cold),
        candidate_matrix.get("maximum_second_warm_seconds") == max(warm),
        candidate_matrix.get("maximum_peak_rss_mb") == max(rss),
        candidate_matrix.get("gates") == expected_producer_gates,
        candidate_matrix.get("passed") is all(expected_producer_gates.values()),
        comparison.get("cases") == expected_cases,
        comparison.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
        comparison.get("candidate_matrix_digest") == candidate_matrix.get("result_digest"),
        comparison.get("candidate_seal_digest") == candidate_seal.get("seal_digest"),
        comparison.get("authority_matrix_digest") == authority_matrix.get("result_digest"),
        comparison.get("authority_seal_digest") == authority_seal.get("seal_digest"),
        comparison.get("results_opened_marker_digest") == results_opened.get("result_digest"),
        comparison.get("failures") == expected_comparison_failures,
        comparison.get("gates") == expected_comparison_gates,
        comparison.get("passed") is all(expected_comparison_gates.values()),
        comparison.get("minimum_recall_at_20") == min(recalls),
        comparison.get("mean_recall_at_20") == sum(recalls) / 60,
        comparison.get("perfect_recall_cases") == sum(value == 1.0 for value in recalls),
        comparison.get("candidate_results_opened") is True,
        comparison.get("producer_contract_digest") == producer_contract["contract_digest"],
        comparison.get("production_promotion_authorized") is False,
        comparison.get("real_forward_outcomes_accessed") is False,
    )):
        failures.append("aggregate producer/comparison reconstruction differs")
    unique = sorted(set(failures))
    deterministic = {
        "schema_version": SCHEMA, "registry_digest": registry["registry_digest"],
        "candidate_matrix_digest": candidate_matrix.get("result_digest"),
        "candidate_seal_digest": candidate_seal.get("seal_digest"),
        "authority_matrix_digest": authority_matrix.get("result_digest"),
        "authority_seal_digest": authority_seal.get("seal_digest"),
        "comparison_digest": comparison.get("result_digest"),
        "comparison_seal_digest": comparison_seal.get("seal_digest"),
        "producer_contract_digest": producer_contract.get("contract_digest"),
        "verification_implementation_sha256": file_fingerprint(Path(__file__)),
        "verified_cases": len(expected_cases), "comparison_gate_passed": comparison.get("passed"),
        "failures": unique, "passed": not unique,
        "real_forward_outcomes_accessed": False,
    }
    payload = {**deterministic, "created_at": datetime.now(timezone.utc).isoformat(),
               "result_digest": stable_hash(deterministic)}
    _atomic_json(args.output_root / "candidate-comparison-verification.json", payload)
    status = "PASS" if payload["passed"] else "FAIL"
    (args.output_root / "candidate-comparison-verification.html").write_text(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>Candidate comparison verification</title></head><body>"
        f"<h1>{status}</h1><p>Independent source, representation, physical generation, "
        f"case, scan, recall, aggregate and seal reconstruction.</p><pre>{escape(json.dumps(payload, indent=2, sort_keys=True))}"
        "</pre></body></html>"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
