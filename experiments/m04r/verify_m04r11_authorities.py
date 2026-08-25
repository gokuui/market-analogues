"""Independent verifier for the sealed M04R-11 authority matrix."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from typing import Any

import pandas as pd

from market_analogues.adapters import file_fingerprint, source_from_spec
from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.certified_packed_search import certified_packed_search_contract
from market_analogues.config import load_config
from market_analogues.distance import representation_distance
from market_analogues.episodes import build_episode
from market_analogues.m04r_certified_search_verification import _certificate_digest
from market_analogues.m04r_full_pack_verification import EVIDENCE_OMITTED as BUILD_OMITTED
from market_analogues.m04r_validation_registry import validate_m04r_validation_registry
from market_analogues.packed_bound_search import (
    PackedBoundQuery,
    bound_proposal_candidate_digest,
    scan_packed_bound_proposals,
    scan_packed_bound_proposals_many,
)
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.representation import represent
from market_analogues.types import EpisodeKey, InstrumentKey, stable_hash


SCHEMA_VERSION = "m04r11-certified-authority-verification-v2"
CASE_OMITTED = {
    "created_at", "proposal_seconds", "amortized_proposal_seconds",
    "exact_seconds", "final_exact_seconds", "frontier_attempt_measurements",
    "peak_rss_mb",
    "result_digest", "checkpoint_integrity_digest",
}
MATRIX_OMITTED = {
    "created_at", "elapsed_seconds", "p95_exact_seconds",
    "maximum_exact_seconds", "total_exact_seconds", "maximum_worker_rss_mb",
    "p95_search_seconds", "maximum_search_seconds", "total_search_seconds",
    "measurements", "performance_gates", "performance_gate_passed",
    "measurement_integrity_digest", "result_digest",
}


def _case_result_digest(payload: dict[str, Any]) -> str:
    return stable_hash({key: value for key, value in payload.items() if key not in CASE_OMITTED})


def _checkpoint_digest(payload: dict[str, Any]) -> str:
    return stable_hash({
        key: value for key, value in payload.items()
        if key not in {"created_at", "checkpoint_integrity_digest"}
    })


def _implementation_manifest() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    runner = Path(__file__).with_name("m04r11_build_authorities.py").resolve()
    paths = [runner, *sorted((repository / "src").rglob("*.py"))]
    files = {
        str(path.relative_to(repository)): file_fingerprint(path)
        for path in paths
    }
    return {"files": files, "digest": stable_hash(files)}


def _frontier_execution_failure(
    payload: dict[str, Any], contract: dict[str, Any],
) -> str | None:
    certificate = payload.get("certificate") or {}
    rounds = certificate.get("rounds") or []
    frontiers = [int(row.get("frontier_rows", -1)) for row in rounds]
    attempts = payload.get("frontier_attempts") or []
    if not frontiers or not attempts or min(frontiers) < 1:
        return "authority frontier evidence is missing"
    policy = contract.get("frontier_overflow_policy") or {}
    primary = int(contract["controls"]["maximum_frontier_rows"])
    eligible = int(certificate.get("eligible_candidates", -1))
    expected = [primary]
    while expected[-1] < eligible:
        expected.append(min(eligible, expected[-1] * int(policy.get("growth_factor", -1))))
    limits = [int(row.get("maximum_frontier_rows", -1)) for row in attempts]
    statuses = [row.get("status") for row in attempts]
    proposal_digests = [row.get("proposal_result_digest") for row in attempts]
    attempt_details_valid = True
    for row, limit in zip(attempts[:-1], limits[:-1], strict=True):
        try:
            attempt_details_valid = attempt_details_valid and all((
                int(row.get("exact_evaluated", -1)) == limit,
                float.fromhex(str(row.get("next_lower_bound_hex")))
                <= float.fromhex(str(row.get("stop_threshold_hex"))),
            ))
        except ValueError:
            attempt_details_valid = False
    final = attempts[-1]
    certificate_next = certificate.get("next_lower_bound")
    attempt_details_valid = attempt_details_valid and all((
        int(final.get("exact_evaluated", -1))
        == int(certificate.get("exact_evaluated", -2)),
        final.get("stop_threshold_hex")
        == float(certificate.get("stop_threshold")).hex(),
        final.get("next_lower_bound_hex")
        == (float(certificate_next).hex() if certificate_next is not None else None),
    ))
    if not all((
        policy.get("schema_version") == "m04r11-frontier-overflow-recovery-v1",
        policy.get("trigger")
        == "certified frontier exceeds configured maximum; no result emitted",
        policy.get("terminal_frontier") == "eligible-universe exhaustion",
        limits == expected[:len(limits)],
        len(limits) <= len(expected),
        statuses[-1] == "certified",
        all(value == "overflow" for value in statuses[:-1]),
        payload.get("frontier_overflow_recovery") is (len(attempts) > 1),
        payload.get("frontier_limit_rows") == limits[-1],
        max(frontiers) <= limits[-1],
        all(isinstance(value, str) and value for value in proposal_digests),
        attempt_details_valid,
    )):
        return "authority frontier execution policy differs"
    return None


def _packed_query(source: Any, case: dict[str, Any]) -> PackedBoundQuery:
    episode = build_episode(
        source, InstrumentKey("nasdaq", str(case["symbol"])), str(case["cutoff"]),
        int(case["lookback"]), str(case["representation_version"]),
    )
    return PackedBoundQuery(
        episode.key.id, episode.key.instrument.source_symbol,
        int(episode.bars.timestamp.iloc[0].value),
        int(episode.bars.timestamp.iloc[-61].value),
        represent(episode), ("A", "B"),
    )


def _proposal_evidence_failures(
    source: Any, cases: list[dict[str, Any]], payloads: list[dict[str, Any]],
    store_root: Path, generation_id: str, primary_limit: int,
) -> tuple[list[str], int]:
    """Independently rescan every primary prefix and every recovery tier."""
    failures: list[str] = []
    by_id = {str(row["query_episode_id"]): row for row in payloads}
    queries = [(case, _packed_query(source, case)) for case in cases]
    primary_reports: dict[str, Any] = {}
    for first in range(0, len(queries), 8):
        chunk = queries[first:first + 8]
        batch = scan_packed_bound_proposals_many(
            store_root, generation_id, [query for _, query in chunk],
            route_quotas={"composite": primary_limit + 1}, block_rows=8_192,
            block_order="reverse", verify_content=False,
        )
        primary_reports.update({
            report.query_episode_id: report for report in batch.reports
        })
    scans = len(range(0, len(queries), 8))
    for case, query in queries:
        payload = by_id.get(str(case["episode_id"]))
        if payload is None:
            failures.append(f"proposal evidence case missing:{case['case_id']}")
            continue
        attempts = payload.get("frontier_attempts") or []
        reports = [primary_reports[query.episode_id]]
        for attempt in attempts[1:]:
            limit = int(attempt["maximum_frontier_rows"])
            reports.append(scan_packed_bound_proposals(
                store_root, generation_id, query,
                route_quotas={"composite": limit + 1}, block_rows=8_192,
                block_order="reverse", verify_content=False,
            ))
            scans += 1
        if len(reports) != len(attempts):
            failures.append(f"proposal attempt count differs:{case['case_id']}")
            continue
        for attempt, report in zip(attempts, reports, strict=True):
            if attempt.get("proposal_result_digest") != report.result_digest:
                failures.append(f"proposal result digest differs:{case['case_id']}")
        final = reports[-1]
        certificate = payload.get("certificate") or {}
        if int(final.eligible_rows) != int(certificate.get("eligible_candidates", -1)):
            failures.append(f"proposal eligible count differs:{case['case_id']}")
        for round_row in certificate.get("rounds") or []:
            required = min(
                int(round_row["frontier_rows"]) + 1, int(final.eligible_rows),
            )
            digest = bound_proposal_candidate_digest(final.candidates[:required])
            if digest != round_row.get("proposal_digest"):
                failures.append(f"proposal round digest differs:{case['case_id']}")
                break
    return failures, scans


def _independent_case_failures(
    payload: dict[str, Any], case: dict[str, Any], contract: dict[str, Any],
    source: Any, packed_provenance_digest: str,
) -> list[str]:
    failures: list[str] = []
    query_id = str(case["episode_id"])
    if not all((
        payload.get("schema_version") == "m04r11-certified-authority-case-v2",
        payload.get("status") == "completed",
        payload.get("contract_digest") == contract.get("contract_digest"),
        payload.get("registry_digest") == contract.get("registry_digest"),
        payload.get("generation_id") == contract.get("generation_id"),
        payload.get("registry_case_id") == case.get("case_id"),
        payload.get("query_symbol") == case.get("symbol"),
        pd.Timestamp(payload.get("query_cutoff")) == pd.Timestamp(case.get("cutoff")),
    )):
        failures.append("authority case identity/provenance wrapper differs")
    instrument = InstrumentKey("nasdaq", str(case["symbol"]))
    bars = source.load(instrument)
    cutoff = pd.Timestamp(case["cutoff"])
    query_bars = bars[pd.to_datetime(bars.timestamp) <= cutoff].tail(int(case["lookback"]))
    query_start = pd.Timestamp(query_bars.timestamp.iloc[0])
    latest_eligible = pd.Timestamp(query_bars.timestamp.iloc[-61])
    if not all((
        pd.Timestamp(payload.get("query_start")) == query_start,
        pd.Timestamp(payload.get("latest_eligible_cutoff")) == latest_eligible,
    )):
        failures.append("authority query temporal wrapper differs")
    key = EpisodeKey(instrument, cutoff, int(case["lookback"]), str(case["representation_version"]))
    if key.id != query_id or payload.get("query_episode_id") != query_id:
        failures.append("query episode identity differs")
    query_episode = build_episode(
        source, instrument, cutoff, int(case["lookback"]),
        str(case["representation_version"]),
    )
    query_representation = represent(query_episode)
    stock_prefix = asdict(causal_prefix_digest(bars, cutoff))
    benchmark = source.load_benchmark()
    benchmark_prefix = asdict(causal_prefix_digest(benchmark, cutoff)) if benchmark is not None else None
    if stock_prefix != case["stock_prefix"] or payload.get("query_stock_prefix") != stock_prefix:
        failures.append("query stock prefix differs")
    if benchmark_prefix != case["benchmark_prefix"] or payload.get("query_benchmark_prefix") != benchmark_prefix:
        failures.append("query benchmark prefix differs")
    request = contract["search_contract"]["request"]
    input_digest = stable_hash({
        "query_stock_prefix": stock_prefix,
        "query_benchmark_prefix": benchmark_prefix,
        "request": {
            "search_datasets": ["nasdaq"],
            "quality_tiers": request["quality_tiers"],
            "top_k": request["top_k"],
            "cross_dataset": request["cross_dataset"],
            "deduplicate_overlaps": request["deduplicate_overlaps"],
            "max_per_instrument": request["max_per_instrument"],
            "minimum_history_gap_bars": request["minimum_history_gap_bars"],
        },
        "packed_provenance_digest": packed_provenance_digest,
    })
    certificate = payload.get("certificate") or {}
    matches = payload.get("matches") or []
    if certificate.get("input_digest") != input_digest:
        failures.append("certificate input digest differs")
    if certificate.get("schema_version") != "m04r-certified-packed-search-v6":
        failures.append("certificate execution schema differs")
    if certificate.get("generation_id") != contract["generation_id"]:
        failures.append("certificate generation differs")
    if certificate.get("contract_digest") != contract[
        "certified_execution_contract"
    ]["digest"]:
        failures.append("certificate search contract differs")
    if certificate.get("result_digest") != _certificate_digest({
        "certificate": certificate, "matches": matches,
    }):
        failures.append("certificate digest differs")
    if payload.get("certificate_digest") != certificate.get("result_digest"):
        failures.append("authority certificate wrapper digest differs")
    if len(matches) != 20 or len({row.get("episode_id") for row in matches}) != 20:
        failures.append("authority does not contain 20 unique matches")
    ordered = [(float(row["total_distance"]), str(row["episode_id"])) for row in matches]
    if ordered != sorted(ordered):
        failures.append("authority order is not distance/episode-ID stable")
    counts = pd.Series([str(row.get("symbol")) for row in matches]).value_counts()
    if len(counts) and int(counts.max()) > int(request["max_per_instrument"]):
        failures.append("authority violates per-instrument diversity cap")
    for match in matches:
        match_cutoff = pd.Timestamp(match["cutoff"])
        if match_cutoff > latest_eligible:
            failures.append("authority admits a temporally ineligible match")
            break
        if str(match["symbol"]) == str(case["symbol"]) and match_cutoff >= query_start:
            failures.append("authority admits an overlapping same-symbol match")
            break
        if match.get("quality_tier") not in request["quality_tiers"]:
            failures.append("authority admits a disallowed quality tier")
            break
        candidate = build_episode(
            source, InstrumentKey("nasdaq", str(match["symbol"])), match_cutoff,
            int(case["lookback"]), str(case["representation_version"]),
            str(match["quality_tier"]),
        )
        total, components, alignment = representation_distance(
            query_representation, represent(candidate),
        )
        component_delta = max(
            abs(float(components[name]) - float(match["component_distances"][name]))
            for name in components
        )
        expected_alignment = [[int(left), int(right)] for left, right in alignment]
        if (
            set(components) != set(match["component_distances"])
            or
            abs(float(total) - float(match["total_distance"])) > 1e-12
            or component_delta > 1e-12
            or expected_alignment != match.get("alignment")
        ):
            failures.append("authority exact match reconstruction differs")
            break
    exact = int(certificate.get("exact_evaluated", -1))
    pruned = int(certificate.get("safely_pruned", -1))
    eligible = int(certificate.get("eligible_candidates", -2))
    if exact + pruned != eligible:
        failures.append("certificate candidate accounting differs")
    next_bound = certificate.get("next_lower_bound")
    stopped = bool(certificate.get("stopped_early"))
    if not (
        (stopped and next_bound is not None
         and float(next_bound) > float(certificate.get("stop_threshold")))
        or (not stopped and pruned == 0)
    ):
        failures.append("certificate has no strict stop or literal exhaustion")
    if float(certificate.get("maximum_quantized_bound_excess", float("inf"))) > 1e-12:
        failures.append("certificate quantized bound exceeds tolerance")
    if payload.get("result_digest") != _case_result_digest(payload):
        failures.append("authority case result digest differs")
    if payload.get("checkpoint_integrity_digest") != _checkpoint_digest(payload):
        failures.append("authority checkpoint integrity digest differs")
    if payload.get("real_forward_outcomes_accessed") is not False:
        failures.append("authority outcome-access marker differs")
    gates = payload.get("gates") or {}
    expected_gate_names = {
        "twenty_matches", "unique_episode_ids", "stable_distance_id_order",
        "quality_tiers_allowed", "per_instrument_cap",
        "same_symbol_overlap_excluded", "candidate_cutoffs_temporally_eligible",
        "certificate_query_equal", "candidate_accounting",
        "strict_stop_or_exhaustion", "quantized_bound_safe",
        "certificate_digest_reconstructed", "frontier_execution_policy",
    }
    if not all((
        set(gates) == expected_gate_names,
        all(value is True for value in gates.values()),
        payload.get("gate_passed") is True,
    )):
        failures.append("authority declared case gates differ")
    frontier_failure = _frontier_execution_failure(payload, contract)
    if frontier_failure:
        failures.append(frontier_failure)
    attempt_measurements = payload.get("frontier_attempt_measurements") or []
    attempts = payload.get("frontier_attempts") or []
    measurement_shape_valid = (
        len(attempt_measurements) == len(attempts)
        and all(
            measured.get("maximum_frontier_rows")
            == semantic.get("maximum_frontier_rows")
            and measured.get("status") == semantic.get("status")
            and float(measured.get("elapsed_seconds", -1)) >= 0
            for measured, semantic in zip(
                attempt_measurements, attempts, strict=True,
            )
        )
    )
    final_attempt_elapsed = (
        float(attempt_measurements[-1].get("elapsed_seconds", -1))
        if attempt_measurements else -1.0
    )
    if not all((
        measurement_shape_valid,
        float(payload.get("exact_seconds", -1)) == sum(
            float(row.get("elapsed_seconds", 0)) for row in attempt_measurements
        ),
        0 <= float(payload.get("final_exact_seconds", -1))
        <= final_attempt_elapsed,
        float(payload.get("proposal_seconds", -1))
        >= float(payload.get("amortized_proposal_seconds", -1)) >= 0,
        float(payload.get("peak_rss_mb", -1)) > 0,
    )):
        failures.append("authority case measurements differ")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    source = source_from_spec(config.datasets["nasdaq"])
    failures = list(validate_m04r_validation_registry(source, args.registry.parent))
    registry = json.loads(args.registry.read_text())
    contract_path = args.authority_root / "authority-contract.json"
    matrix_path = args.authority_root / "authority-matrix.json"
    seal_path = args.authority_root / "SEALED.json"
    if not all(path.exists() for path in (contract_path, matrix_path, seal_path)):
        raise SystemExit("authority contract, matrix, or seal is missing")
    contract = json.loads(contract_path.read_text())
    matrix = json.loads(matrix_path.read_text())
    seal = json.loads(seal_path.read_text())
    contract_without_digest = {key: value for key, value in contract.items() if key != "contract_digest"}
    if stable_hash(contract_without_digest) != contract.get("contract_digest"):
        failures.append("authority contract digest differs")
    if contract.get("schema_version") != "m04r11-authority-build-contract-v2":
        failures.append("authority contract schema differs")
    runner_path = Path(__file__).with_name("m04r11_build_authorities.py")
    if file_fingerprint(runner_path) != contract.get("runner_sha256"):
        failures.append("authority runner source hash differs")
    if _implementation_manifest() != contract.get("implementation_manifest"):
        failures.append("authority implementation source manifest differs")
    expected_execution_contract = certified_packed_search_contract(
        requested_positions=True, vector_lower_bounds=True,
        deferred_alignments=True, compact_scored=True,
    )
    if contract.get("certified_execution_contract") != expected_execution_contract:
        failures.append("authority certified execution contract differs")
    controls = dict(registry["search_contract"]["controls"])
    expected_overflow_policy = {
        "schema_version": "m04r11-frontier-overflow-recovery-v1",
        "trigger": "certified frontier exceeds configured maximum; no result emitted",
        "first_recovery_frontier_rows": int(controls["maximum_frontier_rows"]) * 2,
        "growth_factor": 2,
        "terminal_frontier": "eligible-universe exhaustion",
        "scope": "authority construction only",
        "semantics": (
            "retry the unchanged exact certified search over monotonically larger "
            "bound-sorted frontiers; never emit an approximate result"
        ),
    }
    build = json.loads((args.full_root / "packed-bound-full.json").read_text())
    deterministic_build = {
        key: value for key, value in build.items() if key not in BUILD_OMITTED
    }
    if not all((
        contract.get("registry_digest") == registry.get("registry_digest"),
        contract.get("search_contract") == registry.get("search_contract"),
        contract.get("controls") == controls,
        contract.get("execution_processes") == int(controls["processes"]) == 8,
        contract.get("frontier_overflow_policy") == expected_overflow_policy,
        contract.get("generation_id")
        == registry["search_contract"]["packed_generation_id"],
        contract.get("generation_id") == build.get("generation_id"),
        contract.get("full_build_evidence_digest") == build.get("result_digest"),
        build.get("result_digest") == stable_hash(deterministic_build),
        build.get("gate_passed") is True,
        build.get("shadow_generation") is True,
        build.get("real_forward_outcomes_accessed") is False,
        not (args.full_root / "store" / "active.json").exists(),
        contract.get("authority_root_policy")
        == "write-isolated truth; no candidate result input",
        contract.get("real_forward_outcomes_accessed") is False,
    )):
        failures.append("authority contract input/provenance bindings differ")
    loaded = load_packed_generation(
        args.full_root / "store", str(contract["generation_id"]),
        verify_content=True, validate_records=False,
    )
    cases = list(registry["cases_data"])
    if contract.get("expected_query_episode_ids") != [case["episode_id"] for case in cases]:
        failures.append("authority contract membership/order differs")
    if contract.get("selection_order") != [case["case_id"] for case in cases]:
        failures.append("authority contract selection order differs")
    verified = []
    per_case_failures: dict[str, list[str]] = {}
    for case in cases:
        path = args.authority_root / "cases" / f"{case['episode_id']}.json"
        if not path.exists():
            per_case_failures[case["case_id"]] = ["authority checkpoint is missing"]
            continue
        payload = json.loads(path.read_text())
        observed = _independent_case_failures(
            payload, case, contract, source, str(loaded.manifest["provenance_digest"]),
        )
        if observed:
            per_case_failures[case["case_id"]] = observed
        else:
            verified.append(payload)
    if per_case_failures:
        failures.append("one or more authority cases failed independent reconstruction")
    proposal_failures: list[str] = []
    proposal_scans = 0
    if len(verified) == 60:
        proposal_failures, proposal_scans = _proposal_evidence_failures(
            source, cases, verified, args.full_root / "store",
            str(contract["generation_id"]),
            int(contract["controls"]["maximum_frontier_rows"]),
        )
        failures.extend(proposal_failures)
    matrix_without = {key: value for key, value in matrix.items() if key not in MATRIX_OMITTED}
    if stable_hash(matrix_without) != matrix.get("result_digest"):
        failures.append("authority matrix digest differs")
    expected_rows = [{
        "registry_case_id": case["registry_case_id"],
        "query_episode_id": case["query_episode_id"],
        "authority_digest": case["result_digest"],
        "certificate_digest": case["certificate_digest"],
        "eligible_candidates": case["certificate"]["eligible_candidates"],
        "exact_evaluated": case["certificate"]["exact_evaluated"],
    } for case in verified]
    if matrix.get("cases") != expected_rows:
        failures.append("authority matrix case bindings differ")
    expected_measurements = [{
        "registry_case_id": case["registry_case_id"],
        "proposal_seconds": case["proposal_seconds"],
        "exact_seconds": case["exact_seconds"],
        "search_seconds": float(case["proposal_seconds"])
        + float(case["exact_seconds"]),
        "peak_rss_mb": case["peak_rss_mb"],
    } for case in verified]
    exact_seconds = [float(case["exact_seconds"]) for case in verified]
    search_seconds = [
        float(case["proposal_seconds"]) + float(case["exact_seconds"])
        for case in verified
    ]
    expected_measurement_values = {
        "measurements": expected_measurements,
        "p95_exact_seconds": (
            float(pd.Series(exact_seconds).quantile(.95)) if exact_seconds else None
        ),
        "maximum_exact_seconds": max(exact_seconds, default=None),
        "total_exact_seconds": sum(exact_seconds),
        "p95_search_seconds": (
            float(pd.Series(search_seconds).quantile(.95)) if search_seconds else None
        ),
        "maximum_search_seconds": max(search_seconds, default=None),
        "total_search_seconds": sum(search_seconds),
        "maximum_worker_rss_mb": max(
            (float(case["peak_rss_mb"]) for case in verified), default=0.0,
        ),
    }
    if any(
        matrix.get(key) != value for key, value in expected_measurement_values.items()
    ):
        failures.append("authority measurements do not reconstruct from cases")
    expected_performance_gates = {
        "certified_p95_search_at_most_300_seconds": (
            expected_measurement_values["p95_search_seconds"] is not None
            and float(expected_measurement_values["p95_search_seconds"]) <= 300
        ),
        "certified_maximum_search_at_most_600_seconds": (
            expected_measurement_values["maximum_search_seconds"] is not None
            and float(expected_measurement_values["maximum_search_seconds"]) <= 600
        ),
        "certified_peak_rss_at_most_1536_mb": (
            float(expected_measurement_values["maximum_worker_rss_mb"]) <= 1_536
        ),
    }
    if not all((
        matrix.get("performance_gates") == expected_performance_gates,
        matrix.get("performance_gate_passed")
        is all(expected_performance_gates.values()),
    )):
        failures.append("authority performance gates differ")
    expected_measurement_values["performance_gates"] = expected_performance_gates
    expected_measurement_values["performance_gate_passed"] = all(
        expected_performance_gates.values()
    )
    expected_measurement_digest = stable_hash(expected_measurement_values)
    if matrix.get("measurement_integrity_digest") != expected_measurement_digest:
        failures.append("authority measurement integrity digest differs")
    if not all((
        matrix.get("schema_version") == "m04r11-certified-authority-matrix-v2",
        matrix.get("gate_passed") is True,
        matrix.get("completed_cases") == 60,
        not matrix.get("invalid_cases"),
        not matrix.get("worker_failures"),
        all((matrix.get("gates") or {}).values()),
    )):
        failures.append("authority matrix completion gates differ")
    matrix_ids = [row.get("query_episode_id") for row in matrix.get("cases", [])]
    if matrix_ids != [case["episode_id"] for case in cases]:
        failures.append("authority matrix case order differs")
    seal_without_digest = {key: value for key, value in seal.items() if key != "seal_digest"}
    if stable_hash(seal_without_digest) != seal.get("seal_digest"):
        failures.append("authority seal digest differs")
    if seal.get("authority_matrix_digest") != matrix.get("result_digest"):
        failures.append("authority seal does not bind matrix")
    if seal.get("measurement_integrity_digest") != matrix.get(
        "measurement_integrity_digest"
    ):
        failures.append("authority seal does not bind measurements")
    if not all((
        seal.get("schema_version") == "m04r11-authority-seal-v2",
        seal.get("authority_cases") == 60,
        seal.get("contract_digest") == contract.get("contract_digest"),
        seal.get("registry_digest") == registry.get("registry_digest"),
        seal.get("seal_scope") == "exact authority correctness only",
        seal.get("authority_correctness_sealed") is True,
        seal.get("performance_gate_passed")
        is matrix.get("performance_gate_passed"),
        seal.get("production_promotion_authorized") is False,
        seal.get("candidate_results_opened") is False,
        seal.get("real_forward_outcomes_accessed") is False,
    )):
        failures.append("authority seal contract differs")
    gates = {
        "registry_runtime_validation_passed": not validate_m04r_validation_registry(
            source, args.registry.parent,
        ),
        "physical_generation_rehashed": True,
        "all_60_cases_independently_reconstructed": len(verified) == 60,
        "all_proposal_prefixes_independently_rescanned": (
            len(verified) == 60 and not proposal_failures and proposal_scans >= 8
        ),
        "matrix_and_seal_reconstructed": not any(
            "matrix" in value or "seal" in value for value in failures
        ),
        "candidate_results_remained_unopened": (
            matrix.get("candidate_results_opened") is False
            and seal.get("candidate_results_opened") is False
        ),
        "real_forward_outcomes_excluded": (
            matrix.get("real_forward_outcomes_accessed") is False
            and seal.get("real_forward_outcomes_accessed") is False
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "registry_digest": registry["registry_digest"],
        "contract_digest": contract["contract_digest"],
        "generation_id": contract["generation_id"],
        "authority_matrix_digest": matrix.get("result_digest"),
        "authority_seal_digest": seal.get("seal_digest"),
        "verified_cases": len(verified),
        "authority_correctness_passed": matrix.get("gate_passed") is True,
        "performance_gate_passed": matrix.get("performance_gate_passed") is True,
        "production_promotion_authorized": False,
        "proposal_rescans": proposal_scans,
        "proposal_failures": proposal_failures,
        "per_case_failures": per_case_failures,
        "gates": gates, "failures": list(dict.fromkeys(failures)),
        "real_forward_outcomes_accessed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    payload["passed"] = all(gates.values()) and not payload["failures"]
    payload["result_digest"] = stable_hash({
        key: value for key, value in payload.items()
        if key not in {"created_at", "result_digest"}
    })
    args.output_root.mkdir(parents=True, exist_ok=True)
    json_path = args.output_root / "m04r11-authority-verification.json"
    html_path = args.output_root / "m04r11-authority-verification.html"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    status = "PASS" if payload["passed"] else "FAIL"
    html_path.write_text(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>M04R-11 authority verification</title><style>"
        "body{font-family:system-ui;max-width:1200px;margin:2rem auto}"
        "pre{white-space:pre-wrap}</style></head><body><h1>M04R-11 authority "
        f"verification: {status}</h1><pre>{escape(json.dumps(payload, indent=2, sort_keys=True))}"
        "</pre></body></html>"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(html_path)
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
