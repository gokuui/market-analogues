"""Truth-blind validation helpers for sealed M04R candidate-pool evidence."""

from __future__ import annotations

from math import isfinite
from typing import Any, Mapping

from .packed_bound_search import BoundProposal, bound_proposal_candidate_digest
from .types import stable_hash


CANDIDATE_CASE_SCHEMA = "candidate-recall-case-v2-producer"
LEGACY_SCAN_SCHEMA = "m04r-global-bound-proposal-v1"
CASE_SEMANTIC_OMITTED = {
    "created_at", "cold_seconds", "warm_first_seconds", "warm_second_seconds",
    "cold_task_seconds", "peak_rss_mb", "result_digest",
    "checkpoint_integrity_digest",
}


def candidate_semantic_digest(payload: Mapping[str, Any]) -> str:
    return stable_hash({
        key: value for key, value in payload.items()
        if key not in CASE_SEMANTIC_OMITTED
    })


def candidate_checkpoint_integrity_digest(payload: Mapping[str, Any]) -> str:
    return stable_hash({
        key: value for key, value in payload.items()
        if key not in {"created_at", "checkpoint_integrity_digest"}
    })


def reconstructed_candidate_digest(rows: list[dict[str, Any]]) -> str:
    proposals = [BoundProposal(
        str(row["episode_id"]), str(row["symbol"]), int(row["cutoff_ns"]),
        str(row["quality_tier"]), float.fromhex(str(row["lower_bound_hex"])),
        tuple(str(value) for value in row["routes"]), bool(row["overflow_fallback"]),
    ) for row in rows]
    return bound_proposal_candidate_digest(proposals)


def scan_result_digest(invariant: Mapping[str, Any], proposal_contract_digest: str) -> str:
    return stable_hash({
        "schema_version": invariant["schema_version"],
        "contract_digest": proposal_contract_digest,
        "generation_id": invariant["generation_id"],
        "query_episode_id": invariant["query_episode_id"],
        "rows_scanned": invariant["rows_scanned"],
        "eligible_rows": invariant["eligible_rows"],
        "eligible_main_rows": invariant["eligible_main_rows"],
        "eligible_overflow_rows": invariant["eligible_overflow_rows"],
        "route_counts": invariant["route_counts"],
        "route_quotas": invariant["route_quotas"],
        "candidate_digest": invariant["candidate_digest"],
        "real_forward_outcomes_accessed": False,
    })


def validate_truth_blind_candidate_case(
    payload: Mapping[str, Any], *, expected_case: Mapping[str, Any],
    registry_digest: str, generation_id: str, proposal_contract_digest: str,
    producer_contract_digest: str, route_quotas: Mapping[str, int],
    physical_rows: int, expected_query_start_ns: int | None = None,
    expected_latest_eligible_ns: int | None = None,
    expected_representation_digest: str | None = None,
) -> tuple[str, ...]:
    """Fail closed on any semantic, provenance, measurement or digest drift."""
    failures: list[str] = []
    try:
        rows = list(payload["candidates"])
        invariants = list(payload["scan_invariants"])
        if len(invariants) != 3 or not (invariants[0] == invariants[1] == invariants[2]):
            failures.append("three scan invariants differ")
            return tuple(failures)
        first = invariants[0]
        for invariant in invariants:
            if not all((
                invariant.get("schema_version") == LEGACY_SCAN_SCHEMA,
                invariant.get("generation_id") == generation_id,
                invariant.get("query_episode_id") == expected_case["episode_id"],
                int(invariant.get("rows_scanned", -1)) == physical_rows,
                invariant.get("route_quotas") == dict(route_quotas),
                int(invariant.get("eligible_rows", -1))
                == int(invariant.get("eligible_main_rows", -2))
                + int(invariant.get("eligible_overflow_rows", -3)),
                invariant.get("result_digest")
                == scan_result_digest(invariant, proposal_contract_digest),
            )):
                failures.append("scan identity, accounting or digest differs")
                break
        candidate_ids = [str(row["episode_id"]) for row in rows]
        bounds = [float.fromhex(str(row["lower_bound_hex"])) for row in rows]
        identities_valid = all(
            len(value) == 24 and value == value.lower()
            and len(bytes.fromhex(value)) == 12 for value in candidate_ids
        )
        order_valid = list(zip(bounds, candidate_ids)) == sorted(zip(bounds, candidate_ids))
        routes_valid = all(
            row["routes"] == sorted(set(row["routes"]))
            and bool(row["routes"])
            and set(row["routes"]).issubset(route_quotas)
            for row in rows
        )
        metadata_valid = all(
            isfinite(bound) and bound >= 0
            and row["quality_tier"] in ("A", "B")
            and isinstance(row["cutoff_ns"], int)
            and isinstance(row["overflow_fallback"], bool)
            for row, bound in zip(rows, bounds, strict=True)
        )
        rebuilt_digest = reconstructed_candidate_digest(rows)
        route_counts = {
            route: sum(route in row["routes"] for row in rows)
            for route in route_quotas
        }
        query_start_ns = int(payload["query_start_ns"])
        latest_eligible_ns = int(payload["latest_eligible_ns"])
        violations = {
            "duplicates": len(candidate_ids) - len(set(candidate_ids)),
            "future": sum(int(row["cutoff_ns"]) > latest_eligible_ns for row in rows),
            "same_symbol_overlap": sum(
                row["symbol"] == payload["query_symbol"]
                and int(row["cutoff_ns"]) >= query_start_ns for row in rows
            ),
            "tier": sum(row["quality_tier"] not in ("A", "B") for row in rows),
        }
        cold = float(payload["cold_seconds"])
        warm_first = float(payload["warm_first_seconds"])
        warm_second = float(payload["warm_second_seconds"])
        cold_task = float(payload["cold_task_seconds"])
        peak_rss = float(payload["peak_rss_mb"])
        measurements_valid = all(
            isfinite(value) and value >= 0
            for value in (cold, warm_first, warm_second, cold_task, peak_rss)
        ) and cold_task >= cold
        expected_gates = {
            "cold_advice_supported_and_applied": payload.get("cold_advice_applied") is True,
            "three_scan_digest_and_block_order_invariance": True,
            "internal_eligible_row_accounting": int(first["eligible_rows"]) == int(first["eligible_main_rows"]) + int(first["eligible_overflow_rows"]),
            "physical_row_accounting": int(first["rows_scanned"]) == physical_rows,
            "frozen_route_quotas": payload.get("route_quotas") == dict(route_quotas),
            "zero_temporal_overlap_tier_duplicate_errors": not any(violations.values()),
            "measurements_finite_nonnegative_and_task_contains_cold": measurements_valid,
            "cold_scan_at_most_120_seconds": measurements_valid and cold <= 120.0,
            "second_warm_scan_at_most_60_seconds": measurements_valid and warm_second <= 60.0,
            "rss_at_most_1024_mib": measurements_valid and peak_rss <= 1_024.0,
        }
        expected_identity = all((
            payload.get("schema_version") == CANDIDATE_CASE_SCHEMA,
            payload.get("registry_case_id") == expected_case["case_id"],
            payload.get("query_episode_id") == expected_case["episode_id"],
            payload.get("query_symbol") == expected_case["symbol"],
            payload.get("registry_digest") == registry_digest,
            payload.get("generation_id") == generation_id,
            payload.get("proposal_contract_digest") == proposal_contract_digest,
            payload.get("producer_contract_digest") == producer_contract_digest,
            payload.get("query_stock_prefix") == expected_case["stock_prefix"],
            payload.get("query_benchmark_prefix") == expected_case["benchmark_prefix"],
            expected_query_start_ns is None or query_start_ns == expected_query_start_ns,
            expected_latest_eligible_ns is None or latest_eligible_ns == expected_latest_eligible_ns,
            expected_representation_digest is None
            or payload.get("query_representation_digest") == expected_representation_digest,
            payload.get("real_forward_outcomes_accessed") is False,
        ))
        if not expected_identity:
            failures.append("candidate outer identity or query provenance differs")
        if not all((
            identities_valid, order_valid, routes_valid, metadata_valid,
            len(rows) == int(first["candidate_count"]),
            route_counts == first["route_counts"],
            rebuilt_digest == payload.get("candidate_digest_reconstructed"),
            rebuilt_digest == first["candidate_digest"],
            payload.get("violations") == violations,
        )):
            failures.append("candidate rows, routes, ordering or digest differ")
        if payload.get("gates") != expected_gates or payload.get("passed") is not all(expected_gates.values()):
            failures.append("candidate gates differ")
        if payload.get("result_digest") != candidate_semantic_digest(payload):
            failures.append("candidate semantic digest differs")
        if payload.get("checkpoint_integrity_digest") != candidate_checkpoint_integrity_digest(payload):
            failures.append("candidate checkpoint integrity digest differs")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        failures.append(f"malformed candidate evidence:{type(exc).__name__}:{exc}")
    return tuple(sorted(set(failures)))
