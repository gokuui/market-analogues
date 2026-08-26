"""Frozen contracts for the M04R-11 resident candidate v2 experiment.

This module is deliberately isolated from the terminal v1 producer.  It
contains no runner and opens neither candidate checkpoints nor authority
artifacts.  Producers and the pre-open comparator may share these constants
and pure validators; the final verifier must reconstruct them independently.
"""

from __future__ import annotations

from copy import deepcopy
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

from market_analogues.types import stable_hash


PRODUCER_CONTRACT_SCHEMA = "candidate-recall-resident-producer-contract-v1"
RESIDENT_READY_SCHEMA = "m04r-resident-packed-store-ready-v2"
RESIDENT_CONTENT_SCHEMA = "m04r-resident-packed-store-content-v1"
RESIDENT_BINDING_SCHEMA = "candidate-resident-ready-binding-v2"
SEMANTIC_CASE_SCHEMA = "candidate-recall-semantic-case-v3"
PERFORMANCE_ATTEMPT_SCHEMA = "candidate-resident-performance-attempt-v1"
SEMANTIC_MATRIX_SCHEMA = "candidate-recall-semantic-matrix-v1"
SEMANTIC_SEAL_SCHEMA = "candidate-recall-semantic-seal-v1"
PERFORMANCE_MATRIX_SCHEMA = "candidate-resident-performance-matrix-v1"
PERFORMANCE_FINAL_SCHEMA = "candidate-resident-performance-final-v1"
RUN_LEDGER_EVENT_SCHEMA = "candidate-resident-run-ledger-event-v1"
RUN_LEDGER_HEAD_SCHEMA = "candidate-resident-run-ledger-head-v1"
INCOMPLETE_SCHEMA = "candidate-resident-incomplete-v1"
RUN_COMPLETE_SCHEMA = "candidate-resident-run-complete-v1"
CASE_BUNDLE_SCHEMA = "candidate-resident-case-bundle-v1"
RESULTS_OPENED_SCHEMA = "candidate-recall-results-opened-v2"
COMPARISON_MATRIX_SCHEMA = "candidate-recall-comparison-matrix-v2"
COMPARISON_SEAL_SCHEMA = "candidate-recall-comparison-seal-v2"
COMPARISON_VERIFICATION_SCHEMA = "candidate-recall-comparison-verification-v2"
PREREGISTRATION_SCHEMA = "candidate-recall-resident-preregistration-v1"
PREREGISTRATION_RELATIVE_PATH = (
    "experiments/m04r/m04r11_candidate_v2_preregistered.json"
)

FROZEN_REGISTRY_DIGEST = (
    "0a4da732f91375a091775cb04e6e77c8d136ade47d7f4d16508a2d9a6555361e"
)
FROZEN_GENERATION_ID = (
    "9fc6ae0ec4451133d8006897162f3443803fd30a3c44b78e80a476fbb18bb483"
)
FROZEN_PROPOSAL_CONTRACT_DIGEST = (
    "059db6d78bbe62e7588e1fe2408fe244040b501d190bdf7b211a7ecd7aadb96c"
)
FROZEN_CASE_ORDER_DIGEST = (
    "db1741d50eab19217d6264f051ca3bccf2663c9a54a7e6e846e8689410fe2b47"
)
FROZEN_ROLE_TABLE_DIGEST = (
    "ca97b14d187bfba0332072d1597468cdb163a3c03d57e9d2db16294038437861"
)
FROZEN_EXPOSED_IDS_DIGEST = (
    "777dfbca0624c0aaed998c85ced5da80c1738adf785070837a422fd120585999"
)
FROZEN_CONFIRMATORY_IDS_DIGEST = (
    "266fbd0c6607c126a76cebcb5d1fee6dcad2925d51f48f3b87d6ded28b635564"
)
FROZEN_EXECUTION_ORDER_DIGEST = (
    "2fa117b902416bd222a5d854c95d95a5e9b1bfd47e5452ef3eddd8466e2c7467"
)

FROZEN_ROUTE_QUOTAS = {
    "candle_volatility": 64,
    "coarse": 64,
    "composite": 1_000,
    "market_context": 64,
    "price": 64,
    "stage": 64,
    "structural": 64,
    "volume_shock": 64,
}
FROZEN_REQUEST = {
    "cross_dataset": False,
    "deduplicate_overlaps": True,
    "max_per_instrument": 3,
    "minimum_history_gap_bars": 60,
    "quality_tiers": ["A", "B"],
    "top_k": 20,
}

EXPOSED_PERFORMANCE_CASES = 7
CONFIRMATORY_PERFORMANCE_CASES = 53
TOTAL_RECALL_CASES = 60
EXPOSED_PERFORMANCE_ROLE = "exposed_recovery_regression"
CONFIRMATORY_PERFORMANCE_ROLE = "confirmatory_untouched"
BLIND_RECALL_ROLE = "blind_primary"

FROZEN_EXPOSED_CASES = (
    ("nasdaq-BLDP-current-252", "d107eb49f0ae78d63a90bbce"),
    ("nasdaq-BLDP-historical-252", "f51fc69c1920a76409acc8ff"),
    ("nasdaq-HAS-current-252", "e20fa3265667c4d84c50aa92"),
    ("nasdaq-HAS-historical-252", "2429c38cd8e4f7d8ff1af984"),
    ("nasdaq-FWONA-current-252", "942bba10f93b0a36d4d21766"),
    ("nasdaq-FWONA-historical-252", "224498b96b521ad0e84c703d"),
    ("nasdaq-GABC-current-252", "e3ae797166ab7df0a224edf6"),
)

PREDECESSOR_FAILURE = {
    "schema_version": "candidate-recall-producer-terminal-failure-v1",
    "failure_digest": (
        "64df2ed8f3deff98193552a8b2bb085432c50788e299a8e4e62828a89c8c7843"
    ),
    "producer_contract_digest": (
        "e6d971081f288375df3ad66baad0300eb2d452860b464d6d9c111a6c75d657ad"
    ),
    "status": "terminal_performance_failure_after_interruption",
    "completed_cases": 7,
    "failed_case_id": "nasdaq-GABC-current-252",
    "failed_gates": ["cold_scan_at_most_120_seconds"],
    "candidate_pools_sealed": False,
    "authority_results_opened": False,
    "candidate_authority_comparison_opened": False,
    "resume_authorized": False,
    "claims_policy": {
        "original_combined_gate_passed": False,
        "performance_exposed_cases": 7,
        "recall_holdout_cases_still_blind": 60,
        "remaining_untouched_performance_cases": 53,
    },
}

SCAN_PROTOCOL = {
    "execution": "serial fresh spawned process per query",
    "execution_order": "53 confirmatory-untouched cases then 7 exposed recovery cases",
    "engine": "bounded ordered eight-thread legacy-v1 scan over verified resident mirror",
    "outer_threads": 8,
    "numba_threads_per_scorer": 1,
    "maximum_in_flight_blocks": 8,
    "reduction": "strict requested physical block order into unchanged stable route heaps",
    "cache_advice": "none; resident readiness is explicit and POSIX_FADV_DONTNEED is forbidden",
    "resident_first": {"block_rows": 4_096, "order": "forward"},
    "resident_reverse": {"block_rows": 4_097, "order": "reverse"},
    "resident_repeat": {"block_rows": 4_093, "order": "forward"},
    "primary_performance_attempts_per_case": 1,
    "maximum_semantic_recovery_attempts_per_case": 0,
    "worker_error_policy": "fail fast before the next case; every failure is terminal and no resume is authorized",
}
PERFORMANCE_LIMITS = {
    "resident_first_seconds": 120.0,
    "resident_repeat_seconds": 60.0,
    "worker_rss_mib": 1_536.0,
}
RESIDENT_POLICY = {
    "filesystem": "tmpfs",
    "storage_claim": "tmpfs-backed empirical query latency; RAM residency is not guaranteed when swap is enabled",
    "reserve_bytes": 1_073_741_824,
    "publication_modes": ["copy-and-validate", "validate-existing"],
    "content": "byte-identical generation manifest, rows and overflow sidecar",
    "readiness": "full content-hash verification and pre-touch before any registry query",
    "mutation_detection": "full content-hash verification at readiness and after the final query",
    "attempt_identity": "each attempt holds and verifies an exact regular-file identity lease from start through end",
    "active_pointer_allowed": False,
    "symbolic_links_allowed": False,
    "files_must_be_contained_regular_and_same_tmpfs_st_dev": True,
    "cold_storage_claimed": False,
}

ARTIFACT_SCHEMAS = {
    "producer_contract": PRODUCER_CONTRACT_SCHEMA,
    "resident_ready": RESIDENT_READY_SCHEMA,
    "resident_content": RESIDENT_CONTENT_SCHEMA,
    "resident_binding": RESIDENT_BINDING_SCHEMA,
    "semantic_case": SEMANTIC_CASE_SCHEMA,
    "performance_attempt": PERFORMANCE_ATTEMPT_SCHEMA,
    "run_ledger_event": RUN_LEDGER_EVENT_SCHEMA,
    "run_ledger_head": RUN_LEDGER_HEAD_SCHEMA,
    "incomplete": INCOMPLETE_SCHEMA,
    "semantic_matrix": SEMANTIC_MATRIX_SCHEMA,
    "semantic_seal": SEMANTIC_SEAL_SCHEMA,
    "performance_matrix": PERFORMANCE_MATRIX_SCHEMA,
    "performance_final": PERFORMANCE_FINAL_SCHEMA,
    "run_complete": RUN_COMPLETE_SCHEMA,
    "case_bundle": CASE_BUNDLE_SCHEMA,
    "results_opened": RESULTS_OPENED_SCHEMA,
    "comparison_matrix": COMPARISON_MATRIX_SCHEMA,
    "comparison_seal": COMPARISON_SEAL_SCHEMA,
    "comparison_verification": COMPARISON_VERIFICATION_SCHEMA,
    "preregistration": PREREGISTRATION_SCHEMA,
}

MARKER_AND_RESUME_POLICY = {
    "producer_contract_path": "candidate-contract.json",
    "resident_binding_path": "RESIDENT_READY.json",
    "ledger_events_directory": "ledger/events",
    "ledger_head_path": "ledger/HEAD.json",
    "semantic_cases_directory": "semantic-cases",
    "performance_attempts_directory": "performance-attempts",
    "case_bundles_directory": "case-bundles",
    "case_bundle_filename": "{execution_ordinal:03d}-{query_episode_id}.json",
    "incomplete_marker_path": "INCOMPLETE.json",
    "semantic_matrix_path": "semantic-matrix.json",
    "semantic_seal_path": "SEMANTIC_SEALED.json",
    "performance_matrix_path": "performance-matrix.json",
    "performance_final_path": "PERFORMANCE_FINAL.json",
    "run_complete_path": "RUN_COMPLETE.json",
    "results_opened_path": "RESULTS_OPENED.json",
    "comparison_matrix_path": "candidate-comparison.json",
    "comparison_seal_path": "SEALED.json",
    "comparison_verification_path": "verification.json",
    "resume_authorized": False,
    "partial_ledger_resume_authorized": False,
    "semantic_recovery_authorized": False,
    "incomplete_marker_is_terminal": True,
    "semantic_failure_is_terminal": True,
    "worker_failure_is_terminal": True,
    "existing_case_overwrite_authorized": False,
    "existing_terminal_marker_overwrite_authorized": False,
    "fresh_empty_candidate_root_required": True,
}

PREREGISTRATION_POLICY = {
    "relative_path": PREREGISTRATION_RELATIVE_PATH,
    "must_be_committed_before_launch": True,
    "must_preexist_candidate_output": True,
    "load_from_exact_relative_path_only": True,
    "generate_and_trust_at_launch_forbidden": True,
    "runtime_contract_must_exactly_equal_embedded_contract": True,
    "runtime_source_implementation_environment_must_exactly_match": True,
    "overwrite_at_launch_authorized": False,
    "authority_results_may_be_read_during_generation_or_validation": False,
}

COMPARISON_POLICY = {
    "top_k": 20,
    "minimum_retained_per_case": 19,
    "minimum_retained_aggregate": 1_188,
    "aggregate_denominator": 1_200,
    "twenty_of_twenty_is_descriptive_only": True,
    "results_opened_marker_must_be_written_fsynced_before_authority_read": True,
    "valid_semantic_seal_required": True,
    "terminal_performance_evidence_required": True,
    "performance_pass_required": False,
    "authority_results_opened_before_candidate_semantic_seal": False,
    "comparison_is_one_shot": True,
    "real_forward_outcomes_accessed": False,
}

SEMANTIC_CASE_GATES = (
    "resident_content_matches_contract",
    "query_identity_prefix_and_representation_match",
    "three_scan_digest_and_block_order_invariance",
    "internal_eligible_row_accounting",
    "physical_row_accounting",
    "frozen_route_quotas",
    "candidate_digest_order_and_routes_reconstruct",
    "zero_temporal_overlap_tier_duplicate_errors",
    "real_forward_outcomes_excluded",
)
PERFORMANCE_ATTEMPT_GATES = (
    "same_ready_instance_at_start_and_end",
    "measurements_finite_nonnegative_and_task_contains_scans",
    "resident_first_scan_at_most_120_seconds",
    "resident_repeat_scan_at_most_60_seconds",
    "worker_rss_at_most_1536_mib",
    "primary_attempt_completed",
)
SEMANTIC_MATRIX_GATES = (
    "all_60_semantic_cases_present_in_registry_order",
    "all_semantic_case_gates_passed",
    "mirror_content_unchanged_before_and_after",
    "ledger_semantic_completion_matches_cases",
    "real_forward_outcomes_excluded",
)
PERFORMANCE_MATRIX_GATES = (
    "exact_7_exposed_53_confirmatory_role_partition",
    "all_60_primary_attempts_accounted",
    "all_53_confirmatory_primary_attempts_passed",
    "all_7_exposed_regression_attempts_passed",
    "all_60_operational_limits_passed",
    "single_ready_instance_for_all_primary_attempts",
)
CLAIMS_POLICY = {
    "recall_holdout_cases": 60,
    "recall_claim": "all 60 cases remain blind until the v2 results-opened marker",
    "confirmatory_performance_cases": 53,
    "confirmatory_performance_claim": "only the 53 previously unmeasured cases support the v2 confirmatory performance claim",
    "exposed_performance_cases": 7,
    "exposed_performance_claim": "development/recovery regression only",
    "semantic_seal_independent_of_performance": True,
    "performance_failure_must_not_block_recall_comparison": True,
    "performance_must_be_terminal_before_recall_comparison": True,
    "v1_original_combined_gate_passed": False,
    "production_promotion_authorized": False,
    "real_forward_outcomes_accessed": False,
}

FROZEN_ROW_BYTES = 2_432
FROZEN_OVERFLOW_ROW_BYTES = 32
_HEX = frozenset("0123456789abcdef")
_DIGEST_FIELDS = {
    "manifest_sha256",
    "manifest_digest",
    "provenance_digest",
    "rows_sha256",
    "overflow_sha256",
}
_SOURCE_PACK_FIELDS = {
    "generation_id",
    "source_full_root",
    "manifest_path",
    "manifest_sha256",
    "manifest_digest",
    "provenance_digest",
    "rows_file",
    "rows_path",
    "rows_bytes",
    "rows_sha256",
    "row_count",
    "row_bytes",
    "overflow_file",
    "overflow_path",
    "overflow_bytes",
    "overflow_sha256",
    "overflow_count",
    "overflow_row_bytes",
    "physical_rows",
    "active_pointer_absent",
}


def _resolved(value: Path | str) -> str:
    return str(Path(value).resolve())


def expected_roots(artifact_dir: Path) -> dict[str, str]:
    """Return every persistent and resident root frozen by v2."""
    artifact = Path(artifact_dir).resolve()
    return {
        "registry_root": _resolved(
            artifact / "m04r10" / "nasdaq-untouched-authority-registry"
        ),
        "source_full_root": _resolved(
            artifact / "poc" / "m04r" / "packed-bound-full"
        ),
        "resident_full_root": _resolved(
            Path("/dev/shm/market-analogues/m04r11-candidate-v2")
            / FROZEN_GENERATION_ID
        ),
        "candidate_root": _resolved(artifact / "m04r11" / "candidate-pools-v2"),
        "comparison_root": _resolved(
            artifact / "m04r11" / "candidate-comparison-v2"
        ),
        "verification_root": _resolved(
            artifact / "m04r11" / "candidate-comparison-verification-v2"
        ),
        "predecessor_candidate_root": _resolved(
            artifact / "m04r11" / "candidate-pools-v1"
        ),
        "predecessor_comparison_root": _resolved(
            artifact / "m04r11" / "candidate-comparison-v1"
        ),
        "authority_root": _resolved(
            artifact / "m04r11" / "authorities-sealed-v4"
        ),
    }


def validate_exact_roots(
    observed: Mapping[str, Any], artifact_dir: Path,
) -> tuple[str, ...]:
    expected = expected_roots(artifact_dir)
    failures: list[str] = []
    if set(observed) != set(expected):
        failures.append("root keys differ")
        return tuple(failures)
    try:
        normalized = {key: _resolved(str(observed[key])) for key in expected}
    except (OSError, TypeError, ValueError) as exc:
        return (f"malformed roots:{type(exc).__name__}:{exc}",)
    if normalized != expected:
        failures.append("exact v2 roots differ")
    outputs = tuple(Path(normalized[key]) for key in (
        "candidate_root", "comparison_root", "verification_root",
    ))
    protected = tuple(Path(normalized[key]) for key in (
        "registry_root", "source_full_root", "resident_full_root",
        "predecessor_candidate_root", "predecessor_comparison_root",
        "authority_root",
    ))
    pairs = (
        *((left, right) for left in outputs for right in protected),
        *((left, right) for index, left in enumerate(outputs)
          for right in outputs[index + 1:]),
    )
    if any(
        left == right or left.is_relative_to(right) or right.is_relative_to(left)
        for left, right in pairs
    ):
        failures.append("v2 output roots overlap or alias protected roots")
    return tuple(failures)


def derive_role_table(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    cases = list(registry.get("cases_data", []))
    if len(cases) != TOTAL_RECALL_CASES:
        raise ValueError("v2 role derivation requires exactly 60 registry cases")
    rows: list[dict[str, Any]] = []
    for ordinal, case in enumerate(cases):
        rows.append({
            "ordinal": ordinal,
            "case_id": str(case["case_id"]),
            "query_episode_id": str(case["episode_id"]),
            "performance_role": (
                EXPOSED_PERFORMANCE_ROLE
                if ordinal < EXPOSED_PERFORMANCE_CASES
                else CONFIRMATORY_PERFORMANCE_ROLE
            ),
            "recall_role": BLIND_RECALL_ROLE,
        })
    return rows


def validate_role_table(
    registry: Mapping[str, Any], observed: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[str, ...]:
    failures: list[str] = []
    try:
        cases = list(registry["cases_data"])
        if registry.get("registry_digest") != FROZEN_REGISTRY_DIGEST:
            failures.append("registry digest differs")
        ordered_ids = [str(case["episode_id"]) for case in cases]
        if stable_hash(ordered_ids) != FROZEN_CASE_ORDER_DIGEST:
            failures.append("registry case order differs")
        expected = derive_role_table(registry)
        table = expected if observed is None else [dict(row) for row in observed]
        if table != expected:
            failures.append("performance/recall role table differs")
        if stable_hash(expected) != FROZEN_ROLE_TABLE_DIGEST:
            failures.append("frozen role table digest differs")
        exposed = [
            row["query_episode_id"] for row in expected
            if row["performance_role"] == EXPOSED_PERFORMANCE_ROLE
        ]
        confirmatory = [
            row["query_episode_id"] for row in expected
            if row["performance_role"] == CONFIRMATORY_PERFORMANCE_ROLE
        ]
        if stable_hash(exposed) != FROZEN_EXPOSED_IDS_DIGEST:
            failures.append("exposed performance IDs differ")
        if stable_hash(confirmatory) != FROZEN_CONFIRMATORY_IDS_DIGEST:
            failures.append("confirmatory performance IDs differ")
        if tuple(
            (row["case_id"], row["query_episode_id"]) for row in expected[:7]
        ) != FROZEN_EXPOSED_CASES:
            failures.append("exposed predecessor cases differ")
        execution = confirmatory + exposed
        if stable_hash(execution) != FROZEN_EXECUTION_ORDER_DIGEST:
            failures.append("v2 execution order differs")
    except (KeyError, TypeError, ValueError) as exc:
        failures.append(f"malformed role table:{type(exc).__name__}:{exc}")
    return tuple(sorted(set(failures)))


def execution_query_ids(registry: Mapping[str, Any]) -> list[str]:
    failures = validate_role_table(registry)
    if failures:
        raise ValueError(f"frozen role table differs:{failures}")
    table = derive_role_table(registry)
    return [
        row["query_episode_id"] for role in (
            CONFIRMATORY_PERFORMANCE_ROLE, EXPOSED_PERFORMANCE_ROLE,
        ) for row in table if row["performance_role"] == role
    ]


def validate_predecessor_failure(payload: Mapping[str, Any]) -> tuple[str, ...]:
    failures: list[str] = []
    try:
        deterministic = {
            key: value for key, value in payload.items()
            if key not in {"created_at", "failure_digest"}
        }
        for key, expected in PREDECESSOR_FAILURE.items():
            if payload.get(key) != expected:
                failures.append(f"predecessor failure {key} differs")
        if payload.get("failure_digest") != stable_hash(deterministic):
            failures.append("predecessor failure digest does not reconstruct")
        summaries = list(payload.get("completed_case_summaries", []))
        if len(summaries) != EXPOSED_PERFORMANCE_CASES:
            failures.append("predecessor completed summaries differ")
        else:
            for ordinal, (summary, expected_identity) in enumerate(
                zip(summaries, FROZEN_EXPOSED_CASES, strict=True)
            ):
                expected_case, expected_query = expected_identity
                false_gates = summary.get("false_gates")
                expected_false = (
                    [] if ordinal < EXPOSED_PERFORMANCE_CASES - 1
                    else ["cold_scan_at_most_120_seconds"]
                )
                numeric = (
                    float(summary["cold_seconds"]),
                    float(summary["warm_second_seconds"]),
                    float(summary["peak_rss_mb"]),
                )
                if not all((
                    summary.get("ordinal") == ordinal,
                    summary.get("registry_case_id") == expected_case,
                    summary.get("query_episode_id") == expected_query,
                    summary.get("passed") is (ordinal < 6),
                    false_gates == expected_false,
                    all(isfinite(value) and value >= 0 for value in numeric),
                )):
                    failures.append("predecessor completed summary differs")
                    break
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        failures.append(f"malformed predecessor failure:{type(exc).__name__}:{exc}")
    return tuple(sorted(set(failures)))


def validate_predecessor_paths(artifact_dir: Path) -> tuple[str, ...]:
    roots = expected_roots(artifact_dir)
    candidate = Path(roots["predecessor_candidate_root"])
    comparison = Path(roots["predecessor_comparison_root"])
    failures: list[str] = []
    try:
        failure_path = candidate / "FAILED.json"
        if not failure_path.is_file():
            failures.append("predecessor FAILED.json is missing")
        else:
            import json

            failures.extend(validate_predecessor_failure(
                json.loads(failure_path.read_text())
            ))
        if any((candidate / name).exists() for name in (
            "SEALED.json", "candidate-matrix.json",
        )):
            failures.append("predecessor candidate root is not terminal-unsealed")
        if any((comparison / name).exists() for name in (
            "RESULTS_OPENED.json", "candidate-comparison.json", "SEALED.json",
        )):
            failures.append("predecessor authority comparison was opened")
    except (OSError, TypeError, ValueError) as exc:
        failures.append(f"malformed predecessor paths:{type(exc).__name__}:{exc}")
    return tuple(sorted(set(failures)))


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and value == value.lower() and set(value).issubset(_HEX)
    )


def validate_source_pack_binding(
    payload: Mapping[str, Any], roots: Mapping[str, str],
) -> tuple[str, ...]:
    failures: list[str] = []
    try:
        if set(payload) != _SOURCE_PACK_FIELDS:
            failures.append("source pack fields differ")
            return tuple(failures)
        generation_root = (
            Path(roots["source_full_root"]) / "store" / "generations"
            / FROZEN_GENERATION_ID
        ).resolve()
        rows_file = str(payload["rows_file"])
        overflow_file = str(payload["overflow_file"])
        if any(
            not value or Path(value).name != value
            for value in (rows_file, overflow_file)
        ):
            failures.append("source pack filenames are unsafe")
        expected_paths = {
            "source_full_root": roots["source_full_root"],
            "manifest_path": str((generation_root / "manifest.json").resolve()),
            "rows_path": str((generation_root / rows_file).resolve()),
            "overflow_path": str((generation_root / overflow_file).resolve()),
        }
        if any(_resolved(str(payload[key])) != value for key, value in expected_paths.items()):
            failures.append("source pack paths differ")
        if payload.get("generation_id") != FROZEN_GENERATION_ID:
            failures.append("source pack generation differs")
        if payload.get("manifest_digest") != FROZEN_GENERATION_ID:
            failures.append("source pack manifest digest differs")
        if any(not _is_digest(payload.get(key)) for key in _DIGEST_FIELDS):
            failures.append("source pack digest format differs")
        numeric = {
            key: int(payload[key]) for key in (
                "rows_bytes", "row_count", "row_bytes", "overflow_bytes",
                "overflow_count", "overflow_row_bytes", "physical_rows",
            )
        }
        if not all((
            numeric["row_count"] > 0,
            numeric["overflow_count"] >= 0,
            numeric["row_bytes"] == FROZEN_ROW_BYTES,
            numeric["overflow_row_bytes"] == FROZEN_OVERFLOW_ROW_BYTES,
            numeric["rows_bytes"] == numeric["row_count"] * FROZEN_ROW_BYTES,
            numeric["overflow_bytes"]
            == numeric["overflow_count"] * FROZEN_OVERFLOW_ROW_BYTES,
            numeric["physical_rows"]
            == numeric["row_count"] + numeric["overflow_count"],
            payload.get("active_pointer_absent") is True,
        )):
            failures.append("source pack physical accounting differs")
    except (KeyError, OSError, TypeError, ValueError, OverflowError) as exc:
        failures.append(f"malformed source pack:{type(exc).__name__}:{exc}")
    return tuple(sorted(set(failures)))


def _manifest_valid(payload: Mapping[str, Any]) -> bool:
    try:
        files = dict(payload["files"])
        if not files or set(payload) != {"files", "digest"}:
            return False
        if not all(
            isinstance(name, str) and name and not Path(name).is_absolute()
            and ".." not in Path(name).parts and _is_digest(digest)
            for name, digest in files.items()
        ):
            return False
        return payload.get("digest") == stable_hash(files)
    except (KeyError, TypeError, ValueError):
        return False


def _environment_valid(payload: Mapping[str, Any]) -> bool:
    try:
        deterministic = {key: value for key, value in payload.items() if key != "digest"}
        return (
            bool(deterministic)
            and payload.get("digest") == stable_hash(deterministic)
        )
    except (TypeError, ValueError):
        return False


def producer_contract_digest(payload: Mapping[str, Any]) -> str:
    return stable_hash({
        key: value for key, value in payload.items() if key != "contract_digest"
    })


def semantic_case_digest(payload: Mapping[str, Any]) -> str:
    return stable_hash({
        key: value for key, value in payload.items()
        if key not in {"created_at", "semantic_digest"}
    })


def performance_attempt_digest(payload: Mapping[str, Any]) -> str:
    return stable_hash({
        key: value for key, value in payload.items()
        if key not in {"created_at", "performance_digest"}
    })


def semantic_matrix_digest(payload: Mapping[str, Any]) -> str:
    return stable_hash({
        key: value for key, value in payload.items()
        if key not in {"created_at", "elapsed_seconds", "result_digest"}
    })


def performance_matrix_digest(payload: Mapping[str, Any]) -> str:
    return stable_hash({
        key: value for key, value in payload.items()
        if key not in {"created_at", "elapsed_seconds", "result_digest"}
    })


def terminal_digest(payload: Mapping[str, Any], digest_field: str) -> str:
    return stable_hash({
        key: value for key, value in payload.items()
        if key not in {"created_at", digest_field}
    })


def preregistration_digest(payload: Mapping[str, Any]) -> str:
    return stable_hash({
        key: value for key, value in payload.items()
        if key != "preregistration_digest"
    })


def expected_preregistration_path(repository_root: Path) -> Path:
    """Return the only path from which a v2 runner may load preregistration."""
    return Path(repository_root).resolve() / PREREGISTRATION_RELATIVE_PATH


def build_producer_contract(
    registry: Mapping[str, Any], artifact_dir: Path, *,
    source_pack: Mapping[str, Any], implementation_manifest: Mapping[str, Any],
    environment_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    role_failures = validate_role_table(registry)
    roots = expected_roots(artifact_dir)
    pack_failures = validate_source_pack_binding(source_pack, roots)
    if role_failures or pack_failures:
        raise ValueError(
            f"cannot build frozen v2 contract:{role_failures}:{pack_failures}"
        )
    if not _manifest_valid(implementation_manifest):
        raise ValueError("implementation manifest is invalid")
    if not _environment_valid(environment_manifest):
        raise ValueError("environment manifest is invalid")
    roles = derive_role_table(registry)
    ordered_ids = [str(case["episode_id"]) for case in registry["cases_data"]]
    deterministic: dict[str, Any] = {
        "schema_version": PRODUCER_CONTRACT_SCHEMA,
        "registry_digest": FROZEN_REGISTRY_DIGEST,
        "ordered_query_ids": ordered_ids,
        "ordered_query_ids_digest": FROZEN_CASE_ORDER_DIGEST,
        "role_table": roles,
        "role_table_digest": FROZEN_ROLE_TABLE_DIGEST,
        "exposed_query_ids_digest": FROZEN_EXPOSED_IDS_DIGEST,
        "confirmatory_query_ids_digest": FROZEN_CONFIRMATORY_IDS_DIGEST,
        "execution_query_ids": execution_query_ids(registry),
        "execution_query_ids_digest": FROZEN_EXECUTION_ORDER_DIGEST,
        "generation_id": FROZEN_GENERATION_ID,
        "proposal_contract_digest": FROZEN_PROPOSAL_CONTRACT_DIGEST,
        "route_quotas": deepcopy(FROZEN_ROUTE_QUOTAS),
        "request": deepcopy(FROZEN_REQUEST),
        "roots": roots,
        "predecessor_failure": deepcopy(PREDECESSOR_FAILURE),
        "source_pack": deepcopy(dict(source_pack)),
        "resident_policy": deepcopy(RESIDENT_POLICY),
        "artifact_schemas": deepcopy(ARTIFACT_SCHEMAS),
        "resident_ready_schema": RESIDENT_READY_SCHEMA,
        "resident_binding_schema": RESIDENT_BINDING_SCHEMA,
        "scan_protocol": deepcopy(SCAN_PROTOCOL),
        "performance_limits": deepcopy(PERFORMANCE_LIMITS),
        "semantic_case_schema": SEMANTIC_CASE_SCHEMA,
        "performance_attempt_schema": PERFORMANCE_ATTEMPT_SCHEMA,
        "semantic_matrix_schema": SEMANTIC_MATRIX_SCHEMA,
        "semantic_seal_schema": SEMANTIC_SEAL_SCHEMA,
        "performance_matrix_schema": PERFORMANCE_MATRIX_SCHEMA,
        "performance_final_schema": PERFORMANCE_FINAL_SCHEMA,
        "run_ledger_event_schema": RUN_LEDGER_EVENT_SCHEMA,
        "run_ledger_head_schema": RUN_LEDGER_HEAD_SCHEMA,
        "incomplete_schema": INCOMPLETE_SCHEMA,
        "run_complete_schema": RUN_COMPLETE_SCHEMA,
        "case_bundle_schema": CASE_BUNDLE_SCHEMA,
        "results_opened_schema": RESULTS_OPENED_SCHEMA,
        "comparison_matrix_schema": COMPARISON_MATRIX_SCHEMA,
        "comparison_seal_schema": COMPARISON_SEAL_SCHEMA,
        "comparison_verification_schema": COMPARISON_VERIFICATION_SCHEMA,
        "preregistration_schema": PREREGISTRATION_SCHEMA,
        "preregistration_relative_path": PREREGISTRATION_RELATIVE_PATH,
        "marker_and_resume_policy": deepcopy(MARKER_AND_RESUME_POLICY),
        "preregistration_policy": deepcopy(PREREGISTRATION_POLICY),
        "comparison_policy": deepcopy(COMPARISON_POLICY),
        "semantic_case_gates": list(SEMANTIC_CASE_GATES),
        "performance_attempt_gates": list(PERFORMANCE_ATTEMPT_GATES),
        "semantic_matrix_gates": list(SEMANTIC_MATRIX_GATES),
        "performance_matrix_gates": list(PERFORMANCE_MATRIX_GATES),
        "claims_policy": deepcopy(CLAIMS_POLICY),
        "implementation_manifest": deepcopy(dict(implementation_manifest)),
        "environment_manifest": deepcopy(dict(environment_manifest)),
        "real_forward_outcomes_accessed": False,
    }
    return {**deterministic, "contract_digest": stable_hash(deterministic)}


def validate_producer_contract(
    payload: Mapping[str, Any], registry: Mapping[str, Any], artifact_dir: Path,
    *, expected_source_pack: Mapping[str, Any],
    expected_implementation_manifest: Mapping[str, Any],
    expected_environment_manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    failures: list[str] = []
    try:
        expected = build_producer_contract(
            registry, artifact_dir, source_pack=expected_source_pack,
            implementation_manifest=expected_implementation_manifest,
            environment_manifest=expected_environment_manifest,
        )
        if dict(payload) != expected:
            failures.append("producer contract differs from exact frozen v2 contract")
        if payload.get("contract_digest") != producer_contract_digest(payload):
            failures.append("producer contract digest differs")
        failures.extend(validate_exact_roots(
            dict(payload.get("roots", {})), artifact_dir,
        ))
        failures.extend(validate_role_table(
            registry, payload.get("role_table", []),
        ))
        failures.extend(validate_source_pack_binding(
            dict(payload.get("source_pack", {})), expected_roots(artifact_dir),
        ))
    except (KeyError, TypeError, ValueError, OSError) as exc:
        failures.append(f"malformed producer contract:{type(exc).__name__}:{exc}")
    return tuple(sorted(set(failures)))


def build_preregistration_document(
    registry: Mapping[str, Any], artifact_dir: Path, repository_root: Path, *,
    source_pack: Mapping[str, Any], implementation_manifest: Mapping[str, Any],
    environment_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the deterministic document that must later be committed.

    This pure builder is for the pre-launch preparation step.  A runner must
    never trust its return value directly; it must call
    :func:`load_and_validate_preregistration`, which reads the pre-existing
    document from the single frozen repository path and exact-compares it to
    current source, implementation and environment observations.
    """
    contract = build_producer_contract(
        registry, artifact_dir, source_pack=source_pack,
        implementation_manifest=implementation_manifest,
        environment_manifest=environment_manifest,
    )
    deterministic: dict[str, Any] = {
        "schema_version": PREREGISTRATION_SCHEMA,
        "relative_path": PREREGISTRATION_RELATIVE_PATH,
        "resolved_path": str(expected_preregistration_path(repository_root)),
        "producer_contract": contract,
        "producer_contract_digest": contract["contract_digest"],
        "source_pack_binding_digest": stable_hash(dict(source_pack)),
        "implementation_manifest_digest": implementation_manifest["digest"],
        "environment_manifest_digest": environment_manifest["digest"],
        "artifact_schemas": deepcopy(ARTIFACT_SCHEMAS),
        "marker_and_resume_policy": deepcopy(MARKER_AND_RESUME_POLICY),
        "comparison_policy": deepcopy(COMPARISON_POLICY),
        "preregistration_policy": deepcopy(PREREGISTRATION_POLICY),
        "authority_results_opened": False,
        "candidate_authority_comparison_opened": False,
        "real_forward_outcomes_accessed": False,
    }
    return {
        **deterministic,
        "preregistration_digest": stable_hash(deterministic),
    }


def validate_preregistration_document(
    payload: Mapping[str, Any], registry: Mapping[str, Any], artifact_dir: Path,
    repository_root: Path, *, observed_path: Path,
    expected_source_pack: Mapping[str, Any],
    expected_implementation_manifest: Mapping[str, Any],
    expected_environment_manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    """Exact-validate a pre-existing preregistration against current state."""
    failures: list[str] = []
    try:
        frozen_path = expected_preregistration_path(repository_root)
        if Path(observed_path).resolve() != frozen_path:
            failures.append("preregistration was not loaded from exact committed path")
        expected = build_preregistration_document(
            registry, artifact_dir, repository_root,
            source_pack=expected_source_pack,
            implementation_manifest=expected_implementation_manifest,
            environment_manifest=expected_environment_manifest,
        )
        if dict(payload) != expected:
            failures.append(
                "preregistration differs from current exact frozen v2 contract"
            )
        if payload.get("schema_version") != PREREGISTRATION_SCHEMA:
            failures.append("preregistration schema differs")
        if payload.get("relative_path") != PREREGISTRATION_RELATIVE_PATH:
            failures.append("preregistration relative path differs")
        if payload.get("resolved_path") != str(frozen_path):
            failures.append("preregistration resolved path differs")
        if payload.get("preregistration_digest") != preregistration_digest(payload):
            failures.append("preregistration digest differs")
        contract = dict(payload.get("producer_contract", {}))
        failures.extend(validate_producer_contract(
            contract, registry, artifact_dir,
            expected_source_pack=expected_source_pack,
            expected_implementation_manifest=expected_implementation_manifest,
            expected_environment_manifest=expected_environment_manifest,
        ))
        if payload.get("producer_contract_digest") != contract.get("contract_digest"):
            failures.append("preregistration producer contract digest differs")
        if payload.get("artifact_schemas") != ARTIFACT_SCHEMAS:
            failures.append("preregistration artifact schemas differ")
        if payload.get("marker_and_resume_policy") != MARKER_AND_RESUME_POLICY:
            failures.append("preregistration marker/no-resume policy differs")
        if payload.get("comparison_policy") != COMPARISON_POLICY:
            failures.append("preregistration comparison policy differs")
        if payload.get("preregistration_policy") != PREREGISTRATION_POLICY:
            failures.append("preregistration loading policy differs")
        if any(payload.get(key) is not False for key in (
            "authority_results_opened", "candidate_authority_comparison_opened",
            "real_forward_outcomes_accessed",
        )):
            failures.append("preregistration claims opened truth or outcomes")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        failures.append(f"malformed preregistration:{type(exc).__name__}:{exc}")
    return tuple(sorted(set(failures)))


def load_and_validate_preregistration(
    registry: Mapping[str, Any], artifact_dir: Path, repository_root: Path, *,
    expected_source_pack: Mapping[str, Any],
    expected_implementation_manifest: Mapping[str, Any],
    expected_environment_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Load only the committed preregistration and fail closed on any drift."""
    path = expected_preregistration_path(repository_root)
    if not path.is_file():
        raise ValueError(f"committed v2 preregistration is missing: {path}")
    if path.is_symlink() or path.resolve() != path:
        raise ValueError("committed v2 preregistration path contains a symbolic link")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read committed v2 preregistration: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("committed v2 preregistration must be a JSON object")
    failures = validate_preregistration_document(
        payload, registry, artifact_dir, repository_root, observed_path=path,
        expected_source_pack=expected_source_pack,
        expected_implementation_manifest=expected_implementation_manifest,
        expected_environment_manifest=expected_environment_manifest,
    )
    if failures:
        raise ValueError(f"committed v2 preregistration differs:{failures}")
    return payload
