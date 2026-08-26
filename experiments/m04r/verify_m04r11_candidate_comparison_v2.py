"""Independent, read-only verifier for the sealed M04R-11 v2 comparison.

This file intentionally imports neither the v2 contract nor its producer or
comparator.  Frozen identities and digest rules are repeated here so corruption
of a shared experiment helper cannot make production and verification agree.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
from math import isfinite
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping
from uuid import uuid4

import pandas as pd

from market_analogues.adapters import source_from_spec
from market_analogues.config import load_config
from market_analogues.certified_packed_search import certified_packed_search_contract
from market_analogues.episodes import build_episode
from market_analogues.representation import represent, representation_input_digest
from market_analogues.packed_bound_search import (
    packed_bound_search_contract, packed_bound_threshold_scan_contract,
)
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import InstrumentKey


VERIFICATION_SCHEMA = "candidate-recall-comparison-verification-v2"
PRODUCER_SCHEMA = "candidate-recall-resident-producer-contract-v1"
PREREGISTRATION_SCHEMA = "candidate-recall-resident-preregistration-v1"
RESIDENT_BINDING_SCHEMA = "candidate-resident-ready-binding-v2"
RESIDENT_READY_SCHEMA = "m04r-resident-packed-store-ready-v2"
CASE_BUNDLE_SCHEMA = "candidate-resident-case-bundle-v1"
SEMANTIC_CASE_SCHEMA = "candidate-recall-semantic-case-v3"
PERFORMANCE_ATTEMPT_SCHEMA = "candidate-resident-performance-attempt-v1"
SEMANTIC_MATRIX_SCHEMA = "candidate-recall-semantic-matrix-v1"
SEMANTIC_SEAL_SCHEMA = "candidate-recall-semantic-seal-v1"
PERFORMANCE_MATRIX_SCHEMA = "candidate-resident-performance-matrix-v1"
PERFORMANCE_FINAL_SCHEMA = "candidate-resident-performance-final-v1"
LEDGER_EVENT_SCHEMA = "candidate-resident-run-ledger-event-v1"
LEDGER_HEAD_SCHEMA = "candidate-resident-run-ledger-head-v1"
RUN_COMPLETE_SCHEMA = "candidate-resident-run-complete-v1"
RESULTS_OPENED_SCHEMA = "candidate-recall-results-opened-v2"
COMPARISON_MATRIX_SCHEMA = "candidate-recall-comparison-matrix-v2"
COMPARISON_SEAL_SCHEMA = "candidate-recall-comparison-seal-v2"
AUTHORITY_CONTRACT_SCHEMA = "m04r11-authority-build-contract-v4"
AUTHORITY_CASE_SCHEMA = "m04r11-certified-authority-case-v4"
AUTHORITY_MATRIX_SCHEMA = "m04r11-certified-authority-matrix-v4"
AUTHORITY_SEAL_SCHEMA = "m04r11-authority-seal-v4"
PREREGISTRATION_RELATIVE_PATH = (
    "experiments/m04r/m04r11_candidate_v2_preregistered.json"
)
SEARCH_SCHEMA = "m04r-global-bound-proposal-v1"
RESIDENT_CONTENT_SCHEMA = "m04r-resident-packed-store-content-v1"
VALIDATION_OBSERVATION_SCHEMA = "m04r-resident-validation-observation-v1"
FILE_IDENTITY_LEASE_SCHEMA = "m04r-resident-file-identity-lease-v1"
IMPLEMENTATION_FILES = (
    "experiments/m04r/m04r11_candidate_matrix_v2.py",
    "experiments/m04r/m04r11_candidate_v2_contract.py",
    "experiments/m04r/compare_m04r11_candidate_matrix_v2.py",
    "experiments/m04r/verify_m04r11_candidate_comparison_v2.py",
    "experiments/m04r/prepare_m04r11_resident_mirror.py",
    "experiments/m04r/preregister_m04r11_candidate_v2.py",
    "experiments/m04r/preflight_m04r11_candidate_v2.py",
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
    "candle_volatility": 64, "coarse": 64, "composite": 1_000,
    "market_context": 64, "price": 64, "stage": 64,
    "structural": 64, "volume_shock": 64,
}
FROZEN_REQUEST = {
    "cross_dataset": False, "deduplicate_overlaps": True,
    "max_per_instrument": 3, "minimum_history_gap_bars": 60,
    "quality_tiers": ["A", "B"], "top_k": 20,
}
PERFORMANCE_LIMITS = {
    "resident_first_seconds": 120.0,
    "resident_repeat_seconds": 60.0,
    "worker_rss_mib": 1_536.0,
}
ARTIFACT_SCHEMAS = {
    "producer_contract": PRODUCER_SCHEMA,
    "resident_ready": RESIDENT_READY_SCHEMA,
    "resident_content": "m04r-resident-packed-store-content-v1",
    "resident_binding": RESIDENT_BINDING_SCHEMA,
    "semantic_case": SEMANTIC_CASE_SCHEMA,
    "performance_attempt": PERFORMANCE_ATTEMPT_SCHEMA,
    "run_ledger_event": LEDGER_EVENT_SCHEMA,
    "run_ledger_head": LEDGER_HEAD_SCHEMA,
    "incomplete": "candidate-resident-incomplete-v1",
    "semantic_matrix": SEMANTIC_MATRIX_SCHEMA,
    "semantic_seal": SEMANTIC_SEAL_SCHEMA,
    "performance_matrix": PERFORMANCE_MATRIX_SCHEMA,
    "performance_final": PERFORMANCE_FINAL_SCHEMA,
    "run_complete": RUN_COMPLETE_SCHEMA,
    "case_bundle": CASE_BUNDLE_SCHEMA,
    "results_opened": RESULTS_OPENED_SCHEMA,
    "comparison_matrix": COMPARISON_MATRIX_SCHEMA,
    "comparison_seal": COMPARISON_SEAL_SCHEMA,
    "comparison_verification": VERIFICATION_SCHEMA,
    "preregistration": PREREGISTRATION_SCHEMA,
}
SCAN_PROTOCOL = {
    "execution": "serial fresh spawned process per query",
    "execution_order": "53 confirmatory-untouched cases then 7 exposed recovery cases",
    "engine": "bounded ordered eight-thread legacy-v1 scan over verified resident mirror",
    "outer_threads": 8, "numba_threads_per_scorer": 1,
    "maximum_in_flight_blocks": 8,
    "reduction": "strict requested physical block order into unchanged stable route heaps",
    "cache_advice": (
        "none; resident readiness is explicit and POSIX_FADV_DONTNEED is forbidden"
    ),
    "resident_first": {"block_rows": 4_096, "order": "forward"},
    "resident_reverse": {"block_rows": 4_097, "order": "reverse"},
    "resident_repeat": {"block_rows": 4_093, "order": "forward"},
    "primary_performance_attempts_per_case": 1,
    "maximum_semantic_recovery_attempts_per_case": 0,
    "worker_error_policy": (
        "fail fast before the next case; every failure is terminal and no resume is authorized"
    ),
}
RESIDENT_POLICY = {
    "filesystem": "tmpfs",
    "storage_claim": (
        "tmpfs-backed empirical query latency; RAM residency is not guaranteed when swap is enabled"
    ),
    "reserve_bytes": 1_073_741_824,
    "publication_modes": ["copy-and-validate", "validate-existing"],
    "content": "byte-identical generation manifest, rows and overflow sidecar",
    "readiness": "full content-hash verification and pre-touch before any registry query",
    "mutation_detection": (
        "full content-hash verification at readiness and after the final query"
    ),
    "attempt_identity": (
        "each attempt holds and verifies an exact regular-file identity lease "
        "from start through end"
    ),
    "active_pointer_allowed": False, "symbolic_links_allowed": False,
    "files_must_be_contained_regular_and_same_tmpfs_st_dev": True,
    "cold_storage_claimed": False,
}
COMPARISON_POLICY = {
    "top_k": 20, "minimum_retained_per_case": 19,
    "minimum_retained_aggregate": 1_188, "aggregate_denominator": 1_200,
    "twenty_of_twenty_is_descriptive_only": True,
    "results_opened_marker_must_be_written_fsynced_before_authority_read": True,
    "valid_semantic_seal_required": True,
    "terminal_performance_evidence_required": True,
    "performance_pass_required": False,
    "authority_results_opened_before_candidate_semantic_seal": False,
    "comparison_is_one_shot": True, "real_forward_outcomes_accessed": False,
}
CLAIMS_POLICY = {
    "recall_holdout_cases": 60,
    "recall_claim": "all 60 cases remain blind until the v2 results-opened marker",
    "confirmatory_performance_cases": 53,
    "confirmatory_performance_claim": (
        "only the 53 previously unmeasured cases support the v2 confirmatory performance claim"
    ),
    "exposed_performance_cases": 7,
    "exposed_performance_claim": "development/recovery regression only",
    "semantic_seal_independent_of_performance": True,
    "performance_failure_must_not_block_recall_comparison": True,
    "performance_must_be_terminal_before_recall_comparison": True,
    "v1_original_combined_gate_passed": False,
    "production_promotion_authorized": False,
    "real_forward_outcomes_accessed": False,
}
PREDECESSOR_FAILURE = {
    "schema_version": "candidate-recall-producer-terminal-failure-v1",
    "failure_digest": "64df2ed8f3deff98193552a8b2bb085432c50788e299a8e4e62828a89c8c7843",
    "producer_contract_digest": "e6d971081f288375df3ad66baad0300eb2d452860b464d6d9c111a6c75d657ad",
    "status": "terminal_performance_failure_after_interruption",
    "completed_cases": 7, "failed_case_id": "nasdaq-GABC-current-252",
    "failed_gates": ["cold_scan_at_most_120_seconds"],
    "candidate_pools_sealed": False, "authority_results_opened": False,
    "candidate_authority_comparison_opened": False, "resume_authorized": False,
    "claims_policy": {
        "original_combined_gate_passed": False, "performance_exposed_cases": 7,
        "recall_holdout_cases_still_blind": 60,
        "remaining_untouched_performance_cases": 53,
    },
}
MARKER_AND_RESUME_POLICY = {
    "producer_contract_path": "candidate-contract.json",
    "resident_binding_path": "RESIDENT_READY.json",
    "ledger_events_directory": "ledger/events", "ledger_head_path": "ledger/HEAD.json",
    "semantic_cases_directory": "semantic-cases",
    "performance_attempts_directory": "performance-attempts",
    "case_bundles_directory": "case-bundles",
    "case_bundle_filename": "{execution_ordinal:03d}-{query_episode_id}.json",
    "incomplete_marker_path": "INCOMPLETE.json", "semantic_matrix_path": "semantic-matrix.json",
    "semantic_seal_path": "SEMANTIC_SEALED.json",
    "performance_matrix_path": "performance-matrix.json",
    "performance_final_path": "PERFORMANCE_FINAL.json", "run_complete_path": "RUN_COMPLETE.json",
    "results_opened_path": "RESULTS_OPENED.json", "comparison_matrix_path": "candidate-comparison.json",
    "comparison_seal_path": "SEALED.json", "comparison_verification_path": "verification.json",
    "resume_authorized": False, "partial_ledger_resume_authorized": False,
    "semantic_recovery_authorized": False, "incomplete_marker_is_terminal": True,
    "semantic_failure_is_terminal": True, "worker_failure_is_terminal": True,
    "existing_case_overwrite_authorized": False,
    "existing_terminal_marker_overwrite_authorized": False,
    "fresh_empty_candidate_root_required": True,
}
PREREGISTRATION_POLICY = {
    "relative_path": PREREGISTRATION_RELATIVE_PATH,
    "must_be_committed_before_launch": True, "must_preexist_candidate_output": True,
    "load_from_exact_relative_path_only": True, "generate_and_trust_at_launch_forbidden": True,
    "runtime_contract_must_exactly_equal_embedded_contract": True,
    "runtime_source_implementation_environment_must_exactly_match": True,
    "overwrite_at_launch_authorized": False,
    "authority_results_may_be_read_during_generation_or_validation": False,
}
RESIDENT_LATENCY_SCOPE = (
    "tmpfs-backed empirical query; not durable-storage disk-cold and not "
    "guaranteed unswappable RAM"
)
SEMANTIC_GATES = (
    "resident_content_matches_contract",
    "query_identity_prefix_and_representation_match",
    "three_scan_digest_and_block_order_invariance",
    "internal_eligible_row_accounting", "physical_row_accounting",
    "frozen_route_quotas", "candidate_digest_order_and_routes_reconstruct",
    "zero_temporal_overlap_tier_duplicate_errors",
    "real_forward_outcomes_excluded",
)
PERFORMANCE_GATES = (
    "same_ready_instance_at_start_and_end",
    "measurements_finite_nonnegative_and_task_contains_scans",
    "resident_first_scan_at_most_120_seconds",
    "resident_repeat_scan_at_most_60_seconds",
    "worker_rss_at_most_1536_mib", "primary_attempt_completed",
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
AUTHORITY_CASE_OMITTED = {
    "created_at", "proposal_seconds", "amortized_proposal_seconds",
    "exact_seconds", "final_exact_seconds", "frontier_attempt_measurements",
    "peak_rss_mb", "result_digest", "checkpoint_integrity_digest",
}
AUTHORITY_MATRIX_OMITTED = {
    "created_at", "elapsed_seconds", "p95_exact_seconds",
    "maximum_exact_seconds", "total_exact_seconds", "maximum_worker_rss_mb",
    "p95_search_seconds", "maximum_search_seconds", "total_search_seconds",
    "measurements", "performance_gates", "performance_gate_passed",
    "measurement_integrity_digest", "result_digest",
}
LEDGER_GENESIS = "0" * 64
_HEX = frozenset("0123456789abcdef")


def _hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(raw.encode()).hexdigest()


def _file_hash(path: Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _environment_manifest() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("numpy", "pandas", "numba"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    deterministic = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(), "machine": platform.machine(),
        "packages": packages, "cpu_count": os.cpu_count(),
        "numba_num_threads": os.environ.get("NUMBA_NUM_THREADS"),
    }
    return {**deterministic, "digest": _hash(deterministic)}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if type(value) is not dict:
        raise ValueError(f"JSON object required: {path}")
    return value


def _timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} timestamp is absent")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} timestamp is malformed") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} timestamp is not timezone-aware")
    return parsed


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def expected_roots(artifact_dir: Path) -> dict[str, str]:
    artifact = artifact_dir.resolve()
    return {
        "registry_root": str((
            artifact / "m04r10" / "nasdaq-untouched-authority-registry"
        ).resolve()),
        "source_full_root": str((
            artifact / "poc" / "m04r" / "packed-bound-full"
        ).resolve()),
        "resident_full_root": str((
            Path("/dev/shm/market-analogues/m04r11-candidate-v2")
            / FROZEN_GENERATION_ID
        ).resolve()),
        "candidate_root": str((artifact / "m04r11" / "candidate-pools-v2").resolve()),
        "comparison_root": str((artifact / "m04r11" / "candidate-comparison-v2").resolve()),
        "verification_root": str((
            artifact / "m04r11" / "candidate-comparison-verification-v2"
        ).resolve()),
        "predecessor_candidate_root": str((
            artifact / "m04r11" / "candidate-pools-v1"
        ).resolve()),
        "predecessor_comparison_root": str((
            artifact / "m04r11" / "candidate-comparison-v1"
        ).resolve()),
        "authority_root": str((artifact / "m04r11" / "authorities-sealed-v4").resolve()),
    }


def _roles(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{
        "ordinal": index, "case_id": case["case_id"],
        "query_episode_id": case["episode_id"],
        "performance_role": (
            "exposed_recovery_regression" if index < 7 else "confirmatory_untouched"
        ),
        "recall_role": "blind_primary",
    } for index, case in enumerate(registry["cases_data"])]


def _execution(roles: list[dict[str, Any]]) -> list[str]:
    return [
        row["query_episode_id"] for role in (
            "confirmatory_untouched", "exposed_recovery_regression",
        ) for row in roles if row["performance_role"] == role
    ]


def _manifest_valid(payload: Any, repository_root: Path | None = None) -> bool:
    if type(payload) is not dict or set(payload) != {"files", "digest"}:
        return False
    files = payload.get("files")
    if type(files) is not dict or not files or payload.get("digest") != _hash(files):
        return False
    for relative, digest in files.items():
        if (
            not isinstance(relative, str) or not relative
            or Path(relative).is_absolute() or ".." in Path(relative).parts
            or not _is_digest(digest)
        ):
            return False
        if repository_root is not None:
            path = (repository_root / relative).resolve()
            if not path.is_relative_to(repository_root.resolve()) or (
                not path.is_file() or _file_hash(path) != digest
            ):
                return False
    return True


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and value == value.lower()
        and set(value).issubset(_HEX)
    )


def _validate_source_pack(
    pack: Mapping[str, Any], roots: Mapping[str, str],
) -> None:
    if not all((
        pack.get("generation_id") == FROZEN_GENERATION_ID,
        pack.get("manifest_digest") == FROZEN_GENERATION_ID,
        pack.get("source_full_root") == roots["source_full_root"],
        pack.get("row_bytes") == 2_432,
        pack.get("overflow_row_bytes") == 32,
        pack.get("rows_bytes") == pack.get("row_count", -1) * 2_432,
        pack.get("overflow_bytes") == pack.get("overflow_count", -1) * 32,
        pack.get("physical_rows")
        == pack.get("row_count", -1) + pack.get("overflow_count", -1),
        pack.get("active_pointer_absent") is True,
    )):
        raise ValueError("source pack identity or accounting differs")
    for field in (
        "manifest_sha256", "manifest_digest", "provenance_digest",
        "rows_sha256", "overflow_sha256",
    ):
        if not _is_digest(pack.get(field)):
            raise ValueError("source pack digest differs")
    expected = {
        "manifest_path": "manifest.json", "rows_path": str(pack["rows_file"]),
        "overflow_path": str(pack["overflow_file"]),
    }
    generation = (
        Path(roots["source_full_root"]) / "store" / "generations"
        / FROZEN_GENERATION_ID
    )
    for field, name in expected.items():
        path = Path(str(pack[field]))
        if path.resolve() != (generation / name).resolve() or not path.is_file():
            raise ValueError("source pack path differs")
    checks = (
        (Path(pack["manifest_path"]), pack["manifest_sha256"], None),
        (Path(pack["rows_path"]), pack["rows_sha256"], pack["rows_bytes"]),
        (Path(pack["overflow_path"]), pack["overflow_sha256"], pack["overflow_bytes"]),
    )
    for path, digest, size in checks:
        if _file_hash(path) != digest or (size is not None and path.stat().st_size != size):
            raise ValueError("source pack physical content differs")


def _resident_content(pack: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = Path(str(pack["manifest_path"]))
    manifest = _read(manifest_path)
    files = {
        "manifest": {"bytes": manifest_path.stat().st_size, "sha256": pack["manifest_sha256"]},
        "rows": {"bytes": pack["rows_bytes"], "sha256": pack["rows_sha256"]},
        "overflow": {"bytes": pack["overflow_bytes"], "sha256": pack["overflow_sha256"]},
    }
    return {
        "schema_version": RESIDENT_CONTENT_SCHEMA,
        "generation_id": FROZEN_GENERATION_ID,
        "provenance_digest": pack["provenance_digest"],
        "manifest_digest": pack["manifest_digest"],
        "pack_contract_digest": manifest["pack_contract_digest"],
        "quantized_bound_contract_digest": manifest["quantized_bound_contract_digest"],
        "physical_generation_bytes": sum(int(row["bytes"]) for row in files.values()),
        "source_files": files, "mirror_files": files,
    }


def _query_context(config_path: Path, case: Mapping[str, Any]) -> dict[str, Any]:
    config = load_config(config_path)
    source = source_from_spec(config.datasets["nasdaq"])
    episode = build_episode(
        source, InstrumentKey("nasdaq", str(case["symbol"])), str(case["cutoff"]),
        int(case["lookback"]), str(case["representation_version"]),
    )
    if episode.key.id != case["episode_id"]:
        raise ValueError("rebuilt verification query differs from registry")
    instrument = InstrumentKey("nasdaq", str(case["symbol"]))
    stock = asdict(source.causal_prefix_fingerprint(instrument, str(case["cutoff"])))
    benchmark_raw = source.benchmark_causal_prefix_fingerprint(str(case["cutoff"]))
    benchmark = asdict(benchmark_raw) if benchmark_raw is not None else None
    if stock != case["stock_prefix"] or benchmark != case["benchmark_prefix"]:
        raise ValueError("verification query causal prefix differs from registry")
    return {
        "query_episode_id": episode.key.id,
        "query_symbol": episode.key.instrument.source_symbol,
        "query_start_ns": int(pd.Timestamp(episode.bars.timestamp.iloc[0]).value),
        "latest_eligible_ns": int(latest_eligible_cutoff(episode, 60).value),
        "query_stock_prefix": stock, "query_benchmark_prefix": benchmark,
        "query_representation_digest": representation_input_digest(represent(episode)),
    }


def _candidate_digest(rows: list[dict[str, Any]]) -> str:
    return _hash([{
        "episode_id": str(row["episode_id"]), "symbol": str(row["symbol"]),
        "cutoff_ns": int(row["cutoff_ns"]),
        "quality_tier": str(row["quality_tier"]),
        "lower_bound_hex": float.fromhex(str(row["lower_bound_hex"])).hex(),
        "routes": [str(value) for value in row["routes"]],
        "overflow_fallback": bool(row["overflow_fallback"]),
    } for row in rows])


def _scan_digest(scan: Mapping[str, Any]) -> str:
    return _hash({
        "schema_version": scan["schema_version"],
        "contract_digest": FROZEN_PROPOSAL_CONTRACT_DIGEST,
        "generation_id": scan["generation_id"],
        "query_episode_id": scan["query_episode_id"],
        "rows_scanned": scan["rows_scanned"], "eligible_rows": scan["eligible_rows"],
        "eligible_main_rows": scan["eligible_main_rows"],
        "eligible_overflow_rows": scan["eligible_overflow_rows"],
        "route_counts": scan["route_counts"], "route_quotas": scan["route_quotas"],
        "candidate_digest": scan["candidate_digest"],
        "real_forward_outcomes_accessed": False,
    })


def _certificate_digest(certificate: Mapping[str, Any], matches: list[dict[str, Any]]) -> str:
    deterministic: dict[str, Any] = {
        "schema_version": certificate.get("schema_version"),
        "contract_digest": certificate.get("contract_digest"),
        "generation_id": certificate.get("generation_id"),
        "query_episode_id": certificate.get("query_episode_id"),
        "input_digest": certificate.get("input_digest"),
        "eligible_candidates": certificate.get("eligible_candidates"),
        "exact_evaluated": certificate.get("exact_evaluated"),
        "safely_pruned": certificate.get("safely_pruned"),
        "stopped_early": certificate.get("stopped_early"),
        "stop_threshold_hex": float(certificate.get("stop_threshold")).hex(),
        "next_lower_bound_hex": (
            float(certificate["next_lower_bound"]).hex()
            if certificate.get("next_lower_bound") is not None else None
        ),
        "maximum_quantized_bound_excess_hex": float(
            certificate.get("maximum_quantized_bound_excess")
        ).hex(),
        "rounds": certificate.get("rounds"),
        "matches": [{
            "episode_id": match.get("episode_id"),
            "total_hex": float(match.get("total_distance")).hex(),
            "components": {
                key: float(value).hex() for key, value in sorted(
                    (match.get("component_distances") or {}).items()
                )
            },
            "alignment": match.get("alignment"),
        } for match in matches],
        "real_forward_outcomes_accessed": False,
    }
    if certificate.get("native_bound_accounting") is not None:
        deterministic.update({
            "native_bound_accounting": certificate.get("native_bound_accounting"),
            "minimum_native_pruned_bound_hex": (
                float(certificate["minimum_native_pruned_bound"]).hex()
                if certificate.get("minimum_native_pruned_bound") is not None else None
            ),
            "threshold_closure_passes": certificate.get("threshold_closure_passes") or [],
        })
    return _hash(deterministic)


def _frontier_execution_valid(payload: Mapping[str, Any], contract: Mapping[str, Any]) -> bool:
    try:
        certificate = dict(payload["certificate"])
        rounds = list(certificate.get("rounds") or [])
        frontiers = [int(row.get("frontier_rows", -1)) for row in rounds]
        attempts = list(payload.get("frontier_attempts") or [])
        if not frontiers or not attempts or min(frontiers) < 1:
            return False
        primary = int(contract["controls"]["maximum_frontier_rows"])
        limits = [int(row.get("maximum_frontier_rows", -1)) for row in attempts]
        if limits != [primary] or [row.get("status") for row in attempts] != ["certified"]:
            return False
        closure = list(certificate.get("threshold_closure_passes") or [])
        if payload.get("streaming_threshold_closure_used") is not bool(closure):
            return False
        if payload.get("frontier_limit_rows") != primary or max(frontiers) > primary:
            return False
        if not all(isinstance(row.get("proposal_result_digest"), str) and row["proposal_result_digest"] for row in attempts):
            return False
        previous_upper = None
        previous_threshold = None
        previous_native = primary
        previous_exact = int(rounds[-1].get("exact_rows", -1))
        for index, row in enumerate(closure):
            lower, upper = row.get("lower_exclusive"), float(row.get("upper_inclusive"))
            native = int(row.get("cumulative_native_bound_evaluated", -1))
            exact = int(row.get("cumulative_exact_dtw_evaluated", -1))
            resulting = float(row.get("resulting_threshold"))
            minima = (row.get("minimum_packed_unclassified_bound"), row.get("minimum_native_pruned_bound"))
            if not all((
                lower == previous_upper, upper >= 0,
                index != 0 or upper == float(rounds[-1].get("constrained_threshold")) + 1e-12,
                index == 0 or upper > float(previous_upper),
                index == 0 or upper == previous_threshold + 1e-12,
                int(row.get("admitted_rows", -1)) >= 0,
                native - previous_native == int(row.get("admitted_rows", -1)),
                previous_exact <= exact <= native, int(row.get("selected_rows", -1)) == 20,
                isinstance(row.get("certified"), bool),
                index == len(closure) - 1 or row.get("certified") is False,
                isfinite(resulting) and resulting >= 0,
                not row.get("certified") or all(value is None or float(value) > resulting + 1e-12 for value in minima),
                isinstance(row.get("excluded_prefix_digest"), str),
                isinstance(row.get("admitted_set_digest"), str),
                isinstance(row.get("scan_result_digest"), str),
            )):
                return False
            previous_upper, previous_threshold = upper, resulting
            previous_native, previous_exact = native, exact
        if closure and closure[-1].get("certified") is not True:
            return False
        accounting = certificate.get("native_bound_accounting") or {}
        final_minima = [float(value) for value in (
            closure[-1].get("minimum_packed_unclassified_bound"),
            closure[-1].get("minimum_native_pruned_bound"),
        ) if value is not None] if closure else []
        expected_next = min(final_minima) if final_minima else None
        if closure and not all((
            previous_native == int(accounting.get("native_bound_evaluated", -1)),
            previous_exact == int(accounting.get("exact_dtw_evaluated", -1)),
            previous_threshold == float(certificate.get("stop_threshold")),
            expected_next == certificate.get("next_lower_bound"),
        )):
            return False
        final = attempts[-1]
        next_bound = certificate.get("next_lower_bound")
        return all((
            int(final.get("exact_evaluated", -1)) == int(certificate.get("exact_evaluated", -2)),
            final.get("stop_threshold_hex") == float(certificate.get("stop_threshold")).hex(),
            final.get("next_lower_bound_hex") == (float(next_bound).hex() if next_bound is not None else None),
        ))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _overflow_policy(primary_limit: int) -> dict[str, Any]:
    return {
        "schema_version": "m04r11-streaming-threshold-closure-v1",
        "sorted_prefix_rows": int(primary_limit),
        "trigger": "sorted prefix cannot certify the constrained top-k",
        "terminal_frontier": "strict streamed threshold closure or eligible exhaustion",
        "scope": "authority construction only",
        "semantics": (
            "retain all prefix classifications, stream inclusive packed-bound bands, "
            "apply exact native non-DTW deferral with threshold reopening, and never "
            "emit an approximate result or repeat exact DTW work; native "
            "materialization may repeat only when a raised constrained threshold "
            "reopens a deferred row"
        ),
    }


def _authority_case_gates(payload: Mapping[str, Any], case: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, bool]:
    matches, certificate = list(payload.get("matches", [])), dict(payload.get("certificate", {}))
    ids = [str(row.get("episode_id")) for row in matches]
    ordering = [(float(row["total_distance"]), str(row["episode_id"])) for row in matches]
    next_bound, stopped = certificate.get("next_lower_bound"), bool(certificate.get("stopped_early"))
    stopping = (
        stopped and next_bound is not None and float(next_bound) > float(certificate["stop_threshold"]) + 1e-12
    ) or (not stopped and int(certificate.get("safely_pruned", -1)) == 0)
    gates = {
        "twenty_matches": len(matches) == 20, "unique_episode_ids": len(ids) == len(set(ids)),
        "stable_distance_id_order": ordering == sorted(ordering),
        "quality_tiers_allowed": all(row.get("quality_tier") in {"A", "B"} for row in matches),
        "per_instrument_cap": max((sum(str(row.get("symbol")) == symbol for row in matches) for symbol in {str(row.get("symbol")) for row in matches}), default=0) <= 3,
        "same_symbol_overlap_excluded": all(str(row.get("symbol")) != str(case["symbol"]) or pd.Timestamp(row.get("cutoff")) < pd.Timestamp(payload["query_start"]) for row in matches),
        "candidate_cutoffs_temporally_eligible": all(pd.Timestamp(row.get("cutoff")) <= pd.Timestamp(payload["latest_eligible_cutoff"]) for row in matches),
        "certificate_query_equal": certificate.get("query_episode_id") == case["episode_id"],
        "candidate_accounting": int(certificate.get("exact_evaluated", -1)) + int(certificate.get("safely_pruned", -1)) == int(certificate.get("eligible_candidates", -2)),
        "strict_stop_or_exhaustion": stopping,
        "quantized_bound_safe": float(certificate.get("maximum_quantized_bound_excess", float("inf"))) <= 1e-12,
        "certificate_digest_reconstructed": certificate.get("result_digest") == _certificate_digest(certificate, matches),
    }
    gates["frontier_execution_policy"] = _frontier_execution_valid(payload, contract)
    return gates


def _without(payload: Mapping[str, Any], omitted: set[str]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in omitted}


def _validate_contract_and_prereg(
    registry: Mapping[str, Any], artifact_dir: Path, repository_root: Path,
    candidate_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[str]]:
    roots = expected_roots(artifact_dir)
    prereg_path = repository_root / PREREGISTRATION_RELATIVE_PATH
    if prereg_path.is_symlink() or prereg_path.resolve() != prereg_path.absolute():
        raise ValueError("preregistration exact path is linked or aliased")
    prereg = _read(prereg_path)
    contract = _read(candidate_root / "candidate-contract.json")
    roles = _roles(registry)
    execution = _execution(roles)
    ordered = [str(case["episode_id"]) for case in registry["cases_data"]]
    if not all((
        len(ordered) == 60,
        registry.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
        _hash(ordered) == FROZEN_CASE_ORDER_DIGEST,
        _hash(roles) == FROZEN_ROLE_TABLE_DIGEST,
        _hash(ordered[:7]) == FROZEN_EXPOSED_IDS_DIGEST,
        _hash(ordered[7:]) == FROZEN_CONFIRMATORY_IDS_DIGEST,
        _hash(execution) == FROZEN_EXECUTION_ORDER_DIGEST,
    )):
        raise ValueError("frozen registry, role or execution order differs")
    if contract != prereg.get("producer_contract"):
        raise ValueError("candidate contract differs from preregistration")
    if not all((
        contract.get("schema_version") == PRODUCER_SCHEMA,
        contract.get("contract_digest") == _hash(_without(contract, {"contract_digest"})),
        contract.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
        contract.get("ordered_query_ids") == ordered,
        contract.get("ordered_query_ids_digest") == FROZEN_CASE_ORDER_DIGEST,
        contract.get("role_table") == roles,
        contract.get("role_table_digest") == FROZEN_ROLE_TABLE_DIGEST,
        contract.get("execution_query_ids") == execution,
        contract.get("execution_query_ids_digest") == FROZEN_EXECUTION_ORDER_DIGEST,
        contract.get("generation_id") == FROZEN_GENERATION_ID,
        contract.get("proposal_contract_digest") == FROZEN_PROPOSAL_CONTRACT_DIGEST,
        contract.get("route_quotas") == FROZEN_ROUTE_QUOTAS,
        contract.get("request") == FROZEN_REQUEST,
        contract.get("roots") == roots,
        contract.get("artifact_schemas") == ARTIFACT_SCHEMAS,
        contract.get("resident_policy") == RESIDENT_POLICY,
        contract.get("resident_ready_schema") == RESIDENT_READY_SCHEMA,
        contract.get("resident_binding_schema") == RESIDENT_BINDING_SCHEMA,
        contract.get("scan_protocol") == SCAN_PROTOCOL,
        contract.get("performance_limits") == PERFORMANCE_LIMITS,
        contract.get("semantic_case_schema") == SEMANTIC_CASE_SCHEMA,
        contract.get("performance_attempt_schema") == PERFORMANCE_ATTEMPT_SCHEMA,
        contract.get("semantic_matrix_schema") == SEMANTIC_MATRIX_SCHEMA,
        contract.get("semantic_seal_schema") == SEMANTIC_SEAL_SCHEMA,
        contract.get("performance_matrix_schema") == PERFORMANCE_MATRIX_SCHEMA,
        contract.get("performance_final_schema") == PERFORMANCE_FINAL_SCHEMA,
        contract.get("run_ledger_event_schema") == LEDGER_EVENT_SCHEMA,
        contract.get("run_ledger_head_schema") == LEDGER_HEAD_SCHEMA,
        contract.get("incomplete_schema") == "candidate-resident-incomplete-v1",
        contract.get("run_complete_schema") == RUN_COMPLETE_SCHEMA,
        contract.get("case_bundle_schema") == CASE_BUNDLE_SCHEMA,
        contract.get("results_opened_schema") == RESULTS_OPENED_SCHEMA,
        contract.get("comparison_matrix_schema") == COMPARISON_MATRIX_SCHEMA,
        contract.get("comparison_seal_schema") == COMPARISON_SEAL_SCHEMA,
        contract.get("comparison_verification_schema") == VERIFICATION_SCHEMA,
        contract.get("preregistration_schema") == PREREGISTRATION_SCHEMA,
        contract.get("preregistration_relative_path")
        == PREREGISTRATION_RELATIVE_PATH,
        contract.get("comparison_policy") == COMPARISON_POLICY,
        contract.get("semantic_case_gates") == list(SEMANTIC_GATES),
        contract.get("performance_attempt_gates") == list(PERFORMANCE_GATES),
        contract.get("semantic_matrix_gates") == list(SEMANTIC_MATRIX_GATES),
        contract.get("performance_matrix_gates") == list(PERFORMANCE_MATRIX_GATES),
        contract.get("claims_policy") == CLAIMS_POLICY,
        contract.get("real_forward_outcomes_accessed") is False,
    )):
        raise ValueError("producer contract frozen identity differs")
    _validate_source_pack(contract["source_pack"], roots)
    if not _manifest_valid(contract.get("implementation_manifest"), repository_root):
        raise ValueError("producer code manifest differs from current files")
    if set(contract["implementation_manifest"]["files"]) != set(IMPLEMENTATION_FILES):
        raise ValueError("producer implementation file set differs")
    environment = contract.get("environment_manifest")
    if environment != _environment_manifest():
        raise ValueError("producer environment manifest differs")
    expected_contract = {
        "schema_version": PRODUCER_SCHEMA, "registry_digest": FROZEN_REGISTRY_DIGEST,
        "ordered_query_ids": ordered, "ordered_query_ids_digest": FROZEN_CASE_ORDER_DIGEST,
        "role_table": roles, "role_table_digest": FROZEN_ROLE_TABLE_DIGEST,
        "exposed_query_ids_digest": FROZEN_EXPOSED_IDS_DIGEST,
        "confirmatory_query_ids_digest": FROZEN_CONFIRMATORY_IDS_DIGEST,
        "execution_query_ids": execution, "execution_query_ids_digest": FROZEN_EXECUTION_ORDER_DIGEST,
        "generation_id": FROZEN_GENERATION_ID,
        "proposal_contract_digest": FROZEN_PROPOSAL_CONTRACT_DIGEST,
        "route_quotas": FROZEN_ROUTE_QUOTAS, "request": FROZEN_REQUEST,
        "roots": roots, "predecessor_failure": PREDECESSOR_FAILURE,
        "source_pack": contract["source_pack"], "resident_policy": RESIDENT_POLICY,
        "artifact_schemas": ARTIFACT_SCHEMAS, "resident_ready_schema": RESIDENT_READY_SCHEMA,
        "resident_binding_schema": RESIDENT_BINDING_SCHEMA, "scan_protocol": SCAN_PROTOCOL,
        "performance_limits": PERFORMANCE_LIMITS, "semantic_case_schema": SEMANTIC_CASE_SCHEMA,
        "performance_attempt_schema": PERFORMANCE_ATTEMPT_SCHEMA,
        "semantic_matrix_schema": SEMANTIC_MATRIX_SCHEMA, "semantic_seal_schema": SEMANTIC_SEAL_SCHEMA,
        "performance_matrix_schema": PERFORMANCE_MATRIX_SCHEMA,
        "performance_final_schema": PERFORMANCE_FINAL_SCHEMA,
        "run_ledger_event_schema": LEDGER_EVENT_SCHEMA, "run_ledger_head_schema": LEDGER_HEAD_SCHEMA,
        "incomplete_schema": "candidate-resident-incomplete-v1", "run_complete_schema": RUN_COMPLETE_SCHEMA,
        "case_bundle_schema": CASE_BUNDLE_SCHEMA, "results_opened_schema": RESULTS_OPENED_SCHEMA,
        "comparison_matrix_schema": COMPARISON_MATRIX_SCHEMA,
        "comparison_seal_schema": COMPARISON_SEAL_SCHEMA,
        "comparison_verification_schema": VERIFICATION_SCHEMA,
        "preregistration_schema": PREREGISTRATION_SCHEMA,
        "preregistration_relative_path": PREREGISTRATION_RELATIVE_PATH,
        "marker_and_resume_policy": MARKER_AND_RESUME_POLICY,
        "preregistration_policy": PREREGISTRATION_POLICY,
        "comparison_policy": COMPARISON_POLICY,
        "semantic_case_gates": list(SEMANTIC_GATES),
        "performance_attempt_gates": list(PERFORMANCE_GATES),
        "semantic_matrix_gates": list(SEMANTIC_MATRIX_GATES),
        "performance_matrix_gates": list(PERFORMANCE_MATRIX_GATES),
        "claims_policy": CLAIMS_POLICY, "implementation_manifest": contract["implementation_manifest"],
        "environment_manifest": environment, "real_forward_outcomes_accessed": False,
    }
    expected_contract["contract_digest"] = _hash(expected_contract)
    if contract != expected_contract:
        raise ValueError("producer contract differs from independently reconstructed exact contract")
    expected_prereg = {
        "schema_version": PREREGISTRATION_SCHEMA, "relative_path": PREREGISTRATION_RELATIVE_PATH,
        "resolved_path": str(prereg_path.resolve()), "producer_contract": contract,
        "producer_contract_digest": contract["contract_digest"],
        "source_pack_binding_digest": _hash(contract["source_pack"]),
        "implementation_manifest_digest": contract["implementation_manifest"]["digest"],
        "environment_manifest_digest": environment["digest"], "artifact_schemas": ARTIFACT_SCHEMAS,
        "marker_and_resume_policy": MARKER_AND_RESUME_POLICY,
        "comparison_policy": COMPARISON_POLICY, "preregistration_policy": PREREGISTRATION_POLICY,
        "authority_results_opened": False, "candidate_authority_comparison_opened": False,
        "real_forward_outcomes_accessed": False,
    }
    expected_prereg["preregistration_digest"] = _hash(expected_prereg)
    if prereg != expected_prereg:
        raise ValueError("preregistration differs from independently reconstructed exact document")
    if not all((
        prereg.get("schema_version") == PREREGISTRATION_SCHEMA,
        prereg.get("relative_path") == PREREGISTRATION_RELATIVE_PATH,
        prereg.get("resolved_path") == str(prereg_path.resolve()),
        prereg.get("producer_contract_digest") == contract["contract_digest"],
        prereg.get("source_pack_binding_digest") == _hash(contract["source_pack"]),
        prereg.get("implementation_manifest_digest")
        == contract["implementation_manifest"]["digest"],
        prereg.get("environment_manifest_digest") == environment["digest"],
        prereg.get("artifact_schemas") == ARTIFACT_SCHEMAS,
        prereg.get("comparison_policy") == COMPARISON_POLICY,
        prereg.get("authority_results_opened") is False,
        prereg.get("candidate_authority_comparison_opened") is False,
        prereg.get("real_forward_outcomes_accessed") is False,
        prereg.get("preregistration_digest")
        == _hash(_without(prereg, {"preregistration_digest"})),
    )):
        raise ValueError("preregistration identity or digest differs")
    return contract, prereg, roles, execution


def _validate_resident(
    candidate_root: Path, contract: Mapping[str, Any],
) -> dict[str, Any]:
    resident = _read(candidate_root / "RESIDENT_READY.json")
    observation = resident.get("resident_ready_observation")
    ready_payload = resident.get("resident_ready_payload")
    validation = resident.get("validation_observation")
    if set(resident) != {
        "schema_version", "producer_contract_digest", "resident_ready_path",
        "resident_ready_schema", "resident_content_digest",
        "resident_ready_observation", "resident_ready_payload",
        "resident_ready_bytes_base64", "validation_observation", "generation_id",
        "provenance_digest", "mirror_store_root", "storage_class",
        "latency_scope", "query_specific_inputs_used", "outcomes_or_labels_used",
        "real_forward_outcomes_accessed", "binding_digest",
    }:
        raise ValueError("resident binding fields differ")
    roots = contract["roots"]
    ready_path = Path(roots["resident_full_root"]) / "READY.json"
    store_root = Path(roots["resident_full_root"]) / "store"
    expected_content = _resident_content(contract["source_pack"])
    expected_content_digest = _hash(expected_content)
    try:
        ready_bytes = base64.b64decode(
            str(resident.get("resident_ready_bytes_base64", "")), validate=True,
        )
        decoded_ready = json.loads(ready_bytes)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("durable resident READY byte snapshot differs") from exc
    ready_keys = {
        "schema_version", "mode", "content", "content_digest", "seal", "seal_digest",
        "capacity_observation", "startup_timings", "created_at", "ready_digest",
    }
    seal_keys = {
        "generation_id", "provenance_digest", "pack_contract_digest",
        "quantized_bound_contract_digest", "source_store_root", "mirror_root",
        "mirror_store_root", "source_generation_st_dev", "mirror_generation_st_dev",
        "source_files", "mirror_files", "resident_mount", "mountinfo_path",
        "resident_capacity_bytes", "required_capacity_bytes", "reserve_bytes",
        "physical_generation_bytes", "storage_class", "latency_scope",
        "query_specific_inputs_used", "outcomes_or_labels_used",
        "real_forward_outcomes_accessed", "source_active_pointer_absent",
        "mirror_active_pointer_absent",
    }
    content = ready_payload.get("content") if type(ready_payload) is dict else None
    seal = ready_payload.get("seal") if type(ready_payload) is dict else None
    source_generation = Path(roots["source_full_root"]) / "store/generations" / FROZEN_GENERATION_ID
    expected_file_paths = {
        "source_files": {
            "manifest": source_generation / "manifest.json",
            "rows": source_generation / str(contract["source_pack"]["rows_file"]),
            "overflow": source_generation / str(contract["source_pack"]["overflow_file"]),
        },
        "mirror_files": {
            "manifest": store_root / "generations" / FROZEN_GENERATION_ID / "manifest.json",
            "rows": store_root / "generations" / FROZEN_GENERATION_ID / str(contract["source_pack"]["rows_file"]),
            "overflow": store_root / "generations" / FROZEN_GENERATION_ID / str(contract["source_pack"]["overflow_file"]),
        },
    }
    seal_files_valid = type(seal) is dict
    if seal_files_valid:
        for group, paths in expected_file_paths.items():
            rows = seal.get(group)
            if type(rows) is not dict or set(rows) != set(paths):
                seal_files_valid = False
                break
            for name, path in paths.items():
                row = rows[name]
                content_row = expected_content[group][name]
                if not (
                    type(row) is dict
                    and set(row) == {"path", "bytes", "sha256", "st_dev"}
                    and Path(str(row.get("path"))) == path
                    and row.get("bytes") == content_row["bytes"]
                    and row.get("sha256") == content_row["sha256"]
                    and type(row.get("st_dev")) is int
                ):
                    seal_files_valid = False
                    break
    ready_capacity = ready_payload.get("capacity_observation") if type(ready_payload) is dict else None
    ready_capacity_valid = type(ready_capacity) is dict and set(ready_capacity) == {"before", "after"}
    if ready_capacity_valid:
        ready_capacity_valid = all(
            type(row) is dict
            and set(row) == {"capacity_bytes", "available_bytes", "block_bytes"}
            and all(type(value) is int and value >= 0 for value in row.values())
            and row["available_bytes"] <= row["capacity_bytes"]
            for row in ready_capacity.values()
        )
    ready_timings = ready_payload.get("startup_timings") if type(ready_payload) is dict else None
    ready_timings_valid = type(ready_timings) is dict and set(ready_timings) == {
        "source_content_verification_seconds", "mirror_copy_seconds",
        "mirror_content_verification_seconds", "readiness_seal_seconds",
        "total_before_ready_seconds",
    } and all(type(value) in {int, float} and isfinite(value) and value >= 0 for value in ready_timings.values())
    lease = observation.get("file_identity_lease") if type(observation) is dict else None
    identity = observation.get("ready_identity") if type(observation) is dict else None
    expected_observation_keys = {
        "schema_version", "content_digest", "ready_digest", "seal_digest",
        "ready_file_sha256", "ready_identity", "file_identity_lease",
    }
    expected_lease_names = {
        "ready", "mirror", "store", "generations", "generation",
        "file_manifest", "file_rows", "file_overflow",
    }
    expected_paths = {
        "ready": ready_path, "mirror": Path(roots["resident_full_root"]),
        "store": store_root, "generations": store_root / "generations",
        "generation": store_root / "generations" / FROZEN_GENERATION_ID,
        "file_manifest": store_root / "generations" / FROZEN_GENERATION_ID / "manifest.json",
        "file_rows": store_root / "generations" / FROZEN_GENERATION_ID / str(contract["source_pack"]["rows_file"]),
        "file_overflow": store_root / "generations" / FROZEN_GENERATION_ID / str(contract["source_pack"]["overflow_file"]),
    }
    validation_keys = {
        "schema_version", "ready_digest", "content_digest", "seal_digest", "reserve_bytes",
        "capacity", "mount", "source_content_verification_seconds",
        "mirror_content_verification_seconds", "validation_seconds", "observed_at",
        "observation_digest",
    }
    identity_keys = {"path", "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode"}
    lease_deterministic = _without(lease, {"lease_digest"}) if type(lease) is dict else {}
    lease_files = lease.get("files") if type(lease) is dict else None
    identities_valid = type(lease_files) is dict and set(lease_files) == expected_lease_names
    if identities_valid:
        devices = set()
        for name, row in lease_files.items():
            if type(row) is not dict or set(row) != identity_keys:
                identities_valid = False
                break
            if Path(str(row.get("path"))) != expected_paths[name]:
                identities_valid = False
                break
            if any(type(row.get(key)) is not int for key in identity_keys - {"path"}):
                identities_valid = False
                break
            devices.add(row["st_dev"])
        identities_valid = identities_valid and len(devices) == 1
    capacity = validation.get("capacity") if type(validation) is dict else None
    mount = validation.get("mount") if type(validation) is dict else None
    capacity_valid = type(capacity) is dict and set(capacity) == {
        "capacity_bytes", "available_bytes", "block_bytes",
    } and all(type(value) is int and value >= 0 for value in capacity.values())
    mount_keys = {
        "mount_id", "parent_mount_id", "major_minor", "mount_root", "mount_point",
        "mount_options", "optional_fields", "fs_type", "mount_source", "super_options", "st_dev",
    }
    mount_valid = type(mount) is dict and set(mount) == mount_keys and mount.get("fs_type") == "tmpfs"
    seal_mount = seal.get("resident_mount") if type(seal) is dict else None
    seal_mount_valid = (
        type(seal_mount) is dict and set(seal_mount) == mount_keys
        and seal_mount.get("fs_type") == "tmpfs" and seal_mount == mount
        and seal.get("mirror_generation_st_dev") == seal_mount.get("st_dev")
        and all(row.get("st_dev") == seal_mount.get("st_dev") for row in seal.get("mirror_files", {}).values())
        and all(row.get("st_dev") == seal.get("source_generation_st_dev") for row in seal.get("source_files", {}).values())
    )
    validation_times_valid = type(validation) is dict and all(
        type(validation.get(key)) in {int, float} and isfinite(validation[key]) and validation[key] >= 0
        for key in ("source_content_verification_seconds", "mirror_content_verification_seconds", "validation_seconds")
    )
    if not all((
        resident.get("schema_version") == RESIDENT_BINDING_SCHEMA,
        resident.get("producer_contract_digest") == contract["contract_digest"],
        resident.get("resident_ready_schema") == RESIDENT_READY_SCHEMA,
        resident.get("generation_id") == FROZEN_GENERATION_ID,
        resident.get("provenance_digest") == contract["source_pack"]["provenance_digest"],
        resident.get("resident_content_digest") == expected_content_digest,
        type(ready_payload) is dict and set(ready_payload) == ready_keys,
        decoded_ready == ready_payload,
        ready_payload.get("schema_version") == RESIDENT_READY_SCHEMA,
        ready_payload.get("mode") in {"copy-and-validate", "validate-existing"},
        ready_capacity_valid, ready_timings_valid,
        _timestamp(ready_payload.get("created_at"), "resident READY snapshot") is not None,
        content == expected_content,
        ready_payload.get("content_digest") == _hash(content),
        type(seal) is dict and set(seal) == seal_keys,
        seal_files_valid,
        ready_payload.get("seal_digest") == _hash(seal),
        ready_payload.get("ready_digest") == _hash(_without(ready_payload, {"ready_digest"})),
        sha256(ready_bytes).hexdigest() == observation.get("ready_file_sha256"),
        ready_payload.get("ready_digest") == observation.get("ready_digest"),
        ready_payload.get("seal_digest") == observation.get("seal_digest"),
        ready_payload.get("content_digest") == expected_content_digest,
        seal.get("generation_id") == FROZEN_GENERATION_ID,
        seal.get("provenance_digest") == contract["source_pack"]["provenance_digest"],
        seal.get("pack_contract_digest") == expected_content["pack_contract_digest"],
        seal.get("quantized_bound_contract_digest") == expected_content["quantized_bound_contract_digest"],
        seal.get("source_store_root") == str((Path(roots["source_full_root"]) / "store").resolve()),
        seal.get("mirror_root") == roots["resident_full_root"],
        seal.get("mirror_store_root") == str(store_root),
        seal.get("reserve_bytes") == RESIDENT_POLICY["reserve_bytes"],
        seal.get("physical_generation_bytes") == expected_content["physical_generation_bytes"],
        seal.get("required_capacity_bytes") == seal.get("physical_generation_bytes") + seal.get("reserve_bytes"),
        type(seal.get("resident_capacity_bytes")) is int
        and seal.get("resident_capacity_bytes") >= seal.get("required_capacity_bytes"),
        seal.get("storage_class") == "tmpfs-backed-generation-v1",
        seal.get("latency_scope") == RESIDENT_LATENCY_SCOPE,
        seal.get("source_active_pointer_absent") is True,
        seal.get("mirror_active_pointer_absent") is True,
        all(seal.get(key) is False for key in (
            "query_specific_inputs_used", "outcomes_or_labels_used", "real_forward_outcomes_accessed",
        )),
        Path(str(resident.get("resident_ready_path"))) == ready_path,
        Path(str(resident.get("mirror_store_root"))) == store_root,
        resident.get("storage_class") == "tmpfs-backed-generation-v1",
        resident.get("latency_scope") == RESIDENT_LATENCY_SCOPE,
        type(observation) is dict and set(observation) == expected_observation_keys,
        observation.get("schema_version") == RESIDENT_READY_SCHEMA,
        observation.get("content_digest") == resident.get("resident_content_digest"),
        all(_is_digest(observation.get(key)) for key in ("ready_digest", "seal_digest", "ready_file_sha256")),
        type(identity) is dict and type(lease_files) is dict,
        identity == lease_files.get("ready"), identities_valid,
        set(lease) == {"schema_version", "ready_digest", "ready_file_sha256", "content_digest", "files", "lease_digest"},
        lease.get("schema_version") == FILE_IDENTITY_LEASE_SCHEMA,
        lease.get("ready_digest") == observation.get("ready_digest"),
        lease.get("ready_file_sha256") == observation.get("ready_file_sha256"),
        lease.get("content_digest") == expected_content_digest,
        lease.get("lease_digest") == _hash(lease_deterministic),
        type(validation) is dict and set(validation) == validation_keys,
        validation.get("schema_version") == VALIDATION_OBSERVATION_SCHEMA,
        validation.get("ready_digest") == observation.get("ready_digest"),
        validation.get("content_digest") == resident.get("resident_content_digest"),
        validation.get("seal_digest") == observation.get("seal_digest"),
        validation.get("reserve_bytes")
        == contract.get("resident_policy", {}).get("reserve_bytes"),
        capacity_valid, mount_valid, seal_mount_valid, validation_times_valid,
        _timestamp(validation.get("observed_at"), "resident validation") is not None,
        validation.get("observation_digest")
        == _hash(_without(validation, {"observation_digest"})),
        resident.get("query_specific_inputs_used") is False,
        resident.get("outcomes_or_labels_used") is False,
        resident.get("real_forward_outcomes_accessed") is False,
        resident.get("binding_digest")
        == _hash(_without(resident, {"binding_digest"})),
    )):
        raise ValueError("resident binding differs")
    return resident


def _validate_bundle(
    bundle: Mapping[str, Any], case: Mapping[str, Any], role: Mapping[str, Any],
    contract: Mapping[str, Any], resident: Mapping[str, Any], ordinal: int,
    expected_query: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    query_id = str(case["episode_id"])
    if set(bundle) != {
        "schema_version", "producer_contract_digest", "execution_ordinal",
        "query_episode_id", "semantic", "performance", "bundle_digest",
    }:
        raise ValueError("case bundle fields differ")
    if not all((
        bundle.get("schema_version") == CASE_BUNDLE_SCHEMA,
        bundle.get("producer_contract_digest") == contract["contract_digest"],
        bundle.get("execution_ordinal") == ordinal,
        bundle.get("query_episode_id") == query_id,
        bundle.get("bundle_digest") == _hash(_without(bundle, {"bundle_digest"})),
    )):
        raise ValueError("case bundle identity or digest differs")
    semantic, performance = dict(bundle["semantic"]), dict(bundle["performance"])
    if set(semantic) != {
        "schema_version", "producer_contract_digest", "registry_digest",
        "generation_id", "proposal_contract_digest", "resident_content_digest",
        "registry_case_id", "query_episode_id", "query_symbol", "query_start_ns",
        "latest_eligible_ns", "query_stock_prefix", "query_benchmark_prefix",
        "query_representation_digest", "performance_role", "recall_role",
        "scan_semantics", "candidates", "candidate_digest_reconstructed",
        "violations", "gates", "passed", "real_forward_outcomes_accessed",
        "created_at", "semantic_digest",
    }:
        raise ValueError("semantic case fields differ")
    if not all((
        semantic.get("schema_version") == SEMANTIC_CASE_SCHEMA,
        semantic.get("producer_contract_digest") == contract["contract_digest"],
        semantic.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
        semantic.get("generation_id") == FROZEN_GENERATION_ID,
        semantic.get("proposal_contract_digest") == FROZEN_PROPOSAL_CONTRACT_DIGEST,
        semantic.get("resident_content_digest") == resident["resident_content_digest"],
        semantic.get("registry_case_id") == case["case_id"],
        semantic.get("query_episode_id") == query_id,
        all(semantic.get(key) == value for key, value in expected_query.items()),
        semantic.get("performance_role") == role["performance_role"],
        semantic.get("recall_role") == "blind_primary",
        _timestamp(semantic.get("created_at"), "semantic case") is not None,
        semantic.get("real_forward_outcomes_accessed") is False,
        semantic.get("semantic_digest")
        == _hash(_without(semantic, {"created_at", "semantic_digest"})),
    )):
        raise ValueError("semantic case identity or digest differs")
    candidates = list(semantic.get("candidates", []))
    ids, ordered = [], []
    for row in candidates:
        if set(row) != {
            "episode_id", "symbol", "cutoff_ns", "quality_tier",
            "lower_bound_hex", "routes", "overflow_fallback",
        }:
            raise ValueError("candidate fields differ")
        episode_id = str(row["episode_id"])
        bound = float.fromhex(str(row["lower_bound_hex"]))
        if not all((
            len(episode_id) == 24, episode_id == episode_id.lower(),
            len(bytes.fromhex(episode_id)) == 12, isfinite(bound), bound >= 0,
            type(row.get("cutoff_ns")) is int, row.get("quality_tier") in {"A", "B"},
            type(row.get("overflow_fallback")) is bool,
            row.get("routes") == sorted(set(row.get("routes", []))),
            bool(row.get("routes")), set(row["routes"]).issubset(FROZEN_ROUTE_QUOTAS),
        )):
            raise ValueError("candidate metadata differs")
        ids.append(episode_id)
        ordered.append((bound, episode_id))
    candidate_digest = _candidate_digest(candidates)
    scans = list(semantic.get("scan_semantics", []))
    if len(scans) != 3 or scans[0] != scans[1] or scans[1] != scans[2]:
        raise ValueError("three scan semantics differ")
    route_counts = {
        route: sum(route in row["routes"] for row in candidates)
        for route in FROZEN_ROUTE_QUOTAS
    }
    for scan in scans:
        if set(scan) != {
            "schema_version", "generation_id", "query_episode_id", "rows_scanned",
            "eligible_rows", "eligible_main_rows", "eligible_overflow_rows",
            "route_counts", "route_quotas", "candidate_count",
            "candidate_digest", "result_digest",
        }:
            raise ValueError("scan semantic fields differ")
        if not all((
            scan.get("schema_version") == SEARCH_SCHEMA,
            scan.get("generation_id") == FROZEN_GENERATION_ID,
            scan.get("query_episode_id") == query_id,
            scan.get("rows_scanned") == contract["source_pack"]["physical_rows"],
            scan.get("eligible_rows")
            == scan.get("eligible_main_rows") + scan.get("eligible_overflow_rows"),
            scan.get("route_quotas") == FROZEN_ROUTE_QUOTAS,
            scan.get("route_counts") == route_counts,
            scan.get("candidate_count") == len(candidates),
            scan.get("candidate_digest") == candidate_digest,
            scan.get("result_digest") == _scan_digest(scan),
        )):
            raise ValueError("scan semantics differ")
    violations = {
        "duplicates": len(ids) - len(set(ids)),
        "future": sum(row["cutoff_ns"] > semantic["latest_eligible_ns"] for row in candidates),
        "same_symbol_overlap": sum(
            row["symbol"] == semantic["query_symbol"]
            and row["cutoff_ns"] >= semantic["query_start_ns"] for row in candidates
        ),
        "tier": sum(row["quality_tier"] not in {"A", "B"} for row in candidates),
    }
    expected_semantic_gates = {
        "resident_content_matches_contract": True,
        "query_identity_prefix_and_representation_match": bool(
            all(semantic.get(key) == value for key, value in expected_query.items())
        ),
        "three_scan_digest_and_block_order_invariance": True,
        "internal_eligible_row_accounting": True,
        "physical_row_accounting": True, "frozen_route_quotas": True,
        "candidate_digest_order_and_routes_reconstruct": (
            candidate_digest == semantic.get("candidate_digest_reconstructed")
            and ordered == sorted(ordered) and route_counts == scans[0]["route_counts"]
        ),
        "zero_temporal_overlap_tier_duplicate_errors": not any(violations.values()),
        "real_forward_outcomes_excluded": True,
    }
    if not all((
        tuple(expected_semantic_gates) == SEMANTIC_GATES,
        semantic.get("violations") == violations,
        semantic.get("gates") == expected_semantic_gates,
        semantic.get("passed") is True,
        all(expected_semantic_gates.values()),
    )):
        raise ValueError("semantic gates differ")
    timings = performance.get("timings", {})
    if set(performance) != {
        "schema_version", "producer_contract_digest", "registry_digest",
        "generation_id", "resident_binding_digest", "semantic_digest",
        "registry_case_id", "query_episode_id", "performance_role",
        "attempt_ordinal", "ready_start", "ready_end", "timings", "gates",
        "passed", "real_forward_outcomes_accessed", "created_at",
        "performance_digest",
    } or set(timings) != {
        "resident_first_seconds", "resident_reverse_seconds",
        "resident_repeat_seconds", "task_seconds", "peak_rss_mb",
    }:
        raise ValueError("performance attempt fields differ")
    numeric = [float(value) for value in timings.values()]
    finite = len(numeric) == 5 and all(isfinite(value) and value >= 0 for value in numeric)
    expected_performance_gates = {
        "same_ready_instance_at_start_and_end": (
            performance.get("ready_start") == resident["resident_ready_observation"]
            and performance.get("ready_end") == resident["resident_ready_observation"]
        ),
        "measurements_finite_nonnegative_and_task_contains_scans": (
            finite and timings["task_seconds"] >= sum(
                timings[name] for name in (
                    "resident_first_seconds", "resident_reverse_seconds",
                    "resident_repeat_seconds",
                )
            )
        ),
        "resident_first_scan_at_most_120_seconds": (
            finite and timings["resident_first_seconds"] <= 120.0
        ),
        "resident_repeat_scan_at_most_60_seconds": (
            finite and timings["resident_repeat_seconds"] <= 60.0
        ),
        "worker_rss_at_most_1536_mib": finite and timings["peak_rss_mb"] <= 1_536.0,
        "primary_attempt_completed": performance.get("attempt_ordinal") == 1,
    }
    if not all((
        tuple(expected_performance_gates) == PERFORMANCE_GATES,
        performance.get("schema_version") == PERFORMANCE_ATTEMPT_SCHEMA,
        performance.get("producer_contract_digest") == contract["contract_digest"],
        performance.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
        performance.get("generation_id") == FROZEN_GENERATION_ID,
        performance.get("resident_binding_digest") == resident["binding_digest"],
        performance.get("semantic_digest") == semantic["semantic_digest"],
        performance.get("registry_case_id") == case["case_id"],
        performance.get("query_episode_id") == query_id,
        performance.get("performance_role") == role["performance_role"],
        performance.get("attempt_ordinal") == 1,
        performance.get("gates") == expected_performance_gates,
        performance.get("passed") is all(expected_performance_gates.values()),
        performance.get("real_forward_outcomes_accessed") is False,
        _timestamp(performance.get("created_at"), "performance attempt") is not None,
        performance.get("performance_digest")
        == _hash(_without(performance, {"created_at", "performance_digest"})),
    )):
        raise ValueError("performance attempt differs")
    return semantic, performance


def _load_ledger(root: Path, contract_digest: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = sorted((root / "ledger" / "events").glob("*.json"))
    events, previous = [], LEDGER_GENESIS
    for index, path in enumerate(paths):
        event = _read(path)
        if not all((
            set(event) == {"schema_version", "producer_contract_digest", "event_index", "previous_event_digest", "event_type", "details", "created_at", "event_digest"},
            path.name == f"{index:06d}.json",
            event.get("schema_version") == LEDGER_EVENT_SCHEMA,
            event.get("producer_contract_digest") == contract_digest,
            event.get("event_index") == index,
            event.get("previous_event_digest") == previous,
            _timestamp(event.get("created_at"), "ledger event") is not None,
            event.get("event_digest") == _hash(_without(event, {"event_digest"})),
        )):
            raise ValueError("ledger chain differs")
        events.append(event)
        previous = event["event_digest"]
    head = _read(root / "ledger" / "HEAD.json")
    if not all((
        set(head) == {"schema_version", "producer_contract_digest", "event_count", "last_event_digest", "head_digest"},
        head.get("schema_version") == LEDGER_HEAD_SCHEMA,
        head.get("producer_contract_digest") == contract_digest,
        head.get("event_count") == len(events),
        head.get("last_event_digest") == previous,
        head.get("head_digest") == _hash(_without(head, {"head_digest"})),
    )):
        raise ValueError("ledger head differs")
    return events, head


def _validate_candidate_aggregate(
    registry: Mapping[str, Any], candidate_root: Path, contract: Mapping[str, Any],
    prereg: Mapping[str, Any], roles: list[dict[str, Any]], execution: list[str],
    resident: Mapping[str, Any], expected_queries: Mapping[str, Mapping[str, Any]],
    repository_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    by_id = {str(case["episode_id"]): case for case in registry["cases_data"]}
    role_by_id = {row["query_episode_id"]: row for row in roles}
    expected_paths = [
        candidate_root / "case-bundles" / f"{ordinal:03d}-{query_id}.json"
        for ordinal, query_id in enumerate(execution)
    ]
    if sorted((candidate_root / "case-bundles").glob("*.json")) != sorted(expected_paths):
        raise ValueError("exact case-bundle path set differs")
    semantics, performance, bundles = {}, {}, []
    for ordinal, (query_id, path) in enumerate(zip(execution, expected_paths, strict=True)):
        bundle = _read(path)
        semantic, attempt = _validate_bundle(
            bundle, by_id[query_id], role_by_id[query_id], contract, resident, ordinal,
            expected_queries[query_id],
        )
        bundles.append(bundle)
        semantics[query_id], performance[query_id] = semantic, attempt
    events, head = _load_ledger(candidate_root, contract["contract_digest"])
    expected_types = ["run_started"] + [
        value for _ in execution for value in ("case_started", "case_completed")
    ] + ["run_complete"]
    if [event.get("event_type") for event in events] != expected_types:
        raise ValueError("ledger event sequence differs")
    prereg_path = repository_root / PREREGISTRATION_RELATIVE_PATH
    run_details = events[0].get("details")
    raw_git_binding = run_details.get("git_binding") if type(run_details) is dict else None
    git_binding = raw_git_binding if type(raw_git_binding) is dict else {}
    git_deterministic = _without(git_binding, {"binding_digest"})
    head_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    tracked_clean = not subprocess.run(
        ["git", "diff", "--quiet"], cwd=repository_root,
    ).returncode
    index_clean = not subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=repository_root,
    ).returncode
    expected_run = {
        "preregistration_digest": prereg["preregistration_digest"],
        "git_binding": git_binding,
        "resident_binding_digest": resident["binding_digest"],
        "resident_content_digest": resident["resident_content_digest"],
        "execution_query_ids_digest": contract["execution_query_ids_digest"],
    }
    if not all((
        run_details == expected_run, type(raw_git_binding) is dict,
        set(git_binding) == {"repository_root", "head_commit", "preregistration_relative_path", "preregistration_blob_sha256", "tracked_worktree_clean", "index_clean", "binding_digest"},
        git_binding.get("repository_root") == str(repository_root),
        git_binding.get("head_commit") == head_commit,
        git_binding.get("preregistration_relative_path") == PREREGISTRATION_RELATIVE_PATH,
        git_binding.get("preregistration_blob_sha256") == _file_hash(prereg_path),
        git_binding.get("tracked_worktree_clean") is True and tracked_clean,
        git_binding.get("index_clean") is True and index_clean,
        git_binding.get("binding_digest") == _hash(git_deterministic),
    )):
        raise ValueError("ledger run-start binding differs")
    for ordinal, query_id in enumerate(execution):
        case, role = by_id[query_id], role_by_id[query_id]
        expected_started = {
            "execution_ordinal": ordinal, "registry_case_id": case["case_id"],
            "query_episode_id": query_id, "performance_role": role["performance_role"],
            "implementation_manifest_digest": contract["implementation_manifest"]["digest"],
            "resident_ready_digest": resident["resident_ready_observation"]["ready_digest"],
            "resident_content_digest": resident["resident_content_digest"],
        }
        if events[1 + ordinal * 2].get("details") != expected_started:
            raise ValueError("ledger case-start binding differs")
    completed = [event for event in events if event["event_type"] == "case_completed"]
    for ordinal, (event, bundle) in enumerate(zip(completed, bundles, strict=True)):
        semantic, attempt = bundle["semantic"], bundle["performance"]
        expected = {
            "execution_ordinal": ordinal,
            "registry_case_id": semantic["registry_case_id"],
            "query_episode_id": semantic["query_episode_id"],
            "performance_role": semantic["performance_role"],
            "bundle_digest": bundle["bundle_digest"],
            "semantic_digest": semantic["semantic_digest"],
            "performance_digest": attempt["performance_digest"],
            "semantic_passed": True, "performance_passed": attempt["passed"],
        }
        if event.get("details") != expected:
            raise ValueError("ledger completion differs from bundle")
    ordered = [str(case["episode_id"]) for case in registry["cases_data"]]
    semantic_matrix = _read(candidate_root / "semantic-matrix.json")
    if set(semantic_matrix) != {
        "schema_version", "producer_contract_digest",
        "resident_start_content_digest", "resident_end_content_digest",
        "ordered_query_episode_ids", "semantic_case_digests", "gates", "passed",
        "real_forward_outcomes_accessed", "elapsed_seconds", "created_at",
        "result_digest",
    }:
        raise ValueError("semantic matrix fields differ")
    semantic_gates = {
        "all_60_semantic_cases_present_in_registry_order": True,
        "all_semantic_case_gates_passed": all(semantics[value]["passed"] for value in ordered),
        "mirror_content_unchanged_before_and_after": (
            semantic_matrix.get("resident_start_content_digest")
            == semantic_matrix.get("resident_end_content_digest")
            == resident["resident_content_digest"]
        ),
        "ledger_semantic_completion_matches_cases": (
            [event["details"]["semantic_digest"] for event in completed]
            == [semantics[value]["semantic_digest"] for value in execution]
        ),
        "real_forward_outcomes_excluded": True,
    }
    if not all((
        tuple(semantic_gates) == SEMANTIC_MATRIX_GATES,
        semantic_matrix.get("schema_version") == SEMANTIC_MATRIX_SCHEMA,
        semantic_matrix.get("producer_contract_digest") == contract["contract_digest"],
        semantic_matrix.get("ordered_query_episode_ids") == ordered,
        semantic_matrix.get("semantic_case_digests")
        == [semantics[value]["semantic_digest"] for value in ordered],
        semantic_matrix.get("gates") == semantic_gates,
        semantic_matrix.get("passed") is True,
        semantic_matrix.get("real_forward_outcomes_accessed") is False,
        type(semantic_matrix.get("elapsed_seconds")) in {int, float}
        and isfinite(semantic_matrix["elapsed_seconds"]) and semantic_matrix["elapsed_seconds"] >= 0,
        _timestamp(semantic_matrix.get("created_at"), "semantic matrix") is not None,
        semantic_matrix.get("result_digest") == _hash(_without(
            semantic_matrix, {"created_at", "elapsed_seconds", "result_digest"},
        )),
    )):
        raise ValueError("semantic matrix differs")
    semantic_seal = _read(candidate_root / "SEMANTIC_SEALED.json")
    if set(semantic_seal) != {
        "schema_version", "producer_contract_digest", "semantic_matrix_digest",
        "resident_start_content_digest", "resident_end_content_digest",
        "semantic_cases", "semantic_recall_ready", "performance_independent",
        "authority_results_opened", "production_promotion_authorized",
        "created_at", "seal_digest",
    }:
        raise ValueError("semantic seal fields differ")
    if not all((
        semantic_seal.get("schema_version") == SEMANTIC_SEAL_SCHEMA,
        semantic_seal.get("producer_contract_digest") == contract["contract_digest"],
        semantic_seal.get("semantic_matrix_digest") == semantic_matrix["result_digest"],
        semantic_seal.get("resident_start_content_digest")
        == resident["resident_content_digest"],
        semantic_seal.get("resident_end_content_digest")
        == resident["resident_content_digest"],
        semantic_seal.get("semantic_cases") == 60,
        semantic_seal.get("semantic_recall_ready") is True,
        semantic_seal.get("performance_independent") is True,
        semantic_seal.get("authority_results_opened") is False,
        semantic_seal.get("production_promotion_authorized") is False,
        _timestamp(semantic_seal.get("created_at"), "semantic seal") is not None,
        semantic_seal.get("seal_digest")
        == _hash(_without(semantic_seal, {"created_at", "seal_digest"})),
    )):
        raise ValueError("semantic seal differs")
    attempts = [performance[value] for value in ordered]
    confirmatory = [row for row in attempts if row["performance_role"] == "confirmatory_untouched"]
    exposed = [row for row in attempts if row["performance_role"] == "exposed_recovery_regression"]
    ready_pairs = {(_hash(row["ready_start"]), _hash(row["ready_end"])) for row in attempts}
    performance_gates = {
        "exact_7_exposed_53_confirmatory_role_partition": (
            len(exposed) == 7 and len(confirmatory) == 53
        ),
        "all_60_primary_attempts_accounted": len(attempts) == 60,
        "all_53_confirmatory_primary_attempts_passed": all(row["passed"] for row in confirmatory),
        "all_7_exposed_regression_attempts_passed": all(row["passed"] for row in exposed),
        "all_60_operational_limits_passed": all(row["passed"] for row in attempts),
        "single_ready_instance_for_all_primary_attempts": (
            len(ready_pairs) == 1 and next(iter(ready_pairs))[0] == next(iter(ready_pairs))[1]
        ),
    }
    performance_matrix = _read(candidate_root / "performance-matrix.json")
    if set(performance_matrix) != {
        "schema_version", "producer_contract_digest", "ordered_query_episode_ids",
        "attempt_digests", "confirmatory_query_episode_ids",
        "exposed_query_episode_ids", "gates", "passed", "claims_policy",
        "real_forward_outcomes_accessed", "elapsed_seconds", "created_at",
        "result_digest",
    }:
        raise ValueError("performance matrix fields differ")
    if not all((
        tuple(performance_gates) == PERFORMANCE_MATRIX_GATES,
        performance_matrix.get("schema_version") == PERFORMANCE_MATRIX_SCHEMA,
        performance_matrix.get("producer_contract_digest") == contract["contract_digest"],
        performance_matrix.get("ordered_query_episode_ids") == ordered,
        performance_matrix.get("attempt_digests")
        == [performance[value]["performance_digest"] for value in ordered],
        performance_matrix.get("confirmatory_query_episode_ids")
        == [row["query_episode_id"] for row in confirmatory],
        performance_matrix.get("exposed_query_episode_ids")
        == [row["query_episode_id"] for row in exposed],
        performance_matrix.get("gates") == performance_gates,
        performance_matrix.get("passed") is all(performance_gates.values()),
        performance_matrix.get("claims_policy") == CLAIMS_POLICY,
        performance_matrix.get("real_forward_outcomes_accessed") is False,
        type(performance_matrix.get("elapsed_seconds")) in {int, float}
        and isfinite(performance_matrix["elapsed_seconds"]) and performance_matrix["elapsed_seconds"] >= 0,
        _timestamp(performance_matrix.get("created_at"), "performance matrix") is not None,
        performance_matrix.get("result_digest") == _hash(_without(
            performance_matrix, {"created_at", "elapsed_seconds", "result_digest"},
        )),
    )):
        raise ValueError("performance matrix differs")
    performance_final = _read(candidate_root / "PERFORMANCE_FINAL.json")
    if set(performance_final) != {
        "schema_version", "producer_contract_digest", "performance_matrix_digest",
        "resident_binding_digest", "performance_terminal", "performance_passed",
        "confirmatory_performance_cases", "exposed_regression_cases",
        "claims_policy", "authority_results_opened",
        "production_promotion_authorized", "created_at", "final_digest",
    }:
        raise ValueError("performance final fields differ")
    if not all((
        performance_final.get("schema_version") == PERFORMANCE_FINAL_SCHEMA,
        performance_final.get("producer_contract_digest") == contract["contract_digest"],
        performance_final.get("performance_matrix_digest") == performance_matrix["result_digest"],
        performance_final.get("resident_binding_digest") == resident["binding_digest"],
        performance_final.get("performance_terminal") is True,
        type(performance_final.get("performance_passed")) is bool,
        performance_final.get("performance_passed") is performance_matrix["passed"],
        performance_final.get("confirmatory_performance_cases") == 53,
        performance_final.get("exposed_regression_cases") == 7,
        performance_final.get("claims_policy") == CLAIMS_POLICY,
        performance_final.get("authority_results_opened") is False,
        performance_final.get("production_promotion_authorized") is False,
        _timestamp(performance_final.get("created_at"), "performance final") is not None,
        performance_final.get("final_digest")
        == _hash(_without(performance_final, {"created_at", "final_digest"})),
    )):
        raise ValueError("terminal performance evidence differs")
    final_event = events[-1]
    run_complete = _read(candidate_root / "RUN_COMPLETE.json")
    if set(run_complete) != {
        "schema_version", "producer_contract_digest", "resident_binding_digest",
        "semantic_seal_digest", "performance_final_digest",
        "ledger_last_event_digest", "ledger_head_digest", "semantic_passed",
        "performance_passed", "authority_results_opened",
        "production_promotion_authorized", "created_at", "complete_digest",
    }:
        raise ValueError("run-complete fields differ")
    expected_final_details = {
        "semantic_matrix_digest": semantic_matrix["result_digest"],
        "semantic_seal_digest": semantic_seal["seal_digest"],
        "performance_matrix_digest": performance_matrix["result_digest"],
        "performance_final_digest": performance_final["final_digest"],
        "performance_passed": performance_matrix["passed"],
    }
    if not all((
        final_event.get("details") == expected_final_details,
        run_complete.get("schema_version") == RUN_COMPLETE_SCHEMA,
        run_complete.get("producer_contract_digest") == contract["contract_digest"],
        run_complete.get("resident_binding_digest") == resident["binding_digest"],
        run_complete.get("semantic_seal_digest") == semantic_seal["seal_digest"],
        run_complete.get("performance_final_digest") == performance_final["final_digest"],
        run_complete.get("ledger_last_event_digest") == final_event["event_digest"],
        run_complete.get("ledger_head_digest") == head["head_digest"],
        run_complete.get("semantic_passed") is True,
        run_complete.get("performance_passed") is performance_matrix["passed"],
        run_complete.get("authority_results_opened") is False,
        run_complete.get("production_promotion_authorized") is False,
        _timestamp(run_complete.get("created_at"), "run complete") is not None,
        run_complete.get("complete_digest")
        == _hash(_without(run_complete, {"created_at", "complete_digest"})),
    )):
        raise ValueError("run-complete evidence differs")
    expected_files = {
        "candidate-contract.json", "RESIDENT_READY.json", "ledger/HEAD.json",
        "semantic-matrix.json", "SEMANTIC_SEALED.json", "performance-matrix.json",
        "PERFORMANCE_FINAL.json", "RUN_COMPLETE.json",
        *(str(path.relative_to(candidate_root)) for path in expected_paths),
        *(f"ledger/events/{index:06d}.json" for index in range(122)),
    }
    observed_files: set[str] = set()
    for path in candidate_root.rglob("*"):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise ValueError("candidate tree contains linked or special entry")
        if path.is_file():
            observed_files.add(str(path.relative_to(candidate_root)))
    if observed_files != expected_files or (candidate_root / "INCOMPLETE.json").exists():
        raise ValueError("candidate terminal artifact tree differs")
    return semantics, semantic_seal, performance_final, run_complete


def _validate_authorities(
    registry: Mapping[str, Any], authority_root: Path, repository_root: Path,
    expected_queries: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    contract = _read(authority_root / "authority-contract.json")
    matrix = _read(authority_root / "authority-matrix.json")
    seal = _read(authority_root / "SEALED.json")
    ordered = [str(case["episode_id"]) for case in registry["cases_data"]]
    contract_keys = {
        "schema_version", "registry_digest", "generation_id", "full_build_evidence_digest",
        "search_contract", "certified_execution_contract", "branch_bound_evidence",
        "primary_proposal_contract", "threshold_scan_contract", "controls",
        "frontier_overflow_policy", "execution_processes", "expected_query_episode_ids",
        "selection_order", "authority_root_policy", "runner_sha256", "implementation_manifest",
        "real_forward_outcomes_accessed", "contract_digest",
    }
    authority_builder = repository_root / "experiments/m04r/m04r11_build_authorities.py"
    expected_authority_files = {
        "experiments/m04r/m04r11_build_authorities.py",
        *(str(path.relative_to(repository_root)) for path in (repository_root / "src").rglob("*.py")),
    }
    matrix_keys = {
        "schema_version", "contract_digest", "registry_digest", "generation_id",
        "branch_bound_evidence_digest", "cases", "measurements", "completed_cases",
        "invalid_cases", "p95_exact_seconds", "maximum_exact_seconds", "total_exact_seconds",
        "p95_search_seconds", "maximum_search_seconds", "total_search_seconds",
        "maximum_worker_rss_mb", "streaming_threshold_closure_cases", "worker_failures",
        "elapsed_seconds", "gates", "candidate_results_opened", "real_forward_outcomes_accessed",
        "created_at", "gate_passed", "performance_gates", "performance_gate_passed",
        "measurement_integrity_digest", "result_digest",
    }
    seal_keys = {
        "schema_version", "contract_digest", "registry_digest", "branch_bound_evidence_digest",
        "authority_matrix_digest", "measurement_integrity_digest", "authority_cases", "seal_scope",
        "authority_correctness_sealed", "performance_gate_passed",
        "production_promotion_authorized", "candidate_results_opened",
        "real_forward_outcomes_accessed", "seal_digest",
    }
    measurements = list(matrix.get("measurements", []))
    exact_seconds = [float(row["exact_seconds"]) for row in measurements]
    search_seconds = [float(row["search_seconds"]) for row in measurements]
    matrix_gates = {
        "all_60_authorities_complete": matrix.get("completed_cases") == 60 and matrix.get("invalid_cases") == [],
        "all_case_certificates_pass": True,
        "no_worker_failures": matrix.get("worker_failures") == [],
        "registry_order_exact": [row.get("query_episode_id") for row in matrix.get("cases", [])] == ordered,
        "real_forward_outcomes_excluded": True,
    }
    performance_gates = {
        "certified_p95_search_at_most_300_seconds": matrix.get("p95_search_seconds") is not None and float(matrix["p95_search_seconds"]) <= 300,
        "certified_maximum_search_at_most_600_seconds": matrix.get("maximum_search_seconds") is not None and float(matrix["maximum_search_seconds"]) <= 600,
        "certified_peak_rss_at_most_1536_mb": float(matrix.get("maximum_worker_rss_mb", float("inf"))) <= 1536,
    }
    if not all((
        set(contract) == contract_keys,
        contract.get("schema_version") == AUTHORITY_CONTRACT_SCHEMA,
        contract.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
        contract.get("generation_id") == FROZEN_GENERATION_ID,
        contract.get("expected_query_episode_ids") == ordered,
        contract.get("selection_order") == [case["case_id"] for case in registry["cases_data"]],
        contract.get("search_contract") == registry.get("search_contract"),
        contract.get("controls") == registry.get("search_contract", {}).get("controls"),
        contract.get("execution_processes") == contract.get("controls", {}).get("processes"),
        contract.get("certified_execution_contract") == certified_packed_search_contract(
            requested_positions=True, vector_lower_bounds=True, deferred_alignments=True,
            compact_scored=True, native_bound_deferral=True,
            streaming_threshold_closure=True, branch_aware_packed_bounds=True,
        ),
        contract.get("primary_proposal_contract") == packed_bound_search_contract(branch_aware=True),
        contract.get("threshold_scan_contract") == packed_bound_threshold_scan_contract(branch_aware=True),
        contract.get("frontier_overflow_policy") == _overflow_policy(
            int(contract.get("controls", {}).get("maximum_frontier_rows", -1))
        ),
        type(contract.get("branch_bound_evidence")) is dict,
        contract.get("branch_bound_evidence", {}).get("digest")
        == _hash(_without(contract.get("branch_bound_evidence", {}), {"digest"})),
        contract.get("authority_root_policy") == "write-isolated truth; no candidate result input",
        _manifest_valid(contract.get("implementation_manifest"), repository_root),
        set(contract.get("implementation_manifest", {}).get("files", {})) == expected_authority_files,
        contract.get("runner_sha256") == _file_hash(authority_builder),
        contract.get("real_forward_outcomes_accessed") is False,
        contract.get("contract_digest") == _hash(_without(contract, {"contract_digest"})),
        set(matrix) == matrix_keys,
        matrix.get("schema_version") == AUTHORITY_MATRIX_SCHEMA,
        matrix.get("contract_digest") == contract["contract_digest"],
        matrix.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
        matrix.get("generation_id") == FROZEN_GENERATION_ID,
        matrix.get("completed_cases") == 60, matrix.get("invalid_cases") == [],
        type(matrix.get("elapsed_seconds")) in {int, float}
        and isfinite(matrix["elapsed_seconds"]) and matrix["elapsed_seconds"] >= 0,
        _timestamp(matrix.get("created_at"), "authority matrix") is not None,
        len(measurements) == 60,
        [row.get("registry_case_id") for row in measurements] == [case["case_id"] for case in registry["cases_data"]],
        matrix.get("p95_exact_seconds") == float(pd.Series(exact_seconds).quantile(.95)),
        matrix.get("maximum_exact_seconds") == max(exact_seconds),
        matrix.get("total_exact_seconds") == sum(exact_seconds),
        matrix.get("p95_search_seconds") == float(pd.Series(search_seconds).quantile(.95)),
        matrix.get("maximum_search_seconds") == max(search_seconds),
        matrix.get("total_search_seconds") == sum(search_seconds),
        matrix.get("maximum_worker_rss_mb") == max(float(row["peak_rss_mb"]) for row in measurements),
        matrix.get("gates") == matrix_gates,
        matrix.get("gate_passed") is True,
        matrix.get("performance_gates") == performance_gates,
        matrix.get("performance_gate_passed") is all(performance_gates.values()),
        matrix.get("measurement_integrity_digest") == _hash({
            "measurements": matrix["measurements"], "p95_exact_seconds": matrix["p95_exact_seconds"],
            "maximum_exact_seconds": matrix["maximum_exact_seconds"], "total_exact_seconds": matrix["total_exact_seconds"],
            "p95_search_seconds": matrix["p95_search_seconds"], "maximum_search_seconds": matrix["maximum_search_seconds"],
            "total_search_seconds": matrix["total_search_seconds"], "maximum_worker_rss_mb": matrix["maximum_worker_rss_mb"],
            "performance_gates": matrix["performance_gates"], "performance_gate_passed": matrix["performance_gate_passed"],
        }),
        matrix.get("result_digest") == _hash(_without(matrix, AUTHORITY_MATRIX_OMITTED)),
        set(seal) == seal_keys,
        seal.get("schema_version") == AUTHORITY_SEAL_SCHEMA,
        seal.get("contract_digest") == contract["contract_digest"],
        seal.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
        seal.get("authority_matrix_digest") == matrix["result_digest"],
        seal.get("branch_bound_evidence_digest") == contract.get("branch_bound_evidence", {}).get("digest") == matrix.get("branch_bound_evidence_digest"),
        seal.get("measurement_integrity_digest") == matrix.get("measurement_integrity_digest"),
        seal.get("authority_cases") == 60,
        seal.get("seal_scope") == "exact authority correctness only",
        seal.get("authority_correctness_sealed") is True,
        seal.get("performance_gate_passed") is matrix.get("performance_gate_passed"),
        seal.get("production_promotion_authorized") is False,
        seal.get("candidate_results_opened") is False,
        seal.get("real_forward_outcomes_accessed") is False,
        seal.get("seal_digest") == _hash(_without(seal, {"seal_digest"})),
    )):
        raise ValueError("sealed v4 authority aggregate differs")
    matrix_rows = list(matrix.get("cases", []))
    if len(matrix_rows) != 60 or [row.get("query_episode_id") for row in matrix_rows] != ordered:
        raise ValueError("authority matrix order differs")
    matrix_by_id = {row["query_episode_id"]: row for row in matrix_rows}
    expected_case_paths = sorted(authority_root / "cases" / f"{query_id}.json" for query_id in ordered)
    if sorted((authority_root / "cases").glob("*.json")) != expected_case_paths:
        raise ValueError("authority case path set differs")
    authorities: dict[str, dict[str, Any]] = {}
    for case in registry["cases_data"]:
        query_id = str(case["episode_id"])
        authority = _read(authority_root / "cases" / f"{query_id}.json")
        matches = list(authority.get("matches", []))
        ids = [str(row.get("episode_id")) for row in matches]
        certificate = dict(authority.get("certificate", {}))
        row = matrix_by_id.get(query_id, {})
        expected_case_keys = {
            "schema_version", "status", "contract_digest", "registry_digest", "generation_id",
            "registry_case_id", "query_episode_id", "query_symbol", "query_cutoff", "query_start",
            "latest_eligible_cutoff", "query_stock_prefix", "query_benchmark_prefix",
            "primary_proposal_result_digest", "proposal_result_digest", "proposal_seconds",
            "amortized_proposal_seconds", "frontier_attempts", "streaming_threshold_closure_used",
            "frontier_limit_rows", "frontier_attempt_measurements", "matches", "certificate",
            "certificate_digest", "exact_seconds", "final_exact_seconds", "peak_rss_mb",
            "real_forward_outcomes_accessed", "created_at", "gates", "gate_passed",
            "result_digest", "checkpoint_integrity_digest",
        }
        expected_gates = _authority_case_gates(authority, case, contract)
        expected_context = expected_queries[query_id]
        if not all((
            set(authority) == expected_case_keys,
            authority.get("schema_version") == AUTHORITY_CASE_SCHEMA,
            authority.get("status") == "completed",
            authority.get("contract_digest") == contract["contract_digest"],
            authority.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
            authority.get("generation_id") == FROZEN_GENERATION_ID,
            authority.get("registry_case_id") == case["case_id"],
            authority.get("query_episode_id") == query_id,
            authority.get("query_symbol") == case.get("symbol"),
            authority.get("query_cutoff") == case.get("cutoff"),
            pd.Timestamp(authority.get("query_start")).value == expected_context["query_start_ns"],
            pd.Timestamp(authority.get("latest_eligible_cutoff")).value == expected_context["latest_eligible_ns"],
            authority.get("query_stock_prefix") == case.get("stock_prefix"),
            authority.get("query_benchmark_prefix") == case.get("benchmark_prefix"),
            _timestamp(authority.get("created_at"), "authority case") is not None,
            all(type(authority.get(name)) in {int, float} and isfinite(authority[name]) and authority[name] >= 0 for name in (
                "proposal_seconds", "amortized_proposal_seconds", "exact_seconds",
                "final_exact_seconds", "peak_rss_mb",
            )),
            len(matches) == 20, len(ids) == len(set(ids)),
            authority.get("gates") == expected_gates,
            authority.get("gate_passed") is True and all(expected_gates.values()),
            authority.get("certificate_digest") == certificate.get("result_digest"),
            certificate.get("result_digest") == _certificate_digest(certificate, matches),
            authority.get("result_digest") == _hash(_without(authority, AUTHORITY_CASE_OMITTED)),
            authority.get("checkpoint_integrity_digest") == _hash(_without(
                authority, {"created_at", "checkpoint_integrity_digest"},
            )),
            authority.get("real_forward_outcomes_accessed") is False,
            row.get("registry_case_id") == case["case_id"],
            row.get("authority_digest") == authority.get("result_digest"),
            row.get("certificate_digest") == authority.get("certificate_digest"),
            set(row) == {"registry_case_id", "query_episode_id", "authority_digest", "certificate_digest", "eligible_candidates", "exact_evaluated"},
            row.get("eligible_candidates") == certificate.get("eligible_candidates"),
            row.get("exact_evaluated") == certificate.get("exact_evaluated"),
        )):
            raise ValueError(f"sealed authority case differs:{case['case_id']}")
        authorities[query_id] = authority
    if matrix.get("gates", {}).get("all_case_certificates_pass") is not all(
        value.get("gate_passed") is True for value in authorities.values()
    ):
        raise ValueError("authority matrix case gate aggregate differs")
    for case, measurement in zip(registry["cases_data"], measurements, strict=True):
        authority = authorities[str(case["episode_id"])]
        if not all((
            set(measurement) == {"registry_case_id", "proposal_seconds", "exact_seconds", "search_seconds", "peak_rss_mb"},
            measurement.get("registry_case_id") == case["case_id"],
            measurement.get("proposal_seconds") == authority.get("proposal_seconds"),
            measurement.get("exact_seconds") == authority.get("exact_seconds"),
            measurement.get("search_seconds") == float(authority["proposal_seconds"]) + float(authority["exact_seconds"]),
            measurement.get("peak_rss_mb") == authority.get("peak_rss_mb"),
        )):
            raise ValueError("authority measurement row differs from sealed case")
    if matrix.get("streaming_threshold_closure_cases") != sum(
        bool(value.get("streaming_threshold_closure_used")) for value in authorities.values()
    ):
        raise ValueError("authority streaming-closure aggregate differs")
    expected_files = {
        "authority-contract.json", "authority-matrix.json", "authority-matrix.html", "SEALED.json",
        *(f"cases/{query_id}.json" for query_id in ordered),
    }
    observed_files: set[str] = set()
    for path in authority_root.rglob("*"):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise ValueError("authority tree contains linked or special entry")
        if path.is_file():
            observed_files.add(str(path.relative_to(authority_root)))
    if observed_files != expected_files:
        raise ValueError("authority terminal artifact tree differs")
    return authorities, matrix, seal


def _validate_marker(
    comparison_root: Path, contract: Mapping[str, Any], semantic_seal: Mapping[str, Any],
    performance_final: Mapping[str, Any], run_complete: Mapping[str, Any],
) -> dict[str, Any]:
    marker = _read(comparison_root / "RESULTS_OPENED.json")
    expected = {
        "schema_version": RESULTS_OPENED_SCHEMA,
        "registry_digest": FROZEN_REGISTRY_DIGEST,
        "producer_contract_digest": contract["contract_digest"],
        "semantic_seal_digest": semantic_seal["seal_digest"],
        "performance_final_digest": performance_final["final_digest"],
        "run_complete_digest": run_complete["complete_digest"],
        "status": "authority results about to be opened exactly once",
    }
    if set(marker) != {*expected, "created_at", "result_digest"}:
        raise ValueError("RESULTS_OPENED marker fields differ")
    if not all((
        all(marker.get(key) == value for key, value in expected.items()),
        marker.get("result_digest") == _hash(expected),
        _timestamp(marker.get("created_at"), "RESULTS_OPENED") is not None,
    )):
        raise ValueError("RESULTS_OPENED marker differs")
    return marker


def _validate_comparison(
    registry: Mapping[str, Any], contract: Mapping[str, Any],
    semantics: Mapping[str, Mapping[str, Any]], semantic_seal: Mapping[str, Any],
    performance_final: Mapping[str, Any], authorities: Mapping[str, Mapping[str, Any]],
    authority_matrix: Mapping[str, Any], authority_seal: Mapping[str, Any],
    marker: Mapping[str, Any], comparison_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_cases, failures, total = [], [], 0
    for case in registry["cases_data"]:
        query_id = str(case["episode_id"])
        candidate_ids = {str(row["episode_id"]) for row in semantics[query_id]["candidates"]}
        truth = [str(row["episode_id"]) for row in authorities[query_id]["matches"]]
        retained = [value for value in truth if value in candidate_ids]
        count = len(retained)
        total += count
        case_failures = [] if count >= 19 else ["candidate retained fewer than 19 of 20"]
        expected_cases.append({
            "registry_case_id": case["case_id"], "query_episode_id": query_id,
            "candidate_semantic_digest": semantics[query_id]["semantic_digest"],
            "authority_case_digest": authorities[query_id]["result_digest"],
            "candidate_count": len(candidate_ids), "retained_count": count,
            "recall_at_20": count / 20.0, "perfect_20_of_20": count == 20,
            "missing_authority_episode_ids": [
                value for value in truth if value not in candidate_ids
            ],
            "failures": case_failures, "passed": not case_failures,
        })
        failures.extend(f"{case['case_id']}:{value}" for value in case_failures)
    gates = {
        "all_60_authority_cases_valid": len(expected_cases) == 60,
        "every_case_retains_at_least_19_of_20": all(
            row["retained_count"] >= 19 for row in expected_cases
        ),
        "aggregate_retains_at_least_1188_of_1200": total >= 1_188,
    }
    comparison = _read(comparison_root / "candidate-comparison.json")
    expected = {
        "schema_version": COMPARISON_MATRIX_SCHEMA,
        "registry_digest": FROZEN_REGISTRY_DIGEST,
        "producer_contract_digest": contract["contract_digest"],
        "semantic_seal_digest": semantic_seal["seal_digest"],
        "performance_final_digest": performance_final["final_digest"],
        "performance_passed": performance_final["performance_passed"],
        "results_opened_marker_digest": marker["result_digest"],
        "authority_contract_digest": authority_matrix["contract_digest"],
        "authority_matrix_digest": authority_matrix["result_digest"],
        "authority_seal_digest": authority_seal["seal_digest"],
        "completed_cases": 60, "retained_total": total,
        "retained_denominator": 1_200,
        "minimum_retained_count": min(row["retained_count"] for row in expected_cases),
        "perfect_20_of_20_cases": sum(row["perfect_20_of_20"] for row in expected_cases),
        "perfect_20_of_20_is_descriptive_only": True,
        "cases": expected_cases, "failures": failures, "gates": gates,
        "passed": all(gates.values()), "candidate_results_opened": True,
        "production_promotion_authorized": False,
        "real_forward_outcomes_accessed": False,
    }
    if set(comparison) != {*expected, "created_at", "result_digest"}:
        raise ValueError("comparison matrix fields differ")
    if not all((
        all(comparison.get(key) == value for key, value in expected.items()),
        comparison.get("result_digest") == _hash(expected),
        _timestamp(comparison.get("created_at"), "comparison")
        >= _timestamp(marker.get("created_at"), "RESULTS_OPENED"),
    )):
        raise ValueError("comparison reconstruction differs")
    seal = _read(comparison_root / "SEALED.json")
    seal_expected = {
        "schema_version": COMPARISON_SEAL_SCHEMA,
        "registry_digest": FROZEN_REGISTRY_DIGEST,
        "producer_contract_digest": contract["contract_digest"],
        "comparison_digest": comparison["result_digest"],
        "results_opened_marker_digest": marker["result_digest"],
        "authority_seal_digest": authority_seal["seal_digest"],
        "candidate_results_opened": True,
        "comparison_gate_passed": comparison["passed"],
        "production_promotion_authorized": False,
    }
    if set(seal) != {*seal_expected, "created_at", "seal_digest"}:
        raise ValueError("comparison seal fields differ")
    if not all((
        all(seal.get(key) == value for key, value in seal_expected.items()),
        seal.get("seal_digest") == _hash(seal_expected),
        _timestamp(seal.get("created_at"), "comparison seal")
        >= _timestamp(comparison.get("created_at"), "comparison"),
    )):
        raise ValueError("comparison seal differs")
    observed_files = {
        str(path.relative_to(comparison_root)) for path in comparison_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if observed_files != {"RESULTS_OPENED.json", "candidate-comparison.json", "SEALED.json"} or any(
        path.is_symlink() or (not path.is_file() and not path.is_dir())
        for path in comparison_root.rglob("*")
    ):
        raise ValueError("comparison terminal artifact tree differs")
    return comparison, seal


def verify_comparison(
    *, config: Path, artifact_dir: Path, repository_root: Path, registry_root: Path,
    candidate_root: Path, authority_root: Path, comparison_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Verify all sealed inputs and atomically publish only verification evidence."""
    actual_repository = Path(__file__).resolve().parents[2]
    if repository_root.resolve() != actual_repository:
        raise ValueError("repository root differs from verifier implementation root")
    repository_root = actual_repository
    if config.resolve() != repository_root / "config/datasets.example.yaml":
        raise ValueError("verification config path differs")
    roots = expected_roots(artifact_dir)
    observed = {
        "registry_root": str(registry_root.resolve()),
        "candidate_root": str(candidate_root.resolve()),
        "authority_root": str(authority_root.resolve()),
        "comparison_root": str(comparison_root.resolve()),
        "verification_root": str(output_root.resolve()),
    }
    if any(observed[key] != roots[key] for key in observed):
        raise ValueError("verification input or output root differs")
    protected = [Path(roots[key]) for key in ("authority_root", "candidate_root", "comparison_root", "verification_root")]
    if any(repository_root == path or repository_root.is_relative_to(path) or path.is_relative_to(repository_root) for path in protected):
        raise ValueError("repository and evidence roots overlap")
    output_path = output_root / "verification.json"
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("verification root must be fresh and empty")
    failures: list[str] = []
    digests: dict[str, Any] = {}
    comparison: dict[str, Any] = {}
    try:
        registry = _read(registry_root / "query-registry.json")
        contract, prereg, roles, execution = _validate_contract_and_prereg(
            registry, artifact_dir, repository_root, candidate_root,
        )
        expected_queries = {
            str(case["episode_id"]): _query_context(config, case)
            for case in registry["cases_data"]
        }
        resident = _validate_resident(candidate_root, contract)
        semantics, semantic_seal, performance_final, run_complete = (
            _validate_candidate_aggregate(
                registry, candidate_root, contract, prereg, roles, execution, resident,
                expected_queries, repository_root,
            )
        )
        marker = _validate_marker(
            comparison_root, contract, semantic_seal, performance_final, run_complete,
        )
        authorities, authority_matrix, authority_seal = _validate_authorities(
            registry, authority_root, repository_root, expected_queries,
        )
        comparison, comparison_seal = _validate_comparison(
            registry, contract, semantics, semantic_seal, performance_final,
            authorities, authority_matrix, authority_seal, marker, comparison_root,
        )
        digests = {
            "preregistration_digest": prereg["preregistration_digest"],
            "producer_contract_digest": contract["contract_digest"],
            "semantic_seal_digest": semantic_seal["seal_digest"],
            "performance_final_digest": performance_final["final_digest"],
            "run_complete_digest": run_complete["complete_digest"],
            "results_opened_marker_digest": marker["result_digest"],
            "authority_matrix_digest": authority_matrix["result_digest"],
            "authority_seal_digest": authority_seal["seal_digest"],
            "comparison_digest": comparison["result_digest"],
            "comparison_seal_digest": comparison_seal["seal_digest"],
        }
    except Exception as exc:
        failures.append(f"{type(exc).__name__}:{exc}")
    deterministic = {
        "schema_version": VERIFICATION_SCHEMA,
        "registry_digest": FROZEN_REGISTRY_DIGEST,
        **digests,
        "verified_cases": 60 if not failures else 0,
        "minimum_retained_per_case": 19,
        "minimum_retained_aggregate": 1_188,
        "aggregate_denominator": 1_200,
        "performance_terminal_fail_allowed": True,
        "comparison_gate_passed": comparison.get("passed") if comparison else None,
        "failures": failures, "passed": not failures,
        "verification_implementation_sha256": _file_hash(Path(__file__)),
        "real_forward_outcomes_accessed": False,
    }
    payload = {
        **deterministic, "created_at": datetime.now(timezone.utc).isoformat(),
        "result_digest": _hash(deterministic),
    }
    _atomic_json(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[2] / "config/datasets.example.yaml",
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = verify_comparison(**vars(args))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
