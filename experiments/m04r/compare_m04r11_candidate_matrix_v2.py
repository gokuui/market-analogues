"""One-shot comparison of sealed M04R-11 v2 pools with sealed v4 truth.

All candidate-side evidence is validated before ``RESULTS_OPENED.json`` is
durably published.  Authority JSON is never read through this module before
that transition.  A marker without a final seal is intentionally terminal.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.metadata
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import platform
import sys
from typing import Any, Callable, Mapping
from uuid import uuid4

import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from m04r11_candidate_v2_contract import (  # noqa: E402
    CASE_BUNDLE_SCHEMA, CLAIMS_POLICY, COMPARISON_MATRIX_SCHEMA,
    COMPARISON_POLICY, COMPARISON_SEAL_SCHEMA, FROZEN_GENERATION_ID,
    FROZEN_PROPOSAL_CONTRACT_DIGEST, FROZEN_REGISTRY_DIGEST,
    FROZEN_ROUTE_QUOTAS, PERFORMANCE_ATTEMPT_GATES,
    PERFORMANCE_ATTEMPT_SCHEMA, PERFORMANCE_FINAL_SCHEMA,
    PERFORMANCE_LIMITS, PERFORMANCE_MATRIX_GATES, PERFORMANCE_MATRIX_SCHEMA,
    RESIDENT_BINDING_SCHEMA, RESIDENT_CONTENT_SCHEMA, RESULTS_OPENED_SCHEMA,
    RUN_COMPLETE_SCHEMA,
    RUN_LEDGER_EVENT_SCHEMA, RUN_LEDGER_HEAD_SCHEMA, SEMANTIC_CASE_GATES,
    SEMANTIC_CASE_SCHEMA, SEMANTIC_MATRIX_GATES, SEMANTIC_MATRIX_SCHEMA,
    SEMANTIC_SEAL_SCHEMA, execution_query_ids, expected_preregistration_path,
    load_and_validate_preregistration,
    expected_roots, performance_attempt_digest, performance_matrix_digest,
    semantic_case_digest, semantic_matrix_digest, terminal_digest,
    validate_exact_roots,
    validate_producer_contract, validate_role_table,
)
from market_analogues.adapters import file_fingerprint, source_from_spec  # noqa: E402
from market_analogues.config import load_config  # noqa: E402
from market_analogues.episodes import build_episode  # noqa: E402
from market_analogues.m04r_candidate_evidence import (  # noqa: E402
    reconstructed_candidate_digest, scan_result_digest,
)
from market_analogues.m04r_certified_search_verification import (  # noqa: E402
    _certificate_digest,
)
from market_analogues.packed_bound_store import load_packed_generation  # noqa: E402
from market_analogues.packed_bound_search import SEARCH_SCHEMA_VERSION  # noqa: E402
from market_analogues.representation import (  # noqa: E402
    represent, representation_input_digest,
)
from market_analogues.resident_store import READY_SCHEMA_VERSION  # noqa: E402
from market_analogues.search import latest_eligible_cutoff  # noqa: E402
from market_analogues.types import InstrumentKey, stable_hash  # noqa: E402


AUTHORITY_CONTRACT_SCHEMA = "m04r11-authority-build-contract-v4"
AUTHORITY_CASE_SCHEMA = "m04r11-certified-authority-case-v4"
AUTHORITY_MATRIX_SCHEMA = "m04r11-certified-authority-matrix-v4"
AUTHORITY_SEAL_SCHEMA = "m04r11-authority-seal-v4"
LEDGER_GENESIS = "0" * 64
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


@dataclass(frozen=True)
class PreopenEvidence:
    registry: dict[str, Any]
    contract: dict[str, Any]
    preregistration: dict[str, Any]
    resident_binding: dict[str, Any]
    bundles: tuple[dict[str, Any], ...]
    semantics_by_id: dict[str, dict[str, Any]]
    performance_by_id: dict[str, dict[str, Any]]
    semantic_matrix: dict[str, Any]
    semantic_seal: dict[str, Any]
    performance_matrix: dict[str, Any]
    performance_final: dict[str, Any]
    run_complete: dict[str, Any]
    ledger_head: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably publish one owned artifact without replacing any inode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(
                dict(payload), handle, indent=2, sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _exact_keys(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(payload) != expected:
        raise ValueError(f"{label} fields differ")


def _validate_timestamp(value: Any, label: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{label} timestamp differs")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} timestamp is malformed") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} timestamp lacks timezone")


def _manifest_valid(payload: Mapping[str, Any]) -> bool:
    try:
        files = dict(payload["files"])
        return (
            set(payload) == {"files", "digest"} and bool(files)
            and payload.get("digest") == stable_hash(files)
            and all(
                isinstance(name, str) and name and not Path(name).is_absolute()
                and ".." not in Path(name).parts
                and isinstance(value, str) and len(value) == 64
                for name, value in files.items()
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def _implementation_manifest_from_frozen(
    repository_root: Path, frozen: Mapping[str, Any],
) -> dict[str, Any]:
    if not _manifest_valid(frozen):
        raise ValueError("frozen producer implementation manifest is malformed")
    root = repository_root.resolve()
    files: dict[str, str] = {}
    for relative in frozen["files"]:
        path = (root / str(relative)).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("frozen producer implementation file is absent or unsafe")
        files[str(relative)] = file_fingerprint(path)
    observed = {"files": files, "digest": stable_hash(files)}
    if observed != dict(frozen):
        raise ValueError("producer implementation manifest changed before truth open")
    return observed


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
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "cpu_count": os.cpu_count(),
        "numba_num_threads": os.environ.get("NUMBA_NUM_THREADS"),
    }
    return {**deterministic, "digest": stable_hash(deterministic)}


def _source_pack_binding(source_full_root: Path) -> dict[str, Any]:
    store = source_full_root / "store"
    loaded = load_packed_generation(
        store, FROZEN_GENERATION_ID, verify_content=True,
        validate_records=False,
    )
    manifest = loaded.manifest
    generation = store / "generations" / FROZEN_GENERATION_ID
    manifest_path = generation / "manifest.json"
    rows_path = generation / str(manifest["rows_file"])
    overflow_path = generation / str(manifest["overflow_file"])
    return {
        "generation_id": FROZEN_GENERATION_ID,
        "source_full_root": str(source_full_root.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": file_fingerprint(manifest_path),
        "manifest_digest": str(manifest["manifest_digest"]),
        "provenance_digest": str(manifest["provenance_digest"]),
        "rows_file": str(manifest["rows_file"]),
        "rows_path": str(rows_path.resolve()),
        "rows_bytes": int(manifest["rows_bytes"]),
        "rows_sha256": str(manifest["rows_sha256"]),
        "row_count": len(loaded.rows),
        "row_bytes": int(loaded.rows.dtype.itemsize),
        "overflow_file": str(manifest["overflow_file"]),
        "overflow_path": str(overflow_path.resolve()),
        "overflow_bytes": int(manifest["overflow_bytes"]),
        "overflow_sha256": str(manifest["overflow_sha256"]),
        "overflow_count": len(loaded.overflow),
        "overflow_row_bytes": int(loaded.overflow.dtype.itemsize),
        "physical_rows": len(loaded.rows) + len(loaded.overflow),
        "active_pointer_absent": not (store / "active.json").exists(),
    }


def _expected_resident_content_digest(source_pack: Mapping[str, Any]) -> str:
    manifest_path = Path(str(source_pack["manifest_path"]))
    manifest = _read_json(manifest_path)
    file_content = {
        "manifest": {
            "bytes": manifest_path.stat().st_size,
            "sha256": source_pack["manifest_sha256"],
        },
        "rows": {
            "bytes": source_pack["rows_bytes"],
            "sha256": source_pack["rows_sha256"],
        },
        "overflow": {
            "bytes": source_pack["overflow_bytes"],
            "sha256": source_pack["overflow_sha256"],
        },
    }
    content = {
        "schema_version": RESIDENT_CONTENT_SCHEMA,
        "generation_id": FROZEN_GENERATION_ID,
        "provenance_digest": source_pack["provenance_digest"],
        "manifest_digest": source_pack["manifest_digest"],
        "pack_contract_digest": manifest["pack_contract_digest"],
        "quantized_bound_contract_digest": manifest[
            "quantized_bound_contract_digest"
        ],
        "physical_generation_bytes": sum(
            int(file_content[name]["bytes"])
            for name in ("manifest", "rows", "overflow")
        ),
        "source_files": file_content,
        "mirror_files": file_content,
    }
    return stable_hash(content)


def _expected_query_context(config_path: Path, case: Mapping[str, Any]) -> dict[str, Any]:
    config = load_config(config_path)
    source = source_from_spec(config.datasets["nasdaq"])
    episode = build_episode(
        source, InstrumentKey("nasdaq", str(case["symbol"])),
        str(case["cutoff"]), int(case["lookback"]),
        str(case["representation_version"]),
    )
    if episode.key.id != case["episode_id"]:
        raise ValueError("rebuilt comparison query differs from frozen registry")
    instrument = InstrumentKey("nasdaq", str(case["symbol"]))
    stock_prefix = asdict(source.causal_prefix_fingerprint(
        instrument, str(case["cutoff"]),
    ))
    benchmark_raw = source.benchmark_causal_prefix_fingerprint(str(case["cutoff"]))
    benchmark_prefix = asdict(benchmark_raw) if benchmark_raw is not None else None
    if (
        stock_prefix != case["stock_prefix"]
        or benchmark_prefix != case["benchmark_prefix"]
    ):
        raise ValueError("comparison query causal prefix differs from registry")
    return {
        "query_episode_id": episode.key.id,
        "query_symbol": episode.key.instrument.source_symbol,
        "query_start_ns": int(pd.Timestamp(episode.bars.timestamp.iloc[0]).value),
        "latest_eligible_ns": int(latest_eligible_cutoff(episode, 60).value),
        "query_stock_prefix": stock_prefix,
        "query_benchmark_prefix": benchmark_prefix,
        "query_representation_digest": representation_input_digest(represent(episode)),
    }


def _bundle_digest(payload: Mapping[str, Any]) -> str:
    return stable_hash({key: value for key, value in payload.items() if key != "bundle_digest"})


def _validate_bundle(
    bundle: Mapping[str, Any], *, contract: Mapping[str, Any],
    case: Mapping[str, Any], role: Mapping[str, Any],
    expected_query: Mapping[str, Any], resident: Mapping[str, Any],
    physical_rows: int, execution_ordinal: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _exact_keys(bundle, {
        "schema_version", "producer_contract_digest", "execution_ordinal",
        "query_episode_id", "semantic", "performance", "bundle_digest",
    }, "case bundle")
    if not all((
        bundle.get("schema_version") == CASE_BUNDLE_SCHEMA,
        bundle.get("producer_contract_digest") == contract["contract_digest"],
        bundle.get("execution_ordinal") == execution_ordinal,
        bundle.get("query_episode_id") == case["episode_id"],
        bundle.get("bundle_digest") == _bundle_digest(bundle),
    )):
        raise ValueError("case bundle identity or digest differs")
    semantic = dict(bundle["semantic"])
    performance = dict(bundle["performance"])
    _exact_keys(semantic, {
        "schema_version", "producer_contract_digest", "registry_digest",
        "generation_id", "proposal_contract_digest", "resident_content_digest",
        "registry_case_id", "query_episode_id", "query_symbol", "query_start_ns",
        "latest_eligible_ns", "query_stock_prefix", "query_benchmark_prefix",
        "query_representation_digest", "performance_role", "recall_role",
        "scan_semantics", "candidates", "candidate_digest_reconstructed",
        "violations", "gates", "passed", "real_forward_outcomes_accessed",
        "created_at", "semantic_digest",
    }, "semantic case")
    _validate_timestamp(semantic["created_at"], "semantic case")
    if not all((
        semantic.get("schema_version") == SEMANTIC_CASE_SCHEMA,
        semantic.get("producer_contract_digest") == contract["contract_digest"],
        semantic.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
        semantic.get("generation_id") == FROZEN_GENERATION_ID,
        semantic.get("proposal_contract_digest") == FROZEN_PROPOSAL_CONTRACT_DIGEST,
        semantic.get("resident_content_digest") == resident["resident_content_digest"],
        semantic.get("registry_case_id") == case["case_id"],
        semantic.get("performance_role") == role["performance_role"],
        semantic.get("recall_role") == role["recall_role"],
        semantic.get("real_forward_outcomes_accessed") is False,
        all(semantic.get(key) == value for key, value in expected_query.items()),
        semantic.get("semantic_digest") == semantic_case_digest(semantic),
    )):
        raise ValueError("semantic identity, provenance or digest differs")
    scans = list(semantic["scan_semantics"])
    if len(scans) != 3 or not (scans[0] == scans[1] == scans[2]):
        raise ValueError("semantic scans differ")
    for scan in scans:
        _exact_keys(scan, {
            "schema_version", "generation_id", "query_episode_id", "rows_scanned",
            "eligible_rows", "eligible_main_rows", "eligible_overflow_rows",
            "route_counts", "route_quotas", "candidate_count", "candidate_digest",
            "result_digest",
        }, "scan semantic")
        if not all((
            scan["schema_version"] == SEARCH_SCHEMA_VERSION,
            scan["generation_id"] == FROZEN_GENERATION_ID,
            scan["query_episode_id"] == case["episode_id"],
            scan["rows_scanned"] == physical_rows,
            scan["eligible_rows"] == scan["eligible_main_rows"] + scan["eligible_overflow_rows"],
            scan["route_quotas"] == dict(FROZEN_ROUTE_QUOTAS),
            set(scan["route_counts"]) == set(FROZEN_ROUTE_QUOTAS),
            all(type(value) is int and value >= 0 for value in scan["route_counts"].values()),
            scan["result_digest"] == scan_result_digest(scan, FROZEN_PROPOSAL_CONTRACT_DIGEST),
        )):
            raise ValueError("scan semantic differs")
    candidates = list(semantic["candidates"])
    bounds: list[float] = []
    candidate_ids: list[str] = []
    for row in candidates:
        _exact_keys(row, {
            "episode_id", "symbol", "cutoff_ns", "quality_tier",
            "lower_bound_hex", "routes", "overflow_fallback",
        }, "candidate")
        bound = float.fromhex(str(row["lower_bound_hex"]))
        episode_id = str(row["episode_id"])
        if not all((
            isfinite(bound) and bound >= 0, len(episode_id) == 24,
            episode_id == episode_id.lower(), len(bytes.fromhex(episode_id)) == 12,
            type(row["cutoff_ns"]) is int, row["quality_tier"] in ("A", "B"),
            type(row["overflow_fallback"]) is bool,
            row["routes"] == sorted(set(row["routes"])) and bool(row["routes"]),
            set(row["routes"]).issubset(FROZEN_ROUTE_QUOTAS),
        )):
            raise ValueError("candidate metadata differs")
        bounds.append(bound)
        candidate_ids.append(episode_id)
    if list(zip(bounds, candidate_ids)) != sorted(zip(bounds, candidate_ids)):
        raise ValueError("candidate ordering differs")
    candidate_digest = reconstructed_candidate_digest(candidates)
    route_counts = {
        route: sum(route in row["routes"] for row in candidates)
        for route in FROZEN_ROUTE_QUOTAS
    }
    violations = {
        "duplicates": len(candidate_ids) - len(set(candidate_ids)),
        "future": sum(row["cutoff_ns"] > expected_query["latest_eligible_ns"] for row in candidates),
        "same_symbol_overlap": sum(
            row["symbol"] == expected_query["query_symbol"]
            and row["cutoff_ns"] >= expected_query["query_start_ns"]
            for row in candidates
        ),
        "tier": sum(row["quality_tier"] not in ("A", "B") for row in candidates),
    }
    candidate_valid = all((
        candidate_digest == semantic["candidate_digest_reconstructed"],
        candidate_digest == scans[0]["candidate_digest"],
        len(candidates) == scans[0]["candidate_count"],
        route_counts == scans[0]["route_counts"],
    ))
    semantic_gates = {
        "resident_content_matches_contract": semantic["resident_content_digest"] == resident["resident_content_digest"],
        "query_identity_prefix_and_representation_match": all(semantic.get(key) == value for key, value in expected_query.items()),
        "three_scan_digest_and_block_order_invariance": all(
            row["result_digest"] == scan_result_digest(row, FROZEN_PROPOSAL_CONTRACT_DIGEST)
            for row in scans
        ),
        "internal_eligible_row_accounting": scans[0]["eligible_rows"] == scans[0]["eligible_main_rows"] + scans[0]["eligible_overflow_rows"],
        "physical_row_accounting": scans[0]["rows_scanned"] == physical_rows,
        "frozen_route_quotas": scans[0]["route_quotas"] == dict(FROZEN_ROUTE_QUOTAS),
        "candidate_digest_order_and_routes_reconstruct": candidate_valid,
        "zero_temporal_overlap_tier_duplicate_errors": not any(violations.values()),
        "real_forward_outcomes_excluded": True,
    }
    if tuple(semantic_gates) != SEMANTIC_CASE_GATES or not all((
        semantic["violations"] == violations, semantic["gates"] == semantic_gates,
        semantic["passed"] is True, all(semantic_gates.values()),
    )):
        raise ValueError("semantic case gates differ")

    _exact_keys(performance, {
        "schema_version", "producer_contract_digest", "registry_digest",
        "generation_id", "resident_binding_digest", "semantic_digest",
        "registry_case_id", "query_episode_id", "performance_role",
        "attempt_ordinal", "ready_start", "ready_end", "timings", "gates",
        "passed", "real_forward_outcomes_accessed", "created_at",
        "performance_digest",
    }, "performance attempt")
    _validate_timestamp(performance["created_at"], "performance attempt")
    timings = dict(performance["timings"])
    _exact_keys(timings, {
        "resident_first_seconds", "resident_reverse_seconds",
        "resident_repeat_seconds", "task_seconds", "peak_rss_mb",
    }, "performance timings")
    numeric = [float(value) for value in timings.values()]
    finite = all(isfinite(value) and value >= 0 for value in numeric)
    performance_gates = {
        "same_ready_instance_at_start_and_end": performance["ready_start"] == resident["resident_ready_observation"] and performance["ready_end"] == resident["resident_ready_observation"],
        "measurements_finite_nonnegative_and_task_contains_scans": finite and timings["task_seconds"] >= sum(timings[name] for name in ("resident_first_seconds", "resident_reverse_seconds", "resident_repeat_seconds")),
        "resident_first_scan_at_most_120_seconds": finite and timings["resident_first_seconds"] <= PERFORMANCE_LIMITS["resident_first_seconds"],
        "resident_repeat_scan_at_most_60_seconds": finite and timings["resident_repeat_seconds"] <= PERFORMANCE_LIMITS["resident_repeat_seconds"],
        "worker_rss_at_most_1536_mib": finite and timings["peak_rss_mb"] <= PERFORMANCE_LIMITS["worker_rss_mib"],
        "primary_attempt_completed": len(scans) == 3 and performance["attempt_ordinal"] == 1,
    }
    if tuple(performance_gates) != PERFORMANCE_ATTEMPT_GATES or not all((
        performance["schema_version"] == PERFORMANCE_ATTEMPT_SCHEMA,
        performance["producer_contract_digest"] == contract["contract_digest"],
        performance["registry_digest"] == FROZEN_REGISTRY_DIGEST,
        performance["generation_id"] == FROZEN_GENERATION_ID,
        performance["resident_binding_digest"] == resident["binding_digest"],
        performance["semantic_digest"] == semantic["semantic_digest"],
        performance["registry_case_id"] == case["case_id"],
        performance["query_episode_id"] == case["episode_id"],
        performance["performance_role"] == role["performance_role"],
        performance["real_forward_outcomes_accessed"] is False,
        performance["gates"] == performance_gates,
        performance["passed"] is all(performance_gates.values()),
        performance["performance_digest"] == performance_attempt_digest(performance),
    )):
        raise ValueError("performance attempt differs")
    return semantic, performance


def _load_ledger(root: Path, contract_digest: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = sorted((root / "ledger" / "events").glob("*.json"))
    events: list[dict[str, Any]] = []
    previous = LEDGER_GENESIS
    for index, path in enumerate(paths):
        if path.name != f"{index:06d}.json":
            raise ValueError("ledger filenames are not contiguous")
        event = _read_json(path)
        _exact_keys(event, {
            "schema_version", "producer_contract_digest", "event_index",
            "previous_event_digest", "event_type", "details", "created_at",
            "event_digest",
        }, "ledger event")
        if not isinstance(event.get("details"), dict):
            raise ValueError("ledger event details differ")
        _validate_timestamp(event["created_at"], "ledger event")
        deterministic = {key: value for key, value in event.items() if key != "event_digest"}
        if not all((
            event.get("schema_version") == RUN_LEDGER_EVENT_SCHEMA,
            event.get("event_index") == index,
            event.get("previous_event_digest") == previous,
            event.get("producer_contract_digest") == contract_digest,
            event.get("event_digest") == stable_hash(deterministic),
        )):
            raise ValueError("ledger chain differs")
        events.append(event)
        previous = str(event["event_digest"])
    head = _read_json(root / "ledger" / "HEAD.json")
    _exact_keys(head, {
        "schema_version", "producer_contract_digest", "event_count",
        "last_event_digest", "head_digest",
    }, "ledger head")
    head_deterministic = {key: value for key, value in head.items() if key != "head_digest"}
    if not all((
        head.get("schema_version") == RUN_LEDGER_HEAD_SCHEMA,
        head.get("producer_contract_digest") == contract_digest,
        head.get("event_count") == len(events),
        head.get("last_event_digest") == previous,
        head.get("head_digest") == stable_hash(head_deterministic),
    )):
        raise ValueError("ledger head differs")
    return events, head


def _validate_terminal_performance(
    performance_matrix: Mapping[str, Any],
    performance_final: Mapping[str, Any],
    contract_digest: str, resident_binding_digest: str,
) -> None:
    """Require terminal evidence while deliberately allowing a measured fail."""
    _exact_keys(performance_final, {
        "schema_version", "producer_contract_digest",
        "performance_matrix_digest", "resident_binding_digest",
        "performance_terminal", "performance_passed",
        "confirmatory_performance_cases", "exposed_regression_cases",
        "claims_policy", "authority_results_opened",
        "production_promotion_authorized", "created_at", "final_digest",
    }, "terminal performance")
    _validate_timestamp(performance_final["created_at"], "terminal performance")
    if not all((
        performance_final.get("schema_version") == PERFORMANCE_FINAL_SCHEMA,
        performance_final.get("producer_contract_digest") == contract_digest,
        performance_final.get("performance_matrix_digest")
        == performance_matrix.get("result_digest"),
        performance_final.get("resident_binding_digest")
        == resident_binding_digest,
        performance_final.get("performance_terminal") is True,
        type(performance_final.get("performance_passed")) is bool,
        performance_final.get("performance_passed")
        is performance_matrix.get("passed"),
        performance_final.get("confirmatory_performance_cases") == 53,
        performance_final.get("exposed_regression_cases") == 7,
        performance_final.get("claims_policy") == CLAIMS_POLICY,
        performance_final.get("authority_results_opened") is False,
        performance_final.get("production_promotion_authorized") is False,
        performance_final.get("final_digest")
        == terminal_digest(performance_final, "final_digest"),
    )):
        raise ValueError("terminal performance evidence differs")


def _validate_semantic_seal(
    semantic_matrix: Mapping[str, Any], semantic_seal: Mapping[str, Any],
    contract_digest: str, resident_content_digest: str,
) -> None:
    _exact_keys(semantic_seal, {
        "schema_version", "producer_contract_digest",
        "semantic_matrix_digest", "resident_start_content_digest",
        "resident_end_content_digest", "semantic_cases",
        "semantic_recall_ready", "performance_independent",
        "authority_results_opened", "production_promotion_authorized",
        "created_at", "seal_digest",
    }, "semantic seal")
    _validate_timestamp(semantic_seal["created_at"], "semantic seal")
    if not all((
        semantic_seal.get("schema_version") == SEMANTIC_SEAL_SCHEMA,
        semantic_seal.get("producer_contract_digest") == contract_digest,
        semantic_seal.get("semantic_matrix_digest")
        == semantic_matrix.get("result_digest"),
        semantic_seal.get("resident_start_content_digest")
        == semantic_matrix.get("resident_start_content_digest")
        == resident_content_digest,
        semantic_seal.get("resident_end_content_digest")
        == semantic_matrix.get("resident_end_content_digest")
        == resident_content_digest,
        semantic_seal.get("semantic_cases") == 60,
        semantic_seal.get("semantic_recall_ready") is True,
        semantic_seal.get("performance_independent") is True,
        semantic_seal.get("authority_results_opened") is False,
        semantic_seal.get("production_promotion_authorized") is False,
        semantic_seal.get("seal_digest")
        == terminal_digest(semantic_seal, "seal_digest"),
    )):
        raise ValueError("semantic seal differs")


def _assert_exact_candidate_tree(
    root: Path, execution_query_ids_value: list[str],
) -> None:
    """Reject every extra, missing, linked, or special candidate artifact."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("candidate artifact root differs")
    expected_files = {
        "candidate-contract.json", "RESIDENT_READY.json",
        "ledger/HEAD.json", "semantic-matrix.json", "SEMANTIC_SEALED.json",
        "performance-matrix.json", "PERFORMANCE_FINAL.json",
        "RUN_COMPLETE.json",
        *(f"ledger/events/{index:06d}.json" for index in range(122)),
        *(
            f"case-bundles/{ordinal:03d}-{query_id}.json"
            for ordinal, query_id in enumerate(execution_query_ids_value)
        ),
    }
    expected_directories = {"case-bundles", "ledger", "ledger/events"}
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for path in root.rglob("*"):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            raise ValueError("candidate artifact tree contains a symbolic link")
        if path.is_file():
            observed_files.add(relative)
        elif path.is_dir():
            observed_directories.add(relative)
        else:
            raise ValueError("candidate artifact tree contains a special file")
    if observed_files != expected_files or observed_directories != expected_directories:
        raise ValueError("candidate artifact tree differs from exact sealed tree")


def validate_preopen_candidate(
    *, registry: Mapping[str, Any], candidate_root: Path, artifact_dir: Path,
    repository_root: Path, expected_source_pack: Mapping[str, Any],
    expected_implementation_manifest: Mapping[str, Any],
    expected_environment_manifest: Mapping[str, Any],
    expected_query_contexts: Mapping[str, Mapping[str, Any]],
    expected_ready_observation: Mapping[str, Any] | None = None,
) -> PreopenEvidence:
    """Validate every candidate byte needed for comparison without truth access."""
    roots = expected_roots(artifact_dir)
    if candidate_root.resolve() != Path(roots["candidate_root"]):
        raise ValueError("candidate root differs from exact v2 root")
    if validate_exact_roots(roots, artifact_dir):
        raise ValueError("frozen v2 roots differ")
    if validate_role_table(registry):
        raise ValueError("frozen registry role table differs")
    if (candidate_root / "INCOMPLETE.json").exists():
        raise ValueError("candidate root is terminal incomplete")
    if any((candidate_root / name).exists() for name in (
        "RESULTS_OPENED.json", "candidate-comparison.json", "SEALED.json",
    )):
        raise ValueError("candidate root contains pre-open authority artifacts")
    preregistration = load_and_validate_preregistration(
        registry, artifact_dir, repository_root,
        expected_source_pack=expected_source_pack,
        expected_implementation_manifest=expected_implementation_manifest,
        expected_environment_manifest=expected_environment_manifest,
    )
    contract = _read_json(candidate_root / "candidate-contract.json")
    if contract != preregistration["producer_contract"]:
        raise ValueError("copied candidate contract differs from preregistration")
    contract_failures = validate_producer_contract(
        contract, registry, artifact_dir,
        expected_source_pack=expected_source_pack,
        expected_implementation_manifest=expected_implementation_manifest,
        expected_environment_manifest=expected_environment_manifest,
    )
    if contract_failures:
        raise ValueError(f"candidate contract differs:{contract_failures}")
    resident = _read_json(candidate_root / "RESIDENT_READY.json")
    _exact_keys(resident, {
        "schema_version", "producer_contract_digest", "resident_ready_path",
        "resident_ready_schema", "resident_content_digest",
        "resident_ready_observation", "resident_ready_payload",
        "resident_ready_bytes_base64", "validation_observation", "generation_id",
        "provenance_digest", "mirror_store_root", "storage_class",
        "latency_scope", "query_specific_inputs_used", "outcomes_or_labels_used",
        "real_forward_outcomes_accessed", "binding_digest",
    }, "resident binding")
    resident_deterministic = {key: value for key, value in resident.items() if key != "binding_digest"}
    stored_ready = dict(resident.get("resident_ready_observation", {}))
    ready_payload = dict(resident.get("resident_ready_payload", {}))
    try:
        ready_bytes = base64.b64decode(
            str(resident.get("resident_ready_bytes_base64", "")), validate=True,
        )
        decoded_ready = json.loads(ready_bytes)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("durable resident READY byte snapshot differs") from exc
    stored_lease = dict(stored_ready.get("file_identity_lease", {}))
    lease_deterministic = {
        key: value for key, value in stored_lease.items() if key != "lease_digest"
    }
    validation = dict(resident.get("validation_observation", {}))
    validation_deterministic = {
        key: value for key, value in validation.items()
        if key != "observation_digest"
    }
    if not all((
        resident.get("schema_version") == RESIDENT_BINDING_SCHEMA,
        resident.get("producer_contract_digest") == contract["contract_digest"],
        resident.get("generation_id") == FROZEN_GENERATION_ID,
        resident.get("resident_ready_schema") == contract["resident_ready_schema"],
        Path(str(resident.get("resident_ready_path"))).resolve()
        == Path(roots["resident_full_root"]) / "READY.json",
        Path(str(resident.get("mirror_store_root"))).resolve()
        == Path(roots["resident_full_root"]) / "store",
        resident.get("provenance_digest") == expected_source_pack["provenance_digest"],
        resident.get("resident_content_digest")
        == _expected_resident_content_digest(expected_source_pack),
        stored_ready.get("schema_version") == READY_SCHEMA_VERSION,
        ready_payload.get("schema_version") == READY_SCHEMA_VERSION,
        decoded_ready == ready_payload,
        sha256(ready_bytes).hexdigest() == stored_ready.get("ready_file_sha256"),
        ready_payload.get("ready_digest") == stored_ready.get("ready_digest"),
        ready_payload.get("ready_digest") == stable_hash({
            key: value for key, value in ready_payload.items() if key != "ready_digest"
        }),
        ready_payload.get("content_digest") == stable_hash(ready_payload.get("content")),
        ready_payload.get("seal_digest") == stable_hash(ready_payload.get("seal")),
        stored_ready.get("content_digest")
        == resident.get("resident_content_digest"),
        ready_payload.get("content_digest") == resident.get("resident_content_digest"),
        stored_lease.get("content_digest")
        == resident.get("resident_content_digest"),
        stored_lease.get("ready_digest") == stored_ready.get("ready_digest"),
        stored_lease.get("ready_file_sha256")
        == stored_ready.get("ready_file_sha256"),
        stored_lease.get("lease_digest") == stable_hash(lease_deterministic),
        validation.get("content_digest")
        == resident.get("resident_content_digest"),
        validation.get("ready_digest") == stored_ready.get("ready_digest"),
        validation.get("seal_digest") == stored_ready.get("seal_digest"),
        validation.get("reserve_bytes")
        == contract["resident_policy"]["reserve_bytes"],
        validation.get("observation_digest")
        == stable_hash(validation_deterministic),
        resident.get("query_specific_inputs_used") is False,
        resident.get("outcomes_or_labels_used") is False,
        resident.get("real_forward_outcomes_accessed") is False,
        resident.get("binding_digest") == stable_hash(resident_deterministic),
    )):
        raise ValueError("resident binding differs")
    if expected_ready_observation is not None and resident.get("resident_ready_observation") != dict(expected_ready_observation):
        raise ValueError("resident READY observation differs at comparison boundary")
    execution = execution_query_ids(registry)
    case_by_id = {str(row["episode_id"]): row for row in registry["cases_data"]}
    role_by_id = {str(row["query_episode_id"]): row for row in contract["role_table"]}
    _assert_exact_candidate_tree(candidate_root, execution)
    paths = sorted((candidate_root / "case-bundles").glob("*.json"))
    expected_paths = [
        candidate_root / "case-bundles" / f"{ordinal:03d}-{query_id}.json"
        for ordinal, query_id in enumerate(execution)
    ]
    if paths != sorted(expected_paths):
        raise ValueError("exact 60 case-bundle path set differs")
    bundles: list[dict[str, Any]] = []
    semantics: dict[str, dict[str, Any]] = {}
    performance: dict[str, dict[str, Any]] = {}
    for ordinal, (query_id, path) in enumerate(zip(execution, expected_paths, strict=True)):
        bundle = _read_json(path)
        semantic, attempt = _validate_bundle(
            bundle, contract=contract, case=case_by_id[query_id],
            role=role_by_id[query_id], expected_query=expected_query_contexts[query_id],
            resident=resident, physical_rows=int(expected_source_pack["physical_rows"]),
            execution_ordinal=ordinal,
        )
        bundles.append(bundle)
        semantics[query_id] = semantic
        performance[query_id] = attempt
    events, head = _load_ledger(candidate_root, str(contract["contract_digest"]))
    expected_event_types = ["run_started"] + [
        event_type for _ in execution for event_type in ("case_started", "case_completed")
    ] + ["run_complete"]
    if [row.get("event_type") for row in events] != expected_event_types:
        raise ValueError("ledger event sequence differs")
    run_details = dict(events[0].get("details", {}))
    git_binding = dict(run_details.get("git_binding", {}))
    git_deterministic = {
        key: value for key, value in git_binding.items() if key != "binding_digest"
    }
    prereg_path = expected_preregistration_path(repository_root)
    expected_run_details = {
        "preregistration_digest": preregistration["preregistration_digest"],
        "git_binding": git_binding,
        "resident_binding_digest": resident["binding_digest"],
        "resident_content_digest": resident["resident_content_digest"],
        "execution_query_ids_digest": contract["execution_query_ids_digest"],
    }
    if not all((
        run_details == expected_run_details,
        set(git_binding) == {
            "repository_root", "head_commit", "preregistration_relative_path",
            "preregistration_blob_sha256", "tracked_worktree_clean",
            "index_clean", "binding_digest",
        },
        git_binding.get("repository_root") == str(repository_root.resolve()),
        git_binding.get("preregistration_relative_path")
        == str(prereg_path.relative_to(repository_root.resolve())),
        git_binding.get("preregistration_blob_sha256")
        == file_fingerprint(prereg_path),
        git_binding.get("tracked_worktree_clean") is True,
        git_binding.get("index_clean") is True,
        isinstance(git_binding.get("head_commit"), str)
        and len(git_binding["head_commit"]) == 40,
        git_binding.get("binding_digest") == stable_hash(git_deterministic),
    )):
        raise ValueError("ledger run-start preregistration binding differs")
    for ordinal, query_id in enumerate(execution):
        case = case_by_id[query_id]
        role = role_by_id[query_id]
        expected_started = {
            "execution_ordinal": ordinal,
            "registry_case_id": case["case_id"],
            "query_episode_id": query_id,
            "performance_role": role["performance_role"],
            "implementation_manifest_digest": contract[
                "implementation_manifest"
            ]["digest"],
            "resident_ready_digest": resident["resident_ready_observation"][
                "ready_digest"
            ],
            "resident_content_digest": resident["resident_content_digest"],
        }
        if events[1 + ordinal * 2].get("details") != expected_started:
            raise ValueError("ledger case-start binding differs")
    completed_events = [row for row in events if row.get("event_type") == "case_completed"]
    for ordinal, (event, bundle) in enumerate(zip(completed_events, bundles, strict=True)):
        semantic = bundle["semantic"]
        attempt = bundle["performance"]
        expected_details = {
            "execution_ordinal": ordinal,
            "registry_case_id": semantic["registry_case_id"],
            "query_episode_id": semantic["query_episode_id"],
            "performance_role": semantic["performance_role"],
            "bundle_digest": bundle["bundle_digest"],
            "semantic_digest": semantic["semantic_digest"],
            "performance_digest": attempt["performance_digest"],
            "semantic_passed": True,
            "performance_passed": attempt["passed"],
        }
        if event.get("details") != expected_details:
            raise ValueError("ledger case completion differs from bundle")

    ordered_ids = [str(case["episode_id"]) for case in registry["cases_data"]]
    semantic_matrix = _read_json(candidate_root / "semantic-matrix.json")
    _exact_keys(semantic_matrix, {
        "schema_version", "producer_contract_digest",
        "resident_start_content_digest", "resident_end_content_digest",
        "ordered_query_episode_ids", "semantic_case_digests", "gates",
        "passed", "real_forward_outcomes_accessed", "elapsed_seconds",
        "created_at", "result_digest",
    }, "semantic matrix")
    _validate_timestamp(semantic_matrix["created_at"], "semantic matrix")
    semantic_elapsed = semantic_matrix["elapsed_seconds"]
    if not (
        type(semantic_elapsed) in (int, float)
        and isfinite(float(semantic_elapsed)) and semantic_elapsed >= 0
    ):
        raise ValueError("semantic matrix elapsed time differs")
    semantic_gates = {
        "all_60_semantic_cases_present_in_registry_order": True,
        "all_semantic_case_gates_passed": all(semantics[value]["passed"] is True for value in ordered_ids),
        "mirror_content_unchanged_before_and_after": semantic_matrix.get("resident_start_content_digest") == semantic_matrix.get("resident_end_content_digest") == resident["resident_content_digest"],
        "ledger_semantic_completion_matches_cases": [row["details"]["semantic_digest"] for row in completed_events] == [semantics[value]["semantic_digest"] for value in execution],
        "real_forward_outcomes_excluded": all(semantics[value]["real_forward_outcomes_accessed"] is False for value in ordered_ids),
    }
    if tuple(semantic_gates) != SEMANTIC_MATRIX_GATES or not all((
        semantic_matrix.get("schema_version") == SEMANTIC_MATRIX_SCHEMA,
        semantic_matrix.get("producer_contract_digest") == contract["contract_digest"],
        semantic_matrix.get("ordered_query_episode_ids") == ordered_ids,
        semantic_matrix.get("semantic_case_digests") == [semantics[value]["semantic_digest"] for value in ordered_ids],
        semantic_matrix.get("gates") == semantic_gates,
        semantic_matrix.get("passed") is True,
        semantic_matrix.get("real_forward_outcomes_accessed") is False,
        semantic_matrix.get("result_digest") == semantic_matrix_digest(semantic_matrix),
    )):
        raise ValueError("semantic matrix differs")
    semantic_seal = _read_json(candidate_root / "SEMANTIC_SEALED.json")
    _validate_semantic_seal(
        semantic_matrix, semantic_seal, str(contract["contract_digest"]),
        str(resident["resident_content_digest"]),
    )

    attempts = [performance[value] for value in ordered_ids]
    roles = {row["query_episode_id"]: row["performance_role"] for row in contract["role_table"]}
    confirmatory = [row for row in attempts if roles[row["query_episode_id"]] == "confirmatory_untouched"]
    exposed = [row for row in attempts if roles[row["query_episode_id"]] == "exposed_recovery_regression"]
    ready_pairs = {(stable_hash(row["ready_start"]), stable_hash(row["ready_end"])) for row in attempts}
    performance_gates = {
        "exact_7_exposed_53_confirmatory_role_partition": len(exposed) == 7 and len(confirmatory) == 53,
        "all_60_primary_attempts_accounted": len(attempts) == 60 and all(row["attempt_ordinal"] == 1 for row in attempts),
        "all_53_confirmatory_primary_attempts_passed": all(row["passed"] is True for row in confirmatory),
        "all_7_exposed_regression_attempts_passed": all(row["passed"] is True for row in exposed),
        "all_60_operational_limits_passed": all(row["passed"] is True for row in attempts),
        "single_ready_instance_for_all_primary_attempts": len(ready_pairs) == 1 and next(iter(ready_pairs))[0] == next(iter(ready_pairs))[1],
    }
    performance_matrix = _read_json(candidate_root / "performance-matrix.json")
    _exact_keys(performance_matrix, {
        "schema_version", "producer_contract_digest",
        "ordered_query_episode_ids", "attempt_digests",
        "confirmatory_query_episode_ids", "exposed_query_episode_ids",
        "gates", "passed", "claims_policy",
        "real_forward_outcomes_accessed", "elapsed_seconds", "created_at",
        "result_digest",
    }, "performance matrix")
    _validate_timestamp(performance_matrix["created_at"], "performance matrix")
    performance_elapsed = performance_matrix["elapsed_seconds"]
    if not (
        type(performance_elapsed) in (int, float)
        and isfinite(float(performance_elapsed)) and performance_elapsed >= 0
    ):
        raise ValueError("performance matrix elapsed time differs")
    if tuple(performance_gates) != PERFORMANCE_MATRIX_GATES or not all((
        performance_matrix.get("schema_version") == PERFORMANCE_MATRIX_SCHEMA,
        performance_matrix.get("producer_contract_digest") == contract["contract_digest"],
        performance_matrix.get("ordered_query_episode_ids") == ordered_ids,
        performance_matrix.get("attempt_digests") == [performance[value]["performance_digest"] for value in ordered_ids],
        performance_matrix.get("confirmatory_query_episode_ids") == [row["query_episode_id"] for row in confirmatory],
        performance_matrix.get("exposed_query_episode_ids") == [row["query_episode_id"] for row in exposed],
        performance_matrix.get("gates") == performance_gates,
        performance_matrix.get("passed") is all(performance_gates.values()),
        performance_matrix.get("claims_policy") == CLAIMS_POLICY,
        performance_matrix.get("real_forward_outcomes_accessed") is False,
        performance_matrix.get("result_digest") == performance_matrix_digest(performance_matrix),
    )):
        raise ValueError("performance matrix differs")
    performance_final = _read_json(candidate_root / "PERFORMANCE_FINAL.json")
    _validate_terminal_performance(
        performance_matrix, performance_final, str(contract["contract_digest"]),
        str(resident["binding_digest"]),
    )
    run_complete = _read_json(candidate_root / "RUN_COMPLETE.json")
    _exact_keys(run_complete, {
        "schema_version", "producer_contract_digest",
        "resident_binding_digest", "semantic_seal_digest",
        "performance_final_digest", "ledger_last_event_digest",
        "ledger_head_digest", "semantic_passed", "performance_passed",
        "authority_results_opened", "production_promotion_authorized",
        "created_at", "complete_digest",
    }, "run complete")
    _validate_timestamp(run_complete["created_at"], "run complete")
    final_event = events[-1]
    expected_final_details = {
        "semantic_matrix_digest": semantic_matrix["result_digest"],
        "semantic_seal_digest": semantic_seal["seal_digest"],
        "performance_matrix_digest": performance_matrix["result_digest"],
        "performance_final_digest": performance_final["final_digest"],
        "performance_passed": performance_matrix["passed"],
    }
    if not all((
        final_event.get("event_type") == "run_complete",
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
        run_complete.get("complete_digest") == terminal_digest(run_complete, "complete_digest"),
    )):
        raise ValueError("run-complete evidence differs")
    return PreopenEvidence(
        dict(registry), contract, preregistration, resident, tuple(bundles),
        semantics, performance, semantic_matrix, semantic_seal,
        performance_matrix, performance_final, run_complete, head,
    )


def _authority_case_digest(payload: Mapping[str, Any]) -> str:
    return stable_hash({key: value for key, value in payload.items() if key not in AUTHORITY_CASE_OMITTED})


def _authority_checkpoint_digest(payload: Mapping[str, Any]) -> str:
    return stable_hash({key: value for key, value in payload.items() if key not in {"created_at", "checkpoint_integrity_digest"}})


def _authority_matrix_digest(payload: Mapping[str, Any]) -> str:
    return stable_hash({key: value for key, value in payload.items() if key not in AUTHORITY_MATRIX_OMITTED})


def _authority_measurement_digest(payload: Mapping[str, Any]) -> str:
    return stable_hash({
        "measurements": payload["measurements"],
        "p95_exact_seconds": payload["p95_exact_seconds"],
        "maximum_exact_seconds": payload["maximum_exact_seconds"],
        "total_exact_seconds": payload["total_exact_seconds"],
        "p95_search_seconds": payload["p95_search_seconds"],
        "maximum_search_seconds": payload["maximum_search_seconds"],
        "total_search_seconds": payload["total_search_seconds"],
        "maximum_worker_rss_mb": payload["maximum_worker_rss_mb"],
        "performance_gates": payload["performance_gates"],
        "performance_gate_passed": payload["performance_gate_passed"],
    })


def _authority_case_gates(
    payload: Mapping[str, Any], case: Mapping[str, Any],
) -> dict[str, bool]:
    matches = list(payload.get("matches", []))
    certificate = dict(payload.get("certificate", {}))
    ids = [str(row.get("episode_id")) for row in matches]
    ordering = [
        (float(row["total_distance"]), str(row["episode_id"]))
        for row in matches
    ]
    query_start = pd.Timestamp(payload["query_start"])
    latest_eligible = pd.Timestamp(payload["latest_eligible_cutoff"])
    next_bound = certificate.get("next_lower_bound")
    stopped_early = bool(certificate.get("stopped_early"))
    stopping = (
        stopped_early and next_bound is not None
        and float(next_bound) > float(certificate["stop_threshold"]) + 1e-12
    ) or (
        not stopped_early and int(certificate.get("safely_pruned", -1)) == 0
    )
    return {
        "twenty_matches": len(matches) == 20,
        "unique_episode_ids": len(ids) == len(set(ids)),
        "stable_distance_id_order": ordering == sorted(ordering),
        "quality_tiers_allowed": all(
            row.get("quality_tier") in {"A", "B"} for row in matches
        ),
        "per_instrument_cap": max((
            sum(str(row.get("symbol")) == symbol for row in matches)
            for symbol in {str(row.get("symbol")) for row in matches}
        ), default=0) <= 3,
        "same_symbol_overlap_excluded": all(
            str(row.get("symbol")) != str(case["symbol"])
            or pd.Timestamp(row.get("cutoff")) < query_start
            for row in matches
        ),
        "candidate_cutoffs_temporally_eligible": all(
            pd.Timestamp(row.get("cutoff")) <= latest_eligible for row in matches
        ),
        "certificate_query_equal": certificate.get("query_episode_id")
        == case["episode_id"],
        "candidate_accounting": (
            int(certificate.get("exact_evaluated", -1))
            + int(certificate.get("safely_pruned", -1))
            == int(certificate.get("eligible_candidates", -2))
        ),
        "strict_stop_or_exhaustion": stopping,
        "quantized_bound_safe": float(
            certificate.get("maximum_quantized_bound_excess", float("inf"))
        ) <= 1e-12,
        "certificate_digest_reconstructed": certificate.get("result_digest")
        == _certificate_digest({"certificate": certificate, "matches": matches}),
    }


def _validate_authority_case(
    payload: Mapping[str, Any], case: Mapping[str, Any],
    authority_contract: Mapping[str, Any], matrix_row: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    try:
        matches = list(payload["matches"])
        ids = [str(row["episode_id"]) for row in matches]
        certificate = dict(payload["certificate"])
        expected_gates = _authority_case_gates(payload, case)
        expected_gates["frontier_execution_policy"] = True
        if not all((
            payload.get("schema_version") == AUTHORITY_CASE_SCHEMA,
            payload.get("status") == "completed",
            payload.get("contract_digest") == authority_contract["contract_digest"],
            payload.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
            payload.get("generation_id") == FROZEN_GENERATION_ID,
            payload.get("registry_case_id") == case["case_id"],
            payload.get("query_episode_id") == case["episode_id"],
            payload.get("query_symbol") == case["symbol"],
            payload.get("query_cutoff") == case["cutoff"],
            payload.get("query_stock_prefix") == case["stock_prefix"],
            payload.get("query_benchmark_prefix") == case["benchmark_prefix"],
            len(matches) == 20, len(ids) == len(set(ids)),
            payload.get("gates") == expected_gates,
            payload.get("gate_passed") is True,
            all(expected_gates.values()),
            payload.get("certificate_digest") == certificate.get("result_digest"),
            certificate.get("result_digest") == _certificate_digest({"certificate": certificate, "matches": matches}),
            payload.get("result_digest") == _authority_case_digest(payload),
            payload.get("checkpoint_integrity_digest") == _authority_checkpoint_digest(payload),
            payload.get("real_forward_outcomes_accessed") is False,
            matrix_row.get("registry_case_id") == case["case_id"],
            matrix_row.get("query_episode_id") == case["episode_id"],
            matrix_row.get("authority_digest") == payload.get("result_digest"),
            matrix_row.get("certificate_digest") == payload.get("certificate_digest"),
        )):
            failures.append("sealed authority case differs")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        failures.append(f"malformed authority case:{type(exc).__name__}:{exc}")
    return failures


def _assert_open_marker(output_root: Path, expected_digest: str) -> None:
    marker = _read_json(output_root / "RESULTS_OPENED.json")
    deterministic = {key: value for key, value in marker.items() if key not in {"created_at", "result_digest"}}
    if not all((
        marker.get("schema_version") == RESULTS_OPENED_SCHEMA,
        marker.get("result_digest") == stable_hash(deterministic),
        marker.get("result_digest") == expected_digest,
    )):
        raise ValueError("durable results-opened marker differs")


def _fresh_comparison_root(root: Path) -> None:
    if (root / "SEALED.json").exists():
        raise ValueError("one-shot comparison is already sealed; reopen is forbidden")
    if (root / "RESULTS_OPENED.json").exists():
        raise ValueError("authority was previously opened without a seal; fail closed")
    if root.exists() and any(root.iterdir()):
        raise ValueError("comparison root is not fresh and empty")


def compare_once(
    preopen: PreopenEvidence, *, authority_root: Path, output_root: Path,
    artifact_dir: Path,
    authority_reader: Callable[[Path], dict[str, Any]] = _read_json,
) -> dict[str, Any]:
    """Cross the truth boundary once, compare identities, and seal pass or fail."""
    roots = expected_roots(artifact_dir)
    if authority_root.resolve() != Path(roots["authority_root"]):
        raise ValueError("authority root differs from exact frozen root")
    if output_root.resolve() != Path(roots["comparison_root"]):
        raise ValueError("comparison root differs from exact frozen root")
    _fresh_comparison_root(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    marker_deterministic = {
        "schema_version": RESULTS_OPENED_SCHEMA,
        "registry_digest": FROZEN_REGISTRY_DIGEST,
        "producer_contract_digest": preopen.contract["contract_digest"],
        "semantic_seal_digest": preopen.semantic_seal["seal_digest"],
        "performance_final_digest": preopen.performance_final["final_digest"],
        "run_complete_digest": preopen.run_complete["complete_digest"],
        "status": "authority results about to be opened exactly once",
    }
    marker = {
        **marker_deterministic,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result_digest": stable_hash(marker_deterministic),
    }
    _atomic_json(output_root / "RESULTS_OPENED.json", marker)
    _assert_open_marker(output_root, str(marker["result_digest"]))

    def read_authority(path: Path) -> dict[str, Any]:
        _assert_open_marker(output_root, str(marker["result_digest"]))
        return authority_reader(path)

    authority_contract = read_authority(authority_root / "authority-contract.json")
    authority_matrix = read_authority(authority_root / "authority-matrix.json")
    authority_seal = read_authority(authority_root / "SEALED.json")
    ordered_ids = [str(case["episode_id"]) for case in preopen.registry["cases_data"]]
    contract_deterministic = {
        key: value for key, value in authority_contract.items() if key != "contract_digest"
    }
    matrix_rows = list(authority_matrix.get("cases", []))
    if not all((
        authority_contract.get("schema_version") == AUTHORITY_CONTRACT_SCHEMA,
        authority_contract.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
        authority_contract.get("generation_id") == FROZEN_GENERATION_ID,
        authority_contract.get("expected_query_episode_ids") == ordered_ids,
        authority_contract.get("authority_root_policy") == "write-isolated truth; no candidate result input",
        _manifest_valid(authority_contract.get("implementation_manifest", {})),
        authority_contract.get("real_forward_outcomes_accessed") is False,
        authority_contract.get("contract_digest") == stable_hash(contract_deterministic),
        authority_matrix.get("schema_version") == AUTHORITY_MATRIX_SCHEMA,
        authority_matrix.get("contract_digest") == authority_contract.get("contract_digest"),
        authority_matrix.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
        authority_matrix.get("generation_id") == FROZEN_GENERATION_ID,
        authority_matrix.get("completed_cases") == 60,
        authority_matrix.get("invalid_cases") == [],
        len(matrix_rows) == 60,
        [row.get("query_episode_id") for row in matrix_rows] == ordered_ids,
        authority_matrix.get("gate_passed") is True,
        authority_matrix.get("measurement_integrity_digest")
        == _authority_measurement_digest(authority_matrix),
        authority_matrix.get("result_digest") == _authority_matrix_digest(authority_matrix),
        authority_seal.get("schema_version") == AUTHORITY_SEAL_SCHEMA,
        authority_seal.get("contract_digest") == authority_contract.get("contract_digest"),
        authority_seal.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
        authority_seal.get("authority_matrix_digest") == authority_matrix.get("result_digest"),
        authority_seal.get("measurement_integrity_digest")
        == authority_matrix.get("measurement_integrity_digest"),
        authority_seal.get("authority_cases") == 60,
        authority_seal.get("seal_scope") == "exact authority correctness only",
        authority_seal.get("authority_correctness_sealed") is True,
        authority_seal.get("performance_gate_passed")
        is authority_matrix.get("performance_gate_passed"),
        authority_seal.get("production_promotion_authorized") is False,
        authority_seal.get("candidate_results_opened") is False,
        authority_seal.get("real_forward_outcomes_accessed") is False,
        authority_seal.get("seal_digest") == stable_hash({key: value for key, value in authority_seal.items() if key != "seal_digest"}),
    )):
        raise ValueError("sealed v4 authority aggregate differs")
    matrix_by_id = {str(row["query_episode_id"]): row for row in matrix_rows}
    observed_case_paths = sorted((authority_root / "cases").glob("*.json"))
    expected_case_paths = sorted(
        authority_root / "cases" / f"{query_id}.json"
        for query_id in ordered_ids
    )
    if observed_case_paths != expected_case_paths:
        raise ValueError("sealed v4 authority case path set differs")
    comparison_cases: list[dict[str, Any]] = []
    failures: list[str] = []
    total_retained = 0
    for case in preopen.registry["cases_data"]:
        query_id = str(case["episode_id"])
        authority = read_authority(authority_root / "cases" / f"{query_id}.json")
        case_failures = _validate_authority_case(
            authority, case, authority_contract, matrix_by_id.get(query_id, {}),
        )
        candidate_ids = {
            str(row["episode_id"])
            for row in preopen.semantics_by_id[query_id]["candidates"]
        }
        truth_ids = [str(row["episode_id"]) for row in authority.get("matches", [])]
        retained = [value for value in truth_ids if value in candidate_ids]
        retained_count = len(retained)
        total_retained += retained_count
        if retained_count < int(COMPARISON_POLICY["minimum_retained_per_case"]):
            case_failures.append("candidate retained fewer than 19 of 20")
        comparison_cases.append({
            "registry_case_id": case["case_id"],
            "query_episode_id": query_id,
            "candidate_semantic_digest": preopen.semantics_by_id[query_id]["semantic_digest"],
            "authority_case_digest": authority.get("result_digest"),
            "candidate_count": len(candidate_ids),
            "retained_count": retained_count,
            "recall_at_20": retained_count / 20.0,
            "perfect_20_of_20": retained_count == 20,
            "missing_authority_episode_ids": [value for value in truth_ids if value not in candidate_ids],
            "failures": case_failures,
            "passed": not case_failures,
        })
        failures.extend(f"{case['case_id']}:{value}" for value in case_failures)
    gates = {
        "all_60_authority_cases_valid": len(comparison_cases) == 60 and not any(row["failures"] and any("authority" in value for value in row["failures"]) for row in comparison_cases),
        "every_case_retains_at_least_19_of_20": len(comparison_cases) == 60 and all(row["retained_count"] >= 19 for row in comparison_cases),
        "aggregate_retains_at_least_1188_of_1200": total_retained >= 1_188,
    }
    deterministic = {
        "schema_version": COMPARISON_MATRIX_SCHEMA,
        "registry_digest": FROZEN_REGISTRY_DIGEST,
        "producer_contract_digest": preopen.contract["contract_digest"],
        "semantic_seal_digest": preopen.semantic_seal["seal_digest"],
        "performance_final_digest": preopen.performance_final["final_digest"],
        "performance_passed": preopen.performance_final["performance_passed"],
        "results_opened_marker_digest": marker["result_digest"],
        "authority_contract_digest": authority_contract["contract_digest"],
        "authority_matrix_digest": authority_matrix["result_digest"],
        "authority_seal_digest": authority_seal["seal_digest"],
        "completed_cases": len(comparison_cases),
        "retained_total": total_retained,
        "retained_denominator": 1_200,
        "minimum_retained_count": min((row["retained_count"] for row in comparison_cases), default=None),
        "perfect_20_of_20_cases": sum(row["perfect_20_of_20"] for row in comparison_cases),
        "perfect_20_of_20_is_descriptive_only": True,
        "cases": comparison_cases,
        "failures": failures,
        "gates": gates,
        "passed": all(gates.values()),
        "candidate_results_opened": True,
        "production_promotion_authorized": False,
        "real_forward_outcomes_accessed": False,
    }
    comparison = {
        **deterministic, "created_at": datetime.now(timezone.utc).isoformat(),
        "result_digest": stable_hash(deterministic),
    }
    _atomic_json(output_root / "candidate-comparison.json", comparison)
    seal_deterministic = {
        "schema_version": COMPARISON_SEAL_SCHEMA,
        "registry_digest": FROZEN_REGISTRY_DIGEST,
        "producer_contract_digest": preopen.contract["contract_digest"],
        "comparison_digest": comparison["result_digest"],
        "results_opened_marker_digest": marker["result_digest"],
        "authority_seal_digest": authority_seal["seal_digest"],
        "candidate_results_opened": True,
        "comparison_gate_passed": comparison["passed"],
        "production_promotion_authorized": False,
    }
    seal = {
        **seal_deterministic, "created_at": datetime.now(timezone.utc).isoformat(),
        "seal_digest": stable_hash(seal_deterministic),
    }
    _atomic_json(output_root / "SEALED.json", seal)
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    roots = expected_roots(config.artifact_dir)
    observed = {
        "registry_root": str(args.registry.parent.resolve()),
        "source_full_root": roots["source_full_root"],
        "resident_full_root": roots["resident_full_root"],
        "candidate_root": str(args.candidate_root.resolve()),
        "comparison_root": str(args.output_root.resolve()),
        "verification_root": roots["verification_root"],
        "predecessor_candidate_root": roots["predecessor_candidate_root"],
        "predecessor_comparison_root": roots["predecessor_comparison_root"],
        "authority_root": str(args.authority_root.resolve()),
    }
    if observed != roots:
        raise ValueError("v2 comparator exact roots differ")
    registry = _read_json(args.registry)
    preregistration = _read_json(expected_preregistration_path(Path(__file__).resolve().parents[2]))
    frozen_contract = dict(preregistration["producer_contract"])
    source_pack = _source_pack_binding(Path(roots["source_full_root"]))
    implementation = _implementation_manifest_from_frozen(
        Path(__file__).resolve().parents[2], frozen_contract["implementation_manifest"],
    )
    environment = _environment_manifest()
    contexts = {
        str(case["episode_id"]): _expected_query_context(args.config, case)
        for case in registry["cases_data"]
    }
    preopen = validate_preopen_candidate(
        registry=registry, candidate_root=args.candidate_root,
        artifact_dir=config.artifact_dir,
        repository_root=Path(__file__).resolve().parents[2],
        expected_source_pack=source_pack,
        expected_implementation_manifest=implementation,
        expected_environment_manifest=environment,
        expected_query_contexts=contexts,
        expected_ready_observation=None,
    )
    comparison = compare_once(
        preopen, authority_root=args.authority_root,
        output_root=args.output_root, artifact_dir=config.artifact_dir,
    )
    print(json.dumps({key: value for key, value in comparison.items() if key != "cases"}, indent=2, sort_keys=True))
    return 0 if comparison["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
