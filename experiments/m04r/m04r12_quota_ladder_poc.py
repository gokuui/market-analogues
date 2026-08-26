"""Post-open, development-only route-quota ladder for M04R-11 failures.

This experiment is intentionally retrospective.  It may run only after the
terminal v2 comparison has opened the sealed authorities and failed its recall
gate.  It scans only the failed cases, never reads forward outcomes, and can
never authorize production promotion.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
from math import isfinite
import os
from pathlib import Path
import platform
import resource
import stat
import subprocess
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping
from uuid import uuid4

import pandas as pd

from market_analogues.adapters import source_from_spec
from market_analogues.config import load_config
from market_analogues.episodes import build_episode
from market_analogues.m04r_candidate_evidence import reconstructed_candidate_digest
import market_analogues.packed_bound_search as packed_search
from market_analogues.packed_bound_search import (
    DEFAULT_ROUTE_QUOTAS,
    PackedBoundQuery,
    scan_packed_bound_proposals_threaded,
)
from market_analogues.representation import represent
from market_analogues.resident_store import (
    observe_ready_strict,
    prepare_resident_mirror_observed,
    resident_file_identity_lease,
)
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import InstrumentKey, stable_hash


SCHEMA_VERSION = "m04r12-development-quota-ladder-v1"
CHECKPOINT_SCHEMA = "m04r12-development-quota-ladder-case-config-v1"
PREREGISTRATION_SCHEMA = "m04r12-development-quota-ladder-preregistration-v1"
PRODUCER_SEAL_SCHEMA = "m04r12-development-max-route-producer-seal-v1"
REGISTRY_SCHEMA = "m04r-untouched-authority-registry-v1"
PRODUCER_SCHEMA = "candidate-recall-resident-producer-contract-v1"
RESULTS_OPENED_SCHEMA = "candidate-recall-results-opened-v2"
COMPARISON_SCHEMA = "candidate-recall-comparison-matrix-v2"
COMPARISON_SEAL_SCHEMA = "candidate-recall-comparison-seal-v2"
AUTHORITY_CONTRACT_SCHEMA = "m04r11-authority-build-contract-v4"
AUTHORITY_CASE_SCHEMA = "m04r11-certified-authority-case-v4"
AUTHORITY_MATRIX_SCHEMA = "m04r11-certified-authority-matrix-v4"
AUTHORITY_SEAL_SCHEMA = "m04r11-authority-seal-v4"

FROZEN_REGISTRY_DIGEST = (
    "0a4da732f91375a091775cb04e6e77c8d136ade47d7f4d16508a2d9a6555361e"
)
FROZEN_GENERATION_ID = (
    "9fc6ae0ec4451133d8006897162f3443803fd30a3c44b78e80a476fbb18bb483"
)
FROZEN_PRODUCER_CONTRACT_DIGEST = (
    "aa29be3c4d8cdd31e310c64e85ec4692968b608a5fe2bf1248ba2d0825c9aac2"
)
FROZEN_RESULTS_OPENED_DIGEST = (
    "4664fa8cab2d5210ebeaf6a79561f22e9f7beab4f8d8b4302dfef9b6810bdf32"
)
FROZEN_AUTHORITY_CONTRACT_DIGEST = (
    "032162035fc069c794406a4f801f0f8469c4e51c7ea1f4c912c18f80722bbb6f"
)
FROZEN_AUTHORITY_MATRIX_DIGEST = (
    "ae60b7cda755c2deabc7bf9335701d34eea73073664582933eb5ba984e7f9b29"
)
FROZEN_AUTHORITY_SEAL_DIGEST = (
    "83501b4cf620607c4ee14cc837d1d5c242460dc867aca22a51b3fbbf924be2c9"
)
FROZEN_COMPARISON_DIGEST = (
    "cfda7ef0e785667ce8bc3c566dbd210f5fa008ca35cf9e2fdece0c1330d7b670"
)
FROZEN_COMPARISON_SEAL_DIGEST = (
    "942f045cc5aab617e5a701a956df517d7c17cdf0fdc7ce8832484d0f7fa6c7bc"
)
FROZEN_QUERY_IDS = (
    "3307023dbe2164d025e788da", "3618af07dedd52fb3bdb1ccd",
    "9d7365581643bd93e85beb67", "99a0838725a09570b4a075ff",
)
PREREGISTRATION_RELATIVE_PATH = Path(
    "experiments/m04r/m04r12_quota_ladder_preregistered.json"
)
PERFORMANCE_LIMITS = {
    "forward_first_seconds": 120.0, "reverse_warm_seconds": 60.0,
    "peak_rss_mb": 1_536.0,
}
MAX_SIDECAR_BYTES_PER_CASE = 32 * 1024 * 1024
MAX_OUTPUT_BYTES = 256 * 1024 * 1024

COMPONENT_ROUTES = tuple(
    route for route in DEFAULT_ROUTE_QUOTAS if route != "composite"
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


class QuotaLadderError(ValueError):
    """The post-open development contract was not satisfied."""


@dataclass(frozen=True)
class FailedCase:
    ordinal: int
    registry_case: dict[str, Any]
    comparison_case: dict[str, Any]
    authority_digest: str
    authority_top20: tuple[str, ...]
    baseline_candidate_digest: str
    baseline_candidate_ids: tuple[str, ...]
    baseline_result_digest: str

    @property
    def case_id(self) -> str:
        return str(self.registry_case["case_id"])

    @property
    def query_id(self) -> str:
        return str(self.registry_case["episode_id"])


@dataclass(frozen=True)
class OpenedInputs:
    registry_digest: str
    producer_contract_digest: str
    comparison_digest: str
    comparison_seal_digest: str
    results_opened_digest: str
    authority_seal_digest: str
    generation_id: str
    provenance_digest: str
    resident_reserve_bytes: int
    source_store_root: Path
    resident_root: Path
    failed_cases: tuple[FailedCase, ...]
    original_retained_total: int
    unaffected_retained_total: int


@dataclass(frozen=True)
class ProducerCase:
    ordinal: int
    registry_case: dict[str, Any]
    baseline_candidate_digest: str
    baseline_candidate_ids: tuple[str, ...]
    baseline_result_digest: str

    @property
    def case_id(self) -> str:
        return str(self.registry_case["case_id"])

    @property
    def query_id(self) -> str:
        return str(self.registry_case["episode_id"])


@dataclass(frozen=True)
class ProducerInputs:
    registry_digest: str
    producer_contract_digest: str
    poc_contract_digest: str
    generation_id: str
    provenance_digest: str
    resident_content_digest: str
    resident_ready_digest: str
    resident_reserve_bytes: int
    source_store_root: Path
    resident_root: Path
    cases: tuple[ProducerCase, ...]


def _without(payload: Mapping[str, Any], omitted: Iterable[str]) -> dict[str, Any]:
    omitted_set = set(omitted)
    return {key: value for key, value in payload.items() if key not in omitted_set}


def _read_json(path: Path) -> dict[str, Any]:
    """Read one plain JSON file without following a terminal symlink."""
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise QuotaLadderError(f"cannot open required JSON: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise QuotaLadderError(f"required JSON is not regular: {path}")
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns, value.st_mode,
    )
    if identity(before) != identity(after):
        raise QuotaLadderError(f"required JSON changed while read: {path}")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise QuotaLadderError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(b"".join(chunks), object_pairs_hook=object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QuotaLadderError(f"invalid JSON: {path}") from exc
    if type(value) is not dict:
        raise QuotaLadderError(f"required JSON object differs: {path}")
    return value


def _atomic_create(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish create-only JSON; an existing checkpoint is never replaced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            raw = (
                json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False)
                + "\n"
            ).encode()
            handle.write(raw)
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


def _timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise QuotaLadderError(f"{label} timestamp is absent")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise QuotaLadderError(f"{label} timestamp is malformed") from exc
    if parsed.tzinfo is None:
        raise QuotaLadderError(f"{label} timestamp lacks timezone")
    return parsed


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _validate_git_gate(
    repository: Path, preregistration: Mapping[str, Any],
    *, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    """Require every frozen code input to be tracked and clean in index/worktree."""
    manifest = preregistration.get("code_manifest", {})
    files = manifest.get("files", {}) if type(manifest) is dict else {}
    relative_paths = {
        str(PREREGISTRATION_RELATIVE_PATH),
        str(preregistration.get("script_relative_path", "")),
        *files.keys(),
    }
    if "" in relative_paths or any(
        Path(value).is_absolute() or ".." in Path(value).parts
        for value in relative_paths
    ):
        raise QuotaLadderError("M04R-12 Git-bound path differs")
    paths = sorted(relative_paths)
    commands = (
        ["git", "ls-files", "--error-unmatch", "--", *paths],
        ["git", "diff", "--quiet"],
        ["git", "diff", "--cached", "--quiet"],
    )
    try:
        results = [
            runner(command, cwd=repository, text=True, capture_output=True, check=False)
            for command in commands
        ]
    except (OSError, subprocess.SubprocessError) as exc:
        raise QuotaLadderError("M04R-12 Git gate could not run") from exc
    if any(result.returncode != 0 for result in results):
        raise QuotaLadderError("M04R-12 frozen files are untracked or dirty")


def _environment_binding() -> dict[str, Any]:
    deterministic = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "numba")
        },
    }
    return {**deterministic, "digest": stable_hash(deterministic)}


def _validate_preregistration(
    repository: Path, *, config_path: Path, registry_root: Path,
    candidate_root: Path, source_full_root: Path, resident_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    path = repository / PREREGISTRATION_RELATIVE_PATH
    payload = _read_json(path)
    keys = {
        "schema_version", "status", "post_open_development_poc",
        "truth_inputs_allowed_in_producer", "production_promotion_authorized",
        "real_forward_outcomes_accessed", "repository_root", "script_relative_path",
        "script_sha256", "config_path", "config_sha256", "roots",
        "registry_digest", "producer_contract_digest", "generation_id",
        "provenance_digest", "resident_content_digest", "resident_ready_digest",
        "query_episode_ids", "query_order_digest", "baseline_bindings",
        "quota_configurations_digest", "maximum_route_quotas",
        "scan_protocol", "performance_limits", "output_limits", "environment",
        "code_manifest", "contract_digest",
    }
    deterministic = _without(payload, {"contract_digest"})
    observed_roots = {
        "registry_root": str(registry_root.resolve()),
        "candidate_root": str(candidate_root.resolve()),
        "source_full_root": str(source_full_root.resolve()),
        "resident_root": str(resident_root.resolve()),
        "output_root": str(output_root.resolve()),
    }
    script = repository / "experiments/m04r/m04r12_quota_ladder_poc.py"
    manifest = payload.get("code_manifest", {})
    files = manifest.get("files", {}) if type(manifest) is dict else {}
    manifest_valid = (
        type(files) is dict and bool(files)
        and manifest.get("digest") == stable_hash(files)
        and all(
            isinstance(relative, str)
            and not Path(relative).is_absolute() and ".." not in Path(relative).parts
            and (repository / relative).is_file()
            and _file_sha256(repository / relative) == digest
            for relative, digest in files.items()
        )
    )
    if not all((
        set(payload) == keys,
        payload.get("schema_version") == PREREGISTRATION_SCHEMA,
        payload.get("status") == "frozen_before_m04r12_producer_launch",
        payload.get("post_open_development_poc") is True,
        payload.get("truth_inputs_allowed_in_producer") is False,
        payload.get("production_promotion_authorized") is False,
        payload.get("real_forward_outcomes_accessed") is False,
        payload.get("repository_root") == str(repository.resolve()),
        payload.get("script_relative_path")
        == "experiments/m04r/m04r12_quota_ladder_poc.py",
        payload.get("script_sha256") == _file_sha256(script),
        payload.get("config_path") == str(config_path.resolve()),
        payload.get("config_sha256") == _file_sha256(config_path),
        payload.get("roots") == observed_roots,
        payload.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
        payload.get("producer_contract_digest") == FROZEN_PRODUCER_CONTRACT_DIGEST,
        payload.get("generation_id") == FROZEN_GENERATION_ID,
        payload.get("query_episode_ids") == list(FROZEN_QUERY_IDS),
        payload.get("query_order_digest") == stable_hash(list(FROZEN_QUERY_IDS)),
        payload.get("quota_configurations_digest")
        == stable_hash(list(quota_configurations())),
        payload.get("maximum_route_quotas") == MAX_ROUTE_QUOTAS,
        payload.get("scan_protocol") == {
            "forward": {"block_rows": 4_096, "block_order": "forward"},
            "reverse": {"block_rows": 4_097, "block_order": "reverse"},
            "threads": 8, "verify_content": False,
            "physical_scans_per_case": 2,
        },
        payload.get("performance_limits") == PERFORMANCE_LIMITS,
        payload.get("output_limits") == {
            "maximum_sidecar_bytes_per_case": MAX_SIDECAR_BYTES_PER_CASE,
            "maximum_total_output_bytes": MAX_OUTPUT_BYTES,
        },
        payload.get("environment") == _environment_binding(),
        manifest_valid,
        payload.get("contract_digest") == stable_hash(deterministic),
    )):
        raise QuotaLadderError("M04R-12 preregistration differs")
    _validate_git_gate(repository, payload)
    return payload


def quota_configurations() -> tuple[dict[str, Any], ...]:
    """Return the frozen, ordered ladder; every route is always explicit."""
    values = (
        ("baseline", 1_000, 64),
        ("composite_2000", 2_000, 64),
        ("composite_4000", 4_000, 64),
        ("composite_8000", 8_000, 64),
        ("components_128", 1_000, 128),
        ("components_256", 1_000, 256),
        ("components_512", 1_000, 512),
        ("composite_2000_components_256", 2_000, 256),
        ("composite_4000_components_512", 4_000, 512),
        ("composite_8000_components_512", 8_000, 512),
    )
    values = (*values, *(
        (f"single_{route}_512", 1_000, route)
        for route in COMPONENT_ROUTES
    ))
    result = []
    for ordinal, (name, composite, component) in enumerate(values):
        quotas = {"composite": composite}
        if isinstance(component, str):
            quotas.update({
                route: 512 if route == component else 64
                for route in COMPONENT_ROUTES
            })
        else:
            quotas.update({route: component for route in COMPONENT_ROUTES})
        result.append({"ordinal": ordinal, "name": name, "route_quotas": quotas})
    return tuple(result)


def _validate_plain_tree(
    root: Path, expected_files: set[str], expected_directories: set[str], label: str,
) -> None:
    if root.is_symlink() or not root.is_dir():
        raise QuotaLadderError(f"{label} root is absent or linked")
    files: set[str] = set()
    directories: set[str] = set()
    for path in root.rglob("*"):
        relative = str(path.relative_to(root))
        observed = path.lstat()
        if stat.S_ISLNK(observed.st_mode):
            raise QuotaLadderError(f"{label} tree contains a symlink")
        if stat.S_ISDIR(observed.st_mode):
            directories.add(relative)
        elif stat.S_ISREG(observed.st_mode):
            files.add(relative)
        else:
            raise QuotaLadderError(f"{label} tree contains a special entry")
    if files != expected_files or directories != expected_directories:
        raise QuotaLadderError(f"{label} terminal tree differs")


def _validate_registry(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not all((
        registry.get("schema_version") == REGISTRY_SCHEMA,
        registry.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
        registry.get("passed") is True,
        registry.get("failures") == [],
        registry.get("real_forward_outcomes_accessed") is False,
        isinstance(registry.get("cases_data"), list),
        len(registry.get("cases_data", [])) == 60,
    )):
        raise QuotaLadderError("frozen registry prerequisite differs")
    deterministic = _without(
        registry,
        {"passed", "failures", "registry_digest", "cases_data", "contamination_ledger"},
    )
    expected = stable_hash({
        **deterministic,
        "cases_data": registry["cases_data"],
        "contamination_ledger": registry["contamination_ledger"],
    })
    if expected != registry["registry_digest"]:
        raise QuotaLadderError("frozen registry digest differs")
    ids = [str(case.get("episode_id")) for case in registry["cases_data"]]
    if len(ids) != len(set(ids)):
        raise QuotaLadderError("frozen registry query IDs are not unique")
    return [dict(case) for case in registry["cases_data"]]


def _validate_marker(marker: Mapping[str, Any]) -> None:
    keys = {
        "schema_version", "registry_digest", "producer_contract_digest",
        "semantic_seal_digest", "performance_final_digest", "run_complete_digest",
        "status", "created_at", "result_digest",
    }
    deterministic = _without(marker, {"created_at", "result_digest"})
    if not all((
        set(marker) == keys,
        marker.get("schema_version") == RESULTS_OPENED_SCHEMA,
        marker.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
        marker.get("producer_contract_digest") == FROZEN_PRODUCER_CONTRACT_DIGEST,
        marker.get("status") == "authority results about to be opened exactly once",
        marker.get("result_digest") == stable_hash(deterministic),
        marker.get("result_digest") == FROZEN_RESULTS_OPENED_DIGEST,
    )):
        raise QuotaLadderError("terminal RESULTS_OPENED prerequisite differs")
    _timestamp(marker.get("created_at"), "RESULTS_OPENED")


def _validate_comparison(
    comparison: Mapping[str, Any], seal: Mapping[str, Any], marker: Mapping[str, Any],
) -> list[dict[str, Any]]:
    case_keys = {
        "registry_case_id", "query_episode_id", "candidate_semantic_digest",
        "authority_case_digest", "candidate_count", "retained_count", "recall_at_20",
        "perfect_20_of_20", "missing_authority_episode_ids", "failures", "passed",
    }
    comparison_keys = {
        "schema_version", "registry_digest", "producer_contract_digest",
        "semantic_seal_digest", "performance_final_digest", "performance_passed",
        "results_opened_marker_digest", "authority_contract_digest",
        "authority_matrix_digest", "authority_seal_digest", "completed_cases",
        "retained_total", "retained_denominator", "minimum_retained_count",
        "perfect_20_of_20_cases", "perfect_20_of_20_is_descriptive_only",
        "cases", "failures", "gates", "passed", "candidate_results_opened",
        "production_promotion_authorized", "real_forward_outcomes_accessed",
        "created_at", "result_digest",
    }
    seal_keys = {
        "schema_version", "registry_digest", "producer_contract_digest",
        "comparison_digest", "results_opened_marker_digest",
        "authority_seal_digest", "candidate_results_opened",
        "comparison_gate_passed", "production_promotion_authorized",
        "created_at", "seal_digest",
    }
    deterministic = _without(comparison, {"created_at", "result_digest"})
    seal_deterministic = _without(seal, {"created_at", "seal_digest"})
    cases = comparison.get("cases")
    if not all((
        set(comparison) == comparison_keys,
        comparison.get("schema_version") == COMPARISON_SCHEMA,
        comparison.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
        comparison.get("producer_contract_digest") == FROZEN_PRODUCER_CONTRACT_DIGEST,
        comparison.get("results_opened_marker_digest") == marker.get("result_digest"),
        comparison.get("authority_contract_digest") == FROZEN_AUTHORITY_CONTRACT_DIGEST,
        comparison.get("authority_matrix_digest") == FROZEN_AUTHORITY_MATRIX_DIGEST,
        comparison.get("authority_seal_digest") == FROZEN_AUTHORITY_SEAL_DIGEST,
        comparison.get("candidate_results_opened") is True,
        comparison.get("passed") is False,
        comparison.get("production_promotion_authorized") is False,
        comparison.get("real_forward_outcomes_accessed") is False,
        comparison.get("result_digest") == stable_hash(deterministic),
        comparison.get("result_digest") == FROZEN_COMPARISON_DIGEST,
        isinstance(cases, list), len(cases or []) == 60,
        all(type(row) is dict and set(row) == case_keys for row in (cases or [])),
        set(seal) == seal_keys,
        seal.get("schema_version") == COMPARISON_SEAL_SCHEMA,
        seal.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
        seal.get("comparison_digest") == comparison.get("result_digest"),
        seal.get("results_opened_marker_digest") == marker.get("result_digest"),
        seal.get("authority_seal_digest") == FROZEN_AUTHORITY_SEAL_DIGEST,
        seal.get("candidate_results_opened") is True,
        seal.get("comparison_gate_passed") is False,
        seal.get("production_promotion_authorized") is False,
        seal.get("seal_digest") == stable_hash(seal_deterministic),
        seal.get("seal_digest") == FROZEN_COMPARISON_SEAL_DIGEST,
    )):
        raise QuotaLadderError("terminal failed comparison prerequisite differs")
    marker_time = _timestamp(marker.get("created_at"), "RESULTS_OPENED")
    comparison_time = _timestamp(comparison.get("created_at"), "comparison")
    seal_time = _timestamp(seal.get("created_at"), "comparison seal")
    if not marker_time <= comparison_time <= seal_time:
        raise QuotaLadderError("terminal comparison publication order differs")
    failed = [dict(row) for row in cases if row.get("passed") is False]
    if len(failed) != 4 or any(
        type(row.get("retained_count")) is not int
        or not 0 <= row["retained_count"] < 19
        or row.get("failures") != ["candidate retained fewer than 19 of 20"]
        for row in failed
    ):
        raise QuotaLadderError("comparison does not contain exactly four recall failures")
    if any(row.get("passed") is not True or row.get("retained_count", -1) < 19
           for row in cases if row not in failed):
        raise QuotaLadderError("comparison nonfailed case accounting differs")
    if comparison.get("retained_total") != sum(row["retained_count"] for row in cases):
        raise QuotaLadderError("comparison retained aggregate differs")
    return failed


def _validate_candidate_terminal(
    candidate_root: Path, contract: Mapping[str, Any], marker: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    if not all((
        contract.get("schema_version") == PRODUCER_SCHEMA,
        contract.get("contract_digest") == FROZEN_PRODUCER_CONTRACT_DIGEST,
        contract.get("contract_digest") == stable_hash(_without(contract, {"contract_digest"})),
        contract.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
        contract.get("generation_id") == FROZEN_GENERATION_ID,
        contract.get("real_forward_outcomes_accessed") is False,
        isinstance(contract.get("execution_query_ids"), list),
        len(contract.get("execution_query_ids", [])) == 60,
    )):
        raise QuotaLadderError("terminal producer contract differs")
    semantic_seal = _read_json(candidate_root / "SEMANTIC_SEALED.json")
    performance_final = _read_json(candidate_root / "PERFORMANCE_FINAL.json")
    run_complete = _read_json(candidate_root / "RUN_COMPLETE.json")
    if not all((
        semantic_seal.get("seal_digest") == marker.get("semantic_seal_digest"),
        semantic_seal.get("seal_digest")
        == stable_hash(_without(semantic_seal, {"created_at", "seal_digest"})),
        semantic_seal.get("semantic_recall_ready") is True,
        semantic_seal.get("authority_results_opened") is False,
        semantic_seal.get("production_promotion_authorized") is False,
        performance_final.get("final_digest") == marker.get("performance_final_digest"),
        performance_final.get("final_digest")
        == stable_hash(_without(performance_final, {"created_at", "final_digest"})),
        performance_final.get("performance_terminal") is True,
        performance_final.get("authority_results_opened") is False,
        performance_final.get("production_promotion_authorized") is False,
        run_complete.get("complete_digest") == marker.get("run_complete_digest"),
        run_complete.get("complete_digest")
        == stable_hash(_without(run_complete, {"created_at", "complete_digest"})),
        run_complete.get("semantic_passed") is True,
        run_complete.get("authority_results_opened") is False,
        run_complete.get("production_promotion_authorized") is False,
    )):
        raise QuotaLadderError("candidate terminal chain differs")
    bundles: dict[str, dict[str, Any]] = {}
    for ordinal, query_id in enumerate(contract["execution_query_ids"]):
        bundle = _read_json(
            candidate_root / "case-bundles" / f"{ordinal:03d}-{query_id}.json"
        )
        if not all((
            bundle.get("query_episode_id") == query_id,
            bundle.get("execution_ordinal") == ordinal,
            bundle.get("producer_contract_digest") == contract["contract_digest"],
            bundle.get("bundle_digest") == stable_hash(_without(bundle, {"bundle_digest"})),
            bundle.get("semantic", {}).get("real_forward_outcomes_accessed") is False,
        )):
            raise QuotaLadderError(f"candidate case bundle differs: {query_id}")
        bundles[str(query_id)] = bundle
    expected_files = {
        "candidate-contract.json", "RESIDENT_READY.json", "ledger/HEAD.json",
        "semantic-matrix.json", "SEMANTIC_SEALED.json", "performance-matrix.json",
        "PERFORMANCE_FINAL.json", "RUN_COMPLETE.json",
        *(f"case-bundles/{ordinal:03d}-{query_id}.json"
          for ordinal, query_id in enumerate(contract["execution_query_ids"])),
        *(f"ledger/events/{index:06d}.json" for index in range(122)),
    }
    _validate_plain_tree(
        candidate_root, expected_files,
        {"case-bundles", "ledger", "ledger/events"}, "candidate",
    )
    return bundles


def _validate_candidate_blind(
    candidate_root: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    """Validate candidate evidence without accepting or opening any truth root."""
    contract = _read_json(candidate_root / "candidate-contract.json")
    resident = _read_json(candidate_root / "RESIDENT_READY.json")
    if not all((
        contract.get("schema_version") == PRODUCER_SCHEMA,
        contract.get("contract_digest") == FROZEN_PRODUCER_CONTRACT_DIGEST,
        contract.get("contract_digest") == stable_hash(_without(contract, {"contract_digest"})),
        contract.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
        contract.get("generation_id") == FROZEN_GENERATION_ID,
        contract.get("real_forward_outcomes_accessed") is False,
        isinstance(contract.get("execution_query_ids"), list),
        len(contract.get("execution_query_ids", [])) == 60,
        resident.get("producer_contract_digest") == contract.get("contract_digest"),
        resident.get("generation_id") == FROZEN_GENERATION_ID,
        resident.get("query_specific_inputs_used") is False,
        resident.get("outcomes_or_labels_used") is False,
        resident.get("real_forward_outcomes_accessed") is False,
        resident.get("binding_digest")
        == stable_hash(_without(resident, {"binding_digest"})),
    )):
        raise QuotaLadderError("truth-blind candidate prerequisite differs")
    # Terminal candidate documents remain truth-blind and bind all 60 bundles.
    for name, digest_name, omitted in (
        ("SEMANTIC_SEALED.json", "seal_digest", {"created_at", "seal_digest"}),
        ("PERFORMANCE_FINAL.json", "final_digest", {"created_at", "final_digest"}),
        ("RUN_COMPLETE.json", "complete_digest", {"created_at", "complete_digest"}),
    ):
        payload = _read_json(candidate_root / name)
        if not all((
            payload.get(digest_name) == stable_hash(_without(payload, omitted)),
            payload.get("producer_contract_digest") == contract["contract_digest"],
            payload.get("authority_results_opened") is False,
            payload.get("production_promotion_authorized") is False,
        )):
            raise QuotaLadderError(f"truth-blind candidate terminal differs: {name}")
    bundles: dict[str, dict[str, Any]] = {}
    for ordinal, query_id in enumerate(contract["execution_query_ids"]):
        bundle = _read_json(
            candidate_root / "case-bundles" / f"{ordinal:03d}-{query_id}.json"
        )
        semantic = bundle.get("semantic", {})
        if not all((
            bundle.get("query_episode_id") == query_id,
            bundle.get("execution_ordinal") == ordinal,
            bundle.get("producer_contract_digest") == contract["contract_digest"],
            bundle.get("bundle_digest") == stable_hash(_without(bundle, {"bundle_digest"})),
            semantic.get("query_episode_id") == query_id,
            semantic.get("real_forward_outcomes_accessed") is False,
            semantic.get("passed") is True,
            semantic.get("candidate_digest_reconstructed")
            == reconstructed_candidate_digest(semantic.get("candidates", [])),
        )):
            raise QuotaLadderError(f"truth-blind candidate bundle differs: {query_id}")
        bundles[str(query_id)] = bundle
    expected_files = {
        "candidate-contract.json", "RESIDENT_READY.json", "ledger/HEAD.json",
        "semantic-matrix.json", "SEMANTIC_SEALED.json", "performance-matrix.json",
        "PERFORMANCE_FINAL.json", "RUN_COMPLETE.json",
        *(f"case-bundles/{ordinal:03d}-{query_id}.json"
          for ordinal, query_id in enumerate(contract["execution_query_ids"])),
        *(f"ledger/events/{index:06d}.json" for index in range(122)),
    }
    _validate_plain_tree(
        candidate_root, expected_files,
        {"case-bundles", "ledger", "ledger/events"}, "candidate",
    )
    return contract, bundles, resident


def validate_producer_inputs(
    *, repository: Path, config_path: Path, registry_root: Path,
    candidate_root: Path, source_full_root: Path, resident_root: Path,
    output_root: Path,
) -> tuple[ProducerInputs, dict[str, Any]]:
    """Build producer scope without any comparison or authority parameter."""
    prereg = _validate_preregistration(
        repository, config_path=config_path, registry_root=registry_root,
        candidate_root=candidate_root, source_full_root=source_full_root,
        resident_root=resident_root, output_root=output_root,
    )
    registry = _read_json(registry_root / "query-registry.json")
    registry_cases = _validate_registry(registry)
    contract, bundles, resident = _validate_candidate_blind(candidate_root)
    by_query = {str(case["episode_id"]): case for case in registry_cases}
    bindings = prereg.get("baseline_bindings")
    if not isinstance(bindings, list) or len(bindings) != 4:
        raise QuotaLadderError("preregistered baseline bindings differ")
    cases = []
    for ordinal, (query_id, binding) in enumerate(zip(FROZEN_QUERY_IDS, bindings, strict=True)):
        case = by_query.get(query_id)
        bundle = bundles.get(query_id)
        if case is None or bundle is None or type(binding) is not dict:
            raise QuotaLadderError("preregistered producer case is absent")
        semantic = bundle["semantic"]
        candidates = semantic["candidates"]
        candidate_ids = tuple(str(row["episode_id"]) for row in candidates)
        expected_binding = {
            "ordinal": ordinal, "registry_case_id": case["case_id"],
            "query_episode_id": query_id,
            "semantic_digest": semantic["semantic_digest"],
            "candidate_digest": semantic["candidate_digest_reconstructed"],
            "candidate_count": len(candidate_ids),
            "candidate_ids_digest": stable_hash(list(candidate_ids)),
            "baseline_result_digest": semantic["scan_semantics"][0]["result_digest"],
        }
        if binding != expected_binding:
            raise QuotaLadderError("preregistered candidate baseline differs")
        cases.append(ProducerCase(
            ordinal, dict(case), str(binding["candidate_digest"]), candidate_ids,
            str(binding["baseline_result_digest"]),
        ))
    if not all((
        prereg.get("provenance_digest") == contract["source_pack"]["provenance_digest"],
        prereg.get("resident_content_digest") == resident["resident_content_digest"],
        prereg.get("resident_ready_digest")
        == resident["resident_ready_observation"]["ready_digest"],
    )):
        raise QuotaLadderError("preregistered resident/source binding differs")
    inputs = ProducerInputs(
        registry_digest=FROZEN_REGISTRY_DIGEST,
        producer_contract_digest=FROZEN_PRODUCER_CONTRACT_DIGEST,
        poc_contract_digest=str(prereg["contract_digest"]),
        generation_id=FROZEN_GENERATION_ID,
        provenance_digest=str(prereg["provenance_digest"]),
        resident_content_digest=str(prereg["resident_content_digest"]),
        resident_ready_digest=str(prereg["resident_ready_digest"]),
        resident_reserve_bytes=int(contract["resident_policy"]["reserve_bytes"]),
        source_store_root=source_full_root.resolve() / "store",
        resident_root=resident_root.resolve(), cases=tuple(cases),
    )
    return inputs, prereg


def _validate_authority_terminal(
    authority_root: Path, registry_cases: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    contract = _read_json(authority_root / "authority-contract.json")
    matrix = _read_json(authority_root / "authority-matrix.json")
    seal = _read_json(authority_root / "SEALED.json")
    if not all((
        contract.get("schema_version") == AUTHORITY_CONTRACT_SCHEMA,
        contract.get("contract_digest") == FROZEN_AUTHORITY_CONTRACT_DIGEST,
        contract.get("contract_digest") == stable_hash(_without(contract, {"contract_digest"})),
        contract.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
        contract.get("generation_id") == FROZEN_GENERATION_ID,
        contract.get("real_forward_outcomes_accessed") is False,
        matrix.get("schema_version") == AUTHORITY_MATRIX_SCHEMA,
        matrix.get("contract_digest") == contract.get("contract_digest"),
        matrix.get("result_digest") == FROZEN_AUTHORITY_MATRIX_DIGEST,
        matrix.get("result_digest") == stable_hash(_without(matrix, AUTHORITY_MATRIX_OMITTED)),
        matrix.get("completed_cases") == 60,
        matrix.get("invalid_cases") == [],
        matrix.get("gate_passed") is True,
        matrix.get("candidate_results_opened") is False,
        matrix.get("real_forward_outcomes_accessed") is False,
        seal.get("schema_version") == AUTHORITY_SEAL_SCHEMA,
        seal.get("contract_digest") == contract.get("contract_digest"),
        seal.get("authority_matrix_digest") == matrix.get("result_digest"),
        seal.get("seal_digest") == FROZEN_AUTHORITY_SEAL_DIGEST,
        seal.get("seal_digest") == stable_hash(_without(seal, {"seal_digest"})),
        seal.get("authority_correctness_sealed") is True,
        seal.get("production_promotion_authorized") is False,
        seal.get("candidate_results_opened") is False,
        seal.get("real_forward_outcomes_accessed") is False,
    )):
        raise QuotaLadderError("sealed authority prerequisite differs")
    authorities: dict[str, dict[str, Any]] = {}
    for case in registry_cases:
        query_id = str(case["episode_id"])
        payload = _read_json(authority_root / "cases" / f"{query_id}.json")
        matches = payload.get("matches")
        if not all((
            payload.get("schema_version") == AUTHORITY_CASE_SCHEMA,
            payload.get("status") == "completed",
            payload.get("registry_case_id") == case.get("case_id"),
            payload.get("query_episode_id") == query_id,
            payload.get("contract_digest") == contract.get("contract_digest"),
            payload.get("gate_passed") is True,
            payload.get("real_forward_outcomes_accessed") is False,
            isinstance(matches, list), len(matches or []) == 20,
            len({str(row.get("episode_id")) for row in (matches or [])}) == 20,
            payload.get("result_digest")
            == stable_hash(_without(payload, AUTHORITY_CASE_OMITTED)),
            payload.get("checkpoint_integrity_digest")
            == stable_hash(_without(payload, {"created_at", "checkpoint_integrity_digest"})),
        )):
            raise QuotaLadderError(f"sealed authority case differs: {query_id}")
        authorities[query_id] = payload
    expected_files = {
        "authority-contract.json", "authority-matrix.json", "authority-matrix.html",
        "SEALED.json", *(f"cases/{case['episode_id']}.json" for case in registry_cases),
    }
    _validate_plain_tree(authority_root, expected_files, {"cases"}, "authority")
    return authorities, seal


def derive_failed_cases(
    registry_cases: list[dict[str, Any]], failed_rows: list[dict[str, Any]],
    authorities: Mapping[str, Mapping[str, Any]],
    bundles: Mapping[str, Mapping[str, Any]],
) -> tuple[FailedCase, ...]:
    """Bind the four comparison failures to exact authority and baseline pools."""
    by_case = {str(case["case_id"]): case for case in registry_cases}
    result = []
    for ordinal, row in enumerate(failed_rows):
        case_id = str(row["registry_case_id"])
        case = by_case.get(case_id)
        if case is None or str(case["episode_id"]) != str(row["query_episode_id"]):
            raise QuotaLadderError("failed comparison row is outside the registry")
        query_id = str(case["episode_id"])
        authority = authorities[query_id]
        semantic = bundles[query_id].get("semantic", {})
        candidates = semantic.get("candidates")
        if not isinstance(candidates, list):
            raise QuotaLadderError("baseline semantic candidates are absent")
        candidate_ids = tuple(str(value["episode_id"]) for value in candidates)
        truth = tuple(str(value["episode_id"]) for value in authority["matches"])
        retained = tuple(value for value in truth if value in set(candidate_ids))
        if not all((
            row.get("authority_case_digest") == authority.get("result_digest"),
            row.get("candidate_semantic_digest") == semantic.get("semantic_digest"),
            row.get("candidate_count") == len(set(candidate_ids)),
            row.get("retained_count") == len(retained),
            row.get("missing_authority_episode_ids")
            == [value for value in truth if value not in set(candidate_ids)],
            semantic.get("candidate_digest_reconstructed")
            == reconstructed_candidate_digest(candidates),
        )):
            raise QuotaLadderError("failed comparison recall reconstruction differs")
        result.append(FailedCase(
            ordinal=ordinal, registry_case=dict(case), comparison_case=dict(row),
            authority_digest=str(authority["result_digest"]), authority_top20=truth,
            baseline_candidate_digest=str(semantic["candidate_digest_reconstructed"]),
            baseline_candidate_ids=candidate_ids,
            baseline_result_digest=str(semantic["scan_semantics"][0]["result_digest"]),
        ))
    if len(result) != 4:
        raise QuotaLadderError("quota ladder scope must contain exactly four failed cases")
    return tuple(result)


def validate_opened_inputs(
    *, registry_root: Path, candidate_root: Path, authority_root: Path,
    comparison_root: Path, source_full_root: Path, resident_root: Path,
) -> OpenedInputs:
    """Validate the exact terminal evidence before touching the resident pack."""
    _validate_plain_tree(
        comparison_root,
        {"RESULTS_OPENED.json", "candidate-comparison.json", "SEALED.json"},
        set(), "comparison",
    )
    marker = _read_json(comparison_root / "RESULTS_OPENED.json")
    comparison = _read_json(comparison_root / "candidate-comparison.json")
    comparison_seal = _read_json(comparison_root / "SEALED.json")
    _validate_marker(marker)
    failed_rows = _validate_comparison(comparison, comparison_seal, marker)
    registry = _read_json(registry_root / "query-registry.json")
    registry_cases = _validate_registry(registry)
    contract = _read_json(candidate_root / "candidate-contract.json")
    bundles = _validate_candidate_terminal(candidate_root, contract, marker)
    authorities, authority_seal = _validate_authority_terminal(
        authority_root, registry_cases,
    )
    failed = derive_failed_cases(registry_cases, failed_rows, authorities, bundles)
    roots = contract.get("roots", {})
    source_store = source_full_root.resolve() / "store"
    if not all((
        Path(str(roots.get("source_full_root"))).resolve() == source_full_root.resolve(),
        Path(str(roots.get("resident_full_root"))).resolve() == resident_root.resolve(),
        Path(str(roots.get("candidate_root"))).resolve() == candidate_root.resolve(),
        Path(str(roots.get("authority_root"))).resolve() == authority_root.resolve(),
        Path(str(roots.get("comparison_root"))).resolve() == comparison_root.resolve(),
        contract.get("source_pack", {}).get("provenance_digest"),
        type(contract.get("resident_policy", {}).get("reserve_bytes")) is int,
    )):
        raise QuotaLadderError("exact source/resident/evidence roots differ")
    unaffected = int(comparison["retained_total"]) - sum(
        int(row["retained_count"]) for row in failed_rows
    )
    return OpenedInputs(
        registry_digest=FROZEN_REGISTRY_DIGEST,
        producer_contract_digest=FROZEN_PRODUCER_CONTRACT_DIGEST,
        comparison_digest=FROZEN_COMPARISON_DIGEST,
        comparison_seal_digest=FROZEN_COMPARISON_SEAL_DIGEST,
        results_opened_digest=FROZEN_RESULTS_OPENED_DIGEST,
        authority_seal_digest=str(authority_seal["seal_digest"]),
        generation_id=FROZEN_GENERATION_ID,
        provenance_digest=str(contract["source_pack"]["provenance_digest"]),
        resident_reserve_bytes=int(contract["resident_policy"]["reserve_bytes"]),
        source_store_root=source_store,
        resident_root=resident_root.resolve(), failed_cases=failed,
        original_retained_total=int(comparison["retained_total"]),
        unaffected_retained_total=unaffected,
    )


def validate_resident_once(inputs: OpenedInputs | ProducerInputs) -> dict[str, Any]:
    """Full-hash source and tmpfs mirror once, then return a cheap identity lease."""
    ready, validation = prepare_resident_mirror_observed(
        inputs.source_store_root, inputs.resident_root, inputs.generation_id,
        expected_provenance_digest=inputs.provenance_digest,
        reserve_bytes=inputs.resident_reserve_bytes, validate_existing=True,
    )
    observation = observe_ready_strict(inputs.resident_root / "READY.json")
    lease = resident_file_identity_lease(inputs.resident_root / "READY.json")
    seal = ready.get("seal", {})
    if not all((
        observation.get("payload") == ready,
        ready.get("content_digest") == validation.get("content_digest"),
        ready.get("ready_digest") == validation.get("ready_digest"),
        ready.get("seal_digest") == validation.get("seal_digest"),
        seal.get("generation_id") == inputs.generation_id,
        seal.get("provenance_digest") == inputs.provenance_digest,
        seal.get("mirror_root") == str(inputs.resident_root),
        seal.get("mirror_store_root") == str((inputs.resident_root / "store").resolve()),
        seal.get("storage_class") == "tmpfs-backed-generation-v1",
        seal.get("query_specific_inputs_used") is False,
        seal.get("outcomes_or_labels_used") is False,
        seal.get("real_forward_outcomes_accessed") is False,
        lease.get("ready_digest") == ready.get("ready_digest"),
        lease.get("content_digest") == ready.get("content_digest"),
        not isinstance(inputs, ProducerInputs)
        or ready.get("content_digest") == inputs.resident_content_digest,
        not isinstance(inputs, ProducerInputs)
        or ready.get("ready_digest") == inputs.resident_ready_digest,
    )):
        raise QuotaLadderError("validated resident READY/content differs")
    stable = {
        "ready_digest": ready["ready_digest"],
        "content_digest": ready["content_digest"],
        "seal_digest": ready["seal_digest"],
        "ready_file_sha256": observation["ready_file_sha256"],
        "lease": lease,
        "store_root": str((inputs.resident_root / "store").resolve()),
    }
    return {
        **stable,
        "stable_identity_digest": stable_hash(stable),
        "validation_observation": {
            "observation_digest": validation["observation_digest"],
            "observed_at": validation["observed_at"],
        },
    }


def _stable_resident_snapshot(resident: Mapping[str, Any]) -> dict[str, Any]:
    keys = {
        "ready_digest", "content_digest", "seal_digest", "ready_file_sha256",
        "lease", "store_root", "stable_identity_digest",
    }
    snapshot = {key: resident.get(key) for key in keys}
    deterministic = _without(snapshot, {"stable_identity_digest"})
    if not all((
        set(resident) in (keys, keys | {"validation_observation"}),
        snapshot["stable_identity_digest"] == stable_hash(deterministic),
        isinstance(snapshot["store_root"], str),
        type(snapshot["lease"]) is dict,
        snapshot["lease"].get("ready_digest") == snapshot["ready_digest"],
        snapshot["lease"].get("ready_file_sha256") == snapshot["ready_file_sha256"],
        snapshot["lease"].get("content_digest") == snapshot["content_digest"],
    )):
        raise QuotaLadderError("resident stable identity differs")
    return snapshot


def _assert_resident_identity(
    inputs: ProducerInputs, resident: Mapping[str, Any],
) -> dict[str, Any]:
    """Cheap exact READY plus generation-file identity check around every scan."""
    observation = observe_ready_strict(inputs.resident_root / "READY.json")
    lease = resident_file_identity_lease(inputs.resident_root / "READY.json")
    observed = {
        "ready_digest": observation["ready_digest"],
        "content_digest": observation["content_digest"],
        "seal_digest": observation["seal_digest"],
        "ready_file_sha256": observation["ready_file_sha256"],
        "lease": lease,
        "store_root": str((inputs.resident_root / "store").resolve()),
    }
    observed["stable_identity_digest"] = stable_hash(observed)
    if observed != _stable_resident_snapshot(resident):
        raise QuotaLadderError("resident identity changed around physical scan")
    return observed


def _build_query(config_path: Path, failed: FailedCase) -> PackedBoundQuery:
    config = load_config(config_path)
    source = source_from_spec(config.datasets["nasdaq"])
    case = failed.registry_case
    episode = build_episode(
        source, InstrumentKey("nasdaq", str(case["symbol"])),
        str(case["cutoff"]), int(case["lookback"]),
        str(case["representation_version"]),
    )
    if episode.key.id != failed.query_id:
        raise QuotaLadderError("rebuilt development query episode differs")
    stock = asdict(source.causal_prefix_fingerprint(
        InstrumentKey("nasdaq", str(case["symbol"])), str(case["cutoff"]),
    ))
    raw_benchmark = source.benchmark_causal_prefix_fingerprint(str(case["cutoff"]))
    benchmark = asdict(raw_benchmark) if raw_benchmark is not None else None
    if stock != case.get("stock_prefix") or benchmark != case.get("benchmark_prefix"):
        raise QuotaLadderError("development query causal prefix differs")
    latest = latest_eligible_cutoff(episode, 60)
    return PackedBoundQuery(
        failed.query_id, episode.key.instrument.source_symbol,
        int(pd.Timestamp(episode.bars.timestamp.iloc[0]).value),
        int(latest.value), represent(episode), ("A", "B"),
    )


def recall_accounting(
    authority_top20: Iterable[str], candidate_ids: Iterable[str],
) -> dict[str, Any]:
    truth = tuple(str(value) for value in authority_top20)
    candidates = tuple(str(value) for value in candidate_ids)
    if len(truth) != 20 or len(set(truth)) != 20 or len(candidates) != len(set(candidates)):
        raise QuotaLadderError("recall accounting inputs are not unique")
    candidate_set = set(candidates)
    retained = tuple(value for value in truth if value in candidate_set)
    return {
        "retained_authority_episode_ids": list(retained),
        "missing_authority_episode_ids": [
            value for value in truth if value not in candidate_set
        ],
        "retained_count": len(retained),
        "recall_at_20": len(retained) / 20.0,
        "passes_19_of_20": len(retained) >= 19,
    }


def _candidate_rows(report: Any) -> list[dict[str, Any]]:
    return [{
        "episode_id": row.episode_id, "symbol": row.symbol,
        "cutoff_ns": row.cutoff_ns, "quality_tier": row.quality_tier,
        "lower_bound_hex": row.lower_bound.hex(), "routes": list(row.routes),
        "overflow_fallback": row.overflow_fallback,
    } for row in report.candidates]


MAX_SCAN_SCHEMA = "m04r12-development-max-route-sidecar-v1"
MAX_SCAN_SEAL_SCHEMA = "m04r12-development-max-route-seal-v1"
POC_RESULTS_OPENED_SCHEMA = "m04r12-development-results-opened-v1"
MAX_ROUTE_QUOTAS = {
    "composite": 8_000, **{route: 512 for route in COMPONENT_ROUTES},
}


def _rss_mb() -> float:
    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 1024 ** 2
    except (OSError, ValueError, IndexError):
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1_024


def _ranked_entries(entries: Any) -> list[dict[str, Any]]:
    """Losslessly serialize one route heap in exact score/ID order."""
    ordered = sorted(
        entries,
        key=lambda row: (float(row["route_score"]), bytes(row["episode_id"])),
    )
    return [{
        "rank": rank, "episode_id": bytes(row["episode_id"]).hex(),
        "route_score_hex": float(row["route_score"]).hex(),
        "total_hex": float(row["total"]).hex(),
        "cutoff_ns": int(row["cutoff_ns"]), "symbol_id": int(row["symbol_id"]),
        "quality_tier_code": int(row["quality_tier"]),
        "overflow_fallback": bool(row["overflow"]),
    } for rank, row in enumerate(ordered, start=1)]


def _deserialize_rankings(
    rankings: Mapping[str, Any],
) -> dict[str, Any]:
    row_keys = {
        "rank", "episode_id", "route_score_hex", "total_hex", "cutoff_ns",
        "symbol_id", "quality_tier_code", "overflow_fallback",
    }
    if type(rankings) is not dict or set(rankings) != set(MAX_ROUTE_QUOTAS):
        raise QuotaLadderError("max-route sidecar route set differs")
    heaps: dict[str, Any] = {}
    for route in MAX_ROUTE_QUOTAS:
        rows = rankings.get(route)
        if not isinstance(rows, list):
            raise QuotaLadderError("max-route sidecar route is absent")
        heap = packed_search._empty_entries()
        heap = __import__("numpy").empty(len(rows), dtype=packed_search._ENTRY_DTYPE)
        for index, row in enumerate(rows):
            if type(row) is not dict or set(row) != row_keys or row.get("rank") != index + 1:
                raise QuotaLadderError("max-route sidecar rank differs")
            heap[index]["episode_id"] = bytes.fromhex(str(row["episode_id"]))
            heap[index]["route_score"] = float.fromhex(str(row["route_score_hex"]))
            heap[index]["total"] = float.fromhex(str(row["total_hex"]))
            heap[index]["cutoff_ns"] = int(row["cutoff_ns"])
            heap[index]["symbol_id"] = int(row["symbol_id"])
            heap[index]["quality_tier"] = int(row["quality_tier_code"])
            heap[index]["overflow"] = bool(row["overflow_fallback"])
        if _ranked_entries(heap) != rows:
            raise QuotaLadderError("max-route sidecar ordering is not canonical")
        heaps[route] = heap
    return heaps


def _scan_max_with_sidecar(
    inputs: OpenedInputs | ProducerInputs, resident: Mapping[str, Any], query: PackedBoundQuery,
    *, order: str,
) -> tuple[Any, dict[str, Any]]:
    """Use the exact legacy scanner while observing its final route heaps."""
    task_started, task_start_rss = perf_counter(), _rss_mb()
    captured: dict[str, Any] = {}
    original = packed_search._finalize

    def capture(heaps: Mapping[str, Any], symbols: tuple[str, ...]) -> Any:
        captured["heaps"] = {route: entries.copy() for route, entries in heaps.items()}
        captured["symbols"] = tuple(symbols)
        return original(heaps, symbols)

    if packed_search._finalize is not original:
        raise QuotaLadderError("legacy finalizer is already instrumented")
    packed_search._finalize = capture
    try:
        report = scan_packed_bound_proposals_threaded(
            Path(str(resident["store_root"])), inputs.generation_id, query,
            route_quotas=MAX_ROUTE_QUOTAS,
            block_rows=4_096 if order == "forward" else 4_097,
            block_order=order, threads=8, verify_content=False,
            expected_provenance_digest=inputs.provenance_digest,
        )
    finally:
        packed_search._finalize = original
    if set(captured.get("heaps", {})) != set(MAX_ROUTE_QUOTAS):
        raise QuotaLadderError("legacy max scan did not expose every route heap")
    rankings = {
        route: _ranked_entries(captured["heaps"][route])
        for route in MAX_ROUTE_QUOTAS
    }
    if any(len(rankings[route]) != report.route_counts[route] for route in rankings):
        raise QuotaLadderError("max-route sidecar count differs from scan report")
    return report, {
        "symbols": list(captured["symbols"]), "route_rankings": rankings,
        "task_elapsed_seconds": perf_counter() - task_started,
        "task_peak_rss_mb": max(task_start_rss, _rss_mb(), report.peak_rss_mb),
    }


def _max_scan_path(output_root: Path, failed: FailedCase) -> Path:
    return output_root / "max-scans" / f"{failed.ordinal:02d}-{failed.query_id}.json"


def _max_scan_payload(
    failed: FailedCase | ProducerCase, inputs: ProducerInputs,
    resident: Mapping[str, Any],
    forward: Any, forward_sidecar: Mapping[str, Any], reverse: Any,
    reverse_sidecar: Mapping[str, Any],
) -> dict[str, Any]:
    forward_rows, reverse_rows = _candidate_rows(forward), _candidate_rows(reverse)
    measurements = (
        forward.elapsed_seconds, reverse.elapsed_seconds,
        forward.peak_rss_mb, reverse.peak_rss_mb,
        forward_sidecar.get("task_elapsed_seconds"),
        reverse_sidecar.get("task_elapsed_seconds"),
        forward_sidecar.get("task_peak_rss_mb"),
        reverse_sidecar.get("task_peak_rss_mb"),
    )
    if not all((
        all(type(value) in {int, float} and isfinite(value) and value >= 0
            for value in measurements),
        forward.elapsed_seconds <= forward_sidecar["task_elapsed_seconds"],
        reverse.elapsed_seconds <= reverse_sidecar["task_elapsed_seconds"],
        forward.candidate_digest == reconstructed_candidate_digest(forward_rows),
        reverse.candidate_digest == reconstructed_candidate_digest(reverse_rows),
        forward.candidate_digest == reverse.candidate_digest,
        forward.result_digest == reverse.result_digest,
        forward.route_counts == reverse.route_counts,
        forward_sidecar["symbols"] == reverse_sidecar["symbols"],
        forward_sidecar["route_rankings"] == reverse_sidecar["route_rankings"],
        forward.rows_scanned == reverse.rows_scanned,
        forward.eligible_rows == reverse.eligible_rows,
        forward.eligible_main_rows == reverse.eligible_main_rows,
        forward.eligible_overflow_rows == reverse.eligible_overflow_rows,
        forward.eligible_main_rows + forward.eligible_overflow_rows
        == forward.eligible_rows,
    )):
        raise QuotaLadderError("forward/reverse max-route scan differs")
    durable_sidecar = {
        "symbols": forward_sidecar["symbols"],
        "route_rankings": forward_sidecar["route_rankings"],
    }
    sidecar_bytes = len(json.dumps(
        durable_sidecar, sort_keys=True, separators=(",", ":"),
    ).encode())
    candidate_bytes = len(json.dumps(forward_rows, sort_keys=True, separators=(",", ":")).encode())
    deterministic = {
        "schema_version": MAX_SCAN_SCHEMA,
        "status": "truth_not_used_by_scan_or_sidecar",
        "post_open_development_poc": True,
        "production_promotion_authorized": False,
        "real_forward_outcomes_accessed": False,
        "registry_digest": inputs.registry_digest,
        "producer_contract_digest": inputs.producer_contract_digest,
        "poc_contract_digest": inputs.poc_contract_digest,
        "resident_content_digest": resident["content_digest"],
        "resident_ready_digest": resident["ready_digest"],
        "resident_stable_identity_digest": resident["stable_identity_digest"],
        "registry_case_id": failed.case_id,
        "query_episode_id": failed.query_id,
        "maximum_route_quotas": MAX_ROUTE_QUOTAS,
        "threads": 8, "verify_content": False,
        "forward": {
            "block_rows": forward.block_rows, "block_order": forward.block_order,
            "rows_scanned": forward.rows_scanned,
            "eligible_rows": forward.eligible_rows,
            "eligible_main_rows": forward.eligible_main_rows,
            "eligible_overflow_rows": forward.eligible_overflow_rows,
            "route_counts": dict(forward.route_counts),
            "candidate_count": len(forward_rows),
            "candidate_digest": forward.candidate_digest,
            "result_digest": forward.result_digest,
            "elapsed_seconds": forward.elapsed_seconds,
            "peak_rss_mb": forward.peak_rss_mb,
            "task_elapsed_seconds": forward_sidecar["task_elapsed_seconds"],
            "task_peak_rss_mb": forward_sidecar["task_peak_rss_mb"],
        },
        "reverse": {
            "block_rows": reverse.block_rows, "block_order": reverse.block_order,
            "rows_scanned": reverse.rows_scanned,
            "eligible_rows": reverse.eligible_rows,
            "eligible_main_rows": reverse.eligible_main_rows,
            "eligible_overflow_rows": reverse.eligible_overflow_rows,
            "route_counts": dict(reverse.route_counts),
            "candidate_count": len(reverse_rows),
            "candidate_digest": reverse.candidate_digest,
            "result_digest": reverse.result_digest,
            "elapsed_seconds": reverse.elapsed_seconds,
            "peak_rss_mb": reverse.peak_rss_mb,
            "task_elapsed_seconds": reverse_sidecar["task_elapsed_seconds"],
            "task_peak_rss_mb": reverse_sidecar["task_peak_rss_mb"],
        },
        "candidate_payload_bytes": candidate_bytes,
        "route_sidecar_bytes": sidecar_bytes,
        "symbols": forward_sidecar["symbols"],
        "route_rankings": forward_sidecar["route_rankings"],
        "forward_reverse_exact": True,
        "performance_gates": {
            "forward_first_at_most_120_seconds": (
                forward_sidecar["task_elapsed_seconds"] <= 120.0
            ),
            "reverse_warm_at_most_60_seconds": (
                reverse_sidecar["task_elapsed_seconds"] <= 60.0
            ),
            "peak_rss_at_most_1536_mb": max(
                forward_sidecar["task_peak_rss_mb"],
                reverse_sidecar["task_peak_rss_mb"],
            ) <= 1_536.0,
            "sidecar_at_most_32_mib": sidecar_bytes <= MAX_SIDECAR_BYTES_PER_CASE,
        },
    }
    deterministic["performance_passed"] = all(
        deterministic["performance_gates"].values()
    )
    return {
        **deterministic, "created_at": datetime.now(timezone.utc).isoformat(),
        "result_digest": stable_hash(deterministic),
    }


def _validate_max_scan(
    payload: Mapping[str, Any], failed: FailedCase | ProducerCase,
    inputs: ProducerInputs,
    resident: Mapping[str, Any],
) -> None:
    top_keys = {
        "schema_version", "status", "post_open_development_poc",
        "production_promotion_authorized", "real_forward_outcomes_accessed",
        "registry_digest", "producer_contract_digest", "poc_contract_digest",
        "resident_content_digest", "resident_ready_digest",
        "resident_stable_identity_digest", "registry_case_id",
        "query_episode_id", "maximum_route_quotas", "threads", "verify_content",
        "forward", "reverse", "candidate_payload_bytes", "route_sidecar_bytes",
        "symbols", "route_rankings", "forward_reverse_exact", "performance_gates",
        "performance_passed", "created_at", "result_digest",
    }
    forward_keys = {
        "block_rows", "block_order", "rows_scanned", "eligible_rows",
        "eligible_main_rows", "eligible_overflow_rows", "route_counts",
        "candidate_count", "candidate_digest", "result_digest", "elapsed_seconds",
        "peak_rss_mb", "task_elapsed_seconds", "task_peak_rss_mb",
    }
    reverse_keys = {
        "block_rows", "block_order", "rows_scanned", "eligible_rows",
        "eligible_main_rows", "eligible_overflow_rows", "route_counts",
        "candidate_count", "candidate_digest", "result_digest",
        "elapsed_seconds", "peak_rss_mb", "task_elapsed_seconds", "task_peak_rss_mb",
    }
    forward, reverse = payload.get("forward", {}), payload.get("reverse", {})
    measurements = [
        forward.get(name) for name in (
            "elapsed_seconds", "peak_rss_mb", "task_elapsed_seconds", "task_peak_rss_mb",
        )
    ] + [reverse.get(name) for name in (
        "elapsed_seconds", "peak_rss_mb", "task_elapsed_seconds", "task_peak_rss_mb",
    )]
    gates = {
        "forward_first_at_most_120_seconds": (
            type(forward.get("task_elapsed_seconds")) in {int, float}
            and forward["task_elapsed_seconds"] <= 120.0
        ),
        "reverse_warm_at_most_60_seconds": (
            type(reverse.get("task_elapsed_seconds")) in {int, float}
            and reverse["task_elapsed_seconds"] <= 60.0
        ),
        "peak_rss_at_most_1536_mb": all(
            type(value) in {int, float} for value in (
                forward.get("task_peak_rss_mb"), reverse.get("task_peak_rss_mb"),
            )
        ) and max(forward["task_peak_rss_mb"], reverse["task_peak_rss_mb"]) <= 1_536.0,
        "sidecar_at_most_32_mib": (
            type(payload.get("route_sidecar_bytes")) is int
            and payload["route_sidecar_bytes"] <= MAX_SIDECAR_BYTES_PER_CASE
        ),
    }
    if not all((
        set(payload) == top_keys,
        type(forward) is dict and set(forward) == forward_keys,
        type(reverse) is dict and set(reverse) == reverse_keys,
        payload.get("schema_version") == MAX_SCAN_SCHEMA,
        payload.get("status") == "truth_not_used_by_scan_or_sidecar",
        payload.get("post_open_development_poc") is True,
        payload.get("production_promotion_authorized") is False,
        payload.get("real_forward_outcomes_accessed") is False,
        payload.get("registry_digest") == inputs.registry_digest,
        payload.get("producer_contract_digest") == inputs.producer_contract_digest,
        payload.get("poc_contract_digest") == inputs.poc_contract_digest,
        payload.get("resident_content_digest") == resident["content_digest"],
        payload.get("resident_ready_digest") == resident["ready_digest"],
        payload.get("resident_stable_identity_digest")
        == resident["stable_identity_digest"],
        payload.get("registry_case_id") == failed.case_id,
        payload.get("query_episode_id") == failed.query_id,
        payload.get("maximum_route_quotas") == MAX_ROUTE_QUOTAS,
        payload.get("threads") == 8, payload.get("verify_content") is False,
        forward.get("block_rows") == 4_096, forward.get("block_order") == "forward",
        reverse.get("block_rows") == 4_097, reverse.get("block_order") == "reverse",
        all(type(value) in {int, float} and isfinite(value) and value >= 0
            for value in measurements),
        forward.get("elapsed_seconds") <= forward.get("task_elapsed_seconds"),
        reverse.get("elapsed_seconds") <= reverse.get("task_elapsed_seconds"),
        all(type(forward.get(name)) is int and forward[name] >= 0 for name in (
            "rows_scanned", "eligible_rows", "eligible_main_rows",
            "eligible_overflow_rows", "candidate_count",
        )),
        int(forward.get("eligible_rows", -1))
        == int(forward.get("eligible_main_rows", -2))
        + int(forward.get("eligible_overflow_rows", -3)),
        int(forward.get("rows_scanned", -1)) >= int(forward.get("eligible_rows", 0)),
        all(type(reverse.get(name)) is int and reverse[name] >= 0 for name in (
            "rows_scanned", "eligible_rows", "eligible_main_rows",
            "eligible_overflow_rows", "candidate_count",
        )),
        reverse.get("rows_scanned") == forward.get("rows_scanned"),
        reverse.get("eligible_rows") == forward.get("eligible_rows"),
        reverse.get("eligible_main_rows") == forward.get("eligible_main_rows"),
        reverse.get("eligible_overflow_rows") == forward.get("eligible_overflow_rows"),
        reverse.get("route_counts") == forward.get("route_counts"),
        reverse.get("candidate_count") == forward.get("candidate_count"),
        forward.get("candidate_digest") == reverse.get("candidate_digest"),
        forward.get("result_digest") == reverse.get("result_digest"),
        payload.get("forward_reverse_exact") is True,
        payload.get("performance_gates") == gates,
        payload.get("performance_passed") is all(gates.values()),
        payload.get("result_digest")
        == stable_hash(_without(payload, {"created_at", "result_digest"})),
    )):
        raise QuotaLadderError("existing max-route scan differs")
    _timestamp(payload.get("created_at"), "max-route scan")
    try:
        heaps = _deserialize_rankings(payload.get("route_rankings", {}))
        candidates, route_counts, candidate_digest = packed_search._finalize(
            heaps, tuple(payload.get("symbols", [])),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise QuotaLadderError("max-route checkpoint sidecar differs") from exc
    candidate_rows = _candidate_rows(
        type("Report", (), {"candidates": candidates})()
    )
    sidecar = {
        "symbols": payload.get("symbols"),
        "route_rankings": payload.get("route_rankings"),
    }
    standard_digest = _standard_result_digest(
        failed, inputs, payload, MAX_ROUTE_QUOTAS, route_counts, candidate_digest,
    )
    if any((
        route_counts != forward.get("route_counts"),
        candidate_digest != forward.get("candidate_digest"),
        len(candidates) != forward.get("candidate_count"),
        reconstructed_candidate_digest(candidate_rows) != candidate_digest,
        standard_digest != forward.get("result_digest"),
        len(json.dumps(sidecar, sort_keys=True, separators=(",", ":")).encode())
        != payload.get("route_sidecar_bytes"),
        len(json.dumps(candidate_rows, sort_keys=True, separators=(",", ":")).encode())
        != payload.get("candidate_payload_bytes"),
        len(candidates) > sum(MAX_ROUTE_QUOTAS.values()),
        any(len(heaps[route]) != route_counts[route] for route in MAX_ROUTE_QUOTAS),
    )):
        raise QuotaLadderError("max-route checkpoint heap counts differ")


def _validate_producer_baseline(
    payload: Mapping[str, Any], case: ProducerCase, inputs: ProducerInputs,
) -> None:
    """Prove the preregistered v2 baseline from sealed max heaps before truth opens."""
    heaps = _deserialize_rankings(payload["route_rankings"])
    selected = _select_heaps(heaps, DEFAULT_ROUTE_QUOTAS)
    candidates, route_counts, candidate_digest = packed_search._finalize(
        selected, tuple(payload["symbols"]),
    )
    ids = tuple(row.episode_id for row in candidates)
    result_digest = _standard_result_digest(
        case, inputs, payload, DEFAULT_ROUTE_QUOTAS, route_counts, candidate_digest,
    )
    if not all((
        ids == case.baseline_candidate_ids,
        candidate_digest == case.baseline_candidate_digest,
        result_digest == case.baseline_result_digest,
    )):
        raise QuotaLadderError("producer max heaps do not reproduce preregistered baseline")


def _select_heaps(max_heaps: Mapping[str, Any], quotas: Mapping[str, int]) -> dict[str, Any]:
    if set(quotas) != set(MAX_ROUTE_QUOTAS) or any(
        type(quotas[route]) is not int or not 1 <= quotas[route] <= MAX_ROUTE_QUOTAS[route]
        for route in quotas
    ):
        raise QuotaLadderError("derived route quotas exceed the sealed max scan")
    return {
        route: packed_search._stable_bounded(
            max_heaps[route], packed_search._empty_entries(), quotas[route],
        )
        for route in MAX_ROUTE_QUOTAS
    }


def _standard_result_digest(
    failed: FailedCase | ProducerCase,
    inputs: OpenedInputs | ProducerInputs, max_payload: Mapping[str, Any],
    quotas: Mapping[str, int], route_counts: Mapping[str, int], candidate_digest: str,
) -> str:
    forward = max_payload["forward"]
    return stable_hash({
        "schema_version": packed_search.SEARCH_SCHEMA_VERSION,
        "contract_digest": packed_search.packed_bound_search_contract()["digest"],
        "generation_id": inputs.generation_id,
        "query_episode_id": failed.query_id,
        "rows_scanned": forward["rows_scanned"],
        "eligible_rows": forward["eligible_rows"],
        "eligible_main_rows": forward["eligible_main_rows"],
        "eligible_overflow_rows": forward["eligible_overflow_rows"],
        "route_counts": dict(route_counts), "route_quotas": dict(quotas),
        "candidate_digest": candidate_digest,
        "real_forward_outcomes_accessed": False,
    })


def _truth_route_ranks(
    failed: FailedCase, rankings: Mapping[str, Any],
) -> list[dict[str, Any]]:
    indexed = {
        route: {row["episode_id"]: row for row in rows}
        for route, rows in rankings.items()
    }
    return [{
        "episode_id": episode_id,
        "routes": {
            route: ({
                "rank": indexed[route][episode_id]["rank"],
                "score_hex": indexed[route][episode_id]["route_score_hex"],
            } if episode_id in indexed[route] else None)
            for route in MAX_ROUTE_QUOTAS
        },
    } for episode_id in failed.authority_top20]


def _derived_checkpoint_path(
    output_root: Path, failed: FailedCase, configuration: Mapping[str, Any],
) -> Path:
    return output_root / "derived" / (
        f"{failed.ordinal:02d}-{failed.query_id}--"
        f"{int(configuration['ordinal']):02d}-{configuration['name']}.json"
    )


def _derive_checkpoint(
    failed: FailedCase, configuration: Mapping[str, Any], inputs: OpenedInputs,
    resident: Mapping[str, Any], max_payload: Mapping[str, Any],
) -> dict[str, Any]:
    started = perf_counter()
    peak = _rss_mb()
    heaps = _deserialize_rankings(max_payload["route_rankings"])
    selected = _select_heaps(heaps, configuration["route_quotas"])
    candidates, route_counts, candidate_digest = packed_search._finalize(
        selected, tuple(max_payload["symbols"]),
    )
    rows = _candidate_rows(type("Report", (), {"candidates": candidates})())
    ids = [row["episode_id"] for row in rows]
    if reconstructed_candidate_digest(rows) != candidate_digest:
        raise QuotaLadderError("derived candidate digest differs")
    scan_digest = _standard_result_digest(
        failed, inputs, max_payload, configuration["route_quotas"],
        route_counts, candidate_digest,
    )
    recall = recall_accounting(failed.authority_top20, ids)
    baseline_match = None
    if configuration["name"] == "baseline":
        baseline_match = all((
            candidate_digest == failed.baseline_candidate_digest,
            tuple(ids) == failed.baseline_candidate_ids,
            scan_digest == failed.baseline_result_digest,
            recall["retained_count"] == failed.comparison_case["retained_count"],
        ))
        if not baseline_match:
            raise QuotaLadderError("derived baseline does not reproduce sealed v2")
    elapsed = perf_counter() - started
    candidate_bytes = len(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode())
    deterministic = {
        "schema_version": CHECKPOINT_SCHEMA,
        "status": "retrospective_authority_comparison",
        "post_open_development_poc": True,
        "development_only": True, "diagnostic_truth_opened": True,
        "production_promotion_authorized": False,
        "real_forward_outcomes_accessed": False,
        "registry_digest": inputs.registry_digest,
        "producer_contract_digest": inputs.producer_contract_digest,
        "poc_contract_digest": max_payload["poc_contract_digest"],
        "comparison_digest": inputs.comparison_digest,
        "authority_case_digest": failed.authority_digest,
        "resident_content_digest": resident["content_digest"],
        "resident_ready_digest": resident["ready_digest"],
        "resident_stable_identity_digest": resident["stable_identity_digest"],
        "max_scan_digest": max_payload["result_digest"],
        "registry_case_id": failed.case_id, "query_episode_id": failed.query_id,
        "configuration_ordinal": configuration["ordinal"],
        "configuration_name": configuration["name"],
        "route_quotas": dict(configuration["route_quotas"]),
        "candidate_count": len(candidates), "candidate_digest": candidate_digest,
        "candidate_episode_ids": ids,
        "candidate_membership_digest": stable_hash(sorted(ids)),
        "candidate_payload_bytes": candidate_bytes,
        "marginal_candidate_count_vs_baseline": (
            len(candidates) - len(failed.baseline_candidate_ids)
        ),
        "route_counts": route_counts, "standard_result_digest": scan_digest,
        "derivation_seconds": elapsed,
        "derivation_peak_rss_mb": max(peak, _rss_mb()),
        "physical_scan_elapsed_seconds": max_payload["forward"]["elapsed_seconds"],
        "physical_scan_peak_rss_mb": max_payload["forward"]["peak_rss_mb"],
        "recall": recall,
        "marginal_retained_vs_baseline": (
            recall["retained_count"] - failed.comparison_case["retained_count"]
        ),
        "authority_top20_route_ranks": _truth_route_ranks(
            failed, max_payload["route_rankings"],
        ),
        "baseline_reproduces_terminal_v2": baseline_match,
    }
    return {
        **deterministic, "created_at": datetime.now(timezone.utc).isoformat(),
        "result_digest": stable_hash(deterministic),
    }


def _validate_derived(
    payload: Mapping[str, Any], failed: FailedCase, configuration: Mapping[str, Any],
    inputs: OpenedInputs, resident: Mapping[str, Any],
    max_payload: Mapping[str, Any],
) -> None:
    keys = {
        "schema_version", "status", "post_open_development_poc", "development_only",
        "diagnostic_truth_opened", "production_promotion_authorized",
        "real_forward_outcomes_accessed", "registry_digest",
        "producer_contract_digest", "comparison_digest", "authority_case_digest",
        "poc_contract_digest", "resident_content_digest", "resident_ready_digest",
        "resident_stable_identity_digest",
        "max_scan_digest", "registry_case_id",
        "query_episode_id", "configuration_ordinal", "configuration_name",
        "route_quotas", "candidate_count", "candidate_digest",
        "candidate_episode_ids", "candidate_membership_digest",
        "candidate_payload_bytes", "marginal_candidate_count_vs_baseline",
        "route_counts", "standard_result_digest", "derivation_seconds",
        "derivation_peak_rss_mb", "physical_scan_elapsed_seconds",
        "physical_scan_peak_rss_mb", "recall", "marginal_retained_vs_baseline",
        "authority_top20_route_ranks", "baseline_reproduces_terminal_v2",
        "created_at", "result_digest",
    }
    if type(max_payload) is not dict:
        raise QuotaLadderError("derived validator max payload is not bound")
    heaps = _deserialize_rankings(max_payload["route_rankings"])
    selected = _select_heaps(heaps, configuration["route_quotas"])
    candidates, route_counts, candidate_digest = packed_search._finalize(
        selected, tuple(max_payload["symbols"]),
    )
    rows = _candidate_rows(type("Report", (), {"candidates": candidates})())
    ids = [row["episode_id"] for row in rows]
    recall = recall_accounting(failed.authority_top20, ids)
    standard = _standard_result_digest(
        failed, inputs, max_payload, configuration["route_quotas"],
        route_counts, candidate_digest,
    )
    baseline = None
    if configuration["name"] == "baseline":
        baseline = all((
            candidate_digest == failed.baseline_candidate_digest,
            tuple(ids) == failed.baseline_candidate_ids,
            standard == failed.baseline_result_digest,
            recall["retained_count"] == failed.comparison_case["retained_count"],
        ))
    if not all((
        set(payload) == keys,
        payload.get("schema_version") == CHECKPOINT_SCHEMA,
        payload.get("status") == "retrospective_authority_comparison",
        payload.get("post_open_development_poc") is True,
        payload.get("development_only") is True,
        payload.get("diagnostic_truth_opened") is True,
        payload.get("production_promotion_authorized") is False,
        payload.get("real_forward_outcomes_accessed") is False,
        payload.get("registry_digest") == inputs.registry_digest,
        payload.get("producer_contract_digest") == inputs.producer_contract_digest,
        payload.get("poc_contract_digest") == max_payload.get("poc_contract_digest"),
        payload.get("comparison_digest") == inputs.comparison_digest,
        payload.get("authority_case_digest") == failed.authority_digest,
        payload.get("resident_content_digest") == resident["content_digest"],
        payload.get("resident_ready_digest") == resident["ready_digest"],
        payload.get("resident_stable_identity_digest")
        == resident["stable_identity_digest"],
        payload.get("max_scan_digest") == max_payload.get("result_digest"),
        payload.get("registry_case_id") == failed.case_id,
        payload.get("query_episode_id") == failed.query_id,
        payload.get("configuration_ordinal") == configuration["ordinal"],
        payload.get("configuration_name") == configuration["name"],
        payload.get("route_quotas") == configuration["route_quotas"],
        payload.get("candidate_count") == len(candidates),
        payload.get("candidate_digest") == candidate_digest,
        payload.get("candidate_episode_ids") == ids,
        payload.get("candidate_membership_digest") == stable_hash(sorted(ids)),
        payload.get("candidate_payload_bytes")
        == len(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()),
        payload.get("marginal_candidate_count_vs_baseline")
        == len(candidates) - len(failed.baseline_candidate_ids),
        payload.get("route_counts") == route_counts,
        type(payload.get("route_counts")) is dict,
        set(payload.get("route_counts", {})) == set(MAX_ROUTE_QUOTAS),
        payload.get("standard_result_digest") == standard,
        type(payload.get("derivation_seconds")) in {int, float}
        and isfinite(payload["derivation_seconds"]) and payload["derivation_seconds"] >= 0,
        type(payload.get("derivation_peak_rss_mb")) in {int, float}
        and isfinite(payload["derivation_peak_rss_mb"])
        and payload["derivation_peak_rss_mb"] >= 0,
        payload.get("physical_scan_elapsed_seconds")
        == max_payload["forward"]["elapsed_seconds"],
        payload.get("physical_scan_peak_rss_mb")
        == max_payload["forward"]["peak_rss_mb"],
        payload.get("recall") == recall,
        payload.get("marginal_retained_vs_baseline")
        == recall["retained_count"] - failed.comparison_case["retained_count"],
        payload.get("authority_top20_route_ranks")
        == _truth_route_ranks(failed, max_payload["route_rankings"]),
        payload.get("baseline_reproduces_terminal_v2") is baseline,
        payload.get("result_digest")
        == stable_hash(_without(payload, {"created_at", "result_digest"})),
    )):
        raise QuotaLadderError("existing derived quota checkpoint differs")
    _timestamp(payload.get("created_at"), "derived quota checkpoint")


def _validate_output_tree(output_root: Path, allowed: set[str]) -> None:
    if not output_root.exists():
        return
    files, directories = set(), set()
    for path in output_root.rglob("*"):
        observed, relative = path.lstat(), str(path.relative_to(output_root))
        if stat.S_ISLNK(observed.st_mode):
            raise QuotaLadderError("quota output contains a symlink")
        if stat.S_ISDIR(observed.st_mode):
            directories.add(relative)
        elif stat.S_ISREG(observed.st_mode):
            files.add(relative)
        else:
            raise QuotaLadderError("quota output contains a special entry")
    if not files.issubset(allowed) or not directories.issubset({"max-scans", "derived"}):
        raise QuotaLadderError("quota output tree contains an unexpected artifact")


def _producer_files(output_root: Path, inputs: ProducerInputs) -> set[str]:
    return {
        "POC_CONTRACT.json", "RESIDENT_READY.json", "MAX_SCAN_SEALED.json",
        *(str(_max_scan_path(output_root, case).relative_to(output_root))
          for case in inputs.cases),
    }


def _observed_output_files(output_root: Path) -> set[str]:
    if not output_root.exists():
        return set()
    return {
        str(path.relative_to(output_root))
        for path in output_root.rglob("*") if path.is_file()
    }


def _validate_producer_tree(
    output_root: Path, inputs: ProducerInputs, *, phase: str,
) -> None:
    producer = _producer_files(output_root, inputs)
    if phase == "partial":
        allowed = producer
    elif phase == "sealed":
        allowed = producer
    elif phase == "opened":
        allowed = producer | {"RESULTS_OPENED.json", "quota-ladder.json"} | {
            str(_derived_checkpoint_path(output_root, case, configuration).relative_to(output_root))
            for case in inputs.cases for configuration in quota_configurations()
        }
    else:
        raise QuotaLadderError("unknown quota output phase")
    _validate_output_tree(output_root, allowed)
    observed = _observed_output_files(output_root)
    if phase == "partial" and not observed.issubset(producer):
        raise QuotaLadderError("partial producer tree differs")
    if phase == "partial" and "MAX_SCAN_SEALED.json" in observed and observed != producer:
        raise QuotaLadderError("partial tree contains a premature producer seal")
    if phase == "sealed" and observed != producer:
        raise QuotaLadderError("sealed producer tree differs")
    if phase == "opened" and not (
        producer | {"RESULTS_OPENED.json"}
    ).issubset(observed):
        raise QuotaLadderError("opened comparator lacks sealed producer tree")
    if phase == "opened" and "quota-ladder.json" in observed and observed != allowed:
        raise QuotaLadderError("terminal comparator tree is incomplete")


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def produce_max_evidence(
    *, config_path: Path, output_root: Path, inputs: ProducerInputs,
    preregistration: Mapping[str, Any], resident: Mapping[str, Any],
    query_builder: Callable[[Path, ProducerCase], PackedBoundQuery] = _build_query,
    max_scanner: Callable[..., tuple[Any, dict[str, Any]]] = _scan_max_with_sidecar,
    resident_checker: Callable[[ProducerInputs, Mapping[str, Any]], Any]
    = _assert_resident_identity,
) -> dict[str, Any]:
    """Truth-blind phase: no comparison/authority root is accepted or opened."""
    allowed = {
        "POC_CONTRACT.json", "RESIDENT_READY.json", "MAX_SCAN_SEALED.json",
        *(str(_max_scan_path(output_root, case).relative_to(output_root))
          for case in inputs.cases),
    }
    _validate_producer_tree(output_root, inputs, phase="partial")
    stable_resident = _stable_resident_snapshot(resident)
    snapshots = {
        output_root / "POC_CONTRACT.json": dict(preregistration),
        output_root / "RESIDENT_READY.json": stable_resident,
    }
    for path, expected in snapshots.items():
        if path.exists():
            if _read_json(path) != expected:
                raise QuotaLadderError("producer snapshot differs")
        else:
            _atomic_create(path, expected)
    max_scans = []
    for case in inputs.cases:
        path = _max_scan_path(output_root, case)
        if path.exists():
            payload = _read_json(path)
            _validate_max_scan(payload, case, inputs, resident)
        else:
            query = query_builder(config_path, case)
            resident_checker(inputs, resident)
            forward, forward_sidecar = max_scanner(
                inputs, resident, query, order="forward",
            )
            resident_checker(inputs, resident)
            resident_checker(inputs, resident)
            reverse, reverse_sidecar = max_scanner(
                inputs, resident, query, order="reverse",
            )
            resident_checker(inputs, resident)
            payload = _max_scan_payload(
                case, inputs, resident, forward, forward_sidecar,
                reverse, reverse_sidecar,
            )
            _atomic_create(path, payload)
        _validate_producer_baseline(payload, case, inputs)
        if payload.get("performance_passed") is not True:
            raise QuotaLadderError(f"max-route performance gate failed: {case.case_id}")
        if _tree_bytes(output_root) > MAX_OUTPUT_BYTES:
            raise QuotaLadderError("producer output exceeds the preregistered budget")
        max_scans.append(payload)
    resident_checker(inputs, resident)
    deterministic = {
        "schema_version": PRODUCER_SEAL_SCHEMA,
        "status": "truth_blind_max_route_evidence_complete",
        "post_open_development_poc": True,
        "truth_inputs_opened": False,
        "production_promotion_authorized": False,
        "real_forward_outcomes_accessed": False,
        "poc_contract_digest": inputs.poc_contract_digest,
        "registry_digest": inputs.registry_digest,
        "producer_contract_digest": inputs.producer_contract_digest,
        "resident_content_digest": resident["content_digest"],
        "resident_ready_digest": resident["ready_digest"],
        "resident_stable_identity_digest": resident["stable_identity_digest"],
        "maximum_route_quotas": MAX_ROUTE_QUOTAS,
        "case_query_ids": list(FROZEN_QUERY_IDS),
        "case_order_digest": stable_hash(list(FROZEN_QUERY_IDS)),
        "max_scan_digests": [row["result_digest"] for row in max_scans],
        "all_performance_gates_passed": all(
            row["performance_passed"] for row in max_scans
        ),
    }
    seal = {
        **deterministic, "created_at": datetime.now(timezone.utc).isoformat(),
        "seal_digest": stable_hash(deterministic),
    }
    path = output_root / "MAX_SCAN_SEALED.json"
    if path.exists():
        observed = _read_json(path)
        if _without(observed, {"created_at"}) != _without(seal, {"created_at"}):
            raise QuotaLadderError("producer max-route seal differs")
        seal = observed
    else:
        _atomic_create(path, seal)
    if _tree_bytes(output_root) > MAX_OUTPUT_BYTES:
        raise QuotaLadderError("sealed producer output exceeds budget")
    resident_checker(inputs, resident)
    _validate_producer_tree(output_root, inputs, phase="sealed")
    return seal


def validate_producer_evidence(
    *, output_root: Path, inputs: ProducerInputs,
    preregistration: Mapping[str, Any], resident: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recursively reconstruct every producer checkpoint before truth opens."""
    contract = _read_json(output_root / "POC_CONTRACT.json")
    resident_snapshot = _read_json(output_root / "RESIDENT_READY.json")
    if contract != dict(preregistration) or resident_snapshot != _stable_resident_snapshot(resident):
        raise QuotaLadderError("producer contract or resident snapshot differs")
    max_scans = []
    for case in inputs.cases:
        payload = _read_json(_max_scan_path(output_root, case))
        _validate_max_scan(payload, case, inputs, resident)
        _validate_producer_baseline(payload, case, inputs)
        if payload.get("performance_passed") is not True:
            raise QuotaLadderError("producer checkpoint performance failed")
        max_scans.append(payload)
    seal = _read_json(output_root / "MAX_SCAN_SEALED.json")
    keys = {
        "schema_version", "status", "post_open_development_poc",
        "truth_inputs_opened", "production_promotion_authorized",
        "real_forward_outcomes_accessed", "poc_contract_digest",
        "registry_digest", "producer_contract_digest", "resident_content_digest",
        "resident_ready_digest", "resident_stable_identity_digest",
        "maximum_route_quotas", "case_query_ids",
        "case_order_digest", "max_scan_digests", "all_performance_gates_passed",
        "created_at", "seal_digest",
    }
    if not all((
        set(seal) == keys,
        seal.get("schema_version") == PRODUCER_SEAL_SCHEMA,
        seal.get("status") == "truth_blind_max_route_evidence_complete",
        seal.get("truth_inputs_opened") is False,
        seal.get("production_promotion_authorized") is False,
        seal.get("real_forward_outcomes_accessed") is False,
        seal.get("poc_contract_digest") == inputs.poc_contract_digest,
        seal.get("registry_digest") == inputs.registry_digest,
        seal.get("producer_contract_digest") == inputs.producer_contract_digest,
        seal.get("resident_content_digest") == resident["content_digest"],
        seal.get("resident_ready_digest") == resident["ready_digest"],
        seal.get("resident_stable_identity_digest")
        == resident["stable_identity_digest"],
        seal.get("maximum_route_quotas") == MAX_ROUTE_QUOTAS,
        seal.get("case_query_ids") == list(FROZEN_QUERY_IDS),
        seal.get("case_order_digest") == stable_hash(list(FROZEN_QUERY_IDS)),
        seal.get("max_scan_digests") == [row["result_digest"] for row in max_scans],
        seal.get("all_performance_gates_passed") is True,
        seal.get("seal_digest") == stable_hash(_without(seal, {"created_at", "seal_digest"})),
    )):
        raise QuotaLadderError("producer max-route seal reconstruction differs")
    _timestamp(seal.get("created_at"), "producer max-route seal")
    if _tree_bytes(output_root) > MAX_OUTPUT_BYTES:
        raise QuotaLadderError("producer evidence exceeds output budget")
    _validate_producer_tree(
        output_root, inputs,
        phase="opened" if (output_root / "RESULTS_OPENED.json").exists() else "sealed",
    )
    return max_scans, seal


def _publish_comparator_marker(
    output_root: Path, inputs: ProducerInputs, producer_seal: Mapping[str, Any],
) -> dict[str, Any]:
    deterministic = {
        "schema_version": POC_RESULTS_OPENED_SCHEMA,
        "status": "authority and comparison truth about to be opened",
        "post_open_development_poc": True,
        "global_v2_results_were_already_opened": True,
        "global_v2_results_opened_digest": FROZEN_RESULTS_OPENED_DIGEST,
        "poc_contract_digest": inputs.poc_contract_digest,
        "producer_max_scan_seal_digest": producer_seal["seal_digest"],
        "production_promotion_authorized": False,
        "real_forward_outcomes_accessed": False,
    }
    expected = {
        **deterministic, "created_at": datetime.now(timezone.utc).isoformat(),
        "result_digest": stable_hash(deterministic),
    }
    path = output_root / "RESULTS_OPENED.json"
    if path.exists():
        observed = _read_json(path)
        if _without(observed, {"created_at"}) != _without(expected, {"created_at"}):
            raise QuotaLadderError("comparator RESULTS_OPENED marker differs")
        _timestamp(observed.get("created_at"), "POC RESULTS_OPENED")
        return observed
    _atomic_create(path, expected)
    return expected


def compare_producer_evidence(
    *, output_root: Path, producer_inputs: ProducerInputs,
    preregistration: Mapping[str, Any], resident: Mapping[str, Any],
    truth_loader: Callable[[], OpenedInputs],
) -> dict[str, Any]:
    """Publish marker first, then load truth and derive retrospective evidence."""
    max_scans, producer_seal = validate_producer_evidence(
        output_root=output_root, inputs=producer_inputs,
        preregistration=preregistration, resident=resident,
    )
    marker = _publish_comparator_marker(output_root, producer_inputs, producer_seal)
    # This call is deliberately the first authority/comparison body read in this phase.
    truth = truth_loader()
    if not all((
        tuple(value.query_id for value in truth.failed_cases) == FROZEN_QUERY_IDS,
        tuple(value.query_id for value in producer_inputs.cases) == FROZEN_QUERY_IDS,
        all(
            truth_case.query_id == producer_case.query_id
            and truth_case.case_id == producer_case.case_id
            and truth_case.baseline_candidate_digest == producer_case.baseline_candidate_digest
            and truth_case.baseline_candidate_ids == producer_case.baseline_candidate_ids
            and truth_case.baseline_result_digest == producer_case.baseline_result_digest
            for truth_case, producer_case in zip(
                truth.failed_cases, producer_inputs.cases, strict=True,
            )
        ),
    )):
        raise QuotaLadderError("post-marker truth scope differs from producer scope")
    configurations = quota_configurations()
    derived = []
    for failed, max_payload in zip(truth.failed_cases, max_scans, strict=True):
        for configuration in configurations:
            path = _derived_checkpoint_path(output_root, failed, configuration)
            if path.exists():
                payload = _read_json(path)
                _validate_derived(
                    payload, failed, configuration, truth, resident, max_payload,
                )
            else:
                payload = _derive_checkpoint(
                    failed, configuration, truth, resident, max_payload,
                )
                _atomic_create(path, payload)
            derived.append(payload)
    # Exact componentwise nesting is reconstructed from retained candidate IDs.
    for failed in truth.failed_cases:
        rows = [row for row in derived if row["query_episode_id"] == failed.query_id]
        for smaller in rows:
            for larger in rows:
                if all(
                    smaller["route_quotas"][route] <= larger["route_quotas"][route]
                    for route in MAX_ROUTE_QUOTAS
                ) and not set(smaller["candidate_episode_ids"]).issubset(
                    larger["candidate_episode_ids"]
                ):
                    raise QuotaLadderError("derived candidate sets are not quota-nested")
    by_config = []
    for configuration in configurations:
        rows = [row for row in derived if row["configuration_name"] == configuration["name"]]
        retained = sum(row["recall"]["retained_count"] for row in rows)
        projected = truth.unaffected_retained_total + retained
        by_config.append({
            "configuration_ordinal": configuration["ordinal"],
            "configuration_name": configuration["name"],
            "route_quotas": configuration["route_quotas"],
            "failed_case_retained": retained, "failed_case_denominator": 80,
            "minimum_failed_case_retained": min(row["recall"]["retained_count"] for row in rows),
            "failed_cases_passing_19_of_20": sum(row["recall"]["passes_19_of_20"] for row in rows),
            "projected_frozen_60_retained": projected,
            "projected_frozen_60_denominator": 1_200,
            "projected_recall_gate_passed": projected >= 1_188 and all(
                row["recall"]["passes_19_of_20"] for row in rows
            ),
            "maximum_candidate_count": max(row["candidate_count"] for row in rows),
            "maximum_candidate_payload_bytes": max(row["candidate_payload_bytes"] for row in rows),
            "checkpoint_digests": [row["result_digest"] for row in rows],
        })
    deterministic = {
        "schema_version": SCHEMA_VERSION,
        "status": "post_open_development_poc_complete",
        "post_open_development_poc": True, "development_only": True,
        "diagnostic_truth_opened": True,
        "production_promotion_authorized": False,
        "policy_selection_authorized": False,
        "real_forward_outcomes_accessed": False,
        "poc_contract_digest": producer_inputs.poc_contract_digest,
        "registry_digest": truth.registry_digest,
        "producer_contract_digest": truth.producer_contract_digest,
        "global_results_opened_digest": truth.results_opened_digest,
        "poc_results_opened_digest": marker["result_digest"],
        "comparison_digest": truth.comparison_digest,
        "comparison_seal_digest": truth.comparison_seal_digest,
        "authority_seal_digest": truth.authority_seal_digest,
        "resident_content_digest": resident["content_digest"],
        "resident_ready_digest": resident["ready_digest"],
        "resident_stable_identity_digest": resident["stable_identity_digest"],
        "producer_max_scan_seal_digest": producer_seal["seal_digest"],
        "max_scan_digests": [row["result_digest"] for row in max_scans],
        "configuration_count": len(configurations),
        "derived_checkpoint_count": len(derived),
        "original_retained_total": truth.original_retained_total,
        "unaffected_retained_total": truth.unaffected_retained_total,
        "configurations": by_config,
        "derived_checkpoint_digests": [row["result_digest"] for row in derived],
        "progressive_widening_stopping_certificate": None,
        "claims": (
            "retrospective diagnostics only; a new untouched registry and exact "
            "stopping certificate are mandatory before policy selection"
        ),
    }
    expected = {
        **deterministic, "created_at": datetime.now(timezone.utc).isoformat(),
        "result_digest": stable_hash(deterministic),
    }
    final_path = output_root / "quota-ladder.json"
    if final_path.exists():
        observed = _read_json(final_path)
        if _without(observed, {"created_at"}) != _without(expected, {"created_at"}):
            raise QuotaLadderError("terminal quota ladder reconstruction differs")
        expected = observed
    else:
        _atomic_create(final_path, expected)
    allowed = {
        "POC_CONTRACT.json", "RESIDENT_READY.json", "MAX_SCAN_SEALED.json",
        "RESULTS_OPENED.json", "quota-ladder.json",
        *(str(_max_scan_path(output_root, case).relative_to(output_root))
          for case in producer_inputs.cases),
        *(str(_derived_checkpoint_path(output_root, failed, configuration).relative_to(output_root))
          for failed in truth.failed_cases for configuration in configurations),
    }
    _validate_output_tree(output_root, allowed)
    observed_files = {
        str(path.relative_to(output_root))
        for path in output_root.rglob("*") if path.is_file()
    }
    if observed_files != allowed or _tree_bytes(output_root) > MAX_OUTPUT_BYTES:
        raise QuotaLadderError("terminal comparator tree or budget differs")
    return expected


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--registry-root", type=Path, required=True)
        command.add_argument("--candidate-root", type=Path, required=True)
        command.add_argument("--source-full-root", type=Path, required=True)
        command.add_argument("--resident-root", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)

    common(commands.add_parser("produce"))
    compare = commands.add_parser("compare")
    common(compare)
    compare.add_argument("--authority-root", type=Path, required=True)
    compare.add_argument("--comparison-root", type=Path, required=True)
    args = parser.parse_args()
    input_roots = [
        args.registry_root.resolve(), args.candidate_root.resolve(),
        args.source_full_root.resolve(), args.resident_root.resolve(),
    ]
    if args.command == "compare":
        input_roots.extend((
            args.authority_root.resolve(), args.comparison_root.resolve(),
        ))
    output = args.output_root.resolve()
    if any(
        output == root or output.is_relative_to(root) or root.is_relative_to(output)
        for root in input_roots
    ):
        raise QuotaLadderError("quota output overlaps an input root")
    repository = Path(__file__).resolve().parents[2]
    producer_inputs, prereg = validate_producer_inputs(
        repository=repository, config_path=args.config,
        registry_root=args.registry_root, candidate_root=args.candidate_root,
        source_full_root=args.source_full_root, resident_root=args.resident_root,
        output_root=args.output_root,
    )
    if args.command == "produce":
        resident = validate_resident_once(producer_inputs)
        result = produce_max_evidence(
            config_path=args.config, output_root=args.output_root,
            inputs=producer_inputs, preregistration=prereg, resident=resident,
        )
    else:
        resident = _stable_resident_snapshot(
            _read_json(args.output_root.resolve() / "RESIDENT_READY.json")
        )
        if not all((
            resident["content_digest"] == producer_inputs.resident_content_digest,
            resident["ready_digest"] == producer_inputs.resident_ready_digest,
        )):
            raise QuotaLadderError("offline producer resident snapshot differs from preregistration")
        def load_truth() -> OpenedInputs:
            return validate_opened_inputs(
                registry_root=args.registry_root, candidate_root=args.candidate_root,
                authority_root=args.authority_root,
                comparison_root=args.comparison_root,
                source_full_root=args.source_full_root,
                resident_root=args.resident_root,
            )

        result = compare_producer_evidence(
            output_root=args.output_root, producer_inputs=producer_inputs,
            preregistration=prereg, resident=resident, truth_loader=load_truth,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
