"""Truth-blind M04R-13 producer for exposed-case threaded differentials.

The producer accepts no authority/comparison path.  It creates one fresh root,
runs four cases serially in fresh children, and seals exact search evidence.
Forward outcomes, comparison truth, and promotion are outside this process.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import importlib.metadata
import json
from math import isfinite
import os
import platform
from pathlib import Path
import resource
import stat
import subprocess
import sys
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

import pandas as pd

from market_analogues.adapters import source_from_spec
from market_analogues.certified_packed_search import (
    certified_packed_search, certified_packed_search_contract,
)
from market_analogues.config import load_config
from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.episodes import build_episode
from market_analogues.packed_bound_search import (
    BRANCH_AWARE_SEARCH_SCHEMA_VERSION, BoundProposal, BoundProposalReport,
    PackedBoundQuery, _packed_query_input_digest,
    bound_proposal_candidate_digest, packed_bound_search_contract,
    scan_packed_bound_proposals_threaded,
)
from market_analogues.representation import represent, representation_input_digest
from market_analogues.resident_store import (
    observe_ready_strict, prepare_resident_mirror_observed,
    resident_file_identity_lease,
)
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import InstrumentKey, SearchQuery, stable_hash


PREREG_SCHEMA = "m04r13-threaded-certified-exposed-preregistration-v1"
CASE_SCHEMA = "m04r13-threaded-certified-exposed-case-v1"
SEAL_SCHEMA = "m04r13-threaded-certified-exposed-producer-seal-v1"
FROZEN_QUERY_IDS = (
    "3307023dbe2164d025e788da", "3618af07dedd52fb3bdb1ccd",
    "9d7365581643bd93e85beb67", "99a0838725a09570b4a075ff",
)
FROZEN_CASE_IDS = (
    "nasdaq-JCTC-historical-252", "nasdaq-GBNY-current-252",
    "nasdaq-GBNY-historical-252", "nasdaq-ISPOW-current-252",
)
OUTPUT_RELATIVE = Path("config/data/analogues/m04r13/threaded-certified-exposed-v1")
PREREG_RELATIVE = Path(
    "experiments/m04r/m04r13_threaded_certified_exposed_preregistered.json"
)
DIAGNOSTIC_RELATIVE = Path(
    "config/data/analogues/m04r13/finite-threshold-diagnostic-v1/DIAGNOSTIC.json"
)
DIAGNOSTIC_SCHEMA = "m04r13-truth-blind-finite-threshold-diagnostic-v1"
DIAGNOSTIC_CASE_SCHEMA = "m04r13-truth-blind-finite-threshold-case-v1"
GENERATION_ID = "9fc6ae0ec4451133d8006897162f3443803fd30a3c44b78e80a476fbb18bb483"
PROVENANCE_DIGEST = "83ccfa62ac7ffec03e48d0f8a5634c7b8f1b8b0dde1426be22cc343fe116f62d"
REGISTRY_DIGEST = "0a4da732f91375a091775cb04e6e77c8d136ade47d7f4d16508a2d9a6555361e"
CONFIG_RELATIVE = Path("config/datasets.example.yaml")
REGISTRY_RELATIVE = Path("config/data/analogues/m04r10/nasdaq-untouched-authority-registry")
SOURCE_FULL_RELATIVE = Path("config/data/analogues/poc/m04r/packed-bound-full")
RESIDENT_ROOT = Path("/dev/shm/market-analogues/m04r11-candidate-v2") / GENERATION_ID
PROPOSAL_QUOTA = 16_385
PROPOSAL_THREADS = EXACT_WORKERS = 8
INITIAL_FRONTIER, MAXIMUM_FRONTIER, SEED_ROWS = 1_000, 16_384, 512
LOGICAL_FRONTIERS = (1_000, 2_000, 4_000, 8_000, 16_000, 16_384)
TOLERANCE = 1e-12
FORWARD_PROPOSAL_LIMIT_SECONDS = 120.0
REVERSE_PROPOSAL_LIMIT_SECONDS = 60.0
PROCESS_RSS_LIMIT_MB = 1_536.0
RESIDENT_RESERVE_BYTES = 1024 ** 3
COMPONENT_NAMES = {
    "stage", "price", "candle_volatility", "volume_shock", "market_context",
    "structural", "coarse",
}
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    "NUMBA_NUM_THREADS", "NUMBA_THREADING_LAYER",
)
RUNTIME_FILES = (
    "experiments/m04r/m04r13_threaded_certified_exposed.py",
    "experiments/m04r/compare_m04r13_threaded_certified_exposed.py",
    "experiments/m04r/verify_m04r13_threaded_certified_exposed.py",
    "experiments/m04r/m04r13_finite_threshold_diagnostic.py",
    "experiments/m04r/m04r12_quota_ladder_poc.py",
)


class HarnessError(ValueError):
    pass


@dataclass(frozen=True)
class CaseInput:
    ordinal: int
    registry_case: dict[str, Any]

    @property
    def query_id(self) -> str:
        return str(self.registry_case["episode_id"])

    @property
    def case_id(self) -> str:
        return str(self.registry_case["case_id"])


@dataclass(frozen=True)
class Inputs:
    repository: Path
    config_path: Path
    registry_root: Path
    source_store_root: Path
    resident_root: Path
    output_root: Path
    generation_id: str
    provenance_digest: str
    reserve_bytes: int
    registry_digest: str
    cases: tuple[CaseInput, ...]
    prereg_digest: str


def _without(value: Mapping[str, Any], omitted: set[str]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in omitted}


def _is_digest(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        return False
    try:
        return len(bytes.fromhex(value)) == 32
    except ValueError:
        return False


def _is_utc_iso_timestamp(value: Any) -> bool:
    if type(value) is not str:
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    offset = parsed.utcoffset()
    return all((
        parsed.tzinfo is not None,
        offset is not None,
        offset is not None and offset.total_seconds() == 0,
        parsed.isoformat() == value,
    ))


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024): digest.update(block)
    return digest.hexdigest()


def _read_json_sha(path: Path) -> tuple[dict[str, Any], str]:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise HarnessError(f"cannot open JSON: {path}") from exc
    try:
        before, chunks = os.fstat(descriptor), []
        while block := os.read(descriptor, 1024 * 1024): chunks.append(block)
        after = os.fstat(descriptor)
    finally: os.close(descriptor)
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
        value.st_ctime_ns, value.st_mode,
    )
    if not stat.S_ISREG(before.st_mode) or identity(before) != identity(after):
        raise HarnessError(f"JSON identity differs: {path}")
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise HarnessError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    def invalid_constant(value: str) -> Any:
        raise HarnessError(f"non-finite JSON number: {value}")

    raw = b"".join(chunks)
    try:
        value = json.loads(
            raw, object_pairs_hook=object_pairs,
            parse_constant=invalid_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HarnessError(f"invalid JSON: {path}") from exc
    if type(value) is not dict: raise HarnessError(f"JSON object differs: {path}")
    return value, sha256(raw).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return _read_json_sha(path)[0]


def _atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write((json.dumps(
                dict(payload), indent=2, sort_keys=True, allow_nan=False,
            ) + "\n").encode())
            handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
    finally:
        if temporary.exists(): temporary.unlink()


def _rss() -> float:
    current = 0.0
    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        current = pages * os.sysconf("SC_PAGE_SIZE") / 1024 ** 2
    except (OSError, ValueError, IndexError):
        pass
    observed_high_water = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    high_water = observed_high_water / (1024 ** 2 if sys.platform == "darwin" else 1024)
    return max(current, high_water)


def _run_git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments], cwd=repository, text=True,
        capture_output=True, check=False,
    )


def _runtime_files(repository: Path) -> tuple[str, ...]:
    tracked = _run_git(repository, "ls-files", "--", "src/market_analogues/*.py")
    if tracked.returncode:
        raise HarnessError("cannot enumerate implementation modules")
    return tuple(dict.fromkeys((*RUNTIME_FILES, *tracked.stdout.splitlines())))


def _implementation_git(repository: Path) -> dict[str, Any]:
    """Capture the clean code commit immediately before preregistration."""
    runtime_files = _runtime_files(repository)
    checks = (
        ("ls-files", "--error-unmatch", "--", *runtime_files),
        ("diff", "--quiet"), ("diff", "--cached", "--quiet"),
    )
    if any(_run_git(repository, *command).returncode for command in checks):
        raise HarnessError("preregistration requires clean tracked implementation")
    prereg = repository / PREREG_RELATIVE
    if prereg.exists() or prereg.is_symlink():
        raise HarnessError("preregistration is create-only")
    head = _run_git(repository, "rev-parse", "HEAD")
    if head.returncode or not head.stdout.strip():
        raise HarnessError("cannot resolve implementation commit")
    files = {path: _sha(repository / path) for path in runtime_files}
    deterministic = {
        "implementation_commit": head.stdout.strip(),
        "files": files, "files_digest": stable_hash(files),
    }
    return {**deterministic, "digest": stable_hash(deterministic)}


def _launch_git(repository: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    """Require the clean, unique commit that introduced the preregistration."""
    runtime_files = _runtime_files(repository)
    paths = [*runtime_files, str(PREREG_RELATIVE), str(DIAGNOSTIC_RELATIVE)]
    checks = (
        ("ls-files", "--error-unmatch", "--", *paths),
        ("diff", "--quiet"), ("diff", "--cached", "--quiet"),
    )
    if any(_run_git(repository, *command).returncode for command in checks):
        raise HarnessError("M04R-13 launch requires a clean tracked tree")
    result = _run_git(
        repository, "log", "--diff-filter=A", "--format=%H", "--",
        str(PREREG_RELATIVE),
    )
    introductions = [row for row in result.stdout.splitlines() if row]
    head = _run_git(repository, "rev-parse", "HEAD").stdout.strip()
    diagnostic_result = _run_git(
        repository, "log", "--diff-filter=A", "--format=%H", "--",
        str(DIAGNOSTIC_RELATIVE),
    )
    diagnostic_introductions = [
        row for row in diagnostic_result.stdout.splitlines() if row
    ]
    if (result.returncode or diagnostic_result.returncode
            or len(introductions) != 1 or introductions[0] != head
            or diagnostic_introductions != [head]):
        raise HarnessError("launch HEAD is not the unique preregistration commit")
    parent = _run_git(repository, "rev-parse", f"{head}^").stdout.strip()
    files = {path: _sha(repository / path) for path in runtime_files}
    deterministic = {
        "implementation_commit": parent,
        "files": files, "files_digest": stable_hash(files),
    }
    observed = {**deterministic, "digest": stable_hash(deterministic)}
    if dict(expected) != observed:
        raise HarnessError("implementation differs from preregistration")
    return observed


def _validate_committed_diagnostic_blob(
    repository: Path, binding: Mapping[str, Any],
) -> None:
    head = _run_git(repository, "rev-parse", "HEAD").stdout.strip()
    completed = subprocess.run(
        ["git", "cat-file", "blob", f"{head}:{DIAGNOSTIC_RELATIVE}"],
        cwd=repository, capture_output=True, check=False,
    )
    if not all((
        completed.returncode == 0,
        sha256(completed.stdout).hexdigest() == binding.get("sha256"),
        _sha(repository / DIAGNOSTIC_RELATIVE) == binding.get("sha256"),
    )):
        raise HarnessError("committed finite-threshold diagnostic differs")


def _m12(repository: Path) -> Any:
    path = repository / "experiments/m04r/m04r12_quota_ladder_poc.py"
    spec = importlib.util.spec_from_file_location("m04r13_registry_validator", path)
    if spec is None or spec.loader is None: raise HarnessError("registry validator absent")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module


def _registry_cases(repository: Path, root: Path) -> tuple[str, tuple[CaseInput, ...]]:
    module = _m12(repository)
    registry = _read_json(root / "query-registry.json")
    cases = module._validate_registry(registry)
    by_id = {str(case["episode_id"]): case for case in cases}
    selected = tuple(CaseInput(index, dict(by_id[value]))
                     for index, value in enumerate(FROZEN_QUERY_IDS))
    if tuple(case.case_id for case in selected) != FROZEN_CASE_IDS:
        raise HarnessError("frozen registry case identities differ")
    return str(registry["registry_digest"]), selected


def resident_full(
    source_store_root: Path, resident_root: Path, generation_id: str,
    provenance_digest: str, reserve_bytes: int,
) -> dict[str, Any]:
    ready, validation = prepare_resident_mirror_observed(
        source_store_root, resident_root, generation_id,
        expected_provenance_digest=provenance_digest,
        reserve_bytes=reserve_bytes, validate_existing=True,
    )
    observation = observe_ready_strict(resident_root / "READY.json")
    lease = resident_file_identity_lease(resident_root / "READY.json")
    if not all((observation["payload"] == ready,
                validation["content_digest"] == ready["content_digest"],
                lease["ready_digest"] == ready["ready_digest"])):
        raise HarnessError("resident validation differs")
    value = {
        "ready_digest": ready["ready_digest"],
        "content_digest": ready["content_digest"],
        "seal_digest": ready["seal_digest"],
        "ready_file_sha256": observation["ready_file_sha256"],
        "lease": lease, "store_root": str((resident_root / "store").resolve()),
    }
    result = {**value, "identity_digest": stable_hash(value)}
    validate_resident_snapshot(result)
    return result


def validate_resident_snapshot(value: Mapping[str, Any]) -> None:
    if set(value) != {
        "ready_digest", "content_digest", "seal_digest", "ready_file_sha256",
        "lease", "store_root", "identity_digest",
    }:
        raise HarnessError("resident snapshot fields differ")
    lease = value["lease"]
    if type(lease) is not dict or set(lease) != {
        "schema_version", "ready_digest", "ready_file_sha256", "content_digest",
        "files", "lease_digest",
    }:
        raise HarnessError("resident lease fields differ")
    files = lease["files"]
    identity_keys = {
        "path", "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns",
        "st_mode",
    }
    if type(files) is not dict or not files or any(
        type(identity) is not dict or set(identity) != identity_keys
        or not all(_nonnegative_int(identity[key]) for key in identity_keys - {"path"})
        or not isinstance(identity["path"], str)
        for identity in files.values()
    ):
        raise HarnessError("resident file identities differ")
    if not all((
        lease["ready_digest"] == value["ready_digest"],
        lease["ready_file_sha256"] == value["ready_file_sha256"],
        lease["content_digest"] == value["content_digest"],
        lease["lease_digest"] == stable_hash(_without(lease, {"lease_digest"})),
        value["identity_digest"] == stable_hash(_without(value, {"identity_digest"})),
        all(isinstance(value[key], str) and value[key]
            for key in ("ready_digest", "content_digest", "seal_digest",
                        "ready_file_sha256", "store_root")),
    )):
        raise HarnessError("resident snapshot digest differs")


def lease_exact(inputs: Inputs, resident: Mapping[str, Any]) -> str:
    validate_resident_snapshot(resident)
    observation = observe_ready_strict(inputs.resident_root / "READY.json")
    lease = resident_file_identity_lease(inputs.resident_root / "READY.json")
    value = {
        "ready_digest": observation["ready_digest"],
        "content_digest": observation["content_digest"],
        "seal_digest": observation["seal_digest"],
        "ready_file_sha256": observation["ready_file_sha256"],
        "lease": lease, "store_root": str((inputs.resident_root / "store").resolve()),
    }
    if stable_hash(value) != resident["identity_digest"]:
        raise HarnessError("resident lease changed")
    return str(lease["lease_digest"])


def preregister_payload(
    *, repository: Path, config_path: Path, registry_root: Path,
    source_full_root: Path, resident_root: Path, output_root: Path,
    generation_id: str, provenance_digest: str, reserve_bytes: int,
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise HarnessError("M04R-13 output must be absent at preregistration")
    if not all((
        generation_id == GENERATION_ID, provenance_digest == PROVENANCE_DIGEST,
        reserve_bytes == RESIDENT_RESERVE_BYTES,
        config_path.resolve() == (repository / CONFIG_RELATIVE).resolve(),
        registry_root.resolve() == (repository / REGISTRY_RELATIVE).resolve(),
        source_full_root.resolve() == (repository / SOURCE_FULL_RELATIVE).resolve(),
        resident_root.resolve() == RESIDENT_ROOT.resolve(),
        output_root.resolve() == (repository / OUTPUT_RELATIVE).resolve(),
    )):
        raise HarnessError("frozen generation or resident reserve differs")
    git = _implementation_git(repository)
    registry_digest, cases = _registry_cases(repository, registry_root)
    resident = resident_full(
        source_full_root / "store", resident_root, generation_id,
        provenance_digest, reserve_bytes,
    )
    registry_cases_digest = stable_hash([case.registry_case for case in cases])
    diagnostic_inputs = Inputs(
        repository.resolve(), config_path.resolve(), registry_root.resolve(),
        source_full_root.resolve() / "store", resident_root.resolve(),
        output_root.resolve(), generation_id, provenance_digest, reserve_bytes,
        registry_digest, cases, "diagnostic-preregistration",
    )
    finite_diagnostic = finite_threshold_diagnostic_binding(
        repository, expected_git=git,
        registry_cases_digest=registry_cases_digest, resident=resident,
        expected_query_bindings=diagnostic_query_bindings(diagnostic_inputs),
    )
    deterministic = {
        "schema_version": PREREG_SCHEMA, "status": "frozen_before_producer",
        "development_only": True, "truth_roots_allowed_in_producer": False,
        "production_promotion_authorized": False,
        "real_forward_outcomes_accessed": False,
        "git": git, "config_path": str(config_path.resolve()),
        "config_sha256": _sha(config_path),
        "roots": {
            "registry_root": str(registry_root.resolve()),
            "source_full_root": str(source_full_root.resolve()),
            "resident_root": str(resident_root.resolve()),
            "output_root": str(output_root.resolve()),
        },
        "generation_id": generation_id, "provenance_digest": provenance_digest,
        "reserve_bytes": reserve_bytes, "registry_digest": registry_digest,
        "query_ids": list(FROZEN_QUERY_IDS),
        "registry_cases_digest": registry_cases_digest,
        "resident_content_digest": resident["content_digest"],
        "resident_ready_digest": resident["ready_digest"],
        "resident_identity_digest": resident["identity_digest"],
        "finite_threshold_diagnostic": finite_diagnostic,
        "execution": _execution_policy(), "environment": _environment_binding(),
    }
    payload = {**deterministic, "preregistration_digest": stable_hash(deterministic)}
    validate_preregistration_shape(payload)
    return payload


def _execution_policy() -> dict[str, Any]:
    return {
        "child_processes": "fresh_serial_spawn", "parent_max_workers": 1,
        "proposal_threads": PROPOSAL_THREADS, "exact_workers": EXACT_WORKERS,
        "forward_block_rows": 4_096, "reverse_block_rows": 4_097,
        "route_quotas": {"composite": PROPOSAL_QUOTA},
        "branch_aware": True, "initial_frontier_rows": INITIAL_FRONTIER,
        "maximum_frontier_rows": MAXIMUM_FRONTIER,
        "logical_frontiers": list(LOGICAL_FRONTIERS), "seed_rows": SEED_ROWS,
        "tolerance": TOLERANCE, "streaming_fallback": True,
        "performance_limits": {
            "forward_proposal_seconds": FORWARD_PROPOSAL_LIMIT_SECONDS,
            "reverse_proposal_seconds": REVERSE_PROPOSAL_LIMIT_SECONDS,
            "process_rss_mb": PROCESS_RSS_LIMIT_MB,
        },
    }


def _environment_binding() -> dict[str, Any]:
    deterministic = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(), "machine": platform.machine(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "numba", "pandas")
        },
        "thread_environment": {key: os.environ.get(key) for key in THREAD_ENV_KEYS},
    }
    return {**deterministic, "digest": stable_hash(deterministic)}


def _finite_hex(value: Any, *, optional: bool = False) -> bool:
    if value is None:
        return optional
    if type(value) is not str:
        return False
    try:
        parsed = float.fromhex(value)
    except ValueError:
        return False
    return isfinite(parsed) and parsed.hex() == value


def validate_finite_threshold_diagnostic(
    payload: Mapping[str, Any], *, repository: Path,
    expected_git: Mapping[str, Any], registry_cases_digest: str,
    resident: Mapping[str, Any], expected_query_bindings: Sequence[Mapping[str, Any]],
) -> None:
    required = {
        "schema_version", "status", "development_only",
        "authority_or_outcome_paths_accepted", "production_promotion_authorized",
        "real_forward_outcomes_accessed", "implementation_git", "environment",
        "execution", "diagnostic_path", "roots", "generation_id",
        "provenance_digest", "registry_digest", "registry_cases_digest",
        "config_sha256",
        "query_ids", "case_ids", "resident", "cases",
        "all_thresholds_finite", "all_semantic_checks_passed", "created_at",
        "result_digest",
    }
    roots = payload.get("roots")
    resident_binding = payload.get("resident")
    cases = payload.get("cases")
    deterministic = _without(payload, {"created_at", "result_digest"})
    if not all((
        set(payload) == required,
        payload.get("schema_version") == DIAGNOSTIC_SCHEMA,
        payload.get("status") == "truth_blind_finite_thresholds_verified",
        payload.get("development_only") is True,
        payload.get("authority_or_outcome_paths_accepted") is False,
        payload.get("production_promotion_authorized") is False,
        payload.get("real_forward_outcomes_accessed") is False,
        payload.get("implementation_git") == dict(expected_git),
        payload.get("environment") == _environment_binding(),
        payload.get("execution") == _execution_policy(),
        payload.get("config_sha256") == _sha(repository / CONFIG_RELATIVE),
        payload.get("diagnostic_path")
        == str((repository / DIAGNOSTIC_RELATIVE).resolve()),
        type(roots) is dict,
        roots == {
            "config_path": str((repository / CONFIG_RELATIVE).resolve()),
            "registry_root": str((repository / REGISTRY_RELATIVE).resolve()),
            "source_full_root": str((repository / SOURCE_FULL_RELATIVE).resolve()),
            "resident_root": str(RESIDENT_ROOT.resolve()),
        },
        payload.get("generation_id") == GENERATION_ID,
        payload.get("provenance_digest") == PROVENANCE_DIGEST,
        payload.get("registry_digest") == REGISTRY_DIGEST,
        payload.get("registry_cases_digest") == registry_cases_digest,
        payload.get("query_ids") == list(FROZEN_QUERY_IDS),
        payload.get("case_ids") == list(FROZEN_CASE_IDS),
        type(resident_binding) is dict,
        resident_binding == {
            "content_digest": resident["content_digest"],
            "ready_digest": resident["ready_digest"],
            "identity_digest": resident["identity_digest"],
        },
        type(cases) is list and len(cases) == len(FROZEN_QUERY_IDS),
        len(expected_query_bindings) == len(FROZEN_QUERY_IDS),
        payload.get("all_thresholds_finite") is True,
        payload.get("all_semantic_checks_passed") is True,
        _is_utc_iso_timestamp(payload.get("created_at")),
        payload.get("result_digest") == stable_hash(deterministic),
    )):
        raise HarnessError("finite-threshold diagnostic differs")
    case_keys = {
        "schema_version", "ordinal", "registry_case_id", "query_episode_id",
        "forward_candidate_digest", "forward_result_digest",
        "reverse_candidate_digest", "reverse_result_digest",
        "certificate_result_digest", "case_result_digest", "rounds",
        "query_binding", "stop_threshold_hex", "next_lower_bound_hex",
        "minimum_native_pruned_bound_hex",
        "threshold_closure_passes", "semantic_passed",
        "performance_passed_diagnostic", "all_thresholds_finite",
        "real_forward_outcomes_accessed", "result_digest",
    }
    round_keys = {
        "frontier_rows", "selected_rows", "constrained_threshold_hex",
        "next_lower_bound_hex", "certified",
    }
    closure_keys = {
        "lower_exclusive_hex", "upper_inclusive_hex", "admitted_rows",
        "selected_rows", "resulting_threshold_hex",
        "minimum_packed_unclassified_bound_hex",
        "minimum_native_pruned_bound_hex", "certified",
    }
    for ordinal, case in enumerate(cases):
        rounds = case.get("rounds") if type(case) is dict else None
        closures = case.get("threshold_closure_passes") if type(case) is dict else None
        digests = (
            "forward_candidate_digest", "forward_result_digest",
            "reverse_candidate_digest", "reverse_result_digest",
            "certificate_result_digest", "case_result_digest", "result_digest",
        )
        valid_rounds = type(rounds) is list and bool(rounds) and all(
            type(row) is dict and set(row) == round_keys
            and type(row["frontier_rows"]) is int and row["frontier_rows"] > 0
            and type(row["selected_rows"]) is int and 0 <= row["selected_rows"] <= 20
            and _finite_hex(row["constrained_threshold_hex"])
            and _finite_hex(row["next_lower_bound_hex"], optional=True)
            and type(row["certified"]) is bool
            for row in rounds
        )
        valid_closures = type(closures) is list and all(
            type(row) is dict and set(row) == closure_keys
            and _finite_hex(row["lower_exclusive_hex"], optional=True)
            and _finite_hex(row["upper_inclusive_hex"])
            and type(row["admitted_rows"]) is int and row["admitted_rows"] >= 0
            and type(row["selected_rows"]) is int and 0 <= row["selected_rows"] <= 20
            and _finite_hex(row["resulting_threshold_hex"])
            and _finite_hex(
                row["minimum_packed_unclassified_bound_hex"], optional=True,
            )
            and _finite_hex(row["minimum_native_pruned_bound_hex"], optional=True)
            and type(row["certified"]) is bool
            for row in closures
        )
        if not all((
            type(case) is dict, set(case) == case_keys,
            case.get("schema_version") == DIAGNOSTIC_CASE_SCHEMA,
            case.get("ordinal") == ordinal,
            case.get("registry_case_id") == FROZEN_CASE_IDS[ordinal],
            case.get("query_episode_id") == FROZEN_QUERY_IDS[ordinal],
            case.get("query_binding") == dict(expected_query_bindings[ordinal]),
            all(_is_digest(case.get(key)) for key in digests),
            case.get("forward_candidate_digest")
            == case.get("reverse_candidate_digest"),
            valid_rounds, valid_closures,
            _finite_hex(case.get("stop_threshold_hex")),
            _finite_hex(case.get("next_lower_bound_hex"), optional=True),
            _finite_hex(
                case.get("minimum_native_pruned_bound_hex"), optional=True,
            ),
            case.get("semantic_passed") is True,
            type(case.get("performance_passed_diagnostic")) is bool,
            case.get("all_thresholds_finite") is True,
            case.get("real_forward_outcomes_accessed") is False,
            case.get("result_digest") == stable_hash(
                _without(case, {"result_digest"})
            ),
        )):
            raise HarnessError("finite-threshold diagnostic case differs")
        final_round = rounds[-1]
        if closures:
            final_closure = closures[-1]
            terminal = all((
                final_round["certified"] is False,
                all(row["certified"] is False for row in rounds),
                all(row["certified"] is False for row in closures[:-1]),
                final_closure["certified"] is True,
                final_closure["selected_rows"] == 20,
                final_closure["resulting_threshold_hex"]
                == case["stop_threshold_hex"],
                case["next_lower_bound_hex"] == min(
                    (value for value in (
                        final_closure["minimum_packed_unclassified_bound_hex"],
                        final_closure["minimum_native_pruned_bound_hex"],
                    ) if value is not None),
                    key=float.fromhex, default=None,
                ),
                closures[0]["lower_exclusive_hex"] is None,
                closures[0]["upper_inclusive_hex"] == (
                    float.fromhex(final_round["constrained_threshold_hex"])
                    + TOLERANCE
                ).hex(),
                all(
                    row["upper_inclusive_hex"] == (
                        float.fromhex(prior["resulting_threshold_hex"])
                        + TOLERANCE
                    ).hex()
                    and float.fromhex(row["upper_inclusive_hex"])
                    > float.fromhex(prior["upper_inclusive_hex"])
                    for prior, row in zip(closures, closures[1:])
                ),
            ))
        else:
            terminal = all((
                all(row["certified"] is False for row in rounds[:-1]),
                final_round["certified"] is True,
                final_round["selected_rows"] == 20,
                final_round["constrained_threshold_hex"]
                == case["stop_threshold_hex"],
                final_round["next_lower_bound_hex"]
                == case["next_lower_bound_hex"],
            ))
        frontiers = [row["frontier_rows"] for row in rounds]
        distinct = list(dict.fromkeys(frontiers))
        if not all((
            terminal,
            distinct == list(LOGICAL_FRONTIERS)[:len(distinct)],
            all(left <= right for left, right in zip(frontiers, frontiers[1:])),
            all(
                row["lower_exclusive_hex"]
                == (None if index == 0 else closures[index - 1]["upper_inclusive_hex"])
                for index, row in enumerate(closures)
            ),
        )):
            raise HarnessError("finite-threshold terminal state differs")


def finite_threshold_diagnostic_binding(
    repository: Path, *, expected_git: Mapping[str, Any],
    registry_cases_digest: str, resident: Mapping[str, Any],
    expected_query_bindings: Sequence[Mapping[str, Any]],
    expected_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = repository / DIAGNOSTIC_RELATIVE
    payload, file_sha256 = _read_json_sha(path)
    validate_finite_threshold_diagnostic(
        payload, repository=repository, expected_git=expected_git,
        registry_cases_digest=registry_cases_digest, resident=resident,
        expected_query_bindings=expected_query_bindings,
    )
    binding = {
        "path": str(path.resolve()), "sha256": file_sha256,
        "result_digest": payload["result_digest"],
    }
    if expected_binding is not None and dict(expected_binding) != binding:
        raise HarnessError("finite-threshold diagnostic binding differs")
    return binding


def validate_preregistration_shape(
    prereg: Mapping[str, Any], *, enforce_topology: bool = True,
) -> None:
    required = {
        "schema_version", "status", "development_only",
        "truth_roots_allowed_in_producer", "production_promotion_authorized",
        "real_forward_outcomes_accessed", "git", "config_path", "config_sha256",
        "roots", "generation_id", "provenance_digest", "reserve_bytes",
        "registry_digest", "query_ids", "registry_cases_digest",
        "resident_content_digest", "resident_ready_digest",
        "resident_identity_digest", "finite_threshold_diagnostic",
        "execution", "environment",
        "preregistration_digest",
    }
    roots = prereg.get("roots")
    git = prereg.get("git")
    if not all((
        set(prereg) == required,
        prereg.get("schema_version") == PREREG_SCHEMA,
        prereg.get("status") == "frozen_before_producer",
        prereg.get("development_only") is True,
        prereg.get("truth_roots_allowed_in_producer") is False,
        prereg.get("production_promotion_authorized") is False,
        prereg.get("real_forward_outcomes_accessed") is False,
        type(roots) is dict,
        set(roots or {}) == {"registry_root", "source_full_root", "resident_root",
                             "output_root"},
        all(isinstance(value, str) and Path(value).is_absolute()
            for value in (roots or {}).values()),
        type(git) is dict,
        set(git or {}) == {"implementation_commit", "files", "files_digest", "digest"},
        isinstance((git or {}).get("implementation_commit"), str),
        type((git or {}).get("files")) is dict,
        (git or {}).get("files_digest") == stable_hash((git or {}).get("files", {})),
        (git or {}).get("digest") == stable_hash(_without(git or {}, {"digest"})),
        all(isinstance(path, str) and _is_digest(digest)
            for path, digest in (git or {}).get("files", {}).items()),
        isinstance(prereg.get("config_path"), str)
        and Path(str(prereg.get("config_path"))).is_absolute(),
        all(_is_digest(prereg.get(key)) for key in (
            "config_sha256", "generation_id", "provenance_digest", "registry_digest",
            "registry_cases_digest", "resident_content_digest",
            "resident_ready_digest", "resident_identity_digest",
            "preregistration_digest",
        )),
        prereg.get("generation_id") == GENERATION_ID,
        prereg.get("reserve_bytes") == RESIDENT_RESERVE_BYTES,
        prereg.get("query_ids") == list(FROZEN_QUERY_IDS),
        type(prereg.get("finite_threshold_diagnostic")) is dict,
        set(prereg.get("finite_threshold_diagnostic", {}))
        == {"path", "sha256", "result_digest"},
        type(prereg.get("finite_threshold_diagnostic", {}).get("path")) is str,
        Path(prereg.get("finite_threshold_diagnostic", {}).get("path", "")).is_absolute(),
        all(_is_digest(prereg.get("finite_threshold_diagnostic", {}).get(key))
            for key in ("sha256", "result_digest")),
        prereg.get("execution") == _execution_policy(),
        prereg.get("environment") == _environment_binding(),
        prereg.get("preregistration_digest")
        == stable_hash(_without(prereg, {"preregistration_digest"})),
    )):
        raise HarnessError("M04R-13 preregistration schema differs")
    if enforce_topology:
        repository = Path(__file__).resolve().parents[2]
        expected_roots = {
            "registry_root": str((repository / REGISTRY_RELATIVE).resolve()),
            "source_full_root": str((repository / SOURCE_FULL_RELATIVE).resolve()),
            "resident_root": str(RESIDENT_ROOT.resolve()),
            "output_root": str((repository / OUTPUT_RELATIVE).resolve()),
        }
        resolved = [Path(value).resolve() for value in expected_roots.values()]
        resolved.append((repository / DIAGNOSTIC_RELATIVE).resolve())
        disjoint = all(
            not (left == right or left.is_relative_to(right) or right.is_relative_to(left))
            for index, left in enumerate(resolved) for right in resolved[index + 1:]
        )
        if not all((
            prereg["config_path"] == str((repository / CONFIG_RELATIVE).resolve()),
            prereg["roots"] == expected_roots,
            prereg["registry_digest"] == REGISTRY_DIGEST,
            prereg["provenance_digest"] == PROVENANCE_DIGEST,
            prereg["finite_threshold_diagnostic"]["path"]
            == str((repository / DIAGNOSTIC_RELATIVE).resolve()),
            disjoint,
        )):
            raise HarnessError("M04R-13 preregistration topology differs")


def load_inputs(
    *, repository: Path, config_path: Path, registry_root: Path,
    source_full_root: Path, resident_root: Path, output_root: Path,
) -> tuple[Inputs, dict[str, Any], dict[str, Any]]:
    prereg = _read_json(repository / PREREG_RELATIVE)
    validate_preregistration_shape(prereg)
    git = _launch_git(repository, prereg.get("git", {}))
    _validate_committed_diagnostic_blob(
        repository, prereg.get("finite_threshold_diagnostic", {}),
    )
    registry_digest, cases = _registry_cases(repository, registry_root)
    roots = prereg.get("roots", {})
    deterministic = _without(prereg, {"preregistration_digest"})
    required = {
        "schema_version", "status", "development_only",
        "truth_roots_allowed_in_producer", "production_promotion_authorized",
        "real_forward_outcomes_accessed", "git", "config_path", "config_sha256",
        "roots", "generation_id", "provenance_digest", "reserve_bytes",
        "registry_digest", "query_ids", "registry_cases_digest",
        "resident_content_digest", "resident_ready_digest",
        "resident_identity_digest", "finite_threshold_diagnostic",
        "execution", "environment",
        "preregistration_digest",
    }
    if not all((
        set(prereg) == required,
        prereg.get("schema_version") == PREREG_SCHEMA,
        prereg.get("status") == "frozen_before_producer",
        prereg.get("truth_roots_allowed_in_producer") is False,
        prereg.get("production_promotion_authorized") is False,
        prereg.get("real_forward_outcomes_accessed") is False,
        prereg.get("git") == git,
        prereg.get("config_path") == str(config_path.resolve()),
        prereg.get("config_sha256") == _sha(config_path),
        roots == {
            "registry_root": str(registry_root.resolve()),
            "source_full_root": str(source_full_root.resolve()),
            "resident_root": str(resident_root.resolve()),
            "output_root": str(output_root.resolve()),
        },
        output_root.resolve() == (repository / OUTPUT_RELATIVE).resolve(),
        prereg.get("generation_id") == GENERATION_ID,
        prereg.get("reserve_bytes") == RESIDENT_RESERVE_BYTES,
        prereg.get("execution") == _execution_policy(),
        prereg.get("environment") == _environment_binding(),
        prereg.get("registry_digest") == registry_digest,
        prereg.get("query_ids") == list(FROZEN_QUERY_IDS),
        tuple(case.case_id for case in cases) == FROZEN_CASE_IDS,
        prereg.get("registry_cases_digest")
        == stable_hash([case.registry_case for case in cases]),
        prereg.get("preregistration_digest") == stable_hash(deterministic),
    )): raise HarnessError("M04R-13 preregistration differs")
    resident = resident_full(
        source_full_root / "store", resident_root, GENERATION_ID,
        str(prereg["provenance_digest"]), prereg["reserve_bytes"],
    )
    if not all((resident["content_digest"] == prereg["resident_content_digest"],
                resident["ready_digest"] == prereg["resident_ready_digest"],
                resident["identity_digest"] == prereg["resident_identity_digest"])):
        raise HarnessError("resident differs from preregistration")
    inputs = Inputs(
        repository.resolve(), config_path.resolve(), registry_root.resolve(),
        source_full_root.resolve() / "store", resident_root.resolve(),
        output_root.resolve(), GENERATION_ID, str(prereg["provenance_digest"]),
        prereg["reserve_bytes"], registry_digest, cases,
        str(prereg["preregistration_digest"]),
    )
    finite_threshold_diagnostic_binding(
        repository, expected_git=git,
        registry_cases_digest=str(prereg["registry_cases_digest"]),
        resident=resident,
        expected_query_bindings=diagnostic_query_bindings(inputs),
        expected_binding=prereg["finite_threshold_diagnostic"],
    )
    return inputs, prereg, resident


def _proposal_payload(report: BoundProposalReport) -> dict[str, Any]:
    return {
        "schema_version": report.schema_version, "generation_id": report.generation_id,
        "query_episode_id": report.query_episode_id,
        "candidates": [{
            "episode_id": row.episode_id, "symbol": row.symbol,
            "cutoff_ns": row.cutoff_ns, "quality_tier": row.quality_tier,
            "lower_bound_hex": row.lower_bound.hex(), "routes": list(row.routes),
            "overflow_fallback": row.overflow_fallback,
        } for row in report.candidates],
        "rows_scanned": report.rows_scanned, "eligible_rows": report.eligible_rows,
        "eligible_main_rows": report.eligible_main_rows,
        "eligible_overflow_rows": report.eligible_overflow_rows,
        "route_counts": dict(report.route_counts),
        "route_quotas": dict(report.route_quotas), "block_rows": report.block_rows,
        "block_order": report.block_order, "elapsed_seconds": report.elapsed_seconds,
        "peak_rss_mb": report.peak_rss_mb,
        "candidate_digest": report.candidate_digest,
        "result_digest": report.result_digest,
        "contract_digest": report.contract_digest, "input_digest": report.input_digest,
    }


def _proposal_report(value: Mapping[str, Any]) -> BoundProposalReport:
    _validate_proposal_json_types(value)
    candidates = tuple(BoundProposal(
        row["episode_id"], row["symbol"], row["cutoff_ns"],
        row["quality_tier"], float.fromhex(row["lower_bound_hex"]),
        tuple(row["routes"]), row["overflow_fallback"],
    ) for row in value["candidates"])
    return BoundProposalReport(
        value["schema_version"], value["generation_id"], value["query_episode_id"],
        candidates, value["rows_scanned"], value["eligible_rows"],
        value["eligible_main_rows"], value["eligible_overflow_rows"],
        dict(value["route_counts"]), dict(value["route_quotas"]),
        value["block_rows"], value["block_order"],
        value["elapsed_seconds"], value["peak_rss_mb"],
        value["candidate_digest"], value["result_digest"],
        value["contract_digest"], value["input_digest"],
    )


def _number(value: Any) -> bool:
    return type(value) in {int, float} and isfinite(value)


def _validate_proposal_json_types(value: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "generation_id", "query_episode_id", "candidates",
        "rows_scanned", "eligible_rows", "eligible_main_rows",
        "eligible_overflow_rows", "route_counts", "route_quotas", "block_rows",
        "block_order", "elapsed_seconds", "peak_rss_mb", "candidate_digest",
        "result_digest", "contract_digest", "input_digest",
    }
    candidate_keys = {
        "episode_id", "symbol", "cutoff_ns", "quality_tier", "lower_bound_hex",
        "routes", "overflow_fallback",
    }
    if not all((
        type(value) is dict, set(value) == required,
        all(type(value[key]) is str for key in (
            "schema_version", "generation_id", "query_episode_id", "block_order",
            "candidate_digest", "result_digest", "contract_digest", "input_digest",
        )),
        all(_nonnegative_int(value[key]) for key in (
            "rows_scanned", "eligible_rows", "eligible_main_rows",
            "eligible_overflow_rows", "block_rows",
        )),
        _number(value["elapsed_seconds"]) and value["elapsed_seconds"] >= 0,
        _number(value["peak_rss_mb"]) and value["peak_rss_mb"] >= 0,
        type(value["route_counts"]) is dict,
        type(value["route_quotas"]) is dict,
        set(value["route_counts"]) == {"composite"},
        set(value["route_quotas"]) == {"composite"},
        _nonnegative_int(value["route_counts"]["composite"]),
        _nonnegative_int(value["route_quotas"]["composite"]),
        type(value["candidates"]) is list,
    )):
        raise HarnessError("proposal raw JSON types differ")
    for row in value["candidates"]:
        if not all((
            type(row) is dict, set(row) == candidate_keys,
            type(row["episode_id"]) is str, type(row["symbol"]) is str,
            _nonnegative_int(row["cutoff_ns"]),
            type(row["quality_tier"]) is str,
            type(row["lower_bound_hex"]) is str,
            type(row["routes"]) is list and row["routes"] == ["composite"],
            type(row["overflow_fallback"]) is bool,
        )):
            raise HarnessError("proposal candidate raw JSON types differ")
        try:
            lower = float.fromhex(row["lower_bound_hex"])
        except ValueError as exc:
            raise HarnessError("proposal lower-bound encoding differs") from exc
        if (not isfinite(lower) or lower < 0 or row["lower_bound_hex"].startswith("-")
                or lower.hex() != row["lower_bound_hex"]):
            raise HarnessError("proposal lower-bound encoding differs")


def validate_proposal(
    value: Mapping[str, Any], query: PackedBoundQuery, inputs: Inputs,
) -> BoundProposalReport:
    _validate_proposal_json_types(value)
    report = _proposal_report(value); contract = packed_bound_search_contract(branch_aware=True)["digest"]
    deterministic = {
        "schema_version": BRANCH_AWARE_SEARCH_SCHEMA_VERSION,
        "contract_digest": contract, "generation_id": inputs.generation_id,
        "query_episode_id": query.episode_id, "rows_scanned": report.rows_scanned,
        "eligible_rows": report.eligible_rows,
        "eligible_main_rows": report.eligible_main_rows,
        "eligible_overflow_rows": report.eligible_overflow_rows,
        "route_counts": report.route_counts, "route_quotas": report.route_quotas,
        "candidate_digest": report.candidate_digest,
        "real_forward_outcomes_accessed": False,
        "input_digest": _packed_query_input_digest(query),
    }
    if not all((report.schema_version == BRANCH_AWARE_SEARCH_SCHEMA_VERSION,
                report.contract_digest == contract,
                report.input_digest == deterministic["input_digest"],
                report.generation_id == inputs.generation_id,
                report.query_episode_id == query.episode_id,
                report.route_quotas == {"composite": PROPOSAL_QUOTA},
                report.route_counts == {"composite": len(report.candidates)},
                report.eligible_rows == report.eligible_main_rows + report.eligible_overflow_rows,
                0 <= report.eligible_rows <= report.rows_scanned,
                len(report.candidates) == min(PROPOSAL_QUOTA, report.eligible_rows),
                all(row.routes == ("composite",) and isfinite(row.lower_bound)
                    and row.lower_bound >= 0 for row in report.candidates),
                list(report.candidates) == sorted(
                    report.candidates, key=lambda row: (row.lower_bound, row.episode_id)),
                report.candidate_digest == bound_proposal_candidate_digest(report.candidates),
                isfinite(report.elapsed_seconds) and report.elapsed_seconds >= 0,
                isfinite(report.peak_rss_mb) and report.peak_rss_mb >= 0,
                report.result_digest == stable_hash(deterministic))):
        raise HarnessError("proposal reconstruction differs")
    return report


def _case_context(inputs: Inputs, case: CaseInput) -> tuple[Any, Any, Any, Any]:
    config = load_config(inputs.config_path); source = source_from_spec(config.datasets["nasdaq"])
    raw = case.registry_case
    episode = build_episode(source, InstrumentKey("nasdaq", raw["symbol"]),
                            raw["cutoff"], raw["lookback"], raw["representation_version"])
    request = SearchQuery(episode.key, ("nasdaq",), ("A", "B"), 20, False, True, 3, 60)
    packed = PackedBoundQuery(
        episode.key.id, episode.key.instrument.source_symbol,
        int(episode.bars.timestamp.iloc[0].value),
        int(latest_eligible_cutoff(episode, 60).value), represent(episode),
        request.quality_tiers,
    )
    if episode.key.id != case.query_id: raise HarnessError("query reconstruction differs")
    return source, episode, request, packed


def _match(value: Any) -> dict[str, Any]:
    return {
        "episode_id": value.episode_key.id,
        "symbol": value.episode_key.instrument.source_symbol,
        "cutoff": value.episode_key.cutoff.isoformat(),
        "total_distance": value.total_distance,
        "component_distances": dict(value.component_distances),
        "alignment": [list(item) for item in value.alignment],
        "quality_tier": value.quality_tier,
    }


def query_binding(
    source: Any, episode: Any, request: SearchQuery, packed: PackedBoundQuery,
    provenance_digest: str,
) -> dict[str, Any]:
    benchmark = source.load_benchmark()
    deterministic = {
        "query_stock_prefix": asdict(causal_prefix_digest(
            source.load(episode.key.instrument), episode.key.cutoff,
        )),
        "query_benchmark_prefix": (
            asdict(causal_prefix_digest(benchmark, episode.key.cutoff))
            if benchmark is not None else None
        ),
        "request": {
            "search_datasets": list(request.search_datasets),
            "quality_tiers": list(request.quality_tiers), "top_k": request.top_k,
            "cross_dataset": request.cross_dataset,
            "deduplicate_overlaps": request.deduplicate_overlaps,
            "max_per_instrument": request.max_per_instrument,
            "minimum_history_gap_bars": request.minimum_history_gap_bars,
        },
        "packed_provenance_digest": provenance_digest,
        "query_representation_digest": representation_input_digest(
            packed.representation,
        ),
    }
    value = {
        **deterministic, "packed_query_input_digest": _packed_query_input_digest(packed),
        "certified_input_digest": stable_hash(deterministic),
    }
    try:
        canonical = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise HarnessError("query binding is not canonical JSON") from exc
    if canonical != value:
        raise HarnessError("query binding changes across JSON serialization")
    return value


def diagnostic_query_bindings(inputs: Inputs) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for case in inputs.cases:
        source, episode, request, packed = _case_context(inputs, case)
        values.append(query_binding(
            source, episode, request, packed, inputs.provenance_digest,
        ))
    return values


def _nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def performance_gate(metrics: Mapping[str, Any]) -> bool:
    required = {
        "forward_proposal_seconds", "reverse_proposal_seconds",
        "exact_task_wall_seconds", "case_task_wall_seconds", "process_rss_mb",
    }
    if set(metrics) != required or any(
        type(value) not in {int, float} or not isfinite(value) or value < 0
        for value in metrics.values()
    ):
        raise HarnessError("resource measurement invalid")
    if not all((
        metrics["exact_task_wall_seconds"] <= metrics["case_task_wall_seconds"],
        metrics["forward_proposal_seconds"] <= metrics["case_task_wall_seconds"],
        metrics["reverse_proposal_seconds"] <= metrics["case_task_wall_seconds"],
        metrics["case_task_wall_seconds"] >= (
            metrics["forward_proposal_seconds"]
            + metrics["reverse_proposal_seconds"]
            + metrics["exact_task_wall_seconds"]
        ),
    )):
        raise HarnessError("task timing nesting differs")
    return all((
        metrics["forward_proposal_seconds"] <= FORWARD_PROPOSAL_LIMIT_SECONDS,
        metrics["reverse_proposal_seconds"] <= REVERSE_PROPOSAL_LIMIT_SECONDS,
        metrics["process_rss_mb"] <= PROCESS_RSS_LIMIT_MB,
    ))


def validate_certificate_and_matches(
    certificate: Mapping[str, Any], matches: Sequence[Mapping[str, Any]], query_id: str,
    *, expected_input_digest: str | None = None,
) -> None:
    certificate_keys = {
        "schema_version", "contract_digest", "generation_id", "query_episode_id",
        "input_digest", "eligible_candidates", "exact_evaluated", "safely_pruned",
        "stopped_early", "stop_threshold", "next_lower_bound",
        "maximum_quantized_bound_excess", "materialization_groups", "sparse_symbols",
        "batch_symbols", "rounds", "result_digest", "elapsed_seconds",
        "native_bound_accounting", "minimum_native_pruned_bound",
        "threshold_closure_passes",
    }
    match_keys = {
        "episode_id", "symbol", "cutoff", "total_distance",
        "component_distances", "alignment", "quality_tier",
    }
    if (type(certificate) is not dict or type(matches) is not list
            or set(certificate) != certificate_keys or len(matches) != 20 or any(
        set(row) != match_keys for row in matches
    )):
        raise HarnessError("certified result fields differ")
    if not all((
        all(type(certificate[key]) is str for key in (
            "schema_version", "contract_digest", "generation_id", "query_episode_id",
            "input_digest", "result_digest",
        )),
        all(_nonnegative_int(certificate[key]) for key in (
            "eligible_candidates", "exact_evaluated", "safely_pruned",
            "materialization_groups", "sparse_symbols", "batch_symbols",
        )),
        type(certificate["stopped_early"]) is bool,
        all(type(certificate[key]) is float and isfinite(certificate[key])
            and certificate[key] >= 0 for key in (
                "stop_threshold", "maximum_quantized_bound_excess", "elapsed_seconds",
            )),
        certificate["next_lower_bound"] is None or (
            type(certificate["next_lower_bound"]) is float
            and isfinite(certificate["next_lower_bound"])
            and certificate["next_lower_bound"] >= 0
        ),
        certificate["minimum_native_pruned_bound"] is None or (
            type(certificate["minimum_native_pruned_bound"]) is float
            and isfinite(certificate["minimum_native_pruned_bound"])
            and certificate["minimum_native_pruned_bound"] >= 0
        ),
        type(certificate["rounds"]) is list,
        type(certificate["threshold_closure_passes"]) is list,
        type(certificate["native_bound_accounting"]) is dict,
    )):
        raise HarnessError("certificate raw JSON types differ")
    for row in matches:
        try:
            identifier_valid = len(bytes.fromhex(row["episode_id"])) == 12
            cutoff = pd.Timestamp(row["cutoff"])
        except (TypeError, ValueError) as exc:
            raise HarnessError("match identity encoding differs") from exc
        if not all((
            type(row["episode_id"]) is str, identifier_valid,
            type(row["symbol"]) is str and row["symbol"] != "",
            type(row["cutoff"]) is str and cutoff.tzinfo is not None,
            type(row["total_distance"]) is float
            and isfinite(row["total_distance"]) and row["total_distance"] >= 0,
            type(row["component_distances"]) is dict,
            set(row["component_distances"]) == COMPONENT_NAMES,
            all(type(value) is float and isfinite(value) and value >= 0
                for value in row["component_distances"].values()),
            type(row["alignment"]) is list and len(row["alignment"]) > 0,
            all(type(pair) is list and len(pair) == 2
                and all(_nonnegative_int(index) for index in pair)
                for pair in row["alignment"]),
            type(row["quality_tier"]) is str and row["quality_tier"] in {"A", "B"},
        )):
            raise HarnessError("match raw JSON schema differs")
    contract = certified_packed_search_contract(
        requested_positions=True, vector_lower_bounds=True,
        deferred_alignments=True, compact_scored=True,
        native_bound_deferral=True, streaming_threshold_closure=True,
        branch_aware_packed_bounds=True,
    )
    accounting = certificate["native_bound_accounting"]
    if set(accounting) != {
        "native_bound_evaluated", "exact_dtw_evaluated", "native_bound_pruned",
        "packed_bound_pruned",
    }:
        raise HarnessError("native accounting fields differ")
    rounds = certificate["rounds"]
    round_keys = {
        "frontier_rows", "exact_rows", "next_lower_bound", "constrained_threshold",
        "selected_rows", "certified", "proposal_digest",
    }
    frontiers = [row.get("frontier_rows") for row in rounds]
    eligible = certificate["eligible_candidates"]
    expected_frontiers: list[int] = []
    frontier = min(INITIAL_FRONTIER, eligible)
    while frontier > 0:
        expected_frontiers.append(frontier)
        if frontier >= eligible or frontier >= MAXIMUM_FRONTIER:
            break
        frontier = min(MAXIMUM_FRONTIER, frontier * 2, eligible)
    distinct_frontiers = list(dict.fromkeys(frontiers))
    if not rounds or any(set(row) != round_keys for row in rounds) or (
        distinct_frontiers != expected_frontiers[:len(distinct_frontiers)]
    ) or any(not all((
        _nonnegative_int(row["frontier_rows"]) and row["frontier_rows"] > 0,
        _nonnegative_int(row["exact_rows"]),
        _nonnegative_int(row["selected_rows"]) and row["selected_rows"] <= 20,
        type(row["certified"]) is bool,
        type(row["constrained_threshold"]) is float
        and isfinite(row["constrained_threshold"])
        and row["constrained_threshold"] >= 0,
        row["next_lower_bound"] is None or (
            type(row["next_lower_bound"]) is float
            and isfinite(row["next_lower_bound"])
            and row["next_lower_bound"] >= 0
        ),
        _is_digest(row["proposal_digest"]),
    )) for row in rounds):
        raise HarnessError("logical frontier rounds differ")
    if not all((
        all(left["frontier_rows"] <= right["frontier_rows"]
            and left["exact_rows"] <= right["exact_rows"]
            for left, right in zip(rounds, rounds[1:])),
        all(row["certified"] is False for row in rounds[:-1]),
        all(row["frontier_rows"] < eligible for row in rounds[:-1]),
    )):
        raise HarnessError("logical frontier state differs")
    closures = certificate["threshold_closure_passes"]
    closure_keys = {
        "lower_exclusive", "upper_inclusive", "admitted_rows",
        "cumulative_native_bound_evaluated", "cumulative_exact_dtw_evaluated",
        "selected_rows", "resulting_threshold",
        "minimum_packed_unclassified_bound", "minimum_native_pruned_bound",
        "excluded_prefix_digest", "admitted_set_digest", "scan_result_digest",
        "certified",
    }
    prior: Mapping[str, Any] | None = None
    for row in closures:
        float_fields = ("upper_inclusive", "resulting_threshold")
        optional_float_fields = (
            "lower_exclusive", "minimum_packed_unclassified_bound",
            "minimum_native_pruned_bound",
        )
        if not all((
            type(row) is dict, set(row) == closure_keys,
            all(type(row[key]) is float and isfinite(row[key]) and row[key] >= 0
                for key in float_fields),
            all(row[key] is None or (
                type(row[key]) is float and isfinite(row[key]) and row[key] >= 0
            ) for key in optional_float_fields),
            all(_nonnegative_int(row[key]) for key in (
                "admitted_rows", "cumulative_native_bound_evaluated",
                "cumulative_exact_dtw_evaluated", "selected_rows",
            )),
            row["selected_rows"] <= 20,
            row["cumulative_exact_dtw_evaluated"]
            <= row["cumulative_native_bound_evaluated"],
            all(_is_digest(row[key]) for key in (
                "excluded_prefix_digest", "admitted_set_digest", "scan_result_digest",
            )),
            type(row["certified"]) is bool,
            prior is None or all((
                row["lower_exclusive"] == prior["upper_inclusive"],
                row["upper_inclusive"] > prior["upper_inclusive"],
                row["cumulative_native_bound_evaluated"]
                == prior["cumulative_native_bound_evaluated"] + row["admitted_rows"],
                row["cumulative_exact_dtw_evaluated"]
                >= prior["cumulative_exact_dtw_evaluated"],
            )),
        )):
            raise HarnessError("threshold closure evidence differs")
        prior = row
    final_round = rounds[-1]
    if closures:
        first_closure = closures[0]
        final_closure = closures[-1]
        closure_state_valid = all((
            final_round["certified"] is False,
            final_round["frontier_rows"] == MAXIMUM_FRONTIER,
            final_round["frontier_rows"] < eligible,
            all(row["certified"] is False for row in closures[:-1]),
            final_closure["certified"] is True,
            first_closure["lower_exclusive"] is None,
            first_closure["upper_inclusive"].hex()
            == (final_round["constrained_threshold"] + TOLERANCE).hex(),
            all(row["lower_exclusive"] == prior["upper_inclusive"]
                and row["upper_inclusive"].hex()
                == (prior["resulting_threshold"] + TOLERANCE).hex()
                and row["upper_inclusive"] > prior["upper_inclusive"]
                for prior, row in zip(closures, closures[1:])),
            all((row["resulting_threshold"] + TOLERANCE)
                > row["upper_inclusive"] for row in closures[:-1]),
            first_closure["cumulative_native_bound_evaluated"]
            == final_round["frontier_rows"] + first_closure["admitted_rows"],
            first_closure["cumulative_exact_dtw_evaluated"]
            >= final_round["exact_rows"],
            final_closure["selected_rows"] == len(matches),
            final_closure["cumulative_native_bound_evaluated"]
            == accounting["native_bound_evaluated"],
            final_closure["cumulative_exact_dtw_evaluated"]
            == accounting["exact_dtw_evaluated"],
            final_closure["resulting_threshold"] == certificate["stop_threshold"],
            certificate["minimum_native_pruned_bound"]
            == final_closure["minimum_native_pruned_bound"],
        ))
    else:
        closure_state_valid = all((
            final_round["certified"] is True,
            final_round["exact_rows"] == accounting["exact_dtw_evaluated"],
            final_round["selected_rows"] == len(matches),
            final_round["constrained_threshold"] == certificate["stop_threshold"],
        ))
    final_next = (
        min((value for value in (
            closures[-1]["minimum_packed_unclassified_bound"],
            closures[-1]["minimum_native_pruned_bound"],
        ) if value is not None), default=None)
        if closures else final_round["next_lower_bound"]
    )
    deterministic = {
        "schema_version": contract["schema_version"],
        "contract_digest": contract["digest"], "generation_id": GENERATION_ID,
        "query_episode_id": query_id, "input_digest": certificate["input_digest"],
        "eligible_candidates": certificate["eligible_candidates"],
        "exact_evaluated": certificate["exact_evaluated"],
        "safely_pruned": certificate["safely_pruned"],
        "stopped_early": certificate["stopped_early"],
        "stop_threshold_hex": certificate["stop_threshold"].hex(),
        "next_lower_bound_hex": (
            certificate["next_lower_bound"].hex()
            if certificate["next_lower_bound"] is not None else None
        ),
        "maximum_quantized_bound_excess_hex":
            certificate["maximum_quantized_bound_excess"].hex(),
        "rounds": rounds,
        "matches": [{
            "episode_id": row["episode_id"],
            "total_hex": row["total_distance"].hex(),
            "components": {
                key: value.hex()
                for key, value in sorted(row["component_distances"].items())
            },
            "alignment": row["alignment"],
        } for row in matches],
        "real_forward_outcomes_accessed": False,
        "native_bound_accounting": accounting,
        "minimum_native_pruned_bound_hex": (
            certificate["minimum_native_pruned_bound"].hex()
            if certificate["minimum_native_pruned_bound"] is not None else None
        ),
        "threshold_closure_passes": closures,
    }
    totals = [(row["total_distance"], row["episode_id"]) for row in matches]
    count_fields = (
        "eligible_candidates", "exact_evaluated", "safely_pruned",
        "materialization_groups", "sparse_symbols", "batch_symbols",
    )
    if not all((
        certificate["schema_version"] == contract["schema_version"],
        certificate["contract_digest"] == contract["digest"],
        certificate["generation_id"] == GENERATION_ID,
        certificate["query_episode_id"] == query_id,
        all(_nonnegative_int(certificate[key]) for key in count_fields),
        type(certificate["stopped_early"]) is bool,
        expected_input_digest is None
        or certificate["input_digest"] == expected_input_digest,
        certificate["eligible_candidates"]
        == certificate["exact_evaluated"] + certificate["safely_pruned"],
        certificate["exact_evaluated"] == accounting["exact_dtw_evaluated"],
        all(_nonnegative_int(value) for value in accounting.values()),
        accounting["native_bound_evaluated"]
        == accounting["exact_dtw_evaluated"] + accounting["native_bound_pruned"],
        (accounting["native_bound_pruned"] == 0)
        is (certificate["minimum_native_pruned_bound"] is None),
        certificate["eligible_candidates"]
        == accounting["native_bound_evaluated"] + accounting["packed_bound_pruned"],
        certificate["exact_evaluated"] >= len(matches),
        certificate["materialization_groups"]
        == certificate["sparse_symbols"] + certificate["batch_symbols"],
        closure_state_valid,
        all(row["selected_rows"] <= row["exact_rows"]
            <= certificate["eligible_candidates"] for row in rounds),
        not closures or all((
            closures[-1]["certified"] is True,
            closures[-1]["cumulative_native_bound_evaluated"]
            == accounting["native_bound_evaluated"],
            closures[-1]["cumulative_exact_dtw_evaluated"]
            == accounting["exact_dtw_evaluated"],
        )),
        certificate["stop_threshold"]
        == max(row["total_distance"] for row in matches),
        certificate["next_lower_bound"] == final_next,
        certificate["stopped_early"] is (final_next is not None),
        final_next is None or final_next > certificate["stop_threshold"] + TOLERANCE,
        0 <= certificate["maximum_quantized_bound_excess"] <= TOLERANCE,
        isfinite(certificate["elapsed_seconds"])
        and certificate["elapsed_seconds"] >= 0,
        totals == sorted(totals), len({row["episode_id"] for row in matches}) == 20,
        all(row["quality_tier"] in {"A", "B"} for row in matches),
        certificate["result_digest"] == stable_hash(deterministic),
    )):
        raise HarnessError("certified result reconstruction differs")


def run_case(
    inputs: Inputs, prereg: Mapping[str, Any], resident: Mapping[str, Any],
    case: CaseInput, *, scan: Callable[..., BoundProposalReport] = scan_packed_bound_proposals_threaded,
    certified: Callable[..., Any] = certified_packed_search,
    lease: Callable[[Inputs, Mapping[str, Any]], str] = lease_exact,
    context: Callable[[Inputs, CaseInput], tuple[Any, Any, Any, Any]] = _case_context,
    full_validation: Callable[..., dict[str, Any]] = resident_full,
    binding_builder: Callable[..., dict[str, Any]] = query_binding,
    clock: Callable[[], float] = perf_counter,
) -> dict[str, Any]:
    path = inputs.output_root / "cases" / f"{case.ordinal:02d}-{case.query_id}.json"
    if path.exists(): raise HarnessError("case checkpoint exists; resume is forbidden")
    started, rss = clock(), _rss(); source, episode, request, packed = context(inputs, case)
    binding = binding_builder(source, episode, request, packed, inputs.provenance_digest)
    leases = [lease(inputs, resident)]
    forward = scan(Path(resident["store_root"]), inputs.generation_id, packed,
                   route_quotas={"composite": PROPOSAL_QUOTA}, block_rows=4_096,
                   block_order="forward", threads=PROPOSAL_THREADS, branch_aware=True,
                   verify_content=False, expected_provenance_digest=inputs.provenance_digest)
    leases += [lease(inputs, resident), lease(inputs, resident)]
    reverse = scan(Path(resident["store_root"]), inputs.generation_id, packed,
                   route_quotas={"composite": PROPOSAL_QUOTA}, block_rows=4_097,
                   block_order="reverse", threads=PROPOSAL_THREADS, branch_aware=True,
                   verify_content=False, expected_provenance_digest=inputs.provenance_digest)
    leases.append(lease(inputs, resident))
    fwd, rev = _proposal_payload(forward), _proposal_payload(reverse)
    validate_proposal(fwd, packed, inputs); validate_proposal(rev, packed, inputs)
    semantic_omitted = {"block_rows", "block_order", "elapsed_seconds", "peak_rss_mb"}
    if _without(fwd, semantic_omitted) != _without(rev, semantic_omitted):
        raise HarnessError("forward/reverse proposal mismatch")
    exact_started = clock()
    result = certified(
        episode, source, request, Path(resident["store_root"]), inputs.generation_id,
        store_dataset_id="nasdaq", initial_frontier_rows=INITIAL_FRONTIER,
        maximum_frontier_rows=MAXIMUM_FRONTIER, seed_rows=SEED_ROWS,
        block_rows=4_096, workers=EXACT_WORKERS, sparse_cutoff=8,
        tolerance=TOLERANCE, verify_content=False, requested_positions=True,
        vector_lower_bounds=True, deferred_alignments=True, compact_scored=True,
        native_bound_deferral=True, streaming_threshold_closure=True,
        branch_aware_packed_bounds=True, precomputed_proposal=forward,
    )
    leases.append(lease(inputs, resident))
    certificate = asdict(result.certificate); matches = [_match(value) for value in result.matches]
    validate_certificate_and_matches(
        certificate, matches, case.query_id,
        expected_input_digest=binding["certified_input_digest"],
    )
    if not all((len(matches) == 20, certificate["schema_version"]
                == "m04r-certified-packed-search-v8",
                certificate["eligible_candidates"] == forward.eligible_rows,
                certificate["maximum_quantized_bound_excess"] <= TOLERANCE,
                certificate["eligible_candidates"] == certificate["exact_evaluated"]
                + certificate["safely_pruned"])):
        raise HarnessError("certified semantic gates failed")
    metrics = {
        "forward_proposal_seconds": forward.elapsed_seconds,
        "reverse_proposal_seconds": reverse.elapsed_seconds,
        "exact_task_wall_seconds": clock() - exact_started,
        "case_task_wall_seconds": clock() - started,
        "process_rss_mb": max(rss, _rss(), forward.peak_rss_mb, reverse.peak_rss_mb),
    }
    performance_passed = performance_gate(metrics)
    post_resident = full_validation(
        inputs.source_store_root, inputs.resident_root, inputs.generation_id,
        inputs.provenance_digest, inputs.reserve_bytes,
    )
    if post_resident != resident:
        raise HarnessError("resident full validation changed during case")
    deterministic = {
        "schema_version": CASE_SCHEMA, "status": "truth_blind_case_complete",
        "development_only": True, "truth_opened": False,
        "production_promotion_authorized": False,
        "real_forward_outcomes_accessed": False,
        "preregistration_digest": inputs.prereg_digest,
        "registry_case_id": case.case_id, "query_episode_id": case.query_id,
        "query_binding": binding,
        "resident_identity_digest": resident["identity_digest"],
        "lease_digests": leases, "forward_proposal": fwd, "reverse_proposal": rev,
        "proposal_semantic_exact": True, "certificate": certificate,
        "matches": matches, "rounds": certificate["rounds"],
        "certified": True, "semantic_passed": True,
        "streaming_fallback_used": len(certificate["threshold_closure_passes"]) > 0,
        "metrics": metrics,
        "performance_passed": performance_passed,
    }
    payload = {**deterministic, "created_at": datetime.now(timezone.utc).isoformat(),
               "result_digest": stable_hash(deterministic)}
    _atomic(path, payload); lease(inputs, resident)
    return payload


def _tree(root: Path, expected: set[str]) -> None:
    files, directories = set(), set()
    for path in root.rglob("*"):
        value = path.lstat(); relative = str(path.relative_to(root))
        if stat.S_ISLNK(value.st_mode): raise HarnessError("output symlink forbidden")
        if stat.S_ISDIR(value.st_mode): directories.add(relative)
        elif stat.S_ISREG(value.st_mode): files.add(relative)
        else: raise HarnessError("output special entry forbidden")
    if files != expected or directories != {"cases"}: raise HarnessError("output tree differs")


def _tree_snapshot_digest(root: Path, *, omitted: set[str] | None = None) -> str:
    omitted = omitted or set()
    if root.is_symlink() or not root.is_dir():
        raise HarnessError("output root is linked or absent")
    files: dict[str, str] = {}
    directories: list[str] = []
    for path in root.rglob("*"):
        relative = str(path.relative_to(root))
        if relative in omitted:
            continue
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise HarnessError("output tree contains a symlink")
        if stat.S_ISDIR(mode):
            directories.append(relative)
        elif stat.S_ISREG(mode):
            files[relative] = _sha(path)
        else:
            raise HarnessError("output tree contains a special entry")
    return stable_hash({"files": files, "directories": sorted(directories)})


def _write_incomplete(inputs: Inputs, error_type: str) -> dict[str, Any]:
    _validate_incomplete_prefix(inputs, include_marker=False)
    deterministic = {
        "schema_version": "m04r13-incomplete-v1",
        "status": "terminal_incomplete_new_root_required",
        "development_only": True, "production_promotion_authorized": False,
        "real_forward_outcomes_accessed": False,
        "preregistration_digest": inputs.prereg_digest,
        "output_root": str(inputs.output_root),
        "query_ids": list(FROZEN_QUERY_IDS), "error_type": error_type,
        "partial_tree_digest": _tree_snapshot_digest(
            inputs.output_root, omitted={"INCOMPLETE.json"},
        ),
    }
    payload = {
        **deterministic, "created_at": datetime.now(timezone.utc).isoformat(),
        "result_digest": stable_hash(deterministic),
    }
    _atomic(inputs.output_root / "INCOMPLETE.json", payload)
    _validate_incomplete_prefix(inputs, include_marker=True)
    return payload


def _validate_incomplete_prefix(inputs: Inputs, *, include_marker: bool) -> None:
    expected_cases = [
        f"cases/{case.ordinal:02d}-{case.query_id}.json" for case in inputs.cases
    ]
    files: set[str] = set()
    directories: set[str] = set()
    for path in inputs.output_root.rglob("*"):
        mode = path.lstat().st_mode
        relative = str(path.relative_to(inputs.output_root))
        if stat.S_ISLNK(mode):
            raise HarnessError("partial output contains a symlink")
        if stat.S_ISDIR(mode):
            directories.add(relative)
        elif stat.S_ISREG(mode):
            files.add(relative)
        else:
            raise HarnessError("partial output contains a special entry")
    marker_present = "INCOMPLETE.json" in files
    if marker_present is not include_marker:
        raise HarnessError("partial incomplete-marker state differs")
    files.discard("INCOMPLETE.json")
    if not directories:
        if files:
            raise HarnessError("empty partial root has files")
    else:
        if directories != {"cases"}:
            raise HarnessError("partial directory tree differs")
        foundation = ["RUN_STARTED.json", "CONTRACT.json", "RESIDENT.json"]
        observed_foundation = [value for value in foundation if value in files]
        if observed_foundation != foundation[:len(observed_foundation)]:
            raise HarnessError("partial foundation is not a contiguous prefix")
        observed_cases = [value for value in expected_cases if value in files]
        if observed_cases != expected_cases[:len(observed_cases)]:
            raise HarnessError("partial case tree is not a contiguous prefix")
        sealed = "PRODUCER_SEALED.json" in files
        if observed_cases and len(observed_foundation) != len(foundation):
            raise HarnessError("cases precede complete foundation")
        if sealed and len(observed_cases) != len(expected_cases):
            raise HarnessError("producer seal precedes complete cases")
        expected = {*observed_foundation, *observed_cases}
        if sealed:
            expected.add("PRODUCER_SEALED.json")
        if files != expected:
            raise HarnessError("partial file tree differs")
    if include_marker:
        marker = _read_json(inputs.output_root / "INCOMPLETE.json")
        if not all((
            set(marker) == {
                "schema_version", "status", "development_only",
                "production_promotion_authorized",
                "real_forward_outcomes_accessed", "preregistration_digest",
                "output_root", "query_ids", "error_type",
                "partial_tree_digest", "created_at", "result_digest",
            },
            marker.get("schema_version") == "m04r13-incomplete-v1",
            marker.get("status") == "terminal_incomplete_new_root_required",
            marker.get("development_only") is True,
            marker.get("production_promotion_authorized") is False,
            marker.get("real_forward_outcomes_accessed") is False,
            marker.get("preregistration_digest") == inputs.prereg_digest,
            marker.get("output_root") == str(inputs.output_root),
            marker.get("query_ids") == list(FROZEN_QUERY_IDS),
            type(marker.get("error_type")) is str and marker["error_type"] != "",
            marker.get("partial_tree_digest") == _tree_snapshot_digest(
                inputs.output_root, omitted={"INCOMPLETE.json"},
            ),
            _is_utc_iso_timestamp(marker.get("created_at")),
            marker.get("result_digest") == stable_hash(_without(
                marker, {"created_at", "result_digest"},
            )),
        )):
            raise HarnessError("incomplete marker differs")


def validate_child_boundary(
    inputs: Inputs, prereg: Mapping[str, Any], resident: Mapping[str, Any], ordinal: int,
) -> None:
    if ordinal not in range(len(inputs.cases)):
        raise HarnessError("case ordinal differs")
    expected_cases = {
        f"cases/{case.ordinal:02d}-{case.query_id}.json"
        for case in inputs.cases[:ordinal]
    }
    _tree(
        inputs.output_root,
        {"RUN_STARTED.json", "CONTRACT.json", "RESIDENT.json", *expected_cases},
    )
    if _read_json(inputs.output_root / "CONTRACT.json") != dict(prereg):
        raise HarnessError("child contract snapshot differs")
    if _read_json(inputs.output_root / "RESIDENT.json") != dict(resident):
        raise HarnessError("child resident snapshot differs")
    started = _read_json(inputs.output_root / "RUN_STARTED.json")
    if not all((
        set(started) == {"schema_version", "preregistration_digest", "case_order",
                         "parent_max_workers", "created_at"},
        started.get("schema_version") == "m04r13-run-started-v1",
        started.get("preregistration_digest") == inputs.prereg_digest,
        started.get("case_order") == list(FROZEN_QUERY_IDS),
        started.get("parent_max_workers") == 1,
        _is_utc_iso_timestamp(started.get("created_at")),
    )):
        raise HarnessError("child run marker differs")
    for case in inputs.cases[:ordinal]:
        validate_case(_read_json(
            inputs.output_root / "cases" / f"{case.ordinal:02d}-{case.query_id}.json"
        ), inputs, case, resident)


def validate_case(
    payload: Mapping[str, Any], inputs: Inputs, case: CaseInput,
    resident: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    required = {
        "schema_version", "status", "development_only", "truth_opened",
        "production_promotion_authorized", "real_forward_outcomes_accessed",
        "preregistration_digest", "registry_case_id", "query_episode_id",
        "query_binding",
        "resident_identity_digest", "lease_digests", "forward_proposal",
        "reverse_proposal", "proposal_semantic_exact", "certificate", "matches",
        "rounds", "certified", "streaming_fallback_used", "metrics",
        "semantic_passed", "performance_passed", "created_at", "result_digest",
    }
    if not all((set(payload) == required,
                payload.get("schema_version") == CASE_SCHEMA,
                payload.get("status") == "truth_blind_case_complete",
                payload.get("development_only") is True,
                payload.get("truth_opened") is False,
                payload.get("production_promotion_authorized") is False,
                payload.get("real_forward_outcomes_accessed") is False,
                payload.get("preregistration_digest") == inputs.prereg_digest,
                payload.get("registry_case_id") == case.case_id,
                payload.get("query_episode_id") == case.query_id,
                payload.get("proposal_semantic_exact") is True,
                payload.get("certified") is True,
                payload.get("semantic_passed") is True,
                _is_utc_iso_timestamp(payload.get("created_at")),
                payload.get("result_digest") == stable_hash(_without(
                    payload, {"created_at", "result_digest"})))):
        raise HarnessError("case checkpoint differs")
    source, episode, request, query = _case_context(inputs, case)
    binding = query_binding(source, episode, request, query, inputs.provenance_digest)
    if payload.get("query_binding") != binding:
        raise HarnessError("query/source/request binding differs")
    forward = validate_proposal(payload["forward_proposal"], query, inputs)
    reverse = validate_proposal(payload["reverse_proposal"], query, inputs)
    omitted = {"block_rows", "block_order", "elapsed_seconds", "peak_rss_mb"}
    metrics = payload.get("metrics", {})
    if not (forward.block_rows == 4_096 and forward.block_order == "forward"
            and reverse.block_rows == 4_097 and reverse.block_order == "reverse"
            and len(payload.get("lease_digests", [])) == 5
            and all(value == payload["lease_digests"][0]
                    for value in payload["lease_digests"])
            and (resident is None or all(
                value == resident["lease"]["lease_digest"]
                for value in payload["lease_digests"]
            ))
            and _without(payload["forward_proposal"], omitted)
            == _without(payload["reverse_proposal"], omitted)
            and set(metrics) == {"forward_proposal_seconds", "reverse_proposal_seconds",
                                 "exact_task_wall_seconds", "case_task_wall_seconds",
                                 "process_rss_mb"}
            and all(type(value) in {int, float} and isfinite(value) and value >= 0
                    for value in metrics.values())
            and payload.get("performance_passed") == performance_gate(metrics)
            and metrics["forward_proposal_seconds"] == forward.elapsed_seconds
            and metrics["reverse_proposal_seconds"] == reverse.elapsed_seconds
            and metrics["process_rss_mb"] >= forward.peak_rss_mb
            and metrics["process_rss_mb"] >= reverse.peak_rss_mb
            and payload["certificate"]["elapsed_seconds"]
            <= metrics["exact_task_wall_seconds"]
            and payload.get("rounds") == payload.get("certificate", {}).get("rounds")):
        raise HarnessError("case checkpoint reconstruction differs")
    if payload.get("streaming_fallback_used") is not (
        len(payload["certificate"]["threshold_closure_passes"]) > 0
    ):
        raise HarnessError("streaming fallback flag differs")
    validate_certificate_and_matches(
        payload["certificate"], payload["matches"], case.query_id,
        expected_input_digest=binding["certified_input_digest"],
    )
    return dict(payload)


def seal_producer(inputs: Inputs, prereg: Mapping[str, Any], resident: Mapping[str, Any]) -> dict[str, Any]:
    cases = [validate_case(_read_json(
        inputs.output_root / "cases" / f"{case.ordinal:02d}-{case.query_id}.json"
    ), inputs, case, resident) for case in inputs.cases]
    post_resident = resident_full(
        inputs.source_store_root, inputs.resident_root, inputs.generation_id,
        inputs.provenance_digest, inputs.reserve_bytes,
    )
    if post_resident != resident:
        raise HarnessError("resident full validation changed before producer seal")
    lease_exact(inputs, resident)
    deterministic = {
        "schema_version": SEAL_SCHEMA, "status": "truth_blind_producer_complete",
        "development_only": True, "truth_opened": False,
        "production_promotion_authorized": False,
        "real_forward_outcomes_accessed": False,
        "preregistration_digest": inputs.prereg_digest,
        "resident_identity_digest": resident["identity_digest"],
        "query_ids": list(FROZEN_QUERY_IDS),
        "case_digests": [value["result_digest"] for value in cases],
        "semantic_passed": all(value["semantic_passed"] for value in cases),
        "performance_passed": all(value["performance_passed"] for value in cases),
    }
    payload = {**deterministic, "created_at": datetime.now(timezone.utc).isoformat(),
               "seal_digest": stable_hash(deterministic)}
    expected = {"RUN_STARTED.json", "CONTRACT.json", "RESIDENT.json"} | {
        f"cases/{case.ordinal:02d}-{case.query_id}.json" for case in inputs.cases
    }
    _tree(inputs.output_root, expected)
    lease_exact(inputs, resident)
    # Publication of the terminal producer seal is the last fallible operation.
    _atomic(inputs.output_root / "PRODUCER_SEALED.json", payload)
    return payload


def produce(
    inputs: Inputs, prereg: Mapping[str, Any], resident: Mapping[str, Any],
    *, child_runner: Callable[[Sequence[str]], int], child_command: Callable[[int], Sequence[str]],
) -> dict[str, Any]:
    if inputs.output_root.exists() or inputs.output_root.is_symlink():
        if (inputs.output_root.is_dir() and not inputs.output_root.is_symlink()
                and not (inputs.output_root / "PRODUCER_SEALED.json").exists()
                and not (inputs.output_root / "INCOMPLETE.json").exists()):
            _write_incomplete(inputs, "PartialRootDetected")
        raise HarnessError("fresh absent output root is required; resume forbidden")
    inputs.output_root.mkdir(parents=True)
    try:
        (inputs.output_root / "cases").mkdir()
        _atomic(inputs.output_root / "RUN_STARTED.json", {
            "schema_version": "m04r13-run-started-v1",
            "preregistration_digest": inputs.prereg_digest,
            "case_order": list(FROZEN_QUERY_IDS), "parent_max_workers": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        _atomic(inputs.output_root / "CONTRACT.json", dict(prereg))
        _atomic(inputs.output_root / "RESIDENT.json", dict(resident))
        for ordinal in range(4):
            if child_runner(child_command(ordinal)) != 0:
                raise HarnessError(f"case child failed: {ordinal}")
        return seal_producer(inputs, prereg, resident)
    except BaseException as exc:
        incomplete = inputs.output_root / "INCOMPLETE.json"
        if not incomplete.exists():
            _write_incomplete(inputs, type(exc).__name__)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preregister", "produce", "_case-child"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--registry-root", type=Path, required=True)
        command.add_argument("--source-full-root", type=Path, required=True)
        command.add_argument("--resident-root", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--case-ordinal", type=int)
    args = parser.parse_args(); repository = Path(__file__).resolve().parents[2]
    if args.command == "preregister":
        payload = preregister_payload(
            repository=repository, config_path=args.config,
            registry_root=args.registry_root, source_full_root=args.source_full_root,
            resident_root=args.resident_root, output_root=args.output_root,
            generation_id=GENERATION_ID,
            provenance_digest=PROVENANCE_DIGEST,
            reserve_bytes=RESIDENT_RESERVE_BYTES,
        )
        _atomic(repository / PREREG_RELATIVE, payload); print(json.dumps(payload, indent=2)); return 0
    inputs, prereg, resident = load_inputs(
        repository=repository, config_path=args.config, registry_root=args.registry_root,
        source_full_root=args.source_full_root, resident_root=args.resident_root,
        output_root=args.output_root,
    )
    if args.command == "_case-child":
        if args.case_ordinal not in range(4): raise HarnessError("case ordinal differs")
        validate_child_boundary(inputs, prereg, resident, args.case_ordinal)
        result = run_case(inputs, prereg, resident, inputs.cases[args.case_ordinal])
    else:
        if args.case_ordinal is not None: raise HarnessError("producer rejects case ordinal")
        base = [sys.executable, str(Path(__file__).resolve()), "_case-child",
                "--config", str(args.config), "--registry-root", str(args.registry_root),
                "--source-full-root", str(args.source_full_root),
                "--resident-root", str(args.resident_root),
                "--output-root", str(args.output_root)]
        result = produce(
            inputs, prereg, resident,
            child_runner=lambda command: subprocess.run(command, check=False).returncode,
            child_command=lambda ordinal: [*base, "--case-ordinal", str(ordinal)],
        )
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
