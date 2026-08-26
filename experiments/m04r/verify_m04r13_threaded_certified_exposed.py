"""Independent verifier for the terminal M04R-13 exposed-case differential.

This module deliberately does not import either the M04R-13 producer or its
comparator.  It reconstructs their frozen contracts and terminal evidence from
the committed preregistration, the four SHA-bound v4 authorities, and the
current resident/source bindings.  It writes only a fresh verification root.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import importlib.metadata
import json
from math import isfinite
import os
from pathlib import Path
import platform
import stat
import subprocess
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

import pandas as pd
import numpy as np

from market_analogues.adapters import source_from_spec
from market_analogues.certified_packed_search import certified_packed_search_contract
from market_analogues.config import load_config
from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.episodes import build_episode
from market_analogues.packed_bound_search import (
    BRANCH_AWARE_SEARCH_SCHEMA_VERSION,
    BoundProposal,
    PackedBoundQuery,
    _packed_query_input_digest,
    bound_proposal_candidate_digest,
    packed_bound_search_contract,
    scan_packed_bound_proposals,
    scan_packed_bound_threshold,
)
from market_analogues.packed_bound_store import (
    TIER_NAMES,
    decode_episode_id,
    load_packed_generation,
    packed_branch_aware_lower_bounds,
)
from market_analogues.representation import represent, representation_input_digest
from market_analogues.resident_store import (
    observe_ready_strict,
    resident_file_identity_lease,
)
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import InstrumentKey, SearchQuery, stable_hash


VERIFICATION_SCHEMA = "m04r13-threaded-certified-independent-verification-v1"
PREREG_SCHEMA = "m04r13-threaded-certified-exposed-preregistration-v1"
CASE_SCHEMA = "m04r13-threaded-certified-exposed-case-v1"
PRODUCER_SEAL_SCHEMA = "m04r13-threaded-certified-exposed-producer-seal-v1"
RESULTS_OPENED_SCHEMA = "m04r13-threaded-certified-results-opened-v1"
COMPARISON_SCHEMA = "m04r13-threaded-certified-comparison-v1"
COMPARISON_SEAL_SCHEMA = "m04r13-threaded-certified-comparison-seal-v1"
PREREG_RELATIVE = Path(
    "experiments/m04r/m04r13_threaded_certified_exposed_preregistered.json"
)
DIAGNOSTIC_RELATIVE = Path(
    "config/data/analogues/m04r13/finite-threshold-diagnostic-v1/DIAGNOSTIC.json"
)
DIAGNOSTIC_SCHEMA = "m04r13-truth-blind-finite-threshold-diagnostic-v1"
DIAGNOSTIC_CASE_SCHEMA = "m04r13-truth-blind-finite-threshold-case-v1"
CONFIG_RELATIVE = Path("config/datasets.example.yaml")
REGISTRY_RELATIVE = Path(
    "config/data/analogues/m04r10/nasdaq-untouched-authority-registry"
)
SOURCE_FULL_RELATIVE = Path("config/data/analogues/poc/m04r/packed-bound-full")
OUTPUT_RELATIVE = Path("config/data/analogues/m04r13/threaded-certified-exposed-v1")
GENERATION_ID = "9fc6ae0ec4451133d8006897162f3443803fd30a3c44b78e80a476fbb18bb483"
PROVENANCE_DIGEST = "83ccfa62ac7ffec03e48d0f8a5634c7b8f1b8b0dde1426be22cc343fe116f62d"
REGISTRY_DIGEST = "0a4da732f91375a091775cb04e6e77c8d136ade47d7f4d16508a2d9a6555361e"
RESIDENT_ROOT = Path("/dev/shm/market-analogues/m04r11-candidate-v2") / GENERATION_ID
QUERY_IDS = (
    "3307023dbe2164d025e788da", "3618af07dedd52fb3bdb1ccd",
    "9d7365581643bd93e85beb67", "99a0838725a09570b4a075ff",
)
CASE_IDS = (
    "nasdaq-JCTC-historical-252", "nasdaq-GBNY-current-252",
    "nasdaq-GBNY-historical-252", "nasdaq-ISPOW-current-252",
)
PROPOSAL_QUOTA = 16_385
INITIAL_FRONTIER = 1_000
MAXIMUM_FRONTIER = 16_384
LOGICAL_FRONTIERS = (1_000, 2_000, 4_000, 8_000, 16_000, 16_384)
SEED_ROWS = 512
TOLERANCE = 1e-12
RESIDENT_RESERVE_BYTES = 1024 ** 3
FORWARD_LIMIT = 120.0
REVERSE_LIMIT = 60.0
RSS_LIMIT = 1_536.0
COMPONENT_NAMES = frozenset({
    "stage", "price", "candle_volatility", "volume_shock",
    "market_context", "structural", "coarse",
})
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    "NUMBA_NUM_THREADS", "NUMBA_THREADING_LAYER",
)
REQUIRED_RUNTIME_FILES = frozenset({
    "experiments/m04r/m04r13_threaded_certified_exposed.py",
    "experiments/m04r/compare_m04r13_threaded_certified_exposed.py",
    "experiments/m04r/verify_m04r13_threaded_certified_exposed.py",
    "experiments/m04r/m04r13_finite_threshold_diagnostic.py",
    "experiments/m04r/m04r12_quota_ladder_poc.py",
})
AUTHORITY_VERIFICATION_SHA256 = (
    "b8d879c0f00cf848d39dbd0171d67d9b138f7ebeadd206137566cdddf98a36ee"
)
AUTHORITY_VERIFICATION_DIGEST = (
    "99d11756ed7542714635eca3fb75a43faed93881121d1f22d8d87426f4d6b190"
)
AUTHORITY_MATRIX_DIGEST = (
    "ae60b7cda755c2deabc7bf9335701d34eea73073664582933eb5ba984e7f9b29"
)
AUTHORITY_SEAL_DIGEST = (
    "83501b4cf620607c4ee14cc837d1d5c242460dc867aca22a51b3fbbf924be2c9"
)
AUTHORITY_CASE_BINDINGS = {
    "3307023dbe2164d025e788da": (
        "1aeeddea785923e7e090adc7ffd57fcac410ae42a336c2d03d72236ec01e5b0e",
        "c78e1e8a4f21aaedb8df88ffa9514a1fe0de70c15d67ab010d60b4127a1de91b",
    ),
    "3618af07dedd52fb3bdb1ccd": (
        "c256fe6cba1482a1a090ebfbf89a337700f43bb8150a7c280d7ffe405ac65ed0",
        "b8bcff4c2f1b69109b3d5514d474e2cca4edf0b952f715ab54f4f3cc94dcc127",
    ),
    "9d7365581643bd93e85beb67": (
        "f5f96b97cb2a140abbe2275c12a7aff3cb08b9c462df374e26da00afc59e9e3b",
        "1e704d8a73667eed03488ac990370f274daf06855cac1a1e110c44f09f058141",
    ),
    "99a0838725a09570b4a075ff": (
        "35dc2f84bdc092aec063a6a1c6f8d188489e5c8ed3cbaab3ff16ef94bfaae968",
        "f46cb50c6dbbc3b8af8c521e605e941170bc817588ca9056a82de4fc8945f2f7",
    ),
}


class VerificationError(ValueError):
    """Terminal M04R-13 evidence failed independent reconstruction."""


def _without(value: Mapping[str, Any], omitted: set[str]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in omitted}


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _is_digest(value: Any) -> bool:
    if type(value) is not str or len(value) != 64 or value.lower() != value:
        return False
    try:
        return len(bytes.fromhex(value)) == 32
    except ValueError:
        return False


def _nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _number(value: Any) -> bool:
    return type(value) in {int, float} and isfinite(value)


def _timestamp(value: Any, label: str) -> datetime:
    if type(value) is not str:
        raise VerificationError(f"{label} timestamp is absent")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise VerificationError(f"{label} timestamp is malformed") from exc
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise VerificationError(f"{label} timestamp is not UTC")
    return parsed


def _read_json_sha(path: Path) -> tuple[dict[str, Any], str]:
    """Read one regular file once and parse/hash the exact same bytes."""
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise VerificationError(f"cannot open JSON: {path}") from exc
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns,
        item.st_ctime_ns, item.st_mode,
    )
    if not stat.S_ISREG(before.st_mode) or identity(before) != identity(after):
        raise VerificationError(f"JSON identity differs: {path}")

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in values:
            if key in result:
                raise VerificationError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    def invalid(value: str) -> Any:
        raise VerificationError(f"non-finite JSON number: {value}")

    raw = b"".join(chunks)
    try:
        payload = json.loads(
            raw, object_pairs_hook=pairs, parse_constant=invalid,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSON: {path}") from exc
    if type(payload) is not dict:
        raise VerificationError(f"JSON object differs: {path}")
    return payload, sha256(raw).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return _read_json_sha(path)[0]


def _atomic_create(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write((json.dumps(
                dict(payload), indent=2, sort_keys=True, allow_nan=False,
            ) + "\n").encode())
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


def _exact_tree(root: Path, files: set[str], directories: set[str]) -> None:
    if root.is_symlink() or not root.is_dir():
        raise VerificationError("evidence root is linked or absent")
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        relative = str(path.relative_to(root))
        if stat.S_ISLNK(mode):
            raise VerificationError("evidence tree contains a symlink")
        if stat.S_ISREG(mode):
            observed_files.add(relative)
        elif stat.S_ISDIR(mode):
            observed_directories.add(relative)
        else:
            raise VerificationError("evidence tree contains a special entry")
    if observed_files != files or observed_directories != directories:
        raise VerificationError("terminal evidence tree differs")


def _plain_existing_path(path: Path, label: str) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise VerificationError(f"{label} path is absent") from exc
        if stat.S_ISLNK(mode):
            raise VerificationError(f"{label} path contains a symlink")


def _source_generation_identity(store_root: Path) -> dict[str, Any]:
    generation = store_root / "generations" / GENERATION_ID
    if any((
        store_root.is_symlink(), generation.is_symlink(),
        store_root.resolve() != store_root.absolute(),
        generation.resolve() != generation.absolute(),
        not store_root.is_dir(), not generation.is_dir(),
    )):
        raise VerificationError("durable packed source root differs")
    names = {"manifest.json", "bound-rows.bin", "overflow-exact-fallback.bin"}
    entries = {path.name: path for path in generation.iterdir()}
    if set(entries) != names:
        raise VerificationError("durable packed generation tree differs")
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode")
    result: dict[str, Any] = {}
    for name, path in {"store": store_root, "generation": generation, **entries}.items():
        value = path.lstat()
        if name not in {"store", "generation"} and (
            stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode)
        ):
            raise VerificationError("durable packed generation entry differs")
        result[name] = tuple(getattr(value, field) for field in fields)
    return result


def _execution_policy() -> dict[str, Any]:
    return {
        "child_processes": "fresh_serial_spawn", "parent_max_workers": 1,
        "proposal_threads": 8, "exact_workers": 8,
        "forward_block_rows": 4_096, "reverse_block_rows": 4_097,
        "route_quotas": {"composite": PROPOSAL_QUOTA},
        "branch_aware": True, "initial_frontier_rows": INITIAL_FRONTIER,
        "maximum_frontier_rows": MAXIMUM_FRONTIER,
        "logical_frontiers": list(LOGICAL_FRONTIERS), "seed_rows": SEED_ROWS,
        "tolerance": TOLERANCE, "streaming_fallback": True,
        "performance_limits": {
            "forward_proposal_seconds": FORWARD_LIMIT,
            "reverse_proposal_seconds": REVERSE_LIMIT,
            "process_rss_mb": RSS_LIMIT,
        },
    }


def _environment() -> dict[str, Any]:
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


def _run_git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments], cwd=repository, text=True,
        capture_output=True, check=False,
    )


def _git_blob(repository: Path, commit: str, relative: str) -> bytes:
    completed = subprocess.run(
        ["git", "cat-file", "blob", f"{commit}:{relative}"],
        cwd=repository, capture_output=True, check=False,
    )
    if completed.returncode:
        raise VerificationError(f"historical implementation blob is absent: {relative}")
    return completed.stdout


def _validate_git(repository: Path, expected: Mapping[str, Any]) -> None:
    if type(expected) is not dict or set(expected) != {
        "implementation_commit", "files", "files_digest", "digest",
    }:
        raise VerificationError("preregistered Git binding fields differ")
    files = expected.get("files")
    if type(files) is not dict or not files or any(
        type(path) is not str or Path(path).is_absolute() or ".." in Path(path).parts
        or not _is_digest(digest) for path, digest in files.items()
    ):
        raise VerificationError("preregistered implementation manifest differs")
    tracked_src = _run_git(
        repository, "ls-files", "--", "src/market_analogues/*.py",
    )
    expected_paths = REQUIRED_RUNTIME_FILES | set(tracked_src.stdout.splitlines())
    if tracked_src.returncode or set(files) != expected_paths:
        raise VerificationError("preregistered implementation file set differs")
    prereg = str(PREREG_RELATIVE)
    diagnostic = str(DIAGNOSTIC_RELATIVE)
    checks = (
        ("ls-files", "--error-unmatch", "--", *sorted((*files, prereg, diagnostic))),
        ("diff", "--quiet"), ("diff", "--cached", "--quiet"),
    )
    if any(_run_git(repository, *command).returncode for command in checks):
        raise VerificationError("verification requires a clean tracked tree")
    head = _run_git(repository, "rev-parse", "HEAD")
    parent = _run_git(repository, "rev-parse", "HEAD^")
    introductions = _run_git(
        repository, "log", "--diff-filter=A", "--format=%H", "--", prereg,
    )
    diagnostic_introductions = _run_git(
        repository, "log", "--diff-filter=A", "--format=%H", "--", diagnostic,
    )
    introduced = [row for row in introductions.stdout.splitlines() if row]
    if not all((
        head.returncode == 0, parent.returncode == 0,
        introductions.returncode == 0, diagnostic_introductions.returncode == 0,
        len(introduced) == 1,
        introduced[0] == head.stdout.strip(),
        [row for row in diagnostic_introductions.stdout.splitlines() if row]
        == [head.stdout.strip()],
        expected["implementation_commit"] == parent.stdout.strip(),
        expected["files_digest"] == stable_hash(files),
        expected["digest"] == stable_hash(_without(expected, {"digest"})),
    )):
        raise VerificationError("preregistration commit topology differs")
    for relative, digest in files.items():
        path = repository / relative
        blob = _git_blob(repository, expected["implementation_commit"], relative)
        if not path.is_file() or _sha(path) != digest:
            raise VerificationError(f"implementation blob differs: {relative}")
        if sha256(blob).hexdigest() != digest:
            raise VerificationError(f"historical implementation blob differs: {relative}")


def _validate_committed_diagnostic(
    repository: Path, binding: Mapping[str, Any],
) -> None:
    head = _run_git(repository, "rev-parse", "HEAD").stdout.strip()
    blob = _git_blob(repository, head, str(DIAGNOSTIC_RELATIVE))
    if not all((
        type(binding) is dict,
        _is_digest(binding.get("sha256")),
        sha256(blob).hexdigest() == binding.get("sha256"),
        _sha(repository / DIAGNOSTIC_RELATIVE) == binding.get("sha256"),
    )):
        raise VerificationError("committed finite-threshold diagnostic differs")


def _validate_config_bytes(path: Path, expected_sha256: str) -> None:
    if _sha(path) != expected_sha256:
        raise VerificationError("configuration bytes differ")


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


def _validate_finite_diagnostic(
    prereg: Mapping[str, Any], repository: Path,
) -> None:
    binding = prereg.get("finite_threshold_diagnostic")
    if not all((
        type(binding) is dict,
        set(binding or {}) == {"path", "sha256", "result_digest"},
        type((binding or {}).get("path")) is str,
        (binding or {}).get("path")
        == str((repository / DIAGNOSTIC_RELATIVE).resolve()),
        _is_digest((binding or {}).get("sha256")),
        _is_digest((binding or {}).get("result_digest")),
    )):
        raise VerificationError("finite-threshold diagnostic binding differs")
    payload, observed_sha = _read_json_sha(Path(binding["path"]))
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
    deterministic = _without(payload, {"created_at", "result_digest"})
    if not all((
        observed_sha == binding["sha256"], set(payload) == required,
        payload.get("schema_version") == DIAGNOSTIC_SCHEMA,
        payload.get("status") == "truth_blind_finite_thresholds_verified",
        payload.get("development_only") is True,
        payload.get("authority_or_outcome_paths_accepted") is False,
        payload.get("production_promotion_authorized") is False,
        payload.get("real_forward_outcomes_accessed") is False,
        payload.get("implementation_git") == prereg.get("git"),
        payload.get("environment") == prereg.get("environment"),
        payload.get("execution") == prereg.get("execution"),
        payload.get("config_sha256") == prereg.get("config_sha256"),
        payload.get("diagnostic_path") == binding["path"],
        payload.get("roots") == {
            "config_path": str((repository / CONFIG_RELATIVE).resolve()),
            "registry_root": str((repository / REGISTRY_RELATIVE).resolve()),
            "source_full_root": str((repository / SOURCE_FULL_RELATIVE).resolve()),
            "resident_root": str(RESIDENT_ROOT.resolve()),
        },
        payload.get("generation_id") == GENERATION_ID,
        payload.get("provenance_digest") == PROVENANCE_DIGEST,
        payload.get("registry_digest") == REGISTRY_DIGEST,
        payload.get("registry_cases_digest") == prereg.get("registry_cases_digest"),
        payload.get("query_ids") == list(QUERY_IDS),
        payload.get("case_ids") == list(CASE_IDS),
        payload.get("resident") == {
            "content_digest": prereg.get("resident_content_digest"),
            "ready_digest": prereg.get("resident_ready_digest"),
            "identity_digest": prereg.get("resident_identity_digest"),
        },
        type(payload.get("cases")) is list
        and len(payload["cases"]) == len(QUERY_IDS),
        payload.get("all_thresholds_finite") is True,
        payload.get("all_semantic_checks_passed") is True,
        payload.get("result_digest") == binding["result_digest"],
        payload.get("result_digest") == stable_hash(deterministic),
    )):
        raise VerificationError("finite-threshold diagnostic reconstruction differs")
    _timestamp(payload.get("created_at"), "finite-threshold diagnostic")
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
    expected_bindings = _query_bindings(prereg)
    for ordinal, case in enumerate(payload["cases"]):
        rounds = case.get("rounds") if type(case) is dict else None
        closures = case.get("threshold_closure_passes") if type(case) is dict else None
        digests = (
            "forward_candidate_digest", "forward_result_digest",
            "reverse_candidate_digest", "reverse_result_digest",
            "certificate_result_digest", "case_result_digest", "result_digest",
        )
        if not all((
            type(case) is dict, set(case) == case_keys,
            case.get("schema_version") == DIAGNOSTIC_CASE_SCHEMA,
            case.get("ordinal") == ordinal,
            case.get("registry_case_id") == CASE_IDS[ordinal],
            case.get("query_episode_id") == QUERY_IDS[ordinal],
            case.get("query_binding") == expected_bindings[QUERY_IDS[ordinal]],
            all(_is_digest(case.get(key)) for key in digests),
            case.get("forward_candidate_digest")
            == case.get("reverse_candidate_digest"),
            type(rounds) is list and bool(rounds),
            all(type(row) is dict and set(row) == round_keys
                and type(row["frontier_rows"]) is int and row["frontier_rows"] > 0
                and type(row["selected_rows"]) is int
                and 0 <= row["selected_rows"] <= 20
                and _finite_hex(row["constrained_threshold_hex"])
                and _finite_hex(row["next_lower_bound_hex"], optional=True)
                and type(row["certified"]) is bool for row in rounds),
            type(closures) is list,
            all(type(row) is dict and set(row) == closure_keys
                and _finite_hex(row["lower_exclusive_hex"], optional=True)
                and _finite_hex(row["upper_inclusive_hex"])
                and type(row["admitted_rows"]) is int and row["admitted_rows"] >= 0
                and type(row["selected_rows"]) is int
                and 0 <= row["selected_rows"] <= 20
                and _finite_hex(row["resulting_threshold_hex"])
                and _finite_hex(
                    row["minimum_packed_unclassified_bound_hex"], optional=True,
                )
                and _finite_hex(row["minimum_native_pruned_bound_hex"], optional=True)
                and type(row["certified"]) is bool for row in closures),
            _finite_hex(case.get("stop_threshold_hex")),
            _finite_hex(case.get("next_lower_bound_hex"), optional=True),
            _finite_hex(case.get("minimum_native_pruned_bound_hex"), optional=True),
            case.get("semantic_passed") is True,
            type(case.get("performance_passed_diagnostic")) is bool,
            case.get("all_thresholds_finite") is True,
            case.get("real_forward_outcomes_accessed") is False,
            case.get("result_digest") == stable_hash(
                _without(case, {"result_digest"})
            ),
        )):
            raise VerificationError("finite-threshold diagnostic case differs")
        final_round = rounds[-1]
        if closures:
            final_closure = closures[-1]
            final_next = min(
                (value for value in (
                    final_closure["minimum_packed_unclassified_bound_hex"],
                    final_closure["minimum_native_pruned_bound_hex"],
                ) if value is not None),
                key=float.fromhex, default=None,
            )
            terminal = all((
                all(row["certified"] is False for row in rounds),
                all(row["certified"] is False for row in closures[:-1]),
                final_closure["certified"] is True,
                final_closure["selected_rows"] == 20,
                final_closure["resulting_threshold_hex"]
                == case["stop_threshold_hex"],
                case["next_lower_bound_hex"] == final_next,
                closures[0]["lower_exclusive_hex"] is None,
                closures[0]["upper_inclusive_hex"] == (
                    float.fromhex(final_round["constrained_threshold_hex"])
                    + TOLERANCE
                ).hex(),
                all(row["upper_inclusive_hex"] == (
                    float.fromhex(prior["resulting_threshold_hex"])
                    + TOLERANCE
                ).hex() and float.fromhex(row["upper_inclusive_hex"])
                    > float.fromhex(prior["upper_inclusive_hex"])
                    for prior, row in zip(closures, closures[1:])),
            ))
        else:
            terminal = all((
                all(row["certified"] is False for row in rounds[:-1]),
                final_round["certified"] is True,
                final_round["selected_rows"] == 20,
                final_round["constrained_threshold_hex"]
                == case["stop_threshold_hex"],
                final_round["next_lower_bound_hex"] == case["next_lower_bound_hex"],
            ))
        frontiers = [row["frontier_rows"] for row in rounds]
        if not all((
            terminal,
            list(dict.fromkeys(frontiers))
            == list(LOGICAL_FRONTIERS)[:len(dict.fromkeys(frontiers))],
            all(left <= right for left, right in zip(frontiers, frontiers[1:])),
            all(row["lower_exclusive_hex"]
                == (None if index == 0 else closures[index - 1]["upper_inclusive_hex"])
                for index, row in enumerate(closures)),
        )):
            raise VerificationError("finite-threshold diagnostic terminal differs")


def _validate_preregistration(
    prereg: Mapping[str, Any], repository: Path, *, enforce_topology: bool,
    environment: Mapping[str, Any],
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
    if not all((
        set(prereg) == required,
        prereg.get("schema_version") == PREREG_SCHEMA,
        prereg.get("status") == "frozen_before_producer",
        prereg.get("development_only") is True,
        prereg.get("truth_roots_allowed_in_producer") is False,
        prereg.get("production_promotion_authorized") is False,
        prereg.get("real_forward_outcomes_accessed") is False,
        type(roots) is dict,
        set(roots or {}) == {
            "registry_root", "source_full_root", "resident_root", "output_root",
        },
        all(type(value) is str and Path(value).is_absolute()
            for value in (roots or {}).values()),
        prereg.get("generation_id") == GENERATION_ID,
        prereg.get("provenance_digest") == PROVENANCE_DIGEST,
        prereg.get("reserve_bytes") == RESIDENT_RESERVE_BYTES,
        prereg.get("registry_digest") == REGISTRY_DIGEST,
        prereg.get("query_ids") == list(QUERY_IDS),
        prereg.get("execution") == _execution_policy(),
        prereg.get("environment") == dict(environment),
        all(_is_digest(prereg.get(key)) for key in (
            "config_sha256", "registry_cases_digest", "resident_content_digest",
            "resident_ready_digest", "resident_identity_digest",
            "preregistration_digest",
        )),
        prereg.get("preregistration_digest")
        == stable_hash(_without(prereg, {"preregistration_digest"})),
    )):
        raise VerificationError("preregistration reconstruction differs")
    if enforce_topology:
        expected_roots = {
            "registry_root": str((repository / REGISTRY_RELATIVE).resolve()),
            "source_full_root": str((repository / SOURCE_FULL_RELATIVE).resolve()),
            "resident_root": str(RESIDENT_ROOT.resolve()),
            "output_root": str((repository / OUTPUT_RELATIVE).resolve()),
        }
        if not all((
            prereg["config_path"] == str((repository / CONFIG_RELATIVE).resolve()),
            roots == expected_roots,
            prereg["config_sha256"] == _sha(repository / CONFIG_RELATIVE),
        )):
            raise VerificationError("preregistered path topology differs")
    _validate_finite_diagnostic(prereg, repository)


def _validate_registry(root: Path) -> tuple[list[dict[str, Any]], str]:
    registry = _read_json(root / "query-registry.json")
    if not all((
        registry.get("schema_version") == "m04r-untouched-authority-registry-v1",
        registry.get("registry_digest") == REGISTRY_DIGEST,
        registry.get("passed") is True, registry.get("failures") == [],
        registry.get("real_forward_outcomes_accessed") is False,
        type(registry.get("cases_data")) is list,
        len(registry.get("cases_data", [])) == 60,
    )):
        raise VerificationError("registry prerequisite differs")
    deterministic = _without(registry, {
        "passed", "failures", "registry_digest", "cases_data",
        "contamination_ledger",
    })
    expected = stable_hash({
        **deterministic, "cases_data": registry["cases_data"],
        "contamination_ledger": registry["contamination_ledger"],
    })
    if expected != REGISTRY_DIGEST:
        raise VerificationError("registry digest reconstruction differs")
    by_id = {str(case.get("episode_id")): case for case in registry["cases_data"]}
    if len(by_id) != 60 or any(value not in by_id for value in QUERY_IDS):
        raise VerificationError("registry query identities differ")
    cases = [dict(by_id[value]) for value in QUERY_IDS]
    if tuple(str(case.get("case_id")) for case in cases) != CASE_IDS:
        raise VerificationError("registry case identities differ")
    return cases, stable_hash(cases)


def _query_bindings(prereg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    roots = prereg["roots"]
    cases, cases_digest = _validate_registry(Path(roots["registry_root"]))
    if cases_digest != prereg["registry_cases_digest"]:
        raise VerificationError("registry case digest differs")
    config = load_config(Path(prereg["config_path"]))
    source = source_from_spec(config.datasets["nasdaq"])
    result: dict[str, dict[str, Any]] = {}
    for case in cases:
        instrument = InstrumentKey("nasdaq", str(case["symbol"]))
        episode = build_episode(
            source, instrument, str(case["cutoff"]), int(case["lookback"]),
            str(case["representation_version"]),
        )
        request = SearchQuery(
            episode.key, ("nasdaq",), ("A", "B"), 20,
            False, True, 3, 60,
        )
        representation = represent(episode)
        packed = PackedBoundQuery(
            episode.key.id, episode.key.instrument.source_symbol,
            int(episode.bars.timestamp.iloc[0].value),
            int(latest_eligible_cutoff(episode, 60).value), representation,
            request.quality_tiers,
        )
        benchmark = source.load_benchmark()
        deterministic = {
            "query_stock_prefix": asdict(causal_prefix_digest(
                source.load(instrument), episode.key.cutoff,
            )),
            "query_benchmark_prefix": (
                asdict(causal_prefix_digest(benchmark, episode.key.cutoff))
                if benchmark is not None else None
            ),
            "request": {
                "search_datasets": list(request.search_datasets),
                "quality_tiers": list(request.quality_tiers), "top_k": 20,
                "cross_dataset": False, "deduplicate_overlaps": True,
                "max_per_instrument": 3, "minimum_history_gap_bars": 60,
            },
            "packed_provenance_digest": prereg["provenance_digest"],
            "query_representation_digest": representation_input_digest(
                representation,
            ),
        }
        binding = {
            **deterministic,
            "packed_query_input_digest": _packed_query_input_digest(packed),
            "certified_input_digest": stable_hash(deterministic),
        }
        if not all((
            episode.key.id == case["episode_id"],
            binding["query_stock_prefix"] == case["stock_prefix"],
            binding["query_benchmark_prefix"] == case["benchmark_prefix"],
        )):
            raise VerificationError("query causal binding differs")
        result[episode.key.id] = binding
    return result


def _packed_universe(
    prereg: Mapping[str, Any], cases: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Authenticate returned IDs against the immutable full packed generation."""
    if len(cases) != len(QUERY_IDS):
        raise VerificationError("case set differs before universe authentication")
    requested: dict[str, set[str]] = {}
    for query_id, case in zip(QUERY_IDS, cases, strict=True):
        rows = case.get("matches")
        if type(rows) is not list:
            raise VerificationError("matches are absent before universe authentication")
        identifiers: set[str] = set()
        for row in rows:
            identifier = row.get("episode_id") if type(row) is dict else None
            try:
                valid = type(identifier) is str and len(bytes.fromhex(identifier)) == 12
            except ValueError:
                valid = False
            if not valid:
                raise VerificationError("match identity differs before universe authentication")
            identifiers.add(identifier)
        requested[query_id] = identifiers

    roots = prereg["roots"]
    loaded = load_packed_generation(
        Path(roots["source_full_root"]) / "store", GENERATION_ID,
        expected_provenance_digest=PROVENANCE_DIGEST,
        verify_content=True, validate_records=True,
    )
    if loaded.generation_id != GENERATION_ID or (
        loaded.manifest.get("provenance_digest") != PROVENANCE_DIGEST
    ):
        raise VerificationError("source packed generation binding differs")

    all_identifiers = set().union(*requested.values())
    encoded = np.asarray(
        [np.void(bytes.fromhex(value)) for value in sorted(all_identifiers)],
        dtype="V12",
    )
    selected_main = loaded.rows[np.isin(loaded.rows["episode_id"], encoded)]
    selected_overflow = loaded.overflow[
        np.isin(loaded.overflow["episode_id"], encoded)
    ]
    records: dict[str, tuple[Any, bool]] = {}
    for overflow, selected in ((False, selected_main), (True, selected_overflow)):
        for record in selected:
            identifier = decode_episode_id(record["episode_id"])
            if identifier in records:
                raise VerificationError("packed match identity is duplicated")
            symbol_id = int(record["symbol_id"])
            quality_code = int(record["quality_tier"])
            if symbol_id >= len(loaded.symbols) or quality_code not in TIER_NAMES:
                raise VerificationError("packed match metadata differs")
            records[identifier] = (record, overflow)
    if set(records) != all_identifiers:
        raise VerificationError("returned match is absent from frozen packed generation")

    registry_cases, _ = _validate_registry(Path(roots["registry_root"]))
    config = load_config(Path(prereg["config_path"]))
    source = source_from_spec(config.datasets["nasdaq"])
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for query_id, registry_case in zip(QUERY_IDS, registry_cases, strict=True):
        instrument = InstrumentKey("nasdaq", str(registry_case["symbol"]))
        episode = build_episode(
            source, instrument, str(registry_case["cutoff"]),
            int(registry_case["lookback"]),
            str(registry_case["representation_version"]),
        )
        query = PackedBoundQuery(
            episode.key.id, episode.key.instrument.source_symbol,
            int(episode.bars.timestamp.iloc[0].value),
            int(latest_eligible_cutoff(episode, 60).value), represent(episode),
            ("A", "B"),
        )
        symbol_id = (
            loaded.symbols.index(query.symbol)
            if query.symbol in loaded.symbols else None
        )
        main_identifiers = [
            identifier for identifier in requested[query_id]
            if records[identifier][1] is False
        ]
        main_rows = np.asarray(
            [records[identifier][0] for identifier in main_identifiers],
            dtype=loaded.rows.dtype,
        )
        main_bounds = (
            packed_branch_aware_lower_bounds(query.representation, main_rows).totals
            if len(main_rows) else np.asarray([], dtype=np.float64)
        )
        bounds = {
            identifier: float(value)
            for identifier, value in zip(main_identifiers, main_bounds, strict=True)
        }
        authenticated: dict[str, dict[str, Any]] = {}
        query_bytes = bytes.fromhex(query.episode_id)
        for identifier in requested[query_id]:
            record, overflow = records[identifier]
            cutoff_ns = int(record["cutoff_ns"])
            record_symbol_id = int(record["symbol_id"])
            eligible = all((
                bytes(record["episode_id"]) != query_bytes,
                cutoff_ns <= query.latest_eligible_ns,
                not (
                    symbol_id is not None
                    and record_symbol_id == symbol_id
                    and cutoff_ns >= query.query_start_ns
                ),
                int(record["quality_tier"]) in {
                    1 if value == "A" else 2 for value in query.quality_tiers
                },
            ))
            bound = 0.0 if overflow else bounds[identifier]
            if not isfinite(bound) or bound < 0:
                raise VerificationError("packed match lower bound differs")
            authenticated[identifier] = {
                "symbol": loaded.symbols[record_symbol_id],
                "cutoff_ns": cutoff_ns,
                "quality_tier": TIER_NAMES[int(record["quality_tier"])],
                "overflow_fallback": overflow,
                "lower_bound": bound,
                "eligible": eligible,
            }
        output[query_id] = authenticated
    return output


def _reconstruct_closure_scans(
    prereg: Mapping[str, Any], cases: Sequence[Mapping[str, Any]],
) -> None:
    """Independently replay each recorded closure band in reverse traversal."""
    registry_cases, _ = _validate_registry(Path(prereg["roots"]["registry_root"]))
    config = load_config(Path(prereg["config_path"]))
    source = source_from_spec(config.datasets["nasdaq"])
    store_root = Path(prereg["roots"]["source_full_root"]) / "store"
    for query_id, registry_case, case in zip(
        QUERY_IDS, registry_cases, cases, strict=True,
    ):
        certificate = case.get("certificate")
        closures = (
            certificate.get("threshold_closure_passes")
            if type(certificate) is dict else None
        )
        if not closures:
            continue
        instrument = InstrumentKey("nasdaq", str(registry_case["symbol"]))
        episode = build_episode(
            source, instrument, str(registry_case["cutoff"]),
            int(registry_case["lookback"]),
            str(registry_case["representation_version"]),
        )
        packed = PackedBoundQuery(
            episode.key.id, episode.key.instrument.source_symbol,
            int(episode.bars.timestamp.iloc[0].value),
            int(latest_eligible_cutoff(episode, 60).value), represent(episode),
            ("A", "B"),
        )
        proposal = case.get("forward_proposal")
        rows = proposal.get("candidates") if type(proposal) is dict else None
        if type(rows) is not list or len(rows) < MAXIMUM_FRONTIER:
            raise VerificationError("closure prefix proposal is incomplete")
        excluded = frozenset(
            str(row.get("episode_id")) for row in rows[:MAXIMUM_FRONTIER]
            if type(row) is dict
        )
        if len(excluded) != MAXIMUM_FRONTIER:
            raise VerificationError("closure prefix identity set differs")
        expected_exclusions_digest = stable_hash(sorted(excluded))
        for closure in closures:
            if closure.get("excluded_prefix_digest") != expected_exclusions_digest:
                raise VerificationError("closure exclusion digest differs")
            report = scan_packed_bound_threshold(
                store_root, GENERATION_ID, packed,
                lower_exclusive=closure["lower_exclusive"],
                upper_inclusive=closure["upper_inclusive"],
                excluded_episode_ids=excluded, block_rows=4_097,
                block_order="reverse", branch_aware=True,
                # The universe pass hashes the full generation once; the
                # surrounding identity lease detects mutation during replays.
                verify_content=False,
                expected_provenance_digest=PROVENANCE_DIGEST,
                consume=lambda _rows: None,
            )
            if not all((
                report.query_episode_id == query_id,
                report.input_digest == _packed_query_input_digest(packed),
                report.exclusions_digest == expected_exclusions_digest,
                report.lower_exclusive == closure["lower_exclusive"],
                report.upper_inclusive == closure["upper_inclusive"],
                report.eligible_rows == certificate["eligible_candidates"],
                report.excluded_eligible_rows == len(excluded),
                report.admitted_rows == closure["admitted_rows"],
                report.minimum_above_upper
                == closure["minimum_packed_unclassified_bound"],
                report.admitted_set_digest == closure["admitted_set_digest"],
                report.result_digest == closure["scan_result_digest"],
            )):
                raise VerificationError("closure threshold replay differs")


def _reconstruct_proposals(
    prereg: Mapping[str, Any], cases: Sequence[Mapping[str, Any]],
) -> None:
    """Recompute every proposal through an independent physical traversal."""
    registry_cases, _ = _validate_registry(Path(prereg["roots"]["registry_root"]))
    config = load_config(Path(prereg["config_path"]))
    source = source_from_spec(config.datasets["nasdaq"])
    store_root = Path(prereg["roots"]["source_full_root"]) / "store"
    omitted = {"block_rows", "block_order", "elapsed_seconds", "peak_rss_mb"}
    for query_id, registry_case, case in zip(
        QUERY_IDS, registry_cases, cases, strict=True,
    ):
        instrument = InstrumentKey("nasdaq", str(registry_case["symbol"]))
        episode = build_episode(
            source, instrument, str(registry_case["cutoff"]),
            int(registry_case["lookback"]),
            str(registry_case["representation_version"]),
        )
        packed = PackedBoundQuery(
            episode.key.id, episode.key.instrument.source_symbol,
            int(episode.bars.timestamp.iloc[0].value),
            int(latest_eligible_cutoff(episode, 60).value), represent(episode),
            ("A", "B"),
        )
        report = scan_packed_bound_proposals(
            store_root, GENERATION_ID, packed,
            route_quotas={"composite": PROPOSAL_QUOTA}, block_rows=4_093,
            block_order="reverse", branch_aware=True,
            # Full content was verified by _packed_universe under the same
            # pre/post immutable generation identity observation.
            verify_content=False,
            expected_provenance_digest=PROVENANCE_DIGEST,
        )
        candidates = [{
            "episode_id": row.episode_id, "symbol": row.symbol,
            "cutoff_ns": row.cutoff_ns, "quality_tier": row.quality_tier,
            "lower_bound_hex": row.lower_bound.hex(),
            "routes": list(row.routes),
            "overflow_fallback": row.overflow_fallback,
        } for row in report.candidates]
        observed = {
            "schema_version": report.schema_version,
            "generation_id": report.generation_id,
            "query_episode_id": report.query_episode_id,
            "candidates": candidates, "rows_scanned": report.rows_scanned,
            "eligible_rows": report.eligible_rows,
            "eligible_main_rows": report.eligible_main_rows,
            "eligible_overflow_rows": report.eligible_overflow_rows,
            "route_counts": dict(report.route_counts),
            "route_quotas": dict(report.route_quotas),
            "block_rows": report.block_rows, "block_order": report.block_order,
            "elapsed_seconds": report.elapsed_seconds,
            "peak_rss_mb": report.peak_rss_mb,
            "candidate_digest": report.candidate_digest,
            "result_digest": report.result_digest,
            "contract_digest": report.contract_digest,
            "input_digest": report.input_digest,
        }
        expected = case.get("forward_proposal")
        if type(expected) is not dict or (
            _without(observed, omitted) != _without(expected, omitted)
        ):
            raise VerificationError("independent proposal replay differs")


def _validate_resident(value: Mapping[str, Any]) -> None:
    required = {
        "ready_digest", "content_digest", "seal_digest", "ready_file_sha256",
        "lease", "store_root", "identity_digest",
    }
    lease = value.get("lease")
    if not all((
        set(value) == required, type(lease) is dict,
        set(lease or {}) == {
            "schema_version", "ready_digest", "ready_file_sha256",
            "content_digest", "files", "lease_digest",
        },
        lease.get("schema_version") == "m04r-resident-file-identity-lease-v1",
        type(lease.get("files")) is dict and bool(lease.get("files")),
        lease.get("ready_digest") == value.get("ready_digest"),
        lease.get("ready_file_sha256") == value.get("ready_file_sha256"),
        lease.get("content_digest") == value.get("content_digest"),
        lease.get("lease_digest") == stable_hash(_without(lease, {"lease_digest"})),
        value.get("identity_digest")
        == stable_hash(_without(value, {"identity_digest"})),
        all(_is_digest(value.get(key)) for key in (
            "ready_digest", "content_digest", "seal_digest",
            "ready_file_sha256", "identity_digest",
        )),
        type(value.get("store_root")) is str and Path(value["store_root"]).is_absolute(),
    )):
        raise VerificationError("resident snapshot differs")
    identity_keys = {
        "path", "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns",
        "st_mode",
    }
    if any(
        type(identity) is not dict or set(identity) != identity_keys
        or type(identity["path"]) is not str
        or any(not _nonnegative_int(identity[key])
               for key in identity_keys - {"path"})
        for identity in lease["files"].values()
    ):
        raise VerificationError("resident file identity differs")


def _observe_resident(prereg: Mapping[str, Any]) -> dict[str, Any]:
    roots = prereg["roots"]
    resident_root = Path(roots["resident_root"])
    ready_path = resident_root / "READY.json"
    observed = observe_ready_strict(ready_path)
    lease = resident_file_identity_lease(ready_path)
    source = load_packed_generation(
        Path(roots["source_full_root"]) / "store", GENERATION_ID,
        expected_provenance_digest=PROVENANCE_DIGEST,
        verify_content=True, validate_records=False,
    )
    mirrored = load_packed_generation(
        resident_root / "store", GENERATION_ID,
        expected_provenance_digest=PROVENANCE_DIGEST,
        verify_content=True, validate_records=False,
    )
    observed_after = observe_ready_strict(ready_path)
    lease_after = resident_file_identity_lease(ready_path)
    deterministic = {
        "ready_digest": observed["ready_digest"],
        "content_digest": observed["content_digest"],
        "seal_digest": observed["seal_digest"],
        "ready_file_sha256": observed["ready_file_sha256"],
        "lease": lease, "store_root": str((resident_root / "store").resolve()),
    }
    value = {**deterministic, "identity_digest": stable_hash(deterministic)}
    if not all((
        source.manifest == mirrored.manifest,
        observed_after == observed, lease_after == lease,
        observed["payload"]["content"]["generation_id"] == GENERATION_ID,
        observed["payload"]["content"]["provenance_digest"]
        == PROVENANCE_DIGEST,
        observed["payload"]["seal"]["mirror_store_root"]
        == str((resident_root / "store").resolve()),
        observed["payload"]["seal"]["reserve_bytes"]
        == RESIDENT_RESERVE_BYTES,
    )):
        raise VerificationError("live resident full validation differs")
    _validate_resident(value)
    return value


def _candidate_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    try:
        proposals = tuple(BoundProposal(
            str(row["episode_id"]), str(row["symbol"]), int(row["cutoff_ns"]),
            str(row["quality_tier"]), float.fromhex(str(row["lower_bound_hex"])),
            tuple(row["routes"]), bool(row["overflow_fallback"]),
        ) for row in rows)
    except (KeyError, TypeError, ValueError) as exc:
        raise VerificationError("proposal candidate encoding differs") from exc
    return bound_proposal_candidate_digest(proposals)


def _validate_proposal(
    value: Mapping[str, Any], query_id: str, input_digest: str, order: str,
) -> tuple[BoundProposal, ...]:
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
        type(value.get("candidates")) is list,
        all(type(row) is dict and set(row) == candidate_keys
            for row in value.get("candidates", [])),
    )):
        raise VerificationError("proposal fields differ")
    proposals: list[BoundProposal] = []
    for row in value["candidates"]:
        try:
            lower = float.fromhex(row["lower_bound_hex"])
            identifier = bytes.fromhex(row["episode_id"])
        except (TypeError, ValueError) as exc:
            raise VerificationError("proposal candidate encoding differs") from exc
        if not all((
            len(identifier) == 12, type(row["symbol"]) is str and row["symbol"],
            _nonnegative_int(row["cutoff_ns"]), row["quality_tier"] in {"A", "B"},
            row["routes"] == ["composite"], type(row["overflow_fallback"]) is bool,
            isfinite(lower) and lower >= 0 and lower.hex() == row["lower_bound_hex"],
        )):
            raise VerificationError("proposal candidate semantics differ")
        proposals.append(BoundProposal(
            row["episode_id"], row["symbol"], row["cutoff_ns"],
            row["quality_tier"], lower, ("composite",), row["overflow_fallback"],
        ))
    contract = packed_bound_search_contract(branch_aware=True)["digest"]
    deterministic = {
        "schema_version": BRANCH_AWARE_SEARCH_SCHEMA_VERSION,
        "contract_digest": contract, "generation_id": GENERATION_ID,
        "query_episode_id": query_id, "rows_scanned": value["rows_scanned"],
        "eligible_rows": value["eligible_rows"],
        "eligible_main_rows": value["eligible_main_rows"],
        "eligible_overflow_rows": value["eligible_overflow_rows"],
        "route_counts": value["route_counts"], "route_quotas": value["route_quotas"],
        "candidate_digest": value["candidate_digest"],
        "real_forward_outcomes_accessed": False, "input_digest": input_digest,
    }
    expected_block = 4_096 if order == "forward" else 4_097
    if not all((
        value["schema_version"] == BRANCH_AWARE_SEARCH_SCHEMA_VERSION,
        value["generation_id"] == GENERATION_ID,
        value["query_episode_id"] == query_id,
        value["contract_digest"] == contract, value["input_digest"] == input_digest,
        value["route_quotas"] == {"composite": PROPOSAL_QUOTA},
        value["route_counts"] == {"composite": len(proposals)},
        all(_nonnegative_int(value[key]) for key in (
            "rows_scanned", "eligible_rows", "eligible_main_rows",
            "eligible_overflow_rows", "block_rows",
        )),
        value["eligible_rows"]
        == value["eligible_main_rows"] + value["eligible_overflow_rows"],
        value["rows_scanned"] >= value["eligible_rows"],
        len(proposals) == min(PROPOSAL_QUOTA, value["eligible_rows"]),
        proposals == sorted(proposals, key=lambda row: (row.lower_bound, row.episode_id)),
        value["block_rows"] == expected_block, value["block_order"] == order,
        _number(value["elapsed_seconds"]) and value["elapsed_seconds"] >= 0,
        _number(value["peak_rss_mb"]) and value["peak_rss_mb"] >= 0,
        value["candidate_digest"] == bound_proposal_candidate_digest(proposals),
        value["result_digest"] == stable_hash(deterministic),
    )):
        raise VerificationError("proposal reconstruction differs")
    return tuple(proposals)


def _validate_matches(matches: Any) -> list[dict[str, Any]]:
    keys = {
        "episode_id", "symbol", "cutoff", "total_distance",
        "component_distances", "alignment", "quality_tier",
    }
    if type(matches) is not list or len(matches) != 20:
        raise VerificationError("match count differs")
    for row in matches:
        try:
            identifier_valid = len(bytes.fromhex(row["episode_id"])) == 12
            cutoff = pd.Timestamp(row["cutoff"])
        except (KeyError, TypeError, ValueError) as exc:
            raise VerificationError("match identity encoding differs") from exc
        if not all((
            type(row) is dict, set(row) == keys, identifier_valid,
            type(row["symbol"]) is str and row["symbol"],
            cutoff.isoformat() == row["cutoff"],
            cutoff.tzinfo is None or (
                cutoff.utcoffset() is not None
                and cutoff.utcoffset().total_seconds() == 0
            ),
            type(row["total_distance"]) is float
            and isfinite(row["total_distance"]) and row["total_distance"] >= 0,
            type(row["component_distances"]) is dict
            and set(row["component_distances"]) == COMPONENT_NAMES,
            all(type(value) is float and isfinite(value) and value >= 0
                for value in row["component_distances"].values()),
            type(row["alignment"]) is list and bool(row["alignment"]),
            all(type(pair) is list and len(pair) == 2
                and all(_nonnegative_int(index) for index in pair)
                for pair in row["alignment"]),
            row["quality_tier"] in {"A", "B"},
        )):
            raise VerificationError("match schema differs")
    if [(row["total_distance"], row["episode_id"]) for row in matches] != sorted(
        (row["total_distance"], row["episode_id"]) for row in matches
    ) or len({row["episode_id"] for row in matches}) != 20:
        raise VerificationError("match order or uniqueness differs")
    return matches


def _validate_certificate(
    certificate: Any, matches: list[dict[str, Any]], query_id: str,
    input_digest: str, forward: tuple[BoundProposal, ...], eligible_rows: int,
) -> None:
    required = {
        "schema_version", "contract_digest", "generation_id", "query_episode_id",
        "input_digest", "eligible_candidates", "exact_evaluated", "safely_pruned",
        "stopped_early", "stop_threshold", "next_lower_bound",
        "maximum_quantized_bound_excess", "materialization_groups", "sparse_symbols",
        "batch_symbols", "rounds", "result_digest", "elapsed_seconds",
        "native_bound_accounting", "minimum_native_pruned_bound",
        "threshold_closure_passes",
    }
    if type(certificate) is not dict or set(certificate) != required:
        raise VerificationError("certificate fields differ")
    contract = certified_packed_search_contract(
        requested_positions=True, vector_lower_bounds=True,
        deferred_alignments=True, compact_scored=True,
        native_bound_deferral=True, streaming_threshold_closure=True,
        branch_aware_packed_bounds=True,
    )
    accounting = certificate["native_bound_accounting"]
    if type(accounting) is not dict or set(accounting) != {
        "native_bound_evaluated", "exact_dtw_evaluated", "native_bound_pruned",
        "packed_bound_pruned",
    } or any(not _nonnegative_int(value) for value in accounting.values()):
        raise VerificationError("native-bound accounting differs")
    rounds = certificate["rounds"]
    round_keys = {
        "frontier_rows", "exact_rows", "next_lower_bound", "constrained_threshold",
        "selected_rows", "certified", "proposal_digest",
    }
    if type(rounds) is not list or not rounds:
        raise VerificationError("logical rounds are absent")
    for row in rounds:
        if not all((
            type(row) is dict, set(row) == round_keys,
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
        )):
            raise VerificationError("logical round schema differs")
        required_candidates = min(
            row["frontier_rows"] + 1, certificate["eligible_candidates"],
        )
        if row["proposal_digest"] != bound_proposal_candidate_digest(
            forward[:required_candidates]
        ):
            raise VerificationError("logical proposal prefix digest differs")
        packed_boundary = (
            forward[row["frontier_rows"]].lower_bound
            if row["frontier_rows"] < certificate["eligible_candidates"]
            and len(forward) > row["frontier_rows"] else None
        )
        if packed_boundary is not None and (
            row["next_lower_bound"] is None
            or row["next_lower_bound"] > packed_boundary + TOLERANCE
        ):
            raise VerificationError("logical next bound exceeds packed boundary")
    distinct = list(dict.fromkeys(row["frontier_rows"] for row in rounds))
    expected: list[int] = []
    frontier = min(INITIAL_FRONTIER, certificate["eligible_candidates"])
    while frontier > 0:
        expected.append(frontier)
        if frontier >= certificate["eligible_candidates"] or frontier >= MAXIMUM_FRONTIER:
            break
        frontier = min(
            MAXIMUM_FRONTIER, frontier * 2, certificate["eligible_candidates"],
        )
    if not all((
        distinct == expected[:len(distinct)],
        all(left["frontier_rows"] <= right["frontier_rows"]
            and left["exact_rows"] <= right["exact_rows"]
            for left, right in zip(rounds, rounds[1:])),
        all(row["selected_rows"] <= row["exact_rows"] for row in rounds),
        all(row["certified"] is False for row in rounds[:-1]),
    )):
        raise VerificationError("logical frontier progression differs")
    closures = certificate["threshold_closure_passes"]
    closure_keys = {
        "lower_exclusive", "upper_inclusive", "admitted_rows",
        "cumulative_native_bound_evaluated", "cumulative_exact_dtw_evaluated",
        "selected_rows", "resulting_threshold",
        "minimum_packed_unclassified_bound", "minimum_native_pruned_bound",
        "excluded_prefix_digest", "admitted_set_digest", "scan_result_digest",
        "certified",
    }
    if type(closures) is not list:
        raise VerificationError("threshold closure list differs")
    expected_exclusion_digest = stable_hash(sorted(
        row.episode_id for row in forward[:MAXIMUM_FRONTIER]
    ))
    prior: Mapping[str, Any] | None = None
    for row in closures:
        optional = (
            "lower_exclusive", "minimum_packed_unclassified_bound",
            "minimum_native_pruned_bound",
        )
        if not all((
            type(row) is dict, set(row) == closure_keys,
            all(type(row[key]) is float and isfinite(row[key]) and row[key] >= 0
                for key in ("upper_inclusive", "resulting_threshold")),
            all(row[key] is None or (
                type(row[key]) is float and isfinite(row[key]) and row[key] >= 0
            ) for key in optional),
            all(_nonnegative_int(row[key]) for key in (
                "admitted_rows", "cumulative_native_bound_evaluated",
                "cumulative_exact_dtw_evaluated", "selected_rows",
            )),
            row["selected_rows"] <= 20,
            row["selected_rows"] <= row["cumulative_exact_dtw_evaluated"],
            row["cumulative_exact_dtw_evaluated"]
            <= row["cumulative_native_bound_evaluated"],
            all(_is_digest(row[key]) for key in (
                "excluded_prefix_digest", "admitted_set_digest", "scan_result_digest",
            )),
            row["excluded_prefix_digest"] == expected_exclusion_digest,
            type(row["certified"]) is bool,
            (
                row["cumulative_native_bound_evaluated"]
                == MAXIMUM_FRONTIER + row["admitted_rows"]
                if prior is None else all((
                    row["lower_exclusive"] == prior["upper_inclusive"],
                    row["upper_inclusive"].hex()
                    == (prior["resulting_threshold"] + TOLERANCE).hex(),
                    row["cumulative_native_bound_evaluated"]
                    == prior["cumulative_native_bound_evaluated"]
                    + row["admitted_rows"],
                    row["cumulative_exact_dtw_evaluated"]
                    >= prior["cumulative_exact_dtw_evaluated"],
                ))
            ),
        )):
            raise VerificationError("threshold closure reconstruction differs")
        prior = row
    final_round = rounds[-1]
    if closures:
        final_closure = closures[-1]
        final_next = min((value for value in (
            final_closure["minimum_packed_unclassified_bound"],
            final_closure["minimum_native_pruned_bound"],
        ) if value is not None), default=None)
        closure_valid = all((
            final_round["certified"] is False,
            final_round["frontier_rows"] == MAXIMUM_FRONTIER,
            final_round["frontier_rows"] < certificate["eligible_candidates"],
            all(row["certified"] is False for row in closures[:-1]),
            final_closure["certified"] is True,
            closures[0]["lower_exclusive"] is None,
            closures[0]["upper_inclusive"].hex()
            == (final_round["constrained_threshold"] + TOLERANCE).hex(),
            all((row["resulting_threshold"] + TOLERANCE)
                > row["upper_inclusive"] for row in closures[:-1]),
            closures[0]["cumulative_exact_dtw_evaluated"]
            >= final_round["exact_rows"],
            final_closure["cumulative_native_bound_evaluated"]
            == accounting["native_bound_evaluated"],
            final_closure["cumulative_exact_dtw_evaluated"]
            == accounting["exact_dtw_evaluated"],
            final_closure["resulting_threshold"] == certificate["stop_threshold"],
            certificate["minimum_native_pruned_bound"]
            == final_closure["minimum_native_pruned_bound"],
            final_closure["selected_rows"] == 20,
        ))
    else:
        packed_boundary = (
            forward[final_round["frontier_rows"]].lower_bound
            if final_round["frontier_rows"] < certificate["eligible_candidates"]
            and len(forward) > final_round["frontier_rows"] else None
        )
        final_next = min((value for value in (
            packed_boundary, certificate["minimum_native_pruned_bound"],
        ) if value is not None), default=None)
        closure_valid = all((
            final_round["certified"] is True,
            final_round["exact_rows"] == accounting["exact_dtw_evaluated"],
            final_round["selected_rows"] == 20,
            final_round["constrained_threshold"] == certificate["stop_threshold"],
        ))
    deterministic = {
        "schema_version": contract["schema_version"],
        "contract_digest": contract["digest"], "generation_id": GENERATION_ID,
        "query_episode_id": query_id, "input_digest": input_digest,
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
    final_bound = certificate["next_lower_bound"]
    count_fields = (
        "eligible_candidates", "exact_evaluated", "safely_pruned",
        "materialization_groups", "sparse_symbols", "batch_symbols",
    )
    if not all((
        certificate["schema_version"] == contract["schema_version"],
        certificate["contract_digest"] == contract["digest"],
        certificate["generation_id"] == GENERATION_ID,
        certificate["query_episode_id"] == query_id,
        certificate["input_digest"] == input_digest,
        all(_nonnegative_int(certificate[key]) for key in count_fields),
        type(certificate["stopped_early"]) is bool,
        type(certificate["stop_threshold"]) is float
        and isfinite(certificate["stop_threshold"]),
        final_bound is None or (type(final_bound) is float and isfinite(final_bound)),
        type(certificate["maximum_quantized_bound_excess"]) is float
        and 0 <= certificate["maximum_quantized_bound_excess"] <= TOLERANCE,
        _number(certificate["elapsed_seconds"])
        and certificate["elapsed_seconds"] >= 0,
        certificate["eligible_candidates"] == eligible_rows,
        len(forward) == min(PROPOSAL_QUOTA, eligible_rows),
        certificate["eligible_candidates"]
        == certificate["exact_evaluated"] + certificate["safely_pruned"],
        accounting["native_bound_evaluated"]
        == accounting["exact_dtw_evaluated"] + accounting["native_bound_pruned"],
        (accounting["native_bound_pruned"] == 0)
        is (certificate["minimum_native_pruned_bound"] is None),
        certificate["eligible_candidates"]
        == accounting["native_bound_evaluated"] + accounting["packed_bound_pruned"],
        certificate["exact_evaluated"] == accounting["exact_dtw_evaluated"],
        certificate["materialization_groups"]
        == certificate["sparse_symbols"] + certificate["batch_symbols"],
        closure_valid,
        certificate["stop_threshold"] == matches[-1]["total_distance"],
        final_bound == final_next,
        certificate["stopped_early"] is (final_next is not None),
        final_next is None or final_next > certificate["stop_threshold"] + TOLERANCE,
        certificate["result_digest"] == stable_hash(deterministic),
    )):
        raise VerificationError("certificate reconstruction differs")


def _performance(metrics: Any) -> bool:
    required = {
        "forward_proposal_seconds", "reverse_proposal_seconds",
        "exact_task_wall_seconds", "case_task_wall_seconds", "process_rss_mb",
    }
    if type(metrics) is not dict or set(metrics) != required or any(
        not _number(value) or value < 0 for value in metrics.values()
    ):
        raise VerificationError("performance metrics differ")
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
        raise VerificationError("performance nesting differs")
    return all((
        metrics["forward_proposal_seconds"] <= FORWARD_LIMIT,
        metrics["reverse_proposal_seconds"] <= REVERSE_LIMIT,
        metrics["process_rss_mb"] <= RSS_LIMIT,
    ))


def _validate_case(
    payload: Mapping[str, Any], ordinal: int, prereg: Mapping[str, Any],
    resident: Mapping[str, Any], binding: Mapping[str, Any],
    universe: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    required = {
        "schema_version", "status", "development_only", "truth_opened",
        "production_promotion_authorized", "real_forward_outcomes_accessed",
        "preregistration_digest", "registry_case_id", "query_episode_id",
        "query_binding", "resident_identity_digest", "lease_digests",
        "forward_proposal", "reverse_proposal", "proposal_semantic_exact",
        "certificate", "matches", "rounds", "certified",
        "streaming_fallback_used", "metrics", "semantic_passed",
        "performance_passed", "created_at", "result_digest",
    }
    query_id = QUERY_IDS[ordinal]
    if not all((
        set(payload) == required, payload.get("schema_version") == CASE_SCHEMA,
        payload.get("status") == "truth_blind_case_complete",
        payload.get("development_only") is True, payload.get("truth_opened") is False,
        payload.get("production_promotion_authorized") is False,
        payload.get("real_forward_outcomes_accessed") is False,
        payload.get("preregistration_digest") == prereg["preregistration_digest"],
        payload.get("registry_case_id") == CASE_IDS[ordinal],
        payload.get("query_episode_id") == query_id,
        payload.get("query_binding") == dict(binding),
        payload.get("resident_identity_digest") == resident["identity_digest"],
        type(payload.get("lease_digests")) is list,
        len(payload.get("lease_digests", [])) == 5,
        all(value == resident["lease"]["lease_digest"]
            for value in payload.get("lease_digests", [])),
        payload.get("proposal_semantic_exact") is True,
        payload.get("certified") is True, payload.get("semantic_passed") is True,
        payload.get("result_digest")
        == stable_hash(_without(payload, {"created_at", "result_digest"})),
    )):
        raise VerificationError("case semantic envelope differs")
    _timestamp(payload["created_at"], "case")
    forward = _validate_proposal(
        payload["forward_proposal"], query_id,
        binding["packed_query_input_digest"], "forward",
    )
    reverse = _validate_proposal(
        payload["reverse_proposal"], query_id,
        binding["packed_query_input_digest"], "reverse",
    )
    omitted = {"block_rows", "block_order", "elapsed_seconds", "peak_rss_mb"}
    if _without(payload["forward_proposal"], omitted) != _without(
        payload["reverse_proposal"], omitted,
    ) or forward != reverse:
        raise VerificationError("forward/reverse proposal semantics differ")
    matches = _validate_matches(payload["matches"])
    _validate_certificate(
        payload["certificate"], matches, query_id,
        binding["certified_input_digest"], forward,
        payload["forward_proposal"]["eligible_rows"],
    )
    candidates = {row.episode_id: row for row in forward}
    if len(candidates) != len(forward) or set(universe) != {
        row["episode_id"] for row in matches
    }:
        raise VerificationError("match authentication set differs")
    closures = payload["certificate"]["threshold_closure_passes"]
    prefix_ids = {
        row.episode_id for row in forward[:min(MAXIMUM_FRONTIER, len(forward))]
    }
    for match in matches:
        identifier = match["episode_id"]
        authenticated = universe[identifier]
        proposal = candidates.get(identifier)
        if not all((
            set(authenticated) == {
                "symbol", "cutoff_ns", "quality_tier", "overflow_fallback",
                "lower_bound", "eligible",
            },
            authenticated["eligible"] is True,
            match["symbol"] == authenticated["symbol"],
            pd.Timestamp(match["cutoff"]).value == authenticated["cutoff_ns"],
            match["quality_tier"] == authenticated["quality_tier"],
        )):
            raise VerificationError("match frozen-universe identity differs")
        if proposal is not None and not all((
            proposal.symbol == authenticated["symbol"],
            proposal.cutoff_ns == authenticated["cutoff_ns"],
            proposal.quality_tier == authenticated["quality_tier"],
            proposal.overflow_fallback == authenticated["overflow_fallback"],
            proposal.lower_bound.hex() == authenticated["lower_bound"].hex(),
        )):
            raise VerificationError("match proposal/universe metadata differs")
        if not closures:
            if proposal is None:
                raise VerificationError("non-closure match is absent from proposal")
        elif identifier not in prefix_ids:
            bound = authenticated["lower_bound"]
            admitted = any(
                bound <= row["upper_inclusive"]
                and (row["lower_exclusive"] is None
                     or bound > row["lower_exclusive"])
                and (not authenticated["overflow_fallback"]
                     or row["lower_exclusive"] is None)
                for row in closures
            )
            if not admitted:
                raise VerificationError(
                    "outside-prefix match lacks closure-band admission"
                )
    if not all((
        payload["rounds"] == payload["certificate"]["rounds"],
        payload["streaming_fallback_used"] is (
            len(payload["certificate"]["threshold_closure_passes"]) > 0
        ),
        payload["metrics"]["forward_proposal_seconds"]
        == payload["forward_proposal"]["elapsed_seconds"],
        payload["metrics"]["reverse_proposal_seconds"]
        == payload["reverse_proposal"]["elapsed_seconds"],
        payload["metrics"]["process_rss_mb"]
        >= max(payload["forward_proposal"]["peak_rss_mb"],
               payload["reverse_proposal"]["peak_rss_mb"]),
        payload["certificate"]["elapsed_seconds"]
        <= payload["metrics"]["exact_task_wall_seconds"],
        payload["performance_passed"] == _performance(payload["metrics"]),
    )):
        raise VerificationError("case cross-binding differs")
    return dict(payload)


def _load_authorities(
    authority_root: Path, verification_path: Path,
    *, verification_sha256: str, verification_digest: str,
    case_bindings: Mapping[str, tuple[str, str]],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    verification, observed_verification_sha = _read_json_sha(verification_path)
    if observed_verification_sha != verification_sha256:
        raise VerificationError("authority verification bytes differ")
    if not all((
        verification.get("schema_version")
        == "m04r11-certified-authority-verification-v4",
        verification.get("result_digest") == verification_digest,
        verification.get("passed") is True,
        verification.get("authority_correctness_passed") is True,
        verification.get("production_promotion_authorized") is False,
        verification.get("real_forward_outcomes_accessed") is False,
        verification.get("authority_matrix_digest") == AUTHORITY_MATRIX_DIGEST,
        verification.get("authority_seal_digest") == AUTHORITY_SEAL_DIGEST,
        verification.get("generation_id") == GENERATION_ID,
    )):
        raise VerificationError("authority verification differs")
    if set(case_bindings) != set(QUERY_IDS):
        raise VerificationError("authority binding set differs")
    result: dict[str, dict[str, Any]] = {}
    hashes = {"authority_verification": verification_sha256}
    for query_id in QUERY_IDS:
        expected_sha, expected_digest = case_bindings[query_id]
        path = authority_root / "cases" / f"{query_id}.json"
        payload, observed_sha = _read_json_sha(path)
        if not all((
            observed_sha == expected_sha,
            payload.get("schema_version") == "m04r11-certified-authority-case-v4",
            payload.get("status") == "completed",
            payload.get("query_episode_id") == query_id,
            payload.get("result_digest") == expected_digest,
            payload.get("real_forward_outcomes_accessed") is False,
            type(payload.get("matches")) is list and len(payload["matches"]) == 20,
            type(payload.get("certificate")) is dict,
        )):
            raise VerificationError("authority case differs")
        result[query_id] = payload
        hashes[f"authority:{query_id}"] = observed_sha
    return result, hashes


def _comparison_row(
    candidate: Mapping[str, Any], authority: Mapping[str, Any],
) -> dict[str, Any]:
    certificate = candidate["certificate"]
    authority_certificate = authority["certificate"]
    row = {
        "query_episode_id": candidate["query_episode_id"],
        "matches_equal": candidate["matches"] == authority["matches"],
        "certificate_result_digest_equal_diagnostic":
            certificate["result_digest"] == authority_certificate["result_digest"],
        "accounting_equal": all(
            certificate[key] == authority_certificate[key]
            for key in ("eligible_candidates", "exact_evaluated", "safely_pruned")
        ),
        "candidate_case_digest": candidate["result_digest"],
        "authority_case_digest": authority["result_digest"],
        "stock_prefix_equal": candidate["query_binding"]["query_stock_prefix"]
        == authority["query_stock_prefix"],
        "benchmark_prefix_equal":
        candidate["query_binding"]["query_benchmark_prefix"]
        == authority["query_benchmark_prefix"],
    }
    row["semantic_passed"] = all((
        row["matches_equal"], row["accounting_equal"],
        row["stock_prefix_equal"], row["benchmark_prefix_equal"],
    ))
    return row


def _fresh_output(path: Path, protected: Sequence[Path]) -> None:
    lexical = path.absolute()
    if ".." in path.parts or path.is_symlink() or path.exists():
        raise VerificationError("verification root must be fresh and unaliased")
    resolved = path.resolve()
    for item in protected:
        other = item.resolve()
        if resolved == other or resolved.is_relative_to(other) or other.is_relative_to(resolved):
            raise VerificationError("verification root overlaps protected evidence")
    parent = lexical.parent
    while parent != parent.parent:
        if parent.exists() and parent.is_symlink():
            raise VerificationError("verification ancestry contains a symlink")
        parent = parent.parent


def verify(
    *, repository: Path, producer_root: Path, authority_root: Path,
    authority_verification: Path, output_root: Path,
    repository_prereg: Mapping[str, Any] | None = None,
    enforce_topology: bool = True,
    git_validator: Callable[[Path, Mapping[str, Any]], None] = _validate_git,
    diagnostic_git_validator: Callable[
        [Path, Mapping[str, Any]], None
    ] = _validate_committed_diagnostic,
    config_validator: Callable[[Path, str], None] = _validate_config_bytes,
    prereg_validator: Callable[..., None] = _validate_preregistration,
    environment_loader: Callable[[], Mapping[str, Any]] = _environment,
    binding_loader: Callable[[Mapping[str, Any]], dict[str, dict[str, Any]]]
    = _query_bindings,
    universe_loader: Callable[
        [Mapping[str, Any], Sequence[Mapping[str, Any]]],
        dict[str, dict[str, dict[str, Any]]],
    ] = _packed_universe,
    closure_reconstructor: Callable[
        [Mapping[str, Any], Sequence[Mapping[str, Any]]], None,
    ] = _reconstruct_closure_scans,
    proposal_reconstructor: Callable[
        [Mapping[str, Any], Sequence[Mapping[str, Any]]], None,
    ] = _reconstruct_proposals,
    source_identity_loader: Callable[[Path], Mapping[str, Any]]
    = _source_generation_identity,
    resident_observer: Callable[[Mapping[str, Any]], dict[str, Any]]
    = _observe_resident,
    verification_sha256: str = AUTHORITY_VERIFICATION_SHA256,
    verification_digest: str = AUTHORITY_VERIFICATION_DIGEST,
    authority_case_bindings: Mapping[str, tuple[str, str]]
    = AUTHORITY_CASE_BINDINGS,
) -> dict[str, Any]:
    """Reconstruct a terminal comparison and publish independent evidence."""
    repository = repository.absolute()
    producer_root = producer_root.absolute()
    authority_root = authority_root.absolute()
    authority_verification = authority_verification.absolute()
    output_root = output_root.absolute()
    for path, label in (
        (repository, "repository"), (producer_root, "producer"),
        (authority_root, "authority"),
        (authority_verification, "authority verification"),
    ):
        _plain_existing_path(path, label)
    repository = repository.resolve()
    _fresh_output(output_root, (
        producer_root, authority_root, authority_verification,
    ))
    terminal_files = {
        "RUN_STARTED.json", "CONTRACT.json", "RESIDENT.json",
        "PRODUCER_SEALED.json", "RESULTS_OPENED.json", "COMPARISON.json",
        "COMPARISON_SEALED.json",
        *(f"cases/{index:02d}-{query_id}.json"
          for index, query_id in enumerate(QUERY_IDS)),
    }
    _exact_tree(producer_root, terminal_files, {"cases"})
    documents: dict[str, dict[str, Any]] = {}
    file_hashes: dict[str, str] = {}
    for relative in sorted(terminal_files):
        documents[relative], file_hashes[relative] = _read_json_sha(
            producer_root / relative
        )
    prereg = documents["CONTRACT.json"]
    frozen = dict(repository_prereg) if repository_prereg is not None else _read_json(
        repository / PREREG_RELATIVE
    )
    if prereg != frozen:
        raise VerificationError("producer preregistration snapshot differs")
    _fresh_output(output_root, (
        producer_root, authority_root, authority_verification,
        repository / PREREG_RELATIVE,
        repository / DIAGNOSTIC_RELATIVE,
        Path(prereg["config_path"]),
        Path(prereg["roots"]["registry_root"]),
        Path(prereg["roots"]["source_full_root"]),
        Path(prereg["roots"]["resident_root"]),
    ))
    if producer_root.resolve() != Path(prereg["roots"]["output_root"]).resolve():
        raise VerificationError("producer root differs from preregistered ownership root")
    environment = dict(environment_loader())
    prereg_validator(
        prereg, repository, enforce_topology=enforce_topology,
        environment=environment,
    )
    git_validator(repository, prereg["git"])
    diagnostic_git_validator(repository, prereg["finite_threshold_diagnostic"])
    config_validator(Path(prereg["config_path"]), prereg["config_sha256"])
    started = documents["RUN_STARTED.json"]
    if not all((
        set(started) == {
            "schema_version", "preregistration_digest", "case_order",
            "parent_max_workers", "created_at",
        },
        started.get("schema_version") == "m04r13-run-started-v1",
        started.get("preregistration_digest") == prereg["preregistration_digest"],
        started.get("case_order") == list(QUERY_IDS),
        started.get("parent_max_workers") == 1,
    )):
        raise VerificationError("run-started marker differs")
    started_at = _timestamp(started["created_at"], "run-started")
    resident = documents["RESIDENT.json"]
    _validate_resident(resident)
    live_resident = resident_observer(prereg)
    _validate_resident(live_resident)
    if not all((
        resident == live_resident,
        resident["identity_digest"] == prereg["resident_identity_digest"],
        resident["content_digest"] == prereg["resident_content_digest"],
        resident["ready_digest"] == prereg["resident_ready_digest"],
        resident["store_root"]
        == str((Path(prereg["roots"]["resident_root"]) / "store").resolve()),
    )):
        raise VerificationError("resident/preregistration binding differs")
    bindings = binding_loader(prereg)
    if set(bindings) != set(QUERY_IDS):
        raise VerificationError("query binding set differs")
    raw_cases = [
        documents[f"cases/{ordinal:02d}-{query_id}.json"]
        for ordinal, query_id in enumerate(QUERY_IDS)
    ]
    source_store = Path(prereg["roots"]["source_full_root"]) / "store"
    source_identity = dict(source_identity_loader(source_store))
    universes = universe_loader(prereg, raw_cases)
    if set(universes) != set(QUERY_IDS):
        raise VerificationError("packed-universe query set differs")
    cases = []
    for ordinal, query_id in enumerate(QUERY_IDS):
        case = raw_cases[ordinal]
        cases.append(_validate_case(
            case, ordinal, prereg, resident, bindings[query_id],
            universes[query_id],
        ))
    proposal_reconstructor(prereg, cases)
    closure_reconstructor(prereg, cases)
    if dict(source_identity_loader(source_store)) != source_identity:
        raise VerificationError("durable packed source identity changed")
    producer_seal = documents["PRODUCER_SEALED.json"]
    producer_expected = {
        "schema_version": PRODUCER_SEAL_SCHEMA,
        "status": "truth_blind_producer_complete", "development_only": True,
        "truth_opened": False, "production_promotion_authorized": False,
        "real_forward_outcomes_accessed": False,
        "preregistration_digest": prereg["preregistration_digest"],
        "resident_identity_digest": resident["identity_digest"],
        "query_ids": list(QUERY_IDS),
        "case_digests": [case["result_digest"] for case in cases],
        "semantic_passed": all(case["semantic_passed"] for case in cases),
        "performance_passed": all(case["performance_passed"] for case in cases),
    }
    if not all((
        _without(producer_seal, {"created_at", "seal_digest"}) == producer_expected,
        producer_seal.get("seal_digest") == stable_hash(producer_expected),
    )):
        raise VerificationError("producer seal reconstruction differs")
    producer_sealed_at = _timestamp(producer_seal["created_at"], "producer seal")
    marker = documents["RESULTS_OPENED.json"]
    marker_expected = {
        "schema_version": RESULTS_OPENED_SCHEMA,
        "status": "authority_results_opened_after_producer_seal",
        "development_only": True, "production_promotion_authorized": False,
        "real_forward_outcomes_accessed": False,
        "preregistration_digest": prereg["preregistration_digest"],
        "producer_seal_digest": producer_seal["seal_digest"],
    }
    if not all((
        _without(marker, {"created_at", "result_digest"}) == marker_expected,
        marker.get("result_digest") == stable_hash(marker_expected),
    )):
        raise VerificationError("results-opened marker differs")
    opened_at = _timestamp(marker["created_at"], "results-opened")
    if not started_at <= producer_sealed_at <= opened_at:
        raise VerificationError("producer/opening timestamp order differs")
    authorities, authority_hashes = _load_authorities(
        authority_root, authority_verification,
        verification_sha256=verification_sha256,
        verification_digest=verification_digest,
        case_bindings=authority_case_bindings,
    )
    expected_rows = [
        _comparison_row(case, authorities[case["query_episode_id"]])
        for case in cases
    ]
    semantic = all(row["semantic_passed"] for row in expected_rows)
    performance = producer_seal["performance_passed"]
    comparison = documents["COMPARISON.json"]
    comparison_expected = {
        "schema_version": COMPARISON_SCHEMA, "status": "comparison_complete",
        "development_only": True, "post_open": True,
        "production_promotion_authorized": False,
        "real_forward_outcomes_accessed": False,
        "results_opened_digest": marker["result_digest"],
        "producer_seal_digest": producer_seal["seal_digest"],
        "authority_verification_digest": verification_digest,
        "rows": expected_rows, "semantic_passed": semantic,
        "performance_passed": performance, "passed": semantic and performance,
    }
    if not all((
        _without(comparison, {"created_at", "result_digest"}) == comparison_expected,
        comparison.get("result_digest") == stable_hash(comparison_expected),
    )):
        raise VerificationError("comparison reconstruction differs")
    comparison_at = _timestamp(comparison["created_at"], "comparison")
    comparison_seal = documents["COMPARISON_SEALED.json"]
    comparison_seal_expected = {
        "schema_version": COMPARISON_SEAL_SCHEMA,
        "status": "terminal_comparison_sealed", "development_only": True,
        "production_promotion_authorized": False,
        "real_forward_outcomes_accessed": False,
        "comparison_result_digest": comparison["result_digest"],
        "semantic_passed": semantic, "performance_passed": performance,
        "passed": semantic and performance,
    }
    if not all((
        _without(comparison_seal, {"created_at", "seal_digest"})
        == comparison_seal_expected,
        comparison_seal.get("seal_digest") == stable_hash(comparison_seal_expected),
    )):
        raise VerificationError("comparison seal reconstruction differs")
    comparison_sealed_at = _timestamp(
        comparison_seal["created_at"], "comparison seal",
    )
    if not opened_at <= comparison_at <= comparison_sealed_at:
        raise VerificationError("comparison timestamp order differs")
    all_hashes = {**file_hashes, **authority_hashes}
    deterministic = {
        "schema_version": VERIFICATION_SCHEMA,
        "status": "terminal_evidence_independently_reconstructed",
        "development_only": True, "post_open": True,
        "production_promotion_authorized": False,
        "real_forward_outcomes_accessed": False,
        "preregistration_digest": prereg["preregistration_digest"],
        "producer_seal_digest": producer_seal["seal_digest"],
        "results_opened_digest": marker["result_digest"],
        "comparison_result_digest": comparison["result_digest"],
        "comparison_seal_digest": comparison_seal["seal_digest"],
        "authority_verification_digest": verification_digest,
        "authority_matrix_digest": AUTHORITY_MATRIX_DIGEST,
        "authority_seal_digest": AUTHORITY_SEAL_DIGEST,
        "verified_query_ids": list(QUERY_IDS), "verified_cases": 4,
        "semantic_passed": semantic, "performance_passed": performance,
        "experiment_passed": semantic and performance,
        "verification_passed": True,
        "artifact_sha256": all_hashes,
        "artifact_sha256_digest": stable_hash(all_hashes),
        "verifier_sha256": _sha(Path(__file__)),
    }
    payload = {
        **deterministic, "created_at": datetime.now(timezone.utc).isoformat(),
        "result_digest": stable_hash(deterministic),
    }
    output_root.mkdir(parents=True)
    _atomic_create(output_root / "verification.json", payload)
    html = (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>M04R-13 independent verification</title></head><body>"
        "<h1>PASS: terminal evidence reconstructed</h1>"
        f"<p>Semantic: {escape(str(semantic))}; performance: "
        f"{escape(str(performance))}; development-only post-open evidence.</p>"
        f"<pre>{escape(json.dumps(payload, indent=2, sort_keys=True))}</pre>"
        "</body></html>"
    )
    html_path = output_root / "verification.html"
    temporary = html_path.with_name(f".{html_path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(html.encode())
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, html_path)
        descriptor = os.open(output_root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()
    _exact_tree(output_root, {"verification.json", "verification.html"}, set())
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--authority-verification", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    payload = verify(
        repository=arguments.repository,
        producer_root=arguments.producer_root,
        authority_root=arguments.authority_root,
        authority_verification=arguments.authority_verification,
        output_root=arguments.output_root,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
