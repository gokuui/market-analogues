"""Independent, read-only verifier for the T14-02 scheduler evidence.

This module intentionally does not import the T14-02 producer.  It freezes
the evidence contract independently, reconstructs the terminal result from
leaf bytes, and publishes only a create-only verification receipt.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

from dataclasses import asdict
from market_analogues.adapters import source_from_spec
from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.certified_packed_search import certified_packed_search_contract
from market_analogues.config import load_config
from market_analogues.episodes import build_episode
from market_analogues.packed_bound_search import (
    BoundProposal, PackedBoundQuery, _packed_query_input_digest,
    bound_proposal_candidate_digest, packed_bound_search_contract,
)
from market_analogues.representation import represent, representation_input_digest
from market_analogues.resident_store import observe_ready_strict, resident_file_identity_lease
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import InstrumentKey, SearchQuery, stable_hash


SCHEMA = "m04r14-exact-scheduler-poc-v1"
RUN_SCHEMA = "m04r14-exact-scheduler-run-started-v1"
PROPOSAL_SCHEMA = "m04r14-exact-scheduler-proposal-v1"
ATTEMPT_SCHEMA = "m04r14-exact-scheduler-attempt-v1"
SEMANTICS_SCHEMA = "m04r14-exact-scheduler-semantics-v1"
MEASUREMENTS_SCHEMA = "m04r14-exact-scheduler-measurements-v1"
COMPLETE_SCHEMA = "m04r14-exact-scheduler-complete-v1"
INCOMPLETE_SCHEMA = "m04r14-exact-scheduler-incomplete-v1"
PREREG_SCHEMA = "m04r14-exact-scheduler-preregistration-v1"
VERIFICATION_SCHEMA = "m04r14-exact-scheduler-independent-verification-v1"
FROZEN_QUERY_IDS = (
    "3307023dbe2164d025e788da", "3618af07dedd52fb3bdb1ccd",
    "9d7365581643bd93e85beb67", "99a0838725a09570b4a075ff",
)
FROZEN_CASE_IDS = (
    "nasdaq-JCTC-historical-252", "nasdaq-GBNY-current-252",
    "nasdaq-GBNY-historical-252", "nasdaq-ISPOW-current-252",
)
EXTRA_QUERY_IDS = (
    "2d5fd014631f9d4ce97ac34d", "5d4e4365b63ff184cbf29f9a",
    "f2af4d63144103d695102676", "f9d5f80116223e7573d268fb",
)
EXTRA_CASE_IDS = (
    "nasdaq-OABI-historical-252", "nasdaq-BTCY-current-252",
    "nasdaq-CUE-historical-252", "nasdaq-FRMM-historical-252",
)
THROUGHPUT_QUERY_IDS = FROZEN_QUERY_IDS + EXTRA_QUERY_IDS
THROUGHPUT_CASE_IDS = FROZEN_CASE_IDS + EXTRA_CASE_IDS
WORKERS = (1, 2, 4, 8)
REPETITIONS = 3
PROPOSAL_THREADS = 8
TASKS = 8
NUMERIC_ATOL = 1e-6
ENGINE_TOLERANCE = 1e-12
PREREG_SELECTION_RULE = (
    "per-case max/min<=1.25; score=sum four 3-rep medians; "
    "smallest workers with score<=1.03*fastest"
)
RUN_THROUGHPUT = {"tasks": 8, "distinct_queries": True,
    "maximum_concurrent_exact_tasks": 8, "workers_per_task": 1,
    "proposal_phase": "serial-before-barrier"}
RUN_NUMERIC_POLICY = {"semantic_digest_equality": "exact",
    "scorer_absolute_tolerance_hex": NUMERIC_ATOL.hex(),
    "engine_algorithm_tolerance_hex": ENGINE_TOLERANCE.hex()}
CATALOG_DIGEST = "5aedc6d5eb0890df3040c559cf70cf35123398506577f63e852030db132bd658"
ORACLE_RESULT_DIGEST = "ac2a8c84cf629060156ebc37f0c037a0a6731b1fffca9d349eeebdfd0eca0496"
M11_SELECTION_DIGEST = "6542e99973abbaf81db2ff1e1e5bf107b45802027519642c902169e617c329be"
M11_ALL60_DIGEST = "0eb6aa3d9f3277ca4d1fab829b08981b982626c856683a5616eac92fc57c49a8"
M11_SEMANTIC_SEAL = "976781b3021700c54e64091b669507201f09aa4799b13e1f414af2e6979066a2"
M11_SEMANTIC_MATRIX = "9bc68539092ec0912f5bdcd0c0f063c35370e15757408914e44e0c1178f21af3"
M11_SEMANTIC_SHA = "b9c63061a14c1c04f5ffb8bb7bd8d12b19cd3ec7aa945099cc4f87593ea9c492"
M11_COMPLETE_SHA = "de098543ae3b76e019742dc0d574468fb0bfc75d0afd4d5576b1bfc870c6ebd6"
CATALOG_SHA256 = "e7328fa2e270e7333e3b12411d2bc142c7055cbaa76235dbd25bf90cf6f2fda1"
ORACLE_SHA256 = "d9c714f6f56ae62f6999881d97d10cf49a37d3c5fd0a2b56f75fccb3a6583566"
GENERATION_ID = "9fc6ae0ec4451133d8006897162f3443803fd30a3c44b78e80a476fbb18bb483"
COMPONENT_NAMES = {"stage", "price", "candle_volatility", "volume_shock",
                   "market_context", "structural", "coarse"}
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    "NUMBA_NUM_THREADS", "NUMBA_THREADING_LAYER",
)
RUNTIME_FIXED_FILES = (
    "config/datasets.example.yaml",
    "experiments/m04r/m04r14_exact_scheduler_poc.py",
    "experiments/m04r/verify_m04r14_exact_scheduler_poc.py",
    "experiments/m04r/m04r13_threaded_certified_exposed.py",
    "experiments/m04r/m04r14_evidence_catalog.py",
    "experiments/m04r/m04r14_adversarial_oracle.py",
    "experiments/m04r/compare_m04r11_candidate_matrix_v2.py",
    "experiments/m04r/m04r11_candidate_v2_contract.py",
    "experiments/m04r/m04r11_candidate_matrix_v2.py",
    "experiments/m04r/m04r12_quota_ladder_poc.py",
)
CANDIDATE_RELATIVE = Path("config/data/analogues/m04r14/exact-scheduler-poc-v1")
PREREG_RELATIVE = Path("experiments/m04r/m04r14_exact_scheduler_poc_preregistered.json")
VERIFICATION_RELATIVE = Path(
    "config/data/analogues/m04r14/exact-scheduler-poc-v1-verification"
)


class VerificationError(RuntimeError):
    pass


def _without(value: Mapping[str, Any], omitted: set[str]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in omitted}


def _strict_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise VerificationError("evidence is not strict finite JSON") from exc


def _identity(item: os.stat_result) -> tuple[int, ...]:
    return (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns,
            item.st_ctime_ns, item.st_mode)


def _read_snapshot(path: Path, expected_sha: str | None = None) -> tuple[dict[str, Any], str, tuple[int, ...]]:
    if path.is_symlink():
        raise VerificationError("JSON symlink is forbidden")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        raw = bytearray()
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            raw.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _identity(before) != _identity(after) or not stat.S_ISREG(before.st_mode):
        raise VerificationError("JSON identity changed")
    digest = sha256(raw).hexdigest()
    if expected_sha is not None and digest != expected_sha:
        raise VerificationError("JSON SHA differs")
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise VerificationError("duplicate JSON key")
            result[key] = value
        return result
    try:
        value = json.loads(
            bytes(raw), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                VerificationError(f"nonfinite token {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError("invalid JSON") from exc
    def finite(item: Any) -> bool:
        if type(item) is float:
            return math.isfinite(item)
        if type(item) is list:
            return all(finite(child) for child in item)
        if type(item) is dict:
            return all(finite(child) for child in item.values())
        return True
    if type(value) is not dict or not finite(value):
        raise VerificationError("JSON shape/nonfinite number differs")
    return value, digest, _identity(after)


def _read(path: Path, expected_sha: str | None = None) -> dict[str, Any]:
    return _read_snapshot(path, expected_sha)[0]


def _observed_identity(path: Path) -> tuple[int, ...]:
    if path.is_symlink():
        raise VerificationError("file symlink is forbidden")
    observed = path.stat()
    if not stat.S_ISREG(observed.st_mode):
        raise VerificationError("file is not regular")
    return _identity(observed)


def _sha(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor); digest = sha256()
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block: break
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
            before.st_ctime_ns, before.st_mode) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
            after.st_ctime_ns, after.st_mode) or not stat.S_ISREG(before.st_mode):
        raise VerificationError("file identity changed")
    return digest.hexdigest()


def _keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise VerificationError(f"{label} exact keys differ")
    return value


def _seal(value: Any, label: str) -> dict[str, Any]:
    row = _keys(value, {"state", "digest"}, label)
    if type(row["state"]) is not dict or row["digest"] != stable_hash(row["state"]):
        raise VerificationError(f"{label} seal differs")
    return row


def _timestamp(value: Any) -> bool:
    if type(value) is not str:
        return False
    try: parsed = datetime.fromisoformat(value)
    except ValueError: return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed)


def _rotated(values: Sequence[Any], offset: int) -> tuple[Any, ...]:
    offset %= len(values)
    return tuple(values[offset:]) + tuple(values[:offset])


def _paths(include_complete: bool) -> set[str]:
    result = {"CONTRACT.json", "RUN_STARTED.json", "SEMANTICS.json", "MEASUREMENTS.json"}
    if include_complete: result.add("COMPLETE.json")
    for repetition in range(REPETITIONS):
        for ordinal in range(4):
            base = f"primary/r{repetition}/c{ordinal}"
            result.add(f"{base}/PROPOSAL.json")
            result.update(f"{base}/EXACT-w{worker}.json" for worker in WORKERS)
    result.add("throughput/PROPOSALS_COMPLETE.json")
    for index in range(TASKS):
        base = f"throughput/t{index}"
        result.update({f"{base}/PROPOSAL.json", f"{base}/EXACT-w1.json",
                       f"{base}/CONTROL-w1.json"})
    return result


def _tree(root: Path) -> set[str]:
    result = set()
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise VerificationError("terminal tree contains symlink/special entry")
        if stat.S_ISREG(mode): result.add(str(path.relative_to(root)))
    return result


def _reject_symlink_ancestry(path: Path) -> None:
    absolute = path.absolute()
    for parent in (absolute, *absolute.parents):
        if parent.exists() or parent.is_symlink():
            if parent.is_symlink():
                raise VerificationError("path ancestry contains symlink")


def _absent_canonical(path: Path) -> Path:
    _reject_symlink_ancestry(path)
    if path.exists() or path.is_symlink():
        raise VerificationError("verification root must be absent")
    nearest = path.absolute().parent
    while not nearest.exists():
        nearest = nearest.parent
    return nearest.resolve(strict=True) / path.absolute().relative_to(nearest.absolute())


def _overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _execution_policy() -> dict[str, Any]:
    return {
        "case_ids": list(FROZEN_CASE_IDS), "query_ids": list(FROZEN_QUERY_IDS),
        "throughput_case_ids": list(THROUGHPUT_CASE_IDS),
        "throughput_query_ids": list(THROUGHPUT_QUERY_IDS),
        "workers": list(WORKERS), "repetitions": REPETITIONS,
        "primary_order": [list(_rotated(range(4), row)) for row in range(3)],
        "worker_order": [[list(_rotated(WORKERS, row + case)) for case in range(4)]
                         for row in range(3)],
        "proposal": {"fresh_spawned_child": True, "threads": 8, "cpu_count": 8,
            "quota": 16385, "forward": {"block_rows": 4096, "block_order": "forward"},
            "reverse": {"block_rows": 4097, "block_order": "reverse"},
            "branch_aware": True, "verify_content": False, "serial": True,
            "scans_per_preparation": 2},
        "exact": {"fresh_spawned_child_per_attempt": True,
            "initial_frontier_rows": 1000, "maximum_frontier_rows": 16384,
            "seed_rows": 512, "block_rows": 4096, "sparse_cutoff": 8,
            "requested_positions": True, "vector_lower_bounds": True,
            "deferred_alignments": True, "compact_scored": True,
            "native_bound_deferral": True, "streaming_threshold_closure": True,
            "branch_aware_packed_bounds": True, "precomputed_forward_proposal": True},
        "throughput": {"label": "exact-stage exposed-host p8t1 microbenchmark",
            "distinct_tasks": 8, "workers_per_task": 1,
            "proposal_preparation": "serial-before-durable-barrier",
            "timed_wave": "eight-ready-one-release-spawned-processes",
            "serial_controls": "after-wave-excluded-from-timing",
            "affinity": "one-distinct-logical-cpu-per-child",
            "supervisor_shares_host_capacity_and_is_not_claimed_nonoversubscribed": True,
            "excluded_from_worker_selection": True},
        "selection": {"case_worker_repeat_max_min_ratio": 1.25,
            "score": "sum-of-four-three-repeat-raw-medians", "near_fastest_factor": 1.03,
            "tie": "smallest-worker-within-factor-no-rounding"},
        "numeric": {"semantic_identity": "byte/canonical-digest-exact",
            "evidence_atol_hex": NUMERIC_ATOL.hex(), "evidence_rtol": 0,
            "engine_algorithm_tolerance_hex": ENGINE_TOLERANCE.hex()},
        "resource": {"child_process_swap_kib": 0, "host_vmstat_swap": "context-only",
            "peak_rss": "max-proc-rss-proc-hwm-wait4-ru_maxrss",
            "startup_timeout_seconds": 600, "task_timeout_seconds": 3600,
            "run_hard_limit_seconds": 21600},
        "thread_environment": {"numeric_thread_variables": "1",
            "numba_threading_layer": "preserve-preregistered-observed-value"},
        "lifecycle": {"create_only": True, "resume": False,
            "failure_terminal": INCOMPLETE_SCHEMA, "complete_published_last": True,
            "exact_terminal_tree": sorted(_paths(True))},
        "claims": {"development_only": True, "cases_previously_exposed": True,
            "direct_raw_authority_accessed_by_this_run": False,
            "authority_derived_prerequisite_evidence_accessed_by_this_run": True,
            "forward_outcomes_accessed_by_this_run": False,
            "production_promotion_authorized": False},
        "schemas": {"run": RUN_SCHEMA, "proposal": PROPOSAL_SCHEMA,
            "attempt": ATTEMPT_SCHEMA, "semantics": SEMANTICS_SCHEMA,
            "measurements": MEASUREMENTS_SCHEMA, "complete": COMPLETE_SCHEMA,
            "incomplete": INCOMPLETE_SCHEMA},
    }


def _parse_cpu_list(raw: str) -> list[int]:
    cpus: set[int] = set()
    for part in raw.strip().split(","):
        if not part:
            continue
        bounds = part.split("-", 1)
        try:
            start, end = int(bounds[0]), int(bounds[-1])
        except ValueError as exc:
            raise VerificationError("cgroup CPU list is malformed") from exc
        if start < 0 or end < start:
            raise VerificationError("cgroup CPU list is malformed")
        cpus.update(range(start, end + 1))
    return sorted(cpus)


def _cgroup_cpu_configuration() -> dict[str, Any]:
    try:
        unified = next(line.split("::", 1)[1] for line in
            Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
    except (OSError, StopIteration, IndexError) as exc:
        raise VerificationError("cgroup v2 CPU binding is unavailable") from exc
    cgroup_root = Path("/sys/fs/cgroup")
    current = (cgroup_root / unified.lstrip("/")).resolve(strict=True)
    def inherited(name: str) -> tuple[Path, str]:
        candidate = current
        while candidate == cgroup_root or cgroup_root in candidate.parents:
            path = candidate / name
            try: raw = path.read_text().strip()
            except OSError: raw = ""
            if raw: return path, raw
            if candidate == cgroup_root: break
            candidate = candidate.parent
        raise VerificationError(f"cgroup {name} binding is unavailable")
    cpuset_path, cpuset_raw = inherited("cpuset.cpus.effective")
    cpu_max_path, cpu_max_raw = inherited("cpu.max")
    maximum = cpu_max_raw.split()
    if len(maximum) != 2:
        raise VerificationError("cgroup cpu.max is malformed")
    quota = None if maximum[0] == "max" else int(maximum[0]); period = int(maximum[1])
    stat_path, _ = inherited("cpu.stat")
    if (quota is not None and quota <= 0) or period <= 0:
        raise VerificationError("cgroup cpu.max is invalid")
    return {"schema_version": "m04r14-cgroup-cpu-configuration-v1",
        "cgroup_path": str(current), "cpuset_path": str(cpuset_path),
        "effective_cpus": _parse_cpu_list(cpuset_raw), "cpu_max_path": str(cpu_max_path),
        "quota_usec": quota, "period_usec": period,
        "effective_quota_cpus": None if quota is None else float(quota) / float(period),
        "cpu_stat_path": str(stat_path)}


def _environment() -> dict[str, Any]:
    state = {"python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(), "system": platform.system(),
        "machine": platform.machine(), "packages": {
            name: importlib.metadata.version(name) for name in ("numpy", "pandas", "numba", "pyarrow")},
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "thread_environment": {key: os.environ.get(key) for key in THREAD_ENV_KEYS},
        "cgroup_cpu_configuration": _cgroup_cpu_configuration()}
    return {"state": state, "digest": stable_hash(state)}


def _runtime_paths(repository: Path, commit: str) -> tuple[str, ...]:
    names = subprocess.run(["git", "ls-tree", "-r", "--name-only", commit],
        cwd=repository, text=True, capture_output=True, check=True).stdout.splitlines()
    selected = {name for name in names if name.startswith("src/market_analogues/") and name.endswith(".py")}
    if any(name not in names for name in RUNTIME_FIXED_FILES):
        raise VerificationError("runtime mandatory file is absent")
    selected.update(RUNTIME_FIXED_FILES)
    return tuple(sorted(selected))


def _validate_runtime(value: Any, repository: Path, require_production: bool) -> None:
    if not require_production:
        _keys(value, {"mode", "digest"}, "test runtime")
        return
    row = _seal(value, "runtime"); state = _keys(row["state"],
        {"git_head", "files", "environment", "contracts"}, "runtime state")
    if state["contracts"] != {"schema_version": SCHEMA, "execution_policy": _execution_policy()} \
            or set(state["files"]) != set(_runtime_paths(repository, state["git_head"])) \
            or state["environment"] != _environment():
        raise VerificationError("runtime contract differs")
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repository, text=True, capture_output=True, check=True).stdout.strip()
    current = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository,
        text=True, capture_output=True, check=True).stdout.strip()
    if status or subprocess.run(["git", "merge-base", "--is-ancestor", state["git_head"], current],
            cwd=repository, check=False).returncode:
        raise VerificationError("runtime Git state differs")
    for name, expected in state["files"].items():
        h0 = subprocess.run(["git", "show", f"{state['git_head']}:{name}"], cwd=repository,
            capture_output=True, check=True).stdout
        now = subprocess.run(["git", "show", f"{current}:{name}"], cwd=repository,
            capture_output=True, check=True).stdout
        if sha256(h0).hexdigest() != expected or sha256(now).hexdigest() != expected \
                or _sha(repository / name) != expected:
            raise VerificationError(f"runtime blob drifted: {name}")


def _validate_prereg_lineage(
    contract: Mapping[str, Any], runtime: Mapping[str, Any], repository: Path,
) -> None:
    prereg = (repository / PREREG_RELATIVE).resolve(strict=True)
    if prereg.is_symlink() or _read(prereg) != contract:
        raise VerificationError("canonical preregistration bytes differ")
    h0 = runtime["state"]["git_head"]
    additions = subprocess.run(["git", "log", "--diff-filter=A", "--format=%H", "--",
        str(PREREG_RELATIVE)], cwd=repository, text=True, capture_output=True,
        check=True).stdout.splitlines()
    if len(additions) != 1:
        raise VerificationError("preregistration introduction commit differs")
    h1 = additions[0]
    parents = subprocess.run(["git", "rev-list", "--parents", "-n", "1", h1],
        cwd=repository, text=True, capture_output=True, check=True).stdout.split()
    changed = subprocess.run(["git", "diff-tree", "--no-commit-id", "--name-only", "-r", h1],
        cwd=repository, text=True, capture_output=True, check=True).stdout.splitlines()
    current = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository,
        text=True, capture_output=True, check=True).stdout.strip()
    blob = subprocess.run(["git", "show", f"{h1}:{PREREG_RELATIVE}"], cwd=repository,
        capture_output=True, check=True).stdout
    if parents != [h1, h0] or changed != [str(PREREG_RELATIVE)] \
            or sha256(blob).hexdigest() != _sha(prereg) \
            or subprocess.run(["git", "merge-base", "--is-ancestor", h1, current],
                cwd=repository, check=False).returncode:
        raise VerificationError("preregistration H0/H1 lineage differs")


def _number(value: Any) -> bool:
    return type(value) in {int, float} and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _resource(value: Any) -> None:
    row = _keys(value, {"before", "after", "delta", "peak_rss_mb", "swap_delta_kib"}, "resource")
    snapshot_keys = {"user_cpu_seconds", "system_cpu_seconds", "minor_faults", "major_faults",
        "voluntary_context_switches", "involuntary_context_switches", "max_rss_mb", "swap_kib",
        "cpu_affinity", "thread_environment"}
    for name in ("before", "after"):
        snap = _keys(row[name], snapshot_keys, f"resource {name}")
        if not all(_number(snap[key]) for key in snapshot_keys - {"cpu_affinity", "thread_environment"}) \
                or type(snap["cpu_affinity"]) is not list or type(snap["thread_environment"]) is not dict:
            raise VerificationError("resource snapshot differs")
    numeric = ("user_cpu_seconds", "system_cpu_seconds", "minor_faults", "major_faults",
        "voluntary_context_switches", "involuntary_context_switches", "swap_kib")
    expected_delta = {key: row["after"][key] - row["before"][key] for key in numeric}
    if row["delta"] != expected_delta or row["swap_delta_kib"] != expected_delta["swap_kib"] \
            or row["peak_rss_mb"] != max(row["before"]["max_rss_mb"], row["after"]["max_rss_mb"]):
        raise VerificationError("resource reconstruction differs")


def _host_cpu_qualification(before: Any, after: Any, required_cpus: int = 8) -> dict[str, Any]:
    before = _keys(before, {"configuration", "cpu_stat"}, "CPU before")
    after = _keys(after, {"configuration", "cpu_stat"}, "CPU after")
    configuration = _keys(before["configuration"], {"schema_version", "cgroup_path",
        "cpuset_path", "effective_cpus", "cpu_max_path", "quota_usec", "period_usec",
        "effective_quota_cpus", "cpu_stat_path"}, "CPU configuration")
    if after["configuration"] != configuration \
            or configuration["schema_version"] != "m04r14-cgroup-cpu-configuration-v1":
        raise VerificationError("CPU configuration changed")
    cpus = configuration["effective_cpus"]; quota = configuration["effective_quota_cpus"]
    if type(cpus) is not list or len(cpus) < required_cpus \
            or len(set(cpus)) != len(cpus) \
            or not all(type(cpu) is int and cpu >= 0 for cpu in cpus) \
            or type(configuration["period_usec"]) is not int \
            or configuration["period_usec"] <= 0 \
            or (configuration["quota_usec"] is not None and (
                type(configuration["quota_usec"]) is not int or configuration["quota_usec"] <= 0)) \
            or (quota is not None and (type(quota) is not float or not math.isfinite(quota)
                or quota < required_cpus)):
        raise VerificationError("CPU capacity is below required lane width")
    if type(before["cpu_stat"]) is not dict or type(after["cpu_stat"]) is not dict \
            or set(before["cpu_stat"]) != set(after["cpu_stat"]) \
            or not {"usage_usec", "nr_periods", "nr_throttled", "throttled_usec"} <= set(before["cpu_stat"]):
        raise VerificationError("CPU stat schema differs")
    delta = {}
    for key in before["cpu_stat"]:
        left, right = before["cpu_stat"][key], after["cpu_stat"][key]
        if type(left) is not int or type(right) is not int or left < 0 or right < left:
            raise VerificationError("CPU stat values differ")
        delta[key] = right - left
    if delta["nr_throttled"] != 0 or delta["throttled_usec"] != 0:
        raise VerificationError("CPU throttling invalidates evidence")
    return {"schema_version": "m04r14-host-cpu-qualification-v1",
        "required_cpus": required_cpus, "before": before, "after": after,
        "delta": delta, "performance_valid": True}


def _child_metric(value: Any) -> None:
    row = _keys(value, {"pid", "cpus", "wall_seconds", "user_cpu_seconds",
        "system_cpu_seconds", "minor_faults", "major_faults", "peak_rss_kib",
        "peak_hwm_kib", "wait4_max_rss_kib", "effective_peak_rss_kib",
        "peak_swap_kib", "final_swap_kib"}, "child metric")
    if type(row["pid"]) is not int or row["pid"] <= 0 or type(row["cpus"]) is not list \
            or not row["cpus"] or len(set(row["cpus"])) != len(row["cpus"]) \
            or not all(type(cpu) is int and cpu >= 0 for cpu in row["cpus"]) \
            or not all(_number(row[key]) for key in set(row) - {"pid", "cpus"}) \
            or row["effective_peak_rss_kib"] != max(row["peak_rss_kib"], row["peak_hwm_kib"],
                row["wait4_max_rss_kib"]) or row["peak_swap_kib"] != 0 or row["final_swap_kib"] != 0:
        raise VerificationError("child metric reconstruction differs")


def _attempt_measurement(value: Any, *, production: bool, workers: int) -> None:
    expected = {"wall_seconds", "engine_seconds", "resources"}
    if production:
        expected.add("spawned_process")
    row = _keys(value, expected, "exact measurement")
    wall, engine = row["wall_seconds"], row["engine_seconds"]
    if not _number(wall) or not _number(engine) or wall < engine:
        raise VerificationError("attempt timing differs")
    _resource(row["resources"])
    if production:
        _child_metric(row["spawned_process"])
        child = row["spawned_process"]
        if len(child["cpus"]) != workers or child["wall_seconds"] < wall:
            raise VerificationError("attempt child timing/worker crosslink differs")


def _validate_live_resident(foundation: Mapping[str, Any], resident: Mapping[str, Any]) -> None:
    ready_path = Path(foundation["resident_root"]) / "READY.json"
    observation = observe_ready_strict(ready_path)
    live_lease = resident_file_identity_lease(ready_path)
    if observation["ready_digest"] != resident["ready_digest"] \
            or observation["ready_file_sha256"] != resident["ready_file_sha256"] \
            or observation["content_digest"] != resident["content_digest"] \
            or observation["seal_digest"] != resident["seal_digest"] \
            or live_lease != resident["lease"] \
            or Path(resident["store_root"]).resolve(strict=True) != (
                Path(foundation["resident_root"]) / "store").resolve(strict=True):
        raise VerificationError("live resident identity/content differs")


def _load_validator(repository: Path, relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, repository / relative)
    if spec is None or spec.loader is None:
        raise VerificationError("prerequisite validator cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _validate_prerequisites(repository: Path, prereq: Mapping[str, Any]) -> None:
    if prereq["evidence_catalog_sha256"] != CATALOG_SHA256 \
            or prereq["adversarial_oracle_sha256"] != ORACLE_SHA256:
        raise VerificationError("production prerequisite fixed SHA differs")
    catalog_path = repository / "config/data/analogues/m04r14/evidence-catalog-v1/catalog.json"
    oracle_path = repository / "config/data/analogues/m04r14/adversarial-oracle-v1/oracle.json"
    catalog = _read(catalog_path, CATALOG_SHA256); oracle = _read(oracle_path, ORACLE_SHA256)
    if catalog.get("catalog_digest") != CATALOG_DIGEST \
            or oracle.get("result_digest") != ORACLE_RESULT_DIGEST or oracle.get("passed") is not True:
        raise VerificationError("production prerequisite evidence differs")
    try:
        _load_validator(repository, "experiments/m04r/m04r14_evidence_catalog.py",
            "m04r14_t1402_independent_catalog_validator").validate_catalog(catalog, repository)
        _load_validator(repository, "experiments/m04r/m04r14_adversarial_oracle.py",
            "m04r14_t1402_independent_oracle_validator").validate_payload(
                oracle, require_production=True)
    except BaseException as exc:
        raise VerificationError("production prerequisite reconstruction differs") from exc


def _validate_foundation(value: Any, *, production: bool, repository: Path | None = None) -> None:
    if not production:
        row = _keys(value, {"mode", "resident_identity_digest", "source_lease_digest",
            "prerequisites"}, "test foundation")
        prereq = _keys(row["prerequisites"], {"evidence_catalog_digest",
            "adversarial_oracle_result_digest"}, "test prerequisites")
        if row["mode"] != "test" or prereq != {
                "evidence_catalog_digest": CATALOG_DIGEST,
                "adversarial_oracle_result_digest": ORACLE_RESULT_DIGEST}:
            raise VerificationError("test foundation differs")
        return
    foundation = _keys(value, {"registry_digest", "generation_id", "provenance_digest",
        "resident", "source_store_root", "config_path", "registry_root", "resident_root",
        "causal_input_shas", "prerequisites", "throughput_selection"}, "production foundation")
    prereq = _keys(foundation["prerequisites"], {"evidence_catalog_digest",
        "evidence_catalog_sha256", "adversarial_oracle_result_digest",
        "adversarial_oracle_sha256"}, "production prerequisites")
    if foundation["generation_id"] != GENERATION_ID \
            or prereq["evidence_catalog_digest"] != CATALOG_DIGEST \
            or prereq["adversarial_oracle_result_digest"] != ORACLE_RESULT_DIGEST:
        raise VerificationError("production foundation constants differ")
    resident = _keys(foundation["resident"], {"ready_digest", "content_digest",
        "seal_digest", "ready_file_sha256", "lease", "store_root", "identity_digest"},
        "resident snapshot")
    lease = _keys(resident["lease"], {"schema_version", "ready_digest",
        "ready_file_sha256", "content_digest", "files", "lease_digest"},
        "resident lease")
    if lease["ready_digest"] != resident["ready_digest"] \
            or lease["ready_file_sha256"] != resident["ready_file_sha256"] \
            or lease["content_digest"] != resident["content_digest"] \
            or lease["lease_digest"] != stable_hash(_without(lease, {"lease_digest"})) \
            or resident["identity_digest"] != stable_hash(_without(resident, {"identity_digest"})):
        raise VerificationError("resident snapshot digest differs")
    identity_keys = {"path", "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode"}
    if type(lease["files"]) is not dict or not lease["files"] or any(
            type(row) is not dict or set(row) != identity_keys for row in lease["files"].values()):
        raise VerificationError("resident lease file identities differ")
    if repository is None:
        raise VerificationError("production foundation repository is absent")
    expected = {"config_path": repository / "config/datasets.example.yaml",
        "registry_root": repository / "config/data/analogues/m04r10/nasdaq-untouched-authority-registry",
        "source_store_root": repository / "config/data/analogues/poc/m04r/packed-bound-full/store",
        "resident_root": Path("/dev/shm/market-analogues/m04r11-candidate-v2") / GENERATION_ID}
    if any(Path(foundation[key]).resolve(strict=True) != path.resolve(strict=True)
           for key, path in expected.items()):
        raise VerificationError("production foundation roots differ")
    _validate_live_resident(foundation, resident)
    _validate_prerequisites(repository, prereq)
    selection = _seal(foundation["throughput_selection"], "throughput selection")
    _keys(selection["state"], {"rule", "source_selection_digest", "source_all60_digest",
        "semantic_seal_digest", "semantic_matrix_digest", "full_non_hard_transcript",
        "selected", "historical_authentication", "candidate_root_inventory"},
        "throughput selection state")
    selection_state = selection["state"]
    bundle_root = repository / "config/data/analogues/m04r11/candidate-pools-v2/case-bundles"
    rows = []
    for path in sorted(bundle_root.glob("*.json")):
        bundle = _read(path); semantic = bundle.get("semantic", {})
        scans = semantic.get("scan_semantics")
        if type(scans) is not list or len(scans) != 3 or scans[1:] != scans[:1] * 2:
            raise VerificationError("M11 proposal repetitions differ")
        rows.append({"query_id": bundle.get("query_episode_id"),
            "symbol": semantic.get("query_symbol"), "candidate_count": scans[0].get("candidate_count"),
            "eligible_rows": scans[0].get("eligible_rows"), "rows_scanned": scans[0].get("rows_scanned"),
            "bundle_digest": bundle.get("bundle_digest")})
    all60 = sorted(rows, key=lambda row: row["query_id"])
    non_hard = sorted((row for row in rows if row["query_id"] not in FROZEN_QUERY_IDS),
                      key=lambda row: (row["candidate_count"], row["query_id"]))
    pool = {"schema_version": "m04r14-t14-02-opened-workload-pool-v1",
        "source": "m04r11-candidate-pools-v2/case-bundles",
        "excluded_hard_query_ids": sorted(FROZEN_QUERY_IDS),
        "sort": ["candidate_count:ascending", "query_id:ascending"], "rows": non_hard}
    selected = [non_hard[index] for index in (6, 20, 34, 48)] if len(non_hard) == 56 else []
    m11_root = bundle_root.parent
    if len(rows) != 60 or stable_hash(all60) != M11_ALL60_DIGEST \
            or stable_hash(pool) != M11_SELECTION_DIGEST \
            or selection_state["full_non_hard_transcript"] != non_hard \
            or selection_state["selected"] != selected \
            or selection_state["source_selection_digest"] != M11_SELECTION_DIGEST \
            or selection_state["source_all60_digest"] != M11_ALL60_DIGEST \
            or selection_state["semantic_seal_digest"] != M11_SEMANTIC_SEAL \
            or selection_state["semantic_matrix_digest"] != M11_SEMANTIC_MATRIX \
            or _sha(m11_root / "SEMANTIC_SEALED.json") != M11_SEMANTIC_SHA \
            or _sha(m11_root / "RUN_COMPLETE.json") != M11_COMPLETE_SHA \
            or [row.get("query_id") for row in selected] != list(EXTRA_QUERY_IDS):
        raise VerificationError("throughput selection identities differ")


def _proposal_process(value: Any, task_id: str) -> None:
    row = _keys(value, {"pid", "wall_seconds", "cpus", "ready_evidence",
        "release_evidence", "startup_to_ready_seconds", "user_cpu_seconds",
        "system_cpu_seconds", "minor_faults", "major_faults", "peak_rss_kib",
        "peak_hwm_kib", "wait4_max_rss_kib", "effective_peak_rss_kib",
        "peak_swap_kib", "final_swap_kib", "host_vmstat_swap_before",
        "host_vmstat_swap_after", "host_swap_is_context_only"}, "proposal process")
    ready = _keys(row["ready_evidence"], {"schema_version", "task_id", "pid",
        "cpu_affinity", "thread_environment", "resident_lease_digest",
        "ready_monotonic"}, "proposal READY")
    release = _keys(row["release_evidence"], {"schema_version", "task_id", "pid",
        "ready_digest", "released_monotonic"}, "proposal release")
    numeric = {"wall_seconds", "startup_to_ready_seconds", "user_cpu_seconds",
        "system_cpu_seconds", "minor_faults", "major_faults", "peak_rss_kib",
        "peak_hwm_kib", "wait4_max_rss_kib", "effective_peak_rss_kib",
        "peak_swap_kib", "final_swap_kib"}
    if not all(_number(row[key]) for key in numeric) or type(row["pid"]) is not int \
            or row["pid"] <= 0 or row["cpus"] != ready["cpu_affinity"] \
            or len(row["cpus"]) != 8 or ready["pid"] != row["pid"] \
            or ready["task_id"] != task_id or release["task_id"] != task_id \
            or release["pid"] != row["pid"] or release["ready_digest"] != stable_hash(ready) \
            or not _number(ready["ready_monotonic"]) or not _number(release["released_monotonic"]) \
            or release["released_monotonic"] < ready["ready_monotonic"] \
            or row["startup_to_ready_seconds"] > row["wall_seconds"] \
            or row["effective_peak_rss_kib"] != max(row["peak_rss_kib"], row["peak_hwm_kib"],
                row["wait4_max_rss_kib"]) or row["peak_swap_kib"] != 0 \
            or row["final_swap_kib"] != 0 or row["host_swap_is_context_only"] is not True:
        raise VerificationError("proposal process reconstruction differs")
    for name in ("host_vmstat_swap_before", "host_vmstat_swap_after"):
        vmstat = _keys(row[name], {"pswpin", "pswpout"}, name)
        if any(type(value) is not int or value < 0 for value in vmstat.values()):
            raise VerificationError("proposal vmstat differs")
    if any(row["host_vmstat_swap_after"][key] < row["host_vmstat_swap_before"][key]
           for key in ("pswpin", "pswpout")):
        raise VerificationError("proposal vmstat regressed")


def _child_environment(runtime: Mapping[str, Any]) -> dict[str, Any]:
    observed = runtime["state"]["environment"]["state"]["thread_environment"]
    return {key: observed.get(key) if key == "NUMBA_THREADING_LAYER" else "1"
            for key in THREAD_ENV_KEYS}


def _proposal_process_binding(
    process: Mapping[str, Any], proposal: Mapping[str, Any], runtime: Mapping[str, Any],
) -> None:
    ready = process["ready_evidence"]
    if ready["task_id"] != proposal["task_id"] \
            or ready["resident_lease_digest"] != proposal["resident_lease_digests"][0] \
            or ready["thread_environment"] != _child_environment(runtime):
        raise VerificationError("proposal process causal binding differs")


def _batch_process(
    value: Any, proposal_shas: Sequence[str], attempts: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
) -> None:
    row = _keys(value, {"wall_seconds", "children", "host_vmstat_swap_before",
        "host_vmstat_swap_after", "host_swap_is_context_only", "process_swap_gate_passed",
        "ready_evidence", "release_evidence", "observed_concurrent_children",
        "release_to_all_children_exit_seconds"}, "batch process")
    if type(row["children"]) is not list or len(row["children"]) != 8 \
            or type(row["ready_evidence"]) is not list or len(row["ready_evidence"]) != 8 \
            or row["host_swap_is_context_only"] is not True \
            or row["process_swap_gate_passed"] is not True \
            or row["observed_concurrent_children"] != 8 \
            or not _number(row["wall_seconds"]) \
            or not _number(row["release_to_all_children_exit_seconds"]):
        raise VerificationError("batch process values differ")
    for child in row["children"]: _child_metric(child)
    for name in ("host_vmstat_swap_before", "host_vmstat_swap_after"):
        vmstat = _keys(row[name], {"pswpin", "pswpout"}, name)
        if any(type(value) is not int or value < 0 for value in vmstat.values()):
            raise VerificationError("batch vmstat differs")
    if any(row["host_vmstat_swap_after"][key] < row["host_vmstat_swap_before"][key]
           for key in ("pswpin", "pswpout")) \
            or row["release_to_all_children_exit_seconds"] > row["wall_seconds"]:
        raise VerificationError("batch timing/vmstat differs")
    ready_pids = []; ready_cpus = []; ready_times = []
    for index, ready in enumerate(row["ready_evidence"]):
        _keys(ready, {"schema_version", "task_id", "pid", "workers", "cpu_affinity",
            "thread_environment", "ready_monotonic", "proposal_sha256",
            "resident_lease_digest"}, "batch READY")
        if ready["schema_version"] != "m04r14-exact-child-ready-v1" \
                or ready["task_id"] != f"throughput-t{index}-c{index}" \
                or ready["workers"] != 1 or len(ready["cpu_affinity"]) != 1 \
                or ready["pid"] != row["children"][index]["pid"] \
                or ready["cpu_affinity"] != row["children"][index]["cpus"] \
                or ready["proposal_sha256"] != proposal_shas[index] \
                or ready["resident_lease_digest"] != attempts[index]["lease_before"] \
                or ready["thread_environment"] != _child_environment(runtime) \
                or not _number(ready["ready_monotonic"]):
            raise VerificationError("batch READY values differ")
        ready_pids.append(ready["pid"]); ready_cpus.extend(ready["cpu_affinity"])
        ready_times.append(ready["ready_monotonic"])
    release = _keys(row["release_evidence"], {"schema_version", "released_monotonic",
        "pids", "ready_digests"}, "batch release")
    if release["schema_version"] != "m04r14-exact-child-release-v1" \
            or release["pids"] != ready_pids \
            or not _number(release["released_monotonic"]) \
            or release["released_monotonic"] < max(ready_times) \
            or release["ready_digests"] != [stable_hash(value) for value in row["ready_evidence"]] \
            or [child["pid"] for child in row["children"]] != ready_pids \
            or len(set(ready_pids)) != 8 or len(set(ready_cpus)) != 8:
        raise VerificationError("batch process crosslink differs")


def _proposal_report(value: Any, production: bool) -> None:
    if not production:
        if type(value) is not dict: raise VerificationError("test proposal report differs")
        return
    keys = {"schema_version", "generation_id", "query_episode_id", "candidates", "rows_scanned",
        "eligible_rows", "eligible_main_rows", "eligible_overflow_rows", "route_counts", "route_quotas",
        "block_rows", "block_order", "elapsed_seconds", "peak_rss_mb", "candidate_digest",
        "result_digest", "contract_digest", "input_digest"}
    row = _keys(value, keys, "proposal report")
    candidate_keys = {"episode_id", "symbol", "cutoff_ns", "quality_tier", "lower_bound_hex",
        "routes", "overflow_fallback"}
    candidates = []
    for candidate in row["candidates"]:
        _keys(candidate, candidate_keys, "proposal candidate")
        try: lower = float.fromhex(candidate["lower_bound_hex"])
        except (TypeError, ValueError) as exc: raise VerificationError("proposal bound differs") from exc
        if not math.isfinite(lower) or lower < 0 or lower.hex() != candidate["lower_bound_hex"]:
            raise VerificationError("proposal bound differs")
        candidates.append(BoundProposal(candidate["episode_id"], candidate["symbol"],
            candidate["cutoff_ns"], candidate["quality_tier"], lower,
            tuple(candidate["routes"]), candidate["overflow_fallback"]))
    contract = packed_bound_search_contract(branch_aware=True)["digest"]
    deterministic = {"schema_version": row["schema_version"], "contract_digest": contract,
        "generation_id": row["generation_id"], "query_episode_id": row["query_episode_id"],
        "rows_scanned": row["rows_scanned"], "eligible_rows": row["eligible_rows"],
        "eligible_main_rows": row["eligible_main_rows"],
        "eligible_overflow_rows": row["eligible_overflow_rows"], "route_counts": row["route_counts"],
        "route_quotas": row["route_quotas"], "candidate_digest": row["candidate_digest"],
        "real_forward_outcomes_accessed": False, "input_digest": row["input_digest"]}
    if row["contract_digest"] != contract or row["route_quotas"] != {"composite": 16385} \
            or row["route_counts"] != {"composite": len(candidates)} \
            or row["eligible_rows"] != row["eligible_main_rows"] + row["eligible_overflow_rows"] \
            or len(candidates) != min(16385, row["eligible_rows"]) \
            or candidates != sorted(candidates, key=lambda item: (item.lower_bound, item.episode_id)) \
            or row["candidate_digest"] != bound_proposal_candidate_digest(candidates) \
            or row["result_digest"] != stable_hash(deterministic):
        raise VerificationError("proposal report reconstruction differs")


def _proposal(source: Path | Mapping[str, Any], production: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _keys(_read(source) if isinstance(source, Path) else dict(source),
        {"state", "digest", "measurement", "created_at"}, "proposal leaf")
    if not _timestamp(payload["created_at"]): raise VerificationError("proposal timestamp differs")
    state = _seal(_without(payload, {"measurement", "created_at"}), "proposal")["state"]
    _keys(state, {"schema_version", "task_id", "case_id", "query_id", "query_binding",
        "resident_lease_digests", "source_binding_before", "source_binding_after",
        "resident_snapshot", "forward", "reverse", "semantic_digest"}, "proposal state")
    omitted = {"block_rows", "block_order", "elapsed_seconds", "peak_rss_mb"}
    _proposal_report(state["forward"], production); _proposal_report(state["reverse"], production)
    if state["schema_version"] != PROPOSAL_SCHEMA or state["source_binding_before"] != state["source_binding_after"] \
            or len(state["resident_lease_digests"]) != 4 or len(set(state["resident_lease_digests"])) != 1 \
            or state["forward"]["block_rows"] != 4096 or state["forward"]["block_order"] != "forward" \
            or state["reverse"]["block_rows"] != 4097 or state["reverse"]["block_order"] != "reverse" \
            or _without(state["forward"], omitted) != _without(state["reverse"], omitted) \
            or state["semantic_digest"] != stable_hash(_without(state["forward"], omitted)):
        raise VerificationError("proposal parity differs")
    measurement = payload["measurement"]
    allowed = {"wall_seconds", "forward_seconds", "reverse_seconds", "resources"}
    if production: allowed.add("spawned_process")
    _keys(measurement, allowed, "proposal measurement"); _resource(measurement["resources"])
    if not all(_number(measurement[key]) for key in ("wall_seconds", "forward_seconds", "reverse_seconds")) \
            or measurement["wall_seconds"] < measurement["forward_seconds"] + measurement["reverse_seconds"]:
        raise VerificationError("proposal timing differs")
    if production:
        _proposal_process(measurement["spawned_process"], state["task_id"])
    return state, measurement


def _production_proposal_binding(
    proposal: Mapping[str, Any], foundation: Mapping[str, Any],
    expected_bindings: Mapping[str, Mapping[str, Any]],
) -> None:
    binding = _keys(proposal["query_binding"], {"query_stock_prefix",
        "query_benchmark_prefix", "request", "packed_provenance_digest",
        "query_representation_digest", "packed_query_input_digest",
        "certified_input_digest"}, "query binding")
    request = _keys(binding["request"], {"search_datasets", "quality_tiers", "top_k",
        "cross_dataset", "deduplicate_overlaps", "max_per_instrument",
        "minimum_history_gap_bars"}, "query request")
    deterministic = _without(binding, {"packed_query_input_digest", "certified_input_digest"})
    resident = foundation["resident"]
    expected_lease = resident["lease"]["lease_digest"]
    if binding != expected_bindings.get(proposal["query_id"]) \
            or binding["certified_input_digest"] != stable_hash(deterministic) \
            or binding["packed_provenance_digest"] != foundation["provenance_digest"] \
            or proposal["forward"]["input_digest"] != binding["packed_query_input_digest"] \
            or proposal["reverse"]["input_digest"] != binding["packed_query_input_digest"] \
            or proposal["source_binding_before"] != binding \
            or proposal["source_binding_after"] != binding \
            or proposal["resident_snapshot"] != resident \
            or proposal["resident_lease_digests"] != [expected_lease] * 4 \
            or request != {"search_datasets": ["nasdaq"], "quality_tiers": ["A", "B"],
                "top_k": 20, "cross_dataset": False, "deduplicate_overlaps": True,
                "max_per_instrument": 3, "minimum_history_gap_bars": 60}:
        raise VerificationError("proposal causal binding differs")


def _production_query_bindings(foundation: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    config_path = Path(foundation["config_path"]).resolve(strict=True)
    registry_path = (Path(foundation["registry_root"]) / "query-registry.json").resolve(strict=True)
    causal_shas = foundation["causal_input_shas"]
    if _sha(config_path) != causal_shas["config"] or _sha(registry_path) != causal_shas["registry"]:
        raise VerificationError("production config/registry SHA differs")
    registry = _read(registry_path)
    if registry.get("registry_digest") != foundation["registry_digest"] \
            or type(registry.get("cases_data")) is not list:
        raise VerificationError("production registry binding differs")
    by_query = {row.get("episode_id"): row for row in registry["cases_data"]
                if type(row) is dict}
    if any(query_id not in by_query for query_id in THROUGHPUT_QUERY_IDS):
        raise VerificationError("production registry query is absent")
    config = load_config(config_path); source = source_from_spec(config.datasets["nasdaq"])
    benchmark = source.load_benchmark(); result = {}
    for query_id, case_id in zip(THROUGHPUT_QUERY_IDS, THROUGHPUT_CASE_IDS, strict=True):
        raw = by_query[query_id]
        if raw.get("case_id") != case_id or raw.get("dataset_id") != "nasdaq":
            raise VerificationError("production registry case identity differs")
        episode = build_episode(source, InstrumentKey("nasdaq", raw["symbol"]), raw["cutoff"],
                                raw["lookback"], raw["representation_version"])
        request = SearchQuery(episode.key, ("nasdaq",), ("A", "B"), 20, False, True, 3, 60)
        packed = PackedBoundQuery(episode.key.id, episode.key.instrument.source_symbol,
            int(episode.bars.timestamp.iloc[0].value),
            int(latest_eligible_cutoff(episode, 60).value), represent(episode),
            request.quality_tiers)
        if episode.key.id != query_id:
            raise VerificationError("production query reconstruction differs")
        deterministic = {"query_stock_prefix": asdict(causal_prefix_digest(
                source.load(episode.key.instrument), episode.key.cutoff)),
            "query_benchmark_prefix": None if benchmark is None else asdict(
                causal_prefix_digest(benchmark, episode.key.cutoff)),
            "request": {"search_datasets": list(request.search_datasets),
                "quality_tiers": list(request.quality_tiers), "top_k": request.top_k,
                "cross_dataset": request.cross_dataset,
                "deduplicate_overlaps": request.deduplicate_overlaps,
                "max_per_instrument": request.max_per_instrument,
                "minimum_history_gap_bars": request.minimum_history_gap_bars},
            "packed_provenance_digest": foundation["provenance_digest"],
            "query_representation_digest": representation_input_digest(packed.representation)}
        result[query_id] = {**deterministic,
            "packed_query_input_digest": _packed_query_input_digest(packed),
            "certified_input_digest": stable_hash(deterministic)}
    return result


def _certificate(certificate: Any, matches: Any, query_id: str, input_digest: str, production: bool) -> None:
    if not production:
        if type(certificate) is not dict or certificate.get("result_digest") is None:
            raise VerificationError("test certificate differs")
        return
    cert_keys = {"schema_version", "contract_digest", "generation_id", "query_episode_id", "input_digest",
        "eligible_candidates", "exact_evaluated", "safely_pruned", "stopped_early", "stop_threshold",
        "next_lower_bound", "maximum_quantized_bound_excess", "materialization_groups", "sparse_symbols",
        "batch_symbols", "rounds", "result_digest", "native_bound_accounting",
        "minimum_native_pruned_bound", "threshold_closure_passes"}
    match_keys = {"episode_id", "symbol", "cutoff", "total_distance", "component_distances",
        "alignment", "quality_tier"}
    _keys(certificate, cert_keys, "certificate")
    if type(matches) is not list or len(matches) != 20 or any(type(row) is not dict or set(row) != match_keys for row in matches):
        raise VerificationError("match schema differs")
    accounting = _keys(certificate["native_bound_accounting"], {"native_bound_evaluated",
        "exact_dtw_evaluated", "native_bound_pruned", "packed_bound_pruned"}, "native accounting")
    contract = certified_packed_search_contract(requested_positions=True, vector_lower_bounds=True,
        deferred_alignments=True, compact_scored=True, native_bound_deferral=True,
        streaming_threshold_closure=True, branch_aware_packed_bounds=True)
    totals = [(row["total_distance"], row["episode_id"]) for row in matches]
    if not all(type(accounting[key]) is int and accounting[key] >= 0 for key in accounting) \
            or not all(type(certificate[key]) is int and certificate[key] >= 0 for key in (
                "eligible_candidates", "exact_evaluated", "safely_pruned", "materialization_groups",
                "sparse_symbols", "batch_symbols")) \
            or type(certificate["stopped_early"]) is not bool \
            or not all(type(certificate[key]) is float and math.isfinite(certificate[key])
                and certificate[key] >= 0 for key in ("stop_threshold",
                    "maximum_quantized_bound_excess")):
        raise VerificationError("certificate numeric schema differs")
    for match in matches:
        if type(match["episode_id"]) is not str or len(match["episode_id"]) != 24 \
                or type(match["symbol"]) is not str or not match["symbol"] \
                or type(match["cutoff"]) is not str or type(match["total_distance"]) is not float \
                or not math.isfinite(match["total_distance"]) or match["total_distance"] < 0 \
                or type(match["component_distances"]) is not dict \
                or set(match["component_distances"]) != COMPONENT_NAMES \
                or not all(type(value) is float and math.isfinite(value) and value >= 0
                    for value in match["component_distances"].values()) \
                or type(match["alignment"]) is not list or not match["alignment"] \
                or not all(type(pair) is list and len(pair) == 2 and all(type(index) is int
                    and index >= 0 for index in pair) for pair in match["alignment"]) \
                or match["quality_tier"] not in {"A", "B"}:
            raise VerificationError("match value schema differs")
    rounds = certificate["rounds"]
    round_keys = {"frontier_rows", "exact_rows", "next_lower_bound", "constrained_threshold",
        "selected_rows", "certified", "proposal_digest"}
    if type(rounds) is not list or not rounds or any(type(row) is not dict or set(row) != round_keys
            or type(row["frontier_rows"]) is not int or row["frontier_rows"] <= 0
            or type(row["exact_rows"]) is not int or row["exact_rows"] < 0
            or type(row["selected_rows"]) is not int or not 0 <= row["selected_rows"] <= 20
            or type(row["certified"]) is not bool
            or type(row["constrained_threshold"]) is not float
            or not math.isfinite(row["constrained_threshold"])
            or (row["next_lower_bound"] is not None and (type(row["next_lower_bound"]) is not float
                or not math.isfinite(row["next_lower_bound"])))
            for row in rounds):
        raise VerificationError("certificate rounds differ")
    closures = certificate["threshold_closure_passes"]
    closure_keys = {"lower_exclusive", "upper_inclusive", "admitted_rows",
        "cumulative_native_bound_evaluated", "cumulative_exact_dtw_evaluated", "selected_rows",
        "resulting_threshold", "minimum_packed_unclassified_bound", "minimum_native_pruned_bound",
        "excluded_prefix_digest", "admitted_set_digest", "scan_result_digest", "certified"}
    if type(closures) is not list or any(type(row) is not dict or set(row) != closure_keys
            for row in closures):
        raise VerificationError("certificate closure schema differs")
    if closures:
        if rounds[-1]["certified"] is not False or closures[-1]["certified"] is not True \
                or any(row["certified"] is not False for row in closures[:-1]) \
                or closures[0]["lower_exclusive"] is not None \
                or closures[-1]["selected_rows"] != 20 \
                or closures[-1]["resulting_threshold"] != certificate["stop_threshold"]:
            raise VerificationError("certificate closure terminal state differs")
    elif rounds[-1]["certified"] is not True or rounds[-1]["selected_rows"] != 20 \
            or rounds[-1]["constrained_threshold"] != certificate["stop_threshold"]:
        raise VerificationError("certificate round terminal state differs")
    deterministic = {"schema_version": contract["schema_version"], "contract_digest": contract["digest"],
        "generation_id": certificate["generation_id"], "query_episode_id": query_id,
        "input_digest": certificate["input_digest"], "eligible_candidates": certificate["eligible_candidates"],
        "exact_evaluated": certificate["exact_evaluated"], "safely_pruned": certificate["safely_pruned"],
        "stopped_early": certificate["stopped_early"], "stop_threshold_hex": certificate["stop_threshold"].hex(),
        "next_lower_bound_hex": None if certificate["next_lower_bound"] is None else certificate["next_lower_bound"].hex(),
        "maximum_quantized_bound_excess_hex": certificate["maximum_quantized_bound_excess"].hex(),
        "rounds": certificate["rounds"], "matches": [{"episode_id": row["episode_id"],
            "total_hex": row["total_distance"].hex(), "components": {key: value.hex()
            for key, value in sorted(row["component_distances"].items())}, "alignment": row["alignment"]}
            for row in matches], "real_forward_outcomes_accessed": False,
        "native_bound_accounting": accounting,
        "minimum_native_pruned_bound_hex": None if certificate["minimum_native_pruned_bound"] is None
            else certificate["minimum_native_pruned_bound"].hex(),
        "threshold_closure_passes": certificate["threshold_closure_passes"]}
    if certificate["schema_version"] != contract["schema_version"] \
            or certificate["contract_digest"] != contract["digest"] \
            or certificate["generation_id"] != GENERATION_ID \
            or certificate["query_episode_id"] != query_id \
            or certificate["input_digest"] != input_digest \
            or certificate["eligible_candidates"] != certificate["exact_evaluated"] + certificate["safely_pruned"] \
            or certificate["exact_evaluated"] != accounting["exact_dtw_evaluated"] \
            or accounting["native_bound_evaluated"] != accounting["exact_dtw_evaluated"] + accounting["native_bound_pruned"] \
            or certificate["eligible_candidates"] != accounting["native_bound_evaluated"] + accounting["packed_bound_pruned"] \
            or certificate["stop_threshold"] != max(row["total_distance"] for row in matches) \
            or (accounting["native_bound_pruned"] == 0) is not (
                certificate["minimum_native_pruned_bound"] is None
            ) \
            or certificate["materialization_groups"] != certificate["sparse_symbols"] + certificate["batch_symbols"] \
            or not 0 <= certificate["maximum_quantized_bound_excess"] <= ENGINE_TOLERANCE \
            or totals != sorted(totals) or len({row["episode_id"] for row in matches}) != 20 \
            or certificate["result_digest"] != stable_hash(deterministic):
        raise VerificationError("certificate reconstruction differs")


def _attempt(value: Any, proposal: Mapping[str, Any], workers: int, production: bool) -> dict[str, Any]:
    semantic = _keys(value, {"schema_version", "case_id", "query_id", "workers",
        "proposal_semantic_digest", "certificate", "matches", "certificate_result_digest",
        "match_digest", "lease_before", "lease_after", "source_binding_before",
        "source_binding_after"} if production else {"schema_version", "case_id", "query_id", "workers",
        "proposal_semantic_digest", "certificate", "matches", "certificate_result_digest",
        "match_digest", "lease_before", "lease_after"}, "attempt")
    input_digest = proposal["query_binding"].get("certified_input_digest", "")
    _certificate(semantic["certificate"], semantic["matches"], semantic["query_id"], input_digest, production)
    if semantic["schema_version"] != ATTEMPT_SCHEMA or semantic["workers"] != workers \
            or semantic["case_id"] != proposal["case_id"] or semantic["query_id"] != proposal["query_id"] \
            or semantic["proposal_semantic_digest"] != proposal["semantic_digest"] \
            or semantic["certificate_result_digest"] != semantic["certificate"]["result_digest"] \
            or semantic["match_digest"] != stable_hash(semantic["matches"]) \
            or semantic["lease_before"] != semantic["lease_after"]:
        raise VerificationError("attempt reconstruction differs")
    if production and (semantic["source_binding_before"] != semantic["source_binding_after"]
            or semantic["source_binding_before"] != proposal["source_binding_before"]):
        raise VerificationError("attempt source binding differs")
    return semantic


def _attempt_leaf(source: Path | Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _keys(_read(source) if isinstance(source, Path) else dict(source),
        {"semantic", "measurement", "created_at"}, "attempt leaf")
    if not _timestamp(payload["created_at"]): raise VerificationError("attempt timestamp differs")
    return _seal(payload["semantic"], "attempt semantic")["state"], payload["measurement"]


def _core(value: Mapping[str, Any]) -> dict[str, Any]:
    return _without(value, {"workers", "lease_before", "lease_after"})


def _selection(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values: dict[tuple[str, int], list[float]] = {}
    for row in rows: values.setdefault((row["case_id"], row["workers"]), []).append(row["measurement"]["wall_seconds"])
    summaries = {}; stable = []
    for worker in WORKERS:
        medians = []; ratios = []
        for case in FROZEN_CASE_IDS:
            samples = sorted(values[(case, worker)])
            if len(samples) != 3: raise VerificationError("selection sample count differs")
            medians.append(samples[1]); ratios.append(max(samples) / max(min(samples), 1e-12))
        score = sum(medians); valid = max(ratios) <= 1.25
        summaries[str(worker)] = {"sample_count": 12, "per_case_median_seconds": medians,
            "raw_score_seconds": score, "maximum_case_repeat_ratio": max(ratios), "stable": valid}
        if valid: stable.append(worker)
    if not stable: raise VerificationError("no stable worker")
    fastest = min(summaries[str(worker)]["raw_score_seconds"] for worker in stable)
    selected = min(worker for worker in stable if summaries[str(worker)]["raw_score_seconds"] <= fastest * 1.03)
    state = {"rule": "each case/worker max/min<=1.25; raw score=sum of four three-repetition medians; smallest worker with raw score<=1.03*fastest",
        "workers": summaries, "selected_workers": selected}
    return {"state": state, "digest": stable_hash(state)}


def verify_terminal(root: Path, *, repository: Path, require_production: bool = True) -> dict[str, Any]:
    root = root.resolve(strict=True); repository = repository.resolve(strict=True)
    if root.is_symlink() or _tree(root) != _paths(True) or (root / "INCOMPLETE.json").exists():
        raise VerificationError("terminal exact tree differs")
    complete_raw_value, initial_complete_sha, complete_identity = _read_snapshot(root / "COMPLETE.json")
    complete_raw = _keys(complete_raw_value, {"state", "digest", "created_at"}, "complete")
    if not _timestamp(complete_raw["created_at"]): raise VerificationError("complete timestamp differs")
    complete = _seal(_without(complete_raw, {"created_at"}), "complete")["state"]
    _keys(complete, {"schema_version", "status", "semantic_digest", "measurement_digest",
        "selected_workers", "final_lease", "semantic_passed", "performance_selection_only",
        "production_promotion_authorized", "leaf_manifest", "leaf_manifest_digest"}, "complete state")
    manifest = complete["leaf_manifest"]; expected_paths = sorted(_paths(False))
    if complete["schema_version"] != COMPLETE_SCHEMA or complete["status"] != "complete" \
            or complete["semantic_passed"] is not True or complete["production_promotion_authorized"] is not False \
            or complete["performance_selection_only"] is not True or type(manifest) is not list \
            or [row.get("path") for row in manifest] != expected_paths \
            or any(type(row) is not dict or set(row) != {"path", "sha256"} for row in manifest) \
            or complete["leaf_manifest_digest"] != stable_hash(manifest):
        raise VerificationError("complete reconstruction differs")
    snapshots: dict[str, dict[str, Any]] = {}
    snapshot_identities: dict[str, tuple[int, ...]] = {}
    manifest_shas = {row["path"]: row["sha256"] for row in manifest}
    for row in manifest:
        value, digest, identity = _read_snapshot(root / row["path"], row["sha256"])
        snapshots[row["path"]] = value; snapshot_identities[row["path"]] = identity
        if digest != row["sha256"]:
            raise VerificationError("leaf SHA differs")
    def leaf(relative: str) -> dict[str, Any]:
        return snapshots[relative]
    contract_seal = _seal(leaf("CONTRACT.json"), "contract")
    contract = contract_seal["state"]
    _keys(contract, {"schema_version", "status", "runtime_binding", "foundation", "output_root",
        "query_ids", "throughput_query_ids", "workers", "repetitions", "execution_policy",
        "selection_rule"}, "contract state")
    if contract["schema_version"] != PREREG_SCHEMA or contract["status"] != "frozen_before_run" \
            or contract["output_root"] != str(root) or contract["query_ids"] != list(FROZEN_QUERY_IDS) \
            or contract["throughput_query_ids"] != list(THROUGHPUT_QUERY_IDS) \
            or contract["workers"] != list(WORKERS) or contract["repetitions"] != 3 \
            or contract["execution_policy"] != _execution_policy() \
            or contract["selection_rule"] != PREREG_SELECTION_RULE:
        raise VerificationError("contract reconstruction differs")
    _validate_foundation(contract["foundation"], production=require_production,
                         repository=repository)
    _validate_runtime(contract["runtime_binding"], repository, require_production)
    if require_production:
        _validate_prereg_lineage(contract_seal, contract["runtime_binding"], repository)
    run_raw = _keys(leaf("RUN_STARTED.json"), {"state", "digest", "created_at"}, "run started")
    if not _timestamp(run_raw["created_at"]): raise VerificationError("run timestamp differs")
    run = _seal(_without(run_raw, {"created_at"}), "run started")["state"]
    _keys(run, {"schema_version", "status", "development_only", "cases_previously_exposed",
        "direct_raw_authority_accessed_by_this_run",
        "authority_derived_prerequisite_evidence_accessed_by_this_run",
        "forward_outcomes_accessed_by_this_run",
        "raw_authority_or_outcome_paths_accepted", "production_promotion_authorized", "query_ids",
        "case_ids", "workers", "repetitions", "proposal_threads", "throughput", "numeric_policy",
        "foundation", "runtime_binding", "preregistration_digest"}, "run state")
    if run["schema_version"] != RUN_SCHEMA or run["status"] != "started" \
            or run["development_only"] is not True or run["cases_previously_exposed"] is not True \
            or run["direct_raw_authority_accessed_by_this_run"] is not False \
            or run["authority_derived_prerequisite_evidence_accessed_by_this_run"] is not True \
            or run["forward_outcomes_accessed_by_this_run"] is not False \
            or run["raw_authority_or_outcome_paths_accepted"] is not False \
            or run["production_promotion_authorized"] is not False \
            or run["query_ids"] != list(FROZEN_QUERY_IDS) or run["case_ids"] != list(FROZEN_CASE_IDS) \
            or run["workers"] != list(WORKERS) or run["repetitions"] != 3 \
            or run["proposal_threads"] != 8 or run["throughput"] != RUN_THROUGHPUT \
            or run["numeric_policy"] != RUN_NUMERIC_POLICY \
            or run["foundation"] != contract["foundation"] \
            or run["runtime_binding"] != contract["runtime_binding"] \
            or run["preregistration_digest"] != contract_seal["digest"]:
        raise VerificationError("run reconstruction differs")
    production = require_production
    expected_query_bindings = (
        _production_query_bindings(contract["foundation"]) if production else {}
    )
    primary_semantics = []; primary_measurements = []; references = {}
    for repetition in range(3):
        for ordinal in _rotated(range(4), repetition):
            proposal, proposal_measurement = _proposal(leaf(f"primary/r{repetition}/c{ordinal}/PROPOSAL.json"), production)
            if production:
                _production_proposal_binding(
                    proposal, contract["foundation"], expected_query_bindings,
                )
                _proposal_process_binding(proposal_measurement["spawned_process"], proposal,
                                          contract["runtime_binding"])
            task = f"primary-r{repetition}-c{ordinal}"
            if (proposal["task_id"], proposal["case_id"], proposal["query_id"]) != (task, FROZEN_CASE_IDS[ordinal], FROZEN_QUERY_IDS[ordinal]):
                raise VerificationError("primary proposal identity differs")
            for worker in _rotated(WORKERS, repetition + ordinal):
                semantic_row, measurement_row = _attempt_leaf(leaf(f"primary/r{repetition}/c{ordinal}/EXACT-w{worker}.json"))
                expected_identity = {"lane": "primary", "repetition": repetition,
                    "case_id": proposal["case_id"], "query_id": proposal["query_id"],
                    "workers": worker, "task_id": task}
                if _without(semantic_row, {"attempt"}) != expected_identity:
                    raise VerificationError("primary row identity differs")
                attempt = _attempt(semantic_row["attempt"], proposal, worker, production)
                if production and attempt["lease_before"] != contract["foundation"]["resident"]["lease"]["lease_digest"]:
                    raise VerificationError("primary resident lease differs")
                if references.setdefault(proposal["case_id"], _core(attempt)) != _core(attempt):
                    raise VerificationError("primary semantic parity differs")
                expected_measurement = {"lane": "primary", "repetition": repetition,
                    "case_id": proposal["case_id"], "workers": worker, "task_id": task,
                    "measurement": measurement_row["measurement"], "proposal_measurement": proposal_measurement}
                if measurement_row != expected_measurement: raise VerificationError("primary measurement differs")
                _attempt_measurement(measurement_row["measurement"], production=production,
                                     workers=worker)
                primary_semantics.append(semantic_row); primary_measurements.append(measurement_row)
    throughput_semantics = []; throughput_measurements = []; proposal_digests = []
    for index in range(8):
        proposal, throughput_proposal_measurement = _proposal(
            leaf(f"throughput/t{index}/PROPOSAL.json"), production)
        if production:
            _production_proposal_binding(
                proposal, contract["foundation"], expected_query_bindings,
            )
            _proposal_process_binding(throughput_proposal_measurement["spawned_process"], proposal,
                                      contract["runtime_binding"])
        task = f"throughput-t{index}-c{index}"; proposal_digests.append(proposal["semantic_digest"])
        if (proposal["task_id"], proposal["case_id"], proposal["query_id"]) != (task, THROUGHPUT_CASE_IDS[index], THROUGHPUT_QUERY_IDS[index]):
            raise VerificationError("throughput proposal identity differs")
        control, control_measurement = _attempt_leaf(leaf(f"throughput/t{index}/CONTROL-w1.json"))
        exact_row, measurement_row = _attempt_leaf(leaf(f"throughput/t{index}/EXACT-w1.json"))
        control_attempt = _attempt(control, proposal, 1, production)
        attempt = _attempt(exact_row["attempt"], proposal, 1, production)
        if production and (control_attempt["lease_before"] != contract["foundation"]["resident"]["lease"]["lease_digest"]
                or attempt["lease_before"] != contract["foundation"]["resident"]["lease"]["lease_digest"]):
            raise VerificationError("throughput resident lease differs")
        _attempt_measurement(control_measurement, production=production, workers=1)
        _attempt_measurement(measurement_row["measurement"], production=production, workers=1)
        if _core(control_attempt) != _core(attempt) or (proposal["case_id"] in references and _core(attempt) != references[proposal["case_id"]]):
            raise VerificationError("throughput exact/control parity differs")
        expected = {"lane": "throughput-p8t1", "task_index": index, "case_id": proposal["case_id"],
            "query_id": proposal["query_id"], "workers": 1, "task_id": task, "attempt": attempt}
        expected_measurement = {"lane": "throughput-p8t1", "task_index": index,
            "case_id": proposal["case_id"], "workers": 1, "task_id": task,
            "measurement": measurement_row["measurement"]}
        if exact_row != expected or measurement_row != expected_measurement:
            raise VerificationError("throughput row reconstruction differs")
        throughput_semantics.append(exact_row); throughput_measurements.append(measurement_row)
    barrier = _seal(leaf("throughput/PROPOSALS_COMPLETE.json"), "barrier")
    if barrier["state"] != {"status": "all_eight_proposals_complete_before_timed_wave",
            "query_ids": list(THROUGHPUT_QUERY_IDS), "proposal_semantic_digests": proposal_digests}:
        raise VerificationError("throughput barrier differs")
    semantics = _seal(leaf("SEMANTICS.json"), "semantics")
    measurements = _seal(leaf("MEASUREMENTS.json"), "measurements")
    semantic_state = _keys(semantics["state"], {"schema_version", "status", "primary", "throughput",
        "case_reference_digests", "semantic_passed", "cases_previously_exposed",
        "direct_raw_authority_accessed_by_this_run",
        "authority_derived_prerequisite_evidence_accessed_by_this_run",
        "forward_outcomes_accessed_by_this_run"}, "semantic aggregate")
    measurement_state = _keys(measurements["state"], {"schema_version", "status", "primary", "throughput",
        "throughput_lane", "selection", "host_cpu_qualification",
        "primary_interactive_selected", "event_ledger",
        "event_ledger_digest"}, "measurement aggregate")
    if semantic_state["schema_version"] != SEMANTICS_SCHEMA or semantic_state["status"] != "semantic_pass" \
            or semantic_state["semantic_passed"] is not True or semantic_state["cases_previously_exposed"] is not True \
            or semantic_state["direct_raw_authority_accessed_by_this_run"] is not False \
            or semantic_state["authority_derived_prerequisite_evidence_accessed_by_this_run"] is not True \
            or semantic_state["forward_outcomes_accessed_by_this_run"] is not False \
            or semantic_state["primary"] != primary_semantics or semantic_state["throughput"] != throughput_semantics \
            or semantic_state["case_reference_digests"] != {key: stable_hash(value) for key, value in sorted(references.items())}:
        raise VerificationError("semantic aggregate reconstruction differs")
    selection = _selection(primary_measurements)
    selected_worker = selection["state"]["selected_workers"]
    interactive = [{"case_id": row["case_id"], "repetition": row["repetition"],
        "forward_proposal_seconds": row["proposal_measurement"]["forward_seconds"],
        "selected_exact_seconds": row["measurement"]["wall_seconds"],
        "interactive_seconds": row["proposal_measurement"]["forward_seconds"]
            + row["measurement"]["wall_seconds"],
        "reverse_proposal_role": "semantic-parity-only-excluded"}
        for row in primary_measurements if row["workers"] == selected_worker]
    if measurement_state["schema_version"] != MEASUREMENTS_SCHEMA or measurement_state["status"] != "measured" \
            or measurement_state["primary"] != primary_measurements \
            or measurement_state["throughput"] != throughput_measurements \
            or measurement_state["selection"] != selection \
            or measurement_state["primary_interactive_selected"] != interactive \
            or complete["selected_workers"] != selection["state"]["selected_workers"]:
        raise VerificationError("measurement aggregate reconstruction differs")
    lane = _keys(measurement_state["throughput_lane"], {"label", "wall_seconds", "resources", "tasks",
        "distinct_query_ids", "maximum_concurrent_exact_tasks", "workers_per_task",
        "observed_maximum_active_exact_tasks", "proposals_prepared_serially_with_threads",
        "serial_controls_excluded_from_lane_timing", "spawned_process_evidence"}, "throughput lane")
    _resource(lane["resources"])
    cpu = measurement_state["host_cpu_qualification"]
    _keys(cpu, {"schema_version", "required_cpus", "before", "after", "delta",
                "performance_valid"}, "host CPU qualification")
    if cpu != _host_cpu_qualification(cpu["before"], cpu["after"], required_cpus=8):
        raise VerificationError("host CPU qualification differs")
    if lane["label"] != "exact-stage exposed-host p8t1 microbenchmark" or lane["tasks"] != 8 \
            or lane["distinct_query_ids"] != list(THROUGHPUT_QUERY_IDS) or lane["workers_per_task"] != 1 \
            or lane["maximum_concurrent_exact_tasks"] != 8 \
            or lane["proposals_prepared_serially_with_threads"] != 8 \
            or lane["serial_controls_excluded_from_lane_timing"] is not True \
            or not _number(lane["wall_seconds"]) \
            or type(lane["observed_maximum_active_exact_tasks"]) is not int \
            or not 1 < lane["observed_maximum_active_exact_tasks"] <= 8:
        raise VerificationError("throughput lane values differ")
    if production:
        _batch_process(lane["spawned_process_evidence"],
                       [manifest_shas[f"throughput/t{index}/PROPOSAL.json"] for index in range(8)],
                       [row["attempt"] for row in throughput_semantics],
                       contract["runtime_binding"])
        if lane["wall_seconds"] != lane["spawned_process_evidence"][
                "release_to_all_children_exit_seconds"]:
            raise VerificationError("throughput lane wall differs from released batch")
    elif lane["spawned_process_evidence"] is not None:
        raise VerificationError("test batch process evidence differs")
    ledger = measurement_state["event_ledger"]
    if measurement_state["event_ledger_digest"] != stable_hash(ledger): raise VerificationError("ledger digest differs")
    previous = None
    for sequence, event in enumerate(ledger):
        if event.get("sequence") != sequence or event.get("previous_event_digest") != previous \
                or event.get("event_digest") != stable_hash(_without(event, {"event_digest"})):
            raise VerificationError("ledger chain differs")
        previous = event["event_digest"]
    rows = [_without(event, {"previous_event_digest", "event_digest"}) for event in ledger]
    expected_prefix = []
    for index in range(8):
        task = f"throughput-t{index}-c{index}"
        expected_prefix.extend((
            {"sequence": len(expected_prefix), "event": "proposal_started", "task_id": task,
             "active_proposals": 1},
            {"sequence": len(expected_prefix) + 1, "event": "proposal_completed", "task_id": task,
             "active_proposals": 0, "proposal_semantic_digest": proposal_digests[index]},
        ))
    expected_prefix.append({"sequence": len(expected_prefix), "event": "barrier_released",
                            "barrier_digest": barrier["digest"]})
    if rows[:len(expected_prefix)] != expected_prefix:
        raise VerificationError("proposal event sequence differs")
    cursor = len(expected_prefix); batch = []
    while cursor < len(rows) and rows[cursor].get("event") in {"batch_exact_started", "batch_exact_ended"}:
        batch.append(rows[cursor]); cursor += 1
    tasks = {f"throughput-t{index}-c{index}" for index in range(8)}
    if len(batch) != 16 or {row["task_id"] for row in batch if row["event"] == "batch_exact_started"} != tasks \
            or {row["task_id"] for row in batch if row["event"] == "batch_exact_ended"} != tasks:
        raise VerificationError("batch event sequence differs")
    if any(set(row) != {"sequence", "event", "task_id", "active_exact_tasks"}
           for row in batch):
        raise VerificationError("batch event schema differs")
    observed_maximum = 0
    if production:
        expected_batch = [
            {"sequence": len(expected_prefix) + index, "event": "batch_exact_started",
             "task_id": f"throughput-t{index}-c{index}", "active_exact_tasks": 8}
            for index in range(8)
        ] + [
            {"sequence": len(expected_prefix) + 8 + index, "event": "batch_exact_ended",
             "task_id": f"throughput-t{index}-c{index}", "active_exact_tasks": 0}
            for index in range(8)
        ]
        if batch != expected_batch:
            raise VerificationError("production concurrency event replay differs")
        observed_maximum = 8
    else:
        active = 0; active_tasks = set()
        for event in batch:
            if event["event"] == "batch_exact_started":
                if event["task_id"] in active_tasks: raise VerificationError("duplicate active task")
                active_tasks.add(event["task_id"]); active += 1
            else:
                if event["task_id"] not in active_tasks: raise VerificationError("inactive task ended")
                active_tasks.remove(event["task_id"]); active -= 1
            if event["active_exact_tasks"] != active:
                raise VerificationError("concurrency counter replay differs")
            observed_maximum = max(observed_maximum, active)
        if active or active_tasks: raise VerificationError("concurrency did not close")
    if lane["observed_maximum_active_exact_tasks"] != observed_maximum:
        raise VerificationError("throughput observed concurrency differs from ledger")
    for index in range(8):
        control = _attempt_leaf(leaf(f"throughput/t{index}/CONTROL-w1.json"))[0]
        expected = {"sequence": cursor, "event": "control_completed_after_wave",
            "task_id": f"throughput-t{index}-c{index}", "semantic_digest": stable_hash(_core(control))}
        if cursor >= len(rows) or rows[cursor] != expected: raise VerificationError("control event differs")
        cursor += 1
    for index in range(8):
        attempt = _attempt_leaf(leaf(f"throughput/t{index}/EXACT-w1.json"))[0]["attempt"]
        expected = {"sequence": cursor, "event": "batch_exact_completed",
            "task_id": f"throughput-t{index}-c{index}", "semantic_digest": stable_hash(_core(attempt))}
        if cursor >= len(rows) or rows[cursor] != expected: raise VerificationError("batch completion event differs")
        cursor += 1
    if cursor != len(rows): raise VerificationError("event ledger terminal length differs")
    if semantics["digest"] != complete["semantic_digest"] or measurements["digest"] != complete["measurement_digest"]:
        raise VerificationError("complete aggregate links differ")
    final = complete["final_lease"]
    if production:
        _keys(final, {"resident_identity_digest", "source_binding_digests", "causal_input_shas"}, "final lease")
        expected_sources = {}
        for ordinal in range(8):
            relative = (f"primary/r0/c{ordinal}/PROPOSAL.json" if ordinal < 4
                        else f"throughput/t{ordinal}/PROPOSAL.json")
            expected_sources[str(ordinal)] = stable_hash(
                _proposal(leaf(relative), production)[0]["source_binding_before"])
        if final["source_binding_digests"] != expected_sources or final["resident_identity_digest"] != contract["foundation"]["resident"]["identity_digest"] \
                or final["causal_input_shas"] != contract["foundation"]["causal_input_shas"]:
            raise VerificationError("final lease differs")
    else:
        _keys(final, {"resident_identity_digest", "source_lease_digest"}, "test final lease")
        if final["resident_identity_digest"] != contract["foundation"].get("resident_identity_digest") \
                or final["source_lease_digest"] != contract["foundation"].get("source_lease_digest"):
            raise VerificationError("test final lease crosslink differs")
    if _tree(root) != _paths(True) or _sha(root / "COMPLETE.json") != initial_complete_sha \
            or _observed_identity(root / "COMPLETE.json") != complete_identity \
            or any(_sha(root / row["path"]) != row["sha256"]
                   or _observed_identity(root / row["path"]) != snapshot_identities[row["path"]]
                   for row in manifest):
        raise VerificationError("final terminal manifest recheck differs")
    state = {"schema_version": VERIFICATION_SCHEMA, "status": "verified", "passed": True,
        "candidate_root": str(root), "candidate_complete_sha256": _sha(root / "COMPLETE.json"),
        "candidate_complete_digest": complete_raw["digest"], "terminal_tree_digest": stable_hash(manifest),
        "semantic_digest": semantics["digest"], "measurement_digest": measurements["digest"],
        "selected_workers": complete["selected_workers"], "primary_rows": 48,
        "throughput_rows": 8,
        "direct_raw_authority_accessed_by_verifier": False,
        "authority_derived_prerequisite_evidence_accessed_by_verifier": True,
        "forward_outcomes_accessed_by_verifier": False}
    return {**state, "result_digest": stable_hash(state)}


def _atomic(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink(): raise VerificationError("verification output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _strict_bytes(payload) + b"\n"; descriptor, temporary = tempfile.mkstemp(prefix=".verify.", dir=path.parent)
    temp = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw); handle.flush(); os.fsync(handle.fileno())
        os.link(temp, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally: temp.unlink(missing_ok=True)


def publish_verification(candidate_root: Path, verification_root: Path, *, repository: Path,
                         require_production: bool = True) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    _reject_symlink_ancestry(candidate_root)
    candidate = candidate_root.resolve(strict=True)
    output = _absent_canonical(verification_root)
    if _overlap(candidate, output):
        raise VerificationError("candidate/verification roots overlap")
    if require_production:
        expected_candidate = (repository / CANDIDATE_RELATIVE).resolve(strict=True)
        expected_output = repository / VERIFICATION_RELATIVE
        if candidate != expected_candidate or output != expected_output:
            raise VerificationError("production verification roots differ")
        protected = [repository, repository / ".git", repository / "src",
            repository / "experiments", repository / "config/datasets.example.yaml",
            repository / "config/data/analogues/m04r11",
            repository / "config/data/analogues/m04r13",
            repository / "config/data/analogues/m04r14/evidence-catalog-v1",
            repository / "config/data/analogues/m04r14/adversarial-oracle-v1",
            repository / "config/data/analogues/poc/m04r/packed-bound-full",
            Path("/dev/shm/market-analogues/m04r11-candidate-v2")]
        # The two canonical M14 terminal descendants are deliberately exempt
        # from the broad repository ancestor; every other protected input is not.
        if any(_overlap(output, path.resolve(strict=False)) for path in protected[1:]):
            raise VerificationError("verification output overlaps protected input")
    state = verify_terminal(candidate, repository=repository, require_production=require_production)
    _reject_symlink_ancestry(candidate_root)
    if candidate_root.resolve(strict=True) != candidate or _absent_canonical(verification_root) != output:
        raise VerificationError("verification topology changed before publication")
    verification_root.mkdir(parents=False)
    payload = {"state": state, "digest": stable_hash(state), "created_at": datetime.now(timezone.utc).isoformat()}
    _atomic(verification_root / "VERIFIED.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--verification-root", type=Path, required=True)
    args = parser.parse_args(argv)
    repository = Path(__file__).resolve().parents[2]
    payload = publish_verification(args.candidate_root, args.verification_root,
        repository=repository, require_production=True)
    print(json.dumps(payload, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
