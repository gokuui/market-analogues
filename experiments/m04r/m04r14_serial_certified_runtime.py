"""Authority-free spawned runtime for serial certified-search experiments.

The module deliberately owns no experiment ledger, preregistration, terminal
schema, authority path, or outcome path.  A caller supplies a frozen case table,
persists the returned proposal evidence, binds that exact file, and only then
may launch the one-worker exact child.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

from market_analogues.types import stable_hash


PROPOSAL_REQUEST_SCHEMA = "m04r14-serial-proposal-child-request-v1"
PROPOSAL_READY_SCHEMA = "m04r14-serial-proposal-child-ready-v1"
PROPOSAL_RELEASE_SCHEMA = "m04r14-serial-proposal-child-release-v1"
EXACT_REQUEST_SCHEMA = "m04r14-serial-exact-child-request-v1"
EXACT_READY_SCHEMA = "m04r14-serial-exact-child-ready-v1"
PROPOSAL_THREADS = 8
EXACT_WORKERS = 1
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    "NUMBA_NUM_THREADS", "NUMBA_THREADING_LAYER",
)


class RuntimeErrorEvidence(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeCase:
    ordinal: int
    case_id: str
    query_id: str
    registry_case: dict[str, Any]


@dataclass(frozen=True)
class PreparedCase:
    task_id: str
    case: RuntimeCase
    semantic: dict[str, Any]
    measurement: dict[str, Any]


@dataclass(frozen=True)
class ExactAttempt:
    semantic: dict[str, Any]
    measurement: dict[str, Any]


def _strict_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise RuntimeErrorEvidence("evidence is not strict finite JSON") from exc


def _pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in rows:
        if key in value:
            raise RuntimeErrorEvidence("duplicate JSON key")
        value[key] = item
    return value


def _read(path: Path, expected_sha: str | None = None) -> dict[str, Any]:
    if path.is_symlink():
        raise RuntimeErrorEvidence("JSON symlink is forbidden")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeErrorEvidence("JSON path is not a regular file")
        raw = bytearray()
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block: break
            raw.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns,
                            row.st_ctime_ns, row.st_mode)
    digest = sha256(raw).hexdigest()
    if identity(before) != identity(after) or not stat.S_ISREG(before.st_mode) \
            or (expected_sha is not None and digest != expected_sha):
        raise RuntimeErrorEvidence("JSON identity/SHA changed")
    try:
        value = json.loads(bytes(raw), object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                RuntimeErrorEvidence(f"nonfinite JSON token: {token}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeErrorEvidence("invalid JSON") from exc
    def finite(item: Any) -> bool:
        return not (type(item) is float and not math.isfinite(item)) and (
            type(item) is not list or all(finite(child) for child in item)) and (
            type(item) is not dict or all(finite(child) for child in item.values()))
    if type(value) is not dict or not finite(value):
        raise RuntimeErrorEvidence("JSON shape/nonfinite value differs")
    return value


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeErrorEvidence("create-only target exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".runtime-", dir=path.parent)
    temp = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_strict_bytes(value) + b"\n"); handle.flush(); os.fsync(handle.fileno())
        os.link(temp, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        temp.unlink(missing_ok=True)


def _file_observation(path: Path) -> tuple[str, tuple[int, ...]]:
    if path.is_symlink():
        raise RuntimeErrorEvidence("file symlink is forbidden")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeErrorEvidence("file is not a regular file")
        digest = sha256()
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block: break
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                before.st_ctime_ns, before.st_mode)
    if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                    after.st_ctime_ns, after.st_mode):
        raise RuntimeErrorEvidence("file identity changed")
    return digest.hexdigest(), identity


def _sha(path: Path) -> str:
    return _file_observation(path)[0]


def _reject_symlink_ancestry(path: Path) -> None:
    absolute = path.absolute()
    for value in (absolute, *absolute.parents):
        if value.is_symlink():
            raise RuntimeErrorEvidence("path ancestry contains a symlink")


def _keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise RuntimeErrorEvidence(f"{label} exact keys differ")
    return value


def _finite_number(value: Any) -> bool:
    return type(value) in {int, float} and not isinstance(value, bool) \
        and math.isfinite(value)


def _digest(value: Any, label: str) -> str:
    if type(value) is not str or len(value) != 64 \
            or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeErrorEvidence(f"{label} differs")
    return value


def _thread_environment() -> dict[str, str | None]:
    return {key: os.environ.get(key) if key == "NUMBA_THREADING_LAYER" else "1"
            for key in THREAD_ENV_KEYS}


def _child_environment() -> dict[str, str]:
    value = dict(os.environ)
    for key in THREAD_ENV_KEYS:
        if key != "NUMBA_THREADING_LAYER": value[key] = "1"
    return value


def _proc_memory(pid: int) -> dict[str, int]:
    result = {"VmRSS": 0, "VmHWM": 0, "VmSwap": 0}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            name = line.split(":", 1)[0]
            if name in result: result[name] = int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError):
        pass
    return result


def _vmstat() -> dict[str, int]:
    result = {"pswpin": 0, "pswpout": 0}
    try:
        for line in Path("/proc/vmstat").read_text().splitlines():
            key, raw = line.split()
            if key in result: result[key] = int(raw)
    except (OSError, ValueError):
        pass
    return result


def _contains_forbidden_input(value: Any, seen: set[int] | None = None) -> bool:
    """Reject nested truth-path channels while allowing ordinary descriptive text."""
    if seen is None:
        seen = set()
    if type(value) in {dict, list}:
        identity = id(value)
        if identity in seen:
            return True
        seen.add(identity)
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str or "authorit" in key.lower() or "outcome" in key.lower():
                return True
            if _contains_forbidden_input(item, seen):
                return True
        return False
    if type(value) is list:
        return any(_contains_forbidden_input(item, seen) for item in value)
    if type(value) is str and ("/" in value or "\\" in value):
        components = value.lower().replace("\\", "/").split("/")
        return any("authorit" in item or "outcome" in item for item in components)
    return False


def _validate_case_table(values: Sequence[RuntimeCase]) -> tuple[RuntimeCase, ...]:
    cases = tuple(values)
    if not cases or any(type(case.ordinal) is not int or case.ordinal != index
            or type(case.case_id) is not str or not case.case_id
            or type(case.query_id) is not str or not case.query_id
            or type(case.registry_case) is not dict
            or case.registry_case.get("case_id") != case.case_id
            or case.registry_case.get("episode_id") != case.query_id
            or _contains_forbidden_input(case.registry_case)
            for index, case in enumerate(cases)) \
            or len({case.case_id for case in cases}) != len(cases) \
            or len({case.query_id for case in cases}) != len(cases):
        raise RuntimeErrorEvidence("frozen case table differs")
    return cases


def case_table_payload(cases: Sequence[RuntimeCase]) -> list[dict[str, Any]]:
    return [{"ordinal": row.ordinal, "case_id": row.case_id,
             "query_id": row.query_id, "registry_case": row.registry_case} for row in cases]


class SerialSpawnedRuntime:
    """Serial authority-free proposal/exact child supervisor."""

    def __init__(self, *, repository: Path, cases: Sequence[RuntimeCase],
                 resident_lease_digest: str, scratch_root: Path | None = None,
                 resident_snapshot: Mapping[str, Any] | None = None,
                 proposal_command: Callable[[Path, Path], Sequence[str]] | None = None,
                 exact_command: Callable[[Path, Path], Sequence[str]] | None = None,
                 proposal_validator: Callable[[dict[str, Any], RuntimeCase, str], PreparedCase] | None = None,
                 exact_validator: Callable[[dict[str, Any], PreparedCase], ExactAttempt] | None = None,
                 startup_timeout: float = 600, task_timeout: float = 3600) -> None:
        self.repository = repository.resolve(strict=True)
        supplied_cases = _validate_case_table(cases)
        try:
            frozen_table = json.loads(_strict_bytes(case_table_payload(supplied_cases)))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:  # defensive; bytes are local
            raise RuntimeErrorEvidence("frozen case table is invalid") from exc
        self.cases = _validate_case_table(tuple(RuntimeCase(
            row["ordinal"], row["case_id"], row["query_id"], row["registry_case"],
        ) for row in frozen_table))
        self.case_table = frozen_table
        self.case_table_digest = stable_hash(self.case_table)
        if type(resident_lease_digest) is not str or not resident_lease_digest:
            raise RuntimeErrorEvidence("resident lease digest differs")
        self.resident_lease_digest = resident_lease_digest
        self.resident_snapshot = None if resident_snapshot is None else dict(resident_snapshot)
        self._temporary = None if scratch_root is not None else tempfile.TemporaryDirectory(
            prefix="m04r14-serial-runtime-")
        self.scratch = (scratch_root if scratch_root is not None else Path(self._temporary.name))
        _reject_symlink_ancestry(self.scratch)
        self.scratch.mkdir(parents=True, exist_ok=True)
        if not self.scratch.is_dir() or self.scratch.is_symlink():
            raise RuntimeErrorEvidence("scratch root is not a real directory")
        self.proposal_command = proposal_command
        self.exact_command = exact_command
        self.proposal_validator = proposal_validator or self._default_proposal_validator
        self.exact_validator = exact_validator or self._default_exact_validator
        if type(startup_timeout) not in {int, float} or isinstance(startup_timeout, bool) \
                or type(task_timeout) not in {int, float} or isinstance(task_timeout, bool) \
                or not math.isfinite(startup_timeout) or not math.isfinite(task_timeout) \
                or startup_timeout <= 0 or task_timeout <= 0:
            raise RuntimeErrorEvidence("child timeout policy differs")
        self.startup_timeout = startup_timeout; self.task_timeout = task_timeout
        self._sequence = 0
        self._proposal_files: dict[str, tuple[Path, str, tuple[int, ...]]] = {}

    @staticmethod
    def _default_proposal_validator(payload: dict[str, Any], case: RuntimeCase,
                                    task_id: str) -> PreparedCase:
        row = _keys(payload, {"semantic", "measurement", "created_at"}, "proposal result")
        seal = _keys(row["semantic"], {"state", "digest"}, "proposal semantic")
        if seal["digest"] != stable_hash(seal["state"]):
            raise RuntimeErrorEvidence("proposal semantic seal differs")
        state = seal["state"]
        if (state.get("task_id"), state.get("case_id"), state.get("query_id")) != (
                task_id, case.case_id, case.query_id):
            raise RuntimeErrorEvidence("proposal result identity differs")
        return PreparedCase(task_id, case, state, row["measurement"])

    @staticmethod
    def _default_exact_validator(payload: dict[str, Any], prepared: PreparedCase) -> ExactAttempt:
        row = _keys(payload, {"semantic", "measurement", "created_at"}, "exact result")
        seal = _keys(row["semantic"], {"state", "digest"}, "exact semantic")
        if seal["digest"] != stable_hash(seal["state"]):
            raise RuntimeErrorEvidence("exact semantic seal differs")
        state = seal["state"]
        if (state.get("case_id"), state.get("query_id"), state.get("workers")) != (
                prepared.case.case_id, prepared.case.query_id, 1):
            raise RuntimeErrorEvidence("exact result identity differs")
        return ExactAttempt(state, row["measurement"])

    def _paths(self, prefix: str) -> tuple[Path, Path, Path, Path]:
        value = self._sequence; self._sequence += 1
        return tuple(self.scratch / f"{prefix}-{value:04d}-{name}.json"
                     for name in ("request", "ready", "release", "output"))  # type: ignore[return-value]

    @staticmethod
    def _terminate(process: subprocess.Popen[Any]) -> None:
        if process.returncode is None:
            try: process.kill()
            except ProcessLookupError: pass
        if process.returncode is None:
            try:
                waited, status, _ = os.wait4(process.pid, 0)
                if waited: process.returncode = os.waitstatus_to_exitcode(status)
            except ChildProcessError:
                process.poll()

    def _launch(self, *, command: Sequence[str], request_path: Path,
                ready_path: Path, release_path: Path | None, output_path: Path,
                task_id: str, cpus: list[int], workers: int,
                proposal_sha: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        before_swap = _vmstat(); started = perf_counter(); peak = {"VmRSS": 0, "VmHWM": 0, "VmSwap": 0}
        process: subprocess.Popen[Any] | None = None; usage = None
        try:
            # Avoid preexec_fn: fork briefly inherited the growing producer's
            # address space and polluted ru_maxrss before exec.  taskset execs
            # the child under the frozen affinity while leaving Popen eligible
            # for Python's posix_spawn path.
            affinity_command = ["/usr/bin/taskset", "--cpu-list",
                ",".join(str(cpu) for cpu in cpus), *command]
            process = subprocess.Popen(affinity_command, cwd=self.repository,
                env=_child_environment(), stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            deadline = perf_counter() + self.startup_timeout
            while True:
                try:
                    ready_mode = ready_path.lstat().st_mode
                except FileNotFoundError:
                    ready_mode = None
                if ready_mode is not None:
                    if stat.S_ISLNK(ready_mode) or not stat.S_ISREG(ready_mode):
                        raise RuntimeErrorEvidence("child READY is not a regular file")
                    break
                if process.poll() is not None:
                    raise RuntimeErrorEvidence("child exited before READY")
                if perf_counter() > deadline:
                    raise RuntimeErrorEvidence("child READY timed out")
                time.sleep(0.01)
            ready = _read(ready_path)
            expected = {"schema_version", "task_id", "pid", "workers", "cpu_affinity",
                "thread_environment", "resident_lease_digest", "ready_monotonic"}
            if proposal_sha is not None: expected.add("proposal_sha256")
            _keys(ready, expected, "child READY")
            schema = EXACT_READY_SCHEMA if proposal_sha is not None else PROPOSAL_READY_SCHEMA
            if type(ready["schema_version"]) is not str or ready["schema_version"] != schema \
                    or type(ready["task_id"]) is not str or ready["task_id"] != task_id \
                    or type(ready["pid"]) is not int or ready["pid"] != process.pid \
                    or type(ready["workers"]) is not int or ready["workers"] != workers \
                    or type(ready["cpu_affinity"]) is not list \
                    or any(type(cpu) is not int for cpu in ready["cpu_affinity"]) \
                    or ready["cpu_affinity"] != cpus \
                    or type(ready["thread_environment"]) is not dict \
                    or ready["thread_environment"] != _thread_environment() \
                    or ready["resident_lease_digest"] != self.resident_lease_digest \
                    or not _finite_number(ready["ready_monotonic"]) \
                    or (proposal_sha is not None and ready["proposal_sha256"] != proposal_sha):
                raise RuntimeErrorEvidence("child READY binding differs")
            release_evidence = None
            if release_path is not None:
                release_evidence = {"schema_version": PROPOSAL_RELEASE_SCHEMA,
                    "task_id": task_id, "pid": process.pid,
                    "ready_digest": stable_hash(ready), "released_monotonic": time.monotonic()}
                _atomic(release_path, release_evidence)
            deadline = perf_counter() + self.task_timeout
            while True:
                observed = _proc_memory(process.pid)
                for key in peak: peak[key] = max(peak[key], observed[key])
                waited, status, usage = os.wait4(process.pid, os.WNOHANG)
                if waited:
                    process.returncode = os.waitstatus_to_exitcode(status); break
                if perf_counter() > deadline:
                    raise RuntimeErrorEvidence("child task timed out")
                time.sleep(0.01)
            if process.returncode != 0 or usage is None:
                raise RuntimeErrorEvidence(f"child failed rc={process.returncode}")
            payload = _read(output_path)
        except BaseException:
            if process is not None: self._terminate(process)
            raise
        wait4_rss = int(usage.ru_maxrss * (1024 if sys.platform == "darwin" else 1))
        final_swap = (payload.get("measurement", {}).get("resources", {})
                      .get("after", {}).get("swap_kib"))
        authoritative_peak = max(peak["VmRSS"], peak["VmHWM"])
        evidence = {"pid": process.pid, "cpus": cpus, "workers": workers,
            "wall_seconds": perf_counter() - started, "ready_evidence": ready,
            "release_evidence": release_evidence,
            "user_cpu_seconds": float(usage.ru_utime), "system_cpu_seconds": float(usage.ru_stime),
            "minor_faults": int(usage.ru_minflt), "major_faults": int(usage.ru_majflt),
            "peak_rss_kib": peak["VmRSS"], "peak_hwm_kib": peak["VmHWM"],
            "wait4_max_rss_kib": wait4_rss,
            "effective_peak_rss_kib": authoritative_peak,
            "wait4_max_rss_kib_context_only": True,
            "peak_swap_kib": peak["VmSwap"], "final_swap_kib": final_swap,
            "host_vmstat_swap_before": before_swap,
            "host_vmstat_swap_after": _vmstat(), "host_swap_is_context_only": True}
        if peak["VmSwap"] != 0 or (final_swap is not None and (
                type(final_swap) is not int or final_swap != 0)):
            raise RuntimeErrorEvidence("child used process-attributed swap")
        return payload, evidence

    def prepare(self, ordinal: int, task_id: str) -> PreparedCase:
        if type(ordinal) is not int or ordinal not in range(len(self.cases)) \
                or type(task_id) is not str or not task_id:
            raise RuntimeErrorEvidence("proposal case/task differs")
        if self.proposal_command is None:
            raise RuntimeErrorEvidence("proposal command is not configured")
        case = self.cases[ordinal]; request, ready, release, output = self._paths("proposal")
        _atomic(request, {"schema_version": PROPOSAL_REQUEST_SCHEMA, "ordinal": ordinal,
            "task_id": task_id, "case_table": self.case_table,
            "case_table_digest": self.case_table_digest, "ready_path": str(ready),
            "release_path": str(release), "resident_lease_digest": self.resident_lease_digest,
            "resident_snapshot": self.resident_snapshot})
        affinity = sorted(os.sched_getaffinity(0))
        if len(affinity) < PROPOSAL_THREADS:
            raise RuntimeErrorEvidence("proposal runtime requires eight CPUs")
        payload, process = self._launch(command=self.proposal_command(request, output),
            request_path=request, ready_path=ready, release_path=release,
            output_path=output, task_id=task_id, cpus=affinity[:8], workers=8)
        prepared = self.proposal_validator(payload, case, task_id)
        return PreparedCase(prepared.task_id, prepared.case, prepared.semantic,
                            {**prepared.measurement, "spawned_process": process})

    def bind_proposal(self, prepared: PreparedCase, path: Path) -> None:
        if prepared.task_id in self._proposal_files:
            raise RuntimeErrorEvidence("proposal task bound twice")
        _reject_symlink_ancestry(path)
        resolved = path.resolve(strict=True)
        digest, identity = _file_observation(resolved)
        self._proposal_files[prepared.task_id] = (resolved, digest, identity)

    def exact(self, prepared: PreparedCase, workers: int = 1) -> ExactAttempt:
        if workers != 1:
            raise RuntimeErrorEvidence("serial exact runtime freezes workers=1")
        if self.exact_command is None or prepared.task_id not in self._proposal_files:
            raise RuntimeErrorEvidence("exact attempt lacks persisted proposal binding")
        proposal_path, proposal_sha, proposal_identity = self._proposal_files[prepared.task_id]
        if _file_observation(proposal_path) != (proposal_sha, proposal_identity):
            raise RuntimeErrorEvidence("persisted proposal SHA changed")
        request, ready, _release, output = self._paths("exact")
        _atomic(request, {"schema_version": EXACT_REQUEST_SCHEMA,
            "ordinal": prepared.case.ordinal, "task_id": prepared.task_id,
            "case_id": prepared.case.case_id, "query_id": prepared.case.query_id,
            "workers": 1, "case_table": self.case_table,
            "case_table_digest": self.case_table_digest,
            "proposal_path": str(proposal_path), "proposal_sha256": proposal_sha,
            "ready_path": str(ready), "resident_lease_digest": self.resident_lease_digest,
            "resident_snapshot": self.resident_snapshot})
        cpu = [sorted(os.sched_getaffinity(0))[0]]
        payload, process = self._launch(command=self.exact_command(request, output),
            request_path=request, ready_path=ready, release_path=None,
            output_path=output, task_id=prepared.task_id, cpus=cpu, workers=1,
            proposal_sha=proposal_sha)
        if _file_observation(proposal_path) != (proposal_sha, proposal_identity):
            raise RuntimeErrorEvidence("persisted proposal changed during exact task")
        attempt = self.exact_validator(payload, prepared)
        return ExactAttempt(attempt.semantic, {**attempt.measurement, "spawned_process": process})


class ProductionSerialRuntime(SerialSpawnedRuntime):
    """Production command adapter; still has no authority/outcome inputs."""

    def __init__(self, *, repository: Path, cases: Sequence[RuntimeCase],
                 config_path: Path, registry_root: Path, source_full_root: Path,
                 resident_root: Path, resident_snapshot: Mapping[str, Any],
                 generation_id: str, provenance_digest: str, reserve_bytes: int,
                 registry_digest: str, scratch_root: Path | None = None,
                 startup_timeout: float = 600, task_timeout: float = 3600) -> None:
        self.config_path = config_path.resolve(strict=True)
        self.registry_root = registry_root.resolve(strict=True)
        self.source_full_root = source_full_root.resolve(strict=True)
        self.resident_root = resident_root.resolve(strict=True)
        self.resident_snapshot = dict(resident_snapshot)
        if type(generation_id) is not str or not generation_id:
            raise RuntimeErrorEvidence("generation ID differs")
        _digest(provenance_digest, "provenance digest")
        _digest(registry_digest, "registry digest")
        if type(reserve_bytes) is not int or reserve_bytes < 0:
            raise RuntimeErrorEvidence("resident reserve differs")
        self.generation_id = generation_id; self.provenance_digest = provenance_digest
        self.reserve_bytes = reserve_bytes; self.registry_digest = registry_digest
        lease = self.resident_snapshot.get("lease", {}).get("lease_digest")
        base = [sys.executable, str(Path(__file__).resolve()), None,
            "--config", str(self.config_path), "--registry-root", str(self.registry_root),
            "--source-full-root", str(self.source_full_root),
            "--resident-root", str(self.resident_root), "--generation-id", generation_id,
            "--provenance-digest", provenance_digest, "--reserve-bytes", str(reserve_bytes),
            "--registry-digest", registry_digest]
        def command(mode: str, request: Path, output: Path) -> list[str]:
            value = list(base); value[2] = mode
            return [*value, "--request", str(request), "--output", str(output)]
        super().__init__(repository=repository, cases=cases,
            resident_lease_digest=str(lease), resident_snapshot=self.resident_snapshot,
            scratch_root=scratch_root,
            proposal_command=lambda request, output: command("_proposal-child", request, output),
            exact_command=lambda request, output: command("_exact-child", request, output),
            startup_timeout=startup_timeout, task_timeout=task_timeout)

    @staticmethod
    def _bind_final_swap(measurement: Mapping[str, Any]) -> None:
        try:
            value = measurement["resources"]["after"]["swap_kib"]
        except (KeyError, TypeError) as exc:
            raise RuntimeErrorEvidence("production final swap evidence is absent") from exc
        if type(value) is not int or value != 0:
            raise RuntimeErrorEvidence("production child used process-attributed swap")

    def _validate_prepared_leases(self, prepared: PreparedCase) -> None:
        state = prepared.semantic
        leases = state.get("resident_lease_digests")
        if type(leases) is not list or leases != [self.resident_lease_digest] * 4 \
                or state.get("resident_snapshot") != self.resident_snapshot \
                or type(state.get("query_binding")) is not dict \
                or state.get("source_binding_before") != state["query_binding"] \
                or state.get("source_binding_after") != state["query_binding"]:
            raise RuntimeErrorEvidence("production proposal resident/source lease differs")

    def _validate_attempt_leases(
        self, prepared: PreparedCase, attempt: ExactAttempt,
    ) -> None:
        state = attempt.semantic
        if state.get("lease_before") != self.resident_lease_digest \
                or state.get("lease_after") != self.resident_lease_digest \
                or state.get("source_binding_before") != prepared.semantic.get("query_binding") \
                or state.get("source_binding_after") != prepared.semantic.get("query_binding"):
            raise RuntimeErrorEvidence("production exact resident/source lease differs")

    def prepare(self, ordinal: int, task_id: str) -> PreparedCase:
        prepared = super().prepare(ordinal, task_id)
        self._validate_prepared_leases(prepared)
        self._bind_final_swap(prepared.measurement)
        return prepared

    def exact(self, prepared: PreparedCase, workers: int = 1) -> ExactAttempt:
        attempt = super().exact(prepared, workers)
        self._validate_attempt_leases(prepared, attempt)
        self._bind_final_swap(attempt.measurement)
        return attempt

def _module(repository: Path, relative: str, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, repository / relative)
    if spec is None or spec.loader is None:
        raise RuntimeErrorEvidence(f"runtime dependency is absent: {relative}")
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module


def _child_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--source-full-root", type=Path, required=True)
    parser.add_argument("--resident-root", type=Path, required=True)
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--provenance-digest", required=True)
    parser.add_argument("--reserve-bytes", type=int, required=True)
    parser.add_argument("--registry-digest", required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _child_context(args: argparse.Namespace, request: Mapping[str, Any]) -> tuple[Any, Any, tuple[Any, ...], Any]:
    repository = Path(__file__).resolve().parents[2]
    m13 = _module(repository, "experiments/m04r/m04r13_threaded_certified_exposed.py",
                  f"m04r14_serial_m13_{os.getpid()}")
    table = request.get("case_table")
    if type(table) is not list or request.get("case_table_digest") != stable_hash(table):
        raise RuntimeErrorEvidence("child frozen case table differs")
    for row in table:
        _keys(row, {"ordinal", "case_id", "query_id", "registry_case"},
              "child case-table row")
    cases = tuple(m13.CaseInput(row["ordinal"], dict(row["registry_case"])) for row in table)
    runtime_cases = _validate_case_table(tuple(RuntimeCase(row["ordinal"], row["case_id"],
        row["query_id"], dict(row["registry_case"])) for row in table))
    if any((case.case_id, case.query_id) != (runtime.case_id, runtime.query_id)
           for case, runtime in zip(cases, runtime_cases, strict=True)):
        raise RuntimeErrorEvidence("child registry/case identity differs")
    inputs = m13.Inputs(repository, args.config.resolve(), args.registry_root.resolve(),
        (args.source_full_root / "store").resolve(), args.resident_root.resolve(),
        Path("/nonexistent/m04r14-serial-child"), args.generation_id,
        args.provenance_digest, args.reserve_bytes, args.registry_digest, cases,
        stable_hash({"schema_version": "m04r14-serial-runtime-v1"}))
    resident = request.get("resident_snapshot")
    if type(resident) is not dict:
        raise RuntimeErrorEvidence("child resident snapshot is absent")
    m13.validate_resident_snapshot(resident)
    observation = m13.observe_ready_strict(inputs.resident_root / "READY.json")
    content = observation["payload"]["content"]
    if Path(resident["store_root"]).resolve(strict=True) != (
            inputs.resident_root / "store").resolve(strict=True) \
            or content["generation_id"] != inputs.generation_id \
            or content["provenance_digest"] != inputs.provenance_digest:
        raise RuntimeErrorEvidence("child resident root/generation/provenance differs")
    if m13.lease_exact(inputs, resident) != request["resident_lease_digest"]:
        raise RuntimeErrorEvidence("child resident lease differs")
    return m13, inputs, cases, resident


def _proposal_child(argv: Sequence[str]) -> int:
    args = _child_parser().parse_args(argv); request = _read(args.request)
    _keys(request, {"schema_version", "ordinal", "task_id", "case_table",
        "case_table_digest", "ready_path", "release_path", "resident_lease_digest",
        "resident_snapshot"},
        "proposal child request")
    if request["schema_version"] != PROPOSAL_REQUEST_SCHEMA:
        raise RuntimeErrorEvidence("proposal child request schema differs")
    m13, inputs, cases, resident = _child_context(args, request)
    ordinal = request["ordinal"]
    if type(ordinal) is not int or ordinal not in range(len(cases)):
        raise RuntimeErrorEvidence("proposal child ordinal differs")
    ready = {"schema_version": PROPOSAL_READY_SCHEMA, "task_id": request["task_id"],
        "pid": os.getpid(), "workers": 8, "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "thread_environment": {key: os.environ.get(key) for key in THREAD_ENV_KEYS},
        "resident_lease_digest": m13.lease_exact(inputs, resident),
        "ready_monotonic": time.monotonic()}
    _atomic(Path(request["ready_path"]), ready)
    release_path = Path(request["release_path"]); deadline = time.monotonic() + 600
    while not release_path.is_file():
        if time.monotonic() > deadline: raise RuntimeErrorEvidence("proposal release timed out")
        time.sleep(0.01)
    release = _read(release_path)
    released = release.get("released_monotonic")
    if not _finite_number(released) or released < ready["ready_monotonic"] \
            or release != {"schema_version": PROPOSAL_RELEASE_SCHEMA,
            "task_id": request["task_id"], "pid": os.getpid(),
            "ready_digest": stable_hash(ready),
            "released_monotonic": released}:
        raise RuntimeErrorEvidence("proposal release binding differs")
    scheduler = _module(Path(__file__).resolve().parents[2],
        "experiments/m04r/m04r14_exact_scheduler_poc.py", f"m04r14_serial_scheduler_{os.getpid()}")
    prepared = scheduler.ProductionBackend._prepare_local(
        m13, inputs, resident, cases[ordinal], request["task_id"])
    _atomic(args.output, {"semantic": scheduler._seal(prepared.semantic),
        "measurement": prepared.measurement, "created_at": datetime.now(timezone.utc).isoformat()})
    return 0


def _exact_child(argv: Sequence[str]) -> int:
    args = _child_parser().parse_args(argv); request = _read(args.request)
    _keys(request, {"schema_version", "ordinal", "task_id", "case_id", "query_id",
        "workers", "case_table", "case_table_digest", "proposal_path", "proposal_sha256",
        "ready_path", "resident_lease_digest", "resident_snapshot"}, "exact child request")
    if request["schema_version"] != EXACT_REQUEST_SCHEMA or request["workers"] != 1:
        raise RuntimeErrorEvidence("exact child request schema/workers differ")
    m13, inputs, cases, resident = _child_context(args, request)
    ordinal = request["ordinal"]
    if type(ordinal) is not int or ordinal not in range(len(cases)):
        raise RuntimeErrorEvidence("exact child ordinal differs")
    proposal_path = Path(request["proposal_path"])
    proposal_leaf = _read(proposal_path, request["proposal_sha256"])
    seal = _keys(proposal_leaf, {"state", "digest", "measurement", "created_at"},
                 "persisted proposal leaf")
    if seal["digest"] != stable_hash(seal["state"]):
        raise RuntimeErrorEvidence("persisted proposal seal differs")
    semantic = seal["state"]; case = cases[ordinal]
    if (request["case_id"], request["query_id"], semantic.get("case_id"), semantic.get("query_id")) \
            != (case.case_id, case.query_id, case.case_id, case.query_id):
        raise RuntimeErrorEvidence("exact child case identity differs")
    source, episode, search_request, packed = m13._case_context(inputs, case)
    binding = m13.query_binding(source, episode, search_request, packed, inputs.provenance_digest)
    if semantic.get("query_binding") != binding:
        raise RuntimeErrorEvidence("exact child query binding differs")
    forward = m13._proposal_report(semantic["forward"]); m13.validate_proposal(semantic["forward"], packed, inputs)
    ready = {"schema_version": EXACT_READY_SCHEMA, "task_id": request["task_id"],
        "pid": os.getpid(), "workers": 1, "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "thread_environment": {key: os.environ.get(key) for key in THREAD_ENV_KEYS},
        "resident_lease_digest": m13.lease_exact(inputs, resident),
        "ready_monotonic": time.monotonic(), "proposal_sha256": request["proposal_sha256"]}
    _atomic(Path(request["ready_path"]), ready)
    scheduler = _module(Path(__file__).resolve().parents[2],
        "experiments/m04r/m04r14_exact_scheduler_poc.py", f"m04r14_serial_scheduler_{os.getpid()}")
    prepared = scheduler.PreparedCase(request["task_id"], case.case_id, case.query_id,
        semantic, {}, (source, episode, search_request, packed, forward, case))
    backend = object.__new__(scheduler.ProductionBackend)
    backend.module = m13; backend.inputs = inputs; backend.resident = resident
    attempt = backend._exact_local(prepared, 1)
    _atomic(args.output, {"semantic": scheduler._seal(attempt.semantic),
        "measurement": attempt.measurement, "created_at": datetime.now(timezone.utc).isoformat()})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None: argv = sys.argv[1:]
    if argv and argv[0] == "_proposal-child": return _proposal_child(argv[1:])
    if argv and argv[0] == "_exact-child": return _exact_child(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("describe",))
    args = parser.parse_args(argv)
    if args.mode == "describe":
        print(json.dumps({"proposal_threads": 8, "exact_workers": 1,
            "authority_paths_accepted": False, "outcome_paths_accepted": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
