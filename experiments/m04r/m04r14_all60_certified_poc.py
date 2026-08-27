"""Truth-blind, serial T14-03 orchestrator for all 60 exposed cases.

The backend contract is deliberately small.  ``cases`` is the frozen tuple of
``RuntimeCase`` objects; ``source_lock()`` and ``resident_binding()`` return
payloads with the exact contract key sets; ``prepare``, ``bind_proposal`` and
``exact`` implement the serial runtime API; and ``final_source_lease()`` /
``final_resident_lease()`` return strict-JSON lease observations.  No authority
or forward-outcome path is accepted by this module.

The module CLI has two create-only modes, ``preregister`` and ``run``.  The
run mode is intentionally a Linux main-thread entry point because its
``ITIMER_REAL`` deadline covers every synchronous operation through final
lease observation, preterminal reconstruction, and COMPLETE publication.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import importlib.util
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import tempfile
import sys
import threading
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

from experiments.m04r import m04r14_all60_contract as contract


class All60Error(RuntimeError):
    pass


class _RunHardDeadline(TimeoutError):
    pass


def _run_hard_limit_seconds() -> float:
    return float(contract.EXECUTION_POLICY["run_hard_limit_seconds"])


def _arm_run_deadline() -> tuple[Any, tuple[float, float]]:
    """Arm one process wall timer; production execution is main-thread only."""
    if sys.platform != "linux" or threading.current_thread() is not threading.main_thread():
        raise All60Error("hard-deadline execution requires the Linux main thread")
    seconds = _run_hard_limit_seconds()
    if type(seconds) is not float or not math.isfinite(seconds) or seconds <= 0:
        raise All60Error("run hard deadline differs")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    def expired(_signum: int, _frame: Any) -> None:
        raise _RunHardDeadline("run hard limit exceeded")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    return previous_handler, previous_timer


def _disarm_run_deadline() -> None:
    signal.setitimer(signal.ITIMER_REAL, 0.0)


def _restore_run_deadline(previous_handler: Any,
                          previous_timer: tuple[float, float]) -> None:
    _disarm_run_deadline()
    signal.signal(signal.SIGALRM, previous_handler)
    signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _enter_complete_publication(run_started_monotonic: float) -> set[signal.Signals]:
    """Convert the live deadline into one masked terminal commit section."""
    if perf_counter() - run_started_monotonic > contract.EXECUTION_POLICY["run_hard_limit_seconds"]:
        raise _RunHardDeadline("run hard limit exceeded before COMPLETE publication")
    remaining, _interval = signal.getitimer(signal.ITIMER_REAL)
    if remaining <= 0:
        raise _RunHardDeadline("run hard timer expired before COMPLETE publication")
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
    _disarm_run_deadline()
    # Close the check-to-mask race.  A timer that expired just before masking
    # is consumed and treated as failure; it must not survive to post-COMPLETE.
    if signal.SIGALRM in signal.sigpending():
        signal.sigwait({signal.SIGALRM})
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        raise _RunHardDeadline("run hard timer expired before COMPLETE publication")
    return previous_mask


RUNTIME_FIXED_FILES = (
    "config/datasets.example.yaml",
    "experiments/m04r/m04r14_all60_contract.py",
    "experiments/m04r/m04r14_serial_certified_runtime.py",
    "experiments/m04r/m04r14_all60_certified_poc.py",
    "experiments/m04r/verify_m04r14_all60_certified_poc.py",
    "experiments/m04r/m04r14_exact_scheduler_poc.py",
    "experiments/m04r/m04r13_threaded_certified_exposed.py",
    "experiments/m04r/m04r12_quota_ladder_poc.py",
)


def _module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise All60Error(f"dependency absent: {path}")
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module


class ProductionBackendAdapter:
    """Foundation/lease adapter around the authority-free serial runtime."""
    def __init__(self, runtime: Any, *, source_payload: Mapping[str, Any],
                 resident_payload: Mapping[str, Any], final_observer: Callable[[], tuple[dict[str, Any], dict[str, Any]]],
                 certified_validator: Callable[[Any, Any, str, Mapping[str, Any]], None]):
        self.runtime = runtime; self.cases = runtime.cases
        self._source = dict(source_payload); self._resident = dict(resident_payload)
        self._final_observer = final_observer; self._final: tuple[dict[str, Any], dict[str, Any]] | None = None
        self._certified_validator = certified_validator

    @property
    def task_timeout(self) -> float: return self.runtime.task_timeout
    @task_timeout.setter
    def task_timeout(self, value: float) -> None: self.runtime.task_timeout = value
    @property
    def startup_timeout(self) -> float: return self.runtime.startup_timeout
    @startup_timeout.setter
    def startup_timeout(self, value: float) -> None: self.runtime.startup_timeout = value
    def source_lock(self) -> dict[str, Any]: return dict(self._source)
    def resident_binding(self) -> dict[str, Any]: return dict(self._resident)
    def prepare(self, ordinal: int, task_id: str) -> Any: return self.runtime.prepare(ordinal, task_id)
    def bind_proposal(self, prepared: Any, path: Path) -> None: self.runtime.bind_proposal(prepared, path)
    def exact(self, prepared: Any, workers: int = 1) -> Any: return self.runtime.exact(prepared, workers)
    def _observe_final(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._final is None: self._final = self._final_observer()
        return self._final
    def final_source_lease(self) -> dict[str, Any]: return dict(self._observe_final()[0])
    def final_resident_lease(self) -> dict[str, Any]: return dict(self._observe_final()[1])
    def validate_certified(self, certificate: Any, matches: Any, query_id: str,
                           query_binding: Mapping[str, Any]) -> None:
        self._certified_validator(certificate, matches, query_id, query_binding)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite(value: Any) -> bool:
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_finite(item) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _finite(item) for key, item in value.items())
    return value is None or type(value) in {str, int, bool}


def _digest(value: Any) -> str:
    if not _finite(value):
        raise All60Error("evidence is not strict finite JSON")
    try:
        return contract.stable_digest(value)
    except (TypeError, ValueError) as exc:
        raise All60Error("evidence is not strict finite JSON") from exc


def _without(value: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in keys}


def _keys(value: Any, expected: Sequence[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(expected):
        raise All60Error(f"{label} exact keys differ")
    if not _finite(value):
        raise All60Error(f"{label} is not finite strict JSON")
    if "created_at" in expected:
        raw = value.get("created_at")
        try: parsed = datetime.fromisoformat(raw)
        except (TypeError, ValueError) as exc: raise All60Error(f"{label} timestamp differs") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.isoformat() != raw:
            raise All60Error(f"{label} timestamp differs")
    return value


def _strict_read(path: Path, expected_sha: str | None = None) -> dict[str, Any]:
    try:
        initial = path.lstat()
    except OSError as exc:
        raise All60Error(f"cannot stat evidence: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(initial.st_mode):
        raise All60Error("JSON symlink is forbidden")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise All60Error(f"cannot open evidence: {path}") from exc
    try:
        before = os.fstat(descriptor)
        raw = bytearray()
        while block := os.read(descriptor, 1 << 20):
            raw.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns,
                            row.st_ctime_ns, row.st_mode)
    if not stat.S_ISREG(before.st_mode) or identity(before) != identity(after):
        raise All60Error("evidence identity changed")
    observed_sha = sha256(raw).hexdigest()
    if expected_sha is not None and observed_sha != expected_sha:
        raise All60Error("evidence SHA differs")
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in rows:
            if key in result:
                raise All60Error("duplicate JSON key")
            result[key] = item
        return result
    try:
        value = json.loads(bytes(raw), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                All60Error(f"nonfinite token {token}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise All60Error("invalid JSON") from exc
    if type(value) is not dict or not _finite(value):
        raise All60Error("JSON root/finite shape differs")
    return value


def _sha(path: Path) -> str:
    if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
        raise All60Error("SHA target is not a regular file")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor); digest = sha256()
        while block := os.read(descriptor, 1 << 20): digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns,
                            row.st_ctime_ns, row.st_mode)
    if identity(before) != identity(after) or not stat.S_ISREG(before.st_mode):
        raise All60Error("SHA target identity changed")
    return digest.hexdigest()


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise All60Error(f"create-only target exists: {path}")
    _reject_symlink_ancestry(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise All60Error("create-only parent is not a real directory")
    descriptor, temporary = tempfile.mkstemp(prefix=".all60-", dir=path.parent)
    temp = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(contract.canonical_bytes(value) + b"\n")
            handle.flush(); os.fsync(handle.fileno())
        os.link(temp, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temp.unlink(missing_ok=True)


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value); result[field] = _digest(result)
    return result


def build_preregistration(*, runtime_binding: Mapping[str, Any], roots: Mapping[str, Any]) -> dict[str, Any]:
    """Build the deterministic H0 envelope; publication/H1 is a separate step."""
    required_roots = {"candidate", "config", "registry", "source", "resident"}
    if type(roots) is not dict or set(roots) != required_roots \
            or any(type(value) is not str or not value for value in roots.values()):
        raise All60Error("preregistration roots differ")
    runtime = _keys(dict(runtime_binding), ("state", "digest"), "runtime binding")
    state = runtime["state"]
    if type(state) is not dict or runtime["digest"] != _digest(state) \
            or type(state.get("git_head")) is not str \
            or type(state.get("files")) is not dict \
            or type(state.get("environment")) is not dict \
            or state.get("contracts") != {"descriptor_digest": contract.DESCRIPTOR_DIGEST,
                                           "execution_policy": contract.EXECUTION_POLICY}:
        raise All60Error("runtime H0 binding differs")
    if not set(RUNTIME_FIXED_FILES) <= set(state["files"]):
        raise All60Error("runtime fixed-file manifest differs")
    deterministic = {
        "schema_version": contract.SCHEMAS["contract"],
        "status": "frozen_before_run", "runtime_binding": runtime,
        "roots": dict(roots), "query_ids": list(contract.QUERY_IDS),
        "execution_policy": dict(contract.EXECUTION_POLICY),
        "t14_02_binding": dict(contract.T14_02_BINDING),
        "claims": dict(contract.CLAIMS),
    }
    result = {**deterministic, "preregistration_digest": _digest(deterministic)}
    _keys(result, contract.FIELD_KEYS["contract"], "preregistration")
    return result


def production_runtime_binding(repository: Path) -> dict[str, Any]:
    """Capture a clean tracked H0 implementation/environment manifest."""
    repository = repository.resolve(strict=True)
    try:
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repository, text=True, capture_output=True, check=True).stdout.strip()
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository,
            text=True, capture_output=True, check=True).stdout.strip()
        tracked = subprocess.run(["git", "ls-tree", "-r", "--name-only", head],
            cwd=repository, text=True, capture_output=True, check=True).stdout.splitlines()
    except subprocess.CalledProcessError as exc:
        raise All60Error("cannot capture runtime Git state") from exc
    if status: raise All60Error("runtime H0 requires globally clean Git")
    files = sorted({name for name in tracked if name.startswith("src/market_analogues/")
                    and name.endswith(".py")} | set(RUNTIME_FIXED_FILES))
    if any(name not in tracked for name in files): raise All60Error("runtime file is untracked")
    scheduler = _module(repository / "experiments/m04r/m04r14_exact_scheduler_poc.py",
                        "m04r14_all60_environment_capture")
    state = {"git_head": head,
        "files": {name: _sha(repository / name) for name in files},
        "environment": scheduler._environment_binding(),
        "contracts": {"descriptor_digest": contract.DESCRIPTOR_DIGEST,
                      "execution_policy": contract.EXECUTION_POLICY}}
    return {"state": state, "digest": _digest(state)}


def validate_preregistration(value: Mapping[str, Any]) -> dict[str, Any]:
    row = _keys(dict(value), contract.FIELD_KEYS["contract"], "preregistration")
    if row["schema_version"] != contract.SCHEMAS["contract"] \
            or row["status"] != "frozen_before_run" \
            or tuple(row["query_ids"]) != contract.QUERY_IDS \
            or row["execution_policy"] != contract.EXECUTION_POLICY \
            or row["t14_02_binding"] != contract.T14_02_BINDING \
            or row["claims"] != contract.CLAIMS \
            or row["preregistration_digest"] != _digest(_without(row, "preregistration_digest")):
        raise All60Error("preregistration contract differs")
    # Reconstruct the nested H0 seal; H1 ancestry is supplied by the committed
    # launch tool and is intentionally not self-certified inside this payload.
    roots = row["roots"]
    if type(roots) is not dict or set(roots) != {"candidate", "config", "registry", "source", "resident"} \
            or any(type(value) is not str or not value for value in roots.values()):
        raise All60Error("preregistration roots differ")
    runtime = _keys(row["runtime_binding"], ("state", "digest"),
                    "runtime binding")
    state = runtime["state"]
    if runtime["digest"] != _digest(state) or type(state) is not dict \
            or set(state) != {"git_head", "files", "environment", "contracts"} \
            or state["contracts"] != {"descriptor_digest": contract.DESCRIPTOR_DIGEST,
                                      "execution_policy": contract.EXECUTION_POLICY}:
        raise All60Error("runtime binding digest differs")
    if type(state.get("files")) is not dict or not set(RUNTIME_FIXED_FILES) <= set(state["files"]):
        raise All60Error("runtime fixed-file manifest differs")
    return row


def validate_committed_launch(repository: Path, preregistration_path: Path,
                              expected: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the canonical H0 -> preregistration-only H1 launch boundary."""
    repository = repository.resolve(strict=True); path = preregistration_path.resolve(strict=True)
    try: relative = path.relative_to(repository)
    except ValueError as exc: raise All60Error("preregistration is outside repository") from exc
    if relative.as_posix() != contract.PREREGISTRATION_RELATIVE:
        raise All60Error("preregistration path is not the frozen canonical path")
    observed = validate_preregistration(_strict_read(path))
    if observed != validate_preregistration(expected):
        raise All60Error("committed preregistration content differs")
    h0 = observed["runtime_binding"]["state"]["git_head"]
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository,
            text=True, capture_output=True, check=True).stdout.strip()
        parents = subprocess.run(["git", "rev-list", "--parents", "-n", "1", head],
            cwd=repository, text=True, capture_output=True, check=True).stdout.split()
        changed = subprocess.run(["git", "diff-tree", "--no-commit-id", "--name-only", "-r", head],
            cwd=repository, text=True, capture_output=True, check=True).stdout.splitlines()
        committed = subprocess.run(["git", "show", f"{head}:{relative.as_posix()}"],
            cwd=repository, capture_output=True, check=True).stdout
        status = subprocess.run(["git", "status", "--porcelain"], cwd=repository,
            text=True, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as exc:
        raise All60Error("Git launch binding could not be reconstructed") from exc
    if parents != [head, h0] or changed != [relative.as_posix()] \
            or sha256(committed).hexdigest() != _sha(path) or status:
        raise All60Error("launch is not clean preregistration-only H1 over H0")
    _validate_runtime_state(repository, observed, expected_head=head)
    return observed


def _validate_runtime_state(repository: Path, prereg: Mapping[str, Any], *,
                            expected_head: str | None = None) -> None:
    state = prereg["runtime_binding"]["state"]; h0 = state["git_head"]
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository,
            text=True, capture_output=True, check=True).stdout.strip()
        if expected_head is not None and head != expected_head:
            raise All60Error("runtime HEAD changed")
        if subprocess.run(["git", "merge-base", "--is-ancestor", h0, head], cwd=repository).returncode:
            raise All60Error("runtime H0 is not an ancestor")
        files = state["files"]
        if type(files) is not dict or not files or any(type(k) is not str or type(v) is not str for k,v in files.items()):
            raise All60Error("runtime file manifest differs")
        tracked_h0 = subprocess.run(["git", "ls-tree", "-r", "--name-only", h0],
            cwd=repository, text=True, capture_output=True, check=True).stdout.splitlines()
        expected_files = {name for name in tracked_h0 if name.startswith("src/market_analogues/")
                          and name.endswith(".py")} | set(RUNTIME_FIXED_FILES)
        if set(files) != expected_files:
            raise All60Error("runtime file manifest is not the exact H0 closure")
        for name, expected_sha in files.items():
            h0_raw = subprocess.run(["git", "show", f"{h0}:{name}"], cwd=repository,
                capture_output=True, check=True).stdout
            head_raw = subprocess.run(["git", "show", f"{head}:{name}"], cwd=repository,
                capture_output=True, check=True).stdout
            if sha256(h0_raw).hexdigest() != expected_sha \
                    or sha256(head_raw).hexdigest() != expected_sha \
                    or _sha(repository / name) != expected_sha:
                raise All60Error(f"runtime blob drifted: {name}")
        scheduler = _module(repository / "experiments/m04r/m04r14_exact_scheduler_poc.py",
                            "m04r14_all60_environment")
        if state["environment"] != scheduler._environment_binding():
            raise All60Error("runtime environment drifted")
    except subprocess.CalledProcessError as exc:
        raise All60Error("runtime Git state could not be reconstructed") from exc


def _ordered_all60_registry_cases(m13: Any, repository: Path,
                                  registry_root: Path) -> tuple[str, tuple[Any, ...]]:
    """Validate the complete registry in its frozen T14-03 order."""
    registry = m13._read_json(registry_root / "query-registry.json")
    rows = m13._m12(repository)._validate_registry(registry)
    if type(rows) is not list or type(registry) is not dict \
            or type(registry.get("registry_digest")) is not str:
        raise All60Error("full registry shape/digest differs")
    query_ids = [row.get("episode_id") if type(row) is dict else None for row in rows]
    case_ids = [row.get("case_id") if type(row) is dict else None for row in rows]
    if any(type(value) is not str or not value for value in (*query_ids, *case_ids)):
        raise All60Error("full registry ID encoding differs")
    if len(rows) != 60 or len(query_ids) != len(set(query_ids)) \
            or len(case_ids) != len(set(case_ids)):
        raise All60Error("full registry IDs are not exactly 60 unique cases")
    if tuple(query_ids) != contract.QUERY_IDS:
        raise All60Error("full registry execution order differs")
    ordered = tuple(m13.CaseInput(index, dict(row)) for index, row in enumerate(rows))
    return str(registry["registry_digest"]), ordered


def production_backend(repository: Path, preregistration: Mapping[str, Any]) -> ProductionBackendAdapter:
    """Build the real truth-free backend from the exact frozen M13 inputs."""
    prereg = validate_preregistration(preregistration); repository = repository.resolve(strict=True)
    roots = {key: Path(value).resolve(strict=True) for key, value in prereg["roots"].items()
             if key != "candidate"}
    m13 = _module(repository / "experiments/m04r/m04r13_threaded_certified_exposed.py",
                  "m04r14_all60_m13")
    runtime_module = _module(repository / "experiments/m04r/m04r14_serial_certified_runtime.py",
                             "m04r14_all60_runtime")
    expected = {
        "config": (repository / m13.CONFIG_RELATIVE).resolve(),
        "registry": (repository / m13.REGISTRY_RELATIVE).resolve(),
        "source": (repository / m13.SOURCE_FULL_RELATIVE).resolve(),
        "resident": m13.RESIDENT_ROOT.resolve(),
    }
    if roots != expected: raise All60Error("production roots differ from frozen M13 inputs")
    binding = contract.T14_02_BINDING
    artifact_checks = (("candidate_root", "COMPLETE.json", "candidate_complete_sha256"),
        ("candidate_root", "SEMANTICS.json", "semantic_sha256"),
        ("candidate_root", "MEASUREMENTS.json", "measurement_sha256"),
        ("verification_root", "VERIFIED.json", "verification_sha256"))
    for root_key, name, sha_key in artifact_checks:
        path = repository / binding[root_key] / name
        if _sha(path) != binding[sha_key]: raise All60Error("verified T14-02 binding differs")
    registry_digest, m13_cases = _ordered_all60_registry_cases(
        m13, repository, roots["registry"])
    resident = m13.resident_full(roots["source"] / "store", roots["resident"],
        m13.GENERATION_ID, m13.PROVENANCE_DIGEST, m13.RESIDENT_RESERVE_BYTES)
    runtime_cases = tuple(runtime_module.RuntimeCase(index, case.case_id, case.query_id,
        dict(case.registry_case)) for index, case in enumerate(m13_cases))
    inputs = m13.Inputs(repository, roots["config"], roots["registry"],
        roots["source"] / "store", roots["resident"], Path(prereg["roots"]["candidate"]),
        m13.GENERATION_ID, m13.PROVENANCE_DIGEST, m13.RESIDENT_RESERVE_BYTES,
        registry_digest, m13_cases, prereg["preregistration_digest"])
    def bindings() -> list[str]:
        values = []
        for case in m13_cases:
            source, episode, request, packed = m13._case_context(inputs, case)
            values.append(_digest(m13.query_binding(source, episode, request, packed,
                                                    m13.PROVENANCE_DIGEST)))
        return values
    query_digests = bindings()
    source_payload = {"config_sha256": _sha(roots["config"]),
        "registry_sha256": _sha(roots["registry"] / "query-registry.json"),
        "registry_digest": registry_digest, "generation_id": m13.GENERATION_ID,
        "provenance_digest": m13.PROVENANCE_DIGEST,
        "source_tree_digest": resident["content_digest"],
        "query_binding_digests": query_digests}
    runtime = runtime_module.ProductionSerialRuntime(repository=repository, cases=runtime_cases,
        config_path=roots["config"], registry_root=roots["registry"],
        source_full_root=roots["source"], resident_root=roots["resident"],
        resident_snapshot=resident, generation_id=m13.GENERATION_ID,
        provenance_digest=m13.PROVENANCE_DIGEST, reserve_bytes=m13.RESIDENT_RESERVE_BYTES,
        registry_digest=registry_digest)
    def final() -> tuple[dict[str, Any], dict[str, Any]]:
        _validate_runtime_state(repository, prereg)
        current = m13.resident_full(roots["source"] / "store", roots["resident"],
            m13.GENERATION_ID, m13.PROVENANCE_DIGEST, m13.RESIDENT_RESERVE_BYTES)
        current_bindings = bindings()
        if current_bindings != query_digests: raise All60Error("final query/source bindings drifted")
        return ({"source_tree_digest": current["content_digest"],
                 "query_binding_digests": current_bindings},
                {"identity_digest": current["identity_digest"],
                 "lease_digest": current["lease"]["lease_digest"]})
    def validate_certified(certificate: Any, matches: Any, query_id: str,
                           query_binding: Mapping[str, Any]) -> None:
        m13.validate_certificate_and_matches(certificate, matches, query_id,
            expected_input_digest=query_binding.get("certified_input_digest"))
    return ProductionBackendAdapter(runtime, source_payload=source_payload,
        resident_payload=resident, final_observer=final,
        certified_validator=validate_certified)


def _strip_timing(value: Any) -> Any:
    if type(value) is dict:
        return {key: _strip_timing(item) for key, item in value.items()
                if key not in contract.TIMING_FIELDS and key not in {
                    "block_rows", "block_order", "peak_rss_mb"}}
    if type(value) is list:
        return [_strip_timing(item) for item in value]
    return value


def _proposal_leaf(prepared: Any, created_at: str) -> dict[str, Any]:
    value = {"state": prepared.semantic, "digest": _digest(prepared.semantic),
             "measurement": prepared.measurement, "created_at": created_at}
    return _keys(value, contract.FIELD_KEYS["proposal"], "proposal")


def _exact_leaf(ordinal: int, prepared: Any, attempt: Any, proposal_sha: str,
                created_at: str) -> dict[str, Any]:
    semantic_digest = _digest(attempt.semantic)
    measurement_digest = _digest(attempt.measurement)
    result = _seal({"schema_version": contract.SCHEMAS["exact"], "ordinal": ordinal,
        "query_id": prepared.case.query_id, "case_id": prepared.case.case_id,
        "workers": 1, "proposal_sha256": proposal_sha,
        "semantic": attempt.semantic, "measurement": attempt.measurement,
        "semantic_digest": semantic_digest, "measurement_digest": measurement_digest,
        "created_at": created_at}, "result_digest")
    return _keys(result, contract.FIELD_KEYS["exact"], "exact leaf")


def _case_documents(ordinal: int, prepared: Any, attempt: Any, proposal_sha: str,
                    exact_sha: str, started: float, created_at: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    state = prepared.semantic; exact = attempt.semantic
    forward = _strip_timing(state.get("forward")); reverse = _strip_timing(state.get("reverse"))
    parity = forward == reverse
    source_leases = [state.get("source_binding_before"), state.get("source_binding_after"),
                     exact.get("source_binding_before"), exact.get("source_binding_after")]
    resident_leases = list(state.get("resident_lease_digests", [])) + [
        exact.get("lease_before"), exact.get("lease_after")]
    semantic = {"schema_version": contract.SCHEMAS["case_semantic"],
        "ordinal": ordinal, "query_id": prepared.case.query_id,
        "case_id": prepared.case.case_id, "query_binding": state.get("query_binding"),
        "forward_proposal": forward, "reverse_proposal": reverse,
        "proposal_parity": parity, "certificate": exact.get("certificate"),
        "matches": exact.get("matches"), "source_leases": source_leases,
        "resident_leases": resident_leases}
    semantic = _seal(semantic, "semantic_digest")
    _keys(semantic, contract.FIELD_KEYS["case_semantic"], "case semantic")
    pm = prepared.measurement; em = attempt.measurement
    def seconds(value: Any, label: str) -> float:
        if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
            raise All60Error(f"{label} is not finite nonnegative numeric evidence")
        return float(value)
    forward_seconds = seconds(pm.get("forward_seconds"), "forward seconds")
    reverse_seconds = seconds(pm.get("reverse_seconds"), "reverse seconds")
    exact_seconds = seconds(em.get("wall_seconds"), "exact seconds")
    end_seconds = seconds(perf_counter() - started, "case end-to-end seconds")
    proposal_resources = pm.get("resources", {}); exact_resources = em.get("resources", {})
    proposal_after = proposal_resources.get("after", {}) if type(proposal_resources) is dict else {}
    limits = contract.EXECUTION_POLICY["performance_limits"]
    process = pm.get("spawned_process", {})
    process_peak = process.get("effective_peak_rss_kib", 0) if type(process) is dict else 0
    resource_peak = proposal_resources.get("peak_rss_mb", 0) if type(proposal_resources) is dict else 0
    if type(process_peak) not in {int, float} or type(resource_peak) not in {int, float}:
        raise All60Error("proposal RSS is not numeric")
    proposal_peak_mib = max(float(process_peak) / 1024.0, float(resource_peak))
    process_swap = type(process) is dict and type(process.get("peak_swap_kib")) is int \
        and process["peak_swap_kib"] == 0 and type(process.get("final_swap_kib")) is int \
        and process["final_swap_kib"] == 0
    proposal_gate = process_swap and type(proposal_after.get("swap_kib")) is int \
        and proposal_after["swap_kib"] == 0 \
        and forward_seconds <= limits["forward_proposal_seconds"] \
        and reverse_seconds <= limits["reverse_proposal_seconds"] \
        and proposal_peak_mib <= limits["proposal_process_rss_mb"]
    measurement = {"schema_version": contract.SCHEMAS["case_measurement"],
        "ordinal": ordinal, "query_id": prepared.case.query_id,
        "case_id": prepared.case.case_id,
        "child_process": {"proposal": pm.get("spawned_process"), "exact": em.get("spawned_process")},
        "forward_proposal_seconds": forward_seconds,
        "reverse_proposal_seconds": reverse_seconds, "exact_stage_seconds": exact_seconds,
        "end_to_end_seconds": end_seconds,
        "resources": {"proposal": proposal_resources, "exact": exact_resources},
        "proposal_resource_gate_passed": proposal_gate,
        "exact_stage_slo_passed": exact_seconds <= limits["exact_stage_max_seconds"],
        "end_to_end_slo_passed": end_seconds <= limits["end_to_end_max_seconds"],
        "created_at": created_at}
    measurement = _seal(measurement, "measurement_digest")
    _keys(measurement, contract.FIELD_KEYS["case_measurement"], "case measurement")
    bundle = {"schema_version": contract.SCHEMAS["case_bundle"],
        "ordinal": ordinal, "query_id": prepared.case.query_id,
        "case_id": prepared.case.case_id, "proposal_sha256": proposal_sha,
        "exact_sha256": exact_sha, "semantic_digest": semantic["semantic_digest"],
        "measurement_digest": measurement["measurement_digest"],
        "semantic": semantic, "measurement": measurement, "created_at": created_at}
    bundle = _seal(bundle, "bundle_digest")
    return semantic, measurement, bundle


def _event(sequence: int, name: str, ordinal: int, query_id: str,
           previous: str | None, case_sha: str | None, created_at: str) -> dict[str, Any]:
    return _seal({"schema_version": contract.SCHEMAS["event"], "sequence": sequence,
        "event": name, "ordinal": ordinal, "query_id": query_id,
        "previous_event_digest": previous, "case_bundle_sha256": case_sha,
        "created_at": created_at}, "event_digest")


def _manifest(root: Path, paths: Sequence[str]) -> list[dict[str, str]]:
    if len(paths) != len(set(paths)):
        raise All60Error("manifest paths are not unique")
    return [{"path": relative, "sha256": _sha(root / relative)} for relative in sorted(paths)]


def _overlap(left: Path, right: Path) -> bool:
    left = left.absolute(); right = right.absolute()
    return left == right or left in right.parents or right in left.parents


def _reject_symlink_ancestry(path: Path) -> None:
    for item in (path.absolute(), *path.absolute().parents):
        if item.exists() and item.is_symlink(): raise All60Error("symlinked path ancestry is forbidden")


def execute(output_root: Path, preregistration: Mapping[str, Any], backend: Any,
            *, clock: Callable[[], str] = _now) -> dict[str, Any]:
    """Execute once.  Every failure terminalizes the durable prefix; never resume."""
    prereg = validate_preregistration(preregistration)
    root = output_root.absolute()
    if Path(prereg["roots"]["candidate"]).absolute() != root:
        raise All60Error("candidate root differs from frozen preregistration")
    if root.exists() or root.is_symlink():
        raise All60Error("output root exists; resume/relaunch is forbidden")
    _reject_symlink_ancestry(root)
    protected = [Path(prereg["roots"][key]).absolute() for key in ("config", "registry", "source", "resident")]
    if any(_overlap(root, path) for path in protected): raise All60Error("candidate overlaps protected input")
    completed = 0; stage: str | None = None; foundation_count = 0; aggregate_count = 0
    ledger: str | None = None
    run_started_monotonic = perf_counter()
    semantics: list[dict[str, Any]] = []; measurements: list[dict[str, Any]] = []
    previous_handler, previous_timer = _arm_run_deadline()
    complete_signal_mask: set[signal.Signals] | None = None
    try:
        root.mkdir(parents=True); (root / "events").mkdir(); (root / "cases").mkdir()
        if [case.query_id for case in backend.cases] != list(contract.QUERY_IDS) \
                or [case.ordinal for case in backend.cases] != list(range(60)):
            raise All60Error("backend case order differs")
        _atomic(root / "CONTRACT.json", prereg); foundation_count = 1
        run = _seal({"schema_version": contract.SCHEMAS["run"], "status": "running",
            "preregistration_digest": prereg["preregistration_digest"],
            "query_ids": list(contract.QUERY_IDS),
            "execution_policy": dict(contract.EXECUTION_POLICY), "claims": dict(contract.CLAIMS),
            "created_at": clock()}, "result_digest")
        _atomic(root / "RUN_STARTED.json", run); foundation_count = 2
        source = dict(backend.source_lock()); source["schema_version"] = contract.SCHEMAS["source"]
        source["created_at"] = clock(); source = _seal(_without(source, "result_digest"), "result_digest")
        _keys(source, contract.FIELD_KEYS["source"], "source lock")
        if type(source["query_binding_digests"]) is not list or len(source["query_binding_digests"]) != 60:
            raise All60Error("source query-binding table differs")
        _atomic(root / "SOURCE_LOCK.json", source); foundation_count = 3
        resident = dict(backend.resident_binding()); resident["schema_version"] = contract.SCHEMAS["resident"]
        resident["created_at"] = clock(); resident = _seal(_without(resident, "result_digest"), "result_digest")
        _keys(resident, contract.FIELD_KEYS["resident"], "resident binding")
        resident_identity = _without(resident, "schema_version", "created_at",
                                     "result_digest", "identity_digest")
        if type(resident["lease"]) is not dict \
                or resident["lease"].get("lease_digest") != _digest(_without(resident["lease"], "lease_digest")) \
                or resident["identity_digest"] != _digest(resident_identity):
            raise All60Error("resident identity/lease seal differs")
        _atomic(root / "RESIDENT.json", resident); foundation_count = 4
        for ordinal, case in enumerate(backend.cases):
            remaining = contract.EXECUTION_POLICY["run_hard_limit_seconds"] - (
                perf_counter() - run_started_monotonic)
            if remaining <= 0: raise TimeoutError("run hard limit exceeded")
            if hasattr(backend, "task_timeout"):
                backend.task_timeout = min(float(backend.task_timeout), remaining / 2.0)
            if hasattr(backend, "startup_timeout"):
                backend.startup_timeout = min(float(backend.startup_timeout), remaining / 2.0)
            case_started = perf_counter(); task_id = f"all60-{ordinal:03d}-{case.query_id}"
            started_event = _event(ordinal * 2, "CASE_STARTED", ordinal, case.query_id,
                                   ledger, None, clock())
            started_path = root / f"events/{ordinal * 2:03d}-CASE_STARTED-{case.query_id}.json"
            _atomic(started_path, started_event); ledger = started_event["event_digest"]; stage = "started"
            prepared = backend.prepare(ordinal, task_id)
            if prepared.case != case or prepared.task_id != task_id:
                raise All60Error("prepared identity differs")
            case_dir = root / f"cases/{ordinal:03d}-{case.query_id}"
            proposal_path = case_dir / "PROPOSAL.json"
            _atomic(proposal_path, _proposal_leaf(prepared, clock())); stage = "proposal"
            proposal_sha = _sha(proposal_path); backend.bind_proposal(prepared, proposal_path)
            remaining = contract.EXECUTION_POLICY["run_hard_limit_seconds"] - (
                perf_counter() - run_started_monotonic)
            if remaining <= 0: raise TimeoutError("run hard limit exceeded after proposal")
            if hasattr(backend, "task_timeout"):
                backend.task_timeout = min(float(backend.task_timeout), remaining / 2.0)
            if hasattr(backend, "startup_timeout"):
                backend.startup_timeout = min(float(backend.startup_timeout), remaining / 2.0)
            attempt = backend.exact(prepared, workers=1)
            exact_path = case_dir / "EXACT-w1.json"
            exact_leaf = _exact_leaf(ordinal, prepared, attempt, proposal_sha, clock())
            _atomic(exact_path, exact_leaf); stage = "exact"
            semantic, measurement, bundle = _case_documents(
                ordinal, prepared, attempt, proposal_sha, _sha(exact_path), case_started, clock())
            # CASE contains timing-free semantic plus separate measurement so an
            # independent verifier can reconstruct both digests from raw leaves.
            case_leaf = bundle
            _keys(case_leaf, contract.FIELD_KEYS["case_bundle"], "case bundle")
            _atomic(case_dir / "CASE.json", case_leaf); stage = "case"
            case_sha = _sha(case_dir / "CASE.json")
            completed_event = _event(ordinal * 2 + 1, "CASE_COMPLETED", ordinal,
                                     case.query_id, ledger, case_sha, clock())
            _atomic(root / f"events/{ordinal * 2 + 1:03d}-CASE_COMPLETED-{case.query_id}.json",
                    completed_event)
            ledger = completed_event["event_digest"]; completed += 1; stage = None
            semantics.append(semantic); measurements.append(measurement)
        if perf_counter() - run_started_monotonic > contract.EXECUTION_POLICY["run_hard_limit_seconds"]:
            raise TimeoutError("run hard limit exceeded before aggregates")
        semantic_payload = _seal({"schema_version": contract.SCHEMAS["semantics"],
            "status": "complete", "ordered_case_semantic_digests": [x["semantic_digest"] for x in semantics],
            "forward_reverse_equal": all(x["proposal_parity"] is True for x in semantics),
            "all_certified": all(type(x["certificate"]) is dict for x in semantics),
            "all_finite": all(_finite(x) for x in semantics),
            "semantic_passed": all(x["proposal_parity"] is True and type(x["certificate"]) is dict for x in semantics),
            "claims": dict(contract.CLAIMS), "created_at": clock()}, "semantic_digest")
        _atomic(root / "SEMANTICS.json", semantic_payload); aggregate_count = 1
        raw_metrics = [{key: value for key, value in row.items() if key not in {
            "schema_version", "ordinal", "query_id", "case_id", "measurement_digest"}}
            for row in measurements]
        def exact_resource_pass(row: Mapping[str, Any]) -> bool:
            process = row["child_process"]["exact"]; resources = row["resources"]["exact"]
            return type(process) is dict and type(process.get("peak_swap_kib")) is int \
                and process["peak_swap_kib"] == 0 and type(process.get("final_swap_kib")) is int \
                and process["final_swap_kib"] == 0 and type(resources) is dict \
                and type(resources.get("after")) is dict \
                and type(resources["after"].get("swap_kib")) is int \
                and resources["after"]["swap_kib"] == 0
        resource_pass = all(row["proposal_resource_gate_passed"] is True and exact_resource_pass(row)
                            for row in measurements)
        p95_index = contract.EXECUTION_POLICY["performance_limits"]["p95_order_index_zero_based_for_60"]
        nearest_rank_p95 = lambda values: sorted(values)[p95_index]
        exact_p95 = nearest_rank_p95([x["exact_stage_seconds"] for x in measurements])
        end_p95 = nearest_rank_p95([x["end_to_end_seconds"] for x in measurements])
        limits = contract.EXECUTION_POLICY["performance_limits"]
        slo_pass = all(row["exact_stage_slo_passed"] is True and row["end_to_end_slo_passed"] is True for row in measurements) \
            and exact_p95 <= limits["exact_stage_p95_seconds"] \
            and end_p95 <= limits["end_to_end_p95_seconds"]
        measurement_payload = _seal({"schema_version": contract.SCHEMAS["measurements"],
            "status": "complete", "ordered_case_measurement_digests": [x["measurement_digest"] for x in measurements],
            "raw_case_metrics": raw_metrics,
            "summary": {"cases": 60, "total_end_to_end_seconds": sum(x["end_to_end_seconds"] for x in measurements),
                        "p95_method": limits["p95_method"], "exact_stage_seconds_p95": exact_p95,
                        "case_end_to_end_seconds_p95": end_p95},
            "resource_gates": {"all_passed": resource_pass}, "slo_gates": {"all_passed": slo_pass},
            "performance_passed": resource_pass and slo_pass, "created_at": clock()}, "measurement_digest")
        _atomic(root / "MEASUREMENTS.json", measurement_payload); aggregate_count = 2
        precomplete = tuple(path for path in contract.successful_tree() if path != "COMPLETE.json")
        manifest = _manifest(root, precomplete)
        final_source = backend.final_source_lease(); final_resident = backend.final_resident_lease()
        if type(final_source) is not dict \
                or set(final_source) != {"source_tree_digest", "query_binding_digests"} \
                or final_source.get("source_tree_digest") != source["source_tree_digest"] \
                or final_source.get("query_binding_digests") != source["query_binding_digests"] \
                or type(final_resident) is not dict \
                or set(final_resident) != {"identity_digest", "lease_digest"} \
                or final_resident.get("identity_digest") != resident["identity_digest"] \
                or final_resident.get("lease_digest") != resident["lease"].get("lease_digest"):
            raise All60Error("final source/resident lease drifted")
        if perf_counter() - run_started_monotonic > contract.EXECUTION_POLICY["run_hard_limit_seconds"]:
            raise TimeoutError("run hard limit exceeded before COMPLETE")
        complete = _seal({"schema_version": contract.SCHEMAS["complete"], "status": "complete",
            "preregistration_digest": prereg["preregistration_digest"], "ledger_head_digest": ledger,
            "semantic_digest": semantic_payload["semantic_digest"], "measurement_digest": measurement_payload["measurement_digest"],
            "final_source_lease": final_source, "final_resident_lease": final_resident,
            "leaf_manifest": manifest, "leaf_manifest_digest": _digest(manifest),
            "semantic_passed": semantic_payload["semantic_passed"],
            "performance_passed": measurement_payload["performance_passed"],
            "development_only": True, "production_promotion_authorized": False,
            "created_at": clock()}, "complete_digest")
        _keys(complete, contract.FIELD_KEYS["complete"], "complete")
        if not hasattr(backend, "validate_certified"):
            raise All60Error("backend lacks certified semantic validator")
        _validate_evidence(root, prereg, prospective_complete=complete,
                           certified_validator=backend.validate_certified)
        complete_signal_mask = _enter_complete_publication(run_started_monotonic)
        _atomic(root / "COMPLETE.json", complete)
        # COMPLETE is the final fallible filesystem operation.  Independent
        # validation is deliberately not invoked after terminal publication.
        return complete
    except BaseException as exc:
        # The deadline must fire only once.  Failure sealing is intentionally
        # outside the timed region so a valid durable INCOMPLETE can be fsynced.
        _disarm_run_deadline()
        if not (root / "COMPLETE.json").exists() and not (root / "INCOMPLETE.json").exists():
            if isinstance(exc, _RunHardDeadline):
                # An asynchronous timeout may land between a create-only link
                # and the caller's stage update.  Infer and authenticate the
                # actual durable tree rather than trusting in-memory counters.
                if root.is_dir():
                    (root / "events").mkdir(exist_ok=True); (root / "cases").mkdir(exist_ok=True)
                seal_interrupted_prefix(root, prereg,
                    failure_class=type(exc).__name__, failure_message=str(exc),
                    clock=clock, certified_validator=getattr(backend, "validate_certified", None))
            else:
                paths = contract.incomplete_tree(completed, trailing_stage=stage,
                    foundation_count=foundation_count, aggregate_count=aggregate_count)
                partial = _manifest(root, [path for path in paths if path != "INCOMPLETE.json"])
                incomplete = _seal({"schema_version": contract.SCHEMAS["incomplete"], "status": "incomplete",
                    "preregistration_digest": prereg["preregistration_digest"],
                    "failure_class": type(exc).__name__, "failure_message": str(exc),
                    "completed_cases": completed,
                    "trailing_query_id": None if stage is None else contract.QUERY_IDS[completed],
                    "trailing_stage": stage, "ledger_head_digest": ledger,
                    "partial_tree_manifest": partial, "partial_tree_digest": _digest(partial),
                    "resume_authorized": False, "authority_open_authorized": False,
                    "created_at": clock()}, "incomplete_digest")
                _atomic(root / "INCOMPLETE.json", incomplete)
        raise
    finally:
        _restore_run_deadline(previous_handler, previous_timer)
        if complete_signal_mask is not None:
            signal.pthread_sigmask(signal.SIG_SETMASK, complete_signal_mask)


def _tree(root: Path) -> tuple[str, ...]:
    result: list[str] = []
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise All60Error("terminal tree contains forbidden entry")
        if path.is_file(): result.append(path.relative_to(root).as_posix())
    return tuple(sorted(result))


def _directories(root: Path) -> tuple[str, ...]:
    result = []
    for path in root.rglob("*"):
        if path.is_symlink(): raise All60Error("terminal tree contains forbidden symlink")
        if path.is_dir(): result.append(path.relative_to(root).as_posix())
    return tuple(sorted(result))


def seal_interrupted_prefix(root: Path, preregistration: Mapping[str, Any], *,
                            failure_class: str = "InterruptedProcess",
                            failure_message: str = "producer exited before terminal publication",
                            clock: Callable[[], str] = _now,
                            certified_validator: Callable[[Any, Any, str, Mapping[str, Any]], None] | None = None) -> dict[str, Any]:
    """Validate and terminalize one crash prefix without ever continuing it."""
    prereg = validate_preregistration(preregistration)
    if (root / "COMPLETE.json").exists() or (root / "INCOMPLETE.json").exists():
        raise All60Error("root is already terminal")
    observed = _tree(root); candidates: list[tuple[int, str | None, int, int]] = []
    for foundation in range(5):
        for completed in range(61):
            stages = (None,) if completed == 60 else (None, "started", "proposal", "exact", "case")
            for stage in stages:
                for aggregates in range(3):
                    try:
                        expected = tuple(path for path in contract.incomplete_tree(
                            completed, trailing_stage=stage, foundation_count=foundation,
                            aggregate_count=aggregates) if path != "INCOMPLETE.json")
                    except ValueError:
                        continue
                    expected_dirs = contract.incomplete_directories(completed,
                        trailing_stage=stage)
                    if expected == observed and expected_dirs == _directories(root):
                        candidates.append((completed, stage, foundation, aggregates))
    if len(candidates) != 1:
        raise All60Error("interrupted tree is not one strict gap-free prefix")
    completed, stage, foundation, aggregates = candidates[0]
    for relative in observed:
        _strict_read(root / relative)
    ledger = None
    if foundation >= 1 and _strict_read(root / "CONTRACT.json") != prereg:
        raise All60Error("prefix contract differs")
    if foundation >= 2:
        run = _keys(_strict_read(root / "RUN_STARTED.json"), contract.FIELD_KEYS["run"], "prefix run")
        if run["result_digest"] != _digest(_without(run, "result_digest")) \
                or run["preregistration_digest"] != prereg["preregistration_digest"]:
            raise All60Error("prefix run seal differs")
    for index, (name, schema) in enumerate((("SOURCE_LOCK.json", "source"),
                                             ("RESIDENT.json", "resident")), start=3):
        if foundation >= index:
            leaf = _keys(_strict_read(root / name), contract.FIELD_KEYS[schema], f"prefix {schema}")
            if leaf["result_digest"] != _digest(_without(leaf, "result_digest")):
                raise All60Error(f"prefix {schema} seal differs")
    prefix_source = _strict_read(root / "SOURCE_LOCK.json") if foundation == 4 else None
    prefix_resident = _strict_read(root / "RESIDENT.json") if foundation == 4 else None
    if prefix_resident is not None:
        resident_state = _without(prefix_resident, "schema_version", "created_at",
                                  "result_digest", "identity_digest")
        lease = prefix_resident.get("lease")
        if type(lease) is not dict or lease.get("lease_digest") != _digest(_without(lease, "lease_digest")) \
                or prefix_resident.get("identity_digest") != _digest(resident_state):
            raise All60Error("prefix resident identity differs")
    stages = {None: -1, "started": 0, "proposal": 1, "exact": 2, "case": 3}
    semantic_rows: list[dict[str, Any]] = []
    measurement_rows: list[dict[str, Any]] = []
    for ordinal in range(completed + int(stage is not None)):
        query_id = contract.QUERY_IDS[ordinal]; trailing = ordinal == completed
        depth = stages[stage] if trailing else 4
        started_path = root / f"events/{ordinal * 2:03d}-CASE_STARTED-{query_id}.json"
        started = _keys(_strict_read(started_path), contract.FIELD_KEYS["event"], "prefix start")
        if started != _event(ordinal * 2, "CASE_STARTED", ordinal, query_id,
                             ledger, None, started["created_at"]):
            raise All60Error("prefix CASE_STARTED chain differs")
        ledger = started["event_digest"]
        case_dir = root / f"cases/{ordinal:03d}-{query_id}"
        if depth >= 1:
            proposal = _keys(_strict_read(case_dir / "PROPOSAL.json"),
                             contract.FIELD_KEYS["proposal"], "prefix proposal")
            state = proposal["state"]
            if proposal["digest"] != _digest(state) or type(state) is not dict \
                    or set(state) != {"schema_version", "task_id", "case_id", "query_id",
                        "query_binding", "resident_lease_digests", "source_binding_before",
                        "source_binding_after", "resident_snapshot", "forward", "reverse",
                        "semantic_digest"} \
                    or state.get("query_id") != query_id \
                    or (prefix_source is not None and _digest(state.get("query_binding")) !=
                        prefix_source["query_binding_digests"][ordinal]) \
                    or (prefix_resident is not None and state.get("resident_snapshot") !=
                        _without(prefix_resident, "schema_version", "created_at", "result_digest")) \
                    or state.get("semantic_digest") != _digest(_strip_timing(state.get("forward"))) \
                    or _strip_timing(state.get("forward")) != _strip_timing(state.get("reverse")):
                raise All60Error("prefix proposal seal differs")
        if depth >= 2:
            exact = _keys(_strict_read(case_dir / "EXACT-w1.json"),
                          contract.FIELD_KEYS["exact"], "prefix exact")
            if exact["proposal_sha256"] != _sha(case_dir / "PROPOSAL.json") \
                    or exact["result_digest"] != _digest(_without(exact, "result_digest")) \
                    or exact["semantic_digest"] != _digest(exact["semantic"]) \
                    or exact["measurement_digest"] != _digest(exact["measurement"]) \
                    or exact.get("query_id") != query_id or exact.get("case_id") != state.get("case_id"):
                raise All60Error("prefix exact seal differs")
        if depth >= 3:
            case = _keys(_strict_read(case_dir / "CASE.json"),
                         contract.FIELD_KEYS["case_bundle"], "prefix case")
            if case["proposal_sha256"] != _sha(case_dir / "PROPOSAL.json") \
                    or case["exact_sha256"] != _sha(case_dir / "EXACT-w1.json") \
                    or case["bundle_digest"] != _digest(_without(case, "bundle_digest")) \
                    or case.get("query_id") != query_id or case.get("case_id") != state.get("case_id") \
                    or case.get("semantic_digest") != case.get("semantic", {}).get("semantic_digest") \
                    or case.get("measurement_digest") != case.get("measurement", {}).get("measurement_digest"):
                raise All60Error("prefix case seal differs")
            semantic = case["semantic"]; measurement = case["measurement"]
            exact_semantic = exact["semantic"]
            if semantic.get("semantic_digest") != _digest(_without(semantic, "semantic_digest")) \
                    or measurement.get("measurement_digest") != _digest(_without(measurement, "measurement_digest")) \
                    or semantic.get("certificate") != exact_semantic.get("certificate") \
                    or semantic.get("matches") != exact_semantic.get("matches") \
                    or measurement.get("resources") != {"proposal": proposal["measurement"].get("resources"),
                                                        "exact": exact["measurement"].get("resources")} \
                    or measurement.get("child_process") != {"proposal": proposal["measurement"].get("spawned_process"),
                                                            "exact": exact["measurement"].get("spawned_process")}:
                raise All60Error("prefix case reconstruction differs")
            if certified_validator is None:
                repository = Path(__file__).resolve().parents[2]
                module = _module(repository / "experiments/m04r/m04r13_threaded_certified_exposed.py",
                                 "m04r14_all60_prefix_m13")
                certified_validator = lambda certificate, matches, identity, binding: (
                    module.validate_certificate_and_matches(certificate, matches, identity,
                        expected_input_digest=binding.get("certified_input_digest")))
            try:
                certified_validator(exact["semantic"].get("certificate"),
                    exact["semantic"].get("matches"), query_id, state.get("query_binding"))
            except Exception as exc:
                raise All60Error("prefix certified result differs") from exc
            semantic_rows.append(semantic); measurement_rows.append(measurement)
        if not trailing:
            completed_path = root / f"events/{ordinal * 2 + 1:03d}-CASE_COMPLETED-{query_id}.json"
            event = _keys(_strict_read(completed_path), contract.FIELD_KEYS["event"], "prefix completed")
            if event != _event(ordinal * 2 + 1, "CASE_COMPLETED", ordinal, query_id,
                               ledger, _sha(case_dir / "CASE.json"), event["created_at"]):
                raise All60Error("prefix CASE_COMPLETED chain differs")
            ledger = event["event_digest"]
    if aggregates:
        semantics = _keys(_strict_read(root / "SEMANTICS.json"), contract.FIELD_KEYS["semantics"], "prefix semantics")
        semantic_digests = [row["semantic_digest"] for row in semantic_rows]
        semantic_flags = {
            "forward_reverse_equal": all(row["proposal_parity"] is True for row in semantic_rows),
            "all_certified": all(type(row["certificate"]) is dict for row in semantic_rows),
            "all_finite": all(_finite(row) for row in semantic_rows),
        }
        semantic_passed = all(semantic_flags.values())
        if semantics["schema_version"] != contract.SCHEMAS["semantics"] \
                or semantics["status"] != "complete" \
                or semantics["semantic_digest"] != _digest(_without(semantics, "semantic_digest")) \
                or semantics["ordered_case_semantic_digests"] != semantic_digests \
                or semantics["claims"] != contract.CLAIMS \
                or any(semantics[key] != value for key, value in semantic_flags.items()) \
                or semantics["semantic_passed"] != semantic_passed:
            raise All60Error("prefix semantics seal differs")
    if aggregates == 2:
        measurements = _keys(_strict_read(root / "MEASUREMENTS.json"), contract.FIELD_KEYS["measurements"], "prefix measurements")
        measurement_digests = [row["measurement_digest"] for row in measurement_rows]
        raw_metrics = [{key: value for key, value in row.items() if key not in {
            "schema_version", "ordinal", "query_id", "case_id", "measurement_digest"}}
            for row in measurement_rows]
        exact_values = [row["exact_stage_seconds"] for row in measurement_rows]
        end_values = [row["end_to_end_seconds"] for row in measurement_rows]
        limits = contract.EXECUTION_POLICY["performance_limits"]
        p95_index = limits["p95_order_index_zero_based_for_60"]
        exact_p95 = sorted(exact_values)[p95_index]; end_p95 = sorted(end_values)[p95_index]
        summary = {"cases": 60, "total_end_to_end_seconds": sum(end_values),
            "p95_method": limits["p95_method"], "exact_stage_seconds_p95": exact_p95,
            "case_end_to_end_seconds_p95": end_p95}
        def exact_resource_pass(row: Mapping[str, Any]) -> bool:
            process = row["child_process"]["exact"]; resources = row["resources"]["exact"]
            return type(process) is dict and type(process.get("peak_swap_kib")) is int \
                and process["peak_swap_kib"] == 0 and type(process.get("final_swap_kib")) is int \
                and process["final_swap_kib"] == 0 and type(resources) is dict \
                and type(resources.get("after")) is dict and resources["after"].get("swap_kib") == 0
        resource_pass = all(row["proposal_resource_gate_passed"] is True
                            and exact_resource_pass(row) for row in measurement_rows)
        slo_pass = all(row["exact_stage_slo_passed"] is True
                       and row["end_to_end_slo_passed"] is True for row in measurement_rows) \
            and exact_p95 <= limits["exact_stage_p95_seconds"] \
            and end_p95 <= limits["end_to_end_p95_seconds"]
        if measurements["schema_version"] != contract.SCHEMAS["measurements"] \
                or measurements["status"] != "complete" \
                or measurements["measurement_digest"] != _digest(_without(measurements, "measurement_digest")) \
                or measurements["ordered_case_measurement_digests"] != measurement_digests \
                or measurements["raw_case_metrics"] != raw_metrics \
                or measurements["summary"] != summary \
                or measurements["resource_gates"] != {"all_passed": resource_pass} \
                or measurements["slo_gates"] != {"all_passed": slo_pass} \
                or measurements["performance_passed"] != (resource_pass and slo_pass):
            raise All60Error("prefix measurements seal differs")
    manifest = _manifest(root, observed)
    value = _seal({"schema_version": contract.SCHEMAS["incomplete"], "status": "incomplete",
        "preregistration_digest": prereg["preregistration_digest"],
        "failure_class": failure_class, "failure_message": failure_message,
        "completed_cases": completed,
        "trailing_query_id": None if stage is None else contract.QUERY_IDS[completed],
        "trailing_stage": stage, "ledger_head_digest": ledger,
        "partial_tree_manifest": manifest, "partial_tree_digest": _digest(manifest),
        "resume_authorized": False, "authority_open_authorized": False,
        "created_at": clock()}, "incomplete_digest")
    _atomic(root / "INCOMPLETE.json", value)
    return value


def _validate_evidence(root: Path, preregistration: Mapping[str, Any], *,
                       prospective_complete: Mapping[str, Any] | None = None,
                       certified_validator: Callable[[Any, Any, str, Mapping[str, Any]], None] | None = None) -> dict[str, Any]:
    prereg = validate_preregistration(preregistration)
    if (root / "INCOMPLETE.json").exists():
        raise All60Error("run is terminal incomplete")
    expected_tree = contract.successful_tree() if prospective_complete is None else tuple(
        path for path in contract.successful_tree() if path != "COMPLETE.json")
    if _tree(root) != expected_tree:
        raise All60Error("successful tree differs")
    if _directories(root) != contract.successful_directories():
        raise All60Error("successful directory inventory differs")
    # Reconstruct foundation seals and frozen identities.
    observed_contract = _strict_read(root / "CONTRACT.json")
    if observed_contract != prereg: raise All60Error("persisted contract differs")
    run = _keys(_strict_read(root / "RUN_STARTED.json"), contract.FIELD_KEYS["run"], "run")
    if run["schema_version"] != contract.SCHEMAS["run"] or run["status"] != "running" \
            or run["preregistration_digest"] != prereg["preregistration_digest"] \
            or run["result_digest"] != _digest(_without(run, "result_digest")) \
            or run["query_ids"] != list(contract.QUERY_IDS) \
            or run["execution_policy"] != contract.EXECUTION_POLICY \
            or run["claims"] != contract.CLAIMS:
        raise All60Error("run seal/policy differs")
    foundations: dict[str, dict[str, Any]] = {}
    for name, schema in (("SOURCE_LOCK.json", "source"), ("RESIDENT.json", "resident")):
        leaf = _keys(_strict_read(root / name), contract.FIELD_KEYS[schema], schema)
        if leaf["schema_version"] != contract.SCHEMAS[schema] \
                or leaf["result_digest"] != _digest(_without(leaf, "result_digest")):
            raise All60Error(f"{schema} seal differs")
        foundations[schema] = leaf
    source_foundation = foundations["source"]; resident_foundation = foundations["resident"]
    if type(source_foundation["query_binding_digests"]) is not list \
            or len(source_foundation["query_binding_digests"]) != 60 \
            or any(type(value) is not str or len(value) != 64
                   for value in source_foundation["query_binding_digests"]):
        raise All60Error("source query-binding table differs")
    resident_state = _without(resident_foundation, "schema_version", "created_at",
                              "result_digest", "identity_digest")
    lease = resident_foundation["lease"]
    lease_keys = {"schema_version", "ready_digest", "ready_file_sha256", "content_digest",
                  "files", "lease_digest"}
    identity_keys = {"path", "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode"}
    files = lease.get("files") if type(lease) is dict else None
    if type(lease) is not dict or set(lease) != lease_keys \
            or lease.get("ready_digest") != resident_foundation["ready_digest"] \
            or lease.get("ready_file_sha256") != resident_foundation["ready_file_sha256"] \
            or lease.get("content_digest") != resident_foundation["content_digest"] \
            or type(files) is not dict or not files \
            or any(type(identity) is not dict or set(identity) != identity_keys \
                   or type(identity["path"]) is not str \
                   or any(type(identity[key]) is not int or identity[key] < 0
                          for key in identity_keys - {"path"}) for identity in files.values()) \
            or lease.get("lease_digest") != _digest(_without(lease, "lease_digest")) \
            or resident_foundation["identity_digest"] != _digest(resident_state):
        raise All60Error("resident identity/lease seal differs")
    ledger = None; semantic_digests = []; measurement_digests = []
    semantic_rows: list[dict[str, Any]] = []; measurement_rows: list[dict[str, Any]] = []
    for ordinal, query_id in enumerate(contract.QUERY_IDS):
        directory = root / f"cases/{ordinal:03d}-{query_id}"
        proposal_path = directory / "PROPOSAL.json"; exact_path = directory / "EXACT-w1.json"
        case_path = directory / "CASE.json"
        started = _keys(_strict_read(root / f"events/{ordinal * 2:03d}-CASE_STARTED-{query_id}.json"),
                        contract.FIELD_KEYS["event"], "started event")
        if started != _event(ordinal * 2, "CASE_STARTED", ordinal, query_id,
                             ledger, None, started["created_at"]):
            raise All60Error("CASE_STARTED chain differs")
        ledger = started["event_digest"]
        proposal = _keys(_strict_read(proposal_path), contract.FIELD_KEYS["proposal"], "proposal")
        if proposal["digest"] != _digest(proposal["state"]):
            raise All60Error("proposal seal differs")
        state = proposal["state"]
        proposal_state_keys = {"schema_version", "task_id", "case_id", "query_id",
            "query_binding", "resident_lease_digests", "source_binding_before",
            "source_binding_after", "resident_snapshot", "forward", "reverse",
            "semantic_digest"}
        if type(state) is not dict or set(state) != proposal_state_keys \
                or state.get("schema_version") != "m04r14-exact-scheduler-proposal-v1" \
                or type(state.get("case_id")) is not str or state.get("query_id") != query_id \
                or state.get("source_binding_before") != state.get("query_binding") \
                or state.get("source_binding_after") != state.get("query_binding") \
                or _digest(state.get("query_binding")) != foundations["source"]["query_binding_digests"][ordinal] \
                or state.get("resident_snapshot") != _without(foundations["resident"],
                    "schema_version", "created_at", "result_digest") \
                or type(state.get("resident_lease_digests")) is not list \
                or len(state["resident_lease_digests"]) != 4 \
                or len(set(state["resident_lease_digests"])) != 1 \
                or state["resident_lease_digests"][0] != foundations["resident"]["lease"].get("lease_digest") \
                or state.get("semantic_digest") != _digest(_strip_timing(state.get("forward"))) \
                or _strip_timing(state.get("forward")) != _strip_timing(state.get("reverse")) \
                or state["forward"].get("block_rows") != 4096 \
                or state["forward"].get("block_order") != "forward" \
                or state["reverse"].get("block_rows") != 4097 \
                or state["reverse"].get("block_order") != "reverse":
            raise All60Error("proposal identity differs")
        exact = _keys(_strict_read(exact_path), contract.FIELD_KEYS["exact"], "exact")
        if exact["schema_version"] != contract.SCHEMAS["exact"] \
                or exact["ordinal"] != ordinal or exact["query_id"] != query_id \
                or exact["case_id"] != state.get("case_id") or exact["workers"] != 1 \
                or exact["proposal_sha256"] != _sha(proposal_path) \
                or exact["semantic_digest"] != _digest(exact["semantic"]) \
                or exact["measurement_digest"] != _digest(exact["measurement"]) \
                or exact["result_digest"] != _digest(_without(exact, "result_digest")):
            raise All60Error("exact leaf reconstruction differs")
        case = _keys(_strict_read(case_path), contract.FIELD_KEYS["case_bundle"], "case")
        semantic = _keys(case["semantic"], contract.FIELD_KEYS["case_semantic"], "case semantic")
        measurement = _keys(case["measurement"], contract.FIELD_KEYS["case_measurement"], "case measurement")
        proposal_measurement = proposal["measurement"]; exact_measurement = exact["measurement"]
        expected_sources = [state.get("source_binding_before"), state.get("source_binding_after"),
            exact["semantic"].get("source_binding_before"), exact["semantic"].get("source_binding_after")]
        expected_resident = list(state.get("resident_lease_digests", [])) + [
            exact["semantic"].get("lease_before"), exact["semantic"].get("lease_after")]
        exact_semantic = exact["semantic"]
        if certified_validator is None:
            repository = Path(__file__).resolve().parents[2]
            m13_validator = _module(repository / "experiments/m04r/m04r13_threaded_certified_exposed.py",
                                    "m04r14_all60_terminal_m13")
            certified_validator = lambda certificate, matches, identity, binding: (
                m13_validator.validate_certificate_and_matches(certificate, matches, identity,
                    expected_input_digest=binding.get("certified_input_digest")))
        try:
            certified_validator(exact_semantic.get("certificate"), exact_semantic.get("matches"),
                                query_id, state.get("query_binding"))
        except Exception as exc:
            raise All60Error("certified result reconstruction differs") from exc
        if semantic["semantic_digest"] != _digest(_without(semantic, "semantic_digest")) \
                or semantic["schema_version"] != contract.SCHEMAS["case_semantic"] \
                or measurement["schema_version"] != contract.SCHEMAS["case_measurement"] \
                or case["schema_version"] != contract.SCHEMAS["case_bundle"] \
                or measurement["measurement_digest"] != _digest(_without(measurement, "measurement_digest")) \
                or case["bundle_digest"] != _digest(_without(case, "bundle_digest")) \
                or case["proposal_sha256"] != _sha(proposal_path) \
                or case["exact_sha256"] != _sha(exact_path) \
                or case["semantic_digest"] != semantic["semantic_digest"] \
                or case["measurement_digest"] != measurement["measurement_digest"] \
                or (case["ordinal"], case["query_id"], case["case_id"]) != (ordinal, query_id, state.get("case_id")) \
                or semantic["query_binding"] != state.get("query_binding") \
                or semantic["forward_proposal"] != _strip_timing(state.get("forward")) \
                or semantic["reverse_proposal"] != _strip_timing(state.get("reverse")) \
                or semantic["proposal_parity"] != (semantic["forward_proposal"] == semantic["reverse_proposal"]) \
                or semantic["certificate"] != exact["semantic"].get("certificate") \
                or semantic["matches"] != exact_semantic.get("matches") \
                or semantic["source_leases"] != expected_sources \
                or semantic["resident_leases"] != expected_resident \
                or type(exact_semantic) is not dict \
                or set(exact_semantic) != {"schema_version", "case_id", "query_id", "workers",
                    "proposal_semantic_digest", "certificate", "matches",
                    "certificate_result_digest", "match_digest", "lease_before", "lease_after",
                    "source_binding_before", "source_binding_after"} \
                or exact_semantic.get("schema_version") != "m04r14-exact-scheduler-attempt-v1" \
                or exact_semantic.get("workers") != 1 \
                or exact_semantic.get("proposal_semantic_digest") != state.get("semantic_digest") \
                or type(exact_semantic.get("certificate")) is not dict \
                or exact_semantic["certificate"].get("result_digest") != exact_semantic.get("certificate_result_digest") \
                or type(exact_semantic.get("matches")) is not list \
                or exact_semantic.get("match_digest") != _digest(exact_semantic["matches"]) \
                or measurement["forward_proposal_seconds"] != proposal_measurement.get("forward_seconds") \
                or measurement["reverse_proposal_seconds"] != proposal_measurement.get("reverse_seconds") \
                or measurement["exact_stage_seconds"] != exact_measurement.get("wall_seconds") \
                or measurement["resources"] != {"proposal": proposal_measurement.get("resources"),
                                                "exact": exact_measurement.get("resources")} \
                or measurement["child_process"] != {"proposal": proposal_measurement.get("spawned_process"),
                                                    "exact": exact_measurement.get("spawned_process")}:
            raise All60Error("case reconstruction differs")
        completed = _keys(_strict_read(root / f"events/{ordinal * 2 + 1:03d}-CASE_COMPLETED-{query_id}.json"),
                          contract.FIELD_KEYS["event"], "completed event")
        if completed != _event(ordinal * 2 + 1, "CASE_COMPLETED", ordinal, query_id,
                               ledger, _sha(case_path), completed["created_at"]):
            raise All60Error("CASE_COMPLETED chain differs")
        ledger = completed["event_digest"]
        semantic_digests.append(semantic["semantic_digest"])
        measurement_digests.append(measurement["measurement_digest"])
        semantic_rows.append(semantic); measurement_rows.append(measurement)
    semantics = _keys(_strict_read(root / "SEMANTICS.json"), contract.FIELD_KEYS["semantics"], "semantics")
    measurements = _keys(_strict_read(root / "MEASUREMENTS.json"), contract.FIELD_KEYS["measurements"], "measurements")
    expected_semantic_flags = {
        "forward_reverse_equal": all(x["proposal_parity"] is True for x in semantic_rows),
        "all_certified": all(type(x["certificate"]) is dict for x in semantic_rows),
        "all_finite": all(_finite(x) for x in semantic_rows),
    }
    expected_semantic_pass = all(expected_semantic_flags.values())
    raw_metrics = [{key: value for key, value in row.items() if key not in {
        "schema_version", "ordinal", "query_id", "case_id", "measurement_digest"}}
        for row in measurement_rows]
    exact_values = [x["exact_stage_seconds"] for x in measurement_rows]
    end_values = [x["end_to_end_seconds"] for x in measurement_rows]
    limits = contract.EXECUTION_POLICY["performance_limits"]
    p95_index = limits["p95_order_index_zero_based_for_60"]
    exact_p95 = sorted(exact_values)[p95_index]; end_p95 = sorted(end_values)[p95_index]
    expected_summary = {"cases": 60, "total_end_to_end_seconds": sum(end_values),
        "p95_method": limits["p95_method"], "exact_stage_seconds_p95": exact_p95,
        "case_end_to_end_seconds_p95": end_p95}
    def exact_resource_pass(row: Mapping[str, Any]) -> bool:
        process = row["child_process"]["exact"]; resources = row["resources"]["exact"]
        return type(process) is dict and type(process.get("peak_swap_kib")) is int \
            and process["peak_swap_kib"] == 0 and type(process.get("final_swap_kib")) is int \
            and process["final_swap_kib"] == 0 and type(resources) is dict \
            and type(resources.get("after")) is dict and resources["after"].get("swap_kib") == 0
    resource_pass = all(x["proposal_resource_gate_passed"] is True and exact_resource_pass(x)
                        for x in measurement_rows)
    slo_pass = all(x["exact_stage_slo_passed"] is True and x["end_to_end_slo_passed"] is True
                   for x in measurement_rows) and exact_p95 <= limits["exact_stage_p95_seconds"] \
                   and end_p95 <= limits["end_to_end_p95_seconds"]
    if semantics["schema_version"] != contract.SCHEMAS["semantics"] or semantics["status"] != "complete" \
            or measurements["schema_version"] != contract.SCHEMAS["measurements"] or measurements["status"] != "complete" \
            or semantics["semantic_digest"] != _digest(_without(semantics, "semantic_digest")) \
            or semantics["ordered_case_semantic_digests"] != semantic_digests \
            or semantics["claims"] != contract.CLAIMS \
            or any(semantics[key] != value for key, value in expected_semantic_flags.items()) \
            or semantics["semantic_passed"] != expected_semantic_pass \
            or measurements["measurement_digest"] != _digest(_without(measurements, "measurement_digest")) \
            or measurements["ordered_case_measurement_digests"] != measurement_digests \
            or measurements["raw_case_metrics"] != raw_metrics \
            or measurements["summary"] != expected_summary \
            or measurements["resource_gates"] != {"all_passed": resource_pass} \
            or measurements["slo_gates"] != {"all_passed": slo_pass} \
            or measurements["performance_passed"] != (resource_pass and slo_pass):
        raise All60Error("aggregate reconstruction differs")
    complete = (_strict_read(root / "COMPLETE.json") if prospective_complete is None
                else dict(prospective_complete))
    _keys(complete, contract.FIELD_KEYS["complete"], "complete")
    if complete["schema_version"] != contract.SCHEMAS["complete"] \
            or complete["status"] != "complete" \
            or complete["development_only"] is not True \
            or complete["production_promotion_authorized"] is not False \
            or complete["complete_digest"] != _digest(_without(complete, "complete_digest")) \
            or complete["preregistration_digest"] != prereg["preregistration_digest"] \
            or complete["ledger_head_digest"] != ledger \
            or complete["semantic_digest"] != semantics["semantic_digest"] \
            or complete["measurement_digest"] != measurements["measurement_digest"] \
            or complete["semantic_passed"] != semantics["semantic_passed"] \
            or complete["performance_passed"] != measurements["performance_passed"] \
            or type(complete["final_source_lease"]) is not dict \
            or set(complete["final_source_lease"]) != {"source_tree_digest", "query_binding_digests"} \
            or complete["final_source_lease"].get("source_tree_digest") != foundations["source"]["source_tree_digest"] \
            or complete["final_source_lease"].get("query_binding_digests") != foundations["source"]["query_binding_digests"] \
            or type(complete["final_resident_lease"]) is not dict \
            or set(complete["final_resident_lease"]) != {"identity_digest", "lease_digest"} \
            or complete["final_resident_lease"].get("identity_digest") != foundations["resident"]["identity_digest"] \
            or complete["final_resident_lease"].get("lease_digest") != foundations["resident"]["lease"].get("lease_digest"):
        raise All60Error("complete seal differs")
    expected_paths = tuple(path for path in contract.successful_tree() if path != "COMPLETE.json")
    if type(complete["leaf_manifest"]) is not list \
            or [row.get("path") for row in complete["leaf_manifest"]] != list(sorted(expected_paths)):
        raise All60Error("complete manifest path set/order differs")
    manifest = _manifest(root, expected_paths)
    if manifest != complete["leaf_manifest"] or _digest(manifest) != complete["leaf_manifest_digest"]:
        raise All60Error("complete leaf manifest differs")
    return complete


def validate_terminal(root: Path, preregistration: Mapping[str, Any], *,
                      certified_validator: Callable[[Any, Any, str, Mapping[str, Any]], None] | None = None) -> dict[str, Any]:
    return _validate_evidence(root, preregistration, certified_validator=certified_validator)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preregister", "run"))
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path)
    args = parser.parse_args(argv)
    repository = args.repository.resolve(strict=True)
    canonical = repository / contract.PREREGISTRATION_RELATIVE
    prereg_path = canonical if args.preregistration is None else args.preregistration.resolve()
    if prereg_path != canonical: raise All60Error("preregistration path differs")
    if args.mode == "preregister":
        m13 = _module(repository / "experiments/m04r/m04r13_threaded_certified_exposed.py",
                      "m04r14_all60_prereg_m13")
        roots = {"candidate": str((repository / contract.CANDIDATE_RELATIVE).resolve()),
            "config": str((repository / m13.CONFIG_RELATIVE).resolve()),
            "registry": str((repository / m13.REGISTRY_RELATIVE).resolve()),
            "source": str((repository / m13.SOURCE_FULL_RELATIVE).resolve()),
            "resident": str(m13.RESIDENT_ROOT.resolve())}
        value = build_preregistration(runtime_binding=production_runtime_binding(repository),
                                      roots=roots)
        _atomic(prereg_path, value)
        return 0
    prereg = _strict_read(prereg_path)
    validate_committed_launch(repository, prereg_path, prereg)
    candidate = Path(prereg["roots"]["candidate"])
    backend = production_backend(repository, prereg)
    execute(candidate, prereg, backend)
    return 0


__all__ = ("All60Error", "ProductionBackendAdapter", "build_preregistration", "execute",
           "main", "production_backend", "production_runtime_binding",
           "seal_interrupted_prefix", "validate_committed_launch",
           "validate_preregistration", "validate_terminal")


if __name__ == "__main__":
    raise SystemExit(main())
