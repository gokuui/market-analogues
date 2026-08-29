"""Independent verifier for the truth-blind T14-03 exposed-60 evidence.

Only the machine-readable compatibility contract is shared with the producer.
This module deliberately does not import the producer or serial child runtime.
It validates a terminal tree read-only and writes, when requested, one receipt
to a fresh canonical verifier root.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from market_analogues.adapters import source_from_spec
from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.certified_packed_search import certified_packed_search_contract
from market_analogues.config import load_config
from market_analogues.episodes import build_episode
from market_analogues.packed_bound_search import (
    BoundProposal, PackedBoundQuery, _packed_query_input_digest,
    bound_proposal_candidate_digest, packed_bound_search_contract,
    scan_packed_bound_proposals, scan_packed_bound_threshold,
)
from market_analogues.packed_bound_store import (
    TIER_NAMES, decode_episode_id, load_packed_generation,
    packed_branch_aware_lower_bounds,
)
from market_analogues.representation import represent, representation_input_digest
from market_analogues.resident_store import observe_ready_strict, resident_file_identity_lease
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import InstrumentKey, SearchQuery, stable_hash


def _load_contract():
    path = Path(__file__).with_name("m04r14_all60_contract.py")
    spec = importlib.util.spec_from_file_location("m04r14_all60_contract_for_verifier", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("T14-03 contract cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module


contract = _load_contract()

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
THREAD_ENV_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    "NUMBA_NUM_THREADS", "NUMBA_THREADING_LAYER")
FROZEN_REGISTRY_DIGEST = "0a4da732f91375a091775cb04e6e77c8d136ade47d7f4d16508a2d9a6555361e"


class VerificationError(RuntimeError):
    pass


def _pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in rows:
        if key in result:
            raise VerificationError("duplicate JSON key")
        result[key] = value
    return result


def _finite(value: Any) -> bool:
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_finite(item) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _finite(item) for key, item in value.items())
    return value is None or type(value) in {str, int, bool}


def _identity(row: os.stat_result) -> tuple[int, ...]:
    return (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns,
            row.st_ctime_ns, row.st_mode)


def _read_snapshot(path: Path, expected_sha: str | None = None) -> tuple[dict[str, Any], str, tuple[int, ...]]:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise VerificationError(f"cannot open evidence: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise VerificationError("evidence is not a regular file")
        raw = bytearray()
        while block := os.read(descriptor, 1 << 20):
            raw.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    digest = sha256(raw).hexdigest()
    if _identity(before) != _identity(after) or (expected_sha is not None and digest != expected_sha):
        raise VerificationError("evidence identity/SHA changed")
    try:
        value = json.loads(bytes(raw), object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                VerificationError(f"nonfinite JSON token: {token}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError("invalid JSON") from exc
    if type(value) is not dict or not _finite(value):
        raise VerificationError("JSON root/finite shape differs")
    return value, digest, _identity(after)


def _read(path: Path, expected_sha: str | None = None) -> dict[str, Any]:
    return _read_snapshot(path, expected_sha)[0]


def _file_snapshot(path: Path, expected_sha: str | None = None) -> tuple[str, tuple[int, ...]]:
    """Hash arbitrary implementation/input bytes through exactly one descriptor."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise VerificationError(f"cannot open bound file: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise VerificationError("bound file is not regular")
        digest = sha256()
        while block := os.read(descriptor, 1 << 20):
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    observed = digest.hexdigest()
    if _identity(before) != _identity(after) or (expected_sha is not None and observed != expected_sha):
        raise VerificationError("bound file identity/SHA changed")
    return observed, _identity(after)


def _keys(value: Any, name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(contract.FIELD_KEYS[name]):
        raise VerificationError(f"{name} exact keys differ")
    return value


def _without(value: Mapping[str, Any], *names: str) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in names}


def _digest(value: Any) -> str:
    try:
        return contract.stable_digest(value)
    except (TypeError, ValueError) as exc:
        raise VerificationError("nonfinite digest input") from exc


def _timestamp(value: Any) -> bool:
    if type(value) is not str:
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None \
        and parsed.utcoffset().total_seconds() == 0


def _sealed(row: dict[str, Any], field: str, name: str) -> dict[str, Any]:
    _keys(row, name)
    if row.get("schema_version") != contract.SCHEMAS[name] \
            or row[field] != _digest(_without(row, field)):
        raise VerificationError(f"{name} digest differs")
    if "created_at" in row and not _timestamp(row["created_at"]):
        raise VerificationError(f"{name} timestamp differs")
    return row


def _reject_ancestry(path: Path) -> None:
    current = path.absolute()
    for item in (current, *current.parents):
        if item.exists() and item.is_symlink():
            raise VerificationError("symlinked ancestry is forbidden")


def _tree(root: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    files, directories = [], []
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise VerificationError("terminal tree has a forbidden entry")
        if stat.S_ISREG(mode):
            files.append(path.relative_to(root).as_posix())
        else:
            directories.append(path.relative_to(root).as_posix())
    return tuple(sorted(files)), tuple(sorted(directories))


def _strip_timing(value: Any) -> Any:
    if type(value) is dict:
        return {key: _strip_timing(item) for key, item in value.items()
                if key not in contract.TIMING_FIELDS and key not in {
                    "block_rows", "block_order", "peak_rss_mb"}}
    if type(value) is list:
        return [_strip_timing(item) for item in value]
    return value


def _proposal_report(value: Any) -> list[BoundProposal]:
    keys = {"schema_version", "generation_id", "query_episode_id", "candidates",
        "rows_scanned", "eligible_rows", "eligible_main_rows", "eligible_overflow_rows",
        "route_counts", "route_quotas", "block_rows", "block_order", "elapsed_seconds",
        "peak_rss_mb", "candidate_digest", "result_digest", "contract_digest", "input_digest"}
    if type(value) is not dict or set(value) != keys:
        raise VerificationError("proposal report keys differ")
    candidates = []
    ckeys = {"episode_id", "symbol", "cutoff_ns", "quality_tier", "lower_bound_hex",
             "routes", "overflow_fallback"}
    for row in value["candidates"]:
        if type(row) is not dict or set(row) != ckeys:
            raise VerificationError("proposal candidate differs")
        try:
            lower = float.fromhex(row["lower_bound_hex"])
        except (TypeError, ValueError) as exc:
            raise VerificationError("proposal lower bound differs") from exc
        candidates.append(BoundProposal(row["episode_id"], row["symbol"], row["cutoff_ns"],
            row["quality_tier"], lower, tuple(row["routes"]), row["overflow_fallback"]))
    deterministic = {"schema_version": value["schema_version"],
        "contract_digest": packed_bound_search_contract(branch_aware=True)["digest"],
        "generation_id": value["generation_id"], "query_episode_id": value["query_episode_id"],
        "rows_scanned": value["rows_scanned"], "eligible_rows": value["eligible_rows"],
        "eligible_main_rows": value["eligible_main_rows"],
        "eligible_overflow_rows": value["eligible_overflow_rows"],
        "route_counts": value["route_counts"], "route_quotas": value["route_quotas"],
        "candidate_digest": value["candidate_digest"],
        "real_forward_outcomes_accessed": False, "input_digest": value["input_digest"]}
    if value["contract_digest"] != deterministic["contract_digest"] \
            or value["eligible_rows"] != value["eligible_main_rows"] + value["eligible_overflow_rows"] \
            or candidates != sorted(candidates, key=lambda item: (item.lower_bound, item.episode_id)) \
            or value["candidate_digest"] != bound_proposal_candidate_digest(candidates) \
            or value["result_digest"] != stable_hash(deterministic):
        raise VerificationError("proposal report reconstruction differs")
    return candidates


def _proposal(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    _keys(row, "proposal")
    state = row["state"]
    expected = {"schema_version", "task_id", "case_id", "query_id", "query_binding",
        "resident_lease_digests", "source_binding_before", "source_binding_after",
        "resident_snapshot", "forward", "reverse", "semantic_digest"}
    if type(state) is not dict or set(state) != expected \
            or state["schema_version"] != "m04r14-exact-scheduler-proposal-v1" \
            or row["digest"] != _digest(state) \
            or not _timestamp(row["created_at"]):
        raise VerificationError("proposal wrapper differs")
    _proposal_report(state["forward"]); _proposal_report(state["reverse"])
    if state["source_binding_before"] != state["source_binding_after"] \
            or len(state["resident_lease_digests"]) != 4 \
            or len(set(state["resident_lease_digests"])) != 1 \
            or _strip_timing(state["forward"]) != _strip_timing(state["reverse"]) \
            or state["forward"]["block_rows"] != 4096 or state["forward"]["block_order"] != "forward" \
            or state["reverse"]["block_rows"] != 4097 or state["reverse"]["block_order"] != "reverse" \
            or state["semantic_digest"] != _digest(_strip_timing(state["forward"])):
        raise VerificationError("proposal parity/binding differs")
    if type(row["measurement"]) is not dict or not _finite(row["measurement"]):
        raise VerificationError("proposal measurement differs")
    return state, row["measurement"]


def _certificate(certificate: Any, matches: Any, query_id: str, input_digest: str) -> None:
    cert_keys = {"schema_version", "contract_digest", "generation_id", "query_episode_id",
        "input_digest", "eligible_candidates", "exact_evaluated", "safely_pruned",
        "stopped_early", "stop_threshold", "next_lower_bound",
        "maximum_quantized_bound_excess", "materialization_groups", "sparse_symbols",
        "batch_symbols", "rounds", "result_digest", "native_bound_accounting",
        "minimum_native_pruned_bound", "threshold_closure_passes"}
    match_keys = {"episode_id", "symbol", "cutoff", "total_distance",
        "component_distances", "alignment", "quality_tier"}
    if type(certificate) is not dict or set(certificate) != cert_keys \
            or type(matches) is not list or len(matches) != 20 \
            or any(type(row) is not dict or set(row) != match_keys for row in matches):
        raise VerificationError("certificate/match schema differs")
    accounting = certificate["native_bound_accounting"]
    if type(accounting) is not dict or set(accounting) != {"native_bound_evaluated",
            "exact_dtw_evaluated", "native_bound_pruned", "packed_bound_pruned"}:
        raise VerificationError("certificate accounting differs")
    components = {"stage", "price", "candle_volatility", "volume_shock",
                  "market_context", "structural", "coarse"}
    if any(type(row["total_distance"]) is not float or not math.isfinite(row["total_distance"])
            or row["total_distance"] < 0 or set(row["component_distances"]) != components
            or any(type(value) is not float or not math.isfinite(value) or value < 0
                   for value in row["component_distances"].values())
            or type(row["alignment"]) is not list or not row["alignment"]
            or row["quality_tier"] not in {"A", "B"} for row in matches):
        raise VerificationError("match numeric/content schema differs")
    totals = [(row["total_distance"], row["episode_id"]) for row in matches]
    search_contract = certified_packed_search_contract(requested_positions=True,
        vector_lower_bounds=True, deferred_alignments=True, compact_scored=True,
        native_bound_deferral=True, streaming_threshold_closure=True,
        branch_aware_packed_bounds=True)
    deterministic = {"schema_version": search_contract["schema_version"],
        "contract_digest": search_contract["digest"], "generation_id": certificate["generation_id"],
        "query_episode_id": query_id, "input_digest": input_digest,
        "eligible_candidates": certificate["eligible_candidates"],
        "exact_evaluated": certificate["exact_evaluated"], "safely_pruned": certificate["safely_pruned"],
        "stopped_early": certificate["stopped_early"],
        "stop_threshold_hex": certificate["stop_threshold"].hex(),
        "next_lower_bound_hex": None if certificate["next_lower_bound"] is None else certificate["next_lower_bound"].hex(),
        "maximum_quantized_bound_excess_hex": certificate["maximum_quantized_bound_excess"].hex(),
        "rounds": certificate["rounds"], "matches": [{"episode_id": row["episode_id"],
            "total_hex": row["total_distance"].hex(), "components": {key: value.hex()
                for key, value in sorted(row["component_distances"].items())},
            "alignment": row["alignment"]} for row in matches],
        "real_forward_outcomes_accessed": False, "native_bound_accounting": accounting,
        "minimum_native_pruned_bound_hex": None if certificate["minimum_native_pruned_bound"] is None
            else certificate["minimum_native_pruned_bound"].hex(),
        "threshold_closure_passes": certificate["threshold_closure_passes"]}
    if certificate["contract_digest"] != search_contract["digest"] \
            or certificate["query_episode_id"] != query_id or certificate["input_digest"] != input_digest \
            or totals != sorted(totals) or len({row["episode_id"] for row in matches}) != 20 \
            or certificate["eligible_candidates"] != certificate["exact_evaluated"] + certificate["safely_pruned"] \
            or certificate["exact_evaluated"] != accounting["exact_dtw_evaluated"] \
            or accounting["native_bound_evaluated"] != accounting["exact_dtw_evaluated"] + accounting["native_bound_pruned"] \
            or certificate["eligible_candidates"] != accounting["native_bound_evaluated"] + accounting["packed_bound_pruned"] \
            or certificate["stop_threshold"] != max(row["total_distance"] for row in matches) \
            or certificate["materialization_groups"] != certificate["sparse_symbols"] + certificate["batch_symbols"] \
            or certificate["result_digest"] != stable_hash(deterministic):
        raise VerificationError("certificate reconstruction differs")


def _case(snapshots: Mapping[str, dict[str, Any]], shas: Mapping[str, str],
          ordinal: int, query_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    prefix = f"cases/{ordinal:03d}-{query_id}"
    proposal_relative = f"{prefix}/PROPOSAL.json"
    exact_relative = f"{prefix}/EXACT-w1.json"
    case_relative = f"{prefix}/CASE.json"
    proposal_raw = snapshots[proposal_relative]; proposal_sha = shas[proposal_relative]
    proposal, _proposal_measurement = _proposal(proposal_raw)
    exact = _sealed(snapshots[exact_relative], "result_digest", "exact")
    if exact["ordinal"] != ordinal or exact["query_id"] != query_id or exact["workers"] != 1 \
            or exact["proposal_sha256"] != proposal_sha \
            or exact["semantic_digest"] != _digest(exact["semantic"]) \
            or exact["measurement_digest"] != _digest(exact["measurement"]):
        raise VerificationError("exact leaf reconstruction differs")
    semantic = exact["semantic"]
    attempt_keys = {"schema_version", "case_id", "query_id", "workers",
        "proposal_semantic_digest", "certificate", "matches", "certificate_result_digest",
        "match_digest", "lease_before", "lease_after", "source_binding_before",
        "source_binding_after"}
    if type(semantic) is not dict or set(semantic) != attempt_keys \
            or semantic["schema_version"] != "m04r14-exact-scheduler-attempt-v1" \
            or semantic["case_id"] != exact["case_id"] or semantic["query_id"] != query_id \
            or semantic["workers"] != 1 or semantic["proposal_semantic_digest"] != proposal["semantic_digest"] \
            or semantic["certificate_result_digest"] != semantic["certificate"].get("result_digest") \
            or semantic["match_digest"] != _digest(semantic["matches"]) \
            or semantic["lease_before"] != semantic["lease_after"] \
            or semantic["source_binding_before"] != proposal["query_binding"] \
            or semantic["source_binding_after"] != proposal["query_binding"]:
        raise VerificationError("exact attempt reconstruction differs")
    input_digest = proposal["query_binding"].get("certified_input_digest")
    _certificate(semantic.get("certificate"), semantic.get("matches"), query_id, input_digest)
    case = _sealed(snapshots[case_relative], "bundle_digest", "case_bundle")
    case_semantic = _sealed(case["semantic"], "semantic_digest", "case_semantic")
    case_measurement = _sealed(case["measurement"], "measurement_digest", "case_measurement")
    expected_semantic = {"schema_version": contract.SCHEMAS["case_semantic"],
        "ordinal": ordinal, "query_id": query_id, "case_id": exact["case_id"],
        "query_binding": proposal["query_binding"],
        "forward_proposal": _strip_timing(proposal["forward"]),
        "reverse_proposal": _strip_timing(proposal["reverse"]), "proposal_parity": True,
        "certificate": semantic["certificate"], "matches": semantic["matches"],
        "source_leases": [proposal["source_binding_before"], proposal["source_binding_after"],
            semantic["source_binding_before"], semantic["source_binding_after"]],
        "resident_leases": list(proposal["resident_lease_digests"]) +
            [semantic["lease_before"], semantic["lease_after"]]}
    expected_semantic["semantic_digest"] = _digest(expected_semantic)
    if case_semantic != expected_semantic or case["ordinal"] != ordinal \
            or case["query_id"] != query_id or case["proposal_sha256"] != proposal_sha \
            or case["exact_sha256"] != shas[exact_relative] \
            or case["semantic_digest"] != case_semantic["semantic_digest"] \
            or case["measurement_digest"] != case_measurement["measurement_digest"]:
        raise VerificationError("CASE reconstruction differs")
    return case, case_semantic, case_measurement


def _runtime_envelope(prereg: Mapping[str, Any]) -> Mapping[str, Any]:
    runtime = prereg["runtime_binding"]
    if type(runtime) is not dict or set(runtime) != {"state", "digest"} \
            or runtime["digest"] != _digest(runtime["state"]):
        raise VerificationError("runtime binding differs")
    state = runtime["state"]
    if type(state) is not dict or set(state) != {"git_head", "files", "environment", "contracts"} \
            or type(state.get("git_head")) is not str or type(state.get("files")) is not dict \
            or state["contracts"] != {"descriptor_digest": contract.DESCRIPTOR_DIGEST,
                                      "execution_policy": contract.EXECUTION_POLICY}:
        raise VerificationError("runtime state differs")
    return state


def _runtime_and_lineage(prereg: Mapping[str, Any], repository: Path) -> None:
    state = _runtime_envelope(prereg)
    h0 = state["git_head"]; manifest = state["files"]
    environment = state["environment"]
    if type(environment) is not dict or set(environment) != {"state", "digest"} \
            or environment["digest"] != _digest(environment["state"]):
        raise VerificationError("runtime environment seal differs")
    env_state = environment["state"]
    expected_environment = {"python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(), "system": platform.system(),
        "machine": platform.machine(), "packages": {name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "numba", "pyarrow")},
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "thread_environment": {key: os.environ.get(key) for key in THREAD_ENV_KEYS}}
    if type(env_state) is not dict or set(env_state) != set(expected_environment) | {"cgroup_cpu_configuration"} \
            or any(env_state[key] != value for key, value in expected_environment.items()):
        raise VerificationError("runtime environment differs")
    cgroup = env_state["cgroup_cpu_configuration"]
    if type(cgroup) is not dict or set(cgroup) != {"schema_version", "cgroup_path", "cpuset_path",
            "effective_cpus", "cpu_max_path", "quota_usec", "period_usec",
            "effective_quota_cpus", "cpu_stat_path"}:
        raise VerificationError("runtime cgroup configuration differs")
    try:
        maximum = Path(cgroup["cpu_max_path"]).read_text().strip().split()
        cpus = []
        for part in Path(cgroup["cpuset_path"]).read_text().strip().split(","):
            lo, sep, hi = part.partition("-"); cpus.extend(range(int(lo), int(hi) + 1) if sep else [int(lo)])
        quota = None if maximum[0] == "max" else int(maximum[0]); period = int(maximum[1])
    except (OSError, ValueError, IndexError) as exc:
        raise VerificationError("cannot reconstruct runtime cgroup") from exc
    if cgroup["effective_cpus"] != cpus or cgroup["quota_usec"] != quota \
            or cgroup["period_usec"] != period \
            or cgroup["effective_quota_cpus"] != (None if quota is None else quota / period) \
            or not Path(cgroup["cpu_stat_path"]).is_file():
        raise VerificationError("runtime cgroup drifted")
    tracked = subprocess.run(["git", "ls-tree", "-r", "--name-only", h0], cwd=repository,
        text=True, capture_output=True, check=True).stdout.splitlines()
    expected_files = sorted({name for name in tracked if name.startswith("src/market_analogues/")
                             and name.endswith(".py")} | set(RUNTIME_FIXED_FILES))
    current = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository, text=True,
        capture_output=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repository, text=True, capture_output=True, check=True).stdout.strip()
    if sorted(manifest) != expected_files or dirty or subprocess.run(
            ["git", "merge-base", "--is-ancestor", h0, current], cwd=repository).returncode:
        raise VerificationError("runtime manifest/Git state differs")
    for relative, expected in manifest.items():
        historical = subprocess.run(["git", "show", f"{h0}:{relative}"], cwd=repository,
            capture_output=True, check=True).stdout
        if sha256(historical).hexdigest() != expected \
                or _file_snapshot(repository / relative)[0] != expected:
            raise VerificationError("runtime implementation drifted")
    prereg_path = repository / contract.PREREGISTRATION_RELATIVE
    if _read(prereg_path) != prereg:
        raise VerificationError("canonical preregistration differs")
    additions = subprocess.run(["git", "log", "--diff-filter=A", "--format=%H", "--",
        contract.PREREGISTRATION_RELATIVE], cwd=repository, text=True,
        capture_output=True, check=True).stdout.splitlines()
    if len(additions) != 1:
        raise VerificationError("preregistration H1 differs")
    h1 = additions[0]
    parents = subprocess.run(["git", "rev-list", "--parents", "-n", "1", h1], cwd=repository,
        text=True, capture_output=True, check=True).stdout.split()
    changed = subprocess.run(["git", "diff-tree", "--no-commit-id", "--name-only", "-r", h1],
        cwd=repository, text=True, capture_output=True, check=True).stdout.splitlines()
    if parents != [h1, h0] or changed != [contract.PREREGISTRATION_RELATIVE]:
        raise VerificationError("preregistration H0/H1 lineage differs")


def _registry_case_map(registry: Mapping[str, Any], source_lock: Mapping[str, Any]) \
        -> dict[str, dict[str, Any]]:
    rows = registry.get("cases_data")
    if registry.get("registry_digest") != FROZEN_REGISTRY_DIGEST \
            or source_lock.get("registry_digest") != FROZEN_REGISTRY_DIGEST \
            or type(rows) is not list or len(rows) != 60 \
            or any(type(row) is not dict or type(row.get("episode_id")) is not str for row in rows):
        raise VerificationError("registry binding differs")
    by_id = {row["episode_id"]: row for row in rows}
    if len(by_id) != 60 or set(by_id) != set(contract.QUERY_IDS):
        raise VerificationError("registry query universe differs")
    return by_id


def _production_queries(prereg: Mapping[str, Any], source_lock: Mapping[str, Any]) \
        -> tuple[dict[str, tuple[PackedBoundQuery, dict[str, Any]]], Any, Path]:
    roots = prereg["roots"]
    config_path = Path(roots["config"]).resolve(strict=True)
    registry_path = (Path(roots["registry"]) / "query-registry.json").resolve(strict=True)
    if _file_snapshot(config_path)[0] != source_lock["config_sha256"] \
            or _file_snapshot(registry_path)[0] != source_lock["registry_sha256"]:
        raise VerificationError("causal input SHA differs")
    registry = _read(registry_path)
    by_id = _registry_case_map(registry, source_lock)
    config = load_config(config_path); source = source_from_spec(config.datasets["nasdaq"])
    benchmark = source.load_benchmark(); output = {}
    for query_id in contract.QUERY_IDS:
        raw = by_id[query_id]
        if raw.get("dataset_id") != "nasdaq":
            raise VerificationError("registry dataset differs")
        episode = build_episode(source, InstrumentKey("nasdaq", raw["symbol"]), raw["cutoff"],
                                raw["lookback"], raw["representation_version"])
        request = SearchQuery(episode.key, ("nasdaq",), ("A", "B"), 20, False, True, 3, 60)
        packed = PackedBoundQuery(episode.key.id, episode.key.instrument.source_symbol,
            int(episode.bars.timestamp.iloc[0].value), int(latest_eligible_cutoff(episode, 60).value),
            represent(episode), request.quality_tiers)
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
            "packed_provenance_digest": source_lock["provenance_digest"],
            "query_representation_digest": representation_input_digest(packed.representation)}
        binding = {**deterministic, "packed_query_input_digest": _packed_query_input_digest(packed),
                   "certified_input_digest": stable_hash(deterministic)}
        if episode.key.id != query_id:
            raise VerificationError("causal query reconstruction differs")
        output[query_id] = (packed, binding)
    expected_digests = [stable_hash(output[q][1]) for q in contract.QUERY_IDS]
    if source_lock["query_binding_digests"] != expected_digests:
        raise VerificationError("source query bindings differ")
    return output, source, Path(roots["source"]) / "store"


def _report_payload(report: Any) -> dict[str, Any]:
    return {"schema_version": report.schema_version, "generation_id": report.generation_id,
        "query_episode_id": report.query_episode_id, "candidates": [{"episode_id": x.episode_id,
        "symbol": x.symbol, "cutoff_ns": x.cutoff_ns, "quality_tier": x.quality_tier,
        "lower_bound_hex": x.lower_bound.hex(), "routes": list(x.routes),
        "overflow_fallback": x.overflow_fallback} for x in report.candidates],
        "rows_scanned": report.rows_scanned, "eligible_rows": report.eligible_rows,
        "eligible_main_rows": report.eligible_main_rows,
        "eligible_overflow_rows": report.eligible_overflow_rows,
        "route_counts": dict(report.route_counts), "route_quotas": dict(report.route_quotas),
        "block_rows": report.block_rows, "block_order": report.block_order,
        "elapsed_seconds": report.elapsed_seconds, "peak_rss_mb": report.peak_rss_mb,
        "candidate_digest": report.candidate_digest, "result_digest": report.result_digest,
        "contract_digest": report.contract_digest, "input_digest": report.input_digest}


def _production_replay(prereg: Mapping[str, Any], source_lock: Mapping[str, Any],
                       semantics: Sequence[Mapping[str, Any]]) -> None:
    queries, _source, store_root = _production_queries(prereg, source_lock)
    generation = source_lock["generation_id"]; provenance = source_lock["provenance_digest"]
    generation_root = store_root / "generations" / generation
    def identities() -> tuple[tuple[str, tuple[int, ...]], ...]:
        rows = []
        for path in generation_root.rglob("*"):
            observed = path.lstat()
            if stat.S_ISLNK(observed.st_mode) or not (stat.S_ISREG(observed.st_mode)
                                                       or stat.S_ISDIR(observed.st_mode)):
                raise VerificationError("packed generation contains forbidden entry")
            rows.append((path.relative_to(generation_root).as_posix(), _identity(observed)))
        return tuple(sorted(rows))
    before_identity = identities()
    loaded = load_packed_generation(store_root, generation,
        expected_provenance_digest=provenance, verify_content=True, validate_records=True)
    requested = {row["query_id"]: {m["episode_id"] for m in row["matches"]} for row in semantics}
    wanted = set().union(*requested.values())
    encoded = np.asarray([np.void(bytes.fromhex(x)) for x in wanted], dtype="V12")
    records: dict[str, tuple[Any, bool]] = {}
    for overflow, selected in ((False, loaded.rows[np.isin(loaded.rows["episode_id"], encoded)]),
                               (True, loaded.overflow[np.isin(loaded.overflow["episode_id"], encoded)])):
        for record in selected:
            identifier = decode_episode_id(record["episode_id"])
            if identifier in records: raise VerificationError("packed match is duplicated")
            records[identifier] = (record, overflow)
    if set(records) != wanted:
        raise VerificationError("match absent from durable generation")
    for semantic in semantics:
        query_id = semantic["query_id"]; packed, binding = queries[query_id]
        if semantic["query_binding"] != binding:
            raise VerificationError("case causal binding differs")
        stored = semantic["forward_proposal"]
        quota = stored["route_quotas"]
        replay = scan_packed_bound_proposals(store_root, generation, packed,
            route_quotas=quota, block_rows=4093, block_order="reverse", branch_aware=True,
            verify_content=False, expected_provenance_digest=provenance)
        if _strip_timing(_report_payload(replay)) != _strip_timing(stored):
            raise VerificationError("independent proposal replay differs")
        symbol_id = loaded.symbols.index(packed.symbol) if packed.symbol in loaded.symbols else None
        main_ids = [x for x in requested[query_id] if not records[x][1]]
        main_rows = np.asarray([records[x][0] for x in main_ids], dtype=loaded.rows.dtype)
        totals = packed_branch_aware_lower_bounds(packed.representation, main_rows).totals \
            if len(main_rows) else np.asarray([], dtype=np.float64)
        bounds = dict(zip(main_ids, map(float, totals), strict=True))
        for match in semantic["matches"]:
            record, overflow = records[match["episode_id"]]; cutoff = int(record["cutoff_ns"])
            sid = int(record["symbol_id"]); tier = int(record["quality_tier"])
            eligible = bytes(record["episode_id"]) != bytes.fromhex(query_id) \
                and cutoff <= packed.latest_eligible_ns \
                and not (symbol_id is not None and sid == symbol_id and cutoff >= packed.query_start_ns) \
                and tier in {1, 2}
            if not eligible or loaded.symbols[sid] != match["symbol"] \
                    or pd.Timestamp(match["cutoff"]).value != cutoff \
                    or TIER_NAMES[tier] != match["quality_tier"] \
                    or not math.isfinite(0.0 if overflow else bounds[match["episode_id"]]) \
                    or (0.0 if overflow else bounds[match["episode_id"]]) > match["total_distance"] + 1e-12:
                raise VerificationError("durable match authentication differs")
        closures = semantic["certificate"]["threshold_closure_passes"]
        candidates = stored["candidates"]
        if len(candidates) < 16384: raise VerificationError("proposal frontier is incomplete")
        excluded = frozenset(x["episode_id"] for x in candidates[:16384])
        admitted_matches: set[str] = set()
        if closures:
            excluded_digest = stable_hash(sorted(excluded))
            for closure in closures:
                admitted: set[str] = set()
                def consume(rows: Any) -> None:
                    admitted.update(decode_episode_id(row["episode_id"]) for row in rows)
                report = scan_packed_bound_threshold(store_root, generation, packed,
                    lower_exclusive=closure["lower_exclusive"], upper_inclusive=closure["upper_inclusive"],
                    excluded_episode_ids=excluded, block_rows=4097, block_order="reverse",
                    branch_aware=True, verify_content=False,
                    expected_provenance_digest=provenance, consume=consume)
                if closure["excluded_prefix_digest"] != excluded_digest or not all((
                    report.input_digest == _packed_query_input_digest(packed),
                    report.exclusions_digest == excluded_digest,
                    report.eligible_rows == semantic["certificate"]["eligible_candidates"],
                    report.excluded_eligible_rows == len(excluded),
                    report.admitted_rows == closure["admitted_rows"],
                    report.minimum_above_upper == closure["minimum_packed_unclassified_bound"],
                    report.admitted_set_digest == closure["admitted_set_digest"],
                    report.result_digest == closure["scan_result_digest"])):
                    raise VerificationError("closure scan replay differs")
                admitted_matches.update(admitted)
        if not requested[query_id] <= excluded | admitted_matches:
            raise VerificationError("returned match lacks proposal/closure admission")
    if identities() != before_identity:
        raise VerificationError("packed generation changed during independent replay")


def verify_terminal(root: Path, *, repository: Path, require_production: bool = True,
                    replay_scans: bool = True) -> dict[str, Any]:
    repository = repository.resolve(strict=True); _reject_ancestry(root)
    root = root.resolve(strict=True)
    if _tree(root) != (contract.successful_tree(), contract.successful_directories()) \
            or (root / "INCOMPLETE.json").exists():
        raise VerificationError("exact successful tree differs")
    complete_raw, complete_sha, complete_identity = _read_snapshot(root / "COMPLETE.json")
    complete = _sealed(complete_raw, "complete_digest", "complete")
    manifest = complete["leaf_manifest"]
    expected_paths = sorted(path for path in contract.successful_tree() if path != "COMPLETE.json")
    if type(manifest) is not list or [row.get("path") for row in manifest] != expected_paths \
            or complete["leaf_manifest_digest"] != _digest(manifest):
        raise VerificationError("complete manifest differs")
    snapshots: dict[str, dict[str, Any]] = {}; identities = {}
    manifest_shas = {row["path"]: row["sha256"] for row in manifest
                     if type(row) is dict and set(row) == {"path", "sha256"}}
    for row in manifest:
        if type(row) is not dict or set(row) != {"path", "sha256"}:
            raise VerificationError("manifest row differs")
        value, digest, identity = _read_snapshot(root / row["path"], row["sha256"])
        if digest != row["sha256"]:
            raise VerificationError("manifest SHA differs")
        snapshots[row["path"]] = value; identities[row["path"]] = identity
    prereg = _keys(snapshots["CONTRACT.json"], "contract")
    if prereg["preregistration_digest"] != _digest(_without(prereg, "preregistration_digest")) \
            or tuple(prereg["query_ids"]) != contract.QUERY_IDS \
            or prereg["execution_policy"] != contract.EXECUTION_POLICY \
            or prereg["t14_02_binding"] != contract.T14_02_BINDING \
            or prereg["claims"] != contract.CLAIMS \
            or Path(prereg["roots"]["candidate"]).resolve() != root:
        raise VerificationError("preregistration reconstruction differs")
    if require_production:
        _runtime_and_lineage(prereg, repository)
    run = _sealed(snapshots["RUN_STARTED.json"], "result_digest", "run")
    source = _sealed(snapshots["SOURCE_LOCK.json"], "result_digest", "source")
    resident = _sealed(snapshots["RESIDENT.json"], "result_digest", "resident")
    if run["preregistration_digest"] != prereg["preregistration_digest"] \
            or run["query_ids"] != list(contract.QUERY_IDS) \
            or run["claims"] != contract.CLAIMS or run["execution_policy"] != contract.EXECUTION_POLICY \
            or run["status"] != "running" or source["generation_id"] == "" \
            or source["source_tree_digest"] != resident["content_digest"] \
            or resident["store_root"] != str(Path(prereg["roots"]["resident"]) / "store"):
        raise VerificationError("run binding differs")
    if require_production:
        ready_path = Path(prereg["roots"]["resident"]) / "READY.json"
        observation = observe_ready_strict(ready_path); lease = resident_file_identity_lease(ready_path)
        if resident["ready_digest"] != observation["ready_digest"] \
                or resident["content_digest"] != observation["content_digest"] \
                or resident["ready_file_sha256"] != observation["ready_file_sha256"] \
                or lease.get("ready_digest") != observation["ready_digest"] \
                or lease.get("content_digest") != observation["content_digest"] \
                or lease.get("ready_file_sha256") != observation["ready_file_sha256"] \
                or resident["lease"] != lease:
            raise VerificationError("live resident differs")
    previous = None; semantics = []; measurements = []
    for ordinal, query_id in enumerate(contract.QUERY_IDS):
        started = _sealed(snapshots[f"events/{ordinal*2:03d}-CASE_STARTED-{query_id}.json"],
                           "event_digest", "event")
        case, semantic, measurement = _case(snapshots, manifest_shas, ordinal, query_id)
        proposal_state = snapshots[f"cases/{ordinal:03d}-{query_id}/PROPOSAL.json"]["state"]
        expected_lease = resident["lease"].get("lease_digest")
        if proposal_state["resident_snapshot"] != _without(resident, "schema_version", "created_at", "result_digest") \
                and proposal_state["resident_snapshot"] != resident:
            raise VerificationError("proposal resident snapshot differs")
        if semantic["source_leases"] != [semantic["query_binding"]] * 4 \
                or semantic["resident_leases"] != [expected_lease] * 6:
            raise VerificationError("case source/resident leases differ")
        completed = _sealed(snapshots[f"events/{ordinal*2+1:03d}-CASE_COMPLETED-{query_id}.json"],
                             "event_digest", "event")
        if started["sequence"] != ordinal * 2 or started["event"] != "CASE_STARTED" \
                or started["ordinal"] != ordinal or started["query_id"] != query_id \
                or started["previous_event_digest"] != previous or started["case_bundle_sha256"] is not None:
            raise VerificationError("CASE_STARTED ledger differs")
        case_sha = manifest_shas[f"cases/{ordinal:03d}-{query_id}/CASE.json"]
        if completed["sequence"] != ordinal * 2 + 1 or completed["event"] != "CASE_COMPLETED" \
                or completed["ordinal"] != ordinal or completed["query_id"] != query_id \
                or completed["previous_event_digest"] != started["event_digest"] \
                or completed["case_bundle_sha256"] != case_sha:
            raise VerificationError("CASE_COMPLETED ledger differs")
        previous = completed["event_digest"]; semantics.append(semantic); measurements.append(measurement)
    if require_production:
        if replay_scans is not True:
            raise VerificationError("production replay cannot be disabled")
        _production_replay(prereg, source, semantics)
    semantic_aggregate = _sealed(snapshots["SEMANTICS.json"], "semantic_digest", "semantics")
    measurement_aggregate = _sealed(snapshots["MEASUREMENTS.json"], "measurement_digest", "measurements")
    semantic_flags = {"forward_reverse_equal": all(row["proposal_parity"] is True for row in semantics),
        "all_certified": all(type(row["certificate"]) is dict for row in semantics),
        "all_finite": all(_finite(row) for row in semantics)}
    if semantic_aggregate["ordered_case_semantic_digests"] != [row["semantic_digest"] for row in semantics] \
            or semantic_aggregate["claims"] != contract.CLAIMS \
            or semantic_aggregate["status"] != "complete" or measurement_aggregate["status"] != "complete" \
            or any(semantic_aggregate[key] != value for key, value in semantic_flags.items()) \
            or semantic_aggregate["semantic_passed"] is not all(semantic_flags.values()) \
            or measurement_aggregate["ordered_case_measurement_digests"] != [row["measurement_digest"] for row in measurements]:
        raise VerificationError("aggregate reconstruction differs")
    raw = [{key: value for key, value in row.items() if key not in {
        "schema_version", "ordinal", "query_id", "case_id", "measurement_digest"}}
        for row in measurements]
    limits = contract.EXECUTION_POLICY["performance_limits"]
    p95_index = limits["p95_order_index_zero_based_for_60"]
    exact_values = sorted(row["exact_stage_seconds"] for row in measurements)
    end_values = sorted(row["end_to_end_seconds"] for row in measurements)
    expected_summary = {"cases": 60,
        "total_end_to_end_seconds": sum(row["end_to_end_seconds"] for row in measurements),
        "p95_method": limits["p95_method"],
        "exact_stage_seconds_p95": exact_values[p95_index],
        "case_end_to_end_seconds_p95": end_values[p95_index]}
    def expected_proposal_gate(row: Mapping[str, Any]) -> bool:
        resources = row["resources"]["proposal"]
        after = resources.get("after", {}) if type(resources) is dict else {}
        process = row["child_process"].get("proposal", {})
        process_peak = process.get("effective_peak_rss_kib", 0) if type(process) is dict else 0
        return type(after.get("swap_kib")) is int and after["swap_kib"] == 0 \
            and type(process.get("peak_swap_kib")) is int and process["peak_swap_kib"] == 0 \
            and type(process.get("final_swap_kib")) is int and process["final_swap_kib"] == 0 \
            and row["forward_proposal_seconds"] <= limits["forward_proposal_seconds"] \
            and row["reverse_proposal_seconds"] <= limits["reverse_proposal_seconds"] \
            and float(process_peak) / 1024.0 <= limits["proposal_process_rss_mb"]
    if any(row["proposal_resource_gate_passed"] is not expected_proposal_gate(row)
           or row["exact_stage_slo_passed"] is not (
               row["exact_stage_seconds"] <= limits["exact_stage_max_seconds"])
           or row["end_to_end_slo_passed"] is not (
               row["end_to_end_seconds"] <= limits["end_to_end_max_seconds"])
           for row in measurements):
        raise VerificationError("per-case performance gates differ")
    def exact_resource_gate(row: Mapping[str, Any]) -> bool:
        process = row["child_process"]["exact"]; resources = row["resources"]["exact"]
        return type(process) is dict and type(process.get("peak_swap_kib")) is int \
            and process["peak_swap_kib"] == 0 and type(process.get("final_swap_kib")) is int \
            and process["final_swap_kib"] == 0 and type(resources) is dict \
            and type(resources.get("after")) is dict \
            and type(resources["after"].get("swap_kib")) is int and resources["after"]["swap_kib"] == 0
    resource_pass = all(row["proposal_resource_gate_passed"] and exact_resource_gate(row)
                        for row in measurements)
    slo_pass = all(row["exact_stage_slo_passed"] and row["end_to_end_slo_passed"]
                   for row in measurements) \
        and exact_values[p95_index] <= limits["exact_stage_p95_seconds"] \
        and end_values[p95_index] <= limits["end_to_end_p95_seconds"]
    if measurement_aggregate["raw_case_metrics"] != raw \
            or measurement_aggregate["summary"] != expected_summary \
            or measurement_aggregate["resource_gates"] != {"all_passed": resource_pass} \
            or measurement_aggregate["slo_gates"] != {"all_passed": slo_pass} \
            or measurement_aggregate["performance_passed"] is not (resource_pass and slo_pass):
        raise VerificationError("measurement summary differs")
    if complete["ledger_head_digest"] != previous \
            or complete["status"] != "complete" \
            or complete["preregistration_digest"] != prereg["preregistration_digest"] \
            or complete["semantic_digest"] != semantic_aggregate["semantic_digest"] \
            or complete["measurement_digest"] != measurement_aggregate["measurement_digest"] \
            or complete["semantic_passed"] is not semantic_aggregate["semantic_passed"] \
            or complete["performance_passed"] is not measurement_aggregate["performance_passed"] \
            or complete["development_only"] is not True \
            or complete["production_promotion_authorized"] is not False:
        raise VerificationError("complete aggregate binding differs")
    if require_production and (
            set(complete["final_source_lease"]) != {"source_tree_digest", "query_binding_digests"}
            or complete["final_source_lease"].get("source_tree_digest") != source["source_tree_digest"]
            or complete["final_source_lease"].get("query_binding_digests") != source["query_binding_digests"]
            or set(complete["final_resident_lease"]) != {"identity_digest", "lease_digest"}
            or complete["final_resident_lease"].get("identity_digest") != resident["identity_digest"]
            or complete["final_resident_lease"].get("lease_digest") != resident["lease"].get("lease_digest")):
        raise VerificationError("final source/resident lease differs")
    def observed_identity(path: Path) -> tuple[int, ...]:
        try: row = path.lstat()
        except OSError as exc: raise VerificationError("terminal identity disappeared") from exc
        if not stat.S_ISREG(row.st_mode):
            raise VerificationError("terminal identity is no longer regular")
        return _identity(row)
    if _tree(root) != (contract.successful_tree(), contract.successful_directories()) \
            or observed_identity(root / "COMPLETE.json") != complete_identity \
            or any(observed_identity(root / path) != identity for path, identity in identities.items()):
        raise VerificationError("terminal tree changed during verification")
    state = {"schema_version": contract.SCHEMAS["verifier_receipt"], "status": "verified",
        "passed": True, "candidate_root": str(root), "candidate_complete_sha256": complete_sha,
        "candidate_complete_digest": complete["complete_digest"],
        "terminal_tree_digest": _digest(manifest),
        "semantic_digest": semantic_aggregate["semantic_digest"],
        "measurement_digest": measurement_aggregate["measurement_digest"], "verified_cases": 60,
        "direct_raw_authority_accessed_by_verifier": False,
        "authority_derived_prerequisite_evidence_accessed_by_verifier": True,
        "forward_outcomes_accessed_by_verifier": False, "development_only": True,
        "production_promotion_authorized": False}
    state["result_digest"] = _digest(state)
    return state


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise VerificationError("verification target exists")
    descriptor, temporary = tempfile.mkstemp(prefix=".all60-verify-", dir=path.parent)
    temp = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(contract.canonical_bytes(value) + b"\n"); handle.flush(); os.fsync(handle.fileno())
        os.link(temp, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        temp.unlink(missing_ok=True)


def publish_verification(candidate_root: Path, verifier_root: Path, *, repository: Path,
                         require_production: bool = True) -> dict[str, Any]:
    _reject_ancestry(candidate_root); _reject_ancestry(verifier_root)
    if verifier_root.exists() or verifier_root.is_symlink():
        raise VerificationError("verifier root must be absent")
    candidate = candidate_root.resolve(strict=True); output = verifier_root.absolute()
    if candidate == output or candidate in output.parents or output in candidate.parents:
        raise VerificationError("candidate/verifier roots overlap")
    if require_production:
        expected_candidate = (repository / contract.CANDIDATE_RELATIVE).resolve(strict=True)
        expected_output = (repository / contract.VERIFIER_RELATIVE).absolute()
        if candidate != expected_candidate or output != expected_output:
            raise VerificationError("production roots differ")
    state = verify_terminal(candidate, repository=repository,
                            require_production=require_production)
    verifier_root.mkdir(parents=False)
    receipt = {**state, "created_at": datetime.now(timezone.utc).isoformat()}
    _keys(receipt, "verifier_receipt")
    _atomic(verifier_root / "VERIFIED.json", receipt)
    return receipt


def dry_replay(candidate_root: Path, *, repository: Path,
               require_production: bool = True) -> dict[str, Any]:
    return verify_terminal(candidate_root, repository=repository,
                           require_production=require_production)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--verifier-root", type=Path, required=True)
    args = parser.parse_args(argv)
    repository = Path(__file__).resolve().parents[2]
    receipt = publish_verification(args.candidate_root, args.verifier_root,
                                   repository=repository, require_production=True)
    print(json.dumps(receipt, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
