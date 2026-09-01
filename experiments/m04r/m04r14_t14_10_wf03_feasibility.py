"""Durable, outcome-blind WF-03 certified-retrieval feasibility probes.

The harness deliberately accepts no outcome or authority path.  It freezes three
walk-forward queries, scans the resident packed-bound store in both directions,
runs exact certified completion, and publishes create-only receipts.  A rerun
keeps interrupted attempts and executes only cases without a valid terminal.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import stat
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Any, Mapping, Sequence

import pandas as pd

from market_analogues.adapters import source_from_spec
from market_analogues.certified_packed_search import certified_packed_search
from market_analogues.config import load_config
from market_analogues.episodes import build_episode
from market_analogues.packed_bound_search import (
    BRANCH_AWARE_SEARCH_SCHEMA_VERSION,
    BoundProposal,
    BoundProposalReport,
    PackedBoundQuery,
    _packed_query_input_digest,
    bound_proposal_candidate_digest,
    packed_bound_search_contract,
    scan_packed_bound_proposals_threaded,
)
from market_analogues.representation import represent
from market_analogues.resident_store import (
    observe_ready_strict,
    resident_file_identity_lease,
)
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import InstrumentKey, SearchQuery, stable_hash


SCHEMA = "m04r14-t14-10-wf03-feasibility-preregistration-v1"
RUN_SCHEMA = "m04r14-t14-10-wf03-feasibility-run-v1"
CASE_SCHEMA = "m04r14-t14-10-wf03-feasibility-case-v1"
ATTEMPT_SCHEMA = "m04r14-t14-10-wf03-feasibility-attempt-v1"
ROOT_COMPLETE_SCHEMA = "m04r14-t14-10-wf03-feasibility-complete-v1"
GENERATION_ID = "9fc6ae0ec4451133d8006897162f3443803fd30a3c44b78e80a476fbb18bb483"
PROVENANCE_DIGEST = "83ccfa62ac7ffec03e48d0f8a5634c7b8f1b8b0dde1426be22cc343fe116f62d"
REGISTRY_DIGEST = "784e771be69fe77a8684af6b56815ba701c90164400a2f93e53fa835a5da82c4"
REGISTRY_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-walk-forward-query-registry-v1"
)
REGISTRY_FILE = REGISTRY_RELATIVE / "walk-forward-query-registry.json"
REGISTRY_VERIFICATION_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-walk-forward-query-registry-v1-verification/VERIFIED.json"
)
SCORING_VERIFICATION_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-walk-forward-synthetic-scoring-v1/VERIFIED.json"
)
CONFIG_RELATIVE = Path("config/datasets.example.yaml")
RESIDENT_ROOT = Path("/dev/shm/market-analogues/m04r11-candidate-v2") / GENERATION_ID
OUTPUT_RELATIVE = Path("config/data/analogues/m04r14/t14-10-wf03-feasibility-v1")
PREREGISTRATION_RELATIVE = Path(
    "experiments/m04r/m04r14_t14_10_wf03_feasibility_preregistered.json"
)
PROBES = (
    ("early", "a69def453340e01048a52284", "OHAI", "2014-03-31T00:00:00"),
    ("middle", "4ebfb91d87892612d998e71e", "ACGLP", "2020-08-31T00:00:00"),
    ("late", "b923b64e7f3b36fe2a8d0643", "FRME", "2025-07-31T00:00:00"),
)
PROPOSAL_QUOTA = 16_385
PROPOSAL_THREADS = 8
INITIAL_FRONTIER = 1_000
MAXIMUM_FRONTIER = 16_384
SEED_ROWS = 512
BLOCK_ROWS = 4_096
REVERSE_BLOCK_ROWS = 4_097
EXACT_WORKERS = 1
TOP_K = 20
MAX_PER_INSTRUMENT = 1
MINIMUM_HISTORY_GAP = 60
TOLERANCE = 1e-12
RUNTIME_FILES = (
    "experiments/m04r/m04r14_t14_10_wf03_feasibility.py",
    "src/market_analogues/certified_packed_search.py",
    "src/market_analogues/packed_bound_search.py",
    "src/market_analogues/packed_bound_store.py",
    "src/market_analogues/distance.py",
    "src/market_analogues/episodes.py",
    "src/market_analogues/representation.py",
    "src/market_analogues/resident_store.py",
    "src/market_analogues/search.py",
)


class FeasibilityError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strict_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode() + b"\n"
    except (TypeError, ValueError) as exc:
        raise FeasibilityError("evidence is not strict finite JSON") from exc


def _pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in rows:
        if key in result:
            raise FeasibilityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FeasibilityError(f"regular JSON file required: {path}")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise FeasibilityError(f"regular JSON file required: {path}")
        raw = bytearray()
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            raw.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns,
                            row.st_ctime_ns, row.st_mode)
    if identity(before) != identity(after):
        raise FeasibilityError(f"JSON identity changed while reading: {path}")
    try:
        value = json.loads(bytes(raw), object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                FeasibilityError(f"non-finite JSON token: {token}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FeasibilityError(f"invalid JSON: {path}") from exc
    if type(value) is not dict:
        raise FeasibilityError(f"JSON object required: {path}")
    return value


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FeasibilityError(f"create-only target exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".wf03-", dir=path.parent)
    temp = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_strict_bytes(dict(value)))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temp.unlink(missing_ok=True)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _sealed(value: Mapping[str, Any], field: str = "result_digest") -> dict[str, Any]:
    state = dict(value)
    if field in state:
        raise FeasibilityError(f"seal field already exists: {field}")
    return {**state, field: stable_hash(state)}


def _validate_seal(value: Mapping[str, Any], field: str = "result_digest") -> None:
    if type(value) is not dict or value.get(field) != stable_hash({
        key: item for key, item in value.items() if key != field
    }):
        raise FeasibilityError(f"{field} differs")


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repository, text=True,
                            capture_output=True, check=False)
    if result.returncode:
        raise FeasibilityError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _resource() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    status: dict[str, int] = {"VmRSS": 0, "VmHWM": 0, "VmSwap": 0}
    for line in Path("/proc/self/status").read_text().splitlines():
        key = line.split(":", 1)[0]
        if key in status:
            status[key] = int(line.split()[1])
    return {
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "max_rss_mb": float(usage.ru_maxrss) / 1024.0,
        "rss_kib": status["VmRSS"], "hwm_kib": status["VmHWM"],
        "swap_kib": status["VmSwap"],
        "user_cpu_seconds": float(usage.ru_utime),
        "system_cpu_seconds": float(usage.ru_stime),
    }


def _registry(repository: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    state = _read(repository / REGISTRY_FILE)
    deterministic = {key: value for key, value in state.items() if key != "registry_digest"}
    rows = state.get("queries_data")
    if not all((
        state.get("registry_digest") == REGISTRY_DIGEST,
        state.get("registry_digest") == stable_hash(deterministic),
        state.get("status") == "sealed", state.get("passed") is True,
        state.get("historical_walk_forward_query_outcomes_opened") is False,
        state.get("final_period_result_opened") is False,
        type(rows) is list, len(rows) == 3_936,
    )):
        raise FeasibilityError("walk-forward registry identity/boundary differs")
    by_id = {row.get("episode_id"): row for row in rows if type(row) is dict}
    if len(by_id) != len(rows):
        raise FeasibilityError("walk-forward query IDs are not unique")
    for label, query_id, symbol, cutoff in PROBES:
        row = by_id.get(query_id)
        if row is None or (row.get("symbol"), row.get("cutoff"), row.get("scored")) != (
                symbol, cutoff, True):
            raise FeasibilityError(f"frozen {label} probe differs")
    return state, by_id


def _resident() -> dict[str, Any]:
    ready_path = RESIDENT_ROOT / "READY.json"
    observation = observe_ready_strict(ready_path)
    lease = resident_file_identity_lease(ready_path)
    content = observation["payload"]["content"]
    if content.get("generation_id") != GENERATION_ID \
            or content.get("provenance_digest") != PROVENANCE_DIGEST \
            or lease.get("ready_digest") != observation["ready_digest"]:
        raise FeasibilityError("resident generation/provenance/lease differs")
    state = {
        "root": str(RESIDENT_ROOT.resolve()),
        "store_root": str((RESIDENT_ROOT / "store").resolve()),
        "ready_digest": observation["ready_digest"],
        "ready_file_sha256": observation["ready_file_sha256"],
        "content_digest": observation["content_digest"],
        "seal_digest": observation["seal_digest"],
        "lease": lease,
    }
    return _sealed(state, "identity_digest")


def _proposal_payload(report: BoundProposalReport) -> dict[str, Any]:
    return {
        "schema_version": report.schema_version,
        "generation_id": report.generation_id,
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
        "route_quotas": dict(report.route_quotas),
        "block_rows": report.block_rows, "block_order": report.block_order,
        "elapsed_seconds": report.elapsed_seconds, "peak_rss_mb": report.peak_rss_mb,
        "candidate_digest": report.candidate_digest,
        "result_digest": report.result_digest,
        "contract_digest": report.contract_digest, "input_digest": report.input_digest,
    }


def _proposal_report(value: Mapping[str, Any]) -> BoundProposalReport:
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
    if type(value) is not dict or set(value) != required \
            or type(value["candidates"]) is not list \
            or any(type(value[key]) is not int or isinstance(value[key], bool)
                   or value[key] < 0 for key in (
                       "rows_scanned", "eligible_rows", "eligible_main_rows",
                       "eligible_overflow_rows", "block_rows")) \
            or any(type(value[key]) not in {int, float} or isinstance(value[key], bool)
                   or not math.isfinite(value[key]) or value[key] < 0
                   for key in ("elapsed_seconds", "peak_rss_mb")):
        raise FeasibilityError("proposal JSON fields differ")
    for row in value["candidates"]:
        if type(row) is not dict or set(row) != candidate_keys \
                or type(row["episode_id"]) is not str or type(row["symbol"]) is not str \
                or type(row["cutoff_ns"]) is not int or isinstance(row["cutoff_ns"], bool) \
                or type(row["quality_tier"]) is not str \
                or type(row["lower_bound_hex"]) is not str \
                or row["routes"] != ["composite"] \
                or type(row["overflow_fallback"]) is not bool:
            raise FeasibilityError("proposal candidate JSON fields differ")
    candidates = tuple(BoundProposal(
        row["episode_id"], row["symbol"], row["cutoff_ns"], row["quality_tier"],
        float.fromhex(row["lower_bound_hex"]), tuple(row["routes"]),
        row["overflow_fallback"],
    ) for row in value["candidates"])
    return BoundProposalReport(
        value["schema_version"], value["generation_id"], value["query_episode_id"],
        candidates, value["rows_scanned"], value["eligible_rows"],
        value["eligible_main_rows"], value["eligible_overflow_rows"],
        dict(value["route_counts"]), dict(value["route_quotas"]),
        value["block_rows"], value["block_order"], value["elapsed_seconds"],
        value["peak_rss_mb"], value["candidate_digest"], value["result_digest"],
        value["contract_digest"], value["input_digest"],
    )


def _validate_proposal(value: Mapping[str, Any], query: PackedBoundQuery) -> BoundProposalReport:
    try:
        report = _proposal_report(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise FeasibilityError("proposal JSON encoding differs") from exc
    deterministic = {
        "schema_version": BRANCH_AWARE_SEARCH_SCHEMA_VERSION,
        "contract_digest": packed_bound_search_contract(branch_aware=True)["digest"],
        "generation_id": GENERATION_ID, "query_episode_id": query.episode_id,
        "rows_scanned": report.rows_scanned, "eligible_rows": report.eligible_rows,
        "eligible_main_rows": report.eligible_main_rows,
        "eligible_overflow_rows": report.eligible_overflow_rows,
        "route_counts": report.route_counts, "route_quotas": report.route_quotas,
        "candidate_digest": report.candidate_digest,
        "real_forward_outcomes_accessed": False,
        "input_digest": _packed_query_input_digest(query),
    }
    if not all((
        report.schema_version == BRANCH_AWARE_SEARCH_SCHEMA_VERSION,
        report.contract_digest == deterministic["contract_digest"],
        report.generation_id == GENERATION_ID,
        report.query_episode_id == query.episode_id,
        report.input_digest == deterministic["input_digest"],
        report.route_quotas == {"composite": PROPOSAL_QUOTA},
        report.route_counts == {"composite": len(report.candidates)},
        report.eligible_rows == report.eligible_main_rows + report.eligible_overflow_rows,
        len(report.candidates) == min(PROPOSAL_QUOTA, report.eligible_rows),
        all(row.routes == ("composite",) and math.isfinite(row.lower_bound)
            and row.lower_bound >= 0 for row in report.candidates),
        list(report.candidates) == sorted(
            report.candidates, key=lambda row: (row.lower_bound, row.episode_id)),
        report.candidate_digest == bound_proposal_candidate_digest(report.candidates),
        report.result_digest == stable_hash(deterministic),
    )):
        raise FeasibilityError("proposal reconstruction differs")
    return report


def _proposal_semantics(value: Mapping[str, Any]) -> dict[str, Any]:
    omitted = {"block_rows", "block_order", "elapsed_seconds", "peak_rss_mb"}
    return {key: item for key, item in value.items() if key not in omitted}


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


def _load_m13(repository: Path) -> Any:
    path = repository / "experiments/m04r/m04r13_threaded_certified_exposed.py"
    spec = importlib.util.spec_from_file_location("wf03_m13_validator", path)
    if spec is None or spec.loader is None:
        raise FeasibilityError("certified result validator is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _context(repository: Path, row: Mapping[str, Any]) -> tuple[Any, Any, SearchQuery, PackedBoundQuery]:
    source = source_from_spec(load_config(repository / CONFIG_RELATIVE).datasets["nasdaq"])
    episode = build_episode(source, InstrumentKey("nasdaq", row["symbol"]),
                            row["cutoff"], row["lookback"], row["representation_version"])
    request = SearchQuery(episode.key, ("nasdaq",), ("A", "B"), TOP_K,
                          False, True, MAX_PER_INSTRUMENT, MINIMUM_HISTORY_GAP)
    packed = PackedBoundQuery(
        episode.key.id, episode.key.instrument.source_symbol,
        int(pd.Timestamp(episode.bars.timestamp.iloc[0]).value),
        int(latest_eligible_cutoff(episode, MINIMUM_HISTORY_GAP).value),
        represent(episode), request.quality_tiers,
    )
    if episode.key.id != row["episode_id"]:
        raise FeasibilityError("query episode reconstruction differs")
    return source, episode, request, packed


def _next_attempt(case_root: Path) -> Path:
    attempts = case_root / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    ordinals: list[int] = []
    for path in attempts.iterdir():
        if path.is_dir() and not path.is_symlink() and path.name.startswith("attempt-"):
            try:
                ordinals.append(int(path.name.removeprefix("attempt-")))
            except ValueError:
                raise FeasibilityError("malformed attempt directory")
    target = attempts / f"attempt-{max(ordinals, default=0) + 1:04d}"
    target.mkdir()
    return target


def _case_complete_valid(case_root: Path, query_id: str) -> dict[str, Any] | None:
    terminal = case_root / "COMPLETE.json"
    if not terminal.exists():
        return None
    value = _read(terminal)
    _validate_seal(value, "complete_digest")
    relative = value.get("attempt_relative")
    if type(relative) is not str or not relative.startswith("attempts/attempt-") \
            or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise FeasibilityError("case attempt reference differs")
    attempt = case_root / relative
    attempt_terminal = _read(attempt / "COMPLETE.json")
    _validate_seal(attempt_terminal, "complete_digest")
    manifest = attempt_terminal.get("leaf_manifest")
    if type(manifest) is not list or attempt_terminal.get("leaf_manifest_digest") != stable_hash(manifest):
        raise FeasibilityError("attempt leaf manifest differs")
    for row in manifest:
        if type(row) is not dict or set(row) != {"path", "sha256"} \
                or type(row["path"]) is not str or Path(row["path"]).name != row["path"] \
                or _sha(attempt / row["path"]) != row["sha256"]:
            raise FeasibilityError("attempt leaf differs")
    if value.get("schema_version") != CASE_SCHEMA or value.get("query_id") != query_id \
            or value.get("status") != "complete" \
            or attempt_terminal.get("status") != "complete" \
            or attempt_terminal.get("query_id") != query_id \
            or attempt_terminal.get("complete_digest") != value.get("attempt_complete_digest") \
            or _sha(attempt / "COMPLETE.json") != value.get("attempt_complete_sha256"):
        raise FeasibilityError("case terminal differs")
    return value


def validate_terminal(repository: Path, preregistration: Mapping[str, Any]) -> dict[str, Any]:
    prereg = validate_preregistration(repository, preregistration)
    root = repository / OUTPUT_RELATIVE
    if _read(root / "CONTRACT.json") != prereg:
        raise FeasibilityError("terminal contract differs")
    started = _read(root / "RUN_STARTED.json")
    _validate_seal(started)
    if started.get("schema_version") != RUN_SCHEMA or started.get("status") != "running" \
            or started.get("preregistration_digest") != prereg["preregistration_digest"]:
        raise FeasibilityError("run-started receipt differs")
    cases: list[dict[str, Any]] = []
    for ordinal, (label, query_id, _symbol, _cutoff) in enumerate(PROBES):
        case_root = root / "cases" / f"{ordinal:03d}-{label}-{query_id}"
        value = _case_complete_valid(case_root, query_id)
        if value is None:
            raise FeasibilityError("terminal has an incomplete case")
        cases.append(value)
    complete = _read(root / "COMPLETE.json")
    _validate_seal(complete, "complete_digest")
    expected_manifest = [{
        "path": f"cases/{ordinal:03d}-{label}-{query_id}/COMPLETE.json",
        "sha256": _sha(root / f"cases/{ordinal:03d}-{label}-{query_id}/COMPLETE.json"),
        "complete_digest": cases[ordinal]["complete_digest"],
    } for ordinal, (label, query_id, _symbol, _cutoff) in enumerate(PROBES)]
    if not all((
        complete.get("schema_version") == ROOT_COMPLETE_SCHEMA,
        complete.get("status") == "complete", complete.get("cases") == len(PROBES),
        complete.get("all_certified") is True, complete.get("proposal_parity") is True,
        complete.get("preregistration_digest") == prereg["preregistration_digest"],
        complete.get("case_manifest") == expected_manifest,
        complete.get("case_manifest_digest") == stable_hash(expected_manifest),
        complete.get("historical_walk_forward_query_outcomes_opened") is False,
        complete.get("final_period_result_opened") is False,
        complete.get("production_promotion_authorized") is False,
    )):
        raise FeasibilityError("root terminal reconstruction differs")
    return complete


def _run_case(repository: Path, root: Path, ordinal: int, label: str,
              row: Mapping[str, Any], resident: Mapping[str, Any]) -> dict[str, Any]:
    query_id = str(row["episode_id"])
    case_root = root / "cases" / f"{ordinal:03d}-{label}-{query_id}"
    case_root.mkdir(parents=True, exist_ok=True)
    prior = _case_complete_valid(case_root, query_id)
    if prior is not None:
        return prior
    attempt = _next_attempt(case_root)
    started = perf_counter()
    _atomic(attempt / "RUN_STARTED.json", _sealed({
        "schema_version": ATTEMPT_SCHEMA, "status": "running",
        "query_id": query_id, "case_id": row["case_id"], "label": label,
        "resident_identity_digest": resident["identity_digest"],
        "created_at": _now(),
    }))
    try:
        before = _resource()
        source, episode, request, packed = _context(repository, row)
        m13 = _load_m13(repository)
        binding_before = m13.query_binding(source, episode, request, packed, PROVENANCE_DIGEST)
        lease_before = resident_file_identity_lease(RESIDENT_ROOT / "READY.json")
        forward = scan_packed_bound_proposals_threaded(
            Path(resident["store_root"]), GENERATION_ID, packed,
            route_quotas={"composite": PROPOSAL_QUOTA}, block_rows=BLOCK_ROWS,
            block_order="forward", threads=PROPOSAL_THREADS, branch_aware=True,
            verify_content=False, expected_provenance_digest=PROVENANCE_DIGEST,
        )
        forward_payload = _proposal_payload(forward)
        _validate_proposal(forward_payload, packed)
        _atomic(attempt / "PROPOSAL_FORWARD.json", _sealed({
            "proposal": forward_payload, "lease_before": lease_before["lease_digest"],
            "lease_after": resident_file_identity_lease(
                RESIDENT_ROOT / "READY.json")["lease_digest"], "created_at": _now(),
        }))
        reverse = scan_packed_bound_proposals_threaded(
            Path(resident["store_root"]), GENERATION_ID, packed,
            route_quotas={"composite": PROPOSAL_QUOTA}, block_rows=REVERSE_BLOCK_ROWS,
            block_order="reverse", threads=PROPOSAL_THREADS, branch_aware=True,
            verify_content=False, expected_provenance_digest=PROVENANCE_DIGEST,
        )
        reverse_payload = _proposal_payload(reverse)
        _validate_proposal(reverse_payload, packed)
        if _proposal_semantics(forward_payload) != _proposal_semantics(reverse_payload):
            raise FeasibilityError("forward/reverse proposal semantics differ")
        _atomic(attempt / "PROPOSAL_REVERSE.json", _sealed({
            "proposal": reverse_payload, "forward_candidate_digest": forward.candidate_digest,
            "lease_after": resident_file_identity_lease(
                RESIDENT_ROOT / "READY.json")["lease_digest"], "created_at": _now(),
        }))
        exact_started = perf_counter()
        result = certified_packed_search(
            episode, source, request, Path(resident["store_root"]), GENERATION_ID,
            store_dataset_id="nasdaq", initial_frontier_rows=INITIAL_FRONTIER,
            maximum_frontier_rows=MAXIMUM_FRONTIER, seed_rows=SEED_ROWS,
            block_rows=BLOCK_ROWS, workers=EXACT_WORKERS, tolerance=TOLERANCE,
            verify_content=False, requested_positions=True, vector_lower_bounds=True,
            deferred_alignments=True, compact_scored=True, native_bound_deferral=True,
            streaming_threshold_closure=True, branch_aware_packed_bounds=True,
            precomputed_proposal=forward,
        )
        exact_seconds = perf_counter() - exact_started
        # Canonicalize tuple-bearing dataclasses before the strict raw-JSON
        # validator sees them; this is also exactly what the durable leaf stores.
        certificate = json.loads(json.dumps(asdict(result.certificate), allow_nan=False))
        matches = json.loads(json.dumps(
            [_match(value) for value in result.matches], allow_nan=False,
        ))
        m13.validate_certificate_and_matches(
            certificate, matches, query_id,
            expected_input_digest=binding_before["certified_input_digest"],
        )
        if len({item["symbol"] for item in matches}) != TOP_K:
            raise FeasibilityError("exact result violates one-match-per-symbol contract")
        binding_after = m13.query_binding(source, episode, request, packed, PROVENANCE_DIGEST)
        if binding_after != binding_before:
            raise FeasibilityError("query/source binding changed during case")
        final_lease = resident_file_identity_lease(RESIDENT_ROOT / "READY.json")
        if final_lease["lease_digest"] != lease_before["lease_digest"]:
            raise FeasibilityError("resident identity changed during case")
        _atomic(attempt / "EXACT.json", _sealed({
            "query_binding": binding_before, "certificate": certificate,
            "matches": matches, "exact_wall_seconds": exact_seconds,
            "resident_lease_digest": final_lease["lease_digest"], "created_at": _now(),
        }))
        after = _resource()
        if after["swap_kib"] != 0:
            raise FeasibilityError("process-attributed swap is nonzero")
        manifest = [{"path": name, "sha256": _sha(attempt / name)} for name in (
            "RUN_STARTED.json", "PROPOSAL_FORWARD.json", "PROPOSAL_REVERSE.json", "EXACT.json"
        )]
        complete = _sealed({
            "schema_version": ATTEMPT_SCHEMA, "status": "complete", "query_id": query_id,
            "case_id": row["case_id"], "label": label,
            "proposal_parity": True, "certified": True,
            "distinct_match_symbols": len({item["symbol"] for item in matches}),
            "elapsed_seconds": perf_counter() - started,
            "stage_seconds": {"forward": forward.elapsed_seconds,
                              "reverse": reverse.elapsed_seconds, "exact": exact_seconds},
            "resources_before": before, "resources_after": after,
            "leaf_manifest": manifest, "leaf_manifest_digest": stable_hash(manifest),
            "created_at": _now(),
        }, "complete_digest")
        _atomic(attempt / "COMPLETE.json", complete)
        relative = attempt.relative_to(case_root).as_posix()
        case_complete = _sealed({
            "schema_version": CASE_SCHEMA, "status": "complete", "query_id": query_id,
            "case_id": row["case_id"], "label": label, "attempt_relative": relative,
            "attempt_complete_sha256": _sha(attempt / "COMPLETE.json"),
            "attempt_complete_digest": complete["complete_digest"], "created_at": _now(),
        }, "complete_digest")
        _atomic(case_root / "COMPLETE.json", case_complete)
        return case_complete
    except BaseException as exc:
        failed = attempt / "FAILED.json"
        if not failed.exists():
            _atomic(failed, _sealed({
                "schema_version": ATTEMPT_SCHEMA, "status": "failed",
                "query_id": query_id, "exception_type": type(exc).__name__,
                "exception_message": str(exc), "elapsed_seconds": perf_counter() - started,
                "created_at": _now(),
            }, "failure_digest"))
        raise


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain"):
        raise FeasibilityError("preregistration requires a clean implementation commit")
    head = _git(repository, "rev-parse", "HEAD")
    registry, by_id = _registry(repository)
    verification = _read(repository / REGISTRY_VERIFICATION_RELATIVE)
    scoring = _read(repository / SCORING_VERIFICATION_RELATIVE)
    if verification.get("passed") is not True or verification.get("registry_digest") != REGISTRY_DIGEST \
            or scoring.get("passed") is not True:
        raise FeasibilityError("WF-01/WF-02 verification prerequisite differs")
    state = {
        "schema_version": SCHEMA, "status": "frozen_before_durable_probe_results",
        "implementation_commit": head,
        "runtime_files": {path: _sha(repository / path) for path in RUNTIME_FILES},
        "inputs": {
            "config_path": str((repository / CONFIG_RELATIVE).resolve()),
            "config_sha256": _sha(repository / CONFIG_RELATIVE),
            "registry_path": str((repository / REGISTRY_FILE).resolve()),
            "registry_sha256": _sha(repository / REGISTRY_FILE),
            "registry_digest": registry["registry_digest"],
            "registry_verification_sha256": _sha(repository / REGISTRY_VERIFICATION_RELATIVE),
            "scoring_verification_sha256": _sha(repository / SCORING_VERIFICATION_RELATIVE),
            "generation_id": GENERATION_ID, "provenance_digest": PROVENANCE_DIGEST,
            "resident_root": str(RESIDENT_ROOT.resolve()),
        },
        "probes": [{"label": label, "query_id": query_id,
                    "registry_case": by_id[query_id]}
                   for label, query_id, _symbol, _cutoff in PROBES],
        "execution": {
            "proposal_quota": PROPOSAL_QUOTA, "proposal_threads": PROPOSAL_THREADS,
            "forward": {"block_rows": BLOCK_ROWS, "block_order": "forward"},
            "reverse": {"block_rows": REVERSE_BLOCK_ROWS, "block_order": "reverse"},
            "top_k": TOP_K, "max_per_instrument": MAX_PER_INSTRUMENT,
            "minimum_history_gap_bars": MINIMUM_HISTORY_GAP,
            "initial_frontier_rows": INITIAL_FRONTIER,
            "maximum_frontier_rows": MAXIMUM_FRONTIER, "seed_rows": SEED_ROWS,
            "exact_workers": EXACT_WORKERS, "tolerance_hex": TOLERANCE.hex(),
            "requested_positions": True, "vector_lower_bounds": True,
            "deferred_alignments": True, "compact_scored": True,
            "native_bound_deferral": True, "streaming_threshold_closure": True,
            "branch_aware_packed_bounds": True,
            "output_root": str((repository / OUTPUT_RELATIVE).resolve()),
            "resume": "retain attempts and skip only strictly validated complete cases",
        },
        "claims": {
            "historical_query_retrieval_opened": True,
            "historical_walk_forward_query_outcomes_opened": False,
            "final_period_result_opened": False,
            "authority_or_outcome_paths_accepted": False,
            "development_only": True, "production_promotion_authorized": False,
        },
    }
    return _sealed(state, "preregistration_digest")


def validate_preregistration(repository: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    _validate_seal(value, "preregistration_digest")
    if value.get("schema_version") != SCHEMA \
            or value.get("status") != "frozen_before_durable_probe_results" \
            or value.get("inputs", {}).get("registry_digest") != REGISTRY_DIGEST \
            or value.get("claims", {}).get("historical_walk_forward_query_outcomes_opened") is not False \
            or value.get("claims", {}).get("final_period_result_opened") is not False:
        raise FeasibilityError("preregistration boundary differs")
    head = value.get("implementation_commit")
    if type(head) is not str:
        raise FeasibilityError("implementation commit is absent")
    _git(repository, "merge-base", "--is-ancestor", head, "HEAD")
    for relative, digest in value.get("runtime_files", {}).items():
        if _sha(repository / relative) != digest:
            raise FeasibilityError(f"runtime file differs from preregistration: {relative}")
        blob = subprocess.run(["git", "show", f"{head}:{relative}"], cwd=repository,
                              capture_output=True, check=False)
        if blob.returncode or sha256(blob.stdout).hexdigest() != digest:
            raise FeasibilityError(f"implementation commit binding differs: {relative}")
    _registry(repository)
    for relative, key in ((CONFIG_RELATIVE, "config_sha256"),
                          (REGISTRY_FILE, "registry_sha256"),
                          (REGISTRY_VERIFICATION_RELATIVE, "registry_verification_sha256"),
                          (SCORING_VERIFICATION_RELATIVE, "scoring_verification_sha256")):
        if _sha(repository / relative) != value["inputs"][key]:
            raise FeasibilityError(f"frozen input differs: {relative}")
    return dict(value)


def execute(repository: Path, preregistration: Mapping[str, Any]) -> dict[str, Any]:
    prereg = validate_preregistration(repository, preregistration)
    _registry_state, by_id = _registry(repository)
    resident = _resident()
    if len(os.sched_getaffinity(0)) < PROPOSAL_THREADS:
        raise FeasibilityError("WF-03 proposal requires eight available CPUs")
    root = repository / OUTPUT_RELATIVE
    root.mkdir(parents=True, exist_ok=True)
    contract_path = root / "CONTRACT.json"
    if contract_path.exists():
        if _read(contract_path) != prereg:
            raise FeasibilityError("existing run contract differs")
    else:
        _atomic(contract_path, prereg)
    run_started_path = root / "RUN_STARTED.json"
    if not run_started_path.exists():
        _atomic(root / "RUN_STARTED.json", _sealed({
            "schema_version": RUN_SCHEMA, "status": "running",
            "preregistration_digest": prereg["preregistration_digest"],
            "resident": resident, "resource": _resource(), "created_at": _now(),
        }))
    else:
        run_started = _read(run_started_path)
        _validate_seal(run_started)
        if run_started.get("preregistration_digest") != prereg["preregistration_digest"]:
            raise FeasibilityError("existing run-started receipt differs")
    terminal = root / "COMPLETE.json"
    if terminal.exists():
        return validate_terminal(repository, prereg)
    cases = []
    for ordinal, (label, query_id, _symbol, _cutoff) in enumerate(PROBES):
        cases.append(_run_case(repository, root, ordinal, label, by_id[query_id], resident))
    current_resident = _resident()
    if current_resident["identity_digest"] != resident["identity_digest"]:
        raise FeasibilityError("resident identity changed across feasibility run")
    case_manifest = [{
        "path": f"cases/{ordinal:03d}-{label}-{query_id}/COMPLETE.json",
        "sha256": _sha(root / f"cases/{ordinal:03d}-{label}-{query_id}/COMPLETE.json"),
        "complete_digest": cases[ordinal]["complete_digest"],
    } for ordinal, (label, query_id, _symbol, _cutoff) in enumerate(PROBES)]
    durations = []
    for ordinal, row in enumerate(cases):
        attempt = root / "cases" / f"{ordinal:03d}-{row['label']}-{row['query_id']}" / row["attempt_relative"]
        durations.append(_read(attempt / "COMPLETE.json")["elapsed_seconds"])
    complete = _sealed({
        "schema_version": ROOT_COMPLETE_SCHEMA, "status": "complete",
        "preregistration_digest": prereg["preregistration_digest"],
        "cases": len(cases), "all_certified": True, "proposal_parity": True,
        "case_manifest": case_manifest, "case_manifest_digest": stable_hash(case_manifest),
        "observed_case_seconds": durations,
        "estimated_serial_3936_hours": sum(durations) / len(durations) * 3_936 / 3_600,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
        "final_resident_identity_digest": current_resident["identity_digest"],
        "created_at": _now(),
    }, "complete_digest")
    _atomic(terminal, complete)
    return validate_terminal(repository, prereg)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preregister", "run", "validate"))
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv)
    repository = args.repository.resolve(strict=True)
    prereg_path = repository / PREREGISTRATION_RELATIVE
    if args.mode == "preregister":
        _atomic(prereg_path, build_preregistration(repository))
        return 0
    prereg = _read(prereg_path)
    if args.mode == "validate":
        terminal = validate_terminal(repository, prereg)
        print(json.dumps(terminal, indent=2, sort_keys=True))
        return 0
    result = execute(repository, prereg)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
