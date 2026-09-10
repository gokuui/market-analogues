"""Outcome-blind support pilot for R1-B conditional matching cells."""

from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import os
import shutil
import subprocess
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd

from experiments.m04r import m04r14_r1a_exposure_audit as r1a
from experiments.m04r import audit_m04r14_r1a_exposure_result as r1a_integrity
from market_analogues.adequacy_support import (
    aligned_value, causally_eligible, deterministic_terciles,
    farthest_first_partition, matched_support,
)
from market_analogues.baseline_feature_store import load_feature_generation
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.types import stable_hash


SCHEMA = "m04r14-r1b-support-pilot-v1"
OUTPUT = Path("config/data/analogues/m04r14/r1b-support-pilot-v1")
PREREGISTRATION = Path("experiments/m04r/m04r14_r1b_support_pilot_preregistered.json")
BASELINE_ROOT = Path(
    "config/data/analogues/m04r14/t14-10-wf03-baseline-store-full-v1"
)
BENCHMARK = Path("/home/vinay/code/loser-nasdaq/data/nasdaq/index/IXIC.parquet")
SUPPORT_CAP = 4096
MIN_EPISODE_COVERAGE = 0.90
MIN_LINK_COVERAGE = 0.90
RUNTIME_FILES = (
    "experiments/m04r/m04r14_r1b_support_pilot.py",
    "src/market_analogues/adequacy_support.py",
    "tests/test_adequacy_support.py",
)
KNOWN_INVENTORY = {
    "queries": 3270, "links": 65400,
    "r1a_episode_max_null_p99": 4,
    "cohort_threshold": 5, "cohort_episodes": 369, "cohort_links": 2865,
}
DESIGN = {
    "base_order": ["session_21", "session_42", "session_63"],
    "base_fields": ["session_bin", "query_tier", "liquidity_tertile", "balanced_volatility_rank_tercile", "context_available"],
    "session_bin": "floor(IXIC session ordinal / span), anchored at first IXIC row",
    "volatility": "baseline feature index 2; rank within session-21 bin; value then query-id tie break; balanced terciles",
    "causal_eligibility": "candidate cutoff <= latest eligible; same-symbol candidate cutoff < query start",
    "structure_requested_k": [8, 12, 16],
    "structure_minimum_cell_size": 30,
    "structure_feature_indices": {
        "coarse": [0, 96], "stage_columns_per_block": [0, 1, 2], "structural": [0, 9],
    },
    "structure_scaling": "column median center; IQR scale; IQR below 1e-6 replaced by 1",
    "structure_distance": "squared Euclidean after frozen robust column scaling",
    "structure_partition": "first medoid smallest query id; farthest-first equal-distance tie chooses largest query id; nearest-medoid assignment equal-distance tie chooses earliest selected medoid; undersized merge chooses smallest count then smallest medoid id and targets nearest then smallest medoid id; repeat until every cell size >=30 or one remains; final labels ordered by medoid id",
    "matched_support": "min(4096, product over cells of combination(eligible query count, observed selected-query count)); observed membership must be a duplicate-free subset of causal eligibility",
    "episode_coverage": "episodes with matched support >=4096 divided by all 369 cohort episodes",
    "link_coverage": "observed inbound links attached to supported episodes divided by all 2865 cohort links",
    "minimum_matched_sets": SUPPORT_CAP,
    "minimum_episode_coverage": MIN_EPISODE_COVERAGE,
    "minimum_link_coverage": MIN_LINK_COVERAGE,
    "selection": "first passing base design; then passing structure design with largest retained cells, requested-K tie break",
}
CLAIMS = {
    "support_statistics_opened": True,
    "r1b_test_statistics_opened": False,
    "real_forward_outcomes_accessed": False,
    "adequacy_labels_authorized": False,
    "predictive_claim_authorized": False,
    "production_promotion_authorized": False,
    "passed_meaning": "matching design estimable only",
}


class SupportPilotError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return r1a._load(path)


def _digest(payload: Mapping[str, Any], omitted: set[str]) -> str:
    return stable_hash({key: value for key, value in payload.items() if key not in omitted})


def _locate(episode_ids: Sequence[str], packed_ids: np.ndarray) -> np.ndarray:
    order = np.argsort(packed_ids, kind="stable")
    ordered = packed_ids[order]
    requested = np.asarray([np.void(bytes.fromhex(value)) for value in episode_ids], dtype="V12")
    found = np.searchsorted(ordered, requested)
    if np.any(found >= len(ordered)):
        raise SupportPilotError("one or more query episodes are absent from the packed generation")
    if np.any(ordered[found] != requested):
        raise SupportPilotError("one or more query episodes are absent from the packed generation")
    return order[found]


def _metadata_from_loaded(loaded: Any) -> r1a.Metadata:
    main, overflow = loaded.rows, loaded.overflow
    total = len(main) + len(overflow)
    compact = np.empty(total, dtype=[
        ("episode_id", "V12"), ("cutoff", "<i8"), ("symbol", "<u4"),
    ])
    cursor = 0
    for source in (main, overflow):
        for first in range(0, len(source), 1 << 17):
            block = source[first:first + (1 << 17)]; stop = cursor + len(block)
            compact["episode_id"][cursor:stop] = block["episode_id"]
            compact["cutoff"][cursor:stop] = block["cutoff_ns"]
            compact["symbol"][cursor:stop] = block["symbol_id"]
            cursor = stop
    order = np.lexsort((compact["episode_id"], compact["cutoff"], compact["symbol"]))
    compact = compact[order]
    if len(np.unique(compact["episode_id"])) != total:
        raise SupportPilotError("packed episode identifiers are not unique")
    counts = np.bincount(compact["symbol"], minlength=len(loaded.symbols))
    stops = np.cumsum(counts, dtype=np.int64); starts = stops - counts
    for symbol_id, (first, stop) in enumerate(zip(starts, stops, strict=True)):
        if stop > first and not np.all(compact["symbol"][first:stop] == symbol_id):
            raise SupportPilotError("packed symbol ordering differs")
        if stop - first > 1 and np.any(np.diff(compact["cutoff"][first:stop]) <= 0):
            raise SupportPilotError("packed symbol cutoffs are not strictly ordered")
    return r1a.Metadata(compact["episode_id"], compact["cutoff"], starts, stops, loaded.symbols)


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args), cwd=repository, text=True, capture_output=True, check=False,
    )
    if result.returncode:
        raise SupportPilotError(f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid4().hex}"
    descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        try:
            os.link(temp, path)
        except FileExistsError as error:
            raise SupportPilotError(f"create-only path exists: {path}") from error
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temp.unlink(missing_ok=True)


def _atomic_publish_directory(temp: Path, destination: Path) -> None:
    """Atomically publish a completed directory without replacing a peer result."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.parent / f".{destination.name}.publish.lock"
    descriptor = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if destination.exists() or destination.is_symlink():
            raise SupportPilotError(f"create-only output exists: {destination}")
        temporary_directory = os.open(temp, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(temporary_directory)
        finally:
            os.close(temporary_directory)
        os.rename(temp, destination)
        directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _durable_packed_manifest(repository: Path, generation: str) -> Path:
    path = repository / r1a.PACKED_DURABLE / "generations" / generation / "manifest.json"
    if not path.is_file():
        raise SupportPilotError(f"canonical durable packed manifest is absent: {path}")
    return path


def _select_packed_store(repository: Path, generation: str) -> Path:
    """Use resident bytes only when their manifest is identical to durable authority."""
    durable_manifest = _durable_packed_manifest(repository, generation)
    resident_manifest = r1a.PACKED_RESIDENT / "generations" / generation / "manifest.json"
    if resident_manifest.is_file():
        if _sha(resident_manifest) != _sha(durable_manifest):
            raise SupportPilotError("resident and canonical durable packed manifests differ")
        return r1a.PACKED_RESIDENT
    return repository / r1a.PACKED_DURABLE


def _write_fsynced_text(path: Path, value: str) -> None:
    with path.open("w") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _validate_preregistration(repository: Path, prereg: Mapping[str, Any]) -> None:
    expected_static = {
        "schema_version": "m04r14-r1b-support-pilot-preregistration-v1",
        "execution": {
            "output_root": str(OUTPUT),
            "publication": "flock-serialized create-only atomic directory rename with parent fsync",
        },
        "known_inventory": KNOWN_INVENTORY,
        "design": DESIGN,
        "claims": CLAIMS,
        "prior_exposure": {
            "r1a_graph_and_recurrence_counts_previously_observed": True,
            "terminated_io_preflight_before_support_values": True,
            "support_values_previously_observed": False,
        },
    }
    for key, expected in expected_static.items():
        if prereg.get(key) != expected:
            raise SupportPilotError(f"preregistered {key} contract differs")
    if set(prereg.get("runtime_sha256", {})) != set(RUNTIME_FILES):
        raise SupportPilotError("preregistered runtime file set differs")
    r1a_result = _load(repository / r1a.OUTPUT / "RESULT.json")
    integrity = _load(repository / r1a_integrity.OUTPUT / "VERIFIED.json")
    baseline = _load(repository / BASELINE_ROOT / "RESULT.json")
    expected_inputs = {
        "r1a_result_digest": r1a_result["result_digest"],
        "r1a_integrity_digest": integrity["verification_digest"],
        "r1a_case_manifest_digest": r1a_result["inputs"]["case_manifest_digest"],
        "packed_generation_id": r1a_result["inputs"]["packed_generation_id"],
        "baseline_generation_id": baseline["generation_id"],
    }
    for key, expected in expected_inputs.items():
        if prereg.get("inputs", {}).get(key) != expected:
            raise SupportPilotError(f"preregistered input {key} differs")
    generation = str(r1a_result["inputs"]["packed_generation_id"])
    packed_manifest = _durable_packed_manifest(repository, generation)
    expected_files = {
        str(value.resolve()) for value in (
            repository / r1a.OUTPUT / "RESULT.json",
            repository / r1a_integrity.OUTPUT / "VERIFIED.json",
            repository / r1a.REGISTRY,
            repository / BASELINE_ROOT / "RESULT.json",
            packed_manifest,
            repository / BASELINE_ROOT / "store" / "generations" /
            str(baseline["generation_id"]) / "manifest.json",
            BENCHMARK,
        )
    }
    if set(prereg.get("inputs", {}).get("file_sha256", {})) != expected_files:
        raise SupportPilotError("preregistered input file set differs")


def _require_committed_preregistration(repository: Path, prereg: Mapping[str, Any]) -> str:
    if _git(repository, "status", "--porcelain"):
        raise SupportPilotError("support run requires a clean committed tree")
    relative = PREREGISTRATION.as_posix()
    head = _git(repository, "rev-parse", "HEAD")
    parent = _git(repository, "rev-parse", "HEAD^")
    if parent != prereg.get("implementation_commit"):
        raise SupportPilotError("preregistration is not the sole child of implementation H0")
    changed = _git(repository, "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").splitlines()
    if changed != [relative]:
        raise SupportPilotError("H1 must contain only the preregistration")
    blob = subprocess.run(
        ("git", "show", f"HEAD:{relative}"), cwd=repository,
        capture_output=True, check=False,
    )
    if blob.returncode or blob.stdout != (repository / PREREGISTRATION).read_bytes():
        raise SupportPilotError("working preregistration differs from committed H1")
    return head


def preregister(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(); path = repository / PREREGISTRATION
    if path.exists() or path.is_symlink():
        raise SupportPilotError(f"create-only preregistration exists: {path}")
    if _git(repository, "status", "--porcelain"):
        raise SupportPilotError("preregistration requires a clean implementation commit")
    r1a_result = _load(repository / r1a.OUTPUT / "RESULT.json")
    integrity = _load(repository / r1a_integrity.OUTPUT / "VERIFIED.json")
    baseline_result = _load(repository / BASELINE_ROOT / "RESULT.json")
    generation = str(r1a_result["inputs"]["packed_generation_id"])
    packed_manifest = _durable_packed_manifest(repository, generation)
    input_files = (
        repository / r1a.OUTPUT / "RESULT.json",
        repository / r1a_integrity.OUTPUT / "VERIFIED.json",
        repository / r1a.REGISTRY,
        repository / BASELINE_ROOT / "RESULT.json",
        packed_manifest,
        repository / BASELINE_ROOT / "store" / "generations" /
        str(baseline_result["generation_id"]) / "manifest.json",
        BENCHMARK,
    )
    state = {
        "schema_version": "m04r14-r1b-support-pilot-preregistration-v1",
        "implementation_commit": _git(repository, "rev-parse", "HEAD"),
        "execution": {
            "output_root": str(OUTPUT),
            "publication": "flock-serialized create-only atomic directory rename with parent fsync",
        },
        "inputs": {
            "r1a_result_digest": r1a_result["result_digest"],
            "r1a_integrity_digest": integrity["verification_digest"],
            "r1a_case_manifest_digest": r1a_result["inputs"]["case_manifest_digest"],
            "packed_generation_id": generation,
            "baseline_generation_id": baseline_result["generation_id"],
            "file_sha256": {str(value.resolve()): _sha(value) for value in input_files},
        },
        "runtime_sha256": {value: _sha(repository / value) for value in RUNTIME_FILES},
        "known_inventory": KNOWN_INVENTORY,
        "design": DESIGN,
        "claims": CLAIMS,
        "prior_exposure": {
            "r1a_graph_and_recurrence_counts_previously_observed": True,
            "terminated_io_preflight_before_support_values": True,
            "support_values_previously_observed": False,
        },
    }
    payload = {**state, "preregistration_digest": stable_hash(state)}
    _atomic_create_json(path, payload)
    return payload


def _session_positions(cutoffs: np.ndarray) -> np.ndarray:
    frame = pd.read_parquet(BENCHMARK, columns=["date"])
    timestamps = pd.to_datetime(frame["date"], errors="coerce")
    if timestamps.isna().any():
        raise SupportPilotError("benchmark calendar contains invalid timestamps")
    sessions = np.sort(timestamps.to_numpy(dtype="datetime64[ns]").view(np.int64))
    if len(sessions) < 2 or np.any(np.diff(sessions) <= 0):
        raise SupportPilotError("benchmark sessions are not strictly increasing and unique")
    positions = np.searchsorted(sessions, cutoffs, side="right") - 1
    if np.any(positions < 0) or np.any(sessions[positions] != cutoffs):
        raise SupportPilotError("a query cutoff is absent from benchmark sessions")
    return positions


def _cells(
    span: int | None,
    query_rows: Sequence[Mapping[str, Any]],
    queries: Sequence[r1a.Query],
    session_positions: np.ndarray,
    volatility_terciles: np.ndarray,
) -> tuple[tuple[Any, ...], ...]:
    output = []
    for index, (row, query) in enumerate(zip(query_rows, queries, strict=True)):
        time_cell: Any = query.latest_ns if span is None else int(session_positions[index] // span)
        output.append((
            time_cell,
            str(row["quality_tier"]),
            str(row["liquidity_stratum"]),
            int(volatility_terciles[index]),
            bool(row["context_available"]),
        ))
    return tuple(output)


def _design_result(
    name: str,
    cells: Sequence[Any],
    cohort: Mapping[str, tuple[int, ...]],
    eligible: Mapping[str, tuple[int, ...]],
) -> tuple[dict[str, Any], dict[str, int]]:
    support = {
        episode_id: matched_support(eligible[episode_id], observed, cells, cap=SUPPORT_CAP)
        for episode_id, observed in cohort.items()
    }
    passing = {episode_id for episode_id, value in support.items() if value >= SUPPORT_CAP}
    total_links = sum(len(value) for value in cohort.values())
    supported_links = sum(len(cohort[value]) for value in passing)
    result = {
        "design": name,
        "distinct_cells": len(set(cells)),
        "supported_episodes": len(passing),
        "cohort_episodes": len(cohort),
        "episode_coverage": len(passing) / len(cohort),
        "supported_links": supported_links,
        "cohort_links": total_links,
        "link_coverage": supported_links / total_links,
    }
    result["passes"] = (
        result["episode_coverage"] >= MIN_EPISODE_COVERAGE
        and result["link_coverage"] >= MIN_LINK_COVERAGE
    )
    return result, support


def compute(repository: Path, prereg: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    repository = repository.resolve()
    integrity = _load(repository / r1a_integrity.OUTPUT / "VERIFIED.json")
    if integrity.get("verification_digest") != _digest(
        integrity, {"verification_digest", "created_at"}
    ) or integrity.get("passed") is not True or not all(integrity.get("gates", {}).values()):
        raise SupportPilotError("R1-A full integrity receipt differs")
    r1a_result = _load(repository / r1a.OUTPUT / "RESULT.json")
    if r1a_result.get("result_digest") != _digest(r1a_result, {"result_digest", "elapsed_seconds"}):
        raise SupportPilotError("R1-A result differs")
    if integrity.get("verified_result_digest") != r1a_result["result_digest"]:
        raise SupportPilotError("R1-A result/integrity binding differs")
    if r1a_result["result_digest"] != prereg["inputs"]["r1a_result_digest"] \
            or integrity["verification_digest"] != prereg["inputs"]["r1a_integrity_digest"]:
        raise SupportPilotError("preregistered R1-A authority differs")

    generation = str(r1a_result["inputs"]["packed_generation_id"])
    store = _select_packed_store(repository, generation)
    loaded = load_packed_generation(
        store,
        generation,
        expected_provenance_digest=str(r1a_result["inputs"]["packed_provenance_digest"]),
        verify_content=True,
        validate_records=True,
    )
    metadata = _metadata_from_loaded(loaded)
    registry = _load(repository / r1a.REGISTRY)
    registry_rows = {str(row["episode_id"]): row for row in registry["cases_data"]}
    symbol_ids = {symbol: index for index, symbol in enumerate(metadata.symbols)}

    queries_by_id: dict[str, r1a.Query] = {}
    selected: defaultdict[str, list[str]] = defaultdict(list)
    recurrent_meta: dict[str, tuple[str, int]] = {}
    case_manifest = []
    for path in sorted((repository / r1a.CASES).glob("*.json")):
        case = _load(path)
        query_id = str(case["query_episode_id"])
        registered = registry_rows.get(query_id)
        if registered is None or not all((
            case.get("gate_passed") is True,
            case.get("real_forward_outcomes_accessed") is False,
            case.get("result_digest") == r1a.shadow._deterministic_case_digest(case),
            case.get("checkpoint_integrity_digest") == r1a.shadow._integrity_digest(case),
            case.get("registry_case_id") == registered["case_id"],
            case.get("latest_eligible_cutoff") == registered["latest_eligible_cutoff"],
            len(case.get("matches", [])) == 20,
        )):
            raise SupportPilotError(f"sealed case differs: {path.name}")
        symbol = str(case["query_symbol"])
        query = r1a.Query(
            query_id,
            symbol,
            symbol_ids[symbol],
            int(np.datetime64(case["query_start"], "ns").view(np.int64)),
            int(np.datetime64(case["latest_eligible_cutoff"], "ns").view(np.int64)),
            int(case["certificate"]["eligible_candidates"]),
        )
        queries_by_id[query_id] = query
        for match in case["matches"]:
            episode_id = str(match["episode_id"])
            selected[episode_id].append(query_id)
            recurrent_meta[episode_id] = (
                str(match["symbol"]),
                int(np.datetime64(match["cutoff"], "ns").view(np.int64)),
            )
        case_manifest.append({
            "path": path.relative_to(repository / r1a.CASES.parent).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        })
    if stable_hash(case_manifest) != r1a_result["inputs"]["case_manifest_digest"]:
        raise SupportPilotError("shadow case manifest differs")

    query_ids = sorted(queries_by_id)
    queries = tuple(queries_by_id[value] for value in query_ids)
    query_rows_list = []
    for value in query_ids:
        row = dict(registry_rows[value])
        cutoff = np.datetime64(row["cutoff"], "ns")
        benchmark_prefix = dict(row["benchmark_prefix"])
        stock_prefix = dict(row["stock_prefix"])
        if int(row["lookback"]) != 252 or int(stock_prefix["rows"]) < 252:
            raise SupportPilotError(f"query history contract differs: {value}")
        row["context_available"] = (
            int(benchmark_prefix["rows"]) >= 252
            and np.datetime64(benchmark_prefix["coverage_cutoff"], "ns") >= cutoff
        )
        query_rows_list.append(row)
    query_rows = tuple(query_rows_list)
    query_index = {value: index for index, value in enumerate(query_ids)}
    cohort = {
        episode_id: tuple(sorted(query_index[value] for value in query_ids_selected))
        for episode_id, query_ids_selected in selected.items()
        if len(query_ids_selected) >= 5
    }
    if len(cohort) != 369 or sum(map(len, cohort.values())) != 2865:
        raise SupportPilotError("R1-A recurrent cohort differs")

    eligible: dict[str, tuple[int, ...]] = {}
    for episode_id in cohort:
        symbol, cutoff = recurrent_meta[episode_id]
        eligible[episode_id] = tuple(
            index for index, query in enumerate(queries)
            if causally_eligible(
                candidate_symbol=symbol, candidate_cutoff_ns=cutoff,
                query_symbol=query.symbol, query_start_ns=query.start_ns,
                latest_eligible_ns=query.latest_ns,
            )
        )
        if not set(cohort[episode_id]).issubset(eligible[episode_id]):
            raise SupportPilotError(f"selected queries exceed causal risk set: {episode_id}")

    baseline_result = _load(repository / BASELINE_ROOT / "RESULT.json")
    if baseline_result.get("result_digest") != _digest(baseline_result, {"result_digest"}) \
            or baseline_result.get("passed") is not True \
            or not all(baseline_result.get("gates", {}).values()) \
            or baseline_result.get("outcomes_or_labels_used") is not False:
        raise SupportPilotError("baseline feature authority differs")
    feature_generation = load_feature_generation(
        repository / BASELINE_ROOT / "store",
        str(baseline_result["generation_id"]),
        packed_manifest=loaded.manifest,
        verify_content=True,
    )
    packed_ids = np.concatenate((loaded.rows["episode_id"], loaded.overflow["episode_id"]))
    query_locations = _locate(query_ids, packed_ids)
    main_count = len(loaded.rows)
    volatility = np.asarray([
        aligned_value(
            feature_generation.rows["values"], feature_generation.overflow["values"],
            int(location),
        )[2]
        for location in query_locations
    ], dtype=np.float64)
    query_overflow_count = int(np.sum(query_locations >= main_count))
    missing_volatility_count = int(np.sum(~np.isfinite(volatility)))
    missing_context_count = sum(not bool(row["context_available"]) for row in query_rows)
    query_cutoffs = np.asarray([
        int(np.datetime64(row["cutoff"], "ns").view(np.int64)) for row in query_rows
    ], dtype=np.int64)
    positions = _session_positions(query_cutoffs)
    bin21 = positions // 21
    volatility_terciles = np.full(len(queries), -1, dtype=np.int8)
    for value in np.unique(bin21):
        members = np.flatnonzero(bin21 == value)
        volatility_terciles[members] = deterministic_terciles(
            volatility[members], [query_ids[index] for index in members]
        )

    base_designs = []
    base_cells: dict[str, tuple[tuple[Any, ...], ...]] = {}
    base_support: dict[str, dict[str, int]] = {}
    for name, span in (("session_21", 21), ("session_42", 42), ("session_63", 63)):
        cells = _cells(span, query_rows, queries, positions, volatility_terciles)
        design, support = _design_result(name, cells, cohort, eligible)
        base_designs.append(design); base_cells[name] = cells; base_support[name] = support
    selected_base = next((row["design"] for row in base_designs if row["passes"]), None)

    # Primary N1 support partition deliberately excludes benchmark and relative
    # channels: coarse[0:96] is close/ATR/volume; every fourth stage value is
    # relative return and is excluded. Presence bitfields are not metric values.
    structure = np.zeros((len(queries), 96 + 36 + 9), dtype=np.float64)
    for out_index, location in enumerate(query_locations):
        if location < main_count:
            row = loaded.rows[int(location)]
            stage = row["stage"].astype(np.float64).reshape(12, 4)[:, :3].ravel()
            structure[out_index] = np.r_[
                row["coarse"][:96].astype(np.float64),
                stage,
                row["structural"].astype(np.float64),
            ]
    center = np.median(structure, axis=0)
    scale = np.percentile(structure, 75, axis=0) - np.percentile(structure, 25, axis=0)
    scale[scale < 1e-6] = 1.0
    structure = (structure - center) / scale
    structure_designs = []; structure_support: dict[int, dict[str, int]] = {}
    selected_structure = None; selected_structure_retained = None
    partitions: dict[int, np.ndarray] = {}
    medoid_ids: dict[int, tuple[str, ...]] = {}
    if selected_base is not None:
        for k in (8, 12, 16):
            labels, medoids = farthest_first_partition(
                structure, query_ids, clusters=k, minimum_size=30,
            )
            partitions[k] = labels
            cells = tuple(
                (*base_cells[selected_base][index], int(labels[index]))
                for index in range(len(queries))
            )
            design, support = _design_result(f"{selected_base}_structure_{k}", cells, cohort, eligible)
            structure_support[k] = support
            design["requested_structure_cells"] = k
            design["retained_structure_cells"] = len(medoids)
            medoid_ids[k] = tuple(query_ids[index] for index in medoids)
            structure_designs.append(design)
        supported = [row for row in structure_designs if row["passes"]]
        if supported:
            selected_row = max(
                supported,
                key=lambda row: (row["retained_structure_cells"], row["requested_structure_cells"]),
            )
            selected_structure = selected_row["requested_structure_cells"]
            selected_structure_retained = selected_row["retained_structure_cells"]

    cohort_rows = []
    for episode_id in sorted(cohort, key=lambda value: (-len(cohort[value]), value)):
        row = {
            "episode_id": episode_id,
            "symbol": recurrent_meta[episode_id][0],
            "cutoff": np.datetime_as_string(np.datetime64(recurrent_meta[episode_id][1], "ns")),
            "observed_inbound_queries": len(cohort[episode_id]),
            "causally_eligible_queries": len(eligible[episode_id]),
            "base_support": {name: values[episode_id] for name, values in base_support.items()},
            "structure_support": {
                str(k): values[episode_id] for k, values in structure_support.items()
            },
        }
        cohort_rows.append(row)

    inputs = {
        "r1a_result_digest": r1a_result["result_digest"],
        "r1a_integrity_digest": integrity["verification_digest"],
        "shadow_registry_digest": registry["registry_digest"],
        "shadow_case_manifest_digest": stable_hash(case_manifest),
        "packed_generation_id": generation,
        "baseline_generation_id": baseline_result["generation_id"],
        "benchmark_sha256": _sha(BENCHMARK),
    }
    permitted_files = {
        str((repository / r1a_integrity.OUTPUT / "VERIFIED.json").resolve()): _sha(
            repository / r1a_integrity.OUTPUT / "VERIFIED.json"
        ),
        str((repository / r1a.OUTPUT / "RESULT.json").resolve()): _sha(
            repository / r1a.OUTPUT / "RESULT.json"
        ),
        str((repository / r1a.REGISTRY).resolve()): _sha(repository / r1a.REGISTRY),
        str((repository / BASELINE_ROOT / "RESULT.json").resolve()): _sha(
            repository / BASELINE_ROOT / "RESULT.json"
        ),
        str(_durable_packed_manifest(repository, generation).resolve()): _sha(
            _durable_packed_manifest(repository, generation)
        ),
        str((repository / BASELINE_ROOT / "store" / "generations" /
             str(baseline_result["generation_id"]) / "manifest.json").resolve()): _sha(
            repository / BASELINE_ROOT / "store" / "generations" /
            str(baseline_result["generation_id"]) / "manifest.json"
        ),
        str(BENCHMARK.resolve()): _sha(BENCHMARK),
    }
    query_cell_rows = [{
        "query_episode_id": query_id,
        "base_cells": {name: list(base_cells[name][index]) for name in base_cells},
        "structure_cells": {str(k): int(labels[index]) for k, labels in partitions.items()},
    } for index, query_id in enumerate(query_ids)]
    state = {
        "schema_version": SCHEMA,
        "status": "support_only_complete",
        "passed": selected_base is not None and selected_structure is not None,
        "passed_meaning": "matching design estimable only",
        "preregistration_digest": prereg["preregistration_digest"],
        "support_statistics_opened": True,
        "r1b_test_statistics_opened": False,
        "real_forward_outcomes_accessed": False,
        "adequacy_labels_authorized": False,
        "predictive_claim_authorized": False,
        "production_promotion_authorized": False,
        "inputs": inputs,
        "permitted_input_contract": {
            "authority_manifest_files": permitted_files,
            "shadow_case_files": len(case_manifest),
            "shadow_case_manifest_digest": stable_hash(case_manifest),
            "packed_binary_content_verified_from_manifest": True,
            "baseline_binary_content_verified_from_manifest": True,
            "forbidden_roots": [
                "t14-09 evidence/outcome/card stores",
                "wf03d outcome/prediction stores",
                "wf04 evaluation stores",
                "t14-11 Stockbee stores",
                "t14-12 post-signal stores",
            ],
        },
        "inventory": {
            "queries": len(queries),
            "selected_links": sum(len(value) for value in selected.values()),
            "recurrent_episode_threshold": 5,
            "cohort_episodes": len(cohort),
            "cohort_links": sum(map(len, cohort.values())),
            "query_overflow_rows": query_overflow_count,
            "missing_volatility_queries": missing_volatility_count,
            "missing_context_queries": missing_context_count,
        },
        "support_contract": {
            "minimum_matched_sets": SUPPORT_CAP,
            "minimum_episode_coverage": MIN_EPISODE_COVERAGE,
            "minimum_link_coverage": MIN_LINK_COVERAGE,
            "base_design_order": list(prereg["design"]["base_order"]),
            "calendar_bin_formula": "floor(benchmark_session_ordinal / span), anchored at first IXIC row",
            "volatility_cells": "balanced deterministic within-session21-bin rank terciles; value then query-id tie break",
            "history_fixed": "lookback=252 and stock-prefix rows>=252 for every query",
            "context_availability_reconstructed": True,
            "structure_view": "market-excluded packed coarse[0:96] + stage columns net/volatility/volume + structural; robust column scaling",
            "structure_options": [8, 12, 16],
            "minimum_structure_cell_size": 30,
        },
        "base_designs": base_designs,
        "selected_base_design": selected_base,
        "structure_designs": structure_designs,
        "selected_structure_cells": selected_structure,
        "selected_retained_structure_cells": selected_structure_retained,
        "structure_medoid_query_ids": {str(key): list(value) for key, value in medoid_ids.items()},
        "query_cell_assignments_digest": stable_hash(query_cell_rows),
        "remaining_before_r1b_statistics": [
            "independent support-pilot reconstruction",
            "freeze B0 statistic and shared-priority contracts",
        ],
        "gates": {
            "r1a_full_integrity_bound": True,
            "shadow_cases_reconstructed": True,
            "causal_membership_reconstructed": True,
            "selected_subset_of_eligibility": True,
            "packed_and_baseline_content_verified": True,
            "cohort_threshold_fixed_from_r1a_null_p99": True,
            "support_only_no_scientific_statistics": True,
            "query_covariates_complete": (
                query_overflow_count == 0
                and missing_volatility_count == 0
                and missing_context_count == 0
            ),
            "history_and_context_contract_reconstructed": True,
            "outcomes_excluded_by_permitted_input_contract": True,
            "base_support_decision_available": selected_base is not None,
            "structure_support_decision_available": selected_structure is not None,
        },
    }
    state["gates"]["cohort_threshold_fixed_from_r1a_null_p99"] = (
        float(r1a_result["comparison"]["episode_max"]["null_p99"]) == 4.0
        and int(prereg["known_inventory"]["cohort_threshold"]) == 5
    )
    state["passed"] = all(state["gates"].values())
    state["status"] = "support_only_complete" if state["passed"] else "insufficient_null_support"
    return {**state, "result_digest": stable_hash(state)}, cohort_rows, query_cell_rows


def _html(result: Mapping[str, Any]) -> str:
    rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{:.1%}</td><td>{:.1%}</td><td>{}</td></tr>".format(
            escape(str(row["design"])), row["distinct_cells"], row["episode_coverage"],
            row["link_coverage"], "PASS" if row["passes"] else "insufficient",
        ) for row in result["base_designs"]
    )
    structure_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{:.1%}</td><td>{:.1%}</td><td>{}</td></tr>".format(
            escape(str(row["design"])), row["distinct_cells"], row["episode_coverage"],
            row["link_coverage"], "PASS" if row["passes"] else "insufficient",
        ) for row in result["structure_designs"]
    )
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><title>R1-B support pilot</title><style>body{{font-family:system-ui;max-width:1000px;margin:2rem auto}}table{{border-collapse:collapse;width:100%}}td,th{{padding:.5rem;border-bottom:1px solid #ccc;text-align:left}}.note{{background:#fff5d8;padding:1rem}}</style></head><body><h1>R1-B support-only pilot</h1><p><b>Status:</b> {result['status']}. Selected base design: <code>{result['selected_base_design']}</code>; selected structure K: <code>{result['selected_structure_cells']}</code>.</p><h2>Base support</h2><table><thead><tr><th>Design</th><th>Cells</th><th>Episode coverage</th><th>Link coverage</th><th>Gate</th></tr></thead><tbody>{rows}</tbody></table><h2>Structure-conditioned support</h2><table><thead><tr><th>Design</th><th>Cells</th><th>Episode coverage</th><th>Link coverage</th><th>Gate</th></tr></thead><tbody>{structure_rows}</tbody></table><p class='note'><b>Boundary:</b> this pilot opened and reports matching-support statistics only. Stored distance fields were present in sealed case JSON bytes but were not consumed or summarized. It did not compute R1-B distance/cohesion/specificity test statistics, forward outcomes, predictive evidence or adequacy labels.</p></body></html>"""


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(); output = repository / OUTPUT
    if output.exists() or output.is_symlink():
        raise SupportPilotError(f"create-only output exists: {output}")
    prereg = _load(repository / PREREGISTRATION)
    if prereg.get("preregistration_digest") != _digest(prereg, {"preregistration_digest"}):
        raise SupportPilotError("preregistration digest differs")
    _validate_preregistration(repository, prereg)
    preregistration_commit = _require_committed_preregistration(repository, prereg)
    for path, expected in prereg["inputs"]["file_sha256"].items():
        if _sha(Path(path)) != expected:
            raise SupportPilotError(f"preregistered input changed: {path}")
    for path, expected in prereg["runtime_sha256"].items():
        if _sha(repository / path) != expected:
            raise SupportPilotError(f"preregistered runtime changed: {path}")
    result, cohort, query_cells = compute(repository, prereg)
    result["preregistration_commit"] = preregistration_commit
    temp = output.parent / f".{output.name}.tmp-{os.getpid()}-{uuid4().hex}"
    try:
        temp.mkdir(parents=True)
        _write_fsynced_text(
            temp / "COHORT_SUPPORT.json",
            json.dumps(cohort, indent=2, sort_keys=True) + "\n",
        )
        _write_fsynced_text(
            temp / "QUERY_CELLS.json",
            json.dumps(query_cells, indent=2, sort_keys=True) + "\n",
        )
        result["cohort_support_sha256"] = _sha(temp / "COHORT_SUPPORT.json")
        result["query_cells_sha256"] = _sha(temp / "QUERY_CELLS.json")
        deterministic = {key: value for key, value in result.items() if key != "result_digest"}
        result["result_digest"] = stable_hash(deterministic)
        _write_fsynced_text(
            temp / "RESULT.json", json.dumps(result, indent=2, sort_keys=True) + "\n",
        )
        _write_fsynced_text(temp / "report.html", _html(result))
        _atomic_publish_directory(temp, output)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True); raise
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("preregister", "run")); parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    result = preregister(args.repository) if args.mode == "preregister" else execute(args.repository)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
