"""Independent post-publication reconstruction of the R1-B support-only pilot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
from html import escape
from html.parser import HTMLParser
import json
from math import comb
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd

from experiments.m04r import m04r14_shadow_run as shadow
from market_analogues.adapters import source_from_spec
from market_analogues.baseline_neighbors import recent_return_volatility_at_positions
from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.config import load_config
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.representation import represent, representation_input_digest
from market_analogues.types import Episode, EpisodeKey, InstrumentKey, stable_hash


SCHEMA = "m04r14-r1b-support-result-integrity-verification-v1"
OUTPUT = Path(
    "config/data/analogues/m04r14/"
    "r1b-support-pilot-v2-integrity-verification-v1"
)
PRODUCER_OUTPUT = Path("config/data/analogues/m04r14/r1b-support-pilot-v2")
PREREGISTRATION = Path("experiments/m04r/m04r14_r1b_support_pilot_v2_preregistered.json")
V1_PREREGISTRATION = Path("experiments/m04r/m04r14_r1b_support_pilot_preregistered.json")
V1_OUTPUT = Path("config/data/analogues/m04r14/r1b-support-pilot-v1")
R1A_OUTPUT = Path("config/data/analogues/m04r14/r1a-exposure-audit-v2")
R1A_INTEGRITY = Path(
    "config/data/analogues/m04r14/r1a-exposure-audit-v2-integrity-verification-v1"
)
REGISTRY = Path("config/data/analogues/m04r14/nasdaq-shadow-denominator-v1/query-registry.json")
CASES = Path("config/data/analogues/m04r14/nasdaq-shadow-snapshot-v1/cases")
PACKED_DURABLE = Path("config/data/analogues/poc/m04r/packed-bound-full/store")
PACKED_RESIDENT = Path(
    "/dev/shm/market-analogues/m04r11-candidate-v2/"
    "9fc6ae0ec4451133d8006897162f3443803fd30a3c44b78e80a476fbb18bb483/store"
)
CONFIG = Path("config/datasets.example.yaml")
BENCHMARK = Path("/home/vinay/code/loser-nasdaq/data/nasdaq/index/IXIC.parquet")
PRODUCER_SCHEMA = "m04r14-r1b-support-pilot-v2"
SUPPORT_CAP = 4096
MIN_EPISODE_COVERAGE = 0.90
MIN_LINK_COVERAGE = 0.90
STRUCTURE_OPTIONS = (8, 12, 16)
PRODUCER_RUNTIME_FILES = (
    "experiments/m04r/m04r14_r1b_support_pilot.py",
    "src/market_analogues/adequacy_support.py",
    "tests/test_adequacy_support.py",
)
VERIFIER_RUNTIME_FILES = (
    "experiments/m04r/audit_m04r14_r1b_support_result.py",
    "tests/test_r1b_support_result_audit.py",
)
KNOWN_INVENTORY = {
    "queries": 3270, "links": 65400, "r1a_episode_max_null_p99": 4,
    "cohort_threshold": 5, "cohort_episodes": 369, "cohort_links": 2865,
    "v1_query_ids_present_in_historical_pack": 9,
    "v1_query_ids_absent_from_historical_pack": 3261,
}
V1_HISTORY = {
    "implementation_commit": "d3087a9b0b0605e71ec7d68dd922acade659f7bb",
    "preregistration_commit": "e30e9eff2225bd747ea68e394cbe7298d726cc5a",
    "preregistration_digest": "5aa93df5e7416c01e9a999a2fdf7622236788e6a824e00794d2fcd511112c0e0",
    "failure_stage": "query-to-historical-pack identity lookup before matched-support computation",
    "output_absent": True,
}
DESIGN = {
    "base_order": ["session_21", "session_42", "session_63"],
    "base_fields": [
        "session_bin", "query_tier", "liquidity_tertile",
        "balanced_volatility_rank_tercile", "context_available",
    ],
    "session_bin": "floor(IXIC session ordinal / span), anchored at first IXIC row",
    "volatility": "baseline recent_return_volatility_at_positions output index 2 reconstructed from each exact causal query window; rank within session-21 bin; value then query-id tie break; balanced terciles",
    "causal_eligibility": "candidate cutoff <= latest eligible; same-symbol candidate cutoff < query start",
    "structure_requested_k": [8, 12, 16],
    "structure_minimum_cell_size": 30,
    "structure_feature_indices": {
        "coarse": [0, 96], "stage_columns_per_block": [0, 1, 2],
        "structural": [0, 9],
    },
    "structure_source": "exact unquantized distance-v1 representation reconstructed from each registered causal query window; never substitute a historical packed row",
    "structure_scaling": "column median center; IQR scale; IQR below 1e-6 replaced by 1",
    "structure_distance": "squared Euclidean after frozen robust column scaling",
    "structure_partition": "first medoid smallest query id; farthest-first equal-distance tie chooses largest query id; nearest-medoid assignment equal-distance tie chooses earliest selected medoid; undersized merge chooses smallest count then smallest medoid id and targets nearest then smallest medoid id; repeat until every cell size >=30 or one remains; final labels ordered by medoid id",
    "matched_support": "min(4096, product over cells of combination(eligible query count, observed selected-query count)); observed membership must be a duplicate-free subset of causal eligibility",
    "episode_coverage": "episodes with matched support >=4096 divided by all 369 cohort episodes",
    "link_coverage": "observed inbound links attached to supported episodes divided by all 2865 cohort links",
    "minimum_matched_sets": 4096,
    "minimum_episode_coverage": 0.9,
    "minimum_link_coverage": 0.9,
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


@dataclass(frozen=True)
class Query:
    episode_id: str
    symbol: str
    symbol_id: int
    start_ns: int
    latest_ns: int
    eligible_count: int


@dataclass(frozen=True)
class Metadata:
    episode_ids: np.ndarray
    cutoffs: np.ndarray
    symbol_ids: np.ndarray
    starts: np.ndarray
    stops: np.ndarray
    symbols: tuple[str, ...]


class SupportIntegrityError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SupportIntegrityError(message)


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, expected_type: type) -> Any:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise SupportIntegrityError(f"duplicate JSON key {key}: {path}")
            result[key] = value
        return result

    value = json.loads(
        path.read_bytes(), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            SupportIntegrityError(f"non-finite JSON value {token}: {path}")
        ),
    )
    if not isinstance(value, expected_type):
        raise SupportIntegrityError(f"JSON {expected_type.__name__} required: {path}")
    if expected_type is list and not all(isinstance(row, dict) for row in value):
        raise SupportIntegrityError(f"JSON object list required: {path}")
    return value


def _load_object(path: Path) -> dict[str, Any]:
    return _load_json(path, dict)


def _load_list(path: Path) -> list[dict[str, Any]]:
    return _load_json(path, list)


def _digest(payload: Mapping[str, Any], omitted: set[str]) -> str:
    return stable_hash({key: value for key, value in payload.items() if key not in omitted})


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args), cwd=repository, text=True, capture_output=True, check=False,
    )
    if result.returncode:
        raise SupportIntegrityError(f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _require_clean_committed_files(repository: Path, paths: Sequence[str]) -> str:
    status = _git(repository, "status", "--porcelain", "--untracked-files=all")
    _require(not status, "verifier execution requires a clean committed tree")
    head = _git(repository, "rev-parse", "HEAD")
    for relative in paths:
        working = repository / relative
        _require(working.is_file() and not working.is_symlink(), f"runtime file differs: {relative}")
        blob = subprocess.run(
            ("git", "show", f"{head}:{relative}"), cwd=repository,
            capture_output=True, check=False,
        )
        _require(
            blob.returncode == 0 and blob.stdout == working.read_bytes(),
            f"runtime file is not bound to verifier commit: {relative}",
        )
    return head


def _validate_v1_history(repository: Path) -> None:
    prereg = _load_object(repository / V1_PREREGISTRATION)
    _require(all((
        prereg.get("preregistration_digest") == _digest(prereg, {"preregistration_digest"}),
        prereg.get("preregistration_digest") == V1_HISTORY["preregistration_digest"],
        prereg.get("implementation_commit") == V1_HISTORY["implementation_commit"],
        _git(repository, "rev-parse", f"{V1_HISTORY['preregistration_commit']}^")
            == V1_HISTORY["implementation_commit"],
        _git(
            repository, "diff-tree", "--no-commit-id", "--name-only", "-r",
            V1_HISTORY["preregistration_commit"],
        ) == V1_PREREGISTRATION.as_posix(),
        not (repository / V1_OUTPUT).exists(),
        not (repository / V1_OUTPUT).is_symlink(),
    )), "v1 preregistration history differs")
    committed = subprocess.run(
        ("git", "show", f"{V1_HISTORY['preregistration_commit']}:{V1_PREREGISTRATION}"),
        cwd=repository, capture_output=True, check=False,
    )
    _require(
        committed.returncode == 0
        and committed.stdout == (repository / V1_PREREGISTRATION).read_bytes(),
        "v1 committed preregistration bytes differ",
    )


def _select_packed_store(repository: Path, generation: str) -> Path:
    durable_manifest = repository / PACKED_DURABLE / "generations" / generation / "manifest.json"
    _require(durable_manifest.is_file() and not durable_manifest.is_symlink(), "durable manifest absent")
    resident_manifest = PACKED_RESIDENT / "generations" / generation / "manifest.json"
    if resident_manifest.is_file():
        _require(_sha(resident_manifest) == _sha(durable_manifest), "resident manifest differs")
        return PACKED_RESIDENT
    return repository / PACKED_DURABLE


def _design_result(
    name: str,
    cells: Sequence[Any],
    cohort: Mapping[str, tuple[int, ...]],
    eligible: Mapping[str, tuple[int, ...]],
) -> tuple[dict[str, Any], dict[str, int]]:
    support = {
        episode_id: _matched_support(
            eligible[episode_id], observed, cells, cap=SUPPORT_CAP,
        )
        for episode_id, observed in cohort.items()
    }
    passing = {key for key, value in support.items() if value >= SUPPORT_CAP}
    links = sum(len(value) for value in cohort.values())
    supported_links = sum(len(cohort[key]) for key in passing)
    result = {
        "design": name,
        "distinct_cells": len(set(cells)),
        "supported_episodes": len(passing),
        "cohort_episodes": len(cohort),
        "episode_coverage": len(passing) / len(cohort),
        "supported_links": supported_links,
        "cohort_links": links,
        "link_coverage": supported_links / links,
    }
    result["passes"] = (
        result["episode_coverage"] >= MIN_EPISODE_COVERAGE
        and result["link_coverage"] >= MIN_LINK_COVERAGE
    )
    return result, support


def _atomic_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid4().hex}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise SupportIntegrityError(f"create-only receipt exists: {path}") from error
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_publish_directory(temporary: Path, destination: Path) -> None:
    """Serialize the no-overwrite rename and make its parent durable."""
    lock_path = destination.parent / f".{destination.name}.publish.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if destination.exists() or destination.is_symlink():
            raise SupportIntegrityError(f"create-only output exists: {destination}")
        temporary_descriptor = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(temporary_descriptor)
        finally:
            os.close(temporary_descriptor)
        os.rename(temporary, destination)
        parent = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _causally_eligible(
    candidate_symbol: str, candidate_cutoff_ns: int,
    query_symbol: str, query_start_ns: int, latest_eligible_ns: int,
) -> bool:
    return candidate_cutoff_ns <= latest_eligible_ns and (
        candidate_symbol != query_symbol or candidate_cutoff_ns < query_start_ns
    )


def _matched_support(
    eligible_indices: Sequence[int], observed_indices: Sequence[int],
    cells: Sequence[Any], *, cap: int,
) -> int:
    eligible_values = tuple(map(int, eligible_indices))
    observed_values = tuple(map(int, observed_indices))
    _require(cap >= 1, "support cap must be positive")
    _require(
        len(set(eligible_values)) == len(eligible_values)
        and len(set(observed_values)) == len(observed_values),
        "query membership contains duplicates",
    )
    _require(set(observed_values).issubset(eligible_values), "observed membership is not eligible")
    _require(
        all(0 <= index < len(cells) for index in (*eligible_values, *observed_values)),
        "query membership index is outside cells",
    )
    available = Counter(cells[index] for index in eligible_values)
    selected = Counter(cells[index] for index in observed_values)
    support = 1
    for cell, count in selected.items():
        population = available.get(cell, 0)
        if population < count:
            return 0
        support *= comb(population, count)
        if support >= cap:
            return cap
    return support


def _deterministic_terciles(values: Sequence[float], ids: Sequence[str]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    _require(
        array.ndim == 1 and len(array) == len(ids) and len(set(ids)) == len(ids),
        "tercile inputs differ",
    )
    output = np.full(len(array), -1, dtype=np.int8)
    valid = [index for index, value in enumerate(array) if np.isfinite(value)]
    for rank, index in enumerate(sorted(valid, key=lambda item: (array[item], ids[item]))):
        output[index] = min(2, rank * 3 // max(len(valid), 1))
    return output


def _farthest_first_partition(
    vectors: np.ndarray, ids: Sequence[str], *, clusters: int, minimum_size: int,
) -> tuple[np.ndarray, tuple[int, ...]]:
    values = np.asarray(vectors, dtype=np.float64)
    _require(
        values.ndim == 2 and len(values) == len(ids) and len(set(ids)) == len(ids),
        "partition inputs differ",
    )
    _require(1 <= clusters <= len(values) and minimum_size >= 1, "partition limits differ")
    _require(np.isfinite(values).all(), "partition vectors are non-finite")
    first = min(range(len(ids)), key=lambda index: ids[index])
    medoids = [first]
    nearest = np.sum((values - values[first]) ** 2, axis=1)
    for _ in range(1, clusters):
        available = (index for index in range(len(values)) if index not in medoids)
        selected = max(available, key=lambda index: (nearest[index], ids[index]))
        medoids.append(selected)
        nearest = np.minimum(nearest, np.sum((values - values[selected]) ** 2, axis=1))
    distances = np.column_stack([
        np.sum((values - values[index]) ** 2, axis=1) for index in medoids
    ])
    labels = np.argmin(distances, axis=1).astype(np.int32)
    active = set(range(len(medoids)))
    while len(active) > 1:
        counts = Counter(map(int, labels))
        small = [cell for cell in sorted(active) if counts.get(cell, 0) < minimum_size]
        if not small:
            break
        cell = min(small, key=lambda value: (counts.get(value, 0), ids[medoids[value]]))
        target = min(
            active - {cell},
            key=lambda value: (
                float(np.sum((values[medoids[cell]] - values[medoids[value]]) ** 2)),
                ids[medoids[value]],
            ),
        )
        labels[labels == cell] = target
        active.remove(cell)
    ordered = sorted(active, key=lambda value: ids[medoids[value]])
    remap = {old: new for new, old in enumerate(ordered)}
    return np.asarray([remap[int(value)] for value in labels], dtype=np.int32), tuple(
        medoids[value] for value in ordered
    )


def _metadata_from_loaded(loaded: Any) -> Metadata:
    total = len(loaded.rows) + len(loaded.overflow)
    compact = np.empty(total, dtype=[
        ("episode_id", "V12"), ("cutoff", "<i8"), ("symbol", "<u4"),
    ])
    cursor = 0
    for source in (loaded.rows, loaded.overflow):
        for first in range(0, len(source), 1 << 17):
            block = source[first:first + (1 << 17)]
            stop = cursor + len(block)
            compact["episode_id"][cursor:stop] = block["episode_id"]
            compact["cutoff"][cursor:stop] = block["cutoff_ns"]
            compact["symbol"][cursor:stop] = block["symbol_id"]
            cursor = stop
    order = np.lexsort((compact["episode_id"], compact["cutoff"], compact["symbol"]))
    compact = compact[order]
    _require(len(np.unique(compact["episode_id"])) == total, "packed episode IDs are not unique")
    counts = np.bincount(compact["symbol"], minlength=len(loaded.symbols))
    stops = np.cumsum(counts, dtype=np.int64)
    starts = stops - counts
    for symbol_id, (first, stop) in enumerate(zip(starts, stops, strict=True)):
        _require(
            stop == first or np.all(compact["symbol"][first:stop] == symbol_id),
            "packed symbol ordering differs",
        )
        _require(
            stop - first <= 1 or np.all(np.diff(compact["cutoff"][first:stop]) > 0),
            "packed symbol cutoffs are not strictly increasing",
        )
    return Metadata(
        compact["episode_id"], compact["cutoff"], compact["symbol"],
        starts, stops, tuple(loaded.symbols),
    )


def _pool(metadata: Metadata, latest_ns: int) -> tuple[np.ndarray, int]:
    counts = np.empty(len(metadata.symbols), dtype=np.int64)
    for symbol_id, (first, stop) in enumerate(zip(metadata.starts, metadata.stops, strict=True)):
        counts[symbol_id] = np.searchsorted(
            metadata.cutoffs[first:stop], latest_ns, side="right",
        )
    return np.cumsum(counts, dtype=np.int64), int(counts.sum())


def _verify_risk_set(
    metadata: Metadata, pool_cumulative: np.ndarray, pool_total: int, query: Query,
) -> None:
    first = int(metadata.starts[query.symbol_id])
    before = int(pool_cumulative[query.symbol_id - 1]) if query.symbol_id else 0
    own = int(pool_cumulative[query.symbol_id]) - before
    allowed_own = int(np.searchsorted(
        metadata.cutoffs[first:first + own], query.start_ns, side="left",
    ))
    _require(
        pool_total - (own - allowed_own) == query.eligible_count,
        f"risk-set count differs: {query.episode_id}",
    )


def _session_positions(cutoffs: np.ndarray) -> np.ndarray:
    frame = pd.read_parquet(BENCHMARK, columns=["date"])
    timestamps = pd.to_datetime(frame["date"], errors="coerce")
    _require(not timestamps.isna().any(), "benchmark calendar has invalid timestamps")
    sessions = np.sort(timestamps.to_numpy(dtype="datetime64[ns]").view(np.int64))
    _require(len(sessions) >= 2 and np.all(np.diff(sessions) > 0), "benchmark sessions differ")
    positions = np.searchsorted(sessions, cutoffs, side="right") - 1
    _require(
        np.all(positions >= 0) and np.all(sessions[positions] == cutoffs),
        "query cutoff is absent from benchmark sessions",
    )
    return positions


def _cells(
    span: int, query_rows: Sequence[Mapping[str, Any]], queries: Sequence[Query],
    positions: np.ndarray, volatility_terciles: np.ndarray,
) -> tuple[tuple[Any, ...], ...]:
    return tuple((
        int(positions[index] // span), str(row["quality_tier"]),
        str(row["liquidity_stratum"]), int(volatility_terciles[index]),
        bool(row["context_available"]),
    ) for index, (row, _query) in enumerate(zip(query_rows, queries, strict=True)))


def _reconstruct_query_feature(
    query_id: str, row: Mapping[str, Any], stock: pd.DataFrame, benchmark: pd.DataFrame,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    cutoff = pd.Timestamp(str(row["cutoff"]))
    stock_prefix = asdict(causal_prefix_digest(stock, cutoff))
    benchmark_prefix = asdict(causal_prefix_digest(benchmark, cutoff))
    _require(
        stock_prefix == row["stock_prefix"] and benchmark_prefix == row["benchmark_prefix"],
        f"query causal prefix differs: {query_id}",
    )
    eligible = stock.loc[pd.to_datetime(stock["timestamp"]) <= cutoff]
    lookback = int(row["lookback"])
    _require(len(eligible) >= min(lookback, 126), f"query history differs: {query_id}")
    window = eligible.tail(lookback).copy().reset_index(drop=True)
    actual_cutoff = pd.Timestamp(window["timestamp"].iloc[-1])
    context = benchmark.loc[pd.to_datetime(benchmark["timestamp"]) <= actual_cutoff].copy()
    episode = Episode(
        EpisodeKey(
            InstrumentKey("nasdaq", str(row["symbol"])), actual_cutoff,
            lookback, str(row["representation_version"]),
        ),
        window, context, str(row["quality_tier"]),
    )
    _require(episode.key.id == query_id, f"query identity differs: {query_id}")
    representation = represent(episode)
    stage = representation.stage.astype(np.float64).reshape(12, 4)[:, :3].ravel()
    vector = np.r_[
        representation.coarse[:96].astype(np.float64), stage,
        representation.structural.astype(np.float64),
    ]
    volatility = float(recent_return_volatility_at_positions(
        episode.bars["close"].to_numpy(dtype=np.float64),
        np.asarray([len(episode.bars) - 1], dtype=np.int64),
    )[0, 2])
    return vector, volatility, {
        "query_episode_id": query_id,
        "query_representation_digest": representation_input_digest(representation),
        "stock_prefix_digest": stock_prefix["digest"],
        "benchmark_prefix_digest": benchmark_prefix["digest"],
        "volatility_hex": volatility.hex() if np.isfinite(volatility) else None,
    }


def _reconstruct_query_features(
    repository: Path, query_ids: Sequence[str], query_rows: Sequence[Mapping[str, Any]],
    *, workers: int,
) -> tuple[np.ndarray, np.ndarray, tuple[dict[str, Any], ...]]:
    _require(workers >= 1, "query feature workers must be positive")
    config = load_config(repository / CONFIG)
    spec = config.datasets.get("nasdaq")
    _require(
        spec is not None and spec.benchmark is not None
        and spec.benchmark.path.resolve() == BENCHMARK.resolve(),
        "NASDAQ source/benchmark configuration differs",
    )
    source = source_from_spec(spec)
    benchmark = source.load_benchmark()
    _require(benchmark is not None, "NASDAQ benchmark is absent")

    def one(item: tuple[str, Mapping[str, Any]]) -> tuple[np.ndarray, float, dict[str, Any]]:
        query_id, row = item
        stock = source.load(InstrumentKey("nasdaq", str(row["symbol"])))
        return _reconstruct_query_feature(query_id, row, stock, benchmark)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        reconstructed = tuple(executor.map(one, zip(query_ids, query_rows, strict=True)))
    vectors = np.asarray([value[0] for value in reconstructed], dtype=np.float64)
    volatility = np.asarray([value[1] for value in reconstructed], dtype=np.float64)
    audits = tuple(value[2] for value in reconstructed)
    _require(
        vectors.shape == (len(query_ids), 141) and np.isfinite(vectors).all(),
        "query structure reconstruction differs",
    )
    return vectors, volatility, audits


def _expected_html(result: Mapping[str, Any]) -> str:
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


def _structure_nontrivial(
    selected: Mapping[str, Any], base: Mapping[str, Any], labels: Sequence[int],
) -> bool:
    return (
        int(selected["retained_structure_cells"]) >= 2
        and len(set(map(int, labels))) >= 2
        and int(selected["distinct_cells"]) > int(base["distinct_cells"])
    )


def execute(repository: Path, *, workers: int = 12) -> dict[str, Any]:
    repository = repository.resolve()
    verifier_commit = _require_clean_committed_files(repository, VERIFIER_RUNTIME_FILES)
    output = repository / OUTPUT
    if output.exists() or output.is_symlink():
        raise SupportIntegrityError(f"create-only output exists: {output}")
    prereg = _load_object(repository / PREREGISTRATION)
    result_root = repository / PRODUCER_OUTPUT
    expected_names = {"COHORT_SUPPORT.json", "QUERY_CELLS.json", "RESULT.json", "report.html"}
    _require(
        result_root.is_dir()
        and {path.name for path in result_root.iterdir()} == expected_names
        and all(
            path.is_file() and not path.is_symlink()
            for path in result_root.iterdir()
        ),
        "result directory closure differs",
    )
    result = _load_object(result_root / "RESULT.json")
    cohort_rows = _load_list(result_root / "COHORT_SUPPORT.json")
    query_cells = _load_list(result_root / "QUERY_CELLS.json")

    prereg_digest = _digest(prereg, {"preregistration_digest"})
    _require(prereg["preregistration_digest"] == prereg_digest, "preregistration digest differs")
    _require(set(prereg) == {
        "claims", "design", "execution", "implementation_commit", "inputs",
        "known_inventory", "preregistration_digest", "prior_exposure", "runtime_sha256",
        "schema_version",
    }, "preregistration field closure differs")
    _require(all((
        prereg["schema_version"] == "m04r14-r1b-support-pilot-preregistration-v2",
        prereg["design"] == DESIGN,
        prereg["claims"] == CLAIMS,
        prereg["known_inventory"] == KNOWN_INVENTORY,
        prereg["execution"] == {
            "output_root": str(PRODUCER_OUTPUT),
            "publication": "flock-serialized create-only atomic directory rename with parent fsync",
        },
        prereg["prior_exposure"] == {
            "r1a_graph_and_recurrence_counts_previously_observed": True,
            "terminated_io_preflight_before_support_values": True,
            "v1_execution_failed_on_query_pack_identity_before_support_values": True,
            "v1_history": V1_HISTORY,
            "support_values_previously_observed": False,
        },
        set(prereg["inputs"]) == {
            "file_sha256", "packed_generation_id", "r1a_case_manifest_digest",
            "r1a_integrity_digest", "r1a_result_digest", "source_lock_digest",
        },
        set(prereg["runtime_sha256"]) == set(PRODUCER_RUNTIME_FILES),
    )), "preregistration contract differs")
    h1 = str(result["preregistration_commit"])
    _require(_git(repository, "rev-parse", f"{h1}^") == prereg["implementation_commit"], "H0/H1 parent differs")
    _require(
        _git(repository, "diff-tree", "--no-commit-id", "--name-only", "-r", h1)
        == PREREGISTRATION.as_posix(),
        "H1 is not a sole-preregistration commit",
    )
    committed = subprocess.run(
        ("git", "show", f"{h1}:{PREREGISTRATION}"), cwd=repository,
        capture_output=True, check=False,
    )
    _require(
        committed.returncode == 0
        and committed.stdout == (repository / PREREGISTRATION).read_bytes(),
        "committed preregistration bytes differ",
    )
    for path, expected in prereg["inputs"]["file_sha256"].items():
        _require(_sha(Path(path)) == expected, f"preregistered input differs: {path}")
    for path, expected in prereg["runtime_sha256"].items():
        _require(_sha(repository / path) == expected, f"preregistered runtime differs: {path}")
    _validate_v1_history(repository)

    _require(result["result_digest"] == _digest(result, {"result_digest"}), "result digest differs")
    _require(result["preregistration_digest"] == prereg_digest, "result/preregistration binding differs")
    expected_result_keys = {
        "adequacy_labels_authorized", "base_designs", "cohort_support_sha256", "gates",
        "inputs", "inventory", "passed", "passed_meaning", "permitted_input_contract",
        "predictive_claim_authorized", "preregistration_commit", "preregistration_digest",
        "production_promotion_authorized", "query_cell_assignments_digest", "query_cells_sha256",
        "r1b_test_statistics_opened", "real_forward_outcomes_accessed", "remaining_before_r1b_statistics",
        "result_digest", "schema_version", "selected_base_design", "selected_retained_structure_cells",
        "selected_structure_cells", "status", "structure_designs", "structure_medoid_query_ids",
        "support_contract", "support_statistics_opened",
    }
    _require(set(result) == expected_result_keys, "result field closure differs")
    _require(all((
        result["schema_version"] == PRODUCER_SCHEMA,
        result["status"] == "support_only_complete",
        result["passed_meaning"] == "matching design estimable only",
        result["support_statistics_opened"] is True,
        result["r1b_test_statistics_opened"] is False,
        result["real_forward_outcomes_accessed"] is False,
        result["adequacy_labels_authorized"] is False,
        result["predictive_claim_authorized"] is False,
        result["production_promotion_authorized"] is False,
        result["permitted_input_contract"]["authority_manifest_files"]
            == prereg["inputs"]["file_sha256"],
    )), "result claim/input contract differs")
    _require(set(result["inputs"]) == {
        "benchmark_sha256", "packed_generation_id", "query_feature_audit_digest",
        "r1a_integrity_digest", "r1a_result_digest", "shadow_case_manifest_digest",
        "shadow_registry_digest", "shadow_source_lock_digest",
    }, "result input field closure differs")
    expected_gates = {
        "base_support_decision_available": True,
        "causal_membership_reconstructed": True,
        "cohort_threshold_fixed_from_r1a_null_p99": True,
        "history_and_context_contract_reconstructed": True,
        "outcomes_excluded_by_permitted_input_contract": True,
        "packed_content_and_query_prefixes_verified": True,
        "query_covariates_complete": True,
        "r1a_full_integrity_bound": True,
        "selected_subset_of_eligibility": True,
        "shadow_cases_reconstructed": True,
        "structure_support_decision_available": True,
        "support_only_no_scientific_statistics": True,
    }
    _require(result["gates"] == expected_gates, "producer gate closure differs")
    _require(result["support_contract"] == {
        "minimum_matched_sets": SUPPORT_CAP,
        "minimum_episode_coverage": MIN_EPISODE_COVERAGE,
        "minimum_link_coverage": MIN_LINK_COVERAGE,
        "base_design_order": list(prereg["design"]["base_order"]),
        "calendar_bin_formula": "floor(benchmark_session_ordinal / span), anchored at first IXIC row",
        "volatility_cells": "balanced deterministic within-session21-bin rank terciles; value then query-id tie break",
        "history_fixed": "lookback=252 and stock-prefix rows>=252 for every query; exact query identity and both semantic causal prefixes reconstructed",
        "context_availability_reconstructed": True,
        "structure_view": "market-excluded exact query coarse[0:96] + stage columns net/volatility/volume + structural; robust column scaling",
        "structure_options": [8, 12, 16],
        "minimum_structure_cell_size": 30,
    }, "support contract differs")
    _require(result["remaining_before_r1b_statistics"] == [
        "independent support-pilot reconstruction",
        "freeze B0 statistic and shared-priority contracts",
    ], "remaining-work contract differs")
    _require(_sha(result_root / "COHORT_SUPPORT.json") == result["cohort_support_sha256"], "cohort artifact hash differs")
    _require(_sha(result_root / "QUERY_CELLS.json") == result["query_cells_sha256"], "query-cell artifact hash differs")
    _require(stable_hash(query_cells) == result["query_cell_assignments_digest"], "query-cell semantic digest differs")
    parser = HTMLParser(); parser.feed((result_root / "report.html").read_text())
    _require((result_root / "report.html").read_text() == _expected_html(result), "report HTML differs")

    r1a_result = _load_object(repository / R1A_OUTPUT / "RESULT.json")
    r1a_integrity = _load_object(repository / R1A_INTEGRITY / "VERIFIED.json")
    registry = _load_object(repository / REGISTRY)
    _require(
        r1a_result.get("result_digest")
        == _digest(r1a_result, {"result_digest", "elapsed_seconds"}),
        "R1-A result digest differs",
    )
    _require(all((
        r1a_integrity.get("verification_digest")
            == _digest(r1a_integrity, {"verification_digest", "created_at"}),
        r1a_integrity.get("passed") is True,
        all(r1a_integrity.get("gates", {}).values()),
        r1a_integrity.get("verified_result_digest") == r1a_result["result_digest"],
        registry.get("registry_digest") == _digest(registry, {"registry_digest"}),
        registry["source_lock"].get("source_lock_digest")
            == _digest(registry["source_lock"], {"source_lock_digest"}),
    )), "upstream authority integrity differs")
    _require(all((
        result["inputs"]["r1a_result_digest"] == r1a_result["result_digest"],
        result["inputs"]["r1a_integrity_digest"] == r1a_integrity["verification_digest"],
        result["inputs"]["shadow_registry_digest"] == registry["registry_digest"],
        result["inputs"]["shadow_source_lock_digest"] == registry["source_lock"]["source_lock_digest"],
        result["inputs"]["shadow_case_manifest_digest"] == r1a_result["inputs"]["case_manifest_digest"],
        result["inputs"]["packed_generation_id"] == r1a_result["inputs"]["packed_generation_id"],
        result["inputs"]["benchmark_sha256"] == _sha(BENCHMARK),
        prereg["inputs"]["r1a_result_digest"] == r1a_result["result_digest"],
        prereg["inputs"]["r1a_integrity_digest"] == r1a_integrity["verification_digest"],
        prereg["inputs"]["r1a_case_manifest_digest"] == r1a_result["inputs"]["case_manifest_digest"],
        prereg["inputs"]["packed_generation_id"] == r1a_result["inputs"]["packed_generation_id"],
        prereg["inputs"]["source_lock_digest"] == registry["source_lock"]["source_lock_digest"],
        float(r1a_result["comparison"]["episode_max"]["null_p99"]) == 4.0,
    )), "upstream semantic binding differs")
    generation = str(result["inputs"]["packed_generation_id"])
    durable_manifest = repository / PACKED_DURABLE / "generations" / generation / "manifest.json"
    expected_preregistered_files = {
        str((repository / R1A_OUTPUT / "RESULT.json").resolve()),
        str((repository / R1A_INTEGRITY / "VERIFIED.json").resolve()),
        str((repository / REGISTRY).resolve()),
        str(durable_manifest.resolve()),
        str((repository / V1_PREREGISTRATION).resolve()),
        str((repository / CONFIG).resolve()),
        str(BENCHMARK.resolve()),
    }
    _require(
        set(prereg["inputs"]["file_sha256"]) == expected_preregistered_files,
        "preregistered input path closure differs",
    )
    store = _select_packed_store(repository, generation)
    loaded = load_packed_generation(
        store, generation,
        expected_provenance_digest=str(r1a_result["inputs"]["packed_provenance_digest"]),
        verify_content=True, validate_records=True,
    )
    metadata = _metadata_from_loaded(loaded)
    registry_rows = {str(row["episode_id"]): row for row in registry["cases_data"]}
    _require(
        len(registry["cases_data"]) == len(registry_rows) == 3270,
        "registry query identity closure differs",
    )
    symbol_ids = {symbol: index for index, symbol in enumerate(metadata.symbols)}
    queries_by_id: dict[str, Query] = {}
    selected: defaultdict[str, list[str]] = defaultdict(list)
    recurrent_meta: dict[str, tuple[str, int]] = {}
    case_manifest = []
    for path in sorted((repository / CASES).glob("*.json")):
        case = _load_object(path); query_id = str(case["query_episode_id"])
        registered = registry_rows.get(query_id)
        _require(registered is not None, f"unregistered case: {query_id}")
        _require(all((
            case.get("gate_passed") is True,
            case.get("real_forward_outcomes_accessed") is False,
            case.get("result_digest") == shadow._deterministic_case_digest(case),
            case.get("checkpoint_integrity_digest") == shadow._integrity_digest(case),
            case.get("registry_case_id") == registered["case_id"],
            case.get("latest_eligible_cutoff") == registered["latest_eligible_cutoff"],
            case.get("query_symbol") == registered["symbol"],
            case.get("query_cutoff") == registered["cutoff"],
            case.get("query_stock_prefix") == registered["stock_prefix"],
            case.get("query_benchmark_prefix") == registered["benchmark_prefix"],
            case.get("registry_digest") == registry["registry_digest"],
            case.get("generation_id") == generation,
            len(case.get("matches", [])) == 20,
            len({str(match["episode_id"]) for match in case.get("matches", [])}) == 20,
        )), f"sealed case differs: {query_id}")
        symbol = str(case["query_symbol"])
        queries_by_id[query_id] = Query(
            query_id, symbol, symbol_ids[symbol],
            int(np.datetime64(case["query_start"], "ns").view(np.int64)),
            int(np.datetime64(case["latest_eligible_cutoff"], "ns").view(np.int64)),
            int(case["certificate"]["eligible_candidates"]),
        )
        for match in case["matches"]:
            episode_id = str(match["episode_id"]); selected[episode_id].append(query_id)
            if episode_id in recurrent_meta:
                _require(recurrent_meta[episode_id] == (
                    str(match["symbol"]),
                    int(np.datetime64(match["cutoff"], "ns").view(np.int64)),
                ), f"episode metadata conflicts: {episode_id}")
            recurrent_meta[episode_id] = (
                str(match["symbol"]),
                int(np.datetime64(match["cutoff"], "ns").view(np.int64)),
            )
        case_manifest.append({
            "path": path.relative_to(repository / CASES.parent).as_posix(),
            "bytes": path.stat().st_size, "sha256": _sha(path),
        })
    _require(stable_hash(case_manifest) == result["inputs"]["shadow_case_manifest_digest"], "case manifest differs")
    _require(result["permitted_input_contract"] == {
        "authority_manifest_files": prereg["inputs"]["file_sha256"],
        "shadow_case_files": 3270,
        "shadow_case_manifest_digest": stable_hash(case_manifest),
        "packed_binary_content_verified_from_manifest": True,
        "query_raw_ohlcv_root": str(load_config(repository / CONFIG).datasets["nasdaq"].path),
        "query_raw_ohlcv_usage": "only bars at or before each registered cutoff enter covariates; every semantic stock/benchmark prefix is rehashed against the sealed registry",
        "forbidden_roots": [
            "t14-09 evidence/outcome/card stores", "wf03d outcome/prediction stores",
            "wf04 evaluation stores", "t14-11 Stockbee stores",
            "t14-12 post-signal stores",
        ],
    }, "permitted input contract differs")

    query_ids = sorted(queries_by_id); queries = tuple(queries_by_id[key] for key in query_ids)
    _require(
        len(query_ids) == len(registry_rows) == 3270
        and sum(len(value) for value in selected.values()) == 65400,
        "query/link inventory differs",
    )
    pools = {
        value: _pool(metadata, value)
        for value in sorted({query.latest_ns for query in queries})
    }
    for query in queries:
        _verify_risk_set(metadata, *pools[query.latest_ns], query)
    query_rows = tuple(dict(registry_rows[key]) for key in query_ids)
    index = {key: value for value, key in enumerate(query_ids)}
    cohort = {
        episode_id: tuple(sorted(index[key] for key in values))
        for episode_id, values in selected.items() if len(values) >= 5
    }
    target_ids = np.asarray([bytes.fromhex(value) for value in cohort], dtype="V12")
    packed_mask = np.isin(metadata.episode_ids, target_ids)
    packed_positions = np.flatnonzero(packed_mask)
    _require(len(packed_positions) == len(cohort), "recurrent packed episode inventory differs")
    packed_recurrent = {
        bytes(metadata.episode_ids[position]).hex(): (
            metadata.symbols[int(metadata.symbol_ids[position])],
            int(metadata.cutoffs[position]),
        ) for position in packed_positions
    }
    _require(
        packed_recurrent == {key: recurrent_meta[key] for key in cohort},
        "recurrent episode packed identity differs",
    )
    eligible = {
        episode_id: tuple(
            query_index for query_index, query in enumerate(queries)
            if _causally_eligible(
                recurrent_meta[episode_id][0], recurrent_meta[episode_id][1],
                query.symbol, query.start_ns, query.latest_ns,
            )
        )
        for episode_id in cohort
    }
    _require(all(set(cohort[key]).issubset(eligible[key]) for key in cohort), "causal membership differs")

    for row in query_rows:
        cutoff = np.datetime64(row["cutoff"], "ns")
        row["context_available"] = (
            int(row["benchmark_prefix"]["rows"]) >= 252
            and np.datetime64(row["benchmark_prefix"]["coverage_cutoff"], "ns") >= cutoff
        )
    structure, volatility, audits = _reconstruct_query_features(
        repository, query_ids, query_rows, workers=workers,
    )
    _require(tuple(row["query_feature_audit"] for row in query_cells) == audits, "query feature audits differ")
    _require(stable_hash(audits) == result["inputs"]["query_feature_audit_digest"], "query feature digest differs")
    _require(
        len(query_cells) == len({row["query_episode_id"] for row in query_cells}) == 3270
        and all(set(row) == {
            "base_cells", "query_episode_id", "query_feature_audit", "structure_cells",
        } for row in query_cells)
        and all(set(row["structure_cells"]) == {"8", "12", "16"} for row in query_cells)
        and all(
            len(value["query_feature_audit"]["query_representation_digest"]) == 64
            and np.isfinite(float.fromhex(value["query_feature_audit"]["volatility_hex"]))
            for value in query_cells
        ),
        "query feature artifact closure differs",
    )
    cutoffs = np.asarray([
        int(np.datetime64(row["cutoff"], "ns").view(np.int64)) for row in query_rows
    ], dtype=np.int64)
    positions = _session_positions(cutoffs); bins21 = positions // 21
    volatility_terciles = np.full(len(query_ids), -1, dtype=np.int8)
    for value in np.unique(bins21):
        members = np.flatnonzero(bins21 == value)
        volatility_terciles[members] = _deterministic_terciles(
            volatility[members], [query_ids[item] for item in members],
        )
    base_cells = {
        name: _cells(span, query_rows, queries, positions, volatility_terciles)
        for name, span in (("session_21", 21), ("session_42", 42), ("session_63", 63))
    }
    for ordinal, row in enumerate(query_cells):
        _require(row["query_episode_id"] == query_ids[ordinal], "query order differs")
        _require(
            row["base_cells"] == {key: list(value[ordinal]) for key, value in base_cells.items()},
            f"base cell differs: {query_ids[ordinal]}",
        )
    base_designs = []; base_support = {}
    for name in ("session_21", "session_42", "session_63"):
        design, support = _design_result(name, base_cells[name], cohort, eligible)
        base_designs.append(design); base_support[name] = support
    selected_base = next(row["design"] for row in base_designs if row["passes"])

    center = np.median(structure, axis=0)
    scale = np.percentile(structure, 75, axis=0) - np.percentile(structure, 25, axis=0)
    scale[scale < 1e-6] = 1.0; structure = (structure - center) / scale
    structure_designs = []; structure_support = {}; medoid_ids = {}; partitions = {}
    for k in STRUCTURE_OPTIONS:
        labels, medoids = _farthest_first_partition(
            structure, query_ids, clusters=k, minimum_size=30,
        )
        partitions[k] = labels
        _require(
            [row["structure_cells"][str(k)] for row in query_cells] == labels.tolist(),
            f"structure partition differs: {k}",
        )
        cells = tuple((*base_cells[selected_base][i], int(labels[i])) for i in range(len(query_ids)))
        design, support = _design_result(f"{selected_base}_structure_{k}", cells, cohort, eligible)
        design.update(requested_structure_cells=k, retained_structure_cells=len(medoids))
        structure_designs.append(design); structure_support[k] = support
        medoid_ids[str(k)] = [query_ids[value] for value in medoids]
    supported = [row for row in structure_designs if row["passes"]]
    selected_structure = max(
        supported, key=lambda row: (row["retained_structure_cells"], row["requested_structure_cells"]),
    )
    expected_cohort = [{
        "episode_id": episode_id,
        "symbol": recurrent_meta[episode_id][0],
        "cutoff": np.datetime_as_string(np.datetime64(recurrent_meta[episode_id][1], "ns")),
        "observed_inbound_queries": len(cohort[episode_id]),
        "causally_eligible_queries": len(eligible[episode_id]),
        "base_support": {key: value[episode_id] for key, value in base_support.items()},
        "structure_support": {str(key): value[episode_id] for key, value in structure_support.items()},
    } for episode_id in sorted(cohort, key=lambda key: (-len(cohort[key]), key))]
    _require(cohort_rows == expected_cohort, "cohort support rows differ")
    _require(
        len(cohort_rows) == len({row["episode_id"] for row in cohort_rows}) == 369,
        "cohort artifact closure differs",
    )
    _require(all(set(row) == {
        "base_support", "causally_eligible_queries", "cutoff", "episode_id",
        "observed_inbound_queries", "structure_support", "symbol",
    } for row in cohort_rows), "cohort row field closure differs")
    _require(result["base_designs"] == base_designs, "base support summaries differ")
    _require(result["structure_designs"] == structure_designs, "structure support summaries differ")
    _require(result["selected_base_design"] == selected_base, "selected base differs")
    _require(result["selected_structure_cells"] == selected_structure["requested_structure_cells"], "selected K differs")
    _require(result["selected_retained_structure_cells"] == selected_structure["retained_structure_cells"], "retained K differs")
    _require(result["structure_medoid_query_ids"] == medoid_ids, "structure medoids differ")
    _require(result["inventory"] == {
        "queries": 3270, "selected_links": 65400, "recurrent_episode_threshold": 5,
        "cohort_episodes": 369, "cohort_links": 2865,
        "query_representations_reconstructed": 3270, "query_prefix_mismatches": 0,
        "missing_volatility_queries": 0, "missing_context_queries": 0,
    }, "result inventory differs")
    _require(result["passed"] is True and all(result["gates"].values()), "producer support gates differ")
    selected_k = int(result["selected_structure_cells"])
    selected_base_result = next(
        row for row in base_designs if row["design"] == selected_base
    )
    nontrivial = _structure_nontrivial(
        selected_structure, selected_base_result, partitions[selected_k],
    )
    gates = {
        "preregistration_and_h0_h1_lineage_reconstructed": True,
        "all_frozen_input_and_runtime_hashes_reconstructed": True,
        "result_and_artifact_digests_reconstructed": True,
        "report_html_reconstructed": True,
        "packed_generation_fully_verified": True,
        "all_3270_case_seals_and_manifest_reconstructed": True,
        "causal_eligibility_and_selected_membership_reconstructed": True,
        "all_query_prefixes_features_and_audits_reconstructed": True,
        "base_and_structure_cells_reconstructed": True,
        "all_369_episode_support_counts_reconstructed": True,
        "all_support_summaries_and_selection_reconstructed": True,
        "outcomes_and_r1b_scientific_statistics_excluded": True,
    }
    state = {
        "schema_version": SCHEMA,
        "passed": all(gates.values()),
        "status": "full_support_only_integrity_reconstruction",
        "verified_preregistration_digest": prereg_digest,
        "verified_result_digest": result["result_digest"],
        "verified_queries": len(query_ids),
        "verified_cohort_episodes": len(cohort),
        "verified_cohort_links": sum(map(len, cohort.values())),
        "n0_matching_design_support_verified": True,
        "n0_scientific_execution_authorized": False,
        "structure_partition_nontrivial": nontrivial,
        "n1_nontrivial_matching_design_available": nontrivial,
        "n1_scientific_execution_authorized": False,
        "b2_scientific_test_authorized": False,
        "adequacy_labels_authorized": False,
        "predictive_claim_authorized": False,
        "production_promotion_authorized": False,
        "real_forward_outcomes_accessed": False,
        "verifier_commit": verifier_commit,
        "verifier_runtime_sha256": {
            value: _sha(repository / value) for value in VERIFIER_RUNTIME_FILES
        },
        "interpretation": (
            "producer contract is valid, but retained one structure cell; N1 equals N0"
            if not nontrivial else "producer contract and nontrivial structure support verified"
        ),
        "gates": gates,
    }
    payload = {
        **state,
        "verification_digest": stable_hash(state),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = output.parent / f".{output.name}.tmp-{os.getpid()}-{uuid4().hex}"
    try:
        temporary.mkdir(parents=True)
        _atomic_receipt(temporary / "VERIFIED.json", payload)
        _atomic_publish_directory(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(argv)
    print(json.dumps(execute(args.repository, workers=args.workers), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
