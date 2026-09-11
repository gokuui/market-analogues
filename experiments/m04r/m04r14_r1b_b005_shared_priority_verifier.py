"""Independent replay verifier for the sealed B0-05 shared-priority result.

This runtime deliberately imports neither the B0-05 producer nor its shared-
priority/adequacy validation cores.  It reconstructs the packed causal universe,
the 3,270 query risk sets, the observed concentration metrics, both SHA-256
priority orders, every null selection, all 512 replicate rows, and the final
scientific summaries from the frozen joint preregistration.

Verification is restartable at the producer-shard boundary.  A completed
verification shard binds the verifier commit and the exact producer shard bytes;
an interrupted shard is recomputed, while a published mismatch fails closed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import resource
import subprocess
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

import numpy as np


SCHEMA = "m04r14-r1b-b005-shared-priority-integrity-verification-v1"
CHECKPOINT_SCHEMA = f"{SCHEMA}-shard"
BINDING_SCHEMA = f"{SCHEMA}-binding"
PREREGISTRATION = Path("experiments/m04r/m04r14_r1b_joint_b005_b2_preregistered.json")
PREREGISTRATION_DIGEST = "734e65768f04cc9c8108e2dae480258f5fc48094c1db80517fc0f5bbe62af191"
R1A_PREREGISTRATION = Path("experiments/m04r/m04r14_r1a_exposure_audit_v2_preregistered.json")
PRODUCER = Path("config/data/analogues/m04r14/r1b-b005-shared-priority-v1")
OUTPUT = Path("config/data/analogues/m04r14/r1b-b005-shared-priority-integrity-verification-v1")
PACKED_ROOT = Path("config/data/analogues/poc/m04r/packed-bound-full/store")
CASES = Path("config/data/analogues/m04r14/nasdaq-shadow-snapshot-v1/cases")
VERIFIER_RUNTIME = (
    "experiments/m04r/m04r14_r1b_b005_shared_priority_verifier.py",
    "tests/test_r1b_b005_shared_priority_verifier.py",
)
EPISODE_DOMAIN = "market-analogues/r1b/b0-05/h-episode/v1"
SYMBOL_DOMAIN = "market-analogues/r1b/b0-05/h-symbol/v1"
METRICS = (
    "episode_unique", "episode_max", "episode_top_1_percent_share", "episode_hhi",
    "episode_effective_number", "episode_gini", "symbol_unique", "symbol_max",
    "symbol_top_1_percent_share", "symbol_hhi", "symbol_effective_number",
    "query_any_repeated_symbol_fraction", "repeated_symbol_twice_per_query",
    "repeated_symbol_thrice_per_query", "mean_pairwise_query_episode_overlap",
)
LOWER_METRICS = {
    "episode_unique", "episode_effective_number", "symbol_unique", "symbol_effective_number",
}
PACK_DTYPE = np.dtype([
    ("episode_id", "V12"), ("cutoff_ns", "<i8"), ("symbol_id", "<u4"),
    ("quality_tier", "u1"), ("presence", "u1", (3,)), ("coarse", "<f2", (128,)),
    ("samples_48", "<f2", (19, 48)), ("stage", "<f2", (48,)),
    ("structural", "<f2", (9,)), ("error_radii", "<f4", (41,)), ("padding", "V46"),
])
OVERFLOW_DTYPE = np.dtype([
    ("episode_id", "V12"), ("cutoff_ns", "<i8"), ("symbol_id", "<u4"),
    ("quality_tier", "u1"), ("padding", "V7"),
])


class VerificationError(RuntimeError):
    """A frozen authority or independently reconstructed value differs."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def stable_hash(value: Any) -> str:
    return sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False,
    ).encode("utf-8")).hexdigest()


def without(value: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in keys}


def load_json(path: Path, expected: type = dict) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            require(key not in result, f"duplicate JSON key: {path}/{key}")
            result[key] = item
        return result
    require(path.is_file() and not path.is_symlink(), f"regular JSON required: {path}")
    try:
        value = json.loads(
            path.read_bytes(), object_pairs_hook=pairs,
            parse_constant=lambda token: require(False, f"non-finite JSON: {token}"),
        )
    except (OSError, ValueError) as error:
        raise VerificationError(f"invalid JSON: {path}") from error
    require(isinstance(value, expected), f"JSON {expected.__name__} required: {path}")
    return value


def file_sha(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"regular file required: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def safe_repo_file(repository: Path, relative: str | Path) -> Path:
    name = Path(relative)
    require(not name.is_absolute() and ".." not in name.parts and name.as_posix() not in {"", "."},
            f"unsafe repository path: {relative}")
    root = repository.resolve(); path = root / name
    require(path.resolve().is_relative_to(root), f"repository path escaped: {relative}")
    require(path.is_file() and not path.is_symlink(), f"regular repository file required: {relative}")
    current = path.parent
    while current != root:
        require(not current.is_symlink(), f"symlink parent forbidden: {relative}")
        require(current != current.parent, f"repository ancestry differs: {relative}")
        current = current.parent
    return path


def git(repository: Path, *arguments: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ("git", *arguments), cwd=repository, capture_output=True,
        text=not binary, check=False,
    )
    require(result.returncode == 0, f"git {' '.join(arguments)} failed")
    return result.stdout if binary else result.stdout.strip()


def check_preregistration(repository: Path) -> tuple[dict[str, Any], str, str]:
    prereg = load_json(safe_repo_file(repository, PREREGISTRATION))
    require(set(prereg) == {
        "schema_version", "implementation_commit", "specification",
        "b2_priority_contract_digest", "environment", "runtime_sha256",
        "authorities", "preregistration_digest",
    }, "joint preregistration field closure differs")
    require(prereg["preregistration_digest"] == PREREGISTRATION_DIGEST
            == stable_hash(without(prereg, "preregistration_digest")),
            "joint preregistration digest differs")
    spec = prereg["specification"]["b005"]
    directions = {name: ("lower_is_more_concentrated" if name in LOWER_METRICS
                         else "higher_is_more_concentrated") for name in METRICS}
    require(spec["seed"] == 947221 and spec["replicates"] == 512
            and spec["top_k"] == 20 and spec["shard_replicates"] == 8
            and spec["shards"] == 64 and spec["production_workers"] == 12,
            "frozen B0-05 execution dimensions differ")
    require(spec["domains"] == {"episode": EPISODE_DOMAIN, "symbol": SYMBOL_DOMAIN}
            and spec["metrics"] == directions, "frozen B0-05 hash/metric contract differs")
    require(prereg["specification"]["claims"]["real_forward_outcomes_accessed"] is False,
            "preregistration outcome boundary differs")
    h0 = str(prereg["implementation_commit"])
    commits = str(git(repository, "log", "--format=%H", "--", str(PREREGISTRATION))).splitlines()
    candidates = [commit for commit in commits
                  if str(git(repository, "rev-parse", f"{commit}^")) == h0]
    require(len(candidates) == 1, "joint H0/H1 lineage differs")
    h1 = candidates[0]
    require(str(git(repository, "diff-tree", "--no-commit-id", "--name-only", "-r", h1)).splitlines()
            == [str(PREREGISTRATION)], "joint H1 is not the sole preregistration change")
    require(git(repository, "show", f"{h1}:{PREREGISTRATION}", binary=True)
            == (repository / PREREGISTRATION).read_bytes(), "committed preregistration bytes differ")
    head = str(git(repository, "rev-parse", "HEAD"))
    git(repository, "merge-base", "--is-ancestor", h1, head)
    for relative, expected in prereg["runtime_sha256"].items():
        path = safe_repo_file(repository, relative)
        require(file_sha(path) == expected, f"producer runtime changed: {relative}")
        for commit in (h0, h1, head):
            require(sha256(git(repository, "show", f"{commit}:{relative}", binary=True)).hexdigest()
                    == expected, f"committed producer runtime differs: {relative}/{commit}")
    for relative in VERIFIER_RUNTIME:
        path = safe_repo_file(repository, relative)
        require(git(repository, "show", f"{head}:{relative}", binary=True) == path.read_bytes(),
                f"verifier runtime is not committed: {relative}")
    return prereg, h1, head


@dataclass(frozen=True)
class Universe:
    episode_ids: np.ndarray
    cutoffs: np.ndarray
    symbol_ids: np.ndarray
    local_ordinals: np.ndarray
    symbols: tuple[str, ...]
    starts: np.ndarray
    stops: np.ndarray


@dataclass(frozen=True, order=True)
class Query:
    query_id: str
    symbol_id: int
    start_ns: int
    latest_ns: int
    eligible_count: int


def _manifest_records(prereg: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = prereg["authorities"]["reuse_authority_manifest"]
    require(len(rows) == 15332 and stable_hash(rows)
            == prereg["authorities"]["reuse_authority_manifest_digest"],
            "reuse authority manifest differs")
    records = {str(row["path"]): row for row in rows}
    require(len(records) == len(rows), "reuse authority paths duplicate")
    return records


def reconstruct_universe(repository: Path, prereg: Mapping[str, Any]) -> Universe:
    authorities = prereg["authorities"]
    r1a_path = safe_repo_file(repository, R1A_PREREGISTRATION)
    require(file_sha(r1a_path) == authorities["authority_file_sha256"][str(R1A_PREREGISTRATION)],
            "R1-A preregistration authority changed")
    r1a = load_json(r1a_path)
    require(r1a["preregistration_digest"] == stable_hash(without(r1a, "preregistration_digest")),
            "R1-A preregistration digest differs")
    generation_id = str(r1a["inputs"]["packed_generation_id"])
    generation = repository / PACKED_ROOT / "generations" / generation_id
    require(generation.is_dir() and not generation.is_symlink(), "packed generation missing or linked")
    records = _manifest_records(prereg)
    manifest_relative = (generation / "manifest.json").relative_to(repository).as_posix()
    require(manifest_relative in records, "packed manifest absent from frozen authority")
    manifest_path = safe_repo_file(repository, manifest_relative)
    require(file_sha(manifest_path) == records[manifest_relative]["sha256"]
            and manifest_path.stat().st_size == records[manifest_relative]["bytes"],
            "packed manifest bytes differ")
    manifest = load_json(manifest_path)
    require(manifest["manifest_digest"] == generation_id
            and manifest["provenance_digest"] == r1a["inputs"]["packed_provenance_digest"]
            and manifest["real_forward_outcomes_accessed"] is False,
            "packed manifest identity/outcome boundary differs")
    arrays = []
    for file_field, size_field, sha_field, count_field, dtype in (
        ("rows_file", "rows_bytes", "rows_sha256", "row_count", PACK_DTYPE),
        ("overflow_file", "overflow_bytes", "overflow_sha256", "overflow_count", OVERFLOW_DTYPE),
    ):
        relative = (generation / str(manifest[file_field])).relative_to(repository).as_posix()
        require(relative in records, f"packed {file_field} absent from frozen authority")
        path = safe_repo_file(repository, relative)
        require(path.stat().st_size == manifest[size_field] == records[relative]["bytes"]
                and file_sha(path) == manifest[sha_field] == records[relative]["sha256"],
                f"packed {file_field} bytes differ")
        require(path.stat().st_size == int(manifest[count_field]) * dtype.itemsize,
                f"packed {file_field} row accounting differs")
        arrays.append(np.memmap(path, dtype=dtype, mode="r") if path.stat().st_size
                      else np.empty(0, dtype=dtype))
    total = sum(len(array) for array in arrays)
    require(total == int(manifest["eligible_row_count"]) == 3_786_156,
            "packed candidate inventory differs")
    symbols = tuple(str(value) for value in manifest["symbols"])
    require(len(symbols) == len(set(symbols)) == 11_584, "packed symbol dictionary differs")
    compact = np.empty(total, dtype=[("episode_id", "V12"), ("cutoff", "<i8"), ("symbol", "<u4")])
    cursor = 0
    for source in arrays:
        for first in range(0, len(source), 1 << 17):
            block = source[first:first + (1 << 17)]; stop = cursor + len(block)
            compact["episode_id"][cursor:stop] = block["episode_id"]
            compact["cutoff"][cursor:stop] = block["cutoff_ns"]
            compact["symbol"][cursor:stop] = block["symbol_id"]
            cursor = stop
    require(cursor == total, "packed copy accounting differs")
    order = np.lexsort((compact["episode_id"], compact["cutoff"], compact["symbol"]))
    compact = compact[order]
    require(len(np.unique(compact["episode_id"])) == total, "packed episode IDs duplicate")
    require(int(compact["symbol"].max()) < len(symbols), "packed symbol ID exceeds dictionary")
    counts = np.bincount(compact["symbol"], minlength=len(symbols))
    stops = np.cumsum(counts, dtype=np.int64); starts = stops - counts
    local = np.empty(total, dtype=np.int64)
    prefixes = manifest["provenance"]["source_prefixes"]
    require(set(prefixes) == set(symbols), "packed source-prefix dictionary differs")
    for symbol_id, (raw_first, raw_stop) in enumerate(zip(starts, stops, strict=True)):
        first = int(raw_first); stop = int(raw_stop)
        require(np.all(compact["symbol"][first:stop] == symbol_id), "packed symbol ordering differs")
        require(stop - first < 2 or np.all(np.diff(compact["cutoff"][first:stop]) > 0),
                "packed cutoff ordering differs")
        local[first:stop] = np.arange(stop - first, dtype=np.int64)
        expected = max((int(prefixes[symbols[symbol_id]]["rows"]) - 252) // 5 + 1, 0)
        require(stop - first == expected, f"packed stride accounting differs: {symbols[symbol_id]}")
    return Universe(compact["episode_id"], compact["cutoff"], compact["symbol"],
                    local, symbols, starts, stops)


def _eligible(universe: Universe, query: Query, index: int) -> bool:
    cutoff = int(universe.cutoffs[index]); symbol = int(universe.symbol_ids[index])
    return cutoff <= query.latest_ns and not (symbol == query.symbol_id and cutoff >= query.start_ns)


def _top_share(counts: Sequence[int], fraction: float, population: int) -> float:
    ordered = sorted((int(value) for value in counts if value > 0), reverse=True)
    if not ordered:
        return 0.0
    take = min(len(ordered), max(1, int(np.ceil(population * fraction))))
    return float(sum(ordered[:take]) / sum(ordered))


def _hhi(counts: Sequence[int]) -> float:
    total = float(sum(counts))
    return float(sum((value / total) ** 2 for value in counts)) if total else 0.0


def _gini(counts: Sequence[int], population: int) -> float:
    positive = np.sort(np.asarray([value for value in counts if value > 0], dtype=float))
    require(population >= len(positive) and population >= 1 and bool(len(positive)),
            "invalid Gini population")
    indices = np.arange(population - len(positive) + 1, population + 1, dtype=float)
    return float((2.0 * np.dot(indices, positive) / (population * positive.sum()))
                 - (population + 1.0) / population)


def concentration_metrics(universe: Universe, selections: Sequence[Sequence[int]]) -> dict[str, float | int]:
    require(len(selections) >= 2 and all(len(row) == 20 for row in selections),
            "complete top-20 selections required")
    episodes: Counter[int] = Counter(); symbols: Counter[int] = Counter()
    repeated = twice = thrice = 0
    for selected in selections:
        require(len(set(int(value) for value in selected)) == 20, "selection episode duplicate")
        local = Counter(int(universe.symbol_ids[int(index)]) for index in selected)
        repeated += any(value > 1 for value in local.values())
        twice += sum(value == 2 for value in local.values())
        thrice += sum(value == 3 for value in local.values())
        for raw_index in selected:
            index = int(raw_index); episodes[index] += 1
            symbols[int(universe.symbol_ids[index])] += 1
    episode_counts = list(episodes.values()); symbol_counts = list(symbols.values())
    episode_hhi = _hhi(episode_counts); symbol_hhi = _hhi(symbol_counts)
    queries = len(selections)
    result: dict[str, float | int] = {
        "episode_unique": len(episode_counts), "episode_max": max(episode_counts),
        "episode_top_1_percent_share": _top_share(episode_counts, .01, len(universe.episode_ids)),
        "episode_hhi": episode_hhi, "episode_effective_number": 1.0 / episode_hhi,
        "episode_gini": _gini(episode_counts, len(universe.episode_ids)),
        "symbol_unique": len(symbol_counts), "symbol_max": max(symbol_counts),
        "symbol_top_1_percent_share": _top_share(symbol_counts, .01, len(universe.symbols)),
        "symbol_hhi": symbol_hhi, "symbol_effective_number": 1.0 / symbol_hhi,
        "query_any_repeated_symbol_fraction": repeated / queries,
        "repeated_symbol_twice_per_query": twice / queries,
        "repeated_symbol_thrice_per_query": thrice / queries,
        "mean_pairwise_query_episode_overlap": (
            sum(value * (value - 1) // 2 for value in episode_counts)
            / (queries * (queries - 1) / 2)
        ),
    }
    require(set(result) == set(METRICS), "independent metric closure differs")
    return result


def reconstruct_queries_and_actual(
    repository: Path, prereg: Mapping[str, Any], universe: Universe,
) -> tuple[tuple[Query, ...], dict[str, float | int], str]:
    manifest = prereg["authorities"]["case_manifest"]
    by_path = {str(row["path"]): row for row in manifest}
    require(len(by_path) == len(manifest) == 3270, "frozen case manifest differs")
    case_entries = tuple((repository / CASES).iterdir())
    require(all(path.is_file() and not path.is_symlink() and path.suffix == ".json"
                for path in case_entries), "case directory contains an unsealed entry")
    require(set(by_path) == {path.relative_to(repository).as_posix() for path in case_entries},
            "case directory closure differs")
    symbols = {symbol: index for index, symbol in enumerate(universe.symbols)}
    universe_order = np.argsort(universe.episode_ids, kind="stable")
    sorted_ids = universe.episode_ids[universe_order]
    queries: list[Query] = []; observed: list[tuple[int, ...]] = []; identity = []
    for relative in sorted(by_path):
        record = by_path[relative]; path = safe_repo_file(repository, relative)
        require(path.stat().st_size == record["bytes"] and file_sha(path) == record["sha256"],
                f"case authority bytes differ: {relative}")
        case = load_json(path)
        require(case.get("gate_passed") is True and case.get("real_forward_outcomes_accessed") is False,
                f"case outcome/integrity boundary differs: {relative}")
        query_id = str(case["query_episode_id"]); symbol = str(case["query_symbol"])
        require(symbol in symbols, f"query symbol absent from universe: {query_id}")
        query = Query(
            query_id, symbols[symbol],
            int(np.datetime64(case["query_start"]).astype("datetime64[ns]").astype(np.int64)),
            int(np.datetime64(case["latest_eligible_cutoff"]).astype("datetime64[ns]").astype(np.int64)),
            int(case["certificate"]["eligible_candidates"]),
        )
        matches = case.get("matches")
        require(isinstance(matches, list) and len(matches) == 20, "case top-20 differs")
        distances = [float(row["total_distance"]) for row in matches]
        ids = [str(row["episode_id"]) for row in matches]
        require(all(math.isfinite(value) and value >= 0 for value in distances)
                and list(zip(distances, ids)) == sorted(zip(distances, ids))
                and len(set(ids)) == 20, "case match order/identity differs")
        selected = []
        for rank, match in enumerate(matches, 1):
            episode_id = str(match["episode_id"])
            require(re.fullmatch(r"[0-9a-f]{24}", episode_id) is not None,
                    "noncanonical episode identifier")
            raw_id = np.void(bytes.fromhex(episode_id)); position = int(np.searchsorted(sorted_ids, raw_id))
            require(position < len(sorted_ids) and sorted_ids[position] == raw_id,
                    f"observed episode absent from universe: {episode_id}")
            index = int(universe_order[position]); selected.append(index)
            require(universe.symbols[int(universe.symbol_ids[index])] == str(match["symbol"])
                    and int(universe.cutoffs[index]) == int(np.datetime64(match["cutoff"])
                        .astype("datetime64[ns]").astype(np.int64)),
                    "observed episode metadata differs")
            identity.append({"query_id": query_id, "episode_id": episode_id, "rank": rank})
        require(all(_eligible(universe, query, index) for index in selected),
                "observed episode outside causal risk set")
        local = Counter(int(universe.symbol_ids[index]) for index in selected)
        require(max(local.values()) <= 3, "observed symbol cap differs")
        for left, first in enumerate(selected):
            for second in selected[left + 1:]:
                if int(universe.symbol_ids[first]) == int(universe.symbol_ids[second]):
                    require(abs(int(universe.local_ordinals[first])
                                - int(universe.local_ordinals[second])) * 5 > 251,
                            "observed inclusive-overlap selector differs")
        queries.append(query); observed.append(tuple(selected))
    order = np.argsort(np.asarray([query.query_id for query in queries], dtype="U"), kind="stable")
    queries_sorted = tuple(queries[int(index)] for index in order)
    observed_sorted = tuple(observed[int(index)] for index in order)
    require([query.query_id for query in queries_sorted]
            == prereg["authorities"]["population"]["query_ids"], "query population differs")
    all_cutoffs = np.sort(universe.cutoffs, kind="stable")
    for query in queries_sorted:
        total = int(np.searchsorted(all_cutoffs, query.latest_ns, side="right"))
        first, stop = int(universe.starts[query.symbol_id]), int(universe.stops[query.symbol_id])
        own = universe.cutoffs[first:stop]
        removed = int(np.searchsorted(own, query.latest_ns, side="right")
                      - np.searchsorted(own, query.start_ns, side="left"))
        require(total - removed == query.eligible_count, f"query risk-set certificate differs: {query.query_id}")
    return queries_sorted, concentration_metrics(universe, observed_sorted), stable_hash(identity)


def expected_binding(
    prereg: Mapping[str, Any], h1: str, universe: Universe, queries: Sequence[Query],
    retrieval_identity_digest: str,
) -> dict[str, Any]:
    query_rows = [{"query_id": row.query_id, "symbol_id": row.symbol_id,
                   "start_ns": row.start_ns, "latest_ns": row.latest_ns,
                   "eligible_count": row.eligible_count} for row in queries]
    state = {
        "schema_version": "m04r14-r1b-b005-shared-priority-binding-v1",
        "h1_commit": h1, "joint_preregistration_digest": PREREGISTRATION_DIGEST,
        "runtime_sha256": prereg["runtime_sha256"],
        "episode_ids_sha256": sha256(b"".join(bytes(value) for value in universe.episode_ids)).hexdigest(),
        "symbols_digest": stable_hash(list(universe.symbols)),
        "queries_digest": stable_hash(query_rows), "candidate_episodes": len(universe.episode_ids),
        "candidate_symbols": len(universe.symbols), "queries": len(queries), "replicates": 512,
        "replicate_indices": list(range(512)), "shard_size": 8,
        "shard_ranges": [[first, first + 8] for first in range(0, 512, 8)],
        "seed": 947221, "priority_domains": {"episode": EPISODE_DOMAIN, "symbol": SYMBOL_DOMAIN},
        "metric_directions": prereg["specification"]["b005"]["metrics"],
        "top_k": 20, "max_per_symbol": 3, "real_forward_outcomes_accessed": False,
        "retrieval_identity_digest": retrieval_identity_digest,
    }
    return {**state, "binding_digest": stable_hash(state)}


def _lp16(value: bytes) -> bytes:
    require(len(value) <= 0xFFFF, "priority identifier exceeds uint16")
    return len(value).to_bytes(2, "big") + value


def priority(domain: str, seed: int, replicate: int, identifier: str) -> bytes:
    require(0 <= seed < 2**64 and 0 <= replicate < 2**32, "priority draw outside integer range")
    return sha256(_lp16(domain.encode("utf-8")) + seed.to_bytes(8, "big")
                  + replicate.to_bytes(4, "big") + _lp16(identifier.encode("utf-8"))).digest()


_FORK_IDS: np.ndarray | None = None
_FORK_BUFFER: Any = None


def _hash_range(arguments: tuple[int, int, int, int]) -> int:
    seed, replicate, first, stop = arguments
    require(_FORK_IDS is not None and _FORK_BUFFER is not None, "priority worker uninitialized")
    domain = EPISODE_DOMAIN.encode("utf-8")
    prefix = _lp16(domain) + seed.to_bytes(8, "big") + replicate.to_bytes(4, "big") + (24).to_bytes(2, "big")
    output = np.frombuffer(_FORK_BUFFER, dtype=np.uint8).reshape(-1, 32)
    for index in range(first, stop):
        output[index] = np.frombuffer(sha256(prefix + bytes(_FORK_IDS[index]).hex().encode("ascii")).digest(), dtype=np.uint8)
    return stop - first


class EpisodeHasher:
    def __init__(self, episode_ids: np.ndarray, workers: int):
        require("fork" in mp.get_all_start_methods() and workers >= 1, "fork workers required")
        self.length = len(episode_ids); self.workers = min(workers, self.length)
        context = mp.get_context("fork"); self.buffer = context.RawArray("B", self.length * 32)
        global _FORK_IDS, _FORK_BUFFER
        _FORK_IDS = episode_ids; _FORK_BUFFER = self.buffer
        self.pool = context.Pool(self.workers)
        block = (self.length + self.workers - 1) // self.workers
        self.ranges = tuple((first, min(first + block, self.length))
                            for first in range(0, self.length, block))

    def hashes(self, seed: int, replicate: int) -> np.ndarray:
        completed = self.pool.map(_hash_range, ((seed, replicate, first, stop)
                                                for first, stop in self.ranges))
        require(sum(completed) == self.length, "parallel priority accounting differs")
        return np.frombuffer(self.buffer, dtype=np.uint8).reshape(self.length, 32)

    def __enter__(self) -> "EpisodeHasher":
        return self

    def __exit__(self, kind: Any, value: Any, traceback: Any) -> None:
        if kind is None:
            self.pool.close(); self.pool.join()
        else:
            self.pool.terminate(); self.pool.join()
        global _FORK_IDS, _FORK_BUFFER
        _FORK_IDS = None; _FORK_BUFFER = None


def complete_orders(
    universe: Universe, episode_hashes: np.ndarray, symbol_hashes: np.ndarray,
    episode_id_order: np.ndarray, symbol_id_order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require(episode_hashes.shape == (len(universe.episode_ids), 32)
            and symbol_hashes.shape == (len(universe.symbols), 32)
            and episode_hashes.dtype == symbol_hashes.dtype == np.uint8,
            "priority matrices differ")
    ep = np.ascontiguousarray(episode_hashes).view("V32").reshape(-1)
    sy = np.ascontiguousarray(symbol_hashes).view("V32").reshape(-1)
    global_order = episode_id_order[np.argsort(ep[episode_id_order], kind="stable")]
    symbol_order = symbol_id_order[np.argsort(sy[symbol_id_order], kind="stable")]
    symbol_rank = np.empty(len(symbol_order), dtype=np.int64)
    symbol_rank[symbol_order] = np.arange(len(symbol_order), dtype=np.int64)
    episode_rank = np.empty(len(global_order), dtype=np.int64)
    episode_rank[global_order] = np.arange(len(global_order), dtype=np.int64)
    hierarchical = np.lexsort((episode_rank, symbol_rank[universe.symbol_ids]))
    return (global_order.astype(np.int64, copy=False),
            hierarchical.astype(np.int64, copy=False),
            symbol_order.astype(np.int64, copy=False))


def select_direct(universe: Universe, query: Query, order: Iterable[int]) -> tuple[int, ...]:
    selected: list[int] = []; counts: Counter[int] = Counter()
    for raw_index in order:
        index = int(raw_index)
        if not _eligible(universe, query, index):
            continue
        symbol = int(universe.symbol_ids[index])
        if counts[symbol] >= 3:
            continue
        ordinal = int(universe.local_ordinals[index])
        if any(int(universe.symbol_ids[prior]) == symbol
               and abs(ordinal - int(universe.local_ordinals[prior])) * 5 <= 251
               for prior in selected):
            continue
        selected.append(index); counts[symbol] += 1
        if len(selected) == 20:
            return tuple(selected)
    raise VerificationError(f"independent selector cannot satisfy query: {query.query_id}")


def select_global_batch(
    universe: Universe, queries: Sequence[Query], order: np.ndarray,
) -> tuple[tuple[int, ...], ...]:
    """Exact date-prefix reuse, independently guarded by direct own-symbol fallback."""

    by_latest: dict[int, tuple[int, ...]] = {}; output = []
    for query in queries:
        base = by_latest.get(query.latest_ns)
        if base is None:
            date_query = Query("date-only", -1, query.latest_ns + 1, query.latest_ns, 0)
            base = select_direct(universe, date_query, order)
            by_latest[query.latest_ns] = base
        if any(int(universe.symbol_ids[index]) == query.symbol_id for index in base):
            output.append(select_direct(universe, query, order))
        else:
            output.append(base)
    return tuple(output)


def select_hierarchical_batch(
    universe: Universe, queries: Sequence[Query], hierarchical_order: np.ndarray,
    symbol_order: np.ndarray,
) -> tuple[tuple[int, ...], ...]:
    """Factor the exact hierarchical order into independently checked symbol blocks."""

    counts = universe.stops - universe.starts
    block_stops = np.cumsum(counts[symbol_order], dtype=np.int64)
    block_starts = block_stops - counts[symbol_order]
    common: dict[tuple[int, int], tuple[int, ...]] = {}
    own: dict[tuple[int, int, int], tuple[int, ...]] = {}

    def choices(query: Query, rank: int, symbol: int) -> tuple[int, ...]:
        is_own = symbol == query.symbol_id
        key = ((query.latest_ns, symbol, query.start_ns) if is_own
               else (query.latest_ns, symbol))
        cache = own if is_own else common
        if key in cache:
            return cache[key]
        selected = []
        block = hierarchical_order[int(block_starts[rank]):int(block_stops[rank])]
        for raw_index in block:
            index = int(raw_index); cutoff = int(universe.cutoffs[index])
            if cutoff > query.latest_ns or (is_own and cutoff >= query.start_ns):
                continue
            ordinal = int(universe.local_ordinals[index])
            if any(abs(ordinal - int(universe.local_ordinals[prior])) * 5 <= 251
                   for prior in selected):
                continue
            selected.append(index)
            if len(selected) == 3:
                break
        cache[key] = tuple(selected)
        return cache[key]

    output = []
    for query in queries:
        selected = []
        for rank, raw_symbol in enumerate(symbol_order):
            selected.extend(choices(query, rank, int(raw_symbol)))
            if len(selected) >= 20:
                break
        require(len(selected) >= 20, f"hierarchical selector cannot satisfy query: {query.query_id}")
        output.append(tuple(selected[:20]))
    return tuple(output)


def selection_digest(
    universe: Universe, queries: Sequence[Query], selections: Sequence[Sequence[int]], family: str,
) -> str:
    digest = sha256(f"m04r14-r1b-b005-shared-priority-v1/selection/{family}/v1".encode("ascii"))
    digest.update(len(queries).to_bytes(4, "big"))
    for query, selected in zip(queries, selections, strict=True):
        encoded = query.query_id.encode("utf-8"); digest.update(len(encoded).to_bytes(2, "big")); digest.update(encoded)
        digest.update(len(selected).to_bytes(2, "big"))
        for index in selected:
            digest.update(bytes(universe.episode_ids[int(index)]))
    return digest.hexdigest()


def replay_replicate(
    universe: Universe, queries: Sequence[Query], replicate: int,
    episode_hashes: np.ndarray | None = None,
    episode_id_order: np.ndarray | None = None,
    symbol_id_order: np.ndarray | None = None,
) -> dict[str, Any]:
    if episode_hashes is None:
        episode_hashes = np.asarray([
            np.frombuffer(priority(EPISODE_DOMAIN, 947221, replicate, bytes(value).hex()), dtype=np.uint8)
            for value in universe.episode_ids
        ], dtype=np.uint8)
    symbol_hashes = np.asarray([
        np.frombuffer(priority(SYMBOL_DOMAIN, 947221, replicate, symbol), dtype=np.uint8)
        for symbol in universe.symbols
    ], dtype=np.uint8)
    if episode_id_order is None:
        episode_id_order = np.argsort(universe.episode_ids, kind="stable").astype(np.int64, copy=False)
    if symbol_id_order is None:
        symbol_id_order = np.argsort(np.asarray(universe.symbols, dtype="U"), kind="stable").astype(np.int64, copy=False)
    global_order, hierarchical_order, symbol_order = complete_orders(
        universe, episode_hashes, symbol_hashes, episode_id_order, symbol_id_order,
    )
    global_selected = select_global_batch(universe, queries, global_order)
    hierarchical_selected = select_hierarchical_batch(
        universe, queries, hierarchical_order, symbol_order,
    )
    counters = {
        "episode_hashes": len(universe.episode_ids), "symbol_hashes": len(universe.symbols),
        "global_orders": 1, "hierarchical_orders": 1,
        "per_query_full_universe_hashes": 0, "per_query_full_universe_sorts": 0,
        "query_risk_sets": len(queries), "global_risk_set_filters": len(queries),
        "hierarchical_risk_set_filters": len(queries), "total_risk_set_filters": 2 * len(queries),
    }
    return {
        "replicate": replicate,
        "global": {"metrics": concentration_metrics(universe, global_selected),
                   "selection_digest": selection_digest(universe, queries, global_selected, "global")},
        "hierarchical": {"metrics": concentration_metrics(universe, hierarchical_selected),
                         "selection_digest": selection_digest(universe, queries, hierarchical_selected, "hierarchical")},
        "work_counters": counters,
        "selection_completeness": {"global": True, "hierarchical": True, "exact_top_k": 20},
    }


def validate_producer_shard(path: Path, binding_digest: str, first: int, stop: int,
                            expected_counters: Mapping[str, int]) -> dict[str, Any]:
    value = load_json(path)
    require(set(value) == {"schema_version", "binding_digest", "replicate_range", "rows", "shard_digest"}
            and value["schema_version"] == "m04r14-r1b-b005-shared-priority-shard-v1"
            and value["binding_digest"] == binding_digest and value["replicate_range"] == [first, stop],
            f"producer shard envelope differs: {path.name}")
    require([row.get("replicate") for row in value["rows"]] == list(range(first, stop)),
            f"producer shard replicate accounting differs: {path.name}")
    for row in value["rows"]:
        require(set(row) == {"replicate", "global", "hierarchical", "work_counters", "selection_completeness"}
                and row["selection_completeness"] == {"global": True, "hierarchical": True, "exact_top_k": 20}
                and row["work_counters"] == expected_counters,
                f"producer replicate closure differs: {path.name}")
        for family in ("global", "hierarchical"):
            payload = row[family]
            require(set(payload) == {"metrics", "selection_digest"}
                    and set(payload["metrics"]) == set(METRICS)
                    and all(type(number) in {int, float} and math.isfinite(float(number))
                            for number in payload["metrics"].values())
                    and re.fullmatch(r"[0-9a-f]{64}", payload["selection_digest"]) is not None,
                    f"producer family payload differs: {path.name}/{family}")
    require(value["shard_digest"] == stable_hash(without(value, "shard_digest")),
            f"producer shard digest differs: {path.name}")
    return value


def nearest_rank(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, int(np.ceil(probability * len(ordered))) - 1)]


def summaries(actual: Mapping[str, float | int], rows: Sequence[Mapping[str, Any]],
              directions: Mapping[str, str], family: str) -> dict[str, Any]:
    result = {}
    for metric in METRICS:
        observed = actual[metric]; values = [float(row[family]["metrics"][metric]) for row in rows]
        direction = directions[metric]
        tail = (sum(value >= float(observed) for value in values)
                if direction == "higher_is_more_concentrated"
                else sum(value <= float(observed) for value in values))
        result[metric] = {
            "observed": observed, "null_mean": math.fsum(values) / len(values),
            "null_p05": nearest_rank(values, .05), "null_p50": nearest_rank(values, .50),
            "null_p95": nearest_rank(values, .95), "null_p99": nearest_rank(values, .99),
            "concentration_direction": direction, "inclusive_tail_count": tail,
            "concentration_tail_monte_carlo_p": (1 + tail) / (len(values) + 1),
        }
    return result


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid4().hex}"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise VerificationError(f"create-only verification artifact exists: {path}") from error
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def verification_binding(repository: Path, prereg: Mapping[str, Any], h1: str, verifier_commit: str,
                         binding: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    shard_files = sorted((repository / PRODUCER / "shards").glob("shard-*.json"))
    rows = [{"path": path.name, "bytes": path.stat().st_size, "sha256": file_sha(path)}
            for path in shard_files]
    state = {
        "schema_version": BINDING_SCHEMA, "verifier_commit": verifier_commit,
        "h1_commit": h1, "preregistration_digest": PREREGISTRATION_DIGEST,
        "producer_binding_digest": binding["binding_digest"],
        "producer_binding_sha256": file_sha(repository / PRODUCER / "BINDING.json"),
        "producer_result_digest": result["result_digest"],
        "producer_result_sha256": file_sha(repository / PRODUCER / "RESULT.json"),
        "producer_report_sha256": file_sha(repository / PRODUCER / "report.html"),
        "producer_shards": rows, "producer_shards_digest": stable_hash(rows),
        "replicate_ranges": [[first, first + 8] for first in range(0, 512, 8)],
        "real_forward_outcomes_accessed": False,
    }
    return {**state, "binding_digest": stable_hash(state)}


def rehash_frozen_inputs_and_output(
    repository: Path, prereg: Mapping[str, Any], verification: Mapping[str, Any],
    binding: Mapping[str, Any], result: Mapping[str, Any],
) -> None:
    """Close the long-run mutation window before publishing verification."""

    require(load_json(safe_repo_file(repository, PREREGISTRATION)) == prereg,
            "joint preregistration changed during verification")
    for relative, expected in prereg["runtime_sha256"].items():
        require(file_sha(safe_repo_file(repository, relative)) == expected,
                f"producer runtime changed during verification: {relative}")
    authorities = prereg["authorities"]
    r1a_path = safe_repo_file(repository, R1A_PREREGISTRATION)
    require(file_sha(r1a_path) == authorities["authority_file_sha256"][str(R1A_PREREGISTRATION)],
            "R1-A preregistration changed during verification")
    r1a = load_json(r1a_path); generation = str(r1a["inputs"]["packed_generation_id"])
    records = _manifest_records(prereg)
    packed_prefix = f"{PACKED_ROOT.as_posix()}/generations/{generation}/"
    packed = [row for path, row in records.items() if path.startswith(packed_prefix)]
    require(len(packed) == 3, "packed authority file closure differs")
    for row in packed:
        path = safe_repo_file(repository, row["path"])
        require(path.stat().st_size == row["bytes"] and file_sha(path) == row["sha256"],
                f"packed authority changed during verification: {row['path']}")
    for row in authorities["case_manifest"]:
        path = safe_repo_file(repository, row["path"])
        require(path.stat().st_size == row["bytes"] and file_sha(path) == row["sha256"],
                f"case authority changed during verification: {row['path']}")
    producer_binding = safe_repo_file(repository, PRODUCER / "BINDING.json")
    require(load_json(producer_binding) == binding
            and file_sha(producer_binding) == verification["producer_binding_sha256"],
            "producer binding changed during verification")
    result_path = safe_repo_file(repository, PRODUCER / "RESULT.json")
    require(load_json(result_path) == result
            and file_sha(result_path) == verification["producer_result_sha256"],
            "producer result changed during verification")
    for row in verification["producer_shards"]:
        path = safe_repo_file(repository, PRODUCER / "shards" / row["path"])
        require(path.stat().st_size == row["bytes"] and file_sha(path) == row["sha256"],
                f"producer shard changed during verification: {row['path']}")
    require(file_sha(safe_repo_file(repository, PRODUCER / "report.html"))
            == verification["producer_report_sha256"],
            "producer report changed during verification")


def validate_checkpoint(path: Path, verification_digest: str, source: Mapping[str, Any],
                        first: int, stop: int, expected_rows_digest: str) -> dict[str, Any]:
    value = load_json(path)
    require(set(value) == {"schema_version", "verification_binding_digest", "producer_shard",
                           "replicate_range", "reconstructed_rows_digest", "exact_rows_match",
                           "checkpoint_digest"}
            and value["schema_version"] == CHECKPOINT_SCHEMA
            and value["verification_binding_digest"] == verification_digest
            and value["producer_shard"] == source and value["replicate_range"] == [first, stop]
            and value["exact_rows_match"] is True
            and value["reconstructed_rows_digest"] == expected_rows_digest
            and value["checkpoint_digest"] == stable_hash(without(value, "checkpoint_digest")),
            f"verification checkpoint differs: {path.name}")
    return value


def validate_result_and_rows(
    repository: Path, prereg: Mapping[str, Any], binding: Mapping[str, Any],
    result: Mapping[str, Any], actual: Mapping[str, float | int], all_rows: Sequence[Mapping[str, Any]],
    shard_digests: Sequence[str],
) -> dict[str, Any]:
    expected_counters = prereg["specification"]["b005"]["work_counters_per_replicate"]
    scientific = {
        "actual": dict(actual),
        "global_comparison": summaries(actual, all_rows, binding["metric_directions"], "global"),
        "hierarchical_comparison": summaries(actual, all_rows, binding["metric_directions"], "hierarchical"),
        "replicates_digest": stable_hash(list(all_rows)), "shard_digests": list(shard_digests),
    }
    require(result.get("scientific") == scientific
            and result.get("scientific_digest") == sha256(json.dumps(
                scientific, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")).hexdigest(), "producer scientific reconstruction differs")
    expected_gates = {
        "h1_validated": True, "outcomes_excluded": True,
        "shared_episode_priority_both_nulls": True,
        "one_complete_order_per_family_per_replicate": True,
        "zero_per_query_full_universe_hash_or_sort": True,
        "replicate_accounting_complete": True, "all_15_metrics_complete": True,
        "every_selection_exact_top20_unique": True,
    }
    expected_claims = {
        "adequacy_labels_authorized": False, "predictive_claim_authorized": False,
        "production_promotion_authorized": False, "real_forward_outcomes_accessed": False,
    }
    deterministic = {
        "schema_version": "m04r14-r1b-b005-shared-priority-v1",
        "status": "diagnostic_complete_pending_independent_verification", "passed": True,
        "binding_digest": binding["binding_digest"],
        "joint_preregistration_digest": PREREGISTRATION_DIGEST,
        "inventory": {"queries": 3270, "candidate_episodes": 3_786_156,
                      "candidate_symbols": 11_584, "replicates": 512, "null_families": 2},
        "scientific": scientific, "scientific_digest": result["scientific_digest"],
        "work_counters_per_replicate": expected_counters,
        "work_counters_total": prereg["specification"]["b005"]["work_counters_complete_512"],
        "gates": expected_gates, "claims": expected_claims,
    }
    require(result.get("result_digest") == stable_hash(deterministic)
            and without(result, "result_digest", "performance") == deterministic,
            "producer result digest/envelope differs")
    performance = result.get("performance")
    require(isinstance(performance, dict) and performance.get("workers") == 12
            and type(performance.get("completed_shards_this_invocation")) is int
            and type(performance.get("reused_shards")) is int
            and performance["completed_shards_this_invocation"] >= 0
            and performance["reused_shards"] >= 0
            and performance["completed_shards_this_invocation"] + performance["reused_shards"] == 64
            and all(type(performance.get(key)) in {int, float}
                    and math.isfinite(float(performance[key])) and float(performance[key]) >= 0
                    for key in ("wall_seconds_this_invocation", "cpu_seconds_this_invocation",
                                "self_cpu_seconds_this_invocation", "child_cpu_seconds_this_invocation",
                                "self_maximum_resident_set_kib", "child_maximum_resident_set_kib")),
            "producer performance receipt differs")
    expected_html = "<!doctype html><html lang='en'><head><meta charset='utf-8'><title>B0-05 shared-priority sensitivity</title></head><body>" \
        f"<h1>B0-05: {escape(result['status'])}</h1><p>Result <code>{result['result_digest']}</code>.</p>" \
        "<p>Outcome-blind global-episode and hierarchical-symbol shared-priority sensitivities. No adequacy, prediction, ranking, or trading claim is authorized before independent verification and later stages.</p></body></html>\n"
    report = safe_repo_file(repository, PRODUCER / "report.html")
    require(report.read_text(encoding="utf-8") == expected_html, "producer HTML report differs")
    return scientific


def execute(repository: Path, *, workers: int = 12, stop_after_shards: int | None = None) -> dict[str, Any]:
    started = perf_counter(); repository = repository.resolve()
    require(workers == 12, "production verification requires 12 workers")
    require(not str(git(repository, "status", "--porcelain", "--untracked-files=all")),
            "verification requires a clean committed tree")
    prereg, h1, verifier_commit = check_preregistration(repository)
    producer_root = repository / PRODUCER
    require(producer_root.is_dir() and not producer_root.is_symlink(), "producer output root differs")
    expected_names = {"BINDING.json", "RESULT.json", "report.html", "shards"}
    require({path.name for path in producer_root.iterdir()} == expected_names,
            "producer output closure differs")
    binding = load_json(safe_repo_file(repository, PRODUCER / "BINDING.json"))
    result = load_json(safe_repo_file(repository, PRODUCER / "RESULT.json"))
    universe = reconstruct_universe(repository, prereg)
    queries, actual, retrieval_identity = reconstruct_queries_and_actual(repository, prereg, universe)
    expected = expected_binding(prereg, h1, universe, queries, retrieval_identity)
    require(binding == expected, "producer binding reconstruction differs")
    expected_counters = prereg["specification"]["b005"]["work_counters_per_replicate"]
    all_rows = []; shard_digests = []
    source_by_name = {}
    shard_root = producer_root / "shards"
    require(shard_root.is_dir() and not shard_root.is_symlink(), "producer shard root differs")
    expected_shard_names = {f"shard-{first:04d}-{first + 8:04d}.json" for first in range(0, 512, 8)}
    require({path.name for path in shard_root.iterdir()} == expected_shard_names,
            "producer shard directory closure differs")
    for first in range(0, 512, 8):
        path = safe_repo_file(repository, PRODUCER / "shards" / f"shard-{first:04d}-{first + 8:04d}.json")
        value = validate_producer_shard(path, binding["binding_digest"], first, first + 8, expected_counters)
        all_rows.extend(value["rows"]); shard_digests.append(value["shard_digest"])
        source_by_name[path.name] = {"path": path.name, "bytes": path.stat().st_size, "sha256": file_sha(path)}
    require([row["replicate"] for row in all_rows] == list(range(512)), "producer replicate closure differs")
    verification = verification_binding(repository, prereg, h1, verifier_commit, binding, result)
    require(verification["producer_shards"] == [source_by_name[name] for name in sorted(source_by_name)],
            "verification source binding differs")
    output = repository / OUTPUT
    require(not output.is_symlink(), "verification output root is linked")
    output.mkdir(parents=True, exist_ok=True)
    allowed_output_names = {"BINDING.json", "shards", "VERIFIED.json", "report.html"}
    root_temporary = re.compile(r"^\.(BINDING\.json|VERIFIED\.json|report\.html)\.tmp-\d+-[0-9a-f]{32}$")
    for path in tuple(output.iterdir()):
        if root_temporary.fullmatch(path.name) is not None:
            require(path.is_file() and not path.is_symlink(), "unsafe verifier root temporary artifact")
            path.unlink()
    require({path.name for path in output.iterdir()} <= allowed_output_names,
            "unexpected verification output artifact")
    binding_path = output / "BINDING.json"
    if binding_path.exists() or binding_path.is_symlink():
        require(not binding_path.is_symlink() and load_json(binding_path) == verification,
                "existing verification binding differs")
    else:
        atomic_json(binding_path, verification)
    checkpoints = output / "shards"
    require(not checkpoints.is_symlink(), "verification checkpoint root is linked")
    checkpoints.mkdir(exist_ok=True)
    allowed = {f"verified-{first:04d}-{first + 8:04d}.json" for first in range(0, 512, 8)}
    temporary_pattern = re.compile(r"^\.(verified-\d{4}-\d{4}\.json)\.tmp-\d+-[0-9a-f]{32}$")
    for path in tuple(checkpoints.iterdir()):
        match = temporary_pattern.fullmatch(path.name)
        if match is not None and match.group(1) in allowed:
            require(path.is_file() and not path.is_symlink(), "unsafe verifier temporary artifact")
            path.unlink()
    require({path.name for path in checkpoints.iterdir()} <= allowed,
            "unexpected verification checkpoint artifact")
    missing = []
    checkpoint_rows = []
    for first in range(0, 512, 8):
        name = f"verified-{first:04d}-{first + 8:04d}.json"; path = checkpoints / name
        source = source_by_name[f"shard-{first:04d}-{first + 8:04d}.json"]
        if path.exists() or path.is_symlink():
            require(not path.is_symlink(), f"linked verification checkpoint: {name}")
            checkpoint_rows.append(validate_checkpoint(
                path, verification["binding_digest"], source, first, first + 8,
                stable_hash(all_rows[first:first + 8]),
            ))
        else:
            missing.append((first, first + 8, source))
    completed_now = 0
    episode_id_order = np.argsort(universe.episode_ids, kind="stable").astype(np.int64, copy=False)
    symbol_id_order = np.argsort(np.asarray(universe.symbols, dtype="U"), kind="stable").astype(np.int64, copy=False)
    hasher_context = EpisodeHasher(universe.episode_ids, workers) if missing else nullcontext(None)
    with hasher_context as hasher:
        for first, stop, source in missing:
            producer_rows = all_rows[first:stop]; reconstructed = []
            for replicate in range(first, stop):
                hashes = hasher.hashes(947221, replicate) if hasher is not None else None
                row = replay_replicate(
                    universe, queries, replicate, hashes, episode_id_order, symbol_id_order,
                )
                require(row == producer_rows[replicate - first],
                        f"independent replicate differs: {replicate}")
                reconstructed.append(row)
            state = {
                "schema_version": CHECKPOINT_SCHEMA,
                "verification_binding_digest": verification["binding_digest"],
                "producer_shard": source, "replicate_range": [first, stop],
                "reconstructed_rows_digest": stable_hash(reconstructed), "exact_rows_match": True,
            }
            checkpoint = {**state, "checkpoint_digest": stable_hash(state)}
            atomic_json(checkpoints / f"verified-{first:04d}-{stop:04d}.json", checkpoint)
            checkpoint_rows.append(checkpoint); completed_now += 1
            if stop_after_shards is not None and completed_now >= stop_after_shards:
                return {"status": "incomplete", "completed_shards_this_invocation": completed_now}
    checkpoint_rows = []
    for first in range(0, 512, 8):
        checkpoint_rows.append(validate_checkpoint(
            checkpoints / f"verified-{first:04d}-{first + 8:04d}.json",
            verification["binding_digest"], source_by_name[f"shard-{first:04d}-{first + 8:04d}.json"],
            first, first + 8, stable_hash(all_rows[first:first + 8]),
        ))
    scientific = validate_result_and_rows(
        repository, prereg, binding, result, actual, all_rows, shard_digests,
    )
    rehash_frozen_inputs_and_output(repository, prereg, verification, binding, result)
    gates = {
        "joint_h0_h1_lineage_and_runtime_verified": True,
        "verifier_runtime_committed_and_independent": True,
        "packed_main_overflow_universe_reconstructed": True,
        "all_3270_causal_risk_sets_reconstructed": True,
        "observed_15_metrics_reconstructed": True,
        "all_64_producer_shards_integrity_verified": True,
        "all_512_replicates_independently_replayed": True,
        "global_and_hierarchical_selection_digests_exact": True,
        "scientific_summaries_and_result_digest_reconstructed": True,
        "work_counters_and_restart_bindings_verified": True,
        "outcomes_predictions_and_labels_excluded": True,
    }
    state = {
        "schema_version": SCHEMA, "status": "verified_diagnostic_complete", "passed": True,
        "verifier_commit": verifier_commit, "h1_commit": h1,
        "preregistration_digest": PREREGISTRATION_DIGEST,
        "verification_binding_digest": verification["binding_digest"],
        "producer_binding_digest": binding["binding_digest"],
        "producer_result_digest": result["result_digest"],
        "producer_scientific_digest": result["scientific_digest"],
        "independently_reconstructed_scientific_digest": sha256(json.dumps(
            scientific, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")).hexdigest(),
        "inventory": {"queries": 3270, "candidate_episodes": 3_786_156,
                      "candidate_symbols": 11_584, "replicates": 512,
                      "producer_shards": 64, "null_families": 2, "metrics": 15},
        "checkpoint_digests": [row["checkpoint_digest"] for row in checkpoint_rows],
        "gates": gates,
        "claims": {"b005_complete": True, "adequacy_labels_authorized": False,
                   "predictive_claim_authorized": False, "ranking_change_authorized": False,
                   "production_promotion_authorized": False,
                   "real_forward_outcomes_accessed": False},
    }
    deterministic = {**state, "verification_digest": stable_hash(state)}
    receipt = {**deterministic, "created_at": datetime.now(timezone.utc).isoformat(),
               "performance": {"workers": workers,
                               "wall_seconds_this_invocation": perf_counter() - started,
                               "maximum_resident_set_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                               "completed_shards_this_invocation": completed_now,
                               "reused_shards": 64 - completed_now}}
    receipt_path = output / "VERIFIED.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        require(not receipt_path.is_symlink(), "verification receipt is linked")
        prior = load_json(receipt_path)
        require(without(prior, "created_at", "performance") == deterministic,
                "existing verification receipt differs")
        receipt = prior
    else:
        atomic_json(receipt_path, receipt)
    html = "<!doctype html><html lang='en'><head><meta charset='utf-8'><title>B0-05 independent verification</title></head><body>" \
        f"<h1>B0-05: {escape(receipt['status'])}</h1>" \
        f"<p>Verification <code>{escape(receipt['verification_digest'])}</code>; producer <code>{escape(receipt['producer_result_digest'])}</code>.</p>" \
        "<p>All 512 global and hierarchical shared-priority replicates, selections, metrics, summaries, bindings and causal authorities were independently reconstructed. This verifies an outcome-blind descriptive sensitivity only; it authorizes no adequacy, prediction, ranking, production or trading claim.</p></body></html>\n"
    report_path = output / "report.html"
    if report_path.exists() or report_path.is_symlink():
        require(not report_path.is_symlink() and report_path.read_text(encoding="utf-8") == html,
                "existing verification report differs")
    else:
        temporary = report_path.parent / f".{report_path.name}.tmp-{os.getpid()}-{uuid4().hex}"
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(html); handle.flush(); os.fsync(handle.fileno())
            os.link(temporary, report_path)
            descriptor = os.open(report_path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except FileExistsError as error:
            raise VerificationError("create-only verification report exists") from error
        finally:
            temporary.unlink(missing_ok=True)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(argv)
    print(json.dumps(execute(args.repository, workers=args.workers), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
