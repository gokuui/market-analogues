"""Independent verifier for the frozen R1-B B2 geometry and localization result.

This verifier does not import the B2 producer, joint-contract validator, or
adequacy-localization kernel.  It independently decodes every sealed array,
replays the geometry and all conditional randomization draws, reconstructs the
published decision, and emits a create-only integrity receipt.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import heapq
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Hashable, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd

from market_analogues.adapters import DirectorySource, source_from_spec
from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.config import load_config
from market_analogues.representation import represent, representation_input_digest
from market_analogues.types import Episode, EpisodeKey, InstrumentKey


SCHEMA = "m04r14-r1b-b2-localization-integrity-verification-v1"
PREREG = Path("experiments/m04r/m04r14_r1b_joint_b005_b2_preregistered.json")
GEOMETRY = Path("config/data/analogues/m04r14/r1b-b2-geometry-v1")
PRODUCER = Path("config/data/analogues/m04r14/r1b-b2-localization-v1")
OUTPUT = Path("config/data/analogues/m04r14/r1b-b2-localization-integrity-verification-v1")
CASES = Path("config/data/analogues/m04r14/nasdaq-shadow-snapshot-v1/cases")
RUNTIME = (
    "experiments/m04r/m04r14_r1b_b2_localization_verifier.py",
    "tests/test_r1b_b2_localization_verifier.py",
)
DIMENSIONS = 141
QUERIES = 3270
COHORT = 369
PRIMARY = 357
COHORT_LINKS = 2865
PRIMARY_LINKS = 2791
REPLICATES = 4096
SHARD_SIZE = 32
WORKERS = 12
ARRAYS = (
    "episode_n0", "episode_n1", "episode_k12", "episode_k16",
    "shared_n0", "shared_n1", "episode_n0_breadth",
)
NULLS = ARRAYS[:6]
PRIORITY_DOMAIN = b"R1B-B2-conditional-priority-v1"
PRIORITY_FAMILY = b"N0-N1-common"


class B2VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise B2VerificationError(message)


def stable(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def exact(value: object, keys: Sequence[str], name: str) -> Mapping[str, Any]:
    require(isinstance(value, dict) and set(value) == set(keys), f"{name} field closure differs")
    return value


def snapshot(path: Path) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        raise B2VerificationError(f"unsafe file: {path}") from error
    try:
        info = os.fstat(descriptor)
        require(stat.S_ISREG(info.st_mode), f"regular file required: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(descriptor)


def bound(path: Path, digest: object, size: object | None = None) -> bytes:
    require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
            f"invalid bound digest: {path}")
    content = snapshot(path)
    require(size is None or type(size) is int and len(content) == size, f"bound length differs: {path}")
    require(sha256(content).hexdigest() == digest, f"bound hash differs: {path}")
    return content


def decode(content: bytes, path: Path, expected: type = dict) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {path}/{key}")
            result[key] = value
        return result
    try:
        value = json.loads(content, object_pairs_hook=pairs,
                           parse_constant=lambda token: require(False, f"nonfinite JSON: {path}/{token}"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise B2VerificationError(f"invalid JSON: {path}") from error
    require(isinstance(value, expected), f"JSON {expected.__name__} required: {path}")
    return value


def load(path: Path, expected: type = dict) -> Any:
    return decode(snapshot(path), path, expected)


def file_sha(path: Path) -> str:
    return sha256(snapshot(path)).hexdigest()


def git(root: Path, *args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(("git", *args), cwd=root, capture_output=True,
                            text=not binary, check=False)
    require(result.returncode == 0, f"git {' '.join(args)} failed")
    return result.stdout if binary else result.stdout.strip()


def lineage(root: Path, prereg: Mapping[str, Any]) -> tuple[str, str]:
    require(not str(git(root, "status", "--porcelain", "--untracked-files=all")),
            "verification requires a clean committed tree")
    head = str(git(root, "rev-parse", "HEAD"))
    h0 = prereg["implementation_commit"]
    candidates = []
    for commit in str(git(root, "log", "--format=%H", "--", str(PREREG))).splitlines():
        if str(git(root, "rev-parse", f"{commit}^")) == h0:
            candidates.append(commit)
    require(len(candidates) == 1, "H0/H1 lineage differs")
    h1 = candidates[0]
    require(str(git(root, "diff-tree", "--no-commit-id", "--name-only", "-r", h1)).splitlines()
            == [str(PREREG)], "H1 sole-file closure differs")
    require(git(root, "show", f"{h1}:{PREREG}", binary=True) == snapshot(root / PREREG),
            "committed preregistration differs")
    require(git(root, "merge-base", "--is-ancestor", h1, head) == "", "H1 is not verifier ancestor")
    for relative, expected in prereg["runtime_sha256"].items():
        require(file_sha(root / relative) == expected, f"producer runtime changed: {relative}")
        for commit in (h0, h1, head):
            blob = git(root, "show", f"{commit}:{relative}", binary=True)
            require(sha256(blob).hexdigest() == expected, f"committed runtime changed: {relative}")
    for relative in RUNTIME:
        require(git(root, "show", f"{head}:{relative}", binary=True) == snapshot(root / relative),
                f"verifier runtime is not committed: {relative}")
    return head, h1


def id_digest(ids: Sequence[str]) -> str:
    return sha256(json.dumps(list(ids), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def array_digest(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    return stable({"shape": list(value.shape), "dtype": value.dtype.str,
                   "sha256": sha256(value.tobytes(order="C")).hexdigest()})


def geometry_semantic(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array); digest = sha256()
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.tobytes(order="C")); return digest.hexdigest()


def balanced_array_digest(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array, dtype="<f8")); digest = sha256()
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes()); digest.update(value.tobytes())
    return digest.hexdigest()


def npy(path: Path, record: Mapping[str, Any], shape: tuple[int, ...], dtype: np.dtype[Any]) -> np.ndarray:
    exact(record, ("path", "sha256", "semantic_digest", "shape", "dtype", "order"), "geometry array")
    require(path.name == record["path"] and record["shape"] == list(shape)
            and np.dtype(record["dtype"]) == dtype and record["order"] == "C", "geometry array record differs")
    try:
        value = np.load(io.BytesIO(bound(path, record["sha256"])), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise B2VerificationError(f"invalid NPY: {path}") from error
    require(value.shape == shape and value.dtype == dtype and value.flags.c_contiguous, f"array layout differs: {path}")
    require(record["semantic_digest"] == geometry_semantic(value), f"semantic digest differs: {path}")
    return value


def empirical(queries: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    require(queries.shape == (QUERIES, DIMENSIONS) and candidates.shape == (COHORT, DIMENSIONS)
            and np.isfinite(queries).all() and np.isfinite(candidates).all(), "raw geometry differs")
    q = np.zeros_like(queries); c = np.zeros_like(candidates); constants = []
    for column in range(DIMENSIONS):
        ordered = np.sort(queries[:, column], kind="stable")
        if ordered[0] == ordered[-1]:
            constants.append(column)
            c[candidates[:, column] < ordered[0], column] = -1
            c[candidates[:, column] > ordered[0], column] = 1
            continue
        for source, target in ((queries, q), (candidates, c)):
            left = np.searchsorted(ordered, source[:, column], side="left")
            right = np.searchsorted(ordered, source[:, column], side="right")
            target[:, column] = np.clip((left + right + 1 - (QUERIES + 1)) / (QUERIES - 1), -1, 1)
    q[q == 0] = 0.; c[c == 0] = 0.
    return q, c, constants


def verify_candidate_sources(
    prereg: Mapping[str, Any], raw_candidates: np.ndarray, audit_rows: Sequence[Mapping[str, Any]],
) -> None:
    manifest = prereg["authorities"]["external_source_manifest"]
    records = {row["symbol"]: row for row in manifest["stock_files"]}
    require(len(records) == manifest["stock_files_count"] == len(manifest["required_symbols"])
            and sorted(records) == manifest["required_symbols"], "external source inventory differs")
    for record in (manifest["config"], manifest["benchmark"], *manifest["stock_files"]):
        bound(Path(record["path"]), record["sha256"], record["bytes"])
    config = load_config(Path(manifest["config"]["path"])); spec = config.datasets.get("nasdaq")
    require(spec is not None and spec.adapter == "directory" and spec.benchmark is not None
            and spec.benchmark.path == Path(manifest["benchmark"]["path"]), "NASDAQ source resolver differs")
    source = source_from_spec(spec); require(type(source) is DirectorySource, "NASDAQ directory source differs")
    for symbol, record in records.items():
        require(source._files.get(symbol) == Path(record["path"]), f"stock resolver differs: {symbol}")
    benchmark = source.load_benchmark(); require(benchmark is not None, "benchmark is absent")
    population = prereg["authorities"]["population"]
    require(len(audit_rows) == len(population["episodes"]) == COHORT, "candidate audit inventory differs")

    def one(item: tuple[int, Mapping[str, Any]]) -> tuple[int, np.ndarray, dict[str, Any]]:
        index, row = item; episode_id = row["episode_id"]; cutoff = pd.Timestamp(int(row["cutoff_ns"]))
        stock = source.load(InstrumentKey("nasdaq", row["symbol"])); eligible = stock.loc[pd.to_datetime(stock["timestamp"]) <= cutoff]
        require(len(eligible) >= 252, f"candidate history differs: {episode_id}")
        window = eligible.tail(252).copy().reset_index(drop=True); actual = pd.Timestamp(window["timestamp"].iloc[-1])
        key = EpisodeKey(InstrumentKey("nasdaq", row["symbol"]), actual, 252, "dense-v1")
        require(actual == cutoff and key.id == episode_id, f"candidate identity differs: {episode_id}")
        stock_prefix = asdict(causal_prefix_digest(stock, cutoff)); benchmark_prefix = asdict(causal_prefix_digest(benchmark, cutoff))
        represented = represent(Episode(key, window, benchmark.loc[pd.to_datetime(benchmark["timestamp"]) <= cutoff].copy(), "A"))
        vector = np.r_[represented.coarse[:96].astype(np.float64),
                       represented.stage.astype(np.float64).reshape(12, 4)[:, :3].ravel(),
                       represented.structural[:9].astype(np.float64)]
        vector = np.ascontiguousarray(vector.astype("<f8")); vector[vector == 0] = 0.
        audit = {"episode_id": episode_id, "symbol": row["symbol"], "cutoff_ns": row["cutoff_ns"],
                 "lookback": 252, "representation_version": "dense-v1",
                 "representation_digest": representation_input_digest(represented),
                 "stock_prefix": stock_prefix, "benchmark_prefix": benchmark_prefix}
        return index, vector, audit

    with ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="b2-verifier-source") as pool:
        reconstructed = list(pool.map(one, enumerate(population["episodes"])))
    reconstructed.sort()
    require(np.array_equal(np.asarray([row[1] for row in reconstructed]), raw_candidates),
            "candidate OHLCV reconstruction differs")
    require([row[2] for row in reconstructed] == list(audit_rows), "candidate reconstruction audit differs")
    for record in (manifest["config"], manifest["benchmark"], *manifest["stock_files"]):
        bound(Path(record["path"]), record["sha256"], record["bytes"])


_GQ: np.ndarray | None = None
_GC: np.ndarray | None = None
_GQQ: np.ndarray | None = None
_GQC: np.ndarray | None = None


def distance_chunk(bounds_: tuple[int, int]) -> tuple[int, bool]:
    require(all(value is not None for value in (_GQ, _GC, _GQQ, _GQC)), "distance verifier state missing")
    start, stop = bounds_; q = _GQ; c = _GC; qq = _GQQ; qc = _GQC
    assert q is not None and c is not None and qq is not None and qc is not None
    for index in range(start, stop):
        expected_q = np.asarray([math.sqrt(math.fsum(map(float, (row-q[index])*(row-q[index]))) / DIMENSIONS) for row in q])
        expected_c = np.asarray([math.sqrt(math.fsum(map(float, (row-q[index])*(row-q[index]))) / DIMENSIONS) for row in c])
        if not np.array_equal(expected_q, qq[index]) or not np.array_equal(expected_c, qc[index]):
            return index, False
    return start, True


def specificity(distances: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    result = np.full(distances.shape, np.nan)
    for row in range(len(distances)):
        ids = np.flatnonzero(eligible[row]); values = distances[row, ids]
        ordered = np.sort(values, kind="stable")
        left = np.searchsorted(ordered, values, side="left")
        right = np.searchsorted(ordered, values, side="right")
        result[row, ids] = (left + right) / (2.0 * len(ids))
    return result


def geometry(root: Path, prereg: Mapping[str, Any], h1: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    result = load(root / GEOMETRY / "RESULT.json")
    exact(result, ("arrays", "claims", "geometry_digest", "ids", "implementation_commit", "passed",
                   "preregistration_commit", "preregistration_digest", "schema_version", "source_manifest",
                   "status", "transform_audit"), "geometry result")
    require(result["geometry_digest"] == stable({k: v for k, v in result.items() if k != "geometry_digest"}), "geometry digest differs")
    require(result["schema_version"] == "m04r14-r1b-b2-geometry-v1" and result["passed"] is True
            and result["status"] == "geometry_complete_pending_independent_verification"
            and result["preregistration_commit"] == h1
            and result["preregistration_digest"] == prereg["preregistration_digest"]
            and result["implementation_commit"] == prereg["implementation_commit"], "geometry lineage differs")
    ids_record = exact(result["ids"], ("candidate_ids_digest", "path", "population_digest", "query_ids_digest", "sha256"), "geometry IDs")
    identities = decode(bound(root / GEOMETRY / "IDS.json", ids_record["sha256"]), root / GEOMETRY / "IDS.json")
    exact(identities, ("candidate_ids", "candidate_ids_digest", "candidate_keys", "population_digest", "query_ids", "query_ids_digest", "schema_version"), "geometry identities")
    pop = prereg["authorities"]["population"]
    require(identities["query_ids"] == pop["query_ids"] and identities["candidate_ids"] == pop["cohort_ids"]
            and identities["query_ids_digest"] == ids_record["query_ids_digest"] == id_digest(pop["query_ids"])
            and identities["candidate_ids_digest"] == ids_record["candidate_ids_digest"] == id_digest(pop["cohort_ids"])
            and identities["population_digest"] == ids_record["population_digest"] == pop["population_digest"], "geometry identity binding differs")
    expected_keys = [{"episode_id": row["episode_id"], "dataset_id": "nasdaq", "symbol": row["symbol"],
                      "cutoff_ns": row["cutoff_ns"], "lookback": 252, "representation_version": "dense-v1"}
                     for row in pop["episodes"]]
    require(identities["candidate_keys"] == expected_keys, "geometry candidate keys differ")
    specs = {
        "raw_queries": ((QUERIES, DIMENSIONS), np.dtype("<f8")), "raw_candidates": ((COHORT, DIMENSIONS), np.dtype("<f8")),
        "transformed_queries": ((QUERIES, DIMENSIONS), np.dtype("<f8")), "transformed_candidates": ((COHORT, DIMENSIONS), np.dtype("<f8")),
        "query_pair_distances": ((QUERIES, QUERIES), np.dtype("<f8")), "query_candidate_distances": ((QUERIES, COHORT), np.dtype("<f8")),
        "specificity_ranks": ((QUERIES, COHORT), np.dtype("<f8")), "causal_eligibility": ((QUERIES, COHORT), np.dtype("?")),
    }
    require(set(result["arrays"]) == set(specs), "geometry array inventory differs")
    arrays = {name: npy(root / GEOMETRY / result["arrays"][name]["path"], result["arrays"][name], *specs[name]) for name in specs}
    eligibility = np.zeros((QUERIES, COHORT), dtype=np.bool_); qindex = {q: i for i, q in enumerate(pop["query_ids"])}
    episodes = {row["episode_id"]: row for row in pop["episodes"]}
    for column, episode_id in enumerate(pop["cohort_ids"]):
        values = episodes[episode_id]["eligible_query_ids"]
        require(len(values) == len(set(values)) and set(values) <= set(qindex), "eligibility identities differ")
        eligibility[[qindex[value] for value in values], column] = True
    require(np.array_equal(arrays["causal_eligibility"], eligibility), "causal eligibility differs")
    qz, cz, constants = empirical(arrays["raw_queries"], arrays["raw_candidates"])
    require(np.array_equal(qz, arrays["transformed_queries"]) and np.array_equal(cz, arrays["transformed_candidates"]), "empirical transform differs")
    sealed = prereg["authorities"]["sealed_query_transform"]
    require(constants == sealed["constant_columns"] and balanced_array_digest(arrays["raw_queries"]) == sealed["raw_vector_digest"]
            and balanced_array_digest(qz) == sealed["transformed_digest"], "sealed query geometry differs")
    expected_rank = specificity(arrays["query_candidate_distances"], eligibility)
    require(np.array_equal(np.isnan(expected_rank), np.isnan(arrays["specificity_ranks"]))
            and np.array_equal(expected_rank[eligibility], arrays["specificity_ranks"][eligibility]), "specificity ranks differ")
    frozen_source = prereg["authorities"]["external_source_manifest"]
    require(result["source_manifest"] == {"manifest_digest": frozen_source["manifest_digest"],
            "source_lock_digest": frozen_source["source_lock_digest"], "stock_files_count": frozen_source["stock_files_count"],
            "config_sha256": frozen_source["config"]["sha256"], "benchmark_sha256": frozen_source["benchmark"]["sha256"],
            "stock_files_digest": stable(frozen_source["stock_files"])}, "geometry source binding differs")
    audit = result["transform_audit"]
    exact(audit, ("candidate_audits", "candidate_audits_digest", "constant_columns", "distance_chunk_rows",
                  "distance_shard_binding_digest", "distance_shards", "distance_workers", "full_specificity_denominator",
                  "performance", "query_reconstruction_workers", "reconstructed_query_audits_digest",
                  "sealed_integer_midrank_digest", "sealed_query_audits_digest", "sealed_query_ids_digest",
                  "sealed_raw_query_digest", "sealed_transform_sha256", "sealed_transformed_query_digest",
                  "synthetic_1_12_byte_identity_h0_test", "unsupported_candidates_included"), "geometry transform audit")
    require(audit["sealed_query_ids_digest"] == sealed["query_ids_digest"]
            and audit["sealed_raw_query_digest"] == sealed["raw_vector_digest"]
            and audit["sealed_transformed_query_digest"] == sealed["transformed_digest"]
            and audit["constant_columns"] == constants and audit["full_specificity_denominator"] == COHORT
            and audit["unsupported_candidates_included"] == COHORT-PRIMARY
            and audit["query_reconstruction_workers"] == WORKERS and audit["distance_workers"] == WORKERS
            and audit["distance_chunk_rows"] == 64 and audit["distance_shards"] == 52,
            "geometry transform audit differs")
    require(audit["candidate_audits_digest"] == stable(audit["candidate_audits"]), "candidate audit digest differs")
    verify_candidate_sources(prereg, arrays["raw_candidates"], audit["candidate_audits"])
    global _GQ, _GC, _GQQ, _GQC
    _GQ, _GC = qz, cz; _GQQ = arrays["query_pair_distances"]; _GQC = arrays["query_candidate_distances"]
    ranges = [(start, min(start + 64, QUERIES)) for start in range(0, QUERIES, 64)]
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context("fork")) as pool:
        checked = list(pool.map(distance_chunk, ranges))
    require(all(ok for _, ok in checked), f"geometry distance replay differs at {[i for i, ok in checked if not ok][:3]}")
    require(result["claims"] == {"b2_statistics_computed": False, "predictive_claim_authorized": False,
            "production_promotion_authorized": False, "real_forward_outcomes_accessed": False}, "geometry claims differ")
    require(set(path.name for path in (root/GEOMETRY).iterdir()) == {"RESULT.json", "IDS.json", *(record["path"] for record in result["arrays"].values())},
            "geometry publication closure differs")
    return result, identities, arrays


class Plan:
    def __init__(self, episode_id: str, column: int, observed: tuple[int, ...], groups: tuple[Any, ...], active: tuple[int, ...]):
        self.episode_id, self.column, self.observed, self.groups, self.active = episode_id, column, observed, groups, active


def group(eligible: Sequence[int], observed: Sequence[int], cells: Sequence[Hashable]) -> tuple[tuple[int, tuple[int, ...]], ...]:
    needed = Counter(cells[index] for index in observed)
    ordered = sorted(needed, key=lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")))
    result = tuple((needed[cell], tuple(index for index in eligible if cells[index] == cell)) for cell in ordered)
    require(all(len(members) >= count for count, members in result), "conditional cell unsupported")
    return result


def plans(prereg: Mapping[str, Any], identities: Mapping[str, Any], eligibility: np.ndarray) -> tuple[tuple[str, ...], tuple[Plan, ...], np.ndarray]:
    pop = prereg["authorities"]["population"]; qids = tuple(identities["query_ids"]); cids = tuple(identities["candidate_ids"])
    qi = {q: i for i, q in enumerate(qids)}; ci = {e: i for i, e in enumerate(cids)}
    require(len(qi) == QUERIES and len(ci) == COHORT, "population duplicates")
    rows = pop["query_cells"]; require([row["query_id"] for row in rows] == list(qids), "query cell order differs")
    n0 = [tuple(row["n0"]) for row in rows]
    labels = {k: np.asarray([int(row["labels"][str(k)]) for row in rows], dtype=np.int8) for k in (8, 12, 16)}
    designs = (n0, *([value + (int(row["labels"][str(k)]),) for value, row in zip(n0, rows)] for k in (8, 12, 16)))
    by = {row["episode_id"]: row for row in pop["episodes"]}; primary = tuple(pop["primary_ids"]); output = []
    for episode_id in primary:
        row = by[episode_id]; eligible = tuple(qi[q] for q in row["eligible_query_ids"]); observed = tuple(qi[q] for q in row["observed_query_ids"])
        require(np.array_equal(np.flatnonzero(eligibility[:, ci[episode_id]]), np.asarray(eligible)), "episode eligibility differs")
        groups = tuple(group(eligible, observed, cells) for cells in designs)
        active = tuple(sorted({index for _, members in groups[0] for index in members}))
        output.append(Plan(episode_id, ci[episode_id], observed, groups, active))
    require(len(output) == PRIMARY and sum(len(value.observed) for value in output) == PRIMARY_LINKS, "primary inventory differs")
    return primary, tuple(output), labels[8]


def lp(value: bytes) -> bytes:
    return len(value).to_bytes(4, "big") + value


def priority(contract: str, replicate: int, episode: str | None, query: str, shared: bool) -> bytes:
    parts = [PRIORITY_DOMAIN, bytes.fromhex(contract), PRIORITY_FAMILY,
             b"shared-query" if shared else b"episode", replicate.to_bytes(4, "big")]
    if not shared:
        assert episode is not None; parts.append(episode.encode())
    parts.append(query.encode()); return sha256(b"".join(lp(value) for value in parts)).digest()


def select(groups: Sequence[tuple[int, Sequence[int]]], priorities: Mapping[int, bytes], qids: Sequence[str]) -> tuple[int, ...]:
    chosen = []
    for count, members in groups:
        chosen.extend(heapq.nsmallest(count, members, key=lambda i: (priorities[i], qids[i])))
    result = tuple(sorted(chosen, key=lambda i: qids[i])); require(len(result) == len(set(result)), "draw duplicates")
    return result


def breadth(selected: Sequence[int], labels: np.ndarray) -> float:
    counts = Counter(int(labels[index]) for index in selected); n = len(selected)
    value = math.exp(-math.fsum((count/n) * math.log(count/n) for _, count in sorted(counts.items()))) / min(8, n)
    return min(1.0, value)


class Replay:
    def __init__(self, qids: tuple[str, ...], plans_: tuple[Plan, ...], qq: np.ndarray, ranks: np.ndarray, labels: np.ndarray, contract: str):
        self.qids, self.plans, self.qq, self.ranks, self.labels, self.contract = qids, plans_, qq, ranks, labels, contract


_REPLAY: Replay | None = None


def evaluate(plan: Plan, selected: Sequence[int], state: Replay) -> tuple[float, float]:
    ids = sorted(selected); cohesion = math.fsum(float(state.qq[a, b]) for pos, a in enumerate(ids) for b in ids[pos+1:]) / math.comb(len(ids), 2)
    specific = math.fsum(float(state.ranks[index, plan.column]) for index in ids) / len(ids)
    return cohesion, specific


def replay_chunk(bounds_: tuple[int, int]) -> tuple[int, dict[str, np.ndarray]]:
    require(_REPLAY is not None, "null replay state missing"); state = _REPLAY; start, stop = bounds_
    count = stop-start; output = {name: np.empty((count, PRIMARY, 2), dtype="<f8") for name in NULLS}
    output["episode_n0_breadth"] = np.empty((count, PRIMARY), dtype="<f8")
    shared = [{index: priority(state.contract, replicate, None, q, True) for index, q in enumerate(state.qids)} for replicate in range(start, stop)]
    for position, plan in enumerate(state.plans):
        for local, replicate in enumerate(range(start, stop)):
            episode_priorities = {index: priority(state.contract, replicate, plan.episode_id, state.qids[index], False) for index in plan.active}
            selected_sets = [select(groups, episode_priorities, state.qids) for groups in plan.groups]
            shared_sets = [select(groups, shared[local], state.qids) for groups in plan.groups[:2]]
            for name, selected in zip(NULLS, (*selected_sets, *shared_sets)):
                output[name][local, position] = evaluate(plan, selected, state)
            output["episode_n0_breadth"][local, position] = breadth(selected_sets[0], state.labels)
    return start, output


def load_shards(root: Path, result: Mapping[str, Any], bindings: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    shard_root = root / PRODUCER / "shards"; ranges = [(i, i+SHARD_SIZE) for i in range(0, REPLICATES, SHARD_SIZE)]
    require(len(ranges) == 128 and set(path.name for path in shard_root.iterdir()) == {f"replicates-{a:04d}-{b-1:04d}" for a,b in ranges}, "shard directory closure differs")
    tables = {name: np.empty((REPLICATES, PRIMARY) + (() if name.endswith("breadth") else (2,)), dtype="<f8") for name in ARRAYS}
    manifest = []
    for start, stop in ranges:
        directory = shard_root / f"replicates-{start:04d}-{stop-1:04d}"; seal_path = directory / "SHARD.json"; seal_bytes = snapshot(seal_path); seal = decode(seal_bytes, seal_path)
        exact(seal, ("arrays", "bindings", "episodes", "primary_ids_digest", "replicate_start", "replicate_stop", "schema_version", "shard_digest", "status"), "shard")
        require(seal["shard_digest"] == stable({k:v for k,v in seal.items() if k != "shard_digest"})
                and seal["schema_version"] == "m04r14-r1b-b2-localization-shard-v1" and seal["status"] == "complete"
                and seal["replicate_start"] == start and seal["replicate_stop"] == stop and seal["episodes"] == PRIMARY
                and seal["bindings"] == bindings and seal["primary_ids_digest"] == bindings["primary_ids_digest"], "shard seal differs")
        require(set(seal["arrays"]) == set(ARRAYS) and set(path.name for path in directory.iterdir()) == {"SHARD.json", *(f"{name}.npy" for name in ARRAYS)}, "shard file closure differs")
        for name in ARRAYS:
            record = seal["arrays"][name]; exact(record, ("content_digest", "dtype", "path", "sha256", "shape"), "shard array")
            shape = (SHARD_SIZE, PRIMARY) + (() if name.endswith("breadth") else (2,))
            require(record["path"] == f"{name}.npy" and record["shape"] == list(shape) and record["dtype"] == "<f8", "shard array record differs")
            value = np.load(io.BytesIO(bound(directory / record["path"], record["sha256"])), allow_pickle=False)
            require(value.shape == shape and value.dtype == np.dtype("<f8") and np.isfinite(value).all()
                    and record["content_digest"] == array_digest(value), "shard array content differs")
            if name.endswith("breadth"):
                require(((value > 0) & (value <= 1)).all(), "breadth bound differs")
            else:
                require((value >= 0).all() and ((value[:,:,1] > 0) & (value[:,:,1] < 1)).all(), "null statistic range differs")
            tables[name][start:stop] = value
        manifest.append({"path": f"shards/{directory.name}/SHARD.json", "sha256": sha256(seal_bytes).hexdigest(),
                         "shard_digest": seal["shard_digest"], "start": start, "stop": stop})
    require(result["shards"] == manifest, "result shard manifest differs")
    return tables, manifest


def mean(values: Sequence[float] | np.ndarray) -> float:
    return math.fsum(map(float, values)) / len(values)


def summaries(table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    aggregate = np.asarray([[mean(row[:, metric]) for metric in range(2)] for row in table])
    episode = np.asarray([[mean(table[:, e, metric]) for metric in range(2)] for e in range(PRIMARY)])
    return aggregate, episode


def effect(observed: np.ndarray, null: np.ndarray) -> dict[str, Any]:
    oc, nc = mean(observed[:,0]), mean(null[:,0]); ce = (nc-oc)/nc if nc > 0 else None
    se = mean(null[:,1])-mean(observed[:,1]); needed = (3*len(observed)+4)//5
    cc = int(np.count_nonzero(observed[:,0] < null[:,0])); sc = int(np.count_nonzero(observed[:,1] < null[:,1]))
    return {"observed_cohesion": oc, "null_cohesion": nc, "cohesion_relative_improvement": ce,
            "specificity_improvement": se, "cohesion_improved_episodes": cc,
            "specificity_improved_episodes": sc, "required_improved_episodes": needed,
            "practical_pass": ce is not None and ce >= .05 and se >= .05 and cc >= needed and sc >= needed}


def descriptive(value: Mapping[str, Any]) -> dict[str, Any]:
    return {k:v for k,v in value.items() if k not in {"practical_pass", "required_improved_episodes"}}


def selected_diagnostics(root: Path, prereg: Mapping[str, Any], primary: Sequence[str]) -> dict[str, Any]:
    pop = prereg["authorities"]["population"]; episodes = {row["episode_id"]:row for row in pop["episodes"]}; wanted = set(primary)
    expected = {(e,q) for e in primary for q in episodes[e]["observed_query_ids"]}; require(len(expected)==PRIMARY_LINKS, "selected-link population differs")
    links=[]; seen=set(); manifest=prereg["authorities"]["case_manifest"]; require(len(manifest)==QUERIES, "case manifest differs")
    for record in manifest:
        exact(record,("bytes","path","sha256"),"case record"); relative=Path(record["path"])
        require(not relative.is_absolute() and ".." not in relative.parts and relative.parent==CASES,"case path differs")
        case=decode(bound(root/relative,record["sha256"],record["bytes"]),root/relative); query=case["query_episode_id"]; require(query not in seen,"duplicate case query"); seen.add(query)
        selected=set()
        for match in case["matches"]:
            episode=match["episode_id"]; require(episode not in selected,"duplicate case episode"); selected.add(episode)
            if episode in wanted:
                total=match["total_distance"]; market=match["component_distances"]["market_context"]
                require(all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x>=0 for x in (total,market)),"selected distance differs")
                links.append({"episode_id":episode,"query_id":query,"production_total":float(total),"raw_market_context":float(market)})
    links.sort(key=lambda row:(row["episode_id"],row["query_id"])); require({(r["episode_id"],r["query_id"]) for r in links}==expected and len(links)==PRIMARY_LINKS,"selected links differ")
    rows=[]
    for episode in primary:
        values=[row for row in links if row["episode_id"]==episode]
        rows.append({"episode_id":episode,"selected_links":len(values),"production_total_mean":mean([r["production_total"] for r in values]),"raw_market_context_mean":mean([r["raw_market_context"] for r in values])})
    return {"status":"descriptive_only","scope":{"primary_episodes":PRIMARY,"selected_links":PRIMARY_LINKS},
            "unit":"selected-link means within episode, then equal-episode means","links":links,"links_digest":stable(links),
            "episode_rows":rows,"episode_rows_digest":stable(rows),
            "selected_link_aggregate":{"production_total":mean([r["production_total"] for r in links]),"raw_market_context":mean([r["raw_market_context"] for r in links])},
            "equal_episode_aggregate":{"production_total":mean([r["production_total_mean"] for r in rows]),"raw_market_context":mean([r["raw_market_context_mean"] for r in rows])},
            "claims":{"null_computed":False,"pvalue_computed":False,"decision_input":False,"rescue_authorized":False}}


def scientific(state: Replay, tables: Mapping[str,np.ndarray], diagnostics: Mapping[str,Any]) -> dict[str,Any]:
    observed=np.asarray([evaluate(plan,plan.observed,state) for plan in state.plans]); observed_breadth=np.asarray([breadth(plan.observed,state.labels) for plan in state.plans])
    sums={name:summaries(tables[name]) for name in NULLS}; effects={name:effect(observed,sums[name][1]) for name in NULLS}
    n1agg=sums["episode_n1"][0]; pvalues=[(1+int(np.count_nonzero(n1agg[:,m] <= mean(observed[:,m]))))/(REPLICATES+1) for m in range(2)]
    parities=np.asarray([int(e,16)&1 for e in (p.episode_id for p in state.plans)]); splits={}
    for split in (0,1):
        mask=parities==split; value=effect(observed[mask],sums["episode_n1"][1][mask])
        splits[str(split)]={"episodes":int(mask.sum()),"cohesion_relative_improvement":value["cohesion_relative_improvement"],"specificity_improvement":value["specificity_improvement"],"gate":"both effects >= 0.05; no p-value or episode-count gate","gate_passed":value["cohesion_relative_improvement"] is not None and value["cohesion_relative_improvement"]>=.05 and value["specificity_improvement"]>=.05}
    loeo=[]
    for omitted,plan in enumerate(state.plans):
        mask=np.arange(PRIMARY)!=omitted; value=effect(observed[mask],sums["episode_n1"][1][mask])
        loeo.append({"omitted_episode_id":plan.episode_id,"cohesion_relative_improvement":value["cohesion_relative_improvement"],"specificity_improvement":value["specificity_improvement"]})
    shared_ok={name:effects[name]["cohesion_relative_improvement"] is not None and effects[name]["cohesion_relative_improvement"]>0 and effects[name]["specificity_improvement"]>0 for name in ("shared_n0","shared_n1")}
    robustness=all(row["cohesion_relative_improvement"] is not None and row["cohesion_relative_improvement"]>0 and row["specificity_improvement"]>0 for row in loeo) and all(row["gate_passed"] for row in splits.values()) and all(shared_ok.values())
    reasons=[]
    if any(p>.01 for p in pvalues): reasons.append("both N1 lower-tail p-values must be at most 0.01")
    if not effects["episode_n1"]["practical_pass"]: reasons.append("N1 practical effects or 60 percent episode counts failed")
    if not effects["episode_n0"]["practical_pass"]: reasons.append("N0 corroboration effects or 60 percent episode counts failed")
    status="structurally_localized" if not reasons and robustness else ("unresolved" if not robustness else "not_established")
    published={"structurally_localized":"established_pending_independent_verification","not_established":"not_established_pending_independent_verification","unresolved":"unresolved"}[status]
    decision={"status":status,"reasons":reasons,"n1_pvalues":pvalues,"n1_effects":effects["episode_n1"],"n0_effects":effects["episode_n0"]}
    return {"status":published,"decision":decision,
            "observed":{"statistics_digest":array_digest(observed),"breadth_digest":array_digest(observed_breadth),"cohesion":mean(observed[:,0]),"specificity":mean(observed[:,1]),"breadth":mean(observed_breadth)},
            "primary_effects":{name:effects[name] for name in ("episode_n0","episode_n1")},
            "shared_priority_effects":{name:descriptive(effects[name]) for name in ("shared_n0","shared_n1")},
            "descriptive_effect_sensitivity":{name:{**descriptive(effects[name]),"interpretation":"descriptive only; no p-value, gate, or rescue"} for name in ("episode_k12","episode_k16")},
            "selected_link_production_diagnostics":diagnostics,
            "robustness":{"episode_hash_splits":splits,"leave_one_episode_out":loeo,"all_leave_one_out_effects_strictly_positive":all(row["cohesion_relative_improvement"] is not None and row["cohesion_relative_improvement"]>0 and row["specificity_improvement"]>0 for row in loeo),"shared_n0_effects_strictly_positive":shared_ok["shared_n0"],"shared_n1_effects_strictly_positive":shared_ok["shared_n1"]},
            "n0_breadth":{"replicate_mean":mean(tables["episode_n0_breadth"].ravel(order="C")),"table_digest":array_digest(tables["episode_n0_breadth"])},
            "null_table_digests":{name:array_digest(tables[name]) for name in ARRAYS},
            "claims":{"predictive_claim_authorized":False,"production_promotion_authorized":False,"ranking_change_authorized":False,"outcomes_opened":False,"independent_verification_required":True}}


def publish(path: Path, payload: Mapping[str,Any]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); require(not path.exists() and not path.is_symlink(),"verifier output exists")
    temporary=path.parent/f".{path.name}.tmp-{os.getpid()}-{uuid4().hex}"
    content=(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n").encode(); descriptor=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(descriptor,"wb",closefd=True) as handle: handle.write(content); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary,path); directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY); os.fsync(directory); os.close(directory)
    finally: temporary.unlink(missing_ok=True)


def run(root: Path) -> dict[str,Any]:
    root=root.resolve(); require(not (root/OUTPUT).exists() and not (root/OUTPUT).is_symlink(),"create-only verifier output exists")
    prereg=load(root/PREREG); exact(prereg,("authorities","b2_priority_contract_digest","environment","implementation_commit","preregistration_digest","runtime_sha256","schema_version","specification"),"preregistration")
    require(prereg["preregistration_digest"]==stable({k:v for k,v in prereg.items() if k!="preregistration_digest"}),"preregistration digest differs")
    head,h1=lineage(root,prereg); geo,ids,arrays=geometry(root,prereg,h1); primary,plans_,labels=plans(prereg,ids,arrays["causal_eligibility"])
    result=load(root/PRODUCER/"RESULT.json"); exact(result,("bindings","identity","inventory","performance","result_digest","result_digest_scope","schema_version","scientific","shards"),"B2 result")
    require(result["result_digest"]==stable({k:v for k,v in result.items() if k not in {"result_digest","performance"}}),"B2 result digest differs")
    require(result["schema_version"] == "m04r14-r1b-b2-localization-result-v1"
            and result["result_digest_scope"] == "all fields except result_digest and nonsemantic performance",
            "B2 result schema differs")
    performance = exact(result["performance"], ("claim", "elapsed_seconds", "new_shards", "shard_replicates", "workers"), "B2 performance")
    require(performance["claim"] == "measurement only; no unsupported fastest claim"
            and performance["workers"] == WORKERS and performance["shard_replicates"] == SHARD_SIZE
            and performance["new_shards"] == 128 and isinstance(performance["elapsed_seconds"], (int,float))
            and not isinstance(performance["elapsed_seconds"], bool) and math.isfinite(performance["elapsed_seconds"])
            and performance["elapsed_seconds"] >= 0, "B2 performance record differs")
    bindings={"h1_commit":h1,"preregistration_digest":prereg["preregistration_digest"],"priority_contract_digest":prereg["b2_priority_contract_digest"],"population_digest":prereg["authorities"]["population"]["population_digest"],"runtime_sha256":prereg["runtime_sha256"]["experiments/m04r/m04r14_r1b_b2_localization.py"],"geometry_digest":geo["geometry_digest"],"query_ids_digest":ids["query_ids_digest"],"candidate_ids_digest":ids["candidate_ids_digest"],"primary_ids_digest":stable(list(primary))}
    require(result["bindings"]==bindings and result["inventory"]=={"queries":QUERIES,"cohort_episodes":COHORT,"cohort_links":COHORT_LINKS,"primary_episodes":PRIMARY,"primary_links":PRIMARY_LINKS,"unsupported_episodes":12,"unsupported_links":74,"replicates":REPLICATES,"shards":128},"B2 binding/inventory differs")
    require(result["identity"]=={"primary_ids":list(primary),"unsupported_ids":sorted(set(ids["candidate_ids"])-set(primary))},"B2 identities differ")
    tables,_=load_shards(root,result,bindings)
    global _REPLAY; _REPLAY=Replay(tuple(ids["query_ids"]),plans_,arrays["query_pair_distances"],arrays["specificity_ranks"],labels,prereg["b2_priority_contract_digest"])
    ranges=[(i,i+SHARD_SIZE) for i in range(0,REPLICATES,SHARD_SIZE)]
    with ProcessPoolExecutor(max_workers=WORKERS,mp_context=multiprocessing.get_context("fork")) as pool:
        replayed=list(pool.map(replay_chunk,ranges))
    for start,chunk in replayed:
        for name in ARRAYS: require(np.array_equal(chunk[name],tables[name][start:start+SHARD_SIZE]),f"independent null replay differs: {name}/{start}")
    diagnostics=selected_diagnostics(root,prereg,primary); expected_scientific=scientific(_REPLAY,tables,diagnostics)
    require(result["scientific"]==expected_scientific,"B2 scientific result reconstruction differs")
    require(result["scientific"]["status"]=="established_pending_independent_verification" and result["scientific"]["decision"]["status"]=="structurally_localized","verified scientific decision differs")
    require(result["scientific"]["claims"]=={"predictive_claim_authorized":False,"production_promotion_authorized":False,"ranking_change_authorized":False,"outcomes_opened":False,"independent_verification_required":True},"no-claim boundary differs")
    require(set(path.name for path in (root/PRODUCER).iterdir()) == {"RESULT.json", "shards"}, "B2 publication closure differs")
    require(load(root/PRODUCER/"RESULT.json") == result and load(root/GEOMETRY/"RESULT.json") == geo,
            "sealed producer evidence changed during verification")
    require(lineage(root,prereg) == (head,h1), "lineage changed during verification")
    state={"schema_version":SCHEMA,"passed":True,"status":"verified_structurally_localized","verified_result_digest":result["result_digest"],"verified_result_sha256":file_sha(root/PRODUCER/"RESULT.json"),"verified_geometry_digest":geo["geometry_digest"],"verified_preregistration_digest":prereg["preregistration_digest"],"preregistration_commit":h1,"verifier_commit":head,"verifier_runtime_sha256":{path:file_sha(root/path) for path in RUNTIME},"inventory":result["inventory"],"gates":{"h0_h1_lineage_and_runtime":True,"geometry_exact_replay":True,"all_4096_replicates_exact_replay":True,"all_128_shards_and_digests":True,"decision_and_robustness_reconstructed":True,"breadth_exactly_bounded":True,"selected_link_diagnostics_reconstructed":True,"no_outcome_or_promotion_boundary":True},"claims":{"predictive_claim_authorized":False,"production_promotion_authorized":False,"ranking_change_authorized":False,"outcomes_opened":False}}
    payload={**state,"verification_digest":stable(state),"created_at":datetime.now(timezone.utc).isoformat()}; publish(root/OUTPUT/"VERIFIED.json",payload); return payload


def main(argv: Sequence[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--repository",type=Path,default=Path.cwd()); args=parser.parse_args(argv)
    try: result=run(args.repository)
    except B2VerificationError as error: print(f"B2 verification unresolved: {error}",file=os.sys.stderr); return 2
    print(json.dumps({"passed":result["passed"],"verification_digest":result["verification_digest"]},sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())
