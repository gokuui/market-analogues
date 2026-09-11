"""Independently reconstruct every query in the sealed R2 full mode store."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path
from resource import getrusage, RUSAGE_CHILDREN, RUSAGE_SELF
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.m04r import verify_m04r15_r2_bounded_real_poc as oracle


SCHEMA = "m04r15-r2-full-mode-store-verification-v2"
CONTRACT = Path("config/m04r15-r2-fixed-neighbor-modes-contract-v1.json")
PREREG = Path("experiments/m04r/m04r15_r2_full_mode_store_v2_preregistered.json")
SOURCE = Path("config/data/analogues/m04r14/t14-09-full-outcome-store-v1")
STORE = Path("config/data/analogues/m04r15/r2-full-mode-store-v2")
OUTPUT = Path("config/data/analogues/m04r15/r2-full-mode-store-v2-verification")
RUNTIME = (
    "experiments/m04r/verify_m04r15_r2_full_mode_store.py",
    "experiments/m04r/verify_m04r15_r2_bounded_real_poc.py",
    "experiments/m04r/verify_m04r15_r2_stability_synthetic_gate.py",
    "tests/test_verify_m04r15_r2_full_mode_store.py",
)
PATH_COLUMNS = (
    "benchmark_relative_close_return", "close_return", "contract_digest", "cutoff",
    "episode_id", "expected_session_match", "source_content_digest",
    "source_fingerprint", "step", "timestamp",
)
DICTIONARY_COLUMNS = (
    "contract_digest", "cutoff", "episode_id", "source_content_digest",
    "source_fingerprint", "timestamp",
)
LINK_COLUMNS = (
    "query_case_id", "query_symbol", "match_rank", "matched_episode_id",
    "matched_symbol", "matched_cutoff", "source_fingerprint",
    "outcome_eligibility_json",
)
WORKERS = 12

_TABLE: pa.Table | None = None
_SLICES: dict[str, tuple[int, int]] | None = None
_LINKS: dict[str, tuple[dict[str, Any], ...]] | None = None
_ACTUAL: dict[str, dict[str, Any]] | None = None
_PREREG_DIGEST: str | None = None


class FullModeVerificationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FullModeVerificationError(message)


def _stable(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode() + b"\n"


def _json(path: Path) -> tuple[dict[str, Any], bytes]:
    _require(path.is_file() and not path.is_symlink(), f"regular JSON required: {path}")
    raw = path.read_bytes()

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            _require(key not in value, f"duplicate JSON key: {path}:{key}")
            value[key] = item
        return value

    try:
        value = json.loads(raw, object_pairs_hook=pairs,
                           parse_constant=lambda item: (_ for _ in ()).throw(
                               FullModeVerificationError(f"nonfinite JSON: {path}:{item}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FullModeVerificationError(f"invalid JSON: {path}") from error
    _require(type(value) is dict, f"JSON object required: {path}")
    return value, raw


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20): digest.update(block)
    return digest.hexdigest()


def _git(repository: Path, *args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(("git", *args), cwd=repository, capture_output=True,
                            text=not binary, check=False)
    _require(result.returncode == 0, f"git {' '.join(args)} failed")
    return result.stdout if binary else result.stdout.strip()


def _groups(query_ids: Sequence[str], count: int) -> tuple[tuple[str, ...], ...]:
    values = tuple(sorted(map(str, query_ids)))
    _require(values and len(set(values)) == len(values) and count > 0,
             "query partition inputs differ")
    count = min(count, len(values)); width, remainder = divmod(len(values), count)
    result = []; offset = 0
    for index in range(count):
        size = width + (index < remainder)
        result.append(values[offset:offset + size]); offset += size
    return tuple(result)


def _coverage(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = Counter(view["selection"]["status"] for result in results
                       for view in result["views"].values())
    selected = Counter(str(view["selection"]["selected_k"]) for result in results
                       for view in result["views"].values())
    return {
        "schema_version": "m04r15-r2-full-mode-store-v2-coverage",
        "query_count": len(results), "view_count": len(results) * 2,
        "raw_link_count": sum(result["raw_links"] for result in results),
        "primary_member_count": sum(result["primary_members"] for result in results),
        "dependence_exclusion_count": sum(len(result["dependence_exclusions"])
                                          for result in results),
        "ineligible_member_occurrences": sum(len(view["ineligible_members"])
            for result in results for view in result["views"].values()) // 2,
        "invalid_member_occurrences": sum(len(view["invalid_members"])
            for result in results for view in result["views"].values()),
        "mode_status_counts": dict(sorted(statuses.items())),
        "selected_k_counts": dict(sorted(selected.items())),
    }


def _load_actual(root: Path, prereg: Mapping[str, Any]) -> tuple[
    tuple[str, ...], dict[str, dict[str, Any]], dict[str, Any], dict[str, Any], list[str]
]:
    _require(root.is_dir() and not root.is_symlink(), "regular producer store required")
    expected_entries = {"COVERAGE.json", "RESULTS.jsonl", "RUN_COMPLETED.json",
                        "RUN_STARTED.json", "SEALED.json", "partitions"}
    _require({path.name for path in root.iterdir()} == expected_entries,
             "producer store layout differs")
    seal, seal_raw = _json(root / "SEALED.json")
    seal_state = {key: value for key, value in seal.items() if key != "seal_digest"}
    _require(seal_raw == _canonical(seal) and seal.get("seal_digest") == _stable(seal_state),
             "producer seal differs")
    _require(all((seal.get("status") == "sealed", seal.get("query_count") == 3270,
                  seal.get("view_count") == 6540, seal.get("partition_count") == 128,
                  seal.get("preregistration_digest") == prereg.get("preregistration_digest"),
                  seal.get("real_future_path_modes_computed") is True,
                  seal.get("descriptive_evidence_authorized") is True,
                  seal.get("independent_verification_complete") is False,
                  seal.get("predictive_claim_authorized") is False,
                  seal.get("production_promotion_authorized") is False,
                  seal.get("trading_claim_authorized") is False)), "producer boundary differs")
    results_path = root / "RESULTS.jsonl"
    _require(results_path.is_file() and not results_path.is_symlink(),
             "regular aggregate results required")
    lines = results_path.read_bytes().splitlines(keepends=True)
    _require(len(lines) == 3270 and sum(map(len, lines)) == seal["results_jsonl_bytes"] and
             _sha(root / "RESULTS.jsonl") == seal["results_jsonl_sha256"],
             "aggregate result file differs")
    results = []
    for line in lines:
        value = json.loads(line)
        _require(type(value) is dict and line == _canonical(value),
                 "noncanonical aggregate result row")
        results.append(value)
    query_ids = tuple(str(result.get("query_case_id", "")) for result in results)
    _require(tuple(sorted(set(query_ids))) == query_ids and _stable(results) ==
             seal["query_results_digest"], "aggregate query identities differ")
    actual = dict(zip(query_ids, results, strict=True))
    groups = _groups(query_ids, 128); partition_hashes = []
    for partition_id, group in enumerate(groups):
        directory = root / f"partitions/partition-{partition_id:04d}"
        _require(directory.is_dir() and not directory.is_symlink() and
                 {path.name for path in directory.iterdir()} == {"PARTITION.json"},
                 f"partition layout differs: {partition_id}")
        value, raw = _json(directory / "PARTITION.json")
        state = {key: item for key, item in value.items() if key != "partition_digest"}
        expected_results = [actual[query_id] for query_id in group]
        _require(all((raw == _canonical(value), value.get("partition_id") == partition_id,
                      value.get("contract_digest") == prereg["preregistration_digest"],
                      value.get("query_ids") == list(group),
                      value.get("query_ids_digest") == _stable(list(group)),
                      value.get("results") == expected_results,
                      value.get("results_digest") == _stable(expected_results),
                      value.get("partition_digest") == _stable(state))),
                 f"partition seal differs: {partition_id}")
        partition_hashes.append(sha256(raw).hexdigest())
    _require(_stable(partition_hashes) == seal["partition_sha256_digest"],
             "partition inventory digest differs")
    coverage, coverage_raw = _json(root / "COVERAGE.json")
    expected_coverage = _coverage(results)
    _require(coverage == expected_coverage and coverage_raw == _canonical(coverage) and
             _stable(coverage) == seal["coverage_digest"] and
             sha256(coverage_raw).hexdigest() == seal["coverage_sha256"],
             "independent coverage differs")
    return query_ids, actual, seal, coverage, partition_hashes


def _load_inputs(repository: Path) -> tuple[pa.Table, dict[str, tuple[int, int]],
                                             dict[str, tuple[dict[str, Any], ...]]]:
    path = repository / SOURCE / "future-paths.parquet"
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            table = pq.read_table(handle, columns=list(PATH_COLUMNS),
                                  read_dictionary=list(DICTIONARY_COLUMNS)).combine_chunks()
        after = os.fstat(descriptor)
        _require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                 (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
                 "future-path store changed during independent preload")
    finally:
        os.close(descriptor)
    _require(tuple(table.column_names) == PATH_COLUMNS and table.num_rows == 6_917_999,
             "future-path table differs")
    encoded = table.column("episode_id").chunk(0)
    _require(pa.types.is_dictionary(encoded.type) and encoded.null_count == 0,
             "episode encoding differs")
    codes = encoded.indices.to_numpy(zero_copy_only=True)
    boundaries = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    starts = np.concatenate(([0], boundaries)); stops = np.concatenate((boundaries, [len(codes)]))
    run_codes = codes[starts]
    _require(len(np.unique(run_codes)) == len(run_codes) == 56_325,
             "episode contiguity differs")
    dictionary = encoded.dictionary
    slices = {str(dictionary[int(code)].as_py()): (int(start), int(stop))
              for code, start, stop in zip(run_codes, starts, stops, strict=True)}
    links_path = repository / SOURCE / "query-match-links.parquet"
    descriptor = os.open(links_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            frame = pq.read_table(handle, columns=list(LINK_COLUMNS)).to_pandas()
        after = os.fstat(descriptor)
        _require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                 (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
                 "query-link store changed during independent preload")
    finally:
        os.close(descriptor)
    frame = frame.sort_values(["query_case_id", "match_rank"], kind="stable")
    links = {str(query): tuple(group.to_dict("records"))
             for query, group in frame.groupby("query_case_id", sort=True)}
    _require(len(frame) == 65_400 and len(links) == 3_270 and
             all([int(row["match_rank"]) for row in rows] == list(range(1, 21))
                 for rows in links.values()), "query-link inventory differs")
    return table, slices, links


def _expected(query_id: str) -> dict[str, Any]:
    _require(_TABLE is not None and _SLICES is not None and _LINKS is not None,
             "independent worker inputs absent")
    rows = _LINKS[query_id]; query_symbol = str(rows[0]["query_symbol"]); seen = set()
    primary = []; excluded = []
    for row in rows:
        member = {"rank": int(row["match_rank"]), "episode": str(row["matched_episode_id"]),
                  "symbol": str(row["matched_symbol"]), "cutoff": str(row["matched_cutoff"]),
                  "fingerprint": str(row["source_fingerprint"])}
        if member["symbol"] == query_symbol: excluded.append([member["episode"], "query_symbol_memory"])
        elif member["symbol"] in seen: excluded.append([member["episode"], "duplicate_matched_symbol"])
        else: primary.append(member); seen.add(member["symbol"])
    cohort = oracle._cohort(primary)
    eligibility = {str(row["matched_episode_id"]): json.loads(
        str(row["outcome_eligibility_json"]))["60"] for row in rows}
    eligible = [member for member in primary if eligibility[member["episode"]]["eligible"] is True]
    ineligible = [[member["episode"], eligibility[member["episode"]].get("reason")]
                  for member in primary if eligibility[member["episode"]]["eligible"] is not True]
    grouped = {}
    for member in eligible:
        location = _SLICES.get(member["episode"])
        grouped[member["episode"]] = [] if location is None else _TABLE.slice(
            location[0], location[1] - location[0]).to_pylist()
    views = {}
    for view, field in (("absolute_close_return", "close_return"),
                        ("benchmark_relative_close_return", "benchmark_relative_close_return")):
        paths, invalid = oracle._prepare(eligible, grouped, field)
        views[view] = {"complete_members": len(paths), "invalid_members": invalid,
                       "ineligible_members": ineligible,
                       "selection": oracle._mode(paths, query_id, view)}
    return {"query_case_id": query_id, "raw_links": len(rows),
            "primary_members": len(primary), "dependence_exclusions": excluded,
            "primary_cohort_digest": cohort, "views": views}


def _worker(task: tuple[int, tuple[str, ...], str]) -> dict[str, Any]:
    partition_id, query_ids, producer_sha256 = task
    expected = [_expected(query_id) for query_id in query_ids]
    _require(_ACTUAL is not None, "actual results absent")
    _require(_PREREG_DIGEST is not None, "preregistration identity absent")
    actual = [_ACTUAL[query_id] for query_id in query_ids]
    _require(expected == actual, f"independent query reconstruction differs: {partition_id}")
    return {"partition_id": partition_id, "worker_pid": os.getpid(),
            "query_count": len(query_ids), "query_ids_digest": _stable(list(query_ids)),
            "expected_results_digest": _stable(expected),
            "producer_partition_sha256": producer_sha256,
            "preregistration_digest": _PREREG_DIGEST}


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def _recover_stale(root: Path) -> tuple[str, ...]:
    partitions = root / "partitions"
    if not partitions.exists(): return ()
    interrupted = root / "interrupted"; moved = []
    for candidate in sorted(partitions.iterdir(), key=lambda value: value.name):
        if not candidate.name.startswith(".partition-"): continue
        _require(candidate.is_dir() and not candidate.is_symlink(),
                 "unsafe verification staging path")
        interrupted.mkdir(exist_ok=True)
        destination = interrupted / f"{candidate.name[1:]}-{_stable(candidate.name)[:12]}"
        _require(not destination.exists(), "verification interrupted identity exists")
        os.rename(candidate, destination); moved.append(destination.name)
    if moved: _fsync_directory(interrupted); _fsync_directory(partitions)
    return tuple(moved)


def _publish_partition(root: Path, value: Mapping[str, Any]) -> str:
    partition_id = int(value["partition_id"]); state = dict(value)
    payload = {**state, "verification_digest": _stable(state)}
    directory = root / "partitions" / f"partition-{partition_id:04d}"
    encoded = _canonical(payload)
    if directory.exists():
        existing, raw = _json(directory / "VERIFIED.json")
        _require(raw == encoded and existing == payload, "verification partition differs")
        return "reused"
    (root / "partitions").mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".partition-{partition_id:04d}.",
                                     dir=root / "partitions"))
    try:
        path = temporary / "VERIFIED.json"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
        _fsync_directory(temporary)
        os.rename(temporary, directory)
        _fsync_directory(root / "partitions")
    except BaseException:
        try:
            for child in temporary.iterdir(): child.unlink()
            temporary.rmdir()
        except OSError: pass
        raise
    return "created"


def verify(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True); started = perf_counter()
    _require(str(_git(repository, "status", "--porcelain")) == "", "clean verifier H0 required")
    commit = str(_git(repository, "rev-parse", "HEAD")); runtime = {
        name: _sha(repository / name) for name in RUNTIME}
    for name, digest in runtime.items():
        historical = _git(repository, "show", f"{commit}:{name}", binary=True)
        _require(sha256(historical).hexdigest() == digest,
                 f"verifier runtime differs from H0: {name}")
    contract, _ = _json(repository / CONTRACT); prereg, _ = _json(repository / PREREG)
    prereg_state = {key: value for key, value in prereg.items()
                    if key != "preregistration_digest"}
    contract_state = {key: value for key, value in contract.items() if key != "contract_digest"}
    _require(contract.get("contract_digest") == _stable(contract_state) and
             prereg.get("preregistration_digest") == _stable(prereg_state) and
             prereg.get("contract_digest") == contract.get("contract_digest"),
             "contract/preregistration chain differs")
    _require(_sha(repository / SOURCE / "future-paths.parquet") ==
             prereg["future_paths_sha256"] and _sha(
                 repository / SOURCE / "query-match-links.parquet") ==
             prereg["query_match_links_sha256"], "source SHA-256 differs")
    query_ids, actual, seal, coverage, producer_partition_hashes = _load_actual(
        repository / STORE, prereg)
    started_state, _ = _json(repository / STORE / "RUN_STARTED.json")
    completed_state, _ = _json(repository / STORE / "RUN_COMPLETED.json")
    h0 = str(prereg["h0_implementation_commit"]); h1 = str(started_state.get("h1_commit", ""))
    _require(all((started_state.get("preregistration_digest") == prereg["preregistration_digest"],
                  started_state.get("h0_commit") == h0, len(h1) == 40,
                  str(_git(repository, "rev-parse", f"{h1}^")) == h0,
                  str(_git(repository, "diff-tree", "--no-commit-id", "--name-only", "-r", h1)) == str(PREREG),
                  completed_state.get("created_partition_count") == 128,
                  completed_state.get("preexisting_partition_count") == 0,
                  completed_state.get("configured_workers") == 12,
                  completed_state.get("observed_worker_pid_count") == 12,
                  float(completed_state.get("wall_seconds_this_invocation", 0)) > 0)),
             "producer H0/H1 or execution receipt differs")
    for name, digest in prereg.get("runtime_sha256", {}).items():
        historical = _git(repository, "show", f"{h0}:{name}", binary=True)
        _require(sha256(historical).hexdigest() == digest,
                 f"historical producer runtime differs: {name}")
    groups = _groups(query_ids, 128)
    global _TABLE, _SLICES, _LINKS, _ACTUAL, _PREREG_DIGEST
    _TABLE, _SLICES, _LINKS = _load_inputs(repository); _ACTUAL = actual
    _PREREG_DIGEST = str(prereg["preregistration_digest"])
    root = repository / OUTPUT; root.mkdir(parents=True, exist_ok=True)
    _recover_stale(root)
    existing = set()
    for partition_id, group in enumerate(groups):
        path = root / f"partitions/partition-{partition_id:04d}/VERIFIED.json"
        if path.exists():
            value, raw = _json(path); state = {key: item for key, item in value.items()
                                                  if key != "verification_digest"}
            _require(raw == _canonical(value) and value.get("verification_digest") ==
                     _stable(state) and value.get("query_ids_digest") == _stable(list(group)) and
                     value.get("producer_partition_sha256") == producer_partition_hashes[partition_id] and
                     value.get("preregistration_digest") == prereg["preregistration_digest"],
                     f"existing verification partition differs: {partition_id}")
            existing.add(partition_id)
    missing = [(partition_id, group, producer_partition_hashes[partition_id])
               for partition_id, group in enumerate(groups)
               if partition_id not in existing]; pids = set()
    with ProcessPoolExecutor(max_workers=WORKERS,
            mp_context=multiprocessing.get_context("fork")) as pool:
        futures = {pool.submit(_worker, task): task[0] for task in missing}
        for future in as_completed(futures):
            value = future.result(); _require(value["partition_id"] == futures[future],
                                               "verification worker identity differs")
            pids.add(value.pop("worker_pid")); value["verifier_commit"] = commit
            _publish_partition(root, value)
    receipts = []
    for partition_id, group in enumerate(groups):
        value, raw = _json(root / f"partitions/partition-{partition_id:04d}/VERIFIED.json")
        _require(value.get("query_ids_digest") == _stable(list(group)) and
                 value.get("verifier_commit") == commit and
                 value.get("producer_partition_sha256") == producer_partition_hashes[partition_id] and
                 value.get("preregistration_digest") == prereg["preregistration_digest"],
                 "verification receipt differs")
        receipts.append(sha256(raw).hexdigest())
    children = getrusage(RUSAGE_CHILDREN)
    performance = {"wall_seconds": perf_counter() - started,
        "preexisting_partition_count": len(existing), "created_partition_count": len(missing),
        "configured_workers": WORKERS, "observed_worker_pid_count": len(pids),
        "parent_peak_rss_kib": getrusage(RUSAGE_SELF).ru_maxrss,
        "maximum_child_peak_rss_kib": children.ru_maxrss,
        "children_cpu_seconds": children.ru_utime + children.ru_stime}
    forbidden = {"market_analogues.future_modes", "market_analogues.future_mode_store",
                 "experiments.m04r.m04r15_r2_full_mode_store"}
    _require(not (forbidden & set(sys.modules)), "producer module was imported")
    state = {"schema_version": SCHEMA, "status": "independently_verified", "passed": True,
        "producer_seal_digest": seal["seal_digest"],
        "producer_query_results_digest": seal["query_results_digest"],
        "producer_partition_sha256_digest": _stable(producer_partition_hashes),
        "contract_digest": contract["contract_digest"],
        "preregistration_digest": prereg["preregistration_digest"],
        "verifier_implementation_commit": commit, "verifier_runtime_sha256": runtime,
        "verified_query_count": len(query_ids), "verified_view_count": len(query_ids) * 2,
        "verification_partition_count": len(receipts),
        "verification_partition_sha256_digest": _stable(receipts),
        "coverage": coverage, "producer_modules_imported": False,
        "real_future_path_store_opened": True, "all_query_results_exact": True,
        "descriptive_evidence_authorized": True, "predictive_claim_authorized": False,
        "production_promotion_authorized": False, "trading_claim_authorized": False}
    result = {**state, "performance": performance, "result_digest": _stable(state),
              "created_at": datetime.now(timezone.utc).isoformat()}
    path = root / "VERIFIED.json"
    _require(not path.exists(), "final verification already exists")
    descriptor, name = tempfile.mkstemp(prefix=".VERIFIED.", dir=root); temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(result, indent=2, sort_keys=True,
                                    allow_nan=False).encode() + b"\n")
            handle.flush(); os.fsync(handle.fileno())
        os.rename(temporary, path)
    except BaseException:
        try: temporary.unlink()
        except OSError: pass
        raise
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv); value = verify(args.repository)
    print(json.dumps(value, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
