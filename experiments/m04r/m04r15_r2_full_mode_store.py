"""Preregister, build and seal all R2 fixed-neighbor future-path mode results."""
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
import tempfile
from time import perf_counter
from typing import Any, Mapping, Sequence

import pandas as pd

from market_analogues.future_modes import (
    calendar_quarter, pairwise_l1, prepare_paths, select_modes,
    select_primary_members,
)
from market_analogues.future_mode_store import (
    ArrowPathIndex, canonical_json, partition_query_ids, publish_partition,
    recover_stale_partition_temporaries, stable_digest, validate_partition,
)


SCHEMA = "m04r15-r2-full-mode-store-v1"
PREREG_SCHEMA = "m04r15-r2-full-mode-store-preregistration-v1"
CONTRACT = Path("config/m04r15-r2-fixed-neighbor-modes-contract-v1.json")
ENGINEERING = Path(
    "config/data/analogues/m04r15/r2-full-runner-engineering-gate-v1-verification/VERIFIED.json"
)
STORE = Path("config/data/analogues/m04r14/t14-09-full-outcome-store-v1")
PREREG = Path("experiments/m04r/m04r15_r2_full_mode_store_preregistered.json")
OUTPUT = Path("config/data/analogues/m04r15/r2-full-mode-store-v1")
RUNTIME = (
    "config/m04r15-r2-fixed-neighbor-modes-contract-v1.json",
    "config/data/analogues/m04r15/r2-full-runner-engineering-gate-v1-verification/VERIFIED.json",
    "src/market_analogues/future_modes.py",
    "src/market_analogues/future_mode_store.py",
    "experiments/m04r/m04r15_r2_full_mode_store.py",
    "tests/test_m04r15_r2_full_mode_store.py",
)
LINK_COLUMNS = (
    "query_case_id", "query_symbol", "match_rank", "matched_episode_id",
    "matched_symbol", "matched_cutoff", "source_fingerprint",
    "outcome_eligibility_json",
)
WORKERS = 12
PARTITIONS = 128

_PATH_INDEX: ArrowPathIndex | None = None
_LINKS_BY_QUERY: dict[str, tuple[dict[str, Any], ...]] | None = None
_CONTRACT_DIGEST: str | None = None


class FullModeStoreError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FullModeStoreError(message)


def _json(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"regular JSON required: {path}")
    try:
        value = json.loads(path.read_text())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FullModeStoreError(f"invalid JSON: {path}") from error
    _require(type(value) is dict, f"JSON object required: {path}")
    return value


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _git(repository: Path, *args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ("git", *args), cwd=repository, capture_output=True,
        text=not binary, check=False,
    )
    _require(result.returncode == 0, f"git {' '.join(args)} failed")
    return result.stdout if binary else result.stdout.strip()


def _selection_summary(selection: Any, paths: Sequence[Any]) -> dict[str, Any]:
    candidates = []
    for candidate in selection.candidates:
        stability = candidate.stability
        candidates.append({
            "k": candidate.k,
            "medoid_episode_ids": [paths[index].member.episode_id
                                   for index in candidate.medoid_indices],
            "cluster_sizes": list(candidate.cluster_sizes),
            "mean_silhouette": candidate.mean_silhouette,
            "accepted": candidate.accepted,
            "rejection_reasons": list(candidate.rejection_reasons),
            "stability": None if stability is None else {
                "valid_replicates": stability.valid_replicates,
                "median_adjusted_rand_index": stability.median_adjusted_rand_index,
                "adjusted_rand_indices_digest": stable_digest(
                    list(stability.adjusted_rand_indices)),
                "minimum_adjusted_rand_index": min(stability.adjusted_rand_indices)
                    if stability.adjusted_rand_indices else None,
                "maximum_adjusted_rand_index": max(stability.adjusted_rand_indices)
                    if stability.adjusted_rand_indices else None,
            },
        })
    return {
        "status": selection.status,
        "selected_k": selection.selected_k,
        "medoid_episode_ids": [paths[index].member.episode_id
                               for index in selection.medoid_indices],
        "member_to_mode": {
            path.member.episode_id: int(selection.labels[index])
            for index, path in enumerate(paths)
        },
        "candidates": candidates,
    }


def _cohort_digest(members: Sequence[Any]) -> str:
    return stable_digest([[
        member.match_rank, member.episode_id, member.symbol, member.cutoff,
        member.source_fingerprint,
    ] for member in members])


def _query_result(query_id: str) -> dict[str, Any]:
    _require(_PATH_INDEX is not None and _LINKS_BY_QUERY is not None and
             _CONTRACT_DIGEST is not None, "forked worker state is not installed")
    raw = list(_LINKS_BY_QUERY[query_id])
    query_symbols = {str(row["query_symbol"]) for row in raw}
    _require(len(raw) == 20 and len(query_symbols) == 1,
             f"raw link inventory differs: {query_id}")
    primary, dependence = select_primary_members(next(iter(query_symbols)), raw)
    cohort_digest = _cohort_digest(primary)
    eligibility: dict[str, dict[str, Any]] = {}
    for row in raw:
        try:
            horizon = json.loads(str(row["outcome_eligibility_json"]))["60"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise FullModeStoreError(f"eligibility differs: {query_id}") from error
        _require(type(horizon) is dict and type(horizon.get("eligible")) is bool,
                 f"eligibility state differs: {query_id}")
        eligibility[str(row["matched_episode_id"])] = horizon
    eligible = [member for member in primary
                if eligibility[member.episode_id]["eligible"] is True]
    ineligible = [[member.episode_id, eligibility[member.episode_id].get("reason")]
                  for member in primary
                  if eligibility[member.episode_id]["eligible"] is not True]
    path_rows = {member.episode_id: _PATH_INDEX.rows(member.episode_id)
                 for member in eligible}
    views: dict[str, Any] = {}
    for view_id, field in (
        ("absolute_close_return", "close_return"),
        ("benchmark_relative_close_return", "benchmark_relative_close_return"),
    ):
        paths, invalid = prepare_paths(
            eligible, path_rows, value_field=field, horizon=60,
        )
        if paths:
            blocks = [calendar_quarter(path.member.cutoff) for path in paths]
            selection = select_modes(
                pairwise_l1(paths), [path.member.key for path in paths], blocks,
                contract_digest=_CONTRACT_DIGEST, query_case_id=query_id,
                view_id=view_id, replicates=256,
            )
            summary = _selection_summary(selection, paths)
        else:
            summary = {
                "status": "abstain_insufficient_complete_primary_members",
                "selected_k": 0, "medoid_episode_ids": [],
                "member_to_mode": {}, "candidates": [],
            }
        views[view_id] = {
            "complete_members": len(paths),
            "invalid_members": [[item.member.episode_id, item.reason]
                                for item in invalid],
            "ineligible_members": ineligible,
            "selection": summary,
        }
    return {
        "query_case_id": query_id,
        "raw_links": len(raw),
        "primary_members": len(primary),
        "dependence_exclusions": [[item.member.episode_id, item.reason]
                                  for item in dependence],
        "primary_cohort_digest": cohort_digest,
        "views": views,
    }


def _worker(task: tuple[int, tuple[str, ...]]) -> dict[str, Any]:
    partition_id, query_ids = task
    return {
        "partition_id": partition_id,
        "worker_pid": os.getpid(),
        "results": [_query_result(query_id) for query_id in query_ids],
    }


def _read_links(path: Path) -> tuple[tuple[str, ...], dict[str, tuple[dict[str, Any], ...]]]:
    frame = pd.read_parquet(path, columns=list(LINK_COLUMNS))
    _require(len(frame) == 65_400, "full link row count differs")
    frame = frame.sort_values(["query_case_id", "match_rank"], kind="stable")
    grouped: dict[str, tuple[dict[str, Any], ...]] = {}
    for query_id, group in frame.groupby("query_case_id", sort=True):
        rows = tuple(group.to_dict("records"))
        _require([int(row["match_rank"]) for row in rows] == list(range(1, 21)),
                 f"link ranks differ: {query_id}")
        query_symbols = {str(row["query_symbol"]) for row in rows}
        _require(len(query_symbols) == 1, f"query symbol differs: {query_id}")
        try:
            select_primary_members(next(iter(query_symbols)), rows)
            horizons = [json.loads(str(row["outcome_eligibility_json"]))["60"]
                        for row in rows]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise FullModeStoreError(f"link member/eligibility differs: {query_id}") from error
        _require(all(type(value) is dict and type(value.get("eligible")) is bool
                     for value in horizons),
                 f"link eligibility state differs: {query_id}")
        grouped[str(query_id)] = rows
    query_ids = tuple(grouped)
    _require(len(query_ids) == 3_270 and tuple(sorted(set(query_ids))) == query_ids,
             "full query inventory differs")
    return query_ids, grouped


def _runtime(repository: Path) -> dict[str, str]:
    return {name: _sha(repository / name) for name in RUNTIME}


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    _require(not path.exists() and not path.is_symlink(), f"output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _publish_equal_or_create(path: Path, value: Mapping[str, Any]) -> str:
    encoded = canonical_json(value)
    if path.exists() or path.is_symlink():
        _require(path.is_file() and not path.is_symlink() and path.read_bytes() == encoded,
                 f"existing aggregate artifact differs: {path}")
        return "reused"
    _write_exclusive(path, value)
    return "created"


def preregister(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    _require(str(_git(repository, "status", "--porcelain")) == "",
             "clean H0 worktree required")
    _require(not (repository / PREREG).exists(), "preregistration exists")
    contract = _json(repository / CONTRACT)
    engineering = _json(repository / ENGINEERING)
    _require(all((
        engineering.get("passed") is True,
        engineering.get("contract_digest") == contract.get("contract_digest"),
        engineering.get("full_query_build_authorized") is True,
        engineering.get("predictive_claim_authorized") is False,
    )), "engineering authorization differs")
    links = repository / STORE / "query-match-links.parquet"
    paths = repository / STORE / "future-paths.parquet"
    _require(_sha(links) == contract["upstream"]["query_match_links_sha256"],
             "query-link SHA-256 differs")
    _require(_sha(paths) == contract["upstream"]["future_paths_sha256"],
             "future-path SHA-256 differs")
    query_ids = tuple(sorted(pd.read_parquet(
        links, columns=["query_case_id"],
    )["query_case_id"].astype(str).unique()))
    groups = partition_query_ids(query_ids, PARTITIONS)
    h0 = str(_git(repository, "rev-parse", "HEAD"))
    state = {
        "schema_version": PREREG_SCHEMA,
        "status": "frozen_before_full_mode_computation",
        "h0_implementation_commit": h0,
        "runtime_sha256": _runtime(repository),
        "contract_digest": contract["contract_digest"],
        "engineering_verification_result_digest": engineering["result_digest"],
        "query_match_links_sha256": contract["upstream"]["query_match_links_sha256"],
        "future_paths_sha256": contract["upstream"]["future_paths_sha256"],
        "query_count": len(query_ids),
        "query_inventory_digest": stable_digest(list(query_ids)),
        "partition_count": len(groups),
        "partition_query_counts": [len(group) for group in groups],
        "partition_query_digests": [stable_digest(list(group)) for group in groups],
        "workers": WORKERS,
        "multiprocessing_start_method": "fork",
        "output_root_absent": not (repository / OUTPUT).exists()
            and not (repository / OUTPUT).is_symlink(),
        "real_future_path_modes_computed": False,
        "predictive_claim_authorized": False,
        "production_promotion_authorized": False,
        "trading_claim_authorized": False,
    }
    _require(state["query_count"] == 3_270 and state["output_root_absent"] is True,
             "preregistration population or output state differs")
    value = {**state, "preregistration_digest": stable_digest(state)}
    _write_exclusive(repository / PREREG, value)
    return value


def _validate_preregistration(repository: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    prereg = _json(repository / PREREG)
    state = {key: value for key, value in prereg.items()
             if key != "preregistration_digest"}
    _require(prereg.get("preregistration_digest") == stable_digest(state),
             "preregistration digest differs")
    _require(prereg.get("schema_version") == PREREG_SCHEMA and
             prereg.get("status") == "frozen_before_full_mode_computation",
             "preregistration state differs")
    h1 = str(_git(repository, "rev-parse", "HEAD"))
    parent = str(_git(repository, "rev-parse", "HEAD^"))
    changed = str(_git(
        repository, "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD",
    )).splitlines()
    _require(parent == prereg.get("h0_implementation_commit") and changed == [str(PREREG)],
             "sole-child H1 boundary differs")
    for name, digest in prereg.get("runtime_sha256", {}).items():
        historical = _git(repository, "show", f"{parent}:{name}", binary=True)
        _require(sha256(historical).hexdigest() == digest,
                 f"H0 runtime differs: {name}")
    _require(str(_git(repository, "status", "--porcelain")) == "",
             "clean H1 worktree required")
    return prereg, {"h1_commit": h1, "h0_commit": parent}


def _coverage(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = Counter(
        view["selection"]["status"]
        for result in results for view in result["views"].values()
    )
    selected_k = Counter(
        str(view["selection"]["selected_k"])
        for result in results for view in result["views"].values()
    )
    return {
        "schema_version": f"{SCHEMA}-coverage",
        "query_count": len(results),
        "view_count": sum(len(result["views"]) for result in results),
        "raw_link_count": sum(int(result["raw_links"]) for result in results),
        "primary_member_count": sum(int(result["primary_members"]) for result in results),
        "dependence_exclusion_count": sum(
            len(result["dependence_exclusions"]) for result in results),
        "ineligible_member_occurrences": sum(
            len(view["ineligible_members"])
            for result in results for view in result["views"].values()) // 2,
        "invalid_member_occurrences": sum(
            len(view["invalid_members"])
            for result in results for view in result["views"].values()),
        "mode_status_counts": dict(sorted(statuses.items())),
        "selected_k_counts": dict(sorted(selected_k.items())),
    }


def _seal_store(root: Path, prereg: Mapping[str, Any], groups: Sequence[Sequence[str]],
                *, performance: Mapping[str, Any]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    partition_hashes = []
    for partition_id, group in enumerate(groups):
        directory = root / f"partitions/partition-{partition_id:04d}"
        payload = validate_partition(
            directory, partition_id, group,
            contract_digest=str(prereg["contract_digest"]),
        )
        results.extend(payload["results"])
        partition_hashes.append(_sha(directory / "PARTITION.json"))
    expected_ids = [query_id for group in groups for query_id in group]
    _require([result["query_case_id"] for result in results] == expected_ids,
             "aggregate result inventory differs")
    results_path = root / "RESULTS.jsonl"
    result_bytes = b"".join(canonical_json(result) for result in results)
    if results_path.exists():
        _require(results_path.is_file() and not results_path.is_symlink() and
                 results_path.read_bytes() == result_bytes,
                 "existing aggregate results differ")
    else:
        descriptor, name = tempfile.mkstemp(prefix=".RESULTS.", dir=root)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(result_bytes); handle.flush(); os.fsync(handle.fileno())
            os.rename(temporary, results_path)
        except BaseException:
            try: temporary.unlink()
            except OSError: pass
            raise
    coverage = _coverage(results)
    _publish_equal_or_create(root / "COVERAGE.json", coverage)
    completed = {
        "schema_version": f"{SCHEMA}-run-completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        **performance,
    }
    if not (root / "RUN_COMPLETED.json").exists():
        _write_exclusive(root / "RUN_COMPLETED.json", completed)
    state = {
        "schema_version": SCHEMA,
        "status": "sealed",
        "preregistration_digest": prereg["preregistration_digest"],
        "contract_digest": prereg["contract_digest"],
        "query_count": len(results),
        "view_count": len(results) * 2,
        "partition_count": len(groups),
        "partition_sha256_digest": stable_digest(partition_hashes),
        "query_results_digest": stable_digest(results),
        "results_jsonl_sha256": _sha(results_path),
        "results_jsonl_bytes": results_path.stat().st_size,
        "coverage_digest": stable_digest(coverage),
        "coverage_sha256": _sha(root / "COVERAGE.json"),
        "real_future_path_modes_computed": True,
        "descriptive_evidence_authorized": True,
        "independent_verification_complete": False,
        "predictive_claim_authorized": False,
        "production_promotion_authorized": False,
        "trading_claim_authorized": False,
    }
    seal = {**state, "seal_digest": stable_digest(state)}
    _publish_equal_or_create(root / "SEALED.json", seal)
    return seal


def run(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    prereg, commits = _validate_preregistration(repository)
    contract = _json(repository / CONTRACT)
    engineering = _json(repository / ENGINEERING)
    _require(all((
        contract.get("contract_digest") == prereg.get("contract_digest"),
        engineering.get("result_digest") ==
            prereg.get("engineering_verification_result_digest"),
        engineering.get("full_query_build_authorized") is True,
    )), "run authorization chain differs")
    links_path = repository / STORE / "query-match-links.parquet"
    paths_path = repository / STORE / "future-paths.parquet"
    _require(_sha(links_path) == prereg["query_match_links_sha256"] and
             _sha(paths_path) == prereg["future_paths_sha256"],
             "run input SHA-256 differs")
    query_ids, grouped = _read_links(links_path)
    groups = partition_query_ids(query_ids, int(prereg["partition_count"]))
    _require(stable_digest(list(query_ids)) == prereg["query_inventory_digest"] and
             [stable_digest(list(group)) for group in groups] ==
                prereg["partition_query_digests"],
             "run query partition inventory differs")

    root = repository / OUTPUT
    root.mkdir(parents=True, exist_ok=True)
    started_state = {
        "schema_version": f"{SCHEMA}-run-started",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preregistration_digest": prereg["preregistration_digest"],
        **commits,
    }
    started_path = root / "RUN_STARTED.json"
    if started_path.exists():
        existing = _json(started_path)
        _require(all(existing.get(key) == value for key, value in started_state.items()
                     if key != "created_at"), "existing run identity differs")
    else:
        _write_exclusive(started_path, started_state)
    recover_stale_partition_temporaries(root)

    existing_ids: set[int] = set()
    for partition_id, group in enumerate(groups):
        directory = root / f"partitions/partition-{partition_id:04d}"
        if directory.exists() or directory.is_symlink():
            validate_partition(
                directory, partition_id, group,
                contract_digest=str(prereg["contract_digest"]),
            )
            existing_ids.add(partition_id)
    missing = [(partition_id, group) for partition_id, group in enumerate(groups)
               if partition_id not in existing_ids]
    started = perf_counter()
    worker_pids: set[int] = set()
    global _PATH_INDEX, _LINKS_BY_QUERY, _CONTRACT_DIGEST
    if missing:
        _PATH_INDEX = ArrowPathIndex.load(paths_path)
        _require(_PATH_INDEX.row_count == 6_917_999 and
                 _PATH_INDEX.episode_count == 56_325,
                 "full path index inventory differs")
        _LINKS_BY_QUERY = grouped
        _CONTRACT_DIGEST = str(prereg["contract_digest"])
        with ProcessPoolExecutor(
            max_workers=int(prereg["workers"]),
            mp_context=multiprocessing.get_context("fork"),
        ) as pool:
            futures = {pool.submit(_worker, task): task[0] for task in missing}
            for future in as_completed(futures):
                value = future.result()
                partition_id = int(value["partition_id"])
                _require(partition_id == futures[future], "worker partition identity differs")
                worker_pids.add(int(value["worker_pid"]))
                status = publish_partition(
                    root, partition_id, groups[partition_id], value["results"],
                    contract_digest=str(prereg["contract_digest"]),
                )
                _require(status == "created", "missing partition was not created")
    elapsed = perf_counter() - started
    children = getrusage(RUSAGE_CHILDREN)
    performance = {
        "wall_seconds_this_invocation": elapsed,
        "preexisting_partition_count": len(existing_ids),
        "created_partition_count": len(missing),
        "configured_workers": int(prereg["workers"]),
        "observed_worker_pid_count": len(worker_pids),
        "parent_peak_rss_kib": getrusage(RUSAGE_SELF).ru_maxrss,
        "maximum_child_peak_rss_kib": children.ru_maxrss,
        "children_cpu_seconds": children.ru_utime + children.ru_stime,
    }
    return _seal_store(root, prereg, groups, performance=performance)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preregister", "run"))
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv)
    value = preregister(args.repository) if args.command == "preregister" else run(args.repository)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
