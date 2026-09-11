"""Verify the single-copy, forked, restart-safe architecture required by R2-04."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path
from resource import getrusage, RUSAGE_SELF
import subprocess
import tempfile
from time import perf_counter
from typing import Any, Mapping, Sequence

from market_analogues.future_mode_store import (
    ArrowPathIndex,
    canonical_json,
    install_shared_path_index,
    partition_query_ids,
    publish_partition,
    recover_stale_partition_temporaries,
    shared_episode_digest,
)


SCHEMA = "m04r15-r2-full-runner-engineering-gate-v1"
CONTRACT = Path("config/m04r15-r2-fixed-neighbor-modes-contract-v1.json")
POC_VERIFICATION = Path(
    "config/data/analogues/m04r15/r2-bounded-real-poc-v1-verification/VERIFIED.json"
)
POC_RESULT = Path(
    "config/data/analogues/m04r15/r2-bounded-real-poc-v1/RESULT.json"
)
PATH_STORE = Path(
    "config/data/analogues/m04r14/t14-09-full-outcome-store-v1/future-paths.parquet"
)
OUTPUT = Path(
    "config/data/analogues/m04r15/r2-full-runner-engineering-gate-v1/RESULT.json"
)
RUNTIME = (
    "config/m04r15-r2-fixed-neighbor-modes-contract-v1.json",
    "config/data/analogues/m04r15/r2-bounded-real-poc-v1/RESULT.json",
    "config/data/analogues/m04r15/r2-bounded-real-poc-v1-verification/VERIFIED.json",
    "src/market_analogues/future_mode_store.py",
    "experiments/m04r/m04r15_r2_full_runner_engineering_gate.py",
    "tests/test_future_mode_store.py",
)
WORKERS = 12
PARTITIONS = 12
QUERY_COUNT = 48
MAX_IMAGE_BYTES = 512 << 20
MAX_PEAK_RSS_KIB = 2 << 20
_FORK_BARRIER: Any = None


class EngineeringGateError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EngineeringGateError(message)


def _stable(value: Any) -> str:
    return sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"regular JSON required: {path}")
    value = json.loads(path.read_text())
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


def _partition_result(task: tuple[int, tuple[str, ...], tuple[str, ...]]) -> dict[str, Any]:
    if _FORK_BARRIER is not None:
        _FORK_BARRIER.wait(timeout=30)
    partition_id, query_ids, episode_ids = task
    rows = [{
        "query_case_id": query_id,
        "episode_id": episode_id,
        "episode_rows_digest": shared_episode_digest(episode_id),
    } for query_id, episode_id in zip(query_ids, episode_ids, strict=True)]
    return {
        "partition_id": partition_id,
        "worker_pid": os.getpid(),
        "results": rows,
    }


def _partition_bytes(root: Path, count: int) -> tuple[bytes, ...]:
    return tuple(
        (root / f"partitions/partition-{index:04d}/PARTITION.json").read_bytes()
        for index in range(count)
    )


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    contract = _json(repository / CONTRACT)
    poc = _json(repository / POC_RESULT)
    prior = _json(repository / POC_VERIFICATION)
    _require(all((
        prior.get("passed") is True,
        prior.get("full_query_build_authorized") is True,
        prior.get("producer_result_digest") == poc.get("result_digest"),
        poc.get("contract_digest") == contract.get("contract_digest"),
        contract.get("contract_digest") ==
            "6bec009810afce7c58508fca28179579f5904382376dc9c5bce74aa74f41e08c",
    )), "R2-03 verification chain differs")

    commit = str(_git(repository, "rev-parse", "HEAD"))
    runtime = {name: _sha(repository / name) for name in RUNTIME}
    for name, digest in runtime.items():
        historical = _git(repository, "show", f"{commit}:{name}", binary=True)
        _require(sha256(historical).hexdigest() == digest,
                 f"runtime differs from committed H0: {name}")

    source = repository / PATH_STORE
    _require(_sha(source) == contract["upstream"]["future_paths_sha256"],
             "future-path store SHA-256 differs")
    started = perf_counter()
    index = ArrowPathIndex.load(source)
    _require(index.row_count == contract["population"]["future_path_rows"],
             "future-path row count differs")
    _require(index.episode_count == 56_325, "path-bearing episode count differs")
    _require(index.image_bytes <= MAX_IMAGE_BYTES, "dictionary image exceeds frozen bound")
    install_shared_path_index(index)

    query_ids = tuple(f"engineering-query-{index:03d}" for index in range(QUERY_COUNT))
    groups = partition_query_ids(query_ids, PARTITIONS)
    selected_episodes = index.episode_ids[:QUERY_COUNT]
    episode_groups = tuple(
        selected_episodes[offset:offset + len(group)]
        for offset, group in zip(
            (sum(map(len, groups[:index])) for index in range(len(groups))), groups,
            strict=True,
        )
    )
    tasks = tuple(
        (partition_id, group, episode_group)
        for partition_id, (group, episode_group) in enumerate(zip(
            groups, episode_groups, strict=True,
        ))
    )
    serial = tuple(_partition_result(task) for task in tasks)
    global _FORK_BARRIER
    _FORK_BARRIER = multiprocessing.get_context("fork").Barrier(WORKERS)
    with ProcessPoolExecutor(
        max_workers=WORKERS, mp_context=multiprocessing.get_context("fork"),
    ) as pool:
        parallel = tuple(pool.map(_partition_result, tasks, chunksize=1))
    _FORK_BARRIER = None
    serial_results = tuple(value["results"] for value in serial)
    parallel_results = tuple(value["results"] for value in parallel)
    _require(serial_results == parallel_results, "serial/fork result identity differs")

    with tempfile.TemporaryDirectory(prefix="r2-runner-gate-") as temporary_name:
        temporary = Path(temporary_name)
        serial_root = temporary / "serial"
        resumed_root = temporary / "resumed"
        conflict_root = temporary / "conflict"
        for partition_id, (group, rows) in enumerate(zip(
            groups, serial_results, strict=True,
        )):
            _require(publish_partition(
                serial_root, partition_id, group, rows,
                contract_digest=contract["contract_digest"],
            ) == "created", "serial partition was not created")

        orphan = resumed_root / "partitions/.partition-0004.crashed"
        orphan.mkdir(parents=True)
        (orphan / "PARTITION.json").write_bytes(b'{"partial":true}')
        recovered = recover_stale_partition_temporaries(resumed_root)
        _require(len(recovered) == 1 and not orphan.exists(),
                 "crash-left staging recovery differs")
        for partition_id in range(4):
            publish_partition(
                resumed_root, partition_id, groups[partition_id],
                parallel_results[partition_id],
                contract_digest=contract["contract_digest"],
            )
        prefix_before = _partition_bytes(resumed_root, 4)
        states = tuple(publish_partition(
            resumed_root, partition_id, group, rows,
            contract_digest=contract["contract_digest"],
        ) for partition_id, (group, rows) in enumerate(zip(
            groups, parallel_results, strict=True,
        )))
        _require(states == ("reused",) * 4 + ("created",) * 8,
                 "partial resume state differs")
        _require(_partition_bytes(resumed_root, 4) == prefix_before,
                 "completed prefix was rewritten")
        serial_bytes = _partition_bytes(serial_root, PARTITIONS)
        resumed_bytes = _partition_bytes(resumed_root, PARTITIONS)
        _require(serial_bytes == resumed_bytes,
                 "sequential/resumed fork partition bytes differ")

        publish_partition(
            conflict_root, 0, groups[0], parallel_results[0],
            contract_digest=contract["contract_digest"],
        )
        conflict_path = conflict_root / "partitions/partition-0000/PARTITION.json"
        damaged = json.loads(conflict_path.read_text())
        damaged["results"][0]["episode_id"] = "tampered"
        conflict_path.write_bytes(canonical_json(damaged))
        conflict_refused = False
        try:
            publish_partition(
                conflict_root, 0, groups[0], parallel_results[0],
                contract_digest=contract["contract_digest"],
            )
        except Exception as error:
            conflict_refused = "quarantined" in str(error)
        _require(conflict_refused, "conflicting completed partition was not refused")
        _require(not (conflict_root / "partitions/partition-0000").exists(),
                 "conflicting completed partition remains live")
        _require(len(tuple((conflict_root / "conflicts").iterdir())) == 1,
                 "conflict quarantine inventory differs")
        partition_digest = _stable([sha256(value).hexdigest() for value in serial_bytes])

    elapsed = perf_counter() - started
    peak_rss = getrusage(RUSAGE_SELF).ru_maxrss
    _require(peak_rss <= MAX_PEAK_RSS_KIB, "engineering gate exceeds RSS bound")
    worker_pids = sorted({int(value["worker_pid"]) for value in parallel})
    _require(len(worker_pids) == WORKERS, "not every configured fork processed work")
    state = {
        "schema_version": SCHEMA,
        "status": "full_runner_engineering_gate_passed",
        "passed": True,
        "contract_digest": contract["contract_digest"],
        "r203_verification_result_digest": prior["result_digest"],
        "implementation_commit": commit,
        "runtime_sha256": runtime,
        "architecture": {
            "preload_count": 1,
            "multiprocessing_start_method": "fork",
            "configured_workers": WORKERS,
            "observed_worker_pid_count": len(worker_pids),
            "query_count": QUERY_COUNT,
            "partition_count": PARTITIONS,
            "serial_parallel_results_identical": True,
            "serial_parallel_partition_bytes_identical": True,
            "crash_left_staging_recovered": True,
            "completed_prefix_reused_without_rewrite": True,
            "conflicting_complete_partition_quarantined_and_refused": True,
        },
        "authentic_input": {
            "future_paths_sha256": contract["upstream"]["future_paths_sha256"],
            "row_count": index.row_count,
            "path_bearing_episode_count": index.episode_count,
            "dictionary_image_bytes": index.image_bytes,
            "maximum_dictionary_image_bytes": MAX_IMAGE_BYTES,
        },
        "partition_inventory_digest": partition_digest,
        "performance": {
            "wall_seconds": elapsed,
            "peak_rss_kib": peak_rss,
            "maximum_peak_rss_kib": MAX_PEAK_RSS_KIB,
        },
        "real_future_path_store_opened": True,
        "real_future_path_modes_computed": False,
        "full_query_build_authorized": True,
        "predictive_claim_authorized": False,
        "production_promotion_authorized": False,
        "trading_claim_authorized": False,
    }
    deterministic = {key: value for key, value in state.items() if key != "performance"}
    return {**state, "result_digest": _stable(deterministic)}


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    _require(not path.exists() and not path.is_symlink(), "engineering gate result exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".r2-runner-gate-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write((json.dumps({
                **value, "created_at": datetime.now(timezone.utc).isoformat(),
            }, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    value = execute(args.repository)
    if not args.dry_run:
        _publish(args.repository.resolve(strict=True) / OUTPUT, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
