"""Independently reconstruct the R2-04 single-copy runner engineering gate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA = "m04r15-r2-full-runner-engineering-verification-v1"
PRODUCER_SCHEMA = "m04r15-r2-full-runner-engineering-gate-v1"
PARTITION_SCHEMA = "m04r15-r2-mode-partition-v1"
CONTRACT = Path("config/m04r15-r2-fixed-neighbor-modes-contract-v1.json")
RESULT = Path(
    "config/data/analogues/m04r15/r2-full-runner-engineering-gate-v1/RESULT.json"
)
PATH_STORE = Path(
    "config/data/analogues/m04r14/t14-09-full-outcome-store-v1/future-paths.parquet"
)
OUTPUT = Path(
    "config/data/analogues/m04r15/r2-full-runner-engineering-gate-v1-verification"
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
VERIFIER_RUNTIME = (
    "experiments/m04r/verify_m04r15_r2_full_runner_engineering_gate.py",
    "tests/test_verify_m04r15_r2_full_runner_engineering_gate.py",
)


class EngineeringVerificationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EngineeringVerificationError(message)


def _stable(value: Any) -> str:
    return sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode() + b"\n"


def _json(path: Path) -> tuple[dict[str, Any], bytes]:
    _require(path.is_file() and not path.is_symlink(), f"regular JSON required: {path}")
    raw = path.read_bytes()

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            _require(key not in result, f"duplicate JSON key: {path}:{key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw, object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                EngineeringVerificationError(f"nonfinite JSON: {path}:{item}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EngineeringVerificationError(f"invalid JSON: {path}") from error
    _require(type(value) is dict, f"JSON object required: {path}")
    return value, raw


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


def _validate_summary(result: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    deterministic = {key: value for key, value in result.items()
                     if key not in {"created_at", "performance", "result_digest"}}
    architecture = result.get("architecture")
    authentic = result.get("authentic_input")
    performance = result.get("performance")
    _require(type(architecture) is dict and type(authentic) is dict and
             type(performance) is dict, "engineering result sections differ")
    _require(all((
        result.get("result_digest") == _stable(deterministic),
        result.get("schema_version") == PRODUCER_SCHEMA,
        result.get("status") == "full_runner_engineering_gate_passed",
        result.get("passed") is True,
        result.get("contract_digest") == contract.get("contract_digest"),
        result.get("r203_verification_result_digest") ==
            "3a9cfc0ccb317f7b801f8850a78d4fbf3b342ec5e261738ae7b032c76341bde4",
        architecture.get("preload_count") == 1,
        architecture.get("multiprocessing_start_method") == "fork",
        architecture.get("configured_workers") == 12,
        architecture.get("observed_worker_pid_count") == 12,
        architecture.get("query_count") == 48,
        architecture.get("partition_count") == 12,
        architecture.get("serial_parallel_results_identical") is True,
        architecture.get("serial_parallel_partition_bytes_identical") is True,
        architecture.get("crash_left_staging_recovered") is True,
        architecture.get("completed_prefix_reused_without_rewrite") is True,
        architecture.get("conflicting_complete_partition_quarantined_and_refused") is True,
        authentic.get("future_paths_sha256") ==
            contract.get("upstream", {}).get("future_paths_sha256"),
        authentic.get("row_count") == 6_917_999,
        authentic.get("path_bearing_episode_count") == 56_325,
        authentic.get("maximum_dictionary_image_bytes") == 512 << 20,
        type(authentic.get("dictionary_image_bytes")) is int,
        authentic.get("dictionary_image_bytes", 1 << 60) <= 512 << 20,
        type(performance.get("peak_rss_kib")) is int,
        performance.get("maximum_peak_rss_kib") == 2 << 20,
        performance.get("peak_rss_kib", 1 << 60) <= 2 << 20,
        type(performance.get("wall_seconds")) in {int, float},
        math.isfinite(float(performance.get("wall_seconds", float("nan")))),
        float(performance.get("wall_seconds", 0)) > 0,
        result.get("real_future_path_store_opened") is True,
        result.get("real_future_path_modes_computed") is False,
        result.get("full_query_build_authorized") is True,
        result.get("predictive_claim_authorized") is False,
        result.get("production_promotion_authorized") is False,
        result.get("trading_claim_authorized") is False,
    )), "engineering result summary differs")


def _reconstruct_input_and_partitions(
    source: Path, contract_digest: str,
) -> tuple[int, int, str]:
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), "regular future-path store required")
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            table = pq.read_table(
                handle, columns=list(PATH_COLUMNS),
                read_dictionary=list(DICTIONARY_COLUMNS),
            ).combine_chunks()
        after = os.fstat(descriptor)
        _require((metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns) ==
                 (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
                 "future-path store changed during independent read")
    finally:
        os.close(descriptor)
    _require(tuple(table.column_names) == PATH_COLUMNS and table.num_rows == 6_917_999,
             "independent future-path table differs")
    encoded = table.column("episode_id").chunk(0)
    _require(pa.types.is_dictionary(encoded.type) and encoded.null_count == 0,
             "independent episode encoding differs")
    codes = encoded.indices.to_numpy(zero_copy_only=True)
    boundaries = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    starts = np.concatenate(([0], boundaries)).astype(np.int64)
    stops = np.concatenate((boundaries, [len(codes)])).astype(np.int64)
    run_codes = codes[starts]
    _require(len(np.unique(run_codes)) == len(run_codes) == 56_325,
             "independent episode contiguity differs")
    dictionary = encoded.dictionary
    slices = {
        str(dictionary[int(code)].as_py()): (int(start), int(stop))
        for code, start, stop in zip(run_codes, starts, stops, strict=True)
    }
    selected = tuple(sorted(slices)[:48])
    query_ids = tuple(f"engineering-query-{index:03d}" for index in range(48))
    partition_hashes = []
    for partition_id in range(12):
        query_group = query_ids[partition_id * 4:(partition_id + 1) * 4]
        episode_group = selected[partition_id * 4:(partition_id + 1) * 4]
        results = []
        for query_id, episode_id in zip(query_group, episode_group, strict=True):
            start, stop = slices[episode_id]
            rows = table.slice(start, stop - start).to_pylist()
            results.append({
                "query_case_id": query_id,
                "episode_id": episode_id,
                "episode_rows_digest": _stable(rows),
            })
        state = {
            "schema_version": PARTITION_SCHEMA,
            "partition_id": partition_id,
            "contract_digest": contract_digest,
            "query_ids": list(query_group),
            "query_ids_digest": _stable(list(query_group)),
            "results": results,
            "results_digest": _stable(results),
        }
        payload = {**state, "partition_digest": _stable(state)}
        partition_hashes.append(sha256(_canonical(payload)).hexdigest())
    return table.nbytes, len(slices), _stable(partition_hashes)


def verify(repository: Path, result_path: Path | None = None) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    result, raw = _json((result_path or repository / RESULT).resolve(strict=True))
    contract, _ = _json(repository / CONTRACT)
    contract_state = {key: value for key, value in contract.items()
                      if key != "contract_digest"}
    _require(contract.get("contract_digest") == _stable(contract_state),
             "base R2 contract self-digest differs")
    _validate_summary(result, contract)

    producer_commit = str(result.get("implementation_commit", ""))
    _require(len(producer_commit) == 40, "producer commit differs")
    _git(repository, "cat-file", "-e", f"{producer_commit}^{{commit}}")
    runtime = result.get("runtime_sha256")
    _require(type(runtime) is dict and runtime, "producer runtime inventory differs")
    for name, digest in runtime.items():
        historical = _git(repository, "show", f"{producer_commit}:{name}", binary=True)
        _require(sha256(historical).hexdigest() == digest,
                 f"historical producer runtime differs: {name}")

    source = repository / PATH_STORE
    _require(_sha(source) == contract["upstream"]["future_paths_sha256"],
             "independent future-path SHA-256 differs")
    image_bytes, episode_count, partition_digest = _reconstruct_input_and_partitions(
        source, contract["contract_digest"],
    )
    _require(image_bytes == result["authentic_input"]["dictionary_image_bytes"],
             "independent dictionary image size differs")
    _require(episode_count == result["authentic_input"]["path_bearing_episode_count"],
             "independent episode count differs")
    _require(partition_digest == result.get("partition_inventory_digest"),
             "independent partition inventory differs")

    verifier_commit = str(_git(repository, "rev-parse", "HEAD"))
    verifier_runtime = {name: _sha(repository / name) for name in VERIFIER_RUNTIME}
    for name, digest in verifier_runtime.items():
        historical = _git(repository, "show", f"{verifier_commit}:{name}", binary=True)
        _require(sha256(historical).hexdigest() == digest,
                 f"verifier runtime differs from committed H0: {name}")
    state = {
        "schema_version": SCHEMA,
        "status": "independently_verified",
        "passed": True,
        "producer_result_digest": result["result_digest"],
        "producer_result_sha256": sha256(raw).hexdigest(),
        "producer_implementation_commit": producer_commit,
        "verifier_implementation_commit": verifier_commit,
        "verifier_runtime_sha256": verifier_runtime,
        "contract_digest": contract["contract_digest"],
        "independent_dictionary_image_bytes": image_bytes,
        "independent_path_bearing_episode_count": episode_count,
        "independent_partition_inventory_digest": partition_digest,
        "producer_modules_imported": False,
        "single_copy_fork_architecture_verified": True,
        "restart_and_conflict_semantics_verified": True,
        "real_future_path_store_opened": True,
        "real_future_path_modes_computed": False,
        "full_query_build_authorized": True,
        "predictive_claim_authorized": False,
        "production_promotion_authorized": False,
        "trading_claim_authorized": False,
    }
    return {**state, "result_digest": _stable(state)}


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    _require(not path.exists() and not path.is_symlink(), "verification output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".r2-runner-verification-", dir=path.parent))
    try:
        target = temporary / "VERIFIED.json"
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write((json.dumps({
                **value, "created_at": datetime.now(timezone.utc).isoformat(),
            }, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, path)
    except BaseException:
        try:
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
        except OSError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    value = verify(args.repository, args.result)
    if not args.dry_run:
        _publish(args.repository.resolve(strict=True) / OUTPUT, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
