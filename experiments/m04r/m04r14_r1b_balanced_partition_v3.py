"""Preregister and publish the outcome/retrieval-blind R1-B v3 query partition."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info

from market_analogues.adapters import source_from_spec
from market_analogues.balanced_partition import (
    BalancedPartitionError, array_digest, balanced_recursive_partition,
    id_digest, integer_array_digest, midrank_transform,
)
from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.config import load_config
from market_analogues.representation import represent, representation_input_digest
from market_analogues.types import Episode, EpisodeKey, InstrumentKey, stable_hash


SCHEMA = "m04r14-r1b-balanced-partition-v3"
PREREGISTRATION = Path("experiments/m04r/m04r14_r1b_balanced_v3_preregistered.json")
OUTPUT = Path("config/data/analogues/m04r14/r1b-balanced-partition-v3")
SUPPORT_OUTPUT = Path("config/data/analogues/m04r14/r1b-balanced-support-v3")
REGISTRY = Path("config/data/analogues/m04r14/nasdaq-shadow-denominator-v1/query-registry.json")
CONFIG = Path("config/datasets.example.yaml")
BENCHMARK = Path("/home/vinay/code/loser-nasdaq/data/nasdaq/index/IXIC.parquet")
V2_RESULT = Path("config/data/analogues/m04r14/r1b-support-pilot-v2/RESULT.json")
V2_QUERY_CELLS = Path("config/data/analogues/m04r14/r1b-support-pilot-v2/QUERY_CELLS.json")
V2_COHORT = Path("config/data/analogues/m04r14/r1b-support-pilot-v2/COHORT_SUPPORT.json")
V2_VERIFIED = Path(
    "config/data/analogues/m04r14/r1b-support-pilot-v2-integrity-verification-v1/VERIFIED.json"
)
R1A_RESULT = Path("config/data/analogues/m04r14/r1a-exposure-audit-v2/RESULT.json")
CASES = Path("config/data/analogues/m04r14/nasdaq-shadow-snapshot-v1/cases")
PARTITION_RUNTIME = (
    "experiments/m04r/m04r14_r1b_balanced_partition_v3.py",
    "src/market_analogues/balanced_partition.py",
    "tests/test_balanced_partition.py",
)
SUPPORT_RUNTIME = (
    "experiments/m04r/m04r14_r1b_balanced_support_v3.py",
    "src/market_analogues/adequacy_support.py",
    "tests/test_r1b_balanced_support_v3.py",
)
K_VALUES = (8, 12, 16)
PRIMARY_K = 8
V2_VERIFICATION_DIGEST = "1d14eabe98c2630dd94d5a8fbead62791368477573bcfcc85a80d758c6f0ca03"


class BalancedPartitionRunError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BalancedPartitionRunError(message)


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, expected: type = dict) -> Any:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise BalancedPartitionRunError(f"duplicate JSON key {key}: {path}")
            result[key] = value
        return result
    value = json.loads(
        path.read_bytes(), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            BalancedPartitionRunError(f"non-finite JSON {token}: {path}")
        ),
    )
    _require(isinstance(value, expected), f"JSON {expected.__name__} required: {path}")
    return value


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(("git", *args), cwd=repository, text=True, capture_output=True)
    _require(result.returncode == 0, f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _digest(value: Mapping[str, Any], omitted: set[str]) -> str:
    return stable_hash({key: item for key, item in value.items() if key not in omitted})


def _runtime_environment() -> dict[str, Any]:
    blas = [{
        key: value for key, value in row.items()
        if key in {"user_api", "internal_api", "prefix", "version", "threading_layer", "architecture"}
    } for row in threadpool_info() if row.get("user_api") == "blas"]
    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "platform": platform.platform(), "blas": sorted(blas, key=lambda row: json.dumps(row, sort_keys=True)),
        "partition_blas_threads": 1,
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid4().hex}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise BalancedPartitionRunError(f"create-only path exists: {path}") from error
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _write(path: Path, value: Any) -> None:
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())


def _publish(temporary: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination.parent / f".{destination.name}.publish.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        _require(not destination.exists() and not destination.is_symlink(), f"output exists: {destination}")
        current = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(current)
        finally: os.close(current)
        os.rename(temporary, destination)
        parent = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(parent)
        finally: os.close(parent)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN); os.close(descriptor)


def _clean(repository: Path) -> str:
    _require(not _git(repository, "status", "--porcelain", "--untracked-files=all"), "clean tree required")
    return _git(repository, "rev-parse", "HEAD")


def _feature(
    query_id: str, row: Mapping[str, Any], stock: pd.DataFrame, benchmark: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    cutoff = pd.Timestamp(str(row["cutoff"]))
    stock_prefix = asdict(causal_prefix_digest(stock, cutoff))
    benchmark_prefix = asdict(causal_prefix_digest(benchmark, cutoff))
    _require(stock_prefix == row["stock_prefix"], f"stock prefix differs: {query_id}")
    _require(benchmark_prefix == row["benchmark_prefix"], f"benchmark prefix differs: {query_id}")
    eligible = stock.loc[pd.to_datetime(stock["timestamp"]) <= cutoff]
    lookback = int(row["lookback"])
    _require(lookback == 252 and len(eligible) >= 252, f"query history differs: {query_id}")
    window = eligible.tail(lookback).copy().reset_index(drop=True)
    actual_cutoff = pd.Timestamp(window["timestamp"].iloc[-1])
    context = benchmark.loc[pd.to_datetime(benchmark["timestamp"]) <= actual_cutoff].copy()
    episode = Episode(
        EpisodeKey(InstrumentKey("nasdaq", str(row["symbol"])), actual_cutoff, lookback,
                   str(row["representation_version"])),
        window, context, str(row["quality_tier"]),
    )
    _require(episode.key.id == query_id, f"query identity differs: {query_id}")
    represented = represent(episode)
    stage = represented.stage.astype(np.float64).reshape(12, 4, order="C")[:, :3].ravel(order="C")
    vector = np.r_[represented.coarse[:96].astype(np.float64), stage,
                   represented.structural[:9].astype(np.float64)]
    _require(vector.shape == (141,) and np.isfinite(vector).all(), f"query vector differs: {query_id}")
    return vector, {
        "query_episode_id": query_id,
        "query_representation_digest": representation_input_digest(represented),
        "stock_prefix_digest": stock_prefix["digest"],
        "benchmark_prefix_digest": benchmark_prefix["digest"],
    }


def _features(
    repository: Path, query_ids: Sequence[str], rows: Sequence[Mapping[str, Any]], workers: int,
) -> tuple[np.ndarray, tuple[dict[str, Any], ...]]:
    _require(workers >= 1, "workers must be positive")
    spec = load_config(repository / CONFIG).datasets["nasdaq"]
    _require(spec.benchmark is not None and spec.benchmark.path.resolve() == BENCHMARK.resolve(),
             "benchmark configuration differs")
    source = source_from_spec(spec); benchmark = source.load_benchmark()
    _require(benchmark is not None, "benchmark absent")
    def one(item: tuple[str, Mapping[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
        query_id, row = item
        return _feature(query_id, row, source.load(InstrumentKey("nasdaq", str(row["symbol"]))), benchmark)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        values = tuple(executor.map(one, zip(query_ids, rows, strict=True)))
    return np.asarray([value[0] for value in values], dtype=np.float64), tuple(value[1] for value in values)


def _query_authority(repository: Path) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...], dict[str, Any]]:
    registry = _load(repository / REGISTRY)
    _require(registry["registry_digest"] == _digest(registry, {"registry_digest"}), "registry digest differs")
    source = dict(registry["source_lock"]); source_digest = source.pop("source_lock_digest")
    _require(source_digest == stable_hash(source), "source lock digest differs")
    rows_by_id = {str(row["episode_id"]): row for row in registry["cases_data"]}
    _require(len(registry["cases_data"]) == len(rows_by_id) == 3270, "query inventory differs")
    query_ids = tuple(sorted(rows_by_id))
    return query_ids, tuple(dict(rows_by_id[value]) for value in query_ids), registry


def _partition_input_files(repository: Path) -> tuple[Path, ...]:
    return repository / REGISTRY, repository / CONFIG, BENCHMARK


def _support_input_files(repository: Path) -> tuple[Path, ...]:
    return tuple(repository / path for path in (
        R1A_RESULT, V2_RESULT, V2_QUERY_CELLS, V2_COHORT, V2_VERIFIED,
    ))


def _partition_contract() -> dict[str, Any]:
    return {
        "query_order": "ascending query episode ID", "shape": [3270, 141],
        "column_order": "coarse[0:96] + stage.reshape(12,4,C)[:,0:3].ravel(C) + structural[0:9]",
        "rank": "exact float64 ties; doubled 1-based midrank m=a+b; z=(m-(N+1))/(N-1); constants +0.0",
        "scatter": "locally centered float64 Zc.T@Zc under single-thread BLAS",
        "fallback": "relative eigengap <=1e-10; exact integer n*sum(m^2)-sum(m)^2; lowest column tie",
        "projection": "math.fsum feature order; exact-equal boundary uses query ID; else margin >1e-10*max(1,maxabs)",
        "requested_k": list(K_VALUES), "primary_k": PRIMARY_K,
        "secondary_k": [12, 16], "no_merge_retry_or_coarsening": True,
        "parallel_reconstruction_workers": 12,
    }


def _support_contract() -> dict[str, Any]:
    return {
        "base": "verified v2 session_21 cells", "cap": 4096,
        "minimum_episode_coverage": 0.9, "minimum_link_coverage": 0.9,
        "primary": "K8 only; K12/K16 sensitivity cannot rescue",
        "failure_states": [
            "partition_invalid", "reproducibility_failed", "support_inadequate",
            "support_pass_pending_independent_verification",
        ],
    }


def _claims() -> dict[str, Any]:
    return {
        "partition_only_before_support": True, "real_forward_outcomes_accessed": False,
        "r1b_statistics_opened": False, "b2_execution_authorized": False,
        "adequacy_labels_authorized": False, "predictive_claim_authorized": False,
        "production_promotion_authorized": False,
    }


def preregister(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(); _clean(repository)
    path = repository / PREREGISTRATION
    _require(not path.exists() and not path.is_symlink(), "preregistration exists")
    _require(not (repository / OUTPUT).exists() and not (repository / OUTPUT).is_symlink(),
             "partition output already exists")
    _require(not (repository / SUPPORT_OUTPUT).exists() and not (repository / SUPPORT_OUTPUT).is_symlink(),
             "support output already exists")
    for relative in (*PARTITION_RUNTIME, *SUPPORT_RUNTIME):
        _require((repository / relative).is_file(), f"runtime absent: {relative}")
    partition_files = _partition_input_files(repository)
    support_files = _support_input_files(repository)
    v2_verified = _load(repository / V2_VERIFIED)
    _require(v2_verified.get("verification_digest") == V2_VERIFICATION_DIGEST
             and v2_verified.get("passed") is True
             and v2_verified.get("real_forward_outcomes_accessed") is False,
             "v2 verification authority differs")
    state = {
        "schema_version": "m04r14-r1b-balanced-v3-preregistration-v1",
        "implementation_commit": _git(repository, "rev-parse", "HEAD"),
        "execution": {
            "partition_output": str(OUTPUT), "support_output": str(SUPPORT_OUTPUT),
            "publication": "flock-serialized create-only atomic directory rename with fsync",
        },
        "partition_file_sha256": {str(path.resolve()): _sha(path) for path in partition_files},
        "support_file_sha256": {str(path.resolve()): _sha(path) for path in support_files},
        "runtime_sha256": {
            relative: _sha(repository / relative) for relative in (*PARTITION_RUNTIME, *SUPPORT_RUNTIME)
        },
        "environment": _runtime_environment(),
        "inventory": {"queries": 3270, "cohort_episodes": 369, "cohort_links": 2865},
        "partition_contract": _partition_contract(),
        "support_contract": _support_contract(),
        "claims": _claims(),
        "prior_exposure": {
            "v2_base_support_observed": True, "v2_structure_partition_degenerate": True,
            "v3_real_vectors_or_support_observed": False,
            "v2_verification_digest": v2_verified["verification_digest"],
        },
    }
    payload = {**state, "preregistration_digest": stable_hash(state)}
    _atomic_json(path, payload); return payload


def _validate_h1(repository: Path, prereg: Mapping[str, Any]) -> str:
    head = _clean(repository)
    _require(_git(repository, "rev-parse", "HEAD^") == prereg["implementation_commit"], "H0/H1 differs")
    _require(_git(repository, "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD")
             == PREREGISTRATION.as_posix(), "H1 is not sole preregistration")
    blob = subprocess.run(("git", "show", f"HEAD:{PREREGISTRATION}"), cwd=repository, capture_output=True)
    _require(blob.returncode == 0 and blob.stdout == (repository / PREREGISTRATION).read_bytes(),
             "committed preregistration differs")
    _require(prereg["preregistration_digest"] == _digest(prereg, {"preregistration_digest"}),
             "preregistration digest differs")
    _require(set(prereg) == {
        "claims", "environment", "execution", "implementation_commit", "inventory",
        "partition_contract", "partition_file_sha256", "preregistration_digest",
        "prior_exposure", "runtime_sha256", "schema_version", "support_contract",
        "support_file_sha256",
    }, "preregistration field closure differs")
    _require(all((
        prereg["schema_version"] == "m04r14-r1b-balanced-v3-preregistration-v1",
        prereg["execution"] == {
            "partition_output": str(OUTPUT), "support_output": str(SUPPORT_OUTPUT),
            "publication": "flock-serialized create-only atomic directory rename with fsync",
        },
        prereg["inventory"] == {"queries": 3270, "cohort_episodes": 369, "cohort_links": 2865},
        prereg["partition_contract"] == _partition_contract(),
        prereg["support_contract"] == _support_contract(),
        prereg["claims"] == _claims(),
        prereg["prior_exposure"] == {
            "v2_base_support_observed": True, "v2_structure_partition_degenerate": True,
            "v3_real_vectors_or_support_observed": False,
            "v2_verification_digest": V2_VERIFICATION_DIGEST,
        },
        set(prereg["partition_file_sha256"])
            == {str(path.resolve()) for path in _partition_input_files(repository)},
        set(prereg["support_file_sha256"])
            == {str(path.resolve()) for path in _support_input_files(repository)},
        set(prereg["runtime_sha256"]) == set((*PARTITION_RUNTIME, *SUPPORT_RUNTIME)),
    )), "preregistration contract differs")
    _require(prereg["environment"] == _runtime_environment(), "solver environment differs")
    for group in ("partition_file_sha256", "runtime_sha256"):
        for path, expected in prereg[group].items():
            target = Path(path) if Path(path).is_absolute() else repository / path
            _require(_sha(target) == expected, f"frozen file differs: {path}")
    return head


def _construct_partition_payload(
    vectors: np.ndarray, audits: Sequence[Mapping[str, Any]], query_ids: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    transformed, integers, constant = midrank_transform(vectors)
    transform_rows = {
        "schema_version": "m04r14-r1b-balanced-transform-v1",
        "query_ids_digest": id_digest(query_ids), "query_audits": list(audits),
        "raw_vector_digest": array_digest(vectors),
        "integer_midrank_digest": integer_array_digest(integers),
        "transformed_digest": array_digest(transformed), "shape": list(transformed.shape),
        "constant_columns": list(constant),
    }
    partitions: dict[str, Any] = {}
    assignments = [{"query_episode_id": value, "labels": {}} for value in query_ids]
    for k in K_VALUES:
        try:
            built = balanced_recursive_partition(transformed, integers, query_ids, leaves=k)
            for index, label in enumerate(built.labels):
                assignments[index]["labels"][str(k)] = int(label)
            partitions[str(k)] = {
                "status": "partition_valid", "requested_k": k,
                "retained_k": len(built.leaf_paths),
                "leaf_paths": list(built.leaf_paths), "leaf_sizes": list(built.leaf_sizes),
                "membership_digest": stable_hash([
                    [query_ids[index], int(label)] for index, label in enumerate(built.labels)
                ]),
                "splits": [asdict(value) for value in built.splits],
            }
        except BalancedPartitionError as error:
            partitions[str(k)] = {
                "status": "partition_invalid", "reason": str(error), "requested_k": k,
            }
    return transform_rows, {
        "schema_version": "m04r14-r1b-balanced-partitions-v1",
        "assignments": assignments, "partitions": partitions,
    }


def run(repository: Path, *, workers: int) -> dict[str, Any]:
    _require(workers == 12, "parallel reconstruction requires exactly 12 workers")
    repository = repository.resolve(); output = repository / OUTPUT
    _require(not output.exists() and not output.is_symlink(), "partition output exists")
    prereg = _load(repository / PREREGISTRATION); h1 = _validate_h1(repository, prereg)
    query_ids, rows, registry = _query_authority(repository)
    parallel, parallel_audits = _features(repository, query_ids, rows, workers)
    serial, serial_audits = _features(repository, query_ids, rows, 1)
    parallel_transform, partition_payload = _construct_partition_payload(
        parallel, parallel_audits, query_ids,
    )
    serial_transform, serial_partition_payload = _construct_partition_payload(
        serial, serial_audits, query_ids,
    )
    raw_identical = np.array_equal(parallel, serial) and parallel_audits == serial_audits
    transform_digest_identical = stable_hash(parallel_transform) == stable_hash(serial_transform)
    partition_digest_identical = stable_hash(partition_payload) == stable_hash(serial_partition_payload)
    reproducible = raw_identical and transform_digest_identical and partition_digest_identical
    transform_rows = parallel_transform
    if not reproducible:
        transform_rows["reproducibility_comparison"] = {
            "raw_identical": raw_identical,
            "transform_digest_identical": transform_digest_identical,
            "partition_digest_identical": partition_digest_identical,
            "serial_raw_vector_digest": array_digest(serial),
            "serial_transform_digest": stable_hash(serial_transform),
            "serial_partition_digest": stable_hash(serial_partition_payload),
        }
    partitions = partition_payload["partitions"]
    primary_valid = reproducible \
        and partitions.get(str(PRIMARY_K), {}).get("status") == "partition_valid"
    status = "partition_valid" if primary_valid else (
        "reproducibility_failed" if not reproducible else "partition_invalid"
    )
    state = {
        "schema_version": SCHEMA, "status": status, "passed": primary_valid,
        "preregistration_commit": h1, "preregistration_digest": prereg["preregistration_digest"],
        "registry_digest": registry["registry_digest"], "queries": len(query_ids),
        "serial_parallel_reconstruction_identical": reproducible,
        "serial_parallel_transform_digest_identical": transform_digest_identical,
        "serial_parallel_partition_digest_identical": partition_digest_identical,
        "primary_k": PRIMARY_K, "secondary_k": [12, 16],
        "real_forward_outcomes_accessed": False, "candidate_or_eligibility_inputs_accessed": False,
        "r1b_statistics_opened": False, "b2_execution_authorized": False,
        "adequacy_labels_authorized": False, "predictive_claim_authorized": False,
        "production_promotion_authorized": False,
    }
    temporary = output.parent / f".{output.name}.tmp-{os.getpid()}-{uuid4().hex}"
    try:
        temporary.mkdir(parents=True)
        _write(temporary / "TRANSFORM.json", transform_rows)
        _write(temporary / "PARTITIONS.json", partition_payload)
        state["transform_sha256"] = _sha(temporary / "TRANSFORM.json")
        state["partitions_sha256"] = _sha(temporary / "PARTITIONS.json")
        result = {**state, "result_digest": stable_hash(state),
                  "created_at": datetime.now(timezone.utc).isoformat()}
        _write(temporary / "RESULT.json", result); _publish(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True); raise
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("preregister", "run"))
    parser.add_argument("--repository", type=Path, default=Path.cwd()); parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(argv)
    result = preregister(args.repository) if args.mode == "preregister" else run(args.repository, workers=args.workers)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
