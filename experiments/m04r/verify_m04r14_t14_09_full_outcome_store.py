"""Independently verify every episode in the complete T14-09 outcome store."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import multiprocessing
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import pandas as pd

from experiments.m04r import m04r14_shadow_run as retrieval
from experiments.m04r.m04r14_t14_09_outcome_oracle import (
    prepare_reference_series,
    reference_prepared_episode,
)
from market_analogues.adapters import source_from_spec
from market_analogues.config import load_config
from market_analogues.types import EpisodeKey, InstrumentKey, stable_hash


SCHEMA = "m04r14-t14-09-full-outcome-verification-v1"
STORE_SCHEMA = "m04r14-t14-09-full-outcome-store-v1"
PARTITION_SCHEMA = "m04r14-t14-09-full-outcome-partition-v1"
PREREG_SCHEMA = "m04r14-t14-09-full-outcome-preregistration-v1"
PREREGISTRATION = Path(
    "experiments/m04r/m04r14_t14_09_full_outcome_store_preregistered.json"
)
CACHE = Path("config/data/analogues/m04r14/t14-09-full-outcome-partitions-v1")
OUTPUT = Path("config/data/analogues/m04r14/t14-09-full-outcome-store-v1")
VERIFICATION = Path(
    "config/data/analogues/m04r14/t14-09-full-outcome-store-v1-verification"
)
CONTRACT = Path("config/m04r14-t14-09-outcome-contract.json")
SMOKE_VERIFICATION = Path(
    "config/data/analogues/m04r14/t14-09-outcome-smoke-v1-verification/VERIFIED.json"
)
SEMANTIC = Path(
    "config/data/analogues/m04r14/nasdaq-shadow-interrupted-verification-v1/VERIFIED.json"
)
AUDIT = Path("config/data/analogues/m04r14/nasdaq-shadow-audit-comparison-v1/RESULT.json")
HORIZONS = (5, 10, 20, 40, 60, 126)
PROCESSES = 12
SEMANTIC_DIGEST_SCHEMA = "canonical-json-record-chunks-v1"
SEMANTIC_CHUNK_ROWS = 16384


class FullVerificationError(RuntimeError):
    pass


def _git(repository: Path, *args: str, raw: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=repository, capture_output=True,
        text=not raw, check=True,
    )
    return result.stdout if raw else result.stdout.strip()


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise FullVerificationError(f"regular JSON file required: {path}")
    raw = path.read_bytes()

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise FullVerificationError(f"duplicate JSON key: {path}:{key}")
            result[key] = value
        return result

    value = json.loads(
        raw, object_pairs_hook=pairs,
        parse_constant=lambda item: (_ for _ in ()).throw(
            FullVerificationError(f"non-finite JSON: {path}:{item}")
        ),
    )
    if type(value) is not dict:
        raise FullVerificationError(f"JSON object required: {path}")
    return value, raw


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise FullVerificationError(f"regular file required: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value.item() if hasattr(value, "item") else value


def _records(frame: pd.DataFrame, order: Sequence[str]) -> list[dict[str, Any]]:
    ordered = frame.sort_values(list(order), kind="stable").reset_index(drop=True)
    return _plain(ordered.to_dict("records"))


def _frame_digest(frame: pd.DataFrame, order: Sequence[str]) -> str:
    ordered = frame.sort_values(list(order), kind="stable").reset_index(drop=True)
    digest = sha256()
    digest.update(f"{SEMANTIC_DIGEST_SCHEMA}\0{len(ordered)}\0".encode())
    for start in range(0, len(ordered), SEMANTIC_CHUNK_ROWS):
        records = _plain(
            ordered.iloc[start:start + SEMANTIC_CHUNK_ROWS].to_dict("records")
        )
        payload = json.dumps(
            records, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _record_digest(records: Sequence[Mapping[str, Any]]) -> str:
    digest = sha256()
    digest.update(f"{SEMANTIC_DIGEST_SCHEMA}\0{len(records)}\0".encode())
    for start in range(0, len(records), SEMANTIC_CHUNK_ROWS):
        payload = json.dumps(
            _plain(list(records[start:start + SEMANTIC_CHUNK_ROWS])),
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _receipt_valid(value: Mapping[str, Any], *, timing: bool = False) -> bool:
    omitted = {"result_digest", "created_at"}
    if timing:
        omitted |= {"elapsed_seconds", "partition_elapsed_seconds"}
    return value.get("result_digest") == stable_hash({
        key: item for key, item in value.items() if key not in omitted
    })


def _manifest(root: Path, names: Sequence[str]) -> list[dict[str, Any]]:
    return [{
        "path": name, "bytes": (root / name).stat().st_size,
        "sha256": _sha(root / name),
    } for name in names]


def _sole_child(repository: Path, raw: bytes, h0: str) -> str:
    accepted: list[str] = []
    for line in str(_git(repository, "rev-list", "--all", "--children")).splitlines():
        values = line.split()
        if not values or values[0] != h0:
            continue
        for child in values[1:]:
            lineage = str(_git(
                repository, "rev-list", "--parents", "-n", "1", child,
            )).split()
            changed = str(_git(
                repository, "diff-tree", "--no-commit-id", "--name-only", "-r", child,
            )).splitlines()
            if lineage != [child, h0] or changed != [PREREGISTRATION.as_posix()]:
                continue
            if _git(repository, "show", f"{child}:{PREREGISTRATION}", raw=True) == raw:
                accepted.append(child)
    if len(set(accepted)) != 1:
        raise FullVerificationError("full-store preregistration lifecycle differs")
    return accepted[0]


def _links(repository: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    registry, _ = _read(repository / retrieval.REGISTRY / "query-registry.json")
    semantic, _ = _read(repository / SEMANTIC)
    audit, _ = _read(repository / AUDIT)
    if not all((
        registry.get("passed") is True,
        semantic.get("semantic_passed") is True,
        semantic.get("verified_cases") == 3270,
        semantic.get("verified_matches") == 65400,
        semantic.get("registry_digest") == registry.get("registry_digest"),
        audit.get("passed") is True and audit.get("matching_positions") == 240,
        type(registry.get("cases_data")) is list and len(registry["cases_data"]) == 3270,
    )):
        raise FullVerificationError("retrieval authority differs")
    links: list[dict[str, Any]] = []
    for registered in registry["cases_data"]:
        query_id = str(registered["episode_id"])
        case, _ = _read(repository / retrieval.OUTPUT / "cases" / f"{query_id}.json")
        if not all((
            case.get("query_episode_id") == query_id,
            case.get("registry_case_id") == registered.get("case_id"),
            case.get("gate_passed") is True,
            case.get("result_digest") == retrieval._deterministic_case_digest(case),
            case.get("checkpoint_integrity_digest") == retrieval._integrity_digest(case),
            type(case.get("matches")) is list and len(case["matches"]) == 20,
        )):
            raise FullVerificationError(f"candidate case differs: {registered.get('case_id')}")
        for rank, match in enumerate(case["matches"], 1):
            links.append({
                "query_case_id": str(registered["case_id"]),
                "query_episode_id": query_id,
                "query_symbol": str(registered["symbol"]),
                "query_cutoff": str(case["query_cutoff"]),
                "match_rank": rank,
                "matched_episode_id": str(match["episode_id"]),
                "matched_symbol": str(match["symbol"]),
                "matched_cutoff": str(match["cutoff"]),
                "total_distance": float(match["total_distance"]),
                "match_digest": stable_hash(match),
                "candidate_case_result_digest": str(case["result_digest"]),
            })
    if len(links) != 65400 or len({
        (row["query_episode_id"], row["match_rank"]) for row in links
    }) != 65400:
        raise FullVerificationError("full link inventory differs")
    return links, registry


def _requests(
    repository: Path, links: Sequence[Mapping[str, Any]], registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for link in links:
        row = {
            "episode_id": str(link["matched_episode_id"]),
            "symbol": str(link["matched_symbol"]),
            "cutoff": str(link["matched_cutoff"]),
        }
        if row["episode_id"] in by_id and by_id[row["episode_id"]] != row:
            raise FullVerificationError("conflicting matched episode identity")
        by_id[row["episode_id"]] = row
    base = [by_id[key] for key in sorted(by_id)]
    lock = registry["source_lock"]
    quality_path = Path(str(lock["quality_path"]))
    if _sha(quality_path) != lock.get("quality_sha256"):
        raise FullVerificationError("quality metadata drifted")
    quality = pd.read_parquet(quality_path)
    if not {"symbol", "source_hash"}.issubset(quality.columns) \
            or quality.symbol.astype(str).duplicated().any():
        raise FullVerificationError("quality fingerprint schema differs")
    fingerprints = {
        str(row.symbol): str(row.source_hash) for row in quality.itertuples(index=False)
    }
    try:
        return [{
            **row, "expected_source_fingerprint": fingerprints[row["symbol"]],
        } for row in base]
    except KeyError as exc:
        raise FullVerificationError(f"matched symbol has no source lock: {exc}") from exc


def _groups(requests: Sequence[Mapping[str, Any]]) -> list[tuple[dict[str, Any], ...]]:
    result: list[list[dict[str, Any]]] = [[] for _ in range(PROCESSES)]
    for raw in requests:
        row = dict(raw)
        index = int(sha256(row["symbol"].encode()).hexdigest(), 16) % PROCESSES
        result[index].append(row)
    if any(not group for group in result):
        raise FullVerificationError("full verifier partition is empty")
    return [tuple(sorted(group, key=lambda row: (
        row["symbol"], row["cutoff"], row["episode_id"],
    ))) for group in result]


def _valid_source(frame: pd.DataFrame) -> bool:
    required = {"timestamp", "open", "high", "low", "close"}
    if not required.issubset(frame.columns):
        return False
    numeric = frame[["open", "high", "low", "close"]]
    return not any((
        bool(frame.attrs.get("source_timestamp_reordered")),
        bool(frame.attrs.get("source_duplicate_timestamps")),
        bool(frame["timestamp"].duplicated().any()),
        not bool(frame["timestamp"].is_monotonic_increasing),
        not bool(numeric.apply(lambda column: column.map(math.isfinite)).all().all()),
        bool((numeric <= 0).any().any()),
        bool((frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any()),
        bool((frame["low"] > frame[["open", "close", "high"]].min(axis=1)).any()),
    ))


def _verify_partition_worker(
    repository: str, index: int, requests: tuple[dict[str, Any], ...],
    contract_digest: str, source_content_digest: str,
) -> dict[str, Any]:
    root = Path(repository)
    part = root / CACHE / f"part-{index:02d}"
    observed_outcomes = pd.read_parquet(part / "episode-outcomes.parquet")
    observed_paths = pd.read_parquet(part / "future-paths.parquet")
    config = load_config(root / retrieval.CONFIG)
    source = source_from_spec(config.datasets["nasdaq"])
    benchmark = source.load_benchmark()
    registry, _ = _read(root / retrieval.REGISTRY / "query-registry.json")
    if benchmark is None or not _valid_source(benchmark) \
            or source.benchmark_fingerprint() != registry["source_lock"]["benchmark_sha256"]:
        raise FullVerificationError("benchmark differs during full independent verification")
    reference_benchmark = prepare_reference_series(benchmark)
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for request in requests:
        by_symbol.setdefault(request["symbol"], []).append(request)
    outcomes: list[dict[str, Any]] = []
    paths: list[dict[str, Any]] = []
    for symbol in sorted(by_symbol):
        key = InstrumentKey("nasdaq", symbol)
        stock = source.load(key)
        fingerprint = source.fingerprint(key)
        if not _valid_source(stock) \
                or {row["expected_source_fingerprint"] for row in by_symbol[symbol]} != {fingerprint}:
            raise FullVerificationError(f"source differs during verification: {symbol}")
        reference_stock = prepare_reference_series(stock)
        for request in by_symbol[symbol]:
            cutoff = pd.Timestamp(request["cutoff"])
            if EpisodeKey(key, cutoff, 252, "dense-v1").id != request["episode_id"]:
                raise FullVerificationError("episode identity differs during verification")
            expected_outcomes, expected_paths = reference_prepared_episode(
                reference_stock, reference_benchmark,
                episode_id=request["episode_id"], cutoff=cutoff,
                source_fingerprint=fingerprint, contract_digest=contract_digest,
                source_content_digest=source_content_digest,
            )
            outcomes.extend(expected_outcomes)
            paths.extend(expected_paths)
    outcomes = sorted(outcomes, key=lambda row: (
        row["episode_id"], row["horizon_sessions"],
    ))
    paths = sorted(paths, key=lambda row: (row["episode_id"], row["step"]))
    if _records(observed_outcomes, ("episode_id", "horizon_sessions")) != outcomes:
        raise FullVerificationError(f"independent outcomes differ in partition {index}")
    if _records(observed_paths, ("episode_id", "step")) != paths:
        raise FullVerificationError(f"independent paths differ in partition {index}")
    return {
        "partition": index, "request_count": len(requests),
        "outcome_rows": len(outcomes), "path_rows": len(paths),
        "semantic_digest_schema": SEMANTIC_DIGEST_SCHEMA,
        "outcome_digest": _record_digest(outcomes),
        "path_digest": _record_digest(paths),
    }


def _eligibility(completion: Any, query_cutoff: Any, complete: bool) -> dict[str, Any]:
    if not complete or completion is None:
        return {"eligible": False, "reason": "incomplete_horizon"}
    if pd.Timestamp(completion) > pd.Timestamp(query_cutoff):
        return {"eligible": False, "reason": "outcome_not_yet_observable"}
    return {"eligible": True, "reason": "eligible"}


def verify(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    prereg, prereg_raw = _read(repository / PREREGISTRATION)
    prereg_state = {k: v for k, v in prereg.items() if k != "preregistration_digest"}
    if prereg.get("schema_version") != PREREG_SCHEMA \
            or prereg.get("preregistration_digest") != stable_hash(prereg_state):
        raise FullVerificationError("full-store preregistration differs")
    h0 = str(prereg.get("implementation_h0"))
    h1 = _sole_child(repository, prereg_raw, h0)
    for name, expected in prereg.get("runtime_files", {}).items():
        if sha256(_git(repository, "show", f"{h0}:{name}", raw=True)).hexdigest() != expected \
                or sha256(_git(repository, "show", f"{h1}:{name}", raw=True)).hexdigest() != expected \
                or _sha(repository / name) != expected:
            raise FullVerificationError(f"runtime source drifted: {name}")
    contract, contract_raw = _read(repository / CONTRACT)
    smoke_verified, smoke_raw = _read(repository / SMOKE_VERIFICATION)
    if not all((
        prereg.get("contract_digest") == contract.get("contract_digest"),
        prereg.get("contract_sha256") == sha256(contract_raw).hexdigest(),
        prereg.get("smoke_verification_result_digest") == smoke_verified.get("result_digest"),
        prereg.get("smoke_verification_sha256") == sha256(smoke_raw).hexdigest(),
        _receipt_valid(smoke_verified), smoke_verified.get("full_run_authorized") is True,
    )):
        raise FullVerificationError("full-store prerequisite differs")
    links, registry = _links(repository)
    requests = _requests(repository, links, registry)
    groups = _groups(requests)
    if any((
        prereg.get("registry_digest") != registry.get("registry_digest"),
        prereg.get("source_content_digest") != contract["retrieval_inputs"]["source_content_digest"],
        prereg.get("query_count") != 3270,
        prereg.get("link_count") != len(links),
        prereg.get("link_digest") != stable_hash(links),
        prereg.get("unique_episode_count") != len(requests),
        prereg.get("unique_episode_request_digest") != stable_hash(requests),
        prereg.get("partition_request_counts") != [len(group) for group in groups],
        prereg.get("partition_request_digests") != [stable_hash(group) for group in groups],
    )):
        raise FullVerificationError("full-store preregistered inventory differs")

    cache = repository / CACHE
    expected_cache_names = {"RUN_STARTED.json", "CACHE_SEALED.json"} | {
        f"part-{index:02d}" for index in range(PROCESSES)
    }
    if cache.is_symlink() or not cache.is_dir() \
            or {path.name for path in cache.iterdir()} != expected_cache_names \
            or any(path.is_symlink() for path in cache.iterdir()):
        raise FullVerificationError("full partition cache inventory differs")
    cache_seal, _ = _read(cache / "CACHE_SEALED.json")
    if not all((
        _receipt_valid(cache_seal), cache_seal.get("passed") is True,
        cache_seal.get("partition_count") == PROCESSES,
        cache_seal.get("unique_episodes") == len(requests),
        cache_seal.get("outcome_rows") == len(requests) * len(HORIZONS),
        cache_seal.get("complete_real_forward_outcomes_accessed") is True,
    )):
        raise FullVerificationError("full partition cache seal differs")
    partition_receipts: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        part = cache / f"part-{index:02d}"
        if {path.name for path in part.iterdir()} != {
            "PARTITION.json", "episode-outcomes.parquet", "future-paths.parquet",
        } or any(path.is_symlink() for path in part.iterdir()):
            raise FullVerificationError(f"partition inventory differs: {index}")
        receipt, _ = _read(part / "PARTITION.json")
        if not all((
            _receipt_valid(receipt, timing=True), receipt.get("passed") is True,
            receipt.get("partition") == index,
            receipt.get("request_count") == len(group),
            receipt.get("request_digest") == stable_hash(group),
            receipt.get("outcome_rows") == len(group) * len(HORIZONS),
            receipt.get("semantic_digest_schema") == SEMANTIC_DIGEST_SCHEMA,
            receipt.get("file_manifest") == _manifest(
                part, ("episode-outcomes.parquet", "future-paths.parquet"),
            ),
        )):
            raise FullVerificationError(f"partition receipt differs: {index}")
        partition_receipts.append(receipt)
    if cache_seal.get("partition_result_digests") != [
        row["result_digest"] for row in partition_receipts
    ]:
        raise FullVerificationError("partition cache digest list differs")

    output = repository / OUTPUT
    expected_output_names = {
        "RUN_STARTED.json", "episode-outcomes.parquet", "future-paths.parquet",
        "query-match-links.parquet", "COVERAGE.json", "SEALED.json",
    }
    if output.is_symlink() or not output.is_dir() \
            or {path.name for path in output.iterdir()} != expected_output_names \
            or any(path.is_symlink() for path in output.iterdir()):
        raise FullVerificationError("full outcome output inventory differs")
    started, _ = _read(output / "RUN_STARTED.json")
    seal, seal_raw = _read(output / "SEALED.json")
    expected_files = (
        "episode-outcomes.parquet", "future-paths.parquet",
        "query-match-links.parquet", "COVERAGE.json",
    )
    if not all((
        _receipt_valid(seal, timing=True), seal.get("passed") is True,
        seal.get("preregistration_h1") == h1,
        seal.get("preregistration_digest") == prereg["preregistration_digest"],
        seal.get("contract_digest") == contract["contract_digest"],
        seal.get("source_content_digest") == prereg["source_content_digest"],
        seal.get("partition_cache_result_digest") == cache_seal["result_digest"],
        seal.get("query_count") == 3270, seal.get("query_links") == 65400,
        seal.get("unique_episodes") == len(requests),
        seal.get("unique_symbols") == prereg["unique_symbol_count"],
        seal.get("outcome_rows") == len(requests) * len(HORIZONS),
        seal.get("file_manifest") == _manifest(output, expected_files),
        seal.get("complete_real_forward_outcomes_accessed") is True,
        seal.get("outcomes_affected_retrieval") is False,
        started.get("status") == "partitions_running",
        started.get("preregistration_h1") == h1,
        started.get("preregistration_digest") == prereg["preregistration_digest"],
        started.get("partition_count") == PROCESSES,
        started.get("complete_real_forward_outcomes_accessed") is False,
    )):
        raise FullVerificationError("full outcome seal differs")

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=PROCESSES, mp_context=context) as pool:
        verified = list(pool.map(
            _verify_partition_worker, [str(repository)] * PROCESSES,
            range(PROCESSES), groups,
            [contract["contract_digest"]] * PROCESSES,
            [prereg["source_content_digest"]] * PROCESSES,
        ))
    verified.sort(key=lambda row: row["partition"])
    for row, receipt in zip(verified, partition_receipts, strict=True):
        if any(row[key] != receipt[key] for key in (
            "partition", "request_count", "outcome_rows", "path_rows",
            "semantic_digest_schema", "outcome_digest", "path_digest",
        )):
            raise FullVerificationError("independent partition digest differs")

    outcomes = pd.read_parquet(output / "episode-outcomes.parquet")
    paths = pd.read_parquet(output / "future-paths.parquet")
    observed_links = pd.read_parquet(output / "query-match-links.parquet")
    coverage, _ = _read(output / "COVERAGE.json")
    if outcomes[["episode_id", "horizon_sessions"]].duplicated().any() \
            or paths[["episode_id", "step"]].duplicated().any() \
            or set(outcomes.episode_id.astype(str)) != {row["episode_id"] for row in requests} \
            or len(paths) != sum(row["path_rows"] for row in verified):
        raise FullVerificationError("aggregate episode identity coverage differs")
    for index, group in enumerate(groups):
        ids = {row["episode_id"] for row in group}
        if _frame_digest(
            outcomes.loc[outcomes.episode_id.isin(ids)], ("episode_id", "horizon_sessions"),
        ) != verified[index]["outcome_digest"] or _frame_digest(
            paths.loc[paths.episode_id.isin(ids)], ("episode_id", "step"),
        ) != verified[index]["path_digest"]:
            raise FullVerificationError(f"aggregate differs from partition {index}")

    outcome_by_key = {
        (str(row.episode_id), int(row.horizon_sessions)): row
        for row in outcomes.itertuples(index=False)
    }
    expected_links: list[dict[str, Any]] = []
    for link in links:
        eligibility = {
            str(horizon): _eligibility(
                outcome_by_key[(link["matched_episode_id"], horizon)].completion_timestamp,
                link["query_cutoff"],
                bool(outcome_by_key[(link["matched_episode_id"], horizon)].complete),
            ) for horizon in HORIZONS
        }
        expected_links.append({
            **link,
            "source_fingerprint": str(
                outcome_by_key[(link["matched_episode_id"], 5)].source_fingerprint
            ),
            "outcome_eligibility_json": json.dumps(
                eligibility, sort_keys=True, separators=(",", ":"),
            ),
        })
    expected_links.sort(key=lambda row: (row["query_episode_id"], row["match_rank"]))
    link_records = _records(observed_links, ("query_episode_id", "match_rank"))
    if link_records != expected_links:
        raise FullVerificationError("full query-link outcome join differs")

    outcome_records = _records(outcomes, ("episode_id", "horizon_sessions"))
    coverage_state = {k: v for k, v in coverage.items() if k != "result_digest"}
    expected_coverage = {
        "schema_version": STORE_SCHEMA, "status": "complete",
        "query_links": len(link_records), "unique_episodes": len(requests),
        "unique_symbols": prereg["unique_symbol_count"],
        "outcome_rows": len(outcome_records), "path_rows": len(paths),
        "complete_by_horizon": {
            str(horizon): sum(
                bool(row["complete"]) for row in outcome_records
                if int(row["horizon_sessions"]) == horizon
            ) for horizon in HORIZONS
        },
        "status_counts": {
            key: sum(row["status"] == key for row in outcome_records)
            for key in sorted({str(row["status"]) for row in outcome_records})
        },
        "barrier_counts": {
            key: sum(
                int(row["horizon_sessions"]) == 20 and str(row["barrier_label"]) == key
                for row in outcome_records
            ) for key in sorted({
                str(row["barrier_label"]) for row in outcome_records
                if int(row["horizon_sessions"]) == 20
            })
        },
        "complete_real_forward_outcomes_accessed": True,
    }
    semantic = {
        "semantic_digest_schema": SEMANTIC_DIGEST_SCHEMA,
        "outcome_digest": _frame_digest(outcomes, ("episode_id", "horizon_sessions")),
        "path_digest": _frame_digest(paths, ("episode_id", "step")),
        "link_digest": _frame_digest(observed_links, ("query_episode_id", "match_rank")),
        "coverage_result_digest": coverage.get("result_digest"),
    }
    if coverage_state != expected_coverage \
            or coverage.get("result_digest") != stable_hash(coverage_state) \
            or seal.get("semantic_digests") != semantic:
        raise FullVerificationError("full coverage or semantic digest differs")
    state = {
        "schema_version": SCHEMA, "status": "verified", "passed": True,
        "store_result_digest": seal["result_digest"],
        "store_result_sha256": sha256(seal_raw).hexdigest(),
        "contract_digest": contract["contract_digest"],
        "query_count": 3270, "query_links": 65400,
        "unique_episodes": len(requests),
        "verified_outcome_rows": len(outcome_records),
        "verified_path_rows": len(paths),
        "verified_partitions": PROCESSES,
        "complete_real_forward_outcomes_accessed": True,
        "outcomes_affected_retrieval": False,
        "evidence_cards_authorized": True,
        "production_promotion_authorized": False,
    }
    return {**state, "result_digest": stable_hash(state)}


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FullVerificationError("full verification root exists")
    path.mkdir(parents=False)
    descriptor = os.open(path / "VERIFIED.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json.dumps({
            **value, "created_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    repository = args.repository.resolve(strict=True)
    result = verify(repository)
    if not args.dry_run:
        _publish(repository / VERIFICATION, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
