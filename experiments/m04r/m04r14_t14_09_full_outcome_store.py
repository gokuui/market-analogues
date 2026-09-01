"""Preregister and build the complete T14-09 causal outcome store."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from time import perf_counter
from typing import Any, Mapping, Sequence

import pandas as pd

from experiments.m04r import m04r14_shadow_run as retrieval
from experiments.m04r import m04r14_t14_09_outcome_smoke as smoke
from market_analogues.adapters import source_from_spec
from market_analogues.causal_outcomes import (
    compute_prepared_episode_outcomes,
    outcome_embargo,
    prepare_outcome_sessions,
)
from market_analogues.config import load_config
from market_analogues.types import EpisodeKey, InstrumentKey, stable_hash


SCHEMA = "m04r14-t14-09-full-outcome-store-v1"
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
SMOKE_VERIFICATION = Path(
    "config/data/analogues/m04r14/t14-09-outcome-smoke-v1-verification/VERIFIED.json"
)
RUNTIME_FILES = (
    "experiments/m04r/m04r14_t14_09_full_outcome_store.py",
    "experiments/m04r/verify_m04r14_t14_09_full_outcome_store.py",
    "experiments/m04r/m04r14_t14_09_outcome_oracle.py",
    "experiments/m04r/m04r14_t14_09_outcome_smoke.py",
    "src/market_analogues/causal_outcomes.py",
)
HORIZONS = (5, 10, 20, 40, 60, 126)
PROCESSES = 12
SEMANTIC_DIGEST_SCHEMA = "canonical-json-record-chunks-v1"
SEMANTIC_CHUNK_ROWS = 16384


class FullOutcomeError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(repository: Path, *args: str, raw: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=repository, capture_output=True,
        text=not raw, check=True,
    )
    return result.stdout if raw else result.stdout.strip()


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        return smoke._read(path)
    except Exception as exc:
        raise FullOutcomeError(str(exc)) from exc


def _sha(path: Path) -> str:
    try:
        return smoke._sha(path)
    except Exception as exc:
        raise FullOutcomeError(str(exc)) from exc


def _receipt_valid(value: Mapping[str, Any], *, timing: bool = False) -> bool:
    return smoke._receipt_valid(value, timing=timing)


def _full_links(repository: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    registry, _ = _read(repository / retrieval.REGISTRY / "query-registry.json")
    semantic, _ = _read(repository / smoke.SEMANTIC)
    audit, _ = _read(repository / smoke.AUDIT)
    smoke_verified, _ = _read(repository / SMOKE_VERIFICATION)
    cases = registry.get("cases_data")
    if not all((
        registry.get("passed") is True,
        type(cases) is list and len(cases) == 3270,
        semantic.get("semantic_passed") is True,
        semantic.get("verified_cases") == 3270,
        semantic.get("verified_matches") == 65400,
        semantic.get("registry_digest") == registry.get("registry_digest"),
        audit.get("passed") is True and audit.get("matching_positions") == 240,
        smoke_verified.get("passed") is True,
        smoke_verified.get("full_run_authorized") is True,
        smoke_verified.get("outcomes_affected_retrieval") is False,
        smoke_verified.get("contract_digest") == smoke._read(repository / smoke.CONTRACT)[0].get("contract_digest"),
    )):
        raise FullOutcomeError("sealed retrieval/smoke evidence does not authorize full outcomes")
    links: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    for registered in cases:
        query_id = str(registered["episode_id"])
        if query_id in seen_queries:
            raise FullOutcomeError("duplicate registered query episode")
        seen_queries.add(query_id)
        case, _ = _read(repository / retrieval.OUTPUT / "cases" / f"{query_id}.json")
        if not all((
            case.get("query_episode_id") == query_id,
            case.get("registry_case_id") == registered.get("case_id"),
            case.get("gate_passed") is True,
            case.get("result_digest") == retrieval._deterministic_case_digest(case),
            case.get("checkpoint_integrity_digest") == retrieval._integrity_digest(case),
            type(case.get("matches")) is list and len(case["matches"]) == 20,
        )):
            raise FullOutcomeError(f"retrieval case differs: {registered.get('case_id')}")
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
        raise FullOutcomeError("full query-link inventory differs")
    return links, registry


def _requests(
    repository: Path, links: Sequence[Mapping[str, Any]], registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    try:
        base = smoke._requests(links)
        return smoke._bind_source_fingerprints(repository, base, registry)
    except Exception as exc:
        raise FullOutcomeError(str(exc)) from exc


def _groups(requests: Sequence[Mapping[str, Any]]) -> list[tuple[dict[str, Any], ...]]:
    groups: list[list[dict[str, Any]]] = [[] for _ in range(PROCESSES)]
    symbol_partitions: dict[str, int] = {}
    for raw in requests:
        row = dict(raw)
        partition = int(sha256(row["symbol"].encode()).hexdigest(), 16) % PROCESSES
        prior = symbol_partitions.setdefault(row["symbol"], partition)
        if prior != partition:
            raise FullOutcomeError("symbol partition is unstable")
        groups[partition].append(row)
    if any(not group for group in groups):
        raise FullOutcomeError("all twelve full-run partitions must be nonempty")
    return [tuple(sorted(group, key=lambda row: (
        row["symbol"], row["cutoff"], row["episode_id"],
    ))) for group in groups]


def _runtime_manifest(repository: Path, head: str) -> dict[str, str]:
    tracked = set(str(_git(
        repository, "ls-tree", "-r", "--name-only", head,
    )).splitlines())
    names = sorted({
        name for name in tracked
        if name.startswith("src/market_analogues/") and name.endswith(".py")
    } | set(RUNTIME_FILES))
    if any(name not in tracked for name in names):
        raise FullOutcomeError("runtime manifest contains an uncommitted file")
    return {
        name: sha256(_git(repository, "show", f"{head}:{name}", raw=True)).hexdigest()
        for name in names
    }


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise FullOutcomeError("globally clean Git worktree required")
    if (repository / CACHE).exists() or (repository / OUTPUT).exists():
        raise FullOutcomeError("full outcome cache/output must be absent before preregistration")
    h0 = str(_git(repository, "rev-parse", "HEAD"))
    contract, contract_raw = _read(repository / smoke.CONTRACT)
    smoke_verified, smoke_verified_raw = _read(repository / SMOKE_VERIFICATION)
    if not all((
        smoke_verified.get("passed") is True,
        smoke_verified.get("full_run_authorized") is True,
        smoke_verified.get("contract_digest") == contract.get("contract_digest"),
        smoke_verified.get("production_promotion_authorized") is False,
        _receipt_valid(smoke_verified),
    )):
        raise FullOutcomeError("verified smoke prerequisite differs")
    links, registry = _full_links(repository)
    requests = _requests(repository, links, registry)
    groups = _groups(requests)
    state = {
        "schema_version": PREREG_SCHEMA,
        "status": "frozen_before_complete_real_outcome_store",
        "implementation_h0": h0,
        "runtime_files": _runtime_manifest(repository, h0),
        "contract_digest": contract["contract_digest"],
        "contract_sha256": sha256(contract_raw).hexdigest(),
        "smoke_verification_result_digest": smoke_verified["result_digest"],
        "smoke_verification_sha256": sha256(smoke_verified_raw).hexdigest(),
        "registry_digest": registry["registry_digest"],
        "source_content_digest": contract["retrieval_inputs"]["source_content_digest"],
        "query_count": 3270,
        "link_count": len(links),
        "link_digest": stable_hash(links),
        "unique_episode_count": len(requests),
        "unique_episode_request_digest": stable_hash(requests),
        "unique_symbol_count": len({row["symbol"] for row in requests}),
        "processes": PROCESSES,
        "partition_request_counts": [len(group) for group in groups],
        "partition_request_digests": [stable_hash(group) for group in groups],
        "cache": str((repository / CACHE).resolve()),
        "output": str((repository / OUTPUT).resolve()),
        "retrieval_opened": True,
        "complete_real_forward_outcomes_accessed": False,
        "production_promotion_authorized": False,
    }
    return {**state, "preregistration_digest": stable_hash(state)}


def _sole_child(repository: Path, prereg_raw: bytes, h0: str) -> str:
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
            if _git(repository, "show", f"{child}:{PREREGISTRATION}", raw=True) == prereg_raw:
                accepted.append(child)
    if len(set(accepted)) != 1:
        raise FullOutcomeError("expected one exact full-store preregistration-only child")
    return accepted[0]


def _validate_preregistration(
    repository: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[tuple[dict[str, Any], ...]], str]:
    if _git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise FullOutcomeError("globally clean Git worktree required")
    prereg, prereg_raw = _read(repository / PREREGISTRATION)
    state = {key: value for key, value in prereg.items() if key != "preregistration_digest"}
    if prereg.get("schema_version") != PREREG_SCHEMA \
            or prereg.get("preregistration_digest") != stable_hash(state):
        raise FullOutcomeError("full-store preregistration seal differs")
    h0 = str(prereg.get("implementation_h0"))
    h1 = _sole_child(repository, prereg_raw, h0)
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", h1, "HEAD"], cwd=repository,
    ).returncode:
        raise FullOutcomeError("HEAD does not descend from full-store preregistration")
    for name, expected in prereg.get("runtime_files", {}).items():
        if sha256(_git(repository, "show", f"{h0}:{name}", raw=True)).hexdigest() != expected \
                or sha256(_git(repository, "show", f"{h1}:{name}", raw=True)).hexdigest() != expected \
                or _sha(repository / name) != expected:
            raise FullOutcomeError(f"runtime source drifted: {name}")
    contract, contract_raw = _read(repository / smoke.CONTRACT)
    smoke_verified, smoke_verified_raw = _read(repository / SMOKE_VERIFICATION)
    if not all((
        prereg.get("contract_digest") == contract.get("contract_digest"),
        prereg.get("contract_sha256") == sha256(contract_raw).hexdigest(),
        prereg.get("smoke_verification_result_digest") == smoke_verified.get("result_digest"),
        prereg.get("smoke_verification_sha256") == sha256(smoke_verified_raw).hexdigest(),
        _receipt_valid(smoke_verified), smoke_verified.get("full_run_authorized") is True,
        prereg.get("complete_real_forward_outcomes_accessed") is False,
    )):
        raise FullOutcomeError("full-store prerequisites drifted")
    links, registry = _full_links(repository)
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
        prereg.get("unique_symbol_count") != len({row["symbol"] for row in requests}),
        prereg.get("processes") != PROCESSES,
        prereg.get("partition_request_counts") != [len(group) for group in groups],
        prereg.get("partition_request_digests") != [stable_hash(group) for group in groups],
        prereg.get("cache") != str((repository / CACHE).resolve()),
        prereg.get("output") != str((repository / OUTPUT).resolve()),
    )):
        raise FullOutcomeError("full-store frozen inventory differs")
    return prereg, contract, links, groups, h1


def _cast_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values(["episode_id", "horizon_sessions"], kind="stable").reset_index(drop=True)
    result["horizon_sessions"] = result["horizon_sessions"].astype("int64")
    result["available_sessions"] = result["available_sessions"].astype("int64")
    result["complete"] = result["complete"].astype("bool")
    for column in ("time_to_mfe", "time_to_mae", "barrier_touch_offset"):
        result[column] = result[column].astype("Int64")
    return result


def _cast_paths(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values(["episode_id", "step"], kind="stable").reset_index(drop=True)
    result["step"] = result["step"].astype("int64")
    result["expected_session_match"] = result["expected_session_match"].astype("bool")
    return result


def _frame_digest(
    frame: pd.DataFrame, order: Sequence[str], *, presorted: bool = False,
) -> str:
    ordered = frame.reset_index(drop=True) if presorted else frame.sort_values(
        list(order), kind="stable",
    ).reset_index(drop=True)
    digest = sha256()
    digest.update(f"{SEMANTIC_DIGEST_SCHEMA}\0{len(ordered)}\0".encode())
    for start in range(0, len(ordered), SEMANTIC_CHUNK_ROWS):
        records = smoke._plain(
            ordered.iloc[start:start + SEMANTIC_CHUNK_ROWS].to_dict("records")
        )
        payload = json.dumps(
            records, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _file_manifest(root: Path, names: Sequence[str]) -> list[dict[str, Any]]:
    return [{
        "path": name, "bytes": (root / name).stat().st_size,
        "sha256": _sha(root / name),
    } for name in names]


def _compute_partition(
    repository: str, index: int, requests: tuple[dict[str, Any], ...],
    cache_root: str, prereg_digest: str, h1: str,
    contract_digest: str, source_content_digest: str,
) -> dict[str, Any]:
    root = Path(repository)
    cache = Path(cache_root)
    final = cache / f"part-{index:02d}"
    if final.exists() or final.is_symlink():
        raise FullOutcomeError(f"partition target already exists: {index}")
    temporary = Path(tempfile.mkdtemp(
        prefix=f".t14-09-full-part-{index:02d}-", dir=cache.parent,
    ))
    started = perf_counter()
    try:
        config = load_config(root / retrieval.CONFIG)
        source = source_from_spec(config.datasets["nasdaq"])
        benchmark = source.load_benchmark()
        registry, _ = _read(root / retrieval.REGISTRY / "query-registry.json")
        if benchmark is None \
                or source.benchmark_fingerprint() != registry["source_lock"]["benchmark_sha256"]:
            raise FullOutcomeError("benchmark drifted from full-store source lock")
        if benchmark.attrs.get("source_timestamp_reordered") \
                or benchmark.attrs.get("source_duplicate_timestamps"):
            raise FullOutcomeError("benchmark session order is non-canonical")
        prepared_benchmark = prepare_outcome_sessions(benchmark, "benchmark")
        outcome_rows: list[dict[str, Any]] = []
        path_rows: list[dict[str, Any]] = []
        by_symbol: dict[str, list[dict[str, Any]]] = {}
        for request in requests:
            by_symbol.setdefault(request["symbol"], []).append(request)
        for symbol in sorted(by_symbol):
            key = InstrumentKey("nasdaq", symbol)
            stock = source.load(key)
            if stock.attrs.get("source_timestamp_reordered") \
                    or stock.attrs.get("source_duplicate_timestamps"):
                raise FullOutcomeError(f"non-canonical source order: {symbol}")
            prepared_stock = prepare_outcome_sessions(stock, f"stock:{symbol}")
            fingerprint = source.fingerprint(key)
            expected = {row["expected_source_fingerprint"] for row in by_symbol[symbol]}
            if expected != {fingerprint}:
                raise FullOutcomeError(f"stock source drifted from lock: {symbol}")
            timestamps = set(pd.to_datetime(stock["timestamp"]))
            for request in by_symbol[symbol]:
                cutoff = pd.Timestamp(request["cutoff"])
                if cutoff not in timestamps \
                        or EpisodeKey(key, cutoff, 252, "dense-v1").id != request["episode_id"]:
                    raise FullOutcomeError(f"episode identity differs: {request['episode_id']}")
                bundle = compute_prepared_episode_outcomes(
                    prepared_stock, prepared_benchmark,
                    episode_id=request["episode_id"], cutoff=cutoff,
                    source_fingerprint=fingerprint, contract_digest=contract_digest,
                    source_content_digest=source_content_digest,
                )
                outcome_rows.extend(smoke._plain(bundle.outcomes.to_dict("records")))
                path_rows.extend(smoke._plain(bundle.paths.to_dict("records")))
        outcomes = _cast_outcomes(pd.DataFrame(outcome_rows))
        paths = _cast_paths(pd.DataFrame(path_rows))
        smoke._atomic_parquet(temporary / "episode-outcomes.parquet", outcomes)
        smoke._atomic_parquet(temporary / "future-paths.parquet", paths)
        state = {
            "schema_version": PARTITION_SCHEMA, "status": "sealed", "passed": True,
            "partition": index, "preregistration_h1": h1,
            "preregistration_digest": prereg_digest,
            "request_count": len(requests), "request_digest": stable_hash(requests),
            "symbol_count": len(by_symbol), "outcome_rows": len(outcomes),
            "path_rows": len(paths),
            "semantic_digest_schema": SEMANTIC_DIGEST_SCHEMA,
            "outcome_digest": _frame_digest(
                outcomes, ("episode_id", "horizon_sessions"), presorted=True,
            ),
            "path_digest": _frame_digest(paths, ("episode_id", "step"), presorted=True),
            "file_manifest": _file_manifest(
                temporary, ("episode-outcomes.parquet", "future-paths.parquet"),
            ),
            "elapsed_seconds": perf_counter() - started,
            "real_forward_outcomes_accessed": True,
        }
        deterministic = {k: v for k, v in state.items() if k != "elapsed_seconds"}
        receipt = {**state, "result_digest": stable_hash(deterministic), "created_at": _now()}
        smoke._atomic_json(temporary / "PARTITION.json", receipt)
        os.replace(temporary, final)
        return receipt
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _validate_partition(
    root: Path, index: int, requests: Sequence[Mapping[str, Any]],
    prereg: Mapping[str, Any], h1: str,
) -> dict[str, Any]:
    part = root / f"part-{index:02d}"
    if part.is_symlink() or not part.is_dir() \
            or {path.name for path in part.iterdir()} != {
                "episode-outcomes.parquet", "future-paths.parquet", "PARTITION.json",
            } or any(path.is_symlink() for path in part.iterdir()):
        raise FullOutcomeError(f"partition inventory differs: {index}")
    receipt, _ = _read(part / "PARTITION.json")
    deterministic = {
        k: v for k, v in receipt.items()
        if k not in {"elapsed_seconds", "result_digest", "created_at"}
    }
    expected_files = ("episode-outcomes.parquet", "future-paths.parquet")
    if not all((
        receipt.get("schema_version") == PARTITION_SCHEMA,
        receipt.get("passed") is True,
        receipt.get("partition") == index,
        receipt.get("preregistration_h1") == h1,
        receipt.get("preregistration_digest") == prereg["preregistration_digest"],
        receipt.get("request_count") == len(requests),
        receipt.get("request_digest") == stable_hash(requests),
        receipt.get("outcome_rows") == len(requests) * len(HORIZONS),
        receipt.get("real_forward_outcomes_accessed") is True,
        receipt.get("result_digest") == stable_hash(deterministic),
        receipt.get("file_manifest") == _file_manifest(part, expected_files),
    )):
        raise FullOutcomeError(f"partition seal differs: {index}")
    return receipt


def _start_or_resume_cache(
    repository: Path, prereg: Mapping[str, Any], h1: str,
) -> Path:
    root = repository / CACHE
    expected = {
        "schema_version": SCHEMA, "status": "partitions_running",
        "preregistration_h1": h1,
        "preregistration_digest": prereg["preregistration_digest"],
        "partition_count": PROCESSES,
        "complete_real_forward_outcomes_accessed": False,
    }
    if not root.exists():
        root.mkdir(parents=True)
        smoke._atomic_json(root / "RUN_STARTED.json", {**expected, "created_at": _now()})
    if root.is_symlink() or not root.is_dir():
        raise FullOutcomeError("partition cache root is not a regular directory")
    started, _ = _read(root / "RUN_STARTED.json")
    if {k: v for k, v in started.items() if k != "created_at"} != expected:
        raise FullOutcomeError("partition cache start receipt differs")
    allowed = {"RUN_STARTED.json", "CACHE_SEALED.json"} | {
        f"part-{index:02d}" for index in range(PROCESSES)
    }
    if any(path.name not in allowed or path.is_symlink() for path in root.iterdir()):
        raise FullOutcomeError("partition cache contains an unexpected entry")
    return root


def _seal_cache(
    root: Path, receipts: Sequence[Mapping[str, Any]],
    prereg: Mapping[str, Any], h1: str,
) -> dict[str, Any]:
    state = {
        "schema_version": SCHEMA, "status": "partitions_sealed", "passed": True,
        "preregistration_h1": h1,
        "preregistration_digest": prereg["preregistration_digest"],
        "partition_count": PROCESSES,
        "partition_result_digests": [row["result_digest"] for row in receipts],
        "unique_episodes": sum(int(row["request_count"]) for row in receipts),
        "outcome_rows": sum(int(row["outcome_rows"]) for row in receipts),
        "path_rows": sum(int(row["path_rows"]) for row in receipts),
        "complete_real_forward_outcomes_accessed": True,
    }
    value = {**state, "result_digest": stable_hash(state), "created_at": _now()}
    path = root / "CACHE_SEALED.json"
    if path.exists():
        observed, _ = _read(path)
        if {k: v for k, v in observed.items() if k != "created_at"} \
                != {k: v for k, v in value.items() if k != "created_at"}:
            raise FullOutcomeError("existing cache seal differs")
        return observed
    smoke._atomic_json(path, value)
    return value


def _aggregate(
    repository: Path, cache: Path, prereg: Mapping[str, Any], contract: Mapping[str, Any],
    links: Sequence[Mapping[str, Any]], receipts: Sequence[Mapping[str, Any]],
    cache_seal: Mapping[str, Any], h1: str, elapsed_seconds: float,
) -> dict[str, Any]:
    final = repository / OUTPUT
    if final.exists() or final.is_symlink():
        raise FullOutcomeError("full outcome output already exists")
    temporary = Path(tempfile.mkdtemp(prefix=".t14-09-full-store-", dir=final.parent))
    try:
        run_started, _ = _read(cache / "RUN_STARTED.json")
        smoke._atomic_json(temporary / "RUN_STARTED.json", run_started)
        outcomes = _cast_outcomes(pd.concat([
            pd.read_parquet(cache / f"part-{index:02d}" / "episode-outcomes.parquet")
            for index in range(PROCESSES)
        ], ignore_index=True))
        paths = _cast_paths(pd.concat([
            pd.read_parquet(cache / f"part-{index:02d}" / "future-paths.parquet")
            for index in range(PROCESSES)
        ], ignore_index=True))
        if len(outcomes) != prereg["unique_episode_count"] * len(HORIZONS) \
                or outcomes[["episode_id", "horizon_sessions"]].duplicated().any() \
                or paths[["episode_id", "step"]].duplicated().any():
            raise FullOutcomeError("aggregate episode/path identity differs")
        outcome_by_key = {
            (str(row.episode_id), int(row.horizon_sessions)): (
                row.completion_timestamp, bool(row.complete), str(row.source_fingerprint),
            ) for row in outcomes.itertuples(index=False)
        }
        enriched: list[dict[str, Any]] = []
        for link in links:
            eligibility: dict[str, dict[str, Any]] = {}
            fingerprint: str | None = None
            for horizon in HORIZONS:
                completion, complete, observed_fingerprint = outcome_by_key[
                    (link["matched_episode_id"], horizon)
                ]
                fingerprint = observed_fingerprint
                eligible, reason = outcome_embargo(
                    completion, link["query_cutoff"], complete=complete,
                )
                eligibility[str(horizon)] = {"eligible": eligible, "reason": reason}
            enriched.append({
                **link, "source_fingerprint": fingerprint,
                "outcome_eligibility_json": json.dumps(
                    eligibility, sort_keys=True, separators=(",", ":"),
                ),
            })
        link_frame = pd.DataFrame(enriched).sort_values(
            ["query_episode_id", "match_rank"], kind="stable",
        ).reset_index(drop=True)
        link_frame["match_rank"] = link_frame["match_rank"].astype("int64")
        smoke._atomic_parquet(temporary / "episode-outcomes.parquet", outcomes)
        smoke._atomic_parquet(temporary / "future-paths.parquet", paths)
        smoke._atomic_parquet(temporary / "query-match-links.parquet", link_frame)
        coverage_state = {
            "schema_version": SCHEMA, "status": "complete",
            "query_links": len(link_frame),
            "unique_episodes": int(prereg["unique_episode_count"]),
            "unique_symbols": int(prereg["unique_symbol_count"]),
            "outcome_rows": len(outcomes), "path_rows": len(paths),
            "complete_by_horizon": {
                str(horizon): int(outcomes.loc[
                    outcomes.horizon_sessions == horizon, "complete",
                ].sum()) for horizon in HORIZONS
            },
            "status_counts": {
                str(key): int(value)
                for key, value in outcomes.status.value_counts().sort_index().items()
            },
            "barrier_counts": {
                str(key): int(value) for key, value in outcomes.loc[
                    outcomes.horizon_sessions == 20, "barrier_label",
                ].value_counts(dropna=False).sort_index().items()
            },
            "complete_real_forward_outcomes_accessed": True,
        }
        coverage = {**coverage_state, "result_digest": stable_hash(coverage_state)}
        smoke._atomic_json(temporary / "COVERAGE.json", coverage)
        names = (
            "episode-outcomes.parquet", "future-paths.parquet",
            "query-match-links.parquet", "COVERAGE.json",
        )
        state = {
            "schema_version": SCHEMA, "status": "sealed", "passed": True,
            "preregistration_h1": h1,
            "preregistration_digest": prereg["preregistration_digest"],
            "contract_digest": contract["contract_digest"],
            "source_content_digest": prereg["source_content_digest"],
            "partition_cache_result_digest": cache_seal["result_digest"],
            "query_count": 3270, "query_links": len(link_frame),
            "unique_episodes": int(prereg["unique_episode_count"]),
            "unique_symbols": int(prereg["unique_symbol_count"]),
            "outcome_rows": len(outcomes), "path_rows": len(paths),
            "semantic_digests": {
                "semantic_digest_schema": SEMANTIC_DIGEST_SCHEMA,
                "outcome_digest": _frame_digest(
                    outcomes, ("episode_id", "horizon_sessions"), presorted=True,
                ),
                "path_digest": _frame_digest(
                    paths, ("episode_id", "step"), presorted=True,
                ),
                "link_digest": _frame_digest(
                    link_frame, ("query_episode_id", "match_rank"), presorted=True,
                ),
                "coverage_result_digest": coverage["result_digest"],
            },
            "file_manifest": _file_manifest(temporary, names),
            "partition_elapsed_seconds": elapsed_seconds,
            "complete_real_forward_outcomes_accessed": True,
            "outcomes_affected_retrieval": False,
            "production_promotion_authorized": False,
        }
        deterministic = {
            k: v for k, v in state.items() if k != "partition_elapsed_seconds"
        }
        seal = {**state, "result_digest": stable_hash(deterministic), "created_at": _now()}
        smoke._atomic_json(temporary / "SEALED.json", seal)
        os.replace(temporary, final)
        return seal
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    prereg, contract, links, groups, h1 = _validate_preregistration(repository)
    if os.statvfs(repository).f_bavail * os.statvfs(repository).f_frsize < 5 * 1024 ** 3:
        raise FullOutcomeError("less than 5 GiB free before complete outcome store")
    if (repository / OUTPUT).exists() or (repository / OUTPUT).is_symlink():
        raise FullOutcomeError("complete outcome store is create-only")
    cache = _start_or_resume_cache(repository, prereg, h1)
    started = perf_counter()
    receipts: list[dict[str, Any] | None] = [None] * PROCESSES
    missing: list[int] = []
    for index, requests in enumerate(groups):
        if (cache / f"part-{index:02d}").exists():
            receipts[index] = _validate_partition(cache, index, requests, prereg, h1)
        else:
            missing.append(index)
    if (cache / "CACHE_SEALED.json").exists() and missing:
        raise FullOutcomeError("sealed cache is missing a partition")
    if missing:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=PROCESSES, mp_context=context) as pool:
            futures = {
                pool.submit(
                    _compute_partition, str(repository), index, groups[index], str(cache),
                    prereg["preregistration_digest"], h1, contract["contract_digest"],
                    prereg["source_content_digest"],
                ): index for index in missing
            }
            for future in as_completed(futures):
                index = futures[future]
                future.result()
                receipts[index] = _validate_partition(cache, index, groups[index], prereg, h1)
    complete = [row for row in receipts if row is not None]
    if len(complete) != PROCESSES:
        raise FullOutcomeError("not all partitions completed")
    cache_seal = _seal_cache(cache, complete, prereg, h1)
    return _aggregate(
        repository, cache, prereg, contract, links, complete, cache_seal, h1,
        perf_counter() - started,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--build-preregistration", action="store_true")
    args = parser.parse_args(argv)
    repository = args.repository.resolve(strict=True)
    value = build_preregistration(repository) if args.build_preregistration else execute(repository)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
