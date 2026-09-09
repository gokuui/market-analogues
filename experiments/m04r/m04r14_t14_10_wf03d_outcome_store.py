"""Preregister and build the causal WF-03D analogue-outcome store."""
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

import numpy as np
import pandas as pd

from market_analogues.adapters import source_from_spec
from market_analogues.causal_outcomes import (
    compute_prepared_episode_outcomes, prepare_outcome_sessions,
)
from market_analogues.config import load_config
from market_analogues.types import EpisodeKey, InstrumentKey, stable_hash

from experiments.m04r import m04r14_t14_09_full_outcome_store as old_store
from experiments.m04r import m04r14_t14_09_outcome_smoke as smoke
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_10_wf03d_cross_store_manifest as cross_store
from experiments.m04r import verify_m04r14_t14_09_full_outcome_store as old_verifier
from experiments.m04r import verify_m04r14_t14_10_wf03d_cross_store_manifest as cross_verifier


SCHEMA = "m04r14-t14-10-wf03d-outcome-store-preregistration-v1"
PARTITION_SCHEMA = "m04r14-t14-10-wf03d-outcome-partition-v1"
STORE_SCHEMA = "m04r14-t14-10-wf03d-outcome-store-v1"
PREREGISTRATION_RELATIVE = Path(
    "experiments/m04r/m04r14_t14_10_wf03d_outcome_store_v1_preregistered.json"
)
CACHE_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf03d-outcome-partitions-v1"
)
OUTPUT_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf03d-outcome-store-v1"
)
VERIFICATION_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf03d-outcome-store-v1-verification"
)
OLD_VERIFICATION = old_store.VERIFICATION / "VERIFIED.json"
CONTRACT = smoke.CONTRACT
SOURCE_ACCOUNTING = base.REGISTRY_RELATIVE / "source-accounting.parquet"
HORIZONS = old_store.HORIZONS
PROCESSES = 12
PARTITIONS = 48
EXPECTED_REQUESTS = 274_331
EXPECTED_REUSE = 7_955
EXPECTED_MISSING = 266_376
EXPECTED_LINKS = cross_store.EXPECTED_LINKS
EXPECTED_ELIGIBILITY_ROWS = EXPECTED_LINKS * len(HORIZONS)
RUNTIME_FILES = (
    "experiments/m04r/m04r14_t14_10_wf03d_outcome_store.py",
    "experiments/m04r/verify_m04r14_t14_10_wf03d_outcome_store.py",
    "experiments/m04r/m04r14_t14_10_wf03d_cross_store_manifest.py",
    "experiments/m04r/verify_m04r14_t14_10_wf03d_cross_store_manifest.py",
    "experiments/m04r/m04r14_t14_09_full_outcome_store.py",
    "experiments/m04r/verify_m04r14_t14_09_full_outcome_store.py",
    "experiments/m04r/m04r14_t14_09_outcome_oracle.py",
    "experiments/m04r/m04r14_t14_09_outcome_smoke.py",
    "src/market_analogues/causal_outcomes.py",
    "src/market_analogues/adapters.py",
    "src/market_analogues/types.py",
    "config/m04r14-t14-09-outcome-contract.json",
    "config/datasets.example.yaml",
    "pyproject.toml",
)


class WalkForwardOutcomeError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(repository: Path, *args: str, raw: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=repository, capture_output=True,
        text=not raw, check=False,
    )
    if result.returncode:
        message = result.stderr if not raw else result.stderr.decode(errors="replace")
        raise WalkForwardOutcomeError(message.strip() or "git command failed")
    return result.stdout if raw else result.stdout.strip()


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise WalkForwardOutcomeError(f"regular file required: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _runtime_manifest(repository: Path, head: str) -> dict[str, str]:
    tracked = set(str(_git(
        repository, "ls-tree", "-r", "--name-only", head,
    )).splitlines())
    if any(name not in tracked for name in RUNTIME_FILES):
        raise WalkForwardOutcomeError("runtime manifest contains uncommitted files")
    return {
        name: sha256(_git(repository, "show", f"{head}:{name}", raw=True)).hexdigest()
        for name in RUNTIME_FILES
    }


def _receipt_valid(
    value: Mapping[str, Any], *, timing: bool = False, digest_key: str = "result_digest",
) -> bool:
    omitted = {digest_key, "created_at"}
    if timing:
        omitted |= {"elapsed_seconds", "partition_elapsed_seconds"}
    return value.get(digest_key) == stable_hash({
        key: item for key, item in value.items() if key not in omitted
    })


def _cross_store_inputs(repository: Path) -> dict[str, Any]:
    root = repository / cross_store.OUTPUT_RELATIVE
    result = base._read(root / "RESULT.json")
    manifest = base._read(root / "MANIFEST.json")
    verified_path = repository / cross_verifier.OUTPUT_RELATIVE / "VERIFIED.json"
    verified = base._read(verified_path)
    base._validate_seal(result)
    base._validate_seal(manifest, "manifest_digest")
    base._validate_seal(verified, "verification_digest")
    if not all((
        result.get("passed") is True,
        result.get("link_count") == EXPECTED_LINKS,
        result.get("unique_episode_count") == EXPECTED_REQUESTS,
        verified.get("passed") is True,
        verified.get("producer_result_digest") == result.get("result_digest"),
        verified.get("manifest_digest") == manifest.get("manifest_digest"),
        verified.get("walk_forward_outcome_store_construction_authorized") is True,
        verified.get("outcomes_or_labels_used") is False,
    )):
        raise WalkForwardOutcomeError("verified cross-store input differs")
    return {
        "result_digest": result["result_digest"],
        "result_sha256": _sha(root / "RESULT.json"),
        "manifest_digest": manifest["manifest_digest"],
        "manifest_sha256": _sha(root / "MANIFEST.json"),
        "verification_digest": verified["verification_digest"],
        "verification_sha256": _sha(verified_path),
        "raw_link_sha256": _sha(root / cross_store.LINK_FILE),
        "request_sha256": _sha(root / cross_store.REQUEST_FILE),
        "raw_link_semantic_digest": result["raw_link_semantic_digest"],
        "request_semantic_digest": result["episode_request_semantic_digest"],
    }


def _old_store_inputs(repository: Path) -> dict[str, Any]:
    root = repository / old_store.OUTPUT
    seal = base._read(root / "SEALED.json")
    verified_path = repository / OLD_VERIFICATION
    verified = base._read(verified_path)
    if not all((
        _receipt_valid(seal, timing=True), seal.get("passed") is True,
        seal.get("contract_digest") == base._read(repository / CONTRACT).get("contract_digest"),
        seal.get("source_content_digest") == base._resident()["content_digest"],
        _receipt_valid(verified), verified.get("passed") is True,
        verified.get("store_result_digest") == seal.get("result_digest"),
        verified.get("store_result_sha256") == _sha(root / "SEALED.json"),
        verified.get("outcomes_affected_retrieval") is False,
    )):
        raise WalkForwardOutcomeError("verified reusable outcome store differs")
    return {
        "seal_result_digest": seal["result_digest"],
        "seal_sha256": _sha(root / "SEALED.json"),
        "verification_result_digest": verified["result_digest"],
        "verification_sha256": _sha(verified_path),
        "outcomes_sha256": _sha(root / "episode-outcomes.parquet"),
        "paths_sha256": _sha(root / "future-paths.parquet"),
    }


def _outcome_contract_inputs(repository: Path) -> dict[str, Any]:
    contract_receipt = base._read(repository / smoke.CONTRACT_RECEIPT)
    synthetic = base._read(repository / smoke.SYNTHETIC)
    contract = base._read(repository / CONTRACT)
    if not all((
        _receipt_valid(contract_receipt), contract_receipt.get("passed") is True,
        contract_receipt.get("contract_digest") == contract.get("contract_digest"),
        contract_receipt.get("real_forward_outcomes_accessed") is False,
        _receipt_valid(synthetic, timing=True), synthetic.get("passed") is True,
        synthetic.get("contract_digest") == contract.get("contract_digest"),
        synthetic.get("real_forward_outcomes_accessed") is False,
    )):
        raise WalkForwardOutcomeError("verified outcome contract/synthetic gate differs")
    return {
        "contract_verification_result_digest": contract_receipt["result_digest"],
        "contract_verification_sha256": _sha(repository / smoke.CONTRACT_RECEIPT),
        "synthetic_result_digest": synthetic["result_digest"],
        "synthetic_sha256": _sha(repository / smoke.SYNTHETIC),
    }


def _requests(repository: Path) -> list[dict[str, Any]]:
    cross_root = repository / cross_store.OUTPUT_RELATIVE
    requests = pd.read_parquet(cross_root / cross_store.REQUEST_FILE, engine="pyarrow")
    if tuple(requests.columns) != cross_store.REQUEST_COLUMNS \
            or len(requests) != EXPECTED_REQUESTS \
            or cross_store._semantic_digest(cross_store._plain(requests.to_dict("records"))) \
                != base._read(cross_root / "RESULT.json")["episode_request_semantic_digest"]:
        raise WalkForwardOutcomeError("cross-store episode requests differ")
    registry, _ = base._registry(repository)
    accounting_path = repository / SOURCE_ACCOUNTING
    accounting = pd.read_parquet(accounting_path, engine="pyarrow")
    if stable_hash(cross_store._plain(accounting.to_dict("records"))) \
            != registry.get("source_accounting_digest") \
            or not {"symbol", "source_hash_at_lock", "error"}.issubset(accounting.columns) \
            or accounting.symbol.astype(str).duplicated().any():
        raise WalkForwardOutcomeError("source-accounting identity differs")
    fingerprints = {
        str(row.symbol): str(row.source_hash_at_lock)
        for row in accounting.itertuples(index=False) if row.error is None
    }
    result = []
    for row in requests.itertuples(index=False):
        fingerprint = fingerprints.get(str(row.symbol))
        if not fingerprint:
            raise WalkForwardOutcomeError(f"request lacks locked source: {row.symbol}")
        result.append({
            "episode_id": str(row.episode_id), "dataset_id": str(row.dataset_id),
            "symbol": str(row.symbol), "cutoff": str(row.cutoff),
            "quality_tier": str(row.quality_tier),
            "expected_source_fingerprint": fingerprint,
        })
    return result


def _reusable_identity_matches(
    request: Mapping[str, Any], observed: Any,
    contract_digest: str, source_content_digest: str,
) -> bool:
    def field(name: str) -> Any:
        return observed.get(name) if isinstance(observed, Mapping) else getattr(observed, name)

    return (
        str(field("cutoff")), str(field("source_fingerprint")),
        str(field("contract_digest")), str(field("source_content_digest")),
    ) == (
        str(request["cutoff"]), str(request["expected_source_fingerprint"]),
        contract_digest, source_content_digest,
    )


def _reuse_split(
    repository: Path, requests: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    contract = base._read(repository / CONTRACT)
    source_content_digest = base._resident()["content_digest"]
    identity = pd.read_parquet(
        repository / old_store.OUTPUT / "episode-outcomes.parquet",
        columns=[
            "episode_id", "cutoff", "source_fingerprint", "contract_digest",
            "source_content_digest",
        ], engine="pyarrow",
    ).drop_duplicates()
    if identity.episode_id.astype(str).duplicated().any():
        raise WalkForwardOutcomeError("reusable outcome identity conflicts")
    old = {str(row.episode_id): row for row in identity.itertuples(index=False)}
    reuse: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for request in requests:
        observed = old.get(str(request["episode_id"]))
        target = reuse if observed is not None and _reusable_identity_matches(
            request, observed, contract["contract_digest"], source_content_digest,
        ) else missing
        target.append(dict(request))
    if len(reuse) != EXPECTED_REUSE or len(missing) != EXPECTED_MISSING:
        raise WalkForwardOutcomeError("reuse/missing inventory differs")
    return reuse, missing


def _groups(requests: Sequence[Mapping[str, Any]]) -> list[tuple[dict[str, Any], ...]]:
    groups: list[list[dict[str, Any]]] = [[] for _ in range(PARTITIONS)]
    for raw in requests:
        row = dict(raw)
        index = int(sha256(row["symbol"].encode()).hexdigest(), 16) % PARTITIONS
        groups[index].append(row)
    if any(not group for group in groups):
        raise WalkForwardOutcomeError("every outcome partition must be nonempty")
    return [tuple(sorted(group, key=lambda row: (
        row["symbol"], row["cutoff"], row["episode_id"],
    ))) for group in groups]


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise WalkForwardOutcomeError("globally clean Git worktree required")
    if (repository / CACHE_RELATIVE).exists() or (repository / OUTPUT_RELATIVE).exists():
        raise WalkForwardOutcomeError("outcome cache/output must be absent")
    h0 = str(_git(repository, "rev-parse", "HEAD"))
    contract = base._read(repository / CONTRACT)
    requests = _requests(repository)
    reuse, missing = _reuse_split(repository, requests)
    groups = _groups(missing)
    source = source_from_spec(
        load_config(repository / base.CONFIG_RELATIVE).datasets["nasdaq"]
    )
    benchmark_fingerprint = source.benchmark_fingerprint()
    state = {
        "schema_version": SCHEMA,
        "status": "frozen_before_walk_forward_analogue_outcomes",
        "implementation_h0": h0,
        "runtime_files": _runtime_manifest(repository, h0),
        "cross_store": _cross_store_inputs(repository),
        "reusable_store": _old_store_inputs(repository),
        "outcome_contract_evidence": _outcome_contract_inputs(repository),
        "contract_digest": contract["contract_digest"],
        "contract_sha256": _sha(repository / CONTRACT),
        "source_content_digest": base._resident()["content_digest"],
        "source_accounting_sha256": _sha(repository / SOURCE_ACCOUNTING),
        "source_accounting_digest": base._registry(repository)[0]["source_accounting_digest"],
        "benchmark_fingerprint": benchmark_fingerprint,
        "request_count": len(requests), "request_digest": stable_hash(requests),
        "reuse_count": len(reuse), "reuse_request_digest": stable_hash(reuse),
        "missing_count": len(missing), "missing_request_digest": stable_hash(missing),
        "partition_count": PARTITIONS, "worker_processes": PROCESSES,
        "partition_request_counts": [len(group) for group in groups],
        "partition_request_digests": [stable_hash(group) for group in groups],
        "expected_outcome_rows": len(requests) * len(HORIZONS),
        "expected_link_eligibility_rows": EXPECTED_ELIGIBILITY_ROWS,
        "horizons": list(HORIZONS),
        "cache": str((repository / CACHE_RELATIVE).resolve()),
        "output": str((repository / OUTPUT_RELATIVE).resolve()),
        "historical_analogue_outcomes_accessed": False,
        "historical_query_evaluation_opened": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
    }
    return base._sealed(state, "preregistration_digest")


def _sole_child(repository: Path, raw: bytes, h0: str) -> str:
    accepted: list[str] = []
    for line in str(_git(repository, "rev-list", "--all", "--children")).splitlines():
        values = line.split()
        if not values or values[0] != h0:
            continue
        for child in values[1:]:
            parents = str(_git(repository, "rev-list", "--parents", "-n", "1", child)).split()
            changed = str(_git(
                repository, "diff-tree", "--no-commit-id", "--name-only", "-r", child,
            )).splitlines()
            if parents == [child, h0] and changed == [PREREGISTRATION_RELATIVE.as_posix()] \
                    and _git(repository, "show", f"{child}:{PREREGISTRATION_RELATIVE}", raw=True) == raw:
                accepted.append(child)
    if len(set(accepted)) != 1:
        raise WalkForwardOutcomeError("expected one exact preregistration-only child")
    return accepted[0]


def validate_preregistration(
    repository: Path,
) -> tuple[
    dict[str, Any], dict[str, Any], list[dict[str, Any]],
    list[dict[str, Any]], list[tuple[dict[str, Any], ...]], str,
]:
    if _git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise WalkForwardOutcomeError("globally clean Git worktree required")
    path = repository / PREREGISTRATION_RELATIVE
    raw = path.read_bytes()
    prereg = base._read(path)
    base._validate_seal(prereg, "preregistration_digest")
    if prereg.get("schema_version") != SCHEMA:
        raise WalkForwardOutcomeError("outcome preregistration schema differs")
    h0 = str(prereg.get("implementation_h0"))
    h1 = _sole_child(repository, raw, h0)
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", h1, "HEAD"], cwd=repository,
    ).returncode:
        raise WalkForwardOutcomeError("HEAD does not descend from preregistration")
    for name, expected in prereg.get("runtime_files", {}).items():
        if _sha(repository / name) != expected \
                or sha256(_git(repository, "show", f"{h0}:{name}", raw=True)).hexdigest() != expected \
                or sha256(_git(repository, "show", f"{h1}:{name}", raw=True)).hexdigest() != expected:
            raise WalkForwardOutcomeError(f"runtime source drifted: {name}")
    contract = base._read(repository / CONTRACT)
    requests = _requests(repository)
    reuse, missing = _reuse_split(repository, requests)
    groups = _groups(missing)
    source = source_from_spec(
        load_config(repository / base.CONFIG_RELATIVE).datasets["nasdaq"]
    )
    if any((
        prereg.get("cross_store") != _cross_store_inputs(repository),
        prereg.get("reusable_store") != _old_store_inputs(repository),
        prereg.get("outcome_contract_evidence")
            != _outcome_contract_inputs(repository),
        prereg.get("contract_digest") != contract.get("contract_digest"),
        prereg.get("contract_sha256") != _sha(repository / CONTRACT),
        prereg.get("source_content_digest") != base._resident()["content_digest"],
        prereg.get("source_accounting_sha256") != _sha(repository / SOURCE_ACCOUNTING),
        prereg.get("source_accounting_digest")
            != base._registry(repository)[0]["source_accounting_digest"],
        prereg.get("benchmark_fingerprint") != source.benchmark_fingerprint(),
        prereg.get("request_count") != len(requests),
        prereg.get("request_digest") != stable_hash(requests),
        prereg.get("reuse_count") != len(reuse),
        prereg.get("reuse_request_digest") != stable_hash(reuse),
        prereg.get("missing_count") != len(missing),
        prereg.get("missing_request_digest") != stable_hash(missing),
        prereg.get("partition_request_counts") != [len(group) for group in groups],
        prereg.get("partition_request_digests") != [stable_hash(group) for group in groups],
        prereg.get("historical_analogue_outcomes_accessed") is not False,
        prereg.get("historical_query_evaluation_opened") is not False,
        prereg.get("final_period_result_opened") is not False,
    )):
        raise WalkForwardOutcomeError("outcome preregistration inventory drifted")
    return prereg, contract, reuse, missing, groups, h1


def _compute_partition(
    repository_raw: str, index: int, requests: tuple[dict[str, Any], ...],
    cache_raw: str, preregistration_digest: str, h1: str,
    contract_digest: str, source_content_digest: str, benchmark_fingerprint: str,
) -> dict[str, Any]:
    repository = Path(repository_raw)
    cache = Path(cache_raw)
    final = cache / f"part-{index:02d}"
    if final.exists() or final.is_symlink():
        raise WalkForwardOutcomeError(f"partition target already exists: {index}")
    temporary = Path(tempfile.mkdtemp(
        prefix=f".wf03d-outcomes-{index:02d}-", dir=cache.parent,
    ))
    started = perf_counter()
    try:
        source = source_from_spec(
            load_config(repository / base.CONFIG_RELATIVE).datasets["nasdaq"]
        )
        benchmark = source.load_benchmark()
        if benchmark is None or source.benchmark_fingerprint() != benchmark_fingerprint \
                or benchmark.attrs.get("source_timestamp_reordered") \
                or benchmark.attrs.get("source_duplicate_timestamps"):
            raise WalkForwardOutcomeError("benchmark source drifted")
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
                raise WalkForwardOutcomeError(f"non-canonical stock source: {symbol}")
            fingerprint = source.fingerprint(key)
            if {row["expected_source_fingerprint"] for row in by_symbol[symbol]} \
                    != {fingerprint}:
                raise WalkForwardOutcomeError(f"stock fingerprint drifted: {symbol}")
            prepared_stock = prepare_outcome_sessions(stock, f"stock:{symbol}")
            timestamps = set(pd.to_datetime(stock["timestamp"]))
            for request in by_symbol[symbol]:
                cutoff = pd.Timestamp(request["cutoff"])
                if cutoff not in timestamps \
                        or EpisodeKey(key, cutoff, 252, "dense-v1").id \
                            != request["episode_id"]:
                    raise WalkForwardOutcomeError(
                        f"episode identity differs: {request['episode_id']}"
                    )
                bundle = compute_prepared_episode_outcomes(
                    prepared_stock, prepared_benchmark,
                    episode_id=request["episode_id"], cutoff=cutoff,
                    source_fingerprint=fingerprint,
                    contract_digest=contract_digest,
                    source_content_digest=source_content_digest,
                )
                outcome_rows.extend(smoke._plain(bundle.outcomes.to_dict("records")))
                path_rows.extend(smoke._plain(bundle.paths.to_dict("records")))
        outcomes = old_store._cast_outcomes(pd.DataFrame(outcome_rows))
        paths = old_store._cast_paths(pd.DataFrame(path_rows))
        smoke._atomic_parquet(temporary / "episode-outcomes.parquet", outcomes)
        smoke._atomic_parquet(temporary / "future-paths.parquet", paths)
        state = {
            "schema_version": PARTITION_SCHEMA, "status": "sealed", "passed": True,
            "partition": index, "preregistration_h1": h1,
            "preregistration_digest": preregistration_digest,
            "request_count": len(requests), "request_digest": stable_hash(requests),
            "symbol_count": len(by_symbol),
            "outcome_rows": len(outcomes), "path_rows": len(paths),
            "outcome_digest": old_store._frame_digest(
                outcomes, ("episode_id", "horizon_sessions"), presorted=True,
            ),
            "path_digest": old_store._frame_digest(
                paths, ("episode_id", "step"), presorted=True,
            ),
            "file_manifest": old_store._file_manifest(
                temporary, ("episode-outcomes.parquet", "future-paths.parquet"),
            ),
            "elapsed_seconds": perf_counter() - started,
            "historical_analogue_outcomes_accessed": True,
            "historical_query_evaluation_opened": False,
            "final_period_result_opened": False,
        }
        receipt = {
            **state, "result_digest": stable_hash({
                key: value for key, value in state.items() if key != "elapsed_seconds"
            }), "created_at": _now(),
        }
        smoke._atomic_json(temporary / "PARTITION.json", receipt)
        os.replace(temporary, final)
        return receipt
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _validate_partition(
    cache: Path, index: int, requests: Sequence[Mapping[str, Any]],
    prereg: Mapping[str, Any], h1: str,
) -> dict[str, Any]:
    root = cache / f"part-{index:02d}"
    if root.is_symlink() or not root.is_dir() \
            or {path.name for path in root.iterdir()} != {
                "episode-outcomes.parquet", "future-paths.parquet", "PARTITION.json",
            } or any(path.is_symlink() for path in root.iterdir()):
        raise WalkForwardOutcomeError(f"partition layout differs: {index}")
    receipt = base._read(root / "PARTITION.json")
    if not all((
        _receipt_valid(receipt, timing=True),
        receipt.get("schema_version") == PARTITION_SCHEMA,
        receipt.get("passed") is True,
        receipt.get("partition") == index,
        receipt.get("preregistration_h1") == h1,
        receipt.get("preregistration_digest") == prereg["preregistration_digest"],
        receipt.get("request_count") == len(requests),
        receipt.get("request_digest") == stable_hash(requests),
        receipt.get("outcome_rows") == len(requests) * len(HORIZONS),
        receipt.get("file_manifest") == old_store._file_manifest(
            root, ("episode-outcomes.parquet", "future-paths.parquet"),
        ),
        receipt.get("historical_query_evaluation_opened") is False,
        receipt.get("final_period_result_opened") is False,
    )):
        raise WalkForwardOutcomeError(f"partition seal differs: {index}")
    return receipt


def _start_cache(
    repository: Path, prereg: Mapping[str, Any], h1: str,
) -> Path:
    root = repository / CACHE_RELATIVE
    expected = {
        "schema_version": STORE_SCHEMA, "status": "partitions_running",
        "preregistration_h1": h1,
        "preregistration_digest": prereg["preregistration_digest"],
        "partition_count": PARTITIONS, "worker_processes": PROCESSES,
        "missing_episode_count": prereg["missing_count"],
        "historical_analogue_outcomes_accessed": False,
        "historical_query_evaluation_opened": False,
        "final_period_result_opened": False,
    }
    if not root.exists():
        root.mkdir(parents=True)
        smoke._atomic_json(root / "RUN_STARTED.json", {**expected, "created_at": _now()})
    if root.is_symlink() or not root.is_dir():
        raise WalkForwardOutcomeError("outcome cache is not a regular directory")
    started = base._read(root / "RUN_STARTED.json")
    if {key: value for key, value in started.items() if key != "created_at"} != expected:
        raise WalkForwardOutcomeError("outcome cache start receipt differs")
    allowed = {"RUN_STARTED.json", "CACHE_SEALED.json"} | {
        f"part-{index:02d}" for index in range(PARTITIONS)
    }
    if any(path.name not in allowed or path.is_symlink() for path in root.iterdir()):
        raise WalkForwardOutcomeError("outcome cache contains unexpected entries")
    return root


def _seal_cache(
    cache: Path, receipts: Sequence[Mapping[str, Any]],
    prereg: Mapping[str, Any], h1: str,
) -> dict[str, Any]:
    state = {
        "schema_version": STORE_SCHEMA, "status": "partitions_sealed", "passed": True,
        "preregistration_h1": h1,
        "preregistration_digest": prereg["preregistration_digest"],
        "partition_count": PARTITIONS,
        "partition_result_digests": [row["result_digest"] for row in receipts],
        "computed_episodes": sum(int(row["request_count"]) for row in receipts),
        "computed_outcome_rows": sum(int(row["outcome_rows"]) for row in receipts),
        "computed_path_rows": sum(int(row["path_rows"]) for row in receipts),
        "historical_analogue_outcomes_accessed": True,
        "historical_query_evaluation_opened": False,
        "final_period_result_opened": False,
    }
    value = {**state, "result_digest": stable_hash(state), "created_at": _now()}
    path = cache / "CACHE_SEALED.json"
    if path.exists():
        observed = base._read(path)
        if {key: item for key, item in observed.items() if key != "created_at"} \
                != {key: item for key, item in value.items() if key != "created_at"}:
            raise WalkForwardOutcomeError("outcome cache seal differs")
        return observed
    smoke._atomic_json(path, value)
    return value


def _reused_frames(
    repository: Path, reuse: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any], source_content_digest: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = {str(row["episode_id"]) for row in reuse}
    expected = {str(row["episode_id"]): row for row in reuse}
    old_root = repository / old_store.OUTPUT
    outcomes = pd.read_parquet(old_root / "episode-outcomes.parquet", engine="pyarrow")
    outcomes = outcomes.loc[outcomes.episode_id.astype(str).isin(ids)].copy()
    paths = pd.read_parquet(old_root / "future-paths.parquet", engine="pyarrow")
    paths = paths.loc[paths.episode_id.astype(str).isin(ids)].copy()
    if len(outcomes) != len(ids) * len(HORIZONS) \
            or outcomes[["episode_id", "horizon_sessions"]].duplicated().any() \
            or paths[["episode_id", "step"]].duplicated().any():
        raise WalkForwardOutcomeError("reused outcome/path identity differs")
    identities = outcomes[[
        "episode_id", "cutoff", "source_fingerprint", "contract_digest",
        "source_content_digest",
    ]].drop_duplicates()
    if len(identities) != len(ids):
        raise WalkForwardOutcomeError("reused outcome metadata conflicts")
    for row in identities.itertuples(index=False):
        request = expected[str(row.episode_id)]
        if not all((
            str(row.cutoff) == request["cutoff"],
            str(row.source_fingerprint) == request["expected_source_fingerprint"],
            str(row.contract_digest) == contract["contract_digest"],
            str(row.source_content_digest) == source_content_digest,
        )):
            raise WalkForwardOutcomeError("reused outcome binding differs")
    path_ids = set(paths.episode_id.astype(str))
    if not path_ids.issubset(ids):
        raise WalkForwardOutcomeError("reused path inventory differs")
    return old_store._cast_outcomes(outcomes), old_store._cast_paths(paths)


def _eligibility_table(
    links: pd.DataFrame, outcomes: pd.DataFrame,
    *, expected_links: int = EXPECTED_LINKS,
    expected_requests: int = EXPECTED_REQUESTS,
    horizons: Sequence[int] = HORIZONS,
) -> pd.DataFrame:
    required = {"query_id", "query_cutoff", "method", "rank", "matched_episode_id"}
    if len(links) != expected_links or not required.issubset(links.columns):
        raise WalkForwardOutcomeError("eligibility link inventory differs")
    link_base = links[[
        "query_id", "query_cutoff", "method", "rank", "matched_episode_id",
    ]].copy()
    link_base["query_timestamp"] = pd.to_datetime(link_base["query_cutoff"])
    frames = []
    for horizon in horizons:
        observed = outcomes.loc[
            outcomes.horizon_sessions == horizon,
            ["episode_id", "completion_timestamp", "complete", "status"],
        ].rename(columns={"episode_id": "matched_episode_id"})
        if len(observed) != expected_requests \
                or observed.matched_episode_id.astype(str).duplicated().any():
            raise WalkForwardOutcomeError("horizon outcome inventory differs")
        frame = link_base.merge(
            observed, on="matched_episode_id", how="left", validate="many_to_one",
        )
        completion = pd.to_datetime(frame["completion_timestamp"])
        complete = frame["complete"].fillna(False).astype(bool)
        eligible = complete & completion.notna() & (completion <= frame["query_timestamp"])
        reason = np.where(
            ~complete | completion.isna(), "incomplete_horizon",
            np.where(eligible, "eligible", "outcome_not_yet_observable"),
        )
        frames.append(pd.DataFrame({
            "query_id": frame["query_id"].astype(str),
            "method": frame["method"].astype(str),
            "rank": frame["rank"].astype("int16"),
            "matched_episode_id": frame["matched_episode_id"].astype(str),
            "horizon_sessions": np.full(len(frame), horizon, dtype=np.int16),
            "completion_timestamp": frame["completion_timestamp"],
            "outcome_status": frame["status"].astype(str),
            "eligible": eligible.astype(bool), "reason": reason,
        }))
    result = pd.concat(frames, ignore_index=True)
    method_order = {method: index for index, method in enumerate(cross_store.METHODS)}
    result["_method_order"] = result.method.map(method_order).astype("int8")
    result = result.sort_values(
        ["query_id", "_method_order", "rank", "horizon_sessions"], kind="stable",
    ).drop(columns="_method_order").reset_index(drop=True)
    if len(result) != expected_links * len(horizons) \
            or result[["query_id", "method", "rank", "horizon_sessions"]].duplicated().any() \
            or not set(result.reason.unique()).issubset({
                "eligible", "incomplete_horizon", "outcome_not_yet_observable",
            }):
        raise WalkForwardOutcomeError("eligibility table differs")
    return result


def _coverage(
    outcomes: pd.DataFrame, paths: pd.DataFrame, eligibility: pd.DataFrame,
    reuse_count: int,
) -> dict[str, Any]:
    state = {
        "schema_version": STORE_SCHEMA, "status": "complete",
        "unique_episodes": EXPECTED_REQUESTS,
        "reused_episodes": reuse_count,
        "computed_episodes": EXPECTED_REQUESTS - reuse_count,
        "outcome_rows": len(outcomes), "path_rows": len(paths),
        "link_eligibility_rows": len(eligibility),
        "complete_by_horizon": {
            str(horizon): int(outcomes.loc[
                outcomes.horizon_sessions == horizon, "complete",
            ].sum()) for horizon in HORIZONS
        },
        "status_counts": {
            str(key): int(value)
            for key, value in outcomes.status.value_counts().sort_index().items()
        },
        "eligibility_by_horizon": {
            str(horizon): int(eligibility.loc[
                eligibility.horizon_sessions == horizon, "eligible",
            ].sum()) for horizon in HORIZONS
        },
        "eligibility_reason_counts": {
            str(key): int(value)
            for key, value in eligibility.reason.value_counts().sort_index().items()
        },
        "historical_analogue_outcomes_accessed": True,
        "historical_query_evaluation_opened": False,
        "final_period_result_opened": False,
    }
    return {**state, "result_digest": stable_hash(state)}


def _aggregate(
    repository: Path, cache: Path, prereg: Mapping[str, Any],
    contract: Mapping[str, Any], reuse: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]], cache_seal: Mapping[str, Any],
    h1: str, elapsed_seconds: float,
) -> dict[str, Any]:
    final = repository / OUTPUT_RELATIVE
    if final.exists() or final.is_symlink():
        raise WalkForwardOutcomeError("outcome output already exists")
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{final.name}.", dir=final.parent,
    ))
    try:
        smoke._atomic_json(
            temporary / "RUN_STARTED.json", base._read(cache / "RUN_STARTED.json")
        )
        reused_outcomes, reused_paths = _reused_frames(
            repository, reuse, contract, str(prereg["source_content_digest"]),
        )
        computed_outcomes = [
            pd.read_parquet(cache / f"part-{index:02d}" / "episode-outcomes.parquet")
            for index in range(PARTITIONS)
        ]
        computed_paths = [
            pd.read_parquet(cache / f"part-{index:02d}" / "future-paths.parquet")
            for index in range(PARTITIONS)
        ]
        outcomes = old_store._cast_outcomes(pd.concat(
            [reused_outcomes, *computed_outcomes], ignore_index=True,
        ))
        paths = old_store._cast_paths(pd.concat(
            [reused_paths, *computed_paths], ignore_index=True,
        ))
        if len(outcomes) != prereg["expected_outcome_rows"] \
                or outcomes[["episode_id", "horizon_sessions"]].duplicated().any() \
                or outcomes.episode_id.astype(str).nunique() != EXPECTED_REQUESTS \
                or paths[["episode_id", "step"]].duplicated().any() \
                or not set(paths.episode_id.astype(str)).issubset(
                    set(outcomes.episode_id.astype(str))
                ):
            raise WalkForwardOutcomeError("aggregate outcome/path inventory differs")
        links = pd.read_parquet(
            repository / cross_store.OUTPUT_RELATIVE / cross_store.LINK_FILE,
            engine="pyarrow",
        )
        eligibility = _eligibility_table(links, outcomes)
        coverage = _coverage(outcomes, paths, eligibility, len(reuse))
        smoke._atomic_parquet(temporary / "episode-outcomes.parquet", outcomes)
        smoke._atomic_parquet(temporary / "future-paths.parquet", paths)
        smoke._atomic_parquet(
            temporary / "link-outcome-eligibility.parquet", eligibility,
        )
        smoke._atomic_json(temporary / "COVERAGE.json", coverage)
        names = (
            "episode-outcomes.parquet", "future-paths.parquet",
            "link-outcome-eligibility.parquet", "COVERAGE.json",
        )
        state = {
            "schema_version": STORE_SCHEMA, "status": "sealed", "passed": True,
            "preregistration_h1": h1,
            "preregistration_digest": prereg["preregistration_digest"],
            "contract_digest": contract["contract_digest"],
            "source_content_digest": prereg["source_content_digest"],
            "cross_store_verification_digest": prereg["cross_store"]["verification_digest"],
            "reusable_store_verification_result_digest": prereg[
                "reusable_store"
            ]["verification_result_digest"],
            "partition_cache_result_digest": cache_seal["result_digest"],
            "request_count": EXPECTED_REQUESTS,
            "reused_episodes": len(reuse),
            "computed_episodes": prereg["missing_count"],
            "outcome_rows": len(outcomes), "path_rows": len(paths),
            "link_eligibility_rows": len(eligibility),
            "semantic_digests": {
                "outcome_digest": old_store._frame_digest(
                    outcomes, ("episode_id", "horizon_sessions"), presorted=True,
                ),
                "path_digest": old_store._frame_digest(
                    paths, ("episode_id", "step"), presorted=True,
                ),
                "eligibility_digest": old_store._frame_digest(
                    eligibility,
                    ("query_id", "method", "rank", "horizon_sessions"),
                    presorted=True,
                ),
                "coverage_result_digest": coverage["result_digest"],
            },
            "file_manifest": old_store._file_manifest(temporary, names),
            "partition_elapsed_seconds": elapsed_seconds,
            "historical_analogue_outcomes_accessed": True,
            "historical_query_evaluation_opened": False,
            "final_period_result_opened": False,
            "outcomes_affected_retrieval": False,
            "production_promotion_authorized": False,
            "independent_verification_authorized": True,
        }
        seal = {
            **state, "result_digest": stable_hash({
                key: value for key, value in state.items()
                if key != "partition_elapsed_seconds"
            }), "created_at": _now(),
        }
        smoke._atomic_json(temporary / "SEALED.json", seal)
        os.replace(temporary, final)
        return seal
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _existing_output(
    repository: Path, prereg: Mapping[str, Any], h1: str,
) -> dict[str, Any] | None:
    root = repository / OUTPUT_RELATIVE
    if not root.exists():
        return None
    expected = {
        "RUN_STARTED.json", "episode-outcomes.parquet", "future-paths.parquet",
        "link-outcome-eligibility.parquet", "COVERAGE.json", "SEALED.json",
    }
    if root.is_symlink() or not root.is_dir() \
            or {path.name for path in root.iterdir()} != expected \
            or any(path.is_symlink() for path in root.iterdir()):
        raise WalkForwardOutcomeError("outcome output layout differs")
    seal = base._read(root / "SEALED.json")
    if not all((
        _receipt_valid(seal, timing=True), seal.get("passed") is True,
        seal.get("preregistration_h1") == h1,
        seal.get("preregistration_digest") == prereg["preregistration_digest"],
        seal.get("request_count") == EXPECTED_REQUESTS,
        seal.get("reused_episodes") == EXPECTED_REUSE,
        seal.get("computed_episodes") == EXPECTED_MISSING,
        seal.get("outcome_rows") == EXPECTED_REQUESTS * len(HORIZONS),
        seal.get("link_eligibility_rows") == EXPECTED_ELIGIBILITY_ROWS,
        seal.get("file_manifest") == old_store._file_manifest(root, (
            "episode-outcomes.parquet", "future-paths.parquet",
            "link-outcome-eligibility.parquet", "COVERAGE.json",
        )),
        seal.get("historical_query_evaluation_opened") is False,
        seal.get("final_period_result_opened") is False,
    )):
        raise WalkForwardOutcomeError("outcome terminal publication differs")
    return seal


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    prereg, contract, reuse, _missing, groups, h1 = validate_preregistration(repository)
    prior = _existing_output(repository, prereg, h1)
    if prior is not None:
        return prior
    free = os.statvfs(repository)
    if free.f_bavail * free.f_frsize < 5 * 1024 ** 3:
        raise WalkForwardOutcomeError("less than 5 GiB free before outcome build")
    cache = _start_cache(repository, prereg, h1)
    started = perf_counter()
    receipts: list[dict[str, Any] | None] = [None] * PARTITIONS
    missing_partitions: list[int] = []
    for index, requests in enumerate(groups):
        if (cache / f"part-{index:02d}").exists():
            receipts[index] = _validate_partition(cache, index, requests, prereg, h1)
        else:
            missing_partitions.append(index)
    if (cache / "CACHE_SEALED.json").exists() and missing_partitions:
        raise WalkForwardOutcomeError("sealed cache is missing partitions")
    if missing_partitions:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=PROCESSES, mp_context=context) as pool:
            futures = {
                pool.submit(
                    _compute_partition, str(repository), index, groups[index], str(cache),
                    prereg["preregistration_digest"], h1,
                    contract["contract_digest"], prereg["source_content_digest"],
                    prereg["benchmark_fingerprint"],
                ): index for index in missing_partitions
            }
            for future in as_completed(futures):
                index = futures[future]
                future.result()
                receipts[index] = _validate_partition(
                    cache, index, groups[index], prereg, h1,
                )
                print(
                    f"[wf03d-outcomes] partition={index:02d} "
                    f"complete={sum(row is not None for row in receipts)}/{PARTITIONS}",
                    flush=True,
                )
    complete = [row for row in receipts if row is not None]
    if len(complete) != PARTITIONS:
        raise WalkForwardOutcomeError("not all outcome partitions completed")
    cache_seal = _seal_cache(cache, complete, prereg, h1)
    return _aggregate(
        repository, cache, prereg, contract, reuse, complete, cache_seal, h1,
        perf_counter() - started,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preregister", "run"))
    parser.add_argument("--repository", required=True, type=Path)
    args = parser.parse_args(argv)
    repository = args.repository.resolve(strict=True)
    if args.mode == "preregister":
        base._atomic(
            repository / PREREGISTRATION_RELATIVE,
            build_preregistration(repository),
        )
        return 0
    print(json.dumps(execute(repository), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
