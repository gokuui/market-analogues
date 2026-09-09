"""Independently verify every causal WF-03D analogue outcome and path."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import multiprocessing
import os
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from experiments.m04r.m04r14_t14_09_outcome_oracle import (
    prepare_reference_series, reference_prepared_episode,
)
from market_analogues.adapters import source_from_spec
from market_analogues.config import load_config
from market_analogues.types import EpisodeKey, InstrumentKey, stable_hash

from experiments.m04r import m04r14_t14_09_full_outcome_store as old_store
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_10_wf03d_cross_store_manifest as cross_store
from experiments.m04r import m04r14_t14_10_wf03d_outcome_store as producer
from experiments.m04r import verify_m04r14_t14_09_full_outcome_store as oracle_tools


SCHEMA = "m04r14-t14-10-wf03d-outcome-verification-v1"


class WalkForwardOutcomeVerificationError(RuntimeError):
    pass


def _git(repository: Path, *args: str, raw: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=repository, capture_output=True,
        text=not raw, check=False,
    )
    if result.returncode:
        message = result.stderr if not raw else result.stderr.decode(errors="replace")
        raise WalkForwardOutcomeVerificationError(
            message.strip() or "git command failed"
        )
    return result.stdout if raw else result.stdout.strip()


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise WalkForwardOutcomeVerificationError(f"regular file required: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise WalkForwardOutcomeVerificationError(f"regular JSON required: {path}")
    raw = path.read_bytes()

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise WalkForwardOutcomeVerificationError(
                    f"duplicate JSON key: {path}:{key}"
                )
            result[key] = value
        return result

    value = json.loads(
        raw, object_pairs_hook=pairs,
        parse_constant=lambda item: (_ for _ in ()).throw(
            WalkForwardOutcomeVerificationError(f"non-finite JSON: {path}:{item}")
        ),
    )
    if type(value) is not dict:
        raise WalkForwardOutcomeVerificationError(f"JSON object required: {path}")
    return value, raw


def _receipt_valid(
    value: Mapping[str, Any], *, timing: bool = False,
    digest_key: str = "result_digest",
) -> bool:
    omitted = {digest_key, "created_at"}
    if timing:
        omitted |= {"elapsed_seconds", "partition_elapsed_seconds"}
    return value.get(digest_key) == stable_hash({
        key: item for key, item in value.items() if key not in omitted
    })


def _base_seal_valid(value: Mapping[str, Any], digest_key: str) -> bool:
    return value.get(digest_key) == stable_hash({
        key: item for key, item in value.items() if key != digest_key
    })


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
            if parents == [child, h0] and changed == [producer.PREREGISTRATION_RELATIVE.as_posix()] \
                    and _git(repository, "show", f"{child}:{producer.PREREGISTRATION_RELATIVE}", raw=True) == raw:
                accepted.append(child)
    if len(set(accepted)) != 1:
        raise WalkForwardOutcomeVerificationError("preregistration lifecycle differs")
    return accepted[0]


def _requests(repository: Path) -> list[dict[str, Any]]:
    request_frame = pd.read_parquet(
        repository / cross_store.OUTPUT_RELATIVE / cross_store.REQUEST_FILE,
        engine="pyarrow",
    )
    registry, _ = base._registry(repository)
    accounting = pd.read_parquet(repository / producer.SOURCE_ACCOUNTING, engine="pyarrow")
    if stable_hash(cross_store._plain(accounting.to_dict("records"))) \
            != registry.get("source_accounting_digest") \
            or accounting.symbol.astype(str).duplicated().any():
        raise WalkForwardOutcomeVerificationError("source accounting differs")
    fingerprints = {
        str(row.symbol): str(row.source_hash_at_lock)
        for row in accounting.itertuples(index=False) if row.error is None
    }
    result = []
    for row in request_frame.itertuples(index=False):
        fingerprint = fingerprints.get(str(row.symbol))
        if not fingerprint:
            raise WalkForwardOutcomeVerificationError("request source fingerprint absent")
        result.append({
            "episode_id": str(row.episode_id), "dataset_id": str(row.dataset_id),
            "symbol": str(row.symbol), "cutoff": str(row.cutoff),
            "quality_tier": str(row.quality_tier),
            "expected_source_fingerprint": fingerprint,
        })
    if len(result) != producer.EXPECTED_REQUESTS:
        raise WalkForwardOutcomeVerificationError("request count differs")
    return result


def _reusable_identity_matches(
    request: Mapping[str, Any], observed: Any,
    contract_digest: str, source_content_digest: str,
) -> bool:
    def field(name: str) -> Any:
        return observed.get(name) if isinstance(observed, Mapping) else getattr(observed, name)

    return all((
        str(field("cutoff")) == str(request["cutoff"]),
        str(field("source_fingerprint"))
            == str(request["expected_source_fingerprint"]),
        str(field("contract_digest")) == contract_digest,
        str(field("source_content_digest")) == source_content_digest,
    ))


def _reuse_split(
    repository: Path, requests: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    contract, _ = _read(repository / producer.CONTRACT)
    identity = pd.read_parquet(
        repository / old_store.OUTPUT / "episode-outcomes.parquet",
        columns=[
            "episode_id", "cutoff", "source_fingerprint", "contract_digest",
            "source_content_digest",
        ], engine="pyarrow",
    ).drop_duplicates()
    if identity.episode_id.astype(str).duplicated().any():
        raise WalkForwardOutcomeVerificationError("reusable identity conflicts")
    old = {str(row.episode_id): row for row in identity.itertuples(index=False)}
    content = base._resident()["content_digest"]
    reuse, missing = [], []
    for raw in requests:
        request = dict(raw)
        row = old.get(request["episode_id"])
        target = reuse if row is not None and _reusable_identity_matches(
            request, row, contract["contract_digest"], content,
        ) else missing
        target.append(request)
    if len(reuse) != producer.EXPECTED_REUSE \
            or len(missing) != producer.EXPECTED_MISSING:
        raise WalkForwardOutcomeVerificationError("reuse split differs")
    return reuse, missing


def _groups(requests: Sequence[Mapping[str, Any]]) -> list[tuple[dict[str, Any], ...]]:
    groups: list[list[dict[str, Any]]] = [[] for _ in range(producer.PARTITIONS)]
    for raw in requests:
        row = dict(raw)
        index = int(sha256(row["symbol"].encode()).hexdigest(), 16) % producer.PARTITIONS
        groups[index].append(row)
    return [tuple(sorted(group, key=lambda row: (
        row["symbol"], row["cutoff"], row["episode_id"],
    ))) for group in groups]


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
    repository_raw: str, index: int, requests: tuple[dict[str, Any], ...],
    contract_digest: str, source_content_digest: str, benchmark_fingerprint: str,
) -> dict[str, Any]:
    started = perf_counter()
    repository = Path(repository_raw)
    observed_root = repository / producer.CACHE_RELATIVE / f"part-{index:02d}"
    observed_outcomes = pd.read_parquet(observed_root / "episode-outcomes.parquet")
    observed_paths = pd.read_parquet(observed_root / "future-paths.parquet")
    source = source_from_spec(
        load_config(repository / base.CONFIG_RELATIVE).datasets["nasdaq"]
    )
    benchmark = source.load_benchmark()
    if benchmark is None or not _valid_source(benchmark) \
            or source.benchmark_fingerprint() != benchmark_fingerprint:
        raise WalkForwardOutcomeVerificationError("oracle benchmark differs")
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
                or {row["expected_source_fingerprint"] for row in by_symbol[symbol]} \
                    != {fingerprint}:
            raise WalkForwardOutcomeVerificationError(f"oracle source differs: {symbol}")
        reference_stock = prepare_reference_series(stock)
        for request in by_symbol[symbol]:
            cutoff = pd.Timestamp(request["cutoff"])
            if EpisodeKey(key, cutoff, 252, "dense-v1").id != request["episode_id"]:
                raise WalkForwardOutcomeVerificationError("oracle episode identity differs")
            expected_outcomes, expected_paths = reference_prepared_episode(
                reference_stock, reference_benchmark,
                episode_id=request["episode_id"], cutoff=cutoff,
                source_fingerprint=fingerprint, contract_digest=contract_digest,
                source_content_digest=source_content_digest,
            )
            outcomes.extend(expected_outcomes)
            paths.extend(expected_paths)
    outcomes.sort(key=lambda row: (row["episode_id"], row["horizon_sessions"]))
    paths.sort(key=lambda row: (row["episode_id"], row["step"]))
    if oracle_tools._records(
        observed_outcomes, ("episode_id", "horizon_sessions"),
    ) != outcomes or oracle_tools._records(
        observed_paths, ("episode_id", "step"),
    ) != paths:
        raise WalkForwardOutcomeVerificationError(
            f"independent numeric oracle differs: {index}"
        )
    state = {
        "schema_version": SCHEMA, "status": "partition_verified", "passed": True,
        "partition": index, "request_count": len(requests),
        "outcome_rows": len(outcomes), "path_rows": len(paths),
        "outcome_digest": oracle_tools._record_digest(outcomes),
        "path_digest": oracle_tools._record_digest(paths),
        "elapsed_seconds": perf_counter() - started,
    }
    return {
        **state, "result_digest": stable_hash({
            key: value for key, value in state.items() if key != "elapsed_seconds"
        }), "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _expected_eligibility(
    links: pd.DataFrame, outcomes: pd.DataFrame,
) -> pd.DataFrame:
    base_links = links[[
        "query_id", "query_cutoff", "method", "rank", "matched_episode_id",
    ]].copy()
    base_links["query_timestamp"] = pd.to_datetime(base_links["query_cutoff"])
    frames = []
    for horizon in producer.HORIZONS:
        observed = outcomes.loc[
            outcomes.horizon_sessions == horizon,
            ["episode_id", "completion_timestamp", "complete", "status"],
        ].rename(columns={"episode_id": "matched_episode_id"})
        frame = base_links.merge(
            observed, on="matched_episode_id", how="left", validate="many_to_one",
        )
        completion = pd.to_datetime(frame.completion_timestamp)
        complete = frame.complete.fillna(False).astype(bool)
        eligible = complete & completion.notna() & (completion <= frame.query_timestamp)
        frames.append(pd.DataFrame({
            "query_id": frame.query_id.astype(str),
            "method": frame.method.astype(str),
            "rank": frame["rank"].astype("int16"),
            "matched_episode_id": frame.matched_episode_id.astype(str),
            "horizon_sessions": np.full(len(frame), horizon, dtype=np.int16),
            "completion_timestamp": frame.completion_timestamp,
            "outcome_status": frame.status.astype(str),
            "eligible": eligible.astype(bool),
            "reason": np.where(
                ~complete | completion.isna(), "incomplete_horizon",
                np.where(eligible, "eligible", "outcome_not_yet_observable"),
            ),
        }))
    result = pd.concat(frames, ignore_index=True)
    order = {method: index for index, method in enumerate(cross_store.METHODS)}
    result["_method_order"] = result.method.map(order).astype("int8")
    return result.sort_values(
        ["query_id", "_method_order", "rank", "horizon_sessions"], kind="stable",
    ).drop(columns="_method_order").reset_index(drop=True)


def _verify_lifecycle(
    repository: Path,
) -> tuple[
    dict[str, Any], dict[str, Any], list[dict[str, Any]],
    list[dict[str, Any]], list[tuple[dict[str, Any], ...]], str,
]:
    path = repository / producer.PREREGISTRATION_RELATIVE
    prereg, raw = _read(path)
    state = {key: value for key, value in prereg.items() if key != "preregistration_digest"}
    if prereg.get("schema_version") != producer.SCHEMA \
            or prereg.get("preregistration_digest") != stable_hash(state):
        raise WalkForwardOutcomeVerificationError("preregistration seal differs")
    h0 = str(prereg.get("implementation_h0"))
    h1 = _sole_child(repository, raw, h0)
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", h1, "HEAD"], cwd=repository,
    ).returncode:
        raise WalkForwardOutcomeVerificationError("HEAD lineage differs")
    for name, expected in prereg.get("runtime_files", {}).items():
        if _sha(repository / name) != expected \
                or sha256(_git(repository, "show", f"{h0}:{name}", raw=True)).hexdigest() != expected \
                or sha256(_git(repository, "show", f"{h1}:{name}", raw=True)).hexdigest() != expected:
            raise WalkForwardOutcomeVerificationError(f"runtime drifted: {name}")
    contract, _ = _read(repository / producer.CONTRACT)
    contract_receipt, _ = _read(repository / producer.smoke.CONTRACT_RECEIPT)
    synthetic, _ = _read(repository / producer.smoke.SYNTHETIC)
    expected_contract_evidence = {
        "contract_verification_result_digest": contract_receipt["result_digest"],
        "contract_verification_sha256": _sha(
            repository / producer.smoke.CONTRACT_RECEIPT
        ),
        "synthetic_result_digest": synthetic["result_digest"],
        "synthetic_sha256": _sha(repository / producer.smoke.SYNTHETIC),
    }
    cross_root = repository / cross_store.OUTPUT_RELATIVE
    cross_result, _ = _read(cross_root / "RESULT.json")
    cross_manifest, _ = _read(cross_root / "MANIFEST.json")
    cross_verified_path = repository / producer.cross_verifier.OUTPUT_RELATIVE / "VERIFIED.json"
    cross_verified, _ = _read(cross_verified_path)
    expected_cross = {
        "result_digest": cross_result["result_digest"],
        "result_sha256": _sha(cross_root / "RESULT.json"),
        "manifest_digest": cross_manifest["manifest_digest"],
        "manifest_sha256": _sha(cross_root / "MANIFEST.json"),
        "verification_digest": cross_verified["verification_digest"],
        "verification_sha256": _sha(cross_verified_path),
        "raw_link_sha256": _sha(cross_root / cross_store.LINK_FILE),
        "request_sha256": _sha(cross_root / cross_store.REQUEST_FILE),
        "raw_link_semantic_digest": cross_result["raw_link_semantic_digest"],
        "request_semantic_digest": cross_result["episode_request_semantic_digest"],
    }
    old_root = repository / old_store.OUTPUT
    old_seal, _ = _read(old_root / "SEALED.json")
    old_verified_path = repository / producer.OLD_VERIFICATION
    old_verified, _ = _read(old_verified_path)
    expected_old = {
        "seal_result_digest": old_seal["result_digest"],
        "seal_sha256": _sha(old_root / "SEALED.json"),
        "verification_result_digest": old_verified["result_digest"],
        "verification_sha256": _sha(old_verified_path),
        "outcomes_sha256": _sha(old_root / "episode-outcomes.parquet"),
        "paths_sha256": _sha(old_root / "future-paths.parquet"),
    }
    requests = _requests(repository)
    reuse, missing = _reuse_split(repository, requests)
    groups = _groups(missing)
    if any((
        prereg.get("contract_digest") != contract.get("contract_digest"),
        prereg.get("contract_sha256") != _sha(repository / producer.CONTRACT),
        prereg.get("outcome_contract_evidence") != expected_contract_evidence,
        not _receipt_valid(contract_receipt),
        contract_receipt.get("passed") is not True,
        contract_receipt.get("contract_digest") != contract.get("contract_digest"),
        not _receipt_valid(synthetic, timing=True),
        synthetic.get("passed") is not True,
        synthetic.get("contract_digest") != contract.get("contract_digest"),
        prereg.get("cross_store") != expected_cross,
        not _base_seal_valid(cross_result, "result_digest"),
        not _base_seal_valid(cross_manifest, "manifest_digest"),
        not _base_seal_valid(cross_verified, "verification_digest"),
        cross_verified.get("walk_forward_outcome_store_construction_authorized")
            is not True,
        prereg.get("reusable_store") != expected_old,
        not _receipt_valid(old_seal, timing=True),
        not _receipt_valid(old_verified),
        old_verified.get("store_result_digest") != old_seal.get("result_digest"),
        prereg.get("source_content_digest") != base._resident()["content_digest"],
        prereg.get("source_accounting_sha256")
            != _sha(repository / producer.SOURCE_ACCOUNTING),
        prereg.get("source_accounting_digest")
            != base._registry(repository)[0]["source_accounting_digest"],
        prereg.get("request_digest") != stable_hash(requests),
        prereg.get("reuse_request_digest") != stable_hash(reuse),
        prereg.get("missing_request_digest") != stable_hash(missing),
        prereg.get("partition_request_counts") != [len(group) for group in groups],
        prereg.get("partition_request_digests") != [stable_hash(group) for group in groups],
        prereg.get("historical_analogue_outcomes_accessed") is not False,
        prereg.get("historical_query_evaluation_opened") is not False,
        prereg.get("final_period_result_opened") is not False,
    )):
        raise WalkForwardOutcomeVerificationError("preregistered inventory differs")
    return prereg, contract, reuse, missing, groups, h1


def _verification_root(
    repository: Path, prereg: Mapping[str, Any], h1: str,
) -> Path:
    root = repository / producer.VERIFICATION_RELATIVE
    expected = {
        "schema_version": SCHEMA, "status": "verification_running",
        "preregistration_h1": h1,
        "preregistration_digest": prereg["preregistration_digest"],
        "partition_count": producer.PARTITIONS,
    }
    if not root.exists():
        root.mkdir(parents=True)
        base._atomic(root / "RUN_STARTED.json", {
            **expected, "created_at": datetime.now(timezone.utc).isoformat(),
        })
    if root.is_symlink() or not root.is_dir():
        raise WalkForwardOutcomeVerificationError("verification root differs")
    started, _ = _read(root / "RUN_STARTED.json")
    if {key: value for key, value in started.items() if key != "created_at"} != expected:
        raise WalkForwardOutcomeVerificationError("verification start receipt differs")
    allowed = {"RUN_STARTED.json", "VERIFIED.json"} | {
        f"part-{index:02d}.json" for index in range(producer.PARTITIONS)
    }
    if any(path.name not in allowed or path.is_symlink() for path in root.iterdir()):
        raise WalkForwardOutcomeVerificationError("verification root is not closed")
    return root


def _valid_verifier_receipt(
    path: Path, index: int, requests: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    receipt, _ = _read(path)
    if not all((
        _receipt_valid(receipt, timing=True),
        receipt.get("schema_version") == SCHEMA,
        receipt.get("status") == "partition_verified",
        receipt.get("passed") is True,
        receipt.get("partition") == index,
        receipt.get("request_count") == len(requests),
    )):
        raise WalkForwardOutcomeVerificationError(
            f"verifier partition receipt differs: {index}"
        )
    return receipt


def verify(repository: Path) -> dict[str, Any]:
    started_at = perf_counter()
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain"):
        raise WalkForwardOutcomeVerificationError("verifier requires clean commit")
    verifier_commit = str(_git(repository, "rev-parse", "HEAD"))
    prereg, contract, reuse, _missing, groups, h1 = _verify_lifecycle(repository)
    output = repository / producer.OUTPUT_RELATIVE
    cache = repository / producer.CACHE_RELATIVE
    expected_output = {
        "RUN_STARTED.json", "episode-outcomes.parquet", "future-paths.parquet",
        "link-outcome-eligibility.parquet", "COVERAGE.json", "SEALED.json",
    }
    expected_cache = {"RUN_STARTED.json", "CACHE_SEALED.json"} | {
        f"part-{index:02d}" for index in range(producer.PARTITIONS)
    }
    if output.is_symlink() or not output.is_dir() \
            or {path.name for path in output.iterdir()} != expected_output \
            or cache.is_symlink() or not cache.is_dir() \
            or {path.name for path in cache.iterdir()} != expected_cache:
        raise WalkForwardOutcomeVerificationError("producer file closure differs")
    seal, seal_raw = _read(output / "SEALED.json")
    cache_seal, _ = _read(cache / "CACHE_SEALED.json")
    file_names = (
        "episode-outcomes.parquet", "future-paths.parquet",
        "link-outcome-eligibility.parquet", "COVERAGE.json",
    )
    if not all((
        _receipt_valid(seal, timing=True), seal.get("passed") is True,
        seal.get("preregistration_h1") == h1,
        seal.get("preregistration_digest") == prereg["preregistration_digest"],
        seal.get("contract_digest") == contract["contract_digest"],
        seal.get("request_count") == producer.EXPECTED_REQUESTS,
        seal.get("reused_episodes") == producer.EXPECTED_REUSE,
        seal.get("computed_episodes") == producer.EXPECTED_MISSING,
        seal.get("outcome_rows") == producer.EXPECTED_REQUESTS * len(producer.HORIZONS),
        seal.get("link_eligibility_rows") == producer.EXPECTED_ELIGIBILITY_ROWS,
        seal.get("file_manifest") == old_store._file_manifest(output, file_names),
        seal.get("partition_cache_result_digest") == cache_seal.get("result_digest"),
        seal.get("historical_query_evaluation_opened") is False,
        seal.get("final_period_result_opened") is False,
        seal.get("outcomes_affected_retrieval") is False,
        _receipt_valid(cache_seal), cache_seal.get("passed") is True,
        cache_seal.get("computed_episodes") == producer.EXPECTED_MISSING,
    )):
        raise WalkForwardOutcomeVerificationError("producer seal differs")
    partition_receipts = []
    for index, group in enumerate(groups):
        root = cache / f"part-{index:02d}"
        if root.is_symlink() or {path.name for path in root.iterdir()} != {
            "PARTITION.json", "episode-outcomes.parquet", "future-paths.parquet",
        }:
            raise WalkForwardOutcomeVerificationError("producer partition layout differs")
        receipt, _ = _read(root / "PARTITION.json")
        if not all((
            _receipt_valid(receipt, timing=True), receipt.get("passed") is True,
            receipt.get("partition") == index,
            receipt.get("request_count") == len(group),
            receipt.get("request_digest") == stable_hash(group),
            receipt.get("outcome_rows") == len(group) * len(producer.HORIZONS),
            receipt.get("file_manifest") == old_store._file_manifest(
                root, ("episode-outcomes.parquet", "future-paths.parquet"),
            ),
        )):
            raise WalkForwardOutcomeVerificationError("producer partition seal differs")
        partition_receipts.append(receipt)
    if cache_seal.get("partition_result_digests") != [
        row["result_digest"] for row in partition_receipts
    ]:
        raise WalkForwardOutcomeVerificationError("partition digest closure differs")

    verification_root = _verification_root(repository, prereg, h1)
    verified: list[dict[str, Any] | None] = [None] * producer.PARTITIONS
    pending = []
    for index, group in enumerate(groups):
        value = _valid_verifier_receipt(
            verification_root / f"part-{index:02d}.json", index, group,
        )
        verified[index] = value
        if value is None:
            pending.append(index)
    if pending:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=producer.PROCESSES, mp_context=context,
        ) as pool:
            futures = {
                pool.submit(
                    _verify_partition_worker, str(repository), index, groups[index],
                    contract["contract_digest"], prereg["source_content_digest"],
                    prereg["benchmark_fingerprint"],
                ): index for index in pending
            }
            for future in as_completed(futures):
                index = futures[future]
                value = future.result()
                base._atomic(verification_root / f"part-{index:02d}.json", value)
                verified[index] = value
                print(
                    f"[wf03d-outcome-verifier] partition={index:02d} "
                    f"complete={sum(row is not None for row in verified)}/{producer.PARTITIONS}",
                    flush=True,
                )
    complete = [row for row in verified if row is not None]
    if len(complete) != producer.PARTITIONS:
        raise WalkForwardOutcomeVerificationError("oracle partitions incomplete")
    for independent, produced in zip(complete, partition_receipts, strict=True):
        if any(independent.get(key) != produced.get(key) for key in (
            "partition", "request_count", "outcome_rows", "path_rows",
            "outcome_digest", "path_digest",
        )):
            raise WalkForwardOutcomeVerificationError("oracle digest differs")

    outcomes = old_store._cast_outcomes(pd.read_parquet(
        output / "episode-outcomes.parquet", engine="pyarrow",
    ))
    paths = old_store._cast_paths(pd.read_parquet(
        output / "future-paths.parquet", engine="pyarrow",
    ))
    eligibility = pd.read_parquet(
        output / "link-outcome-eligibility.parquet", engine="pyarrow",
    )
    if len(outcomes) != producer.EXPECTED_REQUESTS * len(producer.HORIZONS) \
            or outcomes[["episode_id", "horizon_sessions"]].duplicated().any() \
            or paths[["episode_id", "step"]].duplicated().any() \
            or len(eligibility) != producer.EXPECTED_ELIGIBILITY_ROWS \
            or eligibility[["query_id", "method", "rank", "horizon_sessions"]].duplicated().any():
        raise WalkForwardOutcomeVerificationError("aggregate coverage differs")
    reuse_ids = {row["episode_id"] for row in reuse}
    old_outcomes = old_store._cast_outcomes(pd.read_parquet(
        repository / old_store.OUTPUT / "episode-outcomes.parquet", engine="pyarrow",
    ))
    old_paths = old_store._cast_paths(pd.read_parquet(
        repository / old_store.OUTPUT / "future-paths.parquet", engine="pyarrow",
    ))
    observed_reuse_outcomes = outcomes.loc[outcomes.episode_id.astype(str).isin(reuse_ids)]
    observed_reuse_paths = paths.loc[paths.episode_id.astype(str).isin(reuse_ids)]
    expected_reuse_outcomes = old_outcomes.loc[
        old_outcomes.episode_id.astype(str).isin(reuse_ids)
    ]
    expected_reuse_paths = old_paths.loc[old_paths.episode_id.astype(str).isin(reuse_ids)]
    if oracle_tools._records(
        observed_reuse_outcomes, ("episode_id", "horizon_sessions"),
    ) != oracle_tools._records(
        expected_reuse_outcomes, ("episode_id", "horizon_sessions"),
    ) or oracle_tools._records(
        observed_reuse_paths, ("episode_id", "step"),
    ) != oracle_tools._records(
        expected_reuse_paths, ("episode_id", "step"),
    ):
        raise WalkForwardOutcomeVerificationError("verified reuse differs")
    for index, group in enumerate(groups):
        ids = {row["episode_id"] for row in group}
        if old_store._frame_digest(
            outcomes.loc[outcomes.episode_id.astype(str).isin(ids)],
            ("episode_id", "horizon_sessions"),
        ) != partition_receipts[index]["outcome_digest"] \
                or old_store._frame_digest(
                    paths.loc[paths.episode_id.astype(str).isin(ids)],
                    ("episode_id", "step"),
                ) != partition_receipts[index]["path_digest"]:
            raise WalkForwardOutcomeVerificationError(
                f"aggregate partition differs: {index}"
            )

    links = pd.read_parquet(
        repository / cross_store.OUTPUT_RELATIVE / cross_store.LINK_FILE,
        engine="pyarrow",
    )
    expected_eligibility = _expected_eligibility(links, outcomes)
    if tuple(eligibility.columns) != tuple(expected_eligibility.columns) \
            or not eligibility.equals(expected_eligibility):
        raise WalkForwardOutcomeVerificationError("causal eligibility differs")
    coverage, _ = _read(output / "COVERAGE.json")
    coverage_state = {key: value for key, value in coverage.items() if key != "result_digest"}
    expected_coverage = {
        "schema_version": producer.STORE_SCHEMA, "status": "complete",
        "unique_episodes": producer.EXPECTED_REQUESTS,
        "reused_episodes": producer.EXPECTED_REUSE,
        "computed_episodes": producer.EXPECTED_MISSING,
        "outcome_rows": len(outcomes), "path_rows": len(paths),
        "link_eligibility_rows": len(eligibility),
        "complete_by_horizon": {
            str(horizon): int(outcomes.loc[
                outcomes.horizon_sessions == horizon, "complete",
            ].sum()) for horizon in producer.HORIZONS
        },
        "status_counts": {
            str(key): int(value)
            for key, value in outcomes.status.value_counts().sort_index().items()
        },
        "eligibility_by_horizon": {
            str(horizon): int(eligibility.loc[
                eligibility.horizon_sessions == horizon, "eligible",
            ].sum()) for horizon in producer.HORIZONS
        },
        "eligibility_reason_counts": {
            str(key): int(value)
            for key, value in eligibility.reason.value_counts().sort_index().items()
        },
        "historical_analogue_outcomes_accessed": True,
        "historical_query_evaluation_opened": False,
        "final_period_result_opened": False,
    }
    semantic = {
        "outcome_digest": old_store._frame_digest(
            outcomes, ("episode_id", "horizon_sessions"), presorted=True,
        ),
        "path_digest": old_store._frame_digest(
            paths, ("episode_id", "step"), presorted=True,
        ),
        "eligibility_digest": old_store._frame_digest(
            eligibility, ("query_id", "method", "rank", "horizon_sessions"),
            presorted=True,
        ),
        "coverage_result_digest": coverage.get("result_digest"),
    }
    if coverage_state != expected_coverage \
            or coverage.get("result_digest") != stable_hash(coverage_state) \
            or seal.get("semantic_digests") != semantic:
        raise WalkForwardOutcomeVerificationError("coverage/semantic seal differs")

    state = {
        "schema_version": SCHEMA, "status": "verified", "passed": True,
        "verifier_commit": verifier_commit,
        "verifier_runtime_sha256": _sha(Path(__file__).resolve()),
        "preregistration_digest": prereg["preregistration_digest"],
        "store_result_digest": seal["result_digest"],
        "store_result_sha256": sha256(seal_raw).hexdigest(),
        "contract_digest": contract["contract_digest"],
        "unique_episodes": producer.EXPECTED_REQUESTS,
        "reused_episodes": producer.EXPECTED_REUSE,
        "independently_recomputed_episodes": producer.EXPECTED_MISSING,
        "verified_outcome_rows": len(outcomes),
        "verified_path_rows": len(paths),
        "verified_link_eligibility_rows": len(eligibility),
        "verified_partitions": producer.PARTITIONS,
        "gates": {
            "all_new_outcomes_match_independent_oracle": True,
            "all_reused_rows_match_prior_verified_store": True,
            "all_aggregate_partition_digests_reconcile": True,
            "all_causal_eligibility_rows_reconstructed": True,
            "all_physical_and_semantic_seals_valid": True,
            "historical_query_evaluation_unopened": True,
            "final_period_unopened": True,
        },
        "historical_analogue_outcomes_accessed": True,
        "historical_query_evaluation_opened": False,
        "final_period_result_opened": False,
        "outcomes_affected_retrieval": False,
        "prediction_store_construction_authorized": True,
        "production_promotion_authorized": False,
        "elapsed_seconds": perf_counter() - started_at,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return {**state, "verification_digest": stable_hash(state)}


def publish(repository: Path) -> Path:
    value = verify(repository)
    root = repository.resolve(strict=True) / producer.VERIFICATION_RELATIVE
    path = root / "VERIFIED.json"
    if path.exists():
        prior, _ = _read(path)
        omitted = {"verification_digest", "elapsed_seconds", "created_at"}
        if {key: item for key, item in prior.items() if key not in omitted} \
                != {key: item for key, item in value.items() if key not in omitted}:
            raise WalkForwardOutcomeVerificationError("verification replay differs")
        return path
    base._atomic(path, value)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    args = parser.parse_args(argv)
    print(publish(args.repository))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
