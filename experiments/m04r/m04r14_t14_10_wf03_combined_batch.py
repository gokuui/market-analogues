"""Preregister and run certified combined retrieval for all 3,936 WF-03 queries."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import stat
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from market_analogues.adapters import CachedOHLCVSource, source_from_spec
from market_analogues.config import load_config
from market_analogues.dtw_component_search import (
    certified_staged_dtw_component_search,
    staged_dtw_component_search_contract,
)
from market_analogues.dtw_sample_store import load_dtw_sample_generation
from market_analogues.episodes import build_episode
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.resident_store import (
    CONTENT_SCHEMA_VERSION,
    observe_ready_strict, prepare_resident_mirror_observed,
    resident_file_identity_lease,
)
from market_analogues.types import InstrumentKey, SearchQuery, stable_hash

from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_10_wf03b_dtw_component_ladder as ladder


SCHEMA = "m04r14-t14-10-wf03-combined-batch-preregistration-v4"
OUTPUT_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf03-combined-batch-v4"
)
PREREGISTRATION_RELATIVE = Path(
    "experiments/m04r/m04r14_t14_10_wf03_combined_batch_v4_preregistered.json"
)
PACKED_SOURCE_STORE_RELATIVE = Path(
    "config/data/analogues/poc/m04r/packed-bound-full/store"
)
RESIDENT_RESERVE_BYTES = 1024 ** 3
BASELINE_VERIFICATION_RELATIVE = Path(
    "config/data/analogues/m04r14/"
    "t14-10-wf03-baseline-batch-v2-verification/VERIFIED.json"
)
TOP_K = 20
SEED_ROWS = 2_048
BLOCK_ROWS = 4_096
THREADS = 8
PRELOAD_WORKERS = 8
TOLERANCE = 1e-12
RUNTIME_FILES = (
    "experiments/m04r/m04r14_t14_10_wf03_combined_batch.py",
    "experiments/m04r/m04r14_t14_10_wf03_feasibility.py",
    "src/market_analogues/adapters.py",
    "src/market_analogues/component_search.py",
    "src/market_analogues/dtw_component_search.py",
    "src/market_analogues/dtw_interval_bound.py",
    "src/market_analogues/dtw_sample_store.py",
    "src/market_analogues/exact_batch.py",
    "src/market_analogues/packed_bound_search.py",
    "src/market_analogues/packed_bound_store.py",
    "src/market_analogues/representation.py",
    "src/market_analogues/resident_store.py",
    "src/market_analogues/search.py",
)


class CombinedBatchError(RuntimeError):
    pass


def _is_digest(value: Any) -> bool:
    return type(value) is str and len(value) == 64 \
        and set(value).issubset("0123456789abcdef")


def _is_attempt_id(value: Any) -> bool:
    if type(value) is not str or not value.startswith("attempt-"):
        return False
    try:
        ordinal = int(value.removeprefix("attempt-"))
    except ValueError:
        return False
    return ordinal >= 1 and value == f"attempt-{ordinal:04d}"


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repository, text=True, capture_output=True,
        check=False,
    )
    if result.returncode:
        raise CombinedBatchError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def _case_path(root: Path, query_id: str) -> Path:
    if len(query_id) != 24 \
            or any(value not in "0123456789abcdef" for value in query_id):
        raise CombinedBatchError("combined batch query ID differs")
    return root / f"{query_id}.json"


def _replace_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _file_identity(path: Path) -> dict[str, int | str]:
    value = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(value.st_mode):
        raise CombinedBatchError(f"regular immutable input required: {path}")
    return {
        "path": str(path.resolve()), "device": value.st_dev,
        "inode": value.st_ino, "size": value.st_size,
        "mtime_ns": value.st_mtime_ns, "ctime_ns": value.st_ctime_ns,
        "mode": value.st_mode,
    }


def _dtw_physical_identity(repository: Path) -> dict[str, Any]:
    root = repository / ladder.DTW_ROOT_RELATIVE / "generations" \
        / ladder.DTW_GENERATION_ID
    files = {
        name: _file_identity(root / name)
        for name in ("manifest.json", "dtw-samples.bin", "dtw-overflow-samples.bin")
    }
    return {"files": files, "digest": stable_hash(files)}


def _packed_semantic_identity(repository: Path) -> dict[str, Any]:
    root = repository / PACKED_SOURCE_STORE_RELATIVE / "generations" \
        / base.GENERATION_ID
    manifest_path = root / "manifest.json"
    manifest = base._read(manifest_path)
    names = {
        "manifest": {
            "bytes": manifest_path.stat().st_size,
            "sha256": base._sha(manifest_path),
        },
        "rows": {
            "bytes": int(manifest["rows_bytes"]),
            "sha256": str(manifest["rows_sha256"]),
        },
        "overflow": {
            "bytes": int(manifest["overflow_bytes"]),
            "sha256": str(manifest["overflow_sha256"]),
        },
    }
    content = {
        "schema_version": CONTENT_SCHEMA_VERSION,
        "generation_id": base.GENERATION_ID,
        "provenance_digest": str(manifest["provenance_digest"]),
        "manifest_digest": str(manifest["manifest_digest"]),
        "pack_contract_digest": str(manifest["pack_contract_digest"]),
        "quantized_bound_contract_digest": str(
            manifest["quantized_bound_contract_digest"]
        ),
        "physical_generation_bytes": sum(
            int(value["bytes"]) for value in names.values()
        ),
        "source_files": names,
        "mirror_files": names,
    }
    if content["generation_id"] != content["manifest_digest"] \
            or content["provenance_digest"] != base.PROVENANCE_DIGEST:
        raise CombinedBatchError("durable packed semantic identity differs")
    return {"content": content, "content_digest": stable_hash(content)}


def _dtw_semantic_identity(repository: Path) -> dict[str, Any]:
    root = repository / ladder.DTW_ROOT_RELATIVE / "generations" \
        / ladder.DTW_GENERATION_ID
    manifest_path = root / "manifest.json"
    manifest = base._read(manifest_path)
    state = {
        "manifest_sha256": base._sha(manifest_path),
        "manifest_digest": manifest["manifest_digest"],
        "provenance_digest": manifest["provenance_digest"],
        "contract_digest": manifest["contract_digest"],
        "rows_bytes": manifest["rows_bytes"],
        "rows_sha256": manifest["rows_sha256"],
        "overflow_bytes": manifest["overflow_bytes"],
        "overflow_sha256": manifest["overflow_sha256"],
        "packed_generation": manifest["packed_generation"],
    }
    if state["manifest_digest"] != ladder.DTW_GENERATION_ID \
            or state["packed_generation"]["manifest_digest"] != base.GENERATION_ID:
        raise CombinedBatchError("durable DTW semantic identity differs")
    return {"state": state, "digest": stable_hash(state)}


def _ensure_resident(
    repository: Path, expected_content_digest: str,
    trusted_attempts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    source_store = repository / PACKED_SOURCE_STORE_RELATIVE
    ready_path = base.RESIDENT_ROOT / "READY.json"
    generation_path = (
        base.RESIDENT_ROOT / "store" / "generations" / base.GENERATION_ID
    )
    validate_existing = (
        ready_path.exists() and not ready_path.is_symlink()
    ) or (generation_path.exists() and not generation_path.is_symlink())
    if ready_path.exists() and not ready_path.is_symlink():
        observed = observe_ready_strict(ready_path)
        lease = resident_file_identity_lease(ready_path)
        trusted = next((
            (attempt_id, attempt) for attempt_id, attempt in (
                trusted_attempts or {}
            ).items()
            if attempt.get("packed_content_digest") == expected_content_digest
            and attempt.get("resident_ready_digest") == observed["ready_digest"]
            and attempt.get("resident_attempt_lease_digest")
                == lease["lease_digest"]
        ), None)
        if trusted is not None and observed["content_digest"] \
                == lease["content_digest"] == expected_content_digest:
            validation = {
                "schema_version": "m04r14-resident-lease-reuse-observation-v1",
                "trusted_attempt_id": trusted[0],
                "content_digest": expected_content_digest,
                "ready_digest": observed["ready_digest"],
                "lease_digest": lease["lease_digest"],
            }
            validation["observation_digest"] = stable_hash(validation)
            return {
                "mode": "reused-fully-validated-unchanged-lease",
                "root": str(base.RESIDENT_ROOT.resolve()),
                "store_root": str((base.RESIDENT_ROOT / "store").resolve()),
                "content_digest": expected_content_digest,
                "ready_digest": observed["ready_digest"],
                "ready_file_sha256": observed["ready_file_sha256"],
                "lease": lease,
                "validation_observation": validation,
            }
    ready, observation = prepare_resident_mirror_observed(
        source_store, base.RESIDENT_ROOT, base.GENERATION_ID,
        expected_provenance_digest=base.PROVENANCE_DIGEST,
        reserve_bytes=RESIDENT_RESERVE_BYTES,
        validate_existing=validate_existing,
    )
    observed = observe_ready_strict(ready_path)
    lease = resident_file_identity_lease(ready_path)
    if ready["content_digest"] != expected_content_digest \
            or observed["content_digest"] != expected_content_digest \
            or lease["content_digest"] != expected_content_digest:
        raise CombinedBatchError("restored resident semantic content differs")
    return {
        "mode": (
            "validated-existing" if validate_existing
            else "restored-after-restart"
        ),
        "root": str(base.RESIDENT_ROOT.resolve()),
        "store_root": str((base.RESIDENT_ROOT / "store").resolve()),
        "content_digest": expected_content_digest,
        "ready_digest": observed["ready_digest"],
        "ready_file_sha256": observed["ready_file_sha256"],
        "lease": lease,
        "validation_observation": observation,
    }


def _verified_inputs(repository: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    dtw_result, dtw_verification = ladder._validate_upstream(repository)
    baseline_verification = base._read(repository / BASELINE_VERIFICATION_RELATIVE)
    base._validate_seal(baseline_verification, "verification_digest")
    if not all((
        baseline_verification.get("passed") is True,
        baseline_verification.get("queries_verified") == 3_936,
        baseline_verification.get("baseline_batch_complete") is True,
        baseline_verification.get("outcomes_or_labels_used") is False,
        baseline_verification.get(
            "historical_walk_forward_query_outcomes_opened"
        ) is False,
        baseline_verification.get("final_period_result_opened") is False,
    )):
        raise CombinedBatchError("verified baseline prerequisite differs")
    return dtw_result, dtw_verification


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain"):
        raise CombinedBatchError("combined batch preregistration requires clean commit")
    output = repository / OUTPUT_RELATIVE
    if output.exists() or output.is_symlink():
        raise CombinedBatchError("combined batch output must be absent before freeze")
    registry, _by_id = base._registry(repository)
    dtw_result, dtw_verification = _verified_inputs(repository)
    baseline_verification = base._read(repository / BASELINE_VERIFICATION_RELATIVE)
    packed_semantic = _packed_semantic_identity(repository)
    dtw_semantic = _dtw_semantic_identity(repository)
    rows = registry["queries_data"]
    head = _git(repository, "rev-parse", "HEAD")
    state = {
        "schema_version": SCHEMA,
        "status": "frozen_before_all_query_combined_retrieval",
        "implementation_commit": head,
        "runtime_files": {
            path: base._sha(repository / path) for path in RUNTIME_FILES
        },
        "inputs": {
            "registry_digest": registry["registry_digest"],
            "registry_sha256": base._sha(repository / base.REGISTRY_FILE),
            "query_ids_digest": stable_hash([
                row["episode_id"] for row in rows
            ]),
            "packed_generation_id": base.GENERATION_ID,
            "packed_provenance_digest": base.PROVENANCE_DIGEST,
            "packed_content_digest": packed_semantic["content_digest"],
            "dtw_generation_id": ladder.DTW_GENERATION_ID,
            "dtw_result_digest": dtw_result["result_digest"],
            "dtw_verification_digest": dtw_verification["verification_digest"],
            "dtw_verification_sha256": base._sha(
                repository / ladder.DTW_VERIFICATION_RELATIVE
            ),
            "dtw_semantic_identity_digest": dtw_semantic["digest"],
            "baseline_verification_digest": baseline_verification[
                "verification_digest"
            ],
            "baseline_verification_sha256": base._sha(
                repository / BASELINE_VERIFICATION_RELATIVE
            ),
        },
        "inventory": {
            "queries": len(rows),
            "scored_queries": sum(bool(row["scored"]) for row in rows),
            "warmup_queries": sum(not bool(row["scored"]) for row in rows),
            "months": len({row["cutoff"] for row in rows}),
        },
        "contract": staged_dtw_component_search_contract(adaptive_seed=True),
        "execution": {
            "query_concurrency": 1,
            "threads_per_query": THREADS,
            "preload_workers": PRELOAD_WORKERS,
            "source_cache_max_entries": None,
            "prepared_symbol_cache": "batch lifetime",
            "initial_seed_rows": SEED_ROWS,
            "seed_policy": "geometric stable-prefix expansion for top-k symbols",
            "maximum_seed_rows": "eligible candidates",
            "block_rows": BLOCK_ROWS,
            "top_k": TOP_K,
            "tolerance_hex": TOLERANCE.hex(),
            "case_publication": "create-only sealed query JSON",
            "progress_publication": "atomic after every completed query",
            "attempt_publication": "create-only per-process attempt receipts",
            "resident_restore": (
                "fully validate existing tmpfs mirror or reconstruct it from "
                "the fully verified durable packed generation"
            ),
            "resume": (
                "accept only fully validated sealed query receipts bound to "
                "durable semantic content; physical leases are attempt-local"
            ),
            "output_root": str(output.resolve()),
        },
        "gates": {
            "all_query_certificates_close": True,
            "twenty_distinct_symbols_per_query": True,
            "strict_bound_closure": True,
            "durable_packed_and_dtw_semantic_content_unchanged": True,
            "resident_and_dtw_physical_identity_unchanged_within_attempt": True,
            "zero_process_swap": True,
            "outcomes_or_labels_excluded": True,
            "independent_verification_required": True,
        },
        "claims": {
            "historical_query_retrieval_opened": True,
            "historical_walk_forward_query_outcomes_opened": False,
            "final_period_result_opened": False,
            "production_promotion_authorized": False,
        },
    }
    return base._sealed(state, "preregistration_digest")


def validate_preregistration(
    repository: Path, value: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    base._validate_seal(value, "preregistration_digest")
    registry, by_id = base._registry(repository)
    dtw_result, dtw_verification = _verified_inputs(repository)
    baseline_verification = base._read(repository / BASELINE_VERIFICATION_RELATIVE)
    packed_semantic = _packed_semantic_identity(repository)
    dtw_semantic = _dtw_semantic_identity(repository)
    rows = registry["queries_data"]
    expected = value.get("inputs", {})
    if not all((
        value.get("schema_version") == SCHEMA,
        value.get("status") == "frozen_before_all_query_combined_retrieval",
        expected.get("registry_digest") == registry["registry_digest"],
        expected.get("registry_sha256")
            == base._sha(repository / base.REGISTRY_FILE),
        expected.get("query_ids_digest") == stable_hash([
            row["episode_id"] for row in rows
        ]),
        expected.get("packed_generation_id") == base.GENERATION_ID,
        expected.get("packed_provenance_digest") == base.PROVENANCE_DIGEST,
        expected.get("packed_content_digest")
            == packed_semantic["content_digest"],
        expected.get("dtw_generation_id") == ladder.DTW_GENERATION_ID,
        expected.get("dtw_result_digest") == dtw_result["result_digest"],
        expected.get("dtw_verification_digest")
            == dtw_verification["verification_digest"],
        expected.get("dtw_verification_sha256")
            == base._sha(repository / ladder.DTW_VERIFICATION_RELATIVE),
        expected.get("dtw_semantic_identity_digest")
            == dtw_semantic["digest"],
        expected.get("baseline_verification_digest")
            == baseline_verification["verification_digest"],
        expected.get("baseline_verification_sha256")
            == base._sha(repository / BASELINE_VERIFICATION_RELATIVE),
        value.get("inventory") == {
            "queries": len(rows),
            "scored_queries": sum(bool(row["scored"]) for row in rows),
            "warmup_queries": sum(not bool(row["scored"]) for row in rows),
            "months": len({row["cutoff"] for row in rows}),
        },
        value.get("execution", {}).get("query_concurrency") == 1,
        value.get("execution", {}).get("threads_per_query") == THREADS,
        value.get("execution", {}).get("preload_workers") == PRELOAD_WORKERS,
        value.get("execution", {}).get("initial_seed_rows") == SEED_ROWS,
        value.get("execution", {}).get("seed_policy")
            == "geometric stable-prefix expansion for top-k symbols",
        value.get("execution", {}).get("maximum_seed_rows")
            == "eligible candidates",
        value.get("execution", {}).get("block_rows") == BLOCK_ROWS,
        value.get("execution", {}).get("top_k") == TOP_K,
        value.get("execution", {}).get("tolerance_hex") == TOLERANCE.hex(),
        value.get("execution", {}).get("output_root")
            == str((repository / OUTPUT_RELATIVE).resolve()),
        value.get("contract")
            == staged_dtw_component_search_contract(adaptive_seed=True),
        set(value.get("runtime_files", {})) == set(RUNTIME_FILES),
    )):
        raise CombinedBatchError("combined batch preregistration differs")
    commit = value.get("implementation_commit")
    if type(commit) is not str:
        raise CombinedBatchError("combined batch implementation commit differs")
    _git(repository, "merge-base", "--is-ancestor", commit, "HEAD")
    for path, digest in value.get("runtime_files", {}).items():
        if type(path) is not str or type(digest) is not str \
                or base._sha(repository / path) != digest:
            raise CombinedBatchError(f"combined batch runtime differs: {path}")
        blob = subprocess.run(
            ["git", "show", f"{commit}:{path}"], cwd=repository,
            capture_output=True, check=False,
        )
        if blob.returncode or sha256(blob.stdout).hexdigest() != digest:
            raise CombinedBatchError(f"combined batch Git binding differs: {path}")
    return registry, by_id


def _matches(result: Any) -> list[dict[str, Any]]:
    return [{
        "episode_id": row.episode_key.id,
        "symbol": row.episode_key.instrument.source_symbol,
        "cutoff": row.episode_key.cutoff.isoformat(),
        "distance_hex": row.total_distance.hex(),
        "quality_tier": row.quality_tier,
    } for row in result.matches]


def _case_semantic_state(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "query_id": value["query_id"],
        "case_id": value["case_id"],
        "packed_generation_id": value["packed_generation_id"],
        "packed_content_digest": value["packed_content_digest"],
        "dtw_generation_id": value["dtw_generation_id"],
        "dtw_semantic_identity_digest": value[
            "dtw_semantic_identity_digest"
        ],
        "certificate_result_digest": value["certificate"]["result_digest"],
        "matches": value["matches"],
    }


def _certificate_result_digest(
    certificate: Mapping[str, Any], matches: Sequence[Mapping[str, Any]],
) -> str:
    deterministic = {
        "schema_version": certificate["schema_version"],
        "contract_digest": certificate["contract_digest"],
        "packed_generation_id": certificate["packed_generation_id"],
        "dtw_generation_id": certificate["dtw_generation_id"],
        "query_episode_id": certificate["query_episode_id"],
        "input_digest": certificate["input_digest"],
        "eligible_candidates": certificate["eligible_candidates"],
        "seed_rows": certificate["seed_rows"],
        "rigid_bound_evaluated": certificate["rigid_bound_evaluated"],
        "rigid_bound_admitted": certificate["rigid_bound_admitted"],
        "dtw_bound_evaluated": certificate["dtw_bound_evaluated"],
        "combined_bound_admitted": certificate["combined_bound_admitted"],
        "exact_evaluated": certificate["exact_evaluated"],
        "native_bound_pruned": certificate["native_bound_pruned"],
        "seed_threshold_hex": certificate["seed_threshold"].hex(),
        "final_threshold_hex": certificate["final_threshold"].hex(),
        "minimum_rigid_pruned_hex": (
            certificate["minimum_rigid_pruned"].hex()
            if certificate["minimum_rigid_pruned"] is not None else None
        ),
        "minimum_combined_pruned_hex": (
            certificate["minimum_combined_pruned"].hex()
            if certificate["minimum_combined_pruned"] is not None else None
        ),
        "maximum_bound_excess_hex": certificate["maximum_bound_excess"].hex(),
        "matches": [{
            "episode_id": row["episode_id"],
            "distance_hex": row["distance_hex"],
        } for row in matches],
        "outcomes_or_labels_used": False,
    }
    return stable_hash(deterministic)


def _validate_case(
    value: Mapping[str, Any], row: Mapping[str, Any],
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        base._validate_seal(value, "case_digest")
        certificate = value["certificate"]
        matches = value["matches"]
        distances = [float.fromhex(item["distance_hex"]) for item in matches]
        minimum_rigid = certificate["minimum_rigid_pruned"]
        minimum_combined = certificate["minimum_combined_pruned"]
        valid = all((
            value["schema_version"] == "m04r14-wf03-combined-batch-case-v4",
            value["status"] == "complete",
            value["query_id"] == row["episode_id"],
            value["case_id"] == row["case_id"],
            value["symbol"] == row["symbol"],
            value["cutoff"] == row["cutoff"],
            value["fold_id"] == row["fold_id"],
            value["fold_role"] == row["fold_role"],
            value["scored"] is bool(row["scored"]),
            value["preregistration_digest"]
                == preregistration["preregistration_digest"],
            value["packed_generation_id"] == base.GENERATION_ID,
            value["packed_content_digest"]
                == preregistration["inputs"]["packed_content_digest"],
            value["dtw_generation_id"] == ladder.DTW_GENERATION_ID,
            value["dtw_semantic_identity_digest"]
                == preregistration["inputs"]["dtw_semantic_identity_digest"],
            _is_attempt_id(value["attempt_id"]),
            _is_digest(value["resident_attempt_lease_digest"]),
            _is_digest(value["dtw_attempt_identity_digest"]),
            value["contract_digest"] == preregistration["contract"]["digest"],
            value["outcomes_or_labels_used"] is False,
            value["historical_walk_forward_query_outcomes_opened"] is False,
            value["final_period_result_opened"] is False,
            certificate["schema_version"]
                == preregistration["contract"]["schema_version"],
            certificate["query_episode_id"] == row["episode_id"],
            certificate["packed_generation_id"] == base.GENERATION_ID,
            certificate["dtw_generation_id"] == ladder.DTW_GENERATION_ID,
            certificate["contract_digest"] == preregistration["contract"]["digest"],
            certificate["seed_rows"] >= TOP_K,
            certificate["rigid_bound_evaluated"]
                == certificate["eligible_candidates"],
            certificate["rigid_bound_admitted"]
                <= certificate["rigid_bound_evaluated"],
            certificate["dtw_bound_evaluated"]
                <= certificate["rigid_bound_admitted"],
            certificate["combined_bound_admitted"]
                <= certificate["rigid_bound_admitted"],
            certificate["exact_evaluated"] >= certificate["seed_rows"],
            certificate["maximum_bound_excess"] <= TOLERANCE,
            certificate["final_threshold"] <= certificate["seed_threshold"]
                + TOLERANCE,
            minimum_rigid is None
                or minimum_rigid > certificate["seed_threshold"],
            minimum_combined is None
                or minimum_combined > certificate["seed_threshold"],
            type(matches) is list and len(matches) == TOP_K,
            len({item["symbol"] for item in matches}) == TOP_K,
            len({item["episode_id"] for item in matches}) == TOP_K,
            distances == sorted(distances),
            np.isfinite(distances).all(),
            distances[-1] == certificate["final_threshold"],
            all(item["quality_tier"] in ("A", "B") for item in matches),
            certificate["result_digest"]
                == _certificate_result_digest(certificate, matches),
            value["semantic_digest"] == stable_hash(_case_semantic_state(value)),
        ))
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise CombinedBatchError(f"combined batch case differs: {row['episode_id']}")
    return dict(value)


def _existing_case(
    path: Path, row: Mapping[str, Any], preregistration: Mapping[str, Any],
) -> dict[str, Any] | None:
    if path.is_symlink():
        raise CombinedBatchError("combined batch case is linked or non-regular")
    if not path.exists():
        return None
    if not path.is_file():
        raise CombinedBatchError("combined batch case is linked or non-regular")
    return _validate_case(base._read(path), row, preregistration)


def _next_attempt(root: Path) -> Path:
    attempts = root / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    if attempts.is_symlink():
        raise CombinedBatchError("combined batch attempts root is linked")
    ordinals = []
    for path in attempts.iterdir():
        if path.is_symlink() or not path.is_dir() \
                or not path.name.startswith("attempt-"):
            raise CombinedBatchError("malformed combined batch attempt")
        try:
            ordinal = int(path.name.removeprefix("attempt-"))
        except ValueError as exc:
            raise CombinedBatchError("malformed combined batch attempt") from exc
        if ordinal < 1 or path.name != f"attempt-{ordinal:04d}":
            raise CombinedBatchError("malformed combined batch attempt")
        ordinals.append(ordinal)
    target = attempts / f"attempt-{max(ordinals, default=0) + 1:04d}"
    target.mkdir()
    return target


def _attempt_history(
    root: Path, preregistration: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    attempts = root / "attempts"
    if not attempts.exists():
        return {}
    if attempts.is_symlink() or not attempts.is_dir():
        raise CombinedBatchError("combined batch attempts root differs")
    output = {}
    for path in attempts.iterdir():
        if path.is_symlink() or not path.is_dir() or not _is_attempt_id(path.name):
            raise CombinedBatchError("malformed combined batch attempt")
        names = {value.name for value in path.iterdir()}
        if not names:
            continue
        unexpected = names - {
            "RUN_STARTED.json", "INTERRUPTED.json", "COMPLETE.json",
        }
        if unexpected and not all(name.startswith(".wf03-") for name in unexpected):
            raise CombinedBatchError("combined batch attempt files differ")
        if "RUN_STARTED.json" not in names:
            if unexpected == names:
                continue
            raise CombinedBatchError("combined batch attempt start is absent")
        started = base._read(path / "RUN_STARTED.json")
        base._validate_seal(started, "attempt_digest")
        if not all((
            started.get("schema_version")
                == "m04r14-wf03-combined-batch-attempt-v4",
            started.get("status") == "running",
            started.get("attempt_id") == path.name,
            started.get("preregistration_digest")
                == preregistration["preregistration_digest"],
            started.get("packed_content_digest")
                == preregistration["inputs"]["packed_content_digest"],
            started.get("dtw_semantic_identity_digest")
                == preregistration["inputs"]["dtw_semantic_identity_digest"],
            _is_digest(started.get("resident_attempt_lease_digest")),
            _is_digest(started.get("dtw_attempt_identity_digest")),
        )):
            raise CombinedBatchError("combined batch attempt start differs")
        terminals = names & {"INTERRUPTED.json", "COMPLETE.json"}
        if len(terminals) > 1:
            raise CombinedBatchError("combined batch attempt terminals differ")
        if terminals:
            terminal_name = next(iter(terminals))
            terminal = base._read(path / terminal_name)
            base._validate_seal(terminal, "attempt_digest")
            if terminal.get("attempt_id") != path.name or terminal.get("status") \
                    != ("complete" if terminal_name == "COMPLETE.json"
                        else "interrupted"):
                raise CombinedBatchError("combined batch attempt terminal differs")
        output[path.name] = started
    return output


def _validate_case_attempt(
    value: Mapping[str, Any], attempts: Mapping[str, Mapping[str, Any]],
) -> None:
    attempt = attempts.get(str(value.get("attempt_id")))
    if attempt is None or value.get("resident_attempt_lease_digest") \
            != attempt.get("resident_attempt_lease_digest") \
            or value.get("dtw_attempt_identity_digest") \
            != attempt.get("dtw_attempt_identity_digest"):
        raise CombinedBatchError("combined batch case attempt binding differs")


def _run_case(
    row: Mapping[str, Any], *, source: CachedOHLCVSource,
    prepared_symbols: dict[str, Any], packed_root: Path, dtw_root: Path,
    packed_content_digest: str, resident_lease_digest: str,
    dtw_identity_digest: str, attempt_id: str,
    repository: Path, preregistration: Mapping[str, Any], cases_root: Path,
) -> dict[str, Any]:
    path = _case_path(cases_root, str(row["episode_id"]))
    existing = _existing_case(path, row, preregistration)
    if existing is not None:
        return existing
    started = perf_counter()
    episode = build_episode(
        source, InstrumentKey("nasdaq", str(row["symbol"])), str(row["cutoff"]),
        int(row["lookback"]), str(row["representation_version"]),
    )
    if episode.key.id != row["episode_id"]:
        raise CombinedBatchError("combined batch query reconstruction differs")
    request = SearchQuery(
        episode.key, ("nasdaq",), ("A", "B"), TOP_K,
        False, True, base.MAX_PER_INSTRUMENT, base.MINIMUM_HISTORY_GAP,
    )
    result = certified_staged_dtw_component_search(
        episode, source, request, packed_root, base.GENERATION_ID,
        dtw_root, ladder.DTW_GENERATION_ID, store_dataset_id="nasdaq",
        seed_rows=SEED_ROWS, block_rows=BLOCK_ROWS,
        rigid_threads=THREADS, dtw_threads=THREADS, exact_workers=THREADS,
        tolerance=TOLERANCE, verify_content=False,
        prepared_symbol_cache=prepared_symbols,
        adaptive_seed=True,
    )
    current_lease = resident_file_identity_lease(
        base.RESIDENT_ROOT / "READY.json"
    )
    if current_lease["content_digest"] != packed_content_digest \
            or current_lease["lease_digest"] != resident_lease_digest \
            or _dtw_physical_identity(repository)["digest"] != dtw_identity_digest:
        raise CombinedBatchError("combined batch attempt input identity changed")
    swap_kib = int(
        Path("/proc/self/status").read_text().split("VmSwap:")[1].split()[0]
    )
    if swap_kib != 0:
        raise CombinedBatchError("combined batch process used swap")
    matches = _matches(result)
    certificate = asdict(result.certificate)
    state = {
        "schema_version": "m04r14-wf03-combined-batch-case-v4",
        "status": "complete",
        "case_id": row["case_id"],
        "query_id": row["episode_id"],
        "symbol": row["symbol"],
        "cutoff": row["cutoff"],
        "fold_id": row["fold_id"],
        "fold_role": row["fold_role"],
        "scored": bool(row["scored"]),
        "preregistration_digest": preregistration["preregistration_digest"],
        "packed_generation_id": base.GENERATION_ID,
        "packed_content_digest": packed_content_digest,
        "dtw_generation_id": ladder.DTW_GENERATION_ID,
        "dtw_semantic_identity_digest": preregistration["inputs"][
            "dtw_semantic_identity_digest"
        ],
        "attempt_id": attempt_id,
        "resident_attempt_lease_digest": resident_lease_digest,
        "dtw_attempt_identity_digest": dtw_identity_digest,
        "contract_digest": preregistration["contract"]["digest"],
        "certificate": certificate,
        "matches": matches,
        "elapsed_seconds": perf_counter() - started,
        "process_swap_kib": swap_kib,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "source_cache_state": source.cache_state(),
        "prepared_symbols": len(prepared_symbols),
        "outcomes_or_labels_used": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
    }
    state["semantic_digest"] = stable_hash(_case_semantic_state(state))
    sealed = base._sealed(state, "case_digest")
    _validate_case(sealed, row, preregistration)
    base._atomic(path, sealed)
    return sealed


def _execute_locked(
    repository: Path, preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    started = perf_counter()
    repository = repository.resolve(strict=True)
    registry, _by_id = validate_preregistration(repository, preregistration)
    root = repository / OUTPUT_RELATIVE
    if root.is_symlink() or root.exists() and not root.is_dir():
        raise CombinedBatchError("combined batch output path differs")
    if root.exists():
        contract_path = root / "CONTRACT.json"
        if not contract_path.exists() and not any(root.iterdir()):
            base._atomic(contract_path, preregistration)
        if base._read(contract_path) != preregistration:
            raise CombinedBatchError("combined batch resume contract differs")
        if (root / "RESULT.json").exists():
            result = base._read(root / "RESULT.json")
            base._validate_seal(result)
            if not all((
                result.get("schema_version")
                    == "m04r14-t14-10-wf03-combined-batch-result-v4",
                result.get("passed") is True,
                result.get("queries") == 3_936,
                result.get("preregistration_digest")
                    == preregistration["preregistration_digest"],
                result.get("packed_content_digest")
                    == preregistration["inputs"]["packed_content_digest"],
            )):
                raise CombinedBatchError("combined batch terminal differs")
            return result
    else:
        root.mkdir(parents=True)
        base._atomic(root / "CONTRACT.json", preregistration)
    cases_root = root / "cases"
    cases_root.mkdir(exist_ok=True)
    if cases_root.is_symlink():
        raise CombinedBatchError("combined batch cases root is linked")
    rows = registry["queries_data"]
    existing_results = []
    attempt_history = _attempt_history(root, preregistration)
    for row in rows:
        value = _existing_case(
            _case_path(cases_root, str(row["episode_id"])), row,
            preregistration,
        )
        if value is not None:
            _validate_case_attempt(value, attempt_history)
            existing_results.append(value)
    resident = _ensure_resident(
        repository, preregistration["inputs"]["packed_content_digest"],
        attempt_history,
    )
    resident_lease_digest = resident["lease"]["lease_digest"]
    packed_root = Path(resident["store_root"])
    dtw_root = repository / ladder.DTW_ROOT_RELATIVE
    dtw_identity_before = _dtw_physical_identity(repository)
    unchanged_resident = (
        resident["mode"] == "reused-fully-validated-unchanged-lease"
    )
    trusted_dtw = any(
        attempt.get("dtw_semantic_identity_digest")
            == preregistration["inputs"]["dtw_semantic_identity_digest"]
        and attempt.get("dtw_attempt_identity_digest")
            == dtw_identity_before["digest"]
        for attempt in attempt_history.values()
    )
    packed = load_packed_generation(
        packed_root, base.GENERATION_ID,
        expected_provenance_digest=base.PROVENANCE_DIGEST,
        verify_content=False, validate_records=not unchanged_resident,
    )
    load_dtw_sample_generation(
        dtw_root, ladder.DTW_GENERATION_ID,
        packed_manifest=packed.manifest, verify_content=not trusted_dtw,
        validate_records=not trusted_dtw,
    )
    current_lease = resident_file_identity_lease(
        base.RESIDENT_ROOT / "READY.json"
    )
    dtw_identity_after = _dtw_physical_identity(repository)
    if current_lease["lease_digest"] != resident_lease_digest \
            or dtw_identity_after["digest"] != dtw_identity_before["digest"]:
        raise CombinedBatchError("combined batch startup input identity changed")
    dtw_identity_digest = dtw_identity_after["digest"]
    attempt = _next_attempt(root)
    attempt_id = attempt.name
    base._atomic(attempt / "RUN_STARTED.json", base._sealed({
        "schema_version": "m04r14-wf03-combined-batch-attempt-v4",
        "status": "running", "attempt_id": attempt_id,
        "preregistration_digest": preregistration["preregistration_digest"],
        "packed_content_digest": resident["content_digest"],
        "resident_mode": resident["mode"],
        "dtw_validation_mode": (
            "reused-fully-validated-unchanged-identity" if trusted_dtw
            else "full-content-and-record-validation"
        ),
        "resident_ready_digest": resident["ready_digest"],
        "resident_attempt_lease_digest": resident_lease_digest,
        "resident_validation_observation_digest": resident[
            "validation_observation"
        ]["observation_digest"],
        "dtw_semantic_identity_digest": preregistration["inputs"][
            "dtw_semantic_identity_digest"
        ],
        "dtw_attempt_identity_digest": dtw_identity_digest,
        "receipts_reused_at_start": len(existing_results),
        "created_at": base._now(),
    }, "attempt_digest"))
    results = []
    current_row: Mapping[str, Any] | None = None
    try:
        raw_source = source_from_spec(
            load_config(repository / base.CONFIG_RELATIVE).datasets["nasdaq"]
        )
        source = CachedOHLCVSource(raw_source, max_entries=None)
        preload_started = perf_counter()
        source.preload(tuple(raw_source.instruments()), workers=PRELOAD_WORKERS)
        preload_seconds = perf_counter() - preload_started
        prepared_symbols: dict[str, Any] = {}
        for completed, row in enumerate(rows, start=1):
            current_row = row
            value = _run_case(
                row, source=source, prepared_symbols=prepared_symbols,
                packed_root=packed_root, dtw_root=dtw_root,
                packed_content_digest=resident["content_digest"],
                resident_lease_digest=resident_lease_digest,
                dtw_identity_digest=dtw_identity_digest,
                attempt_id=attempt_id,
                repository=repository, preregistration=preregistration,
                cases_root=cases_root,
            )
            results.append(value)
            _replace_json(root / "PROGRESS.json", {
                "schema_version": "m04r14-wf03-combined-batch-progress-v4",
                "status": "running" if completed < len(rows) else "publishing",
                "attempt_id": attempt_id,
                "completed_queries": completed,
                "total_queries": len(rows),
                "completed_month_equivalents": completed // 24,
                "last_query_id": value["query_id"],
                "last_case_digest": value["case_digest"],
                "prepared_symbols": len(prepared_symbols),
                "source_cache_state": source.cache_state(),
                "elapsed_seconds": perf_counter() - started,
            })
    except BaseException as exc:
        _replace_json(root / "PROGRESS.json", {
            "schema_version": "m04r14-wf03-combined-batch-progress-v4",
            "status": "interrupted", "attempt_id": attempt_id,
            "completed_queries": len(results),
            "total_queries": len(rows),
            "next_query_id": (
                current_row["episode_id"] if current_row is not None else None
            ),
            "error_type": type(exc).__name__, "error": str(exc),
        })
        base._atomic(attempt / "INTERRUPTED.json", base._sealed({
            "schema_version": "m04r14-wf03-combined-batch-attempt-v4",
            "status": "interrupted", "attempt_id": attempt_id,
            "completed_queries": len(results),
            "error_type": type(exc).__name__, "error": str(exc),
            "created_at": base._now(),
        }, "attempt_digest"))
        raise
    if [value["query_id"] for value in results] != [
        row["episode_id"] for row in rows
    ]:
        raise CombinedBatchError("combined batch result order differs")
    case_manifest = [{
        "query_id": value["query_id"],
        "case_digest": value["case_digest"],
        "sha256": base._sha(_case_path(cases_root, value["query_id"])),
    } for value in results]
    state = {
        "schema_version": "m04r14-t14-10-wf03-combined-batch-result-v4",
        "status": "complete",
        "passed": True,
        "queries": len(results),
        "scored_queries": sum(bool(value["scored"]) for value in results),
        "warmup_queries": sum(not bool(value["scored"]) for value in results),
        "months": preregistration["inventory"]["months"],
        "preregistration_digest": preregistration["preregistration_digest"],
        "packed_generation_id": base.GENERATION_ID,
        "packed_content_digest": resident["content_digest"],
        "dtw_generation_id": ladder.DTW_GENERATION_ID,
        "case_manifest_digest": stable_hash(case_manifest),
        "case_semantic_digest": stable_hash([
            value["semantic_digest"] for value in results
        ]),
        "minimum_eligible_candidates": min(
            value["certificate"]["eligible_candidates"] for value in results
        ),
        "maximum_eligible_candidates": max(
            value["certificate"]["eligible_candidates"] for value in results
        ),
        "preload_seconds": preload_seconds,
        "elapsed_seconds": perf_counter() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "source_cache_state": source.cache_state(),
        "prepared_symbols": len(prepared_symbols),
        "outcomes_or_labels_used": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
        "independent_verification_authorized": True,
        "terminal_attempt_id": attempt_id,
        "attempts": len(tuple((root / "attempts").iterdir())),
    }
    result = base._sealed(state)
    base._atomic(root / "RESULT.json", result)
    base._atomic(attempt / "COMPLETE.json", base._sealed({
        "schema_version": "m04r14-wf03-combined-batch-attempt-v4",
        "status": "complete", "attempt_id": attempt_id,
        "queries": len(results),
        "receipts_reused_at_start": len(existing_results),
        "receipts_computed": len(results) - len(existing_results),
        "result_digest": result["result_digest"],
        "created_at": base._now(),
    }, "attempt_digest"))
    _replace_json(root / "PROGRESS.json", {
        "schema_version": "m04r14-wf03-combined-batch-progress-v4",
        "status": "complete", "completed_queries": len(results),
        "attempt_id": attempt_id,
        "total_queries": len(results),
        "completed_month_equivalents": preregistration["inventory"]["months"],
        "result_digest": result["result_digest"],
    })
    return result


def execute(repository: Path, preregistration: Mapping[str, Any]) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    output = repository / OUTPUT_RELATIVE
    lock_path = output.with_name(f"{output.name}.lock")
    if lock_path.is_symlink():
        raise CombinedBatchError("combined batch producer lock is linked")
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise CombinedBatchError("combined batch producer lock is not regular")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CombinedBatchError(
                "another combined batch producer is already running"
            ) from exc
        try:
            return _execute_locked(repository, preregistration)
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preregister", "run"))
    parser.add_argument("--repository", required=True, type=Path)
    args = parser.parse_args(argv)
    repository = args.repository.resolve(strict=True)
    path = repository / PREREGISTRATION_RELATIVE
    if args.mode == "preregister":
        base._atomic(path, build_preregistration(repository))
        return 0
    result = execute(repository, base._read(path))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
