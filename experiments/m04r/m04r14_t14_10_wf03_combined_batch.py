"""Preregister and run certified combined retrieval for all 3,936 WF-03 queries."""
from __future__ import annotations

import argparse
from dataclasses import asdict
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
from market_analogues.episodes import build_episode
from market_analogues.types import InstrumentKey, SearchQuery, stable_hash

from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_10_wf03b_dtw_component_ladder as ladder


SCHEMA = "m04r14-t14-10-wf03-combined-batch-preregistration-v2"
OUTPUT_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf03-combined-batch-v2"
)
PREREGISTRATION_RELATIVE = Path(
    "experiments/m04r/m04r14_t14_10_wf03_combined_batch_v2_preregistered.json"
)
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
    "src/market_analogues/search.py",
)


class CombinedBatchError(RuntimeError):
    pass


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


def _dtw_identity(repository: Path) -> dict[str, Any]:
    root = repository / ladder.DTW_ROOT_RELATIVE / "generations" \
        / ladder.DTW_GENERATION_ID
    files = {
        name: _file_identity(root / name)
        for name in ("manifest.json", "dtw-samples.bin", "dtw-overflow-samples.bin")
    }
    return {"files": files, "digest": stable_hash(files)}


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
    resident = base._resident()
    dtw_identity = _dtw_identity(repository)
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
            "resident_content_digest": resident["content_digest"],
            "resident_ready_digest": resident["ready_digest"],
            "dtw_generation_id": ladder.DTW_GENERATION_ID,
            "dtw_result_digest": dtw_result["result_digest"],
            "dtw_verification_digest": dtw_verification["verification_digest"],
            "dtw_verification_sha256": base._sha(
                repository / ladder.DTW_VERIFICATION_RELATIVE
            ),
            "dtw_file_identity_digest": dtw_identity["digest"],
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
        "contract": staged_dtw_component_search_contract(),
        "execution": {
            "query_concurrency": 1,
            "threads_per_query": THREADS,
            "preload_workers": PRELOAD_WORKERS,
            "source_cache_max_entries": None,
            "prepared_symbol_cache": "batch lifetime",
            "seed_rows": SEED_ROWS,
            "block_rows": BLOCK_ROWS,
            "top_k": TOP_K,
            "tolerance_hex": TOLERANCE.hex(),
            "case_publication": "create-only sealed query JSON",
            "progress_publication": "atomic after every completed query",
            "resume": "accept only fully validated sealed query receipts",
            "output_root": str(output.resolve()),
        },
        "gates": {
            "all_query_certificates_close": True,
            "twenty_distinct_symbols_per_query": True,
            "strict_bound_closure": True,
            "resident_and_dtw_file_identity_unchanged": True,
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
    resident = base._resident()
    rows = registry["queries_data"]
    expected = value.get("inputs", {})
    if not all((
        value.get("schema_version") == SCHEMA,
        value.get("status") == "frozen_before_all_query_combined_retrieval",
        expected.get("registry_digest") == registry["registry_digest"],
        expected.get("query_ids_digest") == stable_hash([
            row["episode_id"] for row in rows
        ]),
        expected.get("packed_generation_id") == base.GENERATION_ID,
        expected.get("packed_provenance_digest") == base.PROVENANCE_DIGEST,
        expected.get("resident_content_digest") == resident["content_digest"],
        expected.get("resident_ready_digest") == resident["ready_digest"],
        expected.get("dtw_generation_id") == ladder.DTW_GENERATION_ID,
        expected.get("dtw_result_digest") == dtw_result["result_digest"],
        expected.get("dtw_verification_digest")
            == dtw_verification["verification_digest"],
        expected.get("dtw_file_identity_digest")
            == _dtw_identity(repository)["digest"],
        expected.get("baseline_verification_digest")
            == baseline_verification["verification_digest"],
        value.get("execution", {}).get("query_concurrency") == 1,
        value.get("execution", {}).get("threads_per_query") == THREADS,
        value.get("execution", {}).get("seed_rows") == SEED_ROWS,
        value.get("contract") == staged_dtw_component_search_contract(),
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
        "dtw_generation_id": value["dtw_generation_id"],
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
            value["schema_version"] == "m04r14-wf03-combined-batch-case-v2",
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
            value["dtw_generation_id"] == ladder.DTW_GENERATION_ID,
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
    if not path.exists():
        return None
    return _validate_case(base._read(path), row, preregistration)


def _run_case(
    row: Mapping[str, Any], *, source: CachedOHLCVSource,
    prepared_symbols: dict[str, Any], packed_root: Path, dtw_root: Path,
    resident_lease_digest: str, dtw_identity_digest: str,
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
    )
    current_lease = base.resident_file_identity_lease(
        base.RESIDENT_ROOT / "READY.json"
    )
    if current_lease["lease_digest"] != resident_lease_digest \
            or _dtw_identity(repository)["digest"] != dtw_identity_digest:
        raise CombinedBatchError("combined batch immutable input identity changed")
    swap_kib = int(
        Path("/proc/self/status").read_text().split("VmSwap:")[1].split()[0]
    )
    if swap_kib != 0:
        raise CombinedBatchError("combined batch process used swap")
    matches = _matches(result)
    certificate = asdict(result.certificate)
    state = {
        "schema_version": "m04r14-wf03-combined-batch-case-v2",
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
        "dtw_generation_id": ladder.DTW_GENERATION_ID,
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


def execute(repository: Path, preregistration: Mapping[str, Any]) -> dict[str, Any]:
    started = perf_counter()
    repository = repository.resolve(strict=True)
    registry, _by_id = validate_preregistration(repository, preregistration)
    root = repository / OUTPUT_RELATIVE
    if root.is_symlink() or root.exists() and not root.is_dir():
        raise CombinedBatchError("combined batch output path differs")
    if root.exists():
        if base._read(root / "CONTRACT.json") != preregistration:
            raise CombinedBatchError("combined batch resume contract differs")
        if (root / "RESULT.json").exists():
            result = base._read(root / "RESULT.json")
            base._validate_seal(result)
            if result.get("passed") is not True or result.get("queries") != 3_936:
                raise CombinedBatchError("combined batch terminal differs")
            return result
    else:
        root.mkdir(parents=True)
        base._atomic(root / "CONTRACT.json", preregistration)
        base._atomic(root / "RUN_STARTED.json", base._sealed({
            "schema_version": "m04r14-wf03-combined-batch-run-v2",
            "status": "running",
            "preregistration_digest": preregistration["preregistration_digest"],
            "created_at": base._now(),
        }))
    cases_root = root / "cases"
    cases_root.mkdir(exist_ok=True)
    resident = base._resident()
    resident_lease_digest = resident["lease"]["lease_digest"]
    dtw_identity_digest = _dtw_identity(repository)["digest"]
    packed_root = Path(resident["store_root"])
    dtw_root = repository / ladder.DTW_ROOT_RELATIVE
    raw_source = source_from_spec(
        load_config(repository / base.CONFIG_RELATIVE).datasets["nasdaq"]
    )
    source = CachedOHLCVSource(raw_source, max_entries=None)
    preload_started = perf_counter()
    source.preload(tuple(raw_source.instruments()), workers=PRELOAD_WORKERS)
    preload_seconds = perf_counter() - preload_started
    prepared_symbols: dict[str, Any] = {}
    rows = registry["queries_data"]
    results = []
    for completed, row in enumerate(rows, start=1):
        try:
            value = _run_case(
                row, source=source, prepared_symbols=prepared_symbols,
                packed_root=packed_root, dtw_root=dtw_root,
                resident_lease_digest=resident_lease_digest,
                dtw_identity_digest=dtw_identity_digest,
                repository=repository, preregistration=preregistration,
                cases_root=cases_root,
            )
        except Exception as exc:
            _replace_json(root / "PROGRESS.json", {
                "schema_version": "m04r14-wf03-combined-batch-progress-v2",
                "status": "interrupted",
                "completed_queries": len(results),
                "total_queries": len(rows),
                "next_query_id": row["episode_id"],
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            raise
        results.append(value)
        _replace_json(root / "PROGRESS.json", {
            "schema_version": "m04r14-wf03-combined-batch-progress-v2",
            "status": "running" if completed < len(rows) else "publishing",
            "completed_queries": completed,
            "total_queries": len(rows),
            "completed_month_equivalents": completed // 24,
            "last_query_id": value["query_id"],
            "last_case_digest": value["case_digest"],
            "prepared_symbols": len(prepared_symbols),
            "source_cache_state": source.cache_state(),
            "elapsed_seconds": perf_counter() - started,
        })
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
        "schema_version": "m04r14-t14-10-wf03-combined-batch-result-v2",
        "status": "complete",
        "passed": True,
        "queries": len(results),
        "scored_queries": sum(bool(value["scored"]) for value in results),
        "warmup_queries": sum(not bool(value["scored"]) for value in results),
        "months": preregistration["inventory"]["months"],
        "preregistration_digest": preregistration["preregistration_digest"],
        "packed_generation_id": base.GENERATION_ID,
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
    }
    result = base._sealed(state)
    base._atomic(root / "RESULT.json", result)
    _replace_json(root / "PROGRESS.json", {
        "schema_version": "m04r14-wf03-combined-batch-progress-v2",
        "status": "complete", "completed_queries": len(results),
        "total_queries": len(results),
        "completed_month_equivalents": preregistration["inventory"]["months"],
        "result_digest": result["result_digest"],
    })
    return result


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
