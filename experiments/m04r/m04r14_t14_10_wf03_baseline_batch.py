"""Preregister and run outcome-blind baselines for all 3,936 WF-03 queries."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from market_analogues.adapters import source_from_spec
from market_analogues.baseline_feature_store import load_feature_generation
from market_analogues.baseline_neighbors import (
    BaselineRankIndex,
    build_baseline_rank_index,
    deterministic_random_neighbors,
    indexed_recent_return_volatility_neighbors,
    recent_return_volatility,
)
from market_analogues.config import load_config
from market_analogues.episodes import build_episode
from market_analogues.packed_bound_search import _eligible_mask
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import InstrumentKey, stable_hash

from experiments.m04r import m04r14_t14_10_wf03_baseline_poc as bounded
from experiments.m04r import m04r14_t14_10_wf03_baseline_store_full as feature_store
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base


SCHEMA = "m04r14-t14-10-wf03-baseline-batch-preregistration-v1"
OUTPUT_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf03-baseline-batch-v1"
)
PREREGISTRATION_RELATIVE = Path(
    "experiments/m04r/m04r14_t14_10_wf03_baseline_batch_preregistered.json"
)
FEATURE_RESULT_RELATIVE = feature_store.OUTPUT_RELATIVE / "RESULT.json"
FEATURE_VERIFICATION_RELATIVE = Path(
    "config/data/analogues/m04r14/"
    "t14-10-wf03-baseline-store-full-v1-verification/VERIFIED.json"
)
WORKERS = 8
TOP_K = 20
RUNTIME_FILES = (
    "experiments/m04r/m04r14_t14_10_wf03_baseline_batch.py",
    "experiments/m04r/m04r14_t14_10_wf03_feasibility.py",
    "src/market_analogues/baseline_neighbors.py",
    "src/market_analogues/baseline_feature_store.py",
    "src/market_analogues/episodes.py",
    "src/market_analogues/packed_bound_search.py",
    "src/market_analogues/packed_bound_store.py",
    "src/market_analogues/search.py",
)

_SOURCE: Any = None
_RECORDS: np.ndarray | None = None
_FEATURES: np.ndarray | None = None
_RANK_INDEX: BaselineRankIndex | None = None
_SYMBOLS: tuple[str, ...] = ()
_CASES_ROOT: Path | None = None


class BaselineBatchError(RuntimeError):
    pass


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repository, text=True, capture_output=True, check=False,
    )
    if result.returncode:
        raise BaselineBatchError(f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _neighbor_state(rows: Sequence[Any]) -> list[dict[str, Any]]:
    return [{
        "episode_id": row.episode_id,
        "symbol": row.symbol,
        "distance_hex": None if row.distance is None else row.distance.hex(),
        "order_key": row.order_key,
    } for row in rows]


def _case_path(root: Path, query_id: str) -> Path:
    if len(query_id) != 24 or any(value not in "0123456789abcdef" for value in query_id):
        raise BaselineBatchError("baseline batch query ID differs")
    return root / f"{query_id}.json"


def _replace_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def _upstream(repository: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    result = base._read(repository / FEATURE_RESULT_RELATIVE)
    base._validate_seal(result)
    verification = base._read(repository / FEATURE_VERIFICATION_RELATIVE)
    base._validate_seal(verification, "verification_digest")
    if result.get("passed") is not True \
            or verification.get("passed") is not True \
            or verification.get("producer_result_digest") != result["result_digest"] \
            or verification.get("monthly_committee_batch_retrieval_authorized") is not True \
            or result.get("outcomes_or_labels_used") is not False \
            or verification.get("outcomes_or_labels_used") is not False:
        raise BaselineBatchError("full baseline store does not authorize batch")
    return result, verification


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain"):
        raise BaselineBatchError("baseline batch preregistration requires clean commit")
    registry, _by_id = base._registry(repository)
    result, verification = _upstream(repository)
    queries = registry["queries_data"]
    output = repository / OUTPUT_RELATIVE
    if output.exists() or output.is_symlink():
        raise BaselineBatchError("baseline batch output must be absent before freeze")
    query_ids = [row["episode_id"] for row in queries]
    cutoffs = [row["cutoff"] for row in queries]
    state = {
        "schema_version": SCHEMA,
        "status": "frozen_before_all_query_baseline_retrieval",
        "implementation_commit": _git(repository, "rev-parse", "HEAD"),
        "runtime_files": {
            path: base._sha(repository / path) for path in RUNTIME_FILES
        },
        "inputs": {
            "registry_digest": registry["registry_digest"],
            "registry_sha256": base._sha(repository / base.REGISTRY_FILE),
            "query_ids_digest": stable_hash(query_ids),
            "feature_result_digest": result["result_digest"],
            "feature_result_sha256": base._sha(repository / FEATURE_RESULT_RELATIVE),
            "feature_verification_digest": verification["verification_digest"],
            "feature_verification_sha256": base._sha(
                repository / FEATURE_VERIFICATION_RELATIVE
            ),
            "feature_generation_id": result["generation_id"],
            "packed_generation_id": base.GENERATION_ID,
            "packed_provenance_digest": base.PROVENANCE_DIGEST,
        },
        "inventory": {
            "queries": len(queries),
            "scored_queries": sum(bool(row["scored"]) for row in queries),
            "unscored_warmup_queries": sum(not bool(row["scored"]) for row in queries),
            "months": len(set(cutoffs)),
            "queries_per_month": sorted(set(
                cutoffs.count(value) for value in set(cutoffs)
            )),
        },
        "execution": {
            "workers": WORKERS,
            "top_k": TOP_K,
            "rank_index": "three global (feature value, episode ID) orders shared by fork",
            "random": "exact symbol-first equivalent implementation",
            "case_publication": "create-only sealed query JSON",
            "progress": "atomic after every 24 completed queries",
            "resume": "reuse only sealed cases bound to query and upstream generation",
            "output_root": str(output.resolve()),
        },
        "claims": {
            "outcomes_or_labels_used": False,
            "historical_walk_forward_query_outcomes_opened": False,
            "final_period_result_opened": False,
            "production_promotion_authorized": False,
        },
    }
    return base._sealed(state, "preregistration_digest")


def validate_preregistration(
    repository: Path, value: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    base._validate_seal(value, "preregistration_digest")
    registry, by_id = base._registry(repository)
    result, verification = _upstream(repository)
    query_ids = [row["episode_id"] for row in registry["queries_data"]]
    if value.get("schema_version") != SCHEMA \
            or value.get("inputs", {}).get("registry_digest") != registry["registry_digest"] \
            or value.get("inputs", {}).get("query_ids_digest") != stable_hash(query_ids) \
            or value.get("inputs", {}).get("feature_result_digest") != result["result_digest"] \
            or value.get("inputs", {}).get("feature_verification_digest") \
            != verification["verification_digest"]:
        raise BaselineBatchError("baseline batch preregistration differs")
    head = value.get("implementation_commit")
    if type(head) is not str:
        raise BaselineBatchError("baseline batch implementation differs")
    _git(repository, "merge-base", "--is-ancestor", head, "HEAD")
    for path, digest in value["runtime_files"].items():
        blob = subprocess.run(
            ["git", "show", f"{head}:{path}"], cwd=repository,
            capture_output=True, check=False,
        )
        if blob.returncode or sha256(blob.stdout).hexdigest() != digest \
                or base._sha(repository / path) != digest:
            raise BaselineBatchError(f"baseline batch runtime differs: {path}")
    return registry, by_id, result


def _valid_case(path: Path, row: Mapping[str, Any], generation_id: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = base._read(path)
        base._validate_seal(value, "case_digest")
        valid = all((
            value["schema_version"] == "m04r14-wf03-baseline-batch-case-v1",
            value["query_id"] == row["episode_id"],
            value["case_id"] == row["case_id"],
            value["feature_generation_id"] == generation_id,
            value["outcomes_or_labels_used"] is False,
            len(value["random_neighbors"]) == TOP_K,
            len(value["rank_neighbors"]) == TOP_K,
        ))
    except Exception:
        return None
    return value if valid else None


def _run_query(task: tuple[dict[str, Any], str]) -> dict[str, Any]:
    row, generation_id = task
    if _SOURCE is None or _RECORDS is None or _FEATURES is None \
            or _RANK_INDEX is None or not _SYMBOLS or _CASES_ROOT is None:
        raise BaselineBatchError("baseline batch worker is uninitialized")
    path = _case_path(_CASES_ROOT, row["episode_id"])
    existing = _valid_case(path, row, generation_id)
    if existing is not None:
        return existing
    episode = build_episode(
        _SOURCE, InstrumentKey("nasdaq", row["symbol"]), row["cutoff"],
        int(row["lookback"]), str(row["representation_version"]),
    )
    if episode.key.id != row["episode_id"]:
        raise BaselineBatchError("baseline batch query reconstruction differs")
    latest = latest_eligible_cutoff(episode, base.MINIMUM_HISTORY_GAP)
    query = SimpleNamespace(
        episode_id=episode.key.id,
        symbol=episode.key.instrument.source_symbol,
        query_start_ns=int(episode.bars.timestamp.iloc[0].value),
        latest_eligible_ns=int(latest.value),
        quality_tiers=("A", "B"),
    )
    try:
        symbol_id = _SYMBOLS.index(query.symbol)
    except ValueError as exc:
        raise BaselineBatchError("baseline query symbol is absent from packed store") from exc
    eligible = _eligible_mask(_RECORDS, query, symbol_id)
    query_features = recent_return_volatility(
        episode.bars["close"].to_numpy(dtype=np.float64)
    )
    random_rows = deterministic_random_neighbors(
        _RECORDS["episode_id"], _RECORDS["symbol_id"], eligible,
        _SYMBOLS, episode.key.id, top_k=TOP_K,
    )
    rank_rows = indexed_recent_return_volatility_neighbors(
        _RANK_INDEX, _FEATURES, _RECORDS["episode_id"],
        _RECORDS["symbol_id"], eligible, _SYMBOLS, query_features,
        episode.key.id, top_k=TOP_K,
    )
    if len(random_rows) != TOP_K or len(rank_rows) != TOP_K \
            or len({value.symbol for value in random_rows}) != TOP_K \
            or len({value.symbol for value in rank_rows}) != TOP_K:
        raise BaselineBatchError("baseline batch distinct-neighbor gate differs")
    state = {
        "schema_version": "m04r14-wf03-baseline-batch-case-v1",
        "status": "complete",
        "case_id": row["case_id"],
        "query_id": episode.key.id,
        "symbol": row["symbol"],
        "cutoff": row["cutoff"],
        "fold_id": row["fold_id"],
        "fold_role": row["fold_role"],
        "scored": bool(row["scored"]),
        "feature_generation_id": generation_id,
        "query_features_hex": [value.hex() for value in query_features],
        "latest_eligible_ns": int(latest.value),
        "eligible_rows": int(eligible.sum()),
        "eligible_symbols": int(len(np.unique(_RECORDS["symbol_id"][eligible]))),
        "random_neighbors": _neighbor_state(random_rows),
        "rank_neighbors": _neighbor_state(rank_rows),
        "outcomes_or_labels_used": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
    }
    result = base._sealed(state, "case_digest")
    base._atomic(path, result)
    return result


def execute(repository: Path, preregistration: Mapping[str, Any]) -> dict[str, Any]:
    started = perf_counter()
    registry, _by_id, feature_result = validate_preregistration(
        repository, preregistration,
    )
    root = repository / OUTPUT_RELATIVE
    if root.is_symlink() or root.exists() and not root.is_dir():
        raise BaselineBatchError("baseline batch output path differs")
    if root.exists():
        if base._read(root / "CONTRACT.json") != preregistration:
            raise BaselineBatchError("baseline batch resume contract differs")
        result_path = root / "RESULT.json"
        if result_path.exists():
            result = base._read(result_path); base._validate_seal(result)
            return result
    else:
        root.mkdir(parents=True)
        base._atomic(root / "CONTRACT.json", preregistration)
    cases_root = root / "cases"; cases_root.mkdir(exist_ok=True)
    resident = base._resident()
    packed = load_packed_generation(
        Path(resident["store_root"]), base.GENERATION_ID,
        expected_provenance_digest=base.PROVENANCE_DIGEST,
        verify_content=False, validate_records=False,
    )
    feature_generation = load_feature_generation(
        repository / feature_store.OUTPUT_RELATIVE / "store",
        feature_result["generation_id"], packed_manifest=packed.manifest,
        verify_content=False,
    )
    records = np.concatenate((
        bounded._neighbor_records(packed.rows),
        bounded._neighbor_records(packed.overflow),
    ))
    features = np.concatenate((
        feature_generation.rows, feature_generation.overflow,
    ))["values"]
    rank_started = perf_counter()
    rank_index = build_baseline_rank_index(features, records["episode_id"])
    rank_index_seconds = perf_counter() - rank_started
    source = source_from_spec(
        load_config(repository / base.CONFIG_RELATIVE).datasets["nasdaq"]
    )
    global _SOURCE, _RECORDS, _FEATURES, _RANK_INDEX, _SYMBOLS, _CASES_ROOT
    _SOURCE, _RECORDS, _FEATURES, _RANK_INDEX = source, records, features, rank_index
    _SYMBOLS, _CASES_ROOT = packed.symbols, cases_root
    tasks = [(dict(row), feature_generation.generation_id)
             for row in registry["queries_data"]]
    results = []
    with ProcessPoolExecutor(
        max_workers=WORKERS, mp_context=multiprocessing.get_context("fork"),
    ) as executor:
        for completed, result in enumerate(
            executor.map(_run_query, tasks, chunksize=1), start=1,
        ):
            results.append(result)
            if completed % 24 == 0 or completed == len(tasks):
                _replace_json(root / "PROGRESS.json", {
                    "schema_version": "m04r14-wf03-baseline-batch-progress-v1",
                    "status": "running" if completed < len(tasks) else "publishing",
                    "completed_queries": completed,
                    "total_queries": len(tasks),
                    "completed_month_equivalents": completed // 24,
                })
    if [row["query_id"] for row in results] != [row[0]["episode_id"] for row in tasks]:
        raise BaselineBatchError("baseline batch result order differs")
    case_manifest = [{
        "query_id": row["query_id"],
        "case_digest": row["case_digest"],
        "sha256": base._sha(_case_path(cases_root, row["query_id"])),
    } for row in results]
    state = {
        "schema_version": "m04r14-t14-10-wf03-baseline-batch-result-v1",
        "status": "complete",
        "passed": True,
        "queries": len(results),
        "scored_queries": sum(bool(row["scored"]) for row in results),
        "warmup_queries": sum(not bool(row["scored"]) for row in results),
        "months": preregistration["inventory"]["months"],
        "feature_generation_id": feature_generation.generation_id,
        "rank_index_seconds": rank_index_seconds,
        "case_manifest_digest": stable_hash(case_manifest),
        "case_receipt_digest": stable_hash([
            row["case_digest"] for row in results
        ]),
        "minimum_eligible_rows": min(row["eligible_rows"] for row in results),
        "maximum_eligible_rows": max(row["eligible_rows"] for row in results),
        "elapsed_seconds": perf_counter() - started,
        "outcomes_or_labels_used": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
        "independent_verification_authorized": True,
    }
    result = base._sealed(state)
    base._atomic(root / "RESULT.json", result)
    _replace_json(root / "PROGRESS.json", {
        "schema_version": "m04r14-wf03-baseline-batch-progress-v1",
        "status": "complete",
        "completed_queries": len(results),
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
        base._atomic(path, build_preregistration(repository)); return 0
    result = execute(repository, base._read(path))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
