"""Preregister and run the bounded real WF-03 neighbor-baseline gate.

This harness never accepts an outcome or label path.  It builds return/volatility
features for a deterministic 128-symbol slice of the immutable packed generation,
then exercises deterministic-random and rank-L1 retrieval for all three frozen
development probes in original and shuffled physical order.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from market_analogues.baseline_feature_store import (
    FEATURE_DTYPE,
    baseline_feature_store_contract,
    features_for_packed_records,
    validate_feature_records,
)
from market_analogues.baseline_neighbors import (
    deterministic_random_neighbors,
    recent_return_volatility,
    recent_return_volatility_neighbors,
)
from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.packed_bound_search import _eligible_mask
from market_analogues.packed_bound_store import decode_episode_id, load_packed_generation
from market_analogues.resident_store import resident_file_identity_lease
from market_analogues.types import InstrumentKey, stable_hash

from experiments.m04r import m04r14_t14_10_wf03_feasibility as base


SCHEMA = "m04r14-t14-10-wf03-baseline-poc-preregistration-v1"
OUTPUT_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf03-baseline-poc-v1"
)
PREREGISTRATION_RELATIVE = Path(
    "experiments/m04r/m04r14_t14_10_wf03_baseline_poc_preregistered.json"
)
SYMBOLS = 128
WORKERS = 8
RUNTIME_FILES = (
    "experiments/m04r/m04r14_t14_10_wf03_baseline_poc.py",
    "experiments/m04r/m04r14_t14_10_wf03_feasibility.py",
    "src/market_analogues/baseline_feature_store.py",
    "src/market_analogues/baseline_neighbors.py",
    "src/market_analogues/causal_prefix.py",
    "src/market_analogues/packed_bound_search.py",
    "src/market_analogues/packed_bound_store.py",
    "src/market_analogues/resident_store.py",
)

_SOURCE: Any = None
_PACKED: Any = None
_MAXIMUM: pd.Timestamp | None = None


class BaselinePocError(RuntimeError):
    pass


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repository, text=True, capture_output=True, check=False,
    )
    if result.returncode:
        raise BaselinePocError(f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _slice(records: np.ndarray, symbol_id: int) -> np.ndarray:
    values = records["symbol_id"]
    first = int(np.searchsorted(values, symbol_id, side="left"))
    last = int(np.searchsorted(values, symbol_id, side="right"))
    return records[first:last]


def _row_digest(records: np.ndarray) -> str:
    return stable_hash([{
        "episode_id": decode_episode_id(row["episode_id"]),
        "cutoff_ns": int(row["cutoff_ns"]),
        "quality_tier": int(row["quality_tier"]),
    } for row in records])


def select_symbols(loaded: Any, earliest_query: Any, count: int = SYMBOLS) -> list[dict[str, Any]]:
    if count < 20 or count > len(loaded.symbols):
        raise BaselinePocError("bounded baseline symbol count differs")
    overflow_ids = set(int(value) for value in loaded.overflow["symbol_id"])
    query_symbol_id = loaded.symbols.index(earliest_query.symbol)
    eligible_main = _eligible_mask(loaded.rows, earliest_query, query_symbol_id)
    eligible_overflow = _eligible_mask(
        loaded.overflow, earliest_query, query_symbol_id,
    )
    eligible_ids = np.union1d(
        loaded.rows["symbol_id"][eligible_main],
        loaded.overflow["symbol_id"][eligible_overflow],
    ).astype(np.int64).tolist()
    forced = sorted(
        (value for value in eligible_ids if value in overflow_ids),
        key=lambda value: (
            stable_hash({"lane": "baseline-overflow-v1", "symbol": loaded.symbols[value]}),
            value,
        ),
    )
    remaining = sorted(
        (value for value in eligible_ids if value not in overflow_ids),
        key=lambda value: (
            stable_hash({"lane": "baseline-main-v1", "symbol": loaded.symbols[value]}),
            value,
        ),
    )
    chosen = sorted((forced + remaining)[:count])
    if len(chosen) != count:
        raise BaselinePocError("insufficient earliest-probe baseline symbols")
    output = []
    for symbol_id in chosen:
        main = _slice(loaded.rows, symbol_id)
        overflow = _slice(loaded.overflow, symbol_id)
        output.append({
            "symbol": loaded.symbols[symbol_id],
            "symbol_id": symbol_id,
            "forced_overflow": symbol_id in overflow_ids,
            "rows": len(main),
            "overflow_rows": len(overflow),
            "main_slice_digest": _row_digest(main),
            "overflow_slice_digest": _row_digest(overflow),
            "source_prefix": loaded.manifest["provenance"]["source_prefixes"][
                loaded.symbols[symbol_id]
            ],
        })
    return output


def _paths(root: Path, symbol: str) -> tuple[Path, Path, Path]:
    stem = root / sha256(symbol.encode()).hexdigest()
    return (
        stem.with_suffix(".features.bin"),
        stem.with_suffix(".overflow-features.bin"),
        stem.with_suffix(".json"),
    )


def _write_array(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    values.tofile(temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _build_symbol(task: tuple[dict[str, Any], str]) -> dict[str, Any]:
    specification, root_value = task
    if _SOURCE is None or _PACKED is None or _MAXIMUM is None:
        raise BaselinePocError("baseline worker is not initialized")
    root = Path(root_value)
    symbol = specification["symbol"]
    symbol_id = int(specification["symbol_id"])
    frame = _SOURCE.load(InstrumentKey("nasdaq", symbol))
    prefix = asdict(causal_prefix_digest(frame, _MAXIMUM))
    if prefix != specification["source_prefix"]:
        raise BaselinePocError(f"baseline source prefix changed: {symbol}")
    frame = frame[frame.timestamp <= _MAXIMUM].reset_index(drop=True)
    main = _slice(_PACKED.rows, symbol_id)
    overflow = _slice(_PACKED.overflow, symbol_id)
    if _row_digest(main) != specification["main_slice_digest"] \
            or _row_digest(overflow) != specification["overflow_slice_digest"]:
        raise BaselinePocError("baseline packed slice changed")
    main_features = features_for_packed_records(frame, main)
    overflow_features = features_for_packed_records(frame, overflow)
    rows_path, overflow_path, metadata_path = _paths(root, symbol)
    _write_array(rows_path, main_features)
    _write_array(overflow_path, overflow_features)
    state = {
        "schema_version": "m04r14-wf03-baseline-poc-shard-v1",
        "symbol": symbol,
        "symbol_id": symbol_id,
        "source_prefix": prefix,
        "main_slice_digest": specification["main_slice_digest"],
        "overflow_slice_digest": specification["overflow_slice_digest"],
        "rows": len(main_features),
        "overflow_rows": len(overflow_features),
        "missing_rows": int(np.isnan(main_features["values"]).all(axis=1).sum()),
        "missing_overflow_rows": int(
            np.isnan(overflow_features["values"]).all(axis=1).sum()
        ),
        "rows_sha256": base._sha(rows_path),
        "overflow_sha256": base._sha(overflow_path),
    }
    metadata = base._sealed(state, "shard_digest")
    base._atomic(metadata_path, metadata)
    return metadata


def _load_arrays(
    root: Path, selection: Sequence[Mapping[str, Any]], loaded: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    record_main: list[np.ndarray] = []
    record_overflow: list[np.ndarray] = []
    feature_main: list[np.ndarray] = []
    feature_overflow: list[np.ndarray] = []
    for row in selection:
        symbol_id = int(row["symbol_id"])
        rows_path, overflow_path, metadata_path = _paths(root, str(row["symbol"]))
        metadata = base._read(metadata_path)
        base._validate_seal(metadata, "shard_digest")
        main = np.fromfile(rows_path, dtype=FEATURE_DTYPE)
        overflow = np.fromfile(overflow_path, dtype=FEATURE_DTYPE)
        validate_feature_records(main)
        validate_feature_records(overflow)
        if metadata["rows_sha256"] != base._sha(rows_path) \
                or metadata["overflow_sha256"] != base._sha(overflow_path) \
                or len(main) != row["rows"] or len(overflow) != row["overflow_rows"]:
            raise BaselinePocError("baseline shard differs")
        record_main.append(_slice(loaded.rows, symbol_id))
        record_overflow.append(_slice(loaded.overflow, symbol_id))
        feature_main.append(main)
        feature_overflow.append(overflow)
    concatenate = lambda rows, dtype: (
        np.concatenate(rows) if rows else np.empty(0, dtype=dtype)
    )
    return (
        concatenate(record_main, loaded.rows.dtype),
        concatenate(record_overflow, loaded.overflow.dtype),
        concatenate(feature_main, FEATURE_DTYPE),
        concatenate(feature_overflow, FEATURE_DTYPE),
    )


def _neighbor_state(rows: Sequence[Any]) -> list[dict[str, Any]]:
    return [{
        "episode_id": row.episode_id,
        "symbol": row.symbol,
        "distance_hex": None if row.distance is None else row.distance.hex(),
        "order_key": row.order_key,
    } for row in rows]


def _context(repository: Path) -> tuple[Any, Any, list[tuple[str, Any, Any]]]:
    resident = base._resident()
    loaded = load_packed_generation(
        Path(resident["store_root"]), base.GENERATION_ID,
        expected_provenance_digest=base.PROVENANCE_DIGEST,
        verify_content=False, validate_records=False,
    )
    _registry, by_id = base._registry(repository)
    contexts = []
    source = None
    for label, query_id, _symbol, _cutoff in base.PROBES:
        current_source, episode, _request, packed_query = base._context(
            repository, by_id[query_id],
        )
        if source is None:
            source = current_source
        contexts.append((label, episode, packed_query))
    return loaded, source, contexts


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain"):
        raise BaselinePocError("baseline POC preregistration requires a clean commit")
    loaded, _source, contexts = _context(repository)
    selection = select_symbols(loaded, contexts[0][2])
    state = {
        "schema_version": SCHEMA,
        "status": "frozen_before_bounded_baseline_build",
        "implementation_commit": _git(repository, "rev-parse", "HEAD"),
        "runtime_files": {
            path: base._sha(repository / path) for path in RUNTIME_FILES
        },
        "inputs": {
            "packed_generation_id": loaded.generation_id,
            "packed_manifest_digest": loaded.manifest["manifest_digest"],
            "packed_provenance_digest": loaded.manifest["provenance_digest"],
            "probe_ids": [episode.key.id for _label, episode, _query in contexts],
        },
        "selection": selection,
        "selection_digest": stable_hash(selection),
        "feature_store_contract": baseline_feature_store_contract(),
        "execution": {
            "symbols": SYMBOLS,
            "workers": WORKERS,
            "probes": len(contexts),
            "top_k": 20,
            "minimum_history_gap_bars": base.MINIMUM_HISTORY_GAP,
            "original_and_seeded_shuffled_traversal": True,
            "output_root": str((repository / OUTPUT_RELATIVE).resolve()),
        },
        "claims": {
            "outcomes_or_labels_used": False,
            "historical_walk_forward_query_outcomes_opened": False,
            "final_period_result_opened": False,
            "production_promotion_authorized": False,
        },
    }
    return base._sealed(state, "preregistration_digest")


def validate_preregistration(repository: Path, value: Mapping[str, Any]) -> None:
    base._validate_seal(value, "preregistration_digest")
    loaded, _source, contexts = _context(repository)
    selection = select_symbols(loaded, contexts[0][2])
    if value.get("schema_version") != SCHEMA \
            or value.get("selection") != selection \
            or value.get("selection_digest") != stable_hash(selection) \
            or value.get("feature_store_contract") != baseline_feature_store_contract() \
            or value.get("inputs", {}).get("probe_ids") != [
                episode.key.id for _label, episode, _query in contexts
            ]:
        raise BaselinePocError("baseline POC preregistration differs")
    head = value.get("implementation_commit")
    if type(head) is not str:
        raise BaselinePocError("baseline implementation commit differs")
    _git(repository, "merge-base", "--is-ancestor", head, "HEAD")
    for path, digest in value["runtime_files"].items():
        blob = subprocess.run(
            ["git", "show", f"{head}:{path}"], cwd=repository,
            capture_output=True, check=False,
        )
        if blob.returncode or sha256(blob.stdout).hexdigest() != digest \
                or base._sha(repository / path) != digest:
            raise BaselinePocError(f"baseline frozen runtime differs: {path}")


def execute(repository: Path, preregistration: Mapping[str, Any]) -> dict[str, Any]:
    started = perf_counter()
    validate_preregistration(repository, preregistration)
    loaded, source, contexts = _context(repository)
    root = repository / OUTPUT_RELATIVE
    if root.exists() or root.is_symlink():
        raise BaselinePocError("baseline POC output must be absent")
    root.mkdir(parents=True)
    base._atomic(root / "CONTRACT.json", preregistration)
    work = root / "shards"
    work.mkdir()
    maximum = pd.Timestamp(loaded.manifest["provenance"]["benchmark_prefix"][
        "requested_cutoff"
    ])
    lease_before = resident_file_identity_lease(base.RESIDENT_ROOT / "READY.json")
    global _SOURCE, _PACKED, _MAXIMUM
    _SOURCE, _PACKED, _MAXIMUM = source, loaded, maximum
    tasks = [(dict(row), str(work)) for row in preregistration["selection"]]
    with ProcessPoolExecutor(
        max_workers=WORKERS, mp_context=multiprocessing.get_context("fork"),
    ) as executor:
        metadata = list(executor.map(_build_symbol, tasks))
    metadata.sort(key=lambda row: row["symbol_id"])
    main, overflow, main_features, overflow_features = _load_arrays(
        work, preregistration["selection"], loaded,
    )
    records = np.concatenate((main, overflow))
    features = np.concatenate((main_features, overflow_features))["values"]
    if len(records) != len(features) or not len(records):
        raise BaselinePocError("bounded baseline alignment differs")
    probe_results = []
    for ordinal, (label, episode, packed_query) in enumerate(contexts):
        query_symbol_id = loaded.symbols.index(episode.key.instrument.source_symbol)
        eligible = _eligible_mask(records, packed_query, query_symbol_id)
        query_features = recent_return_volatility(
            episode.bars["close"].to_numpy(dtype=np.float64)
        )
        random_rows = deterministic_random_neighbors(
            records["episode_id"], records["symbol_id"], eligible,
            loaded.symbols, episode.key.id, top_k=20,
        )
        rank_rows = recent_return_volatility_neighbors(
            features, records["episode_id"], records["symbol_id"], eligible,
            loaded.symbols, query_features, episode.key.id, top_k=20,
        )
        rng = np.random.default_rng(810_000 + ordinal)
        order = rng.permutation(len(records))
        shuffled_random = deterministic_random_neighbors(
            records["episode_id"][order], records["symbol_id"][order],
            eligible[order], loaded.symbols, episode.key.id, top_k=20,
        )
        shuffled_rank = recent_return_volatility_neighbors(
            features[order], records["episode_id"][order],
            records["symbol_id"][order], eligible[order], loaded.symbols,
            query_features, episode.key.id, top_k=20,
        )
        if random_rows != shuffled_random or rank_rows != shuffled_rank \
                or len(random_rows) != 20 or len(rank_rows) != 20 \
                or len({row.symbol for row in random_rows}) != 20 \
                or len({row.symbol for row in rank_rows}) != 20:
            raise BaselinePocError("bounded baseline neighbor gate differs")
        probe_results.append({
            "ordinal": ordinal,
            "label": label,
            "query_id": episode.key.id,
            "query_features_hex": [value.hex() for value in query_features],
            "eligible_rows": int(eligible.sum()),
            "eligible_symbols": int(len(set(
                int(value) for value in records["symbol_id"][eligible]
            ))),
            "random_neighbors": _neighbor_state(random_rows),
            "rank_neighbors": _neighbor_state(rank_rows),
            "shuffled_traversal_equal": True,
        })
    lease_after = resident_file_identity_lease(base.RESIDENT_ROOT / "READY.json")
    gates = {
        "physical_alignment": len(records) == sum(
            int(row["rows"]) + int(row["overflow_rows"]) for row in metadata
        ),
        "all_feature_rows_finite": bool(np.isfinite(features).all()),
        "twenty_distinct_neighbors_per_method": True,
        "shuffled_traversal_equal": True,
        "resident_lease_unchanged": (
            lease_before["lease_digest"] == lease_after["lease_digest"]
        ),
    }
    if not all(gates.values()):
        raise BaselinePocError("bounded baseline terminal gate differs")
    state = {
        "schema_version": "m04r14-t14-10-wf03-baseline-poc-result-v1",
        "status": "complete",
        "passed": True,
        "gates": gates,
        "selection_digest": preregistration["selection_digest"],
        "symbols": len(metadata),
        "rows": len(main),
        "overflow_rows": len(overflow),
        "feature_content_digest": stable_hash([{
            "symbol": row["symbol"],
            "rows_sha256": row["rows_sha256"],
            "overflow_sha256": row["overflow_sha256"],
        } for row in metadata]),
        "missing_rows": sum(int(row["missing_rows"]) for row in metadata),
        "missing_overflow_rows": sum(
            int(row["missing_overflow_rows"]) for row in metadata
        ),
        "probe_results": probe_results,
        "probe_digest": stable_hash(probe_results),
        "elapsed_seconds": perf_counter() - started,
        "resident_lease_digest": lease_after["lease_digest"],
        "outcomes_or_labels_used": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
        "independent_verification_authorized": True,
    }
    result = base._sealed(state)
    base._atomic(root / "RESULT.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preregister", "run"))
    parser.add_argument("--repository", required=True, type=Path)
    args = parser.parse_args(argv)
    repository = args.repository.resolve(strict=True)
    preregistration_path = repository / PREREGISTRATION_RELATIVE
    if args.mode == "preregister":
        base._atomic(preregistration_path, build_preregistration(repository))
        return 0
    result = execute(repository, base._read(preregistration_path))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
