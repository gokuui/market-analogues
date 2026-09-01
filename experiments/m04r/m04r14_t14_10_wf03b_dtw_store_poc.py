"""Build and verify two clean bounded real DTW-sample shard sets."""
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
from numba import set_num_threads

from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.dtw_interval_bound import (
    DtwIntervalBoundError,
    quantize_dtw_samples,
    quantized_dtw_lower_bound,
)
from market_analogues.dtw_sample_store import (
    DTW_SAMPLE_DTYPE,
    DTW_SAMPLE_ROW_BYTES,
    dtw_sample_lower_bounds,
    dtw_sample_store_contract,
    make_dtw_sample_record_from_quantized,
    make_zero_dtw_sample_record,
    validate_dtw_sample_records,
)
from market_analogues.exact_batch import sliding_exact_representations
from market_analogues.packed_bound_store import decode_episode_id, load_packed_generation
from market_analogues.representation import representation_input_digest
from market_analogues.resident_store import resident_file_identity_lease
from market_analogues.types import EpisodeKey, InstrumentKey, stable_hash

from experiments.m04r import m04r14_t14_10_wf03_feasibility as base


SCHEMA = "m04r14-t14-10-wf03b-dtw-store-poc-preregistration-v1"
OUTPUT_RELATIVE = Path("config/data/analogues/m04r14/t14-10-wf03b-dtw-store-poc-v1")
PREREGISTRATION_RELATIVE = Path(
    "experiments/m04r/m04r14_t14_10_wf03b_dtw_store_poc_preregistered.json"
)
VERIFICATION_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf03b-dtw-bound-poc-v1-verification/VERIFIED.json"
)
SYMBOLS = 64
WORKERS = 8
MAX_PROJECTED_SCAN_SECONDS = 90.0
MAX_PROJECTED_STORE_GIB = 3.1
TOLERANCE = 1e-12
RUNTIME_FILES = (
    "experiments/m04r/m04r14_t14_10_wf03b_dtw_store_poc.py",
    "experiments/m04r/m04r14_t14_10_wf03_feasibility.py",
    "src/market_analogues/adapters.py", "src/market_analogues/causal_prefix.py",
    "src/market_analogues/config.py", "src/market_analogues/context.py",
    "src/market_analogues/dtw_interval_bound.py",
    "src/market_analogues/dtw_sample_store.py",
    "src/market_analogues/exact_batch.py", "src/market_analogues/representation.py",
    "src/market_analogues/packed_bound_store.py",
    "src/market_analogues/quantized_bound.py",
    "src/market_analogues/resident_store.py",
    "src/market_analogues/episodes.py", "src/market_analogues/search.py",
    "src/market_analogues/structural.py", "src/market_analogues/types.py",
)

_SOURCE: Any = None
_BENCHMARK: pd.DataFrame | None = None
_QUERY: Any = None
_QUERY_ID = ""
_MAXIMUM_CUTOFF: pd.Timestamp | None = None
_PACKED: Any = None


class StorePocError(RuntimeError):
    pass


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repository, text=True, capture_output=True, check=False,
    )
    if result.returncode:
        raise StorePocError(f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _stem(symbol: str) -> str:
    return sha256(symbol.encode()).hexdigest()


def _paths(root: Path, symbol: str) -> tuple[Path, Path, Path]:
    stem = root / _stem(symbol)
    return (stem.with_suffix(".rows.bin"), stem.with_suffix(".overflow.bin"),
            stem.with_suffix(".json"))


def _slice(records: np.ndarray, symbol_id: int) -> np.ndarray:
    values = records["symbol_id"]
    first = int(np.searchsorted(values, symbol_id, side="left"))
    last = int(np.searchsorted(values, symbol_id, side="right"))
    return records[first:last]


def _packed_rows_state(records: np.ndarray) -> list[dict[str, Any]]:
    return [{
        "episode_id": decode_episode_id(row["episode_id"]),
        "cutoff_ns": int(row["cutoff_ns"]),
        "quality_tier": int(row["quality_tier"]),
    } for row in records]


def select_symbols(loaded: Any, count: int = SYMBOLS) -> list[dict[str, Any]]:
    if count < 1 or count > len(loaded.symbols):
        raise StorePocError("bounded symbol count differs")
    overflow_ids = set(int(value) for value in loaded.overflow["symbol_id"])
    forced = sorted(overflow_ids, key=lambda index: (
        stable_hash({"lane": "overflow", "symbol": loaded.symbols[index]}), index,
    ))
    remaining = sorted(
        (index for index in range(len(loaded.symbols)) if index not in overflow_ids),
        key=lambda index: (
            stable_hash({"lane": "bounded", "symbol": loaded.symbols[index]}), index,
        ),
    )
    selected = sorted((forced + remaining)[:count])
    output = []
    for symbol_id in selected:
        main = _slice(loaded.rows, symbol_id)
        overflow = _slice(loaded.overflow, symbol_id)
        output.append({
            "symbol": loaded.symbols[symbol_id], "symbol_id": symbol_id,
            "forced_overflow": symbol_id in overflow_ids,
            "rows": len(main), "overflow_rows": len(overflow),
            "main_slice_digest": stable_hash(_packed_rows_state(main)),
            "overflow_slice_digest": stable_hash(_packed_rows_state(overflow)),
            "source_prefix": loaded.manifest["provenance"]["source_prefixes"][
                loaded.symbols[symbol_id]
            ],
        })
    if not any(row["forced_overflow"] for row in output):
        raise StorePocError("bounded selection does not exercise overflow")
    return output


def _prerequisites(repository: Path) -> tuple[Any, dict[str, Any], Any, Any]:
    verified = base._read(repository / VERIFICATION_RELATIVE)
    base._validate_seal(verified, "verification_digest")
    if verified.get("passed") is not True \
            or verified.get("auxiliary_store_authorized") is not True:
        raise StorePocError("real DTW-bound verification prerequisite differs")
    resident = base._resident()
    loaded = load_packed_generation(
        Path(resident["store_root"]), base.GENERATION_ID,
        expected_provenance_digest=base.PROVENANCE_DIGEST,
        verify_content=False, validate_records=False,
    )
    _registry, by_id = base._registry(repository)
    query_id = base.PROBES[0][1]
    source, episode, _request, packed = base._context(repository, by_id[query_id])
    return loaded, verified, source, packed


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain"):
        raise StorePocError("store POC preregistration requires a clean commit")
    loaded, verified, _source, packed = _prerequisites(repository)
    cases = select_symbols(loaded)
    state = {
        "schema_version": SCHEMA, "status": "frozen_before_real_store_builds",
        "implementation_commit": _git(repository, "rev-parse", "HEAD"),
        "runtime_files": {path: base._sha(repository / path) for path in RUNTIME_FILES},
        "inputs": {
            "packed_generation_id": loaded.generation_id,
            "packed_manifest_digest": loaded.manifest["manifest_digest"],
            "packed_rows_sha256": loaded.manifest["rows_sha256"],
            "packed_overflow_sha256": loaded.manifest["overflow_sha256"],
            "packed_provenance_digest": loaded.manifest["provenance_digest"],
            "bound_verification_digest": verified["verification_digest"],
            "bound_verification_sha256": base._sha(repository / VERIFICATION_RELATIVE),
            "query_id": packed.episode_id,
            "query_representation_digest": representation_input_digest(
                packed.representation
            ),
        },
        "selection": cases, "selection_digest": stable_hash(cases),
        "store_contract": dtw_sample_store_contract(),
        "execution": {
            "symbols": SYMBOLS, "workers": WORKERS, "clean_builds": 2,
            "lookback": 252, "stride": 5, "batch_size": 512,
            "scalar_probes_per_symbol": 2,
            "scan_repeats": 3, "max_projected_scan_seconds": MAX_PROJECTED_SCAN_SECONDS,
            "max_projected_store_gib": MAX_PROJECTED_STORE_GIB,
            "tolerance": TOLERANCE,
            "output_root": str((repository / OUTPUT_RELATIVE).resolve()),
        },
        "claims": {
            "historical_walk_forward_query_outcomes_opened": False,
            "final_period_result_opened": False,
            "production_promotion_authorized": False,
        },
    }
    return base._sealed(state, "preregistration_digest")


def validate_preregistration(repository: Path, value: Mapping[str, Any]) -> None:
    base._validate_seal(value, "preregistration_digest")
    loaded, verified, _source, packed = _prerequisites(repository)
    if value.get("schema_version") != SCHEMA \
            or value.get("selection") != select_symbols(loaded) \
            or value.get("selection_digest") != stable_hash(value["selection"]) \
            or value.get("store_contract") != dtw_sample_store_contract() \
            or value.get("inputs", {}).get("bound_verification_digest") \
            != verified["verification_digest"] \
            or value.get("inputs", {}).get("query_representation_digest") \
            != representation_input_digest(packed.representation):
        raise StorePocError("store POC preregistration differs")
    head = value.get("implementation_commit")
    if type(head) is not str:
        raise StorePocError("store POC implementation commit differs")
    _git(repository, "merge-base", "--is-ancestor", head, "HEAD")
    for path, digest in value["runtime_files"].items():
        if base._sha(repository / path) != digest:
            raise StorePocError(f"store POC runtime differs: {path}")
        blob = subprocess.run(
            ["git", "show", f"{head}:{path}"], cwd=repository,
            capture_output=True, check=False,
        )
        if blob.returncode or sha256(blob.stdout).hexdigest() != digest:
            raise StorePocError(f"store POC implementation binding differs: {path}")


def _write_array(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    values.tofile(temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validated_shard(
    root: Path, specification: Mapping[str, Any],
) -> dict[str, Any] | None:
    rows_path, overflow_path, metadata_path = _paths(root, specification["symbol"])
    if not all(path.is_file() and not path.is_symlink()
               for path in (rows_path, overflow_path, metadata_path)):
        return None
    try:
        metadata = base._read(metadata_path)
        base._validate_seal(metadata, "shard_digest")
        main = np.fromfile(rows_path, dtype=DTW_SAMPLE_DTYPE)
        overflow = np.fromfile(overflow_path, dtype=DTW_SAMPLE_DTYPE)
        validate_dtw_sample_records(main); validate_dtw_sample_records(overflow)
        valid = all((
            metadata["symbol"] == specification["symbol"],
            metadata["symbol_id"] == specification["symbol_id"],
            metadata["source_prefix"] == specification["source_prefix"],
            metadata["main_slice_digest"] == specification["main_slice_digest"],
            metadata["overflow_slice_digest"] == specification["overflow_slice_digest"],
            metadata["rows"] == specification["rows"] == len(main),
            metadata["overflow_rows"] == specification["overflow_rows"] == len(overflow),
            metadata["rows_sha256"] == base._sha(rows_path),
            metadata["overflow_sha256"] == base._sha(overflow_path),
        ))
    except Exception:
        return None
    return metadata if valid else None


def _build_symbol(task: tuple[dict[str, Any], str]) -> dict[str, Any]:
    specification, root_value = task
    set_num_threads(1)
    if _SOURCE is None or _BENCHMARK is None or _QUERY is None \
            or _MAXIMUM_CUTOFF is None or _PACKED is None:
        raise StorePocError("store POC worker is not initialized")
    root = Path(root_value)
    symbol = specification["symbol"]
    symbol_id = int(specification["symbol_id"])
    rows_path, overflow_path, metadata_path = _paths(root, symbol)
    existing = _validated_shard(root, specification)
    if existing is not None:
        return existing
    key = InstrumentKey("nasdaq", symbol)
    full = _SOURCE.load(key)
    prefix = asdict(causal_prefix_digest(full, _MAXIMUM_CUTOFF))
    if prefix != specification["source_prefix"]:
        raise StorePocError(f"source prefix changed: {symbol}")
    main = _slice(_PACKED.rows, symbol_id)
    overflow = _slice(_PACKED.overflow, symbol_id)
    if stable_hash(_packed_rows_state(main)) != specification["main_slice_digest"] \
            or stable_hash(_packed_rows_state(overflow)) \
            != specification["overflow_slice_digest"]:
        raise StorePocError("packed symbol slice changed")
    main_ids = {decode_episode_id(row["episode_id"]): index
                for index, row in enumerate(main)}
    overflow_ids = {decode_episode_id(row["episode_id"]): index
                    for index, row in enumerate(overflow)}
    frame = full[full.timestamp <= _MAXIMUM_CUTOFF].reset_index(drop=True)
    batch = sliding_exact_representations(
        frame, _BENCHMARK, lookback=252, stride=5, batch_size=512,
    )
    main_rows = np.zeros(len(main), dtype=DTW_SAMPLE_DTYPE)
    overflow_rows = np.zeros(len(overflow), dtype=DTW_SAMPLE_DTYPE)
    seen_main: set[str] = set(); seen_overflow: set[str] = set()
    zero_rows = 0
    probes: list[tuple[str, dict[str, Any]]] = []
    for position, representation in zip(batch.positions, batch.representations, strict=True):
        cutoff = pd.Timestamp(frame.timestamp.iloc[int(position)])
        episode_id = EpisodeKey(key, cutoff, 252, "dense-v1").id
        if episode_id in main_ids:
            lane = "main"; lane_index = main_ids[episode_id]; seen_main.add(episode_id)
        elif episode_id in overflow_ids:
            lane = "overflow"; lane_index = overflow_ids[episode_id]
            seen_overflow.add(episode_id)
        else:
            raise StorePocError("materialized episode is absent from packed symbol slice")
        try:
            quantized = quantize_dtw_samples(representation)
            record = make_dtw_sample_record_from_quantized(quantized)
        except DtwIntervalBoundError:
            quantized = None
            record = make_zero_dtw_sample_record(); zero_rows += 1
        if lane == "main":
            main_rows[lane_index] = record[0]
        else:
            overflow_rows[lane_index] = record[0]
        probe_key = stable_hash({
            "query_id": _QUERY_ID, "episode_id": episode_id,
            "purpose": "store-scalar-probe-v1",
        })
        if len(probes) < 2 or probe_key < probes[-1][0]:
            scalar = (quantized_dtw_lower_bound(_QUERY, quantized)
                      if quantized is not None else 0.0)
            probes.append((probe_key, {
                "episode_id": episode_id, "lane": lane, "lane_index": lane_index,
                "scalar_bound_hex": float(scalar).hex(),
            }))
            probes.sort(key=lambda item: item[0]); probes = probes[:2]
    if seen_main != set(main_ids) or seen_overflow != set(overflow_ids) \
            or len(batch.positions) != len(main) + len(overflow):
        raise StorePocError("materialized/packed episode inventory differs")
    validate_dtw_sample_records(main_rows)
    validate_dtw_sample_records(overflow_rows)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    _write_array(rows_path, main_rows); _write_array(overflow_path, overflow_rows)
    selected_probes = [row for _key, row in probes]
    state = {
        "schema_version": "m04r14-wf03b-dtw-store-poc-shard-v1",
        "symbol": symbol, "symbol_id": symbol_id,
        "source_prefix": prefix,
        "main_slice_digest": specification["main_slice_digest"],
        "overflow_slice_digest": specification["overflow_slice_digest"],
        "rows": len(main_rows), "overflow_rows": len(overflow_rows),
        "rows_sha256": base._sha(rows_path),
        "overflow_sha256": base._sha(overflow_path),
        "zero_bound_rows": zero_rows, "scalar_probes": selected_probes,
    }
    metadata = base._sealed(state, "shard_digest")
    base._atomic(metadata_path, metadata)
    return metadata


def _build(
    root: Path, selection: Sequence[dict[str, Any]], *, source: Any,
    benchmark: pd.DataFrame, query: Any, query_id: str,
    maximum_cutoff: pd.Timestamp, packed: Any,
) -> list[dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    global _SOURCE, _BENCHMARK, _QUERY, _QUERY_ID, _MAXIMUM_CUTOFF, _PACKED
    _SOURCE, _BENCHMARK, _QUERY, _QUERY_ID = source, benchmark, query, query_id
    _MAXIMUM_CUTOFF, _PACKED = maximum_cutoff, packed
    tasks = [(dict(row), str(root)) for row in selection]
    with ProcessPoolExecutor(
        max_workers=WORKERS, mp_context=multiprocessing.get_context("fork"),
    ) as executor:
        results = list(executor.map(_build_symbol, tasks))
    results.sort(key=lambda row: row["symbol_id"])
    return results


def _load_build(root: Path, selection: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    main = []; overflow = []
    for specification in selection:
        rows_path, overflow_path, metadata_path = _paths(root, specification["symbol"])
        metadata = base._read(metadata_path); base._validate_seal(metadata, "shard_digest")
        if base._sha(rows_path) != metadata["rows_sha256"] \
                or base._sha(overflow_path) != metadata["overflow_sha256"]:
            raise StorePocError("bounded shard content changed")
        main.append(np.fromfile(rows_path, dtype=DTW_SAMPLE_DTYPE))
        overflow.append(np.fromfile(overflow_path, dtype=DTW_SAMPLE_DTYPE))
    return (
        np.concatenate(main) if main else np.empty(0, dtype=DTW_SAMPLE_DTYPE),
        np.concatenate(overflow) if overflow else np.empty(0, dtype=DTW_SAMPLE_DTYPE),
    )


def _semantic_metadata(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def execute(repository: Path, preregistration: Mapping[str, Any]) -> dict[str, Any]:
    validate_preregistration(repository, preregistration)
    loaded, _verified, source, packed_query = _prerequisites(repository)
    benchmark = source.load_benchmark()
    if benchmark is None:
        raise StorePocError("store POC requires benchmark")
    maximum = pd.Timestamp(loaded.manifest["provenance"]["benchmark_prefix"][
        "requested_cutoff"
    ])
    root = repository / OUTPUT_RELATIVE
    if root.is_symlink() or root.exists() and not root.is_dir():
        raise StorePocError("store POC output path differs")
    if root.exists():
        contract = base._read(root / "CONTRACT.json")
        if contract != preregistration:
            raise StorePocError("store POC resume contract differs")
        result_path = root / "RESULT.json"
        if result_path.exists():
            result = base._read(result_path); base._validate_seal(result)
            return result
    else:
        root.mkdir(parents=True)
        base._atomic(root / "CONTRACT.json", preregistration)
    lease_before = resident_file_identity_lease(base.RESIDENT_ROOT / "READY.json")
    started = perf_counter()
    first = _build(
        root / "build-a", preregistration["selection"], source=source,
        benchmark=benchmark, query=packed_query.representation,
        query_id=packed_query.episode_id, maximum_cutoff=maximum, packed=loaded,
    )
    second = _build(
        root / "build-b", preregistration["selection"], source=source,
        benchmark=benchmark, query=packed_query.representation,
        query_id=packed_query.episode_id, maximum_cutoff=maximum, packed=loaded,
    )
    rebuild_equal = _semantic_metadata(first) == _semantic_metadata(second)
    main_a, overflow_a = _load_build(root / "build-a", preregistration["selection"])
    main_b, overflow_b = _load_build(root / "build-b", preregistration["selection"])
    if not rebuild_equal or not np.array_equal(main_a, main_b) \
            or not np.array_equal(overflow_a, overflow_b):
        raise StorePocError("clean DTW shard rebuild differs")
    records_a = np.concatenate((main_a, overflow_a))
    records_b = np.concatenate((main_b, overflow_b))
    if not len(records_a):
        raise StorePocError("bounded DTW shard set is empty")
    dtw_sample_lower_bounds(packed_query.representation, records_a[:1])
    scan_seconds = []
    scans = []
    for records in (records_a, records_b):
        started_scan = perf_counter()
        values = dtw_sample_lower_bounds(packed_query.representation, records)
        scan_seconds.append(perf_counter() - started_scan); scans.append(values)
    for _ in range(2):
        started_scan = perf_counter()
        values = dtw_sample_lower_bounds(packed_query.representation, records_a)
        scan_seconds.append(perf_counter() - started_scan); scans.append(values)
    if any(not np.array_equal(scans[0], values) for values in scans[1:]):
        raise StorePocError("bounded DTW scan is nondeterministic")
    offsets: dict[tuple[str, str], int] = {}
    main_offset = overflow_offset = 0
    for specification, metadata in zip(preregistration["selection"], first, strict=True):
        symbol = specification["symbol"]
        for probe in metadata["scalar_probes"]:
            offset = (main_offset + probe["lane_index"] if probe["lane"] == "main"
                      else len(main_a) + overflow_offset + probe["lane_index"])
            offsets[(symbol, probe["episode_id"])] = offset
        main_offset += metadata["rows"]; overflow_offset += metadata["overflow_rows"]
    scalar_deltas = []
    for metadata in first:
        for probe in metadata["scalar_probes"]:
            offset = offsets[(metadata["symbol"], probe["episode_id"])]
            scalar_deltas.append(abs(
                float.fromhex(probe["scalar_bound_hex"]) - scans[0][offset]
            ))
    maximum_scalar_delta = max(scalar_deltas, default=0.0)
    eligible = int(loaded.manifest["row_count"]) + int(loaded.manifest["overflow_count"])
    median_scan = float(np.median(scan_seconds[1:]))
    projected_scan = median_scan * eligible / len(records_a)
    projected_bytes = eligible * DTW_SAMPLE_ROW_BYTES
    lease_after = resident_file_identity_lease(base.RESIDENT_ROOT / "READY.json")
    alignment = all(
        metadata["rows"] == specification["rows"]
        and metadata["overflow_rows"] == specification["overflow_rows"]
        and metadata["main_slice_digest"] == specification["main_slice_digest"]
        and metadata["overflow_slice_digest"] == specification["overflow_slice_digest"]
        for metadata, specification in zip(first, preregistration["selection"], strict=True)
    )
    gates = {
        "clean_rebuild_equal": rebuild_equal,
        "packed_alignment_passed": alignment
            and sum(row["rows"] for row in first) == len(main_a)
            and sum(row["overflow_rows"] for row in first) == len(overflow_a),
        "scalar_batch_passed": maximum_scalar_delta <= TOLERANCE,
        "scan_deterministic": True,
        "scan_performance_passed": projected_scan <= MAX_PROJECTED_SCAN_SECONDS,
        "capacity_passed": projected_bytes / 1024 ** 3 <= MAX_PROJECTED_STORE_GIB,
        "resident_lease_unchanged": lease_before["lease_digest"] == lease_after["lease_digest"],
    }
    passed = all(gates.values())
    state = {
        "schema_version": "m04r14-t14-10-wf03b-dtw-store-poc-result-v1",
        "status": "complete", "passed": passed, "gates": gates,
        "selection_digest": preregistration["selection_digest"],
        "symbols": len(first), "forced_overflow_symbols": sum(
            row["forced_overflow"] for row in preregistration["selection"]
        ),
        "rows": len(main_a), "overflow_rows": len(overflow_a),
        "zero_bound_rows": sum(row["zero_bound_rows"] for row in first),
        "build_digest": stable_hash(first),
        "shard_content_digest": stable_hash([{
            "symbol": row["symbol"], "rows_sha256": row["rows_sha256"],
            "overflow_sha256": row["overflow_sha256"],
        } for row in first]),
        "scalar_probes": len(scalar_deltas),
        "maximum_scalar_batch_delta": maximum_scalar_delta,
        "scan_seconds": scan_seconds,
        "median_warm_scan_seconds": median_scan,
        "projected_full_scan_seconds": projected_scan,
        "projected_store_bytes": projected_bytes,
        "projected_store_gib": projected_bytes / 1024 ** 3,
        "elapsed_seconds": perf_counter() - started,
        "resident_lease_digest": lease_after["lease_digest"],
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
        "full_store_build_authorized": passed,
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
    path = repository / PREREGISTRATION_RELATIVE
    if args.mode == "preregister":
        base._atomic(path, build_preregistration(repository)); return 0
    result = execute(repository, base._read(path))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
