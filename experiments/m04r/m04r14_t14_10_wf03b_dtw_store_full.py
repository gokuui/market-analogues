"""Build the preregistered full-universe immutable DTW-sample generation."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from hashlib import sha256
import json
import multiprocessing
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from numba import set_num_threads

from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.dtw_interval_bound import DtwIntervalBoundError, quantize_dtw_samples
from market_analogues.dtw_sample_store import (
    DTW_SAMPLE_DTYPE,
    DTW_SAMPLE_ROW_BYTES,
    dtw_sample_lower_bounds,
    dtw_sample_store_contract,
    load_dtw_sample_generation,
    make_dtw_sample_record_from_quantized,
    make_zero_dtw_sample_record,
    validate_dtw_sample_records,
    write_dtw_sample_generation_from_shards,
)
from market_analogues.exact_batch import sliding_exact_representations
from market_analogues.packed_bound_store import decode_episode_id
from market_analogues.representation import representation_input_digest
from market_analogues.resident_store import resident_file_identity_lease
from market_analogues.types import EpisodeKey, InstrumentKey, stable_hash

from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_10_wf03b_dtw_store_poc as bounded


SCHEMA = "m04r14-t14-10-wf03b-dtw-store-full-preregistration-v1"
OUTPUT_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf03b-dtw-store-full-v1"
)
PREREGISTRATION_RELATIVE = Path(
    "experiments/m04r/m04r14_t14_10_wf03b_dtw_store_full_v1_preregistered.json"
)
BOUNDED_VERIFICATION_RELATIVE = Path(
    "config/data/analogues/m04r14/"
    "t14-10-wf03b-dtw-store-poc-v2-verification/VERIFIED.json"
)
WORKERS = 8
BLOCK_ROWS = 4096
MAX_FULL_SCAN_SECONDS = 90.0
MAX_STORE_GIB = 3.1
RUNTIME_FILES = tuple(dict.fromkeys((
    "experiments/m04r/m04r14_t14_10_wf03b_dtw_store_full.py",
    *bounded.RUNTIME_FILES,
)))

_SOURCE: Any = None
_BENCHMARK: pd.DataFrame | None = None
_MAXIMUM_CUTOFF: pd.Timestamp | None = None
_PACKED: Any = None


class FullStoreError(RuntimeError):
    pass


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repository, text=True, capture_output=True,
        check=False,
    )
    if result.returncode:
        raise FullStoreError(f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _prerequisites(repository: Path) -> tuple[Any, dict[str, Any], Any, Any]:
    verification_path = repository / BOUNDED_VERIFICATION_RELATIVE
    verification = base._read(verification_path)
    base._validate_seal(verification, "verification_digest")
    required_gates = (
        "all_raw_rows_equal", "zero_rows_equal", "scalar_probes_equal",
        "scan_deterministic", "scan_performance_passed", "capacity_recomputed",
    )
    if verification.get("passed") is not True \
            or verification.get("full_store_build_authorized") is not True \
            or any(verification.get("gates", {}).get(key) is not True
                   for key in required_gates):
        raise FullStoreError("bounded independent verification does not authorize build")
    loaded, _bound_verification, source, query = bounded._prerequisites(repository)
    return loaded, verification, source, query


def _full_selection(loaded: Any) -> list[dict[str, Any]]:
    output = [
        {
            "symbol": symbol, "symbol_id": symbol_id,
            "source_prefix": loaded.manifest["provenance"]["source_prefixes"][symbol],
        }
        for symbol_id, symbol in enumerate(loaded.symbols)
    ]
    if [row["symbol_id"] for row in output] != list(range(len(loaded.symbols))) \
            or [row["symbol"] for row in output] != list(loaded.symbols):
        raise FullStoreError("full selection does not exactly cover packed symbol order")
    return output


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain"):
        raise FullStoreError("full-store preregistration requires a clean commit")
    loaded, verification, _source, query = _prerequisites(repository)
    selection = _full_selection(loaded)
    total_rows = int(loaded.manifest["row_count"])
    total_overflow = int(loaded.manifest["overflow_count"])
    state = {
        "schema_version": SCHEMA,
        "status": "frozen_before_full_store_build",
        "implementation_commit": _git(repository, "rev-parse", "HEAD"),
        "runtime_files": {
            path: base._sha(repository / path) for path in RUNTIME_FILES
        },
        "inputs": {
            "bounded_verification_digest": verification["verification_digest"],
            "bounded_verification_sha256": base._sha(
                repository / BOUNDED_VERIFICATION_RELATIVE
            ),
            "packed_generation_id": loaded.generation_id,
            "packed_manifest_digest": loaded.manifest["manifest_digest"],
            "packed_rows_sha256": loaded.manifest["rows_sha256"],
            "packed_overflow_sha256": loaded.manifest["overflow_sha256"],
            "packed_provenance_digest": loaded.manifest["provenance_digest"],
            "query_id": query.episode_id,
            "query_representation_digest": representation_input_digest(
                query.representation
            ),
        },
        "selection": selection,
        "selection_digest": stable_hash(selection),
        "store_contract": dtw_sample_store_contract(),
        "execution": {
            "symbols": len(selection), "workers": WORKERS,
            "numba_threads_per_worker": 1, "lookback": 252, "stride": 5,
            "representation_batch_size": 512, "scan_block_rows": BLOCK_ROWS,
            "max_full_scan_seconds": MAX_FULL_SCAN_SECONDS,
            "max_store_gib": MAX_STORE_GIB,
            "output_root": str((repository / OUTPUT_RELATIVE).resolve()),
            "resume_rule": "reuse only independently sealed and revalidated symbol shards",
        },
        "expected": {
            "rows": total_rows, "overflow_rows": total_overflow,
            "total_rows": total_rows + total_overflow,
            "store_bytes": (total_rows + total_overflow) * DTW_SAMPLE_ROW_BYTES,
        },
        "claims": {
            "historical_walk_forward_query_outcomes_opened": False,
            "final_period_result_opened": False,
            "production_promotion_authorized": False,
        },
    }
    return base._sealed(state, "preregistration_digest")


def validate_preregistration(
    repository: Path, preregistration: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], Any, Any]:
    base._validate_seal(preregistration, "preregistration_digest")
    loaded, verification, source, query = _prerequisites(repository)
    selection = _full_selection(loaded)
    expected = {
        "rows": int(loaded.manifest["row_count"]),
        "overflow_rows": int(loaded.manifest["overflow_count"]),
        "total_rows": int(loaded.manifest["row_count"])
        + int(loaded.manifest["overflow_count"]),
        "store_bytes": (
            int(loaded.manifest["row_count"])
            + int(loaded.manifest["overflow_count"])
        ) * DTW_SAMPLE_ROW_BYTES,
    }
    if preregistration.get("schema_version") != SCHEMA \
            or preregistration.get("selection") != selection \
            or preregistration.get("selection_digest") != stable_hash(selection) \
            or preregistration.get("store_contract") != dtw_sample_store_contract() \
            or preregistration.get("expected") != expected \
            or preregistration.get("inputs", {}).get(
                "bounded_verification_digest"
            ) != verification["verification_digest"] \
            or preregistration.get("inputs", {}).get(
                "query_representation_digest"
            ) != representation_input_digest(query.representation):
        raise FullStoreError("full-store preregistration differs")
    head = preregistration.get("implementation_commit")
    if type(head) is not str:
        raise FullStoreError("full-store implementation commit differs")
    _git(repository, "merge-base", "--is-ancestor", head, "HEAD")
    for path, digest in preregistration["runtime_files"].items():
        if base._sha(repository / path) != digest:
            raise FullStoreError(f"full-store runtime differs: {path}")
        blob = subprocess.run(
            ["git", "show", f"{head}:{path}"], cwd=repository,
            capture_output=True, check=False,
        )
        if blob.returncode or sha256(blob.stdout).hexdigest() != digest:
            raise FullStoreError(f"full-store implementation binding differs: {path}")
    return loaded, verification, source, query


def _write_progress(
    path: Path, *, status: str, completed: int, total: int,
    reused: int, rows: int, overflow_rows: int,
) -> None:
    base._atomic(path, {
        "schema_version": "m04r14-t14-10-wf03b-full-build-progress-v1",
        "status": status, "completed_symbols": completed,
        "total_symbols": total, "reused_symbols": reused,
        "rows": rows, "overflow_rows": overflow_rows,
    })


def _validated_shard(
    root: Path, specification: Mapping[str, Any],
) -> dict[str, Any] | None:
    rows_path, overflow_path, metadata_path = bounded._paths(
        root, specification["symbol"],
    )
    if not all(path.is_file() and not path.is_symlink()
               for path in (rows_path, overflow_path, metadata_path)):
        return None
    try:
        metadata = base._read(metadata_path)
        base._validate_seal(metadata, "shard_digest")
        main = np.fromfile(rows_path, dtype=DTW_SAMPLE_DTYPE)
        overflow = np.fromfile(overflow_path, dtype=DTW_SAMPLE_DTYPE)
        validate_dtw_sample_records(main)
        validate_dtw_sample_records(overflow)
        valid = all((
            metadata["symbol"] == specification["symbol"],
            metadata["symbol_id"] == specification["symbol_id"],
            metadata["source_prefix"] == specification["source_prefix"],
            metadata["rows"] == len(main),
            metadata["overflow_rows"] == len(overflow),
            metadata["rows_sha256"] == base._sha(rows_path),
            metadata["overflow_sha256"] == base._sha(overflow_path),
        ))
    except Exception:
        return None
    return metadata if valid else None


def _build_symbol(task: tuple[dict[str, Any], str]) -> dict[str, Any]:
    specification, root_value = task
    set_num_threads(1)
    if _SOURCE is None or _BENCHMARK is None or _MAXIMUM_CUTOFF is None \
            or _PACKED is None:
        raise FullStoreError("full-store worker is not initialized")
    root = Path(root_value)
    existing = _validated_shard(root, specification)
    if existing is not None:
        return existing
    symbol = specification["symbol"]
    symbol_id = int(specification["symbol_id"])
    key = InstrumentKey("nasdaq", symbol)
    full = _SOURCE.load(key)
    prefix = asdict(causal_prefix_digest(full, _MAXIMUM_CUTOFF))
    if prefix != specification["source_prefix"]:
        raise FullStoreError(f"source prefix changed: {symbol}")
    main_packed = bounded._slice(_PACKED.rows, symbol_id)
    overflow_packed = bounded._slice(_PACKED.overflow, symbol_id)
    main_ids = {
        decode_episode_id(row["episode_id"]): index
        for index, row in enumerate(main_packed)
    }
    overflow_ids = {
        decode_episode_id(row["episode_id"]): index
        for index, row in enumerate(overflow_packed)
    }
    if len(main_ids) != len(main_packed) or len(overflow_ids) != len(overflow_packed):
        raise FullStoreError("packed symbol contains duplicate episode")
    frame = full[full.timestamp <= _MAXIMUM_CUTOFF].reset_index(drop=True)
    batch = sliding_exact_representations(
        frame, _BENCHMARK, lookback=252, stride=5, batch_size=512,
    )
    main = np.zeros(len(main_packed), dtype=DTW_SAMPLE_DTYPE)
    overflow = np.zeros(len(overflow_packed), dtype=DTW_SAMPLE_DTYPE)
    seen_main: set[str] = set()
    seen_overflow: set[str] = set()
    zero_rows = 0
    for position, representation in zip(
        batch.positions, batch.representations, strict=True,
    ):
        cutoff = pd.Timestamp(frame.timestamp.iloc[int(position)])
        episode_id = EpisodeKey(key, cutoff, 252, "dense-v1").id
        if episode_id in main_ids:
            lane = main
            lane_index = main_ids[episode_id]
            seen_main.add(episode_id)
        elif episode_id in overflow_ids:
            lane = overflow
            lane_index = overflow_ids[episode_id]
            seen_overflow.add(episode_id)
        else:
            raise FullStoreError("materialized episode is outside packed symbol slice")
        try:
            record = make_dtw_sample_record_from_quantized(
                quantize_dtw_samples(representation)
            )
        except DtwIntervalBoundError:
            record = make_zero_dtw_sample_record()
            zero_rows += 1
        lane[lane_index] = record[0]
    if seen_main != set(main_ids) or seen_overflow != set(overflow_ids) \
            or len(batch.positions) != len(main_packed) + len(overflow_packed):
        raise FullStoreError("materialized/packed episode inventory differs")
    validate_dtw_sample_records(main)
    validate_dtw_sample_records(overflow)
    rows_path, overflow_path, metadata_path = bounded._paths(root, symbol)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    bounded._write_array(rows_path, main)
    bounded._write_array(overflow_path, overflow)
    state = {
        "schema_version": "m04r14-wf03b-dtw-store-full-shard-v1",
        "symbol": symbol, "symbol_id": symbol_id, "source_prefix": prefix,
        "rows": len(main), "overflow_rows": len(overflow),
        "rows_sha256": base._sha(rows_path),
        "overflow_sha256": base._sha(overflow_path),
        "packed_main_slice_digest": stable_hash(
            bounded._packed_rows_state(main_packed)
        ),
        "packed_overflow_slice_digest": stable_hash(
            bounded._packed_rows_state(overflow_packed)
        ),
        "zero_bound_rows": zero_rows,
    }
    metadata = base._sealed(state, "shard_digest")
    base._atomic(metadata_path, metadata)
    return metadata


def _build_shards(
    root: Path, selection: Sequence[dict[str, Any]], *, source: Any,
    benchmark: pd.DataFrame, query: Any, query_id: str,
    maximum_cutoff: pd.Timestamp, packed: Any,
) -> list[dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    del query, query_id
    global _SOURCE, _BENCHMARK, _MAXIMUM_CUTOFF, _PACKED
    _SOURCE = source
    _BENCHMARK = benchmark
    _MAXIMUM_CUTOFF = maximum_cutoff
    _PACKED = packed
    progress_path = root.parent / "PROGRESS.json"
    existing: dict[int, dict[str, Any]] = {}
    for specification in selection:
        value = _validated_shard(root, specification)
        if value is not None:
            existing[int(specification["symbol_id"])] = value
    results = dict(existing)
    rows = sum(int(value["rows"]) for value in existing.values())
    overflow_rows = sum(int(value["overflow_rows"]) for value in existing.values())
    _write_progress(
        progress_path, status="building", completed=len(results),
        total=len(selection), reused=len(existing), rows=rows,
        overflow_rows=overflow_rows,
    )
    missing = [row for row in selection if int(row["symbol_id"]) not in results]
    if missing:
        with ProcessPoolExecutor(
            max_workers=WORKERS,
            mp_context=multiprocessing.get_context("fork"),
        ) as executor:
            futures = {
                executor.submit(_build_symbol, (dict(row), str(root))): row
                for row in missing
            }
            for future in as_completed(futures):
                value = future.result()
                symbol_id = int(value["symbol_id"])
                if symbol_id in results:
                    raise FullStoreError("full-store worker returned duplicate symbol")
                results[symbol_id] = value
                rows += int(value["rows"])
                overflow_rows += int(value["overflow_rows"])
                _write_progress(
                    progress_path, status="building", completed=len(results),
                    total=len(selection), reused=len(existing), rows=rows,
                    overflow_rows=overflow_rows,
                )
    ordered = [results[int(row["symbol_id"])] for row in selection]
    _write_progress(
        progress_path, status="shards_complete", completed=len(ordered),
        total=len(selection), reused=len(existing), rows=rows,
        overflow_rows=overflow_rows,
    )
    return ordered


def _scan_generation(query: Any, generation: Any) -> tuple[float, str, int]:
    digest = sha256()
    count = 0
    started = perf_counter()
    for records in (generation.rows, generation.overflow):
        for first in range(0, len(records), BLOCK_ROWS):
            values = dtw_sample_lower_bounds(
                query, records[first:first + BLOCK_ROWS],
            )
            if not np.isfinite(values).all() or np.any(values < 0):
                raise FullStoreError("full-store scan produced invalid bound")
            digest.update(np.asarray(values, dtype="<f8").tobytes())
            count += len(values)
    return float(perf_counter() - started), digest.hexdigest(), count


def execute(repository: Path, preregistration: Mapping[str, Any]) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    loaded, verification, source, query = validate_preregistration(
        repository, preregistration,
    )
    benchmark = source.load_benchmark()
    if benchmark is None:
        raise FullStoreError("full-store build requires benchmark")
    maximum = pd.Timestamp(loaded.manifest["provenance"]["benchmark_prefix"][
        "requested_cutoff"
    ])
    root = repository / OUTPUT_RELATIVE
    if root.is_symlink() or root.exists() and not root.is_dir():
        raise FullStoreError("full-store output path differs")
    if root.exists():
        if base._read(root / "CONTRACT.json") != preregistration:
            raise FullStoreError("full-store resume contract differs")
        result_path = root / "RESULT.json"
        if result_path.exists():
            result = base._read(result_path)
            base._validate_seal(result)
            return result
    else:
        root.mkdir(parents=True)
        base._atomic(root / "CONTRACT.json", preregistration)
    lease_before = resident_file_identity_lease(base.RESIDENT_ROOT / "READY.json")
    started = perf_counter()
    try:
        metadata = _build_shards(
            root / "work" / "shards", preregistration["selection"],
            source=source, benchmark=benchmark, query=query.representation,
            query_id=query.episode_id, maximum_cutoff=maximum, packed=loaded,
        )
        row_shards = []
        overflow_shards = []
        for specification in preregistration["selection"]:
            row_path, overflow_path, _metadata_path = bounded._paths(
                root / "work" / "shards", specification["symbol"],
            )
            row_shards.append(row_path)
            overflow_shards.append(overflow_path)
        provenance = {
            "preregistration_digest": preregistration["preregistration_digest"],
            "bounded_verification_digest": verification["verification_digest"],
            "packed_generation_id": loaded.generation_id,
            "packed_provenance_digest": loaded.manifest["provenance_digest"],
            "selection_digest": preregistration["selection_digest"],
            "query_representation_digest": preregistration["inputs"][
                "query_representation_digest"
            ],
            "historical_walk_forward_query_outcomes_opened": False,
            "final_period_result_opened": False,
        }
        generation_id = write_dtw_sample_generation_from_shards(
            root / "store", row_shards, overflow_shards,
            packed_manifest=loaded.manifest, provenance=provenance,
        )
        generation = load_dtw_sample_generation(
            root / "store", generation_id, packed_manifest=loaded.manifest,
            verify_content=True, validate_records=True,
        )
        # Compile before the measured complete pass.
        warmup = generation.rows[:min(1, len(generation.rows))]
        if len(warmup):
            dtw_sample_lower_bounds(query.representation, warmup)
        scan_seconds, scan_digest, scanned = _scan_generation(
            query.representation, generation,
        )
        lease_after = resident_file_identity_lease(base.RESIDENT_ROOT / "READY.json")
        expected = preregistration["expected"]
        zero_rows = sum(int(row["zero_bound_rows"]) for row in metadata)
        gates = {
            "complete_symbol_inventory": bool(
                len(metadata) == preregistration["execution"]["symbols"]
                and [int(row["symbol_id"]) for row in metadata]
                == list(range(len(metadata)))
            ),
            "packed_row_alignment": bool(
                sum(int(row["rows"]) for row in metadata) == expected["rows"]
                and sum(int(row["overflow_rows"]) for row in metadata)
                == expected["overflow_rows"]
                and len(generation.rows) == expected["rows"]
                and len(generation.overflow) == expected["overflow_rows"]
            ),
            "content_addressed_generation_valid": bool(
                generation.manifest["manifest_digest"] == generation_id
                and generation.manifest["packed_generation"]["manifest_digest"]
                == loaded.manifest["manifest_digest"]
            ),
            "scan_complete": bool(scanned == expected["total_rows"]),
            "scan_performance_passed": bool(
                scan_seconds <= MAX_FULL_SCAN_SECONDS
            ),
            "capacity_passed": bool(
                generation.manifest["rows_bytes"]
                + generation.manifest["overflow_bytes"]
                == expected["store_bytes"]
                and expected["store_bytes"] / 1024 ** 3 <= MAX_STORE_GIB
            ),
            "resident_lease_unchanged": bool(
                lease_before["lease_digest"] == lease_after["lease_digest"]
            ),
        }
        passed = all(gates.values())
        state = {
            "schema_version": "m04r14-t14-10-wf03b-dtw-store-full-result-v1",
            "status": "complete", "passed": passed, "gates": gates,
            "generation_id": generation_id,
            "generation_manifest_digest": generation.manifest["manifest_digest"],
            "generation_provenance_digest": generation.manifest[
                "provenance_digest"
            ],
            "symbols": len(metadata), "rows": len(generation.rows),
            "overflow_rows": len(generation.overflow),
            "zero_bound_rows": zero_rows,
            "shard_metadata_digest": stable_hash(metadata),
            "scan_seconds": scan_seconds, "scan_digest": scan_digest,
            "store_bytes": expected["store_bytes"],
            "store_gib": expected["store_bytes"] / 1024 ** 3,
            "elapsed_seconds": float(perf_counter() - started),
            "resident_lease_digest": lease_after["lease_digest"],
            "historical_walk_forward_query_outcomes_opened": False,
            "final_period_result_opened": False,
            "production_promotion_authorized": False,
            "full_store_verification_authorized": passed,
        }
        result = base._sealed(state)
        base._atomic(root / "RESULT.json", result)
        _write_progress(
            root / "PROGRESS.json", status="complete",
            completed=len(metadata), total=len(metadata), reused=0,
            rows=len(generation.rows), overflow_rows=len(generation.overflow),
        )
        return result
    except Exception as exc:
        completed = rows = overflow_rows = 0
        progress_path = root / "PROGRESS.json"
        if progress_path.exists():
            try:
                progress = base._read(progress_path)
                completed = int(progress.get("completed_symbols", 0))
                rows = int(progress.get("rows", 0))
                overflow_rows = int(progress.get("overflow_rows", 0))
            except Exception:
                pass
        _write_progress(
            progress_path, status=f"failed:{type(exc).__name__}",
            completed=completed, total=len(preregistration["selection"]),
            reused=0, rows=rows, overflow_rows=overflow_rows,
        )
        raise


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
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
