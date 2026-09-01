"""Independently verify the full-universe immutable DTW-sample generation."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
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
    DtwSampleStoreError,
    dtw_sample_lower_bounds,
    load_dtw_sample_generation,
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


SCHEMA = "m04r14-t14-10-wf03b-dtw-store-full-verifier-preregistration-v1"
PRODUCER_ROOT = Path(
    "config/data/analogues/m04r14/t14-10-wf03b-dtw-store-full-v2"
)
SEED_ROOT = Path(
    "config/data/analogues/m04r14/t14-10-wf03b-dtw-store-full-v1/work/shards"
)
VERIFICATION_ROOT = Path(
    "config/data/analogues/m04r14/t14-10-wf03b-dtw-store-full-v2-verification"
)
PREREGISTRATION = Path(
    "experiments/m04r/"
    "verify_m04r14_t14_10_wf03b_dtw_store_full_v1_preregistered.json"
)
SAMPLE_SYMBOLS = 128
WORKERS = 8
FORWARD_BLOCK_ROWS = 4096
REVERSE_BLOCK_ROWS = 4093
MAX_SCAN_SECONDS = 90.0
RUNTIME_FILES = (
    "experiments/m04r/verify_m04r14_t14_10_wf03b_dtw_store_full.py",
    "experiments/m04r/m04r14_t14_10_wf03_feasibility.py",
    "src/market_analogues/adapters.py",
    "src/market_analogues/causal_prefix.py",
    "src/market_analogues/config.py",
    "src/market_analogues/context.py",
    "src/market_analogues/dtw_interval_bound.py",
    "src/market_analogues/dtw_sample_store.py",
    "src/market_analogues/exact_batch.py",
    "src/market_analogues/packed_bound_store.py",
    "src/market_analogues/representation.py",
    "src/market_analogues/resident_store.py",
    "src/market_analogues/types.py",
)

_SOURCE: Any = None
_BENCHMARK: pd.DataFrame | None = None
_PACKED: Any = None
_SAMPLES: Any = None
_MAXIMUM: pd.Timestamp | None = None


class FullStoreVerificationError(RuntimeError):
    pass


def _probe_query_ids() -> list[str]:
    return [str(probe[1]) for probe in base.PROBES]


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repository, text=True, capture_output=True,
        check=False,
    )
    if result.returncode:
        raise FullStoreVerificationError(f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _stem(symbol: str) -> str:
    return sha256(symbol.encode()).hexdigest()


def _paths(root: Path, symbol: str) -> tuple[Path, Path, Path]:
    stem = root / _stem(symbol)
    return (
        stem.with_suffix(".rows.bin"),
        stem.with_suffix(".overflow.bin"),
        stem.with_suffix(".json"),
    )


def _slice(records: np.ndarray, symbol_id: int) -> np.ndarray:
    values = records["symbol_id"]
    first = int(np.searchsorted(values, symbol_id, side="left"))
    last = int(np.searchsorted(values, symbol_id, side="right"))
    return records[first:last]


def _sha(path: Path, digest: Any | None = None) -> str:
    own = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            own.update(block)
            if digest is not None:
                digest.update(block)
    return own.hexdigest()


def _independent_validate(records: np.ndarray) -> None:
    if records.dtype != DTW_SAMPLE_DTYPE:
        raise FullStoreVerificationError("independent DTW dtype differs")
    expected = np.arange(64, dtype=np.uint8)
    for first in range(0, len(records), 16_384):
        block = records[first:first + 16_384]
        centers = np.asarray(block["centers"])
        radii = np.asarray(block["radii"])
        orders = np.asarray(block["orders"])
        presence_byte = np.asarray(block["presence"], dtype=np.uint8)
        presence = ((presence_byte[:, None] >> np.arange(4)) & 1).astype(bool)
        padding = np.asarray(block["padding"]).view(np.uint8).reshape(len(block), -1)
        if not np.isfinite(centers).all() or not np.isfinite(radii).all() \
                or np.any(radii < 0) or np.any(presence_byte > 15) \
                or np.any(padding != 0):
            raise FullStoreVerificationError("independent DTW values differ")
        if np.any(np.sort(orders, axis=2) != expected):
            raise FullStoreVerificationError("independent DTW permutation differs")
        ordered = np.take_along_axis(centers, orders.astype(np.int64), axis=2)
        if np.any(ordered[:, :, 1:] < ordered[:, :, :-1]):
            raise FullStoreVerificationError("independent DTW order differs")
        absent = ~presence
        if np.any(centers[absent] != 0) or np.any(radii[absent] != 0):
            raise FullStoreVerificationError("independent absent channel differs")


def _producer(repository: Path) -> tuple[dict[str, Any], dict[str, Any], Any, Any]:
    root = repository / PRODUCER_ROOT
    contract = base._read(root / "CONTRACT.json")
    result = base._read(root / "RESULT.json")
    base._validate_seal(contract, "preregistration_digest")
    base._validate_seal(result)
    if result.get("passed") is not True \
            or result.get("full_store_verification_authorized") is not True \
            or not all(result.get("gates", {}).values()):
        raise FullStoreVerificationError("full-store producer does not authorize verifier")
    resident = base._resident()
    packed = load_packed_generation(
        Path(resident["store_root"]), base.GENERATION_ID,
        expected_provenance_digest=base.PROVENANCE_DIGEST,
        verify_content=False, validate_records=False,
    )
    samples = load_dtw_sample_generation(
        root / "store", result["generation_id"],
        packed_manifest=packed.manifest, verify_content=False,
        validate_records=False,
    )
    if len(samples.rows) != result["rows"] \
            or len(samples.overflow) != result["overflow_rows"]:
        raise FullStoreVerificationError("full-store producer counts differ")
    return contract, result, packed, samples


def _metadata(repository: Path, symbols: Sequence[str]) -> list[dict[str, Any]]:
    output = []
    for symbol_id, symbol in enumerate(symbols):
        path = _paths(repository / SEED_ROOT, symbol)[2]
        value = base._read(path)
        base._validate_seal(value, "shard_digest")
        if value.get("symbol") != symbol or value.get("symbol_id") != symbol_id:
            raise FullStoreVerificationError("seed metadata order differs")
        output.append(value)
    return output


def _select(
    symbols: Sequence[str], metadata: Sequence[Mapping[str, Any]], packed: Any,
) -> list[dict[str, Any]]:
    overflow_ids = set(int(value) for value in packed.overflow["symbol_id"])
    zero_ids = {
        index for index, row in enumerate(metadata)
        if int(row["zero_bound_rows"]) > 0
    }
    forced = sorted(overflow_ids | zero_ids)
    remaining = sorted(
        (index for index in range(len(symbols)) if index not in set(forced)),
        key=lambda index: (
            stable_hash({
                "purpose": "full-store-independent-raw-v1",
                "symbol": symbols[index],
            }),
            index,
        ),
    )
    selected = sorted((forced + remaining)[:SAMPLE_SYMBOLS])
    if not overflow_ids.issubset(selected) or not zero_ids.issubset(selected):
        raise FullStoreVerificationError("raw sample omits forced edge symbol")
    main_offsets = []
    overflow_offsets = []
    main_offset = 0
    overflow_offset = 0
    for row in metadata:
        main_offsets.append(main_offset)
        overflow_offsets.append(overflow_offset)
        main_offset += int(row["rows"])
        overflow_offset += int(row["overflow_rows"])
    return [{
        "symbol": symbols[index], "symbol_id": index,
        "rows": int(metadata[index]["rows"]),
        "overflow_rows": int(metadata[index]["overflow_rows"]),
        "main_offset": main_offsets[index],
        "overflow_offset": overflow_offsets[index],
        "zero_bound_rows": int(metadata[index]["zero_bound_rows"]),
        "source_prefix": metadata[index]["source_prefix"],
    } for index in selected]


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain"):
        raise FullStoreVerificationError("verifier preregistration requires clean commit")
    contract, result, packed, samples = _producer(repository)
    metadata = _metadata(repository, packed.symbols)
    selection = _select(packed.symbols, metadata, packed)
    _registry, by_id = base._registry(repository)
    queries = []
    for query_id in _probe_query_ids():
        _source, _episode, _request, packed_query = base._context(
            repository, by_id[query_id],
        )
        queries.append({
            "query_id": query_id,
            "representation_digest": representation_input_digest(
                packed_query.representation
            ),
        })
    state = {
        "schema_version": SCHEMA, "status": "frozen_before_independent_verification",
        "implementation_commit": _git(repository, "rev-parse", "HEAD"),
        "runtime_files": {
            path: base._sha(repository / path) for path in RUNTIME_FILES
        },
        "inputs": {
            "producer_contract_digest": contract["preregistration_digest"],
            "producer_contract_sha256": base._sha(
                repository / PRODUCER_ROOT / "CONTRACT.json"
            ),
            "producer_result_digest": result["result_digest"],
            "producer_result_sha256": base._sha(
                repository / PRODUCER_ROOT / "RESULT.json"
            ),
            "generation_id": samples.generation_id,
            "generation_manifest_digest": samples.manifest["manifest_digest"],
            "packed_manifest_digest": packed.manifest["manifest_digest"],
            "packed_provenance_digest": packed.manifest["provenance_digest"],
        },
        "raw_selection": selection,
        "raw_selection_digest": stable_hash(selection),
        "queries": queries, "queries_digest": stable_hash(queries),
        "execution": {
            "all_symbols": len(packed.symbols), "sample_symbols": SAMPLE_SYMBOLS,
            "workers": WORKERS, "forward_block_rows": FORWARD_BLOCK_ROWS,
            "reverse_block_rows": REVERSE_BLOCK_ROWS,
            "max_scan_seconds": MAX_SCAN_SECONDS,
            "required_complete_scans": len(queries) * 2,
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
) -> tuple[dict[str, Any], Any, Any, list[dict[str, Any]]]:
    base._validate_seal(preregistration, "preregistration_digest")
    contract, result, packed, samples = _producer(repository)
    metadata = _metadata(repository, packed.symbols)
    selection = _select(packed.symbols, metadata, packed)
    expected_query_ids = _probe_query_ids()
    frozen_query_ids = [
        row.get("query_id") for row in preregistration.get("queries", [])
    ]
    expected_execution = {
        "all_symbols": len(packed.symbols), "sample_symbols": SAMPLE_SYMBOLS,
        "workers": WORKERS, "forward_block_rows": FORWARD_BLOCK_ROWS,
        "reverse_block_rows": REVERSE_BLOCK_ROWS,
        "max_scan_seconds": MAX_SCAN_SECONDS,
        "required_complete_scans": len(expected_query_ids) * 2,
    }
    if preregistration.get("schema_version") != SCHEMA \
            or preregistration.get("raw_selection") != selection \
            or preregistration.get("raw_selection_digest") != stable_hash(selection) \
            or frozen_query_ids != expected_query_ids \
            or preregistration.get("queries_digest") \
            != stable_hash(preregistration.get("queries", [])) \
            or preregistration.get("execution") != expected_execution \
            or preregistration.get("inputs", {}).get("producer_contract_digest") \
            != contract["preregistration_digest"] \
            or preregistration.get("inputs", {}).get("producer_result_digest") \
            != result["result_digest"] \
            or preregistration.get("inputs", {}).get("generation_id") \
            != samples.generation_id:
        raise FullStoreVerificationError("verifier preregistration differs")
    head = preregistration.get("implementation_commit")
    if type(head) is not str:
        raise FullStoreVerificationError("verifier implementation commit differs")
    _git(repository, "merge-base", "--is-ancestor", head, "HEAD")
    for path, digest in preregistration["runtime_files"].items():
        if base._sha(repository / path) != digest:
            raise FullStoreVerificationError(f"verifier runtime differs: {path}")
        blob = subprocess.run(
            ["git", "show", f"{head}:{path}"], cwd=repository,
            capture_output=True, check=False,
        )
        if blob.returncode or sha256(blob.stdout).hexdigest() != digest:
            raise FullStoreVerificationError(f"verifier Git binding differs: {path}")
    _registry, by_id = base._registry(repository)
    queries = []
    for frozen in preregistration["queries"]:
        source, _episode, _request, packed_query = base._context(
            repository, by_id[frozen["query_id"]],
        )
        if representation_input_digest(packed_query.representation) \
                != frozen["representation_digest"]:
            raise FullStoreVerificationError("frozen verifier query differs")
        queries.append({"source": source, "packed": packed_query})
    return result, packed, samples, queries


def _audit_shards(
    repository: Path, packed: Any, samples: Any,
) -> dict[str, Any]:
    main_digest = sha256()
    overflow_digest = sha256()
    metadata_rows = []
    rows = 0
    overflow_rows = 0
    zero_rows = 0
    for symbol_id, symbol in enumerate(packed.symbols):
        row_path, overflow_path, metadata_path = _paths(repository / SEED_ROOT, symbol)
        metadata = base._read(metadata_path)
        base._validate_seal(metadata, "shard_digest")
        main = np.fromfile(row_path, dtype=DTW_SAMPLE_DTYPE)
        overflow = np.fromfile(overflow_path, dtype=DTW_SAMPLE_DTYPE)
        _independent_validate(main)
        _independent_validate(overflow)
        rows_sha256 = _sha(row_path, main_digest)
        overflow_sha256 = _sha(overflow_path, overflow_digest)
        if metadata.get("symbol") != symbol \
                or metadata.get("symbol_id") != symbol_id \
                or metadata.get("source_prefix") \
                != packed.manifest["provenance"]["source_prefixes"][symbol] \
                or metadata.get("rows") != len(main) \
                or metadata.get("overflow_rows") != len(overflow) \
                or metadata.get("rows_sha256") != rows_sha256 \
                or metadata.get("overflow_sha256") != overflow_sha256:
            raise FullStoreVerificationError("independent shard audit differs")
        rows += len(main)
        overflow_rows += len(overflow)
        zero_rows += int(metadata["zero_bound_rows"])
        metadata_rows.append(metadata)
    manifest = samples.manifest
    gates = {
        "all_shard_metadata_valid": len(metadata_rows) == len(packed.symbols),
        "shard_counts_match_packed": rows == int(packed.manifest["row_count"])
        and overflow_rows == int(packed.manifest["overflow_count"]),
        "ordered_main_concatenation_equal": main_digest.hexdigest()
        == manifest["rows_sha256"],
        "ordered_overflow_concatenation_equal": overflow_digest.hexdigest()
        == manifest["overflow_sha256"],
    }
    if not all(gates.values()):
        raise FullStoreVerificationError("full shard/concatenation audit failed")
    return {
        "gates": gates, "rows": rows, "overflow_rows": overflow_rows,
        "zero_bound_rows": zero_rows,
        "metadata_digest": stable_hash(metadata_rows),
        "main_sha256": main_digest.hexdigest(),
        "overflow_sha256": overflow_digest.hexdigest(),
    }


def _raw_symbol(specification: Mapping[str, Any]) -> dict[str, Any]:
    set_num_threads(1)
    if _SOURCE is None or _BENCHMARK is None or _PACKED is None \
            or _SAMPLES is None or _MAXIMUM is None:
        raise FullStoreVerificationError("raw verifier worker is not initialized")
    symbol = specification["symbol"]
    symbol_id = int(specification["symbol_id"])
    key = InstrumentKey("nasdaq", symbol)
    full = _SOURCE.load(key)
    if asdict(causal_prefix_digest(full, _MAXIMUM)) \
            != specification["source_prefix"]:
        raise FullStoreVerificationError("raw verifier source prefix differs")
    main_packed = _slice(_PACKED.rows, symbol_id)
    overflow_packed = _slice(_PACKED.overflow, symbol_id)
    main_ids = {
        decode_episode_id(row["episode_id"]): index
        for index, row in enumerate(main_packed)
    }
    overflow_ids = {
        decode_episode_id(row["episode_id"]): index
        for index, row in enumerate(overflow_packed)
    }
    main = np.zeros(len(main_packed), dtype=DTW_SAMPLE_DTYPE)
    overflow = np.zeros(len(overflow_packed), dtype=DTW_SAMPLE_DTYPE)
    seen: set[str] = set()
    zero_rows = 0
    frame = full[full.timestamp <= _MAXIMUM].reset_index(drop=True)
    batch = sliding_exact_representations(
        frame, _BENCHMARK, lookback=252, stride=5, batch_size=512,
    )
    for position, representation in zip(
        batch.positions, batch.representations, strict=True,
    ):
        cutoff = pd.Timestamp(frame.timestamp.iloc[int(position)])
        episode_id = EpisodeKey(key, cutoff, 252, "dense-v1").id
        if episode_id in seen:
            raise FullStoreVerificationError("raw verifier episode repeats")
        seen.add(episode_id)
        if episode_id in main_ids:
            lane = main
            lane_index = main_ids[episode_id]
        elif episode_id in overflow_ids:
            lane = overflow
            lane_index = overflow_ids[episode_id]
        else:
            raise FullStoreVerificationError("raw verifier episode is outside packed slice")
        try:
            record = make_dtw_sample_record_from_quantized(
                quantize_dtw_samples(representation)
            )
        except DtwIntervalBoundError:
            record = make_zero_dtw_sample_record()
            zero_rows += 1
        lane[lane_index] = record[0]
    if seen != set(main_ids) | set(overflow_ids) \
            or len(main) != specification["rows"] \
            or len(overflow) != specification["overflow_rows"] \
            or zero_rows != specification["zero_bound_rows"]:
        raise FullStoreVerificationError("raw verifier inventory differs")
    main_offset = int(specification["main_offset"])
    overflow_offset = int(specification["overflow_offset"])
    stored_main = _SAMPLES.rows[main_offset:main_offset + len(main)]
    stored_overflow = _SAMPLES.overflow[
        overflow_offset:overflow_offset + len(overflow)
    ]
    if not np.array_equal(main, stored_main) \
            or not np.array_equal(overflow, stored_overflow):
        raise FullStoreVerificationError("raw reconstruction/store bytes differ")
    return {
        "symbol": symbol, "symbol_id": symbol_id,
        "rows": len(main), "overflow_rows": len(overflow),
        "zero_bound_rows": zero_rows,
        "main_sha256": sha256(main.tobytes()).hexdigest(),
        "overflow_sha256": sha256(overflow.tobytes()).hexdigest(),
    }


def _raw_reconstruction(
    source: Any, benchmark: pd.DataFrame, packed: Any, samples: Any,
    maximum: pd.Timestamp, selection: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    global _SOURCE, _BENCHMARK, _PACKED, _SAMPLES, _MAXIMUM
    _SOURCE, _BENCHMARK, _PACKED, _SAMPLES = source, benchmark, packed, samples
    _MAXIMUM = maximum
    with ProcessPoolExecutor(
        max_workers=WORKERS, mp_context=multiprocessing.get_context("fork"),
    ) as executor:
        results = list(executor.map(_raw_symbol, reversed(selection)))
    results.sort(key=lambda row: row["symbol_id"])
    return results


def _scan(
    query: Any, samples: Any, *, block_rows: int, reverse: bool,
) -> tuple[np.ndarray, float, str]:
    total = len(samples.rows) + len(samples.overflow)
    output = np.empty(total, dtype=np.float64)
    started = perf_counter()
    base_offset = 0
    for records in (samples.rows, samples.overflow):
        starts = range(0, len(records), block_rows)
        if reverse:
            starts = reversed(list(starts))
        for first in starts:
            values = dtw_sample_lower_bounds(
                query, records[first:first + block_rows],
            )
            output[base_offset + first:base_offset + first + len(values)] = values
        base_offset += len(records)
    elapsed = float(perf_counter() - started)
    return output, elapsed, sha256(output.astype("<f8", copy=False).tobytes()).hexdigest()


def _guard_fixtures(samples: Any, packed: Any) -> dict[str, bool]:
    if not len(samples.rows):
        raise FullStoreVerificationError("guard fixture requires a real row")
    malformed = np.asarray(samples.rows[:1]).copy()
    malformed["orders"][0, 0, 0] = malformed["orders"][0, 0, 1]
    independent_rejected = False
    production_rejected = False
    try:
        _independent_validate(malformed)
    except FullStoreVerificationError:
        independent_rejected = True
    try:
        validate_dtw_sample_records(malformed)
    except DtwSampleStoreError:
        production_rejected = True
    changed = dict(packed.manifest)
    changed["rows_sha256"] = "0" * 64
    binding_rejected = False
    try:
        load_dtw_sample_generation(
            samples.root, samples.generation_id, packed_manifest=changed,
            verify_content=False, validate_records=False,
        )
    except DtwSampleStoreError:
        binding_rejected = True
    return {
        "independent_malformed_order_rejected": independent_rejected,
        "production_malformed_order_rejected": production_rejected,
        "changed_packed_binding_rejected": binding_rejected,
    }


def verify(repository: Path, preregistration: Mapping[str, Any]) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    result, packed, samples, queries = validate_preregistration(
        repository, preregistration,
    )
    lease_before = resident_file_identity_lease(base.RESIDENT_ROOT / "READY.json")
    started = perf_counter()
    fully_loaded = load_dtw_sample_generation(
        repository / PRODUCER_ROOT / "store", samples.generation_id,
        packed_manifest=packed.manifest, verify_content=True, validate_records=True,
    )
    shard_audit = _audit_shards(repository, packed, fully_loaded)
    source = queries[0]["source"]
    benchmark = source.load_benchmark()
    if benchmark is None:
        raise FullStoreVerificationError("raw verifier benchmark is absent")
    maximum = pd.Timestamp(packed.manifest["provenance"]["benchmark_prefix"][
        "requested_cutoff"
    ])
    raw_cases = _raw_reconstruction(
        source, benchmark, packed, fully_loaded, maximum,
        preregistration["raw_selection"],
    )
    if len(fully_loaded.rows):
        dtw_sample_lower_bounds(queries[0]["packed"].representation,
                                fully_loaded.rows[:1])
    scans = []
    for frozen, context in zip(preregistration["queries"], queries, strict=True):
        forward, forward_seconds, forward_digest = _scan(
            context["packed"].representation, fully_loaded,
            block_rows=FORWARD_BLOCK_ROWS, reverse=False,
        )
        reverse, reverse_seconds, reverse_digest = _scan(
            context["packed"].representation, fully_loaded,
            block_rows=REVERSE_BLOCK_ROWS, reverse=True,
        )
        equal = np.array_equal(forward, reverse)
        if not equal:
            raise FullStoreVerificationError("forward/reverse full scan differs")
        scans.append({
            "query_id": frozen["query_id"],
            "forward_seconds": forward_seconds,
            "reverse_seconds": reverse_seconds,
            "forward_digest": forward_digest,
            "reverse_digest": reverse_digest,
            "equal": equal,
        })
    guards = _guard_fixtures(fully_loaded, packed)
    lease_after = resident_file_identity_lease(base.RESIDENT_ROOT / "READY.json")
    gates = {
        **shard_audit["gates"],
        "producer_metadata_digest_equal": shard_audit["metadata_digest"]
        == result["shard_metadata_digest"],
        "producer_zero_rows_equal": shard_audit["zero_bound_rows"]
        == result["zero_bound_rows"],
        "raw_sample_complete": len(raw_cases) == SAMPLE_SYMBOLS,
        "raw_overflow_complete": sum(row["overflow_rows"] for row in raw_cases)
        == result["overflow_rows"],
        "raw_zero_rows_complete": sum(row["zero_bound_rows"] for row in raw_cases)
        == result["zero_bound_rows"],
        "all_full_scans_equal": len(scans)
        == preregistration["execution"]["required_complete_scans"] // 2
        and all(row["equal"] for row in scans),
        "producer_scan_digest_equal": scans[0]["forward_digest"]
        == result["scan_digest"],
        "scan_performance_passed": all(
            max(row["forward_seconds"], row["reverse_seconds"])
            <= MAX_SCAN_SECONDS for row in scans
        ),
        "guard_fixtures_passed": all(guards.values()),
        "capacity_recomputed": (
            len(fully_loaded.rows) + len(fully_loaded.overflow)
        ) * DTW_SAMPLE_ROW_BYTES == result["store_bytes"],
        "resident_lease_unchanged": lease_before["lease_digest"]
        == lease_after["lease_digest"],
    }
    if not all(gates.values()):
        raise FullStoreVerificationError("independent full-store gate failed")
    state = {
        "schema_version": "m04r14-t14-10-wf03b-dtw-store-full-verification-v1",
        "status": "verified", "passed": True, "gates": gates,
        "producer_result_digest": result["result_digest"],
        "generation_id": fully_loaded.generation_id,
        "shard_audit": shard_audit,
        "raw_cases": raw_cases, "raw_case_digest": stable_hash(raw_cases),
        "scans": scans, "scan_digest": stable_hash(scans),
        "guard_fixtures": guards,
        "elapsed_seconds": float(perf_counter() - started),
        "resident_lease_digest": lease_after["lease_digest"],
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
        "retrieval_integration_authorized": True,
    }
    return base._sealed(state, "verification_digest")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preregister", "run"))
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    repository = args.repository.resolve(strict=True)
    path = repository / PREREGISTRATION
    if args.mode == "preregister":
        if args.dry_run:
            raise FullStoreVerificationError("preregistration dry-run is unsupported")
        base._atomic(path, build_preregistration(repository))
        return 0
    receipt = verify(repository, base._read(path))
    if not args.dry_run:
        root = repository / VERIFICATION_ROOT
        if root.exists() or root.is_symlink():
            raise FullStoreVerificationError("verification root already exists")
        root.mkdir(parents=True)
        base._atomic(root / "VERIFIED.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
