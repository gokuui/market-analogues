"""Build and benchmark an overflow-stratified 1% immutable bound pack."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
import math
import multiprocessing
import os
from pathlib import Path
import resource
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from market_analogues.adapters import source_from_spec
from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.config import load_config
from market_analogues.episodes import build_episode
from market_analogues.exact_batch import sliding_exact_representations
from market_analogues.packed_bound_store import (
    OVERFLOW_DTYPE, PACK_DTYPE, decode_episode_id, load_packed_generation,
    make_overflow_record, make_packed_record, packed_bound_store_contract,
    packed_lower_bounds, write_packed_generation,
)
from market_analogues.quantized_bound import QuantizedBoundError, quantize_bound_row
from market_analogues.representation import represent
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import EpisodeKey, InstrumentKey, stable_hash


SCHEMA_VERSION = "m04r-packed-bound-1pct-poc-v1"
SAMPLE_FRACTION = 0.01
FULL_PROJECTED_ROWS = 3_820_000
_SOURCE: Any = None
_BENCHMARK: pd.DataFrame | None = None
_MAXIMUM_CUTOFF: pd.Timestamp | None = None
_SYMBOL_IDS: dict[str, int] = {}
_TIERS: dict[str, str] = {}
_SHARD_ROOT: Path | None = None


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _stem(symbol: str) -> str:
    return sha256(symbol.encode()).hexdigest()


def _shard_paths(symbol: str) -> tuple[Path, Path, Path]:
    assert _SHARD_ROOT is not None
    root = _SHARD_ROOT / _stem(symbol)
    return root.with_suffix(".rows.bin"), root.with_suffix(".overflow.bin"), root.with_suffix(".json")


def _validate_shard(symbol: str) -> dict[str, Any] | None:
    rows_path, overflow_path, metadata_path = _shard_paths(symbol)
    if not all(path.exists() for path in (rows_path, overflow_path, metadata_path)):
        return None
    try:
        metadata = json.loads(metadata_path.read_text())
        expected_prefix = asdict(causal_prefix_digest(
            _SOURCE.load(InstrumentKey("nasdaq", symbol)), _MAXIMUM_CUTOFF,
        ))
        valid = (
            metadata.get("schema_version") == f"{SCHEMA_VERSION}-shard"
            and metadata.get("pack_contract_digest") == packed_bound_store_contract()["digest"]
            and metadata.get("symbol") == symbol
            and metadata.get("symbol_id") == _SYMBOL_IDS[symbol]
            and metadata.get("quality_tier") == _TIERS[symbol]
            and metadata.get("source_prefix") == expected_prefix
            and rows_path.stat().st_size == int(metadata.get("rows", -1)) * PACK_DTYPE.itemsize
            and overflow_path.stat().st_size == int(metadata.get("overflow_rows", -1)) * OVERFLOW_DTYPE.itemsize
            and _sha(rows_path) == metadata.get("rows_sha256")
            and _sha(overflow_path) == metadata.get("overflow_sha256")
        )
    except Exception:
        return None
    return metadata if valid else None


def _build_symbol(symbol: str) -> dict[str, Any]:
    if _SOURCE is None or _BENCHMARK is None or _MAXIMUM_CUTOFF is None:
        raise RuntimeError("packed POC worker is not initialized")
    existing = _validate_shard(symbol)
    if existing is not None:
        return {**existing, "reused": True}
    rows_path, overflow_path, metadata_path = _shard_paths(symbol)
    key = InstrumentKey("nasdaq", symbol)
    full = _SOURCE.load(key)
    prefix = asdict(causal_prefix_digest(full, _MAXIMUM_CUTOFF))
    frame = full[full.timestamp <= _MAXIMUM_CUTOFF].reset_index(drop=True)
    started = perf_counter()
    batch = sliding_exact_representations(
        frame, _BENCHMARK, lookback=252, stride=5, batch_size=512,
    )
    encoded: list[np.ndarray] = []
    overflow: list[np.ndarray] = []
    for position, representation in zip(batch.positions, batch.representations):
        cutoff = pd.Timestamp(frame.timestamp.iloc[position])
        episode_id = EpisodeKey(key, cutoff, 252, "dense-v1").id
        try:
            encoded.append(make_packed_record(
                episode_id, int(cutoff.value), _SYMBOL_IDS[symbol],
                _TIERS[symbol], quantize_bound_row(representation),
            ))
        except QuantizedBoundError:
            overflow.append(make_overflow_record(
                episode_id, int(cutoff.value), _SYMBOL_IDS[symbol], _TIERS[symbol],
            ))
    rows = np.concatenate(encoded) if encoded else np.empty(0, dtype=PACK_DTYPE)
    sidecar = np.concatenate(overflow) if overflow else np.empty(0, dtype=OVERFLOW_DTYPE)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    for path, values in ((rows_path, rows), (overflow_path, sidecar)):
        temporary = path.with_suffix(path.suffix + ".tmp")
        values.tofile(temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
    metadata = {
        "schema_version": f"{SCHEMA_VERSION}-shard",
        "pack_contract_digest": packed_bound_store_contract()["digest"],
        "symbol": symbol, "symbol_id": _SYMBOL_IDS[symbol],
        "quality_tier": _TIERS[symbol], "source_prefix": prefix,
        "rows": len(rows), "overflow_rows": len(sidecar),
        "rows_sha256": _sha(rows_path),
        "overflow_sha256": _sha(overflow_path),
        "elapsed_seconds": perf_counter() - started,
    }
    _json(metadata_path, metadata)
    return {**metadata, "reused": False}


def _load_queries(config: Any, source: Any) -> list[dict[str, Any]]:
    paths = sorted((
        config.artifact_dir / "gate12" / "authorities" / "nasdaq" / "cases"
    ).glob("*.json"))
    if len(paths) != 12:
        raise ValueError(f"require 12 authorities; found {len(paths)}")
    output = []
    for path in paths:
        authority = json.loads(path.read_text())
        meta = authority["query"]
        episode = build_episode(
            source, InstrumentKey("nasdaq", str(meta["symbol"])),
            str(meta["cutoff"]), int(meta["lookback"]),
            str(meta["representation_version"]),
        )
        output.append({
            "episode_id": episode.key.id,
            "symbol": episode.key.instrument.source_symbol,
            "query_start_ns": int(pd.Timestamp(episode.bars.timestamp.iloc[0]).value),
            "latest_ns": int(latest_eligible_cutoff(episode, 60).value),
            "full_eligible_rows": int(
                authority["certificate"]["eligible_candidates"]
            ),
            "representation": represent(episode),
        })
    return output


def _advise_cold(path: Path) -> bool:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return False
    with path.open("rb") as handle:
        os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return True


def _scan(query: dict[str, Any], loaded: Any, block_rows: int) -> dict[str, Any]:
    started = perf_counter()
    top = np.empty(0, dtype=np.float64)
    eligible_rows = 0
    symbol_id = loaded.symbols.index(query["symbol"] ) if query["symbol"] in loaded.symbols else None
    for first in range(0, len(loaded.rows), block_rows):
        block = loaded.rows[first:first + block_rows]
        eligible = block["cutoff_ns"] <= query["latest_ns"]
        if symbol_id is not None:
            eligible &= ~(
                (block["symbol_id"] == symbol_id)
                & (block["cutoff_ns"] >= query["query_start_ns"])
            )
        selected = block[eligible]
        eligible_rows += len(selected)
        if len(selected):
            values = packed_lower_bounds(query["representation"], selected).totals
            merged = np.r_[top, values]
            top = np.sort(merged)[:1_000]
    sidecar = loaded.overflow
    overflow_eligible = sidecar["cutoff_ns"] <= query["latest_ns"]
    if symbol_id is not None:
        overflow_eligible &= ~(
            (sidecar["symbol_id"] == symbol_id)
            & (sidecar["cutoff_ns"] >= query["query_start_ns"])
        )
    overflow_count = int(np.sum(overflow_eligible))
    eligible_rows += overflow_count
    if overflow_count:
        top = np.sort(np.r_[top, np.zeros(overflow_count)])[:1_000]
    elapsed = perf_counter() - started
    factor = query["full_eligible_rows"] / eligible_rows if eligible_rows else float("inf")
    return {
        "query_episode_id": query["episode_id"],
        "eligible_rows": eligible_rows,
        "full_eligible_rows": query["full_eligible_rows"],
        "projection_factor": factor,
        "overflow_eligible_rows": overflow_count,
        "top_1000_digest": sha256(np.asarray(top, dtype="<f8").tobytes()).hexdigest(),
        "minimum_bound": float(top[0]) if len(top) else None,
        "seconds": elapsed,
        "projected_full_seconds": elapsed * factor,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-new-symbols", type=int)
    parser.add_argument("--block-rows", type=int, default=2_048)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.block_rows < 1:
        raise ValueError("workers and block rows must be positive")
    config = load_config(args.config)
    source = source_from_spec(config.datasets["nasdaq"])
    rank_evidence = json.loads((
        config.artifact_dir / "poc" / "m04r" / "quantized-bound-rank-full.json"
    ).read_text())
    quality = pd.read_parquet(config.artifact_dir / "quality" / "nasdaq.parquet")
    tiers = {
        str(row.symbol): str(row.tier) for row in quality.itertuples(index=False)
        if str(row.tier) in {"A", "B"}
    }
    universe = sorted(tiers, key=lambda symbol: sha256(
        f"m04r-packed-bound-1pct:{symbol}".encode(),
    ).hexdigest())
    sample_size = math.ceil(len(universe) * SAMPLE_FRACTION)
    forced = sorted({
        str(row["symbol"]) for row in rank_evidence["overflow_examples"]
        if str(row["symbol"]) in tiers
    })
    selected = forced + [symbol for symbol in universe if symbol not in forced]
    selected = selected[:sample_size]
    selected.sort(key=lambda symbol: sha256(
        f"m04r-packed-bound-1pct:{symbol}".encode(),
    ).hexdigest())
    maximum_cutoff = pd.Timestamp(rank_evidence["benchmark_prefix"]["requested_cutoff"])
    benchmark = source.load_benchmark()
    observed_benchmark = asdict(causal_prefix_digest(benchmark, maximum_cutoff))
    if observed_benchmark != rank_evidence["benchmark_prefix"]:
        raise ValueError("benchmark causal prefix differs from rank evidence")

    global _SOURCE, _BENCHMARK, _MAXIMUM_CUTOFF, _SYMBOL_IDS, _TIERS, _SHARD_ROOT
    _SOURCE, _BENCHMARK, _MAXIMUM_CUTOFF = source, benchmark, maximum_cutoff
    _SYMBOL_IDS = {symbol: index for index, symbol in enumerate(selected)}
    _TIERS = tiers
    _SHARD_ROOT = args.output_root / "work" / "shards"
    if args.restart and (args.output_root / "work").exists():
        import shutil
        shutil.rmtree(args.output_root / "work")

    reusable = {symbol for symbol in selected if _validate_shard(symbol) is not None}
    remaining = [symbol for symbol in selected if symbol not in reusable]
    if args.max_new_symbols is not None:
        remaining = remaining[:args.max_new_symbols]
    started = perf_counter()
    if args.workers == 1:
        results = [_build_symbol(symbol) for symbol in remaining]
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("fork"),
        ) as executor:
            results = list(executor.map(_build_symbol, remaining))
    complete_metadata = []
    for symbol in selected:
        metadata = _validate_shard(symbol)
        if metadata is not None:
            complete_metadata.append(metadata)
    if len(complete_metadata) != len(selected):
        payload = {
            "schema_version": SCHEMA_VERSION, "complete": False,
            "sample_symbols": len(selected), "completed_symbols": len(complete_metadata),
            "new_symbols": len(results), "remaining_symbols": len(selected) - len(complete_metadata),
            "elapsed_seconds": perf_counter() - started,
        }
        _json(args.output_root / "interrupted.json", payload)
        print(json.dumps(payload, indent=2))
        return 3

    rows = []
    overflow = []
    prefixes = {}
    for symbol in selected:
        rows_path, overflow_path, metadata_path = _shard_paths(symbol)
        metadata = json.loads(metadata_path.read_text())
        prefixes[symbol] = metadata["source_prefix"]
        if metadata["rows"]:
            rows.append(np.fromfile(rows_path, dtype=PACK_DTYPE))
        if metadata["overflow_rows"]:
            overflow.append(np.fromfile(overflow_path, dtype=OVERFLOW_DTYPE))
    row_array = np.concatenate(rows) if rows else np.empty(0, dtype=PACK_DTYPE)
    overflow_array = np.concatenate(overflow) if overflow else np.empty(0, dtype=OVERFLOW_DTYPE)
    selection = {
        "method": "sha256 order with known-overflow symbols forced inside fixed ceil(1%) size",
        "fraction": SAMPLE_FRACTION, "universe_count": len(universe),
        "sample_count": len(selected), "forced_overflow_symbols": forced,
        "symbols": selected,
    }
    provenance = {
        "rank_evidence_digest": rank_evidence["result_digest"],
        "source_prefixes": prefixes, "benchmark_prefix": observed_benchmark,
        "selection": selection, "selection_digest": stable_hash(selection),
    }
    generation_started = perf_counter()
    generation_id = write_packed_generation(
        args.output_root / "store", row_array, overflow_array,
        selected, provenance,
    )
    generation_seconds = perf_counter() - generation_started
    validation_started = perf_counter()
    loaded = load_packed_generation(args.output_root / "store")
    validation_seconds = perf_counter() - validation_started
    queries = _load_queries(config, source)
    pack_path = loaded.root / "generations" / loaded.generation_id / "bound-rows.bin"
    cold_advised = _advise_cold(pack_path)
    cold = _scan(queries[0], loaded, args.block_rows)
    warm_first = [_scan(query, loaded, args.block_rows) for query in queries]
    warm_second = [_scan(query, loaded, args.block_rows) for query in queries]
    scan_deterministic = [row["top_1000_digest"] for row in warm_first] == [
        row["top_1000_digest"] for row in warm_second
    ]
    interrupted_path = args.output_root / "interrupted.json"
    resume_evidence = None
    if interrupted_path.exists():
        interrupted = json.loads(interrupted_path.read_text())
        resume_evidence = {
            "schema_version": interrupted.get("schema_version"),
            "sample_symbols": interrupted.get("sample_symbols"),
            "interrupted_completed_symbols": interrupted.get("completed_symbols"),
            "interrupted_remaining_symbols": interrupted.get("remaining_symbols"),
            "resumed_completed_symbols": len(complete_metadata),
            "resume_completed": len(complete_metadata) == len(selected),
        }
    full_bytes = (
        FULL_PROJECTED_ROWS * PACK_DTYPE.itemsize
        + math.ceil(FULL_PROJECTED_ROWS * len(overflow_array) / max(len(row_array), 1))
        * OVERFLOW_DTYPE.itemsize
    )
    maximum_warm = max(row["seconds"] for row in warm_second)
    projected_warm = max(row["projected_full_seconds"] for row in warm_second)
    projected_cold = cold["projected_full_seconds"]
    deterministic = {
        "schema_version": SCHEMA_VERSION,
        "pack_contract_digest": packed_bound_store_contract()["digest"],
        "generation_id": generation_id,
        "sample": selection,
        "rows": len(row_array), "overflow_rows": len(overflow_array),
        "eligible_rows": len(row_array) + len(overflow_array),
        "pack_bytes": int(row_array.nbytes + overflow_array.nbytes),
        "projected_full_rows": FULL_PROJECTED_ROWS,
        "projected_full_bytes": full_bytes,
        "projected_full_gib": full_bytes / 1024 ** 3,
        "cold_cache_advised": cold_advised,
        "cold_scan": cold,
        "warm_scans_first": warm_first,
        "warm_scans_second": warm_second,
        "maximum_warm_seconds": maximum_warm,
        "projected_warm_seconds": projected_warm,
        "projected_cold_seconds": projected_cold,
        "scan_deterministic": scan_deterministic,
        "resume_evidence": resume_evidence,
        "capacity_passed": full_bytes <= 11 * 1024 ** 3,
        "warm_latency_passed": projected_warm <= 300,
        "cold_latency_passed": projected_cold <= 600,
        "overflow_sidecar_exercised": len(overflow_array) > 0,
        "real_forward_outcomes_accessed": False,
    }
    deterministic["poc_passed"] = all((
        deterministic["capacity_passed"], deterministic["warm_latency_passed"],
        deterministic["cold_latency_passed"], deterministic["scan_deterministic"],
        deterministic["overflow_sidecar_exercised"],
    ))
    payload = {
        **deterministic,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "build_seconds": perf_counter() - started,
        "generation_seconds": generation_seconds,
        "validation_seconds": validation_seconds,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "result_digest": stable_hash(deterministic),
    }
    output = args.output_root / "packed-bound-1pct.json"
    _json(output, payload)
    metadata = {key: value for key, value in payload.items() if key not in {
        "warm_scans_first", "warm_scans_second", "sample",
    }}
    html = args.output_root / "packed-bound-1pct.html"
    html.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>M04R packed bound 1%</title><style>body{{font-family:system-ui;max-width:1100px;margin:2rem auto}}pre{{white-space:pre-wrap}}</style></head><body><h1>M04R packed bound 1% POC: {'PASS' if payload['poc_passed'] else 'FAIL'}</h1><p>Overflow-stratified deterministic real NASDAQ sample; no forward outcomes accessed.</p><p>Evidence <code>{payload['result_digest']}</code>.</p><pre>{escape(json.dumps(metadata, indent=2, sort_keys=True))}</pre></body></html>""")
    print(json.dumps(metadata, indent=2))
    return 0 if payload["poc_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
