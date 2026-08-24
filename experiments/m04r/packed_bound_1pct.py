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
import mmap
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
    write_packed_generation_from_shards,
)
from market_analogues.quantized_bound import QuantizedBoundError, quantize_bound_row
from market_analogues.representation import represent
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import EpisodeKey, InstrumentKey, stable_hash


SCHEMA_VERSION = "m04r-packed-bound-1pct-poc-v1"
SAMPLE_FRACTION = 0.01
FULL_PROJECTED_ROWS = 3_820_000
EVIDENCE_NONDETERMINISTIC_FIELDS = {
    "created_at", "build_seconds", "generation_seconds",
    "validation_seconds", "peak_rss_mb", "scan_peak_rss_mb",
    "validation_peak_rss_mb", "inherited_scan_ru_maxrss_mb", "result_digest",
}
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


def _write_evidence_html(path: Path, payload: dict[str, Any], label: str) -> None:
    metadata = {key: value for key, value in payload.items() if key not in {
        "warm_scans_first", "warm_scans_second", "sample",
    }}
    path.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>M04R packed bound {escape(label)}</title><style>body{{font-family:system-ui;max-width:1100px;margin:2rem auto}}pre{{white-space:pre-wrap}}</style></head><body><h1>M04R packed bound {escape(label)}: {'PASS' if payload['gate_passed'] else 'FAIL'}</h1><p>Deterministic real NASDAQ bound pack; no forward outcomes accessed. Full mode remains shadow-only.</p><p>Evidence <code>{payload['result_digest']}</code>.</p><pre>{escape(json.dumps(metadata, indent=2, sort_keys=True))}</pre></body></html>""")


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


def _current_rss_mb() -> float:
    try:
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    except (OSError, ValueError, IndexError):
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return resident_pages * os.sysconf("SC_PAGE_SIZE") / 1024 ** 2


def _scan(
    query: dict[str, Any], loaded: Any, block_rows: int,
    *, bounded_read: bool = False,
) -> dict[str, Any]:
    started = perf_counter()
    top = np.empty(0, dtype=np.float64)
    observed_peak_rss_mb = _current_rss_mb()
    eligible_rows = 0
    symbol_id = (
        loaded.symbols.index(query["symbol"])
        if query["symbol"] in loaded.symbols else None
    )
    pack_path = (
        loaded.root / "generations" / loaded.generation_id / "bound-rows.bin"
    )
    handle = pack_path.open("rb") if bounded_read else None
    try:
        for first in range(0, len(loaded.rows), block_rows):
            count = min(block_rows, len(loaded.rows) - first)
            raw = None
            if handle is not None:
                raw = os.pread(
                    handle.fileno(), count * PACK_DTYPE.itemsize,
                    first * PACK_DTYPE.itemsize,
                )
                if len(raw) != count * PACK_DTYPE.itemsize:
                    raise ValueError("short positional read from packed generation")
                block = np.frombuffer(raw, dtype=PACK_DTYPE, count=count)
            else:
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
                values = packed_lower_bounds(
                    query["representation"], selected,
                ).totals
                merged = np.r_[top, values]
                top = np.sort(merged)[:1_000]
                observed_peak_rss_mb = max(
                    observed_peak_rss_mb, _current_rss_mb(),
                )
                del values, merged
            del selected, block, raw
            if (
                not bounded_read and isinstance(loaded.rows, np.memmap)
                and hasattr(loaded.rows._mmap, "madvise")
            ):
                byte_start = first * PACK_DTYPE.itemsize
                byte_end = (
                    min(first + block_rows, len(loaded.rows))
                    * PACK_DTYPE.itemsize
                )
                page = mmap.PAGESIZE
                advised_start = byte_start - byte_start % page
                advised_end = min(
                    loaded.rows.nbytes,
                    ((byte_end + page - 1) // page) * page,
                )
                loaded.rows._mmap.madvise(
                    mmap.MADV_DONTNEED,
                    advised_start, advised_end - advised_start,
                )
    finally:
        if handle is not None:
            handle.close()
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
        "observed_peak_rss_mb": observed_peak_rss_mb,
    }


def _eligible_count(query: dict[str, Any], loaded: Any) -> dict[str, Any]:
    symbol_id = (
        loaded.symbols.index(query["symbol"])
        if query["symbol"] in loaded.symbols else None
    )
    total = 0
    overflow_total = 0
    for records, overflow in ((loaded.rows, False), (loaded.overflow, True)):
        eligible = records["cutoff_ns"] <= query["latest_ns"]
        if symbol_id is not None:
            eligible &= ~(
                (records["symbol_id"] == symbol_id)
                & (records["cutoff_ns"] >= query["query_start_ns"])
            )
        count = int(np.sum(eligible))
        total += count
        if overflow:
            overflow_total = count
    return {
        "query_episode_id": query["episode_id"],
        "eligible_rows": total,
        "full_eligible_rows": query["full_eligible_rows"],
        "overflow_eligible_rows": overflow_total,
        "row_accounting_matches": total == query["full_eligible_rows"],
    }


def _full_scan_worker(
    config_path: Path, output_root: Path, generation_id: str,
    worst_id: str, block_rows: int,
) -> dict[str, Any]:
    config = load_config(config_path)
    source = source_from_spec(config.datasets["nasdaq"])
    queries = _load_queries(config, source)
    selected = [query for query in queries if query["episode_id"] == worst_id]
    if len(selected) != 1:
        raise ValueError("full scan worker cannot resolve worst authority")
    loaded = load_packed_generation(
        output_root / "store", generation_id,
        verify_content=False, validate_records=False,
    )
    pack_path = (
        loaded.root / "generations" / loaded.generation_id / "bound-rows.bin"
    )
    cold_advised = _advise_cold(pack_path)
    cold = _scan(selected[0], loaded, block_rows, bounded_read=True)
    warm_first = [_scan(selected[0], loaded, block_rows, bounded_read=True)]
    warm_second = [_scan(selected[0], loaded, block_rows, bounded_read=True)]
    return _full_scan_payload(cold_advised, cold, warm_first, warm_second)


def _full_scan_payload(
    cold_advised: bool, cold: dict[str, Any],
    warm_first: list[dict[str, Any]], warm_second: list[dict[str, Any]],
) -> dict[str, Any]:
    scan_rows = (cold, *warm_first, *warm_second)
    return {
        "cold_cache_advised": cold_advised,
        "cold_scan": cold,
        "warm_scans_first": warm_first,
        "warm_scans_second": warm_second,
        "scan_peak_rss_mb": max(
            float(row["observed_peak_rss_mb"]) for row in scan_rows
        ),
        "inherited_ru_maxrss_mb": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        ),
        "scan_io_mode": "bounded positional reads over raw immutable pack",
    }


def _matching_shadow_generation(
    store_root: Path, row_count: int, overflow_count: int,
    symbols: list[str], provenance: dict[str, Any],
) -> str | None:
    for path in sorted((store_root / "generations").glob("*/manifest.json")):
        try:
            manifest = json.loads(path.read_text())
        except Exception:
            continue
        if (
            int(manifest.get("row_count", -1)) == row_count
            and int(manifest.get("overflow_count", -1)) == overflow_count
            and manifest.get("symbols") == symbols
            and manifest.get("provenance_digest") == stable_hash(provenance)
            and manifest.get("pack_contract_digest")
            == packed_bound_store_contract()["digest"]
        ):
            return str(manifest.get("manifest_digest"))
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-new-symbols", type=int)
    parser.add_argument("--block-rows", type=int, default=2_048)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--full-universe", action="store_true")
    parser.add_argument("--scan-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.block_rows < 1:
        raise ValueError("workers and block rows must be positive")
    config = load_config(args.config)
    if args.scan_only:
        if not args.full_universe:
            raise ValueError("scan-only is restricted to the full shadow generation")
        evidence_path = args.output_root / "packed-bound-full.json"
        previous = json.loads(evidence_path.read_text())
        _json(
            args.output_root / "packed-bound-full-inherited-rss-metric-failure.json",
            previous,
        )
        selection = previous.get("benchmark_selection") or {}
        with ProcessPoolExecutor(
            max_workers=1, mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            scan = executor.submit(
                _full_scan_worker, args.config, args.output_root,
                str(previous["generation_id"]),
                str(selection["query_episode_id"]), args.block_rows,
            ).result()
        previous.update({
            "cold_cache_advised": scan["cold_cache_advised"],
            "cold_scan": scan["cold_scan"],
            "warm_scans_first": scan["warm_scans_first"],
            "warm_scans_second": scan["warm_scans_second"],
            "maximum_warm_seconds": max(
                row["seconds"] for row in scan["warm_scans_second"]
            ),
            "projected_warm_seconds": max(
                row["projected_full_seconds"]
                for row in scan["warm_scans_second"]
            ),
            "projected_cold_seconds": scan["cold_scan"]["projected_full_seconds"],
            "scan_deterministic": (
                [row["top_1000_digest"] for row in scan["warm_scans_first"]]
                == [row["top_1000_digest"] for row in scan["warm_scans_second"]]
            ),
            "scan_io_mode": scan["scan_io_mode"],
            "scan_peak_rss_mb": scan["scan_peak_rss_mb"],
            "peak_rss_mb": scan["scan_peak_rss_mb"],
            "inherited_scan_ru_maxrss_mb": scan["inherited_ru_maxrss_mb"],
            "scan_rss_passed": scan["scan_peak_rss_mb"] <= 1_024,
            "warm_latency_passed": max(
                row["projected_full_seconds"]
                for row in scan["warm_scans_second"]
            ) <= 300,
            "cold_latency_passed": (
                scan["cold_scan"]["projected_full_seconds"] <= 600
            ),
            "scan_retry": {
                "reason": "replace inherited ru_maxrss with per-block current RSS",
                "archived_evidence": (
                    "packed-bound-full-inherited-rss-metric-failure.json"
                ),
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        previous["poc_passed"] = all((
            previous["capacity_passed"], previous["warm_latency_passed"],
            previous["cold_latency_passed"], previous["scan_deterministic"],
            previous["overflow_sidecar_exercised"],
            previous["authority_row_accounting_passed"],
            previous["scan_rss_passed"],
        ))
        previous["gate_passed"] = previous["poc_passed"]
        deterministic = {
            key: value for key, value in previous.items()
            if key not in EVIDENCE_NONDETERMINISTIC_FIELDS
        }
        previous["result_digest"] = stable_hash(deterministic)
        _json(evidence_path, previous)
        _write_evidence_html(
            args.output_root / "packed-bound-full.html",
            previous, "full shadow generation",
        )
        print(json.dumps({
            "gate_passed": previous["gate_passed"],
            "generation_id": previous["generation_id"],
            "cold_seconds": previous["projected_cold_seconds"],
            "warm_seconds": previous["projected_warm_seconds"],
            "scan_peak_rss_mb": previous["scan_peak_rss_mb"],
            "inherited_scan_ru_maxrss_mb": previous["inherited_scan_ru_maxrss_mb"],
            "result_digest": previous["result_digest"],
        }, indent=2))
        return 0 if previous["gate_passed"] else 2
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
    sample_fraction = 1.0 if args.full_universe else SAMPLE_FRACTION
    run_schema = (
        "m04r-packed-bound-full-build-v1"
        if args.full_universe else SCHEMA_VERSION
    )
    sample_size = math.ceil(len(universe) * sample_fraction)
    forced = sorted({
        str(row["symbol"]) for row in rank_evidence["overflow_examples"]
        if str(row["symbol"]) in tiers
    }) if not args.full_universe else []
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

    validated_metadata = {
        symbol: metadata for symbol in selected
        if (metadata := _validate_shard(symbol)) is not None
    }
    reusable = set(validated_metadata)
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
    if not remaining and len(validated_metadata) == len(selected):
        complete_metadata = [validated_metadata[symbol] for symbol in selected]
    else:
        complete_metadata = []
        for symbol in selected:
            metadata = _validate_shard(symbol)
            if metadata is not None:
                complete_metadata.append(metadata)
    if len(complete_metadata) != len(selected):
        payload = {
            "schema_version": run_schema, "complete": False,
            "sample_symbols": len(selected), "completed_symbols": len(complete_metadata),
            "new_symbols": len(results), "remaining_symbols": len(selected) - len(complete_metadata),
            "elapsed_seconds": perf_counter() - started,
        }
        _json(args.output_root / "interrupted.json", payload)
        print(json.dumps(payload, indent=2))
        return 3

    row_paths = []
    overflow_paths = []
    row_count = overflow_count = 0
    prefixes = {}
    for symbol in selected:
        rows_path, overflow_path, metadata_path = _shard_paths(symbol)
        metadata = json.loads(metadata_path.read_text())
        prefixes[symbol] = metadata["source_prefix"]
        row_paths.append(rows_path)
        overflow_paths.append(overflow_path)
        row_count += int(metadata["rows"])
        overflow_count += int(metadata["overflow_rows"])
    selection = {
        "method": (
            "complete sha256-ordered A/B universe"
            if args.full_universe else
            "sha256 order with known-overflow symbols forced inside fixed ceil(1%) size"
        ),
        "fraction": sample_fraction, "universe_count": len(universe),
        "sample_count": len(selected), "forced_overflow_symbols": forced,
        "symbols": selected,
    }
    provenance = {
        "rank_evidence_digest": rank_evidence["result_digest"],
        "source_prefixes": prefixes, "benchmark_prefix": observed_benchmark,
        "selection": selection, "selection_digest": stable_hash(selection),
    }
    generation_started = perf_counter()
    if args.full_universe:
        generation_id = _matching_shadow_generation(
            args.output_root / "store", row_count, overflow_count,
            selected, provenance,
        )
        if generation_id is None:
            generation_id = write_packed_generation_from_shards(
                args.output_root / "store", row_paths, overflow_paths,
                row_count, overflow_count, selected, provenance, activate=False,
            )
    else:
        rows = [np.fromfile(path, dtype=PACK_DTYPE) for path in row_paths if path.stat().st_size]
        overflow = [
            np.fromfile(path, dtype=OVERFLOW_DTYPE)
            for path in overflow_paths if path.stat().st_size
        ]
        row_array = np.concatenate(rows) if rows else np.empty(0, dtype=PACK_DTYPE)
        overflow_array = np.concatenate(overflow) if overflow else np.empty(0, dtype=OVERFLOW_DTYPE)
        generation_id = write_packed_generation(
            args.output_root / "store", row_array, overflow_array,
            selected, provenance,
        )
    generation_seconds = perf_counter() - generation_started
    validation_started = perf_counter()
    loaded = load_packed_generation(args.output_root / "store", generation_id)
    validation_seconds = perf_counter() - validation_started
    queries = _load_queries(config, source)
    authority_row_counts = (
        [_eligible_count(query, loaded) for query in queries]
        if args.full_universe else None
    )
    benchmark_queries = queries
    benchmark_selection = None
    if args.full_universe:
        one_percent = json.loads((
            args.output_root.parent / "packed-bound-1pct" / "packed-bound-1pct.json"
        ).read_text())
        worst_id = max(
            one_percent["warm_scans_second"],
            key=lambda row: float(row["projected_full_seconds"]),
        )["query_episode_id"]
        benchmark_queries = [
            query for query in queries if query["episode_id"] == worst_id
        ]
        if len(benchmark_queries) != 1:
            raise ValueError("cannot resolve worst projected 1% authority")
        benchmark_selection = {
            "method": "maximum query-specific projected warm seconds in sealed 1% evidence",
            "source_evidence_digest": one_percent["result_digest"],
            "query_episode_id": worst_id,
        }
    if args.full_universe:
        if isinstance(loaded.rows, np.memmap) and hasattr(loaded.rows._mmap, "madvise"):
            loaded.rows._mmap.madvise(mmap.MADV_DONTNEED)
        del loaded
        with ProcessPoolExecutor(
            max_workers=1, mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            scan_evidence = executor.submit(
                _full_scan_worker, args.config, args.output_root,
                generation_id, benchmark_queries[0]["episode_id"], args.block_rows,
            ).result()
        cold_advised = scan_evidence["cold_cache_advised"]
        cold = scan_evidence["cold_scan"]
        warm_first = scan_evidence["warm_scans_first"]
        warm_second = scan_evidence["warm_scans_second"]
        scan_peak_rss_mb = float(scan_evidence["scan_peak_rss_mb"])
        inherited_scan_ru_maxrss_mb = float(
            scan_evidence["inherited_ru_maxrss_mb"]
        )
    else:
        pack_path = loaded.root / "generations" / loaded.generation_id / "bound-rows.bin"
        cold_advised = _advise_cold(pack_path)
        cold = _scan(benchmark_queries[0], loaded, args.block_rows)
        warm_first = [_scan(query, loaded, args.block_rows) for query in benchmark_queries]
        warm_second = [_scan(query, loaded, args.block_rows) for query in benchmark_queries]
        scan_peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        inherited_scan_ru_maxrss_mb = scan_peak_rss_mb
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
        + math.ceil(FULL_PROJECTED_ROWS * overflow_count / max(row_count, 1))
        * OVERFLOW_DTYPE.itemsize
    )
    maximum_warm = max(row["seconds"] for row in warm_second)
    projected_warm = max(row["projected_full_seconds"] for row in warm_second)
    projected_cold = cold["projected_full_seconds"]
    deterministic = {
        "schema_version": run_schema,
        "pack_contract_digest": packed_bound_store_contract()["digest"],
        "generation_id": generation_id,
        "sample": selection,
        "rows": row_count, "overflow_rows": overflow_count,
        "eligible_rows": row_count + overflow_count,
        "pack_bytes": int(
            row_count * PACK_DTYPE.itemsize
            + overflow_count * OVERFLOW_DTYPE.itemsize
        ),
        "projected_full_rows": FULL_PROJECTED_ROWS,
        "projected_full_bytes": full_bytes,
        "projected_full_gib": full_bytes / 1024 ** 3,
        "cold_cache_advised": cold_advised,
        "cold_scan": cold,
        "warm_scans_first": warm_first,
        "warm_scans_second": warm_second,
        "full_authority_row_counts": authority_row_counts,
        "benchmark_selection": benchmark_selection,
        "authority_row_accounting_passed": (
            authority_row_counts is None
            or all(row["row_accounting_matches"] for row in authority_row_counts)
        ),
        "maximum_warm_seconds": maximum_warm,
        "projected_warm_seconds": projected_warm,
        "projected_cold_seconds": projected_cold,
        "scan_deterministic": scan_deterministic,
        "scan_io_mode": (
            scan_evidence["scan_io_mode"]
            if args.full_universe else "raw mmap blocks"
        ),
        "scan_rss_passed": scan_peak_rss_mb <= 1_024,
        "resume_evidence": resume_evidence,
        "capacity_passed": full_bytes <= 11 * 1024 ** 3,
        "warm_latency_passed": projected_warm <= 300,
        "cold_latency_passed": projected_cold <= 600,
        "overflow_sidecar_exercised": overflow_count > 0,
        "shadow_generation": args.full_universe,
        "real_forward_outcomes_accessed": False,
    }
    deterministic["poc_passed"] = all((
        deterministic["capacity_passed"], deterministic["warm_latency_passed"],
        deterministic["cold_latency_passed"], deterministic["scan_deterministic"],
        deterministic["overflow_sidecar_exercised"],
        deterministic["authority_row_accounting_passed"],
        deterministic["scan_rss_passed"],
    ))
    deterministic["gate_passed"] = deterministic["poc_passed"]
    payload = {
        **deterministic,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "build_seconds": perf_counter() - started,
        "generation_seconds": generation_seconds,
        "validation_seconds": validation_seconds,
        "peak_rss_mb": scan_peak_rss_mb,
        "scan_peak_rss_mb": scan_peak_rss_mb,
        "validation_peak_rss_mb": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        ),
        "inherited_scan_ru_maxrss_mb": inherited_scan_ru_maxrss_mb,
        "result_digest": stable_hash(deterministic),
    }
    evidence_stem = "packed-bound-full" if args.full_universe else "packed-bound-1pct"
    output = args.output_root / f"{evidence_stem}.json"
    _json(output, payload)
    html = args.output_root / f"{evidence_stem}.html"
    label = "full shadow generation" if args.full_universe else "1% POC"
    _write_evidence_html(html, payload, label)
    metadata = {key: value for key, value in payload.items() if key not in {
        "warm_scans_first", "warm_scans_second", "sample",
    }}
    print(json.dumps(metadata, indent=2))
    return 0 if payload["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
