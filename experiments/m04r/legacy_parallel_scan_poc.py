"""Development-only process-partition POC for the frozen legacy-v1 proposal scan."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from html import escape
import json
import multiprocessing
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import numba
import numpy as np
import pandas as pd

from market_analogues.adapters import source_from_spec
from market_analogues.config import load_config
from market_analogues.episodes import build_episode
from market_analogues.packed_bound_search import (
    DEFAULT_ROUTE_QUOTAS, PackedBoundQuery, _eligible_mask, _empty_entries,
    _entries, _finalize, _stable_bounded, packed_bound_search_contract,
)
from market_analogues.packed_bound_store import (
    PACK_DTYPE, load_packed_generation, packed_lower_bounds,
)
from market_analogues.representation import represent
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import InstrumentKey, stable_hash


SCHEMA = "m04r-legacy-parallel-scan-poc-v1"
EXPECTED_QUERY_ID = "cb2bdccd1386790ef8a16bd0"
EXPECTED_CANDIDATE_DIGEST = "c8f51e202205367bf3cfda6a62cd3bcbe0e29e32f5b809d7e88431a194692555"
EXPECTED_RESULT_DIGEST = "bb36c3b770446bc1a62aa3552ab2ca1ccbe1a8c04a90d4db01cedb6e8180b75b"


def _current_rss_mb() -> float:
    resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE") / 1024 ** 2


def _advise_cold(path: Path) -> bool:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return False
    with path.open("rb") as handle:
        os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return True


def _partitions(row_count: int, block_rows: int, processes: int) -> list[tuple[int, ...]]:
    offsets = list(range(0, row_count, block_rows))
    return [tuple(offsets[index::processes]) for index in range(processes)]


def _scan_partition(
    store_root: str, generation_id: str, query: PackedBoundQuery,
    offsets: tuple[int, ...], block_rows: int, block_order: str,
    numba_threads: int,
) -> dict[str, Any]:
    numba.set_num_threads(numba_threads)
    loaded = load_packed_generation(
        Path(store_root), generation_id, verify_content=False, validate_records=False,
    )
    symbol_id = loaded.symbols.index(query.symbol) if query.symbol in loaded.symbols else None
    heaps = {route: _empty_entries() for route in DEFAULT_ROUTE_QUOTAS}
    eligible_rows = 0
    peak_rss = _current_rss_mb()
    ordered_offsets = offsets if block_order == "forward" else tuple(reversed(offsets))
    pack_path = loaded.root / "generations" / generation_id / str(loaded.manifest["rows_file"])
    with pack_path.open("rb") as handle:
        for first in ordered_offsets:
            count = min(block_rows, len(loaded.rows) - first)
            raw = os.pread(handle.fileno(), count * PACK_DTYPE.itemsize, first * PACK_DTYPE.itemsize)
            if len(raw) != count * PACK_DTYPE.itemsize:
                raise ValueError("short partition positional read")
            block = np.frombuffer(raw, dtype=PACK_DTYPE, count=count)
            selected = block[_eligible_mask(block, query, symbol_id)]
            eligible_rows += len(selected)
            if len(selected):
                bounded = packed_lower_bounds(query.representation, selected)
                route_values = {"composite": bounded.totals, **bounded.components}
                for route, quota in DEFAULT_ROUTE_QUOTAS.items():
                    heaps[route] = _stable_bounded(
                        heaps[route],
                        _entries(
                            selected, bounded.totals,
                            np.asarray(route_values[route], dtype=np.float64),
                            overflow=False,
                        ),
                        quota,
                    )
            peak_rss = max(peak_rss, _current_rss_mb())
    return {"heaps": heaps, "eligible_rows": eligible_rows, "peak_rss_mb": peak_rss}


def _merge_partition_heaps(
    partitions: list[dict[str, Any]], loaded: Any, query: PackedBoundQuery,
) -> dict[str, Any]:
    heaps = {route: _empty_entries() for route in DEFAULT_ROUTE_QUOTAS}
    for partition in partitions:
        for route, quota in DEFAULT_ROUTE_QUOTAS.items():
            heaps[route] = _stable_bounded(heaps[route], partition["heaps"][route], quota)
    overflow = np.asarray(loaded.overflow)
    symbol_id = loaded.symbols.index(query.symbol) if query.symbol in loaded.symbols else None
    selected_overflow = overflow[_eligible_mask(overflow, query, symbol_id)]
    if len(selected_overflow):
        zeros = np.zeros(len(selected_overflow), dtype=np.float64)
        incoming = _entries(selected_overflow, zeros, zeros, overflow=True)
        for route, quota in DEFAULT_ROUTE_QUOTAS.items():
            heaps[route] = _stable_bounded(heaps[route], incoming, quota)
    candidates, route_counts, candidate_digest = _finalize(heaps, loaded.symbols)
    eligible_main = sum(int(partition["eligible_rows"]) for partition in partitions)
    deterministic = {
        "schema_version": "m04r-global-bound-proposal-v1",
        "contract_digest": packed_bound_search_contract()["digest"],
        "generation_id": loaded.generation_id,
        "query_episode_id": query.episode_id,
        "rows_scanned": len(loaded.rows) + len(loaded.overflow),
        "eligible_rows": eligible_main + len(selected_overflow),
        "eligible_main_rows": eligible_main,
        "eligible_overflow_rows": len(selected_overflow),
        "route_counts": route_counts, "route_quotas": DEFAULT_ROUTE_QUOTAS,
        "candidate_digest": candidate_digest, "real_forward_outcomes_accessed": False,
    }
    return {
        **deterministic, "candidate_count": len(candidates),
        "result_digest": stable_hash(deterministic),
        "worker_peak_rss_sum_mb": sum(float(row["peak_rss_mb"]) for row in partitions),
        "aggregate_process_rss_mb": (
            sum(float(row["peak_rss_mb"]) for row in partitions) + _current_rss_mb()
        ),
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processes", type=int, choices=(2, 4), required=True)
    parser.add_argument("--numba-threads", type=int, default=1)
    parser.add_argument("--block-rows", type=int, default=4_096)
    args = parser.parse_args()
    config = load_config(args.config)
    source = source_from_spec(config.datasets["nasdaq"])
    episode = build_episode(
        source, InstrumentKey("nasdaq", "CAKE"), "2026-03-30", 252, "dense-v1",
    )
    query = PackedBoundQuery(
        episode.key.id, "CAKE", int(pd.Timestamp(episode.bars.timestamp.iloc[0]).value),
        int(latest_eligible_cutoff(episode, 60).value), represent(episode), ("A", "B"),
    )
    if query.episode_id != EXPECTED_QUERY_ID:
        raise ValueError("development query differs")
    build = json.loads((args.full_root / "packed-bound-full.json").read_text())
    generation_id = str(build["generation_id"])
    store_root = args.full_root / "store"
    loaded = load_packed_generation(store_root, generation_id, verify_content=True, validate_records=False)
    pack_path = loaded.root / "generations" / generation_id / str(loaded.manifest["rows_file"])
    partitions = _partitions(len(loaded.rows), args.block_rows, args.processes)
    advised_cold = _advise_cold(pack_path)
    context = multiprocessing.get_context("spawn")
    scans = []
    with ProcessPoolExecutor(max_workers=args.processes, mp_context=context) as executor:
        for label, order in (("cold", "forward"), ("warm_first", "reverse"), ("warm_second", "forward")):
            started = perf_counter()
            futures = [executor.submit(
                _scan_partition, str(store_root), generation_id, query, offsets,
                args.block_rows, order, args.numba_threads,
            ) for offsets in partitions]
            partition_results = [future.result() for future in futures]
            merged = _merge_partition_heaps(partition_results, loaded, query)
            scans.append({"label": label, "seconds": perf_counter() - started, **merged})
            print(
                f"[{label}] seconds={scans[-1]['seconds']:.2f} "
                f"rss_sum={merged['worker_peak_rss_sum_mb']:.2f}", flush=True,
            )
    invariant_fields = (
        "generation_id", "query_episode_id", "rows_scanned", "eligible_rows",
        "eligible_main_rows", "eligible_overflow_rows", "route_counts",
        "route_quotas", "candidate_count", "candidate_digest", "result_digest",
    )
    repeated = all(
        all(scan[field] == scans[0][field] for field in invariant_fields)
        for scan in scans[1:]
    )
    gates = {
        "cold_advice_applied": advised_cold,
        "three_scan_invariance": repeated,
        "legacy_candidate_digest_exact": scans[0]["candidate_digest"] == EXPECTED_CANDIDATE_DIGEST,
        "legacy_result_digest_exact": scans[0]["result_digest"] == EXPECTED_RESULT_DIGEST,
        "cold_at_most_120_seconds": scans[0]["seconds"] <= 120.0,
        "warm_second_at_most_60_seconds": scans[2]["seconds"] <= 60.0,
        "aggregate_process_rss_at_most_1024_mib": max(
            row["aggregate_process_rss_mb"] for row in scans
        ) <= 1_024.0,
    }
    deterministic = {
        "schema_version": SCHEMA, "query_episode_id": query.episode_id,
        "generation_id": generation_id, "processes": args.processes,
        "numba_threads_per_process": args.numba_threads,
        "block_rows": args.block_rows, "advised_cold": advised_cold,
        "scans": scans, "gates": gates, "passed": all(gates.values()),
        "query_stock_prefix": asdict(source.causal_prefix_fingerprint(
            InstrumentKey("nasdaq", "CAKE"), "2026-03-30",
        )),
        "real_forward_outcomes_accessed": False,
    }
    payload = {**deterministic, "created_at": datetime.now(timezone.utc).isoformat(),
               "result_digest": stable_hash(deterministic)}
    _write(args.output, payload)
    status = "PASS" if payload["passed"] else "FAIL"
    args.output.with_suffix(".html").write_text(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>Legacy parallel scan POC</title></head><body>"
        f"<h1>{status}</h1><p>Development-only CAKE process-partition scan; "
        f"candidate semantics must remain byte-identical.</p><pre>{escape(json.dumps(payload, indent=2, sort_keys=True))}"
        "</pre></body></html>"
    )
    print(json.dumps({"passed": payload["passed"], "gates": gates,
                      "result_digest": payload["result_digest"]}, indent=2))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
