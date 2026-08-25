"""Development-only bounded-thread POC for the frozen legacy-v1 proposal scan."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import resource
from time import perf_counter
from typing import Any, Callable, Iterable

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


SCHEMA = "m04r-legacy-threaded-scan-poc-v1"
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


def _bounded_ordered_map(
    executor: ThreadPoolExecutor, function: Callable[[int], Any],
    values: Iterable[int], maximum_in_flight: int,
) -> Iterable[Any]:
    """Yield in input order while retaining at most ``maximum_in_flight`` tasks."""
    iterator = iter(values)
    pending: list[Future[Any]] = []
    for _ in range(maximum_in_flight):
        try:
            pending.append(executor.submit(function, next(iterator)))
        except StopIteration:
            break
    while pending:
        future = pending.pop(0)
        yield future.result()
        try:
            pending.append(executor.submit(function, next(iterator)))
        except StopIteration:
            pass


def _score_block(
    descriptor: int, row_count: int, query: PackedBoundQuery,
    symbol_id: int | None, first: int, block_rows: int,
) -> tuple[int, dict[str, np.ndarray]]:
    numba.set_num_threads(1)
    count = min(block_rows, row_count - first)
    raw = os.pread(descriptor, count * PACK_DTYPE.itemsize, first * PACK_DTYPE.itemsize)
    if len(raw) != count * PACK_DTYPE.itemsize:
        raise ValueError("short threaded positional read")
    block = np.frombuffer(raw, dtype=PACK_DTYPE, count=count)
    selected = block[_eligible_mask(block, query, symbol_id)]
    if not len(selected):
        return 0, {route: _empty_entries() for route in DEFAULT_ROUTE_QUOTAS}
    bounded = packed_lower_bounds(query.representation, selected)
    route_values = {"composite": bounded.totals, **bounded.components}
    entries = {
        route: _entries(
            selected, bounded.totals,
            np.asarray(route_values[route], dtype=np.float64), overflow=False,
        ) for route in DEFAULT_ROUTE_QUOTAS
    }
    return len(selected), entries


def _scan(
    executor: ThreadPoolExecutor, loaded: Any, query: PackedBoundQuery,
    *, block_rows: int, block_order: str, threads: int,
) -> dict[str, Any]:
    started = perf_counter()
    heaps = {route: _empty_entries() for route in DEFAULT_ROUTE_QUOTAS}
    eligible_main = 0
    peak_rss = _current_rss_mb()
    symbol_id = loaded.symbols.index(query.symbol) if query.symbol in loaded.symbols else None
    offsets = list(range(0, len(loaded.rows), block_rows))
    if block_order == "reverse":
        offsets.reverse()
    pack_path = loaded.root / "generations" / loaded.generation_id / str(loaded.manifest["rows_file"])
    with pack_path.open("rb") as handle:
        function = lambda first: _score_block(
            handle.fileno(), len(loaded.rows), query, symbol_id, first, block_rows,
        )
        for eligible, entries in _bounded_ordered_map(executor, function, offsets, threads):
            eligible_main += eligible
            for route, quota in DEFAULT_ROUTE_QUOTAS.items():
                heaps[route] = _stable_bounded(heaps[route], entries[route], quota)
            peak_rss = max(peak_rss, _current_rss_mb())
    overflow = np.asarray(loaded.overflow)
    selected_overflow = overflow[_eligible_mask(overflow, query, symbol_id)]
    if len(selected_overflow):
        zeros = np.zeros(len(selected_overflow), dtype=np.float64)
        incoming = _entries(selected_overflow, zeros, zeros, overflow=True)
        for route, quota in DEFAULT_ROUTE_QUOTAS.items():
            heaps[route] = _stable_bounded(heaps[route], incoming, quota)
    candidates, route_counts, candidate_digest = _finalize(heaps, loaded.symbols)
    deterministic = {
        "schema_version": "m04r-global-bound-proposal-v1",
        "contract_digest": packed_bound_search_contract()["digest"],
        "generation_id": loaded.generation_id, "query_episode_id": query.episode_id,
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
        "seconds": perf_counter() - started,
        "peak_rss_mb": max(
            peak_rss, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1_024,
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
    parser.add_argument("--threads", type=int, default=4, choices=(2, 4))
    args = parser.parse_args()
    numba.set_num_threads(1)
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
    advised_cold = _advise_cold(pack_path)
    scans = []
    with ThreadPoolExecutor(max_workers=args.threads, thread_name_prefix="legacy-bound") as executor:
        for label, order, block_rows in (
            ("cold", "forward", 4_096),
            ("warm_first", "reverse", 4_097),
            ("warm_second", "forward", 4_093),
        ):
            scan = _scan(
                executor, loaded, query, block_rows=block_rows,
                block_order=order, threads=args.threads,
            )
            scans.append({"label": label, "block_rows": block_rows, **scan})
            print(
                f"[{label}] seconds={scan['seconds']:.2f} rss={scan['peak_rss_mb']:.2f}",
                flush=True,
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
        "cold_advice_applied": advised_cold, "three_scan_invariance": repeated,
        "legacy_candidate_digest_exact": scans[0]["candidate_digest"] == EXPECTED_CANDIDATE_DIGEST,
        "legacy_result_digest_exact": scans[0]["result_digest"] == EXPECTED_RESULT_DIGEST,
        "cold_at_most_120_seconds": scans[0]["seconds"] <= 120.0,
        "warm_second_at_most_60_seconds": scans[2]["seconds"] <= 60.0,
        "peak_rss_at_most_1024_mib": max(row["peak_rss_mb"] for row in scans) <= 1_024.0,
    }
    deterministic = {
        "schema_version": SCHEMA, "query_episode_id": query.episode_id,
        "generation_id": generation_id, "threads": args.threads,
        "numba_threads": numba.get_num_threads(),
        "block_rows": [4_096, 4_097, 4_093],
        "maximum_in_flight_blocks": args.threads, "advised_cold": advised_cold,
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
        "<title>Legacy threaded scan POC</title></head><body>"
        f"<h1>{status}</h1><p>Bounded development-only CAKE thread pipeline; "
        f"ordered reduction must preserve legacy output exactly.</p><pre>{escape(json.dumps(payload, indent=2, sort_keys=True))}"
        "</pre></body></html>"
    )
    print(json.dumps({"passed": payload["passed"], "gates": gates,
                      "result_digest": payload["result_digest"]}, indent=2))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
