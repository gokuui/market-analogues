"""In-memory kernel POC for scanning a 90-value compact bound layout.

This measures arithmetic/layout throughput only.  It intentionally does not
claim cold-disk or 11,584-shard performance; those require a packed mmap POC
after the storage-capacity precondition is met.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
from time import perf_counter

import numpy as np


def scan(block: np.ndarray, query: np.ndarray, weights: tuple[float, float, float]) -> tuple[int, float]:
    values = block.astype(np.float32)
    stage = np.sqrt(np.mean((values[:, :48] - query[:48]) ** 2, axis=1))
    structural = np.sqrt(np.mean((values[:, 48:57] - query[48:57]) ** 2, axis=1))
    coarse_paa = np.sqrt(np.mean((values[:, 57:89] - query[57:89]) ** 2, axis=1))
    score = weights[0] * stage + weights[1] * structural + weights[2] * coarse_paa
    # Force both thresholding and stable small-k selection work.
    survivors = int(np.sum(score < .75))
    kth = float(np.partition(score, min(19, len(score) - 1))[min(19, len(score) - 1)])
    return survivors, kth


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-rows", type=int, default=250_000)
    parser.add_argument("--total-rows", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    block = rng.normal(size=(args.block_rows, 90)).astype(np.float16)
    query = rng.normal(size=90).astype(np.float32)
    repeats = (args.total_rows + args.block_rows - 1) // args.block_rows
    scan(block[: min(10_000, len(block))], query, (.30, .09, .08))
    started = perf_counter()
    survivors = 0
    kth_checksum = 0.0
    processed = 0
    for repeat in range(repeats):
        count = min(args.block_rows, args.total_rows - processed)
        found, kth = scan(block[:count], query, (.30, .09, .08))
        survivors += found
        kth_checksum += kth
        processed += count
    seconds = perf_counter() - started
    payload = {
        "schema_version": "m04r-layout-scan-poc-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "storage_dtype": "float16",
        "dimensions": 90,
        "bytes_per_row": 180,
        "block_rows": args.block_rows,
        "logical_rows_scanned": processed,
        "resident_unique_rows": args.block_rows,
        "resident_block_bytes": int(block.nbytes),
        "seconds": seconds,
        "rows_per_second": processed / seconds,
        "logical_gib_per_second": processed * 180 / seconds / 1024 ** 3,
        "survivor_checksum": survivors,
        "kth_checksum": kth_checksum,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "limitations": [
            "warm in-memory repeated block; not a cold mmap benchmark",
            "synthetic normally distributed fields; not real pruning selectivity",
            "NumPy float16 widening kernel; native SIMD backends remain untested",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
