"""POC for a two-row, threshold-aware implementation of distance-v1 DTW.

The kernel preserves the production tie order and tracks the path length chosen
by the minimum accumulated-cost path.  Early abandonment is safe because all
local costs are nonnegative and no legal path can exceed n + m - 1 positions:
if the cheapest accumulated prefix divided by that maximum final path length is
already above tau, the completed normalized cost cannot beat tau.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
from time import perf_counter

import numpy as np
from numba import njit

from market_analogues.distance import bounded_dtw


@njit(cache=True)
def dtw_two_row(x: np.ndarray, y: np.ndarray, band_fraction: float, tau: float) -> tuple[float, int, bool]:
    n, m = len(x), len(y)
    band = max(abs(n - m), int(max(n, m) * band_fraction), 1)
    previous = np.full(m + 1, np.inf)
    previous_length = np.zeros(m + 1, dtype=np.int64)
    previous[0] = 0.0
    maximum_path_length = n + m - 1
    for i in range(1, n + 1):
        current = np.full(m + 1, np.inf)
        current_length = np.zeros(m + 1, dtype=np.int64)
        first = max(1, i - band)
        last = min(m, i + band)
        row_minimum = np.inf
        for j in range(first, last + 1):
            # Match Python's stable min order: vertical, horizontal, diagonal.
            value = previous[j]
            length = previous_length[j]
            if current[j - 1] < value:
                value = current[j - 1]
                length = current_length[j - 1]
            if previous[j - 1] < value:
                value = previous[j - 1]
                length = previous_length[j - 1]
            squared = 0.0
            for channel in range(x.shape[1]):
                delta = x[i - 1, channel] - y[j - 1, channel]
                squared += delta * delta
            local = np.sqrt(squared / x.shape[1])
            current[j] = value + local
            current_length[j] = length + 1
            if current[j] < row_minimum:
                row_minimum = current[j]
        if np.isfinite(tau) and row_minimum / maximum_path_length > tau:
            return np.inf, 0, True
        previous = current
        previous_length = current_length
    if not np.isfinite(previous[m]):
        return np.inf, 0, False
    return previous[m] / max(previous_length[m], 1), int(previous_length[m]), False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-pairs", type=int, default=2_000)
    parser.add_argument("--abandon-pairs", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    # Compile outside measurements.
    warm = np.zeros((64, 4), dtype=np.float64)
    dtw_two_row(warm, warm, .12, np.inf)

    maximum_delta = 0.0
    path_length_mismatches = 0
    reference_inputs: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(args.reference_pairs):
        left = np.cumsum(rng.normal(size=(64, 4)), axis=0)
        right = left + np.cumsum(rng.normal(scale=.35, size=(64, 4)), axis=0)
        reference_inputs.append((left, right))
    reference_started = perf_counter()
    for left, right in reference_inputs:
        expected, path = bounded_dtw(left, right, .12)
        actual, path_length, abandoned = dtw_two_row(left, right, .12, np.inf)
        if abandoned:
            raise AssertionError("infinite-threshold call abandoned")
        maximum_delta = max(maximum_delta, abs(actual - expected))
        path_length_mismatches += int(path_length != len(path))
    reference_seconds = perf_counter() - reference_started

    native_started = perf_counter()
    for left, right in reference_inputs:
        dtw_two_row(left, right, .12, np.inf)
    native_seconds = perf_counter() - native_started

    abandon_started = perf_counter()
    abandon_count = unsafe_abandons = completed_mismatches = 0
    maximum_completed_delta = 0.0
    for _ in range(args.abandon_pairs):
        left = np.cumsum(rng.normal(size=(64, 4)), axis=0)
        right = left + np.cumsum(rng.normal(scale=rng.uniform(.05, 1.5), size=(64, 4)), axis=0)
        exact, _, _ = dtw_two_row(left, right, .12, np.inf)
        tau = exact * rng.uniform(.5, 1.5)
        value, _, abandoned = dtw_two_row(left, right, .12, tau)
        if abandoned:
            abandon_count += 1
            unsafe_abandons += int(exact <= tau)
        else:
            completed_mismatches += int(abs(value - exact) > 1e-12)
            maximum_completed_delta = max(maximum_completed_delta, abs(value - exact))
    abandon_seconds = perf_counter() - abandon_started

    payload = {
        "schema_version": "m04r-dtw-kernel-poc-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_pairs": args.reference_pairs,
        "maximum_reference_delta": maximum_delta,
        "path_length_mismatches": path_length_mismatches,
        "python_reference_seconds": reference_seconds,
        "numba_kernel_seconds_same_pairs": native_seconds,
        "measured_speedup": reference_seconds / native_seconds,
        "abandon_pairs": args.abandon_pairs,
        "abandoned": abandon_count,
        "unsafe_abandons": unsafe_abandons,
        "completed_mismatches": completed_mismatches,
        "maximum_completed_delta": maximum_completed_delta,
        "abandon_test_seconds": abandon_seconds,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "limitations": [
            "random-walk/adversarial numerical POC, not yet integrated with pair scaling",
            "final-result alignment paths still require the full parent matrix",
            "tests support but do not replace the written early-abandon proof",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2))
    passed = (
        maximum_delta <= 1e-12
        and path_length_mismatches == 0
        and unsafe_abandons == 0
        and completed_mismatches == 0
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
