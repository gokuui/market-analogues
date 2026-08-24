"""Million-pair safety gate for distance-v1's native symmetric LB_Keogh.

The contract contains the algebraic proof.  This executable gate tries to
falsify it on pair-scaled random, structured and adversarial inputs, and first
checks its compiled kernels against the production Python implementation.
"""

from __future__ import annotations

import argparse
from itertools import product
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from numba import njit

from market_analogues.distance import _lb_keogh_one_way, bounded_dtw
from market_analogues.distance_v1_reference import distance_v1_contract
from market_analogues.types import stable_hash


@njit(cache=True)
def _linear_percentile(sorted_values: np.ndarray, quantile: float) -> float:
    location = (len(sorted_values) - 1) * quantile
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    fraction = location - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


@njit(cache=True)
def _pair_scaled(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaled_left = left.copy()
    scaled_right = right.copy()
    joined = np.empty(len(left) + len(right), dtype=np.float64)
    for channel in range(left.shape[1]):
        joined[:len(left)] = left[:, channel]
        joined[len(left):] = right[:, channel]
        ordered = np.sort(joined)
        scale = _linear_percentile(ordered, .75) - _linear_percentile(ordered, .25)
        if scale < 1e-8:
            mean = 0.0
            for value in joined:
                mean += value
            mean /= len(joined)
            variance = 0.0
            for value in joined:
                variance += (value - mean) ** 2
            scale = math.sqrt(variance / len(joined))
        scale = max(scale, 1e-6)
        scaled_left[:, channel] /= scale
        scaled_right[:, channel] /= scale
    return scaled_left, scaled_right


@njit(cache=True)
def _exact_normalized_dtw(left: np.ndarray, right: np.ndarray, band: int) -> float:
    previous = np.full(len(right) + 1, np.inf)
    previous_length = np.zeros(len(right) + 1, dtype=np.int64)
    previous[0] = 0.0
    for i in range(1, len(left) + 1):
        current = np.full(len(right) + 1, np.inf)
        current_length = np.zeros(len(right) + 1, dtype=np.int64)
        first = max(1, i - band)
        last = min(len(right), i + band)
        for j in range(first, last + 1):
            # Production's stable predecessor order: vertical, horizontal,
            # diagonal.  A strict comparison retains the earlier choice.
            value = previous[j]
            length = previous_length[j]
            if current[j - 1] < value:
                value = current[j - 1]
                length = current_length[j - 1]
            if previous[j - 1] < value:
                value = previous[j - 1]
                length = previous_length[j - 1]
            squared = 0.0
            for channel in range(left.shape[1]):
                delta = left[i - 1, channel] - right[j - 1, channel]
                squared += delta * delta
            current[j] = value + math.sqrt(squared / left.shape[1])
            current_length[j] = length + 1
        previous = current
        previous_length = current_length
    return previous[len(right)] / max(previous_length[len(right)], 1)


@njit(cache=True)
def _one_way_bound(query: np.ndarray, candidate: np.ndarray, band: int) -> float:
    total = 0.0
    for index in range(len(candidate)):
        start = max(0, index - band)
        end = min(len(query), index + band + 1)
        squared = 0.0
        for channel in range(query.shape[1]):
            lower = query[start, channel]
            upper = lower
            for position in range(start + 1, end):
                lower = min(lower, query[position, channel])
                upper = max(upper, query[position, channel])
            value = candidate[index, channel]
            deviation = max(lower - value, 0.0) + min(upper - value, 0.0)
            squared += deviation * deviation
        total += math.sqrt(squared / query.shape[1])
    return total / max(len(query) + len(candidate) - 1, 1)


@njit(cache=True)
def _evaluate_batch(
    raw_left: np.ndarray,
    raw_right: np.ndarray,
    band_fraction: float,
    tolerance: float,
) -> tuple[int, float, float, float, float, int]:
    violations = 0
    maximum_excess = 0.0
    maximum_ratio = 0.0
    bound_sum = 0.0
    exact_sum = 0.0
    positive_bounds = 0
    for pair in range(len(raw_left)):
        left, right = _pair_scaled(raw_left[pair], raw_right[pair])
        band = max(abs(len(left) - len(right)), int(max(len(left), len(right)) * band_fraction), 1)
        exact = _exact_normalized_dtw(left, right, band)
        bound = max(_one_way_bound(left, right, band), _one_way_bound(right, left, band))
        excess = bound - exact
        if excess > tolerance:
            violations += 1
            maximum_excess = max(maximum_excess, excess)
        if exact > 1e-15:
            maximum_ratio = max(maximum_ratio, bound / exact)
        positive_bounds += int(bound > 0.0)
        bound_sum += bound
        exact_sum += exact
    return violations, maximum_excess, maximum_ratio, bound_sum, exact_sum, positive_bounds


def _make_batch(
    rng: np.random.Generator,
    count: int,
    left_length: int,
    right_length: int,
    channels: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    left = rng.normal(size=(count, left_length, channels))
    right = rng.normal(size=(count, right_length, channels))
    if mode == "random_walk":
        left = np.cumsum(left, axis=1)
        right = np.cumsum(right, axis=1)
    elif mode == "constant":
        left[:] = rng.normal(size=(count, 1, channels))
        right[:] = rng.normal(size=(count, 1, channels))
    elif mode == "spikes":
        left *= .01
        right *= .01
        left[:, left_length // 2, :] += rng.choice((-1e6, 1e6), size=(count, channels))
        right[:, right_length // 3, :] += rng.choice((-1e6, 1e6), size=(count, channels))
    elif mode == "affine":
        common = left[:, :min(left_length, right_length), :]
        scales = 10.0 ** rng.uniform(-6, 6, size=(count, 1, channels))
        offsets = rng.uniform(-1e9, 1e9, size=(count, 1, channels))
        right[:, :len(common[0]), :] = common * scales + offsets
    elif mode == "reversed":
        shared = left[:, :min(left_length, right_length), :]
        right[:, :len(shared[0]), :] = shared[:, ::-1, :]
    else:
        raise ValueError(f"unknown mode {mode}")
    return left.astype(np.float64), right.astype(np.float64)


def _python_pair_scaled(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaled_left = left.copy()
    scaled_right = right.copy()
    for channel in range(left.shape[1]):
        joined = np.r_[left[:, channel], right[:, channel]]
        scale = np.percentile(joined, 75) - np.percentile(joined, 25)
        if scale < 1e-8:
            scale = np.std(joined)
        scale = max(float(scale), 1e-6)
        scaled_left[:, channel] /= scale
        scaled_right[:, channel] /= scale
    return scaled_left, scaled_right


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=1_000_000)
    parser.add_argument("--reference-pairs", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    parser.add_argument("--reference-tolerance", type=float, default=1e-9)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pairs < 1_000_000:
        raise ValueError("formal gate requires at least 1,000,000 pairs")

    rng = np.random.default_rng(args.seed)
    configurations = (
        (4, 4, 1, "constant"),
        (7, 9, 2, "random_walk"),
        (12, 8, 4, "spikes"),
        (16, 16, 4, "affine"),
        (11, 15, 3, "reversed"),
        (64, 64, 4, "random_walk"),
    )
    # Compile outside all measurements.
    warm_left, warm_right = _make_batch(rng, 1, 4, 4, 1, "constant")
    _evaluate_batch(warm_left, warm_right, .12, args.tolerance)

    reference_maximum_exact_delta = 0.0
    reference_maximum_bound_delta = 0.0
    reference_maximum_exact_relative_delta = 0.0
    reference_maximum_bound_relative_delta = 0.0
    reference_violations = 0
    for index in range(args.reference_pairs):
        n, m, channels, mode = configurations[index % len(configurations)]
        raw_left, raw_right = _make_batch(rng, 1, n, m, channels, mode)
        left, right = _python_pair_scaled(raw_left[0], raw_right[0])
        band = max(abs(n - m), int(max(n, m) * .12), 1)
        expected_exact, _ = bounded_dtw(left, right, .12)
        expected_bound = max(
            _lb_keogh_one_way(left, right, band),
            _lb_keogh_one_way(right, left, band),
        )
        compiled = _evaluate_batch(raw_left, raw_right, .12, args.tolerance)
        compiled_left, compiled_right = _pair_scaled(raw_left[0], raw_right[0])
        compiled_exact = _exact_normalized_dtw(compiled_left, compiled_right, band)
        compiled_bound = max(
            _one_way_bound(compiled_left, compiled_right, band),
            _one_way_bound(compiled_right, compiled_left, band),
        )
        reference_maximum_exact_delta = max(
            reference_maximum_exact_delta, abs(expected_exact - compiled_exact),
        )
        reference_maximum_bound_delta = max(
            reference_maximum_bound_delta, abs(expected_bound - compiled_bound),
        )
        reference_maximum_exact_relative_delta = max(
            reference_maximum_exact_relative_delta,
            abs(expected_exact - compiled_exact) / max(abs(expected_exact), 1.0),
        )
        reference_maximum_bound_relative_delta = max(
            reference_maximum_bound_relative_delta,
            abs(expected_bound - compiled_bound) / max(abs(expected_bound), 1.0),
        )
        reference_violations += compiled[0]

    tiny_sequences = np.asarray(list(product((-1.0, 0.0, 1.0), repeat=3)))
    exhaustive_left = np.repeat(tiny_sequences, len(tiny_sequences), axis=0)[:, :, None]
    exhaustive_right = np.tile(tiny_sequences, (len(tiny_sequences), 1))[:, :, None]
    exhaustive = _evaluate_batch(
        exhaustive_left, exhaustive_right, .12, args.tolerance,
    )

    started = perf_counter()
    tested = violations = positive_bounds = 0
    maximum_excess = maximum_ratio = bound_sum = exact_sum = 0.0
    configuration_counts: dict[str, int] = {}
    batch_index = 0
    while tested < args.pairs:
        n, m, channels, mode = configurations[batch_index % len(configurations)]
        count = min(args.batch_size, args.pairs - tested)
        left, right = _make_batch(rng, count, n, m, channels, mode)
        result = _evaluate_batch(left, right, .12, args.tolerance)
        violations += result[0]
        maximum_excess = max(maximum_excess, result[1])
        maximum_ratio = max(maximum_ratio, result[2])
        bound_sum += result[3]
        exact_sum += result[4]
        positive_bounds += result[5]
        key = f"{n}x{m}x{channels}:{mode}"
        configuration_counts[key] = configuration_counts.get(key, 0) + count
        tested += count
        batch_index += 1
    elapsed = perf_counter() - started

    deterministic = {
        "schema_version": "m04r-distance-v1-bound-gate-v1",
        "distance_contract_digest": distance_v1_contract()["digest"],
        "seed": args.seed,
        "pairs_tested": tested,
        "reference_pairs": args.reference_pairs,
        "configurations": configuration_counts,
        "tolerance": args.tolerance,
        "production_reference_relative_tolerance": args.reference_tolerance,
        "violations": violations,
        "maximum_excess": maximum_excess,
        "maximum_bound_to_exact_ratio": maximum_ratio,
        "positive_bound_fraction": positive_bounds / tested,
        "mean_bound": bound_sum / tested,
        "mean_exact": exact_sum / tested,
        "reference_maximum_exact_delta": reference_maximum_exact_delta,
        "reference_maximum_bound_delta": reference_maximum_bound_delta,
        "reference_maximum_exact_relative_delta": reference_maximum_exact_relative_delta,
        "reference_maximum_bound_relative_delta": reference_maximum_bound_relative_delta,
        "reference_violations": reference_violations,
        "exhaustive_tiny_alphabet_pairs": len(exhaustive_left),
        "exhaustive_tiny_alphabet_violations": exhaustive[0],
        "outcomes_or_labels_used": False,
    }
    payload = {
        **deterministic,
        "elapsed_seconds": elapsed,
        "pairs_per_second": tested / elapsed,
        "result_digest": stable_hash(deterministic),
        "proof_boundary": (
            "Tests try to falsify the bound; the algebraic proof is frozen in the "
            "distance-v1 contract. This gate does not certify the deferred quantized bound."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    passed = (
        violations == 0
        and reference_violations == 0
        and exhaustive[0] == 0
        and reference_maximum_exact_relative_delta <= args.reference_tolerance
        and reference_maximum_bound_relative_delta <= args.reference_tolerance
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
