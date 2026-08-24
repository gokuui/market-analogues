"""Million-pair falsification gate for the full float16 distance-v1 bound."""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from market_analogues.quantized_bound import quantized_bound_contract
from market_analogues.types import stable_hash


def _outward32(values: np.ndarray) -> np.ndarray:
    native = np.asarray(values, dtype=np.float64)
    rounded = native.astype(np.float32)
    return np.where(
        rounded.astype(np.float64) < native,
        np.nextafter(rounded, np.float32(np.inf), dtype=np.float32),
        rounded,
    ).astype(np.float64)


def _evaluate(query: np.ndarray, candidate: np.ndarray, tolerance: float) -> dict[str, float | int]:
    stored = candidate.astype(np.float16).astype(np.float64)
    error = candidate - stored
    error_rms = _outward32(np.sqrt(np.mean(error * error, axis=1)))
    error_max = _outward32(np.max(np.abs(error), axis=1))
    observed = np.sqrt(np.mean((stored - query) ** 2, axis=1))
    numerator = np.maximum(observed - error_rms, 0.0)
    approximate = np.c_[stored, query]
    denominator = np.maximum.reduce((
        np.percentile(approximate, 75, axis=1)
        - np.percentile(approximate, 25, axis=1) + 2.0 * error_max,
        np.std(approximate, axis=1) + error_rms / np.sqrt(2.0),
        np.full(len(query), 1e-6),
    ))
    lower = numerator / denominator
    exact_joined = np.c_[candidate, query]
    exact_scale = (
        np.percentile(exact_joined, 75, axis=1)
        - np.percentile(exact_joined, 25, axis=1)
    )
    exact_scale = np.where(
        exact_scale < 1e-8, np.std(exact_joined, axis=1), exact_scale,
    )
    exact_scale = np.maximum(exact_scale, 1e-6)
    exact = np.sqrt(np.mean((candidate - query) ** 2, axis=1)) / exact_scale
    unscaled_lower = numerator
    unscaled_exact = np.sqrt(np.mean((candidate - query) ** 2, axis=1))
    excess = lower - exact
    unscaled_excess = unscaled_lower - unscaled_exact
    return {
        "violations": int(np.sum(excess > tolerance)),
        "unscaled_violations": int(np.sum(unscaled_excess > tolerance)),
        "maximum_excess": max(float(np.max(excess)), 0.0),
        "maximum_unscaled_excess": max(float(np.max(unscaled_excess)), 0.0),
        "maximum_ratio": float(np.max(np.divide(
            lower, exact, out=np.zeros_like(lower), where=exact > 1e-15,
        ))),
        "positive": int(np.sum(lower > 0)),
    }


def _batch(rng: np.random.Generator, count: int, mode: int) -> tuple[np.ndarray, np.ndarray]:
    query = rng.normal(size=(count, 48))
    candidate = query + rng.normal(
        scale=rng.lognormal(-2, 1, count)[:, None], size=(count, 48),
    )
    if mode == 1:
        query[:] = rng.normal(size=(count, 1))
        candidate[:] = rng.normal(size=(count, 1))
    elif mode == 2:
        query[:] = rng.normal(scale=1e-9, size=(count, 48))
        candidate[:] = query + rng.normal(scale=1e-9, size=(count, 48))
    elif mode == 3:
        query *= .01
        candidate *= .01
        candidate[:, 24] += rng.choice((-60000.0, 60000.0), size=count)
    elif mode == 4:
        centers = rng.choice((0.0, 1.0, -1.0, 1024.0, -1024.0), size=(count, 1))
        ulps = np.spacing(centers.astype(np.float16)).astype(float)
        query = centers + rng.uniform(-1, 1, size=(count, 48)) * ulps
        candidate = centers + rng.uniform(-1, 1, size=(count, 48)) * ulps
    return query, candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pairs < 1_000_000:
        raise ValueError("formal gate requires at least 1,000,000 pairs")
    rng = np.random.default_rng(args.seed)
    totals = {
        "violations": 0, "unscaled_violations": 0, "maximum_excess": 0.0,
        "maximum_unscaled_excess": 0.0, "maximum_ratio": 0.0, "positive": 0,
    }
    started = perf_counter()
    tested = 0
    batch_index = 0
    while tested < args.pairs:
        count = min(args.batch_size, args.pairs - tested)
        query, candidate = _batch(rng, count, batch_index % 5)
        result = _evaluate(query, candidate, args.tolerance)
        for name in ("violations", "unscaled_violations", "positive"):
            totals[name] += int(result[name])
        for name in ("maximum_excess", "maximum_unscaled_excess", "maximum_ratio"):
            totals[name] = max(float(totals[name]), float(result[name]))
        tested += count
        batch_index += 1

    boundary_values = np.asarray([
        -1.0009765625, -5.960464477539063e-08, 0.0,
        5.960464477539063e-08, 1.0009765625,
    ])
    sequences = np.asarray(list(product(boundary_values, repeat=3)))
    expanded = np.tile(sequences, (1, 16))
    left = np.repeat(expanded, len(expanded), axis=0)
    right = np.tile(expanded, (len(expanded), 1))
    boundary = _evaluate(left, right, args.tolerance)
    elapsed = perf_counter() - started
    deterministic = {
        "schema_version": "m04r-quantized-bound-million-gate-v1",
        "contract_digest": quantized_bound_contract()["digest"],
        "seed": args.seed,
        "pairs": tested,
        "tolerance": args.tolerance,
        **totals,
        "boundary_pairs": len(left),
        "boundary_violations": boundary["violations"],
        "boundary_unscaled_violations": boundary["unscaled_violations"],
        "outcomes_or_labels_used": False,
    }
    payload = {
        **deterministic,
        "elapsed_seconds": elapsed,
        "pairs_per_second": tested / elapsed,
        "result_digest": stable_hash(deterministic),
        "proof_boundary": "Falsification evidence supports but does not replace the contract proof.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    passed = not (
        totals["violations"] or totals["unscaled_violations"]
        or boundary["violations"] or boundary["unscaled_violations"]
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
