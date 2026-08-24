"""POC for a full float16, error-corrected distance-v1 lower-bound row."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import resource
from time import perf_counter

import numpy as np
import pandas as pd

from market_analogues.adapters import source_from_spec
from market_analogues.config import load_config
from market_analogues.distance import DistanceConfig, GROUPS
from market_analogues.episodes import build_episode
from market_analogues.exact_batch import batch_representation_lower_bounds, sliding_exact_representations
from market_analogues.representation import Representation, represent
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import InstrumentKey


def _safe_rms(query: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stored = candidates.astype(np.float16).astype(float)
    delta = candidates - stored
    error_rms = np.sqrt(np.mean(delta * delta, axis=1))
    error_max = np.max(np.abs(delta), axis=1)
    observed = np.sqrt(np.mean((stored - query[None, :]) ** 2, axis=1))
    return np.maximum(observed - error_rms, 0.0), error_rms, error_max


def _group_bound(
    query: Representation,
    candidates: tuple[Representation, ...],
    group: str,
    denominator_mode: str,
) -> np.ndarray:
    distances: list[np.ndarray] = []
    included: list[np.ndarray] = []
    for name in GROUPS[group]:
        query_values = query.samples_48.get(name)
        candidate_values = [candidate.samples_48.get(name) for candidate in candidates]
        present = np.asarray([value is not None for value in candidate_values])
        if query_values is None:
            distances.append(np.where(present, 2.0, 0.0))
            included.append(present)
            continue
        matrix = np.stack([value if value is not None else np.zeros(48) for value in candidate_values])
        numerator, error_rms, error_max = _safe_rms(query_values, matrix)
        stored = matrix.astype(np.float16).astype(float)
        joined = np.concatenate((stored, np.broadcast_to(query_values, stored.shape)), axis=1)
        approximate_std = np.std(joined, axis=1)
        # Quantiles are L-infinity Lipschitz and standard deviation is an L2
        # projection norm.  Query values are native, so only half the joined
        # vector contributes candidate quantization error.
        if denominator_mode == "quantile":
            approximate_iqr = np.percentile(joined, 75, axis=1) - np.percentile(joined, 25, axis=1)
            denominator_upper = np.maximum.reduce((
                approximate_iqr + 2.0 * error_max,
                approximate_std + error_rms / np.sqrt(2.0),
                np.full(len(matrix), 1e-6),
            ))
        elif denominator_mode == "statistical":
            range_upper = (
                np.maximum(np.max(stored, axis=1) + error_max, float(np.max(query_values)))
                - np.minimum(np.min(stored, axis=1) - error_max, float(np.min(query_values)))
            )
            denominator_upper = np.maximum(
                np.minimum(
                    range_upper,
                    4.0 * (approximate_std + error_rms / np.sqrt(2.0)),
                ),
                1e-6,
            )
        else:
            raise ValueError(f"unknown denominator mode {denominator_mode}")
        values = numerator / denominator_upper
        values[~present] = 2.0
        distances.append(values)
        included.append(np.ones(len(candidates), dtype=bool))
    count = np.sum(included, axis=0)
    result = np.divide(
        np.sum(distances, axis=0), count,
        out=np.zeros(len(candidates), dtype=float), where=count > 0,
    )
    return .55 * result if group == "price" else result


def _full_bound(
    query: Representation,
    candidates: tuple[Representation, ...],
    denominator_mode: str,
) -> np.ndarray:
    weights = DistanceConfig().weights
    stage_matrix = np.stack([candidate.stage for candidate in candidates])
    structural_matrix = np.stack([candidate.structural for candidate in candidates])
    coarse_matrix = np.stack([candidate.coarse for candidate in candidates])
    stage, _, _ = _safe_rms(query.stage, stage_matrix)
    structural, _, _ = _safe_rms(query.structural, structural_matrix)
    coarse_numerator, coarse_error, _ = _safe_rms(query.coarse, coarse_matrix)
    stored_coarse = coarse_matrix.astype(np.float16).astype(float)
    joined_coarse = np.concatenate((stored_coarse, np.broadcast_to(query.coarse, stored_coarse.shape)), axis=1)
    coarse_denominator_upper = np.maximum(
        np.std(joined_coarse, axis=1) + coarse_error / np.sqrt(2.0), 1e-6,
    )
    total = (
        weights["stage"] * stage
        + weights["structural"] * structural
        + weights["coarse"] * coarse_numerator / coarse_denominator_upper
    )
    for group in GROUPS:
        total += weights[group] * _group_bound(
            query, candidates, group, denominator_mode,
        )
    return total


def _randomized_channel_check(
    pairs: int,
    seed: int,
    denominator_mode: str,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    violations = 0
    maximum_excess = 0.0
    chunk = 2_000
    for first in range(0, pairs, chunk):
        count = min(chunk, pairs - first)
        query = rng.normal(size=(count, 48))
        candidate = query + rng.normal(scale=rng.lognormal(-2, 1, count)[:, None], size=(count, 48))
        query[:3] = np.asarray((np.zeros(48), np.full(48, 1e-7), np.full(48, -1e3)))
        candidate[:3] = np.asarray((np.zeros(48), np.full(48, -1e-7), np.full(48, 1e3)))
        stored = candidate.astype(np.float16).astype(float)
        delta = candidate - stored
        error_rms = np.sqrt(np.mean(delta * delta, axis=1))
        error_max = np.max(np.abs(delta), axis=1)
        numerator = np.maximum(np.sqrt(np.mean((stored - query) ** 2, axis=1)) - error_rms, 0.0)
        approximate_joined = np.concatenate((stored, query), axis=1)
        approximate_std = np.std(approximate_joined, axis=1)
        if denominator_mode == "quantile":
            denominator_upper = np.maximum.reduce((
                np.percentile(approximate_joined, 75, axis=1)
                - np.percentile(approximate_joined, 25, axis=1)
                + 2.0 * error_max,
                approximate_std + error_rms / np.sqrt(2.0),
                np.full(count, 1e-6),
            ))
        elif denominator_mode == "statistical":
            range_upper = (
                np.maximum(np.max(stored, axis=1) + error_max, np.max(query, axis=1))
                - np.minimum(np.min(stored, axis=1) - error_max, np.min(query, axis=1))
            )
            denominator_upper = np.maximum(
                np.minimum(
                    range_upper,
                    4.0 * (approximate_std + error_rms / np.sqrt(2.0)),
                ),
                1e-6,
            )
        else:
            raise ValueError(f"unknown denominator mode {denominator_mode}")
        lower = numerator / denominator_upper
        exact_joined = np.concatenate((candidate, query), axis=1)
        exact_scale = np.percentile(exact_joined, 75, axis=1) - np.percentile(exact_joined, 25, axis=1)
        exact_scale = np.where(exact_scale < 1e-8, np.std(exact_joined, axis=1), exact_scale)
        exact_scale = np.maximum(exact_scale, 1e-6)
        exact = np.sqrt(np.mean((candidate - query) ** 2, axis=1)) / exact_scale
        excess = lower - exact
        violations += int(np.sum(excess > 1e-12))
        maximum_excess = max(maximum_excess, float(np.max(excess)))
    return {
        "pairs": pairs,
        "denominator_mode": denominator_mode,
        "violations": violations,
        "maximum_excess": maximum_excess,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--symbols", type=int, default=48)
    parser.add_argument("--random-pairs", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    source = source_from_spec(config.datasets["nasdaq"])
    authority = json.loads((
        config.artifact_dir / "gate12" / "authorities" / "nasdaq" / "cases" / f"{args.episode_id}.json"
    ).read_text())
    meta = authority["query"]
    query = build_episode(
        source, InstrumentKey("nasdaq", str(meta["symbol"])), str(meta["cutoff"]),
        int(meta["lookback"]), str(meta["representation_version"]),
    )
    query_representation = represent(query)
    latest = latest_eligible_cutoff(query, 60)
    benchmark = source.load_benchmark()
    quality = pd.read_parquet(config.artifact_dir / "quality" / "nasdaq.parquet")
    tiers = dict(zip(quality.symbol.astype(str), quality.tier.astype(str)))
    keys = [key for key in source.instruments() if tiers.get(key.source_symbol, "A") in {"A", "B"}]
    keys.sort(key=lambda key: sha256(f"m04r-full-bound:{key.source_symbol}".encode()).hexdigest())
    selected = keys[:args.symbols]
    selected_symbols = {key.source_symbol for key in selected}
    wanted = {str(row["symbol"]) for row in authority["matches"]}
    selected.extend(key for key in keys if key.source_symbol in wanted - selected_symbols)
    threshold = float(authority["certificate"]["stop_threshold"])
    real_rows = native_pruned = 0
    generation_seconds = native_bound_seconds = 0.0
    mode_metrics = {
        mode: {"pruned": 0, "violations": 0, "maximum_excess": 0.0, "seconds": 0.0}
        for mode in ("quantile", "statistical")
    }
    started = perf_counter()
    for key in selected:
        frame = source.load(key)
        frame = frame[frame.timestamp <= latest].reset_index(drop=True)
        timer = perf_counter()
        batch = sliding_exact_representations(frame, benchmark, lookback=252, stride=5, batch_size=512)
        generation_seconds += perf_counter() - timer
        if not batch.representations:
            continue
        timer = perf_counter()
        native = batch_representation_lower_bounds(query_representation, batch.representations).totals
        native_bound_seconds += perf_counter() - timer
        for mode, metrics in mode_metrics.items():
            timer = perf_counter()
            lower = _full_bound(query_representation, batch.representations, mode)
            metrics["seconds"] += perf_counter() - timer
            excess = lower - native
            metrics["violations"] += int(np.sum(excess > 1e-12))
            metrics["maximum_excess"] = max(
                metrics["maximum_excess"], float(np.max(excess)),
            )
            metrics["pruned"] += int(np.sum(lower >= threshold))
        native_pruned += int(np.sum(native >= threshold))
        real_rows += len(lower)
    randomized = [
        _randomized_channel_check(args.random_pairs, args.seed, mode)
        for mode in mode_metrics
    ]
    for metrics in mode_metrics.values():
        metrics["pruning_fraction"] = metrics["pruned"] / real_rows
        metrics["pruning_retention"] = metrics["pruned"] / native_pruned
        metrics["rows_per_second"] = real_rows / metrics["seconds"]
    value_count = 128 + 19 * 48 + 48 + 9
    # Per-channel RMS/max error radii plus coarse/stage/structural radii are
    # conservatively budgeted as 44 float32 values; masks/IDs are separate.
    bytes_per_row = value_count * 2 + 44 * 4
    payload = {
        "schema_version": "m04r-full-quantized-bound-poc-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "query_episode_id": args.episode_id,
        "symbols": len(selected),
        "real_rows": real_rows,
        "threshold": threshold,
        "native_pruned": native_pruned,
        "native_pruning_fraction": native_pruned / real_rows,
        "modes": mode_metrics,
        "randomized_channel_checks": randomized,
        "layout": {
            "float16_values": value_count,
            "float32_error_values_budget": 44,
            "bytes_per_row_before_ids_masks_metadata": bytes_per_row,
            "projected_3_82m_gib_before_ids_masks_metadata": bytes_per_row * 3_820_000 / 1024 ** 3,
            "projected_5m_gib_before_ids_masks_metadata": bytes_per_row * 5_000_000 / 1024 ** 3,
        },
        "elapsed_seconds": perf_counter() - started,
        "generation_seconds": generation_seconds,
        "native_bound_seconds": native_bound_seconds,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "real_forward_outcomes_accessed": False,
        "interpretation": "POC evidence only; quantile/error derivation requires independent proof review",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2))
    failed = any(metrics["violations"] for metrics in mode_metrics.values()) or any(
        result["violations"] for result in randomized
    )
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
