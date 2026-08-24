"""POC for compact bounds on distance-v1's pair-scaled channel groups."""

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

from compact_bound_poc import _compact_bounds
from market_analogues.adapters import source_from_spec
from market_analogues.config import load_config
from market_analogues.distance import DistanceConfig, GROUPS
from market_analogues.episodes import build_episode
from market_analogues.exact_batch import batch_representation_lower_bounds, sliding_exact_representations
from market_analogues.representation import Representation, represent
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import InstrumentKey


def _outward(values: np.ndarray, *, upper: bool) -> np.ndarray:
    rounded = values.astype(np.float16)
    direction = np.float16(np.inf if upper else -np.inf)
    needs_step = rounded.astype(float) < values if upper else rounded.astype(float) > values
    return np.where(needs_step, np.nextafter(rounded, direction).astype(float), rounded.astype(float))


def _group_bound(
    query: Representation,
    candidates: tuple[Representation, ...],
    group: str,
    paa_count: int,
) -> np.ndarray:
    distances: list[np.ndarray] = []
    included: list[np.ndarray] = []
    block = 48 // paa_count
    for name in GROUPS[group]:
        query_values = query.samples_48.get(name)
        candidate_values = [candidate.samples_48.get(name) for candidate in candidates]
        present = np.asarray([value is not None for value in candidate_values])
        if query_values is None:
            distances.append(np.where(present, 2.0, 0.0))
            included.append(present)
            continue
        matrix = np.stack([value if value is not None else np.zeros(48) for value in candidate_values])
        candidate_paa = matrix.reshape(len(matrix), paa_count, block).mean(axis=2)
        query_paa = query_values.reshape(paa_count, block).mean(axis=1)
        stored = candidate_paa.astype(np.float16).astype(float)
        error = np.sqrt(np.mean((candidate_paa - stored) ** 2, axis=1))
        numerator = np.maximum(
            np.sqrt(np.mean((stored - query_paa) ** 2, axis=1)) - error,
            0.0,
        )
        candidate_min = _outward(np.min(matrix, axis=1), upper=False)
        candidate_max = _outward(np.max(matrix, axis=1), upper=True)
        range_upper = np.maximum(
            np.maximum(candidate_max, float(np.max(query_values)))
            - np.minimum(candidate_min, float(np.min(query_values))),
            1e-6,
        )
        # For any distribution, Q3-Q1 <= 4 sigma by Chebyshev's inequality.
        # Candidate mean and second moment are sufficient to recover the joined
        # variance.  The production version would store outward-rounded float32
        # moment intervals; this POC uses native moments to test tightness first.
        joined_mean = (np.mean(matrix, axis=1) + float(np.mean(query_values))) / 2.0
        joined_second = (
            np.mean(matrix * matrix, axis=1) + float(np.mean(query_values * query_values))
        ) / 2.0
        joined_std = np.sqrt(np.maximum(joined_second - joined_mean * joined_mean, 0.0))
        moment_upper = np.maximum(4.0 * joined_std, 1e-6)
        denominator_upper = np.minimum(range_upper, moment_upper)
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


def _randomized_channel_check(pairs: int, seed: int, paa_count: int) -> dict[str, object]:
    rng = np.random.default_rng(seed + paa_count)
    violations = 0
    maximum_excess = 0.0
    chunk = 2_000
    for first in range(0, pairs, chunk):
        count = min(chunk, pairs - first)
        query = rng.normal(size=(count, 48))
        candidate = query + rng.normal(scale=rng.lognormal(-2, 1, count)[:, None], size=(count, 48))
        query[:3] = np.asarray((np.zeros(48), np.full(48, 1e-7), np.full(48, -1e3)))
        candidate[:3] = np.asarray((np.zeros(48), np.full(48, -1e-7), np.full(48, 1e3)))
        block = 48 // paa_count
        q_paa = query.reshape(count, paa_count, block).mean(axis=2)
        c_paa = candidate.reshape(count, paa_count, block).mean(axis=2)
        stored = c_paa.astype(np.float16).astype(float)
        error = np.sqrt(np.mean((c_paa - stored) ** 2, axis=1))
        numerator = np.maximum(np.sqrt(np.mean((stored - q_paa) ** 2, axis=1)) - error, 0.0)
        lower_min = _outward(np.min(candidate, axis=1), upper=False)
        upper_max = _outward(np.max(candidate, axis=1), upper=True)
        range_upper = np.maximum(
            np.maximum(upper_max, np.max(query, axis=1))
            - np.minimum(lower_min, np.min(query, axis=1)), 1e-6,
        )
        joined_mean = (np.mean(candidate, axis=1) + np.mean(query, axis=1)) / 2.0
        joined_second = (
            np.mean(candidate * candidate, axis=1) + np.mean(query * query, axis=1)
        ) / 2.0
        joined_std = np.sqrt(np.maximum(joined_second - joined_mean * joined_mean, 0.0))
        denominator_upper = np.minimum(range_upper, np.maximum(4.0 * joined_std, 1e-6))
        lower = numerator / denominator_upper
        joined = np.concatenate((query, candidate), axis=1)
        scale = np.percentile(joined, 75, axis=1) - np.percentile(joined, 25, axis=1)
        fallback = np.std(joined, axis=1)
        scale = np.maximum(np.where(scale < 1e-8, fallback, scale), 1e-6)
        exact = np.sqrt(np.mean((candidate - query) ** 2, axis=1)) / scale
        excess = lower - exact
        violations += int(np.sum(excess > 1e-12))
        maximum_excess = max(maximum_excess, float(np.max(excess)))
    return {"pairs": pairs, "paa_count": paa_count, "violations": violations, "maximum_excess": maximum_excess}


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
    keys.sort(key=lambda key: sha256(f"m04r-pair-bound:{key.source_symbol}".encode()).hexdigest())
    selected = keys[:args.symbols]
    selected_symbols = {key.source_symbol for key in selected}
    wanted = {str(row["symbol"]) for row in authority["matches"]}
    selected.extend(key for key in keys if key.source_symbol in wanted - selected_symbols)
    threshold = float(authority["certificate"]["stop_threshold"])
    weights = DistanceConfig().weights
    layout_specs = {
        "all_paa2": {group: 2 for group in GROUPS},
        "all_paa4": {group: 4 for group in GROUPS},
        "price_market_paa4_other_paa2": {
            "price": 4, "market_context": 4, "candle_volatility": 2, "volume_shock": 2,
        },
        "price_paa8_other_paa2": {
            "price": 8, "market_context": 2, "candle_volatility": 2, "volume_shock": 2,
        },
    }
    metrics = {
        name: {"pruned": 0, "violations": 0, "maximum_excess": 0.0}
        for name in layout_specs
    }
    real_rows = native_pruned = 0
    started = perf_counter()
    for key in selected:
        frame = source.load(key)
        frame = frame[frame.timestamp <= latest].reset_index(drop=True)
        batch = sliding_exact_representations(frame, benchmark, lookback=252, stride=5, batch_size=512)
        if not batch.representations:
            continue
        candidates = batch.representations
        native = batch_representation_lower_bounds(query_representation, candidates)
        candidate_stage = np.stack([row.stage for row in candidates])
        candidate_structural = np.stack([row.structural for row in candidates])
        candidate_coarse = np.stack([row.coarse for row in candidates])
        base, _ = _compact_bounds(
            query_representation.stage, query_representation.structural, query_representation.coarse,
            candidate_stage, candidate_structural, candidate_coarse,
        )
        # Remove the coarse term: layouts below account only for stage,
        # structural, and explicitly declared channel groups.
        coarse_only, parts = _compact_bounds(
            query_representation.stage, query_representation.structural, query_representation.coarse,
            candidate_stage, candidate_structural, candidate_coarse,
        )
        base = coarse_only - weights["coarse"] * parts["coarse"]
        for layout, groups in layout_specs.items():
            lower = base.copy()
            for group, paa_count in groups.items():
                lower += weights[group] * _group_bound(query_representation, candidates, group, paa_count)
            excess = lower - native.totals
            metrics[layout]["pruned"] += int(np.sum(lower >= threshold))
            metrics[layout]["violations"] += int(np.sum(excess > 1e-12))
            metrics[layout]["maximum_excess"] = max(metrics[layout]["maximum_excess"], float(np.max(excess)))
        native_pruned += int(np.sum(native.totals >= threshold))
        real_rows += len(candidates)
    channel_counts = {group: len(names) for group, names in GROUPS.items()}
    for layout, groups in layout_specs.items():
        # PAA values are float16.  Two float32 moment values per channel count
        # as four float16-width values; outward error metadata is still an open
        # production-contract detail.
        dimensions = 57 + sum(channel_counts[group] * (paa + 4) for group, paa in groups.items())
        metrics[layout].update({
            "dimensions": dimensions,
            "bytes_per_row": dimensions * 2,
            "projected_3_82m_gib": dimensions * 2 * 3_820_000 / 1024 ** 3,
            "projected_5m_gib": dimensions * 2 * 5_000_000 / 1024 ** 3,
            "pruning_fraction": metrics[layout]["pruned"] / real_rows,
            "pruning_retention": metrics[layout]["pruned"] / native_pruned,
        })
    randomized = [
        _randomized_channel_check(args.random_pairs, args.seed, paa_count)
        for paa_count in (2, 4, 8)
    ]
    payload = {
        "schema_version": "m04r-pair-scaled-bound-poc-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "query_episode_id": args.episode_id,
        "symbols": len(selected),
        "real_rows": real_rows,
        "threshold": threshold,
        "native_pruned": native_pruned,
        "native_pruning_fraction": native_pruned / real_rows,
        "layouts": metrics,
        "randomized_channel_checks": randomized,
        "elapsed_seconds": perf_counter() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "real_forward_outcomes_accessed": False,
        "interpretation": "POC evidence only; range-denominator proof and tests require independent review",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2))
    failed = any(row["violations"] for row in metrics.values()) or any(row["violations"] for row in randomized)
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
