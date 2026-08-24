"""POC for a quantization-safe compact lower-bound layout.

The proposed row stores exact-aligned stage (48), structural (9), coarse PAA
(32), and one coarse-vector RMS value as float16: 90 values / 180 bytes.  The
experiment verifies algebraic bounds against native components on real market
windows and randomized/adversarial vectors.  It is not a production proof.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import resource
from time import perf_counter

import numpy as np

from market_analogues.adapters import source_from_spec
from market_analogues.config import load_config
from market_analogues.distance import DistanceConfig
from market_analogues.episodes import build_episode
from market_analogues.exact_batch import (
    batch_representation_lower_bounds,
    sliding_exact_representations,
)
from market_analogues.representation import represent
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import InstrumentKey, SearchQuery


def _paa32(rows: np.ndarray) -> np.ndarray:
    return np.asarray(rows, dtype=float).reshape(len(rows), 32, 4).mean(axis=2)


def _rms(rows: np.ndarray, axis: int = 1) -> np.ndarray:
    return np.sqrt(np.mean(np.asarray(rows, dtype=float) ** 2, axis=axis))


def _quantized_rms_bound(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    stored = candidates.astype(np.float16).astype(float)
    errors = _rms(candidates - stored)
    observed = _rms(stored - query[None, :])
    return np.maximum(observed - errors, 0.0)


def _compact_bounds(
    query_stage: np.ndarray,
    query_structural: np.ndarray,
    query_coarse: np.ndarray,
    candidate_stage: np.ndarray,
    candidate_structural: np.ndarray,
    candidate_coarse: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    stage = _quantized_rms_bound(query_stage, candidate_stage)
    structural = _quantized_rms_bound(query_structural, candidate_structural)

    query_paa = _paa32(query_coarse[None, :])[0]
    candidate_paa = _paa32(candidate_coarse)
    numerator = _quantized_rms_bound(query_paa, candidate_paa)
    # std(concat(q,c)) <= RMS(concat(q,c)).  Store a conservatively rounded-up
    # candidate RMS; the denominator upper bound therefore cannot overstate the
    # resulting lower bound.
    exact_candidate_rms = _rms(candidate_coarse)
    rounded = exact_candidate_rms.astype(np.float16)
    rounded_up = np.where(
        rounded.astype(float) < exact_candidate_rms,
        np.nextafter(rounded, np.float16(np.inf)).astype(float),
        rounded.astype(float),
    )
    query_rms = float(_rms(query_coarse[None, :])[0])
    denominator_upper = np.maximum(
        np.sqrt((rounded_up ** 2 + query_rms ** 2) / 2.0), 1e-6,
    )
    coarse = numerator / denominator_upper
    weights = DistanceConfig().weights
    total = (
        weights["stage"] * stage
        + weights["structural"] * structural
        + weights["coarse"] * coarse
    )
    return total, {"stage": stage, "structural": structural, "coarse": coarse}


def _randomized_check(pairs: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    violations = 0
    maximum_excess = 0.0
    processed = 0
    chunk = 2_000
    while processed < pairs:
        count = min(chunk, pairs - processed)
        query_stage = rng.normal(size=(count, 48))
        candidate_stage = query_stage + rng.normal(scale=rng.lognormal(-2, 1, count)[:, None], size=(count, 48))
        query_structural = rng.normal(size=(count, 9))
        candidate_structural = query_structural + rng.normal(scale=.3, size=(count, 9))
        query_coarse = rng.normal(size=(count, 128))
        candidate_coarse = query_coarse + rng.normal(scale=.5, size=(count, 128))
        # Include flat, extreme, and nearly-constant rows in every chunk.
        candidate_coarse[0].fill(0.0)
        candidate_coarse[1].fill(1e-7)
        candidate_coarse[2].fill(1e3)
        query_coarse[0].fill(0.0)
        query_coarse[1].fill(-1e-7)
        query_coarse[2].fill(-1e3)
        for row in range(count):
            compact, parts = _compact_bounds(
                query_stage[row], query_structural[row], query_coarse[row],
                candidate_stage[row:row + 1], candidate_structural[row:row + 1],
                candidate_coarse[row:row + 1],
            )
            native_stage = float(_rms((candidate_stage[row] - query_stage[row])[None, :])[0])
            native_structural = float(_rms((candidate_structural[row] - query_structural[row])[None, :])[0])
            joined = np.r_[candidate_coarse[row], query_coarse[row]]
            native_coarse = float(_rms((candidate_coarse[row] - query_coarse[row])[None, :])[0]) / max(float(np.std(joined)), 1e-6)
            native = (
                .30 * native_stage + .09 * native_structural + .08 * native_coarse
            )
            excess = max(
                float(compact[0] - native),
                float(parts["stage"][0] - native_stage),
                float(parts["structural"][0] - native_structural),
                float(parts["coarse"][0] - native_coarse),
            )
            if excess > 1e-12:
                violations += 1
            maximum_excess = max(maximum_excess, excess)
        processed += count
    return {
        "pairs": pairs,
        "seed": seed,
        "violations": violations,
        "maximum_excess": maximum_excess,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--symbols", type=int, default=64)
    parser.add_argument("--random-pairs", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    source = source_from_spec(config.datasets["nasdaq"])
    authority_path = config.artifact_dir / "gate12" / "authorities" / "nasdaq" / "cases" / f"{args.episode_id}.json"
    authority = json.loads(authority_path.read_text())
    meta = authority["query"]
    query = build_episode(
        source, InstrumentKey("nasdaq", str(meta["symbol"])), str(meta["cutoff"]),
        int(meta["lookback"]), str(meta["representation_version"]),
    )
    query_representation = represent(query)
    request = SearchQuery(query.key, ("nasdaq",), ("A", "B"), 20, minimum_history_gap_bars=60)
    latest = latest_eligible_cutoff(query, 60)
    benchmark = source.load_benchmark()
    quality_path = config.artifact_dir / "quality" / "nasdaq.parquet"
    import pandas as pd
    quality = pd.read_parquet(quality_path)
    tiers = dict(zip(quality.symbol.astype(str), quality.tier.astype(str)))
    keys = [key for key in source.instruments() if tiers.get(key.source_symbol, "A") in {"A", "B"}]
    keys.sort(key=lambda key: sha256(f"m04r-bound-poc:{key.source_symbol}".encode()).hexdigest())
    selected = keys[:args.symbols]
    # Always include symbols that contain the authority top-20.
    selected_symbols = {key.source_symbol for key in selected}
    wanted = {str(row["symbol"]) for row in authority["matches"]}
    selected.extend(key for key in keys if key.source_symbol in wanted - selected_symbols)

    started = perf_counter()
    real_rows = 0
    violations = {"total": 0, "stage": 0, "structural": 0, "coarse": 0}
    maximum_excess = {name: 0.0 for name in violations}
    native_pruned = compact_pruned = 0
    threshold = float(authority["certificate"]["stop_threshold"])
    for key in selected:
        frame = source.load(key)
        frame = frame[frame.timestamp <= latest].reset_index(drop=True)
        batch = sliding_exact_representations(
            frame, benchmark, lookback=query.key.lookback, stride=5, batch_size=512,
        )
        if not batch.representations:
            continue
        candidates = batch.representations
        native = batch_representation_lower_bounds(query_representation, candidates)
        candidate_stage = np.stack([row.stage for row in candidates])
        candidate_structural = np.stack([row.structural for row in candidates])
        candidate_coarse = np.stack([row.coarse for row in candidates])
        compact, parts = _compact_bounds(
            query_representation.stage, query_representation.structural,
            query_representation.coarse, candidate_stage, candidate_structural,
            candidate_coarse,
        )
        comparisons = {
            "total": (compact, native.totals),
            "stage": (parts["stage"], native.components["stage"]),
            "structural": (parts["structural"], native.components["structural"]),
            "coarse": (parts["coarse"], native.components["coarse"]),
        }
        for name, (lower, exact) in comparisons.items():
            excess = lower - exact
            violations[name] += int(np.sum(excess > 1e-12))
            maximum_excess[name] = max(maximum_excess[name], float(np.max(excess)))
        native_pruned += int(np.sum(native.totals >= threshold))
        compact_pruned += int(np.sum(compact >= threshold))
        real_rows += len(candidates)

    randomized = _randomized_check(args.random_pairs, args.seed)
    elapsed = perf_counter() - started
    row_bytes = 90 * 2
    projected_rows = 5_000_000
    payload = {
        "schema_version": "m04r-compact-bound-poc-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "query_episode_id": args.episode_id,
        "symbols": len(selected),
        "real_rows": real_rows,
        "threshold": threshold,
        "real_violations": violations,
        "real_maximum_excess": maximum_excess,
        "native_pruned": native_pruned,
        "compact_pruned": compact_pruned,
        "compact_pruning_fraction": compact_pruned / real_rows if real_rows else 0.0,
        "native_pruning_fraction": native_pruned / real_rows if real_rows else 0.0,
        "pruning_retention": compact_pruned / native_pruned if native_pruned else 0.0,
        "randomized": randomized,
        "layout": {
            "stage_values": 48,
            "structural_values": 9,
            "coarse_paa_values": 32,
            "coarse_rms_values": 1,
            "storage_dtype": "float16",
            "bytes_per_row": row_bytes,
            "projected_5m_bytes": projected_rows * row_bytes,
            "projected_5m_gib": projected_rows * row_bytes / 1024 ** 3,
        },
        "elapsed_seconds": elapsed,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "real_forward_outcomes_accessed": False,
        "interpretation": "POC evidence only; randomized testing does not replace an algebraic proof",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2))
    return 0 if not any(violations.values()) and randomized["violations"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
