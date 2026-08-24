"""POC for SFA/DCT partial-energy bounds on pair-scaled exact channels."""

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
from scipy.fft import dct

from compact_bound_poc import _compact_bounds
from market_analogues.adapters import source_from_spec
from market_analogues.config import load_config
from market_analogues.distance import DistanceConfig, GROUPS
from market_analogues.episodes import build_episode
from market_analogues.exact_batch import batch_representation_lower_bounds, sliding_exact_representations
from market_analogues.representation import Representation, represent
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import InstrumentKey


def _group_spectral_bound(
    query: Representation,
    candidates: tuple[Representation, ...],
    group: str,
    coefficients: int,
    denominator_mode: str,
) -> np.ndarray:
    distances: list[np.ndarray] = []
    included: list[np.ndarray] = []
    for name in GROUPS[group]:
        q = query.samples_48.get(name)
        values = [candidate.samples_48.get(name) for candidate in candidates]
        present = np.asarray([value is not None for value in values])
        if q is None:
            distances.append(np.where(present, 2.0, 0.0))
            included.append(present)
            continue
        matrix = np.stack([value if value is not None else np.zeros(48) for value in values])
        q_coeff = dct(q, norm="ortho")[:coefficients]
        c_coeff = dct(matrix, axis=1, norm="ortho")[:, :coefficients]
        stored = c_coeff.astype(np.float16).astype(float)
        error_norm = np.linalg.norm(c_coeff - stored, axis=1)
        numerator = np.maximum(
            np.linalg.norm(stored - q_coeff[None, :], axis=1) - error_norm,
            0.0,
        ) / np.sqrt(48.0)
        joined = np.concatenate((matrix, np.broadcast_to(q, matrix.shape)), axis=1)
        if denominator_mode == "oracle":
            iqr = np.percentile(joined, 75, axis=1) - np.percentile(joined, 25, axis=1)
            std = np.std(joined, axis=1)
            denominator = np.maximum(np.where(iqr < 1e-8, std, iqr), 1e-6)
        elif denominator_mode == "statistical":
            std = np.std(joined, axis=1)
            range_ = np.max(joined, axis=1) - np.min(joined, axis=1)
            denominator = np.maximum(np.minimum(range_, 4.0 * std), 1e-6)
        else:
            raise ValueError(denominator_mode)
        distance = numerator / denominator
        distance[~present] = 2.0
        distances.append(distance)
        included.append(np.ones(len(candidates), dtype=bool))
    count = np.sum(included, axis=0)
    result = np.divide(
        np.sum(distances, axis=0), count,
        out=np.zeros(len(candidates), dtype=float), where=count > 0,
    )
    return .55 * result if group == "price" else result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--symbols", type=int, default=48)
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
    keys.sort(key=lambda key: sha256(f"m04r-spectral:{key.source_symbol}".encode()).hexdigest())
    selected = keys[:args.symbols]
    selected_symbols = {key.source_symbol for key in selected}
    wanted = {str(row["symbol"]) for row in authority["matches"]}
    selected.extend(key for key in keys if key.source_symbol in wanted - selected_symbols)
    threshold = float(authority["certificate"]["stop_threshold"])
    weights = DistanceConfig().weights
    variants = {
        f"dct_{count}_{mode}": {
            "coefficients": count, "denominator": mode,
            "pruned": 0, "violations": 0, "maximum_excess": 0.0,
        }
        for count in (4, 8, 16, 24)
        for mode in ("oracle", "statistical")
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
        native = batch_representation_lower_bounds(query_representation, candidates).totals
        stage = np.stack([row.stage for row in candidates])
        structural = np.stack([row.structural for row in candidates])
        coarse = np.stack([row.coarse for row in candidates])
        base, _ = _compact_bounds(
            query_representation.stage, query_representation.structural, query_representation.coarse,
            stage, structural, coarse,
        )
        for metrics in variants.values():
            lower = base.copy()
            for group in GROUPS:
                lower += weights[group] * _group_spectral_bound(
                    query_representation, candidates, group,
                    int(metrics["coefficients"]), str(metrics["denominator"]),
                )
            excess = lower - native
            metrics["pruned"] += int(np.sum(lower >= threshold))
            metrics["violations"] += int(np.sum(excess > 1e-12))
            metrics["maximum_excess"] = max(float(metrics["maximum_excess"]), float(np.max(excess)))
        native_pruned += int(np.sum(native >= threshold))
        real_rows += len(native)
    for metrics in variants.values():
        count = int(metrics["coefficients"])
        # 90-value stage/structure/coarse base + DCT coefficients and two
        # float32 moment summaries (four float16-width values) per channel.
        dimensions = 90 + 19 * (count + 4)
        metrics.update({
            "dimensions": dimensions,
            "bytes_per_row": dimensions * 2,
            "projected_3_82m_gib": dimensions * 2 * 3_820_000 / 1024 ** 3,
            "pruning_fraction": int(metrics["pruned"]) / real_rows,
            "pruning_retention": int(metrics["pruned"]) / native_pruned,
        })
    payload = {
        "schema_version": "m04r-spectral-bound-poc-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "query_episode_id": args.episode_id,
        "symbols": len(selected),
        "real_rows": real_rows,
        "native_pruning_fraction": native_pruned / real_rows,
        "variants": variants,
        "elapsed_seconds": perf_counter() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "real_forward_outcomes_accessed": False,
        "interpretation": "oracle-denominator variants isolate numerator capacity and are not deployable bounds",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2))
    return 2 if any(int(row["violations"]) for row in variants.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
