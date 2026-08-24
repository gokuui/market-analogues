"""Bounded M04R proposal-ranking POC over the immutable NASDAQ view store.

This is deliberately outside the production package.  It answers one design
question before implementation: can a global, exact-weighted signature score
retain every frozen exact top-20 neighbour without the per-symbol truncation?

The experiment uses two passes.  Pass one finds the authority-neighbour view
distances and builds a deterministic distribution sample.  Pass two computes
global ranks for several predeclared score normalizations and each individual
view.  No forward outcomes are opened.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from market_analogues.adapters import source_from_spec
from market_analogues.config import load_config
from market_analogues.distance import DistanceConfig
from market_analogues.episodes import build_episode
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import InstrumentKey, SearchQuery
from market_analogues.view_search import _load_manifest, _load_record
from market_analogues.view_signatures import (
    episode_view_signature,
    signature_view_distances,
)


VIEW_TO_EXACT = {
    "coarse": "coarse",
    "price_shape": "price",
    "stage": "stage",
    "candle_volatility": "candle_volatility",
    "volume_shock": "volume_shock",
    "market_context": "market_context",
    "structural": "structural",
}
METHODS = ("raw_weighted", "robust_weighted", "percentile_weighted")


@dataclass(frozen=True)
class Target:
    authority_rank: int
    episode_id: str
    symbol: str
    exact_distance: float


def _authority_path(artifact_dir: Path, episode_id: str) -> Path:
    paths = list((artifact_dir / "gate12" / "authorities" / "nasdaq" / "cases").glob(
        f"{episode_id}.json"
    ))
    if len(paths) != 1:
        raise ValueError(f"expected one NASDAQ authority for {episode_id}, found {len(paths)}")
    return paths[0]


def _score(
    matrix: np.ndarray,
    *,
    method: str,
    weights: np.ndarray,
    medians: np.ndarray,
    iqrs: np.ndarray,
    sorted_samples: tuple[np.ndarray, ...],
) -> np.ndarray:
    if method == "raw_weighted":
        transformed = matrix
    elif method == "robust_weighted":
        transformed = (matrix - medians) / iqrs
    elif method == "percentile_weighted":
        transformed = np.column_stack([
            np.searchsorted(sample, matrix[:, column], side="right") / len(sample)
            for column, sample in enumerate(sorted_samples)
        ])
    else:
        raise ValueError(f"unknown method {method}")
    return transformed @ weights


def run_case(config_path: Path, query_episode_id: str, sample_modulus: int) -> dict[str, object]:
    config = load_config(config_path)
    source = source_from_spec(config.datasets["nasdaq"])
    authority_path = _authority_path(config.artifact_dir, query_episode_id)
    authority = json.loads(authority_path.read_text())
    query_meta = authority["query"]
    query = build_episode(
        source,
        InstrumentKey("nasdaq", str(query_meta["symbol"])),
        str(query_meta["cutoff"]),
        int(query_meta["lookback"]),
        str(query_meta["representation_version"]),
    )
    if query.key.id != query_episode_id:
        raise ValueError("reconstructed query ID differs from authority")
    request = SearchQuery(
        query.key, ("nasdaq",), ("A", "B"), 20,
        minimum_history_gap_bars=60,
    )
    latest_ns = int(latest_eligible_cutoff(query, 60).value)
    query_start_ns = int(query.bars.timestamp.iloc[0].value)
    root = config.artifact_dir / "view-store"
    manifest, manifest_digest = _load_manifest(root, "nasdaq")
    records = [
        record for record in manifest["shards"]
        if int(record["lookback"]) == query.key.lookback
        and str(record["quality_tier"]) in request.quality_tiers
    ]
    records.sort(key=lambda row: (str(row["symbol"]), str(row["path"])))
    targets = tuple(
        Target(rank, str(match["episode_id"]), str(match["symbol"]), float(match["total_distance"]))
        for rank, match in enumerate(authority["matches"], 1)
    )
    target_by_symbol: dict[str, set[str]] = {}
    for target in targets:
        target_by_symbol.setdefault(target.symbol, set()).add(target.episode_id)
    query_signature = episode_view_signature(query)

    started = perf_counter()
    view_names: tuple[str, ...] | None = None
    samples: list[np.ndarray] = []
    target_views: dict[str, np.ndarray] = {}
    eligible_rows = 0
    # Pass one: deterministic distribution sample plus exact target view values.
    for record in records:
        shard = _load_record(root, record)
        eligible = shard.cutoffs_ns <= latest_ns
        if str(record["symbol"]) == query.key.instrument.source_symbol:
            eligible &= shard.cutoffs_ns < query_start_ns
        positions = np.flatnonzero(eligible)
        eligible_rows += len(positions)
        if not len(positions):
            continue
        distances = signature_view_distances(query_signature, shard.signatures[positions])
        if view_names is None:
            view_names = tuple(distances)
        matrix = np.column_stack([distances[name] for name in view_names])
        # Sampling is invariant to filesystem order because it uses the row's
        # stable cutoff integer, not a process-local counter or random state.
        sampled = (np.abs(shard.cutoffs_ns[positions] // 86_400_000_000_000) % sample_modulus) == 0
        if sampled.any():
            samples.append(matrix[sampled])
        wanted = target_by_symbol.get(str(record["symbol"]), set())
        if wanted:
            ids = shard.episode_ids[positions].astype(str)
            for episode_id in wanted:
                found = np.flatnonzero(ids == episode_id)
                if len(found) == 1:
                    target_views[episode_id] = matrix[int(found[0])]
    if view_names is None or not samples:
        raise ValueError("view store produced no eligible sample")
    missing = sorted({target.episode_id for target in targets}.difference(target_views))
    if missing:
        raise ValueError(f"authority targets absent from eligible view rows: {missing}")
    sample = np.concatenate(samples)
    medians = np.median(sample, axis=0)
    iqrs = np.percentile(sample, 75, axis=0) - np.percentile(sample, 25, axis=0)
    iqrs = np.maximum(iqrs, 1e-9)
    sorted_samples = tuple(np.sort(sample[:, column]) for column in range(sample.shape[1]))
    exact_weights = DistanceConfig().weights
    weights = np.asarray([exact_weights[VIEW_TO_EXACT[name]] for name in view_names], dtype=float)
    weights /= weights.sum()
    target_matrix = np.stack([target_views[target.episode_id] for target in targets])
    target_scores = {
        method: _score(
            target_matrix, method=method, weights=weights, medians=medians,
            iqrs=iqrs, sorted_samples=sorted_samples,
        )
        for method in METHODS
    }
    method_counts = {method: np.zeros(len(targets), dtype=np.int64) for method in METHODS}
    view_counts = np.zeros((len(targets), len(view_names)), dtype=np.int64)

    # Pass two: exact global rank counts. Floating ties are conservatively
    # counted ahead; ties are reported and should be examined before promotion.
    score_ties = {method: np.zeros(len(targets), dtype=np.int64) for method in METHODS}
    view_ties = np.zeros((len(targets), len(view_names)), dtype=np.int64)
    for record in records:
        shard = _load_record(root, record)
        eligible = shard.cutoffs_ns <= latest_ns
        if str(record["symbol"]) == query.key.instrument.source_symbol:
            eligible &= shard.cutoffs_ns < query_start_ns
        positions = np.flatnonzero(eligible)
        if not len(positions):
            continue
        distances = signature_view_distances(query_signature, shard.signatures[positions])
        matrix = np.column_stack([distances[name] for name in view_names])
        for column in range(len(view_names)):
            values = matrix[:, column, None]
            references = target_matrix[:, column][None, :]
            view_counts[:, column] += np.sum(values < references, axis=0)
            view_ties[:, column] += np.sum(values == references, axis=0)
        for method in METHODS:
            values = _score(
                matrix, method=method, weights=weights, medians=medians,
                iqrs=iqrs, sorted_samples=sorted_samples,
            )[:, None]
            references = target_scores[method][None, :]
            method_counts[method] += np.sum(values < references, axis=0)
            score_ties[method] += np.sum(values == references, axis=0)

    rows: list[dict[str, object]] = []
    for index, target in enumerate(targets):
        method_ranks = {method: int(method_counts[method][index] + 1) for method in METHODS}
        per_view_ranks = {
            name: int(view_counts[index, column] + 1)
            for column, name in enumerate(view_names)
        }
        # At most 20,000 rows: 2,000 globally best from each of seven views,
        # then up to 6,000 raw-weighted rows. Overlap only makes the pool smaller.
        route_admitted = (
            min(per_view_ranks.values()) <= 2_000
            or method_ranks["raw_weighted"] <= 6_000
        )
        rows.append({
            **asdict(target),
            "method_ranks": method_ranks,
            "per_view_ranks": per_view_ranks,
            "route_union_20k_admitted": route_admitted,
            "method_score_ties_including_self": {
                method: int(score_ties[method][index]) for method in METHODS
            },
            "view_ties_including_self": {
                name: int(view_ties[index, column]) for column, name in enumerate(view_names)
            },
        })
    recalls = {
        method: {
            str(pool): sum(row["method_ranks"][method] <= pool for row in rows) / len(rows)
            for pool in (1_000, 5_000, 10_000, 20_000)
        }
        for method in METHODS
    }
    recalls["route_union_20k"] = {
        "20000": sum(bool(row["route_union_20k_admitted"]) for row in rows) / len(rows)
    }
    return {
        "schema_version": "m04r-proposal-rank-poc-v1",
        "query_episode_id": query_episode_id,
        "symbol": query.key.instrument.source_symbol,
        "cutoff": query.key.cutoff.isoformat(),
        "authority_digest": authority["authority_digest"],
        "view_manifest_digest": manifest_digest,
        "eligible_rows": eligible_rows,
        "sample_rows": len(sample),
        "sample_modulus": sample_modulus,
        "view_names": list(view_names),
        "weights": {name: float(weight) for name, weight in zip(view_names, weights)},
        "normalizers": {
            name: {"median": float(medians[column]), "iqr": float(iqrs[column])}
            for column, name in enumerate(view_names)
        },
        "recall": recalls,
        "targets": rows,
        "elapsed_seconds": perf_counter() - started,
        "real_forward_outcomes_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episode-id", action="append", required=True)
    parser.add_argument("--sample-modulus", type=int, default=97)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sample_modulus < 2:
        raise SystemExit("--sample-modulus must be at least 2")
    cases = [run_case(args.config, episode_id, args.sample_modulus) for episode_id in args.episode_id]
    payload = {
        "schema_version": "m04r-proposal-rank-poc-matrix-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cases": cases,
        "real_forward_outcomes_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output),
        "cases": [
            {"symbol": case["symbol"], "recall": case["recall"], "seconds": case["elapsed_seconds"]}
            for case in cases
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
