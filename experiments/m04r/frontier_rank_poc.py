"""Measure exact lower-bound proposal ranks from existing frozen frontiers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from market_analogues.exhaustive import load_frontier_shard


def run_case(artifact_dir: Path, episode_id: str) -> dict[str, object]:
    authority_path = artifact_dir / "gate12" / "authorities" / "nasdaq" / "cases" / f"{episode_id}.json"
    authority = json.loads(authority_path.read_text())
    targets = {str(row["episode_id"]): row for row in authority["matches"]}
    frontier = artifact_dir / "gate12" / "authorities" / "nasdaq" / "frontiers" / episode_id
    manifest = json.loads((frontier / "manifest.json").read_text())
    target_bounds: dict[str, float] = {}
    started = perf_counter()
    shard_paths = [frontier.parent / str(record["path"]) for record in manifest["shards"]]
    for path in shard_paths:
        shard = load_frontier_shard(path)
        wanted = targets.keys() & set(shard.episode_ids.astype(str))
        for target in wanted:
            position = int(np.flatnonzero(shard.episode_ids.astype(str) == target)[0])
            target_bounds[target] = float(shard.lower_bounds[position])
    missing = sorted(targets.keys() - target_bounds.keys())
    if missing:
        raise ValueError(f"targets absent from frontier: {missing}")
    lower_counts = {target: 0 for target in targets}
    tie_counts = {target: 0 for target in targets}
    references = np.asarray([target_bounds[target] for target in targets])
    target_ids = list(targets)
    for path in shard_paths:
        shard = load_frontier_shard(path)
        values = shard.lower_bounds[:, None]
        lower = np.sum(values < references[None, :], axis=0)
        ties = np.sum(values == references[None, :], axis=0)
        for index, target in enumerate(target_ids):
            lower_counts[target] += int(lower[index])
            tie_counts[target] += int(ties[index])
    rows = []
    for authority_rank, target in enumerate(target_ids, 1):
        match = targets[target]
        rows.append({
            "authority_rank": authority_rank,
            "episode_id": target,
            "symbol": match["symbol"],
            "exact_distance": match["total_distance"],
            "exact_lower_bound": target_bounds[target],
            "lower_bound_rank": lower_counts[target] + 1,
            "ties_including_self": tie_counts[target],
        })
    return {
        "query_episode_id": episode_id,
        "symbol": authority["query"]["symbol"],
        "eligible_rows": authority["certificate"]["eligible_candidates"],
        "exact_evaluated": authority["certificate"]["exact_evaluated"],
        "recall": {
            str(pool): sum(row["lower_bound_rank"] <= pool for row in rows) / len(rows)
            for pool in (1_000, 5_000, 10_000, 20_000)
        },
        "maximum_target_rank": max(row["lower_bound_rank"] for row in rows),
        "targets": rows,
        "elapsed_seconds": perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--episode-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "schema_version": "m04r-frontier-rank-poc-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cases": [run_case(args.artifact_dir, episode_id) for episode_id in args.episode_id],
        "real_forward_outcomes_accessed": False,
        "interpretation": "query-specific native lower-bound ceiling, not a query-independent production store",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output),
        "cases": [
            {key: case[key] for key in ("symbol", "recall", "maximum_target_rank", "elapsed_seconds")}
            for case in payload["cases"]
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
