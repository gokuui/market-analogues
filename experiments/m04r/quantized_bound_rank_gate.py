"""Full-universe rank and overflow census for the actual quantized bound.

Rows that cannot be represented by the certified float16/error-radius contract
are not clipped.  They receive routing bound zero (the universally safe lower
bound) and are recorded for a required exact/float32 sidecar in M04R-06.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
import multiprocessing
from pathlib import Path
import resource
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from market_analogues.adapters import source_from_spec
from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.config import load_config
from market_analogues.episodes import build_episode
from market_analogues.exact_batch import sliding_exact_representations
from market_analogues.quantized_bound import (
    QuantizedBoundError, quantize_bound_row, quantized_batch_lower_bounds,
    quantized_bound_contract, quantized_representation_lower_bound,
)
from market_analogues.representation import represent
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import EpisodeKey, InstrumentKey, stable_hash


SCHEMA_VERSION = "m04r-quantized-bound-rank-gate-v1"
ROUTE_QUOTAS = (100, 500, 1_000, 2_000)
_SOURCE: Any = None
_BENCHMARK: pd.DataFrame | None = None
_CASES: list[dict[str, Any]] | None = None
_STRIDE = 5
_MAXIMUM_LATEST: pd.Timestamp | None = None


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_cases(config: Any, source: Any) -> list[dict[str, Any]]:
    root = config.artifact_dir / "gate12" / "authorities" / "nasdaq" / "cases"
    paths = sorted(root.glob("*.json"))
    if len(paths) != 12:
        raise ValueError(f"require 12 NASDAQ authorities; found {len(paths)}")
    causal = json.loads((
        config.artifact_dir / "m04r-causal-prefixes" / "m04r-causal-prefixes.json"
    ).read_text())
    causal_by_id = {row["query_episode_id"]: row for row in causal["cases"]}
    benchmark = source.load_benchmark()
    cases = []
    target_cache = {}
    for path in paths:
        authority = json.loads(path.read_text())
        meta = authority["query"]
        query = build_episode(
            source, InstrumentKey("nasdaq", str(meta["symbol"])),
            str(meta["cutoff"]), int(meta["lookback"]),
            str(meta["representation_version"]),
        )
        if query.key.id != authority["query_episode_id"]:
            raise ValueError(f"query ID drift for {path.name}")
        frozen = causal_by_id[query.key.id]
        if causal_prefix_digest(
            source.load(query.key.instrument), query.key.cutoff,
        ).digest != frozen["stock_prefix"]["digest"]:
            raise ValueError(f"query stock prefix drift for {query.key.id}")
        if causal_prefix_digest(
            benchmark, query.key.cutoff,
        ).digest != frozen["benchmark_prefix"]["digest"]:
            raise ValueError(f"query benchmark prefix drift for {query.key.id}")
        query_representation = represent(query)
        targets = []
        target_scores = []
        for match in authority["matches"]:
            episode_id = str(match["episode_id"])
            if episode_id not in target_cache:
                episode = build_episode(
                    source, InstrumentKey("nasdaq", str(match["symbol"])),
                    str(match["cutoff"]), int(match["lookback"]),
                    str(match["representation_version"]),
                    str(match["quality_tier"]), tuple(match["quality_issues"]),
                )
                if episode.key.id != episode_id:
                    raise ValueError(f"target ID drift for {episode_id}")
                target_cache[episode_id] = represent(episode)
            try:
                packed = quantize_bound_row(target_cache[episode_id])
            except QuantizedBoundError as error:
                raise ValueError(f"authority target overflows: {episode_id}") from error
            score = quantized_representation_lower_bound(
                query_representation, packed,
            ).total
            target_scores.append(score)
            targets.append(match)
        cases.append({
            "authority": authority,
            "query": query,
            "query_representation": query_representation,
            "latest": latest_eligible_cutoff(query, 60),
            "query_start": pd.Timestamp(query.bars.timestamp.iloc[0]),
            "targets": targets,
            "target_scores": target_scores,
        })
    return cases


def _empty_delta(case_count: int) -> dict[str, Any]:
    return {
        "eligible_rows": [0] * case_count,
        "target_seen": [[0] * 20 for _ in range(case_count)],
        "less": [[0] * 20 for _ in range(case_count)],
        "ties": [[0] * 20 for _ in range(case_count)],
        "overflow_rows": 0,
        "overflow_examples": [],
        "prefixes": {},
        "processed_symbols": [],
        "representation_seconds": 0.0,
        "quantization_seconds": 0.0,
        "scoring_seconds": 0.0,
    }


def _merge(left: dict[str, Any], right: dict[str, Any]) -> None:
    for field in ("eligible_rows", "target_seen", "less", "ties"):
        left[field] = (
            np.asarray(left[field], dtype=np.int64)
            + np.asarray(right[field], dtype=np.int64)
        ).tolist()
    left["overflow_rows"] += int(right["overflow_rows"])
    remaining = max(20 - len(left["overflow_examples"]), 0)
    left["overflow_examples"].extend(right["overflow_examples"][:remaining])
    left["prefixes"].update(right["prefixes"])
    left["processed_symbols"].extend(right["processed_symbols"])
    for field in (
        "representation_seconds", "quantization_seconds", "scoring_seconds",
    ):
        left[field] += float(right[field])


def _process_symbol(symbol: str) -> dict[str, Any]:
    if _SOURCE is None or _BENCHMARK is None or _CASES is None or _MAXIMUM_LATEST is None:
        raise RuntimeError("quantized rank worker state is absent")
    delta = _empty_delta(len(_CASES))
    key = InstrumentKey("nasdaq", symbol)
    full = _SOURCE.load(key)
    delta["prefixes"][symbol] = asdict(causal_prefix_digest(full, _MAXIMUM_LATEST))
    frame = full[full.timestamp <= _MAXIMUM_LATEST].reset_index(drop=True)
    started = perf_counter()
    batch = sliding_exact_representations(
        frame, _BENCHMARK, lookback=252, stride=_STRIDE, batch_size=512,
    )
    delta["representation_seconds"] += perf_counter() - started
    delta["processed_symbols"].append(symbol)
    if not batch.representations:
        return delta
    cutoffs = pd.to_datetime(frame.timestamp.iloc[batch.positions]).to_numpy()
    started = perf_counter()
    packed = []
    packed_positions = []
    overflow_positions = []
    for row, representation in enumerate(batch.representations):
        try:
            packed.append(quantize_bound_row(representation))
            packed_positions.append(row)
        except QuantizedBoundError:
            overflow_positions.append(row)
    delta["quantization_seconds"] += perf_counter() - started
    delta["overflow_rows"] = len(overflow_positions)
    for row in overflow_positions[:20]:
        delta["overflow_examples"].append({
            "symbol": symbol,
            "cutoff": pd.Timestamp(cutoffs[row]).isoformat(),
            "routing_bound": 0.0,
        })
    packed_positions_array = np.asarray(packed_positions, dtype=int)
    for case_index, case in enumerate(_CASES):
        eligible = cutoffs <= np.datetime64(case["latest"])
        if symbol == case["query"].key.instrument.source_symbol:
            eligible &= cutoffs < np.datetime64(case["query_start"])
        positions = np.flatnonzero(eligible)
        delta["eligible_rows"][case_index] += len(positions)
        if not len(positions):
            continue
        for target_index, target in enumerate(case["targets"]):
            if str(target["symbol"]) != symbol:
                continue
            found = np.flatnonzero(
                cutoffs[positions] == np.datetime64(pd.Timestamp(target["cutoff"]))
            )
            if len(found) == 1 and EpisodeKey(
                key, pd.Timestamp(target["cutoff"]), 252, "dense-v1",
            ).id == target["episode_id"]:
                delta["target_seen"][case_index][target_index] += 1
        values = np.zeros(len(batch.representations), dtype=np.float64)
        if packed:
            started = perf_counter()
            values[packed_positions_array] = quantized_batch_lower_bounds(
                case["query_representation"], packed,
            ).totals
            delta["scoring_seconds"] += perf_counter() - started
        eligible_values = values[positions, None]
        targets = np.asarray(case["target_scores"], dtype=np.float64)[None, :]
        delta["less"][case_index] = (
            np.asarray(delta["less"][case_index])
            + np.sum(eligible_values < targets, axis=0)
        ).tolist()
        delta["ties"][case_index] = (
            np.asarray(delta["ties"][case_index])
            + np.sum(eligible_values == targets, axis=0)
        ).tolist()
    return delta


def _process_chunk(symbols: list[str]) -> dict[str, Any]:
    if _CASES is None:
        raise RuntimeError("quantized rank cases are absent")
    output = _empty_delta(len(_CASES))
    for symbol in symbols:
        _merge(output, _process_symbol(symbol))
    return output


def _initialize(
    cases: list[dict[str, Any]], symbols: list[str], stride: int,
) -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA_VERSION}-checkpoint",
        "contract_digest": quantized_bound_contract()["digest"],
        "authority_digests": [case["authority"]["authority_digest"] for case in cases],
        "symbols": symbols,
        "stride": stride,
        **_empty_delta(len(cases)),
    }


def _summarize(
    checkpoint: dict[str, Any], cases: list[dict[str, Any]], elapsed: float,
) -> dict[str, Any]:
    rows = []
    maximum_rank = 0
    for case_index, case in enumerate(cases):
        less = np.asarray(checkpoint["less"][case_index], dtype=np.int64)
        ties = np.asarray(checkpoint["ties"][case_index], dtype=np.int64)
        lower = less + 1
        upper = less + ties
        targets = []
        for index, target in enumerate(case["targets"]):
            maximum_rank = max(maximum_rank, int(upper[index]))
            targets.append({
                "authority_rank": index + 1,
                "episode_id": target["episode_id"],
                "symbol": target["symbol"],
                "quantized_bound": case["target_scores"][index],
                "lower_rank": int(lower[index]),
                "upper_rank": int(upper[index]),
                "ties_including_target": int(ties[index]),
            })
        observed = int(checkpoint["eligible_rows"][case_index])
        expected = int(case["authority"]["certificate"]["eligible_candidates"])
        rows.append({
            "query_episode_id": case["authority"]["query_episode_id"],
            "authority_digest": case["authority"]["authority_digest"],
            "symbol": case["authority"]["query"]["symbol"],
            "cutoff": case["authority"]["query"]["cutoff"],
            "eligible_rows": observed,
            "authority_eligible_rows": expected,
            "row_accounting_matches": observed == expected,
            "targets_seen_once": checkpoint["target_seen"][case_index] == [1] * 20,
            "recall": {
                str(quota): sum(int(rank) <= quota for rank in upper) / 20
                for quota in ROUTE_QUOTAS
            },
            "maximum_target_rank": int(np.max(upper)),
            "targets": targets,
        })
    deterministic = {
        "schema_version": SCHEMA_VERSION,
        "contract_digest": quantized_bound_contract()["digest"],
        "authority_cases": rows,
        "authority_case_count": len(rows),
        "universe_symbols": len(checkpoint["symbols"]),
        "source_prefix_count": len(checkpoint["prefixes"]),
        "source_prefixes": checkpoint["prefixes"],
        "benchmark_prefix": checkpoint["benchmark_prefix"],
        "source_scope_digest": stable_hash({
            "stocks": checkpoint["prefixes"],
            "benchmark": checkpoint["benchmark_prefix"],
        }),
        "stride": checkpoint["stride"],
        "overflow_rows": checkpoint["overflow_rows"],
        "overflow_examples": checkpoint["overflow_examples"],
        "overflow_routing_policy": (
            "no quantized row emitted; route with universal safe bound zero and "
            "require exact/float32 sidecar"
        ),
        "maximum_target_rank": maximum_rank,
        "top_100_recall_passed": all(row["recall"]["100"] == 1.0 for row in rows),
        "top_1000_recall_passed": all(row["recall"]["1000"] == 1.0 for row in rows),
        "all_row_accounting_passed": all(
            row["row_accounting_matches"] and row["targets_seen_once"] for row in rows
        ),
        "real_forward_outcomes_accessed": False,
    }
    deterministic["rank_gate_passed"] = (
        deterministic["top_1000_recall_passed"]
        and deterministic["all_row_accounting_passed"]
    )
    return {
        **deterministic,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "representation_seconds": checkpoint["representation_seconds"],
        "quantization_seconds": checkpoint["quantization_seconds"],
        "scoring_seconds": checkpoint["scoring_seconds"],
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "result_digest": stable_hash(deterministic),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--worker-chunk", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    started = perf_counter()
    config = load_config(args.config)
    source = source_from_spec(config.datasets["nasdaq"])
    benchmark = source.load_benchmark()
    cases = _load_cases(config, source)
    quality = pd.read_parquet(config.artifact_dir / "quality" / "nasdaq.parquet")
    tiers = dict(zip(quality.symbol.astype(str), quality.tier.astype(str)))
    symbols = [
        key.source_symbol for key in source.instruments()
        if tiers.get(key.source_symbol, "A") in {"A", "B"}
    ]
    symbols.sort(key=lambda symbol: sha256(
        f"m04r-quantized-bound-rank:{symbol}".encode(),
    ).hexdigest())
    checkpoint_path = args.output.with_suffix(args.output.suffix + ".checkpoint")
    maximum_latest = max(case["latest"] for case in cases)
    expected = _initialize(cases, symbols, 5)
    expected["benchmark_prefix"] = asdict(causal_prefix_digest(
        benchmark, maximum_latest,
    ))
    if checkpoint_path.exists() and not args.restart:
        checkpoint = json.loads(checkpoint_path.read_text())
        for field in ("schema_version", "contract_digest", "authority_digests", "symbols", "stride"):
            if checkpoint[field] != expected[field]:
                raise ValueError(f"checkpoint {field} differs")
        if "benchmark_prefix" not in checkpoint:
            checkpoint["benchmark_prefix"] = expected["benchmark_prefix"]
        elif checkpoint["benchmark_prefix"] != expected["benchmark_prefix"]:
            raise ValueError("checkpoint benchmark causal prefix differs")
        if len(checkpoint["processed_symbols"]) != len(set(checkpoint["processed_symbols"])):
            raise ValueError("checkpoint contains duplicate processed symbols")
        for symbol in checkpoint["processed_symbols"]:
            prefix = causal_prefix_digest(
                source.load(InstrumentKey("nasdaq", symbol)), maximum_latest,
            )
            if asdict(prefix) != checkpoint["prefixes"][symbol]:
                raise ValueError(f"checkpoint causal prefix drift for {symbol}")
    else:
        checkpoint = expected
        _write_json(checkpoint_path, checkpoint)
    completed = set(checkpoint["processed_symbols"])
    remaining = [symbol for symbol in symbols if symbol not in completed]
    chunks = [
        remaining[first:first + args.worker_chunk]
        for first in range(0, len(remaining), args.worker_chunk)
    ]
    global _SOURCE, _BENCHMARK, _CASES, _STRIDE, _MAXIMUM_LATEST
    _SOURCE, _BENCHMARK, _CASES = source, benchmark, cases
    _STRIDE = checkpoint["stride"]
    _MAXIMUM_LATEST = maximum_latest
    executor = ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("fork"),
    )
    try:
        next_checkpoint = args.checkpoint_every
        newly_completed = 0
        for chunk, delta in zip(chunks, executor.map(_process_chunk, chunks)):
            _merge(checkpoint, delta)
            newly_completed += len(chunk)
            if newly_completed >= next_checkpoint:
                _write_json(checkpoint_path, checkpoint)
                print(json.dumps({
                    "processed": len(checkpoint["processed_symbols"]),
                    "total": len(symbols), "last_symbol": chunk[-1],
                    "overflow_rows": checkpoint["overflow_rows"],
                    "elapsed_seconds": perf_counter() - started,
                }), flush=True)
                next_checkpoint += args.checkpoint_every
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    _write_json(checkpoint_path, checkpoint)
    payload = _summarize(checkpoint, cases, perf_counter() - started)
    _write_json(args.output, payload)
    html_path = args.output.with_suffix(".html")
    table = "".join(
        f"<tr><td>{escape(row['symbol'])}</td><td>{escape(row['cutoff'])}</td>"
        f"<td>{row['eligible_rows']:,}</td><td>{row['maximum_target_rank']}</td>"
        f"<td>{row['recall']['100']:.0%}</td><td>{row['recall']['1000']:.0%}</td></tr>"
        for row in payload["authority_cases"]
    )
    metadata = {key: value for key, value in payload.items() if key not in {
        "authority_cases", "source_prefixes", "benchmark_prefix",
    }}
    html_path.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>M04R quantized bound ranks</title><style>body{{font-family:system-ui;max-width:1300px;margin:2rem auto}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.5rem;border-bottom:1px solid #ddd;text-align:left}}pre{{white-space:pre-wrap}}</style></head><body><h1>Full-universe quantized-bound rank gate: {'PASS' if payload['rank_gate_passed'] else 'FAIL'}</h1><p>The predeclared 10k hybrid reserves 1,000 bound rows. Top 100 is a stricter pool-1k diagnostic and is reported separately.</p><p>Overflow rows use safe routing bound zero and require a sidecar; they are never clipped into a certified row.</p><p>Evidence <code>{payload['result_digest']}</code>.</p><pre>{escape(json.dumps(metadata, indent=2, sort_keys=True))}</pre><table><thead><tr><th>Query</th><th>Cutoff</th><th>Rows</th><th>Max rank</th><th>Top 100</th><th>Top 1000</th></tr></thead><tbody>{table}</tbody></table></body></html>""")
    print(json.dumps({
        "output": str(args.output), "maximum_target_rank": payload["maximum_target_rank"],
        "top_100_recall_passed": payload["top_100_recall_passed"],
        "top_1000_recall_passed": payload["top_1000_recall_passed"],
        "rank_gate_passed": payload["rank_gate_passed"],
        "overflow_rows": payload["overflow_rows"],
        "result_digest": payload["result_digest"],
        "elapsed_seconds": payload["elapsed_seconds"],
    }, indent=2))
    return 0 if payload["rank_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
