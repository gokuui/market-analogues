"""M04R-05 compact proposal Pareto gate over frozen NASDAQ authorities.

The evaluator never materializes a second universe store.  It streams each
eligible OHLCV symbol once, projects the shared exact representation into all
predeclared layouts, and accumulates conservative global rank intervals for
every frozen authority target.  Checkpoints are atomic and source-prefix bound.
No forward outcome is read.
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
from market_analogues.distance import representation_distance
from market_analogues.episodes import build_episode
from market_analogues.exact_batch import sliding_exact_representations
from market_analogues.m04r_distance_verification import PRACTICAL_AUTHORITY_TOLERANCE
from market_analogues.proposal_v2 import (
    LAYOUTS, PROPOSAL_POOLS, PROPOSAL_ROUTES, proposal_signature_v2,
    proposal_signatures_v2, proposal_v2_contract, proposal_v2_distances,
    proposal_v2_distances_many, proposal_v2_route_admitted,
    proposal_v2_storage_bytes,
)
from market_analogues.representation import represent
from market_analogues.search import latest_eligible_cutoff
from market_analogues.synthetic import FAMILIES, generate_case, transform_case
from market_analogues.types import EpisodeKey, InstrumentKey, stable_hash


ROUTES = (*PROPOSAL_ROUTES, "composite")
DTYPES = ("float32", "float16")
SCHEMA_VERSION = "m04r-proposal-v2-authority-gate-v1"
_WORKER_SOURCE: Any = None
_WORKER_BENCHMARK: pd.DataFrame | None = None
_WORKER_CASES: list[dict[str, Any]] | None = None
_WORKER_STRIDE = 5
_WORKER_MAXIMUM_LATEST: pd.Timestamp | None = None


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    first = 0
    while first < len(values):
        last = first + 1
        while last < len(values) and sorted_values[last] == sorted_values[first]:
            last += 1
        ranks[order[first:last]] = (first + 1 + last) / 2
        first = last
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def _case_key(case_index: int, dimensions: int, dtype: str) -> str:
    return f"{case_index}:{dimensions}:{dtype}"


def _empty_counter(case_count: int) -> dict[str, Any]:
    shape = (len(ROUTES), 20)
    return {
        _case_key(case_index, dimensions, dtype): {
            "less": np.zeros(shape, dtype=np.int64).tolist(),
            "ties": np.zeros(shape, dtype=np.int64).tolist(),
        }
        for case_index in range(case_count)
        for dimensions in sorted(LAYOUTS)
        for dtype in DTYPES
    }


def _load_authorities(config: Any, source: Any) -> list[dict[str, Any]]:
    root = config.artifact_dir / "gate12" / "authorities" / "nasdaq" / "cases"
    paths = sorted(root.glob("*.json"))
    if len(paths) != 12:
        raise ValueError(f"require 12 NASDAQ authorities; found {len(paths)}")
    causal_path = config.artifact_dir / "m04r-causal-prefixes" / "m04r-causal-prefixes.json"
    causal = json.loads(causal_path.read_text())
    causal_by_id = {row["query_episode_id"]: row for row in causal["cases"]}
    benchmark = source.load_benchmark()
    representation_cache: dict[str, Any] = {}
    cases: list[dict[str, Any]] = []
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
        frozen_prefix = causal_by_id[query.key.id]
        stock_prefix = causal_prefix_digest(source.load(query.key.instrument), query.key.cutoff)
        benchmark_prefix = causal_prefix_digest(benchmark, query.key.cutoff)
        if stock_prefix.digest != frozen_prefix["stock_prefix"]["digest"]:
            raise ValueError(f"query causal stock prefix drift for {query.key.id}")
        if benchmark_prefix.digest != frozen_prefix["benchmark_prefix"]["digest"]:
            raise ValueError(f"query causal benchmark prefix drift for {query.key.id}")
        query_representation = represent(query)
        targets = []
        target_representations = []
        maximum_authority_total_delta = 0.0
        maximum_authority_component_delta = 0.0
        for match in authority["matches"]:
            target_id = str(match["episode_id"])
            if target_id not in representation_cache:
                episode = build_episode(
                    source, InstrumentKey("nasdaq", str(match["symbol"])),
                    str(match["cutoff"]), int(match["lookback"]),
                    str(match["representation_version"]),
                    str(match["quality_tier"]), tuple(match["quality_issues"]),
                )
                if episode.key.id != target_id:
                    raise ValueError(f"authority target ID drift for {target_id}")
                representation_cache[target_id] = represent(episode)
            target_representation = representation_cache[target_id]
            exact_total, exact_components, _ = representation_distance(
                query_representation, target_representation,
            )
            total_delta = abs(exact_total - float(match["total_distance"]))
            component_delta = max(
                abs(exact_components[name] - float(value))
                for name, value in match["component_distances"].items()
            )
            maximum_authority_total_delta = max(
                maximum_authority_total_delta, total_delta,
            )
            maximum_authority_component_delta = max(
                maximum_authority_component_delta, component_delta,
            )
            if total_delta > PRACTICAL_AUTHORITY_TOLERANCE:
                raise ValueError(f"authority exact total drift for {target_id}")
            if component_delta > PRACTICAL_AUTHORITY_TOLERANCE:
                raise ValueError(f"authority exact component drift for {target_id}")
            targets.append(match)
            target_representations.append(target_representation)
        latest = latest_eligible_cutoff(query, 60)
        target_scores: dict[str, dict[str, list[list[float]]]] = {}
        for dimensions in sorted(LAYOUTS):
            query_signature = proposal_signature_v2(query_representation, dimensions)
            target_batch = proposal_signatures_v2(target_representations, dimensions)
            by_dtype = {}
            for dtype in DTYPES:
                vectors = target_batch.vectors.astype(dtype)
                distances = proposal_v2_distances(
                    query_signature, vectors, target_batch.presence,
                )
                by_dtype[dtype] = [distances[route].tolist() for route in ROUTES]
            target_scores[str(dimensions)] = by_dtype
        cases.append({
            "authority": authority,
            "query": query,
            "query_representation": query_representation,
            "query_signatures": {
                dimensions: proposal_signature_v2(query_representation, dimensions)
                for dimensions in sorted(LAYOUTS)
            },
            "latest": latest,
            "query_start": pd.Timestamp(query.bars.timestamp.iloc[0]),
            "targets": targets,
            "target_scores": target_scores,
            "maximum_authority_total_delta": maximum_authority_total_delta,
            "maximum_authority_component_delta": maximum_authority_component_delta,
        })
    return cases


def _synthetic_gate() -> dict[str, Any]:
    rows = []
    for dimensions in sorted(LAYOUTS):
        for dtype in DTYPES:
            for family_index, family in enumerate(sorted(FAMILIES)):
                case = generate_case(family, 800 + family_index)
                query_representation = represent(case.episode)
                query = proposal_signature_v2(query_representation, dimensions)
                # The planted clone is deliberately after 27 same-family
                # decoys: an enumeration/local-cap rule cannot rescue it.
                representations = [
                    represent(transform_case(
                        case, name=f"decoy-{index}", noise=.0007 + index * .00003,
                        seed=900 + index,
                    ).episode)
                    for index in range(27)
                ]
                representations.append(query_representation)
                representations.extend(
                    represent(generate_case(other, 1_000 + index).episode)
                    for index, other in enumerate(sorted(FAMILIES)) if other != family
                )
                batch = proposal_signatures_v2(representations, dimensions)
                distances = proposal_v2_distances(
                    query, batch.vectors.astype(dtype), batch.presence,
                )
                clone_index = 27
                ranks = {
                    route: int(np.sum(values < values[clone_index]) + 1)
                    for route, values in distances.items()
                }
                rows.append({
                    "dimensions": dimensions, "dtype": dtype, "family": family,
                    "clone_input_position": clone_index + 1,
                    "route_ranks": ranks,
                    "passed": ranks["composite"] == 1,
                })
    return {
        "cases": rows,
        "case_count": len(rows),
        "all_passed": all(row["passed"] for row in rows),
        "adversarial_local_cap_depth": 27,
    }


def _initialize_checkpoint(
    contract_digest: str, cases: list[dict[str, Any]], symbols: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA_VERSION}-checkpoint",
        "contract_digest": contract_digest,
        "authority_digests": [case["authority"]["authority_digest"] for case in cases],
        "symbols": symbols,
        "stride": args.stride,
        "processed_symbols": [],
        "prefixes": {},
        "eligible_rows": [0] * len(cases),
        "target_seen": [[0] * 20 for _ in cases],
        "counters": _empty_counter(len(cases)),
        "projection_seconds": {str(dimensions): 0.0 for dimensions in LAYOUTS},
        "scoring_seconds": {
            f"{dimensions}:{dtype}": 0.0
            for dimensions in LAYOUTS for dtype in DTYPES
        },
        "quantization_overflow_rows": {str(dimensions): 0 for dimensions in LAYOUTS},
        "quantization_maximum_absolute_value": {
            str(dimensions): 0.0 for dimensions in LAYOUTS
        },
        "quantization_overflow_examples": {
            str(dimensions): [] for dimensions in LAYOUTS
        },
        "started_at": datetime.now(timezone.utc).isoformat(),
    }


def _validate_checkpoint(
    checkpoint: dict[str, Any], expected: dict[str, Any], source: Any,
    maximum_latest: pd.Timestamp,
) -> None:
    for name in ("schema_version", "contract_digest", "authority_digests", "symbols", "stride"):
        if checkpoint[name] != expected[name]:
            raise ValueError(f"checkpoint {name} differs from this run")
    # Resume never trusts stale accumulated counts: re-hash every completed
    # causal prefix before using them.
    for symbol in checkpoint["processed_symbols"]:
        prefix = causal_prefix_digest(
            source.load(InstrumentKey("nasdaq", symbol)), maximum_latest,
        )
        if asdict(prefix) != checkpoint["prefixes"][symbol]:
            raise ValueError(f"checkpoint causal prefix drift for {symbol}")


def _process_symbol(
    symbol: str, source: Any, benchmark: pd.DataFrame, cases: list[dict[str, Any]],
    checkpoint: dict[str, Any], stride: int, maximum_latest: pd.Timestamp,
) -> None:
    key = InstrumentKey("nasdaq", symbol)
    full = source.load(key)
    prefix = causal_prefix_digest(full, maximum_latest)
    frame = full[full.timestamp <= maximum_latest].reset_index(drop=True)
    batch = sliding_exact_representations(
        frame, benchmark, lookback=252, stride=stride, batch_size=512,
    )
    if not batch.representations:
        checkpoint["prefixes"][symbol] = asdict(prefix)
        checkpoint["processed_symbols"].append(symbol)
        return
    cutoffs = pd.to_datetime(frame.timestamp.iloc[batch.positions]).to_numpy()
    proposal_batches = {}
    for dimensions in sorted(LAYOUTS):
        started = perf_counter()
        proposal_batches[dimensions] = proposal_signatures_v2(
            batch.representations, dimensions,
        )
        checkpoint["projection_seconds"][str(dimensions)] += perf_counter() - started
    eligible_positions = []
    for case_index, case in enumerate(cases):
        eligible = cutoffs <= np.datetime64(case["latest"])
        if symbol == case["query"].key.instrument.source_symbol:
            eligible &= cutoffs < np.datetime64(case["query_start"])
        positions = np.flatnonzero(eligible)
        eligible_positions.append(positions)
        checkpoint["eligible_rows"][case_index] += int(len(positions))
        if not len(positions):
            continue
        target_cutoffs = {
            np.datetime64(pd.Timestamp(target["cutoff"])): target_index
            for target_index, target in enumerate(case["targets"])
            if str(target["symbol"]) == symbol
        }
        for cutoff, target_index in target_cutoffs.items():
            found = np.flatnonzero(cutoffs[positions] == cutoff)
            if len(found) == 1:
                candidate_key = EpisodeKey(
                    key, pd.Timestamp(cutoff), 252, "dense-v1",
                )
                if candidate_key.id == case["targets"][target_index]["episode_id"]:
                    checkpoint["target_seen"][case_index][target_index] += 1
    for dimensions, proposal_batch in proposal_batches.items():
        queries = [case["query_signatures"][dimensions] for case in cases]
        for dtype in DTYPES:
            started = perf_counter()
            vectors = proposal_batch.vectors
            if dtype == "float16":
                float16_limit = np.finfo(np.float16).max
                row_maximum = np.max(np.abs(vectors), axis=1)
                checkpoint["quantization_overflow_rows"][str(dimensions)] += int(
                    np.sum(row_maximum > float16_limit)
                )
                overflow_positions = np.flatnonzero(row_maximum > float16_limit)
                examples = checkpoint["quantization_overflow_examples"][str(dimensions)]
                for row in overflow_positions[:max(20 - len(examples), 0)]:
                    examples.append({
                        "symbol": symbol,
                        "cutoff": pd.Timestamp(cutoffs[row]).isoformat(),
                        "maximum_absolute_value": float(row_maximum[row]),
                    })
                checkpoint["quantization_maximum_absolute_value"][str(dimensions)] = max(
                    checkpoint["quantization_maximum_absolute_value"][str(dimensions)],
                    float(np.max(row_maximum)),
                )
                vectors = np.clip(vectors, -float16_limit, float16_limit)
            distances = proposal_v2_distances_many(
                queries, vectors.astype(dtype),
                proposal_batch.presence,
            )
            checkpoint["scoring_seconds"][f"{dimensions}:{dtype}"] += (
                perf_counter() - started
            )
            for case_index, case in enumerate(cases):
                positions = eligible_positions[case_index]
                if not len(positions):
                    continue
                counter = checkpoint["counters"][_case_key(
                    case_index, dimensions, dtype,
                )]
                less = np.asarray(counter["less"], dtype=np.int64)
                ties = np.asarray(counter["ties"], dtype=np.int64)
                references = np.asarray(
                    case["target_scores"][str(dimensions)][dtype], dtype=float,
                )
                for route_index, route in enumerate(ROUTES):
                    values = distances[route][case_index, positions, None]
                    target_values = references[route_index][None, :]
                    less[route_index] += np.sum(values < target_values, axis=0)
                    ties[route_index] += np.sum(values == target_values, axis=0)
                counter["less"] = less.tolist()
                counter["ties"] = ties.tolist()
    checkpoint["prefixes"][symbol] = asdict(prefix)
    checkpoint["processed_symbols"].append(symbol)


def _symbol_delta(
    symbol: str, source: Any, benchmark: pd.DataFrame, cases: list[dict[str, Any]],
    stride: int, maximum_latest: pd.Timestamp,
) -> dict[str, Any]:
    delta = {
        "eligible_rows": [0] * len(cases),
        "target_seen": [[0] * 20 for _ in cases],
        "counters": _empty_counter(len(cases)),
        "projection_seconds": {str(dimensions): 0.0 for dimensions in LAYOUTS},
        "scoring_seconds": {
            f"{dimensions}:{dtype}": 0.0
            for dimensions in LAYOUTS for dtype in DTYPES
        },
        "quantization_overflow_rows": {str(dimensions): 0 for dimensions in LAYOUTS},
        "quantization_maximum_absolute_value": {
            str(dimensions): 0.0 for dimensions in LAYOUTS
        },
        "quantization_overflow_examples": {
            str(dimensions): [] for dimensions in LAYOUTS
        },
        "prefixes": {},
        "processed_symbols": [],
    }
    _process_symbol(
        symbol, source, benchmark, cases, delta, stride, maximum_latest,
    )
    return delta


def _merge_delta(checkpoint: dict[str, Any], delta: dict[str, Any]) -> None:
    checkpoint["processed_symbols"].extend(delta["processed_symbols"])
    checkpoint["prefixes"].update(delta["prefixes"])
    checkpoint["eligible_rows"] = (
        np.asarray(checkpoint["eligible_rows"], dtype=np.int64)
        + np.asarray(delta["eligible_rows"], dtype=np.int64)
    ).tolist()
    checkpoint["target_seen"] = (
        np.asarray(checkpoint["target_seen"], dtype=np.int64)
        + np.asarray(delta["target_seen"], dtype=np.int64)
    ).tolist()
    for key, counter in checkpoint["counters"].items():
        for field in ("less", "ties"):
            counter[field] = (
                np.asarray(counter[field], dtype=np.int64)
                + np.asarray(delta["counters"][key][field], dtype=np.int64)
            ).tolist()
    for field in ("projection_seconds", "scoring_seconds"):
        for key, value in delta[field].items():
            checkpoint[field][key] += value
    for key, value in delta["quantization_overflow_rows"].items():
        checkpoint["quantization_overflow_rows"][key] += value
    for key, value in delta["quantization_maximum_absolute_value"].items():
        checkpoint["quantization_maximum_absolute_value"][key] = max(
            checkpoint["quantization_maximum_absolute_value"][key], value,
        )
    for key, values in delta["quantization_overflow_examples"].items():
        remaining = max(20 - len(checkpoint["quantization_overflow_examples"][key]), 0)
        checkpoint["quantization_overflow_examples"][key].extend(values[:remaining])


def _worker_chunk(symbols: list[str]) -> dict[str, Any]:
    if (
        _WORKER_SOURCE is None or _WORKER_BENCHMARK is None
        or _WORKER_CASES is None or _WORKER_MAXIMUM_LATEST is None
    ):
        raise RuntimeError("proposal worker state was not initialized")
    combined = {
        "eligible_rows": [0] * len(_WORKER_CASES),
        "target_seen": [[0] * 20 for _ in _WORKER_CASES],
        "counters": _empty_counter(len(_WORKER_CASES)),
        "projection_seconds": {str(dimensions): 0.0 for dimensions in LAYOUTS},
        "scoring_seconds": {
            f"{dimensions}:{dtype}": 0.0
            for dimensions in LAYOUTS for dtype in DTYPES
        },
        "quantization_overflow_rows": {str(dimensions): 0 for dimensions in LAYOUTS},
        "quantization_maximum_absolute_value": {
            str(dimensions): 0.0 for dimensions in LAYOUTS
        },
        "quantization_overflow_examples": {
            str(dimensions): [] for dimensions in LAYOUTS
        },
        "prefixes": {},
        "processed_symbols": [],
    }
    for symbol in symbols:
        delta = _symbol_delta(
            symbol, _WORKER_SOURCE, _WORKER_BENCHMARK, _WORKER_CASES,
            _WORKER_STRIDE, _WORKER_MAXIMUM_LATEST,
        )
        _merge_delta(combined, delta)
    return combined


def _summarize(
    checkpoint: dict[str, Any], cases: list[dict[str, Any]], universe_count: int,
    synthetic: dict[str, Any], elapsed_seconds: float,
) -> dict[str, Any]:
    case_rows = []
    layout_pass = {dimensions: True for dimensions in LAYOUTS}
    dtype_pass = {
        (dimensions, dtype): True for dimensions in LAYOUTS for dtype in DTYPES
    }
    correlations: dict[str, list[float]] = {}
    for case_index, case in enumerate(cases):
        authority = case["authority"]
        layouts = {}
        for dimensions in sorted(LAYOUTS):
            layouts[str(dimensions)] = {}
            for dtype in DTYPES:
                counter = checkpoint["counters"][_case_key(case_index, dimensions, dtype)]
                less = np.asarray(counter["less"], dtype=np.int64)
                ties = np.asarray(counter["ties"], dtype=np.int64)
                lower = less + 1
                upper = less + ties
                pool_recalls = {}
                target_rows = []
                for target_index, target in enumerate(case["targets"]):
                    lower_components = {
                        route: int(lower[route_index, target_index])
                        for route_index, route in enumerate(PROPOSAL_ROUTES)
                    }
                    upper_components = {
                        route: int(upper[route_index, target_index])
                        for route_index, route in enumerate(PROPOSAL_ROUTES)
                    }
                    target_rows.append({
                        "authority_rank": target_index + 1,
                        "episode_id": target["episode_id"],
                        "symbol": target["symbol"],
                        "lower_route_ranks": lower_components,
                        "upper_route_ranks": upper_components,
                        "lower_composite_rank": int(lower[-1, target_index]),
                        "upper_composite_rank": int(upper[-1, target_index]),
                        "ties_including_target": {
                            route: int(ties[route_index, target_index])
                            for route_index, route in enumerate(ROUTES)
                        },
                    })
                for pool in sorted(PROPOSAL_POOLS):
                    admitted = [
                        proposal_v2_route_admitted(
                            row["upper_route_ranks"], row["upper_composite_rank"], pool,
                        )
                        for row in target_rows
                    ]
                    pool_recalls[str(pool)] = {
                        "admitted": int(sum(admitted)),
                        "total": len(admitted),
                        "recall": float(np.mean(admitted)),
                        "conservative_upper_tie_rank": True,
                    }
                passed = (
                    pool_recalls["20000"]["admitted"] == 20
                    and pool_recalls["10000"]["admitted"] >= 19
                )
                dtype_pass[(dimensions, dtype)] &= passed
                target_proposal = case["target_scores"][str(dimensions)][dtype]
                route_correlations = {}
                for route_index, route in enumerate(ROUTES):
                    exact = [
                        float(target["total_distance"])
                        if route == "composite"
                        else float(target["component_distances"][route])
                        for target in case["targets"]
                    ]
                    correlation = _spearman(target_proposal[route_index], exact)
                    route_correlations[route] = correlation
                    if correlation is not None:
                        correlations.setdefault(f"{dimensions}:{dtype}:{route}", []).append(correlation)
                layouts[str(dimensions)][dtype] = {
                    "pool_recalls": pool_recalls,
                    "target_ranks": target_rows,
                    "target_exact_spearman": route_correlations,
                    "passed": passed,
                }
        exact_rows = int(authority["certificate"]["eligible_candidates"])
        observed_rows = int(checkpoint["eligible_rows"][case_index])
        row_accounting = observed_rows == exact_rows
        targets_seen_once = checkpoint["target_seen"][case_index] == [1] * 20
        for dimensions in LAYOUTS:
            layout_pass[dimensions] &= row_accounting and targets_seen_once
            for dtype in DTYPES:
                dtype_pass[(dimensions, dtype)] &= row_accounting and targets_seen_once
        case_rows.append({
            "query_episode_id": authority["query_episode_id"],
            "authority_digest": authority["authority_digest"],
            "symbol": authority["query"]["symbol"],
            "cutoff": authority["query"]["cutoff"],
            "eligible_rows": observed_rows,
            "authority_eligible_rows": exact_rows,
            "row_accounting_matches": row_accounting,
            "targets_seen_once": targets_seen_once,
            "maximum_authority_total_delta": case["maximum_authority_total_delta"],
            "maximum_authority_component_delta": case["maximum_authority_component_delta"],
            "authority_practical_tolerance": PRACTICAL_AUTHORITY_TOLERANCE,
            "layouts": layouts,
        })
    is_full = len(checkpoint["symbols"]) == universe_count
    for dimensions in LAYOUTS:
        layout_pass[dimensions] &= synthetic["all_passed"] and is_full
        for dtype in DTYPES:
            dtype_pass[(dimensions, dtype)] &= synthetic["all_passed"] and is_full
        dtype_pass[(dimensions, "float16")] &= (
            checkpoint["quantization_overflow_rows"][str(dimensions)] == 0
        )
    selected = next(
        (dimensions for dimensions in sorted(LAYOUTS)
         if dtype_pass[(dimensions, "float32")]), None,
    )
    selected_dtype = None
    if selected is not None:
        selected_dtype = (
            "float16" if dtype_pass[(selected, "float16")] else "float32"
        )
    quantization_sensitivity = {}
    for dimensions in sorted(LAYOUTS):
        maximum_rank_delta = 0
        admission_disagreements = {str(pool): 0 for pool in PROPOSAL_POOLS}
        for case in case_rows:
            native_rows = case["layouts"][str(dimensions)]["float32"]["target_ranks"]
            quantized_rows = case["layouts"][str(dimensions)]["float16"]["target_ranks"]
            for native, quantized in zip(native_rows, quantized_rows):
                maximum_rank_delta = max(
                    maximum_rank_delta,
                    abs(native["upper_composite_rank"] - quantized["upper_composite_rank"]),
                    *(abs(
                        native["upper_route_ranks"][route]
                        - quantized["upper_route_ranks"][route]
                    ) for route in PROPOSAL_ROUTES),
                )
                for pool in PROPOSAL_POOLS:
                    native_admitted = proposal_v2_route_admitted(
                        native["upper_route_ranks"], native["upper_composite_rank"], pool,
                    )
                    quantized_admitted = proposal_v2_route_admitted(
                        quantized["upper_route_ranks"],
                        quantized["upper_composite_rank"], pool,
                    )
                    admission_disagreements[str(pool)] += int(
                        native_admitted != quantized_admitted
                    )
        quantization_sensitivity[str(dimensions)] = {
            "maximum_conservative_rank_delta": maximum_rank_delta,
            "route_union_admission_disagreements": admission_disagreements,
            "targets_compared": len(case_rows) * 20,
        }
    deterministic = {
        "schema_version": SCHEMA_VERSION,
        "contract_digest": checkpoint["contract_digest"],
        "authority_cases": case_rows,
        "authority_case_count": len(case_rows),
        "universe_symbols": universe_count,
        "evaluated_symbols": len(checkpoint["symbols"]),
        "is_full_universe": is_full,
        "stride": checkpoint["stride"],
        "source_scope_digest": stable_hash(checkpoint["prefixes"]),
        "source_prefix_count": len(checkpoint["prefixes"]),
        "source_prefixes": checkpoint["prefixes"],
        "synthetic_gate": synthetic,
        "layout_float32_pass": {
            str(dimensions): dtype_pass[(dimensions, "float32")]
            for dimensions in sorted(LAYOUTS)
        },
        "layout_float16_pass": {
            str(dimensions): dtype_pass[(dimensions, "float16")]
            for dimensions in sorted(LAYOUTS)
        },
        "mean_target_exact_spearman": {
            key: float(np.mean(values)) for key, values in sorted(correlations.items())
        },
        "storage_projection": {
            str(dimensions): {
                dtype: {
                    "signature_bytes": proposal_v2_storage_bytes(dimensions, dtype),
                    "projected_3_82m_gib": (
                        proposal_v2_storage_bytes(dimensions, dtype)
                        * 3_820_000 / 1024 ** 3
                    ),
                }
                for dtype in DTYPES
            }
            for dimensions in sorted(LAYOUTS)
        },
        "quantization_sensitivity": quantization_sensitivity,
        "quantization_overflow_rows": checkpoint["quantization_overflow_rows"],
        "quantization_maximum_absolute_value": (
            checkpoint["quantization_maximum_absolute_value"]
        ),
        "quantization_overflow_examples": checkpoint["quantization_overflow_examples"],
        "selected_dimensions": selected,
        "selected_dtype": selected_dtype,
        "selected_signature_bytes": (
            proposal_v2_storage_bytes(selected, selected_dtype)
            if selected is not None and selected_dtype is not None else None
        ),
        "selection_rule": (
            "smallest layout with conservative tie-aware 20/20 at pool 20000, "
            ">=19/20 at pool 10000 in every authority, and perfect synthetic retrieval"
        ),
        "all_cases_passed": selected is not None,
        "real_forward_outcomes_accessed": False,
    }
    return {
        **deterministic,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "projection_seconds": checkpoint["projection_seconds"],
        "scoring_seconds": checkpoint["scoring_seconds"],
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "end_to_end_rows_per_second": (
            max(checkpoint["eligible_rows"]) / elapsed_seconds
            if elapsed_seconds else None
        ),
        "result_digest": stable_hash(deterministic),
    }


def _write_html(path: Path, payload: dict[str, Any]) -> None:
    summary_rows = "".join(
        f"<tr><td>{escape(case['symbol'])}</td><td>{escape(case['cutoff'])}</td>"
        f"<td>{case['eligible_rows']:,}</td><td>{'PASS' if case['row_accounting_matches'] else 'FAIL'}</td>"
        + "".join(
            f"<td>{case['layouts'][str(dimensions)]['float32']['pool_recalls']['10000']['admitted']}/20</td>"
            f"<td>{case['layouts'][str(dimensions)]['float32']['pool_recalls']['20000']['admitted']}/20</td>"
            for dimensions in sorted(LAYOUTS)
        ) + "</tr>"
        for case in payload["authority_cases"]
    )
    columns = "".join(
        f"<th>{dimensions} @10k</th><th>{dimensions} @20k</th>"
        for dimensions in sorted(LAYOUTS)
    )
    headline = "PASS" if payload["all_cases_passed"] else (
        "SAMPLE" if not payload["is_full_universe"] else "FAIL"
    )
    metadata = {key: value for key, value in payload.items() if key not in {
        "authority_cases", "synthetic_gate", "source_prefixes",
    }}
    path.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>M04R-05 proposal v2 gate</title><style>body{{font-family:system-ui;max-width:1500px;margin:2rem auto;line-height:1.45}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.45rem;border-bottom:1px solid #ddd;text-align:left}}code,pre{{overflow-wrap:anywhere;white-space:pre-wrap}}.pass{{color:#075}}.fail{{color:#a20}}</style></head><body><h1>M04R-05 proposal v2 authority gate: {headline}</h1><p>This is {'a bounded development sample, not a selectable result' if not payload['is_full_universe'] else 'the full frozen 12-authority universe evaluation'}. Ranks use the conservative end of every exact-tie interval.</p><p>Selected layout: <b>{payload['selected_dimensions']}</b>; dtype: <b>{payload['selected_dtype']}</b>; evidence digest <code>{payload['result_digest']}</code>.</p><pre>{escape(json.dumps(metadata, indent=2, sort_keys=True))}</pre><table><thead><tr><th>Query</th><th>Cutoff</th><th>Rows</th><th>Accounting</th>{columns}</tr></thead><tbody>{summary_rows}</tbody></table></body></html>""")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--symbols", type=int, default=64,
                        help="deterministic base sample; 0 means full universe")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=16)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-chunk", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    if (
        args.symbols < 0 or args.stride < 1 or args.checkpoint_every < 1
        or args.workers < 1 or args.worker_chunk < 1
    ):
        raise SystemExit(
            "symbols must be >=0; stride, checkpoint-every, workers and worker-chunk "
            "must be positive"
        )
    started = perf_counter()
    config = load_config(args.config)
    source = source_from_spec(config.datasets["nasdaq"])
    benchmark = source.load_benchmark()
    if benchmark is None:
        raise ValueError("NASDAQ proposal gate requires benchmark context")
    cases = _load_authorities(config, source)
    quality = pd.read_parquet(config.artifact_dir / "quality" / "nasdaq.parquet")
    tiers = dict(zip(quality.symbol.astype(str), quality.tier.astype(str)))
    universe = [
        key.source_symbol for key in source.instruments()
        if tiers.get(key.source_symbol, "A") in {"A", "B"}
    ]
    universe.sort(key=lambda symbol: sha256(
        f"m04r-proposal-v2-authorities:{symbol}".encode(),
    ).hexdigest())
    required = {
        str(target["symbol"]) for case in cases for target in case["targets"]
    }
    if args.symbols:
        base = universe[:args.symbols]
        selected = base + [symbol for symbol in universe if symbol in required - set(base)]
    else:
        selected = list(universe)
    contract_digest = proposal_v2_contract()["digest"]
    checkpoint_path = args.output.with_suffix(args.output.suffix + ".checkpoint")
    expected = _initialize_checkpoint(contract_digest, cases, selected, args)
    maximum_latest = max(case["latest"] for case in cases)
    if checkpoint_path.exists() and not args.restart:
        checkpoint = json.loads(checkpoint_path.read_text())
        _validate_checkpoint(checkpoint, expected, source, maximum_latest)
    else:
        checkpoint = expected
        _write_json(checkpoint_path, checkpoint)
    completed = set(checkpoint["processed_symbols"])
    remaining = [symbol for symbol in selected if symbol not in completed]
    chunks = [
        remaining[first:first + args.worker_chunk]
        for first in range(0, len(remaining), args.worker_chunk)
    ]
    global _WORKER_SOURCE, _WORKER_BENCHMARK, _WORKER_CASES
    global _WORKER_STRIDE, _WORKER_MAXIMUM_LATEST
    _WORKER_SOURCE = source
    _WORKER_BENCHMARK = benchmark
    _WORKER_CASES = cases
    _WORKER_STRIDE = args.stride
    _WORKER_MAXIMUM_LATEST = maximum_latest
    if args.workers == 1:
        results = map(_worker_chunk, chunks)
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("fork"),
        )
        results = executor.map(_worker_chunk, chunks)
    try:
        newly_completed = 0
        next_checkpoint = args.checkpoint_every
        for chunk, delta in zip(chunks, results):
            _merge_delta(checkpoint, delta)
            newly_completed += len(chunk)
            if newly_completed >= next_checkpoint:
                _write_json(checkpoint_path, checkpoint)
                print(json.dumps({
                    "processed": len(checkpoint["processed_symbols"]),
                    "total": len(selected), "last_symbol": chunk[-1],
                    "workers": args.workers,
                    "elapsed_seconds": perf_counter() - started,
                }), flush=True)
                next_checkpoint += args.checkpoint_every
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    _write_json(checkpoint_path, checkpoint)
    synthetic = _synthetic_gate()
    payload = _summarize(
        checkpoint, cases, len(universe), synthetic, perf_counter() - started,
    )
    _write_json(args.output, payload)
    _write_html(args.output.with_suffix(".html"), payload)
    print(json.dumps({
        "output": str(args.output), "is_full_universe": payload["is_full_universe"],
        "selected_dimensions": payload["selected_dimensions"],
        "selected_dtype": payload["selected_dtype"],
        "result_digest": payload["result_digest"],
        "elapsed_seconds": payload["elapsed_seconds"],
    }, indent=2))
    return 0 if payload["all_cases_passed"] else (0 if not payload["is_full_universe"] else 2)


if __name__ == "__main__":
    raise SystemExit(main())
