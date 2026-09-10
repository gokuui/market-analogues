"""R1-A outcome-blind exposure and random-priority concentration audit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from multiprocessing import get_context
import json
import math
import os
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np

from experiments.m04r import m04r14_shadow_run as shadow
from market_analogues.adequacy import (
    NullCandidate, concentration_metrics, nearest_rank, random_priority_selection,
)
from market_analogues.packed_bound_store import decode_episode_id, load_packed_generation
from market_analogues.types import stable_hash


SCHEMA = "m04r14-r1a-exposure-audit-v1"
PREREGISTRATION = Path("experiments/m04r/m04r14_r1a_exposure_audit_v2_preregistered.json")
OUTPUT = Path("config/data/analogues/m04r14/r1a-exposure-audit-v2")
REGISTRY = Path("config/data/analogues/m04r14/nasdaq-shadow-denominator-v1/query-registry.json")
REGISTRY_SEAL = Path("config/data/analogues/m04r14/nasdaq-shadow-denominator-v1/SEALED.json")
CASES = Path("config/data/analogues/m04r14/nasdaq-shadow-snapshot-v1/cases")
SEMANTIC_VERIFICATION = Path(
    "config/data/analogues/m04r14/nasdaq-shadow-interrupted-verification-v1/VERIFIED.json"
)
PACKED_DURABLE = Path("config/data/analogues/poc/m04r/packed-bound-full/store")
PACKED_RESULT = Path("config/data/analogues/poc/m04r/packed-bound-full/packed-bound-full.json")
PACKED_RESIDENT = Path(
    "/dev/shm/market-analogues/m04r11-candidate-v2/"
    "9fc6ae0ec4451133d8006897162f3443803fd30a3c44b78e80a476fbb18bb483/store"
)


class AdequacyAuditError(RuntimeError):
    pass


@dataclass(frozen=True)
class Query:
    episode_id: str
    symbol: str
    symbol_id: int
    start_ns: int
    latest_ns: int
    eligible_count: int


@dataclass(frozen=True)
class Metadata:
    episode_ids: np.ndarray
    cutoffs: np.ndarray
    starts: np.ndarray
    stops: np.ndarray
    symbols: tuple[str, ...]


def _load(path: Path) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise AdequacyAuditError(f"duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result
    value = json.loads(
        path.read_bytes(), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            AdequacyAuditError(f"non-finite JSON value {token}: {path}")
        ),
    )
    if not isinstance(value, dict):
        raise AdequacyAuditError(f"JSON object required: {path}")
    return value


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _verify_digest(value: Mapping[str, Any], field: str, omitted: set[str]) -> bool:
    return value.get(field) == stable_hash({
        key: item for key, item in value.items() if key not in omitted
    })


def _extract_metadata(
    store: Path, generation: str, *, expected_provenance_digest: str | None = None,
) -> tuple[Metadata, dict[str, Any]]:
    loaded = load_packed_generation(
        store, generation, expected_provenance_digest=expected_provenance_digest,
        verify_content=True, validate_records=True,
    )
    manifest = loaded.manifest
    if manifest.get("real_forward_outcomes_accessed") is not False:
        raise AdequacyAuditError("packed source is not outcome blind")
    main, overflow = loaded.rows, loaded.overflow
    total = len(main) + len(overflow)
    compact = np.empty(total, dtype=[
        ("episode_id", "V12"), ("cutoff", "<i8"), ("symbol", "<u4"),
    ])
    cursor = 0
    for source in (main, overflow):
        for first in range(0, len(source), 1 << 17):
            block = source[first:first + (1 << 17)]
            stop = cursor + len(block)
            compact["episode_id"][cursor:stop] = block["episode_id"]
            compact["cutoff"][cursor:stop] = block["cutoff_ns"]
            compact["symbol"][cursor:stop] = block["symbol_id"]
            cursor = stop
    order = np.lexsort((compact["episode_id"], compact["cutoff"], compact["symbol"]))
    compact = compact[order]
    if len(np.unique(compact["episode_id"])) != total:
        raise AdequacyAuditError("packed episode identifiers are not unique")
    symbol_count = len(manifest["symbols"])
    counts = np.bincount(compact["symbol"], minlength=symbol_count)
    stops = np.cumsum(counts, dtype=np.int64)
    starts = stops - counts
    for symbol_id, (first, stop) in enumerate(zip(starts, stops, strict=True)):
        if stop > first and not np.all(compact["symbol"][first:stop] == symbol_id):
            raise AdequacyAuditError("packed symbol ordering differs")
        if stop - first > 1 and np.any(np.diff(compact["cutoff"][first:stop]) <= 0):
            raise AdequacyAuditError("packed symbol cutoffs are not strictly ordered")
    return Metadata(
        compact["episode_id"], compact["cutoff"], starts, stops,
        loaded.symbols,
    ), manifest


def _pool(metadata: Metadata, latest_ns: int) -> tuple[np.ndarray, int]:
    counts = np.empty(len(metadata.symbols), dtype=np.int64)
    for symbol_id, (first, stop) in enumerate(zip(metadata.starts, metadata.stops, strict=True)):
        counts[symbol_id] = np.searchsorted(
            metadata.cutoffs[first:stop], latest_ns, side="right",
        )
    return np.cumsum(counts, dtype=np.int64), int(counts.sum())


def _query_mapping(
    metadata: Metadata, pool_cumulative: np.ndarray, pool_total: int, query: Query,
):
    first = int(metadata.starts[query.symbol_id])
    before = int(pool_cumulative[query.symbol_id - 1]) if query.symbol_id else 0
    base_own = int(pool_cumulative[query.symbol_id]) - before
    allowed_own = int(np.searchsorted(
        metadata.cutoffs[first:first + base_own], query.start_ns, side="left",
    ))
    removed = base_own - allowed_own
    eligible = pool_total - removed
    if eligible != query.eligible_count:
        raise AdequacyAuditError(
            f"risk-set count differs for {query.episode_id}: {eligible} != {query.eligible_count}"
        )

    def candidate_at(index: int) -> NullCandidate:
        adjusted = index + removed if index >= before + allowed_own else index
        symbol_id = int(np.searchsorted(pool_cumulative, adjusted, side="right"))
        symbol_before = int(pool_cumulative[symbol_id - 1]) if symbol_id else 0
        local = adjusted - symbol_before
        global_index = int(metadata.starts[symbol_id]) + local
        # Full A/B packs contain every 252-session window at five-session stride.
        # Session coordinates reproduce interval overlap exactly without calendar guesses.
        return NullCandidate(symbol_id, global_index, local * 5, local * 5 + 251)

    return eligible, candidate_at


def _eligible_symbol_count(
    metadata: Metadata, pool_cumulative: np.ndarray, query: Query, symbol_id: int,
) -> int:
    before = int(pool_cumulative[symbol_id - 1]) if symbol_id else 0
    count = int(pool_cumulative[symbol_id]) - before
    if symbol_id == query.symbol_id:
        first = int(metadata.starts[symbol_id])
        count = int(np.searchsorted(
            metadata.cutoffs[first:first + count], query.start_ns, side="left",
        ))
    return count


_WORK: dict[str, Any] = {}


def _simulate(replicates: tuple[int, ...]) -> dict[str, Any]:
    metadata: Metadata = _WORK["metadata"]
    queries: tuple[Query, ...] = _WORK["queries"]
    pools: dict[int, tuple[np.ndarray, int]] = _WORK["pools"]
    observed_positions: dict[int, int] = _WORK["observed_positions"]
    seed = int(_WORK["seed"])
    entity_hits = np.zeros(len(observed_positions), dtype=np.uint32)
    symbol_hits = np.zeros(len(metadata.symbols), dtype=np.uint32)
    rows: list[dict[str, Any]] = []
    for replicate in replicates:
        rng = np.random.default_rng(np.random.SeedSequence([seed, replicate]))
        episode_counts: Counter[int] = Counter()
        symbol_counts: Counter[int] = Counter()
        repeats = twice = thrice = 0
        for query in queries:
            cumulative, total = pools[query.latest_ns]
            eligible, candidate_at = _query_mapping(metadata, cumulative, total, query)
            selected = random_priority_selection(
                rng=rng, eligible_count=eligible, candidate_at=candidate_at,
                top_k=20, max_per_symbol=3,
            )
            local = Counter(item.symbol_id for item in selected)
            repeats += any(value > 1 for value in local.values())
            twice += sum(value == 2 for value in local.values())
            thrice += sum(value == 3 for value in local.values())
            for item in selected:
                episode_counts[item.ordinal] += 1
                symbol_counts[item.symbol_id] += 1
                position = observed_positions.get(item.ordinal)
                if position is not None:
                    entity_hits[position] += 1
                symbol_hits[item.symbol_id] += 1
        pair_intersections = sum(value * (value - 1) // 2 for value in episode_counts.values())
        metrics = concentration_metrics(
            list(episode_counts.values()), list(symbol_counts.values()),
            episode_population=len(metadata.episode_ids),
            symbol_population=len(metadata.symbols),
        )
        metrics.update({
            "replicate": replicate,
            "query_any_repeated_symbol_fraction": repeats / len(queries),
            "repeated_symbol_twice_per_query": twice / len(queries),
            "repeated_symbol_thrice_per_query": thrice / len(queries),
            "mean_pairwise_query_episode_overlap": (
                pair_intersections / (len(queries) * (len(queries) - 1) / 2)
            ),
        })
        rows.append(metrics)
    return {"metrics": rows, "entity_hits": entity_hits, "symbol_hits": symbol_hits}


def _actual(repository: Path, metadata: Metadata, registry: Mapping[str, Any]):
    rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    queries: list[Query] = []
    episode_counts: Counter[str] = Counter()
    episode_meta: dict[str, tuple[str, int, int]] = {}
    symbol_counts: Counter[str] = Counter()
    case_manifest: list[dict[str, Any]] = []
    registry_by_id = {str(row["episode_id"]): row for row in registry["cases_data"]}
    paths = sorted((repository / CASES).glob("*.json"))
    if len(paths) != len(registry_by_id):
        raise AdequacyAuditError("case inventory differs from registry")
    symbol_ids = {symbol: index for index, symbol in enumerate(metadata.symbols)}
    for path in paths:
        case = _load(path)
        query_id = str(case.get("query_episode_id"))
        registered = registry_by_id.get(query_id)
        if registered is None or not all((
            case.get("gate_passed") is True,
            case.get("real_forward_outcomes_accessed") is False,
            case.get("result_digest") == shadow._deterministic_case_digest(case),
            case.get("checkpoint_integrity_digest") == shadow._integrity_digest(case),
            case.get("registry_case_id") == registered["case_id"],
            case.get("latest_eligible_cutoff") == registered["latest_eligible_cutoff"],
            isinstance(case.get("matches"), list) and len(case["matches"]) == 20,
        )):
            raise AdequacyAuditError(f"invalid sealed case: {path.name}")
        distances = [float(item["total_distance"]) for item in case["matches"]]
        identities = [str(item["episode_id"]) for item in case["matches"]]
        if any(not math.isfinite(value) or value < 0 for value in distances) \
                or list(zip(distances, identities)) != sorted(zip(distances, identities)) \
                or len(set(identities)) != 20:
            raise AdequacyAuditError(f"invalid match ordering: {path.name}")
        per_symbol = Counter(str(item["symbol"]) for item in case["matches"])
        if max(per_symbol.values()) > 3:
            raise AdequacyAuditError(f"symbol cap violated: {path.name}")
        symbol = str(case["query_symbol"])
        if symbol not in symbol_ids:
            raise AdequacyAuditError(f"query symbol absent from pack: {symbol}")
        eligible_count = int(case["certificate"]["eligible_candidates"])
        queries.append(Query(
            query_id, symbol, symbol_ids[symbol],
            int(np.datetime64(case["query_start"]).astype("datetime64[ns]").astype(np.int64)),
            int(np.datetime64(case["latest_eligible_cutoff"]).astype("datetime64[ns]").astype(np.int64)),
            eligible_count,
        ))
        repeated = [value for value in per_symbol.values() if value > 1]
        query_rows.append({
            "query_episode_id": query_id, "query_symbol": symbol,
            "distance_rank_1": distances[0], "distance_rank_20": distances[-1],
            "distance_rank20_rank1_ratio": (
                distances[-1] / distances[0] if distances[0] > 0 else None
            ),
            "distinct_symbols": len(per_symbol),
            "symbols_repeated_twice": sum(value == 2 for value in repeated),
            "symbols_repeated_thrice": sum(value == 3 for value in repeated),
        })
        for rank, item in enumerate(case["matches"], 1):
            episode_id = str(item["episode_id"]); match_symbol = str(item["symbol"])
            cutoff_ns = int(np.datetime64(item["cutoff"]).astype("datetime64[ns]").astype(np.int64))
            episode_counts[episode_id] += 1; symbol_counts[match_symbol] += 1
            episode_meta[episode_id] = (match_symbol, cutoff_ns, rank)
            rows.append({"query_episode_id": query_id, "episode_id": episode_id, "rank": rank})
        case_manifest.append({
            "path": path.relative_to(repository / CASES.parent).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        })
    queries.sort(key=lambda item: item.episode_id)
    query_rows.sort(key=lambda item: item["query_episode_id"])
    pair_intersections = sum(value * (value - 1) // 2 for value in episode_counts.values())
    metrics = concentration_metrics(
        list(episode_counts.values()), list(symbol_counts.values()),
        episode_population=len(metadata.episode_ids),
        symbol_population=len(metadata.symbols),
    )
    metrics.update({
        "query_any_repeated_symbol_fraction": sum(
            row["distinct_symbols"] < 20 for row in query_rows
        ) / len(query_rows),
        "repeated_symbol_twice_per_query": sum(
            row["symbols_repeated_twice"] for row in query_rows
        ) / len(query_rows),
        "repeated_symbol_thrice_per_query": sum(
            row["symbols_repeated_thrice"] for row in query_rows
        ) / len(query_rows),
        "mean_pairwise_query_episode_overlap": (
            pair_intersections / (len(query_rows) * (len(query_rows) - 1) / 2)
        ),
    })
    return (
        tuple(queries), query_rows, rows, episode_counts, episode_meta,
        symbol_counts, metrics, case_manifest,
    )


def _locate_observed(metadata: Metadata, episode_meta: Mapping[str, tuple[str, int, int]]):
    symbol_ids = {value: index for index, value in enumerate(metadata.symbols)}
    located: dict[str, int] = {}
    for episode_id, (symbol, cutoff, _) in episode_meta.items():
        symbol_id = symbol_ids[symbol]
        first, stop = int(metadata.starts[symbol_id]), int(metadata.stops[symbol_id])
        local = int(np.searchsorted(metadata.cutoffs[first:stop], cutoff))
        index = first + local
        if index >= stop or int(metadata.cutoffs[index]) != cutoff \
                or decode_episode_id(metadata.episode_ids[index]) != episode_id:
            raise AdequacyAuditError(f"selected episode absent from packed universe: {episode_id}")
        located[episode_id] = index
    return located


def _retrieval_identity(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "query_episode_id": str(row["query_episode_id"]),
        "episode_id": str(row["episode_id"]), "rank": int(row["rank"]),
    } for row in rows]


def _summaries(
    actual: Mapping[str, float | int], null: Sequence[Mapping[str, Any]],
    directions: Mapping[str, str],
):
    output: dict[str, Any] = {}
    for key, observed in actual.items():
        values = [float(row[key]) for row in null]
        direction = directions[key]
        if direction not in {"higher_is_more_concentrated", "lower_is_more_concentrated"}:
            raise AdequacyAuditError(f"invalid metric direction: {key}")
        tail = (
            sum(value >= float(observed) for value in values)
            if direction == "higher_is_more_concentrated"
            else sum(value <= float(observed) for value in values)
        )
        output[key] = {
            "observed": observed, "null_mean": float(np.mean(values)),
            "null_p05": nearest_rank(values, .05), "null_p50": nearest_rank(values, .50),
            "null_p95": nearest_rank(values, .95), "null_p99": nearest_rank(values, .99),
            "concentration_direction": direction,
            "concentration_tail_monte_carlo_p": (1 + tail) / (len(values) + 1),
        }
    return output


def _html(result: Mapping[str, Any]) -> str:
    rows = "".join(
        f"<tr><td>{key}</td><td>{value['observed']:.6g}</td>"
        f"<td>{value['null_p50']:.6g}</td><td>{value['null_p95']:.6g}</td>"
        f"<td>{value['concentration_tail_monte_carlo_p']:.6g}</td></tr>"
        for key, value in result["comparison"].items()
    )
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'>
<title>R1-A analogue exposure audit</title><style>body{{font-family:system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #bbb;padding:.4rem;text-align:right}}td:first-child,th:first-child{{text-align:left}}.boundary{{background:#fff3cd;padding:1rem}}</style></head><body>
<h1>R1-A analogue exposure and concentration audit</h1>
<p><b>Status:</b> {result['status']}. This is outcome-blind and leaves all 65,400 retrieved links unchanged.</p>
<p>{result['inventory']['queries']:,} queries; {result['inventory']['candidate_episodes']:,} candidate episodes; {result['null']['replicates']:,} independent-query random-priority replicates.</p>
<table><thead><tr><th>Metric</th><th>Observed</th><th>Null median</th><th>Null p95</th><th>Concentration-tail p</th></tr></thead><tbody>{rows}</tbody></table>
<p class='boundary'><b>Boundary:</b> this tests whether retrieval concentration exceeds an iid random-priority risk-set null while preserving the causal pool, cap-three rule, and interval overlap. It does not establish chart adequacy or predictive value. R1-B must test matched query nulls, reciprocity and perturbation/view/cutoff stability before labels are allowed.</p>
</body></html>"""


def execute(repository: Path, *, workers: int | None = None) -> dict[str, Any]:
    started = perf_counter(); repository = repository.resolve()
    prereg = _load(repository / PREREGISTRATION)
    prereg_state = {key: value for key, value in prereg.items() if key != "preregistration_digest"}
    if prereg.get("preregistration_digest") != stable_hash(prereg_state):
        raise AdequacyAuditError("preregistration digest differs")
    output = repository / Path(str(prereg["execution"]["output_root"]))
    if output.exists() or output.is_symlink():
        raise AdequacyAuditError(f"create-only output exists: {output}")
    generation = str(prereg["inputs"]["packed_generation_id"])
    resident = PACKED_RESIDENT
    if not (resident / "generations" / generation / "manifest.json").is_file():
        resident = repository / PACKED_DURABLE
    metadata, manifest = _extract_metadata(
        resident, generation,
        expected_provenance_digest=str(prereg["inputs"]["packed_provenance_digest"]),
    )
    registry = _load(repository / REGISTRY); seal = _load(repository / REGISTRY_SEAL)
    verified = _load(repository / SEMANTIC_VERIFICATION)
    packed_result = _load(repository / PACKED_RESULT)
    verified_state = {key: value for key, value in verified.items() if key not in {"result_digest", "created_at"}}
    registry_state = {key: value for key, value in registry.items() if key != "registry_digest"}
    seal_state = {key: value for key, value in seal.items() if key not in {"seal_digest", "created_at"}}
    for relative, expected in prereg["inputs"]["file_sha256"].items():
        if _sha(repository / relative) != expected:
            raise AdequacyAuditError(f"preregistered input changed: {relative}")
    for relative, expected in prereg["runtime_files"].items():
        if _sha(repository / relative) != expected:
            raise AdequacyAuditError(f"preregistered runtime changed: {relative}")
    if not all((
        verified.get("result_digest") == stable_hash(verified_state),
        verified.get("semantic_passed") is True,
        verified.get("real_forward_outcomes_accessed") is False,
        verified.get("verified_cases") == 3270,
        verified.get("verified_matches") == 65400,
        verified.get("source_content_digest") == prereg["inputs"]["packed_content_digest"],
        verified.get("registry_digest") == registry.get("registry_digest"),
        registry.get("registry_digest") == stable_hash(registry_state),
        registry.get("registry_digest") == seal.get("registry_digest"),
        seal.get("seal_digest") == stable_hash(seal_state),
        registry.get("real_forward_outcomes_accessed") is False,
        packed_result.get("result_digest") == prereg["inputs"]["packed_result_digest"],
        packed_result.get("gate_passed") is True,
        packed_result.get("generation_id") == generation,
        packed_result.get("eligible_rows") == len(metadata.episode_ids),
        packed_result.get("real_forward_outcomes_accessed") is False,
    )):
        raise AdequacyAuditError("upstream semantic authority differs")
    queries, query_rows, retrieval_rows, episode_counts, episode_meta, symbol_counts, actual, case_manifest = _actual(
        repository, metadata, registry,
    )
    if stable_hash(case_manifest) != verified.get("case_manifest_digest"):
        raise AdequacyAuditError("case tree differs from upstream semantic verification")
    source_prefixes = manifest.get("provenance", {}).get("source_prefixes", {})
    for symbol_id, symbol in enumerate(metadata.symbols):
        prefix_rows = int(source_prefixes[symbol]["rows"])
        expected_rows = max((prefix_rows - 252) // 5 + 1, 0)
        if int(metadata.stops[symbol_id] - metadata.starts[symbol_id]) != expected_rows:
            raise AdequacyAuditError(f"stride/lookback row accounting differs: {symbol}")
    located = _locate_observed(metadata, episode_meta)
    unique_latest = sorted({query.latest_ns for query in queries})
    pools = {value: _pool(metadata, value) for value in unique_latest}
    for query in queries:
        _query_mapping(metadata, *pools[query.latest_ns], query)
    replicate_count = int(prereg["execution"]["null_replicates"])
    worker_count = workers or int(prereg["execution"]["workers"])
    worker_count = max(1, min(worker_count, replicate_count))
    observed_indices = sorted(located.values())
    observed_positions = {value: index for index, value in enumerate(observed_indices)}
    _WORK.update({
        "metadata": metadata, "queries": queries, "pools": pools,
        "observed_positions": observed_positions, "seed": int(prereg["execution"]["seed"]),
    })
    chunks = tuple(tuple(range(first, replicate_count, worker_count)) for first in range(worker_count))
    if worker_count == 1:
        parts = [_simulate(chunks[0])]
    else:
        with get_context("fork").Pool(worker_count) as pool:
            parts = pool.map(_simulate, chunks)
    null_rows = sorted(
        (row for part in parts for row in part["metrics"]), key=lambda row: row["replicate"],
    )
    entity_hits = sum((part["entity_hits"] for part in parts), np.zeros(len(observed_indices), dtype=np.uint64))
    null_symbol_hits = sum((part["symbol_hits"] for part in parts), np.zeros(len(metadata.symbols), dtype=np.uint64))
    if len(null_rows) != replicate_count:
        raise AdequacyAuditError("null replicate accounting differs")
    comparison = _summaries(actual, null_rows, prereg["metric_directions"])
    latest_values = np.asarray([query.latest_ns for query in queries], dtype=np.int64)
    query_by_symbol = {query.symbol: query for query in queries}
    entity_rows = []
    index_position = {value: index for index, value in enumerate(observed_indices)}
    for episode_id in sorted(episode_counts, key=lambda value: (-episode_counts[value], value)):
        symbol, cutoff, _ = episode_meta[episode_id]
        exposure = int(np.sum(latest_values >= cutoff))
        own = query_by_symbol.get(symbol)
        if own is not None and cutoff >= own.start_ns and cutoff <= own.latest_ns:
            exposure -= 1
        index = located[episode_id]
        expected = float(entity_hits[index_position[index]] / replicate_count)
        entity_rows.append({
            "episode_id": episode_id, "symbol": symbol,
            "cutoff": np.datetime_as_string(np.datetime64(cutoff, "ns")),
            "observed_count": episode_counts[episode_id], "eligible_query_exposure": exposure,
            "observed_per_eligible_query": episode_counts[episode_id] / exposure,
            "null_expected_count": expected,
            "observed_to_null_expected": episode_counts[episode_id] / expected if expected else None,
        })
    symbol_rows = []
    symbol_ids = {symbol: index for index, symbol in enumerate(metadata.symbols)}
    for symbol in sorted(symbol_counts, key=lambda value: (-symbol_counts[value], value)):
        symbol_id = symbol_ids[symbol]
        eligible_counts = [
            _eligible_symbol_count(metadata, pools[q.latest_ns][0], q, symbol_id)
            for q in queries
        ]
        eligible_queries = sum(value > 0 for value in eligible_counts)
        # Full-pack windows are five sessions apart and each spans 252 sessions;
        # ordinal separation >=51 is therefore the exact non-overlap threshold.
        selection_capacity = sum(
            min(3, 1 + (value - 1) // 51) if value else 0
            for value in eligible_counts
        )
        symbol_rows.append({
            "symbol": symbol, "observed_count": symbol_counts[symbol],
            "eligible_query_exposure": eligible_queries,
            "cap_overlap_selection_capacity": selection_capacity,
            "observed_per_capacity": symbol_counts[symbol] / selection_capacity,
            "null_expected_count": float(null_symbol_hits[symbol_id] / replicate_count),
        })
    identity = _retrieval_identity(retrieval_rows)
    finite_ratios = [
        float(row["distance_rank20_rank1_ratio"]) for row in query_rows
        if row["distance_rank20_rank1_ratio"] is not None
    ]
    gates = {
        "upstream_semantics_verified": True,
        "upstream_case_manifest_unchanged": stable_hash(case_manifest) == verified["case_manifest_digest"],
        "case_seals_reconstructed": True,
        "all_risk_set_counts_equal_certificates": True,
        "packed_main_and_overflow_included": (
            len(metadata.episode_ids) == int(manifest["row_count"]) +
            int(manifest["overflow_count"])
        ),
        "stride_5_lookback_252_accounting": True,
        "exact_cap_three_overlap_null": True,
        "retrieval_unchanged": True,
        "outcomes_excluded": True,
        "null_accounting_complete": len(null_rows) == replicate_count,
    }
    if not all(gates.values()):
        raise AdequacyAuditError("one or more audit gates failed")
    state: dict[str, Any] = {
        "schema_version": SCHEMA, "status": "diagnostic_only_r1a_complete",
        "passed": True, "production_promotion_authorized": False,
        "adequacy_labels_authorized": False, "predictive_claim_authorized": False,
        "real_forward_outcomes_accessed": False,
        "preregistration_digest": prereg["preregistration_digest"],
        "inputs": {
            "registry_digest": registry["registry_digest"],
            "semantic_verification_digest": verified["result_digest"],
            "packed_generation_id": generation,
            "packed_provenance_digest": manifest["provenance_digest"],
            "case_manifest_digest": stable_hash(case_manifest),
            "retrieval_identity_digest": stable_hash(identity),
        },
        "inventory": {
            "queries": len(queries), "retrieved_links": len(identity),
            "candidate_episodes": len(metadata.episode_ids),
            "candidate_symbols": len(metadata.symbols),
            "unique_latest_eligible_cutoffs": len(unique_latest),
        },
        "distance_geometry": {
            "rank1_zero_count": len(query_rows) - len(finite_ratios),
            "rank20_rank1_ratio_p10": float(np.quantile(finite_ratios, .1)),
            "rank20_rank1_ratio_p50": float(np.quantile(finite_ratios, .5)),
            "rank20_rank1_ratio_p90": float(np.quantile(finite_ratios, .9)),
        },
        "actual": actual, "comparison": comparison,
        "null": {
            "model": "independent per-query iid continuous random priorities over exact causal risk set, followed by production greedy cap-3 inclusive-overlap selection",
            "replicates": replicate_count, "seed": int(prereg["execution"]["seed"]),
            "metrics_digest": stable_hash(null_rows),
        },
        "most_recurrent_episodes_with_exposure": entity_rows[:100],
        "top_symbols": symbol_rows[:100],
        "gates": gates,
        "remaining_r1": [
            "matched causal query-distance nulls", "directed reciprocity registry",
            "leave-one-group/input-perturbation/nearby-cutoff stability",
            "synthetic positive/novel-path adequacy-label calibration",
        ],
    }
    result = {**state, "result_digest": stable_hash(state)}
    temp = output.parent / f".{output.name}.tmp-{os.getpid()}-{uuid4().hex}"
    try:
        temp.mkdir(parents=True)
        (temp / "NULL_REPLICATES.json").write_text(json.dumps(
            null_rows, indent=2, sort_keys=True, allow_nan=False,
        ) + "\n")
        (temp / "QUERY_DIAGNOSTICS.json").write_text(json.dumps(
            query_rows, indent=2, sort_keys=True, allow_nan=False,
        ) + "\n")
        result["artifacts"] = {
            "null_replicates_sha256": _sha(temp / "NULL_REPLICATES.json"),
            "query_diagnostics_sha256": _sha(temp / "QUERY_DIAGNOSTICS.json"),
        }
        result["elapsed_seconds"] = perf_counter() - started
        deterministic = {k: v for k, v in result.items() if k not in {"result_digest", "elapsed_seconds"}}
        result["result_digest"] = stable_hash(deterministic)
        (temp / "RESULT.json").write_text(json.dumps(
            result, indent=2, sort_keys=True, allow_nan=False,
        ) + "\n")
        (temp / "report.html").write_text(_html(result), encoding="utf-8")
        temp.rename(output)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True); raise
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--workers", type=int)
    args = parser.parse_args(argv)
    print(json.dumps(execute(args.repository, workers=args.workers), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
