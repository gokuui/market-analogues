from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import OrderedDict
from hashlib import sha256
from html import escape
import heapq
import json
import os
from pathlib import Path
import re
from time import perf_counter

import numpy as np
import pandas as pd

from .adapters import OHLCVSource
from .distance import (
    complete_representation_distance, representation_distance_lower_bound,
)
from .exact_batch import (
    EXACT_BATCH_VERSION, batch_representation_lower_bounds,
    sliding_exact_representations,
)
from .representation import represent
from .search import ScoredCandidate, eligible, latest_eligible_cutoff, select_scored
from .types import (
    AnalogueMatch, Episode, EpisodeKey, InstrumentKey, SearchQuery, stable_hash,
)


FRONTIER_SCHEMA_VERSION = 1


class FrontierError(ValueError):
    pass


@dataclass(frozen=True)
class FrontierShardMetadata:
    schema_version: int
    batch_version: str
    query_episode_id: str
    request_scope_digest: str
    dataset_id: str
    symbol: str
    lookback: int
    stride: int
    representation_version: str
    quality_tier: str
    source_fingerprint: str
    benchmark_fingerprint: str | None
    rows: int
    content_digest: str


@dataclass(frozen=True)
class LoadedFrontierShard:
    path: Path
    metadata: FrontierShardMetadata
    episode_ids: np.ndarray
    cutoffs_ns: np.ndarray
    lower_bounds: np.ndarray


@dataclass(frozen=True)
class FrontierBuildResult:
    passed: bool
    query_episode_id: str
    instruments_considered: int
    instruments_built: int
    instruments_reused: int
    quality_skipped: int
    eligible_rows: int
    failures: tuple[str, ...]
    manifest_path: Path
    manifest_digest: str
    seconds: float


@dataclass(frozen=True)
class ExhaustiveCertificate:
    query_episode_id: str
    manifest_digest: str
    eligible_candidates: int
    exact_evaluated: int
    safely_pruned: int
    stopped_early: bool
    stop_threshold: float
    next_lower_bound: float | None
    maximum_recomputed_bound_delta: float
    elapsed_seconds: float


@dataclass(frozen=True)
class ExhaustiveResult:
    matches: tuple[AnalogueMatch, ...]
    certificate: ExhaustiveCertificate


def write_exhaustive_report(
    build: FrontierBuildResult,
    resume: FrontierBuildResult,
    result: ExhaustiveResult,
    path: Path,
    *,
    passed: bool,
    failures: list[str],
    metrics: dict[str, object],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "".join(
        f"<tr><td>{rank}</td><td>{escape(str(match.episode_key.instrument))}</td>"
        f"<td>{escape(match.episode_key.cutoff.isoformat())}</td>"
        f"<td>{match.total_distance:.10g}</td></tr>"
        for rank, match in enumerate(result.matches, 1)
    )
    failure_items = "".join(f"<li>{escape(value)}</li>" for value in failures) or "<li>None</li>"
    status = "PASS" if passed else "FAIL"
    certificate = asdict(result.certificate)
    if not np.isfinite(float(certificate["stop_threshold"])):
        certificate["stop_threshold"] = None
    path.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exhaustive frontier verification</title><style>body{{font-family:system-ui,sans-serif;max-width:1400px;margin:2rem auto;background:#f5f7f8;color:#17202a}}header,section{{background:white;border:1px solid #ddd;border-radius:10px;padding:1.2rem;margin:1rem}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.5rem;border-bottom:1px solid #ddd}}</style></head><body><header><h1>T12-03 exhaustive frontier: {status}</h1><p>Atomic/resumable exact lower-bound shards are globally merged; candidates are reconstructed natively until the next bound cannot beat the complete constrained result.</p></header><section><h2>Metrics</h2><pre>{escape(json.dumps(metrics, indent=2))}</pre></section><section><h2>Matches</h2><table><thead><tr><th>Rank</th><th>Instrument</th><th>Cutoff</th><th>Distance</th></tr></thead><tbody>{rows}</tbody></table></section><section><h2>Failures</h2><ul>{failure_items}</ul></section><section><h2>Build provenance</h2><pre>{escape(json.dumps({"build": asdict(build), "resume": asdict(resume), "certificate": certificate}, indent=2, default=str))}</pre></section></body></html>""")
    return path


def _safe_name(symbol: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", symbol).strip("._") or "symbol"
    return f"{stem[:80]}-{sha256(symbol.encode()).hexdigest()[:10]}"


def frontier_shard_path(root: Path, query_id: str, key: InstrumentKey) -> Path:
    return root / query_id / "shards" / key.dataset_id / f"{_safe_name(key.source_symbol)}.npz"


def _content_digest(
    episode_ids: np.ndarray, cutoffs_ns: np.ndarray, lower_bounds: np.ndarray,
) -> str:
    digest = sha256()
    for episode_id in episode_ids.astype(str):
        digest.update(episode_id.encode())
        digest.update(b"\0")
    digest.update(np.asarray(cutoffs_ns, dtype="<i8").tobytes())
    digest.update(np.asarray(lower_bounds, dtype="<f8").tobytes())
    return digest.hexdigest()


def _write_frontier_shard(
    path: Path,
    metadata: FrontierShardMetadata,
    episode_ids: np.ndarray,
    cutoffs_ns: np.ndarray,
    lower_bounds: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata_json=np.asarray(json.dumps(asdict(metadata), sort_keys=True)),
            episode_ids=episode_ids.astype("U24"),
            cutoffs_ns=np.asarray(cutoffs_ns, dtype=np.int64),
            lower_bounds=np.asarray(lower_bounds, dtype=np.float64),
        )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_frontier_shard(path: Path) -> LoadedFrontierShard:
    try:
        with np.load(path, allow_pickle=False) as payload:
            required = {"metadata_json", "episode_ids", "cutoffs_ns", "lower_bounds"}
            missing = required.difference(payload.files)
            if missing:
                raise FrontierError(f"missing frontier arrays: {sorted(missing)}")
            metadata = FrontierShardMetadata(**json.loads(str(payload["metadata_json"].item())))
            episode_ids = np.asarray(payload["episode_ids"]).astype(str)
            cutoffs_ns = np.asarray(payload["cutoffs_ns"], dtype=np.int64)
            lower_bounds = np.asarray(payload["lower_bounds"], dtype=np.float64)
    except FrontierError:
        raise
    except Exception as exc:
        raise FrontierError(f"cannot load frontier shard {path}: {exc}") from exc
    if metadata.schema_version != FRONTIER_SCHEMA_VERSION:
        raise FrontierError(f"unsupported frontier schema {metadata.schema_version}")
    if metadata.batch_version != EXACT_BATCH_VERSION:
        raise FrontierError(f"unsupported exact batch version {metadata.batch_version}")
    if not (len(episode_ids) == len(cutoffs_ns) == len(lower_bounds) == metadata.rows):
        raise FrontierError("frontier row counts disagree")
    if len(set(episode_ids)) != len(episode_ids):
        raise FrontierError("frontier contains duplicate episode IDs")
    if not np.isfinite(lower_bounds).all() or (lower_bounds < 0).any():
        raise FrontierError("frontier contains invalid lower bounds")
    order = np.lexsort((episode_ids, lower_bounds))
    if not np.array_equal(order, np.arange(len(order))):
        raise FrontierError("frontier rows are not sorted by lower bound and episode ID")
    digest = _content_digest(episode_ids, cutoffs_ns, lower_bounds)
    if digest != metadata.content_digest:
        raise FrontierError("frontier content digest mismatch")
    return LoadedFrontierShard(path, metadata, episode_ids, cutoffs_ns, lower_bounds)


def _metadata_matches(actual: FrontierShardMetadata, expected: dict[str, object]) -> bool:
    return all(getattr(actual, key) == value for key, value in expected.items())


def _request_scope_digest(request: SearchQuery) -> str:
    return stable_hash({
        "search_datasets": request.search_datasets,
        "quality_tiers": request.quality_tiers,
        "cross_dataset": request.cross_dataset,
        "minimum_history_gap_bars": request.minimum_history_gap_bars,
    })


def _quality_digest(quality: pd.DataFrame) -> str:
    columns = [name for name in ("symbol", "tier", "issues") if name in quality]
    records = quality[columns].fillna("").sort_values("symbol").to_dict(orient="records")
    return stable_hash(records)


def build_exact_frontier(
    query: Episode,
    source: OHLCVSource,
    request: SearchQuery,
    quality: pd.DataFrame,
    output_root: Path,
    *,
    stride: int = 5,
    batch_size: int = 128,
    instrument_limit: int | None = None,
    rebuild_invalid: bool = False,
) -> FrontierBuildResult:
    if stride < 1 or batch_size < 1:
        raise ValueError("stride and batch size must be positive")
    started = perf_counter()
    query_representation = represent(query)
    request_digest = _request_scope_digest(request)
    quality_digest = _quality_digest(quality)
    benchmark = source.load_benchmark()
    benchmark_fingerprint = source.benchmark_fingerprint()
    qmap = {str(row.symbol): row for row in quality.itertuples(index=False)}
    instruments = sorted(source.instruments())
    if request.search_datasets:
        instruments = [key for key in instruments if key.dataset_id in request.search_datasets]
    if instrument_limit is not None:
        instruments = instruments[:instrument_limit]
    failures: list[str] = []
    records: list[dict[str, object]] = []
    built = reused = skipped = eligible_rows = 0
    latest = latest_eligible_cutoff(query, request.minimum_history_gap_bars)
    for key in instruments:
        record = qmap.get(key.source_symbol)
        tier = str(record.tier) if record is not None else "A"
        if tier not in request.quality_tiers or tier == "QUARANTINED":
            skipped += 1
            continue
        path = frontier_shard_path(output_root, query.key.id, key)
        expected = {
            "query_episode_id": query.key.id,
            "request_scope_digest": request_digest,
            "dataset_id": key.dataset_id,
            "symbol": key.source_symbol,
            "lookback": query.key.lookback,
            "stride": stride,
            "representation_version": query.key.representation_version,
            "quality_tier": tier,
            "source_fingerprint": source.fingerprint(key),
            "benchmark_fingerprint": benchmark_fingerprint,
        }
        if path.exists():
            try:
                loaded = load_frontier_shard(path)
                if not _metadata_matches(loaded.metadata, expected):
                    raise FrontierError("frontier provenance is stale")
            except FrontierError as exc:
                if not rebuild_invalid:
                    failures.append(f"{key}:{exc}")
                    continue
            else:
                reused += 1
                eligible_rows += loaded.metadata.rows
                records.append({**asdict(loaded.metadata), "path": str(path.relative_to(output_root))})
                continue
        try:
            bars = source.load(key)
            frame = bars[bars.timestamp <= latest].reset_index(drop=True)
            batch = sliding_exact_representations(
                frame, benchmark, lookback=query.key.lookback, stride=stride,
                batch_size=batch_size,
            )
            bounds = batch_representation_lower_bounds(
                query_representation, batch.representations,
            ).totals
            episode_ids: list[str] = []
            cutoffs: list[int] = []
            kept_bounds: list[float] = []
            query_timestamps = set(query.bars.timestamp.astype(str)) if key == query.key.instrument else set()
            for row, position in enumerate(batch.positions):
                window = frame.iloc[
                    int(position) - query.key.lookback + 1:int(position) + 1
                ]
                if query_timestamps and query_timestamps.intersection(window.timestamp.astype(str)):
                    continue
                cutoff = pd.Timestamp(frame.timestamp.iloc[int(position)])
                if cutoff >= query.key.cutoff:
                    continue
                episode_key = EpisodeKey(
                    key, cutoff, query.key.lookback, query.key.representation_version,
                )
                episode_ids.append(episode_key.id)
                cutoffs.append(cutoff.value)
                kept_bounds.append(float(bounds[row]))
            ids = np.asarray(episode_ids, dtype="U24")
            cutoff_ns = np.asarray(cutoffs, dtype=np.int64)
            lower = np.asarray(kept_bounds, dtype=np.float64)
            order = np.lexsort((ids, lower))
            ids, cutoff_ns, lower = ids[order], cutoff_ns[order], lower[order]
            metadata = FrontierShardMetadata(
                FRONTIER_SCHEMA_VERSION, EXACT_BATCH_VERSION,
                query.key.id, request_digest, key.dataset_id, key.source_symbol,
                query.key.lookback, stride, query.key.representation_version,
                tier, expected["source_fingerprint"], benchmark_fingerprint,
                len(ids), _content_digest(ids, cutoff_ns, lower),
            )
            _write_frontier_shard(path, metadata, ids, cutoff_ns, lower)
            loaded = load_frontier_shard(path)
            built += 1
            eligible_rows += metadata.rows
            records.append({**asdict(loaded.metadata), "path": str(path.relative_to(output_root))})
        except Exception as exc:
            failures.append(f"{key}:{type(exc).__name__}:{exc}")
    records.sort(key=lambda item: (str(item["dataset_id"]), str(item["symbol"])))
    digest = sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest_path = output_root / query.key.id / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": FRONTIER_SCHEMA_VERSION,
        "batch_version": EXACT_BATCH_VERSION,
        "query_episode_id": query.key.id,
        "query_source_fingerprint": source.fingerprint(query.key.instrument),
        "request_scope_digest": request_digest,
        "quality_digest": quality_digest,
        "query_instrument": str(query.key.instrument),
        "query_cutoff": query.key.cutoff.isoformat(),
        "lookback": query.key.lookback,
        "stride": stride,
        "representation_version": query.key.representation_version,
        "benchmark_fingerprint": benchmark_fingerprint,
        "eligible_rows": eligible_rows,
        "shards": records,
        "manifest_digest": digest,
        "failures": failures,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    temporary.replace(manifest_path)
    return FrontierBuildResult(
        bool(records) and not failures, query.key.id, len(instruments), built,
        reused, skipped, eligible_rows, tuple(failures), manifest_path, digest,
        perf_counter() - started,
    )


def _load_manifest(root: Path, query: Episode) -> tuple[dict[str, object], list[LoadedFrontierShard]]:
    path = root / query.key.id / "manifest.json"
    if not path.exists():
        raise FrontierError(f"missing frontier manifest {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("query_episode_id") != query.key.id:
        raise FrontierError("frontier query identity is stale")
    records = manifest.get("shards")
    if not isinstance(records, list) or manifest.get("failures"):
        raise FrontierError("frontier manifest is incomplete")
    digest = sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != manifest.get("manifest_digest"):
        raise FrontierError("frontier manifest digest mismatch")
    shards = []
    for record in records:
        shard = load_frontier_shard(root / str(record["path"]))
        if asdict(shard.metadata) != {key: value for key, value in record.items() if key != "path"}:
            raise FrontierError(f"frontier shard metadata changed: {shard.path}")
        shards.append(shard)
    if sum(shard.metadata.rows for shard in shards) != int(manifest.get("eligible_rows", -1)):
        raise FrontierError("frontier eligible-row accounting mismatch")
    return manifest, shards


def exhaustive_frontier_search(
    query: Episode,
    source: OHLCVSource,
    request: SearchQuery,
    frontier_root: Path,
    *,
    quality: pd.DataFrame,
    tolerance: float = 1e-12,
    frontier_batch_rows: int = 256,
    representation_cache_shards: int = 8,
) -> ExhaustiveResult:
    if frontier_batch_rows < 1 or representation_cache_shards < 1:
        raise ValueError("frontier batch rows and representation cache must be positive")
    started = perf_counter()
    manifest, shards = _load_manifest(frontier_root, query)
    if manifest.get("request_scope_digest") != _request_scope_digest(request):
        raise FrontierError("frontier request scope is stale")
    if manifest.get("quality_digest") != _quality_digest(quality):
        raise FrontierError("frontier quality provenance is stale")
    if manifest.get("query_source_fingerprint") != source.fingerprint(query.key.instrument):
        raise FrontierError("frontier query source fingerprint is stale")
    if manifest.get("benchmark_fingerprint") != source.benchmark_fingerprint():
        raise FrontierError("frontier benchmark fingerprint is stale")
    heap: list[tuple[float, str, int, int]] = []
    for shard_index, shard in enumerate(shards):
        if shard.metadata.source_fingerprint != source.fingerprint(
            InstrumentKey(shard.metadata.dataset_id, shard.metadata.symbol)
        ):
            raise FrontierError(f"frontier source fingerprint is stale: {shard.path}")
        if len(shard.lower_bounds):
            heapq.heappush(heap, (
                float(shard.lower_bounds[0]), str(shard.episode_ids[0]),
                shard_index, 0,
            ))
    query_representation = represent(query)
    scored: list[ScoredCandidate] = []
    threshold = float("inf")
    maximum_bound_delta = 0.0
    evaluated = 0
    stopped_early = False
    next_lower: float | None = None
    benchmark = source.load_benchmark()
    latest = latest_eligible_cutoff(query, request.minimum_history_gap_bars)
    cache: OrderedDict[int, tuple[pd.DataFrame, tuple[object, ...], dict[int, int]]] = OrderedDict()

    def materialized(shard_index: int):
        if shard_index in cache:
            cache.move_to_end(shard_index)
            return cache[shard_index]
        shard = shards[shard_index]
        key = InstrumentKey(shard.metadata.dataset_id, shard.metadata.symbol)
        bars = source.load(key)
        frame = bars[bars.timestamp <= latest].reset_index(drop=True)
        batch = sliding_exact_representations(
            frame, benchmark, lookback=shard.metadata.lookback,
            stride=shard.metadata.stride, batch_size=128,
        )
        cutoff_to_row = {
            int(pd.Timestamp(frame.timestamp.iloc[int(position)]).value): row
            for row, position in enumerate(batch.positions)
        }
        value = (frame, batch.representations, cutoff_to_row)
        cache[shard_index] = value
        cache.move_to_end(shard_index)
        while len(cache) > representation_cache_shards:
            cache.popitem(last=False)
        return value

    while heap:
        pending: list[tuple[float, str, int, int]] = []
        if np.isfinite(threshold):
            while (
                heap and heap[0][0] <= threshold
                and len(pending) < frontier_batch_rows
            ):
                item = heapq.heappop(heap)
                pending.append(item)
                shard = shards[item[2]]
                next_row = item[3] + 1
                if next_row < len(shard.lower_bounds):
                    heapq.heappush(heap, (
                        float(shard.lower_bounds[next_row]),
                        str(shard.episode_ids[next_row]), item[2], next_row,
                    ))
            if not pending:
                stopped_early = True
                next_lower = float(heap[0][0])
                break
        else:
            while heap and len(pending) < frontier_batch_rows:
                item = heapq.heappop(heap)
                pending.append(item)
                shard = shards[item[2]]
                next_row = item[3] + 1
                if next_row < len(shard.lower_bounds):
                    heapq.heappush(heap, (
                        float(shard.lower_bounds[next_row]),
                        str(shard.episode_ids[next_row]), item[2], next_row,
                    ))
        grouped: dict[int, list[tuple[float, str, int]]] = {}
        for lower, episode_id, shard_index, row in pending:
            grouped.setdefault(shard_index, []).append((lower, episode_id, row))
        for shard_index in sorted(grouped):
            shard = shards[shard_index]
            key = InstrumentKey(shard.metadata.dataset_id, shard.metadata.symbol)
            frame, representations, cutoff_to_row = materialized(shard_index)
            for lower, expected_id, row in grouped[shard_index]:
                cutoff_ns = int(shard.cutoffs_ns[row])
                batch_row = cutoff_to_row.get(cutoff_ns)
                if batch_row is None:
                    raise FrontierError(f"cannot reconstruct frontier cutoff {cutoff_ns}")
                position = shard.metadata.lookback - 1 + batch_row * shard.metadata.stride
                window = frame.iloc[
                    position - shard.metadata.lookback + 1:position + 1
                ].reset_index(drop=True)
                episode = Episode(
                    EpisodeKey(
                        key, pd.Timestamp(cutoff_ns), shard.metadata.lookback,
                        shard.metadata.representation_version,
                    ),
                    window, None, shard.metadata.quality_tier,
                )
                if episode.key.id != expected_id:
                    raise FrontierError("frontier episode identity changed during reconstruction")
                if not eligible(query, episode, request):
                    raise FrontierError(f"frontier emitted ineligible episode {episode.key.id}")
                candidate_representation = representations[batch_row]
                actual_lower, components, rigid = representation_distance_lower_bound(
                    query_representation, candidate_representation,
                )
                delta = abs(actual_lower - lower)
                maximum_bound_delta = max(maximum_bound_delta, delta)
                if delta > tolerance:
                    raise FrontierError(
                        f"frontier lower bound changed for {episode.key.id}: {delta:.3e}"
                    )
                total, exact_components, path = complete_representation_distance(
                    query_representation, candidate_representation,
                    actual_lower, components, rigid,
                )
                scored.append(ScoredCandidate(AnalogueMatch(
                    episode.key, total, exact_components, path,
                    episode.quality_tier, episode.quality_issues,
                ), episode))
                evaluated += 1
        selected = select_scored(scored, request)
        threshold = (
            max(match.total_distance for match in selected)
            if len(selected) >= request.top_k else float("inf")
        )
    matches = tuple(select_scored(scored, request))
    eligible_count = int(manifest["eligible_rows"])
    certificate = ExhaustiveCertificate(
        query.key.id, str(manifest["manifest_digest"]), eligible_count,
        evaluated, eligible_count - evaluated, stopped_early, threshold,
        next_lower, maximum_bound_delta, perf_counter() - started,
    )
    return ExhaustiveResult(matches, certificate)
