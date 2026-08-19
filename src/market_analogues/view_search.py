from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .fusion import reciprocal_rank_fusion
from .adapters import OHLCVSource
from .representation import represent
from .scan import _quality_issues
from .search import PrunedScoreReport, SearchCandidate, exact_search_pruned, latest_eligible_cutoff
from .types import AnalogueMatch, Episode, EpisodeKey, InstrumentKey, SearchQuery
from .view_signatures import VIEW_SIGNATURE_VERSION, episode_view_signature, signature_view_distances
from .view_store import VIEW_SHARD_SCHEMA_VERSION, LoadedViewShard, ViewShardError, load_view_shard


@dataclass(frozen=True)
class PersistedCandidateHit:
    episode_id: str
    instrument: InstrumentKey
    cutoff: pd.Timestamp
    lookback: int
    representation_version: str
    source_fingerprint: str
    quality_tier: str
    fusion_score: float
    view_distances: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class PersistedSearchReport:
    hits: tuple[PersistedCandidateHit, ...]
    shards_loaded: int
    rows_considered: int
    local_candidates: int
    elapsed_seconds: float
    manifest_digest: str
    benchmark_fingerprint: str | None
    source_versions: tuple[tuple[InstrumentKey, str, str], ...]


@dataclass(frozen=True)
class PersistedExactSearchReport:
    matches: tuple[AnalogueMatch, ...]
    candidate_search: PersistedSearchReport
    pruning: PrunedScoreReport
    candidates_materialized: int
    fingerprints_validated: int
    fingerprint_validation_seconds: float
    materialization_seconds: float
    exact_scoring_seconds: float
    elapsed_seconds: float


def _load_manifest(root: Path, dataset_id: str) -> tuple[dict[str, object], str]:
    path = root / dataset_id / "manifest.json"
    if not path.exists():
        raise ViewShardError(f"missing view-store manifest: {path}")
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        raise ViewShardError(f"invalid view-store manifest {path}: {exc}") from exc
    if payload.get("schema_version") != VIEW_SHARD_SCHEMA_VERSION:
        raise ViewShardError("view-store manifest schema is unsupported")
    if payload.get("signature_version") != VIEW_SIGNATURE_VERSION:
        raise ViewShardError("view-store manifest signature version is stale")
    records = payload.get("shards")
    if not isinstance(records, list):
        raise ViewShardError("view-store manifest has no shard list")
    digest_payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    digest = sha256(digest_payload.encode()).hexdigest()
    if digest != payload.get("manifest_digest"):
        raise ViewShardError("view-store manifest digest mismatch")
    return payload, digest


def _load_record(root: Path, record: dict[str, object]) -> LoadedViewShard:
    relative = Path(str(record["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ViewShardError(f"unsafe shard path in manifest: {relative}")
    shard = load_view_shard(root / relative)
    expected = {key: value for key, value in record.items() if key != "path"}
    if asdict(shard.metadata) != expected:
        raise ViewShardError(f"shard metadata disagrees with manifest: {relative}")
    return shard


def search_view_store(
    query: Episode,
    request: SearchQuery,
    root: Path,
    *,
    candidate_pool: int = 1000,
    per_instrument_view: int = 5,
) -> PersistedSearchReport:
    """Search precomputed signatures without reading raw candidate OHLCV files."""
    if candidate_pool < 1:
        raise ValueError("candidate_pool must be positive")
    if per_instrument_view < 1:
        raise ValueError("per_instrument_view must be positive")
    started = perf_counter()
    dataset_id = query.key.instrument.dataset_id
    payload, manifest_digest = _load_manifest(root, dataset_id)
    if str(payload.get("dataset_id")) != dataset_id:
        raise ViewShardError("view-store manifest dataset disagrees with query")
    if str(payload.get("representation_version")) != query.key.representation_version:
        raise ViewShardError("view-store representation version is stale")
    if payload.get("failures"):
        raise ViewShardError("view-store manifest records an incomplete build")
    if request.search_datasets and dataset_id not in request.search_datasets:
        raise ViewShardError("query dataset is excluded by the search request")
    records = [
        record for record in payload["shards"]
        if int(record["lookback"]) == query.key.lookback
        and str(record["quality_tier"]) in request.quality_tiers
    ]
    records.sort(key=lambda value: (str(value["symbol"]), str(value["path"])))
    query_signature = episode_view_signature(query)
    latest = latest_eligible_cutoff(query, request.minimum_history_gap_bars)
    latest_ns = int(latest.value)
    query_start_ns = int(pd.Timestamp(query.bars.timestamp.iloc[0]).value)
    local_rows: list[dict[str, object]] = []
    rows_considered = 0
    view_names: tuple[str, ...] | None = None

    for record in records:
        shard = _load_record(root, record)
        eligible = shard.cutoffs_ns <= latest_ns
        if shard.metadata.dataset_id == dataset_id and shard.metadata.symbol == query.key.instrument.source_symbol:
            eligible &= shard.cutoffs_ns < query_start_ns
        positions = np.flatnonzero(eligible)
        rows_considered += len(positions)
        if not len(positions):
            continue
        distances = signature_view_distances(query_signature, shard.signatures[positions])
        view_names = tuple(distances)
        selected: set[int] = set()
        ids = shard.episode_ids[positions].astype(str)
        for values in distances.values():
            order = np.lexsort((ids, values))[:per_instrument_view]
            selected.update(int(index) for index in order)
        for local_index in sorted(selected):
            position = int(positions[local_index])
            local_rows.append({
                "episode_id": str(shard.episode_ids[position]),
                "dataset_id": shard.metadata.dataset_id,
                "symbol": shard.metadata.symbol,
                "cutoff_ns": int(shard.cutoffs_ns[position]),
                "lookback": shard.metadata.lookback,
                "representation_version": shard.metadata.representation_version,
                "source_fingerprint": shard.metadata.source_fingerprint,
                "quality_tier": shard.metadata.quality_tier,
                **{name: float(values[local_index]) for name, values in distances.items()},
            })

    if not local_rows or view_names is None:
        source_versions = tuple(
            (
                InstrumentKey(str(record["dataset_id"]), str(record["symbol"])),
                str(record["source_fingerprint"]), str(record["quality_tier"]),
            )
            for record in records
        )
        return PersistedSearchReport(
            (), len(records), rows_considered, 0,
            perf_counter() - started, manifest_digest,
            payload.get("benchmark_fingerprint"), source_versions,
        )
    frame = pd.DataFrame(local_rows)
    if frame.episode_id.duplicated().any():
        raise ViewShardError("local shard union contains duplicate episode IDs")
    fused = reciprocal_rank_fusion(
        frame, view_names, pool_size=candidate_pool,
    ).selected
    hits = tuple(
        PersistedCandidateHit(
            str(row.episode_id),
            InstrumentKey(str(row.dataset_id), str(row.symbol)),
            pd.Timestamp(int(row.cutoff_ns)), int(row.lookback),
            str(row.representation_version), str(row.source_fingerprint),
            str(row.quality_tier),
            float(row.fusion_score),
            tuple((name, float(getattr(row, name))) for name in view_names),
        )
        for row in fused.itertuples(index=False)
    )
    source_versions = tuple(
        (
            InstrumentKey(str(record["dataset_id"]), str(record["symbol"])),
            str(record["source_fingerprint"]), str(record["quality_tier"]),
        )
        for record in records
    )
    return PersistedSearchReport(
        hits, len(records), rows_considered, len(frame),
        perf_counter() - started, manifest_digest,
        payload.get("benchmark_fingerprint"), source_versions,
    )


def persisted_exact_search(
    query: Episode,
    source: OHLCVSource,
    request: SearchQuery,
    root: Path,
    *,
    candidate_pool: int = 1000,
    per_instrument_view: int = 5,
    workers: int = 1,
    use_dtw_bound: bool = False,
    quality: pd.DataFrame | None = None,
) -> PersistedExactSearchReport:
    """Retrieve from verified shards, validate inputs, then exact-safe rerank."""
    if workers < 1:
        raise ValueError("workers must be positive")
    started = perf_counter()
    candidate_search = search_view_store(
        query, request, root, candidate_pool=candidate_pool,
        per_instrument_view=per_instrument_view,
    )
    quality_map = (
        {str(row.symbol): row for row in quality.itertuples(index=False)}
        if quality is not None else {}
    )
    expected_fingerprints: dict[InstrumentKey, str] = {}
    expected_tiers: dict[InstrumentKey, str] = {}
    for instrument, fingerprint, tier in candidate_search.source_versions:
        previous = expected_fingerprints.setdefault(instrument, fingerprint)
        if previous != fingerprint:
            raise ViewShardError(f"inconsistent fingerprints for {instrument}")
        previous_tier = expected_tiers.setdefault(instrument, tier)
        if previous_tier != tier:
            raise ViewShardError(f"inconsistent quality tiers for {instrument}")
    validation_started = perf_counter()
    current_benchmark_fingerprint = source.benchmark_fingerprint()
    if current_benchmark_fingerprint != candidate_search.benchmark_fingerprint:
        raise ViewShardError(
            "benchmark changed after view-store build; rebuild the store"
        )
    for instrument, expected in expected_fingerprints.items():
        actual = source.fingerprint(instrument)
        if actual != expected:
            raise ViewShardError(
                f"source changed after view-store build for {instrument}; rebuild the store"
            )
        quality_record = quality_map.get(instrument.source_symbol)
        if quality_record is not None and str(quality_record.tier) != expected_tiers[instrument]:
            raise ViewShardError(
                f"quality tier changed after view-store build for {instrument}; rebuild the store"
            )
    fingerprint_validation_seconds = perf_counter() - validation_started

    materialization_started = perf_counter()
    benchmark = source.load_benchmark()

    def materialize(hit: PersistedCandidateHit) -> SearchCandidate:
        bars = source.load(hit.instrument)
        window = bars[bars.timestamp <= hit.cutoff].tail(hit.lookback).reset_index(drop=True)
        if len(window) != hit.lookback or pd.Timestamp(window.timestamp.iloc[-1]) != hit.cutoff:
            raise ViewShardError(
                f"cannot reconstruct stored episode {hit.episode_id} from current source"
            )
        quality_record = quality_map.get(hit.instrument.source_symbol)
        episode = Episode(
            EpisodeKey(
                hit.instrument, hit.cutoff, hit.lookback, hit.representation_version,
            ),
            window, benchmark, hit.quality_tier, _quality_issues(quality_record),
        )
        if episode.key.id != hit.episode_id:
            raise ViewShardError(
                f"stored episode ID disagrees with reconstructed key: {hit.episode_id}"
            )
        return SearchCandidate(episode, represent(episode))

    if workers == 1:
        candidates = [materialize(hit) for hit in candidate_search.hits]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            candidates = list(executor.map(materialize, candidate_search.hits))
    materialization_seconds = perf_counter() - materialization_started
    scoring_started = perf_counter()
    matches, pruning = exact_search_pruned(
        query, candidates, request, use_dtw_bound=use_dtw_bound,
    )
    exact_scoring_seconds = perf_counter() - scoring_started
    return PersistedExactSearchReport(
        tuple(matches), candidate_search, pruning, len(candidates),
        len(expected_fingerprints), fingerprint_validation_seconds,
        materialization_seconds, exact_scoring_seconds,
        perf_counter() - started,
    )
