from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .fusion import reciprocal_rank_fusion
from .search import latest_eligible_cutoff
from .types import Episode, InstrumentKey, SearchQuery
from .view_signatures import VIEW_SIGNATURE_VERSION, episode_view_signature, signature_view_distances
from .view_store import VIEW_SHARD_SCHEMA_VERSION, LoadedViewShard, ViewShardError, load_view_shard


@dataclass(frozen=True)
class PersistedCandidateHit:
    episode_id: str
    instrument: InstrumentKey
    cutoff: pd.Timestamp
    lookback: int
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
                **{name: float(values[local_index]) for name, values in distances.items()},
            })

    if not local_rows or view_names is None:
        return PersistedSearchReport((), len(records), rows_considered, 0,
                                     perf_counter() - started, manifest_digest)
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
            float(row.fusion_score),
            tuple((name, float(getattr(row, name))) for name in view_names),
        )
        for row in fused.itertuples(index=False)
    )
    return PersistedSearchReport(
        hits, len(records), rows_considered, len(frame),
        perf_counter() - started, manifest_digest,
    )
