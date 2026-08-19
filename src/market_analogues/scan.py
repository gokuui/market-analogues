from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import importlib.util
import resource
from time import perf_counter

import numpy as np
import pandas as pd

from .adapters import OHLCVSource
from .candidate_views import VIEW_NAMES, sliding_view_distances
from .episodes import build_episode
from .fusion import reciprocal_rank_fusion
from .search import SearchCandidate, exact_search, latest_eligible_cutoff
from .types import AnalogueMatch, Episode, InstrumentKey, SearchQuery


@dataclass(frozen=True)
class CoarseHit:
    instrument: InstrumentKey
    cutoff: pd.Timestamp
    lookback: int
    distance: float
    view_distances: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class ScanReport:
    matches: list[AnalogueMatch]
    instruments_scanned: int
    windows_scanned: int
    coarse_candidates: int
    elapsed_seconds: float
    instruments_considered: int = 0
    quality_skipped: int = 0
    failures: tuple[str, ...] = ()
    peak_rss_mb: float = 0.0


@dataclass(frozen=True)
class CandidateScanReport:
    hits: tuple[CoarseHit, ...]
    instruments_considered: int
    instruments_scanned: int
    quality_skipped: int
    windows_scanned: int
    failures: tuple[str, ...]
    elapsed_seconds: float
    peak_rss_mb: float


def _quality_issues(record: object | None) -> tuple[str, ...]:
    if record is None:
        return ()
    value = getattr(record, "issues", None)
    if value is None or pd.isna(value):
        return ()
    return tuple(part for part in str(value).split(";") if part)


def _row_correlation_distance(rows: np.ndarray, query: np.ndarray) -> np.ndarray:
    left = rows - rows.mean(axis=1, keepdims=True)
    right = query - query.mean()
    denominator = np.sqrt(np.sum(left * left, axis=1)) * np.sqrt(np.sum(right * right))
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = (left @ right) / denominator
    return 1 - np.nan_to_num(correlation, nan=-1.0, posinf=-1.0, neginf=-1.0)


def scan_frame(
    query: Episode,
    candidate_bars: pd.DataFrame,
    instrument: InstrumentKey,
    *,
    stride: int = 5,
    per_instrument: int = 5,
    shape_weight: float = .65,
    backend: str = "auto",
    minimum_gap_bars: int = 60,
) -> tuple[list[CoarseHit], int]:
    """Vectorized, bounded-memory sliding scan inspired by public stock matchers."""
    lookback = query.key.lookback
    latest_eligible = latest_eligible_cutoff(query, minimum_gap_bars)
    eligible = candidate_bars[candidate_bars.timestamp <= latest_eligible].reset_index(drop=True)
    if len(eligible) < lookback:
        return [], 0
    query_close = np.log(query.bars.close.astype(float).clip(lower=1e-12).to_numpy())
    query_path = query_close - query_close[0]
    candidate_close = np.log(eligible.close.astype(float).clip(lower=1e-12).to_numpy())
    all_windows = np.lib.stride_tricks.sliding_window_view(candidate_close, lookback)
    windows = all_windows[::stride]
    positions = np.arange(lookback - 1, len(eligible), stride)
    paths = windows - windows[:, :1]
    use_mass = backend == "mass" or (
        backend == "auto" and len(all_windows) >= 20_000 and importlib.util.find_spec("stumpy") is not None
    )
    if backend not in {"auto", "vector", "mass"}:
        raise ValueError("backend must be auto, vector, or mass")
    if use_mass:
        try:
            import stumpy
            mass_distance = np.asarray(stumpy.mass(query_path, candidate_close), dtype=float)[::stride]
            correlation_distance = np.clip(mass_distance ** 2 / (2 * lookback), 0, 2)
            if not np.isfinite(correlation_distance).all():
                raise ValueError("MASS returned non-finite distances")
        except Exception:
            if backend == "mass":
                raise
            correlation_distance = _row_correlation_distance(paths, query_path)
    else:
        correlation_distance = _row_correlation_distance(paths, query_path)
    magnitude_distance = np.sqrt(np.mean((paths - query_path) ** 2, axis=1))
    # Correlation catches shape under amplitude changes; path RMSE retains the
    # strength of the move. Exact multichannel reranking follows this stage.
    score = shape_weight * correlation_distance + (1 - shape_weight) * magnitude_distance
    if instrument == query.key.instrument:
        query_timestamps = set(query.bars.timestamp.astype(str))
        overlap = np.array([
            bool(query_timestamps.intersection(
                eligible.timestamp.iloc[end - lookback + 1:end + 1].astype(str)
            )) for end in positions
        ])
        score[overlap] = np.inf
    finite = np.flatnonzero(np.isfinite(score))
    if not len(finite):
        return [], len(windows)
    count = min(per_instrument, len(finite))
    chosen = finite[np.argpartition(score[finite], count - 1)[:count]]
    chosen = chosen[np.argsort(score[chosen], kind="stable")]
    return [
        CoarseHit(
            instrument, pd.Timestamp(eligible.timestamp.iloc[positions[index]]),
            lookback, float(score[index]),
        ) for index in chosen
    ], len(windows)


def scan_frame_multiview(
    query: Episode,
    candidate_bars: pd.DataFrame,
    instrument: InstrumentKey,
    *,
    stride: int = 5,
    per_instrument: int = 5,
    minimum_gap_bars: int = 60,
    benchmark: pd.DataFrame | None = None,
) -> tuple[list[CoarseHit], int]:
    """Retain independent local candidates from each cheap chart-information view."""
    latest = latest_eligible_cutoff(query, minimum_gap_bars)
    eligible = candidate_bars[candidate_bars.timestamp <= latest].reset_index(drop=True)
    lookback = query.key.lookback
    if len(eligible) < lookback:
        return [], 0
    views = sliding_view_distances(
        query.bars, eligible, stride=stride, query_benchmark=query.benchmark,
        candidate_benchmark=benchmark,
    )
    positions = np.arange(lookback - 1, len(eligible), stride)
    if instrument == query.key.instrument:
        query_timestamps = set(query.bars.timestamp.astype(str))
        overlap = np.array([
            bool(query_timestamps.intersection(
                eligible.timestamp.iloc[end - lookback + 1:end + 1].astype(str)
            )) for end in positions
        ])
        for values in views.values():
            values[overlap] = np.inf
    chosen: set[int] = set()
    for values in views.values():
        finite = np.flatnonzero(np.isfinite(values))
        count = min(per_instrument, len(finite))
        if count:
            local = finite[np.argpartition(values[finite], count - 1)[:count]]
            chosen.update(int(index) for index in local)
    hits = [
        CoarseHit(
            instrument, pd.Timestamp(eligible.timestamp.iloc[positions[index]]), lookback,
            float(views["price_shape"][index]),
            tuple((name, float(views[name][index])) for name in VIEW_NAMES),
        )
        for index in chosen
    ]
    hits.sort(key=lambda hit: (hit.distance, hit.cutoff))
    return hits, len(positions)


def scan_universe_candidates(
    query: Episode,
    source: OHLCVSource,
    request: SearchQuery,
    *,
    stride: int = 5,
    candidate_pool: int = 1000,
    per_instrument: int = 5,
    quality: pd.DataFrame | None = None,
    instrument_limit: int | None = None,
    scan_backend: str = "auto",
    workers: int = 1,
    candidate_strategy: str = "price",
) -> CandidateScanReport:
    """Scan every eligible instrument and retain a deterministic global pool."""
    if workers < 1:
        raise ValueError("workers must be positive")
    if candidate_strategy not in {"price", "multiview"}:
        raise ValueError("candidate_strategy must be price or multiview")
    started = perf_counter()
    qmap = {}
    if quality is not None and len(quality):
        qmap = {str(row.symbol): row for row in quality.itertuples(index=False)}
    instruments = source.instruments()
    benchmark = source.load_benchmark() if candidate_strategy == "multiview" else None
    if instrument_limit is not None:
        instruments = instruments[:instrument_limit]


    def scan_one(instrument: InstrumentKey) -> tuple[str, list[CoarseHit], int, str | None]:
        record = qmap.get(instrument.source_symbol)
        tier = str(record.tier) if record is not None else "A"
        if tier not in request.quality_tiers or tier == "QUARANTINED":
            return "skipped", [], 0, None
        try:
            bars = source.load(instrument)
            if candidate_strategy == "multiview":
                local_hits, windows = scan_frame_multiview(
                    query, bars, instrument, stride=stride,
                    per_instrument=per_instrument,
                    minimum_gap_bars=request.minimum_history_gap_bars,
                    benchmark=benchmark,
                )
            else:
                local_hits, windows = scan_frame(
                    query, bars, instrument, stride=stride, per_instrument=per_instrument,
                    backend=scan_backend, minimum_gap_bars=request.minimum_history_gap_bars,
                )
            return "scanned", local_hits, windows, None
        except Exception as exc:
            failure = f"{instrument}:{type(exc).__name__}:{exc}"
            return "failed", [], 0, failure

    if workers == 1:
        results = map(scan_one, instruments)
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        results = executor.map(scan_one, instruments)
    hits: list[CoarseHit] = []
    windows_scanned = instruments_scanned = quality_skipped = 0
    failures: list[str] = []
    try:
        for status, local_hits, windows, failure in results:
            if status == "skipped":
                quality_skipped += 1
                continue
            if status == "failed":
                failures.append(str(failure))
                continue
            hits.extend(local_hits)
            windows_scanned += windows
            instruments_scanned += 1
    finally:
        if workers != 1:
            executor.shutdown(wait=True)
    if candidate_strategy == "multiview" and hits:
        rows = []
        by_id: dict[str, CoarseHit] = {}
        for hit in hits:
            identifier = f"{hit.instrument}:{hit.cutoff.isoformat()}:{hit.lookback}"
            by_id[identifier] = hit
            rows.append({"episode_id": identifier, **dict(hit.view_distances)})
        fused = reciprocal_rank_fusion(
            pd.DataFrame(rows), VIEW_NAMES, pool_size=candidate_pool,
        ).selected
        hits = [
            replace(by_id[row.episode_id], distance=-float(row.fusion_score))
            for row in fused.itertuples(index=False)
        ]
    else:
        hits.sort(key=lambda hit: (hit.distance, str(hit.instrument), hit.cutoff))
        hits = hits[:candidate_pool]
    return CandidateScanReport(
        tuple(hits), len(instruments), instruments_scanned, quality_skipped,
        windows_scanned, tuple(failures), perf_counter() - started,
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )


def streaming_search(
    query: Episode,
    source: OHLCVSource,
    request: SearchQuery | None = None,
    *,
    stride: int = 5,
    candidate_pool: int = 1000,
    per_instrument: int = 5,
    quality: pd.DataFrame | None = None,
    instrument_limit: int | None = None,
    scan_backend: str = "auto",
    workers: int = 1,
    candidate_strategy: str = "price",
) -> ScanReport:
    """Search a universe without pre-materializing every historical episode."""
    started = perf_counter()
    request = request or SearchQuery(query.key)
    qmap = {}
    if quality is not None and len(quality):
        qmap = {str(row.symbol): row for row in quality.itertuples(index=False)}
    candidate_scan = scan_universe_candidates(
        query, source, request, stride=stride, candidate_pool=candidate_pool,
        per_instrument=per_instrument, quality=quality,
        instrument_limit=instrument_limit, scan_backend=scan_backend, workers=workers,
        candidate_strategy=candidate_strategy,
    )
    candidates: list[SearchCandidate] = []
    rerank_failures: list[str] = []
    for hit in candidate_scan.hits:
        record = qmap.get(hit.instrument.source_symbol)
        tier = str(record.tier) if record is not None else "A"
        issues = _quality_issues(record)
        try:
            episode = build_episode(
                source, hit.instrument, hit.cutoff, hit.lookback,
                query.key.representation_version, tier, issues,
            )
            candidates.append(SearchCandidate.from_episode(episode))
        except Exception as exc:
            rerank_failures.append(f"{hit.instrument}:{hit.cutoff}:{type(exc).__name__}:{exc}")
    matches = exact_search(query, candidates, request)
    return ScanReport(
        matches, candidate_scan.instruments_scanned, candidate_scan.windows_scanned,
        len(candidate_scan.hits), perf_counter() - started,
        candidate_scan.instruments_considered, candidate_scan.quality_skipped,
        candidate_scan.failures + tuple(rerank_failures), candidate_scan.peak_rss_mb,
    )
