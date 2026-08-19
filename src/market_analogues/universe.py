from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from html import escape
import json
from pathlib import Path
from time import perf_counter

import pandas as pd
from scipy.stats import spearmanr

from .adapters import OHLCVSource
from .episodes import build_episode
from .scan import CandidateScanReport, _quality_issues, scan_universe_candidates
from .search import SearchCandidate, latest_eligible_cutoff, score_candidates, select_scored
from .types import AnalogueMatch, Episode, SearchQuery


@dataclass(frozen=True)
class UniverseVerification:
    passed: bool
    metrics: dict[str, object]
    failures: tuple[str, ...]
    matches: tuple[AnalogueMatch, ...]
    candidate_scan: CandidateScanReport


def _digest(scan: CandidateScanReport) -> str:
    payload = [
        (str(hit.instrument), hit.cutoff.isoformat(), hit.lookback, round(hit.distance, 12))
        for hit in scan.hits
    ]
    return sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def _candidate(
    query: Episode, source: OHLCVSource, hit, qmap: dict[str, object],
) -> SearchCandidate:
    record = qmap.get(hit.instrument.source_symbol)
    tier = str(record.tier) if record is not None else "A"
    episode = build_episode(
        source, hit.instrument, hit.cutoff, hit.lookback,
        query.key.representation_version, tier, _quality_issues(record),
    )
    return SearchCandidate.from_episode(episode)


def verify_universe(
    query: Episode,
    source: OHLCVSource,
    request: SearchQuery,
    quality: pd.DataFrame,
    *,
    stride: int = 5,
    reference_pool: int = 1000,
    comparison_pools: tuple[int, ...] = (50, 100, 200, 500),
    per_instrument: int = 5,
    scan_backend: str = "vector",
    workers: int = 1,
    max_seconds: float = 1800,
    max_rss_mb: float = 4096,
    minimum_pool_recall: float = .90,
    recall_pool: int = 200,
    candidate_strategy: str = "price",
) -> UniverseVerification:
    started = perf_counter()
    scan = scan_universe_candidates(
        query, source, request, stride=stride, candidate_pool=reference_pool,
        per_instrument=per_instrument, quality=quality,
        scan_backend=scan_backend, workers=workers,
        candidate_strategy=candidate_strategy,
    )
    qmap = {str(row.symbol): row for row in quality.itertuples(index=False)}
    candidates: list[SearchCandidate] = []
    build_failures: list[str] = []
    for hit in scan.hits:
        try:
            candidates.append(_candidate(query, source, hit, qmap))
        except Exception as exc:
            build_failures.append(
                f"{hit.instrument}:{hit.cutoff}:{type(exc).__name__}:{exc}"
            )
    reference_request = replace(request, top_k=request.top_k)
    scored = score_candidates(query, candidates, reference_request)
    reference_matches = select_scored(scored, reference_request)
    exact_scores = [item.match.total_distance for item in scored]
    coarse_scores = [hit.distance for hit in scan.hits[:len(scored)]]
    rank_correlation = (
        float(spearmanr(coarse_scores, exact_scores).statistic)
        if len(scored) > 2 else 0.0
    )
    coarse_rank = {item.match.episode_key.id: rank + 1 for rank, item in enumerate(scored)}
    reference_coarse_ranks = [coarse_rank[match.episode_key.id] for match in reference_matches]
    component_means = {
        name: float(sum(match.component_distances[name] for match in reference_matches) / len(reference_matches))
        for name in reference_matches[0].component_distances
    } if reference_matches else {}
    reference_ids = {match.episode_key.id for match in reference_matches}
    pool_recalls: dict[str, float] = {}
    pool_rankings: dict[str, list[str]] = {}
    for size in sorted(set(comparison_pools + (reference_pool,))):
        matches = select_scored(scored[:size], reference_request)
        ids = [match.episode_key.id for match in matches]
        pool_rankings[str(size)] = ids
        pool_recalls[str(size)] = (
            len(reference_ids.intersection(ids)) / len(reference_ids)
            if reference_ids else 0.0
        )

    eligible_tiers = quality.tier.astype(str).isin(request.quality_tiers)
    expected_scanned = int(eligible_tiers.sum())
    latest = latest_eligible_cutoff(query, request.minimum_history_gap_bars)
    elapsed = perf_counter() - started
    throughput = scan.windows_scanned / scan.elapsed_seconds if scan.elapsed_seconds else 0.0
    failures = list(scan.failures) + build_failures
    if scan.instruments_scanned != expected_scanned:
        failures.append(
            f"eligible coverage {scan.instruments_scanned}/{expected_scanned} instruments"
        )
    if scan.quality_skipped != scan.instruments_considered - expected_scanned:
        failures.append("quality skip accounting does not reconcile")
    if scan.windows_scanned <= 0:
        failures.append("no historical windows were scanned")
    if not reference_matches:
        failures.append("reference candidate pool produced no eligible matches")
    if any(match.episode_key.cutoff > latest for match in reference_matches):
        failures.append("a returned match violates the observed-session history gap")
    measured_recall = pool_recalls.get(str(recall_pool))
    if measured_recall is None:
        failures.append(f"recall pool {recall_pool} was not evaluated")
    elif measured_recall < minimum_pool_recall:
        failures.append(
            f"pool saturation recall@{request.top_k}={measured_recall:.3f} "
            f"below {minimum_pool_recall:.3f} for pool {recall_pool} vs {reference_pool}"
        )
    if elapsed > max_seconds:
        failures.append(f"elapsed {elapsed:.2f}s exceeds {max_seconds:.2f}s")
    if scan.peak_rss_mb > max_rss_mb:
        failures.append(f"peak RSS {scan.peak_rss_mb:.1f}MB exceeds {max_rss_mb:.1f}MB")

    metrics: dict[str, object] = {
        "dataset": query.key.instrument.dataset_id,
        "query": str(query.key.instrument),
        "query_cutoff": query.key.cutoff.isoformat(),
        "lookback": query.key.lookback,
        "minimum_history_gap_bars": request.minimum_history_gap_bars,
        "latest_eligible_cutoff": latest.isoformat(),
        "instruments_considered": scan.instruments_considered,
        "expected_quality_eligible": expected_scanned,
        "instruments_scanned": scan.instruments_scanned,
        "quality_skipped": scan.quality_skipped,
        "windows_scanned": scan.windows_scanned,
        "candidate_pool_requested": reference_pool,
        "candidate_pool_built": len(candidates),
        "matches": len(reference_matches),
        "candidate_scan_seconds": scan.elapsed_seconds,
        "total_seconds": elapsed,
        "windows_per_second": throughput,
        "peak_rss_mb": scan.peak_rss_mb,
        "candidate_digest": _digest(scan),
        "coarse_exact_spearman": rank_correlation,
        "reference_match_coarse_ranks": reference_coarse_ranks,
        "reference_component_means": component_means,
        "pool_saturation_recall": pool_recalls,
        "pool_rankings": pool_rankings,
        "scan_backend": scan_backend,
        "candidate_strategy": candidate_strategy,
        "workers": workers,
    }
    return UniverseVerification(
        not failures, metrics, tuple(failures), tuple(reference_matches), scan,
    )


def write_universe_report(result: UniverseVerification, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_rows = "".join(
        f"<tr><th>{escape(str(key))}</th><td><code>{escape(json.dumps(value, default=str))}</code></td></tr>"
        for key, value in result.metrics.items() if key != "pool_rankings"
    )
    match_rows = "".join(
        f"<tr><td>{rank}</td><td>{escape(str(match.episode_key.instrument))}</td>"
        f"<td>{match.episode_key.cutoff.date()}</td><td>{match.total_distance:.6f}</td></tr>"
        for rank, match in enumerate(result.matches, 1)
    )
    failures = "".join(f"<li>{escape(item)}</li>" for item in result.failures) or "<li>None</li>"
    status = "PASS" if result.passed else "FAIL"
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Universe verification — {escape(str(result.metrics['dataset']))}</title><style>body{{font-family:system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;background:#f5f7f8;color:#17202a}}section,header{{background:white;border:1px solid #d5d8dc;border-radius:10px;padding:1.3rem;margin:1rem 0}}table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;padding:.6rem;border-bottom:1px solid #ddd}}.pass{{color:#117864}}.fail{{color:#b03a2e}}code{{white-space:pre-wrap}}</style></head><body><header><h1>Complete-universe verification: <span class="{status.lower()}">{status}</span></h1><p>This verifies universe coverage, temporal eligibility, candidate-pool saturation, latency, memory, and reproducibility metadata. Pool saturation is measured against a larger coarse pool; it is not exhaustive composite-distance recall over every window.</p></header><section><h2>Metrics</h2><table>{metric_rows}</table></section><section><h2>Failures</h2><ul>{failures}</ul></section><section><h2>Reference-pool matches</h2><table><thead><tr><th>#</th><th>Instrument</th><th>Cutoff</th><th>Distance</th></tr></thead><tbody>{match_rows}</tbody></table></section></body></html>"""
    path.write_text(html)
    return path
