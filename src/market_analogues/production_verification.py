from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
import resource
from time import perf_counter

import numpy as np
import pandas as pd

from .adapters import OHLCVSource
from .types import Episode, SearchQuery, stable_hash
from .view_search import persisted_exact_search


@dataclass(frozen=True)
class ProductionSearchVerification:
    passed: bool
    metrics: dict[str, object]
    failures: tuple[str, ...]
    runs: pd.DataFrame


def verify_production_search(
    query: Episode,
    source: OHLCVSource,
    request: SearchQuery,
    view_store_root: Path,
    *,
    quality: pd.DataFrame | None = None,
    candidate_pool: int = 175,
    per_instrument_view: int = 5,
    workers: int = 1,
    repeat: int = 2,
    use_dtw_bound: bool = False,
    max_seconds: float = 300.0,
    max_rss_mb: float = 1024.0,
    tolerance: float = 1e-12,
) -> ProductionSearchVerification:
    """Verify deterministic, bounded persisted retrieval plus exact-safe reranking."""
    if repeat < 2:
        raise ValueError("repeat must be at least 2 to verify determinism")
    started = perf_counter()
    failures: list[str] = []
    rows: list[dict[str, object]] = []
    authority_ids: list[str] | None = None
    authority_distances: dict[str, float] = {}
    authority_components: dict[str, dict[str, float]] = {}

    for run_number in range(1, repeat + 1):
        result = persisted_exact_search(
            query, source, request, view_store_root,
            candidate_pool=candidate_pool,
            per_instrument_view=per_instrument_view,
            workers=workers,
            use_dtw_bound=use_dtw_bound,
            quality=quality,
        )
        ids = [match.episode_key.id for match in result.matches]
        digest = stable_hash({
            "ids": ids,
            "distances": [round(float(match.total_distance), 12) for match in result.matches],
            "components": [
                {key: round(float(value), 12) for key, value in sorted(match.component_distances.items())}
                for match in result.matches
            ],
        })
        maximum_delta = 0.0
        if authority_ids is None:
            authority_ids = ids
            authority_distances = {
                match.episode_key.id: float(match.total_distance) for match in result.matches
            }
            authority_components = {
                match.episode_key.id: dict(match.component_distances)
                for match in result.matches
            }
        else:
            if ids != authority_ids:
                failures.append(
                    f"run {run_number}: ordered match IDs differ from run 1"
                )
            for match in result.matches:
                episode_id = match.episode_key.id
                if episode_id not in authority_distances:
                    continue
                delta = abs(match.total_distance - authority_distances[episode_id])
                maximum_delta = max(maximum_delta, delta)
                if not np.isclose(
                    match.total_distance, authority_distances[episode_id],
                    rtol=tolerance, atol=tolerance,
                ):
                    failures.append(
                        f"run {run_number}: distance differs for {episode_id}"
                    )
                if match.component_distances.keys() != authority_components[episode_id].keys():
                    failures.append(
                        f"run {run_number}: component keys differ for {episode_id}"
                    )
                    continue
                for name, value in match.component_distances.items():
                    component_delta = abs(value - authority_components[episode_id][name])
                    maximum_delta = max(maximum_delta, component_delta)
                    if not np.isclose(
                        value, authority_components[episode_id][name],
                        rtol=tolerance, atol=tolerance,
                    ):
                        failures.append(
                            f"run {run_number}: component {name} differs for {episode_id}"
                        )
        if len(ids) != request.top_k:
            failures.append(
                f"run {run_number}: returned {len(ids)} matches; require {request.top_k}"
            )
        if result.elapsed_seconds > max_seconds:
            failures.append(
                f"run {run_number}: {result.elapsed_seconds:.2f}s exceeds {max_seconds:.2f}s"
            )
        rows.append({
            "run": run_number,
            "digest": digest,
            "matches": len(ids),
            "shards_loaded": result.candidate_search.shards_loaded,
            "rows_considered": result.candidate_search.rows_considered,
            "local_candidates": result.candidate_search.local_candidates,
            "candidate_hits": len(result.candidate_search.hits),
            "fingerprints_validated": result.fingerprints_validated,
            "exact_evaluated": result.pruning.exact_evaluated,
            "safely_pruned": result.pruning.safely_pruned,
            "candidate_seconds": result.candidate_search.elapsed_seconds,
            "fingerprint_seconds": result.fingerprint_validation_seconds,
            "materialization_seconds": result.materialization_seconds,
            "exact_seconds": result.exact_scoring_seconds,
            "elapsed_seconds": result.elapsed_seconds,
            "maximum_repeat_delta": maximum_delta,
            "manifest_digest": result.candidate_search.manifest_digest,
        })

    runs = pd.DataFrame(rows)
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    if peak_rss_mb > max_rss_mb:
        failures.append(f"peak RSS {peak_rss_mb:.1f} MB exceeds {max_rss_mb:.1f} MB")
    if len(runs) and runs.digest.nunique() != 1:
        failures.append("repeat ranking digests differ")
    metrics: dict[str, object] = {
        "dataset": query.key.instrument.dataset_id,
        "query_episode_id": query.key.id,
        "query_symbol": query.key.instrument.source_symbol,
        "cutoff": query.key.cutoff.isoformat(),
        "lookback": query.key.lookback,
        "top_k": request.top_k,
        "candidate_pool": candidate_pool,
        "per_instrument_view": per_instrument_view,
        "workers": workers,
        "repeat": repeat,
        "pruning_mode": "symmetric_multivariate_lb_keogh" if use_dtw_bound else "non_dtw_partial_sum",
        "deterministic_digest": str(runs.digest.iloc[0]) if len(runs) else "",
        "maximum_repeat_delta": float(runs.maximum_repeat_delta.max()) if len(runs) else 0.0,
        "maximum_elapsed_seconds": float(runs.elapsed_seconds.max()) if len(runs) else 0.0,
        "mean_elapsed_seconds": float(runs.elapsed_seconds.mean()) if len(runs) else 0.0,
        "maximum_candidate_seconds": float(runs.candidate_seconds.max()) if len(runs) else 0.0,
        "maximum_fingerprint_seconds": float(runs.fingerprint_seconds.max()) if len(runs) else 0.0,
        "maximum_materialization_seconds": float(runs.materialization_seconds.max()) if len(runs) else 0.0,
        "maximum_exact_seconds": float(runs.exact_seconds.max()) if len(runs) else 0.0,
        "minimum_exact_prune_fraction": float(
            (runs.safely_pruned / (runs.safely_pruned + runs.exact_evaluated)).min()
        ) if len(runs) else 0.0,
        "peak_rss_mb": peak_rss_mb,
        "max_seconds": max_seconds,
        "max_rss_mb": max_rss_mb,
        "tolerance": tolerance,
        "seconds": perf_counter() - started,
    }
    return ProductionSearchVerification(
        bool(len(runs)) and not failures, metrics, tuple(failures), runs,
    )


def write_production_search_report(
    result: ProductionSearchVerification,
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "".join(
        f"<tr><td>{int(row.run)}</td><td><code>{escape(str(row.digest))}</code></td>"
        f"<td>{int(row.shards_loaded):,}</td><td>{int(row.rows_considered):,}</td>"
        f"<td>{int(row.candidate_hits)}</td><td>{int(row.exact_evaluated)}</td>"
        f"<td>{int(row.safely_pruned)}</td><td>{float(row.candidate_seconds):.2f}</td>"
        f"<td>{float(row.fingerprint_seconds):.2f}</td>"
        f"<td>{float(row.materialization_seconds):.2f}</td>"
        f"<td>{float(row.exact_seconds):.2f}</td>"
        f"<td>{float(row.elapsed_seconds):.2f}</td></tr>"
        for row in result.runs.itertuples(index=False)
    )
    failure_items = "".join(
        f"<li>{escape(value)}</li>" for value in result.failures
    ) or "<li>None</li>"
    status = "PASS" if result.passed else "FAIL"
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Production persisted search verification</title><style>body{{font-family:system-ui,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem;background:#f5f7f8;color:#17202a}}section,header{{background:white;border:1px solid #ddd;border-radius:10px;padding:1.3rem;margin:1rem 0}}table{{border-collapse:collapse;width:100%;font-size:.9rem}}th,td{{padding:.55rem;text-align:left;border-bottom:1px solid #ddd}}.pass{{color:#117864}}.fail{{color:#b03a2e}}pre{{overflow:auto}}code{{overflow-wrap:anywhere}}</style></head><body><header><h1>Production persisted search: <span class="{status.lower()}">{status}</span></h1><p>Repeated complete-store retrieval validates every eligible shard, source fingerprint, quality tier and benchmark fingerprint, materializes exact representations, applies the Gate 11B exact-safe reranker, and requires stable ordered results within explicit time and memory budgets.</p></header><section><h2>Aggregate evidence</h2><pre>{escape(json.dumps(result.metrics, indent=2))}</pre></section><section><h2>Repeated runs</h2><table><thead><tr><th>Run</th><th>Digest</th><th>Shards</th><th>Rows</th><th>Pool</th><th>Exact</th><th>Pruned</th><th>Candidate s</th><th>Fingerprint s</th><th>Materialize s</th><th>Exact s</th><th>Total s</th></tr></thead><tbody>{rows}</tbody></table></section><section><h2>Failures</h2><ul>{failure_items}</ul></section></body></html>"""
    path.write_text(html)
    return path
