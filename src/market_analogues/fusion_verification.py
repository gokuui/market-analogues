from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
from time import perf_counter

import pandas as pd

from .adapters import OHLCVSource
from .candidate_views import CANDIDATE_VIEW_VERSION, VIEW_NAMES, episode_view_distances
from .episodes import build_episode
from .fusion import preselect_per_group, reciprocal_rank_fusion
from .types import Episode, EpisodeKey, InstrumentKey


@dataclass(frozen=True)
class FusionVerification:
    passed: bool
    metrics: dict[str, object]
    failures: tuple[str, ...]
    cases: pd.DataFrame


def _candidate_episode(
    source: OHLCVSource,
    benchmark: pd.DataFrame | None,
    dataset: str,
    symbol: str,
    cutoff: object,
    lookback: int,
    representation_version: str,
    quality_tier: str,
) -> Episode:
    instrument = InstrumentKey(dataset, symbol)
    bars = source.load(instrument)
    window = bars[bars.timestamp <= pd.Timestamp(cutoff)].tail(lookback).reset_index(drop=True)
    if len(window) != lookback:
        raise ValueError(f"incomplete candidate window {instrument}:{cutoff}:{lookback}")
    actual = pd.Timestamp(window.timestamp.iloc[-1])
    return Episode(
        EpisodeKey(instrument, actual, lookback, representation_version),
        window, benchmark, quality_tier,
    )


def verify_candidate_fusion(
    source: OHLCVSource,
    oracle_directory: Path,
    *,
    representation_version: str,
    pool_sizes: tuple[int, ...] = (50, 100, 125),
    acceptance_pool: int = 125,
    minimum_recall: float = .95,
    per_instrument_view: int = 5,
) -> FusionVerification:
    """Benchmark the production-cheap view union against Gate 09 exact rankings."""
    started = perf_counter()
    summary_path = oracle_directory / "oracle-summary.parquet"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = pd.read_parquet(summary_path)
    benchmark = source.load_benchmark()
    case_rows: list[dict[str, object]] = []
    failures: list[str] = []
    view_columns = tuple(f"candidate_view_{name}" for name in VIEW_NAMES)

    for case in summary.sort_values("case_id").itertuples(index=False):
        case_started = perf_counter()
        path = oracle_directory / f"{case.case_id}.parquet"
        ranking = pd.read_parquet(path)
        query_symbol = str(case.query).split(":", 1)[-1]
        query = build_episode(
            source, InstrumentKey(str(source.instruments()[0].dataset_id), query_symbol),
            case.cutoff, int(case.lookback), representation_version,
        )
        cached_version = (
            str(ranking.candidate_view_version.iloc[0])
            if "candidate_view_version" in ranking and len(ranking) else ""
        )
        if not set(view_columns).issubset(ranking.columns) or cached_version != CANDIDATE_VIEW_VERSION:
            measured: dict[str, list[float]] = {column: [] for column in view_columns}
            for row in ranking.itertuples(index=False):
                candidate = _candidate_episode(
                    source, benchmark, str(row.dataset_id), str(row.symbol), row.cutoff,
                    int(row.lookback), representation_version, str(row.quality_tier),
                )
                distances = episode_view_distances(query, candidate)
                for name, value in distances.items():
                    measured[f"candidate_view_{name}"].append(value)
            for column, values in measured.items():
                ranking[column] = values
            ranking["candidate_view_version"] = CANDIDATE_VIEW_VERSION
            ranking.to_parquet(path, index=False)

        local_union = preselect_per_group(
            ranking, view_columns, per_view=per_instrument_view,
        )
        target = set(ranking.loc[ranking.oracle_selected, "episode_id"].astype(str))
        recalls: dict[str, float] = {}
        for pool_size in sorted(set(pool_sizes + (acceptance_pool,))):
            fused = reciprocal_rank_fusion(
                local_union, view_columns, pool_size=pool_size,
            ).selected
            found = target.intersection(fused.episode_id.astype(str))
            recalls[str(pool_size)] = len(found) / len(target) if target else 0.0
        accepted = recalls[str(acceptance_pool)] >= minimum_recall
        if not accepted:
            failures.append(
                f"{case.case_id}: recall {recalls[str(acceptance_pool)]:.3f} "
                f"below {minimum_recall:.3f} at pool {acceptance_pool}"
            )
        case_rows.append({
            "case_id": case.case_id,
            "candidates": len(ranking),
            "local_union_candidates": len(local_union),
            "oracle_matches": len(target),
            "pool_recall": recalls,
            "accepted": accepted,
            "seconds": perf_counter() - case_started,
        })

    cases = pd.DataFrame(case_rows)
    metrics: dict[str, object] = {
        "dataset": str(source.instruments()[0].dataset_id),
        "cases": len(cases),
        "cases_passed": int(cases.accepted.sum()) if len(cases) else 0,
        "candidate_views": list(VIEW_NAMES),
        "per_instrument_per_view": per_instrument_view,
        "pool_sizes": sorted(set(pool_sizes + (acceptance_pool,))),
        "acceptance_pool": acceptance_pool,
        "minimum_recall": minimum_recall,
        "minimum_observed_recall": float(min(
            row[str(acceptance_pool)] for row in cases.pool_recall
        )) if len(cases) else 0.0,
        "seconds": perf_counter() - started,
    }
    return FusionVerification(bool(len(cases)) and not failures, metrics, tuple(failures), cases)


def write_fusion_report(result: FusionVerification, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "".join(
        f"<tr><td>{escape(str(row.case_id))}</td><td>{'PASS' if row.accepted else 'FAIL'}</td>"
        f"<td>{int(row.candidates)}</td><td>{int(row.local_union_candidates)}</td>"
        f"<td><code>{escape(json.dumps(row.pool_recall))}</code></td>"
        f"<td>{float(row.seconds):.2f}</td></tr>"
        for row in result.cases.itertuples(index=False)
    )
    failures = "".join(f"<li>{escape(value)}</li>" for value in result.failures) or "<li>None</li>"
    status = "PASS" if result.passed else "FAIL"
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Multi-view candidate verification</title><style>body{{font-family:system-ui,sans-serif;max-width:1300px;margin:2rem auto;padding:0 1rem;background:#f5f7f8;color:#17202a}}section,header{{background:white;border:1px solid #ddd;border-radius:10px;padding:1.3rem;margin:1rem 0}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.6rem;text-align:left;border-bottom:1px solid #ddd}}.pass{{color:#117864}}.fail{{color:#b03a2e}}</style></head><body><header><h1>Cheap multi-view candidate union: <span class="{status.lower()}">{status}</span></h1><p>Independent price, stage, candle/volatility, volume/shock, market-context and structural views are computed without exact DTW. Per-instrument view winners are fused deterministically and compared with the exhaustive, overlap-deduplicated Gate 09 oracle.</p></header><section><h2>Metrics</h2><pre>{escape(json.dumps(result.metrics, indent=2))}</pre></section><section><h2>Cases</h2><table><thead><tr><th>Case</th><th>Status</th><th>Oracle candidates</th><th>Local union</th><th>Pool recall</th><th>Seconds</th></tr></thead><tbody>{rows}</tbody></table></section><section><h2>Failures</h2><ul>{failures}</ul></section></body></html>"""
    path.write_text(html)
    return path
