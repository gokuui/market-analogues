from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .adapters import OHLCVSource
from .episodes import build_episode
from .search import SearchCandidate, exact_search_pruned
from .types import Episode, EpisodeKey, InstrumentKey, SearchQuery


@dataclass(frozen=True)
class PruningVerification:
    passed: bool
    metrics: dict[str, object]
    failures: tuple[str, ...]
    cases: pd.DataFrame


def _candidate_from_row(
    source: OHLCVSource,
    benchmark: pd.DataFrame | None,
    row: object,
    representation_version: str,
) -> SearchCandidate:
    instrument = InstrumentKey(str(row.dataset_id), str(row.symbol))
    bars = source.load(instrument)
    window = bars[bars.timestamp <= pd.Timestamp(row.cutoff)].tail(
        int(row.lookback)
    ).reset_index(drop=True)
    if len(window) != int(row.lookback):
        raise ValueError(
            f"incomplete candidate window {instrument}:{row.cutoff}:{row.lookback}"
        )
    episode = Episode(
        EpisodeKey(
            instrument, pd.Timestamp(window.timestamp.iloc[-1]), int(row.lookback),
            representation_version,
        ),
        window,
        benchmark,
        str(row.quality_tier),
    )
    return SearchCandidate.from_episode(episode)


def _compare_matches(
    label: str,
    matches: list[object],
    expected: pd.DataFrame,
    *,
    tolerance: float,
) -> tuple[list[str], float, float]:
    failures: list[str] = []
    expected_ids = expected.episode_id.astype(str).tolist()
    actual_ids = [match.episode_key.id for match in matches]
    if actual_ids != expected_ids:
        failures.append(
            f"{label} IDs differ: expected {expected_ids}, received {actual_ids}"
        )

    expected_by_id = expected.set_index("episode_id")
    maximum_distance_delta = 0.0
    maximum_component_delta = 0.0
    component_columns = {
        column.removeprefix("component_"): column
        for column in expected.columns if column.startswith("component_")
    }
    for match in matches:
        episode_id = match.episode_key.id
        if episode_id not in expected_by_id.index:
            continue
        row = expected_by_id.loc[episode_id]
        distance_delta = abs(float(match.total_distance) - float(row.exact_distance))
        maximum_distance_delta = max(maximum_distance_delta, distance_delta)
        if not np.isclose(
            match.total_distance, row.exact_distance, rtol=tolerance, atol=tolerance,
        ):
            failures.append(
                f"{label} distance differs for {episode_id}: "
                f"expected {float(row.exact_distance):.17g}, "
                f"received {float(match.total_distance):.17g}"
            )
        for component, column in component_columns.items():
            if component not in match.component_distances:
                failures.append(f"{label} omitted component {component} for {episode_id}")
                continue
            delta = abs(float(match.component_distances[component]) - float(row[column]))
            maximum_component_delta = max(maximum_component_delta, delta)
            if not np.isclose(
                match.component_distances[component], row[column],
                rtol=tolerance, atol=tolerance,
            ):
                failures.append(
                    f"{label} component {component} differs for {episode_id}: "
                    f"expected {float(row[column]):.17g}, "
                    f"received {float(match.component_distances[component]):.17g}"
                )
    return failures, maximum_distance_delta, maximum_component_delta


def verify_exact_safe_pruning(
    source: OHLCVSource,
    oracle_directory: Path,
    *,
    representation_version: str = "dense-v1",
    tolerance: float = 1e-12,
) -> PruningVerification:
    """Differentially verify both pruned scorers against persisted exact oracles."""
    started = perf_counter()
    summary_path = oracle_directory / "oracle-summary.parquet"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = pd.read_parquet(summary_path)
    benchmark = source.load_benchmark()
    dataset_id = str(source.instruments()[0].dataset_id)
    failures: list[str] = []
    case_rows: list[dict[str, object]] = []

    for case in summary.sort_values("case_id").itertuples(index=False):
        case_started = perf_counter()
        ranking_path = oracle_directory / f"{case.case_id}.parquet"
        ranking = pd.read_parquet(ranking_path)
        query_symbol = str(case.query).split(":", 1)[-1]
        query = build_episode(
            source, InstrumentKey(dataset_id, query_symbol), case.cutoff,
            int(case.lookback), representation_version,
        )
        candidates = [
            _candidate_from_row(source, benchmark, row, representation_version)
            for row in ranking.itertuples(index=False)
        ]
        expected = ranking.loc[ranking.oracle_selected].sort_values(
            ["oracle_rank", "episode_id"], kind="stable",
        )
        request = SearchQuery(
            query.key, (dataset_id,), ("A", "B"), int(case.top_k),
            minimum_history_gap_bars=60,
        )

        default_started = perf_counter()
        default_matches, default_report = exact_search_pruned(
            query, candidates, request, use_dtw_bound=False,
        )
        default_seconds = perf_counter() - default_started
        bound_started = perf_counter()
        bound_matches, bound_report = exact_search_pruned(
            query, candidates, request, use_dtw_bound=True,
        )
        bound_seconds = perf_counter() - bound_started

        case_failures: list[str] = []
        default_failures, default_distance_delta, default_component_delta = (
            _compare_matches("default", default_matches, expected, tolerance=tolerance)
        )
        bound_failures, bound_distance_delta, bound_component_delta = _compare_matches(
            "LB_Keogh", bound_matches, expected, tolerance=tolerance,
        )
        case_failures.extend(default_failures)
        case_failures.extend(bound_failures)
        default_ids = [match.episode_key.id for match in default_matches]
        bound_ids = [match.episode_key.id for match in bound_matches]
        if default_ids != bound_ids:
            case_failures.append("default and LB_Keogh rankings differ")
        if default_report.exact_evaluated > default_report.eligible_candidates:
            case_failures.append("default exact evaluation count exceeds eligible candidates")
        if bound_report.exact_evaluated > default_report.exact_evaluated:
            case_failures.append("LB_Keogh evaluated more exact distances than default")
        failures.extend(f"{case.case_id}:{failure}" for failure in case_failures)
        eligible = default_report.eligible_candidates
        case_rows.append({
            "case_id": str(case.case_id),
            "passed": not case_failures,
            "eligible_candidates": eligible,
            "oracle_matches": len(expected),
            "default_exact_evaluated": default_report.exact_evaluated,
            "default_safely_pruned": default_report.safely_pruned,
            "default_prune_fraction": (
                default_report.safely_pruned / eligible if eligible else 0.0
            ),
            "default_seconds": default_seconds,
            "bound_exact_evaluated": bound_report.exact_evaluated,
            "bound_safely_pruned": bound_report.safely_pruned,
            "bound_prune_fraction": (
                bound_report.safely_pruned / eligible if eligible else 0.0
            ),
            "bound_seconds": bound_seconds,
            "dtw_bounds_evaluated": bound_report.dtw_bounds_evaluated,
            "maximum_distance_delta": max(default_distance_delta, bound_distance_delta),
            "maximum_component_delta": max(default_component_delta, bound_component_delta),
            "reconstruction_seconds": perf_counter() - case_started,
            "failures": "; ".join(case_failures),
        })

    cases = pd.DataFrame(case_rows)
    eligible_total = int(cases.eligible_candidates.sum()) if len(cases) else 0
    default_evaluated = int(cases.default_exact_evaluated.sum()) if len(cases) else 0
    bound_evaluated = int(cases.bound_exact_evaluated.sum()) if len(cases) else 0
    metrics: dict[str, object] = {
        "dataset": dataset_id,
        "oracle_directory": str(oracle_directory),
        "cases": len(cases),
        "cases_passed": int(cases.passed.sum()) if len(cases) else 0,
        "eligible_candidates": eligible_total,
        "default_exact_evaluated": default_evaluated,
        "default_safely_pruned": eligible_total - default_evaluated,
        "default_prune_fraction": (
            (eligible_total - default_evaluated) / eligible_total if eligible_total else 0.0
        ),
        "default_scoring_seconds": float(cases.default_seconds.sum()) if len(cases) else 0.0,
        "bound_exact_evaluated": bound_evaluated,
        "bound_safely_pruned": eligible_total - bound_evaluated,
        "bound_prune_fraction": (
            (eligible_total - bound_evaluated) / eligible_total if eligible_total else 0.0
        ),
        "bound_scoring_seconds": float(cases.bound_seconds.sum()) if len(cases) else 0.0,
        "maximum_distance_delta": float(cases.maximum_distance_delta.max()) if len(cases) else 0.0,
        "maximum_component_delta": float(cases.maximum_component_delta.max()) if len(cases) else 0.0,
        "tolerance": tolerance,
        "seconds": perf_counter() - started,
    }
    return PruningVerification(bool(len(cases)) and not failures, metrics, tuple(failures), cases)


def write_pruning_report(result: PruningVerification, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "".join(
        f"<tr><td>{escape(str(row.case_id))}</td>"
        f"<td>{'PASS' if row.passed else 'FAIL'}</td>"
        f"<td>{int(row.eligible_candidates)}</td><td>{int(row.oracle_matches)}</td>"
        f"<td>{int(row.default_exact_evaluated)} "
        f"({float(row.default_prune_fraction):.1%} pruned)</td>"
        f"<td>{float(row.default_seconds):.3f}</td>"
        f"<td>{int(row.bound_exact_evaluated)} "
        f"({float(row.bound_prune_fraction):.1%} pruned)</td>"
        f"<td>{float(row.bound_seconds):.3f}</td>"
        f"<td>{float(row.maximum_distance_delta):.2e}</td>"
        f"<td>{escape(str(row.failures)) or 'None'}</td></tr>"
        for row in result.cases.itertuples(index=False)
    )
    failure_items = "".join(
        f"<li>{escape(value)}</li>" for value in result.failures
    ) or "<li>None</li>"
    status = "PASS" if result.passed else "FAIL"
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exact-safe pruning verification</title><style>body{{font-family:system-ui,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem;background:#f5f7f8;color:#17202a}}section,header{{background:white;border:1px solid #ddd;border-radius:10px;padding:1.3rem;margin:1rem 0}}table{{border-collapse:collapse;width:100%;font-size:.9rem}}th,td{{padding:.55rem;text-align:left;border-bottom:1px solid #ddd;vertical-align:top}}.pass{{color:#117864}}.fail{{color:#b03a2e}}pre{{overflow:auto}}</style></head><body><header><h1>Exact-safe progressive pruning: <span class="{status.lower()}">{status}</span></h1><p>Both the inexpensive non-DTW lower bound and its optional symmetric multivariate LB_Keogh strengthening are differentially checked against every persisted Gate 09 exact-oracle result. A pass requires identical ordered episode IDs and numerically equal total and component distances.</p></header><section><h2>Aggregate evidence</h2><pre>{escape(json.dumps(result.metrics, indent=2))}</pre></section><section><h2>Cases</h2><table><thead><tr><th>Case</th><th>Status</th><th>Eligible</th><th>Top k</th><th>Default exact work</th><th>Default seconds</th><th>LB exact work</th><th>LB seconds</th><th>Max Δ</th><th>Failures</th></tr></thead><tbody>{rows}</tbody></table></section><section><h2>Failures</h2><ul>{failure_items}</ul></section></body></html>"""
    path.write_text(html)
    return path
