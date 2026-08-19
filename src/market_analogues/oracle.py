from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from hashlib import sha256
from html import escape
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .adapters import OHLCVSource
from .episodes import build_episode
from .fusion import oracle_pool_recall
from .scan import _quality_issues
from .search import SearchCandidate, latest_eligible_cutoff, score_candidates, select_scored
from .types import Episode, InstrumentKey, SearchQuery


@dataclass(frozen=True)
class OracleCaseResult:
    case_id: str
    passed: bool
    metrics: dict[str, object]
    failures: tuple[str, ...]
    ranking: pd.DataFrame


@dataclass(frozen=True)
class OracleSuiteResult:
    passed: bool
    metrics: dict[str, object]
    failures: tuple[str, ...]
    cases: tuple[OracleCaseResult, ...]
    liquidity: pd.DataFrame


def liquidity_strata(
    source: OHLCVSource,
    quality: pd.DataFrame,
    *,
    workers: int = 1,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Measure recent dollar-volume only to stratify the verification sample."""
    tiers = quality.set_index("symbol").tier.astype(str).to_dict()
    instruments = [
        key for key in source.instruments() if tiers.get(key.source_symbol, "A") in {"A", "B"}
    ]

    def measure(key: InstrumentKey) -> tuple[dict[str, object] | None, str | None]:
        try:
            bars = source.load(key).tail(252)
            dollar_volume = bars.close.astype(float) * bars.volume.astype(float)
            value = float(dollar_volume.replace([np.inf, -np.inf], np.nan).median())
            return {
                "dataset_id": key.dataset_id,
                "symbol": key.source_symbol,
                "quality_tier": tiers.get(key.source_symbol, "A"),
                "rows": len(source.load(key)),
                "median_dollar_volume_252": value,
            }, None
        except Exception as exc:
            return None, f"{key}:{type(exc).__name__}:{exc}"

    if workers == 1:
        measured = map(measure, instruments)
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        measured = executor.map(measure, instruments)
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    try:
        for row, failure in measured:
            if failure:
                failures.append(failure)
            elif row is not None:
                rows.append(row)
    finally:
        if workers != 1:
            executor.shutdown(wait=True)
    frame = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)
    if len(frame):
        percentile = frame.median_dollar_volume_252.rank(method="first", pct=True)
        frame["liquidity_stratum"] = np.select(
            [percentile <= 1 / 3, percentile <= 2 / 3], ["low", "middle"], default="high",
        )
    return frame, tuple(failures)


def select_stratified_instruments(
    liquidity: pd.DataFrame,
    *,
    per_stratum: int = 2,
    excluded_symbols: set[str] | None = None,
    seed: str = "oracle-v1",
) -> list[InstrumentKey]:
    excluded_symbols = excluded_symbols or set()
    selected: list[InstrumentKey] = []
    for (tier, stratum), group in liquidity.groupby(
        ["quality_tier", "liquidity_stratum"], sort=True,
    ):
        candidates = group[~group.symbol.astype(str).isin(excluded_symbols)].copy()
        candidates["order"] = candidates.symbol.astype(str).map(
            lambda symbol: sha256(f"{seed}:{tier}:{stratum}:{symbol}".encode()).hexdigest()
        )
        for row in candidates.sort_values("order").head(per_stratum).itertuples(index=False):
            selected.append(InstrumentKey(str(row.dataset_id), str(row.symbol)))
    return selected


def _coarse_path_distance(query: Episode, candidate: Episode) -> float:
    x = np.log(query.bars.close.astype(float).clip(lower=1e-12).to_numpy())
    y = np.log(candidate.bars.close.astype(float).clip(lower=1e-12).to_numpy())
    x, y = x - x[0], y - y[0]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        correlation = 0.0 if np.allclose(x, y) else 2.0
    else:
        correlation = float(1 - np.corrcoef(x, y)[0, 1])
    magnitude = float(np.sqrt(np.mean((x - y) ** 2)))
    return .65 * correlation + .35 * magnitude


def _sample_episodes(
    query: Episode,
    source: OHLCVSource,
    instruments: list[InstrumentKey],
    qmap: dict[str, object],
    *,
    windows_per_instrument: int,
) -> tuple[list[Episode], list[str]]:
    latest = latest_eligible_cutoff(query, 60)
    episodes: list[Episode] = []
    failures: list[str] = []
    for instrument in instruments:
        try:
            bars = source.load(instrument)
            eligible = bars[bars.timestamp <= latest]
            available = len(eligible) - query.key.lookback + 1
            if available <= 0:
                continue
            count = min(windows_per_instrument, available)
            ends = np.unique(np.linspace(
                query.key.lookback - 1, len(eligible) - 1, count, dtype=int,
            ))
            record = qmap.get(instrument.source_symbol)
            tier = str(record.tier) if record is not None else "A"
            for end in ends:
                episodes.append(build_episode(
                    source, instrument, eligible.timestamp.iloc[end], query.key.lookback,
                    query.key.representation_version, tier, _quality_issues(record),
                ))
        except Exception as exc:
            failures.append(f"{instrument}:{type(exc).__name__}:{exc}")
    return episodes, failures


def run_oracle_case(
    query: Episode,
    source: OHLCVSource,
    quality: pd.DataFrame,
    instruments: list[InstrumentKey],
    *,
    windows_per_instrument: int = 16,
    top_k: int = 20,
    coarse_pools: tuple[int, ...] = (25, 50, 100),
    minimum_candidates: int = 100,
) -> OracleCaseResult:
    started = perf_counter()
    qmap = {str(row.symbol): row for row in quality.itertuples(index=False)}
    episodes, failures = _sample_episodes(
        query, source, instruments, qmap,
        windows_per_instrument=windows_per_instrument,
    )
    candidates = [SearchCandidate.from_episode(episode) for episode in episodes]
    request = SearchQuery(
        query.key, (query.key.instrument.dataset_id,), ("A", "B"), top_k,
        minimum_history_gap_bars=60,
    )
    scored = score_candidates(query, candidates, request)
    oracle = select_scored(scored, request)
    oracle_ids = {match.episode_key.id for match in oracle}
    oracle_ranks = {match.episode_key.id: rank for rank, match in enumerate(oracle, 1)}
    coarse = [_coarse_path_distance(query, item.episode) for item in scored]
    coarse_order = np.argsort(coarse, kind="stable")
    recalls: dict[str, float] = {}
    for pool in sorted(set(coarse_pools)):
        subset = [scored[position] for position in coarse_order[:pool]]
        matches = select_scored(subset, request)
        ids = {match.episode_key.id for match in matches}
        recalls[str(pool)] = len(ids.intersection(oracle_ids)) / len(oracle_ids) if oracle_ids else 0.0

    rows = []
    coarse_rank = np.empty(len(coarse_order), dtype=int)
    coarse_rank[coarse_order] = np.arange(1, len(coarse_order) + 1)
    for position, item in enumerate(scored):
        rows.append({
            "episode_id": item.episode.key.id,
            "dataset_id": item.episode.key.instrument.dataset_id,
            "symbol": item.episode.key.instrument.source_symbol,
            "cutoff": item.episode.key.cutoff,
            "lookback": item.episode.key.lookback,
            "quality_tier": item.episode.quality_tier,
            "coarse_path_distance": coarse[position],
            "coarse_rank": int(coarse_rank[position]),
            "exact_distance": item.match.total_distance,
            "oracle_selected": item.episode.key.id in oracle_ids,
            "oracle_rank": oracle_ranks.get(item.episode.key.id),
            **{f"component_{key}": value for key, value in item.match.component_distances.items()},
        })
    ranking = pd.DataFrame(rows).sort_values(
        ["exact_distance", "episode_id"], ignore_index=True,
    )
    digest_columns = ["episode_id", "coarse_rank", "exact_distance"]
    digest_payload = ranking[digest_columns].round({"exact_distance": 12}).to_json(
        orient="records", date_format="iso",
    ) if len(ranking) else "[]"
    digest = sha256(digest_payload.encode()).hexdigest()
    component_views = tuple(
        f"component_{name}" for name in (
            "coarse", "stage", "price", "candle_volatility", "volume_shock",
            "market_context", "structural",
        )
    )
    fusion_recalls = oracle_pool_recall(
        ranking, component_views, tuple(sorted(set(coarse_pools + (50,)))),
    ) if len(ranking) else {}
    latest = latest_eligible_cutoff(query, 60)
    if len(scored) < minimum_candidates:
        failures.append(f"only {len(scored)} candidates; require {minimum_candidates}")
    if len(oracle) < top_k:
        failures.append(f"only {len(oracle)} deduplicated oracle matches; require top {top_k}")
    if ranking.episode_id.duplicated().any() if len(ranking) else False:
        failures.append("duplicate episode IDs in oracle")
    if len(ranking) and not np.isfinite(ranking.exact_distance).all():
        failures.append("non-finite exact distances")
    if len(ranking) and (pd.to_datetime(ranking.cutoff) > latest).any():
        failures.append("oracle contains a future/history-gap violation")
    case_id = (
        f"{query.key.instrument.dataset_id}-{query.key.instrument.source_symbol}-"
        f"{query.key.cutoff.date()}-{query.key.lookback}"
    )
    metrics: dict[str, object] = {
        "query": str(query.key.instrument),
        "cutoff": query.key.cutoff.isoformat(),
        "lookback": query.key.lookback,
        "candidate_instruments": len(instruments),
        "candidates_scored": len(scored),
        "top_k": len(oracle),
        "coarse_pool_recall": recalls,
        # This is an architectural upper bound: these component distances are
        # exact and therefore too expensive to compute for every universe window.
        "exact_component_fusion_upper_bound_recall": fusion_recalls,
        "oracle_digest": digest,
        "seconds": perf_counter() - started,
    }
    return OracleCaseResult(case_id, not failures, metrics, tuple(failures), ranking)


def run_oracle_suite(
    source: OHLCVSource,
    quality: pd.DataFrame,
    symbols: tuple[str, ...],
    *,
    lookbacks: tuple[int, ...] = (63, 126, 252),
    cutoff_quantiles: tuple[float, ...] = (.75, 1.0),
    per_stratum: int = 2,
    windows_per_instrument: int = 16,
    top_k: int = 20,
    coarse_pools: tuple[int, ...] = (25, 50, 100),
    workers: int = 1,
    minimum_candidates: int = 100,
    representation_version: str = "dense-v1",
) -> OracleSuiteResult:
    started = perf_counter()
    liquidity, liquidity_failures = liquidity_strata(source, quality, workers=workers)
    instruments = select_stratified_instruments(
        liquidity, per_stratum=per_stratum, excluded_symbols=set(symbols),
    )
    cases: list[OracleCaseResult] = []
    failures = list(liquidity_failures)
    for symbol in symbols:
        bars = source.load(InstrumentKey(source.instruments()[0].dataset_id, symbol))
        minimum = max(lookbacks) - 1
        for quantile in cutoff_quantiles:
            if not 0 <= quantile <= 1:
                failures.append(f"invalid cutoff quantile {quantile}")
                continue
            position = int(round(minimum + quantile * (len(bars) - 1 - minimum)))
            cutoff = bars.timestamp.iloc[position]
            for lookback in lookbacks:
                query = build_episode(
                    source, InstrumentKey(source.instruments()[0].dataset_id, symbol),
                    cutoff, lookback, representation_version,
                )
                case = run_oracle_case(
                    query, source, quality, instruments,
                    windows_per_instrument=windows_per_instrument, top_k=top_k,
                    coarse_pools=coarse_pools, minimum_candidates=minimum_candidates,
                )
                cases.append(case)
                failures.extend(f"{case.case_id}:{failure}" for failure in case.failures)
    metrics: dict[str, object] = {
        "dataset": source.instruments()[0].dataset_id,
        "query_symbols": list(symbols),
        "lookbacks": list(lookbacks),
        "cutoff_quantiles": list(cutoff_quantiles),
        "liquidity_instruments_measured": len(liquidity),
        "stratified_candidate_instruments": len(instruments),
        "cases": len(cases),
        "cases_passed": sum(case.passed for case in cases),
        "total_candidates_scored": sum(int(case.metrics["candidates_scored"]) for case in cases),
        "seconds": perf_counter() - started,
    }
    return OracleSuiteResult(not failures and bool(cases), metrics, tuple(failures), tuple(cases), liquidity)


def write_oracle_artifacts(result: OracleSuiteResult, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    result.liquidity.to_parquet(directory / "liquidity-strata.parquet", index=False)
    summary_rows = []
    for case in result.cases:
        case.ranking.to_parquet(directory / f"{case.case_id}.parquet", index=False)
        summary_rows.append({
            "case_id": case.case_id, "passed": case.passed,
            **case.metrics, "failures": ";".join(case.failures),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_parquet(directory / "oracle-summary.parquet", index=False)
    rows = "".join(
        f"<tr><td>{escape(str(row.case_id))}</td><td>{'PASS' if row.passed else 'FAIL'}</td>"
        f"<td>{int(row.candidates_scored)}</td><td><code>{escape(json.dumps(row.coarse_pool_recall))}</code></td>"
        f"<td>{float(row.seconds):.2f}</td><td><code>{escape(str(row.oracle_digest))}</code></td></tr>"
        for row in summary.itertuples(index=False)
    )
    failures = "".join(f"<li>{escape(item)}</li>" for item in result.failures) or "<li>None</li>"
    status = "PASS" if result.passed else "FAIL"
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exhaustive sampled oracle</title><style>body{{font-family:system-ui,sans-serif;max-width:1300px;margin:2rem auto;padding:0 1rem;background:#f5f7f8;color:#17202a}}section,header{{background:white;border:1px solid #ddd;border-radius:10px;padding:1.3rem;margin:1rem 0}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.6rem;text-align:left;border-bottom:1px solid #ddd}}code{{overflow-wrap:anywhere}}.pass{{color:#117864}}.fail{{color:#b03a2e}}</style></head><body><header><h1>Stratified exhaustive sampled oracle: <span class="{status.lower()}">{status}</span></h1><p>Every candidate in each deterministic market/quality/liquidity/time sample is scored with the exact composite distance. Coarse recall here is measured against that exhaustive sampled ranking.</p></header><section><h2>Suite metrics</h2><pre>{escape(json.dumps(result.metrics, indent=2))}</pre></section><section><h2>Cases</h2><table><thead><tr><th>Case</th><th>Status</th><th>Candidates</th><th>Coarse pool recall</th><th>Seconds</th><th>Digest</th></tr></thead><tbody>{rows}</tbody></table></section><section><h2>Failures</h2><ul>{failures}</ul></section></body></html>"""
    report = directory / "oracle-report.html"
    report.write_text(html)
    return report
