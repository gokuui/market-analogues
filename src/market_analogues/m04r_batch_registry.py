from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from html import escape
import json
from pathlib import Path

import pandas as pd

from .adapters import OHLCVSource
from .gate12_registry import LIQUIDITY_STRATA, QUALITY_TIERS, _exclusions
from .types import EpisodeKey, InstrumentKey, stable_hash


SCHEMA_VERSION = "m04r-batch-query-registry-v1"
SYMBOLS_PER_CELL = 2
EXPECTED_SYMBOLS = len(QUALITY_TIERS) * len(LIQUIDITY_STRATA) * SYMBOLS_PER_CELL
EXPECTED_CASES = EXPECTED_SYMBOLS * 2


@dataclass(frozen=True)
class M04RBatchRegistry:
    passed: bool
    metrics: dict[str, object]
    failures: tuple[str, ...]
    cases: pd.DataFrame
    excluded_symbols: tuple[str, ...]


def _digest_payload(metrics: dict[str, object], cases: list[dict[str, object]]) -> dict[str, object]:
    return {
        key: metrics[key] for key in (
            "schema_version", "dataset", "seed", "symbols_per_cell",
            "lookback", "historical_quantile", "minimum_rows",
            "minimum_future_sessions", "representation_version",
            "maximum_staleness_days", "market_latest_timestamp",
            "exclusion_digest", "benchmark_fingerprint",
        )
    } | {"cases": cases, "real_forward_outcomes_accessed": False}


def build_m04r_batch_registry(
    source: OHLCVSource,
    oracle_directory: Path,
    quality: pd.DataFrame,
    *,
    seed: str = "gate12-query-v1",
    lookback: int = 252,
    historical_quantile: float = .70,
    minimum_rows: int = 1000,
    minimum_future_sessions: int = 60,
    representation_version: str = "dense-v1",
    maximum_staleness_days: int = 120,
) -> M04RBatchRegistry:
    if not 0 < historical_quantile < 1:
        raise ValueError("historical_quantile must be strictly between zero and one")
    if minimum_rows < lookback + minimum_future_sessions:
        raise ValueError("minimum_rows cannot support the requested episode and future gap")
    if maximum_staleness_days < 0:
        raise ValueError("maximum_staleness_days cannot be negative")
    required_quality = {"symbol", "last_timestamp"}
    missing_quality = required_quality.difference(quality.columns)
    if missing_quality:
        raise ValueError(f"quality audit is missing columns: {sorted(missing_quality)}")

    instruments = source.instruments()
    if not instruments:
        raise ValueError("source has no instruments")
    dataset_id = instruments[0].dataset_id
    excluded, exclusion_digest = _exclusions(oracle_directory)
    liquidity_path = oracle_directory / "liquidity-strata.parquet"
    if not liquidity_path.exists():
        raise FileNotFoundError(liquidity_path)
    liquidity = pd.read_parquet(liquidity_path)
    freshness = quality[["symbol", "last_timestamp"]].rename(
        columns={"last_timestamp": "current_last_timestamp"},
    ).copy()
    freshness["symbol"] = freshness.symbol.astype(str)
    freshness["current_last_timestamp"] = pd.to_datetime(
        freshness.current_last_timestamp,
    )
    liquidity = liquidity.merge(
        freshness, on="symbol", how="left", validate="one_to_one",
    )
    market_latest = pd.Timestamp(liquidity.current_last_timestamp.max())
    freshness_floor = market_latest - pd.Timedelta(days=maximum_staleness_days)
    eligible = liquidity[
        liquidity.quality_tier.astype(str).isin(QUALITY_TIERS)
        & liquidity.liquidity_stratum.astype(str).isin(LIQUIDITY_STRATA)
        & (liquidity.rows.astype(int) >= minimum_rows)
        & (liquidity.current_last_timestamp >= freshness_floor)
        & ~liquidity.symbol.astype(str).isin(excluded)
    ].copy()
    eligible["selection_hash"] = eligible.apply(
        lambda row: sha256(
            f"{seed}:{dataset_id}:{row.quality_tier}:"
            f"{row.liquidity_stratum}:{row.symbol}".encode()
        ).hexdigest(),
        axis=1,
    )
    selected = (
        eligible.sort_values(["selection_hash", "symbol"], kind="stable")
        .groupby(["quality_tier", "liquidity_stratum"], sort=True)
        .head(SYMBOLS_PER_CELL)
        .sort_values(
            ["quality_tier", "liquidity_stratum", "selection_hash", "symbol"],
            kind="stable",
        )
    )
    failures: list[str] = []
    counts = selected.groupby(["quality_tier", "liquidity_stratum"]).size()
    for tier in QUALITY_TIERS:
        for stratum in LIQUIDITY_STRATA:
            observed = int(counts.get((tier, stratum), 0))
            if observed != SYMBOLS_PER_CELL:
                failures.append(
                    f"quality/liquidity cell {(tier, stratum)} has {observed}; "
                    f"require {SYMBOLS_PER_CELL}"
                )

    benchmark_fingerprint = source.benchmark_fingerprint()
    rows: list[dict[str, object]] = []
    for selection in selected.itertuples(index=False):
        symbol = str(selection.symbol)
        instrument = InstrumentKey(dataset_id, symbol)
        try:
            bars = source.load(instrument)
            source_fingerprint = source.fingerprint(instrument)
        except Exception as exc:
            failures.append(f"{instrument}:{type(exc).__name__}:{exc}")
            continue
        if len(bars) < minimum_rows:
            failures.append(f"{instrument}: current source has only {len(bars)} rows")
            continue
        historical_position = int(round(
            lookback - 1
            + historical_quantile * (len(bars) - 1 - (lookback - 1))
        ))
        for cutoff_role, position in (
            ("historical", historical_position), ("current", len(bars) - 1),
        ):
            future_sessions = len(bars) - position - 1
            if cutoff_role == "historical" and future_sessions < minimum_future_sessions:
                failures.append(
                    f"{instrument}: historical cutoff has only "
                    f"{future_sessions} future sessions"
                )
                continue
            cutoff = pd.Timestamp(bars.timestamp.iloc[position])
            key = EpisodeKey(instrument, cutoff, lookback, representation_version)
            rows.append({
                "case_id": f"{dataset_id}-{symbol}-{cutoff_role}-{lookback}",
                "dataset_id": dataset_id,
                "symbol": symbol,
                "quality_tier": str(selection.quality_tier),
                "liquidity_stratum": str(selection.liquidity_stratum),
                "median_dollar_volume_252": float(selection.median_dollar_volume_252),
                "rows_at_lock": len(bars),
                "cutoff_role": cutoff_role,
                "cutoff": cutoff.isoformat(),
                "cutoff_position": position,
                "future_sessions_at_lock": future_sessions,
                "lookback": lookback,
                "representation_version": representation_version,
                "episode_id": key.id,
                "source_fingerprint": source_fingerprint,
                "benchmark_fingerprint": benchmark_fingerprint,
                "selection_hash": str(selection.selection_hash),
            })
    cases = (
        pd.DataFrame(rows).sort_values(
            ["quality_tier", "liquidity_stratum", "selection_hash", "symbol", "cutoff_role"],
            kind="stable", ignore_index=True,
        ) if rows else pd.DataFrame()
    )
    if len(cases) != EXPECTED_CASES:
        failures.append(f"created {len(cases)} cases; require {EXPECTED_CASES}")
    if len(cases) and (
        cases.case_id.duplicated().any() or cases.episode_id.duplicated().any()
    ):
        failures.append("registry contains duplicate case or episode IDs")
    case_rows = cases.to_dict(orient="records") if len(cases) else []
    metrics: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset_id,
        "seed": seed,
        "symbols_per_cell": SYMBOLS_PER_CELL,
        "lookback": lookback,
        "historical_quantile": historical_quantile,
        "minimum_rows": minimum_rows,
        "minimum_future_sessions": minimum_future_sessions,
        "representation_version": representation_version,
        "maximum_staleness_days": maximum_staleness_days,
        "market_latest_timestamp": market_latest.isoformat(),
        "excluded_symbols": len(excluded),
        "exclusion_digest": exclusion_digest,
        "selected_symbols": sorted(cases.symbol.unique().tolist()) if len(cases) else [],
        "cases": len(cases),
        "benchmark_fingerprint": benchmark_fingerprint,
        "real_forward_outcomes_accessed": False,
    }
    metrics["registry_digest"] = stable_hash(_digest_payload(metrics, case_rows))
    return M04RBatchRegistry(
        not failures and len(cases) == EXPECTED_CASES,
        metrics, tuple(failures), cases, excluded,
    )


def write_m04r_batch_registry(
    result: M04RBatchRegistry, directory: Path,
) -> tuple[Path, Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "query-registry.json"
    parquet_path = directory / "query-registry.parquet"
    html_path = directory / "query-registry.html"
    result.cases.to_parquet(parquet_path, index=False)
    payload = {
        **result.metrics,
        "passed": result.passed,
        "failures": list(result.failures),
        "excluded_symbols_list": list(result.excluded_symbols),
        "cases_data": result.cases.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    rows = "".join(
        f"<tr><td>{escape(str(row.case_id))}</td><td>{escape(str(row.quality_tier))} / "
        f"{escape(str(row.liquidity_stratum))}</td><td>{escape(str(row.cutoff_role))}</td>"
        f"<td>{escape(str(row.cutoff))}</td><td><code>{escape(str(row.episode_id))}</code></td></tr>"
        for row in result.cases.itertuples(index=False)
    )
    failures = "".join(f"<li>{escape(value)}</li>" for value in result.failures) or "<li>None</li>"
    status = "PASS" if result.passed else "FAIL"
    html_path.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>M04R 24-query batch registry</title><style>body{{font-family:system-ui,sans-serif;max-width:1400px;margin:2rem auto;padding:0 1rem;background:#f5f7f8;color:#17202a}}section,header{{background:white;border:1px solid #ddd;border-radius:10px;padding:1.3rem;margin:1rem 0}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.55rem;text-align:left;border-bottom:1px solid #ddd}}.pass{{color:#117864}}.fail{{color:#b03a2e}}pre{{overflow:auto}}code{{overflow-wrap:anywhere}}</style></head><body><header><h1>M04R 24-query batch registry: <span class="{status.lower()}">{status}</span></h1><p>Two hash-selected symbols per A/B quality and low/middle/high liquidity cell; historical/current cutoffs are frozen without outcomes or setup labels.</p></header><section><h2>Registry metrics</h2><pre>{escape(json.dumps(result.metrics, indent=2))}</pre></section><section><h2>Cases</h2><table><thead><tr><th>Case</th><th>Cell</th><th>Role</th><th>Cutoff</th><th>Episode</th></tr></thead><tbody>{rows}</tbody></table></section><section><h2>Failures</h2><ul>{failures}</ul></section></body></html>""")
    return json_path, parquet_path, html_path


def validate_m04r_batch_registry(
    source: OHLCVSource,
    path: Path,
    quality: pd.DataFrame,
    oracle_directory: Path | None = None,
) -> tuple[str, ...]:
    payload = json.loads(path.read_text())
    failures: list[str] = []
    cases = payload.get("cases_data")
    if payload.get("schema_version") != SCHEMA_VERSION:
        failures.append("registry schema version is unsupported")
    if not isinstance(cases, list):
        return tuple(failures + ["registry has no case list"])
    metrics = {key: payload.get(key) for key in _digest_payload(payload, cases) if key != "cases"}
    if stable_hash(_digest_payload(metrics, cases)) != payload.get("registry_digest"):
        failures.append("registry digest mismatch")
    if payload.get("benchmark_fingerprint") != source.benchmark_fingerprint():
        failures.append("registry benchmark fingerprint is stale")
    excluded = payload.get("excluded_symbols_list")
    if not isinstance(excluded, list) or stable_hash(sorted(excluded)) != payload.get("exclusion_digest"):
        failures.append("registry exclusion digest mismatch")
        excluded = []
    if len(cases) != EXPECTED_CASES:
        failures.append(f"registry has {len(cases)} cases; require {EXPECTED_CASES}")
    frame = pd.DataFrame(cases)
    required = {
        "case_id", "dataset_id", "symbol", "quality_tier", "liquidity_stratum",
        "cutoff_role", "cutoff", "lookback", "representation_version",
        "episode_id", "source_fingerprint", "benchmark_fingerprint",
    }
    if required.difference(frame.columns):
        return tuple(failures + ["registry case fields are incomplete"])
    if frame.case_id.duplicated().any() or frame.episode_id.duplicated().any():
        failures.append("registry contains duplicate case or episode IDs")
    if set(frame.symbol.astype(str)).intersection(map(str, excluded)):
        failures.append("registry includes an excluded Gate 09 symbol")
    cells = frame.drop_duplicates("symbol").groupby(
        ["quality_tier", "liquidity_stratum"],
    ).size()
    for tier in QUALITY_TIERS:
        for stratum in LIQUIDITY_STRATA:
            if int(cells.get((tier, stratum), 0)) != SYMBOLS_PER_CELL:
                failures.append(f"registry cell {(tier, stratum)} does not contain two symbols")
    quality_map = {str(row.symbol): row for row in quality.itertuples(index=False)}
    for symbol, group in frame.groupby("symbol", sort=True):
        if set(group.cutoff_role.astype(str)) != {"historical", "current"} or len(group) != 2:
            failures.append(f"{symbol} does not have one historical and one current case")
        instrument = InstrumentKey(str(group.dataset_id.iloc[0]), str(symbol))
        try:
            fingerprint = source.fingerprint(instrument)
        except Exception as exc:
            failures.append(f"{instrument}:{type(exc).__name__}:{exc}")
            continue
        if set(group.source_fingerprint.astype(str)) != {fingerprint}:
            failures.append(f"source fingerprint is stale for {instrument}")
        quality_row = quality_map.get(str(symbol))
        if quality_row is None:
            failures.append(f"quality audit is missing {instrument}")
        elif hasattr(quality_row, "tier") and set(group.quality_tier.astype(str)) != {
            str(quality_row.tier)
        }:
            failures.append(f"quality tier is stale for {instrument}")
        for row in group.itertuples(index=False):
            key = EpisodeKey(
                instrument, pd.Timestamp(row.cutoff), int(row.lookback),
                str(row.representation_version),
            )
            if key.id != str(row.episode_id):
                failures.append(f"episode identity is stale for {row.case_id}")
            if str(row.benchmark_fingerprint) != source.benchmark_fingerprint():
                failures.append(f"benchmark fingerprint is stale for {row.case_id}")
    forbidden = {"outcome", "forward_return", "setup_label", "winner", "loser"}
    if any(forbidden.intersection(case) for case in cases):
        failures.append("registry contains outcome or setup-label fields")
    if payload.get("real_forward_outcomes_accessed") is not False:
        failures.append("registry outcome-access marker differs")
    if oracle_directory is not None:
        try:
            rebuilt = build_m04r_batch_registry(
                source, oracle_directory, quality,
                seed=str(payload.get("seed")),
                lookback=int(payload.get("lookback")),
                historical_quantile=float(payload.get("historical_quantile")),
                minimum_rows=int(payload.get("minimum_rows")),
                minimum_future_sessions=int(payload.get("minimum_future_sessions")),
                representation_version=str(payload.get("representation_version")),
                maximum_staleness_days=int(payload.get("maximum_staleness_days")),
            )
            if (
                not rebuilt.passed
                or stable_hash(rebuilt.cases.to_dict(orient="records"))
                != stable_hash(cases)
            ):
                failures.append("registry deterministic selection differs")
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(f"registry controls cannot be rebuilt: {exc}")
    return tuple(dict.fromkeys(failures))
