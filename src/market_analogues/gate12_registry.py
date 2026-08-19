from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from html import escape
import json
from pathlib import Path

import pandas as pd
import yaml

from .adapters import OHLCVSource
from .types import EpisodeKey, InstrumentKey, stable_hash


REGISTRY_SCHEMA_VERSION = "gate12-query-registry-v2"
LIQUIDITY_STRATA = ("high", "low", "middle")
QUALITY_TIERS = ("A", "B")


@dataclass(frozen=True)
class Gate12Registry:
    passed: bool
    metrics: dict[str, object]
    failures: tuple[str, ...]
    cases: pd.DataFrame
    excluded_symbols: tuple[str, ...]


def _exclusions(oracle_directory: Path) -> tuple[tuple[str, ...], str]:
    summary_path = oracle_directory / "oracle-summary.parquet"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = pd.read_parquet(summary_path)
    excluded = {str(value).split(":", 1)[-1] for value in summary["query"]}
    for case_id in summary.case_id.astype(str):
        ranking_path = oracle_directory / f"{case_id}.parquet"
        ranking = pd.read_parquet(ranking_path, columns=["symbol"])
        excluded.update(ranking.symbol.astype(str))
    ordered = tuple(sorted(excluded))
    return ordered, stable_hash(list(ordered))


def build_gate12_registry(
    source: OHLCVSource,
    oracle_directory: Path,
    *,
    seed: str = "gate12-query-v1",
    lookback: int = 252,
    historical_quantile: float = .70,
    minimum_rows: int = 1000,
    minimum_future_sessions: int = 60,
    representation_version: str = "dense-v1",
    quality: pd.DataFrame | None = None,
    maximum_staleness_days: int = 120,
) -> Gate12Registry:
    if not 0 < historical_quantile < 1:
        raise ValueError("historical_quantile must be strictly between zero and one")
    if minimum_rows < lookback + minimum_future_sessions:
        raise ValueError("minimum_rows cannot support the requested episode and future gap")
    if maximum_staleness_days < 0:
        raise ValueError("maximum_staleness_days cannot be negative")
    dataset_id = source.instruments()[0].dataset_id
    excluded, exclusion_digest = _exclusions(oracle_directory)
    liquidity_path = oracle_directory / "liquidity-strata.parquet"
    if not liquidity_path.exists():
        raise FileNotFoundError(liquidity_path)
    liquidity = pd.read_parquet(liquidity_path)
    if quality is None:
        current_rows = []
        for row in liquidity.itertuples(index=False):
            instrument = InstrumentKey(dataset_id, str(row.symbol))
            bars = source.load(instrument)
            current_rows.append({
                "symbol": str(row.symbol),
                "current_last_timestamp": pd.Timestamp(bars.timestamp.iloc[-1]),
            })
        freshness = pd.DataFrame(current_rows)
    else:
        required = {"symbol", "last_timestamp"}
        missing = required.difference(quality.columns)
        if missing:
            raise ValueError(f"quality audit is missing freshness columns: {sorted(missing)}")
        freshness = quality[["symbol", "last_timestamp"]].rename(
            columns={"last_timestamp": "current_last_timestamp"},
        ).copy()
        freshness["symbol"] = freshness.symbol.astype(str)
        freshness["current_last_timestamp"] = pd.to_datetime(
            freshness.current_last_timestamp,
        )
    liquidity = liquidity.merge(freshness, on="symbol", how="left", validate="one_to_one")
    market_latest_timestamp = pd.Timestamp(liquidity.current_last_timestamp.max())
    freshness_floor = market_latest_timestamp - pd.Timedelta(days=maximum_staleness_days)
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
        .head(1)
        .sort_values(["quality_tier", "liquidity_stratum"], kind="stable")
    )
    failures: list[str] = []
    expected_cells = {
        (tier, stratum) for tier in QUALITY_TIERS for stratum in LIQUIDITY_STRATA
    }
    actual_cells = {
        (str(row.quality_tier), str(row.liquidity_stratum))
        for row in selected.itertuples(index=False)
    }
    for cell in sorted(expected_cells.difference(actual_cells)):
        failures.append(f"no eligible symbol for quality/liquidity cell {cell}")

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
        cutoffs = (
            ("historical", historical_position),
            ("current", len(bars) - 1),
        )
        for cutoff_role, position in cutoffs:
            future_sessions = len(bars) - position - 1
            if cutoff_role == "historical" and future_sessions < minimum_future_sessions:
                failures.append(
                    f"{instrument}: historical cutoff has only {future_sessions} future sessions"
                )
                continue
            cutoff = pd.Timestamp(bars.timestamp.iloc[position])
            episode_key = EpisodeKey(
                instrument, cutoff, lookback, representation_version,
            )
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
                "episode_id": episode_key.id,
                "source_fingerprint": source_fingerprint,
                "benchmark_fingerprint": benchmark_fingerprint,
                "selection_hash": str(selection.selection_hash),
            })
    cases = pd.DataFrame(rows).sort_values(
        ["quality_tier", "liquidity_stratum", "symbol", "cutoff_role"],
        kind="stable", ignore_index=True,
    ) if rows else pd.DataFrame()
    if len(cases) != 12:
        failures.append(f"created {len(cases)} cases; require 12")
    if len(cases) and cases.case_id.duplicated().any():
        failures.append("registry contains duplicate case IDs")
    case_payload = cases.to_dict(orient="records") if len(cases) else []
    registry_digest = stable_hash({
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "seed": seed,
        "historical_quantile": historical_quantile,
        "minimum_rows": minimum_rows,
        "minimum_future_sessions": minimum_future_sessions,
        "representation_version": representation_version,
        "maximum_staleness_days": maximum_staleness_days,
        "market_latest_timestamp": market_latest_timestamp.isoformat(),
        "exclusion_digest": exclusion_digest,
        "cases": case_payload,
    })
    metrics: dict[str, object] = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "dataset": dataset_id,
        "seed": seed,
        "lookback": lookback,
        "historical_quantile": historical_quantile,
        "minimum_rows": minimum_rows,
        "minimum_future_sessions": minimum_future_sessions,
        "representation_version": representation_version,
        "maximum_staleness_days": maximum_staleness_days,
        "market_latest_timestamp": market_latest_timestamp.isoformat(),
        "excluded_symbols": len(excluded),
        "exclusion_digest": exclusion_digest,
        "selected_symbols": sorted(cases.symbol.unique().tolist()) if len(cases) else [],
        "cases": len(cases),
        "registry_digest": registry_digest,
        "benchmark_fingerprint": benchmark_fingerprint,
    }
    return Gate12Registry(not failures and len(cases) == 12, metrics, tuple(failures), cases, excluded)


def write_gate12_registry(result: Gate12Registry, directory: Path) -> tuple[Path, Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    parquet_path = directory / "query-registry.parquet"
    yaml_path = directory / "query-registry.yaml"
    html_path = directory / "query-registry.html"
    result.cases.to_parquet(parquet_path, index=False)
    payload = {
        **result.metrics,
        "passed": result.passed,
        "failures": list(result.failures),
        "excluded_symbols_list": list(result.excluded_symbols),
        "cases_data": result.cases.to_dict(orient="records"),
    }
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    rows = "".join(
        f"<tr><td>{escape(str(row.case_id))}</td><td>{escape(str(row.quality_tier))} / "
        f"{escape(str(row.liquidity_stratum))}</td><td>{escape(str(row.cutoff))}</td>"
        f"<td>{int(row.future_sessions_at_lock)}</td><td><code>"
        f"{escape(str(row.source_fingerprint))}</code></td></tr>"
        for row in result.cases.itertuples(index=False)
    )
    failure_items = "".join(
        f"<li>{escape(value)}</li>" for value in result.failures
    ) or "<li>None</li>"
    status = "PASS" if result.passed else "FAIL"
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gate 12 independent query registry</title><style>body{{font-family:system-ui,sans-serif;max-width:1400px;margin:2rem auto;padding:0 1rem;background:#f5f7f8;color:#17202a}}section,header{{background:white;border:1px solid #ddd;border-radius:10px;padding:1.3rem;margin:1rem 0}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.55rem;text-align:left;border-bottom:1px solid #ddd}}.pass{{color:#117864}}.fail{{color:#b03a2e}}pre{{overflow:auto}}code{{overflow-wrap:anywhere}}</style></head><body><header><h1>Gate 12 independent registry: <span class="{status.lower()}">{status}</span></h1><p>Queries are deterministically selected from quality/liquidity strata after excluding every Gate 09 query and sampled-candidate symbol. Cutoffs and data fingerprints are frozen before recall measurement.</p></header><section><h2>Registry metrics</h2><pre>{escape(json.dumps(result.metrics, indent=2))}</pre></section><section><h2>Cases</h2><table><thead><tr><th>Case</th><th>Cell</th><th>Cutoff</th><th>Future sessions</th><th>Source fingerprint</th></tr></thead><tbody>{rows}</tbody></table></section><section><h2>Failures</h2><ul>{failure_items}</ul></section></body></html>"""
    html_path.write_text(html)
    return yaml_path, parquet_path, html_path


def validate_gate12_registry(
    source: OHLCVSource,
    yaml_path: Path,
    quality: pd.DataFrame | None = None,
) -> tuple[str, ...]:
    payload = yaml.safe_load(yaml_path.read_text()) or {}
    failures: list[str] = []
    if payload.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        failures.append("registry schema version is unsupported")
    if payload.get("benchmark_fingerprint") != source.benchmark_fingerprint():
        failures.append("registry benchmark fingerprint is stale")
    cases = payload.get("cases_data")
    if not isinstance(cases, list):
        return tuple(failures + ["registry has no case list"])
    digest_payload = {
        key: payload.get(key) for key in (
            "schema_version", "seed", "historical_quantile", "minimum_rows",
            "minimum_future_sessions", "representation_version",
            "maximum_staleness_days", "market_latest_timestamp",
            "exclusion_digest",
        )
    }
    digest_payload["cases"] = cases
    if stable_hash(digest_payload) != payload.get("registry_digest"):
        failures.append("registry digest mismatch")
    excluded = payload.get("excluded_symbols_list")
    if not isinstance(excluded, list) or stable_hash(sorted(excluded)) != payload.get(
        "exclusion_digest"
    ):
        failures.append("registry exclusion digest mismatch")
    if len(cases) != 12:
        failures.append(f"registry has {len(cases)} cases; require 12")
    quality_map = (
        {str(row.symbol): row for row in quality.itertuples(index=False)}
        if quality is not None else {}
    )
    validated: set[tuple[str, str]] = set()
    for case in cases:
        instrument = InstrumentKey(str(case["dataset_id"]), str(case["symbol"]))
        key = (instrument.source_symbol, str(case["source_fingerprint"]))
        if key not in validated:
            validated.add(key)
            if source.fingerprint(instrument) != case["source_fingerprint"]:
                failures.append(f"registry source fingerprint is stale for {instrument}")
            quality_record = quality_map.get(instrument.source_symbol)
            if quality_record is not None and str(quality_record.tier) != str(
                case["quality_tier"]
            ):
                failures.append(f"registry quality tier is stale for {instrument}")
        bars = source.load(instrument)
        position = int(case["cutoff_position"])
        if not 0 <= position < len(bars):
            failures.append(f"registry cutoff position is invalid for {case['case_id']}")
            continue
        cutoff = pd.Timestamp(case["cutoff"])
        if pd.Timestamp(bars.timestamp.iloc[position]) != cutoff:
            failures.append(f"registry cutoff is stale for {case['case_id']}")
        episode_key = EpisodeKey(
            instrument, cutoff, int(case["lookback"]),
            str(case["representation_version"]),
        )
        if episode_key.id != case["episode_id"]:
            failures.append(f"registry episode ID mismatch for {case['case_id']}")
    return tuple(failures)
