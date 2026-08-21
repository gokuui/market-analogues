from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from html import escape
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import pandas as pd
import yaml

from .adapters import OHLCVSource
from .config import DatasetSpec
from .quality import qualify_frame
from .types import InstrumentKey, stable_hash


AVAILABILITY_SCHEMA_VERSION = "data-availability-v1"
LEDGER_SCHEMA_VERSION = "point-in-time-data-ledger-v1"
CAPABILITY_STATUSES = {"available", "partial", "unavailable", "unknown"}
REQUIRED_CAPABILITIES = {
    "source_ohlcv",
    "benchmark_ohlcv",
    "corporate_action_provenance",
    "point_in_time_listing_membership",
    "delisting_returns",
    "ticker_identity_history",
    "point_in_time_sector_industry",
    "point_in_time_event_calendar",
}
QUALITY_COLUMNS = {
    "dataset_id", "symbol", "source_hash", "rows", "first_timestamp",
    "last_timestamp", "duplicate_timestamps", "missing_required", "invalid_ohlc",
    "nonpositive_prices", "negative_volume", "zero_volume",
    "extreme_discontinuities", "max_gap_multiple", "tier", "issues",
}


class DataLedgerError(ValueError):
    pass


@dataclass(frozen=True)
class AvailabilityDeclaration:
    source: Path
    universe_boundary: str
    datasets: dict[str, dict[str, dict[str, str]]]
    digest: str


@dataclass(frozen=True)
class DataLedgerResult:
    passed: bool
    metrics: dict[str, Any]
    failures: tuple[str, ...]
    instruments: pd.DataFrame


def load_availability_declaration(path: str | Path) -> AvailabilityDeclaration:
    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text()) or {}
    if not isinstance(payload, Mapping):
        raise DataLedgerError("availability declaration must be a mapping")
    required_top = {"schema_version", "universe_boundary", "datasets"}
    if set(payload) != required_top:
        raise DataLedgerError(
            f"availability top-level keys differ; missing={sorted(required_top - set(payload))}, "
            f"unknown={sorted(set(payload) - required_top)}"
        )
    if payload["schema_version"] != AVAILABILITY_SCHEMA_VERSION:
        raise DataLedgerError(f"unsupported availability schema: {payload['schema_version']!r}")
    if payload["universe_boundary"] != "configured_source_instruments_only":
        raise DataLedgerError("universe boundary must be configured_source_instruments_only")
    raw_datasets = payload["datasets"]
    if not isinstance(raw_datasets, Mapping) or not raw_datasets:
        raise DataLedgerError("availability datasets must be a non-empty mapping")
    datasets: dict[str, dict[str, dict[str, str]]] = {}
    for dataset_id, raw in raw_datasets.items():
        if not isinstance(raw, Mapping) or set(raw) != {"capabilities"}:
            raise DataLedgerError(f"availability dataset {dataset_id!r} must contain only capabilities")
        capabilities = raw["capabilities"]
        if not isinstance(capabilities, Mapping) or set(capabilities) != REQUIRED_CAPABILITIES:
            raise DataLedgerError(
                f"dataset {dataset_id!r} capability keys differ; "
                f"missing={sorted(REQUIRED_CAPABILITIES - set(capabilities or {}))}, "
                f"unknown={sorted(set(capabilities or {}) - REQUIRED_CAPABILITIES)}"
            )
        normalized: dict[str, dict[str, str]] = {}
        for name, declaration in capabilities.items():
            if not isinstance(declaration, Mapping) or set(declaration) != {"status", "reason"}:
                raise DataLedgerError(f"capability {dataset_id}.{name} must contain status and reason")
            status = declaration["status"]
            reason = declaration["reason"]
            if status not in CAPABILITY_STATUSES:
                raise DataLedgerError(f"unsupported capability status {dataset_id}.{name}={status!r}")
            if not isinstance(reason, str) or not reason.strip():
                raise DataLedgerError(f"capability {dataset_id}.{name} requires a reason")
            normalized[str(name)] = {"status": str(status), "reason": reason.strip()}
        datasets[str(dataset_id)] = normalized
    canonical = {
        "schema_version": AVAILABILITY_SCHEMA_VERSION,
        "universe_boundary": payload["universe_boundary"],
        "datasets": datasets,
    }
    return AvailabilityDeclaration(source, payload["universe_boundary"], datasets, stable_hash(canonical))


def _fingerprint_one(source: OHLCVSource, key: InstrumentKey) -> tuple[str, str | None]:
    try:
        return key.source_symbol, source.fingerprint(key)
    except Exception as exc:
        return key.source_symbol, f"ERROR:{type(exc).__name__}:{exc}"


def _quality_issue_count(quality: pd.DataFrame, prefix: str) -> int:
    return int(quality["issues"].fillna("").str.split(";").explode().str.startswith(prefix).sum())


def build_data_ledger(
    spec: DatasetSpec,
    source: OHLCVSource,
    quality: pd.DataFrame,
    declaration: AvailabilityDeclaration,
    *,
    workers: int = 4,
    as_of: pd.Timestamp | str | None = None,
) -> DataLedgerResult:
    started = perf_counter()
    if workers < 1:
        raise ValueError("workers must be positive")
    if spec.dataset_id not in declaration.datasets:
        raise DataLedgerError(f"dataset {spec.dataset_id!r} has no availability declaration")
    missing_columns = sorted(QUALITY_COLUMNS - set(quality.columns))
    if missing_columns:
        raise DataLedgerError(f"quality frame missing columns: {missing_columns}")
    quality = quality.copy()
    quality["symbol"] = quality["symbol"].astype(str)
    keys = source.instruments()
    source_symbols = [key.source_symbol for key in keys]
    source_set = set(source_symbols)
    duplicate_source_symbols = len(source_symbols) - len(source_set)
    duplicate_quality_symbols = int(quality["symbol"].duplicated().sum())
    quality_set = set(quality["symbol"])
    missing_quality = sorted(source_set - quality_set)
    extra_quality = sorted(quality_set - source_set)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        fingerprint_pairs = list(pool.map(lambda key: _fingerprint_one(source, key), keys))
    live_hashes = dict(fingerprint_pairs)
    quality_map = quality.set_index("symbol", drop=False).to_dict("index")
    rows: list[dict[str, Any]] = []
    fingerprint_mismatches: list[str] = []
    fingerprint_errors: list[str] = []
    for symbol in sorted(source_set | quality_set):
        record = quality_map.get(symbol)
        present = symbol in source_set
        live_hash = live_hashes.get(symbol)
        recorded_hash = str(record.get("source_hash") or "") if record else ""
        issues = str(record.get("issues") or "") if record else "missing_quality_record"
        load_error = issues.startswith("load_error:")
        if not present:
            fingerprint_status = "source_missing"
        elif live_hash is not None and live_hash.startswith("ERROR:"):
            fingerprint_status = "fingerprint_error"
            fingerprint_errors.append(f"{symbol}:{live_hash}")
        elif load_error and not recorded_hash:
            fingerprint_status = "unverified_load_error"
        elif live_hash == recorded_hash:
            fingerprint_status = "verified"
        else:
            fingerprint_status = "mismatch"
            fingerprint_mismatches.append(symbol)
        tier = str(record.get("tier")) if record else "UNACCOUNTED"
        if not present:
            accounting_status = "quality_record_without_source"
        elif record is None:
            accounting_status = "source_without_quality_record"
        elif load_error:
            accounting_status = "quarantined_load_error"
        elif tier == "QUARANTINED":
            accounting_status = "quarantined_quality"
        elif tier in {"A", "B"}:
            accounting_status = "usable"
        else:
            accounting_status = "invalid_quality_tier"
        row = dict(record or {})
        row.update({
            "dataset_id": spec.dataset_id,
            "symbol": symbol,
            "source_present": present,
            "live_source_hash": live_hash,
            "fingerprint_status": fingerprint_status,
            "accounting_status": accounting_status,
        })
        rows.append(row)
    instruments = pd.DataFrame(rows).sort_values("symbol", kind="stable").reset_index(drop=True)

    capabilities = declaration.datasets[spec.dataset_id]
    source_declared = capabilities["source_ohlcv"]["status"]
    benchmark_declared = capabilities["benchmark_ohlcv"]["status"] == "available"
    benchmark_metrics: dict[str, Any] = {
        "declared_status": capabilities["benchmark_ohlcv"]["status"],
        "configured": spec.benchmark is not None,
    }
    benchmark_failures: list[str] = []
    if spec.benchmark is not None:
        if not benchmark_declared:
            benchmark_failures.append("configured benchmark is not declared available")
        try:
            benchmark = source.load_benchmark()
            if benchmark is None:
                raise ValueError("configured benchmark returned no frame")
            benchmark_record = qualify_frame(
                InstrumentKey(spec.dataset_id, "__benchmark__"), benchmark,
                source.benchmark_fingerprint() or "",
            )
            benchmark_metrics.update(asdict(benchmark_record))
            if benchmark_record.tier == "QUARANTINED":
                benchmark_failures.append("configured benchmark is quarantined")
        except Exception as exc:
            benchmark_failures.append(f"benchmark_load_error:{type(exc).__name__}:{exc}")
    elif benchmark_declared:
        benchmark_failures.append("benchmark declared available but none is configured")

    accounted = int(instruments["accounting_status"].isin({
        "usable", "quarantined_quality", "quarantined_load_error",
    }).sum())
    usable = int((instruments["accounting_status"] == "usable").sum())
    quarantined = int(instruments["accounting_status"].isin({
        "quarantined_quality", "quarantined_load_error",
    }).sum())
    limitations = {
        name: value for name, value in capabilities.items()
        if value["status"] != "available"
    }
    as_of_timestamp = pd.Timestamp.now(tz="UTC").tz_localize(None) if as_of is None else pd.Timestamp(as_of)
    if as_of_timestamp.tzinfo is not None:
        as_of_timestamp = as_of_timestamp.tz_convert("UTC").tz_localize(None)
    first_timestamps = pd.to_datetime(quality["first_timestamp"], errors="coerce")
    last_timestamps = pd.to_datetime(quality["last_timestamp"], errors="coerce")
    first_source_timestamp = first_timestamps.min()
    last_source_timestamp = last_timestamps.max()
    freshness_age_days = (
        int((as_of_timestamp.normalize() - last_source_timestamp.normalize()).days)
        if pd.notna(last_source_timestamp) else None
    )
    if freshness_age_days is None:
        freshness_status = "unavailable"
    elif freshness_age_days < -1:
        freshness_status = "future_timestamp_error"
    elif freshness_age_days <= 7:
        freshness_status = "current_within_7_days"
    elif freshness_age_days <= 31:
        freshness_status = "delayed_8_to_31_days"
    else:
        freshness_status = "historical_snapshot_over_31_days"
    usable_last_timestamps = pd.to_datetime(
        instruments.loc[instruments["accounting_status"] == "usable", "last_timestamp"],
        errors="coerce",
    )
    usable_age_days = (as_of_timestamp.normalize() - usable_last_timestamps.dt.normalize()).dt.days
    current_usable = int(usable_age_days.between(-1, 7, inclusive="both").sum())
    current_coverage_fraction = current_usable / usable if usable else 0.0
    failures: list[str] = []
    if source_declared != "available":
        failures.append("configured source OHLCV is not declared available")
    if duplicate_source_symbols:
        failures.append(f"duplicate source symbols: {duplicate_source_symbols}")
    if duplicate_quality_symbols:
        failures.append(f"duplicate quality symbols: {duplicate_quality_symbols}")
    if missing_quality:
        failures.append(f"source instruments without quality rows: {len(missing_quality)}")
    if extra_quality:
        failures.append(f"quality rows without source instruments: {len(extra_quality)}")
    if fingerprint_mismatches:
        failures.append(f"source fingerprints changed since quality audit: {len(fingerprint_mismatches)}")
    if fingerprint_errors:
        failures.append(f"source fingerprint errors: {len(fingerprint_errors)}")
    if accounted != len(source_set):
        failures.append(f"instrument accounting does not reconcile: {accounted}/{len(source_set)}")
    if freshness_status == "future_timestamp_error":
        failures.append("source contains timestamps after the ledger as-of date")
    failures.extend(benchmark_failures)
    elapsed = perf_counter() - started
    metrics: dict[str, Any] = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "dataset": spec.dataset_id,
        "universe_boundary": declaration.universe_boundary,
        "availability_digest": declaration.digest,
        "source_adapter": spec.adapter,
        "source_format": spec.format,
        "source_interval": spec.interval,
        "source_timezone": spec.timezone,
        "ledger_as_of_utc": as_of_timestamp.isoformat(),
        "first_source_timestamp": first_source_timestamp.isoformat() if pd.notna(first_source_timestamp) else None,
        "last_source_timestamp": last_source_timestamp.isoformat() if pd.notna(last_source_timestamp) else None,
        "freshness_age_calendar_days": freshness_age_days,
        "freshness_status": freshness_status,
        "current_usable_instruments_within_7_days": current_usable,
        "current_coverage_fraction_of_usable": current_coverage_fraction,
        "current_after_close_analysis_available": current_usable > 0,
        "source_instruments": len(source_set),
        "quality_rows": len(quality),
        "accounted_instruments": accounted,
        "usable_instruments": usable,
        "quarantined_instruments": quarantined,
        "quarantined_load_errors": int((instruments["accounting_status"] == "quarantined_load_error").sum()),
        "missing_quality_records": len(missing_quality),
        "extra_quality_records": len(extra_quality),
        "fingerprints_verified": int((instruments["fingerprint_status"] == "verified").sum()),
        "fingerprints_unverified_load_error": int((instruments["fingerprint_status"] == "unverified_load_error").sum()),
        "fingerprint_mismatches": len(fingerprint_mismatches),
        "duplicate_timestamp_issues": _quality_issue_count(quality, "duplicate_timestamp:"),
        "split_like_discontinuity_issues": _quality_issue_count(quality, "extreme_discontinuity:"),
        "capabilities": capabilities,
        "declared_limitations": limitations,
        "benchmark": benchmark_metrics,
        "elapsed_seconds": elapsed,
        "instruments_per_second": len(source_set) / elapsed if elapsed else 0.0,
        "instrument_ledger_digest": stable_hash(instruments.fillna("").to_dict("records")),
        "fingerprint_mismatch_examples": fingerprint_mismatches[:20],
        "fingerprint_error_examples": fingerprint_errors[:20],
        "missing_quality_examples": missing_quality[:20],
        "extra_quality_examples": extra_quality[:20],
    }
    return DataLedgerResult(not failures, metrics, tuple(failures), instruments)


def write_data_ledger_artifacts(
    results: list[DataLedgerResult],
    declaration: AvailabilityDeclaration,
    output_dir: Path,
) -> tuple[Path, Path, list[Path]]:
    results = sorted(results, key=lambda result: str(result.metrics["dataset"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_paths: list[Path] = []
    for result in results:
        path = output_dir / f"{result.metrics['dataset']}-instruments.parquet"
        result.instruments.to_parquet(path, index=False)
        parquet_paths.append(path)
    passed = bool(results) and all(result.passed for result in results)
    envelope = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "passed": passed,
        "availability_digest": declaration.digest,
        "datasets": [
            {"passed": result.passed, "metrics": result.metrics, "failures": list(result.failures)}
            for result in results
        ],
    }
    envelope["ledger_digest"] = stable_hash({
        "schema_version": LEDGER_SCHEMA_VERSION,
        "availability_digest": declaration.digest,
        "datasets": [
            {
                "dataset": result.metrics["dataset"],
                "passed": result.passed,
                "source_instruments": result.metrics["source_instruments"],
                "accounted_instruments": result.metrics["accounted_instruments"],
                "instrument_ledger_digest": result.metrics["instrument_ledger_digest"],
                "benchmark_source_hash": result.metrics["benchmark"].get("source_hash"),
                "failures": list(result.failures),
            }
            for result in results
        ],
    })
    machine_path = output_dir / "data-ledger.json"
    machine_path.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")
    rows = "".join(
        f"<tr><td>{escape(str(result.metrics['dataset']))}</td>"
        f"<td class={'pass' if result.passed else 'fail'}>{'PASS' if result.passed else 'FAIL'}</td>"
        f"<td>{result.metrics['source_instruments']}</td><td>{result.metrics['usable_instruments']}</td>"
        f"<td>{result.metrics['quarantined_instruments']}</td>"
        f"<td>{result.metrics['fingerprints_verified']}</td>"
        f"<td>{escape(str(result.metrics['freshness_status']))}</td>"
        f"<td>{len(result.metrics['declared_limitations'])}</td>"
        f"<td>{result.metrics['elapsed_seconds']:.2f}</td></tr>"
        for result in results
    )
    sections = "".join(_dataset_html(result) for result in results)
    html_path = output_dir / "data-ledger.html"
    html_path.write_text(f"""<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>M01 point-in-time data ledger</title><style>body{{font-family:system-ui,sans-serif;max-width:1280px;margin:2rem auto;padding:0 1rem;background:#f4f6f7;color:#17202a;line-height:1.5}}header,section{{background:white;border:1px solid #ddd;border-radius:10px;padding:1.25rem;margin:1rem 0}}table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;vertical-align:top;padding:.55rem;border-bottom:1px solid #ddd}}.pass{{color:#117864;font-weight:700}}.fail{{color:#b03a2e;font-weight:700}}.warn{{border-left:5px solid #9a6700}}code{{overflow-wrap:anywhere}}pre{{white-space:pre-wrap}}</style></head><body><header><h1>M01 available-data and point-in-time ledger: <span class={'pass' if passed else 'fail'}>{'PASS' if passed else 'FAIL'}</span></h1><p>Every configured source instrument is accounted as usable or quarantined and checked against its prior quality fingerprint. PASS means complete accounting inside the declared source boundary; it does not imply fresh prices or survivorship-free exchange membership.</p><p>Availability declaration <code>{declaration.digest}</code> · ledger <code>{envelope['ledger_digest']}</code></p></header><section><h2>Market summary</h2><table><thead><tr><th>Dataset</th><th>Status</th><th>Source</th><th>Usable</th><th>Quarantined</th><th>Hashes verified</th><th>Freshness</th><th>Declared limitations</th><th>Seconds</th></tr></thead><tbody>{rows}</tbody></table></section><section class=\"warn\"><h2>Boundary</h2><p>The universe is <code>{escape(declaration.universe_boundary)}</code>. Missing point-in-time membership, delisting returns, identity history, classifications and events remain explicit limitations and must constrain later evidence claims. A historical-snapshot freshness status means the data remains useful for historical development but cannot answer a current-date query.</p></section>{sections}<section><h2>Canonical machine summary</h2><pre>{escape(json.dumps(envelope, indent=2, sort_keys=True))}</pre></section></body></html>""")
    return machine_path, html_path, parquet_paths


def _dataset_html(result: DataLedgerResult) -> str:
    limitations = "".join(
        f"<tr><td><code>{escape(name)}</code></td><td>{escape(value['status'])}</td><td>{escape(value['reason'])}</td></tr>"
        for name, value in result.metrics["declared_limitations"].items()
    ) or "<tr><td colspan=3>None</td></tr>"
    failures = "".join(f"<li>{escape(item)}</li>" for item in result.failures) or "<li>None</li>"
    benchmark = result.metrics["benchmark"]
    return f"""<section><h2>{escape(str(result.metrics['dataset']))}</h2><p>{result.metrics['accounted_instruments']}/{result.metrics['source_instruments']} source instruments accounted; {result.metrics['fingerprints_verified']} content fingerprints verified and {result.metrics['fingerprints_unverified_load_error']} load-error fingerprints explicitly unverified.</p><p><b>Coverage:</b> {escape(str(result.metrics['first_source_timestamp']))} through {escape(str(result.metrics['last_source_timestamp']))}; as-of age {result.metrics['freshness_age_calendar_days']} calendar days; <code>{escape(str(result.metrics['freshness_status']))}</code>. Current usable instruments: {result.metrics['current_usable_instruments_within_7_days']}/{result.metrics['usable_instruments']}.</p><h3>Declared limitations</h3><table><thead><tr><th>Capability</th><th>Status</th><th>Reason</th></tr></thead><tbody>{limitations}</tbody></table><h3>Benchmark</h3><pre>{escape(json.dumps(benchmark, indent=2, sort_keys=True))}</pre><h3>Failures</h3><ul>{failures}</ul></section>"""
