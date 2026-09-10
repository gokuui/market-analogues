"""Deterministic cross-adapter end-to-end portability verifier."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import escape
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .adapters import source_from_spec
from .config import BenchmarkSpec, DatasetSpec
from .episodes import build_episode
from .report import write_search_report
from .search import SearchCandidate, exact_search
from .search_evidence import build_search_evidence
from .types import InstrumentKey, SearchQuery, stable_hash


@dataclass(frozen=True)
class PortabilityVerification:
    schema_version: str
    passed: bool
    variants: tuple[str, ...]
    ordered_symbols: tuple[str, ...]
    maximum_distance_delta: float
    retrieval_semantics_equal: bool
    same_format_identity_equal: bool
    evidence_rows_equivalent: bool
    evidence_summary_equivalent: bool
    maximum_evidence_numeric_delta: float
    source_files_unchanged: bool
    future_mutation_retrieval_invariant: bool
    future_mutation_outcomes_changed: bool
    reports_valid: bool
    result_digest: str
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bars() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    dates = pd.bdate_range("2018-01-01", periods=340)
    step = np.arange(len(dates), dtype=np.float64)
    market_close = 100.0 * np.exp(np.cumsum(.00035 + .0015 * np.sin(step / 17.0)))

    def frame(close: np.ndarray, phase: float, volume_scale: float) -> pd.DataFrame:
        gap = .0015 * np.sin(step / 9.0 + phase)
        open_ = np.r_[close[0], close[:-1]] * np.exp(gap)
        width = .006 + .002 * (1.0 + np.sin(step / 13.0 + phase))
        high = np.maximum(open_, close) * (1.0 + width)
        low = np.minimum(open_, close) * (1.0 - width)
        volume = np.rint(volume_scale * (1.0 + .25 * np.sin(step / 11.0 + phase)))
        return pd.DataFrame({
            "date": dates, "open": open_, "high": high, "low": low,
            "close": close, "volume": volume.astype(np.int64),
        })

    benchmark = frame(market_close, 0.0, 5_000_000)
    frames: dict[str, pd.DataFrame] = {}
    for index, symbol in enumerate(("QUERY", "ALPHA", "BETA", "GAMMA", "DELTA")):
        phase = .4 * index
        relative = np.exp(np.cumsum(
            .00015 * (index - 2) + .0025 * np.sin(step / (15.0 + index) + phase)
        ))
        frames[symbol] = frame(
            market_close * relative * (1.0 + .02 * index), phase,
            700_000 + 120_000 * index,
        )
    return frames, benchmark


def _write_variant(
    root: Path,
    *,
    adapter: str,
    fmt: str,
    frames: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
) -> DatasetSpec:
    root.mkdir(parents=True, exist_ok=False)
    suffix = ".parquet" if fmt == "parquet" else ".csv"
    benchmark_path = root / f"benchmark{suffix}"
    writer = "to_parquet" if fmt == "parquet" else "to_csv"
    getattr(benchmark, writer)(benchmark_path, index=False)
    if adapter == "directory":
        source_root = root / "symbols"
        source_root.mkdir()
        for symbol, frame in sorted(frames.items()):
            getattr(frame, writer)(source_root / f"{symbol}{suffix}", index=False)
        path = source_root
        symbol_from = "filename"
    else:
        path = root / f"long{suffix}"
        long = pd.concat(
            [frame.assign(symbol=symbol) for symbol, frame in sorted(frames.items())],
            ignore_index=True,
        )
        getattr(long, writer)(path, index=False)
        symbol_from = "column"
    return DatasetSpec(
        dataset_id="portable", adapter=adapter, path=path, format=fmt,
        file_glob=f"*{suffix}", symbol_from=symbol_from,
        symbol_column="symbol", timestamp_column="date",
        benchmark=BenchmarkSpec(benchmark_path, timestamp_column="date", format=fmt),
    )


def _file_digests(root: Path) -> dict[str, str]:
    from hashlib import sha256
    return {
        str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def _normalized_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    normalized = frame.astype(object).where(pd.notna(frame), None)
    records = normalized.to_dict("records")
    for row in records:
        for key, value in list(row.items()):
            if isinstance(value, (np.bool_, np.integer, np.floating)):
                row[key] = value.item()
    return records


def _records_equivalent(
    left: list[dict[str, Any]], right: list[dict[str, Any]], *, tolerance: float,
) -> tuple[bool, float]:
    if len(left) != len(right):
        return False, math.inf
    maximum_delta = 0.0
    for left_row, right_row in zip(left, right):
        if left_row.keys() != right_row.keys():
            return False, math.inf
        for key in left_row:
            a, b = left_row[key], right_row[key]
            if a is None or b is None or isinstance(a, (str, bool)) or isinstance(b, (str, bool)):
                if a != b:
                    return False, maximum_delta
                continue
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                delta = abs(float(a) - float(b))
                maximum_delta = max(maximum_delta, delta)
                if delta > tolerance:
                    return False, maximum_delta
            elif a != b:
                return False, maximum_delta
    return True, maximum_delta


def _run_variant(spec: DatasetSpec, output: Path) -> dict[str, Any]:
    source = source_from_spec(spec)
    keys = {key.source_symbol: key for key in source.instruments()}
    query_cutoff = source.load(keys["QUERY"])["timestamp"].iloc[320]
    candidate_cutoff = source.load(keys["ALPHA"])["timestamp"].iloc[170]
    query = build_episode(source, keys["QUERY"], query_cutoff, 63, "dense-v1")
    candidates = [
        SearchCandidate.from_episode(
            build_episode(source, keys[symbol], candidate_cutoff, 63, "dense-v1")
        )
        for symbol in ("ALPHA", "BETA", "GAMMA", "DELTA")
    ]
    matches = exact_search(
        query, candidates,
        SearchQuery(
            query.key, ("portable",), ("A", "B"), 4,
            max_per_instrument=1, minimum_history_gap_bars=60,
        ),
    )
    evidence = build_search_evidence(source, matches, query_cutoff=query_cutoff)
    report = write_search_report(
        query, matches, output, evidence.summary,
        provenance={
            "evidence_contract_digest": evidence.contract_digest,
            "retrieval_identity_digest": evidence.retrieval_identity_digest,
            "outcome_digest": evidence.outcome_digest,
        },
        outcome_rows=evidence.rows,
        outcome_notice="Synthetic portability verification evidence.",
    )
    report_text = report.read_text()
    return {
        "symbols": [match.episode_key.instrument.source_symbol for match in matches],
        "distances": [float(match.total_distance) for match in matches],
        "components": [dict(sorted(match.component_distances.items())) for match in matches],
        "retrieval_identity_digest": evidence.retrieval_identity_digest,
        "outcome_digest": evidence.outcome_digest,
        "rows": _normalized_records(evidence.rows),
        "summary": _normalized_records(evidence.summary),
        "report_valid": all(text in report_text for text in (
            "Descriptive evidence, not a forecast", "Per-match outcome evidence",
            "retrieval_identity_digest", "benchmark_relative_return",
        )),
    }


def run_portability_verification(root: Path) -> PortabilityVerification:
    """Create four equivalent inputs and independently compare the full output."""
    if root.exists():
        raise FileExistsError(f"verification root already exists: {root}")
    root.mkdir(parents=True)
    frames, benchmark = _bars()
    variants = (
        ("directory-parquet", "directory", "parquet"),
        ("directory-csv", "directory", "csv"),
        ("long-table-parquet", "long_table", "parquet"),
        ("long-table-csv", "long_table", "csv"),
    )
    results: dict[str, dict[str, Any]] = {}
    unchanged = True
    for name, adapter, fmt in variants:
        variant_root = root / "inputs" / name
        spec = _write_variant(
            variant_root, adapter=adapter, fmt=fmt,
            frames=frames, benchmark=benchmark,
        )
        before = _file_digests(variant_root)
        results[name] = _run_variant(spec, root / "reports" / f"{name}.html")
        unchanged = unchanged and before == _file_digests(variant_root)

    reference = results[variants[0][0]]
    failures: list[str] = []
    maximum_delta = 0.0
    semantics_equal = True
    rows_equivalent = True
    summary_equivalent = True
    maximum_evidence_delta = 0.0
    for name, _, _ in variants[1:]:
        result = results[name]
        if result["symbols"] != reference["symbols"]:
            failures.append(f"{name}: ordered neighbour symbols differ")
            semantics_equal = False
        maximum_delta = max(maximum_delta, max(
            abs(left - right)
            for left, right in zip(reference["distances"], result["distances"])
        ))
        for left, right in zip(reference["components"], result["components"]):
            if left.keys() != right.keys() or any(
                abs(float(left[key]) - float(right[key])) > 1e-12 for key in left
            ):
                semantics_equal = False
        row_equal, row_delta = _records_equivalent(
            reference["rows"], result["rows"], tolerance=1e-12,
        )
        summary_equal, summary_delta = _records_equivalent(
            reference["summary"], result["summary"], tolerance=1e-12,
        )
        rows_equivalent = rows_equivalent and row_equal
        summary_equivalent = summary_equivalent and summary_equal
        maximum_evidence_delta = max(
            maximum_evidence_delta, row_delta, summary_delta,
        )
    if maximum_delta > 1e-12:
        failures.append(f"maximum cross-adapter distance delta {maximum_delta} exceeds 1e-12")
        semantics_equal = False
    same_format_identity = (
        results["directory-parquet"]["retrieval_identity_digest"]
        == results["long-table-parquet"]["retrieval_identity_digest"]
        and results["directory-csv"]["retrieval_identity_digest"]
        == results["long-table-csv"]["retrieval_identity_digest"]
    )
    if not semantics_equal:
        failures.append("cross-adapter retrieval semantics differ")
    if not same_format_identity:
        failures.append("same-format adapter retrieval identity differs")
    if not rows_equivalent:
        failures.append("cross-adapter evidence rows exceed numeric tolerance")
    if not summary_equivalent:
        failures.append("cross-adapter evidence summary exceeds numeric tolerance")
    if not unchanged:
        failures.append("a source input changed during verification")

    mutated_frames = {symbol: frame.copy() for symbol, frame in frames.items()}
    cutoff = mutated_frames["ALPHA"].date.iloc[170]
    future = mutated_frames["ALPHA"].date > cutoff
    mutated_frames["ALPHA"].loc[future, ["open", "high", "low", "close"]] *= 1.25
    mutation_root = root / "inputs" / "future-mutated"
    mutation_spec = _write_variant(
        mutation_root, adapter="directory", fmt="parquet",
        frames=mutated_frames, benchmark=benchmark,
    )
    mutated = _run_variant(mutation_spec, root / "reports" / "future-mutated.html")
    mutation_identity = (
        mutated["retrieval_identity_digest"] == reference["retrieval_identity_digest"]
    )
    mutation_outcome = mutated["outcome_digest"] != reference["outcome_digest"]
    if not mutation_identity:
        failures.append("future mutation changed retrieval identity")
    if not mutation_outcome:
        failures.append("future mutation did not change outcome evidence")
    reports_valid = all(result["report_valid"] for result in results.values()) \
        and bool(mutated["report_valid"])
    if not reports_valid:
        failures.append("one or more HTML reports lacks required evidence sections")

    state = {
        "schema_version": "portable-e2e-verification-v1",
        "variants": [name for name, _, _ in variants],
        "ordered_symbols": reference["symbols"],
        "maximum_distance_delta": maximum_delta,
        "retrieval_semantics_equal": semantics_equal,
        "same_format_identity_equal": same_format_identity,
        "evidence_rows_equivalent": rows_equivalent,
        "evidence_summary_equivalent": summary_equivalent,
        "maximum_evidence_numeric_delta": maximum_evidence_delta,
        "source_files_unchanged": unchanged,
        "future_mutation_retrieval_invariant": mutation_identity,
        "future_mutation_outcomes_changed": mutation_outcome,
        "reports_valid": reports_valid,
        "failures": failures,
    }
    digest = stable_hash(state)
    return PortabilityVerification(
        schema_version=state["schema_version"], passed=not failures,
        variants=tuple(state["variants"]),
        ordered_symbols=tuple(state["ordered_symbols"]),
        maximum_distance_delta=maximum_delta,
        retrieval_semantics_equal=semantics_equal,
        same_format_identity_equal=same_format_identity,
        evidence_rows_equivalent=rows_equivalent,
        evidence_summary_equivalent=summary_equivalent,
        maximum_evidence_numeric_delta=maximum_evidence_delta,
        source_files_unchanged=unchanged,
        future_mutation_retrieval_invariant=mutation_identity,
        future_mutation_outcomes_changed=mutation_outcome,
        reports_valid=reports_valid, result_digest=digest,
        failures=tuple(failures),
    )


def write_portability_verification(
    result: PortabilityVerification, root: Path,
) -> tuple[Path, Path]:
    machine = root / "RESULT.json"
    html = root / "report.html"
    payload = result.to_dict()
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    machine.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    rows = "".join(
        f"<tr><th>{escape(str(key))}</th><td><code>{escape(str(value))}</code></td></tr>"
        for key, value in payload.items() if key != "failures"
    )
    failures = "".join(f"<li>{escape(value)}</li>" for value in result.failures) or "<li>None</li>"
    html.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Portable E2E verification</title><style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.6rem;border-bottom:1px solid #ddd;text-align:left}}.pass{{color:#176b37}}.fail{{color:#a61b1b}}code{{overflow-wrap:anywhere}}</style></head><body><h1>Portable end-to-end verification</h1><p class="{'pass' if result.passed else 'fail'}"><b>{'PASS' if result.passed else 'FAIL'}</b></p><p>Directory/long-table and CSV/Parquet inputs were carried through canonical ingestion, causal representation, exact reranking, post-retrieval outcome evidence and HTML publication.</p><table>{rows}</table><h2>Failures</h2><ul>{failures}</ul><p><b>Boundary:</b> This certifies semantic portability on deterministic data. It does not establish predictive value or substitute for the real NSE replay.</p></body></html>""")
    return machine, html
