"""Real-source end-to-end evidence replay from sealed NSE exact authorities."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from .adapters import CachedOHLCVSource, OHLCVSource, source_from_spec
from .authority import (
    authority_universe_digest, load_authority_artifact, validate_authority_artifact,
)
from .config import AppConfig
from .distance import representation_distance
from .episodes import build_episode
from .gate12_registry import validate_gate12_registry
from .report import write_search_report
from .search_evidence import build_search_evidence
from .representation import represent
from .types import AnalogueMatch, EpisodeKey, InstrumentKey, SearchQuery, stable_hash


class NseE2EError(RuntimeError):
    pass


@dataclass(frozen=True)
class NseE2EVerification:
    schema_version: str
    passed: bool
    dataset: str
    cases: int
    current_cases: int
    historical_cases: int
    quality_tiers: tuple[str, ...]
    liquidity_strata: tuple[str, ...]
    exact_matches: int
    evidence_rows: int
    eligible_5: int
    eligible_20: int
    eligible_60: int
    authorities_verified: bool
    benchmark_full_fingerprint_drift: bool
    selected_match_rescore_equal: bool
    maximum_selected_rescore_delta: float
    repeated_evidence_equal: bool
    source_lock_unchanged: bool
    workers: int
    source_universe_digest: str
    result_digest: str
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analogue_match_from_payload(raw: Mapping[str, Any]) -> AnalogueMatch:
    required = {
        "episode_id", "dataset_id", "symbol", "cutoff", "lookback",
        "representation_version", "total_distance", "component_distances",
        "alignment", "quality_tier", "quality_issues",
    }
    missing = required - set(raw)
    if missing:
        raise NseE2EError(f"authority match missing fields: {sorted(missing)}")
    key = EpisodeKey(
        InstrumentKey(str(raw["dataset_id"]), str(raw["symbol"])),
        pd.Timestamp(raw["cutoff"]), int(raw["lookback"]),
        str(raw["representation_version"]),
    )
    if key.id != str(raw["episode_id"]):
        raise NseE2EError("authority match episode identity is invalid")
    return AnalogueMatch(
        episode_key=key,
        total_distance=float(raw["total_distance"]),
        component_distances={
            str(name): float(value)
            for name, value in sorted(dict(raw["component_distances"]).items())
        },
        alignment=[(int(left), int(right)) for left, right in raw["alignment"]],
        quality_tier=str(raw["quality_tier"]),
        quality_issues=tuple(str(value) for value in raw["quality_issues"]),
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _load_registry(
    config: AppConfig, source: OHLCVSource, dataset: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str, bool]:
    quality_path = config.artifact_dir / "quality" / f"{dataset}.parquet"
    registry_root = config.artifact_dir / "gate12" / dataset
    yaml_path = registry_root / "query-registry.yaml"
    parquet_path = registry_root / "query-registry.parquet"
    quality = pd.read_parquet(quality_path)
    failures = validate_gate12_registry(source, yaml_path, quality)
    benchmark_drift = tuple(failures) == ("registry benchmark fingerprint is stale",)
    if failures and not benchmark_drift:
        raise NseE2EError("; ".join(failures))
    payload = yaml.safe_load(yaml_path.read_text()) or {}
    yaml_frame = pd.DataFrame(payload["cases_data"])
    parquet_frame = pd.read_parquet(parquet_path)
    try:
        pd.testing.assert_frame_equal(
            yaml_frame.reset_index(drop=True), parquet_frame.reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError as exc:
        raise NseE2EError(f"registry copies disagree: {exc}") from exc
    if benchmark_drift:
        benchmark = source.load_benchmark()
        if benchmark is None or benchmark["timestamp"].duplicated().any() \
                or not benchmark["timestamp"].is_monotonic_increasing:
            raise NseE2EError("drifted benchmark is absent, duplicated or unordered")
        if pd.Timestamp(benchmark["timestamp"].iloc[-1]) <= pd.Timestamp(
            payload["market_latest_timestamp"]
        ):
            raise NseE2EError("benchmark fingerprint changed without a later session")
    return quality, parquet_frame, str(payload["registry_digest"]), benchmark_drift


def _validate_drifted_authority(
    path: Path,
    *,
    query_episode_id: str,
    request: SearchQuery,
    source_fingerprint: str,
    registry_digest: str,
    universe_digest: str,
) -> None:
    """Validate every immutable binding except the intentionally drifted full benchmark."""
    load_authority_artifact(path)
    payload = json.loads(path.read_text())
    expected = {
        "query_episode_id": query_episode_id,
        "request_digest": stable_hash(asdict(request)),
        "source_fingerprint": source_fingerprint,
        "registry_digest": registry_digest,
        "universe_source_digest": universe_digest,
    }
    stale = [name for name, value in expected.items() if payload.get(name) != value]
    if stale:
        raise NseE2EError(
            "authority non-benchmark provenance is stale: " + ", ".join(stale)
        )


def _case_evidence(
    case: Mapping[str, Any],
    source: OHLCVSource,
    authority_root: Path,
    output_root: Path,
    quality: pd.DataFrame,
    registry_digest: str,
    universe_digest: str,
    benchmark_drift: bool,
) -> dict[str, Any]:
    query = build_episode(
        source, InstrumentKey(str(case["dataset_id"]), str(case["symbol"])),
        str(case["cutoff"]), int(case["lookback"]),
        str(case["representation_version"]),
    )
    if query.key.id != str(case["episode_id"]):
        raise NseE2EError(f"{case['case_id']}: query episode identity changed")
    request = SearchQuery(
        query.key, (str(case["dataset_id"]),), ("A", "B"), 20,
        max_per_instrument=3, minimum_history_gap_bars=60,
    )
    authority_path = authority_root / "cases" / f"{query.key.id}.json"
    if benchmark_drift:
        _validate_drifted_authority(
            authority_path, query_episode_id=query.key.id, request=request,
            source_fingerprint=source.fingerprint(query.key.instrument),
            registry_digest=registry_digest, universe_digest=universe_digest,
        )
        authority = load_authority_artifact(authority_path)
    else:
        authority = validate_authority_artifact(
            authority_path, query=query, request=request,
            source_fingerprint=source.fingerprint(query.key.instrument),
            benchmark_fingerprint=source.benchmark_fingerprint(),
            registry_digest=registry_digest, universe_source_digest=universe_digest,
        )
    payload = json.loads(authority_path.read_text())
    matches = [analogue_match_from_payload(raw) for raw in payload["matches"]]
    if len(matches) != 20:
        raise NseE2EError(f"{case['case_id']}: authority does not contain 20 matches")
    query_representation = represent(query)
    maximum_rescore_delta = 0.0
    for match in matches:
        candidate = build_episode(
            source, match.episode_key.instrument, match.episode_key.cutoff,
            match.episode_key.lookback, match.episode_key.representation_version,
            match.quality_tier, match.quality_issues,
        )
        total, components, alignment = representation_distance(
            query_representation, represent(candidate),
        )
        if set(components) != set(match.component_distances):
            raise NseE2EError(
                f"{case['case_id']}: selected exact component inventory changed"
            )
        maximum_rescore_delta = max(
            maximum_rescore_delta, abs(total - match.total_distance),
            *(abs(components[key] - match.component_distances[key]) for key in components),
        )
        if maximum_rescore_delta > 1e-12 or alignment != match.alignment:
            raise NseE2EError(
                f"{case['case_id']}: selected exact match no longer reproduces"
            )
    evidence = build_search_evidence(source, matches, query_cutoff=query.key.cutoff)
    repeated = build_search_evidence(source, matches, query_cutoff=query.key.cutoff)
    if (
        evidence.retrieval_identity_digest != repeated.retrieval_identity_digest
        or evidence.outcome_digest != repeated.outcome_digest
    ):
        raise NseE2EError(f"{case['case_id']}: repeated evidence digest differs")
    report_path = output_root / "reports" / f"{case['case_id']}.html"
    write_search_report(
        query, matches, report_path, evidence.summary,
        provenance={
            "authority_digest": authority.authority_digest,
            "authority_result_digest": authority.result_digest,
            "registry_digest": registry_digest,
            "source_universe_digest": universe_digest,
            "evidence_contract_digest": evidence.contract_digest,
            "retrieval_identity_digest": evidence.retrieval_identity_digest,
            "outcome_digest": evidence.outcome_digest,
        },
        outcome_rows=evidence.rows,
        outcome_notice=(
            "Exact full-universe NSE authority; configured NIFTY calendar and "
            "relative-return context applied."
        ),
    )
    summary = {
        int(row.horizon_sessions): int(row.eligible_outcomes)
        for row in evidence.summary.itertuples(index=False)
    }
    return {
        "case_id": str(case["case_id"]),
        "cutoff_role": str(case["cutoff_role"]),
        "quality_tier": str(case["quality_tier"]),
        "liquidity_stratum": str(case["liquidity_stratum"]),
        "query_episode_id": query.key.id,
        "authority_digest": authority.authority_digest,
        "retrieval_identity_digest": evidence.retrieval_identity_digest,
        "outcome_digest": evidence.outcome_digest,
        "matches": len(matches),
        "evidence_rows": len(evidence.rows),
        "eligible_5": summary.get(5, 0),
        "eligible_20": summary.get(20, 0),
        "eligible_60": summary.get(60, 0),
        "maximum_selected_rescore_delta": maximum_rescore_delta,
        "report": str(report_path.resolve()),
    }


def run_nse_e2e_verification(
    config: AppConfig,
    output_root: Path,
    *,
    dataset: str = "nse",
    workers: int = 12,
) -> tuple[NseE2EVerification, list[dict[str, Any]]]:
    if output_root.exists():
        raise FileExistsError(f"verification root already exists: {output_root}")
    if dataset not in config.datasets:
        raise NseE2EError(f"unknown dataset: {dataset}")
    if workers < 1:
        raise NseE2EError("workers must be positive")
    output_root.mkdir(parents=True)
    raw_source = source_from_spec(config.datasets[dataset])
    source = CachedOHLCVSource(raw_source, max_entries=None)
    quality, registry, registry_digest, benchmark_drift = _load_registry(
        config, source, dataset,
    )
    first_case = registry.sort_values("case_id").iloc[0]
    universe_request = SearchQuery(
        build_episode(
            source, InstrumentKey(dataset, str(first_case.symbol)),
            str(first_case.cutoff), int(first_case.lookback),
            str(first_case.representation_version),
        ).key,
        (dataset,), ("A", "B"), 20,
        max_per_instrument=3, minimum_history_gap_bars=60,
    )
    before_digest = authority_universe_digest(source, universe_request, quality)
    authority_root = config.artifact_dir / "gate12" / "authorities" / dataset
    cases = [row._asdict() for row in registry.sort_values("case_id").itertuples(index=False)]

    # Preload every query and matched symbol once using the available cores.
    keys = {
        InstrumentKey(dataset, str(case["symbol"])) for case in cases
    }
    for case in cases:
        payload = json.loads(
            (authority_root / "cases" / f"{case['episode_id']}.json").read_text()
        )
        keys.update(
            InstrumentKey(str(match["dataset_id"]), str(match["symbol"]))
            for match in payload["matches"]
        )
    source.preload(tuple(sorted(keys)), workers=workers)

    def execute(case: dict[str, Any]) -> dict[str, Any]:
        return _case_evidence(
            case, source, authority_root, output_root, quality,
            registry_digest, before_digest, benchmark_drift,
        )

    failures: list[str] = []
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="nse-e2e") as executor:
        futures = [(case, executor.submit(execute, case)) for case in cases]
        for case, future in futures:
            try:
                records.append(future.result())
            except Exception as exc:
                failures.append(f"{case['case_id']}: {type(exc).__name__}: {exc}")
    records.sort(key=lambda row: row["case_id"])
    after_digest = authority_universe_digest(source, universe_request, quality)
    unchanged = before_digest == after_digest
    if not unchanged:
        failures.append("NSE source-universe digest changed during replay")
    expected_cases = len(cases) == 12 and len(records) == 12
    if not expected_cases:
        failures.append(f"expected 12 completed cases; received {len(records)}")
    exact_matches = sum(row["matches"] for row in records)
    evidence_rows = sum(row["evidence_rows"] for row in records)
    maximum_rescore_delta = max(
        (row["maximum_selected_rescore_delta"] for row in records), default=float("inf"),
    )
    if exact_matches != 240 or evidence_rows != 720:
        failures.append(
            f"expected 240 matches/720 evidence rows; received {exact_matches}/{evidence_rows}"
        )
    state = {
        "schema_version": "nse-real-e2e-verification-v1",
        "dataset": dataset,
        "registry_digest": registry_digest,
        "source_universe_digest": before_digest,
        "workers": workers,
        "benchmark_full_fingerprint_drift": benchmark_drift,
        "cases": records,
        "source_lock_unchanged": unchanged,
        "failures": failures,
    }
    result_digest = stable_hash(state)
    result = NseE2EVerification(
        schema_version=state["schema_version"], passed=not failures,
        dataset=dataset, cases=len(records),
        current_cases=sum(row["cutoff_role"] == "current" for row in records),
        historical_cases=sum(row["cutoff_role"] == "historical" for row in records),
        quality_tiers=tuple(sorted({row["quality_tier"] for row in records})),
        liquidity_strata=tuple(sorted({row["liquidity_stratum"] for row in records})),
        exact_matches=exact_matches, evidence_rows=evidence_rows,
        eligible_5=sum(row["eligible_5"] for row in records),
        eligible_20=sum(row["eligible_20"] for row in records),
        eligible_60=sum(row["eligible_60"] for row in records),
        authorities_verified=not failures and len(records) == 12,
        benchmark_full_fingerprint_drift=benchmark_drift,
        selected_match_rescore_equal=(maximum_rescore_delta <= 1e-12),
        maximum_selected_rescore_delta=maximum_rescore_delta,
        repeated_evidence_equal=not any("repeated evidence" in value for value in failures),
        source_lock_unchanged=unchanged, workers=workers,
        source_universe_digest=before_digest,
        result_digest=result_digest, failures=tuple(failures),
    )
    return result, records


def write_nse_e2e_verification(
    result: NseE2EVerification,
    records: list[dict[str, Any]],
    output_root: Path,
) -> tuple[Path, Path]:
    machine = output_root / "RESULT.json"
    html = output_root / "report.html"
    payload = {**result.to_dict(), "case_records": records,
               "generated_at": datetime.now(timezone.utc).isoformat()}
    _atomic_json(machine, payload)
    case_rows = "".join(
        "<tr>" + "".join(f"<td>{escape(str(row[key]))}</td>" for key in (
            "case_id", "cutoff_role", "quality_tier", "liquidity_stratum",
            "matches", "eligible_5", "eligible_20", "eligible_60",
        )) + f"<td><a href=\"reports/{escape(row['case_id'])}.html\">report</a></td></tr>"
        for row in records
    )
    failure_rows = "".join(f"<li>{escape(value)}</li>" for value in result.failures) or "<li>None</li>"
    html.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NSE real end-to-end verification</title><style>body{{font-family:system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.55rem;border-bottom:1px solid #ddd;text-align:left}}.pass{{color:#176b37}}.fail{{color:#a61b1b}}code{{overflow-wrap:anywhere}}</style></head><body><h1>NSE real end-to-end verification</h1><p class="{'pass' if result.passed else 'fail'}"><b>{'PASS' if result.passed else 'FAIL'}</b></p><p>All sealed full-universe NSE exact authorities were rebound to the current read-only stock universe, then carried through causal outcome joining and user-facing evidence publication.</p><table><thead><tr><th>Case</th><th>Role</th><th>Tier</th><th>Liquidity</th><th>Matches</th><th>Eligible 5</th><th>Eligible 20</th><th>Eligible 60</th><th>Evidence</th></tr></thead><tbody>{case_rows}</tbody></table><h2>Integrity</h2><p>Cases: {result.cases}; exact matches: {result.exact_matches}; evidence rows: {result.evidence_rows}; workers: {result.workers}; result digest: <code>{result.result_digest}</code>; source-universe digest: <code>{result.source_universe_digest}</code>.</p><p>Benchmark whole-file drift: <code>{result.benchmark_full_fingerprint_drift}</code>. All 240 selected exact matches were reconstructed against the current benchmark with maximum distance/component delta <code>{result.maximum_selected_rescore_delta}</code>.</p><h2>Failures</h2><ul>{failure_rows}</ul><p><b>Boundary:</b> This certifies real NSE 252-session evidence mechanics over 12 internally sealed full-universe authorities and current exact reconstruction of their selected matches. The benchmark file was appended after the old lock, whose contract retained only a whole-file fingerprint; this replay does not claim a newly recomputed exhaustive frontier. It does not establish predictive value, and 63/126-session full-universe authority coverage remains separate.</p></body></html>""")
    return machine, html
