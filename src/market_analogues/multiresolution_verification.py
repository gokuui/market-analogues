from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
import resource
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .adapters import OHLCVSource
from .episodes import build_episode
from .multiresolution import (
    CHANNEL_FIELD_OBSERVATIONS, REQUIRED_HORIZONS, SUMMARY_FIELD_OBSERVATIONS,
    MultiResolutionState, build_multiresolution_state, field_contract, state_manifest,
)
from .synthetic import FAMILIES, generate_case, transform_case
from .types import Episode, EpisodeKey, InstrumentKey, stable_hash


@dataclass(frozen=True)
class MultiResolutionVerification:
    passed: bool
    metrics: dict[str, Any]
    failures: tuple[str, ...]
    real_cases: tuple[dict[str, Any], ...]


def _state_delta(left: MultiResolutionState, right: MultiResolutionState) -> float:
    deltas: list[float] = []
    left_numeric = left.channels.select_dtypes("number").to_numpy(dtype=float)
    right_numeric = right.channels.select_dtypes("number").to_numpy(dtype=float)
    finite = np.isfinite(left_numeric) & np.isfinite(right_numeric)
    deltas.append(float(np.max(np.abs(left_numeric[finite] - right_numeric[finite]))) if finite.any() else 0.0)
    if not np.array_equal(np.isfinite(left_numeric), np.isfinite(right_numeric)):
        return float("inf")
    for horizon in REQUIRED_HORIZONS:
        a, b = left.views[horizon], right.views[horizon]
        for name in a.samples:
            if not np.array_equal(a.samples[name].observed, b.samples[name].observed):
                return float("inf")
            deltas.append(float(np.max(np.abs(a.samples[name].values - b.samples[name].values))))
        for name in a.summary:
            x, y = a.summary[name], b.summary[name]
            if x is None or y is None:
                if x is not y:
                    return float("inf")
            else:
                deltas.append(abs(x - y))
    return max(deltas, default=0.0)


def _future_mutation_pair() -> tuple[MultiResolutionState, MultiResolutionState]:
    case = generate_case(FAMILIES[0], 9001, n=300)
    bars = case.episode.bars.copy()
    benchmark = case.episode.benchmark.copy()
    cutoff_position = 251
    cutoff = pd.Timestamp(bars.timestamp.iloc[cutoff_position])
    key = EpisodeKey(InstrumentKey("synthetic", "future-prefix"), cutoff, 252, "dense-v1")
    original = Episode(key, bars, benchmark)
    mutated_bars = bars.copy()
    mutated_bars.loc[cutoff_position + 1:, ["open", "high", "low", "close", "volume"]] *= 100
    mutated_benchmark = benchmark.copy()
    mutated_benchmark.loc[cutoff_position + 1:, "close"] *= 100
    mutated = Episode(key, mutated_bars, mutated_benchmark)
    return build_multiresolution_state(original), build_multiresolution_state(mutated)


def verify_multiresolution_state(
    real_inputs: list[tuple[str, OHLCVSource, pd.DataFrame, pd.DataFrame]],
    *,
    maximum_total_seconds: float = 60.0,
    maximum_case_seconds: float = 2.0,
    maximum_rss_mb: float = 1024.0,
) -> MultiResolutionVerification:
    started = perf_counter()
    failures: list[str] = []
    contract = field_contract()
    contract_digest = stable_hash(contract)
    synthetic_deltas: list[float] = []
    missing_mask_errors = 0
    state_digests: list[str] = []
    context_names = {
        "benchmark_close", "benchmark_return", "benchmark_path",
        "benchmark_volatility", "benchmark_drawdown", "relative_return", "relative_path",
    }
    for family_index, family in enumerate(FAMILIES):
        case = generate_case(family, 7000 + family_index, n=300)
        base = build_multiresolution_state(case.episode)
        transformed = transform_case(
            case, name="units", price_scale=17.5, volume_scale=100,
        )
        scaled = build_multiresolution_state(transformed.episode)
        synthetic_deltas.append(_state_delta(base, scaled))
        missing_episode = Episode(
            case.episode.key, case.episode.bars.copy(), None,
            case.episode.quality_tier, case.episode.quality_issues,
        )
        missing = build_multiresolution_state(missing_episode)
        for view in missing.views.values():
            missing_mask_errors += sum(
                int(sample.observed.any()) for name, sample in view.samples.items()
                if name in context_names
            )
            missing_mask_errors += int(view.summary["benchmark_context_fraction"] != 0)
            missing_mask_errors += int(view.summary["benchmark_log_return"] is not None)
            missing_mask_errors += int(view.summary["relative_log_return"] is not None)
        state_digests.extend((base.state_digest, missing.state_digest))

    before, after = _future_mutation_pair()
    future_mutation_equal = before.state_digest == after.state_digest
    future_mutation_delta = _state_delta(before, after)
    if not future_mutation_equal or future_mutation_delta != 0:
        failures.append("post-cutoff stock or benchmark mutation changed the state")
    scale_delta = max(synthetic_deltas, default=float("inf"))
    if scale_delta > 1e-6:
        failures.append(f"price/volume unit invariance delta {scale_delta:.3g} exceeds 1e-6")
    if missing_mask_errors:
        failures.append(f"missing market context produced {missing_mask_errors} observed values")

    real_rows: list[dict[str, Any]] = []
    for dataset_id, source, quality, registry in real_inputs:
        qmap = quality.set_index(quality.symbol.astype(str), drop=False).to_dict("index")
        for row in registry.sort_values("case_id", kind="stable").itertuples(index=False):
            case_started = perf_counter()
            symbol = str(row.symbol)
            quality_row = qmap.get(symbol)
            if quality_row is None:
                failure = "missing quality record"
                failures.append(f"{row.case_id}:{failure}")
                real_rows.append({
                    "case_id": str(row.case_id), "dataset": dataset_id,
                    "symbol": symbol, "cutoff": str(row.cutoff),
                    "failures": [failure], "seconds_for_pair": perf_counter() - case_started,
                })
                continue
            raw_issues = quality_row.get("issues")
            issues = tuple(
                filter(None, str(raw_issues).split(";"))
            ) if pd.notna(raw_issues) else ()
            try:
                episode = build_episode(
                    source, InstrumentKey(dataset_id, symbol), row.cutoff, 252,
                    str(row.representation_version), str(quality_row.get("tier") or "A"),
                    issues,
                )
                state = build_multiresolution_state(episode)
                repeat = build_multiresolution_state(episode)
                seconds = perf_counter() - case_started
                manifest = state_manifest(state)
                case_failures: list[str] = []
                if state.state_digest != repeat.state_digest:
                    case_failures.append("repeat state digest differs")
                if tuple(state.views) != REQUIRED_HORIZONS:
                    case_failures.append("required horizon set differs")
                if state.field_contract_digest != contract_digest:
                    case_failures.append("field contract digest differs")
                if pd.Timestamp(state.cutoff) != pd.Timestamp(row.cutoff):
                    case_failures.append("state cutoff differs from registry")
                if seconds > maximum_case_seconds:
                    case_failures.append(
                        f"state pair time {seconds:.3f}s exceeds {maximum_case_seconds:.3f}s"
                    )
                real_rows.append({
                    "case_id": str(row.case_id),
                    "dataset": dataset_id,
                    "symbol": symbol,
                    "cutoff": str(row.cutoff),
                    "state_digest": state.state_digest,
                    "manifest_digest": manifest["manifest_digest"],
                    "field_contract_digest": state.field_contract_digest,
                    "benchmark_context_fraction_252": state.views[252].summary["benchmark_context_fraction"],
                    "quality_tier": state.quality_tier,
                    "quality_issues": list(state.quality_issues),
                    "seconds_for_pair": seconds,
                    "failures": case_failures,
                })
                failures.extend(f"{row.case_id}:{failure}" for failure in case_failures)
                state_digests.append(state.state_digest)
            except Exception as exc:
                failures.append(f"{row.case_id}:{type(exc).__name__}:{exc}")
                real_rows.append({
                    "case_id": str(row.case_id), "dataset": dataset_id,
                    "symbol": symbol, "cutoff": str(row.cutoff),
                    "failures": [f"{type(exc).__name__}:{exc}"],
                    "seconds_for_pair": perf_counter() - case_started,
                })
    elapsed = perf_counter() - started
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    expected_real_cases = sum(len(registry) for _, _, _, registry in real_inputs)
    successful_real_cases = sum(not row["failures"] for row in real_rows)
    case_ids = [str(row["case_id"]) for row in real_rows]
    if expected_real_cases == 0:
        failures.append("no independent real cases were supplied")
    if len(case_ids) != len(set(case_ids)):
        failures.append("independent real case IDs are not unique")
    if successful_real_cases != expected_real_cases:
        failures.append(f"successful real cases {successful_real_cases}/{expected_real_cases}")
    if elapsed > maximum_total_seconds:
        failures.append(f"total elapsed {elapsed:.3f}s exceeds {maximum_total_seconds:.3f}s")
    if peak_rss_mb > maximum_rss_mb:
        failures.append(f"peak RSS {peak_rss_mb:.1f}MB exceeds {maximum_rss_mb:.1f}MB")
    metrics: dict[str, Any] = {
        "schema_version": "m02-multiresolution-verification-v1",
        "required_horizons": list(REQUIRED_HORIZONS),
        "channel_fields": len(CHANNEL_FIELD_OBSERVATIONS),
        "summary_fields": len(SUMMARY_FIELD_OBSERVATIONS),
        "field_contract_fields": len(contract),
        "field_contract_digest": contract_digest,
        "synthetic_families": len(FAMILIES),
        "price_volume_unit_max_delta": scale_delta,
        "future_mutation_digest_equal": future_mutation_equal,
        "future_mutation_max_delta": future_mutation_delta,
        "missing_context_mask_errors": missing_mask_errors,
        "real_cases_expected": expected_real_cases,
        "real_cases_passed": successful_real_cases,
        "real_case_max_seconds": max((row["seconds_for_pair"] for row in real_rows), default=0.0),
        "maximum_case_seconds": maximum_case_seconds,
        "total_seconds": elapsed,
        "maximum_total_seconds": maximum_total_seconds,
        "peak_rss_mb": peak_rss_mb,
        "maximum_rss_mb": maximum_rss_mb,
        "state_set_digest": stable_hash(sorted(state_digests)),
    }
    return MultiResolutionVerification(not failures, metrics, tuple(failures), tuple(real_rows))


def write_multiresolution_verification(
    result: MultiResolutionVerification,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = field_contract()
    contract_path = output_dir / "field-observation-contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    payload = {
        "passed": result.passed,
        "metrics": result.metrics,
        "failures": list(result.failures),
        "real_cases": list(result.real_cases),
    }
    machine_path = output_dir / "m02-multiresolution-verification.json"
    machine_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    rows = "".join(
        f"<tr><td>{escape(str(row['case_id']))}</td><td>{escape(str(row['dataset']))}</td>"
        f"<td>{escape(str(row['cutoff']))}</td><td>{row.get('benchmark_context_fraction_252', 'n/a')}</td>"
        f"<td>{row['seconds_for_pair']:.4f}</td><td>{escape('; '.join(row['failures']) or 'PASS')}</td></tr>"
        for row in result.real_cases
    )
    failures = "".join(f"<li>{escape(failure)}</li>" for failure in result.failures) or "<li>None</li>"
    status = "PASS" if result.passed else "FAIL"
    html_path = output_dir / "m02-multiresolution-verification.html"
    html_path.write_text(f"""<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>M02 multi-resolution verification</title><style>body{{font-family:system-ui,sans-serif;max-width:1300px;margin:2rem auto;padding:0 1rem;background:#f4f6f7;color:#17202a}}header,section{{background:white;border:1px solid #ddd;border-radius:10px;padding:1.25rem;margin:1rem 0}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.5rem;text-align:left;border-bottom:1px solid #ddd}}.pass{{color:#117864}}.fail{{color:#b03a2e}}pre{{white-space:pre-wrap}}</style></head><body><header><h1>M02 six-resolution point-in-time state: <span class={status.lower()}>{status}</span></h1><p>The state retains the complete 252-session derived channel frame and exposes 252/126/63/21/10/5-session summaries plus masked samples. Missing context has a separate mask and cannot become a neutral observed value.</p></header><section><h2>Metrics</h2><pre>{escape(json.dumps(result.metrics, indent=2, sort_keys=True))}</pre></section><section><h2>Failures</h2><ul>{failures}</ul></section><section><h2>Independent real cases</h2><table><thead><tr><th>Case</th><th>Market</th><th>Cutoff</th><th>Context fraction</th><th>Pair seconds</th><th>Status</th></tr></thead><tbody>{rows}</tbody></table></section><section><h2>Field contract</h2><p>Every channel and summary declares source inputs, earliest observation, missingness and a prohibition on future inputs. Digest <code>{result.metrics['field_contract_digest']}</code>.</p></section></body></html>""")
    return machine_path, html_path, contract_path
