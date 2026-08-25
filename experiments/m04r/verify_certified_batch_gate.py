"""Independently verify complete M04R 24-query certified batch evidence."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
from typing import Any

import pandas as pd

from market_analogues.adapters import source_from_spec
from market_analogues.config import load_config
from market_analogues.m04r_batch_registry import validate_m04r_batch_registry
from market_analogues.m04r_certified_search_verification import _certificate_digest
from market_analogues.m04r_full_pack_verification import (
    EVIDENCE_OMITTED as FULL_BUILD_OMITTED,
)
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.types import stable_hash


VERIFICATION_SCHEMA = "m04r-certified-batch-verification-v1"
BATCH_SCHEMA = "m04r-certified-batch-gate-v1"
BATCH_CASE_SCHEMA = "m04r-certified-batch-case-v1"
SCALAR_SCHEMA = "m04r-batch-scalar-controls-v1"
SCALAR_CASE_SCHEMA = "m04r-batch-scalar-control-case-v1"
BATCH_CASE_OMITTED = {"created_at", "exact_seconds", "peak_rss_mb", "result_digest"}
BATCH_OMITTED = {
    "created_at", "started_at", "shared_scan_seconds", "exact_total_seconds",
    "total_seconds", "peak_rss_mb", "result_digest",
}
SCALAR_CASE_OMITTED = {"created_at", "seconds", "peak_rss_mb", "result_digest"}
SCALAR_OMITTED = {
    "created_at", "started_at", "p95_seconds", "maximum_seconds",
    "total_seconds", "peak_rss_mb", "result_digest",
}
SELECTED_CONTROLS = {
    "block_rows": 4_096,
    "workers": 8,
    "initial_frontier_rows": 16_384,
    "maximum_frontier_rows": 32_768,
    "maximum_proposal_rows": 32_769,
    "requested_positions": True,
    "vector_lower_bounds": True,
    "deferred_alignments": True,
    "sorted_joined_iqr_merge": True,
}


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    metrics: dict[str, Any]
    failures: tuple[str, ...]
    result_digest: str


def _matrix_digest(
    payload: dict[str, Any], omitted: set[str], case_omitted: set[str],
) -> str:
    deterministic = {
        key: value for key, value in payload.items() if key not in omitted
    }
    deterministic["cases"] = [{
        key: value for key, value in case.items() if key not in case_omitted
    } for case in payload.get("cases", [])]
    return stable_hash(deterministic)


def verify(
    config_path: Path, full_root: Path, registry_path: Path,
    scalar_path: Path, evidence_path: Path,
) -> VerificationResult:
    failures: list[str] = []
    config = load_config(config_path)
    source = source_from_spec(config.datasets["nasdaq"])
    quality = pd.read_parquet(config.artifact_dir / "quality" / "nasdaq.parquet")
    failures.extend(validate_m04r_batch_registry(
        source, registry_path, quality, config.artifact_dir / "oracles" / "nasdaq",
    ))
    registry = json.loads(registry_path.read_text())
    expected_ids = [str(case["episode_id"]) for case in registry.get("cases_data", [])]
    registry_digest = str(registry.get("registry_digest", ""))

    build = json.loads((full_root / "packed-bound-full.json").read_text())
    build_content = {key: value for key, value in build.items() if key not in FULL_BUILD_OMITTED}
    if build.get("result_digest") != stable_hash(build_content):
        failures.append("full-build evidence digest differs")
    if not all((
        build.get("gate_passed") is True,
        build.get("shadow_generation") is True,
        build.get("real_forward_outcomes_accessed") is False,
        not (full_root / "store" / "active.json").exists(),
    )):
        failures.append("full-build shadow gate differs")
    try:
        loaded = load_packed_generation(
            full_root / "store", str(build.get("generation_id", "")),
            verify_content=True, validate_records=False,
        )
        generation_id = loaded.generation_id
    except Exception as exc:
        failures.append(f"physical generation verification failed: {type(exc).__name__}:{exc}")
        generation_id = ""

    scalar = json.loads(scalar_path.read_text())
    if scalar.get("schema_version") != SCALAR_SCHEMA:
        failures.append("scalar schema differs")
    if scalar.get("result_digest") != _matrix_digest(
        scalar, SCALAR_OMITTED, SCALAR_CASE_OMITTED,
    ):
        failures.append("scalar matrix digest differs")
    if not all((
        scalar.get("gate_passed") is True,
        scalar.get("registry_digest") == registry_digest,
        scalar.get("generation_id") == generation_id,
        scalar.get("completed_query_episode_ids") == expected_ids,
        scalar.get("real_forward_outcomes_accessed") is False,
    )):
        failures.append("scalar matrix binding or gate differs")
    scalar_cases = {
        str(case.get("query_episode_id")): case for case in scalar.get("cases", [])
    }

    evidence = json.loads(evidence_path.read_text())
    if evidence.get("schema_version") != BATCH_SCHEMA:
        failures.append("batch schema differs")
    if evidence.get("result_digest") != _matrix_digest(
        evidence, BATCH_OMITTED, BATCH_CASE_OMITTED,
    ):
        failures.append("batch evidence digest differs")
    if not all((
        evidence.get("registry_digest") == registry_digest,
        evidence.get("generation_id") == generation_id,
        evidence.get("scalar_control_digest") == scalar.get("result_digest"),
        evidence.get("expected_query_episode_ids") == expected_ids,
        evidence.get("completed_query_episode_ids") == expected_ids,
        evidence.get("shared_controls") == SELECTED_CONTROLS,
        evidence.get("real_forward_outcomes_accessed") is False,
    )):
        failures.append("batch evidence binding or controls differ")
    batch_cases = evidence.get("cases") or []
    if len(batch_cases) != 24 or len(scalar_cases) != 24:
        failures.append("batch/scalar case count differs from 24")
    for position, batch_case in enumerate(batch_cases):
        query_id = str(batch_case.get("query_episode_id", ""))
        prefix = f"case {position}:{query_id}"
        scalar_case = scalar_cases.get(query_id)
        if scalar_case is None:
            failures.append(f"{prefix} missing scalar control")
            continue
        scalar_digest = stable_hash({
            key: value for key, value in scalar_case.items()
            if key not in SCALAR_CASE_OMITTED
        })
        batch_digest = stable_hash({
            key: value for key, value in batch_case.items()
            if key not in BATCH_CASE_OMITTED
        })
        if scalar_case.get("schema_version") != SCALAR_CASE_SCHEMA or (
            scalar_case.get("result_digest") != scalar_digest
        ):
            failures.append(f"{prefix} scalar checkpoint digest differs")
        if batch_case.get("schema_version") != BATCH_CASE_SCHEMA or (
            batch_case.get("result_digest") != batch_digest
        ):
            failures.append(f"{prefix} batch checkpoint digest differs")
        if batch_case.get("scalar_result_digest") != scalar_case.get("result_digest"):
            failures.append(f"{prefix} scalar binding differs")
        if batch_case.get("matches") != scalar_case.get("matches"):
            failures.append(f"{prefix} matches differ")
        if batch_case.get("certificate") != scalar_case.get("certificate"):
            failures.append(f"{prefix} certificate differs")
        if not all((
            batch_case.get("certificate_digest") == scalar_case.get("certificate_digest"),
            batch_case.get("certificate_digest") == _certificate_digest(batch_case),
            scalar_case.get("certificate_digest") == _certificate_digest(scalar_case),
            batch_case.get("controls") == SELECTED_CONTROLS,
            batch_case.get("gate_passed") is True,
            batch_case.get("real_forward_outcomes_accessed") is False,
        )):
            failures.append(f"{prefix} certificate/control gate differs")

    exact_total = sum(float(case.get("exact_seconds", 0.0)) for case in batch_cases)
    shared_seconds = float(evidence.get("shared_scan_seconds", 0.0))
    total = shared_seconds + exact_total
    scalar_total = sum(float(case.get("seconds", 0.0)) for case in scalar_cases.values())
    target = .9 * scalar_total
    recomputed_gates = {
        "all_24_completed": len(batch_cases) == 24 and not evidence.get("failed_cases"),
        "all_case_gates_passed": len(batch_cases) == 24 and all(
            case.get("gate_passed") is True for case in batch_cases
        ),
        "at_least_10pct_faster": len(batch_cases) == 24 and total <= target,
        "rss_within_1536_mib": float(evidence.get("peak_rss_mb", 0.0)) <= 1_536,
        "real_forward_outcomes_excluded": all(
            case.get("real_forward_outcomes_accessed") is False for case in batch_cases
        ),
    }
    numeric_equal = all((
        abs(float(evidence.get("exact_total_seconds", 0.0)) - exact_total) <= 1e-9,
        abs(float(evidence.get("total_seconds", 0.0)) - total) <= 1e-9,
        abs(float(evidence.get("scalar_total_seconds", 0.0)) - scalar_total) <= 1e-9,
        abs(float(evidence.get("required_maximum_total_seconds", 0.0)) - target) <= 1e-9,
        evidence.get("gates") == recomputed_gates,
        evidence.get("gate_passed") == all(recomputed_gates.values()),
    ))
    if not numeric_equal:
        failures.append("aggregate timing arithmetic or gates differ")
    metrics = {
        "registry_digest": registry_digest,
        "generation_id": generation_id,
        "completed_cases": len(batch_cases),
        "shared_scan_seconds": shared_seconds,
        "exact_total_seconds": exact_total,
        "total_seconds": total,
        "scalar_total_seconds": scalar_total,
        "required_maximum_total_seconds": target,
        "speedup": scalar_total / total if total else None,
        "peak_rss_mb": float(evidence.get("peak_rss_mb", 0.0)),
        "evidence_digest": evidence.get("result_digest"),
        "physical_generation_verified": bool(generation_id),
        "real_forward_outcomes_accessed": False,
    }
    result_digest = stable_hash({
        "schema_version": VERIFICATION_SCHEMA,
        "passed": not failures,
        "metrics": metrics,
        "failures": failures,
    })
    return VerificationResult(not failures, metrics, tuple(failures), result_digest)


def _write_result(result: VerificationResult, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    machine = output_dir / "certified-batch-verification.json"
    html = output_dir / "certified-batch-verification.html"
    payload = {
        "schema_version": VERIFICATION_SCHEMA,
        "passed": result.passed,
        "metrics": result.metrics,
        "failures": list(result.failures),
        "result_digest": result.result_digest,
    }
    machine.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    status = "PASS" if result.passed else "FAIL"
    css = "pass" if result.passed else "fail"
    html.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>M04R certified batch verification</title><style>body{{font-family:system-ui;max-width:1200px;margin:2rem auto}}pre{{white-space:pre-wrap}}.pass{{color:#075}}.fail{{color:#a20}}</style></head><body><h1>M04R certified batch verification: <span class="{css}">{status}</span></h1><p>Independent reconstruction of registry, physical generation, scalar controls, per-query batch equality, certificate digests and aggregate performance gates.</p><pre>{escape(json.dumps(payload, indent=2, sort_keys=True))}</pre></body></html>""")
    return machine, html


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--scalar", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        args.config, args.full_root, args.registry, args.scalar, args.evidence,
    )
    machine, html = _write_result(result, args.output_dir)
    print(json.dumps({
        "passed": result.passed,
        **result.metrics,
        "result_digest": result.result_digest,
        "failures": result.failures,
    }, indent=2))
    print(html)
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
