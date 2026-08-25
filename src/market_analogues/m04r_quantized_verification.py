from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
from typing import Any

from .quantized_bound import (
    PACKED_ROW_BYTES, branch_aware_quantized_bound_contract,
    quantized_bound_contract,
)
from .types import stable_hash


M04R_QUANTIZED_VERIFIER_SCHEMA = "m04r-quantized-bound-verification-v1"
M04R_BRANCH_AWARE_QUANTIZED_VERIFIER_SCHEMA = (
    "m04r-quantized-bound-verification-v2"
)


@dataclass(frozen=True)
class M04RQuantizedVerificationResult:
    passed: bool
    metrics: dict[str, Any]
    failures: tuple[str, ...]
    contract: dict[str, Any]
    result_digest: str


def _verified_digest(payload: dict[str, Any], omitted: set[str]) -> bool:
    deterministic = {key: value for key, value in payload.items() if key not in omitted}
    return payload.get("result_digest") == stable_hash(deterministic)


def verify_m04r_quantized_bound(
    million_path: Path,
    authority_path: Path,
    *,
    branch_aware: bool = False,
) -> M04RQuantizedVerificationResult:
    million = json.loads(million_path.read_text())
    authority = json.loads(authority_path.read_text())
    contract = (
        branch_aware_quantized_bound_contract()
        if branch_aware else quantized_bound_contract()
    )
    verifier_schema = (
        M04R_BRANCH_AWARE_QUANTIZED_VERIFIER_SCHEMA
        if branch_aware else M04R_QUANTIZED_VERIFIER_SCHEMA
    )
    failures: list[str] = []
    if not _verified_digest(
        million, {"elapsed_seconds", "pairs_per_second", "result_digest", "proof_boundary"},
    ):
        failures.append("million-pair evidence digest differs")
    if not _verified_digest(
        authority, {"elapsed_seconds", "peak_rss_mb", "result_digest"},
    ):
        failures.append("authority evidence digest differs")
    for name, payload in (("million", million), ("authority", authority)):
        if payload.get("contract_digest") != contract["digest"]:
            failures.append(f"{name} contract digest differs")
        if payload.get("outcomes_or_labels_used", payload.get("real_forward_outcomes_accessed")):
            failures.append(f"{name} evidence accessed outcomes or labels")
    expected_schemas = (
        (
            "m04r-quantized-bound-million-gate-v2",
            "m04r-quantized-bound-authority-gate-v2",
        ) if branch_aware else (
            "m04r-quantized-bound-million-gate-v1",
            "m04r-quantized-bound-authority-gate-v1",
        )
    )
    if million.get("schema_version") != expected_schemas[0]:
        failures.append("million evidence schema differs")
    if authority.get("schema_version") != expected_schemas[1]:
        failures.append("authority evidence schema differs")
    if int(million.get("pairs", 0)) < 1_000_000:
        failures.append("fewer than one million randomized pairs")
    if int(million.get("boundary_pairs", 0)) < 15_625:
        failures.append("float16 boundary grid is incomplete")
    for name in (
        "violations", "unscaled_violations", "boundary_violations",
        "boundary_unscaled_violations",
    ):
        if int(million.get(name, -1)) != 0:
            failures.append(f"million gate {name} is nonzero")
    cases = authority.get("authority_cases", [])
    if len(cases) != 12 or int(authority.get("case_count", 0)) != 12:
        failures.append("authority gate does not contain 12 cases")
    if int(authority.get("total_rows", 0)) < 300_000:
        failures.append("authority gate has insufficient real rows")
    if not authority.get("all_cases_passed"):
        failures.append("one or more authority cases failed")
    if int(authority.get("overflow_rows", -1)) != 0:
        failures.append("authority gate contains overflow rows")
    if float(authority.get("maximum_total_excess", 1.0)) > 1e-12:
        failures.append("authority total bound is unsafe")
    if float(authority.get("maximum_component_excess", 1.0)) > 1e-12:
        failures.append("authority component bound is unsafe")
    if float(authority.get("minimum_pruning_retention", 0.0)) < .99:
        failures.append("minimum authority pruning retention is below 99%")
    if int(authority.get("packed_row_bytes", 0)) != PACKED_ROW_BYTES:
        failures.append("authority row-byte projection differs")
    if any(not row.get("passed") for row in cases):
        failures.append("authority case pass flags are incomplete")
    if any(
        not row.get("source_scope_digest")
        or len(row.get("selected_symbols", [])) != int(row.get("symbols", -1))
        or not row.get("benchmark_fingerprint")
        for row in cases
    ):
        failures.append("authority source provenance is incomplete")
    metrics = {
        "schema_version": verifier_schema,
        "contract_digest": contract["digest"],
        "million_pairs": int(million.get("pairs", 0)),
        "boundary_pairs": int(million.get("boundary_pairs", 0)),
        "million_violations": int(million.get("violations", -1)),
        "authority_cases": len(cases),
        "authority_rows": int(authority.get("total_rows", 0)),
        "minimum_pruning_retention": float(
            authority.get("minimum_pruning_retention", 0.0)
        ),
        "maximum_total_excess": float(authority.get("maximum_total_excess", 1.0)),
        "maximum_component_excess": float(
            authority.get("maximum_component_excess", 1.0)
        ),
        "overflow_rows": int(authority.get("overflow_rows", -1)),
        "packed_row_bytes": PACKED_ROW_BYTES,
        "projected_3_82m_gib": PACKED_ROW_BYTES * 3_820_000 / 1024 ** 3,
        "million_evidence_digest": str(million.get("result_digest")),
        "authority_evidence_digest": str(authority.get("result_digest")),
        "real_forward_outcomes_accessed": False,
    }
    deterministic = {
        "schema_version": verifier_schema,
        "metrics": metrics,
        "failures": sorted(failures),
        "contract": contract,
    }
    return M04RQuantizedVerificationResult(
        not failures, metrics, tuple(sorted(failures)), contract,
        stable_hash(deterministic),
    )


def write_m04r_quantized_verification(
    result: M04RQuantizedVerificationResult,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    machine = output_dir / "m04r-quantized-bound.json"
    html = output_dir / "m04r-quantized-bound.html"
    contract = output_dir / "quantized-bound-contract.json"
    payload = {
        "schema_version": result.metrics["schema_version"],
        "passed": result.passed,
        "metrics": result.metrics,
        "failures": list(result.failures),
        "contract": result.contract,
        "result_digest": result.result_digest,
    }
    for path, value in ((machine, payload), (contract, result.contract)):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    status = "PASS" if result.passed else "FAIL"
    failure_items = "".join(
        f"<li>{escape(value)}</li>" for value in result.failures
    ) or "<li>None</li>"
    temporary = html.with_suffix(html.suffix + ".tmp")
    temporary.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>M04R quantized bound</title><style>body{{font-family:system-ui;max-width:1200px;margin:2rem auto;background:#f5f7f8}}header,section{{background:white;padding:1.2rem;margin:1rem;border:1px solid #ddd;border-radius:10px}}pre{{white-space:pre-wrap}}.pass{{color:#117864}}.fail{{color:#b03a2e}}</style></head><body><header><h1>M04R quantized bound: <span class="{status.lower()}">{status}</span></h1><p>Independent proof-supporting gates for the complete float16/error-radius distance-v1 lower-bound row.</p><p>Contract <code>{result.contract['digest']}</code> · Result <code>{result.result_digest}</code></p></header><section><h2>Metrics</h2><pre>{escape(json.dumps(result.metrics, indent=2, sort_keys=True))}</pre></section><section><h2>Failures</h2><ul>{failure_items}</ul></section><section><h2>Contract and proof</h2><pre>{escape(json.dumps(result.contract, indent=2, sort_keys=True))}</pre></section></body></html>""")
    temporary.replace(html)
    return machine, html, contract
