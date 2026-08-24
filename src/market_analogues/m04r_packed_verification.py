from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .packed_bound_store import (
    OVERFLOW_DTYPE, PACK_DTYPE, load_packed_generation,
    packed_bound_store_contract,
)
from .types import stable_hash


SCHEMA_VERSION = "m04r-packed-bound-1pct-verification-v1"
EVIDENCE_OMITTED = {
    "created_at", "build_seconds", "generation_seconds",
    "validation_seconds", "peak_rss_mb", "result_digest",
}


@dataclass(frozen=True)
class PackedBoundVerificationResult:
    passed: bool
    metrics: dict[str, Any]
    failures: tuple[str, ...]
    result_digest: str


def _verify_evidence_digest(evidence: dict[str, Any]) -> bool:
    deterministic = {
        key: value for key, value in evidence.items()
        if key not in EVIDENCE_OMITTED
    }
    return evidence.get("result_digest") == stable_hash(deterministic)


def verify_m04r_packed_bound_poc(
    parallel_evidence_path: Path,
    serial_evidence_path: Path,
    parallel_store_root: Path,
    serial_store_root: Path,
    rank_evidence_path: Path,
) -> PackedBoundVerificationResult:
    parallel = json.loads(parallel_evidence_path.read_text())
    serial = json.loads(serial_evidence_path.read_text())
    rank = json.loads(rank_evidence_path.read_text())
    failures: list[str] = []
    for label, evidence in (("parallel", parallel), ("serial", serial)):
        if evidence.get("schema_version") != "m04r-packed-bound-1pct-poc-v1":
            failures.append(f"{label} packed evidence schema differs")
        if not _verify_evidence_digest(evidence):
            failures.append(f"{label} packed evidence digest differs")
        if evidence.get("pack_contract_digest") != packed_bound_store_contract()["digest"]:
            failures.append(f"{label} packed contract digest differs")
        if evidence.get("real_forward_outcomes_accessed") is not False:
            failures.append(f"{label} packed evidence accessed outcomes")
    generation_id = str(parallel.get("generation_id", ""))
    if not generation_id or serial.get("generation_id") != generation_id:
        failures.append("serial and parallel packed generations differ")
    try:
        parallel_store = load_packed_generation(parallel_store_root)
        serial_store = load_packed_generation(serial_store_root)
    except Exception as exc:
        failures.append(f"physical packed generation is invalid: {type(exc).__name__}:{exc}")
        parallel_store = serial_store = None
    if parallel_store is not None and serial_store is not None:
        if (
            parallel_store.generation_id != generation_id
            or serial_store.generation_id != generation_id
        ):
            failures.append("active packed generation differs from evidence")
        if parallel_store.manifest != serial_store.manifest:
            failures.append("serial and parallel packed manifests differ")
        provenance = parallel_store.manifest.get("provenance", {})
        selection = parallel.get("sample", {})
        selected = [str(value) for value in selection.get("symbols", [])]
        rank_prefixes = rank.get("source_prefixes", {})
        expected_prefixes = {
            symbol: rank_prefixes.get(symbol) for symbol in selected
        }
        if provenance.get("source_prefixes") != expected_prefixes:
            failures.append("packed stock causal prefixes differ from rank evidence")
        if provenance.get("benchmark_prefix") != rank.get("benchmark_prefix"):
            failures.append("packed benchmark causal prefix differs from rank evidence")
        if provenance.get("rank_evidence_digest") != rank.get("result_digest"):
            failures.append("packed rank-evidence binding differs")
        if provenance.get("selection") != selection:
            failures.append("packed selection provenance differs")
        if provenance.get("selection_digest") != stable_hash(selection):
            failures.append("packed selection digest differs")
        if len(parallel_store.rows):
            unused_presence = parallel_store.rows["presence"][:, 2] & np.uint8(0b11111000)
            if bool(unused_presence.any()):
                failures.append("packed presence padding bits are nonzero")
            if parallel_store.rows["padding"].tobytes().strip(b"\0"):
                failures.append("packed main padding bytes are nonzero")
            numeric = (
                "coarse", "samples_48", "stage", "structural", "error_radii",
            )
            if not all(np.isfinite(parallel_store.rows[name]).all() for name in numeric):
                failures.append("packed main numeric fields contain non-finite values")
            if np.any(parallel_store.rows["error_radii"] < 0):
                failures.append("packed error radii contain negative values")
        if len(parallel_store.overflow) and (
            parallel_store.overflow["padding"].tobytes().strip(b"\0")
        ):
            failures.append("packed overflow padding bytes are nonzero")

    selection = parallel.get("sample", {})
    universe_count = int(selection.get("universe_count", 0))
    sample_count = int(selection.get("sample_count", 0))
    symbols = selection.get("symbols", [])
    forced = selection.get("forced_overflow_symbols", [])
    if (
        float(selection.get("fraction", 0)) != .01
        or universe_count != 11_584
        or sample_count != math.ceil(universe_count * .01)
        or len(symbols) != sample_count or len(set(symbols)) != sample_count
        or not forced or not set(forced).issubset(symbols)
    ):
        failures.append("packed 1% sample selection differs")
    rows = int(parallel.get("rows", -1))
    overflow = int(parallel.get("overflow_rows", -1))
    eligible = int(parallel.get("eligible_rows", -1))
    pack_bytes = int(parallel.get("pack_bytes", -1))
    if (
        rows <= 0 or overflow <= 0 or eligible != rows + overflow
        or pack_bytes != rows * PACK_DTYPE.itemsize + overflow * OVERFLOW_DTYPE.itemsize
    ):
        failures.append("packed row/sidecar byte accounting differs")
    projected_rows = int(parallel.get("projected_full_rows", -1))
    projected_bytes = (
        projected_rows * PACK_DTYPE.itemsize
        + math.ceil(projected_rows * overflow / max(rows, 1)) * OVERFLOW_DTYPE.itemsize
    )
    if (
        projected_rows != 3_820_000
        or int(parallel.get("projected_full_bytes", -1)) != projected_bytes
        or float(parallel.get("projected_full_gib", -1)) != projected_bytes / 1024 ** 3
    ):
        failures.append("packed full-store capacity projection differs")
    invariant_fields = (
        "pack_contract_digest", "generation_id", "sample", "rows",
        "overflow_rows", "eligible_rows", "pack_bytes", "projected_full_rows",
        "projected_full_bytes", "projected_full_gib",
    )
    if any(parallel.get(field) != serial.get(field) for field in invariant_fields):
        failures.append("serial and parallel packed structural evidence differs")
    resume = parallel.get("resume_evidence") or {}
    if not (
        resume.get("schema_version") == "m04r-packed-bound-1pct-poc-v1"
        and int(resume.get("sample_symbols", 0)) == sample_count
        and 0 < int(resume.get("interrupted_completed_symbols", 0)) < sample_count
        and int(resume.get("interrupted_remaining_symbols", -1))
        == sample_count - int(resume.get("interrupted_completed_symbols", 0))
        and int(resume.get("resumed_completed_symbols", 0)) == sample_count
        and resume.get("resume_completed") is True
    ):
        failures.append("parallel interrupted/resume evidence differs")

    for label, evidence in (("parallel", parallel), ("serial", serial)):
        first = evidence.get("warm_scans_first", [])
        second = evidence.get("warm_scans_second", [])
        if len(first) != 12 or len(second) != 12:
            failures.append(f"{label} packed scan lacks 12 authorities")
            continue
        identities = [str(row.get("query_episode_id", "")) for row in second]
        if len(set(identities)) != 12 or any(not value for value in identities):
            failures.append(f"{label} packed scan query identities differ")
        for one, two in zip(first, second):
            stable_fields = (
                "query_episode_id", "eligible_rows", "full_eligible_rows",
                "projection_factor", "overflow_eligible_rows", "top_1000_digest",
                "minimum_bound",
            )
            if any(one.get(field) != two.get(field) for field in stable_fields):
                failures.append(f"{label} repeated packed scan differs")
            eligible_rows = int(two.get("eligible_rows", 0))
            full_rows = int(two.get("full_eligible_rows", 0))
            factor = full_rows / eligible_rows if eligible_rows else float("inf")
            seconds = float(two.get("seconds", -1))
            if (
                float(two.get("projection_factor", -1)) != factor
                or float(two.get("projected_full_seconds", -1)) != seconds * factor
            ):
                failures.append(f"{label} query-specific latency projection differs")
        maximum_warm = max(float(row["seconds"]) for row in second)
        projected_warm = max(float(row["projected_full_seconds"]) for row in second)
        cold = evidence.get("cold_scan", {})
        projected_cold = float(cold.get("projected_full_seconds", -1))
        if float(evidence.get("maximum_warm_seconds", -1)) != maximum_warm:
            failures.append(f"{label} maximum warm latency differs")
        if float(evidence.get("projected_warm_seconds", -1)) != projected_warm:
            failures.append(f"{label} projected warm latency differs")
        if float(evidence.get("projected_cold_seconds", -1)) != projected_cold:
            failures.append(f"{label} projected cold latency differs")
        expected_flags = {
            "capacity_passed": int(evidence.get("projected_full_bytes", 0)) <= 11 * 1024 ** 3,
            "warm_latency_passed": projected_warm <= 300,
            "cold_latency_passed": projected_cold <= 600,
            "scan_deterministic": all(
                one.get("top_1000_digest") == two.get("top_1000_digest")
                for one, two in zip(first, second)
            ),
            "overflow_sidecar_exercised": int(evidence.get("overflow_rows", 0)) > 0,
        }
        for field, expected in expected_flags.items():
            if bool(evidence.get(field)) != expected:
                failures.append(f"{label} {field} differs")
        expected_pass = all(expected_flags.values())
        if bool(evidence.get("poc_passed")) != expected_pass or not expected_pass:
            failures.append(f"{label} packed POC does not pass")
        if float(evidence.get("peak_rss_mb", float("inf"))) > 1_024:
            failures.append(f"{label} packed RSS exceeds 1 GiB")
    if (
        len(parallel.get("warm_scans_second", [])) == 12
        and len(serial.get("warm_scans_second", [])) == 12
        and [row.get("top_1000_digest") for row in parallel["warm_scans_second"]]
        != [row.get("top_1000_digest") for row in serial["warm_scans_second"]]
    ):
        failures.append("serial and parallel packed scan results differ")
    unique = tuple(sorted(set(failures)))
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "pack_contract_digest": packed_bound_store_contract()["digest"],
        "generation_id": generation_id,
        "sample_symbols": sample_count,
        "eligible_rows": eligible,
        "overflow_rows": overflow,
        "pack_bytes": pack_bytes,
        "projected_full_gib": parallel.get("projected_full_gib"),
        "parallel_projected_warm_seconds": parallel.get("projected_warm_seconds"),
        "parallel_projected_cold_seconds": parallel.get("projected_cold_seconds"),
        "parallel_peak_rss_mb": parallel.get("peak_rss_mb"),
        "parallel_evidence_digest": parallel.get("result_digest"),
        "serial_evidence_digest": serial.get("result_digest"),
        "real_forward_outcomes_accessed": False,
    }
    result_payload = {
        "schema_version": SCHEMA_VERSION,
        "metrics": metrics, "failures": list(unique),
    }
    return PackedBoundVerificationResult(
        not unique, metrics, unique, stable_hash(result_payload),
    )


def write_m04r_packed_bound_verification(
    result: PackedBoundVerificationResult, output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    machine = output_dir / "m04r-packed-bound-1pct.json"
    html = output_dir / "m04r-packed-bound-1pct.html"
    payload = {
        "schema_version": SCHEMA_VERSION, "passed": result.passed,
        "metrics": result.metrics, "failures": list(result.failures),
        "result_digest": result.result_digest,
    }
    temporary = machine.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(machine)
    status = "PASS" if result.passed else "FAIL"
    failures = "".join(f"<li>{escape(value)}</li>" for value in result.failures) or "<li>None</li>"
    temporary = html.with_suffix(".html.tmp")
    temporary.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>M04R packed store verification</title><style>body{{font-family:system-ui;max-width:1100px;margin:2rem auto}}pre{{white-space:pre-wrap}}.pass{{color:#075}}.fail{{color:#a20}}</style></head><body><h1>M04R packed 1% verification: <span class="{status.lower()}">{status}</span></h1><p>Independent physical-layout, provenance, determinism, capacity, latency, RSS and sidecar verification.</p><p>Result <code>{result.result_digest}</code>.</p><pre>{escape(json.dumps(result.metrics, indent=2, sort_keys=True))}</pre><ul>{failures}</ul></body></html>""")
    temporary.replace(html)
    return machine, html
