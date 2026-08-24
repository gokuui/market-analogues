from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
from typing import Any

from .packed_bound_store import (
    OVERFLOW_DTYPE, PACK_DTYPE, load_packed_generation,
    packed_bound_store_contract,
)
from .types import stable_hash


SCHEMA_VERSION = "m04r-packed-bound-full-verification-v1"
EVIDENCE_OMITTED = {
    "created_at", "build_seconds", "generation_seconds",
    "validation_seconds", "peak_rss_mb", "scan_peak_rss_mb",
    "validation_peak_rss_mb", "inherited_scan_ru_maxrss_mb", "result_digest",
}


@dataclass(frozen=True)
class FullPackVerificationResult:
    passed: bool
    metrics: dict[str, Any]
    failures: tuple[str, ...]
    result_digest: str


def verify_m04r_full_pack(
    evidence_path: Path, store_root: Path, rank_evidence_path: Path,
    one_percent_evidence_path: Path,
) -> FullPackVerificationResult:
    evidence = json.loads(evidence_path.read_text())
    rank = json.loads(rank_evidence_path.read_text())
    one_percent = json.loads(one_percent_evidence_path.read_text())
    failures: list[str] = []
    deterministic = {
        key: value for key, value in evidence.items()
        if key not in EVIDENCE_OMITTED
    }
    if evidence.get("result_digest") != stable_hash(deterministic):
        failures.append("full packed evidence digest differs")
    if evidence.get("schema_version") != "m04r-packed-bound-full-build-v1":
        failures.append("full packed evidence schema differs")
    contract_digest = packed_bound_store_contract()["digest"]
    if evidence.get("pack_contract_digest") != contract_digest:
        failures.append("full packed contract digest differs")
    if evidence.get("real_forward_outcomes_accessed") is not False:
        failures.append("full packed evidence accessed outcomes")
    if evidence.get("shadow_generation") is not True:
        failures.append("full packed generation is not declared shadow-only")
    if evidence.get("scan_io_mode") != "bounded positional reads over raw immutable pack":
        failures.append("full packed scan I/O mode differs")
    if (store_root / "active.json").exists():
        failures.append("full packed shadow store has an active pointer")

    generation_id = str(evidence.get("generation_id", ""))
    try:
        loaded = load_packed_generation(store_root, generation_id)
    except Exception as exc:
        failures.append(f"full physical generation is invalid: {type(exc).__name__}:{exc}")
        loaded = None
    selection = evidence.get("sample", {})
    symbols = [str(value) for value in selection.get("symbols", [])]
    if not (
        selection.get("method") == "complete sha256-ordered A/B universe"
        and float(selection.get("fraction", 0)) == 1.0
        and int(selection.get("universe_count", 0)) == 11_584
        and int(selection.get("sample_count", 0)) == 11_584
        and len(symbols) == 11_584 and len(set(symbols)) == 11_584
        and selection.get("forced_overflow_symbols") == []
    ):
        failures.append("full packed universe selection differs")
    if set(symbols) != set(rank.get("source_prefixes", {})):
        failures.append("full packed symbol universe differs from rank evidence")

    if loaded is not None:
        manifest = loaded.manifest
        provenance = manifest.get("provenance", {})
        if loaded.generation_id != generation_id:
            failures.append("full physical generation ID differs")
        if tuple(symbols) != loaded.symbols:
            failures.append("full packed symbol dictionary differs")
        if provenance.get("rank_evidence_digest") != rank.get("result_digest"):
            failures.append("full pack rank-evidence binding differs")
        if provenance.get("source_prefixes") != rank.get("source_prefixes"):
            failures.append("full pack stock causal prefixes differ")
        if provenance.get("benchmark_prefix") != rank.get("benchmark_prefix"):
            failures.append("full pack benchmark causal prefix differs")
        if provenance.get("selection") != selection:
            failures.append("full pack selection provenance differs")
        if provenance.get("selection_digest") != stable_hash(selection):
            failures.append("full pack selection digest differs")

    rows = int(evidence.get("rows", -1))
    overflow = int(evidence.get("overflow_rows", -1))
    eligible = int(evidence.get("eligible_rows", -1))
    pack_bytes = int(evidence.get("pack_bytes", -1))
    if (
        rows <= 0 or eligible != rows + overflow
        or overflow != int(rank.get("overflow_rows", -2))
        or pack_bytes != rows * PACK_DTYPE.itemsize + overflow * OVERFLOW_DTYPE.itemsize
    ):
        failures.append("full packed row/overflow byte accounting differs")
    if loaded is not None and (
        len(loaded.rows) != rows or len(loaded.overflow) != overflow
        or int(loaded.manifest.get("eligible_row_count", -1)) != eligible
    ):
        failures.append("full physical manifest row accounting differs")
    capacity_passed = pack_bytes <= 11 * 1024 ** 3
    if bool(evidence.get("capacity_passed")) != capacity_passed or not capacity_passed:
        failures.append("full packed capacity gate differs or fails")

    resume = evidence.get("resume_evidence") or {}
    interrupted_count = int(resume.get("interrupted_completed_symbols", 0))
    if not (
        resume.get("schema_version") == "m04r-packed-bound-full-build-v1"
        and int(resume.get("sample_symbols", 0)) == 11_584
        and 0 < interrupted_count < 11_584
        and int(resume.get("interrupted_remaining_symbols", -1))
        == 11_584 - interrupted_count
        and int(resume.get("resumed_completed_symbols", 0)) == 11_584
        and resume.get("resume_completed") is True
    ):
        failures.append("full packed interrupted/resume evidence differs")

    expected_rows = {
        str(case["query_episode_id"]): int(case["eligible_rows"])
        for case in rank.get("authority_cases", [])
    }
    if len(expected_rows) != 12:
        failures.append("rank evidence lacks 12 full authorities")
    row_counts = evidence.get("full_authority_row_counts", [])
    if len(row_counts) != 12:
        failures.append("full pack lacks 12 authority metadata counts")
    else:
        seen = set()
        for row in row_counts:
            query_id = str(row.get("query_episode_id", ""))
            seen.add(query_id)
            expected = expected_rows.get(query_id, -1)
            if (
                int(row.get("eligible_rows", -2)) != expected
                or int(row.get("full_eligible_rows", -3)) != expected
                or row.get("row_accounting_matches") is not True
            ):
                failures.append("full packed authority metadata row accounting differs")
        if seen != set(expected_rows):
            failures.append("full packed metadata authority identities differ")
    authority_accounting = (
        len(row_counts) == 12
        and all(row.get("row_accounting_matches") is True for row in row_counts)
    )
    if bool(evidence.get("authority_row_accounting_passed")) != authority_accounting:
        failures.append("full packed authority accounting summary differs")

    first = evidence.get("warm_scans_first", [])
    second = evidence.get("warm_scans_second", [])
    one_percent_scans = one_percent.get("warm_scans_second", [])
    expected_benchmark_id = (
        str(max(
            one_percent_scans,
            key=lambda row: float(row["projected_full_seconds"]),
        )["query_episode_id"])
        if one_percent_scans else ""
    )
    benchmark_selection = evidence.get("benchmark_selection") or {}
    if not (
        benchmark_selection.get("method")
        == "maximum query-specific projected warm seconds in sealed 1% evidence"
        and benchmark_selection.get("source_evidence_digest")
        == one_percent.get("result_digest")
        and benchmark_selection.get("query_episode_id") == expected_benchmark_id
    ):
        failures.append("full packed worst-authority benchmark selection differs")
    if len(first) != 1 or len(second) != 1:
        failures.append("full pack lacks two worst-authority scans")
    else:
        stable_fields = (
            "query_episode_id", "eligible_rows", "full_eligible_rows",
            "projection_factor", "overflow_eligible_rows", "top_1000_digest",
            "minimum_bound",
        )
        if any(first[0].get(field) != second[0].get(field) for field in stable_fields):
            failures.append("repeated full packed worst-authority scan differs")
        query_id = str(second[0].get("query_episode_id", ""))
        expected = expected_rows.get(query_id, -1)
        if (
            int(second[0].get("eligible_rows", -2)) != expected
            or int(second[0].get("full_eligible_rows", -3)) != expected
            or float(second[0].get("projection_factor", -1)) != 1.0
            or float(second[0].get("projected_full_seconds", -1))
            != float(second[0].get("seconds", -2))
        ):
            failures.append("full packed worst-authority row accounting/projection differs")
        if query_id != expected_benchmark_id:
            failures.append("full packed scan did not use worst projected authority")
    if len(second) == 1:
        maximum_warm = max(float(row["seconds"]) for row in second)
        projected_warm = max(float(row["projected_full_seconds"]) for row in second)
    else:
        maximum_warm = projected_warm = float("inf")
    cold = evidence.get("cold_scan", {})
    projected_cold = float(cold.get("projected_full_seconds", float("inf")))
    if (
        float(evidence.get("maximum_warm_seconds", -1)) != maximum_warm
        or float(evidence.get("projected_warm_seconds", -1)) != projected_warm
        or float(evidence.get("projected_cold_seconds", -1)) != projected_cold
    ):
        failures.append("full packed latency summary differs")
    flags = {
        "warm_latency_passed": projected_warm <= 300,
        "cold_latency_passed": projected_cold <= 600,
        "scan_deterministic": len(first) == 1 and len(second) == 1 and all(
            one.get("top_1000_digest") == two.get("top_1000_digest")
            for one, two in zip(first, second)
        ),
        "overflow_sidecar_exercised": overflow > 0,
        "scan_rss_passed": float(evidence.get("scan_peak_rss_mb", float("inf"))) <= 1_024,
    }
    for field, expected in flags.items():
        if bool(evidence.get(field)) != expected or not expected:
            failures.append(f"full packed {field} differs or fails")
    expected_gate = capacity_passed and authority_accounting and all(flags.values())
    if (
        bool(evidence.get("gate_passed")) != expected_gate
        or bool(evidence.get("poc_passed")) != expected_gate
        or not expected_gate
    ):
        failures.append("full packed gate does not pass")
    peak_rss = float(evidence.get("scan_peak_rss_mb", float("inf")))
    if peak_rss > 1_024:
        failures.append("full packed RSS exceeds 1 GiB")
    unique = tuple(sorted(set(failures)))
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "pack_contract_digest": contract_digest,
        "generation_id": generation_id,
        "universe_symbols": len(symbols),
        "rows": rows, "overflow_rows": overflow,
        "pack_gib": pack_bytes / 1024 ** 3,
        "maximum_warm_seconds": maximum_warm,
        "cold_seconds": projected_cold,
        "peak_rss_mb": peak_rss,
        "evidence_digest": evidence.get("result_digest"),
        "rank_evidence_digest": rank.get("result_digest"),
        "shadow_generation": True,
        "real_forward_outcomes_accessed": False,
    }
    result_payload = {
        "schema_version": SCHEMA_VERSION,
        "metrics": metrics, "failures": list(unique),
    }
    return FullPackVerificationResult(
        not unique, metrics, unique, stable_hash(result_payload),
    )


def write_m04r_full_pack_verification(
    result: FullPackVerificationResult, output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    machine = output_dir / "m04r-packed-bound-full.json"
    html = output_dir / "m04r-packed-bound-full.html"
    payload = {
        "schema_version": SCHEMA_VERSION, "passed": result.passed,
        "metrics": result.metrics, "failures": list(result.failures),
        "result_digest": result.result_digest,
    }
    temporary = machine.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(machine)
    status = "PASS" if result.passed else "FAIL"
    failure_html = "".join(
        f"<li>{escape(value)}</li>" for value in result.failures
    ) or "<li>None</li>"
    temporary = html.with_suffix(".html.tmp")
    temporary.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>M04R full pack verification</title><style>body{{font-family:system-ui;max-width:1100px;margin:2rem auto}}pre{{white-space:pre-wrap}}.pass{{color:#075}}.fail{{color:#a20}}</style></head><body><h1>M04R full shadow pack: <span class="{status.lower()}">{status}</span></h1><p>Independent full-universe physical, provenance, authority-row, performance and shadow-state verification.</p><p>Result <code>{result.result_digest}</code>.</p><pre>{escape(json.dumps(result.metrics, indent=2, sort_keys=True))}</pre><ul>{failure_html}</ul></body></html>""")
    temporary.replace(html)
    return machine, html
