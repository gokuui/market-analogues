from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
from typing import Any

from .packed_bound_search import (
    DEFAULT_ROUTE_QUOTAS, packed_bound_search_contract,
)
from .packed_bound_store import load_packed_generation
from .m04r_quantized_rank_verification import verify_m04r_quantized_ranks
from .types import stable_hash


SCHEMA_VERSION = "m04r-global-bound-proposal-verification-v1"
EVIDENCE_OMITTED = {
    "created_at", "forward_seconds", "reverse_seconds", "peak_rss_mb",
    "result_digest",
}
FULL_BUILD_OMITTED = {
    "created_at", "build_seconds", "generation_seconds",
    "validation_seconds", "peak_rss_mb", "scan_peak_rss_mb",
    "validation_peak_rss_mb", "inherited_scan_ru_maxrss_mb", "result_digest",
}
INVARIANT_FIELDS = (
    "generation_id", "query_episode_id", "rows_scanned", "eligible_rows",
    "eligible_main_rows", "eligible_overflow_rows", "candidate_count",
    "composite_count", "route_counts", "route_quotas", "candidate_digest",
    "top_1000_score_digest", "duplicate_candidates", "future_candidates",
    "overlapping_same_symbol_candidates", "overflow_candidates",
    "composite_episode_ids",
)


@dataclass(frozen=True)
class GlobalProposalVerificationResult:
    passed: bool
    metrics: dict[str, Any]
    failures: tuple[str, ...]
    result_digest: str


def verify_m04r_global_bound_proposal(
    evidence_path: Path,
    store_root: Path,
    full_build_evidence_path: Path,
    rank_evidence_path: Path,
) -> GlobalProposalVerificationResult:
    evidence = json.loads(evidence_path.read_text())
    build = json.loads(full_build_evidence_path.read_text())
    rank = json.loads(rank_evidence_path.read_text())
    failures: list[str] = []
    deterministic = {
        key: value for key, value in evidence.items()
        if key not in EVIDENCE_OMITTED
    }
    if evidence.get("result_digest") != stable_hash(deterministic):
        failures.append("global proposal evidence digest differs")
    if evidence.get("schema_version") != "m04r-global-bound-proposal-gate-v1":
        failures.append("global proposal evidence schema differs")
    contract = packed_bound_search_contract()
    if evidence.get("contract_digest") != contract["digest"]:
        failures.append("global proposal contract digest differs")
    if evidence.get("real_forward_outcomes_accessed") is not False:
        failures.append("global proposal evidence accessed outcomes or labels")
    rank_verification = verify_m04r_quantized_ranks(rank_evidence_path)
    if not rank_verification.passed:
        failures.append("bound-rank evidence fails its independent verifier")
    build_deterministic = {
        key: value for key, value in build.items()
        if key not in FULL_BUILD_OMITTED
    }
    if (
        build.get("schema_version") != "m04r-packed-bound-full-build-v1"
        or build.get("result_digest") != stable_hash(build_deterministic)
    ):
        failures.append("full-build evidence schema or digest differs")

    generation_id = str(evidence.get("generation_id", ""))
    try:
        loaded = load_packed_generation(
            store_root, generation_id, verify_content=True,
            validate_records=False,
        )
    except Exception as exc:
        failures.append(
            f"global proposal physical generation is invalid: {type(exc).__name__}:{exc}"
        )
        loaded = None
    if generation_id != str(build.get("generation_id", "")):
        failures.append("global proposal generation differs from full build")
    if evidence.get("full_build_evidence_digest") != build.get("result_digest"):
        failures.append("global proposal full-build evidence binding differs")
    if build.get("gate_passed") is not True or build.get("shadow_generation") is not True:
        failures.append("bound generation is not a passed shadow generation")
    if (store_root / "active.json").exists():
        failures.append("global proposal unexpectedly activated the shadow generation")
    if loaded is not None:
        provenance = loaded.manifest.get("provenance", {})
        if provenance.get("rank_evidence_digest") != rank.get("result_digest"):
            failures.append("physical generation rank-evidence binding differs")

    forward = evidence.get("forward") or {}
    reverse = evidence.get("reverse") or {}
    invariant = all(forward.get(field) == reverse.get(field) for field in INVARIANT_FIELDS)
    if not invariant:
        failures.append("forward/reverse or unequal-block proposal output differs")
    if int(evidence.get("block_rows_forward", 0)) < 1 or (
        int(evidence.get("block_rows_reverse", 0))
        == int(evidence.get("block_rows_forward", 0))
    ):
        failures.append("proposal evidence lacks unequal positive block sizes")

    query_id = str(evidence.get("query_episode_id", ""))
    selection = evidence.get("selection_method") or {}
    if (
        query_id != str(build.get("benchmark_selection", {}).get("query_episode_id", ""))
        or selection != build.get("benchmark_selection")
    ):
        failures.append("global proposal query was not preregistered by the full-pack gate")
    rank_cases = {
        str(case.get("query_episode_id")): case
        for case in rank.get("authority_cases", [])
    }
    rank_case = rank_cases.get(query_id)
    if len(rank_cases) != 12 or rank_case is None:
        failures.append("global proposal cannot resolve 12-case rank evidence")
        expected_eligible = -1
        target_ids: set[str] = set()
    else:
        expected_eligible = int(rank_case.get("eligible_rows", -1))
        target_ids = {
            str(row.get("episode_id", "")) for row in rank_case.get("targets", [])
        }
    all_rank_targets = [
        int(target.get("upper_rank", 1_001))
        for case in rank_cases.values() for target in case.get("targets", [])
    ]
    all_12_supported = (
        len(rank_cases) == 12 and len(all_rank_targets) == 240
        and max(all_rank_targets, default=1_001) <= 1_000
        and rank.get("top_1000_recall_passed") is True
    )
    composite_ids = [str(value) for value in forward.get("composite_episode_ids", [])]
    retained = target_ids.issubset(composite_ids) and len(target_ids) == 20
    retention = evidence.get("target_retention") or {}
    if not retained or (
        int(retention.get("retained_count", -1)) != 20
        or int(retention.get("target_count", -1)) != 20
        or retention.get("missing_ids") != []
        or retention.get("rank_evidence_digest") != rank.get("result_digest")
        or retention.get("rank_evidence_all_12_top_1000") is not True
        or int(retention.get("rank_evidence_maximum_target_rank", -1))
        != max(all_rank_targets, default=-1)
    ):
        failures.append("global composite does not retain the selected exact authority")
    if not all_12_supported:
        failures.append("all-12 rank evidence does not support composite top-1000")

    expected_rows_scanned = (
        int(loaded.manifest["eligible_row_count"]) if loaded is not None else -1
    )
    row_accounting = (
        int(forward.get("rows_scanned", -1)) == expected_rows_scanned
        and int(forward.get("eligible_rows", -1)) == expected_eligible
        and int(forward.get("eligible_rows", -1))
        == int(forward.get("eligible_main_rows", -2))
        + int(forward.get("eligible_overflow_rows", -3))
    )
    if not row_accounting:
        failures.append("global proposal row accounting differs")
    route_quotas = forward.get("route_quotas") or {}
    route_counts = forward.get("route_counts") or {}
    route_contract = (
        route_quotas == DEFAULT_ROUTE_QUOTAS
        and set(route_counts) == set(DEFAULT_ROUTE_QUOTAS)
        and all(
            int(route_counts.get(route, -1)) == quota
            for route, quota in DEFAULT_ROUTE_QUOTAS.items()
        )
        and int(forward.get("composite_count", -1)) == 1_000
        and len(composite_ids) == 1_000
        and len(set(composite_ids)) == 1_000
        and int(forward.get("candidate_count", -1)) > 1_000
    )
    if not route_contract:
        failures.append("global proposal route quotas or non-displacing union differ")
    clean_selection = all((
        int(forward.get("duplicate_candidates", -1)) == 0,
        int(forward.get("future_candidates", -1)) == 0,
        int(forward.get("overlapping_same_symbol_candidates", -1)) == 0,
        int(forward.get("overflow_candidates", 0)) > 0,
    ))
    if not clean_selection:
        failures.append("global proposals contain duplicates/temporal violations or omit overflow")
    prior_scans = build.get("warm_scans_second") or []
    prior_digest = str(
        prior_scans[0].get("top_1000_digest", "") if prior_scans else ""
    )
    score_equivalent = (
        forward.get("top_1000_score_digest") == prior_digest
        and evidence.get("prior_top_1000_score_digest") == prior_digest
    )
    if not score_equivalent:
        failures.append("global composite scores differ from M04R-06C")
    latency = max(
        float(evidence.get("forward_seconds", float("inf"))),
        float(evidence.get("reverse_seconds", float("inf"))),
    )
    peak_rss = float(evidence.get("peak_rss_mb", float("inf")))
    expected_gates = {
        "full_row_accounting": row_accounting,
        "forward_reverse_and_block_invariance": invariant,
        "top_1000_scores_equal_m04r_06c": score_equivalent,
        "all_selected_authority_targets_retained": retained,
        "all_12_rank_evidence_supports_composite_quota": all_12_supported,
        "no_duplicates_or_temporal_violations": all((
            int(forward.get("duplicate_candidates", -1)) == 0,
            int(forward.get("future_candidates", -1)) == 0,
            int(forward.get("overlapping_same_symbol_candidates", -1)) == 0,
        )),
        "composite_quota_preserved": int(forward.get("composite_count", -1)) == 1_000,
        "overflow_fallback_exercised": int(forward.get("overflow_candidates", 0)) > 0,
        "rss_within_768_mib": peak_rss <= 768,
        "each_scan_within_600_seconds": latency <= 600,
    }
    if evidence.get("gates") != expected_gates:
        failures.append("global proposal reported gate flags differ")
    if bool(evidence.get("gate_passed")) != all(expected_gates.values()):
        failures.append("global proposal overall pass flag differs")
    if not all(expected_gates.values()):
        failures.append("one or more global proposal gates fail")

    unique = tuple(sorted(set(failures)))
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "contract_digest": contract["digest"],
        "generation_id": generation_id,
        "query_episode_id": query_id,
        "rows_scanned": int(forward.get("rows_scanned", -1)),
        "eligible_rows": int(forward.get("eligible_rows", -1)),
        "candidate_count": int(forward.get("candidate_count", -1)),
        "candidate_digest": forward.get("candidate_digest"),
        "top_1000_score_digest": forward.get("top_1000_score_digest"),
        "maximum_authority_target_rank": max(all_rank_targets, default=-1),
        "forward_seconds": float(evidence.get("forward_seconds", -1)),
        "reverse_seconds": float(evidence.get("reverse_seconds", -1)),
        "peak_rss_mb": peak_rss,
        "evidence_digest": evidence.get("result_digest"),
        "real_forward_outcomes_accessed": False,
    }
    result_payload = {
        "schema_version": SCHEMA_VERSION,
        "metrics": metrics,
        "failures": list(unique),
        "contract_digest": contract["digest"],
    }
    return GlobalProposalVerificationResult(
        not unique, metrics, unique, stable_hash(result_payload),
    )


def write_m04r_global_bound_proposal_verification(
    result: GlobalProposalVerificationResult, output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    machine = output_dir / "m04r-global-bound-proposal.json"
    html = output_dir / "m04r-global-bound-proposal.html"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "passed": result.passed,
        "metrics": result.metrics,
        "failures": list(result.failures),
        "result_digest": result.result_digest,
    }
    temporary = machine.with_suffix(machine.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(machine)
    status = "PASS" if result.passed else "FAIL"
    failure_items = "".join(
        f"<li>{escape(value)}</li>" for value in result.failures
    ) or "<li>None</li>"
    temporary = html.with_suffix(html.suffix + ".tmp")
    temporary.write_text(f"""<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>M04R-07 global proposal verification</title><style>body{{font-family:system-ui;max-width:1100px;margin:2rem auto}}pre{{white-space:pre-wrap}}.pass{{color:#075}}.fail{{color:#a20}}</style></head><body><h1>M04R-07 global proposal verification: <span class=\"{status.lower()}\">{status}</span></h1><p>Independent physical-generation, evidence-binding, row-accounting, target-retention, determinism, temporal-safety and resource verification.</p><p>Result <code>{result.result_digest}</code>.</p><pre>{escape(json.dumps(result.metrics, indent=2, sort_keys=True))}</pre><ul>{failure_items}</ul></body></html>""")
    temporary.replace(html)
    return machine, html
