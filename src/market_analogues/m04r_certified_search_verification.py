from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
from typing import Any

from .authority import AUTHORITY_SCHEMA_VERSION
from .certified_packed_search import (
    CERTIFIED_PACKED_SEARCH_VERSION, certified_packed_search_contract,
)
from .m04r_full_pack_verification import EVIDENCE_OMITTED as FULL_BUILD_OMITTED
from .packed_bound_store import load_packed_generation
from .types import stable_hash


SCHEMA_VERSION = "m04r-certified-packed-search-verification-v1"
EVIDENCE_SCHEMA = "m04r-certified-packed-search-gate-v1"
EVIDENCE_OMITTED = {
    "created_at", "run_seconds", "peak_rss_mb", "result_digest",
}
TOTAL_TOLERANCE = 1e-7
COMPONENT_TOLERANCE = 1e-6
BOUND_TOLERANCE = 1e-12


@dataclass(frozen=True)
class CertifiedSearchVerificationResult:
    passed: bool
    metrics: dict[str, Any]
    failures: tuple[str, ...]
    result_digest: str


def _evidence_digest_payload(evidence: dict[str, Any]) -> dict[str, Any]:
    payload = {
        key: value for key, value in evidence.items()
        if key not in EVIDENCE_OMITTED
    }
    payload["runs"] = [
        {key: value for key, value in run.items() if key != "seconds"}
        for run in evidence.get("runs", [])
    ]
    return payload


def _certificate_digest(run: dict[str, Any]) -> str:
    certificate = run.get("certificate") or {}
    matches = run.get("matches") or []
    deterministic = {
        "schema_version": certificate.get("schema_version"),
        "contract_digest": certificate.get("contract_digest"),
        "generation_id": certificate.get("generation_id"),
        "query_episode_id": certificate.get("query_episode_id"),
        "input_digest": certificate.get("input_digest"),
        "eligible_candidates": certificate.get("eligible_candidates"),
        "exact_evaluated": certificate.get("exact_evaluated"),
        "safely_pruned": certificate.get("safely_pruned"),
        "stopped_early": certificate.get("stopped_early"),
        "stop_threshold_hex": float(certificate.get("stop_threshold")).hex(),
        "next_lower_bound_hex": (
            float(certificate["next_lower_bound"]).hex()
            if certificate.get("next_lower_bound") is not None else None
        ),
        "maximum_quantized_bound_excess_hex": float(
            certificate.get("maximum_quantized_bound_excess")
        ).hex(),
        "rounds": certificate.get("rounds"),
        "matches": [{
            "episode_id": match.get("episode_id"),
            "total_hex": float(match.get("total_distance")).hex(),
            "components": {
                key: float(value).hex()
                for key, value in sorted(
                    (match.get("component_distances") or {}).items()
                )
            },
            "alignment": match.get("alignment"),
        } for match in matches],
        "real_forward_outcomes_accessed": False,
    }
    return stable_hash(deterministic)


def _authority_deltas(
    matches: list[dict[str, Any]], authority_matches: list[dict[str, Any]],
) -> tuple[bool, bool, float, float]:
    if len(matches) != len(authority_matches) or not matches:
        return False, False, float("inf"), float("inf")
    ids_equal = [row.get("episode_id") for row in matches] == [
        row.get("episode_id") for row in authority_matches
    ]
    alignments_equal = [row.get("alignment") for row in matches] == [
        row.get("alignment") for row in authority_matches
    ]
    total_delta = max(
        abs(float(left.get("total_distance")) - float(right.get("total_distance")))
        for left, right in zip(matches, authority_matches)
    )
    component_delta = 0.0
    for left, right in zip(matches, authority_matches):
        left_components = left.get("component_distances") or {}
        right_components = right.get("component_distances") or {}
        if set(left_components) != set(right_components):
            return ids_equal, alignments_equal, total_delta, float("inf")
        component_delta = max(component_delta, *(
            abs(float(left_components[name]) - float(right_components[name]))
            for name in left_components
        ))
    return ids_equal, alignments_equal, total_delta, component_delta


def verify_m04r_certified_packed_search(
    evidence_path: Path,
    store_root: Path,
    full_build_evidence_path: Path,
    authority_path: Path,
) -> CertifiedSearchVerificationResult:
    evidence = json.loads(evidence_path.read_text())
    build = json.loads(full_build_evidence_path.read_text())
    authority = json.loads(authority_path.read_text())
    failures: list[str] = []
    contract = certified_packed_search_contract()

    if evidence.get("schema_version") != EVIDENCE_SCHEMA:
        failures.append("certified-search evidence schema differs")
    if evidence.get("contract_digest") != contract["digest"]:
        failures.append("certified-search contract digest differs")
    if evidence.get("result_digest") != stable_hash(_evidence_digest_payload(evidence)):
        failures.append("certified-search evidence digest differs")
    if evidence.get("real_forward_outcomes_accessed") is not False:
        failures.append("certified-search evidence accessed outcomes or labels")

    authority_content = dict(authority)
    authority_content.pop("authority_digest", None)
    authority_valid = all((
        authority.get("schema_version") == AUTHORITY_SCHEMA_VERSION,
        authority.get("authority_digest") == stable_hash(authority_content),
        authority.get("result_digest") == stable_hash(authority.get("matches")),
        authority.get("result_digest") == authority.get("repeated_digest"),
    ))
    if not authority_valid:
        failures.append("frozen authority integrity differs")
    if evidence.get("authority_digest") != authority.get("authority_digest"):
        failures.append("certified-search authority binding differs")

    build_deterministic = {
        key: value for key, value in build.items()
        if key not in FULL_BUILD_OMITTED
    }
    build_valid = all((
        build.get("schema_version") == "m04r-packed-bound-full-build-v1",
        build.get("result_digest") == stable_hash(build_deterministic),
        build.get("gate_passed") is True,
        build.get("shadow_generation") is True,
        build.get("real_forward_outcomes_accessed") is False,
    ))
    if not build_valid:
        failures.append("full packed-build evidence integrity or gate differs")
    if evidence.get("full_build_evidence_digest") != build.get("result_digest"):
        failures.append("certified-search full-build binding differs")

    generation_id = str(evidence.get("generation_id", ""))
    if generation_id != str(build.get("generation_id", "")):
        failures.append("certified-search generation differs from full build")
    try:
        loaded = load_packed_generation(
            store_root, generation_id, verify_content=True, validate_records=False,
        )
    except Exception as exc:
        failures.append(
            f"certified-search physical generation is invalid: "
            f"{type(exc).__name__}:{exc}"
        )
        loaded = None
    if (store_root / "active.json").exists():
        failures.append("certified-search unexpectedly activated shadow generation")
    if loaded is not None and loaded.generation_id != generation_id:
        failures.append("certified-search physical generation ID differs")

    query_id = str(evidence.get("query_episode_id", ""))
    selected_query = str(
        (build.get("benchmark_selection") or {}).get("query_episode_id", "")
    )
    if query_id != selected_query or query_id != authority.get("query_episode_id"):
        failures.append("certified-search query is not the preregistered worst authority")
    authority_matches = authority.get("matches") or []
    runs = evidence.get("runs") or []
    required_runs = int(evidence.get("required_runs", -1))
    repeats_complete = (
        required_runs >= 2 and len(runs) == required_runs
        and int(evidence.get("completed_runs", -1)) == required_runs
    )
    if not repeats_complete:
        failures.append("certified-search repeated runs are incomplete")

    controls = [run.get("controls") or {} for run in runs]
    controls_independent = (
        len(controls) >= 2
        and len({(row.get("block_rows"), row.get("workers")) for row in controls})
        == len(controls)
        and all(int(row.get("block_rows", 0)) > 0 for row in controls)
        and all(int(row.get("workers", 0)) > 0 for row in controls)
    )
    if not controls_independent:
        failures.append("certified-search repeats lack distinct valid controls")

    stable_fields = ("matches", "certificate", "certificate_digest")
    repeated_identical = bool(runs) and all(
        run.get(field) == runs[0].get(field)
        for run in runs[1:] for field in stable_fields
    )
    if not repeated_identical:
        failures.append("certified-search repeated results differ")

    all_ids_equal = True
    all_alignments_equal = True
    maximum_total_delta = 0.0
    maximum_component_delta = 0.0
    certificates_valid = True
    accounting = True
    stopping = True
    bound_safe = True
    group_accounting = True
    for run in runs:
        matches = run.get("matches") or []
        ids, alignments, total_delta, component_delta = _authority_deltas(
            matches, authority_matches,
        )
        all_ids_equal &= ids
        all_alignments_equal &= alignments
        maximum_total_delta = max(maximum_total_delta, total_delta)
        maximum_component_delta = max(maximum_component_delta, component_delta)
        certificate = run.get("certificate") or {}
        expected_digest = _certificate_digest(run)
        certificates_valid &= all((
            certificate.get("schema_version") == CERTIFIED_PACKED_SEARCH_VERSION,
            certificate.get("contract_digest") == contract["digest"],
            certificate.get("generation_id") == generation_id,
            certificate.get("query_episode_id") == query_id,
            certificate.get("result_digest") == expected_digest,
            run.get("certificate_digest") == expected_digest,
        ))
        eligible = int(certificate.get("eligible_candidates", -1))
        exact = int(certificate.get("exact_evaluated", -1))
        pruned = int(certificate.get("safely_pruned", -1))
        accounting &= eligible > 0 and exact >= len(matches) and exact + pruned == eligible
        next_bound = certificate.get("next_lower_bound")
        threshold = float(certificate.get("stop_threshold", float("inf")))
        rounds = certificate.get("rounds") or []
        stopping &= all((
            len(matches) == 20,
            certificate.get("stopped_early") is True,
            next_bound is not None and float(next_bound) > threshold,
            bool(rounds) and rounds[-1].get("certified") is True,
            int(rounds[-1].get("selected_rows", -1)) == len(matches),
            float(rounds[-1].get("constrained_threshold", float("inf")))
            == threshold,
            rounds[-1].get("next_lower_bound") == next_bound,
        ))
        bound_safe &= float(
            certificate.get("maximum_quantized_bound_excess", float("inf"))
        ) <= BOUND_TOLERANCE
        groups = int(certificate.get("materialization_groups", -1))
        group_accounting &= groups > 0 and groups == (
            int(certificate.get("sparse_symbols", -2))
            + int(certificate.get("batch_symbols", -3))
        )
    if not certificates_valid:
        failures.append("certified-search certificate digest or binding differs")
    if not accounting:
        failures.append("certified-search candidate accounting differs")
    if not stopping:
        failures.append("certified-search strict stopping certificate differs")
    if not bound_safe:
        failures.append("certified-search quantized bound exceeds native bound")
    if not group_accounting:
        failures.append("certified-search materialization-group accounting differs")
    if not all_ids_equal:
        failures.append("certified-search ordered IDs differ from exact authority")
    if not all_alignments_equal:
        failures.append("certified-search alignments differ from exact authority")
    if maximum_total_delta > TOTAL_TOLERANCE:
        failures.append("certified-search total distance exceeds practical tolerance")
    if maximum_component_delta > COMPONENT_TOLERANCE:
        failures.append("certified-search component distance exceeds practical tolerance")

    seconds = [float(run.get("seconds", float("inf"))) for run in runs]
    reported_seconds = evidence.get("run_seconds") or []
    runtime_valid = seconds == reported_seconds and bool(seconds) and max(seconds) <= 600
    rss_valid = float(evidence.get("peak_rss_mb", float("inf"))) <= 1_536
    if not runtime_valid:
        failures.append("certified-search runtime evidence differs or exceeds 600 seconds")
    if not rss_valid:
        failures.append("certified-search RSS exceeds 1536 MiB")

    failed_attempt = evidence.get("failed_attempt") or {}
    failure_preserved = all((
        failed_attempt.get("status") == "interrupted performance failure",
        int(failed_attempt.get("elapsed_before_interrupt_seconds", 0)) >= 600,
        int(failed_attempt.get("observed_rss_mb", 0)) > 0,
        int(failed_attempt.get("exit_code", 0)) == 130,
    ))
    if not failure_preserved:
        failures.append("certified-search rejected performance attempt is not preserved")

    expected_gates = {
        "required_repeats_complete": repeats_complete,
        "repeated_results_identical": repeated_identical,
        "ordered_ids_equal_authority": all_ids_equal,
        "alignments_equal_authority": all_alignments_equal,
        "total_delta_within_1e_7": maximum_total_delta <= TOTAL_TOLERANCE,
        "component_delta_within_1e_6": (
            maximum_component_delta <= COMPONENT_TOLERANCE
        ),
        "maximum_runtime_within_600_seconds": runtime_valid,
        "rss_within_1536_mib": rss_valid,
        "candidate_accounting": accounting,
        "strict_stopping": stopping,
    }
    if evidence.get("gates") != expected_gates:
        failures.append("certified-search reported gate flags differ")
    if evidence.get("gate_passed") != all(expected_gates.values()):
        failures.append("certified-search overall gate flag differs")
    if not all(expected_gates.values()):
        failures.append("one or more certified-search gates fail")

    unique = tuple(sorted(set(failures)))
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "contract_digest": contract["digest"],
        "generation_id": generation_id,
        "query_episode_id": query_id,
        "completed_runs": len(runs),
        "certificate_digest": (
            runs[0].get("certificate_digest") if runs else None
        ),
        "eligible_candidates": (
            int((runs[0].get("certificate") or {}).get("eligible_candidates", -1))
            if runs else -1
        ),
        "exact_evaluated": (
            int((runs[0].get("certificate") or {}).get("exact_evaluated", -1))
            if runs else -1
        ),
        "maximum_total_delta": maximum_total_delta,
        "maximum_component_delta": maximum_component_delta,
        "maximum_seconds": max(seconds, default=-1),
        "peak_rss_mb": float(evidence.get("peak_rss_mb", -1)),
        "physical_generation_verified": loaded is not None,
        "failed_attempt_preserved": failure_preserved,
        "evidence_digest": evidence.get("result_digest"),
        "real_forward_outcomes_accessed": False,
    }
    result_payload = {
        "schema_version": SCHEMA_VERSION,
        "metrics": metrics,
        "failures": list(unique),
        "contract_digest": contract["digest"],
    }
    return CertifiedSearchVerificationResult(
        not unique, metrics, unique, stable_hash(result_payload),
    )


def write_m04r_certified_search_verification(
    result: CertifiedSearchVerificationResult, output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    machine = output_dir / "m04r-certified-packed-search.json"
    html = output_dir / "m04r-certified-packed-search.html"
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
    temporary.write_text(f"""<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>M04R-08A certified packed search verification</title><style>body{{font-family:system-ui;max-width:1100px;margin:2rem auto}}pre{{white-space:pre-wrap}}.pass{{color:#075}}.fail{{color:#a20}}</style></head><body><h1>M04R-08A independent verification: <span class=\"{status.lower()}\">{status}</span></h1><p>Independent evidence, authority, physical-generation, certificate-digest, repeatability, numeric, stopping, accounting and resource verification. Forward outcomes and setup labels are excluded.</p><p>Result <code>{result.result_digest}</code>.</p><pre>{escape(json.dumps(result.metrics, indent=2, sort_keys=True))}</pre><ul>{failure_items}</ul></body></html>""")
    temporary.replace(html)
    return machine, html
