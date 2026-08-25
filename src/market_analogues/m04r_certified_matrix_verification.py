from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
from typing import Any

import numpy as np

from .authority import AUTHORITY_SCHEMA_VERSION
from .certified_packed_search import certified_packed_search_contract
from .m04r_certified_search_verification import _certificate_digest
from .m04r_full_pack_verification import EVIDENCE_OMITTED as FULL_BUILD_OMITTED
from .packed_bound_store import load_packed_generation
from .types import stable_hash


SCHEMA_VERSION = "m04r-certified-packed-search-all12-verification-v1"
MATRIX_SCHEMA = "m04r-certified-packed-search-all12-v1"
CASE_SCHEMA = "m04r-certified-packed-search-case-v1"
CASE_OMITTED = {"created_at", "seconds", "peak_rss_mb", "result_digest"}
MATRIX_OMITTED = {
    "created_at", "started_at", "p95_seconds", "maximum_seconds",
    "total_seconds", "peak_rss_mb", "result_digest",
}
TOTAL_TOLERANCE = 1e-7
COMPONENT_TOLERANCE = 1e-6
BOUND_TOLERANCE = 1e-12
LEGACY_CONTROLS = {"block_rows": 2_048, "workers": 8}
V5_CONTROLS = {
    "block_rows": 2_048,
    "workers": 8,
    "initial_frontier_rows": 16_384,
    "requested_positions": True,
    "hybrid_requested_positions": False,
    "vector_lower_bounds": True,
    "deferred_alignments": True,
}


@dataclass(frozen=True)
class CertifiedMatrixVerificationResult:
    passed: bool
    evidence_gate_passed: bool
    metrics: dict[str, Any]
    failures: tuple[str, ...]
    performance_failures: tuple[str, ...]
    result_digest: str


def _contract_for_controls(
    controls: dict[str, Any],
) -> dict[str, Any] | None:
    if controls == LEGACY_CONTROLS:
        return certified_packed_search_contract()
    if controls == V5_CONTROLS:
        return certified_packed_search_contract(
            requested_positions=True, vector_lower_bounds=True,
            deferred_alignments=True,
        )
    return None


def _case_deterministic(case: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in case.items() if key not in CASE_OMITTED
    }


def _matrix_deterministic(evidence: dict[str, Any]) -> dict[str, Any]:
    payload = {
        key: value for key, value in evidence.items()
        if key not in MATRIX_OMITTED
    }
    payload["cases"] = [
        _case_deterministic(case) for case in evidence.get("cases", [])
    ]
    return payload


def _authority_valid(authority: dict[str, Any]) -> bool:
    content = dict(authority)
    claimed = content.pop("authority_digest", None)
    return all((
        authority.get("schema_version") == AUTHORITY_SCHEMA_VERSION,
        claimed == stable_hash(content),
        authority.get("result_digest") == stable_hash(authority.get("matches")),
        authority.get("result_digest") == authority.get("repeated_digest"),
    ))


def _comparison(
    matches: list[dict[str, Any]], expected: list[dict[str, Any]],
) -> dict[str, Any]:
    same_length = len(matches) == len(expected) and len(matches) == 20
    ids_equal = same_length and [row.get("episode_id") for row in matches] == [
        row.get("episode_id") for row in expected
    ]
    alignments_equal = same_length and [row.get("alignment") for row in matches] == [
        row.get("alignment") for row in expected
    ]
    component_names_equal = same_length and all(
        set(left.get("component_distances") or {})
        == set(right.get("component_distances") or {})
        for left, right in zip(matches, expected)
    )
    if not same_length:
        total_delta = component_delta = None
    else:
        total_delta = max(
            abs(float(left["total_distance"]) - float(right["total_distance"]))
            for left, right in zip(matches, expected)
        )
        component_delta = (
            max(
                abs(
                    float(left["component_distances"][name])
                    - float(right["component_distances"][name])
                )
                for left, right in zip(matches, expected)
                for name in left["component_distances"]
            ) if component_names_equal else None
        )
    return {
        "ordered_ids_equal_authority": ids_equal,
        "alignments_equal_authority": alignments_equal,
        "component_names_equal_authority": component_names_equal,
        "maximum_total_delta": total_delta,
        "maximum_component_delta": component_delta,
    }


def _case_gates(
    case: dict[str, Any], comparison: dict[str, Any],
) -> dict[str, bool]:
    certificate = case.get("certificate") or {}
    try:
        return {
            "ordered_ids_equal_authority": bool(
                comparison["ordered_ids_equal_authority"]
            ),
            "alignments_equal_authority": bool(
                comparison["alignments_equal_authority"]
            ),
            "component_names_equal_authority": bool(
                comparison["component_names_equal_authority"]
            ),
            "total_delta_within_1e_7": (
                comparison["maximum_total_delta"] is not None
                and comparison["maximum_total_delta"] <= TOTAL_TOLERANCE
            ),
            "component_delta_within_1e_6": (
                comparison["maximum_component_delta"] is not None
                and comparison["maximum_component_delta"] <= COMPONENT_TOLERANCE
            ),
            "candidate_accounting": (
                int(certificate["exact_evaluated"])
                + int(certificate["safely_pruned"])
                == int(certificate["eligible_candidates"])
            ),
            "strict_stopping": (
                certificate["next_lower_bound"] is not None
                and float(certificate["next_lower_bound"])
                > float(certificate["stop_threshold"])
            ),
            "quantized_bound_safe": (
                float(certificate["maximum_quantized_bound_excess"])
                <= BOUND_TOLERANCE
            ),
            "runtime_within_600_seconds": float(case["seconds"]) <= 600,
            "rss_within_1536_mib": float(case["peak_rss_mb"]) <= 1_536,
        }
    except (KeyError, TypeError, ValueError):
        return {}


def verify_m04r_certified_matrix(
    evidence_path: Path,
    store_root: Path,
    full_build_evidence_path: Path,
    authority_dir: Path,
) -> CertifiedMatrixVerificationResult:
    evidence = json.loads(evidence_path.read_text())
    build = json.loads(full_build_evidence_path.read_text())
    failures: list[str] = []
    performance_failures: list[str] = []
    cases = evidence.get("cases") or []
    profile_controls = (
        cases[0].get("controls") if cases
        and isinstance(cases[0].get("controls"), dict) else {}
    )
    contract = _contract_for_controls(profile_controls)
    if contract is None:
        failures.append("certified matrix execution controls are unsupported")
        contract = certified_packed_search_contract()

    if evidence.get("schema_version") != MATRIX_SCHEMA:
        failures.append("certified matrix evidence schema differs")
    if evidence.get("contract_digest") != contract["digest"]:
        failures.append("certified matrix contract digest differs")
    if evidence.get("result_digest") != stable_hash(_matrix_deterministic(evidence)):
        failures.append("certified matrix evidence digest differs")
    if evidence.get("real_forward_outcomes_accessed") is not False:
        failures.append("certified matrix accessed outcomes or setup labels")

    build_content = {
        key: value for key, value in build.items()
        if key not in FULL_BUILD_OMITTED
    }
    build_valid = all((
        build.get("schema_version") == "m04r-packed-bound-full-build-v1",
        build.get("result_digest") == stable_hash(build_content),
        build.get("gate_passed") is True,
        build.get("shadow_generation") is True,
        build.get("real_forward_outcomes_accessed") is False,
    ))
    if not build_valid:
        failures.append("full packed-build evidence integrity or gate differs")
    if evidence.get("full_build_evidence_digest") != build.get("result_digest"):
        failures.append("certified matrix full-build binding differs")
    generation_id = str(evidence.get("generation_id", ""))
    if generation_id != str(build.get("generation_id", "")):
        failures.append("certified matrix generation differs from full build")
    try:
        loaded = load_packed_generation(
            store_root, generation_id, verify_content=True, validate_records=False,
        )
    except Exception as exc:
        failures.append(
            f"certified matrix physical generation is invalid: "
            f"{type(exc).__name__}:{exc}"
        )
        loaded = None
    if (store_root / "active.json").exists():
        failures.append("certified matrix unexpectedly activated shadow generation")

    authority_paths = sorted(authority_dir.glob("*.json"))
    authorities = [json.loads(path.read_text()) for path in authority_paths]
    if len(authorities) != 12:
        failures.append(f"expected 12 authorities, found {len(authorities)}")
    if not all(_authority_valid(authority) for authority in authorities):
        failures.append("one or more frozen authorities fail integrity checks")
    expected_ids = [str(authority.get("query_episode_id", "")) for authority in authorities]
    authority_by_id = {
        str(authority.get("query_episode_id", "")): authority
        for authority in authorities
    }
    case_ids = [str(case.get("query_episode_id", "")) for case in cases]
    if (
        evidence.get("expected_query_episode_ids") != expected_ids
        or evidence.get("completed_query_episode_ids") != expected_ids
        or case_ids != expected_ids
        or int(evidence.get("completed_cases", -1)) != 12
        or evidence.get("failed_cases") != []
    ):
        failures.append("certified matrix case order/completion accounting differs")

    all_case_evidence_valid = True
    all_correct = True
    all_case_gate_flags: list[bool] = []
    seconds: list[float] = []
    rss_values: list[float] = []
    heavy_cases: list[dict[str, Any]] = []
    for case in cases:
        query_id = str(case.get("query_episode_id", ""))
        authority = authority_by_id.get(query_id)
        if authority is None:
            failures.append(f"matrix case has no authority: {query_id}")
            all_case_evidence_valid = False
            continue
        comparison = _comparison(case.get("matches") or [], authority.get("matches") or [])
        gates = _case_gates(case, comparison)
        certificate = case.get("certificate") or {}
        try:
            certificate_digest = _certificate_digest(case)
            case_valid = all((
                case.get("schema_version") == CASE_SCHEMA,
                case.get("status") == "completed",
                case.get("result_digest") == stable_hash(_case_deterministic(case)),
                case.get("contract_digest") == contract["digest"],
                case.get("generation_id") == generation_id,
                case.get("full_build_evidence_digest") == build.get("result_digest"),
                case.get("authority_digest") == authority.get("authority_digest"),
                case.get("real_forward_outcomes_accessed") is False,
                case.get("controls") == profile_controls,
                certificate.get("schema_version") == contract["schema_version"],
                certificate.get("contract_digest") == contract["digest"],
                certificate.get("generation_id") == generation_id,
                certificate.get("query_episode_id") == query_id,
                certificate.get("result_digest") == certificate_digest,
                case.get("certificate_digest") == certificate_digest,
                int(certificate.get("eligible_candidates", -1))
                == int((authority.get("certificate") or {}).get("eligible_candidates", -2)),
                int(certificate.get("materialization_groups", -1))
                == int(certificate.get("sparse_symbols", -2))
                + int(certificate.get("batch_symbols", -3)),
                bool(certificate.get("rounds")),
                (certificate.get("rounds") or [{}])[-1].get("certified") is True,
                case.get("gates") == gates,
                case.get("gate_passed") == all(gates.values()),
                all(case.get(key) == value for key, value in comparison.items()),
            ))
        except (KeyError, TypeError, ValueError):
            case_valid = False
        if not case_valid:
            failures.append(f"certified matrix case evidence differs: {query_id}")
        all_case_evidence_valid &= case_valid
        correctness_names = (
            "ordered_ids_equal_authority", "alignments_equal_authority",
            "component_names_equal_authority", "total_delta_within_1e_7",
            "component_delta_within_1e_6", "candidate_accounting",
            "strict_stopping", "quantized_bound_safe",
        )
        case_correct = bool(gates) and all(gates.get(name) for name in correctness_names)
        all_correct &= case_correct
        all_case_gate_flags.append(bool(gates) and all(gates.values()))
        seconds.append(float(case.get("seconds", float("inf"))))
        rss_values.append(float(case.get("peak_rss_mb", float("inf"))))
        if not gates.get("runtime_within_600_seconds", False):
            performance_failures.append(
                f"{query_id} runtime {float(case.get('seconds', 0)):.2f}s exceeds 600s"
            )
            heavy_cases.append({
                "query_episode_id": query_id,
                "seconds": float(case.get("seconds", 0)),
                "exact_evaluated": int(certificate.get("exact_evaluated", -1)),
            })

    p95 = float(np.percentile(seconds, 95)) if seconds else float("inf")
    maximum = max(seconds, default=float("inf"))
    total = sum(seconds)
    peak_rss = max(rss_values, default=float("inf"))
    if p95 > 300:
        performance_failures.append(f"p95 runtime {p95:.2f}s exceeds 300s")
    if maximum > 600:
        performance_failures.append(f"maximum runtime {maximum:.2f}s exceeds 600s")
    if peak_rss > 1_536:
        performance_failures.append(f"peak RSS {peak_rss:.2f} MiB exceeds 1536 MiB")
    bindings = {
        (
            case.get("query_episode_id"),
            (case.get("certificate") or {}).get("input_digest"),
            case.get("certificate_digest"),
        ) for case in cases
    }
    expected_matrix_gates = {
        "all_12_completed": len(cases) == 12 and case_ids == expected_ids,
        "all_case_gates_passed": len(cases) == 12 and all(all_case_gate_flags),
        "all_certificates_unique_to_query": len(bindings) == len(cases),
        "p95_within_300_seconds": p95 <= 300,
        "maximum_within_600_seconds": maximum <= 600,
        "rss_within_1536_mib": peak_rss <= 1_536,
    }
    if evidence.get("gates") != expected_matrix_gates:
        failures.append("certified matrix reported gate flags differ")
    expected_gate_passed = all(expected_matrix_gates.values())
    if evidence.get("gate_passed") != expected_gate_passed:
        failures.append("certified matrix overall gate flag differs")
    reported_numeric = all((
        abs(float(evidence.get("p95_seconds", float("inf"))) - p95) <= 1e-12,
        abs(float(evidence.get("maximum_seconds", float("inf"))) - maximum) <= 1e-12,
        abs(float(evidence.get("total_seconds", float("inf"))) - total) <= 1e-9,
        abs(float(evidence.get("peak_rss_mb", float("inf"))) - peak_rss) <= 1e-12,
    ))
    if not reported_numeric:
        failures.append("certified matrix aggregate numeric metrics differ")
    if not all_case_evidence_valid:
        failures.append("one or more certified matrix case artifacts are invalid")
    if not all_correct:
        failures.append("one or more certified matrix cases fail exact correctness")

    unique = tuple(sorted(set(failures)))
    performance = tuple(sorted(set(performance_failures)))
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "contract_digest": contract["digest"],
        "generation_id": generation_id,
        "completed_cases": len(cases),
        "all_cases_exact": all_correct,
        "p95_seconds": p95,
        "maximum_seconds": maximum,
        "total_seconds": total,
        "peak_rss_mb": peak_rss,
        "heavy_cases": heavy_cases,
        "evidence_gate_passed": expected_gate_passed,
        "physical_generation_verified": loaded is not None,
        "evidence_digest": evidence.get("result_digest"),
        "real_forward_outcomes_accessed": False,
    }
    result_payload = {
        "schema_version": SCHEMA_VERSION,
        "metrics": metrics,
        "failures": list(unique),
        "performance_failures": list(performance),
        "contract_digest": contract["digest"],
    }
    return CertifiedMatrixVerificationResult(
        not unique, expected_gate_passed, metrics, unique, performance,
        stable_hash(result_payload),
    )


def write_m04r_certified_matrix_verification(
    result: CertifiedMatrixVerificationResult, output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    machine = output_dir / "m04r-certified-packed-search-all12.json"
    html = output_dir / "m04r-certified-packed-search-all12.html"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "passed": result.passed,
        "evidence_gate_passed": result.evidence_gate_passed,
        "metrics": result.metrics,
        "failures": list(result.failures),
        "performance_failures": list(result.performance_failures),
        "result_digest": result.result_digest,
    }
    temporary = machine.with_suffix(machine.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(machine)
    evidence_status = "PASS" if result.evidence_gate_passed else "REJECTED"
    verifier_status = "PASS" if result.passed else "FAIL"
    items = "".join(
        f"<li>{escape(value)}</li>"
        for value in (*result.failures, *result.performance_failures)
    ) or "<li>None</li>"
    temporary = html.with_suffix(html.suffix + ".tmp")
    temporary.write_text(f"""<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>M04R-08B all-12 verification</title><style>body{{font-family:system-ui;max-width:1100px;margin:2rem auto}}pre{{white-space:pre-wrap}}.pass{{color:#075}}.fail{{color:#a20}}</style></head><body><h1>Independent evidence verification: <span class=\"{'pass' if result.passed else 'fail'}\">{verifier_status}</span></h1><h2>Certified-search baseline decision: <span class=\"{'pass' if result.evidence_gate_passed else 'fail'}\">{evidence_status}</span></h2><p>A verifier PASS means the evidence is authentic and internally correct. The baseline remains rejected when its frozen performance gates fail. No forward outcomes or setup labels are accessed.</p><p>Result <code>{result.result_digest}</code>.</p><pre>{escape(json.dumps(result.metrics, indent=2, sort_keys=True))}</pre><ul>{items}</ul></body></html>""")
    temporary.replace(html)
    return machine, html
