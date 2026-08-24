from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
from typing import Any

from .proposal_v2 import (
    LAYOUTS, PROPOSAL_POOLS, PROPOSAL_ROUTES, proposal_v2_contract,
    proposal_v2_route_admitted, proposal_v2_storage_bytes,
)
from .types import stable_hash


M04R_PROPOSAL_VERIFIER_SCHEMA = "m04r-proposal-v2-verification-v1"
EVIDENCE_NONDETERMINISTIC_FIELDS = {
    "created_at", "elapsed_seconds", "projection_seconds", "scoring_seconds",
    "peak_rss_mb", "end_to_end_rows_per_second", "result_digest",
}


@dataclass(frozen=True)
class M04RProposalVerificationResult:
    passed: bool
    metrics: dict[str, Any]
    failures: tuple[str, ...]
    contract: dict[str, Any]
    result_digest: str


def verify_m04r_proposal_v2(
    evidence_path: Path,
) -> M04RProposalVerificationResult:
    evidence = json.loads(evidence_path.read_text())
    contract = proposal_v2_contract()
    failures: list[str] = []
    deterministic = {
        key: value for key, value in evidence.items()
        if key not in EVIDENCE_NONDETERMINISTIC_FIELDS
    }
    if evidence.get("result_digest") != stable_hash(deterministic):
        failures.append("proposal evidence digest differs")
    if evidence.get("schema_version") != "m04r-proposal-v2-authority-gate-v1":
        failures.append("proposal evidence schema differs")
    if evidence.get("contract_digest") != contract["digest"]:
        failures.append("proposal contract digest differs")
    if evidence.get("real_forward_outcomes_accessed") is not False:
        failures.append("proposal evidence accessed outcomes or labels")
    if not evidence.get("is_full_universe"):
        failures.append("proposal evidence is not a full-universe evaluation")
    evaluated = int(evidence.get("evaluated_symbols", 0))
    universe = int(evidence.get("universe_symbols", -1))
    if evaluated != universe or universe < 11_000:
        failures.append("proposal universe symbol accounting differs")
    prefixes = evidence.get("source_prefixes", {})
    if (
        not isinstance(prefixes, dict)
        or len(prefixes) != universe
        or int(evidence.get("source_prefix_count", -1)) != universe
        or evidence.get("source_scope_digest") != stable_hash(prefixes)
    ):
        failures.append("proposal source-prefix scope digest differs")

    synthetic = evidence.get("synthetic_gate", {})
    synthetic_cases = synthetic.get("cases", [])
    if (
        not synthetic.get("all_passed")
        or int(synthetic.get("case_count", 0)) != 30
        or len(synthetic_cases) != 30
        or int(synthetic.get("adversarial_local_cap_depth", 0)) < 27
        or any(
            not row.get("passed")
            or int(row.get("clone_input_position", 0)) != 28
            or int(row.get("route_ranks", {}).get("composite", 0)) != 1
            for row in synthetic_cases
        )
    ):
        failures.append("proposal synthetic/local-cap gate is incomplete")

    cases = evidence.get("authority_cases", [])
    if len(cases) != 12 or int(evidence.get("authority_case_count", 0)) != 12:
        failures.append("proposal evidence does not contain 12 authorities")
    dtype_pass = {
        (dimensions, dtype): True
        for dimensions in sorted(LAYOUTS) for dtype in ("float32", "float16")
    }
    target_count = 0
    for case in cases:
        if (
            not case.get("authority_digest")
            or not case.get("row_accounting_matches")
            or not case.get("targets_seen_once")
            or int(case.get("eligible_rows", -1))
            != int(case.get("authority_eligible_rows", -2))
        ):
            failures.append(
                f"authority accounting differs for {case.get('query_episode_id', '?')}"
            )
        layouts = case.get("layouts", {})
        for dimensions in sorted(LAYOUTS):
            layout = layouts.get(str(dimensions), {})
            for dtype in ("float32", "float16"):
                result = layout.get(dtype, {})
                targets = result.get("target_ranks", [])
                target_count += len(targets)
                if len(targets) != 20:
                    failures.append(
                        f"layout {dimensions}/{dtype} lacks 20 authority targets"
                    )
                    dtype_pass[(dimensions, dtype)] = False
                    continue
                recalculated = {}
                for pool in sorted(PROPOSAL_POOLS):
                    admitted = 0
                    for target in targets:
                        lower_routes = target.get("lower_route_ranks", {})
                        upper_routes = target.get("upper_route_ranks", {})
                        ties = target.get("ties_including_target", {})
                        if (
                            set(lower_routes) != set(PROPOSAL_ROUTES)
                            or set(upper_routes) != set(PROPOSAL_ROUTES)
                            or set(ties) != {*PROPOSAL_ROUTES, "composite"}
                            or any(
                                int(lower_routes[route]) < 1
                                or int(upper_routes[route]) < int(lower_routes[route])
                                or int(ties[route]) < 1
                                for route in PROPOSAL_ROUTES
                            )
                            or int(target.get("lower_composite_rank", 0)) < 1
                            or int(target.get("upper_composite_rank", 0))
                            < int(target.get("lower_composite_rank", 0))
                            or int(ties.get("composite", 0)) < 1
                        ):
                            failures.append(
                                f"invalid conservative rank interval in {dimensions}/{dtype}"
                            )
                        admitted += int(proposal_v2_route_admitted(
                            {route: int(upper_routes[route]) for route in PROPOSAL_ROUTES},
                            int(target["upper_composite_rank"]), pool,
                        ))
                    reported = result.get("pool_recalls", {}).get(str(pool), {})
                    if (
                        int(reported.get("admitted", -1)) != admitted
                        or int(reported.get("total", -1)) != 20
                        or not reported.get("conservative_upper_tie_rank")
                    ):
                        failures.append(
                            f"reported pool {pool} recall differs in {dimensions}/{dtype}"
                        )
                    recalculated[pool] = admitted
                passed = recalculated[20_000] == 20 and recalculated[10_000] >= 19
                if bool(result.get("passed")) != passed:
                    failures.append(
                        f"layout pass flag differs in {dimensions}/{dtype}"
                    )
                dtype_pass[(dimensions, dtype)] &= passed

    overflow = evidence.get("quantization_overflow_rows", {})
    for dimensions in sorted(LAYOUTS):
        dtype_pass[(dimensions, "float16")] &= int(
            overflow.get(str(dimensions), -1)
        ) == 0
        storage = evidence.get("storage_projection", {}).get(str(dimensions), {})
        for dtype in ("float32", "float16"):
            if int(storage.get(dtype, {}).get("signature_bytes", -1)) != (
                proposal_v2_storage_bytes(dimensions, dtype)
            ):
                failures.append(f"storage bytes differ for {dimensions}/{dtype}")
    reported_float32 = evidence.get("layout_float32_pass", {})
    reported_float16 = evidence.get("layout_float16_pass", {})
    for dimensions in sorted(LAYOUTS):
        if bool(reported_float32.get(str(dimensions))) != dtype_pass[
            (dimensions, "float32")
        ]:
            failures.append(f"float32 summary differs for layout {dimensions}")
        if bool(reported_float16.get(str(dimensions))) != dtype_pass[
            (dimensions, "float16")
        ]:
            failures.append(f"float16 summary differs for layout {dimensions}")
    selected = next((
        dimensions for dimensions in sorted(LAYOUTS)
        if dtype_pass[(dimensions, "float32")]
    ), None)
    selected_dtype = None if selected is None else (
        "float16" if dtype_pass[(selected, "float16")] else "float32"
    )
    if evidence.get("selected_dimensions") != selected:
        failures.append("selected proposal dimensions differ")
    if evidence.get("selected_dtype") != selected_dtype:
        failures.append("selected proposal dtype differs")
    expected_bytes = (
        proposal_v2_storage_bytes(selected, selected_dtype)
        if selected is not None and selected_dtype is not None else None
    )
    if evidence.get("selected_signature_bytes") != expected_bytes:
        failures.append("selected proposal bytes differ")
    if bool(evidence.get("all_cases_passed")) != (selected is not None):
        failures.append("overall proposal pass flag differs")
    if selected is None:
        failures.append("no proposal layout satisfies the frozen selection rule")

    metrics = {
        "schema_version": M04R_PROPOSAL_VERIFIER_SCHEMA,
        "contract_digest": contract["digest"],
        "authority_cases": len(cases),
        "authority_target_layout_rows": target_count,
        "universe_symbols": universe,
        "selected_dimensions": selected,
        "selected_dtype": selected_dtype,
        "selected_signature_bytes": expected_bytes,
        "float16_overflow_rows": overflow.get(str(selected)) if selected else None,
        "evidence_digest": evidence.get("result_digest"),
        "real_forward_outcomes_accessed": False,
    }
    verifier_payload = {
        "schema_version": M04R_PROPOSAL_VERIFIER_SCHEMA,
        "metrics": metrics,
        "failures": sorted(set(failures)),
        "contract": contract,
    }
    unique_failures = tuple(sorted(set(failures)))
    return M04RProposalVerificationResult(
        not unique_failures, metrics, unique_failures, contract,
        stable_hash(verifier_payload),
    )


def write_m04r_proposal_verification(
    result: M04RProposalVerificationResult,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    machine = output_dir / "m04r-proposal-v2.json"
    html = output_dir / "m04r-proposal-v2.html"
    contract_path = output_dir / "proposal-v2-contract.json"
    payload = {
        "schema_version": M04R_PROPOSAL_VERIFIER_SCHEMA,
        "passed": result.passed,
        "metrics": result.metrics,
        "failures": list(result.failures),
        "contract": result.contract,
        "result_digest": result.result_digest,
    }
    for path, value in ((machine, payload), (contract_path, result.contract)):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    status = "PASS" if result.passed else "FAIL"
    failures = "".join(
        f"<li>{escape(value)}</li>" for value in result.failures
    ) or "<li>None</li>"
    temporary = html.with_suffix(html.suffix + ".tmp")
    temporary.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>M04R-05 proposal v2</title><style>body{{font-family:system-ui;max-width:1200px;margin:2rem auto;background:#f5f7f8}}header,section{{background:white;padding:1.2rem;margin:1rem;border:1px solid #ddd;border-radius:10px}}pre{{white-space:pre-wrap}}.pass{{color:#075}}.fail{{color:#a20}}</style></head><body><header><h1>M04R-05 proposal v2: <span class="{status.lower()}">{status}</span></h1><p>Independent verification of the full-universe compact proposal Pareto selection.</p><p>Contract <code>{result.contract['digest']}</code> · Result <code>{result.result_digest}</code></p></header><section><h2>Metrics</h2><pre>{escape(json.dumps(result.metrics, indent=2, sort_keys=True))}</pre></section><section><h2>Failures</h2><ul>{failures}</ul></section><section><h2>Frozen contract</h2><pre>{escape(json.dumps(result.contract, indent=2, sort_keys=True))}</pre></section></body></html>""")
    temporary.replace(html)
    return machine, html, contract_path
