from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
import math
from pathlib import Path
from typing import Any

from .quantized_bound import quantized_bound_contract
from .types import stable_hash


SCHEMA_VERSION = "m04r-quantized-bound-rank-verification-v1"
OMITTED = {
    "created_at", "elapsed_seconds", "representation_seconds",
    "quantization_seconds", "scoring_seconds", "peak_rss_mb", "result_digest",
}


@dataclass(frozen=True)
class QuantizedRankVerificationResult:
    passed: bool
    metrics: dict[str, Any]
    failures: tuple[str, ...]
    result_digest: str


def verify_m04r_quantized_ranks(path: Path) -> QuantizedRankVerificationResult:
    evidence = json.loads(path.read_text())
    failures = []
    deterministic = {
        key: value for key, value in evidence.items() if key not in OMITTED
    }
    if evidence.get("result_digest") != stable_hash(deterministic):
        failures.append("quantized rank evidence digest differs")
    if evidence.get("schema_version") != "m04r-quantized-bound-rank-gate-v1":
        failures.append("quantized rank evidence schema differs")
    contract = quantized_bound_contract()
    if evidence.get("contract_digest") != contract["digest"]:
        failures.append("quantized bound contract digest differs")
    if evidence.get("real_forward_outcomes_accessed") is not False:
        failures.append("quantized rank evidence accessed outcomes")
    universe = int(evidence.get("universe_symbols", 0))
    prefixes = evidence.get("source_prefixes", {})
    benchmark_prefix = evidence.get("benchmark_prefix", {})
    if (
        universe < 11_000 or len(prefixes) != universe
        or int(evidence.get("source_prefix_count", -1)) != universe
        or not benchmark_prefix.get("digest")
        or int(benchmark_prefix.get("rows", 0)) <= 0
        or evidence.get("source_scope_digest") != stable_hash({
            "stocks": prefixes, "benchmark": benchmark_prefix,
        })
    ):
        failures.append("quantized rank source-prefix scope differs")
    cases = evidence.get("authority_cases", [])
    if len(cases) != 12 or int(evidence.get("authority_case_count", 0)) != 12:
        failures.append("quantized rank evidence lacks 12 authorities")
    maximum = 0
    all_top_500 = True
    all_top_100 = True
    all_top_1000 = True
    all_accounting = True
    query_ids = [str(case.get("query_episode_id", "")) for case in cases]
    authority_digests = [str(case.get("authority_digest", "")) for case in cases]
    if len(set(query_ids)) != len(query_ids) or any(not value for value in query_ids):
        failures.append("quantized authority query IDs are absent or duplicated")
    if (
        len(set(authority_digests)) != len(authority_digests)
        or any(not value for value in authority_digests)
    ):
        failures.append("quantized authority digests are absent or duplicated")
    if int(evidence.get("stride", -1)) != 5:
        failures.append("quantized rank stride differs")
    for case in cases:
        eligible_rows = int(case.get("eligible_rows", -1))
        accounting = (
            case.get("row_accounting_matches") is True
            and case.get("targets_seen_once") is True
            and eligible_rows
            == int(case.get("authority_eligible_rows", -2))
            and eligible_rows > 0
        )
        all_accounting &= accounting
        if not accounting:
            failures.append(
                f"rank row accounting differs for {case.get('query_episode_id', '?')}"
            )
        targets = case.get("targets", [])
        if len(targets) != 20:
            failures.append("quantized authority does not contain 20 targets")
            continue
        upper = []
        episode_ids = [str(target.get("episode_id", "")) for target in targets]
        if len(set(episode_ids)) != 20 or any(not value for value in episode_ids):
            failures.append("quantized target IDs are absent or duplicated")
        for target in targets:
            lower_rank = int(target.get("lower_rank", 0))
            upper_rank = int(target.get("upper_rank", 0))
            ties = int(target.get("ties_including_target", 0))
            score = float(target.get("quantized_bound", float("nan")))
            if (
                lower_rank < 1 or upper_rank < lower_rank
                or upper_rank > eligible_rows or ties < 1
                or upper_rank - lower_rank + 1 != ties
                or not math.isfinite(score) or score < 0
            ):
                failures.append("quantized target rank interval is invalid")
            upper.append(upper_rank)
        case_maximum = max(upper)
        maximum = max(maximum, case_maximum)
        if int(case.get("maximum_target_rank", -1)) != case_maximum:
            failures.append("reported quantized case maximum rank differs")
        for quota in (100, 500, 1_000, 2_000):
            recall = sum(rank <= quota for rank in upper) / 20
            if float(case.get("recall", {}).get(str(quota), -1)) != recall:
                failures.append(f"reported quantized recall differs at {quota}")
        all_top_100 &= all(rank <= 100 for rank in upper)
        all_top_500 &= all(rank <= 500 for rank in upper)
        all_top_1000 &= all(rank <= 1_000 for rank in upper)
    if int(evidence.get("maximum_target_rank", -1)) != maximum:
        failures.append("reported quantized maximum target rank differs")
    if bool(evidence.get("top_100_recall_passed")) != all_top_100:
        failures.append("top-100 quantized summary differs")
    if bool(evidence.get("top_1000_recall_passed")) != all_top_1000:
        failures.append("top-1000 quantized summary differs")
    if bool(evidence.get("all_row_accounting_passed")) != all_accounting:
        failures.append("quantized accounting summary differs")
    expected_pass = all_top_1000 and all_accounting
    if bool(evidence.get("rank_gate_passed")) != expected_pass:
        failures.append("quantized rank pass flag differs")
    overflow_rows = int(evidence.get("overflow_rows", -1))
    if overflow_rows < 0 or (
        overflow_rows and not evidence.get("overflow_examples")
    ):
        failures.append("quantized overflow census is incomplete")
    for example in evidence.get("overflow_examples", []):
        if (
            not example.get("symbol") or not example.get("cutoff")
            or float(example.get("routing_bound", float("nan"))) != 0.0
        ):
            failures.append("quantized overflow example is invalid")
    if evidence.get("overflow_routing_policy") != (
        "no quantized row emitted; route with universal safe bound zero and "
        "require exact/float32 sidecar"
    ):
        failures.append("quantized overflow routing policy differs")
    if not expected_pass:
        failures.append("top-1000 quantized bound route does not retain all targets")
    unique = tuple(sorted(set(failures)))
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "contract_digest": contract["digest"],
        "authority_cases": len(cases),
        "universe_symbols": universe,
        "benchmark_prefix_digest": benchmark_prefix.get("digest"),
        "maximum_target_rank": maximum,
        "top_100_recall_passed": all_top_100,
        "top_500_recall_passed": all_top_500,
        "top_1000_recall_passed": all_top_1000,
        "overflow_rows": overflow_rows,
        "evidence_digest": evidence.get("result_digest"),
        "real_forward_outcomes_accessed": False,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "metrics": metrics,
        "failures": list(unique),
        "contract_digest": contract["digest"],
    }
    return QuantizedRankVerificationResult(
        not unique, metrics, unique, stable_hash(payload),
    )


def write_m04r_quantized_rank_verification(
    result: QuantizedRankVerificationResult, output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    machine = output_dir / "m04r-quantized-bound-ranks.json"
    html = output_dir / "m04r-quantized-bound-ranks.html"
    payload = {
        "schema_version": SCHEMA_VERSION, "passed": result.passed,
        "metrics": result.metrics, "failures": list(result.failures),
        "result_digest": result.result_digest,
    }
    temporary = machine.with_suffix(machine.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(machine)
    status = "PASS" if result.passed else "FAIL"
    failures = "".join(
        f"<li>{escape(value)}</li>" for value in result.failures
    ) or "<li>None</li>"
    temporary = html.with_suffix(html.suffix + ".tmp")
    temporary.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>M04R quantized ranks</title><style>body{{font-family:system-ui;max-width:1100px;margin:2rem auto}}pre{{white-space:pre-wrap}}.pass{{color:#075}}.fail{{color:#a20}}</style></head><body><h1>Quantized bound rank gate: <span class="{status.lower()}">{status}</span></h1><p>Independent verification of full-universe target ranks and fail-closed overflow routing.</p><p>Result <code>{result.result_digest}</code>.</p><pre>{escape(json.dumps(result.metrics, indent=2, sort_keys=True))}</pre><ul>{failures}</ul></body></html>""")
    temporary.replace(html)
    return machine, html
