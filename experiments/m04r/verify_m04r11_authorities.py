"""Independent verifier for the sealed M04R-11 authority matrix."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from typing import Any

import pandas as pd

from market_analogues.adapters import file_fingerprint, source_from_spec
from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.config import load_config
from market_analogues.m04r_certified_search_verification import _certificate_digest
from market_analogues.m04r_validation_registry import validate_m04r_validation_registry
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.types import EpisodeKey, InstrumentKey, stable_hash


SCHEMA_VERSION = "m04r11-certified-authority-verification-v1"
CASE_OMITTED = {
    "created_at", "proposal_seconds", "exact_seconds", "peak_rss_mb",
    "result_digest", "checkpoint_integrity_digest",
}
MATRIX_OMITTED = {
    "created_at", "elapsed_seconds", "p95_exact_seconds",
    "maximum_exact_seconds", "total_exact_seconds", "maximum_worker_rss_mb",
    "result_digest",
}


def _case_result_digest(payload: dict[str, Any]) -> str:
    return stable_hash({key: value for key, value in payload.items() if key not in CASE_OMITTED})


def _checkpoint_digest(payload: dict[str, Any]) -> str:
    return stable_hash({
        key: value for key, value in payload.items()
        if key not in {"created_at", "checkpoint_integrity_digest"}
    })


def _independent_case_failures(
    payload: dict[str, Any], case: dict[str, Any], contract: dict[str, Any],
    source: Any, packed_provenance_digest: str,
) -> list[str]:
    failures: list[str] = []
    query_id = str(case["episode_id"])
    instrument = InstrumentKey("nasdaq", str(case["symbol"]))
    bars = source.load(instrument)
    cutoff = pd.Timestamp(case["cutoff"])
    query_bars = bars[pd.to_datetime(bars.timestamp) <= cutoff].tail(int(case["lookback"]))
    query_start = pd.Timestamp(query_bars.timestamp.iloc[0])
    latest_eligible = pd.Timestamp(query_bars.timestamp.iloc[-61])
    key = EpisodeKey(instrument, cutoff, int(case["lookback"]), str(case["representation_version"]))
    if key.id != query_id or payload.get("query_episode_id") != query_id:
        failures.append("query episode identity differs")
    stock_prefix = asdict(causal_prefix_digest(bars, cutoff))
    benchmark = source.load_benchmark()
    benchmark_prefix = asdict(causal_prefix_digest(benchmark, cutoff)) if benchmark is not None else None
    if stock_prefix != case["stock_prefix"] or payload.get("query_stock_prefix") != stock_prefix:
        failures.append("query stock prefix differs")
    if benchmark_prefix != case["benchmark_prefix"] or payload.get("query_benchmark_prefix") != benchmark_prefix:
        failures.append("query benchmark prefix differs")
    request = contract["search_contract"]["request"]
    input_digest = stable_hash({
        "query_stock_prefix": stock_prefix,
        "query_benchmark_prefix": benchmark_prefix,
        "request": {
            "search_datasets": ["nasdaq"],
            "quality_tiers": request["quality_tiers"],
            "top_k": request["top_k"],
            "cross_dataset": request["cross_dataset"],
            "deduplicate_overlaps": request["deduplicate_overlaps"],
            "max_per_instrument": request["max_per_instrument"],
            "minimum_history_gap_bars": request["minimum_history_gap_bars"],
        },
        "packed_provenance_digest": packed_provenance_digest,
    })
    certificate = payload.get("certificate") or {}
    matches = payload.get("matches") or []
    if certificate.get("input_digest") != input_digest:
        failures.append("certificate input digest differs")
    if certificate.get("generation_id") != contract["generation_id"]:
        failures.append("certificate generation differs")
    if certificate.get("contract_digest") != contract["search_contract"][
        "certified_search_contract_digest"
    ]:
        failures.append("certificate search contract differs")
    if certificate.get("result_digest") != _certificate_digest({
        "certificate": certificate, "matches": matches,
    }):
        failures.append("certificate digest differs")
    if len(matches) != 20 or len({row.get("episode_id") for row in matches}) != 20:
        failures.append("authority does not contain 20 unique matches")
    ordered = [(float(row["total_distance"]), str(row["episode_id"])) for row in matches]
    if ordered != sorted(ordered):
        failures.append("authority order is not distance/episode-ID stable")
    counts = pd.Series([str(row.get("symbol")) for row in matches]).value_counts()
    if len(counts) and int(counts.max()) > int(request["max_per_instrument"]):
        failures.append("authority violates per-instrument diversity cap")
    for match in matches:
        match_cutoff = pd.Timestamp(match["cutoff"])
        if match_cutoff > latest_eligible:
            failures.append("authority admits a temporally ineligible match")
            break
        if str(match["symbol"]) == str(case["symbol"]) and match_cutoff >= query_start:
            failures.append("authority admits an overlapping same-symbol match")
            break
        if match.get("quality_tier") not in request["quality_tiers"]:
            failures.append("authority admits a disallowed quality tier")
            break
    exact = int(certificate.get("exact_evaluated", -1))
    pruned = int(certificate.get("safely_pruned", -1))
    eligible = int(certificate.get("eligible_candidates", -2))
    if exact + pruned != eligible:
        failures.append("certificate candidate accounting differs")
    next_bound = certificate.get("next_lower_bound")
    stopped = bool(certificate.get("stopped_early"))
    if not (
        (stopped and next_bound is not None
         and float(next_bound) > float(certificate.get("stop_threshold")))
        or (not stopped and pruned == 0)
    ):
        failures.append("certificate has no strict stop or literal exhaustion")
    if float(certificate.get("maximum_quantized_bound_excess", float("inf"))) > 1e-12:
        failures.append("certificate quantized bound exceeds tolerance")
    if payload.get("result_digest") != _case_result_digest(payload):
        failures.append("authority case result digest differs")
    if payload.get("checkpoint_integrity_digest") != _checkpoint_digest(payload):
        failures.append("authority checkpoint integrity digest differs")
    if payload.get("real_forward_outcomes_accessed") is not False:
        failures.append("authority outcome-access marker differs")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    source = source_from_spec(config.datasets["nasdaq"])
    failures = list(validate_m04r_validation_registry(source, args.registry.parent))
    registry = json.loads(args.registry.read_text())
    contract_path = args.authority_root / "authority-contract.json"
    matrix_path = args.authority_root / "authority-matrix.json"
    seal_path = args.authority_root / "SEALED.json"
    if not all(path.exists() for path in (contract_path, matrix_path, seal_path)):
        raise SystemExit("authority contract, matrix, or seal is missing")
    contract = json.loads(contract_path.read_text())
    matrix = json.loads(matrix_path.read_text())
    seal = json.loads(seal_path.read_text())
    contract_without_digest = {key: value for key, value in contract.items() if key != "contract_digest"}
    if stable_hash(contract_without_digest) != contract.get("contract_digest"):
        failures.append("authority contract digest differs")
    runner_path = Path(__file__).with_name("m04r11_build_authorities.py")
    if file_fingerprint(runner_path) != contract.get("runner_sha256"):
        failures.append("authority runner source hash differs")
    loaded = load_packed_generation(
        args.full_root / "store", str(contract["generation_id"]),
        verify_content=True, validate_records=False,
    )
    cases = list(registry["cases_data"])
    if contract.get("expected_query_episode_ids") != [case["episode_id"] for case in cases]:
        failures.append("authority contract membership/order differs")
    verified = []
    per_case_failures: dict[str, list[str]] = {}
    for case in cases:
        path = args.authority_root / "cases" / f"{case['episode_id']}.json"
        if not path.exists():
            per_case_failures[case["case_id"]] = ["authority checkpoint is missing"]
            continue
        payload = json.loads(path.read_text())
        observed = _independent_case_failures(
            payload, case, contract, source, str(loaded.manifest["provenance_digest"]),
        )
        if observed:
            per_case_failures[case["case_id"]] = observed
        else:
            verified.append(payload)
    if per_case_failures:
        failures.append("one or more authority cases failed independent reconstruction")
    matrix_without = {key: value for key, value in matrix.items() if key not in MATRIX_OMITTED}
    if stable_hash(matrix_without) != matrix.get("result_digest"):
        failures.append("authority matrix digest differs")
    matrix_ids = [row.get("query_episode_id") for row in matrix.get("cases", [])]
    if matrix_ids != [case["episode_id"] for case in cases]:
        failures.append("authority matrix case order differs")
    seal_without_digest = {key: value for key, value in seal.items() if key != "seal_digest"}
    if stable_hash(seal_without_digest) != seal.get("seal_digest"):
        failures.append("authority seal digest differs")
    if seal.get("authority_matrix_digest") != matrix.get("result_digest"):
        failures.append("authority seal does not bind matrix")
    gates = {
        "registry_runtime_validation_passed": not validate_m04r_validation_registry(
            source, args.registry.parent,
        ),
        "physical_generation_rehashed": True,
        "all_60_cases_independently_reconstructed": len(verified) == 60,
        "matrix_and_seal_reconstructed": not any(
            "matrix" in value or "seal" in value for value in failures
        ),
        "candidate_results_remained_unopened": (
            matrix.get("candidate_results_opened") is False
            and seal.get("candidate_results_opened") is False
        ),
        "real_forward_outcomes_excluded": (
            matrix.get("real_forward_outcomes_accessed") is False
            and seal.get("real_forward_outcomes_accessed") is False
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "registry_digest": registry["registry_digest"],
        "contract_digest": contract["contract_digest"],
        "generation_id": contract["generation_id"],
        "authority_matrix_digest": matrix.get("result_digest"),
        "authority_seal_digest": seal.get("seal_digest"),
        "verified_cases": len(verified),
        "per_case_failures": per_case_failures,
        "gates": gates, "failures": list(dict.fromkeys(failures)),
        "real_forward_outcomes_accessed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    payload["passed"] = all(gates.values()) and not payload["failures"]
    payload["result_digest"] = stable_hash({
        key: value for key, value in payload.items()
        if key not in {"created_at", "result_digest"}
    })
    args.output_root.mkdir(parents=True, exist_ok=True)
    json_path = args.output_root / "m04r11-authority-verification.json"
    html_path = args.output_root / "m04r11-authority-verification.html"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    status = "PASS" if payload["passed"] else "FAIL"
    html_path.write_text(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>M04R-11 authority verification</title><style>"
        "body{font-family:system-ui;max-width:1200px;margin:2rem auto}"
        "pre{white-space:pre-wrap}</style></head><body><h1>M04R-11 authority "
        f"verification: {status}</h1><pre>{escape(json.dumps(payload, indent=2, sort_keys=True))}"
        "</pre></body></html>"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(html_path)
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
