"""Open sealed candidate pools once and compare them with sealed exact truth."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from typing import Any

from market_analogues.adapters import file_fingerprint
from market_analogues.adapters import source_from_spec
from market_analogues.config import load_config
from market_analogues.episodes import build_episode
from market_analogues.m04r_candidate_evidence import (
    candidate_checkpoint_integrity_digest, candidate_semantic_digest,
    reconstructed_candidate_digest as reconstruct_truth_blind_candidate_digest,
    scan_result_digest,
    validate_truth_blind_candidate_case,
)
from market_analogues.m04r_validation_registry import validate_m04r_validation_registry
from market_analogues.representation import represent, representation_input_digest
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import InstrumentKey
from market_analogues.types import stable_hash


SCHEMA = "candidate-recall-comparison-v2"
FROZEN_REGISTRY_DIGEST = "0a4da732f91375a091775cb04e6e77c8d136ade47d7f4d16508a2d9a6555361e"
FROZEN_GENERATION_ID = "9fc6ae0ec4451133d8006897162f3443803fd30a3c44b78e80a476fbb18bb483"
FROZEN_PROPOSAL_CONTRACT_DIGEST = "059db6d78bbe62e7588e1fe2408fe244040b501d190bdf7b211a7ecd7aadb96c"
FROZEN_CASE_ORDER_DIGEST = "db1741d50eab19217d6264f051ca3bccf2663c9a54a7e6e846e8689410fe2b47"
FROZEN_ROUTE_QUOTAS = {
    "candle_volatility": 64, "coarse": 64, "composite": 1_000,
    "market_context": 64, "price": 64, "stage": 64,
    "structural": 64, "volume_shock": 64,
}
FROZEN_REQUEST = {
    "cross_dataset": False, "deduplicate_overlaps": True,
    "max_per_instrument": 3, "minimum_history_gap_bars": 60,
    "quality_tiers": ["A", "B"], "top_k": 20,
}
PRODUCER_CONTRACT_SCHEMA = "candidate-recall-producer-contract-v2"
PRODUCER_SCAN_PROTOCOL = {
    "execution": "serial fresh spawned process per query",
    "engine": "bounded ordered four-thread legacy-v1 scan",
    "outer_threads": 4,
    "numba_threads_per_scorer": 1,
    "maximum_in_flight_blocks": 4,
    "reduction": "strict requested physical block order into unchanged stable route heaps",
    "cold": {
        "advice": "POSIX_FADV_DONTNEED", "block_rows": 4_096,
        "order": "forward",
    },
    "warm_first": {"block_rows": 4_097, "order": "reverse"},
    "warm_second": {"block_rows": 4_093, "order": "forward"},
}
PRODUCER_PERFORMANCE_LIMITS = {
    "cold_seconds": 120.0, "warm_second_seconds": 60.0,
    "rss_mib": 1_024.0,
}
AUTHORITY_CASE_OMITTED = {
    "created_at", "proposal_seconds", "amortized_proposal_seconds",
    "exact_seconds", "final_exact_seconds", "frontier_attempt_measurements",
    "peak_rss_mb", "result_digest", "checkpoint_integrity_digest",
}
CASE_OMITTED = {
    "created_at", "cold_seconds", "warm_first_seconds", "warm_second_seconds",
    "cold_task_seconds", "peak_rss_mb", "result_digest",
    "checkpoint_integrity_digest",
}
MATRIX_OMITTED = {"created_at", "elapsed_seconds", "result_digest"}


def candidate_case_digest(payload: dict[str, Any]) -> str:
    return candidate_semantic_digest(payload)


def candidate_matrix_digest(payload: dict[str, Any]) -> str:
    return stable_hash({key: value for key, value in payload.items() if key not in MATRIX_OMITTED})


def seal_digest(payload: dict[str, Any]) -> str:
    return stable_hash({key: value for key, value in payload.items() if key not in {"created_at", "seal_digest"}})


def _refuse_failed_candidate_root(candidate_root: Path) -> None:
    failure_path = candidate_root / "FAILED.json"
    if not failure_path.exists():
        return
    payload = json.loads(failure_path.read_text())
    deterministic = {
        key: value for key, value in payload.items()
        if key not in {"created_at", "failure_digest"}
    }
    if not all((
        payload.get("schema_version")
        == "candidate-recall-producer-terminal-failure-v1",
        payload.get("status") == "terminal_performance_failure_after_interruption",
        payload.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
        payload.get("producer_contract_digest")
        == "e6d971081f288375df3ad66baad0300eb2d452860b464d6d9c111a6c75d657ad",
        payload.get("candidate_pools_sealed") is False,
        payload.get("authority_results_opened") is False,
        payload.get("candidate_authority_comparison_opened") is False,
        payload.get("resume_authorized") is False,
        payload.get("failure_digest") == stable_hash(deterministic),
    )):
        raise ValueError("candidate root contains invalid terminal failure evidence")
    raise ValueError("candidate root is terminally failed; truth comparison is forbidden")


def checkpoint_integrity_digest(payload: dict[str, Any]) -> str:
    return candidate_checkpoint_integrity_digest(payload)


def _scan_digest(invariant: dict[str, Any]) -> str:
    return scan_result_digest(invariant, FROZEN_PROPOSAL_CONTRACT_DIGEST)


def validate_candidate_case(
    candidate: dict[str, Any], registry_case: dict[str, Any],
    authority: dict[str, Any], contract: dict[str, Any],
) -> list[str]:
    failures = list(validate_truth_blind_candidate_case(
        candidate, expected_case=registry_case,
        registry_digest=FROZEN_REGISTRY_DIGEST,
        generation_id=FROZEN_GENERATION_ID,
        proposal_contract_digest=FROZEN_PROPOSAL_CONTRACT_DIGEST,
        producer_contract_digest=str(contract["contract_digest"]),
        route_quotas=contract["route_quotas"],
        physical_rows=int(contract["physical_rows"]),
    ))
    try:
        if authority.get("result_digest") != stable_hash({
            key: value for key, value in authority.items() if key not in AUTHORITY_CASE_OMITTED
        }):
            failures.append("authority case digest differs")
        invariants = candidate.get("scan_invariants") or []
        if not invariants or int(invariants[0]["eligible_rows"]) != int(authority["certificate"]["eligible_candidates"]):
            failures.append("candidate/authority eligible accounting differs")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        failures.append(f"malformed authority binding:{type(exc).__name__}:{exc}")
    return failures


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def reconstructed_candidate_digest(rows: list[dict[str, Any]]) -> str:
    return reconstruct_truth_blind_candidate_digest(rows)


def _expected_producer_implementation_manifest() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    producer = Path(__file__).with_name("m04r11_candidate_matrix.py").resolve()
    paths = [producer, *sorted((repository / "src").rglob("*.py"))]
    files = {
        str(path.relative_to(repository)): file_fingerprint(path) for path in paths
    }
    return {"files": files, "digest": stable_hash(files)}


def validate_preopen_producer_contract(
    contract: dict[str, Any], registry: dict[str, Any],
) -> tuple[str, ...]:
    """Validate all execution/code bindings before authority truth is opened."""
    failures: list[str] = []
    try:
        manifest_path = Path(str(contract["physical_manifest_path"]))
        ordered_ids = [str(case["episode_id"]) for case in registry["cases_data"]]
        if not all((
            contract.get("schema_version") == PRODUCER_CONTRACT_SCHEMA,
            contract.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
            contract.get("ordered_query_ids") == ordered_ids,
            contract.get("ordered_query_ids_digest") == FROZEN_CASE_ORDER_DIGEST,
            contract.get("generation_id") == FROZEN_GENERATION_ID,
            contract.get("proposal_contract_digest")
            == FROZEN_PROPOSAL_CONTRACT_DIGEST,
            contract.get("route_quotas") == FROZEN_ROUTE_QUOTAS,
            contract.get("request") == FROZEN_REQUEST,
            contract.get("scan_protocol") == PRODUCER_SCAN_PROTOCOL,
            contract.get("performance_limits") == PRODUCER_PERFORMANCE_LIMITS,
            contract.get("prefix_policy")
            == "worker recomputes and exactly matches frozen stock and benchmark causal prefixes",
            contract.get("implementation_manifest")
            == _expected_producer_implementation_manifest(),
            contract.get("real_forward_outcomes_accessed") is False,
            type(contract.get("physical_rows")) is int
            and int(contract["physical_rows"]) > 0,
            manifest_path.is_file(),
            file_fingerprint(manifest_path)
            == contract.get("physical_manifest_sha256"),
            contract.get("contract_digest") == stable_hash({
                key: value for key, value in contract.items()
                if key != "contract_digest"
            }),
        )):
            failures.append("producer execution, code or physical binding differs")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        failures.append(f"malformed producer contract:{type(exc).__name__}:{exc}")
    return tuple(failures)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    expected_candidate_root = config.artifact_dir / "m04r11" / "candidate-pools-v1"
    expected_authority_root = config.artifact_dir / "m04r11" / "authorities-sealed-v4"
    expected_output_root = config.artifact_dir / "m04r11" / "candidate-comparison-v1"
    if not all((
        args.candidate_root.resolve() == expected_candidate_root.resolve(),
        args.authority_root.resolve() == expected_authority_root.resolve(),
        args.output_root.resolve() == expected_output_root.resolve(),
    )):
        raise ValueError("one-shot candidate, authority or comparison path differs")
    _refuse_failed_candidate_root(args.candidate_root)
    output_resolved = args.output_root.resolve()
    for protected in (
        args.registry.parent.resolve(), args.candidate_root.resolve(),
        args.authority_root.resolve(),
    ):
        if output_resolved == protected or output_resolved.is_relative_to(protected) or protected.is_relative_to(output_resolved):
            raise ValueError("comparison output root overlaps a sealed input root")
    comparison_seal_path = args.output_root / "SEALED.json"
    opened_marker_path = args.output_root / "RESULTS_OPENED.json"
    if comparison_seal_path.exists():
        sealed = json.loads(comparison_seal_path.read_text())
        result = json.loads((args.output_root / "candidate-comparison.json").read_text())
        if not all((
            sealed.get("seal_digest") == seal_digest(sealed),
            sealed.get("comparison_digest") == result.get("result_digest"),
            opened_marker_path.is_file(),
            sealed.get("results_opened_marker_digest")
            == json.loads(opened_marker_path.read_text()).get("result_digest"),
        )):
            raise ValueError("existing comparison seal is invalid")
        print("candidate comparison is already sealed; refusing to reopen one-shot evidence")
        return 0
    if opened_marker_path.exists():
        raise ValueError("candidate truth was previously opened without a final seal; fail closed")
    registry = json.loads(args.registry.read_text())
    candidate_matrix = json.loads((args.candidate_root / "candidate-matrix.json").read_text())
    candidate_seal = json.loads((args.candidate_root / "SEALED.json").read_text())
    producer_contract = json.loads((args.candidate_root / "candidate-contract.json").read_text())
    failures: list[str] = []
    producer_contract_failures = validate_preopen_producer_contract(
        producer_contract, registry,
    )
    if producer_contract_failures:
        raise ValueError(
            f"candidate producer contract differs before truth open:{producer_contract_failures}"
        )
    if not all((
        len(registry["cases_data"]) == 60,
        registry.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
        stable_hash([case["episode_id"] for case in registry["cases_data"]]) == FROZEN_CASE_ORDER_DIGEST,
        producer_contract.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
        producer_contract.get("generation_id") == FROZEN_GENERATION_ID,
        producer_contract.get("proposal_contract_digest") == FROZEN_PROPOSAL_CONTRACT_DIGEST,
        producer_contract.get("route_quotas") == FROZEN_ROUTE_QUOTAS,
        producer_contract.get("request") == FROZEN_REQUEST,
        producer_contract.get("ordered_query_ids_digest") == FROZEN_CASE_ORDER_DIGEST,
        registry.get("search_contract", {}).get("fast_route_quotas") == FROZEN_ROUTE_QUOTAS,
        registry.get("search_contract", {}).get("request") == FROZEN_REQUEST,
        candidate_matrix.get("result_digest") == candidate_matrix_digest(candidate_matrix),
        candidate_seal.get("seal_digest") == seal_digest(candidate_seal),
        candidate_seal.get("candidate_pools_sealed") is True,
        candidate_seal.get("candidate_matrix_digest") == candidate_matrix.get("result_digest"),
        candidate_seal.get("comparison_results_opened") is False,
    )):
        raise ValueError("candidate seal prerequisite differs")
    source = source_from_spec(config.datasets["nasdaq"])
    registry_failures = validate_m04r_validation_registry(source, args.registry.parent)
    if registry_failures:
        raise ValueError(f"sealed registry differs:{registry_failures}")
    candidate_payloads: dict[str, dict[str, Any]] = {}
    for registry_case in registry["cases_data"]:
        query_id = str(registry_case["episode_id"])
        candidate = json.loads((args.candidate_root / "cases" / f"{query_id}.json").read_text())
        episode = build_episode(
            source, InstrumentKey("nasdaq", str(registry_case["symbol"])),
            str(registry_case["cutoff"]), int(registry_case["lookback"]),
            str(registry_case["representation_version"]),
        )
        blind_failures = validate_truth_blind_candidate_case(
            candidate, expected_case=registry_case,
            registry_digest=FROZEN_REGISTRY_DIGEST,
            generation_id=FROZEN_GENERATION_ID,
            proposal_contract_digest=FROZEN_PROPOSAL_CONTRACT_DIGEST,
            producer_contract_digest=producer_contract["contract_digest"],
            route_quotas=producer_contract["route_quotas"],
            physical_rows=int(producer_contract["physical_rows"]),
            expected_query_start_ns=int(episode.bars.timestamp.iloc[0].value),
            expected_latest_eligible_ns=int(latest_eligible_cutoff(episode, 60).value),
            expected_representation_digest=representation_input_digest(represent(episode)),
        )
        if blind_failures:
            raise ValueError(f"truth-blind candidate prerequisite differs:{registry_case['case_id']}:{blind_failures}")
        candidate_payloads[query_id] = candidate
    ordered_candidates = [
        candidate_payloads[str(case["episode_id"])] for case in registry["cases_data"]
    ]
    preopen_producer_gates = {
        "all_60_candidate_pools_sealed": (
            len(ordered_candidates) == 60 and candidate_matrix.get("worker_failures") == []
        ),
        "all_safety_determinism_and_resource_gates": (
            len(ordered_candidates) == 60 and all(row["passed"] for row in ordered_candidates)
        ),
    }
    if not all((
        candidate_matrix.get("schema_version") == "candidate-recall-matrix-v2-producer",
        candidate_matrix.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
        candidate_matrix.get("producer_contract_digest") == producer_contract["contract_digest"],
        candidate_matrix.get("generation_id") == FROZEN_GENERATION_ID,
        candidate_matrix.get("proposal_contract_digest") == FROZEN_PROPOSAL_CONTRACT_DIGEST,
        candidate_matrix.get("completed_cases") == 60,
        candidate_matrix.get("case_result_digests") == [row["result_digest"] for row in ordered_candidates],
        candidate_matrix.get("maximum_cold_seconds") == max(row["cold_seconds"] for row in ordered_candidates),
        candidate_matrix.get("maximum_second_warm_seconds") == max(row["warm_second_seconds"] for row in ordered_candidates),
        candidate_matrix.get("maximum_peak_rss_mb") == max(row["peak_rss_mb"] for row in ordered_candidates),
        candidate_matrix.get("gates") == preopen_producer_gates,
        candidate_matrix.get("passed") is all(preopen_producer_gates.values()),
        candidate_seal.get("producer_contract_digest") == producer_contract["contract_digest"],
        candidate_seal.get("completed_cases") == 60,
        candidate_seal.get("candidate_pools_sealed") is True,
        candidate_seal.get("comparison_results_opened") is False,
        candidate_seal.get("production_promotion_authorized") is False,
    )):
        raise ValueError("aggregate truth-blind candidate prerequisite differs")
    marker_deterministic = {
        "schema_version": "candidate-recall-results-opened-v1",
        "registry_digest": FROZEN_REGISTRY_DIGEST,
        "candidate_matrix_digest": candidate_matrix["result_digest"],
        "candidate_seal_digest": candidate_seal["seal_digest"],
        "producer_contract_digest": producer_contract["contract_digest"],
        "status": "authority results about to be opened exactly once",
    }
    marker = {
        **marker_deterministic, "created_at": datetime.now(timezone.utc).isoformat(),
        "result_digest": stable_hash(marker_deterministic),
    }
    _atomic_json(opened_marker_path, marker)
    authority_matrix = json.loads((args.authority_root / "authority-matrix.json").read_text())
    authority_seal = json.loads((args.authority_root / "SEALED.json").read_text())
    if not all((
        authority_matrix.get("gate_passed") is True,
        authority_seal.get("authority_correctness_sealed") is True,
        authority_seal.get("seal_digest") == seal_digest(authority_seal),
        authority_seal.get("authority_matrix_digest") == authority_matrix.get("result_digest"),
        authority_seal.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
    )):
        raise ValueError("authority seal prerequisite differs after results-opened marker")
    cases: list[dict[str, Any]] = []
    authority_rows = {
        str(row["query_episode_id"]): row for row in authority_matrix["cases"]
    }
    candidate_case_digests: list[str] = []
    cold_seconds: list[float] = []
    warm_seconds: list[float] = []
    peak_rss: list[float] = []
    for registry_case in registry["cases_data"]:
        query_id = str(registry_case["episode_id"])
        candidate = candidate_payloads[query_id]
        authority = json.loads((args.authority_root / "cases" / f"{query_id}.json").read_text())
        candidate_ids = [str(row["episode_id"]) for row in candidate["candidates"]]
        truth_ids = [str(row["episode_id"]) for row in authority["matches"]]
        retained = [value for value in truth_ids if value in set(candidate_ids)]
        case_failures = validate_candidate_case(
            candidate, registry_case, authority, producer_contract,
        )
        authority_row = authority_rows.get(query_id) or {}
        if not all((
            authority_row.get("registry_case_id") == registry_case["case_id"],
            authority_row.get("authority_digest") == authority.get("result_digest"),
            authority_row.get("certificate_digest") == authority.get("certificate_digest"),
        )):
            case_failures.append("authority matrix/case binding differs")
        if candidate.get("passed") is not True and not any(
            value == "candidate safety/determinism/performance gate failed"
            for value in case_failures
        ):
            case_failures.append("candidate safety/determinism/performance gate failed")
        recall = len(retained) / 20.0
        if recall < .95:
            case_failures.append("candidate recall below 19/20")
        cases.append({
            "registry_case_id": registry_case["case_id"], "query_episode_id": query_id,
            "candidate_case_digest": candidate["result_digest"],
            "authority_case_digest": authority["result_digest"],
            "candidate_count": len(candidate_ids), "retained_count": len(retained),
            "recall_at_20": recall,
            "missing_authority_episode_ids": [value for value in truth_ids if value not in set(retained)],
            "failures": case_failures, "passed": not case_failures,
        })
        failures.extend(f"{registry_case['case_id']}:{value}" for value in case_failures)
        candidate_case_digests.append(candidate["result_digest"])
        cold_seconds.append(float(candidate["cold_seconds"]))
        warm_seconds.append(float(candidate["warm_second_seconds"]))
        peak_rss.append(float(candidate["peak_rss_mb"]))
    recalls = [row["recall_at_20"] for row in cases]
    expected_producer_gates = {
        "all_60_candidate_pools_sealed": len(cases) == 60 and candidate_matrix.get("worker_failures") == [],
        "all_safety_determinism_and_resource_gates": len(cases) == 60 and all(
            json.loads((args.candidate_root / "cases" / f"{row['query_episode_id']}.json").read_text())["passed"]
            for row in cases
        ),
    }
    if not all((
        candidate_matrix.get("schema_version") == "candidate-recall-matrix-v2-producer",
        candidate_matrix.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
        candidate_matrix.get("producer_contract_digest") == producer_contract["contract_digest"],
        candidate_matrix.get("generation_id") == FROZEN_GENERATION_ID,
        candidate_matrix.get("proposal_contract_digest") == FROZEN_PROPOSAL_CONTRACT_DIGEST,
        candidate_matrix.get("completed_cases") == 60,
        candidate_matrix.get("case_result_digests") == candidate_case_digests,
        candidate_matrix.get("maximum_cold_seconds") == max(cold_seconds),
        candidate_matrix.get("maximum_second_warm_seconds") == max(warm_seconds),
        candidate_matrix.get("maximum_peak_rss_mb") == max(peak_rss),
        candidate_matrix.get("gates") == expected_producer_gates,
        candidate_matrix.get("passed") is all(expected_producer_gates.values()),
        candidate_seal.get("producer_contract_digest") == producer_contract["contract_digest"],
        candidate_seal.get("completed_cases") == 60,
        candidate_seal.get("candidate_pools_sealed") is True,
        candidate_seal.get("comparison_results_opened") is False,
        candidate_seal.get("production_promotion_authorized") is False,
    )):
        failures.append("aggregate candidate matrix or seal differs")
    gates = {
        "all_60_comparisons_valid": len(cases) == 60 and not failures,
        "every_case_at_least_19_of_20": len(cases) == 60 and min(recalls, default=0) >= .95,
        "mean_recall_at_least_99_percent": len(cases) == 60 and sum(recalls) / 60 >= .99,
    }
    deterministic = {
        "schema_version": SCHEMA, "registry_digest": registry["registry_digest"],
        "candidate_matrix_digest": candidate_matrix["result_digest"],
        "candidate_seal_digest": candidate_seal["seal_digest"],
        "producer_contract_digest": producer_contract["contract_digest"],
        "results_opened_marker_digest": marker["result_digest"],
        "comparison_implementation_sha256": file_fingerprint(Path(__file__)),
        "authority_matrix_digest": authority_matrix["result_digest"],
        "authority_seal_digest": authority_seal["seal_digest"],
        "completed_cases": len(cases), "minimum_recall_at_20": min(recalls, default=None),
        "mean_recall_at_20": sum(recalls) / len(recalls) if recalls else None,
        "perfect_recall_cases": sum(value == 1.0 for value in recalls),
        "cases": cases, "failures": failures, "gates": gates,
        "passed": all(gates.values()), "candidate_results_opened": True,
        "production_promotion_authorized": False, "real_forward_outcomes_accessed": False,
    }
    payload = {**deterministic, "created_at": datetime.now(timezone.utc).isoformat(),
               "result_digest": stable_hash(deterministic)}
    _atomic_json(args.output_root / "candidate-comparison.json", payload)
    status = "PASS" if payload["passed"] else "FAIL"
    (args.output_root / "candidate-comparison.html").write_text(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>M04R-11 candidate comparison</title><style>body{font-family:system-ui;"
        "max-width:1200px;margin:2rem auto}pre{white-space:pre-wrap}</style></head><body>"
        f"<h1>{status}</h1><p>One-time comparison of sealed truth-blind candidate pools "
        f"with sealed exact authorities.</p><pre>{escape(json.dumps(payload, indent=2, sort_keys=True))}"
        "</pre></body></html>"
    )
    comparison_seal_deterministic = {
        "schema_version": "candidate-recall-comparison-seal-v1",
        "registry_digest": FROZEN_REGISTRY_DIGEST,
        "candidate_seal_digest": candidate_seal["seal_digest"],
        "authority_seal_digest": authority_seal["seal_digest"],
        "comparison_digest": payload["result_digest"],
        "results_opened_marker_digest": marker["result_digest"],
        "candidate_results_opened": True,
        "comparison_gate_passed": payload["passed"],
        "production_promotion_authorized": False,
    }
    comparison_seal = {
        **comparison_seal_deterministic,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seal_digest": stable_hash(comparison_seal_deterministic),
    }
    _atomic_json(comparison_seal_path, comparison_seal)
    print(json.dumps({key: value for key, value in payload.items() if key != "cases"}, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
