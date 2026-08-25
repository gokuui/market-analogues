"""Build and seal the M04R-11 truth-first 60-case authority matrix."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
import multiprocessing
from pathlib import Path
import resource
from time import perf_counter
import traceback
from typing import Any

import numba
import pandas as pd

from market_analogues.adapters import file_fingerprint, source_from_spec
from market_analogues.certified_packed_search import (
    CertifiedFrontierOverflow,
    certified_packed_search,
    certified_packed_search_contract,
)
from market_analogues.config import load_config
from market_analogues.episodes import build_episode
from market_analogues.m04r_certified_search_verification import _certificate_digest
from market_analogues.m04r_full_pack_verification import EVIDENCE_OMITTED as BUILD_OMITTED
from market_analogues.m04r_validation_registry import validate_m04r_validation_registry
from market_analogues.packed_bound_search import PackedBoundQuery, scan_packed_bound_proposals_many
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.representation import represent
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import InstrumentKey, SearchQuery, stable_hash


CONTRACT_SCHEMA = "m04r11-authority-build-contract-v2"
CASE_SCHEMA = "m04r11-certified-authority-case-v2"
MATRIX_SCHEMA = "m04r11-certified-authority-matrix-v2"
SEAL_SCHEMA = "m04r11-authority-seal-v2"
FRONTIER_OVERFLOW_MESSAGE = (
    "certified frontier exceeds configured maximum; no result emitted"
)
OVERFLOW_POLICY_SCHEMA = "m04r11-frontier-overflow-recovery-v1"
CASE_RESULT_OMITTED = {
    "created_at", "proposal_seconds", "amortized_proposal_seconds",
    "exact_seconds", "final_exact_seconds", "frontier_attempt_measurements",
    "peak_rss_mb",
    "result_digest", "checkpoint_integrity_digest",
}
MATRIX_RESULT_OMITTED = {
    "created_at", "elapsed_seconds", "p95_exact_seconds",
    "maximum_exact_seconds", "total_exact_seconds", "maximum_worker_rss_mb",
    "p95_search_seconds", "maximum_search_seconds", "total_search_seconds",
    "measurements", "performance_gates", "performance_gate_passed",
    "measurement_integrity_digest", "result_digest",
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _match(value: Any) -> dict[str, Any]:
    return {
        "episode_id": value.episode_key.id,
        "symbol": value.episode_key.instrument.source_symbol,
        "cutoff": value.episode_key.cutoff.isoformat(),
        "total_distance": float(value.total_distance),
        "component_distances": {
            key: float(item) for key, item in sorted(value.component_distances.items())
        },
        "alignment": [[int(left), int(right)] for left, right in value.alignment],
        "quality_tier": value.quality_tier,
    }


def case_result_digest(payload: dict[str, Any]) -> str:
    return stable_hash({
        key: value for key, value in payload.items() if key not in CASE_RESULT_OMITTED
    })


def checkpoint_integrity_digest(payload: dict[str, Any]) -> str:
    return stable_hash({
        key: value for key, value in payload.items()
        if key not in {"created_at", "checkpoint_integrity_digest"}
    })


def matrix_result_digest(payload: dict[str, Any]) -> str:
    return stable_hash({
        key: value for key, value in payload.items()
        if key not in MATRIX_RESULT_OMITTED
    })


def measurement_integrity_digest(payload: dict[str, Any]) -> str:
    return stable_hash({
        "measurements": payload["measurements"],
        "p95_exact_seconds": payload["p95_exact_seconds"],
        "maximum_exact_seconds": payload["maximum_exact_seconds"],
        "total_exact_seconds": payload["total_exact_seconds"],
        "p95_search_seconds": payload["p95_search_seconds"],
        "maximum_search_seconds": payload["maximum_search_seconds"],
        "total_search_seconds": payload["total_search_seconds"],
        "maximum_worker_rss_mb": payload["maximum_worker_rss_mb"],
        "performance_gates": payload["performance_gates"],
        "performance_gate_passed": payload["performance_gate_passed"],
    })


def _implementation_manifest() -> dict[str, Any]:
    """Bind authority evidence to all local Python implementation sources."""
    repository = Path(__file__).resolve().parents[2]
    paths = [Path(__file__).resolve(), *sorted((repository / "src").rglob("*.py"))]
    files = {
        str(path.relative_to(repository)): file_fingerprint(path)
        for path in paths
    }
    return {"files": files, "digest": stable_hash(files)}


def _overflow_policy(primary_limit: int) -> dict[str, Any]:
    return {
        "schema_version": OVERFLOW_POLICY_SCHEMA,
        "trigger": FRONTIER_OVERFLOW_MESSAGE,
        "first_recovery_frontier_rows": int(primary_limit) * 2,
        "growth_factor": 2,
        "terminal_frontier": "eligible-universe exhaustion",
        "scope": "authority construction only",
        "semantics": (
            "retry the unchanged exact certified search over monotonically larger "
            "bound-sorted frontiers; never emit an approximate result"
        ),
    }


def _frontier_execution_valid(
    payload: dict[str, Any], contract: dict[str, Any],
) -> bool:
    certificate = payload.get("certificate") or {}
    rounds = certificate.get("rounds") or []
    final_frontiers = [int(row.get("frontier_rows", -1)) for row in rounds]
    attempts = payload.get("frontier_attempts") or []
    if not final_frontiers or not attempts or min(final_frontiers) < 1:
        return False
    primary_limit = int(contract["controls"]["maximum_frontier_rows"])
    eligible = int(certificate.get("eligible_candidates", -1))
    expected_limits = [primary_limit]
    while expected_limits[-1] < eligible:
        expected_limits.append(min(eligible, expected_limits[-1] * 2))
    limits = [int(row.get("maximum_frontier_rows", -1)) for row in attempts]
    if limits != expected_limits[:len(limits)] or len(limits) > len(expected_limits):
        return False
    statuses = [row.get("status") for row in attempts]
    if statuses[-1] != "certified" or any(value != "overflow" for value in statuses[:-1]):
        return False
    if payload.get("frontier_overflow_recovery") is not (len(attempts) > 1):
        return False
    if payload.get("frontier_limit_rows") != limits[-1]:
        return False
    if max(final_frontiers) > limits[-1]:
        return False
    proposal_digests = [row.get("proposal_result_digest") for row in attempts]
    if not all(isinstance(value, str) and value for value in proposal_digests):
        return False
    for row, limit in zip(attempts[:-1], limits[:-1], strict=True):
        next_hex = row.get("next_lower_bound_hex")
        threshold_hex = row.get("stop_threshold_hex")
        if not all((
            int(row.get("exact_evaluated", -1)) == limit,
            isinstance(next_hex, str), isinstance(threshold_hex, str),
            float.fromhex(next_hex) <= float.fromhex(threshold_hex),
        )):
            return False
    final = attempts[-1]
    certificate_next = certificate.get("next_lower_bound")
    return all((
        int(final.get("exact_evaluated", -1))
        == int(certificate.get("exact_evaluated", -2)),
        final.get("stop_threshold_hex")
        == float(certificate.get("stop_threshold")).hex(),
        final.get("next_lower_bound_hex")
        == (float(certificate_next).hex() if certificate_next is not None else None),
    ))


def _case_gates(
    case: dict[str, Any], matches: list[dict[str, Any]], certificate: dict[str, Any],
) -> dict[str, bool]:
    query_start = pd.Timestamp(case["cutoff"]) - pd.Timedelta(days=10_000)
    # The exact same-symbol overlap condition can be checked from the frozen
    # lookback start stored by the runner; the broad fallback is never used in
    # a completed artifact.
    if case.get("query_start") is not None:
        query_start = pd.Timestamp(case["query_start"])
    ids = [str(row.get("episode_id")) for row in matches]
    ordering = [
        (float(row["total_distance"]), str(row["episode_id"])) for row in matches
    ]
    next_bound = certificate.get("next_lower_bound")
    stopped_early = bool(certificate.get("stopped_early"))
    stopping = (
        stopped_early and next_bound is not None
        and float(next_bound) > float(certificate.get("stop_threshold"))
    ) or (
        not stopped_early and int(certificate.get("safely_pruned", -1)) == 0
    )
    same_symbol_valid = all(
        str(row.get("symbol")) != str(case["symbol"])
        or pd.Timestamp(row.get("cutoff")) < query_start
        for row in matches
    )
    latest_eligible = case.get("latest_eligible_cutoff")
    temporal_valid = latest_eligible is not None and all(
        pd.Timestamp(row.get("cutoff")) <= pd.Timestamp(latest_eligible)
        for row in matches
    )
    gates = {
        "twenty_matches": len(matches) == 20,
        "unique_episode_ids": len(ids) == len(set(ids)),
        "stable_distance_id_order": ordering == sorted(ordering),
        "quality_tiers_allowed": all(row.get("quality_tier") in {"A", "B"} for row in matches),
        "per_instrument_cap": max(
            (sum(str(item.get("symbol")) == symbol for item in matches)
             for symbol in {str(item.get("symbol")) for item in matches}),
            default=0,
        ) <= 3,
        "same_symbol_overlap_excluded": same_symbol_valid,
        "candidate_cutoffs_temporally_eligible": temporal_valid,
        "certificate_query_equal": certificate.get("query_episode_id") == case["episode_id"],
        "candidate_accounting": (
            int(certificate.get("exact_evaluated", -1))
            + int(certificate.get("safely_pruned", -1))
            == int(certificate.get("eligible_candidates", -2))
        ),
        "strict_stop_or_exhaustion": stopping,
        "quantized_bound_safe": float(
            certificate.get("maximum_quantized_bound_excess", float("inf"))
        ) <= 1e-12,
        "certificate_digest_reconstructed": (
            certificate.get("result_digest")
            == _certificate_digest({"certificate": certificate, "matches": matches})
        ),
    }
    return gates


def _case_checkpoint_valid(
    payload: dict[str, Any], case: dict[str, Any], contract: dict[str, Any],
) -> bool:
    try:
        certificate = payload["certificate"]
        matches = payload["matches"]
        validation_case = {
            **case, "query_start": payload.get("query_start"),
            "latest_eligible_cutoff": payload.get("latest_eligible_cutoff"),
        }
        expected_gates = _case_gates(validation_case, matches, certificate)
        expected_gates["frontier_execution_policy"] = _frontier_execution_valid(
            payload, contract,
        )
        return all((
            payload.get("schema_version") == CASE_SCHEMA,
            payload.get("status") == "completed",
            payload.get("contract_digest") == contract["contract_digest"],
            payload.get("registry_digest") == contract["registry_digest"],
            payload.get("generation_id") == contract["generation_id"],
            payload.get("registry_case_id") == case["case_id"],
            payload.get("query_episode_id") == case["episode_id"],
            payload.get("query_stock_prefix") == case["stock_prefix"],
            payload.get("query_benchmark_prefix") == case["benchmark_prefix"],
            certificate.get("generation_id") == contract["generation_id"],
            certificate.get("contract_digest")
            == contract["certified_execution_contract"]["digest"],
            payload.get("gates") == expected_gates,
            payload.get("gate_passed") is True,
            payload.get("result_digest") == case_result_digest(payload),
            payload.get("checkpoint_integrity_digest") == checkpoint_integrity_digest(payload),
            payload.get("real_forward_outcomes_accessed") is False,
        ))
    except (KeyError, TypeError, ValueError):
        return False


def _certified_with_overflow_recovery(
    episode: Any, source: Any, request: SearchQuery, full_root: Path,
    generation_id: str, controls: dict[str, Any], packed_query: PackedBoundQuery,
    primary_proposal: Any,
) -> tuple[Any, list[dict[str, Any]], float, Any, list[dict[str, Any]]]:
    """Escalate only the operational frontier; exact semantics stay unchanged."""
    primary_limit = int(controls["maximum_frontier_rows"])
    frontier_limit = primary_limit
    proposal = primary_proposal
    recovery_proposal_seconds = 0.0
    attempts: list[dict[str, Any]] = []
    attempt_measurements: list[dict[str, Any]] = []
    common = {
        "store_dataset_id": "nasdaq",
        "seed_rows": int(controls["seed_rows"]),
        "block_rows": int(controls["block_rows"]),
        "workers": int(controls["exact_workers_per_process"]),
        "sparse_cutoff": 8,
        "verify_content": False,
        "requested_positions": True,
        "vector_lower_bounds": True,
        "deferred_alignments": True,
        "compact_scored": True,
    }
    while True:
        initial = int(controls["initial_frontier_rows"])
        attempt_started = perf_counter()
        try:
            result = certified_packed_search(
                episode, source, request, full_root / "store", generation_id,
                initial_frontier_rows=initial,
                maximum_frontier_rows=frontier_limit,
                precomputed_proposal=proposal,
                **common,
            )
        except CertifiedFrontierOverflow as exc:
            attempt_measurements.append({
                "maximum_frontier_rows": frontier_limit,
                "elapsed_seconds": perf_counter() - attempt_started,
                "status": "overflow",
            })
            attempts.append({
                "maximum_frontier_rows": frontier_limit,
                "proposal_result_digest": proposal.result_digest,
                "exact_evaluated": exc.exact_evaluated,
                "stop_threshold_hex": exc.stop_threshold.hex(),
                "next_lower_bound_hex": (
                    exc.next_lower_bound.hex()
                    if exc.next_lower_bound is not None else None
                ),
                "status": "overflow",
            })
            eligible = int(proposal.eligible_rows)
            if frontier_limit >= eligible:
                raise RuntimeError(
                    "certified search reported frontier overflow at eligible exhaustion"
                ) from exc
            frontier_limit = min(eligible, frontier_limit * 2)
            recovery = scan_packed_bound_proposals_many(
                full_root / "store", generation_id, [packed_query],
                route_quotas={"composite": frontier_limit + 1},
                block_rows=int(controls["block_rows"]), verify_content=False,
            )
            recovery_proposal_seconds += float(recovery.elapsed_seconds)
            proposal = recovery.reports[0]
            continue
        attempts.append({
            "maximum_frontier_rows": frontier_limit,
            "proposal_result_digest": proposal.result_digest,
            "exact_evaluated": result.certificate.exact_evaluated,
            "stop_threshold_hex": result.certificate.stop_threshold.hex(),
            "next_lower_bound_hex": (
                result.certificate.next_lower_bound.hex()
                if result.certificate.next_lower_bound is not None else None
            ),
            "status": "certified",
        })
        attempt_measurements.append({
            "maximum_frontier_rows": frontier_limit,
            "elapsed_seconds": perf_counter() - attempt_started,
            "status": "certified",
        })
        return result, attempts, recovery_proposal_seconds, proposal, attempt_measurements


def _worker(
    config_path: str, full_root: str, output_root: str,
    generation_id: str, contract: dict[str, Any],
    cases: tuple[dict[str, Any], ...], controls: dict[str, Any],
) -> dict[str, Any]:
    started = perf_counter()
    numba.set_num_threads(int(controls["numba_threads_per_process"]))
    config = load_config(config_path)
    source = source_from_spec(config.datasets["nasdaq"])
    episodes = []
    requests = []
    packed_queries = []
    expanded_cases = []
    for raw in cases:
        case = dict(raw)
        episode = build_episode(
            source, InstrumentKey("nasdaq", str(case["symbol"])),
            str(case["cutoff"]), int(case["lookback"]),
            str(case["representation_version"]),
        )
        case["query_start"] = pd.Timestamp(episode.bars.timestamp.iloc[0]).isoformat()
        request = SearchQuery(
            episode.key, ("nasdaq",), ("A", "B"), 20,
            False, True, 3, 60,
        )
        case["latest_eligible_cutoff"] = latest_eligible_cutoff(episode, 60).isoformat()
        episodes.append(episode)
        requests.append(request)
        expanded_cases.append(case)
        packed_queries.append(PackedBoundQuery(
            episode.key.id, episode.key.instrument.source_symbol,
            int(episode.bars.timestamp.iloc[0].value),
            int(latest_eligible_cutoff(episode, 60).value),
            represent(episode), request.quality_tiers,
        ))
    proposal = scan_packed_bound_proposals_many(
        Path(full_root) / "store", generation_id, packed_queries,
        route_quotas={"composite": int(controls["maximum_frontier_rows"]) + 1},
        block_rows=int(controls["block_rows"]), verify_content=False,
    )
    completed = []
    for case, episode, request, packed_query, query_proposal in zip(
        expanded_cases, episodes, requests, packed_queries, proposal.reports, strict=True,
    ):
        result, attempts, recovery_seconds, final_proposal, attempt_measurements = (
            _certified_with_overflow_recovery(
                episode, source, request, Path(full_root), generation_id, controls,
                packed_query, query_proposal,
            )
        )
        matches = [_match(value) for value in result.matches]
        certificate = json.loads(json.dumps(asdict(result.certificate)))
        final_exact_seconds = float(certificate.pop("elapsed_seconds"))
        exact_seconds = sum(
            float(row["elapsed_seconds"]) for row in attempt_measurements
        )
        payload: dict[str, Any] = {
            "schema_version": CASE_SCHEMA, "status": "completed",
            "contract_digest": contract["contract_digest"],
            "registry_digest": contract["registry_digest"],
            "generation_id": generation_id,
            "registry_case_id": case["case_id"],
            "query_episode_id": case["episode_id"],
            "query_symbol": case["symbol"], "query_cutoff": case["cutoff"],
            "query_start": case["query_start"],
            "latest_eligible_cutoff": case["latest_eligible_cutoff"],
            "query_stock_prefix": case["stock_prefix"],
            "query_benchmark_prefix": case["benchmark_prefix"],
            "primary_proposal_result_digest": query_proposal.result_digest,
            "proposal_result_digest": final_proposal.result_digest,
            "proposal_seconds": (
                float(proposal.elapsed_seconds) + recovery_seconds
            ),
            "amortized_proposal_seconds": (
                float(proposal.elapsed_seconds) / len(cases) + recovery_seconds
            ),
            "frontier_attempts": attempts,
            "frontier_overflow_recovery": len(attempts) > 1,
            "frontier_limit_rows": attempts[-1]["maximum_frontier_rows"],
            "frontier_attempt_measurements": attempt_measurements,
            "matches": matches, "certificate": certificate,
            "certificate_digest": certificate["result_digest"],
            "exact_seconds": exact_seconds,
            "final_exact_seconds": final_exact_seconds,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "real_forward_outcomes_accessed": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        gates = _case_gates(case, matches, certificate)
        gates["frontier_execution_policy"] = _frontier_execution_valid(payload, contract)
        payload["gates"] = gates
        payload["gate_passed"] = all(gates.values())
        payload["result_digest"] = case_result_digest(payload)
        payload["checkpoint_integrity_digest"] = checkpoint_integrity_digest(payload)
        path = Path(output_root) / "cases" / f"{case['episode_id']}.json"
        _atomic_json(path, payload)
        completed.append(str(case["episode_id"]))
        print(
            f"[authority:{case['case_id']}] "
            f"{'PASS' if payload['gate_passed'] else 'FAIL'} "
            f"exact={exact_seconds:.2f}s rows={certificate['exact_evaluated']} "
            f"recovery={len(attempts) > 1}",
            flush=True,
        )
        if not payload["gate_passed"]:
            raise RuntimeError(f"authority gate failed: {case['case_id']}")
    return {
        "query_episode_ids": completed,
        "proposal_seconds": float(proposal.elapsed_seconds),
        "elapsed_seconds": perf_counter() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }


def _groups(cases: list[dict[str, Any]], workers: int) -> list[tuple[dict[str, Any], ...]]:
    weighted = sorted(
        cases,
        key=lambda case: (
            -int(case.get("active_source_universe", 0)),
            sha256(str(case["case_id"]).encode()).hexdigest(),
        ),
    )
    groups: list[list[dict[str, Any]]] = [[] for _ in range(workers)]
    loads = [0] * workers
    for case in weighted:
        index = min(range(workers), key=lambda value: (loads[value], len(groups[value]), value))
        groups[index].append(case)
        loads[index] += int(case.get("active_source_universe", 0))
    return [tuple(group) for group in groups if group]


def _render(path: Path, payload: dict[str, Any]) -> None:
    status = "PASS — SEALED" if payload.get("gate_passed") else "INCOMPLETE/FAIL"
    css = "pass" if payload.get("gate_passed") else "fail"
    path.write_text(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>M04R-11 certified authority matrix</title><style>"
        "body{font-family:system-ui;max-width:1200px;margin:2rem auto}"
        ".pass{color:#075}.fail{color:#a20}pre{white-space:pre-wrap}"
        "</style></head><body><h1>M04R-11 truth-first authorities: "
        f"<span class=\"{css}\">{status}</span></h1><p>All exact certified "
        "authorities are built and sealed before any one-shot candidate-system "
        "comparison is authorized.</p><pre>"
        f"{escape(json.dumps(payload, indent=2, sort_keys=True))}</pre></body></html>"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=8)
    args = parser.parse_args()
    config = load_config(args.config)
    source = source_from_spec(config.datasets["nasdaq"])
    registry_root = args.registry.parent
    failures = validate_m04r_validation_registry(source, registry_root)
    if failures:
        raise ValueError(f"sealed registry differs: {failures}")
    registry = json.loads(args.registry.read_text())
    cases = list(registry["cases_data"])
    if len(cases) != 60 or args.processes < 1 or args.processes > len(cases):
        raise ValueError("authority runner requires 60 cases and 1..60 processes")
    if "candidate" in args.authority_root.name.lower():
        raise ValueError("authority root name must not be a candidate-result root")
    build_path = args.full_root / "packed-bound-full.json"
    build = json.loads(build_path.read_text())
    deterministic_build = {key: value for key, value in build.items() if key not in BUILD_OMITTED}
    if not all((
        build.get("result_digest") == stable_hash(deterministic_build),
        build.get("gate_passed") is True,
        build.get("shadow_generation") is True,
        build.get("real_forward_outcomes_accessed") is False,
        not (args.full_root / "store" / "active.json").exists(),
    )):
        raise ValueError("frozen shadow build evidence differs")
    generation_id = str(build["generation_id"])
    if generation_id != str(registry["search_contract"]["packed_generation_id"]):
        raise ValueError("full build generation differs from frozen registry")
    load_packed_generation(
        args.full_root / "store", generation_id,
        verify_content=True, validate_records=False,
    )
    controls = dict(registry["search_contract"]["controls"])
    if args.processes != int(controls["processes"]):
        raise ValueError("authority process count differs from frozen execution control")
    implementation = _implementation_manifest()
    execution_search_contract = certified_packed_search_contract(
        requested_positions=True, vector_lower_bounds=True,
        deferred_alignments=True, compact_scored=True,
    )
    contract: dict[str, Any] = {
        "schema_version": CONTRACT_SCHEMA,
        "registry_digest": registry["registry_digest"],
        "generation_id": generation_id,
        "full_build_evidence_digest": build["result_digest"],
        "search_contract": registry["search_contract"],
        "certified_execution_contract": execution_search_contract,
        "controls": controls,
        "frontier_overflow_policy": _overflow_policy(
            int(controls["maximum_frontier_rows"]),
        ),
        "execution_processes": args.processes,
        "expected_query_episode_ids": [case["episode_id"] for case in cases],
        "selection_order": [case["case_id"] for case in cases],
        "authority_root_policy": "write-isolated truth; no candidate result input",
        "runner_sha256": file_fingerprint(Path(__file__)),
        "implementation_manifest": implementation,
        "real_forward_outcomes_accessed": False,
    }
    contract["contract_digest"] = stable_hash(contract)
    args.authority_root.mkdir(parents=True, exist_ok=True)
    contract_path = args.authority_root / "authority-contract.json"
    if contract_path.exists():
        existing = json.loads(contract_path.read_text())
        if existing != contract:
            raise ValueError("existing authority build contract differs; refuse mixed resume")
    else:
        _atomic_json(contract_path, contract)
    seal_path = args.authority_root / "SEALED.json"
    if seal_path.exists():
        seal = json.loads(seal_path.read_text())
        matrix = json.loads((args.authority_root / "authority-matrix.json").read_text())
        seal_without_digest = {
            key: value for key, value in seal.items() if key != "seal_digest"
        }
        if not all((
            seal.get("schema_version") == SEAL_SCHEMA,
            seal.get("contract_digest") == contract["contract_digest"],
            seal.get("registry_digest") == contract["registry_digest"],
            seal.get("authority_matrix_digest") == matrix.get("result_digest"),
            seal.get("measurement_integrity_digest")
            == matrix.get("measurement_integrity_digest"),
            seal.get("seal_digest") == stable_hash(seal_without_digest),
            matrix.get("result_digest") == matrix_result_digest(matrix),
            matrix.get("measurement_integrity_digest")
            == measurement_integrity_digest(matrix),
            matrix.get("contract_digest") == contract["contract_digest"],
            matrix.get("registry_digest") == contract["registry_digest"],
            matrix.get("generation_id") == contract["generation_id"],
            matrix.get("completed_cases") == 60,
            matrix.get("gate_passed") is True,
            seal.get("authority_cases") == 60,
            seal.get("authority_correctness_sealed") is True,
            seal.get("seal_scope") == "exact authority correctness only",
            seal.get("performance_gate_passed")
            is matrix.get("performance_gate_passed"),
            seal.get("production_promotion_authorized") is False,
            seal.get("candidate_results_opened") is False,
            seal.get("real_forward_outcomes_accessed") is False,
        )):
            raise ValueError("authority seal is corrupt")
        print(json.dumps(seal, indent=2, sort_keys=True))
        return 0

    case_by_id = {str(case["episode_id"]): case for case in cases}
    valid: dict[str, dict[str, Any]] = {}
    for query_id, case in case_by_id.items():
        path = args.authority_root / "cases" / f"{query_id}.json"
        if path.exists():
            payload = json.loads(path.read_text())
            if _case_checkpoint_valid(payload, case, contract):
                valid[query_id] = payload
    remaining = [case for case in cases if str(case["episode_id"]) not in valid]
    print(
        f"[m04r11] valid={len(valid)}/60 remaining={len(remaining)} "
        f"processes={min(args.processes, len(remaining)) if remaining else 0}",
        flush=True,
    )
    started = perf_counter()
    worker_results = []
    worker_failures: list[dict[str, Any]] = []
    if remaining:
        groups = _groups(remaining, min(args.processes, len(remaining)))
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(groups), mp_context=context) as executor:
            futures = {
                executor.submit(
                    _worker, str(args.config), str(args.full_root),
                    str(args.authority_root), generation_id, contract, group, controls,
                ): group for group in groups
            }
            for future in as_completed(futures):
                group = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # preserve other groups and emit failed evidence
                    failure = {
                        "case_ids": [str(case["case_id"]) for case in group],
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": "".join(traceback.format_exception(exc)),
                    }
                    worker_failures.append(failure)
                    print(
                        f"[m04r11] group failed cases={failure['case_ids']} "
                        f"error={failure['error_type']}: {failure['message']}",
                        flush=True,
                    )
                    continue
                worker_results.append(result)
                print(
                    f"[m04r11] group complete cases={len(result['query_episode_ids'])} "
                    f"proposal={result['proposal_seconds']:.2f}s "
                    f"wall={result['elapsed_seconds']:.2f}s",
                    flush=True,
                )
    ordered = []
    invalid = []
    for case in cases:
        path = args.authority_root / "cases" / f"{case['episode_id']}.json"
        if not path.exists():
            invalid.append(f"missing:{case['case_id']}")
            continue
        payload = json.loads(path.read_text())
        if not _case_checkpoint_valid(payload, case, contract):
            invalid.append(f"invalid:{case['case_id']}")
            continue
        ordered.append(payload)
    seconds = [float(case["exact_seconds"]) for case in ordered]
    search_seconds = [
        float(case["proposal_seconds"]) + float(case["exact_seconds"])
        for case in ordered
    ]
    matrix: dict[str, Any] = {
        "schema_version": MATRIX_SCHEMA,
        "contract_digest": contract["contract_digest"],
        "registry_digest": registry["registry_digest"],
        "generation_id": generation_id,
        "cases": [{
            "registry_case_id": case["registry_case_id"],
            "query_episode_id": case["query_episode_id"],
            "authority_digest": case["result_digest"],
            "certificate_digest": case["certificate_digest"],
            "eligible_candidates": case["certificate"]["eligible_candidates"],
            "exact_evaluated": case["certificate"]["exact_evaluated"],
        } for case in ordered],
        "measurements": [{
            "registry_case_id": case["registry_case_id"],
            "proposal_seconds": case["proposal_seconds"],
            "exact_seconds": case["exact_seconds"],
            "search_seconds": float(case["proposal_seconds"])
            + float(case["exact_seconds"]),
            "peak_rss_mb": case["peak_rss_mb"],
        } for case in ordered],
        "completed_cases": len(ordered),
        "invalid_cases": invalid,
        "p95_exact_seconds": float(pd.Series(seconds).quantile(.95)) if seconds else None,
        "maximum_exact_seconds": max(seconds, default=None),
        "total_exact_seconds": sum(seconds),
        "p95_search_seconds": (
            float(pd.Series(search_seconds).quantile(.95)) if search_seconds else None
        ),
        "maximum_search_seconds": max(search_seconds, default=None),
        "total_search_seconds": sum(search_seconds),
        "maximum_worker_rss_mb": max(
            [float(case["peak_rss_mb"]) for case in ordered],
            default=0.0,
        ),
        "frontier_overflow_recovery_cases": sum(
            bool(case.get("frontier_overflow_recovery")) for case in ordered
        ),
        "worker_failures": worker_failures,
        "elapsed_seconds": perf_counter() - started,
        "gates": {
            "all_60_authorities_complete": len(ordered) == 60 and not invalid,
            "all_case_certificates_pass": all(case["gate_passed"] for case in ordered),
            "no_worker_failures": not worker_failures,
            "registry_order_exact": [case["query_episode_id"] for case in ordered]
            == [case["episode_id"] for case in cases],
            "real_forward_outcomes_excluded": True,
        },
        "candidate_results_opened": False,
        "real_forward_outcomes_accessed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    matrix["gate_passed"] = all(matrix["gates"].values())
    matrix["performance_gates"] = {
        "certified_p95_search_at_most_300_seconds": (
            matrix["p95_search_seconds"] is not None
            and float(matrix["p95_search_seconds"]) <= 300
        ),
        "certified_maximum_search_at_most_600_seconds": (
            matrix["maximum_search_seconds"] is not None
            and float(matrix["maximum_search_seconds"]) <= 600
        ),
        "certified_peak_rss_at_most_1536_mb": (
            float(matrix["maximum_worker_rss_mb"]) <= 1_536
        ),
    }
    matrix["performance_gate_passed"] = all(matrix["performance_gates"].values())
    matrix["measurement_integrity_digest"] = measurement_integrity_digest(matrix)
    matrix["result_digest"] = matrix_result_digest(matrix)
    _atomic_json(args.authority_root / "authority-matrix.json", matrix)
    _render(args.authority_root / "authority-matrix.html", matrix)
    if matrix["gate_passed"]:
        seal = {
            "schema_version": SEAL_SCHEMA,
            "contract_digest": contract["contract_digest"],
            "registry_digest": registry["registry_digest"],
            "authority_matrix_digest": matrix["result_digest"],
            "measurement_integrity_digest": matrix["measurement_integrity_digest"],
            "authority_cases": 60,
            "seal_scope": "exact authority correctness only",
            "authority_correctness_sealed": True,
            "performance_gate_passed": matrix["performance_gate_passed"],
            "production_promotion_authorized": False,
            "candidate_results_opened": False,
            "real_forward_outcomes_accessed": False,
        }
        seal["seal_digest"] = stable_hash(seal)
        _atomic_json(seal_path, seal)
    print(json.dumps(matrix, indent=2, sort_keys=True))
    return 0 if matrix["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
