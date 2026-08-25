"""Run the complete M04R 24-query shared-proposal certified batch gate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import resource
from typing import Any

import pandas as pd

from certified_batch_scalar_controls import (
    CASE_OMITTED as SCALAR_CASE_OMITTED,
    MATRIX_OMITTED as SCALAR_MATRIX_OMITTED,
    _case_valid as scalar_case_valid,
    _match,
)
from market_analogues.adapters import source_from_spec
from market_analogues.certified_packed_search import certified_packed_search
from market_analogues.config import load_config
from market_analogues.episodes import build_episode
from market_analogues.m04r_batch_registry import validate_m04r_batch_registry
from market_analogues.m04r_certified_search_verification import _certificate_digest
from market_analogues.m04r_full_pack_verification import (
    EVIDENCE_OMITTED as FULL_BUILD_OMITTED,
)
from market_analogues.packed_bound_search import (
    PackedBoundQuery, scan_packed_bound_proposals_many,
)
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.representation import represent
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import InstrumentKey, SearchQuery, stable_hash


CASE_SCHEMA = "m04r-certified-batch-case-v1"
EVIDENCE_SCHEMA = "m04r-certified-batch-gate-v1"
CASE_OMITTED = {"created_at", "exact_seconds", "peak_rss_mb", "result_digest"}
EVIDENCE_OMITTED = {
    "created_at", "started_at", "shared_scan_seconds", "exact_total_seconds",
    "total_seconds", "peak_rss_mb", "result_digest",
}


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _render(path: Path, payload: dict[str, Any]) -> None:
    status = "PASS" if payload["gate_passed"] else "INCOMPLETE/FAIL"
    css = "pass" if payload["gate_passed"] else "fail"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>M04R complete certified batch gate</title><style>body{{font-family:system-ui;max-width:1200px;margin:2rem auto}}pre{{white-space:pre-wrap}}.pass{{color:#075}}.fail{{color:#a20}}</style></head><body><h1>M04R complete certified batch: <span class="{css}">{status}</span></h1><p>One shared maximum-frontier proposal scan plus isolated certified exact completion for the frozen outcome-blind 24-query registry. Every match and certificate is compared to its optimized scalar control.</p><pre>{escape(json.dumps(payload, indent=2, sort_keys=True))}</pre></body></html>""")
    temporary.replace(path)


def _canonical_certificate(result: Any) -> tuple[dict[str, Any], float]:
    payload = asdict(result.certificate)
    seconds = float(payload.pop("elapsed_seconds"))
    return json.loads(json.dumps(payload)), seconds


def _valid_batch_checkpoint(
    payload: dict[str, Any], scalar: dict[str, Any], registry_digest: str,
    generation_id: str, shared_controls: dict[str, Any],
) -> bool:
    deterministic = {
        key: value for key, value in payload.items() if key not in CASE_OMITTED
    }
    try:
        gates = {
            "matches_equal_scalar": payload["matches"] == scalar["matches"],
            "certificate_equal_scalar": payload["certificate"] == scalar["certificate"],
            "certificate_digest_equal_scalar": (
                payload["certificate_digest"] == scalar["certificate_digest"]
            ),
            "certificate_reconstructed": (
                payload["certificate_digest"] == _certificate_digest(payload)
            ),
            "rss_within_1536_mib": float(payload["peak_rss_mb"]) <= 1_536,
        }
        return all((
            payload.get("schema_version") == CASE_SCHEMA,
            payload.get("registry_digest") == registry_digest,
            payload.get("generation_id") == generation_id,
            payload.get("query_episode_id") == scalar["query_episode_id"],
            payload.get("scalar_result_digest") == scalar["result_digest"],
            payload.get("controls") == shared_controls,
            payload.get("result_digest") == stable_hash(deterministic),
            payload.get("gates") == gates,
            payload.get("gate_passed") == all(gates.values()),
            payload.get("real_forward_outcomes_accessed") is False,
        ))
    except (KeyError, TypeError, ValueError):
        return False


def _aggregate(
    cases: list[dict[str, Any]], expected_ids: list[str], failures: list[dict[str, str]],
    *, registry_digest: str, generation_id: str, scalar_digest: str,
    scalar_total: float, shared_scan_seconds: float, shared_report_digest: str,
    shared_controls: dict[str, Any], peak_rss_mb: float, started_at: str,
) -> dict[str, Any]:
    completed_ids = [str(case["query_episode_id"]) for case in cases]
    exact_total = sum(float(case["exact_seconds"]) for case in cases)
    total = shared_scan_seconds + exact_total
    target = .9 * scalar_total
    gates = {
        "all_24_completed": len(cases) == 24 and completed_ids == expected_ids and not failures,
        "all_case_gates_passed": len(cases) == 24 and all(case["gate_passed"] for case in cases),
        "at_least_10pct_faster": len(cases) == 24 and total <= target,
        "rss_within_1536_mib": peak_rss_mb <= 1_536,
        "real_forward_outcomes_excluded": all(
            case["real_forward_outcomes_accessed"] is False for case in cases
        ),
    }
    payload = {
        "schema_version": EVIDENCE_SCHEMA,
        "registry_digest": registry_digest,
        "generation_id": generation_id,
        "scalar_control_digest": scalar_digest,
        "scalar_total_seconds": scalar_total,
        "required_maximum_total_seconds": target,
        "shared_controls": shared_controls,
        "shared_proposal_report_digest": shared_report_digest,
        "expected_query_episode_ids": expected_ids,
        "completed_query_episode_ids": completed_ids,
        "completed_cases": len(cases),
        "failed_cases": failures,
        "cases": cases,
        "shared_scan_seconds": shared_scan_seconds,
        "exact_total_seconds": exact_total,
        "total_seconds": total,
        "speedup": scalar_total / total if total else None,
        "elapsed_reduction_fraction": 1.0 - total / scalar_total if scalar_total else None,
        "peak_rss_mb": peak_rss_mb,
        "gates": gates,
        "gate_passed": all(gates.values()),
        "real_forward_outcomes_accessed": False,
        "started_at": started_at,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    deterministic = {
        key: value for key, value in payload.items() if key not in EVIDENCE_OMITTED
    }
    deterministic["cases"] = [{
        key: value for key, value in case.items() if key not in CASE_OMITTED
    } for case in cases]
    payload["result_digest"] = stable_hash(deterministic)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--scalar-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--block-rows", type=int, default=4_096)
    parser.add_argument("--initial-frontier-rows", type=int, default=16_384)
    parser.add_argument("--maximum-frontier-rows", type=int, default=32_768)
    args = parser.parse_args()
    if not all((
        args.workers > 0, args.block_rows > 0,
        args.maximum_frontier_rows >= args.initial_frontier_rows >= 512,
    )):
        raise ValueError("batch controls are invalid")
    config = load_config(args.config)
    source = source_from_spec(config.datasets["nasdaq"])
    quality = pd.read_parquet(config.artifact_dir / "quality" / "nasdaq.parquet")
    registry_failures = validate_m04r_batch_registry(
        source, args.registry, quality, config.artifact_dir / "oracles" / "nasdaq",
    )
    if registry_failures:
        raise ValueError(f"batch registry invalid: {registry_failures}")
    registry = json.loads(args.registry.read_text())
    registry_digest = str(registry["registry_digest"])
    registry_cases = list(registry["cases_data"])
    expected_ids = [str(case["episode_id"]) for case in registry_cases]

    build = json.loads((args.full_root / "packed-bound-full.json").read_text())
    build_content = {key: value for key, value in build.items() if key not in FULL_BUILD_OMITTED}
    if not all((
        build.get("result_digest") == stable_hash(build_content),
        build.get("gate_passed") is True,
        build.get("shadow_generation") is True,
        build.get("real_forward_outcomes_accessed") is False,
        not (args.full_root / "store" / "active.json").exists(),
    )):
        raise ValueError("full packed-build evidence differs")
    generation_id = str(build["generation_id"])
    load_packed_generation(
        args.full_root / "store", generation_id,
        verify_content=True, validate_records=False,
    )

    scalar_matrix = json.loads(
        (args.scalar_root / "certified-batch-scalar-controls.json").read_text()
    )
    scalar_deterministic = {
        key: value for key, value in scalar_matrix.items()
        if key not in SCALAR_MATRIX_OMITTED
    }
    scalar_deterministic["cases"] = [{
        key: value for key, value in case.items()
        if key not in SCALAR_CASE_OMITTED
    } for case in scalar_matrix.get("cases", [])]
    if not all((
        scalar_matrix.get("gate_passed") is True,
        scalar_matrix.get("registry_digest") == registry_digest,
        scalar_matrix.get("generation_id") == generation_id,
        scalar_matrix.get("completed_query_episode_ids") == expected_ids,
        scalar_matrix.get("result_digest") == stable_hash(scalar_deterministic),
    )):
        raise ValueError("scalar control matrix differs")
    scalar_cases = {
        str(case["query_episode_id"]): case for case in scalar_matrix["cases"]
    }
    authority_dir = config.artifact_dir / "gate12" / "authorities" / "nasdaq" / "cases"
    authorities = {}
    for path in authority_dir.glob("*.json"):
        authority = json.loads(path.read_text())
        content = dict(authority)
        claimed = content.pop("authority_digest", None)
        if claimed != stable_hash(content):
            raise ValueError(f"authority integrity differs: {path.name}")
        authorities[str(authority["query_episode_id"])] = authority
    scalar_controls = dict(scalar_matrix["controls"])
    for case in registry_cases:
        query_id = str(case["episode_id"])
        scalar = scalar_cases.get(query_id)
        if scalar is None or not scalar_case_valid(
            scalar, case, build, registry_digest, scalar_controls,
            authorities.get(query_id),
        ):
            raise ValueError(f"scalar checkpoint differs: {query_id}")

    episodes = []
    requests = []
    packed_queries = []
    for case in registry_cases:
        episode = build_episode(
            source, InstrumentKey("nasdaq", str(case["symbol"])),
            str(case["cutoff"]), int(case["lookback"]),
            str(case["representation_version"]),
        )
        request = SearchQuery(
            episode.key, ("nasdaq",), ("A", "B"), 20,
            False, True, 3, 60,
        )
        episodes.append(episode)
        requests.append(request)
        packed_queries.append(PackedBoundQuery(
            episode.key.id, episode.key.instrument.source_symbol,
            int(episode.bars.timestamp.iloc[0].value),
            int(latest_eligible_cutoff(episode, 60).value),
            represent(episode), request.quality_tiers,
        ))

    shared_controls = {
        "block_rows": args.block_rows,
        "workers": args.workers,
        "initial_frontier_rows": args.initial_frontier_rows,
        "maximum_frontier_rows": args.maximum_frontier_rows,
        "maximum_proposal_rows": args.maximum_frontier_rows + 1,
        "requested_positions": True,
        "vector_lower_bounds": True,
        "deferred_alignments": True,
        "sorted_joined_iqr_merge": True,
    }
    started_at = datetime.now(timezone.utc).isoformat()
    print("[shared-scan] start 24 queries", flush=True)
    shared = scan_packed_bound_proposals_many(
        args.full_root / "store", generation_id, packed_queries,
        route_quotas={"composite": args.maximum_frontier_rows + 1},
        block_rows=args.block_rows, verify_content=False,
    )
    print(
        f"[shared-scan] PASS {shared.elapsed_seconds:.2f}s "
        f"rss={shared.peak_rss_mb:.2f} MiB", flush=True,
    )
    if tuple(expected_ids) != shared.query_episode_ids:
        raise ValueError("shared report query order differs")

    cases_dir = args.output_root / "cases"
    evidence_path = args.output_root / "certified-batch-gate.json"
    completed: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    for case in registry_cases:
        query_id = str(case["episode_id"])
        checkpoint = cases_dir / f"{query_id}.json"
        if not checkpoint.exists():
            continue
        try:
            payload = json.loads(checkpoint.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if _valid_batch_checkpoint(
            payload, scalar_cases[query_id], registry_digest,
            generation_id, shared_controls,
        ):
            completed[query_id] = payload

    for case, episode, request, proposal in zip(
        registry_cases, episodes, requests, shared.reports,
    ):
        query_id = str(case["episode_id"])
        if query_id in completed:
            print(f"[{query_id}] exact resume verified", flush=True)
            continue
        print(f"[{query_id}] exact start {case['symbol']} {case['cutoff_role']}", flush=True)
        try:
            result = certified_packed_search(
                episode, source, request, args.full_root / "store", generation_id,
                store_dataset_id="nasdaq",
                initial_frontier_rows=args.initial_frontier_rows,
                maximum_frontier_rows=args.maximum_frontier_rows,
                seed_rows=512, block_rows=args.block_rows, workers=args.workers,
                sparse_cutoff=8, verify_content=False,
                requested_positions=True, vector_lower_bounds=True,
                deferred_alignments=True, precomputed_proposal=proposal,
            )
            matches = [_match(value) for value in result.matches]
            certificate, exact_seconds = _canonical_certificate(result)
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            scalar = scalar_cases[query_id]
            gates = {
                "matches_equal_scalar": matches == scalar["matches"],
                "certificate_equal_scalar": certificate == scalar["certificate"],
                "certificate_digest_equal_scalar": (
                    result.certificate.result_digest == scalar["certificate_digest"]
                ),
                "certificate_reconstructed": (
                    result.certificate.result_digest == _certificate_digest({
                        "certificate": certificate, "matches": matches,
                    })
                ),
                "rss_within_1536_mib": peak <= 1_536,
            }
            payload = {
                "schema_version": CASE_SCHEMA,
                "registry_digest": registry_digest,
                "generation_id": generation_id,
                "query_episode_id": query_id,
                "registry_case_id": case["case_id"],
                "scalar_result_digest": scalar["result_digest"],
                "controls": shared_controls,
                "proposal_result_digest": proposal.result_digest,
                "matches": matches,
                "certificate": certificate,
                "certificate_digest": result.certificate.result_digest,
                "exact_seconds": exact_seconds,
                "peak_rss_mb": peak,
                "gates": gates,
                "gate_passed": all(gates.values()),
                "real_forward_outcomes_accessed": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            payload["result_digest"] = stable_hash({
                key: value for key, value in payload.items() if key not in CASE_OMITTED
            })
            _write(checkpoint, payload)
            completed[query_id] = payload
            print(
                f"[{query_id}] {'PASS' if payload['gate_passed'] else 'FAIL'} "
                f"{exact_seconds:.2f}s exact={certificate['exact_evaluated']}",
                flush=True,
            )
        except Exception as exc:
            failure = {
                "query_episode_id": query_id,
                "type": type(exc).__name__, "message": str(exc),
            }
            failures.append(failure)
            print(f"[{query_id}] ERROR {failure['type']}:{failure['message']}", flush=True)
        ordered = [completed[value] for value in expected_ids if value in completed]
        evidence = _aggregate(
            ordered, expected_ids, failures, registry_digest=registry_digest,
            generation_id=generation_id, scalar_digest=scalar_matrix["result_digest"],
            scalar_total=float(scalar_matrix["total_seconds"]),
            shared_scan_seconds=shared.elapsed_seconds,
            shared_report_digest=shared.result_digest,
            shared_controls=shared_controls,
            peak_rss_mb=max(
                shared.peak_rss_mb,
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            ),
            started_at=started_at,
        )
        _write(evidence_path, evidence)
        _render(evidence_path.with_suffix(".html"), evidence)

    ordered = [completed[value] for value in expected_ids if value in completed]
    evidence = _aggregate(
        ordered, expected_ids, failures, registry_digest=registry_digest,
        generation_id=generation_id, scalar_digest=scalar_matrix["result_digest"],
        scalar_total=float(scalar_matrix["total_seconds"]),
        shared_scan_seconds=shared.elapsed_seconds,
        shared_report_digest=shared.result_digest,
        shared_controls=shared_controls,
        peak_rss_mb=max(
            shared.peak_rss_mb,
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        ),
        started_at=started_at,
    )
    _write(evidence_path, evidence)
    _render(evidence_path.with_suffix(".html"), evidence)
    print(json.dumps({
        "gate_passed": evidence["gate_passed"],
        "completed_cases": evidence["completed_cases"],
        "failed_cases": evidence["failed_cases"],
        "shared_scan_seconds": evidence["shared_scan_seconds"],
        "exact_total_seconds": evidence["exact_total_seconds"],
        "total_seconds": evidence["total_seconds"],
        "scalar_total_seconds": evidence["scalar_total_seconds"],
        "speedup": evidence["speedup"],
        "peak_rss_mb": evidence["peak_rss_mb"],
        "result_digest": evidence["result_digest"],
    }, indent=2), flush=True)
    return 0 if evidence["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
