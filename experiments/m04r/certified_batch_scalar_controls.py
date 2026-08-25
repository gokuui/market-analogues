"""Build resumable scalar certified controls for the frozen M04R 24-query registry."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import resource
from typing import Any

import numpy as np
import pandas as pd

from market_analogues.adapters import source_from_spec
from market_analogues.certified_packed_search import (
    certified_packed_search, certified_packed_search_contract,
)
from market_analogues.config import load_config
from market_analogues.episodes import build_episode
from market_analogues.m04r_batch_registry import validate_m04r_batch_registry
from market_analogues.m04r_certified_search_verification import _certificate_digest
from market_analogues.m04r_full_pack_verification import (
    EVIDENCE_OMITTED as FULL_BUILD_OMITTED,
)
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.types import InstrumentKey, SearchQuery, stable_hash


CASE_SCHEMA = "m04r-batch-scalar-control-case-v1"
MATRIX_SCHEMA = "m04r-batch-scalar-controls-v1"
CASE_OMITTED = {"created_at", "seconds", "peak_rss_mb", "result_digest"}
MATRIX_OMITTED = {
    "created_at", "started_at", "p95_seconds", "maximum_seconds",
    "total_seconds", "peak_rss_mb", "result_digest",
}


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _match(match: Any) -> dict[str, Any]:
    return {
        "episode_id": match.episode_key.id,
        "symbol": match.episode_key.instrument.source_symbol,
        "cutoff": match.episode_key.cutoff.isoformat(),
        "total_distance": match.total_distance,
        "component_distances": dict(sorted(match.component_distances.items())),
        "alignment": [[int(left), int(right)] for left, right in match.alignment],
        "quality_tier": match.quality_tier,
    }


def _authority_equal(matches: list[dict[str, Any]], authority: dict[str, Any] | None) -> bool:
    if authority is None:
        return True
    expected = authority.get("matches") or []
    if len(matches) != len(expected) or not matches:
        return False
    if [row["episode_id"] for row in matches] != [row["episode_id"] for row in expected]:
        return False
    if [row["alignment"] for row in matches] != [row["alignment"] for row in expected]:
        return False
    for left, right in zip(matches, expected):
        if abs(float(left["total_distance"]) - float(right["total_distance"])) > 1e-7:
            return False
        if set(left["component_distances"]) != set(right["component_distances"]):
            return False
        if any(
            abs(float(left["component_distances"][name]) - float(right["component_distances"][name])) > 1e-6
            for name in left["component_distances"]
        ):
            return False
    return True


def _case_valid(
    payload: dict[str, Any], case: dict[str, Any], build: dict[str, Any],
    registry_digest: str, controls: dict[str, Any],
    authority: dict[str, Any] | None,
) -> bool:
    certificate = payload.get("certificate") or {}
    deterministic = {
        key: value for key, value in payload.items() if key not in CASE_OMITTED
    }
    try:
        gates = {
            "authority_equal_when_available": _authority_equal(
                payload.get("matches") or [], authority,
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
                float(certificate["maximum_quantized_bound_excess"]) <= 1e-12
            ),
            "runtime_within_600_seconds": float(payload["seconds"]) <= 600,
            "rss_within_1536_mib": float(payload["peak_rss_mb"]) <= 1_536,
        }
        return all((
            payload.get("schema_version") == CASE_SCHEMA,
            payload.get("registry_digest") == registry_digest,
            payload.get("generation_id") == build["generation_id"],
            payload.get("full_build_evidence_digest") == build["result_digest"],
            payload.get("query_episode_id") == case["episode_id"],
            payload.get("registry_case_id") == case["case_id"],
            payload.get("controls") == controls,
            payload.get("result_digest") == stable_hash(deterministic),
            certificate.get("result_digest") == _certificate_digest(payload),
            payload.get("certificate_digest") == _certificate_digest(payload),
            payload.get("gates") == gates,
            payload.get("gate_passed") == all(gates.values()),
            payload.get("real_forward_outcomes_accessed") is False,
        ))
    except (KeyError, TypeError, ValueError):
        return False


def _render(path: Path, payload: dict[str, Any]) -> None:
    status = "PASS" if payload["gate_passed"] else "INCOMPLETE/FAIL"
    css = "pass" if payload["gate_passed"] else "fail"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>M04R 24-query scalar controls</title><style>body{{font-family:system-ui;max-width:1200px;margin:2rem auto}}pre{{white-space:pre-wrap}}.pass{{color:#075}}.fail{{color:#a20}}</style></head><body><h1>M04R scalar controls: <span class="{css}">{status}</span></h1><p>Certified scalar baseline for the outcome-blind 24-query registry. No forward outcome or setup label is accessed.</p><pre>{escape(json.dumps(payload, indent=2, sort_keys=True))}</pre></body></html>""")
    temporary.replace(path)


def _aggregate(
    cases: list[dict[str, Any]], expected_ids: list[str], failures: list[dict[str, str]],
    *, registry_digest: str, build: dict[str, Any], controls: dict[str, Any],
    started_at: str,
) -> dict[str, Any]:
    seconds = [float(case["seconds"]) for case in cases]
    completed_ids = [str(case["query_episode_id"]) for case in cases]
    p95 = float(np.percentile(seconds, 95)) if seconds else None
    maximum = max(seconds) if seconds else None
    peak = max((float(case["peak_rss_mb"]) for case in cases), default=0.0)
    gates = {
        "all_24_completed": len(cases) == 24 and completed_ids == expected_ids and not failures,
        "all_case_gates_passed": len(cases) == 24 and all(case["gate_passed"] for case in cases),
        "all_certificate_inputs_unique": len({
            (case["query_episode_id"], case["certificate"]["input_digest"])
            for case in cases
        }) == len(cases),
        "p95_within_300_seconds": p95 is not None and p95 <= 300,
        "maximum_within_600_seconds": maximum is not None and maximum <= 600,
        "rss_within_1536_mib": peak <= 1_536,
    }
    payload = {
        "schema_version": MATRIX_SCHEMA,
        "registry_digest": registry_digest,
        "generation_id": build["generation_id"],
        "full_build_evidence_digest": build["result_digest"],
        "controls": controls,
        "expected_query_episode_ids": expected_ids,
        "completed_query_episode_ids": completed_ids,
        "completed_cases": len(cases),
        "failed_cases": failures,
        "cases": cases,
        "p95_seconds": p95,
        "maximum_seconds": maximum,
        "total_seconds": sum(seconds),
        "peak_rss_mb": peak,
        "gates": gates,
        "gate_passed": all(gates.values()),
        "real_forward_outcomes_accessed": False,
        "started_at": started_at,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    deterministic = {
        key: value for key, value in payload.items() if key not in MATRIX_OMITTED
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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--query-ids", nargs="+")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--block-rows", type=int, default=4_096)
    parser.add_argument("--initial-frontier-rows", type=int, default=16_384)
    args = parser.parse_args()
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
    load_packed_generation(
        args.full_root / "store", str(build["generation_id"]),
        verify_content=True, validate_records=False,
    )
    authority_dir = config.artifact_dir / "gate12" / "authorities" / "nasdaq" / "cases"
    authorities: dict[str, dict[str, Any]] = {}
    for path in authority_dir.glob("*.json"):
        authority = json.loads(path.read_text())
        content = dict(authority)
        claimed = content.pop("authority_digest", None)
        if claimed != stable_hash(content):
            raise ValueError(f"authority integrity differs: {path.name}")
        authorities[str(authority["query_episode_id"])] = authority

    expected_ids = [str(case["episode_id"]) for case in registry_cases]
    selected_ids = set(args.query_ids or expected_ids)
    unknown = selected_ids.difference(expected_ids)
    if unknown:
        raise ValueError(f"unknown query IDs: {sorted(unknown)}")
    controls = {
        "block_rows": args.block_rows,
        "workers": args.workers,
        "initial_frontier_rows": args.initial_frontier_rows,
        "requested_positions": True,
        "hybrid_requested_positions": False,
        "vector_lower_bounds": True,
        "deferred_alignments": True,
        "sorted_joined_iqr_merge": True,
    }
    contract = certified_packed_search_contract(
        requested_positions=True, vector_lower_bounds=True,
        deferred_alignments=True,
    )
    cases_dir = args.output_root / "cases"
    matrix_path = args.output_root / "certified-batch-scalar-controls.json"
    started_at = datetime.now(timezone.utc).isoformat()
    completed: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    for case in registry_cases:
        query_id = str(case["episode_id"])
        checkpoint = cases_dir / f"{query_id}.json"
        if checkpoint.exists():
            try:
                payload = json.loads(checkpoint.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if _case_valid(
                payload, case, build, registry_digest, controls,
                authorities.get(query_id),
            ):
                completed[query_id] = payload

    for case in registry_cases:
        query_id = str(case["episode_id"])
        if query_id not in selected_ids:
            continue
        checkpoint = cases_dir / f"{query_id}.json"
        if query_id in completed:
            print(f"[{query_id}] resume verified", flush=True)
            continue
        print(f"[{query_id}] start {case['symbol']} {case['cutoff_role']}", flush=True)
        try:
            query = build_episode(
                source, InstrumentKey("nasdaq", str(case["symbol"])),
                str(case["cutoff"]), int(case["lookback"]),
                str(case["representation_version"]),
            )
            request = SearchQuery(
                query.key, ("nasdaq",), ("A", "B"), 20,
                False, True, 3, 60,
            )
            result = certified_packed_search(
                query, source, request, args.full_root / "store",
                str(build["generation_id"]), store_dataset_id="nasdaq",
                initial_frontier_rows=args.initial_frontier_rows,
                maximum_frontier_rows=32_768, seed_rows=512,
                block_rows=args.block_rows, workers=args.workers,
                sparse_cutoff=8, verify_content=False,
                requested_positions=True, vector_lower_bounds=True,
                deferred_alignments=True,
            )
            matches = [_match(value) for value in result.matches]
            certificate = asdict(result.certificate)
            seconds = float(certificate.pop("elapsed_seconds"))
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            gates = {
                "authority_equal_when_available": _authority_equal(
                    matches, authorities.get(query_id),
                ),
                "candidate_accounting": (
                    certificate["exact_evaluated"] + certificate["safely_pruned"]
                    == certificate["eligible_candidates"]
                ),
                "strict_stopping": (
                    certificate["next_lower_bound"] is not None
                    and certificate["next_lower_bound"] > certificate["stop_threshold"]
                ),
                "quantized_bound_safe": certificate["maximum_quantized_bound_excess"] <= 1e-12,
                "runtime_within_600_seconds": seconds <= 600,
                "rss_within_1536_mib": peak <= 1_536,
            }
            payload = {
                "schema_version": CASE_SCHEMA,
                "contract_digest": contract["digest"],
                "registry_digest": registry_digest,
                "registry_case_id": case["case_id"],
                "generation_id": build["generation_id"],
                "full_build_evidence_digest": build["result_digest"],
                "query_episode_id": query_id,
                "authority_digest": (
                    authorities[query_id]["authority_digest"]
                    if query_id in authorities else None
                ),
                "controls": controls,
                "matches": matches,
                "certificate": certificate,
                "certificate_digest": result.certificate.result_digest,
                "seconds": seconds,
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
                f"{seconds:.2f}s exact={certificate['exact_evaluated']}", flush=True,
            )
        except Exception as exc:
            failure = {
                "query_episode_id": query_id,
                "type": type(exc).__name__, "message": str(exc),
            }
            failures.append(failure)
            print(f"[{query_id}] ERROR {failure['type']}:{failure['message']}", flush=True)
        ordered = [completed[value] for value in expected_ids if value in completed]
        matrix = _aggregate(
            ordered, expected_ids, failures, registry_digest=registry_digest,
            build=build, controls=controls, started_at=started_at,
        )
        _write(matrix_path, matrix)
        _render(matrix_path.with_suffix(".html"), matrix)

    ordered = [completed[value] for value in expected_ids if value in completed]
    matrix = _aggregate(
        ordered, expected_ids, failures, registry_digest=registry_digest,
        build=build, controls=controls, started_at=started_at,
    )
    _write(matrix_path, matrix)
    _render(matrix_path.with_suffix(".html"), matrix)
    print(json.dumps({
        "gate_passed": matrix["gate_passed"],
        "completed_cases": matrix["completed_cases"],
        "failed_cases": matrix["failed_cases"],
        "p95_seconds": matrix["p95_seconds"],
        "maximum_seconds": matrix["maximum_seconds"],
        "total_seconds": matrix["total_seconds"],
        "peak_rss_mb": matrix["peak_rss_mb"],
        "result_digest": matrix["result_digest"],
    }, indent=2), flush=True)
    selected_complete = selected_ids.issubset(completed)
    return 0 if selected_complete and not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
