"""Run resumable M04R-08B certified completion over all NASDAQ authorities."""

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

from market_analogues.adapters import source_from_spec
from market_analogues.certified_packed_search import (
    certified_packed_search, certified_packed_search_contract,
)
from market_analogues.config import load_config
from market_analogues.episodes import build_episode
from market_analogues.m04r_full_pack_verification import (
    EVIDENCE_OMITTED as FULL_BUILD_OMITTED,
)
from market_analogues.m04r_certified_search_verification import _certificate_digest
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.types import InstrumentKey, SearchQuery, stable_hash


CASE_SCHEMA = "m04r-certified-packed-search-case-v1"
MATRIX_SCHEMA = "m04r-certified-packed-search-all12-v1"
CASE_OMITTED = {"created_at", "seconds", "peak_rss_mb", "result_digest"}
MATRIX_OMITTED = {
    "created_at", "started_at", "p95_seconds", "maximum_seconds",
    "total_seconds", "peak_rss_mb", "result_digest",
}
TOTAL_TOLERANCE = 1e-7
COMPONENT_TOLERANCE = 1e-6
BOUND_TOLERANCE = 1e-12


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _render(path: Path, payload: dict[str, Any]) -> None:
    status = "PASS" if payload.get("gate_passed") else "INCOMPLETE/FAIL"
    css = "pass" if payload.get("gate_passed") else "fail"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(f"""<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>M04R-08B all-12 certified search</title><style>body{{font-family:system-ui;max-width:1200px;margin:2rem auto}}pre{{white-space:pre-wrap}}.pass{{color:#075}}.fail{{color:#a20}}</style></head><body><h1>M04R-08B all-12 certified search: <span class=\"{css}\">{status}</span></h1><p>Development-authority correctness/performance evidence only. Forward outcomes and setup labels are excluded.</p><pre>{escape(json.dumps(payload, indent=2, sort_keys=True))}</pre></body></html>""")
    temporary.replace(path)


def _deterministic(
    payload: dict[str, Any], omitted: set[str],
) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in omitted}


def _matrix_deterministic(payload: dict[str, Any]) -> dict[str, Any]:
    deterministic = _deterministic(payload, MATRIX_OMITTED)
    deterministic["cases"] = [
        _deterministic(case, CASE_OMITTED) for case in payload.get("cases", [])
    ]
    return deterministic


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


def _comparison(
    matches: list[dict[str, Any]], expected: list[dict[str, Any]],
) -> dict[str, Any]:
    same_length = len(matches) == len(expected) and bool(matches)
    ids_equal = same_length and [row["episode_id"] for row in matches] == [
        row["episode_id"] for row in expected
    ]
    alignments_equal = same_length and [row["alignment"] for row in matches] == [
        row["alignment"] for row in expected
    ]
    component_shape = same_length and all(
        set(left["component_distances"]) == set(right["component_distances"])
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
            ) if component_shape else None
        )
    return {
        "ordered_ids_equal_authority": ids_equal,
        "alignments_equal_authority": alignments_equal,
        "component_names_equal_authority": component_shape,
        "maximum_total_delta": total_delta,
        "maximum_component_delta": component_delta,
    }


def _case_payload(
    result: Any,
    authority: dict[str, Any],
    build: dict[str, Any],
    controls: dict[str, Any],
    requested_positions: bool,
    hybrid_requested_positions: bool,
    vector_lower_bounds: bool,
    deferred_alignments: bool,
) -> dict[str, Any]:
    matches = [_match(value) for value in result.matches]
    comparison = _comparison(matches, authority["matches"])
    certificate = asdict(result.certificate)
    seconds = float(certificate.pop("elapsed_seconds"))
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    gates = {
        "ordered_ids_equal_authority": comparison["ordered_ids_equal_authority"],
        "alignments_equal_authority": comparison["alignments_equal_authority"],
        "component_names_equal_authority": (
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
        "runtime_within_600_seconds": seconds <= 600,
        "rss_within_1536_mib": peak_rss <= 1_536,
    }
    payload = {
        "schema_version": CASE_SCHEMA,
        "contract_digest": certified_packed_search_contract(
            requested_positions=requested_positions,
            hybrid_requested_positions=hybrid_requested_positions,
            vector_lower_bounds=vector_lower_bounds,
            deferred_alignments=deferred_alignments,
        )["digest"],
        "generation_id": build["generation_id"],
        "full_build_evidence_digest": build["result_digest"],
        "query_episode_id": authority["query_episode_id"],
        "authority_digest": authority["authority_digest"],
        "controls": controls,
        "matches": matches,
        **comparison,
        "certificate": certificate,
        "certificate_digest": result.certificate.result_digest,
        "seconds": seconds,
        "peak_rss_mb": peak_rss,
        "gates": gates,
        "gate_passed": all(gates.values()),
        "status": "completed",
        "real_forward_outcomes_accessed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    payload["result_digest"] = stable_hash(_deterministic(payload, CASE_OMITTED))
    return payload


def _valid_checkpoint(
    payload: dict[str, Any], authority: dict[str, Any], build: dict[str, Any],
    expected_controls: dict[str, Any] | None = None,
) -> bool:
    requested_positions = bool(
        (payload.get("controls") or {}).get("requested_positions", False)
    )
    hybrid_requested_positions = bool(
        (payload.get("controls") or {}).get(
            "hybrid_requested_positions", False,
        )
    )
    vector_lower_bounds = bool(
        (payload.get("controls") or {}).get("vector_lower_bounds", False)
    )
    deferred_alignments = bool(
        (payload.get("controls") or {}).get("deferred_alignments", False)
    )
    contract = certified_packed_search_contract(
        requested_positions=requested_positions,
        hybrid_requested_positions=hybrid_requested_positions,
        vector_lower_bounds=vector_lower_bounds,
        deferred_alignments=deferred_alignments,
    )
    if not all((
        payload.get("schema_version") == CASE_SCHEMA,
        payload.get("status") == "completed",
        payload.get("contract_digest") == contract["digest"],
        payload.get("generation_id") == build.get("generation_id"),
        payload.get("full_build_evidence_digest") == build.get("result_digest"),
        payload.get("query_episode_id") == authority.get("query_episode_id"),
        payload.get("authority_digest") == authority.get("authority_digest"),
        expected_controls is None or payload.get("controls") == expected_controls,
        payload.get("result_digest") == stable_hash(
            _deterministic(payload, CASE_OMITTED)
        ),
        payload.get("real_forward_outcomes_accessed") is False,
    )):
        return False
    comparison = _comparison(payload.get("matches") or [], authority["matches"])
    if not all(payload.get(key) == value for key, value in comparison.items()):
        return False
    certificate = payload.get("certificate") or {}
    try:
        expected_gates = {
            "ordered_ids_equal_authority": comparison["ordered_ids_equal_authority"],
            "alignments_equal_authority": comparison[
                "alignments_equal_authority"
            ],
            "component_names_equal_authority": comparison[
                "component_names_equal_authority"
            ],
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
            "runtime_within_600_seconds": float(payload["seconds"]) <= 600,
            "rss_within_1536_mib": float(payload["peak_rss_mb"]) <= 1_536,
        }
        certificate_valid = all((
            certificate.get("generation_id") == build.get("generation_id"),
            certificate.get("query_episode_id") == authority.get("query_episode_id"),
            certificate.get("contract_digest") == contract["digest"],
            certificate.get("result_digest") == _certificate_digest(payload),
            payload.get("certificate_digest") == _certificate_digest(payload),
        ))
    except (KeyError, TypeError, ValueError):
        return False
    return all((
        certificate_valid,
        payload.get("gates") == expected_gates,
        payload.get("gate_passed") == all(expected_gates.values()),
    ))


def _aggregate(
    cases: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    *,
    build: dict[str, Any],
    expected_ids: list[str],
    started_at: str,
    requested_positions: bool,
    hybrid_requested_positions: bool,
    vector_lower_bounds: bool,
    deferred_alignments: bool,
) -> dict[str, Any]:
    seconds = [float(case["seconds"]) for case in cases]
    completed_ids = [str(case["query_episode_id"]) for case in cases]
    p95 = float(np.percentile(seconds, 95)) if seconds else None
    maximum = max(seconds) if seconds else None
    peak_rss = max(
        (float(case["peak_rss_mb"]) for case in cases), default=0.0,
    )
    gates = {
        "all_12_completed": (
            len(cases) == 12 and completed_ids == expected_ids and not failures
        ),
        "all_case_gates_passed": len(cases) == 12 and all(
            case.get("gate_passed") is True for case in cases
        ),
        "all_certificates_unique_to_query": len({
            (
                case["query_episode_id"],
                case["certificate"]["input_digest"],
                case["certificate_digest"],
            ) for case in cases
        }) == len(cases),
        "p95_within_300_seconds": p95 is not None and p95 <= 300,
        "maximum_within_600_seconds": maximum is not None and maximum <= 600,
        "rss_within_1536_mib": peak_rss <= 1_536,
    }
    payload = {
        "schema_version": MATRIX_SCHEMA,
        "contract_digest": certified_packed_search_contract(
            requested_positions=requested_positions,
            hybrid_requested_positions=hybrid_requested_positions,
            vector_lower_bounds=vector_lower_bounds,
            deferred_alignments=deferred_alignments,
        )["digest"],
        "generation_id": build["generation_id"],
        "full_build_evidence_digest": build["result_digest"],
        "expected_query_episode_ids": expected_ids,
        "completed_query_episode_ids": completed_ids,
        "completed_cases": len(cases),
        "failed_cases": failures,
        "cases": cases,
        "p95_seconds": p95,
        "maximum_seconds": maximum,
        "total_seconds": sum(seconds),
        "peak_rss_mb": peak_rss,
        "gates": gates,
        "gate_passed": all(gates.values()),
        "real_forward_outcomes_accessed": False,
        "started_at": started_at,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    payload["result_digest"] = stable_hash(_matrix_deterministic(payload))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--block-rows", type=int, default=2_048)
    parser.add_argument("--initial-frontier-rows", type=int, default=8_192)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--query-ids", nargs="+")
    parser.add_argument("--requested-positions", action="store_true")
    parser.add_argument("--hybrid-requested-positions", action="store_true")
    parser.add_argument("--vector-lower-bounds", action="store_true")
    parser.add_argument("--deferred-alignments", action="store_true")
    args = parser.parse_args()
    if args.requested_positions and args.hybrid_requested_positions:
        raise ValueError("requested-position modes are mutually exclusive")
    if args.vector_lower_bounds and not args.requested_positions:
        raise ValueError("vector lower bounds require requested positions")
    if args.deferred_alignments and not args.vector_lower_bounds:
        raise ValueError("deferred alignments require vector lower bounds")
    if (
        args.workers < 1 or args.block_rows < 1
        or args.initial_frontier_rows < 512
        or args.initial_frontier_rows > 32_768
    ):
        raise ValueError("workers, block rows and frontier limits are invalid")
    if args.case_limit is not None and args.case_limit < 1:
        raise ValueError("case limit must be positive")
    config = load_config(args.config)
    build = json.loads((args.full_root / "packed-bound-full.json").read_text())
    build_content = {
        key: value for key, value in build.items()
        if key not in FULL_BUILD_OMITTED
    }
    if not all((
        build.get("schema_version") == "m04r-packed-bound-full-build-v1",
        build.get("result_digest") == stable_hash(build_content),
        build.get("gate_passed") is True,
        build.get("shadow_generation") is True,
        build.get("real_forward_outcomes_accessed") is False,
    )):
        raise ValueError("full packed-build evidence integrity or gate differs")
    if (args.full_root / "store" / "active.json").exists():
        raise ValueError("shadow generation unexpectedly has an active pointer")
    load_packed_generation(
        args.full_root / "store", str(build["generation_id"]),
        verify_content=True, validate_records=False,
    )
    authority_dir = (
        config.artifact_dir / "gate12" / "authorities" / "nasdaq" / "cases"
    )
    authority_paths = sorted(authority_dir.glob("*.json"))
    if len(authority_paths) != 12:
        raise ValueError(f"expected 12 NASDAQ authorities, found {len(authority_paths)}")
    authorities = [json.loads(path.read_text()) for path in authority_paths]
    for authority in authorities:
        content = dict(authority)
        claimed = content.pop("authority_digest", None)
        if not all((
            authority.get("schema_version") == "gate12-authority-v1",
            claimed == stable_hash(content),
            authority.get("result_digest") == stable_hash(authority.get("matches")),
            authority.get("result_digest") == authority.get("repeated_digest"),
        )):
            raise ValueError(
                f"authority integrity differs: {authority.get('query_episode_id')}"
            )
    expected_ids = [str(authority["query_episode_id"]) for authority in authorities]
    if args.query_ids:
        requested_ids = set(args.query_ids)
        unknown = requested_ids.difference(expected_ids)
        if unknown:
            raise ValueError(f"unknown query IDs: {sorted(unknown)}")
        selected = [
            authority for authority in authorities
            if str(authority["query_episode_id"]) in requested_ids
        ]
    else:
        selected = authorities[:args.case_limit] if args.case_limit else authorities
    source = source_from_spec(config.datasets["nasdaq"])
    cases_dir = args.output_root / "cases"
    matrix_path = args.output_root / "certified-packed-search-all12.json"
    started_at = datetime.now(timezone.utc).isoformat()
    completed: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    controls = {
        "block_rows": args.block_rows,
        "workers": args.workers,
        "initial_frontier_rows": args.initial_frontier_rows,
        "requested_positions": args.requested_positions,
        "hybrid_requested_positions": args.hybrid_requested_positions,
        "vector_lower_bounds": args.vector_lower_bounds,
        "deferred_alignments": args.deferred_alignments,
    }

    for authority in authorities:
        query_id = str(authority["query_episode_id"])
        checkpoint = cases_dir / f"{query_id}.json"
        if not checkpoint.exists():
            continue
        try:
            payload = json.loads(checkpoint.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if _valid_checkpoint(payload, authority, build, controls):
            completed[query_id] = payload

    for authority in selected:
        query_id = str(authority["query_episode_id"])
        if query_id in completed:
            print(f"[{query_id}] resume verified", flush=True)
            continue
        metadata = authority["query"]
        print(
            f"[{query_id}] start {metadata['symbol']} {metadata['cutoff']}",
            flush=True,
        )
        try:
            query = build_episode(
                source, InstrumentKey("nasdaq", str(metadata["symbol"])),
                str(metadata["cutoff"]), int(metadata["lookback"]),
                str(metadata["representation_version"]),
            )
            request = SearchQuery(
                query.key, ("nasdaq",), ("A", "B"), 20,
                False, True, 3, 60,
            )
            result = certified_packed_search(
                query, source, request, args.full_root / "store",
                str(build["generation_id"]), store_dataset_id="nasdaq",
                initial_frontier_rows=args.initial_frontier_rows,
                maximum_frontier_rows=32_768,
                seed_rows=512,
                block_rows=args.block_rows,
                workers=args.workers,
                sparse_cutoff=8,
                verify_content=False,
                requested_positions=args.requested_positions,
                hybrid_requested_positions=args.hybrid_requested_positions,
                vector_lower_bounds=args.vector_lower_bounds,
                deferred_alignments=args.deferred_alignments,
            )
            payload = _case_payload(
                result, authority, build,
                controls,
                args.requested_positions,
                args.hybrid_requested_positions,
                args.vector_lower_bounds,
                args.deferred_alignments,
            )
            _write(cases_dir / f"{query_id}.json", payload)
            completed[query_id] = payload
            print(
                f"[{query_id}] {'PASS' if payload['gate_passed'] else 'FAIL'} "
                f"{payload['seconds']:.2f}s exact="
                f"{payload['certificate']['exact_evaluated']}",
                flush=True,
            )
        except Exception as exc:
            failure = {
                "query_episode_id": query_id,
                "type": type(exc).__name__,
                "message": str(exc),
            }
            failures.append(failure)
            print(f"[{query_id}] ERROR {failure['type']}:{failure['message']}", flush=True)
        ordered = [
            completed[query_id] for query_id in expected_ids if query_id in completed
        ]
        matrix = _aggregate(
            ordered, failures, build=build, expected_ids=expected_ids,
            started_at=started_at,
            requested_positions=args.requested_positions,
            hybrid_requested_positions=args.hybrid_requested_positions,
            vector_lower_bounds=args.vector_lower_bounds,
            deferred_alignments=args.deferred_alignments,
        )
        _write(matrix_path, matrix)
        _render(matrix_path.with_suffix(".html"), matrix)

    ordered = [completed[query_id] for query_id in expected_ids if query_id in completed]
    matrix = _aggregate(
        ordered, failures, build=build, expected_ids=expected_ids,
        started_at=started_at,
        requested_positions=args.requested_positions,
        hybrid_requested_positions=args.hybrid_requested_positions,
        vector_lower_bounds=args.vector_lower_bounds,
        deferred_alignments=args.deferred_alignments,
    )
    _write(matrix_path, matrix)
    _render(matrix_path.with_suffix(".html"), matrix)
    print(json.dumps({
        "gate_passed": matrix["gate_passed"],
        "completed_cases": matrix["completed_cases"],
        "failed_cases": matrix["failed_cases"],
        "p95_seconds": matrix["p95_seconds"],
        "maximum_seconds": matrix["maximum_seconds"],
        "peak_rss_mb": matrix["peak_rss_mb"],
        "result_digest": matrix["result_digest"],
    }, indent=2), flush=True)
    expected_selected_complete = all(
        str(authority["query_episode_id"]) in completed for authority in selected
    )
    return 0 if expected_selected_complete and not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
