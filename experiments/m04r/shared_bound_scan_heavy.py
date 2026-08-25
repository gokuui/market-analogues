"""Compare one shared heavy-authority pack scan with repeated scalar scans."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from market_analogues.adapters import source_from_spec
from market_analogues.certified_packed_search import certified_packed_search_contract
from market_analogues.config import load_config
from market_analogues.episodes import build_episode
from market_analogues.m04r_certified_search_verification import _certificate_digest
from market_analogues.packed_bound_search import (
    PackedBoundQuery, packed_bound_batch_search_contract,
    scan_packed_bound_proposals, scan_packed_bound_proposals_many,
)
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.representation import represent
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import InstrumentKey, SearchQuery, stable_hash


SCHEMA_VERSION = "m04r-shared-bound-scan-heavy-v1"
CASE_OMITTED = {"created_at", "seconds", "peak_rss_mb", "result_digest"}
EVIDENCE_OMITTED = {
    "created_at", "batch_seconds", "scalar_seconds", "scalar_total_seconds",
    "elapsed_seconds", "speedup", "peak_rss_mb", "result_digest",
}


def _case_valid(case: dict[str, Any]) -> bool:
    deterministic = {
        key: value for key, value in case.items() if key not in CASE_OMITTED
    }
    certificate = case.get("certificate") or {}
    controls = case.get("controls") or {}
    return all((
        case.get("schema_version") == "m04r-certified-packed-search-case-v1",
        case.get("status") == "completed",
        case.get("gate_passed") is True,
        case.get("real_forward_outcomes_accessed") is False,
        case.get("result_digest") == stable_hash(deterministic),
        case.get("certificate_digest") == _certificate_digest(case),
        certificate.get("result_digest") == _certificate_digest(case),
        certificate.get("contract_digest") == certified_packed_search_contract(
            requested_positions=True,
        )["digest"],
        controls.get("requested_positions") is True,
        controls.get("hybrid_requested_positions") is False,
        int(controls.get("initial_frontier_rows", -1)) == 16_384,
        bool(certificate.get("rounds")),
        certificate.get("rounds")[-1].get("certified") is True,
    ))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_html(path: Path, payload: dict[str, Any]) -> None:
    css = "pass" if payload["gate_passed"] else "fail"
    status = "PASS" if payload["gate_passed"] else "FAIL"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>M04R-08C shared heavy scan</title><style>body{{font-family:system-ui;max-width:1200px;margin:2rem auto}}pre{{white-space:pre-wrap}}.pass{{color:#075}}.fail{{color:#a20}}</style></head><body><h1>M04R-08C shared heavy scan: <span class="{css}">{status}</span></h1><p>Proposal-scan evidence only. Exact final matches remain bound to the certified input checkpoints; outcomes and setup labels are excluded.</p><pre>{escape(json.dumps(payload, indent=2, sort_keys=True))}</pre></body></html>""")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--case-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-rows", type=int, default=2_048)
    args = parser.parse_args()
    if len(args.case_paths) < 2 or args.block_rows < 1:
        raise ValueError("at least two cases and positive block rows are required")
    cases = [json.loads(path.read_text()) for path in args.case_paths]
    if not all(_case_valid(case) for case in cases):
        raise ValueError("one or more certified input checkpoints are invalid")
    query_ids = [str(case["query_episode_id"]) for case in cases]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("certified input query IDs must be unique")

    config = load_config(args.config)
    source = source_from_spec(config.datasets["nasdaq"])
    authority_dir = (
        config.artifact_dir / "gate12" / "authorities" / "nasdaq" / "cases"
    )
    authorities = {
        str(value["query_episode_id"]): value for value in (
            json.loads(path.read_text()) for path in authority_dir.glob("*.json")
        )
    }
    build = json.loads((args.full_root / "packed-bound-full.json").read_text())
    generation_id = str(build["generation_id"])
    if any(str(case["generation_id"]) != generation_id for case in cases):
        raise ValueError("certified input generation differs")
    load_packed_generation(
        args.full_root / "store", generation_id,
        verify_content=True, validate_records=False,
    )

    queries = []
    for query_id in query_ids:
        authority = authorities.get(query_id)
        if authority is None:
            raise ValueError(f"missing frozen authority: {query_id}")
        authority_content = dict(authority)
        authority_digest = authority_content.pop("authority_digest", None)
        if not all((
            authority.get("schema_version") == "gate12-authority-v1",
            authority_digest == stable_hash(authority_content),
            authority.get("result_digest") == stable_hash(authority.get("matches")),
            authority.get("result_digest") == authority.get("repeated_digest"),
            authority_digest == next(
                case["authority_digest"] for case in cases
                if case["query_episode_id"] == query_id
            ),
        )):
            raise ValueError(f"frozen authority integrity differs: {query_id}")
        metadata = authority["query"]
        episode = build_episode(
            source, InstrumentKey("nasdaq", str(metadata["symbol"])),
            str(metadata["cutoff"]), int(metadata["lookback"]),
            str(metadata["representation_version"]),
        )
        request = SearchQuery(
            episode.key, ("nasdaq",), ("A", "B"), 20,
            False, True, 3, 60,
        )
        queries.append(PackedBoundQuery(
            episode.key.id, episode.key.instrument.source_symbol,
            int(episode.bars.timestamp.iloc[0].value),
            int(latest_eligible_cutoff(
                episode, request.minimum_history_gap_bars,
            ).value),
            represent(episode), request.quality_tiers,
        ))

    quotas = {"composite": 16_385}
    overall_started = perf_counter()
    batch = scan_packed_bound_proposals_many(
        args.full_root / "store", generation_id, queries,
        route_quotas=quotas, block_rows=args.block_rows, verify_content=False,
    )
    scalars = [scan_packed_bound_proposals(
        args.full_root / "store", generation_id, query,
        route_quotas=quotas, block_rows=args.block_rows, verify_content=False,
    ) for query in queries]
    elapsed = perf_counter() - overall_started
    comparisons = []
    for case, shared, scalar in zip(cases, batch.reports, scalars):
        certificate = case["certificate"]
        expected_proposal = certificate["rounds"][-1]["proposal_digest"]
        comparisons.append({
            "query_episode_id": case["query_episode_id"],
            "candidate_digest": shared.candidate_digest,
            "expected_candidate_digest": expected_proposal,
            "eligible_rows": shared.eligible_rows,
            "expected_eligible_rows": certificate["eligible_candidates"],
            "candidate_digest_matches_certificate": (
                shared.candidate_digest == expected_proposal
            ),
            "eligible_rows_match_certificate": (
                shared.eligible_rows == certificate["eligible_candidates"]
            ),
            "shared_matches_scalar": all((
                shared.candidates == scalar.candidates,
                shared.candidate_digest == scalar.candidate_digest,
                shared.result_digest == scalar.result_digest,
                shared.eligible_rows == scalar.eligible_rows,
                shared.route_counts == scalar.route_counts,
            )),
        })
    scalar_seconds = [report.elapsed_seconds for report in scalars]
    scalar_total = sum(scalar_seconds)
    speedup = scalar_total / batch.elapsed_seconds
    gates = {
        "all_inputs_valid": True,
        "all_shared_reports_match_scalar": all(
            row["shared_matches_scalar"] for row in comparisons
        ),
        "all_candidate_digests_match_certificates": all(
            row["candidate_digest_matches_certificate"] for row in comparisons
        ),
        "all_eligible_counts_match_certificates": all(
            row["eligible_rows_match_certificate"] for row in comparisons
        ),
        "physical_row_accounting": (
            batch.logical_rows_evaluated
            == batch.physical_rows_scanned * len(queries)
        ),
        "shared_at_least_10pct_faster": batch.elapsed_seconds <= .9 * scalar_total,
        "rss_within_1536_mib": batch.peak_rss_mb <= 1_536,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "batch_contract_digest": packed_bound_batch_search_contract()["digest"],
        "generation_id": generation_id,
        "query_episode_ids": query_ids,
        "controls": {"block_rows": args.block_rows, "route_quotas": quotas},
        "comparisons": comparisons,
        "physical_rows_scanned": batch.physical_rows_scanned,
        "logical_rows_evaluated": batch.logical_rows_evaluated,
        "batch_seconds": batch.elapsed_seconds,
        "scalar_seconds": scalar_seconds,
        "scalar_total_seconds": scalar_total,
        "elapsed_seconds": elapsed,
        "speedup": speedup,
        "peak_rss_mb": batch.peak_rss_mb,
        "batch_result_digest": batch.result_digest,
        "gates": gates,
        "gate_passed": all(gates.values()),
        "real_forward_outcomes_accessed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    payload["result_digest"] = stable_hash({
        key: value for key, value in payload.items()
        if key not in EVIDENCE_OMITTED
    })
    _write_json(args.output, payload)
    _write_html(args.output.with_suffix(".html"), payload)
    print(json.dumps({
        "gate_passed": payload["gate_passed"],
        "batch_seconds": payload["batch_seconds"],
        "scalar_total_seconds": payload["scalar_total_seconds"],
        "speedup": payload["speedup"],
        "peak_rss_mb": payload["peak_rss_mb"],
        "result_digest": payload["result_digest"],
    }, indent=2), flush=True)
    return 0 if payload["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
