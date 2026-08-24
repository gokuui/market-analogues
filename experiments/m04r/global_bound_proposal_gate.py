"""Run M04R-07 global bound proposal selection against the full shadow pack."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from market_analogues.adapters import source_from_spec
from market_analogues.config import load_config
from market_analogues.episodes import build_episode
from market_analogues.packed_bound_search import (
    PackedBoundQuery, packed_bound_search_contract, scan_packed_bound_proposals,
)
from market_analogues.representation import represent
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import InstrumentKey, stable_hash


SCHEMA_VERSION = "m04r-global-bound-proposal-gate-v1"
NONDETERMINISTIC = {"created_at", "forward_seconds", "reverse_seconds", "peak_rss_mb", "result_digest"}


def _json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_query(config: Any, source: Any, episode_id: str) -> tuple[PackedBoundQuery, dict[str, Any]]:
    paths = sorted((
        config.artifact_dir / "gate12" / "authorities" / "nasdaq" / "cases"
    ).glob("*.json"))
    cases = [json.loads(path.read_text()) for path in paths]
    selected = [case for case in cases if case["query_episode_id"] == episode_id]
    if len(selected) != 1:
        raise ValueError("cannot resolve preregistered full-pack benchmark query")
    authority = selected[0]
    metadata = authority["query"]
    episode = build_episode(
        source, InstrumentKey(str(metadata["dataset_id"]), str(metadata["symbol"])),
        str(metadata["cutoff"]), int(metadata["lookback"]),
        str(metadata["representation_version"]),
    )
    query = PackedBoundQuery(
        episode.key.id, episode.key.instrument.source_symbol,
        int(pd.Timestamp(episode.bars.timestamp.iloc[0]).value),
        int(latest_eligible_cutoff(episode, 60).value), represent(episode),
        ("A", "B"),
    )
    return query, authority


def _report_payload(report: Any, query: PackedBoundQuery) -> dict[str, Any]:
    composite = [row for row in report.candidates if "composite" in row.routes]
    score_digest = sha256(np.asarray(
        [row.lower_bound for row in composite], dtype="<f8",
    ).tobytes()).hexdigest()
    return {
        "generation_id": report.generation_id,
        "query_episode_id": report.query_episode_id,
        "rows_scanned": report.rows_scanned,
        "eligible_rows": report.eligible_rows,
        "eligible_main_rows": report.eligible_main_rows,
        "eligible_overflow_rows": report.eligible_overflow_rows,
        "candidate_count": len(report.candidates),
        "composite_count": len(composite),
        "route_counts": dict(report.route_counts),
        "route_quotas": dict(report.route_quotas),
        "candidate_digest": report.candidate_digest,
        "top_1000_score_digest": score_digest,
        "duplicate_candidates": len(report.candidates) - len({
            row.episode_id for row in report.candidates
        }),
        "future_candidates": sum(
            row.cutoff_ns > query.latest_eligible_ns for row in report.candidates
        ),
        "overlapping_same_symbol_candidates": sum(
            row.symbol == query.symbol and row.cutoff_ns >= query.query_start_ns
            for row in report.candidates
        ),
        "overflow_candidates": sum(row.overflow_fallback for row in report.candidates),
        "composite_episode_ids": [row.episode_id for row in composite],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-rows", type=int, default=2_048)
    args = parser.parse_args()

    config = load_config(args.config)
    build = json.loads((args.full_root / "packed-bound-full.json").read_text())
    rank = json.loads((
        config.artifact_dir / "poc" / "m04r" / "quantized-bound-rank-full.json"
    ).read_text())
    source = source_from_spec(config.datasets["nasdaq"])
    benchmark_id = str(build["benchmark_selection"]["query_episode_id"])
    query, authority = _load_query(config, source, benchmark_id)
    generation_id = str(build["generation_id"])
    store_root = args.full_root / "store"

    forward = scan_packed_bound_proposals(
        store_root, generation_id, query, block_rows=args.block_rows,
        block_order="forward", verify_content=True,
    )
    reverse = scan_packed_bound_proposals(
        store_root, generation_id, query, block_rows=args.block_rows * 2 + 1,
        block_order="reverse", verify_content=False,
    )
    forward_payload = _report_payload(forward, query)
    reverse_payload = _report_payload(reverse, query)
    rank_case = next(
        case for case in rank["authority_cases"]
        if case["query_episode_id"] == benchmark_id
    )
    target_ids = [str(row["episode_id"]) for row in rank_case["targets"]]
    composite_ids = set(forward_payload["composite_episode_ids"])
    target_retention = {
        "target_count": len(target_ids),
        "retained_count": sum(value in composite_ids for value in target_ids),
        "missing_ids": sorted(set(target_ids) - composite_ids),
        "rank_evidence_digest": rank["result_digest"],
        "rank_evidence_all_12_top_1000": bool(rank["top_1000_recall_passed"]),
        "rank_evidence_maximum_target_rank": int(rank["maximum_target_rank"]),
    }
    prior_score_digest = str(build["warm_scans_second"][0]["top_1000_digest"])
    invariant_fields = (
        "generation_id", "query_episode_id", "rows_scanned", "eligible_rows",
        "eligible_main_rows", "eligible_overflow_rows", "candidate_count",
        "composite_count", "route_counts", "route_quotas", "candidate_digest",
        "top_1000_score_digest", "duplicate_candidates", "future_candidates",
        "overlapping_same_symbol_candidates", "overflow_candidates",
        "composite_episode_ids",
    )
    invariant = all(
        forward_payload[field] == reverse_payload[field]
        for field in invariant_fields
    )
    expected_eligible = int(authority["certificate"]["eligible_candidates"])
    gates = {
        "full_row_accounting": forward.eligible_rows == expected_eligible,
        "forward_reverse_and_block_invariance": invariant,
        "top_1000_scores_equal_m04r_06c": (
            forward_payload["top_1000_score_digest"] == prior_score_digest
        ),
        "all_selected_authority_targets_retained": target_retention["retained_count"] == 20,
        "all_12_rank_evidence_supports_composite_quota": (
            target_retention["rank_evidence_all_12_top_1000"]
            and target_retention["rank_evidence_maximum_target_rank"] <= 1_000
        ),
        "no_duplicates_or_temporal_violations": all((
            forward_payload["duplicate_candidates"] == 0,
            forward_payload["future_candidates"] == 0,
            forward_payload["overlapping_same_symbol_candidates"] == 0,
        )),
        "composite_quota_preserved": forward_payload["composite_count"] == 1_000,
        "overflow_fallback_exercised": forward_payload["overflow_candidates"] > 0,
        "rss_within_768_mib": max(forward.peak_rss_mb, reverse.peak_rss_mb) <= 768,
        "each_scan_within_600_seconds": max(
            forward.elapsed_seconds, reverse.elapsed_seconds,
        ) <= 600,
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract_digest": packed_bound_search_contract()["digest"],
        "generation_id": generation_id,
        "full_build_evidence_digest": build["result_digest"],
        "query_episode_id": benchmark_id,
        "selection_method": build["benchmark_selection"],
        "block_rows_forward": args.block_rows,
        "block_rows_reverse": args.block_rows * 2 + 1,
        "forward": forward_payload,
        "reverse": reverse_payload,
        "target_retention": target_retention,
        "prior_top_1000_score_digest": prior_score_digest,
        "gates": gates,
        "gate_passed": all(gates.values()),
        "forward_seconds": forward.elapsed_seconds,
        "reverse_seconds": reverse.elapsed_seconds,
        "peak_rss_mb": max(forward.peak_rss_mb, reverse.peak_rss_mb),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "real_forward_outcomes_accessed": False,
    }
    deterministic = {key: value for key, value in payload.items() if key not in NONDETERMINISTIC}
    payload["result_digest"] = stable_hash(deterministic)
    _json(args.output, payload)
    html = args.output.with_suffix(".html")
    html.write_text(f"""<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>M04R-07 global bound proposal gate</title><style>body{{font-family:system-ui;max-width:1200px;margin:2rem auto}}pre{{white-space:pre-wrap}}.pass{{color:#075}}.fail{{color:#a20}}</style></head><body><h1>M04R-07 global bound proposal: <span class=\"{'pass' if payload['gate_passed'] else 'fail'}\">{'PASS' if payload['gate_passed'] else 'FAIL'}</span></h1><p>Full shadow-pack selection with forward/reverse and unequal-block verification. No outcomes or setup labels are accessed.</p><p>Evidence <code>{payload['result_digest']}</code>.</p><pre>{escape(json.dumps({key: value for key, value in payload.items() if key not in ('forward', 'reverse')}, indent=2, sort_keys=True))}</pre></body></html>""")
    print(json.dumps({
        "gate_passed": payload["gate_passed"],
        "candidate_digest": forward.candidate_digest,
        "candidate_count": len(forward.candidates),
        "forward_seconds": forward.elapsed_seconds,
        "reverse_seconds": reverse.elapsed_seconds,
        "peak_rss_mb": payload["peak_rss_mb"],
        "result_digest": payload["result_digest"],
    }, indent=2))
    return 0 if payload["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
