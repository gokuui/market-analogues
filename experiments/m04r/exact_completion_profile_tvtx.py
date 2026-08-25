"""Profile frozen TVTX exact completion without changing its certificate schema."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, Callable

from market_analogues import certified_packed_search as completion
from market_analogues.adapters import source_from_spec
from market_analogues.certified_packed_search import _score_new_proposals
from market_analogues.config import load_config
from market_analogues.episodes import build_episode
from market_analogues.m04r_certified_search_verification import _certificate_digest
from market_analogues.packed_bound_search import (
    PackedBoundQuery, scan_packed_bound_proposals,
)
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.representation import represent
from market_analogues.search import latest_eligible_cutoff, select_scored
from market_analogues.types import InstrumentKey, SearchQuery, stable_hash


SCHEMA_VERSION = "m04r-exact-completion-profile-tvtx-v1"
CASE_OMITTED = {"created_at", "seconds", "peak_rss_mb", "result_digest"}
EVIDENCE_OMITTED = {
    "created_at", "scan_seconds", "seed_seconds", "completion_seconds",
    "exact_wall_seconds", "phase_cumulative_seconds", "group_cumulative_seconds",
    "result_digest",
}


class TimedSource:
    def __init__(self, source: Any, record: Callable[[str, float], None]):
        self._source = source
        self._record = record

    def load(self, key: Any) -> Any:
        started = perf_counter()
        try:
            return self._source.load(key)
        finally:
            self._record("source_load", perf_counter() - started)

    def load_benchmark(self) -> Any:
        started = perf_counter()
        try:
            return self._source.load_benchmark()
        finally:
            self._record("benchmark_load", perf_counter() - started)


def _valid_case(case: dict[str, Any]) -> bool:
    deterministic = {
        key: value for key, value in case.items() if key not in CASE_OMITTED
    }
    certificate = case.get("certificate") or {}
    return all((
        case.get("gate_passed") is True,
        case.get("result_digest") == stable_hash(deterministic),
        case.get("certificate_digest") == _certificate_digest(case),
        certificate.get("result_digest") == _certificate_digest(case),
        int(certificate.get("exact_evaluated", -1)) == 9_309,
        bool(certificate.get("rounds")),
        certificate.get("rounds")[-1].get("certified") is True,
    ))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    html = path.with_suffix(".html")
    temporary = html.with_suffix(html.suffix + ".tmp")
    temporary.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>M04R exact-completion profile</title><style>body{{font-family:system-ui;max-width:1200px;margin:2rem auto}}pre{{white-space:pre-wrap}}.pass{{color:#075}}.fail{{color:#a20}}</style></head><body><h1>TVTX exact-completion profile: <span class="{'pass' if payload['gate_passed'] else 'fail'}">{'PASS' if payload['gate_passed'] else 'FAIL'}</span></h1><p>Cumulative phase timings overlap across worker threads and are diagnostic, not certificate latency. Outcomes and setup labels are excluded.</p><pre>{escape(json.dumps(payload, indent=2, sort_keys=True))}</pre></body></html>""")
    temporary.replace(html)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--block-rows", type=int, default=2_048)
    args = parser.parse_args()
    if args.workers < 1 or args.block_rows < 1:
        raise ValueError("workers and block rows must be positive")
    case = json.loads(args.case.read_text())
    if not _valid_case(case):
        raise ValueError("certified TVTX checkpoint is invalid")
    query_id = str(case["query_episode_id"])
    config = load_config(args.config)
    authority = next((
        json.loads(path.read_text())
        for path in (
            config.artifact_dir / "gate12" / "authorities" / "nasdaq" / "cases"
        ).glob("*.json")
        if json.loads(path.read_text())["query_episode_id"] == query_id
    ), None)
    if authority is None or authority.get("authority_digest") != case.get("authority_digest"):
        raise ValueError("frozen authority binding differs")
    build = json.loads((args.full_root / "packed-bound-full.json").read_text())
    generation_id = str(build["generation_id"])
    loaded = load_packed_generation(
        args.full_root / "store", generation_id,
        verify_content=True, validate_records=False,
    )
    source = source_from_spec(config.datasets["nasdaq"])
    metadata = authority["query"]
    query = build_episode(
        source, InstrumentKey("nasdaq", str(metadata["symbol"])),
        str(metadata["cutoff"]), int(metadata["lookback"]),
        str(metadata["representation_version"]),
    )
    request = SearchQuery(
        query.key, ("nasdaq",), ("A", "B"), 20,
        False, True, 3, 60,
    )
    packed_query = PackedBoundQuery(
        query.key.id, query.key.instrument.source_symbol,
        int(query.bars.timestamp.iloc[0].value),
        int(latest_eligible_cutoff(
            query, request.minimum_history_gap_bars,
        ).value),
        represent(query), request.quality_tiers,
    )
    proposal = scan_packed_bound_proposals(
        args.full_root / "store", generation_id, packed_query,
        route_quotas={"composite": 16_385}, block_rows=args.block_rows,
        verify_content=False,
    )
    frontier = proposal.candidates[:16_384]

    lock = Lock()
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    group_totals: dict[str, float] = defaultdict(float)
    group_counts: dict[str, int] = defaultdict(int)

    def record(name: str, seconds: float) -> None:
        with lock:
            totals[name] += seconds
            counts[name] += 1

    originals: dict[str, Any] = {}
    for name in (
        "_prefix_matches", "exact_representations_at_positions",
        "representation_distance_lower_bound", "complete_representation_distance",
        "eligible", "_score_group",
    ):
        originals[name] = getattr(completion, name)

    def timed(name: str, function: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*values: Any, **options: Any) -> Any:
            started = perf_counter()
            try:
                return function(*values, **options)
            finally:
                record(name, perf_counter() - started)
        return wrapper

    original_group = originals["_score_group"]

    def timed_group(*values: Any, **options: Any) -> Any:
        proposals = values[1]
        mode = "dense" if len(proposals) >= 8 else "sparse"
        started = perf_counter()
        try:
            return original_group(*values, **options)
        finally:
            with lock:
                group_totals[mode] += perf_counter() - started
                group_counts[mode] += 1

    for name, function in originals.items():
        if name != "_score_group":
            setattr(completion, name, timed(name, function))
    setattr(completion, "_score_group", timed_group)
    timed_source = TimedSource(source, record)
    try:
        exact_started = perf_counter()
        seed_started = perf_counter()
        seed, seed_excess, seed_sparse, seed_batch = _score_new_proposals(
            frontier[:512], query=query, source=timed_source, request=request,
            store_dataset_id="nasdaq", manifest=loaded.manifest,
            workers=args.workers, sparse_cutoff=8, tolerance=1e-12,
            requested_positions=True, hybrid_requested_positions=False,
        )
        seed_seconds = perf_counter() - seed_started
        selected = select_scored(seed, request)
        threshold = max(row.total_distance for row in selected)
        seed_ids = {row.match.episode_key.id for row in seed}
        pending = [
            row for row in frontier
            if row.episode_id not in seed_ids and row.lower_bound <= threshold
        ]
        completion_started = perf_counter()
        rest, rest_excess, rest_sparse, rest_batch = _score_new_proposals(
            pending, query=query, source=timed_source, request=request,
            store_dataset_id="nasdaq", manifest=loaded.manifest,
            workers=args.workers, sparse_cutoff=8, tolerance=1e-12,
            requested_positions=True, hybrid_requested_positions=False,
        )
        completion_seconds = perf_counter() - completion_started
        exact_wall = perf_counter() - exact_started
    finally:
        for name, function in originals.items():
            setattr(completion, name, function)

    scored = seed + rest
    matches = select_scored(scored, request)
    match_ids = [row.episode_key.id for row in matches]
    expected_ids = [str(row["episode_id"]) for row in case["matches"]]
    expected_threshold = float(case["certificate"]["stop_threshold"])
    next_lower = float(proposal.candidates[16_384].lower_bound)
    gates = {
        "candidate_digest_matches_certificate": (
            proposal.candidate_digest
            == case["certificate"]["rounds"][-1]["proposal_digest"]
        ),
        "eligible_count_matches_certificate": (
            proposal.eligible_rows == case["certificate"]["eligible_candidates"]
        ),
        "exact_count_matches_certificate": len(scored) == 9_309,
        "ordered_ids_match_certificate": match_ids == expected_ids,
        "threshold_matches_certificate": threshold == expected_threshold,
        "strict_stopping": next_lower > threshold,
        "quantized_bound_safe": max(seed_excess, rest_excess) <= 1e-12,
        "group_accounting": (
            seed_sparse + seed_batch + rest_sparse + rest_batch
            == sum(group_counts.values())
        ),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generation_id": generation_id,
        "query_episode_id": query_id,
        "certified_case_digest": case["result_digest"],
        "controls": {
            "workers": args.workers, "block_rows": args.block_rows,
            "frontier_rows": 16_384, "seed_rows": 512,
            "requested_positions": True,
        },
        "eligible_rows": proposal.eligible_rows,
        "candidate_digest": proposal.candidate_digest,
        "scan_seconds": proposal.elapsed_seconds,
        "seed_seconds": seed_seconds,
        "completion_seconds": completion_seconds,
        "exact_wall_seconds": exact_wall,
        "seed_rows": len(seed),
        "pending_rows": len(rest),
        "exact_rows": len(scored),
        "sparse_groups": seed_sparse + rest_sparse,
        "dense_groups": seed_batch + rest_batch,
        "phase_call_counts": dict(sorted(counts.items())),
        "phase_cumulative_seconds": dict(sorted(totals.items())),
        "group_call_counts": dict(sorted(group_counts.items())),
        "group_cumulative_seconds": dict(sorted(group_totals.items())),
        "gates": gates,
        "gate_passed": all(gates.values()),
        "real_forward_outcomes_accessed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    payload["result_digest"] = stable_hash({
        key: value for key, value in payload.items() if key not in EVIDENCE_OMITTED
    })
    _write(args.output, payload)
    print(json.dumps({
        "gate_passed": payload["gate_passed"],
        "scan_seconds": payload["scan_seconds"],
        "exact_wall_seconds": payload["exact_wall_seconds"],
        "phase_cumulative_seconds": payload["phase_cumulative_seconds"],
        "group_cumulative_seconds": payload["group_cumulative_seconds"],
        "result_digest": payload["result_digest"],
    }, indent=2), flush=True)
    return 0 if payload["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
