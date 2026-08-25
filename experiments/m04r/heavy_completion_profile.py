"""Measure exact-completion group and batch amplification for a frozen heavy case."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from market_analogues.adapters import source_from_spec
from market_analogues.certified_packed_search import certified_packed_search_contract
from market_analogues.config import load_config
from market_analogues.episodes import build_episode
from market_analogues.packed_bound_search import (
    PackedBoundQuery, scan_packed_bound_proposals,
)
from market_analogues.representation import represent
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import InstrumentKey, SearchQuery, stable_hash


SCHEMA_VERSION = "m04r-heavy-completion-profile-v1"
OMITTED = {"created_at", "scan_seconds", "metadata_seconds", "result_digest"}


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _render(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(f"""<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>M04R heavy completion profile</title><style>body{{font-family:system-ui;max-width:1100px;margin:2rem auto}}pre{{white-space:pre-wrap}}</style></head><body><h1>M04R heavy completion profile</h1><p>Development performance evidence only; no outcomes or setup labels are accessed.</p><pre>{escape(json.dumps(payload, indent=2, sort_keys=True))}</pre></body></html>""")
    temporary.replace(path)


def _quantiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {name: 0.0 for name in ("minimum", "p50", "p90", "p95", "maximum")}
    array = np.asarray(values, dtype=float)
    return {
        "minimum": float(array.min()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(array.max()),
    }


def _group_summary(rows: list[dict[str, Any]], sparse_cutoff: int) -> dict[str, Any]:
    sparse = [row for row in rows if row["requested_rows"] < sparse_cutoff]
    batch = [row for row in rows if row["requested_rows"] >= sparse_cutoff]
    requested_batch = sum(int(row["requested_rows"]) for row in batch)
    generated_batch = sum(int(row["available_stride_windows"]) for row in batch)
    return {
        "group_count": len(rows),
        "requested_rows": sum(int(row["requested_rows"]) for row in rows),
        "request_count_quantiles": _quantiles([
            int(row["requested_rows"]) for row in rows
        ]),
        "sparse_cutoff": sparse_cutoff,
        "sparse_groups": len(sparse),
        "sparse_requested_rows": sum(int(row["requested_rows"]) for row in sparse),
        "batch_groups": len(batch),
        "batch_requested_rows": requested_batch,
        "batch_generated_windows": generated_batch,
        "batch_window_amplification": (
            generated_batch / requested_batch if requested_batch else 0.0
        ),
        "batch_extra_windows": generated_batch - requested_batch,
        "largest_groups": sorted(
            rows, key=lambda row: (-int(row["requested_rows"]), row["symbol"]),
        )[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--query-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--block-rows", type=int, default=2_048)
    parser.add_argument("--sparse-cutoff", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1 or args.block_rows < 1 or args.sparse_cutoff < 1:
        raise ValueError("workers, block rows and sparse cutoff must be positive")
    config = load_config(args.config)
    build = json.loads((args.full_root / "packed-bound-full.json").read_text())
    authority_path = (
        config.artifact_dir / "gate12" / "authorities" / "nasdaq"
        / "cases" / f"{args.query_id}.json"
    )
    baseline_path = args.matrix_root / "cases" / f"{args.query_id}.json"
    authority = json.loads(authority_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    source = source_from_spec(config.datasets["nasdaq"])
    metadata = authority["query"]
    query = build_episode(
        source, InstrumentKey("nasdaq", str(metadata["symbol"])),
        str(metadata["cutoff"]), int(metadata["lookback"]),
        str(metadata["representation_version"]),
    )
    request = SearchQuery(
        query.key, ("nasdaq",), ("A", "B"), 20, False, True, 3, 60,
    )
    last_round = baseline["certificate"]["rounds"][-1]
    frontier_rows = int(last_round["frontier_rows"])
    threshold = float(baseline["certificate"]["stop_threshold"])
    packed_query = PackedBoundQuery(
        query.key.id, query.key.instrument.source_symbol,
        int(pd.Timestamp(query.bars.timestamp.iloc[0]).value),
        int(latest_eligible_cutoff(query, request.minimum_history_gap_bars).value),
        represent(query), request.quality_tiers,
    )
    print(
        f"scan {args.query_id} composite={frontier_rows + 1}", flush=True,
    )
    scan_started = perf_counter()
    proposal = scan_packed_bound_proposals(
        args.full_root / "store", str(build["generation_id"]), packed_query,
        route_quotas={"composite": frontier_rows + 1},
        block_rows=args.block_rows, verify_content=True,
    )
    scan_seconds = perf_counter() - scan_started
    selected = [
        row for row in proposal.candidates[:frontier_rows]
        if row.lower_bound <= threshold
    ]
    expected_exact = int(baseline["certificate"]["exact_evaluated"])
    if len(selected) != expected_exact:
        raise ValueError(
            f"profile frontier selects {len(selected)} rows; expected {expected_exact}"
        )
    grouped: dict[str, list[Any]] = {}
    for row in selected:
        grouped.setdefault(row.symbol, []).append(row)
    benchmark_prefix = proposal.query_episode_id == args.query_id
    maximum_cutoff = pd.Timestamp(
        json.loads(
            (
                args.full_root / "store" / "generations"
                / str(build["generation_id"]) / "manifest.json"
            ).read_text()
        )["provenance"]["benchmark_prefix"]["requested_cutoff"]
    )
    latest = min(
        latest_eligible_cutoff(query, request.minimum_history_gap_bars),
        maximum_cutoff,
    )

    def one(item: tuple[str, list[Any]]) -> dict[str, Any]:
        symbol, proposals = item
        frame = source.load(InstrumentKey("nasdaq", symbol))
        eligible_rows = int((frame.timestamp <= latest).sum())
        available = max(
            (eligible_rows - query.key.lookback) // 5 + 1, 0,
        )
        return {
            "symbol": symbol,
            "requested_rows": len(proposals),
            "eligible_source_rows": eligible_rows,
            "available_stride_windows": available,
        }

    print(f"load metadata groups={len(grouped)}", flush=True)
    metadata_started = perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(one, sorted(grouped.items())))
    metadata_seconds = perf_counter() - metadata_started
    summary = _group_summary(rows, args.sparse_cutoff)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_digest": certified_packed_search_contract()["digest"],
        "generation_id": build["generation_id"],
        "full_build_evidence_digest": build["result_digest"],
        "query_episode_id": args.query_id,
        "authority_digest": authority["authority_digest"],
        "baseline_case_digest": baseline["result_digest"],
        "baseline_certificate_digest": baseline["certificate_digest"],
        "baseline_seconds": baseline["seconds"],
        "baseline_exact_evaluated": expected_exact,
        "threshold": threshold,
        "frontier_rows": frontier_rows,
        "first_omitted_bound": last_round["next_lower_bound"],
        "scan_candidate_digest": proposal.candidate_digest,
        "scan_seconds": scan_seconds,
        "metadata_seconds": metadata_seconds,
        "source_metadata_workers": args.workers,
        "group_summary": summary,
        "query_binding_valid": benchmark_prefix,
        "selected_rows_equal_baseline_exact": len(selected) == expected_exact,
        "real_forward_outcomes_accessed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    deterministic = {
        key: value for key, value in payload.items() if key not in OMITTED
    }
    payload["result_digest"] = stable_hash(deterministic)
    _write(args.output, payload)
    _render(args.output.with_suffix(".html"), payload)
    print(json.dumps({
        "query_episode_id": args.query_id,
        "scan_seconds": scan_seconds,
        "metadata_seconds": metadata_seconds,
        **summary,
        "result_digest": payload["result_digest"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
