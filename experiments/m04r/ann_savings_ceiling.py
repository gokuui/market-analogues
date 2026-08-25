"""Compute the maximum exact work that any perfect M04R-09 proposer could save.

The audit uses the frozen final certified threshold for each query and the
complete sorted composite lower-bound prefix.  Every candidate whose admissible
lower bound is at most that threshold must still be exact-scored, regardless of
how an ANN or other proposer seeds the search.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import resource
from typing import Any

import pandas as pd

from market_analogues.adapters import source_from_spec
from market_analogues.config import load_config
from market_analogues.episodes import build_episode
from market_analogues.m04r_batch_registry import validate_m04r_batch_registry
from market_analogues.m04r_full_pack_verification import (
    EVIDENCE_OMITTED as FULL_BUILD_OMITTED,
)
from market_analogues.packed_bound_search import (
    PackedBoundQuery,
    scan_packed_bound_proposals_many,
)
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.representation import represent
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import InstrumentKey, SearchQuery, stable_hash


SCHEMA = "m04r-ann-savings-ceiling-v2"
OMITTED = {"created_at", "shared_scan_seconds", "peak_rss_mb", "result_digest"}


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _render(path: Path, payload: dict[str, Any]) -> None:
    status = "PASS" if payload["gate_passed"] else "FAIL"
    css = "pass" if payload["gate_passed"] else "fail"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>M04R-09 ANN savings ceiling</title><style>"
        "body{font-family:system-ui;max-width:1200px;margin:2rem auto}"
        "pre{white-space:pre-wrap}.pass{color:#075}.fail{color:#a20}"
        "</style></head><body><h1>M04R-09 ANN savings ceiling: "
        f"<span class=\"{css}\">{status}</span></h1>"
        "<p>Outcome-blind audit of the maximum exact work removable by an "
        "ideal zero-cost proposer. Rows whose certified lower bound is no "
        "greater than final τ remain mandatory.</p><pre>"
        f"{escape(json.dumps(payload, indent=2, sort_keys=True))}"
        "</pre></body></html>"
    )
    temporary.replace(path)


def evidence_digest(payload: dict[str, Any]) -> str:
    return stable_hash({key: value for key, value in payload.items() if key not in OMITTED})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--block-rows", type=int, default=4_096)
    parser.add_argument("--maximum-frontier-rows", type=int, default=32_768)
    parser.add_argument("--material-fraction", type=float, default=0.10)
    args = parser.parse_args()
    if not all((
        args.block_rows > 0,
        args.maximum_frontier_rows >= 512,
        0 < args.material_fraction < 1,
    )):
        raise ValueError("ANN ceiling controls are invalid")

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
    build_content = {
        key: value for key, value in build.items() if key not in FULL_BUILD_OMITTED
    }
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

    batch = json.loads((args.batch_root / "certified-batch-gate.json").read_text())
    baseline_by_id = {
        str(case["query_episode_id"]): case for case in batch.get("cases", [])
    }
    if not all((
        batch.get("gate_passed") is True,
        batch.get("registry_digest") == registry_digest,
        batch.get("generation_id") == generation_id,
        batch.get("completed_query_episode_ids") == expected_ids,
        batch.get("real_forward_outcomes_accessed") is False,
        set(baseline_by_id) == set(expected_ids),
    )):
        raise ValueError("certified batch baseline differs")

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
        packed_queries.append(PackedBoundQuery(
            episode.key.id, episode.key.instrument.source_symbol,
            int(episode.bars.timestamp.iloc[0].value),
            int(latest_eligible_cutoff(episode, 60).value),
            represent(episode), request.quality_tiers,
        ))

    print(f"[shared-scan] start {len(packed_queries)} queries", flush=True)
    shared = scan_packed_bound_proposals_many(
        args.full_root / "store", generation_id, packed_queries,
        route_quotas={"composite": args.maximum_frontier_rows + 1},
        block_rows=args.block_rows, verify_content=False,
    )
    print(
        f"[shared-scan] PASS {shared.elapsed_seconds:.2f}s "
        f"rss={shared.peak_rss_mb:.2f} MiB", flush=True,
    )
    if shared.query_episode_ids != tuple(expected_ids):
        raise ValueError("shared proposal query order differs")

    cases: list[dict[str, Any]] = []
    for registry_case, proposal in zip(registry_cases, shared.reports):
        query_id = str(registry_case["episode_id"])
        baseline = baseline_by_id[query_id]
        certificate = baseline["certificate"]
        threshold = float(certificate["stop_threshold"])
        forced_rows = sum(
            candidate.lower_bound <= threshold for candidate in proposal.candidates
        )
        exact_rows = int(certificate["exact_evaluated"])
        next_lower = certificate["next_lower_bound"]
        excess_rows = exact_rows - forced_rows
        exact_seconds = float(baseline["exact_seconds"])
        optimistic_seconds = (
            exact_seconds * excess_rows / exact_rows if exact_rows else 0.0
        )
        gates = {
            "proposal_covers_certificate_frontier": (
                len(proposal.candidates)
                >= min(int(certificate["eligible_candidates"]), args.maximum_frontier_rows + 1)
            ),
            "certified_next_bound_strict": (
                next_lower is not None and float(next_lower) > threshold
            ),
            "forced_rows_include_final_top_k": forced_rows >= 20,
            "current_exact_rows_cover_forced_rows": excess_rows >= 0,
            "seed_excess_no_more_than_492": excess_rows <= 492,
        }
        row = {
            "registry_case_id": registry_case["case_id"],
            "query_episode_id": query_id,
            "symbol": registry_case["symbol"],
            "cutoff_role": registry_case["cutoff_role"],
            "stop_threshold": threshold,
            "current_exact_rows": exact_rows,
            "unavoidable_exact_rows": forced_rows,
            "perfect_proposer_maximum_saved_rows": excess_rows,
            "current_exact_seconds": exact_seconds,
            "optimistic_linear_saved_seconds": optimistic_seconds,
            "gates": gates,
            "gate_passed": all(gates.values()),
        }
        cases.append(row)
        print(
            f"[{registry_case['case_id']}] forced={forced_rows} "
            f"current={exact_rows} maximum_saved={excess_rows}", flush=True,
        )

    current_exact_rows = sum(row["current_exact_rows"] for row in cases)
    forced_exact_rows = sum(row["unavoidable_exact_rows"] for row in cases)
    maximum_saved_rows = current_exact_rows - forced_exact_rows
    optimistic_saved_seconds = sum(
        row["optimistic_linear_saved_seconds"] for row in cases
    )
    current_total_seconds = float(batch["total_seconds"])
    optimistic_fraction = optimistic_saved_seconds / current_total_seconds
    gates = {
        "all_24_cases_audited": len(cases) == 24,
        "all_case_gates_passed": all(row["gate_passed"] for row in cases),
        "real_forward_outcomes_excluded": True,
    }
    payload = {
        "schema_version": SCHEMA,
        "registry_digest": registry_digest,
        "generation_id": generation_id,
        "baseline_batch_digest": batch["result_digest"],
        "controls": {
            "block_rows": args.block_rows,
            "maximum_frontier_rows": args.maximum_frontier_rows,
            "maximum_proposal_rows": args.maximum_frontier_rows + 1,
            "material_fraction": args.material_fraction,
        },
        "shared_scan_seconds": shared.elapsed_seconds,
        "shared_report_digest": shared.result_digest,
        "current_exact_rows": current_exact_rows,
        "unavoidable_exact_rows": forced_exact_rows,
        "perfect_proposer_maximum_saved_rows": maximum_saved_rows,
        "perfect_proposer_maximum_saved_row_fraction": (
            maximum_saved_rows / current_exact_rows if current_exact_rows else 0.0
        ),
        "optimistic_linear_saved_seconds": optimistic_saved_seconds,
        "current_complete_batch_seconds": current_total_seconds,
        "optimistic_complete_batch_reduction_fraction": optimistic_fraction,
        "material_improvement_possible_under_optimistic_linear_model": (
            optimistic_fraction >= args.material_fraction
        ),
        "interpretation": (
            "The optimistic time projection assumes exact time scales linearly "
            "with removed rows and assigns zero ANN build/query overhead. It is "
            "an upper-bound screening model, not a latency measurement."
        ),
        "cases": cases,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "memory_policy": (
            "measured, not a hard acceptance gate; allocation/process failure "
            "still fails the experiment"
        ),
        "gates": gates,
        "gate_passed": all(gates.values()),
        "real_forward_outcomes_accessed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    payload["result_digest"] = evidence_digest(payload)
    _write(args.output_root / "ann-savings-ceiling.json", payload)
    _render(args.output_root / "ann-savings-ceiling.html", payload)
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
