"""Measure whether a smaller deterministic seed improves certified completion.

This is a development-only M04R-09 experiment.  It never reads forward
outcomes and retains the packed lower-bound scan as the correctness mechanism.
"""

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

from certified_batch_scalar_controls import _match
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
    PackedBoundQuery,
    scan_packed_bound_proposals_many,
)
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.representation import represent
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import InstrumentKey, SearchQuery, stable_hash


SCHEMA = "m04r-seed-rows-poc-v2"
DEFAULT_CASE_IDS = (
    "nasdaq-IHRT-current-252",
    "nasdaq-UONEK-current-252",
    "nasdaq-AOUT-historical-252",
)
OMITTED = {
    "created_at", "shared_scan_seconds", "peak_rss_mb", "result_digest",
}
CASE_OMITTED = {"elapsed_seconds", "peak_rss_mb"}


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
        "<title>M04R-09 deterministic seed POC</title><style>"
        "body{font-family:system-ui;max-width:1200px;margin:2rem auto}"
        "pre{white-space:pre-wrap}.pass{color:#075}.fail{color:#a20}"
        "</style></head><body><h1>M04R-09 deterministic seed POC: "
        f"<span class=\"{css}\">{status}</span></h1>"
        "<p>Outcome-blind development comparison. Smaller seeds are accepted "
        "only when final matches are unchanged and the replacement stopping "
        "certificate reconstructs exactly.</p><pre>"
        f"{escape(json.dumps(payload, indent=2, sort_keys=True))}"
        "</pre></body></html>"
    )
    temporary.replace(path)


def _digest(payload: dict[str, Any]) -> str:
    deterministic = {key: value for key, value in payload.items() if key not in OMITTED}
    deterministic["cases"] = [
        {key: value for key, value in case.items() if key not in CASE_OMITTED}
        for case in payload["cases"]
    ]
    return stable_hash(deterministic)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case-ids", nargs="+", default=list(DEFAULT_CASE_IDS))
    parser.add_argument("--seed-rows", nargs="+", type=int, default=[128])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--block-rows", type=int, default=4_096)
    parser.add_argument("--initial-frontier-rows", type=int, default=16_384)
    parser.add_argument("--maximum-frontier-rows", type=int, default=32_768)
    args = parser.parse_args()
    if not all(
        args.workers > 0
        and args.block_rows > 0
        and 20 <= value <= args.initial_frontier_rows
        for value in args.seed_rows
    ):
        raise ValueError("seed POC controls are invalid")
    if len(set(args.seed_rows)) != len(args.seed_rows):
        raise ValueError("seed row variants must be unique")

    config = load_config(args.config)
    source = source_from_spec(config.datasets["nasdaq"])
    quality = pd.read_parquet(config.artifact_dir / "quality" / "nasdaq.parquet")
    failures = validate_m04r_batch_registry(
        source, args.registry, quality, config.artifact_dir / "oracles" / "nasdaq",
    )
    if failures:
        raise ValueError(f"batch registry invalid: {failures}")
    registry = json.loads(args.registry.read_text())
    registry_digest = str(registry["registry_digest"])
    by_case_id = {str(case["case_id"]): case for case in registry["cases_data"]}
    if len(set(args.case_ids)) != len(args.case_ids):
        raise ValueError("case IDs must be unique")
    unknown = set(args.case_ids).difference(by_case_id)
    if unknown:
        raise ValueError(f"unknown case IDs: {sorted(unknown)}")
    selected = [by_case_id[value] for value in args.case_ids]

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
        batch.get("real_forward_outcomes_accessed") is False,
        all(str(case["episode_id"]) in baseline_by_id for case in selected),
    )):
        raise ValueError("certified batch baseline differs")

    episodes = []
    requests = []
    packed_queries = []
    for case in selected:
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

    print(f"[shared-scan] start {len(selected)} queries", flush=True)
    shared = scan_packed_bound_proposals_many(
        args.full_root / "store", generation_id, packed_queries,
        route_quotas={"composite": args.maximum_frontier_rows + 1},
        block_rows=args.block_rows, verify_content=False,
    )
    print(
        f"[shared-scan] PASS {shared.elapsed_seconds:.2f}s "
        f"rss={shared.peak_rss_mb:.2f} MiB", flush=True,
    )

    result_cases: list[dict[str, Any]] = []
    for case, episode, request, proposal in zip(
        selected, episodes, requests, shared.reports,
    ):
        baseline = baseline_by_id[str(case["episode_id"])]
        for seed_rows in args.seed_rows:
            print(f"[{case['case_id']}] seed={seed_rows} start", flush=True)
            result = certified_packed_search(
                episode, source, request, args.full_root / "store", generation_id,
                store_dataset_id="nasdaq",
                initial_frontier_rows=args.initial_frontier_rows,
                maximum_frontier_rows=args.maximum_frontier_rows,
                seed_rows=seed_rows, block_rows=args.block_rows,
                workers=args.workers, sparse_cutoff=8, verify_content=False,
                requested_positions=True, vector_lower_bounds=True,
                deferred_alignments=True, precomputed_proposal=proposal,
            )
            matches = [_match(value) for value in result.matches]
            certificate = json.loads(json.dumps(asdict(result.certificate)))
            elapsed_seconds = float(certificate.pop("elapsed_seconds"))
            peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            replacement_digest = str(certificate["result_digest"])
            gates = {
                "matches_equal_baseline": matches == baseline["matches"],
                "certificate_reconstructed": replacement_digest == _certificate_digest({
                    "certificate": certificate, "matches": matches,
                }),
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
            }
            row = {
                "registry_case_id": case["case_id"],
                "query_episode_id": case["episode_id"],
                "symbol": case["symbol"],
                "cutoff_role": case["cutoff_role"],
                "seed_rows": seed_rows,
                "baseline_exact_rows": baseline["certificate"]["exact_evaluated"],
                "exact_rows": certificate["exact_evaluated"],
                "exact_row_reduction": (
                    int(baseline["certificate"]["exact_evaluated"])
                    - int(certificate["exact_evaluated"])
                ),
                "baseline_exact_seconds": baseline["exact_seconds"],
                "elapsed_seconds": elapsed_seconds,
                "peak_rss_mb": peak_rss_mb,
                "memory_policy": (
                    "measured, not a hard acceptance gate; allocation/process "
                    "failure still fails the experiment"
                ),
                "first_threshold": certificate["rounds"][0]["constrained_threshold"],
                "final_threshold": certificate["stop_threshold"],
                "rounds": certificate["rounds"],
                "replacement_certificate_digest": replacement_digest,
                "gates": gates,
                "gate_passed": all(gates.values()),
            }
            result_cases.append(row)
            print(
                f"[{case['case_id']}] seed={seed_rows} "
                f"{'PASS' if row['gate_passed'] else 'FAIL'} "
                f"rows={row['exact_rows']} seconds={elapsed_seconds:.2f}",
                flush=True,
            )

    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    payload = {
        "schema_version": SCHEMA,
        "registry_digest": registry_digest,
        "generation_id": generation_id,
        "baseline_batch_digest": batch["result_digest"],
        "selected_case_ids": list(args.case_ids),
        "seed_rows": list(args.seed_rows),
        "controls": {
            "workers": args.workers,
            "block_rows": args.block_rows,
            "initial_frontier_rows": args.initial_frontier_rows,
            "maximum_frontier_rows": args.maximum_frontier_rows,
            "requested_positions": True,
            "vector_lower_bounds": True,
            "deferred_alignments": True,
            "sorted_joined_iqr_merge": True,
        },
        "shared_scan_seconds": shared.elapsed_seconds,
        "shared_report_digest": shared.result_digest,
        "cases": result_cases,
        "peak_rss_mb": peak_rss_mb,
        "all_matches_equal": all(
            case["gates"]["matches_equal_baseline"] for case in result_cases
        ),
        "gate_passed": bool(result_cases) and all(
            case["gate_passed"] for case in result_cases
        ),
        "real_forward_outcomes_accessed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    payload["result_digest"] = _digest(payload)
    _write(args.output_root / "seed-rows-poc.json", payload)
    _render(args.output_root / "seed-rows-poc.html", payload)
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
