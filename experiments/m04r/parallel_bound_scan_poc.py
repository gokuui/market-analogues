"""POC query-group process parallelism for the certified bound scan."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from html import escape
import json
import multiprocessing
from pathlib import Path
import resource
from time import perf_counter
from typing import Any

import numba
import pandas as pd

from ann_savings_ceiling import evidence_digest as ceiling_digest
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


SCHEMA = "m04r-parallel-bound-scan-poc-v1"
OMITTED = {"created_at", "elapsed_seconds", "peak_rss_mb", "result_digest"}


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _render(path: Path, payload: dict[str, Any]) -> None:
    if not payload["correctness_passed"]:
        status, css = "FAIL", "fail"
    elif payload["performance_improved"]:
        status, css = "PASS — FASTER", "pass"
    else:
        status, css = "PASS — NOT FASTER", "warn"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>M04R-09 parallel bound scan POC</title><style>"
        "body{font-family:system-ui;max-width:1200px;margin:2rem auto}"
        "pre{white-space:pre-wrap}.pass{color:#075}.fail{color:#a20}"
        ".warn{color:#960}</style></head><body>"
        "<h1>M04R-09 parallel bound scan: "
        f"<span class=\"{css}\">{status}</span></h1>"
        "<p>Outcome-blind query-group process parallelism. Repeated physical "
        "reads are permitted; every per-query proposal digest and forced-row "
        "count must equal the frozen serial evidence.</p><pre>"
        f"{escape(json.dumps(payload, indent=2, sort_keys=True))}"
        "</pre></body></html>"
    )
    temporary.replace(path)


def evidence_digest(payload: dict[str, Any]) -> str:
    return stable_hash({key: value for key, value in payload.items() if key not in OMITTED})


def _scan_group(
    store_root: str,
    generation_id: str,
    queries: tuple[PackedBoundQuery, ...],
    thresholds: dict[str, float],
    maximum_frontier_rows: int,
    block_rows: int,
    numba_threads: int,
) -> dict[str, Any]:
    numba.set_num_threads(numba_threads)
    report = scan_packed_bound_proposals_many(
        Path(store_root), generation_id, queries,
        route_quotas={"composite": maximum_frontier_rows + 1},
        block_rows=block_rows, verify_content=False,
    )
    cases = []
    for query_report in report.reports:
        threshold = thresholds[query_report.query_episode_id]
        cases.append({
            "query_episode_id": query_report.query_episode_id,
            "proposal_result_digest": query_report.result_digest,
            "candidate_digest": query_report.candidate_digest,
            "eligible_rows": query_report.eligible_rows,
            "forced_rows": sum(
                candidate.lower_bound <= threshold
                for candidate in query_report.candidates
            ),
        })
    return {
        "query_episode_ids": list(report.query_episode_ids),
        "elapsed_seconds": report.elapsed_seconds,
        "peak_rss_mb": report.peak_rss_mb,
        "physical_rows_scanned": report.physical_rows_scanned,
        "logical_rows_evaluated": report.logical_rows_evaluated,
        "numba_threads": numba.get_num_threads(),
        "cases": cases,
    }


def _round_robin_groups(
    queries: list[PackedBoundQuery], workers: int,
) -> list[tuple[PackedBoundQuery, ...]]:
    groups = [[] for _ in range(workers)]
    for index, query in enumerate(queries):
        groups[index % workers].append(query)
    return [tuple(group) for group in groups if group]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--ceiling", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--numba-threads", type=int, default=2)
    parser.add_argument("--block-rows", type=int, default=4_096)
    parser.add_argument("--maximum-frontier-rows", type=int, default=32_768)
    args = parser.parse_args()
    if not all((
        args.processes > 0,
        args.numba_threads > 0,
        args.block_rows > 0,
        args.maximum_frontier_rows >= 512,
    )):
        raise ValueError("parallel scan controls are invalid")

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
    if args.processes > len(expected_ids):
        raise ValueError("process count exceeds query count")

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
    loaded = load_packed_generation(
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
        set(baseline_by_id) == set(expected_ids),
    )):
        raise ValueError("certified batch baseline differs")

    ceiling = json.loads(args.ceiling.read_text())
    if not all((
        ceiling.get("gate_passed") is True,
        ceiling.get("registry_digest") == registry_digest,
        ceiling.get("generation_id") == generation_id,
        ceiling.get("result_digest") == ceiling_digest(ceiling),
        len(ceiling.get("cases", [])) == 24,
    )):
        raise ValueError("ANN ceiling evidence differs")
    ceiling_by_id = {
        str(case["query_episode_id"]): case for case in ceiling["cases"]
    }
    thresholds = {
        query_id: float(baseline_by_id[query_id]["certificate"]["stop_threshold"])
        for query_id in expected_ids
    }

    queries = []
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
        queries.append(PackedBoundQuery(
            episode.key.id, episode.key.instrument.source_symbol,
            int(episode.bars.timestamp.iloc[0].value),
            int(latest_eligible_cutoff(episode, 60).value),
            represent(episode), request.quality_tiers,
        ))
    groups = _round_robin_groups(queries, args.processes)
    print(
        f"[parallel-scan] start processes={len(groups)} "
        f"numba_threads={args.numba_threads} group_sizes={[len(group) for group in groups]}",
        flush=True,
    )
    started = perf_counter()
    group_results = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(groups), mp_context=context) as executor:
        futures = [executor.submit(
            _scan_group, str(args.full_root / "store"), generation_id,
            group, thresholds, args.maximum_frontier_rows,
            args.block_rows, args.numba_threads,
        ) for group in groups]
        for future in as_completed(futures):
            result = future.result()
            group_results.append(result)
            print(
                f"[parallel-scan] group PASS ids={len(result['query_episode_ids'])} "
                f"seconds={result['elapsed_seconds']:.2f} "
                f"rss={result['peak_rss_mb']:.2f} MiB", flush=True,
            )
    elapsed_seconds = perf_counter() - started

    parallel_by_id = {
        str(case["query_episode_id"]): case
        for group in group_results for case in group["cases"]
    }
    cases = []
    for registry_case in registry_cases:
        query_id = str(registry_case["episode_id"])
        parallel = parallel_by_id.get(query_id, {})
        baseline = baseline_by_id[query_id]
        ceiling_case = ceiling_by_id[query_id]
        gates = {
            "proposal_result_digest_equal": (
                parallel.get("proposal_result_digest")
                == baseline["proposal_result_digest"]
            ),
            "forced_rows_equal": (
                parallel.get("forced_rows")
                == ceiling_case["unavoidable_exact_rows"]
            ),
            "eligible_rows_equal": (
                parallel.get("eligible_rows")
                == baseline["certificate"]["eligible_candidates"]
            ),
        }
        cases.append({
            "registry_case_id": registry_case["case_id"],
            "query_episode_id": query_id,
            "parallel_proposal_result_digest": parallel.get("proposal_result_digest"),
            "baseline_proposal_result_digest": baseline["proposal_result_digest"],
            "forced_rows": parallel.get("forced_rows"),
            "eligible_rows": parallel.get("eligible_rows"),
            "gates": gates,
            "gate_passed": all(gates.values()),
        })

    baseline_seconds = float(ceiling["shared_scan_seconds"])
    gates = {
        "all_24_cases_returned": set(parallel_by_id) == set(expected_ids),
        "all_case_gates_passed": all(case["gate_passed"] for case in cases),
        "all_workers_used_requested_numba_threads": all(
            group["numba_threads"] == args.numba_threads for group in group_results
        ),
        "physical_row_accounting": sum(
            group["physical_rows_scanned"] for group in group_results
        ) == len(groups) * (len(loaded.rows) + len(loaded.overflow)),
        "real_forward_outcomes_excluded": True,
    }
    correctness_passed = all(gates.values())
    payload = {
        "schema_version": SCHEMA,
        "registry_digest": registry_digest,
        "generation_id": generation_id,
        "baseline_ceiling_digest": ceiling["result_digest"],
        "controls": {
            "processes": args.processes,
            "numba_threads_per_process": args.numba_threads,
            "total_requested_compute_threads": (
                args.processes * args.numba_threads
            ),
            "block_rows": args.block_rows,
            "maximum_frontier_rows": args.maximum_frontier_rows,
            "group_assignment": "query input order round-robin",
            "process_start_method": "spawn",
        },
        "group_results": group_results,
        "cases": cases,
        "baseline_serial_seconds": baseline_seconds,
        "elapsed_seconds": elapsed_seconds,
        "speedup": baseline_seconds / elapsed_seconds,
        "elapsed_reduction_fraction": 1.0 - elapsed_seconds / baseline_seconds,
        "performance_improved": elapsed_seconds < baseline_seconds,
        "peak_rss_mb": max(
            (group["peak_rss_mb"] for group in group_results), default=0.0
        ),
        "sum_worker_peak_rss_mb": sum(
            group["peak_rss_mb"] for group in group_results
        ),
        "memory_policy": (
            "measured, not a hard acceptance gate; allocation/process failure "
            "still fails the experiment"
        ),
        "gates": gates,
        "correctness_passed": correctness_passed,
        "gate_passed": correctness_passed,
        "real_forward_outcomes_accessed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    payload["result_digest"] = evidence_digest(payload)
    _write(args.output_root / "parallel-bound-scan-poc.json", payload)
    _render(args.output_root / "parallel-bound-scan-poc.html", payload)
    print(
        f"[parallel-scan] {'PASS' if correctness_passed else 'FAIL'} "
        f"elapsed={elapsed_seconds:.2f}s speedup={payload['speedup']:.4f}x",
        flush=True,
    )
    return 0 if correctness_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
