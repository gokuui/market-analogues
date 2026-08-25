"""Run complete certified query groups concurrently across spawned processes."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
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


SCHEMA = "m04r-parallel-certified-batch-poc-v1"
OMITTED = {"created_at", "elapsed_seconds", "peak_rss_mb", "result_digest"}
GROUP_OMITTED = {
    "elapsed_seconds", "proposal_seconds", "exact_total_seconds", "peak_rss_mb",
}
CASE_OMITTED = {"exact_seconds", "peak_rss_mb"}


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
        "<title>M04R-09 parallel certified batch POC</title><style>"
        "body{font-family:system-ui;max-width:1200px;margin:2rem auto}"
        "pre{white-space:pre-wrap}.pass{color:#075}.fail{color:#a20}"
        ".warn{color:#960}</style></head><body>"
        "<h1>M04R-09 parallel certified batch: "
        f"<span class=\"{css}\">{status}</span></h1>"
        "<p>Outcome-blind, cost-balanced query groups perform proposal scanning "
        "and exact certified completion inside isolated spawned processes. "
        "Every match and complete certificate must equal the frozen batch "
        "control.</p><pre>"
        f"{escape(json.dumps(payload, indent=2, sort_keys=True))}"
        "</pre></body></html>"
    )
    temporary.replace(path)


def evidence_digest(payload: dict[str, Any]) -> str:
    deterministic = {key: value for key, value in payload.items() if key not in OMITTED}
    deterministic["groups"] = []
    for group in payload["groups"]:
        stable_group = {
            key: value for key, value in group.items() if key not in GROUP_OMITTED
        }
        stable_group["cases"] = [
            {key: value for key, value in case.items() if key not in CASE_OMITTED}
            for case in group["cases"]
        ]
        deterministic["groups"].append(stable_group)
    deterministic["cases"] = [
        {key: value for key, value in case.items() if key not in CASE_OMITTED}
        for case in payload["cases"]
    ]
    return stable_hash(deterministic)


def _balanced_groups(
    cases: list[dict[str, Any]], baseline_by_id: dict[str, dict[str, Any]],
    workers: int, proposal_seconds_per_query: float,
) -> list[tuple[dict[str, Any], ...]]:
    weighted = sorted(
        cases,
        key=lambda case: (
            -(
                float(baseline_by_id[str(case["episode_id"])]["exact_seconds"])
                + proposal_seconds_per_query
            ),
            str(case["case_id"]),
        ),
    )
    groups: list[list[dict[str, Any]]] = [[] for _ in range(workers)]
    loads = [0.0] * workers
    for case in weighted:
        index = min(
            range(workers), key=lambda value: (loads[value], len(groups[value]), value),
        )
        groups[index].append(case)
        loads[index] += (
            float(baseline_by_id[str(case["episode_id"])]["exact_seconds"])
            + proposal_seconds_per_query
        )
    return [tuple(group) for group in groups if group]


def _run_group(
    config_path: str,
    full_root: str,
    generation_id: str,
    cases: tuple[dict[str, Any], ...],
    baselines: dict[str, dict[str, Any]],
    maximum_frontier_rows: int,
    initial_frontier_rows: int,
    block_rows: int,
    numba_threads: int,
    exact_workers: int,
) -> dict[str, Any]:
    started = perf_counter()
    numba.set_num_threads(numba_threads)
    config = load_config(Path(config_path))
    source = source_from_spec(config.datasets["nasdaq"])
    episodes = []
    requests = []
    packed_queries = []
    for case in cases:
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

    proposal = scan_packed_bound_proposals_many(
        Path(full_root) / "store", generation_id, packed_queries,
        route_quotas={"composite": maximum_frontier_rows + 1},
        block_rows=block_rows, verify_content=False,
    )
    result_cases = []
    exact_total_seconds = 0.0
    for case, episode, request, query_proposal in zip(
        cases, episodes, requests, proposal.reports,
    ):
        query_id = str(case["episode_id"])
        baseline = baselines[query_id]
        result = certified_packed_search(
            episode, source, request, Path(full_root) / "store", generation_id,
            store_dataset_id="nasdaq",
            initial_frontier_rows=initial_frontier_rows,
            maximum_frontier_rows=maximum_frontier_rows,
            seed_rows=512, block_rows=block_rows, workers=exact_workers,
            sparse_cutoff=8, verify_content=False,
            requested_positions=True, vector_lower_bounds=True,
            deferred_alignments=True, precomputed_proposal=query_proposal,
        )
        matches = [_match(value) for value in result.matches]
        certificate = json.loads(json.dumps(asdict(result.certificate)))
        exact_seconds = float(certificate.pop("elapsed_seconds"))
        exact_total_seconds += exact_seconds
        certificate_digest = str(certificate["result_digest"])
        gates = {
            "proposal_result_digest_equal": (
                query_proposal.result_digest == baseline["proposal_result_digest"]
            ),
            "matches_equal": matches == baseline["matches"],
            "certificate_equal": certificate == baseline["certificate"],
            "certificate_digest_equal": (
                certificate_digest == baseline["certificate_digest"]
            ),
            "certificate_reconstructed": (
                certificate_digest == _certificate_digest({
                    "certificate": certificate, "matches": matches,
                })
            ),
        }
        result_cases.append({
            "registry_case_id": case["case_id"],
            "query_episode_id": query_id,
            "proposal_result_digest": query_proposal.result_digest,
            "matches": matches,
            "certificate": certificate,
            "certificate_digest": certificate_digest,
            "exact_seconds": exact_seconds,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "gates": gates,
            "gate_passed": all(gates.values()),
        })
        print(
            f"[{case['case_id']}] exact "
            f"{'PASS' if result_cases[-1]['gate_passed'] else 'FAIL'} "
            f"seconds={exact_seconds:.2f}", flush=True,
        )
    return {
        "query_episode_ids": [str(case["episode_id"]) for case in cases],
        "numba_threads": numba.get_num_threads(),
        "exact_workers": exact_workers,
        "proposal_result_digest": proposal.result_digest,
        "proposal_seconds": proposal.elapsed_seconds,
        "exact_total_seconds": exact_total_seconds,
        "elapsed_seconds": perf_counter() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "cases": result_cases,
        "gate_passed": all(case["gate_passed"] for case in result_cases),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--numba-threads", type=int, default=1)
    parser.add_argument("--exact-workers", type=int, default=1)
    parser.add_argument("--block-rows", type=int, default=4_096)
    parser.add_argument("--initial-frontier-rows", type=int, default=16_384)
    parser.add_argument("--maximum-frontier-rows", type=int, default=32_768)
    args = parser.parse_args()
    if not all((
        args.processes > 0,
        args.numba_threads > 0,
        args.exact_workers > 0,
        args.block_rows > 0,
        args.maximum_frontier_rows >= args.initial_frontier_rows >= 512,
    )):
        raise ValueError("parallel certified controls are invalid")

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
        set(baseline_by_id) == set(expected_ids),
        batch.get("real_forward_outcomes_accessed") is False,
    )):
        raise ValueError("certified batch baseline differs")

    proposal_seconds_per_query = float(batch["shared_scan_seconds"]) / len(expected_ids)
    groups = _balanced_groups(
        registry_cases, baseline_by_id, args.processes, proposal_seconds_per_query,
    )
    estimated_loads = [sum(
        float(baseline_by_id[str(case["episode_id"])]["exact_seconds"])
        + proposal_seconds_per_query
        for case in group
    ) for group in groups]
    print(
        f"[parallel-batch] start groups={len(groups)} "
        f"sizes={[len(group) for group in groups]} "
        f"estimated={[round(value, 2) for value in estimated_loads]}",
        flush=True,
    )
    started = perf_counter()
    group_results = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(groups), mp_context=context) as executor:
        futures = [executor.submit(
            _run_group, str(args.config), str(args.full_root), generation_id,
            group,
            {str(case["episode_id"]): baseline_by_id[str(case["episode_id"])]
             for case in group},
            args.maximum_frontier_rows, args.initial_frontier_rows,
            args.block_rows, args.numba_threads, args.exact_workers,
        ) for group in groups]
        for future in as_completed(futures):
            group = future.result()
            group_results.append(group)
            print(
                f"[parallel-batch] group "
                f"{'PASS' if group['gate_passed'] else 'FAIL'} "
                f"ids={len(group['query_episode_ids'])} "
                f"proposal={group['proposal_seconds']:.2f}s "
                f"exact={group['exact_total_seconds']:.2f}s "
                f"wall={group['elapsed_seconds']:.2f}s", flush=True,
            )
    elapsed_seconds = perf_counter() - started

    query_order = {query_id: index for index, query_id in enumerate(expected_ids)}
    group_results.sort(key=lambda group: min(
        query_order[query_id] for query_id in group["query_episode_ids"]
    ))
    parallel_by_id = {
        str(case["query_episode_id"]): case
        for group in group_results for case in group["cases"]
    }
    ordered_cases = [parallel_by_id[query_id] for query_id in expected_ids]
    gates = {
        "all_24_cases_returned": set(parallel_by_id) == set(expected_ids),
        "all_case_gates_passed": all(case["gate_passed"] for case in ordered_cases),
        "all_groups_passed": all(group["gate_passed"] for group in group_results),
        "worker_controls_equal": all(
            group["numba_threads"] == args.numba_threads
            and group["exact_workers"] == args.exact_workers
            for group in group_results
        ),
        "real_forward_outcomes_excluded": True,
    }
    correctness_passed = all(gates.values())
    baseline_seconds = float(batch["total_seconds"])
    payload = {
        "schema_version": SCHEMA,
        "registry_digest": registry_digest,
        "generation_id": generation_id,
        "baseline_batch_digest": batch["result_digest"],
        "controls": {
            "processes": args.processes,
            "numba_threads_per_process": args.numba_threads,
            "exact_workers_per_process": args.exact_workers,
            "block_rows": args.block_rows,
            "initial_frontier_rows": args.initial_frontier_rows,
            "maximum_frontier_rows": args.maximum_frontier_rows,
            "group_assignment": "LPT by prior exact seconds plus equal proposal cost",
            "process_start_method": "spawn",
        },
        "estimated_group_loads": estimated_loads,
        "groups": group_results,
        "cases": ordered_cases,
        "baseline_serial_batch_seconds": baseline_seconds,
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
    _write(args.output_root / "parallel-certified-batch-poc.json", payload)
    _render(args.output_root / "parallel-certified-batch-poc.html", payload)
    print(
        f"[parallel-batch] {'PASS' if correctness_passed else 'FAIL'} "
        f"elapsed={elapsed_seconds:.2f}s speedup={payload['speedup']:.4f}x",
        flush=True,
    )
    return 0 if correctness_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
