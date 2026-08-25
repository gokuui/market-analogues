"""Resumable selected M04R-09 parallel certified batch gate."""

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

import pandas as pd

from market_analogues.adapters import source_from_spec
from market_analogues.config import load_config
from market_analogues.m04r_batch_registry import validate_m04r_batch_registry
from market_analogues.m04r_certified_search_verification import _certificate_digest
from market_analogues.m04r_full_pack_verification import (
    EVIDENCE_OMITTED as FULL_BUILD_OMITTED,
)
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.types import stable_hash
from parallel_certified_batch_poc import _balanced_groups, _run_group


GROUP_SCHEMA = "m04r-parallel-certified-group-v2"
EVIDENCE_SCHEMA = "m04r-parallel-certified-batch-gate-v1"
GROUP_OMITTED = {
    "created_at", "elapsed_seconds", "proposal_seconds", "exact_total_seconds",
    "peak_rss_mb", "result_digest", "checkpoint_integrity_digest",
}
CASE_OMITTED = {"exact_seconds", "peak_rss_mb"}
EVIDENCE_OMITTED = {
    "created_at", "launch_elapsed_seconds", "peak_rss_mb", "result_digest",
}


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _render(path: Path, payload: dict[str, Any]) -> None:
    status = "PASS" if payload["gate_passed"] else "INCOMPLETE/FAIL"
    css = "pass" if payload["gate_passed"] else "fail"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>M04R-09 resumable parallel certified gate</title><style>"
        "body{font-family:system-ui;max-width:1200px;margin:2rem auto}"
        "pre{white-space:pre-wrap}.pass{color:#075}.fail{color:#a20}"
        "</style></head><body><h1>M04R-09 parallel certified gate: "
        f"<span class=\"{css}\">{status}</span></h1>"
        "<p>Eight deterministic outcome-blind query groups use atomic validated "
        "checkpoints. Every proposal, final match and complete certificate is "
        "bound to the frozen serial control.</p><pre>"
        f"{escape(json.dumps(payload, indent=2, sort_keys=True))}"
        "</pre></body></html>"
    )
    temporary.replace(path)


def group_digest(payload: dict[str, Any]) -> str:
    deterministic = {
        key: value for key, value in payload.items() if key not in GROUP_OMITTED
    }
    deterministic["cases"] = [
        {key: value for key, value in case.items() if key not in CASE_OMITTED}
        for case in payload["cases"]
    ]
    return stable_hash(deterministic)


def checkpoint_integrity_digest(payload: dict[str, Any]) -> str:
    return stable_hash({
        key: value for key, value in payload.items()
        if key not in {"created_at", "checkpoint_integrity_digest"}
    })


def evidence_digest(payload: dict[str, Any]) -> str:
    return stable_hash({
        key: value for key, value in payload.items() if key not in EVIDENCE_OMITTED
    })


def _controls(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "processes": args.processes,
        "numba_threads_per_process": args.numba_threads,
        "exact_workers_per_process": args.exact_workers,
        "block_rows": args.block_rows,
        "initial_frontier_rows": args.initial_frontier_rows,
        "maximum_frontier_rows": args.maximum_frontier_rows,
        "seed_rows": 512,
        "requested_positions": True,
        "vector_lower_bounds": True,
        "deferred_alignments": True,
        "sorted_joined_iqr_merge": True,
        "group_assignment": "LPT by prior exact seconds plus equal proposal cost",
        "process_start_method": "spawn",
    }


def _group_id(
    index: int, query_ids: list[str], registry_digest: str,
    generation_id: str, baseline_digest: str, controls: dict[str, Any],
) -> str:
    return stable_hash({
        "index": index,
        "query_episode_ids": query_ids,
        "registry_digest": registry_digest,
        "generation_id": generation_id,
        "baseline_batch_digest": baseline_digest,
        "controls": controls,
    })[:16]


def _wrap_group(
    raw: dict[str, Any], *, index: int, group_id: str,
    registry_digest: str, generation_id: str, baseline_digest: str,
    controls: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": GROUP_SCHEMA,
        "group_index": index,
        "group_id": group_id,
        "registry_digest": registry_digest,
        "generation_id": generation_id,
        "baseline_batch_digest": baseline_digest,
        "controls": controls,
        **raw,
        "real_forward_outcomes_accessed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    payload["result_digest"] = group_digest(payload)
    payload["checkpoint_integrity_digest"] = checkpoint_integrity_digest(payload)
    return payload


def valid_group(
    payload: dict[str, Any], *, index: int, group_id: str,
    query_ids: list[str], registry_digest: str, generation_id: str,
    baseline_digest: str, controls: dict[str, Any],
    baselines: dict[str, dict[str, Any]],
) -> bool:
    try:
        if not all((
            payload.get("schema_version") == GROUP_SCHEMA,
            payload.get("group_index") == index,
            payload.get("group_id") == group_id,
            payload.get("query_episode_ids") == query_ids,
            payload.get("registry_digest") == registry_digest,
            payload.get("generation_id") == generation_id,
            payload.get("baseline_batch_digest") == baseline_digest,
            payload.get("controls") == controls,
            payload.get("numba_threads") == controls["numba_threads_per_process"],
            payload.get("exact_workers") == controls["exact_workers_per_process"],
            payload.get("real_forward_outcomes_accessed") is False,
            payload.get("result_digest") == group_digest(payload),
            payload.get("checkpoint_integrity_digest")
            == checkpoint_integrity_digest(payload),
            payload.get("gate_passed") is True,
        )):
            return False
        cases = payload["cases"]
        if [case["query_episode_id"] for case in cases] != query_ids:
            return False
        for case in cases:
            baseline = baselines[case["query_episode_id"]]
            expected_gates = {
                "proposal_result_digest_equal": (
                    case["proposal_result_digest"]
                    == baseline["proposal_result_digest"]
                ),
                "matches_equal": case["matches"] == baseline["matches"],
                "certificate_equal": (
                    case["certificate"] == baseline["certificate"]
                ),
                "certificate_digest_equal": (
                    case["certificate_digest"] == baseline["certificate_digest"]
                ),
                "certificate_reconstructed": (
                    case["certificate_digest"] == _certificate_digest({
                        "certificate": case["certificate"],
                        "matches": case["matches"],
                    })
                ),
            }
            if not all((
                case.get("gates") == expected_gates,
                case.get("gate_passed") == all(expected_gates.values()),
            )):
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def _aggregate(
    groups: list[dict[str, Any]], *, expected_ids: list[str],
    registry_digest: str, generation_id: str, baseline_digest: str,
    baseline_seconds: float, controls: dict[str, Any],
    launch_elapsed_seconds: float,
) -> dict[str, Any]:
    group_summaries = [{
        "group_index": group["group_index"],
        "group_id": group["group_id"],
        "query_episode_ids": group["query_episode_ids"],
        "checkpoint_result_digest": group["result_digest"],
        "checkpoint_integrity_digest": group["checkpoint_integrity_digest"],
        "proposal_seconds": group["proposal_seconds"],
        "exact_total_seconds": group["exact_total_seconds"],
        "elapsed_seconds": group["elapsed_seconds"],
        "peak_rss_mb": group["peak_rss_mb"],
        "gate_passed": group["gate_passed"],
    } for group in groups]
    cases = [{
        "registry_case_id": case["registry_case_id"],
        "query_episode_id": case["query_episode_id"],
        "group_id": group["group_id"],
        "proposal_result_digest": case["proposal_result_digest"],
        "certificate_digest": case["certificate_digest"],
        "exact_seconds": case["exact_seconds"],
        "gate_passed": case["gate_passed"],
    } for group in groups for case in group["cases"]]
    returned_ids = [case["query_episode_id"] for case in cases]
    maximum_group_seconds = max(
        (float(group["elapsed_seconds"]) for group in groups), default=float("inf"),
    )
    gates = {
        "all_8_groups_completed": len(groups) == 8,
        "all_24_cases_completed_once": (
            len(returned_ids) == 24 and set(returned_ids) == set(expected_ids)
            and len(set(returned_ids)) == 24
        ),
        "all_group_gates_passed": all(group["gate_passed"] for group in groups),
        "all_case_gates_passed": all(case["gate_passed"] for case in cases),
        "at_least_50pct_faster_than_serial_batch": (
            maximum_group_seconds <= 0.5 * baseline_seconds
        ),
        "real_forward_outcomes_excluded": True,
    }
    payload = {
        "schema_version": EVIDENCE_SCHEMA,
        "registry_digest": registry_digest,
        "generation_id": generation_id,
        "baseline_batch_digest": baseline_digest,
        "baseline_serial_batch_seconds": baseline_seconds,
        "required_maximum_group_seconds": 0.5 * baseline_seconds,
        "controls": controls,
        "expected_query_episode_ids": expected_ids,
        "groups": group_summaries,
        "cases": cases,
        "maximum_group_seconds": maximum_group_seconds,
        "certified_speedup": baseline_seconds / maximum_group_seconds,
        "certified_elapsed_reduction_fraction": (
            1.0 - maximum_group_seconds / baseline_seconds
        ),
        "launch_elapsed_seconds": launch_elapsed_seconds,
        "peak_rss_mb": max(
            (float(group["peak_rss_mb"]) for group in groups), default=0.0,
        ),
        "sum_worker_peak_rss_mb": sum(
            float(group["peak_rss_mb"]) for group in groups
        ),
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
    return payload


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
        args.processes == 8,
        args.numba_threads == 1,
        args.exact_workers == 1,
        args.block_rows == 4_096,
        args.initial_frontier_rows == 16_384,
        args.maximum_frontier_rows == 32_768,
    )):
        raise ValueError("selected parallel gate controls differ")

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
        set(baseline_by_id) == set(expected_ids),
        batch.get("real_forward_outcomes_accessed") is False,
    )):
        raise ValueError("certified batch baseline differs")
    baseline_digest = str(batch["result_digest"])
    baseline_seconds = float(batch["total_seconds"])
    proposal_seconds_per_query = float(batch["shared_scan_seconds"]) / 24
    groups = _balanced_groups(
        registry_cases, baseline_by_id, 8, proposal_seconds_per_query,
    )
    controls = _controls(args)

    specifications = []
    checkpoints = args.output_root / "groups"
    completed: dict[int, dict[str, Any]] = {}
    for index, group in enumerate(groups):
        query_ids = [str(case["episode_id"]) for case in group]
        group_id = _group_id(
            index, query_ids, registry_digest, generation_id,
            baseline_digest, controls,
        )
        checkpoint = checkpoints / f"{index:02d}-{group_id}.json"
        specifications.append((index, group_id, group, query_ids, checkpoint))
        if not checkpoint.exists():
            continue
        try:
            payload = json.loads(checkpoint.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if valid_group(
            payload, index=index, group_id=group_id, query_ids=query_ids,
            registry_digest=registry_digest, generation_id=generation_id,
            baseline_digest=baseline_digest, controls=controls,
            baselines=baseline_by_id,
        ):
            completed[index] = payload
            print(f"[group {index}] resume verified {group_id}", flush=True)

    pending = [spec for spec in specifications if spec[0] not in completed]
    launch_started = perf_counter()
    if pending:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(pending), mp_context=context) as executor:
            futures = {}
            for index, group_id, group, query_ids, checkpoint in pending:
                future = executor.submit(
                    _run_group, str(args.config), str(args.full_root), generation_id,
                    group,
                    {query_id: baseline_by_id[query_id] for query_id in query_ids},
                    args.maximum_frontier_rows, args.initial_frontier_rows,
                    args.block_rows, args.numba_threads, args.exact_workers,
                )
                futures[future] = (
                    index, group_id, query_ids, checkpoint,
                )
            for future in as_completed(futures):
                index, group_id, query_ids, checkpoint = futures[future]
                raw = future.result()
                payload = _wrap_group(
                    raw, index=index, group_id=group_id,
                    registry_digest=registry_digest, generation_id=generation_id,
                    baseline_digest=baseline_digest, controls=controls,
                )
                if not valid_group(
                    payload, index=index, group_id=group_id, query_ids=query_ids,
                    registry_digest=registry_digest, generation_id=generation_id,
                    baseline_digest=baseline_digest, controls=controls,
                    baselines=baseline_by_id,
                ):
                    raise ValueError(f"completed group {index} fails validation")
                _write(checkpoint, payload)
                completed[index] = payload
                print(
                    f"[group {index}] checkpoint PASS {group_id} "
                    f"wall={payload['elapsed_seconds']:.2f}s", flush=True,
                )
    launch_elapsed = perf_counter() - launch_started
    ordered = [completed[index] for index in range(len(groups)) if index in completed]
    evidence = _aggregate(
        ordered, expected_ids=expected_ids, registry_digest=registry_digest,
        generation_id=generation_id, baseline_digest=baseline_digest,
        baseline_seconds=baseline_seconds, controls=controls,
        launch_elapsed_seconds=launch_elapsed,
    )
    _write(args.output_root / "parallel-certified-batch-gate.json", evidence)
    _render(args.output_root / "parallel-certified-batch-gate.html", evidence)
    print(
        f"[parallel-gate] {'PASS' if evidence['gate_passed'] else 'FAIL'} "
        f"groups={len(ordered)} max={evidence['maximum_group_seconds']:.2f}s "
        f"speedup={evidence['certified_speedup']:.4f}x", flush=True,
    )
    return 0 if evidence["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
