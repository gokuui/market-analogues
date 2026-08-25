"""Measure per-query certified latency without changing exact search semantics."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from html import escape
import json
import multiprocessing
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import pandas as pd

from market_analogues.types import stable_hash

try:
    from experiments.m04r.m04r11_build_authorities import _worker
except ModuleNotFoundError:  # Direct script execution adds this directory to sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from m04r11_build_authorities import _worker


SCHEMA_VERSION = "m04r12-scalar-certified-scheduler-gate-v1"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _timing_valid(
    payload: dict[str, Any], *, case: dict[str, Any],
    authority_digest: str, contract_digest: str,
) -> bool:
    deterministic = {
        key: value for key, value in payload.items()
        if key not in {"created_at", "result_digest"}
    }
    return all((
        payload.get("schema_version") == "m04r12-scalar-case-timing-v1",
        payload.get("registry_case_id") == case["case_id"],
        payload.get("query_episode_id") == case["episode_id"],
        payload.get("authority_digest") == authority_digest,
        payload.get("contract_digest") == contract_digest,
        payload.get("result_digest") == stable_hash(deterministic),
        float(payload.get("end_to_end_seconds", -1)) > 0,
    ))


def _render(path: Path, payload: dict[str, Any]) -> None:
    status = "PASS" if payload["passed"] else "FAIL"
    css = "#075" if payload["passed"] else "#a20"
    path.write_text(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>M04R-12 scalar certified scheduler gate</title><style>"
        "body{font-family:system-ui;max-width:1200px;margin:2rem auto}"
        "pre{white-space:pre-wrap}</style></head><body>"
        f"<h1 style=\"color:{css}\">Scalar scheduler: {status}</h1>"
        "<p>Each frozen query receives its own complete proposal scan and exact "
        "certified completion. Semantic result and certificate digests must equal "
        "the sealed M04R-11 authority before latency can pass.</p><pre>"
        f"{escape(json.dumps(payload, indent=2, sort_keys=True))}"
        "</pre></body></html>"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=8)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text())
    cases = list(registry["cases_data"])
    if len(cases) != 60 or not 1 <= args.processes <= 60:
        raise ValueError("scalar scheduler gate requires 60 cases and 1..60 processes")
    authority_contract = json.loads(
        (args.authority_root / "authority-contract.json").read_text()
    )
    authority_matrix = json.loads(
        (args.authority_root / "authority-matrix.json").read_text()
    )
    authority_seal = json.loads((args.authority_root / "SEALED.json").read_text())
    if not all((
        authority_matrix.get("gate_passed") is True,
        authority_matrix.get("completed_cases") == 60,
        authority_seal.get("authority_correctness_sealed") is True,
        authority_seal.get("authority_matrix_digest")
        == authority_matrix.get("result_digest"),
        authority_seal.get("production_promotion_authorized") is False,
    )):
        raise ValueError("sealed authority correctness prerequisite differs")
    controls = dict(authority_contract["controls"])
    generation_id = str(authority_contract["generation_id"])
    expected = {
        str(row["query_episode_id"]): json.loads((
            args.authority_root / "cases" / f"{row['query_episode_id']}.json"
        ).read_text())
        for row in authority_matrix["cases"]
    }
    output_cases = args.output_root / "cases"
    timing_root = args.output_root / "timings"
    valid_timings: dict[str, dict[str, Any]] = {}
    remaining = []
    for case in cases:
        query_id = str(case["episode_id"])
        authority = expected[query_id]
        case_path = output_cases / f"{query_id}.json"
        timing_path = timing_root / f"{query_id}.json"
        if case_path.exists() and timing_path.exists():
            observed_case = json.loads(case_path.read_text())
            observed_timing = json.loads(timing_path.read_text())
            if (
                observed_case.get("result_digest") == authority["result_digest"]
                and _timing_valid(
                    observed_timing, case=case,
                    authority_digest=authority["result_digest"],
                    contract_digest=authority_contract["contract_digest"],
                )
            ):
                valid_timings[query_id] = observed_timing
                continue
        remaining.append(case)
    print(
        f"[scalar-scheduler] valid={len(valid_timings)}/60 "
        f"remaining={len(remaining)} processes={args.processes}",
        flush=True,
    )
    failures: list[str] = []
    started = perf_counter()
    if remaining:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(args.processes, len(remaining)), mp_context=context,
        ) as executor:
            futures = {
                executor.submit(
                    _worker, str(args.config), str(args.full_root),
                    str(args.output_root), generation_id, authority_contract,
                    (case,), controls,
                ): case for case in remaining
            }
            for future in as_completed(futures):
                case = futures[future]
                query_id = str(case["episode_id"])
                try:
                    worker = future.result()
                    observed = json.loads(
                        (output_cases / f"{query_id}.json").read_text()
                    )
                    authority = expected[query_id]
                    if observed.get("result_digest") != authority["result_digest"]:
                        raise ValueError("scalar result differs from sealed authority")
                    timing = {
                        "schema_version": "m04r12-scalar-case-timing-v1",
                        "registry_case_id": case["case_id"],
                        "query_episode_id": query_id,
                        "authority_digest": authority["result_digest"],
                        "contract_digest": authority_contract["contract_digest"],
                        "proposal_seconds": observed["proposal_seconds"],
                        "exact_seconds": observed["exact_seconds"],
                        "end_to_end_seconds": worker["elapsed_seconds"],
                        "peak_rss_mb": worker["peak_rss_mb"],
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                    deterministic = {
                        key: value for key, value in timing.items()
                        if key != "created_at"
                    }
                    timing["result_digest"] = stable_hash(deterministic)
                    _atomic_json(timing_root / f"{query_id}.json", timing)
                    valid_timings[query_id] = timing
                    print(
                        f"[scalar:{case['case_id']}] PASS "
                        f"wall={timing['end_to_end_seconds']:.2f}s",
                        flush=True,
                    )
                except Exception as exc:
                    failures.append(f"{case['case_id']}:{type(exc).__name__}:{exc}")
                    print(f"[scalar:{case['case_id']}] FAIL {exc}", flush=True)
    ordered = [valid_timings[str(case["episode_id"])] for case in cases
               if str(case["episode_id"]) in valid_timings]
    walls = [float(row["end_to_end_seconds"]) for row in ordered]
    p95 = float(pd.Series(walls).quantile(.95)) if walls else None
    maximum = max(walls, default=None)
    peak_rss = max((float(row["peak_rss_mb"]) for row in ordered), default=0.0)
    gates = {
        "all_60_semantically_equal": len(ordered) == 60 and not failures,
        "p95_end_to_end_at_most_300_seconds": p95 is not None and p95 <= 300,
        "maximum_end_to_end_at_most_600_seconds": (
            maximum is not None and maximum <= 600
        ),
        "peak_rss_at_most_1536_mb": peak_rss <= 1_536,
    }
    deterministic = {
        "schema_version": SCHEMA_VERSION,
        "contract_digest": authority_contract["contract_digest"],
        "authority_matrix_digest": authority_matrix["result_digest"],
        "authority_seal_digest": authority_seal["seal_digest"],
        "registry_digest": registry["registry_digest"],
        "generation_id": generation_id,
        "processes": args.processes,
        "completed_cases": len(ordered),
        "p95_end_to_end_seconds": p95,
        "maximum_end_to_end_seconds": maximum,
        "maximum_worker_rss_mb": peak_rss,
        "case_timing_digests": [row["result_digest"] for row in ordered],
        "failures": failures,
        "gates": gates,
        "passed": all(gates.values()),
        "real_forward_outcomes_accessed": False,
    }
    payload = {
        **deterministic,
        "elapsed_seconds": perf_counter() - started,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result_digest": stable_hash(deterministic),
    }
    _atomic_json(args.output_root / "scalar-scheduler-gate.json", payload)
    _render(args.output_root / "scalar-scheduler-gate.html", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
