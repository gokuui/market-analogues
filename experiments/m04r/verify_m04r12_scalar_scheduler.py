"""Independent verifier for the M04R-12 scalar scheduler timing gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from typing import Any

import pandas as pd

from market_analogues.types import stable_hash


SCHEMA_VERSION = "m04r12-scalar-scheduler-verification-v1"
CASE_OMITTED = {
    "created_at", "proposal_seconds", "amortized_proposal_seconds",
    "exact_seconds", "final_exact_seconds", "frontier_attempt_measurements",
    "peak_rss_mb", "result_digest", "checkpoint_integrity_digest",
}


def _case_result_digest(payload: dict[str, Any]) -> str:
    return stable_hash({
        key: value for key, value in payload.items() if key not in CASE_OMITTED
    })


def _timing_result_digest(payload: dict[str, Any]) -> str:
    return stable_hash({
        key: value for key, value in payload.items()
        if key not in {"created_at", "result_digest"}
    })


def _gate_result_digest(payload: dict[str, Any]) -> str:
    return stable_hash({
        key: value for key, value in payload.items()
        if key not in {"created_at", "elapsed_seconds", "result_digest"}
    })


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text())
    cases = list(registry["cases_data"])
    contract = json.loads(
        (args.authority_root / "authority-contract.json").read_text()
    )
    matrix = json.loads((args.authority_root / "authority-matrix.json").read_text())
    seal = json.loads((args.authority_root / "SEALED.json").read_text())
    gate = json.loads((args.gate_root / "scalar-scheduler-gate.json").read_text())
    failures: list[str] = []
    timing_digests: list[str] = []
    timings: list[dict[str, Any]] = []
    for case in cases:
        query_id = str(case["episode_id"])
        authority_path = args.authority_root / "cases" / f"{query_id}.json"
        scalar_path = args.gate_root / "cases" / f"{query_id}.json"
        timing_path = args.gate_root / "timings" / f"{query_id}.json"
        if not all(path.is_file() for path in (
            authority_path, scalar_path, timing_path,
        )):
            failures.append(f"case artifact missing:{case['case_id']}")
            continue
        authority = json.loads(authority_path.read_text())
        scalar = json.loads(scalar_path.read_text())
        timing = json.loads(timing_path.read_text())
        timing_digest = _timing_result_digest(timing)
        if not all((
            authority.get("result_digest") == _case_result_digest(authority),
            scalar.get("result_digest") == _case_result_digest(scalar),
            scalar.get("result_digest") == authority.get("result_digest"),
            scalar.get("certificate_digest") == authority.get("certificate_digest"),
            scalar.get("contract_digest") == contract.get("contract_digest"),
            scalar.get("real_forward_outcomes_accessed") is False,
            timing.get("schema_version") == "m04r12-scalar-case-timing-v1",
            timing.get("registry_case_id") == case["case_id"],
            timing.get("query_episode_id") == query_id,
            timing.get("authority_digest") == authority.get("result_digest"),
            timing.get("contract_digest") == contract.get("contract_digest"),
            timing.get("result_digest") == timing_digest,
            float(timing.get("proposal_seconds", -1))
            == float(scalar.get("proposal_seconds", -2)),
            float(timing.get("exact_seconds", -1))
            == float(scalar.get("exact_seconds", -2)),
            float(timing.get("peak_rss_mb", -1))
            == float(scalar.get("peak_rss_mb", -2)),
            float(timing.get("end_to_end_seconds", -1))
            >= float(timing.get("proposal_seconds", 0))
            + float(timing.get("exact_seconds", 0)),
        )):
            failures.append(f"case/timing binding differs:{case['case_id']}")
        timing_digests.append(str(timing.get("result_digest")))
        timings.append(timing)
    walls = [float(row["end_to_end_seconds"]) for row in timings]
    p95 = float(pd.Series(walls).quantile(.95)) if walls else None
    maximum = max(walls, default=None)
    peak_rss = max((float(row["peak_rss_mb"]) for row in timings), default=0.0)
    expected_gates = {
        "all_60_semantically_equal": len(timings) == 60 and not failures,
        "p95_end_to_end_at_most_300_seconds": p95 is not None and p95 <= 300,
        "maximum_end_to_end_at_most_600_seconds": (
            maximum is not None and maximum <= 600
        ),
        "peak_rss_at_most_1536_mb": peak_rss <= 1_536,
    }
    if not all((
        len(cases) == 60,
        contract.get("registry_digest") == registry.get("registry_digest"),
        matrix.get("completed_cases") == 60,
        matrix.get("gate_passed") is True,
        seal.get("authority_correctness_sealed") is True,
        seal.get("authority_matrix_digest") == matrix.get("result_digest"),
        gate.get("schema_version") == "m04r12-scalar-certified-scheduler-gate-v1",
        gate.get("contract_digest") == contract.get("contract_digest"),
        gate.get("authority_matrix_digest") == matrix.get("result_digest"),
        gate.get("authority_seal_digest") == seal.get("seal_digest"),
        gate.get("registry_digest") == registry.get("registry_digest"),
        gate.get("generation_id") == contract.get("generation_id"),
        gate.get("processes") == 4,
        gate.get("completed_cases") == 60,
        gate.get("case_timing_digests") == timing_digests,
        gate.get("p95_end_to_end_seconds") == p95,
        gate.get("maximum_end_to_end_seconds") == maximum,
        gate.get("maximum_worker_rss_mb") == peak_rss,
        gate.get("failures") == [],
        gate.get("gates") == expected_gates,
        gate.get("passed") is True and all(expected_gates.values()),
        gate.get("real_forward_outcomes_accessed") is False,
        gate.get("result_digest") == _gate_result_digest(gate),
    )):
        failures.append("aggregate scalar scheduler gate differs")
    unique = sorted(set(failures))
    deterministic = {
        "schema_version": SCHEMA_VERSION,
        "registry_digest": registry["registry_digest"],
        "contract_digest": contract["contract_digest"],
        "authority_matrix_digest": matrix["result_digest"],
        "authority_seal_digest": seal["seal_digest"],
        "scalar_gate_digest": gate.get("result_digest"),
        "verified_cases": len(timings),
        "p95_end_to_end_seconds": p95,
        "maximum_end_to_end_seconds": maximum,
        "maximum_worker_rss_mb": peak_rss,
        "failures": unique,
        "passed": not unique,
        "real_forward_outcomes_accessed": False,
    }
    payload = {
        **deterministic,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result_digest": stable_hash(deterministic),
    }
    _atomic_json(args.output_root / "m04r12-scalar-scheduler-verification.json", payload)
    status = "PASS" if payload["passed"] else "FAIL"
    html = args.output_root / "m04r12-scalar-scheduler-verification.html"
    html.write_text(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>M04R-12 scalar scheduler verification</title></head><body>"
        f"<h1>{status}</h1><p>Independent reconstruction of every authority, "
        "scalar result, timing checkpoint, aggregate performance gate and digest."
        f"</p><pre>{escape(json.dumps(payload, indent=2, sort_keys=True))}"
        "</pre></body></html>"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(html)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
