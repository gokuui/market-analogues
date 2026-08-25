"""Independent verifier for the selected resumable parallel certified gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import resource
from typing import Any

import pandas as pd

from certified_batch_search_gate import (
    CASE_OMITTED as BASELINE_CASE_OMITTED,
    EVIDENCE_OMITTED as BASELINE_EVIDENCE_OMITTED,
)
from market_analogues.adapters import source_from_spec
from market_analogues.config import load_config
from market_analogues.m04r_batch_registry import validate_m04r_batch_registry
from market_analogues.m04r_certified_search_verification import _certificate_digest
from market_analogues.m04r_full_pack_verification import (
    EVIDENCE_OMITTED as FULL_BUILD_OMITTED,
)
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.types import stable_hash


SCHEMA = "m04r-parallel-certified-batch-verification-v1"
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
VERIFICATION_OMITTED = {"created_at", "peak_rss_mb", "result_digest"}


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
        "<title>M04R-09 parallel gate verification</title><style>"
        "body{font-family:system-ui;max-width:1200px;margin:2rem auto}"
        "pre{white-space:pre-wrap}.pass{color:#075}.fail{color:#a20}"
        "</style></head><body><h1>M04R-09 parallel gate verification: "
        f"<span class=\"{css}\">{status}</span></h1>"
        "<p>Independent reconstruction of physical generation, frozen serial "
        "baseline, group schedule/IDs, checkpoint/result/certificate digests, "
        "performance arithmetic and outcome exclusion.</p><pre>"
        f"{escape(json.dumps(payload, indent=2, sort_keys=True))}"
        "</pre></body></html>"
    )
    temporary.replace(path)


def _group_digest(payload: dict[str, Any]) -> str:
    deterministic = {
        key: value for key, value in payload.items() if key not in GROUP_OMITTED
    }
    deterministic["cases"] = [
        {key: value for key, value in case.items() if key not in CASE_OMITTED}
        for case in payload["cases"]
    ]
    return stable_hash(deterministic)


def _checkpoint_integrity_digest(payload: dict[str, Any]) -> str:
    return stable_hash({
        key: value for key, value in payload.items()
        if key not in {"created_at", "checkpoint_integrity_digest"}
    })


def _evidence_digest(payload: dict[str, Any]) -> str:
    return stable_hash({
        key: value for key, value in payload.items() if key not in EVIDENCE_OMITTED
    })


def _baseline_digest(payload: dict[str, Any]) -> str:
    deterministic = {
        key: value for key, value in payload.items()
        if key not in BASELINE_EVIDENCE_OMITTED
    }
    deterministic["cases"] = [{
        key: value for key, value in case.items()
        if key not in BASELINE_CASE_OMITTED
    } for case in payload["cases"]]
    return stable_hash(deterministic)


def _verification_digest(payload: dict[str, Any]) -> str:
    return stable_hash({
        key: value for key, value in payload.items()
        if key not in VERIFICATION_OMITTED
    })


def _balanced_ids(
    registry_cases: list[dict[str, Any]],
    baseline_by_id: dict[str, dict[str, Any]],
    proposal_seconds_per_query: float,
) -> list[list[str]]:
    weighted = sorted(
        registry_cases,
        key=lambda case: (
            -(
                float(baseline_by_id[str(case["episode_id"])]["exact_seconds"])
                + proposal_seconds_per_query
            ),
            str(case["case_id"]),
        ),
    )
    groups: list[list[str]] = [[] for _ in range(8)]
    loads = [0.0] * 8
    for case in weighted:
        index = min(
            range(8), key=lambda value: (loads[value], len(groups[value]), value),
        )
        query_id = str(case["episode_id"])
        groups[index].append(query_id)
        loads[index] += (
            float(baseline_by_id[query_id]["exact_seconds"])
            + proposal_seconds_per_query
        )
    return groups


def _expected_group_id(
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    failures: list[dict[str, str]] = []
    config = load_config(args.config)
    source = source_from_spec(config.datasets["nasdaq"])
    quality = pd.read_parquet(config.artifact_dir / "quality" / "nasdaq.parquet")
    registry_failures = validate_m04r_batch_registry(
        source, args.registry, quality, config.artifact_dir / "oracles" / "nasdaq",
    )
    if registry_failures:
        failures.append({"scope": "registry", "reason": str(registry_failures)})
    registry = json.loads(args.registry.read_text())
    registry_digest = str(registry["registry_digest"])
    registry_cases = list(registry["cases_data"])
    expected_ids = [str(case["episode_id"]) for case in registry_cases]

    build = json.loads((args.full_root / "packed-bound-full.json").read_text())
    build_content = {
        key: value for key, value in build.items() if key not in FULL_BUILD_OMITTED
    }
    build_ok = all((
        build.get("result_digest") == stable_hash(build_content),
        build.get("gate_passed") is True,
        build.get("shadow_generation") is True,
        build.get("real_forward_outcomes_accessed") is False,
        not (args.full_root / "store" / "active.json").exists(),
    ))
    if not build_ok:
        failures.append({"scope": "build", "reason": "full build evidence differs"})
    generation_id = str(build["generation_id"])
    try:
        loaded = load_packed_generation(
            args.full_root / "store", generation_id,
            verify_content=True, validate_records=False,
        )
        physical_generation_ok = loaded.generation_id == generation_id
    except Exception as exc:  # verifier reports integrity exceptions as evidence
        physical_generation_ok = False
        failures.append({"scope": "generation", "reason": repr(exc)})

    baseline = json.loads(
        (args.batch_root / "certified-batch-gate.json").read_text()
    )
    baseline_digest = str(baseline.get("result_digest", ""))
    baseline_by_id = {
        str(case["query_episode_id"]): case for case in baseline.get("cases", [])
    }
    baseline_ok = all((
        baseline.get("gate_passed") is True,
        baseline.get("registry_digest") == registry_digest,
        baseline.get("generation_id") == generation_id,
        baseline.get("completed_query_episode_ids") == expected_ids,
        set(baseline_by_id) == set(expected_ids),
        baseline.get("real_forward_outcomes_accessed") is False,
        baseline_digest == _baseline_digest(baseline),
    ))
    if not baseline_ok:
        failures.append({"scope": "baseline", "reason": "serial baseline differs"})

    evidence = json.loads(
        (args.gate_root / "parallel-certified-batch-gate.json").read_text()
    )
    controls = evidence.get("controls", {})
    expected_controls = {
        "processes": 8,
        "numba_threads_per_process": 1,
        "exact_workers_per_process": 1,
        "block_rows": 4_096,
        "initial_frontier_rows": 16_384,
        "maximum_frontier_rows": 32_768,
        "seed_rows": 512,
        "requested_positions": True,
        "vector_lower_bounds": True,
        "deferred_alignments": True,
        "sorted_joined_iqr_merge": True,
        "group_assignment": "LPT by prior exact seconds plus equal proposal cost",
        "process_start_method": "spawn",
    }
    group_ids = _balanced_ids(
        registry_cases, baseline_by_id,
        float(baseline["shared_scan_seconds"]) / 24,
    ) if baseline_ok else []

    checkpoints = []
    group_summaries = {int(group["group_index"]): group for group in evidence.get("groups", [])}
    for index, query_ids in enumerate(group_ids):
        group_id = _expected_group_id(
            index, query_ids, registry_digest, generation_id,
            baseline_digest, expected_controls,
        )
        path = args.gate_root / "groups" / f"{index:02d}-{group_id}.json"
        if not path.exists():
            failures.append({"scope": f"group-{index}", "reason": "checkpoint missing"})
            continue
        try:
            group = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            failures.append({"scope": f"group-{index}", "reason": repr(exc)})
            continue
        case_ids = [case.get("query_episode_id") for case in group.get("cases", [])]
        group_ok = all((
            group.get("schema_version") == GROUP_SCHEMA,
            group.get("group_index") == index,
            group.get("group_id") == group_id,
            group.get("query_episode_ids") == query_ids,
            case_ids == query_ids,
            group.get("registry_digest") == registry_digest,
            group.get("generation_id") == generation_id,
            group.get("baseline_batch_digest") == baseline_digest,
            group.get("controls") == expected_controls,
            group.get("numba_threads") == 1,
            group.get("exact_workers") == 1,
            group.get("gate_passed") is True,
            group.get("real_forward_outcomes_accessed") is False,
            group.get("result_digest") == _group_digest(group),
            group.get("checkpoint_integrity_digest")
            == _checkpoint_integrity_digest(group),
        ))
        for case in group.get("cases", []):
            query_id = str(case.get("query_episode_id"))
            serial = baseline_by_id.get(query_id, {})
            expected_gates = {
                "proposal_result_digest_equal": (
                    case.get("proposal_result_digest")
                    == serial.get("proposal_result_digest")
                ),
                "matches_equal": case.get("matches") == serial.get("matches"),
                "certificate_equal": (
                    case.get("certificate") == serial.get("certificate")
                ),
                "certificate_digest_equal": (
                    case.get("certificate_digest")
                    == serial.get("certificate_digest")
                ),
                "certificate_reconstructed": (
                    case.get("certificate_digest") == _certificate_digest({
                        "certificate": case.get("certificate"),
                        "matches": case.get("matches"),
                    })
                ),
            }
            group_ok = group_ok and all((
                case.get("gates") == expected_gates,
                case.get("gate_passed") == all(expected_gates.values()),
            ))
        summary = group_summaries.get(index, {})
        group_ok = group_ok and all((
            summary.get("group_id") == group_id,
            summary.get("query_episode_ids") == query_ids,
            summary.get("checkpoint_result_digest") == group.get("result_digest"),
            summary.get("checkpoint_integrity_digest")
            == group.get("checkpoint_integrity_digest"),
            summary.get("proposal_seconds") == group.get("proposal_seconds"),
            summary.get("exact_total_seconds") == group.get("exact_total_seconds"),
            summary.get("elapsed_seconds") == group.get("elapsed_seconds"),
            summary.get("peak_rss_mb") == group.get("peak_rss_mb"),
            summary.get("gate_passed") is True,
        ))
        if not group_ok:
            failures.append({"scope": f"group-{index}", "reason": "checkpoint differs"})
        checkpoints.append(group)

    maximum_group_seconds = max(
        (float(group["elapsed_seconds"]) for group in checkpoints),
        default=float("inf"),
    )
    baseline_seconds = float(baseline.get("total_seconds", 0.0))
    evidence_ok = all((
        evidence.get("schema_version") == EVIDENCE_SCHEMA,
        evidence.get("registry_digest") == registry_digest,
        evidence.get("generation_id") == generation_id,
        evidence.get("baseline_batch_digest") == baseline_digest,
        evidence.get("controls") == expected_controls,
        evidence.get("expected_query_episode_ids") == expected_ids,
        evidence.get("baseline_serial_batch_seconds") == baseline_seconds,
        evidence.get("required_maximum_group_seconds") == 0.5 * baseline_seconds,
        len(evidence.get("groups", [])) == 8,
        len(evidence.get("cases", [])) == 24,
        maximum_group_seconds == evidence.get("maximum_group_seconds"),
        evidence.get("certified_speedup") == baseline_seconds / maximum_group_seconds,
        evidence.get("certified_elapsed_reduction_fraction")
        == 1.0 - maximum_group_seconds / baseline_seconds,
        evidence.get("gate_passed") is True,
        evidence.get("real_forward_outcomes_accessed") is False,
        evidence.get("result_digest") == _evidence_digest(evidence),
    ))
    if not evidence_ok:
        failures.append({"scope": "aggregate", "reason": "gate evidence differs"})

    gates = {
        "registry_reconstructed": not registry_failures,
        "full_build_evidence_valid": build_ok,
        "physical_generation_rehashed": physical_generation_ok,
        "serial_baseline_rehashed": baseline_ok,
        "controls_and_schedule_reconstructed": (
            controls == expected_controls and len(group_ids) == 8
        ),
        "all_8_group_checkpoints_reconstructed": (
            len(checkpoints) == 8
            and not any(item["scope"].startswith("group-") for item in failures)
        ),
        "all_24_matches_and_certificates_equal": (
            sum(len(group["cases"]) for group in checkpoints) == 24
        ),
        "aggregate_and_performance_reconstructed": evidence_ok,
        "real_forward_outcomes_excluded": all((
            build.get("real_forward_outcomes_accessed") is False,
            baseline.get("real_forward_outcomes_accessed") is False,
            evidence.get("real_forward_outcomes_accessed") is False,
            all(group.get("real_forward_outcomes_accessed") is False for group in checkpoints),
        )),
    }
    payload = {
        "schema_version": SCHEMA,
        "registry_digest": registry_digest,
        "generation_id": generation_id,
        "baseline_batch_digest": baseline_digest,
        "parallel_gate_digest": evidence.get("result_digest"),
        "verified_group_digests": [
            group["result_digest"] for group in checkpoints
        ],
        "verified_query_episode_ids": [
            case["query_episode_id"] for group in checkpoints for case in group["cases"]
        ],
        "verified_maximum_group_seconds": maximum_group_seconds,
        "verified_speedup": (
            baseline_seconds / maximum_group_seconds
            if maximum_group_seconds != float("inf") else None
        ),
        "failures": failures,
        "gates": gates,
        "gate_passed": not failures and all(gates.values()),
        "real_forward_outcomes_accessed": False,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    payload["result_digest"] = _verification_digest(payload)
    _write(args.output_root / "parallel-certified-batch-verification.json", payload)
    _render(args.output_root / "parallel-certified-batch-verification.html", payload)
    print(
        f"[parallel-verifier] {'PASS' if payload['gate_passed'] else 'FAIL'} "
        f"failures={len(failures)} digest={payload['result_digest']}", flush=True,
    )
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
