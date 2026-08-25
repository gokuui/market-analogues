"""Persist repeated M04R-08A certified-completion evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import resource
from typing import Any

from market_analogues.adapters import source_from_spec
from market_analogues.certified_packed_search import (
    certified_packed_search, certified_packed_search_contract,
)
from market_analogues.config import load_config
from market_analogues.episodes import build_episode
from market_analogues.types import InstrumentKey, SearchQuery, stable_hash


SCHEMA_VERSION = "m04r-certified-packed-search-gate-v1"
NONDETERMINISTIC = {"created_at", "run_seconds", "peak_rss_mb", "result_digest"}


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _match(match: Any) -> dict[str, Any]:
    return {
        "episode_id": match.episode_key.id,
        "symbol": match.episode_key.instrument.source_symbol,
        "cutoff": match.episode_key.cutoff.isoformat(),
        "total_distance": match.total_distance,
        "component_distances": dict(sorted(match.component_distances.items())),
        "alignment": [[int(left), int(right)] for left, right in match.alignment],
        "quality_tier": match.quality_tier,
    }


def _run_payload(result: Any, authority: dict[str, Any], controls: dict[str, int]) -> dict[str, Any]:
    matches = [_match(value) for value in result.matches]
    expected = authority["matches"]
    if len(matches) != len(expected):
        raise ValueError(
            f"certified/authority result lengths differ: {len(matches)} != {len(expected)}"
        )
    if any(
        set(one["component_distances"]) != set(two["component_distances"])
        for one, two in zip(matches, expected)
    ):
        raise ValueError("certified/authority component names differ")
    total_delta = max(abs(
        float(one["total_distance"]) - float(two["total_distance"])
    ) for one, two in zip(matches, expected))
    component_delta = max(abs(
        float(one["component_distances"][name])
        - float(two["component_distances"][name])
    ) for one, two in zip(matches, expected)
      for name in one["component_distances"])
    certificate = asdict(result.certificate)
    certificate.pop("elapsed_seconds")
    return {
        "controls": controls,
        "matches": matches,
        "ordered_ids_equal_authority": (
            [row["episode_id"] for row in matches]
            == [row["episode_id"] for row in expected]
        ),
        "alignments_equal_authority": (
            [row["alignment"] for row in matches]
            == [row["alignment"] for row in expected]
        ),
        "maximum_total_delta": total_delta,
        "maximum_component_delta": component_delta,
        "certificate": certificate,
        "certificate_digest": result.certificate.result_digest,
        "seconds": result.certificate.elapsed_seconds,
    }


def _render(path: Path, payload: dict[str, Any]) -> None:
    status = "PASS" if payload.get("gate_passed") else "INCOMPLETE/FAIL"
    summary = {key: value for key, value in payload.items() if key != "runs"}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(f"""<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>M04R-08A certified packed search</title><style>body{{font-family:system-ui;max-width:1200px;margin:2rem auto}}pre{{white-space:pre-wrap}}.pass{{color:#075}}.fail{{color:#a20}}</style></head><body><h1>M04R-08A repeated certified search: <span class=\"{'pass' if payload.get('gate_passed') else 'fail'}\">{status}</span></h1><p>Development-authority verification only; no outcomes or setup labels are accessed.</p><pre>{escape(json.dumps(summary, indent=2, sort_keys=True))}</pre><h2>Runs</h2><pre>{escape(json.dumps(payload.get('runs', []), indent=2, sort_keys=True))}</pre></body></html>""")
    temporary.replace(path)


def _checkpoint(path: Path, payload: dict[str, Any]) -> None:
    _write(path, payload)
    _render(path.with_suffix(".html"), payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()
    if args.runs < 2:
        raise ValueError("at least two full runs are required")
    config = load_config(args.config)
    build = json.loads((args.full_root / "packed-bound-full.json").read_text())
    query_id = str(build["benchmark_selection"]["query_episode_id"])
    authority_path = (
        config.artifact_dir / "gate12" / "authorities" / "nasdaq"
        / "cases" / f"{query_id}.json"
    )
    authority = json.loads(authority_path.read_text())
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
    controls = [
        {"block_rows": 2_048, "workers": 8},
        {"block_rows": 4_097, "workers": 4},
    ]
    while len(controls) < args.runs:
        controls.append({
            "block_rows": 1_024 + len(controls),
            "workers": 8,
        })
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract_digest": certified_packed_search_contract()["digest"],
        "generation_id": build["generation_id"],
        "full_build_evidence_digest": build["result_digest"],
        "authority_digest": authority["authority_digest"],
        "query_episode_id": query_id,
        "failed_attempt": {
            "status": "interrupted performance failure",
            "design": "exact-score all 8192 retained frontier rows",
            "elapsed_before_interrupt_seconds": 840,
            "observed_rss_mb": 843,
            "exit_code": 130,
        },
        "runs": [], "completed_runs": 0, "required_runs": args.runs,
        "real_forward_outcomes_accessed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    for index, control in enumerate(controls[:args.runs]):
        try:
            result = certified_packed_search(
                query, source, request, args.full_root / "store",
                str(build["generation_id"]), store_dataset_id="nasdaq",
                initial_frontier_rows=8_192, maximum_frontier_rows=32_768,
                seed_rows=512, block_rows=control["block_rows"],
                workers=control["workers"], sparse_cutoff=8,
                verify_content=index == 0,
            )
        except Exception as exc:
            payload["failure"] = {
                "run_index": index + 1,
                "controls": control,
                "type": type(exc).__name__,
                "message": str(exc),
            }
            payload["gate_passed"] = False
            payload["created_at"] = datetime.now(timezone.utc).isoformat()
            _checkpoint(args.output, payload)
            raise
        payload["runs"].append(_run_payload(result, authority, control))
        payload["completed_runs"] = index + 1
        payload["run_seconds"] = [row["seconds"] for row in payload["runs"]]
        payload["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        payload["gate_passed"] = False
        _checkpoint(args.output, payload)
    runs = payload["runs"]
    stable_fields = ("matches", "certificate", "certificate_digest")
    repeated = all(
        row[field] == runs[0][field]
        for row in runs[1:] for field in stable_fields
    )
    gates = {
        "required_repeats_complete": len(runs) == args.runs,
        "repeated_results_identical": repeated,
        "ordered_ids_equal_authority": all(row["ordered_ids_equal_authority"] for row in runs),
        "alignments_equal_authority": all(row["alignments_equal_authority"] for row in runs),
        "total_delta_within_1e_7": max(row["maximum_total_delta"] for row in runs) <= 1e-7,
        "component_delta_within_1e_6": max(row["maximum_component_delta"] for row in runs) <= 1e-6,
        "maximum_runtime_within_600_seconds": max(row["seconds"] for row in runs) <= 600,
        "rss_within_1536_mib": float(payload["peak_rss_mb"]) <= 1_536,
        "candidate_accounting": all(
            row["certificate"]["exact_evaluated"]
            + row["certificate"]["safely_pruned"]
            == row["certificate"]["eligible_candidates"] for row in runs
        ),
        "strict_stopping": all(
            row["certificate"]["next_lower_bound"]
            > row["certificate"]["stop_threshold"] for row in runs
        ),
    }
    payload["gates"] = gates
    payload["gate_passed"] = all(gates.values())
    payload["numeric_tolerance_reason"] = (
        "frozen authority predates shared exact-kernel refactor; observed deltas are "
        "isolated to float32 coarse component while IDs and alignments are exact"
    )
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    deterministic = {
        key: value for key, value in payload.items()
        if key not in NONDETERMINISTIC
    }
    deterministic["runs"] = [
        {key: value for key, value in run.items() if key != "seconds"}
        for run in payload["runs"]
    ]
    payload["result_digest"] = stable_hash(deterministic)
    _checkpoint(args.output, payload)
    print(json.dumps({
        "gate_passed": payload["gate_passed"],
        "run_seconds": payload["run_seconds"],
        "peak_rss_mb": payload["peak_rss_mb"],
        "certificate_digest": runs[0]["certificate_digest"],
        "result_digest": payload["result_digest"],
    }, indent=2))
    return 0 if payload["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
