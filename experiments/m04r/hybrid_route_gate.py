"""Bound-assisted rescue experiment after compact proposal-v2 rejection.

This does not relabel M04R-05 as passed.  It combines the sealed compact ranks
with the already-built full-universe native lower-bound frontiers to decide
whether a bound-first packed architecture deserves the M04R-06 performance POC.
The route allocation is fixed at 10% bound, 9% for each of seven compact
components, and 27% compact composite at every pool size.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from frontier_rank_poc import run_case

from market_analogues.proposal_v2 import (
    LAYOUTS, PROPOSAL_POOLS, PROPOSAL_ROUTES, proposal_v2_contract,
)
from market_analogues.types import stable_hash


SCHEMA_VERSION = "m04r-bound-assisted-route-gate-v1"


def _quotas(pool: int) -> dict[str, int]:
    if pool not in PROPOSAL_POOLS:
        raise ValueError(f"unsupported pool {pool}")
    return {
        "certified_bound": pool * 10 // 100,
        "per_component": pool * 9 // 100,
        "composite": pool * 27 // 100,
    }


def _compact_digest_valid(payload: dict[str, Any]) -> bool:
    omitted = {
        "created_at", "elapsed_seconds", "projection_seconds", "scoring_seconds",
        "peak_rss_mb", "end_to_end_rows_per_second", "result_digest",
    }
    deterministic = {key: value for key, value in payload.items() if key not in omitted}
    return payload.get("result_digest") == stable_hash(deterministic)


def _admitted(
    target: dict[str, Any], bound_rank: int, pool: int,
) -> bool:
    quotas = _quotas(pool)
    return (
        bound_rank <= quotas["certified_bound"]
        or int(target["upper_composite_rank"]) <= quotas["composite"]
        or min(int(value) for value in target["upper_route_ranks"].values())
        <= quotas["per_component"]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--compact-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    compact = json.loads(args.compact_evidence.read_text())
    if not _compact_digest_valid(compact):
        raise ValueError("compact proposal evidence digest differs")
    if compact.get("contract_digest") != proposal_v2_contract()["digest"]:
        raise ValueError("compact proposal contract differs")
    if not compact.get("is_full_universe") or len(compact.get("authority_cases", [])) != 12:
        raise ValueError("compact evidence is not the full 12-authority result")
    if compact.get("selected_dimensions") is not None:
        raise ValueError("hybrid rescue is only valid after compact selection rejection")

    bound_cases = []
    by_query = {}
    for case_index, compact_case in enumerate(compact["authority_cases"], 1):
        case = run_case(args.artifact_dir, compact_case["query_episode_id"])
        if case["eligible_rows"] != compact_case["eligible_rows"]:
            raise ValueError(
                f"bound/compact row accounting differs for {case['query_episode_id']}"
            )
        by_query[case["query_episode_id"]] = case
        bound_cases.append(case)
        print(json.dumps({
            "frontiers_completed": case_index,
            "frontiers_total": len(compact["authority_cases"]),
            "symbol": case["symbol"],
            "maximum_target_rank": case["maximum_target_rank"],
        }), flush=True)

    case_rows = []
    pass_by_layout = {
        (dimensions, dtype): True
        for dimensions in sorted(LAYOUTS) for dtype in ("float32", "float16")
    }
    for compact_case in compact["authority_cases"]:
        bound = by_query[compact_case["query_episode_id"]]
        bound_targets = {row["episode_id"]: row for row in bound["targets"]}
        layouts = {}
        for dimensions in sorted(LAYOUTS):
            layouts[str(dimensions)] = {}
            for dtype in ("float32", "float16"):
                compact_targets = compact_case["layouts"][str(dimensions)][dtype][
                    "target_ranks"
                ]
                targets = []
                for target in compact_targets:
                    bound_target = bound_targets[target["episode_id"]]
                    upper_bound_rank = (
                        int(bound_target["lower_bound_rank"])
                        + int(bound_target["ties_including_self"]) - 1
                    )
                    targets.append({
                        "authority_rank": target["authority_rank"],
                        "episode_id": target["episode_id"],
                        "symbol": target["symbol"],
                        "compact_upper_route_ranks": target["upper_route_ranks"],
                        "compact_upper_composite_rank": target["upper_composite_rank"],
                        "native_bound_lower_rank": bound_target["lower_bound_rank"],
                        "native_bound_upper_rank": upper_bound_rank,
                    })
                recalls = {}
                for pool in sorted(PROPOSAL_POOLS):
                    admitted = sum(_admitted({
                        "upper_route_ranks": target["compact_upper_route_ranks"],
                        "upper_composite_rank": target["compact_upper_composite_rank"],
                    }, target["native_bound_upper_rank"], pool) for target in targets)
                    recalls[str(pool)] = {
                        "admitted": admitted, "total": 20,
                        "recall": admitted / 20,
                        "quotas": _quotas(pool),
                    }
                passed = (
                    recalls["20000"]["admitted"] == 20
                    and recalls["10000"]["admitted"] >= 19
                )
                pass_by_layout[(dimensions, dtype)] &= passed
                layouts[str(dimensions)][dtype] = {
                    "pool_recalls": recalls,
                    "targets": targets,
                    "passed": passed,
                }
        case_rows.append({
            "query_episode_id": compact_case["query_episode_id"],
            "authority_digest": compact_case["authority_digest"],
            "symbol": compact_case["symbol"],
            "cutoff": compact_case["cutoff"],
            "eligible_rows": compact_case["eligible_rows"],
            "maximum_native_bound_target_rank": bound["maximum_target_rank"],
            "layouts": layouts,
        })
    # Float16 remains in the sensitivity table but is not selectable after any
    # full-universe source overflow in the compact experiment.
    for dimensions in sorted(LAYOUTS):
        pass_by_layout[(dimensions, "float16")] &= int(
            compact["quantization_overflow_rows"][str(dimensions)]
        ) == 0
    selected = next((
        dimensions for dimensions in sorted(LAYOUTS)
        if pass_by_layout[(dimensions, "float32")]
    ), None)
    deterministic = {
        "schema_version": SCHEMA_VERSION,
        "compact_contract_digest": proposal_v2_contract()["digest"],
        "compact_evidence_digest": compact["result_digest"],
        "route_allocation": {
            str(pool): _quotas(pool) for pool in sorted(PROPOSAL_POOLS)
        },
        "authority_cases": case_rows,
        "authority_case_count": len(case_rows),
        "maximum_native_bound_target_rank": max(
            row["maximum_native_bound_target_rank"] for row in case_rows
        ),
        "layout_float32_pass": {
            str(dimensions): pass_by_layout[(dimensions, "float32")]
            for dimensions in sorted(LAYOUTS)
        },
        "layout_float16_pass": {
            str(dimensions): pass_by_layout[(dimensions, "float16")]
            for dimensions in sorted(LAYOUTS)
        },
        "smallest_bound_assisted_dimensions": selected,
        "recall_gate_passed": selected is not None,
        "architecture_status": (
            "eligible for M04R-06 packed bound-first performance POC; not a "
            "compact-only M04R-05 pass and not a production promotion"
        ),
        "requires_quantized_bound_rank_confirmation": True,
        "requires_packed_performance_confirmation": True,
        "real_forward_outcomes_accessed": False,
    }
    payload = {
        **deterministic,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": perf_counter() - started,
        "result_digest": stable_hash(deterministic),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    rows = "".join(
        f"<tr><td>{escape(row['symbol'])}</td><td>{escape(row['cutoff'])}</td>"
        f"<td>{row['maximum_native_bound_target_rank']}</td>"
        + "".join(
            f"<td>{row['layouts'][str(dimensions)]['float32']['pool_recalls']['10000']['admitted']}/20</td>"
            f"<td>{row['layouts'][str(dimensions)]['float32']['pool_recalls']['20000']['admitted']}/20</td>"
            for dimensions in sorted(LAYOUTS)
        ) + "</tr>" for row in case_rows
    )
    columns = "".join(
        f"<th>{dimensions} @10k</th><th>{dimensions} @20k</th>"
        for dimensions in sorted(LAYOUTS)
    )
    html = args.output.with_suffix(".html")
    html.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>M04R bound-assisted route gate</title><style>body{{font-family:system-ui;max-width:1400px;margin:2rem auto}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.45rem;border-bottom:1px solid #ddd;text-align:left}}pre{{white-space:pre-wrap}}</style></head><body><h1>Bound-assisted route recall: {'PASS' if deterministic['recall_gate_passed'] else 'FAIL'}</h1><p>This is a routing POC, not a compact-only pass or production promotion. It uses a fixed 10% native-certified-bound route and still requires quantized-rank and packed-performance confirmation.</p><p>Evidence <code>{payload['result_digest']}</code>.</p><pre>{escape(json.dumps({key: value for key, value in payload.items() if key != 'authority_cases'}, indent=2, sort_keys=True))}</pre><table><thead><tr><th>Query</th><th>Cutoff</th><th>Max bound rank</th>{columns}</tr></thead><tbody>{rows}</tbody></table></body></html>""")
    print(json.dumps({
        "output": str(args.output),
        "recall_gate_passed": payload["recall_gate_passed"],
        "smallest_bound_assisted_dimensions": selected,
        "maximum_native_bound_target_rank": payload["maximum_native_bound_target_rank"],
        "result_digest": payload["result_digest"],
        "elapsed_seconds": payload["elapsed_seconds"],
    }, indent=2))
    return 0 if payload["recall_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
