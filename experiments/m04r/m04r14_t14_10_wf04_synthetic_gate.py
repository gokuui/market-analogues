"""Exercise WF-04 joins, purging and aggregates without real query outcomes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence

import numpy as np
import pandas as pd

from market_analogues.types import stable_hash
from market_analogues.walk_forward_evaluation import (
    continuous_score_record,
    fold_purge_cutoffs,
    probability_score_record,
    query_inside_purged_fold,
    split_abstention_reasons,
)

from experiments.m04r import m04r14_t14_09_outcome_smoke as smoke
from experiments.m04r import m04r14_t14_10_wf04_nonfinal_evaluation as target


SCHEMA = "m04r14-t14-10-wf04-nonfinal-synthetic-verification-v2"
OUTPUT_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf04-nonfinal-synthetic-v2/VERIFIED.json"
)


class SyntheticEvaluationError(RuntimeError):
    pass


def _score_fixture() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(target.SCORED_NONFINAL_FOLDS):
        for month_index in range(6):
            route = target.PRIMARY_CLASSES[(fold_index + month_index) % 3]
            query_id = f"{fold}-{month_index}"
            for lane_index, lane in enumerate(target.PROBABILITY_LANES):
                favorable = .5 - lane_index * .01
                adverse = .3 + lane_index * .005
                rows.append({
                    "query_id": query_id, "query_cutoff": f"2020-{month_index + 1:02d}-28",
                    "month": f"2020-{month_index + 1:02d}", "calendar_year": 2020,
                    "fold_id": fold, "lane": lane, "query_regime": "synthetic",
                    "quality_liquidity_cell": "A|low", "purged_evaluation_included": True,
                    "multiclass_evaluable": True, "directional_evaluable": route != "no_touch",
                    "route_status": route,
                    "multiclass_brier": (lane_index + 1) / 100 + month_index / 1000,
                    "multiclass_log_loss": (lane_index + 1) / 10,
                    "directional_brier": (lane_index + 1) / 50,
                    "directional_log_loss": (lane_index + 1) / 20,
                    "literal_selective": False, "research_dynamic_selective": True,
                    "risk_score": float(month_index + 1),
                    "favorable_probability": favorable, "adverse_probability": adverse,
                    "no_touch_probability": 1 - favorable - adverse,
                })
    return pd.DataFrame(rows)


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=repository,
        text=True, capture_output=True, check=True,
    ).stdout
    if status:
        raise SyntheticEvaluationError("clean H0 worktree required")
    h0 = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    checks: list[str] = []
    probability = probability_score_record([.6, .3, .1], [.7, .3], "favorable_first")
    if not probability["multiclass_evaluable"] or abs(probability["multiclass_brier"] - .26) > 1e-15:
        raise SyntheticEvaluationError("probability oracle differs")
    checks.append("multiclass_and_directional_score")
    excluded = probability_score_record(
        [.6, .3, .1], [.7, .3], "ambiguous_same_first_touch_bar",
    )
    if excluded["multiclass_evaluable"] or not math.isnan(excluded["multiclass_brier"]):
        raise SyntheticEvaluationError("route exclusion differs")
    checks.append("ambiguous_route_exclusion")
    continuous = continuous_score_record(0., {.1: 1., .25: 2., .5: 3., .75: 4., .9: 5.})
    if abs(continuous["central_80_score"] - 14.) > 1e-14 \
            or continuous["central_80_covered"] is not False:
        raise SyntheticEvaluationError("continuous score differs")
    checks.append("quantile_interval_score")
    sessions = pd.bdate_range("2019-01-01", "2021-12-31")
    folds = [
        {"fold_id": "a", "start": "2019-01-01", "end": "2019-12-31"},
        {"fold_id": "b", "start": "2020-01-01", "end": "2020-12-31"},
        {"fold_id": "final", "start": "2021-01-01", "end": "2021-12-31"},
    ]
    limits = fold_purge_cutoffs(sessions, folds, purge_sessions=5)
    prior = sessions[sessions < pd.Timestamp("2020-01-01")]
    if limits["a"] != prior[-6] or not query_inside_purged_fold(prior[-6], "a", limits) \
            or query_inside_purged_fold(prior[-5], "a", limits) or limits["final"] is not None:
        raise SyntheticEvaluationError("purge boundary differs")
    checks.append("session_purge_boundary")
    permanent, dynamic = split_abstention_reasons(
        "poor_data_quality|failed_calibration|historically_novel_query",
    )
    if len(permanent) != 2 or dynamic != ("historically_novel_query",):
        raise SyntheticEvaluationError("abstention partition differs")
    checks.append("literal_and_dynamic_abstention_separation")
    scores = _score_fixture()
    fold_metrics = target._fold_metrics(scores)
    stability = target._stability_metrics(scores)
    calibration = target._calibration(scores)
    risk = target._risk_coverage(scores)
    inference = target._inference(scores)
    if len(fold_metrics) != 35 or len(risk) != 50 or len(inference) != 25 \
            or not np.isfinite(calibration.ece).all() \
            or set(stability.dimension) != {
                "fold", "calendar_year", "benchmark_regime", "quality_liquidity_cell",
            }:
        raise SyntheticEvaluationError("aggregate surface differs")
    checks.append("fold_calibration_stability_risk_inference_surface")
    reversed_inference = target._inference(scores.iloc[::-1].reset_index(drop=True))
    pd.testing.assert_frame_equal(inference, reversed_inference, check_exact=True)
    checks.append("input_order_invariance")
    state = {
        "schema_version": SCHEMA, "status": "verified", "passed": True,
        "implementation_h0": h0, "checks": checks, "check_count": len(checks),
        "real_query_outcomes_accessed": False, "final_period_result_opened": False,
        "production_promotion_authorized": False,
    }
    result = {**state, "result_digest": stable_hash(state), "created_at": datetime.now(timezone.utc).isoformat()}
    output = repository / OUTPUT_RELATIVE
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SyntheticEvaluationError("synthetic receipt already exists")
    smoke._atomic_json(output, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(execute(args.repository), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
