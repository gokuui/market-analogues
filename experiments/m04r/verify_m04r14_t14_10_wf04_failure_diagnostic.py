"""Independently audit every WF-04 post-hoc diagnostic row."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from market_analogues.types import stable_hash

from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_10_wf04_failure_diagnostic as target


SCHEMA = "m04r14-t14-10-wf04-failure-diagnostic-verification-v1"
OUTPUT_RELATIVE = Path("config/data/analogues/m04r14/t14-10-wf04-failure-diagnostic-v1-verification")
TOLERANCE = 5e-13


class DiagnosticVerificationError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20): digest.update(block)
    return digest.hexdigest()


def _valid(value: Mapping[str, Any], *, timing: bool = False) -> bool:
    omitted = {"result_digest", "created_at"} | ({"elapsed_seconds"} if timing else set())
    return value.get("result_digest") == stable_hash({k: v for k, v in value.items() if k not in omitted})


def _close(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right): return True
    if isinstance(left, (float, np.floating, int, np.integer)) and isinstance(right, (float, np.floating, int, np.integer)):
        return abs(float(left) - float(right)) <= TOLERANCE
    return str(left) == str(right)


def _source_scores(repository: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = target._scores(repository)
    raw = pd.read_parquet(repository / target.predictions.OUTPUT_RELATIVE / "raw-predictions.parquet")
    raw = raw.loc[(raw.prefix == 20) & (raw.lane == "composite") & raw.query_id.isin(set(scores.query_id))]
    return scores, raw


def _verify_comparisons(repository: Path, observed: pd.DataFrame) -> None:
    scores, raw = _source_scores(repository)
    by_key = {(str(r.query_id), str(r.lane)): r for r in scores.itertuples(index=False)}
    raw_by_id = {str(r.query_id): r for r in raw.itertuples(index=False)}
    expected_keys = {
        (str(query_id), baseline) for query_id in scores.query_id.unique() for baseline in target.BASELINES
    }
    if set(zip(observed.query_id.astype(str), observed.baseline_lane.astype(str))) != expected_keys:
        raise DiagnosticVerificationError("comparison inventory differs")
    for row in observed.itertuples(index=False):
        model = by_key[(str(row.query_id), "composite")]
        baseline = by_key[(str(row.query_id), str(row.baseline_lane))]
        source = raw_by_id[str(row.query_id)]
        checks = {
            "composite_brier": model.multiclass_brier, "baseline_brier": baseline.multiclass_brier,
            "composite_log_loss": model.multiclass_log_loss, "baseline_log_loss": baseline.multiclass_log_loss,
            "brier_difference": model.multiclass_brier - baseline.multiclass_brier,
            "log_loss_difference": model.multiclass_log_loss - baseline.multiclass_log_loss,
            "route_status": model.route_status, "fold_id": model.fold_id,
            "risk_score": model.risk_score, "effective_rows": model.effective_rows,
            "neighborhood_instability": source.neighborhood_instability,
        }
        if any(not _close(getattr(row, name), expected) for name, expected in checks.items()):
            raise DiagnosticVerificationError(f"comparison formula differs: {row.query_id}:{row.baseline_lane}")


def _scoped(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "validation_pooled": return frame.loc[frame.fold_id.isin(target.nonfinal.VALIDATION_FOLDS)]
    return frame.loc[frame.fold_id == scope]


def _verify_segments(comparisons: pd.DataFrame, observed: pd.DataFrame) -> None:
    usable = comparisons.loc[comparisons.multiclass_evaluable]
    columns = {
        "overall": None, "true_route": "route_status", "calendar_year": "calendar_year",
        "benchmark_regime": "query_regime", "quality_tier": "quality_tier",
        "liquidity_stratum": "liquidity_stratum", "quality_liquidity_cell": "quality_liquidity_cell",
        "dynamic_selectivity": "research_dynamic_selective", "distance_quintile": "distance_quintile",
        "effective_sample_quintile": "effective_sample_quintile", "instability_quintile": "instability_quintile",
    }
    for row in observed.itertuples(index=False):
        group = _scoped(usable, str(row.scope)); group = group.loc[group.baseline_lane == row.baseline_lane]
        column = columns[str(row.dimension)]
        if column is not None: group = group.loc[group[column].astype(str) == str(row.value)]
        model = group.composite_brier.mean(); baseline = group.baseline_brier.mean()
        checks = {
            "query_pairs": len(group), "month_pairs": group.month.nunique(),
            "composite_mean_brier": model, "baseline_mean_brier": baseline,
            "mean_brier_difference": model - baseline, "brier_skill": 1 - model / baseline,
            "composite_improved_fraction": (group.brier_difference < 0).mean(),
            "mean_log_loss_difference": group.log_loss_difference.mean(),
        }
        if len(group) < target.MINIMUM_SEGMENT_ROWS or any(
            not _close(getattr(row, name), expected) for name, expected in checks.items()
        ):
            raise DiagnosticVerificationError(f"segment formula differs: {row.scope}:{row.dimension}:{row.value}")


def _auc(probability: pd.Series, truth: pd.Series) -> float:
    positives = int(truth.sum()); negatives = len(truth) - positives
    if not positives or not negatives: return float("nan")
    ranks = probability.rank(method="average").to_numpy(float)
    return float((ranks[truth.to_numpy(bool)].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _verify_behavior(scores: pd.DataFrame, observed: pd.DataFrame) -> None:
    usable = scores.loc[scores.multiclass_evaluable]
    probability_column = dict(zip(target.CLASSES, (
        "favorable_probability", "adverse_probability", "no_touch_probability",
    )))
    for row in observed.itertuples(index=False):
        group = _scoped(usable, str(row.scope)); group = group.loc[group.lane == row.lane]
        truth = group.route_status.eq(row.class_name); probability = group[probability_column[row.class_name]].astype(float)
        checks = {
            "rows": len(group), "observed_frequency": truth.mean(), "mean_probability": probability.mean(),
            "probability_bias": probability.mean() - truth.mean(),
            "class_brier_component": np.square(probability - truth.astype(float)).mean(),
            "probability_standard_deviation": probability.std(ddof=0),
            "one_vs_rest_auc": _auc(probability, truth),
        }
        if any(not _close(getattr(row, name), expected) for name, expected in checks.items()):
            raise DiagnosticVerificationError(f"probability behavior differs: {row.scope}:{row.lane}:{row.class_name}")


def _verify_monthly(comparisons: pd.DataFrame, observed: pd.DataFrame) -> None:
    usable = comparisons.loc[comparisons.multiclass_evaluable]
    expected = usable.groupby(["fold_id", "month", "baseline_lane"], sort=True).agg(
        query_pairs=("query_id", "size"), mean_brier_difference=("brier_difference", "mean"),
        composite_improved_fraction=("brier_difference", lambda x: (x < 0).mean()),
    ).reset_index().sort_values(["month", "baseline_lane"], kind="stable").reset_index(drop=True)
    if list(expected.columns) != list(observed.columns) or len(expected) != len(observed):
        raise DiagnosticVerificationError("monthly inventory differs")
    for left, right in zip(expected.itertuples(index=False), observed.itertuples(index=False)):
        if any(not _close(a, b) for a, b in zip(left, right)):
            raise DiagnosticVerificationError("monthly formula differs")


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repository,
                      text=True, capture_output=True, check=True).stdout:
        raise DiagnosticVerificationError("clean worktree required")
    started = perf_counter(); root = repository / target.OUTPUT_RELATIVE
    seal = base._read(root / "SEALED.json"); summary = base._read(root / "SUMMARY.json")
    if not _valid(seal, timing=True) or not _valid(summary) or seal.get("file_manifest") != target._manifest(root):
        raise DiagnosticVerificationError("diagnostic seals differ")
    comparisons = pd.read_parquet(root / "query-comparisons.parquet")
    segments = pd.read_parquet(root / "segment-performance.parquet")
    behavior = pd.read_parquet(root / "probability-behavior.parquet")
    monthly = pd.read_parquet(root / "monthly-stability.parquet")
    _verify_comparisons(repository, comparisons); _verify_segments(comparisons, segments)
    scores, _ = _source_scores(repository); _verify_behavior(scores, behavior); _verify_monthly(comparisons, monthly)
    if summary.get("query_count") != comparisons.query_id.nunique() \
            or summary.get("new_model_validation_performed") is not False \
            or summary.get("production_promotion_authorized") is not False:
        raise DiagnosticVerificationError("summary boundary differs")
    state = {
        "schema_version": SCHEMA, "status": "verified", "passed": True,
        "store_result_digest": seal["result_digest"], "store_result_sha256": _sha(root / "SEALED.json"),
        "verified_query_comparison_rows": len(comparisons), "verified_segment_rows": len(segments),
        "verified_probability_behavior_rows": len(behavior), "verified_monthly_rows": len(monthly),
        "gates": {"all_query_loss_pairs_reconstructed": True, "all_segment_aggregates_reconstructed": True,
                  "all_probability_behavior_reconstructed": True, "all_monthly_aggregates_reconstructed": True,
                  "posthoc_nonpromotion_boundary_retained": True, "all_physical_seals_valid": True},
        "posthoc_diagnosis_only": True, "new_model_validation_performed": False,
        "production_promotion_authorized": False, "elapsed_seconds": perf_counter() - started,
        "created_at": _now(),
    }
    result = {**state, "verification_digest": stable_hash(state)}
    output = repository / OUTPUT_RELATIVE
    if output.exists():
        existing = base._read(output / "VERIFIED.json")
        ignored = {"created_at", "elapsed_seconds", "verification_digest"}
        if {k: v for k, v in existing.items() if k not in ignored} != {k: v for k, v in result.items() if k not in ignored}:
            raise DiagnosticVerificationError("existing verification differs")
        return existing
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        target.smoke._atomic_json(temporary / "VERIFIED.json", result); os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True); raise
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv); print(json.dumps(execute(args.repository), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
