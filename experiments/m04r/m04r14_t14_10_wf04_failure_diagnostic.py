"""Describe why verified WF-04 V1 failed without tuning or rescuing it."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from market_analogues.types import stable_hash

from experiments.m04r import m04r14_t14_09_outcome_smoke as smoke
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_10_wf03d_prediction_store as predictions
from experiments.m04r import m04r14_t14_10_wf04_final_evaluation as final
from experiments.m04r import m04r14_t14_10_wf04_nonfinal_evaluation as nonfinal


SCHEMA = "m04r14-t14-10-wf04-failure-diagnostic-v1"
OUTPUT_RELATIVE = Path("config/data/analogues/m04r14/t14-10-wf04-failure-diagnostic-v1")
OUTPUT_FILES = (
    "query-comparisons.parquet", "segment-performance.parquet",
    "probability-behavior.parquet", "monthly-stability.parquet",
    "SUMMARY.json", "index.html",
)
BASELINES = nonfinal.COMPARISON_BASELINES
CLASSES = nonfinal.PRIMARY_CLASSES
MINIMUM_SEGMENT_ROWS = 30


class FailureDiagnosticError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20): digest.update(block)
    return digest.hexdigest()


def _seal(value: Mapping[str, Any], *, timing: bool = False) -> dict[str, Any]:
    omitted = {"elapsed_seconds"} if timing else set()
    result = dict(value); result["result_digest"] = stable_hash({k: v for k, v in result.items() if k not in omitted})
    result["created_at"] = _now(); return result


def _valid(value: Mapping[str, Any], *, timing: bool = False) -> bool:
    omitted = {"result_digest", "created_at"} | ({"elapsed_seconds"} if timing else set())
    return value.get("result_digest") == stable_hash({k: v for k, v in value.items() if k not in omitted})


def _manifest(root: Path) -> list[dict[str, Any]]:
    return [{"path": name, "bytes": (root / name).stat().st_size, "sha256": _sha(root / name)} for name in OUTPUT_FILES]


def _verified_inputs(repository: Path) -> dict[str, Any]:
    nr = repository / nonfinal.OUTPUT_RELATIVE; fr = repository / final.OUTPUT_RELATIVE
    nv = repository / nonfinal.VERIFICATION_RELATIVE / "VERIFIED.json"
    fv = repository / final.VERIFICATION_RELATIVE / "VERIFIED.json"
    ns, fs, nver, fver = map(base._read, (nr / "SEALED.json", fr / "SEALED.json", nv, fv))
    decision = base._read(fr / "DECISION.json")
    if not all((
        nonfinal._valid_seal(ns, timing=True), ns.get("passed") is True,
        final._valid(fs, timing=True), fs.get("passed") is True,
        nver.get("passed") is True, fver.get("passed") is True,
        fver.get("store_result_digest") == fs.get("result_digest"),
        decision.get("research_calibration_pass") is False,
        decision.get("production_promotion_authorized") is False,
        final._valid(decision),
    )):
        raise FailureDiagnosticError("verified failed WF-04 boundary differs")
    head = str(nonfinal._git(repository, "rev-parse", "HEAD"))
    if nonfinal._git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise FailureDiagnosticError("clean worktree required")
    runtime = (
        "experiments/m04r/m04r14_t14_10_wf04_failure_diagnostic.py",
        "experiments/m04r/verify_m04r14_t14_10_wf04_failure_diagnostic.py",
    )
    return {
        "nonfinal_result_digest": ns["result_digest"], "nonfinal_verification_digest": nver["verification_digest"],
        "final_result_digest": fs["result_digest"], "final_verification_digest": fver["verification_digest"],
        "decision_result_digest": decision["result_digest"],
        "nonfinal_scores_sha256": _sha(nr / "query-scores.parquet"),
        "final_scores_sha256": _sha(fr / "query-scores.parquet"),
        "raw_predictions_sha256": _sha(repository / predictions.OUTPUT_RELATIVE / "raw-predictions.parquet"),
        "implementation_head": head,
        "runtime_sha256": {name: _sha(repository / name) for name in runtime},
    }


def _scores(repository: Path) -> pd.DataFrame:
    non = pd.read_parquet(repository / nonfinal.OUTPUT_RELATIVE / "query-scores.parquet")
    non = non.loc[non.purged_evaluation_included]
    fin = pd.read_parquet(repository / final.OUTPUT_RELATIVE / "query-scores.parquet")
    result = pd.concat([non, fin], ignore_index=True)
    if result.duplicated(["query_id", "lane"]).any() or set(result.lane) != set(nonfinal.PROBABILITY_LANES):
        raise FailureDiagnosticError("score inventory differs")
    return result


def _diagnostic_features(repository: Path, query_ids: set[str]) -> pd.DataFrame:
    raw = pd.read_parquet(repository / predictions.OUTPUT_RELATIVE / "raw-predictions.parquet")
    raw = raw.loc[(raw.prefix == 20) & (raw.lane == "composite") & raw.query_id.isin(query_ids), [
        "query_id", "neighborhood_instability", "novelty_threshold", "favorable_rows",
        "adverse_rows", "no_touch_rows", "ambiguous_rows", "censored_rows", "unavailable_rows",
    ]]
    if raw.query_id.duplicated().any() or len(raw) != len(query_ids):
        raise FailureDiagnosticError("diagnostic prediction inventory differs")
    return raw


def _quintile(series: pd.Series) -> pd.Series:
    valid = series.notna() & np.isfinite(series.astype(float))
    result = pd.Series("unavailable", index=series.index, dtype=object)
    if valid.any():
        percentile = series.loc[valid].rank(method="first", pct=True)
        result.loc[valid] = np.minimum(5, np.ceil(percentile * 5).astype(int)).map(lambda x: f"Q{x}")
    return result


def query_comparisons(scores: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    composite = scores.loc[scores.lane == "composite"].merge(features, on="query_id", validate="one_to_one")
    metadata = [
        "query_id", "query_cutoff", "month", "calendar_year", "fold_id", "route_status",
        "query_regime", "quality_tier", "liquidity_stratum", "quality_liquidity_cell",
        "research_dynamic_selective", "risk_score", "eligible_rows", "effective_rows",
        "neighborhood_instability", "novelty_threshold", "favorable_rows", "adverse_rows",
        "no_touch_rows", "ambiguous_rows", "censored_rows", "unavailable_rows",
        "multiclass_evaluable", "multiclass_brier", "multiclass_log_loss",
    ]
    composite = composite[metadata].rename(columns={
        "multiclass_brier": "composite_brier", "multiclass_log_loss": "composite_log_loss",
    })
    rows = []
    for baseline in BASELINES:
        other = scores.loc[scores.lane == baseline, [
            "query_id", "multiclass_brier", "multiclass_log_loss",
        ]].rename(columns={"multiclass_brier": "baseline_brier", "multiclass_log_loss": "baseline_log_loss"})
        joined = composite.merge(other, on="query_id", validate="one_to_one")
        joined["baseline_lane"] = baseline
        joined["brier_difference"] = joined.composite_brier - joined.baseline_brier
        joined["log_loss_difference"] = joined.composite_log_loss - joined.baseline_log_loss
        rows.append(joined)
    result = pd.concat(rows, ignore_index=True)
    for name, column in (
        ("distance_quintile", "risk_score"), ("effective_sample_quintile", "effective_rows"),
        ("instability_quintile", "neighborhood_instability"),
    ):
        result[name] = result.groupby(["fold_id", "baseline_lane"], sort=True)[column].transform(_quintile)
    return result.sort_values(["query_id", "baseline_lane"], kind="stable").reset_index(drop=True)


def _scope_frames(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    result = [(fold, group) for fold, group in frame.groupby("fold_id", sort=True)]
    result.append(("validation_pooled", frame.loc[frame.fold_id.isin(nonfinal.VALIDATION_FOLDS)]))
    return result


def segment_performance(comparisons: pd.DataFrame) -> pd.DataFrame:
    evaluable = comparisons.loc[comparisons.multiclass_evaluable].copy()
    dimensions = (
        ("overall", None), ("true_route", "route_status"), ("calendar_year", "calendar_year"),
        ("benchmark_regime", "query_regime"), ("quality_tier", "quality_tier"),
        ("liquidity_stratum", "liquidity_stratum"), ("quality_liquidity_cell", "quality_liquidity_cell"),
        ("dynamic_selectivity", "research_dynamic_selective"), ("distance_quintile", "distance_quintile"),
        ("effective_sample_quintile", "effective_sample_quintile"),
        ("instability_quintile", "instability_quintile"),
    )
    rows: list[dict[str, Any]] = []
    for scope, scoped in _scope_frames(evaluable):
        for baseline, paired in scoped.groupby("baseline_lane", sort=True):
            for dimension, column in dimensions:
                groups = [("all", paired)] if column is None else paired.groupby(column, sort=True, dropna=False)
                for value, group in groups:
                    if len(group) < MINIMUM_SEGMENT_ROWS: continue
                    base_loss = float(group.baseline_brier.mean()); model_loss = float(group.composite_brier.mean())
                    rows.append({
                        "scope": scope, "baseline_lane": baseline, "dimension": dimension,
                        "value": str(value), "query_pairs": len(group),
                        "month_pairs": group.month.nunique(), "composite_mean_brier": model_loss,
                        "baseline_mean_brier": base_loss, "mean_brier_difference": model_loss - base_loss,
                        "brier_skill": 1 - model_loss / base_loss,
                        "composite_improved_fraction": float((group.brier_difference < 0).mean()),
                        "mean_log_loss_difference": float(group.log_loss_difference.mean()),
                    })
    return pd.DataFrame(rows).sort_values(
        ["scope", "baseline_lane", "dimension", "value"], kind="stable",
    ).reset_index(drop=True)


def _auc(probability: pd.Series, truth: pd.Series) -> float:
    positives = int(truth.sum()); negatives = len(truth) - positives
    if positives == 0 or negatives == 0: return float("nan")
    ranks = probability.rank(method="average").to_numpy(dtype=float)
    return float((ranks[truth.to_numpy(dtype=bool)].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def probability_behavior(scores: pd.DataFrame) -> pd.DataFrame:
    usable = scores.loc[scores.multiclass_evaluable].copy()
    rows: list[dict[str, Any]] = []
    for scope, scoped in _scope_frames(usable):
        for lane, group in scoped.groupby("lane", sort=True):
            for class_name, column in zip(CLASSES, (
                "favorable_probability", "adverse_probability", "no_touch_probability",
            )):
                truth = group.route_status.eq(class_name); probability = group[column].astype(float)
                rows.append({
                    "scope": scope, "lane": lane, "class_name": class_name, "rows": len(group),
                    "observed_frequency": float(truth.mean()), "mean_probability": float(probability.mean()),
                    "probability_bias": float(probability.mean() - truth.mean()),
                    "class_brier_component": float(np.square(probability - truth.astype(float)).mean()),
                    "probability_standard_deviation": float(probability.std(ddof=0)),
                    "one_vs_rest_auc": _auc(probability, truth),
                })
    return pd.DataFrame(rows).sort_values(["scope", "lane", "class_name"], kind="stable").reset_index(drop=True)


def monthly_stability(comparisons: pd.DataFrame) -> pd.DataFrame:
    usable = comparisons.loc[comparisons.multiclass_evaluable]
    rows = []
    for (fold, month, baseline), group in usable.groupby(["fold_id", "month", "baseline_lane"], sort=True):
        rows.append({
            "fold_id": fold, "month": month, "baseline_lane": baseline, "query_pairs": len(group),
            "mean_brier_difference": group.brier_difference.mean(),
            "composite_improved_fraction": (group.brier_difference < 0).mean(),
        })
    return pd.DataFrame(rows).sort_values(["month", "baseline_lane"], kind="stable").reset_index(drop=True)


def summary(
    inputs: Mapping[str, Any], comparisons: pd.DataFrame, segments: pd.DataFrame,
    behavior: pd.DataFrame, monthly: pd.DataFrame,
) -> dict[str, Any]:
    overall = segments.loc[segments.dimension == "overall"]
    selected = overall.loc[overall.baseline_lane == "unconditional_market_frequency"].set_index("scope")
    classes = segments.loc[
        (segments.dimension == "true_route")
        & (segments.baseline_lane == "unconditional_market_frequency")
        & segments.scope.isin(("validation_pooled", final.FINAL_FOLD))
    ]
    final_behavior = behavior.loc[
        (behavior.scope == final.FINAL_FOLD)
        & behavior.lane.isin(("composite", "unconditional_market_frequency"))
    ]
    stability = monthly.loc[monthly.baseline_lane == "unconditional_market_frequency"]
    return _seal({
        "schema_version": SCHEMA, "status": "posthoc_diagnosis_complete",
        "verified_inputs": dict(inputs), "query_count": comparisons.query_id.nunique(),
        "evaluable_comparison_rows": int(comparisons.multiclass_evaluable.sum()),
        "minimum_segment_rows": MINIMUM_SEGMENT_ROWS,
        "composite_vs_unconditional_by_scope": selected[[
            "query_pairs", "mean_brier_difference", "brier_skill", "composite_improved_fraction",
        ]].to_dict("index"),
        "classwise_composite_vs_unconditional": classes[[
            "scope", "value", "query_pairs", "mean_brier_difference", "brier_skill",
        ]].to_dict("records"),
        "final_probability_behavior": final_behavior[[
            "lane", "class_name", "observed_frequency", "mean_probability", "probability_bias",
            "probability_standard_deviation", "one_vs_rest_auc",
        ]].to_dict("records"),
        "months_composite_better_than_unconditional": int((stability.mean_brier_difference < 0).sum()),
        "months_compared_to_unconditional": len(stability),
        "diagnosis": [
            "composite_conditional_adjustments_do_not_add_stable_resolution_over_expanding_base_rates",
            "analogue_probabilities_are_sharper_but_excess_loss_changes_sign_across_folds_and_true_classes",
            "rare_no_touch_cases_often_benefit_but_the_gain_does_not_offset_losses_on_common_routes",
            "posthoc_segments_are_hypothesis_generating_only_and_cannot_rescue_or_validate_V1",
        ],
        "v1_research_calibration_pass": False, "new_model_validation_performed": False,
        "production_promotion_authorized": False,
    })


def _html(summary_value: Mapping[str, Any], segments: pd.DataFrame, behavior: pd.DataFrame) -> str:
    overall = segments.loc[(segments.dimension == "overall") & segments.scope.isin((*nonfinal.VALIDATION_FOLDS, "validation_pooled", final.FINAL_FOLD))]
    rows = "".join("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in (
        r.scope, r.baseline_lane, r.query_pairs, f"{r.composite_mean_brier:.6f}",
        f"{r.baseline_mean_brier:.6f}", f"{100*r.brier_skill:.2f}%", f"{100*r.composite_improved_fraction:.1f}%",
    )) + "</tr>" for r in overall.itertuples(index=False))
    probs = behavior.loc[(behavior.scope == final.FINAL_FOLD) & behavior.lane.isin(("composite", "unconditional_market_frequency"))]
    probability_rows = "".join("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in (
        r.lane, r.class_name, f"{100*r.observed_frequency:.1f}%", f"{100*r.mean_probability:.1f}%",
        f"{100*r.probability_bias:+.1f}pp", f"{r.one_vs_rest_auc:.3f}",
    )) + "</tr>" for r in probs.itertuples(index=False))
    return f"""<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>WF-04 failure diagnosis</title><style>body{{font-family:system-ui;max-width:1250px;margin:2rem auto;line-height:1.45}}table{{border-collapse:collapse;width:100%;margin-bottom:2rem}}th,td{{padding:.4rem;border-bottom:1px solid #ddd;text-align:left}}.warn{{background:#fff4dc;padding:.8rem;border-left:5px solid #b9770e}}.fail{{color:#a11}}</style></head><body>
<h1>WF-04 post-hoc failure diagnosis</h1><p class=\"warn\">This report explains a failed, consumed holdout. Cells are descriptive and cannot validate a threshold, segment, strategy or V2.</p>
<p class=\"fail\"><b>Conclusion:</b> conditional analogue adjustments did not add stable resolution over expanding base rates. The method sometimes helped rare no-touch outcomes, but those gains did not offset unstable losses on common adverse/favorable routes.</p>
<p>Composite beat unconditional in {summary_value['months_composite_better_than_unconditional']} of {summary_value['months_compared_to_unconditional']} evaluated calendar months. No model promotion is authorized.</p>
<h2>Paired Brier performance</h2><table><thead><tr><th>Scope</th><th>Baseline</th><th>Pairs</th><th>Composite</th><th>Baseline</th><th>Skill</th><th>Queries improved</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Final probability behavior</h2><table><thead><tr><th>Lane</th><th>Class</th><th>Observed</th><th>Predicted</th><th>Bias</th><th>AUC</th></tr></thead><tbody>{probability_rows}</tbody></table>
<h2>Interpretation</h2><ul>{''.join(f'<li>{escape(x)}</li>' for x in summary_value['diagnosis'])}</ul></body></html>"""


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True); root = repository / OUTPUT_RELATIVE
    inputs = _verified_inputs(repository)
    if root.exists():
        seal = base._read(root / "SEALED.json")
        if not _valid(seal, timing=True) or seal.get("file_manifest") != _manifest(root):
            raise FailureDiagnosticError("existing diagnostic differs")
        observed_inputs = _verified_inputs(repository)
        if {k: v for k, v in observed_inputs.items() if k != "implementation_head"} != {
            k: v for k, v in seal["verified_inputs"].items() if k != "implementation_head"
        }:
            raise FailureDiagnosticError("existing diagnostic inputs drifted")
        return seal
    scores = _scores(repository); features = _diagnostic_features(repository, set(scores.query_id))
    comparisons = query_comparisons(scores, features)
    segments = segment_performance(comparisons)
    behavior = probability_behavior(scores)
    monthly = monthly_stability(comparisons)
    summary_value = summary(inputs, comparisons, segments, behavior, monthly)
    frames = {
        "query-comparisons.parquet": comparisons, "segment-performance.parquet": segments,
        "probability-behavior.parquet": behavior, "monthly-stability.parquet": monthly,
    }
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        for name, frame in frames.items(): smoke._atomic_parquet(temporary / name, frame)
        smoke._atomic_json(temporary / "SUMMARY.json", summary_value)
        (temporary / "index.html").write_text(_html(summary_value, segments, behavior), encoding="utf-8")
        seal = _seal({
            "schema_version": SCHEMA, "status": "sealed", "passed": True,
            "verified_inputs": inputs, "query_comparison_rows": len(comparisons),
            "segment_rows": len(segments), "probability_behavior_rows": len(behavior),
            "monthly_rows": len(monthly), "summary_result_digest": summary_value["result_digest"],
            "file_manifest": _manifest(temporary), "elapsed_seconds": 0.0,
            "posthoc_diagnosis_only": True, "new_model_validation_performed": False,
            "production_promotion_authorized": False,
        }, timing=True)
        smoke._atomic_json(temporary / "SEALED.json", seal); os.replace(temporary, root); return seal
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True); raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv); print(json.dumps(execute(args.repository), indent=2, sort_keys=True, allow_nan=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
