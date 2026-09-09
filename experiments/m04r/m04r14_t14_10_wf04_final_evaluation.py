"""Open the frozen WF-04 final holdout exactly once and publish its decision."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
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

from experiments.m04r import m04r14_t14_09_outcome_smoke as smoke
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_10_wf03d_prediction_store as predictions
from experiments.m04r import m04r14_t14_10_wf04_nonfinal_evaluation as nonfinal


SCHEMA = "m04r14-t14-10-wf04-final-preregistration-v1"
STORE_SCHEMA = "m04r14-t14-10-wf04-final-evaluation-v1"
PREREGISTRATION_RELATIVE = Path(
    "experiments/m04r/m04r14_t14_10_wf04_final_evaluation_v1_preregistered.json"
)
OUTPUT_RELATIVE = Path("config/data/analogues/m04r14/t14-10-wf04-final-evaluation-v1")
CACHE_RELATIVE = Path("config/data/analogues/m04r14/t14-10-wf04-final-open-v1-cache")
VERIFICATION_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf04-final-evaluation-v1-verification"
)
NONFINAL_ROOT = nonfinal.OUTPUT_RELATIVE
NONFINAL_VERIFICATION = nonfinal.VERIFICATION_RELATIVE / "VERIFIED.json"
FINAL_FOLD = nonfinal.FINAL_FOLD
FINAL_START = predictions.FINAL_START
EXPECTED_FINAL_QUERIES = predictions.EXPECTED_FINAL_QUERIES
OUTPUT_FILES = (
    "final-query-outcomes.parquet", "final-query-paths.parquet",
    "query-scores.parquet", "continuous-scores.parquet", "fold-metrics.parquet",
    "stability-metrics.parquet", "calibration-bins.parquet", "risk-coverage.parquet",
    "inference.parquet", "interval-coverage.parquet", "path-metrics.parquet",
    "COVERAGE.json", "DECISION.json", "index.html",
)
RUNTIME_FILES = (
    "experiments/m04r/m04r14_t14_10_wf04_final_evaluation.py",
    "experiments/m04r/verify_m04r14_t14_10_wf04_final_evaluation.py",
    "experiments/m04r/m04r14_t14_10_wf04_nonfinal_evaluation.py",
    "experiments/m04r/verify_m04r14_t14_10_wf04_nonfinal_evaluation.py",
    "experiments/m04r/m04r14_t14_10_wf03d_prediction_store.py",
    "src/market_analogues/walk_forward_evaluation.py",
    "src/market_analogues/walk_forward_scoring.py",
    "config/m04r14-t14-10-walk-forward-contract.json",
    "pyproject.toml",
)


class FinalEvaluationError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(repository: Path, *args: str, raw: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=repository, capture_output=True, text=not raw, check=False,
    )
    if result.returncode:
        error = result.stderr if not raw else result.stderr.decode(errors="replace")
        raise FinalEvaluationError(error.strip() or "git command failed")
    return result.stdout if raw else result.stdout.strip()


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise FinalEvaluationError(f"regular file required: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _seal(state: Mapping[str, Any], key: str = "result_digest", *, timing: bool = False) -> dict[str, Any]:
    omitted = {"elapsed_seconds"} if timing else set()
    result = dict(state)
    result[key] = stable_hash({name: value for name, value in result.items() if name not in omitted})
    result["created_at"] = _now()
    return result


def _valid(value: Mapping[str, Any], key: str = "result_digest", *, timing: bool = False) -> bool:
    omitted = {key, "created_at"} | ({"elapsed_seconds"} if timing else set())
    return value.get(key) == stable_hash({name: item for name, item in value.items() if name not in omitted})


def _manifest(root: Path, names: Sequence[str]) -> list[dict[str, Any]]:
    return [{"path": name, "bytes": (root / name).stat().st_size, "sha256": _sha(root / name)} for name in names]


def _runtime_manifest(repository: Path, head: str) -> dict[str, str]:
    tracked = set(str(_git(repository, "ls-tree", "-r", "--name-only", head)).splitlines())
    if any(name not in tracked for name in RUNTIME_FILES):
        raise FinalEvaluationError("runtime manifest contains uncommitted files")
    return {
        name: sha256(_git(repository, "show", f"{head}:{name}", raw=True)).hexdigest()
        for name in RUNTIME_FILES
    }


def _verified_inputs(repository: Path) -> dict[str, Any]:
    prediction_seal_path = repository / predictions.OUTPUT_RELATIVE / "SEALED.json"
    prediction_seal = base._read(prediction_seal_path)
    prediction_verification_path = repository / predictions.VERIFICATION_RELATIVE / "VERIFIED.json"
    prediction_verification = base._read(prediction_verification_path)
    nonfinal_seal_path = repository / NONFINAL_ROOT / "SEALED.json"
    nonfinal_seal = base._read(nonfinal_seal_path)
    nonfinal_verification_path = repository / NONFINAL_VERIFICATION
    nonfinal_verification = base._read(nonfinal_verification_path)
    if not all((
        _valid(prediction_seal, timing=True), prediction_seal.get("passed") is True,
        prediction_seal.get("final_period_result_opened") is False,
        prediction_verification.get("passed") is True,
        prediction_verification.get("final_period_result_opened") is False,
        nonfinal._valid_seal(nonfinal_seal, timing=True), nonfinal_seal.get("passed") is True,
        nonfinal_seal.get("final_period_result_opened") is False,
        nonfinal_verification.get("passed") is True,
        nonfinal_verification.get("final_single_open_authorized") is True,
        nonfinal_verification.get("final_period_result_opened") is False,
        nonfinal_verification.get("store_result_digest") == nonfinal_seal.get("result_digest"),
        nonfinal_verification.get("verification_digest") == stable_hash({
            key: value for key, value in nonfinal_verification.items() if key != "verification_digest"
        }),
    )):
        raise FinalEvaluationError("verified prediction/non-final boundary differs")
    prediction_root = repository / predictions.OUTPUT_RELATIVE
    return {
        "prediction_store_result_digest": prediction_seal["result_digest"],
        "prediction_store_sha256": _sha(prediction_seal_path),
        "prediction_verification_digest": prediction_verification["verification_digest"],
        "prediction_verification_sha256": _sha(prediction_verification_path),
        "nonfinal_store_result_digest": nonfinal_seal["result_digest"],
        "nonfinal_store_sha256": _sha(nonfinal_seal_path),
        "nonfinal_verification_digest": nonfinal_verification["verification_digest"],
        "nonfinal_verification_sha256": _sha(nonfinal_verification_path),
        "raw_predictions_sha256": _sha(prediction_root / "raw-predictions.parquet"),
        "baseline_predictions_sha256": _sha(prediction_root / "baseline-predictions.parquet"),
        "continuous_predictions_sha256": _sha(prediction_root / "continuous-predictions.parquet"),
        "path_predictions_sha256": _sha(prediction_root / "path-predictions.parquet"),
        "source_fingerprints_digest": stable_hash(predictions._source_fingerprints(repository)),
    }


def _assert_final_unopened(repository: Path) -> None:
    if (repository / OUTPUT_RELATIVE).exists() or (repository / CACHE_RELATIVE).exists():
        raise FinalEvaluationError("final outcome namespace already exists")
    expected = set((*predictions.PREDICTION_FILES, "PREDICTIONS_SEALED.json"))
    cache = repository / predictions.CACHE_RELATIVE
    final_months = sorted(cache.glob("month-2024-*")) + sorted(cache.glob("month-2025-*"))
    if len(final_months) != 20 or any(
        path.is_symlink() or not path.is_dir() or {child.name for child in path.iterdir()} != expected
        for path in final_months
    ):
        raise FinalEvaluationError("D4 final prediction-only boundary differs")


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise FinalEvaluationError("globally clean Git worktree required")
    _assert_final_unopened(repository)
    h0 = str(_git(repository, "rev-parse", "HEAD"))
    registry = nonfinal._registry(repository)
    final = registry.loc[registry.fold_id == FINAL_FOLD]
    if len(final) != EXPECTED_FINAL_QUERIES or final.query_id.duplicated().any():
        raise FinalEvaluationError("final registry inventory differs")
    contract = base._read(repository / nonfinal.CONTRACT_RELATIVE)
    state = {
        "schema_version": SCHEMA,
        "status": "frozen_immediately_before_single_final_open",
        "implementation_h0": h0,
        "runtime_files": _runtime_manifest(repository, h0),
        "verified_inputs": _verified_inputs(repository),
        "walk_forward_contract_digest": contract["contract_digest"],
        "final_query_count": len(final),
        "final_query_ids_digest": stable_hash(sorted(final.query_id.astype(str))),
        "final_months": sorted(final.month.unique().tolist()),
        "final_outcome_namespace_absent": True,
        "score_formulas": "unchanged_verified_WF04_nonfinal_kernel",
        "primary_lane": "composite_forced_score",
        "final_inference_scope": "20_calendar_month_mean_losses",
        "holm_final_family": list(nonfinal.COMPARISON_BASELINES),
        "significance_gate_interpretation": "both_bootstrap_and_DM_Holm_pvalues_below_0.05_vs_unconditional_and_price_only",
        "interval_gate_interpretation": "no_Holm_rejection_for_any_composite_final_interval_test",
        "literal_minimum_nonabstained_fraction": 0.5,
        "minimum_final_evaluable_queries": 400,
        "nonfinal_failures_are_irreversible": True,
        "representation_threshold_neighbor_and_prediction_changes": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
    }
    return _seal(state, "preregistration_digest")


def _sole_child(repository: Path, raw: bytes, h0: str) -> str:
    accepted: list[str] = []
    for line in str(_git(repository, "rev-list", "--all", "--children")).splitlines():
        values = line.split()
        if not values or values[0] != h0:
            continue
        for child in values[1:]:
            parents = str(_git(repository, "rev-list", "--parents", "-n", "1", child)).split()
            changed = str(_git(repository, "diff-tree", "--no-commit-id", "--name-only", "-r", child)).splitlines()
            if parents == [child, h0] and changed == [PREREGISTRATION_RELATIVE.as_posix()] \
                    and _git(repository, "show", f"{child}:{PREREGISTRATION_RELATIVE}", raw=True) == raw:
                accepted.append(child)
    if len(set(accepted)) != 1:
        raise FinalEvaluationError("expected one exact preregistration-only child")
    return accepted[0]


def validate_preregistration(repository: Path) -> tuple[dict[str, Any], pd.DataFrame, str]:
    if _git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise FinalEvaluationError("globally clean Git worktree required")
    raw = (repository / PREREGISTRATION_RELATIVE).read_bytes()
    prereg = base._read(repository / PREREGISTRATION_RELATIVE)
    if prereg.get("schema_version") != SCHEMA or not _valid(prereg, "preregistration_digest"):
        raise FinalEvaluationError("final preregistration differs")
    h0 = str(prereg["implementation_h0"])
    h1 = _sole_child(repository, raw, h0)
    if subprocess.run(["git", "merge-base", "--is-ancestor", h1, "HEAD"], cwd=repository).returncode:
        raise FinalEvaluationError("HEAD does not descend from final preregistration")
    for name, expected in prereg["runtime_files"].items():
        if _sha(repository / name) != expected \
                or sha256(_git(repository, "show", f"{h0}:{name}", raw=True)).hexdigest() != expected:
            raise FinalEvaluationError(f"runtime source drifted: {name}")
    if prereg["verified_inputs"] != _verified_inputs(repository):
        raise FinalEvaluationError("verified inputs drifted")
    registry = nonfinal._registry(repository)
    final = registry.loc[registry.fold_id == FINAL_FOLD].copy()
    if len(final) != prereg["final_query_count"] \
            or stable_hash(sorted(final.query_id.astype(str))) != prereg["final_query_ids_digest"]:
        raise FinalEvaluationError("final registry drifted")
    return prereg, final, h1


def _month_manifest(root: Path) -> list[dict[str, Any]]:
    return _manifest(root, ("query-outcomes.parquet", "query-paths.parquet"))


def _open_final_outcomes(
    repository: Path, registry: pd.DataFrame, prereg: Mapping[str, Any], h1: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = repository / CACHE_RELATIVE
    if not cache.exists():
        cache.mkdir(parents=True)
        smoke._atomic_json(cache / "OPEN_STARTED.json", _seal({
            "schema_version": STORE_SCHEMA, "status": "final_open_started",
            "preregistration_digest": prereg["preregistration_digest"],
            "preregistration_h1": h1, "final_period_result_opened": True,
        }))
    started = base._read(cache / "OPEN_STARTED.json")
    if not _valid(started) or started.get("preregistration_digest") != prereg["preregistration_digest"]:
        raise FinalEvaluationError("final-open cache boundary differs")
    fingerprints = predictions._source_fingerprints(repository)
    regimes = predictions._benchmark_regimes(repository, nonfinal._registry(repository))
    regime_map = dict(zip(regimes.month.astype(str), regimes.regime.astype(str)))
    for month, rows in registry.groupby("month", sort=True):
        month_root = cache / f"month-{month}"
        seal_path = month_root / "CLOSED.json"
        if seal_path.exists():
            seal = base._read(seal_path)
            if not _valid(seal, timing=True) or seal.get("file_manifest") != _month_manifest(month_root):
                raise FinalEvaluationError(f"cached final month differs: {month}")
            continue
        if month_root.exists():
            raise FinalEvaluationError(f"partial unsealed final month requires inspection: {month}")
        began = perf_counter()
        outcomes, paths = predictions._compute_query_outcomes(
            repository, rows, regime_map[str(month)], fingerprints,
        )
        temporary = Path(tempfile.mkdtemp(prefix=f".month-{month}.", dir=cache))
        try:
            smoke._atomic_parquet(temporary / "query-outcomes.parquet", outcomes)
            smoke._atomic_parquet(temporary / "query-paths.parquet", paths)
            seal = _seal({
                "schema_version": STORE_SCHEMA, "status": "final_month_closed", "passed": True,
                "month": str(month), "query_count": len(rows), "outcome_rows": len(outcomes),
                "path_rows": len(paths), "file_manifest": _month_manifest(temporary),
                "elapsed_seconds": perf_counter() - began, "final_period_result_opened": True,
            }, timing=True)
            smoke._atomic_json(temporary / "CLOSED.json", seal)
            os.replace(temporary, month_root)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    outcomes = pd.concat([
        pd.read_parquet(path / "query-outcomes.parquet", engine="pyarrow")
        for path in sorted(cache.glob("month-*"))
    ], ignore_index=True).sort_values(["query_id", "horizon_sessions"], kind="stable").reset_index(drop=True)
    paths = pd.concat([
        pd.read_parquet(path / "query-paths.parquet", engine="pyarrow")
        for path in sorted(cache.glob("month-*"))
    ], ignore_index=True).sort_values(["query_id", "step"], kind="stable").reset_index(drop=True)
    if outcomes.query_id.nunique() != EXPECTED_FINAL_QUERIES or paths.query_id.nunique() != EXPECTED_FINAL_QUERIES:
        raise FinalEvaluationError("final outcome inventory differs")
    return outcomes, paths


def _score_final(
    raw: pd.DataFrame, baselines: pd.DataFrame, forecasts: pd.DataFrame,
    outcomes: pd.DataFrame, registry: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fake_fold = "validation_3"
    raw = raw.copy(); baselines = baselines.copy(); forecasts = forecasts.copy(); registry = registry.copy()
    raw["fold_id"] = fake_fold; baselines["fold_id"] = fake_fold
    forecasts["fold_id"] = fake_fold; registry["fold_id"] = fake_fold
    cutoffs = {fake_fold: None}
    scores = nonfinal._probability_scores(raw, baselines, outcomes, registry, cutoffs)
    continuous = nonfinal._continuous_scores(forecasts, outcomes, registry, cutoffs)
    scores["fold_id"] = FINAL_FOLD; continuous["fold_id"] = FINAL_FOLD
    scores["purged_evaluation_included"] = True
    continuous["purged_evaluation_included"] = True
    return scores, continuous


def _final_scope(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[frame.scope == "validation_3"].copy()
    result["scope"] = FINAL_FOLD
    if "value" in result and "dimension" in result:
        result.loc[(result.dimension == "fold") & (result.value == "validation_3"), "value"] = FINAL_FOLD
    return result.reset_index(drop=True)


def _complete_nonfinal_fold_surface(scores: pd.DataFrame) -> pd.DataFrame:
    """Feed legacy aggregate helpers all required scopes without changing the selected result."""
    copies = []
    for fold in nonfinal.SCORED_NONFINAL_FOLDS:
        copy = scores.copy()
        copy["fold_id"] = fold
        copy["query_id"] = copy.query_id.astype(str) + "|aggregate-adapter|" + fold
        copies.append(copy)
    return pd.concat(copies, ignore_index=True)


def _aggregates(
    scores: pd.DataFrame, continuous: pd.DataFrame, forecasts: pd.DataFrame,
    paths: pd.DataFrame, registry: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    fake_scores = scores.copy(); fake_scores["fold_id"] = "validation_3"
    complete_scores = _complete_nonfinal_fold_surface(scores)
    fake_continuous = continuous.copy(); fake_continuous["fold_id"] = "validation_3"
    fake_registry = registry.copy(); fake_registry["fold_id"] = "validation_3"
    fake_forecasts = forecasts.copy(); fake_forecasts["fold_id"] = "validation_3"
    return {
        "fold-metrics.parquet": _final_scope(nonfinal._fold_metrics(fake_scores)),
        "stability-metrics.parquet": _final_scope(nonfinal._stability_metrics(fake_scores).assign(scope="validation_3")),
        "calibration-bins.parquet": _final_scope(nonfinal._calibration(fake_scores)),
        "risk-coverage.parquet": _final_scope(nonfinal._risk_coverage(complete_scores)),
        "inference.parquet": _final_scope(nonfinal._inference(complete_scores)),
        "interval-coverage.parquet": _final_scope(nonfinal._interval_coverage(fake_continuous)),
        "path-metrics.parquet": _final_scope(nonfinal._path_metrics(
            fake_forecasts, paths, fake_registry, {"validation_3": None},
        )),
    }


def _coverage(scores: pd.DataFrame, continuous: pd.DataFrame, prereg: Mapping[str, Any]) -> dict[str, Any]:
    primary = scores.loc[scores.lane == "composite"]
    return _seal({
        "schema_version": STORE_SCHEMA, "status": "final_coverage",
        "preregistration_digest": prereg["preregistration_digest"],
        "final_query_rows": len(primary),
        "multiclass_evaluable_queries": int(primary.multiclass_evaluable.sum()),
        "directional_evaluable_queries": int(primary.directional_evaluable.sum()),
        "route_status_counts": {str(k): int(v) for k, v in primary.route_status.value_counts(dropna=False).sort_index().items()},
        "literal_selective_queries": int(primary.literal_selective.sum()),
        "research_dynamic_selective_queries": int(primary.research_dynamic_selective.sum()),
        "literal_nonabstained_fraction": float(primary.literal_selective.mean()),
        "continuous_rows": len(continuous), "continuous_evaluable_rows": int(continuous.evaluable.sum()),
        "final_query_outcomes_opened": True,
    })


def _decision(
    repository: Path, metrics: Mapping[str, pd.DataFrame], coverage: Mapping[str, Any],
    prereg: Mapping[str, Any],
) -> dict[str, Any]:
    final_inference = metrics["inference.parquet"]
    final_folds = metrics["fold-metrics.parquet"]
    nonfinal_inference = pd.read_parquet(repository / NONFINAL_ROOT / "inference.parquet")
    nonfinal_folds = pd.read_parquet(repository / NONFINAL_ROOT / "fold-metrics.parquet")
    nonfinal_interval = pd.read_parquet(repository / NONFINAL_ROOT / "interval-coverage.parquet")
    final_interval = metrics["interval-coverage.parquet"]
    final_pairs = final_inference.set_index("baseline_lane")
    baseline_names = list(nonfinal.COMPARISON_BASELINES)
    positive_every = bool((final_pairs.loc[baseline_names, "brier_skill"] > 0).all())
    significance = bool(all(
        final_pairs.loc[lane, "bootstrap_lower_holm_pvalue"] < .05
        and final_pairs.loc[lane, "dm_lower_holm_pvalue"] < .05
        for lane in ("unconditional_market_frequency", "price_only")
    ))
    pooled = nonfinal_inference.loc[nonfinal_inference.scope == "validation_pooled"].set_index("baseline_lane")
    validation_positive = bool(all(
        pooled.loc[lane, "brier_skill"] > 0
        for lane in ("unconditional_market_frequency", "price_only")
    ))
    unconditional = nonfinal_inference.loc[
        (nonfinal_inference.baseline_lane == "unconditional_market_frequency")
        & nonfinal_inference.scope.isin(nonfinal.VALIDATION_FOLDS)
    ]
    no_negative_fold = bool((unconditional.brier_skill > 0).all() and len(unconditional) == 3)
    final_lane = final_folds.set_index("lane")
    log_not_worse = bool(
        final_lane.loc["composite", "mean_multiclass_log_loss"]
        <= final_lane.loc["unconditional_market_frequency", "mean_multiclass_log_loss"]
    )
    interval_columns = [
        "exact_marginal_holm_reject", "christoffersen_independence_holm_reject",
        "dynamic_binary_holm_reject",
    ]
    final_primary_interval = final_interval.loc[final_interval.lane == "composite"]
    interval_ok = bool(not final_primary_interval[interval_columns].to_numpy(dtype=bool).any())
    nonfinal_primary_interval = nonfinal_interval.loc[
        (nonfinal_interval.scope == "validation_pooled") & (nonfinal_interval.lane == "composite")
    ]
    nonfinal_interval_ok = bool(not nonfinal_primary_interval[interval_columns].to_numpy(dtype=bool).any())
    gates = {
        "final_brier_skill_positive_vs_every_baseline": positive_every,
        "final_Holm_one_sided_both_tests_vs_unconditional_and_price_only_below_0_05": significance,
        "validation_pooled_brier_skill_positive_vs_unconditional_and_price_only": validation_positive,
        "no_validation_fold_negative_brier_skill_vs_unconditional": no_negative_fold,
        "final_log_loss_not_worse_than_unconditional": log_not_worse,
        "nominal_composite_interval_coverage_not_rejected_after_Holm_final": interval_ok,
        "nominal_composite_interval_coverage_not_rejected_after_Holm_validation": nonfinal_interval_ok,
        "minimum_final_nonabstained_fraction_0_5": coverage["literal_nonabstained_fraction"] >= .5,
        "minimum_final_evaluable_queries_400": coverage["multiclass_evaluable_queries"] >= 400,
        "all_leakage_manifest_and_censor_checks_pass": True,
    }
    research_pass = bool(all(gates.values()))
    state = {
        "schema_version": STORE_SCHEMA, "status": "final_acceptance_decision",
        "preregistration_digest": prereg["preregistration_digest"], "gates": gates,
        "research_calibration_pass": research_pass,
        "product_calibration_pass": False,
        "product_failure_reason": "missing_point_in_time_sector_membership_delisting_and_event_cluster_controls",
        "literal_abstention_contract_contradiction_retained": True,
        "nonfinal_failure_irreversible": not (validation_positive and no_negative_fold),
        "final_period_result_opened": True,
        "production_promotion_authorized": False,
    }
    return _seal(state)


def _html(metrics: Mapping[str, pd.DataFrame], coverage: Mapping[str, Any], decision: Mapping[str, Any]) -> str:
    rows = []
    for row in metrics["fold-metrics.parquet"].itertuples(index=False):
        rows.append("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in (
            row.lane, row.multiclass_evaluable_rows,
            f"{row.mean_multiclass_brier:.8f}", f"{row.mean_multiclass_log_loss:.8f}",
        )) + "</tr>")
    gates = "".join(
        f"<li>{escape(name)}: <b>{'PASS' if passed else 'FAIL'}</b></li>"
        for name, passed in decision["gates"].items()
    )
    return f"""<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>WF-04 final evaluation</title>
<style>body{{font-family:system-ui;max-width:1200px;margin:2rem auto;line-height:1.45}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.45rem;border-bottom:1px solid #ddd;text-align:left}}.warn{{background:#fff4dc;padding:.8rem;border-left:5px solid #b9770e}}</style></head><body>
<h1>WF-04 final single-open evaluation</h1><p class=\"warn\">Research calibration pass: <b>{decision['research_calibration_pass']}</b>. Production promotion is not authorized.</p>
<p>Final queries: {coverage['final_query_rows']}; multiclass evaluable: {coverage['multiclass_evaluable_queries']}; literal non-abstained fraction: {coverage['literal_nonabstained_fraction']:.3f}.</p>
<h2>Acceptance gates</h2><ul>{gates}</ul><h2>Final proper scores</h2><table><thead><tr><th>Lane</th><th>Rows</th><th>Mean Brier</th><th>Mean log loss</th></tr></thead><tbody>{''.join(rows)}</tbody></table></body></html>"""


def _validate_existing(repository: Path, prereg: Mapping[str, Any], h1: str) -> dict[str, Any] | None:
    root = repository / OUTPUT_RELATIVE
    if not root.exists():
        return None
    expected = set((*OUTPUT_FILES, "SEALED.json"))
    if root.is_symlink() or not root.is_dir() or {path.name for path in root.iterdir()} != expected:
        raise FinalEvaluationError("existing final output layout differs")
    seal = base._read(root / "SEALED.json")
    if not _valid(seal, timing=True) or seal.get("preregistration_h1") != h1 \
            or seal.get("file_manifest") != _manifest(root, OUTPUT_FILES):
        raise FinalEvaluationError("existing final output seal differs")
    return seal


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    prereg, registry, h1 = validate_preregistration(repository)
    if existing := _validate_existing(repository, prereg, h1):
        return existing
    started = perf_counter()
    outcomes, paths = _open_final_outcomes(repository, registry, prereg, h1)
    prediction_root = repository / predictions.OUTPUT_RELATIVE
    final_ids = set(registry.query_id)
    raw = pd.read_parquet(prediction_root / "raw-predictions.parquet").loc[lambda x: x.query_id.isin(final_ids)]
    baselines = pd.read_parquet(prediction_root / "baseline-predictions.parquet").loc[lambda x: x.query_id.isin(final_ids)]
    forecasts = pd.read_parquet(prediction_root / "continuous-predictions.parquet").loc[lambda x: x.query_id.isin(final_ids)]
    path_forecasts = pd.read_parquet(prediction_root / "path-predictions.parquet").loc[lambda x: x.query_id.isin(final_ids)]
    scores, continuous = _score_final(raw, baselines, forecasts, outcomes, registry)
    metrics = _aggregates(scores, continuous, path_forecasts, paths, registry)
    coverage = _coverage(scores, continuous, prereg)
    decision = _decision(repository, metrics, coverage, prereg)
    frames = {
        "final-query-outcomes.parquet": outcomes, "final-query-paths.parquet": paths,
        "query-scores.parquet": scores, "continuous-scores.parquet": continuous, **metrics,
    }
    final = repository / OUTPUT_RELATIVE
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{final.name}.", dir=final.parent))
    try:
        for name, frame in frames.items():
            smoke._atomic_parquet(temporary / name, frame)
        smoke._atomic_json(temporary / "COVERAGE.json", coverage)
        smoke._atomic_json(temporary / "DECISION.json", decision)
        (temporary / "index.html").write_text(_html(metrics, coverage, decision), encoding="utf-8")
        state = _seal({
            "schema_version": STORE_SCHEMA, "status": "sealed", "passed": True,
            "preregistration_h1": h1, "preregistration_digest": prereg["preregistration_digest"],
            "prediction_store_result_digest": prereg["verified_inputs"]["prediction_store_result_digest"],
            "nonfinal_store_result_digest": prereg["verified_inputs"]["nonfinal_store_result_digest"],
            "final_query_count": len(registry), "outcome_rows": len(outcomes), "path_rows": len(paths),
            "query_score_rows": len(scores), "continuous_score_rows": len(continuous),
            "coverage_result_digest": coverage["result_digest"], "decision_result_digest": decision["result_digest"],
            "research_calibration_pass": decision["research_calibration_pass"],
            "file_manifest": _manifest(temporary, OUTPUT_FILES),
            "elapsed_seconds": perf_counter() - started, "final_period_result_opened": True,
            "independent_verification_authorized": True, "production_promotion_authorized": False,
        }, timing=True)
        smoke._atomic_json(temporary / "SEALED.json", state)
        os.replace(temporary, final)
        return state
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("preregister", "run"):
        child = sub.add_parser(command); child.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "preregister":
        value = build_preregistration(args.repository)
        smoke._atomic_json(args.repository / PREREGISTRATION_RELATIVE, value)
    else:
        value = execute(args.repository)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
