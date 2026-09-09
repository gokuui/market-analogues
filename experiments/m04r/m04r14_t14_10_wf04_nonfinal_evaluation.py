"""Preregister and evaluate only the purged non-final WF-04 folds."""
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

from market_analogues.adapters import file_fingerprint, source_from_spec
from market_analogues.config import load_config
from market_analogues.types import stable_hash
from market_analogues.walk_forward_evaluation import (
    DYNAMIC_ABSTENTION_REASONS,
    PERMANENT_PRODUCT_BLOCKERS,
    PRIMARY_CLASSES,
    continuous_score_record,
    fold_purge_cutoffs,
    probability_score_record,
    query_inside_purged_fold,
    split_abstention_reasons,
)
from market_analogues.walk_forward_scoring import (
    calendar_month_mean_losses,
    diebold_mariano_hac_lower_pvalue,
    equal_count_calibration,
    holm_adjust,
    interval_coverage_tests,
    moving_block_bootstrap_lower_pvalue,
    risk_coverage_curve,
)

from experiments.m04r import m04r14_t14_09_outcome_smoke as smoke
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_10_wf03d_prediction_store as predictions


SCHEMA = "m04r14-t14-10-wf04-nonfinal-preregistration-v1"
STORE_SCHEMA = "m04r14-t14-10-wf04-nonfinal-evaluation-v1"
PREREGISTRATION_RELATIVE = Path(
    "experiments/m04r/m04r14_t14_10_wf04_nonfinal_evaluation_v1_preregistered.json"
)
OUTPUT_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf04-nonfinal-evaluation-v1"
)
VERIFICATION_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf04-nonfinal-evaluation-v1-verification"
)
CONTRACT_RELATIVE = Path("config/m04r14-t14-10-walk-forward-contract.json")
SYNTHETIC_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-walk-forward-synthetic-scoring-v1/VERIFIED.json"
)
WF04_SYNTHETIC_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf04-nonfinal-synthetic-v1/VERIFIED.json"
)
REGISTRY_RELATIVE = predictions.REGISTRY_RELATIVE
PREDICTION_ROOT = predictions.OUTPUT_RELATIVE
PREDICTION_VERIFICATION = predictions.VERIFICATION_RELATIVE / "VERIFIED.json"
FINAL_FOLD = "final_untouched"
WARMUP_FOLD = "warmup"
SCORED_NONFINAL_FOLDS = ("development", "validation_1", "validation_2", "validation_3")
VALIDATION_FOLDS = ("validation_1", "validation_2", "validation_3")
PROBABILITY_LANES = (
    "composite", "composite_unweighted", "price_only", "deterministic_random",
    "recent_return_volatility", "unconditional_market_frequency", "regime_only_frequency",
)
CONTINUOUS_LANES = predictions.NEIGHBOR_LANES
COMPARISON_BASELINES = (
    "unconditional_market_frequency", "regime_only_frequency",
    "deterministic_random", "recent_return_volatility", "price_only",
)
CONTINUOUS_MEASURES = predictions.MEASURES
PATH_MEASURES = predictions.PATH_MEASURES
RUNTIME_FILES = (
    "experiments/m04r/m04r14_t14_10_wf04_nonfinal_evaluation.py",
    "experiments/m04r/verify_m04r14_t14_10_wf04_nonfinal_evaluation.py",
    "experiments/m04r/m04r14_t14_10_wf04_synthetic_gate.py",
    "src/market_analogues/walk_forward_evaluation.py",
    "src/market_analogues/walk_forward_scoring.py",
    "experiments/m04r/m04r14_t14_10_wf03d_prediction_store.py",
    "experiments/m04r/verify_m04r14_t14_10_wf03d_prediction_store.py",
    "config/m04r14-t14-10-walk-forward-contract.json",
    "config/m04r14-t14-10-synthetic-scoring-spec.json",
    "pyproject.toml",
)
OUTPUT_FILES = (
    "query-scores.parquet", "continuous-scores.parquet", "fold-metrics.parquet",
    "stability-metrics.parquet", "calibration-bins.parquet", "risk-coverage.parquet",
    "inference.parquet", "interval-coverage.parquet", "path-metrics.parquet",
    "COVERAGE.json", "index.html",
)


class WalkForwardEvaluationError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(repository: Path, *args: str, raw: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=repository, capture_output=True,
        text=not raw, check=False,
    )
    if result.returncode:
        error = result.stderr if not raw else result.stderr.decode(errors="replace")
        raise WalkForwardEvaluationError(error.strip() or "git command failed")
    return result.stdout if raw else result.stdout.strip()


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise WalkForwardEvaluationError(f"regular file required: {path}")
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


def _valid_seal(value: Mapping[str, Any], key: str = "result_digest", *, timing: bool = False) -> bool:
    omitted = {key, "created_at"} | ({"elapsed_seconds"} if timing else set())
    return value.get(key) == stable_hash({
        name: item for name, item in value.items() if name not in omitted
    })


def _runtime_manifest(repository: Path, head: str) -> dict[str, str]:
    tracked = set(str(_git(repository, "ls-tree", "-r", "--name-only", head)).splitlines())
    if any(name not in tracked for name in RUNTIME_FILES):
        raise WalkForwardEvaluationError("runtime manifest contains uncommitted files")
    return {
        name: sha256(_git(repository, "show", f"{head}:{name}", raw=True)).hexdigest()
        for name in RUNTIME_FILES
    }


def _manifest(root: Path, names: Sequence[str]) -> list[dict[str, Any]]:
    return [{"path": name, "bytes": (root / name).stat().st_size, "sha256": _sha(root / name)} for name in names]


def _verified_inputs(repository: Path) -> dict[str, Any]:
    root = repository / PREDICTION_ROOT
    seal_path = root / "SEALED.json"
    verification_path = repository / PREDICTION_VERIFICATION
    seal = base._read(seal_path)
    verification = base._read(verification_path)
    contract_path = repository / CONTRACT_RELATIVE
    contract = base._read(contract_path)
    synthetic_path = repository / SYNTHETIC_RELATIVE
    synthetic = base._read(synthetic_path)
    wf04_synthetic_path = repository / WF04_SYNTHETIC_RELATIVE
    wf04_synthetic = base._read(wf04_synthetic_path)
    expected_gates = {
        "all_baselines_reconstructed_from_closed_prior_receipts_only": True,
        "all_continuous_quantiles_reconstructed": True,
        "all_final_query_outcomes_unopened": True,
        "all_month_prediction_before_outcome_orders_valid": True,
        "all_neighbor_formulas_reconstructed": True,
        "all_nonfinal_query_outcomes_match_independent_oracle": True,
        "all_physical_seals_valid": True,
        "all_pointwise_paths_reconstructed": True,
    }
    if not all((
        _valid_seal(seal, timing=True), seal.get("passed") is True,
        seal.get("evaluation_metrics_opened") is False,
        seal.get("final_period_result_opened") is False,
        verification.get("passed") is True,
        verification.get("evaluation_authorized") is True,
        verification.get("final_period_result_opened") is False,
        verification.get("gates") == expected_gates,
        verification.get("store_result_digest") == seal.get("result_digest"),
        verification.get("verification_digest") == stable_hash({
            key: value for key, value in verification.items() if key != "verification_digest"
        }),
        contract.get("contract_digest") == "2c65288b45dd79627a1e16f0a4c2d9449919d0602974d7a52440415bd7f96a8e",
        synthetic.get("passed") is True,
        synthetic.get("final_period_result_opened") is False,
        wf04_synthetic.get("passed") is True,
        wf04_synthetic.get("final_period_result_opened") is False,
    )):
        raise WalkForwardEvaluationError("verified prediction/scoring boundary differs")
    expected_layout = set((*predictions.PREDICTION_FILES, "PREDICTIONS_SEALED.json"))
    cache = repository / predictions.CACHE_RELATIVE
    final_months = sorted(cache.glob("month-2024-*") ) + sorted(cache.glob("month-2025-*") )
    if len(final_months) != 20 or any(
        path.is_symlink() or not path.is_dir() or {child.name for child in path.iterdir()} != expected_layout
        for path in final_months
    ):
        raise WalkForwardEvaluationError("final prediction-only month boundary differs")
    return {
        "prediction_store_result_digest": seal["result_digest"],
        "prediction_store_sha256": _sha(seal_path),
        "prediction_verification_digest": verification["verification_digest"],
        "prediction_verification_sha256": _sha(verification_path),
        "walk_forward_contract_digest": contract["contract_digest"],
        "walk_forward_contract_sha256": _sha(contract_path),
        "synthetic_scoring_result_digest": synthetic["result_digest"],
        "synthetic_scoring_sha256": _sha(synthetic_path),
        "wf04_synthetic_result_digest": wf04_synthetic["result_digest"],
        "wf04_synthetic_implementation_h0": wf04_synthetic["implementation_h0"],
        "wf04_synthetic_sha256": _sha(wf04_synthetic_path),
        "raw_predictions_sha256": _sha(root / "raw-predictions.parquet"),
        "baseline_predictions_sha256": _sha(root / "baseline-predictions.parquet"),
        "continuous_predictions_sha256": _sha(root / "continuous-predictions.parquet"),
        "path_predictions_sha256": _sha(root / "path-predictions.parquet"),
        "nonfinal_query_outcomes_sha256": _sha(root / "nonfinal-query-outcomes.parquet"),
        "nonfinal_query_paths_sha256": _sha(root / "nonfinal-query-paths.parquet"),
    }


def _registry(repository: Path) -> pd.DataFrame:
    frame = pd.read_parquet(repository / REGISTRY_RELATIVE, engine="pyarrow").copy()
    if len(frame) != predictions.EXPECTED_QUERIES or frame.episode_id.astype(str).duplicated().any():
        raise WalkForwardEvaluationError("registry differs")
    frame["query_id"] = frame.episode_id.astype(str)
    frame["cutoff"] = pd.to_datetime(frame.cutoff)
    frame["month"] = frame.cutoff.dt.to_period("M").astype(str)
    frame["calendar_year"] = frame.cutoff.dt.year.astype(int)
    frame["quality_liquidity_cell"] = frame.quality_tier.astype(str) + "|" + frame.liquidity_stratum.astype(str)
    return frame.sort_values(["cutoff", "query_id"], kind="stable").reset_index(drop=True)


def _calendar(repository: Path) -> tuple[pd.DatetimeIndex, str, str]:
    config = load_config(repository / base.CONFIG_RELATIVE)
    spec = config.datasets["nasdaq"]
    source = source_from_spec(spec)
    benchmark = source.load_benchmark()
    if benchmark is None or benchmark.attrs.get("source_timestamp_reordered") \
            or benchmark.attrs.get("source_duplicate_timestamps"):
        raise WalkForwardEvaluationError("benchmark calendar differs")
    sessions = pd.DatetimeIndex(pd.to_datetime(benchmark.timestamp)).sort_values().unique()
    if spec.benchmark is None:
        raise WalkForwardEvaluationError("NASDAQ benchmark is unavailable")
    benchmark_path = spec.benchmark.path
    return sessions, stable_hash([stamp.isoformat() for stamp in sessions]), file_fingerprint(benchmark_path)


def _fold_contract(repository: Path, sessions: Sequence[pd.Timestamp]) -> tuple[list[dict[str, Any]], dict[str, pd.Timestamp | None]]:
    contract = base._read(repository / CONTRACT_RELATIVE)
    folds = list(contract["temporal_protocol"]["folds"])
    cutoffs = fold_purge_cutoffs(sessions, folds, purge_sessions=126)
    return folds, cutoffs


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise WalkForwardEvaluationError("globally clean Git worktree required")
    if (repository / OUTPUT_RELATIVE).exists():
        raise WalkForwardEvaluationError("non-final evaluation output must be absent")
    h0 = str(_git(repository, "rev-parse", "HEAD"))
    inputs = _verified_inputs(repository)
    if inputs["wf04_synthetic_implementation_h0"] != h0:
        raise WalkForwardEvaluationError("WF-04 synthetic receipt is not bound to implementation H0")
    registry = _registry(repository)
    sessions, calendar_digest, benchmark_fingerprint = _calendar(repository)
    folds, cutoffs = _fold_contract(repository, sessions)
    registry["purged_evaluation_included"] = [
        bool(row.fold_id in SCORED_NONFINAL_FOLDS and query_inside_purged_fold(
            row.cutoff, str(row.fold_id), cutoffs,
        )) for row in registry.itertuples(index=False)
    ]
    counts = {
        fold: int(registry.loc[
            (registry.fold_id == fold) & registry.purged_evaluation_included
        ].shape[0]) for fold in SCORED_NONFINAL_FOLDS
    }
    state = {
        "schema_version": SCHEMA,
        "status": "frozen_before_nonfinal_score_calculation",
        "implementation_h0": h0,
        "runtime_files": _runtime_manifest(repository, h0),
        "verified_inputs": inputs,
        "benchmark_calendar_digest": calendar_digest,
        "benchmark_source_fingerprint": benchmark_fingerprint,
        "folds": folds,
        "purge_sessions": 126,
        "purge_cutoffs": {
            key: value.isoformat() if value is not None else None for key, value in cutoffs.items()
        },
        "included_query_counts": counts,
        "included_query_count": sum(counts.values()),
        "probability_lanes": list(PROBABILITY_LANES),
        "continuous_lanes": list(CONTINUOUS_LANES),
        "comparison_baselines": list(COMPARISON_BASELINES),
        "primary_prefix": 20,
        "primary_horizon_sessions": 20,
        "inference_unit": "calendar_month_mean_loss",
        "bootstrap_resamples": 10_000,
        "bootstrap_seed": 20_260_901,
        "bootstrap_block_length_months": 3,
        "dm_hac_lags_months": 3,
        "literal_selective_rule": "all_D4_abstention_reasons",
        "research_dynamic_selective_rule": list(DYNAMIC_ABSTENTION_REASONS),
        "permanent_product_blockers": list(PERMANENT_PRODUCT_BLOCKERS),
        "literal_minimum_nonabstained_gate_structurally_possible": False,
        "contract_contradiction_disposition": "retain_literal_failure_and_report_dynamic_diagnostic_without_promotion",
        "nonfinal_query_outcomes_accessed_for_scoring": False,
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
        raise WalkForwardEvaluationError("expected one exact preregistration-only child")
    return accepted[0]


def validate_preregistration(repository: Path) -> tuple[dict[str, Any], pd.DataFrame, dict[str, pd.Timestamp | None], str]:
    if _git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise WalkForwardEvaluationError("globally clean Git worktree required")
    path = repository / PREREGISTRATION_RELATIVE
    raw = path.read_bytes()
    prereg = base._read(path)
    if prereg.get("schema_version") != SCHEMA or not _valid_seal(prereg, "preregistration_digest"):
        raise WalkForwardEvaluationError("non-final preregistration differs")
    h0 = str(prereg["implementation_h0"])
    h1 = _sole_child(repository, raw, h0)
    if subprocess.run(["git", "merge-base", "--is-ancestor", h1, "HEAD"], cwd=repository).returncode:
        raise WalkForwardEvaluationError("HEAD does not descend from preregistration")
    for name, expected in prereg["runtime_files"].items():
        if _sha(repository / name) != expected \
                or sha256(_git(repository, "show", f"{h0}:{name}", raw=True)).hexdigest() != expected:
            raise WalkForwardEvaluationError(f"runtime source drifted: {name}")
    if prereg["verified_inputs"] != _verified_inputs(repository):
        raise WalkForwardEvaluationError("verified inputs drifted")
    registry = _registry(repository)
    sessions, calendar_digest, fingerprint = _calendar(repository)
    _, cutoffs = _fold_contract(repository, sessions)
    observed_cutoffs = {key: value.isoformat() if value is not None else None for key, value in cutoffs.items()}
    if prereg["benchmark_calendar_digest"] != calendar_digest \
            or prereg["benchmark_source_fingerprint"] != fingerprint \
            or prereg["purge_cutoffs"] != observed_cutoffs:
        raise WalkForwardEvaluationError("calendar/purge identity drifted")
    return prereg, registry, cutoffs, h1


def _route_value(row: Any) -> str | None:
    label = row.barrier_label
    if isinstance(label, str) and label in set(PRIMARY_CLASSES + ("ambiguous_same_first_touch_bar", "censored")):
        return label
    if str(row.barrier_status) == "censored" or str(row.status).startswith("censored"):
        return "censored"
    return None


def _probability_scores(
    raw: pd.DataFrame, baselines: pd.DataFrame, outcomes: pd.DataFrame,
    registry: pd.DataFrame, cutoffs: Mapping[str, pd.Timestamp | None],
) -> pd.DataFrame:
    primary = raw.loc[raw.prefix == 20].copy()
    common = [
        "query_id", "query_case_id", "query_cutoff", "month", "fold_id", "scored", "lane",
        "favorable_probability", "adverse_probability", "no_touch_probability",
        "directional_favorable_probability", "directional_adverse_probability",
        "nearest_composite_distance", "eligible_rows", "effective_rows",
        "abstention_reasons", "selective_lane",
    ]
    primary = primary[common]
    baseline = baselines.copy()
    baseline["nearest_composite_distance"] = np.nan
    baseline["eligible_rows"] = baseline.prior_eligible_rows
    baseline["effective_rows"] = baseline.prior_eligible_rows.astype(float)
    baseline["abstention_reasons"] = ""
    baseline["selective_lane"] = True
    baseline = baseline.rename(columns={"query_case_id": "query_case_id"})[common]
    prediction = pd.concat([primary, baseline], ignore_index=True)
    if set(prediction.lane) != set(PROBABILITY_LANES) or prediction.duplicated(["query_id", "lane"]).any():
        raise WalkForwardEvaluationError("probability prediction inventory differs")
    observed = outcomes.loc[outcomes.horizon_sessions == 20, [
        "query_id", "barrier_label", "barrier_status", "status", "complete", "query_regime",
    ]]
    if observed.query_id.duplicated().any():
        raise WalkForwardEvaluationError("query outcome horizon is not unique")
    meta = registry[[
        "query_id", "quality_tier", "liquidity_stratum", "quality_liquidity_cell", "calendar_year",
    ]]
    joined = prediction.merge(observed, on="query_id", how="left", validate="many_to_one").merge(
        meta, on="query_id", how="left", validate="many_to_one",
    )
    rows: list[dict[str, Any]] = []
    for row in joined.itertuples(index=False):
        if row.fold_id == FINAL_FOLD:
            raise WalkForwardEvaluationError("final prediction entered non-final scorer")
        route = _route_value(row)
        scores = probability_score_record(
            (row.favorable_probability, row.adverse_probability, row.no_touch_probability),
            (row.directional_favorable_probability, row.directional_adverse_probability),
            route,
        )
        permanent, dynamic = split_abstention_reasons(row.abstention_reasons)
        included = bool(row.fold_id in SCORED_NONFINAL_FOLDS and query_inside_purged_fold(
            row.query_cutoff, str(row.fold_id), cutoffs,
        ))
        rows.append({
            "query_id": row.query_id, "query_cutoff": row.query_cutoff, "month": row.month,
            "calendar_year": int(row.calendar_year), "fold_id": row.fold_id, "lane": row.lane,
            "query_regime": row.query_regime, "quality_tier": row.quality_tier,
            "liquidity_stratum": row.liquidity_stratum,
            "quality_liquidity_cell": row.quality_liquidity_cell,
            "purged_evaluation_included": included,
            "literal_selective": bool(row.selective_lane),
            "research_dynamic_selective": not dynamic,
            "permanent_product_blockers": "|".join(permanent),
            "dynamic_abstention_reasons": "|".join(dynamic),
            "risk_score": row.nearest_composite_distance,
            "eligible_rows": int(row.eligible_rows), "effective_rows": float(row.effective_rows),
            "favorable_probability": float(row.favorable_probability),
            "adverse_probability": float(row.adverse_probability),
            "no_touch_probability": float(row.no_touch_probability),
            "directional_favorable_probability": float(row.directional_favorable_probability),
            "directional_adverse_probability": float(row.directional_adverse_probability),
            **scores,
        })
    return pd.DataFrame(rows).sort_values(["query_id", "lane"], kind="stable").reset_index(drop=True)


def _continuous_scores(
    forecasts: pd.DataFrame, outcomes: pd.DataFrame, registry: pd.DataFrame,
    cutoffs: Mapping[str, pd.Timestamp | None],
) -> pd.DataFrame:
    meta = registry[["query_id", "cutoff", "fold_id", "calendar_year"]].rename(columns={"cutoff": "registry_cutoff"})
    values = outcomes[["query_id", "horizon_sessions", "complete", *CONTINUOUS_MEASURES]]
    joined = forecasts.merge(values, on=["query_id", "horizon_sessions"], how="left", validate="many_to_one").merge(
        meta, on="query_id", how="left", validate="many_to_one",
    )
    if (joined.fold_id_x != joined.fold_id_y).any() or (pd.to_datetime(joined.query_cutoff) != joined.registry_cutoff).any():
        raise WalkForwardEvaluationError("continuous query metadata differs")
    rows: list[dict[str, Any]] = []
    for row in joined.itertuples(index=False):
        if row.fold_id_x == FINAL_FOLD:
            raise WalkForwardEvaluationError("final prediction entered continuous scorer")
        observed = getattr(row, row.measure) if bool(row.complete) else None
        score = continuous_score_record(observed, {
            .1: row.q10, .25: row.q25, .5: row.q50, .75: row.q75, .9: row.q90,
        })
        rows.append({
            "query_id": row.query_id, "query_cutoff": row.query_cutoff, "month": row.month,
            "calendar_year": int(row.calendar_year), "fold_id": row.fold_id_x, "lane": row.lane,
            "horizon_sessions": int(row.horizon_sessions), "measure": row.measure,
            "purged_evaluation_included": bool(
                row.fold_id_x in SCORED_NONFINAL_FOLDS and query_inside_purged_fold(
                    row.query_cutoff, str(row.fold_id_x), cutoffs,
                )
            ),
            "observed": float(observed) if observed is not None and pd.notna(observed) else np.nan,
            "q10": row.q10, "q25": row.q25, "q50": row.q50, "q75": row.q75, "q90": row.q90,
            **score,
        })
    return pd.DataFrame(rows).sort_values(
        ["query_id", "lane", "horizon_sessions", "measure"], kind="stable",
    ).reset_index(drop=True)


def _fold_metrics(scores: pd.DataFrame) -> pd.DataFrame:
    usable = scores.loc[scores.purged_evaluation_included].sort_values(
        ["query_id", "lane"], kind="stable",
    ).copy()
    rows: list[dict[str, Any]] = []
    scopes = [(fold, usable.loc[usable.fold_id == fold]) for fold in SCORED_NONFINAL_FOLDS]
    scopes.append(("validation_pooled", usable.loc[usable.fold_id.isin(VALIDATION_FOLDS)]))
    for scope, frame in scopes:
        for lane, group in frame.groupby("lane", sort=True):
            primary = group.loc[group.multiclass_evaluable]
            directional = group.loc[group.directional_evaluable]
            rows.append({
                "scope": scope, "lane": lane, "query_rows": len(group),
                "multiclass_evaluable_rows": len(primary),
                "mean_multiclass_brier": primary.multiclass_brier.mean(),
                "mean_multiclass_log_loss": primary.multiclass_log_loss.mean(),
                "directional_evaluable_rows": len(directional),
                "mean_directional_brier": directional.directional_brier.mean(),
                "mean_directional_log_loss": directional.directional_log_loss.mean(),
                "literal_selective_rows": int(group.literal_selective.sum()),
                "research_dynamic_selective_rows": int(group.research_dynamic_selective.sum()),
            })
    return pd.DataFrame(rows).sort_values(["scope", "lane"], kind="stable").reset_index(drop=True)


def _stability_metrics(scores: pd.DataFrame) -> pd.DataFrame:
    usable = scores.loc[
        scores.purged_evaluation_included & scores.multiclass_evaluable
    ].sort_values(["query_id", "lane"], kind="stable").copy()
    rows: list[dict[str, Any]] = []
    for dimension, column in (
        ("fold", "fold_id"), ("calendar_year", "calendar_year"),
        ("benchmark_regime", "query_regime"), ("quality_liquidity_cell", "quality_liquidity_cell"),
    ):
        for (value, lane), group in usable.groupby([column, "lane"], sort=True, dropna=False):
            rows.append({
                "dimension": dimension, "value": str(value), "lane": lane, "rows": len(group),
                "mean_multiclass_brier": group.multiclass_brier.mean(),
                "mean_multiclass_log_loss": group.multiclass_log_loss.mean(),
            })
    return pd.DataFrame(rows).sort_values(["dimension", "value", "lane"], kind="stable").reset_index(drop=True)


def _calibration(scores: pd.DataFrame) -> pd.DataFrame:
    usable = scores.loc[
        scores.purged_evaluation_included & scores.multiclass_evaluable
    ].sort_values(["query_id", "lane"], kind="stable").copy()
    rows: list[dict[str, Any]] = []
    scopes = [(fold, usable.loc[usable.fold_id == fold]) for fold in SCORED_NONFINAL_FOLDS]
    scopes.append(("validation_pooled", usable.loc[usable.fold_id.isin(VALIDATION_FOLDS)]))
    for scope, frame in scopes:
        for lane, group in frame.groupby("lane", sort=True):
            for class_name, probability_column in zip(
                PRIMARY_CLASSES, ("favorable_probability", "adverse_probability", "no_touch_probability"),
            ):
                bins, ece = equal_count_calibration(
                    group[probability_column].to_numpy(),
                    (group.route_status == class_name).astype(int).to_numpy(),
                    bins=10, minimum_rows=30,
                )
                for row in bins.itertuples(index=False):
                    rows.append({
                        "scope": scope, "lane": lane, "class_name": class_name,
                        **row._asdict(), "ece": ece,
                    })
    return pd.DataFrame(rows).sort_values(
        ["scope", "lane", "class_name", "bin"], kind="stable",
    ).reset_index(drop=True)


def _risk_coverage(scores: pd.DataFrame) -> pd.DataFrame:
    composite = scores.loc[
        scores.purged_evaluation_included & scores.multiclass_evaluable & (scores.lane == "composite")
    ].sort_values("query_id", kind="stable").copy()
    if composite.risk_score.isna().any():
        raise WalkForwardEvaluationError("composite risk score is missing")
    rows: list[dict[str, Any]] = []
    scopes = [(fold, composite.loc[composite.fold_id == fold]) for fold in SCORED_NONFINAL_FOLDS]
    scopes.append(("validation_pooled", composite.loc[composite.fold_id.isin(VALIDATION_FOLDS)]))
    for scope, group in scopes:
        curve = risk_coverage_curve(group.multiclass_brier, group.risk_score)
        for row in curve.itertuples(index=False):
            rows.append({"scope": scope, **row._asdict()})
    return pd.DataFrame(rows).sort_values(["scope", "requested_coverage"], kind="stable").reset_index(drop=True)


def _inference(scores: pd.DataFrame) -> pd.DataFrame:
    usable = scores.loc[
        scores.purged_evaluation_included & scores.multiclass_evaluable
    ].sort_values(["query_id", "lane"], kind="stable")
    rows: list[dict[str, Any]] = []
    scopes = [(fold, usable.loc[usable.fold_id == fold]) for fold in SCORED_NONFINAL_FOLDS]
    scopes.append(("validation_pooled", usable.loc[usable.fold_id.isin(VALIDATION_FOLDS)]))
    for scope, frame in scopes:
        composite = frame.loc[
            frame.lane == "composite", ["query_id", "query_cutoff", "multiclass_brier"],
        ].sort_values("query_id", kind="stable")
        for baseline_lane in COMPARISON_BASELINES:
            baseline = frame.loc[
                frame.lane == baseline_lane, ["query_id", "multiclass_brier"],
            ].sort_values("query_id", kind="stable")
            paired = composite.merge(baseline, on="query_id", suffixes=("_model", "_baseline"), validate="one_to_one")
            monthly = calendar_month_mean_losses(
                paired.query_cutoff.tolist(), paired.multiclass_brier_model,
                paired.multiclass_brier_baseline,
            )
            mean_difference, bootstrap_p = moving_block_bootstrap_lower_pvalue(
                monthly.loss_difference, resamples=10_000, block_length=3, seed=20_260_901,
            )
            dm_mean, dm_statistic, dm_p = diebold_mariano_hac_lower_pvalue(
                monthly.loss_difference, lags=3,
            )
            rows.append({
                "scope": scope, "model_lane": "composite", "baseline_lane": baseline_lane,
                "query_pairs": len(paired), "month_pairs": len(monthly),
                "model_mean_brier": paired.multiclass_brier_model.mean(),
                "baseline_mean_brier": paired.multiclass_brier_baseline.mean(),
                "brier_skill": 1.0 - paired.multiclass_brier_model.mean() / paired.multiclass_brier_baseline.mean(),
                "mean_monthly_loss_difference": mean_difference,
                "bootstrap_lower_pvalue": bootstrap_p,
                "dm_mean_difference": dm_mean, "dm_statistic": dm_statistic,
                "dm_lower_pvalue": dm_p,
            })
    result = pd.DataFrame(rows)
    for source in ("bootstrap_lower_pvalue", "dm_lower_pvalue"):
        result[source.replace("_pvalue", "_holm_pvalue")] = np.nan
        result[source.replace("_pvalue", "_holm_reject")] = False
    for scope, positions in result.groupby("scope", sort=True).groups.items():
        selected = result.loc[positions]
        for source in ("bootstrap_lower_pvalue", "dm_lower_pvalue"):
            adjusted = holm_adjust(dict(zip(selected.baseline_lane, selected[source])))
            result.loc[positions, source.replace("_pvalue", "_holm_pvalue")] = [
                adjusted[name][0] for name in selected.baseline_lane
            ]
            result.loc[positions, source.replace("_pvalue", "_holm_reject")] = [
                adjusted[name][1] for name in selected.baseline_lane
            ]
    return result.sort_values(["scope", "baseline_lane"], kind="stable").reset_index(drop=True)


def _interval_coverage(continuous: pd.DataFrame) -> pd.DataFrame:
    usable = continuous.loc[continuous.purged_evaluation_included & continuous.evaluable]
    rows: list[dict[str, Any]] = []
    scopes = [(fold, usable.loc[usable.fold_id == fold]) for fold in SCORED_NONFINAL_FOLDS]
    scopes.append(("validation_pooled", usable.loc[usable.fold_id.isin(VALIDATION_FOLDS)]))
    for scope, frame in scopes:
        for (lane, horizon, measure), group in frame.groupby(
            ["lane", "horizon_sessions", "measure"], sort=True,
        ):
            ordered = group.sort_values(["query_cutoff", "query_id"], kind="stable")
            for interval, nominal, column in (
                ("central_50", .5, "central_50_covered"),
                ("central_80", .8, "central_80_covered"),
            ):
                tests = interval_coverage_tests(
                    ordered[column].astype(bool).to_numpy(), nominal_coverage=nominal, dynamic_lags=3,
                )
                rows.append({
                    "scope": scope, "lane": lane, "horizon_sessions": int(horizon),
                    "measure": measure, "interval": interval, **tests.__dict__,
                })
    result = pd.DataFrame(rows)
    for source in (
        "exact_marginal_pvalue", "christoffersen_independence_pvalue", "dynamic_binary_pvalue",
    ):
        result[source.replace("_pvalue", "_holm_pvalue")] = np.nan
        result[source.replace("_pvalue", "_holm_reject")] = False
    for scope, positions in result.groupby("scope", sort=True).groups.items():
        selected = result.loc[positions]
        for source in (
            "exact_marginal_pvalue", "christoffersen_independence_pvalue", "dynamic_binary_pvalue",
        ):
            adjusted = holm_adjust({str(index): float(value) for index, value in zip(positions, selected[source])})
            result.loc[positions, source.replace("_pvalue", "_holm_pvalue")] = [
                adjusted[str(index)][0] for index in positions
            ]
            result.loc[positions, source.replace("_pvalue", "_holm_reject")] = [
                adjusted[str(index)][1] for index in positions
            ]
    return result.sort_values(
        ["scope", "lane", "horizon_sessions", "measure", "interval"], kind="stable",
    ).reset_index(drop=True)


def _path_metrics(
    forecasts: pd.DataFrame, paths: pd.DataFrame, registry: pd.DataFrame,
    cutoffs: Mapping[str, pd.Timestamp | None],
) -> pd.DataFrame:
    meta = registry[["query_id", "cutoff", "fold_id"]]
    allowed = meta.loc[meta.fold_id.isin(SCORED_NONFINAL_FOLDS)].copy()
    allowed["included"] = [
        query_inside_purged_fold(row.cutoff, str(row.fold_id), cutoffs)
        for row in allowed.itertuples(index=False)
    ]
    allowed_ids = set(allowed.loc[allowed.included, "query_id"])
    forecast = forecasts.loc[forecasts.query_id.isin(allowed_ids)]
    observed = paths.loc[paths.query_id.isin(allowed_ids)]
    rows: list[dict[str, Any]] = []
    fold_map = dict(zip(allowed.query_id, allowed.fold_id))
    for measure in PATH_MEASURES:
        prediction_column = f"{measure}_median"
        joined = forecast[["query_id", "lane", "step", prediction_column]].merge(
            observed[["query_id", "step", measure]], on=["query_id", "step"],
            how="left", validate="many_to_one",
        )
        valid = np.isfinite(joined[prediction_column]) & np.isfinite(joined[measure])
        joined = joined.loc[valid].copy()
        joined["fold_id"] = joined.query_id.map(fold_map)
        joined["absolute_error"] = (joined[prediction_column] - joined[measure]).abs()
        for (fold, lane), group in joined.groupby(["fold_id", "lane"], sort=True):
            rows.append({
                "scope": fold, "lane": lane, "measure": measure, "observed_steps": len(group),
                "mean_absolute_error": group.absolute_error.mean(),
                "median_absolute_error": group.absolute_error.median(),
            })
        pooled = joined.loc[joined.fold_id.isin(VALIDATION_FOLDS)]
        for lane, group in pooled.groupby("lane", sort=True):
            rows.append({
                "scope": "validation_pooled", "lane": lane, "measure": measure,
                "observed_steps": len(group), "mean_absolute_error": group.absolute_error.mean(),
                "median_absolute_error": group.absolute_error.median(),
            })
    return pd.DataFrame(rows).sort_values(["scope", "lane", "measure"], kind="stable").reset_index(drop=True)


def _coverage(scores: pd.DataFrame, continuous: pd.DataFrame, prereg: Mapping[str, Any]) -> dict[str, Any]:
    primary = scores.loc[scores.lane == "composite"]
    included = primary.loc[primary.purged_evaluation_included]
    route_counts = included.route_status.value_counts(dropna=False).sort_index().to_dict()
    return _seal({
        "schema_version": STORE_SCHEMA,
        "status": "nonfinal_coverage",
        "preregistration_digest": prereg["preregistration_digest"],
        "nonfinal_scored_query_rows": len(primary),
        "purged_evaluation_query_rows": len(included),
        "multiclass_evaluable_queries": int(included.multiclass_evaluable.sum()),
        "directional_evaluable_queries": int(included.directional_evaluable.sum()),
        "route_status_counts": {str(key): int(value) for key, value in route_counts.items()},
        "literal_selective_queries": int(included.literal_selective.sum()),
        "research_dynamic_selective_queries": int(included.research_dynamic_selective.sum()),
        "literal_minimum_nonabstained_gate_structurally_possible": False,
        "continuous_rows": len(continuous),
        "continuous_evaluable_rows": int(continuous.loc[continuous.purged_evaluation_included, "evaluable"].sum()),
        "final_query_outcomes_opened": False,
    })


def _html(fold_metrics: pd.DataFrame, coverage: Mapping[str, Any]) -> str:
    rows = []
    for row in fold_metrics.itertuples(index=False):
        rows.append(
            "<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in (
                row.scope, row.lane, row.multiclass_evaluable_rows,
                f"{row.mean_multiclass_brier:.8f}", f"{row.mean_multiclass_log_loss:.8f}",
            )) + "</tr>"
        )
    return """<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>WF-04 non-final evaluation</title>
<style>body{font-family:system-ui;max-width:1200px;margin:2rem auto;line-height:1.45}table{border-collapse:collapse;width:100%}th,td{padding:.45rem;border-bottom:1px solid #ddd;text-align:left}.warn{background:#fff4dc;padding:.8rem;border-left:5px solid #b9770e}</style></head><body>
<h1>WF-04 non-final evaluation</h1><p>Final 2024–August-2025 outcomes remain unopened. These results cannot tune the final test.</p>
<p class=\"warn\">The literal V1 non-abstained gate is structurally impossible because two product blockers are always present. The forced-score evidence remains valid; the diagnostic dynamic-selective count cannot promote V1.</p>
<p>Purged queries: %d; multiclass evaluable: %d; literal selective: %d; dynamic-selective diagnostic: %d.</p>
<table><thead><tr><th>Scope</th><th>Lane</th><th>Rows</th><th>Mean Brier</th><th>Mean log loss</th></tr></thead><tbody>%s</tbody></table></body></html>""" % (
        coverage["purged_evaluation_query_rows"], coverage["multiclass_evaluable_queries"],
        coverage["literal_selective_queries"], coverage["research_dynamic_selective_queries"],
        "".join(rows),
    )


def _validate_existing(repository: Path, prereg: Mapping[str, Any], h1: str) -> dict[str, Any] | None:
    root = repository / OUTPUT_RELATIVE
    if not root.exists():
        return None
    expected = set((*OUTPUT_FILES, "SEALED.json"))
    if root.is_symlink() or not root.is_dir() or {path.name for path in root.iterdir()} != expected:
        raise WalkForwardEvaluationError("existing output layout differs")
    seal = base._read(root / "SEALED.json")
    if not _valid_seal(seal, timing=True) or seal.get("passed") is not True \
            or seal.get("preregistration_h1") != h1 \
            or seal.get("preregistration_digest") != prereg["preregistration_digest"] \
            or seal.get("file_manifest") != _manifest(root, OUTPUT_FILES):
        raise WalkForwardEvaluationError("existing output seal differs")
    return seal


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    prereg, registry, cutoffs, h1 = validate_preregistration(repository)
    if existing := _validate_existing(repository, prereg, h1):
        return existing
    started = perf_counter()
    root = repository / PREDICTION_ROOT
    raw = pd.read_parquet(root / "raw-predictions.parquet", engine="pyarrow")
    baselines = pd.read_parquet(root / "baseline-predictions.parquet", engine="pyarrow")
    forecasts = pd.read_parquet(root / "continuous-predictions.parquet", engine="pyarrow")
    path_forecasts = pd.read_parquet(root / "path-predictions.parquet", engine="pyarrow")
    outcomes = pd.read_parquet(root / "nonfinal-query-outcomes.parquet", engine="pyarrow")
    paths = pd.read_parquet(root / "nonfinal-query-paths.parquet", engine="pyarrow")
    if set(raw.loc[pd.to_datetime(raw.query_cutoff) >= predictions.FINAL_START, "query_id"]).intersection(outcomes.query_id):
        raise WalkForwardEvaluationError("final query outcome entered non-final namespace")
    nonfinal_raw = raw.loc[pd.to_datetime(raw.query_cutoff) < predictions.FINAL_START]
    nonfinal_baselines = baselines.loc[pd.to_datetime(baselines.query_cutoff) < predictions.FINAL_START]
    nonfinal_forecasts = forecasts.loc[pd.to_datetime(forecasts.query_cutoff) < predictions.FINAL_START]
    nonfinal_path_forecasts = path_forecasts.loc[path_forecasts.query_id.isin(set(nonfinal_raw.query_id))]
    scores = _probability_scores(nonfinal_raw, nonfinal_baselines, outcomes, registry, cutoffs)
    continuous = _continuous_scores(nonfinal_forecasts, outcomes, registry, cutoffs)
    fold_metrics = _fold_metrics(scores)
    stability = _stability_metrics(scores)
    calibration = _calibration(scores)
    risk = _risk_coverage(scores)
    inference = _inference(scores)
    interval = _interval_coverage(continuous)
    path_metrics = _path_metrics(nonfinal_path_forecasts, paths, registry, cutoffs)
    coverage = _coverage(scores, continuous, prereg)
    frames = {
        "query-scores.parquet": scores,
        "continuous-scores.parquet": continuous,
        "fold-metrics.parquet": fold_metrics,
        "stability-metrics.parquet": stability,
        "calibration-bins.parquet": calibration,
        "risk-coverage.parquet": risk,
        "inference.parquet": inference,
        "interval-coverage.parquet": interval,
        "path-metrics.parquet": path_metrics,
    }
    final = repository / OUTPUT_RELATIVE
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{final.name}.", dir=final.parent))
    try:
        for name, frame in frames.items():
            smoke._atomic_parquet(temporary / name, frame)
        smoke._atomic_json(temporary / "COVERAGE.json", coverage)
        (temporary / "index.html").write_text(_html(fold_metrics, coverage), encoding="utf-8")
        state = {
            "schema_version": STORE_SCHEMA, "status": "sealed", "passed": True,
            "preregistration_h1": h1,
            "preregistration_digest": prereg["preregistration_digest"],
            "prediction_store_result_digest": prereg["verified_inputs"]["prediction_store_result_digest"],
            "query_score_rows": len(scores), "continuous_score_rows": len(continuous),
            "fold_metric_rows": len(fold_metrics), "stability_metric_rows": len(stability),
            "calibration_bin_rows": len(calibration), "inference_rows": len(inference),
            "interval_coverage_rows": len(interval), "path_metric_rows": len(path_metrics),
            "coverage_result_digest": coverage["result_digest"],
            "file_manifest": _manifest(temporary, OUTPUT_FILES),
            "elapsed_seconds": perf_counter() - started,
            "nonfinal_query_outcomes_accessed_for_scoring": True,
            "final_period_result_opened": False,
            "acceptance_decision_opened": False,
            "production_promotion_authorized": False,
            "independent_verification_authorized": True,
        }
        seal = _seal(state, timing=True)
        smoke._atomic_json(temporary / "SEALED.json", seal)
        os.replace(temporary, final)
        return seal
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("preregister", "run"):
        child = sub.add_parser(command)
        child.add_argument("--repository", type=Path, required=True)
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
