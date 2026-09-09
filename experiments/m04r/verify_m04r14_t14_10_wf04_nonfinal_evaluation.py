"""Independently reconstruct the purged WF-04 non-final evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import binomtest, chi2, norm

from market_analogues.types import stable_hash

from experiments.m04r import m04r14_t14_09_outcome_smoke as smoke
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_10_wf04_nonfinal_evaluation as target


SCHEMA = "m04r14-t14-10-wf04-nonfinal-verification-v2"
NUMERIC_TOLERANCE = 5e-13


class IndependentEvaluationVerificationError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise IndependentEvaluationVerificationError(f"regular file required: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _valid(value: Mapping[str, Any], key: str = "result_digest", *, timing: bool = False) -> bool:
    omitted = {key, "created_at"} | ({"elapsed_seconds"} if timing else set())
    return value.get(key) == stable_hash({name: item for name, item in value.items() if name not in omitted})


def _probability_oracle(probabilities: Sequence[float], directional: Sequence[float], route: str) -> dict[str, Any]:
    probability = np.asarray(probabilities, dtype=np.float64)
    binary = np.asarray(directional, dtype=np.float64)
    if probability.shape != (3,) or binary.shape != (2,) or not np.isfinite(probability).all() \
            or not np.isfinite(binary).all() or (probability <= 0).any() or (binary <= 0).any() \
            or abs(float(probability.sum()) - 1) > 1e-15 or abs(float(binary.sum()) - 1) > 1e-15:
        raise IndependentEvaluationVerificationError("invalid probability row")
    multiclass = route in target.PRIMARY_CLASSES
    directional_evaluable = route in ("favorable_first", "adverse_first")
    if multiclass:
        position = target.PRIMARY_CLASSES.index(route)
        truth = np.zeros(3); truth[position] = 1
        brier = float(np.square(probability - truth).sum())
        log = float(-math.log(probability[position]))
    else:
        brier = log = float("nan")
    if directional_evaluable:
        position = ("favorable_first", "adverse_first").index(route)
        truth = np.zeros(2); truth[position] = 1
        binary_brier = float(np.square(binary - truth).sum())
        binary_log = float(-math.log(binary[position]))
    else:
        binary_brier = binary_log = float("nan")
    return {
        "route_status": route, "multiclass_evaluable": multiclass,
        "multiclass_brier": brier, "multiclass_log_loss": log,
        "directional_evaluable": directional_evaluable,
        "directional_brier": binary_brier, "directional_log_loss": binary_log,
    }


def _reason_oracle(value: str) -> tuple[str, str, bool]:
    reasons = sorted(filter(None, str(value or "").split("|")))
    permanent = [name for name in reasons if name in target.PERMANENT_PRODUCT_BLOCKERS]
    dynamic = [name for name in reasons if name in target.DYNAMIC_ABSTENTION_REASONS]
    if len(permanent) + len(dynamic) != len(reasons):
        raise IndependentEvaluationVerificationError("unknown abstention reason")
    return "|".join(permanent), "|".join(dynamic), not dynamic


def _route(row: Any) -> str:
    label = row.barrier_label
    allowed = set(target.PRIMARY_CLASSES + ("ambiguous_same_first_touch_bar", "censored"))
    if isinstance(label, str) and label in allowed:
        return label
    if str(row.barrier_status) == "censored" or str(row.status).startswith("censored"):
        return "censored"
    return "unavailable"


def _included(cutoff: Any, fold: str, limits: Mapping[str, str | None]) -> bool:
    if fold not in target.SCORED_NONFINAL_FOLDS:
        return False
    limit = limits[fold]
    return limit is None or pd.Timestamp(cutoff) <= pd.Timestamp(limit)


def _query_score_oracle(repository: Path, prereg: Mapping[str, Any]) -> pd.DataFrame:
    root = repository / target.PREDICTION_ROOT
    raw = pd.read_parquet(root / "raw-predictions.parquet")
    raw = raw.loc[(raw.prefix == 20) & (pd.to_datetime(raw.query_cutoff) < target.predictions.FINAL_START)]
    baselines = pd.read_parquet(root / "baseline-predictions.parquet")
    baselines = baselines.loc[pd.to_datetime(baselines.query_cutoff) < target.predictions.FINAL_START]
    outcomes = pd.read_parquet(root / "nonfinal-query-outcomes.parquet")
    outcomes = outcomes.loc[outcomes.horizon_sessions == 20]
    registry = pd.read_parquet(repository / target.REGISTRY_RELATIVE).copy()
    registry["query_id"] = registry.episode_id.astype(str)
    registry["calendar_year"] = pd.to_datetime(registry.cutoff).dt.year.astype(int)
    registry["quality_liquidity_cell"] = registry.quality_tier.astype(str) + "|" + registry.liquidity_stratum.astype(str)
    observed = {str(row.query_id): row for row in outcomes.itertuples(index=False)}
    metadata = {str(row.query_id): row for row in registry.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    probability_columns = (
        "favorable_probability", "adverse_probability", "no_touch_probability",
    )
    directional_columns = ("directional_favorable_probability", "directional_adverse_probability")
    for row in raw.itertuples(index=False):
        outcome = observed[str(row.query_id)]; meta = metadata[str(row.query_id)]
        permanent, dynamic, research_selective = _reason_oracle(row.abstention_reasons)
        scores = _probability_oracle(
            [getattr(row, name) for name in probability_columns],
            [getattr(row, name) for name in directional_columns], _route(outcome),
        )
        rows.append({
            "query_id": row.query_id, "query_cutoff": row.query_cutoff, "month": row.month,
            "calendar_year": int(meta.calendar_year), "fold_id": row.fold_id, "lane": row.lane,
            "query_regime": outcome.query_regime, "quality_tier": meta.quality_tier,
            "liquidity_stratum": meta.liquidity_stratum,
            "quality_liquidity_cell": meta.quality_liquidity_cell,
            "purged_evaluation_included": _included(row.query_cutoff, row.fold_id, prereg["purge_cutoffs"]),
            "literal_selective": bool(row.selective_lane),
            "research_dynamic_selective": research_selective,
            "permanent_product_blockers": permanent, "dynamic_abstention_reasons": dynamic,
            "risk_score": float(row.nearest_composite_distance),
            "eligible_rows": int(row.eligible_rows), "effective_rows": float(row.effective_rows),
            **{name: float(getattr(row, name)) for name in probability_columns + directional_columns},
            **scores,
        })
    for row in baselines.itertuples(index=False):
        outcome = observed[str(row.query_id)]; meta = metadata[str(row.query_id)]
        scores = _probability_oracle(
            [getattr(row, name) for name in probability_columns],
            [getattr(row, name) for name in directional_columns], _route(outcome),
        )
        rows.append({
            "query_id": row.query_id, "query_cutoff": row.query_cutoff, "month": row.month,
            "calendar_year": int(meta.calendar_year), "fold_id": row.fold_id, "lane": row.lane,
            "query_regime": outcome.query_regime, "quality_tier": meta.quality_tier,
            "liquidity_stratum": meta.liquidity_stratum,
            "quality_liquidity_cell": meta.quality_liquidity_cell,
            "purged_evaluation_included": _included(row.query_cutoff, row.fold_id, prereg["purge_cutoffs"]),
            "literal_selective": True, "research_dynamic_selective": True,
            "permanent_product_blockers": "", "dynamic_abstention_reasons": "",
            "risk_score": float("nan"), "eligible_rows": int(row.prior_eligible_rows),
            "effective_rows": float(row.prior_eligible_rows),
            **{name: float(getattr(row, name)) for name in probability_columns + directional_columns},
            **scores,
        })
    return pd.DataFrame(rows).sort_values(["query_id", "lane"], kind="stable").reset_index(drop=True)


def _continuous_oracle(repository: Path, prereg: Mapping[str, Any]) -> pd.DataFrame:
    root = repository / target.PREDICTION_ROOT
    forecasts = pd.read_parquet(root / "continuous-predictions.parquet")
    forecasts = forecasts.loc[pd.to_datetime(forecasts.query_cutoff) < target.predictions.FINAL_START]
    outcomes = pd.read_parquet(root / "nonfinal-query-outcomes.parquet")
    outcome = {(str(row.query_id), int(row.horizon_sessions)): row for row in outcomes.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    for row in forecasts.itertuples(index=False):
        observed_row = outcome[(str(row.query_id), int(row.horizon_sessions))]
        observed = getattr(observed_row, str(row.measure))
        observed_available = bool(
            observed_row.complete and pd.notna(observed) and np.isfinite(float(observed))
        )
        quantiles = np.asarray([row.q10, row.q25, row.q50, row.q75, row.q90], dtype=np.float64)
        valid = bool(observed_available and np.isfinite(quantiles).all())
        if valid and (np.diff(quantiles) < 0).any():
            raise IndependentEvaluationVerificationError("non-monotonic quantiles")
        state: dict[str, Any] = {
            "evaluable": valid,
            "pinball_q10": np.nan, "pinball_q25": np.nan, "pinball_q50": np.nan,
            "pinball_q75": np.nan, "pinball_q90": np.nan, "median_absolute_error": np.nan,
            "central_50_score": np.nan, "central_50_width": np.nan, "central_50_covered": None,
            "central_80_score": np.nan, "central_80_width": np.nan, "central_80_covered": None,
        }
        if valid:
            actual = float(observed)
            for q, forecast in zip((.1, .25, .5, .75, .9), quantiles):
                error = actual - forecast
                state[f"pinball_q{int(q*100):02d}"] = float(max(q * error, (q - 1) * error))
            state["median_absolute_error"] = abs(actual - quantiles[2])
            for name, lower, upper, coverage in (
                ("central_50", quantiles[1], quantiles[3], .5),
                ("central_80", quantiles[0], quantiles[4], .8),
            ):
                penalty = 0.0
                if actual < lower: penalty = 2 * (lower - actual) / (1 - coverage)
                elif actual > upper: penalty = 2 * (actual - upper) / (1 - coverage)
                state[f"{name}_score"] = float(upper - lower + penalty)
                state[f"{name}_width"] = float(upper - lower)
                state[f"{name}_covered"] = bool(lower <= actual <= upper)
        rows.append({
            "query_id": row.query_id, "query_cutoff": row.query_cutoff, "month": row.month,
            "calendar_year": pd.Timestamp(row.query_cutoff).year, "fold_id": row.fold_id,
            "lane": row.lane, "horizon_sessions": int(row.horizon_sessions), "measure": row.measure,
            "purged_evaluation_included": _included(row.query_cutoff, row.fold_id, prereg["purge_cutoffs"]),
            "observed": float(observed) if observed_available else np.nan,
            "q10": row.q10, "q25": row.q25, "q50": row.q50, "q75": row.q75, "q90": row.q90,
            **state,
        })
    return pd.DataFrame(rows).sort_values(
        ["query_id", "lane", "horizon_sessions", "measure"], kind="stable",
    ).reset_index(drop=True)


def _assert_frame(name: str, expected: pd.DataFrame, observed: pd.DataFrame) -> float:
    if list(expected.columns) != list(observed.columns) or len(expected) != len(observed):
        raise IndependentEvaluationVerificationError(f"{name} schema/count differs")
    maximum = 0.0
    for column in expected.columns:
        left, right = expected[column], observed[column]
        if pd.api.types.is_numeric_dtype(left.dtype) and pd.api.types.is_numeric_dtype(right.dtype):
            a, b = left.to_numpy(dtype=float), right.to_numpy(dtype=float)
            if not np.array_equal(np.isnan(a), np.isnan(b)):
                raise IndependentEvaluationVerificationError(f"{name}:{column} missingness differs")
            finite = np.isfinite(a) & np.isfinite(b)
            delta = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
            maximum = max(maximum, delta)
            if delta > NUMERIC_TOLERANCE:
                raise IndependentEvaluationVerificationError(f"{name}:{column} delta {delta}")
        else:
            if not left.fillna("<NA>").astype(str).equals(right.fillna("<NA>").astype(str)):
                raise IndependentEvaluationVerificationError(f"{name}:{column} values differ")
    return maximum


def _simple_aggregates(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = scores.loc[scores.purged_evaluation_included].sort_values(
        ["query_id", "lane"], kind="stable",
    )
    folds: list[dict[str, Any]] = []
    scopes = [(fold, usable.loc[usable.fold_id == fold]) for fold in target.SCORED_NONFINAL_FOLDS]
    scopes.append(("validation_pooled", usable.loc[usable.fold_id.isin(target.VALIDATION_FOLDS)]))
    for scope, frame in scopes:
        for lane in sorted(frame.lane.unique()):
            group = frame.loc[frame.lane == lane]
            primary = group.loc[group.multiclass_evaluable]
            directional = group.loc[group.directional_evaluable]
            folds.append({
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
    stability: list[dict[str, Any]] = []
    primary = usable.loc[usable.multiclass_evaluable]
    for dimension, column in (
        ("fold", "fold_id"), ("calendar_year", "calendar_year"),
        ("benchmark_regime", "query_regime"), ("quality_liquidity_cell", "quality_liquidity_cell"),
    ):
        keys = primary[[column, "lane"]].drop_duplicates().sort_values([column, "lane"], kind="stable")
        for key in keys.itertuples(index=False, name=None):
            group = primary.loc[(primary[column] == key[0]) & (primary.lane == key[1])]
            stability.append({
                "dimension": dimension, "value": str(key[0]), "lane": key[1], "rows": len(group),
                "mean_multiclass_brier": group.multiclass_brier.mean(),
                "mean_multiclass_log_loss": group.multiclass_log_loss.mean(),
            })
    return (
        pd.DataFrame(folds).sort_values(["scope", "lane"], kind="stable").reset_index(drop=True),
        pd.DataFrame(stability).sort_values(["dimension", "value", "lane"], kind="stable").reset_index(drop=True),
    )


def _calibration_oracle(scores: pd.DataFrame) -> pd.DataFrame:
    usable = scores.loc[
        scores.purged_evaluation_included & scores.multiclass_evaluable
    ].sort_values(["query_id", "lane"], kind="stable")
    rows: list[dict[str, Any]] = []
    scopes = [(fold, usable.loc[usable.fold_id == fold]) for fold in target.SCORED_NONFINAL_FOLDS]
    scopes.append(("validation_pooled", usable.loc[usable.fold_id.isin(target.VALIDATION_FOLDS)]))
    for scope, frame in scopes:
        for lane in sorted(frame.lane.unique()):
            group = frame.loc[frame.lane == lane]
            for class_name, column in zip(target.PRIMARY_CLASSES, ("favorable_probability", "adverse_probability", "no_touch_probability")):
                probability = group[column].to_numpy(float)
                outcome = (group.route_status == class_name).to_numpy(int)
                bins = min(10, max(1, len(group) // 30))
                bin_rows = []
                for number, positions in enumerate(np.array_split(np.argsort(probability, kind="stable"), bins), 1):
                    mean = float(probability[positions].mean()); observed = float(outcome[positions].mean())
                    bin_rows.append({
                        "scope": scope, "lane": lane, "class_name": class_name, "bin": number,
                        "rows": len(positions), "minimum_probability": float(probability[positions].min()),
                        "maximum_probability": float(probability[positions].max()),
                        "mean_probability": mean, "observed_frequency": observed,
                        "absolute_gap": abs(mean - observed),
                    })
                ece = sum(row["rows"] * row["absolute_gap"] for row in bin_rows) / len(group)
                for row in bin_rows: rows.append({**row, "ece": ece})
    return pd.DataFrame(rows).sort_values(["scope", "lane", "class_name", "bin"], kind="stable").reset_index(drop=True)


def _risk_oracle(scores: pd.DataFrame) -> pd.DataFrame:
    data = scores.loc[
        scores.purged_evaluation_included & scores.multiclass_evaluable & (scores.lane == "composite")
    ].sort_values("query_id", kind="stable")
    rows = []
    scopes = [(fold, data.loc[data.fold_id == fold]) for fold in target.SCORED_NONFINAL_FOLDS]
    scopes.append(("validation_pooled", data.loc[data.fold_id.isin(target.VALIDATION_FOLDS)]))
    for scope, group in scopes:
        order = np.argsort(group.risk_score.to_numpy(float), kind="stable")
        for coverage in np.linspace(.1, 1., 10):
            retained = max(1, int(math.ceil(coverage * len(group))))
            selected = order[:retained]
            rows.append({
                "scope": scope, "requested_coverage": coverage, "retained_rows": retained,
                "empirical_coverage": retained / len(group),
                "mean_loss": float(group.multiclass_brier.to_numpy(float)[selected].mean()),
                "maximum_retained_risk_score": float(group.risk_score.to_numpy(float)[selected].max()),
            })
    return pd.DataFrame(rows).sort_values(["scope", "requested_coverage"], kind="stable").reset_index(drop=True)


def _bootstrap(values: np.ndarray) -> float:
    observed = float(values.mean()); centered = values - observed
    starts = np.arange(len(values) - 2); rng = np.random.Generator(np.random.PCG64(20_260_901))
    count = 0; blocks = int(math.ceil(len(values) / 3))
    for _ in range(10_000):
        chosen = rng.choice(starts, size=blocks, replace=True)
        sample = np.concatenate([centered[start:start + 3] for start in chosen])[:len(values)]
        count += bool(float(sample.mean()) <= observed)
    return (count + 1) / 10_001


def _dm(values: np.ndarray) -> tuple[float, float]:
    centered = values - values.mean(); n = len(values)
    long_run = float(centered @ centered / n)
    for lag in range(1, 4):
        long_run += 2 * (1 - lag / 4) * float(centered[lag:] @ centered[:-lag] / n)
    long_run = max(0., long_run)
    statistic = float(values.mean() / math.sqrt(long_run / n)) if long_run else (
        -math.inf if values.mean() < 0 else (math.inf if values.mean() > 0 else 0.)
    )
    return statistic, float(norm.cdf(statistic))


def _holm(values: Mapping[str, float]) -> dict[str, tuple[float, bool]]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0])); running = 0.; result = {}
    for index, (name, value) in enumerate(ordered):
        running = max(running, (len(ordered) - index) * value)
        adjusted = min(1., running); result[name] = (adjusted, adjusted < .05)
    return result


def _inference_oracle(scores: pd.DataFrame) -> pd.DataFrame:
    usable = scores.loc[
        scores.purged_evaluation_included & scores.multiclass_evaluable
    ].sort_values(["query_id", "lane"], kind="stable")
    rows = []
    scopes = [(fold, usable.loc[usable.fold_id == fold]) for fold in target.SCORED_NONFINAL_FOLDS]
    scopes.append(("validation_pooled", usable.loc[usable.fold_id.isin(target.VALIDATION_FOLDS)]))
    for scope, frame in scopes:
        model = frame.loc[frame.lane == "composite", ["query_id", "query_cutoff", "multiclass_brier"]]
        for baseline_lane in target.COMPARISON_BASELINES:
            baseline = frame.loc[frame.lane == baseline_lane, ["query_id", "multiclass_brier"]]
            paired = model.merge(baseline, on="query_id", suffixes=("_model", "_baseline"), validate="one_to_one")
            paired["month"] = pd.to_datetime(paired.query_cutoff).dt.to_period("M").astype(str)
            monthly = paired.groupby("month", sort=True)[["multiclass_brier_model", "multiclass_brier_baseline"]].mean()
            differences = (monthly.multiclass_brier_model - monthly.multiclass_brier_baseline).to_numpy(float)
            statistic, dm_p = _dm(differences)
            rows.append({
                "scope": scope, "model_lane": "composite", "baseline_lane": baseline_lane,
                "query_pairs": len(paired), "month_pairs": len(monthly),
                "model_mean_brier": paired.multiclass_brier_model.mean(),
                "baseline_mean_brier": paired.multiclass_brier_baseline.mean(),
                "brier_skill": 1 - paired.multiclass_brier_model.mean() / paired.multiclass_brier_baseline.mean(),
                "mean_monthly_loss_difference": differences.mean(), "bootstrap_lower_pvalue": _bootstrap(differences),
                "dm_mean_difference": differences.mean(), "dm_statistic": statistic, "dm_lower_pvalue": dm_p,
            })
    result = pd.DataFrame(rows)
    for source in ("bootstrap_lower_pvalue", "dm_lower_pvalue"):
        result[source.replace("_pvalue", "_holm_pvalue")] = np.nan
        result[source.replace("_pvalue", "_holm_reject")] = False
    for scope, positions in result.groupby("scope", sort=True).groups.items():
        for source in ("bootstrap_lower_pvalue", "dm_lower_pvalue"):
            adjusted = _holm(dict(zip(result.loc[positions, "baseline_lane"], result.loc[positions, source])))
            result.loc[positions, source.replace("_pvalue", "_holm_pvalue")] = [adjusted[name][0] for name in result.loc[positions, "baseline_lane"]]
            result.loc[positions, source.replace("_pvalue", "_holm_reject")] = [adjusted[name][1] for name in result.loc[positions, "baseline_lane"]]
    return result.sort_values(["scope", "baseline_lane"], kind="stable").reset_index(drop=True)


def _coverage_test(hits: np.ndarray, nominal: float) -> dict[str, Any]:
    hits = hits.astype(int); successes = int(hits.sum())
    previous, current = hits[:-1], hits[1:]
    n00 = int(((previous == 0) & (current == 0)).sum()); n01 = int(((previous == 0) & (current == 1)).sum())
    n10 = int(((previous == 1) & (current == 0)).sum()); n11 = int(((previous == 1) & (current == 1)).sum())
    def likelihood(s: int, n: int) -> float:
        if n == 0: return 0.
        return float(xlogy(s, s / n) + xlogy(n - s, 1 - s / n))
    independent = likelihood(n01 + n11, n00 + n01 + n10 + n11)
    markov = likelihood(n01, n00 + n01) + likelihood(n11, n10 + n11)
    centered = hits.astype(float) - nominal; response = centered[3:]
    design = np.column_stack([np.ones(len(response)), centered[2:-1], centered[1:-2], centered[:-3]])
    coefficients = np.linalg.pinv(design.T @ design) @ design.T @ response
    dq = float(response @ design @ coefficients / (nominal * (1 - nominal)))
    return {
        "observations": len(hits), "successes": successes, "empirical_coverage": hits.mean(),
        "exact_marginal_pvalue": float(binomtest(successes, len(hits), nominal).pvalue),
        "christoffersen_independence_pvalue": float(chi2.sf(max(0., 2 * (markov - independent)), 1)),
        "dynamic_binary_pvalue": float(chi2.sf(max(0., dq), 4)),
    }


def _interval_oracle(continuous: pd.DataFrame) -> pd.DataFrame:
    data = continuous.loc[continuous.purged_evaluation_included & continuous.evaluable]
    rows = []
    scopes = [(fold, data.loc[data.fold_id == fold]) for fold in target.SCORED_NONFINAL_FOLDS]
    scopes.append(("validation_pooled", data.loc[data.fold_id.isin(target.VALIDATION_FOLDS)]))
    for scope, frame in scopes:
        keys = frame[["lane", "horizon_sessions", "measure"]].drop_duplicates().sort_values(["lane", "horizon_sessions", "measure"])
        for lane, horizon, measure in keys.itertuples(index=False, name=None):
            group = frame.loc[(frame.lane == lane) & (frame.horizon_sessions == horizon) & (frame.measure == measure)].sort_values(["query_cutoff", "query_id"], kind="stable")
            for interval, nominal, column in (("central_50", .5, "central_50_covered"), ("central_80", .8, "central_80_covered")):
                rows.append({"scope": scope, "lane": lane, "horizon_sessions": int(horizon), "measure": measure, "interval": interval, **_coverage_test(group[column].to_numpy(bool), nominal)})
    result = pd.DataFrame(rows)
    for source in ("exact_marginal_pvalue", "christoffersen_independence_pvalue", "dynamic_binary_pvalue"):
        result[source.replace("_pvalue", "_holm_pvalue")] = np.nan
        result[source.replace("_pvalue", "_holm_reject")] = False
    for scope, positions in result.groupby("scope", sort=True).groups.items():
        for source in ("exact_marginal_pvalue", "christoffersen_independence_pvalue", "dynamic_binary_pvalue"):
            adjusted = _holm({str(i): float(result.loc[i, source]) for i in positions})
            result.loc[positions, source.replace("_pvalue", "_holm_pvalue")] = [adjusted[str(i)][0] for i in positions]
            result.loc[positions, source.replace("_pvalue", "_holm_reject")] = [adjusted[str(i)][1] for i in positions]
    return result.sort_values(["scope", "lane", "horizon_sessions", "measure", "interval"], kind="stable").reset_index(drop=True)


def _path_oracle(repository: Path, prereg: Mapping[str, Any]) -> pd.DataFrame:
    root = repository / target.PREDICTION_ROOT
    forecast = pd.read_parquet(root / "path-predictions.parquet")
    paths = pd.read_parquet(root / "nonfinal-query-paths.parquet")
    registry = pd.read_parquet(repository / target.REGISTRY_RELATIVE).copy()
    registry["query_id"] = registry.episode_id.astype(str)
    registry["included"] = [_included(row.cutoff, row.fold_id, prereg["purge_cutoffs"]) for row in registry.itertuples(index=False)]
    fold = dict(zip(registry.query_id, registry.fold_id)); allowed = set(registry.loc[registry.included, "query_id"])
    forecast = forecast.loc[forecast.query_id.isin(allowed)]; paths = paths.loc[paths.query_id.isin(allowed)]
    rows = []
    for measure in target.PATH_MEASURES:
        column = f"{measure}_median"
        joined = forecast[["query_id", "lane", "step", column]].merge(paths[["query_id", "step", measure]], on=["query_id", "step"], how="left", validate="many_to_one")
        joined = joined.loc[np.isfinite(joined[column]) & np.isfinite(joined[measure])].copy()
        joined["fold_id"] = joined.query_id.map(fold); joined["absolute_error"] = (joined[column] - joined[measure]).abs()
        for (scope, lane), group in joined.groupby(["fold_id", "lane"], sort=True):
            rows.append({"scope": scope, "lane": lane, "measure": measure, "observed_steps": len(group), "mean_absolute_error": group.absolute_error.mean(), "median_absolute_error": group.absolute_error.median()})
        pooled = joined.loc[joined.fold_id.isin(target.VALIDATION_FOLDS)]
        for lane, group in pooled.groupby("lane", sort=True):
            rows.append({"scope": "validation_pooled", "lane": lane, "measure": measure, "observed_steps": len(group), "mean_absolute_error": group.absolute_error.mean(), "median_absolute_error": group.absolute_error.median()})
    return pd.DataFrame(rows).sort_values(["scope", "lane", "measure"], kind="stable").reset_index(drop=True)


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repository, text=True, capture_output=True, check=True).stdout:
        raise IndependentEvaluationVerificationError("clean worktree required")
    started = perf_counter(); output = repository / target.OUTPUT_RELATIVE
    prereg = base._read(repository / target.PREREGISTRATION_RELATIVE)
    seal = base._read(output / "SEALED.json")
    if not _valid(prereg, "preregistration_digest") or not _valid(seal, timing=True) \
            or seal.get("passed") is not True or seal.get("final_period_result_opened") is not False \
            or seal.get("preregistration_digest") != prereg.get("preregistration_digest") \
            or seal.get("file_manifest") != target._manifest(output, target.OUTPUT_FILES):
        raise IndependentEvaluationVerificationError("preregistration/store seal differs")
    query_expected = _query_score_oracle(repository, prereg)
    continuous_expected = _continuous_oracle(repository, prereg)
    query_observed = pd.read_parquet(output / "query-scores.parquet")
    continuous_observed = pd.read_parquet(output / "continuous-scores.parquet")
    deltas = {
        "query_scores": _assert_frame("query_scores", query_expected, query_observed),
        "continuous_scores": _assert_frame("continuous_scores", continuous_expected, continuous_observed),
    }
    fold, stability = _simple_aggregates(query_expected)
    expected = {
        "fold_metrics": fold, "stability_metrics": stability,
        "calibration_bins": _calibration_oracle(query_expected),
        "risk_coverage": _risk_oracle(query_expected),
        "inference": _inference_oracle(query_expected),
        "interval_coverage": _interval_oracle(continuous_expected),
        "path_metrics": _path_oracle(repository, prereg),
    }
    for name, frame in expected.items():
        deltas[name] = _assert_frame(name, frame, pd.read_parquet(output / f"{name.replace('_', '-')}.parquet"))
    coverage = base._read(output / "COVERAGE.json")
    composite = query_expected.loc[(query_expected.lane == "composite") & query_expected.purged_evaluation_included]
    if not _valid(coverage) or coverage.get("final_query_outcomes_opened") is not False \
            or coverage.get("purged_evaluation_query_rows") != len(composite) \
            or coverage.get("multiclass_evaluable_queries") != int(composite.multiclass_evaluable.sum()) \
            or coverage.get("literal_selective_queries") != 0:
        raise IndependentEvaluationVerificationError("coverage receipt differs")
    state = {
        "schema_version": SCHEMA, "status": "verified", "passed": True,
        "preregistration_digest": prereg["preregistration_digest"],
        "store_result_digest": seal["result_digest"], "store_result_sha256": _sha(output / "SEALED.json"),
        "verified_query_score_rows": len(query_expected),
        "verified_continuous_score_rows": len(continuous_expected),
        "maximum_numeric_delta": max(deltas.values()), "numeric_tolerance": NUMERIC_TOLERANCE,
        "component_maximum_deltas": deltas,
        "gates": {
            "all_query_probability_scores_reconstructed": True,
            "all_continuous_scores_reconstructed": True,
            "all_fold_stability_calibration_risk_aggregates_reconstructed": True,
            "all_dependence_aware_inference_reconstructed": True,
            "all_interval_coverage_tests_reconstructed": True,
            "all_path_metrics_reconstructed": True,
            "literal_abstention_contradiction_retained": True,
            "all_final_query_outcomes_unopened": True,
            "all_physical_seals_valid": True,
        },
        "elapsed_seconds": perf_counter() - started,
        "nonfinal_evaluation_verified": True,
        "final_single_open_authorized": True,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
        "created_at": _now(),
    }
    result = {**state, "verification_digest": stable_hash(state)}
    final = repository / target.VERIFICATION_RELATIVE
    if final.exists():
        existing = base._read(final / "VERIFIED.json")
        omitted = {"created_at", "elapsed_seconds", "verification_digest"}
        if {key: value for key, value in existing.items() if key not in omitted} != {
            key: value for key, value in result.items() if key not in omitted
        }:
            raise IndependentEvaluationVerificationError("existing verification differs")
        return existing
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{final.name}.", dir=final.parent))
    try:
        smoke._atomic_json(temporary / "VERIFIED.json", result)
        os.replace(temporary, final)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True); raise
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(execute(args.repository), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
