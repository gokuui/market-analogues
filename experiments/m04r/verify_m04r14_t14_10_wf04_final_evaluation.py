"""Independently reconstruct the WF-04 final outcomes, scores, and decision."""
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

from market_analogues.adapters import source_from_spec
from market_analogues.config import load_config
from market_analogues.types import InstrumentKey, stable_hash

from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_10_wf04_final_evaluation as target
from experiments.m04r import verify_m04r14_t14_09_full_outcome_store as oracle_tools
from experiments.m04r import verify_m04r14_t14_10_wf04_nonfinal_evaluation as scoring_oracle
from experiments.m04r.m04r14_t14_09_outcome_oracle import (
    prepare_reference_series,
    reference_prepared_episode,
)


SCHEMA = "m04r14-t14-10-wf04-final-verification-v1"
NUMERIC_TOLERANCE = 5e-13


class FinalVerificationError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise FinalVerificationError(f"regular file required: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _valid(value: Mapping[str, Any], key: str = "result_digest", *, timing: bool = False) -> bool:
    omitted = {key, "created_at"} | ({"elapsed_seconds"} if timing else set())
    return value.get(key) == stable_hash({name: item for name, item in value.items() if name not in omitted})


def _outcome_oracle(repository: Path, registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = source_from_spec(load_config(repository / target.nonfinal.base.CONFIG_RELATIVE).datasets["nasdaq"])
    benchmark = source.load_benchmark()
    if benchmark is None:
        raise FinalVerificationError("oracle benchmark unavailable")
    reference_benchmark = prepare_reference_series(benchmark)
    contract = base._read(repository / target.predictions.outcome_store.CONTRACT)
    content_digest = target.predictions.base._resident()["content_digest"]
    fingerprints = target.predictions._source_fingerprints(repository)
    regimes = target.predictions._benchmark_regimes(repository, target.nonfinal._registry(repository))
    regime_map = dict(zip(regimes.month.astype(str), regimes.regime.astype(str)))
    expected_outcomes: list[dict[str, Any]] = []
    expected_paths: list[dict[str, Any]] = []
    for symbol, queries in registry.groupby("symbol", sort=True):
        key = InstrumentKey("nasdaq", str(symbol))
        stock = source.load(key); fingerprint = source.fingerprint(key)
        if fingerprints.get(str(symbol)) != fingerprint:
            raise FinalVerificationError(f"oracle source differs: {symbol}")
        ordered = stock.sort_values("timestamp", kind="stable").reset_index(drop=True)
        positions = {stamp: index for index, stamp in enumerate(pd.to_datetime(ordered.timestamp))}
        for query in queries.itertuples(index=False):
            position = positions.get(pd.Timestamp(query.cutoff))
            if position is None:
                raise FinalVerificationError(f"oracle cutoff absent: {query.case_id}")
            local = ordered.iloc[max(0, position - 20):min(len(ordered), position + 127)].reset_index(drop=True)
            outcomes, paths = reference_prepared_episode(
                prepare_reference_series(local), reference_benchmark,
                episode_id=str(query.query_id), cutoff=pd.Timestamp(query.cutoff),
                source_fingerprint=fingerprint, contract_digest=contract["contract_digest"],
                source_content_digest=content_digest,
            )
            for row in outcomes:
                expected_outcomes.append({**row, "query_id": str(query.query_id), "query_regime": regime_map[str(query.month)]})
            for row in paths:
                expected_paths.append({**row, "query_id": str(query.query_id)})
    expected_outcomes.sort(key=lambda row: (row["query_id"], row["horizon_sessions"]))
    expected_paths.sort(key=lambda row: (row["query_id"], row["step"]))
    return pd.DataFrame(expected_outcomes), pd.DataFrame(expected_paths)


def _query_scores(repository: Path, outcomes: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    root = repository / target.predictions.OUTPUT_RELATIVE
    final_ids = set(registry.query_id.astype(str))
    raw = pd.read_parquet(root / "raw-predictions.parquet")
    raw = raw.loc[(raw.prefix == 20) & raw.query_id.astype(str).isin(final_ids)]
    baselines = pd.read_parquet(root / "baseline-predictions.parquet")
    baselines = baselines.loc[baselines.query_id.astype(str).isin(final_ids)]
    observed = {str(row.query_id): row for row in outcomes.loc[outcomes.horizon_sessions == 20].itertuples(index=False)}
    metadata = {str(row.query_id): row for row in registry.itertuples(index=False)}
    probability_columns = ("favorable_probability", "adverse_probability", "no_touch_probability")
    directional_columns = ("directional_favorable_probability", "directional_adverse_probability")
    rows: list[dict[str, Any]] = []
    for is_baseline, frame in ((False, raw), (True, baselines)):
        for row in frame.itertuples(index=False):
            outcome = observed[str(row.query_id)]; meta = metadata[str(row.query_id)]
            if is_baseline:
                permanent = dynamic = ""; research = True
                risk = float("nan"); eligible = int(row.prior_eligible_rows); effective = float(eligible)
                literal = True
            else:
                permanent, dynamic, research = scoring_oracle._reason_oracle(row.abstention_reasons)
                risk = float(row.nearest_composite_distance); eligible = int(row.eligible_rows)
                effective = float(row.effective_rows); literal = bool(row.selective_lane)
            score = scoring_oracle._probability_oracle(
                [getattr(row, name) for name in probability_columns],
                [getattr(row, name) for name in directional_columns], scoring_oracle._route(outcome),
            )
            rows.append({
                "query_id": row.query_id, "query_cutoff": row.query_cutoff, "month": row.month,
                "calendar_year": int(meta.calendar_year), "fold_id": target.FINAL_FOLD, "lane": row.lane,
                "query_regime": outcome.query_regime, "quality_tier": meta.quality_tier,
                "liquidity_stratum": meta.liquidity_stratum,
                "quality_liquidity_cell": meta.quality_liquidity_cell,
                "purged_evaluation_included": True, "literal_selective": literal,
                "research_dynamic_selective": research, "permanent_product_blockers": permanent,
                "dynamic_abstention_reasons": dynamic, "risk_score": risk,
                "eligible_rows": eligible, "effective_rows": effective,
                **{name: float(getattr(row, name)) for name in probability_columns + directional_columns}, **score,
            })
    return pd.DataFrame(rows).sort_values(["query_id", "lane"], kind="stable").reset_index(drop=True)


def _continuous_scores(repository: Path, outcomes: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    root = repository / target.predictions.OUTPUT_RELATIVE
    final_ids = set(registry.query_id.astype(str))
    forecasts = pd.read_parquet(root / "continuous-predictions.parquet")
    forecasts = forecasts.loc[forecasts.query_id.astype(str).isin(final_ids)]
    observed_rows = {(str(row.query_id), int(row.horizon_sessions)): row for row in outcomes.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    for row in forecasts.itertuples(index=False):
        observed_row = observed_rows[(str(row.query_id), int(row.horizon_sessions))]
        observed = getattr(observed_row, str(row.measure))
        available = bool(observed_row.complete and pd.notna(observed) and np.isfinite(float(observed)))
        quantiles = np.asarray([row.q10, row.q25, row.q50, row.q75, row.q90], dtype=np.float64)
        valid = bool(available and np.isfinite(quantiles).all())
        if valid and (np.diff(quantiles) < 0).any():
            raise FinalVerificationError("non-monotonic quantiles")
        state: dict[str, Any] = {
            "evaluable": valid, "pinball_q10": np.nan, "pinball_q25": np.nan,
            "pinball_q50": np.nan, "pinball_q75": np.nan, "pinball_q90": np.nan,
            "median_absolute_error": np.nan, "central_50_score": np.nan,
            "central_50_width": np.nan, "central_50_covered": None,
            "central_80_score": np.nan, "central_80_width": np.nan, "central_80_covered": None,
        }
        if valid:
            actual = float(observed)
            for q, forecast in zip((.1, .25, .5, .75, .9), quantiles):
                error = actual - forecast
                state[f"pinball_q{int(q * 100):02d}"] = float(max(q * error, (q - 1) * error))
            state["median_absolute_error"] = abs(actual - quantiles[2])
            for name, lower, upper, nominal in (
                ("central_50", quantiles[1], quantiles[3], .5),
                ("central_80", quantiles[0], quantiles[4], .8),
            ):
                penalty = 0.0
                if actual < lower: penalty = 2 * (lower - actual) / (1 - nominal)
                elif actual > upper: penalty = 2 * (actual - upper) / (1 - nominal)
                state[f"{name}_score"] = float(upper - lower + penalty)
                state[f"{name}_width"] = float(upper - lower)
                state[f"{name}_covered"] = bool(lower <= actual <= upper)
        rows.append({
            "query_id": row.query_id, "query_cutoff": row.query_cutoff, "month": row.month,
            "calendar_year": pd.Timestamp(row.query_cutoff).year, "fold_id": target.FINAL_FOLD,
            "lane": row.lane, "horizon_sessions": int(row.horizon_sessions), "measure": row.measure,
            "purged_evaluation_included": True, "observed": float(observed) if available else np.nan,
            "q10": row.q10, "q25": row.q25, "q50": row.q50, "q75": row.q75, "q90": row.q90, **state,
        })
    return pd.DataFrame(rows).sort_values(
        ["query_id", "lane", "horizon_sessions", "measure"], kind="stable",
    ).reset_index(drop=True)


def _final_scope(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[frame.scope == "validation_3"].copy(); result["scope"] = target.FINAL_FOLD
    if "value" in result and "dimension" in result:
        result.loc[(result.dimension == "fold") & (result.value == "validation_3"), "value"] = target.FINAL_FOLD
    return result.reset_index(drop=True)


def _path_metrics(repository: Path, paths: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    root = repository / target.predictions.OUTPUT_RELATIVE
    forecast = pd.read_parquet(root / "path-predictions.parquet")
    forecast = forecast.loc[forecast.query_id.isin(set(registry.query_id))]
    rows: list[dict[str, Any]] = []
    for measure in target.nonfinal.PATH_MEASURES:
        column = f"{measure}_median"
        joined = forecast[["query_id", "lane", "step", column]].merge(
            paths[["query_id", "step", measure]], on=["query_id", "step"], how="left", validate="many_to_one",
        )
        joined = joined.loc[np.isfinite(joined[column]) & np.isfinite(joined[measure])].copy()
        joined["absolute_error"] = (joined[column] - joined[measure]).abs()
        for lane, group in joined.groupby("lane", sort=True):
            rows.append({"scope": target.FINAL_FOLD, "lane": lane, "measure": measure,
                         "observed_steps": len(group), "mean_absolute_error": group.absolute_error.mean(),
                         "median_absolute_error": group.absolute_error.median()})
    return pd.DataFrame(rows).sort_values(["scope", "lane", "measure"], kind="stable").reset_index(drop=True)


def _assert_frame(name: str, expected: pd.DataFrame, observed: pd.DataFrame) -> float:
    try:
        return scoring_oracle._assert_frame(name, expected, observed)
    except Exception as error:
        raise FinalVerificationError(str(error)) from error


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repository,
                      text=True, capture_output=True, check=True).stdout:
        raise FinalVerificationError("clean worktree required")
    started = perf_counter(); output = repository / target.OUTPUT_RELATIVE
    prereg = base._read(repository / target.PREREGISTRATION_RELATIVE)
    seal = base._read(output / "SEALED.json")
    if not _valid(prereg, "preregistration_digest") or not _valid(seal, timing=True) \
            or seal.get("passed") is not True or seal.get("final_period_result_opened") is not True \
            or seal.get("file_manifest") != target._manifest(output, target.OUTPUT_FILES):
        raise FinalVerificationError("preregistration/store seal differs")
    registry = target.nonfinal._registry(repository).loc[lambda x: x.fold_id == target.FINAL_FOLD].copy()
    expected_outcomes, expected_paths = _outcome_oracle(repository, registry)
    observed_outcomes = pd.read_parquet(output / "final-query-outcomes.parquet")
    observed_paths = pd.read_parquet(output / "final-query-paths.parquet")
    if oracle_tools._records(observed_outcomes, ("query_id", "horizon_sessions")) != oracle_tools._records(expected_outcomes, ("query_id", "horizon_sessions")) \
            or oracle_tools._records(observed_paths, ("query_id", "step")) != oracle_tools._records(expected_paths, ("query_id", "step")):
        raise FinalVerificationError("independent final outcome oracle differs")
    query = _query_scores(repository, expected_outcomes, registry)
    continuous = _continuous_scores(repository, expected_outcomes, registry)
    deltas = {
        "query_scores": _assert_frame("query_scores", query, pd.read_parquet(output / "query-scores.parquet")),
        "continuous_scores": _assert_frame("continuous_scores", continuous, pd.read_parquet(output / "continuous-scores.parquet")),
    }
    fake_query = query.copy(); fake_query["fold_id"] = "validation_3"
    complete_query = target._complete_nonfinal_fold_surface(query)
    fake_continuous = continuous.copy(); fake_continuous["fold_id"] = "validation_3"
    fold, stability = scoring_oracle._simple_aggregates(fake_query)
    expected = {
        "fold-metrics": _final_scope(fold),
        "stability-metrics": _final_scope(stability.assign(scope="validation_3")),
        "calibration-bins": _final_scope(scoring_oracle._calibration_oracle(fake_query)),
        "risk-coverage": _final_scope(scoring_oracle._risk_oracle(complete_query)),
        "inference": _final_scope(scoring_oracle._inference_oracle(complete_query)),
        "interval-coverage": _final_scope(scoring_oracle._interval_oracle(fake_continuous)),
        "path-metrics": _path_metrics(repository, expected_paths, registry),
    }
    for name, frame in expected.items():
        deltas[name] = _assert_frame(name, frame, pd.read_parquet(output / f"{name}.parquet"))
    coverage = base._read(output / "COVERAGE.json"); decision = base._read(output / "DECISION.json")
    primary = query.loc[query.lane == "composite"]
    if not _valid(coverage) or coverage.get("final_query_rows") != target.EXPECTED_FINAL_QUERIES \
            or coverage.get("multiclass_evaluable_queries") != int(primary.multiclass_evaluable.sum()) \
            or coverage.get("literal_selective_queries") != int(primary.literal_selective.sum()) \
            or coverage.get("final_query_outcomes_opened") is not True:
        raise FinalVerificationError("final coverage receipt differs")
    reconstructed_decision = target._decision(repository, {f"{name}.parquet": frame for name, frame in expected.items()}, coverage, prereg)
    omitted = {"created_at", "result_digest"}
    if not _valid(decision) or {k: v for k, v in decision.items() if k not in omitted} != {
        k: v for k, v in reconstructed_decision.items() if k not in omitted
    }:
        raise FinalVerificationError("final acceptance decision differs")
    state = {
        "schema_version": SCHEMA, "status": "verified", "passed": True,
        "preregistration_digest": prereg["preregistration_digest"],
        "store_result_digest": seal["result_digest"], "store_result_sha256": _sha(output / "SEALED.json"),
        "verified_final_query_count": len(registry), "verified_outcome_rows": len(expected_outcomes),
        "verified_path_rows": len(expected_paths), "maximum_numeric_delta": max(deltas.values()),
        "numeric_tolerance": NUMERIC_TOLERANCE, "component_maximum_deltas": deltas,
        "gates": {
            "all_final_outcomes_recomputed_from_raw_OHLCV": True,
            "all_final_probability_and_continuous_scores_reconstructed": True,
            "all_final_aggregates_and_inference_reconstructed": True,
            "final_acceptance_decision_reconstructed": True,
            "nonfinal_failures_and_literal_abstention_contradiction_retained": True,
            "all_physical_seals_valid": True,
        },
        "research_calibration_pass": bool(decision["research_calibration_pass"]),
        "final_period_result_opened": True, "production_promotion_authorized": False,
        "elapsed_seconds": perf_counter() - started, "created_at": _now(),
    }
    result = {**state, "verification_digest": stable_hash(state)}
    final = repository / target.VERIFICATION_RELATIVE
    if final.exists():
        existing = base._read(final / "VERIFIED.json")
        ignored = {"created_at", "elapsed_seconds", "verification_digest"}
        if {k: v for k, v in existing.items() if k not in ignored} != {k: v for k, v in result.items() if k not in ignored}:
            raise FinalVerificationError("existing final verification differs")
        return existing
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{final.name}.", dir=final.parent))
    try:
        target.smoke._atomic_json(temporary / "VERIFIED.json", result); os.replace(temporary, final)
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
