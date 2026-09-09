"""Preregister and build prediction-before-outcome WF-03D evidence."""
from __future__ import annotations

import argparse
from dataclasses import asdict
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
import pyarrow.parquet as pq

from market_analogues.adapters import source_from_spec
from market_analogues.causal_outcomes import (
    compute_prepared_episode_outcomes,
    prepare_outcome_sessions,
)
from market_analogues.config import load_config
from market_analogues.types import InstrumentKey, stable_hash
from market_analogues.walk_forward_predictions import (
    PRIMARY_CLASSES,
    QUANTILES,
    continuous_prediction,
    directional_prediction,
    pointwise_path_prediction,
    route_prediction,
)
from market_analogues.walk_forward_scoring import (
    abstention_reasons,
    expanding_frequency,
    regime_frequency,
)

from experiments.m04r import m04r14_t14_09_full_outcome_store as old_store
from experiments.m04r import m04r14_t14_09_outcome_smoke as smoke
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_10_wf03d_cross_store_manifest as cross_store
from experiments.m04r import m04r14_t14_10_wf03d_outcome_store as outcome_store


SCHEMA = "m04r14-t14-10-wf03d-prediction-store-preregistration-v1"
MONTH_SCHEMA = "m04r14-t14-10-wf03d-prediction-month-v1"
STORE_SCHEMA = "m04r14-t14-10-wf03d-prediction-store-v1"
PREREGISTRATION_RELATIVE = Path(
    "experiments/m04r/m04r14_t14_10_wf03d_prediction_store_v1_preregistered.json"
)
CACHE_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf03d-prediction-months-v1"
)
OUTPUT_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf03d-prediction-store-v1"
)
VERIFICATION_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf03d-prediction-store-v1-verification"
)
SYNTHETIC_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf03d-prediction-synthetic-v1/VERIFIED.json"
)
WALK_FORWARD_CONTRACT = Path("config/m04r14-t14-10-walk-forward-contract.json")
REGISTRY_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-walk-forward-query-registry-v1/query-registry.parquet"
)
REGISTRY_VERIFICATION = Path(
    "config/data/analogues/m04r14/t14-10-walk-forward-query-registry-v1-verification/VERIFIED.json"
)
SOURCE_ACCOUNTING = base.REGISTRY_RELATIVE / "source-accounting.parquet"
FINAL_START = pd.Timestamp("2024-01-01")
EXPECTED_QUERIES = 3_936
EXPECTED_MONTHS = 164
EXPECTED_NONFINAL_QUERIES = 3_456
EXPECTED_FINAL_QUERIES = 480
PREFIXES = (5, 10, 15, 20)
NEIGHBOR_LANES = (
    "composite", "composite_unweighted", "price_only",
    "deterministic_random", "recent_return_volatility",
)
MEASURES = (
    "close_return", "benchmark_relative_return",
    "maximum_favorable_excursion", "maximum_adverse_excursion",
)
PATH_MEASURES = (
    "close_return", "benchmark_relative_close_return",
    "atr_normalized_close_move", "high_return", "low_return",
)
RUNTIME_FILES = (
    "experiments/m04r/m04r14_t14_10_wf03d_prediction_store.py",
    "experiments/m04r/verify_m04r14_t14_10_wf03d_prediction_store.py",
    "experiments/m04r/m04r14_t14_10_wf03d_prediction_synthetic_gate.py",
    "experiments/m04r/m04r14_t14_10_wf03d_prediction_listener.py",
    "src/market_analogues/walk_forward_predictions.py",
    "src/market_analogues/walk_forward_scoring.py",
    "experiments/m04r/m04r14_t14_10_wf03d_outcome_store.py",
    "experiments/m04r/verify_m04r14_t14_10_wf03d_outcome_store.py",
    "config/m04r14-t14-10-walk-forward-contract.json",
    "pyproject.toml",
)


class WalkForwardPredictionError(RuntimeError):
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
        raise WalkForwardPredictionError(error.strip() or "git command failed")
    return result.stdout if raw else result.stdout.strip()


def _sha(path: Path) -> str:
    digest = sha256()
    if path.is_symlink() or not path.is_file():
        raise WalkForwardPredictionError(f"regular file required: {path}")
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _seal(state: Mapping[str, Any], key: str = "result_digest", *, timing: bool = False) -> dict[str, Any]:
    omitted = {"elapsed_seconds"} if timing else set()
    return {
        **state,
        key: stable_hash({name: value for name, value in state.items() if name not in omitted}),
        "created_at": _now(),
    }


def _valid_seal(value: Mapping[str, Any], key: str = "result_digest", *, timing: bool = False) -> bool:
    omitted = {key, "created_at"} | (
        {"elapsed_seconds", "partition_elapsed_seconds"} if timing else set()
    )
    return value.get(key) == stable_hash({
        name: item for name, item in value.items() if name not in omitted
    })


def _runtime_manifest(repository: Path, head: str) -> dict[str, str]:
    tracked = set(str(_git(repository, "ls-tree", "-r", "--name-only", head)).splitlines())
    if any(name not in tracked for name in RUNTIME_FILES):
        raise WalkForwardPredictionError("runtime manifest contains uncommitted files")
    return {
        name: sha256(_git(repository, "show", f"{head}:{name}", raw=True)).hexdigest()
        for name in RUNTIME_FILES
    }


def _frame_digest(frame: pd.DataFrame, order: Sequence[str]) -> str:
    ordered = frame.sort_values(list(order), kind="stable").reset_index(drop=True)
    return stable_hash(_plain(ordered.to_dict("records")))


def _registry(repository: Path) -> pd.DataFrame:
    frame = pd.read_parquet(repository / REGISTRY_RELATIVE, engine="pyarrow")
    required = {
        "case_id", "episode_id", "symbol", "cutoff", "fold_id", "fold_role",
        "scored", "quality_tier", "liquidity_stratum",
    }
    if len(frame) != EXPECTED_QUERIES or not required.issubset(frame.columns) \
            or frame.episode_id.astype(str).duplicated().any():
        raise WalkForwardPredictionError("walk-forward registry differs")
    frame = frame.copy()
    frame["query_id"] = frame.episode_id.astype(str)
    frame["month"] = pd.to_datetime(frame.cutoff).dt.to_period("M").astype(str)
    return frame.sort_values(["cutoff", "query_id"], kind="stable").reset_index(drop=True)


def _verified_inputs(repository: Path) -> dict[str, Any]:
    outcome_root = repository / outcome_store.OUTPUT_RELATIVE
    outcome_seal = base._read(outcome_root / "SEALED.json")
    verified_path = repository / outcome_store.VERIFICATION_RELATIVE / "VERIFIED.json"
    verified = base._read(verified_path)
    registry_verified = base._read(repository / REGISTRY_VERIFICATION)
    contract = base._read(repository / WALK_FORWARD_CONTRACT)
    synthetic = base._read(repository / SYNTHETIC_RELATIVE)
    if not all((
        _valid_seal(outcome_seal, timing=True), outcome_seal.get("passed") is True,
        verified.get("verification_digest") == stable_hash({
            key: value for key, value in verified.items() if key != "verification_digest"
        }),
        verified.get("passed") is True,
        verified.get("store_result_digest") == outcome_seal.get("result_digest"),
        verified.get("prediction_store_construction_authorized") is True,
        verified.get("historical_query_evaluation_opened") is False,
        verified.get("final_period_result_opened") is False,
        registry_verified.get("passed") is True,
        registry_verified.get("historical_walk_forward_query_outcomes_opened") is False,
        contract.get("contract_digest") == "2c65288b45dd79627a1e16f0a4c2d9449919d0602974d7a52440415bd7f96a8e",
        synthetic.get("passed") is True,
        _valid_seal(synthetic),
        synthetic.get("real_query_outcomes_accessed") is False,
        synthetic.get("final_period_result_opened") is False,
    )):
        raise WalkForwardPredictionError("verified D3/registry boundary differs")
    return {
        "outcome_store_result_digest": outcome_seal["result_digest"],
        "outcome_store_seal_sha256": _sha(outcome_root / "SEALED.json"),
        "outcome_verification_digest": verified["verification_digest"],
        "outcome_verification_sha256": _sha(verified_path),
        "registry_verification_digest": registry_verified["result_digest"],
        "registry_verification_sha256": _sha(repository / REGISTRY_VERIFICATION),
        "walk_forward_contract_digest": contract["contract_digest"],
        "walk_forward_contract_sha256": _sha(repository / WALK_FORWARD_CONTRACT),
        "synthetic_result_digest": synthetic["result_digest"],
        "synthetic_implementation_h0": synthetic["implementation_h0"],
        "synthetic_sha256": _sha(repository / SYNTHETIC_RELATIVE),
        "raw_links_sha256": _sha(
            repository / cross_store.OUTPUT_RELATIVE / cross_store.LINK_FILE
        ),
        "analogue_outcomes_sha256": _sha(outcome_root / "episode-outcomes.parquet"),
        "analogue_paths_sha256": _sha(outcome_root / "future-paths.parquet"),
        "analogue_eligibility_sha256": _sha(outcome_root / "link-outcome-eligibility.parquet"),
    }


def _benchmark_regimes(repository: Path, registry: pd.DataFrame) -> pd.DataFrame:
    source = source_from_spec(load_config(repository / base.CONFIG_RELATIVE).datasets["nasdaq"])
    benchmark = source.load_benchmark()
    if benchmark is None or benchmark.attrs.get("source_timestamp_reordered") \
            or benchmark.attrs.get("source_duplicate_timestamps"):
        raise WalkForwardPredictionError("benchmark source differs")
    bars = benchmark.sort_values("timestamp", kind="stable").reset_index(drop=True)
    timestamps = pd.to_datetime(bars.timestamp)
    close = bars.close.to_numpy(dtype=np.float64)
    positions = {stamp: index for index, stamp in enumerate(timestamps)}
    monthly: list[dict[str, Any]] = []
    prior_volatility: list[float] = []
    for month, rows in registry.groupby("month", sort=True):
        cutoff = pd.Timestamp(rows.cutoff.iloc[0])
        position = positions.get(cutoff)
        if position is None or position < 126:
            raise WalkForwardPredictionError(f"benchmark cutoff lacks history: {month}")
        trailing_return = float(close[position] / close[position - 126] - 1.0)
        returns = np.diff(np.log(close[position - 20:position + 1]))
        volatility = float(np.std(returns, ddof=0))
        trend = "up" if trailing_return > 0 else ("down" if trailing_return < 0 else "flat")
        if len(prior_volatility) < 2:
            volatility_bucket = "insufficient_prior_reference"
            lower = upper = None
        else:
            lower, upper = np.quantile(
                np.asarray(prior_volatility), [1 / 3, 2 / 3], method="linear",
            )
            volatility_bucket = "low" if volatility <= lower else (
                "mid" if volatility <= upper else "high"
            )
            lower, upper = float(lower), float(upper)
        monthly.append({
            "month": str(month), "cutoff": cutoff.isoformat(),
            "benchmark_return_126": trailing_return,
            "benchmark_volatility_20": volatility,
            "prior_volatility_cutoffs": len(prior_volatility),
            "lower_tercile": lower, "upper_tercile": upper,
            "trend": trend, "volatility_bucket": volatility_bucket,
            "regime": f"trend={trend}|volatility={volatility_bucket}",
        })
        prior_volatility.append(volatility)
    result = pd.DataFrame(monthly)
    if len(result) != EXPECTED_MONTHS:
        raise WalkForwardPredictionError("benchmark regime month count differs")
    return result


def _query_overlap(repository: Path, registry: pd.DataFrame) -> list[str]:
    ids = pd.read_parquet(
        repository / outcome_store.OUTPUT_RELATIVE / "episode-outcomes.parquet",
        columns=["episode_id"], engine="pyarrow",
    ).episode_id.astype(str).drop_duplicates()
    return sorted(set(registry.query_id).intersection(ids))


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise WalkForwardPredictionError("globally clean Git worktree required")
    if (repository / CACHE_RELATIVE).exists() or (repository / OUTPUT_RELATIVE).exists():
        raise WalkForwardPredictionError("prediction cache/output must be absent")
    h0 = str(_git(repository, "rev-parse", "HEAD"))
    verified_inputs = _verified_inputs(repository)
    if verified_inputs["synthetic_implementation_h0"] != h0:
        raise WalkForwardPredictionError("synthetic verifier is not bound to implementation H0")
    registry = _registry(repository)
    regimes = _benchmark_regimes(repository, registry)
    overlap = _query_overlap(repository, registry)
    final_ids = set(registry.loc[pd.to_datetime(registry.cutoff) >= FINAL_START, "query_id"])
    final_overlap = sorted(final_ids.intersection(overlap))
    if final_overlap:
        raise WalkForwardPredictionError("final registry outcomes already exist in analogue namespace")
    months = registry.month.drop_duplicates().tolist()
    state = {
        "schema_version": SCHEMA,
        "status": "frozen_before_historical_query_outcome_access",
        "implementation_h0": h0,
        "runtime_files": _runtime_manifest(repository, h0),
        "verified_inputs": verified_inputs,
        "query_count": len(registry), "month_count": len(months),
        "nonfinal_query_count": int((pd.to_datetime(registry.cutoff) < FINAL_START).sum()),
        "final_query_count": int((pd.to_datetime(registry.cutoff) >= FINAL_START).sum()),
        "month_order": months, "registry_digest": _frame_digest(registry, ("cutoff", "query_id")),
        "regime_digest": _frame_digest(regimes, ("month",)),
        "registry_episode_overlap_count": len(overlap),
        "registry_episode_overlap_digest": stable_hash(overlap),
        "final_registry_episode_overlap_count": len(final_overlap),
        "registry_episode_overlap_policy": "analogue_namespace_only_never_baseline_or_evaluation_state",
        "baseline_state_source": "only_prior_month_closed_query_outcome_receipts",
        "final_period_policy": "seal_all_final_predictions_without_opening_any_final_query_outcome",
        "primary_classes": list(PRIMARY_CLASSES), "prefixes": list(PREFIXES),
        "neighbor_lanes": list(NEIGHBOR_LANES), "continuous_measures": list(MEASURES),
        "path_measures": list(PATH_MEASURES), "quantiles": list(QUANTILES),
        "expected_neighbor_prediction_rows": len(registry) * len(NEIGHBOR_LANES) * len(PREFIXES),
        "expected_baseline_prediction_rows": len(registry) * 2,
        "expected_continuous_prediction_rows": len(registry) * len(NEIGHBOR_LANES) * 6 * len(MEASURES),
        "expected_path_prediction_rows": len(registry) * len(NEIGHBOR_LANES) * 126,
        "historical_analogue_outcomes_accessed": True,
        "historical_query_outcomes_accessed": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
    }
    if state["nonfinal_query_count"] != EXPECTED_NONFINAL_QUERIES \
            or state["final_query_count"] != EXPECTED_FINAL_QUERIES:
        raise WalkForwardPredictionError("final/nonfinal registry split differs")
    return _seal(state, "preregistration_digest")


def _sole_child(repository: Path, raw: bytes, h0: str) -> str:
    accepted = []
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
        raise WalkForwardPredictionError("expected one exact preregistration-only child")
    return accepted[0]


def validate_preregistration(repository: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, str]:
    if _git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise WalkForwardPredictionError("globally clean Git worktree required")
    raw = (repository / PREREGISTRATION_RELATIVE).read_bytes()
    prereg = base._read(repository / PREREGISTRATION_RELATIVE)
    if not _valid_seal(prereg, "preregistration_digest") or prereg.get("schema_version") != SCHEMA:
        raise WalkForwardPredictionError("prediction preregistration seal differs")
    h0 = str(prereg["implementation_h0"])
    h1 = _sole_child(repository, raw, h0)
    if subprocess.run(["git", "merge-base", "--is-ancestor", h1, "HEAD"], cwd=repository).returncode:
        raise WalkForwardPredictionError("HEAD does not descend from prediction preregistration")
    for name, expected in prereg["runtime_files"].items():
        if _sha(repository / name) != expected \
                or sha256(_git(repository, "show", f"{h0}:{name}", raw=True)).hexdigest() != expected:
            raise WalkForwardPredictionError(f"runtime source drifted: {name}")
    registry = _registry(repository)
    regimes = _benchmark_regimes(repository, registry)
    overlap = _query_overlap(repository, registry)
    final_ids = set(registry.loc[pd.to_datetime(registry.cutoff) >= FINAL_START, "query_id"])
    observed = {
        "verified_inputs": _verified_inputs(repository),
        "registry_digest": _frame_digest(registry, ("cutoff", "query_id")),
        "regime_digest": _frame_digest(regimes, ("month",)),
        "registry_episode_overlap_count": len(overlap),
        "registry_episode_overlap_digest": stable_hash(overlap),
        "final_registry_episode_overlap_count": len(final_ids.intersection(overlap)),
    }
    if any(prereg.get(key) != value for key, value in observed.items()):
        raise WalkForwardPredictionError("prediction preregistration inputs drifted")
    return prereg, registry, regimes, h1


def _load_prediction_inputs(repository: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    links = pd.read_parquet(
        repository / cross_store.OUTPUT_RELATIVE / cross_store.LINK_FILE,
        engine="pyarrow",
    )
    outcomes = pd.read_parquet(
        repository / outcome_store.OUTPUT_RELATIVE / "episode-outcomes.parquet",
        engine="pyarrow",
    )
    eligibility = pd.read_parquet(
        repository / outcome_store.OUTPUT_RELATIVE / "link-outcome-eligibility.parquet",
        engine="pyarrow",
    )
    if len(links) != cross_store.EXPECTED_LINKS \
            or len(outcomes) != outcome_store.EXPECTED_REQUESTS * len(outcome_store.HORIZONS) \
            or len(eligibility) != cross_store.EXPECTED_LINKS * len(outcome_store.HORIZONS):
        raise WalkForwardPredictionError("prediction input row counts differ")
    return links, outcomes, eligibility


def _lane_source(lane: str) -> tuple[str, bool]:
    return ("composite", False) if lane == "composite_unweighted" else (lane, True)


def _primary_predictions(
    month_registry: pd.DataFrame, links: pd.DataFrame, outcomes: pd.DataFrame,
    eligibility: pd.DataFrame, prior_distances: Sequence[float],
    *, prepared: pd.DataFrame | None = None,
) -> pd.DataFrame:
    merged = prepared if prepared is not None else _prepare_primary(links, outcomes, eligibility)
    rows: list[dict[str, Any]] = []
    for query in month_registry.itertuples(index=False):
        query_rows = merged.loc[merged.query_id == query.query_id]
        lane_predictions: dict[tuple[Any, ...], Any] = {}
        for lane in NEIGHBOR_LANES:
            method, weighted = _lane_source(lane)
            source_rows = query_rows.loc[query_rows.method == method].sort_values("rank")
            if len(source_rows) != 20:
                raise WalkForwardPredictionError(f"neighbor lane incomplete: {query.query_id}:{lane}")
            for prefix in PREFIXES:
                selected = source_rows.iloc[:prefix]
                routes = [
                    str(label) if bool(is_eligible) else (
                        "censored" if str(reason) == "incomplete_horizon" else None
                    )
                    for label, is_eligible, reason in zip(
                        selected.barrier_label, selected.eligible, selected.reason,
                    )
                ]
                prediction = route_prediction(
                    routes, [int(value) for value in selected["rank"]], weighted=weighted,
                )
                lane_predictions[(lane, prefix)] = prediction
                lane_predictions[(lane, prefix, "routes")] = routes
        composite = lane_predictions[("composite", 20)]
        prefix_probabilities = [
            lane_predictions[("composite", prefix)].probabilities[0]
            for prefix in PREFIXES
        ]
        nearest_hex = query_rows.loc[
            (query_rows.method == "composite") & (query_rows["rank"] == 1), "distance_hex",
        ].iloc[0]
        nearest = float.fromhex(str(nearest_hex))
        reasons = abstention_reasons(
            eligible_primary_rows=composite.eligible_rows,
            nearest_distance=nearest,
            prior_nearest_distances=prior_distances,
            favorable_prefix_probabilities=prefix_probabilities,
        )
        novelty_threshold = float(np.quantile(
            np.asarray(prior_distances), .95, method="linear",
        )) if len(prior_distances) >= 250 else np.nan
        instability = max(prefix_probabilities) - min(prefix_probabilities)
        for lane in NEIGHBOR_LANES:
            for prefix in PREFIXES:
                prediction = lane_predictions[(lane, prefix)]
                method, weighted = _lane_source(lane)
                directional = directional_prediction(
                    lane_predictions[(lane, prefix, "routes")],
                    list(range(1, prefix + 1)), weighted=weighted,
                )
                rows.append({
                    "query_id": query.query_id, "query_case_id": query.case_id,
                    "query_cutoff": query.cutoff, "month": query.month,
                    "fold_id": query.fold_id, "scored": bool(query.scored),
                    "lane": lane, "prefix": prefix,
                    "weighted": lane != "composite_unweighted",
                    **asdict(prediction),
                    "favorable_probability": prediction.probabilities[0],
                    "adverse_probability": prediction.probabilities[1],
                    "no_touch_probability": prediction.probabilities[2],
                    "directional_favorable_probability": directional.favorable_probability,
                    "directional_adverse_probability": directional.adverse_probability,
                    "directional_eligible_rows": directional.eligible_rows,
                    "directional_effective_rows": directional.effective_rows,
                    "nearest_composite_distance": nearest,
                    "novelty_reference_rows": len(prior_distances),
                    "novelty_threshold": novelty_threshold,
                    "neighborhood_instability": instability,
                    "abstention_reasons": "|".join(reasons),
                    "forced_score_lane": True,
                    "selective_lane": not reasons,
                })
    result = pd.DataFrame(rows).drop(columns="probabilities")
    return result.sort_values(["query_id", "lane", "prefix"], kind="stable").reset_index(drop=True)


def _prepare_primary(
    links: pd.DataFrame, outcomes: pd.DataFrame, eligibility: pd.DataFrame,
) -> pd.DataFrame:
    horizon = outcomes.loc[outcomes.horizon_sessions == 20, [
        "episode_id", "barrier_label",
    ]].rename(columns={"episode_id": "matched_episode_id"})
    eligible = eligibility.loc[eligibility.horizon_sessions == 20, [
        "query_id", "method", "rank", "eligible", "reason",
    ]]
    return links.merge(
        eligible, on=["query_id", "method", "rank"], how="left", validate="one_to_one",
    ).merge(horizon, on="matched_episode_id", how="left", validate="many_to_one")


def _baseline_predictions(
    month_registry: pd.DataFrame, prior_outcomes: pd.DataFrame,
    regime: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for query in month_registry.itertuples(index=False):
        cutoff = pd.Timestamp(query.cutoff)
        prior = prior_outcomes.loc[
            (pd.to_datetime(prior_outcomes.completion_timestamp) <= cutoff)
            & prior_outcomes.barrier_label.isin(PRIMARY_CLASSES)
        ] if len(prior_outcomes) else prior_outcomes
        labels = prior.barrier_label.astype(str).tolist()
        unconditional = expanding_frequency(labels, PRIMARY_CLASSES)
        regimes = prior.regime.astype(str).tolist() if len(prior) else []
        regime_values, fallback = regime_frequency(
            labels, regimes, regime, PRIMARY_CLASSES,
        )
        unconditional_directional = directional_prediction(
            labels, list(range(1, len(labels) + 1)), weighted=False,
        ) if labels else directional_prediction([None], [1], weighted=False)
        same_regime_labels = [
            label for label, value in zip(labels, regimes) if value == regime
        ]
        regime_directional_labels = labels if fallback else same_regime_labels
        regime_directional = directional_prediction(
            regime_directional_labels,
            list(range(1, len(regime_directional_labels) + 1)), weighted=False,
        ) if regime_directional_labels else directional_prediction([None], [1], weighted=False)
        same = sum(value == regime for value in regimes)
        for lane, probabilities, directional, used_fallback in (
            ("unconditional_market_frequency", unconditional, unconditional_directional, False),
            ("regime_only_frequency", regime_values, regime_directional, fallback),
        ):
            rows.append({
                "query_id": query.query_id, "query_case_id": query.case_id,
                "query_cutoff": query.cutoff, "month": query.month,
                "fold_id": query.fold_id, "scored": bool(query.scored),
                "lane": lane, "prior_eligible_rows": len(prior),
                "prior_same_regime_rows": same, "query_regime": regime,
                "fallback_to_unconditional": bool(used_fallback),
                "favorable_probability": float(probabilities[0]),
                "adverse_probability": float(probabilities[1]),
                "no_touch_probability": float(probabilities[2]),
                "directional_favorable_probability": directional.favorable_probability,
                "directional_adverse_probability": directional.adverse_probability,
                "directional_eligible_rows": directional.eligible_rows,
                "directional_effective_rows": directional.effective_rows,
                "forced_score_lane": True,
            })
    return pd.DataFrame(rows).sort_values(["query_id", "lane"], kind="stable").reset_index(drop=True)


def _continuous_predictions(
    month_registry: pd.DataFrame, links: pd.DataFrame, outcomes: pd.DataFrame,
    eligibility: pd.DataFrame, *, prepared: pd.DataFrame | None = None,
) -> pd.DataFrame:
    joined = prepared if prepared is not None else _prepare_continuous(
        links, outcomes, eligibility,
    )
    rows: list[dict[str, Any]] = []
    for query in month_registry.itertuples(index=False):
        query_rows = joined.loc[joined.query_id == query.query_id]
        for lane in NEIGHBOR_LANES:
            method, weighted = _lane_source(lane)
            lane_rows = query_rows.loc[query_rows.method == method]
            for horizon in outcome_store.HORIZONS:
                selected = lane_rows.loc[lane_rows.horizon_sessions == horizon].sort_values("rank")
                if len(selected) != 20:
                    raise WalkForwardPredictionError("continuous lane is incomplete")
                for measure in MEASURES:
                    values = [
                        float(value) if bool(ok) and pd.notna(value) else None
                        for value, ok in zip(selected[measure], selected.eligible)
                    ]
                    prediction = continuous_prediction(
                        values, [int(value) for value in selected["rank"]], weighted=weighted,
                    )
                    rows.append({
                        "query_id": query.query_id, "query_cutoff": query.cutoff,
                        "month": query.month, "fold_id": query.fold_id,
                        "lane": lane, "horizon_sessions": int(horizon),
                        "measure": measure, "weighted": weighted,
                        "eligible_rows": prediction.eligible_rows,
                        "effective_rows": prediction.effective_rows,
                        **{f"q{int(q * 100):02d}": value for q, value in zip(QUANTILES, prediction.quantiles)},
                    })
    return pd.DataFrame(rows).sort_values(
        ["query_id", "lane", "horizon_sessions", "measure"], kind="stable",
    ).reset_index(drop=True)


def _prepare_continuous(
    links: pd.DataFrame, outcomes: pd.DataFrame, eligibility: pd.DataFrame,
) -> pd.DataFrame:
    return links[["query_id", "method", "rank", "matched_episode_id"]].merge(
        eligibility[["query_id", "method", "rank", "horizon_sessions", "eligible"]],
        on=["query_id", "method", "rank"], how="left", validate="one_to_many",
    ).merge(
        outcomes[["episode_id", "horizon_sessions", *MEASURES]].rename(
            columns={"episode_id": "matched_episode_id"}
        ), on=["matched_episode_id", "horizon_sessions"], how="left", validate="many_to_one",
    )


class _PathIndex:
    def __init__(self, path: Path):
        parquet = pq.ParquetFile(path)
        total = parquet.metadata.num_rows
        self.step = np.empty(total, dtype=np.int16)
        self.timestamp = np.empty(total, dtype="datetime64[ns]")
        self.expected = np.empty(total, dtype=bool)
        self.values = {name: np.empty(total, dtype=np.float64) for name in PATH_MEASURES}
        self.slices: dict[str, tuple[int, int]] = {}
        offset = 0
        active: str | None = None
        active_start = 0
        columns = ["episode_id", "step", "timestamp", "expected_session_match", *PATH_MEASURES]
        for index in range(parquet.metadata.num_row_groups):
            frame = parquet.read_row_group(index, columns=columns).to_pandas()
            count = len(frame); end = offset + count
            ids = frame.episode_id.astype(str).to_numpy()
            changes = np.flatnonzero(ids[1:] != ids[:-1]) + 1
            starts = np.r_[0, changes]; stops = np.r_[changes, count]
            for start, stop in zip(starts, stops):
                episode = str(ids[start])
                global_start, global_stop = offset + int(start), offset + int(stop)
                if active is not None and episode != active:
                    self.slices[active] = (active_start, global_start)
                    active = None
                if active is None:
                    active, active_start = episode, global_start
                if stop < count:
                    self.slices[episode] = (active_start, global_stop)
                    active = None
            self.step[offset:end] = frame.step.to_numpy(dtype=np.int16)
            self.timestamp[offset:end] = pd.to_datetime(frame.timestamp).to_numpy(dtype="datetime64[ns]")
            self.expected[offset:end] = frame.expected_session_match.to_numpy(dtype=bool)
            for name in PATH_MEASURES:
                self.values[name][offset:end] = frame[name].to_numpy(dtype=np.float64)
            offset = end
        if active is not None:
            self.slices[active] = (active_start, offset)
        if offset != total:
            raise WalkForwardPredictionError("path index row count differs")


def _path_predictions(
    month_registry: pd.DataFrame, links: pd.DataFrame, paths: _PathIndex,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for query in month_registry.itertuples(index=False):
        query_links = links.loc[links.query_id == query.query_id]
        query_timestamp = np.datetime64(pd.Timestamp(query.cutoff), "ns")
        for lane in NEIGHBOR_LANES:
            method, weighted = _lane_source(lane)
            selected = query_links.loc[query_links.method == method].sort_values("rank")
            matrices = {name: np.full((20, 126), np.nan) for name in PATH_MEASURES}
            ranks = selected["rank"].astype(int).tolist()
            for row_index, match in enumerate(selected.itertuples(index=False)):
                bounds = paths.slices.get(str(match.matched_episode_id))
                if bounds is None:
                    continue
                start, stop = bounds
                valid = paths.expected[start:stop] & (paths.timestamp[start:stop] <= query_timestamp)
                steps = paths.step[start:stop][valid].astype(int)
                within = (steps >= 1) & (steps <= 126)
                for name in PATH_MEASURES:
                    matrices[name][row_index, steps[within] - 1] = paths.values[name][start:stop][valid][within]
            summaries = {
                name: _path_matrix_summary(matrix, ranks, weighted)
                for name, matrix in matrices.items()
            }
            for step in range(126):
                payload: dict[str, Any] = {
                    "query_id": query.query_id, "query_cutoff": query.cutoff,
                    "month": query.month, "fold_id": query.fold_id,
                    "lane": lane, "step": step + 1, "weighted": weighted,
                }
                for name in PATH_MEASURES:
                    median, count, ess = summaries[name]
                    payload[f"{name}_median"] = median[step]
                    payload[f"{name}_rows"] = count[step]
                    payload[f"{name}_effective_rows"] = ess[step]
                rows.append(payload)
    return pd.DataFrame(rows).sort_values(["query_id", "lane", "step"], kind="stable").reset_index(drop=True)


def _path_matrix_summary(
    matrix: np.ndarray, ranks: Sequence[int], weighted: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if matrix.shape != (len(ranks), 126):
        raise WalkForwardPredictionError("path matrix shape differs")
    weights = np.asarray([
        2.0 ** (-(int(rank) - 1) / 10.0) if weighted else 1.0
        for rank in ranks
    ], dtype=np.float64)[:, None]
    valid = np.isfinite(matrix)
    order = np.argsort(np.where(valid, matrix, np.inf), axis=0, kind="stable")
    sorted_values = np.take_along_axis(matrix, order, axis=0)
    sorted_weights = np.take_along_axis(
        np.broadcast_to(weights, matrix.shape), order, axis=0,
    ) * np.take_along_axis(valid, order, axis=0)
    cumulative = np.cumsum(sorted_weights, axis=0)
    total = cumulative[-1]
    positions = np.argmax(cumulative >= .5 * total, axis=0)
    median = sorted_values[positions, np.arange(matrix.shape[1])]
    count = valid.sum(axis=0).astype(np.int16)
    square_total = np.square(sorted_weights).sum(axis=0)
    ess = np.divide(
        np.square(total), square_total, out=np.zeros_like(total), where=square_total > 0,
    )
    median[count == 0] = np.nan
    return median, count, ess


def _month_file_manifest(root: Path, names: Sequence[str]) -> list[dict[str, Any]]:
    return [{"path": name, "bytes": (root / name).stat().st_size, "sha256": _sha(root / name)} for name in names]


PREDICTION_FILES = (
    "raw-predictions.parquet", "baseline-predictions.parquet",
    "continuous-predictions.parquet", "path-predictions.parquet",
)


def _build_month_predictions(
    repository: Path, cache: Path, prereg: Mapping[str, Any], h1: str,
    month: str, month_registry: pd.DataFrame, regime: Mapping[str, Any],
    links: pd.DataFrame, outcomes: pd.DataFrame, eligibility: pd.DataFrame,
    paths: _PathIndex, prior_outcomes: pd.DataFrame, prior_distances: Sequence[float],
    primary_prepared: pd.DataFrame, continuous_prepared: pd.DataFrame,
) -> Path:
    final = cache / f"month-{month}"
    if final.exists():
        return final
    temporary = Path(tempfile.mkdtemp(prefix=f".prediction-{month}-", dir=cache.parent))
    started = perf_counter()
    try:
        primary = _primary_predictions(
            month_registry, links, outcomes, eligibility, prior_distances,
            prepared=primary_prepared,
        )
        baselines = _baseline_predictions(month_registry, prior_outcomes, str(regime["regime"]))
        continuous = _continuous_predictions(
            month_registry, links, outcomes, eligibility, prepared=continuous_prepared,
        )
        path_predictions = _path_predictions(month_registry, links, paths)
        frames = dict(zip(PREDICTION_FILES, (primary, baselines, continuous, path_predictions)))
        for name, frame in frames.items():
            smoke._atomic_parquet(temporary / name, frame)
        state = {
            "schema_version": MONTH_SCHEMA, "status": "predictions_sealed", "passed": True,
            "month": month, "query_count": len(month_registry),
            "query_digest": stable_hash(month_registry.query_id.astype(str).tolist()),
            "preregistration_h1": h1,
            "preregistration_digest": prereg["preregistration_digest"],
            "regime": _plain(regime),
            "prior_closed_query_outcome_rows": len(prior_outcomes),
            "prior_novelty_reference_rows": len(prior_distances),
            "neighbor_prediction_rows": len(primary),
            "baseline_prediction_rows": len(baselines),
            "continuous_prediction_rows": len(continuous),
            "path_prediction_rows": len(path_predictions),
            "semantic_digests": {
                "neighbor": _frame_digest(primary, ("query_id", "lane", "prefix")),
                "baseline": _frame_digest(baselines, ("query_id", "lane")),
                "continuous": _frame_digest(continuous, ("query_id", "lane", "horizon_sessions", "measure")),
                "path": _frame_digest(path_predictions, ("query_id", "lane", "step")),
            },
            "file_manifest": _month_file_manifest(temporary, PREDICTION_FILES),
            "elapsed_seconds": perf_counter() - started,
            "historical_analogue_outcomes_accessed": True,
            "query_outcomes_accessed_before_prediction_seal": False,
            "final_period_result_opened": False,
        }
        smoke._atomic_json(temporary / "PREDICTIONS_SEALED.json", _seal(state, timing=True))
        os.replace(temporary, final)
        return final
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _source_fingerprints(repository: Path) -> dict[str, str]:
    accounting = pd.read_parquet(repository / SOURCE_ACCOUNTING, engine="pyarrow")
    return {
        str(row.symbol): str(row.source_hash_at_lock)
        for row in accounting.itertuples(index=False) if row.error is None
    }


def _compute_query_outcomes(
    repository: Path, month_registry: pd.DataFrame, regime: str,
    fingerprints: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    contract = base._read(repository / outcome_store.CONTRACT)
    source_content_digest = base._resident()["content_digest"]
    source = source_from_spec(load_config(repository / base.CONFIG_RELATIVE).datasets["nasdaq"])
    benchmark = source.load_benchmark()
    if benchmark is None:
        raise WalkForwardPredictionError("benchmark unavailable for query outcomes")
    prepared_benchmark = prepare_outcome_sessions(benchmark, "benchmark")
    outcome_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    for symbol, query_rows in month_registry.groupby("symbol", sort=True):
        key = InstrumentKey("nasdaq", str(symbol))
        stock = source.load(key)
        fingerprint = source.fingerprint(key)
        if fingerprints.get(str(symbol)) != fingerprint:
            raise WalkForwardPredictionError(f"query source fingerprint differs: {symbol}")
        prepared_stock = prepare_outcome_sessions(stock, f"stock:{symbol}")
        timestamps = set(pd.to_datetime(stock.timestamp))
        for query in query_rows.itertuples(index=False):
            cutoff = pd.Timestamp(query.cutoff)
            if cutoff not in timestamps:
                raise WalkForwardPredictionError(f"query cutoff absent: {query.case_id}")
            bundle = compute_prepared_episode_outcomes(
                prepared_stock, prepared_benchmark, episode_id=query.query_id,
                cutoff=cutoff, source_fingerprint=fingerprint,
                contract_digest=contract["contract_digest"],
                source_content_digest=source_content_digest,
            )
            observed = bundle.outcomes.copy()
            observed["query_id"] = query.query_id
            observed["query_regime"] = regime
            outcome_rows.extend(_plain(observed.to_dict("records")))
            paths = bundle.paths.copy(); paths["query_id"] = query.query_id
            path_rows.extend(_plain(paths.to_dict("records")))
    outcomes = pd.DataFrame(outcome_rows).sort_values(["query_id", "horizon_sessions"], kind="stable").reset_index(drop=True)
    paths = pd.DataFrame(path_rows).sort_values(["query_id", "step"], kind="stable").reset_index(drop=True)
    return outcomes, paths


def _close_month(
    repository: Path, root: Path, month_registry: pd.DataFrame,
    regime: str, fingerprints: Mapping[str, str],
) -> None:
    closed = root / "MONTH_CLOSED.json"
    if closed.exists():
        return
    prediction_seal = base._read(root / "PREDICTIONS_SEALED.json")
    if not _valid_seal(prediction_seal, timing=True):
        raise WalkForwardPredictionError("prediction seal invalid before query outcome access")
    started = perf_counter()
    outcomes, paths = _compute_query_outcomes(
        repository, month_registry, regime, fingerprints,
    )
    stage = Path(tempfile.mkdtemp(prefix=f".outcome-{prediction_seal['month']}-", dir=root.parent))
    try:
        smoke._atomic_parquet(stage / "query-outcomes.parquet", outcomes)
        smoke._atomic_parquet(stage / "query-paths.parquet", paths)
        for name in ("query-outcomes.parquet", "query-paths.parquet"):
            target = root / name
            if target.exists():
                if _sha(target) != _sha(stage / name):
                    raise WalkForwardPredictionError(
                        f"partial month outcome differs: {prediction_seal['month']}:{name}"
                    )
                (stage / name).unlink()
            else:
                os.replace(stage / name, target)
        state = {
            "schema_version": MONTH_SCHEMA, "status": "month_closed", "passed": True,
            "month": prediction_seal["month"],
            "prediction_seal_digest": prediction_seal["result_digest"],
            "prediction_seal_sha256": _sha(root / "PREDICTIONS_SEALED.json"),
            "query_count": len(month_registry),
            "outcome_rows": len(outcomes), "path_rows": len(paths),
            "outcome_digest": _frame_digest(outcomes, ("query_id", "horizon_sessions")),
            "path_digest": _frame_digest(paths, ("query_id", "step")),
            "file_manifest": _month_file_manifest(root, ("query-outcomes.parquet", "query-paths.parquet")),
            "elapsed_seconds": perf_counter() - started,
            "predictions_sealed_before_query_outcome_access": True,
            "historical_query_outcomes_accessed": True,
            "final_period_result_opened": False,
        }
        smoke._atomic_json(closed, _seal(state, timing=True))
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _prior_query_outcomes(cache: Path, months: Sequence[str]) -> pd.DataFrame:
    frames = []
    for month in months:
        root = cache / f"month-{month}"
        if (root / "MONTH_CLOSED.json").exists():
            frame = pd.read_parquet(root / "query-outcomes.parquet", engine="pyarrow")
            frame = frame.loc[frame.horizon_sessions == 20, [
                "query_id", "completion_timestamp", "barrier_label", "query_regime",
            ]].rename(columns={"query_regime": "regime"})
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["query_id", "completion_timestamp", "barrier_label", "regime"]
    )


def _validate_existing_month(root: Path, month: str, final: bool) -> None:
    base_files = set(PREDICTION_FILES) | {"PREDICTIONS_SEALED.json"}
    observed = {path.name for path in root.iterdir()} if root.is_dir() else set()
    allowed = base_files if final else base_files | {
        "query-outcomes.parquet", "query-paths.parquet", "MONTH_CLOSED.json",
    }
    if root.is_symlink() or not root.is_dir() or not base_files.issubset(observed) \
            or not observed.issubset(allowed):
        raise WalkForwardPredictionError(f"month layout differs: {month}")
    prediction = base._read(root / "PREDICTIONS_SEALED.json")
    if not _valid_seal(prediction, timing=True) or prediction.get("month") != month \
            or prediction.get("file_manifest") != _month_file_manifest(root, PREDICTION_FILES):
        raise WalkForwardPredictionError(f"month prediction seal differs: {month}")
    if final and observed != base_files:
        raise WalkForwardPredictionError(f"final month contains query outcomes: {month}")
    if not final and "MONTH_CLOSED.json" in observed:
        if observed != allowed:
            raise WalkForwardPredictionError(f"closed month layout differs: {month}")
        closed = base._read(root / "MONTH_CLOSED.json")
        if not _valid_seal(closed, timing=True) \
                or closed.get("prediction_seal_digest") != prediction.get("result_digest") \
                or closed.get("predictions_sealed_before_query_outcome_access") is not True \
                or closed.get("file_manifest") != _month_file_manifest(
                    root, ("query-outcomes.parquet", "query-paths.parquet")
                ):
            raise WalkForwardPredictionError(f"month close seal differs: {month}")
    elif not final and not observed.issubset(
        base_files | {"query-outcomes.parquet", "query-paths.parquet"}
    ):
        raise WalkForwardPredictionError(f"partial month layout differs: {month}")


def _start_cache(repository: Path, prereg: Mapping[str, Any], h1: str) -> Path:
    root=repository/CACHE_RELATIVE
    state={
        "schema_version":STORE_SCHEMA,"status":"months_running",
        "preregistration_h1":h1,"preregistration_digest":prereg["preregistration_digest"],
        "month_order":prereg["month_order"],"historical_query_outcomes_accessed":False,
        "final_period_result_opened":False,
    }
    if not root.exists():
        root.mkdir(parents=True); smoke._atomic_json(root/"RUN_STARTED.json",{**state,"created_at":_now()})
    if root.is_symlink() or not root.is_dir(): raise WalkForwardPredictionError("prediction cache differs")
    started=base._read(root/"RUN_STARTED.json")
    if {k:v for k,v in started.items() if k!="created_at"}!=state:
        raise WalkForwardPredictionError("prediction cache start receipt differs")
    allowed={"RUN_STARTED.json"}|{f"month-{month}" for month in prereg["month_order"]}
    if any(path.name not in allowed or path.is_symlink() for path in root.iterdir()):
        raise WalkForwardPredictionError("prediction cache contains unexpected entries")
    return root


def _aggregate(repository: Path, cache: Path, prereg: Mapping[str, Any], h1: str, months: Sequence[str], elapsed: float) -> dict[str, Any]:
    final = repository / OUTPUT_RELATIVE
    if final.exists():
        names=("raw-predictions.parquet","baseline-predictions.parquet","continuous-predictions.parquet","path-predictions.parquet","nonfinal-query-outcomes.parquet","nonfinal-query-paths.parquet")
        if final.is_symlink() or not final.is_dir() \
                or {path.name for path in final.iterdir()}!=set(names)|{"SEALED.json"}:
            raise WalkForwardPredictionError("existing prediction output layout differs")
        seal=base._read(final/"SEALED.json")
        if not _valid_seal(seal,timing=True) or seal.get("passed") is not True \
                or seal.get("preregistration_h1")!=h1 \
                or seal.get("file_manifest")!=_month_file_manifest(final,names):
            raise WalkForwardPredictionError("existing prediction output seal differs")
        return seal
    temporary = Path(tempfile.mkdtemp(prefix=f".{final.name}.", dir=final.parent))
    try:
        primary=[]; baselines=[]; continuous=[]; paths=[]; query_outcomes=[]; query_paths=[]; receipts=[]
        for month in months:
            root=cache/f"month-{month}"; prediction=base._read(root/"PREDICTIONS_SEALED.json")
            receipts.append(prediction["result_digest"])
            primary.append(pd.read_parquet(root/PREDICTION_FILES[0])); baselines.append(pd.read_parquet(root/PREDICTION_FILES[1]))
            continuous.append(pd.read_parquet(root/PREDICTION_FILES[2])); paths.append(pd.read_parquet(root/PREDICTION_FILES[3]))
            if pd.Timestamp(f"{month}-01") < FINAL_START:
                query_outcomes.append(pd.read_parquet(root/"query-outcomes.parquet")); query_paths.append(pd.read_parquet(root/"query-paths.parquet"))
        frames={
            "raw-predictions.parquet":pd.concat(primary,ignore_index=True),
            "baseline-predictions.parquet":pd.concat(baselines,ignore_index=True),
            "continuous-predictions.parquet":pd.concat(continuous,ignore_index=True),
            "path-predictions.parquet":pd.concat(paths,ignore_index=True),
            "nonfinal-query-outcomes.parquet":pd.concat(query_outcomes,ignore_index=True),
            "nonfinal-query-paths.parquet":pd.concat(query_paths,ignore_index=True),
        }
        expected_rows = {
            "raw-predictions.parquet": prereg["expected_neighbor_prediction_rows"],
            "baseline-predictions.parquet": prereg["expected_baseline_prediction_rows"],
            "continuous-predictions.parquet": prereg["expected_continuous_prediction_rows"],
            "path-predictions.parquet": prereg["expected_path_prediction_rows"],
            "nonfinal-query-outcomes.parquet": EXPECTED_NONFINAL_QUERIES * len(outcome_store.HORIZONS),
        }
        if any(len(frames[name]) != count for name, count in expected_rows.items()):
            raise WalkForwardPredictionError("aggregate prediction row counts differ")
        for name,frame in frames.items(): smoke._atomic_parquet(temporary/name,frame)
        state={
            "schema_version":STORE_SCHEMA,"status":"sealed","passed":True,
            "preregistration_h1":h1,"preregistration_digest":prereg["preregistration_digest"],
            "month_count":len(months),"query_count":EXPECTED_QUERIES,
            "nonfinal_query_count":EXPECTED_NONFINAL_QUERIES,"final_query_count":EXPECTED_FINAL_QUERIES,
            "neighbor_prediction_rows":len(frames["raw-predictions.parquet"]),
            "baseline_prediction_rows":len(frames["baseline-predictions.parquet"]),
            "continuous_prediction_rows":len(frames["continuous-predictions.parquet"]),
            "path_prediction_rows":len(frames["path-predictions.parquet"]),
            "nonfinal_query_outcome_rows":len(frames["nonfinal-query-outcomes.parquet"]),
            "nonfinal_query_path_rows":len(frames["nonfinal-query-paths.parquet"]),
            "month_prediction_result_digests":receipts,
            "file_manifest":_month_file_manifest(temporary,tuple(frames)),
            "elapsed_seconds":elapsed,
            "historical_query_outcomes_accessed_month_by_month_after_prediction_seal":True,
            "final_period_result_opened":False,"predictions_affected_retrieval":False,
            "evaluation_metrics_opened":False,"production_promotion_authorized":False,
            "independent_verification_authorized":True,
        }
        seal=_seal(state,timing=True); smoke._atomic_json(temporary/"SEALED.json",seal)
        os.replace(temporary,final); return seal
    except BaseException:
        shutil.rmtree(temporary,ignore_errors=True); raise


def execute(repository: Path) -> dict[str, Any]:
    repository=repository.resolve(strict=True)
    prereg,registry,regimes,h1=validate_preregistration(repository)
    cache=_start_cache(repository,prereg,h1)
    links,outcomes,eligibility=_load_prediction_inputs(repository)
    primary_prepared=_prepare_primary(links,outcomes,eligibility)
    continuous_prepared=_prepare_continuous(links,outcomes,eligibility)
    path_index=_PathIndex(repository/outcome_store.OUTPUT_RELATIVE/"future-paths.parquet")
    fingerprints=_source_fingerprints(repository)
    months=prereg["month_order"]; started=perf_counter(); prior_distances=[]
    regime_map={str(row.month):_plain(row._asdict()) for row in regimes.itertuples(index=False)}
    for index,month in enumerate(months):
        month_registry=registry.loc[registry.month==month]
        prior_outcomes=_prior_query_outcomes(cache,months[:index])
        root=cache/f"month-{month}"
        if root.exists():
            _validate_existing_month(root,month,pd.Timestamp(f"{month}-01")>=FINAL_START)
        else:
            root=_build_month_predictions(
                repository,cache,prereg,h1,month,month_registry,regime_map[month],
                links,outcomes,eligibility,path_index,prior_outcomes,prior_distances,
                primary_prepared,continuous_prepared,
            )
            if pd.Timestamp(f"{month}-01")<FINAL_START:
                _close_month(repository,root,month_registry,regime_map[month]["regime"],fingerprints)
        if pd.Timestamp(f"{month}-01")<FINAL_START and not (root/"MONTH_CLOSED.json").exists():
            _close_month(repository,root,month_registry,regime_map[month]["regime"],fingerprints)
        if bool(month_registry.scored.iloc[0]):
            primary=pd.read_parquet(root/"raw-predictions.parquet",columns=["lane","prefix","nearest_composite_distance"])
            prior_distances.extend(primary.loc[(primary.lane=="composite")&(primary.prefix==20),"nearest_composite_distance"].tolist())
        print(f"[wf03d-predictions] month={month} complete={index+1}/{len(months)}",flush=True)
    return _aggregate(repository,cache,prereg,h1,months,perf_counter()-started)


def main(argv: Sequence[str] | None=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest="command",required=True)
    for command in ("preregister","run"):
        child=sub.add_parser(command); child.add_argument("--repository",required=True,type=Path)
    args=parser.parse_args(argv)
    if args.command=="preregister":
        value=build_preregistration(args.repository); smoke._atomic_json(args.repository/PREREGISTRATION_RELATIVE,value)
    else: value=execute(args.repository)
    print(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)); return 0


if __name__=="__main__":
    raise SystemExit(main())
