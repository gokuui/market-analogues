"""Decision Step 1+2: does chart similarity add forecast value, and could M2 detect it?

Exploratory kill test on already-consumed WF-04/M1 development evidence. It opens
no new untouched outcome and authorizes nothing; it only decides whether more time
on the predictive track is justified.

Part A  Forward-chained ablation of the M1 mixture family: the best mixture with
        the chart retrievers (composite, price_only) versus the best mixture
        without them, each selected only on earlier folds.
Part B  A plain causal feature model (gradient boosting and logistic regression)
        trained on universe-wide history with the same 20-session 2-ATR/1-ATR
        barrier label, refit monthly on labels completed before each month. The
        best forward-selected blend of it with the non-chart components is then
        compared with the same blend plus the chart retrievers.
Part C  Month-block bootstrap of paired Brier differences plus the number of
        prospective months needed to detect the development effect (Step 2).

Stop rules are fixed in STOP_RULES before any result is printed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


CLASSES = ("favorable_first", "adverse_first", "no_touch")
HELD_FOLDS = ("validation_1", "validation_2", "validation_3", "final_untouched")
ALL_FOLDS = ("development", *HELD_FOLDS)
MIXTURE_COMPONENTS = ("matched_causal_history", "composite", "price_only",
                      "recent_return_volatility")
CHART_COMPONENTS = ("composite", "price_only")
HORIZON = 20
ATR_LOOKBACK = 20
FAVORABLE_ATR = 2.0
ADVERSE_ATR = 1.0
HISTORY = 252
REGISTRY = Path("config/data/analogues/m04r14/t14-10-walk-forward-query-registry-v1/"
                "query-registry.parquet")
OUTPUT = Path("config/data/analogues/decision/step1-chart-value.json")

# Frozen before any real result is computed. Skill points are percentage points of
# Brier skill relative to the matched causal base rate on the same held rows.
STOP_RULES = {
    "minimum_chart_increment_skill_points": 0.5,
    "require_bootstrap_90pct_lower_bound_above_zero": True,
    "minimum_positive_held_folds": 3,
    "maximum_prospective_months_for_near_term_m2": 18,
    "power": 0.8,
    "one_sided_alpha": 0.05,
}

FEATURES = (
    "ret_5", "ret_20", "ret_63", "ret_126", "ret_252",
    "vol_20", "vol_63", "atr_pct_20",
    "dist_high_252", "dist_low_252", "dist_sma_50", "dist_sma_200",
    "volume_ratio_20_252", "log_dollar_volume_20", "range_contraction_10_63",
    "close_location_20", "up_volume_share_50",
    "bench_ret_20", "bench_ret_63", "bench_dist_sma_200", "bench_vol_20", "rs_63",
)


class DecisionError(RuntimeError):
    pass


# --------------------------------------------------------------------------- time


def session_key(values: Iterable[object]) -> pd.DatetimeIndex:
    """Calendar-date key that is identical for tz-aware and naive daily stamps."""
    stamps = pd.DatetimeIndex(pd.to_datetime(list(values)))
    if stamps.tz is not None:
        stamps = stamps.tz_localize(None)
    return stamps.normalize()


# ----------------------------------------------------------------- features/labels


def _log_ratio(series: pd.Series, periods: int) -> pd.Series:
    return np.log(series / series.shift(periods))


def benchmark_features(benchmark: pd.DataFrame) -> pd.DataFrame:
    close = benchmark["close"].astype(float)
    log_return = np.log(close / close.shift(1))
    frame = pd.DataFrame({
        "bench_ret_20": _log_ratio(close, 20),
        "bench_ret_63": _log_ratio(close, 63),
        "bench_dist_sma_200": close / close.rolling(200).mean() - 1.0,
        "bench_vol_20": log_return.rolling(20).std(),
    })
    frame.index = session_key(benchmark["timestamp"])
    return frame


def stock_features(stock: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame:
    """Features at every session using only bars at or before that session."""
    open_, high, low, close, volume = (stock[c].astype(float) for c in
                                       ("open", "high", "low", "close", "volume"))
    previous = close.shift(1)
    log_return = np.log(close / previous)
    true_range = pd.concat([(high - low), (high - previous).abs(),
                            (low - previous).abs()], axis=1).max(axis=1)
    spread = (high - low).replace(0.0, np.nan)
    up_volume = volume.where(close > previous, 0.0)
    frame = pd.DataFrame({
        "ret_5": _log_ratio(close, 5), "ret_20": _log_ratio(close, 20),
        "ret_63": _log_ratio(close, 63), "ret_126": _log_ratio(close, 126),
        "ret_252": _log_ratio(close, 252),
        "vol_20": log_return.rolling(20).std(), "vol_63": log_return.rolling(63).std(),
        "atr_pct_20": true_range.rolling(ATR_LOOKBACK).mean() / close,
        "dist_high_252": close / high.rolling(HISTORY).max() - 1.0,
        "dist_low_252": close / low.rolling(HISTORY).min() - 1.0,
        "dist_sma_50": close / close.rolling(50).mean() - 1.0,
        "dist_sma_200": close / close.rolling(200).mean() - 1.0,
        "volume_ratio_20_252": np.log1p(volume.rolling(20).mean())
        - np.log1p(volume.rolling(HISTORY).median()),
        "log_dollar_volume_20": np.log1p((close * volume).rolling(20).median()),
        "range_contraction_10_63": (spread / close).rolling(10).mean()
        / (spread / close).rolling(63).mean(),
        "close_location_20": ((close - low) / spread).rolling(20).mean(),
        "up_volume_share_50": up_volume.rolling(50).sum()
        / volume.rolling(50).sum().replace(0.0, np.nan),
    })
    del open_
    frame.index = session_key(stock["timestamp"])
    frame = frame.join(bench, how="left")
    frame["rs_63"] = frame["ret_63"] - frame["bench_ret_63"]
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame.iloc[:HISTORY] = np.nan
    return frame.loc[:, list(FEATURES)]


def barrier_labels(stock: pd.DataFrame, bench_sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Vectorized T14-09 20-session barrier label and completion session.

    Matches causal_outcomes: ATR is the mean true range over the 20 sessions ending
    at the origin; +2 ATR before -1 ATR is favorable, the reverse adverse, a
    same-bar double touch ambiguous (dropped), neither no_touch. The 20 future
    stock sessions must be exactly the next 20 benchmark sessions.
    """
    keys = session_key(stock["timestamp"])
    high = stock["high"].to_numpy(float)
    low = stock["low"].to_numpy(float)
    close = stock["close"].to_numpy(float)
    n = len(close)
    empty = pd.DataFrame(columns=["label", "completion"], index=keys[:0])
    if n <= max(ATR_LOOKBACK, HORIZON) + 1:
        return empty
    previous = np.r_[np.nan, close[:-1]]
    tr = np.nanmax(np.c_[high - low, np.abs(high - previous), np.abs(low - previous)], axis=1)
    tr[0] = np.nan
    atr = pd.Series(tr).rolling(ATR_LOOKBACK).mean().to_numpy()
    origins = np.arange(n - HORIZON)
    future_high = np.lib.stride_tricks.sliding_window_view(high[1:], HORIZON)[:len(origins)]
    future_low = np.lib.stride_tricks.sliding_window_view(low[1:], HORIZON)[:len(origins)]
    favorable = close[origins] + FAVORABLE_ATR * atr[origins]
    adverse = close[origins] - ADVERSE_ATR * atr[origins]
    up = future_high >= favorable[:, None]
    down = future_low <= adverse[:, None]
    never = HORIZON + 1
    first_up = np.where(up.any(axis=1), up.argmax(axis=1), never)
    first_down = np.where(down.any(axis=1), down.argmax(axis=1), never)
    label = np.full(len(origins), "no_touch", dtype=object)
    label[first_up < first_down] = "favorable_first"
    label[first_down < first_up] = "adverse_first"
    label[(first_up == first_down) & (first_up < never)] = "ambiguous"
    position = bench_sessions.get_indexer(keys)
    mapped = position >= 0
    continuous = (mapped[origins] & mapped[origins + HORIZON]
                  & (position[origins + HORIZON] - position[origins] == HORIZON))
    valid = (np.isfinite(atr[origins]) & (atr[origins] > 0) & continuous
             & (label != "ambiguous"))
    return pd.DataFrame({"label": label[valid],
                         "completion": keys[origins + HORIZON][valid]},
                        index=keys[origins][valid])


@dataclass(frozen=True)
class SymbolJob:
    dataset_config: str
    dataset: str
    symbol: str
    stride: int
    query_cutoffs: tuple[str, ...]


def _source(config_path: str, dataset: str):
    from market_analogues.adapters import source_from_spec
    from market_analogues.config import load_config
    return source_from_spec(load_config(config_path).datasets[dataset])


_WORKER_CACHE: dict[tuple[str, str], tuple[object, pd.DataFrame, pd.DatetimeIndex]] = {}


def _worker_context(config_path: str, dataset: str):
    key = (config_path, dataset)
    if key not in _WORKER_CACHE:
        source = _source(config_path, dataset)
        benchmark = source.load_benchmark()
        if benchmark is None:
            raise DecisionError("benchmark is required for the feature model")
        _WORKER_CACHE[key] = (source, benchmark_features(benchmark),
                              session_key(benchmark["timestamp"]))
    return _WORKER_CACHE[key]


def symbol_rows(job: SymbolJob) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Universe training rows and exact query-cutoff feature rows for one symbol."""
    from market_analogues.types import InstrumentKey
    source, bench, sessions = _worker_context(job.dataset_config, job.dataset)
    stock = source.load(InstrumentKey(job.dataset, job.symbol))
    stock = stock.dropna(subset=["open", "high", "low", "close", "volume"])
    stock = stock.loc[(stock[["open", "high", "low", "close"]] > 0).all(axis=1)]
    stock = stock.drop_duplicates("timestamp", keep=False).reset_index(drop=True)
    features = stock_features(stock, bench)
    labels = barrier_labels(stock, sessions)
    phase = sum(map(ord, job.symbol)) % job.stride
    sampled = labels.iloc[phase::job.stride]
    train = features.join(sampled, how="inner").dropna(subset=list(FEATURES))
    train.insert(0, "symbol", job.symbol)
    train.index.name = "cutoff"
    queries = features.reindex(session_key(job.query_cutoffs))
    queries.insert(0, "symbol", job.symbol)
    queries.index.name = "cutoff"
    return train.reset_index(), queries.reset_index()


def build_feature_tables(config_path: str, dataset: str, symbols: Sequence[str],
                         query_cutoffs: Mapping[str, Sequence[str]], stride: int,
                         workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    jobs = [SymbolJob(config_path, dataset, s, stride, tuple(query_cutoffs.get(s, ())))
            for s in symbols]
    trains, queries = [], []
    if workers <= 1:
        results = map(symbol_rows, jobs)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        results = executor.map(symbol_rows, jobs, chunksize=16)
    for done, (train, query) in enumerate(results, start=1):
        trains.append(train)
        if len(query):
            queries.append(query)
        if done % 500 == 0:
            print(f"features: {done}/{len(jobs)} symbols", file=sys.stderr)
    if workers > 1:
        executor.shutdown()
    return (pd.concat(trains, ignore_index=True),
            pd.concat(queries, ignore_index=True) if queries else pd.DataFrame())


# --------------------------------------------------------------------- scoring


def one_hot(labels: Sequence[str]) -> np.ndarray:
    index = {name: i for i, name in enumerate(CLASSES)}
    truth = np.zeros((len(labels), 3))
    for row, label in enumerate(labels):
        truth[row, index[label]] = 1.0
    return truth


def brier(probabilities: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.square(probabilities - truth).sum(axis=1)


def simplex(names: Sequence[str], step: int = 10) -> list[dict[str, float]]:
    """All non-negative tenths-grid weights over ``names`` summing to one."""
    def rec(remaining: int, depth: int) -> Iterable[tuple[int, ...]]:
        if depth == 1:
            yield (remaining,)
            return
        for value in range(remaining + 1):
            for rest in rec(remaining - value, depth - 1):
                yield (value, *rest)
    return [{name: value / step for name, value in zip(names, combo)}
            for combo in rec(step, len(names))]


def mix(weights: Mapping[str, float], matrices: Mapping[str, np.ndarray]) -> np.ndarray:
    return sum(weight * matrices[name] for name, weight in weights.items())


def forward_select(grid: Sequence[Mapping[str, float]], matrices: Mapping[str, np.ndarray],
                   truth: np.ndarray, folds: np.ndarray, eligible: np.ndarray,
                   held: str, training_folds: Sequence[str]) -> tuple[dict[str, float], np.ndarray]:
    """Choose weights on earlier folds only; ties choose the earlier grid entry."""
    training = eligible & np.isin(folds, list(training_folds))
    if not training.any():
        raise DecisionError(f"no training rows before {held}")
    losses = [brier(mix(w, matrices), truth)[training].mean() for w in grid]
    best = dict(grid[int(np.argmin(losses))])
    return best, mix(best, matrices)


def month_block_bootstrap(differences: np.ndarray, months: np.ndarray,
                          replicates: int = 4000, seed: int = 20260926) -> dict[str, float]:
    """Mean paired difference with calendar-month block resampling."""
    labels, inverse = np.unique(months, return_inverse=True)
    sums = np.bincount(inverse, weights=differences, minlength=len(labels))
    counts = np.bincount(inverse, minlength=len(labels)).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(labels), size=(replicates, len(labels)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {
        "mean": float(differences.mean()),
        "lower_90": float(np.quantile(boot, 0.05)),
        "upper_90": float(np.quantile(boot, 0.95)),
        "months": int(len(labels)),
    }


def months_needed(differences: np.ndarray, months: np.ndarray, effect: float,
                  power: float = STOP_RULES["power"],
                  alpha: float = STOP_RULES["one_sided_alpha"]) -> dict[str, float]:
    """Prospective months needed to detect ``effect`` with month-level dependence."""
    from scipy.stats import norm
    frame = pd.DataFrame({"d": differences, "m": months})
    monthly = frame.groupby("m")["d"].mean()
    queries_per_month = float(frame.groupby("m").size().mean())
    sd = float(monthly.std(ddof=1))
    if not math.isfinite(sd) or effect <= 0:
        needed = math.inf
    else:
        needed = math.ceil(((norm.ppf(1 - alpha) + norm.ppf(power)) * sd / effect) ** 2)
    return {"monthly_sd": sd, "effect": float(effect),
            "queries_per_month_in_development": queries_per_month,
            "months_needed": needed}


# ----------------------------------------------------------------- feature model


def fit_predict_monthly(train: pd.DataFrame, queries: pd.DataFrame, kind: str,
                        max_rows: int, seed: int = 7) -> np.ndarray:
    """Refit at each query month on labels completed before that month's first query."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    output = np.full((len(queries), 3), np.nan)
    months = queries["cutoff"].dt.to_period("M")
    rng = np.random.default_rng(seed)
    for month in sorted(months.unique()):
        rows = np.flatnonzero((months == month).to_numpy())
        start = queries["cutoff"].iloc[rows].min()
        available = train.loc[train["completion"] < start]
        if len(available) > max_rows:
            available = available.iloc[np.sort(rng.choice(len(available), max_rows, replace=False))]
        if available["label"].nunique() < 3:
            continue
        x = available.loc[:, list(FEATURES)].to_numpy(float)
        y = available["label"].to_numpy()
        if kind == "gbm":
            model = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200,
                l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                random_state=seed)
            x_query = queries.loc[:, list(FEATURES)].to_numpy(float)[rows]
        elif kind == "logistic":
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
            medians = np.nanmedian(x, axis=0)
            x = np.where(np.isnan(x), medians, x)
            x_query = queries.loc[:, list(FEATURES)].to_numpy(float)[rows]
            x_query = np.where(np.isnan(x_query), medians, x_query)
        else:
            raise DecisionError(f"unknown model kind: {kind}")
        model.fit(x, y)
        order = [list(model.classes_).index(name) for name in CLASSES]
        output[rows] = model.predict_proba(x_query)[:, order]
    output = np.clip(output, 1e-6, None)
    return output / output.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------- evaluation


def evaluate(queries: pd.DataFrame, matrices: Mapping[str, np.ndarray],
             truth: np.ndarray) -> dict[str, object]:
    """Forward-chained held-fold comparison of every method against the base rate."""
    folds = queries["fold_id"].astype(str).to_numpy()
    eligible = (queries["multiclass_evaluable"].to_numpy(bool)
                & queries["purged_evaluation_included"].to_numpy(bool))
    months = pd.to_datetime(queries["query_cutoff"]).dt.strftime("%Y-%m").to_numpy()
    families = {
        "with_chart": simplex(MIXTURE_COMPONENTS),
        "no_chart": [w for w in simplex(MIXTURE_COMPONENTS)
                     if all(w[name] == 0 for name in CHART_COMPONENTS)],
    }
    has_features = "gbm" in matrices
    if has_features:
        # Both sides may recalibrate the universe-trained model with the base rate,
        # so the chart increment cannot be earned by calibration alone.
        non_chart = ("gbm", "matched_causal_history", "recent_return_volatility")
        families["features_no_chart"] = simplex(non_chart)
        families["features_with_chart"] = simplex((*non_chart, *CHART_COMPONENTS))
    held_predictions: dict[str, np.ndarray] = {name: np.full_like(truth, np.nan)
                                               for name in families}
    selections: dict[str, dict[str, dict[str, float]]] = {name: {} for name in families}
    for position, held in enumerate(HELD_FOLDS):
        mask = folds == held
        earlier = ALL_FOLDS[:position + 1]
        for name, grid in families.items():
            usable = eligible.copy()
            if name.startswith("features_"):
                # Monthly-refit GBM forecasts are causal for every fold, so earlier
                # folds are genuine out-of-sample selection data.
                usable &= np.isfinite(matrices["gbm"]).all(axis=1)
            weights, prediction = forward_select(grid, matrices, truth, folds, usable,
                                                 held, earlier)
            selections[name][held] = weights
            held_predictions[name][mask] = prediction[mask]
    methods = {
        "matched_base_rate": matrices["matched_causal_history"],
        "composite_only": matrices["composite"],
        "no_chart_mixture": held_predictions["no_chart"],
        "with_chart_mixture": held_predictions["with_chart"],
        "m1_frozen_weights_in_sample": matrices["m1_candidate"],
    }
    if has_features:
        methods.update({"gbm": matrices["gbm"], "logistic": matrices["logistic"],
                        "features_no_chart": held_predictions["features_no_chart"],
                        "features_with_chart": held_predictions["features_with_chart"]})
    held = eligible & np.isin(folds, HELD_FOLDS)
    for name in methods:
        held &= np.isfinite(methods[name]).all(axis=1)
    base = brier(methods["matched_base_rate"], truth)
    table = {}
    for name, probabilities in methods.items():
        loss = brier(probabilities, truth)
        table[name] = {
            "pooled_skill_points": float(100 * (1 - loss[held].mean() / base[held].mean())),
            "per_fold_skill_points": {
                fold: float(100 * (1 - loss[held & (folds == fold)].mean()
                                   / base[held & (folds == fold)].mean()))
                for fold in HELD_FOLDS if (held & (folds == fold)).any()
            },
        }

    def paired(better: str, worse: str) -> dict[str, object]:
        diff = (brier(methods[worse], truth) - brier(methods[better], truth))[held]
        scale = 100 / base[held].mean()
        stats = month_block_bootstrap(diff * scale, months[held])
        per_fold = {fold: float(diff[folds[held] == fold].mean() * scale)
                    for fold in HELD_FOLDS if (folds[held] == fold).any()}
        return {**stats, "unit": "skill_points", "per_fold": per_fold,
                "positive_folds": int(sum(v > 0 for v in per_fold.values()))}

    comparisons = {"chart_increment_over_mixture": paired("with_chart_mixture", "no_chart_mixture")}
    if has_features:
        comparisons["chart_increment_over_feature_model"] = paired(
            "features_with_chart", "features_no_chart")
        comparisons["current_mixture_vs_features_no_chart"] = paired(
            "with_chart_mixture", "features_no_chart")
    effect_diff = (base - brier(methods["with_chart_mixture"], truth))[held]
    power = months_needed(effect_diff * 100 / base[held].mean(), months[held],
                          effect=float(effect_diff.mean() * 100 / base[held].mean()))
    return {"held_rows": int(held.sum()), "methods": table, "comparisons": comparisons,
            "forward_selected_weights": selections, "prospective_power": power}


def decide(result: Mapping[str, object]) -> dict[str, object]:
    rules = STOP_RULES
    verdicts = {}
    for name, comparison in result["comparisons"].items():
        if not name.startswith("chart_increment"):
            continue
        verdicts[name] = bool(
            comparison["mean"] >= rules["minimum_chart_increment_skill_points"]
            and (comparison["lower_90"] > 0
                 or not rules["require_bootstrap_90pct_lower_bound_above_zero"])
            and comparison["positive_folds"] >= rules["minimum_positive_held_folds"])
    months = result["prospective_power"]["months_needed"]
    chart_adds_value = bool(verdicts) and all(verdicts.values())
    return {
        "chart_increment_passes": verdicts,
        "chart_adds_value": chart_adds_value,
        "m2_near_term_feasible": bool(months <= rules["maximum_prospective_months_for_near_term_m2"]),
        "recommendation": (
            "continue_predictive_track_event_conditioned" if chart_adds_value
            and months <= rules["maximum_prospective_months_for_near_term_m2"]
            else "chart_signal_real_but_slow_to_confirm" if chart_adds_value
            else "stop_predictive_track_keep_descriptive_tool_if_used"),
    }


# ----------------------------------------------------------------------- driver


def load_development_matrices(repository: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    sys.path.insert(0, str(repository))
    from experiments.m04r import m04r15_m1_candidate_development_gate as gate
    content, _ = gate.validate_inputs(repository)
    queries, matrices, truth = gate.load_development(content)
    matrices = dict(matrices)
    matrices["m1_candidate"] = matrices.pop("candidate")
    return queries, matrices, truth


def attach_feature_models(repository: Path, config: Path, queries: pd.DataFrame,
                          matrices: dict[str, np.ndarray], stride: int, workers: int,
                          max_rows: int, symbol_limit: int | None) -> dict[str, object]:
    registry = pd.read_parquet(repository / REGISTRY, columns=["episode_id", "symbol", "cutoff"])
    registry = registry.rename(columns={"episode_id": "query_id"})
    registry["query_id"] = registry["query_id"].astype(str)
    joined = queries[["query_id"]].merge(registry, on="query_id", how="left", validate="one_to_one")
    if joined["symbol"].isna().any():
        raise DecisionError("registry lacks symbols for some development queries")
    joined["key"] = session_key(joined["cutoff"])
    source = _source(str(config), "nasdaq")
    symbols = sorted(k.source_symbol for k in source.instruments())
    if symbol_limit:
        wanted = set(joined["symbol"])
        symbols = sorted(wanted | set(symbols[:symbol_limit]))
    cutoffs = joined.groupby("symbol")["cutoff"].apply(lambda s: [str(v) for v in s]).to_dict()
    train, query_features = build_feature_tables(str(config), "nasdaq", symbols, cutoffs,
                                                 stride, workers)
    query_features = query_features.drop_duplicates(["symbol", "cutoff"])
    aligned = joined.merge(query_features, left_on=["symbol", "key"],
                           right_on=["symbol", "cutoff"], how="left", suffixes=("", "_f"))
    aligned["cutoff"] = pd.to_datetime(queries["query_cutoff"].to_numpy())
    if aligned["cutoff"].dt.tz is not None:
        aligned["cutoff"] = aligned["cutoff"].dt.tz_localize(None)
    aligned["cutoff"] = aligned["cutoff"].dt.normalize()
    train["completion"] = pd.to_datetime(train["completion"])
    for kind in ("gbm", "logistic"):
        print(f"fitting {kind} monthly", file=sys.stderr)
        matrices[kind] = fit_predict_monthly(train, aligned, kind, max_rows)
    missing = int(aligned[list(FEATURES)].isna().all(axis=1).sum())
    return {"training_rows": int(len(train)), "symbols": len(symbols),
            "queries_without_features": missing, "stride": stride,
            "max_rows_per_fit": max_rows,
            "training_label_share": train["label"].value_counts(normalize=True).round(4).to_dict()}


def print_summary(result: Mapping[str, object]) -> None:
    print("\nHeld-fold Brier skill vs matched base rate (percentage points)")
    for name, row in result["evaluation"]["methods"].items():
        folds = "  ".join(f"{v:+.2f}" for v in row["per_fold_skill_points"].values())
        print(f"  {name:32s} pooled {row['pooled_skill_points']:+.2f}   folds {folds}")
    print("\nPaired increments (skill points, month-block 90% interval)")
    for name, row in result["evaluation"]["comparisons"].items():
        print(f"  {name:38s} {row['mean']:+.2f} [{row['lower_90']:+.2f}, {row['upper_90']:+.2f}]"
              f"  positive folds {row['positive_folds']}/{len(row['per_fold'])}")
    power = result["evaluation"]["prospective_power"]
    print(f"\nStep 2: months of new data to detect {power['effect']:+.2f} points at 80% power:"
          f" {power['months_needed']}")
    print(f"\nDecision: {json.dumps(result['decision'], indent=2)}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("config/datasets.example.yaml"))
    parser.add_argument("--skip-feature-model", action="store_true")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-rows-per-fit", type=int, default=400_000)
    parser.add_argument("--symbol-limit", type=int, default=None,
                        help="smoke test: universe = query symbols plus the first N symbols")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    repository = args.repository.resolve()
    queries, matrices, truth = load_development_matrices(repository)
    feature_info = None
    if not args.skip_feature_model:
        feature_info = attach_feature_models(repository, args.config, queries, matrices,
                                             args.stride, args.workers,
                                             args.max_rows_per_fit, args.symbol_limit)
    evaluation = evaluate(queries, matrices, truth)
    result = {"schema": "decision-step1-chart-value-v1", "stop_rules": STOP_RULES,
              "feature_model": feature_info, "evaluation": evaluation,
              "claim_boundary": "exploratory kill test on consumed development evidence only"}
    result["decision"] = decide(evaluation)
    output = repository / (args.output or OUTPUT)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, default=float) + "\n")
    print_summary(result)
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
