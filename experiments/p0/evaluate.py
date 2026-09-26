"""P0 walk-forward forecaster and leaderboard metrics on the label store.

Model: one gradient-boosted binary head per cumulative threshold P(Y >= k),
k = 1..5, refit every test year on rows whose 20-session outcome completed before
that year began. Heads are repaired to be monotone and turned into class
probabilities; expected R is the class-probability-weighted mean realized R of
each class in the training rows.

Vault: origins on or after VAULT_START are never read here. NASDAQ origins from
2024-01 to 2025-08 were opened by WF-04 and are reported as a separate
"previously_opened" segment.

Usage:
    PYTHONPATH=.:src .venv/bin/python -m experiments.p0.evaluate --dataset nasdaq \
        --name baseline
    # extra feature columns from a parquet keyed by (symbol, date):
    ... --name e1 --extra config/data/analogues/p0/nasdaq/e1.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from experiments.decision.chart_value import FEATURES

VAULT_START = pd.Timestamp("2025-09-01")
PREVIOUSLY_OPENED = {"nasdaq": (pd.Timestamp("2024-01-01"), pd.Timestamp("2025-08-31"))}
FIRST_TEST_YEAR = {"nasdaq": 2006, "nse": 2006}
K = 5  # cumulative heads for classes 1..5
LIQUID = {"nasdaq": {"close": 5.0, "dollar_vol_20": 2e6},
          "nse": {"close": 20.0, "dollar_vol_20": 2e7}}
ROUND_TRIP_COST = {"nasdaq": 0.003, "nse": 0.005}
TOP_K = 10
MIN_TRAIN_ROWS = 50_000
NW_LAG = 5
ROOT = Path("config/data/analogues/p0")


# ------------------------------------------------------------------ data


def load_store(dataset: str, extra: Path | None = None) -> tuple[pd.DataFrame, list[str]]:
    store = pd.read_parquet(ROOT / dataset / "store.parquet")
    store = store.loc[store["date"] < VAULT_START].reset_index(drop=True)
    features = list(FEATURES)
    if extra is not None:
        more = pd.read_parquet(extra)
        cols = [c for c in more.columns if c not in ("symbol", "date")]
        store = store.merge(more, on=["symbol", "date"], how="left")
        features += cols
    liquid = LIQUID[dataset]
    store["liquid"] = ((store["close"] >= liquid["close"])
                       & (store["dollar_vol_20"] >= liquid["dollar_vol_20"]))
    return store, features


# ------------------------------------------------------------------ model


def cumulative_to_classes(q: np.ndarray) -> np.ndarray:
    q = np.minimum.accumulate(np.clip(q, 1e-6, 1 - 1e-6), axis=1)
    edges = np.c_[np.ones(len(q)), q, np.zeros(len(q))]
    return edges[:, :-1] - edges[:, 1:]


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str],
                label: str, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    x = train[features].to_numpy(np.float32)
    y = train[label].to_numpy()
    xt = test[features].to_numpy(np.float32)
    q = np.zeros((len(test), K))
    for k in range(1, K + 1):
        model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
            random_state=seed)
        model.fit(x, (y >= k).astype(np.int8))
        q[:, k - 1] = model.predict_proba(xt)[:, 1]
    probs = cumulative_to_classes(q)
    real = train[label.replace("y20", "realized_r20")].to_numpy(float)
    class_r = np.array([np.nanmean(real[y == c]) if (y == c).any() else 0.0
                        for c in range(K + 1)])
    return probs, probs @ class_r


def walk_forward(store: pd.DataFrame, features: list[str], dataset: str, label: str,
                 max_rows: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labelled = store.loc[store[label] >= 0]
    out = []
    for year in range(FIRST_TEST_YEAR[dataset], VAULT_START.year + 1):
        start = pd.Timestamp(year=year, month=1, day=1)
        stop = min(pd.Timestamp(year=year + 1, month=1, day=1), VAULT_START)
        train = labelled.loc[labelled["completion"] < start]
        scored = store.loc[(store["y20"] >= 0) | (store["y20_pen"] >= 0)]
        test = scored.loc[(scored["date"] >= start) & (scored["date"] < stop)]
        if test.empty or len(train) < MIN_TRAIN_ROWS:
            continue
        if len(train) > max_rows:
            train = train.iloc[np.sort(rng.choice(len(train), max_rows, replace=False))]
        began = time.time()
        probs, er = fit_predict(train, test, features, label, seed)
        base = np.bincount(train[label].to_numpy(), minlength=K + 1) / len(train)
        frame = test[["symbol", "date"]].copy()
        for c in range(K + 1):
            frame[f"p{c}"] = probs[:, c].astype(np.float32)
            frame[f"clim{c}"] = np.float32(base[c])
        frame["er"] = er.astype(np.float32)
        out.append(frame)
        print(f"{year}: train {len(train):,} test {len(test):,} "
              f"{time.time() - began:.0f}s", file=sys.stderr)
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------ metrics


def rps(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    cum_pred = np.cumsum(probs, axis=1)[:, :-1]
    cum_obs = (np.arange(K)[None, :] >= y[:, None]).astype(float)
    return ((cum_pred - cum_obs) ** 2).sum(axis=1) / K


def newey_west_t(series: np.ndarray, lag: int = NW_LAG) -> float:
    x = np.asarray(series, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return float("nan")
    d = x - x.mean()
    var = d @ d / n
    for j in range(1, lag + 1):
        var += 2 * (1 - j / (lag + 1)) * (d[j:] @ d[:-j]) / n
    return float(x.mean() / np.sqrt(var / n)) if var > 0 else float("nan")


def daily_metrics(frame: pd.DataFrame, label: str, real_col: str, dataset: str) -> pd.DataFrame:
    probs = frame[[f"p{c}" for c in range(K + 1)]].to_numpy(float)
    clim = frame[[f"clim{c}" for c in range(K + 1)]].to_numpy(float)
    y = frame[label].to_numpy()
    frame = frame.assign(rps_model=rps(probs, y), rps_clim=rps(clim, y))
    cost_r = ROUND_TRIP_COST[dataset] / frame["r_pct"].astype(float)
    frame["net_r"] = frame[real_col].astype(float) - cost_r
    rows = []
    for date, day in frame.groupby("date", sort=True):
        row = {"date": date, "n": len(day),
               "rps_model": day["rps_model"].mean(), "rps_clim": day["rps_clim"].mean()}
        # same-date climatology: the day's realized class frequencies (reference only)
        freq = np.bincount(day[label].to_numpy(), minlength=K + 1) / len(day)
        row["rps_date_clim"] = rps(np.repeat(freq[None, :], len(day), 0),
                                   day[label].to_numpy()).mean()
        for target in (real_col, "exret5", "exret20"):
            ok = day[target].notna()
            row[f"ic_{target}"] = (day.loc[ok, "er"].rank().corr(day.loc[ok, target].rank())
                                   if ok.sum() > 20 else np.nan)
        liquid = day.loc[day["liquid"]]
        if len(liquid) >= 5 * TOP_K:
            top = liquid.nlargest(TOP_K, "er")
            row["top_net_r"] = top["net_r"].mean()
            row["all_liquid_net_r"] = liquid["net_r"].mean()
            row["top_minus_all"] = row["top_net_r"] - row["all_liquid_net_r"]
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(daily: pd.DataFrame) -> dict[str, float]:
    def skill(ref: str) -> float:
        return float(100 * (1 - daily["rps_model"].sum() / daily[ref].sum()))
    out = {"dates": int(len(daily)), "rows": int(daily["n"].sum()),
           "rps_skill_vs_train_clim_pct": skill("rps_clim"),
           "rps_skill_vs_same_date_clim_pct": skill("rps_date_clim"),
           "rps_skill_daily_t": newey_west_t(daily["rps_clim"] - daily["rps_model"])}
    for col in [c for c in daily.columns if c.startswith("ic_")]:
        out[f"{col}_mean"] = float(daily[col].mean())
        out[f"{col}_t"] = newey_west_t(daily[col].to_numpy())
    for col in ("top_net_r", "all_liquid_net_r", "top_minus_all"):
        if col in daily:
            out[f"{col}_mean"] = float(daily[col].mean())
            out[f"{col}_t"] = newey_west_t(daily[col].to_numpy())
    return out


def report(pred: pd.DataFrame, store: pd.DataFrame, dataset: str) -> dict[str, object]:
    cols = ["symbol", "date", "y20", "y20_pen", "realized_r20", "realized_r20_pen",
            "exret5", "exret20", "r_pct", "liquid"]
    frame = pred.merge(store[cols], on=["symbol", "date"], how="left")
    result: dict[str, object] = {}
    variants = {"censored": ("y20", "realized_r20"), "penalized": ("y20_pen", "realized_r20_pen")}
    for variant, (label, real_col) in variants.items():
        sub = frame.loc[frame[label] >= 0]
        daily = daily_metrics(sub, label, real_col, dataset)
        daily["year"] = pd.to_datetime(daily["date"]).dt.year
        block = {"all": summarize(daily)}
        if dataset in PREVIOUSLY_OPENED:
            lo, hi = PREVIOUSLY_OPENED[dataset]
            inside = daily["date"].between(lo, hi)
            block["before_previously_opened"] = summarize(daily.loc[daily["date"] < lo])
            block["previously_opened"] = summarize(daily.loc[inside])
        block["by_year"] = {int(y): {k: round(v, 4) for k, v in summarize(d).items()
                                     if k in ("rps_skill_vs_train_clim_pct",
                                              "ic_" + real_col + "_mean",
                                              "top_minus_all_mean")}
                            for y, d in daily.groupby("year")}
        result[variant] = block
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(LIQUID), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--extra", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=1_000_000)
    parser.add_argument("--label", default="y20", choices=("y20", "y20_pen"))
    args = parser.parse_args(argv)
    store, features = load_store(args.dataset, args.extra)
    pred = walk_forward(store, features, args.dataset, args.label, args.max_rows)
    out_dir = ROOT / args.dataset / "runs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(out_dir / "predictions.parquet", index=False)
    result = {"name": args.name, "dataset": args.dataset, "features": features,
              "train_label": args.label, "vault_start": str(VAULT_START.date()),
              **report(pred, store, args.dataset)}
    (out_dir / "report.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(json.dumps({v: result[v]["all"] for v in ("censored", "penalized")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
