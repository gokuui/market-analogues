"""Causal daily model scores for every NSE stock-day, for portfolio-level tests.

For each calendar year Y from START_YEAR on, the baseline and E1+E2 models are
fit on label-store rows whose 20-session outcome completed before Y began, then
score every stock-day in Y. The within-date percentile of each score across all
scored stocks is added, so an external filter can compare a signal to that day's
cross-section without looking ahead.

Output: config/data/analogues/p0/nse/daily_scores.parquet
    symbol, date, er_baseline, er_e1e2, pct_baseline, pct_e1e2
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from experiments.decision.chart_value import FEATURES, benchmark_features, session_key
from experiments.p0.entries_gonogo import feature_columns
from experiments.p0.evaluate import ROOT, clean_store, fit_predict
from experiments.p0.features import REQUIRED, p0_stock_features
from experiments.p1.build_extra import load_clean
from experiments.p1.features import E1, E2, e1_features, e2_features

START_YEAR = 2015
END = pd.Timestamp("2026-02-11")
_CTX: dict[str, object] = {}


def _context(config: str):
    if not _CTX:
        from market_analogues.adapters import source_from_spec
        from market_analogues.config import load_config
        source = source_from_spec(load_config(config).datasets["nse"])
        bench = source.load_benchmark().dropna(subset=["close"])
        keys = session_key(bench["timestamp"])
        _CTX.update(source=source, bench_features=benchmark_features(bench),
                    bench_close=pd.Series(bench["close"].to_numpy(float), index=keys))
    return _CTX


def symbol_frame(args) -> pd.DataFrame | None:
    config, symbol = args
    ctx = _context(config)
    stock = load_clean(ctx["source"], "nse", symbol)
    if stock is None or len(stock) < 300:
        return None
    keys = session_key(stock["timestamp"])
    rows = np.flatnonzero((keys >= pd.Timestamp(START_YEAR - 1, 12, 1)) & (keys <= END))
    rows = rows[rows >= 252]
    if not len(rows):
        return None
    base = p0_stock_features(stock, ctx["bench_features"]).iloc[rows].reset_index(drop=True)
    e1 = e1_features(stock, ctx["bench_close"].reindex(keys).to_numpy()).iloc[rows]
    e2 = e2_features(stock, rows)
    frame = pd.concat([base, e1.reset_index(drop=True), e2.reset_index(drop=True)], axis=1)
    frame.insert(0, "date", keys[rows])
    frame.insert(0, "symbol", symbol)
    frame = frame.loc[frame[list(REQUIRED)].notna().all(axis=1)]
    floats = frame.select_dtypes("float64").columns
    frame[floats] = frame[floats].astype("float32")
    return frame


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/datasets.example.yaml")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--max-rows", type=int, default=1_000_000)
    args = parser.parse_args(argv)
    config = str(Path(args.config).resolve())
    from market_analogues.adapters import source_from_spec
    from market_analogues.config import load_config
    symbols = sorted(k.source_symbol for k in
                     source_from_spec(load_config(config).datasets["nse"]).instruments())
    started = time.time()
    frames = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, frame in enumerate(pool.map(symbol_frame, [(config, s) for s in symbols],
                                              chunksize=8), 1):
            if frame is not None:
                frames.append(frame)
            if done % 500 == 0:
                print(f"features {done}/{len(symbols)} {time.time() - started:.0f}s", file=sys.stderr)
    days = pd.concat(frames, ignore_index=True)
    breadth = pd.read_parquet(ROOT / "nse" / "breadth.parquet")
    days = days.merge(breadth, on="date", how="left")
    store = pd.read_parquet(ROOT / "nse" / "store.parquet")
    store = clean_store(store.loc[store["y20"] >= 0]).merge(
        pd.read_parquet(ROOT / "nse" / "p1_features.parquet"), on=["symbol", "date"], how="left")
    rng = np.random.default_rng(7)
    out = []
    for year in range(START_YEAR, END.year + 1):
        test = days.loc[days["date"].dt.year == year]
        train = store.loc[store["completion"] < pd.Timestamp(year, 1, 1)]
        if len(train) > args.max_rows:
            train = train.iloc[np.sort(rng.choice(len(train), args.max_rows, replace=False))]
        scored = test[["symbol", "date"]].copy()
        for name in ("baseline", "e1e2"):
            _, er = fit_predict(train, test, feature_columns(name), "y20")
            scored[f"er_{name}"] = er.astype("float32")
        out.append(scored)
        print(f"{year}: scored {len(test):,} stock-days {time.time() - started:.0f}s", file=sys.stderr)
    scores = pd.concat(out, ignore_index=True)
    for name in ("baseline", "e1e2"):
        scores[f"pct_{name}"] = scores.groupby("date")[f"er_{name}"].rank(pct=True).astype("float32")
    scores.to_parquet(ROOT / "nse" / "daily_scores.parquet", index=False)
    print(f"rows {len(scores):,} symbols {scores['symbol'].nunique()} in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
