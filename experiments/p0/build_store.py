"""P0 label store: causal features plus R-ladder labels on a common date grid.

Every STRIDE-th benchmark session is an origin date, and every eligible stock on
that date gets a row, so evaluation can compare models cross-sectionally per date.
Features are the 22 causal features from the decision study; labels come from
experiments.p0.labels. A label is kept only when the stock's bars t .. t+H+1 are
exactly the next benchmark sessions (no halts or gaps).

Delisting: when a stock's file ends before the snapshot end, origins whose window
runs past the end are censored. The *_pen columns instead mark likely failures
(last close under the price floor, or under half its final 60-session high) as a
stop-out, so results can be reported with and without that penalty.

Usage:
    PYTHONPATH=.:src .venv/bin/python -m experiments.p0.build_store --dataset nasdaq
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from experiments.decision.chart_value import FEATURES, benchmark_features, session_key
from experiments.p0.features import REQUIRED, p0_stock_features
from experiments.p0.labels import R_MULT, ladder_labels, true_range_atr

STRIDE = 5
HORIZONS = (5, 10, 20, 40)
PRIMARY = 20
EXCESS_HORIZONS = (5, 20)
FAILURE_PENALTY = 0.5  # likely failures exit at half their last close
MARKETS = {
    # snapshot_end: last stock session in the source files (verified 2026-09-26)
    "nasdaq": {"snapshot_end": "2026-03-30", "price_floor": 1.0},
    "nse": {"snapshot_end": "2026-02-11", "price_floor": 5.0},
}
OUTPUT_ROOT = Path("config/data/analogues/p0")

_CTX: dict[str, object] = {}


def _context(config: str, dataset: str):
    if not _CTX:
        from market_analogues.adapters import source_from_spec
        from market_analogues.config import load_config
        source = source_from_spec(load_config(config).datasets[dataset])
        bench = source.load_benchmark()
        bench = bench.dropna(subset=["open", "close"]).reset_index(drop=True)
        _CTX.update(source=source, bench_features=benchmark_features(bench),
                    sessions=session_key(bench["timestamp"]),
                    bench_open=bench["open"].to_numpy(float))
    return _CTX


def symbol_rows(args: tuple[str, str, str]) -> pd.DataFrame | None:
    config, dataset, symbol = args
    from market_analogues.types import InstrumentKey
    ctx = _context(config, dataset)
    sessions: pd.DatetimeIndex = ctx["sessions"]
    from market_analogues.adapters import SourceError
    try:
        stock = ctx["source"].load(InstrumentKey(dataset, symbol))
    except SourceError as error:  # unreadable or incomplete file: skipped, logged
        print(f"skip {symbol}: {error}", file=sys.stderr)
        return None
    stock = stock.dropna(subset=["open", "high", "low", "close", "volume"])
    stock = stock.loc[(stock[["open", "high", "low", "close"]] > 0).all(axis=1)]
    stock = stock.drop_duplicates("timestamp", keep=False).reset_index(drop=True)
    if len(stock) < 300:
        return None
    keys = session_key(stock["timestamp"])
    features = p0_stock_features(stock, ctx["bench_features"])
    o, h, l, c = (stock[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    v = stock["volume"].to_numpy(float)
    pos = sessions.get_indexer(keys)
    n = len(c)
    frame = pd.DataFrame(index=keys)
    frame["close"] = c
    frame["dollar_vol_20"] = pd.Series(c * v).rolling(20).median().to_numpy()
    for horizon in HORIZONS:
        lab = ladder_labels(o, h, l, c, horizon)
        cont = np.zeros(n, dtype=bool)
        idx = np.arange(n - horizon - 1)
        cont[idx] = ((pos[idx] >= 0) & (pos[idx + horizon + 1] >= 0)
                     & (pos[idx + horizon + 1] - pos[idx] == horizon + 1))
        cls = np.where(cont, lab["cls"], -1)
        frame[f"y{horizon}"] = cls
        frame[f"realized_r{horizon}"] = np.where(cont, lab["realized_r"], np.nan)
        if horizon == PRIMARY:
            frame["r_pct"] = lab["r_pct"]
            frame["mfe_r20"] = np.where(cont, lab["mfe_r"], np.nan)
            frame["mae_r20"] = np.where(cont, lab["mae_r"], np.nan)
            completion = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
            completion[idx] = keys.values[idx + horizon + 1]
            frame["completion"] = completion
    bench_open = ctx["bench_open"]
    for horizon in EXCESS_HORIZONS:
        ex = np.full(n, np.nan)
        idx = np.arange(max(n - horizon - 1, 0))
        ok = ((pos[idx] >= 0) & (pos[idx + horizon + 1] >= 0)
              & (pos[idx + horizon + 1] - pos[idx] == horizon + 1))
        i = idx[ok]
        ex[i] = (np.log(o[i + horizon + 1] / o[i + 1])
                 - np.log(bench_open[pos[i] + horizon + 1] / bench_open[pos[i] + 1]))
        frame[f"exret{horizon}"] = ex
    # delisting-aware penalty variant of the primary label
    market = MARKETS[dataset]
    ended = keys[-1] < pd.Timestamp(market["snapshot_end"]) - pd.Timedelta(days=15)
    last_close = c[-1]
    likely_failure = bool(ended and (last_close < market["price_floor"]
                                     or last_close < 0.5 * c[-60:].max()))
    frame["y20_pen"] = frame["y20"].to_numpy()
    frame["realized_r20_pen"] = frame["realized_r20"].to_numpy()
    if likely_failure:
        tail = np.arange(max(n - PRIMARY - 1, 0), n - 1)
        tail = tail[frame["y20"].to_numpy()[tail] < 0]  # only censored origins
        entry = o[tail + 1]
        risk = R_MULT * true_range_atr(h, l, c)[tail]
        exit_price = FAILURE_PENALTY * last_close
        frame.iloc[tail, frame.columns.get_loc("y20_pen")] = 0
        frame.iloc[tail, frame.columns.get_loc("realized_r20_pen")] = np.minimum(
            (exit_price - entry) / risk, -1.0)
    frame["ended"] = ended
    frame["likely_failure"] = likely_failure
    frame = frame.join(features)
    grid = pos >= 0
    grid[grid] = pos[grid] % STRIDE == 0
    frame = frame.loc[grid]
    frame = frame.dropna(subset=list(REQUIRED))
    frame = frame.loc[frame["y20"].ge(0) | frame["y20_pen"].ge(0)]
    if frame.empty:
        return None
    frame.index.name = "date"
    frame = frame.reset_index()
    frame.insert(0, "symbol", symbol)
    floats = frame.select_dtypes("float64").columns
    frame[floats] = frame[floats].astype("float32")
    return frame


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/datasets.example.yaml")
    parser.add_argument("--dataset", choices=sorted(MARKETS), required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--symbol-limit", type=int, default=None)
    args = parser.parse_args(argv)
    config = str(Path(args.config).resolve())
    from market_analogues.adapters import source_from_spec
    from market_analogues.config import load_config
    source = source_from_spec(load_config(config).datasets[args.dataset])
    symbols = sorted(k.source_symbol for k in source.instruments())
    if args.symbol_limit:
        symbols = symbols[:args.symbol_limit]
    started = time.time()
    frames = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [(config, args.dataset, s) for s in symbols]
        for done, frame in enumerate(pool.map(symbol_rows, jobs, chunksize=16), 1):
            if frame is not None:
                frames.append(frame)
            if done % 1000 == 0:
                print(f"{done}/{len(jobs)} symbols", file=sys.stderr)
    store = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"],
                                                             ignore_index=True)
    out_dir = OUTPUT_ROOT / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "store.parquet"
    store.to_parquet(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    y = store.loc[store["y20"] >= 0, "y20"]
    manifest = {
        "dataset": args.dataset, "rows": int(len(store)), "symbols": int(store["symbol"].nunique()),
        "dates": [str(store["date"].min().date()), str(store["date"].max().date())],
        "stride": STRIDE, "horizons": list(HORIZONS), "sha256": digest,
        "y20_base_rates": {f"C{k}": round(float((y == k).mean()), 4) for k in range(6)},
        "likely_failure_symbols": int(store.loc[store["likely_failure"], "symbol"].nunique()),
        "ended_symbols": int(store.loc[store["ended"], "symbol"].nunique()),
        "seconds": round(time.time() - started, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
