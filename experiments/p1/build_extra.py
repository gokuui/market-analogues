"""Build P1 candidate features for every label-store row, plus daily market breadth.

Outputs (per market, under config/data/analogues/p0/<market>/):
  p1_features.parquet   symbol, date, E1 + E2 + breadth columns (store grid dates)
  breadth.parquet       date, breadth_50, breadth_50_chg_20 (every session)

Usage:
    PYTHONPATH=.:src .venv/bin/python -m experiments.p1.build_extra --dataset nasdaq
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from experiments.decision.chart_value import session_key
from experiments.p0.build_store import STRIDE
from experiments.p0.evaluate import ROOT
from experiments.p1.features import BREADTH, E1, E2, e1_features, e2_features

_CTX: dict[str, object] = {}


def _context(config: str, dataset: str):
    if not _CTX:
        from market_analogues.adapters import source_from_spec
        from market_analogues.config import load_config
        source = source_from_spec(load_config(config).datasets[dataset])
        bench = source.load_benchmark().dropna(subset=["close"])
        keys = session_key(bench["timestamp"])
        _CTX.update(source=source, sessions=keys,
                    bench_close=pd.Series(bench["close"].to_numpy(float), index=keys))
    return _CTX


def load_clean(source, dataset: str, symbol: str) -> pd.DataFrame | None:
    from market_analogues.adapters import SourceError
    from market_analogues.types import InstrumentKey
    try:
        stock = source.load(InstrumentKey(dataset, symbol))
    except SourceError:
        return None
    stock = stock.dropna(subset=["open", "high", "low", "close", "volume"])
    stock = stock.loc[(stock[["open", "high", "low", "close"]] > 0).all(axis=1)]
    return stock.drop_duplicates("timestamp", keep=False).reset_index(drop=True)


def features_at(stock: pd.DataFrame, bench_close: pd.Series, rows: np.ndarray) -> pd.DataFrame:
    """E1 + E2 at the given row positions of a cleaned stock frame."""
    keys = session_key(stock["timestamp"])
    e1 = e1_features(stock, bench_close.reindex(keys).to_numpy()).iloc[rows]
    e1.index = rows
    e2 = e2_features(stock, rows)
    frame = e1.join(e2)
    frame.insert(0, "date", keys[rows])
    return frame


def symbol_job(args):
    config, dataset, symbol = args
    ctx = _context(config, dataset)
    stock = load_clean(ctx["source"], dataset, symbol)
    if stock is None or len(stock) < 300:
        return None, None
    keys = session_key(stock["timestamp"])
    pos = ctx["sessions"].get_indexer(keys)
    rows = np.flatnonzero((pos >= 0) & (pos % STRIDE == 0))
    rows = rows[rows >= 252]
    frame = features_at(stock, ctx["bench_close"], rows) if len(rows) else None
    if frame is not None:
        frame.insert(0, "symbol", symbol)
        floats = frame.select_dtypes("float64").columns
        frame[floats] = frame[floats].astype("float32")
    close = stock["close"].astype(float)
    sma50 = close.rolling(50).mean()
    ok = sma50.notna().to_numpy()
    above = pd.DataFrame({"date": keys[ok], "above": (close > sma50).to_numpy()[ok]})
    return frame, above


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/datasets.example.yaml")
    parser.add_argument("--dataset", required=True, choices=("nasdaq", "nse"))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--symbol-limit", type=int, default=None)
    args = parser.parse_args(argv)
    config = str(Path(args.config).resolve())
    store_symbols = pd.read_parquet(ROOT / args.dataset / "store.parquet",
                                    columns=["symbol"])["symbol"].unique()
    from market_analogues.adapters import source_from_spec
    from market_analogues.config import load_config
    source = source_from_spec(load_config(config).datasets[args.dataset])
    # breadth uses every configured symbol; features only the label-store symbols
    symbols = sorted(k.source_symbol for k in source.instruments())
    if args.symbol_limit:
        symbols = symbols[:args.symbol_limit]
    wanted = set(store_symbols)
    started = time.time()
    frames, counts = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [(config, args.dataset, s) for s in symbols]
        for done, (frame, above) in enumerate(pool.map(symbol_job, jobs, chunksize=8), 1):
            if frame is not None and frame["symbol"].iat[0] in wanted:
                frames.append(frame)
            if above is not None:
                counts.append(above)
            if done % 1000 == 0:
                print(f"{done}/{len(jobs)} symbols {time.time() - started:.0f}s", file=sys.stderr)
    daily = pd.concat(counts).groupby("date")["above"].agg(["sum", "count"])
    daily = daily.loc[daily["count"] >= 50]
    breadth = pd.DataFrame({"breadth_50": daily["sum"] / daily["count"]})
    breadth["breadth_50_chg_20"] = breadth["breadth_50"] - breadth["breadth_50"].shift(20)
    breadth.index.name = "date"
    breadth = breadth.reset_index()
    out = ROOT / args.dataset
    breadth.to_parquet(out / "breadth.parquet", index=False)
    feats = pd.concat(frames, ignore_index=True).merge(breadth, on="date", how="left")
    feats.to_parquet(out / "p1_features.parquet", index=False)
    print(f"rows {len(feats):,} symbols {feats['symbol'].nunique()} "
          f"columns {len(E1) + len(E2) + len(BREADTH)} in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
