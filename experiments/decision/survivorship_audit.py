"""Decision Step 3: how survivor-biased is the configured source universe?

A point-in-time US listing universe loses several percent of its names every
year to delistings, mergers and failures. If almost no configured symbol stops
trading before the snapshot end, the files are mostly survivors and every "what
happened next" outcome is biased toward the stocks that lived. This audit reads
only first/last sessions and the final bars of each symbol; it opens no study
outcome and needs no sealed artifact.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd


OUTPUT = Path("config/data/analogues/decision/step3-survivorship.json")
ENDED_EARLY_SESSIONS = 10
# Frozen reading guide. Published US-listing studies put yearly attrition in the
# mid-single-digit percent range; well below that implies a survivor snapshot.
LOW_ATTRITION_PERCENT = 2.0


def _summary(args: tuple[str, str, str]) -> dict[str, object] | None:
    config_path, dataset, symbol = args
    from market_analogues.adapters import source_from_spec
    from market_analogues.config import load_config
    from market_analogues.types import InstrumentKey
    source = source_from_spec(load_config(config_path).datasets[dataset])
    frame = source.load(InstrumentKey(dataset, symbol)).dropna(subset=["close", "volume"])
    frame = frame.loc[frame["close"] > 0]
    if frame.empty:
        return None
    close = frame["close"].to_numpy(float)
    tail = close[-61:]
    stamps = pd.to_datetime(frame["timestamp"])
    if stamps.dt.tz is not None:
        stamps = stamps.dt.tz_localize(None)
    return {
        "symbol": symbol, "first": stamps.iloc[0].normalize(), "last": stamps.iloc[-1].normalize(),
        "bars": int(len(frame)), "last_close": float(close[-1]),
        "final_60_log_return": float(np.log(tail[-1] / tail[0])) if len(tail) > 1 else np.nan,
        "median_dollar_volume_final_60": float(
            (frame["close"] * frame["volume"]).iloc[-60:].median()),
    }


def summarize(symbols: pd.DataFrame, benchmark_sessions: pd.DatetimeIndex) -> dict[str, object]:
    # The stock snapshot can end before the benchmark file; measure "ended early"
    # against the last session any symbol reached, not the benchmark's last session.
    benchmark_sessions = benchmark_sessions[benchmark_sessions <= symbols["last"].max()]
    end = benchmark_sessions.max()
    threshold = benchmark_sessions[max(0, len(benchmark_sessions) - 1 - ENDED_EARLY_SESSIONS)]
    symbols = symbols.copy()
    symbols["ended_early"] = symbols["last"] < threshold
    years = range(int(symbols["first"].dt.year.min()) + 1, int(end.year) + 1)
    by_year = []
    for year in years:
        start = pd.Timestamp(year=year, month=1, day=1)
        stop = pd.Timestamp(year=year + 1, month=1, day=1)
        active = symbols.loc[(symbols["first"] < start) & (symbols["last"] >= start)]
        ended = active.loc[active["ended_early"] & (active["last"] < stop)]
        by_year.append({
            "year": year, "active_at_start": int(len(active)), "ended": int(len(ended)),
            "attrition_percent": float(100 * len(ended) / len(active)) if len(active) else None,
        })
    usable = [row["attrition_percent"] for row in by_year
              if row["attrition_percent"] is not None and row["active_at_start"] >= 100
              and row["year"] < end.year]
    median_attrition = float(np.median(usable)) if usable else None
    ended = symbols.loc[symbols["ended_early"]]
    survived = symbols.loc[~symbols["ended_early"]]
    return {
        "snapshot_end": str(end.date()), "symbols": int(len(symbols)),
        "ended_early_symbols": int(len(ended)),
        "median_yearly_attrition_percent": median_attrition,
        "by_year": by_year,
        "final_60_log_return_median": {
            "ended_early": float(ended["final_60_log_return"].median()) if len(ended) else None,
            "survivors": float(survived["final_60_log_return"].median()) if len(survived) else None,
        },
        "ended_early_last_close_below_1_percent": float(
            100 * (ended["last_close"] < 1).mean()) if len(ended) else None,
        "reading": (
            "likely_survivor_snapshot_outcomes_biased_upward"
            if median_attrition is not None and median_attrition < LOW_ATTRITION_PERCENT
            else "attrition_present_check_whether_delisting_returns_are_recorded"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("config/datasets.example.yaml"))
    parser.add_argument("--dataset", default="nasdaq")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    from market_analogues.adapters import source_from_spec
    from market_analogues.config import load_config
    config = str(args.config.resolve())
    source = source_from_spec(load_config(config).datasets[args.dataset])
    benchmark = source.load_benchmark()
    if benchmark is None:
        print("benchmark required to define the session calendar", file=sys.stderr)
        return 2
    sessions = pd.DatetimeIndex(pd.to_datetime(benchmark["timestamp"]))
    if sessions.tz is not None:
        sessions = sessions.tz_localize(None)
    jobs = [(config, args.dataset, key.source_symbol) for key in source.instruments()]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = [row for row in executor.map(_summary, jobs, chunksize=32) if row]
    result = summarize(pd.DataFrame(rows), sessions.normalize())
    output = args.repository.resolve() / (args.output or OUTPUT)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "by_year"}, indent=2, default=str))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
