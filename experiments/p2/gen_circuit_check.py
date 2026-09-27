"""Do the live gen books survive circuit-lock realism over 2015+?

Mirrors loser's scripts/live/pilot_backtest.run_gen config (1L capital, 3 positions,
2.5% risk, 33% max position, liquidity 5%, 650-day warmup) and toggles the same
two circuit flags as its apply_circuit(). Run with loser's interpreter:
    /home/vinay/code/loser/.venv/bin/python experiments/p2/gen_circuit_check.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import pandas as pd

LOSER = Path("/home/vinay/code/loser")
sys.path.insert(0, str(LOSER))

from src.backtest.config import (BacktestConfig, DataConfig, ExecutionConfig,  # noqa: E402
                                 PerformanceConfig, PortfolioConfig, PositionSizingConfig)
from src.backtest.engine import BacktestEngine  # noqa: E402
from src.backtest.strategies.gen498_momentum import Gen498MomentumStrategy  # noqa: E402
from src.backtest.strategies.gen_momentum import (Gen182MomentumStrategy,  # noqa: E402
                                                  Gen191MomentumStrategy)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from portfolio_filter import segment_metrics  # noqa: E402

START = "2015-01-01"
SPLIT = pd.Timestamp("2025-09-01")
BOOKS = {"gen498": Gen498MomentumStrategy, "gen182": Gen182MomentumStrategy,
         "gen191": Gen191MomentumStrategy}
SOURCES = {"validated": LOSER / "data/validated", "truedata": LOSER / "data/validated_truedata"}
OUT = Path(__file__).resolve().parents[2] / "config/data/analogues/p0/nse/gen_circuit"


def run(book: str, source: str, circuit: bool) -> dict[str, object]:
    vdir = SOURCES[source]
    data_start = (pd.Timestamp(START) - pd.Timedelta(days=650)).strftime("%Y-%m-%d")
    cal = pd.to_datetime(pd.read_parquet(vdir / "reliance.parquet", columns=["date"])["date"])
    warmup = int(((cal >= pd.Timestamp(data_start)) & (cal < pd.Timestamp(START))).sum())
    config = BacktestConfig(
        portfolio=PortfolioConfig(initial_capital=100_000, max_positions=3,
                                  position_sizing=PositionSizingConfig(
                                      risk_per_trade_pct=2.5, max_position_pct=33.0,
                                      liquidity_filter_monthly_pct=5.0)),
        execution=ExecutionConfig(commission_pct=0.1, slippage_pct=0.05, entry_on="open",
                                  exit_on="open", skip_circuit_locked=circuit,
                                  defer_circuit_lock_exit=circuit, circuit_lock_range_pct=0.1),
        data=DataConfig(validated_dir=vdir, start_date=data_start, end_date="2026-12-31",
                        min_history_days=200, min_confidence=1, min_warmup_days=warmup),
        performance=PerformanceConfig(enable_parallel=True))
    started = time.time()
    result = BacktestEngine(config, BOOKS[book]()).run()
    equity = result.equity_curve.set_index("date")["equity"].sort_index()
    equity.index = pd.to_datetime(equity.index)
    equity = equity.loc[START:]
    trades = result.trade_df
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{book}__{source}__{'on' if circuit else 'off'}"
    trades.to_csv(OUT / f"{tag}_trades.csv", index=False)
    equity.to_csv(OUT / f"{tag}_equity.csv")
    return {"book": book, "source": source, "circuit": circuit,
            "seconds": round(time.time() - started),
            "to_2025_08": segment_metrics(equity.loc[:SPLIT - pd.Timedelta(days=1)]),
            "from_2025_09": segment_metrics(equity.loc[SPLIT - pd.Timedelta(days=1):]),
            "full": segment_metrics(equity),
            "trades": int(len(trades))}


def main() -> int:
    global OUT
    if "--strict" in sys.argv:
        import strict_lock
        strict_lock.install()
        OUT = OUT.with_name(OUT.name + "_strict")
    results = []
    for source in SOURCES:
        for book in BOOKS:
            for circuit in (False, True):
                res = run(book, source, circuit)
                results.append(res)
                print(json.dumps(res, default=str), flush=True)
                (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
