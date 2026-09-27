"""P2 portfolio test: does a causal model-score entry filter improve loser momentum strategies?

Run with loser's interpreter; this script imports loser's engine but modifies no loser file:
    /home/vinay/code/loser/.venv/bin/python experiments/p2/portfolio_filter.py

For each strategy (mom atr0.7 h60 p3 = S-2, mom BestV2) and each filter
(none, baseline score, E1+E2 score), a full 2015+ backtest runs with circuit-lock
realism ON (locked entry bars don't fill; limit-down exits wait). The filter
drops a candidate signal when its within-date score percentile is below the
1/3 quantile of the strategy's own candidate-signal percentiles over the trailing
250 sessions (no filtering until 30 candidates have been seen). Dropped signals
free their slot for the next-ranked signal, as the engine ranks after on_bar.

Scores come from experiments/p2/daily_scores.py: each year is scored by models
fit only on outcomes completed before that year.

Segments: development (to 2025-08-31) and the previously held-back vault
(2025-09-01 onward, already opened once at the trade level).
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

LOSER = Path("/home/vinay/code/loser")
sys.path.insert(0, str(LOSER))

from src.backtest.config import (BacktestConfig, DataConfig, ExecutionConfig,  # noqa: E402
                                 PerformanceConfig, PortfolioConfig, PositionSizingConfig)
from src.backtest.engine import BacktestEngine  # noqa: E402
from src.backtest.strategies.momentum import MomentumStrategy  # noqa: E402

HERE = Path(__file__).resolve().parents[2]
SCORES = HERE / "config/data/analogues/p0/nse/daily_scores.parquet"
OUT = HERE / "config/data/analogues/p0/nse/p2_portfolio"
VAULT_START = pd.Timestamp("2025-09-01")
SKIP_QUANTILE = 1 / 3
WINDOW_SESSIONS = 250
MIN_HISTORY = 30
STRATEGIES = {
    "mom_atr07_p3": dict(atr_stop_multiplier=0.7, trailing_exit_type="atr", max_holding_days=60),
    "mom_bestv2_p3": dict(atr_stop_multiplier=1.0, trailing_exit_type="atr", max_holding_days=60,
                          stop_loss_pct=5.0),
}


def make_config(circuit: bool) -> BacktestConfig:
    return BacktestConfig(
        portfolio=PortfolioConfig(
            initial_capital=100_000, max_positions=3,
            position_sizing=PositionSizingConfig(risk_per_trade_pct=1.0, max_position_pct=10.0,
                                                 liquidity_filter_monthly_pct=5.0)),
        execution=ExecutionConfig(commission_pct=0.1, slippage_pct=0.05, entry_on="open",
                                  exit_on="open", skip_circuit_locked=circuit,
                                  defer_circuit_lock_exit=circuit, circuit_lock_range_pct=0.1),
        data=DataConfig(validated_dir=LOSER / "data/validated", start_date="2015-01-01",
                        end_date="2026-12-31", min_history_days=200, min_confidence=1),
        performance=PerformanceConfig(enable_parallel=True))


class ScoreFilter:
    def __init__(self, pct: dict[tuple[str, pd.Timestamp], float]):
        self.pct = pct
        self.history: list[tuple[pd.Timestamp, float]] = []
        self.seen = self.dropped = self.unscored = 0

    def __call__(self, signals, date):
        date = pd.Timestamp(date).normalize()
        cutoff = date - pd.Timedelta(days=int(WINDOW_SESSIONS * 1.45))
        self.history = [(d, p) for d, p in self.history if d >= cutoff]
        past = [p for _, p in self.history]
        threshold = np.quantile(past, SKIP_QUANTILE) if len(past) >= MIN_HISTORY else -np.inf
        kept = []
        for s in signals:
            self.seen += 1
            p = self.pct.get((s.symbol, date))
            if p is None:
                self.unscored += 1
                kept.append(s)
                continue
            self.history.append((date, p))
            if p >= threshold:
                kept.append(s)
            else:
                self.dropped += 1
        return kept


def filtered_strategy(params: dict, score_filter: ScoreFilter | None) -> MomentumStrategy:
    class Filtered(MomentumStrategy):
        def on_bar(self, **kw):
            signals = super().on_bar(**kw) or []
            return score_filter(signals, kw["date"]) if score_filter else signals
    return Filtered(**params)


def segment_metrics(equity: pd.Series) -> dict[str, float]:
    equity = equity.dropna()
    if len(equity) < 20:
        return {}
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    dd = (equity / equity.cummax() - 1).min()
    daily = equity.pct_change().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan
    return {"cagr_pct": 100 * cagr, "max_dd_pct": -100 * dd,
            "calmar": cagr / -dd if dd < 0 else np.nan, "sharpe": sharpe,
            "start": str(equity.index[0].date()), "end": str(equity.index[-1].date())}


def run(name: str, filter_name: str, circuit: bool, pct_maps: dict) -> dict[str, object]:
    score_filter = ScoreFilter(pct_maps[filter_name]) if filter_name != "none" else None
    started = time.time()
    result = BacktestEngine(make_config(circuit), filtered_strategy(STRATEGIES[name], score_filter)).run()
    equity = result.equity_curve.set_index("date")["equity"]
    equity.index = pd.to_datetime(equity.index)
    trades = result.trade_df
    tag = f"{name}__{filter_name}__{'circuit' if circuit else 'nocircuit'}"
    OUT.mkdir(parents=True, exist_ok=True)
    trades.to_csv(OUT / f"{tag}_trades.csv", index=False)
    equity.to_csv(OUT / f"{tag}_equity.csv")
    entry = pd.to_datetime(trades["entry_date"]) if len(trades) else pd.Series(dtype="datetime64[ns]")
    dev_trades = trades.loc[entry < VAULT_START] if len(trades) else trades
    return {
        "strategy": name, "filter": filter_name, "circuit_realism": circuit,
        "seconds": round(time.time() - started),
        "engine_stats": {k: float(getattr(result.stats, k)) for k in
                         ("cagr", "max_drawdown_pct", "calmar_ratio", "sharpe_ratio",
                          "win_rate", "total_trades")},
        "development": segment_metrics(equity.loc[equity.index < VAULT_START]),
        "vault": segment_metrics(equity.loc[equity.index >= VAULT_START - pd.Timedelta(days=1)]),
        "dev_trades": int(len(dev_trades)),
        "dev_mean_trade_pct": float(dev_trades["pnl_pct"].mean()) if len(dev_trades) else None,
        "filter_counts": ({"seen": score_filter.seen, "dropped": score_filter.dropped,
                           "unscored": score_filter.unscored} if score_filter else None),
    }


def main() -> int:
    scores = pd.read_parquet(SCORES, columns=["symbol", "date", "pct_baseline", "pct_e1e2"])
    keys = list(zip(scores["symbol"], pd.to_datetime(scores["date"])))
    pct_maps = {"baseline": dict(zip(keys, scores["pct_baseline"].astype(float))),
                "e1e2": dict(zip(keys, scores["pct_e1e2"].astype(float)))}
    del scores
    results = []
    plan = [(s, f, True) for s in STRATEGIES for f in ("none", "baseline", "e1e2")]
    plan += [(s, "none", False) for s in STRATEGIES]  # reference: original engine behaviour
    for name, filter_name, circuit in plan:
        res = run(name, filter_name, circuit, pct_maps)
        results.append(res)
        print(json.dumps({k: res[k] for k in ("strategy", "filter", "circuit_realism", "seconds",
                                                "development", "dev_trades", "filter_counts")},
                         default=str), flush=True)
        (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
