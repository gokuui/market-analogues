"""Re-run the loser CLAUDE.md leaderboard strategies with circuit-lock realism off and on.

Definitions come from loser's validate_all_strategies.py registry (git 708eda5^):
the same make_config (capital, positions, 1% risk, 10% max position, liquidity 5%,
2015-01-01 start, data/validated). Blends are rebuilt from their legs the way
build_blend did: the sum of each leg's own-capital equity.
Run with loser's interpreter:
    /home/vinay/code/loser/.venv/bin/python experiments/p2/leaderboard_circuit_check.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import pandas as pd

LOSER = Path("/home/vinay/code/loser")
sys.path.insert(0, str(LOSER))
# Three leaderboard strategies were deleted from loser's release/prod-v1 branch
# (7bdbc73); they are restored verbatim from 7bdbc73^ with absolute imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))  # restored_strategies/

from src.backtest.config import (BacktestConfig, DataConfig, ExecutionConfig,  # noqa: E402
                                 PerformanceConfig, PortfolioConfig, PositionSizingConfig)
from src.backtest.engine import BacktestEngine  # noqa: E402
from restored_strategies.elder_impulse_breakout import ElderImpulseBreakoutStrategy  # noqa: E402
from src.backtest.strategies.high_52w_momentum import High52WMomentumStrategy  # noqa: E402
from restored_strategies.low_vol_trend import LowVolTrendStrategy  # noqa: E402
from src.backtest.strategies.ml_feature_strategy import MLFeatureStrategy  # noqa: E402
from src.backtest.strategies.momentum import MomentumStrategy  # noqa: E402
from src.backtest.strategies.tight_range_continuation import TightRangeContinuationStrategy  # noqa: E402
from restored_strategies.trend_quality_breakout import TrendQualityBreakoutStrategy  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from portfolio_filter import segment_metrics  # noqa: E402

SPLIT = pd.Timestamp("2025-09-01")
OUT = Path(__file__).resolve().parents[2] / "config/data/analogues/p0/nse/leaderboard_circuit"
FEATURES = str(LOSER / "data/features")


def mlf(**kw):
    base = dict(pro_score_min=2, kul_score_min=4, max_atr_pct=3.0, stop_loss_pct=3.0,
                max_holding_days=21, breakout_pct=5.0, trailing_exit_type="sma10",
                ma_stack_required=True, features_dir=FEATURES)
    base.update(kw)
    return MLFeatureStrategy(**base)


# leg name -> (factory, capital, max_positions)
LEGS = {
    "mom07_30k": (lambda: MomentumStrategy(atr_stop_multiplier=0.7, trailing_exit_type="atr",
                                           max_holding_days=60), 30_000, 3),
    "h52w_70k": (lambda: High52WMomentumStrategy(roc_min=5.0, atr_stop_multiplier=1.0,
                                                 max_holding_days=90), 70_000, 3),
    "tr_70k": (lambda: TightRangeContinuationStrategy(), 70_000, 3),
    "elder50_50k": (lambda: ElderImpulseBreakoutStrategy(er_threshold=0.50), 50_000, 3),
    "tq_50k": (lambda: TrendQualityBreakoutStrategy(er_threshold=0.50, max_holding_days=14,
                                                    stop_loss_pct=3.0, atr_stop_multiplier=0.7),
               50_000, 3),
    "elderlv_50k": (lambda: ElderImpulseBreakoutStrategy(er_threshold=0.40, max_atr_pct=2.0),
                    50_000, 3),
    "lvt_50k": (lambda: LowVolTrendStrategy(max_atr_pct=2.5, er_threshold=0.40), 50_000, 3),
    "mlf_atr3_p5": (lambda: mlf(er_threshold=0.50), 100_000, 5),
    "elder50_p3": (lambda: ElderImpulseBreakoutStrategy(er_threshold=0.50), 100_000, 3),
    "mom_v2_nostop": (lambda: MomentumStrategy(atr_stop_multiplier=1.0, trailing_exit_type="atr",
                                               max_holding_days=60), 100_000, 3),
    "vcp_er55_p5": (lambda: mlf(er_threshold=0.55, vcp_contraction_required=True), 100_000, 5),
    "vcp_rs75_p5": (lambda: mlf(er_threshold=0.55, vcp_contraction_required=True,
                                rs_percentile_min=75), 100_000, 5),
    "lvt_p3": (lambda: LowVolTrendStrategy(max_atr_pct=2.5, er_threshold=0.40), 100_000, 3),
}
# leaderboard rank/number -> legs
ROWS = {
    "#1 mom30+h52w70 blend": ["mom07_30k", "h52w_70k"],
    "#2 mom30+tr70 blend": ["mom07_30k", "tr_70k"],
    "#14 ml_features atr3.0 p5 h21": ["mlf_atr3_p5"],
    "#23 elder_impulse er50 p3": ["elder50_p3"],
    "#18 momentum v2 (no stop) p3": ["mom_v2_nostop"],
    "#19 VCP+er55 p5 h21": ["vcp_er55_p5"],
    "#9 ElderLV+LVT blend": ["elderlv_50k", "lvt_50k"],
    "#8 Elder+TQ blend": ["elder50_50k", "tq_50k"],
    "#20 VCP+RS75 p5 h21": ["vcp_rs75_p5"],
    "#25 lvt atr2.5 er40 p3": ["lvt_p3"],
}


def config(capital: int, positions: int, circuit: bool) -> BacktestConfig:
    return BacktestConfig(
        portfolio=PortfolioConfig(initial_capital=capital, max_positions=positions,
                                  position_sizing=PositionSizingConfig(
                                      risk_per_trade_pct=1.0, max_position_pct=10.0,
                                      liquidity_filter_monthly_pct=5.0)),
        execution=ExecutionConfig(commission_pct=0.1, slippage_pct=0.05, entry_on="open",
                                  exit_on="open", skip_circuit_locked=circuit,
                                  defer_circuit_lock_exit=circuit, circuit_lock_range_pct=0.1),
        data=DataConfig(validated_dir=LOSER / "data/validated", start_date="2015-01-01",
                        end_date="2026-12-31", min_history_days=200, min_confidence=1),
        performance=PerformanceConfig(enable_parallel=True))


def main() -> int:
    global OUT
    if "--strict" in sys.argv:
        import strict_lock
        strict_lock.install()
        OUT = OUT.with_name(OUT.name + "_strict")
    OUT.mkdir(parents=True, exist_ok=True)
    curves: dict[tuple[str, bool], pd.Series] = {}
    for leg, (factory, capital, positions) in LEGS.items():
        for circuit in (False, True):
            path = OUT / f"{leg}__{'on' if circuit else 'off'}_equity.csv"
            if path.exists():
                curves[(leg, circuit)] = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
                continue
            started = time.time()
            result = BacktestEngine(config(capital, positions, circuit), factory()).run()
            equity = result.equity_curve.set_index("date")["equity"].sort_index()
            equity.index = pd.to_datetime(equity.index)
            equity.to_csv(path)
            curves[(leg, circuit)] = equity
            print(f"{leg} circuit={circuit} {time.time() - started:.0f}s", file=sys.stderr, flush=True)
    rows = []
    for row, legs in ROWS.items():
        for circuit in (False, True):
            parts = [curves[(leg, circuit)] for leg in legs]
            frame = pd.concat(parts, axis=1).ffill()
            for i, leg in enumerate(legs):
                frame.iloc[:, i] = frame.iloc[:, i].fillna(LEGS[leg][1])
            equity = frame.sum(axis=1)
            rows.append({"row": row, "circuit": circuit,
                         "to_2025_08": segment_metrics(equity.loc[:SPLIT - pd.Timedelta(days=1)]),
                         "from_2025_09": segment_metrics(equity.loc[SPLIT - pd.Timedelta(days=1):])})
    (OUT / "results.json").write_text(json.dumps(rows, indent=2, default=str))
    for r in rows:
        a, b = r["to_2025_08"], r["from_2025_09"]
        print(f"{r['row']:34s} {'ON ' if r['circuit'] else 'off'} CAGR {a['cagr_pct']:6.1f} "
              f"DD {a['max_dd_pct']:5.1f} Cal {a['calmar']:5.2f} | 25/09+ CAGR {b.get('cagr_pct', 0):6.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
