"""Trend Quality Breakout strategy.

Uses Kaufman Efficiency Ratio to measure trend "cleanness" — only enters
when price is moving efficiently (low noise, strong directional bias).
Combined with EMA ribbon ordering and 52w high proximity for ultra-selective entries.

Key insight: Most momentum strategies use ROC (rate of change) which measures
magnitude but not quality. A stock can have high ROC from one gap-up day
followed by choppy action. Kaufman ER measures how much of the total path
was directional — filtering for CLEAN trends that are more likely to continue.

Different from existing momentum strategies:
- momentum.py: uses multi-period ROC thresholds
- high_52w: uses ROC + proximity to 52w high
- filtered_momentum: uses Minervini template
- This: uses Kaufman Efficiency + MA stack + rolling high breakout

Entry conditions:
1. Kaufman Efficiency Ratio (50-period) > 0.4 (strong directional trend)
2. MA stack: Close > EMA10 > EMA21 > SMA50 > SMA150 (all timeframes aligned)
3. Close at or near 50-day high (within 3%)
4. SMA50 slope positive (confirmed uptrend)
5. Volume above average (not a low-volume drift)

Exit: SMA10 trailing + ATR stop + time stop
"""
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.backtest.strategy import Strategy, Signal, SignalAction
from src.backtest.indicators import IndicatorSpec
from src.backtest.exit_rules import StopLossRule, TrailingMARule, TrailingATRStopRule, TimeStopRule

if TYPE_CHECKING:
    from ..portfolio import Portfolio


class TrendQualityBreakoutStrategy(Strategy):
    """Buy when trend quality is exceptionally high — clean, efficient trends."""

    def __init__(
        self,
        er_period: int = 50,
        er_threshold: float = 0.4,
        ma_fast: int = 50,
        ma_slow: int = 150,
        breakout_period: int = 50,
        breakout_pct: float = 3.0,
        volume_mult: float = 1.0,
        atr_period: int = 14,
        atr_stop_multiplier: float = 1.5,
        stop_loss_pct: float = 5.0,
        max_holding_days: int = 60,
        trailing_exit_type: str = 'sma10',
    ):
        self.er_period = er_period
        self.er_threshold = er_threshold
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.breakout_period = breakout_period
        self.breakout_pct = breakout_pct
        self.volume_mult = volume_mult
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_days = max_holding_days
        self.trailing_exit_type = trailing_exit_type

        self.ma_fast_col = f'sma_{ma_fast}'
        self.ma_slow_col = f'sma_{ma_slow}'
        self.atr_col = f'atr_{atr_period}'
        self.vol_sma_col = 'volume_sma_20'

    def indicators(self) -> list[IndicatorSpec]:
        specs = [
            IndicatorSpec('sma', {'period': self.ma_fast}),
            IndicatorSpec('sma', {'period': self.ma_slow}),
            IndicatorSpec('sma', {'period': 10}),
            IndicatorSpec('ema', {'period': 10}),
            IndicatorSpec('ema', {'period': 21}),
            IndicatorSpec('atr', {'period': self.atr_period, 'output_col': self.atr_col}),
            IndicatorSpec('rolling_high', {'period': self.breakout_period, 'output_col': f'high_{self.breakout_period}'}),
            IndicatorSpec('volume_sma', {'period': 20}),
        ]
        return specs

    def _kaufman_er(self, close_arr, idx, period):
        """Calculate Kaufman Efficiency Ratio at index.
        ER = abs(close[i] - close[i-period]) / sum(abs(close[j] - close[j-1]) for j in range)
        Range: 0 (choppy) to 1 (perfectly trending)
        """
        if idx < period:
            return float('nan')
        direction = abs(close_arr[idx] - close_arr[idx - period])
        volatility = 0.0
        for j in range(idx - period + 1, idx + 1):
            v = abs(close_arr[j] - close_arr[j - 1])
            if v != v:  # NaN check
                return float('nan')
            volatility += v
        if volatility == 0:
            return float('nan')
        return direction / volatility

    def on_bar(
        self,
        date: pd.Timestamp,
        universe: dict[str, pd.DataFrame],
        universe_idx: dict[str, int],
        portfolio: "Portfolio",
        current_prices: dict[str, dict] = None,
        universe_arrays: dict[str, dict] = None,
        **kwargs,
    ) -> list[Signal]:
        signals = []
        use_fast = universe_arrays is not None
        min_bars = max(self.ma_slow, self.er_period, self.breakout_period) + 30

        for symbol, df in universe.items():
            if portfolio.has_position(symbol):
                continue
            if symbol not in universe_idx:
                continue
            idx = universe_idx[symbol]
            if idx < min_bars:
                continue

            if use_fast and symbol in universe_arrays:
                arrays = universe_arrays[symbol]
                try:
                    close = arrays['close'][idx]
                    volume = arrays['volume'][idx]
                    ma_fast = arrays[self.ma_fast_col][idx]
                    ma_slow = arrays[self.ma_slow_col][idx]
                    ema10 = arrays['ema_10'][idx]
                    ema21 = arrays['ema_21'][idx]
                    sma10 = arrays['sma_10'][idx]
                    atr = arrays[self.atr_col][idx]
                    high_n = arrays[f'high_{self.breakout_period}'][idx]
                    vol_sma = arrays[self.vol_sma_col][idx]
                    close_arr = arrays['close']
                except (KeyError, IndexError):
                    continue

                # NaN check
                if any(v != v for v in [close, ma_fast, ma_slow, ema10, ema21, atr, high_n, vol_sma]):
                    continue

                # FILTER 1: MA Stack — all timeframes aligned
                if not (close > ema10 > ema21 > ma_fast > ma_slow):
                    continue

                # FILTER 2: Kaufman Efficiency Ratio
                er = self._kaufman_er(close_arr, idx, self.er_period)
                if er != er or er < self.er_threshold:
                    continue

                # FILTER 3: Near N-day high
                if high_n <= 0:
                    continue
                pct_from_high = (high_n - close) / high_n * 100
                if pct_from_high > self.breakout_pct:
                    continue

                # FILTER 4: SMA50 rising
                ma_fast_prev = arrays[self.ma_fast_col][idx - 5]
                if ma_fast_prev != ma_fast_prev or ma_fast <= ma_fast_prev:
                    continue

                # FILTER 5: Volume above average
                if vol_sma <= 0 or volume < vol_sma * self.volume_mult:
                    continue

                # Calculate stop
                stop_price = close - atr * self.atr_stop_multiplier
                max_stop = close * (1 - self.stop_loss_pct / 100)
                stop_price = max(stop_price, max_stop)

                score = er * 100  # Higher efficiency = better

                signals.append(Signal(
                    symbol=symbol,
                    action=SignalAction.BUY,
                    date=date,
                    stop_price=stop_price,
                    score=score,
                    metadata={
                        'kaufman_er': er,
                        'pct_from_high': pct_from_high,
                    }
                ))

            else:
                continue  # Fast path only

        return signals

    def exit_rules(self) -> list:
        rules = [StopLossRule()]
        if self.trailing_exit_type == 'sma10':
            rules.append(TrailingMARule(ma_column='sma_10'))
        elif self.trailing_exit_type == 'ema10':
            rules.append(TrailingMARule(ma_column='ema_10'))
        rules.append(TimeStopRule(max_holding_days=self.max_holding_days))
        return rules
