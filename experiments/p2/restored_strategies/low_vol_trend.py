"""Low Volatility Trend strategy.

Only buys stocks with very low ATR% (daily volatility < threshold) that are
in strong uptrends near highs. The low volatility filter structurally limits
per-trade loss — if a stock moves 1.5% per day, a 3% stop = 2 days of
adverse movement, keeping losses tiny.

Designed for Track B prop trading: ultra-low DD at the cost of lower CAGR.

Entry conditions:
1. ATR% < max_atr_pct (low volatility stock)
2. Stage 2: Close > SMA50 > SMA150
3. Kaufman ER > threshold (clean trend, not choppy)
4. Near 50-day high (within breakout_pct%)
5. Close > EMA10 > EMA21 (short-term momentum aligned)

Exit: EMA10 trailing + tight stop + short time stop
"""
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.backtest.strategy import Strategy, Signal, SignalAction
from src.backtest.indicators import IndicatorSpec
from src.backtest.exit_rules import StopLossRule, TrailingMARule, TimeStopRule

if TYPE_CHECKING:
    from ..portfolio import Portfolio


class LowVolTrendStrategy(Strategy):
    """Buy low-volatility stocks in clean uptrends."""

    def __init__(
        self,
        max_atr_pct: float = 2.0,
        er_period: int = 50,
        er_threshold: float = 0.40,
        ma_fast: int = 50,
        ma_slow: int = 150,
        breakout_period: int = 50,
        breakout_pct: float = 3.0,
        atr_period: int = 14,
        stop_loss_pct: float = 3.0,
        max_holding_days: int = 14,
        trailing_exit_type: str = 'ema10',
    ):
        self.max_atr_pct = max_atr_pct
        self.er_period = er_period
        self.er_threshold = er_threshold
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.breakout_period = breakout_period
        self.breakout_pct = breakout_pct
        self.atr_period = atr_period
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_days = max_holding_days
        self.trailing_exit_type = trailing_exit_type

        self.ma_fast_col = f'sma_{ma_fast}'
        self.ma_slow_col = f'sma_{ma_slow}'
        self.atr_col = f'atr_{atr_period}'

    def indicators(self) -> list[IndicatorSpec]:
        return [
            IndicatorSpec('sma', {'period': self.ma_fast}),
            IndicatorSpec('sma', {'period': self.ma_slow}),
            IndicatorSpec('sma', {'period': 10}),
            IndicatorSpec('ema', {'period': 10}),
            IndicatorSpec('ema', {'period': 21}),
            IndicatorSpec('atr', {'period': self.atr_period, 'output_col': self.atr_col}),
            IndicatorSpec('rolling_high', {'period': self.breakout_period, 'output_col': f'high_{self.breakout_period}'}),
        ]

    def _kaufman_er(self, close_arr, idx, period):
        if idx < period:
            return float('nan')
        direction = abs(close_arr[idx] - close_arr[idx - period])
        volatility = 0.0
        for j in range(idx - period + 1, idx + 1):
            v = abs(close_arr[j] - close_arr[j - 1])
            if v != v:
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
                    ma_fast = arrays[self.ma_fast_col][idx]
                    ma_slow = arrays[self.ma_slow_col][idx]
                    ema10 = arrays['ema_10'][idx]
                    ema21 = arrays['ema_21'][idx]
                    atr = arrays[self.atr_col][idx]
                    high_n = arrays[f'high_{self.breakout_period}'][idx]
                    close_arr = arrays['close']
                except (KeyError, IndexError):
                    continue

                if any(v != v for v in [close, ma_fast, ma_slow, ema10, ema21, atr, high_n]):
                    continue
                if close <= 0:
                    continue

                # FILTER 1: Low volatility — ATR% < threshold
                atr_pct = atr / close * 100
                if atr_pct > self.max_atr_pct:
                    continue

                # FILTER 2: Stage 2 uptrend
                if not (close > ma_fast > ma_slow):
                    continue

                # FILTER 3: Short-term momentum aligned
                if not (close > ema10 > ema21):
                    continue

                # FILTER 4: Kaufman ER
                er = self._kaufman_er(close_arr, idx, self.er_period)
                if er != er or er < self.er_threshold:
                    continue

                # FILTER 5: Near N-day high
                if high_n <= 0:
                    continue
                pct_from_high = (high_n - close) / high_n * 100
                if pct_from_high > self.breakout_pct:
                    continue

                stop_price = close * (1 - self.stop_loss_pct / 100)
                # Score: lower ATR% = better (less volatile = safer)
                score = (self.max_atr_pct - atr_pct) * 100

                signals.append(Signal(
                    symbol=symbol,
                    action=SignalAction.BUY,
                    date=date,
                    stop_price=stop_price,
                    score=score,
                    metadata={'atr_pct': atr_pct, 'kaufman_er': er}
                ))
            else:
                continue

        return signals

    def exit_rules(self) -> list:
        rules = [StopLossRule()]
        if self.trailing_exit_type == 'ema10':
            rules.append(TrailingMARule(ma_column='ema_10'))
        elif self.trailing_exit_type == 'sma10':
            rules.append(TrailingMARule(ma_column='sma_10'))
        rules.append(TimeStopRule(max_holding_days=self.max_holding_days))
        return rules
