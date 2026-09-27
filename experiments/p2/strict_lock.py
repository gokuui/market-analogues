"""Stricter circuit-lock rule for loser's engine, installed by monkeypatch (loser unchanged).

loser's guard treats any bar with (high-low)/low < 0.1% as locked. A flat bar can
also be a thin stock printing one price without hitting a circuit band. Under the
strict rule a flat bar only blocks a fill when the open is also at least
MIN_MOVE away from the previous close in the locked direction: up for entries,
down for exits. NSE's narrowest band is 2%. Otherwise the bar's high is nudged
just above the 0.1% range for that call only, so the engine's own guard lets the
fill through at the unchanged open price.
"""
from __future__ import annotations

MIN_MOVE = 0.019
_prev_close: dict[str, float] = {}


def _relax(prices: dict, symbols, direction: int, range_pct: float) -> dict:
    changed = None
    for symbol in symbols:
        bar = prices.get(symbol)
        if not bar:
            continue
        hi, lo, op = bar.get("high"), bar.get("low"), bar.get("open")
        if hi is None or lo is None or not lo or op is None:
            continue
        if (hi - lo) / lo * 100.0 >= range_pct:
            continue  # not flat: the engine's guard does not fire anyway
        prev = _prev_close.get(symbol)
        locked = prev is not None and prev > 0 and direction * (op / prev - 1.0) >= MIN_MOVE
        if not locked:
            if changed is None:
                changed = dict(prices)
            bar = dict(bar)
            bar["high"] = lo * (1 + 2 * range_pct / 100.0)
            changed[symbol] = bar
    return changed if changed is not None else prices


def install() -> None:
    from src.backtest.portfolio import Portfolio

    original_entries = Portfolio.execute_entries
    original_exits = Portfolio.execute_exits

    def execute_exits(self, date, prices):
        pct = self.exec_config.circuit_lock_range_pct
        symbols = [o.symbol for o in self.pending_orders if not o.is_entry]
        return original_exits(self, date, _relax(prices, symbols, -1, pct))

    def execute_entries(self, date, prices, universe_idx):
        pct = self.exec_config.circuit_lock_range_pct
        symbols = [o.symbol for o in self.pending_orders if o.is_entry]
        try:
            return original_entries(self, date, _relax(prices, symbols, +1, pct), universe_idx)
        finally:
            for symbol, bar in prices.items():  # today's close is tomorrow's previous close
                close = bar.get("close")
                if close:
                    _prev_close[symbol] = close

    Portfolio.execute_exits = execute_exits
    Portfolio.execute_entries = execute_entries
