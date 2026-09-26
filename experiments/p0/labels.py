"""P0 outcome labels: the stop-aware R-ladder and companion targets.

Origin t is a completed session; everything about the entry is known at its close.
Entry is the next session's open E. One unit of risk is R = R_MULT * ATR20(t), the
mean true range of the 20 sessions ending at t. The stop is E - R.

Bars t+1 .. t+H are scanned. A bar whose low reaches the stop ends the trade; if the
same bar also reaches a target, the stop counts first (conservative), and a gap
open below the stop exits at that open. The class is the highest whole-R level the
high reached strictly before the stopping bar:

    C0 stopped before +1R   C1 neither stop nor +1R by H   C2 [1R, 2R)
    C3 [2R, 3R)             C4 [3R, 5R)                    C5 >= 5R

realized_r is a simple stop-plus-time-exit payoff: the stop exit in R if stopped,
otherwise (close[t+H] - E) / R.
"""
from __future__ import annotations

import numpy as np

R_MULT = 1.25
ATR_LOOKBACK = 20
LEVELS = (1.0, 2.0, 3.0, 5.0)  # class boundaries above C1
N_CLASSES = 6


def true_range_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                   lookback: int = ATR_LOOKBACK) -> np.ndarray:
    if len(close) == 0:
        return np.array([], dtype=float)
    previous = np.r_[np.nan, close[:-1]]
    tr = np.fmax(high - low, np.fmax(np.abs(high - previous), np.abs(low - previous)))
    tr[0] = np.nan
    csum = np.cumsum(np.nan_to_num(tr))
    count = np.cumsum(np.isfinite(tr))
    atr = np.full(len(tr), np.nan)
    atr[lookback:] = (csum[lookback:] - csum[:-lookback]) / lookback
    full = np.r_[np.zeros(lookback), (count[lookback:] - count[:-lookback])] == lookback
    atr[~full] = np.nan
    return atr


def _class_from_reach(reach: np.ndarray, stopped: np.ndarray) -> np.ndarray:
    """reach = best high before the stop, in R above entry."""
    level = np.searchsorted(np.asarray(LEVELS), reach, side="right")  # 0..4
    cls = np.where(level == 0, np.where(stopped, 0, 1), level + 1)
    return cls.astype(np.int8)


def ladder_labels(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                  close: np.ndarray, horizon: int) -> dict[str, np.ndarray]:
    """Vectorized labels for every origin 0..n-horizon-2 (others are NaN / -1)."""
    n = len(close)
    atr = true_range_atr(high, low, close)
    out_cls = np.full(n, -1, dtype=np.int8)
    out_real = np.full(n, np.nan)
    out_mfe = np.full(n, np.nan)
    out_mae = np.full(n, np.nan)
    out_r_pct = np.full(n, np.nan)
    m = n - horizon - 1  # origins with bars t+1 .. t+horizon available
    if m <= 0:
        return {"cls": out_cls, "realized_r": out_real, "mfe_r": out_mfe,
                "mae_r": out_mae, "r_pct": out_r_pct}
    origins = np.arange(m)
    entry = open_[origins + 1]
    risk = R_MULT * atr[origins]
    stop = entry - risk
    win_o = np.lib.stride_tricks.sliding_window_view(open_[1:], horizon)[:m]
    win_h = np.lib.stride_tricks.sliding_window_view(high[1:], horizon)[:m]
    win_l = np.lib.stride_tricks.sliding_window_view(low[1:], horizon)[:m]
    hit = win_l <= stop[:, None]
    stopped = hit.any(axis=1)
    first = np.where(stopped, hit.argmax(axis=1), horizon)
    # best high strictly before the stopping bar (none if stopped on the entry bar)
    running = np.maximum.accumulate(win_h, axis=1)
    before = running[np.arange(m), np.maximum(first - 1, 0)]
    reach = np.where(first > 0, (before - entry) / risk, -np.inf)
    cls = _class_from_reach(reach, stopped)
    gap_open = win_o[np.arange(m), np.minimum(first, horizon - 1)]
    stop_exit = np.where(gap_open <= stop, gap_open, stop)
    realized = np.where(stopped, (stop_exit - entry) / risk,
                        (close[origins + horizon] - entry) / risk)
    valid = np.isfinite(risk) & (risk > 0) & np.isfinite(entry) & (entry > 0)
    out_cls[origins[valid]] = cls[valid]
    out_real[origins[valid]] = realized[valid]
    out_mfe[origins[valid]] = ((win_h.max(axis=1) - entry) / risk)[valid]
    out_mae[origins[valid]] = ((win_l.min(axis=1) - entry) / risk)[valid]
    out_r_pct[origins[valid]] = (risk / entry)[valid]
    return {"cls": out_cls, "realized_r": out_real, "mfe_r": out_mfe,
            "mae_r": out_mae, "r_pct": out_r_pct}


def ladder_label_scalar(open_, high, low, close, origin: int, horizon: int
                        ) -> tuple[int, float] | None:
    """Independent loop oracle for one origin, used only by tests."""
    atr = true_range_atr(np.asarray(high, float), np.asarray(low, float),
                         np.asarray(close, float))[origin]
    if origin + horizon + 1 > len(close) or not np.isfinite(atr) or atr <= 0:
        return None
    entry = open_[origin + 1]
    risk = R_MULT * atr
    stop = entry - risk
    best = -np.inf
    for j in range(origin + 1, origin + horizon + 1):
        if low[j] <= stop:
            exit_price = open_[j] if open_[j] <= stop else stop
            reach = (best - entry) / risk if best > -np.inf else -np.inf
            level = sum(reach >= x for x in LEVELS)
            return (0 if level == 0 else level + 1), (exit_price - entry) / risk
        best = max(best, high[j])
    reach = (best - entry) / risk
    level = sum(reach >= x for x in LEVELS)
    return (1 if level == 0 else level + 1), (close[origin + horizon] - entry) / risk
