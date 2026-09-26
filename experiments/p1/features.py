"""P1 candidate features: E1 anchored/event features and E2 swing legs.

Every value at row t uses only bars 0..t. Breadth is cross-sectional and is joined
later from a daily series built over the whole universe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.p0.labels import true_range_atr

ZIGZAG_ATR = 1.5
LEGS = 3
CAP_DAYS = 252

E1 = ("trend_r2_63", "trend_r2_252", "info_discreteness_252", "close_loc_5", "close_loc_60",
      "gap_sigma_0", "gap_sigma_max_20", "days_since_gap2", "breakout_vol_max_10",
      "days_since_burst4", "burst4_count_20", "new_high_count_63", "dist_sma150",
      "sma150_slope_20", "stage", "mansfield_rs", "mansfield_rs_chg_20")
E2 = tuple([f"{side}_{k}_{what}" for side in ("down", "up") for k in range(1, LEGS + 1)
            for what in ("depth", "bars", "vol")]
           + ["contraction_count", "last_down_ratio", "pivot_dist_atr",
              "current_leg_atr", "current_leg_bars", "pivots_120"])
BREADTH = ("breadth_50", "breadth_50_chg_20")


def _days_since(flag: np.ndarray) -> np.ndarray:
    idx = np.arange(len(flag))
    last = np.maximum.accumulate(np.where(flag, idx, -1))
    return np.where(last >= 0, np.minimum(idx - last, CAP_DAYS), CAP_DAYS).astype(float)


def e1_features(stock: pd.DataFrame, bench_close: np.ndarray) -> pd.DataFrame:
    o, h, l, c, v = (stock[k].astype(float).to_numpy() for k in
                     ("open", "high", "low", "close", "volume"))
    s = pd.Series(c)
    logp = np.log(s)
    t = pd.Series(np.arange(len(c)), dtype=float)
    out = {}
    for n in (63, 252):
        r = logp.rolling(n).corr(t)
        out[f"trend_r2_{n}"] = (np.sign(r) * r ** 2).to_numpy()
    ret1 = logp.diff()
    out["info_discreteness_252"] = (np.sign(logp - logp.shift(252))
                                    * (-np.sign(ret1)).rolling(252).mean()).to_numpy()
    for n in (5, 60):
        lo, hi = pd.Series(l).rolling(n).min(), pd.Series(h).rolling(n).max()
        out[f"close_loc_{n}"] = ((s - lo) / (hi - lo).replace(0, np.nan)).to_numpy()
    sigma = ret1.rolling(252).std()
    gap = (np.log(pd.Series(o) / s.shift(1)) / sigma)
    out["gap_sigma_0"] = gap.to_numpy()
    out["gap_sigma_max_20"] = gap.rolling(20).max().to_numpy()
    out["days_since_gap2"] = _days_since((gap > 2).to_numpy())
    vs = pd.Series(v)
    vol_ratio = vs / vs.shift(1).rolling(50).mean().replace(0, np.nan)
    breakout = s > pd.Series(h).shift(1).rolling(50).max()
    out["breakout_vol_max_10"] = (vol_ratio.where(breakout, 0.0)).rolling(10).max().to_numpy()
    burst = (s / s.shift(1) >= 1.04).to_numpy()
    out["days_since_burst4"] = _days_since(burst)
    out["burst4_count_20"] = pd.Series(burst.astype(float)).rolling(20).sum().to_numpy()
    new_high = (s >= s.rolling(252).max()).astype(float)
    out["new_high_count_63"] = new_high.rolling(63).sum().to_numpy()
    sma150 = s.rolling(150).mean()
    dist = s / sma150 - 1
    slope = sma150 / sma150.shift(20) - 1
    out["dist_sma150"] = dist.to_numpy()
    out["sma150_slope_20"] = slope.to_numpy()
    rising, falling = slope > 0.01, slope < -0.01
    above = dist > 0
    stage = np.select([above & rising, above & ~rising, ~above & falling], [2, 3, 4], 1).astype(float)
    stage[~np.isfinite(slope.to_numpy())] = np.nan
    out["stage"] = stage
    rs = s / pd.Series(bench_close)
    mrs = rs / rs.rolling(250).mean() - 1
    out["mansfield_rs"] = mrs.to_numpy()
    out["mansfield_rs_chg_20"] = (mrs - mrs.shift(20)).to_numpy()
    return pd.DataFrame(out).replace([np.inf, -np.inf], np.nan)


def e2_features(stock: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    """Online ZigZag swing legs, evaluated only at the requested row positions."""
    h, l, c, v = (stock[k].astype(float).to_numpy() for k in ("high", "low", "close", "volume"))
    atr = true_range_atr(h, l, c)
    vol50 = pd.Series(v).rolling(50).mean().to_numpy()
    cumv = np.r_[0.0, np.cumsum(v)]
    want = np.zeros(len(c), dtype=bool)
    want[rows] = True
    result = np.full((len(c), len(E2)), np.nan)
    pivots: list[tuple[int, float, int]] = []  # (bar, price, +1 high / -1 low)
    direction, ext, ext_i = 0, np.nan, 0
    for t in range(len(c)):
        thr = ZIGZAG_ATR * atr[t]
        if np.isfinite(thr) and thr > 0:
            if direction == 0:
                direction, ext, ext_i = 1, h[t], t
            elif direction == 1:
                if h[t] >= ext:
                    ext, ext_i = h[t], t
                elif ext - l[t] >= thr:
                    pivots.append((ext_i, ext, 1))
                    direction, ext, ext_i = -1, l[t], t
            else:
                if l[t] <= ext:
                    ext, ext_i = l[t], t
                elif h[t] - ext >= thr:
                    pivots.append((ext_i, ext, -1))
                    direction, ext, ext_i = 1, h[t], t
        if not want[t] or len(pivots) < 2 or not np.isfinite(atr[t]) or atr[t] <= 0:
            continue
        a = atr[t]
        tail = pivots[-60:]  # enough history for 3+3 legs and the 120-bar pivot count
        legs = [(p0, p1) for p0, p1 in zip(tail[:-1], tail[1:])]
        row = []
        for side in (-1, 1):  # down legs end at a low (-1), up legs end at a high (+1)
            chosen = [leg for leg in reversed(legs) if leg[1][2] == side][:LEGS]
            for k in range(LEGS):
                if k < len(chosen):
                    (i0, p0, _), (i1, p1, _) = chosen[k]
                    bars = max(i1 - i0, 1)
                    vol = (cumv[i1 + 1] - cumv[i0 + 1]) / bars
                    row += [abs(p1 - p0) / a, float(bars),
                            vol / vol50[t] if vol50[t] > 0 else np.nan]
                else:
                    row += [np.nan, np.nan, np.nan]
        downs = [abs(p1[1] - p0[1]) for p0, p1 in reversed(legs) if p1[2] == -1]
        count = 1 if downs else 0
        for later, earlier in zip(downs, downs[1:]):
            if later < earlier:
                count += 1
            else:
                break
        ratio = downs[0] / downs[1] if len(downs) > 1 and downs[1] > 0 else np.nan
        highs = [p for p in tail if p[2] == 1]
        pivot_dist = (c[t] - highs[-1][1]) / a if highs else np.nan
        last_i, last_p, _ = pivots[-1]
        recent = sum(1 for p in tail if p[0] > t - 120)
        row += [float(count), ratio, pivot_dist, (c[t] - last_p) / a, float(t - last_i),
                float(recent)]
        result[t] = row
    return pd.DataFrame(result[rows], columns=list(E2), index=rows)
