from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd


@dataclass(frozen=True)
class DirectionalEvent:
    threshold: float
    direction: int
    extreme_index: int
    confirmation_index: int
    extreme_price: float
    confirmation_price: float
    amplitude: float
    duration: int
    volume_ratio: float


def directional_events(bars: pd.DataFrame, thresholds: tuple[float, ...] = (0.03, 0.06, 0.12)) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame(columns=[f.name for f in DirectionalEvent.__dataclass_fields__.values()])
    close = bars["close"].astype(float).to_numpy()
    volume = bars["volume"].astype(float).to_numpy()
    events: list[DirectionalEvent] = []
    for threshold in thresholds:
        # Begin by tracking a high. This does not assert that the series is in
        # an uptrend; an initial decline is confirmed causally from bar zero.
        mode = 1
        extreme_idx = 0
        extreme = close[0]
        start_idx = 0
        for i in range(1, len(close)):
            price = close[i]
            if mode >= 0:
                if price >= extreme:
                    extreme, extreme_idx = price, i
                elif price <= extreme * (1 - threshold):
                    base_vol = pd.Series(volume[max(start_idx, 0):i + 1]).median()
                    events.append(DirectionalEvent(
                        threshold, -1, extreme_idx, i, extreme, price,
                        price / extreme - 1, i - extreme_idx,
                        float(volume[i] / base_vol) if base_vol else 0.0,
                    ))
                    mode, extreme, extreme_idx, start_idx = -1, price, i, i
            if mode <= 0:
                if price <= extreme:
                    extreme, extreme_idx = price, i
                elif price >= extreme * (1 + threshold):
                    base_vol = pd.Series(volume[max(start_idx, 0):i + 1]).median()
                    events.append(DirectionalEvent(
                        threshold, 1, extreme_idx, i, extreme, price,
                        price / extreme - 1, i - extreme_idx,
                        float(volume[i] / base_vol) if base_vol else 0.0,
                    ))
                    mode, extreme, extreme_idx, start_idx = 1, price, i, i
    return pd.DataFrame([asdict(x) for x in events]).sort_values(
        ["confirmation_index", "threshold"], ignore_index=True
    ) if events else pd.DataFrame(columns=list(DirectionalEvent.__dataclass_fields__))
