from __future__ import annotations

import numpy as np
import pandas as pd

from market_analogues.structural import directional_events


def test_directional_events_use_confirmation_not_future_extreme() -> None:
    close = np.array([100, 102, 105, 109, 108, 103, 98, 99, 103, 108], dtype=float)
    bars = pd.DataFrame({"close": close, "volume": np.full(len(close), 1000.0)})
    events = directional_events(bars, thresholds=(0.05,))
    down = events.iloc[0]
    assert down.direction == -1
    assert down.extreme_index == 3
    assert down.confirmation_index == 5
    assert down.confirmation_index > down.extreme_index


def test_event_prefix_is_unchanged_when_future_is_mutated() -> None:
    close = np.array([100, 106, 99, 92, 98, 105, 97, 110], dtype=float)
    bars = pd.DataFrame({"close": close, "volume": np.arange(len(close)) + 100.0})
    prefix = directional_events(bars.iloc[:6], thresholds=(0.05,))
    mutated = bars.copy()
    mutated.loc[6:, "close"] *= 10
    again = directional_events(mutated.iloc[:6], thresholds=(0.05,))
    pd.testing.assert_frame_equal(prefix, again)
