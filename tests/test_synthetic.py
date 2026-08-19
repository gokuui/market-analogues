from __future__ import annotations

import numpy as np

from market_analogues.synthetic import FAMILIES, generate_case, transform_case


def test_generator_is_deterministic_and_ohlcv_valid() -> None:
    for family in FAMILIES:
        a = generate_case(family, 7)
        b = generate_case(family, 7)
        assert a.episode.bars.equals(b.episode.bars)
        bars = a.episode.bars
        assert (bars.high >= bars[["open", "close"]].max(axis=1)).all()
        assert (bars.low <= bars[["open", "close"]].min(axis=1)).all()
        assert (bars[["open", "high", "low", "close", "volume"]] > 0).all().all()


def test_transform_preserves_valid_bars() -> None:
    case = generate_case("trend_contraction_breakout", 3)
    transformed = transform_case(case, name="scaled", price_scale=9.0, volume_scale=100.0)
    bars = transformed.episode.bars
    assert np.isfinite(bars[["open", "high", "low", "close", "volume"]]).all().all()
    assert (bars.high >= bars[["open", "close"]].max(axis=1)).all()
    assert (bars.low <= bars[["open", "close"]].min(axis=1)).all()
