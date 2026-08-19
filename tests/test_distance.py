from __future__ import annotations

import numpy as np

from market_analogues.distance import bounded_dtw, representation_distance
from market_analogues.representation import represent
from market_analogues.synthetic import generate_case, transform_case


def test_bounded_dtw_identity_and_symmetry() -> None:
    x = np.sin(np.linspace(0, 4 * np.pi, 80))
    shifted = np.roll(x, 3)
    reverse = -x
    identity, path = bounded_dtw(x, x)
    xy, _ = bounded_dtw(x, shifted)
    yx, _ = bounded_dtw(shifted, x)
    opposite, _ = bounded_dtw(x, reverse)
    assert identity == 0
    assert len(path) == len(x)
    assert xy == pytest.approx(yx)
    assert xy < opposite


def test_exact_unit_transform_is_closer_than_reversed_path() -> None:
    base = generate_case("trend_contraction_breakout", 5)
    scaled = transform_case(base, name="scaled", price_scale=13, volume_scale=70)
    reversed_case = transform_case(base, name="reverse", reverse_returns=True)
    positive, _, _ = representation_distance(represent(base.episode), represent(scaled.episode))
    negative, _, _ = representation_distance(represent(base.episode), represent(reversed_case.episode))
    assert positive < 1e-8
    assert positive < negative


import pytest
