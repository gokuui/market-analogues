from __future__ import annotations

import numpy as np
import pytest

from market_analogues.distance import bounded_dtw


def test_local_dtw_order_agrees_with_aeon_reference() -> None:
    aeon = pytest.importorskip("aeon.distances")
    x = np.sin(np.linspace(0, 3 * np.pi, 64))
    candidates = [x, np.roll(x, 3), -x]
    local = [bounded_dtw(x, value, .12)[0] for value in candidates]
    trusted = [aeon.dtw_distance(x, value, window=.12) for value in candidates]
    assert np.argsort(local).tolist() == np.argsort(trusted).tolist()
