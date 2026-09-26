from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.p0 import evaluate as ev


def test_cumulative_heads_become_valid_class_probabilities():
    q = np.array([[0.6, 0.7, 0.2, 0.1, 0.3], [0.9, 0.5, 0.4, 0.2, 0.05]])
    p = ev.cumulative_to_classes(q)
    assert np.allclose(p.sum(axis=1), 1.0) and (p >= 0).all()


def test_rps_is_zero_for_a_certain_correct_forecast():
    y = np.array([0, 3, 5])
    probs = np.eye(6)[y]
    assert np.allclose(ev.rps(probs, y), 0.0)
    assert (ev.rps(np.full((3, 6), 1 / 6), y) > 0).all()


def _synthetic_store(n_dates=400, per_date=150, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2000-01-03", periods=n_dates * 5)[::5]
    rows = len(dates) * per_date
    frame = pd.DataFrame({
        "symbol": np.tile([f"S{i}" for i in range(per_date)], len(dates)),
        "date": np.repeat(dates, per_date)})
    frame["completion"] = frame["date"] + pd.Timedelta(days=30)
    frame["y20"] = rng.integers(0, 6, rows)
    frame["y20_pen"] = frame["y20"]
    frame["realized_r20"] = frame["y20"] - 1.0 + rng.normal(0, 0.5, rows)
    frame["realized_r20_pen"] = frame["realized_r20"]
    for col in ("exret5", "exret20"):
        frame[col] = rng.normal(0, 0.05, rows)
    frame["r_pct"] = 0.05
    frame["liquid"] = True
    frame["noise"] = rng.normal(size=rows)
    frame["leak"] = frame["y20"] + rng.normal(0, 0.3, rows)
    return frame


def _skill(store, features):
    ev.FIRST_TEST_YEAR["synthetic"] = 2002
    ev.ROUND_TRIP_COST["synthetic"] = 0.0
    pred = ev.walk_forward(store, features, "synthetic", "y20", max_rows=100_000)
    return ev.report(pred, store, "synthetic")["censored"]["all"]


def test_harness_detects_a_leaking_feature_and_ignores_noise(monkeypatch):
    monkeypatch.setattr(ev, "VAULT_START", pd.Timestamp("2006-01-01"))
    monkeypatch.setattr(ev, "MIN_TRAIN_ROWS", 5_000)
    store = _synthetic_store()
    leak = _skill(store, ["leak"])
    noise = _skill(store, ["noise"])
    assert leak["rps_skill_vs_train_clim_pct"] > 30
    assert leak["top_minus_all_mean"] > 1.0
    assert abs(noise["rps_skill_vs_train_clim_pct"]) < 1.0
    assert abs(noise["ic_realized_r20_mean"]) < 0.02
