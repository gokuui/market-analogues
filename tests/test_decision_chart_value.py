from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.decision import chart_value as cv
from market_analogues import causal_outcomes
from tests.conftest import make_bars


def _bars(n: int = 600, seed: int = 3) -> pd.DataFrame:
    return make_bars(n, seed).rename(columns={"date": "timestamp"})


def test_barrier_labels_match_frozen_causal_outcome_semantics():
    stock = _bars()
    sessions = cv.session_key(stock["timestamp"])
    labels = cv.barrier_labels(stock, sessions)
    prepared = causal_outcomes.prepare_outcome_sessions(stock, "stock")
    checked = 0
    for origin in range(25, len(stock) - cv.HORIZON, 7):
        atr = causal_outcomes._origin_atr(prepared, origin)
        future = prepared.iloc[origin + 1:origin + 1 + cv.HORIZON]
        expected = causal_outcomes._barrier(future, float(prepared.at[origin, "close"]),
                                            atr, True)[0]
        key = sessions[origin]
        if expected.startswith("ambiguous"):
            assert key not in labels.index
        else:
            assert labels.at[key, "label"] == expected
            assert labels.at[key, "completion"] == sessions[origin + cv.HORIZON]
        checked += 1
    assert checked > 50


def test_barrier_labels_drop_origins_whose_future_skips_a_session():
    stock = _bars(200)
    sessions = cv.session_key(stock["timestamp"])
    gapped = stock.drop(index=100).reset_index(drop=True)
    labels = cv.barrier_labels(gapped, sessions)
    missing = sessions[100]
    for origin in range(80, 100):
        assert sessions[origin] not in labels.index
    assert missing not in labels.index
    later = sum(sessions[origin] in labels.index for origin in range(101, 150))
    assert later >= 45  # origins after the gap keep their labels (bar ambiguous touches)


def test_stock_features_are_causal():
    stock = _bars()
    bench = cv.benchmark_features(_bars(seed=9))
    base = cv.stock_features(stock, bench)
    mutated = stock.copy()
    mutated.loc[400:, ["open", "high", "low", "close"]] *= 3.0
    mutated.loc[400:, "volume"] *= 10
    after = cv.stock_features(mutated, bench)
    pd.testing.assert_frame_equal(base.iloc[:400], after.iloc[:400])
    assert base.iloc[:cv.HISTORY].isna().all().all()
    assert base.iloc[cv.HISTORY:].notna().all(axis=1).mean() > 0.95
    assert not base.iloc[400:].equals(after.iloc[400:])


def test_simplex_matches_m1_grid_size():
    grid = cv.simplex(cv.MIXTURE_COMPONENTS)
    assert len(grid) == 286
    assert all(abs(sum(w.values()) - 1) < 1e-12 for w in grid)
    no_chart = [w for w in grid if w["composite"] == 0 and w["price_only"] == 0]
    assert len(no_chart) == 11


def _synthetic(chart_informative: bool, seed: int = 0):
    rng = np.random.default_rng(seed)
    months = pd.period_range("2019-01", periods=60, freq="M")
    rows = []
    for index, month in enumerate(months):
        fold = cv.ALL_FOLDS[min(index // 12, 4)]
        for _ in range(60):
            rows.append({"query_cutoff": month.to_timestamp() + pd.Timedelta(days=int(rng.integers(0, 27))),
                         "fold_id": fold})
    queries = pd.DataFrame(rows)
    queries["multiclass_evaluable"] = True
    queries["purged_evaluation_included"] = True
    n = len(queries)
    latent = rng.integers(0, 3, n)
    truth_labels = np.where(rng.random(n) < 0.55, latent, rng.integers(0, 3, n))
    truth = np.eye(3)[truth_labels]
    base = np.tile([0.3, 0.3, 0.4], (n, 1))

    def noisy(informative: bool, strength: float) -> np.ndarray:
        signal = np.eye(3)[latent] if informative else np.eye(3)[rng.integers(0, 3, n)]
        p = (1 - strength) * base + strength * signal
        return p / p.sum(axis=1, keepdims=True)

    matrices = {
        "matched_causal_history": base,
        "recent_return_volatility": noisy(True, 0.1),
        "composite": noisy(chart_informative, 0.5),
        "price_only": noisy(chart_informative, 0.5),
    }
    matrices["m1_candidate"] = cv.mix(
        {"matched_causal_history": .4, "composite": .1, "price_only": .2,
         "recent_return_volatility": .3}, matrices)
    matrices["gbm"] = noisy(True, 0.1)
    matrices["logistic"] = noisy(True, 0.05)
    return queries, matrices, truth


def test_evaluate_detects_informative_chart_components():
    queries, matrices, truth = _synthetic(chart_informative=True)
    result = cv.evaluate(queries, matrices, truth)
    decision = cv.decide(result)
    assert result["comparisons"]["chart_increment_over_mixture"]["lower_90"] > 0
    assert result["comparisons"]["chart_increment_over_feature_model"]["lower_90"] > 0
    assert decision["chart_adds_value"] is True
    for weights in result["forward_selected_weights"]["no_chart"].values():
        assert weights["composite"] == 0 and weights["price_only"] == 0


def test_evaluate_stops_when_chart_components_are_noise():
    queries, matrices, truth = _synthetic(chart_informative=False)
    result = cv.evaluate(queries, matrices, truth)
    decision = cv.decide(result)
    assert decision["chart_adds_value"] is False
    assert decision["recommendation"] == "stop_predictive_track_keep_descriptive_tool_if_used"
    increment = result["comparisons"]["chart_increment_over_mixture"]
    assert increment["mean"] < cv.STOP_RULES["minimum_chart_increment_skill_points"]


def test_evaluate_without_feature_model():
    queries, matrices, truth = _synthetic(chart_informative=True)
    for name in ("gbm", "logistic"):
        matrices.pop(name)
    result = cv.evaluate(queries, matrices, truth)
    assert set(result["comparisons"]) == {"chart_increment_over_mixture"}


def test_forward_selection_never_uses_held_or_later_folds():
    queries, matrices, truth = _synthetic(chart_informative=True)
    later = queries["fold_id"].isin(["validation_3", "final_untouched"]).to_numpy()
    tampered = truth.copy()
    tampered[later] = np.roll(tampered[later], 1, axis=1)
    folds = queries["fold_id"].to_numpy()
    eligible = np.ones(len(queries), bool)
    grid = cv.simplex(cv.MIXTURE_COMPONENTS)
    first = cv.forward_select(grid, matrices, truth, folds, eligible, "validation_2",
                              ("development", "validation_1"))[0]
    second = cv.forward_select(grid, matrices, tampered, folds, eligible, "validation_2",
                               ("development", "validation_1"))[0]
    assert first == second


def test_months_needed_scales_with_effect_and_noise():
    rng = np.random.default_rng(1)
    months = np.repeat(np.arange(40), 50).astype(str)
    diff = rng.normal(0.5, 5.0, len(months))
    small = cv.months_needed(diff, months, effect=0.25)["months_needed"]
    large = cv.months_needed(diff, months, effect=1.0)["months_needed"]
    assert small > large > 0
    assert cv.months_needed(diff, months, effect=0.0)["months_needed"] == float("inf")


def test_month_block_bootstrap_brackets_mean():
    rng = np.random.default_rng(2)
    months = np.repeat(np.arange(30), 20).astype(str)
    diff = rng.normal(1.0, 1.0, len(months))
    stats = cv.month_block_bootstrap(diff, months)
    assert stats["lower_90"] < stats["mean"] < stats["upper_90"]
    assert stats["months"] == 30


def test_symbol_rows_end_to_end(tmp_path):
    root = tmp_path / "bars"
    root.mkdir()
    for index, symbol in enumerate(("AAA", "BBB")):
        make_bars(700, seed=index + 11).to_parquet(root / f"{symbol}.parquet", index=False)
    bench = make_bars(700, seed=99)
    bench.to_parquet(tmp_path / "bench.parquet", index=False)
    config = tmp_path / "config.yaml"
    config.write_text(
        "artifact_dir: out\n"
        "datasets:\n"
        "  nasdaq:\n"
        "    adapter: directory\n"
        f"    path: {root}\n"
        "    format: parquet\n"
        "    timestamp_column: date\n"
        "    benchmark:\n"
        f"      path: {tmp_path / 'bench.parquet'}\n"
    )
    cutoff = str(make_bars(700, seed=11)["date"].iloc[500].date())
    train, queries = cv.build_feature_tables(str(config), "nasdaq", ["AAA", "BBB"],
                                             {"AAA": [cutoff]}, stride=5, workers=1)
    assert set(train["symbol"]) == {"AAA", "BBB"}
    assert set(train["label"]) <= set(cv.CLASSES)
    assert (pd.to_datetime(train["completion"]) > pd.to_datetime(train["cutoff"])).all()
    assert len(queries) == 1 and queries[list(cv.FEATURES)].notna().all(axis=1).iloc[0]
    queries["cutoff"] = pd.to_datetime(queries["cutoff"])
    probabilities = cv.fit_predict_monthly(train, queries, "logistic", max_rows=10_000)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
