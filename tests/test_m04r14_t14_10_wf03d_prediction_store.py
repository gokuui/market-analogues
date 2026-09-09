from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import m04r14_t14_10_wf03d_prediction_store as store  # noqa: E402
from experiments.m04r import m04r14_t14_10_wf03d_prediction_listener as listener  # noqa: E402


def _registry() -> pd.DataFrame:
    return pd.DataFrame([{
        "query_id": "q1", "case_id": "case-1", "cutoff": "2020-03-31T00:00:00",
        "month": "2020-03", "fold_id": "validation_2", "scored": True,
    }])


def _links() -> pd.DataFrame:
    rows = []
    for method in ("composite", "price_only", "deterministic_random", "recent_return_volatility"):
        for rank in range(1, 21):
            rows.append({
                "query_id": "q1", "method": method, "rank": rank,
                "matched_episode_id": f"e{rank}",
                "distance_hex": float(rank / 10).hex(),
            })
    return pd.DataFrame(rows)


def _outcomes(future_label: str = "favorable_first", future_value: float = 999.0) -> pd.DataFrame:
    rows = []
    for rank in range(1, 21):
        for horizon in store.outcome_store.HORIZONS:
            rows.append({
                "episode_id": f"e{rank}", "horizon_sessions": horizon,
                "barrier_label": future_label if rank == 20 else (
                    "favorable_first" if rank % 2 else "adverse_first"
                ),
                "close_return": future_value if rank == 20 else rank / 100,
                "benchmark_relative_return": rank / 200,
                "maximum_favorable_excursion": rank / 50,
                "maximum_adverse_excursion": -rank / 70,
            })
    return pd.DataFrame(rows)


def _eligibility() -> pd.DataFrame:
    rows = []
    for method in ("composite", "price_only", "deterministic_random", "recent_return_volatility"):
        for rank in range(1, 21):
            for horizon in store.outcome_store.HORIZONS:
                rows.append({
                    "query_id": "q1", "method": method, "rank": rank,
                    "horizon_sessions": horizon, "eligible": rank != 20,
                    "reason": "eligible" if rank != 20 else "outcome_not_yet_observable",
                })
    return pd.DataFrame(rows)


def test_ineligible_analogue_future_cannot_change_primary_prediction() -> None:
    first = store._primary_predictions(
        _registry(), _links(), _outcomes(), _eligibility(), [],
    )
    second = store._primary_predictions(
        _registry(), _links(), _outcomes("no_touch", -999.0), _eligibility(), [],
    )
    pd.testing.assert_frame_equal(first, second)


def test_ineligible_analogue_future_cannot_change_continuous_prediction() -> None:
    first = store._continuous_predictions(_registry(), _links(), _outcomes(), _eligibility())
    second = store._continuous_predictions(
        _registry(), _links(), _outcomes("no_touch", -999.0), _eligibility(),
    )
    pd.testing.assert_frame_equal(first, second)


def test_baseline_uses_only_completed_prior_receipt_rows() -> None:
    prior = pd.DataFrame([
        {"query_id": "old", "completion_timestamp": "2020-03-30T00:00:00",
         "barrier_label": "favorable_first", "regime": "r"},
        {"query_id": "future", "completion_timestamp": "2020-04-01T00:00:00",
         "barrier_label": "adverse_first", "regime": "r"},
        {"query_id": "ambiguous", "completion_timestamp": "2020-03-01T00:00:00",
         "barrier_label": "ambiguous_same_first_touch_bar", "regime": "r"},
    ])
    observed = store._baseline_predictions(_registry(), prior, "r")
    unconditional = observed.loc[observed.lane == "unconditional_market_frequency"].iloc[0]
    assert unconditional.prior_eligible_rows == 1
    assert unconditional.favorable_probability == .6
    assert unconditional.adverse_probability == .2
    assert unconditional.no_touch_probability == .2


def test_final_month_validator_rejects_any_query_outcome_file(tmp_path) -> None:
    # Layout rejection is checked before seal parsing.
    root = tmp_path / "month-2024-01"; root.mkdir()
    for name in (*store.PREDICTION_FILES, "PREDICTIONS_SEALED.json", "query-outcomes.parquet"):
        (root / name).touch()
    try:
        store._validate_existing_month(root, "2024-01", True)
    except store.WalkForwardPredictionError as error:
        assert "layout" in str(error) or "final month" in str(error)
    else:
        raise AssertionError("final query outcome file was accepted")


def test_vectorized_path_summary_matches_scalar_formula_to_float_roundoff() -> None:
    matrix = pd.DataFrame({
        0: [3.0, 1.0, None, 2.0],
        1: [None, None, None, None],
        2: [4.0, 4.0, 1.0, None],
    }).to_numpy(dtype=float)
    padded = np.full((4, 126), np.nan)
    padded[:, :3] = matrix
    ranks = [1, 3, 7, 20]
    for weighted in (False, True):
        median, count, ess = store._path_matrix_summary(padded, ranks, weighted)
        for step in range(126):
            values = [None if pd.isna(value) else float(value) for value in padded[:, step]]
            expected = store.pointwise_path_prediction(values, ranks, weighted=weighted)
            assert (pd.isna(median[step]) and pd.isna(expected[0])) or median[step] == expected[0]
            assert count[step] == expected[1]
            assert np.isclose(ess[step], expected[2], rtol=0, atol=4e-15)


def test_listener_handles_already_exited_pid_without_polling() -> None:
    assert listener._wait_once(2_147_483_647) == "already_exited"
