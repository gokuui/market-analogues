from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r.m04r14_t14_09_outcome_oracle import reference_episode
from market_analogues.causal_outcomes import (
    CausalOutcomeError,
    compute_episode_outcomes,
    deduplicate_episode_requests,
    outcome_embargo,
)


CONTRACT = json.loads((ROOT / "config/m04r14-t14-09-outcome-contract.json").read_text())
BINDINGS = {
    "episode_id": "synthetic-episode",
    "source_fingerprint": "source-fingerprint",
    "contract_digest": CONTRACT["contract_digest"],
    "source_content_digest": "source-content",
}


def _frames(rows: int = 180, cutoff_index: int = 30) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    dates = pd.bdate_range("2020-01-01", periods=rows)
    stock = pd.DataFrame({
        "timestamp": dates, "open": 100.0, "high": 101.0,
        "low": 99.0, "close": 100.0, "volume": 1000.0,
    })
    benchmark = pd.DataFrame({
        "timestamp": dates, "open": 200.0, "high": 202.0,
        "low": 198.0, "close": 200.0, "volume": 10000.0,
    })
    return stock, benchmark, dates[cutoff_index]


def _compute(stock: pd.DataFrame, benchmark: pd.DataFrame, cutoff: pd.Timestamp):
    return compute_episode_outcomes(stock, benchmark, cutoff=cutoff, **BINDINGS)


def _primary(bundle) -> pd.Series:
    return bundle.outcomes.loc[bundle.outcomes.horizon_sessions == 20].iloc[0]


def _null_normalized(records: list[dict]) -> list[dict]:
    return [{
        key: None if pd.isna(value) else value for key, value in row.items()
    } for row in records]


@pytest.mark.parametrize(
    "kind,expected_label,expected_offset",
    [
        ("favorable", "favorable_first", 3),
        ("adverse", "adverse_first", 2),
        ("ambiguous", "ambiguous_same_first_touch_bar", 4),
        ("no_touch", "no_touch", None),
    ],
)
def test_primary_barriers_and_separate_oracle_agree_exactly(
    kind: str, expected_label: str, expected_offset: int | None,
) -> None:
    stock, benchmark, cutoff = _frames()
    origin = int(stock.index[stock.timestamp == cutoff][0])
    if kind == "favorable":
        stock.loc[origin + 3, "high"] = 104.0
    elif kind == "adverse":
        stock.loc[origin + 2, "low"] = 98.0
    elif kind == "ambiguous":
        stock.loc[origin + 4, ["high", "low"]] = [104.0, 98.0]
    bundle = _compute(stock, benchmark, cutoff)
    expected_outcomes, expected_paths = reference_episode(
        stock, benchmark, cutoff=cutoff, **BINDINGS,
    )
    assert _null_normalized(bundle.outcomes.to_dict("records")) == expected_outcomes
    assert _null_normalized(bundle.paths.to_dict("records")) == expected_paths
    primary = _primary(bundle)
    assert primary.barrier_label == expected_label
    if expected_offset is None:
        assert pd.isna(primary.barrier_touch_offset)
    else:
        assert primary.barrier_touch_offset == expected_offset
    assert primary.origin_atr == 2.0
    assert primary.favorable_barrier_price == 104.0
    assert primary.adverse_barrier_price == 98.0


def test_returns_excursions_and_relative_return_are_hand_calculated() -> None:
    stock, benchmark, cutoff = _frames()
    origin = int(stock.index[stock.timestamp == cutoff][0])
    stock.loc[origin + 5, ["open", "high", "low", "close"]] = [109.0, 112.0, 108.0, 110.0]
    benchmark.loc[origin + 5, ["open", "high", "low", "close"]] = [208.0, 212.0, 207.0, 210.0]
    row = _compute(stock, benchmark, cutoff).outcomes.iloc[0]
    assert row.close_return == pytest.approx(.10)
    assert row.benchmark_relative_return == pytest.approx(1.10 / 1.05 - 1)
    assert row.maximum_favorable_excursion == pytest.approx(.12)
    assert row.maximum_adverse_excursion == pytest.approx(-.01)
    assert row.mfe_atr == pytest.approx(6.0)
    assert row.mae_atr == pytest.approx(-.5)
    assert row.time_to_mfe == 5
    assert row.time_to_mae == 1


def test_incomplete_horizons_preserve_paths_but_null_complete_measures() -> None:
    stock, benchmark, cutoff = _frames(rows=36, cutoff_index=30)
    bundle = _compute(stock, benchmark, cutoff)
    five = bundle.outcomes.iloc[0]
    ten = bundle.outcomes.iloc[1]
    assert bool(five.complete) is True
    assert bool(ten.complete) is False
    assert ten.status == "source_end_before_horizon"
    assert ten.available_sessions == 5
    assert pd.isna(ten.close_return)
    assert len(bundle.paths) == 5


def test_missing_stock_session_censors_instead_of_stretching_horizon() -> None:
    stock, benchmark, cutoff = _frames()
    origin = int(stock.index[stock.timestamp == cutoff][0])
    stock = stock.drop(index=origin + 2).reset_index(drop=True)
    bundle = _compute(stock, benchmark, cutoff)
    assert not bool(bundle.outcomes.iloc[0].complete)
    assert bundle.outcomes.iloc[0].status == "suspension_or_missing_session"
    assert bool(bundle.paths.iloc[1].expected_session_match) is False


def test_missing_benchmark_origin_retains_stock_measure_only() -> None:
    stock, benchmark, cutoff = _frames()
    benchmark = benchmark.loc[benchmark.timestamp != cutoff].reset_index(drop=True)
    row = _compute(stock, benchmark, cutoff).outcomes.iloc[0]
    assert bool(row.complete) is True
    assert row.close_return == 0.0
    assert pd.isna(row.benchmark_relative_return)
    assert row.benchmark_status == "benchmark_missing_relative_only"


def test_missing_benchmark_endpoint_does_not_censor_stock_measure() -> None:
    stock, benchmark, cutoff = _frames()
    origin = int(stock.index[stock.timestamp == cutoff][0])
    benchmark = benchmark.drop(index=origin + 5).reset_index(drop=True)
    row = _compute(stock, benchmark, cutoff).outcomes.iloc[0]
    assert bool(row.complete) is True
    assert row.close_return == 0.0
    assert pd.isna(row.benchmark_relative_return)
    assert row.benchmark_status == "benchmark_missing_relative_only"


def test_atr_is_causal_and_insufficient_history_only_disables_atr_measures() -> None:
    stock, benchmark, cutoff = _frames(cutoff_index=10)
    bundle = _compute(stock, benchmark, cutoff)
    primary = _primary(bundle)
    assert pd.isna(primary.origin_atr)
    assert primary.barrier_label == "censored"
    assert primary.barrier_status == "insufficient_atr_history"
    assert bool(primary.complete) is True
    assert primary.close_return == 0.0
    changed = stock.copy()
    changed.loc[changed.timestamp > cutoff, "high"] *= 1.5
    assert _primary(_compute(changed, benchmark, cutoff)).origin_atr is None


def test_invalid_ohlc_and_conflicting_duplicate_identity_fail_closed() -> None:
    stock, benchmark, cutoff = _frames()
    stock.loc[0, "close"] = 0.0
    with pytest.raises(CausalOutcomeError, match="non-positive"):
        _compute(stock, benchmark, cutoff)
    request = {
        "dataset": "nasdaq", "episode_id": "one",
        "source_fingerprint": "fingerprint", "symbol": "ABC", "cutoff": "2020-01-01",
    }
    assert deduplicate_episode_requests([request, deepcopy(request)]) == [request]
    changed = deepcopy(request)
    changed["symbol"] = "XYZ"
    with pytest.raises(CausalOutcomeError, match="conflicting"):
        deduplicate_episode_requests([request, changed])


def test_non_monotonic_and_duplicate_sessions_fail_closed() -> None:
    stock, benchmark, cutoff = _frames()
    reordered = stock.iloc[[1, 0, *range(2, len(stock))]].reset_index(drop=True)
    with pytest.raises(CausalOutcomeError, match="not increasing"):
        _compute(reordered, benchmark, cutoff)
    duplicate = pd.concat([stock.iloc[:1], stock], ignore_index=True)
    with pytest.raises(CausalOutcomeError, match="duplicate"):
        _compute(duplicate, benchmark, cutoff)


def test_deduplication_is_input_order_invariant() -> None:
    rows = [{
        "dataset": "nasdaq", "episode_id": value,
        "source_fingerprint": f"fp-{value}", "symbol": value,
    } for value in ("c", "a", "b")]
    assert deduplicate_episode_requests(rows) == deduplicate_episode_requests(list(reversed(rows)))


def test_outcome_embargo_requires_full_horizon_completion() -> None:
    assert outcome_embargo("2020-02-01", "2020-02-01", complete=True) == (True, "eligible")
    assert outcome_embargo("2020-02-02", "2020-02-01", complete=True) == (
        False, "outcome_not_yet_observable",
    )
    assert outcome_embargo("2020-01-01", "2020-02-01", complete=False) == (
        False, "incomplete_horizon",
    )


def test_price_scale_invariance_for_returns_barrier_labels_and_atr_units() -> None:
    stock, benchmark, cutoff = _frames()
    origin = int(stock.index[stock.timestamp == cutoff][0])
    stock.loc[origin + 3, "high"] = 104.0
    original = _compute(stock, benchmark, cutoff)
    scaled = stock.copy()
    scaled[["open", "high", "low", "close"]] *= 17.0
    transformed = _compute(scaled, benchmark, cutoff)
    for field in (
        "close_return", "maximum_favorable_excursion", "maximum_adverse_excursion",
        "mfe_atr", "mae_atr", "barrier_label", "barrier_touch_offset",
    ):
        assert _null_normalized(original.outcomes[[field]].to_dict("records")) \
            == _null_normalized(transformed.outcomes[[field]].to_dict("records"))
