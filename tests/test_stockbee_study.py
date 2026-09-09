from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from market_analogues.stockbee_study import (
    StockbeeStudyError,
    clustered_winners,
    symbol_risk_rows,
)
from market_analogues.types import stable_hash
from experiments.m04r import m04r14_t14_11_stockbee_risk_set as risk_set
from experiments.m04r import verify_m04r14_t14_11_stockbee_risk_set as risk_verifier


def _bars(rows: int = 340) -> pd.DataFrame:
    close = np.full(rows, 100.0); open_ = close.copy()
    high = np.full(rows, 101.0); low = np.full(rows, 99.0)
    start = 252
    open_[start] = 100.; high[start] = 105.; low[start] = 99.; close[start] = 104.
    close[start + 20] = 125.; open_[start + 20] = 124.; high[start + 20] = 126.; low[start + 20] = 123.
    close[start + 62] = 130.; open_[start + 62] = 129.; high[start + 62] = 131.; low[start + 62] = 128.
    return pd.DataFrame({
        "timestamp": pd.bdate_range("2020-01-01", periods=rows),
        "open": open_, "high": np.maximum(high, close), "low": np.minimum(low, close),
        "close": close, "volume": np.full(rows, 20_000.),
    })


def test_symbol_kernel_uses_previous_close_and_inclusive_horizon() -> None:
    result = symbol_risk_rows(_bars(), "SYN")
    row21 = result.loc[(result.start_position == 252) & (result.horizon_sessions == 21)].iloc[0]
    row63 = result.loc[(result.start_position == 252) & (result.horizon_sessions == 63)].iloc[0]
    assert row21.forward_close_return == pytest.approx(.25)
    assert row63.forward_close_return == pytest.approx(.30)
    assert row21.winner_25pct and row63.winner_25pct
    assert row21.up_close_4pct_start_day
    assert row21.true_range_4pct_start_day
    assert row21.bullish_range_expansion_4pct_start_day
    assert row21.up_close_4pct_first_5_count == 1
    assert row21.up_close_4pct_pre_start_20_sessions == 0


def test_symbol_kernel_has_exact_history_and_future_boundaries() -> None:
    bars = _bars(); result = symbol_risk_rows(bars, "SYN")
    for horizon, group in result.groupby("horizon_sessions"):
        assert group.start_position.min() == 252
        assert group.start_position.max() == len(bars) - horizon
        assert len(group) == len(bars) - horizon - 252 + 1
    assert result.prior_return_63.notna().all()
    assert result.prior_volatility_20.notna().all()


def test_event_clustering_keeps_earliest_and_peak() -> None:
    rows = pd.DataFrame({
        "symbol": ["A"] * 5 + ["B"], "horizon_sessions": [21] * 6,
        "start": pd.bdate_range("2024-01-01", periods=6),
        "start_position": [10, 11, 12, 15, 16, 10],
        "winner_25pct": [True] * 6,
        "forward_close_return": [.25, .4, .3, .26, .28, .5],
    })
    result = clustered_winners(rows)
    a = result.loc[result.symbol == "A"].sort_values("start_position")
    assert a.start_position.tolist() == [10, 15]
    assert a.event_run_length.tolist() == [3, 2]
    assert a.event_peak_return.tolist() == [.4, .28]
    assert len(result.loc[result.symbol == "B"]) == 1


def test_invalid_order_is_rejected_and_invalid_ohlc_window_is_excluded() -> None:
    bars = _bars(); bars.loc[2, "timestamp"] = bars.loc[1, "timestamp"]
    with pytest.raises(StockbeeStudyError, match="timestamps"):
        symbol_risk_rows(bars, "BAD")
    bars = _bars(); bars.loc[2, "high"] = 50
    assert not symbol_risk_rows(bars, "BAD").empty
    bars = _bars(600); bars.loc[260, "high"] = 50
    result = symbol_risk_rows(bars, "BAD")
    assert 252 not in set(result.start_position)
    assert 261 not in set(result.start_position)


def test_stockbee_contract_self_seal_and_offset_words_agree() -> None:
    contract = json.loads((ROOT / "config/m04r14-t14-11-stockbee-contract.json").read_text())
    digest = contract.pop("contract_digest")
    assert digest == stable_hash(contract)
    assert contract["outcomes"]["return_formula"] == "close_at_t_plus_horizon_minus_1_divided_by_close_at_t_minus_1_minus_1"
    assert contract["exposures"]["windows_relative_to_start"]["full_move"] == [0, "horizon_minus_1"]


def test_independent_scalar_oracle_matches_vector_kernel() -> None:
    bars = _bars(); vector = symbol_risk_rows(bars, "SYN")
    for horizon in (21, 63):
        for position in (252, 270):
            observed = vector.loc[
                (vector.horizon_sessions == horizon) & (vector.start_position == position)
            ].iloc[0]
            expected = risk_verifier._scalar_row(bars, "SYN", position, horizon)
            for name, value in expected.items():
                if isinstance(value, float):
                    assert observed[name] == pytest.approx(value, abs=5e-13)
                else:
                    assert observed[name] == value


def test_symbol_sharding_is_deterministic_and_complete() -> None:
    symbols = [f"S{i}" for i in range(1000)]
    first = [risk_set._shard(symbol) for symbol in symbols]
    second = [risk_set._shard(symbol) for symbol in reversed(symbols)][::-1]
    assert first == second
    assert set(first) == set(range(risk_set.SHARDS))
