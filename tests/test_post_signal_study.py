from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_analogues.post_signal_study import PostSignalStudyError, symbol_post_signal_panel
from market_analogues.types import stable_hash


ROOT = Path(__file__).resolve().parents[1]


def _fixture(rows: int = 340) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2020-01-01", periods=rows)
    close = np.full(rows, 100.); open_ = np.full(rows, 100.)
    high = np.full(rows, 101.); low = np.full(rows, 99.); volume = np.full(rows, 20_000.)
    signal = 252
    open_[signal] = 100.; close[signal] = 104.; high[signal] = 105.; low[signal] = 99.
    open_[signal + 1] = 110.; close[signal + 1] = 110.; high[signal + 1] = 112.; low[signal + 1] = 108.
    close[signal + 5] = 115.; high[signal + 5] = 116.; low[signal + 5] = 99.
    if rows > signal + 20:
        close[signal + 20] = 121.; high[signal + 20] = 122.; low[signal + 20] = 98.
    if rows > signal + 60:
        close[signal + 60] = 132.; high[signal + 60] = 134.; low[signal + 60] = 97.
    stock = pd.DataFrame({
        "timestamp": dates, "open": open_, "high": np.maximum.reduce((high, open_, close)),
        "low": np.minimum.reduce((low, open_, close)), "close": close, "volume": volume,
    })
    market_close = np.linspace(200., 230., rows); market_open = market_close.copy()
    market_open[signal + 1] = 210.
    if rows > signal + 20: market_close[signal + 20] = 220.
    benchmark = pd.DataFrame({
        "date": dates, "open": market_open, "high": np.maximum(market_open, market_close) + 1,
        "low": np.minimum(market_open, market_close) - 1, "close": market_close,
    })
    return stock, benchmark


def _row(panel: pd.DataFrame, position: int = 252) -> pd.Series:
    return panel.loc[panel.signal_position == position].iloc[0]


def test_contract_self_seal_and_strictly_post_signal_clock() -> None:
    contract = json.loads((ROOT / "config/m04r14-t14-12-post-signal-contract.json").read_text())
    digest = contract.pop("contract_digest")
    assert digest == stable_hash(contract)
    assert contract["execution_and_outcomes"]["entry_timestamp"] == "next_exact_benchmark_session_open"
    assert contract["causal_signal"]["future_availability_cannot_affect_signal_inclusion"]


def test_next_open_outcomes_exclude_signal_day_and_match_manual_values() -> None:
    stock, benchmark = _fixture(); row = _row(symbol_post_signal_panel(stock, benchmark, "SYN"))
    assert row.up_close_signal_event and row.bullish_range_expansion_signal_event
    assert row.entry_open_20 == 110.
    assert row.endpoint_close_return_20 == pytest.approx(121 / 110 - 1)
    market_gross = benchmark.iloc[272].close / benchmark.iloc[253].open
    assert row.benchmark_relative_log_return_20 == pytest.approx(np.log(121 / 110) - np.log(market_gross))
    assert row.maximum_favorable_excursion_20 == pytest.approx(122 / 110 - 1)
    assert row.maximum_adverse_excursion_20 == pytest.approx(98 / 110 - 1)
    assert row.status_20 == "complete"


def test_refractory_rule_is_causal_and_signal_specific() -> None:
    stock, benchmark = _fixture(); stock.loc[260, ["open", "close", "high", "low"]] = [100., 105., 106., 99.]
    panel = symbol_post_signal_panel(stock, benchmark, "SYN")
    first, second = _row(panel, 252), _row(panel, 260)
    assert first.up_close_signal_event
    assert second.up_close_4pct and not second.up_close_at_risk and not second.up_close_signal_event


def test_future_mutation_changes_outcome_not_causal_signal_or_matching_state() -> None:
    stock, benchmark = _fixture(); original = _row(symbol_post_signal_panel(stock, benchmark, "SYN"))
    changed = stock.copy(); changed.loc[272, ["open", "close", "high", "low"]] = [150., 150., 151., 149.]
    mutated = _row(symbol_post_signal_panel(changed, benchmark, "SYN"))
    causal = [
        "signal_date", "signal_position", "prior_close", "prior_return_63", "prior_volatility_20",
        "prior_median_dollar_volume_20", "signal_close", "signal_atr_20", "up_close_4pct",
        "up_close_at_risk", "up_close_signal_event", "bullish_range_expansion_4pct",
        "bullish_range_expansion_at_risk", "bullish_range_expansion_signal_event", "investable",
        "benchmark_signal_day_return", "benchmark_return_20", "benchmark_return_63", "benchmark_volatility_20",
    ]
    for column in causal: assert original[column] == mutated[column]
    assert original.endpoint_close_return_20 != mutated.endpoint_close_return_20


def test_missing_market_session_and_invalid_future_censor_without_removing_signal() -> None:
    stock, benchmark = _fixture(); missing = benchmark.drop(index=260).reset_index(drop=True)
    market_row = _row(symbol_post_signal_panel(stock, missing, "SYN"))
    assert market_row.up_close_signal_event and not market_row.complete_20
    assert market_row.status_20 == "missing_benchmark_session"
    invalid = stock.copy(); invalid.loc[260, "high"] = 1.
    invalid_row = _row(symbol_post_signal_panel(invalid, benchmark, "SYN"))
    assert invalid_row.up_close_signal_event and not invalid_row.complete_20
    assert invalid_row.status_20 == "invalid_future_ohlcv"


def test_signal_day_mutation_and_same_bar_barrier_ambiguity() -> None:
    stock, benchmark = _fixture()
    no_signal = stock.copy(); no_signal.loc[252, ["open", "high", "low", "close"]] = [100., 101., 99., 100.]
    row = _row(symbol_post_signal_panel(no_signal, benchmark, "SYN"))
    assert not row.up_close_signal_event and not row.bullish_range_expansion_signal_event
    ambiguous = stock.copy(); ambiguous.loc[253, ["open", "high", "low", "close"]] = [110., 125., 90., 110.]
    assert _row(symbol_post_signal_panel(ambiguous, benchmark, "SYN")).barrier_code_20 == 3


def test_incomplete_source_is_right_censored_and_bad_benchmark_rejected() -> None:
    stock, benchmark = _fixture(270); row = _row(symbol_post_signal_panel(stock, benchmark, "SYN"))
    assert not row.complete_20 and row.status_20 == "source_end_before_horizon"
    bad = benchmark.copy(); bad.loc[2, "date"] = bad.loc[1, "date"]
    with pytest.raises(PostSignalStudyError, match="benchmark"):
        symbol_post_signal_panel(stock, bad, "SYN")
