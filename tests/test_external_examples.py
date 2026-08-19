import csv
import io

import numpy as np
import pandas as pd

from market_analogues.adapters import OHLCVSource
from market_analogues.external_examples import (
    analyze_kullamagi_examples, normalize_setup, parse_kullamagi_positions,
)
from market_analogues.types import InstrumentKey


class _MemorySource(OHLCVSource):
    def __init__(self) -> None:
        self.frames = {}
        for number, symbol in enumerate(("AAA", "BBB", "CCC", "DDD"), 1):
            rng = np.random.default_rng(number)
            close = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, 190)))
            prior = np.r_[close[0], close[:-1]]
            frame = pd.DataFrame({
                "timestamp": pd.date_range("2020-01-01", periods=190, freq="B"),
                "open": prior,
                "high": np.maximum(prior, close) * 1.005,
                "low": np.minimum(prior, close) * 0.995,
                "close": close,
                "volume": rng.integers(10_000, 1_000_000, 190),
            })
            frame.attrs.update(symbol=symbol, dataset_id="test", interval="1d")
            self.frames[symbol] = frame

    def instruments(self) -> list[InstrumentKey]:
        return [InstrumentKey("test", symbol) for symbol in self.frames]

    def load(self, key: InstrumentKey) -> pd.DataFrame:
        return self.frames[key.source_symbol].copy()

    def fingerprint(self, key: InstrumentKey) -> str:
        return key.source_symbol


def _example_csv(source: _MemorySource) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Entry Date", "Exit Date", "Symbol", "Long/Short", "Setup",
        "Chart At Entry", "Chart After Exit", "Time", "Entry Price",
        "Stop Loss", "Exit Price", "Risk", "Reward", "% Move",
        "R-Multiple", "Result", "Notes",
    ])
    rows = [
        ("AAA", 140, "EP", "Win"),
        ("BBB", 150, "Episodic Pivot", "Loss"),
        ("CCC", 160, "Breakout", "Win"),
        ("DDD", 170, "Breakout", "Loss"),
        ("AAA", 180, "Breakout", "Loss"),
    ]
    for symbol, position, setup, result in rows:
        entry = source.frames[symbol].timestamp.iloc[position]
        writer.writerow([
            entry.strftime("%Y-%m-%d"), "", symbol, "Long", setup,
            "https://example.test/chart", "", "", "", "", "", "", "",
            "", "", result, "",
        ])
    return output.getvalue().encode()


def test_kullamagi_parser_normalizes_aliases_and_ignores_summary_rows() -> None:
    text = "\n".join([
        "Entry Date,Exit Date,Symbol,Long/Short,Setup,Chart At Entry,Chart After Exit,Time,Entry Price,Stop Loss,Exit Price,Risk,Reward,% Move,R-Multiple,Result,Notes",
        '"Monday, March 8, 2021",,ABCD,Long,EP,https://example.test/chart,,,,,,,,,,Win,note',
        '"Tuesday, March 9, 2021",,EFGH,Short,Parabolic Short,https://example.test/two,,,,,,,,,,Loss,note',
        "TOTAL CLOSED TRADES,2,,,,,,,,,,,,,,,,",
    ])
    records = parse_kullamagi_positions(text)
    assert len(records) == 2
    assert records[0].symbol == "ABCD"
    assert records[0].setup == "episodic_pivot"
    assert records[1].setup == "parabolic"
    assert records[1].side == "Short"


def test_setup_normalization_keeps_declared_groups_explicit() -> None:
    assert normalize_setup("Episodic Pivot") == "episodic_pivot"
    assert normalize_setup("EP - Sector") == "episodic_pivot"
    assert normalize_setup("Bounce off MA") == "moving_average_reaction"
    assert normalize_setup("Breakout") == "breakout"


def test_external_analysis_is_causal_cross_symbol_and_outcome_blind() -> None:
    source = _MemorySource()
    result = analyze_kullamagi_examples(
        source, _example_csv(source), "test-v1",
        source_url="https://example.test/sheet.csv", lookback=126, top_k=1,
        minimum_history_gap_bars=5, permutations=10, seed=7,
    )

    assert result.passed
    assert result.metrics["source_url"] == "https://example.test/sheet.csv"
    assert result.metrics["outcomes_used_in_similarity"] is False
    assert result.metrics["cutoff_policy"] == "previous_completed_session"
    assert set(result.coverage.status) == {"usable"}
    assert not result.neighbours.empty
    assert (result.neighbours.query_symbol != result.neighbours.candidate_symbol).all()
    assert "query_result" not in result.neighbours
    assert "candidate_result" not in result.neighbours
    causal = result.neighbours[result.neighbours["mode"] == "causal_5_sessions"]
    assert (
        pd.to_datetime(causal.candidate_cutoff)
        < pd.to_datetime(causal.query_cutoff)
    ).all()
    assert "top_k_setup_purity" in result.metrics["modes"]["retrospective"]
