from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from market_analogues.external_examples import ExternalExampleResult, LabelledExample
from market_analogues.yahoo_examples import (
    YahooFetchResult, fetch_yahoo_examples, write_external_example_workbook,
    yahoo_symbol,
)


def _fake_download(tickers, **kwargs) -> pd.DataFrame:
    symbols = [tickers] if isinstance(tickers, str) else list(tickers)
    index = pd.date_range("2019-01-01", periods=400, freq="B", name="Date")
    frames = {}
    for number, symbol in enumerate(symbols, 1):
        close = 100 + number + np.arange(len(index)) * 0.1
        frames[symbol] = pd.DataFrame({
            "Open": close - 0.1, "High": close + 0.5,
            "Low": close - 0.5, "Close": close,
            "Volume": np.full(len(index), 100_000 + number),
        }, index=index)
    return pd.concat(frames, axis=1)


def _record(row: int, symbol: str, setup: str) -> LabelledExample:
    return LabelledExample(
        row, pd.Timestamp("2020-07-01"), symbol, "Long", setup, setup,
        "Win", f"https://example.test/{symbol}",
    )


def test_yahoo_fetch_maps_symbols_caches_bars_and_benchmark(tmp_path: Path) -> None:
    records = (_record(1, "BRK.B", "breakout"), _record(2, "AAA", "breakout"))
    result = fetch_yahoo_examples(
        records, tmp_path / "cache", start=pd.Timestamp("2019-01-01"),
        end=pd.Timestamp("2021-01-01"), downloader=_fake_download,
    )

    assert yahoo_symbol("BRK.B") == "BRK-B"
    assert result.manifest["requested_symbols"] == 2
    assert result.manifest["available_symbols"] == 2
    assert result.source.load_benchmark() is not None
    assert len(result.source.instruments()) == 2
    assert len(result.source.load(result.source.instruments()[0])) == 400
    assert set(result.coverage.status) == {"downloaded"}


def test_excel_workbook_contains_copyable_analysis_sheets(tmp_path: Path) -> None:
    records = (_record(1, "AAA", "breakout"), _record(2, "BBB", "breakout"))
    fetch = fetch_yahoo_examples(
        records, tmp_path / "cache", start=pd.Timestamp("2019-01-01"),
        end=pd.Timestamp("2021-01-01"), downloader=_fake_download,
    )
    mode = {
        "queries": 1, "top_k": 1, "top1_setup_agreement": 1.0,
        "top_k_setup_purity": 1.0, "top1_side_agreement": 1.0,
        "candidate_frequency_expected_agreement": 0.5,
        "majority_setup_accuracy": 1.0,
        "macro_top1_agreement_minimum_3_queries": 0.0,
        "random_candidate_top1_p_value": 0.1,
        "random_candidate_top_k_p_value": 0.1,
        "per_setup": {
            "breakout": {"queries": 1, "top1_agreement": 1.0, "top_k_purity": 1.0},
        },
    }
    neighbours = pd.DataFrame([{
        "mode": "causal_5_sessions", "query_index": 0,
        "candidate_index": 1, "rank": 1, "query_symbol": "AAA",
        "query_entry_date": "2020-07-01", "query_cutoff": "2020-06-30",
        "query_setup": "breakout", "query_side": "Long",
        "query_chart_url": "https://example.test/AAA",
        "candidate_symbol": "BBB", "candidate_entry_date": "2020-01-01",
        "candidate_cutoff": "2019-12-31", "candidate_setup": "breakout",
        "candidate_side": "Long",
        "candidate_chart_url": "https://example.test/BBB",
        "same_setup": True, "same_side": True, "total_distance": 0.2,
    }])
    result = ExternalExampleResult({
        "source_rows": 2, "usable_unique_episodes": 2,
        "usable_unique_symbols": 2, "minimum_history_gap_bars": 5,
        "outcomes_used_in_similarity": False,
        "source_url": "https://example.test/sheet",
        "analysis_input_sha256": "abc", "modes": {
            "retrospective": mode, "causal_5_sessions": mode,
        },
    }, neighbours, pd.DataFrame([{"symbol": "AAA", "status": "usable"}]), True, ())
    workbook_path = write_external_example_workbook(
        result, YahooFetchResult(
            fetch.source, fetch.manifest, fetch.manifest_path, fetch.coverage,
        ), tmp_path / "analysis.xlsx", target_purity=0.75,
    )

    workbook = openpyxl.load_workbook(workbook_path, read_only=True)
    assert {
        "Summary", "Query Summary", "Top1 Matches", "Causal TopK",
        "Setup Performance", "Trade Coverage", "Yahoo Coverage",
    } == set(workbook.sheetnames)
    assert workbook["Summary"]["A2"].value == "Analysis completed"
