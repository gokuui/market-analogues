from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from market_analogues.packed_bound_store import (
    OVERFLOW_DTYPE,
    PACK_DTYPE,
    TIER_CODES,
)
from market_analogues.types import stable_hash

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import m04r14_t14_10_wf03_baseline_poc as subject
from experiments.m04r import verify_m04r14_t14_10_wf03_baseline_poc as verifier
from market_analogues.baseline_feature_store import features_for_packed_records
from market_analogues.packed_bound_search import _eligible_mask


def _records(dtype: np.dtype, rows: list[tuple[int, int]]) -> np.ndarray:
    values = np.zeros(len(rows), dtype=dtype)
    for ordinal, (symbol_id, cutoff) in enumerate(rows):
        values[ordinal]["episode_id"] = np.void((ordinal + 1).to_bytes(12, "big"))
        values[ordinal]["symbol_id"] = symbol_id
        values[ordinal]["cutoff_ns"] = cutoff
        values[ordinal]["quality_tier"] = TIER_CODES["A"]
    return values


def test_slice_retains_physical_symbol_order() -> None:
    rows = _records(PACK_DTYPE, [(0, 1), (0, 2), (2, 1), (3, 1)])
    assert subject._slice(rows, 0)["cutoff_ns"].tolist() == [1, 2]
    assert len(subject._slice(rows, 1)) == 0
    assert subject._slice(rows, 3)["cutoff_ns"].tolist() == [1]


def test_selection_is_deterministic_and_earliest_eligible() -> None:
    symbols = tuple(f"S{value:03d}" for value in range(30))
    rows = _records(PACK_DTYPE, [(value, 5) for value in range(30)])
    overflow = _records(OVERFLOW_DTYPE, [(0, 4), (1, 4)])
    prefixes = {symbol: {"digest": stable_hash(symbol)} for symbol in symbols}
    loaded = SimpleNamespace(
        rows=rows, overflow=overflow, symbols=symbols,
        manifest={"provenance": {"source_prefixes": prefixes}},
    )
    query = SimpleNamespace(
        episode_id=f"{999:024x}", query_start_ns=9, latest_eligible_ns=5,
        quality_tiers=("A", "B"), symbol="S029",
    )
    first = subject.select_symbols(loaded, query, count=20)
    second = subject.select_symbols(loaded, query, count=20)
    assert first == second
    assert [row["symbol_id"] for row in first] == sorted(
        row["symbol_id"] for row in first
    )
    assert {0, 1}.issubset(row["symbol_id"] for row in first)
    assert all(row["rows"] == 1 for row in first)


def test_independent_features_and_eligibility_match_producer_paths() -> None:
    import pandas as pd

    timestamps = pd.date_range("2020-01-01", periods=80, freq="D")
    frame = pd.DataFrame({
        "timestamp": timestamps,
        "close": np.exp(np.arange(80, dtype=float) * .01),
    })
    rows = _records(PACK_DTYPE, [(0, int(timestamps[63].value)),
                                 (1, int(timestamps[79].value))])
    observed = features_for_packed_records(frame, rows)["values"]
    expected = verifier._independent_feature_rows(frame, rows)
    np.testing.assert_allclose(observed, expected, rtol=0, atol=0)

    query = SimpleNamespace(
        episode_id=f"{999:024x}", query_start_ns=int(timestamps[70].value),
        latest_eligible_ns=int(timestamps[75].value), quality_tiers=("A", "B"),
    )
    np.testing.assert_equal(
        verifier._independent_eligible(rows, query, 0),
        _eligible_mask(rows, query, 0),
    )
