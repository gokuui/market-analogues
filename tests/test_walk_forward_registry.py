from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from market_analogues.types import EpisodeKey, InstrumentKey
from market_analogues.walk_forward_registry import (
    canonical_month_cutoffs,
    measure_symbol_prefixes,
    selection_digest,
    stratify_and_select,
)


def _bars(dates: pd.DatetimeIndex, scale: float = 1.0) -> pd.DataFrame:
    close = scale * (20 + np.arange(len(dates), dtype=float) / 100)
    return pd.DataFrame({
        "date": dates, "open": close - .1, "high": close + .2,
        "low": close - .2, "close": close,
        "volume": np.arange(len(dates), dtype=float) + 1_000,
    })


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_month_cutoffs_are_last_observed_session() -> None:
    dates = pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-27", "2020-02-28"])
    benchmark = pd.DataFrame({"timestamp": dates})
    assert canonical_month_cutoffs(benchmark, "2020-01-01", "2020-02-29") == (
        pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-28"),
    )


def test_prefix_measurement_is_invariant_to_future_append(tmp_path: Path) -> None:
    dates = pd.bdate_range("2019-01-01", periods=310)
    cutoff = dates[280]
    locked_last = dates[-1]
    path = tmp_path / "TEST.parquet"
    original = _bars(dates)
    original.to_parquet(path, index=False)
    lock = {
        "symbol": "TEST", "rows_at_lock": len(original),
        "first_timestamp_at_lock": dates[0].isoformat(),
        "last_timestamp_at_lock": locked_last.isoformat(),
        "source_hash_at_lock": _sha(path),
    }
    before, before_source = measure_symbol_prefixes(
        "TEST", path, lock, [cutoff], locked_last,
    )
    future_dates = pd.bdate_range(locked_last + pd.Timedelta(days=1), periods=5)
    appended = pd.concat([original, _bars(future_dates, scale=10)], ignore_index=True)
    appended.to_parquet(path, index=False)
    after, after_source = measure_symbol_prefixes(
        "TEST", path, lock, [cutoff], locked_last,
    )
    assert before == after
    assert before_source["unchanged_file_raw_hash_matches_lock"] is True
    assert after_source["unchanged_file_raw_hash_matches_lock"] is None
    assert after_source["timestamp_identity_through_lock"] is True


def test_quality_is_causal_and_uses_existing_a_b_rules(tmp_path: Path) -> None:
    dates = pd.bdate_range("2019-01-01", periods=300)
    cutoff = dates[-1]
    frame = _bars(dates)
    frame.loc[100, "close"] *= 4
    frame.loc[100, "high"] = frame.loc[100, "close"] + .2
    path = tmp_path / "JUMP.parquet"
    frame.to_parquet(path, index=False)
    lock = {
        "symbol": "JUMP", "rows_at_lock": len(frame),
        "first_timestamp_at_lock": dates[0].isoformat(),
        "last_timestamp_at_lock": dates[-1].isoformat(),
        "source_hash_at_lock": _sha(path),
    }
    rows, source = measure_symbol_prefixes("JUMP", path, lock, [cutoff], cutoff)
    assert source["error"] is None
    assert rows[0]["quality_tier"] == "B"
    assert rows[0]["extreme_discontinuities_at_cutoff"] == 2


def test_liquidity_terciles_and_hash_selection_are_deterministic() -> None:
    cutoff = pd.Timestamp("2020-01-31")
    rows = []
    for quality in ("A", "B"):
        for index in range(15):
            rows.append({
                "cutoff": cutoff.isoformat(), "symbol": f"{quality}{index:02d}",
                "quality_tier": quality, "rows_at_cutoff": 300,
                "zero_volume_at_cutoff": 0, "extreme_discontinuities_at_cutoff": 0,
                "median_dollar_volume_252": float(index * 2 + (0 if quality == "A" else 1)),
            })
    candidates = pd.DataFrame(rows)
    folds = [{
        "fold_id": "development", "role": "diagnostic",
        "start": "2020-01-01", "end": "2020-12-31",
    }]
    enriched, queries, accounting = stratify_and_select(
        candidates, [cutoff], contract_digest="d" * 64, folds=folds,
        target_per_cell=4, lookback=252, representation_version="dense-v1",
    )
    assert enriched.groupby("liquidity_stratum").size().to_dict() == {
        "high": 10, "low": 10, "middle": 10,
    }
    assert len(queries) == 24
    assert accounting.shortfall.sum() == 0
    assert queries.case_id.is_unique and queries.episode_id.is_unique
    first = queries.iloc[0]
    expected_hash = selection_digest(
        "d" * 64, "nasdaq", cutoff.isoformat(), first.quality_tier,
        first.liquidity_stratum, first.symbol,
    )
    assert first.selection_hash == expected_hash
    expected_episode = EpisodeKey(
        InstrumentKey("nasdaq", first.symbol), cutoff, 252, "dense-v1",
    ).id
    assert first.episode_id == expected_episode


def test_underfilled_cell_is_not_backfilled() -> None:
    cutoff = pd.Timestamp("2020-01-31")
    candidates = pd.DataFrame([{
        "cutoff": cutoff.isoformat(), "symbol": f"A{index}", "quality_tier": "A",
        "rows_at_cutoff": 300, "zero_volume_at_cutoff": 0,
        "extreme_discontinuities_at_cutoff": 0,
        "median_dollar_volume_252": float(index),
    } for index in range(9)])
    _, queries, accounting = stratify_and_select(
        candidates, [cutoff], contract_digest="e" * 64, folds=[],
        target_per_cell=4, lookback=252, representation_version="dense-v1",
    )
    assert len(queries) == 9
    assert int(accounting.shortfall.sum()) == 15
    assert not (queries.quality_tier == "B").any()
