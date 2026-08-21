from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_analogues.multiresolution import (
    CHANNEL_FIELD_OBSERVATIONS,
    REQUIRED_HORIZONS,
    SAMPLE_POINTS,
    SUMMARY_FIELD_OBSERVATIONS,
    build_multiresolution_state,
    field_contract,
    state_manifest,
)
from market_analogues.types import Episode, EpisodeKey, InstrumentKey


def _episode(
    bars: pd.DataFrame,
    *,
    cutoff_position: int = 251,
    benchmark: pd.DataFrame | None = None,
) -> Episode:
    canonical = bars.rename(columns={"date": "timestamp"}).reset_index(drop=True)
    cutoff = pd.Timestamp(canonical.timestamp.iloc[cutoff_position])
    key = EpisodeKey(InstrumentKey("test", "AAA"), cutoff, 252, "dense-v1")
    return Episode(key, canonical, benchmark)


def test_state_contains_complete_six_resolution_contract(bars: pd.DataFrame) -> None:
    state = build_multiresolution_state(_episode(bars))
    assert tuple(state.views) == REQUIRED_HORIZONS
    assert len(state.channels) == 252
    contract = field_contract()
    channel_names = {field.name for field in CHANNEL_FIELD_OBSERVATIONS}
    summary_names = {field.name for field in SUMMARY_FIELD_OBSERVATIONS}
    assert set(state.channels) - {"timestamp"} == channel_names
    assert channel_names | summary_names == set(contract)
    assert all(not spec["future_inputs_allowed"] for spec in contract.values())
    assert all(spec["available_by_primary_decision"] for spec in contract.values())
    for horizon, view in state.views.items():
        assert view.observed_sessions == horizon
        assert len(view.samples) == len(channel_names)
        assert set(view.summary) == summary_names
        assert all(sample.values.shape == (SAMPLE_POINTS,) for sample in view.samples.values())
        assert all(sample.observed.shape == (SAMPLE_POINTS,) for sample in view.samples.values())


def test_missing_context_is_masked_and_never_encoded_as_observed_zero(bars: pd.DataFrame) -> None:
    state = build_multiresolution_state(_episode(bars))
    context = {
        "benchmark_close", "benchmark_return", "benchmark_path",
        "benchmark_volatility", "benchmark_drawdown", "relative_return", "relative_path",
    }
    for view in state.views.values():
        assert view.summary["benchmark_context_fraction"] == 0
        assert view.summary["benchmark_log_return"] is None
        assert view.summary["relative_log_return"] is None
        for name in context:
            sample = view.samples[name]
            assert not sample.observed.any()
            assert np.count_nonzero(sample.values) == 0


def test_future_stock_and_benchmark_mutation_has_exactly_zero_effect(bars: pd.DataFrame) -> None:
    canonical_benchmark = bars.rename(columns={"date": "timestamp"})[["timestamp", "close"]].copy()
    original = _episode(bars, cutoff_position=251, benchmark=canonical_benchmark)
    mutated_bars = bars.copy()
    mutated_bars.loc[252:, ["open", "high", "low", "close", "volume"]] *= 100
    mutated_benchmark = canonical_benchmark.copy()
    mutated_benchmark.loc[252:, "close"] *= 100
    mutated = _episode(mutated_bars, cutoff_position=251, benchmark=mutated_benchmark)
    before = build_multiresolution_state(original)
    after = build_multiresolution_state(mutated)
    assert before.state_digest == after.state_digest
    pd.testing.assert_frame_equal(before.channels, after.channels)


def test_price_and_volume_units_do_not_change_multiresolution_state(bars: pd.DataFrame) -> None:
    base_episode = _episode(bars)
    scaled = bars.copy()
    scaled[["open", "high", "low", "close"]] *= 17.5
    scaled["volume"] *= 100
    scaled_episode = _episode(scaled)
    base = build_multiresolution_state(base_episode)
    other = build_multiresolution_state(scaled_episode)
    np.testing.assert_allclose(
        base.channels.select_dtypes("number"),
        other.channels.select_dtypes("number"),
        atol=1e-10,
        equal_nan=True,
    )
    for horizon in REQUIRED_HORIZONS:
        left, right = base.views[horizon], other.views[horizon]
        assert left.summary.keys() == right.summary.keys()
        for name in left.summary:
            if left.summary[name] is None or right.summary[name] is None:
                assert left.summary[name] is right.summary[name]
            else:
                assert left.summary[name] == pytest.approx(right.summary[name], abs=1e-10)
        for name in left.samples:
            np.testing.assert_array_equal(left.samples[name].observed, right.samples[name].observed)
            np.testing.assert_allclose(left.samples[name].values, right.samples[name].values, atol=1e-6)


def test_constant_price_and_zero_volume_have_no_infinite_state(bars: pd.DataFrame) -> None:
    constant = bars.copy()
    constant[["open", "high", "low", "close"]] = 100.0
    constant["volume"] = 0
    state = build_multiresolution_state(_episode(constant))
    assert not np.isinf(state.channels.select_dtypes("number").to_numpy()).any()
    for view in state.views.values():
        for sample in view.samples.values():
            assert np.isfinite(sample.values).all()
        assert all(value is None or np.isfinite(value) for value in view.summary.values())


@pytest.mark.parametrize("problem", ["duplicate", "unsorted"])
def test_invalid_session_order_is_rejected(bars: pd.DataFrame, problem: str) -> None:
    broken = bars.copy()
    if problem == "duplicate":
        broken.loc[100, "date"] = broken.loc[99, "date"]
    else:
        left = broken.loc[99, "date"]
        broken.loc[99, "date"] = broken.loc[100, "date"]
        broken.loc[100, "date"] = left
    with pytest.raises(ValueError, match="duplicate|sorted"):
        build_multiresolution_state(_episode(broken))


def test_timezone_aware_sessions_preserve_cutoff_boundary(bars: pd.DataFrame) -> None:
    aware = bars.copy()
    aware["date"] = pd.to_datetime(aware["date"]).dt.tz_localize("America/New_York")
    state = build_multiresolution_state(_episode(aware))
    expected = pd.Timestamp(aware.date.iloc[251]).isoformat()
    assert state.cutoff == expected
    assert state.views[252].end_timestamp == expected
    assert all(pd.Timestamp(view.end_timestamp) <= pd.Timestamp(state.cutoff) for view in state.views.values())


def test_corporate_action_warning_is_preserved_not_silently_corrected(bars: pd.DataFrame) -> None:
    episode = _episode(bars)
    warned = Episode(
        episode.key, episode.bars, episode.benchmark, "B",
        ("extreme_discontinuity:1", "corporate_action_provenance:unknown"),
    )
    state = build_multiresolution_state(warned)
    manifest = state_manifest(state)
    assert state.quality_tier == "B"
    assert manifest["quality_issues"] == [
        "extreme_discontinuity:1", "corporate_action_provenance:unknown",
    ]
