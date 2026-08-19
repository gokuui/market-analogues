from time import perf_counter

import numpy as np
import pandas as pd
import pytest

from market_analogues.distance import representation_distance_lower_bound
from market_analogues.exact_batch import (
    batch_representation_lower_bounds, sliding_exact_representations,
)
from market_analogues.representation import Representation, represent
from market_analogues.synthetic import generate_case
from market_analogues.types import Episode, EpisodeKey, InstrumentKey


def _reference(
    bars: pd.DataFrame,
    benchmark: pd.DataFrame | None,
    position: int,
    lookback: int,
) -> Representation:
    window = bars.iloc[position - lookback + 1:position + 1].reset_index(drop=True)
    cutoff = pd.Timestamp(window.timestamp.iloc[-1])
    context = (
        benchmark[benchmark.timestamp <= cutoff].copy()
        if benchmark is not None else None
    )
    episode = Episode(
        EpisodeKey(InstrumentKey("test", "X"), cutoff, lookback, "dense-v1"),
        window, context,
    )
    return represent(episode)


def _assert_representation_equal(
    actual: Representation,
    expected: Representation,
) -> None:
    np.testing.assert_allclose(actual.coarse, expected.coarse, rtol=0, atol=0)
    np.testing.assert_allclose(actual.stage, expected.stage, rtol=0, atol=1e-12)
    np.testing.assert_allclose(actual.structural, expected.structural, rtol=0, atol=1e-12)
    for collection in ("samples_48", "samples_64"):
        left = getattr(actual, collection)
        right = getattr(expected, collection)
        assert left.keys() == right.keys()
        for name in left:
            if left[name] is None or right[name] is None:
                assert left[name] is right[name]
            else:
                np.testing.assert_allclose(left[name], right[name], rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    ("benchmark_mode", "lookback"),
    [("full", 126), ("partial", 126), ("missing", 126), ("missing", 63)],
)
def test_sliding_exact_representations_and_bounds_match_scalar(
    benchmark_mode: str,
    lookback: int,
) -> None:
    case = generate_case("volatile_reversal", 7, n=300)
    benchmark = case.episode.benchmark if benchmark_mode != "missing" else None
    if benchmark_mode == "partial":
        benchmark = benchmark.copy()
        benchmark.loc[benchmark.index[::7], "close"] = np.nan
    batch = sliding_exact_representations(
        case.episode.bars, benchmark, lookback=lookback, stride=29,
    )
    references = tuple(
        _reference(case.episode.bars, benchmark, int(position), lookback)
        for position in batch.positions
    )
    for actual, expected in zip(batch.representations, references):
        _assert_representation_equal(actual, expected)

    query = references[-1]
    vectorized = batch_representation_lower_bounds(query, batch.representations[:-1])
    for row, candidate in enumerate(batch.representations[:-1]):
        total, components, rigid = representation_distance_lower_bound(query, candidate)
        assert vectorized.totals[row] == pytest.approx(total, abs=1e-12)
        assert vectorized.rigid_price[row] == pytest.approx(rigid, abs=1e-12)
        for name, value in components.items():
            assert vectorized.components[name][row] == pytest.approx(value, abs=1e-12)


def test_exact_batch_handles_constant_zero_volume_and_chunk_identity() -> None:
    case = generate_case("steady_trend", 4, n=280)
    bars = case.episode.bars.copy()
    bars[["open", "high", "low", "close"]] = 100.0
    bars["volume"] = 0.0
    whole = sliding_exact_representations(
        bars, None, lookback=126, stride=7,
    )
    chunked = sliding_exact_representations(
        bars, None, lookback=126, stride=7, batch_size=4,
    )
    np.testing.assert_array_equal(whole.positions, chunked.positions)
    assert len(whole.representations) == len(chunked.representations)
    for actual, expected in zip(chunked.representations, whole.representations):
        _assert_representation_equal(actual, expected)
    for row, position in enumerate(whole.positions):
        _assert_representation_equal(
            whole.representations[row], _reference(bars, None, int(position), 126),
        )


def test_exact_batch_ignores_future_benchmark_mutation() -> None:
    case = generate_case("rounded_base", 11, n=320)
    cutoff_position = 240
    original = sliding_exact_representations(
        case.episode.bars.iloc[:cutoff_position + 1],
        case.episode.benchmark, lookback=126, stride=13,
    )
    changed_benchmark = case.episode.benchmark.copy()
    changed_benchmark.loc[
        changed_benchmark.timestamp > case.episode.bars.timestamp.iloc[cutoff_position],
        "close",
    ] *= 100
    changed = sliding_exact_representations(
        case.episode.bars.iloc[:cutoff_position + 1],
        changed_benchmark, lookback=126, stride=13,
    )
    for actual, expected in zip(changed.representations, original.representations):
        _assert_representation_equal(actual, expected)


def test_exact_batch_is_materially_faster_than_scalar_reference() -> None:
    case = generate_case("trend_contraction_breakout", 13, n=520)
    started = perf_counter()
    batch = sliding_exact_representations(
        case.episode.bars, case.episode.benchmark, lookback=126, stride=5,
    )
    batch_seconds = perf_counter() - started
    started = perf_counter()
    references = [
        _reference(
            case.episode.bars, case.episode.benchmark, int(position), 126,
        )
        for position in batch.positions
    ]
    scalar_seconds = perf_counter() - started
    assert len(references) == len(batch.representations)
    assert scalar_seconds / batch_seconds >= 5


@pytest.mark.parametrize("keyword", ["lookback", "stride", "batch_size"])
def test_exact_batch_rejects_invalid_controls(keyword: str) -> None:
    case = generate_case("steady_trend", 1, n=126)
    arguments = {"lookback": 63, "stride": 5, "batch_size": 10}
    arguments[keyword] = 0
    with pytest.raises(ValueError):
        sliding_exact_representations(
            case.episode.bars, case.episode.benchmark, **arguments,
        )
