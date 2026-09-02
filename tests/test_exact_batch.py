from dataclasses import replace
from time import perf_counter

import numpy as np
import pandas as pd
import pytest

from market_analogues.distance import (
    representation_distance, representation_distance_lower_bound,
)
from market_analogues.exact_batch import (
    _structural_rows, _structural_rows_reference, batch_exact_price_distances,
    batch_exact_rigid_price_distances, batch_representation_lower_bounds,
    exact_price_representations_at_positions, exact_representations_at_positions,
    sliding_exact_representations,
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


def test_compiled_structural_rows_exactly_match_python_reference() -> None:
    rng = np.random.default_rng(20260825)
    paths = [
        rng.normal(0, .02, size=(64, 252)).cumsum(axis=1),
        np.zeros((8, 252)),
        np.full((8, 252), np.nan),
    ]
    for threshold in (.03, .06, .12):
        boundary = np.zeros((8, 252))
        boundary[:, 1] = np.log(1 - threshold)
        boundary[:, 2] = np.log((1 - threshold) * (1 + threshold))
        paths.append(boundary)
    for path in paths:
        np.testing.assert_array_equal(
            _structural_rows(path), _structural_rows_reference(path),
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


def test_batched_exact_price_matches_scalar_across_channel_shapes() -> None:
    query = represent(generate_case("trend_contraction_breakout", 991).episode)
    candidates = [
        represent(generate_case(family, 1_000 + index).episode)
        for index, family in enumerate((
            "steady_trend", "rounded_base", "failed_breakout",
            "volatile_reversal", "trend_contraction_breakout",
        ))
    ]
    candidates.append(replace(
        candidates[0],
        samples_64={**candidates[0].samples_64, "relative_path": None},
    ))
    rigid = np.asarray([
        representation_distance_lower_bound(query, candidate)[2]
        for candidate in candidates
    ])
    actual = batch_exact_price_distances(query, candidates, rigid)
    expected = np.asarray([
        representation_distance(query, candidate)[1]["price"]
        for candidate in candidates
    ])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-15)


def test_batched_exact_price_handles_absent_channels_and_empty_batch() -> None:
    original = represent(generate_case("steady_trend", 1_050).episode)
    absent = replace(
        original,
        samples_64={name: None for name in original.samples_64},
    )
    actual = batch_exact_price_distances(absent, [absent], np.asarray([0.0]))
    np.testing.assert_array_equal(actual, np.asarray([0.0]))
    assert batch_exact_price_distances(
        absent, [], np.empty(0, dtype=np.float64),
    ).shape == (0,)


def test_specialized_exact_price_projection_is_exactly_equal() -> None:
    case = generate_case("volatile_reversal", 1_080, n=420)
    positions = np.asarray([251, 279, 331, 400])
    complete = exact_representations_at_positions(
        case.episode.bars, case.episode.benchmark,
        positions=positions, lookback=252,
    )
    specialized = exact_price_representations_at_positions(
        case.episode.bars, case.episode.benchmark,
        positions=positions, lookback=252,
    )
    query = complete[-1]
    for full, price in zip(complete, specialized, strict=True):
        for name, values in price.samples_48.items():
            np.testing.assert_array_equal(values, full.samples_48[name])
        for name, values in price.samples_64.items():
            np.testing.assert_array_equal(values, full.samples_64[name])
    expected_rigid = batch_representation_lower_bounds(
        query, list(complete[:-1]),
    ).rigid_price
    actual_rigid = batch_exact_rigid_price_distances(
        query, list(specialized[:-1]),
    )
    np.testing.assert_array_equal(actual_rigid, expected_rigid)
    np.testing.assert_array_equal(
        batch_exact_price_distances(query, list(specialized[:-1]), actual_rigid),
        np.asarray([
            representation_distance(query, candidate)[1]["price"]
            for candidate in complete[:-1]
        ]),
    )


@pytest.mark.parametrize("benchmark_mode", ["full", "partial", "missing"])
def test_requested_position_batch_matches_scalar_and_sliding(
    benchmark_mode: str,
) -> None:
    case = generate_case("volatile_reversal", 17, n=420)
    benchmark = case.episode.benchmark if benchmark_mode != "missing" else None
    if benchmark_mode == "partial":
        benchmark = benchmark.copy()
        benchmark.loc[benchmark.index[::11], "close"] = np.nan
    sliding = sliding_exact_representations(
        case.episode.bars, benchmark, lookback=126, stride=5,
    )
    selected_indices = np.asarray([17, 2, 41, 8, 29])
    positions = sliding.positions[selected_indices]
    requested = exact_representations_at_positions(
        case.episode.bars, benchmark, positions=positions, lookback=126,
    )
    assert len(requested) == len(positions)
    for actual, index, position in zip(requested, selected_indices, positions):
        _assert_representation_equal(actual, sliding.representations[int(index)])
        _assert_representation_equal(
            actual,
            _reference(case.episode.bars, benchmark, int(position), 126),
        )


@pytest.mark.parametrize(
    "positions",
    [np.asarray([[62]]), np.asarray([62, 62]), np.asarray([61]), np.asarray([999])],
)
def test_requested_position_batch_rejects_invalid_positions(
    positions: np.ndarray,
) -> None:
    case = generate_case("steady_trend", 19, n=180)
    with pytest.raises(ValueError):
        exact_representations_at_positions(
            case.episode.bars, case.episode.benchmark,
            positions=positions, lookback=63,
        )


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
