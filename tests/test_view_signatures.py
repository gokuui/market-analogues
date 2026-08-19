from __future__ import annotations

import numpy as np

from market_analogues.synthetic import generate_case, transform_case
from market_analogues.view_signatures import (
    SIGNATURE_DIMENSIONS, episode_view_signature, signature_view_distances,
    sliding_episode_signatures,
)


def test_view_signature_is_fixed_finite_and_unit_invariant() -> None:
    base = generate_case("trend_contraction_breakout", 3, n=126)
    transformed = transform_case(
        base, name="scaled", price_scale=7, volume_scale=11,
    )
    left = episode_view_signature(base.episode)
    right = episode_view_signature(transformed.episode)
    assert left.vector.shape == (SIGNATURE_DIMENSIONS,)
    assert np.isfinite(left.vector).all()
    distances = signature_view_distances(left, right.vector)
    assert max(float(value[0]) for value in distances.values()) < 1e-5


def test_batch_and_single_signature_distances_agree() -> None:
    query = episode_view_signature(generate_case("rounded_base", 2, n=126).episode)
    candidates = [
        episode_view_signature(generate_case("rounded_base", seed, n=126).episode)
        for seed in (3, 4, 5)
    ]
    matrix = np.vstack([candidate.vector for candidate in candidates])
    batch = signature_view_distances(query, matrix)
    for index, candidate in enumerate(candidates):
        single = signature_view_distances(query, candidate.vector)
        for view in batch:
            np.testing.assert_allclose(batch[view][index], single[view][0], rtol=1e-12)


def test_vectorized_sliding_signatures_match_episode_reference() -> None:
    case = generate_case("volatile_reversal", 7, n=180)
    bars = case.episode.bars
    positions, matrix = sliding_episode_signatures(
        bars, case.episode.benchmark, lookback=63, stride=11,
    )
    for row, position in enumerate(positions):
        episode = case.episode
        episode.bars = bars.iloc[position - 62:position + 1].reset_index(drop=True)
        reference = episode_view_signature(episode).vector
        np.testing.assert_allclose(matrix[row], reference, rtol=2e-5, atol=2e-5)


def test_float16_storage_preserves_view_distances_and_top_candidates() -> None:
    query = episode_view_signature(generate_case("steady_trend", 1, n=126).episode)
    candidates = np.vstack([
        episode_view_signature(generate_case(family, seed, n=126).episode).vector
        for family in (
            "steady_trend", "rounded_base", "volatile_reversal",
            "failed_breakout", "trend_contraction_breakout",
        )
        for seed in range(1, 7)
    ])
    full = signature_view_distances(query, candidates)
    quantized = signature_view_distances(query, candidates.astype(np.float16).astype(np.float32))
    for view in full:
        np.testing.assert_allclose(quantized[view], full[view], rtol=5e-3, atol=5e-3)
        assert set(np.argsort(full[view])[:10]) == set(np.argsort(quantized[view])[:10])
