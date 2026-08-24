from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from market_analogues.distance import representation_distance
from market_analogues.exact_aligned_features import (
    EXACT_FEATURE_DIMENSIONS, episode_exact_features, exact_feature_contract,
    exact_features_to_representation, representation_to_exact_features,
    sliding_exact_features,
)
from market_analogues.representation import Representation, represent
from market_analogues.synthetic import FAMILIES, generate_case, transform_case
from market_analogues.types import Episode


def _assert_representation_equal(left: Representation, right: Representation) -> None:
    np.testing.assert_array_equal(left.coarse, right.coarse)
    np.testing.assert_allclose(left.stage, right.stage, rtol=0, atol=0)
    np.testing.assert_allclose(left.structural, right.structural, rtol=0, atol=0)
    for collection_name in ("samples_48", "samples_64"):
        left_collection = getattr(left, collection_name)
        right_collection = getattr(right, collection_name)
        assert left_collection.keys() == right_collection.keys()
        for name in left_collection:
            if left_collection[name] is None or right_collection[name] is None:
                assert left_collection[name] is right_collection[name]
            else:
                np.testing.assert_allclose(
                    left_collection[name], right_collection[name], rtol=0, atol=0,
                )


def test_exact_feature_contract_is_frozen_and_explicitly_not_legacy_proxy() -> None:
    contract = exact_feature_contract()
    assert contract["dimensions"] == EXACT_FEATURE_DIMENSIONS == 1376
    assert contract["lossless_for_distance_v1_fields"] is True
    assert contract["legacy_view_signature_equivalent"] is False
    assert contract["outcomes_or_labels_used"] is False
    assert contract["digest"] == "3769b076a4fb234c71407638fdaaa0f8d325d8525182d6cba5d58be636c0a8ea"


def test_exact_feature_round_trip_preserves_all_values_and_masks() -> None:
    case = generate_case("rounded_base", 201)
    representation = represent(case.episode)
    features = representation_to_exact_features(representation)
    rebuilt = exact_features_to_representation(features)
    assert np.isfinite(features.vector).all()
    _assert_representation_equal(representation, rebuilt)
    assert representation_distance(representation, rebuilt)[0] == 0.0


def test_scalar_and_sliding_features_are_identical_for_every_family() -> None:
    for seed, family in enumerate(sorted(FAMILIES), 211):
        case = generate_case(family, seed, n=310)
        positions, matrix = sliding_exact_features(
            case.episode.bars, case.episode.benchmark, lookback=126, stride=37,
        )
        for row, position in enumerate(positions):
            bars = case.episode.bars.iloc[position - 125:position + 1].reset_index(drop=True)
            cutoff = pd.Timestamp(bars.timestamp.iloc[-1])
            benchmark = case.episode.benchmark[
                case.episode.benchmark.timestamp <= cutoff
            ].copy()
            scalar = episode_exact_features(Episode(
                case.episode.key, bars, benchmark,
                case.episode.quality_tier, case.episode.quality_issues,
            ))
            np.testing.assert_array_equal(matrix[row], scalar.vector)


def test_missing_constant_zero_volume_and_chunked_identity() -> None:
    case = generate_case("steady_trend", 221, n=280)
    bars = case.episode.bars.copy()
    bars[["open", "high", "low", "close"]] = 100.0
    bars["volume"] = 0.0
    positions, whole = sliding_exact_features(
        bars, None, lookback=126, stride=7,
    )
    chunk_positions, chunked = sliding_exact_features(
        bars, None, lookback=126, stride=7, batch_size=4,
    )
    np.testing.assert_array_equal(positions, chunk_positions)
    np.testing.assert_array_equal(whole, chunked)
    assert np.isfinite(whole).all()


def test_price_volume_unit_invariance_and_future_mutation() -> None:
    case = generate_case("trend_contraction_breakout", 231, n=300)
    scaled = transform_case(
        case, name="scaled", price_scale=17.0, volume_scale=31.0,
    )
    left = represent(case.episode)
    right = represent(scaled.episode)
    assert representation_distance(left, right)[0] < 1e-6

    cutoff = 240
    cutoff_timestamp = case.episode.bars.timestamp.iloc[cutoff]
    positions, before = sliding_exact_features(
        case.episode.bars, case.episode.benchmark, lookback=126, stride=11,
    )
    changed_bars = case.episode.bars.copy()
    changed_bars.loc[changed_bars.index > cutoff, "close"] *= 100
    changed_benchmark = case.episode.benchmark.copy()
    changed_benchmark.loc[
        changed_benchmark.timestamp > cutoff_timestamp, "close",
    ] *= 100
    changed_positions, after = sliding_exact_features(
        changed_bars, changed_benchmark, lookback=126, stride=11,
    )
    np.testing.assert_array_equal(positions, changed_positions)
    causal_rows = positions <= cutoff
    np.testing.assert_array_equal(before[causal_rows], after[causal_rows])


def test_parallel_and_serial_feature_digests_are_identical() -> None:
    cases = [generate_case(name, 240 + index) for index, name in enumerate(sorted(FAMILIES))]
    serial = [episode_exact_features(case.episode).digest for case in cases]
    with ThreadPoolExecutor(max_workers=4) as executor:
        parallel = list(executor.map(
            episode_exact_features, [case.episode for case in cases],
        ))
    assert [item.digest for item in parallel] == serial


def test_planted_clones_rank_first_and_critical_negatives_do_not_invert() -> None:
    for index, family in enumerate(sorted(FAMILIES), 251):
        base = generate_case(family, index)
        query = represent(base.episode)
        clone = exact_features_to_representation(episode_exact_features(base.episode))
        positive = represent(transform_case(
            base, name="positive", price_scale=7.3, volume_scale=31,
        ).episode)
        reverse = represent(transform_case(
            base, name="reverse", reverse_returns=True,
        ).episode)
        context = represent(transform_case(
            base, name="context", context_flip=True,
        ).episode)
        clone_distance = representation_distance(query, clone)[0]
        positive_distance = representation_distance(query, positive)[0]
        negative_distances = (
            representation_distance(query, reverse)[0],
            representation_distance(query, context)[0],
        )
        assert clone_distance == 0.0
        assert clone_distance <= positive_distance
        assert all(positive_distance < value for value in negative_distances)
