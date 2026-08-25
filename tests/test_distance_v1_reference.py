from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from market_analogues.distance import (
    bounded_dtw, bounded_dtw_distance, representation_distance,
)
from market_analogues.distance_v1_reference import (
    distance_v1_contract, reference_bounded_dtw, reference_representation_distance,
)
from market_analogues.exact_batch import sliding_exact_representations
from market_analogues.representation import Representation, represent
from market_analogues.synthetic import FAMILIES, generate_case, transform_case
from market_analogues.types import Episode


def _assert_parity(left: Representation, right: Representation) -> None:
    expected = reference_representation_distance(left, right)
    actual_total, actual_components, actual_path = representation_distance(left, right)
    assert actual_total == pytest.approx(expected.total, abs=2e-12, rel=2e-12)
    assert actual_components.keys() == expected.components.keys()
    for name, value in expected.components.items():
        assert actual_components[name] == pytest.approx(value, abs=2e-12, rel=2e-12)
    assert tuple(actual_path) == expected.alignment


def test_frozen_contract_is_complete_and_digest_bound() -> None:
    contract = distance_v1_contract()
    assert contract["schema_version"] == "market-analogue-distance-v1-contract"
    assert sum(contract["weights"].values()) == pytest.approx(1.0)
    assert contract["dtw"]["predecessor_tie_order"] == [
        "vertical", "horizontal", "diagonal",
    ]
    assert contract["outcomes_or_labels_used"] is False
    assert contract["digest"] == "458e42807d5b69faf5616b4a037c4c0f52003344f9678e14b9549eaa878575a6"


def test_reference_matches_production_across_all_synthetic_families() -> None:
    families = sorted(FAMILIES)
    representations = [
        represent(generate_case(name, index + 50).episode)
        for index, name in enumerate(families)
    ]
    for index, left in enumerate(representations):
        _assert_parity(left, left)
        _assert_parity(left, representations[(index + 1) % len(representations)])


def test_reference_matches_missing_constant_and_extreme_finite_inputs() -> None:
    base = represent(generate_case("steady_trend", 71).episode)
    other = represent(generate_case("volatile_reversal", 72).episode)
    left_48 = dict(base.samples_48)
    right_48 = dict(other.samples_48)
    left_64 = dict(base.samples_64)
    right_64 = dict(other.samples_64)
    left_48["benchmark_path"] = None
    right_48["benchmark_return"] = None
    left_64["relative_path"] = None
    right_64["relative_path"] = None
    changed_left = replace(
        base,
        coarse=np.full_like(base.coarse, 1e8),
        samples_48=left_48,
        samples_64=left_64,
        stage=np.zeros_like(base.stage),
    )
    changed_right = replace(
        other,
        coarse=np.full_like(other.coarse, 1e8 + 16),
        samples_48=right_48,
        samples_64=right_64,
        stage=np.zeros_like(other.stage),
    )
    _assert_parity(changed_left, changed_right)


@pytest.mark.parametrize("shape", [(17, 23, 1), (64, 64, 4), (9, 14, 3)])
def test_independent_dtw_matches_cost_and_exact_tie_path(shape: tuple[int, int, int]) -> None:
    n, m, channels = shape
    rng = np.random.default_rng(n * 100 + m)
    left = np.round(rng.normal(size=(n, channels)), 2)
    right = np.round(rng.normal(size=(m, channels)), 2)
    expected, expected_path = reference_bounded_dtw(left, right)
    actual, actual_path = bounded_dtw(left, right)
    assert actual == pytest.approx(expected, abs=2e-15, rel=2e-15)
    assert tuple(actual_path) == expected_path
    assert bounded_dtw_distance(left, right) == actual


def test_compiled_dtw_distance_matches_tied_and_one_dimensional_paths() -> None:
    rows = (
        np.zeros(64),
        np.r_[np.zeros(31), np.ones(33)],
        np.tile(np.asarray([[0.0, 1.0], [1.0, 0.0]]), (32, 1)),
    )
    for left in rows:
        right = np.roll(left, 3, axis=0)
        expected, _ = bounded_dtw(left, right)
        assert bounded_dtw_distance(left, right) == expected


def test_reference_identity_symmetry_and_unit_invariance() -> None:
    base = generate_case("trend_contraction_breakout", 81)
    scaled = transform_case(base, name="scaled", price_scale=17, volume_scale=31)
    left, right = represent(base.episode), represent(scaled.episode)
    identity = reference_representation_distance(left, left)
    forward = reference_representation_distance(left, right)
    reverse = reference_representation_distance(right, left)
    assert identity.total == 0.0
    assert forward.total == pytest.approx(reverse.total, abs=2e-12)
    assert forward.components == pytest.approx(reverse.components, abs=2e-12)
    assert forward.total < 1e-6


def test_reference_matches_production_on_sliding_batch_representations() -> None:
    case = generate_case("rounded_base", 91, n=310)
    batch = sliding_exact_representations(
        case.episode.bars, case.episode.benchmark, lookback=126, stride=37,
    )
    query = batch.representations[-1]
    for candidate in batch.representations[:-1]:
        _assert_parity(query, candidate)


def test_future_benchmark_rows_do_not_change_reference_distance() -> None:
    left_case = generate_case("steady_trend", 101)
    right_case = generate_case("rounded_base", 102)
    cutoff = pd.Timestamp(left_case.episode.key.cutoff)
    benchmark = left_case.episode.benchmark.copy()
    future = benchmark.iloc[[-1]].copy()
    future["timestamp"] = cutoff + pd.offsets.BDay(5)
    future["close"] *= 100
    extended = pd.concat([benchmark, future], ignore_index=True)
    original = represent(left_case.episode)
    changed = represent(Episode(
        left_case.episode.key, left_case.episode.bars.copy(), extended,
        left_case.episode.quality_tier, left_case.episode.quality_issues,
    ))
    right = represent(right_case.episode)
    assert reference_representation_distance(original, right).total == pytest.approx(
        reference_representation_distance(changed, right).total, abs=0.0,
    )
