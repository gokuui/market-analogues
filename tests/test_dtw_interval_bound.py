from dataclasses import replace

import numpy as np
import pytest

from market_analogues.distance import representation_distance_lower_bound
from market_analogues.dtw_interval_bound import (
    DTW_CHANNELS,
    DtwIntervalBoundError,
    dtw_interval_bound_contract,
    exact_price_component,
    quantize_dtw_samples,
    quantized_dtw_lower_bound,
    validate_quantized_dtw_samples,
)
from market_analogues.quantized_bound import (
    branch_aware_quantized_representation_lower_bound,
    quantize_bound_row,
)
from market_analogues.representation import represent
from market_analogues.synthetic import FAMILIES, generate_case


@pytest.mark.parametrize("query_family", FAMILIES)
@pytest.mark.parametrize("candidate_family", FAMILIES)
def test_quantized_dtw_bound_is_safe_across_synthetic_families(
    query_family: str, candidate_family: str,
) -> None:
    query = represent(generate_case(query_family, 80_000).episode)
    candidate = represent(generate_case(candidate_family, 90_000).episode)
    row = quantize_dtw_samples(candidate)
    dtw_lower = quantized_dtw_lower_bound(query, row)
    rigid = branch_aware_quantized_representation_lower_bound(
        query, quantize_bound_row(candidate),
    ).components["price"]
    combined = rigid + 0.45 * dtw_lower
    exact = exact_price_component(query, candidate)
    native = representation_distance_lower_bound(query, candidate)[1]["price"]
    assert 0 <= rigid <= native + 1e-12
    assert 0 <= dtw_lower
    assert combined <= exact + 1e-12


def test_quantized_samples_enclose_exact_values_and_identity_is_zero() -> None:
    representation = represent(generate_case("trend_contraction_breakout", 91_000).episode)
    row = quantize_dtw_samples(representation)
    validate_quantized_dtw_samples(row)
    for index, name in enumerate(DTW_CHANNELS):
        values = representation.samples_64[name]
        centers = row.centers[index].astype(np.float64)
        assert np.max(np.abs(values - centers)) <= row.channel_error_radii[index]
    assert quantized_dtw_lower_bound(representation, row) == 0
    assert dtw_interval_bound_contract()["outcomes_or_labels_used"] is False


def test_missing_channels_are_skipped_exactly() -> None:
    query = represent(generate_case("rounded_base", 92_000).episode)
    candidate = represent(generate_case("volatile_reversal", 93_000).episode)
    query_samples = dict(query.samples_64)
    candidate_samples = dict(candidate.samples_64)
    for name in DTW_CHANNELS:
        query_samples[name] = None
        candidate_samples[name] = None
    absent_query = replace(query, samples_64=query_samples)
    absent_candidate = replace(candidate, samples_64=candidate_samples)
    row = quantize_dtw_samples(absent_candidate)
    assert quantized_dtw_lower_bound(absent_query, row) == 0


def test_quantized_dtw_bound_is_safe_for_random_finite_paths() -> None:
    base_query = represent(generate_case("rounded_base", 95_000).episode)
    base_candidate = represent(generate_case("volatile_reversal", 96_000).episode)
    rng = np.random.default_rng(97_000)
    for iteration in range(128):
        query_samples = dict(base_query.samples_64)
        candidate_samples = dict(base_candidate.samples_64)
        for channel, name in enumerate(DTW_CHANNELS):
            scale = 10.0 ** rng.uniform(-4.0, 2.0)
            query_samples[name] = rng.normal(size=64) * scale
            candidate_samples[name] = rng.normal(size=64) * scale
            if (iteration + channel) % 17 == 0:
                candidate_samples[name] = np.full(64, rng.normal() * scale)
            if (iteration + channel) % 23 == 0:
                query_samples[name] = None
        query = replace(base_query, samples_64=query_samples)
        candidate = replace(base_candidate, samples_64=candidate_samples)
        rigid = branch_aware_quantized_representation_lower_bound(
            query, quantize_bound_row(candidate),
        ).components["price"]
        combined = rigid + 0.45 * quantized_dtw_lower_bound(
            query, quantize_dtw_samples(candidate),
        )
        assert combined <= exact_price_component(query, candidate) + 1e-12


def test_malformed_rows_and_band_fail_closed() -> None:
    query = represent(generate_case("rounded_base", 94_000).episode)
    row = quantize_dtw_samples(query)
    row.centers[0, 0] = np.nan
    with pytest.raises(DtwIntervalBoundError, match="row differs"):
        quantized_dtw_lower_bound(query, row)
    fresh = quantize_dtw_samples(query)
    with pytest.raises(DtwIntervalBoundError, match="band"):
        quantized_dtw_lower_bound(query, fresh, band_fraction=-0.1)
