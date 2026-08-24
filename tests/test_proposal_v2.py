from __future__ import annotations

import numpy as np
import pytest

from market_analogues.proposal_v2 import (
    LAYOUTS, PROPOSAL_POOLS, PROPOSAL_ROUTES, ProposalSignatureV2,
    proposal_signature_v2, proposal_signatures_v2, proposal_v2_contract, proposal_v2_distances,
    proposal_v2_distances_many, proposal_v2_route_admitted,
    proposal_v2_storage_bytes,
)
from market_analogues.representation import represent
from market_analogues.synthetic import FAMILIES, generate_case, transform_case


def _signature(family: str, seed: int, dimensions: int) -> ProposalSignatureV2:
    return proposal_signature_v2(
        represent(generate_case(family, seed).episode), dimensions,
    )


def test_proposal_v2_contract_dimensions_routes_and_bytes_are_frozen() -> None:
    contract = proposal_v2_contract()
    assert set(LAYOUTS) == {192, 240, 320}
    assert contract["digest"] == "dcb44ebf809fdb994b8bb7eac901f880187171778b9544986e0f20122781b969"
    assert contract["route_union"]["routes"] == ["composite", *PROPOSAL_ROUTES]
    assert contract["per_query_weights"] is False
    assert contract["outcomes_or_labels_used"] is False
    for dimensions, layout in LAYOUTS.items():
        materialized = sum(field[2] for field in contract["layouts"][str(dimensions)]["field_order"])
        assert materialized == layout.dimensions == dimensions
        assert proposal_v2_storage_bytes(dimensions, "float32") == dimensions * 4 + 3
        assert proposal_v2_storage_bytes(dimensions, "float16") == dimensions * 2 + 3


def test_proposal_signatures_are_finite_and_have_explicit_missingness() -> None:
    case = generate_case("rounded_base", 611)
    without_market = type(case.episode)(
        case.episode.key, case.episode.bars, None,
        case.episode.quality_tier, case.episode.quality_issues,
    )
    for dimensions in LAYOUTS:
        complete = proposal_signature_v2(represent(case.episode), dimensions)
        missing = proposal_signature_v2(represent(without_market), dimensions)
        assert complete.vector.shape == (dimensions,)
        assert complete.presence.shape == missing.presence.shape == (19,)
        assert np.isfinite(complete.vector).all()
        assert complete.presence.sum() > missing.presence.sum()
        distances = proposal_v2_distances(
            complete, missing.vector, missing.presence,
        )
        assert distances["market_context"][0] > 0
        assert all(np.isfinite(value).all() for value in distances.values())


def test_batch_and_scalar_projection_are_byte_identical() -> None:
    representations = [
        represent(generate_case(family, 601 + index).episode)
        for index, family in enumerate(FAMILIES)
    ]
    for dimensions in LAYOUTS:
        batch = proposal_signatures_v2(representations, dimensions)
        scalar = [proposal_signature_v2(item, dimensions) for item in representations]
        np.testing.assert_array_equal(
            batch.vectors, np.stack([item.vector for item in scalar]),
        )
        np.testing.assert_array_equal(
            batch.presence, np.stack([item.presence for item in scalar]),
        )


def test_price_proxy_has_distinct_projected_close_path_share() -> None:
    base = _signature("steady_trend", 612, 192)
    query = ProposalSignatureV2(
        192, np.zeros(192, dtype=np.float32), np.ones_like(base.presence),
    )
    contract_fields = proposal_v2_contract()["layouts"]["192"]["field_order"]
    slices = {}
    offset = 0
    for group, name, count in contract_fields:
        slices[(group, name)] = slice(offset, offset + count)
        offset += count
    close_candidate = np.zeros(192, dtype=np.float32)
    return_candidate = np.zeros(192, dtype=np.float32)
    ramp = np.linspace(-0.2, 0.2, slices[("price", "close_path")].stop - slices[("price", "close_path")].start)
    close_candidate[slices[("price", "close_path")]] += ramp
    return_candidate[slices[("price", "return")]] += ramp
    close_score = proposal_v2_distances(
        query, close_candidate, query.presence,
    )["price"][0]
    return_score = proposal_v2_distances(
        query, return_candidate, query.presence,
    )["price"][0]
    assert close_score > 5 * return_score


@pytest.mark.parametrize("dimensions", sorted(LAYOUTS))
def test_every_synthetic_family_has_clone_first_and_critical_ordering(
    dimensions: int,
) -> None:
    for seed, family in enumerate(sorted(FAMILIES), 621):
        case = generate_case(family, seed)
        query = proposal_signature_v2(represent(case.episode), dimensions)
        candidates = [
            proposal_signature_v2(represent(case.episode), dimensions),
            proposal_signature_v2(represent(transform_case(
                case, name="near-positive", price_scale=7.3, volume_scale=31,
                noise=.0005, seed=seed,
            ).episode), dimensions),
            proposal_signature_v2(represent(transform_case(
                case, name="reverse", reverse_returns=True,
            ).episode), dimensions),
            proposal_signature_v2(represent(transform_case(
                case, name="context", context_flip=True,
            ).episode), dimensions),
        ]
        matrix = np.stack([item.vector for item in candidates])
        masks = np.stack([item.presence for item in candidates])
        scores = proposal_v2_distances(query, matrix, masks)["composite"]
        assert scores[0] == 0.0
        assert scores[1] < scores[2]
        assert scores[1] < scores[3]
        assert np.flatnonzero(scores == scores.min()).tolist() == [0]

        unit_copy = proposal_signature_v2(represent(transform_case(
            case, name="unit-copy", price_scale=7.3, volume_scale=31,
        ).episode), dimensions)
        unit_score = proposal_v2_distances(
            query, unit_copy.vector, unit_copy.presence,
        )["composite"][0]
        assert unit_score < 1e-5


def test_float16_quantization_preserves_synthetic_nearest_family() -> None:
    for dimensions in LAYOUTS:
        signatures = [
            _signature(family, 650 + index, dimensions)
            for index, family in enumerate(sorted(FAMILIES))
        ]
        float32 = np.stack([item.vector for item in signatures])
        float16 = float32.astype(np.float16)
        masks = np.stack([item.presence for item in signatures])
        for index, query in enumerate(signatures):
            native = proposal_v2_distances(query, float32, masks)["composite"]
            quantized = proposal_v2_distances(query, float16, masks)["composite"]
            assert np.argmin(native) == np.argmin(quantized) == index
            assert abs(float(native[index] - quantized[index])) < 1e-3


def test_distance_rows_are_order_invariant_and_reject_nonfinite_values() -> None:
    query = _signature("failed_breakout", 671, 240)
    signatures = [
        _signature(family, 672 + index, 240)
        for index, family in enumerate(FAMILIES)
    ]
    matrix = np.stack([item.vector for item in signatures])
    masks = np.stack([item.presence for item in signatures])
    order = np.asarray([3, 0, 4, 1, 2])
    original = proposal_v2_distances(query, matrix, masks)
    shuffled = proposal_v2_distances(query, matrix[order], masks[order])
    for route in (*PROPOSAL_ROUTES, "composite"):
        np.testing.assert_array_equal(original[route][order], shuffled[route])
    matrix[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        proposal_v2_distances(query, matrix, masks)


def test_multi_query_scoring_is_identical_to_repeated_scalar_calls() -> None:
    representations = [
        represent(generate_case(family, 690 + index).episode)
        for index, family in enumerate(FAMILIES)
    ]
    missing_case = generate_case("steady_trend", 699)
    missing_episode = type(missing_case.episode)(
        missing_case.episode.key, missing_case.episode.bars, None,
        missing_case.episode.quality_tier, missing_case.episode.quality_issues,
    )
    representations.append(represent(missing_episode))
    for dimensions in LAYOUTS:
        batch = proposal_signatures_v2(representations, dimensions)
        queries = [
            proposal_signature_v2(representations[index], dimensions)
            for index in (0, 2, 5)
        ]
        combined = proposal_v2_distances_many(
            queries, batch.vectors, batch.presence,
        )
        for query_index, query in enumerate(queries):
            scalar = proposal_v2_distances(query, batch.vectors, batch.presence)
            for route in (*PROPOSAL_ROUTES, "composite"):
                np.testing.assert_allclose(
                    combined[route][query_index], scalar[route], rtol=0, atol=0,
                )


def test_route_union_uses_only_frozen_positive_rank_quotas() -> None:
    far = {route: 99_999 for route in PROPOSAL_ROUTES}
    for pool, quotas in PROPOSAL_POOLS.items():
        assert proposal_v2_route_admitted(far, quotas["composite"], pool)
        near = dict(far)
        near["volume_shock"] = quotas["per_component"]
        assert proposal_v2_route_admitted(near, 99_999, pool)
        assert not proposal_v2_route_admitted(far, 99_999, pool)
    with pytest.raises(ValueError, match="routes"):
        proposal_v2_route_admitted({"stage": 1}, 1, 1_000)
    with pytest.raises(ValueError, match="one-based"):
        proposal_v2_route_admitted({route: 1 for route in PROPOSAL_ROUTES}, 0, 1_000)
