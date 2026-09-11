from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from market_analogues.analogue_candidate import (
    AnalogueCandidateError,
    MIXTURE_WEIGHTS,
    candidate_empirical_distribution,
    candidate_probabilities,
    matched_causal_probability,
    matched_causal_probabilities_batch,
)
from market_analogues.types import stable_hash


def row(index: int, label: str, *, completion: str = "2026-01-01",
        regime: str = "up", quality: str = "A", liquidity: str = "high") -> dict[str, object]:
    return {
        "query_id": f"q{index}", "origin_cutoff": "2025-12-01",
        "completion_timestamp": completion, "label": label,
        "market_regime": regime, "prefix_quality_class": quality,
        "trailing_liquidity_cell": liquidity,
    }


def test_contract_digest_and_false_claim_boundary() -> None:
    value = json.loads(Path("config/analogue-candidate-mixture-v1.json").read_text())
    digest = value.pop("contract_digest")
    assert digest == stable_hash(value)
    assert value["candidate"]["component_weights"] == MIXTURE_WEIGHTS
    assert not any(value["claims"].values())


def test_matched_probability_uses_exact_cell_at_thirty_and_excludes_future() -> None:
    outcomes = [row(i, ("favorable_first", "adverse_first", "no_touch")[i % 3])
                for i in range(30)]
    outcomes.append(row(30, "favorable_first", completion="2026-02-01"))
    result = matched_causal_probability(
        outcomes, query_cutoff="2026-01-31", market_regime="up",
        prefix_quality_class="A", trailing_liquidity_cell="high",
    )
    assert result.fallback_level == "exact" and result.support_rows == 30
    assert result.mature_evaluable_rows == 30
    np.testing.assert_allclose(result.probabilities, [1 / 3, 1 / 3, 1 / 3])


def test_matched_probability_fallback_order_and_uniform_no_history() -> None:
    outcomes = [row(i, "favorable_first", quality="B") for i in range(30)]
    result = matched_causal_probability(
        outcomes, query_cutoff="2026-01-31", market_regime="up",
        prefix_quality_class="A", trailing_liquidity_cell="high",
    )
    assert result.fallback_level == "regime_and_liquidity"
    assert result.support_rows == 30 and result.probabilities[0] > .95
    empty = matched_causal_probability(
        [], query_cutoff="2026-01-31", market_regime="up",
        prefix_quality_class="A", trailing_liquidity_cell="high",
    )
    assert empty.fallback_level == "uniform_no_history"
    assert empty.probabilities == pytest.approx((1 / 3, 1 / 3, 1 / 3))


def test_incremental_batch_matches_scalar_for_every_query_and_input_order() -> None:
    outcomes = [
        row(i, ("favorable_first", "adverse_first", "no_touch")[i % 3],
            completion=f"2026-01-{i % 28 + 1:02d}", quality="A" if i % 2 else "B")
        for i in range(60)
    ]
    queries = [
        {
            "query_id": f"target-{i}", "query_cutoff": cutoff,
            "market_regime": "up", "prefix_quality_class": quality,
            "trailing_liquidity_cell": "high",
        }
        for i, (cutoff, quality) in enumerate((
            ("2026-01-10", "A"), ("2026-01-31", "A"), ("2026-01-20", "B"),
        ))
    ]
    batch = matched_causal_probabilities_batch(list(reversed(outcomes)), list(reversed(queries)))
    for query in queries:
        scalar = matched_causal_probability(
            outcomes, query_cutoff=query["query_cutoff"], market_regime="up",
            prefix_quality_class=query["prefix_quality_class"],
            trailing_liquidity_cell="high",
        )
        assert batch[query["query_id"]] == scalar


def test_completion_must_follow_origin() -> None:
    invalid = row(1, "no_touch", completion="2025-11-30")
    with pytest.raises(AnalogueCandidateError, match="follow"):
        matched_causal_probability(
            [invalid], query_cutoff="2026-01-31", market_regime="up",
            prefix_quality_class="A", trailing_liquidity_cell="high",
        )


def test_matched_probability_rejects_duplicate_or_bad_rows() -> None:
    duplicated = [row(1, "no_touch"), row(1, "no_touch")]
    with pytest.raises(AnalogueCandidateError, match="unique"):
        matched_causal_probability(
            duplicated, query_cutoff="2026-01-31", market_regime="up",
            prefix_quality_class="A", trailing_liquidity_cell="high",
        )
    with pytest.raises(AnalogueCandidateError, match="fields"):
        matched_causal_probability(
            [{"query_id": "x"}], query_cutoff="2026-01-31", market_regime="up",
            prefix_quality_class="A", trailing_liquidity_cell="high",
        )


def test_candidate_probability_is_exact_frozen_convex_mixture() -> None:
    components = {
        "matched_causal_history": [.2, .3, .5], "composite": [.8, .1, .1],
        "price_only": [.1, .8, .1], "recent_return_volatility": [.1, .2, .7],
    }
    expected = sum(MIXTURE_WEIGHTS[name] * np.asarray(values)
                   for name, values in components.items())
    np.testing.assert_array_equal(candidate_probabilities(components), expected)
    assert sum(candidate_probabilities(components)) == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize("components", [
    {"matched_causal_history": [.2, .3, .5]},
    {
        "matched_causal_history": [.2, .3, .5], "composite": [.8, .1, .1],
        "price_only": [.1, .8, .1], "recent_return_volatility": [0., .2, .8],
    },
])
def test_candidate_probability_rejects_incomplete_or_invalid_components(components) -> None:
    with pytest.raises(AnalogueCandidateError):
        candidate_probabilities(components)


def test_continuous_empirical_mixture_preserves_component_mass_and_order() -> None:
    components = {
        "matched_causal_history": ([1., 2.], [1., 1.]),
        "composite": ([3.], [7.]),
        "price_only": ([4., 5.], [1., 3.]),
        "recent_return_volatility": ([6.], [2.]),
    }
    values, weights = candidate_empirical_distribution(components)
    np.testing.assert_array_equal(values, [1., 2., 3., 4., 5., 6.])
    assert weights.sum() == pytest.approx(1.0, abs=1e-15)
    assert weights[:2].sum() == pytest.approx(.4)
    assert weights[2] == pytest.approx(.1)
    assert weights[3:5].sum() == pytest.approx(.2)
    assert weights[5] == pytest.approx(.3)


def test_continuous_empirical_mixture_refuses_empty_or_nonfinite_input() -> None:
    valid = {
        name: ([1.], [1.]) for name in MIXTURE_WEIGHTS
    }
    invalid = dict(valid); invalid["composite"] = ([], [])
    with pytest.raises(AnalogueCandidateError):
        candidate_empirical_distribution(invalid)
    invalid = dict(valid); invalid["composite"] = ([np.nan], [1.])
    with pytest.raises(AnalogueCandidateError):
        candidate_empirical_distribution(invalid)
