from __future__ import annotations

import pytest

from market_analogues.benchmark import (
    available_methods, benchmark_synthetic, correlation_path,
    euclidean_path, local_bounded_dtw,
)
from market_analogues.representation import represent
from market_analogues.synthetic import generate_case, transform_case


@pytest.mark.parametrize("distance", [euclidean_path, correlation_path, local_bounded_dtw])
def test_reference_distance_identity_and_negative_ordering(distance) -> None:
    case = generate_case("rounded_base", 2)
    positive = transform_case(case, name="scaled", price_scale=9, volume_scale=50)
    negative = transform_case(case, name="reverse", reverse_returns=True)
    base = represent(case.episode)
    assert distance(base, represent(positive.episode)) == pytest.approx(0, abs=1e-10)
    assert distance(base, represent(negative.episode)) > 0


def test_small_comparison_runs_without_optional_dependencies() -> None:
    methods = {name: function for name, function in available_methods(False).items() if name != "composite_local"}
    results = benchmark_synthetic(seeds_per_family=1, methods=methods)
    assert {result.method for result in results} == set(methods)
    assert all(result.metrics["exact_clone_rank1"] == 1 for result in results)
