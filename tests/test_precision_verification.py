import numpy as np

from market_analogues.exact_storage_feasibility import quantize_representation
from market_analogues.precision_verification import _graded_ndcg
from market_analogues.representation import represent
from market_analogues.synthetic import generate_case


def test_float16_quantization_is_finite_and_detectably_approximate() -> None:
    original = represent(generate_case("volatile_reversal", 7, n=126).episode)
    quantized = quantize_representation(original, "float16")
    assert np.isfinite(quantized.coarse).all()
    assert np.isfinite(quantized.stage).all()
    assert any(
        not np.array_equal(original.samples_48[name], quantized.samples_48[name])
        for name in original.samples_48
        if original.samples_48[name] is not None
    )


def test_graded_ndcg_is_one_for_same_order_and_near_one_for_adjacent_swap() -> None:
    expected = [f"item-{index}" for index in range(20)]
    swapped = expected.copy()
    swapped[15], swapped[16] = swapped[16], swapped[15]
    assert _graded_ndcg(expected, expected) == 1.0
    assert .999 < _graded_ndcg(expected, swapped) < 1.0
