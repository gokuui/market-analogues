from __future__ import annotations

import numpy as np
import pytest

from market_analogues.balanced_partition import (
    BalancedPartitionError,
    array_digest,
    balanced_recursive_partition,
    midrank_transform,
)


def test_midrank_transform_exact_ties_constants_and_range() -> None:
    values = np.asarray([
        [1.0, 7.0, 4.0],
        [1.0, 7.0, 3.0],
        [3.0, 7.0, 2.0],
        [2.0, 7.0, 1.0],
    ])
    transformed, doubled, constant = midrank_transform(values)
    assert constant == (1,)
    np.testing.assert_array_equal(transformed[:, 1], 0.0)
    np.testing.assert_array_equal(doubled[:, 1], 5)
    np.testing.assert_array_equal(doubled[:, 0], [3, 3, 8, 6])
    assert transformed[0, 0] == transformed[1, 0]
    assert transformed[0, 0] == pytest.approx(-2.0 / 3.0)
    assert transformed[2, 0] == 1.0
    np.testing.assert_allclose(transformed[:, 2], [1.0, 1 / 3, -1 / 3, -1])


def test_midrank_transform_refuses_nonfinite_or_single_row() -> None:
    with pytest.raises(BalancedPartitionError, match="finite 2D"):
        midrank_transform(np.asarray([[1.0]]))
    with pytest.raises(BalancedPartitionError, match="finite 2D"):
        midrank_transform(np.asarray([[1.0], [np.nan]]))


@pytest.mark.parametrize("leaves", [8, 12, 16])
def test_partition_is_exact_balanced_complete_and_repeatable(leaves: int) -> None:
    rng = np.random.default_rng(9127)
    raw = rng.normal(size=(327, 19))
    raw[:, 2] = 5.0
    raw[8:14, 6] = raw[7, 6]
    transformed, doubled, _ = midrank_transform(raw)
    ids = tuple(f"q-{index:04d}" for index in range(len(raw)))
    first = balanced_recursive_partition(transformed, doubled, ids, leaves=leaves)
    second = balanced_recursive_partition(transformed, doubled, ids, leaves=leaves)
    np.testing.assert_array_equal(first.labels, second.labels)
    assert first.leaf_paths == second.leaf_paths
    assert first.leaf_sizes == second.leaf_sizes
    assert first.splits == second.splits
    assert len(first.leaf_paths) == len(set(first.leaf_paths)) == leaves
    assert len(first.splits) == leaves - 1
    assert set(first.leaf_sizes) <= {len(raw) // leaves, (len(raw) + leaves - 1) // leaves}
    assert sum(first.leaf_sizes) == len(raw)


def test_partition_tie_boundary_is_resolved_by_id() -> None:
    raw = np.asarray([[0.0], [0.0], [1.0], [1.0]])
    transformed, doubled, _ = midrank_transform(raw)
    result = balanced_recursive_partition(
        transformed, doubled, ("d", "a", "c", "b"), leaves=2,
    )
    assert result.labels.tolist() == [0, 0, 1, 1]


@pytest.mark.parametrize(
    ("leaves", "expected"),
    [
        (8, {408: 2, 409: 6}),
        (12, {272: 6, 273: 6}),
        (16, {204: 10, 205: 6}),
    ],
)
def test_frozen_inventory_has_exact_leaf_size_multiset(
    leaves: int, expected: dict[int, int],
) -> None:
    rng = np.random.default_rng(321)
    transformed, doubled, _constant = midrank_transform(rng.normal(size=(3270, 5)))
    result = balanced_recursive_partition(
        transformed, doubled, tuple(f"q-{index:04d}" for index in range(3270)),
        leaves=leaves,
    )
    assert {size: result.leaf_sizes.count(size) for size in set(result.leaf_sizes)} == expected


def test_partition_refuses_noncanonical_transformed_input() -> None:
    with pytest.raises(BalancedPartitionError, match="canonical midrank"):
        balanced_recursive_partition(
            np.asarray([[0.1], [0.2], [0.3], [0.4]]), np.asarray([[2], [4], [6], [8]]),
            ("a", "b", "c", "d"), leaves=2,
        )


def test_repeated_eigenvalue_uses_lowest_ss_feature_fallback() -> None:
    transformed = np.asarray([
        [-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0],
    ])
    _canonical, doubled, _constant = midrank_transform(transformed)
    result = balanced_recursive_partition(
        transformed, doubled, ("a", "b", "c", "d"), leaves=2,
    )
    assert result.splits[0].axis_mode == "exact_variance_fallback"
    assert result.splits[0].pivot_feature == 0


def test_all_constant_node_fails_closed() -> None:
    with pytest.raises(BalancedPartitionError, match="no non-constant"):
        transformed, doubled, _constant = midrank_transform(np.zeros((8, 3)))
        balanced_recursive_partition(
            transformed, doubled, tuple(f"q-{index}" for index in range(8)), leaves=2,
        )


def test_axis_digest_binds_shape_and_values() -> None:
    base = np.asarray([1.0, 2.0])
    assert array_digest(base) == array_digest(base.copy())
    assert array_digest(base) != array_digest(base.reshape(1, 2))
    assert array_digest(base) != array_digest(np.asarray([2.0, 1.0]))
