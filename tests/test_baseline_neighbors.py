import numpy as np

from market_analogues.baseline_neighbors import (
    baseline_neighbor_contract,
    deterministic_random_neighbors,
    recent_return_volatility,
    recent_return_volatility_at_positions,
    recent_return_volatility_neighbors,
)


def _ids(count: int) -> np.ndarray:
    return np.asarray([np.void(value.to_bytes(12, "big")) for value in range(count)])


def test_recent_features_use_frozen_windows() -> None:
    close = np.exp(np.arange(64, dtype=float) * .01)
    observed = recent_return_volatility(close)
    np.testing.assert_allclose(observed[:2], [np.exp(.20) - 1, np.exp(.63) - 1])
    assert observed[2] < 1e-15
    assert np.isnan(recent_return_volatility(np.ones(63))).all()


def test_vector_recent_features_match_independent_windows() -> None:
    rng = np.random.default_rng(221)
    close = np.exp(np.cumsum(rng.normal(0, .02, 180)))
    positions = np.asarray([63, 64, 90, 179], dtype=np.int64)
    observed = recent_return_volatility_at_positions(close, positions)
    expected = []
    for position in positions:
        window = close[position - 63:position + 1]
        expected.append([
            window[-1] / window[-21] - 1,
            window[-1] / window[0] - 1,
            np.std(np.diff(np.log(window[-21:])), ddof=1),
        ])
    np.testing.assert_allclose(observed, expected, rtol=0, atol=0)


def test_random_neighbors_are_permutation_invariant_and_distinct() -> None:
    ids = _ids(8); symbol_ids = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])
    eligible = np.asarray([True, True, True, False, True, True, True, True])
    symbols = ("A", "B", "C", "D"); query = f"{999:024x}"
    expected = deterministic_random_neighbors(
        ids, symbol_ids, eligible, symbols, query, top_k=3,
    )
    order = np.asarray([6, 1, 4, 0, 7, 3, 2, 5])
    observed = deterministic_random_neighbors(
        ids[order], symbol_ids[order], eligible[order], symbols, query, top_k=3,
    )
    assert observed == expected
    assert len({row.symbol for row in observed}) == 3
    assert baseline_neighbor_contract()["outcomes_or_labels_used"] is False


def test_rank_l1_neighbors_match_scalar_ordinal_oracle() -> None:
    ids = _ids(7)
    symbol_ids = np.asarray([0, 0, 1, 2, 3, 4, 5])
    symbols = tuple("ABCDEF")
    features = np.asarray([
        [0., 2., 1.], [1., 1., 2.], [2., 0., 3.], [3., 3., 0.],
        [1., 2., 3.], [2., 1., 0.], [np.nan, 0., 0.],
    ])
    eligible = np.asarray([True, True, True, True, True, True, True])
    query_id = f"{99:024x}"; query = np.asarray([1.5, 1.5, 1.5])
    observed = recent_return_volatility_neighbors(
        features, ids, symbol_ids, eligible, symbols, query, query_id, top_k=4,
    )
    positions = np.arange(6)
    scalar_ranks = np.zeros((6, 3)); query_ranks = []
    for column in range(3):
        rows = [(features[p, column], bytes(ids[p]), p) for p in positions]
        rows.append((query[column], bytes.fromhex(query_id), -1))
        rows.sort()
        for rank, (_, _, position) in enumerate(rows):
            value = rank / 6
            if position < 0: query_ranks.append(value)
            else: scalar_ranks[position, column] = value
    distance = np.abs(scalar_ranks - np.asarray(query_ranks)).sum(axis=1)
    expected = []
    seen = set()
    for position in sorted(positions, key=lambda p: (distance[p], bytes(ids[p]))):
        symbol = symbol_ids[position]
        if symbol in seen: continue
        seen.add(symbol); expected.append((bytes(ids[position]).hex(), float(distance[position])))
        if len(expected) == 4: break
    assert [(row.episode_id, row.distance) for row in observed] == expected
