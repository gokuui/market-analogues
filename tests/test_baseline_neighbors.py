import numpy as np
from hashlib import sha256

from market_analogues.baseline_neighbors import (
    baseline_neighbor_contract,
    build_baseline_rank_index,
    deterministic_random_neighbors,
    indexed_recent_return_volatility_neighbors,
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


def test_random_symbol_first_optimization_matches_exhaustive_oracle() -> None:
    rng = np.random.default_rng(991)
    symbols = tuple(f"S{value:03d}" for value in range(100))
    symbol_ids = np.repeat(np.arange(100, dtype=np.uint32), 31)
    ids = _ids(len(symbol_ids))
    eligible = rng.random(len(ids)) > .18
    query_id = f"{88_001:024x}"

    def digest(domain: bytes, value: bytes) -> bytes:
        result = sha256()
        result.update(domain); result.update(b"\0")
        result.update(bytes.fromhex(query_id)); result.update(b"\0")
        result.update(value)
        return result.digest()

    best = {}
    for position in np.flatnonzero(eligible):
        symbol_id = int(symbol_ids[position]); raw = bytes(ids[position])
        key = digest(b"wf03-random-episode-v1", raw)
        if symbol_id not in best or (key, raw) < best[symbol_id]:
            best[symbol_id] = (key, raw)
    ordered = sorted(best, key=lambda value: (
        digest(b"wf03-random-symbol-v1", symbols[value].encode()), symbols[value],
    ))[:20]
    expected = [(best[value][1].hex(), symbols[value], best[value][0].hex())
                for value in ordered]
    observed = deterministic_random_neighbors(
        ids, symbol_ids, eligible, symbols, query_id,
    )
    assert [(row.episode_id, row.symbol, row.order_key) for row in observed] == expected


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


def test_indexed_rank_neighbors_exactly_match_exhaustive_sort() -> None:
    rng = np.random.default_rng(1138)
    count = 2200
    ids = _ids(count)
    symbol_ids = rng.integers(0, 137, count, dtype=np.uint32)
    symbols = tuple(f"S{value:03d}" for value in range(137))
    features = rng.normal(size=(count, 3))
    features[:12] = np.asarray([.5, -.5, 1.])
    features[-3:] = np.nan
    eligible = rng.random(count) > .22
    query = np.asarray([.5, .1, -.2])
    query_id = f"{44_321:024x}"
    expected = recent_return_volatility_neighbors(
        features, ids, symbol_ids, eligible, symbols, query, query_id,
    )
    index = build_baseline_rank_index(features, ids)
    observed = indexed_recent_return_volatility_neighbors(
        index, features, ids, symbol_ids, eligible, symbols, query, query_id,
    )
    assert observed == expected

    order = rng.permutation(count)
    shuffled_index = build_baseline_rank_index(features[order], ids[order])
    shuffled = indexed_recent_return_volatility_neighbors(
        shuffled_index, features[order], ids[order], symbol_ids[order],
        eligible[order], symbols, query, query_id,
    )
    assert shuffled == expected
