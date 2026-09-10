from itertools import permutations

import numpy as np
import pandas as pd

from market_analogues.adequacy import (
    NullCandidate, concentration_metrics, greedy_accept, intervals_overlap,
    random_priority_selection,
)
from experiments.m04r.m04r14_r1a_exposure_audit import (
    Metadata, Query, _pool, _query_mapping, _retrieval_identity,
)
from market_analogues.certified_packed_search import CompactScoredCandidate, _select_compact_scored
from market_analogues.types import AnalogueMatch, EpisodeKey, InstrumentKey, SearchQuery


def _candidate(symbol: int, ordinal: int, start: int, cutoff: int) -> NullCandidate:
    return NullCandidate(symbol, ordinal, start, cutoff)


def test_inclusive_overlap_and_cap_three() -> None:
    rows = [
        _candidate(0, 0, 0, 2), _candidate(0, 1, 2, 4),
        _candidate(0, 2, 3, 5), _candidate(0, 3, 6, 8),
        _candidate(0, 4, 9, 11), _candidate(1, 5, 0, 2),
    ]
    assert intervals_overlap(rows[0], rows[1])
    assert [row.ordinal for row in greedy_accept(rows, top_k=4)] == [0, 2, 3, 5]


def test_random_priority_sampler_matches_exact_permutation_distribution() -> None:
    rows = (
        _candidate(0, 0, 0, 2), _candidate(0, 1, 1, 3),
        _candidate(0, 2, 4, 6), _candidate(1, 3, 0, 2),
    )
    exact: dict[tuple[int, ...], int] = {}
    for order in permutations(rows):
        key = tuple(item.ordinal for item in greedy_accept(order, top_k=2))
        exact[key] = exact.get(key, 0) + 1
    rng = np.random.default_rng(918273)
    observed: dict[tuple[int, ...], int] = {}
    trials = 48_000
    for _ in range(trials):
        key = tuple(item.ordinal for item in random_priority_selection(
            rng=rng, eligible_count=len(rows), candidate_at=rows.__getitem__, top_k=2,
        ))
        observed[key] = observed.get(key, 0) + 1
    assert set(observed) == set(exact)
    for key, count in exact.items():
        assert abs(observed[key] / trials - count / 24) < 0.012


def test_concentration_metrics_include_unselected_episode_population() -> None:
    result = concentration_metrics(
        [3, 1], [2, 2], episode_population=4, symbol_population=2,
    )
    assert result["episode_unique"] == 2
    assert result["episode_max"] == 3
    assert result["episode_top_1_percent_share"] == 0.75
    assert result["episode_gini"] == 0.625
    assert result["symbol_hhi"] == 0.5


def test_random_priority_output_always_satisfies_production_constraints() -> None:
    rows = tuple(
        _candidate(symbol, ordinal, window * 2, window * 2 + 5)
        for ordinal, (symbol, window) in enumerate(
            ([(0, value) for value in range(8)] +
             [(1, value) for value in range(8)] +
             [(2, value) for value in range(8)])
        )
    )
    rng = np.random.default_rng(281)
    for _ in range(200):
        selected = random_priority_selection(
            rng=rng, eligible_count=len(rows), candidate_at=rows.__getitem__, top_k=6,
        )
        assert len(selected) == 6
        assert max(sum(item.symbol_id == symbol for item in selected) for symbol in range(3)) <= 3
        assert all(
            not intervals_overlap(left, right)
            for index, left in enumerate(selected) for right in selected[index + 1:]
        )


def test_query_mapping_removes_only_overlapping_query_symbol_suffix() -> None:
    ids = np.asarray([np.void(bytes([value]) * 12) for value in range(5)])
    metadata = Metadata(
        ids, np.asarray([10, 20, 30, 10, 20], dtype=np.int64),
        np.asarray([0, 3]), np.asarray([3, 5]), ("A", "B"),
    )
    cumulative, total = _pool(metadata, 25)
    query = Query("q", "A", 0, 15, 25, 3)
    eligible, candidate_at = _query_mapping(metadata, cumulative, total, query)
    assert eligible == 3
    assert [(candidate_at(i).symbol_id, candidate_at(i).ordinal) for i in range(3)] == [
        (0, 0), (1, 3), (1, 4),
    ]


def test_local_greedy_rule_matches_production_compact_selector() -> None:
    geometry = (
        _candidate(0, 0, 0, 2), _candidate(0, 1, 2, 4),
        _candidate(0, 2, 3, 5), _candidate(0, 3, 6, 8),
        _candidate(0, 4, 9, 11), _candidate(1, 5, 0, 2),
    )
    instruments = (InstrumentKey("x", "A"), InstrumentKey("x", "B"))
    compact = []
    for distance, row in enumerate(geometry):
        key = EpisodeKey(
            instruments[row.symbol_id], pd.Timestamp("2000-01-01") + pd.Timedelta(days=row.ordinal),
            252, "dense-v1",
        )
        compact.append(CompactScoredCandidate(
            AnalogueMatch(key, float(distance), {}), instruments[row.symbol_id],
            row.start_ns, row.cutoff_ns,
        ))
    query_key = EpisodeKey(InstrumentKey("x", "Q"), pd.Timestamp("2001-01-01"), 252, "dense-v1")
    production = _select_compact_scored(
        compact, SearchQuery(query_key, top_k=4, max_per_instrument=3),
    )
    local = greedy_accept(geometry, top_k=4, max_per_symbol=3)
    assert [item.episode_key.cutoff.day - 1 for item in production] == [item.ordinal for item in local]


def test_retrieval_identity_projects_only_immutable_link_fields() -> None:
    assert _retrieval_identity([{
        "query_episode_id": "q", "episode_id": "e", "rank": 2, "distance": 0.4,
    }]) == [{"query_episode_id": "q", "episode_id": "e", "rank": 2}]
