from __future__ import annotations

import math

import numpy as np
import pandas as pd

from market_analogues.search import (
    SearchCandidate, exact_search, exact_search_pruned, latest_eligible_cutoff, score_candidates,
    select_scored,
)
from market_analogues.synthetic import FAMILIES, transform_case, verification_corpus
from market_analogues.types import SearchQuery


def _ndcg(labels: list[int], k: int = 10) -> float:
    dcg = sum(value / math.log2(rank + 2) for rank, value in enumerate(labels[:k]))
    ideal = sum(value / math.log2(rank + 2) for rank, value in enumerate(sorted(labels, reverse=True)[:k]))
    return dcg / ideal if ideal else 1.0


def test_exact_clone_rank_and_family_retrieval() -> None:
    corpus = verification_corpus(seeds_per_family=5)
    candidates = [SearchCandidate.from_episode(case.episode) for case in corpus]
    rank_one = 0
    recalls: list[float] = []
    ndcgs: list[float] = []
    for original in corpus[::2]:
        query = transform_case(
            original, name="query", price_scale=3.5, volume_scale=11,
            time_shift_days=1000,
        ).episode
        request = SearchQuery(query.key, top_k=10)
        matches = exact_search(query, candidates, request)
        if matches[0].episode_key.instrument == original.episode.key.instrument:
            rank_one += 1
        labels = [int(m.episode_key.instrument.source_symbol.startswith(original.family + "-")) for m in matches]
        recalls.append(sum(labels) / 5)
        ndcgs.append(_ndcg(labels))
    assert rank_one == len(corpus[::2])
    assert sum(recalls) / len(recalls) >= .80
    assert sum(ndcgs) / len(ndcgs) >= .90


def test_context_flip_is_a_hard_negative() -> None:
    original = verification_corpus(seeds_per_family=1)[0]
    query = transform_case(original, name="later-query", time_shift_days=1000)
    positive = transform_case(original, name="positive", noise=.0001, seed=4)
    negative = transform_case(original, name="context-negative", noise=.0001, context_flip=True, seed=4)
    candidates = [SearchCandidate.from_episode(x.episode) for x in [positive, negative]]
    result = exact_search(query.episode, candidates, SearchQuery(query.episode.key, top_k=2))
    assert result[0].episode_key == positive.episode.key


def test_contemporaneous_and_future_candidates_are_not_historical() -> None:
    original = verification_corpus(seeds_per_family=1)[0]
    same_time = transform_case(original, name="same-time")
    later = transform_case(original, name="future", time_shift_days=10)
    candidates = [SearchCandidate.from_episode(x.episode) for x in [same_time, later]]
    result = exact_search(original.episode, candidates, SearchQuery(original.episode.key, top_k=10))
    assert result == []


def test_candidate_must_have_room_for_observed_outcome_horizon() -> None:
    original = verification_corpus(seeds_per_family=1)[0]
    query = transform_case(original, name="query", time_shift_days=30)
    candidate = transform_case(original, name="recent")
    result = exact_search(
        query.episode, [SearchCandidate.from_episode(candidate.episode)],
        SearchQuery(query.episode.key, top_k=10, minimum_history_gap_bars=60),
    )
    assert result == []


def test_history_gap_uses_observed_sessions_instead_of_business_day_estimate() -> None:
    query = transform_case(
        verification_corpus(seeds_per_family=1)[0], name="later", time_shift_days=3000,
    ).episode
    query.bars.loc[200:, "timestamp"] += pd.Timedelta(days=10)
    query.key = type(query.key)(
        query.key.instrument, query.bars.timestamp.iloc[-1], query.key.lookback,
        query.key.representation_version,
    )
    assert latest_eligible_cutoff(query, 60) == query.bars.timestamp.iloc[-61]


def test_overlapping_results_are_deduplicated() -> None:
    base = verification_corpus(seeds_per_family=1)[0]
    query = transform_case(base, name="later", time_shift_days=3000).episode
    bars = base.episode.bars
    first = base.episode
    second = transform_case(base, name="overlap").episode
    # The episodes have different symbols by default; assign the same one so
    # temporal overlap represents duplicate evidence from one instrument.
    from market_analogues.types import Episode, EpisodeKey
    second.key = EpisodeKey(first.key.instrument, second.key.cutoff, second.key.lookback, second.key.representation_version)
    candidates = [SearchCandidate.from_episode(x) for x in [first, second]]
    result = exact_search(query, candidates, SearchQuery(query.key, top_k=10))
    assert len(result) == 1
    pruned, _ = exact_search_pruned(query, candidates, SearchQuery(query.key, top_k=10))
    assert [match.episode_key.id for match in pruned] == [match.episode_key.id for match in result]


def test_reusing_candidate_scores_preserves_exact_search_ranking() -> None:
    corpus = verification_corpus(seeds_per_family=2)
    query = transform_case(corpus[0], name="later", time_shift_days=3000).episode
    candidates = [SearchCandidate.from_episode(case.episode) for case in corpus]
    request = SearchQuery(query.key, top_k=5)
    direct = exact_search(query, candidates, request)
    reused = select_scored(score_candidates(query, candidates, request), request)
    assert [match.episode_key for match in reused] == [match.episode_key for match in direct]
    assert [match.total_distance for match in reused] == [match.total_distance for match in direct]


def test_safe_pruning_is_identical_to_brute_force() -> None:
    base_query = verification_corpus(seeds_per_family=1)[0]
    query = transform_case(base_query, name="later", time_shift_days=3000).episode
    corpus = verification_corpus(seeds_per_family=5)
    candidates = [SearchCandidate.from_episode(case.episode) for case in corpus]
    request = SearchQuery(query.key, ("synthetic",), ("A",), 5,
                          minimum_history_gap_bars=0)
    brute = exact_search(query, candidates, request)
    pruned, report = exact_search_pruned(query, candidates, request)
    assert [match.episode_key.id for match in pruned] == [
        match.episode_key.id for match in brute
    ]
    np.testing.assert_allclose(
        [match.total_distance for match in pruned],
        [match.total_distance for match in brute],
    )
    assert report.exact_evaluated + report.safely_pruned == report.eligible_candidates
    assert report.safely_pruned > 0
    assert report.dtw_bounds_evaluated == 0
    stronger, stronger_report = exact_search_pruned(
        query, candidates, request, use_dtw_bound=True,
    )
    assert [match.episode_key.id for match in stronger] == [
        match.episode_key.id for match in brute
    ]
    assert stronger_report.exact_evaluated <= report.exact_evaluated
    assert stronger_report.dtw_bounds_evaluated > 0
