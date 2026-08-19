from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .distance import (
    DistanceConfig, complete_representation_distance, representation_distance,
    representation_distance_lower_bound, representation_dtw_lower_bound,
)
from .representation import Representation, represent
from .types import AnalogueMatch, Episode, SearchQuery


@dataclass(frozen=True)
class SearchCandidate:
    episode: Episode
    representation: Representation

    @classmethod
    def from_episode(cls, episode: Episode) -> "SearchCandidate":
        return cls(episode, represent(episode))


@dataclass(frozen=True)
class ScoredCandidate:
    match: AnalogueMatch
    episode: Episode


@dataclass(frozen=True)
class PrunedScoreReport:
    scored: tuple[ScoredCandidate, ...]
    eligible_candidates: int
    exact_evaluated: int
    safely_pruned: int
    dtw_bounds_evaluated: int


def _overlaps(a: Episode, b: Episode) -> bool:
    if a.key.instrument != b.key.instrument:
        return False
    left = set(a.bars.timestamp.astype(str))
    return bool(left.intersection(b.bars.timestamp.astype(str)))


def latest_eligible_cutoff(query: Episode, minimum_history_gap_bars: int) -> pd.Timestamp:
    """Return a cutoff separated by observed query sessions, not weekday guesses."""
    if minimum_history_gap_bars <= 0:
        return query.key.cutoff
    timestamps = query.bars.timestamp.drop_duplicates().sort_values().reset_index(drop=True)
    if len(timestamps) > minimum_history_gap_bars:
        return pd.Timestamp(timestamps.iloc[-minimum_history_gap_bars - 1])
    # Only used when a caller requests a gap longer than the query episode.
    return query.key.cutoff - pd.offsets.BDay(minimum_history_gap_bars)


def eligible(query: Episode, candidate: Episode, request: SearchQuery) -> bool:
    if query.key.id == candidate.key.id or _overlaps(query, candidate):
        return False
    if candidate.key.cutoff >= query.key.cutoff:
        return False
    latest_eligible = latest_eligible_cutoff(query, request.minimum_history_gap_bars)
    if candidate.key.cutoff > latest_eligible:
        return False
    if candidate.quality_tier not in request.quality_tiers:
        return False
    if request.search_datasets and candidate.key.instrument.dataset_id not in request.search_datasets:
        return False
    if not request.cross_dataset and candidate.key.instrument.dataset_id != query.key.instrument.dataset_id:
        return False
    return True


def score_candidates(
    query: Episode,
    candidates: list[SearchCandidate],
    request: SearchQuery | None = None,
    config: DistanceConfig | None = None,
) -> list[ScoredCandidate]:
    request = request or SearchQuery(query.key)
    query_representation = represent(query)
    scored: list[ScoredCandidate] = []
    for candidate in candidates:
        if not eligible(query, candidate.episode, request):
            continue
        distance, components, path = representation_distance(
            query_representation, candidate.representation, config,
        )
        scored.append(ScoredCandidate(AnalogueMatch(
            candidate.episode.key, distance, components, path,
            candidate.episode.quality_tier, candidate.episode.quality_issues,
        ), candidate.episode))
    return scored


def select_scored(scored: list[ScoredCandidate], request: SearchQuery) -> list[AnalogueMatch]:
    ranked = sorted(
        scored, key=lambda item: (item.match.total_distance, item.match.episode_key.id),
    )
    selected: list[ScoredCandidate] = []
    per_instrument: dict[object, int] = {}
    for item in ranked:
        match, episode = item.match, item.episode
        instrument = episode.key.instrument
        if per_instrument.get(instrument, 0) >= request.max_per_instrument:
            continue
        if request.deduplicate_overlaps and any(
            _overlaps(episode, chosen.episode) for chosen in selected
        ):
            continue
        selected.append(item)
        per_instrument[instrument] = per_instrument.get(instrument, 0) + 1
        if len(selected) >= request.top_k:
            break
    return [item.match for item in selected]


def exact_search(
    query: Episode,
    candidates: list[SearchCandidate],
    request: SearchQuery | None = None,
    config: DistanceConfig | None = None,
) -> list[AnalogueMatch]:
    request = request or SearchQuery(query.key)
    return select_scored(score_candidates(query, candidates, request, config), request)


def score_candidates_pruned(
    query: Episode,
    candidates: list[SearchCandidate],
    request: SearchQuery | None = None,
    config: DistanceConfig | None = None,
    *,
    use_dtw_bound: bool = False,
) -> PrunedScoreReport:
    """Preserve the exact constrained result while safely avoiding some DTW work."""
    request = request or SearchQuery(query.key)
    config = config or DistanceConfig()
    query_representation = represent(query)
    bounded: list[tuple[float, str, SearchCandidate, dict[str, float], float]] = []
    for candidate in candidates:
        if not eligible(query, candidate.episode, request):
            continue
        lower, components, rigid = representation_distance_lower_bound(
            query_representation, candidate.representation, config,
        )
        bounded.append((lower, candidate.episode.key.id, candidate, components, rigid))
    bounded.sort(key=lambda item: (item[0], item[1]))

    scored: list[ScoredCandidate] = []
    threshold = float("inf")
    evaluated = 0
    dtw_bounds_evaluated = 0
    for lower, _, candidate, components, rigid in bounded:
        # Strict comparison preserves total-distance/episode-ID tie behavior.
        if lower > threshold:
            break
        dtw_lower = 0.0
        if use_dtw_bound:
            dtw_lower = representation_dtw_lower_bound(
                query_representation, candidate.representation,
                config.dtw_band_fraction,
            )
            dtw_bounds_evaluated += 1
        strengthened_lower = lower + config.weights.get("price", 0.0) * .45 * dtw_lower
        if strengthened_lower > threshold:
            continue
        components = dict(components)
        components["price"] += .45 * dtw_lower
        total, exact_components, path = complete_representation_distance(
            query_representation, candidate.representation, strengthened_lower,
            components, rigid, config,
        )
        scored.append(ScoredCandidate(AnalogueMatch(
            candidate.episode.key, total, exact_components, path,
            candidate.episode.quality_tier, candidate.episode.quality_issues,
        ), candidate.episode))
        evaluated += 1
        selected = select_scored(scored, request)
        if len(selected) >= request.top_k:
            threshold = max(match.total_distance for match in selected)
        else:
            # A newly evaluated lower-distance candidate can overlap several
            # previously selected episodes (or consume a per-instrument slot),
            # temporarily reopening the constrained top-k. The old threshold
            # is then unsafe because a farther replacement may be required.
            threshold = float("inf")
    return PrunedScoreReport(
        tuple(scored), len(bounded), evaluated, len(bounded) - evaluated,
        dtw_bounds_evaluated,
    )


def exact_search_pruned(
    query: Episode,
    candidates: list[SearchCandidate],
    request: SearchQuery | None = None,
    config: DistanceConfig | None = None,
    *,
    use_dtw_bound: bool = False,
) -> tuple[list[AnalogueMatch], PrunedScoreReport]:
    request = request or SearchQuery(query.key)
    report = score_candidates_pruned(
        query, candidates, request, config, use_dtw_bound=use_dtw_bound,
    )
    return select_scored(list(report.scored), request), report
