from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .distance import representation_distance
from .gates import GateReport
from .index import CoarseIndex, recall_at_k
from .representation import represent
from .search import SearchCandidate, exact_search
from .synthetic import generate_case, transform_case, verification_corpus
from .types import Episode, EpisodeKey, InstrumentKey, SearchQuery, VerificationReport


THRESHOLDS = {
    "exact_clone_rank1": 1.00,
    "hard_invariance": 1.00,
    "pair_ordering": .95,
    "ndcg_at_10": .90,
    "family_recall_at_10": .80,
    "critical_negative_rate_max": .02,
    "cross_dataset_drop_max": .05,
    "coarse_index_roundtrip_recall_at_50": .98,
}


def _ndcg(labels: list[int], relevant: int, k: int = 10) -> float:
    dcg = sum(value / math.log2(rank + 2) for rank, value in enumerate(labels[:k]))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(relevant, k)))
    return dcg / ideal if ideal else 1.0


def run_synthetic_verifier(seeds_per_family: int = 5) -> VerificationReport:
    corpus = verification_corpus(seeds_per_family)
    candidates = [SearchCandidate.from_episode(case.episode) for case in corpus]
    sample = corpus[::2]
    clone_hits = invariant_hits = pair_hits = negative_errors = 0
    recalls: list[float] = []
    cross_recalls: list[float] = []
    ndcgs: list[float] = []
    cross_candidates: list[SearchCandidate] = []
    for case in corpus:
        original = case.episode
        instrument = InstrumentKey("synthetic-other-market", original.key.instrument.source_symbol)
        key = EpisodeKey(instrument, original.key.cutoff, original.key.lookback, original.key.representation_version)
        cross_candidates.append(SearchCandidate.from_episode(Episode(
            key, original.bars.copy(), original.benchmark.copy() if original.benchmark is not None else None,
        )))
    for original in sample:
        positive = transform_case(
            original, name="positive", price_scale=7.3, volume_scale=31,
            time_shift_days=1000,
        )
        reversed_case = transform_case(original, name="reverse", reverse_returns=True)
        context_case = transform_case(original, name="context-flip", context_flip=True)
        a = represent(original.episode)
        p = represent(positive.episode)
        reverse_distance, _, _ = representation_distance(a, represent(reversed_case.episode))
        context_distance, _, _ = representation_distance(a, represent(context_case.episode))
        positive_distance, _, _ = representation_distance(a, p)
        invariant_hits += int(positive_distance < 1e-8)
        pair_hits += int(positive_distance < reverse_distance and positive_distance < context_distance)
        negative_errors += int(reverse_distance <= positive_distance or context_distance <= positive_distance)

        matches = exact_search(
            positive.episode, candidates, SearchQuery(positive.episode.key, top_k=10),
        )
        clone_hits += int(matches[0].episode_key.instrument == original.episode.key.instrument)
        labels = [int(m.episode_key.instrument.source_symbol.startswith(original.family + "-")) for m in matches]
        recalls.append(sum(labels) / seeds_per_family)
        ndcgs.append(_ndcg(labels, seeds_per_family))
        cross = exact_search(
            positive.episode, cross_candidates,
            SearchQuery(
                positive.episode.key, ("synthetic-other-market",), ("A",), 10,
                cross_dataset=True,
            ),
        )
        cross_labels = [int(m.episode_key.instrument.source_symbol.startswith(original.family + "-")) for m in cross]
        cross_recalls.append(sum(cross_labels) / seeds_per_family)

    matrix = np.vstack([candidate.representation.coarse for candidate in candidates])
    ids = [candidate.episode.key.id for candidate in candidates]
    oracle = CoarseIndex(ids, matrix)
    portable = CoarseIndex(ids, matrix.copy())
    k = min(50, len(ids))
    index_recall = recall_at_k(oracle, portable, matrix[::3], k)
    extended = generate_case("trend_contraction_breakout", 999, n=272).episode
    cutoff_position = 251
    cutoff = pd.Timestamp(extended.bars.timestamp.iloc[cutoff_position])
    prefix_key = EpisodeKey(
        extended.key.instrument, cutoff, cutoff_position + 1,
        extended.key.representation_version,
    )
    before = Episode(
        prefix_key, extended.bars.iloc[:cutoff_position + 1].copy(),
        extended.benchmark.iloc[:cutoff_position + 1].copy() if extended.benchmark is not None else None,
    )
    mutated_bars = extended.bars.copy()
    mutated_bars.loc[cutoff_position + 1:, ["open", "high", "low", "close", "volume"]] *= 100
    after = Episode(
        prefix_key, mutated_bars.iloc[:cutoff_position + 1].copy(),
        extended.benchmark.iloc[:cutoff_position + 1].copy() if extended.benchmark is not None else None,
    )
    future_delta = float(np.max(np.abs(represent(before).coarse - represent(after).coarse)))
    n = len(sample)
    same_market_recall = float(np.mean(recalls))
    cross_market_recall = float(np.mean(cross_recalls))
    metrics = {
        "cases": n,
        "exact_clone_rank1": clone_hits / n,
        "hard_invariance": invariant_hits / n,
        "pair_ordering": pair_hits / n,
        "ndcg_at_10": float(np.mean(ndcgs)),
        "family_recall_at_10": same_market_recall,
        "critical_negative_rate": negative_errors / (2 * n),
        "cross_dataset_drop": max(0.0, same_market_recall - cross_market_recall),
        "coarse_index_roundtrip_recall_at_50": index_recall,
        "future_mutation_delta": future_delta,
    }
    failures = []
    for name in ["exact_clone_rank1", "hard_invariance", "pair_ordering", "ndcg_at_10",
                 "family_recall_at_10", "coarse_index_roundtrip_recall_at_50"]:
        if metrics[name] < THRESHOLDS[name]:
            failures.append(f"{name}={metrics[name]:.4f} below {THRESHOLDS[name]:.4f}")
    for metric, threshold_name in [
        ("critical_negative_rate", "critical_negative_rate_max"),
        ("cross_dataset_drop", "cross_dataset_drop_max"),
    ]:
        if metrics[metric] > THRESHOLDS[threshold_name]:
            failures.append(f"{metric}={metrics[metric]:.4f} above {THRESHOLDS[threshold_name]:.4f}")
    if metrics["future_mutation_delta"] != 0:
        failures.append("future_mutation_delta must be exactly zero")
    return VerificationReport("synthetic-retrieval-v1", not failures, metrics, failures)


def write_verification_gate(report: VerificationReport, directory: Path) -> Path:
    return GateReport(
        task="03_synthetic_retrieval", passed=report.passed,
        metrics=report.metrics, failures=report.failures,
    ).write(directory)
