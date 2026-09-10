from __future__ import annotations

from pathlib import Path

import pandas as pd

from market_analogues.adapters import DirectorySource
from market_analogues.config import BenchmarkSpec, DatasetSpec
from market_analogues.search_evidence import (
    DISPLAY_HORIZONS,
    build_search_evidence,
    retrieval_identity_digest,
)
from market_analogues.types import AnalogueMatch, EpisodeKey, InstrumentKey


def _source(
    root: Path, bars: pd.DataFrame, *, benchmark: bool = True,
) -> DirectorySource:
    root.mkdir(exist_ok=True)
    bars.to_parquet(root / "AAA.parquet", index=False)
    shifted = bars.copy()
    shifted[["open", "high", "low", "close"]] *= 1.2
    shifted.to_parquet(root / "BBB.parquet", index=False)
    benchmark_path = root.parent / "benchmark.parquet"
    if benchmark:
        market = bars.copy()
        market[["open", "high", "low", "close"]] *= .8
        market.to_parquet(benchmark_path, index=False)
    spec = DatasetSpec(
        "demo", "directory", root, "parquet", timestamp_column="date",
        benchmark=BenchmarkSpec(benchmark_path) if benchmark else None,
    )
    return DirectorySource(spec)


def _matches(bars: pd.DataFrame) -> list[AnalogueMatch]:
    cutoffs = [bars.date.iloc[80], bars.date.iloc[100]]
    return [
        AnalogueMatch(
            EpisodeKey(InstrumentKey("demo", symbol), cutoff, 63, "dense-v1"),
            total_distance=distance,
            component_distances={"price": distance},
        )
        for symbol, cutoff, distance in zip(("AAA", "BBB"), cutoffs, (.1, .2))
    ]


def test_benchmark_aware_search_evidence_is_observable_and_summarized(
    tmp_path: Path, bars: pd.DataFrame,
) -> None:
    source = _source(tmp_path / "bars", bars)
    matches = _matches(bars)
    evidence = build_search_evidence(
        source, matches, query_cutoff=bars.date.iloc[250],
    )
    assert evidence.benchmark_available is True
    assert len(evidence.rows) == len(matches) * len(DISPLAY_HORIZONS)
    assert evidence.rows.observable_at_query.all()
    assert set(evidence.rows.calendar_validation) == {"configured_benchmark"}
    assert evidence.summary.eligible_outcomes.tolist() == [2, 2, 2]
    assert evidence.summary.benchmark_sample_size.tolist() == [2, 2, 2]
    assert evidence.retrieval_identity_digest == retrieval_identity_digest(matches)


def test_future_outcome_mutation_cannot_change_retrieval_identity(
    tmp_path: Path, bars: pd.DataFrame,
) -> None:
    matches = _matches(bars)
    original = build_search_evidence(
        _source(tmp_path / "original", bars), matches,
        query_cutoff=bars.date.iloc[250],
    )
    changed = bars.copy()
    changed.loc[changed.date > bars.date.iloc[100], ["open", "high", "low", "close"]] *= 1.5
    mutated = build_search_evidence(
        _source(tmp_path / "mutated", changed), matches,
        query_cutoff=bars.date.iloc[250],
    )
    assert original.retrieval_identity_digest == mutated.retrieval_identity_digest
    assert original.outcome_digest != mutated.outcome_digest


def test_benchmark_free_search_is_explicitly_degraded(
    tmp_path: Path, bars: pd.DataFrame,
) -> None:
    evidence = build_search_evidence(
        _source(tmp_path / "bars", bars, benchmark=False), _matches(bars),
        query_cutoff=bars.date.iloc[250],
    )
    assert evidence.benchmark_available is False
    assert set(evidence.rows.calendar_validation) == {
        "unavailable_without_benchmark",
    }
    assert evidence.summary.benchmark_sample_size.tolist() == [0, 0, 0]
