from __future__ import annotations

import pandas as pd

from market_analogues.adapters import DirectorySource
from market_analogues.config import DatasetSpec
from market_analogues.episodes import build_episode
from market_analogues.types import InstrumentKey, SearchQuery
from market_analogues.view_search import search_view_store
from market_analogues.view_store import build_view_store


def test_persisted_search_is_deterministic_and_temporally_eligible(
    directory_dataset, tmp_path,
) -> None:
    source = DirectorySource(DatasetSpec(
        "test", "directory", directory_dataset, "parquet", timestamp_column="date",
    ))
    quality = pd.DataFrame({
        "symbol": ["AAA", "BBB"], "tier": ["A", "B"], "issues": ["", ""],
    })
    root = tmp_path / "store"
    build_view_store(
        source, quality, root, lookbacks=(63,), stride=5, workers=2,
    )
    query = build_episode(
        source, InstrumentKey("test", "AAA"), "2021-12-31", 63, "dense-v1",
    )
    request = SearchQuery(
        query.key, ("test",), ("A", "B"), 5, minimum_history_gap_bars=20,
    )
    first = search_view_store(
        query, request, root, candidate_pool=12, per_instrument_view=3,
    )
    second = search_view_store(
        query, request, root, candidate_pool=12, per_instrument_view=3,
    )
    assert first.manifest_digest == second.manifest_digest
    assert [(hit.episode_id, hit.fusion_score) for hit in first.hits] == [
        (hit.episode_id, hit.fusion_score) for hit in second.hits
    ]
    assert first.shards_loaded == 2
    assert first.rows_considered > 0
    assert len(first.hits) <= 12
    assert all(hit.cutoff < query.key.cutoff for hit in first.hits)
    same_symbol = [hit for hit in first.hits if hit.instrument == query.key.instrument]
    assert all(hit.cutoff < query.bars.timestamp.iloc[0] for hit in same_symbol)
