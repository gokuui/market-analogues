from __future__ import annotations

import pandas as pd

from market_analogues.adapters import DirectorySource
from market_analogues.config import DatasetSpec
from market_analogues.episodes import build_episode, candidate_positions
from market_analogues.types import InstrumentKey


def _source(path) -> DirectorySource:
    return DirectorySource(DatasetSpec(
        dataset_id="test", adapter="directory", path=path, format="parquet",
        timestamp_column="date",
    ))


def test_candidate_positions_respect_minimum_and_stride() -> None:
    assert candidate_positions(125) == []
    assert candidate_positions(140, minimum=126, stride=5) == [125, 130, 135]


def test_episode_is_cut_off_strictly_and_has_stable_id(directory_dataset) -> None:
    source = _source(directory_dataset)
    key = InstrumentKey("test", "AAA")
    cutoff = pd.Timestamp("2020-10-01")
    first = build_episode(source, key, cutoff, 126, "dense-v1")
    second = build_episode(source, key, cutoff, 126, "dense-v1")

    assert len(first.bars) == 126
    assert first.bars.timestamp.max() <= cutoff
    assert first.key.id == second.key.id
    assert first.key.cutoff == first.bars.timestamp.iloc[-1]
