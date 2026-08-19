import pandas as pd

from market_analogues.types import EpisodeKey, InstrumentKey


def test_episode_id_is_deterministic():
    key = EpisodeKey(InstrumentKey("x", "ABC"), pd.Timestamp("2024-01-01"), 252, "v1")
    assert key.id == key.id
    assert len(key.id) == 24

