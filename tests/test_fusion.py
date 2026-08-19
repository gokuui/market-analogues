from __future__ import annotations

import pandas as pd
import pytest

from market_analogues.fusion import (
    oracle_pool_recall, preselect_per_group, reciprocal_rank_fusion,
)


def test_rank_fusion_is_deterministic_under_input_shuffle() -> None:
    frame = pd.DataFrame({
        "episode_id": ["d", "a", "c", "b"],
        "price": [1.0, 0.0, 3.0, 2.0],
        "volume": [0.0, 3.0, 2.0, 1.0],
    })
    first = reciprocal_rank_fusion(
        frame, ("price", "volume"), pool_size=3,
    ).selected.episode_id.tolist()
    second = reciprocal_rank_fusion(
        frame.sample(frac=1, random_state=9), ("price", "volume"), pool_size=3,
    ).selected.episode_id.tolist()
    assert first == second
    assert {"a", "d"}.issubset(first)


def test_oracle_recall_uses_explicit_deduplicated_membership() -> None:
    frame = pd.DataFrame({
        "episode_id": ["a", "b", "c", "d"],
        "view": [3.0, 0.0, 1.0, 2.0],
        "oracle_selected": [True, False, True, False],
    })
    assert oracle_pool_recall(frame, ("view",), (1, 2, 4)) == {
        "1": 0.0, "2": 0.5, "4": 1.0,
    }


def test_rank_fusion_rejects_invalid_inputs() -> None:
    frame = pd.DataFrame({"episode_id": ["a"], "view": [0.0]})
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion(frame, ("view",), pool_size=0)
    with pytest.raises(ValueError, match="missing"):
        reciprocal_rank_fusion(frame, ("other",), pool_size=1)


def test_preselection_unions_each_groups_independent_view_winners() -> None:
    frame = pd.DataFrame({
        "episode_id": ["a", "b", "c", "d"],
        "symbol": ["x", "x", "y", "y"],
        "price": [0.0, 2.0, 3.0, 1.0],
        "volume": [2.0, 0.0, 1.0, 3.0],
    })
    selected = preselect_per_group(frame, ("price", "volume"), per_view=1)
    assert set(selected.episode_id) == {"a", "b", "c", "d"}
