from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import m04r14_t14_10_wf03d_exclusion_audit as subject


def test_groups_maps_all_four_frozen_methods() -> None:
    groups = subject._groups(
        {"retrieval": {"matches": [{"episode_id": "c"}]}},
        {"matches": [{"episode_id": "p"}]},
        {
            "random_neighbors": [{"episode_id": "r"}],
            "rank_neighbors": [{"episode_id": "v"}],
        },
    )
    assert list(groups) == [
        "composite", "price_only", "deterministic_random",
        "recent_return_volatility",
    ]
    assert [groups[key][0]["episode_id"] for key in groups] == ["c", "p", "r", "v"]


def test_groups_rejects_missing_method() -> None:
    with pytest.raises(subject.ExclusionAuditError, match="layout"):
        subject._groups(
            {"retrieval": {"matches": []}}, {"matches": []},
            {"random_neighbors": []},
        )
