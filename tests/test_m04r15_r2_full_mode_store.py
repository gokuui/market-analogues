from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.m04r import m04r15_r2_bounded_real_poc as bounded
from experiments.m04r import m04r15_r2_full_mode_store as full
from market_analogues.future_mode_store import ArrowPathIndex


def _path_row(episode: str, step: int, level: float) -> dict[str, object]:
    return {
        "benchmark_relative_close_return": level + step / 1000,
        "close_return": level + step / 100,
        "contract_digest": "contract", "cutoff": "2020-01-31",
        "episode_id": episode, "expected_session_match": True,
        "source_content_digest": "content",
        "source_fingerprint": f"fingerprint-{episode}", "step": step,
        "timestamp": f"2020-{2 + (step - 1) // 28:02d}-{(step - 1) % 28 + 1:02d}",
    }


def test_selection_summary_is_exactly_the_bounded_poc_shape() -> None:
    from market_analogues.future_modes import Member, PreparedPath, pairwise_l1, select_modes
    paths = tuple(PreparedPath(
        Member(index + 1, f"e{index}", f"S{index}", "2020-01-31", "f"),
        (level,) * 60, tuple(f"t{step}" for step in range(60)),
    ) for index, level in enumerate((-1.1, -1.0, -.9, .9, 1.0, 1.1)))
    selection = select_modes(
        pairwise_l1(paths), [path.member.key for path in paths],
        [f"20{index:02d}-Q1" for index in range(6)],
        contract_digest="a" * 64, query_case_id="q", view_id="v", replicates=16,
    )
    assert full._selection_summary(selection, paths) == bounded._selection_summary(
        selection, paths,
    )


@pytest.mark.parametrize("eligible_count", [2, 20])
def test_query_result_has_two_frozen_views_and_outcome_blind_cohort(
    tmp_path: Path, eligible_count: int,
) -> None:
    path = tmp_path / "paths.parquet"
    episodes = tuple(f"episode-{index}" for index in range(20))
    pd.DataFrame([
        _path_row(episode, step, -1.0 if index < 10 else 1.0)
        for index, episode in enumerate(episodes) for step in range(1, 61)
    ]).to_parquet(path, index=False)
    full._PATH_INDEX = ArrowPathIndex.load(path)
    full._CONTRACT_DIGEST = "a" * 64
    full._LINKS_BY_QUERY = {"query": tuple({
        "query_case_id": "query", "query_symbol": "QUERY", "match_rank": rank,
        "matched_episode_id": episodes[rank - 1],
        "matched_symbol": f"S{rank}", "matched_cutoff": "2020-01-31",
        "source_fingerprint": f"fingerprint-{episodes[rank - 1]}",
        "outcome_eligibility_json": json.dumps({"60": {
            "eligible": rank <= eligible_count,
            "reason": None if rank <= eligible_count else "incomplete_horizon",
        }}),
    } for rank in range(1, 21))}
    result = full._query_result("query")
    assert result["raw_links"] == result["primary_members"] == 20
    assert result["dependence_exclusions"] == []
    assert tuple(result["views"]) == (
        "absolute_close_return", "benchmark_relative_close_return",
    )
    assert all(view["complete_members"] == eligible_count
               for view in result["views"].values())
    if eligible_count == 2:
        assert all(
            view["selection"] == {
                "status": "abstain_insufficient_complete_primary_members",
                "selected_k": 0, "medoid_episode_ids": [],
                "member_to_mode": {}, "candidates": [],
            }
            for view in result["views"].values()
        )


def test_coverage_counts_each_ineligible_member_once_across_two_views() -> None:
    value = {
        "query_case_id": "q", "raw_links": 20, "primary_members": 18,
        "dependence_exclusions": [["x", "duplicate_matched_symbol"], ["y", "query_symbol_memory"]],
        "views": {
            name: {"ineligible_members": [["z", "source_end"]], "invalid_members": [],
                   "selection": {"status": "one_mode_fallback", "selected_k": 1}}
            for name in ("absolute_close_return", "benchmark_relative_close_return")
        },
    }
    coverage = full._coverage([value])
    assert coverage["query_count"] == 1
    assert coverage["raw_link_count"] == 20
    assert coverage["dependence_exclusion_count"] == 2
    assert coverage["ineligible_member_occurrences"] == 1
    assert coverage["mode_status_counts"] == {"one_mode_fallback": 2}
