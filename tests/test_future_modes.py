from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from math import fsum

import pytest
from sklearn.metrics import adjusted_rand_score

from market_analogues.future_modes import (
    FutureModeError, Member, PreparedPath, adjusted_rand_index,
    bootstrap_stability, calendar_quarter, mean_silhouette, pairwise_l1, pam,
    prepare_paths, select_modes, select_primary_members,
)


def member(rank: int, episode: str, symbol: str) -> dict:
    return {
        "match_rank": rank, "matched_episode_id": episode,
        "matched_symbol": symbol, "matched_cutoff": "2020-01-02",
        "source_fingerprint": "source-1",
    }


def rows(episode: str, values: list[float], *, field: str = "close_return") -> list[dict]:
    return [{
        "step": step, "episode_id": episode, "cutoff": "2020-01-02",
        "timestamp": f"2020-04-{step:02d}", "expected_session_match": True,
        "contract_digest": "contract", "source_content_digest": "content",
        "source_fingerprint": "source-1", field: value,
    } for step, value in enumerate(values, 1)]


def prepared(values: list[list[float]]):
    raw = [member(i + 1, f"e{i}", f"S{i}") for i in range(len(values))]
    primary, excluded = select_primary_members("Q", raw)
    paths, invalid = prepare_paths(
        primary, {f"e{i}": rows(f"e{i}", value) for i, value in enumerate(values)},
        value_field="close_return", horizon=len(values[0]),
    )
    assert not excluded and not invalid
    return paths


def brute_objective(matrix, medoids):
    return fsum(min(row[index] for index in medoids) for row in matrix)


def test_member_selection_is_outcome_blind_rank_ordered_and_discloses_exclusions() -> None:
    raw = [
        member(4, "same", "Q"), member(3, "dup-late", "A"),
        member(1, "first", "A"), member(2, "other", "B"),
    ]
    primary, excluded = select_primary_members("Q", list(reversed(raw)))
    assert [(item.match_rank, item.episode_id) for item in primary] == [(1, "first"), (2, "other")]
    assert [(item.member.episode_id, item.reason) for item in excluded] == [
        ("dup-late", "duplicate_matched_symbol"), ("same", "query_symbol_memory"),
    ]


def test_member_selection_rejects_duplicate_rank_and_episode() -> None:
    with pytest.raises(FutureModeError, match="rank"):
        select_primary_members("Q", [member(1, "a", "A"), member(1, "b", "B")])
    with pytest.raises(FutureModeError, match="episode"):
        select_primary_members("Q", [member(1, "a", "A"), member(2, "a", "B")])


def test_path_preparation_requires_every_bound_expected_finite_step() -> None:
    primary, _ = select_primary_members("Q", [member(1, "ok", "A"), member(2, "bad", "B")])
    path_rows = {"ok": rows("ok", [0.1, 0.2, 0.3]), "bad": rows("bad", [0.1, 0.2, 0.3])}
    path_rows["bad"][1]["expected_session_match"] = False
    path_rows["bad"][2]["close_return"] = float("nan")
    complete, invalid = prepare_paths(primary, path_rows, value_field="close_return", horizon=3)
    assert [item.member.episode_id for item in complete] == ["ok"]
    assert invalid[0].member.episode_id == "bad"
    assert invalid[0].reason == "nonfinite_value+unexpected_session"


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda r: r.pop(), "missing_step"),
        (lambda r: r.append(deepcopy(r[-1])), "duplicate_step"),
        (lambda r: r[0].update(timestamp=r[1]["timestamp"]), "timestamp_sequence"),
        (lambda r: r[0].update(source_fingerprint="other"), "source_fingerprint_binding"),
        (lambda r: r[0].update(cutoff="2019-01-01"), "cutoff_binding"),
    ],
)
def test_path_preparation_rejects_censor_and_binding_failures(mutation, reason: str) -> None:
    primary, _ = select_primary_members("Q", [member(1, "e", "A")])
    values = rows("e", [0.1, 0.2, 0.3])
    mutation(values)
    complete, invalid = prepare_paths(primary, {"e": values}, value_field="close_return", horizon=3)
    assert not complete
    assert reason in invalid[0].reason


def test_pairwise_l1_and_pam_match_bruteforce_for_separated_families() -> None:
    paths = prepared([
        [0.0, 0.0, 0.0], [0.01, 0.0, 0.01], [-0.01, 0.0, 0.0],
        [1.0, 1.0, 1.0], [1.01, 1.0, 1.01], [0.99, 1.0, 1.0],
    ])
    matrix = pairwise_l1(paths)
    keys = [path.member.key for path in paths]
    result = pam(matrix, keys, 2)
    optimum = min(brute_objective(matrix, choice) for choice in combinations(range(6), 2))
    assert result.objective == optimum
    assert result.medoid_indices == (0, 3)
    assert result.labels == (0, 0, 0, 1, 1, 1)
    assert mean_silhouette(matrix, result.labels) > 0.98


def test_pam_and_distance_are_input_order_and_weight_deterministic() -> None:
    paths = prepared([[0.0, 0.0], [0.1, 0.1], [1.0, 1.0], [1.1, 1.1]])
    matrix = pairwise_l1(paths)
    keys = [path.member.key for path in paths]
    first = pam(matrix, keys, 2, weights=[2, 1, 3, 1])
    order = [3, 1, 0, 2]
    shuffled_matrix = [[matrix[left][right] for right in order] for left in order]
    shuffled = pam(
        shuffled_matrix, [keys[index] for index in order], 2,
        weights=[[2, 1, 3, 1][index] for index in order],
    )
    assert {keys[index] for index in first.medoid_indices} == {
        [keys[index] for index in order][medoid] for medoid in shuffled.medoid_indices
    }
    first_members = {keys[i]: first.labels[i] for i in range(4)}
    shuffled_medoids = [order[index] for index in shuffled.medoid_indices]
    normalized = {
        keys[order[i]]: keys[shuffled_medoids[shuffled.labels[i]]]
        for i in range(4)
    }
    assert {key: keys[first.medoid_indices[label]] for key, label in first_members.items()} == normalized


def test_pam_ties_choose_lowest_frozen_member_key() -> None:
    paths = prepared([[0.0, 0.0], [2.0, 2.0]])
    result = pam(pairwise_l1(paths), [path.member.key for path in paths], 1)
    assert result.medoid_indices == (0,)


def test_distance_and_pam_reject_nonfinite_asymmetric_or_invalid_inputs() -> None:
    invalid_path = PreparedPath(
        Member(1, "e", "S", "2020-01-02", "source-1"),
        (0.0, float("inf")), ("2020-01-03", "2020-01-06"),
    )
    with pytest.raises(FutureModeError, match="nonfinite"):
        pairwise_l1([invalid_path])
    with pytest.raises(FutureModeError, match="symmetry"):
        pam([[0.0, 1.0], [2.0, 0.0]], [(1, "a"), (2, "b")], 1)
    with pytest.raises(FutureModeError, match="weights"):
        pam([[0.0]], [(1, "a")], 1, weights=[0.0])


def test_silhouette_single_mode_and_singletons_are_explicit() -> None:
    matrix = ((0.0, 1.0, 4.0), (1.0, 0.0, 3.0), (4.0, 3.0, 0.0))
    assert mean_silhouette(matrix, [0, 0, 0]) == 0.0
    assert mean_silhouette(matrix, [0, 0, 1]) == pytest.approx((0.75 + 2 / 3 + 0.0) / 3)


def test_calendar_quarter_and_adjusted_rand_are_exact() -> None:
    assert calendar_quarter("2024-01-31T00:00:00") == "2024-Q1"
    assert calendar_quarter("2024-12-30") == "2024-Q4"
    with pytest.raises(FutureModeError, match="cutoff"):
        calendar_quarter("not-a-date")
    assert adjusted_rand_index([0, 0, 1, 1], [7, 7, 3, 3]) == 1.0
    assert adjusted_rand_index([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(-0.5)


@pytest.mark.parametrize("left,right", [
    ([0, 0, 0, 1, 1, 2], [1, 1, 0, 0, 2, 2]),
    ([0, 0, 1, 1, 2, 2], [2, 2, 1, 1, 0, 0]),
    ([0, 0, 0, 0], [1, 1, 1, 1]),
])
def test_adjusted_rand_matches_independent_sklearn(left, right) -> None:
    assert adjusted_rand_index(left, right) == pytest.approx(adjusted_rand_score(left, right))


def test_stable_three_family_selection_passes_all_frozen_gates() -> None:
    paths = prepared([
        [-1.01] * 12, [-1.0] * 12, [-0.99] * 12,
        [-0.01] * 12, [0.0] * 12, [0.01] * 12,
        [0.99] * 12, [1.0] * 12, [1.01] * 12,
    ])
    matrix = pairwise_l1(paths)
    keys = [path.member.key for path in paths]
    blocks = [f"20{20 + index // 4}-Q{index % 4 + 1}" for index in range(9)]
    result = select_modes(
        matrix, keys, blocks, contract_digest="a" * 64,
        query_case_id="query", view_id="absolute", replicates=64,
    )
    assert result.status == "stable_multiple_modes"
    assert result.selected_k == 3
    assert result.medoid_indices == (1, 4, 7)
    selected = next(value for value in result.candidates if value.k == 3)
    assert selected.accepted is True
    assert selected.stability is not None
    assert selected.stability.valid_replicates >= 52
    assert selected.stability.median_adjusted_rand_index == 1.0


def test_date_confounded_modes_fail_block_diversity_and_fall_back() -> None:
    paths = prepared([
        [-1.01] * 8, [-1.0] * 8, [-0.99] * 8,
        [0.99] * 8, [1.0] * 8, [1.01] * 8,
    ])
    matrix = pairwise_l1(paths)
    keys = [path.member.key for path in paths]
    base = pam(matrix, keys, 2)
    stability = bootstrap_stability(
        matrix, keys, ["2020-Q1"] * 3 + ["2020-Q2"] * 3, base,
        contract_digest="b" * 64, query_case_id="query", view_id="absolute",
        replicates=32,
    )
    assert stability.valid_replicates == 0
    assert stability.median_adjusted_rand_index is None
    result = select_modes(
        matrix, keys, ["2020-Q1"] * 3 + ["2020-Q2"] * 3,
        contract_digest="b" * 64, query_case_id="query", view_id="absolute",
        replicates=32,
    )
    assert result.status == "one_mode_fallback"
    assert result.selected_k == 1
    assert "fewer_than_80_percent_valid_block_bootstraps" in result.candidates[0].rejection_reasons


def test_tight_single_family_falls_back_and_tiny_cohort_abstains() -> None:
    paths = prepared([[0.0] * 6 for _ in range(6)])
    result = select_modes(
        pairwise_l1(paths), [path.member.key for path in paths],
        [f"202{i}-Q1" for i in range(6)], contract_digest="c" * 64,
        query_case_id="query", view_id="absolute", replicates=16,
    )
    assert result.status == "one_mode_fallback"
    assert result.selected_k == 1
    assert "mode_smaller_than_3" in result.candidates[0].rejection_reasons
    tiny = prepared([[0.0] * 6, [1.0] * 6])
    abstain = select_modes(
        pairwise_l1(tiny), [path.member.key for path in tiny], ["2020-Q1", "2021-Q1"],
        contract_digest="d" * 64, query_case_id="query", view_id="absolute",
        replicates=16,
    )
    assert abstain.status == "abstain_insufficient_complete_primary_members"
    assert abstain.selected_k == 0


def test_block_stability_is_exactly_repeatable_and_order_invariant() -> None:
    paths = prepared([
        [-1.0] * 5, [-0.9] * 5, [-1.1] * 5,
        [1.0] * 5, [0.9] * 5, [1.1] * 5,
    ])
    matrix = pairwise_l1(paths)
    keys = [path.member.key for path in paths]
    blocks = ["2019-Q1", "2020-Q1", "2021-Q1", "2019-Q3", "2020-Q3", "2021-Q3"]
    base = pam(matrix, keys, 2)
    first = bootstrap_stability(
        matrix, keys, blocks, base, contract_digest="e" * 64,
        query_case_id="query", view_id="relative", replicates=32,
    )
    second = bootstrap_stability(
        matrix, keys, blocks, base, contract_digest="e" * 64,
        query_case_id="query", view_id="relative", replicates=32,
    )
    assert first == second
    order = [5, 2, 4, 1, 3, 0]
    reordered_matrix = [[matrix[left][right] for right in order] for left in order]
    reordered_base = pam(reordered_matrix, [keys[i] for i in order], 2)
    reordered = bootstrap_stability(
        reordered_matrix, [keys[i] for i in order], [blocks[i] for i in order],
        reordered_base, contract_digest="e" * 64, query_case_id="query",
        view_id="relative", replicates=32,
    )
    assert first.adjusted_rand_indices == reordered.adjusted_rand_indices
    assert first.median_adjusted_rand_index == reordered.median_adjusted_rand_index
