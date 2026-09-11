from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import math

import numpy as np
import pytest

from market_analogues.adequacy_localization import (
    LocalizationError, PreparedConditionalDraw, PreparedLocalization, assemble_replicate_chunks, candidate_specificity_ranks,
    chart_distances, conditional_draw, conditional_priority,
    deterministic_null_summary, effect_summary, empirical_chart_transform,
    entropy_breadth, episode_parities, episode_specificity, equal_episode_mean,
    localization_decision, plus_one_lower_p, query_cohesion,
    required_improved_count,
)
from market_analogues.balanced_partition import midrank_transform


def charts(column):
    result = np.zeros((len(column), 141))
    result[:, 0] = column
    return result


def test_empirical_mapping_exact_ties_gaps_range_constants_and_signed_zero():
    queries = charts([-0., 0., 2., 4.])
    candidates = charts([-100., -0., 0., 1., 2., 3., 4., 100.])
    candidates[:, 1] = [-900, 0, -0., 900, 0, 0, 0, 0]
    qz, ez, constants = empirical_chart_transform(queries, candidates)
    np.testing.assert_array_equal(qz[:, 0], [-2 / 3, -2 / 3, 1 / 3, 1.])
    np.testing.assert_array_equal(ez[:, 0], [-1, -2 / 3, -2 / 3, 0, 1 / 3, 2 / 3, 1, 1])
    assert constants == tuple(range(1, 141))
    np.testing.assert_array_equal(ez[:, 1], [-1, 0, 0, 1, 0, 0, 0, 0])
    assert not np.signbit(qz[qz == 0]).any()
    assert not np.signbit(ez[ez == 0]).any()
    np.testing.assert_array_equal(qz, midrank_transform(queries)[0])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_geometry_refused(bad):
    values = charts([0., bad])
    with pytest.raises(LocalizationError):
        empirical_chart_transform(values, charts([1.]))
    with pytest.raises(LocalizationError):
        empirical_chart_transform(charts([0., 1.]), values)
    with pytest.raises(LocalizationError):
        chart_distances(values, charts([1.]))


def test_rms_direct_oracle_and_chunk_independence():
    values = np.random.default_rng(4).uniform(-1, 1, (7, 141))
    actual = chart_distances(values, values)
    oracle = np.asarray([[math.sqrt(math.fsum(float(x - y) * float(x - y) for x, y in zip(a, b)) / 141) for b in values] for a in values])
    np.testing.assert_array_equal(actual, oracle)
    np.testing.assert_array_equal(actual, actual.T)
    np.testing.assert_array_equal(actual.diagonal(), np.zeros(7))
    np.testing.assert_array_equal(actual, np.concatenate([chart_distances(chunk, values) for chunk in np.array_split(values, 3)]))
    one_dimension = chart_distances(charts([0.]), charts([1.]))
    assert one_dimension[0, 0] == math.sqrt(1 / 141)


def test_specificity_full_eligible_cohort_exact_ties_and_query_denominators():
    distances = np.asarray([[1., 1., 9., 0.], [1., 2., 3., 4.], [1., 2., 3., 4.]])
    eligible = np.asarray([[True, True, True, False], [False, True, False, False], [False] * 4])
    ranks = candidate_specificity_ranks(distances, eligible)
    np.testing.assert_array_equal(ranks[0, :3], [1 / 3, 1 / 3, 5 / 6])
    assert ranks[1, 1] == .5
    assert np.isnan(ranks[0, 3]) and np.isnan(ranks[2]).all()
    assert episode_specificity(ranks, [1, 0], 1) == (1 / 3 + .5) / 2
    with pytest.raises(LocalizationError):
        episode_specificity(ranks, [0], 3)
    with pytest.raises(LocalizationError):
        candidate_specificity_ranks(distances, eligible.astype(int))


def test_specificity_denominator_keeps_all_369_cohort_episodes() -> None:
    distances = np.arange(369, dtype=np.float64)[None, :]
    eligible = np.ones((1, 369), dtype=np.bool_)
    ranks = candidate_specificity_ranks(distances, eligible)
    assert ranks.shape == (1, 369)
    assert ranks[0, 0] == .5 / 369
    assert ranks[0, 356] == 356.5 / 369
    # The 12 support-excluded episodes remain in the rank denominator.
    assert ranks[0, 357] == 357.5 / 369
    assert ranks[0, 368] == 368.5 / 369


def test_recurrence_does_not_weight_episodes_and_cohesion_uses_pairs_once():
    distances = np.asarray([[0., 1., 2.], [1., 0., 6.], [2., 6., 0.]])
    assert query_cohesion(distances, [2, 0, 1]) == 3
    assert query_cohesion(distances, [0, 1]) == 1
    assert equal_episode_mean([3, 1]) == 2  # Not pair-weighted (2.5).
    with pytest.raises(LocalizationError):
        query_cohesion(distances, [0])
    with pytest.raises(LocalizationError):
        query_cohesion(distances, [0, 0])


def test_entropy_effective_k8_count():
    assert entropy_breadth([0, 1, 2, 3], [0, 0, 0, 0]) == .25
    assert entropy_breadth([0, 1, 2, 3], [0, 1, 2, 3]) == 1
    assert entropy_breadth([0], [7]) == 1
    with pytest.raises(LocalizationError):
        entropy_breadth([0], [8])


def test_priority_exact_wire_format_and_full_hash_reproducibility():
    parts = [b"R1B-B2-conditional-priority-v1", bytes.fromhex("ab" * 32), b"N0-N1-common", b"episode", (17).to_bytes(4, "big"), b"episode", b"query"]
    expected = sha256(b"".join(len(part).to_bytes(4, "big") + part for part in parts)).digest()
    assert conditional_priority("ab" * 32, 17, "episode", "query") == expected
    assert len(expected) == 32
    assert conditional_priority("ab" * 32, 17, "e1", "q", shared_query=True) == conditional_priority("ab" * 32, 17, "e2", "q", shared_query=True)
    assert conditional_priority("ab" * 32, 17, "e1", "q") != conditional_priority("ab" * 32, 17, "e2", "q")
    assert conditional_priority("ab" * 32, 17, "ab", "c") != conditional_priority("ab" * 32, 17, "a", "bc")
    for invalid in (-1, 2**32, True):
        with pytest.raises(LocalizationError):
            conditional_priority("ab" * 32, invalid, "episode", "query")


def test_conditional_draw_exact_cells_observed_feasible_and_order_invariant():
    kwargs = dict(eligible=list(range(8)), observed=[0, 1, 4], cells=[0] * 4 + [1] * 4,
                  query_ids=[f"q{i}" for i in range(8)], episode_id="e", contract_digest="ab" * 32)
    selections = []
    for replica in range(200):
        draw = conditional_draw(**kwargs, replicate=replica)
        assert len(draw) == 3 and len(set(draw)) == 3
        assert Counter(kwargs["cells"][i] for i in draw) == {0: 2, 1: 1}
        assert draw == conditional_draw(**{**kwargs, "eligible": list(reversed(range(8))), "observed": [4, 1, 0]}, replicate=replica)
        selections.append(draw)
    assert (0, 1, 4) in selections
    # Every one of the 6*4 admissible choices occurs for this fixed test stream.
    assert len(set(selections)) == 24
    with pytest.raises(LocalizationError):
        conditional_draw(**{**kwargs, "eligible": [0, 1]}, replicate=0)


def test_n0_n1_common_priority_restricts_same_order_within_refined_cells():
    base = dict(eligible=list(range(8)), observed=[0, 4], query_ids=[f"q{i}" for i in range(8)],
                episode_id="e", contract_digest="ab" * 32, replicate=0)
    order = sorted(range(8), key=lambda i: (conditional_priority("ab" * 32, 0, "e", f"q{i}"), f"q{i}"))
    n0 = conditional_draw(**base, cells=[0] * 8)
    n1 = conditional_draw(**base, cells=[0] * 4 + [1] * 4)
    assert n0 == tuple(sorted(order[:2]))
    assert n1 == tuple(sorted([next(i for i in order if i < 4), next(i for i in order if i >= 4)]))


@pytest.mark.parametrize("shared_query", [False, True])
def test_prepared_draw_byte_identity_with_scalar_and_chunk_restart(shared_query):
    arguments = dict(eligible=list(range(15)), observed=[0, 1, 6], cells=[0] * 5 + [1] * 5 + [2] * 5,
                     query_ids=[f"q{index:02d}" for index in range(15)], episode_id="e", contract_digest="ab" * 32,
                     shared_query=shared_query)
    prepared = PreparedConditionalDraw(**arguments)
    oracle = np.asarray([conditional_draw(**arguments, replicate=i) for i in range(48)], dtype=np.int64)
    np.testing.assert_array_equal(prepared.batch(0, 48), oracle)
    with ThreadPoolExecutor(max_workers=12) as pool:
        chunks = list(pool.map(lambda start: prepared.batch(start, start + 4), range(0, 48, 4)))
    assert sha256(np.concatenate(chunks).tobytes()).digest() == sha256(oracle.tobytes()).digest()
    with pytest.raises(LocalizationError):
        prepared.batch(4, 4)


def test_plus_one_threshold_ties_and_exact_sixty_percent_boundary():
    values = np.ones(4096)
    values[:39] = 0
    assert plus_one_lower_p(0, values) == 40 / 4097 <= .01
    values[39] = 0
    assert plus_one_lower_p(0, values) == 41 / 4097 > .01
    assert plus_one_lower_p(1, values) == 1
    assert required_improved_count(357) == 215
    assert required_improved_count(5) == 3
    assert required_improved_count(6) == 4


def test_effect_practical_threshold_equality_and_count_boundary():
    summary = effect_summary(np.asarray([[.95, 0.]] * 5), np.asarray([[1., .05]] * 5))
    assert summary.practical_pass
    assert summary.specificity_improvement == .05
    obs = np.ones((357, 2))
    means = np.ones((357, 2))
    obs[:214] = 0
    assert not effect_summary(obs, means).practical_pass
    obs[214] = 0
    assert effect_summary(obs, means).practical_pass
    zero = effect_summary(np.zeros((2, 2)), np.zeros((2, 2)))
    assert zero.cohesion_relative_improvement is None and not zero.practical_pass


def test_invalid_null_entries_cannot_hide_in_valid_means():
    table = np.asarray([[[-.1, .2]], [[.9, .8]]])
    with pytest.raises(LocalizationError, match="nonnegative"):
        deterministic_null_summary(table)
    table = np.asarray([[[.1, -.1]], [[.9, 1.1]]])
    with pytest.raises(LocalizationError, match="nonnegative"):
        deterministic_null_summary(table)


def test_assembly_restart_holes_overlap_and_serial_twelve_worker_determinism():
    def make_chunk(start):
        return start, np.asarray([[[start / 12, .25], [.5, .75]]])
    with ThreadPoolExecutor(max_workers=12) as pool:
        chunks = list(pool.map(make_chunk, range(12)))
    serial = assemble_replicate_chunks([make_chunk(i) for i in range(12)], replicates=12, episodes=2)
    parallel = assemble_replicate_chunks(list(reversed(chunks)), replicates=12, episodes=2)
    np.testing.assert_array_equal(serial, parallel)
    for a, b in zip(deterministic_null_summary(serial), deterministic_null_summary(parallel)):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(LocalizationError, match="gaps"):
        assemble_replicate_chunks(chunks[:-1], replicates=12, episodes=2)
    with pytest.raises(LocalizationError, match="overlap"):
        assemble_replicate_chunks(chunks + chunks[:1], replicates=12, episodes=2)


def test_prepared_batches_identical_to_scalar_oracle_and_twelve_workers():
    generator = np.random.default_rng(199)
    queries = generator.uniform(-1, 1, (20, 141))
    candidates = generator.uniform(-1, 1, (3, 141))
    dqq = chart_distances(queries, queries)
    dqe = chart_distances(queries, candidates)
    eligible = np.ones((20, 3), dtype=bool)
    ranks = candidate_specificity_ranks(dqe, eligible)
    labels = np.arange(20) % 8
    prepared = PreparedLocalization(dqq, ranks, eligible, labels)
    selections = np.asarray([generator.choice(20, 5, replace=False) for _ in range(48)])
    oracle = np.asarray([[query_cohesion(dqq, selected), episode_specificity(ranks, selected, 1)] for selected in selections])
    np.testing.assert_array_equal(prepared.evaluate(1, selections), oracle)
    np.testing.assert_array_equal(prepared.breadth(selections), [entropy_breadth(row, labels) for row in selections])
    chunks = np.array_split(selections, 12)
    with ThreadPoolExecutor(max_workers=12) as pool:
        parallel = np.concatenate(list(pool.map(lambda rows: prepared.evaluate(1, rows), chunks)))
    assert sha256(parallel.tobytes()).digest() == sha256(oracle.tobytes()).digest()
    # Validated state cannot be changed through the original caller arrays.
    dqq[:] = 0
    ranks[:] = np.nan
    eligible[:] = False
    labels[:] = 99
    np.testing.assert_array_equal(prepared.evaluate(1, selections), oracle)
    for bad in (np.asarray([[0, 0]]), np.asarray([[0, 21]]), np.asarray([[0., 1.]])):
        with pytest.raises(LocalizationError):
            prepared.evaluate(1, bad)


def test_prepared_refuses_bad_geometry_and_ineligible_selection():
    dqq = np.asarray([[0., 1.], [1., 0.]])
    eligible = np.asarray([[True], [False]])
    ranks = np.asarray([[.5], [np.nan]])
    prepared = PreparedLocalization(dqq, ranks, eligible, [0, 1])
    with pytest.raises(LocalizationError, match="ineligible"):
        prepared.evaluate(0, np.asarray([[0, 1]]))
    with pytest.raises(LocalizationError, match="symmetric"):
        PreparedLocalization(np.asarray([[0., 1.], [2., 0.]]), ranks, eligible, [0, 1])
    with pytest.raises(LocalizationError, match="percentiles"):
        PreparedLocalization(dqq, np.asarray([[.5], [.5]]), eligible, [0, 1])


@pytest.fixture(scope="module")
def valid_decision():
    ids = [f"{index:024x}" for index in range(357)]
    observed = np.tile([.2, .2], (357, 1))
    null = np.broadcast_to([.5, .5], (4096, 357, 2)).copy()
    return dict(
        episode_ids=ids, observed=observed, n0_null=null, n1_null=null,
        shared_query_n0_null=null, shared_query_n1_null=null,
        prerequisite_verified=True,
    )


def test_valid_decision_and_episodes_split_not_queries(valid_decision):
    result = localization_decision(**valid_decision)
    assert result.status == "structurally_localized"
    assert result.n1_pvalues == (1 / 4097, 1 / 4097)
    parity = episode_parities(valid_decision["episode_ids"])
    np.testing.assert_array_equal(parity, np.arange(357) % 2)


def test_episode_parity_refuses_noncanonical_or_contract_digest_width_ids():
    with pytest.raises(LocalizationError, match="24-hex"):
        episode_parities(["a" * 64])
    with pytest.raises(LocalizationError, match="24-hex"):
        episode_parities(["A" * 24])


@pytest.mark.parametrize("mutation, expected", [
    ("missing_replicate", "unresolved"), ("unverified", "unresolved"),
    ("secondary_k", "unresolved"), ("n0_failure", "not_established"),
    ("shared_failure", "unresolved"), ("zero_cohesion", "not_established"),
    ("n1_p_failure", "not_established"), ("split_failure", "unresolved"),
    ("loeo_failure", "unresolved"),
])
def test_decision_taxonomy(valid_decision, mutation, expected):
    case = dict(valid_decision)
    if mutation == "missing_replicate":
        case["n1_null"] = case["n1_null"][:-1]
    elif mutation == "unverified":
        case["prerequisite_verified"] = False
    elif mutation == "secondary_k":
        case["primary_k"] = 12
    elif mutation == "n0_failure":
        case["n0_null"] = np.broadcast_to([.21, .21], (4096, 357, 2))
    elif mutation == "shared_failure":
        case["shared_query_n1_null"] = np.broadcast_to([.1, .1], (4096, 357, 2))
    elif mutation == "zero_cohesion":
        case["observed"] = np.zeros((357, 2))
        case["n1_null"] = np.zeros((4096, 357, 2))
    elif mutation == "n1_p_failure":
        case["n1_null"] = case["n1_null"].copy()
        case["n1_null"][:40] = .1
    elif mutation == "split_failure":
        case["observed"] = case["observed"].copy()
        case["observed"][::2] = .49
    elif mutation == "loeo_failure":
        case["observed"] = np.tile([.5, .5], (357, 1))
        case["observed"][0] = [.1, .1]
    result = localization_decision(**case)
    assert result.status == expected, result
    if mutation == "loeo_failure":
        assert "leave-one-episode-out positive effects failed" in result.reasons


@pytest.mark.parametrize("name", ["shared_query_n0_null", "shared_query_n1_null"])
def test_each_shared_query_design_is_independently_required(valid_decision, name):
    case = dict(valid_decision)
    case[name] = np.broadcast_to([.1, .1], (4096, 357, 2))
    result = localization_decision(**case)
    assert result.status == "unresolved"
    assert any("shared-query" in reason for reason in result.reasons)
