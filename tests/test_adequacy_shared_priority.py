"""Synthetic proofs for the B0-05 shared-priority primitive.

These tests intentionally contain no repository data paths.  They test the
mathematical null mechanics before an experiment can bind it to evidence.
"""

from __future__ import annotations

from itertools import permutations

import pandas as pd
import pytest

from market_analogues.adequacy_shared_priority import (
    EPISODE_DOMAIN,
    SYMBOL_DOMAIN,
    SelectionConfig,
    SharedPriorityCandidate,
    SharedPriorityError,
    episode_priority,
    evaluate_query_selections,
    exact_bucket_order,
    global_order,
    greedy_accept,
    hierarchical_order,
    intervals_overlap,
    priority_digest,
    priority_preimage,
    select_global,
    select_hierarchical,
    selection_semantic_digest,
    symbol_priority,
)
from market_analogues.certified_packed_search import (
    CompactScoredCandidate,
    _select_compact_scored,
)
from market_analogues.types import AnalogueMatch, EpisodeKey, InstrumentKey, SearchQuery


def row(episode: str, symbol: str, start: int, cutoff: int) -> SharedPriorityCandidate:
    return SharedPriorityCandidate(episode, symbol, start, cutoff)


def digest(rank: int, suffix: int = 0) -> bytes:
    """A synthetic full SHA-sized digest whose lexicographic rank is explicit."""

    return rank.to_bytes(30, "big") + suffix.to_bytes(2, "big")


def test_hash_preimage_and_digest_golden_vectors_are_binary_unambiguous() -> None:
    assert priority_preimage(
        domain=EPISODE_DOMAIN, seed=947221, replicate=17, identifier="episode-α",
    ).hex() == (
        "00276d61726b65742d616e616c6f677565732f7231622f62302d30352f682d"
        "657069736f64652f763100000000000e741500000011000a657069736f64652dceb1"
    )
    assert episode_priority(seed=947221, replicate=17, episode_id="episode-α").hex() == (
        "7298340f24b15fef861b5a6f45deddd5f4ba8f547e764072dde1e13cc12c5777"
    )
    assert priority_preimage(
        domain=SYMBOL_DOMAIN, seed=947221, replicate=17, identifier="SYM",
    ).hex() == (
        "00266d61726b65742d616e616c6f677565732f7231622f62302d30352f682d"
        "73796d626f6c2f763100000000000e741500000011000353594d"
    )
    assert symbol_priority(seed=947221, replicate=17, symbol_id="SYM").hex() == (
        "cd7b16d905810e1a97b14fcfffd5102b10ad6c2c8b8f6903ce128c2fa3c3b05b"
    )
    assert priority_digest(
        domain=EPISODE_DOMAIN, seed=947222, replicate=17, identifier="episode-α",
    ) != episode_priority(seed=947221, replicate=17, episode_id="episode-α")
    assert priority_digest(
        domain=SYMBOL_DOMAIN, seed=947221, replicate=17, identifier="episode-α",
    ) != episode_priority(seed=947221, replicate=17, episode_id="episode-α")


@pytest.mark.parametrize("kwargs", [
    {"domain": "", "seed": 0, "replicate": 0, "identifier": "x"},
    {"domain": "d", "seed": -1, "replicate": 0, "identifier": "x"},
    {"domain": "d", "seed": 2**64, "replicate": 0, "identifier": "x"},
    {"domain": "d", "seed": 0, "replicate": -1, "identifier": "x"},
    {"domain": "d", "seed": 0, "replicate": 2**32, "identifier": "x"},
    {"domain": "d", "seed": 0, "replicate": 0, "identifier": "x" * 65536},
])
def test_hash_contract_fails_closed_for_ambiguous_or_out_of_range_inputs(kwargs: dict[str, object]) -> None:
    with pytest.raises(SharedPriorityError):
        priority_preimage(**kwargs)  # type: ignore[arg-type]


def test_inclusive_overlap_cap_three_and_top_k_are_exact_and_configurable() -> None:
    rows = (
        row("a0", "A", 0, 2), row("a1", "A", 2, 4),  # inclusive overlap
        row("a2", "A", 3, 5), row("a3", "A", 6, 8),
        row("a4", "A", 9, 11), row("b0", "B", 0, 2),
    )
    assert intervals_overlap(rows[0], rows[1])
    assert tuple(item.episode_id for item in greedy_accept(
        rows, config=SelectionConfig(top_k=4, max_per_symbol=3),
    )) == ("a0", "a2", "a3", "b0")
    with pytest.raises(SharedPriorityError, match="cannot satisfy"):
        greedy_accept(rows[:2], config=SelectionConfig(top_k=3, max_per_symbol=3))


def test_bucket_order_is_identical_to_full_256_bit_sort_on_exhaustive_tiny_orders() -> None:
    rows = tuple(row(f"e{i}", "A" if i < 3 else "B", i * 3, i * 3 + 1) for i in range(5))
    # Include same 16-bit prefixes and a complete-digest collision so both
    # full-digest and ID tie-break pieces are exercised.
    priorities = {
        "e0": bytes.fromhex("00010000" + "00" * 28),
        "e1": bytes.fromhex("0001ffff" + "00" * 28),
        "e2": bytes.fromhex("0001ffff" + "00" * 28),
        "e3": bytes.fromhex("0002" + "00" * 30),
        "e4": bytes.fromhex("ffff" + "00" * 30),
    }
    provider = lambda candidate: priorities[candidate.episode_id]
    expected = tuple(sorted(rows, key=lambda candidate: (provider(candidate), candidate.episode_id)))
    observed = exact_bucket_order(rows, key_for=lambda candidate: (provider(candidate), candidate.episode_id))
    assert observed == expected
    assert tuple(item.episode_id for item in observed) == ("e0", "e1", "e2", "e3", "e4")
    # Every permutation must preserve the same result; this proves the bucket
    # implementation does not accidentally depend on arrival order.
    for permuted in permutations(rows):
        assert exact_bucket_order(
            permuted, key_for=lambda candidate: (provider(candidate), candidate.episode_id),
        ) == expected


def _production_ids(
    ordered: tuple[SharedPriorityCandidate, ...], top_k: int, cap: int,
) -> tuple[str, ...]:
    instruments = {
        symbol: InstrumentKey("synthetic", symbol)
        for symbol in {candidate.symbol_id for candidate in ordered}
    }
    scored: list[CompactScoredCandidate] = []
    episode_id_by_key: dict[str, str] = {}
    for distance, candidate in enumerate(ordered):
        key = EpisodeKey(
            instruments[candidate.symbol_id],
            pd.Timestamp("2000-01-01") + pd.Timedelta(days=distance),
            252,
            "synthetic-v1",
        )
        item = CompactScoredCandidate(
            AnalogueMatch(key, float(distance), {}),
            instruments[candidate.symbol_id],
            candidate.start_coordinate,
            candidate.cutoff_coordinate,
        )
        scored.append(item)
        episode_id_by_key[key.id] = candidate.episode_id
    query_key = EpisodeKey(
        InstrumentKey("synthetic", "QUERY"), pd.Timestamp("2010-01-01"), 252, "synthetic-v1",
    )
    selected = _select_compact_scored(
        scored, SearchQuery(query_key, top_k=top_k, max_per_instrument=cap),
    )
    return tuple(episode_id_by_key[item.episode_key.id] for item in selected)


def test_both_priority_families_match_production_selector_exhaustively_with_collisions() -> None:
    rows = (
        row("a0", "A", 0, 2), row("a1", "A", 2, 4),
        row("a2", "A", 5, 7), row("b0", "B", 0, 2),
    )
    tied = lambda _: bytes(32)
    for permuted in permutations(rows):
        for cap in (1, 2, 3):
            for top_k in (1, 2, 3):
                global_ranked = global_order(
                    permuted, seed=0, replicate=0, priority_for=tied,
                )
                hierarchical_ranked = hierarchical_order(
                    permuted, seed=0, replicate=0,
                    episode_priority_for=tied,
                    symbol_priority_for=lambda _: bytes(32),
                )
                for ranked in (global_ranked, hierarchical_ranked):
                    production = _production_ids(ranked, top_k, cap)
                    if len(production) < top_k:
                        with pytest.raises(SharedPriorityError, match="cannot satisfy"):
                            greedy_accept(
                                ranked,
                                config=SelectionConfig(top_k=top_k, max_per_symbol=cap),
                            )
                    else:
                        local = greedy_accept(
                            ranked,
                            config=SelectionConfig(top_k=top_k, max_per_symbol=cap),
                        )
                        assert tuple(candidate.episode_id for candidate in local) == production


def test_global_and_hierarchical_orders_use_full_hashes_and_explicit_id_ties() -> None:
    rows = (
        row("z", "B", 0, 1), row("a", "A", 2, 3), row("m", "A", 4, 5),
    )
    episode = {"z": digest(7), "a": digest(7), "m": digest(1)}
    symbols = {"A": digest(9), "B": digest(9)}
    assert tuple(item.episode_id for item in global_order(
        rows, seed=0, replicate=0, priority_for=lambda candidate: episode[candidate.episode_id],
    )) == ("m", "a", "z")
    assert tuple(item.episode_id for item in hierarchical_order(
        rows, seed=0, replicate=0,
        episode_priority_for=lambda candidate: episode[candidate.episode_id],
        symbol_priority_for=lambda symbol: symbols[symbol],
    )) == ("m", "a", "z")
    # A symbol hash collision is resolved by symbol ID before episode priority.
    symbols["B"] = digest(1)
    assert tuple(item.episode_id for item in hierarchical_order(
        rows, seed=0, replicate=0,
        episode_priority_for=lambda candidate: episode[candidate.episode_id],
        symbol_priority_for=lambda symbol: symbols[symbol],
    )) == ("z", "m", "a")


def test_hierarchical_null_has_intended_symbol_weighted_marginals() -> None:
    """Global samples episodes uniformly; hierarchical samples symbols uniformly."""

    rows = (row("a1", "A", 0, 1), row("a2", "A", 3, 4), row("b1", "B", 0, 1))
    global_counts = {candidate.episode_id: 0 for candidate in rows}
    hierarchical_counts = {candidate.episode_id: 0 for candidate in rows}
    for episode_order in permutations(("a1", "a2", "b1")):
        episode = {identifier: digest(rank) for rank, identifier in enumerate(episode_order)}
        selected = select_global(
            rows, seed=0, replicate=0, config=SelectionConfig(top_k=1),
            priority_for=lambda candidate: episode[candidate.episode_id],
        )
        global_counts[selected[0].episode_id] += 1
        for symbol_order in permutations(("A", "B")):
            symbols = {identifier: digest(rank) for rank, identifier in enumerate(symbol_order)}
            selected = select_hierarchical(
                rows, seed=0, replicate=0, config=SelectionConfig(top_k=1),
                episode_priority_for=lambda candidate: episode[candidate.episode_id],
                symbol_priority_for=lambda symbol: symbols[symbol],
            )
            hierarchical_counts[selected[0].episode_id] += 1
    assert global_counts == {"a1": 2, "a2": 2, "b1": 2}
    assert hierarchical_counts == {"a1": 3, "a2": 3, "b1": 6}


def test_common_episode_priority_is_not_recomputed_by_null_family() -> None:
    candidate = row("E", "S", 0, 1)
    expected = episode_priority(seed=947221, replicate=11, episode_id="E")
    assert global_order((candidate,), seed=947221, replicate=11)[0] == candidate
    # The hierarchical API accepts the same exact H_episode value independently
    # of its separate H_symbol and has no family-specific episode domain.
    assert hierarchical_order(
        (candidate,), seed=947221, replicate=11,
        episode_priority_for=lambda _: expected,
        symbol_priority_for=lambda _: symbol_priority(seed=947221, replicate=11, symbol_id="S"),
    )[0] == candidate


def test_default_orders_equal_explicit_full_hash_sort_for_every_tiny_input_order() -> None:
    rows = (
        row("e3", "C", 0, 1), row("e1", "A", 3, 4),
        row("e4", "B", 6, 7), row("e2", "A", 9, 10),
    )
    seed, replicate = 947221, 23
    expected_global = tuple(sorted(
        rows,
        key=lambda candidate: (
            episode_priority(seed=seed, replicate=replicate, episode_id=candidate.episode_id),
            candidate.episode_id,
        ),
    ))
    expected_hierarchical = tuple(sorted(
        rows,
        key=lambda candidate: (
            symbol_priority(seed=seed, replicate=replicate, symbol_id=candidate.symbol_id),
            candidate.symbol_id,
            episode_priority(seed=seed, replicate=replicate, episode_id=candidate.episode_id),
            candidate.episode_id,
        ),
    ))
    for permuted in permutations(rows):
        assert global_order(permuted, seed=seed, replicate=replicate) == expected_global
        assert hierarchical_order(permuted, seed=seed, replicate=replicate) == expected_hierarchical


def test_rejects_duplicate_bad_geometry_bad_digest_and_partial_selection() -> None:
    duplicate = (row("same", "A", 0, 1), row("same", "B", 2, 3))
    bad_geometry = (row("x", "A", 3, 2),)
    with pytest.raises(SharedPriorityError, match="duplicate"):
        global_order(duplicate, seed=1, replicate=1)
    with pytest.raises(SharedPriorityError, match="start_coordinate"):
        global_order(bad_geometry, seed=1, replicate=1)
    with pytest.raises(SharedPriorityError, match="32 bytes"):
        global_order((row("x", "A", 0, 1),), seed=1, replicate=1, priority_for=lambda _: b"x")
    with pytest.raises(SharedPriorityError, match="seed"):
        global_order((row("x", "A", 0, 1),), seed=-1, replicate=1, priority_for=lambda _: digest(0))
    with pytest.raises(SharedPriorityError, match="cannot satisfy"):
        select_global((row("x", "A", 0, 1),), seed=1, replicate=1, config=SelectionConfig(top_k=2))


def test_serial_and_twelve_worker_results_are_semantically_identical() -> None:
    risk_sets = {
        f"q{index:02d}": tuple(
            row(f"e{candidate}", f"S{candidate % 4}", candidate * 10, candidate * 10 + 1)
            for candidate in range(12)
        )
        for index in range(31)
    }
    config = SelectionConfig(top_k=7, max_per_symbol=3)
    serial = evaluate_query_selections(risk_sets, seed=947221, replicate=511, config=config)
    parallel = evaluate_query_selections(risk_sets, seed=947221, replicate=511, config=config, workers=12)
    assert parallel == serial
    assert selection_semantic_digest(parallel) == selection_semantic_digest(serial)
    assert len({row.query_id for row in parallel}) == 31


def test_shared_episode_identity_and_priority_are_closed_across_queries() -> None:
    shared = tuple(row(f"e{i}", f"S{i % 3}", 5 * i, 5 * i + 2) for i in range(8))
    selections = evaluate_query_selections(
        {"q1": shared, "q2": tuple(reversed(shared))}, seed=947221, replicate=9,
        config=SelectionConfig(top_k=5, max_per_symbol=3), workers=12,
    )
    assert selections[0].global_episode_ids == selections[1].global_episode_ids
    assert selections[0].hierarchical_episode_ids == selections[1].hierarchical_episode_ids

    mutated = list(shared)
    mutated[0] = row("e0", "WRONG", 0, 2)
    with pytest.raises(SharedPriorityError, match="shared episode_id"):
        evaluate_query_selections(
            {"q1": shared, "q2": tuple(mutated)}, seed=1, replicate=2,
            config=SelectionConfig(top_k=3),
        )


def test_semantic_digest_is_order_insensitive_but_content_sensitive() -> None:
    rows = (
        row("e0", "A", 0, 1), row("e1", "B", 2, 3), row("e2", "C", 4, 5),
    )
    selections = evaluate_query_selections(
        {"b": rows, "a": rows}, seed=5, replicate=6, config=SelectionConfig(top_k=2),
    )
    assert selection_semantic_digest(selections) == selection_semantic_digest(tuple(reversed(selections)))
    modified = list(selections)
    modified[0] = modified[0].__class__(
        query_id=modified[0].query_id,
        global_episode_ids=("not-the-same",),
        hierarchical_episode_ids=modified[0].hierarchical_episode_ids,
    )
    assert selection_semantic_digest(selections) != selection_semantic_digest(tuple(modified))
