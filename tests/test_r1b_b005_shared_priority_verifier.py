from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.m04r import m04r14_r1b_b005_shared_priority_verifier as verify


def universe(*, symbols: int = 10, per_symbol: int = 5) -> verify.Universe:
    ids = []; cutoffs = []; symbol_ids = []; ordinals = []; starts = []; stops = []
    cursor = 0
    for symbol in range(symbols):
        starts.append(cursor)
        for local in range(per_symbol):
            ids.append(np.void((symbol * 10_000 + local + 1).to_bytes(12, "big")))
            cutoffs.append(100 + local * 10); symbol_ids.append(symbol)
            ordinals.append(local * 60); cursor += 1
        stops.append(cursor)
    return verify.Universe(
        np.asarray(ids, dtype="V12"), np.asarray(cutoffs, dtype="<i8"),
        np.asarray(symbol_ids, dtype="<u4"), np.asarray(ordinals, dtype="<i8"),
        tuple(f"S{value:02d}" for value in range(symbols)),
        np.asarray(starts, dtype="<i8"), np.asarray(stops, dtype="<i8"),
    )


def queries(candidate_universe: verify.Universe):
    return (
        verify.Query("Q0", 0, 90, 1_000, len(candidate_universe.episode_ids) - 5),
        verify.Query("Q1", 1, 1_000, 1_000, len(candidate_universe.episode_ids)),
    )


def test_independent_priority_golden_vectors() -> None:
    assert verify.priority(
        verify.EPISODE_DOMAIN, 947221, 17, "episode-α",
    ).hex() == "7298340f24b15fef861b5a6f45deddd5f4ba8f547e764072dde1e13cc12c5777"
    assert verify.priority(
        verify.SYMBOL_DOMAIN, 947221, 17, "SYM",
    ).hex() == "cd7b16d905810e1a97b14fcfffd5102b10ad6c2c8b8f6903ce128c2fa3c3b05b"


def test_full_sha_orders_use_explicit_entity_ties() -> None:
    u = universe(symbols=8, per_symbol=5)
    episode = np.zeros((40, 32), dtype=np.uint8)
    episode[:, 0] = np.arange(40, dtype=np.uint8) % 4
    symbol = np.zeros((8, 32), dtype=np.uint8)
    symbol[:, 0] = np.arange(8, dtype=np.uint8) % 2
    episode_order = np.argsort(u.episode_ids, kind="stable")
    symbol_order = np.argsort(np.asarray(u.symbols, dtype="U"), kind="stable")
    global_order, hierarchical, observed_symbol_order = verify.complete_orders(
        u, episode, symbol, episode_order, symbol_order,
    )
    expected_global = sorted(range(40), key=lambda index: (
        bytes(episode[index]), bytes(u.episode_ids[index]),
    ))
    expected_hierarchical = sorted(range(40), key=lambda index: (
        bytes(symbol[int(u.symbol_ids[index])]), u.symbols[int(u.symbol_ids[index])],
        bytes(episode[index]), bytes(u.episode_ids[index]),
    ))
    assert list(global_order) == expected_global
    assert list(hierarchical) == expected_hierarchical
    assert list(observed_symbol_order) == sorted(range(8), key=lambda index: (
        bytes(symbol[index]), u.symbols[index],
    ))


def test_direct_selector_enforces_causal_exclusion_cap_and_inclusive_overlap() -> None:
    u = universe(symbols=8, per_symbol=5)
    u.local_ordinals[:5] = np.asarray([0, 50, 51, 102, 153])
    selected = verify.select_direct(
        u, verify.Query("Q", 7, 1_001, 1_000, len(u.episode_ids)), range(len(u.episode_ids)),
    )
    assert selected[:3] == (0, 2, 3)
    excluded = verify.select_direct(
        u, verify.Query("OWN", 0, 131, 1_000, len(u.episode_ids) - 2), range(len(u.episode_ids)),
    )
    assert all(not (int(u.symbol_ids[index]) == 0 and int(u.cutoffs[index]) >= 131)
               for index in excluded)


def test_replay_replicate_is_deterministic_and_complete() -> None:
    u = universe(); q = queries(u)
    first = verify.replay_replicate(u, q, 7)
    second = verify.replay_replicate(u, q, 7)
    assert first == second
    assert first["selection_completeness"] == {
        "global": True, "hierarchical": True, "exact_top_k": 20,
    }
    assert first["work_counters"] == {
        "episode_hashes": 50, "symbol_hashes": 10,
        "global_orders": 1, "hierarchical_orders": 1,
        "per_query_full_universe_hashes": 0, "per_query_full_universe_sorts": 0,
        "query_risk_sets": 2, "global_risk_set_filters": 2,
        "hierarchical_risk_set_filters": 2, "total_risk_set_filters": 4,
    }
    assert set(first["global"]["metrics"]) == set(verify.METRICS)
    assert len(first["global"]["selection_digest"]) == 64


def test_batched_selector_optimizations_equal_direct_oracle() -> None:
    u = universe(symbols=12, per_symbol=8)
    q = (
        verify.Query("A", 0, 125, 150, 0), verify.Query("B", 4, 1_000, 150, 0),
        verify.Query("C", 9, 135, 1_000, 0), verify.Query("D", 11, 1_000, 1_000, 0),
    )
    episode = np.asarray([[*verify.priority(
        verify.EPISODE_DOMAIN, 947221, 9, bytes(value).hex(),
    )] for value in u.episode_ids], dtype=np.uint8)
    symbol = np.asarray([[*verify.priority(
        verify.SYMBOL_DOMAIN, 947221, 9, value,
    )] for value in u.symbols], dtype=np.uint8)
    orders = verify.complete_orders(
        u, episode, symbol, np.argsort(u.episode_ids, kind="stable"),
        np.argsort(np.asarray(u.symbols, dtype="U"), kind="stable"),
    )
    assert verify.select_global_batch(u, q, orders[0]) == tuple(
        verify.select_direct(u, row, orders[0]) for row in q
    )
    assert verify.select_hierarchical_batch(u, q, orders[1], orders[2]) == tuple(
        verify.select_direct(u, row, orders[1]) for row in q
    )


def test_episode_hasher_matches_independent_serial_priorities() -> None:
    u = universe(symbols=4, per_symbol=7)
    expected = b"".join(verify.priority(
        verify.EPISODE_DOMAIN, 947221, 23, bytes(value).hex(),
    ) for value in u.episode_ids)
    with verify.EpisodeHasher(u.episode_ids, 12) as hasher:
        actual = hasher.hashes(947221, 23).copy()
    assert actual.tobytes() == expected


def test_metrics_match_hand_counted_concentration() -> None:
    u = universe(symbols=10, per_symbol=5)
    selected = (tuple(range(20)), tuple(range(20, 40)))
    metrics = verify.concentration_metrics(u, selected)
    assert metrics["episode_unique"] == 40
    assert metrics["episode_max"] == 1
    assert metrics["symbol_unique"] == 8
    assert metrics["symbol_max"] == 5
    assert metrics["query_any_repeated_symbol_fraction"] == 1.0
    assert metrics["mean_pairwise_query_episode_overlap"] == 0.0


def valid_shard(path: Path) -> tuple[dict, dict]:
    counters = {
        "episode_hashes": 10, "symbol_hashes": 2, "global_orders": 1,
        "hierarchical_orders": 1, "per_query_full_universe_hashes": 0,
        "per_query_full_universe_sorts": 0, "query_risk_sets": 2,
        "global_risk_set_filters": 2, "hierarchical_risk_set_filters": 2,
        "total_risk_set_filters": 4,
    }
    rows = []
    for replicate in range(2):
        family = {"metrics": {name: 1.0 for name in verify.METRICS},
                  "selection_digest": sha256(str(replicate).encode()).hexdigest()}
        rows.append({"replicate": replicate, "global": family, "hierarchical": family,
                     "work_counters": counters,
                     "selection_completeness": {"global": True, "hierarchical": True,
                                                "exact_top_k": 20}})
    state = {"schema_version": "m04r14-r1b-b005-shared-priority-shard-v1",
             "binding_digest": "a" * 64, "replicate_range": [0, 2], "rows": rows}
    value = {**state, "shard_digest": verify.stable_hash(state)}
    path.write_text(json.dumps(value))
    return value, counters


def test_shard_and_checkpoint_digests_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "shard.json"; value, counters = valid_shard(path)
    assert verify.validate_producer_shard(path, "a" * 64, 0, 2, counters) == value
    value["rows"][0]["global"]["metrics"][verify.METRICS[0]] = 2.0
    path.write_text(json.dumps(value))
    with pytest.raises(verify.VerificationError, match="digest"):
        verify.validate_producer_shard(path, "a" * 64, 0, 2, counters)

    source = {"path": "shard.json", "bytes": 1, "sha256": "b" * 64}
    state = {"schema_version": verify.CHECKPOINT_SCHEMA,
             "verification_binding_digest": "c" * 64, "producer_shard": source,
             "replicate_range": [0, 2], "reconstructed_rows_digest": "d" * 64,
             "exact_rows_match": True}
    checkpoint = {**state, "checkpoint_digest": verify.stable_hash(state)}
    check = tmp_path / "check.json"; check.write_text(json.dumps(checkpoint))
    assert verify.validate_checkpoint(check, "c" * 64, source, 0, 2, "d" * 64) == checkpoint
    checkpoint["exact_rows_match"] = False; check.write_text(json.dumps(checkpoint))
    with pytest.raises(verify.VerificationError):
        verify.validate_checkpoint(check, "c" * 64, source, 0, 2, "d" * 64)


def test_summaries_use_inclusive_directional_plus_one_tail() -> None:
    actual = {name: 2.0 for name in verify.METRICS}
    rows = []
    for value in (1.0, 2.0, 3.0):
        rows.append({"global": {"metrics": {name: value for name in verify.METRICS}}})
    directions = {name: "higher_is_more_concentrated" for name in verify.METRICS}
    directions["episode_unique"] = "lower_is_more_concentrated"
    result = verify.summaries(actual, rows, directions, "global")
    assert result["episode_max"]["inclusive_tail_count"] == 2
    assert result["episode_max"]["concentration_tail_monte_carlo_p"] == .75
    assert result["episode_unique"]["inclusive_tail_count"] == 2


def test_verifier_has_no_producer_or_shared_core_import() -> None:
    source = Path(verify.__file__).read_text()
    assert "m04r14_r1b_b005_shared_priority as" not in source
    assert "market_analogues.adequacy" not in source
    assert "market_analogues.adequacy_shared_priority" not in source
