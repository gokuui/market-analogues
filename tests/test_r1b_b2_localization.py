from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pytest

from experiments.m04r import m04r14_r1b_b2_localization as runner
from experiments.m04r import m04r14_r1b_b2_geometry as geometry
from market_analogues.adequacy_localization import (
    EffectSummary,
    PreparedLocalization,
    conditional_draw,
    conditional_priority,
)
from market_analogues.types import stable_hash


def synthetic_state(tmp_path: Path, *, queries: int = 16, episodes: int = 3) -> runner.RunState:
    query_ids = tuple(f"q{index:03d}" for index in range(queries))
    cohort_ids = tuple(f"{index + 1:024x}" for index in range(episodes))
    base = np.arange(queries, dtype=np.float64)
    distances = np.abs(base[:, None] - base[None, :]) / queries
    ranks = np.empty((queries, episodes), dtype=np.float64)
    for episode in range(episodes):
        ranks[:, episode] = (np.roll(np.arange(queries), episode) + .5) / queries
    eligibility = np.ones((queries, episodes), dtype=np.bool_)
    labels = np.arange(queries, dtype=np.int8) % 8
    localization = PreparedLocalization(distances, ranks, eligibility, labels)
    n0 = [(index % 2,) for index in range(queries)]
    cells = (
        n0,
        [value + (int(labels[index]),) for index, value in enumerate(n0)],
        [value + (index % 12,) for index, value in enumerate(n0)],
        [value + (index % 16,) for index, value in enumerate(n0)],
    )
    plans = []
    for episode, episode_id in enumerate(cohort_ids):
        observed = tuple(sorted({episode % queries, (episode + 2) % queries,
                                 (episode + 5) % queries, (episode + 8) % queries}))
        eligible = tuple(range(queries))
        designs = tuple(runner._groups(eligible, observed, value) for value in cells)
        active = tuple(sorted({index for _, members in designs[0].groups for index in members}))
        plans.append(runner.EpisodePlan(episode_id, episode, observed, active, designs))
    bindings = runner.Bindings(
        h1_commit="1" * 40, preregistration_digest="2" * 64,
        priority_contract_digest="ab" * 32, population_digest="3" * 64,
        runtime_sha256="4" * 64, geometry_digest="5" * 64,
        query_ids_digest=stable_hash(list(query_ids)), candidate_ids_digest=stable_hash(list(cohort_ids)),
        primary_ids_digest=stable_hash(list(cohort_ids)),
    )
    return runner.RunState(query_ids, cohort_ids, cohort_ids, tuple(plans), localization,
                           bindings, tmp_path / "output", tuple(runner._lp(value) for value in query_ids))


def test_runner_priority_is_exact_core_wire_format_for_both_schemes(tmp_path):
    state = synthetic_state(tmp_path)
    for replicate in (0, 17, 2**16):
        episode = state.primary_ids[1]; query = state.query_ids[7]
        prefix = runner._priority_prefix(state.bindings.priority_contract_digest, replicate, b"episode")
        assert runner._priority(prefix, episode, query) == conditional_priority(
            state.bindings.priority_contract_digest, replicate, episode, query)
        optimized = runner._priorities(sha256(prefix + runner._lp(episode)), (7,), state.query_priority_suffixes)
        assert optimized[7] == conditional_priority(
            state.bindings.priority_contract_digest, replicate, episode, query)
        prefix = runner._priority_prefix(state.bindings.priority_contract_digest, replicate, b"shared-query")
        assert runner._priority(prefix, None, query) == conditional_priority(
            state.bindings.priority_contract_digest, replicate, episode, query, shared_query=True)
        assert runner._priorities(sha256(prefix), (7,), state.query_priority_suffixes)[7] == conditional_priority(
            state.bindings.priority_contract_digest, replicate, episode, query, shared_query=True)


def test_chunk_matches_scalar_oracle_for_all_six_designs(tmp_path):
    state = synthetic_state(tmp_path)
    actual = runner._compute_chunk(state, 5, 9)
    for episode_position, plan in enumerate(state.plans):
        for design, groups in enumerate(plan.designs):
            cells = [("unused", index) for index in range(len(state.query_ids))]
            for cell_index, (_, members) in enumerate(groups.groups):
                for index in members:
                    cells[index] = ("occupied", cell_index)
            for replicate in range(5, 9):
                observed = list(plan.observed)
                eligible = sorted({index for _, members in groups.groups for index in members})
                draw = conditional_draw(
                    eligible=eligible, observed=observed, cells=cells,
                    query_ids=state.query_ids, episode_id=plan.episode_id,
                    contract_digest=state.bindings.priority_contract_digest,
                    replicate=replicate,
                )
                expected = state.localization.evaluate(plan.candidate_index, np.asarray([draw]))[0]
                np.testing.assert_array_equal(actual[runner.NULL_TABLES[design]][replicate - 5, episode_position], expected)
                if design < 2:
                    shared = conditional_draw(
                        eligible=eligible, observed=observed, cells=cells,
                        query_ids=state.query_ids, episode_id=plan.episode_id,
                        contract_digest=state.bindings.priority_contract_digest,
                        replicate=replicate, shared_query=True,
                    )
                    expected_shared = state.localization.evaluate(plan.candidate_index, np.asarray([shared]))[0]
                    np.testing.assert_array_equal(actual[runner.NULL_TABLES[4 + design]][replicate - 5, episode_position], expected_shared)
        n0_draws = []
        for replicate in range(5, 9):
            prefix = runner._priority_prefix(state.bindings.priority_contract_digest, replicate, b"episode")
            priority = {index: runner._priority(prefix, plan.episode_id, state.query_ids[index]) for index in plan.active}
            n0_draws.append(runner._select(plan.designs[0], priority, state.query_ids))
        expected_breadth = state.localization.breadth(np.asarray(n0_draws))
        np.testing.assert_array_equal(actual["episode_n0_breadth"][:, episode_position], expected_breadth)


def test_common_priorities_are_hashed_once_not_once_per_design(tmp_path, monkeypatch):
    state = synthetic_state(tmp_path, queries=12, episodes=2)
    calls = 0
    original = runner._priorities

    def counted(base, members, suffixes):
        nonlocal calls
        members = tuple(members)
        calls += len(members)
        return original(base, members, suffixes)

    monkeypatch.setattr(runner, "_priorities", counted)
    runner._compute_chunk(state, 0, 3)
    expected = 3 * (len(state.query_ids) + sum(len(plan.active) for plan in state.plans))
    assert calls == expected
    # Four episode designs and two shared designs reuse those same hashes.
    assert calls < 3 * sum(len(plan.active) * 6 for plan in state.plans)


def test_synthetic_one_and_twelve_chunk_identity(tmp_path):
    state = synthetic_state(tmp_path)
    assert runner.synthetic_worker_identity(state, 0, 12)
    serial = runner._compute_chunk(state, 0, 12)
    pieces = [runner._compute_chunk(state, start, start + 1) for start in reversed(range(12))]
    for name in runner.ARRAYS:
        parallel = np.concatenate(list(reversed([piece[name] for piece in pieces])))
        assert runner._npy_bytes(serial[name]) == runner._npy_bytes(parallel)


def test_actual_fork_process_pool_matches_serial_bytes(tmp_path):
    state = synthetic_state(tmp_path)
    serial = runner._compute_chunk(state, 0, 12)
    runner._PROCESS_STATE = state
    bounds = list(reversed([(replicate, replicate + 1) for replicate in range(12)]))
    with ProcessPoolExecutor(max_workers=12, mp_context=multiprocessing.get_context("fork")) as pool:
        list(pool.map(runner._process_shard, bounds))
    pieces = []
    for start, stop in sorted(bounds):
        _, arrays, _ = runner._load_shard(state, start, stop)
        pieces.append(arrays)
    for name in runner.ARRAYS:
        actual = np.concatenate([piece[name] for piece in pieces], axis=0)
        assert runner._npy_bytes(actual) == runner._npy_bytes(serial[name])


def test_shard_is_create_only_restart_validated_and_tamper_evident(tmp_path):
    state = synthetic_state(tmp_path)
    seal = runner._publish_shard(state, 0, 2)
    loaded, arrays, seal_sha = runner._load_shard(state, 0, 2)
    assert loaded == seal and set(arrays) == set(runner.ARRAYS)
    assert seal_sha == sha256((state.output / "shards" / runner._shard_name(0, 2) / "SHARD.json").read_bytes()).hexdigest()
    with pytest.raises(runner.B2RunnerError, match="already exists"):
        runner._publish_shard(state, 0, 2)
    path = state.output / "shards" / runner._shard_name(0, 2) / "episode_n1.npy"
    raw = bytearray(path.read_bytes()); raw[-1] ^= 1; path.write_bytes(raw)
    with pytest.raises(runner.B2RunnerError, match="file hash differs"):
        runner._load_shard(state, 0, 2)


def test_shard_loader_detects_metadata_toctou(tmp_path, monkeypatch):
    state = synthetic_state(tmp_path)
    runner._publish_shard(state, 0, 2)
    original = runner._snapshot_bytes
    calls = 0

    def changing(path):
        nonlocal calls
        content = original(path)
        if path.name == "SHARD.json":
            calls += 1
            if calls == 2:
                value = json.loads(content)
                return runner._json_bytes({**value, "status": "changed"})
        return content

    monkeypatch.setattr(runner, "_snapshot_bytes", changing)
    with pytest.raises(runner.B2RunnerError, match="metadata changed"):
        runner._load_shard(state, 0, 2)


def test_shard_bindings_cover_h1_runtime_geometry_ids_and_range(tmp_path):
    state = synthetic_state(tmp_path)
    runner._publish_shard(state, 9, 11)
    seal = json.loads((state.output / "shards" / runner._shard_name(9, 11) / "SHARD.json").read_text())
    assert seal["replicate_start"] == 9 and seal["replicate_stop"] == 11
    assert seal["bindings"] == runner._binding_dict(state.bindings)
    assert set(seal["bindings"]) == {
        "h1_commit", "preregistration_digest", "priority_contract_digest", "population_digest",
        "runtime_sha256", "geometry_digest", "query_ids_digest", "candidate_ids_digest",
        "primary_ids_digest",
    }
    changed = replace(state, bindings=replace(state.bindings, geometry_digest="9" * 64))
    with pytest.raises(runner.B2RunnerError, match="binding differs"):
        runner._load_shard(changed, 9, 11)


def test_atomic_create_only_refuses_existing_and_dangling_symlink(tmp_path):
    path = tmp_path / "value.json"
    runner._atomic_json(path, {"a": 1})
    with pytest.raises(runner.B2RunnerError, match="exists"):
        runner._atomic_json(path, {"a": 2})
    link = tmp_path / "link.json"; link.symlink_to(tmp_path / "absent")
    with pytest.raises(runner.B2RunnerError, match="exists"):
        runner._atomic_json(link, {"a": 3})


def test_authenticated_bytes_reject_mutation_and_symlink(tmp_path):
    path = tmp_path / "value.npy"
    content = runner._npy_bytes(np.arange(4, dtype="<f8"))
    path.write_bytes(content)
    digest = sha256(content).hexdigest()
    assert runner._bound_bytes(path, digest, len(content)) == content
    path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    with pytest.raises(runner.B2RunnerError, match="bound file hash differs"):
        runner._bound_bytes(path, digest, len(content))
    link = tmp_path / "link.npy"; link.symlink_to(path)
    with pytest.raises(runner.B2RunnerError, match="unsafe"):
        runner._bound_bytes(link, digest, len(content))


def test_restart_discards_only_exact_owned_unpublished_staging(tmp_path):
    owned = tmp_path / f".replicates-0000-0031.tmp-999-{'a' * 32}"
    owned.mkdir(); (owned / "partial.npy").write_bytes(b"partial")
    runner._cleanup_unpublished_staging(tmp_path, {"replicates-0000-0031"})
    assert not owned.exists()
    foreign = tmp_path / f".replicates-0032-0063.tmp-999-{'b' * 32}"
    foreign.mkdir()
    with pytest.raises(runner.B2RunnerError, match="incomplete shard evidence"):
        runner._cleanup_unpublished_staging(tmp_path, {"replicates-0000-0031"})
    assert foreign.exists()
    symlink_root = tmp_path / "symlink-root"; symlink_root.mkdir()
    symlink = symlink_root / f".replicates-0000-0031.tmp-999-{'c' * 32}"
    symlink.symlink_to(tmp_path)
    with pytest.raises(runner.B2RunnerError, match="unsafe localization staging"):
        runner._cleanup_unpublished_staging(symlink_root, {"replicates-0000-0031"})


def test_failed_shard_computation_removes_unpublished_staging(tmp_path, monkeypatch):
    state = synthetic_state(tmp_path)
    monkeypatch.setattr(runner, "_compute_chunk", lambda *args: (_ for _ in ()).throw(RuntimeError("killed")))
    with pytest.raises(RuntimeError, match="killed"):
        runner._publish_shard(state, 0, 2)
    shard_root = state.output / "shards"
    assert list(shard_root.iterdir()) == []


def test_groups_preserve_exact_observed_cell_counts_and_fail_closed():
    cells = [(0,), (0,), (1,), (1,), (1,)]
    groups = runner._groups(range(5), [0, 2, 3], cells)
    assert sorted((count, len(members)) for count, members in groups.groups) == [(1, 2), (2, 3)]
    with pytest.raises(runner.B2RunnerError, match="lacks support"):
        runner._groups([0, 1], [0, 2], cells)


def test_full_cohort_eligibility_includes_unsupported_columns_and_drives_exact_ranks():
    query_ids = ("q0", "q1", "q2")
    candidate_ids = ("e0", "e1", "unsupported")
    episodes = (
        {"episode_id": "e0", "eligible_query_ids": ["q0", "q1"]},
        {"episode_id": "e1", "eligible_query_ids": ["q0", "q2"]},
        {"episode_id": "unsupported", "eligible_query_ids": ["q0", "q1", "q2"]},
    )
    eligibility = runner._frozen_eligibility(query_ids, candidate_ids, episodes)
    np.testing.assert_array_equal(eligibility, [
        [True, True, True], [True, False, True], [False, True, True],
    ])
    distances = np.asarray([[.3, .2, .1], [.1, 99., .2], [99., .3, .1]])
    ranks = runner.candidate_specificity_ranks(distances, eligibility)
    # The unsupported column participates in every eligible denominator.
    np.testing.assert_array_equal(ranks[0], [5 / 6, 1 / 2, 1 / 6])
    altered = eligibility.copy(); altered[0, 2] = False
    changed = runner.candidate_specificity_ranks(distances, altered)
    assert not np.array_equal(ranks[0, :2], changed[0, :2])


def test_exact_key_closure_refuses_resigned_extras():
    assert runner._exact_keys({"a": 1}, ("a",), "fixture") == {"a": 1}
    with pytest.raises(runner.B2RunnerError, match="field closure"):
        runner._exact_keys({"a": 1, "extra": 2}, ("a",), "fixture")


def test_selection_full_hash_ties_break_by_query_identity():
    groups = runner.DesignGroups(((2, (0, 1, 2)),))
    priorities = {0: b"x" * 32, 1: b"x" * 32, 2: b"x" * 32}
    # Canonical identity, not caller/index order, breaks the forced full-hash tie.
    assert runner._select(groups, priorities, ("z", "a", "m")) == (1, 2)


def test_array_digest_binds_shape_dtype_and_c_order_bytes():
    array = np.arange(12, dtype="<f8").reshape(3, 4)
    assert runner._array_digest(array) != runner._array_digest(array.reshape(4, 3))
    assert runner._array_digest(array) == stable_hash({
        "shape": [3, 4], "dtype": "<f8", "sha256": sha256(array.tobytes(order="C")).hexdigest(),
    })
    assert runner._geometry_array_digest(array) == geometry._array_semantic_digest(array)


def test_runner_declares_only_frozen_geometry_and_b2_outputs():
    assert runner.OUTPUT == Path(runner.joint.OUTPUTS["b2"])
    assert runner.GEOMETRY == Path(runner.joint.OUTPUTS["geometry"])
    assert runner.REPLICATES == 4096 and runner.PRIMARY_EPISODES == 357
    assert runner.NULL_TABLES == (
        "episode_n0", "episode_n1", "episode_k12", "episode_k16", "shared_n0", "shared_n1",
    )


def test_published_decision_taxonomy_is_exact_and_fail_closed():
    assert runner._published_status("structurally_localized") == "established_pending_independent_verification"
    assert runner._published_status("not_established") == "not_established_pending_independent_verification"
    assert runner._published_status("unresolved") == "unresolved"
    with pytest.raises(runner.B2RunnerError, match="unknown localization decision"):
        runner._published_status("passed")


def test_secondary_effect_output_has_no_gate_or_rescue_boolean():
    value = EffectSummary(.2, .4, .5, .1, 4, 4, 3, True)
    descriptive = runner._descriptive_effect(value)
    assert "practical_pass" not in descriptive
    assert "required_improved_episodes" not in descriptive


def test_split_report_uses_only_frozen_effect_gate():
    observed = np.asarray([[.4, .4], [.1, .1]])
    null = np.asarray([[.5, .5], [.5, .5]])
    split = runner._split_effect(observed, null, np.asarray([True, False]))
    assert split["gate_passed"] is True
    assert set(split) == {
        "episodes", "cohesion_relative_improvement", "specificity_improvement", "gate", "gate_passed",
    }
    assert "episode-count" in split["gate"] and "no p-value" in split["gate"]


def test_selected_link_diagnostics_are_exact_h1_bound_descriptive_only(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "QUERIES", 2)
    monkeypatch.setattr(runner, "PRIMARY_EPISODES", 1)
    monkeypatch.setattr(runner, "PRIMARY_LINKS", 2)
    episode_id = "e" * 24
    query_ids = ["q0", "q1"]
    case_root = tmp_path / runner.joint.CASES
    case_root.mkdir(parents=True)
    manifest = []
    for index, (total, market) in enumerate(((2.0, .4), (4.0, .8))):
        path = case_root / f"q{index}.json"
        path.write_text(json.dumps({
            "query_episode_id": query_ids[index],
            "matches": [{"episode_id": episode_id, "total_distance": total,
                         "component_distances": {"market_context": market}}],
        }))
        manifest.append({"path": path.relative_to(tmp_path).as_posix(),
                         "bytes": path.stat().st_size, "sha256": runner._file_sha(path)})
    prereg = {"authorities": {
        "case_manifest": manifest,
        "population": {
            "query_ids": query_ids, "primary_ids": [episode_id],
            "episodes": [{"episode_id": episode_id, "observed_query_ids": query_ids,
                          "observed_count": 2}],
        },
    }}
    result = runner._selected_link_diagnostics(tmp_path, prereg, [episode_id])
    assert result["scope"] == {"primary_episodes": 1, "selected_links": 2}
    assert all(set(row) == {"episode_id", "query_id", "production_total", "raw_market_context"}
               for row in result["links"])
    assert result["selected_link_aggregate"]["production_total"] == 3.0
    assert result["selected_link_aggregate"]["raw_market_context"] == pytest.approx(.6)
    assert result["equal_episode_aggregate"]["production_total"] == 3.0
    assert result["equal_episode_aggregate"]["raw_market_context"] == pytest.approx(.6)
    assert result["claims"] == {"null_computed": False, "pvalue_computed": False,
                                "decision_input": False, "rescue_authorized": False}
    assert result["links_digest"] == stable_hash(result["links"])
    assert result["episode_rows_digest"] == stable_hash(result["episode_rows"])
    (case_root / "q0.json").write_text("{}")
    with pytest.raises(runner.B2RunnerError, match="bound file (length|hash) differs"):
        runner._selected_link_diagnostics(tmp_path, prereg, [episode_id])


def test_secondary_diagnostics_cannot_change_the_localization_decision(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPLICATES", 2)
    monkeypatch.setattr(runner, "SHARD_REPLICATES", 1)
    state = synthetic_state(tmp_path)
    ranges = [(0, 1), (1, 2)]
    for bounds in ranges:
        runner._publish_shard(state, *bounds)
    observed, breadth = runner._observed(state)
    first = runner._aggregate(state, ranges, observed, breadth, {"marker": "first"})
    second = runner._aggregate(state, ranges, observed, breadth, {"marker": "mutated"})
    assert first["scientific"]["decision"] == second["scientific"]["decision"]
    assert first["scientific"]["status"] == second["scientific"]["status"]
    assert (first["scientific"]["selected_link_production_diagnostics"]
            != second["scientific"]["selected_link_production_diagnostics"])


def test_production_parallelism_and_shard_layout_are_not_adaptable(tmp_path):
    with pytest.raises(runner.B2RunnerError, match="exactly 12"):
        runner.run(tmp_path, workers=1)
    with pytest.raises(runner.B2RunnerError, match="frozen 32"):
        runner.run(tmp_path, shard_replicates=64)
