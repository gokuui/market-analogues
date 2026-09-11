from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.m04r import m04r14_r1b_b005_shared_priority as run
from market_analogues.adequacy_shared_priority import (
    EPISODE_DOMAIN, SYMBOL_DOMAIN, SelectionConfig, SharedPriorityCandidate,
    select_global, select_hierarchical,
)
from market_analogues.types import stable_hash


def universe(*, symbols: int = 10, per_symbol: int = 4) -> run.CandidateUniverse:
    ids = []; cutoffs = []; symbol_ids = []; ordinals = []; starts = []; stops = []
    cursor = 0
    for symbol in range(symbols):
        starts.append(cursor)
        for local in range(per_symbol):
            ids.append(np.void((symbol * 10_000 + local + 1).to_bytes(12, "big")))
            cutoffs.append(100 + local * 10)
            symbol_ids.append(symbol)
            # Separate intervals so cap-three, rather than overlap, is binding.
            ordinals.append(local * 60)
            cursor += 1
        stops.append(cursor)
    return run.CandidateUniverse(
        np.asarray(ids, dtype="V12"), np.asarray(cutoffs, dtype="<i8"),
        np.asarray(symbol_ids, dtype="<u4"), np.asarray(ordinals, dtype="<i8"),
        tuple(f"S{value:02d}" for value in range(symbols)),
        np.asarray(starts, dtype="<i8"), np.asarray(stops, dtype="<i8"),
    )


def queries(candidate_universe: run.CandidateUniverse) -> tuple[run.QueryRisk, ...]:
    # All candidates are eligible except the four same-symbol candidates for Q0.
    return (
        run.QueryRisk("Q0", 0, 90, 1_000, len(candidate_universe.episode_ids) - 4),
        run.QueryRisk("Q1", 1, 1_000, 1_000, len(candidate_universe.episode_ids)),
    )


def hashes(values: list[str], domain: str, replicate: int = 0) -> np.ndarray:
    return run.priority_array(
        tuple(value.encode() for value in values), domain=domain,
        seed=947221, replicate=replicate, workers=1,
    )


def binding(candidate_universe: run.CandidateUniverse, query_rows, *, replicates=2, shard_size=1):
    state = {
        "schema_version": run.BINDING_SCHEMA, "h1_commit": "1" * 40,
        "joint_preregistration_digest": "2" * 64, "runtime_sha256": {"runner": "3" * 64},
        "episode_ids_sha256": sha256(b"".join(bytes(v) for v in candidate_universe.episode_ids)).hexdigest(),
        "symbols_digest": stable_hash(list(candidate_universe.symbols)),
        "queries_digest": stable_hash([q.query_id for q in query_rows]),
        "candidate_episodes": len(candidate_universe.episode_ids),
        "candidate_symbols": len(candidate_universe.symbols), "queries": len(query_rows),
        "replicates": replicates, "replicate_indices": list(range(replicates)),
        "shard_size": shard_size, "seed": 947221,
        "shard_ranges": [list(value) for value in run._shard_ranges(replicates, shard_size)],
        "priority_domains": {"episode": EPISODE_DOMAIN, "symbol": SYMBOL_DOMAIN},
        "metric_directions": run.joint.specification()["b005"]["metrics"],
        "top_k": 20, "max_per_symbol": 3,
        "real_forward_outcomes_accessed": False, "retrieval_identity_digest": "4" * 64,
    }
    return {**state, "binding_digest": stable_hash(state)}


def actual_metrics(candidate_universe, query_rows):
    selections = []
    for query in query_rows:
        eligible = [i for i in range(len(candidate_universe.episode_ids))
                    if run._eligible(candidate_universe, query, i)]
        selections.append(tuple(eligible[:20]))
    return run.metrics_for_selections(candidate_universe, selections)


def test_priority_array_serial_parallel_byte_identity():
    values = [f"episode-{value:04d}" for value in range(137)]
    one = run.priority_array(tuple(v.encode() for v in values), domain=EPISODE_DOMAIN,
                             seed=947221, replicate=17, workers=1)
    twelve = run.priority_array(tuple(v.encode() for v in values), domain=EPISODE_DOMAIN,
                                seed=947221, replicate=17, workers=12)
    assert one.tobytes() == twelve.tobytes()


def test_priority_array_rejects_duplicate_and_bad_workers():
    with pytest.raises(run.SharedPriorityRunError, match="unique"):
        run.priority_array((b"x", b"x"), domain=EPISODE_DOMAIN, seed=1, replicate=0)
    with pytest.raises(run.SharedPriorityRunError, match="workers"):
        run.priority_array((b"x",), domain=EPISODE_DOMAIN, seed=1, replicate=0, workers=0)


def test_persistent_production_hasher_matches_core_full_sha():
    u = universe(symbols=4, per_symbol=7)
    expected = hashes([bytes(value).hex() for value in u.episode_ids], EPISODE_DOMAIN, 23)
    with run.ForkEpisodeHasher(u.episode_ids, 12) as hasher:
        actual = hasher.hashes(seed=947221, replicate=23).copy()
    assert actual.tobytes() == expected.tobytes()


def test_complete_orders_match_full_sha_reference_with_forced_ties():
    u = universe(symbols=8, per_symbol=5)
    ep = hashes([bytes(value).hex() for value in u.episode_ids], EPISODE_DOMAIN)
    sy = hashes(list(u.symbols), SYMBOL_DOMAIN)
    # Deliberate complete SHA ties exercise explicit identifier tie breaking.
    ep[[1, 7, 19]] = ep[1]
    sy[[2, 6]] = sy[2]
    orders = run.build_complete_orders(u, seed=947221, replicate=0,
                                       episode_hashes=ep, symbol_hashes=sy)
    ep_map = {bytes(value).hex(): ep[index].tobytes() for index, value in enumerate(u.episode_ids)}
    sy_map = {value: sy[index].tobytes() for index, value in enumerate(u.symbols)}
    candidates = tuple(SharedPriorityCandidate(
        bytes(u.episode_ids[index]).hex(), u.symbols[int(u.symbol_ids[index])],
        int(u.local_ordinals[index]) * 5, int(u.local_ordinals[index]) * 5 + 251,
    ) for index in range(len(u.episode_ids)))
    reference_global = select_global(
        candidates, seed=947221, replicate=0, config=SelectionConfig(top_k=20),
        priority_for=lambda row: ep_map[row.episode_id],
    )
    reference_hierarchical = select_hierarchical(
        candidates, seed=947221, replicate=0, config=SelectionConfig(top_k=20),
        episode_priority_for=lambda row: ep_map[row.episode_id],
        symbol_priority_for=lambda symbol: sy_map[symbol],
    )
    selected_global = run.select_from_order(u, queries(u)[1], orders.global_indices)
    selected_hierarchical = run.select_from_order(u, queries(u)[1], orders.hierarchical_indices)
    assert [bytes(u.episode_ids[i]).hex() for i in selected_global] == [r.episode_id for r in reference_global]
    assert [bytes(u.episode_ids[i]).hex() for i in selected_hierarchical] == [r.episode_id for r in reference_hierarchical]
    assert orders.counters == {
        "episode_hashes": 40, "symbol_hashes": 8, "global_orders": 1,
        "hierarchical_orders": 1, "per_query_full_universe_hashes": 0,
        "per_query_full_universe_sorts": 0,
    }


def test_risk_filter_and_inclusive_overlap_are_exact():
    u = universe(symbols=8, per_symbol=5)
    # Force adjacent ordinal windows for symbol zero: endpoints overlap.
    u.local_ordinals[:5] = np.asarray([0, 50, 51, 102, 153])
    q = run.QueryRisk("Q", 7, 1_001, 1_000, len(u.episode_ids))
    selected = run.select_from_order(u, q, range(len(u.episode_ids)),
                                     config=SelectionConfig(top_k=4, max_per_symbol=3))
    assert selected[:3] == (0, 2, 3)
    assert selected[3] == 5
    excluded = run.select_from_order(
        u, run.QueryRisk("OWN", 0, 131, 1_000, len(u.episode_ids) - 2),
        range(len(u.episode_ids)), config=SelectionConfig(top_k=3),
    )
    assert all(not (int(u.symbol_ids[i]) == 0 and int(u.cutoffs[i]) >= 131) for i in excluded)


def test_compute_replicate_exact_top20_unique_and_worker_identity():
    u = universe(); qs = queries(u)
    one = run.compute_replicate(u, qs, seed=947221, replicate=5, workers=1)
    twelve = run.compute_replicate(u, qs, seed=947221, replicate=5, workers=12)
    assert run._canonical_scientific_bytes(one) == run._canonical_scientific_bytes(twelve)
    assert one["selection_completeness"] == {
        "global": True, "hierarchical": True, "exact_top_k": 20,
    }
    assert one["work_counters"]["query_risk_sets"] == 2
    assert one["work_counters"]["global_risk_set_filters"] == 2
    assert one["work_counters"]["hierarchical_risk_set_filters"] == 2
    assert one["work_counters"]["total_risk_set_filters"] == 4
    assert set(one["global"]["metrics"]) == set(run.joint.METRICS)


def test_cached_query_filters_equal_direct_filter_for_both_complete_orders():
    u = universe(symbols=12, per_symbol=8)
    qs = (
        run.QueryRisk("A", 0, 125, 150, 0),
        run.QueryRisk("B", 4, 1_000, 150, 0),
        run.QueryRisk("C", 9, 135, 1_000, 0),
        run.QueryRisk("D", 11, 1_000, 1_000, 0),
    )
    ep = hashes([bytes(value).hex() for value in u.episode_ids], EPISODE_DOMAIN, 7)
    sy = hashes(list(u.symbols), SYMBOL_DOMAIN, 7)
    orders = run.build_complete_orders(u, seed=947221, replicate=7,
                                       episode_hashes=ep, symbol_hashes=sy)
    config = SelectionConfig(top_k=20, max_per_symbol=3)
    direct_global = tuple(run.select_from_order(u, q, orders.global_indices, config=config) for q in qs)
    direct_hierarchical = tuple(
        run.select_from_order(u, q, orders.hierarchical_indices, config=config) for q in qs
    )
    assert run.select_global_queries(u, qs, orders.global_indices, config=config) == direct_global
    assert run.select_hierarchical_queries(u, qs, orders, config=config) == direct_hierarchical


@pytest.mark.parametrize("bad", [
    (tuple(range(19)),), (tuple([0] * 20),), (tuple(range(19)) + (999,),),
])
def test_selection_completeness_refuses_partial_duplicate_and_outside(bad):
    with pytest.raises(run.SharedPriorityRunError):
        run.validate_complete_selections(universe(), bad, top_k=20)


def test_metrics_reject_incomplete_shape():
    with pytest.raises(run.SharedPriorityRunError):
        run.metrics_for_selections(universe(), [tuple(range(20))])


def test_query_count_validation_uses_exact_same_symbol_exclusion():
    u = universe(); qs = queries(u)
    run._validate_queries(u, qs)
    wrong = (run.QueryRisk("Q0", 0, 90, 1_000, len(u.episode_ids)), qs[1])
    with pytest.raises(run.SharedPriorityRunError, match="risk-set count"):
        run._validate_queries(u, wrong)


def test_run_resume_is_create_only_and_scientifically_identical(tmp_path: Path):
    u = universe(); qs = queries(u); bind = binding(u, qs)
    actual = actual_metrics(u, qs); output = tmp_path / "run"
    partial = run.run_experiment(
        output, binding=bind, universe=u, queries=qs, actual=actual,
        workers=1, stop_after_shards=1,
    )
    assert partial["status"] == "incomplete"
    first_bytes = (output / "shards/shard-0000-0001.json").read_bytes()
    resumed = run.run_experiment(
        output, binding=bind, universe=u, queries=qs, actual=actual, workers=12,
    )
    assert (output / "shards/shard-0000-0001.json").read_bytes() == first_bytes
    assert resumed["gates"]["every_selection_exact_top20_unique"] is True
    assert resumed["claims"] == {
        "adequacy_labels_authorized": False, "predictive_claim_authorized": False,
        "production_promotion_authorized": False, "real_forward_outcomes_accessed": False,
    }
    again = run.run_experiment(
        output, binding=bind, universe=u, queries=qs, actual=actual, workers=1,
    )
    assert again["scientific_digest"] == resumed["scientific_digest"]
    assert again["result_digest"] == resumed["result_digest"]


def test_corrupt_completed_shard_is_refused_not_replaced(tmp_path: Path):
    u = universe(); qs = queries(u); bind = binding(u, qs, replicates=1)
    output = tmp_path / "run"
    run.run_experiment(output, binding=bind, universe=u, queries=qs,
                       actual=actual_metrics(u, qs), workers=1)
    path = output / "shards/shard-0000-0001.json"
    value = json.loads(path.read_text()); value["rows"][0]["replicate"] = 9
    path.write_text(json.dumps(value))
    corrupt = path.read_bytes()
    with pytest.raises(run.SharedPriorityRunError):
        run.run_experiment(output, binding=bind, universe=u, queries=qs,
                           actual=actual_metrics(u, qs), workers=1)
    assert path.read_bytes() == corrupt


def test_foreign_or_partial_shard_artifact_is_refused(tmp_path: Path):
    u = universe(); qs = queries(u); bind = binding(u, qs, replicates=1)
    output = tmp_path / "run"; (output / "shards").mkdir(parents=True)
    run.atomic_json(output / "BINDING.json", bind)
    (output / "shards/.shard.tmp").write_text("partial")
    with pytest.raises(run.SharedPriorityRunError, match="unexpected"):
        run.run_experiment(output, binding=bind, universe=u, queries=qs,
                           actual=actual_metrics(u, qs), workers=1)


def test_exact_unpublished_shard_staging_is_discarded_on_resume(tmp_path: Path):
    u = universe(); qs = queries(u); bind = binding(u, qs, replicates=1)
    output = tmp_path / "run"; (output / "shards").mkdir(parents=True)
    run.atomic_json(output / "BINDING.json", bind)
    staging = output / "shards" / f".shard-0000-0001.json.tmp-123-{'0' * 32}"
    staging.write_text("unpublished")
    result = run.run_experiment(output, binding=bind, universe=u, queries=qs,
                                actual=actual_metrics(u, qs), workers=1)
    assert result["passed"] is True and not staging.exists()


@pytest.mark.parametrize("name", ["RESULT.json", "report.html"])
def test_result_and_report_symlinks_are_refused(tmp_path: Path, name: str):
    u = universe(); qs = queries(u); bind = binding(u, qs, replicates=1)
    output = tmp_path / name.replace(".", "-")
    target = tmp_path / f"target-{name}"; target.write_text("{}" if name.endswith("json") else "")
    output.mkdir(); (output / name).symlink_to(target)
    with pytest.raises(run.SharedPriorityRunError, match="symlink"):
        run.run_experiment(output, binding=bind, universe=u, queries=qs,
                           actual=actual_metrics(u, qs), workers=1)


def test_changed_resume_binding_is_refused(tmp_path: Path):
    u = universe(); qs = queries(u); bind = binding(u, qs, replicates=1)
    output = tmp_path / "run"
    run.run_experiment(output, binding=bind, universe=u, queries=qs,
                       actual=actual_metrics(u, qs), workers=1)
    changed = dict(bind); changed["seed"] += 1
    changed["binding_digest"] = stable_hash({k: v for k, v in changed.items() if k != "binding_digest"})
    with pytest.raises(run.SharedPriorityRunError, match="binding"):
        run.run_experiment(output, binding=changed, universe=u, queries=qs,
                           actual=actual_metrics(u, qs), workers=1)


def test_shard_tail_p_is_plus_one_and_direction_specific():
    actual = {name: 2.0 for name in run.joint.METRICS}
    rows = [{"global": {"metrics": {name: value for name in run.joint.METRICS}}}
            for value in (1.0, 2.0, 3.0)]
    directions = {name: "higher_is_more_concentrated" for name in run.joint.METRICS}
    directions["episode_unique"] = "lower_is_more_concentrated"
    summary = run._summaries(actual, rows, directions, "global")
    assert summary["episode_max"]["inclusive_tail_count"] == 2
    assert summary["episode_max"]["concentration_tail_monte_carlo_p"] == .75
    assert summary["episode_unique"]["inclusive_tail_count"] == 2


def test_atomic_json_refuses_existing_file_and_symlink(tmp_path: Path):
    path = tmp_path / "value.json"; run.atomic_json(path, {"x": 1})
    with pytest.raises(run.SharedPriorityRunError, match="exists"):
        run.atomic_json(path, {"x": 1})
    link = tmp_path / "link.json"; link.symlink_to(tmp_path / "missing")
    with pytest.raises(run.SharedPriorityRunError, match="exists"):
        run.atomic_json(link, {"x": 1})
