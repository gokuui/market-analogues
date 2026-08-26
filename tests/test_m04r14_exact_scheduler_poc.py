from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from threading import Lock
import sys
import time
import tempfile
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/m04r/m04r14_exact_scheduler_poc.py"
SPEC = importlib.util.spec_from_file_location("m04r14_exact_scheduler_poc", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
scheduler = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scheduler
SPEC.loader.exec_module(scheduler)


def _resources(seconds: float = 0.01) -> dict[str, object]:
    snapshot = {
        "user_cpu_seconds": 1.0, "system_cpu_seconds": 0.5,
        "minor_faults": 10, "major_faults": 0,
        "voluntary_context_switches": 2,
        "involuntary_context_switches": 1, "max_rss_mb": 100.0,
        "swap_kib": 0, "cpu_affinity": list(range(8)),
        "thread_environment": {
            key: None for key in scheduler.THREAD_ENV_KEYS
        },
    }
    after = dict(snapshot)
    after.update({
        "user_cpu_seconds": 1.0 + seconds, "minor_faults": 11,
        "max_rss_mb": 101.0,
    })
    return scheduler._resource_delta(snapshot, after)


class FakeBackend:
    case_ids = scheduler.FROZEN_CASE_IDS
    query_ids = scheduler.FROZEN_QUERY_IDS
    throughput_case_ids = scheduler.THROUGHPUT_CASE_IDS
    throughput_query_ids = scheduler.THROUGHPUT_QUERY_IDS

    def __init__(self, *, mutate: bool = False, unstable: bool = False) -> None:
        self.mutate = mutate
        self.unstable = unstable
        self.prepares: list[tuple[int, str]] = []
        self.exacts: list[tuple[str, int]] = []
        self._lock = Lock()
        self.active = 0
        self.maximum_active = 0

    def foundation(self) -> dict[str, object]:
        return {
            "mode": "test", "resident_identity_digest": "a" * 64,
            "source_lease_digest": "b" * 64,
            "prerequisites": {
                "evidence_catalog_digest": scheduler.CATALOG_DIGEST,
                "adversarial_oracle_result_digest": scheduler.ORACLE_RESULT_DIGEST,
            },
        }

    def prepare(self, case_ordinal: int, task_id: str) -> scheduler.PreparedCase:
        self.prepares.append((case_ordinal, task_id))
        case_id = self.throughput_case_ids[case_ordinal]
        query_id = self.throughput_query_ids[case_ordinal]
        proposal_digest = scheduler.stable_hash({"query_id": query_id})
        def report(order: str, rows: int) -> dict[str, object]:
            return {
                "schema_version": "test-branch-aware-v2",
                "generation_id": "test-generation", "query_episode_id": query_id,
                "candidates": [{
                    "episode_id": query_id, "symbol": case_id.split("-")[1],
                    "cutoff_ns": 1, "quality_tier": "A",
                    "lower_bound_hex": float(0).hex(), "routes": ["composite"],
                    "overflow_fallback": False,
                }],
                "rows_scanned": 1, "eligible_rows": 1,
                "eligible_main_rows": 1, "eligible_overflow_rows": 0,
                "route_counts": {"composite": 1},
                "route_quotas": {"composite": 16385}, "block_rows": rows,
                "block_order": order, "elapsed_seconds": 0.01,
                "peak_rss_mb": 100.0,
                "candidate_digest": scheduler.stable_hash([query_id]),
                "result_digest": proposal_digest, "contract_digest": "c" * 64,
                "input_digest": "d" * 64,
            }
        forward = report("forward", 4096); reverse = report("reverse", 4097)
        proposal_semantic = scheduler.stable_hash(scheduler._without(
            forward, {"block_rows", "block_order", "elapsed_seconds", "peak_rss_mb"},
        ))
        binding = {"digest": scheduler.stable_hash(query_id)}
        semantic = {
            "schema_version": scheduler.PROPOSAL_SCHEMA,
            "task_id": task_id, "case_id": case_id, "query_id": query_id,
            "query_binding": binding,
            "resident_lease_digests": ["a" * 64] * 4,
            "source_binding_before": binding, "source_binding_after": binding,
            "resident_snapshot": {"identity_digest": "a" * 64},
            "forward": forward, "reverse": reverse,
            "semantic_digest": proposal_semantic,
        }
        return scheduler.PreparedCase(
            task_id, case_id, query_id, semantic,
            {"wall_seconds": 0.02, "forward_seconds": 0.01,
             "reverse_seconds": 0.01, "resources": _resources()},
        )

    def exact(
        self, prepared: scheduler.PreparedCase, workers: int,
    ) -> scheduler.ExactAttempt:
        with self._lock:
            self.active += 1; self.maximum_active = max(self.maximum_active, self.active)
            occurrence = sum(
                prior == (prepared.case_id, workers) for prior in self.exacts
            )
            self.exacts.append((prepared.case_id, workers))
        if prepared.task_id.startswith("throughput") and occurrence >= 1:
            time.sleep(0.015)
        base = {
            "schema_version": scheduler.ATTEMPT_SCHEMA,
            "case_id": prepared.case_id, "query_id": prepared.query_id,
            "workers": workers,
            "proposal_semantic_digest": prepared.semantic["semantic_digest"],
            "certificate": {"result_digest": scheduler.stable_hash(prepared.query_id)},
            "matches": [{"episode_id": prepared.query_id,
                         "total_distance_hex": float(1.0).hex()}],
            "certificate_result_digest": scheduler.stable_hash(prepared.query_id),
            "match_digest": "",
            "lease_before": "a" * 64, "lease_after": "a" * 64,
        }
        base["match_digest"] = scheduler.stable_hash(base["matches"])
        if self.mutate and workers == 4:
            base["matches"][0]["episode_id"] = "f" * 24
        nominal = {1: 0.40, 2: 0.25, 4: 0.10, 8: 0.105}[workers]
        repetition = int(prepared.task_id.split("-r")[1][0]) \
            if "primary-r" in prepared.task_id else 0
        factor = (1.0, 1.02, 0.99)[repetition]
        if self.unstable and workers == 4 and repetition == 2:
            factor = 2.0
        measurement = {
            "wall_seconds": float(nominal * factor),
            "engine_seconds": float(nominal * factor * 0.9),
            "resources": _resources(nominal),
        }
        with self._lock:
            self.active -= 1
        return scheduler.ExactAttempt(base, measurement)

    def final_lease(self) -> dict[str, str]:
        return {"resident_identity_digest": "a" * 64,
                "source_lease_digest": "b" * 64}


class LightProcessBackend(scheduler.ProductionBackend):
    def __init__(self, root: Path) -> None:
        self.repository = ROOT
        self._scratch = tempfile.TemporaryDirectory(
            prefix="m04r14-light-child-", dir=root,
        )
        self._child_counter = 0
        self._child_lock = Lock()
        self._proposal_leaves = {}

    def _command(self, request: Path, output: Path) -> list[str]:
        program = r'''
import json, os, pathlib, sys, time
from market_analogues.types import stable_hash
request = json.loads(pathlib.Path(sys.argv[1]).read_text())
ready = {
    "schema_version": "m04r14-exact-child-ready-v1",
    "task_id": request["task_id"], "pid": os.getpid(),
    "workers": request["workers"],
    "cpu_affinity": sorted(os.sched_getaffinity(0)),
    "ready_monotonic": time.monotonic(),
    "thread_environment": {
        key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
            "NUMBA_NUM_THREADS", "NUMBA_THREADING_LAYER",
        )
    }, "proposal_sha256": request["proposal_sha256"],
    "resident_lease_digest": "a" * 64,
}
pathlib.Path(request["ready_path"]).write_text(json.dumps(ready))
release = request["release_path"]
while release is not None and not pathlib.Path(release).exists(): time.sleep(0.005)
proposal = json.loads(pathlib.Path(request["proposal_path"]).read_text())["state"]
certificate_digest = stable_hash(request["query_id"])
matches = [{"episode_id": request["query_id"],
            "total_distance_hex": float(1).hex()}]
state = {
    "schema_version": "m04r14-exact-scheduler-attempt-v1",
    "case_id": request["case_id"], "query_id": request["query_id"],
    "workers": request["workers"],
    "proposal_semantic_digest": proposal["semantic_digest"],
    "certificate": {"result_digest": certificate_digest},
    "matches": matches, "certificate_result_digest": certificate_digest,
    "match_digest": stable_hash(matches), "lease_before": "a"*64,
    "lease_after": "a"*64,
}
payload = {
    "semantic": {"state": state, "digest": stable_hash(state)},
    "measurement": {
        "wall_seconds": 0.01, "engine_seconds": 0.009,
        "resources": {
            "before": {"user_cpu_seconds":0.,"system_cpu_seconds":0.,
              "minor_faults":0,"major_faults":0,"voluntary_context_switches":0,
              "involuntary_context_switches":0,"max_rss_mb":1.,"swap_kib":0,
              "cpu_affinity":sorted(os.sched_getaffinity(0)),
              "thread_environment":ready["thread_environment"]},
            "after": {"user_cpu_seconds":0.,"system_cpu_seconds":0.,
              "minor_faults":0,"major_faults":0,"voluntary_context_switches":0,
              "involuntary_context_switches":0,"max_rss_mb":1.,"swap_kib":0,
              "cpu_affinity":sorted(os.sched_getaffinity(0)),
              "thread_environment":ready["thread_environment"]},
            "delta":{"user_cpu_seconds":0.,"system_cpu_seconds":0.,
              "minor_faults":0,"major_faults":0,"voluntary_context_switches":0,
              "involuntary_context_switches":0,"swap_kib":0},
            "peak_rss_mb":1.,"swap_delta_kib":0,
        },
    }, "created_at": "2026-08-26T00:00:00+00:00",
}
pathlib.Path(sys.argv[2]).write_text(json.dumps(payload))
'''
        return [sys.executable, "-c", program, str(request), str(output)]


class HangingProcessBackend(LightProcessBackend):
    def _command(self, request: Path, output: Path) -> list[str]:
        return [sys.executable, "-c", "import time; time.sleep(60)"]


class ExitingProcessBackend(LightProcessBackend):
    def _command(self, request: Path, output: Path) -> list[str]:
        return [sys.executable, "-c", "raise SystemExit(7)"]


class LightProposalBackend(scheduler.ProductionBackend):
    def __init__(self, root: Path) -> None:
        self.repository = ROOT
        self.config_path = self.registry_root = self.source_full_root = root
        self.resident_root = root
        self._scratch = tempfile.TemporaryDirectory(prefix="m04r14-light-proposal-", dir=root)
        self._child_counter = 0; self._child_lock = Lock()
        self._source_bindings = {}; self._proposal_leaves = {}
        self.resident = {"identity_digest": "a" * 64}
        self.cases = tuple(
            SimpleNamespace(case_id=case_id, query_id=query_id)
            for case_id, query_id in zip(
                scheduler.THROUGHPUT_CASE_IDS, scheduler.THROUGHPUT_QUERY_IDS,
                strict=True,
            )
        )

    def _proposal_command(self, request: Path, output: Path) -> list[str]:
        program = r'''
import json, os, pathlib, sys, time
from market_analogues.types import stable_hash
request = json.loads(pathlib.Path(sys.argv[1]).read_text())
case_id, query_id = sys.argv[3], sys.argv[4]
ready = {
  "schema_version":"m04r14-proposal-child-ready-v1",
  "task_id":request["task_id"], "pid":os.getpid(),
  "cpu_affinity":sorted(os.sched_getaffinity(0)),
  "thread_environment":{k:os.environ.get(k) for k in (
    "OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS",
    "NUMBA_NUM_THREADS","NUMBA_THREADING_LAYER")},
  "resident_lease_digest":"a"*64, "ready_monotonic":time.monotonic(),
}
pathlib.Path(request["ready_path"]).write_text(json.dumps(ready))
while not pathlib.Path(request["release_path"]).exists(): time.sleep(.005)
binding={"digest":"b"*64}
report={"block_rows":4096,"block_order":"forward","elapsed_seconds":.01,
        "peak_rss_mb":1.0,"result_digest":"c"*64,"candidates":[]}
reverse=dict(report); reverse.update(block_rows=4097,block_order="reverse")
semantic={"schema_version":"m04r14-exact-scheduler-proposal-v1",
 "task_id":request["task_id"],"case_id":case_id,"query_id":query_id,
 "query_binding":binding,"resident_lease_digests":["a"*64]*4,
 "source_binding_before":binding,"source_binding_after":binding,
 "resident_snapshot":request["resident_snapshot"],"forward":report,"reverse":reverse,
 "semantic_digest":stable_hash({k:v for k,v in report.items() if k not in {
   "block_rows","block_order","elapsed_seconds","peak_rss_mb"}})}
snapshot={"user_cpu_seconds":0.,"system_cpu_seconds":0.,"minor_faults":0,
 "major_faults":0,"voluntary_context_switches":0,"involuntary_context_switches":0,
 "max_rss_mb":1.,"swap_kib":0,"cpu_affinity":sorted(os.sched_getaffinity(0)),
 "thread_environment":ready["thread_environment"]}
measurement={"wall_seconds":.02,"forward_seconds":.01,"reverse_seconds":.01,
 "resources":{"before":snapshot,"after":snapshot,"delta":{
 "user_cpu_seconds":0.,"system_cpu_seconds":0.,"minor_faults":0,"major_faults":0,
 "voluntary_context_switches":0,"involuntary_context_switches":0,"swap_kib":0},
 "peak_rss_mb":1.,"swap_delta_kib":0}}
payload={"semantic":{"state":semantic,"digest":stable_hash(semantic)},
 "measurement":measurement,"created_at":"2026-08-26T00:00:00+00:00"}
pathlib.Path(sys.argv[2]).write_text(json.dumps(payload))
'''
        ordinal = json.loads(request.read_text())["case_ordinal"]
        case = self.cases[ordinal]
        return [sys.executable, "-c", program, str(request), str(output),
                case.case_id, case.query_id]

    def _prepared_from_proposal_child(
        self, case_ordinal: int, task_id: str, semantic: dict[str, object],
        measurement: dict[str, object],
    ) -> scheduler.PreparedCase:
        case = self.cases[case_ordinal]
        return scheduler.PreparedCase(
            task_id, case.case_id, case.query_id, semantic, measurement,
        )


class HangingProposalBackend(LightProposalBackend):
    def _proposal_command(self, request: Path, output: Path) -> list[str]:
        return [sys.executable, "-c", "import time; time.sleep(60)"]


class ExitingProposalBackend(LightProposalBackend):
    def _proposal_command(self, request: Path, output: Path) -> list[str]:
        return [sys.executable, "-c", "raise SystemExit(7)"]


def _light_prepared(backend: LightProcessBackend, root: Path, index: int):
    task = f"light-{index}"
    semantic = {"semantic_digest": scheduler.stable_hash(task)}
    prepared = scheduler.PreparedCase(
        task, scheduler.THROUGHPUT_CASE_IDS[index],
        scheduler.THROUGHPUT_QUERY_IDS[index], semantic, {},
    )
    leaf = root / f"proposal-{index}.json"
    scheduler._atomic(leaf, {**scheduler._seal(semantic), "measurement": {}})
    backend.bind_proposal(prepared, leaf)
    return prepared


@pytest.fixture()
def completed(tmp_path: Path) -> tuple[Path, FakeBackend, dict[str, object]]:
    root = tmp_path / "scheduler"
    backend = FakeBackend()
    result = scheduler.execute(
        root, backend, runtime_binding=scheduler.test_foundation(),
    )
    return root, backend, result


def test_full_matrix_and_create_only_checkpoints(
    completed: tuple[Path, FakeBackend, dict[str, object]],
) -> None:
    root, backend, result = completed
    assert result["state"]["status"] == "complete"
    assert result["state"]["selected_workers"] == 4
    assert len(backend.prepares) == 4 * 3 + 8
    assert len(backend.exacts) == 4 * 4 * 3 + 8 + 8
    assert backend.maximum_active > 1
    for repetition in range(3):
        for ordinal in range(4):
            task = root / "primary" / f"r{repetition}" / f"c{ordinal}"
            assert (task / "PROPOSAL.json").is_file()
            assert len(list(task.glob("EXACT-w*.json"))) == 4
    assert len(list((root / "throughput").glob("t*/PROPOSAL.json"))) == 8
    assert len(list((root / "throughput").glob("t*/CONTROL-w1.json"))) == 8
    assert len(list((root / "throughput").glob("t*/EXACT-w1.json"))) == 8
    assert (root / "throughput/PROPOSALS_COMPLETE.json").is_file()
    assert (root / "SEMANTICS.json").is_file()
    assert (root / "MEASUREMENTS.json").is_file()
    assert not (root / "INCOMPLETE.json").exists()
    assert scheduler.validate_terminal(root) == result


def test_deterministic_rotation_and_distinct_throughput_queries(
    completed: tuple[Path, FakeBackend, dict[str, object]],
) -> None:
    root, backend, _ = completed
    assert [ordinal for ordinal, _ in backend.prepares[:12]] == [
        0, 1, 2, 3, 1, 2, 3, 0, 2, 3, 0, 1,
    ]
    assert [ordinal for ordinal, _ in backend.prepares[12:]] == list(range(8))
    barrier = json.loads((root / "throughput/PROPOSALS_COMPLETE.json").read_text())
    assert barrier["state"]["query_ids"] == list(scheduler.THROUGHPUT_QUERY_IDS)
    assert len(set(barrier["state"]["query_ids"])) == 8


def test_event_ledger_proves_serial_proposals_then_batch(
    completed: tuple[Path, FakeBackend, dict[str, object]],
) -> None:
    root, _, _ = completed
    measurements = json.loads((root / "MEASUREMENTS.json").read_text())["state"]
    ledger = measurements["event_ledger"]
    assert [row["sequence"] for row in ledger] == list(range(len(ledger)))
    barrier_index = next(
        index for index, row in enumerate(ledger)
        if row["event"] == "barrier_released"
    )
    assert sum(row["event"] == "proposal_started" for row in ledger) == 8
    assert all(row.get("active_proposals", 0) <= 1 for row in ledger)
    assert not any(
        row["event"].startswith("proposal_") for row in ledger[barrier_index + 1:]
    )
    assert measurements["throughput_lane"][
        "observed_maximum_active_exact_tasks"
    ] > 1
    assert measurements["throughput_lane"][
        "serial_controls_excluded_from_lane_timing"
    ] is True


def test_worker_selection_uses_only_primary_raw_case_medians(
    completed: tuple[Path, FakeBackend, dict[str, object]],
) -> None:
    root, _, _ = completed
    measurements = json.loads((root / "MEASUREMENTS.json").read_text())["state"]
    selection = measurements["selection"]["state"]
    assert selection["selected_workers"] == 4
    assert selection["workers"]["4"]["sample_count"] == 12
    assert selection["workers"]["4"]["raw_score_seconds"] == pytest.approx(0.4)
    assert "throughput" not in selection


def test_semantic_drift_fails_closed_with_incomplete(tmp_path: Path) -> None:
    root = tmp_path / "mutated"
    with pytest.raises(scheduler.SchedulerError, match="semantic.*differ"):
        scheduler.execute(root, FakeBackend(mutate=True))
    assert not (root / "COMPLETE.json").exists()
    incomplete = json.loads((root / "INCOMPLETE.json").read_text())
    assert incomplete["state"]["status"] == "incomplete"


def test_unstable_fast_worker_is_excluded_by_frozen_rule(tmp_path: Path) -> None:
    root = tmp_path / "unstable"
    scheduler.execute(root, FakeBackend(unstable=True))
    selection = json.loads((root / "MEASUREMENTS.json").read_text())[
        "state"
    ]["selection"]["state"]
    assert selection["workers"]["4"]["stable"] is False
    assert selection["selected_workers"] == 8


def test_output_root_is_create_only(completed: tuple[Path, FakeBackend, dict[str, object]]) -> None:
    root, _, _ = completed
    with pytest.raises(scheduler.SchedulerError, match="must be absent"):
        scheduler.execute(root, FakeBackend())


def test_resource_counter_regression_rejected() -> None:
    before = _resources()["before"]
    after = copy.deepcopy(before)
    after["minor_faults"] -= 1
    with pytest.raises(scheduler.SchedulerError, match="regressed"):
        scheduler._resource_delta(before, after)


def test_numeric_contract_distinguishes_algorithm_and_evidence_tolerances() -> None:
    assert scheduler.NUMERIC_ATOL == 1e-6
    assert scheduler.ENGINE_TOLERANCE == 1e-12
    assert scheduler.NUMERIC_ATOL.hex() != scheduler.ENGINE_TOLERANCE.hex()


def test_cgroup_cpu_quota_and_throttling_fail_performance_closed() -> None:
    before = scheduler._cgroup_cpu_snapshot()
    after = copy.deepcopy(before)
    constrained = copy.deepcopy(before)
    constrained["configuration"]["effective_quota_cpus"] = 1.0
    constrained["configuration"]["quota_usec"] = constrained[
        "configuration"
    ]["period_usec"]
    with pytest.raises(scheduler.SchedulerError, match="capacity"):
        scheduler._host_cpu_qualification(constrained, constrained)
    after["cpu_stat"]["nr_throttled"] += 1
    after["cpu_stat"]["throttled_usec"] += 10
    with pytest.raises(scheduler.SchedulerError, match="throttling"):
        scheduler._host_cpu_qualification(before, after)


def test_backend_rejects_wrong_or_duplicate_query_identity(tmp_path: Path) -> None:
    backend = FakeBackend()
    backend.throughput_query_ids = (*scheduler.THROUGHPUT_QUERY_IDS[:-1],
                                    scheduler.THROUGHPUT_QUERY_IDS[0])
    with pytest.raises(scheduler.SchedulerError, match="throughput identity"):
        scheduler.execute(tmp_path / "wrong", backend)
    assert not (tmp_path / "wrong").exists()


def test_terminal_validator_rejects_leaf_mutation_and_extra(
    completed: tuple[Path, FakeBackend, dict[str, object]],
) -> None:
    root, _, _ = completed
    leaf = root / "primary/r0/c0/EXACT-w1.json"
    original = leaf.read_bytes()
    leaf.write_bytes(original + b" ")
    with pytest.raises(scheduler.SchedulerError, match="input SHA"):
        scheduler.validate_terminal(root)
    leaf.write_bytes(original)
    (root / "EXTRA.json").write_text("{}")
    with pytest.raises(scheduler.SchedulerError, match="exact tree"):
        scheduler.validate_terminal(root)


def test_terminal_validator_reconstructs_preregistered_selection_rule(
    completed: tuple[Path, FakeBackend, dict[str, object]],
) -> None:
    root, _, _ = completed
    contract_path = root / "CONTRACT.json"
    contract = json.loads(contract_path.read_text())
    contract["state"]["selection_rule"] = "forged choose anything"
    contract["digest"] = scheduler.stable_hash(contract["state"])
    contract_path.write_text(json.dumps(contract, sort_keys=True))
    complete_path = root / "COMPLETE.json"
    complete = json.loads(complete_path.read_text())
    for row in complete["state"]["leaf_manifest"]:
        if row["path"] == "CONTRACT.json":
            row["sha256"] = scheduler._sha(contract_path)
    complete["state"]["leaf_manifest_digest"] = scheduler.stable_hash(
        complete["state"]["leaf_manifest"]
    )
    complete["digest"] = scheduler.stable_hash(complete["state"])
    complete_path.write_text(json.dumps(complete, sort_keys=True))
    with pytest.raises(scheduler.SchedulerError, match="contract reconstruction"):
        scheduler.validate_terminal(root)


def test_production_shaped_terminal_requires_child_and_source_evidence(
    tmp_path: Path,
) -> None:
    class ProductionShapedFake(FakeBackend):
        def foundation(self) -> dict[str, object]:
            return {
                "resident": {"identity_digest": "a" * 64},
                "causal_input_shas": {"case": "b" * 64},
            }

        def final_lease(self) -> dict[str, object]:
            binding_digest = scheduler.stable_hash({
                "digest": scheduler.stable_hash(
                    scheduler.THROUGHPUT_QUERY_IDS[0]
                )
            })
            return {
                "resident_identity_digest": "a" * 64,
                "source_binding_digests": {
                    str(index): binding_digest for index in range(8)
                },
                "causal_input_shas": {"case": "b" * 64},
            }

    root = tmp_path / "production-shaped"
    with pytest.raises(
        scheduler.SchedulerError,
        match="runtime seal|production proposal process|live backend binding",
    ):
        scheduler.execute(root, ProductionShapedFake())
    assert not (root / "COMPLETE.json").exists()
    assert (root / "INCOMPLETE.json").is_file()


def test_real_serial_child_gets_worker_sized_affinity_and_wait4_metrics(
    tmp_path: Path,
) -> None:
    backend = LightProcessBackend(tmp_path)
    prepared = _light_prepared(backend, tmp_path, 0)
    attempts, evidence = backend._launch_children(((prepared, 4),))
    assert attempts[0].semantic["case_id"] == scheduler.THROUGHPUT_CASE_IDS[0]
    assert attempts[0].semantic["workers"] == 4
    child = evidence["children"][0]
    assert len(child["cpus"]) == 4
    assert child["user_cpu_seconds"] >= 0
    assert child["peak_hwm_kib"] >= 0
    assert child["peak_swap_kib"] == child["final_swap_kib"] == 0


def test_real_p8_children_wait_for_one_release_barrier(tmp_path: Path) -> None:
    backend = LightProcessBackend(tmp_path)
    prepared = [_light_prepared(backend, tmp_path, index) for index in range(8)]
    attempts, evidence = backend._launch_children(
        tuple((value, 1) for value in prepared),
    )
    assert len(attempts) == 8
    assert evidence["observed_concurrent_children"] == 8
    assert len({row["pid"] for row in evidence["ready_evidence"]}) == 8
    assert all(len(row["cpu_affinity"]) == 1 for row in evidence["ready_evidence"])
    assert evidence["release_evidence"]["schema_version"].endswith("release-v1")
    assert evidence["release_to_all_children_exit_seconds"] > 0


def test_child_ready_timeout_kills_and_reaps_every_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = HangingProcessBackend(tmp_path)
    prepared = [_light_prepared(backend, tmp_path, index) for index in range(2)]
    monkeypatch.setattr(scheduler, "CHILD_STARTUP_TIMEOUT_SECONDS", 0.1)
    with pytest.raises(scheduler.SchedulerError, match="READY barrier"):
        backend._launch_children(tuple((value, 1) for value in prepared))
    assert not list(Path(backend._scratch.name).glob("ready-*.json"))


def test_child_ready_schema_failure_kills_and_reaps_every_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = LightProcessBackend(tmp_path)
    prepared = [_light_prepared(backend, tmp_path, index) for index in range(2)]
    original_read = scheduler._read_json
    child_pids: list[int] = []

    def forged_ready(path: Path, **kwargs: object) -> dict[str, object]:
        payload = original_read(path, **kwargs)
        if path.name.startswith("ready-"):
            child_pids.append(payload["pid"])
            payload["unexpected"] = "must fail closed"
        return payload

    monkeypatch.setattr(scheduler, "_read_json", forged_ready)
    with pytest.raises(scheduler.SchedulerError, match="READY exact schema"):
        backend._launch_children(tuple((value, 1) for value in prepared))
    assert len(child_pids) == 2
    assert all(not Path(f"/proc/{pid}").exists() for pid in child_pids)


def test_child_monitor_exception_kills_and_reaps_every_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = LightProcessBackend(tmp_path)
    prepared = [_light_prepared(backend, tmp_path, index) for index in range(2)]
    original_popen = scheduler.subprocess.Popen
    processes: list[object] = []

    def recording_popen(*args: object, **kwargs: object) -> object:
        process = original_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(scheduler.subprocess, "Popen", recording_popen)
    monkeypatch.setattr(
        scheduler, "_proc_memory",
        lambda _pid: (_ for _ in ()).throw(RuntimeError("injected monitor failure")),
    )
    with pytest.raises(RuntimeError, match="monitor failure"):
        backend._launch_children(tuple((value, 1) for value in prepared))
    assert len(processes) == 2
    assert all(process.returncode is not None for process in processes)
    assert all(not Path(f"/proc/{process.pid}").exists() for process in processes)


def test_strict_json_rejects_duplicate_nonfinite_and_symlink(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}')
    with pytest.raises(scheduler.SchedulerError, match="duplicate"):
        scheduler._read_json(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a":NaN}')
    with pytest.raises(scheduler.SchedulerError, match="nonfinite"):
        scheduler._read_json(nonfinite)
    alias = tmp_path / "alias.json"; alias.symlink_to(duplicate)
    with pytest.raises(scheduler.SchedulerError, match="symlink"):
        scheduler._read_json(alias)


def test_real_opened_workload_selection_reconstructs_exact_quartiles() -> None:
    binding = scheduler._throughput_selection_binding(ROOT)["state"]
    assert len(binding["full_non_hard_transcript"]) == 56
    assert [row["query_id"] for row in binding["selected"]] == list(
        scheduler.THROUGHPUT_EXTRA_QUERY_IDS
    )
    assert binding["source_selection_digest"] == scheduler.THROUGHPUT_SELECTION_DIGEST


def test_fresh_proposal_child_uses_eight_cpus_and_records_process_evidence(
    tmp_path: Path,
) -> None:
    backend = LightProposalBackend(tmp_path)
    prepared = backend.prepare(0, "proposal-light-0")
    process = prepared.measurement["spawned_process"]
    assert prepared.query_id == scheduler.THROUGHPUT_QUERY_IDS[0]
    assert len(process["cpus"]) == scheduler.PROPOSAL_THREADS
    assert process["ready_evidence"]["pid"] == process["pid"]
    assert process["release_evidence"]["pid"] == process["pid"]
    assert process["peak_swap_kib"] == process["final_swap_kib"] == 0


def test_proposal_process_validator_rejects_self_consistent_wrong_context(
    tmp_path: Path,
) -> None:
    backend = LightProposalBackend(tmp_path)
    process = copy.deepcopy(
        backend.prepare(0, "proposal-context").measurement["spawned_process"]
    )
    process["ready_evidence"]["task_id"] = "forged-task"
    process["ready_evidence"]["thread_environment"] = {}
    process["ready_evidence"]["ready_monotonic"] = "not-a-number"
    process["release_evidence"]["task_id"] = "forged-task"
    process["release_evidence"]["ready_digest"] = scheduler.stable_hash(
        process["ready_evidence"]
    )
    with pytest.raises(scheduler.SchedulerError, match="barrier crosslink"):
        scheduler._validate_proposal_process(process)


def test_proposal_child_ready_timeout_kills_and_reaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = HangingProposalBackend(tmp_path)
    monkeypatch.setattr(scheduler, "CHILD_STARTUP_TIMEOUT_SECONDS", 0.1)
    with pytest.raises(scheduler.SchedulerError, match="READY timed out"):
        backend.prepare(0, "proposal-hang")


def test_proposal_child_early_exit_fails_immediately(tmp_path: Path) -> None:
    backend = ExitingProposalBackend(tmp_path)
    started = time.monotonic()
    with pytest.raises(scheduler.SchedulerError, match="exited before READY"):
        backend.prepare(0, "proposal-exit")
    assert time.monotonic() - started < 5


def test_proposal_ready_schema_failure_kills_and_reaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = LightProposalBackend(tmp_path)
    original_read = scheduler._read_json
    child_pids: list[int] = []

    def forged_ready(path: Path, **kwargs: object) -> dict[str, object]:
        payload = original_read(path, **kwargs)
        if path.name.startswith("proposal-ready-"):
            child_pids.append(payload["pid"])
            payload["unexpected"] = True
        return payload

    monkeypatch.setattr(scheduler, "_read_json", forged_ready)
    with pytest.raises(scheduler.SchedulerError, match="READY exact schema"):
        backend.prepare(0, "proposal-schema")
    assert len(child_pids) == 1
    assert not Path(f"/proc/{child_pids[0]}").exists()


def test_exact_child_early_death_fails_immediately_and_reaps_siblings(
    tmp_path: Path,
) -> None:
    backend = ExitingProcessBackend(tmp_path)
    prepared = [_light_prepared(backend, tmp_path, index) for index in range(2)]
    started = time.monotonic()
    with pytest.raises(scheduler.SchedulerError, match="exited before READY"):
        backend._launch_children(tuple((value, 1) for value in prepared))
    assert time.monotonic() - started < 5


def test_terminal_rejects_coordinated_resealed_aggregate_extra_claim(
    completed: tuple[Path, FakeBackend, dict[str, object]],
) -> None:
    root, _, _ = completed
    semantics_path = root / "SEMANTICS.json"
    semantics = json.loads(semantics_path.read_text())
    semantics["state"]["forged_unvalidated_claim"] = True
    semantics["digest"] = scheduler.stable_hash(semantics["state"])
    semantics_path.write_bytes(scheduler._strict_bytes(semantics) + b"\n")
    complete_path = root / "COMPLETE.json"
    complete = json.loads(complete_path.read_text())
    complete["state"]["semantic_digest"] = semantics["digest"]
    for row in complete["state"]["leaf_manifest"]:
        if row["path"] == "SEMANTICS.json":
            row["sha256"] = scheduler._sha(semantics_path)
    complete["state"]["leaf_manifest_digest"] = scheduler.stable_hash(
        complete["state"]["leaf_manifest"]
    )
    complete["digest"] = scheduler.stable_hash(complete["state"])
    complete_path.write_bytes(scheduler._strict_bytes(complete) + b"\n")
    with pytest.raises(scheduler.SchedulerError, match="semantics state exact schema"):
        scheduler.validate_terminal(root)


def test_terminal_rejects_forged_final_lease_and_duplicate_manifest(
    completed: tuple[Path, FakeBackend, dict[str, object]],
) -> None:
    root, _, _ = completed
    complete_path = root / "COMPLETE.json"
    original = complete_path.read_bytes()
    complete = json.loads(original)
    complete["state"]["final_lease"] = {"forged": True}
    complete["digest"] = scheduler.stable_hash(complete["state"])
    complete_path.write_bytes(scheduler._strict_bytes(complete) + b"\n")
    with pytest.raises(scheduler.SchedulerError, match="final lease"):
        scheduler.validate_terminal(root)
    complete_path.write_bytes(original)
    complete = json.loads(original)
    complete["state"]["leaf_manifest"].append(
        copy.deepcopy(complete["state"]["leaf_manifest"][0])
    )
    complete["state"]["leaf_manifest_digest"] = scheduler.stable_hash(
        complete["state"]["leaf_manifest"]
    )
    complete["digest"] = scheduler.stable_hash(complete["state"])
    complete_path.write_bytes(scheduler._strict_bytes(complete) + b"\n")
    with pytest.raises(scheduler.SchedulerError, match="manifest paths"):
        scheduler.validate_terminal(root)


def test_terminal_rejects_rehashed_forged_event_sequence(
    completed: tuple[Path, FakeBackend, dict[str, object]],
) -> None:
    root, _, _ = completed
    measurements_path = root / "MEASUREMENTS.json"
    measurements = json.loads(measurements_path.read_text())
    ledger = measurements["state"]["event_ledger"]
    ledger[0]["event"] = "forged_event"
    previous = None
    for event in ledger:
        event["previous_event_digest"] = previous
        event["event_digest"] = scheduler.stable_hash(
            scheduler._without(event, {"event_digest"})
        )
        previous = event["event_digest"]
    measurements["state"]["event_ledger_digest"] = scheduler.stable_hash(ledger)
    measurements["digest"] = scheduler.stable_hash(measurements["state"])
    measurements_path.write_bytes(scheduler._strict_bytes(measurements) + b"\n")
    complete_path = root / "COMPLETE.json"
    complete = json.loads(complete_path.read_text())
    complete["state"]["measurement_digest"] = measurements["digest"]
    for row in complete["state"]["leaf_manifest"]:
        if row["path"] == "MEASUREMENTS.json":
            row["sha256"] = scheduler._sha(measurements_path)
    complete["state"]["leaf_manifest_digest"] = scheduler.stable_hash(
        complete["state"]["leaf_manifest"]
    )
    complete["digest"] = scheduler.stable_hash(complete["state"])
    complete_path.write_bytes(scheduler._strict_bytes(complete) + b"\n")
    with pytest.raises(scheduler.SchedulerError, match="proposal event sequence"):
        scheduler.validate_terminal(root)


def test_terminal_reconstructs_concurrency_counters(
    completed: tuple[Path, FakeBackend, dict[str, object]],
) -> None:
    root, _, _ = completed
    measurements_path = root / "MEASUREMENTS.json"
    measurements = json.loads(measurements_path.read_text())
    ledger = measurements["state"]["event_ledger"]
    for event in ledger:
        if event["event"] in {"batch_exact_started", "batch_exact_ended"}:
            event["active_exact_tasks"] = -999
    measurements["state"]["throughput_lane"][
        "observed_maximum_active_exact_tasks"
    ] = 2
    previous = None
    for event in ledger:
        event["previous_event_digest"] = previous
        event["event_digest"] = scheduler.stable_hash(
            scheduler._without(event, {"event_digest"})
        )
        previous = event["event_digest"]
    measurements["state"]["event_ledger_digest"] = scheduler.stable_hash(ledger)
    measurements["digest"] = scheduler.stable_hash(measurements["state"])
    measurements_path.write_bytes(scheduler._strict_bytes(measurements) + b"\n")
    complete_path = root / "COMPLETE.json"
    complete = json.loads(complete_path.read_text())
    complete["state"]["measurement_digest"] = measurements["digest"]
    for row in complete["state"]["leaf_manifest"]:
        if row["path"] == "MEASUREMENTS.json":
            row["sha256"] = scheduler._sha(measurements_path)
    complete["state"]["leaf_manifest_digest"] = scheduler.stable_hash(
        complete["state"]["leaf_manifest"]
    )
    complete["digest"] = scheduler.stable_hash(complete["state"])
    complete_path.write_bytes(scheduler._strict_bytes(complete) + b"\n")
    with pytest.raises(scheduler.SchedulerError, match="active counter"):
        scheduler.validate_terminal(root)


def test_terminal_final_recheck_detects_split_read_mutation(
    completed: tuple[Path, FakeBackend, dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _, _ = completed
    target = root / "primary/r0/c0/PROPOSAL.json"
    original_read = scheduler._read_json
    race_fired = False

    def transient_read(path: Path, **kwargs: object) -> dict[str, object]:
        nonlocal race_fired
        if path == target and not race_fired:
            race_fired = True
            original = path.read_bytes()
            payload = json.loads(original)
            payload["created_at"] = "2030-01-01T00:00:00+00:00"
            path.write_bytes(scheduler._strict_bytes(payload) + b"\n")
            try:
                return original_read(path, **kwargs)
            finally:
                path.write_bytes(original)
        return original_read(path, **kwargs)

    monkeypatch.setattr(scheduler, "_read_json", transient_read)
    with pytest.raises(scheduler.SchedulerError, match="input SHA"):
        scheduler.validate_terminal(root)
    assert race_fired


def test_historical_m11_adapter_rejects_rehashed_contract_forgery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = scheduler._read_json
    def forged(path: Path, **kwargs: object) -> dict[str, object]:
        value = original(path, **kwargs)
        if path.name == "candidate-contract.json":
            value["forged_claim"] = True
            value["contract_digest"] = scheduler.stable_hash(
                scheduler._without(value, {"contract_digest"})
            )
        return value
    monkeypatch.setattr(scheduler, "_read_json", forged)
    with pytest.raises(scheduler.SchedulerError, match="historical.*contract"):
        scheduler._authenticate_m11_historical(ROOT)


def test_runtime_validator_rejects_empty_self_sealed_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {
        "git_head": "0" * 40, "files": {},
        "environment": scheduler._environment_binding(),
        "contracts": {
            "schema_version": scheduler.SCHEMA,
            "execution_policy": scheduler._execution_policy(),
        },
    }
    monkeypatch.setattr(scheduler, "_runtime_file_set", lambda *_: ("mandatory.py",))
    with pytest.raises(scheduler.SchedulerError, match="runtime manifest contract"):
        scheduler._validate_runtime(scheduler._seal(state), ROOT)


def test_h1_must_be_one_preregistration_only_commit(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=repository, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "T14 test"],
        cwd=repository, check=True,
    )
    (repository / "implementation.py").write_text("frozen = True\n")
    subprocess.run(["git", "add", "implementation.py"], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "H0"], cwd=repository, check=True,
    )
    h0 = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    preregistration = repository / "preregistered.json"
    preregistration.write_text('{"frozen":true}\n')
    subprocess.run(["git", "add", "preregistered.json"], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "H1"], cwd=repository, check=True,
    )
    scheduler._validate_h1_commit(repository, preregistration, h0)
    (repository / "unrelated.txt").write_text("must not share H1\n")
    subprocess.run(["git", "add", "unrelated.txt"], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-q", "--amend", "--no-edit"],
        cwd=repository, check=True,
    )
    with pytest.raises(scheduler.SchedulerError, match="preregistration-only H1"):
        scheduler._validate_h1_commit(repository, preregistration, h0)


def test_strict_reader_rejects_overflow_to_infinity(tmp_path: Path) -> None:
    path = tmp_path / "overflow.json"
    path.write_text('{"value":1e999}')
    with pytest.raises(scheduler.SchedulerError, match="nonfinite"):
        scheduler._read_json(path)


def test_complete_is_not_published_when_preterminal_validation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise scheduler.SchedulerError("injected preterminal failure")
    monkeypatch.setattr(scheduler, "validate_terminal", fail)
    root = tmp_path / "preterminal"
    with pytest.raises(scheduler.SchedulerError, match="preterminal failure"):
        scheduler.execute(root, FakeBackend())
    assert not (root / "COMPLETE.json").exists()
    assert (root / "INCOMPLETE.json").is_file()


def test_run_hard_deadline_is_rechecked_after_final_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SlowFinalLease(FakeBackend):
        def final_lease(self) -> dict[str, str]:
            time.sleep(0.3)
            return super().final_lease()

    monkeypatch.setattr(scheduler, "RUN_HARD_LIMIT_SECONDS", 0.2)
    root = tmp_path / "deadline"
    with pytest.raises(scheduler.SchedulerError, match="hard limit"):
        scheduler.execute(root, SlowFinalLease())
    assert not (root / "COMPLETE.json").exists()
    assert (root / "INCOMPLETE.json").is_file()


def test_terminal_rejects_rehashed_forged_lane_resources(
    completed: tuple[Path, FakeBackend, dict[str, object]],
) -> None:
    root, _, _ = completed
    measurements_path = root / "MEASUREMENTS.json"
    measurements = json.loads(measurements_path.read_text())
    measurements["state"]["throughput_lane"]["resources"]["forged"] = True
    measurements["digest"] = scheduler.stable_hash(measurements["state"])
    measurements_path.write_bytes(scheduler._strict_bytes(measurements) + b"\n")
    complete_path = root / "COMPLETE.json"
    complete = json.loads(complete_path.read_text())
    complete["state"]["measurement_digest"] = measurements["digest"]
    for row in complete["state"]["leaf_manifest"]:
        if row["path"] == "MEASUREMENTS.json":
            row["sha256"] = scheduler._sha(measurements_path)
    complete["state"]["leaf_manifest_digest"] = scheduler.stable_hash(
        complete["state"]["leaf_manifest"]
    )
    complete["digest"] = scheduler.stable_hash(complete["state"])
    complete_path.write_bytes(scheduler._strict_bytes(complete) + b"\n")
    with pytest.raises(scheduler.SchedulerError, match="resource evidence"):
        scheduler.validate_terminal(root)


def test_partial_spawn_failure_kills_and_reaps_started_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = HangingProcessBackend(tmp_path)
    prepared = [_light_prepared(backend, tmp_path, index) for index in range(2)]
    original = scheduler.subprocess.Popen
    started: list[object] = []
    def fail_second(*args: object, **kwargs: object) -> object:
        if started:
            raise OSError("injected second Popen failure")
        process = original(*args, **kwargs); started.append(process); return process
    monkeypatch.setattr(scheduler.subprocess, "Popen", fail_second)
    with pytest.raises(OSError, match="second Popen"):
        backend._launch_children(tuple((value, 1) for value in prepared))
    assert len(started) == 1
    with pytest.raises(ProcessLookupError):
        os.kill(started[0].pid, 0)


def test_release_publication_failure_kills_and_reaps_all_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = LightProcessBackend(tmp_path)
    prepared = [_light_prepared(backend, tmp_path, index) for index in range(2)]
    original = scheduler._atomic
    pids: list[int] = []
    original_popen = scheduler.subprocess.Popen
    def observed_popen(*args: object, **kwargs: object) -> object:
        process = original_popen(*args, **kwargs); pids.append(process.pid); return process
    def fail_release(path: Path, payload: object) -> None:
        if path.name.startswith("release-"):
            raise OSError("injected release publication failure")
        original(path, payload)
    monkeypatch.setattr(scheduler.subprocess, "Popen", observed_popen)
    monkeypatch.setattr(scheduler, "_atomic", fail_release)
    with pytest.raises(OSError, match="release publication"):
        backend._launch_children(tuple((value, 1) for value in prepared))
    assert len(pids) == 2
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
