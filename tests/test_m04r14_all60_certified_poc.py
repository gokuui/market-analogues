from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.m04r import m04r14_all60_certified_poc as producer
from experiments.m04r import m04r14_all60_contract as contract


FIXED_TIME = "2026-08-26T00:00:00+00:00"
LEASE_STATE = {"schema_version": "m04r-resident-packed-store-lease-v1",
    "ready_digest": "7"*64, "ready_file_sha256": "8"*64,
    "content_digest": "9"*64,
    "files": {"pack": {"path": "/dev/shm/store/pack", "st_dev": 1,
        "st_ino": 2, "st_size": 3, "st_mtime_ns": 4, "st_ctime_ns": 5,
        "st_mode": 33188}}}
LEASE_DIGEST = contract.stable_digest(LEASE_STATE)


def _runtime_binding():
    state = {"git_head": "1" * 40,
             "files": {name: "2"*64 for name in producer.RUNTIME_FIXED_FILES},
             "environment": {"state": {"python": "3.12"}, "digest": "3"*64},
             "contracts": {"descriptor_digest": contract.DESCRIPTOR_DIGEST,
                           "execution_policy": contract.EXECUTION_POLICY}}
    return {"state": state, "digest": contract.stable_digest(state)}


def _prereg(root: Path):
    return producer.build_preregistration(runtime_binding=_runtime_binding(), roots={
        "candidate": str(root), "config": "/inputs/config.json",
        "registry": "/inputs/registry", "source": "/inputs/source",
        "resident": "/dev/shm/resident",
    })


class FakeBackend:
    def __init__(self, *, fail_at=None, fail_stage=None, parity=True,
                 performance=True, drift=False):
        self.cases = tuple(SimpleNamespace(ordinal=i, case_id=f"case-{i}",
            query_id=query, registry_case={}) for i, query in enumerate(contract.QUERY_IDS))
        self.fail_at = fail_at; self.fail_stage = fail_stage
        self.parity = parity; self.performance = performance; self.drift = drift
        self.bound = {}; self.calls = []

    def source_lock(self):
        return {"config_sha256": "1"*64, "registry_sha256": "2"*64,
            "registry_digest": "3"*64, "generation_id": "generation",
            "provenance_digest": "4"*64, "source_tree_digest": "5"*64,
            "query_binding_digests": [contract.stable_digest({"digest": "e"*64})]*60}

    def resident_binding(self):
        value = {"ready_digest": "7"*64, "ready_file_sha256": "8"*64,
            "content_digest": "9"*64, "seal_digest": "a"*64,
            "lease": {**LEASE_STATE, "lease_digest": LEASE_DIGEST}, "store_root": "/dev/shm/store",
            }
        return {**value, "identity_digest": contract.stable_digest(value)}

    def prepare(self, ordinal, task_id):
        self.calls.append(("prepare", ordinal))
        if self.fail_at == ordinal and self.fail_stage in {"prepare", "hang"}:
            raise TimeoutError("fake child timed out")
        case = self.cases[ordinal]
        report = {"candidates": [{"episode_id": case.query_id}],
                  "candidate_digest": contract.stable_digest([case.query_id]),
                  "block_rows": 4096, "block_order": "forward",
                  "elapsed_seconds": .01, "peak_rss_mb": 1.0}
        reverse = deepcopy(report); reverse["block_rows"] = 4097
        reverse["block_order"] = "reverse"
        if not self.parity and ordinal == 0:
            reverse["candidates"] = []
        semantic = {"schema_version": "m04r14-exact-scheduler-proposal-v1",
            "task_id": task_id, "case_id": case.case_id,
            "query_id": case.query_id, "query_binding": {"digest": "e"*64},
            "resident_lease_digests": [LEASE_DIGEST]*4,
            "resident_snapshot": self.resident_binding(),
            "source_binding_before": {"digest": "e"*64},
            "source_binding_after": {"digest": "e"*64},
            "forward": report, "reverse": reverse}
        semantic["semantic_digest"] = contract.stable_digest(producer._strip_timing(report))
        resources = {"after": {"swap_kib": 0 if self.performance else 1}}
        return SimpleNamespace(task_id=task_id, case=case, semantic=semantic,
            measurement={"forward_seconds": .01, "reverse_seconds": .01,
                         "resources": resources,
                         "spawned_process": {"workers": 8, "effective_peak_rss_kib": 1024,
                                             "peak_swap_kib": 0, "final_swap_kib": 0}})

    def bind_proposal(self, prepared, path):
        self.calls.append(("bind", prepared.case.ordinal)); self.bound[prepared.task_id] = path
        if self.fail_at == prepared.case.ordinal and self.fail_stage == "bind":
            raise RuntimeError("fake lease drift")

    def exact(self, prepared, workers=1):
        self.calls.append(("exact", prepared.case.ordinal))
        assert workers == 1 and prepared.task_id in self.bound
        if self.fail_at == prepared.case.ordinal and self.fail_stage == "exact":
            raise RuntimeError("fake exact child died")
        case = prepared.case
        matches = [{"episode_id": case.query_id, "total_distance_hex": float(1).hex()}]
        semantic = {"schema_version": "m04r14-exact-scheduler-attempt-v1",
            "case_id": case.case_id, "query_id": case.query_id,
            "workers": 1, "proposal_semantic_digest": prepared.semantic["semantic_digest"],
            "certificate": {"result_digest": "f"*64},
            "matches": matches,
            "certificate_result_digest": "f"*64,
            "match_digest": contract.stable_digest(matches),
            "lease_before": LEASE_DIGEST, "lease_after": LEASE_DIGEST,
            "source_binding_before": {"digest": "e"*64},
            "source_binding_after": {"digest": "e"*64}}
        swap = 0 if self.performance else 1
        return SimpleNamespace(semantic=semantic,
            measurement={"wall_seconds": .02, "engine_seconds": .01,
                "resources": {"after": {"swap_kib": swap}},
                "spawned_process": {"workers": 1, "peak_swap_kib": 0,
                                    "final_swap_kib": 0}})

    def final_source_lease(self):
        return {"source_tree_digest": ("0" if self.drift else "5")*64,
                "query_binding_digests": [contract.stable_digest({"digest": "e"*64})]*60}

    def final_resident_lease(self):
        identity = self.resident_binding()["identity_digest"]
        return {"identity_digest": "0"*64 if self.drift else identity,
                "lease_digest": LEASE_DIGEST}

    @staticmethod
    def validate_certified(certificate, matches, query_id, query_binding):
        assert certificate == {"result_digest": "f"*64}
        assert type(matches) is list and matches[0]["episode_id"] == query_id


def _run(tmp_path: Path, backend=None):
    root = tmp_path / "evidence"; prereg = _prereg(root)
    result = producer.execute(root, prereg, backend or FakeBackend(), clock=lambda: FIXED_TIME)
    return root, prereg, result


def test_fake_all60_happy_path_and_strict_tree(tmp_path: Path):
    backend = FakeBackend(); root, prereg, result = _run(tmp_path, backend)
    assert result["semantic_passed"] is True and result["performance_passed"] is True
    assert tuple(sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())) == contract.successful_tree()
    assert producer.validate_terminal(root, prereg,
        certified_validator=backend.validate_certified) == result
    assert backend.calls == [item for ordinal in range(60) for item in (
        ("prepare", ordinal), ("bind", ordinal), ("exact", ordinal))]


def test_timing_free_semantic_digest_is_deterministic(tmp_path: Path, monkeypatch):
    _, _, first = _run(tmp_path / "a")
    _, _, second = _run(tmp_path / "b")
    assert first["semantic_digest"] == second["semantic_digest"]


def test_m13_raw_validator_view_restores_only_ephemeral_elapsed_seconds() -> None:
    semantic = {"result_digest": "f" * 64}
    assert producer._m13_certificate_view(semantic) == {
        "result_digest": "f" * 64, "elapsed_seconds": 0.0,
    }
    assert semantic == {"result_digest": "f" * 64}
    with pytest.raises(producer.All60Error, match="timing-free"):
        producer._m13_certificate_view({**semantic, "elapsed_seconds": 0.01})


@pytest.mark.parametrize("stage,expected", [
    ("prepare", "started"), ("hang", "started"),
    ("bind", "proposal"), ("exact", "proposal"),
])
def test_child_failure_or_hang_terminalizes_gap_free_prefix(tmp_path: Path, stage: str, expected: str):
    root = tmp_path / stage; prereg = _prereg(root)
    with pytest.raises(Exception):
        producer.execute(root, prereg, FakeBackend(fail_at=7, fail_stage=stage), clock=lambda: FIXED_TIME)
    incomplete = producer._strict_read(root / "INCOMPLETE.json")
    assert incomplete["completed_cases"] == 7 and incomplete["trailing_stage"] == expected
    assert incomplete["resume_authorized"] is False
    assert producer._tree(root) == contract.incomplete_tree(7, trailing_stage=expected)


def test_semantic_and_performance_failures_remain_separate(tmp_path: Path):
    _, _, performance = _run(tmp_path / "perf", FakeBackend(performance=False))
    assert performance["semantic_passed"] is True and performance["performance_passed"] is False
    root = tmp_path / "semantic" / "evidence"; prereg = _prereg(root)
    with pytest.raises(producer.All60Error, match="proposal identity"):
        producer.execute(root, prereg, FakeBackend(parity=False), clock=lambda: FIXED_TIME)
    assert producer._strict_read(root / "INCOMPLETE.json")["resume_authorized"] is False


def test_holes_extras_symlinks_partial_json_and_reordering_are_rejected(tmp_path: Path):
    root, prereg, _ = _run(tmp_path)
    victim = root / f"cases/000-{contract.QUERY_IDS[0]}/EXACT-w1.json"
    raw = victim.read_bytes(); victim.unlink()
    with pytest.raises(producer.All60Error, match="tree"):
        producer.validate_terminal(root, prereg)
    victim.write_bytes(raw); (root / "EXTRA.json").write_text("{}")
    with pytest.raises(producer.All60Error, match="tree"):
        producer.validate_terminal(root, prereg)
    (root / "EXTRA.json").unlink(); victim.unlink(); victim.symlink_to(root / "COMPLETE.json")
    with pytest.raises(producer.All60Error, match="forbidden|tree"):
        producer.validate_terminal(root, prereg)
    victim.unlink(); victim.write_text("{")
    with pytest.raises(producer.All60Error):
        producer.validate_terminal(root, prereg)
    bad = FakeBackend(); bad.cases = tuple(reversed(bad.cases))
    other = tmp_path / "reordered"; pre = _prereg(other)
    with pytest.raises(producer.All60Error, match="order"):
        producer.execute(other, pre, bad, clock=lambda: FIXED_TIME)


def test_lease_drift_failure_and_root_relaunch_are_terminal(tmp_path: Path):
    root = tmp_path / "drift"; prereg = _prereg(root)
    with pytest.raises(RuntimeError):
        producer.execute(root, prereg, FakeBackend(fail_at=3, fail_stage="bind"), clock=lambda: FIXED_TIME)
    with pytest.raises(producer.All60Error, match="exists"):
        producer.execute(root, prereg, FakeBackend(), clock=lambda: FIXED_TIME)


def test_final_lease_drift_terminalizes_after_two_aggregates(tmp_path: Path):
    root = tmp_path / "final-drift"; prereg = _prereg(root)
    with pytest.raises(producer.All60Error, match="lease drifted"):
        producer.execute(root, prereg, FakeBackend(drift=True), clock=lambda: FIXED_TIME)
    terminal = producer._strict_read(root / "INCOMPLETE.json")
    assert terminal["completed_cases"] == 60
    assert producer._tree(root) == contract.incomplete_tree(60, aggregate_count=2)


@pytest.mark.parametrize("leaf,field,digest_field", [
    ("SEMANTICS.json", "semantic_passed", "semantic_digest"),
    ("MEASUREMENTS.json", "performance_passed", "measurement_digest"),
])
def test_crash_recovery_reconstructs_aggregate_meaning(
    tmp_path: Path, leaf: str, field: str, digest_field: str,
):
    root = tmp_path / leaf.split(".")[0]; prereg = _prereg(root)
    with pytest.raises(producer.All60Error, match="lease drifted"):
        producer.execute(root, prereg, FakeBackend(drift=True), clock=lambda: FIXED_TIME)
    (root / "INCOMPLETE.json").unlink()
    path = root / leaf; value = producer._strict_read(path)
    value[field] = not value[field]
    value[digest_field] = contract.stable_digest({key: item for key, item in value.items()
                                                  if key != digest_field})
    path.write_bytes(contract.canonical_bytes(value) + b"\n")
    with pytest.raises(producer.All60Error, match="prefix semantics|prefix measurements"):
        producer.seal_interrupted_prefix(root, prereg, clock=lambda: FIXED_TIME,
            certified_validator=FakeBackend.validate_certified)
    assert not (root / "INCOMPLETE.json").exists()


def test_crash_prefix_recovery_validates_chain_and_never_resumes(tmp_path: Path):
    root = tmp_path / "recover"; prereg = _prereg(root)
    with pytest.raises(RuntimeError):
        producer.execute(root, prereg, FakeBackend(fail_at=4, fail_stage="exact"), clock=lambda: FIXED_TIME)
    (root / "INCOMPLETE.json").unlink()
    terminal = producer.seal_interrupted_prefix(root, prereg, clock=lambda: FIXED_TIME,
        certified_validator=FakeBackend.validate_certified)
    assert terminal["completed_cases"] == 4 and terminal["trailing_stage"] == "proposal"
    with pytest.raises(producer.All60Error, match="terminal"):
        producer.seal_interrupted_prefix(root, prereg,
            certified_validator=FakeBackend.validate_certified)
    forged = tmp_path / "forged"; pre = _prereg(forged)
    with pytest.raises(RuntimeError):
        producer.execute(forged, pre, FakeBackend(fail_at=4, fail_stage="exact"), clock=lambda: FIXED_TIME)
    (forged / "INCOMPLETE.json").unlink()
    event_path = forged / f"events/002-CASE_STARTED-{contract.QUERY_IDS[1]}.json"
    event = producer._strict_read(event_path); event["previous_event_digest"] = "0"*64
    event["event_digest"] = contract.stable_digest({k:v for k,v in event.items() if k != "event_digest"})
    event_path.write_bytes(contract.canonical_bytes(event) + b"\n")
    with pytest.raises(producer.All60Error, match="chain"):
        producer.seal_interrupted_prefix(forged, pre,
            certified_validator=FakeBackend.validate_certified)


def test_nonfinite_and_bool_measurements_fail_before_case_checkpoint(tmp_path: Path):
    class Invalid(FakeBackend):
        def prepare(self, ordinal, task_id):
            value = super().prepare(ordinal, task_id)
            if ordinal == 0: value.measurement["forward_seconds"] = True
            return value
    root = tmp_path / "invalid"; prereg = _prereg(root)
    with pytest.raises(producer.All60Error, match="numeric"):
        producer.execute(root, prereg, Invalid(), clock=lambda: FIXED_TIME)
    assert producer._strict_read(root / "INCOMPLETE.json")["trailing_stage"] == "exact"


def test_prereg_digest_and_runtime_h0_are_reconstructed(tmp_path: Path):
    value = _prereg(tmp_path / "candidate")
    forged = deepcopy(value); forged["runtime_binding"]["state"]["git_head"] = "2"*40
    with pytest.raises(producer.All60Error): producer.validate_preregistration(forged)
    forged = deepcopy(value); forged["query_ids"][0], forged["query_ids"][1] = forged["query_ids"][1], forged["query_ids"][0]
    forged["preregistration_digest"] = contract.stable_digest({k:v for k,v in forged.items() if k != "preregistration_digest"})
    with pytest.raises(producer.All60Error): producer.validate_preregistration(forged)


def test_api_contains_no_authority_or_outcome_inputs():
    source = Path(producer.__file__).read_text()
    assert "--authority" not in source and "--outcome" not in source


def test_manifest_omission_and_empty_directory_are_rejected(tmp_path: Path):
    root, prereg, _ = _run(tmp_path / "manifest")
    complete = producer._strict_read(root / "COMPLETE.json")
    complete["leaf_manifest"] = complete["leaf_manifest"][:-1]
    complete["leaf_manifest_digest"] = contract.stable_digest(complete["leaf_manifest"])
    complete["complete_digest"] = contract.stable_digest({k:v for k,v in complete.items()
                                                           if k != "complete_digest"})
    (root / "COMPLETE.json").write_bytes(contract.canonical_bytes(complete) + b"\n")
    with pytest.raises(producer.All60Error, match="manifest"):
        producer.validate_terminal(root, prereg,
            certified_validator=FakeBackend.validate_certified)
    other, pre, _ = _run(tmp_path / "directory")
    (other / "unexpected-empty").mkdir()
    with pytest.raises(producer.All60Error, match="directory"):
        producer.validate_terminal(other, pre,
            certified_validator=FakeBackend.validate_certified)


def test_fifo_and_symlinked_output_fail_without_blocking(tmp_path: Path):
    fifo = tmp_path / "fifo.json"; fifo.parent.mkdir(exist_ok=True); fifo_path = str(fifo)
    import os
    os.mkfifo(fifo_path)
    with pytest.raises(producer.All60Error): producer._strict_read(fifo)
    real = tmp_path / "real"; real.mkdir(); link = tmp_path / "linked"; link.symlink_to(real)
    root = link / "candidate"; prereg = _prereg(root)
    with pytest.raises(producer.All60Error, match="symlink"):
        producer.execute(root, prereg, FakeBackend(), clock=lambda: FIXED_TIME)


def test_preregister_and_run_cli_wiring_is_canonical_and_truth_blind(
    tmp_path: Path, monkeypatch,
):
    repository = tmp_path / "repo"; repository.mkdir()
    resident = tmp_path / "resident"; resident.mkdir()
    fake_m13 = SimpleNamespace(CONFIG_RELATIVE=Path("config/datasets.example.yaml"),
        REGISTRY_RELATIVE=Path("registry"), SOURCE_FULL_RELATIVE=Path("source"),
        RESIDENT_ROOT=resident)
    monkeypatch.setattr(producer, "_module", lambda *args, **kwargs: fake_m13)
    monkeypatch.setattr(producer, "production_runtime_binding",
                        lambda repo: _runtime_binding())
    assert producer.main(["preregister", "--repository", str(repository)]) == 0
    prereg_path = repository / contract.PREREGISTRATION_RELATIVE
    prereg = producer.validate_preregistration(producer._strict_read(prereg_path))
    assert prereg["roots"]["candidate"] == str(
        (repository / contract.CANDIDATE_RELATIVE).resolve())
    calls = []
    sentinel = object()
    monkeypatch.setattr(producer, "validate_committed_launch",
        lambda repo, path, value: calls.append(("h1", repo, path)) or value)
    monkeypatch.setattr(producer, "production_backend",
        lambda repo, value: calls.append(("backend", repo)) or sentinel)
    monkeypatch.setattr(producer, "execute",
        lambda root, value, backend: calls.append(("execute", root, backend)) or {})
    assert producer.main(["run", "--repository", str(repository)]) == 0
    assert [row[0] for row in calls] == ["h1", "backend", "execute"]
    assert calls[-1][2] is sentinel


def test_production_registry_preflight_uses_all_60_in_canonical_registry_order() -> None:
    """Regression for the failed launch: M13's helper selects only four cases."""
    m13 = producer._module(
        ROOT / "experiments/m04r/m04r13_threaded_certified_exposed.py",
        "m04r14_all60_real_registry_test",
    )
    digest, cases = producer._ordered_all60_registry_cases(
        m13, ROOT, ROOT / m13.REGISTRY_RELATIVE,
    )
    assert digest == "0a4da732f91375a091775cb04e6e77c8d136ade47d7f4d16508a2d9a6555361e"
    assert len(cases) == 60
    assert tuple(case.ordinal for case in cases) == tuple(range(60))
    assert tuple(case.query_id for case in cases) == contract.QUERY_IDS
    assert tuple(case.query_id for case in m13._registry_cases(ROOT, ROOT / m13.REGISTRY_RELATIVE)[1]) \
        != contract.QUERY_IDS


def test_global_deadline_is_partitioned_across_startup_and_task(
    tmp_path: Path, monkeypatch,
):
    class TimedBackend(FakeBackend):
        def __init__(self):
            super().__init__(); self.startup_timeout = 10_000.0
            self.task_timeout = 10_000.0; self.observed = []
        def prepare(self, ordinal, task_id):
            self.observed.append(("proposal", self.startup_timeout, self.task_timeout))
            return super().prepare(ordinal, task_id)
        def exact(self, prepared, workers=1):
            self.observed.append(("exact", self.startup_timeout, self.task_timeout))
            return super().exact(prepared, workers)
    tick = iter(float(value) for value in range(10_000)).__next__
    monkeypatch.setattr(producer, "perf_counter", tick)
    backend = TimedBackend(); _run(tmp_path, backend)
    proposal, exact = backend.observed[:2]
    assert 0 < proposal[1] == proposal[2] <= 3600
    assert 0 < exact[1] == exact[2] < proposal[1]
    assert all(next_row[1] <= prior[1]
               for prior, next_row in zip(backend.observed, backend.observed[1:]))


def test_complete_is_last_and_failed_publication_terminalizes_prefix(
    tmp_path: Path, monkeypatch,
):
    root = tmp_path / "complete-failure"; prereg = _prereg(root)
    real_atomic = producer._atomic
    def injected(path, value):
        if path.name == "COMPLETE.json": raise OSError("injected COMPLETE failure")
        return real_atomic(path, value)
    monkeypatch.setattr(producer, "_atomic", injected)
    with pytest.raises(OSError, match="COMPLETE"):
        producer.execute(root, prereg, FakeBackend(), clock=lambda: FIXED_TIME)
    assert not (root / "COMPLETE.json").exists()
    assert producer._tree(root) == contract.incomplete_tree(60, aggregate_count=2)
    terminal = producer._strict_read(root / "INCOMPLETE.json")
    assert terminal["completed_cases"] == 60 and terminal["resume_authorized"] is False


def test_linux_hard_deadline_interrupts_final_observer_once_and_restores_signal(
    tmp_path: Path, monkeypatch,
):
    if sys.platform != "linux": pytest.skip("Linux ITIMER_REAL contract")
    class SleepingFinal(FakeBackend):
        entered_final = False
        def final_source_lease(self):
            self.entered_final = True
            time.sleep(5)
            return super().final_source_lease()
    root = tmp_path / "hard-deadline"; prereg = _prereg(root); backend = SleepingFinal()
    monkeypatch.setattr(producer, "_run_hard_limit_seconds", lambda: 1.0)
    original_handler = signal.getsignal(signal.SIGALRM)
    original_timer = signal.getitimer(signal.ITIMER_REAL)
    marker = []
    def prior_handler(_signum, _frame): marker.append("unexpected")
    signal.signal(signal.SIGALRM, prior_handler)
    signal.setitimer(signal.ITIMER_REAL, 30.0)
    started = time.monotonic()
    try:
        with pytest.raises(producer._RunHardDeadline, match="hard limit"):
            producer.execute(root, prereg, backend, clock=lambda: FIXED_TIME)
        elapsed = time.monotonic() - started
        assert backend.entered_final is True and elapsed < 2.5
        assert signal.getsignal(signal.SIGALRM) is prior_handler
        restored, interval = signal.getitimer(signal.ITIMER_REAL)
        assert 29.0 <= restored <= 30.0 and interval == 0.0 and marker == []
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, original_handler)
        signal.setitimer(signal.ITIMER_REAL, *original_timer)
    assert not (root / "COMPLETE.json").exists()
    assert producer._strict_read(root / "INCOMPLETE.json")["completed_cases"] == 60


def test_complete_atomic_is_masked_terminal_success_after_post_link_delay(
    tmp_path: Path, monkeypatch,
):
    if sys.platform != "linux": pytest.skip("Linux ITIMER_REAL contract")
    root = tmp_path / "complete-delay"; prereg = _prereg(root); backend = FakeBackend()
    monkeypatch.setattr(producer, "_run_hard_limit_seconds", lambda: 1.0)
    real_atomic = producer._atomic
    def delayed_after_link(path, value):
        real_atomic(path, value)
        if path.name == "COMPLETE.json": time.sleep(1.25)
    monkeypatch.setattr(producer, "_atomic", delayed_after_link)
    result = producer.execute(root, prereg, backend, clock=lambda: FIXED_TIME)
    assert result["status"] == "complete"
    assert (root / "COMPLETE.json").is_file()
    assert not (root / "INCOMPLETE.json").exists()
    assert producer.validate_terminal(root, prereg,
        certified_validator=backend.validate_certified) == result
