from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "m04r14_serial_runtime_test",
    ROOT / "experiments/m04r/m04r14_serial_certified_runtime.py",
)
assert SPEC is not None and SPEC.loader is not None
runtime = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = runtime
SPEC.loader.exec_module(runtime)


HELPER = r'''
import hashlib,json,os,sys,time
from pathlib import Path
mode,behavior,request_path,output_path=sys.argv[1:]
request=json.loads(Path(request_path).read_text())
Path(request_path+'.pid').write_text(str(os.getpid()))
if behavior=='early': raise SystemExit(7)
schema=('m04r14-serial-proposal-child-ready-v1' if mode=='proposal'
        else 'm04r14-serial-exact-child-ready-v1')
workers=8 if mode=='proposal' else 1
ready={'schema_version':schema,'task_id':request['task_id'],'pid':os.getpid(),
 'workers':workers,'cpu_affinity':sorted(os.sched_getaffinity(0)),
 'thread_environment':{k:(os.environ.get(k) if k=='NUMBA_THREADING_LAYER' else '1')
  for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMBA_NUM_THREADS','NUMBA_THREADING_LAYER')},
 'resident_lease_digest':request['resident_lease_digest'],'ready_monotonic':time.monotonic()}
if mode=='exact': ready['proposal_sha256']=request['proposal_sha256']
if behavior=='wrong-worker': ready['workers']=2
if behavior=='bool-worker': ready['workers']=True
if behavior=='wrong-affinity': ready['cpu_affinity']=[999]
if behavior=='wrong-schema': ready['schema_version']='wrong'
if behavior=='fifo-ready':
 os.mkfifo(request['ready_path']); time.sleep(30)
Path(request['ready_path']).write_text(json.dumps(ready))
if behavior=='hang': time.sleep(30)
if mode=='proposal':
 while not Path(request['release_path']).exists(): time.sleep(.005)
 case=request['case_table'][request['ordinal']]
 state={'task_id':request['task_id'],'case_id':case['case_id'],'query_id':case['query_id'],
        'query_binding':{'test':True},'forward':{},'reverse':{}}
 payload={'semantic':{'state':state,'digest':hashlib.sha256(json.dumps(state,sort_keys=True,
  separators=(',',':')).encode()).hexdigest()},'measurement':{'wall_seconds':.01},
  'created_at':'2026-01-01T00:00:00+00:00'}
else:
 state={'case_id':request['case_id'],'query_id':request['query_id'],'workers':1,
        'certificate':{'result_digest':'x'},'matches':[]}
 payload={'semantic':{'state':state,'digest':hashlib.sha256(json.dumps(state,sort_keys=True,
  separators=(',',':')).encode()).hexdigest()},'measurement':{'wall_seconds':.01},
  'created_at':'2026-01-01T00:00:00+00:00'}
Path(output_path).write_text(json.dumps(payload))
'''


def _helper(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "child.py"; path.write_text(HELPER); return path


def _cases(count: int = 60):
    return tuple(runtime.RuntimeCase(index, f"case-{index}", f"query-{index}",
        {"case_id": f"case-{index}", "episode_id": f"query-{index}"})
        for index in range(count))


def _instance(tmp_path: Path, *, proposal: str = "ok", exact: str = "ok",
              startup: float = 2, task: float = 2):
    helper = _helper(tmp_path)
    command = lambda mode, behavior: lambda request, output: [
        sys.executable, str(helper), mode, behavior, str(request), str(output)]
    return runtime.SerialSpawnedRuntime(repository=ROOT, cases=_cases(),
        resident_lease_digest="lease", scratch_root=tmp_path / "scratch",
        proposal_command=command("proposal", proposal),
        exact_command=command("exact", exact), startup_timeout=startup,
        task_timeout=task)


def _persist(instance, prepared, path: Path) -> None:
    state = prepared.semantic
    runtime._atomic(path, {"state": state, "digest": runtime.stable_hash(state),
        "measurement": prepared.measurement, "created_at": "2026-01-01T00:00:00+00:00"})
    instance.bind_proposal(prepared, path)


def test_serial_success_supports_arbitrary_ordinal_59(tmp_path: Path) -> None:
    instance = _instance(tmp_path)
    prepared = instance.prepare(59, "task-59")
    assert prepared.case.ordinal == 59
    assert prepared.measurement["spawned_process"]["workers"] == 8
    assert len(prepared.measurement["spawned_process"]["cpus"]) == 8
    proposal = tmp_path / "published" / "PROPOSAL.json"
    _persist(instance, prepared, proposal)
    attempt = instance.exact(prepared)
    assert attempt.semantic["query_id"] == "query-59"
    assert attempt.measurement["spawned_process"]["workers"] == 1
    assert len(attempt.measurement["spawned_process"]["cpus"]) == 1


def test_spawned_child_rss_excludes_large_parent_preexec_high_water(tmp_path: Path) -> None:
    retained_parent_evidence = bytearray(160 * 1024 * 1024)
    retained_parent_evidence[0] = 1
    instance = _instance(tmp_path)
    prepared = instance.prepare(0, "rss-boundary")
    process = prepared.measurement["spawned_process"]
    assert process["wait4_max_rss_kib_context_only"] is True
    assert process["effective_peak_rss_kib"] < 128 * 1024
    assert retained_parent_evidence[0] == 1


def test_case_table_rejects_holes_duplicates_and_wrong_worker(tmp_path: Path) -> None:
    values = list(_cases(2)); values[1] = runtime.RuntimeCase(3, "case-1", "query-1", {})
    with pytest.raises(runtime.RuntimeErrorEvidence, match="case table"):
        runtime.SerialSpawnedRuntime(repository=ROOT, cases=values,
            resident_lease_digest="lease", scratch_root=tmp_path)
    instance = _instance(tmp_path / "valid")
    with pytest.raises(runtime.RuntimeErrorEvidence, match="workers=1"):
        instance.exact(runtime.PreparedCase("x", instance.cases[0], {}, {}), 2)


@pytest.mark.parametrize("behavior", ["wrong-worker", "bool-worker", "wrong-affinity", "wrong-schema"])
def test_ready_binding_fails_closed(tmp_path: Path, behavior: str) -> None:
    instance = _instance(tmp_path, proposal=behavior)
    with pytest.raises(runtime.RuntimeErrorEvidence, match="READY binding"):
        instance.prepare(0, "task")


@pytest.mark.parametrize("behavior", ["wrong-worker", "bool-worker", "wrong-affinity", "wrong-schema"])
def test_exact_ready_binding_fails_closed(tmp_path: Path, behavior: str) -> None:
    instance = _instance(tmp_path, exact=behavior)
    prepared = instance.prepare(0, "task")
    _persist(instance, prepared, tmp_path / "PROPOSAL.json")
    with pytest.raises(runtime.RuntimeErrorEvidence, match="READY binding"):
        instance.exact(prepared)


def test_early_exit_and_hang_are_killed_and_reaped(tmp_path: Path) -> None:
    early = _instance(tmp_path / "early", proposal="early", startup=.5, task=.5)
    with pytest.raises(runtime.RuntimeErrorEvidence, match="exited before READY"):
        early.prepare(0, "early")
    hanging = _instance(tmp_path / "hang", proposal="hang", startup=.5, task=.1)
    with pytest.raises(runtime.RuntimeErrorEvidence, match="task timed out"):
        hanging.prepare(0, "hang")
    pid_file = next((tmp_path / "hang" / "scratch").glob("*.pid"))
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_exact_child_early_exit_is_reaped(tmp_path: Path) -> None:
    instance = _instance(tmp_path, exact="early", startup=.5, task=.5)
    prepared = instance.prepare(0, "task")
    _persist(instance, prepared, tmp_path / "PROPOSAL.json")
    with pytest.raises(runtime.RuntimeErrorEvidence, match="exited before READY"):
        instance.exact(prepared)


def test_proposal_sha_drift_is_rejected_before_exact_spawn(tmp_path: Path) -> None:
    instance = _instance(tmp_path); prepared = instance.prepare(0, "task")
    proposal = tmp_path / "PROPOSAL.json"; _persist(instance, prepared, proposal)
    proposal.write_text("{}\n")
    with pytest.raises(runtime.RuntimeErrorEvidence, match="SHA changed"):
        instance.exact(prepared)


def test_proposal_a_b_a_identity_drift_is_rejected(tmp_path: Path) -> None:
    instance = _instance(tmp_path); prepared = instance.prepare(0, "task")
    proposal = tmp_path / "PROPOSAL.json"; _persist(instance, prepared, proposal)
    original = proposal.read_bytes(); observed = proposal.stat()
    proposal.write_bytes(b"{}\n"); proposal.write_bytes(original)
    os.utime(proposal, ns=(observed.st_atime_ns, observed.st_mtime_ns + 1_000_000))
    with pytest.raises(runtime.RuntimeErrorEvidence, match="SHA changed"):
        instance.exact(prepared)


def test_nonregular_ready_fails_immediately_and_reaps(tmp_path: Path) -> None:
    instance = _instance(tmp_path, proposal="fifo-ready", startup=10, task=10)
    started = time.monotonic()
    with pytest.raises(runtime.RuntimeErrorEvidence, match="READY is not a regular"):
        instance.prepare(0, "task")
    assert time.monotonic() - started < 2
    pid = int(next((tmp_path / "scratch").glob("*.pid")).read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_scratch_and_proposal_symlink_ancestry_are_rejected(tmp_path: Path) -> None:
    actual = tmp_path / "actual"; actual.mkdir(); alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    with pytest.raises(runtime.RuntimeErrorEvidence, match="symlink"):
        runtime.SerialSpawnedRuntime(repository=ROOT, cases=_cases(),
            resident_lease_digest="lease", scratch_root=alias)
    instance = _instance(tmp_path / "valid"); prepared = instance.prepare(0, "task")
    target = tmp_path / "target.json"; _persist(instance, prepared, target)
    # A second task proves an aliased proposal is rejected before it is recorded.
    second = instance.prepare(1, "task-1"); link = tmp_path / "link.json"; link.symlink_to(target)
    with pytest.raises(runtime.RuntimeErrorEvidence, match="symlink"):
        instance.bind_proposal(second, link)


def test_case_table_rejects_authority_or_outcome_fields_and_bad_timeouts(tmp_path: Path) -> None:
    forged = list(_cases(1)); row = dict(forged[0].registry_case); row["authority_path"] = "/forbidden"
    forged[0] = runtime.RuntimeCase(0, "case-0", "query-0", row)
    with pytest.raises(runtime.RuntimeErrorEvidence, match="case table"):
        runtime.SerialSpawnedRuntime(repository=ROOT, cases=forged,
            resident_lease_digest="lease", scratch_root=tmp_path / "scratch")
    for nested in ({"metadata": {"outcome_path": "/truth/results.json"}},
                   {"metadata": {"input_path": "/sealed/authorities/case.json"}}):
        row = dict(_cases(1)[0].registry_case); row.update(nested)
        with pytest.raises(runtime.RuntimeErrorEvidence, match="case table"):
            runtime.SerialSpawnedRuntime(repository=ROOT,
                cases=(runtime.RuntimeCase(0, "case-0", "query-0", row),),
                resident_lease_digest="lease", scratch_root=tmp_path / "nested")
    for value in (True, 0, -1, float("inf")):
        with pytest.raises(runtime.RuntimeErrorEvidence, match="timeout"):
            runtime.SerialSpawnedRuntime(repository=ROOT, cases=_cases(1),
                resident_lease_digest="lease", scratch_root=tmp_path / f"s-{value}",
                startup_timeout=value)


def test_case_table_is_finite_and_snapshotted_against_caller_mutation(tmp_path: Path) -> None:
    row = {"case_id": "case-0", "episode_id": "query-0", "metadata": {"rank": 1}}
    supplied = (runtime.RuntimeCase(0, "case-0", "query-0", row),)
    instance = runtime.SerialSpawnedRuntime(repository=ROOT, cases=supplied,
        resident_lease_digest="lease", scratch_root=tmp_path / "valid")
    row["metadata"]["rank"] = 99
    assert instance.case_table[0]["registry_case"]["metadata"]["rank"] == 1
    invalid = {"case_id": "case-0", "episode_id": "query-0", "rank": float("nan")}
    with pytest.raises(runtime.RuntimeErrorEvidence, match="finite JSON"):
        runtime.SerialSpawnedRuntime(repository=ROOT,
            cases=(runtime.RuntimeCase(0, "case-0", "query-0", invalid),),
            resident_lease_digest="lease", scratch_root=tmp_path / "invalid")


def test_api_and_cli_accept_no_authority_or_outcome_paths() -> None:
    source = (ROOT / "experiments/m04r/m04r14_serial_certified_runtime.py").read_text()
    assert "--authority" not in source and "--outcome" not in source
    result = subprocess.run([sys.executable,
        str(ROOT / "experiments/m04r/m04r14_serial_certified_runtime.py"), "describe"],
        cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)["authority_paths_accepted"] is False


def test_child_context_never_runs_full_resident_validation() -> None:
    source = (ROOT / "experiments/m04r/m04r14_serial_certified_runtime.py").read_text()
    child = source[source.index("def _child_context"):source.index("def _proposal_child")]
    assert "resident_full" not in child
    assert "validate_resident_snapshot" in child
    assert "lease_exact" in child


def test_strict_readers_reject_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "evidence.json"
    os.mkfifo(fifo)
    results = []

    def exercise(reader):
        try:
            reader(fifo)
        except BaseException as exc:
            results.append(exc)

    for reader in (runtime._read, runtime._sha):
        thread = threading.Thread(target=exercise, args=(reader,))
        thread.start(); thread.join(timeout=1)
        assert not thread.is_alive()
    assert len(results) == 2
    assert all(isinstance(exc, runtime.RuntimeErrorEvidence) for exc in results)


@pytest.mark.parametrize("value", [1, 0.0, None])
def test_production_final_swap_binding_is_exact(value) -> None:
    with pytest.raises(runtime.RuntimeErrorEvidence, match="swap"):
        runtime.ProductionSerialRuntime._bind_final_swap(
            {"resources": {"after": {"swap_kib": value}}})
    runtime.ProductionSerialRuntime._bind_final_swap(
        {"resources": {"after": {"swap_kib": 0}}})
