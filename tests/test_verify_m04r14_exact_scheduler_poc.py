from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module


verifier = _module(
    "verify_m04r14_exact_scheduler_poc_test",
    ROOT / "experiments/m04r/verify_m04r14_exact_scheduler_poc.py",
)
runner_test = _module(
    "m04r14_runner_fixture_for_verifier",
    ROOT / "tests/test_m04r14_exact_scheduler_poc.py",
)
runner = runner_test.scheduler


@pytest.fixture()
def candidate(tmp_path: Path) -> Path:
    root = tmp_path / "candidate"
    runner.execute(root, runner_test.FakeBackend())
    return root


def _write(path: Path, value: object) -> None:
    path.write_bytes(verifier._strict_bytes(value) + b"\n")


def _rebind_complete(root: Path, changed: Path) -> None:
    complete_path = root / "COMPLETE.json"
    complete = json.loads(complete_path.read_text())
    relative = str(changed.relative_to(root))
    for row in complete["state"]["leaf_manifest"]:
        if row["path"] == relative:
            row["sha256"] = verifier._sha(changed)
    complete["state"]["leaf_manifest_digest"] = verifier.stable_hash(
        complete["state"]["leaf_manifest"]
    )
    complete["digest"] = verifier.stable_hash(complete["state"])
    _write(complete_path, complete)


def _rebind_aggregate(root: Path, name: str, value: dict[str, object]) -> None:
    path = root / name
    value["digest"] = verifier.stable_hash(value["state"])
    _write(path, value)
    complete = json.loads((root / "COMPLETE.json").read_text())
    field = "semantic_digest" if name == "SEMANTICS.json" else "measurement_digest"
    complete["state"][field] = value["digest"]
    for row in complete["state"]["leaf_manifest"]:
        if row["path"] == name:
            row["sha256"] = verifier._sha(path)
    complete["state"]["leaf_manifest_digest"] = verifier.stable_hash(
        complete["state"]["leaf_manifest"]
    )
    complete["digest"] = verifier.stable_hash(complete["state"])
    _write(root / "COMPLETE.json", complete)


def test_independent_verifier_accepts_fixture_and_publishes_create_only(
    candidate: Path, tmp_path: Path,
) -> None:
    output = tmp_path / "verification"
    receipt = verifier.publish_verification(
        candidate, output, repository=ROOT, require_production=False,
    )
    assert receipt["state"]["status"] == "verified"
    assert receipt["state"]["direct_raw_authority_accessed_by_verifier"] is False
    assert receipt["state"][
        "authority_derived_prerequisite_evidence_accessed_by_verifier"
    ] is True
    assert receipt["state"]["passed"] is True
    assert receipt["state"]["primary_rows"] == 48
    assert receipt["state"]["throughput_rows"] == 8
    assert (output / "VERIFIED.json").is_file()
    with pytest.raises(verifier.VerificationError, match="must be absent"):
        verifier.publish_verification(
            candidate, output, repository=ROOT, require_production=False,
        )


def test_fresh_root_replay_is_deterministically_equal(
    candidate: Path, tmp_path: Path,
) -> None:
    first = verifier.publish_verification(
        candidate, tmp_path / "v1", repository=ROOT, require_production=False,
    )
    second = verifier.publish_verification(
        candidate, tmp_path / "v2", repository=ROOT, require_production=False,
    )
    assert first["state"] == second["state"]
    assert first["digest"] == second["digest"]
    assert first["state"]["result_digest"] == second["state"]["result_digest"]


def test_verifier_source_has_no_runner_import() -> None:
    source = (ROOT / "experiments/m04r/verify_m04r14_exact_scheduler_poc.py").read_text()
    assert "import m04r14_exact_scheduler_poc" not in source
    assert "from m04r14_exact_scheduler_poc" not in source


def test_failure_occurs_before_verification_publication(
    candidate: Path, tmp_path: Path,
) -> None:
    (candidate / "EXTRA.json").write_text("{}")
    output = tmp_path / "never-created"
    with pytest.raises(verifier.VerificationError, match="tree"):
        verifier.publish_verification(
            candidate, output, repository=ROOT, require_production=False,
        )
    assert not output.exists()


def test_rejects_coordinated_aggregate_extra_claim(candidate: Path) -> None:
    semantics = json.loads((candidate / "SEMANTICS.json").read_text())
    semantics["state"]["forged_claim"] = True
    _rebind_aggregate(candidate, "SEMANTICS.json", semantics)
    with pytest.raises(verifier.VerificationError, match="semantic aggregate.*keys"):
        verifier.verify_terminal(candidate, repository=ROOT, require_production=False)


def test_rejects_coordinated_proposal_and_attempt_rehash(candidate: Path) -> None:
    proposal_path = candidate / "primary/r0/c0/PROPOSAL.json"
    proposal = json.loads(proposal_path.read_text())
    proposal["state"]["semantic_digest"] = "f" * 64
    proposal["digest"] = verifier.stable_hash(proposal["state"])
    _write(proposal_path, proposal); _rebind_complete(candidate, proposal_path)
    with pytest.raises(verifier.VerificationError, match="proposal parity"):
        verifier.verify_terminal(candidate, repository=ROOT, require_production=False)


def test_rejects_rehashed_match_mutation(candidate: Path) -> None:
    path = candidate / "primary/r0/c0/EXACT-w1.json"
    leaf = json.loads(path.read_text())
    attempt = leaf["semantic"]["state"]["attempt"]
    attempt["matches"][0]["total_distance_hex"] = float(1.1).hex()
    attempt["match_digest"] = verifier.stable_hash(attempt["matches"])
    leaf["semantic"]["digest"] = verifier.stable_hash(leaf["semantic"]["state"])
    _write(path, leaf); _rebind_complete(candidate, path)
    with pytest.raises(verifier.VerificationError, match="semantic parity"):
        verifier.verify_terminal(candidate, repository=ROOT, require_production=False)


def test_rejects_rehashed_selection_and_resource_mutations(candidate: Path) -> None:
    original = (candidate / "MEASUREMENTS.json").read_bytes()
    measurements = json.loads(original)
    measurements["state"]["selection"]["state"]["selected_workers"] = 8
    measurements["state"]["selection"]["digest"] = verifier.stable_hash(
        measurements["state"]["selection"]["state"]
    )
    _rebind_aggregate(candidate, "MEASUREMENTS.json", measurements)
    with pytest.raises(verifier.VerificationError, match="measurement aggregate"):
        verifier.verify_terminal(candidate, repository=ROOT, require_production=False)
    # Rebuild the fixture because COMPLETE was coherently rebound above.
    candidate = candidate.parent / "candidate-resource"
    runner.execute(candidate, runner_test.FakeBackend())
    measurements = json.loads((candidate / "MEASUREMENTS.json").read_text())
    measurements["state"]["throughput_lane"]["resources"]["forged"] = True
    _rebind_aggregate(candidate, "MEASUREMENTS.json", measurements)
    with pytest.raises(verifier.VerificationError, match="resource.*keys"):
        verifier.verify_terminal(candidate, repository=ROOT, require_production=False)


def test_rejects_rehashed_barrier_and_event_forgery(candidate: Path) -> None:
    barrier_path = candidate / "throughput/PROPOSALS_COMPLETE.json"
    barrier = json.loads(barrier_path.read_text())
    barrier["state"]["status"] = "forged"
    barrier["digest"] = verifier.stable_hash(barrier["state"])
    _write(barrier_path, barrier); _rebind_complete(candidate, barrier_path)
    with pytest.raises(verifier.VerificationError, match="barrier"):
        verifier.verify_terminal(candidate, repository=ROOT, require_production=False)
    candidate = candidate.parent / "candidate-ledger"
    runner.execute(candidate, runner_test.FakeBackend())
    measurements = json.loads((candidate / "MEASUREMENTS.json").read_text())
    ledger = measurements["state"]["event_ledger"]; ledger[0]["event"] = "forged"
    previous = None
    for event in ledger:
        event["previous_event_digest"] = previous
        event["event_digest"] = verifier.stable_hash(
            verifier._without(event, {"event_digest"})
        ); previous = event["event_digest"]
    measurements["state"]["event_ledger_digest"] = verifier.stable_hash(ledger)
    _rebind_aggregate(candidate, "MEASUREMENTS.json", measurements)
    with pytest.raises(verifier.VerificationError, match="proposal event"):
        verifier.verify_terminal(candidate, repository=ROOT, require_production=False)


def test_rejects_rehashed_final_lease_and_duplicate_manifest(candidate: Path) -> None:
    complete_path = candidate / "COMPLETE.json"; original = complete_path.read_bytes()
    complete = json.loads(original); complete["state"]["final_lease"] = {"forged": True}
    complete["digest"] = verifier.stable_hash(complete["state"]); _write(complete_path, complete)
    with pytest.raises(verifier.VerificationError, match="final lease"):
        verifier.verify_terminal(candidate, repository=ROOT, require_production=False)
    complete_path.write_bytes(original); complete = json.loads(original)
    complete["state"]["leaf_manifest"].append(copy.deepcopy(complete["state"]["leaf_manifest"][0]))
    complete["state"]["leaf_manifest_digest"] = verifier.stable_hash(complete["state"]["leaf_manifest"])
    complete["digest"] = verifier.stable_hash(complete["state"]); _write(complete_path, complete)
    with pytest.raises(verifier.VerificationError, match="complete reconstruction"):
        verifier.verify_terminal(candidate, repository=ROOT, require_production=False)


def test_strict_reader_rejects_duplicate_nonfinite_and_symlink(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"; duplicate.write_text('{"a":1,"a":2}')
    with pytest.raises(verifier.VerificationError, match="duplicate"):
        verifier._read(duplicate)
    nonfinite = tmp_path / "nonfinite.json"; nonfinite.write_text('{"a":1e999}')
    with pytest.raises(verifier.VerificationError, match="nonfinite"):
        verifier._read(nonfinite)
    alias = tmp_path / "alias.json"; alias.symlink_to(duplicate)
    with pytest.raises(verifier.VerificationError, match="symlink"):
        verifier._read(alias)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (("cpuset", "capacity"), ("quota", "capacity"), ("throttle", "throttling")),
)
def test_host_cpu_qualification_fails_closed_after_coordinated_rehash(
    candidate: Path, mutation: str, message: str,
) -> None:
    measurements = json.loads((candidate / "MEASUREMENTS.json").read_text())
    cpu = measurements["state"]["host_cpu_qualification"]
    if mutation == "cpuset":
        cpus = cpu["before"]["configuration"]["effective_cpus"][:7]
        cpu["before"]["configuration"]["effective_cpus"] = cpus
        cpu["after"]["configuration"]["effective_cpus"] = list(cpus)
    elif mutation == "quota":
        for side in ("before", "after"):
            configuration = cpu[side]["configuration"]
            configuration["effective_quota_cpus"] = 7.0
            configuration["quota_usec"] = configuration["period_usec"] * 7
    else:
        cpu["after"]["cpu_stat"]["nr_throttled"] += 1
        cpu["after"]["cpu_stat"]["throttled_usec"] += 10
        cpu["delta"] = {
            key: cpu["after"]["cpu_stat"][key] - cpu["before"]["cpu_stat"][key]
            for key in cpu["before"]["cpu_stat"]
        }
    _rebind_aggregate(candidate, "MEASUREMENTS.json", measurements)
    with pytest.raises(verifier.VerificationError, match=message):
        verifier.verify_terminal(candidate, repository=ROOT, require_production=False)


def test_proposal_ready_release_and_vmstat_crosslinks_are_independent(
    tmp_path: Path,
) -> None:
    backend = runner_test.LightProposalBackend(tmp_path)
    prepared = backend.prepare(0, "proposal-light-0")
    process = prepared.measurement["spawned_process"]
    verifier._proposal_process(process, prepared.task_id)
    runtime = {"state": {"environment": {"state": {"thread_environment": {
        key: None for key in verifier.THREAD_ENV_KEYS
    }}}}}
    # Numeric child variables are frozen to one; only threading layer inherits.
    verifier._proposal_process_binding(process, prepared.semantic, runtime)
    for field, value, message in (
        ("task_id", "wrong", "causal binding"),
        ("resident_lease_digest", "f" * 64, "causal binding"),
        ("ready_monotonic", process["release_evidence"]["released_monotonic"] + 1,
         "reconstruction"),
    ):
        forged = copy.deepcopy(process); forged["ready_evidence"][field] = value
        forged["release_evidence"]["ready_digest"] = verifier.stable_hash(
            forged["ready_evidence"]
        )
        validator = verifier._proposal_process_binding if field != "ready_monotonic" \
            else verifier._proposal_process
        with pytest.raises(verifier.VerificationError, match=message):
            if validator is verifier._proposal_process_binding:
                validator(forged, prepared.semantic, runtime)
            else:
                validator(forged, prepared.task_id)
    forged = copy.deepcopy(process)
    forged["host_vmstat_swap_after"]["pswpin"] = max(
        0, forged["host_vmstat_swap_before"]["pswpin"] - 1,
    )
    if forged["host_vmstat_swap_before"]["pswpin"] == 0:
        forged["host_vmstat_swap_before"]["pswpin"] = 1
    with pytest.raises(verifier.VerificationError, match="vmstat regressed"):
        verifier._proposal_process(forged, prepared.task_id)


def test_production_query_resident_and_source_causal_binding_rejects_mutation() -> None:
    request = {"search_datasets": ["nasdaq"], "quality_tiers": ["A", "B"],
        "top_k": 20, "cross_dataset": False, "deduplicate_overlaps": True,
        "max_per_instrument": 3, "minimum_history_gap_bars": 60}
    deterministic = {"query_stock_prefix": {"digest": "a" * 64},
        "query_benchmark_prefix": {"digest": "b" * 64}, "request": request,
        "packed_provenance_digest": "c" * 64,
        "query_representation_digest": "d" * 64}
    binding = {**deterministic, "packed_query_input_digest": "e" * 64,
               "certified_input_digest": verifier.stable_hash(deterministic)}
    resident = {"lease": {"lease_digest": "f" * 64}, "identity_digest": "1" * 64}
    proposal = {"query_binding": binding, "source_binding_before": binding,
        "source_binding_after": binding, "resident_snapshot": resident,
        "resident_lease_digests": ["f" * 64] * 4,
        "forward": {"input_digest": "e" * 64}, "reverse": {"input_digest": "e" * 64}}
    foundation = {"resident": resident, "provenance_digest": "c" * 64}
    expected = {"query": binding}; proposal["query_id"] = "query"
    verifier._production_proposal_binding(proposal, foundation, expected)
    for mutate in ("query", "resident", "source"):
        forged = copy.deepcopy(proposal)
        if mutate == "query": forged["query_binding"]["certified_input_digest"] = "0" * 64
        elif mutate == "resident": forged["resident_snapshot"]["identity_digest"] = "0" * 64
        else: forged["source_binding_after"] = {"forged": True}
        with pytest.raises(verifier.VerificationError, match="causal binding"):
            verifier._production_proposal_binding(forged, foundation, expected)


def test_production_certificate_uses_timing_free_semantic_schema() -> None:
    contract = verifier.certified_packed_search_contract(
        requested_positions=True, vector_lower_bounds=True,
        deferred_alignments=True, compact_scored=True,
        native_bound_deferral=True, streaming_threshold_closure=True,
        branch_aware_packed_bounds=True,
    )
    matches = [{"episode_id": f"{index:024d}", "symbol": f"S{index}",
        "cutoff": "2020-01-01", "total_distance": 1.0,
        "component_distances": {name: 0.0 for name in verifier.COMPONENT_NAMES},
        "alignment": [[0, 0]], "quality_tier": "A"} for index in range(20)]
    certificate = {"schema_version": contract["schema_version"],
        "contract_digest": contract["digest"], "generation_id": verifier.GENERATION_ID,
        "query_episode_id": "query", "input_digest": "input",
        "eligible_candidates": 20, "exact_evaluated": 20, "safely_pruned": 0,
        "stopped_early": True, "stop_threshold": 1.0, "next_lower_bound": 1.1,
        "maximum_quantized_bound_excess": 0.0, "materialization_groups": 0,
        "sparse_symbols": 0, "batch_symbols": 0,
        "rounds": [{"frontier_rows": 20, "exact_rows": 20, "next_lower_bound": 1.1,
            "constrained_threshold": 1.0, "selected_rows": 20, "certified": True,
            "proposal_digest": "proposal"}],
        "native_bound_accounting": {"native_bound_evaluated": 20,
            "exact_dtw_evaluated": 20, "native_bound_pruned": 0,
            "packed_bound_pruned": 0},
        "minimum_native_pruned_bound": None, "threshold_closure_passes": []}
    deterministic = {"schema_version": contract["schema_version"],
        "contract_digest": contract["digest"], "generation_id": verifier.GENERATION_ID,
        "query_episode_id": "query", "input_digest": "input", "eligible_candidates": 20,
        "exact_evaluated": 20, "safely_pruned": 0, "stopped_early": True,
        "stop_threshold_hex": float(1.0).hex(), "next_lower_bound_hex": float(1.1).hex(),
        "maximum_quantized_bound_excess_hex": float(0.0).hex(),
        "rounds": certificate["rounds"], "matches": [{"episode_id": row["episode_id"],
            "total_hex": row["total_distance"].hex(), "components": {key: value.hex()
            for key, value in sorted(row["component_distances"].items())},
            "alignment": row["alignment"]} for row in matches],
        "real_forward_outcomes_accessed": False,
        "native_bound_accounting": certificate["native_bound_accounting"],
        "minimum_native_pruned_bound_hex": None, "threshold_closure_passes": []}
    certificate["result_digest"] = verifier.stable_hash(deterministic)
    verifier._certificate(certificate, matches, "query", "input", True)
    forged = copy.deepcopy(certificate); forged["elapsed_seconds"] = 0.1
    with pytest.raises(verifier.VerificationError, match="certificate exact keys"):
        verifier._certificate(forged, matches, "query", "input", True)


def test_producer_attempt_keeps_timing_outside_semantic_certificate(
    candidate: Path,
) -> None:
    leaf = json.loads((candidate / "primary/r0/c0/EXACT-w1.json").read_text())
    certificate = leaf["semantic"]["state"]["attempt"]["certificate"]
    measurement = leaf["measurement"]["measurement"]
    assert "elapsed_seconds" not in certificate
    assert type(measurement["engine_seconds"]) is float
    assert 0 <= measurement["engine_seconds"] <= measurement["wall_seconds"]
    verifier.verify_terminal(candidate, repository=ROOT, require_production=False)


def test_production_runtime_schema_rejects_empty_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"git_head": "0" * 40, "files": {},
        "environment": verifier._environment(),
        "contracts": {"schema_version": verifier.SCHEMA,
                      "execution_policy": verifier._execution_policy()}}
    monkeypatch.setattr(verifier, "_runtime_paths", lambda *_: ("mandatory.py",))
    with pytest.raises(verifier.VerificationError, match="runtime contract"):
        verifier._validate_runtime(
            {"state": state, "digest": verifier.stable_hash(state)}, ROOT, True,
        )


def test_manifest_snapshot_rejects_transient_a_b_a_identity_swap(
    candidate: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = candidate / "primary/r0/c0/PROPOSAL.json"
    original = target.read_bytes(); observed = target.stat()
    real_selection = verifier._selection

    def swap_then_restore(rows):
        target.write_bytes(b"{" + b" " * (len(original) - 2) + b"}")
        target.write_bytes(original)
        # Make the A->B->A identity transition deterministic even on a coarse FS.
        os.utime(target, ns=(observed.st_atime_ns, observed.st_mtime_ns + 1_000_000))
        return real_selection(rows)

    monkeypatch.setattr(verifier, "_selection", swap_then_restore)
    with pytest.raises(verifier.VerificationError, match="manifest recheck"):
        verifier.verify_terminal(candidate, repository=ROOT, require_production=False)


def test_rejects_coordinated_contract_selection_rule_forgery(candidate: Path) -> None:
    contract_path = candidate / "CONTRACT.json"
    contract = json.loads(contract_path.read_text())
    contract["state"]["selection_rule"] = "choose forged worker"
    contract["digest"] = verifier.stable_hash(contract["state"])
    _write(contract_path, contract); _rebind_complete(candidate, contract_path)
    run_path = candidate / "RUN_STARTED.json"; run = json.loads(run_path.read_text())
    run["state"]["preregistration_digest"] = contract["digest"]
    run["digest"] = verifier.stable_hash(run["state"])
    _write(run_path, run); _rebind_complete(candidate, run_path)
    with pytest.raises(verifier.VerificationError, match="contract reconstruction"):
        verifier.verify_terminal(candidate, repository=ROOT, require_production=False)


@pytest.mark.parametrize("field", ["throughput", "numeric_policy"])
def test_rejects_coordinated_run_policy_forgery(candidate: Path, field: str) -> None:
    path = candidate / "RUN_STARTED.json"; run = json.loads(path.read_text())
    run["state"][field]["forged"] = True
    run["digest"] = verifier.stable_hash(run["state"])
    _write(path, run); _rebind_complete(candidate, path)
    with pytest.raises(verifier.VerificationError, match="run reconstruction"):
        verifier.verify_terminal(candidate, repository=ROOT, require_production=False)


@pytest.mark.parametrize("mutation", ["lane", "interactive", "concurrency"])
def test_rejects_coordinated_lane_and_interactive_forgery(
    candidate: Path, mutation: str,
) -> None:
    measurements = json.loads((candidate / "MEASUREMENTS.json").read_text())
    state = measurements["state"]
    if mutation == "lane":
        state["throughput_lane"]["proposals_prepared_serially_with_threads"] = 7
    elif mutation == "interactive":
        state["primary_interactive_selected"][0]["interactive_seconds"] += 1.0
    else:
        state["throughput_lane"]["observed_maximum_active_exact_tasks"] = 7
    _rebind_aggregate(candidate, "MEASUREMENTS.json", measurements)
    with pytest.raises(verifier.VerificationError, match="measurement aggregate|throughput lane|concurrency"):
        verifier.verify_terminal(candidate, repository=ROOT, require_production=False)


def test_attempt_timing_requires_nonnegative_nested_and_child_crosslinks(
    candidate: Path,
) -> None:
    measurement = json.loads(
        (candidate / "primary/r0/c0/EXACT-w1.json").read_text()
    )["measurement"]["measurement"]
    forged = copy.deepcopy(measurement); forged["wall_seconds"] = -1.0
    with pytest.raises(verifier.VerificationError, match="attempt timing"):
        verifier._attempt_measurement(forged, production=False, workers=1)
    forged = copy.deepcopy(measurement); forged["engine_seconds"] = forged["wall_seconds"] + 1.0
    with pytest.raises(verifier.VerificationError, match="attempt timing"):
        verifier._attempt_measurement(forged, production=False, workers=1)


def test_test_foundation_exact_schema_rejects_coordinated_extra(candidate: Path) -> None:
    contract_path = candidate / "CONTRACT.json"
    contract = json.loads(contract_path.read_text())
    contract["state"]["foundation"]["forged"] = True
    contract["digest"] = verifier.stable_hash(contract["state"])
    _write(contract_path, contract); _rebind_complete(candidate, contract_path)
    run_path = candidate / "RUN_STARTED.json"; run = json.loads(run_path.read_text())
    run["state"]["foundation"] = contract["state"]["foundation"]
    run["state"]["preregistration_digest"] = contract["digest"]
    run["digest"] = verifier.stable_hash(run["state"])
    _write(run_path, run); _rebind_complete(candidate, run_path)
    with pytest.raises(verifier.VerificationError, match="test foundation.*keys"):
        verifier.verify_terminal(candidate, repository=ROOT, require_production=False)


def test_verification_output_symlink_ancestry_cannot_alias_candidate(
    candidate: Path, tmp_path: Path,
) -> None:
    alias = tmp_path / "alias"; alias.symlink_to(candidate, target_is_directory=True)
    output = alias / "verification-inside-candidate"
    with pytest.raises(verifier.VerificationError, match="symlink"):
        verifier.publish_verification(
            candidate, output, repository=ROOT, require_production=False,
        )
    assert not output.exists()


def test_environment_binding_exactly_matches_runner_independent_implementation() -> None:
    # The verifier implementation does not import the producer; the test compares
    # the independently constructed contracts directly.
    assert verifier._environment() == runner._environment_binding()
    assert verifier._environment()["state"]["cgroup_cpu_configuration"] == \
        runner._cgroup_cpu_snapshot()["configuration"]


def test_preregistration_lineage_requires_unique_direct_prereg_only_h1(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"; repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repository, check=True)
    (repository / "base.txt").write_text("base\n")
    subprocess.run(["git", "add", "base.txt"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "h0"], cwd=repository, check=True)
    h0 = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository,
        text=True, capture_output=True, check=True).stdout.strip()
    contract = {"state": {"frozen": True}, "digest": verifier.stable_hash({"frozen": True})}
    prereg = repository / verifier.PREREG_RELATIVE; prereg.parent.mkdir(parents=True)
    _write(prereg, contract)
    subprocess.run(["git", "add", str(verifier.PREREG_RELATIVE)], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "h1 prereg only"], cwd=repository, check=True)
    runtime = {"state": {"git_head": h0}}
    verifier._validate_prereg_lineage(contract, runtime, repository)
    (repository / "extra.txt").write_text("forged sibling\n")
    subprocess.run(["git", "add", "extra.txt"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "--amend", "-qm", "forged h1"], cwd=repository, check=True)
    with pytest.raises(verifier.VerificationError, match="lineage"):
        verifier._validate_prereg_lineage(contract, runtime, repository)


def test_live_resident_binding_uses_protected_ready_and_file_identities() -> None:
    resident_root = Path("/dev/shm/market-analogues/m04r11-candidate-v2") / verifier.GENERATION_ID
    ready = resident_root / "READY.json"
    observation = verifier.observe_ready_strict(ready)
    lease = verifier.resident_file_identity_lease(ready)
    state = {"ready_digest": observation["ready_digest"],
        "content_digest": observation["content_digest"],
        "seal_digest": observation["seal_digest"],
        "ready_file_sha256": observation["ready_file_sha256"],
        "lease": lease, "store_root": str((resident_root / "store").resolve())}
    resident = {**state, "identity_digest": verifier.stable_hash(state)}
    foundation = {"resident_root": str(resident_root)}
    verifier._validate_live_resident(foundation, resident)
    forged = copy.deepcopy(resident); forged["lease"]["files"]["ready"]["st_ino"] += 1
    forged["lease"]["lease_digest"] = verifier.stable_hash(
        verifier._without(forged["lease"], {"lease_digest"})
    )
    with pytest.raises(verifier.VerificationError, match="live resident"):
        verifier._validate_live_resident(foundation, forged)


def test_prerequisite_fixed_shas_bind_real_protected_inputs() -> None:
    catalog = ROOT / "config/data/analogues/m04r14/evidence-catalog-v1/catalog.json"
    oracle = ROOT / "config/data/analogues/m04r14/adversarial-oracle-v1/oracle.json"
    assert verifier._sha(catalog) == verifier.CATALOG_SHA256
    assert verifier._sha(oracle) == verifier.ORACLE_SHA256
    forged = {"evidence_catalog_sha256": "0" * 64,
        "adversarial_oracle_sha256": verifier.ORACLE_SHA256}
    with pytest.raises(verifier.VerificationError, match="fixed SHA"):
        verifier._validate_prerequisites(ROOT, forged)
