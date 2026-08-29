from __future__ import annotations

import copy
from dataclasses import dataclass
import importlib.util
import ast
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest

from market_analogues.certified_packed_search import certified_packed_search_contract
from market_analogues.packed_bound_search import packed_bound_search_contract
from market_analogues.types import stable_hash


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module


verifier = load("m04r14_all60_verifier_tested",
    "experiments/m04r/verify_m04r14_all60_certified_poc.py")
contract = verifier.contract
experiments = ModuleType("experiments"); experiments.__path__ = [str(ROOT / "experiments")]
m04r = ModuleType("experiments.m04r"); m04r.__path__ = [str(ROOT / "experiments/m04r")]
experiments.m04r = m04r; m04r.m04r14_all60_contract = contract
sys.modules.setdefault("experiments", experiments)
sys.modules.setdefault("experiments.m04r", m04r)
sys.modules["experiments.m04r.m04r14_all60_contract"] = contract
producer = load("m04r14_all60_producer_fixture",
    "experiments/m04r/m04r14_all60_certified_poc.py")


def proposal_report(query_id: str, order: str, rows: int) -> dict:
    search = packed_bound_search_contract(branch_aware=True)
    deterministic = {"schema_version": search["schema_version"],
        "contract_digest": search["digest"], "generation_id": "generation",
        "query_episode_id": query_id, "rows_scanned": 0, "eligible_rows": 0,
        "eligible_main_rows": 0, "eligible_overflow_rows": 0,
        "route_counts": {"composite": 0}, "route_quotas": {"composite": 16385},
        "candidate_digest": stable_hash([]), "real_forward_outcomes_accessed": False,
        "input_digest": "packed-input"}
    return {**{key: value for key, value in deterministic.items()
               if key != "real_forward_outcomes_accessed"},
        "candidates": [], "block_rows": rows,
        "block_order": order, "elapsed_seconds": 0.01, "peak_rss_mb": 1.0,
        "result_digest": stable_hash(deterministic)}


def certificate(query_id: str, input_digest: str) -> tuple[dict, list[dict]]:
    matches = [{"episode_id": f"{index + 1:024x}", "symbol": f"S{index}",
        "cutoff": "2020-01-01T00:00:00", "total_distance": float(index + 1),
        "component_distances": {name: float(index + 1) for name in (
            "stage", "price", "candle_volatility", "volume_shock", "market_context",
            "structural", "coarse")}, "alignment": [[0, 0]], "quality_tier": "A"}
        for index in range(20)]
    search = certified_packed_search_contract(requested_positions=True,
        vector_lower_bounds=True, deferred_alignments=True, compact_scored=True,
        native_bound_deferral=True, streaming_threshold_closure=True,
        branch_aware_packed_bounds=True)
    accounting = {"native_bound_evaluated": 20, "exact_dtw_evaluated": 20,
        "native_bound_pruned": 0, "packed_bound_pruned": 0}
    row = {"schema_version": search["schema_version"], "contract_digest": search["digest"],
        "generation_id": "generation", "query_episode_id": query_id,
        "input_digest": input_digest, "eligible_candidates": 20,
        "exact_evaluated": 20, "safely_pruned": 0, "stopped_early": True,
        "stop_threshold": 20.0, "next_lower_bound": 21.0,
        "maximum_quantized_bound_excess": 0.0, "materialization_groups": 20,
        "sparse_symbols": 20, "batch_symbols": 0,
        "rounds": [{"frontier_rows": 20, "exact_rows": 20,
            "next_lower_bound": 21.0, "constrained_threshold": 20.0,
            "selected_rows": 20, "certified": True, "proposal_digest": "a" * 64}],
        "native_bound_accounting": accounting, "minimum_native_pruned_bound": None,
        "threshold_closure_passes": []}
    deterministic = {"schema_version": search["schema_version"],
        "contract_digest": search["digest"], "generation_id": "generation",
        "query_episode_id": query_id, "input_digest": input_digest,
        "eligible_candidates": 20, "exact_evaluated": 20, "safely_pruned": 0,
        "stopped_early": True, "stop_threshold_hex": (20.0).hex(),
        "next_lower_bound_hex": (21.0).hex(),
        "maximum_quantized_bound_excess_hex": (0.0).hex(), "rounds": row["rounds"],
        "matches": [{"episode_id": match["episode_id"],
            "total_hex": match["total_distance"].hex(), "components": {key: value.hex()
                for key, value in sorted(match["component_distances"].items())},
            "alignment": match["alignment"]} for match in matches],
        "real_forward_outcomes_accessed": False, "native_bound_accounting": accounting,
        "minimum_native_pruned_bound_hex": None, "threshold_closure_passes": []}
    row["result_digest"] = stable_hash(deterministic)
    return row, matches


@dataclass(frozen=True)
class Case:
    ordinal: int
    case_id: str
    query_id: str


class Backend:
    def __init__(self):
        self.cases = tuple(Case(i, f"case-{i:03d}", query_id)
                           for i, query_id in enumerate(contract.QUERY_IDS))

    @staticmethod
    def binding(ordinal):
        return {"certified_input_digest": f"input-{ordinal}",
                "packed_query_input_digest": "packed-input"}

    def source_lock(self):
        return {"config_sha256": "a" * 64, "registry_sha256": "b" * 64,
            "registry_digest": "c" * 64, "generation_id": "generation",
            "provenance_digest": "d" * 64, "source_tree_digest": "c" * 64,
            "query_binding_digests": [stable_hash(self.binding(i)) for i in range(60)]}

    def resident_binding(self):
        lease_state = {"ready_digest": "a" * 64, "ready_file_sha256": "b" * 64,
                       "content_digest": "c" * 64,
                       "schema_version": "m04r-resident-file-identity-lease-v1",
                       "files": {"ready": {"path": "/resident/READY.json",
                           "st_dev": 1, "st_ino": 2, "st_size": 3,
                           "st_mtime_ns": 4, "st_ctime_ns": 5,
                           "st_mode": 33188}}}
        lease = {**lease_state, "lease_digest": stable_hash(lease_state)}
        state = {"ready_digest": "a" * 64, "ready_file_sha256": "b" * 64,
            "content_digest": "c" * 64, "seal_digest": "d" * 64, "lease": lease,
            "store_root": "/resident/store"}
        return {**state, "identity_digest": stable_hash(state)}

    def prepare(self, ordinal, task_id):
        case = self.cases[ordinal]
        lease_digest = self.resident_binding()["lease"]["lease_digest"]
        binding = self.binding(ordinal)
        forward = proposal_report(case.query_id, "forward", 4096)
        reverse = proposal_report(case.query_id, "reverse", 4097)
        semantic = {"schema_version": "m04r14-exact-scheduler-proposal-v1", "task_id": task_id,
            "case_id": case.case_id, "query_id": case.query_id, "query_binding": binding,
            "resident_lease_digests": [lease_digest] * 4,
            "source_binding_before": binding, "source_binding_after": binding,
            "resident_snapshot": self.resident_binding(), "forward": forward, "reverse": reverse,
            "semantic_digest": stable_hash(verifier._strip_timing(forward))}
        measurement = {"forward_seconds": .01, "reverse_seconds": .01,
            "wall_seconds": .03, "resources": {"after": {"swap_kib": 0}},
            "spawned_process": {"pid": ordinal + 1, "effective_peak_rss_kib": 1024,
                "peak_swap_kib": 0, "final_swap_kib": 0}}
        return SimpleNamespace(task_id=task_id, case=case, semantic=semantic,
                               measurement=measurement)

    def bind_proposal(self, prepared, path):
        assert path.exists()

    def exact(self, prepared, workers=1):
        cert, matches = certificate(prepared.case.query_id,
                                    prepared.semantic["query_binding"]["certified_input_digest"])
        semantic = {"schema_version": "m04r14-exact-scheduler-attempt-v1",
            "case_id": prepared.case.case_id,
            "query_id": prepared.case.query_id, "workers": 1,
            "proposal_semantic_digest": prepared.semantic["semantic_digest"],
            "certificate": cert, "matches": matches,
            "certificate_result_digest": cert["result_digest"],
            "match_digest": stable_hash(matches),
            "lease_before": self.resident_binding()["lease"]["lease_digest"],
            "lease_after": self.resident_binding()["lease"]["lease_digest"],
            "source_binding_before": prepared.semantic["query_binding"],
            "source_binding_after": prepared.semantic["query_binding"]}
        measurement = {"wall_seconds": .02, "engine_seconds": .01,
            "resources": {"after": {"swap_kib": 0}},
            "spawned_process": {"pid": prepared.case.ordinal + 100,
                "effective_peak_rss_kib": 1024, "peak_swap_kib": 0,
                "final_swap_kib": 0}}
        return SimpleNamespace(semantic=semantic, measurement=measurement)

    def final_source_lease(self):
        return {"source_tree_digest": "c" * 64,
                "query_binding_digests": [stable_hash(self.binding(i)) for i in range(60)]}
    def final_resident_lease(self):
        resident = self.resident_binding()
        return {"identity_digest": resident["identity_digest"],
                "lease_digest": resident["lease"]["lease_digest"]}

    @staticmethod
    def validate_certified(observed_certificate, observed_matches, query_id, query_binding):
        verifier._certificate(observed_certificate, observed_matches, query_id,
                              query_binding["certified_input_digest"])


@pytest.fixture()
def candidate(tmp_path: Path) -> Path:
    root = tmp_path / "candidate"
    runtime_state = {"git_head": "0" * 40,
                     "files": {name: "a" * 64 for name in producer.RUNTIME_FIXED_FILES},
                     "environment": {"test": True}, "contracts": {
                         "descriptor_digest": contract.DESCRIPTOR_DIGEST,
                         "execution_policy": contract.EXECUTION_POLICY}}
    runtime = {"state": runtime_state, "digest": stable_hash(runtime_state)}
    prereg = producer.build_preregistration(runtime_binding=runtime, roots={
        "candidate": str(root), "config": "/config", "registry": "/registry",
        "source": "/source", "resident": "/resident"})
    producer.execute(root, prereg, Backend(), clock=lambda: "2026-08-26T00:00:00+00:00")
    return root


def test_production_shaped_tree_dry_replay_and_fresh_receipt(candidate: Path, tmp_path: Path) -> None:
    state = verifier.dry_replay(candidate, repository=ROOT, require_production=False)
    assert state["passed"] is True and state["verified_cases"] == 60
    assert state["direct_raw_authority_accessed_by_verifier"] is False
    assert state["authority_derived_prerequisite_evidence_accessed_by_verifier"] is True
    assert verifier.dry_replay(candidate, repository=ROOT, require_production=False) == state
    receipt = verifier.publish_verification(candidate, tmp_path / "verification",
        repository=ROOT, require_production=False)
    assert receipt["result_digest"] == state["result_digest"]
    with pytest.raises(verifier.VerificationError, match="absent"):
        verifier.publish_verification(candidate, tmp_path / "verification",
            repository=ROOT, require_production=False)


def test_leaf_mutation_and_event_reorder_fail(candidate: Path) -> None:
    proposal = candidate / f"cases/000-{contract.QUERY_IDS[0]}/PROPOSAL.json"
    proposal.write_bytes(proposal.read_bytes() + b" ")
    with pytest.raises(verifier.VerificationError, match="SHA"):
        verifier.verify_terminal(candidate, repository=ROOT, require_production=False)


def test_incomplete_and_symlink_fail_before_success(candidate: Path, tmp_path: Path) -> None:
    complete = candidate / "COMPLETE.json"; complete.unlink()
    (candidate / "INCOMPLETE.json").write_text("{}")
    with pytest.raises(verifier.VerificationError, match="tree"):
        verifier.verify_terminal(candidate, repository=ROOT, require_production=False)
    alias = tmp_path / "alias"; alias.symlink_to(candidate, target_is_directory=True)
    with pytest.raises(verifier.VerificationError, match="symlink"):
        verifier.verify_terminal(alias, repository=ROOT, require_production=False)


def test_production_mode_mandates_independent_replay(candidate: Path, monkeypatch) -> None:
    resident = json.loads((candidate / "RESIDENT.json").read_text())
    monkeypatch.setattr(verifier, "_runtime_and_lineage", lambda *_args: None)
    monkeypatch.setattr(verifier, "observe_ready_strict", lambda _path: {
        "ready_digest": resident["ready_digest"], "content_digest": resident["content_digest"],
        "ready_file_sha256": resident["ready_file_sha256"]})
    monkeypatch.setattr(verifier, "resident_file_identity_lease", lambda _path: resident["lease"])
    called = []
    monkeypatch.setattr(verifier, "_production_replay", lambda *args: called.append(args))
    verifier.verify_terminal(candidate, repository=ROOT, require_production=True)
    assert len(called) == 1 and len(called[0][2]) == 60
    with pytest.raises(verifier.VerificationError, match="cannot be disabled"):
        verifier.verify_terminal(candidate, repository=ROOT, require_production=True,
                                 replay_scans=False)


def test_production_runtime_envelope_matches_actual_two_key_preregistration(candidate: Path) -> None:
    prereg = json.loads((candidate / "CONTRACT.json").read_text())
    state = verifier._runtime_envelope(prereg)
    assert set(prereg["runtime_binding"]) == {"state", "digest"}
    assert state["contracts"] == {
        "descriptor_digest": contract.DESCRIPTOR_DIGEST,
        "execution_policy": contract.EXECUTION_POLICY,
    }
    incompatible = copy.deepcopy(prereg)
    incompatible["runtime_binding"]["schema_version"] = "not-in-the-contract"
    with pytest.raises(verifier.VerificationError, match="runtime binding"):
        verifier._runtime_envelope(incompatible)


def test_verifier_does_not_import_producer_or_serial_runtime() -> None:
    source = (ROOT / "experiments/m04r/verify_m04r14_all60_certified_poc.py").read_text()
    imported = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import): imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom): imported.append(node.module or "")
    assert not any("m04r14_all60_certified_poc" in name for name in imported)
    assert not any("m04r14_serial_certified_runtime" in name for name in imported)


def test_registry_map_requires_frozen_exact_60_without_hidden_duplicates() -> None:
    rows = [{"episode_id": query_id} for query_id in contract.QUERY_IDS]
    registry = {"registry_digest": verifier.FROZEN_REGISTRY_DIGEST, "cases_data": rows}
    source = {"registry_digest": verifier.FROZEN_REGISTRY_DIGEST}
    assert tuple(verifier._registry_case_map(registry, source)) == contract.QUERY_IDS
    with pytest.raises(verifier.VerificationError, match="binding"):
        verifier._registry_case_map({**registry, "registry_digest": "0" * 64}, source)
    with pytest.raises(verifier.VerificationError, match="binding"):
        verifier._registry_case_map({**registry, "cases_data": rows + [rows[0]]}, source)
    duplicate = rows[:-1] + [rows[0]]
    with pytest.raises(verifier.VerificationError, match="universe"):
        verifier._registry_case_map({**registry, "cases_data": duplicate}, source)
