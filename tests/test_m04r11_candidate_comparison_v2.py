from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from market_analogues.m04r_certified_search_verification import _certificate_digest
from market_analogues.types import stable_hash


def _module():
    path = (
        Path(__file__).parents[1] / "experiments" / "m04r"
        / "compare_m04r11_candidate_matrix_v2.py"
    )
    spec = importlib.util.spec_from_file_location(
        "compare_m04r11_candidate_matrix_v2", path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _registry() -> dict:
    cases = []
    for index in range(60):
        cases.append({
            "case_id": f"case-{index:02d}",
            "episode_id": f"{index + 1:024x}",
            "symbol": f"Q{index}",
            "cutoff": "2022-01-01",
            "stock_prefix": {"digest": f"stock-{index}"},
            "benchmark_prefix": {"digest": f"benchmark-{index}"},
        })
    return {
        "registry_digest": (
            "0a4da732f91375a091775cb04e6e77c8d136ade47d7f4d16508a2d9a6555361e"
        ),
        "cases_data": cases,
    }


def _truth_ids(case_index: int) -> list[str]:
    return [f"{case_index + 1:08x}{rank + 10_000:016x}" for rank in range(20)]


def _preopen(module, registry: dict, retained_counts: list[int]):
    semantics = {}
    for index, case in enumerate(registry["cases_data"]):
        truth = _truth_ids(index)
        retained = truth[:retained_counts[index]]
        fillers = [
            f"{0xF0000000 + index:08x}{rank + 50_000:016x}"
            for rank in range(20 - len(retained))
        ]
        semantics[case["episode_id"]] = {
            "semantic_digest": f"{index + 200:064x}",
            "candidates": [{"episode_id": value} for value in retained + fillers],
        }
    performance_final = {
        "performance_passed": False,
        "final_digest": "f" * 64,
    }
    return module.PreopenEvidence(
        registry=registry,
        contract={"contract_digest": "c" * 64},
        preregistration={},
        resident_binding={},
        bundles=(),
        semantics_by_id=semantics,
        performance_by_id={},
        semantic_matrix={},
        semantic_seal={"seal_digest": "s" * 64},
        performance_matrix={"passed": False},
        performance_final=performance_final,
        run_complete={"complete_digest": "r" * 64},
        ledger_head={},
    )


def _authority_case(module, case: dict, contract_digest: str, index: int) -> dict:
    matches = [{
        "episode_id": episode_id,
        "symbol": f"S{rank}",
        "cutoff": "2020-01-01",
        "total_distance": float(rank + 1),
        "component_distances": {"price": float(rank + 1)},
        "alignment": [[0, 0]],
        "quality_tier": "A",
    } for rank, episode_id in enumerate(_truth_ids(index))]
    certificate = {
        "schema_version": "certified-search-certificate-v1",
        "contract_digest": "e" * 64,
        "generation_id": module.FROZEN_GENERATION_ID,
        "query_episode_id": case["episode_id"],
        "input_digest": f"{index + 500:064x}",
        "eligible_candidates": 20,
        "exact_evaluated": 20,
        "safely_pruned": 0,
        "stopped_early": False,
        "stop_threshold": 20.0,
        "next_lower_bound": None,
        "maximum_quantized_bound_excess": 0.0,
        "rounds": [],
    }
    certificate["result_digest"] = _certificate_digest({
        "certificate": certificate, "matches": matches,
    })
    payload = {
        "schema_version": module.AUTHORITY_CASE_SCHEMA,
        "status": "completed",
        "contract_digest": contract_digest,
        "registry_digest": module.FROZEN_REGISTRY_DIGEST,
        "generation_id": module.FROZEN_GENERATION_ID,
        "registry_case_id": case["case_id"],
        "query_episode_id": case["episode_id"],
        "query_symbol": case["symbol"],
        "query_cutoff": case["cutoff"],
        "query_start": "2021-01-01",
        "latest_eligible_cutoff": "2022-01-01",
        "query_stock_prefix": case["stock_prefix"],
        "query_benchmark_prefix": case["benchmark_prefix"],
        "matches": matches,
        "certificate": certificate,
        "certificate_digest": certificate["result_digest"],
        "real_forward_outcomes_accessed": False,
        "created_at": "before-comparison",
    }
    gates = module._authority_case_gates(payload, case)
    gates["frontier_execution_policy"] = True
    payload["gates"] = gates
    payload["gate_passed"] = all(gates.values())
    payload["result_digest"] = module._authority_case_digest(payload)
    payload["checkpoint_integrity_digest"] = (
        module._authority_checkpoint_digest(payload)
    )
    return payload


def _write_authority(module, artifact_dir: Path, registry: dict) -> Path:
    root = Path(module.expected_roots(artifact_dir)["authority_root"])
    root.joinpath("cases").mkdir(parents=True)
    implementation_files = {"authority.py": "a" * 64}
    authority_contract = {
        "schema_version": module.AUTHORITY_CONTRACT_SCHEMA,
        "registry_digest": module.FROZEN_REGISTRY_DIGEST,
        "generation_id": module.FROZEN_GENERATION_ID,
        "expected_query_episode_ids": [
            case["episode_id"] for case in registry["cases_data"]
        ],
        "authority_root_policy": "write-isolated truth; no candidate result input",
        "implementation_manifest": {
            "files": implementation_files,
            "digest": stable_hash(implementation_files),
        },
        "real_forward_outcomes_accessed": False,
    }
    authority_contract["contract_digest"] = stable_hash(authority_contract)
    rows = []
    for index, case in enumerate(registry["cases_data"]):
        payload = _authority_case(
            module, case, authority_contract["contract_digest"], index,
        )
        (root / "cases" / f"{case['episode_id']}.json").write_text(
            json.dumps(payload)
        )
        rows.append({
            "registry_case_id": case["case_id"],
            "query_episode_id": case["episode_id"],
            "authority_digest": payload["result_digest"],
            "certificate_digest": payload["certificate_digest"],
        })
    matrix = {
        "schema_version": module.AUTHORITY_MATRIX_SCHEMA,
        "contract_digest": authority_contract["contract_digest"],
        "registry_digest": module.FROZEN_REGISTRY_DIGEST,
        "generation_id": module.FROZEN_GENERATION_ID,
        "cases": rows,
        "completed_cases": 60,
        "invalid_cases": [],
        "measurements": [],
        "p95_exact_seconds": 1.0,
        "maximum_exact_seconds": 1.0,
        "total_exact_seconds": 60.0,
        "p95_search_seconds": 2.0,
        "maximum_search_seconds": 2.0,
        "total_search_seconds": 120.0,
        "maximum_worker_rss_mb": 100.0,
        "performance_gates": {"synthetic_performance": True},
        "performance_gate_passed": True,
        "gate_passed": True,
        "created_at": "before-comparison",
    }
    matrix["measurement_integrity_digest"] = (
        module._authority_measurement_digest(matrix)
    )
    matrix["result_digest"] = module._authority_matrix_digest(matrix)
    seal = {
        "schema_version": module.AUTHORITY_SEAL_SCHEMA,
        "contract_digest": authority_contract["contract_digest"],
        "registry_digest": module.FROZEN_REGISTRY_DIGEST,
        "authority_matrix_digest": matrix["result_digest"],
        "measurement_integrity_digest": matrix["measurement_integrity_digest"],
        "authority_cases": 60,
        "seal_scope": "exact authority correctness only",
        "authority_correctness_sealed": True,
        "performance_gate_passed": True,
        "production_promotion_authorized": False,
        "candidate_results_opened": False,
        "real_forward_outcomes_accessed": False,
    }
    seal["seal_digest"] = stable_hash(seal)
    (root / "authority-contract.json").write_text(json.dumps(authority_contract))
    (root / "authority-matrix.json").write_text(json.dumps(matrix))
    (root / "SEALED.json").write_text(json.dumps(seal))
    return root


def test_marker_is_durable_before_every_authority_read_and_terminal_fail_allowed(
    tmp_path: Path,
) -> None:
    module = _module()
    registry = _registry()
    artifact_dir = tmp_path / "artifacts"
    authority_root = _write_authority(module, artifact_dir, registry)
    output_root = Path(module.expected_roots(artifact_dir)["comparison_root"])
    preopen = _preopen(module, registry, [20] * 60)
    reads: list[str] = []

    def instrumented_reader(path: Path) -> dict:
        marker_path = output_root / "RESULTS_OPENED.json"
        assert marker_path.is_file()
        marker = json.loads(marker_path.read_text())
        deterministic = {
            key: value for key, value in marker.items()
            if key not in {"created_at", "result_digest"}
        }
        assert marker["result_digest"] == stable_hash(deterministic)
        reads.append(path.name)
        return json.loads(path.read_text())

    result = module.compare_once(
        preopen, authority_root=authority_root, output_root=output_root,
        artifact_dir=artifact_dir, authority_reader=instrumented_reader,
    )
    assert reads[:3] == [
        "authority-contract.json", "authority-matrix.json", "SEALED.json",
    ]
    assert len(reads) == 63
    assert result["passed"] is True
    assert result["performance_passed"] is False
    assert result["retained_total"] == 1_200
    assert (output_root / "SEALED.json").is_file()


def test_terminal_performance_fail_is_valid_but_nonterminal_is_rejected() -> None:
    module = _module()
    matrix = {"result_digest": "m" * 64, "passed": False}
    deterministic = {
        "schema_version": module.PERFORMANCE_FINAL_SCHEMA,
        "producer_contract_digest": "c" * 64,
        "performance_matrix_digest": matrix["result_digest"],
        "resident_binding_digest": "b" * 64,
        "performance_terminal": True,
        "performance_passed": False,
        "confirmatory_performance_cases": 53,
        "exposed_regression_cases": 7,
        "claims_policy": module.CLAIMS_POLICY,
        "authority_results_opened": False,
        "production_promotion_authorized": False,
    }
    final = {
        **deterministic, "created_at": "2026-08-26T00:00:00+00:00",
        "final_digest": stable_hash(deterministic),
    }
    module._validate_terminal_performance(
        matrix, final, "c" * 64, "b" * 64,
    )
    nonterminal = deepcopy(final)
    nonterminal["performance_terminal"] = False
    nonterminal["final_digest"] = module.terminal_digest(
        nonterminal, "final_digest",
    )
    with pytest.raises(ValueError, match="terminal performance"):
        module._validate_terminal_performance(
            matrix, nonterminal, "c" * 64, "b" * 64,
        )

    wrong_resident = deepcopy(final)
    wrong_resident["resident_binding_digest"] = "d" * 64
    wrong_resident["final_digest"] = module.terminal_digest(
        wrong_resident, "final_digest",
    )
    with pytest.raises(ValueError, match="terminal performance"):
        module._validate_terminal_performance(
            matrix, wrong_resident, "c" * 64, "b" * 64,
        )


def test_semantic_seal_binds_both_resident_boundaries() -> None:
    module = _module()
    matrix = {
        "result_digest": "m" * 64,
        "resident_start_content_digest": "r" * 64,
        "resident_end_content_digest": "r" * 64,
    }
    deterministic = {
        "schema_version": module.SEMANTIC_SEAL_SCHEMA,
        "producer_contract_digest": "c" * 64,
        "semantic_matrix_digest": matrix["result_digest"],
        "resident_start_content_digest": "r" * 64,
        "resident_end_content_digest": "r" * 64,
        "semantic_cases": 60,
        "semantic_recall_ready": True,
        "performance_independent": True,
        "authority_results_opened": False,
        "production_promotion_authorized": False,
    }
    seal = {
        **deterministic, "created_at": "2026-08-26T00:00:00+00:00",
        "seal_digest": stable_hash(deterministic),
    }
    module._validate_semantic_seal(matrix, seal, "c" * 64, "r" * 64)
    changed = deepcopy(seal)
    changed["resident_end_content_digest"] = "x" * 64
    changed["seal_digest"] = module.terminal_digest(changed, "seal_digest")
    with pytest.raises(ValueError, match="semantic seal"):
        module._validate_semantic_seal(
            matrix, changed, "c" * 64, "r" * 64,
        )


def test_create_only_publication_preserves_preexisting_inode_bytes(
    tmp_path: Path,
) -> None:
    module = _module()
    path = tmp_path / "RESULTS_OPENED.json"
    original = b"preexisting-owner\n"
    path.write_bytes(original)
    inode = path.stat().st_ino
    with pytest.raises(FileExistsError):
        module._atomic_json(path, {"replacement": True})
    assert path.read_bytes() == original
    assert path.stat().st_ino == inode


def test_exact_candidate_tree_rejects_extra_files_directories_and_links(
    tmp_path: Path,
) -> None:
    module = _module()
    root = tmp_path / "candidate"
    execution = ["1" * 24]
    files = {
        "candidate-contract.json", "RESIDENT_READY.json", "ledger/HEAD.json",
        "semantic-matrix.json", "SEMANTIC_SEALED.json",
        "performance-matrix.json", "PERFORMANCE_FINAL.json",
        "RUN_COMPLETE.json", "case-bundles/000-111111111111111111111111.json",
        *(f"ledger/events/{index:06d}.json" for index in range(122)),
    }
    for relative in files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    module._assert_exact_candidate_tree(root, execution)

    extra = root / "unexpected"
    extra.mkdir()
    with pytest.raises(ValueError, match="exact sealed tree"):
        module._assert_exact_candidate_tree(root, execution)
    extra.rmdir()
    link = root / "linked"
    link.symlink_to(root / "RUN_COMPLETE.json")
    with pytest.raises(ValueError, match="symbolic link"):
        module._assert_exact_candidate_tree(root, execution)


def test_ledger_rejects_extra_fields_even_when_digest_is_recomputed(
    tmp_path: Path,
) -> None:
    module = _module()
    contract_digest = "c" * 64
    events = tmp_path / "ledger" / "events"
    events.mkdir(parents=True)
    event = {
        "schema_version": module.RUN_LEDGER_EVENT_SCHEMA,
        "producer_contract_digest": contract_digest,
        "event_index": 0,
        "previous_event_digest": module.LEDGER_GENESIS,
        "event_type": "run_started",
        "details": {},
        "created_at": "2026-08-26T00:00:00+00:00",
        "unexpected": "self-consistent extension",
    }
    event["event_digest"] = stable_hash(event)
    (events / "000000.json").write_text(json.dumps(event))
    head_deterministic = {
        "schema_version": module.RUN_LEDGER_HEAD_SCHEMA,
        "producer_contract_digest": contract_digest,
        "event_count": 1,
        "last_event_digest": event["event_digest"],
    }
    (tmp_path / "ledger" / "HEAD.json").write_text(json.dumps({
        **head_deterministic, "head_digest": stable_hash(head_deterministic),
    }))
    with pytest.raises(ValueError, match="ledger event fields differ"):
        module._load_ledger(tmp_path, contract_digest)


def test_nested_candidate_corruption_breaks_atomic_bundle_digest() -> None:
    module = _module()
    bundle = {
        "schema_version": module.CASE_BUNDLE_SCHEMA,
        "producer_contract_digest": "c" * 64,
        "execution_ordinal": 0,
        "query_episode_id": "1" * 24,
        "semantic": {"candidates": [{"episode_id": "2" * 24}]},
        "performance": {},
    }
    bundle["bundle_digest"] = module._bundle_digest(bundle)
    bundle["semantic"]["candidates"][0]["episode_id"] = "3" * 24
    with pytest.raises(ValueError, match="bundle identity or digest"):
        module._validate_bundle(
            bundle, contract={"contract_digest": "c" * 64},
            case={"episode_id": "1" * 24}, role={}, expected_query={},
            resident={}, physical_rows=1, execution_ordinal=0,
        )


@pytest.mark.parametrize(
    ("retained", "per_case_pass", "aggregate_pass"),
    [
        ([18] + [20] * 59, False, True),
        ([19] * 13 + [20] * 47, True, False),
    ],
)
def test_recall_thresholds_are_independent_and_exact(
    tmp_path: Path, retained: list[int], per_case_pass: bool,
    aggregate_pass: bool,
) -> None:
    module = _module()
    registry = _registry()
    artifact_dir = tmp_path / "artifacts"
    authority_root = _write_authority(module, artifact_dir, registry)
    output_root = Path(module.expected_roots(artifact_dir)["comparison_root"])
    result = module.compare_once(
        _preopen(module, registry, retained), authority_root=authority_root,
        output_root=output_root, artifact_dir=artifact_dir,
    )
    assert result["gates"][
        "every_case_retains_at_least_19_of_20"
    ] is per_case_pass
    assert result["gates"][
        "aggregate_retains_at_least_1188_of_1200"
    ] is aggregate_pass
    assert result["perfect_20_of_20_is_descriptive_only"] is True
    assert result["passed"] is False
    assert (output_root / "SEALED.json").is_file()


def test_reopen_and_partial_open_are_both_fail_closed(tmp_path: Path) -> None:
    module = _module()
    registry = _registry()
    artifact_dir = tmp_path / "artifacts"
    authority_root = _write_authority(module, artifact_dir, registry)
    output_root = Path(module.expected_roots(artifact_dir)["comparison_root"])
    preopen = _preopen(module, registry, [20] * 60)
    module.compare_once(
        preopen, authority_root=authority_root, output_root=output_root,
        artifact_dir=artifact_dir,
    )
    with pytest.raises(ValueError, match="reopen is forbidden"):
        module.compare_once(
            preopen, authority_root=authority_root, output_root=output_root,
            artifact_dir=artifact_dir,
        )

    partial_artifact = tmp_path / "partial-artifacts"
    partial_root = Path(module.expected_roots(partial_artifact)["comparison_root"])
    partial_root.mkdir(parents=True)
    (partial_root / "RESULTS_OPENED.json").write_text("{}")
    partial_authority = _write_authority(module, partial_artifact, registry)
    with pytest.raises(ValueError, match="previously opened without a seal"):
        module.compare_once(
            preopen, authority_root=partial_authority,
            output_root=partial_root, artifact_dir=partial_artifact,
        )


def test_results_opened_race_never_replaces_foreign_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    registry = _registry()
    artifact_dir = tmp_path / "artifacts"
    authority_root = _write_authority(module, artifact_dir, registry)
    output_root = Path(module.expected_roots(artifact_dir)["comparison_root"])
    original_fresh = module._fresh_comparison_root
    sentinel = b"foreign marker owner\n"

    def race(root: Path) -> None:
        original_fresh(root)
        root.mkdir(parents=True)
        (root / "RESULTS_OPENED.json").write_bytes(sentinel)

    monkeypatch.setattr(module, "_fresh_comparison_root", race)
    reads: list[Path] = []
    with pytest.raises(FileExistsError):
        module.compare_once(
            _preopen(module, registry, [20] * 60),
            authority_root=authority_root, output_root=output_root,
            artifact_dir=artifact_dir,
            authority_reader=lambda path: reads.append(path) or {},
        )
    assert reads == []
    assert (output_root / "RESULTS_OPENED.json").read_bytes() == sentinel


@pytest.mark.parametrize("owned_name", ["candidate-comparison.json", "SEALED.json"])
def test_comparison_and_seal_races_preserve_foreign_owned_bytes(
    tmp_path: Path, owned_name: str,
) -> None:
    module = _module()
    registry = _registry()
    artifact_dir = tmp_path / "artifacts"
    authority_root = _write_authority(module, artifact_dir, registry)
    output_root = Path(module.expected_roots(artifact_dir)["comparison_root"])
    sentinel = f"foreign owner:{owned_name}\n".encode()
    reads = 0

    def racing_reader(path: Path) -> dict:
        nonlocal reads
        reads += 1
        if reads == 63:
            (output_root / owned_name).write_bytes(sentinel)
        return json.loads(path.read_text())

    with pytest.raises(FileExistsError):
        module.compare_once(
            _preopen(module, registry, [20] * 60),
            authority_root=authority_root, output_root=output_root,
            artifact_dir=artifact_dir, authority_reader=racing_reader,
        )
    assert reads == 63
    assert (output_root / owned_name).read_bytes() == sentinel
