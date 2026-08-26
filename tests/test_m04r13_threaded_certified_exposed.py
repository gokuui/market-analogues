from __future__ import annotations

import importlib.util
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
from typing import Any
from types import SimpleNamespace

import numpy as np
import pandas as pd

import pytest

from market_analogues.types import stable_hash
from market_analogues.representation import Representation
from market_analogues.causal_prefix import CausalPrefixDigest


REPOSITORY = Path(__file__).resolve().parents[1]
PRODUCER_PATH = REPOSITORY / "experiments/m04r/m04r13_threaded_certified_exposed.py"
COMPARATOR_PATH = REPOSITORY / "experiments/m04r/compare_m04r13_threaded_certified_exposed.py"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


producer = _load("m04r13_threaded_certified_exposed", PRODUCER_PATH)
comparator = _load("compare_m04r13_threaded_certified_exposed", COMPARATOR_PATH)


def _components(value: float) -> dict[str, float]:
    return {name: value for name in sorted(producer.COMPONENT_NAMES)}


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _proposal(query_id: str, block_rows: int, block_order: str) -> dict[str, Any]:
    contract = producer.packed_bound_search_contract(branch_aware=True)["digest"]
    cutoff_ns = pd.Timestamp("2019-01-01", tz="UTC").value
    rows = tuple(producer.BoundProposal(
        f"{index:024x}", f"S{index}", cutoff_ns, "A", float(index),
        ("composite",), False,
    ) for index in range(40))
    candidates = [{
        "episode_id": row.episode_id, "symbol": row.symbol,
        "cutoff_ns": row.cutoff_ns, "quality_tier": row.quality_tier,
        "lower_bound_hex": row.lower_bound.hex(), "routes": list(row.routes),
        "overflow_fallback": row.overflow_fallback,
    } for row in rows]
    candidate_digest = producer.bound_proposal_candidate_digest(rows)
    deterministic = {
        "schema_version": producer.BRANCH_AWARE_SEARCH_SCHEMA_VERSION,
        "contract_digest": contract,
        "generation_id": producer.GENERATION_ID,
        "query_episode_id": query_id, "rows_scanned": 40,
        "eligible_rows": 40, "eligible_main_rows": 40,
        "eligible_overflow_rows": 0, "route_counts": {"composite": 40},
        "route_quotas": {"composite": producer.PROPOSAL_QUOTA},
        "candidate_digest": candidate_digest,
        "real_forward_outcomes_accessed": False, "input_digest": "input",
    }
    return {
        "schema_version": deterministic["schema_version"],
        "generation_id": producer.GENERATION_ID,
        "query_episode_id": query_id, "candidates": candidates, "rows_scanned": 40,
        "eligible_rows": 40, "eligible_main_rows": 40,
        "eligible_overflow_rows": 0, "route_counts": {"composite": 40},
        "route_quotas": {"composite": producer.PROPOSAL_QUOTA},
        "block_rows": block_rows, "block_order": block_order,
        "elapsed_seconds": 0.1, "peak_rss_mb": 10.0,
        "candidate_digest": candidate_digest,
        "result_digest": stable_hash(deterministic), "contract_digest": contract,
        "input_digest": "input",
    }


def _producer_fixture(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    resident_root = root / "resident-mirror"
    ready_digest, content_digest = "a" * 64, "b" * 64
    file_identity = {
        "path": str(resident_root / "READY.json"), "st_dev": 1, "st_ino": 2,
        "st_size": 3, "st_mtime_ns": 4, "st_ctime_ns": 5, "st_mode": 32_768,
    }
    lease_deterministic = {
        "schema_version": "m04r-resident-file-identity-lease-v1",
        "ready_digest": ready_digest, "ready_file_sha256": "d" * 64,
        "content_digest": content_digest,
        "files": {"ready": file_identity},
    }
    lease = {**lease_deterministic, "lease_digest": stable_hash(lease_deterministic)}
    resident_deterministic = {
        "ready_digest": ready_digest, "content_digest": content_digest,
        "seal_digest": "c" * 64, "ready_file_sha256": "d" * 64, "lease": lease,
        "store_root": str((resident_root / "store").resolve()),
    }
    resident = {
        **resident_deterministic,
        "identity_digest": stable_hash(resident_deterministic),
    }
    git_deterministic = {
        "implementation_commit": "0" * 40, "files": {},
        "files_digest": stable_hash({}),
    }
    git = {**git_deterministic, "digest": stable_hash(git_deterministic)}
    prereg_deterministic = {
        "schema_version": producer.PREREG_SCHEMA,
        "status": "frozen_before_producer", "development_only": True,
        "truth_roots_allowed_in_producer": False,
        "production_promotion_authorized": False,
        "real_forward_outcomes_accessed": False, "git": git,
        "config_path": str((root / "config.yaml").resolve()),
        "config_sha256": "e" * 64,
        "roots": {
            "registry_root": str((root / "registry").resolve()),
            "source_full_root": str((root / "source").resolve()),
            "resident_root": str(resident_root.resolve()),
            "output_root": str(root.resolve()),
        },
        "generation_id": producer.GENERATION_ID, "provenance_digest": "1" * 64,
        "reserve_bytes": producer.RESIDENT_RESERVE_BYTES,
        "registry_digest": "f" * 64,
        "query_ids": list(producer.FROZEN_QUERY_IDS),
        "registry_cases_digest": "2" * 64,
        "resident_content_digest": content_digest,
        "resident_ready_digest": ready_digest,
        "resident_identity_digest": resident["identity_digest"],
        "finite_threshold_diagnostic": {
            "path": str((root / "finite-threshold-diagnostic.json").resolve()),
            "sha256": "3" * 64, "result_digest": "4" * 64,
        },
        "execution": producer._execution_policy(),
        "environment": producer._environment_binding(),
    }
    prereg = {
        **prereg_deterministic,
        "preregistration_digest": stable_hash(prereg_deterministic),
    }
    _write(root / "CONTRACT.json", prereg)
    _write(root / "RESIDENT.json", resident)
    _write(root / "RUN_STARTED.json", {
        "schema_version": "m04r13-run-started-v1",
        "preregistration_digest": prereg["preregistration_digest"],
        "case_order": list(producer.FROZEN_QUERY_IDS), "parent_max_workers": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
    })
    authorities: dict[str, dict[str, Any]] = {}
    case_digests = []
    for index, query_id in enumerate(producer.FROZEN_QUERY_IDS):
        forward = _proposal(query_id, 4_096, "forward")
        reverse = _proposal(query_id, 4_097, "reverse")
        matches = [{
            "episode_id": f"{row:024x}", "symbol": f"S{row}",
            "cutoff": "2019-01-01T00:00:00+00:00", "total_distance": float(row),
            "component_distances": _components(float(row)),
            "alignment": [[0, 0]], "quality_tier": "A",
        } for row in range(20)]
        contract = comparator.certified_packed_search_contract(
            requested_positions=True, vector_lower_bounds=True,
            deferred_alignments=True, compact_scored=True,
            native_bound_deferral=True, streaming_threshold_closure=True,
            branch_aware_packed_bounds=True,
        )
        rounds = [{
            "frontier_rows": 40, "exact_rows": 20, "next_lower_bound": None,
            "constrained_threshold": 19.0, "selected_rows": 20,
            "certified": True, "proposal_digest": forward["candidate_digest"],
        }]
        accounting = {
            "native_bound_evaluated": 20, "exact_dtw_evaluated": 20,
            "native_bound_pruned": 0, "packed_bound_pruned": 20,
        }
        certificate = {
            "schema_version": contract["schema_version"],
            "contract_digest": contract["digest"],
            "generation_id": producer.GENERATION_ID,
            "query_episode_id": query_id, "input_digest": f"input-{index}",
            "eligible_candidates": 40,
            "exact_evaluated": 20, "safely_pruned": 20,
            "stopped_early": False, "stop_threshold": 19.0,
            "next_lower_bound": None, "maximum_quantized_bound_excess": 0.0,
            "materialization_groups": 1, "sparse_symbols": 1, "batch_symbols": 0,
            "rounds": rounds, "elapsed_seconds": 0.5,
            "native_bound_accounting": accounting,
            "minimum_native_pruned_bound": None, "threshold_closure_passes": [],
        }
        certificate_deterministic = {
            "schema_version": contract["schema_version"],
            "contract_digest": contract["digest"], "generation_id": producer.GENERATION_ID,
            "query_episode_id": query_id, "input_digest": f"input-{index}",
            "eligible_candidates": 40, "exact_evaluated": 20, "safely_pruned": 20,
            "stopped_early": False, "stop_threshold_hex": float(19).hex(),
            "next_lower_bound_hex": None,
            "maximum_quantized_bound_excess_hex": float(0).hex(), "rounds": rounds,
            "matches": [{
                "episode_id": row["episode_id"],
                "total_hex": float(row["total_distance"]).hex(),
                "components": {
                    key: value.hex()
                    for key, value in sorted(row["component_distances"].items())
                },
                "alignment": row["alignment"],
            } for row in matches],
            "real_forward_outcomes_accessed": False,
            "native_bound_accounting": accounting,
            "minimum_native_pruned_bound_hex": None, "threshold_closure_passes": [],
        }
        certificate["result_digest"] = stable_hash(certificate_deterministic)
        binding = {
            "query_stock_prefix": {"digest": f"stock-{index}"},
            "query_benchmark_prefix": {"digest": f"benchmark-{index}"},
            "request": {"top_k": 20}, "packed_provenance_digest": "p" * 64,
            "query_representation_digest": f"representation-{index}",
            "packed_query_input_digest": "input",
            "certified_input_digest": f"input-{index}",
        }
        deterministic = {
            "schema_version": producer.CASE_SCHEMA,
            "status": "truth_blind_case_complete", "development_only": True,
            "truth_opened": False, "production_promotion_authorized": False,
            "real_forward_outcomes_accessed": False,
            "preregistration_digest": prereg["preregistration_digest"],
            "registry_case_id": producer.FROZEN_CASE_IDS[index],
            "query_episode_id": query_id,
            "query_binding": binding,
            "resident_identity_digest": resident["identity_digest"],
            "lease_digests": [lease["lease_digest"]] * 5, "forward_proposal": forward,
            "reverse_proposal": reverse, "proposal_semantic_exact": True,
            "certificate": certificate, "matches": matches,
            "rounds": certificate["rounds"], "certified": True,
            "streaming_fallback_used": False,
            "metrics": {
                "forward_proposal_seconds": 0.1, "reverse_proposal_seconds": 0.1,
                "exact_task_wall_seconds": 1.0, "case_task_wall_seconds": 2.0,
                "process_rss_mb": 10.0,
            },
            "semantic_passed": True, "performance_passed": True,
        }
        case = {
            **deterministic, "created_at": "2026-01-01T00:00:00+00:00",
            "result_digest": stable_hash(deterministic),
        }
        _write(root / "cases" / f"{index:02d}-{query_id}.json", case)
        case_digests.append(case["result_digest"])
        authorities[query_id] = {
            "schema_version": "m04r11-certified-authority-case-v4",
            "status": "completed", "query_episode_id": query_id,
            "matches": matches, "certificate": certificate,
            "result_digest": comparator.AUTHORITY_CASE_BINDINGS[query_id][1],
            "real_forward_outcomes_accessed": False,
            "query_stock_prefix": binding["query_stock_prefix"],
            "query_benchmark_prefix": binding["query_benchmark_prefix"],
        }
    seal_deterministic = {
        "schema_version": producer.SEAL_SCHEMA,
        "status": "truth_blind_producer_complete", "development_only": True,
        "truth_opened": False, "production_promotion_authorized": False,
        "real_forward_outcomes_accessed": False,
        "preregistration_digest": prereg["preregistration_digest"],
        "resident_identity_digest": resident["identity_digest"],
        "query_ids": list(producer.FROZEN_QUERY_IDS),
        "case_digests": case_digests,
        "semantic_passed": True, "performance_passed": True,
    }
    _write(root / "PRODUCER_SEALED.json", {
        **seal_deterministic, "created_at": "2026-01-01T00:00:00+00:00",
        "seal_digest": stable_hash(seal_deterministic),
    })
    return prereg, authorities


def _fixture_bindings(root: Path) -> dict[str, dict[str, Any]]:
    return {
        query_id: producer._read_json(
            root / "cases" / f"{index:02d}-{query_id}.json"
        )["query_binding"]
        for index, query_id in enumerate(producer.FROZEN_QUERY_IDS)
    }


def _fixture_universe(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for index, query_id in enumerate(producer.FROZEN_QUERY_IDS):
        case = producer._read_json(
            root / "cases" / f"{index:02d}-{query_id}.json"
        )
        result[query_id] = {
            row["episode_id"]: {
                "episode_id": row["episode_id"], "symbol": row["symbol"],
                "cutoff_ns": row["cutoff_ns"],
                "quality_tier": row["quality_tier"],
                "branch_aware_bound": float.fromhex(row["lower_bound_hex"]),
                "overflow": row["overflow_fallback"],
            }
            for row in case["forward_proposal"]["candidates"]
        }
    return result


def _truth_free_test_kwargs(root: Path, prereg: dict[str, Any]) -> dict[str, Any]:
    universe = _fixture_universe(root)
    return {
        "repository_prereg": prereg,
        "git_validator": lambda *_args: None,
        "binding_loader": lambda _prereg: _fixture_bindings(root),
        "prereg_validator": lambda value: producer.validate_preregistration_shape(
            value, enforce_topology=False,
        ),
        "universe_loader": lambda _prereg, requested, _cases: (
            {
                query_id: {
                    episode_id: universe[query_id][episode_id]
                    for episode_id in episode_ids
                    if episode_id in universe[query_id]
                }
                for query_id, episode_ids in requested.items()
            },
            {query_id: [] for query_id in requested},
        ),
    }


def test_default_match_universe_uses_local_numeric_dependencies_truth_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_id = producer.FROZEN_QUERY_IDS[0]
    retained_id = "01" * 12
    row_dtype = np.dtype([
        ("episode_id", "V12"), ("symbol_id", "<i4"),
        ("cutoff_ns", "<i8"), ("quality_tier", "u1"),
    ])
    rows = np.zeros(1, dtype=row_dtype)
    rows["episode_id"][0] = np.void(bytes.fromhex(retained_id))
    rows["symbol_id"][0] = 0
    rows["cutoff_ns"][0] = 123
    rows["quality_tier"][0] = 1
    loaded = SimpleNamespace(
        rows=rows, overflow=np.zeros(0, dtype=row_dtype), symbols=("S0",),
    )
    case = producer.CaseInput(0, {
        "episode_id": query_id, "case_id": producer.FROZEN_CASE_IDS[0],
    })
    observed_load: dict[str, Any] = {}

    def load_generation(*_args: Any, **kwargs: Any) -> Any:
        observed_load.update(kwargs)
        return loaded

    monkeypatch.setattr(
        comparator, "_source_generation_identity", lambda _root: {"stable": True},
    )
    monkeypatch.setattr(comparator, "load_packed_generation", load_generation)
    monkeypatch.setattr(
        producer, "_registry_cases",
        lambda *_args: (producer.REGISTRY_DIGEST, (case,)),
    )
    monkeypatch.setattr(
        producer, "_case_context",
        lambda *_args: (
            None, None, None,
            SimpleNamespace(symbol="QUERY", representation=object()),
        ),
    )
    monkeypatch.setattr(
        comparator, "_eligible_mask",
        lambda *_args: np.asarray([True], dtype=np.bool_),
    )
    monkeypatch.setattr(
        comparator, "packed_branch_aware_lower_bounds",
        lambda *_args: SimpleNamespace(totals=np.asarray([0.25])),
    )
    # Third-party modules are comparator dependencies, not producer API.
    monkeypatch.delattr(producer, "np", raising=False)
    monkeypatch.delattr(producer, "pd")
    prereg = {
        "roots": {
            "source_full_root": str(tmp_path / "source"),
            "registry_root": str(tmp_path / "registry"),
            "resident_root": str(tmp_path / "resident"),
            "output_root": str(tmp_path / "output"),
        },
        "config_path": str(tmp_path / "config.yaml"),
        "registry_digest": producer.REGISTRY_DIGEST,
        "reserve_bytes": producer.RESIDENT_RESERVE_BYTES,
        "preregistration_digest": "diagnostic",
    }

    universe, closure_evidence = comparator.reconstruct_match_universe(
        prereg, {query_id: {retained_id}}, [{
            "query_episode_id": query_id,
            "certificate": {"threshold_closure_passes": []},
        }],
    )

    assert observed_load["verify_content"] is True
    assert observed_load["validate_records"] is True
    assert universe == {query_id: {retained_id: {
        "episode_id": retained_id, "symbol": "S0", "cutoff_ns": 123,
        "quality_tier": "A", "branch_aware_bound": 0.25,
        "overflow": False,
    }}}
    assert closure_evidence == {query_id: []}


def _rehash_certificate(certificate: dict[str, Any], matches: list[dict[str, Any]]) -> str:
    return stable_hash({
        "schema_version": certificate["schema_version"],
        "contract_digest": certificate["contract_digest"],
        "generation_id": certificate["generation_id"],
        "query_episode_id": certificate["query_episode_id"],
        "input_digest": certificate["input_digest"],
        "eligible_candidates": certificate["eligible_candidates"],
        "exact_evaluated": certificate["exact_evaluated"],
        "safely_pruned": certificate["safely_pruned"],
        "stopped_early": certificate["stopped_early"],
        "stop_threshold_hex": float(certificate["stop_threshold"]).hex(),
        "next_lower_bound_hex": (
            float(certificate["next_lower_bound"]).hex()
            if certificate["next_lower_bound"] is not None else None
        ),
        "maximum_quantized_bound_excess_hex":
            float(certificate["maximum_quantized_bound_excess"]).hex(),
        "rounds": certificate["rounds"],
        "matches": [{
            "episode_id": row["episode_id"],
            "total_hex": float(row["total_distance"]).hex(),
            "components": {key: float(value).hex()
                           for key, value in sorted(row["component_distances"].items())},
            "alignment": row["alignment"],
        } for row in matches],
        "real_forward_outcomes_accessed": False,
        "native_bound_accounting": certificate["native_bound_accounting"],
        "minimum_native_pruned_bound_hex": (
            float(certificate["minimum_native_pruned_bound"]).hex()
            if certificate["minimum_native_pruned_bound"] is not None else None
        ),
        "threshold_closure_passes": certificate["threshold_closure_passes"],
    })


def _rewrite_case_and_seal(root: Path, case: dict[str, Any]) -> None:
    query_id = case["query_episode_id"]
    index = producer.FROZEN_QUERY_IDS.index(query_id)
    case["result_digest"] = stable_hash(producer._without(
        case, {"created_at", "result_digest"},
    ))
    _write(root / "cases" / f"{index:02d}-{query_id}.json", case)
    seal_path = root / "PRODUCER_SEALED.json"
    seal = producer._read_json(seal_path)
    seal["case_digests"][index] = case["result_digest"]
    seal["seal_digest"] = stable_hash(producer._without(
        seal, {"created_at", "seal_digest"},
    ))
    _write(seal_path, seal)


def test_frozen_preregistration_digest_serialization() -> None:
    payload = {"schema_version": producer.PREREG_SCHEMA, "query_ids": ["a", "b"]}
    frozen = {**payload, "preregistration_digest": stable_hash(payload)}
    assert frozen["preregistration_digest"] == stable_hash(
        producer._without(frozen, {"preregistration_digest"})
    )
    assert str(producer.PREREG_RELATIVE) not in producer.RUNTIME_FILES
    assert "experiments/m04r/verify_m04r13_threaded_certified_exposed.py" \
        in producer.RUNTIME_FILES
    assert "experiments/m04r/m04r13_finite_threshold_diagnostic.py" \
        in producer.RUNTIME_FILES


def test_comparator_durably_marks_before_truth_loader(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    prereg, authorities = _producer_fixture(root)
    observed: list[str] = []

    def load_truth(_authority: Path, _verification: Path) -> dict[str, dict[str, Any]]:
        marker = root / "RESULTS_OPENED.json"
        assert marker.is_file()
        assert producer._read_json(marker)["schema_version"] == comparator.RESULTS_OPENED_SCHEMA
        observed.append("truth-loaded-after-marker")
        return authorities

    seal = comparator.compare(
        root, tmp_path / "truth", tmp_path / "verification.json",
        truth_loader=load_truth, **_truth_free_test_kwargs(root, prereg),
    )
    assert observed == ["truth-loaded-after-marker"]
    assert seal["passed"] is True
    assert (root / "COMPARISON.json").is_file()
    assert (root / "COMPARISON_SEALED.json").is_file()


def test_comparator_rejects_producer_corruption_before_truth(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    prereg, _ = _producer_fixture(root)
    case = root / "cases" / f"00-{producer.FROZEN_QUERY_IDS[0]}.json"
    payload = json.loads(case.read_text())
    payload["truth_opened"] = True
    _write(case, payload)
    truth_opened = False

    def forbidden(_authority: Path, _verification: Path) -> dict[str, dict[str, Any]]:
        nonlocal truth_opened
        truth_opened = True
        return {}

    with pytest.raises(comparator.ComparisonError):
        comparator.compare(
            root, tmp_path / "truth", tmp_path / "verification.json",
            truth_loader=forbidden, **_truth_free_test_kwargs(root, prereg),
        )
    assert truth_opened is False
    assert not (root / "RESULTS_OPENED.json").exists()


def test_producer_cli_has_no_truth_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", [
        str(PRODUCER_PATH), "produce", "--authority-root", "/forbidden",
    ])
    with pytest.raises(SystemExit) as error:
        producer.main()
    assert error.value.code == 2


def _run_case_fixture(tmp_path: Path) -> tuple[Any, dict[str, Any], Any, Any]:
    query_id = producer.FROZEN_QUERY_IDS[0]
    case = producer.CaseInput(0, {"episode_id": query_id, "case_id": "case-0"})
    inputs = producer.Inputs(
        tmp_path, tmp_path / "config.toml", tmp_path / "registry",
        tmp_path / "source/store", tmp_path / "resident", tmp_path / "output",
        producer.GENERATION_ID, "p" * 64, 1024, "registry", (case,), "prereg",
    )
    (inputs.output_root / "cases").mkdir(parents=True)
    representation = Representation(
        pd.DataFrame(), np.zeros(1), {"price": np.zeros(1)},
        {"price": np.zeros(1)}, np.zeros(1), np.zeros(1),
    )
    packed = producer.PackedBoundQuery(query_id, "AAA", 0, 1, representation)
    episode_key = SimpleNamespace(
        id=query_id,
        instrument=SimpleNamespace(source_symbol="AAA"),
        cutoff=pd.Timestamp("2020-01-01", tz="UTC"),
    )
    episode = SimpleNamespace(key=episode_key)
    context = lambda _inputs, _case: (object(), episode, object(), packed)
    resident_base = {
        "ready_digest": "r", "content_digest": "c", "seal_digest": "s",
        "ready_file_sha256": "f", "lease": {"lease_digest": "l"},
        "store_root": str(tmp_path / "resident/store"),
    }
    resident = {**resident_base, "identity_digest": stable_hash(resident_base)}
    return inputs, resident, case, context


def _partial_foundation(inputs: Any, resident: dict[str, Any]) -> None:
    _write(inputs.output_root / "RUN_STARTED.json", {
        "schema_version": "m04r13-run-started-v1",
        "preregistration_digest": inputs.prereg_digest,
        "case_order": list(producer.FROZEN_QUERY_IDS), "parent_max_workers": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
    })
    _write(inputs.output_root / "CONTRACT.json", {"synthetic": True})
    _write(inputs.output_root / "RESIDENT.json", resident)


def _valid_report(
    query: Any, *, order: str, block: int, elapsed: float = 0.1,
    rows_scanned: int = 0,
) -> Any:
    candidates = tuple(producer.BoundProposal(
        f"{index:024x}", f"S{index}", index, "A", float(index),
        ("composite",), False,
    ) for index in range(20))
    contract = producer.packed_bound_search_contract(branch_aware=True)["digest"]
    input_digest = producer._packed_query_input_digest(query)
    candidate_digest = producer.bound_proposal_candidate_digest(candidates)
    deterministic = {
        "schema_version": producer.BRANCH_AWARE_SEARCH_SCHEMA_VERSION,
        "contract_digest": contract, "generation_id": producer.GENERATION_ID,
        "query_episode_id": query.episode_id, "rows_scanned": rows_scanned or 20,
        "eligible_rows": 20, "eligible_main_rows": 20, "eligible_overflow_rows": 0,
        "route_counts": {"composite": 20},
        "route_quotas": {"composite": producer.PROPOSAL_QUOTA},
        "candidate_digest": candidate_digest,
        "real_forward_outcomes_accessed": False, "input_digest": input_digest,
    }
    return producer.BoundProposalReport(
        producer.BRANCH_AWARE_SEARCH_SCHEMA_VERSION, producer.GENERATION_ID,
        query.episode_id, candidates, rows_scanned or 20, 20, 20, 0,
        {"composite": 20},
        {"composite": producer.PROPOSAL_QUOTA}, block, order, elapsed, 10.0,
        candidate_digest, stable_hash(deterministic), contract, input_digest,
    )


def _binding(*args: Any) -> dict[str, Any]:
    packed = args[3]
    return {
        "query_stock_prefix": {"digest": "stock"},
        "query_benchmark_prefix": {"digest": "benchmark"},
        "request": {"top_k": 20}, "packed_provenance_digest": "p" * 64,
        "query_representation_digest": "representation",
        "packed_query_input_digest": producer._packed_query_input_digest(packed),
        "certified_input_digest": "input",
    }


def _certified_result(query_id: str) -> tuple[Any, dict[str, Any]]:
    matches = []
    for index in range(20):
        key = SimpleNamespace(
            id=f"{index:024x}", instrument=SimpleNamespace(source_symbol="SYN"),
            cutoff=pd.Timestamp("2019-01-01", tz="UTC"),
        )
        matches.append(SimpleNamespace(
            episode_key=key, total_distance=float(index),
            component_distances=_components(float(index)), alignment=((0, 0),),
            quality_tier="A",
        ))
    match_payloads = [producer._match(value) for value in matches]
    contract = producer.certified_packed_search_contract(
        requested_positions=True, vector_lower_bounds=True,
        deferred_alignments=True, compact_scored=True,
        native_bound_deferral=True, streaming_threshold_closure=True,
        branch_aware_packed_bounds=True,
    )
    proposal_rows = tuple(producer.BoundProposal(
        f"{index:024x}", f"S{index}", index, "A", float(index),
        ("composite",), False,
    ) for index in range(20))
    rounds = [{
        "frontier_rows": 20, "exact_rows": 20, "next_lower_bound": None,
        "constrained_threshold": 19.0, "selected_rows": 20,
        "certified": True, "proposal_digest":
            producer.bound_proposal_candidate_digest(proposal_rows),
    }]
    accounting = {
        "native_bound_evaluated": 20, "exact_dtw_evaluated": 20,
        "native_bound_pruned": 0, "packed_bound_pruned": 0,
    }
    certificate = {
        "schema_version": contract["schema_version"],
        "contract_digest": contract["digest"], "generation_id": producer.GENERATION_ID,
        "query_episode_id": query_id, "input_digest": "input", "eligible_candidates": 20,
        "exact_evaluated": 20, "safely_pruned": 0, "stopped_early": False,
        "stop_threshold": 19.0, "next_lower_bound": None,
        "maximum_quantized_bound_excess": 0.0, "materialization_groups": 1,
        "sparse_symbols": 1, "batch_symbols": 0, "rounds": rounds,
        "elapsed_seconds": 0.5, "native_bound_accounting": accounting,
        "minimum_native_pruned_bound": None, "threshold_closure_passes": [],
    }
    deterministic = {
        "schema_version": contract["schema_version"], "contract_digest": contract["digest"],
        "generation_id": producer.GENERATION_ID, "query_episode_id": query_id,
        "input_digest": "input", "eligible_candidates": 20, "exact_evaluated": 20,
        "safely_pruned": 0, "stopped_early": False,
        "stop_threshold_hex": float(19).hex(), "next_lower_bound_hex": None,
        "maximum_quantized_bound_excess_hex": float(0).hex(), "rounds": rounds,
        "matches": [{
            "episode_id": row["episode_id"],
            "total_hex": float(row["total_distance"]).hex(),
            "components": {
                key: value.hex()
                for key, value in sorted(row["component_distances"].items())
            },
            "alignment": row["alignment"],
        } for row in match_payloads],
        "real_forward_outcomes_accessed": False,
        "native_bound_accounting": accounting,
        "minimum_native_pruned_bound_hex": None, "threshold_closure_passes": [],
    }
    certificate["result_digest"] = stable_hash(deterministic)
    return SimpleNamespace(
        certificate=SimpleNamespace(**certificate), matches=tuple(matches),
    ), certificate


def test_case_lease_mutation_fails_without_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, resident, case, context = _run_case_fixture(tmp_path)
    query = context(inputs, case)[3]
    calls = 0

    def lease(_inputs: Any, _resident: Any) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise producer.HarnessError("resident lease changed")
        return "lease"

    scan = lambda *args, **kwargs: _valid_report(
        query, order=kwargs["block_order"], block=kwargs["block_rows"],
    )
    with pytest.raises(producer.HarnessError, match="lease changed"):
        producer.run_case(
            inputs, {}, resident, case, scan=scan, lease=lease, context=context,
            full_validation=lambda *args: resident,
            binding_builder=_binding,
        )
    assert not any((inputs.output_root / "cases").iterdir())


def test_case_reverse_mismatch_fails_before_exact(
    tmp_path: Path,
) -> None:
    inputs, resident, case, context = _run_case_fixture(tmp_path)
    query = context(inputs, case)[3]

    def mismatched(*args: Any, **kwargs: Any) -> Any:
        return _valid_report(
            query, order=kwargs["block_order"], block=kwargs["block_rows"],
            rows_scanned=21 if kwargs["block_order"] == "reverse" else 0,
        )

    with pytest.raises(producer.HarnessError):
        producer.run_case(
            inputs, {}, resident, case, scan=mismatched,
            certified=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("exact must not run")
            ),
            lease=lambda *args: "lease", context=context,
            full_validation=lambda *args: resident,
            binding_builder=_binding,
        )
    assert not any((inputs.output_root / "cases").iterdir())


def test_timing_failure_still_seals_semantic_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, resident, case, context = _run_case_fixture(tmp_path)
    query = context(inputs, case)[3]
    result, certificate = _certified_result(case.query_id)
    monkeypatch.setattr(producer, "asdict", lambda value: certificate)

    def scan(*args: Any, **kwargs: Any) -> Any:
        elapsed = 121.0 if kwargs["block_order"] == "forward" else 61.0
        return _valid_report(
            query, order=kwargs["block_order"], block=kwargs["block_rows"],
            elapsed=elapsed,
        )

    payload = producer.run_case(
        inputs, {}, resident, case, scan=scan,
        certified=lambda *args, **kwargs: result,
        lease=lambda *args: "lease", context=context,
        full_validation=lambda *args: resident,
        binding_builder=_binding,
        clock=iter((0.0, 182.0, 183.0, 184.0)).__next__,
    )
    assert payload["certified"] is True
    assert payload["semantic_passed"] is True
    assert payload["performance_passed"] is False
    assert (inputs.output_root / "cases" / f"00-{case.query_id}.json").is_file()


def test_run_case_canonicalizes_dataclass_tuple_evidence(tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    inputs, resident, case, context = _run_case_fixture(tmp_path)
    query = context(inputs, case)[3]
    result, certificate = _certified_result(case.query_id)
    certificate["rounds"] = tuple(certificate["rounds"])
    certificate["threshold_closure_passes"] = tuple()
    monkeypatch.setattr(producer, "asdict", lambda value: certificate)
    scan = lambda *args, **kwargs: _valid_report(
        query, order=kwargs["block_order"], block=kwargs["block_rows"],
    )

    payload = producer.run_case(
        inputs, {}, resident, case, scan=scan,
        certified=lambda *args, **kwargs: result,
        lease=lambda *args: "lease", context=context,
        full_validation=lambda *args: resident,
        binding_builder=_binding,
        clock=iter((0.0, 1.0, 2.0, 3.0)).__next__,
    )

    assert type(payload["certificate"]["rounds"]) is list
    assert type(payload["certificate"]["threshold_closure_passes"]) is list


def test_existing_partial_root_is_terminal_and_never_resumed(tmp_path: Path) -> None:
    inputs, resident, _case, _context = _run_case_fixture(tmp_path)
    _partial_foundation(inputs, resident)
    # _run_case_fixture created the partial output/cases tree.
    with pytest.raises(producer.HarnessError, match="resume forbidden"):
        producer.produce(
            inputs, {}, resident, child_runner=lambda _command: 0,
            child_command=lambda ordinal: [str(ordinal)],
        )
    marker = producer._read_json(inputs.output_root / "INCOMPLETE.json")
    assert marker["status"] == "terminal_incomplete_new_root_required"


def test_prereg_git_boundary_binds_implementation_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "a.py").write_text("stable\n")
    monkeypatch.setattr(producer, "_runtime_files", lambda _repository: ("a.py",))
    monkeypatch.setattr(producer, "_sha", lambda _path: "a" * 64)

    def prereg_git(_repository: Path, *arguments: str) -> Any:
        stdout = "impl\n" if arguments[:2] == ("rev-parse", "HEAD") else ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(producer, "_run_git", prereg_git)
    frozen = producer._implementation_git(tmp_path)
    assert frozen["implementation_commit"] == "impl"

    def launch_git(_repository: Path, *arguments: str) -> Any:
        if arguments[:3] == ("log", "--diff-filter=A", "--format=%H"):
            stdout = "launch\n"
        elif arguments[:2] == ("rev-parse", "HEAD"):
            stdout = "launch\n"
        elif arguments[:2] == ("rev-parse", "launch^"):
            stdout = "impl\n"
        else:
            stdout = ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(producer, "_run_git", launch_git)
    assert producer._launch_git(tmp_path, frozen) == frozen


def test_prereg_git_boundary_rejects_later_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(producer, "_runtime_files", lambda _repository: ("a.py",))
    monkeypatch.setattr(producer, "_sha", lambda _path: "a" * 64)

    def run_git(_repository: Path, *arguments: str) -> Any:
        if arguments[:3] == ("log", "--diff-filter=A", "--format=%H"):
            stdout = "launch\n"
        elif arguments[:2] == ("rev-parse", "HEAD"):
            stdout = "later\n"
        else:
            stdout = ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(producer, "_run_git", run_git)
    with pytest.raises(producer.HarnessError, match="unique preregistration commit"):
        producer._launch_git(tmp_path, {})


def test_launch_git_rejects_untracked_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(producer, "_runtime_files", lambda _repository: ("a.py",))

    def run_git(_repository: Path, *arguments: str) -> Any:
        untracked = arguments[:2] == ("ls-files", "--error-unmatch") \
            and str(producer.DIAGNOSTIC_RELATIVE) in arguments
        return SimpleNamespace(
            returncode=1 if untracked else 0, stdout="", stderr="",
        )

    monkeypatch.setattr(producer, "_run_git", run_git)
    with pytest.raises(producer.HarnessError, match="clean tracked tree"):
        producer._launch_git(tmp_path, {})


def test_child_boundary_rejects_extra_entry(tmp_path: Path) -> None:
    inputs, resident, _case, _context = _run_case_fixture(tmp_path)
    prereg = {"preregistration_digest": inputs.prereg_digest}
    _write(inputs.output_root / "RUN_STARTED.json", {
        "schema_version": "m04r13-run-started-v1",
        "preregistration_digest": inputs.prereg_digest,
        "case_order": list(producer.FROZEN_QUERY_IDS), "parent_max_workers": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
    })
    _write(inputs.output_root / "CONTRACT.json", prereg)
    _write(inputs.output_root / "RESIDENT.json", resident)
    (inputs.output_root / "unexpected").write_text("x")
    with pytest.raises(producer.HarnessError, match="tree differs"):
        producer.validate_child_boundary(inputs, prereg, resident, 0)


def test_child_failure_is_serial_and_terminal(tmp_path: Path) -> None:
    inputs, resident, _case, _context = _run_case_fixture(tmp_path)
    # Producer itself requires the output to be absent.
    for child in list(inputs.output_root.rglob("*"))[::-1]:
        if child.is_dir():
            child.rmdir()
        else:
            child.unlink()
    inputs.output_root.rmdir()
    order: list[int] = []

    def run(command: Any) -> int:
        ordinal = int(command[0])
        order.append(ordinal)
        return 1 if ordinal == 1 else 0

    with pytest.raises(producer.HarnessError, match="case child failed: 1"):
        producer.produce(
            inputs, {"frozen": True}, resident, child_runner=run,
            child_command=lambda ordinal: [str(ordinal)],
        )
    assert order == [0, 1]
    assert (inputs.output_root / "INCOMPLETE.json").is_file()


def test_exact_worker_failure_never_writes_case(tmp_path: Path) -> None:
    inputs, resident, case, context = _run_case_fixture(tmp_path)
    query = context(inputs, case)[3]
    scan = lambda *args, **kwargs: _valid_report(
        query, order=kwargs["block_order"], block=kwargs["block_rows"],
    )
    with pytest.raises(RuntimeError, match="worker failed"):
        producer.run_case(
            inputs, {}, resident, case, scan=scan,
            certified=lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("worker failed")
            ),
            lease=lambda *args: "lease", context=context,
            full_validation=lambda *args: resident,
            binding_builder=_binding,
        )
    assert not any((inputs.output_root / "cases").iterdir())


def test_nonfinite_resource_measurement_fails_without_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, resident, case, context = _run_case_fixture(tmp_path)
    query = context(inputs, case)[3]
    result, certificate = _certified_result(case.query_id)
    monkeypatch.setattr(producer, "asdict", lambda value: certificate)
    monkeypatch.setattr(producer, "_rss", lambda: float("nan"))
    scan = lambda *args, **kwargs: _valid_report(
        query, order=kwargs["block_order"], block=kwargs["block_rows"],
    )
    with pytest.raises(producer.HarnessError, match="resource measurement invalid"):
        producer.run_case(
            inputs, {}, resident, case, scan=scan,
            certified=lambda *args, **kwargs: result,
            lease=lambda *args: "lease", context=context,
            full_validation=lambda *args: resident,
            binding_builder=_binding,
        )
    assert not any((inputs.output_root / "cases").iterdir())


def test_candidate_mismatch_seals_terminal_false_comparison(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    prereg, authorities = _producer_fixture(root)
    authorities[producer.FROZEN_QUERY_IDS[0]]["matches"][0]["symbol"] = "DIFFERENT"
    seal = comparator.compare(
        root, tmp_path / "truth", tmp_path / "verification.json",
        truth_loader=lambda *_args: authorities,
        **_truth_free_test_kwargs(root, prereg),
    )
    assert seal["passed"] is False
    comparison = producer._read_json(root / "COMPARISON.json")
    assert comparison["rows"][0]["semantic_passed"] is False


def test_truth_loader_failure_is_post_marker_terminal(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    prereg, _authorities = _producer_fixture(root)

    def fail(*_args: Any) -> Any:
        assert (root / "RESULTS_OPENED.json").is_file()
        raise comparator.ComparisonError("truth corrupt")

    with pytest.raises(comparator.ComparisonError, match="truth corrupt"):
        comparator.compare(
            root, tmp_path / "truth", tmp_path / "verification.json",
            truth_loader=fail, **_truth_free_test_kwargs(root, prereg),
        )
    assert (root / "RESULTS_OPENED.json").is_file()
    assert (root / "COMPARISON_INCOMPLETE.json").is_file()
    assert not (root / "COMPARISON.json").exists()


def test_rehashed_prereg_policy_mutation_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path
    config = tmp_path / "config.toml"
    config.write_text("[synthetic]\n")
    registry_root = tmp_path / "registry"
    source_root = tmp_path / "source"
    resident_root = tmp_path / "resident"
    output_root = repository / producer.OUTPUT_RELATIVE
    cases = tuple(producer.CaseInput(
        index, {"episode_id": query_id,
                "case_id": producer.FROZEN_CASE_IDS[index]},
    ) for index, query_id in enumerate(producer.FROZEN_QUERY_IDS))
    git = {"implementation_commit": "i", "files": {}, "files_digest": "f",
           "digest": "g"}
    resident_base = {
        "ready_digest": "r", "content_digest": "c", "seal_digest": "s",
        "ready_file_sha256": "f", "lease": {"lease_digest": "l"},
        "store_root": str(resident_root / "store"),
    }
    resident = {**resident_base, "identity_digest": stable_hash(resident_base)}
    deterministic = {
        "schema_version": producer.PREREG_SCHEMA, "status": "frozen_before_producer",
        "development_only": True, "truth_roots_allowed_in_producer": False,
        "production_promotion_authorized": False,
        "real_forward_outcomes_accessed": False, "git": git,
        "config_path": str(config.resolve()), "config_sha256": producer._sha(config),
        "roots": {
            "registry_root": str(registry_root.resolve()),
            "source_full_root": str(source_root.resolve()),
            "resident_root": str(resident_root.resolve()),
            "output_root": str(output_root.resolve()),
        },
        "generation_id": producer.GENERATION_ID, "provenance_digest": "p" * 64,
        "reserve_bytes": producer.RESIDENT_RESERVE_BYTES,
        "registry_digest": "registry", "query_ids": list(producer.FROZEN_QUERY_IDS),
        "registry_cases_digest": stable_hash([case.registry_case for case in cases]),
        "resident_content_digest": resident["content_digest"],
        "resident_ready_digest": resident["ready_digest"],
        "resident_identity_digest": resident["identity_digest"],
        "finite_threshold_diagnostic": {
            "path": str((repository / producer.DIAGNOSTIC_RELATIVE).resolve()),
            "sha256": "d" * 64, "result_digest": "e" * 64,
        },
        "execution": producer._execution_policy(),
    }
    prereg = {**deterministic, "preregistration_digest": stable_hash(deterministic)}
    mutated = json.loads(json.dumps(prereg))
    mutated["execution"]["initial_frontier_rows"] = 999
    mutated["preregistration_digest"] = stable_hash(producer._without(
        mutated, {"preregistration_digest"},
    ))
    monkeypatch.setattr(producer, "_read_json", lambda _path: mutated)
    monkeypatch.setattr(producer, "_launch_git", lambda _repository, _expected: git)
    monkeypatch.setattr(producer, "_validate_committed_diagnostic_blob", lambda *_: None)
    monkeypatch.setattr(
        producer, "_registry_cases", lambda _repository, _root: ("registry", cases),
    )
    monkeypatch.setattr(producer, "resident_full", lambda *args: resident)
    monkeypatch.setattr(
        producer, "finite_threshold_diagnostic_binding",
        lambda *_args, **_kwargs: prereg["finite_threshold_diagnostic"],
    )
    monkeypatch.setattr(
        producer, "diagnostic_query_bindings",
        lambda _inputs: [{} for _case in producer.FROZEN_QUERY_IDS],
    )
    with pytest.raises(producer.HarnessError, match="preregistration (schema )?differs"):
        producer.load_inputs(
            repository=repository, config_path=config, registry_root=registry_root,
            source_full_root=source_root, resident_root=resident_root,
            output_root=output_root,
        )


def test_load_inputs_accepts_exact_preregistration_with_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = REPOSITORY.resolve()
    config = (repository / producer.CONFIG_RELATIVE).resolve()
    registry_root = (repository / producer.REGISTRY_RELATIVE).resolve()
    source_root = (repository / producer.SOURCE_FULL_RELATIVE).resolve()
    resident_root = producer.RESIDENT_ROOT.resolve()
    output_root = (repository / producer.OUTPUT_RELATIVE).resolve()
    cases = tuple(producer.CaseInput(
        index, {"episode_id": query_id,
                "case_id": producer.FROZEN_CASE_IDS[index]},
    ) for index, query_id in enumerate(producer.FROZEN_QUERY_IDS))
    git_deterministic = {
        "implementation_commit": "0" * 40, "files": {},
        "files_digest": stable_hash({}),
    }
    git = {
        **git_deterministic, "digest": stable_hash(git_deterministic),
    }
    resident = {
        "content_digest": "a" * 64, "ready_digest": "b" * 64,
        "identity_digest": "c" * 64,
    }
    deterministic = {
        "schema_version": producer.PREREG_SCHEMA,
        "status": "frozen_before_producer", "development_only": True,
        "truth_roots_allowed_in_producer": False,
        "production_promotion_authorized": False,
        "real_forward_outcomes_accessed": False, "git": git,
        "config_path": str(config), "config_sha256": producer._sha(config),
        "roots": {
            "registry_root": str(registry_root),
            "source_full_root": str(source_root),
            "resident_root": str(resident_root),
            "output_root": str(output_root),
        },
        "generation_id": producer.GENERATION_ID,
        "provenance_digest": producer.PROVENANCE_DIGEST,
        "reserve_bytes": producer.RESIDENT_RESERVE_BYTES,
        "registry_digest": producer.REGISTRY_DIGEST,
        "query_ids": list(producer.FROZEN_QUERY_IDS),
        "registry_cases_digest": stable_hash(
            [case.registry_case for case in cases]
        ),
        "resident_content_digest": resident["content_digest"],
        "resident_ready_digest": resident["ready_digest"],
        "resident_identity_digest": resident["identity_digest"],
        "finite_threshold_diagnostic": {
            "path": str((repository / producer.DIAGNOSTIC_RELATIVE).resolve()),
            "sha256": "d" * 64, "result_digest": "e" * 64,
        },
        "execution": producer._execution_policy(),
        "environment": producer._environment_binding(),
    }
    prereg = {
        **deterministic, "preregistration_digest": stable_hash(deterministic),
    }
    monkeypatch.setattr(producer, "_read_json", lambda _path: prereg)
    monkeypatch.setattr(
        producer, "_launch_git", lambda _repository, _expected: git,
    )
    monkeypatch.setattr(producer, "_validate_committed_diagnostic_blob", lambda *_: None)
    monkeypatch.setattr(
        producer, "_registry_cases",
        lambda _repository, _root: (producer.REGISTRY_DIGEST, cases),
    )
    monkeypatch.setattr(producer, "resident_full", lambda *args: resident)
    monkeypatch.setattr(
        producer, "finite_threshold_diagnostic_binding",
        lambda *_args, **_kwargs: prereg["finite_threshold_diagnostic"],
    )

    monkeypatch.setattr(
        producer, "diagnostic_query_bindings",
        lambda _inputs: [{} for _case in producer.FROZEN_QUERY_IDS],
    )
    inputs, observed_prereg, observed_resident = producer.load_inputs(
        repository=repository, config_path=config, registry_root=registry_root,
        source_full_root=source_root, resident_root=resident_root,
        output_root=output_root,
    )
    assert observed_prereg == prereg
    assert observed_prereg["environment"] == producer._environment_binding()
    assert observed_resident == resident
    assert inputs.registry_digest == producer.REGISTRY_DIGEST


def test_registry_binding_uses_canonical_registry_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = [{"episode_id": query_id,
              "case_id": producer.FROZEN_CASE_IDS[index]}
             for index, query_id in enumerate(producer.FROZEN_QUERY_IDS)]
    monkeypatch.setattr(
        producer, "_m12", lambda _repository: SimpleNamespace(
            _validate_registry=lambda _registry: cases,
        ),
    )
    monkeypatch.setattr(
        producer, "_read_json", lambda _path: {
            "registry_digest": "canonical-registry", "cases": cases,
        },
    )
    digest, selected = producer._registry_cases(tmp_path, tmp_path / "registry")
    assert digest == "canonical-registry"
    assert tuple(case.query_id for case in selected) == producer.FROZEN_QUERY_IDS


def test_logical_round_certificate_digest_difference_is_diagnostic_only(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    prereg, authorities = _producer_fixture(root)
    authorities[producer.FROZEN_QUERY_IDS[0]]["certificate"]["result_digest"] = (
        "different-logical-round-digest"
    )
    seal = comparator.compare(
        root, tmp_path / "truth", tmp_path / "verification.json",
        truth_loader=lambda *_args: authorities,
        **_truth_free_test_kwargs(root, prereg),
    )
    comparison = producer._read_json(root / "COMPARISON.json")
    assert seal["passed"] is True
    assert comparison["rows"][0]["certificate_result_digest_equal_diagnostic"] is False


def test_comparator_git_gate_precedes_marker_and_truth(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    prereg, _authorities = _producer_fixture(root)
    truth_opened = False

    def truth(*_args: Any) -> Any:
        nonlocal truth_opened
        truth_opened = True
        return {}

    kwargs = _truth_free_test_kwargs(root, prereg)
    kwargs["git_validator"] = lambda *_args: (_ for _ in ()).throw(
        producer.HarnessError("git differs")
    )
    with pytest.raises(producer.HarnessError, match="git differs"):
        comparator.compare(
            root, tmp_path / "truth", tmp_path / "verification.json",
            truth_loader=truth, **kwargs,
        )
    assert truth_opened is False
    assert not (root / "RESULTS_OPENED.json").exists()


def test_comparator_query_binding_mismatch_precedes_truth(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    prereg, _authorities = _producer_fixture(root)
    bindings = _fixture_bindings(root)
    bindings[producer.FROZEN_QUERY_IDS[0]] = {
        **bindings[producer.FROZEN_QUERY_IDS[0]], "certified_input_digest": "changed",
    }
    truth_opened = False

    def truth(*_args: Any) -> Any:
        nonlocal truth_opened
        truth_opened = True
        return {}

    with pytest.raises(comparator.ComparisonError, match="semantic fields differ"):
        kwargs = _truth_free_test_kwargs(root, prereg)
        kwargs["binding_loader"] = lambda _prereg: bindings
        comparator.compare(
            root, tmp_path / "truth", tmp_path / "verification.json",
            truth_loader=truth, **kwargs,
        )
    assert truth_opened is False
    assert not (root / "RESULTS_OPENED.json").exists()


def test_authority_prefix_mismatch_is_terminal_false(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    prereg, authorities = _producer_fixture(root)
    authorities[producer.FROZEN_QUERY_IDS[0]]["query_stock_prefix"] = {
        "digest": "different"
    }
    seal = comparator.compare(
        root, tmp_path / "truth", tmp_path / "verification.json",
        truth_loader=lambda *_args: authorities,
        **_truth_free_test_kwargs(root, prereg),
    )
    assert seal["passed"] is False


def test_query_binding_is_json_canonical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, _resident, case, context = _run_case_fixture(tmp_path)
    source, episode, _request, packed = context(inputs, case)
    request = SimpleNamespace(
        search_datasets=("nasdaq",), quality_tiers=("A", "B"), top_k=20,
        cross_dataset=False, deduplicate_overlaps=True, max_per_instrument=3,
        minimum_history_gap_bars=60,
    )
    prefix = CausalPrefixDigest(
        "canonical-ohlcv-prefix-v1", "2020-01-01", "2020-01-01", 1, "d",
    )
    monkeypatch.setattr(producer, "causal_prefix_digest", lambda *_args: prefix)
    monkeypatch.setattr(producer, "representation_input_digest", lambda _value: "rep")
    source = SimpleNamespace(load_benchmark=lambda: object(), load=lambda _key: object())
    value = producer.query_binding(source, episode, request, packed, "p" * 64)
    assert value["request"]["search_datasets"] == ["nasdaq"]
    assert value["request"]["quality_tiers"] == ["A", "B"]
    assert json.loads(json.dumps(value, allow_nan=False)) == value


@pytest.mark.parametrize("bad_exact", [-1, True])
def test_comparator_rejects_rehashed_invalid_accounting_before_truth(
    tmp_path: Path, bad_exact: Any,
) -> None:
    root = tmp_path / "evidence"
    prereg, _authorities = _producer_fixture(root)
    query_id = producer.FROZEN_QUERY_IDS[0]
    path = root / "cases" / f"00-{query_id}.json"
    case = producer._read_json(path)
    certificate = case["certificate"]
    certificate["exact_evaluated"] = bad_exact
    certificate["safely_pruned"] = 40 - bad_exact
    certificate["native_bound_accounting"] = {
        "native_bound_evaluated": bad_exact,
        "exact_dtw_evaluated": bad_exact,
        "native_bound_pruned": 0, "packed_bound_pruned": 40 - bad_exact,
    }
    certificate["rounds"][0]["exact_rows"] = bad_exact
    certificate["result_digest"] = _rehash_certificate(certificate, case["matches"])
    _rewrite_case_and_seal(root, case)
    truth_opened = False

    def truth(*_args: Any) -> Any:
        nonlocal truth_opened
        truth_opened = True
        return {}

    with pytest.raises(comparator.ComparisonError):
        comparator.compare(
            root, tmp_path / "truth", tmp_path / "verification.json",
            truth_loader=truth, **_truth_free_test_kwargs(root, prereg),
        )
    assert truth_opened is False
    assert not (root / "RESULTS_OPENED.json").exists()


@pytest.mark.parametrize("raw", ['{"value":1,"value":2}', '{"value":NaN}'])
def test_strict_json_rejects_duplicate_keys_and_nonfinite(
    tmp_path: Path, raw: str,
) -> None:
    path = tmp_path / "bad.json"
    path.write_text(raw)
    with pytest.raises(producer.HarnessError):
        producer._read_json(path)


def test_rss_uses_process_high_water(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        producer.resource, "getrusage",
        lambda _who: SimpleNamespace(ru_maxrss=2_048 * 1_024),
    )
    assert producer._rss() >= 2_048.0


def test_exact_prereg_schema_rejects_rehashed_extra_and_limit_drift(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    prereg, _authorities = _producer_fixture(root)
    producer.validate_preregistration_shape(prereg, enforce_topology=False)
    for mutate in (
        lambda value: value.update({"unexpected": True}),
        lambda value: value["execution"]["performance_limits"].update(
            {"process_rss_mb": 1_535.0}
        ),
    ):
        changed = json.loads(json.dumps(prereg))
        mutate(changed)
        changed["preregistration_digest"] = stable_hash(producer._without(
            changed, {"preregistration_digest"},
        ))
        with pytest.raises(producer.HarnessError, match="schema differs"):
            producer.validate_preregistration_shape(changed, enforce_topology=False)


def test_producer_incomplete_is_digest_bound(tmp_path: Path) -> None:
    inputs, resident, _case, _context = _run_case_fixture(tmp_path)
    _partial_foundation(inputs, resident)
    with pytest.raises(producer.HarnessError):
        producer.produce(
            inputs, {}, resident, child_runner=lambda _command: 0,
            child_command=lambda ordinal: [str(ordinal)],
        )
    payload = producer._read_json(inputs.output_root / "INCOMPLETE.json")
    assert payload["result_digest"] == stable_hash(producer._without(
        payload, {"created_at", "result_digest"},
    ))
    assert payload["partial_tree_digest"] == producer._tree_snapshot_digest(
        inputs.output_root, omitted={"INCOMPLETE.json"},
    )


def test_comparison_incomplete_is_digest_bound(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    prereg, _authorities = _producer_fixture(root)
    with pytest.raises(RuntimeError, match="truth failure"):
        comparator.compare(
            root, tmp_path / "truth", tmp_path / "verification.json",
            truth_loader=lambda *_args: (_ for _ in ()).throw(
                RuntimeError("truth failure")
            ),
            **_truth_free_test_kwargs(root, prereg),
        )
    payload = producer._read_json(root / "COMPARISON_INCOMPLETE.json")
    assert payload["result_digest"] == stable_hash(producer._without(
        payload, {"created_at", "result_digest"},
    ))
    assert payload["partial_tree_digest"] == producer._tree_snapshot_digest(
        root, omitted={"COMPARISON_INCOMPLETE.json"},
    )


def test_semantic_pass_performance_fail_is_terminal_and_not_passed(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    prereg, authorities = _producer_fixture(root)
    query_id = producer.FROZEN_QUERY_IDS[0]
    case = producer._read_json(root / "cases" / f"00-{query_id}.json")
    case["forward_proposal"]["elapsed_seconds"] = 121.0
    case["metrics"]["forward_proposal_seconds"] = 121.0
    case["metrics"]["case_task_wall_seconds"] = 123.0
    case["performance_passed"] = False
    _rewrite_case_and_seal(root, case)
    producer_seal = producer._read_json(root / "PRODUCER_SEALED.json")
    producer_seal["performance_passed"] = False
    producer_seal["seal_digest"] = stable_hash(producer._without(
        producer_seal, {"created_at", "seal_digest"},
    ))
    _write(root / "PRODUCER_SEALED.json", producer_seal)
    seal = comparator.compare(
        root, tmp_path / "truth", tmp_path / "verification.json",
        truth_loader=lambda *_args: authorities,
        **_truth_free_test_kwargs(root, prereg),
    )
    assert seal["semantic_passed"] is True
    assert seal["performance_passed"] is False
    assert seal["passed"] is False


def test_terminal_comparison_recursive_tree_rejects_extra(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    prereg, authorities = _producer_fixture(root)
    comparator.compare(
        root, tmp_path / "truth", tmp_path / "verification.json",
        truth_loader=lambda *_args: authorities,
        **_truth_free_test_kwargs(root, prereg),
    )
    (root / "unexpected.txt").write_text("x")
    with pytest.raises(comparator.ComparisonError, match="tree differs"):
        comparator.validate_terminal_comparison(
            root, tmp_path / "truth", tmp_path / "verification.json",
            **_truth_free_test_kwargs(root, prereg),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (("rows_scanned", "40"), ("eligible_rows", True),
     ("overflow_fallback", "false")),
)
def test_proposal_json_types_never_coerce_numeric_strings_or_bools(
    field: str, value: Any,
) -> None:
    payload = _proposal(producer.FROZEN_QUERY_IDS[0], 4_096, "forward")
    if field == "overflow_fallback":
        payload["candidates"][0][field] = value
    else:
        payload[field] = value
    with pytest.raises(producer.HarnessError, match="proposal .*JSON types differ"):
        producer._proposal_report(payload)


@pytest.mark.parametrize("failure", ("nested_timing", "fallback_flag"))
def test_comparator_rejects_impossible_timing_and_fallback_before_truth(
    tmp_path: Path, failure: str,
) -> None:
    root = tmp_path / "evidence"
    prereg, _authorities = _producer_fixture(root)
    query_id = producer.FROZEN_QUERY_IDS[0]
    path = root / "cases" / f"00-{query_id}.json"
    case = producer._read_json(path)
    if failure == "nested_timing":
        case["metrics"]["case_task_wall_seconds"] = 1.1
    else:
        assert case["certificate"]["threshold_closure_passes"] == []
        case["streaming_fallback_used"] = True
    _rewrite_case_and_seal(root, case)
    truth_opened = False

    def truth(*_args: Any) -> Any:
        nonlocal truth_opened
        truth_opened = True
        return {}

    with pytest.raises(comparator.ComparisonError):
        comparator.compare(
            root, tmp_path / "truth", tmp_path / "verification.json",
            truth_loader=truth, **_truth_free_test_kwargs(root, prereg),
        )
    assert truth_opened is False
    assert not (root / "RESULTS_OPENED.json").exists()


def test_post_marker_non_loader_exception_terminalizes_and_refuses_reopen(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    prereg, authorities = _producer_fixture(root)
    malformed = dict(authorities)
    malformed.pop(producer.FROZEN_QUERY_IDS[-1])
    with pytest.raises(KeyError):
        comparator.compare(
            root, tmp_path / "truth", tmp_path / "verification.json",
            truth_loader=lambda *_args: malformed,
            **_truth_free_test_kwargs(root, prereg),
        )
    marker = producer._read_json(root / "RESULTS_OPENED.json")
    incomplete = producer._read_json(root / "COMPARISON_INCOMPLETE.json")
    assert incomplete["results_opened_digest"] == marker["result_digest"]
    assert incomplete["error_type"] == "KeyError"
    reopened_truth = False

    def forbidden(*_args: Any) -> Any:
        nonlocal reopened_truth
        reopened_truth = True
        return authorities

    with pytest.raises(comparator.ComparisonError, match="tree differs"):
        comparator.compare(
            root, tmp_path / "truth", tmp_path / "verification.json",
            truth_loader=forbidden, **_truth_free_test_kwargs(root, prereg),
        )
    assert reopened_truth is False


@pytest.mark.parametrize("partial_state", ("empty", "foundation", "sealed"))
def test_producer_incomplete_accepts_only_exact_prefix_states(
    tmp_path: Path, partial_state: str,
) -> None:
    cases = tuple(producer.CaseInput(
        index, {"episode_id": query_id, "case_id": f"case-{index}"},
    ) for index, query_id in enumerate(producer.FROZEN_QUERY_IDS))
    root = tmp_path / partial_state
    inputs = producer.Inputs(
        tmp_path, tmp_path / "config", tmp_path / "registry",
        tmp_path / "source/store", tmp_path / "resident", root,
        producer.GENERATION_ID, producer.PROVENANCE_DIGEST,
        producer.RESIDENT_RESERVE_BYTES, producer.REGISTRY_DIGEST, cases,
        "p" * 64,
    )
    root.mkdir()
    if partial_state != "empty":
        (root / "cases").mkdir()
        _write(root / "RUN_STARTED.json", {"prefix": 0})
        _write(root / "CONTRACT.json", {"prefix": 1})
    if partial_state == "sealed":
        _write(root / "RESIDENT.json", {"prefix": 2})
        for case in cases:
            _write(root / "cases" / f"{case.ordinal:02d}-{case.query_id}.json", {})
        _write(root / "PRODUCER_SEALED.json", {"sealed": True})
    payload = producer._write_incomplete(inputs, "SyntheticFailure")
    assert payload["status"] == "terminal_incomplete_new_root_required"
    assert payload["partial_tree_digest"] == producer._tree_snapshot_digest(
        root, omitted={"INCOMPLETE.json"},
    )
    producer._validate_incomplete_prefix(inputs, include_marker=True)


def test_terminal_validator_rejects_rehashed_garbage_status_and_recursive_case(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    prereg, authorities = _producer_fixture(root)
    comparator.compare(
        root, tmp_path / "truth", tmp_path / "verification.json",
        truth_loader=lambda *_args: authorities,
        **_truth_free_test_kwargs(root, prereg),
    )
    comparison_path = root / "COMPARISON.json"
    comparison = producer._read_json(comparison_path)
    comparison["status"] = "garbage_but_rehashed"
    comparison["result_digest"] = stable_hash(producer._without(
        comparison, {"created_at", "result_digest"},
    ))
    _write(comparison_path, comparison)
    seal_path = root / "COMPARISON_SEALED.json"
    seal = producer._read_json(seal_path)
    seal["comparison_result_digest"] = comparison["result_digest"]
    seal["seal_digest"] = stable_hash(producer._without(
        seal, {"created_at", "seal_digest"},
    ))
    _write(seal_path, seal)
    with pytest.raises(comparator.ComparisonError, match="comparison reconstruction"):
        comparator.validate_terminal_comparison(
            root, tmp_path / "truth", tmp_path / "verification.json",
            truth_loader=lambda *_args: authorities,
            **_truth_free_test_kwargs(root, prereg),
        )

    # Restore a valid terminal comparison, then corrupt and rehash a producer case.
    _write(comparison_path, {
        **comparison, "status": "comparison_complete",
        "result_digest": stable_hash(producer._without(
            {**comparison, "status": "comparison_complete"},
            {"created_at", "result_digest"},
        )),
    })
    comparison = producer._read_json(comparison_path)
    seal["comparison_result_digest"] = comparison["result_digest"]
    seal["seal_digest"] = stable_hash(producer._without(
        seal, {"created_at", "seal_digest"},
    ))
    _write(seal_path, seal)
    query_id = producer.FROZEN_QUERY_IDS[0]
    case_path = root / "cases" / f"00-{query_id}.json"
    case = producer._read_json(case_path)
    case["status"] = "garbage_but_rehashed"
    _rewrite_case_and_seal(root, case)
    with pytest.raises(comparator.ComparisonError, match="case semantic fields differ"):
        comparator.validate_terminal_comparison(
            root, tmp_path / "truth", tmp_path / "verification.json",
            truth_loader=lambda *_args: authorities,
            **_truth_free_test_kwargs(root, prereg),
        )


@pytest.mark.parametrize(
    "drift", ("registry", "provenance", "root", "environment"),
)
def test_prereg_topology_rejects_fully_rehashed_frozen_drift(
    tmp_path: Path, drift: str,
) -> None:
    fixture_root = tmp_path / "fixture"
    prereg, _authorities = _producer_fixture(fixture_root)
    repository = PRODUCER_PATH.resolve().parents[2]
    prereg["config_path"] = str((repository / producer.CONFIG_RELATIVE).resolve())
    prereg["roots"] = {
        "registry_root": str((repository / producer.REGISTRY_RELATIVE).resolve()),
        "source_full_root": str(
            (repository / producer.SOURCE_FULL_RELATIVE).resolve()
        ),
        "resident_root": str(producer.RESIDENT_ROOT.resolve()),
        "output_root": str((repository / producer.OUTPUT_RELATIVE).resolve()),
    }
    prereg["registry_digest"] = producer.REGISTRY_DIGEST
    prereg["provenance_digest"] = producer.PROVENANCE_DIGEST
    prereg["environment"] = producer._environment_binding()
    prereg["finite_threshold_diagnostic"]["path"] = str(
        (repository / producer.DIAGNOSTIC_RELATIVE).resolve()
    )
    prereg["preregistration_digest"] = stable_hash(producer._without(
        prereg, {"preregistration_digest"},
    ))
    producer.validate_preregistration_shape(prereg)

    if drift == "registry":
        prereg["registry_digest"] = "a" * 64
    elif drift == "provenance":
        prereg["provenance_digest"] = "b" * 64
    elif drift == "root":
        prereg["roots"]["output_root"] = str((tmp_path / "different").resolve())
    else:
        prereg["environment"]["python_version"] = "0.0.0"
        prereg["environment"]["digest"] = stable_hash(producer._without(
            prereg["environment"], {"digest"},
        ))
    prereg["preregistration_digest"] = stable_hash(producer._without(
        prereg, {"preregistration_digest"},
    ))
    with pytest.raises(producer.HarnessError):
        producer.validate_preregistration_shape(prereg)


@pytest.mark.parametrize("field", ("episode_id", "symbol", "cutoff", "quality_tier"))
def test_comparator_cross_binds_match_identity_metadata_before_truth(
    tmp_path: Path, field: str,
) -> None:
    root = tmp_path / "evidence"
    prereg, _authorities = _producer_fixture(root)
    query_id = producer.FROZEN_QUERY_IDS[0]
    path = root / "cases" / f"00-{query_id}.json"
    case = producer._read_json(path)
    row = case["matches"][-1]
    row[field] = {
        "episode_id": "f" * 24,
        "symbol": "WRONG",
        "cutoff": "2018-12-31T00:00:00+00:00",
        "quality_tier": "B",
    }[field]
    case["certificate"]["result_digest"] = _rehash_certificate(
        case["certificate"], case["matches"],
    )
    _rewrite_case_and_seal(root, case)
    truth_opened = False

    def truth(*_args: Any) -> Any:
        nonlocal truth_opened
        truth_opened = True
        return {}

    with pytest.raises(comparator.ComparisonError):
        comparator.compare(
            root, tmp_path / "truth", tmp_path / "verification.json",
            truth_loader=truth, **_truth_free_test_kwargs(root, prereg),
        )
    assert truth_opened is False
    assert not (root / "RESULTS_OPENED.json").exists()


def test_comparator_reconstructs_round_proposal_prefix_digest_before_truth(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    prereg, _authorities = _producer_fixture(root)
    query_id = producer.FROZEN_QUERY_IDS[0]
    path = root / "cases" / f"00-{query_id}.json"
    case = producer._read_json(path)
    assert case["certificate"]["rounds"][0]["proposal_digest"] != "f" * 64
    case["certificate"]["rounds"][0]["proposal_digest"] = "f" * 64
    case["rounds"] = case["certificate"]["rounds"]
    case["certificate"]["result_digest"] = _rehash_certificate(
        case["certificate"], case["matches"],
    )
    _rewrite_case_and_seal(root, case)
    with pytest.raises(
        comparator.ComparisonError, match="certified proposal cross-binding differs",
    ):
        comparator.validate_producer_before_truth(
            root, **_truth_free_test_kwargs(root, prereg),
        )


def test_comparator_binds_exact_frozen_registry_case_before_truth(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    prereg, _authorities = _producer_fixture(root)
    query_id = producer.FROZEN_QUERY_IDS[0]
    path = root / "cases" / f"00-{query_id}.json"
    case = producer._read_json(path)
    case["registry_case_id"] = "nasdaq-WRONG-current-252"
    _rewrite_case_and_seal(root, case)
    with pytest.raises(comparator.ComparisonError, match="case semantic fields differ"):
        comparator.validate_producer_before_truth(
            root, **_truth_free_test_kwargs(root, prereg),
        )


@pytest.mark.parametrize("corruption", ("self_consistent_flags", "authority_truth"))
def test_terminal_comparison_reconstructs_rows_from_reloaded_truth(
    tmp_path: Path, corruption: str,
) -> None:
    root = tmp_path / "evidence"
    prereg, authorities = _producer_fixture(root)
    truth = lambda *_args: authorities
    comparator.compare(
        root, tmp_path / "truth", tmp_path / "verification.json",
        truth_loader=truth, **_truth_free_test_kwargs(root, prereg),
    )
    assert comparator.validate_terminal_comparison(
        root, tmp_path / "truth", tmp_path / "verification.json",
        truth_loader=truth, **_truth_free_test_kwargs(root, prereg),
    )["passed"] is True
    if corruption == "authority_truth":
        authorities[producer.FROZEN_QUERY_IDS[0]]["matches"][0]["symbol"] = "CHANGED"
    else:
        comparison_path = root / "COMPARISON.json"
        comparison = producer._read_json(comparison_path)
        comparison["rows"][0]["matches_equal"] = False
        comparison["rows"][0]["semantic_passed"] = False
        comparison["semantic_passed"] = False
        comparison["passed"] = False
        comparison["result_digest"] = stable_hash(producer._without(
            comparison, {"created_at", "result_digest"},
        ))
        _write(comparison_path, comparison)
        seal_path = root / "COMPARISON_SEALED.json"
        seal = producer._read_json(seal_path)
        seal["comparison_result_digest"] = comparison["result_digest"]
        seal["semantic_passed"] = False
        seal["passed"] = False
        seal["seal_digest"] = stable_hash(producer._without(
            seal, {"created_at", "seal_digest"},
        ))
        _write(seal_path, seal)
    with pytest.raises(comparator.ComparisonError, match="comparison reconstruction"):
        comparator.validate_terminal_comparison(
            root, tmp_path / "truth", tmp_path / "verification.json",
            truth_loader=truth, **_truth_free_test_kwargs(root, prereg),
        )


@pytest.mark.parametrize(
    ("failed_publication", "expected_prefix"),
    (
        ("RUN_STARTED.json", set()),
        ("CONTRACT.json", {"RUN_STARTED.json"}),
        ("RESIDENT.json", {"RUN_STARTED.json", "CONTRACT.json"}),
    ),
)
def test_initial_publication_failure_terminalizes_exact_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    failed_publication: str, expected_prefix: set[str],
) -> None:
    inputs, resident, _case, _context = _run_case_fixture(tmp_path)
    for child in sorted(inputs.output_root.rglob("*"), reverse=True):
        if child.is_dir():
            child.rmdir()
        else:
            child.unlink()
    inputs.output_root.rmdir()
    original_atomic = producer._atomic
    failed = False

    def fail_once(path: Path, payload: Any) -> None:
        nonlocal failed
        if path.name == failed_publication and not failed:
            failed = True
            raise OSError("injected publication failure")
        original_atomic(path, payload)

    monkeypatch.setattr(producer, "_atomic", fail_once)
    with pytest.raises(OSError, match="injected publication failure"):
        producer.produce(
            inputs, {"frozen": True}, resident,
            child_runner=lambda _command: 0,
            child_command=lambda ordinal: [str(ordinal)],
        )
    observed = {
        str(path.relative_to(inputs.output_root))
        for path in inputs.output_root.rglob("*") if path.is_file()
    }
    assert observed == expected_prefix | {"INCOMPLETE.json"}
    producer._validate_incomplete_prefix(inputs, include_marker=True)


def test_comparator_cli_revalidates_existing_terminal_with_authorities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "COMPARISON_SEALED.json").write_text("{}")
    authority = tmp_path / "authority"
    verification = tmp_path / "verification.json"
    observed: list[tuple[Path, Path, Path]] = []
    monkeypatch.setattr(
        comparator, "validate_terminal_comparison",
        lambda producer_root, authority_root, verification_path: (
            observed.append((producer_root, authority_root, verification_path))
            or {"passed": True}
        ),
    )
    monkeypatch.setattr(
        comparator, "compare",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("terminal must be revalidated, not compared again")
        ),
    )
    monkeypatch.setattr(sys, "argv", [
        str(COMPARATOR_PATH), "--producer-root", str(root),
        "--authority-root", str(authority),
        "--authority-verification", str(verification),
    ])
    assert comparator.main() == 0
    assert observed == [(root, authority, verification)]


def _synthetic_fallback_certificate(
    tmp_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = tmp_path / "fallback-fixture"
    _prereg, _authorities = _producer_fixture(root)
    query_id = producer.FROZEN_QUERY_IDS[0]
    case = producer._read_json(root / "cases" / f"00-{query_id}.json")
    certificate = case["certificate"]
    matches = case["matches"]
    certificate["eligible_candidates"] = 20_000
    certificate["exact_evaluated"] = 25
    certificate["safely_pruned"] = 19_975
    certificate["stopped_early"] = True
    certificate["next_lower_bound"] = 20.0
    certificate["minimum_native_pruned_bound"] = 21.0
    certificate["native_bound_accounting"] = {
        "native_bound_evaluated": 16_394,
        "exact_dtw_evaluated": 25,
        "native_bound_pruned": 16_369,
        "packed_bound_pruned": 3_606,
    }
    certificate["rounds"] = [{
        "frontier_rows": frontier, "exact_rows": 20,
        "next_lower_bound": 20.0, "constrained_threshold": 19.0,
        "selected_rows": 18, "certified": False,
        "proposal_digest": "a" * 64,
    } for frontier in producer.LOGICAL_FRONTIERS]
    first_upper = 19.0 + producer.TOLERANCE
    first_result = 19.5
    certificate["threshold_closure_passes"] = [
        {
            "lower_exclusive": None, "upper_inclusive": first_upper,
            "admitted_rows": 5,
            "cumulative_native_bound_evaluated": 16_389,
            "cumulative_exact_dtw_evaluated": 22,
            "selected_rows": 19, "resulting_threshold": first_result,
            "minimum_packed_unclassified_bound": 19.25,
            "minimum_native_pruned_bound": 19.4,
            "excluded_prefix_digest": "b" * 64,
            "admitted_set_digest": "c" * 64,
            "scan_result_digest": "d" * 64, "certified": False,
        },
        {
            "lower_exclusive": first_upper,
            "upper_inclusive": first_result + producer.TOLERANCE,
            "admitted_rows": 5,
            "cumulative_native_bound_evaluated": 16_394,
            "cumulative_exact_dtw_evaluated": 25,
            "selected_rows": 20, "resulting_threshold": 19.0,
            "minimum_packed_unclassified_bound": 20.0,
            "minimum_native_pruned_bound": 21.0,
            "excluded_prefix_digest": "e" * 64,
            "admitted_set_digest": "f" * 64,
            "scan_result_digest": "1" * 64, "certified": True,
        },
    ]
    certificate["result_digest"] = _rehash_certificate(certificate, matches)
    return certificate, matches


def test_genuine_synthetic_streaming_fallback_certificate_is_accepted(
    tmp_path: Path,
) -> None:
    certificate, matches = _synthetic_fallback_certificate(tmp_path)
    producer.validate_certificate_and_matches(
        certificate, matches, producer.FROZEN_QUERY_IDS[0],
    )


def test_constrained_selected_rows_may_decrease_between_states(
    tmp_path: Path,
) -> None:
    certificate, matches = _synthetic_fallback_certificate(tmp_path)
    certificate["rounds"][0]["selected_rows"] = 20
    certificate["rounds"][1]["selected_rows"] = 17
    first, final = certificate["threshold_closure_passes"]
    first["admitted_rows"] = 3
    first["cumulative_native_bound_evaluated"] = 16_387
    first["cumulative_exact_dtw_evaluated"] = 21
    middle = {
        **first,
        "lower_exclusive": first["upper_inclusive"],
        "upper_inclusive": first["resulting_threshold"] + producer.TOLERANCE,
        "admitted_rows": 2,
        "cumulative_native_bound_evaluated": 16_389,
        "cumulative_exact_dtw_evaluated": 22,
        "selected_rows": 18,
        "resulting_threshold": 20.0,
        "excluded_prefix_digest": "2" * 64,
        "admitted_set_digest": "3" * 64,
        "scan_result_digest": "4" * 64,
    }
    first["selected_rows"] = 19
    final["lower_exclusive"] = middle["upper_inclusive"]
    final["upper_inclusive"] = middle["resulting_threshold"] + producer.TOLERANCE
    certificate["threshold_closure_passes"] = [first, middle, final]
    certificate["result_digest"] = _rehash_certificate(certificate, matches)
    producer.validate_certificate_and_matches(
        certificate, matches, producer.FROZEN_QUERY_IDS[0],
    )


def test_repeated_max_frontier_with_changing_native_minimum_is_accepted(
    tmp_path: Path,
) -> None:
    certificate, matches = _synthetic_fallback_certificate(tmp_path)
    certificate["rounds"][-1]["next_lower_bound"] = 1.12949
    certificate["rounds"].append({
        **certificate["rounds"][-1], "exact_rows": 21,
        "next_lower_bound": 1.07794,
    })
    certificate["result_digest"] = _rehash_certificate(certificate, matches)
    producer.validate_certificate_and_matches(
        certificate, matches, producer.FROZEN_QUERY_IDS[0],
    )


def test_real_source_naive_match_cutoffs_are_canonical_and_accepted(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    _prereg, _authorities = _producer_fixture(root)
    query_id = producer.FROZEN_QUERY_IDS[0]
    case = producer._read_json(root / "cases" / f"00-{query_id}.json")
    for match in case["matches"]:
        match["cutoff"] = pd.Timestamp(match["cutoff"]).tz_localize(None).isoformat()
    producer.validate_certificate_and_matches(
        case["certificate"], case["matches"], query_id,
    )
    case["matches"][0]["cutoff"] = "2019-01-01T05:30:00+05:30"
    with pytest.raises(producer.HarnessError, match="match raw JSON schema differs"):
        producer.validate_certificate_and_matches(
            case["certificate"], case["matches"], query_id,
        )


def test_zero_native_pruned_rejects_forged_native_minimum_and_final_next(
    tmp_path: Path,
) -> None:
    root = tmp_path / "native-accounting"
    _prereg, _authorities = _producer_fixture(root)
    query_id = producer.FROZEN_QUERY_IDS[0]
    case = producer._read_json(root / "cases" / f"00-{query_id}.json")
    certificate = case["certificate"]
    certificate["minimum_native_pruned_bound"] = 20.0
    certificate["next_lower_bound"] = 20.0
    certificate["stopped_early"] = True
    certificate["rounds"][-1]["next_lower_bound"] = 20.0
    certificate["result_digest"] = _rehash_certificate(
        certificate, case["matches"],
    )
    with pytest.raises(producer.HarnessError, match="reconstruction differs"):
        producer.validate_certificate_and_matches(
            certificate, case["matches"], query_id,
        )


@pytest.mark.parametrize(
    "invalid_band", ("first_lower", "first_upper", "later_no_progress"),
)
def test_rehashed_closure_admission_band_fails_closed(
    tmp_path: Path, invalid_band: str,
) -> None:
    certificate, matches = _synthetic_fallback_certificate(tmp_path)
    closures = certificate["threshold_closure_passes"]
    if invalid_band == "first_lower":
        closures[0]["lower_exclusive"] = 0.0
    elif invalid_band == "first_upper":
        closures[0]["upper_inclusive"] += 0.1
        closures[1]["lower_exclusive"] = closures[0]["upper_inclusive"]
    else:
        closures[0]["resulting_threshold"] = 19.0
        closures[1]["upper_inclusive"] = 19.0 + producer.TOLERANCE
    certificate["result_digest"] = _rehash_certificate(certificate, matches)
    with pytest.raises(producer.HarnessError, match="differs"):
        producer.validate_certificate_and_matches(
            certificate, matches, producer.FROZEN_QUERY_IDS[0],
        )


def test_rehashed_round_cannot_continue_after_frontier_covers_eligible(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    prereg, _authorities = _producer_fixture(root)
    query_id = producer.FROZEN_QUERY_IDS[0]
    path = root / "cases" / f"00-{query_id}.json"
    case = producer._read_json(path)
    first = case["certificate"]["rounds"][0]
    first["certified"] = False
    case["certificate"]["rounds"].append({**first, "certified": True})
    case["rounds"] = case["certificate"]["rounds"]
    case["certificate"]["result_digest"] = _rehash_certificate(
        case["certificate"], case["matches"],
    )
    _rewrite_case_and_seal(root, case)
    with pytest.raises(comparator.ComparisonError, match="certificate/match"):
        comparator.validate_producer_before_truth(
            root, **_truth_free_test_kwargs(root, prereg),
        )


def test_rehashed_closure_cannot_follow_a_certified_round(tmp_path: Path) -> None:
    certificate, matches = _synthetic_fallback_certificate(tmp_path)
    certificate["rounds"][-1]["certified"] = True
    certificate["result_digest"] = _rehash_certificate(certificate, matches)
    with pytest.raises(producer.HarnessError, match="reconstruction differs"):
        producer.validate_certificate_and_matches(
            certificate, matches, producer.FROZEN_QUERY_IDS[0],
        )


@pytest.mark.parametrize("invalid_bound", ("round_next", "stop_threshold"))
def test_rehashed_certificate_bound_reconstruction_fails_closed(
    tmp_path: Path, invalid_bound: str,
) -> None:
    root = tmp_path / "evidence"
    prereg, _authorities = _producer_fixture(root)
    query_id = producer.FROZEN_QUERY_IDS[0]
    path = root / "cases" / f"00-{query_id}.json"
    case = producer._read_json(path)
    certificate = case["certificate"]
    if invalid_bound == "round_next":
        certificate["rounds"][0]["next_lower_bound"] = 20.0
        certificate["next_lower_bound"] = 20.0
        certificate["minimum_native_pruned_bound"] = 21.0
        certificate["stopped_early"] = True
    else:
        certificate["stop_threshold"] = 20.0
        certificate["rounds"][0]["constrained_threshold"] = 20.0
    case["rounds"] = certificate["rounds"]
    certificate["result_digest"] = _rehash_certificate(certificate, case["matches"])
    _rewrite_case_and_seal(root, case)
    with pytest.raises(comparator.ComparisonError):
        comparator.validate_producer_before_truth(
            root, **_truth_free_test_kwargs(root, prereg),
        )


def test_round_effective_next_cannot_exceed_packed_boundary() -> None:
    payload = _proposal(producer.FROZEN_QUERY_IDS[0], 4_096, "forward")
    report = producer._proposal_report(payload)
    frontier = 20
    certificate = {
        "rounds": [{
            "frontier_rows": frontier,
            "next_lower_bound": report.candidates[frontier].lower_bound + 0.1,
            "proposal_digest": producer.bound_proposal_candidate_digest(
                report.candidates[:frontier + 1],
            ),
        }],
        "threshold_closure_passes": [{}],
    }
    with pytest.raises(
        comparator.ComparisonError, match="proposal cross-binding differs",
    ):
        comparator._validate_round_proposal_bindings(certificate, report)


def test_closure_rescued_match_authenticates_outside_proposal_prefix() -> None:
    report = producer._proposal_report(
        _proposal(producer.FROZEN_QUERY_IDS[0], 4_096, "forward")
    )
    outside_id = "f" * 24
    match = {
        "episode_id": outside_id, "symbol": "RESCUED",
        "cutoff": "2018-01-01T00:00:00+00:00", "quality_tier": "B",
    }
    universe = {outside_id: {
        "episode_id": outside_id, "symbol": "RESCUED",
        "cutoff_ns": pd.Timestamp(match["cutoff"]).value,
        "quality_tier": "B", "branch_aware_bound": 0.5,
        "overflow": False,
    }}
    closures = [{"lower_exclusive": None, "upper_inclusive": 1.0}]
    comparator._validate_match_universe_cross_binding(
        [match], report, closures, universe,
    )
    with pytest.raises(comparator.ComparisonError, match="outside proposal"):
        comparator._validate_match_universe_cross_binding(
            [match], report, [], universe,
        )


@pytest.mark.parametrize("forgery", ("episode_id", "symbol", "cutoff", "quality_tier"))
def test_closure_rescued_match_forged_metadata_fails_before_truth(
    forgery: str,
) -> None:
    report = producer._proposal_report(
        _proposal(producer.FROZEN_QUERY_IDS[0], 4_096, "forward")
    )
    outside_id = "f" * 24
    match = {
        "episode_id": outside_id, "symbol": "RESCUED",
        "cutoff": "2018-01-01T00:00:00+00:00", "quality_tier": "B",
    }
    universe = {outside_id: {
        "episode_id": outside_id, "symbol": "RESCUED",
        "cutoff_ns": pd.Timestamp(match["cutoff"]).value,
        "quality_tier": "B", "branch_aware_bound": 0.5,
        "overflow": False,
    }}
    changed = dict(match)
    changed[forgery] = {
        "episode_id": "e" * 24, "symbol": "FORGED",
        "cutoff": "2018-01-02T00:00:00+00:00", "quality_tier": "A",
    }[forgery]
    with pytest.raises(comparator.ComparisonError):
        comparator._validate_match_universe_cross_binding(
            [changed], report,
            [{"lower_exclusive": None, "upper_inclusive": 1.0}], universe,
        )


def test_copied_sealed_producer_root_is_rejected_before_truth(tmp_path: Path) -> None:
    root = tmp_path / "owned"
    prereg, _authorities = _producer_fixture(root)
    copied = tmp_path / "copied"
    shutil.copytree(root, copied)
    truth_opened = False

    def truth(*_args: Any) -> Any:
        nonlocal truth_opened
        truth_opened = True
        return {}

    with pytest.raises(comparator.ComparisonError, match="preregistered output root"):
        comparator.compare(
            copied, tmp_path / "truth", tmp_path / "verification.json",
            truth_loader=truth, **_truth_free_test_kwargs(copied, prereg),
        )
    assert truth_opened is False
    assert not (copied / "RESULTS_OPENED.json").exists()


def test_authority_sha_and_json_share_one_identity_bound_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "authority.json"
    raw = b'{"schema_version":"synthetic","value":1}\n'
    path.write_bytes(raw)
    real_open = comparator.os.open
    opened = 0

    def observed_open(*args: Any, **kwargs: Any) -> int:
        nonlocal opened
        opened += 1
        return real_open(*args, **kwargs)

    monkeypatch.setattr(comparator.os, "open", observed_open)
    monkeypatch.setattr(
        producer, "_sha", lambda _path: (_ for _ in ()).throw(
            AssertionError("separate authority hash read forbidden")
        ),
    )
    monkeypatch.setattr(
        producer, "_read_json", lambda _path: (_ for _ in ()).throw(
            AssertionError("second authority parse read forbidden")
        ),
    )
    assert comparator._read_sha_json(path, sha256(raw).hexdigest())["value"] == 1
    assert opened == 1


@pytest.mark.parametrize(
    "created_at",
    (["2026-01-01"], {"utc": True}, "2026-01-01T00:00:00",
     "2026-01-01T01:00:00+01:00"),
)
def test_rehashed_case_created_at_requires_canonical_utc(
    tmp_path: Path, created_at: Any,
) -> None:
    root = tmp_path / "evidence"
    prereg, _authorities = _producer_fixture(root)
    query_id = producer.FROZEN_QUERY_IDS[0]
    path = root / "cases" / f"00-{query_id}.json"
    case = producer._read_json(path)
    case["created_at"] = created_at
    _rewrite_case_and_seal(root, case)
    with pytest.raises(comparator.ComparisonError, match="semantic fields differ"):
        comparator.validate_producer_before_truth(
            root, **_truth_free_test_kwargs(root, prereg),
        )


def test_rehashed_closure_excluded_prefix_digest_is_exact(tmp_path: Path) -> None:
    certificate, matches = _synthetic_fallback_certificate(tmp_path)
    forward = producer._proposal_report(
        _proposal(producer.FROZEN_QUERY_IDS[0], 4_096, "forward")
    )
    excluded = stable_hash(sorted(row.episode_id for row in forward.candidates))
    reports = []
    for closure in certificate["threshold_closure_passes"]:
        closure["excluded_prefix_digest"] = excluded
        reports.append({
            "excluded_prefix_digest": excluded,
            "admitted_rows": closure["admitted_rows"],
            "admitted_set_digest": closure["admitted_set_digest"],
            "scan_result_digest": closure["scan_result_digest"],
            "minimum_packed_unclassified_bound":
                closure["minimum_packed_unclassified_bound"],
            "eligible_rows": certificate["eligible_candidates"],
            "excluded_eligible_rows": forward.eligible_rows,
        })
    certificate["result_digest"] = _rehash_certificate(certificate, matches)
    comparator._validate_closure_scan_evidence(certificate, forward, reports)
    certificate["threshold_closure_passes"][0]["excluded_prefix_digest"] = "0" * 64
    certificate["result_digest"] = _rehash_certificate(certificate, matches)
    with pytest.raises(comparator.ComparisonError, match="closure scan evidence differs"):
        comparator._validate_closure_scan_evidence(certificate, forward, reports)


def test_durable_source_generation_identity_detects_mutation(tmp_path: Path) -> None:
    generation = (
        tmp_path / "store/generations" / producer.GENERATION_ID
    )
    generation.mkdir(parents=True)
    for name in ("manifest.json", "bound-rows.bin", "overflow-exact-fallback.bin"):
        (generation / name).write_bytes(name.encode())
    before = comparator._source_generation_identity(tmp_path / "store")
    (generation / "bound-rows.bin").write_bytes(b"mutated")
    after = comparator._source_generation_identity(tmp_path / "store")
    assert after != before
