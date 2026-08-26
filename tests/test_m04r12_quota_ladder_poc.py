from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from market_analogues.packed_bound_search import BoundProposal


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments/m04r/m04r12_quota_ladder_poc.py"
)
SPEC = importlib.util.spec_from_file_location("m04r12_quota_ladder_poc", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _entry(identifier: int, score: float, *, total: float | None = None):
    row = np.empty(1, dtype=module.packed_search._ENTRY_DTYPE)
    row[0]["episode_id"] = identifier.to_bytes(12, "big")
    row[0]["cutoff_ns"] = identifier
    row[0]["symbol_id"] = 0
    row[0]["quality_tier"] = 1
    row[0]["total"] = score if total is None else total
    row[0]["route_score"] = score
    row[0]["overflow"] = False
    return row


def _heaps(composite: int = 1_002, component: int = 65):
    result = {}
    first = 1
    for route in module.MAX_ROUTE_QUOTAS:
        count = composite if route == "composite" else component
        rows = [_entry(first + offset, float(offset)) for offset in range(count)]
        result[route] = np.concatenate(rows)
        first += count
    return result


def _inputs(failed):
    return module.OpenedInputs(
        registry_digest="r" * 64, producer_contract_digest="p" * 64,
        comparison_digest="c" * 64, comparison_seal_digest="s" * 64,
        results_opened_digest="o" * 64, authority_seal_digest="a" * 64,
        generation_id="g" * 64, provenance_digest="v" * 64,
        resident_reserve_bytes=0, source_store_root=Path("/source/store"),
        resident_root=Path("/resident"), failed_cases=(failed,),
        original_retained_total=10, unaffected_retained_total=0,
    )


def test_quota_ladder_is_exact_ordered_and_bounded():
    configurations = module.quota_configurations()
    assert [row["name"] for row in configurations[:10]] == [
        "baseline", "composite_2000", "composite_4000", "composite_8000",
        "components_128", "components_256", "components_512",
        "composite_2000_components_256", "composite_4000_components_512",
        "composite_8000_components_512",
    ]
    assert len(configurations) == 10 + len(module.COMPONENT_ROUTES)
    assert configurations[0]["route_quotas"] == module.DEFAULT_ROUTE_QUOTAS
    assert configurations[9]["route_quotas"] == module.MAX_ROUTE_QUOTAS
    for route in module.COMPONENT_ROUTES:
        single = next(row for row in configurations if row["name"] == f"single_{route}_512")
        assert single["route_quotas"][route] == 512
        assert all(
            single["route_quotas"][other] == (512 if other == route else 64)
            for other in module.COMPONENT_ROUTES
        )


def test_rank_sidecar_preserves_score_id_ties_overflow_and_prefixes():
    tied = np.concatenate((
        _entry(3, 0.0), _entry(1, 0.0), _entry(2, 0.0), _entry(4, 1.0),
    ))
    tied[0]["overflow"] = True
    tied[1]["overflow"] = True
    tied[2]["overflow"] = True
    rankings = {route: module._ranked_entries(tied) for route in module.MAX_ROUTE_QUOTAS}
    assert [row["episode_id"] for row in rankings["composite"][:3]] == [
        value.to_bytes(12, "big").hex() for value in (1, 2, 3)
    ]
    restored = module._deserialize_rankings(rankings)
    quotas = {route: 2 for route in module.MAX_ROUTE_QUOTAS}
    selected = module._select_heaps(restored, quotas)
    assert {
        bytes(row["episode_id"]).hex() for row in selected["composite"]
    } == {value.to_bytes(12, "big").hex() for value in (1, 2)}
    assert all(bool(row["overflow"]) for row in selected["composite"])
    larger = module._select_heaps(restored, {route: 3 for route in module.MAX_ROUTE_QUOTAS})
    assert {
        bytes(row["episode_id"]) for row in selected["composite"]
    }.issubset({bytes(row["episode_id"]) for row in larger["composite"]})


def test_recall_accounting_is_authority_ordered_and_strict():
    truth = [f"{value:024x}" for value in range(20)]
    result = module.recall_accounting(truth, reversed(truth[1:]))
    assert result["retained_authority_episode_ids"] == truth[1:]
    assert result["missing_authority_episode_ids"] == truth[:1]
    assert result["retained_count"] == 19
    assert result["passes_19_of_20"] is True
    with pytest.raises(module.QuotaLadderError, match="not unique"):
        module.recall_accounting(truth, [truth[0], truth[0]])


def _terminal_documents(monkeypatch):
    now = datetime.now(timezone.utc).isoformat()
    marker = {
        "schema_version": module.RESULTS_OPENED_SCHEMA,
        "registry_digest": module.FROZEN_REGISTRY_DIGEST,
        "producer_contract_digest": module.FROZEN_PRODUCER_CONTRACT_DIGEST,
        "semantic_seal_digest": "e" * 64, "performance_final_digest": "f" * 64,
        "run_complete_digest": "u" * 64,
        "status": "authority results about to be opened exactly once",
        "created_at": now,
    }
    marker["result_digest"] = module.stable_hash(
        module._without(marker, {"created_at"})
    )
    monkeypatch.setattr(module, "FROZEN_RESULTS_OPENED_DIGEST", marker["result_digest"])
    rows = []
    for index in range(60):
        retained = 18 if index < 4 else 20
        rows.append({
            "registry_case_id": f"case-{index}",
            "query_episode_id": f"{index:024x}",
            "candidate_semantic_digest": "d" * 64,
            "authority_case_digest": "a" * 64, "candidate_count": 1_000,
            "retained_count": retained, "recall_at_20": retained / 20,
            "perfect_20_of_20": retained == 20,
            "missing_authority_episode_ids": [] if retained == 20 else ["0" * 24] * 2,
            "failures": [] if retained == 20 else ["candidate retained fewer than 19 of 20"],
            "passed": retained == 20,
        })
    comparison = {
        "schema_version": module.COMPARISON_SCHEMA,
        "registry_digest": module.FROZEN_REGISTRY_DIGEST,
        "producer_contract_digest": module.FROZEN_PRODUCER_CONTRACT_DIGEST,
        "semantic_seal_digest": "e" * 64, "performance_final_digest": "f" * 64,
        "performance_passed": True,
        "results_opened_marker_digest": marker["result_digest"],
        "authority_contract_digest": module.FROZEN_AUTHORITY_CONTRACT_DIGEST,
        "authority_matrix_digest": module.FROZEN_AUTHORITY_MATRIX_DIGEST,
        "authority_seal_digest": module.FROZEN_AUTHORITY_SEAL_DIGEST,
        "completed_cases": 60, "retained_total": sum(row["retained_count"] for row in rows),
        "retained_denominator": 1_200, "minimum_retained_count": 18,
        "perfect_20_of_20_cases": 56, "perfect_20_of_20_is_descriptive_only": True,
        "cases": rows, "failures": ["failure"],
        "gates": {"all_60_authority_cases_valid": True,
                  "every_case_retains_at_least_19_of_20": False,
                  "aggregate_retains_at_least_1188_of_1200": False},
        "passed": False, "candidate_results_opened": True,
        "production_promotion_authorized": False,
        "real_forward_outcomes_accessed": False, "created_at": now,
    }
    comparison["result_digest"] = module.stable_hash(
        module._without(comparison, {"created_at"})
    )
    monkeypatch.setattr(module, "FROZEN_COMPARISON_DIGEST", comparison["result_digest"])
    seal = {
        "schema_version": module.COMPARISON_SEAL_SCHEMA,
        "registry_digest": module.FROZEN_REGISTRY_DIGEST,
        "producer_contract_digest": module.FROZEN_PRODUCER_CONTRACT_DIGEST,
        "comparison_digest": comparison["result_digest"],
        "results_opened_marker_digest": marker["result_digest"],
        "authority_seal_digest": module.FROZEN_AUTHORITY_SEAL_DIGEST,
        "candidate_results_opened": True, "comparison_gate_passed": False,
        "production_promotion_authorized": False, "created_at": now,
    }
    seal["seal_digest"] = module.stable_hash(module._without(seal, {"created_at"}))
    monkeypatch.setattr(module, "FROZEN_COMPARISON_SEAL_DIGEST", seal["seal_digest"])
    return marker, comparison, seal


def test_terminal_refuses_missing_marker_passed_or_promoted_comparison(monkeypatch):
    with pytest.raises(module.QuotaLadderError, match="RESULTS_OPENED"):
        module._validate_marker({})
    marker, comparison, seal = _terminal_documents(monkeypatch)
    module._validate_marker(marker)
    assert len(module._validate_comparison(comparison, seal, marker)) == 4
    for target, key in ((comparison, "passed"), (comparison, "production_promotion_authorized"),
                        (seal, "production_promotion_authorized")):
        corrupted = dict(target)
        corrupted[key] = True
        with pytest.raises(module.QuotaLadderError, match="failed comparison"):
            module._validate_comparison(
                corrupted if target is comparison else comparison,
                corrupted if target is seal else seal, marker,
            )


def test_derived_baseline_reconstructs_standard_digest_and_truth_ranks():
    heaps = _heaps()
    rankings = {route: module._ranked_entries(rows) for route, rows in heaps.items()}
    selected = module._select_heaps(heaps, module.DEFAULT_ROUTE_QUOTAS)
    candidates, _, digest = module.packed_search._finalize(selected, ("SYM",))
    ids = tuple(row.episode_id for row in candidates)
    truth = ids[:20]
    case = {"case_id": "case", "episode_id": "0" * 24}
    failed = module.FailedCase(
        0, case, {"retained_count": 20}, "a" * 64, truth,
        digest, ids, "",
    )
    inputs = _inputs(failed)
    max_payload = {
        "result_digest": "m" * 64, "poc_contract_digest": "p" * 64,
        "symbols": ["SYM"],
        "route_rankings": rankings,
        "forward": {"rows_scanned": 10, "eligible_rows": 10,
                    "eligible_main_rows": 10, "eligible_overflow_rows": 0,
                    "elapsed_seconds": 1.0, "peak_rss_mb": 2.0},
    }
    failed = module.FailedCase(
        0, case, {"retained_count": 20}, "a" * 64, truth, digest, ids,
        module._standard_result_digest(
            failed, inputs, max_payload, module.DEFAULT_ROUTE_QUOTAS,
            {route: len(rows) if len(rows) < module.DEFAULT_ROUTE_QUOTAS[route]
             else module.DEFAULT_ROUTE_QUOTAS[route] for route, rows in heaps.items()},
            digest,
        ),
    )
    inputs = _inputs(failed)
    payload = module._derive_checkpoint(
        failed, module.quota_configurations()[0], inputs,
        {"content_digest": "z" * 64, "ready_digest": "y" * 64,
         "stable_identity_digest": "x" * 64}, max_payload,
    )
    assert payload["baseline_reproduces_terminal_v2"] is True
    assert payload["recall"]["retained_count"] == 20
    assert payload["candidate_episode_ids"] == list(ids)
    assert payload["authority_top20_route_ranks"][0]["episode_id"] == truth[0]


def test_create_only_output_is_idempotent_but_never_overwritten(tmp_path):
    path = tmp_path / "checkpoint.json"
    module._atomic_create(path, {"value": 1})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        module._atomic_create(path, {"value": 2})
    assert path.read_bytes() == original


def _producer_fixture():
    heaps = {}
    base = np.concatenate([_entry(index + 1, float(index)) for index in range(20)])
    for route in module.MAX_ROUTE_QUOTAS:
        heaps[route] = base.copy()
    candidates, route_counts, candidate_digest = module.packed_search._finalize(
        heaps, ("SYM",),
    )
    ids = tuple(row.episode_id for row in candidates)
    cases = tuple(
        module.ProducerCase(
            ordinal, {
                "case_id": f"case-{ordinal}", "episode_id": query_id,
                "symbol": "SYM", "cutoff": "2020-01-01", "lookback": 252,
                "representation_version": "v1",
            }, candidate_digest, ids, "pending",
        )
        for ordinal, query_id in enumerate(module.FROZEN_QUERY_IDS)
    )
    inputs = module.ProducerInputs(
        registry_digest="r" * 64, producer_contract_digest="p" * 64,
        poc_contract_digest="q" * 64, generation_id="g" * 64,
        provenance_digest="v" * 64, resident_content_digest="c" * 64,
        resident_ready_digest="d" * 64, resident_reserve_bytes=0,
        source_store_root=Path("/source/store"), resident_root=Path("/resident"),
        cases=cases,
    )
    stable = {
        "content_digest": "c" * 64, "ready_digest": "d" * 64,
        "seal_digest": "s" * 64, "ready_file_sha256": "f" * 64,
        "lease": {
            "ready_digest": "d" * 64, "content_digest": "c" * 64,
            "ready_file_sha256": "f" * 64,
        },
        "store_root": "/resident/store",
    }
    resident = {
        **stable, "stable_identity_digest": module.stable_hash(stable),
        "validation_observation": {
            "observation_digest": "e" * 64,
            "observed_at": datetime.now(timezone.utc).isoformat(),
        },
    }

    def scanner(_inputs, _resident, query, *, order):
        case = next(value for value in cases if value.query_id == query)
        block_rows = 4_096 if order == "forward" else 4_097
        max_stub = {"forward": {
            "rows_scanned": 100, "eligible_rows": 20,
            "eligible_main_rows": 20, "eligible_overflow_rows": 0,
        }}
        result_digest = module._standard_result_digest(
            case, inputs, max_stub, module.MAX_ROUTE_QUOTAS,
            route_counts, candidate_digest,
        )
        report = SimpleNamespace(
            candidates=candidates, candidate_digest=candidate_digest,
            result_digest=result_digest, route_counts=route_counts,
            rows_scanned=100, eligible_rows=20, eligible_main_rows=20,
            eligible_overflow_rows=0, block_rows=block_rows, block_order=order,
            elapsed_seconds=1.0, peak_rss_mb=100.0,
        )
        sidecar = {
            "symbols": ["SYM"],
            "route_rankings": {
                route: module._ranked_entries(rows) for route, rows in heaps.items()
            },
            "task_elapsed_seconds": 2.0 if order == "forward" else 1.5,
            "task_peak_rss_mb": 110.0,
        }
        return report, sidecar

    # Bind the exact baseline result digest produced by the standard reconstruction.
    rebound = []
    for case in cases:
        baseline_stub = {"forward": {
            "rows_scanned": 100, "eligible_rows": 20,
            "eligible_main_rows": 20, "eligible_overflow_rows": 0,
        }}
        rebound.append(module.ProducerCase(
            case.ordinal, case.registry_case, case.baseline_candidate_digest,
            case.baseline_candidate_ids, module._standard_result_digest(
                case, inputs, baseline_stub, module.DEFAULT_ROUTE_QUOTAS,
                {route: 20 for route in module.MAX_ROUTE_QUOTAS},
                case.baseline_candidate_digest,
            ),
        ))
    inputs = module.ProducerInputs(
        **{**inputs.__dict__, "cases": tuple(rebound)}
    )
    cases = inputs.cases
    return inputs, resident, scanner


def test_fresh_max_path_and_strict_resume_integration(tmp_path):
    inputs, resident, scanner = _producer_fixture()
    prereg = {"contract_digest": inputs.poc_contract_digest}
    seal = module.produce_max_evidence(
        config_path=Path("/config"), output_root=tmp_path, inputs=inputs,
        preregistration=prereg, resident=resident,
        query_builder=lambda _path, case: case.query_id, max_scanner=scanner,
        resident_checker=lambda *_: None,
    )
    assert seal["all_performance_gates_passed"] is True
    assert len(list((tmp_path / "max-scans").glob("*.json"))) == 4
    resumed = module.produce_max_evidence(
        config_path=Path("/config"), output_root=tmp_path, inputs=inputs,
        preregistration=prereg, resident=resident,
        query_builder=lambda *_: pytest.fail("resume rescanned"),
        max_scanner=lambda *_args, **_kwargs: pytest.fail("resume rescanned"),
        resident_checker=lambda *_: None,
    )
    assert resumed == seal
    module.validate_producer_evidence(
        output_root=tmp_path, inputs=inputs,
        preregistration=prereg, resident=resident,
    )


def test_resident_snapshot_is_stable_across_independent_process_observations():
    _inputs_value, resident, _scanner = _producer_fixture()
    later = {
        **resident,
        "validation_observation": {
            "observation_digest": "9" * 64,
            "observed_at": "2030-01-01T00:00:00+00:00",
        },
    }
    code = f"""
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location('m04r12_child', {str(SCRIPT)!r})
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
print(json.dumps(mod._stable_resident_snapshot(json.loads(sys.stdin.read())), sort_keys=True))
"""
    observations = []
    for value in (resident, later):
        result = subprocess.run(
            [sys.executable, "-c", code], input=json.dumps(value), text=True,
            capture_output=True, check=True,
        )
        observations.append(result.stdout)
    assert observations[0] == observations[1]


def test_resident_mutation_between_scan_and_postcheck_fails(tmp_path):
    inputs, resident, scanner = _producer_fixture()
    state = {"mutated": False}

    def mutating_scanner(*args, **kwargs):
        result = scanner(*args, **kwargs)
        state["mutated"] = True
        return result

    def checker(_inputs, _resident):
        if state["mutated"]:
            raise module.QuotaLadderError("resident identity changed around physical scan")

    with pytest.raises(module.QuotaLadderError, match="resident identity changed"):
        module.produce_max_evidence(
            config_path=Path("/config"), output_root=tmp_path, inputs=inputs,
            preregistration={}, resident=resident,
            query_builder=lambda _path, case: case.query_id,
            max_scanner=mutating_scanner, resident_checker=checker,
        )


def test_baseline_must_match_before_producer_seal(tmp_path):
    inputs, resident, scanner = _producer_fixture()
    bad = module.ProducerInputs(**{
        **inputs.__dict__,
        "cases": (
            module.ProducerCase(**{
                **inputs.cases[0].__dict__, "baseline_result_digest": "0" * 64,
            }),
            *inputs.cases[1:],
        ),
    })
    with pytest.raises(module.QuotaLadderError, match="preregistered baseline"):
        module.produce_max_evidence(
            config_path=Path("/config"), output_root=tmp_path, inputs=bad,
            preregistration={}, resident=resident,
            query_builder=lambda _path, case: case.query_id, max_scanner=scanner,
            resident_checker=lambda *_: None,
        )
    assert not (tmp_path / "MAX_SCAN_SEALED.json").exists()
    assert not (tmp_path / "RESULTS_OPENED.json").exists()


def test_max_resource_failure_and_output_budget_fail_closed(tmp_path, monkeypatch):
    inputs, resident, scanner = _producer_fixture()

    def slow(*args, **kwargs):
        report, sidecar = scanner(*args, **kwargs)
        sidecar["task_elapsed_seconds"] = 121.0
        return report, sidecar

    with pytest.raises(module.QuotaLadderError, match="performance gate failed"):
        module.produce_max_evidence(
            config_path=Path("/config"), output_root=tmp_path / "slow",
            inputs=inputs, preregistration={}, resident=resident,
            query_builder=lambda _path, case: case.query_id, max_scanner=slow,
            resident_checker=lambda *_: None,
        )
    monkeypatch.setattr(module, "MAX_OUTPUT_BYTES", 1)
    with pytest.raises(module.QuotaLadderError, match="output exceeds"):
        module.produce_max_evidence(
            config_path=Path("/config"), output_root=tmp_path / "budget",
            inputs=inputs, preregistration={}, resident=resident,
            query_builder=lambda _path, case: case.query_id, max_scanner=scanner,
            resident_checker=lambda *_: None,
        )


def test_tampered_rehashed_max_resume_is_reconstructed_and_rejected(tmp_path):
    inputs, resident, scanner = _producer_fixture()
    module.produce_max_evidence(
        config_path=Path("/config"), output_root=tmp_path, inputs=inputs,
        preregistration={}, resident=resident,
        query_builder=lambda _path, case: case.query_id, max_scanner=scanner,
        resident_checker=lambda *_: None,
    )
    path = module._max_scan_path(tmp_path, inputs.cases[0])
    payload = module._read_json(path)
    payload["route_rankings"]["composite"][0]["total_hex"] = float(99).hex()
    payload["result_digest"] = module.stable_hash(
        module._without(payload, {"created_at", "result_digest"})
    )
    path.write_text(__import__("json").dumps(payload))
    with pytest.raises(module.QuotaLadderError, match="checkpoint"):
        module.validate_producer_evidence(
            output_root=tmp_path, inputs=inputs,
            preregistration={}, resident=resident,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.__setitem__("status", "other"), "max-route"),
        (lambda row: row.__setitem__("resident_ready_digest", "0" * 64), "max-route"),
        (lambda row: row["route_rankings"].__setitem__("extra", []), "max-route"),
        (lambda row: row["reverse"].__setitem__("eligible_rows", 19), "max-route"),
    ],
)
def test_rehashed_max_contract_corruption_is_rejected(tmp_path, mutation, message):
    inputs, resident, scanner = _producer_fixture()
    module.produce_max_evidence(
        config_path=Path("/config"), output_root=tmp_path, inputs=inputs,
        preregistration={}, resident=resident,
        query_builder=lambda _path, case: case.query_id, max_scanner=scanner,
        resident_checker=lambda *_: None,
    )
    path = module._max_scan_path(tmp_path, inputs.cases[0])
    payload = module._read_json(path)
    mutation(payload)
    payload["result_digest"] = module.stable_hash(
        module._without(payload, {"created_at", "result_digest"})
    )
    path.write_text(__import__("json").dumps(payload))
    with pytest.raises(module.QuotaLadderError, match=message):
        module.validate_producer_evidence(
            output_root=tmp_path, inputs=inputs,
            preregistration={}, resident=resident,
        )


def _truth_from_producer(inputs):
    failed = tuple(module.FailedCase(
        case.ordinal, case.registry_case, {"retained_count": 20},
        "a" * 64, case.baseline_candidate_ids[:20],
        case.baseline_candidate_digest, case.baseline_candidate_ids,
        case.baseline_result_digest,
    ) for case in inputs.cases)
    return module.OpenedInputs(
        registry_digest=inputs.registry_digest,
        producer_contract_digest=inputs.producer_contract_digest,
        comparison_digest="x" * 64, comparison_seal_digest="y" * 64,
        results_opened_digest=module.FROZEN_RESULTS_OPENED_DIGEST,
        authority_seal_digest="a" * 64, generation_id=inputs.generation_id,
        provenance_digest=inputs.provenance_digest, resident_reserve_bytes=0,
        source_store_root=inputs.source_store_root, resident_root=inputs.resident_root,
        failed_cases=failed, original_retained_total=1_200,
        unaffected_retained_total=1_120,
    )


def test_marker_precedes_truth_read_and_terminal_resume_is_recursive(tmp_path):
    inputs, resident, scanner = _producer_fixture()
    prereg = {}
    module.produce_max_evidence(
        config_path=Path("/config"), output_root=tmp_path, inputs=inputs,
        preregistration=prereg, resident=resident,
        query_builder=lambda _path, case: case.query_id, max_scanner=scanner,
        resident_checker=lambda *_: None,
    )
    calls = []

    def truth_loader():
        marker = tmp_path / "RESULTS_OPENED.json"
        assert marker.is_file()
        assert module._read_json(marker)["status"] == "authority and comparison truth about to be opened"
        calls.append("truth")
        return _truth_from_producer(inputs)

    later_resident = {
        **resident,
        "validation_observation": {
            "observation_digest": "8" * 64,
            "observed_at": "2031-01-01T00:00:00+00:00",
        },
    }
    terminal = module.compare_producer_evidence(
        output_root=tmp_path, producer_inputs=inputs,
        preregistration=prereg, resident=later_resident, truth_loader=truth_loader,
    )
    assert calls == ["truth"]
    assert terminal["production_promotion_authorized"] is False
    resumed = module.compare_producer_evidence(
        output_root=tmp_path, producer_inputs=inputs,
        preregistration=prereg, resident=resident, truth_loader=truth_loader,
    )
    assert resumed == terminal
    # A rehashed derived artifact must fail recursive terminal resume.
    path = next((tmp_path / "derived").glob("*.json"))
    payload = module._read_json(path)
    payload["candidate_count"] += 1
    payload["result_digest"] = module.stable_hash(
        module._without(payload, {"created_at", "result_digest"})
    )
    path.write_text(__import__("json").dumps(payload))
    with pytest.raises(module.QuotaLadderError, match="derived"):
        module.compare_producer_evidence(
            output_root=tmp_path, producer_inputs=inputs,
            preregistration=prereg, resident=resident,
            truth_loader=lambda: _truth_from_producer(inputs),
        )


def test_pre_marker_tree_corruption_blocks_truth_loader(tmp_path):
    inputs, resident, scanner = _producer_fixture()
    module.produce_max_evidence(
        config_path=Path("/config"), output_root=tmp_path, inputs=inputs,
        preregistration={}, resident=resident,
        query_builder=lambda _path, case: case.query_id, max_scanner=scanner,
        resident_checker=lambda *_: None,
    )
    (tmp_path / "unexpected.json").write_text("{}")
    with pytest.raises(module.QuotaLadderError, match="unexpected artifact"):
        module.compare_producer_evidence(
            output_root=tmp_path, producer_inputs=inputs,
            preregistration={}, resident=resident,
            truth_loader=lambda: pytest.fail("truth read preceded tree validation"),
        )
    (tmp_path / "unexpected.json").unlink()
    (tmp_path / "linked.json").symlink_to(tmp_path / "POC_CONTRACT.json")
    with pytest.raises(module.QuotaLadderError, match="symlink"):
        module.compare_producer_evidence(
            output_root=tmp_path, producer_inputs=inputs,
            preregistration={}, resident=resident,
            truth_loader=lambda: pytest.fail("truth read preceded symlink validation"),
        )


def test_rehashed_derived_status_corruption_is_rejected(tmp_path):
    inputs, resident, scanner = _producer_fixture()
    module.produce_max_evidence(
        config_path=Path("/config"), output_root=tmp_path, inputs=inputs,
        preregistration={}, resident=resident,
        query_builder=lambda _path, case: case.query_id, max_scanner=scanner,
        resident_checker=lambda *_: None,
    )
    module.compare_producer_evidence(
        output_root=tmp_path, producer_inputs=inputs, preregistration={},
        resident=resident, truth_loader=lambda: _truth_from_producer(inputs),
    )
    path = next((tmp_path / "derived").glob("*.json"))
    payload = module._read_json(path)
    payload["status"] = "other"
    payload["result_digest"] = module.stable_hash(
        module._without(payload, {"created_at", "result_digest"})
    )
    path.write_text(__import__("json").dumps(payload))
    with pytest.raises(module.QuotaLadderError, match="derived"):
        module.compare_producer_evidence(
            output_root=tmp_path, producer_inputs=inputs, preregistration={},
            resident=resident, truth_loader=lambda: _truth_from_producer(inputs),
        )


def test_git_gate_requires_tracked_clean_frozen_files(tmp_path):
    prereg = {
        "script_relative_path": "experiments/m04r/m04r12_quota_ladder_poc.py",
        "code_manifest": {"files": {"src/market_analogues/types.py": "x" * 64}},
    }
    calls = []

    def clean(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    module._validate_git_gate(tmp_path, prereg, runner=clean)
    assert [call[1:3] for call in calls] == [
        ["ls-files", "--error-unmatch"], ["diff", "--quiet"],
        ["diff", "--cached"],
    ]

    def dirty(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, "", "")

    with pytest.raises(module.QuotaLadderError, match="untracked or dirty"):
        module._validate_git_gate(tmp_path, prereg, runner=dirty)


def test_producer_cli_does_not_accept_truth_paths(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        str(SCRIPT), "produce", "--config", "x", "--registry-root", "x",
        "--candidate-root", "x", "--source-full-root", "x",
        "--resident-root", "x", "--output-root", "x",
        "--authority-root", "forbidden",
    ])
    with pytest.raises(SystemExit) as raised:
        module.main()
    assert raised.value.code == 2


def test_real_preregistration_and_truth_blind_input_binding_read_only(monkeypatch):
    repository = Path(__file__).resolve().parents[1]
    artifact = repository / "config/data/analogues"
    resident = Path(
        "/dev/shm/market-analogues/m04r11-candidate-v2"
    ) / module.FROZEN_GENERATION_ID
    prereg_path = repository / module.PREREGISTRATION_RELATIVE_PATH
    prereg_payload = module._read_json(prereg_path)
    try:
        module._validate_git_gate(repository, prereg_payload)
    except module.QuotaLadderError:
        assert subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", str(
                module.PREREGISTRATION_RELATIVE_PATH
            )], cwd=repository, capture_output=True,
        ).returncode != 0
    monkeypatch.setattr(module, "_validate_git_gate", lambda *_args, **_kwargs: None)
    inputs, prereg = module.validate_producer_inputs(
        repository=repository,
        config_path=repository / "config/datasets.example.yaml",
        registry_root=artifact / "m04r10/nasdaq-untouched-authority-registry",
        candidate_root=artifact / "m04r11/candidate-pools-v2",
        source_full_root=artifact / "poc/m04r/packed-bound-full",
        resident_root=resident,
        output_root=artifact / "m04r12/development-quota-ladder-v1",
    )
    assert tuple(case.query_id for case in inputs.cases) == module.FROZEN_QUERY_IDS
    assert inputs.resident_content_digest == prereg["resident_content_digest"]
    assert inputs.resident_ready_digest == prereg["resident_ready_digest"]
    assert prereg["truth_inputs_allowed_in_producer"] is False
