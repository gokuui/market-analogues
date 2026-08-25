from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

from market_analogues.certified_packed_search import certified_packed_search_contract
from market_analogues.m04r_certified_search_verification import (
    _certificate_digest, _evidence_digest_payload,
    verify_m04r_certified_packed_search,
    write_m04r_certified_search_verification,
)
from market_analogues.m04r_full_pack_verification import EVIDENCE_OMITTED
from market_analogues.types import stable_hash


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True))


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, dict[str, object]]:
    query_id = "a" * 24
    generation = "b" * 64
    contract = certified_packed_search_contract()["digest"]
    matches = [{
        "episode_id": f"{index + 1:024x}",
        "symbol": f"S{index:02d}",
        "cutoff": "2020-01-01T00:00:00",
        "total_distance": 0.1 + index / 100,
        "component_distances": {"coarse": 0.2 + index / 100},
        "alignment": [[0, 0], [1, 1]],
        "quality_tier": "A",
    } for index in range(20)]
    authority: dict[str, object] = {
        "schema_version": "gate12-authority-v1",
        "query_episode_id": query_id,
        "matches": matches,
        "result_digest": stable_hash(matches),
        "repeated_digest": stable_hash(matches),
    }
    authority["authority_digest"] = stable_hash(authority)
    authority_path = tmp_path / "authority.json"
    _write(authority_path, authority)

    build: dict[str, object] = {
        "schema_version": "m04r-packed-bound-full-build-v1",
        "generation_id": generation,
        "gate_passed": True,
        "shadow_generation": True,
        "real_forward_outcomes_accessed": False,
        "benchmark_selection": {"query_episode_id": query_id},
    }
    build["result_digest"] = stable_hash({
        key: value for key, value in build.items() if key not in EVIDENCE_OMITTED
    })
    build_path = tmp_path / "build.json"
    _write(build_path, build)

    certificate: dict[str, object] = {
        "schema_version": "m04r-certified-packed-search-v1",
        "contract_digest": contract,
        "generation_id": generation,
        "query_episode_id": query_id,
        "input_digest": "c" * 64,
        "eligible_candidates": 100,
        "exact_evaluated": 20,
        "safely_pruned": 80,
        "stopped_early": True,
        "stop_threshold": 0.29,
        "next_lower_bound": 0.3,
        "maximum_quantized_bound_excess": 0.0,
        "materialization_groups": 4,
        "sparse_symbols": 3,
        "batch_symbols": 1,
        "rounds": [{
            "frontier_rows": 50,
            "exact_rows": 20,
            "next_lower_bound": 0.3,
            "constrained_threshold": 0.29,
            "selected_rows": 20,
            "certified": True,
            "proposal_digest": "d" * 64,
        }],
    }
    run: dict[str, object] = {
        "controls": {"block_rows": 7, "workers": 2},
        "matches": matches,
        "ordered_ids_equal_authority": True,
        "alignments_equal_authority": True,
        "maximum_total_delta": 0.0,
        "maximum_component_delta": 0.0,
        "certificate": certificate,
        "seconds": 2.0,
    }
    digest = _certificate_digest(run)
    certificate["result_digest"] = digest
    run["certificate_digest"] = digest
    second = copy.deepcopy(run)
    second["controls"] = {"block_rows": 11, "workers": 1}
    second["seconds"] = 2.5
    gates = {
        "required_repeats_complete": True,
        "repeated_results_identical": True,
        "ordered_ids_equal_authority": True,
        "alignments_equal_authority": True,
        "total_delta_within_1e_7": True,
        "component_delta_within_1e_6": True,
        "maximum_runtime_within_600_seconds": True,
        "rss_within_1536_mib": True,
        "candidate_accounting": True,
        "strict_stopping": True,
    }
    evidence: dict[str, object] = {
        "schema_version": "m04r-certified-packed-search-gate-v1",
        "contract_digest": contract,
        "generation_id": generation,
        "full_build_evidence_digest": build["result_digest"],
        "authority_digest": authority["authority_digest"],
        "query_episode_id": query_id,
        "failed_attempt": {
            "status": "interrupted performance failure",
            "elapsed_before_interrupt_seconds": 840,
            "observed_rss_mb": 843,
            "exit_code": 130,
        },
        "runs": [run, second],
        "completed_runs": 2,
        "required_runs": 2,
        "run_seconds": [2.0, 2.5],
        "peak_rss_mb": 100.0,
        "gates": gates,
        "gate_passed": True,
        "real_forward_outcomes_accessed": False,
    }
    evidence["result_digest"] = stable_hash(_evidence_digest_payload(evidence))
    evidence_path = tmp_path / "evidence.json"
    _write(evidence_path, evidence)
    store = tmp_path / "store"
    store.mkdir()
    return evidence_path, store, build_path, authority_path, evidence


def test_certified_search_verifier_passes_and_writes(
    tmp_path: Path, monkeypatch,
) -> None:
    evidence, store, build, authority, payload = _fixture(tmp_path)
    monkeypatch.setattr(
        "market_analogues.m04r_certified_search_verification.load_packed_generation",
        lambda *args, **kwargs: SimpleNamespace(generation_id="b" * 64),
    )
    result = verify_m04r_certified_packed_search(
        evidence, store, build, authority,
    )
    assert result.passed, result.failures
    machine, html = write_m04r_certified_search_verification(
        result, tmp_path / "report",
    )
    assert json.loads(machine.read_text())["passed"] is True
    assert "PASS" in html.read_text()

    payload["runs"][1]["matches"][0]["total_distance"] += 0.01
    payload["result_digest"] = stable_hash(_evidence_digest_payload(payload))
    _write(evidence, payload)
    tampered = verify_m04r_certified_packed_search(
        evidence, store, build, authority,
    )
    assert not tampered.passed
    assert any("repeated results differ" in value for value in tampered.failures)
    assert any("certificate digest" in value for value in tampered.failures)
