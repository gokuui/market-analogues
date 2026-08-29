from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "experiments/m04r/m04r14_performance_contract_v1.json"


def test_performance_contract_separates_interactive_and_batch_lanes() -> None:
    value = json.loads(CONTRACT.read_text())
    assert value["schema_version"] == "m04r14-performance-contract-v1"
    assert value["interactive_lane"]["end_to_end_seconds_p95_max"] <= 210
    assert value["interactive_lane"]["semantic_pass_rate_required"] == 1.0
    assert value["batch_lane"]["processes"] == 8
    assert value["batch_lane"]["speedup_vs_serial_service_sum_min"] >= 2.0
    assert value["stability_lane"]["complete_sixty_case_runs_required"] == 3


def test_evidence_bindings_resolve_to_immutable_artifacts() -> None:
    value = json.loads(CONTRACT.read_text())["evidence_bindings"]
    artifact = ROOT / "config/data/analogues"
    paths_and_fields = {
        "v3_complete_digest": (artifact / "m04r14/all60-certified-development-v3/COMPLETE.json", "complete_digest"),
        "throughput_result_digest": (artifact / "m04r14/throughput-development-poc-v1/RESULT.json", "result_digest"),
        "throughput_verification_digest": (artifact / "m04r14/throughput-development-poc-v1-verification/VERIFIED.json", "result_digest"),
        "authority_comparison_digest": (artifact / "m04r14/exposed-authority-comparison-v1/COMPARISON.json", "result_digest"),
        "authority_comparison_verification_digest": (artifact / "m04r14/exposed-authority-comparison-v1-verification/VERIFIED.json", "result_digest"),
    }
    for binding, (path, field) in paths_and_fields.items():
        assert value[binding] == json.loads(path.read_text())[field]
    ready = Path("/dev/shm/market-analogues/m04r11-candidate-v2/9fc6ae0ec4451133d8006897162f3443803fd30a3c44b78e80a476fbb18bb483/READY.json")
    if ready.exists():
        assert value["resident_content_digest"] == json.loads(ready.read_text())["content_digest"]


def test_performance_contract_does_not_reintroduce_memory_optimization_gate() -> None:
    value = json.loads(CONTRACT.read_text())
    assert value["stability_lane"]["rss_is_observed_not_a_fixed_failure_ceiling"] is True
    assert value["stability_lane"]["oom_events_max"] == 0
    assert value["stability_lane"]["process_swap_bytes_max"] == 0
    assert "1536" in value["claim_boundary"]["memory_policy"]


def test_growth_load_is_not_misrepresented_as_generalization() -> None:
    value = json.loads(CONTRACT.read_text())
    assert "not-generalization-evidence" in value["operational_lane"]["growth_load_definition"]
    assert value["claim_boundary"]["production_promotion_authorized"] is False
