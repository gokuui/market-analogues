from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import m04r14_t14_10_wf03_composite_width_poc as producer
from experiments.m04r import verify_m04r14_t14_10_wf03_composite_width_poc as subject


def _topology(groups: int, threads: int, query_id: str) -> dict:
    return {
        "schema_version": "m04r14-wf03-composite-width-run-v1",
        "status": "complete", "groups": groups,
        "threads_per_group": threads, "total_threads": 12,
        "all_zero_swap": True, "preregistration_digest": "pre",
        "cases": [{"query_id": query_id}],
        "outcomes_or_labels_used": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
    }


def test_result_validator_rejects_unmeasured_selection() -> None:
    p4 = {"result_digest": "p4", "wall_seconds": 20.0}
    p12 = {"result_digest": "p12", "wall_seconds": 10.0}
    result = {
        "schema_version": "m04r14-t14-10-wf03-composite-width-result-v1",
        "status": "complete", "passed": True,
        "gates": {
            "all_twelve_queries_equal": True,
            "all_twenty_four_certificates_close": True,
            "zero_process_swap": True,
            "outcomes_or_labels_excluded": True,
        },
        "preregistration_digest": "pre", "p4t3_digest": "p4",
        "p12t1_digest": "p12", "p4t3_wall_seconds": 20.0,
        "p12t1_wall_seconds": 10.0, "p12_over_p4_speedup": 2.0,
        "selected_topology": "p4t3", "outcomes_or_labels_used": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
    }
    with pytest.raises(subject.CompositeWidthVerificationError):
        subject._validate_result(result, {"preregistration_digest": "pre"}, p4, p12)


def test_output_loader_rejects_query_reordering(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / producer.OUTPUT_RELATIVE
    root.mkdir(parents=True)
    contract = {"preregistration_digest": "pre"}
    (root / "CONTRACT.json").write_text("{}")
    monkeypatch.setattr(subject.base, "_read", lambda path: (
        contract if path.name == "CONTRACT.json" else
        _topology(4, 3, "b") if path.name == "P4T3.json" else
        _topology(12, 1, "a") if path.name == "P12T1.json" else {}
    ))
    monkeypatch.setattr(subject.base, "_validate_seal", lambda value: None)
    with pytest.raises(subject.CompositeWidthVerificationError):
        subject._load_outputs(tmp_path, contract, [{"episode_id": "a"}])
