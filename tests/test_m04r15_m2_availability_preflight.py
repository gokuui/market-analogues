from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from experiments.m04r import m04r15_m2_availability_preflight as preflight
from market_analogues.untouched_availability import DateMetadata


def contract() -> dict:
    return {
        "contract_digest": "contract",
        "upstream": {"consumed_query_boundary": "2025-08-29"},
        "dataset": {
            "required_stock_columns": ["date", "open", "high", "low", "close", "volume"],
            "minimum_history_rows_at_cutoff": 312,
            "minimum_eligible_stock_files_each_cutoff": 2,
        },
        "prospective_schedule": {
            "initial_consumed_boundary_purge_sessions": 60,
            "evaluation_months": 12,
            "maximum_disclosed_horizon_sessions": 60,
        },
    }


def metadata(last_date: str, path: str = "A.parquet") -> DateMetadata:
    return DateMetadata(
        path=path, bytes=10, rows=500, row_groups=1, first_date="2020-01-02",
        last_date=last_date,
        schema_names=("date", "open", "high", "low", "close", "volume"),
    )


def test_current_style_stale_source_is_explicitly_blocked() -> None:
    sessions = pd.bdate_range("2000-01-03", "2026-05-08")
    state = preflight.semantic_state(
        contract(), {"created_at": "2026-09-11T06:00:00+00:00", "result_digest": "m1"},
        {"verification_digest": "v1"}, [metadata("2026-03-30")], sessions,
        "commit", {"runtime": "hash"}, pd.Timestamp("2026-09-11T07:00:00Z"),
    )
    assert state["status"] == "source_extension_required"
    assert state["readiness_passed"] is False
    assert state["registry_creation_authorized"] is False
    assert state["schedule"]["observed_cutoff_count"] == 0
    assert state["blocking_reasons"] == [
        "fewer_than_twelve_post_freeze_benchmark_month_end_cutoffs",
        "stock_source_does_not_cover_every_required_cutoff",
        "final_cutoff_lacks_sixty_subsequent_benchmark_sessions",
        "stock_source_does_not_extend_through_final_maturity",
    ]
    assert state["source_values_opened"] is False
    assert state["predictive_claim_authorized"] is False


def test_complete_future_calendar_and_stock_coverage_is_ready() -> None:
    sessions = pd.bdate_range("2000-01-03", "2028-01-31")
    state = preflight.semantic_state(
        contract(), {"created_at": "2026-09-11T06:00:00+00:00", "result_digest": "m1"},
        {"verification_digest": "v1"},
        [metadata("2028-01-31", "A.parquet"), metadata("2028-01-31", "B.parquet")],
        sessions, "commit", {"runtime": "hash"}, pd.Timestamp("2028-02-01T00:00:00Z"),
    )
    assert state["status"] == "ready_for_registry"
    assert state["readiness_passed"] is True
    assert state["registry_creation_authorized"] is True
    assert state["schedule"]["observed_cutoff_count"] == 12
    assert state["schedule"]["final_sixty_session_maturity"] is not None
    assert state["schedule"]["eligible_stock_files_at_final_maturity"] == 2
    assert not state["blocking_reasons"]


def test_schema_closure_and_create_only_publication(tmp_path: Path) -> None:
    sessions = pd.bdate_range("2025-01-01", "2028-01-31")
    broken = metadata("2028-01-31")
    broken = DateMetadata(**{**broken.__dict__, "schema_names": ("date", "close")})
    with pytest.raises(preflight.PreflightError, match="schema closure"):
        preflight.semantic_state(
            contract(), {"created_at": "2026-09-11", "result_digest": "m1"},
            {"verification_digest": "v1"}, [broken], sessions, "commit", {},
            pd.Timestamp("2028-02-01T00:00:00Z"),
        )
    path = tmp_path / "RESULT.json"
    preflight.publish(path, {"status": "blocked"})
    with pytest.raises(preflight.PreflightError, match="create-only"):
        preflight.publish(path, {"status": "blocked"})


def test_prerequisite_snapshot_is_restart_safe_and_immutable(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    first = preflight.publish_restart_safe(path, {"status": "metadata_only"})
    assert preflight.publish_restart_safe(path, {"status": "metadata_only"}) == first
    with pytest.raises(preflight.PreflightError, match="differs"):
        preflight.publish_restart_safe(path, {"status": "changed"})


def test_frozen_contract_digest_and_claim_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    value = preflight.decode((root / preflight.CONTRACT).read_bytes(), root / preflight.CONTRACT)
    state = {key: item for key, item in value.items() if key != "contract_digest"}
    assert value["contract_digest"] == preflight.stable(state)
    assert value["availability_access"]["source_values_opened_by_preflight"] is False
    assert not any(value["claims"].values())
