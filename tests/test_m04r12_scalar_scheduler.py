from __future__ import annotations

import importlib.util
from pathlib import Path

from market_analogues.types import stable_hash


def _module():
    path = (
        Path(__file__).parents[1] / "experiments" / "m04r"
        / "m04r12_scalar_scheduler_gate.py"
    )
    spec = importlib.util.spec_from_file_location("m04r12_scalar_scheduler_gate", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scalar_timing_checkpoint_binds_case_authority_and_measurement() -> None:
    module = _module()
    case = {"case_id": "case-1", "episode_id": "a" * 24}
    payload = {
        "schema_version": "m04r12-scalar-case-timing-v1",
        "registry_case_id": case["case_id"],
        "query_episode_id": case["episode_id"],
        "authority_digest": "authority",
        "contract_digest": "contract",
        "proposal_seconds": 10.0,
        "exact_seconds": 2.0,
        "end_to_end_seconds": 13.0,
        "peak_rss_mb": 500.0,
        "created_at": "now",
    }
    payload["result_digest"] = stable_hash({
        key: value for key, value in payload.items() if key != "created_at"
    })
    assert module._timing_valid(
        payload, case=case, authority_digest="authority",
        contract_digest="contract",
    )
    payload["end_to_end_seconds"] = 14.0
    assert not module._timing_valid(
        payload, case=case, authority_digest="authority",
        contract_digest="contract",
    )
