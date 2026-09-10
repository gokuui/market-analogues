import pytest

from market_analogues.nse_e2e_verification import (
    NseE2EError, _valid_receipt, analogue_match_from_payload,
    validate_nse_e2e_result_payload,
)
from market_analogues.types import stable_hash


def _payload() -> dict:
    return {
        "episode_id": "c391f79ad03b6f94e8ea711a",
        "dataset_id": "demo", "symbol": "AAA",
        "cutoff": "2020-01-02T00:00:00", "lookback": 63,
        "representation_version": "dense-v1", "total_distance": .25,
        "component_distances": {"price": .25},
        "alignment": [[0, 0], [1, 1]],
        "quality_tier": "A", "quality_issues": [],
    }


def test_authority_match_payload_requires_exact_episode_identity() -> None:
    payload = _payload()
    # Use the implementation-derived ID rather than a hand-maintained hash.
    payload.pop("episode_id")
    from market_analogues.types import EpisodeKey, InstrumentKey
    import pandas as pd
    payload["episode_id"] = EpisodeKey(
        InstrumentKey("demo", "AAA"), pd.Timestamp(payload["cutoff"]),
        payload["lookback"], payload["representation_version"],
    ).id
    match = analogue_match_from_payload(payload)
    assert match.episode_key.id == payload["episode_id"]
    payload["episode_id"] = "tampered"
    with pytest.raises(NseE2EError, match="identity"):
        analogue_match_from_payload(payload)


def test_fresh_verification_receipt_excludes_only_publication_measurements() -> None:
    state = {"passed": True, "verified_cases": 12}
    receipt = {
        **state, "result_digest": stable_hash(state),
        "elapsed_seconds": 1.25, "created_at": "now",
    }
    assert _valid_receipt(
        receipt, omitted={"result_digest", "elapsed_seconds", "created_at"},
    )
    receipt["verified_cases"] = 11
    assert not _valid_receipt(
        receipt, omitted={"result_digest", "elapsed_seconds", "created_at"},
    )


def test_persisted_nse_result_is_fully_reconstructible() -> None:
    record = {
        "cutoff_role": "current", "quality_tier": "A",
        "liquidity_stratum": "high", "matches": 20, "evidence_rows": 60,
        "eligible_5": 20, "eligible_20": 19, "eligible_60": 18,
        "maximum_selected_rescore_delta": 0.0,
    }
    state = {
        "schema_version": "nse-real-e2e-verification-v2", "dataset": "nse",
        "registry_digest": "registry", "source_universe_digest": "universe",
        "workers": 12, "benchmark_full_fingerprint_drift": False,
        "cases": [record], "source_lock_unchanged": True,
        "authority_generation": "fresh-prefix-locked-v1",
        "authority_verification_digest": "verified",
        "prefix_lock_cutoff": "2026-02-11T00:00:00", "failures": [],
    }
    payload = {
        **{key: value for key, value in state.items() if key != "cases"},
        "case_records": [record], "result_digest": stable_hash(state),
        "passed": True, "cases": 1, "current_cases": 1, "historical_cases": 0,
        "quality_tiers": ["A"], "liquidity_strata": ["high"],
        "exact_matches": 20, "evidence_rows": 60,
        "eligible_5": 20, "eligible_20": 19, "eligible_60": 18,
        "maximum_selected_rescore_delta": 0.0,
        "selected_match_rescore_equal": True,
    }
    assert validate_nse_e2e_result_payload(payload) == ()
    payload["eligible_20"] = 20
    assert "derived field differs: eligible_20" in validate_nse_e2e_result_payload(payload)
