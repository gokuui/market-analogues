import pytest

from market_analogues.nse_e2e_verification import (
    NseE2EError, _valid_receipt, analogue_match_from_payload,
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
