from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.m04r import verify_m04r15_m2_real_source_launcher_gate as verifier


ROOT = Path(__file__).resolve().parents[1]


def test_independent_verifier_reconstructs_frozen_wait_result() -> None:
    result, evidence = verifier.verify_result(ROOT)
    assert result["status"] == "verified_wait_state"
    assert evidence["live_prerequisites"] == {
        "credential_present": False, "capacity_passed": True,
        "canonical_stock_source_present": True,
        "canonical_benchmark_present": True,
    }


def test_seal_check_rejects_tampering() -> None:
    result = json.loads((ROOT / verifier.RESULT).read_text())
    result["decision"] = copy.deepcopy(result["decision"])
    result["decision"]["registry_callback_calls"] = 1
    with pytest.raises(verifier.VerificationError, match="seal differs"):
        verifier.validate_seal(result, "result_digest")


def test_expected_identities_are_distinct_and_fixed() -> None:
    assert len({verifier.EXPECTED_CONTRACT, verifier.EXPECTED_AVAILABILITY,
                verifier.EXPECTED_AVAILABILITY_VERIFICATION,
                verifier.EXPECTED_RESULT}) == 4
    assert all(len(value) == 64 for value in (
        verifier.EXPECTED_CONTRACT, verifier.EXPECTED_AVAILABILITY,
        verifier.EXPECTED_AVAILABILITY_VERIFICATION, verifier.EXPECTED_RESULT,
        verifier.EXPECTED_RESULT_SHA256,
    ))
