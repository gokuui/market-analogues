from __future__ import annotations

import ast
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.m04r import verify_m04r15_m0_analogue_improvement_synthetic_gate as verifier


ROOT = Path(__file__).resolve().parents[1]


def test_verifier_does_not_import_producer_or_project_scientific_code() -> None:
    tree = ast.parse((ROOT / verifier.VERIFIER_RUNTIME[0]).read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any(name.startswith("market_analogues") for name in imported)
    assert not any("m04r15_m0_analogue_improvement_synthetic_gate" in name for name in imported)


def test_strict_decoder_rejects_duplicate_nonfinite_and_nonobject(tmp_path: Path) -> None:
    with pytest.raises(verifier.VerificationError, match="duplicate"):
        verifier.decode_json(b'{"a":1,"a":2}', tmp_path / "x")
    with pytest.raises(verifier.VerificationError, match="nonfinite"):
        verifier.decode_json(b'{"a":NaN}', tmp_path / "x")
    with pytest.raises(verifier.VerificationError, match="object"):
        verifier.decode_json(b'[]', tmp_path / "x")


def test_independent_positive_and_negative_decisions_match_sealed_result() -> None:
    sealed = verifier.read_json(ROOT / verifier.RESULT)
    _, candidate, _, locked, _ = verifier.fixture()
    assert verifier.independent_decision(candidate) == sealed["positive_decision"]
    assert verifier.independent_decision(locked) == sealed["negative_decision"]
    assert sealed["positive_decision"]["passed"] is True
    assert sealed["negative_decision"]["passed"] is False


def test_independent_block_bootstrap_is_seeded_and_uses_plus_one() -> None:
    values = np.full(48, -.42)
    first = verifier.block_inference(values, 20_260_911)
    second = verifier.block_inference(values, 20_260_911)
    assert first == second
    assert first["one_sided_lower_pvalue"] == 1 / 1001
    assert first["simultaneous_upper"] == pytest.approx(-.42)


def test_holm_is_independently_reconstructed() -> None:
    adjusted = verifier.holm({"b": .03, "a": .01, "c": .04})
    assert adjusted == {"a": (.03, True), "b": (.06, False), "c": (.06, False)}


def test_calibration_decomposition_purge_and_crps_oracles() -> None:
    calibration = verifier.calibration_oracle()
    assert calibration["passed"] is True
    assert calibration["class_mean_residual"] == {
        "favorable_first": 0.0, "adverse_first": 0.0, "no_touch": 0.0,
    }
    decomposition = verifier.brier_decomposition_oracle()
    assert all(np.isfinite(value) and value >= 0 for value in decomposition.values())
    assert verifier.purge_cutoffs()["f3"] is None
    assert verifier.crps_quadratic([-1., 0., 2.], [1., 2., 4.], .5) == pytest.approx(
        59 / 98, abs=1e-15,
    )


def test_producer_result_self_digest_and_runtime_are_verified() -> None:
    value = verifier.verify_producer(ROOT)
    sealed = value["producer"]
    state = {key: item for key, item in sealed.items()
             if key not in {"result_digest", "created_at"}}
    assert sealed["result_digest"] == verifier.stable(state)
    assert value["producer_file_sha256"] == sha256((ROOT / verifier.RESULT).read_bytes()).hexdigest()


def test_verify_producer_rejects_claim_escalation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = verifier.snapshot
    sealed = verifier.read_json(ROOT / verifier.RESULT)
    sealed["trading_claim_authorized"] = True
    forged = (json.dumps(sealed, sort_keys=True) + "\n").encode()

    def snapshot(path: Path) -> bytes:
        return forged if path == ROOT / verifier.RESULT else original(path)

    monkeypatch.setattr(verifier, "snapshot", snapshot)
    with pytest.raises(verifier.VerificationError):
        verifier.verify_producer(ROOT)


def test_atomic_publish_is_create_only(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    verifier.atomic_publish(output, {"passed": True})
    with pytest.raises(verifier.VerificationError, match="create-only"):
        verifier.atomic_publish(output, {"passed": True})


def test_verification_state_is_deterministic_and_keeps_claims_false() -> None:
    verified = verifier.verify_producer(ROOT)
    hashes = {name: "0" * 64 for name in verifier.VERIFIER_RUNTIME}
    first = verifier.verification_state(ROOT, verified, "1" * 40, hashes)
    second = verifier.verification_state(ROOT, verified, "1" * 40, hashes)
    assert first == second
    assert first["predictive_claim_authorized"] is False
    assert first["production_promotion_authorized"] is False
    assert first["trading_claim_authorized"] is False


def test_validate_rejects_verification_digest_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    forged = {
        "schema_version": verifier.SCHEMA,
        "verification_digest": "0" * 64,
        "created_at": "2026-09-11T00:00:00+00:00",
    }
    path = tmp_path / "VERIFIED.json"
    path.write_text(json.dumps(forged))
    monkeypatch.setattr(verifier, "OUTPUT", path.relative_to(tmp_path))
    with pytest.raises(verifier.VerificationError, match="digest"):
        verifier.validate(tmp_path)
