from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from experiments.m04r import verify_m04r15_m2_prospective_listener_synthetic_gate as verifier


ROOT = Path(__file__).resolve().parents[1]


def test_verifier_imports_neither_project_nor_producer() -> None:
    tree = ast.parse((ROOT / verifier.VERIFIER_RUNTIME[0]).read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any(name.startswith("market_analogues") for name in imports)
    assert not any("m04r15_m2_prospective_listener_synthetic_gate" in name
                   for name in imports)


def test_independent_oracle_matches_all_synthetic_checks() -> None:
    result, evidence = verifier.verify_result(ROOT)
    assert result["checks"] == evidence["reconstruction"]["checks"]
    assert result["inventory"] == evidence["reconstruction"]["inventory"]
    assert evidence["reconstruction"]["prediction_seal_digest"] == (
        "bee3515cbbed89d0dae64e96888683e22e7220720a9f441df33b1f9e90c5e624"
    )


def test_resealed_claim_escalation_reaches_claim_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = verifier.snapshot
    result = json.loads((ROOT / verifier.RESULT).read_text())
    result["real_prediction_created"] = True
    state = {key: value for key, value in result.items()
             if key not in {"result_digest", "created_at", "elapsed_seconds"}}
    result["result_digest"] = verifier.stable(state)
    forged = json.dumps(result).encode()
    monkeypatch.setattr(
        verifier, "snapshot",
        lambda path: forged if path == ROOT / verifier.RESULT else original(path),
    )
    with pytest.raises(verifier.VerificationError, match="claim boundary"):
        verifier.verify_result(ROOT)


def test_strict_json_and_create_only_receipt(tmp_path: Path) -> None:
    with pytest.raises(verifier.VerificationError, match="duplicate"):
        verifier.decode(b'{"a":1,"a":2}', tmp_path / "x")
    with pytest.raises(verifier.VerificationError, match="nonfinite"):
        verifier.decode(b'{"a":NaN}', tmp_path / "x")
    path = tmp_path / "VERIFIED.json"
    verifier.publish(path, {"passed": True})
    with pytest.raises(verifier.VerificationError, match="create-only"):
        verifier.publish(path, {"passed": True})
