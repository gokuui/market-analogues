from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from experiments.m04r import verify_m04r15_m2_prospective_payload_adapter_gate as verifier


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
    assert not any("m04r15_m2_prospective_payload_adapter_gate" in name for name in imports)


def test_independent_reconstruction_matches_every_adapter_digest() -> None:
    result, evidence = verifier.verify_result(ROOT)
    for field in (
        "checks", "inventory", "source_document_digest", "registry_document_digest",
        "prediction_document_digest", "prospective_prediction_seal_digest",
    ):
        assert result[field] == evidence["reconstruction"][field]
    assert result["inventory"]["causal_history_rows"] == 3683


def test_resealed_claim_escalation_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
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
    path = tmp_path / "VERIFIED.json"
    verifier.publish(path, {"passed": True})
    with pytest.raises(verifier.VerificationError, match="create-only"):
        verifier.publish(path, {"passed": True})
