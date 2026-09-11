from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from experiments.m04r import verify_m04r15_m2_availability_preflight as verifier


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
    assert not any("m04r15_m2_availability_preflight" in name for name in imports)


def test_independent_reconstruction_verifies_blocked_result() -> None:
    result, evidence = verifier.verify_result(ROOT)
    rebuilt = evidence["reconstruction"]
    assert result["status"] == rebuilt["status"] == "source_extension_required"
    assert result["schedule"] == rebuilt["schedule"]
    assert result["blocking_reasons"] == rebuilt["blocking_reasons"]
    assert result["registry_creation_authorized"] is False
    assert evidence["metadata"]["source_values_opened"] is False


def test_metadata_mutation_changes_independent_reconstruction() -> None:
    result, evidence = verifier.verify_result(ROOT)
    changed = copy.deepcopy(evidence["metadata"])
    changed["stock_metadata"][0]["last_date"] = "2099-12-31"
    rebuilt = verifier.independent_reconstruction(
        result, changed, evidence["contract"],
        verifier.decode(verifier.snapshot(ROOT / verifier.M1_RESULT), ROOT / verifier.M1_RESULT),
    )
    assert rebuilt["source_metadata"]["stock_metadata_digest"] != result["source_metadata"][
        "stock_metadata_digest"
    ]


def test_resealed_claim_escalation_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    original = verifier.snapshot
    result = json.loads((ROOT / verifier.RESULT).read_text())
    result["predictive_claim_authorized"] = True
    state = {key: value for key, value in result.items()
             if key not in {"result_digest", "created_at"}}
    result["result_digest"] = verifier.stable(state)
    forged = json.dumps(result).encode()
    monkeypatch.setattr(
        verifier, "snapshot",
        lambda path: forged if path == ROOT / verifier.RESULT else original(path),
    )
    with pytest.raises(verifier.VerificationError, match="claim boundary"):
        verifier.verify_result(ROOT)


def test_strict_json_and_create_only_publication(tmp_path: Path) -> None:
    with pytest.raises(verifier.VerificationError, match="duplicate"):
        verifier.decode(b'{"a":1,"a":2}', tmp_path / "x")
    with pytest.raises(verifier.VerificationError, match="nonfinite"):
        verifier.decode(b'{"a":NaN}', tmp_path / "x")
    path = tmp_path / "VERIFIED.json"
    verifier.publish(path, {"passed": True})
    with pytest.raises(verifier.VerificationError, match="create-only"):
        verifier.publish(path, {"passed": True})
