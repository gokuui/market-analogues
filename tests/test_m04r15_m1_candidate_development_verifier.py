from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from experiments.m04r import verify_m04r15_m1_candidate_development_gate as verifier


ROOT=Path(__file__).resolve().parents[1]


def test_verifier_has_no_project_or_producer_import()->None:
    tree=ast.parse((ROOT/verifier.VERIFIER_RUNTIME[0]).read_text());imports=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):imports.extend(alias.name for alias in node.names)
        elif isinstance(node,ast.ImportFrom) and node.module:imports.append(node.module)
    assert not any(name.startswith("market_analogues") for name in imports)
    assert not any("m04r15_m1_candidate_development_gate" in name for name in imports)


def test_strict_decoder_refuses_duplicate_nonfinite_and_nonobject(tmp_path:Path)->None:
    with pytest.raises(verifier.VerificationError,match="duplicate"):verifier.decode(b'{"a":1,"a":2}',tmp_path/"x")
    with pytest.raises(verifier.VerificationError,match="nonfinite"):verifier.decode(b'{"a":NaN}',tmp_path/"x")
    with pytest.raises(verifier.VerificationError,match="object"):verifier.decode(b'[]',tmp_path/"x")


def test_independent_reconstruction_matches_frozen_producer()->None:
    result,evidence=verifier.verify_producer(ROOT);rebuilt=evidence["reconstruction"]
    assert result["selected_weights"]==verifier.WEIGHTS
    assert rebuilt["selected"]["metrics"]==result["fold_metrics"]
    assert rebuilt["leave_one_fold_out"]==result["leave_one_fold_out"]
    assert result["minimum_fold_two_comparator_skill"]>.014


def test_all_input_bytes_are_independently_manifest_bound()->None:
    raw,_=verifier.inputs(ROOT);result=json.loads((ROOT/verifier.RESULT).read_text())
    from hashlib import sha256
    assert result["input_sha256"]=={name:sha256(value).hexdigest() for name,value in raw.items()}


def test_publication_is_create_only(tmp_path:Path)->None:
    path=tmp_path/"VERIFIED.json";verifier.publish(path,{"passed":True})
    with pytest.raises(verifier.VerificationError,match="create-only"):verifier.publish(path,{"passed":True})


def test_claim_mutation_is_rejected(monkeypatch:pytest.MonkeyPatch)->None:
    original=verifier.snapshot;result=json.loads((ROOT/verifier.RESULT).read_text());result["trading_claim_authorized"]=True
    state={key:value for key,value in result.items() if key not in {"result_digest","created_at"}}
    result["result_digest"]=verifier.stable(state);forged=json.dumps(result).encode()
    monkeypatch.setattr(verifier,"snapshot",lambda path:forged if path==ROOT/verifier.RESULT else original(path))
    with pytest.raises(verifier.VerificationError,match="claim boundary"):
        verifier.verify_producer(ROOT)


def test_extra_producer_field_is_rejected_even_with_resealed_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original=verifier.snapshot;result=json.loads((ROOT/verifier.RESULT).read_text())
    result["undeclared_field"]=True
    state={key:value for key,value in result.items() if key not in {"result_digest","created_at"}}
    result["result_digest"]=verifier.stable(state);forged=json.dumps(result).encode()
    monkeypatch.setattr(verifier,"snapshot",lambda path:forged if path==ROOT/verifier.RESULT else original(path))
    with pytest.raises(verifier.VerificationError,match="field closure"):
        verifier.verify_producer(ROOT)
