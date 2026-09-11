from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from experiments.m04r import m04r14_r1b_joint_b005_b2_closure as closure


def test_stable_hash_is_canonical_and_rejects_nonfinite() -> None:
    assert closure.stable({"z": 1, "a": [True]}) == sha256(
        b'{"a":[true],"z":1}').hexdigest()
    with pytest.raises(ValueError):
        closure.stable({"x": float("nan")})


def test_decode_rejects_duplicate_and_nonfinite(tmp_path: Path) -> None:
    with pytest.raises(closure.ClosureError, match="duplicate"):
        closure.decode(b'{"a":1,"a":2}', tmp_path / "x")
    with pytest.raises(closure.ClosureError, match="nonfinite"):
        closure.decode(b'{"a":NaN}', tmp_path / "x")


def test_snapshot_rejects_symlink(tmp_path: Path) -> None:
    value = tmp_path / "value"; value.write_bytes(b"x")
    link = tmp_path / "link"; link.symlink_to(value)
    with pytest.raises(closure.ClosureError, match="unsafe"):
        closure.snapshot(link)


def test_load_bound_refuses_mutation(tmp_path: Path) -> None:
    path = tmp_path / "value.json"; path.write_text('{"a":1}')
    record = {"path": "value.json", "sha256": sha256(path.read_bytes()).hexdigest()}
    assert closure.load_bound(tmp_path, record)[0] == {"a": 1}
    path.write_text('{"a":2}')
    with pytest.raises(closure.ClosureError, match="hash differs"):
        closure.load_bound(tmp_path, record)


def test_receipt_self_digest_and_claim_boundary() -> None:
    state = {
        "passed": True, "status": "verified_structurally_localized",
        "verifier_commit": closure.VERIFIER_COMMIT, "gates": {"a": True},
        "claims": {"predictive_claim_authorized": False,
                   "production_promotion_authorized": False,
                   "ranking_change_authorized": False},
    }
    receipt = {**state, "verification_digest": closure.stable(state), "created_at": "now"}
    closure.verify_receipt(receipt, b005=False)
    receipt["claims"]["predictive_claim_authorized"] = True
    with pytest.raises(closure.ClosureError):
        closure.verify_receipt(receipt, b005=False)


def test_create_only_lock(tmp_path: Path) -> None:
    path = tmp_path / "LOCKED.json"
    closure.publish(path, {"passed": True})
    assert json.loads(path.read_text()) == {"passed": True}
    with pytest.raises(closure.ClosureError, match="already exists"):
        closure.publish(path, {"passed": False})


def test_validate_locked_checks_historical_runtime_and_claims(tmp_path: Path, monkeypatch) -> None:
    evidence = {"x": 1}
    runtime = {path: b"committed-" + path.encode() for path in closure.RUNTIME}
    state = {
        "schema_version": closure.SCHEMA,
        "status": "joint_b005_b2_verified_and_locked",
        "passed": True,
        "closure_commit": "c" * 40,
        "runtime_sha256": {path: sha256(value).hexdigest() for path, value in runtime.items()},
        "supersedes_closure_digest": closure.SUPERSEDED_CLOSURE_DIGEST,
        "evidence": evidence,
        "scope": closure.SCOPE,
        "claims": {
            "structural_localization_verified": True,
            "b005_sensitivity_verified": True,
            "adequacy_labels_authorized": False,
            "predictive_claim_authorized": False,
            "ranking_change_authorized": False,
            "production_promotion_authorized": False,
            "trading_claim_authorized": False,
            "real_forward_outcomes_accessed": False,
        },
    }
    receipt = {**state, "closure_digest": closure.stable(state), "created_at": "now"}
    path = tmp_path / closure.OUTPUT; path.parent.mkdir(parents=True); path.write_text(json.dumps(receipt))
    monkeypatch.setattr(closure, "validate", lambda root: evidence)
    monkeypatch.setattr(closure, "git", lambda root, *args, binary=False:
                        runtime[args[1].split(":", 1)[1]] if args[0] == "show" else "")
    assert closure.validate_locked(tmp_path)["closure_digest"] == receipt["closure_digest"]
    bad_claim = json.loads(json.dumps(receipt))
    bad_claim["claims"]["adequacy_labels_authorized"] = True
    path.write_text(json.dumps(bad_claim))
    with pytest.raises(closure.ClosureError):
        closure.validate_locked(tmp_path)
    for key in ("locked", "deferred_nonblocking_research"):
        changed = json.loads(json.dumps(state))
        changed["scope"][key] = []
        changed_receipt = {**changed, "closure_digest": closure.stable(changed), "created_at": "now"}
        path.write_text(json.dumps(changed_receipt))
        with pytest.raises(closure.ClosureError, match="scope"):
            closure.validate_locked(tmp_path)


def test_real_lock_validates_after_publication() -> None:
    root = Path(__file__).resolve().parents[1]
    if not (root / closure.OUTPUT).exists():
        pytest.skip("v2 create-only lock has not been published yet")
    receipt = closure.validate_locked(root)
    assert receipt["passed"] is True
