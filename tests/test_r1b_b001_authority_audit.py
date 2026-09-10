from __future__ import annotations

import inspect
import json
from pathlib import Path
import subprocess

import pandas as pd
import pytest

from experiments.m04r import m04r14_r1b_b001_authority_audit as audit
from market_analogues.types import stable_hash


def test_strict_json_rejects_duplicate_nonfinite_and_wrong_shape(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"; duplicate.write_text('{"x":1,"x":2}')
    with pytest.raises(audit.B001AuthorityError, match="duplicate JSON"):
        audit._load(duplicate)
    nonfinite = tmp_path / "nonfinite.json"; nonfinite.write_text('{"x":NaN}')
    with pytest.raises(audit.B001AuthorityError, match="non-finite JSON"):
        audit._load(nonfinite)
    wrong = tmp_path / "wrong.json"; wrong.write_text("[]")
    with pytest.raises(audit.B001AuthorityError, match="JSON dict required"):
        audit._load(wrong)


def test_dynamic_path_allows_only_frozen_d1_roots(tmp_path: Path) -> None:
    allowed = tmp_path / audit.SOURCE_ROOTS[0] / "q.json"
    allowed.parent.mkdir(parents=True); allowed.write_text("{}")
    assert audit._authorized_dynamic_path(tmp_path, allowed.relative_to(tmp_path).as_posix()) == allowed
    outside = tmp_path / "elsewhere" / "q.json"; outside.parent.mkdir(); outside.write_text("{}")
    with pytest.raises(audit.B001AuthorityError, match="outside permitted"):
        audit._authorized_dynamic_path(tmp_path, outside.relative_to(tmp_path).as_posix())
    forbidden = tmp_path / "config/data/analogues/m04r14/t14-10-wf03d-outcome-store-v1/x.json"
    forbidden.parent.mkdir(parents=True); forbidden.write_text("{}")
    with pytest.raises(audit.B001AuthorityError, match="forbidden authority"):
        audit._authorized_dynamic_path(tmp_path, forbidden.relative_to(tmp_path).as_posix())
    with pytest.raises(audit.B001AuthorityError, match="unsafe dynamic"):
        audit._authorized_dynamic_path(tmp_path, "../escape.json")
    linked = tmp_path / audit.SOURCE_ROOTS[0] / "linked.json"
    linked.symlink_to(allowed)
    with pytest.raises(audit.B001AuthorityError, match="absent or linked"):
        audit._authorized_dynamic_path(tmp_path, linked.relative_to(tmp_path).as_posix())


def test_semantic_digest_is_order_and_null_sensitive() -> None:
    rows = [{"b": 2, "a": 1}, {"a": None, "b": 3}]
    assert audit._semantic_digest(rows) == audit._semantic_digest([
        {"a": 1, "b": 2}, {"b": 3, "a": None},
    ])
    assert audit._semantic_digest(rows) != audit._semantic_digest(list(reversed(rows)))
    assert audit._semantic_digest(rows) != audit._semantic_digest(rows[:1])


def test_source_match_routes_are_exact() -> None:
    values = [{"episode_id": str(index)} for index in range(20)]
    assert audit._source_matches({"corrected_matches": values}, "composite", True) == values
    assert audit._source_matches({"retrieval": {"matches": values}}, "composite", False) == values
    assert audit._source_matches({"matches": values}, "price_only", False) == values
    assert audit._source_matches({"random_neighbors": values}, "deterministic_random", False) == values
    assert audit._source_matches({"rank_neighbors": values}, "recent_return_volatility", False) == values
    with pytest.raises(audit.B001AuthorityError, match="source matches"):
        audit._source_matches({"matches": values[:19]}, "price_only", False)


def test_claims_are_reuse_only_and_deny_all_scientific_authority() -> None:
    claims = audit._claims()
    assert claims["reuse_authority_only"] is True
    assert claims["scientific_statistics_opened"] is False
    for key, value in claims.items():
        if key != "reuse_authority_only":
            assert value is False


def test_obsolete_authority_substitution_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prereg = {
        "implementation_commit": "h0",
        "selected_authorities": {"d1": "t14-10-wf03d-exclusion-repair-full-v1"},
        "frozen_authorities": audit.FROZEN_AUTHORITIES,
        "frozen_authorities_digest": stable_hash(audit.FROZEN_AUTHORITIES),
        "allowed_file_sha256": {},
        "dynamic_read_manifests": {
            "shadow_case_root": ".", "shadow_cases": [],
            "d1_source_root": ".", "d1_source_artifacts": [],
            "d1_upstream_source_artifacts": [],
        },
    }
    monkeypatch.setattr(audit, "_core_files", lambda _repository: ())
    monkeypatch.setattr(audit, "_case_manifest", lambda _repository: [])
    monkeypatch.setattr(audit.pd, "read_parquet", lambda *_args, **_kwargs: pd.DataFrame({"source_artifact_path": []}))
    monkeypatch.setattr(audit, "_source_manifest", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(audit, "_upstream_source_manifest", lambda *_args: [])
    monkeypatch.setattr(audit, "_contract", lambda *_args, **_kwargs: dict(prereg))
    with pytest.raises(audit.B001AuthorityError, match="obsolete authority"):
        audit._validate_contract(tmp_path, prereg)


def test_contract_mutation_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    expected = {
        "implementation_commit": "h0", "selected_authorities": {"d1": "v2"},
        "frozen_authorities": audit.FROZEN_AUTHORITIES,
        "frozen_authorities_digest": stable_hash(audit.FROZEN_AUTHORITIES),
        "allowed_file_sha256": {},
        "dynamic_read_manifests": {
            "shadow_case_root": ".", "shadow_cases": [],
            "d1_source_root": ".", "d1_source_artifacts": [],
            "d1_upstream_source_artifacts": [],
        },
    }
    monkeypatch.setattr(audit, "_core_files", lambda _repository: ())
    monkeypatch.setattr(audit, "_case_manifest", lambda _repository: [])
    monkeypatch.setattr(audit.pd, "read_parquet", lambda *_args, **_kwargs: pd.DataFrame(
        {"source_artifact_path": []}
    ))
    monkeypatch.setattr(audit, "_source_manifest", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(audit, "_upstream_source_manifest", lambda *_args: [])
    monkeypatch.setattr(audit, "_contract", lambda *_args, **_kwargs: dict(expected))
    monkeypatch.setattr(audit, "_validate_manifest", lambda *_args, **_kwargs: None)
    mutated = {**expected, "inventory": {"shadow_queries": 3269}}
    with pytest.raises(audit.B001AuthorityError, match="frozen authority contract"):
        audit._validate_contract(tmp_path, mutated)


@pytest.mark.parametrize(("lane", "field"), [
    ("r1a", "result_digest"), ("walk_forward", "registry_digest"),
    ("d1_v2", "effective_inventory_digest"), ("d2", "raw_link_digest"),
    ("packed", "generation_id"),
])
def test_frozen_authority_map_is_cryptographically_bound(lane: str, field: str) -> None:
    frozen = json.loads(json.dumps(audit.FROZEN_AUTHORITIES))
    digest = stable_hash(frozen)
    frozen[lane][field] = "0" * 64
    assert stable_hash(frozen) != digest


def test_manifest_mutation_and_unsafe_entry_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "case.json"; path.write_text('{"case":1}')
    manifest = [{"path": path.name, "bytes": path.stat().st_size,
                 "sha256": audit._sha(path)}]
    audit._validate_manifest(tmp_path, manifest, expected_count=1, label="fixture")
    path.write_text('{"case":2}')
    with pytest.raises(audit.B001AuthorityError,
                       match="manifest entry differs|changed across reads"):
        audit._validate_manifest(tmp_path, manifest, expected_count=1, label="fixture")
    unsafe = [{"path": "../case.json", "bytes": 0, "sha256": "0" * 64}]
    with pytest.raises(audit.B001AuthorityError, match="unsafe fixture"):
        audit._validate_manifest(tmp_path, unsafe, expected_count=1, label="fixture")


def test_repaired_lane_provenance_closure_keeps_all_original_cases() -> None:
    upstream = [
        {"path": f"upstream/{name}.json", "bytes": 2, "sha256": name * 64}
        for name in ("a", "b", "c")
    ]
    effective = [
        upstream[0], upstream[2],
        {"path": "repairs/b.json", "bytes": 3, "sha256": "d" * 64},
    ]
    union = audit._source_union_manifest(effective, upstream, expected_count=4)
    assert [row["path"] for row in union] == [
        "repairs/b.json", "upstream/a.json", "upstream/b.json", "upstream/c.json",
    ]
    assert "upstream/b.json" in {row["path"] for row in union}
    with pytest.raises(audit.B001AuthorityError, match="union count"):
        audit._source_union_manifest(effective, upstream, expected_count=3)


def _result_payload() -> dict[str, object]:
    return {
        "passed": True, "producer_gate_passed": True,
        "status": "producer_gate_pass_pending_independent_verification",
        "result_digest": "a" * 64,
        "inventory": {
            "shadow_queries": 3270, "shadow_links": 65400,
            "walk_forward_queries": 3936, "effective_links": 314880,
            "packed_candidate_episodes": 3786156,
        },
        "gates": {"outcome_blind": True},
    }


def test_atomic_publication_is_create_only(tmp_path: Path) -> None:
    output = tmp_path / "result"
    payload = _result_payload()
    audit._publish(output, payload)
    assert {path.name for path in output.iterdir()} == {"RESULT.json", "report.html"}
    assert json.loads((output / "RESULT.json").read_text()) == payload
    first_report = (output / "report.html").read_bytes()
    assert b"verification pending" in first_report
    assert b"No outcome, prediction" in first_report
    assert audit._report(payload).encode() == first_report
    with pytest.raises(audit.B001AuthorityError, match="create-only"):
        audit._publish(output, {**payload, "passed": False})
    assert json.loads((output / "RESULT.json").read_text()) == payload
    assert (output / "report.html").read_bytes() == first_report


def test_atomic_rename_is_no_replace(tmp_path: Path) -> None:
    source = tmp_path / "source"; source.mkdir(); (source / "x").write_text("new")
    target = tmp_path / "target"; target.mkdir(); (target / "x").write_text("old")
    with pytest.raises(audit.B001AuthorityError, match="create-only"):
        audit._rename_noreplace(source, target)
    assert (target / "x").read_text() == "old"
    assert (source / "x").read_text() == "new"


def test_actual_opened_path_manifest_has_exact_closure(tmp_path: Path) -> None:
    audit._OPENED_PATHS.clear()
    fixed = tmp_path / "fixed.json"; fixed.write_text("{}")
    runtime = tmp_path / "runtime.py"; runtime.write_text("pass\n")
    case = tmp_path / audit.SHADOW_CASES.parent / "cases/case.json"
    case.parent.mkdir(parents=True); case.write_text("{}")
    source = tmp_path / "source.json"; source.write_text("{}")
    prereg_path = tmp_path / audit.PREREGISTRATION
    prereg_path.parent.mkdir(parents=True); prereg_path.write_text("{}")
    prereg = {
        "allowed_file_sha256": {"fixed.json": audit._sha(fixed)},
        "runtime_sha256": {"runtime.py": audit._sha(runtime)},
        "dynamic_read_manifests": {
            "shadow_cases": [{"path": "cases/case.json", "bytes": case.stat().st_size,
                              "sha256": audit._sha(case)}],
            "d1_source_artifacts": [{"path": "source.json", "bytes": source.stat().st_size,
                                     "sha256": audit._sha(source)}],
            "d1_upstream_source_artifacts": [{
                "path": "source.json", "bytes": source.stat().st_size,
                "sha256": audit._sha(source),
            }],
        },
        "d1_source_union_artifact_count": 1,
    }
    prereg["d1_source_union_manifest_digest"] = stable_hash(
        prereg["dynamic_read_manifests"]["d1_source_artifacts"]
    )
    audit._load(prereg_path)
    manifest = audit._opened_path_manifest(tmp_path, prereg)
    assert len(manifest) == 5
    assert stable_hash(manifest) == stable_hash(sorted(manifest, key=lambda row: row["path"]))
    extra = tmp_path / "unapproved.json"; extra.write_text("{}")
    audit._sha(extra)
    with pytest.raises(audit.B001AuthorityError, match="path closure"):
        audit._opened_path_manifest(tmp_path, prereg)


def test_final_payload_semantics_remain_pending_and_non_scientific() -> None:
    prereg = {
        "preregistration_digest": "p", "frozen_authorities_digest": "f",
        "inventory": {}, "allowed_file_sha256": {}, "dynamic_read_roots": [],
        "shadow_case_manifest_digest": "s", "d1_source_manifest_digest": "d",
        "d1_upstream_source_manifest_digest": "u",
        "d1_source_union_manifest_digest": "x",
    }
    opened = [{"path": "authority.json", "bytes": 2, "sha256": "a" * 64}]
    state = audit._result_state(prereg, "h1", {"authority": "digest"},
                                {"outcome_blind": True}, opened)
    assert state["status"] == "producer_gate_pass_pending_independent_verification"
    assert state["producer_gate_passed"] is True and state["passed"] is True
    assert state["independent_verification_complete"] is False
    assert state["b001_complete"] is False and state["later_stage_authorized"] is False
    assert state["opened_path_manifest_digest"] == stable_hash(opened)
    assert state["reuse_authority_only"] is True
    assert state["scientific_statistics_opened"] is False
    assert state["predictive_claim_authorized"] is False
    with pytest.raises(audit.B001AuthorityError, match="gate did not pass"):
        audit._result_state(prereg, "h1", {}, {"outcome_blind": False}, opened)


def _git(root: Path, *args: str) -> None:
    subprocess.run(("git", *args), cwd=root, check=True, capture_output=True)


def test_h1_requires_clean_sole_child_preregistration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _git(tmp_path, "init", "-q"); _git(tmp_path, "config", "user.email", "x@example.invalid")
    _git(tmp_path, "config", "user.name", "Audit")
    for relative in audit.RUNTIME:
        path = tmp_path / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(relative)
    _git(tmp_path, "add", "."); _git(tmp_path, "commit", "-qm", "H0")
    h0 = subprocess.run(("git", "rev-parse", "HEAD"), cwd=tmp_path, check=True,
                        capture_output=True, text=True).stdout.strip()
    prereg = {
        "implementation_commit": h0,
        "runtime_sha256": {relative: audit._sha(tmp_path / relative) for relative in audit.RUNTIME},
    }
    state = dict(prereg); prereg["preregistration_digest"] = stable_hash(state)
    path = tmp_path / audit.PREREGISTRATION; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prereg, sort_keys=True))
    _git(tmp_path, "add", str(audit.PREREGISTRATION)); _git(tmp_path, "commit", "-qm", "H1")
    assert audit._validate_h1(tmp_path, prereg) == subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=tmp_path, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    (tmp_path / audit.RUNTIME[0]).write_text("mutation")
    with pytest.raises(audit.B001AuthorityError, match="clean committed"):
        audit._validate_h1(tmp_path, prereg)


def test_h1_rejects_non_sole_child_commit(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q"); _git(tmp_path, "config", "user.email", "x@example.invalid")
    _git(tmp_path, "config", "user.name", "Audit")
    for relative in audit.RUNTIME:
        path = tmp_path / relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    _git(tmp_path, "add", "."); _git(tmp_path, "commit", "-qm", "H0")
    h0 = subprocess.run(("git", "rev-parse", "HEAD"), cwd=tmp_path, check=True,
                        capture_output=True, text=True).stdout.strip()
    prereg = {"implementation_commit": h0,
              "runtime_sha256": {path: audit._sha(tmp_path / path) for path in audit.RUNTIME}}
    prereg["preregistration_digest"] = stable_hash(prereg)
    target = tmp_path / audit.PREREGISTRATION; target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(prereg, sort_keys=True))
    extra = tmp_path / "extra.txt"; extra.write_text("not sole")
    _git(tmp_path, "add", "."); _git(tmp_path, "commit", "-qm", "bad H1")
    with pytest.raises(audit.B001AuthorityError, match="sole-child"):
        audit._validate_h1(tmp_path, prereg)


def test_module_has_no_forbidden_scientific_data_imports() -> None:
    source = inspect.getsource(audit)
    for forbidden in (
        "market_analogues.causal_outcomes", "market_analogues.evidence_cards",
        "m04r14_t14_11", "m04r14_t14_12", "wf03d_prediction", "wf04_",
    ):
        assert f"import {forbidden}" not in source and f"from {forbidden}" not in source
