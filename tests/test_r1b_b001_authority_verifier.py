from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from experiments.m04r import m04r14_r1b_b001_authority_verifier as v


@pytest.mark.parametrize("content, message", [
    ('{"x":1,"x":2}', "duplicate JSON"), ('{"x":NaN}', "nonfinite JSON"),
    ('[]', "JSON object"), ('{"x":', "invalid JSON"),
])
def test_strict_json(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "value.json"; path.write_text(content)
    with pytest.raises(v.VerificationError, match=message):
        v.load(path)


def test_frozen_contract_rejects_self_consistent_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    value = {"frozen_authorities": {"d1": "v2"}}
    value["frozen_authorities_digest"] = v.digest(value["frozen_authorities"])
    value["preregistration_digest"] = v.digest(value)
    monkeypatch.setattr(v, "PREREG_DIGEST", value["preregistration_digest"])
    v.check_prereg(value)
    altered = deepcopy(value); altered["frozen_authorities"]["d1"] = "v1"
    altered["frozen_authorities_digest"] = v.digest(altered["frozen_authorities"])
    altered["preregistration_digest"] = v.digest(v.without(altered, "preregistration_digest"))
    with pytest.raises(v.VerificationError, match="frozen preregistration"):
        v.check_prereg(altered)


@pytest.mark.parametrize("key,replacement", [
    ("query_count", 3935), ("query_count", 3936.0), ("manifest_digest", "other"),
    ("outcomes_or_labels_used", True), ("outcomes_or_labels_used", 0),
    ("production_promotion_authorized", True),
])
def test_authority_semantic_fields_reject_count_lineage_claim_and_type_mutations(key, replacement) -> None:
    expected = {"query_count": 3936, "manifest_digest": "sealed", "outcomes_or_labels_used": False,
                "production_promotion_authorized": False}
    v.fields(expected, expected, "fixture")
    changed = {**expected, key: replacement}
    with pytest.raises(v.VerificationError, match="field binding"):
        v.fields(changed, expected, "fixture")
    missing = {k: value for k, value in expected.items() if k != key}
    with pytest.raises(v.VerificationError, match="field binding"):
        v.fields(missing, expected, "fixture")


@pytest.mark.parametrize("gates", [{}, {"a": False}, {"a": 1}, [], None])
def test_receipt_gates_require_nonempty_strict_true(gates) -> None:
    v.receipt_gates({"gates": {"a": True}}, "fixture")
    with pytest.raises(v.VerificationError, match="receipt gates"):
        v.receipt_gates({"gates": gates}, "fixture")


def test_paths_reject_escape_and_symlink_ancestor(tmp_path: Path) -> None:
    normal = tmp_path / "normal"; normal.mkdir(); (normal / "item").write_text("x")
    assert v.safe_path(tmp_path, "normal/item") == normal / "item"
    linked = tmp_path / "linked"; linked.symlink_to(normal, target_is_directory=True)
    for relative in ("../escape", "/tmp/escape", "linked/item", "normal/../normal/item"):
        with pytest.raises(v.VerificationError):
            v.safe_path(tmp_path, relative)


def _query() -> dict:
    return {"episode_id": "q", "case_id": "c", "symbol": "QUERY", "cutoff": "2026-03-30T00:00:00",
            "fold_id": "f", "fold_role": "development"}


def _match() -> dict:
    return {"episode_id": "a" * 24, "symbol": "OLD", "cutoff": "2020-01-01T00:00:00",
            "total_distance": 0.25}


def test_link_reconstruction_binds_every_provenance_column() -> None:
    query, match = _query(), _match()
    lane = {"method": "composite", "kind": "original", "effective_matches_digest": "md"}
    row = v.link_record(query, lane, match, 1, ("OLD", "2020-01-01T00:00:00", "A"),
                        "src.json", "sha", "semantic", 1609459200000000000)
    assert tuple(row) == v.LINK_COLUMNS
    assert row == {
        "query_id": "q", "query_case_id": "c", "query_symbol": "QUERY",
        "query_cutoff": "2026-03-30T00:00:00", "fold_id": "f", "fold_role": "development",
        "method": "composite", "rank": 1, "matched_episode_id": "a" * 24,
        "matched_symbol": "OLD", "matched_cutoff": "2020-01-01T00:00:00", "quality_tier": "A",
        "distance_hex": "0x1.0000000000000p-2", "latest_eligible_ns": 1609459200000000000,
        "match_digest": v.digest(match), "effective_matches_digest": "md", "resolution_kind": "original",
        "source_artifact_path": "src.json", "source_artifact_sha256": "sha", "source_artifact_digest": "semantic",
    }
    for field in v.LINK_COLUMNS:
        altered = deepcopy(row); altered[field] = "changed"
        assert v.table_digest([altered]) != v.table_digest([row]), field


@pytest.mark.parametrize("change,identity,latest,message", [
    ({"symbol": "QUERY"}, ("QUERY", "2020-01-01T00:00:00", "A"), 1609459200000000000, "symbol/exclusion"),
    ({}, ("WRONG", "2020-01-01T00:00:00", "A"), 1609459200000000000, "symbol/exclusion"),
    ({}, ("OLD", "2020-01-02T00:00:00", "A"), 1609459200000000000, "source cutoff"),
    ({}, ("OLD", "2020-01-01T00:00:00", "A"), 1, "causal cutoff"),
    ({"total_distance": float("inf")}, ("OLD", "2020-01-01T00:00:00", "A"), 1609459200000000000, "distance"),
    ({"total_distance": -1}, ("OLD", "2020-01-01T00:00:00", "A"), 1609459200000000000, "distance"),
])
def test_link_mutations_fail(change, identity, latest, message) -> None:
    lane = {"method": "composite", "kind": "original", "effective_matches_digest": "md"}
    with pytest.raises(v.VerificationError, match=message):
        v.link_record(_query(), lane, {**_match(), **change}, 1, identity, "src", "sha", "digest", latest)


def test_all_source_routes_and_unknown_method() -> None:
    matches = [{"episode_id": str(i)} for i in range(20)]
    for method, payload in (
        ("composite", {"retrieval": {"matches": matches}}), ("price_only", {"matches": matches}),
        ("deterministic_random", {"random_neighbors": matches}), ("recent_return_volatility", {"rank_neighbors": matches}),
    ):
        assert v.extract_matches(payload, method, False) == matches
        assert v.extract_matches({"corrected_matches": matches}, method, True) == matches
    with pytest.raises(v.VerificationError, match="unknown"):
        v.extract_matches({}, "other", False)
    with pytest.raises(v.VerificationError, match="inventory"):
        v.extract_matches({"matches": matches[:19]}, "price_only", False)


def _records() -> np.ndarray:
    rows = np.zeros(2, dtype=v.PACK_DTYPE)
    rows["episode_id"] = [np.void(bytes.fromhex("01" * 12)), np.void(bytes.fromhex("02" * 12))]
    rows["symbol_id"] = [0, 1]; rows["cutoff_ns"] = [1, 2]; rows["quality_tier"] = [1, 2]
    return rows


@pytest.mark.parametrize("field,value,message", [
    ("symbol_id", 3, "metadata"), ("quality_tier", 0, "metadata"),
    ("error_radii", -1, "numerical"), ("coarse", float("nan"), "numerical"),
])
def test_independent_packed_binary_record_mutations(field, value, message) -> None:
    rows = _records(); v.check_records(rows, 2, overflow=False)
    rows[field][0] = value
    with pytest.raises(v.VerificationError, match=message):
        v.check_records(rows, 2, overflow=False)


def test_packed_order_and_lookup_identity() -> None:
    rows = _records()
    columns = {key: rows[key] for key in ("episode_id", "symbol_id", "cutoff_ns", "quality_tier")}
    columns["symbols"] = ["A", "B"]
    assert v.packed_lookup(columns, ["02" * 12]) == {"02" * 12: ("B", "1970-01-01T00:00:00.000000002", "B")}
    with pytest.raises(v.VerificationError, match="absent"):
        v.packed_lookup(columns, ["03" * 12])
    with pytest.raises(v.VerificationError, match="absent"):
        v.packed_lookup(columns, ["00" * 12])
    with pytest.raises(v.VerificationError, match="order"):
        v.check_records(rows[::-1], 2, overflow=False)


def test_table_digest_order_null_and_chunk_boundaries() -> None:
    rows = [{"x": i, "y": None} for i in range(16385)]
    assert v.table_digest(rows) == v.table_digest([{ "y": None, "x": i} for i in range(16385)])
    for changed in (rows[::-1], rows[:-1], [*rows[:-1], {"x": 16384, "y": 0}]):
        assert v.table_digest(rows) != v.table_digest(changed)


def _state() -> dict:
    prereg = {"allowed_file_sha256": {}, "dynamic_read_roots": [],
              "shadow_case_manifest_digest": "s", "d1_source_manifest_digest": "d",
              "d1_upstream_source_manifest_digest": "u", "d1_source_union_manifest_digest": "union",
              "preregistration_digest": "p", "frozen_authorities_digest": "f",
              "inventory": {"shadow_queries": 3270, "shadow_links": 65400,
                            "walk_forward_queries": 3936, "effective_links": 314880,
                            "packed_candidate_episodes": 3786156}}
    return v.result_state(prereg, "h1", {"source": "a"}, [{"path": "x", "sha256": "sha", "bytes": 2}])


@pytest.mark.parametrize("key", [
    "b001_complete", "later_stage_authorized", "independent_verification_complete",
    *v.DENIED,
])
def test_self_consistent_producer_claim_mutation_rejected(key: str) -> None:
    expected = _state()
    actual = {**deepcopy(expected), "created_at": "2026-09-10T22:28:45+00:00", "result_digest": v.digest(expected)}
    v.compare_result(actual, expected)
    actual[key] = True
    actual["result_digest"] = v.digest(v.without(actual, "created_at", "result_digest"))
    with pytest.raises(v.VerificationError, match="reconstruction"):
        v.compare_result(actual, expected)


@pytest.mark.parametrize("field", ["authority_digests", "inventory", "opened_path_manifest", "gates"])
def test_resigned_result_content_mutation_rejected(field: str) -> None:
    expected = _state()
    actual = deepcopy(expected)
    actual[field] = [] if field == "opened_path_manifest" else {}
    actual["result_digest"] = v.digest(actual)
    actual["created_at"] = "2026-09-10T22:28:45+00:00"
    with pytest.raises(v.VerificationError, match="reconstruction"):
        v.compare_result(actual, expected)


def test_report_is_exact_and_states_pending() -> None:
    payload = _state(); payload["result_digest"] = v.digest(payload)
    html = v.report(payload)
    assert "verification pending" in html and "65,400" in html and "314,880" in html
    assert html.endswith("</body></html>\n")
    payload["inventory"]["shadow_links"] -= 1
    assert v.report(payload) != html


def test_report_mutation_and_symlink_rejected(tmp_path: Path) -> None:
    payload = _state(); payload["result_digest"] = v.digest(payload)
    path = tmp_path / "report.html"; path.write_text(v.report(payload))
    v.compare_report(path, payload)
    path.write_text(path.read_text().replace("verification pending", "complete"))
    with pytest.raises(v.VerificationError, match="HTML report differs"):
        v.compare_report(path, payload)
    alias = tmp_path / "alias.html"; alias.symlink_to(path)
    with pytest.raises(v.VerificationError, match="regular producer report"):
        v.compare_report(alias, payload)


def test_sealed_directory_identity_and_extra_artifact_rejected(tmp_path: Path) -> None:
    folder = tmp_path / "sealed"; folder.mkdir()
    path = folder / "input.json"; path.write_text('{"x":1}')
    files = [{"path": path.name, "bytes": path.stat().st_size, "sha256": v.file_sha(path)}]
    state = {"files": files, "manifest_digest": v.digest(files)}
    seal = {**state, "seal_digest": v.digest(state), "created_at": "now"}
    (folder / "SEALED.json").write_text(json.dumps(seal))
    assert v.sealed(tmp_path, Path("sealed")) == seal
    path.write_text('{"x":2}')
    with pytest.raises(v.VerificationError, match="manifest differs"):
        v.sealed(tmp_path, Path("sealed"))
    (folder / "extra.json").write_text("{}")
    with pytest.raises(v.VerificationError, match="directory closure"):
        v.sealed(tmp_path, Path("sealed"))


def test_publication_is_create_only_and_complete(tmp_path: Path) -> None:
    path = tmp_path / "receipt/VERIFIED.json"
    v.publish(path, {"passed": True, "x": [1, 2]})
    original = path.read_bytes()
    assert v.load(path) == {"passed": True, "x": [1, 2]}
    with pytest.raises(v.VerificationError, match="already exists"):
        v.publish(path, {"passed": False})
    assert path.read_bytes() == original
    assert [p.name for p in path.parent.iterdir()] == ["VERIFIED.json"]


def test_no_producer_or_shared_validation_imports() -> None:
    tree = ast.parse(Path(v.__file__).read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(name.startswith(("market_analogues", "experiments", "tests")) for name in imports)
    assert "eval(" not in Path(v.__file__).read_text() and "exec(" not in Path(v.__file__).read_text()


def test_h0_h1_current_runtime_and_sole_file_lineage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True).stdout.decode().strip()
    git("init"); git("config", "user.email", "test@example.invalid"); git("config", "user.name", "Test")
    (tmp_path / "producer.py").write_text("a = 1\n")
    git("add", "."); git("commit", "-m", "H0")
    h0 = git("rev-parse", "HEAD")
    monkeypatch.setattr(v, "PREREG", Path("prereg.json"))
    monkeypatch.setattr(v, "RUNTIME", ("verifier.py",))
    prereg = {"implementation_commit": h0, "runtime_sha256": {"producer.py": v.file_sha(tmp_path / "producer.py")}}
    (tmp_path / "prereg.json").write_text(json.dumps(prereg))
    git("add", "."); git("commit", "-m", "H1")
    h1 = git("rev-parse", "HEAD")
    (tmp_path / "verifier.py").write_text("b = 2\n")
    git("add", "."); git("commit", "-m", "Verifier")
    assert v.lineage(tmp_path, prereg) == (git("rev-parse", "HEAD"), h1)
    (tmp_path / "producer.py").write_text("a = 2\n")
    with pytest.raises(v.VerificationError, match="clean"):
        v.lineage(tmp_path, prereg)
    git("add", "."); git("commit", "-m", "Changed runtime")
    with pytest.raises(v.VerificationError, match="runtime changed"):
        v.lineage(tmp_path, prereg)
