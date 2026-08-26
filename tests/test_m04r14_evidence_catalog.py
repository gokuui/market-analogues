from __future__ import annotations

import copy
import importlib.util
import json
from hashlib import sha256
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY / "experiments/m04r/m04r14_evidence_catalog.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("m04r14_evidence_catalog", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


catalog = _load()


def _fake_runtime_manifest(_repository: Path) -> dict[str, Any]:
    files = {catalog.RUNTIME_FILES[0]: "1" * 64}
    deterministic = {
        "implementation_commit": "2" * 40,
        "files": files,
        "files_digest": catalog._stable_hash(files),
    }
    return {**deterministic, "digest": catalog._stable_hash(deterministic)}


def _build(repository: Path = REPOSITORY, **kwargs: Any) -> dict[str, Any]:
    return catalog.build_catalog(
        repository, runtime_manifest_loader=_fake_runtime_manifest, **kwargs,
    )


def _validate(payload: dict[str, Any], repository: Path = REPOSITORY, **kwargs: Any) -> None:
    catalog.validate_catalog(
        payload, repository, runtime_manifest_loader=_fake_runtime_manifest, **kwargs,
        stored_runtime_validator=lambda _root, value: (
            catalog._validate_runtime_manifest(value)
        ),
    )


@pytest.fixture(scope="module")
def real_catalog() -> dict[str, Any]:
    return _build()


def test_real_catalog_has_exact_active_superseded_claim_boundaries(
    real_catalog: dict[str, Any],
) -> None:
    _validate(real_catalog)
    assert real_catalog["active_run"] == "v2"
    assert real_catalog["superseded_runs"] == ["v1"]
    assert real_catalog["root_order"] == list(catalog.ROOT_ORDER)
    assert set(real_catalog["roots"]) == set(catalog.KNOWN_ROOT_FILES)
    by_version = {row["version"]: row for row in real_catalog["runs"]}
    assert by_version["v1"]["disposition"] == "superseded_pre_open"
    assert by_version["v1"]["results_opened"] is False
    assert by_version["v1"]["terminal_state"] \
        == "truth_blind_producer_sealed_no_comparison_artifact"
    assert by_version["v2"]["disposition"] == "active_terminal_development_evidence"
    assert by_version["v2"]["results_opened"] is True
    for version in ("v1", "v2"):
        claims = by_version[version]["claims"]
        assert claims["proposal_resource_gate_passed"] is True
        assert claims["exact_stage_slo_passed"] is None
        assert claims["end_to_end_slo_passed"] is None
        assert claims["development_only"] is True
        assert claims["production_promotion_authorized"] is False
        assert claims["finite_diagnostic_certificate_digest_equal"] is True
    assert by_version["v1"]["claims"]["authority_trace_digest_equal"] is None
    assert by_version["v2"]["claims"]["authority_semantic_equal"] is True
    assert by_version["v2"]["claims"]["authority_trace_digest_equal"] is False
    assert real_catalog["v2_independent_verification"][
        "deterministic_replay_equal"
    ] is True
    assert real_catalog["production_promotion_authorized"] is False
    assert all(not Path(row["path"]).is_absolute()
               for row in real_catalog["roots"].values())
    assert catalog.OUTPUT_RELATIVE == Path(
        "config/data/analogues/m04r14/evidence-catalog-v1/catalog.json"
    )


def test_catalog_validation_rejects_relabelled_claim(
    real_catalog: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = catalog._without(real_catalog, {"created_at", "catalog_digest"})
    monkeypatch.setattr(
        catalog, "_deterministic_catalog", lambda _root, **_kwargs: expected,
    )
    changed = copy.deepcopy(real_catalog)
    changed["runs"][1]["claims"]["exact_stage_slo_passed"] = True
    changed["catalog_digest"] = catalog._stable_hash(catalog._without(
        changed, {"created_at", "catalog_digest"},
    ))
    with pytest.raises(catalog.CatalogError, match="reconstruction"):
        _validate(changed)


@pytest.mark.parametrize(
    "raw, message",
    (
        (b'{"a":1,"a":2}', "duplicate JSON key"),
        (b'{"value":NaN}', "non-finite JSON value"),
        (b'{"value":Infinity}', "non-finite JSON value"),
        (b'{"nested":[1e999]}', "non-finite JSON value"),
    ),
)
def test_strict_json_rejects_duplicate_and_nonfinite(
    raw: bytes, message: str,
) -> None:
    with pytest.raises(catalog.CatalogError, match=message):
        catalog._strict_json_bytes(raw, "fixture")


def test_create_only_output_preserves_existing_bytes(
    tmp_path: Path, real_catalog: dict[str, Any],
) -> None:
    output = tmp_path / "catalog.json"
    catalog.write_catalog(output, real_catalog)
    observed = output.read_bytes()
    assert json.loads(observed)["catalog_digest"] == real_catalog["catalog_digest"]
    with pytest.raises(catalog.CatalogError, match="absent"):
        catalog.write_catalog(output, {"replacement": True})
    assert output.read_bytes() == observed


def test_output_cannot_overlap_immutable_evidence(
    tmp_path: Path, real_catalog: dict[str, Any],
) -> None:
    evidence = tmp_path / "immutable"
    evidence.mkdir()
    with pytest.raises(catalog.CatalogError, match="overlaps"):
        catalog.write_catalog(
            evidence / "new-catalog.json", real_catalog,
            protected_roots=(evidence,),
        )
    assert list(evidence.iterdir()) == []


def test_frozen_file_hash_drift_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    changed = copy.deepcopy(catalog.KNOWN_ROOT_FILES)
    changed["finite-threshold-diagnostic-v1"]["DIAGNOSTIC.json"] = "0" * 64
    with pytest.raises(catalog.CatalogError, match="immutable evidence hash"):
        _build(known_root_files=changed)


def _copy_evidence(tmp_path: Path) -> tuple[Path, dict[str, dict[str, str]]]:
    repository = tmp_path / "repository"
    target = repository / catalog.EVIDENCE_RELATIVE
    target.parent.mkdir(parents=True)
    shutil.copytree(REPOSITORY / catalog.EVIDENCE_RELATIVE, target)
    return repository, copy.deepcopy(catalog.KNOWN_ROOT_FILES)


def _rewrite_json(path: Path, payload: dict[str, Any]) -> str:
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.write_bytes(raw)
    return sha256(raw).hexdigest()


def test_copy_mutate_and_rehash_file_map_cannot_bypass_internal_digest(
    tmp_path: Path,
) -> None:
    repository, known = _copy_evidence(tmp_path)
    relative = Path("finite-threshold-diagnostic-v2/DIAGNOSTIC.json")
    path = repository / catalog.EVIDENCE_RELATIVE / relative
    payload = json.loads(path.read_text())
    payload["cases"][0]["all_thresholds_finite"] = False
    known[relative.parts[0]][relative.parts[1]] = _rewrite_json(path, payload)
    with pytest.raises(catalog.CatalogError, match="producer boundary|diagnostic boundary"):
        _build(repository, known_root_files=known)


def test_copy_mutate_recompute_case_digest_fails_producer_seal_crosslink(
    tmp_path: Path,
) -> None:
    repository, known = _copy_evidence(tmp_path)
    root_name = "threaded-certified-exposed-v2"
    relative = f"cases/00-{catalog.QUERY_IDS[0]}.json"
    path = repository / catalog.EVIDENCE_RELATIVE / root_name / relative
    payload = json.loads(path.read_text())
    payload["metrics"]["case_task_wall_seconds"] += 1.0
    payload["result_digest"] = catalog._stable_hash(catalog._without(
        payload, {"created_at", "result_digest"},
    ))
    known[root_name][relative] = _rewrite_json(path, payload)
    with pytest.raises(catalog.CatalogError, match="producer aggregate"):
        _build(repository, known_root_files=known)


def test_runtime_manifest_rejects_unbound_or_malformed_commit() -> None:
    value = _fake_runtime_manifest(REPOSITORY)
    value["implementation_commit"] = "not-a-commit"
    value["digest"] = catalog._stable_hash(catalog._without(value, {"digest"}))
    with pytest.raises(catalog.CatalogError, match="runtime manifest"):
        catalog._validate_runtime_manifest(value)


def test_stored_h0_manifest_accepts_clean_h1_unrelated_commit_and_rejects_blob_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    runtime = repository / catalog.RUNTIME_FILES[0]
    runtime.parent.mkdir(parents=True)
    runtime.write_text("committed runtime\n")
    subprocess.run(("git", "init", "-q", str(repository)), check=True)
    subprocess.run(
        ("git", "-C", str(repository), "config", "user.email", "test@example.invalid"),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(repository), "config", "user.name", "Catalog Test"),
        check=True,
    )
    subprocess.run(("git", "-C", str(repository), "add", catalog.RUNTIME_FILES[0]), check=True)
    subprocess.run(
        ("git", "-C", str(repository), "commit", "-qm", "runtime"), check=True,
    )
    def lifecycle_reconstruction(
        root: Path, *,
        runtime_manifest_loader: Any = catalog._git_runtime_manifest,
        stored_runtime_manifest: Any = None,
        stored_runtime_validator: Any = catalog._validate_stored_runtime_manifest,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        if stored_runtime_manifest is None:
            runtime_manifest = dict(runtime_manifest_loader(root))
        else:
            runtime_manifest = dict(stored_runtime_manifest)
            stored_runtime_validator(root, runtime_manifest)
        return {"runtime_manifest": runtime_manifest, "sentinel": "stable evidence"}

    monkeypatch.setattr(catalog, "_deterministic_catalog", lifecycle_reconstruction)
    payload = catalog.build_catalog(repository)
    manifest = payload["runtime_manifest"]
    catalog._validate_runtime_manifest(manifest)
    assert manifest["files"][catalog.RUNTIME_FILES[0]] \
        == sha256(b"committed runtime\n").hexdigest()
    unrelated = repository / "unrelated.txt"
    unrelated.write_text("later unrelated commit\n")
    subprocess.run(("git", "-C", str(repository), "add", "unrelated.txt"), check=True)
    subprocess.run(
        ("git", "-C", str(repository), "commit", "-qm", "unrelated H1"), check=True,
    )
    catalog.validate_catalog(payload, repository)

    runtime.write_text("committed runtime drift\n")
    subprocess.run(("git", "-C", str(repository), "add", catalog.RUNTIME_FILES[0]), check=True)
    subprocess.run(
        ("git", "-C", str(repository), "commit", "-qm", "runtime drift"), check=True,
    )
    with pytest.raises(catalog.CatalogError, match="stored runtime blob differs"):
        catalog.validate_catalog(payload, repository)


def test_final_resnapshot_detects_post_validation_evidence_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, known = _copy_evidence(tmp_path)
    path = repository / catalog.EVIDENCE_RELATIVE \
        / "finite-threshold-diagnostic-v1/DIAGNOSTIC.json"
    original = catalog._validate_verifications

    def mutate_after_validation(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original(*args, **kwargs)
        path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(catalog, "_validate_verifications", mutate_after_validation)
    with pytest.raises(catalog.CatalogError, match="changed during catalog build"):
        _build(repository, known_root_files=known)
