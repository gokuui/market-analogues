from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _module():
    path = (
        Path(__file__).parents[1] / "experiments" / "m04r"
        / "preregister_m04r11_candidate_v2.py"
    )
    spec = importlib.util.spec_from_file_location(
        "preregister_m04r11_candidate_v2", path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _roots(artifact: Path, registry: Path, source: Path) -> dict[str, str]:
    return {
        "registry_root": str(registry.parent.resolve()),
        "source_full_root": str(source.resolve()),
        "resident_full_root": "/dev/shm/resident",
        "candidate_root": str((artifact / "candidate").resolve()),
        "comparison_root": str((artifact / "comparison").resolve()),
        "verification_root": str((artifact / "verification").resolve()),
        "predecessor_candidate_root": str((artifact / "previous-candidate").resolve()),
        "predecessor_comparison_root": str((artifact / "previous-comparison").resolve()),
        "authority_root": str((artifact / "authorities-sealed-v4").resolve()),
    }


def test_create_only_publication_never_replaces_existing_bytes(
    tmp_path: Path,
) -> None:
    module = _module()
    path = tmp_path / "fixed.json"
    assert module._publish_create_only(path, {"value": 1}) is True
    original = path.read_bytes()
    assert module._publish_create_only(path, {"value": 2}) is False
    assert path.read_bytes() == original


def test_generator_is_authority_blind_and_existing_file_is_validation_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    repository = tmp_path / "repository"
    artifact = repository / "config" / "data" / "analogues"
    config_path = repository / module.CONFIG_RELATIVE_PATH
    registry_path = repository / module.REGISTRY_RELATIVE_PATH
    source = artifact / "poc" / "m04r" / "packed-bound-full"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("config")
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(json.dumps({"registry": True}))
    source.mkdir(parents=True)
    roots = _roots(artifact, registry_path, source)
    fixed = repository / "experiments" / "m04r" / "fixed.json"
    reads: list[Path] = []
    original_read_json = module._read_json

    def recording_read_json(path: Path):
        reads.append(path.resolve())
        return original_read_json(path)

    monkeypatch.setattr(
        module, "load_config",
        lambda path: SimpleNamespace(
            artifact_dir=artifact, datasets={"nasdaq": object()},
        ),
    )
    monkeypatch.setattr(module, "expected_roots", lambda path: dict(roots))
    monkeypatch.setattr(module, "validate_exact_roots", lambda observed, path: ())
    monkeypatch.setattr(module, "validate_predecessor_paths", lambda path: ())
    monkeypatch.setattr(module, "validate_role_table", lambda registry: ())
    monkeypatch.setattr(
        module, "validate_m04r_validation_registry", lambda source, path: (),
    )
    monkeypatch.setattr(module, "source_from_spec", lambda spec: object())
    monkeypatch.setattr(module.producer, "_source_pack_binding", lambda path: {"source": 1})
    monkeypatch.setattr(module.producer, "_implementation_manifest", lambda: {"code": 1})
    monkeypatch.setattr(module.producer, "_environment_manifest", lambda: {"env": 1})
    monkeypatch.setattr(module, "expected_preregistration_path", lambda repo: fixed)
    monkeypatch.setattr(module, "_read_json", recording_read_json)
    monkeypatch.setattr(
        module, "build_preregistration_document",
        lambda *args, **kwargs: {"preregistration_digest": "digest"},
    )

    def strict_load(*args, **kwargs):
        reads.append(fixed.resolve())
        return json.loads(fixed.read_text())

    monkeypatch.setattr(module, "load_and_validate_preregistration", strict_load)

    first, created = module.create_or_validate_preregistration(
        repository_root=repository, config_path=config_path,
        registry_path=registry_path, source_full_root=source,
    )
    assert created is True
    original = fixed.read_bytes()
    second, created = module.create_or_validate_preregistration(
        repository_root=repository, config_path=config_path,
        registry_path=registry_path, source_full_root=source,
    )
    assert created is False
    assert second == first
    assert fixed.read_bytes() == original

    authority = Path(roots["authority_root"])
    assert all(path != authority and not path.is_relative_to(authority) for path in reads)
    source_text = inspect.getsource(module)
    assert '"authority_root"' not in source_text
    assert "--authority" not in source_text


def test_generator_rejects_existing_v2_output_before_any_publication(
    tmp_path: Path,
) -> None:
    module = _module()
    roots = {
        "candidate_root": str(tmp_path / "candidate"),
        "comparison_root": str(tmp_path / "comparison"),
        "verification_root": str(tmp_path / "verification"),
    }
    Path(roots["candidate_root"]).mkdir()
    with pytest.raises(ValueError, match="already exists"):
        module._assert_v2_outputs_absent(roots)
