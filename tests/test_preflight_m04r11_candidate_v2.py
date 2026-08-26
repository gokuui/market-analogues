from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _module():
    path = (
        Path(__file__).parents[1] / "experiments" / "m04r"
        / "preflight_m04r11_candidate_v2.py"
    )
    spec = importlib.util.spec_from_file_location(
        "preflight_m04r11_candidate_v2", path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _development_registry(module) -> dict:
    return {
        "schema_version": "m04r-batch-query-registry-v1",
        "passed": True,
        "failures": [],
        "real_forward_outcomes_accessed": False,
        "cases_data": [{
            "case_id": module.CAKE_CASE_ID,
            "episode_id": module.CAKE_EPISODE_ID,
            "symbol": module.CAKE_SYMBOL,
            "cutoff": module.CAKE_CUTOFF,
            "dataset_id": "nasdaq",
            "lookback": 252,
            "representation_version": "dense-v1",
        }],
    }


def test_only_exact_exposed_cake_registry_query_is_permitted(
    tmp_path: Path,
) -> None:
    module = _module()
    repository = tmp_path / "repository"
    registry_path = repository / module.DEVELOPMENT_REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(json.dumps(_development_registry(module)))

    _, case = module._load_development_case(
        repository_root=repository, registry_path=registry_path,
        case_id=module.CAKE_CASE_ID, frozen_query_ids=set(),
    )
    assert case["episode_id"] == module.CAKE_EPISODE_ID
    with pytest.raises(ValueError, match="only exposed CAKE"):
        module._load_development_case(
            repository_root=repository, registry_path=registry_path,
            case_id="nasdaq-other", frozen_query_ids=set(),
        )
    with pytest.raises(ValueError, match="untouched"):
        module._load_development_case(
            repository_root=repository, registry_path=registry_path,
            case_id=module.CAKE_CASE_ID,
            frozen_query_ids={module.CAKE_EPISODE_ID},
        )
    with pytest.raises(ValueError, match="registry path differs"):
        module._load_development_case(
            repository_root=repository, registry_path=tmp_path / "copy.json",
            case_id=module.CAKE_CASE_ID, frozen_query_ids=set(),
        )


def test_preflight_output_must_be_fresh_and_disjoint_from_every_frozen_root(
    tmp_path: Path,
) -> None:
    module = _module()
    roots = {
        "candidate_root": str((tmp_path / "protected" / "candidate").resolve()),
        "authority_root": str((tmp_path / "protected" / "authority").resolve()),
    }
    safe = tmp_path / "poc" / "cake"
    module._assert_poc_output(safe, roots)
    for unsafe in (
        Path(roots["candidate_root"]),
        Path(roots["candidate_root"]) / "child",
        tmp_path / "protected",
    ):
        with pytest.raises(ValueError, match="overlaps"):
            module._assert_poc_output(unsafe, roots)
    safe.mkdir(parents=True)
    (safe / "existing").write_text("no overwrite")
    with pytest.raises(ValueError, match="fresh and empty"):
        module._assert_poc_output(safe, roots)


def _worker_evidence(module):
    ready = {
        "content_digest": "content",
        "file_identity_lease": {"lease_digest": "lease"},
    }
    scans = [{
        "candidate_digest": module.EXPECTED_CANDIDATE_DIGEST,
        "result_digest": module.EXPECTED_SCAN_RESULT_DIGEST,
    } for _ in range(3)]
    semantic = {
        "scan_semantics": scans,
        "candidate_digest_reconstructed": module.EXPECTED_CANDIDATE_DIGEST,
        "gates": {name: True for name in module.SEMANTIC_CASE_GATES},
        "passed": True,
        "real_forward_outcomes_accessed": False,
    }
    semantic["semantic_digest"] = module.semantic_case_digest(semantic)
    performance = {
        "ready_start": ready,
        "ready_end": deepcopy(ready),
        "gates": {name: True for name in module.PERFORMANCE_ATTEMPT_GATES},
        "passed": True,
        "real_forward_outcomes_accessed": False,
    }
    performance["performance_digest"] = module.performance_attempt_digest(performance)
    resident = {
        "resident_ready_observation": deepcopy(ready),
        "resident_content_digest": "content",
    }
    return {"semantic": semantic, "performance": performance}, resident


def test_worker_result_requires_exact_digests_all_gates_and_stable_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setattr(
        module.producer, "_strict_validate_case_evidence",
        lambda *args, **kwargs: None,
    )
    result, resident = _worker_evidence(module)
    semantic, performance = module._validate_worker_result(
        result, contract={}, case={}, context={}, resident=resident,
        physical_rows=1,
    )
    assert semantic["passed"] is True and performance["passed"] is True

    changed = deepcopy(result)
    changed["semantic"]["scan_semantics"][1]["candidate_digest"] = "wrong"
    changed["semantic"]["semantic_digest"] = module.semantic_case_digest(
        changed["semantic"],
    )
    with pytest.raises(ValueError, match="digest, gate or identity"):
        module._validate_worker_result(
            changed, contract={}, case={}, context={}, resident=resident,
            physical_rows=1,
        )

    changed = deepcopy(result)
    changed["performance"]["gates"][module.PERFORMANCE_ATTEMPT_GATES[0]] = False
    changed["performance"]["performance_digest"] = module.performance_attempt_digest(
        changed["performance"],
    )
    with pytest.raises(ValueError, match="digest, gate or identity"):
        module._validate_worker_result(
            changed, contract={}, case={}, context={}, resident=resident,
            physical_rows=1,
        )


def test_git_binding_uses_every_preregistered_implementation_file() -> None:
    module = _module()
    source = Path(module.__file__).read_text()
    assert 'implementation_files=contract["implementation_manifest"]["files"]' in source
    assert 'get_context("spawn")' in source


def test_run_preflight_reaches_git_binding_and_worker_with_frozen_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    repository = tmp_path / "repository"
    config_path = repository / module.CONFIG_RELATIVE_PATH
    development_registry = (
        repository / module.DEVELOPMENT_REGISTRY_RELATIVE_PATH
    )
    artifact_dir = repository / "artifacts"
    source_root = artifact_dir / "source"
    resident_root = artifact_dir / "resident"
    frozen_registry_root = artifact_dir / "registry"
    output_root = artifact_dir / "poc" / "preflight"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("synthetic")
    development_registry.parent.mkdir(parents=True, exist_ok=True)
    development_registry.write_text(json.dumps(_development_registry(module)))
    frozen_registry_root.mkdir(parents=True)
    (frozen_registry_root / "query-registry.json").write_text(
        json.dumps({"registry_digest": "registry", "cases_data": []}),
    )

    roots = {
        "registry_root": str(frozen_registry_root.resolve()),
        "source_full_root": str(source_root.resolve()),
        "resident_full_root": str(resident_root.resolve()),
        "candidate_root": str((artifact_dir / "candidate").resolve()),
        "comparison_root": str((artifact_dir / "comparison").resolve()),
        "verification_root": str((artifact_dir / "verification").resolve()),
        "predecessor_candidate_root": str((artifact_dir / "candidate-v1").resolve()),
        "predecessor_comparison_root": str((artifact_dir / "comparison-v1").resolve()),
        "authority_root": str((artifact_dir / "authority").resolve()),
    }
    implementation = {
        "files": {"experiments/m04r/preflight_m04r11_candidate_v2.py": "hash"},
        "digest": "implementation",
    }
    environment = {"digest": "environment"}
    source_pack = {"provenance_digest": "provenance", "physical_rows": 1}
    contract = {
        "contract_digest": "contract",
        "implementation_manifest": implementation,
        "ordered_query_ids": ["untouched-query"],
        "resident_policy": {"reserve_bytes": 1},
        "route_quotas": {"composite": 1_000},
        "scan_protocol": {},
    }
    preregistration = {
        "producer_contract": contract,
        "preregistration_digest": "preregistration",
    }
    worker_result, resident = _worker_evidence(module)
    resident.update({"binding_digest": "binding"})
    ready = {"schema_version": "ready"}
    validation_observation = {"observation_digest": "observation"}
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        module, "load_config", lambda _path: SimpleNamespace(artifact_dir=artifact_dir),
    )
    monkeypatch.setattr(module, "expected_roots", lambda _artifact: roots)
    monkeypatch.setattr(module.producer, "_source_pack_binding", lambda _root: source_pack)
    monkeypatch.setattr(module.producer, "_implementation_manifest", lambda: implementation)
    monkeypatch.setattr(module.producer, "_environment_manifest", lambda: environment)
    monkeypatch.setattr(
        module, "load_and_validate_preregistration",
        lambda *args, **kwargs: preregistration,
    )

    def bind_git(_repository, _path, *, implementation_files):
        calls["git_files"] = implementation_files
        return {"binding_digest": "git"}

    monkeypatch.setattr(module.producer, "_git_preregistration_binding", bind_git)
    monkeypatch.setattr(
        module, "_augment_case_with_causal_prefixes",
        lambda _config, case: (case, {"query": "context"}),
    )
    monkeypatch.setattr(
        module, "prepare_resident_mirror_observed",
        lambda *args, **kwargs: (ready, validation_observation),
    )
    monkeypatch.setattr(module.producer, "_resident_binding", lambda *args, **kwargs: resident)
    monkeypatch.setattr(module.producer, "_validate_resident_binding", lambda *args: ())
    monkeypatch.setattr(
        module.producer, "_ready_observation",
        lambda _path: resident["resident_ready_observation"],
    )
    monkeypatch.setattr(
        module.producer, "_strict_validate_case_evidence",
        lambda *args, **kwargs: None,
    )

    class Future:
        def result(self):
            return worker_result

    class Executor:
        def __init__(self, **kwargs):
            calls["executor"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, *args):
            calls["worker"] = function
            calls["worker_args"] = args
            return Future()

    monkeypatch.setattr(module, "ProcessPoolExecutor", Executor)

    evidence = module.run_preflight(
        repository_root=repository, config_path=config_path,
        development_registry_path=development_registry,
        case_id=module.CAKE_CASE_ID, source_full_root=source_root,
        resident_root=resident_root, output_root=output_root,
    )

    assert calls["git_files"] == implementation["files"]
    assert calls["worker"] is module.producer._worker
    assert evidence["passed"] is True
    assert json.loads((output_root / "preflight.json").read_text()) == evidence
