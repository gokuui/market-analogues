from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

import pytest

import test_m04r13_threaded_certified_exposed as support


REPOSITORY = Path(__file__).resolve().parents[1]
PATH = REPOSITORY / "experiments/m04r/m04r13_finite_threshold_diagnostic.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("m04r13_finite_diagnostic", PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


diagnostic = _load()
producer = support.producer


def _fixture(tmp_path: Path) -> tuple[
    Path, dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]],
]:
    root = tmp_path / "producer"
    prereg, _authorities = support._producer_fixture(root)
    resident = producer._read_json(root / "RESIDENT.json")
    cases = [producer._read_json(
        root / "cases" / f"{ordinal:02d}-{query_id}.json"
    ) for ordinal, query_id in enumerate(producer.FROZEN_QUERY_IDS)]
    summaries = [diagnostic.summarize_case(case) for case in cases]
    for summary in summaries:
        summary["rounds"][0]["frontier_rows"] = producer.INITIAL_FRONTIER
        summary["result_digest"] = producer.stable_hash(producer._without(
            summary, {"result_digest"},
        ))
    return root, prereg, resident, cases, summaries


def _payload(
    prereg: dict[str, Any], resident: dict[str, Any],
    cases: list[dict[str, Any]], summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    return diagnostic.diagnostic_payload(
        repository=REPOSITORY, implementation_git=prereg["git"],
        registry_digest=producer.REGISTRY_DIGEST,
        registry_cases_digest=prereg["registry_cases_digest"],
        resident=resident, cases=summaries,
    )


def test_finite_case_summary_and_complete_payload_validate(tmp_path: Path) -> None:
    _root, prereg, resident, cases, summaries = _fixture(tmp_path)
    payload = _payload(prereg, resident, cases, summaries)
    producer.validate_finite_threshold_diagnostic(
        payload, repository=REPOSITORY, expected_git=prereg["git"],
        registry_cases_digest=prereg["registry_cases_digest"], resident=resident,
        expected_query_bindings=[case["query_binding"] for case in cases],
    )


def test_nonfinite_certificate_cannot_be_summarized(tmp_path: Path) -> None:
    _root, _prereg, _resident, cases, _summaries = _fixture(tmp_path)
    cases[0]["certificate"]["rounds"][0]["constrained_threshold"] = float("inf")
    with pytest.raises(diagnostic.DiagnosticError, match="strict-JSON"):
        diagnostic.summarize_case(cases[0])


def test_rehashed_diagnostic_terminal_drift_fails_closed(tmp_path: Path) -> None:
    _root, prereg, resident, cases, summaries = _fixture(tmp_path)
    payload = _payload(prereg, resident, cases, summaries)
    payload["cases"][0]["rounds"][-1]["certified"] = False
    case = payload["cases"][0]
    case["result_digest"] = producer.stable_hash(producer._without(
        case, {"result_digest"},
    ))
    payload["result_digest"] = producer.stable_hash(producer._without(
        payload, {"created_at", "result_digest"},
    ))
    with pytest.raises(producer.HarnessError, match="terminal state"):
        producer.validate_finite_threshold_diagnostic(
            payload, repository=REPOSITORY, expected_git=prereg["git"],
            registry_cases_digest=prereg["registry_cases_digest"], resident=resident,
            expected_query_bindings=[case["query_binding"] for case in cases],
        )


def test_private_scratch_rejects_protected_and_symlink_ancestry(
    tmp_path: Path,
) -> None:
    with pytest.raises(diagnostic.DiagnosticError, match="overlaps"):
        diagnostic.validate_scratch_root(
            REPOSITORY, REPOSITORY / producer.SOURCE_FULL_RELATIVE,
        )
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(outside, target_is_directory=True)
    with pytest.raises(diagnostic.DiagnosticError, match="aliased|symlink"):
        diagnostic.validate_scratch_root(REPOSITORY, alias)


def test_fixed_output_rejects_symlinked_parent(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    parent = tmp_path / "config/data/analogues/m04r13"
    parent.mkdir(parents=True)
    (parent / "finite-threshold-diagnostic-v1").symlink_to(
        outside, target_is_directory=True,
    )
    with pytest.raises(diagnostic.DiagnosticError, match="aliased|symlink"):
        diagnostic.validate_diagnostic_output_path(tmp_path)


def test_malformed_child_summary_cannot_poison_fixed_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "DIAGNOSTIC.json"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    cases = tuple(producer.CaseInput(
        ordinal, {"episode_id": query_id, "case_id": producer.FROZEN_CASE_IDS[ordinal]},
    ) for ordinal, query_id in enumerate(producer.FROZEN_QUERY_IDS))
    resident = {
        "content_digest": "a" * 64, "ready_digest": "b" * 64,
        "identity_digest": "c" * 64,
    }
    monkeypatch.setattr(
        diagnostic, "validate_diagnostic_output_path", lambda _repository: output,
    )
    monkeypatch.setattr(producer, "_implementation_git", lambda _repository: {})
    monkeypatch.setattr(
        producer, "_registry_cases",
        lambda *_args: (producer.REGISTRY_DIGEST, cases),
    )
    monkeypatch.setattr(producer, "resident_full", lambda *_args: resident)
    monkeypatch.setattr(diagnostic.tempfile, "mkdtemp", lambda **_kwargs: str(scratch))
    monkeypatch.setattr(
        diagnostic, "validate_scratch_root", lambda _repository, value: value,
    )

    def runner(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        result = Path(command[command.index("--result-path") + 1])
        result.write_text('{"bad":true}\n')
        return SimpleNamespace(returncode=0)

    with pytest.raises(diagnostic.DiagnosticError, match="case order"):
        diagnostic.run_diagnostic(tmp_path, runner=runner)
    assert not output.exists()


def test_preregister_payload_embeds_exact_diagnostic_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / producer.CONFIG_RELATIVE
    config.parent.mkdir(parents=True)
    config.write_text("synthetic: true\n")
    registry = tmp_path / producer.REGISTRY_RELATIVE
    registry.mkdir(parents=True)
    source = tmp_path / producer.SOURCE_FULL_RELATIVE
    source.mkdir(parents=True)
    cases = tuple(producer.CaseInput(
        ordinal, {"episode_id": query_id,
                  "case_id": producer.FROZEN_CASE_IDS[ordinal]},
    ) for ordinal, query_id in enumerate(producer.FROZEN_QUERY_IDS))
    files: dict[str, str] = {}
    git_deterministic = {
        "implementation_commit": "0" * 40, "files": files,
        "files_digest": producer.stable_hash(files),
    }
    git = {
        **git_deterministic, "digest": producer.stable_hash(git_deterministic),
    }
    resident = {
        "content_digest": "a" * 64, "ready_digest": "b" * 64,
        "identity_digest": "c" * 64,
    }
    binding = {
        "path": str((tmp_path / producer.DIAGNOSTIC_RELATIVE).resolve()),
        "sha256": "d" * 64, "result_digest": "e" * 64,
    }
    monkeypatch.setattr(producer, "_implementation_git", lambda _repository: git)
    monkeypatch.setattr(
        producer, "_registry_cases",
        lambda *_args: (producer.REGISTRY_DIGEST, cases),
    )
    monkeypatch.setattr(producer, "resident_full", lambda *_args: resident)
    monkeypatch.setattr(
        producer, "diagnostic_query_bindings", lambda _inputs: [{}, {}, {}, {}],
    )
    monkeypatch.setattr(
        producer, "finite_threshold_diagnostic_binding",
        lambda *_args, **_kwargs: binding,
    )
    validate_shape = producer.validate_preregistration_shape
    monkeypatch.setattr(
        producer, "validate_preregistration_shape",
        lambda payload: validate_shape(payload, enforce_topology=False),
    )
    payload = producer.preregister_payload(
        repository=tmp_path, config_path=config, registry_root=registry,
        source_full_root=source, resident_root=producer.RESIDENT_ROOT,
        output_root=tmp_path / producer.OUTPUT_RELATIVE,
        generation_id=producer.GENERATION_ID,
        provenance_digest=producer.PROVENANCE_DIGEST,
        reserve_bytes=producer.RESIDENT_RESERVE_BYTES,
    )
    assert payload["finite_threshold_diagnostic"] == binding
    assert payload["preregistration_digest"] == producer.stable_hash(
        producer._without(payload, {"preregistration_digest"})
    )
