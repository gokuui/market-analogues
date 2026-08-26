"""Create or validate the fixed M04R-11 candidate-v2 preregistration.

This program is deliberately authority-blind.  It validates only configuration,
the frozen registry, durable source pack, predecessor terminal evidence, current
implementation/environment, and the absence of all v2 output roots.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping
from uuid import uuid4

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import m04r11_candidate_matrix_v2 as producer  # noqa: E402
from m04r11_candidate_v2_contract import (  # noqa: E402
    build_preregistration_document, expected_preregistration_path,
    expected_roots, load_and_validate_preregistration,
    validate_exact_roots, validate_predecessor_paths, validate_role_table,
)
from market_analogues.adapters import source_from_spec  # noqa: E402
from market_analogues.config import load_config  # noqa: E402
from market_analogues.m04r_validation_registry import (  # noqa: E402
    validate_m04r_validation_registry,
)


CONFIG_RELATIVE_PATH = "config/datasets.example.yaml"
REGISTRY_RELATIVE_PATH = (
    "config/data/analogues/m04r10/nasdaq-untouched-authority-registry/"
    "query-registry.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _assert_exact_inputs(
    repository_root: Path, config_path: Path, registry_path: Path,
    source_full_root: Path, artifact_dir: Path,
) -> dict[str, str]:
    repository = repository_root.resolve()
    if config_path.resolve() != repository / CONFIG_RELATIVE_PATH:
        raise ValueError("preregistration config path differs")
    if registry_path.resolve() != repository / REGISTRY_RELATIVE_PATH:
        raise ValueError("preregistration registry path differs")
    roots = expected_roots(artifact_dir)
    observed = dict(roots)
    observed["registry_root"] = str(registry_path.parent.resolve())
    observed["source_full_root"] = str(source_full_root.resolve())
    failures = validate_exact_roots(observed, artifact_dir)
    if failures:
        raise ValueError(f"preregistration exact roots differ:{failures}")
    return roots


def _assert_v2_outputs_absent(roots: Mapping[str, str]) -> None:
    for key in ("candidate_root", "comparison_root", "verification_root"):
        path = Path(roots[key])
        if path.exists() or path.is_symlink():
            raise ValueError(f"v2 output root already exists before preregistration: {key}")


def _publish_create_only(path: Path, payload: Mapping[str, Any]) -> bool:
    """Atomically publish via hard link; return False if another writer won."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                return False
            raise
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return True
    finally:
        if temporary.exists():
            temporary.unlink()


def create_or_validate_preregistration(
    *, repository_root: Path, config_path: Path, registry_path: Path,
    source_full_root: Path,
) -> tuple[dict[str, Any], bool]:
    """Create once or strictly validate the existing fixed document."""
    repository = repository_root.resolve()
    config = load_config(config_path)
    roots = _assert_exact_inputs(
        repository, config_path, registry_path, source_full_root,
        config.artifact_dir,
    )
    _assert_v2_outputs_absent(roots)
    predecessor_failures = validate_predecessor_paths(config.artifact_dir)
    if predecessor_failures:
        raise ValueError(
            f"predecessor terminal evidence differs:{predecessor_failures}"
        )
    registry = _read_json(registry_path)
    role_failures = validate_role_table(registry)
    if role_failures:
        raise ValueError(f"frozen registry roles differ:{role_failures}")
    registry_failures = validate_m04r_validation_registry(
        source_from_spec(config.datasets["nasdaq"]), registry_path.parent,
    )
    if registry_failures:
        raise ValueError(f"frozen registry validation differs:{registry_failures}")
    source_pack = producer._source_pack_binding(source_full_root)
    implementation = producer._implementation_manifest()
    environment = producer._environment_manifest()
    path = expected_preregistration_path(repository)
    if path.exists() or path.is_symlink():
        payload = load_and_validate_preregistration(
            registry, config.artifact_dir, repository,
            expected_source_pack=source_pack,
            expected_implementation_manifest=implementation,
            expected_environment_manifest=environment,
        )
        _assert_v2_outputs_absent(roots)
        return payload, False
    payload = build_preregistration_document(
        registry, config.artifact_dir, repository,
        source_pack=source_pack,
        implementation_manifest=implementation,
        environment_manifest=environment,
    )
    _assert_v2_outputs_absent(roots)
    created = _publish_create_only(path, payload)
    validated = load_and_validate_preregistration(
        registry, config.artifact_dir, repository,
        expected_source_pack=source_pack,
        expected_implementation_manifest=implementation,
        expected_environment_manifest=environment,
    )
    if validated != payload:
        raise ValueError("published preregistration differs from generated document")
    _assert_v2_outputs_absent(roots)
    return validated, created


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--source-full-root", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    payload, created = create_or_validate_preregistration(
        repository_root=repository, config_path=args.config,
        registry_path=args.registry, source_full_root=args.source_full_root,
    )
    print(json.dumps({
        "created": created,
        "validated": True,
        "path": str(expected_preregistration_path(repository)),
        "preregistration_digest": payload["preregistration_digest"],
        "authority_files_opened": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
