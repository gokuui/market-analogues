from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest


def _module():
    path = (
        Path(__file__).parents[1] / "experiments/m04r"
        / "verify_m04r11_candidate_comparison_v2_post_open.py"
    )
    spec = importlib.util.spec_from_file_location("post_open_verifier", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _layout(tmp_path: Path):
    module = _module()
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    (repository / ".gitignore").write_text("config/data/analogues/\n")
    subprocess.run(["git", "add", ".gitignore"], cwd=repository, check=True)
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid",
         "commit", "-qm", "ignore artifacts"],
        cwd=repository, check=True,
    )
    artifact = repository / "config/data/analogues"
    roots = module.frozen.expected_roots(artifact)
    output = artifact / module.POST_OPEN_RELATIVE_ROOT
    return module, repository, artifact, roots, output


def test_git_ignored_repository_child_roots_are_allowed(tmp_path: Path) -> None:
    module, repository, artifact, roots, output = _layout(tmp_path)
    unrelated = artifact / "unrelated"
    unrelated.mkdir(parents=True)
    (unrelated / "tracked.txt").write_text("unrelated")
    subprocess.run(["git", "add", "-f", str(unrelated / "tracked.txt")], cwd=repository, check=True)
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid",
         "commit", "-qm", "unrelated tracked artifact"],
        cwd=repository, check=True,
    )
    module._validate_root_layout(
        repository=repository, artifact_dir=Path("config/data/analogues"),
        roots=roots, output_root=output,
    )


def test_unignored_or_tracked_exact_root_is_rejected(tmp_path: Path) -> None:
    module, repository, artifact, roots, output = _layout(tmp_path)
    (repository / ".gitignore").write_text("other/\n")
    subprocess.run(["git", "add", ".gitignore"], cwd=repository, check=True)
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid",
         "commit", "-qm", "remove artifact ignore"],
        cwd=repository, check=True,
    )
    with pytest.raises(ValueError, match="not Git-ignored"):
        module._validate_root_layout(
            repository=repository, artifact_dir=artifact, roots=roots,
            output_root=output,
        )

    (repository / ".gitignore").write_text("config/data/analogues/\n")
    subprocess.run(["git", "add", ".gitignore"], cwd=repository, check=True)
    candidate = Path(roots["candidate_root"])
    candidate.mkdir(parents=True)
    tracked = candidate / "tracked.json"
    tracked.write_text("{}")
    subprocess.run(["git", "add", "-f", str(tracked)], cwd=repository, check=True)
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid",
         "commit", "-qm", "tracked exact evidence"],
        cwd=repository, check=True,
    )
    with pytest.raises(ValueError, match="contains tracked files"):
        module._validate_root_layout(
            repository=repository, artifact_dir=artifact, roots=roots,
            output_root=output,
        )


def test_output_overlap_and_symbolic_alias_are_rejected(tmp_path: Path) -> None:
    module, repository, artifact, roots, output = _layout(tmp_path)
    with pytest.raises(ValueError, match="overlap or alias"):
        module._validate_root_layout(
            repository=repository, artifact_dir=artifact, roots=roots,
            output_root=Path(roots["candidate_root"]),
        )
    candidate = Path(roots["candidate_root"])
    candidate.mkdir(parents=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(candidate, output)
    with pytest.raises(ValueError, match="symbolic-link alias"):
        module._validate_root_layout(
            repository=repository, artifact_dir=artifact, roots=roots,
            output_root=output,
        )

    output.unlink()
    actual_candidate = Path(roots["candidate_root"])
    alias = artifact / "candidate-alias"
    os.symlink(actual_candidate, alias)
    with pytest.raises(ValueError, match="symbolic-link alias"):
        module._validate_root_layout(
            repository=repository, artifact_dir=artifact, roots=roots,
            output_root=output, observed_input_roots=(alias,),
        )


def test_ignore_policy_must_be_tracked_clean_repository_gitignore(
    tmp_path: Path,
) -> None:
    module, repository, artifact, roots, output = _layout(tmp_path)
    (repository / ".gitignore").write_text("other/\n")
    subprocess.run(["git", "add", ".gitignore"], cwd=repository, check=True)
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid",
         "commit", "-qm", "remove repository artifact ignore"],
        cwd=repository, check=True,
    )
    info_exclude = repository / ".git/info/exclude"
    info_exclude.write_text(info_exclude.read_text() + "\nconfig/data/analogues/\n")
    with pytest.raises(ValueError, match="not from the tracked repository"):
        module._validate_root_layout(
            repository=repository, artifact_dir=artifact, roots=roots,
            output_root=output,
        )

    (repository / ".gitignore").write_text("config/data/analogues/\n")
    with pytest.raises(ValueError, match="not committed and clean"):
        module._validate_root_layout(
            repository=repository, artifact_dir=artifact, roots=roots,
            output_root=output,
        )


def test_lexical_parent_alias_is_rejected(tmp_path: Path) -> None:
    module, repository, _artifact, roots, output = _layout(tmp_path)
    with pytest.raises(ValueError, match="lexical parent alias"):
        module._validate_root_layout(
            repository=repository,
            artifact_dir=Path("config/data/../data/analogues"),
            roots=roots, output_root=output,
        )


def test_tree_prescan_rejects_required_file_symlink_before_read(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path / "candidate"
    authority = tmp_path / "authority.json"
    root.mkdir()
    authority.write_text('{"truth": true}')
    os.symlink(authority, root / "candidate-contract.json")
    with pytest.raises(ValueError, match="symbolic link"):
        module._prescan_exact_tree(
            root, expected_files={"candidate-contract.json"},
            expected_directories=set(), label="candidate",
        )

    comparison = tmp_path / "comparison"
    comparison.mkdir()
    os.symlink(authority, comparison / "RESULTS_OPENED.json")
    with pytest.raises(ValueError, match="symbolic link"):
        module._prescan_exact_tree(
            comparison, expected_files={"RESULTS_OPENED.json"},
            expected_directories=set(), label="comparison",
        )

    authority_root = tmp_path / "authority"
    (authority_root / "cases").mkdir(parents=True)
    os.symlink(authority, authority_root / "cases/query.json")
    with pytest.raises(ValueError, match="symbolic link"):
        module._prescan_exact_tree(
            authority_root, expected_files={"cases/query.json"},
            expected_directories={"cases"}, label="authority",
        )


def _frozen_manifest_fixture(module, tmp_path: Path):
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    for index, relative in enumerate(module.frozen.IMPLEMENTATION_FILES):
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# fixed implementation {index}\n")
    source = repository / "src/market_analogues/example.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("VALUE = 1\n")
    (source.parent / "not-code.txt").write_text("not manifested\n")
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid",
         "commit", "-qm", "implementation"],
        cwd=repository, check=True,
    )
    prereg = repository / module.frozen.PREREGISTRATION_RELATIVE_PATH
    prereg.write_text("{}\n")
    subprocess.run(["git", "add", str(prereg)], cwd=repository, check=True)
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid",
         "commit", "-qm", "preregister"],
        cwd=repository, check=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    names = [*module.frozen.IMPLEMENTATION_FILES, "src/market_analogues/example.py"]
    files = {
        relative: sha256((repository / relative).read_bytes()).hexdigest()
        for relative in names
    }
    manifest = {"files": files, "digest": module.frozen._hash(files)}
    contract = {"implementation_manifest": manifest}
    candidate = tmp_path / "candidate"
    event = {
        "details": {"git_binding": {
            "head_commit": head,
            "preregistration_blob_sha256": sha256(prereg.read_bytes()).hexdigest(),
        }},
    }
    event_path = candidate / "ledger/events/000000.json"
    event_path.parent.mkdir(parents=True)
    event_path.write_text(json.dumps(event))
    return repository, candidate, contract


def test_real_shaped_manifest_superset_is_verified_and_scoped(
    tmp_path: Path,
) -> None:
    module = _module()
    repository, candidate, contract = _frozen_manifest_fixture(module, tmp_path)
    original = module.frozen.IMPLEMENTATION_FILES
    binding = module._frozen_git_binding(repository, candidate, contract)
    assert binding["implementation_blobs_verified"] == len(original) + 1
    assert module.frozen.IMPLEMENTATION_FILES == original
    with module._frozen_real_manifest_view(contract["implementation_manifest"]["files"]):
        assert set(module.frozen.IMPLEMENTATION_FILES) == set(
            contract["implementation_manifest"]["files"],
        )
    assert module.frozen.IMPLEMENTATION_FILES == original


def test_tampered_or_omitted_frozen_manifest_file_fails(
    tmp_path: Path,
) -> None:
    module = _module()
    repository, candidate, contract = _frozen_manifest_fixture(module, tmp_path)
    tampered = deepcopy(contract)
    first = next(iter(tampered["implementation_manifest"]["files"]))
    tampered["implementation_manifest"]["files"][first] = "0" * 64
    tampered["implementation_manifest"]["digest"] = module.frozen._hash(
        tampered["implementation_manifest"]["files"],
    )
    with pytest.raises(ValueError, match="blob differs"):
        module._frozen_git_binding(repository, candidate, tampered)

    omitted = deepcopy(contract)
    del omitted["implementation_manifest"]["files"][
        "src/market_analogues/example.py"
    ]
    omitted["implementation_manifest"]["digest"] = module.frozen._hash(
        omitted["implementation_manifest"]["files"],
    )
    with pytest.raises(ValueError, match="file set differs"):
        module._frozen_git_binding(repository, candidate, omitted)


def _authority_history_fixture(module, tmp_path: Path):
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    builder = repository / "experiments/m04r/m04r11_build_authorities.py"
    source = repository / "src/market_analogues/packed_bound_search.py"
    builder.parent.mkdir(parents=True)
    source.parent.mkdir(parents=True)
    builder.write_text("# sealed builder\n")
    source.write_text("SEALED = True\n")
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    commit_environment = {
        **os.environ,
        "GIT_AUTHOR_DATE": "2026-01-01T00:00:00+00:00",
        "GIT_COMMITTER_DATE": "2026-01-01T00:00:00+00:00",
    }
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid",
         "commit", "-qm", "sealed authority implementation"],
        cwd=repository, check=True, env=commit_environment,
    )
    sealed_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    relative_files = (
        "experiments/m04r/m04r11_build_authorities.py",
        "src/market_analogues/packed_bound_search.py",
    )
    files = {
        relative: sha256((repository / relative).read_bytes()).hexdigest()
        for relative in relative_files
    }
    module.FROZEN_AUTHORITY_IMPLEMENTATION_COMMIT = sealed_commit
    module.FROZEN_AUTHORITY_IMPLEMENTATION_MANIFEST_DIGEST = (
        module.frozen._hash(files)
    )
    authority_root = tmp_path / "authority"
    authority_root.mkdir()
    (authority_root / "authority-contract.json").write_text(json.dumps({
        "runner_sha256": files[relative_files[0]],
        "implementation_manifest": {
            "files": files, "digest": module.frozen._hash(files),
        },
    }))
    (authority_root / "authority-matrix.json").write_text(json.dumps({
        "created_at": "2026-01-02T00:00:00+00:00",
    }))

    source.write_text("SEALED = False\n")
    (source.parent / "m04r_candidate_evidence.py").write_text("LATER = 1\n")
    (source.parent / "resident_store.py").write_text("LATER = 2\n")
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    later_environment = {
        **os.environ,
        "GIT_AUTHOR_DATE": "2026-01-03T00:00:00+00:00",
        "GIT_COMMITTER_DATE": "2026-01-03T00:00:00+00:00",
    }
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid",
         "commit", "-qm", "later implementation drift"],
        cwd=repository, check=True, env=later_environment,
    )
    return repository, authority_root, files


def test_historical_authority_tree_repairs_only_real_worktree_drift(
    tmp_path: Path,
) -> None:
    module = _module()
    repository, authority_root, files = _authority_history_fixture(
        module, tmp_path,
    )
    manifest = {"files": files, "digest": module.frozen._hash(files)}
    assert module.frozen._manifest_valid(manifest, repository) is False
    assert set(files) != {
        "experiments/m04r/m04r11_build_authorities.py",
        *(str(path.relative_to(repository)) for path in (
            repository / "src"
        ).rglob("*.py")),
    }

    binding, observed_files = module._authority_historical_binding(
        repository, authority_root,
    )
    assert binding["implementation_blobs_verified"] == 2
    assert binding["unique_matching_preseal_git_tree"] is True
    with module._historical_authority_tree(
        repository, binding, observed_files,
    ) as historical:
        assert module.frozen._manifest_valid(manifest, historical) is True
        assert set(files) == {
            "experiments/m04r/m04r11_build_authorities.py",
            *(str(path.relative_to(historical)) for path in (
                historical / "src"
            ).rglob("*.py")),
        }


def test_historical_authority_manifest_tamper_or_omission_fails(
    tmp_path: Path,
) -> None:
    module = _module()
    repository, authority_root, files = _authority_history_fixture(
        module, tmp_path,
    )
    contract_path = authority_root / "authority-contract.json"
    contract = json.loads(contract_path.read_text())
    contract["implementation_manifest"]["files"][
        "src/market_analogues/packed_bound_search.py"
    ] = "0" * 64
    contract["implementation_manifest"]["digest"] = module.frozen._hash(
        contract["implementation_manifest"]["files"],
    )
    contract_path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="manifest is not the frozen artifact"):
        module._authority_historical_binding(repository, authority_root)

    contract["implementation_manifest"]["files"] = dict(files)
    del contract["implementation_manifest"]["files"][
        "src/market_analogues/packed_bound_search.py"
    ]
    contract["implementation_manifest"]["digest"] = module.frozen._hash(
        contract["implementation_manifest"]["files"],
    )
    contract_path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="manifest is not the frozen artifact"):
        module._authority_historical_binding(repository, authority_root)


def test_canonical_authority_verification_is_exactly_bound(
    tmp_path: Path,
) -> None:
    module = _module()
    root = tmp_path / "authority-verification"
    root.mkdir()
    contract = {"contract_digest": "1" * 64}
    matrix = {
        "created_at": "2026-01-01T00:00:00+00:00",
        "contract_digest": contract["contract_digest"],
        "result_digest": "2" * 64,
        "performance_gate_passed": False,
    }
    seal = {"seal_digest": "3" * 64}
    deterministic = {
        "schema_version": "m04r11-certified-authority-verification-v4",
        "registry_digest": module.frozen.FROZEN_REGISTRY_DIGEST,
        "contract_digest": contract["contract_digest"],
        "generation_id": module.frozen.FROZEN_GENERATION_ID,
        "authority_matrix_digest": matrix["result_digest"],
        "authority_seal_digest": seal["seal_digest"],
        "verified_cases": 60,
        "authority_correctness_passed": True,
        "performance_gate_passed": False,
        "production_promotion_authorized": False,
        "proposal_rescans": 8,
        "proposal_failures": [],
        "per_case_failures": {},
        "gates": {
            "all_60_cases_independently_reconstructed": True,
            "all_proposal_prefixes_independently_rescanned": True,
            "candidate_results_remained_unopened": True,
            "matrix_and_seal_reconstructed": True,
            "physical_generation_rehashed": True,
            "real_forward_outcomes_excluded": True,
            "registry_runtime_validation_passed": True,
        },
        "failures": [],
        "real_forward_outcomes_accessed": False,
        "passed": True,
    }
    evidence = {
        **deterministic,
        "created_at": "2026-01-01T00:00:01+00:00",
        "result_digest": module.frozen._hash(deterministic),
    }
    module.FROZEN_AUTHORITY_MATRIX_DIGEST = matrix["result_digest"]
    module.FROZEN_AUTHORITY_SEAL_DIGEST = seal["seal_digest"]
    module.FROZEN_AUTHORITY_VERIFICATION_DIGEST = evidence["result_digest"]
    path = root / "m04r11-authority-verification.json"
    path.write_text(json.dumps(evidence))
    binding = module._authority_verification_binding(
        root, contract_digest=contract["contract_digest"],
        matrix=matrix, seal=seal,
    )
    assert binding["all_seven_gates_passed"] is True

    evidence["proposal_rescans"] = 7
    evidence["result_digest"] = module.frozen._hash({
        key: value for key, value in evidence.items()
        if key not in {"created_at", "result_digest"}
    })
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="verification evidence differs"):
        module._authority_verification_binding(
            root, contract_digest=contract["contract_digest"],
            matrix=matrix, seal=seal,
        )
