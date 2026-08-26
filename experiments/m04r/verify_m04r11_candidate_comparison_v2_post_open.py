"""Post-open repair verifier for the terminal M04R-11 v2 comparison.

The preregistered verifier could not run on the real configured layout because
it rejected every evidence root below the repository, even when that root was
Git-ignored and contained no tracked files.  This separately named verifier
does not alter or relabel any frozen artifact.  It reuses the preregistered
independent validations, replaces only that impossible root rule, and records
that this entry point was authored after authority results had already opened.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Iterator, Mapping
from uuid import uuid4


EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import verify_m04r11_candidate_comparison_v2 as frozen  # noqa: E402


POST_OPEN_SCHEMA = "candidate-recall-comparison-post-open-verification-v1"
POST_OPEN_RELATIVE_ROOT = Path(
    "m04r11/candidate-comparison-post-open-verification-v2"
)
AUTHORITY_VERIFICATION_RELATIVE_ROOT = Path("m04r11/authority-verification-v4")
FROZEN_AUTHORITY_IMPLEMENTATION_COMMIT = (
    "b86f899d4f4d95932591635c5f508398633fe078"
)
FROZEN_AUTHORITY_IMPLEMENTATION_MANIFEST_DIGEST = (
    "b6f928c177a01be4f945e9ead262f7f6b0dc4279d95afa15ef3f71a948a5b154"
)
FROZEN_AUTHORITY_MATRIX_DIGEST = (
    "ae60b7cda755c2deabc7bf9335701d34eea73073664582933eb5ba984e7f9b29"
)
FROZEN_AUTHORITY_SEAL_DIGEST = (
    "83501b4cf620607c4ee14cc837d1d5c242460dc867aca22a51b3fbbf924be2c9"
)
FROZEN_AUTHORITY_VERIFICATION_DIGEST = (
    "99d11756ed7542714635eca3fb75a43faed93881121d1f22d8d87426f4d6b190"
)
REPAIR_RELATIVE_PATH = (
    "experiments/m04r/verify_m04r11_candidate_comparison_v2_post_open.py"
)


def _run_git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *arguments], cwd=repository, check=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _plain_ancestry(path: Path, repository: Path) -> None:
    """Reject aliases and special entries in every existing path component."""
    target = path if path.is_absolute() else repository / path
    current = Path(target.anchor)
    for part in target.parts[1:]:
        current /= part
        if not current.exists() and not current.is_symlink():
            continue
        observed = current.lstat()
        if stat.S_ISLNK(observed.st_mode):
            raise ValueError(f"evidence path contains symbolic-link alias: {current}")
        if current == target:
            if not (stat.S_ISDIR(observed.st_mode) or stat.S_ISREG(observed.st_mode)):
                raise ValueError(f"evidence path is special: {current}")
        elif not stat.S_ISDIR(observed.st_mode):
            raise ValueError(f"evidence ancestry is not a directory: {current}")


def _git_ignored_untracked_root(repository: Path, root: Path) -> None:
    relative = str(root.relative_to(repository))
    ignored = _run_git(repository, "check-ignore", "-v", "--no-index", "--", relative)
    if ignored.returncode != 0:
        raise ValueError(f"repository-resident evidence root is not Git-ignored: {relative}")
    try:
        provenance = ignored.stdout.decode().split("\t", 1)[0].rsplit(":", 2)[0]
    except (IndexError, UnicodeDecodeError) as exc:
        raise ValueError("Git-ignore provenance is malformed") from exc
    if (repository / provenance).resolve() != repository / ".gitignore":
        raise ValueError("evidence ignore rule is not from the tracked repository .gitignore")
    tracked = _run_git(repository, "ls-files", "--", relative)
    if tracked.returncode != 0 or tracked.stdout.strip():
        raise ValueError(f"evidence root contains tracked files: {relative}")


def _validate_root_layout(
    *, repository: Path, artifact_dir: Path, roots: Mapping[str, str],
    output_root: Path, observed_input_roots: Iterable[Path] = (),
    additional_roots: Iterable[Path] = (),
) -> None:
    repository = repository.resolve()
    if ".." in artifact_dir.parts:
        raise ValueError("artifact directory contains a lexical parent alias")
    lexical_artifact = artifact_dir if artifact_dir.is_absolute() else repository / artifact_dir
    expected_artifact = repository / "config/data/analogues"
    if lexical_artifact != expected_artifact:
        raise ValueError("artifact directory is not the exact repository artifact path")
    _plain_ancestry(lexical_artifact, repository)
    if lexical_artifact.resolve() != expected_artifact:
        raise ValueError("artifact directory resolves through an alias")
    ignore_tracked = _run_git(
        repository, "ls-files", "--error-unmatch", "--", ".gitignore",
    )
    ignore_worktree = _run_git(repository, "diff", "--quiet", "--", ".gitignore")
    ignore_index = _run_git(
        repository, "diff", "--cached", "--quiet", "--", ".gitignore",
    )
    if any(value.returncode for value in (
        ignore_tracked, ignore_worktree, ignore_index,
    )):
        raise ValueError("evidence ignore policy is not committed and clean")

    for observed in observed_input_roots:
        if ".." in observed.parts:
            raise ValueError("evidence input contains a lexical parent alias")
        _plain_ancestry(observed, repository)

    evidence = [
        Path(roots[key]) for key in (
            "registry_root", "source_full_root", "candidate_root",
            "authority_root", "comparison_root",
        )
    ] + list(additional_roots) + [output_root]
    resolved = [path.resolve() for path in evidence]
    for path in evidence:
        _plain_ancestry(path, repository)
    for index, left in enumerate(resolved):
        for right in resolved[index + 1:]:
            if left == right or left.is_relative_to(right) or right.is_relative_to(left):
                raise ValueError("evidence and output roots overlap or alias")
    if any(repository == path or repository.is_relative_to(path) for path in resolved):
        raise ValueError("repository is inside an evidence root")
    for path in resolved:
        if path.is_relative_to(repository):
            _git_ignored_untracked_root(repository, path)

    if output_root.exists() or output_root.is_symlink():
        observed = output_root.lstat()
        if not stat.S_ISDIR(observed.st_mode) or any(output_root.iterdir()):
            raise ValueError("post-open verification root must be absent or plain-empty")


def _prescan_exact_tree(
    root: Path, *, expected_files: Iterable[str], expected_directories: Iterable[str],
    label: str,
) -> None:
    expected_file_set = set(expected_files)
    expected_directory_set = set(expected_directories)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{label} root is linked or absent")
    observed_files: set[str] = set()
    observed_directories: set[str] = set()

    def visit(directory: Path) -> None:
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                relative = str(path.relative_to(root))
                observed = path.lstat()
                if stat.S_ISLNK(observed.st_mode):
                    raise ValueError(f"{label} tree contains a symbolic link")
                if stat.S_ISDIR(observed.st_mode):
                    observed_directories.add(relative)
                    visit(path)
                elif stat.S_ISREG(observed.st_mode):
                    observed_files.add(relative)
                else:
                    raise ValueError(f"{label} tree contains a special entry")

    visit(root)
    if observed_files != expected_file_set or observed_directories != expected_directory_set:
        raise ValueError(f"{label} terminal artifact tree differs before any read")


def _require_plain_file(path: Path, label: str) -> None:
    _plain_ancestry(path, Path(__file__).resolve().parents[2])
    try:
        observed = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is absent") from exc
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
        raise ValueError(f"{label} is linked or special")


def _frozen_git_binding(
    repository: Path, candidate_root: Path, contract: Mapping[str, Any],
) -> dict[str, Any]:
    event_path = candidate_root / "ledger/events/000000.json"
    event = frozen._read(event_path)
    details = event.get("details") if type(event.get("details")) is dict else {}
    binding = details.get("git_binding") if type(details.get("git_binding")) is dict else {}
    head = str(binding.get("head_commit", ""))
    commit = _run_git(repository, "rev-parse", "--verify", f"{head}^{{commit}}")
    if commit.returncode != 0 or commit.stdout.decode().strip() != head:
        raise ValueError("frozen launch HEAD is not an available Git commit")
    prereg = frozen.PREREGISTRATION_RELATIVE_PATH
    changed = _run_git(
        repository, "diff-tree", "--no-commit-id", "--name-only", "-r", head,
    )
    if changed.returncode != 0 or changed.stdout.decode().splitlines() != [prereg]:
        raise ValueError("frozen launch HEAD is not the sole-file preregistration commit")
    manifest = contract.get("implementation_manifest")
    if type(manifest) is not dict or set(manifest) != {"files", "digest"}:
        raise ValueError("frozen implementation manifest fields differ")
    expected_files = dict(manifest.get("files", {}))
    if not expected_files or manifest.get("digest") != frozen._hash(expected_files):
        raise ValueError("frozen implementation manifest digest differs")
    source_tree = _run_git(
        repository, "ls-tree", "-r", "--name-only", head, "--", "src",
    )
    if source_tree.returncode != 0:
        raise ValueError("cannot enumerate frozen source implementation tree")
    frozen_source_files = {
        value for value in source_tree.stdout.decode().splitlines()
        if value.endswith(".py")
    }
    exact_manifest_files = set(frozen.IMPLEMENTATION_FILES) | frozen_source_files
    if set(expected_files) != exact_manifest_files:
        raise ValueError("frozen implementation manifest file set differs")
    for relative, digest in expected_files.items():
        if (
            not isinstance(relative, str) or not relative
            or Path(relative).is_absolute() or ".." in Path(relative).parts
            or not isinstance(digest, str) or len(digest) != 64
        ):
            raise ValueError("frozen implementation manifest entry differs")
        blob = _run_git(repository, "show", f"{head}:{relative}")
        if blob.returncode != 0 or sha256(blob.stdout).hexdigest() != digest:
            raise ValueError(f"frozen implementation blob differs: {relative}")
    prereg_blob = _run_git(repository, "show", f"{head}:{prereg}")
    if prereg_blob.returncode != 0 or (
        sha256(prereg_blob.stdout).hexdigest()
        != binding.get("preregistration_blob_sha256")
    ):
        raise ValueError("frozen preregistration blob differs")
    deterministic = {
        "frozen_head_commit": head,
        "frozen_preregistration_blob_sha256": binding["preregistration_blob_sha256"],
        "frozen_implementation_manifest_digest": contract["implementation_manifest"]["digest"],
        "implementation_blobs_verified": len(expected_files),
        "sole_file_preregistration_commit_verified": True,
    }
    return {**deterministic, "binding_digest": frozen._hash(deterministic)}


def _repair_implementation_binding(repository: Path) -> dict[str, Any]:
    tracked = _run_git(repository, "ls-files", "--error-unmatch", "--", REPAIR_RELATIVE_PATH)
    worktree = _run_git(repository, "diff", "--quiet")
    index = _run_git(repository, "diff", "--cached", "--quiet")
    head = _run_git(repository, "rev-parse", "--verify", "HEAD")
    if any(value.returncode for value in (tracked, worktree, index, head)):
        raise ValueError("post-open verifier must be tracked at a clean Git HEAD")
    deterministic = {
        "head_commit": head.stdout.decode().strip(),
        "relative_path": REPAIR_RELATIVE_PATH,
        "file_sha256": _file_sha256(repository / REPAIR_RELATIVE_PATH),
        "tracked_worktree_clean": True,
        "index_clean": True,
    }
    return {**deterministic, "binding_digest": frozen._hash(deterministic)}


def _authority_historical_binding(
    repository: Path, authority_root: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Bind the sealed authority implementation to its unique pre-seal Git tree."""
    contract = frozen._read(authority_root / "authority-contract.json")
    matrix = frozen._read(authority_root / "authority-matrix.json")
    manifest = contract.get("implementation_manifest")
    if type(manifest) is not dict or set(manifest) != {"files", "digest"}:
        raise ValueError("authority implementation manifest fields differ")
    files = dict(manifest.get("files", {}))
    if not files or manifest.get("digest") != frozen._hash(files):
        raise ValueError("authority implementation manifest digest differs")
    if manifest["digest"] != FROZEN_AUTHORITY_IMPLEMENTATION_MANIFEST_DIGEST:
        raise ValueError("authority implementation manifest is not the frozen artifact")
    expected_fixed = "experiments/m04r/m04r11_build_authorities.py"
    if expected_fixed not in files:
        raise ValueError("authority runner is absent from implementation manifest")
    sealed_at = frozen._timestamp(matrix.get("created_at"), "authority matrix")
    revisions = _run_git(repository, "rev-list", "--all")
    if revisions.returncode != 0:
        raise ValueError("cannot enumerate Git history for sealed authority")
    matches: list[tuple[str, str]] = []
    for commit in revisions.stdout.decode().splitlines():
        tree = _run_git(
            repository, "ls-tree", "-r", "--name-only", commit, "--", "src",
        )
        if tree.returncode != 0:
            continue
        names = {
            value for value in tree.stdout.decode().splitlines()
            if value.endswith(".py")
        } | {expected_fixed}
        if names != set(files):
            continue
        if any(
            (blob := _run_git(repository, "show", f"{commit}:{relative}")).returncode
            or sha256(blob.stdout).hexdigest() != digest
            for relative, digest in files.items()
        ):
            continue
        timestamp = _run_git(repository, "show", "-s", "--format=%cI", commit)
        if timestamp.returncode != 0:
            continue
        committed_at = frozen._timestamp(
            timestamp.stdout.decode().strip(), "authority implementation commit",
        )
        if committed_at <= sealed_at:
            matches.append((commit, committed_at.isoformat()))
    if len(matches) != 1 or matches[0][0] != FROZEN_AUTHORITY_IMPLEMENTATION_COMMIT:
        raise ValueError(
            "authority implementation manifest has no unique pre-seal Git tree"
        )
    commit, committed_at = matches[0]
    runner_digest = files[expected_fixed]
    if contract.get("runner_sha256") != runner_digest:
        raise ValueError("authority runner digest differs from historical manifest")
    deterministic = {
        "historical_commit": commit,
        "historical_commit_time": committed_at,
        "authority_matrix_created_at": sealed_at.isoformat(),
        "implementation_manifest_digest": manifest["digest"],
        "implementation_blobs_verified": len(files),
        "runner_sha256": runner_digest,
        "unique_matching_preseal_git_tree": True,
    }
    return (
        {**deterministic, "binding_digest": frozen._hash(deterministic)},
        files,
    )


def _authority_verification_binding(
    verification_root: Path, *, contract_digest: str,
    matrix: Mapping[str, Any], seal: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = frozen._read(
        verification_root / "m04r11-authority-verification.json"
    )
    expected_gates = {
        "all_60_cases_independently_reconstructed": True,
        "all_proposal_prefixes_independently_rescanned": True,
        "candidate_results_remained_unopened": True,
        "matrix_and_seal_reconstructed": True,
        "physical_generation_rehashed": True,
        "real_forward_outcomes_excluded": True,
        "registry_runtime_validation_passed": True,
    }
    expected_keys = {
        "schema_version", "registry_digest", "contract_digest", "generation_id",
        "authority_matrix_digest", "authority_seal_digest", "verified_cases",
        "authority_correctness_passed", "performance_gate_passed",
        "production_promotion_authorized", "proposal_rescans",
        "proposal_failures", "per_case_failures", "gates", "failures",
        "real_forward_outcomes_accessed", "created_at", "passed", "result_digest",
    }
    if not all((
        set(evidence) == expected_keys,
        evidence.get("schema_version") == "m04r11-certified-authority-verification-v4",
        evidence.get("registry_digest") == frozen.FROZEN_REGISTRY_DIGEST,
        evidence.get("contract_digest") == contract_digest,
        evidence.get("generation_id") == frozen.FROZEN_GENERATION_ID,
        evidence.get("authority_matrix_digest") == matrix.get("result_digest"),
        evidence.get("authority_seal_digest") == seal.get("seal_digest"),
        matrix.get("result_digest") == FROZEN_AUTHORITY_MATRIX_DIGEST,
        seal.get("seal_digest") == FROZEN_AUTHORITY_SEAL_DIGEST,
        evidence.get("verified_cases") == 60,
        evidence.get("authority_correctness_passed") is True,
        evidence.get("performance_gate_passed") is matrix.get("performance_gate_passed"),
        evidence.get("production_promotion_authorized") is False,
        evidence.get("proposal_rescans") == 8,
        evidence.get("proposal_failures") == [],
        evidence.get("per_case_failures") == {},
        evidence.get("gates") == expected_gates,
        evidence.get("failures") == [],
        evidence.get("real_forward_outcomes_accessed") is False,
        evidence.get("passed") is True,
        frozen._timestamp(evidence.get("created_at"), "authority verification")
        >= frozen._timestamp(matrix.get("created_at"), "authority matrix"),
        evidence.get("result_digest")
        == frozen._hash(frozen._without(evidence, {"created_at", "result_digest"})),
        evidence.get("result_digest") == FROZEN_AUTHORITY_VERIFICATION_DIGEST,
    )):
        raise ValueError("canonical authority verification evidence differs")
    deterministic = {
        "schema_version": evidence["schema_version"],
        "result_digest": evidence["result_digest"],
        "verified_cases": 60,
        "all_seven_gates_passed": True,
        "authority_correctness_passed": True,
        "performance_gate_passed": evidence["performance_gate_passed"],
        "production_promotion_authorized": False,
    }
    return {**deterministic, "binding_digest": frozen._hash(deterministic)}


@contextmanager
def _historical_authority_tree(
    repository: Path, binding: Mapping[str, Any], files: Mapping[str, str],
) -> Iterator[Path]:
    """Materialize only already-verified historical blobs for frozen validation."""
    with tempfile.TemporaryDirectory(prefix="m04r11-authority-history-") as value:
        root = Path(value)
        commit = str(binding["historical_commit"])
        for relative, digest in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            blob = _run_git(repository, "show", f"{commit}:{relative}")
            if blob.returncode != 0 or sha256(blob.stdout).hexdigest() != digest:
                raise ValueError("historical authority blob changed during materialization")
            path.write_bytes(blob.stdout)
        yield root


@contextmanager
def _frozen_real_manifest_view(files: Iterable[str]) -> Iterator[None]:
    """Correct only the frozen verifier's seven-file manifest-shape defect."""
    original = frozen.IMPLEMENTATION_FILES
    frozen.IMPLEMENTATION_FILES = tuple(files)
    try:
        yield
    finally:
        frozen.IMPLEMENTATION_FILES = original


class _FrozenHeadSubprocess:
    def __init__(self, head: str):
        self.head = head

    def run(self, arguments: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        if arguments == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(arguments, 0, self.head + "\n", "")
        if arguments in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
            return subprocess.CompletedProcess(arguments, 0, "", "")
        raise ValueError(f"unexpected frozen verifier subprocess request: {arguments}")


@contextmanager
def _frozen_head_view(head: str) -> Iterator[None]:
    original = frozen.subprocess
    frozen.subprocess = _FrozenHeadSubprocess(head)  # type: ignore[assignment]
    try:
        yield
    finally:
        frozen.subprocess = original


def _publish_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            raw = (json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_post_open(
    *, config: Path, artifact_dir: Path, repository_root: Path,
    registry_root: Path, candidate_root: Path, authority_root: Path,
    comparison_root: Path, output_root: Path,
) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    if repository_root.resolve() != repository:
        raise ValueError("repository root differs from post-open verifier root")
    if config.resolve() != repository / "config/datasets.example.yaml":
        raise ValueError("post-open verification config path differs")
    roots = frozen.expected_roots(artifact_dir)
    observed = {
        "registry_root": str(registry_root.resolve()),
        "candidate_root": str(candidate_root.resolve()),
        "authority_root": str(authority_root.resolve()),
        "comparison_root": str(comparison_root.resolve()),
    }
    if any(observed[key] != roots[key] for key in observed):
        raise ValueError("post-open verification input root differs")
    expected_output = artifact_dir.resolve() / POST_OPEN_RELATIVE_ROOT
    authority_verification_root = (
        artifact_dir.resolve() / AUTHORITY_VERIFICATION_RELATIVE_ROOT
    )
    if output_root.resolve() != expected_output:
        raise ValueError("post-open verification output root differs")
    _validate_root_layout(
        repository=repository, artifact_dir=artifact_dir, roots=roots,
        output_root=output_root, observed_input_roots=(
            registry_root, candidate_root, authority_root, comparison_root,
        ),
        additional_roots=(authority_verification_root,),
    )
    repair_binding = _repair_implementation_binding(repository)

    _require_plain_file(registry_root / "query-registry.json", "registry JSON")
    registry = frozen._read(registry_root / "query-registry.json")
    roles = frozen._roles(registry)
    execution = frozen._execution(roles)
    ordered = [str(case["episode_id"]) for case in registry["cases_data"]]
    candidate_files = {
        "candidate-contract.json", "RESIDENT_READY.json", "ledger/HEAD.json",
        "semantic-matrix.json", "SEMANTIC_SEALED.json", "performance-matrix.json",
        "PERFORMANCE_FINAL.json", "RUN_COMPLETE.json",
        *(f"case-bundles/{ordinal:03d}-{query_id}.json" for ordinal, query_id in enumerate(execution)),
        *(f"ledger/events/{index:06d}.json" for index in range(122)),
    }
    _prescan_exact_tree(
        candidate_root, expected_files=candidate_files,
        expected_directories={"case-bundles", "ledger", "ledger/events"},
        label="candidate",
    )
    _prescan_exact_tree(
        comparison_root,
        expected_files={"RESULTS_OPENED.json", "candidate-comparison.json", "SEALED.json"},
        expected_directories=set(), label="comparison",
    )
    _prescan_exact_tree(
        authority_root,
        expected_files={
            "authority-contract.json", "authority-matrix.json",
            "authority-matrix.html", "SEALED.json",
            *(f"cases/{query_id}.json" for query_id in ordered),
        },
        expected_directories={"cases"}, label="authority",
    )
    _prescan_exact_tree(
        authority_verification_root,
        expected_files={
            "m04r11-authority-verification.json",
            "m04r11-authority-verification.html",
        },
        expected_directories=set(), label="authority verification",
    )
    untrusted_contract = frozen._read(candidate_root / "candidate-contract.json")
    source_pack = dict(untrusted_contract.get("source_pack", {}))
    for name in ("manifest_path", "rows_path", "overflow_path"):
        _require_plain_file(Path(str(source_pack.get(name, ""))), f"source pack {name}")

    frozen_binding = _frozen_git_binding(
        repository, candidate_root, untrusted_contract,
    )
    manifest_files = untrusted_contract["implementation_manifest"]["files"]
    with _frozen_real_manifest_view(manifest_files):
        contract, prereg, roles, execution = frozen._validate_contract_and_prereg(
            registry, artifact_dir, repository, candidate_root,
        )
    expected_queries = {
        str(case["episode_id"]): frozen._query_context(config, case)
        for case in registry["cases_data"]
    }
    resident = frozen._validate_resident(candidate_root, contract)
    with _frozen_head_view(frozen_binding["frozen_head_commit"]):
        semantics, semantic_seal, performance_final, run_complete = (
            frozen._validate_candidate_aggregate(
                registry, candidate_root, contract, prereg, roles, execution,
                resident, expected_queries, repository,
            )
        )
    marker = frozen._validate_marker(
        comparison_root, contract, semantic_seal, performance_final, run_complete,
    )
    authority_binding, authority_files = _authority_historical_binding(
        repository, authority_root,
    )
    with _historical_authority_tree(
        repository, authority_binding, authority_files,
    ) as authority_repository:
        authorities, authority_matrix, authority_seal = frozen._validate_authorities(
            registry, authority_root, authority_repository, expected_queries,
        )
    authority_verification = _authority_verification_binding(
        authority_verification_root,
        contract_digest=str(authority_matrix["contract_digest"]),
        matrix=authority_matrix, seal=authority_seal,
    )
    comparison, comparison_seal = frozen._validate_comparison(
        registry, contract, semantics, semantic_seal, performance_final,
        authorities, authority_matrix, authority_seal, marker, comparison_root,
    )

    deterministic = {
        "schema_version": POST_OPEN_SCHEMA,
        "status": "post_open_evidence_validation_complete",
        "repair_scope": (
            "replaced the impossible descendant-root rejection with exact ignored-root "
            "and pre-read topology validation"
        ),
        "original_preregistered_verifier_result_produced": False,
        "post_open_entrypoint_was_preregistered": False,
        "authority_results_were_already_opened": True,
        "frozen_candidate_and_comparison_artifacts_modified": False,
        "frozen_verifier_manifest_validated": True,
        "frozen_git_binding": frozen_binding,
        "historical_authority_implementation_binding": authority_binding,
        "canonical_authority_verification_binding": authority_verification,
        "repair_implementation_binding": repair_binding,
        "preregistration_digest": prereg["preregistration_digest"],
        "producer_contract_digest": contract["contract_digest"],
        "semantic_seal_digest": semantic_seal["seal_digest"],
        "performance_final_digest": performance_final["final_digest"],
        "run_complete_digest": run_complete["complete_digest"],
        "results_opened_marker_digest": marker["result_digest"],
        "authority_matrix_digest": authority_matrix["result_digest"],
        "authority_seal_digest": authority_seal["seal_digest"],
        "comparison_digest": comparison["result_digest"],
        "comparison_seal_digest": comparison_seal["seal_digest"],
        "verified_cases": 60,
        "retained_total": comparison["retained_total"],
        "retained_denominator": comparison["retained_denominator"],
        "failed_cases": sum(not row["passed"] for row in comparison["cases"]),
        "comparison_gate_passed": comparison["passed"],
        "evidence_validation_passed": True,
        "production_promotion_authorized": False,
        "real_forward_outcomes_accessed": False,
    }
    payload = {
        **deterministic, "created_at": datetime.now(timezone.utc).isoformat(),
        "result_digest": frozen._hash(deterministic),
    }
    _publish_create_only(output_root / "post-open-verification.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[2] / "config/datasets.example.yaml",
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = verify_post_open(**vars(args))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["evidence_validation_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
